import argparse
import os
import time
from math import ceil
from pathlib import Path

import torch
from pyhessian import hessian

from airbench import CifarLoader, evaluate
from airbench.lib_nofrills import hyp, make_net94
from airbench.muon import Muon
from airbench.utils import LookaheadState, init_whitening_conv


class _MultiOptimizer:
    """Wrapper to step multiple optimizers together."""
    def __init__(self, optimizers):
        self.optimizers = optimizers
        # Expose param_groups for scheduler compatibility
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)


def _build_paths():
    repo_root = Path(__file__).resolve().parent
    cifar10_path = (repo_root / "cifar10").as_posix()

    username = os.getenv("USER")
    checkpoint_folder = Path(f"/work/scratch/{username}/deep-learning/checkpoints/")
    return cifar10_path, checkpoint_folder


def _make_optimizer(model, *, optimizer_name, batch_size, learning_rate, momentum, weight_decay, bias_scaler, nesterov):
    optimizer_name = optimizer_name.lower()

    if optimizer_name in {"sgd", "sgd_nesterov", "nesterov"}:
        kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
        lr = learning_rate / kilostep_scale
        wd = weight_decay * batch_size / kilostep_scale
        lr_biases = lr * bias_scaler

        norm_biases = [p for k, p in model.named_parameters() if "norm" in k and p.requires_grad]
        other_params = [p for k, p in model.named_parameters() if "norm" not in k and p.requires_grad]
        param_configs = [
            dict(params=norm_biases, lr=lr_biases, weight_decay=wd / lr_biases),
            dict(params=other_params, lr=lr, weight_decay=wd / lr),
        ]
        return torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    if optimizer_name == "adam":
        # Note: PyTorch's Adam uses L2-style weight_decay (not decoupled like AdamW).
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if optimizer_name == "muon":
        # Muon only for 4D conv filters; SGD for biases/norms (matches airbench94_muon.py)
        filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
        norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
        other_1d = [p for n, p in model.named_parameters()
                    if len(p.shape) != 4 and "norm" not in n and p.requires_grad]
        # Use same LR scaling as airbench94_muon.py for the SGD part
        wd_sgd = 2e-6 * batch_size
        bias_lr = 0.053
        head_lr = 0.67
        param_configs = [
            dict(params=norm_biases, lr=bias_lr, weight_decay=wd_sgd / bias_lr),
            dict(params=other_1d, lr=head_lr, weight_decay=wd_sgd / head_lr),
        ]
        opt_sgd = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)
        opt_muon = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=nesterov)
        # Return a wrapper that steps both
        return _MultiOptimizer([opt_sgd, opt_muon])

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _train_model(*, cifar10_path, optimizer_name, epochs, batch_size, learning_rate, momentum, weight_decay,
                 bias_scaler, label_smoothing, whiten_bias_epochs, lr_peak, lr_end, nesterov, verbose):
    train_loader = CifarLoader(
        cifar10_path,
        train=True,
        batch_size=batch_size,
        aug=dict(flip=hyp["aug"].get("flip", False), translate=hyp["aug"].get("translate", 0)),
        altflip=True,
    )
    test_loader = CifarLoader(cifar10_path, train=False, batch_size=2000)

    total_train_steps = ceil(len(train_loader) * epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

    model = make_net94()

    # Initialize the whitening layer using training images (mirrors airbench/utils.py)
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)

    optimizer = _make_optimizer(
        model,
        optimizer_name=optimizer_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        bias_scaler=bias_scaler,
        nesterov=nesterov,
    )

    # For Muon, we do manual LR scheduling (like airbench94_muon.py); for others, use LambdaLR
    use_manual_lr = isinstance(optimizer, _MultiOptimizer)

    if use_manual_lr:
        # Store initial LRs for manual decay
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        scheduler = None
    else:
        def get_lr(step):
            warmup_steps = int(total_train_steps * lr_peak)
            warmdown_steps = total_train_steps - warmup_steps
            if step < warmup_steps:
                frac = step / max(1, warmup_steps)
                return 0.2 * (1 - frac) + 1.0 * frac
            frac = (step - warmup_steps) / max(1, warmdown_steps)
            return 1.0 * (1 - frac) + lr_end * frac

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps + 1, device="cuda") / total_train_steps) ** 3
    lookahead_state = LookaheadState(model)

    current_steps = 0
    for epoch in range(ceil(epochs)):
        # Keep parity with airbench training: freeze whitening bias after some epochs
        model[0].bias.requires_grad = (epoch < whiten_bias_epochs)

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected during training (optimizer={optimizer_name}, step={current_steps})."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # LR scheduling: manual for Muon (before step), LambdaLR for others (after step)
            if use_manual_lr:
                # Linear decay to 0 (matches airbench94_muon.py - LR set before step)
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * (1 - current_steps / total_train_steps)

            optimizer.step()

            # Debug/safety: catch non-finite parameters early (Muon can diverge abruptly).
            if current_steps < 50:
                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        raise RuntimeError(
                            f"Non-finite parameter detected after optimizer step "
                            f"(optimizer={optimizer_name}, step={current_steps}, param={name})."
                        )

            if not use_manual_lr:
                scheduler.step()

            current_steps += 1

            if current_steps % 5 == 0:
                lookahead_state.update(model, decay=alpha_schedule[current_steps].item())

            if current_steps >= total_train_steps:
                lookahead_state.update(model, decay=1.0)
                break

        if verbose:
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = loss.item() / batch_size
            val_acc = evaluate(model, test_loader, tta_level=0)
            print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)

        if current_steps >= total_train_steps:
            break

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "adam", "muon"], default="sgd")
    parser.add_argument("--checkpoint-name", default=None)  # defaults to optimizer name
    parser.add_argument("--epochs", type=float, default=hyp["opt"]["train_epochs"])
    parser.add_argument("--batch-size", type=int, default=hyp["opt"]["batch_size"])
    parser.add_argument("--lr", type=float, default=hyp["opt"]["lr"])
    parser.add_argument("--momentum", type=float, default=hyp["opt"]["momentum"])
    parser.add_argument("--weight-decay", type=float, default=hyp["opt"]["weight_decay"])
    parser.add_argument("--bias-scaler", type=float, default=hyp["opt"]["bias_scaler"])
    parser.add_argument("--label-smoothing", type=float, default=hyp["opt"]["label_smoothing"])
    parser.add_argument("--whiten-bias-epochs", type=float, default=hyp["opt"]["whiten_bias_epochs"])
    parser.add_argument("--lr-peak", type=float, default=0.23)
    parser.add_argument("--lr-end", type=float, default=0.07)
    parser.add_argument("--muon-nesterov", action="store_true", default=True)
    parser.add_argument("--no-muon-nesterov", dest="muon_nesterov", action="store_false")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for training/data order/augmentations.",
    )
    parser.add_argument(
        "--hessian-seed",
        type=int,
        default=12345,
        help="RNG seed used right before Hessian/eigenvalue estimation for reproducibility.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_name is None:
        args.checkpoint_name = args.optimizer

    cifar10_path, checkpoint_folder = _build_paths()
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_folder / f"{args.checkpoint_name}.pt"

    # Make training/data order reproducible across optimizers/runs.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"{time.ctime()} Training starting (optimizer={args.optimizer})...", flush=True)
    model = _train_model(
        cifar10_path=cifar10_path,
        optimizer_name=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        bias_scaler=args.bias_scaler,
        label_smoothing=args.label_smoothing,
        whiten_bias_epochs=args.whiten_bias_epochs,
        lr_peak=args.lr_peak,
        lr_end=args.lr_end,
        nesterov=args.muon_nesterov,
        verbose=not args.quiet,
    )
    model.eval()

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    # Hessian evaluation on fixed splits (no shuffle, no aug) to avoid data leakage
    train_loader_hess = CifarLoader(
        cifar10_path,
        train=True,
        batch_size=512,
        aug=dict(flip=False, translate=0, cutout=0),
        altflip=False,
        shuffle=False,
        drop_last=False,
    )
    test_loader_hess = CifarLoader(
        cifar10_path,
        train=False,
        batch_size=512,
        aug=dict(flip=False, translate=0, cutout=0),
        altflip=False,
        shuffle=False,
        drop_last=False,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # pyhessian's eigenvalue routines typically use randomness (e.g., init vectors).
    # Re-seed here so sharpness comparisons are not confounded by RNG noise.
    torch.manual_seed(args.hessian_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.hessian_seed)

    print("Computing Hessian (train split)...", flush=True)
    hessian_comp_train = hessian(model, criterion, dataloader=train_loader_hess, cuda=True)
    top_eigenvalues_train, _ = hessian_comp_train.eigenvalues(top_n=1)
    print("Top Eigenvalue (train): ", top_eigenvalues_train[-1])

    print("Computing Hessian (test split)...", flush=True)
    hessian_comp_test = hessian(model, criterion, dataloader=test_loader_hess, cuda=True)
    top_eigenvalues_test, _ = hessian_comp_test.eigenvalues(top_n=1)
    print("Top Eigenvalue (test): ", top_eigenvalues_test[-1])


if __name__ == "__main__":
    main()
