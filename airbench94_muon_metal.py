"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#            User-configurable flags        #
#############################################

# (i) Dynamically choose the parameter count (approximate target).
#     Set to an int like 1_300_000 for ~1.3M params; None keeps the default architecture.
TARGET_PARAMS = 500_000#None

# (ii) Save each trained model before starting the next run (checkpoint includes seed & param count).
SAVE_EACH_MODEL = True

# (iii) Skip the warmup run entirely.
SKIP_WARMUP = True

#############################################
#                  Setup                    #
#############################################

import os
import sys
import time
import random
import math
import numpy as np

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

if not torch.cuda.is_available():
    assert torch.backends.mps.is_available(), "Neither CUDA nor MPS is available"
    assert hasattr(torch, "compile"), "torch.compile not available in this PyTorch version"

# torch.backends.cudnn.benchmark = True

#############################################
#              Utility helpers              #
#############################################

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def round_to_multiple(x, multiple=8):
    return max(multiple, int(round(x / multiple) * multiple))

def total_params_for_widths(w1, w2, w3, whiten_kernel_size=2):
    # constants from architecture
    k = whiten_kernel_size
    whiten_width = 2 * 3 * (k ** 2)  # fixed at 24 for k=2
    total = 0
    # Whiten conv: weight + bias
    total += whiten_width * 3 * k * k + whiten_width
    # Group 1
    total += 9 * whiten_width * w1         # conv1
    total += 9 * w1 * w1                   # conv2
    total += 2 * w1 * 2                    # two BNs (weight+bias)
    # Group 2
    total += 9 * w1 * w2
    total += 9 * w2 * w2
    total += 2 * w2 * 2
    # Group 3
    total += 9 * w2 * w3
    total += 9 * w3 * w3
    total += 2 * w3 * 2
    # Head
    total += 10 * w3
    return int(total)

def choose_widths_for_target_params(target_params: int | None, base=(64, 256, 256)):
    if not target_params or target_params <= 0:
        return dict(block1=base[0], block2=base[1], block3=base[2])

    # Binary search over a multiplier that scales all blocks (rounded to multiples of 8)
    low, high = 0.25, 8.0
    best = None
    for _ in range(32):
        mid = (low + high) / 2
        w1 = round_to_multiple(base[0] * mid)
        w2 = round_to_multiple(base[1] * mid)
        w3 = round_to_multiple(base[2] * mid)
        params = total_params_for_widths(w1, w2, w3)
        cand = (abs(params - target_params), (w1, w2, w3), params)
        if (best is None) or (cand[0] < best[0]):
            best = cand
        if params < target_params:
            low = mid
        else:
            high = mid
    (_, (w1, w2, w3), _) = best
    return dict(block1=w1, block2=w2, block3=w3)

def count_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def save_checkpoint(model, seed: int, n_params: int, tta_val_acc: float, run_idx: int, widths: dict):
    ckpt_dir = os.path.join("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    fname = f"cifarnet_seed{seed}_params{n_params}_run{run_idx}.pt"
    path = os.path.join(ckpt_dir, fname)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "n_params": n_params,
            "widths": widths,
            "tta_val_acc": tta_val_acc,
            "code": code,
        },
        path,
    )
    print(f"Saved checkpoint: {os.path.abspath(path)}")

#############################################
#               Muon optimizer              #
#############################################

@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    # X = G.bfloat16()
    # X = G.float()
    X = G.half()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("mps"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        # self.images = (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.normalize = self.normalize = T.Normalize(
            CIFAR_MEAN.to(self.images.device, dtype=self.images.dtype),
            CIFAR_STD.to(self.images.device, dtype=self.images.dtype),
        )
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]

        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self, widths: dict | None = None, whiten_kernel_size: int = 2):
        super().__init__()
        widths = widths or dict(block1=64, block2=256, block3=256)
        self.widths = widths
        self.whiten_kernel_size = whiten_kernel_size

        whiten_width = 2 * 3 * whiten_kernel_size**2  # keep fixed (24 when k=2)
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            mod.half()
            # if isinstance(mod, BatchNorm):
            #     mod.float()
            # else:
            #     # mod.float()
            #     mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        dev = self.whiten.weight.device
        with torch.no_grad():
            train_images = train_images.float().to("cpu")
            c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
            patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
            patches_flat = patches.view(len(patches), -1)
            est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
            eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
            eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
            weights = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
            self.whiten.weight.data.copy_(weights.to(dev))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["run   ", "epoch", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

def main(run, model, seed=None):
    if seed is not None:
        seed_everything(seed)

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if run == "warmup":
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=False)
    optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code
    time_seconds = 0.0
    def start_timer():
        global _t0
        _t0 = time.perf_counter()
    def stop_timer():
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        nonlocal time_seconds
        time_seconds += time.perf_counter() - _t0

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        ####################
        #     Training     #
        ####################
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None

    ####################
    #  TTA Evaluation  #
    ####################
    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":

    # Pick widths based on TARGET_PARAMS
    widths = choose_widths_for_target_params(TARGET_PARAMS)
    model = CifarNet(widths=widths).to("mps").to(memory_format=torch.channels_last)

    # Re-use the compiled model between runs to save non-data-dependent compilation time
    model.compile()

    # Report chosen architecture size
    n_params = count_model_params(model)
    print(f"Chosen widths: {widths} | Total params: {n_params:,}")

    print_columns(logging_columns_list, is_head=True)

    # Optional warmup pass
    if not SKIP_WARMUP:
        main("warmup", model)

    # Training runs
    num_runs = 5
    acc_list = []
    base_seed = random.randint(0, 2 ** 16)
    for run in range(num_runs):
        seed = base_seed #+ run  # New seed per run

        tta_acc = main(run, model, seed=seed)
        acc_list.append(tta_acc)
        if SAVE_EACH_MODEL:
            save_checkpoint(model, seed=seed, n_params=n_params, tta_val_acc=tta_acc, run_idx=run, widths=widths)

    accs = torch.tensor(acc_list)
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

    # Log code and accuracies as before
    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs, widths=widths, n_params=n_params), log_path)
    print(os.path.abspath(log_path))

