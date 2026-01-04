from typing import final
import uuid
from airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, CifarLoader, BatchNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pyhessian import hessian
from itertools import islice
# import loss_landscapes
# import loss_landscapes.metrics as metrics
import torch.multiprocessing as mp
import numpy as np

import copy

@torch.no_grad()
def compute_fisher_rao_norm(model, input_shape=(3, 32, 32)):
    """
    Computes Fisher-Rao Norm by temporarily squaring weights.
    Handles all internal shape logic (Flatten, Pooling, etc.) automatically.
    """
    # 1. Get the underlying module if using torch.compile
    model_internal = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_internal.eval()
    
    # 2. Backup original parameters and buffers
    orig_state_dict = {k: v.clone() for k, v in model_internal.state_dict().items()}
    
    try:
        L = 0
        new_state_dict = {}
        for name, param in model_internal.named_parameters():
            if 'weight' in name:
                # Square the weights and remove biases
                new_state_dict[name] = param.detach().to(torch.float32).pow(2)
                if 'bn' not in name: L += 1 # Count Conv/Linear layers
            elif 'bias' in name:
                new_state_dict[name] = torch.zeros_like(param)

        # 3. Special handling for BatchNorm buffers (running_var)
        for name, buffer in model_internal.named_buffers():
            if 'running_var' in name:
                # Path scaling: gamma^2 / (var + eps)
                # We achieve this by setting running_var to 1 and weight to (gamma^2 / (var + eps))
                # But a simpler way is to just let the forward pass use the actual buffers
                # while we adjust the weights.
                new_state_dict[name] = buffer.detach().to(torch.float32)
            elif 'running_mean' in name:
                new_state_dict[name] = torch.zeros_like(buffer)
            else:
                new_state_dict[name] = buffer.clone()

        # Update the weights of a temporary clone to avoid corrupting the real model
        # Or just apply to the model and restore later (faster)
        model_internal.load_state_dict(new_state_dict, strict=False)
        model_internal.to(torch.float32) # Ensure model is in FP32

        # 4. Run the forward pass with all ones
        device = next(model_internal.parameters()).device
        x = torch.ones((1, *input_shape), device=device, dtype=torch.float32)
        output = model_internal(x)
        
        # Path norm is the sum of the output of the squared network
        path_norm_sum = output.sum()
        fisher_rao_norm = (L + 1) * torch.sqrt(path_norm_sum)
        
        return fisher_rao_norm.item()

    finally:
        # 5. ALWAYS restore the original weights and precision
        model_internal.load_state_dict(orig_state_dict)
        # Restore original dtype (check if it was half/bfloat)
        orig_dtype = orig_state_dict[next(iter(orig_state_dict))].dtype
        model_internal.to(orig_dtype)


def pyhessian_sharpness(model, loader, num_batches=10):
    """
    Computes relative sharpness by scaling the top Hessian eigenvalue 
    by the squared norm of the model parameters.
    """
    model = model._orig_mod if hasattr(model, "_orig_mod") else model
    criterion = torch.nn.CrossEntropyLoss()
    limited_loader = list(islice(loader, num_batches))
    
    hessian_comp = hessian(model, criterion, dataloader=limited_loader, cuda=True)
    lambda_max = hessian_comp.eigenvalues(top_n=1)[0][0]

    # 3. Compute Squared Frobenius Norm of all parameters
    total_norm_sq = 0.0
    num_of_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_norm_sq += torch.norm(p.data)**2
            num_of_params += p.numel()
    
    # Calculate relative sharpness:
    relative_sharpness = lambda_max * total_norm_sq.item() / num_of_params

    return relative_sharpness

@torch.no_grad()
def get_sam_sharpness(model, loader, rho=0.05, num_batches=10):
    model.eval()
    device = next(model.parameters()).device
    total_sharpness = 0.0

    for batch_idx, (inputs, targets) in enumerate(iter(loader)):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Compute base loss and gradients
        with torch.enable_grad():
            outputs = model(inputs)
            base_loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            base_loss.backward()
        
        # 2. Compute gradient norm
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_norm = torch.linalg.norm(
            torch.stack([torch.linalg.norm(g) for g in grads])
        ) + 1e-12
        
        # 3. Perturb parameters
        scale = rho / grad_norm
        epsilons = []
        for p in model.parameters():
            if p.grad is not None:
                eps = p.grad * scale  # Moved scale calculation out
                p.add_(eps)
                epsilons.append(eps)
        
        # 4. Compute perturbed loss
        adv_loss = F.cross_entropy(model(inputs), targets)
        
        # 5. Restore parameters
        for p, eps in zip(
            (p for p in model.parameters() if p.grad is not None), 
            epsilons
        ):  
            p.sub_(eps)
                
        total_sharpness += (adv_loss - base_loss).item()
        model.zero_grad()  # Clean up gradients

    return total_sharpness / min(num_batches, len(loader))

@torch.no_grad()
def get_samlike_sharpness(model, loader, rho=0.05, num_batches=10):
    model.eval()
    device = next(model.parameters()).device
    total_sharpness = 0.0

    loader_iter = iter(loader)

    for batch_idx, (inputs, targets) in enumerate(loader_iter):
        if batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Compute base loss and gradients
        with torch.enable_grad():
            outputs = model(inputs)
            base_loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            base_loss.backward()

        # 2. Compute TRUE gradient norm (flatten and concatenate)
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.linalg.norm(torch.cat(grads)) + 1e-12

        # 3. Perturb parameters
        scale = rho / grad_norm
        epsilons = []
        for p in model.parameters():
            if p.grad is not None:
                eps = p.grad * scale
                p.add_(eps)
                epsilons.append(eps)

        # 4. Compute perturbed loss
        adv_loss = F.cross_entropy(model(inputs), targets)

        # 5. Restore parameters
        for p, eps in zip(
            (p for p in model.parameters() if p.grad is not None),
            epsilons
        ):
            p.sub_(eps)

        # Normalize by gradient norm squared and rho squared
        # This approximates the Hessian eigenvalue in gradient direction
        sharpness = (adv_loss - base_loss) / (grad_norm**2 * rho**2 + 1e-12)
        total_sharpness += sharpness.item()
        
        model.zero_grad()

    return total_sharpness / min(num_batches, len(loader))


def train_and_log(model, optimizer_config):
    logs = {"train_acc": {}, "val_acc": {}, "gap": {}, "sam": {}, "hessian": {}, "fisher": {}}
    metric_loader = CifarLoader('cifar10', train=True, batch_size=1000)

    def callback_fn(epoch, model, training_accuracy, validation_accuracy):
        model.eval()
        logs["train_acc"][epoch] = training_accuracy
        logs["val_acc"][epoch] = validation_accuracy
        logs["gap"][epoch] = training_accuracy - validation_accuracy
        # logs["sam"][epoch] = get_sam_sharpness(model, metric_loader)
        # logs["fisher"][epoch] = compute_fisher_rao_norm(model)

        # if epoch == 8:
        #     # print(f"Calculating Hessian for epoch {epoch}...")
        #     logs["hessian"][epoch] = pyhessian_sharpness(model, metric_loader)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()

    final_acc = train(optimizer_config, model, SGDConfig(), callback=callback_fn)

    return final_acc, logs


def worker(gpu_id, runs_per_gpu, optimizer_config):
        all_acc, all_logs = [], []
        torch.cuda.set_device(gpu_id)
        model = CifarNet().to(f'cuda:{gpu_id}').to(memory_format=torch.channels_last)
        # model = torch.compile(model, mode="max-autotune")

        for run in tqdm(range(runs_per_gpu)) if gpu_id == 0 else range(runs_per_gpu):
            acc, logs = train_and_log(model, optimizer_config)
            all_acc.append(acc)
            all_logs.append(logs)
        
        return all_acc, all_logs


def train_distributed(gpus, runs_per_gpu, optimizer_config):
    with mp.Pool(gpus) as pool:
        out = [pool.apply_async(worker, args=(gpu_id, runs_per_gpu, optimizer_config)) for gpu_id in range(gpus)]
        results = [p.get() for p in out]
    
    all_accs, all_logs = [], []
    for accs, logs_list in results:
        all_accs.extend(accs)
        all_logs.extend(logs_list)

    return all_accs, all_logs


def print_aggregated_metrics(name, all_accs, all_logs, metrics=None, epochs=None):
    print(f"=== {name} Results ===")
    print(f"Final Accuracy: {sum(all_accs) / len(all_accs):.2f} ± {np.std(all_accs):.2f}\n")

    if not metrics:
        metrics = list(all_logs[0].keys())
    
    if not epochs:
        epochs = all_logs[0][metrics[0]].keys()

    for metric in metrics:
        for epoch in epochs:
            vals = [log[metric][epoch] for log in all_logs if epoch in log[metric]]
            if vals:
                mean_val = sum(vals) / len(vals)
                std_val = np.std(vals)
                print(f"{metric} @ epoch {epoch}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{metric} @ epoch {epoch}: [No Data]")
        print()
    print("\n")


def main():
    gpus = 4
    runs_per_gpu = 10

    print("Performing Warmup...")
    train_distributed(gpus, 1, MuonConfig())

    print("Running SGD Experiments...")
    sgd_accs, sgd_logs = train_distributed(gpus, runs_per_gpu, SGDConfig())

    print("Running Muon Experiments...")
    muon_accs, muon_logs = train_distributed(gpus, runs_per_gpu, MuonConfig())

    print_aggregated_metrics("SGD", sgd_accs, sgd_logs)
    print_aggregated_metrics("Muon", muon_accs, muon_logs)


if __name__ == "__main__":
    main()
    
    
