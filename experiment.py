import uuid
from airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, CifarLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pyhessian import hessian
from itertools import islice
import loss_landscapes
import loss_landscapes.metrics as metrics

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

# @torch.no_grad()
# def get_sam_sharpness(model, loader, rho=0.05, num_batches=10):
#     model.eval()
#     device = next(model.parameters()).device
#     total_sharpness = 0.0

#     g = torch.Generator()
#     g.manual_seed(42) 
#     loader_iter = iter(loader)
    
#     for batch_idx, (inputs, targets) in enumerate(loader_iter):
#         if batch_idx >= num_batches:
#             break
            
#         inputs, targets = inputs.to(device), targets.to(device)
        
#         # 1. Compute base loss and gradients
#         with torch.enable_grad():
#             outputs = model(inputs)
#             base_loss = F.cross_entropy(outputs, targets)
#             model.zero_grad()
#             base_loss.backward()
        
#         # 2. Compute gradient norm
#         grads = [p.grad for p in model.parameters() if p.grad is not None]
#         grad_norm = torch.linalg.norm(
#             torch.stack([torch.linalg.norm(g) for g in grads])
#         ) + 1e-12
        
#         # 3. Perturb parameters
#         scale = rho / grad_norm
#         epsilons = []
#         for p in model.parameters():
#             if p.grad is not None:
#                 eps = p.grad * scale  # Moved scale calculation out
#                 p.add_(eps)
#                 epsilons.append(eps)
        
#         # 4. Compute perturbed loss
#         adv_loss = F.cross_entropy(model(inputs), targets)
        
#         # 5. Restore parameters
#         for p, eps in zip(
#             (p for p in model.parameters() if p.grad is not None), 
#             epsilons
#         ):  
#             p.sub_(eps)
                
#         total_sharpness += (adv_loss - base_loss).item()
#         model.zero_grad()  # Clean up gradients

#     return total_sharpness / min(num_batches, len(loader))

@torch.no_grad()
def get_sam_sharpness(model, loader, rho=0.05, num_batches=10):
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


sam_values = {}
hessian_values = {}
fisher_values = {}
def sam_callback(name):
    hessian_loader = CifarLoader('cifar10', train=True, batch_size=1000)
    sam_values[name] = {i: [] for i in range(9)}
    hessian_values[name] = {i: [] for i in range(9)}
    fisher_values[name] = {i: [] for i in range(9)}


    def callback_test(epoch, model, *args):
        model.eval()
        sam_sharp = get_sam_sharpness(model, hessian_loader)
        sam_values[name][epoch].append(sam_sharp)

        if epoch == 8:
            print(f"Calculating Hessian and Fisher for epoch {epoch}...")
            hess_sharp = pyhessian_sharpness(model, hessian_loader)
            fisher_sharp = compute_fisher_rao_norm(model)
        else:
            hess_sharp = -1.0
            fisher_sharp = -1.0
        hessian_values[name][epoch].append(hess_sharp)
        fisher_values[name][epoch].append(fisher_sharp)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()
    
    return callback_test


if __name__ == "__main__":

    device = "cuda"
    model = CifarNet().to(device).to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")
    num_runs = 2

    # 1. Warmup
    print("Performing Warmup...")
    train("Warmup", model, MuonConfig(batch_size=2000))

    sgd_callback = sam_callback("SGD")
    sgd_accs = torch.tensor([train(run, model, SGDConfig(), callback=sgd_callback) for run in tqdm(range(num_runs))])

    muon_callback = sam_callback("Muon")
    muon_accs = torch.tensor([train(run, model, MuonConfig(), callback=muon_callback) for run in tqdm(range(num_runs))])

    print("\nSGD  - Mean: %.4f    Std: %.4f" % (sgd_accs.mean(), sgd_accs.std()))
    print(f" SAM0 - Mean: {torch.tensor(sam_values['SGD'][0]).mean()}    Std: {torch.tensor(sam_values['SGD'][0]).std()}")
    print(f" SAM8 - Mean: {torch.tensor(sam_values['SGD'][8]).mean()}    Std: {torch.tensor(sam_values['SGD'][8]).std()}")
    print(f" HESS0 - Mean: {torch.tensor(hessian_values['SGD'][0]).mean()}    Std: {torch.tensor(hessian_values['SGD'][0]).std()}")
    print(f" HESS8 - Mean: {torch.tensor(hessian_values['SGD'][8]).mean()}    Std: {torch.tensor(hessian_values['SGD'][8]).std()}")
    print(f" Fish0 - Mean: {torch.tensor(fisher_values['SGD'][0]).mean()}    Std: {torch.tensor(fisher_values['SGD'][0]).std()}")
    print(f" Fish8 - Mean: {torch.tensor(fisher_values['SGD'][8]).mean()}    Std: {torch.tensor(fisher_values['SGD'][8]).std()}")

    print("\nMuon - Mean: %.4f    Std: %.4f" % (muon_accs.mean(), muon_accs.std()))
    print(f" SAM0 - Mean: {torch.tensor(sam_values['Muon'][0]).mean()}    Std: {torch.tensor(sam_values['Muon'][0]).std()}")
    print(f" SAM8 - Mean: {torch.tensor(sam_values['Muon'][8]).mean()}    Std: {torch.tensor(sam_values['Muon'][8]).std()}")
    print(f" HESS0 - Mean: {torch.tensor(hessian_values['Muon'][0]).mean()}    Std: {torch.tensor(hessian_values['Muon'][0]).std()}")
    print(f" HESS8 - Mean: {torch.tensor(hessian_values['Muon'][8]).mean()}    Std: {torch.tensor(hessian_values['Muon'][8]).std()}")
    print(f" Fish0 - Mean: {torch.tensor(fisher_values['Muon'][0]).mean()}    Std: {torch.tensor(fisher_values['Muon'][0]).std()}")
    print(f" Fish8 - Mean: {torch.tensor(fisher_values['Muon'][8]).mean()}    Std: {torch.tensor(fisher_values['Muon'][8]).std()}")