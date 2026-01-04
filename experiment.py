import uuid
from airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, CifarLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pyhessian import hessian
from itertools import islice
import copy
import numpy as np

def pyhessian_sharpness(model, loader, num_batches=10):
    """
    Computes relative sharpness by scaling the top Hessian eigenvalue 
    by the squared norm of the model parameters.
    """

    # 2. Compute the top eigenvalue (Sharpness)
    criterion = torch.nn.CrossEntropyLoss()
    # Ensure loader is accessible for PyHessian
    limited_loader = list(islice(loader, num_batches))
    
    hessian_comp = hessian(model, criterion, dataloader=limited_loader, cuda=True)
    lambda_max = hessian_comp.eigenvalues(top_n=1)[0][0]
    print(f"Top Hessian Eigenvalue: {lambda_max}")

    # 3. Compute ||W||_2^2 (Squared Frobenius Norm of all parameters)
    # This is required for normalizing to "Relative Flatness"
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm_sq += torch.norm(p.data)**2
    
    # Calculate relative sharpness: lambda_max / ||W||_2^2
    relative_sharpness = lambda_max / total_norm_sq.item()

    # Clean up to save VRAM
    # del model_to_use
    # torch.cuda.empty_cache()

    # model_to_use.train()

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
def sam_callback(name):
    hessian_loader = CifarLoader('cifar10', train=True, batch_size=1000)
    sam_values[name] = {i: [] for i in range(9)}
    hessian_values[name] = {i: [] for i in range(9)}


    def callback_test(epoch, model, *args):
        model.eval()
        sam_sharp = get_sam_sharpness(model, hessian_loader)
        sam_values[name][epoch].append(sam_sharp)

        # print(f"Calculating Hessian for epoch {epoch}...")
        if epoch == 8:
            hess_sharp = pyhessian_sharpness(model, hessian_loader)
        else:
            hess_sharp = -1.0
        hessian_values[name][epoch].append(hess_sharp)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}  Hessian Top Eigenvalue: {hess_sharp}")
        model.train()
    
    return callback_test


if __name__ == "__main__":

    device = "cuda"
    model = CifarNet().to(device).to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune")
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

    print("\nMuon - Mean: %.4f    Std: %.4f" % (muon_accs.mean(), muon_accs.std()))
    print(f" SAM0 - Mean: {torch.tensor(sam_values['Muon'][0]).mean()}    Std: {torch.tensor(sam_values['Muon'][0]).std()}")
    print(f" SAM8 - Mean: {torch.tensor(sam_values['Muon'][8]).mean()}    Std: {torch.tensor(sam_values['Muon'][8]).std()}")
    print(f" HESS0 - Mean: {torch.tensor(hessian_values['Muon'][0]).mean()}    Std: {torch.tensor(hessian_values['Muon'][0]).std()}")
    print(f" HESS8 - Mean: {torch.tensor(hessian_values['Muon'][8]).mean()}    Std: {torch.tensor(hessian_values['Muon'][8]).std()}")