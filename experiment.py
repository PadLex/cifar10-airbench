import uuid
from airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, CifarLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def get_sam_sharpness(model, loader, rho=0.05, num_batches=10):
    model.eval()
    device = next(model.parameters()).device
    total_sharpness = 0.0

    g = torch.Generator()
    g.manual_seed(42) 
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

sam_values = {}
def sam_callback(name):
    hessian_loader = CifarLoader('cifar10', train=True, batch_size=1000)
    sam_values[name] = {i: [] for i in range(8)}

    def callback_test(epoch, model, *args):
        model.eval()
        sam_sharp = get_sam_sharpness(model, hessian_loader)
        sam_values[name][epoch].append(sam_sharp)

        # print(f"[{name}] Epoch {epoch}  SAM Sharpness: {sam_sharp}")
        model.train()
    
    return callback_test


if __name__ == "__main__":

    device = "cuda"
    model = CifarNet().to(device).to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")
    num_runs = 30

    # 1. Warmup
    print("Performing Warmup...")
    train("Warmup", model, MuonConfig(batch_size=2000))

    # 3. Test SGD
    sgd_callback = sam_callback("SGD")
    sgd_accs = torch.tensor([train(run, model, SGDConfig(), callback=sgd_callback) for run in tqdm(range(num_runs))])
    print("\nSGD  - Mean: %.4f    Std: %.4f" % (sgd_accs.mean(), sgd_accs.std()))
    print(f" SAM0 - Mean: {torch.tensor(sam_values['SGD'][0]).mean()}    Std: {torch.tensor(sam_values['SGD'][0]).std()}")
    print(f" SAM7 - Mean: {torch.tensor(sam_values['SGD'][7]).mean()}    Std: {torch.tensor(sam_values['SGD'][7]).std()}")\
    

    # 2. Test Muon
    muon_callback = sam_callback("Muon")
    muon_accs = torch.tensor([train(run, model, MuonConfig(), callback=muon_callback) for run in tqdm(range(num_runs))])
    print("\nMuon - Mean: %.4f    Std: %.4f" % (muon_accs.mean(), muon_accs.std()))
    print(f" SAM0 - Mean: {torch.tensor(sam_values['Muon'][0]).mean()}    Std: {torch.tensor(sam_values['Muon'][0]).std()}")
    print(f" SAM7 - Mean: {torch.tensor(sam_values['Muon'][7]).mean()}    Std: {torch.tensor(sam_values['Muon'][7]).std()}")