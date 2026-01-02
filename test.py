import torch
from airbench import train94, train96, infer, evaluate, CifarLoader
from pyhessian import hessian


def get_sam_sharpness(model, loader, rho=0.05):
    # Ensure we start with clean gradients
    model.zero_grad()
    
    # 1. Get a batch and compute loss
    # Using next(iter()) is fast, but make sure the loader is small!
    images, labels = next(iter(loader))
    images, labels = images.cuda(), labels.cuda()
    
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    
    # 2. Compute the gradient norm, filtering out None grads
    # This handles BatchNorm and other non-differentiable params safely
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    
    # 3. Perturb weights in the worst-case direction
    scale = rho / (grad_norm + 1e-12)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                e_w = p.grad * scale
                p.add_(e_w) 
                p.adv_eps = e_w # Store it to undo later

    # 4. Compute loss at the "sharp" spot
    adv_loss = torch.nn.functional.cross_entropy(model(images), labels)
    
    # 5. Restore original weights
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'adv_eps'):
                p.sub_(p.adv_eps)
                del p.adv_eps # Clean up
    
    model.zero_grad()
    return (adv_loss - loss).item()

hessian_loader = CifarLoader('cifar10', train=True, batch_size=1000)
hessian_criterion = torch.nn.CrossEntropyLoss()
def callback_test(step, model):
    if step % 50 == 0:
        model.eval()
        print(f"Step {step}")
        sam_sharp = get_sam_sharpness(model, hessian_loader, rho=0.05)
        print(f"  SAM Sharpness: {sam_sharp}")

    # if step % 200 == 0:
    #     hessian_comp = hessian(model, hessian_criterion, dataloader=hessian_loader, cuda=True)
    #     (top_eigenvalue, ), _ = hessian_comp.eigenvalues(top_n=1)
    #     print(f"  Sharpness {top_eigenvalue}")
        
    model.train()
        


net = train96(label_smoothing=0, callback=callback_test) # train this network without label smoothing to get a better confidence signal
