import torch
from airbench import infer, evaluate, CifarLoader
from airbench.lib_nofrills import train94, make_net94
from pyhessian import hessian
from pathlib import Path
import os
import time

CHECKPOINT_NAME = "muon"

username = os.getenv("USER")
CHECKPOINT_FOLDER = Path(f"/work/scratch/{username}/deep-learning/checkpoints/")

REPO_ROOT = Path(__file__).resolve().parent
CIFAR10_PATH = (REPO_ROOT / "cifar10").as_posix()

train_loader_train = CifarLoader(
    CIFAR10_PATH,
    train=True,
    batch_size=1024,
    aug=dict(flip=False, translate=0),
    altflip=True,
)

loader = CifarLoader(
    CIFAR10_PATH,
    train=True,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
)
train_loader_hess = CifarLoader(
    CIFAR10_PATH,
    train=True,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
    shuffle=False,
    drop_last=False,
)

test_loader_hess = CifarLoader(
    CIFAR10_PATH,
    train=False,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
    shuffle=False,
    drop_last=False,
)

print(f"{time.ctime()} Training starting...", flush=True)
net = train94(train_loader=train_loader_train, label_smoothing=0)
net.eval()

# save
CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), CHECKPOINT_FOLDER / f"{CHECKPOINT_NAME}.pt")

logits = infer(net, loader)
conf = logits.log_softmax(1).amax(1) # confidence
print(conf)

# get hessian and compute sharpness
print("Computing Hessian...")
criterion = torch.nn.CrossEntropyLoss()

print("Computing Hessian (train split)...", flush=True)
hessian_comp_train = hessian(net, criterion, dataloader=train_loader_hess, cuda=True)
top_eigenvalues_train, _ = hessian_comp_train.eigenvalues(top_n=1)
print("Top Eigenvalue (train): ", top_eigenvalues_train[-1])

print("Computing Hessian (test split)...", flush=True)
hessian_comp_test = hessian(net, criterion, dataloader=test_loader_hess, cuda=True)
top_eigenvalues_test, _ = hessian_comp_test.eigenvalues(top_n=1)
print("Top Eigenvalue (test): ", top_eigenvalues_test[-1])
