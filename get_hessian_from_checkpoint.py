import torch
from airbench import CifarLoader
from airbench.lib_nofrills import make_net94
from pyhessian import hessian
from pathlib import Path
import os
import time

CHECKPOINT_NAME = "muon"

username = os.getenv("USER")
CHECKPOINT_FOLDER = Path(f"/work/scratch/{username}/deep-learning/checkpoints/")
CHECKPOINT_PATH = CHECKPOINT_FOLDER / f"{CHECKPOINT_NAME}.pt"

REPO_ROOT = Path(__file__).resolve().parent
CIFAR10_PATH = (REPO_ROOT / "cifar10").as_posix()

train_loader = CifarLoader(
    CIFAR10_PATH,
    train=True,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
    shuffle=False,
    drop_last=False,
)

test_loader = CifarLoader(
    CIFAR10_PATH,
    train=False,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
    shuffle=False,
    drop_last=False,
)

print(f"{time.ctime()} Loading checkpoint...", flush=True)
net = make_net94()
net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cuda'))
net.eval()

criterion = torch.nn.CrossEntropyLoss()

print("Computing Hessian (train split)...", flush=True)
hessian_comp_train = hessian(net, criterion, dataloader=train_loader, cuda=True)
top_eigenvalues_train, _ = hessian_comp_train.eigenvalues(top_n=1)
print("Top Eigenvalue (train): ", top_eigenvalues_train[-1])

print("Computing Hessian (test split)...", flush=True)
hessian_comp_test = hessian(net, criterion, dataloader=test_loader, cuda=True)
top_eigenvalues_test, _ = hessian_comp_test.eigenvalues(top_n=1)
print("Top Eigenvalue (test): ", top_eigenvalues_test[-1])
