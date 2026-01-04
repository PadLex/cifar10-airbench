import torch
from airbench import infer, evaluate, CifarLoader
from airbench.lib_nofrills import train94, make_net94
from pathlib import Path
import os
import time

CHECKPOINT_NAME = "test"

username = os.getenv("USER")
CHECKPOINT_FOLDER = Path(f"/work/scratch/{username}/deep-learning/checkpoints/")

loader = CifarLoader(
    'cifar10',
    train=True,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
)

print(f"{time.ctime()} Training starting...", flush=True)
net = train94(label_smoothing=0)

# save
CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), CHECKPOINT_FOLDER / f"{CHECKPOINT_NAME}.pt")

logits = infer(net, loader)
conf = logits.log_softmax(1).amax(1) # confidence
print(conf)
