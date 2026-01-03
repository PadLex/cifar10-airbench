import torch
from airbench import infer, evaluate, CifarLoader
from airbench.lib_nofrills import train94, make_net94
from pathlib import Path

CHECKPOINT_FOLDER = Path("/work/scratch/deep-learning/checkpoints/")
CHECKPOINT_NAME = "test"

loader = CifarLoader(
    'cifar10',
    train=True,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
)

net = train94(label_smoothing=0)

# save
CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), CHECKPOINT_FOLDER / f"{CHECKPOINT_NAME}.pt")

logits = infer(net, loader)
conf = logits.log_softmax(1).amax(1) # confidence
print(conf)
