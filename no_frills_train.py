import torch
from airbench import infer, evaluate, CifarLoader
from airbench.lib_nofrills import train94, make_net94

loader = CifarLoader(
    'cifar10',
    train=True,
    batch_size=512,
    aug=dict(flip=False, translate=0, cutout=0),
    altflip=False,
)

net = train_nofrills(label_smoothing=0)
logits = infer(net, loader)
conf = logits.log_softmax(1).amax(1) # confidence
print(conf)
