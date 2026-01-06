import torch.multiprocessing as mp
import wandb
import torch
from airbench94_muon import CifarNet, train, MuonConfig, SGDConfig, AdamConfig

NUM_GPUS = 4
NUM_RUNS = 512


# 1. Define your sweep config (keep as you had it)
muon_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'muon_lr': {'min': 0.05, 'max': 0.5},
        'muon_momentum': {'min': 0.0, 'max': 0.9},
        'sgd_momentum': {'min': 0.0, 'max': 0.95},
        'bias_lr': {'min': 0.01, 'max': 0.1},
        'head_lr': {'min': 0.1, 'max': 1.0},
    }
}


sgd_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'bias_lr': {'min': 0.01, 'max': 0.1},
        'head_lr': {'min': 0.1, 'max': 1.0},
        'filter_lr': {'min': 0.01, 'max': 0.5},
        'momentum': {'min': 0.0, 'max': 0.95},
    }
}


adam_sweep = {
    'method': 'bayes',
    'run_cap': NUM_RUNS,
    'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
    'parameters': {
        'bias_lr': {'min': 0.01, 'max': 0.1},
        'head_lr': {'min': 0.1, 'max': 1.0},
        'filter_lr': {'min': 0.0001, 'max': 0.5},
        'beta1': {'min': 0.8, 'max': 0.99},
        'beta2': {'min': 0.99, 'max': 0.9999},
    }
}


def sweep_worker(sweep_id, device_id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model = CifarNet().to(f'cuda').to(memory_format=torch.channels_last)
    # model = torch.compile(model, mode="max-autotune")
    

    def train_wrapper():
        with wandb.init() as run:
            # Extract parameters from wandb.config
            acc = train(run, model, optimizer(**wandb.config), epochs=16)
            wandb.log({"val_accuracy": acc})

    # Start the agent on this specific process
    wandb.agent(sweep_id, function=train_wrapper)


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep=muon_sweep, project="muon-tuning")
    optimizer = MuonConfig

    # sweep_id = wandb.sweep(sweep=sgd_sweep, project="sgd-tuning")
    # optimizer = SGDConfig

    # sweep_id = wandb.sweep(sweep=adam_sweep, project="adam-tuning")
    # optimizer = AdamConfig
    

    processes = []
    for i in range(NUM_GPUS):
        p = mp.Process(target=sweep_worker, args=(sweep_id, i))
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()