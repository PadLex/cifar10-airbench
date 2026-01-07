#!/bin/bash
#SBATCH --job-name=airbench_cifar10_train
#SBATCH --account=deep_learning
#SBATCH --time=01:00:00
#SBATCH --output=slurm/out/%x/%j.log
#SBATCH --error=slurm/out/%x/%j.log
#SBATCH --mail-user=blipshitz@ethz.ch
#SBATCH --mail-type=END,FAIL

# Ensure we are in the root directory of the repo
# This assumes the sbatch is submitted from the root or we navigate to it.
# If submitted from root:
# cd . 

nvidia-smi

# Run the training script
uv run train_get_hessian.py
