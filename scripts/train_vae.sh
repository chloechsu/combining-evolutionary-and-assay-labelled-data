#!/bin/bash
#
# Trains a DeepSequence VAE model.
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=vae
#SBATCH --output=logs/vae.out
#SBATCH --gres=gpu:1              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=48000M              # memory per node
#SBATCH --time=0-36:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dataset=$1
seed=$2

THEANO_FLAGS='floatX=float32,device=cuda' python src/vae_train.py \
    ${dataset}.a2m ${dataset}_${seed} $seed
