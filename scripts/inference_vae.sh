#!/bin/bash
#
# Infers average ELBO values from DeepSequence VAE models
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=vae_inf
#SBATCH --gres=gpu:1              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=48000M              # memory per node
#SBATCH --time=0-36:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dataset=$1
seqsfile=$2
model=$3

THEANO_FLAGS='floatX=float32,device=cuda' python src/vae_inference.py \
    $model $seqsfile data/$dataset/wt.fasta \
    inference/${dataset}/vae/${model}
