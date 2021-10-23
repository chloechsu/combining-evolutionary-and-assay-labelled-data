#!/bin/bash
#
# Infers pseudo log likelihood approximations from ESM Transformer models
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=esm_inf
#SBATCH --gres=gpu:1              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=32000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dataset=$1
model_name=$2
model_location=$3

python src/esm_inference.py data/$dataset/seqs.fasta \
    data/$dataset/wt.fasta inference/${dataset}/esm/${model_name} \
    --model_location $model_location
    --toks_per_batch 4096;
