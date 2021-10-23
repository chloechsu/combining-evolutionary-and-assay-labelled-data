#!/bin/bash
#
# Infers log likelihoods from UniRep LSTM models
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=unirep_inf
#SBATCH --output=logs/unirep_inf.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=128000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

model_path=$1
data_path=$2
output_dir=$3
flags=$4

python src/unirep_inference.py $model_path $data_path $output_dir $flags --batch_size 32
