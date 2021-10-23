#!/bin/bash
#
# Evaluates the predictive performance of a predictor
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=evaluate
#SBATCH --output=logs/evaluate.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=16        # CPU cores/threads
#SBATCH --mem=128000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
#SBATCH --priority=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dataset=$1
predictor=$2
n_seeds=$3
n_train=$4
n_threads=16
kwargs=$5

python src/evaluate.py $dataset $predictor \
	--n_threads=$n_threads --n_seeds=$n_seeds \
	--n_train=$n_train --joint_training $kwargs
