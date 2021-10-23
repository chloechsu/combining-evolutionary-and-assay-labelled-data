#!/bin/bash
#
# Finetunes the ESM-1b Transformer model on supervised data.
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=esm_finetune
#SBATCH --gres=gpu:1              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=32000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dataset=$1
runid=$2
epochs=$3
n_train=$4
seed=$5
model_name="esm1b"
init_model="/mnt/esm_weights/${model_name}.pt"
kwargs=$6

python src/esm_finetune.py data_esm/${dataset}/data.csv \
	data_esm/${dataset}/wt.fasta \
	/mnt/inference/${dataset}/esm_finetune/${runid} \
	--epochs $epochs --n_train $n_train --toks_per_batch 512 \
	--model_location $init_model --seed $seed \
	$kwargs
