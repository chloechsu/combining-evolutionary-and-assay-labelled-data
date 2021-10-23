#!/bin/bash
#
# Fine-tunes the UniRep model on evolutionary data.
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=evotune_unirep
#SBATCH --output=logs/uni_tune.out
#SBATCH --gres=gpu:1              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=128000M              # memory per node
#SBATCH --time=0-48:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

seqspath=$1
savedir=$2
steps=$3
initial_weights_dir=$4

python src/unirep_evotune.py $seqspath $savedir \
	--initial_weights_dir $initial_weights_dir \
	--num_steps $steps --batch_size 128 --max_seq_len 500
