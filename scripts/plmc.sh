#!/bin/bash
#
# Estimates couplings model from alignment with plmc package
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=plmc
#SBATCH --output=logs/plmc.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=2         # CPU cores/threads
#SBATCH --mem=4000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROTEIN=$1
DATASET=$2

PLMC_DIR="$HOME"
    
mkdir -p inference/$DATASET/plmc

$PLMC_DIR/plmc/bin/plmc \
    -o inference/$DATASET/plmc/uniref100.model_params \
    -c inference/$DATASET/plmc/uniref100.EC \
    -f $PROTEIN \
    -le 16.2 -lh 0.01 -m 200 -t 0.2 \
    -g alignments/$PROTEIN.a2m
