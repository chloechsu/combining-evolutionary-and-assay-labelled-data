#!/bin/bash
#
# Infers log-likelihoods from an HMM and writes results to a CSV file.
#
#SBATCH --cluster=<clustername>
#SBATCH --partition=<partitionname>      
#SBATCH --account=<accountname>
#SBATCH --job-name=hmminf
#SBATCH --output=logs/hmminf.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=1         # CPU cores/threads
#SBATCH --mem=4000M               # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading

dataset=$1
hmmpath=$2
runid=$3
dir="inference/${dataset}/hmm"

mkdir -p $dir
hmmsearch --tblout ${dir}/${runid}.tblout $hmmpath data/${dataset}/seqs.fasta
python scripts/tblout2csv.py ${dir}/${runid}.tblout ${dir}/${runid}.csv
