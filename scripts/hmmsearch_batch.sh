#!/bin/bash
#SBATCH --cluster=beef            # Don"t change
#SBATCH --partition=long          # Don"t change
#SBATCH --account=researcher      # Don"t change
#SBATCH --job-name=hmmsearch
#SBATCH --output=hmmsearch.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=32000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

hmmdir=$1
hmmprefix=$2
seqfile=$3  # fasta file of sequences
outdir=$4

mkdir -p $outdir

for hmmpath in $hmmdir/$hmmprefix*.hmm; do
    hmmfile="${hmmpath##*/}"
    hmmsearch --tblout $outdir/$hmmfile.tblout $hmmpath $seqfile
    python scripts/tblout2csv.py $outdir/$hmmfile.tblout $outdir/$hmmfile.csv
done
