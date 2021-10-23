#!/bin/bash
#
# Runs jackhmmer search with bitscore thresholds
#
#SBATCH --cluster=<clustername> 
#SBATCH --partition=<partitionname>
#SBATCH --account=<accountname>
#SBATCH --job-name=jackhmmer
#SBATCH --output=jackhmmer.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=48000M              # memory per node
#SBATCH --time=0-24:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


## USAGE
## Create a directory, and put the WT sequence in wt.fasta in the directory
## sbatch jackhmmer.sh <dir> <bitscore_threshold> <niter>

dir=$1
bitscore=$2   # e.g. 0.5
niter=$3
seqdb=$4      # location for e.g. uniref100 or uniref90 fasta files

query="$dir/wt.fasta"
tblout="$dir/targets.tblout"
alignmentfile="$dir/alignment.sto"
hmmprefix="$dir/iter"
aliprefix="$dir/iter"

wtseq=$(sed 1d $query)
seqlen=${#wtseq}
bitscore=$(echo "$seqlen*$bitscore" | bc)   # scale bitscore by seqlen
echo "$bitscore"

#EVcouplings defaults
jackhmmer -N $niter \
    --incT $bitscore --incdomT $bitscore -T $bitscore --domT $bitscore \
    --popen 0.02 --pextend 0.4 --mx BLOSUM62 \
    --tblout $tblout -A $alignmentfile --noali --notextw\
    --chkhmm $hmmprefix --chkali $aliprefix \
    --cpu $SLURM_CPUS_PER_TASK \
    $query $seqdb

# convert tblout to target id list
targetidfile="$dir/target_ids.txt"
python scripts/tblout2ids.py $tblout $targetidfile

# fetch sequences
fastafile="$dir/target_seqs.fasta"
txtfile="$dir/target_seqs.txt"
esl-sfetch -o $fastafile -f $seqdb $targetidfile
python scripts/fasta2txt.py $fastafile $txtfile

# split into train and validation
python scripts/randsplit.py $txtfile 0.2

python src/sto2a2m.py $query $alignmentfile ${dir}/alignment
for (( i=1; i<=$niter; i++ ))
do
    python src/sto2a2m.py $query $aliprefix-$i.sto $aliprefix-$i
done
