# Code for ''A systematic assessment of methods for combining evolutionary and assay-labelled data for protein fitness prediction''

### CONFIDENTIAL FOR REVIEW PURPOSES ONLY

## Contents
- Overview
- Repo contents
- System requirements
- Installation guide
- Demo
- Jackhmmer search
- Fitness data
- Density models
- Predictors
- Results

## Overview
This repo is a collection of code, scripts, and data for evaluating the methods
that combine evolutionary and assay-labelled data for protein fitness prediction.

## Repo contents
There are several main components of the repo.
- `data`: Processed protein fitness data.
- `alignments`: Processed multiple sequence alignments.
- `scripts`: Scripts for retrieving evolutionary sequences, using jackhmmer.
- `src`: Pipelines for training various evolutionary models, mostly based on existing
  packages but with custom modifications when necessary. Evolutionary models
  include Potts model (EVmutation), VAE (DeepSequence), LSTM (UniRep), and
  Transformer (ESM).
  Also includes the evaluation and comparison framework of different predictors.
- `src/predictors`: Implementations of `predictors`. Each predictor represents a prediction
  strategy for protein fitness that depends on evolutionary data,
  assay-labelled data, or both.
- `inference`: Directory for intermediate files such as inferred sequence log likelihoods.
- `results`: Directory for results as csv files.
- `environment.yml`: Software dependencies for conda environment.

## System requirements

### Hardware requirements
Some of the methods, in particular DeepSequence VAE, UniRep mLSTM, and ESM
Transformer, require GPU for training and inference. The GPU code in this repo
has been tested on NVIDIA Quadro RTX 8000 GPU.

Evaluating all the methods each with 20 random seeds, 19 data sets, and 10
training setups would require a relatively long time on a single core. Our
evaluation code supports multiprocessing and has been tested on 32 cores.

For storing all intermediate files for all methods and all data sets,
approximately 100G of disk space will be needed.

### Software requirements
The code has been tested on Ubuntu 18.04.5 LTS (Bionic Beaver) with conda 4.10.0
and Python 3.8.5.
The (optional) slurm scripts have been tested on slurm 17.11.12.
The list of software dependencies are provided in the `environment.yml` file.

## Installation guide

1. Create the conda environment from the environment.yml file:
```
    conda env create -f environment.yml
```

2. Activate the new conda environment:
```
    conda activate protein_fitness_prediction
```

3. Install the [plmc package](https://github.com/debbiemarkslab/plmc):
```
    cd $HOME (or use another directory for plmc <directory_to_install_plmc> and
modify `scripts/plmc.sh` accordingly with the custom directory)
    git clone https://github.com/debbiemarkslab/plmc.git
    cd plmc
    make all-openmp
```

The installation should finish in a few minutes.

## Demo 
The one-hot linear model is the simplest example as it only requires
assay-labelled data. To evaluate the one-hot linear model on the Poly(A)-binding
protein (PABP) data with 240 training examples and 20 seeds on a single core:
```
    python src/evaluate.py PABP_YEAST_Fields2013-linear onehot --n_seeds=20 --n_threads=1 --n_train=240
```

When the program finishes, the results from the 20 runs will be available in the
file `results/PABP_YEAST_Fields2013-linear/results.csv`.

As another example that involves both evolutionary and assay-lablled data, here
we show the process to evaluate the augmented Potts model on the same protein.

The multiple sequence alignments (MSAs) are available in the `alignments`
directory for all proteins used in our assessment. For other proteins, 
MSAs can be retrieved by jackhmmer search (see the jackhmmer search section).

From the MSA, first run PLMC to estimate the couplings model:
```
    bash scripts/plmc.sh PABP_YEAST PABP_YEAST_Fields2013-linear 
```
The resulting models are saved at `inference/PABP_YEAST_Fields2013-linear/plmc`.

Then, similar to the one-hot linear model evaluation, run:
```
    python src/evaluate.py PABP_YEAST_Fields2013-linear ev+onehot --n_seeds=20 --n_threads=1 --n_train=240
```
The evaluation should finish in a few minutes, and all results will be saved to
`results/PABP_YEAST_Fields2013-linear/results.csv`.

Here, `ev+onehot` refers to the augmented Potts model. Other models and data
sets can also be similarly evaluated as long as the corresponding prerequisite
files are present in the inference directory.

## Jackhmmer search
1. Downloaded UniRef100 in fasta format from [UniProt](https://www.uniprot.org/downloads). 
2. Index the uniref100 fasta file into ssi with
```
    esl-sfetch --index <seqdb>.fasta
```
3. Set the file location of the fasta file in `scripts/jackhmmer.sh`.
4. To run jackhmmer, use `scripts/jackhmmer.sh` to search the local fasta
file.  In addition to running jackhmmer search, the script also implicitly calls
the other file conversion scripts. For example, it extracts target ids from the
jackhmmer tabular output by calling `scripts/tblout2ids.py`; converts the
fasta output to list of sequences by `scripts/fasta2txt.py`; and splits the
sequences into train and validation with `scripts/randsplit.py`.
5. The outputs of the jackhmmer script will be in
`jackhmmer/<dataset>/<run_id>`, where each iteration's alignment is saved as
`iter-<N>.a2m` and the final alignment is saved as `alignment.a2m`. The list of
full length target sequences is at `target_seqs.fasta` and `target_seqs.txt`.

## Fitness data
In the `data` directory, each subdirectory (e.g. `data/sarkisyan`) represents a
dataset of interest. In the subdirectory, there are two key files.
- `wt.fasta` documents the WT sequence.
- `data.csv` contains three columns: `seq`, `log_fitness`, `n_mut`.
  `seq` is the sequence with mutation, and should be the same length as WT seq.
  `log_fitness` is the log enrichment ratio or other log-scale fitness values,
  where higher is better.
  `n_mut` is how many mutations away the sequence is from WT, where 0 indicates WT.

## Density models

### Potts model
For learning a Potts model (EVmutation / plmc) from an MSA, see ``scripts/plmc.sh``.
The resulting couplings model files (saved to the inference directory) can be
directly parsed by our correpsonding `ev` and `ev+onehot` predictors.

### DeepSequence VAE
1. Install the [DeepSequence
package](https://github.com/debbiemarkslab/DeepSequence.git).
2. Put the DeepSequence package directory as `WORKING_DIR` in both `src/train_vae.py`
and `src/inference_vae.py`.
3. Use `scripts/train_vae.sh` for training a VAE model from an MSA.
4. For retrieving ELBOs from VAEs, see `scripts/inference_vae.sh`.
5. The saved elbo files in the inference directory can be parsed by the
corresponding `vae` and `vae+onehot` predictors.

### ESM
1. Follow the instructions from the [ESM
repo](https://github.com/facebookresearch/esm.git) to download the pre-trained
model weights.
2. Put the downloaded pre-trained weights location into
`scripts/inference_esm.sh`.
3. To retrieve ESM Transformer approximate pseudo-log likelihoods for sequences
in a fasta file, see `scripts/inference_esm.sh`. The results will be in the
inference directory and can be used by the `esm` and `esm+onehot` predictors.

### UniRep
1. Download the pre-trained UniRep weights (1900-unit) from the [UniRep
repo](https://github.com/churchlab/UniRep#obtaining-weight-files).
2. Put the location for the downloaded weights into `scripts/evotune_unirep.sh`.
3. Use the `scripts/evotune_unirep.sh` script to evotune the UniRep model with
an MSA file as `seqspath`.
4. Use `scripts/inference_unirep.sh` to calculate log-likelihoods from an
evotuned unirep model.

## Predictors
Each type of predictor is represented by a Python class.
The base predictor class, `BasePredictor` is defined at `src/predictors/base_predictors.py`.
All predictor classes inherit from this class and have the `train` and `predict` methods.
The `JointPredictor` class is a meta predictor that combines the features from multiple
existing predictor classes, and can be easily specified by the sub-predictor names.
See `src/predictors/__init__.py` for a full list of implemented predictors.

## Results 
The raw results for all methods and all data sets are provided in the
`results/all_results.csv` file.
