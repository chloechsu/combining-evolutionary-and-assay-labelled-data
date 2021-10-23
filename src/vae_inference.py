'''
Infers average ELBO values from DeepSequence VAE models.
Based on open source code from DeepSequence repo.
'''
import argparse
import numpy as np
import os
import pathlib
from shutil import copyfile
import sys
import time

import utils

WORKING_DIR="" # Put in the DeepSequence directory here
N_ELBO_SAMPLES=400

module_path = os.path.abspath(WORKING_DIR)
if module_path not in sys.path:
        sys.path.append(module_path)

from DeepSequence.model import VariationalAutoencoder
from DeepSequence import helper
from DeepSequence import train

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "final_decode_nonlin":  "sigmoid",
    "final_pwm_scale"   :   True,
    "n_pat"             :   4,
    "r_seed"            :   12345,
    "conv_pat"          :   True,
    "d_c_size"          :   40
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_prefix", type=str)
    parser.add_argument("fasta_file", type=str)
    parser.add_argument("wt_fasta_file", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    data_helper = helper.DataHelper(
            working_dir=WORKING_DIR,
            alignment_file=args.fasta_file,
            calc_weights=False,
    )

    vae_model   = VariationalAutoencoder(data_helper,
        batch_size                     =   model_params["bs"],
        encoder_architecture           =   [model_params["encode_dim_zero"],
                                                model_params["encode_dim_one"]],
        decoder_architecture           =   [model_params["decode_dim_zero"],
                                                model_params["decode_dim_one"]],
        n_latent                       =   model_params["n_latent"],
        logit_p                        =   model_params["logit_p"],
        sparsity                       =   model_params["sparsity"],
        encode_nonlinearity_type       =   "relu",
        decode_nonlinearity_type       =   "relu",
        final_decode_nonlinearity      =   model_params["final_decode_nonlin"],
        final_pwm_scale                =   model_params["final_pwm_scale"],
        conv_decoder_size              =   model_params["d_c_size"],
        convolve_patterns              =   model_params["conv_pat"],
        n_patterns                     =   model_params["n_pat"],
        random_seed                    =   model_params["r_seed"],
        working_dir                    =   WORKING_DIR,
        )
    vae_model.load_parameters(args.model_prefix)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    focuscols = set(data_helper.uniprot_focus_cols_list)

    seqs = utils.read_fasta(os.path.join(WORKING_DIR, "datasets",
        args.fasta_file))
    wt, des = utils.read_fasta(args.wt_fasta_file, return_ids=True)
    wt = wt[0]
    des = des[0]
    offset = int(des.split('/')[-1].split('-')[0])
    delta_elbos = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        if i % 100 == 0:
            print(f'Computed elbos for {i} out of {len(seqs)} seqs')
            np.savetxt(os.path.join(args.output_dir, "elbo.npy"), delta_elbos)
        mut_tups = utils.seq2mutation_fromwt(s, wt, offset=offset)
        mut_tups = [t for t in mut_tups if t[0] in focuscols]
        delta_elbos[i] = data_helper.delta_elbo(vae_model, mut_tups,
                N_pred_iterations=N_ELBO_SAMPLES) 
    np.savetxt(os.path.join(args.output_dir, "elbo.npy"), delta_elbos)


if __name__ == "__main__":
    main()
