'''
Trains a DeepSequence VAE model.
Based on open source code from DeepSequence repo.
'''

import argparse
import numpy as np
import os
import sys
import time

WORKING_DIR=""  # Put in the deepsequence directory
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
    "conv_pat"          :   True,
    "d_c_size"          :   40
}

train_params = {
    "num_updates"       :   300000,
    "save_progress"     :   True,
    "verbose"           :   True,
    "save_parameters"   :   False,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('alignment_file', type=str)
    parser.add_argument('job_string', type=str)
    parser.add_argument('seed', type=int)
    args = parser.parse_args()
    data_helper = helper.DataHelper(
            working_dir=WORKING_DIR,
            alignment_file=args.alignment_file,
            calc_weights=True
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
        random_seed                    =   args.seed,
        working_dir                    =   WORKING_DIR,
        )

    data_params = {'alignment_file': args.alignment_file}
    job_string = args.job_string

    train.train(data_helper, vae_model,
        num_updates             =   train_params["num_updates"],
        save_progress           =   train_params["save_progress"],
        save_parameters         =   train_params["save_parameters"],
        verbose                 =   train_params["verbose"],
        job_string              =   job_string)

    vae_model.save_parameters(file_prefix=job_string)


if __name__ == "__main__":
    main()
