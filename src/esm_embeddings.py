'''
Extracts embeddings from ESM models.
'''
import argparse
from collections import defaultdict
import os
import pathlib

import numpy as np
import pandas as pd
import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, BatchConverter

from utils import read_fasta, save

criterion = torch.nn.CrossEntropyLoss(reduction='none')


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "wt_fasta_file",
        type=pathlib.Path,
        help="FASTA file for WT",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output dir",
    )
    parser.add_argument(
        "--model_location",
        type=str,
        help="model location",
        default="/mnt/esm_weights/esm1b/esm1b_t33_650M_UR50S.pt"
    )
    parser.add_argument(
        "--save_hidden", type=bool, default=False, help="whether to save rep"
    )
    parser.add_argument(
        "--toks_per_batch", type=int, default=4096, help="maximum batch size"
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    batch_converter = alphabet.get_batch_converter()
    padding_idx = torch.tensor(alphabet.padding_idx)

    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=batch_converter, batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    repr_layers = [model.num_layers]   # extract last layer

    label_vals = []
    avg_rep_vals = []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers,
                    return_contacts=False)
            # [B, T, E]
            final_layer = out["representations"][model.num_layers]
            notpad = torch.unsqueeze(toks != padding_idx, 2)
            avg_rep = (final_layer * notpad).mean(dim=1).to(
                    device="cpu").numpy()
            avg_rep_vals.append(avg_rep)
            label_vals.append(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    avg_rep_vals = np.concatenate(avg_rep_vals, axis=0)
    label_vals = np.concatenate(label_vals)
    np.savetxt(os.path.join(args.output_dir, 'labels.npy'),
            label_vals, fmt="%s")
    save(os.path.join(args.output_dir, 'rep.npy'), avg_rep_vals)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
