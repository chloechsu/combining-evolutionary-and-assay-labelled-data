'''
Extracts UniProt ids from .sto alignment files.
'''

import argparse
from evcouplings.align.alignment import Alignment
from evcouplings.align.protocol import modify_alignment, cut_sequence
import numpy as np


from utils import read_fasta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sto_alignment_file', type=str, help='input filepath for .sto')
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    with open(args.sto_alignment_file) as a:
        ali_raw = Alignment.from_file(a, "stockholm")
    ids = np.unique(ali_raw.ids)
    with open(args.output_file, 'w') as f:
        for i in ids:
            f.write(i.split('|')[1])
            f.write('\n')


if __name__ == "__main__":
    main()
