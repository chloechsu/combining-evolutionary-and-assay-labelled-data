'''
Converts .sto alignment format to .a2m format.
'''
import argparse
from evcouplings.align.alignment import Alignment
from evcouplings.align.protocol import modify_alignment, cut_sequence
import numpy as np


from utils import read_fasta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_seq_file', type=str,
            help='input filepath for the target sequence in fasta')
    parser.add_argument('sto_alignment_file', type=str, help='input filepath for .sto')
    parser.add_argument('output_prefix', type=str, help='output filepath prefix')
    args = parser.parse_args()

    with open(args.sto_alignment_file) as a:
        ali_raw = Alignment.from_file(a, "stockholm")

    # center alignment around focus/search sequence
    focus_cols = np.array([c != "-" for c in ali_raw[0]])
    focus_ali = ali_raw.select(columns=focus_cols)

    target_seq, target_id = read_fasta(args.target_seq_file, return_ids=True)
    assert len(target_seq) == 1, 'more than 1 target seq'
    target_seq = target_seq[0]
    target_id = target_id[0]
    assert len(target_seq) == len(focus_ali[0]), (
        f'{len(focus_cols)} focus cols, expected {len(target_seq)}')

    target_seq_index = 0
    region_start = 0
    kwargs = {
            'prefix': args.output_prefix,
            'seqid_filter': None,
            'hhfilter': None,
            'minimum_sequence_coverage': 50,
            'minimum_column_coverage': 70,    # The default is 70 but use 0 to cover all columns
            'compute_num_effective_seqs': False,
            'theta': 0.8,
    }
    mod_outcfg, ali = modify_alignment(
        focus_ali, target_seq_index, target_id, region_start, **kwargs)


if __name__ == "__main__":
    main()
