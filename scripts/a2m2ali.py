'''
Script for converting .a2m alignment file format to .ali.
To be used with the integrative Potts model coDCA package.
'''

import argparse
from Bio import SeqIO

alphabet = "ACDEFGHIKLMNPQRSTVWY"
aa_to_int = {c: i for i, c in enumerate(alphabet)}


def seq2ints(s, focus_cols):
    ints = [1+aa_to_int.get(s[i], 20) for i in focus_cols]
    return ' '.join([str(a) for a in ints])

def seqdistance(s1, s2):
    assert len(s1) == len(s2)
    diff = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff += 1
    return float(diff) / float(len(s1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--theta', default=0.2, type=float)
    args = parser.parse_args()
    records = SeqIO.parse(args.input_file, "fasta")
    count = 0
    wt = str(list(records)[0].seq)
    focus_cols = [i for i in range(len(wt)) if wt[i].isupper()]
    records = SeqIO.parse(args.input_file, "fasta")
    with open(args.output_file, 'w+') as seq_list:
        for record in records:
            seq_list.write(seq2ints(str(record.seq), focus_cols))
            seq_list.write('\n')
            count += 1
    print("Converted %i records" % count)
