'''
Converts txt file to fasta.
'''

import argparse
from Bio import SeqIO
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()
    count = 0
    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
        with open(args.output_file, 'w') as out:
            for s in df.seq.values:
                out.write(f'>id_{count}\n')
                out.write(s + '\n')
                count += 1
    else:
        with open(args.input_file, 'r') as in_file:
            with open(args.output_file, 'w') as out:
                for line in in_file.readlines():
                    out.write(f'>id_{count}\n')
                    out.write(line)
                    count += 1
    print("Converted %i records" % count)
