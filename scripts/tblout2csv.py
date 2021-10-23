'''
Converts tblout outputs from hmmer to csv.
'''

import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    column_names=['target', 'accession', 'query_name', 'accession_1',
            'E-value_full', 'score_full', 'bias_full', 'E-value_domain',
            'score_domain', 'bias_domain', 'exp', 'reg', 'clu', 'ov', 'env', 'dom',
            'rep', 'inc', 'target_description']
    df = pd.read_csv(args.input_file, delim_whitespace=True, comment='#',
            index_col=False, names=column_names)
    df.to_csv(args.output_file, index=False)
