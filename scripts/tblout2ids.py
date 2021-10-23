'''
Converts tblout outputs from hmmer to uniprot ids.
'''
import argparse
import csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    with open(args.input_file) as tbl_file:
        with open(args.output_file, 'w+') as id_list:
            for line in tbl_file.readlines():
                # skip comments
                if line[0] == '#' or len(line) == 0:
                    continue
                id_list.write(line.split(' ')[0])
                id_list.write('\n')
