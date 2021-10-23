'''
Converts fasta files to txt format.
'''
import argparse
from Bio import SeqIO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--extract_ids', action='store_true')
    args = parser.parse_args()
    records = SeqIO.parse(args.input_file, "fasta")
    count = 0
    with open(args.output_file, 'w+') as seq_list:
        for record in records:
            if args.extract_ids:
                ids = str(record.id).split('|')
                if len(ids) == 1:
                    seq_list.write(ids[0])
                else:
                    seq_list.write(ids[1])
            else:
                seq_list.write(str(record.seq))
            seq_list.write('\n')
            count += 1
    #SeqIO.write(records, args.output_file, "tab")
    print("Converted %i records" % count)
