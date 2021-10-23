'''
Converts ESM model state checkpoints to pytorch model data.
'''

import argparse
import os
import pathlib
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert model state to model data for ESM."
    )
    parser.add_argument(
        "model_data_location",
        type=pathlib.Path,
        help="Model data filepath",
    )
    parser.add_argument(
        "model_state_location",
        type=pathlib.Path,
        help="Model state filepath",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory",
    )
    return parser



def main(args):
	model_data = torch.load(args.model_data_location, map_location='cpu')
	state = torch.load(args.model_state_location, map_location='cpu')
	model_data['model'] = state
	args.output_dir.mkdir(parents=True, exist_ok=True)
	torch.save(model_data, os.path.join(args.output_dir, 'model_data.pt'))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
