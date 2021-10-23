import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from utils import seqs_to_onehot, get_wt_seq, read_fasta, seq2effect, mutant2seq
from predictors.base_predictors import BaseRegressionPredictor

# EVMutation imports
from couplings_model import CouplingsModel


class EVPredictor(BaseRegressionPredictor):
    """plmc mutation effect prediction."""

    def __init__(self, dataset_name, model_name='uniref100',
        reg_coef=1e-8, ignore_gaps=False, **kwargs):
        super(EVPredictor, self).__init__(dataset_name, reg_coef=reg_coef,
                **kwargs)
        self.ignore_gaps = ignore_gaps
        self.couplings_model_path = os.path.join('inference', dataset_name,
                'plmc', model_name + '.model_params')
        self.couplings_model = CouplingsModel(self.couplings_model_path)
        wtseqs, wtids = read_fasta(os.path.join('data', dataset_name,
            'wt.fasta'), return_ids=True)
        if '/' in wtids[0]:
            self.offset = int(wtids[0].split('/')[-1].split('-')[0])
        else:
            self.offset = 1
        expected_wt = wtseqs[0]
        for pf, pm in self.couplings_model.index_map.items():
            if expected_wt[pf-self.offset] != self.couplings_model.target_seq[pm]:
                print(f'WT and model target seq mismatch at {pf}')

    def seq2score(self, seqs):
        return seq2effect(seqs, self.couplings_model, self.offset,
                ignore_gaps=self.ignore_gaps)

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)
