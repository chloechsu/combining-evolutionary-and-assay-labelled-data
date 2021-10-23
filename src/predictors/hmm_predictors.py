import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from utils import seqs_to_onehot, read_fasta
from predictors.base_predictors import BaseRegressionPredictor


class HMMPredictor(BaseRegressionPredictor):
    """HMM likelihood as features in regression."""

    def __init__(self, dataset_name, model_name='uniref100_b0.5',
        reg_coef=1e-8, **kwargs):
        super(HMMPredictor, self).__init__(dataset_name, reg_coef=reg_coef,
             **kwargs)
        seqs_path = os.path.join('data', dataset_name, 'seqs.fasta')
        hmm_seqs = read_fasta(seqs_path)
        id2seq = pd.Series(index=np.arange(len(hmm_seqs)), data=hmm_seqs,
                name='seq')

        hmm_data_path = os.path.join('inference', dataset_name, 'hmm',
                f'{model_name}.csv')
        ll = pd.read_csv(hmm_data_path)[['target', 'score_full']]
        ll['id'] = ll['target'].apply(lambda x: int(x.replace('id_', '')))
        ll = ll.join(id2seq, on='id', how='left')
        self.seq2score_dict = dict(zip(ll.seq, ll.score_full))

    def seq2score(self, seqs):
        scores = np.array([self.seq2score_dict.get(s, 0.0) for s in seqs])
        return scores

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)


class BLOSUM62HMMPredictor(HMMPredictor):
    def __init__(self, dataset_name, reg_coef=1e-8, **kwargs):
        super(BLOSUM62Predictor, self).__init__(dataset_name,
                model_name='blosum62', reg_coef=reg_coef, **kwargs)
