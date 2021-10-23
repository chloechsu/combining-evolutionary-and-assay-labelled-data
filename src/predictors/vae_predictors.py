import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from utils import seqs_to_onehot, get_wt_seq, read_fasta, seq2effect, mutant2seq
from predictors.base_predictors import BaseRegressionPredictor


class VaePredictor(BaseRegressionPredictor):
    "deepseq vae prediction."""

    def __init__(self, dataset_name, reg_coef=1e-8, **kwargs):
        super(VaePredictor, self).__init__(dataset_name, reg_coef=reg_coef, **kwargs)
        path = os.path.join('inference', dataset_name, 'vae', 'elbo.npy')
        if os.path.exists(path):
            delta_elbo = np.loadtxt(path)
            seqs_path = os.path.join('inference', dataset_name, 'vae', 'seqs.fasta')
            # if not os.path.exists(seqs_path):
            #     seqs_path = os.path.join('data', dataset_name, 'seqs.fasta')
            seqs = read_fasta(seqs_path)
            assert len(delta_elbo) == len(seqs), 'file length mismatch'
            self.seq2score_dict = dict(zip(seqs, delta_elbo))
        else:
            df = pd.read_csv(os.path.join('inference', dataset_name,
                'vae_predictions.csv'))
            df = df[np.isfinite(df.mutation_effect_prediction_vae_ensemble)]
            wtseqs, wtids = read_fasta(os.path.join('data', dataset_name,
                'wt.fasta'), return_ids=True)
            offset = int(wtids[0].split('/')[-1].split('-')[0])
            wt = wtseqs[0]
            seqs = [mutant2seq(m, wt, offset) for m in df.mutant.values]
            self.seq2score_dict = dict(zip(seqs,
                df.mutation_effect_prediction_vae_ensemble))

    def seq2score(self, seqs):
        scores = np.array([self.seq2score_dict.get(s, 0.0) for s in seqs])
        #return np.nan_to_num(scores)
        return scores

    def seq2feat(self, seqs):
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)
