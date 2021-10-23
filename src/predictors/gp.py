import math
import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter
from skopt.learning.gaussian_process.kernels import Matern

from model import _delta_hamiltonian


class EVMaternKernel(Matern):
    """Representation: int vector of sequence len."""
    
    def __init__(self, couplings_model, **kwargs):
        super().__init__(**kwargs)
        self.couplings_model = couplings_model
        self.J = couplings_model.J_ij
        self.h = couplings_model.h_i
    
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter(
            "length_scale", "numeric", "fixed")
        
    def seq2rep(self, seqs):
        reps = np.zeros((len(seqs), len(seqs[0])), dtype=int)
        for i, s in enumerate(seqs):
            reps[i] = [self.couplings_model.alphabet_map[x] for x in s]
        return reps
        
    def dist(self, X, Y):
        dists = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                pos = []
                subs = []
                for l in range(X.shape[1]):
                    if X[i, l] != Y[j, l]:
                        pos.append(l)
                        subs.append(Y[j, l])
                        dists[i, j] += np.abs(
                                self.h[l, X[i, l]] - self.h[l, Y[j, l]])
        return dists

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            dists = self.dist(X, X) 
        else:
            dists = self.dist(X, Y)
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
        dists /= self.length_scale ** 2

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-dists ** 2 / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient
            else:
                raise NotImplementedError
        else:
            return K
        
    def gradient_x(self, x, X_train):
        raise NotImplementedError
