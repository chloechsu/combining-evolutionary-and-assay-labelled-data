import logging
import numpy as np
import torch
from torch.nn.parameter import Parameter

_lim_val = 36.0
eps = np.finfo(np.float64).resolution


def logexp(t):
    return eps + torch.where(t > _lim_val, t,
            torch.log(torch.exp(torch.clamp(t, -_lim_val, _lim_val)) + 1.))


def inv_logexp(t):
    return np.where(t>_lim_val, t, np.log(np.exp(t + eps) - 1))


class Sech(torch.autograd.Function):
     """Implementation of sech(x) = 2 / (e^x + e^(-x))."""

     @staticmethod
     def forward(ctx, x):
         cosh = torch.cosh(x)
         sech = 1. / cosh
         sech = torch.where(torch.isinf(cosh), torch.zeros_like(sech), sech)
         ctx.save_for_backward(x, sech)
         return sech

     @staticmethod
     def backward(ctx, grad_output):
         x, sech = ctx.saved_tensors
         return -sech * torch.tanh(x) * grad_output


class TanhSingleWarpingTerm(torch.nn.Module):
    """A tanh mapping with scaling and translation.

    Maps y to a * tanh(b * (y + c)), where a, b, c are positive scalars.
    The parameters are pre_a, pre_b, and c, initialized uniformly in [-1, 1].
    """

    def __init__(self):
        super(TanhSingleWarpingTerm, self).__init__()
        self.pre_a = Parameter(torch.Tensor(1))
        self.pre_b = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        # Initialize according to warpedLMM code.
        torch.nn.init.normal_(self.pre_a)
        torch.nn.init.normal_(self.pre_b)
        with torch.no_grad():
            self.pre_a.abs_()
            self.pre_b.abs_()
        torch.nn.init.uniform_(self.c, -0.5, 0.5)

    def set_parameters(self, a, b, c):
        self.pre_a.data.fill_(inv_logexp(a).item())
        self.pre_b.data.fill_(inv_logexp(b).item())
        self.c.data.fill_(c)

    def get_parameters(self):
        return (logexp(self.pre_a).detach().item(),
            logexp(self.pre_b).detach().item(), self.c.detach().item())

    def forward(self, y):
        a = logexp(self.pre_a)
        b = logexp(self.pre_b)
        return a * torch.tanh(b * (y + self.c))

    def jacobian(self, y):
        """Returns df/dy evaluated at the y.

        df/dy = a * b * sech^2 (b * (y + c)).
        """
        a = logexp(self.pre_a)
        b = logexp(self.pre_b)
        sech = Sech.apply
        return a * b * (sech(b * (y + self.c)) ** 2)



class TanhWarpingLayer(torch.nn.Module):
    """
    A warping layer combining linear and tanh mappings.

    Maps y to d * y + a_1 * tanh(b_1 * (y + c_1)) + ... + a_n * tanh(
      b_n * (y + c_n)) where all d, a_i, b_i are positive scalars.
    """
    def __init__(self, num_warping_terms):
        super(TanhWarpingLayer, self).__init__()
        self.num_warping_terms = num_warping_terms
        warping_terms = []
        for i in range(num_warping_terms):
            warping_terms.append(TanhSingleWarpingTerm())
        self.warping_terms = torch.nn.ModuleList(warping_terms)
        self.pre_d = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.pre_d)
        with torch.no_grad():
            self.pre_d.abs_()
        for t in self.warping_terms:
            t.reset_parameters()

    def set_parameters(self, a, b, c, d):
        """Sets parameters of the warping layer.

        Args:
            a, b, c are arrays of length num_warping_terms, d is a scalar.
        """
        if len(a) != self.num_warping_terms or len(b) != self.num_warping_terms or (
                len(c) != self.num_warping_terms):
            raise ValueError("Expected %d warping terms", self.num_warping_terms)
        self.pre_d.data.fill_(inv_logexp(d).item())
        for i, t in enumerate(self.warping_terms):
            t.set_parameters(a[i], b[i], c[i])

    def get_parameters(self):
        """Returns parameters of the warping layer.
        
        Returns:
            Warping parameters a,b,c,d.
        """
        d = logexp(self.pre_d).detach().item()
        a = np.zeros(self.num_warping_terms)
        b = np.zeros(self.num_warping_terms)
        c = np.zeros(self.num_warping_terms)
        for i, t in enumerate(self.warping_terms):
            a[i], b[i], c[i] = t.get_parameters()
        return a, b, c, d

    def write_parameters(self):
        """Writes parameters to logging.debug."""
        a, b, c, d = self.get_parameters()
        logging.debug(f"a: {a}")
        logging.debug(f"b: {b}")
        logging.debug(f"c: {c}")
        logging.debug(f"d: {d}")

    def forward(self, y):
        s = logexp(self.pre_d) * y
        for warping_term in self.warping_terms:
            s += warping_term.forward(y)
        return s

    def extra_repr(self):
        return "num_warping_terms={}".format(self.num_warping_terms)

    def jacobian(self, y):
        """Returns df/dy evaluated at the y."""
        jcb = logexp(self.pre_d) * torch.ones_like(y)
        for warping_term in self.warping_terms:
            jcb += warping_term.jacobian(y)
        return jcb

    def numpy_fn(self):
        a, b, c, d = self.get_parameters()
        def fn(x):
            s = d * x
            for i in range(self.num_warping_terms):
                s += a[i] * np.tanh(b[i] * (x + c[i]))
            return s
        return fn

