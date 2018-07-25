import math
from math import gamma
from operator import mul
from functools import reduce
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from numpy.linalg import norm
# Numpy imports
import numpy as np
from numpy import (array, asanyarray, ones, arange, log, diag, vstack, exp,
        asarray, ndarray, zeros, isscalar)
from scipy.special import (psi, polygamma, gammaln)

euler = -1*psi(1)

import sys
try:
    # python 2
    MAXINT = sys.maxint
except AttributeError:
    # python 3
    MAXINT = sys.maxsize

class Dirichlet(object): 
    
    def __init__(self, alpha):
        self.corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
        # Mid-points of triangle sides opposite of each corner
        self.midpoints = [(self.corners[(i + 1) % 3] + self.corners[(i + 2) % 3]) / 2.0 for i in range(3)]
        self.triangle = tri.Triangulation(self.corners[:, 0], self.corners[:, 1])
        self._alpha = np.array(alpha)
        #out = "New Dirichlet Distribution created with alpha = " + str(self._alpha)
        #print(out)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def getAlpha(self):
        return self.alpha
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])
    def draw_pdf_contours(self, nlevels=200, subdiv=8, **kwargs):
        vertexlabels = ('1','2','3')
        refiner = tri.UniformTriRefiner(self.triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)
        pvals = [self.pdf(self.xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
        plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, 0.75**0.5)
        #Print numbers for corners
        axes = plt.gca()
        axes.text(-0.05, -0.05, vertexlabels[0])
        axes.text(1.05, -0.05, vertexlabels[1])
        axes.text(0.5, np.sqrt(3) / 2 + 0.05, vertexlabels[2])
        plt.axis('off')
    def xy2bc(self, xy, tol=1.e-3):
        '''Converts 2D Cartesian coordinates to barycentric.'''
        s = [(self.corners[i] - self.midpoints[i]).dot(xy - self.midpoints[i]) / 0.75 \
             for i in range(3)]
        return np.clip(s, tol, 1.0 - tol)
    
    def mle(self, D, tol=1e-7, maxiter=None):
        '''Mean and precision alternating method for MLE of Dirichlet
        distribution'''
        N, K = D.shape
        logp = log(D).mean(axis=0)
        a0 = self._init_a(D)
        s0 = a0.sum()
        if s0 < 0:
            a0 = a0/s0
            s0 = 1
        elif s0 == 0:
            a0 = ones(a.shape) / len(a)
            s0 = 1
        m0 = a0/s0

        # Start updating
        if maxiter is None:
            maxiter = MAXINT
        for i in range(maxiter):
            a1 = self._fit_s(D, a0, logp, tol=tol)
            s1 = sum(a1)
            a1 = self._fit_m(D, a1, logp, tol=tol)
            m = a1/s1
            # if norm(a1-a0) < tol:
            if abs(self.loglikelihood(D, a1)-self.loglikelihood(D, a0)) < tol: # much faster
                return a1
            a0 = a1
        raise Exception('Failed to converge after {} iterations, values are {}.'
                        .format(maxiter, a1))
    def loglikelihood(self, D, a):
        '''Compute log likelihood of Dirichlet distribution, i.e. log p(D|a).
        Parameters
        ----------
        D : 2D array
            where ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.
        a : array
            Parameters for the Dirichlet distribution.
        Returns
        -------
        logl : float
            The log likelihood of the Dirichlet distribution'''
        N, K = D.shape
        logp = log(D).mean(axis=0)
        return N*(gammaln(a.sum()) - gammaln(a).sum() + ((a - 1)*logp).sum())    
    # Helper functions for mle
    def _trigamma(self, x):
        return polygamma(1, x)
    def _init_a(self, D):
        '''Initial guess for Dirichlet alpha parameters given data D'''
        E = D.mean(axis=0)
        E2 = (D**2).mean(axis=0)
        return ((E[0] - E2[0])/(E2[0]-E[0]**2)) * E

        raise Exception('Failed to converge after {} iterations, s is {}'
                .format(maxiter, s1))
    def _fit_s(self, D, a0, logp, tol=1e-7, maxiter=1000):
        '''Assuming a fixed mean for Dirichlet distribution, maximize likelihood
        for preicision a.k.a. s'''
        N, K = D.shape
        s1 = a0.sum()
        m = a0 / s1
        mlogp = (m*logp).sum()
        for i in range(maxiter):
            s0 = s1
            g = psi(s1) - (m*psi(s1*m)).sum() + mlogp
            h = self._trigamma(s1) - ((m**2)*self._trigamma(s1*m)).sum()

            if g + s1 * h < 0:
                s1 = 1/(1/s0 + g/h/(s0**2))
            if s1 <= 0:
                s1 = s0 * exp(-g/(s0*h + g)) # Newton on log s
            if s1 <= 0:
                s1 = 1/(1/s0 + g/((s0**2)*h + 2*s0*g)) # Newton on 1/s
            if s1 <= 0:
                s1 = s0 - g/h # Newton
            if s1 <= 0:
                raise Exception('Unable to update s from {}'.format(s0))

            a = s1 * m
            if abs(s1 - s0) < tol:
                return a
        
    def _fit_m(self, D, a0, logp, tol=1e-7, maxiter=1000):
        '''With fixed precision s, maximize mean m'''
        N,K = D.shape
        s = a0.sum()

        for i in range(maxiter):
            m = a0 / s
            a1 = self._ipsi(logp + (m*(psi(a0) - logp)).sum())
            a1 = a1/a1.sum() * s

            if norm(a1 - a0) < tol:
                return a1
            a0 = a1
        raise Exception('Failed to converge after {} iterations, s is {}'
                .format(maxiter, s))
    def _ipsi(self, y, tol=1.48e-9, maxiter=10):
        '''Inverse of psi (digamma) using Newton's method. For the purposes
        of Dirichlet MLE, since the parameters a[i] must always
        satisfy a > 0, we define ipsi :: R -> (0,inf).'''
        y = asanyarray(y, dtype='float')
        x0 = self._piecewise(y, [y >= -2.22, y < -2.22],
                [(lambda x: exp(x) + 0.5), (lambda x: -1/(x+euler))])
        for i in range(maxiter):
            x1 = x0 - (psi(x0) - y)/self._trigamma(x0)
            if norm(x1 - x0) < tol:
                return x1
            x0 = x1
        raise Exception(
            'Unable to converge in {} iterations, value is {}'.format(maxiter, x1))
    def _trigamma(self, x):
        return polygamma(1, x)
    def _piecewise(self, x, condlist, funclist, *args, **kw):
        '''Fixed version of numpy.piecewise for 0-d arrays'''
        x = asanyarray(x)
        n2 = len(funclist)
        if isscalar(condlist) or \
                (isinstance(condlist, np.ndarray) and condlist.ndim == 0) or \
                (x.ndim > 0 and condlist[0].ndim == 0):
            condlist = [condlist]
        condlist = [asarray(c, dtype=bool) for c in condlist]
        n = len(condlist)

        zerod = False
        # This is a hack to work around problems with NumPy's
        #  handling of 0-d arrays and boolean indexing with
        #  numpy.bool_ scalars
        if x.ndim == 0:
            x = x[None]
            zerod = True
            newcondlist = []
            for k in range(n):
                if condlist[k].ndim == 0:
                    condition = condlist[k][None]
                else:
                    condition = condlist[k]
                newcondlist.append(condition)
            condlist = newcondlist

        if n == n2-1:  # compute the "otherwise" condition.
            totlist = condlist[0]
            for k in range(1, n):
                totlist |= condlist[k]
            condlist.append(~totlist)
            n += 1
        if (n != n2):
            raise ValueError(
                    "function list and condition list must be the same")

        y = zeros(x.shape, x.dtype)
        for k in range(n):
            item = funclist[k]
            if not callable(item):
                y[condlist[k]] = item
            else:
                vals = x[condlist[k]]
                if vals.size > 0:
                    y[condlist[k]] = item(vals, *args, **kw)
        if zerod:
            y = y.squeeze()
        return y




