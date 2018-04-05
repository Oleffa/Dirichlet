import math
from math import gamma
from operator import mul
import numpy as np
from functools import reduce
import matplotlib.tri as tri
import matplotlib.pyplot as plt

class Dirichlet(object): 
    
    def __init__(self, alpha):
        self.corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
        # Mid-points of triangle sides opposite of each corner
        self.midpoints = [(self.corners[(i + 1) % 3] + self.corners[(i + 2) % 3]) / 2.0 for i in range(3)]
        self.triangle = tri.Triangulation(self.corners[:, 0], self.corners[:, 1])
        self._alpha = np.array(alpha)
        out = "New Dirichlet Distribution created with alpha = " + str(self._alpha)
        print(out)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
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
