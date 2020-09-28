import numpy as np
import pandas as pd 

from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull

import multiprocessing as mp

from sklearn.base import BaseEstimator, TransformerMixin


class Rubber_Band( BaseEstimator, TransformerMixin ):
    """
    Applies a rubber band correction to the input matrix of spectra.
    Must be supplied as a shape (n_samples, n_wavenumbers)
    """

    def __init__(self, n_jobs = 4):

        self.n_jobs = n_jobs
   

    def transform(self, x):

        return self.x - self.baseline


    def rubberband_baseline(spectrum):

        wn = np.arange(len(spectrum))
        points = np.column_stack([wn, spectrum])

        verts = ConvexHull(points).vertices

        # Rotate convex hull vertices until they start from the lowest one
        verts = np.roll(verts, -verts.argmin())
        # Leave only the ascending part
        verts = verts[:verts.argmax()]

        baseline = np.interp(wn, wn[verts], spectrum[verts])

        return baseline


    def fit(self, y):

        self.y = y

        pool = mp.Pool(processes=self.n_jobs)

        self.baseline = np.array([pool.apply(self.rubberband_baseline, args=(spectrum)) 
        for spectrum in np.apply_along_axis(lambda row: row, axis = 0, arr=self.y)])

        return self