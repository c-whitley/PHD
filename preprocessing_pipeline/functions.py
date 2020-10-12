import numpy as np
import pandas as pd 

from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull

import multiprocessing as mp



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