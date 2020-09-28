import pandas as pd
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull
from scipy import interpolate

import multiprocessing as mp


class PCA_denoiser( BaseEstimator, TransformerMixin ):

    #Class Constructor 
    def __init__( self, n_components):

        self.n_components = n_components 

    
    #Return self nothing else to do here    
    def transform( self, X):

        pca_values = self.denoiser.fit(X)
        
        filtered = np.dot(self.denoiser.transform(X)[:,:self.n_components]
            , self.denoiser.components_[:self.n_components,:])

        values = np.add(filtered, np.mean(X, axis = 0).values.reshape(1,-1))

        return values

    
    #Method that describes what we need this transformer to do
    def fit( self, X, y = None ):

        self.denoiser = PCA(self.n_components)

        return self





class Rubber_Band( BaseEstimator, TransformerMixin ):
    """
    Applies a rubber band correction to the input matrix of spectra.
    Must be supplied as a shape (n_samples, n_wavenumbers)
    """

    def __init__(self, n_jobs = 4):

        self.n_jobs = n_jobs
   

    def transform(self, x):

        return self.x - self.baseline


    def rubberband_baseline(spectrum, wn):

        points = np.column_stack([wn, spectrum])

        verts = ConvexHull(points).vertices

        # Rotate convex hull vertices until they start from the lowest one
        verts = np.roll(verts, -verts.argmin())
        # Leave only the ascending part
        verts = verts[:verts.argmax()]

        baseline = np.interp(wn, wn[verts], spectrum[verts])

        return baseline


    def fit(self, x, y):

        if isinstance(y, pd.DataFrame):

            self.y = x.values
            self.wn = x.columns

        self.y = y

        pool = mp.Pool(processes=self.n_jobs)

        self.baseline = np.array([pool.apply(self.rubberband_baseline, args=(spectrum, self.wn)) 
        for spectrum in np.apply_along_axis(lambda row: row, axis = 0, arr=self.y)])

        return self


def truncate(df, transformer = None, start = 1000, end = 1800, paraffin = (1340, 1490)):

    wn = df.columns
    tab = df.copy()

    starti, endi = [np.argmin(np.abs(i - wn)) for i in [start, end]]

    # Only use wavenumbers within specified range
    tab = tab.iloc[:,starti:endi]

    if not paraffin == None:

        psi, pei = [np.argmin(np.abs(i - wn)) for i in [paraffin[0], paraffin[1]]]

        # Drop paraffin
        tab = tab.drop(wn[psi:pei].values.squeeze(), axis = 1)


    if transformer == None:

        return tab

    else:

        trans = transformer.fit_transform(tab.T).T
        tab.iloc[:, tab.columns] = trans

        return tab


def interpolate_DF(DF, x_new):
    """Interpolate inputted dataframes using new x values.

    Args:
        DF (DataFrame): Old Dataframe for interpolation
        x_new (Array): New X values required

    Returns:
        DataFrame: Dataframe with interpolated values
    """

    f=interpolate.interp1d(DF.columns, DF.values, fill_value="extrapolate")

    return pd.DataFrame(f(x_new),index=DF.index, columns=x_new)


def im_classification_plot(clf, image, thresh = 0.1):

    tab = image.reshape(-1, image.shape[-1])

    prob_predictions = clf.predict_proba(tab)
    prob_predictions_im = prob_predictions.reshape(image.shape[0], image.shape[1], -1)

    max_probs = np.expand_dims(prob_predictions_im.max(axis = -1),-1)

    preds=clf.predict(tab)
    unique_classes = clf.classes_

    if len(unique_classes) < 10:
        colours = mpl.cm.get_cmap("tab10").colors
    else:
        colours = mpl.cm.get_cmap("tab20").colors

    # What are each of the colours going to be?
    colour_mappings = dict(zip(unique_classes, colours))

    encoded_im = np.array([colour_mappings[class_name] for class_name in preds])
    preds_im=encoded_im.reshape(image.shape[0], image.shape[1], -1)

    rgba = np.concatenate([preds_im, max_probs], axis=-1)
    rgba = np.where(max_probs<thresh, [1,1,1,1], rgba)

    fig, axes = plt.subplots(2,3, figsize=(15,8))
    axes = axes.flatten()

    axes[0].set_title("FTIR Image")
    im=axes[0].imshow(image.sum(axis=2))
    divider=make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(im, cax)
    cbar.set_label("Absorbance")

    axes[1].set_title("Predictions")
    axes[1].imshow(rgba, interpolation='none')

    values = list(colour_mappings.values())

    patches = [mpatches.Patch(color=colour, label=label) for label, colour in colour_mappings.items()]
    axes[1].legend(handles=patches, bbox_to_anchor=(1, 1), loc=2)#3, borderaxespad=0. )

    for i, class_name in enumerate(unique_classes):

        im=axes[i+2].imshow(prob_predictions_im[:,:,i])
        axes[i+2].set_title(class_name)
        divider=make_axes_locatable(axes[i+2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(im, cax)
        cbar.set_label("Probability")

    return fig

    