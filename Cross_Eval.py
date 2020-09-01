import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from mlxtend.evaluate import scoring

from itertools import product



def get_weights(column):
    """[summary]

    Args:
        column ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    counts = column.value_counts()
    weight_dict = dict(zip(counts.index, [1/(counts.values.sum()/n) for n in counts.values]))
    
    return column.map(weight_dict).values


def bayes_search(clf, param_dict, X, y, cv, **kwargs):
    """ Perform Bayesian hyperparameters optimisation for a given classifier.

    Args:
        clf (Sklearn Classifier): [description]
        param_dict ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]
        cv ([type]): [description]

    Returns:
        BayesSearchCV Object: [description]
    """
    
    fit_params = {
        "clf__sample_weight": kwargs.get("sample_weight", None),
    }

    opt = BayesSearchCV(
        clf
        , param_dict
        , verbose = kwargs.get("verbose", 1)
        , n_iter=3
        , cv=cv
        , return_train_score = True
        #, fit_params = fit_params
        , refit = True
    )
    #opt.set_params(**fit_params)
    
    opt.fit(X, y, groups = kwargs.get("groups", None))
    
    return opt


def plotCM(y_true, y_pred, display_labels, include_values=True, cmap='viridis',
         xticks_rotation='horizontal', values_format=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)
        if values_format is None:
            values_format = '.2g'

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text_[i, j] = ax.text(j, i,
                                       format(cm[i, j], values_format),
                                       ha="center", va="center",
                                       color=color)

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    
    return fig


def score(clf, X_valid, y_true, opt=True):

    if roc_auc_score(y_true, clf.predict_proba(X_valid)[:,0]) > 0.5:

        y_prob = clf.predict_proba(X_valid)[:,0]

    else:

        y_prob = clf.predict_proba(X_valid)[:,1]

    return score_func(y_true, y_prob, opt=opt)


def score_func(y_true, y_prob, opt=True):

    scores = dict()

    scores["FPR_curve"], scores["TPR_curve"], scores["Thresholds"] = roc_curve(y_true, y_prob)
    scores["AUC"] = roc_auc_score(y_true, y_prob, average = "weighted")


    _scores = np.array([np.sqrt(((1-t)**2)+((0-(f))**2)) for f, t in zip(scores["FPR_curve"], scores["TPR_curve"])])
    threshi = np.argmin(_scores)

    scores["opt"] = scores["Thresholds"][threshi]


    if opt:
        scores["y_pred"] = np.array([0 if score < scores["opt"] else 1 for score in y_prob])
    else:
        scores["y_pred"] = np.array([0 if score < 0.5 else 1 for score in y_prob])

    scores["y_true"] = y_true
    scores["y_prob"] = y_prob
    scores["prec_curve"], scores["rec_curve"], _ = precision_recall_curve(y_true.squeeze(), y_prob)

    scores["Precision"] = precision_score(y_true, scores["y_pred"])
    scores["Recall"] = recall_score(scores["y_true"], scores["y_pred"])
    scores["F1"] = f1_score(scores["y_true"], scores["y_pred"])

    scores["Specificity"] = scoring(scores["y_true"], scores["y_pred"], metric = 'specificity')
    scores["Sensitivity"] = scoring(scores["y_true"], scores["y_pred"], metric = 'sensitivity')

    return scores


def patient_score(clf, X_valid, y_true, patient_column="Patient_Number"):

    scores = dict()

    def agg_scores(columns):
    
        return columns.sum()/columns.size

    #print(y_prob)

    y_pred = clf.predict_proba(X_valid)[:,0]

    if roc_auc_score(y_true, clf.predict_proba(X_valid)[:,0]) > 0.5:

        y_prob = clf.predict_proba(X_valid)[:,0]

    else:
        y_prob = clf.predict_proba(X_valid)[:,1]

    #y_prob = clf.predict_proba(X_valid)[:,1]

    patient_preds = pd.DataFrame({"predictions": y_prob}, index = X_valid.index)
    patient_preds["true"] = y_true.values

    y_true = patient_preds.groupby(patient_column).agg(agg_scores).loc[:, "true"]
    y_pred = patient_preds.groupby(patient_column).agg(agg_scores).loc[:,"predictions"]

    return score_func(y_true, y_pred)
