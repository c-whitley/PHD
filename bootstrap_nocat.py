suffix = input('Suffix: ')

from tqdm.notebook import tqdm
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

import glob
import os
import sys
import pickle

from tqdm.notebook import tqdm
from datetime import datetime

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer, RobustScaler, MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.preprocessing import normalize, robust_scale, minmax_scale

from sklearn.preprocessing import OneHotEncoder,KBinsDiscretizer, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, GroupKFold, KFold, train_test_split, LeaveOneOut

from sklearn.metrics import make_scorer, confusion_matrix, roc_auc_score, roc_curve, plot_confusion_matrix, f1_score, recall_score, accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer

import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence

from patsy import dmatrices

import theano

from Preprocessing_Methods import *

total = pd.read_pickle('./Original_Data.pickle')
train_indices = pd.read_pickle('./train_indices.pickle')
#total = pd.read_hdf('/condor_data/sgcwhitl/Bayesian/Datasets/Original.hdf', key='Data')
total = truncate(total, start=1000, end=1800)

patient_ids = total.reset_index()['Patient_nu '].unique()

total['1yeardeath'] = (total.reset_index()['survival (months)']<12).values & (total.reset_index('DiedvsAlive')['DiedvsAlive']=='Died').values

total = total.set_index('1yeardeath', append=True)

y = '1yeardeath'
npca = 20


numeric_pipe = Pipeline([
("Normalise spectra", FunctionTransformer(minmax_scale, kw_args = {"axis": 1})),
("Scaler", RobustScaler()),
("PCA", PCA(npca)),
])

categorical_pipe = Pipeline([
    ("OneHot", OneHotEncoder())
])

#in_df = total.reset_index('ASMA').dropna(subset=['ASMA']).sample(20000)
in_df = total#.sample(20000)
in_df.columns = [str(col) for col in in_df.columns]

ct = make_column_transformer(
    (numeric_pipe,     make_column_selector(dtype_include=np.number)),
    (categorical_pipe, make_column_selector(dtype_include=object))
)

bootstrap_n = 0
total_data = []
traces = []

for i, row in train_indices.iloc[:1,:].iterrows():

    print(i)

    train_data = in_df.query(f"Patient_Number in {list(row['Train_pats'])}")
    test_data = in_df.query(f"Patient_Number in {list(row['Test_pats'])}")

    #columns = np.concatenate([[f'PCA{i}' for i in range(1,npca+1)], [f'ASMA:{t}' for t in train_data['ASMA'].unique()]])
    columns = [f'PCA{i}' for i in range(1,npca+1)]

    # Transform FTIR data to PCA components
    X_train = pd.DataFrame(ct.fit_transform(train_data), columns=columns, index=train_data.index)
    X_test = pd.DataFrame(ct.transform(test_data), columns=columns, index=test_data.index)

    Y_train = pd.DataFrame([1.0 if el==True else 0.0 for el in train_data.index.get_level_values(y)], index=train_data.index)
    Y_test  = pd.DataFrame([1.0 if el==True else 0.0 for el in test_data.index.get_level_values(y)], index=test_data.index)


    ################################ Bayesian ################################

    ncat = 3
    ncon = npca

    with pm.Model() as logistic_model:

        data_ = pm.Data('Pred', X_train.T)
        obs = pm.Data('Observed', Y_train.values.T)

        ɛ = pm.HalfNormal('ɛ', sd=100)
    
        # Continuous variables for each PC
        β1 = pm.Normal("β1", mu=0, sigma=10, shape = (ncon+1,1))

        # β.T + ɛ
        z = pm.math.dot(β1[1:].T, data_)

        # Probability of parameter P given the data
        p = pm.Deterministic('P', pm.math.sigmoid(z + (β1[0] + ɛ)))
        observed = pm.Bernoulli("p", p, observed=obs)

        #start=pm.find_MAP()

        callback = CheckParametersConvergence(diff='absolute')
        approx = pm.fit(n=50000, callbacks=[callback])

    with logistic_model:

        trace = approx.sample(10000)

        # update values of predictors:
        pm.set_data({"Pred": X_test.T})
        # use the updated values and predict outcomes and probabilities:
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["P"], samples=1000)


    model_preds = posterior_predictive["P"].squeeze()

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    results = {'y_test': Y_test,
               'BLR_Posterior': pd.DataFrame(model_preds.T, index=test_data.index),
               'LR_Preds': pd.DataFrame(lr.predict_proba(X_test), index=test_data.index)}

    total_data.append(results)
    traces.append(posterior_predictive)


with open(f'./results/New/{y}_2206_{suffix}.pickle', 'wb') as f:

    output_dict = {'Predictions': total_data,
                   'Trace': traces}

    pickle.dump(output_dict, f)#

print("Finished")