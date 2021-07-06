suffix = input('Suffix = ')

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
# total = pd.read_hdf('/mnt/c/Users/conor/Git_Projects/PHD/Tumour_df_raw_0804.hdf5', key='Data')

train_indices = pd.read_pickle('./train_indices_10.pickle')
#total = pd.read_hdf('/condor_data/sgcwhitl/Bayesian/Datasets/Original.hdf', key='Data')
total = truncate(total, start=1000, end=1800)

patient_ids = total.reset_index()['Patient_nu '].unique()

total['1yeardeath'] = (total.reset_index()['survival (months)']<12).values & (total.reset_index('DiedvsAlive')['DiedvsAlive']=='Died').values
total = total.set_index('1yeardeath', append=True)

y = '2year'
npca = 5

numeric_pipe = Pipeline([
#("Normalise spectra", FunctionTransformer(robust_scale, kw_args = {"axis": 1})),
#("Normalise spectra", FunctionTransformer(normalize, kw_args = {"axis": 1})),
("Normalise spectra", FunctionTransformer(normalize, kw_args = {"axis": 1})),
#("Scaler", MinMaxScaler()),
("PCA", PCA(npca)),
])

categorical_pipe = Pipeline([
    ("OneHot", OneHotEncoder())
])

in_df = total.reset_index('ASMA').dropna(subset=['ASMA']).sample(10000)
in_df.columns = [str(col) for col in in_df.columns]

ct = make_column_transformer(
    (numeric_pipe,     make_column_selector(dtype_include=np.number)),
    (categorical_pipe, make_column_selector(dtype_include=object))
)

#patient_ids = total.reset_index()['Patient_nu '].unique()

bootstrap_n = 0
total_data = []
traces = []

for i, row in train_indices.sample(frac=1).iloc[:100,:].iterrows():

    train_data = in_df.query(f"Patient_Number in {list(row['Train_pats'])}")
    test_data = in_df.query(f"Patient_Number in {list(row['Test_pats'])}")

    columns = np.concatenate([[f'PCA{i}' for i in range(1,npca+1)], [f'ASMA:{t}' for t in train_data['ASMA'].unique()]])

    # Transform FTIR data to PCA components
    X_train = pd.DataFrame(ct.fit_transform(train_data), columns=columns, index=train_data.index)
    X_test = pd.DataFrame(ct.transform(test_data), columns=columns, index=test_data.index)

    Y_train = train_data.index.get_level_values(y).astype(np.int)
    Y_test  = test_data.index.get_level_values(y).astype(np.int)
    #Y_test  = pd.DataFrame([1 if el==True else 0 for el in test_data.index.get_level_values(y)], index=test_data.index)

    ################################ Bayesian ################################

    ncat = 3
    ncon = X_train.shape[1]-ncat

    with pm.Model() as logistic_model:

        data_ = pm.Data('X', X_train.T)
        obs = pm.Data('Observed', Y_train.values.T)

        ɛ = pm.HalfNormal('ɛ', sd=1,  shape = (ncat+ncon+1,1))
        #ɛ = pm.HalfNormal('ɛ', sd=1)

        #ɛ = pm.HalfStudentT('ɛ', nu=1, sigma=1,  shape = (ncat+ncon+1,1))
    
        # Continuous variables for each PC
        β1 = pm.Normal("β1", mu=0, sigma=1, shape = (ncon+1,1))

        # Categorical variables 
        β2 = pm.Categorical("β2", [1/ncat for _ in range(1, ncat+1)], shape=(ncat,1))

        # β.T + ɛ
        z = pm.math.dot(pm.math.concatenate([(β1[1:]+ɛ[1:-ncat]), (β2 + ɛ[-ncat:])]).T, data_)
        #z = pm.math.dot(pm.math.concatenate([β1[1:], β2]).T, data_)

        # Probability of parameter P given the data
        p = pm.Deterministic('P', pm.math.sigmoid(z + (β1[0] + ɛ[0])))
        observed = pm.Bernoulli("p", p, observed=obs)

        start=pm.find_MAP()

        #trace = pm.sample(10000, tune=100, start=start, step=step)
        trace = pm.sample(1000, tune=2000, start=start, init="adapt_diag", cores=1, chains=1)


    with logistic_model:
        # update values of predictors:
        pm.set_data({"X": X_test.T})
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

with open(f'./results/New/{y}_2306_{suffix}.pickle', 'wb') as f:
    pickle.dump(total_data, f)

"""
    model_str = 'y~'+'+'.join(X_train.columns)
    
    bootstrap_n += 1

    with pm.Model() as logistic_model:

        print('A')
        data = pm.Data('Data', X_train.values)
        print('B')
        pm.glm.GLM.from_formula(model_str
                               ,data=data
                               ,family=pm.glm.families.Binomial())

        print('C')
        #trace = pm.sample(10000, tune=100, start=start, step=step)
        trace = pm.sample(100, tune=100, init="adapt_diag", cores=1)
        print('D')
    break

    with logistic_model:

        # update values of predictors:
        pm.set_data({"data": X_test.T})
        # use the updated values and predict outcomes and probabilities:
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["P"], samples=1000)

    model_preds = posterior_predictive["P"].squeeze()

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)

    results = {'y_test': Y_test,
               'BLR_Posterior': pd.DataFrame(model_preds.T, index=test_data.index),
               'LR_Preds': pd.DataFrame(lr.predict_proba(X_test), index=test_data.index)}

    total_data.append(results)
with open(f'./{y}_0306_{suffix}.pickle', 'wb') as f:
    #pickle.dump(total_data, f)
    pass

"""