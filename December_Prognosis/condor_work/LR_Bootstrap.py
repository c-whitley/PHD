from tqdm.notebook import tqdm
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import glob
import os
import sys
import pickle

from tqdm.notebook import tqdm
from datetime import datetime

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import normalize, robust_scale, minmax_scale

from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer

import pymc3 as pm

print('Test')

total = pd.read_hdf('/condor_data/sgcwhitl/Bayesian/Datasets/Original.hdf', key='Data')
total = total.iloc[:,(995<total.columns) & (total.columns<1805)]

patient_ids = total.reset_index()['Patient_nu '].unique()

#total['1yeardeath'] = ((total.reset_index()['survival (months)']<12) & (total.reset_index()['Alive']==False)).values
#total = total.set_index('1yeardeath', append=True)

y = '1yeardeath'
suffix = 'Original_100_0.2T_m1_Bern2'


numeric_pipe = Pipeline([
#("Normalise spectra", FunctionTransformer(robust_scale, kw_args = {"axis": 1})),
#("Normalise spectra", FunctionTransformer(normalize, kw_args = {"axis": 1})),
("Normalise spectra", FunctionTransformer(minmax_scale, kw_args = {"axis": 1})),
#("Scaler", RobustScaler()),
("PCA", PCA(0.99)),
])

categorical_pipe = Pipeline([
    ("OneHot", OneHotEncoder())
])

in_df = total.reset_index(['ASMA']).dropna().sample(2000)
in_df.columns = [str(col) for col in in_df.columns]

ct = make_column_transformer(
    (numeric_pipe,     make_column_selector(dtype_include=np.number)),
    (categorical_pipe, make_column_selector(dtype_include=object))
)

patient_ids = total.reset_index()['Patient_nu '].unique()

bootstrap_n = 0
total_data = []

while bootstrap_n < 10:

    print(bootstrap_n)

    split_check = False

    while split_check == False:

        train_pat, test_pat = train_test_split(patient_ids, test_size=0.2)

        train_data = in_df.query(f"Patient_Number in {list(train_pat)}")
        test_data = in_df.query(f"Patient_Number in {list(test_pat)}")

        # Unique predictor variables in set
        trainn_un = train_data.reset_index(y).iloc[:,0].unique()
        testn_un = test_data.reset_index(y).iloc[:,0].unique()

        if len(trainn_un) == len(testn_un) == 2:

            split_check = True
    try:
        X_train = ct.fit_transform(train_data)
        X_test = ct.transform(test_data)

    except ValueError as e:

        continue

    Y_train = pd.DataFrame([1 if el==True else 0 for el in train_data.index.get_level_values(y)], index=train_data.index)
    Y_test  = pd.DataFrame([1 if el==True else 0 for el in test_data.index.get_level_values(y)], index=test_data.index)

    bootstrap_n += 1

    ################################ Bayesian ################################

    ncat = 3
    ncon = X_train.shape[1]-ncat

    with pm.Model() as logistic_model:

        data_ = pm.Data('Pred', X_train.T)
        obs = pm.Data('Observed', np.array(Y_train.values == True).T)

        ɛ = pm.HalfNormal('ɛ', sd=1,  shape = (ncat+ncon+1,1))
        #ɛ = pm.HalfStudentT('ɛ', nu=1, sigma=1,  shape = (ncat+ncon+1,1))
    
        # Continuous variables for each PC
        β1 = pm.Normal("β1", mu=1, sigma=1, shape = (ncon+1,1))

        # Categorical variables 
        β2 = pm.Categorical("β2", [1/ncat for _ in range(1, ncat+1)], shape=(ncat,1))

        # β.T + ɛ
        z = pm.math.dot(pm.math.concatenate([(β1[1:]+ɛ[1:-ncat]), (β2 + ɛ[-ncat:])]).T, data_)

        # Probability of parameter P given the data
        p = pm.Deterministic('P', pm.math.sigmoid(z + (β1[0] + ɛ[0])))
        observed = pm.Bernoulli("p", p, observed=obs)

        start=pm.find_MAP()

        #trace = pm.sample(10000, tune=100, start=start, step=step)
        trace = pm.sample(1000, tune=1000, start=start, init="adapt_diag", cores=1)


    with logistic_model:
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

with open(f'./results/New/{y}_1706_{suffix}.pickle', 'wb') as f:
    pickle.dump(total_data, f)