import os
import string
import copy

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np 
import pandas as pd 
import seaborn as sns

from scipy.stats import sem

from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, confusion_matrix

metadata = pd.read_excel('/mnt/c/Users/conor/Git_Projects/PHD/metadata_0206.xlsx')

def km_calculate(df, duration_col, pred_col, censored_col, thresh=0.5, patient=False):

    kmf = KaplanMeierFitter()

    #fig, ax = plt.subplots()

    kts = []
    kfs = []
    ps = []

    for index, row in df.groupby(level=0):

        duration = row[duration_col]
        pred = (row[pred_col] > thresh)
        death_obs = (row[censored_col]=='Died')

        stat, p = sm.duration.survdiff(duration, death_obs, pred)

        ps.append((stat,p))

        try:
            kmf.fit(duration.loc[pred].values, death_obs.loc[pred].values, label=True)
            kts.append(kmf.survival_function_)
            kmf.fit(duration.loc[~pred].values, death_obs.loc[~pred].values, label=False)
            kfs.append(kmf.survival_function_)

        except ValueError as e:

            print(index, e)
            continue

    indxs = [df.index for df in kts]

    indxmin, indxmax = np.min(np.concatenate(indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    tsi = np.array([np.interp(new_x, df.index, df.values.squeeze()) for df in kts])

    indxs = [df.index for df in kfs]

    indxmin, indxmax = np.min(np.concatenate(indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    fsi = np.array([np.interp(new_x, df.index, df.values.squeeze()) for df in kfs])

    return tsi, fsi, ps


def km_calculateB(df, duration_col, pred_col, censored_col, thresh=0.5):

    kmf = KaplanMeierFitter()

    #fig, ax = plt.subplots()

    kts = []
    kfs = []
    ps = []

    for index, row in df.groupby(level=0):
    #for index, row in results.iterrows():

        #row.index = pd.MultiIndex.from_frame(row.index.to_frame().reset_index(drop=True).merge(metadata, on = 'REF'))

        duration = row[pred_col].reset_index(duration_col)[duration_col]
        pred = (row[pred_col].mean(axis=1) > thresh)

        death_obs = (row[pred_col].reset_index(censored_col)[censored_col] == 'Died')

        stat, p = sm.duration.survdiff(duration, death_obs, pred)

        ps.append((stat,p))

        try:
            kmf.fit(duration.loc[pred,:], death_obs.loc[pred,:], label=True)
            kts.append(kmf.survival_function_)
            kmf.fit(duration.loc[~pred,:], death_obs.loc[~pred,:], label=False)
            kfs.append(kmf.survival_function_)

        except ValueError as e:

            print(index, e)
            continue

    indxs = [df.index for df in kts]

    indxmin, indxmax = np.min(np.concatenate(indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    tsi = np.array([np.interp(new_x, df.index, df.values.squeeze()) for df in kts])

    indxs = [df.index for df in kfs]

    indxmin, indxmax = np.min(np.concatenate(indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    fsi = np.array([np.interp(new_x, df.index, df.values.squeeze()) for df in kfs])

    return tsi, fsi


def km_plots(tsi, fsi, axin=None, ste=False, median=False):

    if axin:
        ax=axin
    else:
        fig, ax = plt.subplots()

    for name, res, c in zip(['1yeardeath predicted true', '1yeardeath predicted false'], [tsi, fsi], colors.TABLEAU_COLORS.values()):

        if median:
            m = np.median(res, axis=0)
        else:
            m = np.mean(res, axis=0)

        if ste:
            error = sem(res, axis=0)

        else:
            error = np.std(res,axis=0)


        ax.plot(m, ds='steps', c=c, label=name)
        ax.plot(m+error, ds='steps', ls='--', c=c, alpha=0.5)
        ax.plot(m-error, ds='steps', ls='--', c=c, alpha=0.5)

        ax.set_xlabel('Time (months)')
        ax.set_ylabel("Cumulative Survival")
        ax.set_ylim(0,1)

    return ax


def stats_plot(df, fig, axes=None):

    if(axes is None):
        
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(8,4), sharey=True)
        fig.subplots_adjust(wspace=0.1)

    for col_name, ax, lab in zip(df, axes.flatten(), ['A','B','C','D','E','F']):

        ax.hist(df[col_name], edgecolor='black')
        ax.text(0.05, 0.8, lab, size='x-large', transform=ax.transAxes)
        if lab in ['A', 'D']:
            ax.set_ylabel('Frequency')

        if lab in ['D', 'E', 'F']:
            ax.set_xlabel('Score')

    print(dict(zip(df.columns, ['A','B','C','D','E','F'])))

    return fig


def box_plot(stats):

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10,6))
    fig.subplots_adjust(wspace=0.2)

    for ax, name, let in zip(axes.flatten(), stats.columns, string.ascii_uppercase):
        ax = sns.boxplot(data=stats.reset_index()
        , x='Vars'
        , y=name
        , ax=ax
        , whis=1)

        ax.set_xticklabels(['ASMA', 'ASMA+FTIR','FTIR'])
        ax.set_xlabel(None)

        if let in ['A', 'D']:
            ax.set_ylabel('Score')
        else:
            ax.set_ylabel(None)


        if name == 'MCC':
            ax.set_ylim(-1.1,1.1)
        
        else:
            ax.set_ylim(-0.1,1.1)

        ax.text(-0.05, 1.05, let, size='large', transform=ax.transAxes)

    print(list(zip(stats.columns, string.ascii_uppercase)))

    return fig


def calc_stats(datasets, weight=True, patient=False, best_threshold=0.5):


    output = {}

    for colour, (name, dataset) in zip(['tab:blue', 'tab:orange', 'tab:green'], datasets.items()):

        stats = {'AUC':[]
        ,'F1':[]
        ,'MCC':[]
        ,'Specificity':[]
        ,'Sensitivity':[]
        ,'PPV':[]
        ,'NPV':[]
        ,'Thresh':[]}

        for split, split_df in dataset.groupby(level=0):

            if patient:

                split_df = split_df.groupby('Patient_Number').median()

            ts = np.linspace(0.2,0.9,10)
            #f1 = pd.Series({t: f1_score(y_true=split_df['Y_true'], y_pred=split_df['Preds']>t) for t in ts})
            #best_threshold = f1.idxmax()
            best_threshold = 0.5

            preds = np.array([1 if el > best_threshold else 0 for el in split_df['Preds']])

            if weight:
                cm = confusion_matrix(split_df['Y_true'], preds, sample_weight=split_df['Weights'])
                stats['AUC'].append(roc_auc_score(split_df['Y_true'], split_df['Preds'], sample_weight=split_df['Weights']))
                stats['F1'].append(f1_score(y_true=split_df['Y_true'], y_pred=preds, sample_weight=split_df['Weights']))

            else:
                cm = confusion_matrix(split_df['Y_true'], preds)
                stats['AUC'].append(roc_auc_score(split_df['Y_true'], split_df['Preds']))
                stats['F1'].append(f1_score(y_true=split_df['Y_true'], y_pred=preds))

            #cm = confusion_matrix(split_df['Y_true'], preds)
            tn, fp, fn, tp = cm.flatten()

            stats['MCC'].append(((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            stats['Specificity'].append(tn/(tn+fp))
            stats['Sensitivity'].append(tp/(tp+fn))
            stats['PPV'].append(tp/(tp+fp))
            stats['NPV'].append(tn/(tn+fn))
            stats['Thresh'].append(best_threshold)
        
        output[name] = pd.DataFrame(stats)

    return pd.concat(output, names=['Vars'])


def data_plot(datasets, weight=True, patient=False):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))

    for colour, (name, dataset) in zip(['tab:blue', 'tab:orange', 'tab:green'], datasets.items()):

        aucs = []
        roc_curves = []
        prec_curves = []

        for split, split_df in dataset.groupby(level=0):

            if patient:

                split_df = split_df.groupby('Patient_Number').median()

            if weight:
                fpr, tpr, thr = roc_curve(split_df['Y_true'], split_df['Preds'],sample_weight=split_df['Weights'])
                prec, recall, _ = precision_recall_curve(split_df['Y_true'], split_df['Preds'],sample_weight=split_df['Weights'])
                auc = roc_auc_score(split_df['Y_true'], split_df['Preds'],sample_weight=split_df['Weights'])
            else:
                fpr, tpr, thr = roc_curve(split_df['Y_true'], split_df['Preds'])
                prec, recall, _ = precision_recall_curve(split_df['Y_true'], split_df['Preds'])
                auc = roc_auc_score(split_df['Y_true'], split_df['Preds'])

                
            roc_curves.append(np.interp(np.linspace(0,1,100), fpr, tpr))
            prec_curves.append(np.interp(np.linspace(0,1,100), prec, recall))
            aucs.append(auc)
        
        ax1.plot([0,1], [0,1], ls='--', c='black')
        ax1.plot(np.linspace(0,1,100), np.median(np.array(roc_curves), axis=0), c=colour, label=f'{name} - AUC:{np.median(aucs):0.2f}')
        ax1.plot(np.linspace(0,1,100), np.mean(np.array(roc_curves), axis=0),c=colour, ls='--', label=f'{name} - AUC:{np.mean(aucs):0.2f}')
        ax1.set_title('ROC')
        ax1.set_xlabel('False positive rate')
        ax1.set_ylabel('True positive rate')
        ax1.legend(bbox_to_anchor=(0.1, -0.35, 2, 0), loc="lower center", mode="expand", ncol=3)

        ax2.plot(np.linspace(0,1,100), np.median(np.array(prec_curves), axis=0),c=colour, label=f'{name}')
        ax2.plot(np.linspace(0,1,100), np.mean(np.array(prec_curves), axis=0),c=colour, ls='--', label=f'{name}')
        ax2.set_title('PR')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        #ax2.legend()

    return fig


def km_curve(dataset, patient=False):

    ps = []
    kmf = KaplanMeierFitter()

    for t in np.linspace(0.2,0.9,30):

        indf = dataset.reset_index(['DiedvsAlive', 'survival (months)'])
        indf['DiedvsAlive'] = (indf['DiedvsAlive']=='Died')

        if patient:
            indf = indf.groupby('Patient_Number').mean()

        pred = (indf['Preds'] > t)

        try:
            st, p = sm.duration.survdiff(indf.loc[:, 'survival (months)'], indf.loc[:, 'DiedvsAlive'], pred)

        except:
            continue

        ps.append(p)

    thresh = np.linspace(0.2,0.9,30)[np.argmin(ps)]
    pred = (indf['Preds'] > thresh)

    st, p = sm.duration.survdiff(indf.loc[:, 'survival (months)'], indf.loc[:, 'DiedvsAlive'], pred)

    t_ = copy.deepcopy(kmf.fit(indf.loc[pred, 'survival (months)'], indf.loc[pred, 'DiedvsAlive'], label=True))
    f_ = copy.deepcopy(kmf.fit(indf.loc[~pred, 'survival (months)'], indf.loc[~pred, 'DiedvsAlive'], label=False))

    return {'Thresh': thresh
           ,'P-value': p
           ,'True_curve': t_
           ,'False_curve': f_}


def plot_kcmo(kmco, ax=None, logp=True):

    if not ax:
        fig, ax = plt.subplots()

    kmco['True_curve'].plot(legend=False, c='tab:green', ax=ax)
    kmco['False_curve'].plot(legend=False, c='tab:red', xlabel='Timeline', ylabel='Survival Probability', ax=ax)

    if logp:
        p = kmco['P-value']
        ax.text(0.7, 0.8, f'Log-rank p - {p:0.2f}', transform=ax.transAxes)