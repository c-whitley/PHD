import os
import string
import copy
from lifelines.fitters import coxph_fitter

import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import sem

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, confusion_matrix, matthews_corrcoef, auc


metadata = pd.read_excel(
    '/mnt/c/Users/conor/Git_Projects/PHD/metadata_0206.xlsx')


def pr_auc_score(y_true, y_pred, sample_weight=None):

    if sample_weight is not None:
        prec, rec, th = precision_recall_curve(y_true, y_pred, sample_weight=sample_weight)
    
    else:
        prec, rec, th = precision_recall_curve(y_true, y_pred)

    return auc(x=rec, y=prec)


def km_calculate(df, duration_col, pred_col, censored_col, thresh=0.5, patient=False):

    kmf = KaplanMeierFitter()

    #fig, ax = plt.subplots()

    kts = []
    kfs = []
    ps = []

    for index, row in df.groupby(level=0):

        duration = row[duration_col]
        pred = (row[pred_col] > thresh)
        death_obs = (row[censored_col] == 'Died')

        stat, p = sm.duration.survdiff(duration, death_obs, pred)

        ps.append((stat, p))

        try:
            kmf.fit(duration.loc[pred].values,
                    death_obs.loc[pred].values, label=True)
            kts.append(kmf.survival_function_)
            kmf.fit(duration.loc[~pred].values,
                    death_obs.loc[~pred].values, label=False)
            kfs.append(kmf.survival_function_)

        except ValueError as e:

            print(index, e)
            continue

    indxs = [df.index for df in kts]

    indxmin, indxmax = np.min(np.concatenate(
        indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    tsi = np.array([np.interp(new_x, df.index, df.values.squeeze())
                   for df in kts])

    indxs = [df.index for df in kfs]

    indxmin, indxmax = np.min(np.concatenate(
        indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    fsi = np.array([np.interp(new_x, df.index, df.values.squeeze())
                   for df in kfs])

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

        death_obs = (row[pred_col].reset_index(
            censored_col)[censored_col] == 'Died')

        stat, p = sm.duration.survdiff(duration, death_obs, pred)

        ps.append((stat, p))

        try:
            kmf.fit(duration.loc[pred, :], death_obs.loc[pred, :], label=True)
            kts.append(kmf.survival_function_)
            kmf.fit(duration.loc[~pred, :],
                    death_obs.loc[~pred, :], label=False)
            kfs.append(kmf.survival_function_)

        except ValueError as e:

            print(index, e)
            continue

    indxs = [df.index for df in kts]

    indxmin, indxmax = np.min(np.concatenate(
        indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    tsi = np.array([np.interp(new_x, df.index, df.values.squeeze())
                   for df in kts])

    indxs = [df.index for df in kfs]

    indxmin, indxmax = np.min(np.concatenate(
        indxs)), np.max(np.concatenate(indxs))

    new_x = np.linspace(indxmin, indxmax, 100)
    fsi = np.array([np.interp(new_x, df.index, df.values.squeeze())
                   for df in kfs])

    return tsi, fsi


def km_plots(tsi, fsi, axin=None, ste=False, median=False):

    if axin:
        ax = axin
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
            error = np.std(res, axis=0)

        ax.plot(m, ds='steps', c=c, label=name)
        ax.plot(m+error, ds='steps', ls='--', c=c, alpha=0.5)
        ax.plot(m-error, ds='steps', ls='--', c=c, alpha=0.5)

        ax.set_xlabel('Time (months)')
        ax.set_ylabel("Cumulative Survival")
        ax.set_ylim(0, 1)

    return ax


def stats_plot(df, fig, axes=None):

    if(axes is None):

        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(8, 4), sharey=True)
        fig.subplots_adjust(wspace=0.1)

    for col_name, ax, lab in zip(df, axes.flatten(), string.ascii_uppercase[2:]):

        ax.hist(df[col_name], edgecolor='black')
        ax.text(0.05, 0.8, lab, size='x-large', transform=ax.transAxes)
        if lab in ['A', 'D']:
            ax.set_ylabel('Frequency')

        if lab in ['D', 'E', 'F']:
            ax.set_xlabel('Score')

    print(dict(zip(df.columns, string.ascii_uppercase)))

    return fig


def box_plot(stats):
    """_summary_

    Args:
        stats (_type_): _description_

    Returns:
        _type_: _description_
    """
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(8, 12))
    fig.subplots_adjust(wspace=0.2)
    letters = string.ascii_uppercase#[2:]

    for ax, name, let in zip(axes.flatten(), stats.columns, letters):
        ax = sns.boxplot(data=stats.reset_index(),
                         x='Vars', y=name, ax=ax, whis=1)

        ax.set_xticklabels(['ASMA', 'ASMA+FTIR', 'FTIR'])
        ax.set_xlabel(None)

        if let in ['C', 'E', 'G', 'A']:
            ax.set_ylabel('Score')
        else:
            ax.set_ylabel(None)

        if name in ['MCC','AUPRC']:
            ax.set_ylim(-1.1, 1.1)

        else:
            ax.set_ylim(-0.1, 1.1)

        ax.text(-0.05, 1.05, let, size='x-large', transform=ax.transAxes)

    print(list(zip(stats.columns, letters)))

    return fig


def calc_stats(datasets, weight=True, patient=False, threshold='class'):
    """_summary_

    Returns:
        _type_: _description_
    """

    output = {}

    for name, dataset in datasets.items():

        stats = {'AUROC': [], 'AUPRC':[], 'F1': [], 'MCC': [], 'Specificity': [],
                 'Sensitivity': [], 'PPV': [], 'NPV': [], 'Thresh': []}

        #dataset['Preds'] = 1-dataset['Preds']

        if threshold == 'prog':

            best_threshold = get_opt_prog_thresh(dataset).idxmin()

        for split, split_df in dataset.groupby(level=0):

            if patient:

                split_df = split_df.groupby('Patient_Number').median()

            if threshold == 'class':

                ts = np.linspace(
                    split_df['Preds'].min(), split_df['Preds'].max(), 30)
                #f1 = pd.Series({t: f1_score(y_true=split_df['Y_true'], y_pred=(
                    #split_df['Preds'] > t), sample_weight=split_df['Weights'], average='weighted') for t in ts})
                m = pd.Series({t: matthews_corrcoef(y_true=split_df['Y_true'], y_pred=(split_df['Preds']>t), sample_weight=split_df['Weights']) for t in ts})
                best_threshold = m.idxmax()
                preds = np.array(
                    [1 if el > best_threshold else 0 for el in split_df['Preds']])

            elif threshold == 'prog':

                preds = np.array(
                    [1 if el > best_threshold else 0 for el in split_df['Preds']])

            if weight:
                t,f = get_baseline(split_df)
                cm = confusion_matrix(
                    y_true=split_df['Y_true'], y_pred=preds, sample_weight=split_df['Weights'])

                stats['AUROC'].append(roc_auc_score(
                    split_df['Y_true'], split_df['Preds'], sample_weight=split_df['Weights']))

                stats['AUPRC'].append(pr_auc_score(
                    split_df['Y_true'], split_df['Preds'])-(t/(t+f)))

                stats['F1'].append(f1_score(y_true=split_df['Y_true'], y_pred=preds,
                                   sample_weight=split_df['Weights'], average='weighted'))

            else:

                t,f = get_baseline(split_df)
                cm = confusion_matrix(y_true=split_df['Y_true'], y_pred=preds)

                stats['AUROC'].append(roc_auc_score(
                    split_df['Y_true'], split_df['Preds']))

                stats['AUPRC'].append(pr_auc_score(
                    split_df['Y_true'], split_df['Preds'])-(t/(t+f)))

                stats['F1'].append(
                    f1_score(y_true=split_df['Y_true'], y_pred=preds))

            #cm = confusion_matrix(split_df['Y_true'], preds)
            tn, fp, fn, tp = cm.ravel()

            stats['MCC'].append(matthews_corrcoef(
                split_df['Y_true'], y_pred=preds, sample_weight=split_df['Weights']))
            stats['Specificity'].append(tn/(tn+fp))
            stats['Sensitivity'].append(tp/(tp+fn))
            stats['PPV'].append(tp/(tp+fp))
            stats['NPV'].append(tn/(tn+fn))
            stats['Thresh'].append(best_threshold)

            del cm

        output[name] = pd.DataFrame(stats)

    return pd.concat(output, names=['Vars'])


def data_plot(datasets, weight=True, patient=False):
    """_summary_

    Args:
        datasets (_type_): _description_
        weight (bool, optional): _description_. Defaults to True.
        patient (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    for colour, (name, dataset) in zip(['tab:blue', 'tab:orange', 'tab:green'], datasets.items()):

        aurocs = []
        auprcs = []
        roc_curves = []
        prec_curves = []
        baselines = []

        for _, split_df in dataset.groupby(level=0):

            if patient:

                split_df = split_df.groupby('Patient_Number').median()

            if weight:
                fpr, tpr, _ = roc_curve(
                    split_df['Y_true'], split_df['Preds'], sample_weight=split_df['Weights'])
                prec, recall, _ = precision_recall_curve(
                    split_df['Y_true'], split_df['Preds'], sample_weight=split_df['Weights'])
                auroc = roc_auc_score(
                    split_df['Y_true'], split_df['Preds'], sample_weight=split_df['Weights'])
                auprc = pr_auc_score(
                    split_df['Y_true'], split_df['Preds'], sample_weight=split_df['Weights'])

                t,f = get_baseline(split_df)
                baselines.append(0.5)
            
            else:
                
                fpr, tpr, _ = roc_curve(
                    split_df['Y_true'], split_df['Preds'])
                prec, recall, _ = precision_recall_curve(
                    split_df['Y_true'], split_df['Preds'])
                auroc = roc_auc_score(split_df['Y_true'], split_df['Preds'])
                auprc = pr_auc_score(split_df['Y_true'], split_df['Preds'])

                t,f = get_baseline(split_df)
                baselines.append(t/(t+f))

            roc_curves.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
            prec_curves.append(np.interp(np.linspace(0, 1, 100), prec, recall))
            aurocs.append(auroc)
            auprcs.append(auprc-(t/(t+f)))

        ax1.plot(np.linspace(0, 1, 100), np.median(np.array(roc_curves),
                 axis=0), c=colour, label=f'{name} - {np.median(aurocs):0.2f}')


        ax1.text(-0.05, 1.05, 'A', size='x-large', transform=ax1.transAxes)
        ax1.set_xlabel('False positive rate')
        ax1.set_ylabel('True positive rate')
        ax1.legend(title='AUROC scores', bbox_to_anchor=(-0.5, -0.5, 2, 0),
                   loc="lower center", ncol=1)

        ax2.plot(np.linspace(0, 1, 100), np.median(
            np.array(prec_curves), axis=0), c=colour, label=f'{name} - {np.median(auprcs):0.2f}')
        ax2.legend(title='AUPRC scores', bbox_to_anchor=(-0.5, -0.5, 2, 0),
                   loc="lower center", ncol=1)


        ax2.text(-0.05, 1.05, 'B', size='x-large', transform=ax2.transAxes)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        #ax2.legend()
    
    #if not weight:
    baseline = np.median(baselines)
    ax2.plot([0, 1], [baseline, baseline], c='black', ls='--')

    ax1.plot([0, 1], [0, 1], ls='--', c='black')

    return fig


def km_curve(dataset, patient=False, thresh=None):
    """_summary_

    Args:
        dataset (_type_): _description_
        patient (bool, optional): _description_. Defaults to False.
        thresh (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    ps = []
    kmf = KaplanMeierFitter()

    indf = dataset.reset_index(['DiedvsAlive', 'survival (months)'])
    indf['DiedvsAlive'] = (indf['DiedvsAlive'] == 'Died')

    if patient:
        indf = indf.groupby('Patient_Number').median()

    if thresh:

        pred = (indf['Preds'] > thresh)

    else:

        thrshs = np.linspace(0.01, 0.9, 40)

        for t in thrshs:

            pred = (indf['Preds'] > t)

            try:
                st, p = sm.duration.survdiff(
                    indf.loc[:, 'survival (months)'], indf.loc[:, 'DiedvsAlive'], pred)

            except:
                continue

            ps.append(p)

        thresh = thrshs[np.argmin(ps)]
        pred = (indf['Preds'] > thresh)

    st, p = sm.duration.survdiff(
        indf.loc[:, 'survival (months)'], indf.loc[:, 'DiedvsAlive'], pred)

    t_ = copy.deepcopy(kmf.fit(
        indf.loc[pred, 'survival (months)'], indf.loc[pred, 'DiedvsAlive'], label=True))
    f_ = copy.deepcopy(kmf.fit(
        indf.loc[~pred, 'survival (months)'], indf.loc[~pred, 'DiedvsAlive'], label=False))

    return {'Thresh': thresh, 'P-value': p, 'True_curve': t_, 'False_curve': f_}


def plot_kcmo(kmco, ax=None, logp=True):
    """_summary_

    Args:
        kmco (_type_): _description_
        ax (_type_, optional): _description_. Defaults to None.
        logp (bool, optional): _description_. Defaults to True.
    """    
    if not ax:
        fig, ax = plt.subplots()

    kmco['True_curve'].plot_survival_function(legend=False, c='tab:red', ax=ax)
    kmco['False_curve'].plot_survival_function(
        legend=False, c='tab:green', xlabel='Timeline', ylabel='Survival Probability', ax=ax)

    if logp:
        p = kmco['P-value']
        if p < 0.01:
            ax.text(0.6, 0.8, f'p - < 0.01', transform=ax.transAxes)
        else:
            ax.text(0.6, 0.8, f'p - {p:0.3f}', transform=ax.transAxes)


def lifelines_plots(dataset, patient=False, ax=None, thresh=0.5, add_risk=False):
    """_summary_

    Args:
        dataset (_type_): _description_
        patient (bool, optional): _description_. Defaults to False.
        ax (_type_, optional): _description_. Defaults to None.
        thresh (float, optional): _description_. Defaults to 0.5.
        add_risk (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if not ax:
        fig, ax = plt.subplots(figsize=(7, 6))

    indf = dataset.reset_index(['DiedvsAlive', 'survival (months)'])
    indf['DiedvsAlive'] = 1*(indf['DiedvsAlive'] == 'Died')
    indf.index = indf.index.set_names('Split', level=0)

    if patient:
        indf = indf.groupby('Patient_Number').median()
        #indf = indf.groupby(['Split', 'Patient_Number']).median()

    T = indf['survival (months)']
    E = indf['DiedvsAlive']
    P = (indf['Preds'] > thresh)

    results = logrank_test(T[P], T[~P], E[P], E[~P], alpha=.99)
    #results.print_summary(style="latex", decimals=0.2)
    cph = CoxPHFitter()
    cph.fit(indf[['DiedvsAlive', 'survival (months)', 'Preds']],
            duration_col='survival (months)', event_col='DiedvsAlive')

    kmf = KaplanMeierFitter()
    kmf_T = kmf.fit(T[P], event_observed=E[P], label='Death within one year')
    ax = kmf.plot_survival_function(
        legend=False, c='tab:red', ax=ax, show_censors=True)

    kmf = KaplanMeierFitter()
    kmf_F = kmf.fit(T[~P], event_observed=E[~P], label='Lived beyond one year')
    ax = kmf.plot_survival_function(
        legend=False, c='tab:blue', ax=ax, show_censors=True)

    if add_risk:
        add_at_risk_counts(kmf_T, kmf_F, ax=ax)
    plt.tight_layout()
    ax.set_ylabel('Survival probability')
    ax.set_xlabel('Survival (months)')

    return {'KM': {'True': kmf_T, 'False': kmf_F}, 'Logrank': results, 'Cox': cph}


def lifelines_plots2(dataset, patient=False, ax=None, thresh=0.5, add_risk=False):
    """_summary_

    Args:
        dataset (_type_): _description_
        patient (bool, optional): _description_. Defaults to False.
        ax (_type_, optional): _description_. Defaults to None.
        thresh (float, optional): _description_. Defaults to 0.5.
        add_risk (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(7, 5))

    indf = dataset.reset_index(['DiedvsAlive', 'survival (months)'])
    indf['DiedvsAlive'] = 1*(indf['DiedvsAlive'] == 'Died')

    all_folds = []
    t = np.linspace(0, 120, 100)

    sk = 0

    for fold_i, fold_df in indf.groupby(level=0):

        if patient:
            fold_df = fold_df.groupby('Patient_Number').median()

        fold_dict = {}

        T = fold_df['survival (months)']
        E = fold_df['DiedvsAlive']
        P = (fold_df['Preds'] > thresh)

        if P.nunique() == 1:
            sk += 1
            continue

        results = logrank_test(T[P], T[~P], E[P], E[~P], alpha=.99)
        fold_dict['Results'] = results

        kmf = KaplanMeierFitter()
        kmf_T = kmf.fit(T[P], event_observed=E[P], timeline=t,
                        label='Death within one year')
        fold_dict['Death'] = kmf_T

        kmf = KaplanMeierFitter()
        kmf_F = kmf.fit(T[~P], event_observed=E[~P],
                        timeline=t, label='Lived beyond one year')
        fold_dict['Alive'] = kmf_F

        fold_dict['T_Curve'] = kmf_T.survival_function_
        fold_dict['F_Curve'] = kmf_F.survival_function_
        fold_dict['Timeline'] = kmf_T.survival_function_

        cph = CoxPHFitter()
        cph.fit(fold_df[['DiedvsAlive', 'survival (months)', 'Preds', 'Weights']],
                duration_col='survival (months)', event_col='DiedvsAlive', weights_col='Weights',robust=True)

        fold_dict['cox'] = cph

        all_folds.append(fold_dict)

    med_T = pd.concat([fold['T_Curve'] for fold in all_folds], axis=1)
    med_F = pd.concat([fold['F_Curve'] for fold in all_folds], axis=1)
    ps = [fold['Results'].p_value for fold in all_folds]
    cs = [fold['cox'].summary['p'] for fold in all_folds]

    print(np.median(cs))

    for med in [med_T, med_F]:

        median = med.median(axis=1)

        quants = np.quantile(med, [0.25, 0.75], axis=1).squeeze()  # [0,:]
        # return quants
        #        u_quantile, l_quantile = np.quantile(med, [0.33,0.67], axis=0).squeeze()[0,:]
        x = med.index
        ax.plot(x, median)
        ax.fill_between(x, quants[0, :], quants[1, :], alpha=0.3)

    print(sk)
    return all_folds


def get_opt_prog_thresh(dataset):

    def thresh_test(dataset, patient=False, thresh=0.5, add_risk=False):

        indf = dataset.reset_index(['DiedvsAlive', 'survival (months)'])
        indf['DiedvsAlive'] = 1*(indf['DiedvsAlive'] == 'Died')

        if patient:
            indf = indf.groupby('Patient_Number').median()

        T = indf['survival (months)']
        E = indf['DiedvsAlive']
        P = (indf['Preds'] > thresh)

        results = logrank_test(T[P], T[~P], E[P], E[~P], alpha=.99)
        #results.print_summary(style="latex", decimals=0.2)
        return results.p_value

    ths = {}

    for th in np.linspace(dataset['Preds'].min(), dataset['Preds'].max(), 30):

        try:
            p = thresh_test(dataset, patient=True, thresh=th)
            ths[th] = p

        except ValueError as error:
            continue

    ths = pd.Series(ths)

    return ths


def get_baseline(split_df):

    counts = split_df[['Y_true']].value_counts()

    return counts[1], counts[0]