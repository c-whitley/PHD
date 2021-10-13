import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np 
import pandas as pd 

from scipy.stats import sem

from lifelines import KaplanMeierFitter

metadata = pd.read_excel('/mnt/c/Users/conor/Git_Projects/PHD/metadata_0206.xlsx')

def km_calculate(df, duration_col, pred_col, censored_col, thresh=0.5):

    kmf = KaplanMeierFitter()

    #fig, ax = plt.subplots()

    kts = []
    kfs = []
    ps = []

    for index, row in df.groupby('Split'):
    #for index, row in results.iterrows():

        duration = row[pred_col].reset_index(duration_col)[duration_col]
        pred = (row[pred_col].iloc[:,1] > thresh)

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

    return tsi, fsi, ps


def km_calculateB(df, duration_col, pred_col, censored_col, thresh=0.5):

    kmf = KaplanMeierFitter()

    #fig, ax = plt.subplots()

    kts = []
    kfs = []
    ps = []

    for index, row in df.groupby('Split'):
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


def stats_plot(df):

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


def survival_analysis(gb):

    for name, df in gb():

        pass

