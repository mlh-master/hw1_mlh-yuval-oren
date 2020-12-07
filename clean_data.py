# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    c_ctg=CTG_features.drop(extra_feature,axis=1)
    c_ctg_nan=c_ctg.apply(pd.to_numeric, errors='coerce')
    vec = ()
    mydict = dict()
    for column in c_ctg_nan:
        mydict[column] = c_ctg_nan[column].dropna()

    c_ctg = mydict

    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_ctg_new = CTG_features.drop(extra_feature, axis=1)
    c_ctg_new_nan = c_ctg_new.apply(pd.to_numeric, errors='coerce')
    c_cdf = c_ctg_new_nan

    for column in c_ctg_new_nan:
        p = c_ctg_new_nan[column].value_counts(normalize=True, ascending=True)
        u = np.unique(c_ctg_new_nan[column])
        u = pd.DataFrame(u).dropna()
        u = np.ravel(u)
        p = np.ravel(p)
        idx = c_ctg_new_nan[column].isna()
        size = np.count_nonzero(idx)
        idx = np.where(idx == 1)

        c_cdf[column].iloc[idx] = np.random.choice(u, size=size, p=p)


    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    d_summary = dict()
    for column in c_feat:
        d_summary[column] = c_feat[column].describe()

    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_new = c_feat.to_dict('Series')
    c_no_outlier = c_new
    for column in c_new:
        Q1 = d_summary[column]["25%"]
        Q3 = d_summary[column]["75%"]
        IQR = Q3 - Q1
        a = int(Q3 + 1.5 * IQR)
        b = int(Q1 - 1.5 * IQR)
        c = np.array(c_new[column])
        high = c[c >= a]
        low = c[c <= b]
        c_no_outlier[column] = c_no_outlier[column].replace(high, np.nan)
        c_no_outlier[column] = c_no_outlier[column].replace(low, np.nan)

    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """
    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    filt_feature = ()
    c = np.array(c_cdf[feature])
    filt_feature = c[c < thresh]
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    nsd_res = {} #CTG_features
    if (flag == True) & (mode=='none'):
        nsd_res=CTG_features
        plot_scaled = pd.DataFrame(nsd_res).loc[:, (x, y)].plot.hist(bins=100, title='Unscaled Histogram')
        plot_scaled.set_xlabel('bins')
        plt.show()
        
    if mode == 'standard':
        mean = ()
        std = ()
        for column in CTG_features:
            mean = CTG_features.loc[:,column].mean()
            std = CTG_features.loc[:,column].std()
            nsd_res[column] = (CTG_features.loc[:,column] - mean) / std
        if flag == True:
            plot_scaled = pd.DataFrame(nsd_res).loc[:, (x, y)].plot.hist(bins=100, title='Scaled Standard Histogram')
            plot_scaled.set_xlabel('bins')
            plt.show()
    if mode == 'mean':
        mean = ()
        xmax = ()
        xmin = ()
        for column in CTG_features:
            mean = CTG_features.loc[:,column].mean()
            xmax = CTG_features.loc[:,column].max()
            xmin = CTG_features.loc[:,column].min()
            nsd_res[column] = (CTG_features.loc[:,column] - mean) / (xmax - xmin)
        if flag == True:
            plot_scaled = pd.DataFrame(nsd_res).loc[:, (x, y)].plot.hist(bins=100, title='Scaled Mean Histogram')
            plot_scaled.set_xlabel('bins')
            plt.show()
    if mode == 'MinMax':
        xmin = ()
        xmax = ()
        for column in CTG_features:
            xmax = CTG_features.loc[:,column].max()
            xmin = CTG_features.loc[:,column].min()
            nsd_res[column] = (CTG_features.loc[:,column] - xmin) / (xmax - xmin)
        if flag == True:
            plot_scaled = pd.DataFrame(nsd_res).loc[:, (x, y)].plot.hist(bins=100, title='Scaled MinMax Histogram')
            plot_scaled.set_xlabel('bins')
            plt.show()
    return pd.DataFrame(nsd_res)
