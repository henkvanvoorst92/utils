import pandas as pd
import numpy as np
import pickle
import time
import os
#import statsmodels as sm
from scipy import stats
import pingouin as pg


# make a splitvar
def create_splitvar(df, varlist, drop_na=True):
    df_out = df[varlist].sum(axis=1) >= len(varlist)
    return df_out


# split data in 2 groups for analysis
def split_data(df, splitvar):
    dct_df = {}
    for un in df[splitvar].dropna().unique():
        dct_df[un] = df[df[splitvar] == un]
    return dct_df


# Perform chi2 test on var, use splitvar to make 2 groups
def Chi_squared(df, var, splitvar):
    crosstab = pd.crosstab(index=df[var], columns=df[splitvar])
    chi2, pvalue, ddof, expected = stats.chi2_contingency(crosstab)
    return chi2, pvalue, crosstab


# Perform ANOVA test on var, use split_dct to retrieve data per group
def ANOVA(var, split_dct):
    tmp = []
    for k, v in split_dct.items():
        tmp.append(v[var].dropna())
    statistic, p = stats.f_oneway(*tmp)
    return statistic, p, tmp


# non-parametric version of ANOVA (kruskall-wallis)
def KKW(var, split_dct):
    tmp = []
    for k, v in split_dct.items():
        tmp.append(v[var].dropna())
    statistic, p = stats.kruskal(*tmp)
    return statistic, p, tmp


# Perform Mann-Withney-U test on two groups of var in different DataFrames
def MWU(df1, df2, var):
    t, p = stats.mannwhitneyu(df1[var], df2[var], alternative='two-sided',  nan_policy='omit')
    return t, p


# Perform T-test test on two groups of var in different DataFrames
def T_test(df1, df2, var):
    t, p = stats.ttest_ind(df1[var], df2[var], nan_policy='omit')
    return t, p


# Create a crosstab
def crosstab_descr(crosstab):
    rel = crosstab / crosstab.sum(axis=0)
    df_out = pd.DataFrame(index=crosstab.index.astype(int),
                          columns=crosstab.columns)
    for col in rel.columns:
        for ix in rel.index:
            absol = crosstab.loc[ix][col].astype(int)
            perc = round(rel.loc[ix][col] * 100, 1)
            # catnames.append(int(ix))
            df_out.loc[ix, col] = "{} ({}%)".format(absol, perc)
    return df_out


# Descriptive statistics of 2 normal disctributed groups (mean, std)
def normdist_descr(x1, x2, digits=0):
    m1, std1 = round(x1.mean(),digits), round(x1.std(), digits)
    m2, std2 = round(x2.mean(),digits), round(x2.std(), digits)
    if digits==0:
        m1, std1 = int(m1), int(std1)
        m2, std2 = int(m2), int(std2)

    return ["{} ({})".format(m1, std1), "{}({})".format(m2, std2)]


# Descescriptive statistics of 2 non-normal disctributed groups (median, 25-75 IQR)
def non_normdist_descr(x1, x2, digits=1):
    x1, x2 = x1.dropna(), x2.dropna()
    m1, (iq1a, iq1b) = round(x1.median(),digits), np.percentile(x1, [25, 75]).round(digits)
    m2, (iq2a, iq2b) = round(x2.median(),digits), np.percentile(x2, [25, 75]).round(digits)
    if digits==0:
        m1, iq1a, iq1b = int(m1), int(iq1a), int(iq1b)
        m2, iq2a, iq2b = int(m2), int(iq2a), int(iq2b)

    out1 = "{} ({};{})".format(m1, int(iq1a), int(iq1b))
    out2 = "{} ({};{})".format(m2, int(iq2a), int(iq2b))
    return [out1, out2]


def multigroup_descr(dist, list_of_xs, digits=None):
    out = []
    for x in list_of_xs:
        if dist == 'normal':
            if digits is None:
                m, std = int(x.mean()), str(int(round(x.std())))
                out.append("{} ({})".format(m, std))
            else:
                m, std = int(x.mean()), str(round(x.std()))
                out.append("{} ({})".format(m, std))
        else:
            if digits is None:
                m, (iq_low, iq_high) = \
                    round(x.median()), np.percentile(x, [25, 75])
                out.append("{} ({};{})".format(int(m), int(round(iq_low)), int(round(iq_high))))
            else:
                m, (iq_low, iq_high) = \
                    round(x.median(), digits), np.percentile(x, [25, 75])
                out.append("{} ({};{})".format(m, round(iq_low, digits), round(iq_high, digits)))
    return out


def to_categories(df, var, addname='_cat'):
    df[var + addname] = df[var].astype('category').cat.codes
    df[var + addname] = df[var + addname].replace(-1, np.NaN)
    return df


# input to retrieve all statistics are:
# a dataframe df with all the data
# a variable in the dataframe to split the data into two groups
# a dictionary (dct_type) with variable name - analysis type (mwu, t-test, Chi2, ANOVA, kkw)
def all_statistics(df, splitvar, dct_type,
                   verbal=False, digits=None,
                   t_test_digits=1,
                   mwu_digits=1):
    dct = split_data(df, splitvar)
    ks = list(dct.keys())
    no_groups = len(ks)
    df1, df2 = dct[ks[0]], dct[ks[1]]

    ixs = ks.copy()
    ixs.extend(['Statistic', 'p_value', 'Stat_technique',
                'p<0.05', 'p<0.001', 'p<0.0001', 'p<1e-5'])
    df_out = pd.DataFrame(index=ixs)
    for var, datatype in dct_type.items():
        if verbal:
            print(var, datatype)

        if datatype == 'mwu':
            statistic, pvalue = MWU(df1, df2, var)
            descr = non_normdist_descr(df1[var], df2[var], digits=mwu_digits)
            # paste all output in a new df row
            newrow = descr
            newrow.extend([statistic, pvalue, 'MWU', pvalue < 0.05,
                           pvalue < 0.001, pvalue < 0.0001, pvalue < 1e-5])
            df_out[var] = newrow

        elif datatype == 't-test':
            statistic, pvalue = T_test(df1, df2, var)
            descr = normdist_descr(df1[var], df2[var], digits=t_test_digits)
            newrow = descr
            newrow.extend([statistic, pvalue, 'T-test', pvalue < 0.05,
                           pvalue < 0.001, pvalue < 0.0001, pvalue < 1e-5])
            df_out[var] = newrow

        if datatype == 'Chi2':
            statistic, pvalue, crosstab = Chi_squared(df, var, splitvar)
            d = crosstab_descr(crosstab)

            newrow = list(np.repeat(np.NaN, no_groups))
            newrow.extend([statistic, pvalue, 'Chi2', pvalue < 0.05,
                           pvalue < 0.001, pvalue < 0.0001, pvalue < 1e-5])
            df_out[var] = newrow
            # concatenation along columns since index = columns of final df
            df_out = pd.concat([df_out,
                                d.reset_index(drop=True).T], axis=1).loc[ixs]  # keep order of indices

        if datatype == 'ANOVA':
            # compute anova statistic an p-value for
            # continuous x over multiple groups
            statistic, pvalue, list_of_xs = ANOVA(var, dct)
            descr = multigroup_descr('normal', list_of_xs)
            newrow = descr
            newrow.extend([statistic, pvalue, 'ANOVA',
                           pvalue < 0.05, pvalue < 0.001, pvalue < 0.0001, pvalue < 1e-5])
            df_out[var] = newrow

        # kruskall-wallis test for non-parametric version of ANOVA
        if datatype == 'KKW':
            statistic, pvalue, list_of_xs = KKW(var, dct)
            descr = multigroup_descr('non-normal', list_of_xs, digits=digits)
            newrow = descr
            newrow.extend([statistic, pvalue, 'Kruskall_Wallis',
                           pvalue < 0.05, pvalue < 0.001, pvalue < 0.0001, pvalue < 1e-5])
            df_out[var] = newrow

    # compute the amount of datapoints in each
    # full
    lastrow = []
    for k, v in dct.items():
        lastrow.append(v.shape[0])
    reps = df_out.shape[0] - len(lastrow)
    lastrow.extend(list(np.repeat(np.NaN, reps)))
    df_out['total'] = lastrow

    if False in df_out.columns:
        df_out = df_out.rename(index={False: 'all_other', True: splitvar})

    return df_out.T


# In dct_type all the variables where descriptive statistics
# are required for are given as keys
# the values are one of t-test, mwu, Chi2 or ANOVA
# representing the test to perform
dct_type = {'togroin': 't-test',
            'NIHSS_BL': 'mwu',
            'ASPECTS_BL': 'mwu',
            'age1': 't-test',
            'premrs': 'Chi2',
            'prev_str': 'Chi2',
            'occlsegment_c_short': 'Chi2',
            'prev_dm': 'Chi2',
            'ivtrom': 'Chi2',
            'collaterals': 'Chi2',
            'reg_part': 'Chi2'}


def describe_CI(df, confidence=.95):
    df_out = df.describe(percentiles=[.025, 0.25, .5, .75, .975]) \
        .loc[['mean', '50%', 'std', '25%', '75%', '2.5%', '97.5%']]
    n = df.shape[0]
    m = df.mean()
    se = stats.sem(df, nan_policy='omit')
    z = stats.t.ppf((1 + confidence) / 2, n - 1)
    low = m - z * se
    hi = m + z * se
    df_out.loc[len(df_out)] = se
    df_out.loc[len(df_out)] = low
    df_out.loc[len(df_out)] = hi
    df_out.index = ['mean', 'median', 'Std', 'p25', 'p75',
                    'p2_5', 'p97_5', 'sem', 'CI_low', 'CI_high']
    return df_out


################# OLD CODE from here ############################
def continuous_descriptive(df):
    df_out = pd.DataFrame(columns=df.columns)

    minimum = list(df.min())
    df_out.loc[len(df_out)] = minimum

    maximum = list(df.max())
    df_out.loc[len(df_out)] = maximum

    mean = list(df.mean())
    df_out.loc[len(df_out)] = mean

    median = list(df.median())
    df_out.loc[len(df_out)] = median

    std = list(df.std())
    df_out.loc[len(df_out)] = std

    skewness = list(df.skew())
    df_out.loc[len(df_out)] = skewness

    kurtosis = list(df.kurt())
    df_out.loc[len(df_out)] = kurtosis

    df_out.index = ['min', 'max', 'mean', 'median', 'std', 'skew', 'kurt']

    return df_out


def anderson_sign(data, distribution):
    statistic = stats.anderson(data, dist=distribution).statistic
    p05 = stats.anderson(data, dist=distribution).critical_values[2]
    if statistic < p05:
        outcome = 'rejected_p>0.05'
    else:
        outcome = 'accepted_p<0.05'

    return outcome


def distribution_tests(df):  # input column params

    df_out = pd.DataFrame(columns=['shapiro', 'kolmogirov', 'dagostino',
                                   'anderson_norm', 'anderson_exp',
                                   'anderson_log'])

    for c in df.columns:
        data = df[c].values.astype('float')

        shapiro = stats.shapiro(data)[1]
        kolmogirov = stats.kstest(data, cdf='norm').pvalue
        dagostino = stats.normaltest(data).pvalue
        anderson_norm = anderson_sign(data, 'norm')
        anderson_exp = anderson_sign(data, 'expon')
        anderson_log = anderson_sign(data, 'logistic')

        df_out.loc[len(df_out)] = [shapiro, kolmogirov, dagostino,
                                   anderson_norm, anderson_exp, anderson_log]

    return df_out


def bland_altman_values(data1, data2):
    """
    data 1: list, pd series, or np arr measurements of method 1
    data 2: list, pd series, or np arr measurements of method 2
    returns bland altman analysis results with mean difference, standard deviation, and 95%CI (LoA)
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    agreement = 1.96
    high, low = md + agreement * sd, md - agreement * sd

    return md, low, high

def correlation(x, y, analysis='pearson'):
    """
    Computes the correlation coefficient
    x,y: arrays with x to relate to y
    analysis: any of pearson (standard regression), spearman (rank ordered), kendall tau (ordinal data)
    returns
    """
    if analysis == 'pearson':
        res = stats.pearsonr(x,y)
    elif analysis=='spearman':
        res = stats.spearmanr(x,y)
    elif analysis=='kendall':
        res = stats.kendalltau(x,y)
    return res

def get_significance_symbol(p_value, sign_dct):
    symbol = ''
    for p_thresh, sym in sign_dct.items():
        if p_value<p_thresh:
            symbol = sym
    return symbol

def rank_variables(df_col, dct_type):
    out = []
    for term in df_col:
        val = np.NaN
        for k,v in dct_type.items():
            if k in term or k==term:
                val = v
        out.append(val)
    return out

def final_results_column(df, digits=1, estimate_var='Estimate', lower_var='lower', upper_var='upper', p_var='p_value'):
    #creates a single column for a results table
    def format_row(row):
        estimate = round(row[estimate_var], digits)
        lower_ci = round(row[lower_var], digits)
        upper_ci = round(row[upper_var], digits)
        p_value = row[p_var]

        # Determine the significance stars
        if p_value < 0.00001:
            stars = '***'
        elif p_value < 0.001:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = ''

        # Format the new column
        return f"{estimate} ({lower_ci}; {upper_ci}){stars}"

    # Apply the formatting function to each row
    #df['Formatted'] = df.apply(format_row, axis=1)
    return df.apply(format_row, axis=1)

def add_model_performance(data,
                          perf_vars=['Log_Likelihood', 'Pseudo_R_Squared', 'AIC', 'X2_brant', 'brant_p_value'],
                         digits=[0,2,0,1,3]):

    ix, out = [],[]
    for (pv,digit) in zip(perf_vars,digits):
        ix.append(pv)
        if digit>0:
            pv_out = round(data[pv][0],digit)
        else:
            pv_out = round(data[pv][0])
        out.append(pv_out)

    out = pd.DataFrame(data=out, index=ix)
    return out