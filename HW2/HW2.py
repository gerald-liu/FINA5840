#! /usr/bin/env python
# -*-coding: utf-8 -*-

'''FINA 5840 Homework 2
'''

__author__ = 'Gerald W. Liu'

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

'''Question 1
Use Pandas to read the HW2.xlsx (provided) into Dataframes 'equity' and 'factor'.
Convert them into numpy arrays and construct the corresponding simple returns.
'''
df_equity = pd.read_excel('HW2.xlsx',sheet_name='equity', index_col=0)
df_factor = pd.read_excel('HW2.xlsx',sheet_name='factor', index_col=0)

# returns
equity = df_equity.pct_change().dropna().values # (N, n_eq) = (133, 20)
factor = df_factor.pct_change().dropna().values # (N, n_f) = (133, 24)

'''Question 2
Set the following parameters: The required explanatory power 'reqExp' as 0.8;
the required minimum correlation for the factor with the eigen portfolio 'reqCorr' as 0.4;
the maximum allowed between-factor correlation 'reqFcorr' as 0.7.
'''
REQ_EXP = 0.8 # required explanatory power
REQ_CORR = 0.4 # corr(factor, PC) to maximize, acceptable min = 0.4
REQ_FCORR = 0.7 # corr(factor, factor) to minimize, acceptable max = 0.7

'''Question 3
Perform a PCA analysis on the equity returns using numpy,
find the minimum number of principal components to cover the required explanatory power (0.8).
'''
omega = np.cov(equity.T) # np.cov([x1, x2, ..., xn].T) = (20, 20)
lambdas, weights = np.linalg.eig(omega) # (20,), (20, 20)

pc_idx = np.argsort(lambdas)[::-1] # sort descending by explanatory power
lambdas = lambdas[pc_idx]
weights = weights[:, pc_idx]

var_exp = lambdas / np.sum(lambdas) # % of variance explained
n_pc = np.argmax(np.cumsum(var_exp) >= 0.8) + 1 # (cumsum>=0.8) = [F, F, T]

wgt_req = weights[:, :n_pc] # (n_eq, n_pc) = (20, 3)
pc = equity @ wgt_req # (133, 20) @ (20, 3) = (133, 3)

'''Question 4
Find the most relevant factors to represent the principalcomponents.
The algorithm as follows:
PC1: run each factor correlation with PC1.
For the first factor, if the correlation (absolute) is greater than 'reqCorr', keep it.
For the 2nd factor onward, the correlation needs to be greater than 'reqCorr' but less than the 'reqFcorr' to be kept.
After PC1, you must have some factors in the list already, go on for PC2 and then PC3:
For each factor, keep those with correlation greater than 'reqCorr' but less than the 'reqFcorr'.
'''
f_idx = set()

for i in range(pc.shape[1]):
    corr = np.zeros(factor.shape[1])
    for j in range(factor.shape[1]): # j: original factor indices
        corr[j], _ = pearsonr(pc[:, i], factor[:, j])
        
    corr_idx = np.argsort(abs(corr))[::-1]

    for _, k in np.ndenumerate(corr_idx): # k: sort descending by corr(f, PC)
        # keep if corr(f, PC) > REQ_CORR
        if abs(corr[k]) > REQ_CORR:
            keep = True
            for l in f_idx:
                corr_f, _ = pearsonr(factor[:, k], factor[:, l])
                # discard if corr(f, any_f_taken) >= REQ_FCORR
                if abs(corr_f) >= REQ_FCORR:
                    keep = False
            if keep:
                f_idx.add(k)

f_req = factor[:, list(f_idx)]
f_names = df_factor.columns[list(f_idx)]

'''Question 5
With the list of factors from Q4, normalize (standardize) their returns.
Standardize the return for the equity indexes as well.
'''
f_req_norm = (f_req - f_req.mean(axis=0)) / f_req.std(axis=0) # (133, 10)
eq_norm = (equity - equity.mean(axis=0)) / equity.std(axis=0) # (133, 20)

'''Question 6
Run a for loop for each equity index over the standardized factors from Q4: 
OLS with intercept, 
retrieve the beta, t-value and R-squared and keep them into 3 different lists. 
Output all of them into beta.csv, tvalue.csv, Rsq.csv.
'''
beta = np.zeros((eq_norm.shape[1], f_req_norm.shape[1])) # (20, 10)
tvalue = np.zeros((eq_norm.shape[1], f_req_norm.shape[1])) # (20, 10)
rsq = np.zeros(eq_norm.shape[1]) # (20,)

for i in range(eq_norm.shape[1]):
    X = sm.add_constant(f_req_norm) # add a column -> (133, 11)
    y = eq_norm[:, i] # (133, 1)
    model = sm.OLS(y, X).fit()
    beta[i, :] = model.params[1:] # exclude intercept
    tvalue[i, :] = model.tvalues[1:]
    rsq[i] = model.rsquared

df_beta = pd.DataFrame(beta, index=df_equity.columns, columns=f_names)
df_tvalue = pd.DataFrame(tvalue, index=df_equity.columns, columns=f_names)
df_rsq = pd.DataFrame(rsq, index=df_equity.columns, columns=['R-squared'])

df_beta.to_csv('beta.csv')
df_tvalue.to_csv('tvalue.csv')
df_rsq.to_csv('Rsq.csv')
