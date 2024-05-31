#! /usr/bin/env python
# -*-coding: utf-8 -*-

__author__ = 'Gerald W. Liu'

'''FINA 5840 Homework 3
Use the HSI stocks monthly return series from HW1 to find the weight vector 
using the following smart beta schemes.
Use an upper bound weight of 10% for each stock.
a. Maximum Diversification Ratio (MDR)
b. Global Minimum Variance (GMV)
c. Maximum Sharpe Ratio (MSR)
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_excel('../HSI.xlsx', index_col=0)
df = df.interpolate(method='spline', order=3)
df_monthly = df.resample('M').last()
data = df_monthly.pct_change().iloc[1:]

# vol = sqrt(w @ cov @ w)
def vol(w, cov):
    return np.sqrt(w @ cov @ w) # w.T = w for np 1d arrays

# partial deriv of vol_p over w
def pd_vol_w(w, cov):
    return cov @ w / vol(w, cov)

# marginal risk contribution
def MRC(w, cov):
    return w * pd_vol_w(w, cov) # element-wise product

# Risk Parity (Equal Risk Contribution) objective: minimize |MRC - Avg RC|
def ERC_func(w, cov):
    return np.sum((MRC(w, cov) - vol(w, cov)/cov.shape[0])**2)

# Max Diversification Ratio objective: maximize DI = (w.T std)/(w.T omega w)
def MDR_func(w, cov):
    return - (w @ np.sqrt(np.diag(cov))) / vol(w, cov)

# Global Min Variance objective: minimize variance = w.T omega w
def GMV_func(w, cov):
    return w @ cov @ w

# Max Decorrelation objective: minimize (w.T corr_mat w)
def MDC_func(w, corr):
    return w @ corr @ w

# Max Sharpe Ratio objective: maximize Sharpe = (w.T mu - rf)/(w.T omega w)
def MSR_func(w, cov, mu, rf):
    return - (w @ mu - rf) / vol(w, cov)

# weights optimizer
def pfl_optimizer(func, n, args=(), long_only=True):
    wgt = np.ones(n) / n # equal weights for init guess
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)}) # constraint: sum(w)=1
    if long_only:
        bnds = ((0, 0.1),) * n # upper bound weight of 10% for each stock
    else:
        bnds = ((-0.1, 0.1),) * n
    
    w = minimize(
            func, x0=wgt, args=args,
            method='SLSQP', constraints=cons, bounds=bnds, tol=1e-30
        )
    return w.x

n = data.shape[1]
cov = data.cov()
corr = data.corr()
mu = data.mean()
rf = 0.04

w_ERC = pfl_optimizer(ERC_func, n, args=(cov,))
w_MDR = pfl_optimizer(MDR_func, n, args=(cov,))
w_GMV = pfl_optimizer(GMV_func, n, args=(cov,))
w_MDC = pfl_optimizer(MDC_func, n, args=(corr,))
w_MSR = pfl_optimizer(MSR_func, n, args=(cov, mu, rf))

df_weights = pd.DataFrame(
    np.column_stack((w_ERC, w_MDR, w_GMV, w_MDC, w_MSR)),
    index=data.columns, columns=['ERC', 'MDR', 'GMV', 'MDC', 'MSR']
)
df_weights.to_csv('weights.csv', float_format='%.6f')
