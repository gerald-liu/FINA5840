__author__ = 'Gerald W. Liu'

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# risk_measure = {'var', 'cvar'}
class EfficientFrontier:
    def __init__(self, df_price: pd.DataFrame, risk_measure: str, alpha=0.01):
        self.df_price = df_price
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.lb = 0 # weight > 0, cannot short
        self.ub = 1
        self.n = self.df_price.shape[1] # number of assets
        
        self.ret_raw = df_price.pct_change().iloc[1:]
        self.ret_sample_idx = np.random.choice(self.ret_raw.index, size=10000, replace=True)
        self.ret_sample = self.ret_raw.loc[self.ret_sample_idx]
        self.E_ret = self.ret_sample.mean() # sample mean of each asset

        self.mu_range = np.linspace(self.E_ret.min(), self.E_ret.max(), 100)
    
    def weighted_return(self, w):
        return self.ret_sample @ w
    
    # objective function for (historical) VaR
    # minimized -VaR
    def VaR(self, w, q):
        return - self.weighted_return(w).quantile(q)

    # objective function for ES (CVaR)
    # minimize -CVaR
    def CVaR(self, w, q):
        pfl_ret = self.weighted_return(w)
        return - pfl_ret[pfl_ret < pfl_ret.quantile(q)].mean()

    def frontier(self):
        wgt = {} # weights
        iter_len = self.mu_range.shape[0]
        cvar_range = np.zeros(iter_len)

        for i in range(iter_len):
            mu = self.mu_range[i]
            wgt[mu] = []
            x_0 = np.ones(self.n) / self.n

            # produce a tuple of shape (n, 2)
            bnds = ((self.lb, self.ub), ) # ((a, b, ...), ) for tuple addition
            for _ in range(self.n - 1):
                bnds += ((self.lb, self.ub), )

            constr = (
                {
                    'type': 'eq', # equal
                    'fun': lambda x: 1 - np.sum(x) # sum(w) = 1
                },
                {
                    'type': 'eq',
                    'fun': lambda x: mu - (self.E_ret.values @ x) # mu = ER.T @ w
                }
            )
            
            if self.risk_measure == 'var':
                # minimize() always optimizes the first arg of the input function
                w = minimize(
                    self.VaR, x_0,
                    method='SLSQP',constraints=constr, bounds=bnds,
                    args=(self.alpha)
                )
                
            elif self.risk_measure == 'cvar':
                w = minimize(
                    self.CVaR, x_0,
                    method='SLSQP',constraints=constr, bounds=bnds,
                    args=(self.alpha)
                )

            cvar_range[i] = self.CVaR(w.x, self.alpha)
            wgt[mu].extend(np.squeeze(w.x))

        wgt_vec = pd.DataFrame.from_dict(wgt, orient='columns').T

        return {'cvar_range': cvar_range, 'weights': wgt_vec}
