"""FINA 5840 Lecture 4 Class Assignment 2
Efficient Frontier construction with scipy
"""

__author__ = "Gerald W. Liu"


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# risk_measure = {'vol', 'var', 'es'}
class EfficientFrontier:
    def __init__(self, df_price: pd.DataFrame, risk_measure: str = 'vol'):
        self.df_price = df_price
        self.risk_measure = risk_measure
        self.alpha = 0.05
        self.lb = 0 # weight > 0, cannot short
        self.ub = 1
        self.n = self.df_price.shape[1] # number of assets
        # self.mu_range = np.arange(0.0055, 0.013, 0.0002)
        self.mu_range = np.arange(0.0065, 0.0105, 0.0002)
        self.ret = df_price.pct_change().dropna()
        self.E_ret = self.ret.mean() # sample mean of each asset
        self.omega = self.ret.cov() # covariance matrix

    def get_mu_range(self):
        return self.mu_range
    
    def weighted_return(self, w):
        return self.ret @ w

    # mean_variance frontier, objective function for variance
    def vol(self, w, cov_mat):
        return w.T @ cov_mat @ w
    
    # objective function for (historical) VaR
    def VaR(self, w, q):
        return self.weighted_return(w).quantile(q)

    # objective function for ES (cVaR)
    def ES(self, w, q):
        pfl_ret = self.weighted_return(w)
        return pfl_ret[pfl_ret < pfl_ret.quantile(q)].mean()

    def frontier(self):
        wgt = {} # weights
        iter_len = self.mu_range.shape[0]
        vol_range = np.zeros(iter_len)

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

            if self.risk_measure == 'vol':
                # minimize() always optimizes the first arg of the input function
                w = minimize(
                    self.vol, x_0,
                    method='SLSQP',constraints=constr, bounds=bnds,
                    args=(self.omega)
                )

            elif self.risk_measure == 'var':
                w = minimize(
                    self.VaR, x_0,
                    method='SLSQP',constraints=constr, bounds=bnds,
                    args=(self.alpha)
                )
                
            elif self.risk_measure == 'es':
                w = minimize(
                    self.ES, x_0,
                    method='SLSQP',constraints=constr, bounds=bnds,
                    args=(self.alpha)
                )

            vol_range[i] = self.vol(w.x, self.omega)
            wgt[mu].extend(np.squeeze(w.x))

        wgt_vec = pd.DataFrame.from_dict(wgt, orient='columns').T

        return {'vol_range': vol_range, 'weights': wgt_vec}
    

if __name__ == "__main__":
    raw_data = pd.read_excel('../HSI.xlsx', sheet_name='data', index_col=0)
    data = raw_data.resample('M').last().iloc[:, :5]
    risk_measures = ['vol', 'var', 'es']

    fig, axs = plt.subplots(3, 1, figsize=(5, 9), layout='constrained')

    for i in range(len(risk_measures)):
        rm = risk_measures[i]
        print(f'risk measure: {rm}')

        ef = EfficientFrontier(df_price=data, risk_measure=rm)
        frontier = ef.frontier()
        std_range = np.sqrt(frontier['vol_range'])
        mu_range = ef.get_mu_range()

        min_risk_idx = np.argmin(std_range)
        optimal_mu = mu_range[min_risk_idx]
        optimal_std = std_range[min_risk_idx]
        optimal_w = frontier['weights'].iloc[min_risk_idx, :]
        
        print(f'optimal weights: {optimal_w}')
        print(f'where mu = {optimal_mu} and std = {optimal_std}')

        ax = axs[i]
        ax.plot(std_range, mu_range)
        ax.set_title(f'Mean-Variance Frontier using {rm}')
        ax.set_xlabel("std")
        ax.set_ylabel("mean")

    fig.savefig('efficient_frontier.png')
