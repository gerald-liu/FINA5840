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
    def __init__(self,
        df_price: pd.DataFrame,
        mu_range: np.ndarray,
        risk_measure: str = 'vol',
    ):
        self.df_price = df_price
        self.risk_measure = risk_measure
        self.alpha = 0.05
        self.lb = 0 # weight > 0, cannot short
        self.ub = 1
        self.n = self.df_price.shape[1] # number of assets
        # self.mu_range = np.arange(0.0055, 0.013, 0.0002)
        self.mu_range = mu_range
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
    data = raw_data.resample('W').last().iloc[:, :20]

    df_returns = data.pct_change().dropna()
    df_mean = df_returns.mean()
    df_std = df_returns.std()
    df_mean_min = df_mean.min()
    df_mean_max = df_mean.max()
    print(f'min mean: {df_mean.idxmin()}, {df_mean_min :.4f}')
    print(f'max mean: {df_mean.idxmax()}, {df_mean_max :.4f}')
    print(f'min std: {df_std.idxmin()}, {df_std.min() :.4f}')

    risk_measures = ['vol', 'var', 'es']
    mu_rng = np.linspace(df_mean_min, df_mean_max, 100)

    fig1, axs1 = plt.subplots(3, 1, figsize=(8, 9), layout='constrained')
    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 9), layout='constrained')

    for i in range(len(risk_measures)):
        rm = risk_measures[i]
        print(f'\nrisk measure: {rm}')

        ef = EfficientFrontier(df_price=data, mu_range=mu_rng, risk_measure=rm)
        frontier = ef.frontier()
        std_range = np.sqrt(frontier['vol_range'])
        mu_range = ef.get_mu_range()

        min_risk_idx = np.argmin(std_range)
        min_risk_mu = mu_range[min_risk_idx]
        min_risk_std = std_range[min_risk_idx]
        min_risk_w = frontier['weights'].iloc[min_risk_idx, :]
        
        print(f'min risk when mu = {min_risk_mu :.4f} and std = {min_risk_std :.4f}')
        print(f'min risk Sharpe: {min_risk_mu / min_risk_std :.4f}')

        ax1 = axs1[i]
        ax1.plot(std_range, mu_range)
        ax1.set_title(f'Mean-Variance Frontier using {rm}')
        ax1.set_xlabel("std")
        ax1.set_ylabel("mean")

        ax2 = axs2[i]
        ax2.bar([ticker[:-3] for ticker in data.columns], min_risk_w)
        ax2.set_title(f'Min-risk weights using {rm}')

    fig1.savefig('efficient_frontier.png')
    fig2.savefig('min_risk_weights.png')
