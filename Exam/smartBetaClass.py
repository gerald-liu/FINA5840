__author__ = 'Gerald W. Liu'

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# smart_beta_scheme = {'MDR', 'RP', 'GMV', 'MSR'}
class SmartBeta:
    def __init__(self, stock_prices: pd.DataFrame, smart_beta_scheme: str):
        self.df_price = stock_prices
        self.smart_beta_scheme = smart_beta_scheme

        self.ret = self.df_price.pct_change().iloc[1:]
        self.mu = self.ret.mean()
        self.cov_matrix = self.ret.cov()
        self.n = self.ret.shape[1]

        self.w = np.ones(self.n) / self.n # equal weights by default
        self.ER = self.w @ self.mu

        self.lb = 0.01
        self.ub = 0.1
        self.tol = 1e-10

        self.rf = 0.02 # default value for risk-free rate

    def update_w(self, w):
        self.w = w
        self.ER = self.w @ self.mu

    # vol = sqrt(w @ cov @ w)
    def vol(self, w, cov):
        return np.sqrt(w @ cov @ w) # w.T = w for np 1d arrays
    
    # partial deriv of vol_p over w
    def pd_vol_w(self, w, cov):
        return cov @ w / self.vol(w, cov)

    # marginal risk contribution
    def MRC(self, w, cov):
        return w * self.pd_vol_w(w, cov) # element-wise product

    # Max Diversification Ratio objective: maximize DI = (w.T std)/(w.T omega w)
    # maximize DI = minimize -DI
    def MDR_func(self, w, cov):
        return - (w @ np.sqrt(np.diag(cov))) / self.vol(w, cov)
    
    # Risk Parity (Equal Risk Contribution) objective: minimize |MRC - Avg RC|
    def RP_func(self, w, cov):
        return np.sum((self.MRC(w, cov) - self.vol(w, cov)/cov.shape[0])**2)
    
    # Global Min Variance objective: minimize variance = w.T omega w
    def GMV_func(self, w, cov):
        return w @ cov @ w
    
    # Max Sharpe Ratio objective: maximize Sharpe = (w.T mu - rf)/(w.T omega w)
    # maximize Sharpe = minimize -Sharpe
    def MSR_func(self, w, cov, mu, rf):
        return - (w @ mu - rf) / self.vol(w, cov)
    

    def Function_SmartBeta(self):
        wgt = np.ones(self.n) / self.n # equal weights for init guess
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)}) # constraint: sum(w)=1
        bnds = ((self.lb, self.ub),) * self.n
        
        func, args = None, ()
        match self.smart_beta_scheme:
            case 'MDR':
                func = self.MDR_func
                args = (self.cov_matrix,)
            case 'RP':
                func = self.RP_func
                args = (self.cov_matrix,)
            case 'GMV':
                func = self.GMV_func
                args = (self.cov_matrix,)
            case 'MSR':
                func = self.MSR_func
                args = (self.cov_matrix, self.mu, self.rf)

        w = minimize(
                func, x0=wgt, args=args,
                method='SLSQP', constraints=cons, bounds=bnds, tol=1e-30
            )
        
        # update the member ER
        self.update_w(w.x)

        return w.x
    