__author__ = 'Gerald W. Liu'

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm

df_equity = pd.read_excel('HW2.xlsx',sheet_name='equity', index_col=0)
df_factor = pd.read_excel('HW2.xlsx',sheet_name='factor', index_col=0)

# returns
equity = df_equity.pct_change().dropna().values # (N, n_eq) = (133, 20)
factor = df_factor.pct_change().dropna().values # (N, n_f) = (133, 24)

omega = np.cov(equity.T) # np.cov([x1, x2, ..., xn].T) = (20, 20)
lambdas, weights = np.linalg.eig(omega) # (20,), (20, 20)

pc_idx = np.argsort(lambdas)[::-1] # sort descending by explanatory power
lambdas = lambdas[pc_idx]
weights = weights[:, pc_idx]

# Question 1
n_pc = 5
wgt_req = weights[:, :n_pc] # (n_eq, n_pc) = (20, 5)
pc = equity @ wgt_req # (133, 20) @ (20, 5) = (133, 5)

# Question 2
X_pc = sm.add_constant(pc) # add a column -> (133, 6), y = (133, 20)
model_eq_pc = sm.OLS(equity, X_pc).fit() # params = (6, 20)
beta_eq_pc = model_eq_pc.params[1:].T # exclude intercept, (20, 5)

df_beta_eq_pc = pd.DataFrame(beta_eq_pc, index=df_equity.columns, columns=np.arange(1,6))
df_beta_eq_pc.to_csv('beta.csv', header=False)

# Question 3
pfl = equity @ beta_eq_pc # (133, 20) @ (20, 5) = (133, 5)
model_pfl_pc = sm.OLS(pfl, X_pc).fit() # params = (6, 5)
beta_pfl_pc = model_pfl_pc.params[1:].T # exclude intercept, (5, 5)
beta_pfl_pc_diag = np.diag(beta_pfl_pc)
print('Question III. 3 betas =', beta_pfl_pc_diag)

# Question 4
X_eq = sm.add_constant(equity) # add a column -> (133, 21), y = (133, 5)
model_pc_eq = sm.OLS(pc, X_eq).fit() # params = (21, 5)
beta_pc_eq = model_pc_eq.params[1:].T # exclude intercept, (5, 20)

df_beta_pc_eq = pd.DataFrame(beta_pc_eq, index=np.arange(1,6), columns=df_equity.columns)
df_beta_pc_eq.to_csv('beta2.csv', header=True)

rhs = np.array([1, 1, 1, 1, 1]).reshape(-1, 1)
beta2_inv = np.linalg.pinv(beta_pc_eq)
weights = np.squeeze(beta2_inv @ rhs)
print('Question III. 4 W =\n', weights)

