__author__ = 'Gerald W. Liu'

import pandas as pd
from matplotlib import pyplot as plt
from EFF import EfficientFrontier as EF

data = pd.read_excel('./biggestETFData.xlsx', sheet_name='US-only', index_col=0)
ef = EF(df_price=data, risk_measure='cvar', alpha=0.01)

frontier = ef.frontier() # {'cvar_range': cvar_range, 'weights': wgt_vec}

# EFF_output = pd.DataFrame({'CVaR': frontier['cvar_range'], 'Mean': ef.mu_range})
# EFF_output.to_csv('EFF_output.csv')

plt.plot(frontier['cvar_range'], ef.mu_range)
plt.title(f'Mean-CVaR Frontier')
plt.xlabel('CVaR')
plt.ylabel('Mean')
plt.savefig('EFF.png')
