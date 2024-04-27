#%%
import numpy as np
import pandas as pd

#%%
df_raw = pd.read_csv('crsp-monthly.txt', delim_whitespace=True)
df_raw = df_raw.drop_duplicates(['date', 'TICKER'])
# %%
df = df_raw.pivot(index='date', columns='TICKER', values='PRC')
# %%
df = df.drop('date')
# %%
df.index = pd.to_datetime(df.index, format="%Y%m%d")
# %%
df = df.replace('.', np.nan).ffill()
# %%
df = df.astype(float)
df[df<0] = np.nan
df = df.dropna(axis=1, how='all') # drop cols with all NA
# %%
df_a = df.drop(df.columns[np.std(df, axis=0) == 0], axis=1)
# %%
# df_a.loc['1975-04-30'].to_frame().T

# np.random.seed(42)
# data = np.random.normal(0,1, size=[100,3])
# covmat = pd.DataFrame(data).cov()
## pandas cov uses ddof=1, whereas np.cov(data.T) uses ddof=0
# covmat.shape

