#! /usr/bin/env python
# -*-coding: utf-8 -*-

'''FINA 5840 Homework 1
1. Use Pandas to read the daily prices of HSI constituents (given file), with the date being the index. 
    Interpolate missing entries with spline method of order=3.
2. Create a weekly (every Friday), and monthly (every monthend) dataframes
3. Use Pandas to calculate the corresponding daily, weekly and monthly returns of the above 3 dataframes
4. For the weekly dataframe, calculate the covariance matrix. OUTPUT the file as covHSI.csv
5. For the daily returns, use matplotlib to plot histogram for Tencent (700 HK), 
    you will need to take away all the 'NaN'. Use bins=100. Normalize the histogram (using density='True'), 
    make the title '700 HK', font size=10. Output the histogram and paste it in word, name it Graph.doc
6. Using the daily return dataframe: For each stock, for each month (of each year), 
    calculate the standard deviation of daily returns. Output the result into a dataframe as HSI_vol.csv. 
    As a training exercise, please first define an empty dataframe, 
    then use LOOP to get the standard deviation of the each stock for each month (of each year) 
    and then concatenate to the dataframe.
7. Rework Question 6 using the 'resample' function from pandas.
'''

__author__ = 'Gerald W. Liu'

import pandas as pd
from matplotlib import pyplot as plt

# Question 1
df = pd.read_excel('../HSI.xlsx', index_col=0)
df = df.interpolate(method='spline', order=3)

# Question 2
df_weekly = df.resample('W').last()
df_monthly = df.resample('M').last()

# Question 3
# df_daily_ret = (df/df.shift(1) - 1).iloc[1:]
df_daily_ret = df.pct_change(periods=1).iloc[1:]
df_weekly_ret = df_weekly.pct_change(periods=1).iloc[1:]
df_monthly_ret = df_monthly.pct_change(periods=1).iloc[1:]

# Question 4
df_weekly_ret_cov = df_weekly_ret.cov()
df_weekly_ret_cov.to_csv('covHSI.csv')

# Question 5
tencent = df_daily_ret['700 HK'].dropna()
plt.hist(tencent, bins=100, density=True)
plt.title('700 HK', fontsize=10)
plt.savefig('700_HK.png', dpi=300)
plt.show()

# Question 6
df_monthly_std = pd.DataFrame(columns=df_daily_ret.columns).rename_axis('Date')
start_year = df_daily_ret.index[0].year
end_year = df_daily_ret.index[-1].year

for y in range(start_year, end_year + 1):
    df_y = df_daily_ret[df_daily_ret.index.year == y]
    for m in range(1, 13):
        df_m = df_y[df_y.index.month == m]
        if df_m.empty:
            continue
        month_idx = df_m.index[-1].strftime('%Y-%m-%d')
        df_monthly_std.loc[month_idx] = df_m.std()

df_monthly_std.to_csv('HSI_vol.csv')

# Question 7
df_monthly_std_1 = df_daily_ret.resample('M').std()
df_monthly_std_1.to_csv('HSI_vol_1.csv')
