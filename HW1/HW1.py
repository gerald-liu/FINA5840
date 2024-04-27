import pandas as pd
from matplotlib import pyplot as plt

# Question 1
df = pd.read_excel('HSI.xlsx', index_col=0)
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
