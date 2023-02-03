import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


# extract csv file of YoY change in inflation % of consumer prices
dataset = pd.read_csv("P_Data_Extract_From_World_Development_Indicators/Annual_Data.csv")

# select only first row (has data) and annual columns
dataset = pd.DataFrame(dataset.iloc[0,4:]).reset_index()

# rename columns
dataset.rename(columns={'index': 'date', 0: 'Annual_Inflation%'}, inplace=True)

# remove brackets from date col
dataset["date"] = dataset["date"].str.replace(" \\[.*\\]", "")
# change date col to index and datetime
dataset["date"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d")
dataset["Annual_Inflation%"] = dataset["Annual_Inflation%"].astype(float)

# set index to new date col
dataset.set_index("date", inplace=True)
# print(dataset.info())
# print(dataset.tail())

# period of high inflation in the 1970s, inflation starting to increase with 4% in 2021
# sns.displot(dataset)
# sns.lineplot(data=dataset, x=dataset.index, y="Annual_Inflation%")


# plt.show()

# explore years of high inflation
# print(dataset[dataset["Annual_Inflation%"] > 6])

# train and test split
split_index = int(round(dataset.shape[0] * 0.75, 0))
train_set = pd.DataFrame(dataset.iloc[:split_index])
test_set = pd.DataFrame(dataset.iloc[split_index:])

# print(train_set)
# print(test_set)
# ad fuller test, does pass test but not necessarily stationary from EDA plots
ad_results = adfuller(train_set['Annual_Inflation%'])
# print(ad_results)
# (-2.1665169105174336, 0.21862147991454323, 2, 59, {'1%': -3.5463945337644063, '5%': -2.911939409384601, '10%': -2.5936515282964665}, 195.73539928447815)

# find optimal diff to make time series stationary
train_diff = train_set.diff()
sns.lineplot(data=train_diff, x=train_diff.index, y="Annual_Inflation%")
# plt.show()
# optimal is diff x2


# create plots of acf and pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(train_set, lags=15, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(train_set, lags=15, zero=False, ax=ax2, method='ywm')

plt.show()

# based on plots, may be looking at ARMA (3,3)
# order_aic_bic=pd.DataFrame({'p':[], 'q':[], 'AIC':[], 'BIC':[]})
# for p in range(4):
#     # Loop over q values from 0-2
#     for q in range(4):
#         try:
#             # create and fit ARMA(p,q) model
#             model = ARIMA(train_set, order=(p,1,q))
#             results = model.fit()
            
#             # Print order and results
#             print(p, q, results.aic, results.bic)
            
#         except:
#             print(p, q, None, None)
# ARMA (2,1,0) is best from this
# 0 0 177.85893120220547 179.6655936919758
# 0 1 167.40747531928073 171.02080029882137
# 0 2 166.9913747172528 172.41136218656376
# 1 0 176.1059684891346 179.71929346867523
# 1 1 168.65623823378127 174.07622570309223
# 1 2 166.16006460685935 173.38671456594062
# 2 0 164.03705373748522 169.4570412067962
# 2 1 165.84276033639014 173.0694102954714
# 2 2 166.729739203939 175.7630516527906

best_model = ARIMA(train_set, order=(2,1,0))
results = best_model.fit()

# plots KDE, resids
# results.plot_diagnostics()
# plt.show()

# summary results to compare coeffs, Prob(Q) < 0.05 (reject null), Prob(JB) < 0.05 (reject null)
# print(results.summary())
# fail to reject null

# seasonal patterns
# acf does not show any significant evidence for seasonality


# forecast results
train_forecast = results.get_forecast(steps=test_set.shape[0], dynamic=True).predicted_mean
train_forecast = pd.Series(train_forecast,index=train_forecast.index)
train_forecast = pd.DataFrame(train_forecast)
train_forecast.columns=["Annual_Inflation%"]

print(test_set)
print(train_forecast)
plt.plot(train_set.index, train_set, label='observed')
plt.plot(train_forecast.index, train_forecast, color='r', label='forecast')
plt.plot(test_set.index, test_set, color='g', label='actual')
plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Annual_Inflation%")
plt.show()

# looks like there is a seasonality factor, will need to find optimal period
