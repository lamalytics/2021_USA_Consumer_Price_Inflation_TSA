import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
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
print(ad_results)
# (-2.1665169105174336, 0.21862147991454323, 2, 59, {'1%': -3.5463945337644063, '5%': -2.911939409384601, '10%': -2.5936515282964665}, 195.73539928447815)

# find optimal diff to make time series stationary
# train_diff = train_set.diff().dropna()
sns.lineplot(data=train_set, x=train_set.index, y="Annual_Inflation%")
# plt.show()
# optimal is diff x2


# create plots of acf and pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(train_set, lags=18, zero=False, ax=ax1)
# 8 period seasonality based on ACF

# Plot the PACF of df
plot_pacf(train_set, lags=18, zero=False, ax=ax2, method='ywm')

plt.show()

# based on plots, may be looking at ARMA (3,3)
# order_aic_bic=pd.DataFrame({'p':[], 'q':[], 'AIC':[], 'BIC':[]})
# for p in range(4):
#     # Loop over q values from 0-2
#     for q in range(4):
#         try:
#             # create and fit ARMA(p,q) model
#             model = ARIMA(train_set, order=(p,0,q))
#             results = model.fit()
            
#             # Print order and results
#             print(p, q, results.aic, results.bic)
            
#         except:
#             print(p, q, None, None)
# ARMA (1,0,1) is best from this
# 0 0 231.7562689102466 235.41355170322478
# 0 1 186.919057887745 192.40498207721228
# 0 2 173.37902755856703 180.69359314452342
# 0 3 171.32000458167386 180.46321156411932
# 1 0 182.15735680986313 187.64328099933041
# 1 1 167.7095375105422 175.02410309649858
# 1 2 169.62369418225194 178.7669011646974
# 1 3 170.93606556497144 181.907913943906
# 2 0 175.93028090621235 183.24484649216873
# 2 1 169.6521647170993 178.79537169954477
# 2 2 171.52828217503057 182.50013055396514
# 2 3 172.89205842321098 185.69254819863465
# 3 0 169.35055446363245 178.49376144607794
# 3 1 171.0932825427185 182.06513092165306
# 3 2 172.36111097228255 185.16160074770622
# 3 3 172.25146308637767 186.88059425829044
# decomp = seasonal_decompose(train_set['Annual_Inflation%'], 
#                             period=4)
# decomp.plot()
# plt.show()

# find the optimal seasonal order thru trial and error
# can use grid search for optimal (takes high cpu power)
best_model = SARIMAX(train_set, order=(1,0,1), seasonal_order=(0,1,0,16))
results = best_model.fit()

# plots KDE, resids
# results.plot_diagnostics()
# plt.show()

# summary results to compare coeffs, Prob(Q) < 0.05 (reject null), Prob(JB) < 0.05 (reject null)
# print(results.summary())
# fail to reject null


# forecast results
train_forecast = results.get_forecast(steps=test_set.shape[0]).predicted_mean
train_forecast = pd.Series(train_forecast,index=train_forecast.index)
train_forecast = pd.DataFrame(train_forecast)
train_forecast.columns=["Annual_Inflation%"]

# model error metrics
mse = mean_squared_error(test_set,train_forecast)
print("MSE: ", mse)
mape = mean_absolute_percentage_error(test_set,train_forecast)
print("MAPE: ", mape)

# expanded forecast
test_forecast = results.get_forecast(steps=test_set.shape[0] + 10).predicted_mean
test_forecast =  pd.Series(test_forecast,index=test_forecast.index)
test_forecast = pd.DataFrame(test_forecast)
# slice to select the forecasted years not in dataset
test_forecast = test_forecast.iloc[test_set.shape[0]:]
train_forecast.columns=["Annual_Inflation%"]

# plot 3 plots of observed data as a whole, the train_set vs test_set, and 10-year
plt.plot(dataset.index, dataset, label='observed')
plt.plot(train_forecast.index, train_forecast, color='r', label='forecast')
plt.plot(test_forecast.index, test_forecast, color='g', label='10-year')
plt.xlabel("Date")
plt.xticks(rotation=90)
plt.ylabel("Annual_Inflation%")
plt.show()

# looks like there is a seasonality factor, will need to find optimal period
