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
dataset["date"] = pd.to_datetime(dataset["date"], format="%Y-%m-%d").dt.strftime('%m-%d-%Y')
dataset["Annual_Inflation%"] = dataset["Annual_Inflation%"].astype(float)

# set index to new date col
dataset.set_index("date", inplace=True)
# print(dataset.info())
# print(dataset.tail())

# period of high inflation in the 1970s, inflation starting to increase with 4% in 2021
sns.displot(dataset)
sns.scatterplot(dataset)


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

# 1 order of diff helps, but may need more
train_diff = train_set.diff()
sns.scatterplot(train_diff)
# plt.show()


# create plots of acf and pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(train_set, lags=15, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(train_set, lags=15, zero=False, ax=ax2, method='ywm')

plt.show()

# based on plots, may be looking at ARMA (3,3)
# order_aic_bic=pd.DataFrame({'p':[], 'q':[], 'AIC':[], 'BIC':[]})
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):
        try:
            # create and fit ARMA(p,q) model
            model = ARIMA(train_set, order=(p,0,q))
            results = model.fit()
            
            # Print order and results
            print(p, q, results.aic, results.bic)
            
        except:
            print(p, q, None, None)
# ARMA (1,0,1) is best from this
# 0 0 231.7562689102466 235.41355170322478
# 0 1 186.919057887745 192.40498207721228
# 0 2 173.37902755856703 180.69359314452342
# 1 0 182.15735680986313 187.64328099933041
# 1 1 167.7095375105422 175.02410309649858
# 1 2 169.62369418225194 178.7669011646974
# 2 0 175.93028090621235 183.24484649216873
# 2 1 169.6521647170993 178.79537169954477
# 2 2 171.52828217503057 182.50013055396514
# order_df = pd.DataFrame(order_aic_bic, 
#                         columns=['p','q', 'AIC', 'BIC'])

# # Print order_df in order of increasing AIC
# print(order_aic_bic.sort_values('AIC'))

# # Print order_df in order of increasing BIC
# print(order_aic_bic.sort_values('BIC'))




