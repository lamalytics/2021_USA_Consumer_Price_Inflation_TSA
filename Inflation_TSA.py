import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# extract csv file of YoY change in inflation % of consumer prices
dataset = pd.read_csv("P_Data_Extract_From_World_Development_Indicators/Annual_Data.csv")

# select only first row (has data) and annual columns
dataset = pd.DataFrame(dataset.iloc[0,4:]).reset_index()

# rename columns
dataset.rename(columns={'index': 'date', 0: 'Annual_Inflation%'}, inplace=True)

# remove brackets from date col
dataset["date"] = dataset["date"].str.replace(" \\[.*\\]", "")
# change date col to index and datetime
dataset["date"] = pd.to_datetime("01-01-" + dataset["date"])
dataset["Annual_Inflation%"] = dataset["Annual_Inflation%"].astype(float)

# set index to new date col
dataset.set_index("date", inplace=True)
print(dataset.info())
print(dataset.tail())

# period of high inflation in the 1970s, inflation starting to increase with 4% in 2021
sns.displot(dataset)
sns.scatterplot(dataset)


plt.show()

# explore years of high inflation
# print(dataset[dataset["Annual_Inflation%"] > 6])

# train and test split

# ad fuller test, does pass test but not necessarily stationary from EDA plots
ad_results = adfuller(dataset['Annual_Inflation%'])
print(ad_results)
# (-2.1665169105174336, 0.21862147991454323, 2, 59, {'1%': -3.5463945337644063, '5%': -2.911939409384601, '10%': -2.5936515282964665}, 195.73539928447815)

# 1 order of diff helps, but may need more
dataset_diff = dataset.diff()
sns.scatterplot(dataset_diff)
plt.show()


# create plots of acf and pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(dataset, lags=10, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(dataset, lags=10, zero=False, ax=ax2)

plt.show()

# based on plots, may be looking at ARMA (3,3)


