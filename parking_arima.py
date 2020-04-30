#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:00:13 2020

@author: crjo1001
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import re

dataset = pd.read_csv('/mnt/SmarterCityLab/Processed_Data/CSV_Files/1700_ParkingOccupancy.csv',delimiter=";",parse_dates=['Tidpunkt'])

dataset.insert(5,'date',dataset['Tidpunkt'].dt.date)
dataset.insert(6,'month',dataset['Tidpunkt'].dt.month)
dataset.insert(7,'day',dataset['Tidpunkt'].dt.day)
dataset.insert(8,'hour',dataset['Tidpunkt'].dt.hour)
dataset.insert(9,'minute',dataset['Tidpunkt'].dt.minute)
dataset.insert(10,'weekday',dataset['Tidpunkt'].dt.dayofweek+1)

dataset['month'] = dataset.month.map("{:02}".format)
dataset['day'] = dataset.day.map("{:02}".format)
dataset['hour'] = dataset.hour.map("{:02}".format)
dataset['minute'] = dataset.minute.map("{:02}".format)

dataset['date_str'] = dataset['date'].apply(str)
dataset['weekday_str'] = dataset['weekday'].apply(str)
dataset['month_str'] = dataset['month'].apply(str)
dataset['day_str'] = dataset['day'].apply(str)
dataset['hour_str'] = dataset['hour'].apply(str)
dataset['minute_str'] = dataset['minute'].apply(str)

#dataset.info()

dataset.insert(11,'weekdayhour',dataset['weekday_str']+'_'+dataset['hour_str'])
dataset.insert(12,'dayhour',dataset['day_str']+'_'+dataset['hour_str'])
dataset.insert(13,'monthday',dataset['month_str']+'_'+dataset['day_str'])
dataset.insert(14,'datehour',dataset['date_str']+'_'+dataset['hour_str'])
dataset.insert(15,'datetime',dataset['date_str']+' '+dataset['hour_str']+':'+dataset['minute_str'])


dataset = dataset.rename(columns={"Beläggning (%)": "Beläggning"})
dataset.info()

# =============================================================================
# välj kolumn som plotten ska baseras på samt vilken anläggning du vill följa
# =============================================================================

dataset.Anläggning.unique()

plotvar = 'Tidpunkt'
anlaggning = 'S t Nicolai'
start_date = date(2019, 6, 1)
end_date = date(2019, 8, 1)

mask = (dataset['date'] > start_date) & (dataset['date'] <= end_date)

plotdata_beta = dataset.loc[mask]
 
# plotdata_beta = dataset[start_dt <= (dataset['date']) & (dataset['date']) <= end_dt]
plotdata_small = plotdata_beta[[plotvar, 'Anläggning', 'antal_lediga_platser']]
# new_set = dataset_small.set_index([plotvar])

# set_pivot = new_set.pivot(columns='Anläggning', values='Beläggning (%)')
# set_pivot = set_pivot.drop(columns='Klubbstugorna')

# set_corr = set_pivot.corr()

plotdata = plotdata_small[[plotvar, 'Anläggning', 'antal_lediga_platser']].groupby([plotvar, 'Anläggning'], as_index=False).mean()
plotdata = plotdata[(plotdata['Anläggning']==anlaggning)]
plotdata = plotdata.sort_values([plotvar],ascending=[True])

sns.set(style='whitegrid')
sns.lineplot(x=plotvar,
              y='antal_lediga_platser',
              hue='Anläggning',
              data=plotdata,
              alpha=0.8)


plotdata = plotdata.drop(columns='Anläggning')
plotdata = plotdata.set_index([plotvar])
# plotdata = plotdata.set_index()

pd.plotting.autocorrelation_plot(plotdata)

from statsmodels.tsa import arima_model
from sklearn.metrics import mean_squared_error

model = arima_model.ARIMA(plotdata, order=(3,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

X = plotdata.antal_lediga_platser
size = int(len(X) *0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
	model = arima_model.ARIMA(history, order=(3,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# test.info()
test2 = pd.Series.reset_index(test,drop=True)
plt.figure(dpi=1200)
plt.plot(predictions, color='red')
plt.plot(test2, alpha=0.8, linewidth=1)
plt.show()


model_fit.save('/mnt/SmarterCityLab/Graphs/test.pkl')


preds = pd.DataFrame(predictions)
preds.columns=['Prediction']
# preds.rename({'0': 'Prediction'}, axis='columns', inplace=True)
outcome = pd.DataFrame(test.reset_index())
results = pd.concat([outcome, preds], axis = 1)

start_date_str = start_date.strftime("%Y-%m-%d")
start_date_str = re.sub('-','',start_date_str)
end_date_str = end_date.strftime("%Y-%m-%d")
end_date_str = re.sub('-','',end_date_str)

filespec = anlaggning + ' ' + start_date_str + '_' + end_date_str

results.to_csv('/mnt/SmarterCityLab/Graphs/ParkingOccupancy_'+filespec+'.csv',sep=";")
# plt.savefig('/mnt/SmarterCityLab/Graphs/'+anlaggning+'.png', dpi=1200)
