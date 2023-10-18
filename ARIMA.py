#!/usr/bin/env python
# coding: utf-8

# # ARIMA

import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

dir = "C:\\Users\\user\\Downloads\\Programs\\EDCA2\\data"
output_dir = "C:\\Users\\user\\Downloads\\Programs\\EDCA2\\output"
os.chdir(dir)

files = os.listdir(dir)

def convert_date_format(input_date):
    #input is DD/MM/YYYY 
    list = input_date.split(" ")
    print(list)
    converted_date = ""
    
    if(len(list)==1):
        print(list[0])
        ans = list[0].split("-")
        print(ans)
        converted_date = str(ans[1]) + "/" + str(ans[0]) + "/" + str(ans[2]) + " 00:00"
    else:
        print(list[0])
        ans = list[0].split("-")
        print(ans)
        converted_date = str(ans[1]) + "/" + str(ans[0]) + "/" + str(ans[2])
    return converted_date

ans = []
for i in range(0, 7):#len(files)):
    os.chdir(dir)
    file_name = files[i]
    df = pd.read_csv(files[i], sep=";", dtype={'Date': parse_as_string})
    
    if(file_name.split("_")[0]=='1'): #data
        start_date = convert_date_format(df.Date[0])
        end_date = convert_date_format(df.Date[len(df)-1])
        timestamps = pd.date_range(start=start_date, end=end_date, freq='h')  
        length = len(timestamps)
        df_right =  pd.DataFrame({'Discharge': [np.nan] * length, 'h': [np.nan] * length, 'Upstream': [np.nan] * length}, index = timestamps)
        
        for j in range(0, len(df)):
            print(j)
            date_now = pd.Timestamp(convert_date_format(df.iloc[j]['Date']))
  
            df_right.loc[date_now]['Discharge'] = df.iloc[j]['Discharge']
            df_right.loc[date_now]['h'] = df.iloc[j]['h']
            df_right.loc[date_now]['Upstream'] = df.iloc[j]['Upstream']
        
        os.chdir(output_dir)
        df_right.to_csv(files[i]+'.csv', index=True, sep = ";")
        ans.append(df_right)
        
        
    #elif(file_name.split("_")[0]=='2'):
     #   continue
            
    
"""

for i in range(0, 1):#len(files)):
    os.chdir(dir)
    
    df = pd.read_csv(files[i], sep =";")
    file_name = files[i]
    stew_name = file_name.split("_")[0]
    
    fig_dir = "D:\Wageningen\Period 1\EDCA\Part 2\Raw Data\Stuw\\fig"
    os.chdir(fig_dir)
    
    #Plotting Discharge Before ARIMA
    df.plot(x='Date', y='Discharge', kind='line', figsize=(10,3))
    plt.xlabel('Time')
    plt.ylabel('Discharge')
    plt.title('Discharge ' + stew_name)
    plt.savefig('Discharge_'+stew_name+".png")
    #plt.show()
       
    mask = df['Discharge'].notnull()
    
    id_start = id_last = 0
    id_temp_start = id_temp_last = 0
    for i in range(0, len(mask)):
        if(not mask[i]):
            if(id_temp_last-id_temp_start > id_last - id_start):
                id_last = id_temp_last
                id_start = id_temp_start
                
            id_temp_start=i+1
            id_temp_last=i+1
        else:
            id_temp_last+=1
            
    if(id_temp_last-id_temp_start > id_last - id_start):
        id_last = id_temp_last
        id_start = id_temp_start
        
    id_temp_start=i+1
    id_temp_last=i+1
            
    df_full = df
    
    df = df.loc[id_start: id_last-5000]
        
    
    #Plotting Longest discharge
    df.plot(x='Date', y='Discharge', kind='line', figsize=(10,3))
    plt.xlabel('Time')
    plt.ylabel('Discharge')
    plt.title('Discharge ' + stew_name)
    plt.savefig('Longest Discharge_'+stew_name+".png")
    #plt.show()
    
    lags = len(df)-1
    acf_values = acf(x=df.Discharge, missing = "None", nlags=lags)
    
    conf_level = 1.96 / np.sqrt(len(df))  # 95% confidence level
    lower_bound = -conf_level
    upper_bound = conf_level
    
    # Plot ACF with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.stem(range(lags + 1), acf_values, markerfmt='ro', basefmt=" ", linefmt='r-', use_line_collection=True)
    plt.axhline(y=lower_bound, linestyle='--', color='gray', label='95% Confidence Interval')
    plt.axhline(y=upper_bound, linestyle='--', color='gray')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function (ACF) '+ stew_name)
    plt.legend()
    plt.savefig('ACF_'+stew_name+".png")
    #plt.show()
            
    arima_model = ARIMA(df.Discharge, order = (5,2,5))
    arima_fit = arima_model.fit()
    
    forecast_steps = 50000
    
    # Get forecast values and confidence intervals
    forecast = arima_fit.get_forecast(steps=forecast_steps)
    forecast_index = range(len(df), len(df) + forecast_steps)
    forecast_values = forecast.predicted_mean
    lower_conf_int = forecast.conf_int()['lower Discharge']  # Replace 'column_name' with your column name
    upper_conf_int = forecast.conf_int()['upper Discharge']  # Replace 'column_name' with your column name
    
    # Plot the original series and forecasts with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df.Discharge, label='Original Series')
    plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
    plt.fill_between(forecast_index, lower_conf_int, upper_conf_int, color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ARIMA Model Forecast')
    plt.legend()
    plt.show()

    #print(f'ARIMA Model Test Data MSE: {np.mean((predictions.values - test2.values)**2):.3f}')
    
"""
'''
plot_acf(df['Discharge'].dropna(), lags=20)
plot_acf(df['Upstream'].dropna(), lags=20)
plot_acf(df['Downstream'].dropna(), lags=20)
plot_acf(df['Valve'].dropna(), lags=20)
'''
    
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate example data (you can replace this with your own time series data)
np.random.seed(0)
data = np.random.randn(100)  # Generate 100 random data points

# Create a pandas DataFrame
df = pd.DataFrame(data, columns=['value'])

# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(df['value'])
plt.title('Original Time Series Data')
plt.show()

# Fit ARIMA model
# ARIMA(p, d, q) - p: order of AR terms, d: degree of differencing, q: order of MA terms
p, d, q = 1, 1, 1  # Example values for p, d, and q

model = ARIMA(df['value'], order=(p, d, q))
model_fit = model.fit()

# Print summary of the model
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()

# Forecast next 10 data points
forecast_steps = 10
forecast = model_fit.get_forecast(steps=forecast_steps)

# Print forecast
print("Forecasted values for the next", forecast_steps, "steps:")
print(forecast.predicted_mean)

# Plot the original data and forecast
plt.figure(figsize=(12, 6))
plt.plot(df['value'], label='Original Data')
plt.plot(np.arange(len(df), len(df) + forecast_steps), forecast.predicted_mean, label='Forecast', color='red')
plt.title('Original Data and Forecast')
plt.legend()
plt.show()
'''