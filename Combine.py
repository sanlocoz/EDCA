# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:05:43 2023

@author: user
"""
#!/usr/bin/env python
# coding: utf-8

# # ARIMA

import numpy as np
import seaborn as sns
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

dir = "C:\\Users\\user\\Downloads\\Programs\\EDCA2\\big_data"
os.chdir(dir)

files = os.listdir(dir)

df = pd.read_csv(files[0], sep=";", index_col= 0)
df_cleaned = df.dropna()

title = ['Discharge Inlaat Baakse Beek', 'h Inlaat Baakse Beek','Upstream Inlaat Baakse Beek',
         'Discharge Stuw Baakseweg', 'h Stuw Baakseweg','Upstream Stuw Baakseweg',
         'Discharge Stuw Bobbink', 'h Stuw Bobbink','Upstream Stuw Bobbink',
         'Discharge Stuw Hilge', 'h Stuw Hilge','Upstream Stuw Hilge',
         'Discharge Stuw Horsterkamp', 'h Stuw Horsterkamp','Upstream Stuw Horsterkamp',
         'Discharge Stuw Veengoot', 'h Stuw Veengoot','Upstream Stuw Veengoot',
         'Discharge Strockhorsterdijk', 'h Strockhorsterdijk','Upstream Strockhorsterdijk',
         'Discharge Wientjesvoort', 'h Wientjesvoort','Upstream Wientjesvoort',
         'Discharge Baakse Beek Elterweg', 'h Baakse Beek Elterweg',
         'Discharge Groene Kanaal', 'h Groene Kanaal',
         'Discharge Heeckerenbeek', 'h Heeckerenbeek','Upstream Heeckerenbeek',
         'ET Potential', 'ET Actual','Precipitation',
         ]
upstream = [2]
downstream = [11]

lags = 51

r_square = np.zeros(shape=(lags, len(upstream)*2))


for i in range(len(upstream)):
    df_selection = df.iloc[:,list([upstream[i], downstream[i]])]
    
    split_index = 100969

    df_before_meandering = df_selection.iloc[0:100969,]
    df_after_meandering = df_selection.iloc[100969:,]
    
    
    
    for lag in range(0, lags):
        print(lag)
        #Before meandering plot
        df_before_meandering['lag'] = df_before_meandering.iloc[:,0].shift(lag)
        df_before_meandering_r = df_before_meandering.dropna()
       
        sns.set_theme(style="darkgrid", font="serif") 
        sns_plot_before = sns.jointplot(x = df_before_meandering.iloc[:,2], y = df_before_meandering.iloc[:,1], data=df_before_meandering, kind='reg')
        
        r, p = stats.pearsonr(df_before_meandering_r.iloc[:,2], df_before_meandering_r.iloc[:,1])
        sns_plot_before.ax_joint.annotate(f'$r^2 = {r**2:.5f}$',
                            xy=(0.1, 0.9), xycoords='axes fraction',
                            ha='left', va='center',
                            bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
        r_square[lag, i*2+0] = r**2
        sns_plot_before.set_axis_labels(title[upstream[i]], title[downstream[i]], fontsize=12)  # Set the desired labels and fontsize
        sns_plot_before.fig.suptitle("Lag "+str(lag)+" Before Meandering", y=1)
        
        plt.show()
        sns_plot_before.savefig("before"+ str(lag) + ".png")
        plt.close()
        
        #After meandering plot
        df_after_meandering['lag'] = df_after_meandering.iloc[:,0].shift(lag)
        df_after_meandering_r = df_after_meandering.dropna()
       
        sns.set_theme(style="darkgrid", font="serif") 
        sns_plot_after = sns.jointplot(x = df_after_meandering.iloc[:,2], y = df_after_meandering.iloc[:,1], data=df_after_meandering, kind='reg')
        
        r, p = stats.pearsonr(df_after_meandering_r.iloc[:,2], df_after_meandering_r.iloc[:,1])
        sns_plot_after.ax_joint.annotate(f'$r^2 = {r**2:.5f}$',
                            xy=(0.1, 0.9), xycoords='axes fraction',
                            ha='left', va='center',
                            bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
        r_square[lag, i*2+1] = r**2
        
        sns_plot_after.set_axis_labels(title[upstream[i]], title[downstream[i]], fontsize=12)  # Set the desired labels and fontsize
        sns_plot_after.fig.suptitle("Lag "+str(lag)+" After Meandering", y=1)
        
        plt.show()
        
        sns_plot_after.savefig("after"+ str(lag) + ".png")
        
        plt.close()
        
        
            
        #sns_plot = sns.jointplot(x = df_after_meandering.iloc[:,0], y = df_after_meandering.iloc[:,1], data=df_selection, kind='scatter')

        
np.savetxt('r_square.csv', r_square, delimiter=';')


