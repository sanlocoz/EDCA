#!/usr/bin/env python
# coding: utf-8

# # ARIMA

import os
import pandas as pd

dir = "D:\Wageningen\Period 1\EDCA\Part 2\Raw Data\Stuw\data"
os.chdir(dir)

files = os.listdir(dir)
files

for file in files:
    df = pd.read_csv('file')
    df

