# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:20:29 2019

@author: User
"""

import pandas as pd
from sklearn.model_selection import train_test_split


test_data = pd.read_csv(r"C:\Users\User\Desktop\workspace\DATA SCIENCE\Kaggle\DATA\DS Bowl_test.csv")

test_data.shape

train, test = train_test_split(test_data, test_size=0.00005)

train.shape
test.shape

type(test)

test.to_excel(r"C:\Users\User\Desktop\workspace\DATA SCIENCE\Kaggle\DATA\StratData.xlsx", 
              sheet_name='KaggleGO4IT', index=False)

strat_data = pd.read_csv(r"C:\Users\User\Desktop\workspace\DATA SCIENCE\Kaggle\DATA\StratData.csv")
y=strat_data['y']

_, strat_test = train_test_split(strat_data, test_size=0.2, stratify=y)

strat_test.shape

train_data = pd.read_csv(r"C:\Users\User\Desktop\workspace\DATA SCIENCE\Kaggle\DATA\trainDSBowl2019.csv")

start_mem = train_data.memory_usage().sum() / 1024**2  

train_data.shape

_, test = train_test_split(train_data, test_size=0.0005)

test.shape

test.to_excel(r"C:\Users\User\Desktop\workspace\DATA SCIENCE\Kaggle\DATA\RandSplNoStrat5600.xlsx", 
              sheet_name='5671rows_0.25%', index=False)




