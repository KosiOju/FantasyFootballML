# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:46:23 2021

@author: agbajumo
"""

# =========== practice document =============
import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder


df = pd.read_csv('ffmlDf_20-21')

print(df.dtypes)
        
enc = OrdinalEncoder()
enc.fit(df[['playerName']])

df[['playerName']] = enc.transform(df[['playerName']])

print(df.dtypes)
print(df.head())
print(enc.get_params(deep=True))


