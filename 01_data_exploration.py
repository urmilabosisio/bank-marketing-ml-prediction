# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:15:24 2026

@author: 15EG3019NL
"""


import pandas as pd
from ucimlrepo import fetch_ucirepo

# 1. Download the dataset from UCI
bank = fetch_ucirepo(id=222)

# 2. Separate features and target
X = bank.data.features
y = bank.data.targets

# 3. Combine data into a single DataFrame
df = pd.concat([X, y], axis=1)

# 4. Explore the dataset
print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET DIMENSIONS")
print(df.shape)

print("\nVARIABLES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTARGET DISTRIBUTION")
print(df["y"].value_counts())

print("\nTARGET PERCENTAGES")
print(df["y"].value_counts(normalize=True) * 100)
