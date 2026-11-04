# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:26:56 2026

@author: 15EG3019NL
"""


import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. LOAD DATA
# ==========================================

bank = fetch_ucirepo(id=222)

X = bank.data.features.copy()

y = bank.data.targets["y"].map({
    "no": 0,
    "yes": 1
}).astype(int)

# ==========================================
# 2. SELECT FEATURES
# ==========================================

# Use information available before the
# current marketing campaign.

features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "pdays",
    "previous",
    "poutcome"
]

X = X[features].copy()

# ==========================================
# 3. HANDLE SPECIAL VALUES
# ==========================================

# pdays = -1 means no previous contact.
# Some dataset loaders may represent it as NaN.

no_previous_contact = (
    X["pdays"].eq(-1) | X["pdays"].isna()
)

X["was_contacted_before"] = (
    ~no_previous_contact
).astype(int)

X.loc[no_previous_contact, "pdays"] = np.nan

# Keep "unknown" as an explicit category
# rather than silently replacing it.

# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = (
    X_train.select_dtypes(include="number")
    .columns.tolist()
)

categorical_features = (
    X_train.select_dtypes(exclude="number")
    .columns.tolist()
)

print("NUMERIC FEATURES")
print(numeric_features)

print("\nCATEGORICAL FEATURES")
print(categorical_features)

# ==========================================
# 6. NUMERICAL PIPELINE
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ==========================================
# 7. CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="missing"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

# ==========================================
# 8. COMBINE TRANSFORMATIONS
# ==========================================

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# IMPORTANT: fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training parameters
X_test_processed = preprocessor.transform(X_test)

# ==========================================
# 9. VERIFY RESULTS
# ==========================================

print("\nORIGINAL TRAIN SHAPE")
print(X_train.shape)

print("\nPROCESSED TRAIN SHAPE")
print(X_train_processed.shape)

print("\nPROCESSED TEST SHAPE")
print(X_test_processed.shape)

print("\nTARGET DISTRIBUTION - TRAIN")
print(y_train.value_counts(normalize=True).round(3))

print("\nTARGET DISTRIBUTION - TEST")
print(y_test.value_counts(normalize=True).round(3))

print("\nFIRST 10 PROCESSED FEATURE NAMES")
print(preprocessor.get_feature_names_out()[:10])
