# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:19:48 2026

@author: 15EG3019NL
"""


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ucimlrepo import fetch_ucirepo

# ==========================================
# 1. LOAD DATA
# ==========================================

bank = fetch_ucirepo(id=222)

df = pd.concat(
    [bank.data.features, bank.data.targets],
    axis=1
)

# Folder for charts
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

print("Dataset shape:", df.shape)

# ==========================================
# 2. DATA QUALITY
# ==========================================

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATED ROWS")
print(df.duplicated().sum())

print("\nNUMERICAL SUMMARY")
print(df.describe())

# ==========================================
# 3. TARGET DISTRIBUTION
# ==========================================

target_counts = df["y"].value_counts()
target_percent = (
    df["y"].value_counts(normalize=True) * 100
)

print("\nTARGET COUNTS")
print(target_counts)

print("\nTARGET PERCENTAGES")
print(target_percent.round(2))

plt.figure(figsize=(7, 5))

target_counts.reindex(
    ["no", "yes"], fill_value=0
).plot(kind="bar")

plt.title("Bank Deposit Subscription")
plt.xlabel("Subscription")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(output_dir / "01_target_distribution.png")
plt.show()

# ==========================================
# 4. AGE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["age"].dropna(),
    bins=30,
    edgecolor="black"
)

plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(output_dir / "02_age_distribution.png")
plt.show()

# ==========================================
# 5. SUBSCRIPTION RATE BY JOB
# ==========================================

job_rate = (
    df.groupby("job")["y"]
      .apply(lambda x: (x == "yes").mean() * 100)
      .sort_values()
)

print("\nSUBSCRIPTION RATE BY JOB (%)")
print(job_rate.round(2))

plt.figure(figsize=(10, 6))

job_rate.plot(kind="barh")

plt.title("Subscription Rate by Profession")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Profession")
plt.tight_layout()

plt.savefig(output_dir / "03_subscription_by_job.png")
plt.show()
