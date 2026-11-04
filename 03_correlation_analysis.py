# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:24:05 2026

@author: 15EG3019NL
"""


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA
# ==========================================

bank = fetch_ucirepo(id=222)

X = bank.data.features.copy()
y = bank.data.targets["y"].map({
    "no": 0,
    "yes": 1
})

# Split before supervised exploratory analysis
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save charts in the project folder
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(exist_ok=True)

# ==========================================
# 2. CORRELATION MATRIX
# ==========================================

numeric = X_train.select_dtypes(
    include="number"
).copy()

numeric["subscription"] = y_train

corr_matrix = numeric.corr(method="pearson")

print("\nCORRELATION MATRIX")
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    corr_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

columns = corr_matrix.columns

ax.set_xticks(range(len(columns)))
ax.set_yticks(range(len(columns)))

ax.set_xticklabels(columns, rotation=45, ha="right")
ax.set_yticklabels(columns)

for i in range(len(columns)):
    for j in range(len(columns)):
        value = corr_matrix.iloc[i, j]
        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            fontsize=8
        )

fig.colorbar(im, ax=ax, label="Pearson correlation")

plt.title("Correlation Matrix - Training Data")
plt.tight_layout()

plt.savefig(
    output_dir / "04_correlation_matrix.png",
    dpi=200
)
plt.show()

# ==========================================
# 3. CORRELATION WITH THE TARGET
# ==========================================

target_corr = (
    corr_matrix["subscription"]
    .drop("subscription")
    .sort_values()
)

print("\nCORRELATION WITH TARGET")
print(target_corr.round(3))

plt.figure(figsize=(9, 5))

target_corr.plot(kind="barh")

plt.title("Numerical Features vs Subscription")
plt.xlabel("Pearson Correlation")
plt.ylabel("Features")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()

plt.savefig(
    output_dir / "05_target_correlations.png",
    dpi=200
)
plt.show()
