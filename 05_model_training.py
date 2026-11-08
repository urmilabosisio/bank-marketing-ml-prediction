# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:13:10 2026

@author: 15EG3019NL
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

# ==========================================
# 1. LOAD DATA
# ==========================================

bank = fetch_ucirepo(id=222)

X = bank.data.features.copy()
y = bank.data.targets["y"].map({
    "no": 0,
    "yes": 1
})

# Use only selected pre-contact features
features = [
    "age", "job", "marital", "education",
    "default", "balance", "housing", "loan",
    "pdays", "previous", "poutcome"
]

X = X[features].copy()

X["was_contacted_before"] = (
    X["pdays"].notna() &
    (X["pdays"] != -1)
).astype(int)

X["pdays"] = X["pdays"].replace(-1, np.nan)

# ==========================================
# 2. TRAIN / VALIDATION / TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print("Training:", X_train_fit.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ==========================================
# 3. PREPROCESSING
# ==========================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline,
     categorical_features)
])

# ==========================================
# 4. MODELS
# ==========================================

models = {
    "Dummy": DummyClassifier(
        strategy="prior"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
}

results = []

for name, estimator in models.items():

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", estimator)
    ])

    model.fit(X_train_fit, y_train_fit)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_val, y_proba
        ),
        "PR_AUC": average_precision_score(
            y_val, y_proba
        )
    })

    if name == "Logistic Regression":
        logistic_model = model
        logistic_predictions = y_pred
        logistic_probabilities = y_proba

# ==========================================
# 5. COMPARE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_val,
    logistic_predictions,
    display_labels=["No", "Yes"],
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# 7. PRECISION-RECALL CURVE
# ==========================================

PrecisionRecallDisplay.from_predictions(
    y_val,
    logistic_probabilities
)

plt.title("Logistic Regression - Precision Recall")
plt.tight_layout()
plt.show()

# Keep X_test / y_test untouched until
# final model selection and evaluation.


# ==========================================
# 8. THRESHOLD OPTIMIZATION
# ==========================================

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_val,
    logistic_probabilities
)

# The final precision/recall point has no threshold.
precision = precision[:-1]
recall = recall[:-1]

# Calculate F1 for each threshold.
f1 = 2 * precision * recall / (
    precision + recall + 1e-12
)

# Recall achieved by our original model.
baseline_recall = recall_score(
    y_val,
    logistic_predictions
)

# Only accept thresholds that preserve recall.
valid = recall >= baseline_recall - 1e-10

# Maximize F1 subject to the recall constraint.
valid_indices = np.flatnonzero(valid)

best_idx = valid_indices[
    np.argmax(f1[valid])
]

best_threshold = thresholds[best_idx]

print("\nTHRESHOLD OPTIMIZATION")
print(f"Original threshold: 0.5000")
print(f"Optimal threshold: {best_threshold:.4f}")

# Predictions using the optimized threshold.
optimized_predictions = (
    logistic_probabilities >= best_threshold
).astype(int)

# ==========================================
# 9. EVALUATE OPTIMIZED MODEL
# ==========================================

print("\nOPTIMIZED METRICS")

print(
    "Accuracy:",
    round(accuracy_score(
        y_val, optimized_predictions
    ), 4)
)

print(
    "Precision:",
    round(precision_score(
        y_val, optimized_predictions,
        zero_division=0
    ), 4)
)

print(
    "Recall:",
    round(recall_score(
        y_val, optimized_predictions
    ), 4)
)

print(
    "F1-score:",
    round(f1_score(
        y_val, optimized_predictions
    ), 4)
)

# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_val,
    optimized_predictions,
    display_labels=["No", "Yes"],
    cmap="Blues"
)

plt.title("Optimized Logistic Regression")
plt.tight_layout()
plt.show()



# ==========================================
# 11. RANDOM FOREST & GRADIENT BOOSTING
# ==========================================

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import clone

# Preprocessing for HistGradientBoosting:
# ordinal encoding instead of one-hot encoding.

categorical_hgb = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor_hgb = ColumnTransformer([
    ("numeric", SimpleImputer(
        strategy="median"
    ), numeric_features),
    ("categorical", categorical_hgb,
     categorical_features)
])

advanced_models = {
    "Random Forest": Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessing", preprocessor_hgb),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42
        ))
    ])
}

advanced_results = []

for name, model in advanced_models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_fit, y_train_fit)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    advanced_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_val, probabilities
        ),
        "PR_AUC": average_precision_score(
            y_val, probabilities
        )
    })

    print(f"{name} completed.")

# ==========================================
# 12. FINAL MODEL COMPARISON
# ==========================================

advanced_df = pd.DataFrame(advanced_results)

comparison_df = pd.concat([
    results_df,
    advanced_df
], ignore_index=True)

print("\nFINAL MODEL COMPARISON")
print(
    comparison_df.round(4).to_string(index=False)
)

print("\nBEST MODEL BY PR_AUC:")

best_model_name = comparison_df.loc[
    comparison_df["PR_AUC"].idxmax(),
    "Model"
]

print(best_model_name)

# Save validation results
comparison_df.to_csv(
    "model_comparison.csv",
    index=False
)


# ==========================================
# 13. IMPROVE GRADIENT BOOSTING
# ==========================================

from sklearn.metrics import precision_recall_curve

# Identify categorical columns after preprocessing.
n_numeric = len(numeric_features)
n_categorical = len(categorical_features)

categorical_indices = list(
    range(n_numeric, n_numeric + n_categorical)
)

# Tell HistGradientBoosting which features
# represent categories, not ordered numbers.
gradient_model = advanced_models["Gradient Boosting"]

gradient_model.set_params(
    classifier__categorical_features=categorical_indices
)

# Refit Gradient Boosting with correct
# treatment of categorical features.
gradient_model.fit(X_train_fit, y_train_fit)

# Random Forest is already fitted.
forest_model = advanced_models["Random Forest"]

# ==========================================
# 14. THRESHOLD OPTIMIZATION
# ==========================================

candidate_models = {
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model,
    "Gradient Boosting": gradient_model
}

# Minimum acceptable recall:
# original Logistic Regression performance.
min_recall = 0.6229

optimized_results = []

for name, model in candidate_models.items():

    probabilities = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = (
        precision_recall_curve(
            y_val,
            probabilities
        )
    )

    # Last PR point has no threshold.
    p = precision[:-1]
    r = recall[:-1]

    f1_scores = (
        2 * p * r / (p + r + 1e-12)
    )

    # Keep only thresholds respecting recall.
    valid = r >= min_recall

    valid_indices = np.flatnonzero(valid)

    best_idx = valid_indices[
        np.argmax(f1_scores[valid])
    ]

    best_threshold = thresholds[best_idx]

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    optimized_results.append({
        "Model": name,
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(
            y_val, predictions
        ),
        "Precision": precision_score(
            y_val, predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions
        ),
        "F1": f1_score(
            y_val, predictions
        ),
        "ROC_AUC": roc_auc_score(
            y_val, probabilities
        ),
        "PR_AUC": average_precision_score(
            y_val, probabilities
        )
    })

# ==========================================
# 15. COMPARE OPTIMIZED MODELS
# ==========================================

optimized_df = pd.DataFrame(optimized_results)

optimized_df = optimized_df.sort_values(
    by="F1",
    ascending=False
)

print("\nOPTIMIZED MODEL COMPARISON")

print(
    optimized_df.round(4).to_string(index=False)
)

best_model_name = optimized_df.iloc[0]["Model"]
best_threshold = optimized_df.iloc[0]["Threshold"]

print("\nBEST MODEL:", best_model_name)
print("BEST THRESHOLD:", round(best_threshold, 4))

optimized_df.to_csv(
    "optimized_model_comparison.csv",
    index=False
)


# ==========================================
# 16. HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.base import clone

# Use the corrected Gradient Boosting pipeline.
tuning_model = clone(gradient_model)

param_distributions = {
    "classifier__learning_rate": [
        0.02, 0.05, 0.1
    ],
    "classifier__max_iter": [
        100, 150, 250
    ],
    "classifier__max_leaf_nodes": [
        7, 15, 31
    ],
    "classifier__min_samples_leaf": [
        10, 20, 40
    ],
    "classifier__l2_regularization": [
        0.0, 1.0, 5.0
    ]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=2,
    verbose=1,
    refit=True
)

# Tune exclusively on the training subset.
search.fit(X_train_fit, y_train_fit)

print("\nBEST PARAMETERS")
print(search.best_params_)

print("\nBEST CROSS-VALIDATION PR-AUC")
print(round(search.best_score_, 4))

# ==========================================
# 17. VALIDATION
# ==========================================

best_model = search.best_estimator_

probabilities = best_model.predict_proba(
    X_val
)[:, 1]

print("\nTUNED MODEL ROC-AUC")
print(round(
    roc_auc_score(y_val, probabilities), 4
))

print("\nTUNED MODEL PR-AUC")
print(round(
    average_precision_score(y_val, probabilities), 4
))

# ==========================================
# 18. THRESHOLD OPTIMIZATION
# ==========================================

precision, recall, thresholds = (
    precision_recall_curve(
        y_val, probabilities
    )
)

p = precision[:-1]
r = recall[:-1]

f1_values = (
    2 * p * r / (p + r + 1e-12)
)

# Preserve our previous recall requirement.
min_recall = 0.6229

valid_indices = np.flatnonzero(
    r >= min_recall
)

best_idx = valid_indices[
    np.argmax(f1_values[valid_indices])
]

tuned_threshold = thresholds[best_idx]

predictions = (
    probabilities >= tuned_threshold
).astype(int)

print("\nTUNED MODEL RESULTS")

print("Threshold:", round(tuned_threshold, 4))

print("Accuracy:", round(
    accuracy_score(y_val, predictions), 4
))

print("Precision:", round(
    precision_score(
        y_val, predictions, zero_division=0
    ), 4
))

print("Recall:", round(
    recall_score(y_val, predictions), 4
))

print("F1:", round(
    f1_score(y_val, predictions), 4
))

# ==========================================
# 19. SAVE TUNING RESULTS
# ==========================================

cv_results = pd.DataFrame(search.cv_results_)

cv_results.to_csv(
    "gradient_boosting_tuning.csv",
    index=False
)


# ==========================================
# 20. FINAL MODEL TRAINING
# ==========================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.base import clone
import joblib

# Select the pre-tuning Gradient Boosting.
final_model = clone(
    advanced_models["Gradient Boosting"]
)

# Preserve categorical-feature handling.
final_model.set_params(
    classifier__categorical_features=categorical_indices
)

# Refit using all non-test observations.
final_model.fit(X_train, y_train)

# Fixed threshold chosen on validation data.
final_threshold = 0.1028

# ==========================================
# 21. FINAL TEST EVALUATION
# ==========================================

test_proba = final_model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_proba >= final_threshold
).astype(int)

print("\nFINAL TEST RESULTS")

print("Accuracy:", round(
    accuracy_score(y_test, test_pred), 4
))

print("Precision:", round(
    precision_score(
        y_test, test_pred, zero_division=0
    ), 4
))

print("Recall:", round(
    recall_score(y_test, test_pred), 4
))

print("F1-score:", round(
    f1_score(y_test, test_pred), 4
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, test_proba), 4
))

print("Average Precision:", round(
    average_precision_score(
        y_test, test_proba
    ), 4
))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    test_pred,
    target_names=["No", "Yes"],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, test_pred))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    display_labels=["No", "Yes"],
    cmap="Blues"
)

plt.title("Final Gradient Boosting - Test Set")
plt.tight_layout()
plt.savefig(
    "final_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ==========================================
# 22. SAVE FINAL MODEL
# ==========================================

joblib.dump(
    {
        "model": final_model,
        "threshold": final_threshold,
        "features": features + [
            "was_contacted_before"
        ]
    },
    "bank_marketing_model.joblib"
)

print("\nFinal model saved successfully.")


# ==========================================
# 23. FEATURE IMPORTANCE
# ==========================================

from sklearn.inspection import permutation_importance

importance = permutation_importance(
    final_model,
    X_val,
    y_val,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=2
)

importance_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.round(4).to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()
plt.xlabel("Decrease in Average Precision")
plt.title("Permutation Feature Importance")
plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

