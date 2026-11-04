# Bank Marketing Prediction — Machine Learning Project

### Predictive Analytics | Python · Scikit-learn · Statistical Modeling

## Project Overview

This project explores the application of supervised machine learning to predict whether a bank customer will subscribe to a term deposit following a marketing campaign.

Using the **Bank Marketing Dataset** from the UCI Machine Learning Repository, the analysis combines exploratory data analysis, statistical reasoning, feature engineering, and predictive modeling.

The project aims to develop an interpretable and reproducible machine learning workflow that could support data-driven marketing decisions.

## Business Problem

Marketing campaigns require time and resources, and contacting every customer may not be the most efficient strategy.

A predictive classification model can help identify customers with a higher likelihood of subscribing to a term deposit.

**Main research question:**

*Can historical customer characteristics and previous marketing interactions help predict term deposit subscriptions?*

The target variable is binary:

- **1 — Yes:** The customer subscribed to a term deposit.
- **0 — No:** The customer did not subscribe.

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The complete `bank-full` dataset contains:

- 45,211 observations.
- 16 explanatory variables.
- One binary target variable.

The dataset includes demographic information, financial characteristics, and marketing campaign data.

An important challenge is **class imbalance**, since customers who subscribe represent a minority of observations.

## Technologies

| Category | Tools |
|---|---|
| Programming | Python |
| Data Manipulation | pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | scikit-learn |
| Data Source | UCI Machine Learning Repository |
| Development Environment | Spyder |
| Version Control | Git, GitHub |

## Methodology

### 1. Data Exploration

The initial analysis focuses on understanding the dataset structure and identifying potential data quality issues.

Main activities:

- Importing data with `ucimlrepo`.
- Exploring data types and descriptive statistics.
- Identifying missing values and duplicate records.
- Examining the distribution of the target variable.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis investigates customer characteristics and their relationship with deposit subscriptions.

Visualizations include:

- Target class distribution.
- Customer age distribution.
- Subscription rates by profession.
- Numerical feature correlations.

Conditional subscription rates are used to compare customer groups with different sample sizes.

### 3. Statistical Analysis

Pearson correlation coefficients are used to explore linear associations between numerical features.

For two variables \(X\) and \(Y\), Pearson's coefficient is defined as:

\[
\rho_{X,Y} = \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}
\]

The analysis considers the limitations of correlation, including nonlinearity, confounding factors, and the distinction between correlation and causation.

### 4. Data Preprocessing

The preprocessing workflow is being developed using `scikit-learn`.

The planned transformations include:

- Stratified training and test split (80/20).
- Numerical missing-value imputation.
- Standardization with `StandardScaler`.
- One-Hot Encoding for categorical variables.
- Feature engineering for previous campaign interactions.
- Reproducible preprocessing with `Pipeline` and `ColumnTransformer`.

Transformations are fitted exclusively on training data to prevent data leakage.

### 5. Predictive Modeling — Planned

The following supervised learning models will be implemented and compared:

**Logistic Regression**

A statistical classification baseline offering interpretable coefficients and estimated subscription probabilities.

**Decision Tree**

A nonlinear classification model using recursive feature-based decisions.

**Random Forest**

An ensemble learning algorithm designed to improve predictive robustness by combining multiple decision trees.

### 6. Model Evaluation — Planned

Model performance will be evaluated using:

- Confusion Matrix
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Precision-Recall AUC

Particular attention will be given to the minority class and the business implications of false positives and false negatives.

## Data Leakage Prevention

A central methodological consideration is ensuring that predictions use only information available at the intended decision time.

For example, the variable `duration` represents the duration of the latest phone call. Because this information is not known before the call, it is excluded from the pre-contact prediction model.

Other campaign-related variables are also evaluated according to their availability at prediction time.

This distinction is important for building realistic predictive systems.

## Project Structure

```text
bank-marketing-ml-prediction/
│
├── 01_data_exploration.py
├── 02_exploratory_analysis.py
├── 03_correlation_analysis.py
├── 04_preprocessing.py
│
├── figures/
│   ├── 01_target_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_subscription_by_job.png
│   ├── 04_correlation_matrix.png
│   └── 05_target_correlations.png
│
└── README.md
```

*The structure represents the intended organization of the project. Some preprocessing outputs and visualizations are still being validated.*

## How to Run

1. Install Python and the required dependencies:

   `pip install pandas numpy matplotlib scikit-learn ucimlrepo`

2. Clone or download this repository.

3. Run the scripts in numerical order, starting with:

   `python 01_data_exploration.py`

4. Continue with the exploratory analysis, correlation analysis, and preprocessing scripts.

The dataset is retrieved from the UCI repository through the `ucimlrepo` package.

## Project Status

**Work in progress**

- [x] Dataset exploration
- [x] Initial exploratory data analysis
- [x] Target imbalance investigation
- [x] Correlation analysis implementation
- [ ] Validate correlation analysis and preprocessing outputs
- [ ] Logistic Regression
- [ ] Decision Tree and Random Forest
- [ ] Model comparison and hyperparameter tuning
- [ ] Final business insights and recommendations

## Expected Business Value

Once completed, this project will demonstrate how predictive modeling can support:

- Customer segmentation.
- Identification of promising marketing leads.
- More informed campaign planning.
- Evaluation of marketing decision thresholds.
- Quantitative assessment of classification trade-offs.

The analysis is educational and does not represent a production-ready banking decision system.

## Author

**Urmila Bosisio**

PhD Researcher in Applied Mathematics

Background in numerical analysis, constructive approximation, adaptive algorithms, and scientific computing.

Interested in Data Science, Artificial Intelligence, Machine Learning, and Advanced Analytics.

**GitHub:** https://github.com/urmilabosisio
