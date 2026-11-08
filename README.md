# Bank Marketing Prediction — End-to-End Machine Learning Project

### Predictive Analytics | Python · Scikit-learn · Statistical Modeling · Gradient Boosting

## Project Overview

This project explores the application of supervised machine learning to predict whether a bank customer will subscribe to a term deposit following a marketing campaign.

Using the **Bank Marketing Dataset** from the UCI Machine Learning Repository, the analysis combines exploratory data analysis, statistical reasoning, feature engineering, predictive modeling, hyperparameter optimization, and model interpretability.

The project implements an **end-to-end machine learning workflow**, from data exploration to final model evaluation on an independent test set.

The final selected model is a **Histogram-based Gradient Boosting Classifier**, achieving:

- **ROC-AUC: 0.7421**
- **Average Precision: 0.3906**
- **Recall: 61.91%**
- **Precision: 24.72%**
- **F1-score: 0.3533**
- **Accuracy: 73.48%**

These results were obtained on an independent test set of **9,043 observations**.

## Business Problem

Marketing campaigns require time and resources, and contacting every customer may not be the most efficient strategy.

A predictive classification model can help identify customers with a higher likelihood of subscribing to a term deposit.

**Main research question:**

*Can historical customer characteristics and previous marketing interactions help predict term deposit subscriptions?*

The target variable is binary:

- **1 — Yes:** The customer subscribed to a term deposit.
- **0 — No:** The customer did not subscribe.

Because positive responses are relatively uncommon, the project prioritizes identifying potential subscribers rather than maximizing accuracy alone.

The selected decision threshold balances precision and recall while emphasizing the detection of customers likely to subscribe.

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The complete `bank-full` dataset contains:

- 45,211 observations.
- 16 explanatory variables.
- One binary target variable.

The dataset includes demographic information, financial characteristics, and marketing campaign data.

An important challenge is **class imbalance**, since approximately 11.7% of customers subscribed to a term deposit.

### Features Used for Modeling

The predictive models use the following features:

| Feature | Description |
|---|---|
| age | Customer age |
| job | Employment category |
| marital | Marital status |
| education | Education level |
| default | Credit default status |
| balance | Average yearly account balance |
| housing | Housing loan indicator |
| loan | Personal loan indicator |
| pdays | Days since the previous campaign contact |
| previous | Number of contacts before the current campaign |
| poutcome | Outcome of the previous marketing campaign |
| was_contacted_before | Engineered indicator of previous contact |

Variables describing the current telephone interaction, such as `duration`, are excluded from the pre-contact predictive model.

## Technologies

| Category | Tools |
|---|---|
| Programming | Python |
| Data Manipulation | pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | scikit-learn |
| Preprocessing | Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder |
| Classification | Logistic Regression, Random Forest, HistGradientBoostingClassifier |
| Model Optimization | RandomizedSearchCV, StratifiedKFold |
| Model Evaluation | ROC-AUC, Average Precision, Precision, Recall, F1-score |
| Model Interpretability | Permutation Feature Importance |
| Model Serialization | joblib |
| Data Source | UCI Machine Learning Repository |
| Development Environment | Spyder |
| Version Control | Git, GitHub |

## Methodology

### 1. Data Exploration

The initial analysis focuses on understanding the dataset structure and identifying potential data quality issues.

Main activities:

- Importing data using `ucimlrepo`.
- Exploring data types and descriptive statistics.
- Identifying missing values and duplicate records.
- Examining the distribution of the target variable.
- Investigating class imbalance.

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
\rho_{X,Y} =
\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
\]

The analysis considers the limitations of correlation, including nonlinearity, confounding factors, and the distinction between correlation and causation.

### 4. Data Preprocessing

A reproducible preprocessing workflow was implemented using `scikit-learn`.

The main transformations include:

- Stratified training and test split (80/20).
- Additional training/validation split.
- Numerical missing-value imputation using the median.
- Categorical missing-value imputation using the most frequent value.
- Standardization with `StandardScaler`.
- One-Hot Encoding for Logistic Regression and Random Forest.
- Ordinal Encoding with explicit categorical feature handling for Histogram-based Gradient Boosting.
- Feature engineering for previous campaign interactions.
- Preprocessing pipelines using `Pipeline` and `ColumnTransformer`.

Preprocessing transformations are fitted on training data to avoid data leakage.

The independent test set is reserved for the final model evaluation.

### 5. Predictive Modeling

Four classification approaches were evaluated.

**Dummy Classifier**

A baseline model predicting the majority class.

It demonstrates why accuracy can be misleading for imbalanced datasets.

**Logistic Regression**

A linear classification model used as a first predictive baseline.

The model was trained with class balancing to improve the detection of the minority class.

**Random Forest**

An ensemble learning algorithm combining multiple decision trees to capture nonlinear relationships and feature interactions.

**Histogram-based Gradient Boosting**

A boosting algorithm using sequential decision trees to improve predictive performance on tabular data.

This model was selected for the final evaluation based on its validation F1-score under a predefined recall constraint.

### 6. Initial Model Comparison

The following results were obtained on the validation set using the default classification threshold of 0.5.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0.8830 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1170 |
| Logistic Regression | 0.7099 | 0.2286 | 0.6229 | 0.3344 | 0.7222 | 0.3556 |
| Random Forest | 0.8393 | 0.3598 | 0.4792 | 0.4110 | 0.7439 | 0.3990 |
| Gradient Boosting | 0.8956 | 0.6939 | 0.1928 | 0.3018 | 0.7509 | 0.4132 |

The initial Gradient Boosting model showed strong ranking performance but low recall at the default threshold.

This motivated further investigation of decision threshold optimization.

The Gradient Boosting categorical feature configuration was subsequently corrected, so its later results are not directly identical to the initial configuration.

### 7. Decision Threshold Optimization

Because the dataset is imbalanced, using the default classification threshold of 0.5 does not necessarily provide the best trade-off between precision and recall.

The decision threshold was optimized on the validation set by maximizing F1-score subject to a minimum recall constraint of approximately 62.29%.

After correcting categorical feature handling, the optimized models produced:

| Model | Threshold | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.1028 | 0.2495 | 0.6276 | 0.3571 | 0.7474 | 0.4047 |
| Random Forest | 0.3680 | 0.2370 | 0.6342 | 0.3451 | 0.7439 | 0.3990 |
| Logistic Regression | 0.4993 | 0.2276 | 0.6238 | 0.3335 | — | — |

Under the recall constraint, Gradient Boosting achieved the highest validation F1-score.

### 8. Hyperparameter Tuning

Hyperparameter optimization was performed using `RandomizedSearchCV` and stratified three-fold cross-validation.

The search evaluated 15 randomly sampled parameter configurations, resulting in 45 cross-validation fits.

The optimization metric was **Average Precision**, which is particularly informative for imbalanced binary classification.

The best parameter configuration found was:

```python
{
    "min_samples_leaf": 20,
    "max_leaf_nodes": 15,
    "max_iter": 250,
    "learning_rate": 0.1,
    "l2_regularization": 5.0
}
```

The best cross-validation Average Precision was:

**0.3719**

The tuned model achieved the following validation results:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7471 |
| Average Precision | 0.4073 |
| Precision | 0.2433 |
| Recall | 0.6295 |
| F1-score | 0.3510 |

Although tuning slightly improved Average Precision compared with the corrected pre-tuning model, it did not improve F1-score under the recall constraint.

Therefore, the corrected pre-tuning Gradient Boosting configuration was retained for the final evaluation.

### 9. Final Model Evaluation

The selected Gradient Boosting model was retrained on the full non-test training dataset.

The previously selected decision threshold of **0.1028** was retained.

Final performance was evaluated on an independent test set of **9,043 observations**.

| Metric | Final Test Result |
|---|---:|
| Accuracy | 0.7348 |
| Precision | 0.2472 |
| Recall | 0.6191 |
| F1-score | 0.3533 |
| ROC-AUC | 0.7421 |
| Average Precision | 0.3906 |

The relatively small differences between validation and test performance suggest that the model maintains reasonably consistent predictive behavior on held-out observations.

However, this random holdout evaluation does not establish performance on future marketing campaigns.

### 10. Confusion Matrix

The final model produced the following confusion matrix:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| **Actual No** | 5,990 | 1,995 |
| **Actual Yes** | 403 | 655 |

The model correctly identified **655 out of 1,058 actual subscribers**, corresponding to a recall of **61.91%**.

However, it also classified 1,995 non-subscribers as potential subscribers.

This highlights the trade-off between detecting potential customers and avoiding unnecessary marketing contacts.

### 11. Model Interpretability

Permutation Feature Importance was used to investigate which variables contributed most strongly to the model's predictive performance.

The analysis measured the average decrease in Average Precision after randomly permuting individual features.

| Feature | Importance |
|---|---:|
| poutcome | 0.1805 |
| age | 0.0576 |
| pdays | 0.0449 |
| housing | 0.0393 |
| balance | 0.0275 |
| previous | 0.0178 |
| job | 0.0160 |
| loan | 0.0096 |
| marital | 0.0084 |
| education | 0.0062 |
| default | 0.0000 |
| was_contacted_before | 0.0000 |

The previous campaign outcome (`poutcome`) was the most influential predictive feature in this experiment.

Customer age, time since previous contact, housing loan status, and account balance also contributed to the predictions.

Permutation importance measures predictive dependence rather than causal influence.

The importance analysis was conducted on observations subsequently included in the final model's training data, so these importance estimates should be interpreted as exploratory rather than independent out-of-sample measurements.

## Data Leakage Prevention

A central methodological consideration is ensuring that predictions use only information available at the intended decision time.

For example, the variable `duration` represents the duration of the latest phone call. Because this information is not known before the call, it is excluded from the pre-contact prediction model.

Other campaign-related variables were also evaluated according to their availability at prediction time.

Preprocessing transformations were included in scikit-learn pipelines to prevent information from the validation or test data from influencing fitted preprocessing parameters.

Model selection and threshold optimization were performed before the final test evaluation.

The test set was not used for hyperparameter optimization or threshold selection.

## Project Structure

```text
bank-marketing-ml-prediction/
│
├── 01_data_exploration.py
├── 02_exploratory_analysis.py
├── 03_correlation_analysis.py
├── 04_preprocessing.py
├── 05_model_training.py
│
├── figures/
│   ├── 01_target_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_subscription_by_job.png
│   ├── 04_correlation_matrix.png
│   └── 05_target_correlations.png
│
├── results/
│   ├── model_comparison.csv
│   ├── optimized_model_comparison.csv
│   ├── gradient_boosting_tuning.csv
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── final_confusion_matrix.png
│
├── bank_marketing_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

The folder structure above represents the recommended repository organization. Generated outputs may need to be moved into the appropriate directories before publication.

## How to Run

1. Install Python and the required dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn ucimlrepo joblib
```

2. Clone or download this repository.

3. Run the scripts in numerical order, starting with:

```bash
python 01_data_exploration.py
```

4. Continue through the preprocessing and model training stages.

5. Execute:

```bash
python 05_model_training.py
```

The training script retrieves the dataset from UCI, performs preprocessing, compares classification models, evaluates the selected model, and generates model evaluation outputs.

For reproducibility, use a consistent Python environment and record the installed package versions.

## Project Status

**Completed — Model Development and Evaluation**

- [x] Dataset exploration
- [x] Exploratory data analysis
- [x] Target imbalance investigation
- [x] Correlation analysis
- [x] Preprocessing pipeline
- [x] Logistic Regression
- [x] Random Forest
- [x] Gradient Boosting
- [x] Model comparison
- [x] Hyperparameter tuning
- [x] Decision threshold optimization
- [x] Final test evaluation
- [x] Exploratory feature importance analysis
- [ ] Repository packaging and reproducibility verification
- [ ] Independent feature importance evaluation
- [ ] Optional deployment and API integration

## Business Insights

The analysis suggests that previous marketing interactions contain important predictive information for identifying potential term deposit subscribers.

The results also demonstrate that accuracy alone is not sufficient when evaluating models on imbalanced datasets.

By prioritizing recall, the model can identify a substantial proportion of actual subscribers, although this comes at the cost of additional false-positive predictions.

The decision threshold should ultimately depend on business factors such as:

- Cost per customer contact.
- Expected revenue from a successful subscription.
- Marketing team capacity.
- Customer engagement policies.
- Costs associated with missed opportunities.

## Limitations and Future Improvements

Despite promising initial results, several limitations remain:

- The dataset represents historical marketing campaigns and may not reflect current customer behavior.
- The evaluation uses a random train/test split rather than a temporal validation strategy.
- The dataset contains imbalanced classes.
- The selected threshold was optimized using validation data and may require recalibration when data distributions change.
- Permutation feature importance was exploratory and should be repeated with an appropriate independent evaluation procedure.
- Predictive relationships should not be interpreted as causal effects.
- Economic optimization requires additional business information.

Potential improvements include cost-sensitive learning, probability calibration, model monitoring, temporal validation, and deployment through a REST API.

## Expected Business Value

This project demonstrates how supervised machine learning can support:

- Customer segmentation.
- Identification of promising marketing leads.
- More informed campaign planning.
- Evaluation of marketing decision thresholds.
- Quantitative assessment of classification trade-offs.

The project is educational and does not represent a production-ready banking decision system.

## Author

**Urmila Bosisio**

PhD Researcher in Applied Mathematics

University of Milan

Background in numerical analysis, constructive approximation, adaptive algorithms, and scientific computing.

Interested in Data Science, Artificial Intelligence, Machine Learning, and Advanced Analytics.

**GitHub:** https://github.com/urmilabosisio
