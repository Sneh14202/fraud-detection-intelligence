# Fraud Detection & Transaction Risk Intelligence Platform

An end-to-end machine learning project for detecting fraudulent credit card transactions, optimizing fraud-investigation decisions under severe class imbalance, explaining model predictions, and deploying an interactive transaction-risk scoring application.

---

## Project Overview

Fraud detection is a highly imbalanced classification problem because fraudulent transactions represent only a very small fraction of all transactions.

This project develops a complete fraud-risk intelligence workflow rather than treating fraud detection as a simple binary classification problem.

The project covers:

* Data understanding and quality assessment
* Exploratory data analysis
* Severe class-imbalance analysis
* Feature engineering
* Baseline classification
* Multiple model comparison
* Hyperparameter tuning
* Imbalance-aware modeling
* Precision-recall analysis
* Decision-threshold optimization
* Probability calibration
* Final holdout validation
* SHAP explainability
* Transaction risk segmentation
* Fraud-investigation prioritization
* Business-value analysis
* Automated testing
* Streamlit deployment

The final application allows a user to enter or select a transaction and receive:

1. Estimated fraud probability
2. Risk segment
3. Investigation priority
4. Dataset ground truth for example transactions
5. SHAP-based explanation of the prediction

---

## Business Problem

Traditional classification accuracy is not sufficient for fraud detection.

Because legitimate transactions vastly outnumber fraudulent transactions, a model could achieve high accuracy while failing to identify a meaningful number of fraudulent transactions.

The objective of this project is to build a system that can:

* Identify suspicious transactions
* Evaluate model performance using metrics appropriate for imbalanced classification
* Adjust the classification threshold according to operational objectives
* Produce calibrated fraud probabilities
* Prioritize transactions for investigation
* Explain individual model predictions
* Translate model predictions into an operational risk workflow

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from the ULB Machine Learning Group.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

* 284,807 transactions
* 30 input features
* 1 binary target variable
* 492 fraudulent transactions

The target variable is:

```text
Class
```

where:

```text
0 = Legitimate transaction
1 = Fraudulent transaction
```

The fraud rate is approximately:

```text
0.172%
```

This corresponds to an extremely imbalanced classification problem.

The original dataset contains:

```text
Time
V1 ... V28
Amount
Class
```

The `V1` through `V28` variables are anonymized PCA-derived features.

`Time` represents elapsed time, while `Amount` represents the transaction amount.

---

## Project Architecture

```text
fraud-detection-intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_model_tuning.ipynb
│   ├── 08_imbalance_strategy.ipynb
│   ├── 09_threshold_optimization.ipynb
│   ├── 10_calibration.ipynb
│   ├── 11_final_validation.ipynb
│   ├── 12_shap_explainability.ipynb
│   ├── 13_risk_segmentation.ipynb
│   ├── 14_business_intelligence.ipynb
│   ├── 15_final_report.ipynb
│   └── 17_test_src_modules.ipynb
│
├── outputs/
│
├── reports/
│   ├── figures/
│   └── executive_summary.txt
│
├── src/
│   ├── feature_engineering.py
│   ├── risk_scoring.py
│   └── model_utils.py
│
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_risk_scoring.py
│   └── test_model_utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Machine Learning Workflow

```text
Raw Data
    ↓
Data Understanding
    ↓
Data Quality
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Baseline Model
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
Imbalance Strategy
    ↓
Threshold Optimization
    ↓
Probability Calibration
    ↓
Final Validation
    ↓
SHAP Explainability
    ↓
Risk Segmentation
    ↓
Business Decisioning
    ↓
Streamlit Application
```

---

## 1. Data Understanding

The first notebook establishes the structure of the dataset.

Key checks include:

* Dataset dimensions
* Data types
* Target distribution
* Number of fraud transactions
* Number of legitimate transactions
* Feature uniqueness
* Descriptive statistics
* Class imbalance ratio

The dataset contains 284,807 observations and only 492 fraud cases.

The approximate fraud-to-legitimate imbalance is:

```text
1 : 578
```

This makes accuracy an inadequate standalone metric.

---

## 2. Data Quality

The data-quality workflow checks for:

* Missing values
* Non-numeric columns
* Duplicate records
* Infinite values
* Constant features
* Near-constant features
* Invalid target values
* Negative transaction amounts
* Negative elapsed time
* Feature ranges

The goal is to identify potential data-quality problems before model development.

---

## 3. Exploratory Data Analysis

The EDA workflow investigates differences between fraudulent and legitimate transactions.

Analysis includes:

* Class distribution
* Transaction amount distributions
* Log-transformed transaction amounts
* Fraud rate by transaction-amount decile
* Fraud activity over time
* Time-based fraud patterns
* Feature-level mean comparisons
* Standardized feature differences
* Correlations with the fraud target
* Correlation heatmap

Generated figures include:

```text
reports/figures/class_distribution.png
reports/figures/fraud_rate_by_amount_decile.png
reports/figures/fraud_rate_over_time.png
```

---

## 4. Feature Engineering

Several features were created to provide the models with additional representations of transaction timing and amount.

### Amount Log Transformation

```python
Amount_Log = log1p(Amount)
```

This provides a compressed representation of highly skewed transaction amounts.

### Time in Hours

```python
Time_Hours = Time / 3600
```

### Cyclical Time Encoding

Transaction time was represented using sine and cosine transformations:

```python
Hour_Sin = sin(2π × Hour_Of_Day / 24)

Hour_Cos = cos(2π × Hour_Of_Day / 24)
```

The final modeling dataset contains:

```text
34 model features
```

---

## 5. Baseline Model

The baseline model is a Logistic Regression pipeline using:

```text
StandardScaler
        +
LogisticRegression
```

The dataset was split using a stratified train/test split.

The baseline establishes a reference point before comparing more complex models.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

For this project, particular attention is given to:

```text
PR-AUC
Recall
Precision
F1-score
```

because of the severe class imbalance.

---

## 6. Model Comparison

Three primary model families were compared:

### Logistic Regression

Provides a relatively simple linear baseline.

### Random Forest

Captures nonlinear relationships and interactions through an ensemble of decision trees.

### Gradient Boosting

Builds sequential decision trees to improve predictive performance.

The comparison workflow evaluates models using:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC

The complete comparison results are stored in:

```text
outputs/model_comparison.csv
```

---

## 7. Hyperparameter Tuning

RandomizedSearchCV with stratified cross-validation was used to tune the candidate models.

The primary optimization metric was:

```text
Average Precision
```

The tuning process explored parameters including:

### Logistic Regression

* Regularization strength
* L1/L2 penalty
* Solver

### Random Forest

* Number of estimators
* Maximum depth
* Minimum samples per split
* Minimum samples per leaf
* Maximum features

### Gradient Boosting

* Number of estimators
* Learning rate
* Maximum depth
* Minimum samples per split
* Minimum samples per leaf
* Subsample

The tuning results are stored in:

```text
outputs/tuned_model_comparison.csv
```

---

## 8. Imbalance Strategy

The project evaluates class-weighted approaches for handling the severe imbalance.

Class-weighted versions were evaluated for:

* Logistic Regression
* Random Forest

The objective is to give greater importance to the minority fraud class during model training.

SMOTE was not used in this project.

Results are stored in:

```text
outputs/imbalance_strategy_results.csv
outputs/model_comparison_with_imbalance.csv
```

---

## 9. Threshold Optimization

A probability of 0.50 is not necessarily an appropriate fraud-investigation threshold.

Therefore, thresholds from:

```text
0.01 to 0.50
```

were evaluated.

For each threshold, the project calculates:

* Precision
* Recall
* F1-score
* True positives
* False positives
* False negatives
* True negatives
* Number of flagged transactions

This demonstrates an important fraud-detection concept:

> The model estimates risk; the threshold determines the operational decision.

Results are stored in:

```text
outputs/gradient_boosting_threshold_analysis.csv
outputs/logistic_threshold_analysis.csv
outputs/random_forest_threshold_analysis.csv
outputs/best_f1_threshold_comparison.csv
outputs/practical_threshold_table.csv
```

---

## 10. Probability Calibration

The selected Gradient Boosting model was calibrated using:

```python
CalibratedClassifierCV(
    estimator=tuned_gb,
    method="sigmoid",
    cv=3
)
```

Calibration is important because a fraud-risk application needs meaningful probabilities rather than only ranking transactions.

For example, a probability of:

```text
0.80
```

should represent substantially greater estimated risk than:

```text
0.10
```

Calibration performance was evaluated using:

* Brier score
* Log loss
* ROC-AUC
* PR-AUC

Calibration outputs include:

```text
outputs/calibration_comparison.csv
outputs/calibrated_probability_percentiles.csv
outputs/calibrated_risk_counts.csv
outputs/calibration_summary.csv
```

---

## 11. Final Validation Strategy

A fresh:

```text
70% Development
15% Validation
15% Test
```

split was used with:

```text
random_state = 123
```

The workflow was:

```text
Development data
        ↓
Train tuned Gradient Boosting
        ↓
Calibrate model
        ↓
Validation data
        ↓
Select probability threshold
        ↓
Test data
        ↓
Final evaluation
```

The test set was used only for the final evaluation.

This separation helps reduce the risk of selecting a model or threshold based on the final test results.

Final outputs:

```text
outputs/final_metrics.csv
outputs/final_threshold_analysis.csv
outputs/final_validation_split_summary.csv
outputs/final_model_configuration.csv
```

The final model is stored as:

```text
models/final_fraud_model.joblib
```

---

## 12. SHAP Explainability

SHAP was used to explain model predictions.

The project provides two levels of explanation.

### Global Explanation

The project calculates mean absolute SHAP values to identify features that have the largest average contribution to model predictions.

Output:

```text
outputs/shap_global_feature_importance.csv
```

Visualization:

```text
reports/figures/shap_summary_plot.png
```

### Individual Transaction Explanation

A specific transaction can also be explained using its individual SHAP values.

Output:

```text
outputs/shap_transaction_explanation.csv
```

Visualization:

```text
reports/figures/shap_transaction_explanation.png
```

Important limitation:

The `V1` through `V28` variables are anonymized PCA-derived features. Therefore, their individual SHAP contributions should not be interpreted as direct business concepts such as "suspicious merchant" or "unusual location."

SHAP explains how the model used the available features. It does not establish causality.

---

## 13. Risk Segmentation

The calibrated fraud probabilities are converted into operational risk categories.

The project uses the following thresholds:

| Fraud Probability | Risk Segment |
| ----------------: | ------------ |
|              < 1% | Very Low     |
|          1% – <5% | Low          |
|         5% – <10% | Medium       |
|        10% – <50% | High         |
|              ≥50% | Critical     |

These thresholds are **project-defined operational categories**, not industry-standard fraud classifications.

---

## 14. Investigation Priority

Each risk segment is mapped to an investigation priority:

| Risk Segment | Priority |
| ------------ | -------: |
| Very Low     |        5 |
| Low          |        4 |
| Medium       |        3 |
| High         |        2 |
| Critical     |        1 |

A lower priority number represents greater urgency within this project's operational framework.

The purpose is to transform model probabilities into an actionable investigation queue.

Outputs include:

```text
outputs/risk_segment_summary.csv
outputs/fraud_investigation_queue.csv
outputs/top_100_investigation_queue.csv
outputs/top_n_investigation_analysis.csv
outputs/fraud_risk_scored_transactions.csv
```

---

## 15. Business Intelligence

The project includes a hypothetical cost-sensitive decision framework.

The analysis assumes:

```text
Investigation cost = $25
Fraud loss avoided per true positive = $500
Missed fraud cost per false negative = $500
```

The hypothetical net-value calculation is:

```text
Net Value =
    TP × Loss Avoided
    - (TP + FP) × Investigation Cost
    - FN × Missed Fraud Cost
```

These values are **illustrative assumptions for portfolio analysis**.

They are not estimates of actual financial losses or investigation costs for a real organization.

The analysis compares thresholds according to this hypothetical business-value framework.

Outputs include:

```text
outputs/business_threshold_analysis.csv
outputs/business_vs_f1_threshold_comparison.csv
outputs/business_value_summary.csv
outputs/executive_threshold_table.csv
outputs/business_assumptions.csv
```

---

## 16. Executive Reporting

The project consolidates major analytical results into executive-oriented outputs.

Generated files include:

```text
outputs/executive_kpis.csv
outputs/executive_shap_drivers.csv
outputs/executive_risk_segments.csv
outputs/project_summary.csv
outputs/consolidated_final_report.csv
reports/executive_summary.txt
```

These provide a compact view of:

* Model performance
* Risk distribution
* Investigation priorities
* Major model drivers
* Business-value analysis
* Project configuration

---

## 17. Streamlit Application

The project includes an interactive Streamlit application.

Run it using:

```bash
streamlit run app/app.py
```

The application supports three analysis modes:

### Example Fraud Transaction

Loads an actual fraud transaction from the processed dataset.

### Example Legitimate Transaction

Loads an actual legitimate transaction from the processed dataset.

### Manual Input

Allows a user to enter:

* Time
* Transaction Amount
* V1–V28

The application then performs the same feature engineering and model prediction workflow.

### Application Output

For each transaction, the application displays:

* Fraud probability
* Risk segment
* Investigation priority
* Dataset ground truth for example transactions
* SHAP explanation

---

## 18. Reusable Python Modules

The project separates reusable ML logic from the Streamlit presentation layer.

### `src/feature_engineering.py`

Contains:

```python
engineer_features()
```

Responsible for creating:

* Amount_Log
* Time_Hours
* Hour_Sin
* Hour_Cos

### `src/risk_scoring.py`

Contains:

```python
assign_risk_segment()
assign_priority()
```

Responsible for converting probabilities into operational risk categories.

### `src/model_utils.py`

Contains:

```python
load_model()
predict_fraud_probability()
```

Responsible for loading the model and generating fraud probabilities.

This structure prevents the same logic from being duplicated across notebooks and applications.

---

## 19. Automated Testing

The project includes pytest-based tests.

Tests cover:

### Feature Engineering

* Correct output dimensions
* Expected feature columns
* Engineered Amount_Log feature

### Risk Scoring

* Risk-category assignment
* Priority mapping
* Boundary conditions

### Model Utilities

* Probability extraction from a model

Run the tests with:

```bash
python -m pytest tests -v
```

---

## 20. Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd fraud-detection-intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## 21. Running the Project

### Run the Streamlit application

```bash
streamlit run app/app.py
```

### Run automated tests

```bash
python -m pytest tests -v
```

### Explore the notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in numerical order.

---

## 22. Reproducibility

The project uses:

```text
random_state = 42
```

for many model-development operations.

The final validation workflow uses:

```text
random_state = 123
```

for the final 70/15/15 development-validation-test split.

The final test set is kept separate from model and threshold selection.

---

## 23. Limitations

### Anonymized Features

The `V1`–`V28` variables do not have human-readable business meanings.

### Historical Dataset

The dataset represents a historical fraud-detection setting and may not represent current transaction behavior.

### Hypothetical Business Costs

The business-value analysis uses illustrative assumptions rather than organization-specific financial data.

### Threshold Stability

The appropriate operational threshold can change as:

* Fraud prevalence changes
* Investigation capacity changes
* Fraud losses change
* Customer behavior changes
* Model performance changes

### Model Drift

A production fraud system would require ongoing monitoring for:

* Data drift
* Concept drift
* Fraud-pattern changes
* Calibration degradation
* Precision/recall changes

---

## 24. Future Improvements

Potential production-level extensions include:

* Real-time transaction scoring API
* Model monitoring
* Data-drift detection
* Automated model retraining
* Feature-store integration
* Investigation feedback loops
* Cost-sensitive learning
* Advanced ensemble models
* Time-aware validation
* Model registry
* Docker deployment
* CI/CD pipeline
* Cloud deployment
* Database integration
* Real-time fraud alerts
* Investigator feedback tracking

---

## 25. Key Skills Demonstrated

### Python

* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn
* joblib
* SHAP
* Streamlit
* pytest

### Machine Learning

* Logistic Regression
* Random Forest
* Gradient Boosting
* Hyperparameter tuning
* Cross-validation
* Imbalanced classification
* Probability calibration
* Threshold optimization

### Data Science

* Exploratory data analysis
* Feature engineering
* Classification metrics
* Precision-recall analysis
* Model validation
* Explainable AI

### Business Analytics

* Cost-sensitive decisioning
* Risk segmentation
* Investigation prioritization
* Business-value modeling
* Executive reporting

### Software Engineering

* Modular Python code
* Reusable source modules
* Automated testing
* Project structure
* Model serialization
* Streamlit application development

---

## 26. Interview Discussion Points

### Why isn't accuracy enough?

Because fraud represents only a tiny fraction of the transactions. A model could classify nearly everything as legitimate and still obtain high accuracy.

### Why use PR-AUC?

Precision-recall analysis focuses more directly on performance for the minority positive class.

### Why optimize the threshold?

The model produces probabilities, while the business needs decisions. Changing the threshold changes the tradeoff between false positives and false negatives.

### Why calibrate probabilities?

A risk-scoring application benefits from probabilities that better represent estimated event likelihood rather than only producing a ranking.

### Why use SHAP?

SHAP provides a framework for explaining how individual features contributed to a model prediction.

### Why separate validation and test data?

The validation set is used for model and threshold decisions, while the test set provides a final evaluation after those decisions.

### Why create risk segments?

A probability alone may not be operationally convenient. Risk bands allow transactions to be grouped into actionable categories.

### Why create reusable modules?

Separating feature engineering, risk scoring, and prediction logic makes the project easier to test, maintain, reuse, and deploy.

---

## 27. Project Outcome

This project demonstrates an end-to-end machine learning workflow for an extremely imbalanced fraud-detection problem.

Rather than stopping at model training, the project extends through:

```text
Prediction
    ↓
Probability Calibration
    ↓
Threshold Optimization
    ↓
Risk Segmentation
    ↓
Investigation Prioritization
    ↓
Explainability
    ↓
Business Decisioning
    ↓
Deployment
    ↓
Automated Testing
```

The result is a complete **Fraud Detection & Transaction Risk Intelligence Platform** demonstrating practical data science, machine learning, explainability, business analytics, and deployment skills.

---

## Disclaimer

This project is an educational and portfolio implementation.

The business-value assumptions, risk thresholds, and operational categories are illustrative and should not be interpreted as production fraud-management policies or financial estimates.
