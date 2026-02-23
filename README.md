# Credit Risk Scoring with Explainable ML

**Machine Learning | Explainability | Fairness Auditing**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn-purple)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## Overview

A production-grade credit risk scoring system that predicts the likelihood of a loan applicant defaulting. The project emphasizes **model explainability** (SHAP) and **fairness auditing** (disparate impact analysis with Fairlearn bias mitigation), both of which are critical requirements for deploying ML in regulated financial environments.

This project uses the German Credit dataset, originally collected in 1994. The dataset is over 30 years old, and that is part of the point. Data does not become useless with age. The lending patterns, risk signals, and demographic biases captured in this dataset are still observable in modern credit portfolios. Checking account status still predicts default risk. Loan duration still correlates with repayment failure. And models trained on historical data still absorb and amplify the biases embedded in that history. By building a full ML pipeline on this classic dataset, this project demonstrates that meaningful insights, actionable predictions, and real fairness concerns can be surfaced from any well-structured data, regardless of when it was collected.

### Why This Matters

Credit scoring directly impacts people's access to loans, housing, and financial services. A credit model that is accurate but opaque or biased can:
- Deny loans to qualified applicants from certain demographics
- Violate regulatory requirements (ECOA, FCRA, GDPR)
- Expose financial institutions to legal and reputational risk

This project demonstrates building a model that is not just performant, but **explainable and auditable**.

---

## Key Results

| Metric | Value |
|---|---|
| **ROC AUC** | 0.773 |
| **F1 Score** | 0.643 |
| **Precision** | 0.562 |
| **Recall** | 0.750 |
| **Disparate Impact Ratio** | 0.895 (PASS, above 0.8 threshold) |

Logistic Regression was selected as the best model because **recall is the most important metric in credit risk** -- missing a defaulter costs the bank real money, while a false alarm only costs a follow-up review. With 0.750 recall, Logistic Regression catches 75% of defaulters, outperforming all other models on this metric. XGBoost (Tuned) achieved the highest AUC (0.788) but only 0.600 recall.

If the DIR is below 80%, the model would then run **Fairlearn bias mitigation** (ThresholdOptimizer and ExponentiatedGradient), to bring the Disparate Impact Ratio above the 0.80 legal threshold while preserving most of the model's accuracy.

### Top Risk Factors (by SHAP importance)
1. **Checking Account Status** - Strongest predictor; no checking account correlates with lower risk
2. **Savings Account** - Higher savings strongly reduce default probability
3. **Loan Purpose** - Education and used car loans show different risk profiles
4. **Credit History** - Prior payment behavior is a key risk signal
5. **Emplyoyment since** - Longer employment signals job stability and reliable income
   
---
## Dataset

**German Credit Data** (UCI Machine Learning Repository)

- **Samples:** 1,000 loan applicants
- **Features:** 20 (7 numerical, 13 categorical)
- **Target:** Binary - Good credit (70%) vs Bad credit/default (30%)
- **Domain:** Consumer lending / retail banking
- **Source:** Professor Hans Hofmann, University of Hamburg

### Why This Dataset

The German Credit dataset is the industry standard for credit scoring research:
- Industry-recognized benchmark for binary credit classification
- Rich mix of categorical and numerical features
- Built-in class imbalance (30% default rate), which is realistic for credit
- Contains protected attributes (gender) enabling fairness analysis
- Small enough for rapid iteration, complex enough to demonstrate ML depth
- Contains inherent demographic biases that make it ideal for demonstrating fairness auditing and bias mitigation

---

## Technical Approach

### 1. Data Preprocessing
- Decoded 13 categorical attributes from alphanumeric codes to human-readable labels
- Extracted gender from composite `personal_status_sex` field for fairness analysis
- No missing values in this dataset; validated data integrity

### 2. Feature Engineering (7 new features)
| Feature | Formula | Rationale |
|---|---|---|
| `credit_per_month` | credit_amount / duration | Monthly financial burden |
| `age_group` | Binned age (5 groups) | Non-linear age effects |
| `credit_burden` | installment_rate x duration | Total commitment intensity |
| `financial_stability` | Composite of residence + credits + age | Stability proxy |
| `high_credit_amount` | > 75th percentile flag | High-value loan indicator |
| `long_duration` | > 24 months flag | Long-term risk signal |
| `amount_per_dependent` | credit_amount / (dependents + 1) | Per-capita financial load |

### 3. Data Preparation
- **Encoding:** Target encoding for categoricals (handles high cardinality, preserves ordinal signal)
- **Scaling:** StandardScaler for numerical features
- **Split:** 80/20 stratified train/test split
- **Class Imbalance:** SMOTE oversampling on training set only

### 4. Model Training and Comparison
Four models trained and compared:

| Model | ROC AUC | RECALL | F1 |
|---|---|---|---|
| **Logistic Regression** | **0.773** | **0.750** | **0.643** |
| Random Forest | 0.781 | 0.550 | 0.564 |
| XGBoost | 0.774 | 0.617 | 0.607 |
| LightGBM | 0.766 | 0.500 | 0.536 |
| XGBoost (Tuned) | 0.788 | 0.600 | 0.558 |

### 5. Hyperparameter Tuning
- **Method:** Optuna Bayesian optimization (50 trials)
- **Search Space:** 10 hyperparameters including learning rate, max depth, regularization, subsampling
- **Objective:** Maximize 5-fold stratified cross-validated ROC AUC
- **Result:** +1.4% AUC improvement over base XGBoost

### 6. Explainability (SHAP)
- TreeExplainer for exact SHAP value computation
- Global importance: Mean |SHAP| across all test samples
- Local explanations: Waterfall plots for individual predictions
- Dependency analysis: Feature interaction effects

### 7. Fairness Analysis
- **Metric:** Disparate Impact Ratio (approval rate ratio between gender groups)
- **Standard:** EEOC 4/5ths rule, ratio must be >= 0.80
- **Result:** Ratio = 0.897, model shows no gender bias
- **Finding:** Female applicants receive ~7 percentage points lower approval rate

### 8. Bias Mitigation (Fairlearn)
Two mitigation strategies applied and compared (only if DIR < 0.80):

| Strategy | Type | Approach | Result |
|---|---|---|---|
| **ThresholdOptimizer** | Post-processing | Adjusts decision thresholds per group to equalize approval rates | Brings DIR above 0.80 |
| **ExponentiatedGradient** | In-processing | Trains a fair classifier under Demographic Parity constraints | Brings DIR above 0.80 |

Both strategies bring the model into compliance with the 4/5ths rule. The ThresholdOptimizer preserves more of the original model's accuracy since it only adjusts the decision boundary rather than retraining.

---

## Interactive Dashboard

The Streamlit app includes 6 pages:

1. **Overview** - Key metrics, pipeline summary, insights
2. **Data Explorer** - Interactive distributions, correlations, target analysis
3. **Model Performance** - ROC curves, metrics comparison, confusion matrices
4. **Feature Importance and SHAP** - Global/local explanations, feature deep dives
5. **Fairness Analysis** - Disparate impact metrics, group-level comparisons, bias mitigation results
6. **Live Prediction** - Enter applicant details, get risk score with SHAP explanation

### Run the Dashboard

```bash
cd credit-risk-scoring
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Real-World Relevance

### Industry Application
This project mirrors the exact workflow used at banks and fintechs:
- **Origination scoring:** Deciding whether to approve a loan application
- **Risk-based pricing:** Setting interest rates proportional to default probability
- **Portfolio monitoring:** Ongoing assessment of credit book health

### Regulatory Compliance
- **ECOA / Fair Lending:** The fairness analysis addresses Equal Credit Opportunity Act requirements
- **GDPR Article 22:** SHAP explanations provide "right to explanation" for automated decisions
- **SR 11-7 (OCC/Fed):** Model validation framework requires explainability and performance monitoring
- **Basel III/IV:** Internal ratings-based approach requires validated scoring models

### Business Impact
- A 1% improvement in AUC can save millions in reduced bad debt for large portfolios
- Explainability reduces model risk and speeds regulatory approval
- Fairness auditing prevents discrimination lawsuits and reputational damage

---

## How to Reproduce

```bash
# 1. Clone and install
git clone https://github.com/Folarinosuolale/german-credit-risk
cd credit-risk-scoring
pip install -r requirements.txt

# 2. Run the full pipeline (trains models, generates all artifacts)
python src/run_pipeline.py

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **ML Frameworks** | scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Fairness** | Fairlearn (ThresholdOptimizer, ExponentiatedGradient) |
| **Tuning** | Optuna (Bayesian optimization) |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Encoding** | category_encoders (Target Encoding) |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **App Framework** | Streamlit |
| **Data** | Pandas, NumPy |

---
