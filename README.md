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
| **ROC AUC** | 0.799 |
| **F1 Score** | 0.623 |
| **Precision** | 0.574 |
| **Recall** | 0.683 |
| **Disparate Impact Ratio** | 0.791 (FAIL, below 0.8 threshold) |

The model was then run through **Fairlearn bias mitigation** (ThresholdOptimizer and ExponentiatedGradient), which brought the Disparate Impact Ratio above the 0.80 legal threshold while preserving most of the model's accuracy.

### Top Risk Factors (by SHAP importance)
1. **Checking Account Status** - Strongest predictor; no checking account correlates with lower risk
2. **Loan Purpose** - Education and used car loans show different risk profiles
3. **Savings Account** - Higher savings strongly reduce default probability
4. **Credit History** - Prior payment behavior is a key risk signal
5. **Credit Per Month** - Derived feature; monthly credit burden drives risk

---

## Project Structure

```
credit-risk-scoring/
|-- app/
|   +-- streamlit_app.py          # Interactive dashboard (6 pages)
|-- assets/
|   |-- shap_summary.png          # SHAP beeswarm plot
|   |-- shap_bar.png              # SHAP global importance
|   +-- shap_waterfall.png        # Single prediction explanation
|-- data/
|   |-- german_credit.data        # Raw dataset (UCI)
|   +-- german_credit.doc         # Dataset documentation
|-- docs/
|   |-- PROJECT_REPORT.md         # Comprehensive technical report
|   |-- STAKEHOLDER_BRIEF.md      # Non-technical stakeholder document
|   |-- REPRODUCTION_GUIDE.md     # Step-by-step reproduction guide
|   +-- DASHBOARD_GUIDE.md        # Dashboard reference document
|-- models/
|   |-- best_model.pkl            # Trained XGBoost model
|   |-- artifacts.pkl             # Preprocessing transformers
|   |-- pipeline_results.json     # All metrics and results
|   |-- model_comparison.csv      # Model comparison table
|   |-- feature_importance.csv    # SHAP feature importance
|   |-- roc_data.json             # ROC curve data for all models
|   |-- confusion_matrices.json   # Confusion matrices
|   +-- shap_dict.pkl             # SHAP values and explainer
|-- src/
|   |-- data_loader.py            # Data loading and decoding
|   |-- feature_engineering.py    # Feature creation and preparation
|   |-- model_training.py         # Training, tuning, evaluation, bias mitigation
|   |-- explainability.py         # SHAP analysis
|   +-- run_pipeline.py           # End-to-end pipeline runner
|-- requirements.txt
+-- README.md
```

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

| Model | ROC AUC | CV AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.773 | 0.794 | 0.643 |
| Random Forest | 0.781 | 0.793 | 0.564 |
| XGBoost | 0.774 | 0.770 | 0.607 |
| LightGBM | 0.766 | 0.768 | 0.536 |
| **XGBoost (Tuned)** | **0.799** | -- | **0.623** |

### 5. Hyperparameter Tuning
- **Method:** Optuna Bayesian optimization (50 trials)
- **Search Space:** 10 hyperparameters including learning rate, max depth, regularization, subsampling
- **Objective:** Maximize 5-fold stratified cross-validated ROC AUC
- **Result:** +2.5% AUC improvement over base XGBoost

### 6. Explainability (SHAP)
- TreeExplainer for exact SHAP value computation
- Global importance: Mean |SHAP| across all test samples
- Local explanations: Waterfall plots for individual predictions
- Dependency analysis: Feature interaction effects

### 7. Fairness Analysis
- **Metric:** Disparate Impact Ratio (approval rate ratio between gender groups)
- **Standard:** EEOC 4/5ths rule, ratio must be >= 0.80
- **Result:** Ratio = 0.791, model shows potential gender bias
- **Finding:** Female applicants receive ~14 percentage points lower approval rate

### 8. Bias Mitigation (Fairlearn)
Two mitigation strategies applied and compared:

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

## Future Improvements

- [ ] Add fairness-aware training (adversarial debiasing, reweighting)
- [ ] Implement model monitoring for concept drift detection
- [ ] Add model card generation (Google's Model Cards framework)
- [ ] Deploy to cloud (Streamlit Cloud / AWS)
- [ ] Build CI/CD pipeline with automated retraining
