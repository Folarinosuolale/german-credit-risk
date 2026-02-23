# Credit Risk Scoring - Technical Report

## 1. Executive Summary

This report documents the development of a credit risk scoring model that predicts loan default probability using the German Credit dataset. The project delivers an end-to-end machine learning pipeline with a focus on **model explainability** and **fairness auditing**, both of which are non-negotiable requirements for deploying ML in financial services.

**Key deliverables:**
- Tuned XGBoost classifier achieving **0.799 ROC AUC**
- SHAP-based explainability for global and per-applicant insights
- Fairness audit revealing **gender bias** (Disparate Impact Ratio = 0.791, below the 0.80 threshold)
- Bias mitigation via Fairlearn's ThresholdOptimizer and ExponentiatedGradient, bringing the model into compliance
- Interactive Streamlit dashboard for stakeholder exploration and live predictions

---

## 2. Problem Statement

### Business Context
Financial institutions use credit scoring to make lending decisions: approve or reject applications, set interest rates, and manage portfolio risk. A poorly performing model leads to:
- **False Negatives (Type II):** Approving applicants who default, resulting in direct financial loss
- **False Positives (Type I):** Rejecting creditworthy applicants, resulting in lost revenue and customer dissatisfaction

In regulated markets, models must also be **explainable** (why was someone rejected?) and **fair** (no discrimination against protected groups).

### Objective
Build a binary classification model that:
1. Accurately predicts default risk (maximize AUC and recall for the minority class)
2. Provides human-readable explanations for each decision
3. Passes fairness checks across demographic groups

---

## 3. Data Description

### Source
**German Credit Data** - UCI Machine Learning Repository (Statlog version)
- Contributed by Professor Hans Hofmann, University of Hamburg
- 1,000 samples, 20 input features, 1 binary target

### Target Variable
- **0 (Good Credit):** 700 samples (70%) - applicant did not default
- **1 (Bad Credit):** 300 samples (30%) - applicant defaulted

The 30% default rate creates a moderate class imbalance, requiring careful handling.

### Feature Overview

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | checking_account | Categorical (4) | Status of existing checking account |
| 2 | duration_months | Numerical | Loan duration in months |
| 3 | credit_history | Categorical (5) | Past credit behavior |
| 4 | purpose | Categorical (11) | Reason for the loan |
| 5 | credit_amount | Numerical | Loan amount in DM |
| 6 | savings_account | Categorical (5) | Savings account/bonds balance |
| 7 | employment_since | Categorical (5) | Years at current employment |
| 8 | installment_rate | Numerical | Installment rate as % of income |
| 9 | personal_status_sex | Categorical (5) | Gender and marital status |
| 10 | other_debtors | Categorical (3) | Other debtors/guarantors |
| 11 | residence_since | Numerical | Years at current residence |
| 12 | property | Categorical (4) | Most valuable property owned |
| 13 | age | Numerical | Age in years |
| 14 | other_installment_plans | Categorical (3) | Other installment plans |
| 15 | housing | Categorical (3) | Housing situation |
| 16 | num_existing_credits | Numerical | Number of existing credits |
| 17 | job | Categorical (4) | Employment type |
| 18 | num_dependents | Numerical | Number of dependents |
| 19 | telephone | Categorical (2) | Has registered telephone |
| 20 | foreign_worker | Categorical (2) | Is foreign worker |

### Data Quality
- **Missing values:** 0 (dataset is complete)
- **Duplicates:** None detected
- **Outliers:** Credit amount has right skew (max 18,424 DM, median 2,320 DM); handled implicitly by tree-based models

---

## 4. Exploratory Data Analysis - Key Insights

### Insight 1: Checking Account is the Strongest Signal
Applicants with **no checking account** have a 14.5% default rate vs. 54% for those with an overdrawn account (< 0 DM). This aligns with banking intuition, since checking account status is a proxy for financial discipline and cash flow management.

### Insight 2: Loan Duration Drives Risk
Longer loans (> 24 months) have significantly higher default rates. The relationship is nearly monotonic, making it a strong candidate for feature engineering.

### Insight 3: Purpose Matters
Education and vacation loans show higher default rates than car purchases or furniture. This reflects the collateral effect: secured or tangible purchases tend to carry lower risk.

### Insight 4: Age-Risk Relationship is Non-Linear
Default rate is highest for ages 20-30, decreases through middle age, and increases slightly for 60+. This motivated creating age group bins rather than using raw age.

### Insight 5: Class Imbalance is Moderate
At 70/30 split, the imbalance is not extreme but still biases models toward the majority class. I address this with SMOTE and class-weighted objectives.

---

## 5. Feature Engineering

Seven derived features were created to capture domain-specific risk signals:

### 5.1 Credit Per Month
`credit_amount / duration_months`

**Rationale:** Monthly financial burden is more predictive than total loan amount alone. A 10,000 DM loan over 6 months is very different from the same amount over 48 months.

### 5.2 Age Group
Binned age into 5 groups: Young (<25), Young Adult (25-35), Middle Aged (35-45), Senior (45-60), Elderly (60+)

**Rationale:** Captures the non-linear age-risk relationship discovered in EDA.

### 5.3 Credit Burden
`installment_rate x duration_months`

**Rationale:** Combines payment intensity with commitment length. A high installment rate over many months represents sustained financial pressure.

### 5.4 Financial Stability Score
`residence_since + num_existing_credits + (age / 20).clip(4)`

**Rationale:** Composite proxy for overall financial stability. Longer residence, more credit history, and greater age all correlate with stability.

### 5.5-5.7 Binary Flags
- `high_credit_amount`: Above 75th percentile (captures tail risk)
- `long_duration`: Above 24 months (captures time-horizon risk)
- `amount_per_dependent`: Credit per household member (captures per-capita burden)

---

## 6. Data Preparation Pipeline

### 6.1 Encoding Strategy
**Target Encoding** was chosen over one-hot encoding because:
- Handles high-cardinality features (e.g., `purpose` has 11 categories)
- Preserves ordinal information in the encoding
- Produces a single column per feature (no dimensionality explosion)
- Smoothing parameter (0.3) prevents overfitting to rare categories

### 6.2 Scaling
StandardScaler applied to numerical features for zero mean and unit variance. Required for Logistic Regression; tree models are invariant but benefit during SHAP visualization.

### 6.3 Train/Test Split
80/20 stratified split ensuring equal default rates in both sets (30% in each).

### 6.4 Class Imbalance Handling
**SMOTE (Synthetic Minority Oversampling Technique)** applied to training set only:
- Generates synthetic default cases by interpolating between existing defaults
- Applied **after** splitting to prevent data leakage
- Balances the training set to 50/50 for unbiased learning
- Test set remains at natural 70/30 distribution for realistic evaluation

---

## 7. Model Training and Comparison

### 7.1 Models Evaluated

| Model | Rationale for Inclusion |
|---|---|
| **Logistic Regression** | Baseline; highly interpretable; industry standard in credit scoring |
| **Random Forest** | Ensemble of decision trees; handles non-linearities; robust |
| **XGBoost** | Gradient boosting; state-of-the-art for tabular data; regularized |
| **LightGBM** | Efficient gradient boosting; leaf-wise growth; fast training |

All models used `class_weight='balanced'` or equivalent to account for imbalance even after SMOTE.

### 7.2 Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.750 | 0.563 | 0.750 | 0.643 | 0.773 | 0.794 |
| Random Forest | 0.745 | 0.579 | 0.550 | 0.564 | 0.781 | 0.793 |
| XGBoost | 0.760 | 0.597 | 0.617 | 0.607 | 0.774 | 0.770 |
| LightGBM | 0.740 | 0.577 | 0.500 | 0.536 | 0.766 | 0.768 |
| **XGBoost (Tuned)** | **0.760** | **0.574** | **0.683** | **0.623** | **0.799** | -- |

### 7.3 Analysis
- Logistic Regression performs surprisingly well because credit scoring has strong linear signals
- XGBoost after tuning achieves the best AUC (0.799) and balanced F1 (0.623)
- The tuned model gains +2.5% AUC over base XGBoost through Optuna optimization
- Recall improved significantly (0.617 to 0.683), meaning fewer risky applicants are approved

---

## 8. Hyperparameter Tuning

### 8.1 Optimization Strategy
**Optuna** with TPE (Tree-structured Parzen Estimator) sampler, a Bayesian optimization method that:
- Efficiently explores the hyperparameter space
- Prunes unpromising trials early
- Converges faster than grid or random search

### 8.2 Search Space
| Parameter | Range | Best Value |
|---|---|---|
| n_estimators | 100-500 | 167 |
| max_depth | 3-10 | 10 |
| learning_rate | 0.01-0.3 (log) | 0.061 |
| subsample | 0.6-1.0 | 0.646 |
| colsample_bytree | 0.6-1.0 | 0.641 |
| min_child_weight | 1-10 | 2 |
| gamma | 0-5 | 0.308 |
| reg_alpha | 1e-8 to 10 (log) | 1.003 |
| reg_lambda | 1e-8 to 10 (log) | ~0 |
| scale_pos_weight | 1-5 | 2.593 |

### 8.3 Key Tuning Insights
- **scale_pos_weight of ~2.6:** The model assigns 2.6x weight to the minority (default) class, explaining the recall boost
- **Strong L1 regularization (reg_alpha of ~1):** Drives sparsity, effectively performing built-in feature selection
- **Moderate subsampling (~0.64):** Prevents overfitting by using only 64% of features and samples per tree
- **50 trials** provided sufficient convergence; marginal gains flattened after roughly 35 trials

---

## 9. Model Explainability (SHAP)

### 9.1 Method
**SHAP (SHapley Additive exPlanations)** with TreeExplainer:
- Exact computation of Shapley values for tree-based models
- Polynomial time complexity (fast for XGBoost)
- Provides both global and local explanations

### 9.2 Global Feature Importance (Top 10)

| Rank | Feature | Mean |SHAP| | Interpretation |
|---|---|---|---|
| 1 | checking_account | 0.845 | Most influential, reflects financial health |
| 2 | purpose | 0.386 | Loan purpose determines risk profile |
| 3 | savings_account | 0.376 | Financial cushion reduces risk |
| 4 | credit_history | 0.316 | Past behavior predicts future behavior |
| 5 | credit_per_month | 0.287 | Monthly burden is a key stress indicator |
| 6 | employment_since | 0.272 | Job stability correlates with payment reliability |
| 7 | credit_burden | 0.270 | Engineered feature captures commitment stress |
| 8 | other_installment_plans | 0.269 | Existing obligations increase risk |
| 9 | duration_months | 0.250 | Longer duration means more uncertainty |
| 10 | personal_status_sex | 0.240 | Demographic signal (raises fairness concerns) |

### 9.3 Key SHAP Insights
- **Checking account** dominates with 2x the importance of the next feature. This makes domain sense as it directly reflects cash management.
- Two engineered features (`credit_per_month`, `credit_burden`) rank in the top 7, validating the feature engineering approach.
- `personal_status_sex` appears in top 10. This is a **fairness red flag** as it encodes gender.

---

## 10. Fairness Analysis

### 10.1 Methodology
Computed **Disparate Impact Ratio** (DIR) across gender groups:

DIR = (approval_rate_disadvantaged_group) / (approval_rate_advantaged_group)

The EEOC 4/5ths rule states DIR >= 0.80 for a model to be considered fair.

### 10.2 Results

| Metric | Male | Female |
|---|---|---|
| Sample Count | 139 | 61 |
| Approval Rate | 70.5% | 55.7% |
| Actual Default Rate | 28.8% | 32.8% |
| Predicted Default Rate | 29.5% | 44.3% |
| Average Risk Score | 0.363 | 0.446 |
| Accuracy | 76.3% | 72.1% |
| AUC | 0.809 | 0.781 |

**Disparate Impact Ratio: 0.791 (FAILS 4/5ths rule)**

### 10.3 Analysis
- Female applicants are predicted to default at **44.3%** vs actual rate of **32.8%**, meaning the model over-predicts risk for women
- Male applicants are predicted at **29.5%** vs actual **28.8%**, which is much better calibrated
- The 14.8 percentage point gap in approval rates creates legal and ethical risk
- The gap is partly driven by `personal_status_sex` being in the top 10 features

### 10.4 Bias Mitigation (Fairlearn)

Two mitigation strategies were implemented and compared:

#### Strategy 1: ThresholdOptimizer (Post-processing)
- **Method:** Adjusts the decision threshold independently for each demographic group so that approval rates are equalized
- **Constraint:** Demographic Parity
- **How it works:** Instead of using a single 0.5 threshold for all applicants, the optimizer finds different thresholds per group that minimize overall error while satisfying the fairness constraint
- **Advantage:** Does not require retraining the model. The original XGBoost model's predictions are preserved; only the decision boundary changes.

#### Strategy 2: ExponentiatedGradient (In-processing)
- **Method:** Trains a new classifier (Logistic Regression base) using a constrained optimization approach that incorporates Demographic Parity directly into the learning objective
- **Constraint:** Demographic Parity
- **How it works:** Iteratively adjusts sample weights to reduce the gap in approval rates between groups while maintaining predictive accuracy
- **Advantage:** Produces a model that is inherently fair, rather than applying a post-hoc correction

#### Mitigation Results

Both strategies successfully bring the Disparate Impact Ratio above the 0.80 legal threshold. The ThresholdOptimizer preserves more of the original model's accuracy since it only adjusts the decision boundary. The ExponentiatedGradient achieves stronger fairness guarantees but with a larger accuracy trade-off.

#### Key Insight: The Accuracy-Fairness Trade-off

Correcting bias always comes with some cost to overall accuracy. This is not a flaw in the mitigation method; it reflects the fact that the original model was exploiting biased patterns in the data to boost its predictions. Removing that exploitation necessarily reduces the metric that benefited from it. In regulated industries, this trade-off is not optional. A model that fails the 4/5ths rule cannot be deployed regardless of its accuracy.

---

## 11. Model Card

| Field | Details |
|---|---|
| **Model Name** | Credit Risk Scorer v1.0 |
| **Model Type** | XGBoost Binary Classifier |
| **Training Data** | German Credit (UCI), 800 samples |
| **Evaluation Data** | German Credit (UCI), 200 samples |
| **Input Features** | 27 (20 original + 7 engineered) |
| **Output** | Default probability (0.0 - 1.0) |
| **Primary Metric** | ROC AUC = 0.799 |
| **Intended Use** | Loan application risk assessment |
| **Bias Mitigation** | Fairlearn ThresholdOptimizer and ExponentiatedGradient applied |
| **Limitations** | (1) Small dataset (1,000 rows); (2) Historical data from 1994; (3) German-specific feature encodings |
| **Ethical Considerations** | Unmitigated model fails the 4/5ths fairness rule for gender. Mitigation strategies bring the model into compliance. Production deployment should use the mitigated version with ongoing fairness monitoring. |

---

## 12. Conclusions

1. **XGBoost** with Optuna tuning delivers the best performance (AUC 0.799), confirming gradient boosting's strength on tabular credit data.
2. **SHAP explainability** reveals that checking account status is the dominant risk factor, a finding that aligns with domain expertise and builds stakeholder trust.
3. **Feature engineering** adds measurable value. Two engineered features rank in the top 7 by importance.
4. **Fairness analysis** exposes gender bias that would violate EEOC guidelines. This is a critical finding that would block production deployment and requires mitigation.
5. **Bias mitigation** with Fairlearn (ThresholdOptimizer and ExponentiatedGradient) successfully brings the model into compliance, demonstrating that the accuracy-fairness trade-off is manageable.
6. The **interactive dashboard** enables non-technical stakeholders to explore the model, understand decisions, and audit for bias.

### Business Recommendations
- Deploy with **threshold calibration** per demographic group to achieve fairness
- Implement **model monitoring** for drift detection (both performance and fairness metrics)
- Use **SHAP waterfall plots** in adverse action notices to satisfy regulatory explainability requirements
- Retrain periodically on updated data to capture evolving credit patterns
