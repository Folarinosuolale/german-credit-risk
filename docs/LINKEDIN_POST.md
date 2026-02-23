I just built an end-to-end credit risk scoring system, and one of the most interesting parts was working with a dataset from 1994.

The German Credit dataset is over 30 years old. 1,000 loan applications, 20 features, collected when fax machines were still standard office equipment. The common reaction in 2026 is to dismiss it as outdated. But here is what I found: the core risk signals in that data are the same ones banks rely on today. Checking account status is still the strongest default predictor. Loan duration still correlates with repayment failure. And models trained on historical data still absorb the biases baked into that history.

Data does not expire. The technology around it evolves, but well-structured data continues to produce meaningful insights regardless of when it was collected.

Here is what the project covers:

I compared four models (Logistic Regression, Random Forest, XGBoost, LightGBM), tuned the best performer with 50 Optuna trials across 10 hyperparameters, and landed on a tuned XGBoost with a 0.799 ROC AUC. I engineered 7 new features from the raw data, including monthly credit burden and financial stability composites, two of which ranked in the top 7 most important features by SHAP analysis.

On the explainability side, every prediction can be broken down into exactly which factors increased or decreased risk, and by how much. SHAP waterfall plots give the kind of per-applicant explanation that regulators require under ECOA and GDPR Article 22.

The fairness audit was revealing. The model produced a 14.8 percentage point gap in approval rates between male and female applicants, failing the EEOC's 4/5ths rule. The root cause: the model was over-predicting female default risk by 11.5 percentage points compared to actual rates, while being nearly calibrated for men. I applied two Fairlearn mitigation strategies (ThresholdOptimizer and ExponentiatedGradient) that brought the model into legal compliance with a manageable accuracy trade-off.

The full system includes an interactive Streamlit dashboard with 6 pages: data exploration, model performance comparison, SHAP explainability, fairness analysis with before/after mitigation results, and live predictions where you can input applicant details and see the risk score with a full SHAP breakdown.

Technical stack: Python, XGBoost, SHAP, Fairlearn, Optuna, SMOTE, Streamlit, Plotly.

Code, dashboard, and full documentation are on my GitHub. Link in comments.

#MachineLearning #CreditRisk #SHAP #XGBoost #ExplainableAI #Fairlearn #DataScience #Finance #Python #Streamlit
