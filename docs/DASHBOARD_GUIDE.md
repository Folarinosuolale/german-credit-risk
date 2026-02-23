# Dashboard Insights and Values Guide

A complete reference for every metric, chart, and value displayed across the six tabs of the Credit Risk Scoring Dashboard. Written so that anyone, regardless of technical background, can understand what they are seeing and what it means.

---

## Tab 1: Overview

### The Five Headline Metrics

These five numbers sit at the top of the dashboard and summarize the model's overall performance at predicting which loan applicants will default.

**ROC AUC: 0.799**
This is the single most important number on the dashboard. It measures how well the model can tell the difference between someone who will repay and someone who will default. The scale goes from 0.5 (no better than flipping a coin) to 1.0 (perfect prediction every time). Our score of 0.799 means that if you randomly pick one person who defaulted and one who did not, the model will correctly identify which is which about 80% of the time. In the banking industry, scores between 0.70 and 0.85 are considered strong performers.

**F1 Score: 0.623**
This number captures the balance between two competing goals: catching defaulters (recall) and not falsely accusing good borrowers (precision). It is the harmonic mean of those two metrics. A perfect score is 1.0. Our 0.623 means the model has a reasonable but not perfect balance, which is typical for credit risk problems where the data is imbalanced (far more good borrowers than bad ones).

**Precision: 0.574**
When the model flags someone as likely to default, it is correct 57.4% of the time. The other 42.6% are false alarms, meaning good borrowers who were unnecessarily flagged. This number matters because every false alarm is a qualified applicant who might get rejected or face unnecessary scrutiny.

**Recall: 0.683**
Of all the people who actually defaulted in the test set, the model successfully identified 68.3% of them. The remaining 31.7% slipped through as "predicted good" when they were actually bad. This is the more costly type of error in banking because each missed defaulter results in a direct financial loss.

**Avg Precision: 0.640**
This metric summarizes how well the model holds up across all possible decision thresholds. It is particularly useful for imbalanced datasets like ours (70% good, 30% bad). A higher number means the model maintains good precision even as it tries to catch more and more defaulters.

### Key Insights Panel

**Best Model: XGBoost (Tuned)** - Out of the four algorithms tested, XGBoost with hyperparameter tuning performed best. Tuning means I systematically tested 50 different configurations to find the one that produces the most accurate predictions.

**Fairness Alert: Disparate Impact Ratio = 0.791** - This flags that the model treats male and female applicants differently. The legal threshold is 0.800. Our model falls just below at 0.791, meaning female applicants are being approved at a rate that is less than 80% of the male approval rate. This would need to be fixed before the model could be used in production.

**Top Risk Factor: Checking Account Status** - Among all 27 features the model uses, the status of the applicant's checking account has the most influence on the prediction. Applicants with no checking account actually default at lower rates (about 14.5%) than those with overdrawn accounts (about 54%).

### Project Pipeline Table

This table shows the seven stages of the machine learning pipeline used to build this model:

- **Data**: 1,000 historical loan applications with 20 attributes each
- **Preprocessing**: Converting text categories into numbers the model can read, standardizing scales, and generating synthetic samples to balance the dataset
- **Feature Engineering**: Creating 7 new calculated fields from the raw data (like monthly payment amount and financial stability score)
- **Models Compared**: Four different algorithms were trained and tested
- **Tuning**: The best algorithm (XGBoost) was fine-tuned by testing 50 different parameter combinations
- **Explainability**: SHAP analysis was run to explain every prediction the model makes
- **Fairness**: The model was audited for bias across gender groups

---

## Tab 2: Data Explorer

### Distributions Sub-tab

**Numerical Feature Histogram (left side)**
Shows how applicants are spread across different values of the selected feature. Blue bars represent borrowers who repaid. Red bars represent borrowers who defaulted. Where the two colors overlap heavily, the feature does not help distinguish between good and bad borrowers. Where they separate, the feature carries useful signal.

Key observations by feature:
- **age**: Defaulters skew younger (average age 33.9) compared to non-defaulters (average 36.2)
- **credit_amount**: Defaulters borrow more on average. The distribution for defaults has a longer right tail
- **duration_months**: Defaulters have longer loan terms on average (24.8 months vs 19.2 months)
- **installment_rate**: Fairly similar across both groups, meaning it is a weaker standalone predictor

**Categorical Feature Bar Chart (right side)**
Shows the count of good and bad borrowers within each category of the selected feature. Categories where the red bars are proportionally large relative to blue bars are higher-risk categories.

Key observations by feature:
- **checking_account**: "< 0 DM" (overdrawn) has the highest proportion of defaults. "No checking account" has the lowest.
- **purpose**: "Education" and "Others" categories have higher default proportions than "Car (new)" or "Furniture/equipment"
- **savings_account**: "Unknown / no savings" has a high default count, while ">= 1000 DM" savings has very few defaults

### Correlations Sub-tab

**Correlation Heatmap**
Each cell shows how strongly two numerical features move together, on a scale from -1.0 to +1.0:
- **+1.0 (dark red)**: Both features increase together perfectly
- **0 (white)**: No linear relationship
- **-1.0 (dark blue)**: One increases as the other decreases

In this dataset, no pair of features is highly correlated (all values are below 0.8 in absolute terms), which means each feature contributes unique information to the model. The strongest relationship is between credit_amount and duration_months (around 0.62), which makes intuitive sense: larger loans tend to be repaid over longer periods.

### Target Analysis Sub-tab

**Default Rate by Feature (dual-axis chart)**
Red bars (left axis) show the default rate (percentage who failed to repay) for each category. Blue line (right axis) shows the sample count in each group.

This dual view is critical because default rates in very small groups can be unreliable. For example, if a category has only 5 people and 3 defaulted, the 60% rate looks alarming, but it is based on too few observations to be statistically meaningful. Always check the blue line alongside the red bars.

Key findings:
- **checking_account**: The default rate varies from about 14% (no checking account) to 54% (overdrawn), a spread of 40 percentage points. This is the strongest single predictor in the dataset.
- **credit_history**: "Critical account" paradoxically shows lower default rates. This is because in the German Credit dataset, "critical" means the applicant has other existing credits, which actually indicates an established credit track record.
- **gender**: Male applicants default at about 29%, female applicants at about 33%. The gap is moderate.

**Dataset Summary Metrics**
- **Total Samples (1,000)**: The number of loan applications in the dataset
- **Default Rate (30.0%)**: Three in ten applicants defaulted on their loans
- **Features (21)**: The number of data columns (excluding the target variable)

**Descriptive Statistics Table**
Each row represents a statistical property:
- **count**: Number of non-missing values (1,000 for all, confirming no missing data)
- **mean**: The average value
- **std**: Standard deviation, measuring how spread out values are
- **min / max**: The smallest and largest observed values
- **25% / 50% / 75%**: The values at the 25th, 50th (median), and 75th percentiles

---

## Tab 3: Model Performance

### ROC Curves Sub-tab

**What the chart shows**: Five curves, one per model, plotting the trade-off between catching defaulters (y-axis: True Positive Rate) and falsely flagging good borrowers (x-axis: False Positive Rate).

**How to read it**: A curve that bows toward the top-left corner is a better model. The gray dashed diagonal line represents a model that is no better than random guessing. The further above that line a curve sits, the better the model performs.

**The AUC numbers in the legend**: Each model's curve is labeled with its AUC (Area Under the Curve). This single number summarizes the entire curve into one value between 0.5 and 1.0.

Model results:
- **XGBoost (Tuned): 0.7989** - Best overall. The thicker blue line.
- **Random Forest: 0.7814** - Second best. Good at ranking risk but less balanced.
- **Logistic Regression: 0.7727** - Surprisingly competitive for a simpler model.
- **XGBoost (base): 0.7738** - Before tuning, XGBoost performs similarly to the others.
- **LightGBM: 0.7660** - Lowest AUC in this comparison, though still well above random.

### Metrics Comparison Sub-tab

**Comparison Table**
Each row is a model, each column is a metric. Green-highlighted cells mark the best score in each column. Red-highlighted cells mark the worst.

Important observation: No single model wins every metric. This is common. Logistic Regression has the best recall (0.750) but worst precision (0.563), meaning it catches the most defaulters but also creates the most false alarms. XGBoost (Tuned) has the best AUC (0.799) and best overall balance.

**Bar Chart**
Visual comparison of four key metrics (AUC, F1, Precision, Recall) across all models. Makes it easy to see at a glance where each model is strong and weak.

### Confusion Matrices Sub-tab

**The 2x2 Grid**
For the selected model, this shows exactly how its 200 test predictions broke down:

| | Predicted Good | Predicted Bad |
|---|---|---|
| **Actually Good** | True Negatives (correct approvals) | False Positives (unnecessary rejections) |
| **Actually Bad** | False Negatives (missed defaults) | True Positives (correctly caught defaults) |

For XGBoost (Tuned):
- **True Negatives: ~114** good borrowers correctly identified as good
- **False Positives: ~26** good borrowers incorrectly flagged as risky
- **False Negatives: ~19** defaulters that the model missed and would have approved
- **True Positives: ~41** defaulters correctly identified

The business impact framing: Each false negative (missed defaulter) represents a potential loss equal to the full loan amount. Each false positive (rejected good borrower) represents lost interest income. In banking, a missed defaulter typically costs 5 to 10 times more than a rejected good borrower, which is why recall is prioritized.

---

## Tab 4: Feature Importance & SHAP

### Global Importance Sub-tab

**Horizontal Bar Chart**
Features sorted from most influential (top) to least influential (bottom). Bar length represents the mean absolute SHAP value, which measures how much that feature changes the model's prediction on average across all test applicants.

Top features and what they tell us:
1. **checking_account (0.845)**: Dominates all other features with more than 2x the impact. An overdrawn checking account dramatically increases predicted risk, while having no checking account decreases it.
2. **purpose (0.386)**: The reason for borrowing significantly shapes the risk prediction. Tangible purchases (cars, furniture) are lower risk than intangible ones (education, vacation).
3. **savings_account (0.376)**: A strong savings balance provides a safety net that the model recognizes as risk-reducing.
4. **credit_history (0.316)**: Past repayment behavior is a strong predictor of future behavior.
5. **credit_per_month (0.287)**: An engineered feature (loan amount divided by duration). This measures monthly payment burden, which turned out to be a better predictor than either raw amount or raw duration alone.
6. **employment_since (0.272)**: Longer employment signals job stability and reliable income.
7. **credit_burden (0.270)**: Another engineered feature (installment rate times duration). Measures sustained financial commitment pressure.

The **error bars** on each bar show variability. A long error bar means the feature matters a lot for some applicants but very little for others. For example, checking_account has a long error bar because it dramatically affects predictions for applicants with overdrawn accounts but has less impact for those in good standing.

### SHAP Plots Sub-tab

**Beeswarm Plot (left)**
Each dot is one applicant. Dots to the right of center mean the feature pushed that applicant's prediction toward default. Dots to the left mean it pushed toward good credit. The color indicates the feature's actual value for that applicant (red = high, blue = low).

For example, for "duration_months" you can see red dots (high duration values) clustered on the right (toward default), confirming that longer loans are riskier.

**Bar Plot (right)**
A simpler aggregate view showing the average impact of each feature, without the directional detail.

**Waterfall Plot (bottom)**
This shows the complete story of one specific applicant's prediction. Starting from the base value (the model's average output if it knew nothing about the applicant), each bar shows how one feature shifted the prediction up (toward default, in red) or down (toward good credit, in blue). The bars accumulate to produce the final prediction at the top.

This type of visualization is used in regulatory compliance to explain individual loan decisions. If an applicant asks "Why was I rejected?", the waterfall plot provides the answer.

### Feature Deep Dive Sub-tab

**Histogram (left)**: Shows the distribution of SHAP values for the selected feature across all test applicants. A spread of values far from zero means the feature has a strong and variable effect.

**Scatter Plot (right)**: Each dot is one applicant, with the y-axis showing the SHAP value for the selected feature. Red dots indicate the feature increased risk for that applicant; blue dots indicate it decreased risk.

**Three metric cards below the charts**:
- **Mean |SHAP|**: Average absolute impact of this feature across all applicants
- **Max SHAP (risk-increasing)**: The largest risk-increasing effect this feature had on any single applicant
- **Min SHAP (risk-decreasing)**: The largest risk-decreasing effect this feature had on any single applicant

The percentage breakdown tells you for what proportion of applicants this feature increased vs decreased risk.

---

## Tab 5: Fairness & Bias Mitigation

### Headline Metrics (Before Mitigation)

**Disparate Impact Ratio**
This is the approval rate of the disadvantaged group divided by the approval rate of the advantaged group. Values below 0.800 indicate potential bias.

**Threshold (4/5ths Rule): 0.800**
The legal standard from the Equal Employment Opportunity Commission (EEOC), adapted for fair lending. If the ratio falls below 0.800, the lending practice is presumed to have a disparate impact and the institution must justify or correct it.

**Status**
Whether the model passes or fails the 4/5ths rule threshold.

### Metrics by Gender Group Table

This table shows detailed metrics for each gender group including approval rate, actual vs predicted default rates, average risk score, accuracy, and AUC. The key thing to look for is whether the predicted default rate closely matches the actual default rate for each group. A large gap between predicted and actual rates for one group (but not the other) indicates miscalibration that is the root cause of bias.

### Approval Rate Chart
Bars show each group's approval rate. The red dashed line marks the 4/5ths threshold (80% of the highest approval rate). If any bar falls below this line, the model fails the fairness test for that group.

### Actual vs Predicted Default Rate Chart
Yellow bars show actual default rates. Red bars show what the model predicts. When the red bar is much taller than the yellow bar for a group, the model over-estimates that group's risk, leading to unfair rejections.

### Bias Mitigation Results (Fairlearn)

This section shows the results of two concrete mitigation strategies applied using Fairlearn:

**ThresholdOptimizer (Post-processing)**
Adjusts decision thresholds per group to equalize approval rates without retraining the model. The original XGBoost model's predictions are preserved; only the decision boundary changes. This is the recommended approach for production because it preserves the most accuracy.

**ExponentiatedGradient (In-processing)**
Trains a new classifier (Logistic Regression base) with Demographic Parity constraints built directly into the learning objective. This produces a model that is inherently fair from the start, but may have a larger accuracy trade-off.

For each mitigation method, the dashboard shows:
- **Mitigated DIR**: The new Disparate Impact Ratio after applying the mitigation
- **Overall Accuracy**: The model's accuracy after mitigation (expect a small decrease as the trade-off for fairness)
- **4/5ths Rule**: Whether the mitigated model passes the legal threshold
- **Group-level metrics table**: Approval rates, default rates, and accuracy per gender group after mitigation

### Before vs After Comparison Table

A summary table comparing the original unmitigated model against both mitigation strategies. This makes it easy to see the accuracy-fairness trade-off for each approach. The key insight is that correcting bias always costs some overall accuracy, because the original model was exploiting biased patterns to boost its scores.

---

## Tab 6: Live Prediction

### The Input Form
Twelve fields that mirror a simplified loan application. The form is divided into three sections:
- **Account & Credit**: Financial account status, loan amount, and duration
- **Personal Info**: Age, employment, housing, and job type
- **Loan Details**: Purpose, credit history, installment rate, and other existing loans

### The Risk Gauge
A semicircular gauge showing the predicted probability of default:
- **Green zone (0-30%)**: Low risk. The model is fairly confident this applicant will repay.
- **Yellow zone (30-60%)**: Moderate risk. The applicant has some risk factors but also some protective factors.
- **Red zone (60-100%)**: High risk. The model predicts a strong likelihood of default.
- **The red line at 50%**: The decision boundary. Above this line, the recommendation is to reject the application.

### The Three Result Cards

**APPROVED / HIGH RISK**: The binary decision. Based on the 50% threshold, the model either recommends approval or flags the applicant as high risk.

**Risk Score**: The exact probability of default, expressed as a percentage. For example, 34.2% means the model estimates a roughly one-in-three chance this applicant will default.

**Risk Category**: A simplified label (Low / Medium / High) based on where the risk score falls in the three zones.

**Confidence**: How certain the model is about its decision. Calculated as the maximum of (risk score, 1 - risk score). A confidence of 92% means the model is very sure. A confidence of 52% means the model is barely leaning one way, and the decision should be reviewed more carefully by a human.

### The SHAP Explanation Chart
A horizontal bar chart showing the top 10 factors that influenced this specific prediction:
- **Red bars (pointing right)**: Factors that increased this applicant's default risk
- **Green bars (pointing left)**: Factors that decreased this applicant's default risk
- **Bar length**: How strongly each factor influenced the decision

The plain language summary beneath the chart names the top factors working against and in favor of the applicant. This is the explanation that would appear in an adverse action notice if the application were rejected.
