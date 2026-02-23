# Credit Risk Scoring Model - Stakeholder Brief

## What Is This?

We built an intelligent system that helps banks and lending institutions decide whether to approve or reject a loan application. Think of it as a highly experienced loan officer that can review thousands of applications in seconds, but unlike a human, it can clearly explain every single decision it makes and can be audited for fairness. As part of that audit, I tested it for gender bias, found a problem, and applied two correction methods to fix it.

---

## The Problem I Solved

When someone walks into a bank and applies for a loan, the bank needs to answer one question: **"Will this person pay us back?"**

Getting this wrong is expensive in both directions:

- **Approve someone who defaults:** The bank loses the money it lent out. Across a large portfolio, bad approvals can cost millions.
- **Reject someone who would have paid back:** The bank misses out on interest income, and the applicant misses out on a loan they deserved.

Traditionally, loan officers make these decisions using gut instinct, simple checklists, or basic scorecards. These approaches have three major problems:

1. **Inconsistency** - Two different officers might make different decisions for the same applicant
2. **Opacity** - When someone gets rejected, the bank often cannot clearly explain why
3. **Hidden bias** - Without realizing it, the process may treat certain groups (by gender, age, or background) unfairly

Our model addresses all three.

---

## What I Built

### The Model

We trained a machine learning model on historical data from 1,000 past loan applicants. The model learned the patterns that distinguish borrowers who repay from those who default. It looks at 27 different data points about each applicant, including:

- How much money is in their checking and savings accounts
- What the loan is for (car, education, home repair, etc.)
- How long they have been at their current job
- Their credit history (have they paid back past loans on time?)
- How much they want to borrow and for how long

The model then produces a **risk score between 0% and 100%**. A score of 15% means the applicant has a low chance of defaulting. A score of 72% means high risk.

### How Good Is It?

The model correctly identifies risky borrowers **80% of the time** (measured by a standard industry metric called ROC AUC, where our score is 0.799 out of 1.0). For context, a perfect model would score 1.0, a random coin flip would score 0.5, and most production credit models at banks score between 0.70 and 0.85.

### The Dashboard

We built an interactive web application where anyone on the team can:

- **Explore the data** - See charts and distributions of applicant characteristics
- **Compare models** - View how different algorithms performed against each other
- **Understand decisions** - See exactly which factors drove any given prediction
- **Test live scenarios** - Enter a hypothetical applicant's details and get an instant risk assessment with a full explanation
- **Review fairness** - Check whether the model treats men and women equally

---

## What I Discovered

### The Top 5 Factors That Predict Loan Default

| Rank | Factor | What It Means |
|------|--------|--------------|
| 1 | **Checking account status** | Applicants with no checking account actually default less. Those with overdrawn accounts default at 4x the rate. |
| 2 | **Loan purpose** | Education and vacation loans are riskier. Car and furniture loans (backed by a physical asset) are safer. |
| 3 | **Savings balance** | More savings = lower risk. Applicants with over 1,000 DM in savings rarely default. |
| 4 | **Credit history** | People who have paid back previous loans on time are far more likely to pay back the next one. |
| 5 | **Monthly payment burden** | It is not just the total amount that matters. How much someone pays each month relative to the loan size is a stronger signal. |

### A Fairness Concern I Found

When I audited the model for fairness across gender groups, I found that:

- **Male applicants** were approved at a rate of **70.5%**
- **Female applicants** were approved at only **55.7%**

This gap exceeds the legal threshold set by US employment and lending regulations (the "4/5ths rule"). In plain terms: if the group with the highest approval rate gets approved 70% of the time, then every other group should be approved at least 56% of the time (80% of 70%). Female applicants barely fall below this at 55.7%.

**This is an important finding.** It means the model, in its original form, should not be deployed to production without correcting this bias.

### How I Fixed It

We applied two bias correction methods using a tool called Fairlearn (an open-source library from Microsoft designed specifically for this purpose):

1. **Threshold Adjustment:** Instead of using the same approval cutoff for everyone, I adjusted the cutoff for each group so that approval rates are equalized. This is like recalibrating a scale that was reading differently for different people. The original model's intelligence is preserved; only the decision boundary changes.

2. **Fair Retraining:** I retrained a new model with a built-in fairness constraint that forces it to treat both groups equitably from the start.

Both methods brought the approval rate ratio above the legal threshold. The first method preserves more of the original model's accuracy, making it the recommended approach for production deployment.

---

## Why This Matters for the Business

### Regulatory Compliance

Banks are legally required to:
- **Explain loan rejections** to applicants (Equal Credit Opportunity Act)
- **Prove they are not discriminating** against protected groups (Fair Lending laws)
- **Validate their models** and show regulators how they work (OCC/Federal Reserve guidelines)

Our model does all three. The SHAP explanations can be used directly in adverse action notices (the letters banks send when they reject an application). The fairness audit demonstrates proactive compliance monitoring.

### Financial Impact

For a bank processing 100,000 loan applications per year:
- A **1% improvement in default detection** could prevent $2-5M in annual losses (depending on average loan size)
- **Faster, automated scoring** reduces the cost per application from $50-100 (manual review) to under $1
- **Consistent decisions** reduce the risk of costly discrimination lawsuits

### Competitive Advantage

Institutions that can make faster, fairer, more transparent lending decisions win more customers and face fewer regulatory problems. This model provides the foundation for that capability.

---

## What I Recommend Next

1. **Deploy the bias-corrected model** using the threshold adjustment method, which passed the fairness test while preserving the most accuracy
2. **Pilot the dashboard** with a small team of loan officers to validate the model's predictions against their experience
3. **Integrate with existing systems** by connecting the model's API to the loan origination workflow
4. **Set up fairness monitoring** so I can track both performance and fairness on an ongoing basis as new data comes in, catching any drift in bias over time
5. **Retrain quarterly** to keep the model current with changing economic conditions and borrower behavior, re-running the fairness audit each cycle

---

## Technical Details (For Those Who Want Them)

| Item | Detail |
|------|--------|
| Algorithm | XGBoost (gradient-boosted decision trees) |
| Training data | 1,000 historical loan applications (German Credit, UCI) |
| Features used | 27 (20 original + 7 I engineered from the raw data) |
| Accuracy metric | ROC AUC = 0.799 |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Fairness metric (before) | Disparate Impact Ratio = 0.791 (below 0.80 threshold) |
| Fairness metric (after) | Disparate Impact Ratio >= 0.80 (above threshold, PASS) |
| Bias mitigation | Fairlearn (ThresholdOptimizer, ExponentiatedGradient) |
| Dashboard | Streamlit web application with 6 interactive pages |
| Languages/Tools | Python, XGBoost, SHAP, Fairlearn, scikit-learn, Plotly, Streamlit |

---

## Questions?

This document provides a high-level overview. For full technical details including methodology, model comparison, hyperparameter tuning results, and complete fairness analysis, refer to the **Technical Report** (PROJECT_REPORT.md).
