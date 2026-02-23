# Reproduction Guide - Credit Risk Scoring Model

A step-by-step walkthrough to reproduce this project from scratch on your own machine.

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10 or later** installed on your system
- **pip** (Python package manager, comes with Python)
- **Git** (optional, for cloning the repository)
- A terminal or command prompt
- About **2 GB of free disk space** (for packages and model artifacts)

To check your Python version, open a terminal and run:
```bash
python3 --version
```

If you see something like `Python 3.10.x` or higher, you are good to go.

---

## Step 1: Get the Project Files

**Option A: Clone from Git (if hosted in a repository)**
```bash
git clone <repository-url>
cd credit-risk-scoring
```

**Option B: Copy the folder manually**
If you received the project as a zip file or folder, extract it and navigate to the project root:
```bash
cd /path/to/credit-risk-scoring
```

You should see the following structure:
```
credit-risk-scoring/
    app/
    assets/
    data/
    docs/
    models/
    src/
    requirements.txt
    README.md
```

---

## Step 2: Create a Virtual Environment (Recommended)

It is best practice to isolate project dependencies so they do not conflict with other Python projects on your machine.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, confirming the virtual environment is active.

---

## Step 3: Install Dependencies

All required Python packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

This will install the following (and their sub-dependencies):
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Machine learning utilities
- **xgboost, lightgbm** - Gradient boosting models
- **shap** - Model explainability
- **matplotlib, seaborn, plotly** - Visualization
- **streamlit** - Web dashboard framework
- **imbalanced-learn** - SMOTE for class imbalance
- **category_encoders** - Feature encoding
- **optuna** - Hyperparameter tuning
- **joblib** - Model serialization

Installation typically takes 2-5 minutes depending on your internet speed.

---

## Step 4: Verify the Dataset

The German Credit dataset should already be in the `data/` folder:
- `data/german_credit.data` - The raw dataset (1,000 rows, space-separated)
- `data/german_credit.doc` - Dataset documentation from UCI

If for some reason the data files are missing, you can download them manually:
```bash
# Download from UCI Machine Learning Repository
curl -o data/german_credit.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
curl -o data/german_credit.doc https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc
```

---

## Step 5: Run the Full Pipeline

This is the main step. It executes the entire machine learning workflow:

```bash
python src/run_pipeline.py
```

**What happens during this step:**

1. **Data Loading** (a few seconds) - Reads the raw dataset, decodes categorical variables from alphanumeric codes to readable labels, extracts gender for fairness analysis.

2. **Feature Engineering** (a few seconds) - Creates 7 new features from the existing data, including credit burden, financial stability score, and monthly payment amount.

3. **Data Preparation** (a few seconds) - Splits data into 80% training and 20% test sets, applies target encoding to categorical features, scales numerical features, and applies SMOTE oversampling to balance the training set.

4. **Model Training** (about 30 seconds) - Trains four different models (Logistic Regression, Random Forest, XGBoost, LightGBM), runs 5-fold cross-validation on each, and evaluates all models on the test set.

5. **Hyperparameter Tuning** (1-3 minutes) - Runs 50 Optuna trials to find the best XGBoost configuration. This is the longest step.

6. **SHAP Explainability** (about 30 seconds) - Computes SHAP values for every feature and every test sample, generates summary plots, bar plots, and waterfall plots, and saves all visualizations as images.

7. **Fairness Analysis** (a few seconds) - Computes approval rates, default rates, and accuracy for each gender group, calculates the disparate impact ratio, and checks against the 4/5ths rule.

8. **Bias Mitigation** (about 30 seconds) - Applies two Fairlearn mitigation strategies (ThresholdOptimizer and ExponentiatedGradient) to correct any detected gender bias. Computes mitigated approval rates and accuracy for comparison.

9. **Saving Artifacts** (a few seconds) - Saves the trained model, preprocessing transformers, all metrics (including mitigation results), SHAP values, ROC curve data, and confusion matrices.

**Expected total runtime:** 3-5 minutes.

**Expected output:** The terminal will print progress for each step, model comparison results, SHAP feature importance rankings, and fairness metrics. At the end you should see `All artifacts saved. Pipeline complete!`

**Files generated in `models/`:**
- `best_model.pkl` - The trained XGBoost model
- `artifacts.pkl` - Preprocessing transformers (encoder, scaler)
- `pipeline_results.json` - All metrics in JSON format
- `model_comparison.csv` - Performance comparison table
- `feature_importance.csv` - SHAP feature rankings
- `roc_data.json` - ROC curve coordinates for all models
- `confusion_matrices.json` - Confusion matrix data
- `shap_dict.pkl` - Full SHAP values and explainer object

**Files generated in `assets/`:**
- `shap_summary.png` - Beeswarm plot showing feature impact direction
- `shap_bar.png` - Bar chart of global feature importance
- `shap_waterfall.png` - Waterfall plot for a single prediction

---

## Step 6: Launch the Dashboard

Start the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

Your terminal will display:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open that URL in your web browser. You will see the interactive dashboard with six pages accessible from the sidebar:

1. **Overview** - Summary metrics and key insights
2. **Data Explorer** - Interactive charts for exploring the dataset
3. **Model Performance** - ROC curves, metrics table, confusion matrices
4. **Feature Importance and SHAP** - Global and local feature explanations
5. **Fairness Analysis** - Gender bias audit with visualizations
6. **Live Prediction** - Enter applicant details and get a risk score

To stop the dashboard, press `Ctrl+C` in the terminal.

---

## Step 7: Explore and Modify (Optional)

### Changing the Number of Tuning Trials

To speed up or extend hyperparameter tuning, edit `src/run_pipeline.py` and change the `n_trials` parameter:

```python
# Faster (less optimal)
results = run_full_pipeline(n_trials=20)

# More thorough (takes longer)
results = run_full_pipeline(n_trials=100)
```

### Using a Different Dataset

To use your own credit data:
1. Format it as a CSV with the same column structure
2. Update the column names and attribute mappings in `src/data_loader.py`
3. Adjust the feature engineering in `src/feature_engineering.py` if needed
4. Re-run the pipeline

### Adding a New Model

To add a new model to the comparison:
1. Open `src/model_training.py`
2. Add your model to the `get_base_models()` function
3. Re-run the pipeline

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"
You are likely not in the virtual environment. Run `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows) and try again.

### "FileNotFoundError: german_credit.data"
Make sure you are running commands from the project root directory (the folder containing `requirements.txt`). The pipeline expects to find `data/german_credit.data` relative to the root.

### Streamlit shows a blank page or error
Make sure the pipeline completed successfully first (Step 5). The dashboard depends on model artifacts that the pipeline generates.

### SHAP plots are not showing in the dashboard
Verify that the `assets/` folder contains `shap_summary.png`, `shap_bar.png`, and `shap_waterfall.png`. If not, re-run the pipeline.

### Slow performance on older machines
The Optuna tuning step is the most compute-intensive. Reduce `n_trials` from 50 to 20 for faster execution with slightly less optimal results.

---

## File Reference

| File | Purpose |
|------|---------|
| `src/data_loader.py` | Loads raw data, decodes coded attributes, extracts gender |
| `src/feature_engineering.py` | Creates 7 derived features, encodes and scales data |
| `src/model_training.py` | Trains 4 models, tunes XGBoost, computes fairness metrics |
| `src/explainability.py` | Computes and visualizes SHAP values |
| `src/run_pipeline.py` | Orchestrates the full end-to-end workflow |
| `app/streamlit_app.py` | Interactive web dashboard (6 pages) |
| `requirements.txt` | Python package dependencies |
