"""
Model training, hyperparameter tuning, and evaluation for credit scoring.
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM.
Uses Optuna for hyperparameter optimization on the best model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_base_models():
    """Return dictionary of base models to compare."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, scale_pos_weight=2.33, eval_metric='logloss',
            verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, is_unbalance=True, verbose=-1
        )
    }


def evaluate_model(model, X_test, y_test):
    """Compute comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)

    return {
        'metrics': metrics,
        'report': report,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr, roc_thresholds),
        'pr_curve': (prec_curve, rec_curve, pr_thresholds),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


def compare_models(X_train, X_test, y_train, y_test, use_smote=True):
    """Train and compare all base models. Returns results dict."""
    models = get_base_models()
    results = {}

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    for name, model in models.items():
        # Train
        model.fit(X_train_res, y_train_res)

        # Cross-validation on original training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

        # Evaluate on test
        eval_results = evaluate_model(model, X_test, y_test)
        eval_results['cv_auc_mean'] = cv_scores.mean()
        eval_results['cv_auc_std'] = cv_scores.std()
        eval_results['model'] = model

        results[name] = eval_results
        print(f"{name}: AUC={eval_results['metrics']['roc_auc']:.4f}, "
              f"CV-AUC={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


def tune_best_model(X_train, y_train, X_test, y_test, n_trials=50):
    """Hyperparameter tuning with Optuna for XGBoost."""

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
        }

        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='roc_auc')
        return scores.mean()

    study = optuna.create_study(direction='maximize', study_name='xgboost_credit_scoring')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Train final model with best params
    best_params = study.best_params
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['verbosity'] = 0

    best_model = XGBClassifier(**best_params)
    best_model.fit(X_res, y_res)

    eval_results = evaluate_model(best_model, X_test, y_test)
    eval_results['model'] = best_model
    eval_results['best_params'] = best_params
    eval_results['study'] = study

    return best_model, eval_results


def compute_fairness_metrics(model, X_test, y_test, gender_test):
    """Compute fairness metrics across gender groups."""
    if gender_test is None:
        return None

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fairness = {}
    for group in gender_test.unique():
        mask = gender_test == group
        if mask.sum() == 0:
            continue
        group_metrics = {
            'count': int(mask.sum()),
            'approval_rate': float(1 - y_pred[mask].mean()),
            'actual_default_rate': float(y_test[mask].mean()),
            'predicted_default_rate': float(y_pred[mask].mean()),
            'avg_risk_score': float(y_proba[mask].mean()),
            'accuracy': float(accuracy_score(y_test[mask], y_pred[mask])),
        }
        if y_test[mask].nunique() > 1:
            group_metrics['auc'] = float(roc_auc_score(y_test[mask], y_proba[mask]))
        fairness[group] = group_metrics

    # Compute disparate impact (if both groups exist)
    groups = list(fairness.keys())
    if len(groups) >= 2:
        rates = [fairness[g]['approval_rate'] for g in groups]
        fairness['disparate_impact_ratio'] = min(rates) / max(rates) if max(rates) > 0 else 0
        # 4/5ths rule: ratio should be >= 0.8 for fairness
        fairness['passes_four_fifths_rule'] = fairness['disparate_impact_ratio'] >= 0.8

    return fairness


def mitigate_bias(model, X_train, y_train, X_test, y_test, gender_train, gender_test):
    """Apply bias mitigation using Fairlearn's ThresholdOptimizer and ExponentiatedGradient."""
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
    )

    mitigation_results = {}

    # --- Strategy 1: Post-processing threshold adjustment ---
    threshold_optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        prefit=True,
        predict_method="predict_proba",
    )
    threshold_optimizer.fit(X_train, y_train, sensitive_features=gender_train)
    y_pred_mitigated = threshold_optimizer.predict(X_test, sensitive_features=gender_test)

    # Compute mitigated fairness metrics
    mitigated_approval = 1 - y_pred_mitigated.mean()
    mitigated_metrics_by_group = {}
    for group in gender_test.unique():
        mask = gender_test == group
        mitigated_metrics_by_group[group] = {
            'count': int(mask.sum()),
            'approval_rate': float(1 - y_pred_mitigated[mask].mean()),
            'actual_default_rate': float(y_test[mask].mean()),
            'predicted_default_rate': float(y_pred_mitigated[mask].mean()),
            'accuracy': float(accuracy_score(y_test[mask], y_pred_mitigated[mask])),
        }

    rates = [v['approval_rate'] for v in mitigated_metrics_by_group.values()]
    mitigated_dir = min(rates) / max(rates) if max(rates) > 0 else 0

    mitigation_results['threshold_optimizer'] = {
        'method': 'ThresholdOptimizer (Post-processing)',
        'description': 'Adjusts decision thresholds per group to equalize approval rates',
        'metrics_by_group': mitigated_metrics_by_group,
        'disparate_impact_ratio': mitigated_dir,
        'passes_four_fifths_rule': mitigated_dir >= 0.8,
        'overall_accuracy': float(accuracy_score(y_test, y_pred_mitigated)),
        'dp_difference': float(demographic_parity_difference(
            y_test, y_pred_mitigated, sensitive_features=gender_test
        )),
        'dp_ratio': float(demographic_parity_ratio(
            y_test, y_pred_mitigated, sensitive_features=gender_test
        )),
    }

    # --- Strategy 2: In-processing with ExponentiatedGradient ---
    from sklearn.linear_model import LogisticRegression as LR_Fair
    base_estimator = LR_Fair(max_iter=1000, random_state=42)
    exp_grad = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
    )
    exp_grad.fit(X_train, y_train, sensitive_features=gender_train)
    y_pred_eg = exp_grad.predict(X_test)

    eg_metrics_by_group = {}
    for group in gender_test.unique():
        mask = gender_test == group
        eg_metrics_by_group[group] = {
            'count': int(mask.sum()),
            'approval_rate': float(1 - y_pred_eg[mask].mean()),
            'actual_default_rate': float(y_test[mask].mean()),
            'predicted_default_rate': float(y_pred_eg[mask].mean()),
            'accuracy': float(accuracy_score(y_test[mask], y_pred_eg[mask])),
        }

    eg_rates = [v['approval_rate'] for v in eg_metrics_by_group.values()]
    eg_dir = min(eg_rates) / max(eg_rates) if max(eg_rates) > 0 else 0

    mitigation_results['exponentiated_gradient'] = {
        'method': 'ExponentiatedGradient (In-processing)',
        'description': 'Trains a fair classifier using Demographic Parity constraints',
        'metrics_by_group': eg_metrics_by_group,
        'disparate_impact_ratio': eg_dir,
        'passes_four_fifths_rule': eg_dir >= 0.8,
        'overall_accuracy': float(accuracy_score(y_test, y_pred_eg)),
        'dp_difference': float(demographic_parity_difference(
            y_test, y_pred_eg, sensitive_features=gender_test
        )),
        'dp_ratio': float(demographic_parity_ratio(
            y_test, y_pred_eg, sensitive_features=gender_test
        )),
    }

    print("\n  Bias Mitigation Results:")
    for name, res in mitigation_results.items():
        print(f"    {res['method']}:")
        print(f"      DIR: {res['disparate_impact_ratio']:.3f} "
              f"({'PASS' if res['passes_four_fifths_rule'] else 'FAIL'})")
        print(f"      Overall Accuracy: {res['overall_accuracy']:.3f}")

    return mitigation_results


def save_model(model, artifacts, path='models/'):
    """Save model and preprocessing artifacts."""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, 'best_model.pkl'))
    joblib.dump(artifacts, os.path.join(path, 'artifacts.pkl'))
    print(f"Model and artifacts saved to {path}")


def load_model(path='models/'):
    """Load saved model and artifacts."""
    model = joblib.load(os.path.join(path, 'best_model.pkl'))
    artifacts = joblib.load(os.path.join(path, 'artifacts.pkl'))
    return model, artifacts
