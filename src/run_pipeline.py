"""
Main pipeline runner. Executes the full credit scoring workflow:
1. Load & decode data
2. Feature engineering
3. Feature preparation
4. Model training & comparison
5. Hyperparameter tuning
6. SHAP explainability
7. Fairness analysis
8. Bias mitigation (Fairlearn)
9. Save all artifacts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import joblib

from src.data_loader import load_raw_data, decode_features, extract_gender, get_feature_types
from src.feature_engineering import create_derived_features, prepare_features
from src.model_training import (
    compare_models, tune_best_model, compute_fairness_metrics,
    evaluate_model, save_model, mitigate_bias
)
from src.explainability import (
    compute_shap_values, get_feature_importance_df,
    plot_shap_summary, plot_shap_bar, explain_single_prediction
)


def run_full_pipeline(data_path=None, n_trials=50):
    """Execute the complete credit scoring pipeline."""

    print("=" * 60)
    print("CREDIT SCORING MODEL - FULL PIPELINE")
    print("=" * 60)

    # ------ 1. Load Data ------
    print("\n[1/8] Loading and decoding data...")
    df_raw = load_raw_data(data_path)
    df = decode_features(df_raw)
    df = extract_gender(df)

    print(f"  Dataset shape: {df.shape}")
    print(f"  Target distribution:\n{df['target'].value_counts().to_string()}")
    print(f"  Default rate: {df['target'].mean():.2%}")

    # Save EDA stats
    eda_stats = {
        'shape': list(df.shape),
        'default_rate': float(df['target'].mean()),
        'target_distribution': df['target'].value_counts().to_dict(),
        'missing_values': int(df.isnull().sum().sum()),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numerical_stats': {},
        'categorical_stats': {},
    }

    cat_cols, num_cols = get_feature_types(df)
    for col in num_cols:
        eda_stats['numerical_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'skewness': float(df[col].skew()),
        }
    for col in cat_cols:
        eda_stats['categorical_stats'][col] = df[col].value_counts().to_dict()

    # ------ 2. Feature Engineering ------
    print("\n[2/8] Engineering features...")
    df_feat = create_derived_features(df)
    print(f"  Features after engineering: {df_feat.shape[1]}")
    new_features = [c for c in df_feat.columns if c not in df.columns]
    print(f"  New features created: {new_features}")

    # ------ 3. Prepare Features ------
    print("\n[3/8] Preparing features (encoding, scaling, splitting)...")
    X_train, X_test, y_train, y_test, artifacts = prepare_features(df_feat)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Train default rate: {y_train.mean():.2%}")
    print(f"  Test default rate: {y_test.mean():.2%}")

    # ------ 4. Model Comparison ------
    print("\n[4/8] Training and comparing models...")
    comparison_results = compare_models(X_train, X_test, y_train, y_test, use_smote=True)

    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': res['metrics']['accuracy'],
            'Precision': res['metrics']['precision'],
            'Recall': res['metrics']['recall'],
            'F1 Score': res['metrics']['f1_score'],
            'ROC AUC': res['metrics']['roc_auc'],
            'Avg Precision': res['metrics']['avg_precision'],
            'CV AUC Mean': res['cv_auc_mean'],
            'CV AUC Std': res['cv_auc_std'],
        }
        for name, res in comparison_results.items()
    }).T

    print("\n  Model Comparison:")
    print(comparison_df.round(4).to_string())

    # ------ 5. Hyperparameter Tuning ------
    print(f"\n[5/8] Tuning XGBoost with Optuna ({n_trials} trials)...")
    best_model, tuned_results = tune_best_model(
        X_train, y_train, X_test, y_test, n_trials=n_trials
    )
    print(f"  Tuned AUC: {tuned_results['metrics']['roc_auc']:.4f}")
    print(f"  Best params: {tuned_results['best_params']}")

    # ------ 6. SHAP Explainability ------
    print("\n[6/8] Computing SHAP explanations...")
    shap_dict = compute_shap_values(best_model, X_train, X_test, artifacts['feature_names'])
    importance_df = get_feature_importance_df(shap_dict, X_test)
    print("  Top 10 features by SHAP importance:")
    print(importance_df.head(10).to_string(index=False))

    # Save SHAP plots
    os.makedirs('assets', exist_ok=True)
    plot_shap_summary(shap_dict, X_test, save_path='assets/shap_summary.png')
    plot_shap_bar(shap_dict, X_test, save_path='assets/shap_bar.png')
    explain_single_prediction(shap_dict, X_test, index=0, save_path='assets/shap_waterfall.png')
    print("  SHAP plots saved to assets/")

    # ------ 7. Fairness Analysis ------
    print("\n[7/8] Computing fairness metrics...")
    fairness = compute_fairness_metrics(
        best_model, X_test, y_test, artifacts['gender_test']
    )
    if fairness:
        for group, metrics in fairness.items():
            if isinstance(metrics, dict):
                print(f"\n  {group}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {group}: {metrics}")

    # ------ 8. Bias Mitigation ------
    print("\n[8/8] Running bias mitigation (Fairlearn)...")
    mitigation_results = mitigate_bias(
        best_model, X_train, y_train, X_test, y_test,
        artifacts['gender_train'], artifacts['gender_test']
    )

    # ------ Save Everything ------
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    save_model(best_model, artifacts, path='models/')

    # Save comparison results
    comparison_df.to_csv('models/model_comparison.csv')

    # Save importance
    importance_df.to_csv('models/feature_importance.csv', index=False)

    # Save all results as JSON-serializable dict
    pipeline_results = {
        'eda_stats': eda_stats,
        'comparison': {
            name: {k: v for k, v in res['metrics'].items()}
            for name, res in comparison_results.items()
        },
        'tuned_metrics': tuned_results['metrics'],
        'best_params': {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v)) for k, v in tuned_results['best_params'].items()},
        'feature_importance': importance_df.head(15).to_dict(orient='records'),
        'fairness': {},
        'mitigation': {},
    }
    if fairness:
        for k, v in fairness.items():
            if isinstance(v, dict):
                pipeline_results['fairness'][k] = {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv for kk, vv in v.items()}
            else:
                pipeline_results['fairness'][k] = float(v) if isinstance(v, (float, np.floating, bool, np.bool_)) else v

    # Save mitigation results
    for method_name, method_results in mitigation_results.items():
        pipeline_results['mitigation'][method_name] = {
            'method': method_results['method'],
            'description': method_results['description'],
            'disparate_impact_ratio': method_results['disparate_impact_ratio'],
            'passes_four_fifths_rule': method_results['passes_four_fifths_rule'],
            'overall_accuracy': method_results['overall_accuracy'],
            'dp_difference': method_results['dp_difference'],
            'dp_ratio': method_results['dp_ratio'],
            'metrics_by_group': method_results['metrics_by_group'],
        }

    with open('models/pipeline_results.json', 'w') as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    # Save ROC curve data for all models + tuned
    roc_data = {}
    for name, res in comparison_results.items():
        fpr, tpr, _ = res['roc_curve']
        roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': res['metrics']['roc_auc']}
    fpr, tpr, _ = tuned_results['roc_curve']
    roc_data['XGBoost (Tuned)'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': tuned_results['metrics']['roc_auc']}
    with open('models/roc_data.json', 'w') as f:
        json.dump(roc_data, f)

    # Save confusion matrices
    cm_data = {}
    for name, res in comparison_results.items():
        cm_data[name] = res['confusion_matrix'].tolist()
    cm_data['XGBoost (Tuned)'] = tuned_results['confusion_matrix'].tolist()
    with open('models/confusion_matrices.json', 'w') as f:
        json.dump(cm_data, f)

    # Save SHAP values for app
    joblib.dump(shap_dict, 'models/shap_dict.pkl')

    print("\nAll artifacts saved. Pipeline complete!")
    print("=" * 60)

    return pipeline_results


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    results = run_full_pipeline(n_trials=50)
