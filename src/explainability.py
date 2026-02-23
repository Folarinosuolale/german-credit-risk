"""
Model explainability using SHAP (SHapley Additive exPlanations).
Provides global and local feature importance explanations.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def compute_shap_values(model, X_train, X_test, feature_names=None):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_value = explainer.expected_value

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'expected_value': expected_value,
        'feature_names': feature_names or X_test.columns.tolist(),
    }


def get_feature_importance_df(shap_dict, X_test):
    """Get feature importance as a DataFrame sorted by mean |SHAP|."""
    shap_values = shap_dict['shap_values']
    feature_names = shap_dict['feature_names']

    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'std_shap': np.abs(shap_values).std(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    return importance


def plot_shap_summary(shap_dict, X_test, save_path=None):
    """Generate SHAP summary plot (beeswarm)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_dict['shap_values'], X_test,
        feature_names=shap_dict['feature_names'],
        show=False, max_display=15
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_shap_bar(shap_dict, X_test, save_path=None):
    """Generate SHAP bar plot (global importance)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_dict['shap_values'], X_test,
        feature_names=shap_dict['feature_names'],
        plot_type='bar', show=False, max_display=15
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def explain_single_prediction(shap_dict, X_test, index=0, save_path=None):
    """Generate SHAP waterfall plot for a single prediction."""
    shap_values = shap_dict['shap_values']
    expected_value = shap_dict['expected_value']
    feature_names = shap_dict['feature_names']

    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=expected_value,
        data=X_test.iloc[index].values,
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_shap_dependence(shap_dict, X_test, feature, interaction_feature=None, save_path=None):
    """Generate SHAP dependence plot for a feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_dict['shap_values'],
        X_test, feature_names=shap_dict['feature_names'],
        interaction_index=interaction_feature,
        show=False, ax=ax
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
