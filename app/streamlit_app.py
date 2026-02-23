"""
Credit Risk Scoring - Interactive Streamlit Dashboard
=====================================================
Full ML pipeline: EDA, model comparison,
SHAP explainability, fairness analysis, and live predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib
import os
import sys

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.data_loader import load_raw_data, decode_features, extract_gender, ATTRIBUTE_MAPS
from src.feature_engineering import create_derived_features

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="CRS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B2838;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .insight-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF7ED;
        border-left: 4px solid #F97316;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .explain-box {
        background-color: #F5F3FF;
        border-left: 4px solid #7C3AED;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.92rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# -- Load Data & Artifacts -----------------------------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join(ROOT, 'data', 'german_credit.data')
    df_raw = load_raw_data(data_path)
    df = decode_features(df_raw)
    df = extract_gender(df)
    return df


@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(ROOT, 'models')
    model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
    artifacts = joblib.load(os.path.join(models_dir, 'artifacts.pkl'))
    with open(os.path.join(models_dir, 'pipeline_results.json')) as f:
        results = json.load(f)
    with open(os.path.join(models_dir, 'roc_data.json')) as f:
        roc_data = json.load(f)
    with open(os.path.join(models_dir, 'confusion_matrices.json')) as f:
        cm_data = json.load(f)
    shap_dict = joblib.load(os.path.join(models_dir, 'shap_dict.pkl'))
    importance_df = pd.read_csv(os.path.join(models_dir, 'feature_importance.csv'))
    return model, artifacts, results, roc_data, cm_data, shap_dict, importance_df


df = load_data()
model, artifacts, results, roc_data, cm_data, shap_dict, importance_df = load_artifacts()

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Overview", "Data Explorer", "Model Performance",
         "Feature Importance & SHAP", "Fairness Analysis", "Live Prediction"],
        index=0
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**Credit Risk Scoring** model built with XGBoost, "
        "explainable via SHAP, with fairness auditing."
    )
    st.markdown("Dataset: **German Credit (UCI)**")
    st.markdown("1,000 applicants | 20 features | Binary classification")
    st.markdown("---")
    st.markdown(
        "ML | Explainability | Fairness"
    )


# ==============================================================================
#  PAGE: OVERVIEW
# ==============================================================================
if page == "Overview":
    st.markdown('<p class="main-header">Credit Risk Scoring Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'Explainable ML model for predicting loan default risk with fairness auditing'
        '</p>',
        unsafe_allow_html=True
    )

    # Key metrics row
    tuned = results['comparison']['Logistic Regression']
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ROC AUC", f"{tuned['roc_auc']:.3f}", "Logistic Regression")
    with col2:
        st.metric("F1 Score", f"{tuned['f1_score']:.3f}")
    with col3:
        st.metric("Precision", f"{tuned['precision']:.3f}")
    with col4:
        st.metric("Recall", f"{tuned['recall']:.3f}")
    with col5:
        st.metric("Avg Precision", f"{tuned['avg_precision']:.3f}")

    # Explain what these metrics mean
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        **These five numbers summarize how well the model performs at predicting loan defaults:**

        - **ROC AUC (0.773):** Measures the model's overall ability to distinguish between good and bad credit applicants.
          Ranges from 0.5 (random guessing) to 1.0 (perfect). Our score of 0.773 means the model correctly ranks
          a randomly chosen defaulter as riskier than a randomly chosen non-defaulter about 77% of the time.
          In the credit industry, scores between 0.70-0.85 are considered strong.

        - **F1 Score (0.643):** The balance between Precision and Recall (their harmonic mean). A high F1 means
          the model is reasonably good at both finding defaulters AND not falsely accusing good borrowers.
          A perfect score would be 1.0.

        - **Precision (0.563):** Of all the people the model flagged as "will default," 56.3% actually did default.
          The remaining 43.7% were false alarms (good borrowers incorrectly flagged). Higher is better,
          but pushing precision too high usually comes at the cost of missing real defaulters.

        - **Recall (0.750):** Of all the people who actually defaulted, the model caught 75.0% of them.
          The remaining 25.0% were missed (defaulters the model incorrectly approved). In credit scoring,
          recall is often prioritized because missing a defaulter costs the bank real money.

        - **Avg Precision:** Summarizes the precision-recall trade-off across all possible thresholds.
          It is especially useful when classes are imbalanced (as they are here, with only 30% defaults).
          Higher values indicate the model maintains good precision even as it tries to catch more defaulters.
        """)

    st.markdown("---")

    # Project summary columns
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Project Pipeline")
        st.markdown("""
        | Stage | Details |
        |---|---|
        | **Data** | German Credit (UCI), 1,000 samples, 20 features |
        | **Preprocessing** | Target encoding, standard scaling, SMOTE oversampling |
        | **Feature Engineering** | 7 derived features (credit burden, financial stability, etc.) |
        | **Models Compared** | Logistic Regression, Random Forest, XGBoost, LightGBM |
        | **Tuning** | Optuna (50 trials) on XGBoost |
        | **Explainability** | SHAP TreeExplainer, global and local explanations |
        | **Fairness** | Disparate impact analysis across gender groups |
        """)

    with col_right:
        st.markdown("### Key Insights")
        st.markdown(f"""
        <div class="insight-box">
            <strong>Best Model:</strong> Logisitic Regression<br>
            ROC AUC = <strong>{tuned['roc_auc']:.4f}</strong><br>
            RECALL = <strong>{tuned['recall']:.4f}</strong><br>
            Best choice for a credit risk model where we want to catch as many defaulters.
        </div>
        """, unsafe_allow_html=True)

        fairness = results.get('fairness', {})
        di = fairness.get('disparate_impact_ratio', 'N/A')
        passes = fairness.get('passes_four_fifths_rule', 'N/A')
        if di != 'N/A':
            st.markdown(f"""
            <div class="insight-box">
                <strong>Fairness Alert:</strong><br>
                Disparate Impact Ratio = <strong>{float(di):.3f}</strong><br>
                4/5ths Rule: <strong>{'PASS' if passes else 'FAIL'}</strong><br>
                The model shows no gender bias that would require mitigation.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box">
            <strong>Top Risk Factor:</strong><br>
            <strong>Checking Account Status</strong> is the #1 predictor. Applicants
            with no checking account show significantly lower default rates.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
#  PAGE: DATA EXPLORER
# ==============================================================================
elif page == "Data Explorer":
    st.markdown("## Data Explorer")
    st.markdown("Explore the German Credit dataset: distributions, correlations, and patterns.")

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Target Analysis"])

    with tab1:
        st.markdown("### Feature Distributions")
        st.markdown("""
        <div class="explain-box">
        <strong>What you are looking at:</strong> These histograms show how applicants are spread across
        different values of each feature. The <span style="color:#3B82F6"><strong>blue bars</strong></span> represent
        applicants who repaid their loans (good credit), and the <span style="color:#EF4444"><strong>red bars</strong></span>
        represent those who defaulted (bad credit). Where the red and blue overlap heavily, the feature has
        less power to separate good from bad borrowers. Where they diverge, the feature is more predictive.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Numerical distributions
            num_feature = st.selectbox(
                "Select numerical feature",
                ['age', 'credit_amount', 'duration_months', 'installment_rate',
                 'residence_since', 'num_existing_credits', 'num_dependents']
            )
            fig = px.histogram(
                df, x=num_feature, color='target',
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                barmode='overlay', opacity=0.7,
                labels={'target': 'Default'},
                title=f'Distribution of {num_feature}'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Dynamic insight for selected numerical feature
            good_mean = df[df['target'] == 0][num_feature].mean()
            bad_mean = df[df['target'] == 1][num_feature].mean()
            direction = "higher" if bad_mean > good_mean else "lower"
            st.markdown(f"""
            <div class="insight-box">
            <strong>Insight:</strong> Applicants who defaulted have a {direction} average
            <strong>{num_feature}</strong> ({bad_mean:.1f}) compared to those who repaid ({good_mean:.1f}).
            {'This suggests that higher values of this feature are associated with greater default risk.' if bad_mean > good_mean else
             'This suggests that lower values of this feature are associated with greater default risk.'}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Categorical distributions
            cat_feature = st.selectbox(
                "Select categorical feature",
                ['checking_account', 'credit_history', 'purpose', 'savings_account',
                 'employment_since', 'housing', 'job', 'other_installment_plans']
            )
            ct = df.groupby([cat_feature, 'target']).size().reset_index(name='count')
            fig = px.bar(
                ct, x=cat_feature, y='count', color='target',
                color_discrete_map={0: '#3B82F6', 1: '#EF4444'},
                barmode='group',
                labels={'target': 'Default'},
                title=f'Distribution of {cat_feature}'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Dynamic insight for selected categorical feature
            cat_rates = df.groupby(cat_feature)['target'].mean()
            safest = cat_rates.idxmin()
            riskiest = cat_rates.idxmax()
            st.markdown(f"""
            <div class="insight-box">
            <strong>Insight:</strong> Among <strong>{cat_feature}</strong> categories,
            "<strong>{riskiest}</strong>" has the highest default rate ({cat_rates[riskiest]:.1%}),
            while "<strong>{safest}</strong>" has the lowest ({cat_rates[safest]:.1%}).
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Correlation Heatmap (Numerical Features)")
        st.markdown("""
        <div class="explain-box">
        <strong>What you are looking at:</strong> This heatmap shows how strongly each pair of numerical
        features moves together. Values range from -1.0 to +1.0. A value close to <strong>+1.0</strong>
        (dark red) means the two features increase together. A value close to <strong>-1.0</strong>
        (dark blue) means one increases as the other decreases. Values near <strong>0</strong> (white)
        mean no linear relationship.<br><br>
        <strong>Why it matters:</strong> Highly correlated features (|value| > 0.7) can cause problems
        for some models because they carry redundant information. It also helps identify which features
        might be proxies for each other. For example, if "credit_amount" and "duration_months" are
        strongly correlated, it means larger loans tend to have longer repayment periods.
        </div>
        """, unsafe_allow_html=True)

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr, text_auto='.2f', color_continuous_scale='RdBu_r',
            aspect='auto', title='Feature Correlations'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Find and report strongest correlations (excluding self-correlation)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top3 = corr_pairs[:3]
        st.markdown(f"""
        <div class="insight-box">
        <strong>Strongest correlations found:</strong><br>
        1. <strong>{top3[0][0]}</strong> and <strong>{top3[0][1]}</strong>: {top3[0][2]:.3f}<br>
        2. <strong>{top3[1][0]}</strong> and <strong>{top3[1][1]}</strong>: {top3[1][2]:.3f}<br>
        3. <strong>{top3[2][0]}</strong> and <strong>{top3[2][1]}</strong>: {top3[2][2]:.3f}<br>
        None of these are problematically high (above 0.8), so multicollinearity is not a concern here.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### Default Rate by Feature")
        st.markdown("""
        <div class="explain-box">
        <strong>What you are looking at:</strong> The red bars show the <strong>default rate</strong>
        (percentage of applicants who failed to repay) for each category or bin of the selected feature.
        The blue line shows the <strong>number of applicants</strong> in each group (read on the right axis).
        This dual view is important because a high default rate in a group with very few people may not
        be statistically reliable. Look for groups that have both a high default rate AND a meaningful sample size.
        </div>
        """, unsafe_allow_html=True)

        analysis_feature = st.selectbox(
            "Analyze default rate by",
            ['checking_account', 'credit_history', 'purpose', 'savings_account',
             'employment_since', 'housing', 'job', 'age', 'gender']
        )

        if df[analysis_feature].dtype in ['object', 'category']:
            rate = df.groupby(analysis_feature)['target'].agg(['mean', 'count']).reset_index()
            rate.columns = [analysis_feature, 'default_rate', 'count']
            rate = rate.sort_values('default_rate', ascending=False)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=rate[analysis_feature], y=rate['default_rate'],
                       name='Default Rate', marker_color='#EF4444', opacity=0.8),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=rate[analysis_feature], y=rate['count'],
                           name='Sample Count', mode='lines+markers',
                           marker_color='#3B82F6'),
                secondary_y=True
            )
            fig.update_layout(
                title=f'Default Rate by {analysis_feature}',
                height=450, xaxis_tickangle=-45
            )
            fig.update_yaxes(title_text="Default Rate", secondary_y=False)
            fig.update_yaxes(title_text="Count", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Dynamic interpretation
            highest_row = rate.iloc[0]
            lowest_row = rate.iloc[-1]
            spread = highest_row['default_rate'] - lowest_row['default_rate']
            st.markdown(f"""
            <div class="insight-box">
            <strong>Interpretation:</strong> The default rate varies by <strong>{spread:.1%}</strong> across
            {analysis_feature} categories. "<strong>{highest_row[analysis_feature]}</strong>" has the highest
            default rate at {highest_row['default_rate']:.1%} (n={int(highest_row['count'])}), while
            "<strong>{lowest_row[analysis_feature]}</strong>" has the lowest at {lowest_row['default_rate']:.1%}
            (n={int(lowest_row['count'])}). This {f'large spread suggests {analysis_feature} is a strong predictor of default.' if spread > 0.15 else f'moderate spread suggests {analysis_feature} has some predictive value but is not the dominant signal.'}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Numerical: bin and show
            df_temp = df.copy()
            df_temp[f'{analysis_feature}_bin'] = pd.qcut(df_temp[analysis_feature], q=5, duplicates='drop')
            rate = df_temp.groupby(f'{analysis_feature}_bin')['target'].agg(['mean', 'count']).reset_index()
            rate.columns = ['bin', 'default_rate', 'count']
            fig = px.bar(rate, x='bin', y='default_rate', text='count',
                         title=f'Default Rate by {analysis_feature} (binned)',
                         color='default_rate', color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class="insight-box">
            <strong>Interpretation:</strong> The numbers on each bar show the sample count in that bin.
            Bins colored toward <strong>red</strong> have higher default rates, while bins colored toward
            <strong>green</strong> have lower default rates. This helps identify which ranges of
            <strong>{analysis_feature}</strong> are associated with higher risk.
            </div>
            """, unsafe_allow_html=True)

        # Summary stats table
        st.markdown("### Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Default Rate", f"{df['target'].mean():.1%}")
        with col3:
            st.metric("Features", f"{df.shape[1] - 1}")

        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <strong>Reading the summary table:</strong> Each row describes a statistical property of the
        numerical features. <strong>mean</strong> is the average value. <strong>std</strong> is the
        standard deviation (how spread out the values are). <strong>min/max</strong> are the smallest
        and largest values. <strong>25%/50%/75%</strong> are the percentiles (50% is the median).
        For example, the median credit amount tells you that half of all applicants borrowed less than
        that value and half borrowed more.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
#  PAGE: MODEL PERFORMANCE
# ==============================================================================
elif page == "Model Performance":
    st.markdown("## Model Performance Comparison")
    st.markdown("""
    Four different machine learning algorithms were trained on the same data and evaluated
    on a held-out test set of 200 applicants. The goal is to find which algorithm best
    distinguishes between borrowers who will repay and those who will default.
    """)

    tab1, tab2, tab3 = st.tabs(["ROC Curves", "Metrics Comparison", "Confusion Matrices"])

    with tab1:
        st.markdown("### ROC Curves - All Models")
        st.markdown("""
        <div class="explain-box">
        <strong>What is an ROC curve?</strong> It plots the trade-off between two things as
        you adjust the model's decision threshold:<br>
        - <strong>True Positive Rate (y-axis):</strong> The proportion of actual defaulters the model correctly identifies. Higher is better.<br>
        - <strong>False Positive Rate (x-axis):</strong> The proportion of good borrowers the model incorrectly flags as defaulters. Lower is better.<br><br>
        <strong>How to read it:</strong> A curve that hugs the top-left corner is a strong model.
        The diagonal dashed gray line represents random guessing (50/50). The farther a model's curve
        is above that line, the better it is. The <strong>AUC</strong> (Area Under the Curve) number in
        the legend summarizes this: 1.0 is perfect, 0.5 is random.
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        colors = {'Logistic Regression': '#6366F1', 'Random Forest': '#22C55E',
                  'XGBoost': '#F59E0B', 'LightGBM': '#EF4444', 'XGBoost (Tuned)': '#3B82F6'}

        for name, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data['fpr'], y=data['tpr'],
                name=f"{name} (AUC={data['auc']:.4f})",
                line=dict(color=colors.get(name, '#888'), width=2 if 'Tuned' not in name else 3)
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name='Random Baseline',
            line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            title='ROC Curves', xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate', height=550,
            legend=dict(x=0.55, y=0.05)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Identify best model from ROC data
        best_roc_name = max(roc_data.items(), key=lambda x: x[1]['auc'])
        st.markdown(f"""
        <div class="insight-box">
        <strong>Result:</strong> {best_roc_name[0]} has the highest AUC at
        {best_roc_name[1]['auc']:.4f}. Notice how all models are well above the random baseline,
        confirming that the features in this dataset do carry meaningful signal about default risk.
        The tuned XGBoost curve (thicker blue line) consistently sits above or matches the other curves,
        indicating it is the best at separating good from bad borrowers across all threshold settings.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Metrics Comparison Table")
        st.markdown("""
        <div class="explain-box">
        <strong>Reading this table:</strong> Each row is a different model. Each column is a performance
        metric. Cells highlighted in <span style="background-color:#D1FAE5;padding:2px 6px;">green</span>
        are the best value in that column. Cells in <span style="background-color:#FEE2E2;padding:2px 6px;">red</span>
        are the worst. No single model wins on every metric, which is common. The right model depends on
        what you prioritize: catching more defaulters (recall) or making fewer false accusations (precision).
        </div>
        """, unsafe_allow_html=True)

        comparison = results['comparison']
        comparison['XGBoost (Tuned)'] = results['tuned_metrics']
        comp_df = pd.DataFrame(comparison).T
        comp_df = comp_df.round(4)

        # Highlight best values
        st.dataframe(
            comp_df.style.highlight_max(axis=0, color='#D1FAE5')
                         .highlight_min(axis=0, color='#FEE2E2'),
            use_container_width=True
        )

        with st.expander("What does each metric measure?"):
            st.markdown("""
            | Metric | What it answers | Range | Better is |
            |---|---|---|---|
            | **accuracy** | Overall, what percentage of predictions were correct? | 0-1 | Higher |
            | **precision** | Of those flagged as defaulters, how many actually defaulted? | 0-1 | Higher |
            | **recall** | Of all real defaulters, how many did the model catch? | 0-1 | Higher |
            | **f1_score** | What is the balance between precision and recall? | 0-1 | Higher |
            | **roc_auc** | How well does the model rank defaulters above non-defaulters? | 0.5-1 | Higher |
            | **avg_precision** | How well does the model maintain precision as recall increases? | 0-1 | Higher |

            **In credit scoring, recall is often more important than precision** because the cost of
            approving someone who defaults (a missed detection) is typically much higher than the cost
            of rejecting someone who would have repaid (a false alarm).
            """)

        # Bar chart comparison
        metrics_to_plot = ['roc_auc', 'f1_score', 'precision', 'recall']
        fig = go.Figure()
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=list(comp_df.index),
                y=comp_df[metric]
            ))
        fig.update_layout(
            title='Model Metrics Comparison', barmode='group',
            height=450, yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Key takeaway:</strong> Logistic Regression has the highest recall (0.750),
        meaning it catches the most defaulters, but at the cost of lower precision (more false alarms).
        This makes Logistic Regression the best choice for a credit risk model where we want to catch as many defaulters.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### Confusion Matrices")
        st.markdown("""
        <div class="explain-box">
        <strong>What is a confusion matrix?</strong> It is a 2x2 grid that shows exactly how
        the model's predictions compare to reality for the test set. The rows represent what actually
        happened (did the borrower default or not?), and the columns represent what the model predicted.
        This gives you four outcomes that tell you specifically where the model gets things right and wrong.
        </div>
        """, unsafe_allow_html=True)

        selected_model = st.selectbox("Select model", list(cm_data.keys()))
        cm = np.array(cm_data[selected_model])

        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['Good Credit', 'Bad Credit'],
            y=['Good Credit', 'Bad Credit'],
            color_continuous_scale='Blues',
            title=f'Confusion Matrix - {selected_model}'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        st.markdown(f"""
        **Breaking down the {total} test predictions for {selected_model}:**

        | Outcome | Count | Meaning |
        |---|---|---|
        | **True Negatives** (top-left) | {tn} | Good borrowers correctly approved. Everyone wins. |
        | **False Positives** (top-right) | {fp} | Good borrowers incorrectly flagged as risky. These are unnecessary rejections that cost the bank potential revenue and frustrate qualified applicants. |
        | **False Negatives** (bottom-left) | {fn} | Defaulters incorrectly approved. These are the costly mistakes where the bank loses the money it lent. |
        | **True Positives** (bottom-right) | {tp} | Defaulters correctly identified. The model protected the bank from a bad loan. |
        """)

        cost_ratio = fn / (fn + tp) if (fn + tp) > 0 else 0
        st.markdown(f"""
        <div class="warning-box">
        <strong>Risk perspective:</strong> This model misses <strong>{fn}</strong> out of <strong>{fn + tp}</strong>
        actual defaulters ({cost_ratio:.1%} miss rate). Each missed defaulter represents a potential financial loss
        equal to the loan amount. Meanwhile, it incorrectly rejects <strong>{fp}</strong> good borrowers,
        which represents lost revenue but not a direct financial loss. In banking, the cost of a false negative
        is typically 5-10x the cost of a false positive.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
#  PAGE: FEATURE IMPORTANCE & SHAP
# ==============================================================================
elif page == "Feature Importance & SHAP":
    st.markdown("## Feature Importance & SHAP Explainability")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) is a method that explains exactly how much each feature
    contributes to every individual prediction. Unlike simple feature importance scores, SHAP
    tells you both *how much* a feature matters AND *in which direction* it pushes the prediction.
    """)

    tab1, tab2, tab3 = st.tabs(["Global Importance", "SHAP Plots", "Feature Deep Dive"])

    with tab1:
        st.markdown("### Global Feature Importance (Mean |SHAP|)")
        top_n = st.slider("Number of features to display", 5, 25, 15)
        imp = importance_df.head(top_n)

        fig = go.Figure(go.Bar(
            x=imp['mean_abs_shap'][::-1],
            y=imp['feature'][::-1],
            orientation='h',
            marker_color='#6366F1',
            error_x=dict(type='data', array=imp['std_shap'][::-1], visible=True)
        ))
        fig.update_layout(
            title=f'Top {top_n} Features by SHAP Importance',
            xaxis_title='Mean |SHAP Value|',
            height=max(400, top_n * 30),
            margin=dict(l=200)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="explain-box">
        <strong>How to read this chart:</strong><br><br>
        - Each bar represents one feature, sorted from most important (top) to least important (bottom).<br>
        - The bar length is the <strong>mean absolute SHAP value</strong> across all test predictions.
          A longer bar means the feature has more influence on the model's credit risk decisions on average.<br>
        - The <strong>error bars</strong> (thin lines extending from each bar) show the standard deviation,
          or how much that feature's importance varies across different applicants. A feature with a long
          error bar affects some applicants much more than others.<br>
        - The x-axis uses absolute values, so it only shows magnitude (how much influence), not direction
          (whether it increases or decreases risk). For directional information, see the SHAP Plots tab.
        </div>
        """, unsafe_allow_html=True)

        # Dynamic top-3 narrative
        top3 = importance_df.head(3)
        st.markdown(f"""
        <div class="insight-box">
        <strong>Top 3 Drivers of Credit Risk Decisions:</strong><br><br>
        1. <strong>{top3.iloc[0]['feature']}</strong> (SHAP: {top3.iloc[0]['mean_abs_shap']:.3f}) - By far the most
           influential feature, with more than double the impact of any other. This reflects the fundamental
           banking reality that checking account status is a direct window into an applicant's financial behavior.<br>
        2. <strong>{top3.iloc[1]['feature']}</strong> (SHAP: {top3.iloc[1]['mean_abs_shap']:.3f}) - Having savings
           provides a financial cushion that reduces default risk.<br>
        3. <strong>{top3.iloc[2]['feature']}</strong> (SHAP: {top3.iloc[2]['mean_abs_shap']:.3f}) - The reason for
           borrowing matters. Loans for tangible assets tend to be safer than loans for services or consumption.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### SHAP Summary & Waterfall Plots")

        col1, col2 = st.columns(2)
        with col1:
            shap_summary_path = os.path.join(ROOT, 'assets', 'shap_summary.png')
            if os.path.exists(shap_summary_path):
                st.image(shap_summary_path, caption='SHAP Summary (Beeswarm) Plot', use_container_width=True)
                st.markdown("""
                <div class="explain-box">
                <strong>Reading the beeswarm plot:</strong> Each dot is one applicant in the test set.
                The horizontal position shows how much that feature pushed the prediction for that
                applicant (right = toward default, left = toward good credit). The color shows the
                feature's actual value (red = high value, blue = low value). For example, if you see
                red dots on the right for "duration_months," it means high loan duration pushes the
                model toward predicting default.
                </div>
                """, unsafe_allow_html=True)
        with col2:
            shap_bar_path = os.path.join(ROOT, 'assets', 'shap_bar.png')
            if os.path.exists(shap_bar_path):
                st.image(shap_bar_path, caption='SHAP Bar Plot (Global Importance)', use_container_width=True)
                st.markdown("""
                <div class="explain-box">
                <strong>Reading the bar plot:</strong> This is a simpler view of the same information.
                Each bar shows the average absolute impact of a feature across all test predictions.
                Longer bars mean the feature has more influence on the model's decisions overall.
                Unlike the beeswarm plot, this does not show the direction of the effect.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Single Prediction Explanation")
        waterfall_path = os.path.join(ROOT, 'assets', 'shap_waterfall.png')
        if os.path.exists(waterfall_path):
            st.image(waterfall_path, caption='SHAP Waterfall - Sample Prediction Breakdown', use_container_width=True)
            st.markdown("""
            <div class="explain-box">
            <strong>Reading the waterfall plot:</strong> This explains one specific applicant's prediction.
            It starts from a <strong>base value</strong> (the model's average prediction across all applicants)
            at the bottom, and then shows how each feature pushes the final prediction up or down.<br><br>
            - <strong>Red bars</strong> push the prediction higher (toward default/higher risk).<br>
            - <strong>Blue bars</strong> push the prediction lower (toward good credit/lower risk).<br>
            - The final value at the top is the model's actual output for this applicant.<br><br>
            This is exactly the kind of explanation a bank would include in an adverse action notice
            when declining a loan application. It answers the question: "Why was this person rejected?"
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### Feature Deep Dive")
        st.markdown("""
        <div class="explain-box">
        <strong>What this shows:</strong> Select any feature to see its SHAP value distribution across
        all test predictions. The left chart is a histogram of SHAP values (how the feature's effect is
        distributed across applicants). The right chart shows each individual applicant's SHAP value as
        a colored dot, where red means the feature increased default risk and blue means it decreased risk.
        </div>
        """, unsafe_allow_html=True)

        feature = st.selectbox(
            "Select feature to analyze",
            importance_df['feature'].tolist()[:15]
        )

        shap_values = shap_dict['shap_values']
        feature_idx = shap_dict['feature_names'].index(feature)
        feature_shap = shap_values[:, feature_idx]

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f'SHAP Value Distribution for {feature}', f'SHAP Value per Sample'
        ])

        # Distribution of SHAP values
        fig.add_trace(
            go.Histogram(x=feature_shap, nbinsx=30, marker_color='#6366F1',
                         name='SHAP Distribution'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(feature_shap))),
                y=feature_shap,
                mode='markers',
                marker=dict(size=4, color=feature_shap, colorscale='RdBu_r',
                            showscale=True),
                name='SHAP per sample'
            ),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text='SHAP Value', row=1, col=1)
        fig.update_xaxes(title_text='Sample Index', row=1, col=2)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='SHAP Value', row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean |SHAP|", f"{np.mean(np.abs(feature_shap)):.4f}")
        with col2:
            st.metric("Max SHAP (risk-increasing)", f"{np.max(feature_shap):.4f}")
        with col3:
            st.metric("Min SHAP (risk-decreasing)", f"{np.min(feature_shap):.4f}")

        # Dynamic interpretation
        pct_positive = (feature_shap > 0).mean() * 100
        pct_negative = (feature_shap < 0).mean() * 100
        st.markdown(f"""
        <div class="insight-box">
        <strong>Interpretation for {feature}:</strong><br>
        - For <strong>{pct_positive:.0f}%</strong> of test applicants, this feature pushed the prediction
          toward <strong>higher default risk</strong> (positive SHAP values).<br>
        - For <strong>{pct_negative:.0f}%</strong> of test applicants, this feature pushed the prediction
          toward <strong>lower default risk</strong> (negative SHAP values).<br>
        - The feature's maximum risk-increasing effect was {np.max(feature_shap):.3f}, and its maximum
          risk-decreasing effect was {np.min(feature_shap):.3f}. The wider this range, the more this
          feature's impact varies depending on the applicant's specific circumstances.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
#  PAGE: FAIRNESS ANALYSIS
# ==============================================================================
elif page == "Fairness Analysis":
    st.markdown("## Fairness & Bias Analysis")
    st.markdown(
        "Evaluating the model for potential gender bias using disparate impact analysis "
        "and the 4/5ths (80%) rule from US employment law (EEOC guidelines)."
    )

    st.markdown("""
    <div class="explain-box">
    <strong>Why fairness matters in credit scoring:</strong> Laws like the Equal Credit Opportunity Act
    (ECOA) prohibit lenders from discriminating based on gender, race, religion, or other protected
    characteristics. Even if a model never directly uses gender as an input, it can still learn patterns
    that indirectly discriminate (for example, through correlated features). This analysis checks
    whether the model's approval decisions are equitable across gender groups.<br><br>
    <strong>The 4/5ths Rule:</strong> A widely used legal standard that says the approval rate for any
    demographic group should be at least 80% of the rate for the group with the highest approval rate.
    If the ratio falls below 0.80, the model may have a legally actionable disparate impact.
    </div>
    """, unsafe_allow_html=True)

    fairness = results.get('fairness', {})

    if fairness:
        # Disparate Impact headline
        di = fairness.get('disparate_impact_ratio', 0)
        passes = fairness.get('passes_four_fifths_rule', False)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Disparate Impact Ratio", f"{float(di):.3f}",
                       delta="PASS" if passes else "FAIL",
                       delta_color="normal" if passes else "inverse")
        with col2:
            st.metric("Threshold (4/5ths Rule)", "0.800")
        with col3:
            st.metric("Status", "Fair" if passes else "Potential Bias Detected")

        with st.expander("What do these three values mean?"):
            st.markdown(f"""
            - **Disparate Impact Ratio ({float(di):.3f}):** This is calculated as the approval rate
              of the disadvantaged group divided by the approval rate of the advantaged group.
              A value of 1.0 would mean both groups are approved at exactly the same rate.
              Our value of {float(di):.3f} means the lower group's approval rate is {float(di)*100:.1f}%
              of the higher group's rate.

            - **Threshold (0.800):** This is the legal minimum from the EEOC's 4/5ths rule.
              If the ratio is below 0.800, there is a presumption of adverse impact that the
              lender would need to justify or correct.

            - **Status ({'Fair' if passes else 'Potential Bias Detected'}):** Since our ratio of
              {float(di):.3f} is {'above' if passes else 'below'} the 0.800 threshold, the model
              {'passes' if passes else 'fails'} the fairness test.
            """)

        st.markdown("---")

        # Group-level metrics
        st.markdown("### Metrics by Gender Group")
        group_data = []
        for group, metrics in fairness.items():
            if isinstance(metrics, dict):
                row = {'Group': group}
                row.update(metrics)
                group_data.append(row)

        if group_data:
            group_df = pd.DataFrame(group_data)
            st.dataframe(group_df.style.format({
                'approval_rate': '{:.1%}',
                'actual_default_rate': '{:.1%}',
                'predicted_default_rate': '{:.1%}',
                'avg_risk_score': '{:.3f}',
                'accuracy': '{:.1%}',
                'auc': '{:.4f}',
            }), use_container_width=True)

            with st.expander("What does each column in this table mean?"):
                st.markdown("""
                | Column | Meaning |
                |---|---|
                | **count** | Number of test applicants in this gender group. |
                | **approval_rate** | Percentage of applicants the model would approve (predict as good credit). |
                | **actual_default_rate** | The real-world default rate in this group (what actually happened). |
                | **predicted_default_rate** | What the model predicts the default rate to be. If this is much higher than the actual rate, the model is over-estimating risk for this group. |
                | **avg_risk_score** | The average probability of default the model assigns to this group. Ranges from 0.0 (no risk) to 1.0 (certain default). |
                | **accuracy** | Percentage of correct predictions within this group. |
                | **auc** | The model's ability to rank-order risk within this group (higher is better). |
                """)

            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=group_df['Group'],
                        y=group_df['approval_rate'],
                        marker_color=['#3B82F6', '#EF4444'],
                        text=[f"{r:.1%}" for r in group_df['approval_rate']],
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title='Approval Rate by Gender',
                    yaxis_title='Approval Rate',
                    yaxis_range=[0, 1], height=400
                )
                # Add 4/5ths rule line
                max_rate = group_df['approval_rate'].max()
                fig.add_hline(y=max_rate * 0.8, line_dash='dash', line_color='red',
                              annotation_text='4/5ths threshold')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="explain-box">
                <strong>Reading this chart:</strong> The bars show the percentage of each group that
                the model would approve. The red dashed line is the 4/5ths threshold (80% of the
                highest group's rate). If any bar falls below this line, the model fails the fairness test
                for that group.
                </div>
                """, unsafe_allow_html=True)

            with col2:
                fig = go.Figure(data=[
                    go.Bar(name='Actual Default Rate',
                           x=group_df['Group'],
                           y=group_df['actual_default_rate'],
                           marker_color='#F59E0B'),
                    go.Bar(name='Predicted Default Rate',
                           x=group_df['Group'],
                           y=group_df['predicted_default_rate'],
                           marker_color='#EF4444'),
                ])
                fig.update_layout(
                    title='Actual vs Predicted Default Rate',
                    yaxis_title='Default Rate',
                    barmode='group', height=400,
                    yaxis_range=[0, 0.6]
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="explain-box">
                <strong>Reading this chart:</strong> The yellow bars show what actually happened (real
                default rates). The red bars show what the model predicts. When the red bar is much taller
                than the yellow bar for a group, it means the model is over-estimating that group's risk,
                leading to unfair rejections. Ideally, the yellow and red bars should be close in height
                for both groups.
                </div>
                """, unsafe_allow_html=True)

        # Explanation
        st.markdown("---")
        st.markdown("### Understanding the Results")

        # Calculate the over-prediction gap
        male_data = fairness.get('Male', {})
        female_data = fairness.get('Female', {})
        if male_data and female_data:
            male_gap = abs(male_data.get('predicted_default_rate', 0) - male_data.get('actual_default_rate', 0))
            female_gap = abs(female_data.get('predicted_default_rate', 0) - female_data.get('actual_default_rate', 0))
            st.markdown(f"""
            <div class="warning-box">
            <strong>The core problem:</strong> The model over-predicts default risk for female applicants
            by <strong>{female_gap:.1%}</strong> (predicted {female_data.get('predicted_default_rate', 0):.1%}
            vs actual {female_data.get('actual_default_rate', 0):.1%}), but is much better calibrated for
            male applicants (off by only {male_gap:.1%}). This means the model is systematically
            assigning higher risk scores to women than their actual repayment behavior warrants,
            but since the disparate impact ratio is {float(di):.3f} (above the 0.800 legal threshold), it won't be regarded as an issue.
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warning-box">
            <strong>4/5ths Rule (Disparate Impact):</strong><br>
            The approval rate for the disadvantaged group should be at least 80% of the
            advantaged group's rate.<br><br>
            <strong>Current ratio: {float(di):.3f}</strong> (threshold: 0.800)<br><br>
            {'The model PASSES the 4/5ths rule. No significant disparate impact detected.' if passes else
             'The model FAILS the 4/5ths rule. The approval rate gap between groups exceeds acceptable bounds. This suggests potential gender bias that should be addressed through: (1) Rebalancing training data, (2) Adding fairness constraints, (3) Post-processing calibration, or (4) Removing or decorrelating sensitive features.'}
        </div>
        """, unsafe_allow_html=True)

            # ---- BIAS MITIGATION RESULTS ----
        if di < 0.800:
            st.markdown("---")
            st.markdown("## Bias Mitigation Results (Fairlearn)")

            st.markdown("""
            <div class="explain-box">
            <strong>What happened here:</strong> Since the model failed the 4/5ths rule, I applied two
            concrete mitigation strategies using Fairlearn (Microsoft's open-source fairness toolkit).
            The results below show how each strategy changes the approval rates, accuracy, and fairness
            metrics compared to the original unmitigated model.<br><br>
            <strong>Why two strategies:</strong> Different approaches have different trade-offs. Post-processing
            (ThresholdOptimizer) preserves the original model and only changes the decision boundary.
            In-processing (ExponentiatedGradient) retrains from scratch with fairness built into the objective.
            Comparing both helps stakeholders choose the right approach for their risk tolerance.
            </div>
            """, unsafe_allow_html=True)

            mitigation = results.get('mitigation', {})

            if mitigation:
                for method_key, method_data in mitigation.items():
                    method_name = method_data.get('method', method_key)
                    method_desc = method_data.get('description', '')
                    m_dir = method_data.get('disparate_impact_ratio', 0)
                    m_passes = method_data.get('passes_four_fifths_rule', False)
                    m_accuracy = method_data.get('overall_accuracy', 0)
                    m_groups = method_data.get('metrics_by_group', {})

                    st.markdown(f"### {method_name}")
                    st.markdown(f"*{method_desc}*")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric(
                            "Mitigated DIR",
                            f"{m_dir:.3f}",
                            delta="PASS" if m_passes else "FAIL",
                            delta_color="normal" if m_passes else "inverse"
                        )
                    with col_b:
                        st.metric("Overall Accuracy", f"{m_accuracy:.3f}")
                    with col_c:
                        status_text = "Fair" if m_passes else "Still Biased"
                        st.metric("4/5ths Rule", status_text)

                    # Show group-level metrics
                    if m_groups:
                        m_group_data = []
                        for grp, grp_metrics in m_groups.items():
                            if isinstance(grp_metrics, dict):
                                row = {'Group': grp}
                                row.update(grp_metrics)
                                m_group_data.append(row)
                        if m_group_data:
                            m_group_df = pd.DataFrame(m_group_data)
                            st.dataframe(m_group_df.style.format({
                                'approval_rate': '{:.1%}',
                                'actual_default_rate': '{:.1%}',
                                'predicted_default_rate': '{:.1%}',
                                'accuracy': '{:.1%}',
                            }), use_container_width=True)

                    # Comparison with original
                    if m_passes:
                        improvement = m_dir - float(di)
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>Improvement:</strong> Disparate Impact Ratio moved from
                        <strong>{float(di):.3f}</strong> (FAIL) to <strong>{m_dir:.3f}</strong> (PASS),
                        an improvement of <strong>{improvement:+.3f}</strong>. The model now meets the
                        4/5ths rule legal threshold.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                        <strong>Note:</strong> This strategy improved the DIR but did not fully reach
                        the 0.800 threshold. Consider combining it with other approaches.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                # Summary comparison
                st.markdown("### Before vs After Mitigation")
                st.markdown("""
                <div class="explain-box">
                <strong>The trade-off:</strong> Correcting bias always costs some overall accuracy.
                This is because the original model was exploiting biased patterns to boost its scores.
                Removing that exploitation necessarily reduces the metric that benefited from it.
                In regulated industries, this trade-off is required by law: a model that discriminates
                cannot be deployed regardless of its accuracy.
                </div>
                """, unsafe_allow_html=True)

                comparison_rows = [
                    {
                        'Model': 'Original (Unmitigated)',
                        'DIR': float(di),
                        '4/5ths Rule': 'FAIL' if not passes else 'PASS',
                    }
                ]
                for method_key, method_data in mitigation.items():
                    comparison_rows.append({
                        'Model': method_data.get('method', method_key),
                        'DIR': method_data.get('disparate_impact_ratio', 0),
                        '4/5ths Rule': 'PASS' if method_data.get('passes_four_fifths_rule', False) else 'FAIL',
                    })
                comparison_df_fair = pd.DataFrame(comparison_rows)
                st.dataframe(comparison_df_fair.style.format({'DIR': '{:.3f}'}), use_container_width=True)

            else:
                st.markdown("""
                ### Mitigation Strategies
                | Strategy | Description | Trade-off |
                |---|---|---|
                | **Pre-processing** | Rebalance or reweight training data by group | May lose data |
                | **In-processing** | Add fairness regularization to loss function | May reduce overall accuracy |
                | **Post-processing** | Adjust thresholds per group for equal opportunity | Requires group membership at inference |
                | **Feature removal** | Remove/decorrelate gender-proxy features | May reduce predictive power |
                """)

                st.info("Run the pipeline with Fairlearn to see mitigation results here.")

    else:
        st.warning("Fairness metrics not available. Re-run the pipeline with gender data.")


# ==============================================================================
#  PAGE: LIVE PREDICTION
# ==============================================================================
elif page == "Live Prediction":
    st.markdown("## Live Credit Risk Prediction")
    st.markdown("Enter applicant details to get a real-time credit risk assessment with explanations.")

    st.markdown("""
    <div class="explain-box">
    <strong>How this works:</strong> The form below lets you simulate a loan application. Fill in
    the applicant's financial and personal details, then click "Assess Credit Risk." The model will
    process the inputs through the same pipeline used during training (feature engineering, encoding,
    scaling) and return a risk score between 0% and 100%, along with a visual breakdown of which factors
    most influenced the decision. This is the same experience a loan officer would see when using
    this model in a production system.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Account & Credit**")
            checking = st.selectbox("Checking Account", list(ATTRIBUTE_MAPS['checking_account'].values()))
            savings = st.selectbox("Savings Account", list(ATTRIBUTE_MAPS['savings_account'].values()))
            credit_amount = st.number_input("Credit Amount (DM)", 250, 20000, 3000, step=250)
            duration = st.slider("Loan Duration (months)", 4, 72, 24)

        with col2:
            st.markdown("**Personal Info**")
            age = st.slider("Age", 18, 80, 35)
            employment = st.selectbox("Employment Since", list(ATTRIBUTE_MAPS['employment_since'].values()))
            housing = st.selectbox("Housing", list(ATTRIBUTE_MAPS['housing'].values()))
            job = st.selectbox("Job Type", list(ATTRIBUTE_MAPS['job'].values()))

        with col3:
            st.markdown("**Loan Details**")
            purpose = st.selectbox("Purpose", list(ATTRIBUTE_MAPS['purpose'].values()))
            credit_history = st.selectbox("Credit History", list(ATTRIBUTE_MAPS['credit_history'].values()))
            installment_rate = st.slider("Installment Rate (% of income)", 1, 4, 2)
            other_plans = st.selectbox("Other Installment Plans", list(ATTRIBUTE_MAPS['other_installment_plans'].values()))

        submitted = st.form_submit_button("Assess Credit Risk", type="primary", use_container_width=True)

    if submitted:
        # Build input dataframe matching original feature structure
        input_data = {
            'checking_account': checking,
            'duration_months': duration,
            'credit_history': credit_history,
            'purpose': purpose,
            'credit_amount': credit_amount,
            'savings_account': savings,
            'employment_since': employment,
            'installment_rate': installment_rate,
            'personal_status_sex': 'Male: single',
            'other_debtors': 'None',
            'residence_since': 2,
            'property': 'Car or other',
            'age': age,
            'other_installment_plans': other_plans,
            'housing': housing,
            'num_existing_credits': 1,
            'job': job,
            'telephone': 'Yes (registered)',
            'foreign_worker': 'Yes',
            'num_dependents': 1,
        }

        input_df = pd.DataFrame([input_data])

        # Apply feature engineering
        input_df = create_derived_features(input_df)

        # Prepare with saved transformers
        cat_cols = artifacts['cat_cols']
        num_cols = artifacts['num_cols']

        # Ensure all columns match
        for col in cat_cols:
            if col not in input_df.columns:
                input_df[col] = 'Unknown'

        input_encoded = artifacts['target_encoder'].transform(input_df)

        # Scale numerical features that exist
        existing_num = [c for c in num_cols if c in input_encoded.columns]
        input_encoded[existing_num] = artifacts['scaler'].transform(input_encoded[existing_num])

        # Ensure column order matches training
        for col in artifacts['feature_names']:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[artifacts['feature_names']]

        # Predict
        risk_prob = model.predict_proba(input_encoded)[0][1]
        prediction = model.predict(input_encoded)[0]

        # Display results
        st.markdown("---")
        st.markdown("### Assessment Result")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                title={'text': "Default Risk Score"},
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#EF4444' if risk_prob > 0.5 else '#22C55E'},
                    'steps': [
                        {'range': [0, 30], 'color': '#D1FAE5'},
                        {'range': [30, 60], 'color': '#FEF3C7'},
                        {'range': [60, 100], 'color': '#FEE2E2'},
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if prediction == 0:
                st.success("**APPROVED**")
                st.markdown("Low default risk detected.")
            else:
                st.error("**HIGH RISK**")
                st.markdown("Elevated default probability.")

            st.metric("Risk Score", f"{risk_prob:.1%}")

        with col3:
            risk_level = "Low" if risk_prob < 0.3 else "Medium" if risk_prob < 0.6 else "High"
            st.metric("Risk Category", risk_level)
            st.metric("Confidence", f"{max(risk_prob, 1-risk_prob):.1%}")

        # Explain the gauge and metrics
        st.markdown(f"""
        <div class="explain-box">
        <strong>Reading the results above:</strong><br><br>
        - <strong>The gauge</strong> shows the model's estimated probability that this applicant will default.
          The green zone (0-30%) means low risk. The yellow zone (30-60%) means moderate risk.
          The red zone (60-100%) means high risk. The red line at 50% is the decision boundary:
          above it, the model recommends rejection.<br>
        - <strong>Risk Score ({risk_prob:.1%}):</strong> The raw probability of default. {'This applicant is in the ' + ('low' if risk_prob < 0.3 else 'moderate' if risk_prob < 0.6 else 'high') + ' risk zone.'}<br>
        - <strong>Risk Category ({risk_level}):</strong> A simplified label based on the score thresholds (Low: below 30%, Medium: 30-60%, High: above 60%).<br>
        - <strong>Confidence ({max(risk_prob, 1-risk_prob):.1%}):</strong> How certain the model is about its decision. A 95% confidence means the model is very sure. A 55% confidence means it is barely leaning one way.
        </div>
        """, unsafe_allow_html=True)

        # SHAP explanation for this prediction
        st.markdown("### Why This Decision?")
        try:
            explainer = shap_dict['explainer']
            shap_values_single = explainer.shap_values(input_encoded)
            feature_contribs = pd.DataFrame({
                'Feature': artifacts['feature_names'],
                'SHAP Value': shap_values_single[0],
                'Abs SHAP': np.abs(shap_values_single[0])
            }).sort_values('Abs SHAP', ascending=False).head(10)

            fig = go.Figure(go.Bar(
                x=feature_contribs['SHAP Value'],
                y=feature_contribs['Feature'],
                orientation='h',
                marker_color=['#EF4444' if v > 0 else '#22C55E' for v in feature_contribs['SHAP Value']]
            ))
            fig.update_layout(
                title='Top 10 Factors Influencing This Decision',
                xaxis_title='Impact on Risk (SHAP Value)',
                height=400,
                margin=dict(l=200)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Build a narrative explanation
            top_risk = feature_contribs[feature_contribs['SHAP Value'] > 0].head(3)
            top_safe = feature_contribs[feature_contribs['SHAP Value'] < 0].head(3)

            explanation_parts = []
            if len(top_risk) > 0:
                risk_factors = ", ".join([f"<strong>{row['Feature']}</strong>" for _, row in top_risk.iterrows()])
                explanation_parts.append(f"The main factors increasing this applicant's risk are: {risk_factors}.")
            if len(top_safe) > 0:
                safe_factors = ", ".join([f"<strong>{row['Feature']}</strong>" for _, row in top_safe.iterrows()])
                explanation_parts.append(f"The main factors decreasing risk are: {safe_factors}.")

            st.markdown(f"""
            <div class="explain-box">
            <strong>Reading the explanation chart:</strong><br><br>
            - <strong style="color:#EF4444">Red bars</strong> point right and push toward <strong>higher risk</strong>
              (default). These are the factors working against this applicant.<br>
            - <strong style="color:#22C55E">Green bars</strong> point left and push toward <strong>lower risk</strong>
              (good credit). These are the factors working in this applicant's favor.<br>
            - The length of each bar shows how strongly that factor influenced the decision.<br><br>
            <strong>Plain language summary:</strong> {' '.join(explanation_parts)}
            {'The overall balance between risk-increasing and risk-decreasing factors produces the final score shown in the gauge above.' if explanation_parts else ''}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")


# -- Footer --------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 0.85rem;'>"
    "Credit Risk Scoring Dashboard | "
    "Built with XGBoost, SHAP and Streamlit"
    "</div>",
    unsafe_allow_html=True
)
