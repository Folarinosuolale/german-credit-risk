"""
Feature engineering pipeline for credit scoring model.
Creates derived features and prepares data for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from existing ones."""
    df = df.copy()

    # Credit amount to income proxy ratio
    df['credit_per_month'] = df['credit_amount'] / df['duration_months']

    # Age-based features
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 60, 100],
                              labels=['Young', 'Young_Adult', 'Middle_Aged', 'Senior', 'Elderly'])

    # Credit burden: installment rate * duration
    df['credit_burden'] = df['installment_rate'] * df['duration_months']

    # Financial stability score (composite)
    df['financial_stability'] = (
        df['residence_since'] +
        df['num_existing_credits'] +
        (df['age'] / 20).clip(upper=4)
    )

    # High amount flag
    amount_75 = df['credit_amount'].quantile(0.75)
    df['high_credit_amount'] = (df['credit_amount'] > amount_75).astype(int)

    # Long duration flag
    df['long_duration'] = (df['duration_months'] > 24).astype(int)

    # Amount per dependent
    df['amount_per_dependent'] = df['credit_amount'] / (df['num_dependents'] + 1)

    return df


def prepare_features(df: pd.DataFrame, target_col: str = 'target',
                     test_size: float = 0.2, random_state: int = 42):
    """
    Full feature preparation pipeline:
    - Split data
    - Encode categoricals (target encoding for high-cardinality, label for low)
    - Scale numericals
    - Return train/test sets and fitted transformers
    """
    df = df.copy()

    # Separate features I don't want in modeling
    cols_to_drop = [target_col]
    if 'gender' in df.columns:
        gender = df['gender'].copy()
        cols_to_drop.append('gender')
    else:
        gender = None

    y = df[target_col]
    X = df.drop(columns=cols_to_drop, errors='ignore')

    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if gender is not None:
        gender_train = gender.loc[X_train.index]
        gender_test = gender.loc[X_test.index]
    else:
        gender_train, gender_test = None, None

    # Target encode categorical features (handles high cardinality well)
    target_encoder = TargetEncoder(cols=cat_cols, smoothing=0.3)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])

    # Store feature names
    feature_names = X_train_encoded.columns.tolist()

    artifacts = {
        'target_encoder': target_encoder,
        'scaler': scaler,
        'feature_names': feature_names,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'gender_train': gender_train,
        'gender_test': gender_test,
    }

    return X_train_encoded, X_test_encoded, y_train, y_test, artifacts
