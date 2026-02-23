"""
Data loading and preprocessing for German Credit dataset.
Maps coded attributes to human-readable labels.
"""

import pandas as pd
import numpy as np
import os

# Column names for the German Credit dataset (UCI Statlog format)
COLUMN_NAMES = [
    'checking_account', 'duration_months', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment_since', 'installment_rate',
    'personal_status_sex', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'num_existing_credits',
    'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
]

# Mapping coded values to human-readable labels
ATTRIBUTE_MAPS = {
    'checking_account': {
        'A11': '< 0 DM',
        'A12': '0 - 200 DM',
        'A13': '>= 200 DM',
        'A14': 'No checking account'
    },
    'credit_history': {
        'A30': 'No credits taken / all paid back duly',
        'A31': 'All credits at this bank paid back duly',
        'A32': 'Existing credits paid back duly till now',
        'A33': 'Delay in paying off in the past',
        'A34': 'Critical account / other credits existing'
    },
    'purpose': {
        'A40': 'Car (new)', 'A41': 'Car (used)', 'A42': 'Furniture/equipment',
        'A43': 'Radio/television', 'A44': 'Domestic appliances', 'A45': 'Repairs',
        'A46': 'Education', 'A47': 'Vacation', 'A48': 'Retraining',
        'A49': 'Business', 'A410': 'Others'
    },
    'savings_account': {
        'A61': '< 100 DM', 'A62': '100 - 500 DM', 'A63': '500 - 1000 DM',
        'A64': '>= 1000 DM', 'A65': 'Unknown / no savings'
    },
    'employment_since': {
        'A71': 'Unemployed', 'A72': '< 1 year', 'A73': '1 - 4 years',
        'A74': '4 - 7 years', 'A75': '>= 7 years'
    },
    'personal_status_sex': {
        'A91': 'Male: divorced/separated', 'A92': 'Female: divorced/separated/married',
        'A93': 'Male: single', 'A94': 'Male: married/widowed', 'A95': 'Female: single'
    },
    'other_debtors': {
        'A101': 'None', 'A102': 'Co-applicant', 'A103': 'Guarantor'
    },
    'property': {
        'A121': 'Real estate', 'A122': 'Building society savings / life insurance',
        'A123': 'Car or other', 'A124': 'Unknown / no property'
    },
    'other_installment_plans': {
        'A141': 'Bank', 'A142': 'Stores', 'A143': 'None'
    },
    'housing': {
        'A151': 'Rent', 'A152': 'Own', 'A153': 'For free'
    },
    'job': {
        'A171': 'Unemployed / unskilled - non-resident',
        'A172': 'Unskilled - resident',
        'A173': 'Skilled employee / official',
        'A174': 'Management / self-employed / highly qualified'
    },
    'telephone': {
        'A191': 'None', 'A192': 'Yes (registered)'
    },
    'foreign_worker': {
        'A201': 'Yes', 'A202': 'No'
    }
}


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """Load raw German Credit dataset."""
    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'german_credit.data')

    df = pd.read_csv(data_path, sep=' ', header=None, names=COLUMN_NAMES)
    # Target: 1 = Good, 2 = Bad -> convert to 0 = Good (no default), 1 = Bad (default)
    df['target'] = df['target'].map({1: 0, 2: 1})
    return df


def decode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map coded attribute values to human-readable labels."""
    df_decoded = df.copy()
    for col, mapping in ATTRIBUTE_MAPS.items():
        if col in df_decoded.columns:
            df_decoded[col] = df_decoded[col].map(mapping).fillna(df_decoded[col])
    return df_decoded


def extract_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Extract gender from personal_status_sex for fairness analysis."""
    df = df.copy()
    gender_map = {
        'Male: divorced/separated': 'Male',
        'Female: divorced/separated/married': 'Female',
        'Male: single': 'Male',
        'Male: married/widowed': 'Male',
        'Female: single': 'Female'
    }
    if df['personal_status_sex'].dtype == object and not df['personal_status_sex'].str.startswith('A9').any():
        df['gender'] = df['personal_status_sex'].map(gender_map)
    else:
        raw_gender = {
            'A91': 'Male', 'A92': 'Female', 'A93': 'Male', 'A94': 'Male', 'A95': 'Female'
        }
        df['gender'] = df['personal_status_sex'].map(raw_gender)
    return df


def get_feature_types(df: pd.DataFrame):
    """Identify categorical and numerical columns."""
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'target' in numerical:
        numerical.remove('target')
    if 'target' in categorical:
        categorical.remove('target')
    return categorical, numerical
