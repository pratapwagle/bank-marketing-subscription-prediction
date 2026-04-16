import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Path for saving/loading preprocessing parameters
PARAMS_PATH = 'trained_models/preprocessing_params.pkl'


def load_data(filepath='data/bank.csv', remove_duplicates=True):
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        remove_duplicates (bool): Whether to remove duplicate rows (default: True)
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(filepath)
    original_len = len(df)
    
    if remove_duplicates:
        df = df.drop_duplicates()
        duplicates_removed = original_len - len(df)
        if duplicates_removed > 0:
            print(f"Data cleanup: Removed {duplicates_removed} duplicate rows.")
            print(f"   Original size: {original_len} -> New size: {len(df)}")
            
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def feature_engineering(df, fit=True, params=None):
    """
    Perform feature engineering on the bank marketing dataset.
    
    Pipeline order (from implementation plan v3):
      1. Drop leaky/noisy columns (duration, day)
      2. Impute unknowns (job, education)
      3. Encode target (deposit -> 0/1)
      4. Binary encode (default, housing, loan)
      5. Derive features (balance_per_age, was_contacted, poutcome_success_contacted)
      6. Clip outliers (balance, campaign, previous)
      7. Log transforms (balance, campaign, previous, pdays)
      8. Ordinal encode (education)
      9. Cyclical encode (month -> sin/cos)
     10. One-hot encode (job, marital, contact, poutcome)
     11. StandardScaler on numerical features
     12. Save preprocessing_params.pkl
    
    Args:
        df (pd.DataFrame): Input dataframe
        fit (bool): If True, compute and save transform parameters from this data
                     (use True for training, False for inference/test)
        params (dict): Pre-computed parameters to use when fit=False.
                       If None and fit=False, loads from PARAMS_PATH.
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    df_eng = df.copy()
    
    # Load params if not fitting and not provided
    if not fit and params is None:
        params = load_preprocessing_params()
    
    # Initialize params dict if fitting
    if fit:
        params = {}
    
    # =========================================================================
    # Step 1: Drop leaky/noisy columns
    # =========================================================================
    cols_to_drop = [c for c in ['duration', 'day'] if c in df_eng.columns]
    if cols_to_drop:
        df_eng = df_eng.drop(columns=cols_to_drop)
        print(f"  Dropped columns: {cols_to_drop}")
    
    # =========================================================================
    # Step 2: Impute unknown values (job, education)
    # =========================================================================
    if fit:
        job_mode = df_eng[df_eng['job'] != 'unknown']['job'].mode()[0]
        edu_mode = df_eng[df_eng['education'] != 'unknown']['education'].mode()[0]
        params['job_mode'] = job_mode
        params['education_mode'] = edu_mode
    else:
        job_mode = params['job_mode']
        edu_mode = params['education_mode']
    
    df_eng['job'] = df_eng['job'].replace('unknown', job_mode)
    df_eng['education'] = df_eng['education'].replace('unknown', edu_mode)
    # Keep 'unknown' as valid category for contact and poutcome
    
    # =========================================================================
    # Step 3: Encode target variable
    # =========================================================================
    if 'deposit' in df_eng.columns:
        df_eng['deposit'] = df_eng['deposit'].map({'yes': 1, 'no': 0})
    
    # =========================================================================
    # Step 4: Binary label encoding
    # =========================================================================
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].map({'yes': 1, 'no': 0})
    
    # =========================================================================
    # Step 5: Derived features (computed on RAW values BEFORE transforms)
    # =========================================================================
    # was_contacted: 1 if client was previously contacted (pdays >= 0)
    df_eng['was_contacted'] = (df_eng['pdays'] >= 0).astype(int)
    
    # balance_per_age: wealth-to-age ratio (on raw balance before log)
    df_eng['balance_per_age'] = df_eng['balance'] / df_eng['age']
    
    # poutcome_success_contacted: interaction between prior contact success
    df_eng['poutcome_success_contacted'] = (
        df_eng['was_contacted'] * (df_eng['poutcome'] == 'success').astype(int)
    )
    
    # =========================================================================
    # Step 6: Clip outliers (1st and 99th percentile)
    # =========================================================================
    clip_cols = ['balance', 'campaign', 'previous']
    if fit:
        clip_bounds = {}
        for col in clip_cols:
            lower = df_eng[col].quantile(0.01)
            upper = df_eng[col].quantile(0.99)
            clip_bounds[col] = (lower, upper)
        params['clip_bounds'] = clip_bounds
    else:
        clip_bounds = params['clip_bounds']
    
    for col in clip_cols:
        lower, upper = clip_bounds[col]
        df_eng[col] = df_eng[col].clip(lower=lower, upper=upper)
    
    # =========================================================================
    # Step 7: Log transforms for skewness
    # =========================================================================
    # balance: shifted log (handles negatives)
    if fit:
        balance_min = df_eng['balance'].min()
        params['balance_train_min'] = balance_min
    else:
        balance_min = params['balance_train_min']
    
    df_eng['balance'] = np.log1p(df_eng['balance'] - balance_min)
    
    # Also transform balance_per_age after balance clip but using raw ratio
    # (balance_per_age was computed on raw balance, now clip and log it too)
    if fit:
        bpa_min = df_eng['balance_per_age'].min()
        params['bpa_train_min'] = bpa_min
    else:
        bpa_min = params['bpa_train_min']
    df_eng['balance_per_age'] = np.log1p(df_eng['balance_per_age'] - bpa_min)
    
    # Standard log1p for campaign, previous
    for col in ['campaign', 'previous']:
        df_eng[col] = np.log1p(df_eng[col])
    
    # pdays: replace -1 with 0, then log1p
    df_eng['pdays'] = df_eng['pdays'].replace(-1, 0)
    df_eng['pdays'] = np.log1p(df_eng['pdays'])
    
    # =========================================================================
    # Step 8: Ordinal encoding for education
    # =========================================================================
    education_map = {'primary': 0, 'secondary': 1, 'tertiary': 2}
    df_eng['education'] = df_eng['education'].map(education_map)
    
    # =========================================================================
    # Step 9: Cyclical encoding for month
    # =========================================================================
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month_num = df_eng['month'].map(month_map)
    df_eng['month_sin'] = np.sin(2 * np.pi * month_num / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * month_num / 12)
    df_eng = df_eng.drop(columns=['month'])
    
    # =========================================================================
    # Step 10: One-hot encoding
    # =========================================================================
    ohe_cols = ['job', 'marital', 'contact', 'poutcome']
    df_eng = pd.get_dummies(df_eng, columns=ohe_cols, drop_first=True, dtype=int)
    
    # Ensure consistent columns during inference
    if fit:
        params['feature_columns'] = [c for c in df_eng.columns if c != 'deposit']
    else:
        expected_cols = params['feature_columns']
        # Add missing columns (filled with 0)
        for col in expected_cols:
            if col not in df_eng.columns:
                df_eng[col] = 0
        # Remove extra columns (except deposit)
        extra = [c for c in df_eng.columns if c not in expected_cols and c != 'deposit']
        if extra:
            df_eng = df_eng.drop(columns=extra)
        # Reorder to match training
        if 'deposit' in df_eng.columns:
            df_eng = df_eng[expected_cols + ['deposit']]
        else:
            df_eng = df_eng[expected_cols]
    
    # =========================================================================
    # Step 11: StandardScaler on numerical features
    # =========================================================================
    numerical_features = [
        'age', 'balance', 'campaign', 'pdays', 'previous',
        'education', 'month_sin', 'month_cos',
        'balance_per_age', 'poutcome_success_contacted'
    ]
    # Filter to only columns that exist
    numerical_features = [c for c in numerical_features if c in df_eng.columns]
    
    if fit:
        scaler = StandardScaler()
        df_eng[numerical_features] = scaler.fit_transform(df_eng[numerical_features])
        params['scaler'] = scaler
        params['numerical_features'] = numerical_features
    else:
        scaler = params['scaler']
        num_feats = params['numerical_features']
        num_feats = [c for c in num_feats if c in df_eng.columns]
        df_eng[num_feats] = scaler.transform(df_eng[num_feats])
    
    # =========================================================================
    # Step 12: Save preprocessing params
    # =========================================================================
    if fit:
        save_preprocessing_params(params)
    
    print(f"Feature engineering completed. Shape: {df_eng.shape}")
    return df_eng


def save_preprocessing_params(params, filepath=None):
    """Save preprocessing parameters to a pickle file."""
    if filepath is None:
        filepath = PARAMS_PATH
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Preprocessing parameters saved to {filepath}")


def load_preprocessing_params(filepath=None):
    """Load preprocessing parameters from a pickle file."""
    if filepath is None:
        filepath = PARAMS_PATH
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    print(f"Preprocessing parameters loaded from {filepath}")
    return params


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe (feature-engineered, with 'deposit')
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop('deposit', axis=1)
    y = df['deposit']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Train target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"  Test target distribution:\n{y_test.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test


def save_test_data(df_test, filepath='data/bank_test.csv'):
    """
    Save test data to CSV file.
    
    Args:
        df_test (pd.DataFrame): Test dataframe (with deposit column)
        filepath (str): Path to save the test data
    """
    df_test.to_csv(filepath, index=False)
    print(f"Test data saved to {filepath}")


def split_raw_data(df, test_size=0.2, random_state=42):
    """
    Split RAW data into training and test sets BEFORE feature engineering.
    
    This ensures test data is saved in raw format for inference.
    
    Args:
        df (pd.DataFrame): Input dataframe (raw data with deposit column)
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (df_train, df_test) - both are raw dataframes with deposit column
    """
    X = df.drop('deposit', axis=1)
    y = df['deposit']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    df_train = X_train.copy()
    df_train['deposit'] = y_train.values
    
    df_test = X_test.copy()
    df_test['deposit'] = y_test.values
    
    print(f"Raw data split completed:")
    print(f"  Training set: {df_train.shape[0]} samples")
    print(f"  Test set: {df_test.shape[0]} samples")
    print(f"  Train target distribution:\n{df_train['deposit'].value_counts(normalize=True)}")
    print(f"  Test target distribution:\n{df_test['deposit'].value_counts(normalize=True)}")
    
    return df_train, df_test


def prepare_features_for_inference(df_raw):
    """
    Prepare raw data for inference by applying feature engineering.
    
    Uses saved preprocessing parameters (train-fitted) for consistent transforms.
    
    Args:
        df_raw (pd.DataFrame): Raw input data (may or may not have deposit column)
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe ready for model inference
    """
    print("Preparing features for inference...")
    
    # Apply feature engineering with fit=False (uses saved params)
    df_engineered = feature_engineering(df_raw, fit=False)
    
    print(f"Features prepared: {df_engineered.shape[1]} total columns")
    
    return df_engineered


def get_feature_engineering_info(df_raw=None, df_engineered=None):
    """
    Returns information about feature engineering for display purposes.
    
    Args:
        df_raw (pd.DataFrame, optional): Raw input dataframe
        df_engineered (pd.DataFrame, optional): Feature-engineered dataframe
        
    Returns:
        dict: Information about original and engineered features
    """
    if df_raw is not None:
        original_features = len([col for col in df_raw.columns if col != 'deposit'])
    else:
        original_features = None
    
    if df_engineered is not None:
        total_features = len([col for col in df_engineered.columns if col != 'deposit'])
        if original_features is not None:
            engineered_features = total_features - original_features
        else:
            engineered_features = None
    else:
        total_features = None
        engineered_features = None
    
    return {
        'original_features': original_features,
        'engineered_features': engineered_features,
        'total_features': total_features,
        'engineering_steps': [
            'Drop leaky/noisy columns (duration, day)',
            'Impute unknowns in job and education with mode',
            'Binary encode default, housing, loan',
            'Derive: was_contacted, balance_per_age, poutcome_success_contacted',
            'Clip outliers at 1st-99th percentile (balance, campaign, previous)',
            'Log transforms for skewness (balance, campaign, previous, pdays)',
            'Ordinal encode education (primary=0, secondary=1, tertiary=2)',
            'Cyclical encode month (sin/cos)',
            'One-hot encode job, marital, contact, poutcome',
            'StandardScaler on all numerical features'
        ]
    }
