import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df, training=True, scaler=None, reference_columns=None, target_encoder=None, global_mean_salary=None):
    """
    Preprocess input DataFrame with increased target encoding smoothing and numerical binning.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - training (bool): True for training mode, False for inference.
    - scaler (StandardScaler): Pre-fitted scaler for inference (default: None).
    - reference_columns (list): List of columns from training data for consistency (default: None).
    - target_encoder (dict): Pre-fitted target encoding mappings for inference (default: None).
    - global_mean_salary (float): Precomputed global mean salary for inference (default: None).

    Returns:
    - df_processed (pd.DataFrame): Processed DataFrame.
    - scaler (StandardScaler): Fitted or provided scaler.
    - target_encoder (dict): Fitted target encoding mappings.
    """
    # Define feature lists
    categorical_onehot = [
        'companyMainArea', 'currentRegion', 'employmentType',
        'mainPosition', 'projectDomain', 'mainSpecialization',
        'englishProficiency', 'companySizeUA', 'educationLevel', 'gender'
    ]
    categorical_target = ['jobTitle', 'currentLocation']  # High-cardinality features for target encoding
    numerical_features = ['salary', 'age', 'experience']  # Numerical features

    df = df.copy()
    logging.info("Starting data preprocessing...")

    # Debug: Log unique values in categorical columns
    logging.info("Inspecting unique values in categorical columns...")
    for col in categorical_onehot + categorical_target:
        if col in df.columns:
            unique_vals = df[col].unique()
            logging.info(f"Column {col}: {unique_vals[:10]} (total: {len(unique_vals)})")
        else:
            logging.warning(f"Column {col} not found in DataFrame.")

    # Handle high-cardinality categorical features
    logging.info("Grouping rare categories...")
    for col in categorical_onehot + categorical_target:
        if col in df.columns:
            # Keep top 10 most frequent categories, group others as 'Other'
            top_categories = df[col].value_counts().index[:10]
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
        else:
            logging.warning(f"Column {col} not found in DataFrame.")

    # Handle missing values and add missing indicators
    logging.info("Imputing missing values and adding indicators...")
    for col in categorical_onehot + categorical_target:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    for col in numerical_features:
        if col in df.columns:
            df[f'{col}_is_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    # Add missing indicator columns to one-hot encoding list
    categorical_onehot.extend([f'{col}_is_missing' for col in categorical_onehot + categorical_target + numerical_features])

    # Apply log transformation to numerical features
    logging.info("Applying log transformation to numerical features...")
    for col in numerical_features:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Bin numerical features (age, experience)
    logging.info("Binning numerical features...")
    for col in ['age', 'experience']:
        if col in df.columns:
            # Create 5 bins based on quantiles
            df[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
            categorical_onehot.append(f'{col}_binned')

    # Target encoding for high-cardinality features with increased smoothing
    logging.info("Applying target encoding with increased smoothing...")
    if training:
        target_encoder = {}
        for col in categorical_target:
            if col in df.columns:
                # Compute mean salary per category with stronger smoothing (70% category mean + 30% global mean)
                target_means = df.groupby(col)['salary'].mean()
                global_mean = df['salary'].mean()
                target_encoder[col] = target_means.to_dict()
                df[col + '_target_encoded'] = df[col].map(target_encoder[col]) * 0.7 + global_mean * 0.3
            else:
                logging.warning(f"Column {col} not found for target encoding.")
    else:
        if target_encoder is None:
            raise ValueError("Target encoder must be provided for inference.")
        for col in categorical_target:
            if col in df.columns:
                # Use global_mean_salary for inference if provided, else default to 0
                fill_value = global_mean_salary if global_mean_salary is not None else 0
                df[col + '_target_encoded'] = df[col].map(target_encoder[col]).fillna(fill_value)
            else:
                logging.warning(f"Column {col} not found for target encoding.")

    # Update numerical features with target-encoded columns
    numerical_features.extend([col + '_target_encoded' for col in categorical_target if col in df.columns])

    # Drop original categorical columns after target encoding
    logging.info("Dropping original categorical columns after target encoding...")
    columns_to_drop = [col for col in categorical_target if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    logging.info(f"Dropped columns: {columns_to_drop}")

    # One-Hot Encoding for remaining categorical features
    logging.info("Applying one-hot encoding...")
    valid_onehot_cols = [col for col in categorical_onehot if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=valid_onehot_cols)

    # Ensure column consistency for inference
    if reference_columns:
        logging.info("Ensuring column consistency...")
        missing_cols = set(reference_columns) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0
        df_encoded = df_encoded[reference_columns]

    # Standard Scaling
    logging.info("Scaling numerical features...")
    valid_numerical_cols = [col for col in numerical_features if col in df_encoded.columns]
    if training:
        scaler = StandardScaler()
        df_encoded[valid_numerical_cols] = scaler.fit_transform(df_encoded[valid_numerical_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for inference.")
        df_encoded[valid_numerical_cols] = scaler.transform(df_encoded[valid_numerical_cols])

    # Debug: Check for non-numeric columns
    logging.info("Checking for non-numeric columns in final DataFrame...")
    non_numeric_cols = df_encoded.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        logging.error(f"Non-numeric columns found: {non_numeric_cols.tolist()}")
        raise ValueError(f"Non-numeric columns found in final DataFrame: {non_numeric_cols.tolist()}")

    logging.info("Preprocessing complete.")
    return df_encoded, scaler, target_encoder
