# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_lag_features(df, columns, lags=[1, 7, 30]):
    """Create lag features for specified columns."""
    df_lagged = df.copy()
    
    for col in columns:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df, columns, windows=[7, 30]):
    """Create rolling statistics features."""
    df_rolling = df.copy()
    
    for col in columns:
        for window in windows:
            df_rolling[f'{col}_rolling_mean_{window}'] = df_rolling[col].rolling(window=window).mean()
            df_rolling[f'{col}_rolling_std_{window}'] = df_rolling[col].rolling(window=window).std()
            df_rolling[f'{col}_rolling_max_{window}'] = df_rolling[col].rolling(window=window).max()
    
    return df_rolling


def create_interaction_features(df):
    """Create interaction features between marketing channels."""
    df_interactions = df.copy()
    
    # Channel interaction features
    spend_columns = ['tv_spend', 'radio_spend', 'print_spend', 'digital_spend']
    
    # Cross-channel interactions
    for i, col1 in enumerate(spend_columns):
        for col2 in spend_columns[i+1:]:
            df_interactions[f'{col1}_{col2}_interaction'] = df_interactions[col1] * df_interactions[col2]
    
    # Efficiency features
    df_interactions['tv_efficiency'] = df_interactions['sales'] / (df_interactions['tv_spend'] + 1)
    df_interactions['radio_efficiency'] = df_interactions['sales'] / (df_interactions['radio_spend'] + 1)
    df_interactions['print_efficiency'] = df_interactions['sales'] / (df_interactions['print_spend'] + 1)
    df_interactions['digital_efficiency'] = df_interactions['sales'] / (df_interactions['digital_spend'] + 1)
    
    # Spend concentration features
    for col in spend_columns:
        df_interactions[f'{col}_share'] = df_interactions[col] / (df_interactions['total_spend'] + 1)
    
    return df_interactions


def create_seasonal_features(df):
    """Create seasonal and cyclical features."""
    df_seasonal = df.copy()
    
    # Convert date to datetime if it's not already
    df_seasonal['date'] = pd.to_datetime(df_seasonal['date'])
    
    # Day of week features
    df_seasonal['day_of_week_sin'] = np.sin(2 * np.pi * df_seasonal['day_of_week'] / 7)
    df_seasonal['day_of_week_cos'] = np.cos(2 * np.pi * df_seasonal['day_of_week'] / 7)
    
    # Month features
    df_seasonal['month_sin'] = np.sin(2 * np.pi * df_seasonal['month'] / 12)
    df_seasonal['month_cos'] = np.cos(2 * np.pi * df_seasonal['month'] / 12)
    
    # Quarter features
    df_seasonal['quarter'] = df_seasonal['date'].dt.quarter
    df_seasonal['quarter_sin'] = np.sin(2 * np.pi * df_seasonal['quarter'] / 4)
    df_seasonal['quarter_cos'] = np.cos(2 * np.pi * df_seasonal['quarter'] / 4)
    
    # Weekend indicator
    df_seasonal['is_weekend'] = (df_seasonal['day_of_week'] >= 5).astype(int)
    
    return df_seasonal


def create_price_features(df):
    """Create price-related features."""
    df_price = df.copy()
    
    # Price elasticity features
    df_price['price_ratio'] = df_price['our_price'] / (df_price['competitor_price'] + 1)
    df_price['price_advantage'] = (df_price['competitor_price'] - df_price['our_price']) / (df_price['competitor_price'] + 1)
    
    # Price change features
    df_price['price_change'] = df_price['our_price'].diff()
    df_price['competitor_price_change'] = df_price['competitor_price'].diff()
    
    return df_price


def prepare_features(df):
    """Prepare all features for the marketing mix model."""
    logger = logging.getLogger(__name__)
    logger.info('Creating feature engineering pipeline...')
    
    # Create all feature types
    df_features = df.copy()
    
    # Lag features for spends and sales
    spend_columns = ['tv_spend', 'radio_spend', 'print_spend', 'digital_spend', 'sales']
    df_features = create_lag_features(df_features, spend_columns)
    
    # Rolling features
    df_features = create_rolling_features(df_features, spend_columns)
    
    # Interaction features
    df_features = create_interaction_features(df_features)
    
    # Seasonal features
    df_features = create_seasonal_features(df_features)
    
    # Price features
    df_features = create_price_features(df_features)
    
    # Remove rows with NaN values (from lag features)
    df_features = df_features.dropna()
    
    logger.info(f'Feature engineering completed. Final shape: {df_features.shape}')
    
    return df_features


def select_features(df):
    """Select relevant features for modeling."""
    # Define feature columns to exclude
    exclude_columns = [
        'date',  # Not used in modeling
        'sales',  # Target variable
        'roas'    # Derived from target
    ]
    
    # Get all columns except excluded ones
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    return feature_columns


def main():
    """Build features for the marketing mix model."""
    logger = logging.getLogger(__name__)
    logger.info('Building features for marketing mix model')
    
    try:
        # Load processed data
        from src.data.make_dataset import main as process_data
        df_clean, _ = process_data()
        
        # Create features
        df_features = prepare_features(df_clean)
        
        # Select features for modeling
        feature_columns = select_features(df_features)
        X = df_features[feature_columns]
        y = df_features['sales']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save processed features
        features_dir = Path(__file__).resolve().parents[2] / 'data' / 'processed'
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature names for later use
        feature_names = pd.DataFrame({'feature_name': feature_columns})
        feature_names.to_csv(features_dir / 'feature_names.csv', index=False)
        
        # Save scaled data
        train_data = pd.DataFrame(X_train_scaled, columns=feature_columns)
        train_data['sales'] = y_train.values
        train_data.to_csv(features_dir / 'train_data.csv', index=False)
        
        test_data = pd.DataFrame(X_test_scaled, columns=feature_columns)
        test_data['sales'] = y_test.values
        test_data.to_csv(features_dir / 'test_data.csv', index=False)
        
        logger.info(f'Features built successfully. Training set: {X_train.shape}, Test set: {X_test.shape}')
        logger.info(f'Number of features: {len(feature_columns)}')
        
        return (X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, scaler)
        
    except Exception as e:
        logger.error(f'Error in feature engineering: {e}')
        raise


@click.command()
def cli_main():
    """CLI interface for feature engineering."""
    main()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli_main()
