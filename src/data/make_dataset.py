# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datetime import datetime, timedelta


def generate_sample_marketing_data(n_days=365):
    """Generate sample marketing mix data for demonstration."""
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate marketing spend data
    data = {
        'date': dates,
        'tv_spend': np.random.normal(50000, 15000, n_days),
        'radio_spend': np.random.normal(20000, 8000, n_days),
        'print_spend': np.random.normal(15000, 5000, n_days),
        'digital_spend': np.random.normal(30000, 12000, n_days),
        'sales': np.random.normal(1000000, 200000, n_days),
        'competitor_price': np.random.normal(50, 5, n_days),
        'our_price': np.random.normal(55, 3, n_days),
        'seasonality': np.sin(2 * np.pi * np.arange(n_days) / 365) * 0.2 + 1,
        'holiday': np.random.choice([0, 1], n_days, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic relationships
    df['sales'] = (
        df['tv_spend'] * 0.3 +
        df['radio_spend'] * 0.2 +
        df['print_spend'] * 0.1 +
        df['digital_spend'] * 0.4 +
        df['seasonality'] * 200000 +
        df['holiday'] * 50000 +
        np.random.normal(0, 50000, n_days)
    )
    
    return df


def clean_data(df):
    """Clean and prepare the marketing data."""
    df_clean = df.copy()
    
    # Ensure no negative values for spends
    spend_columns = ['tv_spend', 'radio_spend', 'print_spend', 'digital_spend']
    for col in spend_columns:
        df_clean[col] = df_clean[col].clip(lower=0)
    
    # Ensure sales is positive
    df_clean['sales'] = df_clean['sales'].clip(lower=1000)
    
    # Add derived features
    df_clean['total_spend'] = df_clean[spend_columns].sum(axis=1)
    df_clean['roas'] = df_clean['sales'] / df_clean['total_spend']
    
    # Add price difference
    df_clean['price_diff'] = df_clean['our_price'] - df_clean['competitor_price']
    
    # Add day of week and month features
    df_clean['day_of_week'] = pd.to_datetime(df_clean['date']).dt.dayofweek
    df_clean['month'] = pd.to_datetime(df_clean['date']).dt.month
    
    # Remove any infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    return df_clean


def main(input_filepath=None, output_filepath=None):
    """Runs data processing scripts to turn raw data into cleaned data ready for analysis."""
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    
    try:
        # Generate sample data (in real scenario, this would load from input_filepath)
        logger.info('Generating sample marketing mix data...')
        df_raw = generate_sample_marketing_data()
        
        # Clean the data
        logger.info('Cleaning and preparing data...')
        df_clean = clean_data(df_raw)
        
        # Save processed data
        if output_filepath:
            df_clean.to_csv(output_filepath, index=False)
            logger.info(f'Processed data saved to {output_filepath}')
        else:
            # Save to default location
            processed_dir = Path(__file__).resolve().parents[2] / 'data' / 'processed'
            processed_dir.mkdir(parents=True, exist_ok=True)
            output_path = processed_dir / 'marketing_data_processed.csv'
            df_clean.to_csv(output_path, index=False)
            logger.info(f'Processed data saved to {output_path}')
        
        # Create cleaning report
        cleaning_report = {
            'original_shape': df_raw.shape,
            'cleaned_shape': df_clean.shape,
            'rows_removed': df_raw.shape[0] - df_clean.shape[0],
            'columns_added': df_clean.shape[1] - df_raw.shape[1],
            'date_range': f"{df_clean['date'].min()} to {df_clean['date'].max()}",
            'total_spend_range': f"${df_clean['total_spend'].min():,.0f} - ${df_clean['total_spend'].max():,.0f}",
            'sales_range': f"${df_clean['sales'].min():,.0f} - ${df_clean['sales'].max():,.0f}"
        }
        
        logger.info('Data processing completed successfully')
        logger.info(f'Cleaning report: {cleaning_report}')
        
        return df_clean, cleaning_report
        
    except Exception as e:
        logger.error(f'Error in data processing: {e}')
        raise


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), required=False)
@click.argument('output_filepath', type=click.Path(), required=False)
def cli_main(input_filepath, output_filepath):
    """CLI interface for data processing."""
    main(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli_main()
