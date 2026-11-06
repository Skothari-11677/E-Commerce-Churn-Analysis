import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the customers DataFrame.
    
    Steps:
    - Validates CustomerID format
    - Clips Age to 18-75
    - Fills missing Gender with 'Unknown'
    - Parses SignupDate as datetime
    - Drops duplicates
    
    Args:
        df: Raw customers DataFrame
        
    Returns:
        Cleaned customers DataFrame
    """
    logger.info("Starting customer data cleaning...")
    initial_rows = len(df)
    
    df_clean = df.copy()
    
    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Validate CustomerID (C + digits), drop invalid
    valid_id_mask = df_clean['CustomerID'].str.match(r'^C\d+$', na=False)
    df_clean = df_clean[valid_id_mask]
    
    # Clip Age
    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].clip(lower=18, upper=75)
        
    # Fill missing Gender
    if 'Gender' in df_clean.columns:
        null_genders = df_clean['Gender'].isnull().sum()
        if null_genders > 0:
            logger.info(f"Filled {null_genders} missing Gender values with 'Unknown'")
        df_clean['Gender'] = df_clean['Gender'].fillna('Unknown')
        
    # Parse SignupDate
    if 'SignupDate' in df_clean.columns:
        df_clean['SignupDate'] = pd.to_datetime(df_clean['SignupDate'], errors='coerce')
        df_clean = df_clean.dropna(subset=['SignupDate'])
        
    final_rows = len(df_clean)
    logger.info(f"Customer cleaning complete. Rows: {initial_rows} -> {final_rows}")
    
    return df_clean


def clean_orders(df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the orders DataFrame.
    
    Steps:
    - Validates OrderID
    - Drops orphans (CustomerIDs not in customers_df)
    - Validates OrderDate >= SignupDate
    - Clips OrderAmount outliers
    - Validates Quantity >= 1
    - Casts boolean flags
    - Drops duplicates
    - Parses OrderDate
    
    Args:
        df: Raw orders DataFrame
        customers_df: Cleaned customers DataFrame
        
    Returns:
        Cleaned orders DataFrame
    """
    logger.info("Starting order data cleaning...")
    initial_rows = len(df)
    
    df_clean = df.copy()
    
    df_clean = df_clean.drop_duplicates()
    
    # Validate OrderID
    valid_id_mask = df_clean['OrderID'].str.match(r'^ORD\d+$', na=False)
    df_clean = df_clean[valid_id_mask]
    
    # Drop orphans
    valid_customers = set(customers_df['CustomerID'])
    df_clean = df_clean[df_clean['CustomerID'].isin(valid_customers)]
    
    # Parse OrderDate
    if 'OrderDate' in df_clean.columns:
        df_clean['OrderDate'] = pd.to_datetime(df_clean['OrderDate'], errors='coerce')
        df_clean = df_clean.dropna(subset=['OrderDate'])
        
    # Validate OrderDate >= SignupDate
    if 'OrderDate' in df_clean.columns and 'SignupDate' in customers_df.columns:
        merged = df_clean[['OrderID', 'CustomerID', 'OrderDate']].merge(
            customers_df[['CustomerID', 'SignupDate']], on='CustomerID', how='inner'
        )
        valid_orders = merged[merged['OrderDate'] >= merged['SignupDate']]['OrderID']
        df_clean = df_clean[df_clean['OrderID'].isin(valid_orders)]
        
    # Clip OrderAmount at 99.5th percentile
    if 'OrderAmount' in df_clean.columns:
        threshold = df_clean['OrderAmount'].quantile(0.995)
        df_clean['OrderAmount'] = df_clean['OrderAmount'].clip(upper=threshold)
        
    # Quantity >= 1
    if 'Quantity' in df_clean.columns:
        df_clean['Quantity'] = df_clean['Quantity'].clip(lower=1)
        
    # Cast boolean flags
    for col in ['DiscountUsed', 'ReturnFlag']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(bool)
            
    final_rows = len(df_clean)
    logger.info(f"Order cleaning complete. Rows: {initial_rows} -> {final_rows}")
    
    return df_clean


def clean_behaviour(df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the behaviour DataFrame.
    
    Steps:
    - Drops orphans (CustomerIDs not in customers_df)
    - Clips ranges for behavioural metrics
    - Fills missing values with median
    
    Args:
        df: Raw behaviour DataFrame
        customers_df: Cleaned customers DataFrame
        
    Returns:
        Cleaned behaviour DataFrame
    """
    logger.info("Starting behaviour data cleaning...")
    initial_rows = len(df)
    
    df_clean = df.copy()
    
    # Drop orphans
    valid_customers = set(customers_df['CustomerID'])
    df_clean = df_clean[df_clean['CustomerID'].isin(valid_customers)]
    
    # Clip ranges
    if 'SessionCount_30d' in df_clean.columns:
        df_clean['SessionCount_30d'] = df_clean['SessionCount_30d'].clip(lower=0, upper=50)
        
    if 'CartAbandonmentRate' in df_clean.columns:
        df_clean['CartAbandonmentRate'] = df_clean['CartAbandonmentRate'].clip(lower=0.0, upper=1.0)
        
    if 'AvgSessionDuration' in df_clean.columns:
        df_clean['AvgSessionDuration'] = df_clean['AvgSessionDuration'].clip(lower=0.0, upper=60.0)
        
    # Fill missing values with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        nulls = df_clean[col].isnull().sum()
        if nulls > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            logger.info(f"Filled {nulls} missing values in {col} with median")
            
    final_rows = len(df_clean)
    logger.info(f"Behaviour cleaning complete. Rows: {initial_rows} -> {final_rows}")
    
    return df_clean


def run_full_cleaning(raw_dir: str, output_dir: str) -> dict:
    """
    Runs the full cleaning pipeline on raw datasets and saves the outputs.
    
    Args:
        raw_dir: Directory containing raw CSV files
        output_dir: Directory to save cleaned CSV files
        
    Returns:
        Dictionary containing the cleaned DataFrames
    """
    logger.info(f"Loading raw data from {raw_dir}")
    
    try:
        customers_df = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
        orders_df = pd.read_csv(os.path.join(raw_dir, "orders.csv"))
        behaviour_df = pd.read_csv(os.path.join(raw_dir, "behaviour_support.csv"))
    except FileNotFoundError as e:
        logger.error(f"Error loading raw data: {e}")
        raise
        
    clean_customers_df = clean_customers(customers_df)
    clean_orders_df = clean_orders(orders_df, clean_customers_df)
    clean_behaviour_df = clean_behaviour(behaviour_df, clean_customers_df)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving cleaned data to {output_dir}")
    clean_customers_df.to_csv(os.path.join(output_dir, "customers_cleaned.csv"), index=False)
    clean_orders_df.to_csv(os.path.join(output_dir, "orders_cleaned.csv"), index=False)
    clean_behaviour_df.to_csv(os.path.join(output_dir, "behaviour_support_cleaned.csv"), index=False)
    
    return {
        'customers': clean_customers_df,
        'orders': clean_orders_df,
        'behaviour': clean_behaviour_df
    }

if __name__ == '__main__':
    # Example usage
    raw_data_dir = "../data/raw"
    cleaned_data_dir = "../data/processed"
    if os.path.exists(raw_data_dir):
        run_full_cleaning(raw_data_dir, cleaned_data_dir)
    else:
        logger.warning(f"Raw data directory {raw_data_dir} not found. Skipping execution.")
