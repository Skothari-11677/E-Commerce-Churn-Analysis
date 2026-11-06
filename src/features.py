"""
This module engineers customer-level features from cleaned order and behaviour data.
Design: Features are computed using ONLY orders BEFORE a cutoff date to prevent data leakage.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_rfm_features(orders_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute RFM features for each customer using orders before the cutoff date.
    
    Args:
        orders_df (pd.DataFrame): Order history dataframe.
        cutoff_date (pd.Timestamp): Date to truncate data.
        
    Returns:
        pd.DataFrame: RFM features indexed by CustomerID.
    """
    pre_cutoff = orders_df[orders_df['OrderDate'] < cutoff_date]
    if pre_cutoff.empty:
        return pd.DataFrame()
        
    rfm = pre_cutoff.groupby('CustomerID').agg(
        Recency=('OrderDate', lambda x: (cutoff_date - x.max()).days),
        Frequency=('OrderID', 'count'),
        Monetary=('OrderAmount', 'sum'),
        OrderAmountStd=('OrderAmount', 'std'),
    )
    
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    
    def avg_days_between(dates):
        if len(dates) > 1:
            return (dates.max() - dates.min()).days / (len(dates) - 1)
        return np.nan
        
    avg_days = pre_cutoff.groupby('CustomerID')['OrderDate'].apply(avg_days_between)
    rfm['AvgDaysBetweenOrders'] = avg_days
    
    return rfm

def compute_behavioural_features(orders_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute behavioural features like UniqueCategories, DiscountUsageRate, etc.
    
    Args:
        orders_df (pd.DataFrame): Order history dataframe.
        cutoff_date (pd.Timestamp): Date to truncate data.
        
    Returns:
        pd.DataFrame: Behavioural features indexed by CustomerID.
    """
    pre_cutoff = orders_df[orders_df['OrderDate'] < cutoff_date]
    if pre_cutoff.empty:
        return pd.DataFrame()
        
    behaviors = pre_cutoff.groupby('CustomerID').agg(
        UniqueCategories=('ProductCategory', 'nunique'),
        TotalOrders=('OrderID', 'count'),
        DiscountUsedCount=('DiscountUsed', lambda x: x.sum()),
        ReturnCount=('ReturnFlag', lambda x: x.sum())
    )
    behaviors['DiscountUsageRate'] = behaviors['DiscountUsedCount'] / behaviors['TotalOrders']
    behaviors['ReturnRate'] = behaviors['ReturnCount'] / behaviors['TotalOrders']
    
    dominant_cat = pre_cutoff.groupby('CustomerID')['ProductCategory'].agg(lambda x: x.value_counts().idxmax())
    behaviors['DominantCategory'] = dominant_cat
    
    return behaviors[['UniqueCategories', 'DiscountUsageRate', 'ReturnRate', 'DominantCategory']]

def compute_rfm_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM scores and segments based on quintiles.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame containing RFM features.
        
    Returns:
        pd.DataFrame: DataFrame with R_Score, F_Score, M_Score, RFM_Score, CustomerSegment.
    """
    df = rfm_df.copy()
    
    # Recency: lower is better (5 is most recent)
    df['R_Score'] = pd.qcut(df['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency: higher is better
    df['F_Score'] = pd.qcut(df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: higher is better
    df['M_Score'] = pd.qcut(df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
    
    def get_segment(score):
        if score >= 13: return 'Champion'
        elif score >= 10: return 'Loyal'
        elif score >= 7: return 'Potential Loyalist'
        elif score >= 5: return 'At Risk'
        else: return 'Lost'
        
    df['CustomerSegment'] = df['RFM_Score'].apply(get_segment)
    return df

def build_feature_matrix(customers_df: pd.DataFrame, orders_df: pd.DataFrame, behaviour_df: pd.DataFrame, cutoff_date: pd.Timestamp = pd.Timestamp('2024-01-01')) -> pd.DataFrame:
    """
    Master function to build the complete feature matrix.
    
    Args:
        customers_df: Demographic data.
        orders_df: Order data.
        behaviour_df: Behavioural support data.
        cutoff_date: Date to truncate data.
        
    Returns:
        pd.DataFrame: Complete feature matrix indexed by CustomerID.
    """
    logging.info(f"Building feature matrix with cutoff date {cutoff_date}")
    
    rfm = compute_rfm_features(orders_df, cutoff_date)
    rfm = compute_rfm_segments(rfm)
    
    behaviors = compute_behavioural_features(orders_df, cutoff_date)
    
    # Merge
    features = pd.merge(rfm, behaviors, left_index=True, right_index=True)
    
    # Demographics
    if 'CustomerID' in customers_df.columns:
        cust_df = customers_df.set_index('CustomerID')
    else:
        cust_df = customers_df
    
    features = features.merge(cust_df[['Age', 'Gender', 'Region', 'AcquisitionChannel', 'SignupDate']], 
                              left_index=True, right_index=True, how='left')
    
    # Behaviour Support
    if 'CustomerID' in behaviour_df.columns:
        b_df = behaviour_df.set_index('CustomerID')
    else:
        b_df = behaviour_df
    features = features.merge(b_df, left_index=True, right_index=True, how='left')
    
    features['DaysSinceSignup'] = (cutoff_date - pd.to_datetime(features['SignupDate'])).dt.days
    
    logging.info(f"Feature matrix built with shape {features.shape}")
    return features

if __name__ == '__main__':
    logging.info("Features module loaded.")
