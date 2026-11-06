import pandas as pd
import numpy as np
from faker import Faker
import os
from datetime import datetime, timedelta

def get_product_catalog() -> pd.DataFrame:
    """
    Generates a synthetic product catalog with 8 categories and 25 products per category.
    Returns a DataFrame containing ProductID, ProductCategory, and ProductName.
    """
    categories = ['Electronics Accessories', 'Groceries', 'Clothing', 'Beauty', 
                  'Books', 'Home & Garden', 'Sports', 'Furniture']
    cat_prefixes = ['ELEC', 'GROC', 'CLOT', 'BEAU', 'BOOK', 'HOME', 'SPOR', 'FURN']
    
    catalog = []
    fake = Faker()
    Faker.seed(42)
    
    for cat, prefix in zip(categories, cat_prefixes):
        for i in range(1, 26):
            prod_id = f"{prefix}{i:03d}"
            prod_name = f"{cat} - {fake.word().capitalize()} {fake.word().capitalize()}"
            catalog.append({
                'ProductID': prod_id,
                'ProductCategory': cat,
                'ProductName': prod_name
            })
    return pd.DataFrame(catalog)

def generate_all_data(output_dir: str, n_customers: int = 10000, seed: int = 42) -> dict:
    """
    Generates fully synthetic e-commerce datasets: customers, orders, and behaviour_support.
    
    Args:
        output_dir (str): Directory where the CSV files will be saved.
        n_customers (int): Number of customers to generate.
        seed (int): Random seed for reproducibility.
        
    Returns:
        dict: A dictionary containing the 'customers', 'orders', and 'behaviour_support' DataFrames.
    """
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    
    # ==========================================
    # 1. Generate Customers
    # ==========================================
    customer_ids = [f"C{i:04d}" for i in range(1, n_customers + 1)]
    ages = np.clip(np.random.normal(35, 12, n_customers), 18, 75).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=n_customers, p=[0.45, 0.45, 0.10])
    regions = np.random.choice(
        ['North', 'South', 'East', 'West', 'Central', 'Northeast'], 
        size=n_customers, 
        p=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1]
    )
    channels = np.random.choice(
        ['Organic', 'Paid Search', 'Social Media', 'Referral', 'Email'], 
        size=n_customers, 
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)
    total_days = (end_date - start_date).days
    
    signup_days_offset = np.random.randint(0, total_days + 1, n_customers)
    signup_dates = [start_date + timedelta(days=int(d)) for d in signup_days_offset]
    
    customers_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'Gender': genders,
        'Region': regions,
        'AcquisitionChannel': channels,
        'SignupDate': signup_dates
    })
    
    # ==========================================
    # 2. Generate Orders
    # ==========================================
    segments = np.random.choice(['Heavy', 'Moderate', 'Light'], size=n_customers, p=[0.2, 0.5, 0.3])
    
    order_counts = []
    for seg in segments:
        if seg == 'Heavy':
            order_counts.append(np.random.poisson(10)) # approx 8-12
        elif seg == 'Moderate':
            order_counts.append(np.random.poisson(5))  # approx 4-6
        else:
            order_counts.append(np.random.poisson(2))  # approx 1-3
            
    order_counts = np.clip(order_counts, 0, None)
    
    catalog_df = get_product_catalog()
    
    order_data = []
    order_id_counter = 1
    
    # Category spending parameters: (mean, std, min)
    cat_params = {
        'Electronics Accessories': (80, 30, 10),
        'Groceries': (35, 15, 5),
        'Clothing': (60, 25, 10),
        'Beauty': (40, 20, 5),
        'Books': (20, 8, 5),
        'Home & Garden': (70, 30, 10),
        'Sports': (55, 20, 10),
        'Furniture': (200, 80, 30)
    }
    
    for i, cid in enumerate(customer_ids):
        num_orders = order_counts[i]
        c_signup = signup_dates[i]
        days_active = (end_date - c_signup).days
        
        if days_active <= 0:
            continue
            
        for _ in range(num_orders):
            o_date = c_signup + timedelta(days=int(np.random.randint(0, days_active + 1)))
            
            prod = catalog_df.sample(1).iloc[0]
            cat = prod['ProductCategory']
            
            mean_amt, std_amt, min_amt = cat_params[cat]
            amt = max(min_amt, np.random.normal(mean_amt, std_amt))
            qty = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.1, 0.05, 0.05])
            
            discount = np.random.rand() < 0.12
            ret = np.random.rand() < 0.08
            
            order_data.append({
                'OrderID': f"ORD{order_id_counter:05d}",
                'CustomerID': cid,
                'OrderDate': o_date,
                'ProductCategory': cat,
                'ProductID': prod['ProductID'],
                'ProductName': prod['ProductName'],
                'OrderAmount': round(amt, 2),
                'Quantity': qty,
                'DiscountUsed': discount,
                'ReturnFlag': ret
            })
            order_id_counter += 1
            
    orders_df = pd.DataFrame(order_data)
    
    # ==========================================
    # 3. Generate Behaviour & Support
    # ==========================================
    customer_returns = orders_df.groupby('CustomerID')['ReturnFlag'].sum().to_dict()
    
    beh_data = []
    for i, cid in enumerate(customer_ids):
        seg = segments[i]
        if seg == 'Heavy':
            sess = int(np.random.uniform(15, 30))
            cart_ab = np.random.uniform(0.1, 0.3)
        elif seg == 'Moderate':
            sess = int(np.random.uniform(5, 15))
            cart_ab = np.random.uniform(0.3, 0.5)
        else:
            sess = int(np.random.uniform(1, 5))
            cart_ab = np.random.uniform(0.4, 0.8)
            
        avg_dur = min(30, max(2, sess * np.random.uniform(0.5, 1.5)))
        
        tickets = np.random.poisson(1.5)
        if cid in customer_returns and customer_returns[cid] > 0:
            tickets += 2
            
        res_time = 0
        if tickets > 0:
            res_time = round(np.random.uniform(1, 72), 1)
            
        beh_data.append({
            'CustomerID': cid,
            'SessionCount_30d': sess,
            'CartAbandonmentRate': round(cart_ab, 3),
            'AvgSessionDuration': round(avg_dur, 1),
            'SupportTicketCount': tickets,
            'LastSupportResolutionTime': res_time
        })
        
    beh_df = pd.DataFrame(beh_data)
    
    # ==========================================
    # Save Data
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    
    customers_df.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)
    orders_df.to_csv(os.path.join(output_dir, 'orders.csv'), index=False)
    beh_df.to_csv(os.path.join(output_dir, 'behaviour_support.csv'), index=False)
    
    return {
        'customers': customers_df,
        'orders': orders_df,
        'behaviour_support': beh_df
    }

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'raw')
    print(f"Generating data to {output_dir}...")
    dfs = generate_all_data(output_dir)
    print("Data generation complete.")
    for k, df in dfs.items():
        print(f"{k}: {df.shape}")
