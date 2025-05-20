import pandas as pd

def preprocess_for_prophet(df, product_id=1000, branch='South'):
    # Filter for product and branch
    targeted_data = df[(df['ProductID'] == product_id) & (df['Ship Branch'] == branch)]
    # Rename for Prophet
    target_data_prophet = targeted_data.rename(columns={'Order Date': 'ds', 'Ext Total Sales': 'y'})
    target_data_prophet['ds'] = pd.to_datetime(target_data_prophet['ds'])
    target_data_prophet['y'] = pd.to_numeric(target_data_prophet['y'], errors='coerce')
    # Retain extra columns for downstream use
    keep_cols = ['ds', 'y', 'Product', 'Region', 'SalesOffice', 'Sales Head', 'Regional Manager', 'ProductID', 'Ship Branch']
    for col in keep_cols:
        if col not in target_data_prophet.columns:
            target_data_prophet[col] = None
    branch_data = target_data_prophet[keep_cols]
    return branch_data
