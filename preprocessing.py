import pandas as pd

def preprocess_for_prophet(df, product_id=1000, branch='South'):
    # For synthetic_sales_data.csv, ProductID and Ship Branch are always present
    targeted_data = df[df['ProductID'] == product_id]
    target_data_prophet = targeted_data.rename(columns={'Order Date': 'ds', 'Ext Total Sales': 'y'})
    target_data_prophet['ds'] = pd.to_datetime(target_data_prophet['ds'])
    target_data_prophet['y'] = pd.to_numeric(target_data_prophet['y'], errors='coerce')
    branch_data = target_data_prophet[target_data_prophet['Ship Branch'] == branch]
    return branch_data[['ds', 'y']]
