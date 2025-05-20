import pandas as pd
import os

def load_and_merge_data():
    # Only use synthetic_sales_data.csv for the new workflow
    files = ["synthetic_sales_data.csv"]
    dfs = []
    for fname in files:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            # Standardize and select only the required columns
            df = df.rename(columns={
                'Order Date': 'Order Date',
                'Product': 'Product',
                'Region': 'Region',
                'SalesOffice': 'SalesOffice',
                'Price': 'Price',
                'Sales Head': 'Sales Head',
                'Regional Manager': 'Regional Manager',
                'Year': 'Year',
                'Month': 'Month'
            })
            # If Order Date is not present, create it from Year/Month
            if 'Order Date' not in df.columns and 'Year' in df.columns and 'Month' in df.columns:
                df['Order Date'] = pd.to_datetime(df[['Year', 'Month']].astype(str).agg('-'.join, axis=1) + '-01')
            else:
                df['Order Date'] = pd.to_datetime(df['Order Date'])
            # Only keep the required columns
            keep_cols = ['Order Date', 'Product', 'Region', 'SalesOffice', 'Price', 'Sales Head', 'Regional Manager']
            df = df[keep_cols]
            # Assign numeric ProductID as int32
            df['ProductID'] = df['Product'].astype('category').cat.codes.astype('int32') + 1000
            # For compatibility with rest of code
            df['Ship Branch'] = df['Region']
            df['Ext Total Sales'] = df['Price']
            dfs.append(df)
    if dfs:
        df_merged = pd.concat(dfs, ignore_index=True)
        df_merged['Order Date'] = pd.to_datetime(df_merged['Order Date'], errors='coerce', dayfirst=True)
        return df_merged
    return pd.DataFrame()

def get_productid_to_name_mapping():
    fname = "synthetic_sales_data.csv"
    if not os.path.exists(fname):
        return {}
    df = pd.read_csv(fname)
    cat = pd.Categorical(df['Product'])
    mapping = {code + 1000: name for code, name in enumerate(cat.categories)}
    return mapping
