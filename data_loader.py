import pandas as pd
import os

def load_and_merge_data():
    # Only use synthetic_sales_data.csv for the new workflow
    files = ["synthetic_sales_data.csv"]
    dfs = []
    for fname in files:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            # For synthetic_sales_data.csv, create required columns
            df['Order Date'] = pd.to_datetime(df[['Year', 'Month']].astype(str).agg('-'.join, axis=1) + '-01')
            # Assign numeric ProductID as int32 to avoid OverflowError
            df['ProductID'] = df['Product'].astype('category').cat.codes.astype('int32') + 1000
            df['Ship Branch'] = df['Region']
            df['Ext Total Sales'] = df['Price']
            # No Ship Date or Total QTY in this synthetic data
            dfs.append(df[['ProductID','Order Date', 'Ship Branch', 'Ext Total Sales']])
    if dfs:
        df_merged = pd.concat(dfs, ignore_index=True)
        df_merged['Order Date'] = pd.to_datetime(df_merged['Order Date'], errors='coerce', dayfirst=True)
        return df_merged
    return pd.DataFrame()
