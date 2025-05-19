import pandas as pd
import os

def load_and_merge_data():
    files = ["Data_2020.csv", "Data_2021.csv", "SQL Data 5 Years Order Date.csv"]
    dfs = []
    for fname in files:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            if "OrderDate" in df.columns:
                df = df.rename(columns={
                    'OrderDate': 'Order Date',
                    'ShipBranch': 'Ship Branch',
                    'ShipDate': 'Ship Date',
                    'TotalQTY': 'Total QTY',
                    'ExtTotalSales': 'Ext Total Sales'
                })
            dfs.append(df[[c for c in ['ProductID','Order Date', 'Ship Branch', 'Ship Date', 'Total QTY', 'Ext Total Sales'] if c in df.columns]])
    if dfs:
        df_merged = pd.concat(dfs, ignore_index=True)
        df_merged['Order Date'] = pd.to_datetime(df_merged['Order Date'], errors='coerce', dayfirst=True)
        df_merged['Ship Date'] = pd.to_datetime(df_merged['Ship Date'], errors='coerce', dayfirst=True)
        return df_merged
    return pd.DataFrame()
