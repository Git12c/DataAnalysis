import pandas as pd

def preprocess_for_prophet(df, product_id=500000, branch='Branch 1'):
    targeted_data = df[df['ProductID'] == product_id]
    target_data_prophet = targeted_data.rename(columns={'Ship Date': 'ds', 'Ext Total Sales': 'y'})
    target_data_prophet['ds'] = pd.to_datetime(target_data_prophet['ds'])
    target_data_prophet['y'] = (
        target_data_prophet['y']
        .replace('[\$,]', '', regex=True)
        .replace('[()]', '', regex=True)
        .astype(str)
    )
    target_data_prophet['y'] = target_data_prophet['y'].apply(lambda x: -float(x) if '(' in x or ')' in x else float(x))
    branch_data = target_data_prophet[target_data_prophet['Ship Branch'] == branch]
    return branch_data[['ds', 'y']]
