import pandas as pd
import glob 
import os

#path = "/home/reda/AQI/data/raw"

def data_merge(in_path, out_path):

    files = glob.glob(os.path.join(in_path, "*.csv")) #[f1, f2, ..., fn]

    dfs = [] # [df1, df2, ..., dfn]
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['datetime'] = pd.to_datetime(merged_df[['year', 'month', 'day', 'hour']])

    merged_df = merged_df.sort_values(['station', 'datetime']).reset_index(drop=True)


    print(merged_df.shape)
    print(merged_df['station'].value_counts())
    print(merged_df.head())

    merged_df.to_csv(out_path)

    return merged_df