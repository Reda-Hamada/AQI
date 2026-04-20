import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

class DataPreprocessing:
    def __init__(self, file_path, target_col="PM2.5"):
        self.file_path = file_path
        self.target_col = target_col
        self.df = None
        self.scaler = RobustScaler()

    def loading(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    def create_target_column(self):
        self.df[self.target_col] = self.df[self.target_col].shift(-1)
        self.df.dropna(inplace=True)
        return self.df

    def handle_datetime(self):
        self.df['datetime'] = pd.to_datetime(
            self.df[['year', 'month', 'day', 'hour']]
        )
        self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        # self.df.drop(columns=['year', 'month', 'day', 'hour', 'datetime'], inplace=True)

        return self.df

    def handle_missing_value(self):
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns

        self.df[numeric_cols] = self.df[numeric_cols].interpolate(limit=6)
        self.df[categorical_cols] = self.df[categorical_cols].fillna(method='bfill')

        self.df.dropna(inplace=True)
        return self.df

    def handle_encoding(self):
        self.df = pd.get_dummies(self.df, drop_first=True)
        return self.df

    def handle_scaling(self):
        return self.df

    def save_processed(self, output_path):
        self.df.to_csv(output_path, index=False)

    def process(self, output_path):
        self.loading()
        self.handle_datetime()
        self.create_target_column()
        self.handle_missing_value()
        self.handle_encoding()

        self.save_processed(output_path)
        return self.df
