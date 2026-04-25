import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class TimeSeriesDataset:
    def __init__(self, file_path, target_col="PM2.5", seq_len=24, test_ratio=0.2):
        self.file_path = file_path
        self.target_col = target_col
        self.seq_len = seq_len
        self.test_ratio = test_ratio

        self.df = None
        self.scaler = RobustScaler() # Added Scaler here

    def load_data(self):
        self.df = pd.read_csv(self.file_path) # processed_data_v1.csv
        return self.df

    def split_data(self):
        split_idx = int(len(self.df) * (1 - self.test_ratio))

        train_df = self.df[:split_idx].reset_index(drop=True)
        test_df = self.df[split_idx:].reset_index(drop=True)

        return train_df, test_df

    def scale_data(self, train_df, test_df):
        # Fit ONLY on training data to prevent Data Leakage
        feature_cols = [col for col in train_df.columns if col != "station"]

        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])

        # ONLY transform test (using the rules learned from train)
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])

        return train_df, test_df

    def create_sequences(self, data):
        X, y = [], []

        for station in data['station'].unique():
            station_data = data[data['station'] == station].drop(columns=['station']).reset_index(drop=True)
            target_idx = station_data.columns.get_loc(self.target_col)
            values = station_data.values

            for i in range(len(station_data) - self.seq_len):
                X.append(values[i:i + self.seq_len])
                y.append(values[i + self.seq_len, target_idx])
                
        return np.array(X), np.array(y)

    def get_data(self):
        self.load_data()

        train_df, test_df = self.split_data()

        # Added scaling step here to prevent data leakage
        train_df, test_df = self.scale_data(train_df, test_df)

        X_train, y_train = self.create_sequences(train_df)
        X_test, y_test = self.create_sequences(test_df)

        return X_train, y_train, X_test, y_test
