import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

class TimeSeriesDataset:
    def __init__(self, file_path, target_col="AQI", seq_len=24, test_ratio=0.2):
        self.file_path = file_path
        self.target_col = target_col
        self.seq_len = seq_len
        self.test_ratio = test_ratio

        self.df = None
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()

    def load_data(self):
        self.df = pd.read_csv(self.file_path) # processed_data_v1.csv
        return self.df

    def split_data(self):
        if self.df is None or self.df.empty:
            raise ValueError("Data is empty. Call load_data() before split_data().")

        data = self.df.copy()

        if "datetime" in data.columns:
            sort_cols = ["datetime"]
        else:
            sort_cols = [c for c in ["year", "month", "day", "hour"] if c in data.columns]

        if "station" in data.columns:
            by_cols = ["station"] + sort_cols if sort_cols else ["station"]
            data = data.sort_values(by=by_cols).reset_index(drop=True)

            train_parts, test_parts = [], []
            for _, station_df in data.groupby("station", sort=False):
                n_rows = len(station_df)
                if n_rows < 2:
                    continue

                split_idx = int(n_rows * (1 - self.test_ratio))
                split_idx = min(max(split_idx, 1), n_rows - 1)

                train_parts.append(station_df.iloc[:split_idx])
                test_parts.append(station_df.iloc[split_idx:])

            if not train_parts or not test_parts:
                raise ValueError("Split produced empty train/test sets; check data size and test_ratio.")

            train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
            test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
            return train_df, test_df

        if sort_cols:
            data = data.sort_values(by=sort_cols).reset_index(drop=True)

        split_idx = int(len(data) * (1 - self.test_ratio))
        split_idx = min(max(split_idx, 1), len(data) - 1)
        train_df = data.iloc[:split_idx].reset_index(drop=True)
        test_df = data.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    def scale_data(self, train_df, test_df):
        train_df = train_df.copy()
        test_df = test_df.copy()

        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in train data.")

        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col not in numeric_cols:
            raise ValueError(f"Target column '{self.target_col}' must be numeric for scaling.")

        feature_cols = [col for col in numeric_cols if col != self.target_col]

        # Fit ONLY on training data to prevent data leakage.
        if feature_cols:
            train_df[feature_cols] = self.feature_scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = self.feature_scaler.transform(test_df[feature_cols])

        train_df[[self.target_col]] = self.target_scaler.fit_transform(train_df[[self.target_col]])
        test_df[[self.target_col]] = self.target_scaler.transform(test_df[[self.target_col]])

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
