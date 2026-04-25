import numpy as np
import pandas as pd
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

    # REMOVED: This causes Data Leakage (double shift) because dataset.py already shifts for sequences.
    # We leave the code commented out for reference.
    # def create_target_column(self):
    #     self.df[self.target_col] = self.df[self.target_col].shift(-1)
    #     self.df.dropna(inplace=True)
    #     return self.df

    def handle_datetime(self):
        self.df["datetime"] = pd.to_datetime(self.df[["year", "month", "day", "hour"]])
        self.df["day_of_week"] = self.df["datetime"].dt.dayofweek
        self.df["hour_sin"] = np.sin(2 * np.pi * self.df["hour"] / 24)
        self.df["hour_cos"] = np.cos(2 * np.pi * self.df["hour"] / 24)
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)
        return self.df


    def handle_missing_value(self):
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        self.df[numeric_cols] = self.df[numeric_cols].interpolate(limit=6)
        self.df[categorical_cols] = self.df[categorical_cols].fillna(method="bfill")

        self.df.dropna(inplace=True)
        return self.df


    def handle_encoding(self):
        categorical_cols = [
            col for col in self.df.select_dtypes(include=["object"]).columns
            if col != "station"
        ]
        if categorical_cols:
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        self.df = self.df.drop(columns=["datetime"])
        return self.df

    # REMOVED: Scaling the entire dataset before splitting causes Data Leakage from the test set.
    # Scaling is now handled in dataset.py after the Train/Test split.
    # def handle_scaling(self):
    #     return self.df

    def save_processed(self, output_path):
        self.df.to_csv(output_path, index=False)

    def process(self, output_path):
        self.loading()
        self.handle_datetime()
        self.handle_missing_value()
        # self.create_target_column() # Commented out to prevent Data Leakage (Future Shift)
        self.handle_encoding()
        # self.handle_scaling() # Commented out, moved to dataset.py to prevent Data Leakage
        self.save_processed(output_path)
        return self.df
