import numpy as np
import pandas as pd
import re

def add_target(self):
    aqi_table = pd.read_csv("/home/reda/AQI/data/raw/calc-AQI.csv")

    # Parse "low–high" strings into (low, high) tuples per cell
    def parse_range(s):
        parts = re.split(r"\s*[–-]\s*", str(s).strip())
        vals = [float(x) for x in parts]
        return (min(vals), max(vals))

    parsed = aqi_table.applymap(parse_range)

    # Build clean bracket arrays from the AQI table
    aqi_low  = np.array([v[0] for v in parsed["AQI"]])
    aqi_high = np.array([v[1] for v in parsed["AQI"]])

    pollutants = ["PM2.5", "PM10", "CO", "SO2", "NO2"]

    # Precompute bracket edges per pollutant
    bracket_edges = {}
    for gas in pollutants:
        c_lows  = np.array([v[0] for v in parsed[gas]])
        c_highs = np.array([v[1] for v in parsed[gas]])
        bracket_edges[gas] = (c_lows, c_highs)

    def compute_sub_aqi(series: pd.Series, gas: str) -> pd.Series:
        c_lows, c_highs = bracket_edges[gas]
        values = series.to_numpy(dtype=float)

        # Use the low edge of each bracket as the bin boundary for pd.cut
        # Add one extra edge beyond the last bracket
        bin_edges = np.append(c_lows, c_highs[-1])
        bin_edges = np.unique(bin_edges)  # ensure sorted & no duplicates

        # Assign each value to a bracket index (0-based)
        # pd.cut returns NaN for out-of-range; we handle clipping separately
        labels = np.arange(len(bin_edges) - 1)
        bracket_idx = pd.cut(
            series,
            bins=bin_edges,
            labels=labels,
            include_lowest=True
        ).astype("Int64")  # nullable int to preserve NaN

        # Clip out-of-range values to nearest bracket
        too_low  = pd.notna(series) & (series < bin_edges[0])
        too_high = pd.notna(series) & (series > bin_edges[-1])
        bracket_idx[too_low]  = 0
        bracket_idx[too_high] = len(c_lows) - 1

        # Vectorized linear interpolation
        valid = bracket_idx.notna()
        result = pd.array([pd.NA] * len(series), dtype="Float64")

        idx_arr = bracket_idx[valid].to_numpy(dtype=int)
        val_arr = values[valid.to_numpy()]

        cl = c_lows[idx_arr]
        ch = c_highs[idx_arr]
        il = aqi_low[idx_arr]
        ih = aqi_high[idx_arr]

        # Avoid division by zero for degenerate brackets
        denom = ch - cl
        interpolated = np.where(
            denom == 0,
            ih,
            (ih - il) / denom * (val_arr - cl) + il
        )

        result[valid.to_numpy()] = interpolated
        return pd.array(result, dtype="Float64")

    # Compute sub-AQI for each pollutant as a full column, then take row-wise max
    sub_aqi_df = pd.DataFrame({
        gas: compute_sub_aqi(self.df[gas], gas)
        for gas in pollutants
    })

    self.df["AQI"] = sub_aqi_df.max(axis=1, skipna=True)
    return self.df