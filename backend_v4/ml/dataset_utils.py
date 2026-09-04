"""
SkyGuard AI - Dataset Utilities & Feature Engineering
Includes:
- Chronological data loading & cleaning
- Temporal, rolling, rate-of-change, and cross-sensor meteorological features
- Reproducible synthetic fault benchmark generator (spikes, drops, drift, bias, stuck, out-of-bounds, multivariate)
- Zero-leakage chronological train/val/test splits
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ["temperature", "pressure", "humidity"]

# All engineered features for ML models
ENGINEERED_FEATURES = [
    # Base sensors
    "temperature",
    "pressure",
    "humidity",
    # Temporal diurnal/annual cycles
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Rate of change (1st diff)
    "temp_diff",
    "press_diff",
    "hum_diff",
    # Rolling volatility & local deviation (1-hour window = 6 steps)
    "temp_roll_std",
    "press_roll_std",
    "hum_roll_std",
    "temp_roll_dev",
    # Multi-hour rolling volatility (3-hour window = 18 steps)
    "temp_roll_std_3h",
    "hum_roll_std_3h",
    # Sensor flatline / frozen streaks
    "temp_flat_streak",
    "hum_flat_streak",
    "press_flat_streak",
    # Cross-sensor physical thermodynamics
    "dew_point",
    "dew_point_spread",
    "vpd"
]


def calculate_dew_point(temp_c, rh_percent):
    """Magnus formula approximation for dew point temperature."""
    rh_clamped = np.clip(rh_percent, 1.0, 100.0)
    a = 17.625
    b = 243.04
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh_clamped / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point


def calculate_vpd(temp_c, rh_percent):
    """Vapor Pressure Deficit (VPD) in hPa."""
    rh_clamped = np.clip(rh_percent, 0.0, 100.0)
    # Saturation vapor pressure (Tetens formula) in hPa
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    # Actual vapor pressure
    ea = es * (rh_clamped / 100.0)
    vpd = np.maximum(0.0, es - ea)
    return vpd


def load_and_clean_data(csv_path, start_year=2021, end_year=2023):
    """
    Loads Jena climate data, cleans invalid records, and selects primary sensors.
    """
    print(f"Loading raw data from: {csv_path} (Years: {start_year}-{end_year})...")
    df = pd.read_csv(
        csv_path,
        usecols=["Date Time", "T (degC)", "p (mbar)", "rh (%)"],
        low_memory=False
    )

    df.rename(
        columns={
            "Date Time": "timestamp",
            "T (degC)": "temperature",
            "p (mbar)": "pressure",
            "rh (%)": "humidity"
        },
        inplace=True
    )

    # Convert timestamps and sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp", "temperature", "pressure", "humidity"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filter year range
    df = df[(df["timestamp"].dt.year >= start_year) & (df["timestamp"].dt.year <= end_year)].copy()
    df.reset_index(drop=True, inplace=True)

    # Physical boundary sanitization for raw baseline training
    df = df[
        (df["temperature"] >= -40.0) & (df["temperature"] <= 55.0) &
        (df["pressure"] >= 800.0) & (df["pressure"] <= 1100.0) &
        (df["humidity"] >= 0.0) & (df["humidity"] <= 100.0)
    ].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"Cleaned dataset: {len(df):,} records spanning {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def engineer_features(df):
    """
    Computes temporal, rate-of-change, rolling, and cross-sensor thermodynamic features.
    """
    data = df.copy()

    # Temporal cyclic features
    hour = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60.0
    month = data["timestamp"].dt.month
    data["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    data["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    data["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    data["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    # Rate of change (10-min first differences)
    data["temp_diff"] = data["temperature"].diff().fillna(0.0)
    data["press_diff"] = data["pressure"].diff().fillna(0.0)
    data["hum_diff"] = data["humidity"].diff().fillna(0.0)

    # Rolling statistics over 6 steps (1 hour window)
    window = 6
    roll_temp_mean = data["temperature"].rolling(window, min_periods=1).mean()
    data["temp_roll_std"] = data["temperature"].rolling(window, min_periods=1).std().fillna(0.0)
    data["press_roll_std"] = data["pressure"].rolling(window, min_periods=1).std().fillna(0.0)
    data["hum_roll_std"] = data["humidity"].rolling(window, min_periods=1).std().fillna(0.0)

    # Multi-hour rolling volatility (18 steps = 3 hours window)
    data["temp_roll_std_3h"] = data["temperature"].rolling(18, min_periods=1).std().fillna(0.0)
    data["hum_roll_std_3h"] = data["humidity"].rolling(18, min_periods=1).std().fillna(0.0)

    # Sensor flatline / frozen streak durations (steps of zero variance)
    t_flat = (data["temp_diff"].abs() < 1e-4)
    data["temp_flat_streak"] = t_flat.groupby((~t_flat).cumsum()).cumsum()

    h_flat = (data["hum_diff"].abs() < 1e-4)
    data["hum_flat_streak"] = h_flat.groupby((~h_flat).cumsum()).cumsum()

    p_flat = (data["press_diff"].abs() < 1e-4)
    data["press_flat_streak"] = p_flat.groupby((~p_flat).cumsum()).cumsum()

    # Normalized local rolling deviation
    data["temp_roll_dev"] = (data["temperature"] - roll_temp_mean) / (data["temp_roll_std"] + 1e-3)

    # Thermodynamic cross-sensor features
    data["dew_point"] = calculate_dew_point(data["temperature"].values, data["humidity"].values)
    data["dew_point_spread"] = data["temperature"] - data["dew_point"]
    data["vpd"] = calculate_vpd(data["temperature"].values, data["humidity"].values)

    # Replace any potential inf/nan
    data.fillna(0.0, inplace=True)
    data.replace([np.inf, -np.inf], 0.0, inplace=True)

    return data


def inject_synthetic_faults(df, anomaly_rate=0.04, seed=42):
    """
    Creates a realistic, reproducible synthetic fault benchmark.
    Fault categories:
    1. Spikes: Sudden sharp positive/negative impulses (1-2 steps)
    2. Drops: Sudden loss of reading / abrupt drop (1-3 steps)
    3. Drift: Gradual calibration drift over time (20-50 steps)
    4. Bias / Level shift: Persistent constant offset (30-60 steps)
    5. Stuck Sensor: Constant flatlined values while environment fluctuates (20-40 steps)
    6. Out-of-Bounds: Impossible values (e.g. RH > 115%, P < 750)
    7. Multivariate Inconsistency: Physical law violations (e.g. negative dew point spread)
    """
    rng = np.random.RandomState(seed)
    data = df.copy().reset_index(drop=True)
    n = len(data)

    labels = np.zeros(n, dtype=int)
    fault_types = ["Normal"] * n

    # Target number of anomalous time-steps
    target_anomalies = int(n * anomaly_rate)
    current_anomalies = 0

    attempts = 0
    max_attempts = target_anomalies * 5

    while current_anomalies < target_anomalies and attempts < max_attempts:
        attempts += 1
        fault_choice = rng.choice([
            "spike", "drop", "drift", "bias", "stuck", "out_of_bounds", "multivariate"
        ])

        idx = rng.randint(50, n - 70)
        # Avoid overlapping already injected regions
        if np.any(labels[idx - 5: idx + 60] == 1):
            continue

        if fault_choice == "spike":
            sensor = rng.choice(["temperature", "pressure", "humidity"])
            duration = rng.randint(1, 3)
            if sensor == "temperature":
                magnitude = rng.choice([-1, 1]) * rng.uniform(8.0, 18.0)
            elif sensor == "pressure":
                magnitude = rng.choice([-1, 1]) * rng.uniform(15.0, 35.0)
            else:
                magnitude = rng.choice([-1, 1]) * rng.uniform(25.0, 45.0)

            for step in range(duration):
                data.loc[idx + step, sensor] += magnitude
                labels[idx + step] = 1
                fault_types[idx + step] = f"Spike ({sensor})"
            current_anomalies += duration

        elif fault_choice == "drop":
            sensor = rng.choice(["humidity", "pressure", "temperature"])
            duration = rng.randint(1, 4)
            if sensor == "humidity":
                for step in range(duration):
                    data.loc[idx + step, "humidity"] = rng.uniform(0.0, 4.0)
                    labels[idx + step] = 1
                    fault_types[idx + step] = "Drop (humidity)"
            elif sensor == "pressure":
                for step in range(duration):
                    data.loc[idx + step, "pressure"] -= rng.uniform(25.0, 50.0)
                    labels[idx + step] = 1
                    fault_types[idx + step] = "Drop (pressure)"
            else:
                for step in range(duration):
                    data.loc[idx + step, "temperature"] -= rng.uniform(12.0, 20.0)
                    labels[idx + step] = 1
                    fault_types[idx + step] = "Drop (temperature)"
            current_anomalies += duration

        elif fault_choice == "drift":
            sensor = rng.choice(["temperature", "humidity", "pressure"])
            duration = rng.randint(25, 55)
            slope = rng.choice([-1, 1]) * rng.uniform(0.15, 0.45)
            for step in range(duration):
                data.loc[idx + step, sensor] += slope * step
                labels[idx + step] = 1
                fault_types[idx + step] = f"Drift ({sensor})"
            current_anomalies += duration

        elif fault_choice == "bias":
            sensor = rng.choice(["temperature", "pressure", "humidity"])
            duration = rng.randint(30, 60)
            if sensor == "temperature":
                offset = rng.choice([-1, 1]) * rng.uniform(6.0, 12.0)
            elif sensor == "pressure":
                offset = rng.choice([-1, 1]) * rng.uniform(15.0, 25.0)
            else:
                offset = rng.choice([-1, 1]) * rng.uniform(20.0, 35.0)

            for step in range(duration):
                data.loc[idx + step, sensor] += offset
                labels[idx + step] = 1
                fault_types[idx + step] = f"Bias ({sensor})"
            current_anomalies += duration

        elif fault_choice == "stuck":
            sensor = rng.choice(["temperature", "pressure", "humidity"])
            duration = rng.randint(25, 45)
            frozen_val = data.loc[idx, sensor]
            for step in range(duration):
                data.loc[idx + step, sensor] = frozen_val
                labels[idx + step] = 1
                fault_types[idx + step] = f"Stuck ({sensor})"
            current_anomalies += duration

        elif fault_choice == "out_of_bounds":
            sensor = rng.choice(["temperature", "pressure", "humidity"])
            duration = rng.randint(1, 3)
            if sensor == "humidity":
                val = rng.uniform(112.0, 140.0)
            elif sensor == "pressure":
                val = rng.choice([rng.uniform(600.0, 800.0), rng.uniform(1150.0, 1300.0)])
            else:
                val = rng.choice([rng.uniform(55.0, 75.0), rng.uniform(-50.0, -35.0)])

            for step in range(duration):
                data.loc[idx + step, sensor] = val
                labels[idx + step] = 1
                fault_types[idx + step] = f"OutOfBounds ({sensor})"
            current_anomalies += duration

        elif fault_choice == "multivariate":
            # Physically impossible thermodynamic combinations
            duration = rng.randint(10, 25)
            for step in range(duration):
                # E.g., scorching hot temperature combined with saturated air and impossible dew point
                data.loc[idx + step, "temperature"] = rng.uniform(42.0, 52.0)
                data.loc[idx + step, "humidity"] = rng.uniform(92.0, 99.0)
                data.loc[idx + step, "pressure"] = rng.uniform(1035.0, 1055.0)
                labels[idx + step] = 1
                fault_types[idx + step] = "Multivariate (Physics violation)"
            current_anomalies += duration

    # Recalculate all derived features with the injected sensor readings
    data = engineer_features(data)
    data["is_anomaly"] = labels
    data["fault_type"] = fault_types

    actual_rate = (labels.sum() / n) * 100.0
    print(f"Injected {labels.sum():,} synthetic anomalies ({actual_rate:.2f}% anomaly prevalence, seed={seed}).")
    return data


def prepare_train_val_test(csv_path, start_year=2021, end_year=2023):
    """
    Creates chronological train/val/test splits with zero leakage.
    - Train (70%): Clean baseline data.
    - Val (15%): Evaluated with synthetic benchmark (Seed 42) for model & threshold tuning.
    - Test (15%): Evaluated with synthetic benchmark (Seed 123) for final holdout testing.
    """
    df_clean = load_and_clean_data(csv_path, start_year=start_year, end_year=end_year)
    n = len(df_clean)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    raw_train = df_clean.iloc[:train_end].copy().reset_index(drop=True)
    raw_val = df_clean.iloc[train_end:val_end].copy().reset_index(drop=True)
    raw_test = df_clean.iloc[val_end:].copy().reset_index(drop=True)

    print("\n--- Chronological Split Boundaries ---")
    print(f"Train : {len(raw_train):,} rows ({raw_train['timestamp'].min()} to {raw_train['timestamp'].max()})")
    print(f"Val   : {len(raw_val):,} rows ({raw_val['timestamp'].min()} to {raw_val['timestamp'].max()})")
    print(f"Test  : {len(raw_test):,} rows ({raw_test['timestamp'].min()} to {raw_test['timestamp'].max()})")

    # Feature engineer clean train
    train_df = engineer_features(raw_train)
    train_df["is_anomaly"] = 0
    train_df["fault_type"] = "Normal"

    # Inject synthetic benchmarks into validation and test sets with distinct seeds
    val_df = inject_synthetic_faults(raw_val, anomaly_rate=0.045, seed=42)
    test_df = inject_synthetic_faults(raw_test, anomaly_rate=0.045, seed=123)

    # Fit StandardScaler ONLY on clean training set to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(train_df[ENGINEERED_FEATURES])

    return train_df, val_df, test_df, scaler
