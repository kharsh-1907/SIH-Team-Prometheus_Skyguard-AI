import os
import io
import zipfile
import requests
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ==================================================
# SKYGUARD AI - MODEL TRAINING
# ==================================================

BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

os.makedirs(MODEL_DIR, exist_ok=True)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "isolation_forest.joblib"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.joblib"
)


# ==================================================
# DATASET URL
# ==================================================

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "matthewjansen/bgc-jena-weather-station-dataset-20172024"
)


# ==================================================
# FEATURES
# ==================================================

FEATURES = [
    "temperature",
    "pressure",
    "humidity"
]


# ==================================================
# DOWNLOAD DATASET
# ==================================================

print("=" * 50)
print("SKYGUARD AI - MODEL TRAINING")
print("=" * 50)

print("\nDownloading BGC-Jena dataset...")

response = requests.get(
    DATASET_URL,
    timeout=120
)

response.raise_for_status()

print("Dataset downloaded successfully.")


# ==================================================
# READ ZIP FILE
# ==================================================

print("\nReading dataset ZIP...")

with zipfile.ZipFile(
    io.BytesIO(response.content)
) as z:

    files = z.namelist()

    print("Files found:", len(files))

    # Find main CSV
    csv_files = [
        file
        for file in files
        if file.lower().endswith(".csv")
        and "jena_climate_2017_2024.csv" in file
    ]

    if not csv_files:

        raise FileNotFoundError(
            "jena_climate_2017_2024.csv not found in dataset"
        )

    csv_file = csv_files[0]

    print("Using:", csv_file)

    with z.open(csv_file) as file:

        df = pd.read_csv(file)


print("\nDataset loaded successfully.")

print("Original shape:", df.shape)


# ==================================================
# DATE CONVERSION
# ==================================================

print("\nConverting timestamp...")

df["Date Time"] = pd.to_datetime(
    df["Date Time"],
    dayfirst=True,
    errors="coerce"
)


# ==================================================
# SELECT 2023 DATA
# ==================================================

print("Filtering 2023 data...")

df = df[
    (df["Date Time"].dt.year == 2023)
].copy()


print(
    "2023 rows:",
    len(df)
)


# ==================================================
# SELECT REQUIRED COLUMNS
# ==================================================

data = df[
    [
        "Date Time",
        "T (degC)",
        "p (mbar)",
        "rh (%)"
    ]
].copy()


# ==================================================
# RENAME COLUMNS
# ==================================================

data.rename(
    columns={
        "Date Time": "timestamp",
        "T (degC)": "temperature",
        "p (mbar)": "pressure",
        "rh (%)": "humidity"
    },
    inplace=True
)


# ==================================================
# CLEAN DATA
# ==================================================

print("\nCleaning data...")

data.dropna(
    inplace=True
)

data.drop_duplicates(
    inplace=True
)

data.reset_index(
    drop=True,
    inplace=True
)


print(
    "Clean dataset shape:",
    data.shape
)


# ==================================================
# CREATE FEATURE DATA
# ==================================================

X = data[
    FEATURES
].copy()


# ==================================================
# STANDARD SCALER
# ==================================================

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ==================================================
# ISOLATION FOREST
# ==================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

model.fit(
    X_scaled
)


# ==================================================
# SAVE MODEL
# ==================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


# ==================================================
# TEST MODEL
# ==================================================

print("\nTesting model...")

predictions = model.predict(
    X_scaled
)

anomaly_scores = model.decision_function(
    X_scaled
)


normal_count = (
    predictions == 1
).sum()

anomaly_count = (
    predictions == -1
).sum()


anomaly_rate = (
    anomaly_count /
    len(predictions)
) * 100


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print(
    "Total readings :",
    len(data)
)

print(
    "Normal         :",
    normal_count
)

print(
    "Anomalies      :",
    anomaly_count
)

print(
    "Anomaly rate   :",
    round(anomaly_rate, 3),
    "%"
)

print("\nModel:")
print(MODEL_PATH)

print("\nScaler:")
print(SCALER_PATH)

print("=" * 50)