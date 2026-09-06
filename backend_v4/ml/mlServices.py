"""
To start ML: python ml/mlServices.py

"""
import os
import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# --------------------------------------------------
# PATHS & CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Model path hierarchy: Phase 2 -> SkyGuard Validated -> Legacy
PHASE2_MODEL_PATH = os.path.join(MODEL_DIR, "phase2_model.joblib")
PHASE2_SCALER_PATH = os.path.join(MODEL_DIR, "phase2_scaler.joblib")
SKYGUARD_MODEL_PATH = os.path.join(MODEL_DIR, "skyguard_anomaly_model.joblib")
SKYGUARD_SCALER_PATH = os.path.join(MODEL_DIR, "skyguard_scaler.joblib")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
LEGACY_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Station-aware sliding history buffers (last 24 readings = 4 hours of 10-min data)
station_buffers = defaultdict(lambda: deque(maxlen=24))

# Station-aware dynamic sensor health (0 - 100%)
station_health = defaultdict(lambda: {"temperature": 100, "pressure": 100, "humidity": 100})

# Thread-safety lock for concurrent requests
buffer_lock = threading.Lock()

# Global ML runtime state
model = None
scaler = None
metadata = {}
threshold = 0.6172
model_name = "Two-Tier Isolation Forest"
scoring_type = "isolation_forest"
FEATURES = [
    "temperature", "pressure", "humidity",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "temp_diff", "press_diff", "hum_diff",
    "temp_roll_std", "press_roll_std", "hum_roll_std", "temp_roll_dev",
    "temp_roll_std_3h", "hum_roll_std_3h",
    "temp_flat_streak", "hum_flat_streak", "press_flat_streak",
    "dew_point", "dew_point_spread", "vpd"
]


# --------------------------------------------------
# THERMODYNAMIC UTILITIES
# --------------------------------------------------

def calculate_dew_point(temp_c, rh_percent):
    """Magnus formula approximation for dew point temperature (°C)."""
    rh_clamped = max(1.0, min(100.0, float(rh_percent)))
    a = 17.625
    b = 243.04
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh_clamped / 100.0)
    return float((b * alpha) / (a - alpha))


def calculate_vpd(temp_c, rh_percent):
    """Vapor Pressure Deficit (VPD) in hPa."""
    rh_clamped = max(0.0, min(100.0, float(rh_percent)))
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    ea = es * (rh_clamped / 100.0)
    return float(max(0.0, es - ea))


# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------

def load_model():
    """Loads model, scaler, and threshold metadata with fallback cascade."""
    global model, scaler, metadata, threshold, model_name, scoring_type, FEATURES

    if os.path.exists(PHASE2_MODEL_PATH) and os.path.exists(PHASE2_SCALER_PATH):
        target_model_path = PHASE2_MODEL_PATH
        target_scaler_path = PHASE2_SCALER_PATH
    elif os.path.exists(SKYGUARD_MODEL_PATH) and os.path.exists(SKYGUARD_SCALER_PATH):
        target_model_path = SKYGUARD_MODEL_PATH
        target_scaler_path = SKYGUARD_SCALER_PATH
    else:
        target_model_path = LEGACY_MODEL_PATH
        target_scaler_path = LEGACY_SCALER_PATH

    if not os.path.exists(target_model_path) or not os.path.exists(target_scaler_path):
        print("ML Artifacts not found:", target_model_path)
        return False

    try:
        model = joblib.load(target_model_path)
        scaler = joblib.load(target_scaler_path)
        print(f"Loaded ML model ({type(model).__name__}) from: {target_model_path}")
        print(f"Loaded Scaler ({type(scaler).__name__}) from: {target_scaler_path}")

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
            threshold = float(metadata.get("threshold", 0.6172))
            model_name = metadata.get("model_name", type(model).__name__)
            scoring_type = metadata.get("scoring_type", "isolation_forest")
            FEATURES = metadata.get("engineered_features", FEATURES)
            print(f"Loaded metadata: Model={model_name}, Threshold={threshold:.6f}, Features={len(FEATURES)}")
        else:
            threshold = 0.6172
            model_name = type(model).__name__
            scoring_type = "isolation_forest"

        return True
    except Exception as e:
        print("Error loading ML artifacts:", str(e))
        return False


# Eagerly load model on startup
load_model()


# --------------------------------------------------
# TIER 1: DETERMINISTIC SENSOR & PHYSICS DETECTION
# --------------------------------------------------

def evaluate_tier1(temperature, pressure, humidity, dew_point_spread, vpd, temp_diff, press_diff, hum_diff, temp_flat_streak, press_flat_streak, hum_flat_streak):
    """
    Tier 1 Deterministic Rules:
    - Physical Range Bounds
    - Rate of Change Limits
    - Thermodynamic Inconsistencies
    - Stuck Sensor Flatlines
    Returns: (is_anomaly, fault_type, affected_sensor, severity)
    """
    # 1. Physical range bounds
    if temperature < -40.0 or temperature > 55.0:
        return True, "OutOfBounds", "temperature", "High"
    if pressure < 800.0 or pressure > 1100.0:
        return True, "OutOfBounds", "pressure", "High"
    if humidity < 0.0 or humidity > 100.0:
        return True, "OutOfBounds", "humidity", "High"

    # 2. Maximum plausible 10-minute rate of change
    t_roc = abs(temp_diff) > 6.0
    p_roc = abs(press_diff) > 4.0
    h_roc = abs(hum_diff) > 30.0

    if sum([t_roc, p_roc, h_roc]) > 1:
        return True, "RateOfChange", "multiple", "High"
    if t_roc:
        return True, "RateOfChange", "temperature", "High"
    if p_roc:
        return True, "RateOfChange", "pressure", "High"
    if h_roc:
        return True, "RateOfChange", "humidity", "High"

    # 3. Thermodynamic consistency
    if dew_point_spread < -0.5:
        # Air temperature cannot physically be below dew point
        return True, "ThermodynamicInconsistency", "multiple", "Medium"
    if vpd < 0.0:
        return True, "ThermodynamicInconsistency", "humidity", "Medium"

    # 4. Stuck sensor flatline detection (>= 6 readings = 1 hour constant)
    # Non-saturated humidity check prevents false alarms in fog / saturated rain
    if temp_flat_streak >= 6:
        return True, "StuckSensor", "temperature", "Medium"
    if press_flat_streak >= 6:
        return True, "StuckSensor", "pressure", "Medium"
    if hum_flat_streak >= 6 and humidity < 98.0:
        return True, "StuckSensor", "humidity", "Medium"

    return False, "Normal", "none", "Normal"


# --------------------------------------------------
# FEATURE EXTRACTION & SLIDING BUFFER
# --------------------------------------------------

def extract_online_features(temperature, pressure, humidity, station_id="AWS-24567", dt=None):
    """
    Computes all 22 engineered features online using station-aware sliding history buffer.
    Thread-safe and cold-start resilient.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # 1. Cyclic temporal features
    hour = dt.hour + dt.minute / 60.0
    month = dt.month
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)

    with buffer_lock:
        buf = station_buffers[station_id]

        if len(buf) > 0:
            prev = buf[-1]
            temp_diff = float(temperature - prev["temperature"])
            press_diff = float(pressure - prev["pressure"])
            hum_diff = float(humidity - prev["humidity"])

            # 1-hour rolling window (up to 6 readings)
            recent_6 = list(buf)[-5:] + [{"temperature": temperature, "pressure": pressure, "humidity": humidity}]
            temps_6 = [r["temperature"] for r in recent_6]
            press_6 = [r["pressure"] for r in recent_6]
            hums_6 = [r["humidity"] for r in recent_6]

            temp_roll_std = float(np.std(temps_6))
            press_roll_std = float(np.std(press_6))
            hum_roll_std = float(np.std(hums_6))
            temp_roll_mean = float(np.mean(temps_6))
            temp_roll_dev = float((temperature - temp_roll_mean) / (temp_roll_std + 1e-3))

            # 3-hour rolling window (up to 18 readings)
            recent_18 = list(buf)[-17:] + [{"temperature": temperature, "pressure": pressure, "humidity": humidity}]
            temps_18 = [r["temperature"] for r in recent_18]
            hums_18 = [r["humidity"] for r in recent_18]
            temp_roll_std_3h = float(np.std(temps_18))
            hum_roll_std_3h = float(np.std(hums_18))

            # Flatline streak counters
            temp_flat_streak = 0
            hum_flat_streak = 0
            press_flat_streak = 0
            for r in reversed(list(buf)):
                if abs(r["temperature"] - temperature) < 1e-4:
                    temp_flat_streak += 1
                else:
                    break
            for r in reversed(list(buf)):
                if abs(r["humidity"] - humidity) < 1e-4:
                    hum_flat_streak += 1
                else:
                    break
            for r in reversed(list(buf)):
                if abs(r["pressure"] - pressure) < 1e-4:
                    press_flat_streak += 1
                else:
                    break
        else:
            # Cold-start fallback
            temp_diff = 0.0
            press_diff = 0.0
            hum_diff = 0.0
            temp_roll_std = 0.0
            press_roll_std = 0.0
            hum_roll_std = 0.0
            temp_roll_dev = 0.0
            temp_roll_std_3h = 0.0
            hum_roll_std_3h = 0.0
            temp_flat_streak = 0
            hum_flat_streak = 0
            press_flat_streak = 0

        # Thermodynamic cross-sensor features
        dew_point = calculate_dew_point(temperature, humidity)
        dew_point_spread = float(temperature - dew_point)
        vpd = calculate_vpd(temperature, humidity)

        # Append current reading to station buffer
        buf.append({
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,
            "timestamp": dt
        })

    feature_dict = {
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
        "temp_diff": temp_diff,
        "press_diff": press_diff,
        "hum_diff": hum_diff,
        "temp_roll_std": temp_roll_std,
        "press_roll_std": press_roll_std,
        "hum_roll_std": hum_roll_std,
        "temp_roll_dev": temp_roll_dev,
        "temp_roll_std_3h": temp_roll_std_3h,
        "hum_roll_std_3h": hum_roll_std_3h,
        "temp_flat_streak": float(temp_flat_streak),
        "hum_flat_streak": float(hum_flat_streak),
        "press_flat_streak": float(press_flat_streak),
        "dew_point": dew_point,
        "dew_point_spread": dew_point_spread,
        "vpd": vpd
    }

    return feature_dict


# --------------------------------------------------
# FAULT ATTRIBUTION & SENSOR HEALTH SCORER
# --------------------------------------------------

def attribute_ml_fault(features, anomaly_score, threshold):
    """
    Attributes Tier 2 ML anomalies to specific failure modes and affected sensors.
    """
    temp_dev = abs(features.get("temp_roll_dev", 0.0))
    temp_std = features.get("temp_roll_std", 0.0)
    hum_std = features.get("hum_roll_std", 0.0)
    press_std = features.get("press_roll_std", 0.0)
    dew_spread = features.get("dew_point_spread", 0.0)

    # Spike: sudden high volatility in 1-hour window
    if temp_std > 2.5:
        return "Spike", "temperature"
    if press_std > 2.0:
        return "Spike", "pressure"
    if hum_std > 12.0:
        return "Spike", "humidity"

    # Drop: negative deviation
    if features.get("hum_diff", 0.0) < -15.0:
        return "Drop", "humidity"
    if features.get("temp_diff", 0.0) < -4.0:
        return "Drop", "temperature"
    if features.get("press_diff", 0.0) < -2.5:
        return "Drop", "pressure"

    # Drift or Bias: persistent deviation from rolling baseline
    if temp_dev > 3.0:
        return "Drift", "temperature"

    # Multivariate / thermodynamic mismatch
    if dew_spread < 0.2:
        return "Multivariate", "multiple"

    # General statistical anomaly
    return "Bias", "multiple"


def update_sensor_health(station_id, is_anomaly, anomaly_type, affected_sensor, severity):
    """
    Dynamically computes sensor health (0 - 100%) per sensor.
    Reduces health on faults based on severity; allows gradual recovery on normal readings.
    """
    with buffer_lock:
        current = station_health[station_id]

        if not is_anomaly:
            # Gradual recovery: +5% per normal reading up to 100%
            current["temperature"] = min(100, current["temperature"] + 5)
            current["pressure"] = min(100, current["pressure"] + 5)
            current["humidity"] = min(100, current["humidity"] + 5)
        else:
            # Determine penalty based on severity
            penalty = 35 if severity == "High" else (20 if severity == "Medium" else 10)
            if anomaly_type == "StuckSensor":
                penalty = max(penalty, 40)
            elif anomaly_type == "OutOfBounds":
                penalty = max(penalty, 50)

            sensors_to_penalize = []
            if affected_sensor == "multiple":
                sensors_to_penalize = ["temperature", "pressure", "humidity"]
            elif affected_sensor in current:
                sensors_to_penalize = [affected_sensor]

            for s in sensors_to_penalize:
                current[s] = max(0, current[s] - penalty)

        return dict(current)


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "service": "SkyGuard AI ML Service (Phase 2 Two-Tier)",
        "model": model_name,
        "model_loaded": model is not None and scaler is not None,
        "threshold": float(threshold),
        "features_count": len(FEATURES),
        "active_stations": len(station_buffers),
        "scoring_type": scoring_type
    })


@app.route("/reset", methods=["POST"])
def reset():
    with buffer_lock:
        station_buffers.clear()
        station_health.clear()
    return jsonify({
        "success": True,
        "message": "History buffers and sensor health reset successfully."
    })


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        global model, scaler, threshold
        if model is None or scaler is None:
            if not load_model():
                return jsonify({
                    "success": False,
                    "message": "ML model or scaler is not loaded on server."
                }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "Request body must be valid JSON."
            }), 400

        for param in ["temperature", "pressure", "humidity"]:
            if param not in data or data[param] is None:
                return jsonify({
                    "success": False,
                    "message": f"Missing required parameter: {param}"
                }), 400
            if isinstance(data[param], str) and data[param].strip() == "":
                return jsonify({
                    "success": False,
                    "message": f"Parameter '{param}' cannot be empty or whitespace"
                }), 400
            try:
                val = float(data[param])
                if np.isnan(val) or np.isinf(val):
                    return jsonify({
                        "success": False,
                        "message": f"Parameter '{param}' cannot be NaN or Infinite"
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "message": f"Parameter '{param}' must be a valid numeric value"
                }), 400

        temperature = float(data["temperature"])
        pressure = float(data["pressure"])
        humidity = float(data["humidity"])
        station_id = str(data.get("stationId", data.get("station_id", "AWS-24567")))

        # Parse timestamp if supplied
        dt = None
        if "timestamp" in data and data["timestamp"]:
            try:
                dt = datetime.fromisoformat(str(data["timestamp"]).replace("Z", "+00:00"))
            except Exception:
                dt = None
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Feature Extraction
        feature_dict = extract_online_features(temperature, pressure, humidity, station_id=station_id, dt=dt)

        # --------------------------------------------------
        # TIER 1: DETERMINISTIC SENSOR & PHYSICS EVALUATION
        # --------------------------------------------------
        t1_anomaly, t1_type, t1_sensor, t1_severity = evaluate_tier1(
            temperature=temperature,
            pressure=pressure,
            humidity=humidity,
            dew_point_spread=feature_dict["dew_point_spread"],
            vpd=feature_dict["vpd"],
            temp_diff=feature_dict["temp_diff"],
            press_diff=feature_dict["press_diff"],
            hum_diff=feature_dict["hum_diff"],
            temp_flat_streak=feature_dict["temp_flat_streak"],
            press_flat_streak=feature_dict["press_flat_streak"],
            hum_flat_streak=feature_dict["hum_flat_streak"]
        )

        # --------------------------------------------------
        # TIER 2: MACHINE LEARNING EVALUATION
        # --------------------------------------------------
        feature_vector = [feature_dict[col] for col in FEATURES]
        # Format as pandas DataFrame to prevent scikit-learn unconsumed feature-name warnings
        input_df = pd.DataFrame([feature_vector], columns=FEATURES)
        input_scaled = scaler.transform(input_df)

        # Compute model anomaly score
        if scoring_type == "isolation_forest" or hasattr(model, "score_samples"):
            # Score: higher means more anomalous
            ml_anomaly_score = float(-model.score_samples(input_scaled)[0])
            t2_anomaly = ml_anomaly_score >= threshold
        elif scoring_type == "pca_reconstruction_mse" or hasattr(model, "inverse_transform"):
            recon = model.inverse_transform(model.transform(input_scaled))
            ml_anomaly_score = float(np.mean((input_scaled - recon) ** 2))
            t2_anomaly = ml_anomaly_score >= threshold
        elif scoring_type == "mlp_reconstruction_mse":
            recon = model.predict(input_scaled)
            ml_anomaly_score = float(np.mean((input_scaled - recon) ** 2))
            t2_anomaly = ml_anomaly_score >= threshold
        else:
            ml_anomaly_score = float(-model.decision_function(input_scaled)[0])
            t2_anomaly = ml_anomaly_score >= threshold

        # --------------------------------------------------
        # COMBINED TWO-TIER DECISION & ATTRIBUTION
        # --------------------------------------------------
        if t1_anomaly:
            is_anomaly = True
            tier_source = "Tier 1 (Physics/Rules)"
            anomaly_type = t1_type
            affected_sensor = t1_sensor
            severity = t1_severity
            # Normalize display score for Tier 1
            combined_score = max(ml_anomaly_score, threshold * 1.5)
        elif t2_anomaly:
            is_anomaly = True
            tier_source = "Tier 2 (ML)"
            anomaly_type, affected_sensor = attribute_ml_fault(feature_dict, ml_anomaly_score, threshold)
            # Calibrated ML severity
            if ml_anomaly_score >= threshold * 1.35:
                severity = "High"
            elif ml_anomaly_score >= threshold * 1.15:
                severity = "Medium"
            else:
                severity = "Low"
            combined_score = ml_anomaly_score
        else:
            is_anomaly = False
            tier_source = "None"
            anomaly_type = "Normal"
            affected_sensor = "none"
            severity = "Normal"
            combined_score = ml_anomaly_score

        # Dynamic Sensor Health
        health_scores = update_sensor_health(station_id, is_anomaly, anomaly_type, affected_sensor, severity)

        prediction_val = -1 if is_anomaly else 1

        # Comprehensive response preserving backward-compatible data object
        return jsonify({
            "success": True,
            "status": "Anomaly" if is_anomaly else "Normal",
            "anomaly_score": float(combined_score),
            "severity": severity,
            "anomaly_type": anomaly_type,
            "affected_sensor": affected_sensor,
            "timestamp": dt.isoformat(),
            "sensor_health": health_scores,
            "tier": tier_source,
            "data": {
                "temperature": temperature,
                "pressure": pressure,
                "humidity": humidity,
                "prediction": int(prediction_val),
                "anomaly": bool(is_anomaly),
                "label": "Anomaly" if is_anomaly else "Normal",
                "severity": severity,
                "anomalyScore": float(combined_score),
                "threshold": float(threshold),
                "model": model_name,
                "tier": tier_source,
                "anomalyType": anomaly_type,
                "affectedSensor": affected_sensor,
                "sensorHealth": health_scores
            }
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


# --------------------------------------------------
# MICROSERVICE ENTRYPOINT
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ML Model Successfully Integrated & Running")
    print("=" * 60)
    print("Model Architecture : Two-Tier (Deterministic Rules + ML)")
    print("ML Engine          :", model_name)
    print("Engineered Features:", len(FEATURES))
    print("Decision Threshold :", threshold)
    print("Server starting on Render")
    print("=" * 60)

app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    debug=False
)