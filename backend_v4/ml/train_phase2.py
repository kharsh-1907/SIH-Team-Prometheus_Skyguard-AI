"""
SkyGuard AI - Phase 2 Production ML Training Pipeline
Two-Tier Anomaly Detection Architecture:
- Tier 1: Deterministic Physical Guardrails & Sensor Fault Rules
- Tier 2: 22-Feature Machine Learning Anomaly Detection
Benchmarks candidate models, optimizes thresholds strictly on validation data,
evaluates on unbiased holdout test set, and serializes production artifacts.
"""
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor

from dataset_utils import (
    prepare_train_val_test,
    ENGINEERED_FEATURES
)

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ML_DIR, "model")
DATA_PATH = os.path.join(ML_DIR, "data", "jena_climate.csv")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
PHASE2_MODEL_PATH = os.path.join(MODEL_DIR, "phase2_model.joblib")
PHASE2_SCALER_PATH = os.path.join(MODEL_DIR, "phase2_scaler.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)


def compute_tier1_mask(df):
    """
    Tier 1: Deterministic Physics & Sensor Fault Detection Rules.
    Returns a boolean mask of Tier 1 anomalies.
    """
    # 1. Physical range bounds
    oob = (
        (df["temperature"] < -40.0) | (df["temperature"] > 55.0) |
        (df["pressure"] < 800.0) | (df["pressure"] > 1100.0) |
        (df["humidity"] < 0.0) | (df["humidity"] > 100.0)
    )

    # 2. Maximum plausible 10-minute rate of change
    roc = (
        (df["temp_diff"].abs() > 6.0) |
        (df["press_diff"].abs() > 4.0) |
        (df["hum_diff"].abs() > 30.0)
    )

    # 3. Thermodynamic physical consistency
    thermo = (
        (df["dew_point_spread"] < -0.5) |
        (df["vpd"] < 0.0)
    )

    # 4. Stuck sensor flatline detection (>= 6 consecutive 10-min readings = 1 hr)
    # Non-saturated humidity exception prevents false alarms in foggy/rainy conditions
    stuck = (
        (df["temp_flat_streak"] >= 6) |
        (df["press_flat_streak"] >= 6) |
        ((df["hum_flat_streak"] >= 6) & (df["humidity"] < 98.0))
    )

    return (oob | roc | thermo | stuck).values


def optimize_two_tier_threshold(y_true, tier1_flags, scores, percentiles=None):
    """
    Finds optimal score threshold for Tier 2 that maximizes combined (Tier 1 | Tier 2) F1-score on validation data.
    """
    if percentiles is None:
        percentiles = np.linspace(80.0, 99.8, 120)

    candidate_thresholds = np.percentile(scores, percentiles)
    best_f1 = -1.0
    best_thresh = candidate_thresholds[0]
    best_metrics = {}

    for thresh in candidate_thresholds:
        combined_pred = tier1_flags | (scores >= thresh)
        f1 = f1_score(y_true, combined_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            prec = precision_score(y_true, combined_pred, zero_division=0)
            rec = recall_score(y_true, combined_pred, zero_division=0)
            cm = confusion_matrix(y_true, combined_pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            best_metrics = {
                "threshold": float(thresh),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "fpr": float(fpr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }

    return best_thresh, best_metrics


def train_and_evaluate():
    print("=" * 75)
    print("SKYGUARD AI - PHASE 2 TWO-TIER ML PIPELINE TRAINING & BENCHMARKING")
    print("=" * 75)

    # 1. Prepare chronological dataset splits
    train_df, val_df, test_df, scaler = prepare_train_val_test(
        DATA_PATH, start_year=2022, end_year=2023
    )

    # Prepare DataFrames with column names to prevent scikit-learn unconsumed feature-name warnings
    X_train_df = train_df[ENGINEERED_FEATURES]
    X_val_df = val_df[ENGINEERED_FEATURES]
    X_test_df = test_df[ENGINEERED_FEATURES]

    X_train = scaler.transform(X_train_df)
    X_val = scaler.transform(X_val_df)
    X_test = scaler.transform(X_test_df)

    y_val = val_df["is_anomaly"].values
    y_test = test_df["is_anomaly"].values

    # Tier 1 masks
    t1_val = compute_tier1_mask(val_df)
    t1_test = compute_tier1_mask(test_df)

    val_t1_tp = int((t1_val & (y_val == 1)).sum())
    val_t1_fp = int((t1_val & (y_val == 0)).sum())
    test_t1_tp = int((t1_test & (y_test == 1)).sum())
    test_t1_fp = int((t1_test & (y_test == 0)).sum())

    print("\n--- Tier 1 Deterministic Baseline ---")
    print(f"Val Set  -> TP: {val_t1_tp}, FP: {val_t1_fp} (Precision: {val_t1_tp / max(1, val_t1_tp + val_t1_fp):.2%})")
    print(f"Test Set -> TP: {test_t1_tp}, FP: {test_t1_fp} (Precision: {test_t1_tp / max(1, test_t1_tp + test_t1_fp):.2%})")

    # 2. Candidate Models Benchmarking
    print("\n" + "-" * 75)
    print("TRAINING & EVALUATING TIER 2 CANDIDATE MODELS")
    print("-" * 75)

    candidates = {}

    # Candidate 1: Isolation Forest
    print("\n[1/5] Training Isolation Forest...")
    t0 = time.perf_counter()
    iso = IsolationForest(n_estimators=100, contamination=0.045, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_fit_time = time.perf_counter() - t0
    iso_score_fn = lambda m, X: -m.score_samples(X)
    candidates["Isolation Forest"] = (iso, iso_score_fn, iso_fit_time, "isolation_forest")

    # Candidate 2: PCA Reconstruction
    print("[2/5] Training PCA Reconstruction Error (6 components)...")
    t0 = time.perf_counter()
    pca = PCA(n_components=6, random_state=42)
    pca.fit(X_train)
    pca_fit_time = time.perf_counter() - t0
    pca_score_fn = lambda m, X: np.mean((X - m.inverse_transform(m.transform(X))) ** 2, axis=1)
    candidates["PCA Reconstruction"] = (pca, pca_score_fn, pca_fit_time, "pca_reconstruction_mse")

    # Candidate 3: One-Class SVM (12k subsample for training feasibility)
    print("[3/5] Training One-Class SVM (RBF kernel, 12k support)...")
    t0 = time.perf_counter()
    sub_idx = np.random.RandomState(42).choice(len(X_train), size=min(12000, len(X_train)), replace=False)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.045)
    ocsvm.fit(X_train[sub_idx])
    ocsvm_fit_time = time.perf_counter() - t0
    ocsvm_score_fn = lambda m, X: -m.decision_function(X)
    candidates["One-Class SVM"] = (ocsvm, ocsvm_score_fn, ocsvm_fit_time, "negative_decision_function")

    # Candidate 4: Local Outlier Factor (LOF with novelty=True)
    print("[4/5] Training Local Outlier Factor (LOF, 15k support)...")
    t0 = time.perf_counter()
    lof_idx = np.random.RandomState(42).choice(len(X_train), size=min(15000, len(X_train)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=25, novelty=True, contamination=0.045, n_jobs=-1)
    lof.fit(X_train[lof_idx])
    lof_fit_time = time.perf_counter() - t0
    lof_score_fn = lambda m, X: -m.score_samples(X)
    candidates["Local Outlier Factor (LOF)"] = (lof, lof_score_fn, lof_fit_time, "negative_score_samples")

    # Candidate 5: MLP Autoencoder
    print("[5/5] Training MLP Deep Autoencoder (22 -> 14 -> 6 -> 14 -> 22)...")
    t0 = time.perf_counter()
    ae = MLPRegressor(
        hidden_layer_sizes=(14, 6, 14),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=7
    )
    ae.fit(X_train, X_train)
    ae_fit_time = time.perf_counter() - t0
    ae_score_fn = lambda m, X: np.mean((X - m.predict(X)) ** 2, axis=1)
    candidates["MLP Autoencoder"] = (ae, ae_score_fn, ae_fit_time, "mlp_reconstruction_mse")

    # Benchmark on Validation Set
    print("\n" + "=" * 75)
    print("VALIDATION BENCHMARK RESULTS (Tier 1 + Tier 2 Combined)")
    print("=" * 75)

    benchmark_rows = []
    trained_artifacts = {}

    for name, (model, score_fn, fit_time, score_type) in candidates.items():
        t_infer_start = time.perf_counter()
        val_scores = score_fn(model, X_val)
        t_infer_end = time.perf_counter()
        latency_ms = ((t_infer_end - t_infer_start) / len(X_val)) * 1000.0

        pr_auc = average_precision_score(y_val, val_scores)
        best_thresh, metrics = optimize_two_tier_threshold(y_val, t1_val, val_scores)

        benchmark_rows.append({
            "model": name,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "pr_auc": round(pr_auc, 4),
            "fpr": round(metrics["fpr"], 4),
            "latency_ms": round(latency_ms, 4),
            "fit_time_s": round(fit_time, 2),
            "threshold": best_thresh,
            "scoring_type": score_type
        })

        trained_artifacts[name] = {
            "model": model,
            "score_fn": score_fn,
            "threshold": best_thresh,
            "scoring_type": score_type,
            "val_scores": val_scores
        }

    benchmarks_df = pd.DataFrame(benchmark_rows)
    print(benchmarks_df.to_string(index=False))

    # Select best model strictly by Validation F1
    benchmarks_df.sort_values(by=["f1", "pr_auc"], ascending=[False, False], inplace=True)
    best_row = benchmarks_df.iloc[0]
    best_model_name = best_row["model"]
    best_threshold = best_row["threshold"]
    best_scoring_type = best_row["scoring_type"]
    best_model = trained_artifacts[best_model_name]["model"]
    best_score_fn = trained_artifacts[best_model_name]["score_fn"]

    print("\n" + "=" * 75)
    print(f"WINNING PRODUCTION MODEL: {best_model_name}")
    print(f"Validation F1: {best_row['f1']:.4f} | Precision: {best_row['precision']:.4f} | Recall: {best_row['recall']:.4f}")
    print(f"Optimal Score Threshold: {best_threshold:.6f}")
    print("=" * 75)

    # 3. Final Evaluation on Unbiased Holdout Test Set
    t_test_start = time.perf_counter()
    test_scores = best_score_fn(best_model, X_test)
    test_latency_ms = ((time.perf_counter() - t_test_start) / len(X_test)) * 1000.0

    test_preds = t1_test | (test_scores >= best_threshold)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_pr_auc = average_precision_score(y_test, test_scores)
    cm_test = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm_test.ravel()
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n--- UNBIASED HOLDOUT TEST SET PERFORMANCE ---")
    print(f"Precision : {test_prec:.4f} ({test_prec * 100:.2f}%)")
    print(f"Recall    : {test_rec:.4f} ({test_rec * 100:.2f}%)")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"PR-AUC    : {test_pr_auc:.4f}")
    print(f"FPR       : {test_fpr:.4f} ({test_fpr * 100:.2f}%)")
    print(f"Latency   : {test_latency_ms:.4f} ms/sample")
    print(f"Confusion : TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Detailed Fault Type Breakdown
    test_analysis_df = test_df.copy()
    test_analysis_df["predicted"] = test_preds.astype(int)
    test_analysis_df["ml_score"] = test_scores
    test_analysis_df["t1_flag"] = t1_test.astype(int)

    print("\n--- TEST DETECTION BREAKDOWN BY FAULT TYPE ---")
    fault_groups = test_analysis_df.groupby("fault_type")
    fault_breakdown = {}
    for ftype, grp in fault_groups:
        total = len(grp)
        detected = int((grp["predicted"] == 1).sum())
        t1_det = int((grp["t1_flag"] == 1).sum())
        rate = (detected / total) * 100.0 if total > 0 else 0.0
        fault_breakdown[ftype] = {
            "total": total,
            "detected": detected,
            "tier1_detected": t1_det,
            "detection_rate_pct": round(rate, 2)
        }
        print(f"  {ftype:<35}: {detected:>4}/{total:<4} ({rate:6.2f}%) [Tier 1 caught: {t1_det}]")

    # False Positive Analysis
    fps_df = test_analysis_df[(test_analysis_df["is_anomaly"] == 0) & (test_analysis_df["predicted"] == 1)]
    print(f"\nFalse Positive Analysis: {len(fps_df)} normal readings flagged out of {tn + fp} (FPR: {test_fpr:.2%})")

    # 4. Save Artifacts & Metadata
    print("\n" + "=" * 75)
    print("SERIALIZING PRODUCTION ARTIFACTS")
    print("=" * 75)

    joblib.dump(best_model, PHASE2_MODEL_PATH)
    joblib.dump(scaler, PHASE2_SCALER_PATH)

    # Also keep skyguard_anomaly_model.joblib pointing to this validated best model
    LEGACY_TARGET_MODEL = os.path.join(MODEL_DIR, "skyguard_anomaly_model.joblib")
    LEGACY_TARGET_SCALER = os.path.join(MODEL_DIR, "skyguard_scaler.joblib")
    joblib.dump(best_model, LEGACY_TARGET_MODEL)
    joblib.dump(scaler, LEGACY_TARGET_SCALER)

    metadata = {
        "model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "scoring_type": best_scoring_type,
        "threshold": float(best_threshold),
        "two_tier_architecture": {
            "tier1": {
                "rules": [
                    "Physical range bounds (T: -40 to 55 C, P: 800 to 1100 hPa, RH: 0 to 100 %)",
                    "Rate of change bounds (|dT| > 6 C, |dP| > 4 hPa, |dRH| > 30 %)",
                    "Thermodynamic consistency (T < Tdew - 0.5 C, VPD < 0)",
                    "Stuck sensor detection (streak >= 6 readings = 1 hr, non-saturated RH)"
                ]
            },
            "tier2": {
                "model": best_model_name,
                "features_count": len(ENGINEERED_FEATURES),
                "features": ENGINEERED_FEATURES
            }
        },
        "validation_metrics": best_row.to_dict(),
        "test_metrics": {
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1": float(test_f1),
            "pr_auc": float(test_pr_auc),
            "fpr": float(test_fpr),
            "latency_ms": float(test_latency_ms),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        },
        "fault_breakdown": fault_breakdown,
        "all_experiments": benchmark_rows,
        "engineered_features": ENGINEERED_FEATURES,
        "features_count": len(ENGINEERED_FEATURES),
        "training_period": "2022-01-01 to 2023-05-27",
        "validation_period": "2023-05-27 to 2023-09-13",
        "test_period": "2023-09-13 to 2023-12-31"
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Phase 2 Model   : {PHASE2_MODEL_PATH}")
    print(f"Saved Phase 2 Scaler  : {PHASE2_SCALER_PATH}")
    print(f"Saved Model Metadata  : {METADATA_PATH}")
    print("=" * 75)
    print("PHASE 2 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 75)

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
