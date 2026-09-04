"""
SkyGuard AI - Model Experimentation & Validation Engine
Experiments with:
1. Isolation Forest
2. One-Class SVM
3. Local Outlier Factor (LOF) with Novelty Detection
4. PCA Reconstruction Error
5. MLP Deep Autoencoder

Compares:
- Precision, Recall, F1, PR-AUC, False Positive Rate (FPR), Inference Latency (ms/sample)
- Optimizes threshold strictly on validation set
- Evaluates winning model on test set
- Performs detailed error analysis
- Saves best model, scaler, threshold, and metadata
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
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from dataset_utils import (
    prepare_train_val_test,
    ENGINEERED_FEATURES
)

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ML_DIR, "model")
DATA_PATH = os.path.join(ML_DIR, "data", "jena_climate.csv")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

os.makedirs(MODEL_DIR, exist_ok=True)


def optimize_threshold(y_true, scores, percentiles=None):
    """
    Finds the optimal threshold on validation scores to maximize F1-score.
    """
    if percentiles is None:
        percentiles = np.linspace(85.0, 99.8, 120)

    candidate_thresholds = np.percentile(scores, percentiles)
    best_f1 = -1.0
    best_thresh = candidate_thresholds[0]
    best_metrics = {}

    for thresh in candidate_thresholds:
        y_pred = (scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
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


# Native sklearn models are used directly for full pickling portability


def run_experiments():
    print("=" * 70)
    print("SKYGUARD AI - PHASE 2 MODEL EXPERIMENTATION & BENCHMARKING")
    print("=" * 70)

    # 1. Prepare data
    train_df, val_df, test_df, scaler = prepare_train_val_test(
        DATA_PATH, start_year=2022, end_year=2023
    )

    X_train = scaler.transform(train_df[ENGINEERED_FEATURES])
    X_val = scaler.transform(val_df[ENGINEERED_FEATURES])
    X_test = scaler.transform(test_df[ENGINEERED_FEATURES])

    y_val = val_df["is_anomaly"].values
    y_test = test_df["is_anomaly"].values

    print(f"\nFeature count: {X_train.shape[1]}")
    print(f"Train size: {X_train.shape[0]:,}, Val size: {X_val.shape[0]:,}, Test size: {X_test.shape[0]:,}")

    models = {}

    # 1. Isolation Forest
    print("\n[1/5] Training Isolation Forest...")
    t0 = time.perf_counter()
    iso = IsolationForest(n_estimators=100, contamination=0.045, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    iso_fit_time = time.perf_counter() - t0
    # Higher score = more anomalous
    iso_score_fn = lambda m, X: -m.score_samples(X)
    models["Isolation Forest"] = (iso, iso_score_fn, iso_fit_time)

    # 2. One-Class SVM (subsampled to 12,000 for realistic RBF training time)
    print("[2/5] Training One-Class SVM (RBF kernel, 12k support subsample)...")
    t0 = time.perf_counter()
    sub_idx = np.random.RandomState(42).choice(len(X_train), size=min(12000, len(X_train)), replace=False)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.045)
    ocsvm.fit(X_train[sub_idx])
    ocsvm_fit_time = time.perf_counter() - t0
    ocsvm_score_fn = lambda m, X: -m.decision_function(X)
    models["One-Class SVM"] = (ocsvm, ocsvm_score_fn, ocsvm_fit_time)

    # 3. Local Outlier Factor (LOF) with Novelty=True
    print("[3/5] Training Local Outlier Factor (LOF, novelty=True, 15k subsample)...")
    t0 = time.perf_counter()
    lof_idx = np.random.RandomState(42).choice(len(X_train), size=min(15000, len(X_train)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=25, novelty=True, contamination=0.045, n_jobs=-1)
    lof.fit(X_train[lof_idx])
    lof_fit_time = time.perf_counter() - t0
    lof_score_fn = lambda m, X: -m.score_samples(X)
    models["Local Outlier Factor (LOF)"] = (lof, lof_score_fn, lof_fit_time)

    # 4. PCA Reconstruction Error
    print("[4/5] Training PCA Reconstruction Error...")
    t0 = time.perf_counter()
    pca_model = PCA(n_components=6, random_state=42)
    pca_model.fit(X_train)
    pca_fit_time = time.perf_counter() - t0
    pca_score_fn = lambda m, X: np.mean((X - m.inverse_transform(m.transform(X))) ** 2, axis=1)
    models["PCA Reconstruction"] = (pca_model, pca_score_fn, pca_fit_time)

    # 5. MLP Deep Autoencoder
    print("[5/5] Training MLP Deep Autoencoder (22 -> 14 -> 6 -> 14 -> 22)...")
    t0 = time.perf_counter()
    ae_model = MLPRegressor(
        hidden_layer_sizes=(14, 6, 14),
        activation="relu",
        solver="adam",
        max_iter=120,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=7
    )
    ae_model.fit(X_train, X_train)
    ae_fit_time = time.perf_counter() - t0
    ae_score_fn = lambda m, X: np.mean((X - m.predict(X)) ** 2, axis=1)
    models["MLP Autoencoder"] = (ae_model, ae_score_fn, ae_fit_time)

    print("\n" + "=" * 70)
    print("VALIDATION BENCHMARK RESULTS")
    print("=" * 70)

    val_results = []
    trained_artifacts = {}

    for name, (model, score_fn, fit_time) in models.items():
        # Measure inference latency
        t_infer_start = time.perf_counter()
        val_scores = score_fn(model, X_val)
        t_infer_end = time.perf_counter()
        latency_ms_sample = ((t_infer_end - t_infer_start) / len(X_val)) * 1000.0

        # PR-AUC
        pr_auc = average_precision_score(y_val, val_scores)

        # Optimize threshold
        best_thresh, metrics = optimize_threshold(y_val, val_scores)

        val_results.append({
            "model": name,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "pr_auc": round(pr_auc, 4),
            "fpr": round(metrics["fpr"], 4),
            "latency_ms": round(latency_ms_sample, 4),
            "fit_time_s": round(fit_time, 2),
            "threshold": best_thresh
        })

        trained_artifacts[name] = {
            "model": model,
            "score_fn": score_fn,
            "threshold": best_thresh,
            "val_scores": val_scores
        }

    results_df = pd.DataFrame(val_results)
    print(results_df.to_string(index=False))

    # Select best model based on F1, PR-AUC and low FPR
    # Sort primarily by F1 descending, secondarily by PR-AUC
    results_df.sort_values(by=["f1", "pr_auc"], ascending=[False, False], inplace=True)
    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    best_threshold = best_row["threshold"]

    print("\n" + "=" * 70)
    print(f"SELECTED BEST MODEL: {best_model_name}")
    print(f"Reason: Highest Validation F1 ({best_row['f1']:.4f}) and PR-AUC ({best_row['pr_auc']:.4f}) with low FPR ({best_row['fpr']:.4f})")
    print("=" * 70)

    # Final Holdout Test Evaluation of the winning model
    best_model = trained_artifacts[best_model_name]["model"]
    best_score_fn = trained_artifacts[best_model_name]["score_fn"]

    t0_test = time.perf_counter()
    test_scores = best_score_fn(best_model, X_test)
    test_latency_ms = ((time.perf_counter() - t0_test) / len(X_test)) * 1000.0

    test_preds = (test_scores >= best_threshold).astype(int)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    test_pr_auc = average_precision_score(y_test, test_scores)
    cm_test = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm_test.ravel()
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n--- FINAL TEST SET METRICS (Unbiased Holdout) ---")
    print(f"Precision : {test_prec:.4f}")
    print(f"Recall    : {test_rec:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"PR-AUC    : {test_pr_auc:.4f}")
    print(f"FPR       : {test_fpr:.4f}")
    print(f"Latency   : {test_latency_ms:.4f} ms/sample")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Detailed Error Analysis on Test Set
    print("\n" + "=" * 70)
    print("TEST SET ERROR ANALYSIS")
    print("=" * 70)

    test_analysis_df = test_df.copy()
    test_analysis_df["predicted"] = test_preds
    test_analysis_df["score"] = test_scores

    # Breakdown by fault type
    print("\n--- Detection Performance by Fault Type ---")
    fault_groups = test_analysis_df.groupby("fault_type")
    for ftype, grp in fault_groups:
        total = len(grp)
        detected = (grp["predicted"] == 1).sum()
        rate = (detected / total) * 100.0 if total > 0 else 0.0
        print(f"  {ftype:<35}: {detected:>4}/{total:<4} ({rate:6.2f}%)")

    # False positive analysis
    fps = test_analysis_df[(test_analysis_df["is_anomaly"] == 0) & (test_analysis_df["predicted"] == 1)]
    print(f"\nFalse Positives Count: {len(fps)} out of {tn + fp} normal points (FPR: {test_fpr:.2%})")
    if len(fps) > 0:
        print("FP Sample Summary:")
        print(fps[["temperature", "pressure", "humidity", "temp_diff", "dew_point_spread", "score"]].describe().T[["mean", "min", "max"]])

    # False negative analysis
    fns = test_analysis_df[(test_analysis_df["is_anomaly"] == 1) & (test_analysis_df["predicted"] == 0)]
    print(f"\nFalse Negatives Count: {len(fns)} out of {tp + fn} anomaly points (FN Rate: {fn / (tp + fn):.2%})")
    if len(fns) > 0:
        print("FN by Fault Type:")
        print(fns["fault_type"].value_counts())

    # Save final model and metadata
    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS & METADATA")
    print("=" * 70)

    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "skyguard_anomaly_model.joblib")
    SCALER_PATH = os.path.join(MODEL_DIR, "skyguard_scaler.joblib")

    joblib.dump(best_model, FINAL_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    scoring_type = "pca_reconstruction_mse" if "PCA" in best_model_name else (
        "mlp_reconstruction_mse" if "Autoencoder" in best_model_name else (
            "negative_score_samples" if "Forest" in best_model_name or "LOF" in best_model_name else "negative_decision_function"
        )
    )

    metadata = {
        "model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "scoring_type": scoring_type,
        "threshold": float(best_threshold),
        "engineered_features": ENGINEERED_FEATURES,
        "features_count": len(ENGINEERED_FEATURES),
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
        "all_experiments": val_results,
        "training_period": "2022-01-01 to 2023-05-27",
        "validation_period": "2023-05-27 to 2023-09-13",
        "test_period": "2023-09-13 to 2023-12-31"
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved best model to : {FINAL_MODEL_PATH}")
    print(f"Saved scaler to     : {SCALER_PATH}")
    print(f"Saved metadata to   : {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    run_experiments()
