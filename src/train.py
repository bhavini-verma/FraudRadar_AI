"""
VoiceGuard — Model Training
src/train.py

Loads features.csv → trains Random Forest + Logistic Regression + Ensemble
Saves model.pkl and scaler.pkl to models/
Prints accuracy, F1, EER, confusion matrix
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble          import RandomForestClassifier, VotingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.model_selection   import train_test_split, GridSearchCV
from sklearn.metrics           import (accuracy_score, f1_score,
                                       classification_report,
                                       confusion_matrix, roc_auc_score,
                                       roc_curve)
from imblearn.over_sampling    import SMOTE

# ── CONFIG ──────────────────────────────────────────
FEATURES_CSV = "features/features.csv"
MODEL_DIR    = "models"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
# ────────────────────────────────────────────────────


def compute_eer(y_true, y_scores):
    """Equal Error Rate — point where FPR == FNR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return round(eer * 100, 2)


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   Saved → {path}")


def plot_roc(y_test, y_scores, path):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#00cfff", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   Saved → {path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load Data ─────────────────────────────────
    print("Loading features.csv ...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"   Total samples: {len(df)}  |  Real: {(df.label==0).sum()}  Fake: {(df.label==1).sum()}")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # ── 2. Split ──────────────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    print(f"\nSplit → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 3. Scale ──────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print("   Scaler saved → models/scaler.pkl")

    # ── 4. SMOTE if imbalanced ────────────────────────
    real_count = (y_train == 0).sum()
    fake_count = (y_train == 1).sum()
    ratio = min(real_count, fake_count) / max(real_count, fake_count)
    if ratio < 0.7:
        print(f"\nClass imbalance detected (ratio={ratio:.2f}) — applying SMOTE")
        sm = SMOTE(random_state=RANDOM_STATE)
        sm = SMOTE(random_state=42, k_neighbors=min(4, sum(y_train==0)-1))
X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"   After SMOTE → {len(X_train)} samples")

    # ── 5. Train Models ───────────────────────────────
    print("\nTraining models ...")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

    # Ensemble Voting
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr)], voting="soft")
    ensemble.fit(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────
    models = {"Random Forest": rf, "Logistic Regression": lr, "Ensemble": ensemble}

    print("\n" + "="*55)
    for name, model in models.items():
        y_pred   = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        acc  = accuracy_score(y_test, y_pred) * 100
        f1   = f1_score(y_test, y_pred) * 100
        auc  = roc_auc_score(y_test, y_scores) * 100
        eer  = compute_eer(y_test, y_scores)
        fn   = confusion_matrix(y_test, y_pred)[1][0]  # fake predicted as real

        print(f"\n{name}")
        print(f"  Accuracy : {acc:.2f}%")
        print(f"  F1 Score : {f1:.2f}%")
        print(f"  ROC-AUC  : {auc:.2f}%")
        print(f"  EER      : {eer}%")
        print(f"  False Negatives (AI→Real): {fn}  ← minimize this")
        print(classification_report(y_test, y_pred,
                                    target_names=["Real", "Fake"]))
    print("="*55)

    # ── 7. Save best model (Random Forest) ───────────
    joblib.dump(rf, f"{MODEL_DIR}/model.pkl")
    print("\n✅ Best model saved → models/model.pkl")

    # ── 8. Plots ──────────────────────────────────────
    cm = confusion_matrix(y_test, rf.predict(X_test))
    plot_confusion_matrix(cm, "Confusion Matrix — Random Forest",
                          f"{RESULTS_DIR}/confusion_matrix.png")

    rf_scores = rf.predict_proba(X_test)[:, 1]
    plot_roc(y_test, rf_scores, f"{RESULTS_DIR}/roc_curve.png")

    print("\nDone. Run predict.py to test on individual files.")


if __name__ == "__main__":
    main()