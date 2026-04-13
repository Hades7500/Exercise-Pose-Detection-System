import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# ─────────────────────────────────────────────
# FEATURE COLUMNS
# These must match exactly what extract_landmarks.py outputs
# ─────────────────────────────────────────────
ANGLE_FEATURES = [
    "left_elbow_angle", "right_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "elbow_symmetry", "knee_symmetry",
    "shoulder_symmetry", "hip_symmetry",
    "torso_lean",
]

LANDMARK_FEATURES = [
    f"{name}_{axis}"
    for name in [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    for axis in ["x", "y", "z"]
]

ALL_FEATURES = ANGLE_FEATURES + LANDMARK_FEATURES

EXERCISES = ["squat", "pushup"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def balance_classes(df: pd.DataFrame, label_col: str = "form_label") -> pd.DataFrame:
    majority_size = df[label_col].value_counts().max()
    balanced_parts = []
    for label, group in df.groupby(label_col):
        if len(group) < majority_size:
            upsampled = resample(group, replace=True,
                                 n_samples=majority_size, random_state=42)
            balanced_parts.append(upsampled)
        else:
            balanced_parts.append(group)
    return pd.concat(balanced_parts).sample(frac=1, random_state=42)


def get_available_features(df: pd.DataFrame) -> list:
    return [f for f in ALL_FEATURES if f in df.columns]


def train_single_model(df: pd.DataFrame, exercise: str, output_dir: str):
    print(f"\n{'='*55}")
    print(f"  Training: {exercise.upper()}")
    print(f"{'='*55}")

    subset = df[df["exercise"] == exercise].copy()

    if len(subset) == 0:
        print(f"  ⚠️  No data found for {exercise}. Skipping.")
        return None

    # ── Label distribution before balancing ──
    print("\n  Label distribution (raw):")
    print(subset["form_label"].value_counts().to_string(header=False))

    # ── Balance classes ──
    subset = balance_classes(subset)
    print(f"\n  After balancing: {len(subset)} samples per class (approx)")

    # ── Features & labels ──
    features = get_available_features(subset)
    X = subset[features].fillna(0).values
    y = subset["form_label"].values

    # Encode string labels → integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n  Classes: {list(le.classes_)}")
    print(f"  Features: {len(features)}")
    print(f"  Total samples: {len(X)}")

    # ── Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ── Train Random Forest ──
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()

    print(f"\n  Test accuracy: {acc:.1%}")

    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
    print(f"  5-fold CV:     {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_,
                                 zero_division=0))

    # ── Confusion matrix plot ──
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_, ax=ax)
    ax.set_title(f"{exercise} — Confusion Matrix (test set)")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"{exercise}_confusion_matrix.png")
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"\n  Confusion matrix saved → {cm_path}")

    # ── Feature importance plot ──
    importances = clf.feature_importances_
    top_n = min(15, len(features))
    top_idx = np.argsort(importances)[-top_n:][::-1]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh([features[i] for i in top_idx[::-1]],
             importances[top_idx[::-1]], color="#4C72B0")
    ax2.set_title(f"{exercise} — Top {top_n} Feature Importances")
    ax2.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, f"{exercise}_feature_importance.png")
    fig2.savefig(fi_path, dpi=120)
    plt.close(fig2)
    print(f"  Feature importance saved → {fi_path}")

    # ── Save model bundle ──
    bundle = {
        "model":    clf,
        "encoder":  le,
        "features": features,
        "exercise": exercise,
        "accuracy": acc,
        "cv_mean":  cv_scores.mean(),
    }

    model_path = os.path.join(output_dir, f"{exercise}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  ✅  Model saved → {model_path}")
    return bundle


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📂  Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"    Rows: {len(df):,}")
    print(f"    Exercises found: {df['exercise'].unique().tolist()}")
    print(f"    Labels found: {df['form_label'].unique().tolist()}")

    # Check we have the required columns
    missing = [c for c in ["exercise", "form_label"] if c not in df.columns]
    if missing:
        print(f"\n❌  Missing columns: {missing}")
        print("    Make sure you ran extract_landmarks.py first.")
        return

    results = {}
    for exercise in EXERCISES:
        bundle = train_single_model(df, exercise, output_dir)
        if bundle:
            results[exercise] = bundle

    # ── Summary ──
    print(f"\n{'='*55}")
    print("  TRAINING COMPLETE — Summary")
    print(f"{'='*55}")
    for ex, b in results.items():
        print(f"  {ex:<15}  acc={b['accuracy']:.1%}   cv={b['cv_mean']:.1%}")
    print(f"\n  Models saved to: {output_dir}/")
    print("  Next step: run your Streamlit app — the coach will load automatically.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="landmarks.csv")
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()
    main(args.csv, args.output_dir)