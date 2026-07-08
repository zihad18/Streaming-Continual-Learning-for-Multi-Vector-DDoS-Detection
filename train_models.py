import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from river import forest

MODEL_DIR = "models"
DATA_DIR = "data"

N_TREES = int(os.getenv("N_TREES", "5"))
RECORDS_PER_TREE = int(os.getenv("RECORDS_PER_TREE", "3000"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "60000"))

TRAIN_ANOMALY_RATIO = float(os.getenv("TRAIN_ANOMALY_RATIO", "0.45"))
STREAM_ANOMALY_RATIO = float(os.getenv("STREAM_ANOMALY_RATIO", "0.35"))
STREAM_SIZE = int(os.getenv("STREAM_SIZE", "20000"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def row_to_dict(row):
    return {f"f{i}": float(v) for i, v in enumerate(row)}


def decode_value(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def load_kdd(sample_size=SAMPLE_SIZE):
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    labels = df["labels"].apply(decode_value)
    y = (labels != "normal.").astype(int)

    X = df.drop(columns=["labels"])
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(decode_value)

    print("Original label distribution:")
    print(labels.value_counts().head(20))

    print("\nBinary label distribution:")
    print(y.value_counts().rename(index={0: "normal", 1: "anomaly"}))

    return X, y, labels


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.drop(columns=cat_cols).columns.tolist()

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])


def make_mixed_binary_sample(X, y, n_records, anomaly_ratio, random_state):
    n_anom = int(round(n_records * anomaly_ratio))
    n_norm = n_records - n_anom

    normal_idx = np.flatnonzero(y.values == 0)
    anomaly_idx = np.flatnonzero(y.values == 1)

    if len(normal_idx) == 0 or len(anomaly_idx) == 0:
        raise ValueError("Both normal and anomalous records are required.")

    rng = np.random.default_rng(random_state)

    chosen_norm = rng.choice(normal_idx, size=n_norm, replace=n_norm > len(normal_idx))
    chosen_anom = rng.choice(anomaly_idx, size=n_anom, replace=n_anom > len(anomaly_idx))

    chosen = np.concatenate([chosen_norm, chosen_anom])
    rng.shuffle(chosen)

    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)


def train_sanity_check(model, X_tree, y_tree, tree_id):
    tp = tn = fp = fn = 0

    for x, y in zip(X_tree, y_tree.values):
        pred = model.predict_one(row_to_dict(x))
        pred = 0 if pred is None else int(pred)

        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / max(total, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    print(
        f"Model {tree_id} train sanity check: "
        f"TP={tp}, TN={tn}, FP={fp}, FN={fn}, "
        f"accuracy={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}"
    )

    if tp == 0:
        print(f"WARNING: Model {tree_id} learned no anomaly class.")


def make_stream_with_blocks(X, y, n_records, anomaly_ratio, random_state):
    rng = np.random.default_rng(random_state)
    block_size = 500

    blocks = []
    label_blocks = []

    for block_id, start in enumerate(range(0, n_records, block_size)):
        cur_size = min(block_size, n_records - start)

        cur_ratio = float(np.clip(
            rng.normal(anomaly_ratio, 0.12),
            0.10,
            0.70
        ))

        bx, by = make_mixed_binary_sample(
            X,
            y,
            cur_size,
            cur_ratio,
            random_state + 1000 + block_id,
        )

        blocks.append(bx)
        label_blocks.append(by)

    X_stream = pd.concat(blocks, ignore_index=True)
    y_stream = pd.concat(label_blocks, ignore_index=True)

    return X_stream, y_stream


def main():
    X_raw, y_raw, labels_raw = load_kdd()

    X_train_pool, X_test_pool, y_train_pool, y_test_pool = train_test_split(
        X_raw,
        y_raw,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    preprocessor = build_preprocessor(X_train_pool)
    preprocessor.fit(X_train_pool)
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")

    print("\nTraining models with Adaptive Random Forest:")
    print(f"RECORDS_PER_TREE={RECORDS_PER_TREE}")
    print(f"TRAIN_ANOMALY_RATIO={TRAIN_ANOMALY_RATIO}")

    for tree_id in range(N_TREES):
        model = forest.ARFClassifier(
            n_models=10,
            seed=RANDOM_STATE + tree_id,
        )

        X_tree_raw, y_tree = make_mixed_binary_sample(
            X_train_pool,
            y_train_pool,
            RECORDS_PER_TREE,
            TRAIN_ANOMALY_RATIO,
            RANDOM_STATE + tree_id,
        )

        X_tree = preprocessor.transform(X_tree_raw).toarray()

        print(
            f"\nModel {tree_id}: "
            f"normal={(y_tree == 0).sum()}, "
            f"anomaly={(y_tree == 1).sum()}"
        )

        for x, y in zip(X_tree, y_tree.values):
            model.learn_one(row_to_dict(x), int(y))

        train_sanity_check(model, X_tree, y_tree, tree_id)

        joblib.dump(model, f"{MODEL_DIR}/tree_{tree_id}.pkl")

    X_stream_raw, y_stream = make_stream_with_blocks(
        X_test_pool,
        y_test_pool,
        STREAM_SIZE,
        STREAM_ANOMALY_RATIO,
        RANDOM_STATE + 999,
    )

    X_stream_raw = X_stream_raw.copy()
    X_stream_raw["label"] = y_stream.values
    X_stream_raw.to_csv(f"{DATA_DIR}/kdd_stream.csv", index=False)

    X_stream_raw.drop(columns=["label"]).to_csv(
        f"{DATA_DIR}/kdd_stream_unlabeled.csv",
        index=False,
    )

    print("\nStream distribution:")
    print(
        X_stream_raw["label"]
        .value_counts(normalize=True)
        .rename(index={0: "normal", 1: "anomaly"})
    )

    print("\nTraining complete.")
    print(f"Saved models in: {MODEL_DIR}/")
    print(f"Saved labeled stream: {DATA_DIR}/kdd_stream.csv")
    print(f"Saved unlabeled stream: {DATA_DIR}/kdd_stream_unlabeled.csv")


if __name__ == "__main__":
    main()