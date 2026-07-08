import json
import os
import joblib
import numpy as np
from collections import Counter

from pyspark.sql import SparkSession

MODEL_DIR = os.getenv("MODEL_DIR", "models")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "kdd_stream")
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
N_TREES = int(os.getenv("N_TREES", "5"))
ERROR_THRESHOLD = float(os.getenv("ERROR_THRESHOLD", 0.35))

# If LOCAL_MODE=true, Spark runs with local[5].
# On Dataproc/cloud, run with LOCAL_MODE=false.
LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"


def row_to_dict(row):
    return {f"f{i}": float(v) for i, v in enumerate(row)}


def safe_json_loads(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


class RealTimeOAW:
    """
    Label-free drift detector.

    It does not use true labels. It watches an online anomaly signal:
        1 = model thinks current record is suspicious
        0 = model thinks current record is normal

    This is closer to production, where incoming traffic has no label.
    """
    def __init__(self, Ath=1.5, Dth=2.0, Ls=200, La=500):
        self.Ath = Ath
        self.Dth = Dth
        self.Ls = Ls
        self.La = La

        self.condition = "Normal"
        self.signal_stream = []
        self.adapt_win = []
        self.j_ar = None

    def compute_ar(self, window):
        return float(np.mean(window)) if len(window) > 0 else 0.0

    def update(self, anomaly_signal, sample):
        self.signal_stream.append(int(anomaly_signal))

        if len(self.signal_stream) < 2 * self.Ls:
            return self.condition, False, []

        current_window = self.signal_stream[-self.Ls:]
        previous_window = self.signal_stream[-2 * self.Ls:-self.Ls]

        ar_current = self.compute_ar(current_window)
        ar_previous = max(self.compute_ar(previous_window), 1e-6)

        retrain = False
        retrain_data = []

        if self.condition == "Normal":
            if ar_current >= self.Ath * ar_previous:
                self.condition = "Alert"
                self.adapt_win = [sample]

        elif self.condition == "Alert":
            if ar_current >= self.Dth * ar_previous:
                self.condition = "Drift"
                self.j_ar = ar_current
                self.adapt_win.append(sample)
                retrain = True
                retrain_data = self.adapt_win.copy()

            elif ar_current < self.Ath * ar_previous or len(self.adapt_win) >= self.La:
                self.condition = "Normal"
                self.adapt_win = []

            else:
                self.adapt_win.append(sample)

        elif self.condition == "Drift":
            if self.j_ar is None:
                self.j_ar = ar_current

            if ar_current >= self.Ath * self.j_ar or len(self.adapt_win) >= self.La:
                retrain = True
                retrain_data = self.adapt_win.copy()
                self.condition = "Normal"
                self.adapt_win = []
                self.j_ar = None
            else:
                self.adapt_win.append(sample)

        return self.condition, retrain, retrain_data


class SparkHoeffdingEnsemble:
    def __init__(self, n_trees=N_TREES):
        self.preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")

        self.models = [
            joblib.load(f"{MODEL_DIR}/tree_{i}.pkl")
            for i in range(n_trees)
        ]

        self.detectors = [
            RealTimeOAW(Ath=1.5, Dth=2.0, Ls=200, La=500)
            for _ in range(n_trees)
        ]

        self.n_trees = n_trees
        self.total = 0
        self.correct = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.seen_normal = 0
        self.seen_anomaly = 0

        self.tree_total = [0] * n_trees
        self.tree_correct = [0] * n_trees
        self.tree_drifts = [0] * n_trees

    def majority_vote(self, preds):
        return Counter(preds).most_common(1)[0][0]

    def prediction_and_score(self, model, x_dict):
        proba = model.predict_proba_one(x_dict) or {}
        pred = model.predict_one(x_dict)

        if pred is None:
            pred = max(proba, key=proba.get) if len(proba) > 0 else 0
        pred = int(pred)

        # Error-likelihood proxy — NOT "probability this is an attack".
        # Low confidence in whatever class was predicted = high chance the
        # prediction is wrong, regardless of whether that class is normal or attack.
        confidence = float(proba.get(pred, 1.0)) if len(proba) > 0 else 0.5
        error_score = 1.0 - confidence

        return pred, error_score
    def update_metrics(self, y, pred):
        self.total += 1
        if y == 0:
            self.seen_normal += 1
        elif y == 1:
            self.seen_anomaly += 1
        if pred == y:
            self.correct += 1
        if y == 1 and pred == 1:
            self.tp += 1
        elif y == 0 and pred == 0:
            self.tn += 1
        elif y == 0 and pred == 1:
            self.fp += 1
        elif y == 1 and pred == 0:
            self.fn += 1

    def print_metrics(self, rid, y, final_pred, tree_preds, scores, states):
        accuracy = self.correct / max(self.total, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        print("=" * 80)
        print("Record ID:", rid)
        if y is not None:
            print("True label:", y)
        print("Tree predictions:", tree_preds)
        print("Tree anomaly scores:", [round(s, 4) for s in scores])
        print("Final ensemble prediction:", final_pred)
        print("Tree states:", states)

        if y is not None:
            print("Ensemble accuracy:", round(accuracy, 4))
            print("Precision:", round(precision, 4))
            print("Recall:", round(recall, 4))
            print("F1:", round(f1, 4))
            print(f"Seen labels: normal={self.seen_normal}, anomaly={self.seen_anomaly}")
            print(f"Confusion matrix: TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn}")

        for tree_id in range(self.n_trees):
            if y is not None:
                acc = self.tree_correct[tree_id] / max(self.tree_total[tree_id], 1)
                print(
                    f"Tree {tree_id}: accuracy={acc:.4f}, "
                    f"drifts={self.tree_drifts[tree_id]}"
                )
            else:
                print(f"Tree {tree_id}: drifts={self.tree_drifts[tree_id]}")

    def process_pdf(self, pdf):
        if len(pdf) == 0:
            return

        has_label = "label" in pdf.columns
        labels = pdf["label"].astype(int).values if has_label else [None] * len(pdf)
        record_ids = pdf["record_id"].astype(int).values if "record_id" in pdf.columns else range(self.total, self.total + len(pdf))

        features = pdf.drop(columns=["label", "record_id"], errors="ignore")

        cat_cols = self.preprocessor.transformers_[0][2]

        for col in cat_cols:
            if col in features.columns:
                features[col] = features[col].astype(str)


        X = self.preprocessor.transform(features).toarray()

        for idx, x in enumerate(X):
            y = None if labels[idx] is None else int(labels[idx])
            rid = int(record_ids[idx])
            x_dict = row_to_dict(x)

            tree_preds = []
            anomaly_scores = []

            for model in self.models:
                pred, score = self.prediction_and_score(model, x_dict)
                tree_preds.append(pred)
                anomaly_scores.append(score)

            final_pred = self.majority_vote(tree_preds)

            if y is not None:
                self.update_metrics(y, final_pred)
            else:
                self.total += 1

            states = []

            for tree_id, model in enumerate(self.models):
                pred = tree_preds[tree_id]
                score = anomaly_scores[tree_id]

                if y is not None:
                    self.tree_total[tree_id] += 1
                    if pred == y:
                        self.tree_correct[tree_id] += 1

                # Label-free signal for drift detection.
                # Class 1/attack probability above 0.5 is treated as suspicious.
                error_signal = int(score >= ERROR_THRESHOLD)

                condition, retrain, retrain_data = self.detectors[tree_id].update(
                    anomaly_signal=error_signal,
                    sample=(x_dict, pred)
                )

                if retrain and len(retrain_data) > 0:
                    self.tree_drifts[tree_id] += 1
                    print(
                        f"[DRIFT] Tree {tree_id} updating with "
                        f"{len(retrain_data)} pseudo-labeled adaptive records"
                    )

                    for old_x, pseudo_y in retrain_data:
                        model.learn_one(old_x, int(pseudo_y))

                states.append(condition)

            if self.total % 500 == 0:
                self.print_metrics(rid, y, final_pred, tree_preds, anomaly_scores, states)


ensemble = SparkHoeffdingEnsemble(n_trees=N_TREES)


def foreach_batch_function(batch_df, batch_id):
    pdf = batch_df.toPandas()

    if len(pdf) == 0:
        return

    records = []
    for _, row in pdf.iterrows():
        records.append(safe_json_loads(row["value"]))

    import pandas as pd
    parsed_pdf = pd.DataFrame(records)
    ensemble.process_pdf(parsed_pdf)


def main():
    builder = (
        SparkSession.builder
        .appName("KDD-Spark-Hoeffding-Ensemble-LabelFreeDrift")
        .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2"
)
    )

    if LOCAL_MODE:
        builder = builder.master("local[5]")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    values_df = kafka_df.selectExpr("CAST(value AS STRING) as value")

    query = (
        values_df.writeStream
        .foreachBatch(foreach_batch_function)
        .outputMode("append")
        .option("checkpointLocation", "checkpoints/kdd_stream_consumer")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()