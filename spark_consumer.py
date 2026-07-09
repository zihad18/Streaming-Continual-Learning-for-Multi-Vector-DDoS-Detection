import socket
import time
import json
import os
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import joblib

from pyspark import TaskContext, SparkFiles
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)

# ---------------------------- Environment / Config ----------------------------
MODEL_DIR = os.getenv("MODEL_DIR", ".")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "kdd_stream")
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "10.160.0.8:9092")
N_TREES = int(os.getenv("N_TREES", "5"))
ERROR_THRESHOLD = float(os.getenv("ERROR_THRESHOLD", "0.35"))

LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-bucket-name")

# ----------------------------- Metrics Schema --------------------------------
metrics_schema = StructType([
    StructField("node_host", StringType(), True),
    StructField("partition_id", IntegerType(), True),
    StructField("record_id", IntegerType(), True),
    StructField("true_label", IntegerType(), True),
    StructField("final_pred", IntegerType(), True),
    StructField("accuracy", DoubleType(), True),
    StructField("precision", DoubleType(), True),
    StructField("recall", DoubleType(), True),
    StructField("f1", DoubleType(), True),
    StructField("latency_ms", DoubleType(), True),
    StructField("event_time", DoubleType(), True),
])


# ---------------------------- Helper Functions -------------------------------
def row_to_dict(row):
    return {f"f{i}": float(v) for i, v in enumerate(row)}


def safe_json_loads(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def get_model_file(filename):
    p1 = os.path.join(MODEL_DIR, filename)
    p2 = SparkFiles.get(filename)
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Cannot find {filename}. Tried {p1} and {p2}")


# --------------------------- RealTimeOAW (unchanged) -------------------------
class RealTimeOAW:
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


# ------------------------- SparkHoeffdingEnsemble ----------------------------
class SparkHoeffdingEnsemble:
    def __init__(self, n_trees=N_TREES):
        self.preprocessor = joblib.load(get_model_file("preprocessor.pkl"))
        self.models = [
            joblib.load(get_model_file(f"tree_{i}.pkl"))
            for i in range(n_trees)
        ]
        self.detectors = [
            RealTimeOAW(Ath=1.5, Dth=2.0, Ls=200, La=500)
            for _ in range(n_trees)
        ]
        self.n_trees = n_trees
        self.executor = ThreadPoolExecutor(max_workers=n_trees)

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
            pred = max(proba, key=proba.get) if proba else 0
        pred = int(pred)
        confidence = float(proba.get(pred, 1.0)) if proba else 0.5
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
        print("Executor processed records:", self.total)
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

    def process_records(self, records):
        # FIX: return [] instead of None when empty
        if len(records) == 0:
            return []

        pdf = pd.DataFrame(records)
        has_label = "label" in pdf.columns
        labels = pdf["label"].astype(int).values if has_label else [None] * len(pdf)
        record_ids = (
            pdf["record_id"].astype(int).values
            if "record_id" in pdf.columns
            else range(self.total, self.total + len(pdf))
        )
        features = pdf.drop(columns=["label", "record_id"], errors="ignore")

        # Preprocess
        cat_cols = self.preprocessor.transformers_[0][2]
        for col in cat_cols:
            if col in features.columns:
                features[col] = features[col].astype(str)
        X = self.preprocessor.transform(features).toarray()

        metrics = []
        for idx, x in enumerate(X):
            t0 = time.time()                     # start time for latency
            y = None if labels[idx] is None else int(labels[idx])
            rid = int(record_ids[idx])
            x_dict = row_to_dict(x)

            # Parallel predictions
            results = list(self.executor.map(
                lambda model: self.prediction_and_score(model, x_dict),
                self.models
            ))
            tree_preds = [pred for pred, _ in results]
            anomaly_scores = [score for _, score in results]
            final_pred = self.majority_vote(tree_preds)

            if y is not None:
                self.update_metrics(y, final_pred)
            else:
                self.total += 1

            states = []
            # Sequential drift detection & retraining
            for tree_id, model in enumerate(self.models):
                pred = tree_preds[tree_id]
                score = anomaly_scores[tree_id]

                if y is not None:
                    self.tree_total[tree_id] += 1
                    if pred == y:
                        self.tree_correct[tree_id] += 1

                # ---- Correct drift logic (restored) ----
                prior_condition = self.detectors[tree_id].condition
                if prior_condition == "Normal" or y is None:
                    error_signal = int(score >= ERROR_THRESHOLD)
                    y_S = final_pred
                else:
                    error_signal = int(y != pred)
                    y_S = y

                condition, retrain, retrain_data = self.detectors[tree_id].update(
                    anomaly_signal=error_signal,
                    sample=(x_dict, y_S)          # store feature vector and pseudo‑label
                )
                # ----------------------------------------

                if retrain and len(retrain_data) > 0:
                    self.tree_drifts[tree_id] += 1
                    print(f"[DRIFT] Tree {tree_id} updating with {len(retrain_data)} "
                          f"pseudo-labeled adaptive records")
                    for old_x, pseudo_y in retrain_data:
                        model.learn_one(old_x, int(pseudo_y))

                states.append(condition)

            # Compute aggregated metrics
            accuracy = self.correct / max(self.total, 1)
            precision = self.tp / max(self.tp + self.fp, 1)
            recall = self.tp / max(self.tp + self.fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            latency_ms = (time.time() - t0) * 1000

            metrics.append({
                "record_id": rid,
                "true_label": -1 if y is None else y,
                "final_pred": final_pred,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency_ms": latency_ms,
                "event_time": t0,                 # use start time
            })

            if self.total % 500 == 0:
                self.print_metrics(rid, y, final_pred, tree_preds, anomaly_scores, states)

        return metrics


# ---------------------------- Spark Processing ------------------------------
_executor_ensemble = None


def process_partition(rows):
    global _executor_ensemble
    if _executor_ensemble is None:
        _executor_ensemble = SparkHoeffdingEnsemble(n_trees=N_TREES)

    records = []
    for row in rows:
        records.append(safe_json_loads(row["value"]))

    metrics = _executor_ensemble.process_records(records)
    if not metrics:
        return

    tc = TaskContext.get()
    partition_id = tc.partitionId() if tc else -1
    host = socket.gethostname()

    for m in metrics:
        yield Row(
            node_host=host,
            partition_id=partition_id,
            record_id=m["record_id"],
            true_label=m["true_label"],
            final_pred=m["final_pred"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            latency_ms=m["latency_ms"],
            event_time=m["event_time"],
        )


def foreach_batch_function(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return

    metrics_rdd = batch_df.rdd.mapPartitions(process_partition)
    metrics_df = batch_df.sparkSession.createDataFrame(metrics_rdd, schema=metrics_schema)

    # ---- Local vs GCS output ----
    if LOCAL_MODE:
        output_path = "file:///C:/temp/metrics"
    else:
        output_path = f"gs://{GCS_BUCKET}/metrics/batch_id={batch_id}"

    metrics_df.write.mode("append").json(output_path)
    print(f"[Batch {batch_id}] Metrics written to {output_path}")


def main():
    builder = SparkSession.builder.appName("GCP-Distributed-KDD-Hoeffding-Ensemble-Threaded")
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

    # ---- Checkpoint location: local or GCS ----
    if LOCAL_MODE:
        checkpoint = "file:///C:/temp/metrics"
    else:
        checkpoint = f"gs://{GCS_BUCKET}/checkpoints/kdd_stream_consumer"

    query = (
        values_df.writeStream
        .foreachBatch(foreach_batch_function)
        .outputMode("append")
        .option("checkpointLocation", checkpoint)
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()