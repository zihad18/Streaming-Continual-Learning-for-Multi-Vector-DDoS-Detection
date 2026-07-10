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
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "10.160.0.2:9092")
N_TREES = int(os.getenv("N_TREES", "5"))
ERROR_THRESHOLD = float(os.getenv("ERROR_THRESHOLD", "0.35"))

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
BUCKET = os.getenv("BUCKET", "gs://kdd-streaming-bucket")

# ----------------------------- Metrics Schema --------------------------------
metrics_schema = StructType([
    StructField("node_host", StringType(), True),
    StructField("partition_id", IntegerType(), True),
    StructField("record_id", IntegerType(), True),
    StructField("true_label", IntegerType(), True),
    StructField("final_pred", IntegerType(), True),
    StructField("prob_1", DoubleType(), True),
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
    # Try SparkFiles location (from addFile) first, then local path
    try:
        return SparkFiles.get(filename)
    except Exception:
        p1 = os.path.join(MODEL_DIR, filename)
        if os.path.exists(p1):
            return p1
        raise FileNotFoundError(f"Cannot find {filename}. Tried SparkFiles and {p1}")

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
    # FIX: Load models using get_model_file() – this will load from SparkFiles
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
        prob_1 = float(proba.get(1, 0.0)) if proba else 0.0
        return pred, error_score, prob_1

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

        print("=" * 80, flush=True)
        print("Executor processed records:", self.total, flush=True)
        print("Record ID:", rid, flush=True)
        if y is not None:
            print("True label:", y, flush=True)
        print("Tree predictions:", tree_preds, flush=True)
        print("Tree anomaly scores:", [round(s, 4) for s in scores], flush=True)
        print("Final ensemble prediction:", final_pred, flush=True)
        print("Tree states:", states, flush=True)
        if y is not None:
            print("Ensemble accuracy:", round(accuracy, 4), flush=True)
            print("Precision:", round(precision, 4), flush=True)
            print("Recall:", round(recall, 4), flush=True)
            print("F1:", round(f1, 4), flush=True)
            print(f"Seen labels: normal={self.seen_normal}, anomaly={self.seen_anomaly}", flush=True)
            print(f"Confusion matrix: TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn}", flush=True)

    def process_records(self, records):
        if len(records) == 0:
            return []
        local_counter = 0 
        pdf = pd.DataFrame(records)
        has_label = "label" in pdf.columns
        labels = pdf["label"].astype(int).values if has_label else [None] * len(pdf)
        record_ids = (
            pdf["record_id"].astype(int).values
            if "record_id" in pdf.columns
            else range(self.total, self.total + len(pdf))
        )
        features = pdf.drop(columns=["label", "record_id"], errors="ignore")

        cat_cols = self.preprocessor.transformers_[0][2]
        for col in cat_cols:
            if col in features.columns:
                features[col] = features[col].astype(str)
        X = self.preprocessor.transform(features).toarray()

        metrics = []
        for idx, x in enumerate(X):
            local_counter += 1
            if local_counter % 100 == 0:
                print(f"[EXECUTOR:{socket.gethostname()}] Processed {local_counter} records in this partition", flush=True)
            t0 = time.time()
            y = None if labels[idx] is None else int(labels[idx])
            rid = int(record_ids[idx])
            x_dict = row_to_dict(x)

            results = list(self.executor.map(
                lambda model: self.prediction_and_score(model, x_dict),
                self.models
            ))
            
            tree_preds = [res[0] for res in results]
            anomaly_scores = [res[1] for res in results]
            tree_probs_1 = [res[2] for res in results]
            
            final_pred = self.majority_vote(tree_preds)
            ensemble_prob_1 = sum(tree_probs_1) / len(tree_probs_1) if tree_probs_1 else 0.0

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

                prior_condition = self.detectors[tree_id].condition
                if prior_condition == "Normal" or y is None:
                    error_signal = int(score >= ERROR_THRESHOLD)
                    y_S = final_pred
                else:
                    error_signal = int(y != pred)
                    y_S = y

                condition, retrain, retrain_data = self.detectors[tree_id].update(
                    anomaly_signal=error_signal,
                    sample=(x_dict, y_S)
                )

                if retrain and len(retrain_data) > 0:
                    self.tree_drifts[tree_id] += 1
                    print(f"[DRIFT] Tree {tree_id} updating with {len(retrain_data)} "
                          f"pseudo-labeled adaptive records")
                    for old_x, pseudo_y in retrain_data:
                        model.learn_one(old_x, int(pseudo_y))

                states.append(condition)

            accuracy = self.correct / max(self.total, 1)
            precision = self.tp / max(self.tp + self.fp, 1)
            recall = self.tp / max(self.tp + self.fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            latency_ms = (time.time() - t0) * 1000

            metrics.append({
                "record_id": rid,
                "true_label": -1 if y is None else y,
                "final_pred": final_pred,
                "prob_1": ensemble_prob_1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency_ms": latency_ms,
                "event_time": t0,
            })

            if self.total % 500 == 0:
                self.print_metrics(rid, y, final_pred, tree_preds, anomaly_scores, states)

        return metrics

# ---------------------------- Spark Processing ------------------------------
_executor_ensemble = None

def process_partition(rows):
    global _executor_ensemble
    if _executor_ensemble is None:
        # Now each executor loads the models once from SparkFiles
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
            prob_1=m["prob_1"],
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            latency_ms=m["latency_ms"],
            event_time=m["event_time"],
        )

def foreach_batch_function(batch_df, batch_id):
    print(f"[DRIVER] Batch {batch_id} received", flush=True)

    if batch_df.rdd.isEmpty():
        print(f"[DRIVER] Batch {batch_id} is EMPTY", flush=True)
        return

    count = batch_df.count()
    print(f"[DRIVER] Batch {batch_id}: {count} Kafka records", flush=True)

    metrics_rdd = batch_df.rdd.mapPartitions(process_partition)

    metrics_df = batch_df.sparkSession.createDataFrame(
        metrics_rdd,
        schema=metrics_schema
    )

    if LOCAL_MODE:
        output_path = "file:///C:/temp/metrics"
    else:
        output_path = f"{BUCKET}/metrics/batch_id={batch_id}"

    # Write as one fast file (coalesce to 1 partition)
    metrics_df.write.mode("append").json(output_path)
    print(f"[DRIVER] Batch {batch_id} metrics written to {output_path}", flush=True)

# ---------------------------- MAIN (FIXED) ----------------------------------
def main():
    builder = SparkSession.builder.appName("GCP-Distributed-KDD-Hoeffding-Ensemble-Threaded")
    if LOCAL_MODE:
        builder = builder.master("local[5]")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # -------------------------------------------------------------
    # FIX 1: Distribute model files to all executors using addFile()
    # No loading on the driver – this avoids OOM.
    # -------------------------------------------------------------
    model_files = ["preprocessor.pkl"] + [f"tree_{i}.pkl" for i in range(N_TREES)]
    for fname in model_files:
        # If MODEL_DIR is a local path, add that file
        # If MODEL_DIR is a GCS path, you can use a gs:// URI directly
        file_path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(file_path):
            spark.sparkContext.addFile(file_path)
        else:
            # If not found, assume it's already in the working directory
            spark.sparkContext.addFile(fname)
    print("[DRIVER] Model files distributed via SparkFiles.", flush=True)

    # -------------------------------------------------------------
    # FIX 2: Read ALL records in ONE single batch and stop.
    # -------------------------------------------------------------
    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", "30000")  # Bigger than your dataset
        .load()
    )
    values_df = kafka_df.selectExpr("CAST(value AS STRING) as value")

    if LOCAL_MODE:
        checkpoint = "file:///C:/temp/metrics"
    else:
        checkpoint = f"{BUCKET}/checkpoints2/kdd_stream_consumer"

    # -------------------------------------------------------------
    # FIX 3: trigger(once=True) -> one batch only
    # -------------------------------------------------------------
    query = (
        values_df.writeStream
        .foreachBatch(foreach_batch_function)
        .outputMode("append")
        .trigger(once=True)
        .option("checkpointLocation", checkpoint)
        .start()
    )

    query.awaitTermination()

if __name__ == "__main__":
    main()