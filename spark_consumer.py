import json
import os
import joblib
import numpy as np
from collections import deque, Counter
from river.drift import ADWIN

from pyspark.sql import SparkSession

MODEL_DIR = os.getenv("MODEL_DIR", "models")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "kdd_stream")
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"
N_TREES = int(os.getenv("N_TREES", "5"))

LABEL_DELAY_SIZE = int(os.getenv("LABEL_DELAY", "500"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))  # <-- New env var

def row_to_dict(row):
    return {f"f{i}": float(v) for i, v in enumerate(row)}

def safe_json_loads(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


class SparkHoeffdingEnsemble:
    def __init__(self, n_trees=N_TREES):
        # Load your actual ensemble models
        self.preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
        self.models = [
            joblib.load(f"{MODEL_DIR}/tree_{i}.pkl")
            for i in range(n_trees)
        ]
        self.n_trees = n_trees

        # Buffer to hold (features, true_label)
        self.label_buffer = deque(maxlen=LABEL_DELAY_SIZE)

        # --- PER-TREE ADWIN DETECTORS (Monitoring CONFIDENCE, not ERROR) ---
        self.drift_detectors = [ADWIN() for _ in range(n_trees)]

        # Per-tree metrics
        self.tree_total = [0] * n_trees
        self.tree_correct = [0] * n_trees
        self.tree_drifts = [0] * n_trees

        # Global ensemble metrics
        self.total = 0
        self.correct = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.seen_normal = 0
        self.seen_anomaly = 0
        self.drift_count = 0

    def majority_vote(self, preds):
        return Counter(preds).most_common(1)[0][0]

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

    def print_metrics(self, rid, y, final_pred, confidence, tree_preds, tree_accs):
        accuracy = self.correct / max(self.total, 1)
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        print("=" * 80)
        print(f"Record ID: {rid} | True: {y} | Ensemble Pred: {final_pred} | Conf: {confidence:.2f}")
        print(f"Tree preds: {tree_preds}")
        print(f"Tree accuracies: {[round(a, 3) for a in tree_accs]}")
        print(f"Tree drifts: {self.tree_drifts}")
        print(f"Global: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        print(f"Confusion: TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn}")
        print("=" * 80)

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

            # 1. PREDICT WITH ALL TREES
            tree_preds = []
            for model in self.models:
                pred = model.predict_one(x_dict)
                tree_preds.append(0 if pred is None else int(pred))
            
            final_pred = self.majority_vote(tree_preds)
            confidence = tree_preds.count(final_pred) / self.n_trees

            # 2. UPDATE ENSEMBLE METRICS (if label available)
            if y is not None:
                self.update_metrics(y, final_pred)

            # 3. STORE FOR LABEL LATENCY
            if y is not None:
                self.label_buffer.append((x_dict, y))

            # 4. PER-TREE DRIFT DETECTION USING CONFIDENCE (UNSUPERVISED SIGNAL)
            if len(self.label_buffer) == self.label_buffer.maxlen:
                old_x, old_y = self.label_buffer[0]  # Oldest record (label just arrived)
                
                # Loop over each tree independently
                for tree_id in range(self.n_trees):
                    # ============================================================
                    # CRITICAL CHANGE: Use Confidence, NOT Prediction Error
                    # ============================================================
                    # Get probability for the historical record
                    proba = self.models[tree_id].predict_proba_one(old_x) or {}
                    old_confidence = max(proba.values()) if proba else 0.5
                    
                    # Drift signal: 1 = uncertain (low confidence), 0 = confident
                    # This mimics Option B's label-free drift detection
                    uncertainty_signal = 1 if old_confidence < CONFIDENCE_THRESHOLD else 0
                    
                    # Update this tree's ADWIN with the uncertainty signal
                    self.drift_detectors[tree_id].update(uncertainty_signal)
                    
                    # Update per-tree accuracy metrics (still using true labels)
                    if y is not None:
                        old_pred = self.models[tree_id].predict_one(old_x)
                        old_pred = 0 if old_pred is None else int(old_pred)
                        self.tree_total[tree_id] += 1
                        if old_pred == old_y:
                            self.tree_correct[tree_id] += 1

                    # --- TARGETED DRIFT HANDLING FOR THIS SPECIFIC TREE ---
                    if self.drift_detectors[tree_id].drift_detected:
                        self.tree_drifts[tree_id] += 1
                        self.drift_count += 1
                        
                        # ADWIN's width = number of recent records in the NEW concept
                        drifted_window_size = self.drift_detectors[tree_id].width
                        
                        print(f"[DRIFT] Tree {tree_id} detected shift at record {rid} (Confidence-based)")
                        print(f"   [PROOF] Low-confidence period spans last {drifted_window_size} records.")
                        
                        # Extract the EXACT drifted window from the buffer
                        drifted_records = list(self.label_buffer)[-drifted_window_size:]
                        
                        # Retrain ONLY this tree on anomalies within that specific window
                        anomaly_count = 0
                        for drifted_x, drifted_y in drifted_records:
                            if drifted_y == 1:  # Only attack records in the new concept
                                self.models[tree_id].learn_one(drifted_x, drifted_y)
                                anomaly_count += 1
                        
                        print(f"   Retrained Tree {tree_id} on {anomaly_count} anomalies from the drifted window.")
                        
                        # Reset this tree's ADWIN
                        self.drift_detectors[tree_id].reset()

                # CRITICAL: After checking all trees, remove the oldest record from the buffer
                self.label_buffer.popleft()

            # 5. LOGGING
            if self.total % 500 == 0:
                tree_accs = [
                    self.tree_correct[i] / max(self.tree_total[i], 1) 
                    for i in range(self.n_trees)
                ]
                self.print_metrics(rid, y, final_pred, confidence, tree_preds, tree_accs)


# --- Instantiate and Run Spark ---
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
        .appName("KDD-PerTree-ADWIN-ConfidenceDrift")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.2")
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