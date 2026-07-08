# Streaming-Continual-Learning-for-Multi-Vector-DDoS-Detection

---

# Real-Time Label-Free Drift Detection Ensemble with PySpark and River

This project implements a streaming anomaly detection pipeline using an ensemble of online machine learning models. It addresses a core production machine learning challenge: **detecting and adapting to data drift when true labels are unavailable in real time**.

By monitoring a label-free "anomaly signal" (prediction confidence drops), individual models within a PySpark streaming consumer dynamically detect concept drift and retrain themselves on pseudo-labeled streaming data.

---

## ## Architecture Overview

The system architecture consists of three decoupled components running in a streaming loop:

```
[ kdd_stream.csv ] ---> ( kdd_producer.py ) ---> [ Kafka Topic ]
                                                        |
                                                        v
[ Updated Models ] <--- ( spark_consumer.py ) <----------+

```

1. **`train_models.py` (The Initializer):** Samples the classic KDD Cup '99 dataset, builds a scikit-learn preprocessing pipeline, trains an ensemble of `river` Streaming Adaptive Random Forest (`ARFClassifier`) models on distinct data splits, and generates a streaming evaluation dataset with artificial block-wise distribution drifts.
2. **`kdd_producer.py` (The Data Streamer):** Reads the evaluation dataset and streams records into a Apache Kafka topic, simulating production traffic with or without labels.
3. **`spark_consumer.py` (The Core Engine):** A PySpark Structured Streaming application that consumes the Kafka topic. It runs ensemble inference via majority voting and uses a customized **Real-Time Online Anomaly Windowing (OAW)** detector to adaptively self-train models when confidence flags a potential drift.

---

## ## Core Features

- **Label-Free Drift Detection:** Uses the `RealTimeOAW` detector to monitor model prediction confidence (error-likelihood proxy) rather than true target labels, mimicking an authentic production scenario.
- **Online Ensemble Learning:** Leverages `river`'s streaming classifiers to incrementally learn (`learn_one`) from data on the fly without requiring massive batch retraining.
- **Dynamic Pseudo-Labeling:** When a specific tree's detector shifts into a `Drift` state, the tree backfills its recent historical window of adaptive records using its own predictions as pseudo-labels to self-correct.
- **Hybrid Execution:** Built on PySpark Structured Streaming using `foreachBatch`, allowing it to run smoothly on a local machine or scale seamlessly to cloud environments like Google Cloud Dataproc.

---

## ## Prerequisites & Environment Setup

### 1. System Dependencies

Ensure you have the following installed on your system:

- Python 3.8+
- Apache Kafka (running locally or accessible via a server cluster)
- Java Development Kit (JDK 8 or 11) required for Apache Spark

### 2. Python Package Requirements

Install the required packages using pip:

```bash
pip install pyspark sklearn joblib numpy pandas river kafka-python

```

### 3. Environment Variables

The components are highly configurable via environment variables:

| Variable          | Default          | Description                                                                     |
| ----------------- | ---------------- | ------------------------------------------------------------------------------- |
| `MODEL_DIR`       | `models`         | Directory where models and preprocessors are saved.                             |
| `KAFKA_SERVER`    | `localhost:9092` | Kafka broker bootstrap server location.                                         |
| `KAFKA_TOPIC`     | `kdd_stream`     | The target Kafka topic for data ingestion.                                      |
| `LOCAL_MODE`      | `true`           | If `true`, forces Spark to run locally using `local[5]`.                        |
| `ERROR_THRESHOLD` | `0.35`           | Threshold below which prediction uncertainty triggers an anomaly signal.        |
| `SEND_LABEL`      | `true`           | Set to `false` to simulate a pure production stream without true ground truths. |

---

## ## Execution Guide

Follow these steps sequentially to execute the entire streaming pipeline:

### Step 1: Initialize and Train Base Models

Run the training script to fetch the dataset, fit the preprocessor, train the initial tree ensemble, and generate the drift-heavy streaming data file.

```bash
python train_models.py

```

### Step 2: Start the PySpark Stream Consumer

Launch the Spark consumer first so it is listening to the Kafka topic and ready to process incoming micro-batches immediately.

```bash
python spark_consumer.py

```

### Step 3: Start the Kafka Producer Stream

In a separate terminal, execute the producer to begin streaming data records into the Kafka ecosystem.

```bash
python kdd_producer.py

```

---

## ## Understanding the Code Components

### `RealTimeOAW` (Within `spark_consumer.py`)

This state machine keeps track of a moving window (`Ls`) of model error signals. It transitions between three distinct states based on an Anomaly Ratio ($ar$):

- **`Normal`**: System performance is within expected baselines.
- **`Alert`**: The recent window shows an escalation of low-confidence predictions ($ar_{current} \ge Ath \times ar_{previous}$). It starts collecting data into an adaptive window buffer.
- **`Drift`**: Performance degrades further ($ar_{current} \ge Dth \times ar_{previous}$). This state triggers an online retraining loop utilizing the collected buffer of records paired with their predicted pseudo-labels.

### Metrics Monitoring

Every 500 records processed by the ensemble, the Spark consumer outputs a detailed breakdown of the streaming metrics directly to stdout:

> **Note:** If `SEND_LABEL` is true, the consumer will output real-time validation metrics including **Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix** alongside individual tree status tracking.

---
