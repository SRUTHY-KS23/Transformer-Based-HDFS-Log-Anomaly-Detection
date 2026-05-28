
# Transformer-Based HDFS Log Anomaly Detection

## Overview

This project implements a Transformer-based anomaly detection framework for Hadoop Distributed File System (HDFS) logs.

The system parses raw HDFS logs, constructs structured event sequences, and applies a Transformer model with self-attention mechanisms to identify abnormal system behavior. The framework is designed to improve anomaly detection accuracy, scalability, and interpretability in distributed computing environments.

## Features

* HDFS log parsing and preprocessing
* Structured log template generation
* Sequence construction using block IDs
* Transformer-based anomaly detection
* Self-attention based sequence modeling
* Detection of abnormal system behavior
* Streamlit-based visualization interface
* Performance evaluation using precision, recall, accuracy, and F1-score


## Tech Stack

* Python 3.10
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Streamlit


## Dataset

The project uses the publicly available HDFS log dataset from the LogHub repository.



## System Workflow

Raw HDFS Logs
→ Log Parsing
→ Template Extraction
→ Sequence Construction
→ Transformer Modeling
→ Anomaly Detection
→ Result Visualization



## Project Modules

### 1. Log Parsing Module

Converts raw log messages into structured log templates and event IDs.

### 2. Sequence Construction Module

Groups events into ordered sequences using HDFS block identifiers.

### 3. Transformer Modeling Module

Uses self-attention mechanisms to learn normal system behavior patterns.

### 4. Anomaly Detection Module

Identifies abnormal log sequences based on anomaly scores.

### 5. Result Generation Module

Displays anomaly detection results and evaluation metrics.


## Run the Application

```bash
streamlit run app.py
```

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

-

## Future Enhancements

* Real-time log stream analysis
* Explainable AI visualizations
* Cross-system anomaly detection
* Distributed deployment support
* Hybrid Transformer architectures



## License

This project is developed for academic and research purposes.
