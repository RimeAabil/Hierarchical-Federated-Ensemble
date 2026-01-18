# Breast Cancer Learning Comparison

This repository presents a comparative study of three distributed machine learning approaches on the Breast Cancer Wisconsin dataset.

## 🎯 Objective
To evaluate and compare the performance of **Traditional Federated Learning**, **Hierarchical Federated Ensemble Learning**, and **Centralized Stacking** in terms of classification accuracy.

## 📂 Repository Structure

```
breast-cancer-learning-comparison/
├── federated_learning/           # [NEW] Traditional FedAvg baseline (Logistic Regression)
│   ├── client.py
│   ├── server.py
│   ├── model.py
│   └── train.py
│
├── ensemble_federated_learning/  # [EXISTING] Hierarchical Federated Ensemble (HFEL)
│   ├── main.py
│   ├── aggregators/
│   ├── clients/
│   └── meta_learner/
│
├── ensemble_stacking/            # [EXISTING] Centralized Stacking Benchmark
│   └── centralized_benchmark_stacking.py
│
└── README.md
```

## 📊 Method Comparison & Results

We evaluated all three methods using consistent data splits and random seeds.

| Method | Accuracy | Description |
| :--- | :--- | :--- |
| **Centralized Ensemble Stacking** | **98.25%** | The "Gold Standard". All data available centrally. Uses Stacking (LR, RF, GNB) with an LR Meta-Learner. |
| **Traditional Federated Learning** | **97.37%** | Single global Logistic Regression model trained via **FedAvg**. Surprisingly robust on this dataset. |
| **Ensemble Federated Learning** | **95.61%** | A 3-tier hierarchical system. Integrates heterogeneous updates (RF, GNB, LR) via a Meta-Learner. |

### Analysis
1.  **Centralized Stacking (98.25%)** achieved the highest accuracy, as expected, benefiting from direct access to the entire dataset and ensemble diversity.
2.  **Traditional FedAvg (97.37%)** performed remarkably well, close to the centralized benchmark. This suggests that for the Breast Cancer dataset, a linear decision boundary (Logistic Regression) is highly effective and FedAvg aggregates it efficiently even with data partitioning.
3.  **Ensemble Federated Learning (95.61%)** demonstrates the ability to combine *heterogeneous* models (RF, GNB, LR) without sharing raw data. While slightly lower than the single-model FedAvg here, it offers robustness (diversified models) and improved privacy (tiered aggregation). The Meta-Learner successfully boosted the performance of individual distributed aggregators (which scored ~92-93%) to 95.6%.

## 🚀 How to Run

### 1. Traditional Federated Learning
```bash
python federated_learning/train.py
```

### 2. Ensemble Federated Learning (HFEL)
```bash
python ensemble_federated_learning/main.py
```

### 3. Centralized Stacking
```bash
python ensemble_stacking/centralized_benchmark_stacking.py
```

## 📋 Requirements
- Python 3.8+
- scikit-learn
- numpy
- pandas
