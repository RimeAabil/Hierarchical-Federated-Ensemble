# Hierarchical Federated Ensemble Learning vs. Centralized Stacking

This project implements and compares two advanced machine learning architectures for the Breast Cancer Wisconsin dataset: **Hierarchical Federated Ensemble Learning (HFEL)** and **Centralized Stacking**.

## 1. Approaches Evaluated

### A. Hierarchical Federated Ensemble Learning (HFEL)
A privacy-preserving, distributed architecture designed to train a strong global model without sharing raw patient data.

- **Structure**:
    - **Tier 1 (Clients)**: Individual nodes (e.g., hospitals) holding local private data partitions. They train improved local models using Federated Learning (FedAvg).
    - **Tier 2 (Aggregators)**: Intermediate servers that manage subgroups of clients. Each aggregator specializes in a different model type (Heterogeneous Ensemble):
        - Aggregator 0: **Gaussian Naive Bayes (GNB)**
        - Aggregator 1: **Logistic Regression (LR)**
        - Aggregator 2: **Random Forest (RF)**
    - **Tier 3 (Meta-Learner)**: A central server that combines the predictions from all Aggregators using a **Stacking** technique (Logistic Regression Meta-learner) to produce the final prediction.

- **Process**:
    1. Clients train locally and send encrypted/hashed updates (weights) to their Aggregator.
    2. Aggregators update their distinct global models (GNB, LR, RF) via FedAvg.
    3. Aggregators make predictions on validation data.
    4. The Meta-Learner learns how to weigh these predictions to optimize accuracy.

### B. Centralized Stacking (Benchmark)
A traditional, non-distributed approach where all data is gathered in one location.

- **Structure**:
    - A single server accesses the entire dataset ($X, y$).
    - Trains the same three base models (GNB, LR, RF) on the full training set.
    - Trains a Stacking Classifier on top of them.
- **Purpose**: Serves as the theoretical "upper bound" or "Gold Standard" for performance, as it faces no privacy constraints or data fragmentation.

---

## 2. Methodology & Results

We evaluated both approaches on the Breast Cancer Wisconsin dataset with an 80/20 train/test split.

### Comparative Results

| Metric | Centralized Stacking (Benchmark) | Federated Stacking (HFEL) | Delta (Privacy Cost) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **98.25%** | **95.61%** | -2.64% |
| **Precision** | > 98% | ~94.7% | -3.3% |
| **Recall** | > 99% | ~98.6% | -0.4% |

### Detailed Performance Breakdown

#### Centralized Stacking (98.25%)
- Access to 100% of data allowed the **Logistic Regression** base learner to achieve 98.25% on its own.
- The Stacking ensemble matched this peak performance.
- **Why it wins**: It can see all correlations and outliers simultaneously, optimizing the decision boundary perfectly.

#### Federated Stacking (95.61%)
- **Aggregators' Performance**:
    - Aggregator 0 (GNB): ~92.9%
    - Aggregator 1 (LR): ~92.1%
    - Aggregator 2 (RF): ~92.9%
- **Ensemble Boost**: The Meta-Learner successfully combined these weaker distributed models to reach **95.61%**, outperforming any single distributed aggregator by ~3%.
- **Why it is slightly lower**:
    - **Non-IID Data**: Clients have different slices of data. Averaging weights (FedAvg) is harder when data distributions vary.
    - **Privacy**: No raw data leaves the client. We only share weight updates, which contain less information than the raw rows.

---

## 3. Analysis & Conclusion

### Which is "Better"?

1.  **For Accuracy**: **Centralized Stacking** is the winner (98.25%). If you are allowed to pool all data into one database, this is the best mathematical approach.
2.  **For Real-World Feasibility**: **Federated Stacking** is the winner. In healthcare, privacy laws (GDPR, HIPAA) often forbid sending patient records to a central server.
    - The HFEL approach achieves **95.6% accuracy** (very close to the gold standard) **without ever seeing a patient's file**.
    - This makes it the superior choice for practical, privacy-sensitive deployments.

### Key Takeaway
The **Stacking Ensemble** mechanism proved crucial in the Federated setting. While individual federated models struggled (scoring ~92%), the **Meta-Learner** was able to correct their mistakes and boost the final accuracy to **95.6%**, proving the power of hierarchical ensembles in distributed systems.

---

## 4. How to Run

### Run Centralized Benchmark
```bash
cd hierarchical_federated_ensemble/centralized_ensemble_experiment
python centralized_benchmark_stacking.py
```

### Run Federated Simulation
**Option A: Direct Simulation** (Simple)
```bash
cd hierarchical_federated_ensemble
python main.py
```

**Option B: Socket-Based Distributed Mock** (Realistic)
```bash
cd hierarchical_federated_ensemble
python socket_implementation/launcher.py
```
*(Requires creating the `socket_implementation` folder logic first)*
