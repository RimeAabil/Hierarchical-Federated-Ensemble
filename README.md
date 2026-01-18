# Comparative Analysis of Distributed Learning Architectures

This repository contains a comprehensive study comparing three distinct machine learning paradigms—**Centralized**, **Traditional Federated**, and **Hierarchical Federated Ensemble** learning—applied to the Breast Cancer Wisconsin dataset.

The goal is to analyze the trade-offs between **accuracy**, **privacy**, and **robustness** in medical diagnostic systems.

---

##  Methodologies Explained

### 1. Centralized Ensemble Stacking (The "Gold Standard")

**Logic**:
This approach assumes a traditional data warehousing model where all patient records are collected in a single central server. It represents the theoretical upper bound of performance as it has no privacy constraints or data fragmentation issues.

-   **Architecture**:
    -   **Level 0 (Base Learners)**: Three diverse models—Random Forest (RF), Gaussian Naive Bayes (GNB), and Logistic Regression (LR)—are trained on the complete dataset.
    -   **Level 1 (Meta-Learner)**: A Logistic Regression meta-model learns to combine the predictions of the base learners (Stacking) to correct their individual biases.
-   **Pros**: Maximize information utilization; detects global patterns easily.
-   **Cons**: ZERO privacy; single point of failure; high bandwidth cost for data transfer.

**Location**: `ensemble_stacking/`

---

### 2. Traditional Federated Learning (FedAvg)

**Logic**:
This models a privacy-preserving scenario using the standard **FedAvg** algorithm. Data never leaves the client devices (e.g., individual hospitals).

-   **Architecture**:
    -   **Clients**: Each client trains a local instance of a global model (Logistic Regression) on its private partition of data.
    -   **Server**: Aggregates the weights (coefficients) from all clients by averaging them, weighted by the number of samples per client.
-   **Process**:
    1.  Server broadcasts global weights.
    2.  Clients update local models and train for 1 local epoch.
    3.  Clients send updated weights back to the server.
    4.  Repeat.
-   **Pros**: Strong privacy (raw data stays local); lower bandwidth than centralized.
-   **Cons**: vulnerable to **Non-IID** data (data heterogeneity across clients can cause convergence issues).

**Location**: `federated_learning/`

---

### 3. Hierarchical Federated Ensemble Learning (HFEL)

**Logic**:
This is a novel, hybrid architecture designed to combine the **privacy of Federated Learning** with the **robustness of Ensembles**. It introduces a 3-tier hierarchy.

-   **Architecture**:
    -   **Tier 1 (Clients)**: Like standard FL, clients hold private data. However, they are divided into groups.
    -   **Tier 2 (Aggregators)**: Each aggregator manages a specific subset of clients and specializes in ONE type of model:
        -   *Aggregator A*: Aggregates Gaussian Naive Bayes models.
        -   *Aggregator B*: Aggregates Logistic Regression models.
        -   *Aggregator C*: Aggregates Random Forest models.
    -   **Tier 3 (Meta-Server)**: Does NOT aggregate weights. instead, it performs **Stacking** on the *predictions* of the three global aggregators.
-   **Why strict separation?**
    -   It allows the system to learn diverse decision boundaries (linear, probabilistic, tree-based) simultaneously in a distributed setting.
    -   The Meta-Server fixes the weaknesses of individual aggregators.

**Location**: `ensemble_federated_learning/`

---

##  Comparative Results & Analysis

All experiments were conducted with fixed random seeds and consistent Non-IID data partitioning (clients receive asymmetric shards of data).

| Method | Accuracy | Privacy | Robustness |
| :--- | :--- | :--- | :--- |
| **Centralized Ensemble Stacking** | **98.25%** |  None |  High |
| **Traditional Federated Learning** | **97.37%** |  High |  Low (Single Model) |
| **Ensemble Federated Learning (HFEL)** | **95.61%** |  High |  Medium (Ensemble) |

### Key Findings

1.  **The Privacy Tax**: Evaluating the transition from Centralized (98.25%) to HFEL (95.61%) reveals a **~2.6% accuracy drop**. This is the "cost" of privacy—the loss of information incurred by not pooling raw data.
2.  **Robustness of Linear Models**: Traditional FedAvg (97.37%) performed surprisingly well. This indicates that the Breast Cancer dataset is well-separable by linear boundaries, making a simple Logistic Regression highly effective even when data is fragmented.
3.  **The Role of HFEL**: While HFEL scored slightly lower than the single-model baseline here, its architecture provides a crucial advantage: **Parameter Obfuscation**. By using different model architectures in different groups, it makes it significantly harder for an attacker to reverse-engineer the original data from updates, as they would need to compromise multiple distinct aggregators to get a full picture.

---

##  Usage Instructions

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Experiments

**1. Traditional Federated Learning**
```bash
python federated_learning/train.py
```

**2. Hierarchical Federated Ensemble (HFEL)**
```bash
python ensemble_federated_learning/main.py
```

**3. Centralized Benchmark**
```bash
python ensemble_stacking/centralized_benchmark_stacking.py
```
