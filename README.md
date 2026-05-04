# Smart Adaptive Environment Monitoring & Decision System

An intelligent IoT and AI-driven framework designed to monitor environmental conditions, learn complex patterns, and execute autonomous decisions based on real-time sensor data, as outlined in **Scenario 1.pdf**.

---

## 🚀 Key Features
* **Multi-Sensor Monitoring**: Tracks temperature, gas levels, light, and humidity.
* **Intelligent Classification**: Categorizes environments into Normal, Warning, or Dangerous states using SOM.
* **Autonomous Response**: Executes adaptive decisions like activating fans, alarms, or irrigation.
* **Graph-Based Modeling**: Utilizes **Graph Neural Networks (GNN)** to model sensor relationships.

---

## 🧠 AI & Machine Learning Stack

The system employs a sophisticated pipeline to process data and optimize performance:

| Component | Methodology | Purpose |
| :--- | :--- | :--- |
| **Preprocessing** | PCA | Dimensionality reduction and noise removal. |
| **Clustering** | SOM | Environment state classification. |
| **Adaptive Learning** | ART1 / ART2 | Detection of new, unseen binary or continuous patterns. |
| **Prediction** | RBF Network | Forecasting temperature trends and gas danger levels. |
| **Optimization** | Genetic Algorithm | Fine-tuning ANN weights, RBF centers, and Fuzzy rules. |

---

## 🛠 How it Works

1. **Data Acquisition**: Sensors act as nodes in a GNN to capture spatial relationships.
2. **Pattern Recognition**: The system uses Fuzzy Logic to convert sensor data into human-like rules.
3. **Continuous Evolution**: Through **Reinforcement Learning**, the system improves its decision-making accuracy over time.

---
