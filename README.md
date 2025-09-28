# 🛡️ Behavioral Fraud Detection Pipeline

This project implements a robust fraud detection pipeline using behavioral, velocity, and anomaly-based features. It flags high-risk transactions and evaluates model performance using precision, recall, PR-AUC, ROC-AUC, and ranking metrics like Precision@K.

---

## 🚀 Features

- Behavioral scoring (typing speed, navigation speed, geo-distance)
- Velocity features (1h, 24h, 30d transaction counts)
- Amount anomaly detection (user-level rolling averages)
- Unique device tracking per user
- Model scoring using XGBoost pipeline
- Precision@K, PR-AUC, ROC-AUC, and alert generation

---

## 📁 Folder Structure
fraud-detection-pipeline/ 
├── run_fraud_pipeline.py 
├── requirements.txt 
├── README.md 
├── data/ 
│   └── sample_transactions.csv 
├── artifacts/ 
│   └── xgb_behavioral_pipeline.joblib 
├── logs/ 
│   └── model_metrics.txt 
├── alerts.csv


---

## 🧠 Metrics Explained

| Metric              | Description |
|---------------------|-------------|
| **PR-AUC**          | Measures ranking quality for fraud detection. Higher is better. |
| **ROC-AUC**         | Measures overall classification ability. 0.5 = random. |
| **Precision@1%**    | Of the top 1% scored transactions, how many are actual frauds. |
| **Top-k threshold** | Minimum score required to be in the top 1%. |

---

## 📦 How to Run

1. Clone the repo  
2. Place your dataset in the `data/` folder  
3. Run the pipeline:
4. Alerts will be saved to alerts.csv
5. Metrics will be printed and optionally saved to logs/model_metrics.txt

```bash
python run_fraud_pipeline.py

---
Sample Output
PR-AUC: 0.5718
ROC-AUC: 0.5274
Precision@1%: 0.8165
Top-k threshold: 0.9925

---
