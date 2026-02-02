# 📊 Early Loan Prepayment Prediction using Hybrid Machine Learning Model
Incorporating Dynamic Borrower Behavior into Machine Learning Models for Forecasting Early Prepayment of  Loans and Customer Retention in Retail Lending. 

## 📌 Project Overview
Early loan prepayment presents a major challenge for financial institutions as it disrupts expected cash flows, interest income, and asset–liability management. Traditional prepayment models typically rely on static borrower and loan characteristics captured at origination, which limits their ability to adapt to evolving borrower behavior over time.

This project investigates early loan prepayment prediction by comparing traditional machine learning models, deep learning sequence-based models, and a novel hybrid neural network architecture that combines static loan attributes with temporal behavioral patterns.

The project was completed as part of the **Applied Research module** for the **MSc in Artificial Intelligence** and emphasizes building models that are both **academically rigorous and practically deployable**.

---

## 🎯 Research Objectives
- Predict early loan prepayment using supervised learning techniques  
- Compare static, sequential, and hybrid modeling approaches  
- Analyze performance using business-relevant metrics  
- Demonstrate the benefit of integrating borrower attributes with behavioral dynamics  

---

## 🧠 Modeling Approaches

### 1️⃣ Tabular (Static) Models
- **XGBoost**
- **Artificial Neural Network (ANN)**  

These models use borrower and loan characteristics such as credit score, loan-to-value ratio, debt-to-income ratio, interest rate, and loan term.

---

### 2️⃣ Sequence-Based Models
- **Long Short-Term Memory (LSTM)**
- **1D Convolutional Neural Network (CNN)**  

Sequence models are trained on synthetically generated fixed-length behavioral sequences designed to approximate repayment dynamics while avoiding label leakage.

---

### 3️⃣ Hybrid Model (Proposed Contribution)
The hybrid architecture consists of:
- A dense neural network for static tabular features  
- An LSTM encoder for sequential behavioral data  
- A feature fusion layer that jointly learns from both inputs  

This design allows the model to capture both **long-term borrower risk** and **short-term repayment behavior**.

---

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|----------|-----------|--------|----------|---------|
| XGBoost | ~0.67 | ~0.68 | ~0.66 | ~0.67 | ~0.73 |
| ANN | ~0.65 | ~0.65 | ~0.68 | ~0.66 | ~0.71 |
| LSTM | ~0.71 | **1.00** | ~0.43 | ~0.61 | ~0.99 |
| CNN | ~0.62 | **~1.00** | ~0.25 | ~0.40 | ~0.97 |
| **Hybrid Model** | **~0.80** | **~0.998** | ~0.61 | **~0.75** | **~0.96** |

### 🔹 Key Observations
- The **Hybrid Model outperforms all baseline models**
- Achieves **near-perfect precision**, making it suitable for conservative lending strategies
- Demonstrates strong ranking capability through ROC-AUC performance

---

## ⚙️ Methodology Highlights
- Proxy definition of early prepayment based on repayment duration  
- Feature selection designed to avoid label leakage  
- Synthetic behavioral sequence generation under real-world data constraints  
- Z-score normalization and one-hot encoding  
- Stratified 80/20 train–test split  
- Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC  

---

## 📂 Repository Structure
