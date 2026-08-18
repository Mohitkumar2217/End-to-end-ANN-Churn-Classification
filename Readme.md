# End-to-End Customer Churn Prediction using Artificial Neural Networks (ANN)

An end-to-end Machine Learning and Deep Learning web application designed to predict the likelihood of bank customer churn. Built with **TensorFlow/Keras**, **Scikit-learn**, and deployed interactively using **Streamlit**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Running the Web Application](#-running-the-web-application)
- [TensorBoard Visualization](#-tensorboard-visualization)
- [License](#-license)

---

## 📖 Overview
Customer churn is one of the most critical metrics for growing businesses and financial institutions. This project implements an Artificial Neural Network (ANN) trained on bank customer demographics, account information, and behavioral data to accurately classify whether a customer is at risk of exiting the bank.

---

## 🏗 Project Architecture

```
                      +-----------------------------+
                      | Raw Customer Input / Data   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Preprocessing & Encoders    |
                      | - LabelEncoder (Gender)     |
                      | - OneHotEncoder (Geography) |
                      | - StandardScaler (Features) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Deep Neural Network (ANN)   |
                      | - Dense Layers (ReLU)       |
                      | - Dropout / Batch Norm      |
                      | - Output Layer (Sigmoid)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Streamlit Web App Interface |
                      | - Probability Output        |
                      | - Churn Classification      |
                      +-----------------------------+
```

---

## ✨ Features
- **Data Preprocessing Pipeline:** Automated handling of categorical variables (`OneHotEncoder`, `LabelEncoder`) and numeric standard scaling (`StandardScaler`).
- **Deep Learning Classifier:** Custom multi-layer ANN built using TensorFlow/Keras with early stopping and learning rate callbacks.
- **Interactive UI:** Clean, intuitive Streamlit web interface with real-time churn probability scoring.
- **Model Monitoring:** Experiment tracking and metric logging via TensorBoard.

---

## 🛠 Tech Stack
- **Language:** Python 3.10 / 3.11 / 3.12
- **Deep Learning:** TensorFlow, Keras
- **Machine Learning & Preprocessing:** Scikit-learn, Pandas, NumPy
- **Web Framework:** Streamlit
- **Monitoring & Logging:** TensorBoard
- **Serialization:** Pickle

---

## 📁 Directory Structure
```text
├── logs/                         # TensorBoard training logs
├── experiments/
│   └── churn_classification.ipynb # Exploratory data analysis & model experiments
├── app.py                        # Streamlit web application
├── model.h5                      # Trained TensorFlow/Keras ANN model
├── label_encoder_gender.pkl      # Pickled LabelEncoder for Gender
├── onehot_encoder_geo.pkl        # Pickled OneHotEncoder for Geography
├── scaler.pkl                    # Pickled StandardScaler for feature normalization
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10, 3.11, or 3.12 installed on your machine (or via WSL Ubuntu).

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/end-to-end-ann-churn-prediction.git
cd end-to-end-ann-churn-prediction
```

### 3. Set Up a Virtual Environment
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate on Linux/WSL/macOS:
source venv/bin/activate

# Activate on Windows PowerShell:
# .\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📋 Requirements (`requirements.txt`)
```text
tensorflow>=2.15.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.30.0
tensorboard>=2.15.0
```

---

## 🏋️ Model Training & Evaluation

1. Open the training notebook or run your training script:
   ```bash
   jupyter notebook experiments/churn_classification.ipynb
   ```
2. The pipeline:
   - Loads the Churn Modeling dataset.
   - Encodes categorical columns (`Geography`, `Gender`).
   - Normalizes feature scales via `StandardScaler`.
   - Compiles and fits the ANN using `BinaryCrossentropy` and `Adam` optimizer.
   - Saves model weights (`model.h5`) and preprocessing transformers (`.pkl`).

---

## 🖥 Running the Web Application

Launch the Streamlit web application with:

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser. Enter customer details to obtain real-time churn predictions and probability estimates.

---

## 📊 TensorBoard Visualization

To inspect loss curves, accuracy metrics, and weight distributions during training:

```bash
tensorboard --logdir=logs/fit
```
Then open `http://localhost:6006` in your browser.

--- 