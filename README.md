# Pre-maint: Predictive Maintenance using NoSQL & Machine Learning

## Overview

This project is an **independent implementation of a predictive maintenance system** that analyzes industrial machine sensor data and predicts the probability of machine failure using machine learning models.

The system processes machine parameters such as temperature, torque, rotational speed, and tool wear to determine whether a machine is likely to fail.

It is designed as a **practical, end-to-end data-driven application** combining machine learning, data preprocessing, and NoSQL-based storage.

---

## Key Features

- Upload custom CSV datasets for analysis
- Store and manage data using MongoDB (NoSQL)
- Data preprocessing and cleaning pipeline
- Machine learning models for failure prediction
- Visualizations for insights and feature relationships
- Web-based interface for interaction

---

## Machine Learning Models Used

### Random Forest Classifier
- Ensemble-based model using multiple decision trees
- Handles non-linear relationships well
- Reduces overfitting
- Provides feature importance insights

### Support Vector Machine (SVM)
- Effective for binary classification tasks
- Works well with structured and high-dimensional data
- Suitable for predicting machine failure (0 or 1)

---

## Data Handling

### Dataset Support
- Accepts CSV datasets with machine sensor parameters:
  - Air Temperature
  - Process Temperature
  - Rotational Speed
  - Torque
  - Tool Wear
  - Failure Label

### Storage
- Data is stored in **MongoDB**
- Each dataset row is stored as a document
- Enables flexible and scalable data handling

---

## Data Preprocessing

- Handling missing values
- Removing irrelevant features
- Formatting data for ML models
- Feature-label separation

---

## Project Workflow

Dataset Upload → MongoDB Storage → Data Cleaning → Model Training → Prediction → Evaluation

---

## Technologies Used

- Python
- Flask
- MongoDB
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- HTML, CSS

---

## Setup & Installation

```bash
git clone https://github.com/Sudhanshu-Shekhar-028/Pre-maint.git
cd Pre-maint
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
