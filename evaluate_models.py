import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix
)
import os
import sys

def load_and_preprocess_data():
    file_path = 'dataset/ai4i2020.csv'
    
    if not os.path.exists(file_path):
        fallback_path = 'dataset/dataset.csv'
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            print(f"Error: Dataset not found.")
            sys.exit(1)
            
    df = pd.read_csv(file_path)
    
    # Remove variables that cause data leakage
    product_id_col = 'Product_ID' if 'Product_ID' in df.columns else 'Product ID'
    leakage_cols = ['UDI', product_id_col, 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    cols_to_drop = [col for col in leakage_cols if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
        
    target_col = 'Machine_failure' if 'Machine_failure' in df.columns else 'Machine failure'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Format confusion matrix as a string for the table
    cm_str = f"TN:{cm[0][0]} FP:{cm[0][1]} FN:{cm[1][0]} TP:{cm[1][1]}"
    
    return acc, prec, rec, cm_str

def main():
    print("Loading data and training models...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight="balanced",
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Train SVM
    svm_model = SVC(
        kernel="rbf", 
        class_weight="balanced",
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    
    # Get metrics
    rf_acc, rf_prec, rf_rec, rf_cm = get_metrics(rf_model, X_test, y_test)
    svm_acc, svm_prec, svm_rec, svm_cm = get_metrics(svm_model, X_test, y_test)
    
    # Print Comparison Table
    print("\n" + "="*85)
    print(f"{'Model Comparison Table':^85}")
    print("="*85)
    
    # Header
    print(f"{'Model':<18} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Confusion Matrix (Test)'}")
    print("-" * 85)
    
    # Random Forest Row
    print(f"{'Random Forest':<18} | {rf_acc:.4f}     | {rf_prec:.4f}     | {rf_rec:.4f}     | {rf_cm}")
    
    # SVM Row
    print(f"{'SVM (RBF)':<18} | {svm_acc:.4f}     | {svm_prec:.4f}     | {svm_rec:.4f}     | {svm_cm}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
