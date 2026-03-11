import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix, 
    classification_report
)
import os
import sys

def main():
    file_path = 'dataset/ai4i2020.csv'
    
    # Fallback for testing purposes if ai4i2020.csv isn't found
    if not os.path.exists(file_path):
        fallback_path = 'dataset/dataset.csv'
        if os.path.exists(fallback_path):
            print(f"Warning: '{file_path}' not found. Using '{fallback_path}' instead.\n")
            file_path = fallback_path
        else:
            print(f"Error: Dataset '{file_path}' not found.")
            sys.exit(1)

    print("1. Loading dataset...")
    df = pd.read_csv(file_path)

    print("2. Performing preprocessing...")
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Dropping them...")
        df.dropna(inplace=True)
    else:
        print("No missing values found.")

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Removing them...")
        df.drop_duplicates(inplace=True)
    else:
        print("No duplicate rows found.")

    # Handle variations in column names (e.g., from original UCI dataset)
    target_col = 'Machine_failure' if 'Machine_failure' in df.columns else 'Machine failure'
    product_id_col = 'Product_ID' if 'Product_ID' in df.columns else 'Product ID'
    
    # Remove UDI and Product_ID
    cols_to_drop = []
    if 'UDI' in df.columns:
        cols_to_drop.append('UDI')
    if product_id_col in df.columns:
        cols_to_drop.append(product_id_col)
    
    # Also remove "Type" if it exists, as it's non-numeric and wasn't requested for encoding
    if 'Type' in df.columns:
        cols_to_drop.append('Type')

    print(f"Dropping columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    print("\n3. Separating features and target variable...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("\n4. Splitting dataset into training and testing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n5. Feature scaling (StandardScaler)...")
    scaler = StandardScaler()
    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform both train and test sets
    X_test_scaled = scaler.transform(X_test)

    print("\n6. Initializing models...")
    # Model 1: RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight="balanced"
    )

    # Model 2: Support Vector Machine (SVC)
    svm_model = SVC(
        kernel="rbf", 
        class_weight="balanced",
        random_state=42
    )

    print("\n7. Evaluating Models...")
    
    # ------------------ Random Forest ------------------
    print("\n" + "="*20)
    print("Random Forest Results")
    print("="*20)
    
    # Cross Validation
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Average Cross-Validation Accuracy (CV=5): {rf_cv_scores.mean():.4f}")
    
    # Train model
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    rf_y_pred = rf_model.predict(X_test_scaled)
    
    # Print metrics
    print(f"\nAccuracy:  {accuracy_score(y_test, rf_y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, rf_y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, rf_y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, rf_y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, rf_y_pred))


    # ------------------ Support Vector Machine ------------------
    print("\n" + "="*20)
    print("SVM Results")
    print("="*20)

    # Cross Validation
    svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Average Cross-Validation Accuracy (CV=5): {svm_cv_scores.mean():.4f}")

    # Train model
    svm_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    svm_y_pred = svm_model.predict(X_test_scaled)
    
    # Print metrics
    print(f"\nAccuracy:  {accuracy_score(y_test, svm_y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, svm_y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, svm_y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, svm_y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, svm_y_pred))

if __name__ == "__main__":
    main()
