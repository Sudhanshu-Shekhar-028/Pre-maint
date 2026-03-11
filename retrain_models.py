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

def load_data(file_path='dataset/ai4i2020.csv'):
    print("1. Loading dataset...")
    if not os.path.exists(file_path):
        fallback_path = 'dataset/dataset.csv'
        if os.path.exists(fallback_path):
            print(f"Warning: '{file_path}' not found. Using '{fallback_path}' instead.\n")
            file_path = fallback_path
        else:
            print(f"Error: Dataset '{file_path}' not found.")
            sys.exit(1)
            
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    print("2. Performing preprocessing...")
    
    # Remove variables that cause data leakage
    cols_to_drop = []
    
    # Handle variations in column names
    product_id_col = 'Product_ID' if 'Product_ID' in df.columns else 'Product ID'
    
    # Drop "Type" as well since it is a categorical string and shouldn't be scaled as a float
    leakage_cols = ['UDI', product_id_col, 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    for col in leakage_cols:
        if col in df.columns:
            cols_to_drop.append(col)
            
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Removing them...")
        df.drop_duplicates(inplace=True)

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Dropping them...")
        df.dropna(inplace=True)
        
    # Separate features and target
    print("\n3. Separating features and target variable...")
    target_col = 'Machine_failure' if 'Machine_failure' in df.columns else 'Machine failure'
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print("\n4. Class distribution of Machine_failure:")
    print(y.value_counts())
    
    # Split dataset
    print("\n5. Splitting dataset into training and testing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling
    print("\n6. Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_models(X_train, y_train):
    print("\n7. Training models...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight="balanced",
        n_jobs=-1
    )
    
    svm_model = SVC(
        kernel="rbf", 
        class_weight="balanced",
        random_state=42
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    print("Training SVM...")
    svm_model.fit(X_train, y_train)
    
    return rf_model, svm_model

def evaluate_models(rf_model, svm_model, X_train, y_train, X_test, y_test):
    print("\n8. Evaluating models...\n")
    
    models = {
        "Random Forest": rf_model,
        "SVM": svm_model
    }
    
    for name, model in models.items():
        print("="*24)
        print(f"{name} Performance")
        print("="*24)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy: {test_acc:.4f}")
        
        # Overfitting Check
        if (train_acc - test_acc) > 0.05:
            print("\nWarning: Potential overfitting detected.")
            
        print("\nTraining Confusion Matrix:")
        print(confusion_matrix(y_train, y_train_pred))
        
        print("\nTesting Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        
        print("\nClassification Report (Test Data)")
        print(classification_report(y_test, y_test_pred))
        
        # Cross-validation
        # Using the scaled X_train and y_train for CV to prevent leakage
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"\nAverage Cross-Validation Accuracy (CV=5): {cv_scores.mean():.4f}")
        
        print("-" * 25 + "\n")

def main():
    df = load_data('dataset/ai4i2020.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    rf_model, svm_model = train_models(X_train, y_train)
    evaluate_models(rf_model, svm_model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
