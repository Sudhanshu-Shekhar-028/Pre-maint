import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_svm_model():
    input_file = 'dataset/cleaned_dataset.csv'
    
    try:
        # Load cleaned dataset
        print(f"Loading cleaned dataset from {input_file}...")
        df = pd.read_csv(input_file)
        
        target_col = 'Machine_failure' if 'Machine_failure' in df.columns else 'Machine failure'
        
        cols_to_drop = [target_col]
        for col in ['UDI', 'Product ID', 'Type']:
            if col in df.columns:
                cols_to_drop.append(col)
                
        X = df.drop(columns=cols_to_drop)
        y = df[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split dataset using train_test_split
        print("\nSplitting dataset into train and test sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Support Vector Machine (SVC)
        print("\nTraining Support Vector Machine (SVC)...")
        # Using class_weight='balanced' for imbalanced dataset
        svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
        svm_model.fit(X_train, y_train)
        
        # Predict machine failure
        print("Making predictions on the test set...")
        y_pred = svm_model.predict(X_test)
        
        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Failure (0)', 'Failure (1)']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0][0]}\tFalse Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}\tTrue Positives: {cm[1][1]}")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Please ensure the cleaned dataset exists. Run preprocessing.py first if needed.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    train_svm_model()
