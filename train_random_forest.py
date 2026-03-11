import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    # File paths
    input_file = 'dataset/cleaned_dataset.csv'
    
    try:
        # 1. Load cleaned dataset
        print(f"Loading cleaned dataset from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Handle column naming variations
        target_col = 'Machine_failure' if 'Machine_failure' in df.columns else 'Machine failure'
        
        # We need to exclude non-numeric features if they weren't removed during preprocessing
        # UDI, Product ID, Type are likely not useful predictors without encoding
        cols_to_drop = [target_col]
        for col in ['UDI', 'Product ID', 'Type']:
            if col in df.columns:
                cols_to_drop.append(col)
                
        X = df.drop(columns=cols_to_drop)
        y = df[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # 2. Split dataset using train_test_split
        print("\nSplitting dataset into train and test sets (80/20)...")
        # stratify=y ensures the same proportion of machine failures in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        # 3. Train RandomForestClassifier
        print("\nTraining RandomForestClassifier...")
        # class_weight='balanced' helps deal with the imbalanced dataset
        # since machine failures are rare
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1 # Use all available cores
        )
        rf_model.fit(X_train, y_train)
        
        # 4. Predict machine failure
        print("Making predictions on the test set...")
        y_pred = rf_model.predict(X_test)
        
        # 5. Print accuracy and detailed metrics
        # (Requirements specified printing accuracy, but we'll print a bit more for context)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Failure (0)', 'Failure (1)']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0][0]}\tFalse Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}\tTrue Positives: {cm[1][1]}")
        
        # Also show top 5 feature importances
        feature_importances = pd.DataFrame(
            rf_model.feature_importances_,
            index=X.columns,
            columns=['Importance']
        ).sort_values('Importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importances.head())
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Please ensure the cleaned dataset exists. Run preprocessing.py first if needed.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    train_model()
