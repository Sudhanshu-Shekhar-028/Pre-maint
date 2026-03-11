import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data():
    file_path = 'dataset/dataset.csv'
    output_path = 'dataset/cleaned_dataset.csv'
    
    # 1. Load dataset
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Initial shape: {df.shape}")
    
    # Check if target column is named 'Machine failure' (with space) or 'Machine_failure'
    target_col = 'Machine failure'
    if target_col not in df.columns and 'Machine_failure' in df.columns:
        target_col = 'Machine_failure'
        
    # Rename column to machine_failure for consistency if it has space
    if target_col == 'Machine failure':
        df.rename(columns={'Machine failure': 'Machine_failure'}, inplace=True)
        target_col = 'Machine_failure'
    
    # 2. Handle missing values
    # For simplicity, we drop rows with missing values. 
    # Alternatively, you could use imputation strategies.
    print("Handling missing values...")
    missing_before = df.isnull().sum().sum()
    df.dropna(inplace=True)
    print(f"Dropped {missing_before} missing values. Shape: {df.shape}")
    
    # 3. Remove duplicate rows
    print("Removing duplicates...")
    duplicates_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Dropped {duplicates_before} duplicate rows. Shape: {df.shape}")
    
    # 4. Separate features and target column before normalization
    # to avoid normalizing the target or non-numerical IDs
    print("Separating features and target...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 5. Normalize numerical features
    print("Normalizing numerical features...")
    # Identify numerical columns in X
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # We might want to exclude ID columns or binary flags from normalization
    # Looking at typical predictive maintenance datasets:
    cols_to_exclude = ['UDI', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    numeric_features_to_scale = [col for col in numeric_cols if col not in cols_to_exclude]
    
    scaler = StandardScaler()
    X[numeric_features_to_scale] = scaler.fit_transform(X[numeric_features_to_scale])
    
    print(f"Normalized columns: {numeric_features_to_scale}")
    
    # Recombine to save the full cleaned dataset
    cleaned_df = pd.concat([X, y], axis=1)
    
    # 6. Save cleaned dataset
    print(f"Saving cleaned dataset to {output_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    
    print("Preprocessing completed successfully!")
    print(f"Final shape: {cleaned_df.shape}")

if __name__ == "__main__":
    preprocess_data()
