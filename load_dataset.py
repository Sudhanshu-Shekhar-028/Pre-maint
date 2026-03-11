import pandas as pd

def load_and_inspect():
    file_path = 'dataset/dataset.csv'
    try:
        # Load CSV dataset using pandas
        df = pd.read_csv(file_path)
        
        # Print dataset shape
        print(f"Dataset shape: {df.shape}\n")
        
        # Print first 5 rows
        print("First 5 rows:")
        print(df.head())
        print()
        
        # Print column names
        print("Column names:")
        print(df.columns.tolist())
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the 'dataset' directory exists and contains 'dataset.csv'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    load_and_inspect()
