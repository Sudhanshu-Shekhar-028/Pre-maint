import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import sys
import os

def connect_to_mongodb():
    """Connects to the local MongoDB instance and returns the database and collection."""
    print("Connecting to MongoDB...")
    try:
        # Connect to MongoDB at the specified address and port
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        
        # Check if the connection is successful
        client.admin.command('ismaster')
        print("✅ Successfully connected to MongoDB!")
        
        # Access the specified database and collection
        db = client['predictive_maintenance']
        collection = db['sensor_data']
        
        return client, collection
    
    except ConnectionFailure:
        print("❌ Error: Could not connect to local MongoDB at mongodb://localhost:27017/")
        print("Please ensure MongoDB is installed and running on your system.")
        raise
    except Exception as e:
        print(f"❌ An unexpected error occurred during connection: {e}")
        raise

def load_dataset():
    """Loads the dataset using pandas, converts it to JSON documents, and returns the records."""
    file_path = '../dataset/ai4i2020.csv'
    
    # Try alternate paths if running from the root instead of src/
    if not os.path.exists(file_path):
        fallback_path = 'dataset/ai4i2020.csv'
        if not os.path.exists(fallback_path):
            fallback_path = 'dataset/dataset.csv'
            
        if os.path.exists(fallback_path):
            file_path = fallback_path
        else:
            print(f"❌ Error: Dataset 'dataset/ai4i2020.csv' not found.")
            raise FileNotFoundError(f"Could not locate dataset at {file_path} or fallbacks.")
            
    print(f"Loading dataset from {file_path}...")
    try:
        # Load the dataset using pandas
        df = pd.read_csv(file_path)
        
        # Print dataset shape
        print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Convert the dataset into a list of JSON documents (records)
        records = df.to_dict(orient='records')
        return records
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def insert_data(collection, records):
    """Clears the existing collection and inserts the new records."""
    try:
        # Before inserting data, clear the existing collection to prevent duplicate records
        existing_count = collection.count_documents({})
        if existing_count > 0:
            print(f"Clearing {existing_count} existing records from the collection...")
            collection.delete_many({})
            
        print(f"Inserting {len(records)} records into MongoDB...")
        
        # Insert the dataset rows into the MongoDB collection
        result = collection.insert_many(records)
        inserted_count = len(result.inserted_ids)
        
        print(f"✅ Successfully inserted {inserted_count} records.")
        
        # Confirm that the collection now contains the same number of documents as the dataset rows
        final_count = collection.count_documents({})
        print(f"Verification: Collection now contains {final_count} documents.")
        
        if final_count == len(records):
            print("✅ Verification passed: Document count matches dataset row count.")
        else:
            print("❌ Verification failed: Document count does NOT match dataset row count.")
            
        print("\nThe inserted data can be viewed inside MongoDB Compass by connecting to: mongodb://localhost:27017/")
            
    except OperationFailure as e:
        print(f"❌ MongoDB Operation failed during insertion: {e}")
        raise
    except Exception as e:
        print(f"❌ An unexpected error occurred during insertion: {e}")
        raise

def main():
    client = None
    try:
        print("="*50)
        print("MongoDB Data Ingestion Pipeline")
        print("="*50 + "\n")
        
        # 1. Connect to MongoDB
        client, collection = connect_to_mongodb()
        print("-" * 50)
        
        # 2. Load the dataset
        records = load_dataset()
        print("-" * 50)
        
        # 3. Insert the data
        insert_data(collection, records)
        print("-" * 50)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)
        
    finally:
        # Close the connection if it was established
        if client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    main()
