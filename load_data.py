from load_pinecone import load_csv_to_pinecone
import os
import pandas as pd

def main():
    csv_path = 'youtube_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    try:
        print(f"Found CSV file with size: {os.path.getsize(csv_path)} bytes")
        
        df = pd.read_csv(csv_path)
        print(f"CSV contains {len(df)} rows")
        print("\nSample of data:")
        print(df.head(2))
        
        print("\nBeginning upload to Pinecone...")
        load_csv_to_pinecone(csv_path)
        print("Data loading complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 