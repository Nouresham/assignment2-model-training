import os
import pandas as pd

def main():
    data_path = '/mnt/azureml/cr/j/'
    print(f"Searching for parquet files in: {data_path}")
    
    for f in os.listdir(data_path):
        if f.endswith('.parquet'):
            file_path = os.path.join(data_path, f)
            print(f"\nFound: {file_path}")
            df = pd.read_parquet(file_path)
            print(f"Shape: {df.shape}")
            print(f"\nALL COLUMN NAMES:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col}")
            break
    else:
        print("No parquet files found in the mounted path")

if __name__ == "__main__":
    main()
