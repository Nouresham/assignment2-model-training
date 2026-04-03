import os
import pandas as pd

# Find and load parquet
data_path = '/mnt/azureml/cr/j/'
print(f"Looking in: {data_path}")

for file in os.listdir('/mnt/azureml/cr/j/'):
    if file.endswith('.parquet'):
        file_path = os.path.join('/mnt/azureml/cr/j/', file)
        print(f"\nFile: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"Shape: {df.shape}")
        print(f"\nALL COLUMNS:")
        for col in df.columns:
            print(f"  - {col} ({df[col].dtype})")
        
        # Check first few rows
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        break
