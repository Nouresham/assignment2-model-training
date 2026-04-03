import os
import pandas as pd
import numpy as np

# Find the parquet file
data_path = os.environ.get('AZUREML_DATAREFERENCE_data', '/mnt/azureml/cr/j/')
print(f"Looking for data in: {data_path}")

# List all files
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.parquet'):
            file_path = os.path.join(root, file)
            print(f"Found: {file_path}")
            df = pd.read_parquet(file_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nData types:")
            print(df.dtypes)
            print(f"\nNumeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
            break
