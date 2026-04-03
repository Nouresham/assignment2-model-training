import os
import pandas as pd

# Find parquet file
data_path = os.environ.get('AZUREML_DATAREFERENCE_data', '/mnt/azureml/cr/j/')
print(f"Searching in: {data_path}")

for file in os.listdir('/mnt/azureml/cr/j/'):
    if file.endswith('.parquet'):
        file_path = os.path.join('/mnt/azureml/cr/j/', file)
        print(f"Found: {file_path}")
        df = pd.read_parquet(file_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        break
