import pandas as pd
import os

# Find the raw data
for f in os.listdir('/home/azureuser/lab3-azure-databricks./data/'):
    if f.endswith('.parquet'):
        print(f"Checking: {f}")
        df = pd.read_parquet(f'/home/azureuser/lab3-azure-databricks./data/{f}')
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:20]}")
        print(f"Has 'overall': {'overall' in df.columns}")
        break
