import argparse
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEBUG - LISTING DATA STRUCTURE")
    print("=" * 60)
    
    # Find parquet file
    print(f"\nData path: {args.data}")
    files = os.listdir(args.data)
    print(f"Files: {files}")
    
    parquet_file = None
    for f in files:
        if f.endswith('.parquet'):
            parquet_file = os.path.join(args.data, f)
            break
    
    if parquet_file is None:
        print("ERROR: No parquet file found!")
        return
    
    print(f"\nLoading: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    print(f"\nShape: {df.shape}")
    print(f"\nALL COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col} - type: {df[col].dtype}")
    
    print(f"\nFirst 2 rows:")
    print(df.head(2))
    
    print(f"\nData types summary:")
    print(df.dtypes.value_counts())
    
    # Check for sentiment columns
    print(f"\nLooking for sentiment columns:")
    sent_cols = [c for c in df.columns if 'sentiment' in c.lower()]
    print(f"  Found: {sent_cols}")
    
    print(f"\nLooking for length columns:")
    len_cols = [c for c in df.columns if 'length' in c.lower()]
    print(f"  Found: {len_cols}")
    
    print(f"\nLooking for label column (overall):")
    if 'overall' in df.columns:
        print(f"  Found: overall")
        print(f"  Values: {df['overall'].value_counts().to_dict()}")
    else:
        print("  NOT FOUND!")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
