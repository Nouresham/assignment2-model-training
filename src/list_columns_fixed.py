import os
import pandas as pd

def main():
    # The data is passed as an argument, not hardcoded
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Data path: {args.data}")
    
    # List all files in the data path
    print(f"\nFiles in data path:")
    for f in os.listdir(args.data):
        print(f"  - {f}")
        if f.endswith('.parquet'):
            file_path = os.path.join(args.data, f)
            print(f"\nLoading: {file_path}")
            df = pd.read_parquet(file_path)
            print(f"Shape: {df.shape}")
            print(f"\nALL COLUMN NAMES ({len(df.columns)} columns):")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col}")
            
            # Show first few values of important columns
            print(f"\nSample data (first row):")
            for col in df.columns[:10]:
                print(f"  {col}: {df[col].iloc[0]}")
            break

if __name__ == "__main__":
    main()
