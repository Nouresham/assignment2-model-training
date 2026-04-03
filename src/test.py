import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

print(f"Data path: {args.data}")
print(f"Path exists: {os.path.exists(args.data)}")

if os.path.exists(args.data):
    print("Contents:")
    for item in os.listdir(args.data):
        print(f"  - {item}")
else:
    print("Path does not exist!")
