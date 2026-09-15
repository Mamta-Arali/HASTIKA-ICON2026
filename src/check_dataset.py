import pandas as pd
from pathlib import Path


# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"


# Dataset file paths
files = [
    "binary_train.csv",
    "binary_validation_inputs.csv",
    "multiclass_train.csv",
    "multiclass_validation_inputs.csv"
]


print("HASTIKA dataset checking started")
print("-" * 50)


for file_name in files:
    file_path = DATA_FOLDER / file_name

    print("\nChecking:", file_name)

    if file_path.exists():
        data = pd.read_csv(file_path)

        print("File found successfully")
        print("Number of rows:", len(data))
        print("Column names:", list(data.columns))
        print("First 3 rows:")
        print(data.head(3))

    else:
        print("File not found:", file_path)


print("\nDataset checking completed")