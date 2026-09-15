import pandas as pd
from pathlib import Path


# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"


# File paths
binary_train_path = DATA_FOLDER / "binary_train.csv"
multiclass_train_path = DATA_FOLDER / "multiclass_train.csv"


print("=" * 60)
print("HASTIKA DATASET ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# TASK A: BINARY DATASET
# --------------------------------------------------

print("\nTASK A: BINARY CLASSIFICATION")
print("-" * 60)

binary_data = pd.read_csv(binary_train_path)

print("Dataset shape:", binary_data.shape)
print("Columns:", list(binary_data.columns))

print("\nFirst 5 rows:")
print(binary_data.head())

print("\nMissing values:")
print(binary_data.isnull().sum())

print("\nLabel counts:")
print(binary_data["Label"].value_counts())

print("\nLabel percentages:")
print(binary_data["Label"].value_counts(normalize=True) * 100)

print("\nDuplicate comments:")
print(binary_data["Comment"].duplicated().sum())


# --------------------------------------------------
# TASK B: MULTICLASS DATASET
# --------------------------------------------------

print("\n\nTASK B: MULTICLASS CLASSIFICATION")
print("-" * 60)

multiclass_data = pd.read_csv(multiclass_train_path)

print("Dataset shape:", multiclass_data.shape)
print("Columns:", list(multiclass_data.columns))

print("\nFirst 5 rows:")
print(multiclass_data.head())

print("\nMissing values:")
print(multiclass_data.isnull().sum())

print("\nHate category counts:")
print(multiclass_data["Hate Category"].value_counts())

print("\nHate category percentages:")
print(
    multiclass_data["Hate Category"].value_counts(normalize=True) * 100
)

print("\nDuplicate comments:")
print(multiclass_data["Comment"].duplicated().sum())


# --------------------------------------------------
# EXAMPLE COMMENTS
# --------------------------------------------------

print("\n\nEXAMPLE COMMENTS")
print("-" * 60)

print("\nBinary dataset examples:")

for index, row in binary_data.head(5).iterrows():
    print("\nID:", row["id"])
    print("Comment:", row["Comment"])
    print("Label:", row["Label"])


print("\nMulticlass dataset examples:")

for index, row in multiclass_data.head(5).iterrows():
    print("\nID:", row["id"])
    print("Comment:", row["Comment"])
    print("Category:", row["Hate Category"])


print("\n\nAnalysis completed successfully.")