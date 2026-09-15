import pandas as pd
import re
from pathlib import Path


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"

# Create processed folder if it does not exist
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(text):
    """
    Cleans Kannada-English code-mixed text.

    We preserve:
    - Kannada characters
    - English characters
    - Numbers
    - Basic punctuation
    """

    # Convert missing values to empty text
    if pd.isna(text):
        return ""

    # Convert value to string
    text = str(text)

    # Remove links
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Replace multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at the beginning and end
    text = text.strip()

    return text


# --------------------------------------------------
# TASK A PREPROCESSING
# --------------------------------------------------

print("=" * 60)
print("PREPROCESSING HASTIKA DATASET")
print("=" * 60)

binary_train_path = RAW_FOLDER / "binary_train.csv"

binary_data = pd.read_csv(binary_train_path)

print("\nTask A original shape:", binary_data.shape)

# Clean comments
binary_data["Cleaned_Comment"] = binary_data["Comment"].apply(clean_text)

# Remove rows where comment is empty
binary_data = binary_data[binary_data["Cleaned_Comment"].str.strip() != ""]

# Save processed Task A dataset
binary_output_path = PROCESSED_FOLDER / "binary_train_processed.csv"

binary_data.to_csv(binary_output_path, index=False, encoding="utf-8")

print("Task A processed shape:", binary_data.shape)
print("Task A saved to:", binary_output_path)


# --------------------------------------------------
# TASK B PREPROCESSING
# --------------------------------------------------

multiclass_train_path = RAW_FOLDER / "multiclass_train.csv"

multiclass_data = pd.read_csv(multiclass_train_path)

print("\nTask B original shape:", multiclass_data.shape)

# Clean comments
multiclass_data["Cleaned_Comment"] = multiclass_data["Comment"].apply(
    clean_text
)

# Remove rows where comment is empty
multiclass_data = multiclass_data[
    multiclass_data["Cleaned_Comment"].str.strip() != ""
]

# Save processed Task B dataset
multiclass_output_path = (
    PROCESSED_FOLDER / "multiclass_train_processed.csv"
)

multiclass_data.to_csv(
    multiclass_output_path,
    index=False,
    encoding="utf-8"
)

print("Task B processed shape:", multiclass_data.shape)
print("Task B saved to:", multiclass_output_path)


# --------------------------------------------------
# SHOW EXAMPLES
# --------------------------------------------------

print("\n" + "=" * 60)
print("BEFORE AND AFTER EXAMPLES")
print("=" * 60)

for index, row in binary_data.head(5).iterrows():
    print("\nOriginal:")
    print(row["Comment"])

    print("Cleaned:")
    print(row["Cleaned_Comment"])


print("\nPreprocessing completed successfully!")