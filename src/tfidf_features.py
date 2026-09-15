import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"

OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD TASK A DATASET
# --------------------------------------------------

dataset_path = PROCESSED_FOLDER / "binary_train_processed.csv"

data = pd.read_csv(dataset_path)

print("=" * 60)
print("TF-IDF FEATURE CREATION")
print("=" * 60)

print("\nDataset shape:", data.shape)

print("\nColumns:")
print(data.columns)


# --------------------------------------------------
# SELECT TEXT AND LABELS
# --------------------------------------------------

X_text = data["Cleaned_Comment"].fillna("")
y = data["Label"]


# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

X_train_text, X_validation_text, y_train, y_validation = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining comments:", len(X_train_text))
print("Validation comments:", len(X_validation_text))


# --------------------------------------------------
# CREATE TF-IDF VECTORIZER
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True
)


# --------------------------------------------------
# FIT ONLY ON TRAINING DATA
# --------------------------------------------------

X_train_tfidf = vectorizer.fit_transform(X_train_text)

X_validation_tfidf = vectorizer.transform(X_validation_text)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nTF-IDF feature creation completed!")

print("\nTraining feature matrix shape:")
print(X_train_tfidf.shape)

print("\nValidation feature matrix shape:")
print(X_validation_tfidf.shape)

print("\nNumber of vocabulary words and phrases:")
print(len(vectorizer.vocabulary_))


# --------------------------------------------------
# SHOW SAMPLE FEATURES
# --------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

print("\nFirst 30 TF-IDF features:")
print(feature_names[:30])


# --------------------------------------------------
# SAVE SPLIT INFORMATION
# --------------------------------------------------

train_data = pd.DataFrame({
    "Comment": X_train_text,
    "Label": y_train
})

validation_data = pd.DataFrame({
    "Comment": X_validation_text,
    "Label": y_validation
})

train_data.to_csv(
    OUTPUT_FOLDER / "task_a_train_split.csv",
    index=False,
    encoding="utf-8"
)

validation_data.to_csv(
    OUTPUT_FOLDER / "task_a_validation_split.csv",
    index=False,
    encoding="utf-8"
)


print("\nTrain and validation splits saved.")
print("\nProgram completed successfully!")