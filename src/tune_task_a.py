import pandas as pd

from pathlib import Path
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "binary_train_processed.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "task_a"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("TASK A MODEL TUNING")
print("=" * 60)

data = pd.read_csv(DATASET_PATH)

X = data["Cleaned_Comment"].fillna("")
y = data["Label"]


# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# WORD TF-IDF
# --------------------------------------------------

print("\nCreating word-level TF-IDF...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    lowercase=True,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.98,
    max_features=50000,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_validation_word = word_vectorizer.transform(X_validation)

print("Word features:", X_train_word.shape)


# --------------------------------------------------
# CHARACTER TF-IDF
# --------------------------------------------------

print("\nCreating character-level TF-IDF...")

character_vectorizer = TfidfVectorizer(
    analyzer="char",
    lowercase=True,
    ngram_range=(2, 6),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train_character = character_vectorizer.fit_transform(X_train)
X_validation_character = character_vectorizer.transform(
    X_validation
)

print("Character features:", X_train_character.shape)


# --------------------------------------------------
# COMBINE FEATURES
# --------------------------------------------------

X_train_combined = hstack([
    X_train_word,
    X_train_character
])

X_validation_combined = hstack([
    X_validation_word,
    X_validation_character
])

print("Combined features:", X_train_combined.shape)


# --------------------------------------------------
# TEST DIFFERENT C VALUES
# --------------------------------------------------

c_values = [0.5, 1.0, 1.5, 2.0, 3.0]

results = []

best_model = None
best_score = -1
best_c = None


for c_value in c_values:

    print("\n" + "-" * 60)
    print("Training model with C =", c_value)

    model = LogisticRegression(
        C=c_value,
        max_iter=1500,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    model.fit(X_train_combined, y_train)

    predictions = model.predict(X_validation_combined)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))

    results.append({
        "C": c_value,
        "Accuracy": accuracy,
        "Macro-F1": macro_f1
    })

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = model
        best_c = c_value


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results_data = pd.DataFrame(results)

results_path = OUTPUT_FOLDER / "tuning_results.csv"

results_data.to_csv(
    results_path,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("\n" + "=" * 60)
print("TUNING COMPLETED")
print("=" * 60)

print("Best C value:", best_c)
print("Best Macro-F1:", round(best_score, 4))

print("\nAll results:")
print(results_data)

print("\nResults saved to:")
print(results_path)