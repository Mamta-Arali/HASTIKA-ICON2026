import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# PROJECT PATHS
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
# LOAD DATASET
# --------------------------------------------------

print("=" * 60)
print("TASK A: HATE VS NON-HATE CLASSIFICATION")
print("=" * 60)

data = pd.read_csv(DATASET_PATH)

print("\nDataset shape:", data.shape)


# --------------------------------------------------
# SELECT INPUT AND OUTPUT
# --------------------------------------------------

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

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_validation))


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)

print("Training feature shape:", X_train_tfidf.shape)
print("Validation feature shape:", X_validation_tfidf.shape)


# --------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_validation_tfidf)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_validation, predictions)

precision = precision_score(
    y_validation,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_validation,
    predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_validation,
    predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Macro-F1:  {macro_f1:.4f}")


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification report:")

print(
    classification_report(
        y_validation,
        predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

print("\nCreating confusion matrix...")

matrix = confusion_matrix(
    y_validation,
    predictions,
    labels=["Hate", "Non-Hate"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Hate", "Non-Hate"]
)

display.plot()

plt.title("Task A Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = OUTPUT_FOLDER / "confusion_matrix.png"

plt.savefig(confusion_matrix_path)

print("Confusion matrix saved to:")
print(confusion_matrix_path)

plt.show()


# --------------------------------------------------
# SAVE VALIDATION PREDICTIONS
# --------------------------------------------------

prediction_data = pd.DataFrame({
    "Actual_Label": y_validation.values,
    "Predicted_Label": predictions
})

prediction_path = OUTPUT_FOLDER / "validation_predictions.csv"

prediction_data.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8"
)

print("\nValidation predictions saved to:")
print(prediction_path)


# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

metrics_data = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        macro_f1
    ]
})

metrics_path = OUTPUT_FOLDER / "baseline_metrics.csv"

metrics_data.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8"
)

print("\nMetrics saved to:")
print(metrics_path)

print("\nTask A baseline training completed successfully!")