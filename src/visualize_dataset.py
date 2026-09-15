import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"

# Output folder
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

# Create output folder if it does not exist
OUTPUT_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

binary_path = DATA_FOLDER / "binary_train.csv"
multiclass_path = DATA_FOLDER / "multiclass_train.csv"

binary_data = pd.read_csv(binary_path)
multiclass_data = pd.read_csv(multiclass_path)


# --------------------------------------------------
# TASK A GRAPH
# --------------------------------------------------

print("Creating Task A graph...")

binary_counts = binary_data["Label"].value_counts()

plt.figure(figsize=(8, 5))

binary_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Task A: Hate vs Non-Hate")
plt.xlabel("Label")
plt.ylabel("Number of Comments")

plt.xticks(rotation=0)
plt.tight_layout()

task_a_graph_path = OUTPUT_FOLDER / "task_a_label_distribution.png"

plt.savefig(task_a_graph_path)
plt.show()

print("Task A graph saved at:")
print(task_a_graph_path)


# --------------------------------------------------
# TASK B GRAPH
# --------------------------------------------------

print("\nCreating Task B graph...")

multiclass_counts = multiclass_data["Hate Category"].value_counts()

plt.figure(figsize=(10, 6))

multiclass_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Task B: Hate Category Distribution")
plt.xlabel("Hate Category")
plt.ylabel("Number of Comments")

plt.xticks(rotation=45)
plt.tight_layout()

task_b_graph_path = OUTPUT_FOLDER / "task_b_category_distribution.png"

plt.savefig(task_b_graph_path)
plt.show()

print("Task B graph saved at:")
print(task_b_graph_path)


print("\nBoth graphs created successfully!")