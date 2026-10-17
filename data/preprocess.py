import pandas as pd
from pathlib import Path

# Step 1: Dataset information
DATASET_NAME = "Mushroom Dataset"
file_path = Path("data/agaricus-lepiota.data")

# Step 2: Check whether the dataset file exists
if not file_path.exists():
    raise FileNotFoundError(f"Dataset file not found: {file_path}")

# Step 3: Define column names
columns = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

# Step 4: Load the dataset
df = pd.read_csv(file_path, header=None, names=columns, na_values=["?"])

print("Dataset:", DATASET_NAME)
print("Rows and columns:", df.shape)

print("\nFirst five rows:")
print(df.head())

# Step 5: Count missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nTotal missing values:", df.isnull().sum().sum())

# Step 6: Fill missing categorical values with the mode
for column in df.columns:
    if df[column].isnull().any():
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

# Step 7: Verify cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

# Step 8: Save the cleaned dataset
output_path = Path("data/mushroom_clean.csv")
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)
print("Final dataset shape:", df.shape)
