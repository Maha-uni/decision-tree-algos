import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

print("Original Dataset:")
print(df.head())

# 2. Handle missing values
for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())

print("\nMissing values handled.")

# 3. Separate features and target
target_column = df.columns[-1]

X = df.drop(columns=[target_column])
y = df[target_column]

# 4. Convert categorical features into numeric values
X = pd.get_dummies(X, drop_first=False)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train a C4.5-style decision tree
# sklearn uses Gini or Information Gain, not Gain Ratio.
# Therefore, this model is a baseline, not true C4.5.
model = DecisionTreeClassifier(criterion="entropy", random_state=42)

model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model
print("\nActual values:")
print(y_test.to_list())

print("\nPredicted values:")
print(y_pred.tolist())

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
