import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

# 1. Load the cleaned dataset
df = pd.read_csv("data/mushroom_clean.csv")

print("Dataset: Mushroom")
print("Dataset shape:", df.shape)


# 2. Separate features and target
X = df.drop(columns=["class"])
y = df["class"]


# 3. Convert categorical features into numeric columns
X = pd.get_dummies(X)


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 5. Create the CART-style decision tree
model = DecisionTreeClassifier(criterion="gini", random_state=42)


# 6. Train the model
model.fit(X_train, y_train)


# 7. Make predictions
predictions = model.predict(X_test)


# 8. Evaluate the model
print("\nAlgorithm: CART-style Decision Tree")
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# 9. Visualize a small part of the tree
plt.figure(figsize=(18, 8))

plot_tree(
    model,
    max_depth=3,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    fontsize=7,
)

plt.tight_layout()
plt.savefig("data/cart_tree.png")
print("\nTree image saved to data/cart_tree.png")

plt.show()
