import pandas as pd
import math

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATASET_NAME = "Mushroom Dataset"


def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    result = 0

    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)

        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy


def build_id3(data, features, target):
    labels = data[target]

    # Agar tamam examples aik hi class ke hain
    if labels.nunique() == 1:
        return labels.iloc[0]

    # Agar features khatam ho gaye
    if not features:
        return labels.mode()[0]

    # Har feature ka Information Gain calculate karo
    gains = {feature: information_gain(data, feature, target) for feature in features}

    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}
    remaining_features = [feature for feature in features if feature != best_feature]

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]

        if subset.empty:
            tree[best_feature][value] = labels.mode()[0]
        else:
            tree[best_feature][value] = build_id3(subset, remaining_features, target)

    return tree


def predict_one(tree, row, default):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = row[feature]

    if value not in tree[feature]:
        return default

    return predict_one(tree[feature][value], row, default)


# Load preprocessed dataset
df = pd.read_csv("data/mushroom_clean.csv")

print("Dataset:", DATASET_NAME)
print("Dataset shape:", df.shape)

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

features = list(X.columns)
tree = build_id3(X_train, features, "class")
default_class = X_train["class"].mode()[0]

predictions = [predict_one(tree, row, default_class) for _, row in X_test.iterrows()]

print("\nAlgorithm: ID3")
print("Split method: Information Gain")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))
