# decision-tree-algos

Implementation of ID3,CART, and C4.5 algorithms using python.

# Decision Tree Algorithms

## Project Overview

This project implements three decision tree algorithms:

- ID3
- C4.5
- CART

The project uses the Mushroom Dataset to classify mushrooms
as edible or poisonous.

## Dataset

Dataset: Mushroom Dataset

Files:

- agaricus-lepiota.data
- agaricus-lepiota.names

The dataset contains mushroom features and their classes.

## Preprocessing

The preprocessing module handles dataset loading,
column names, missing values, and feature preparation
as required by the implementations.

## Algorithms

- ID3: Uses Information Gain.
- C4.5: Uses Gain Ratio.
- CART: Uses Gini Impurity and binary splits.

Note: These descriptions should match the actual
implementation in each Python file.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn (if used in the code)

## Project Structure

decision-tree-algos/
├── data/
├── ID3.py
├── C45.py
├── CART.py
├── preprocessor.py
├── README.md
└── requirements.txt

## How to Run

1. Install Python.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the required algorithm:
   python ID3.py
   python C45.py
   python CART.py

## Evaluation

The algorithms should be evaluated using the same
train-test split and suitable classification metrics.

## Author

Maha Zubair
