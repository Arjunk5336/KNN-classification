# Iris KNN Classifier with PCA Visualization

This project demonstrates the implementation of a K-Nearest Neighbors (KNN) classifier on the classic Iris dataset. The script evaluates model accuracy for different k values, selects the best performing one, and visualizes the results using a confusion matrix and a PCA-reduced decision boundary plot.


Workflow:

Loads the Iris dataset using Pandas.

Preprocesses features (standard scaling) and encodes target labels.

Splits the dataset into training and test sets.

Trains KNN models with k from 1 to 20.

Plots accuracy vs. k to find the optimal value.

Displays a confusion matrix for the best-performing model.

Reduces feature dimensions using PCA (2D) for visualization.

Plots decision boundaries using the PCA-transformed data.


Visual Output:

Accuracy Plot: Shows KNN accuracy for different values of k.

Confusion Matrix: Displays prediction accuracy by class.

PCA Decision Boundary Plot: Shows how the classifier separates classes in 2D space.


Dataset:

The script expects the Iris dataset in CSV format, typically containing:

Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

Target: Species
