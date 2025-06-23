from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

def train_and_evaluate_rf(df, target_column, features=None, n_estimators=10, test_size=0.2, random_state=42):
    """
    Trains and evaluates a Random Forest classifier for the given target variable.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - target_column: string, name of the target variable (e.g., 'depression', 'stress', 'anxiety')
    - features: list of feature column names (optional; if None, uses default 4 features)
    - n_estimators: number of trees in the forest
    - test_size: proportion of data to use for testing
    - random_state: random seed for reproducibility
    
    Returns:
    - model: trained RandomForestClassifier
    """
    if features is None:
        features = [ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes" ]

    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"\n--- Evaluation for {target_column.upper()} ---")
    print("Training samples shape:", X_train.shape)
    print("Test samples shape:", X_test.shape)
    print(classification_report(y_test, y_pred, zero_division=1))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Test Accuracy for {target_column}: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Training Accuracy for {target_column}: {clf.score(X_train, y_train):.4f}")

    return clf


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def lda_and_rf_classification(df, target_col, n_components=2, n_estimators=100, test_size=0.2, random_state=42):
    """
    Applies LDA for dimensionality reduction and trains a Random Forest classifier on the transformed data.

    Parameters:
    - df: DataFrame with features and labels
    - target_col: Target column name (e.g., 'depression', 'stress', 'anxiety')
    - n_components: Number of LDA components
    - n_estimators: Number of trees in Random Forest
    - test_size: Test split ratio
    - random_state: Seed for reproducibility
    """
    # Drop target columns and metadata to get features
    drop_cols = ['depression', 'anxiety', 'stress', 'Database Collectors', 'File Number user', 'Directory']
    features = df.drop(columns=drop_cols, errors='ignore')
    target = df[target_col]

    # Apply LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(features, target)

    # Add LDA components to DataFrame (optional)
    df['LDA1'] = X_lda[:, 0]
    if n_components > 1:
        df['LDA2'] = X_lda[:, 1]

    # Scatter plot of LDA
    plt.scatter(X_lda[:, 0], X_lda[:, 1] if n_components > 1 else [0]*len(X_lda), c=target, cmap='viridis')
    plt.xlabel('LDA1')
    if n_components > 1:
        plt.ylabel('LDA2')
    plt.title(f'LDA Scatter Plot for {target_col}')
    plt.colorbar(label=target_col)
    plt.show()

    # Classification using Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X_lda, target, test_size=test_size, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    train_pred = clf.predict(X_train)

    print(f"\n--- {target_col.upper()} Classification ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")

    return clf  # Return model if needed


from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def pca_rf_classifier(data, target_col, threshold, n_components=2, n_estimators=100, test_size=0.2, random_state=42):
    """
    Apply PCA and train a Random Forest classifier on a binarized target.

    Parameters:
    - data: DataFrame
    - target_col: Target column name ('depression', 'stress', 'anxiety')
    - threshold: Cut-off value to binarize the target
    - n_components: PCA components
    - n_estimators: Number of trees in RF
    - test_size: Fraction of test data
    - random_state: Reproducibility

    Returns:
    - Trained classifier and accuracy
    """

    # Prepare feature and binary target
    X = data.drop(['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'], axis=1)
    X = (X - np.mean(X)) / np.std(X)  # Standardization
    y = np.where(data[target_col] > threshold, 1, 0)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Optional: 3D Scatter plot for visual understanding (if depression)
    if target_col == 'depression':
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], data['depression'], c=data['depression'], cmap=plt.cm.jet)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('Depression')
        plt.colorbar(scatter)
        plt.title("3D PCA Scatter (Depression vs Components)")
        plt.show()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=random_state)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {target_col.upper()} Classification (Threshold: {threshold}) ---")
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # Percentage breakdown
    pct_pos = (np.sum(y_test) / len(y_test)) * 100
    pct_neg = 100 - pct_pos
    print(f"Positive Class (> {threshold}): {pct_pos:.2f}% | Negative Class: {pct_neg:.2f}%")

    return clf, acc

