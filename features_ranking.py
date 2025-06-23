import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def analyze_emotion_with_rf(df, emotion, threshold, n_estimators=100):
    """
    Trains a Random Forest classifier on the dataset and prints:
    - Percentage of affected and unaffected subjects based on the threshold
    - Top feature importances for the given emotion

    Parameters:
    - df: merged DataFrame
    - emotion: one of ['depression', 'anxiety', 'stress']
    - threshold: threshold to classify binary label
    - n_estimators: number of trees in the forest
    """

    # Prepare features and target
    feature_cols = df.columns.difference(['depression', 'anxiety', 'stress', 'Database Collectors', 
                                          'File Number user', 'Directory', 'Suject'])
    X = df[feature_cols]
    y = np.where(df[emotion] > threshold, 1, 0)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt', random_state=42)
    rf.fit(X, y)

    # Calculate % of affected/unaffected
    pct_affected = np.sum(y) / len(y) * 100
    pct_unaffected = 100 - pct_affected
    print(f"\n--- {emotion.upper()} Analysis ---")
    print(f"Percentage of {emotion} (>{threshold}): {pct_affected:.2f}%")
    print(f"Percentage of non-{emotion}: {pct_unaffected:.2f}%")

    # Show top 10 feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Important Features:")
    print(importances.head(10))
