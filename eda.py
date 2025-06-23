import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_with_targets(df, target_columns=["depression", "anxiety", "stress"], feature_columns=None):
    """
    Plot correlations between feature columns and psychological targets (depression, anxiety, stress).
    
    Parameters:
    - df: DataFrame containing both features and targets
    - target_columns: list of target variable names
    - feature_columns: optional list of feature columns (if None, inferred automatically)
    
    Returns:
    - dict of correlation Series for each target
    """
    correlations = {}

    if feature_columns is None:
        # Exclude target columns to get features
        feature_columns = [col for col in df.columns if col not in target_columns]

    for target in target_columns:
        # Compute correlations
        corr_series = df[feature_columns].corrwith(df[target])
        correlations[target] = corr_series
        
        # Plot bar chart
        plt.figure(figsize=(10, 4))
        plt.bar(corr_series.index, corr_series.values)
        plt.xticks(rotation=45)
        plt.ylabel(f"Correlation with {target.capitalize()}")
        plt.title(f"Correlation between {target.capitalize()} and Features")
        plt.tight_layout()
        plt.show()

    # Optional: heatmap for univariate analysis (e.g., top correlated features with depression)
    for target in target_columns:
        sorted_corr = correlations[target].sort_values(ascending=False)
        top_features = sorted_corr[:7].index.tolist() + sorted_corr[-7:].index.tolist()
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[top_features].corr(), cmap="coolwarm", annot=True)
        plt.title(f"Heatmap of Top Correlated Features with {target.capitalize()}")
        plt.show()

        print(f"\nTop 7 positively correlated features with {target}:\n{sorted_corr[:7]}")
        print(f"\nTop 7 negatively correlated features with {target}:\n{sorted_corr[-7:]}")
        
    return correlations
