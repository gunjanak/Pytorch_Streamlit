from sklearn.preprocessing import StandardScaler
import numpy as np

# Normalize data using StandardScaler
def normalize_with_sklearn(df, columns):
    """
    Normalize specified columns in the DataFrame using sklearn's StandardScaler.

    Args:
        df (DataFrame): Input DataFrame.
        columns (list): List of columns to normalize.

    Returns:
        DataFrame: Normalized DataFrame.
        scaler (StandardScaler): Fitted StandardScaler object for inverse transformation.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# Denormalize predictions
def denormalize_with_sklearn(predictions, scaler, column_index):
    print("******************")
    print("Trying to denormalize")
    """
    Denormalize predictions using the fitted StandardScaler.

    Args:
        predictions (numpy.ndarray): Normalized predictions (1D array for a single column).
        scaler (StandardScaler): Fitted StandardScaler object.
        column_index (int): Index of the column in the scaler to reverse-transform.

    Returns:
        numpy.ndarray: Denormalized predictions.
    """
    # Ensure predictions are reshaped to 2D for inverse_transform
    predictions = predictions.reshape(-1, 1)  # (n_samples, 1)
    
    # Create a dummy array to match scaler's fitted columns
    dummy_array = np.zeros((predictions.shape[0], scaler.n_features_in_))
    
    # Populate the specific column to denormalize
    dummy_array[:, column_index] = predictions.flatten()
    
    # Apply inverse_transform and extract the denormalized column
    denormalized = scaler.inverse_transform(dummy_array)[:, column_index]
    
    return denormalized

