import os
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.io import arff
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel as rbf


# Path to the datasets directory
DATASETS_DIR = "src/datasets/promise"
CORAL_DIR = os.path.join(DATASETS_DIR, "coral")
MMD_DIR = os.path.join(DATASETS_DIR, "mmd")

# Ensure the aligned dataset directories exists
os.makedirs(CORAL_DIR, exist_ok=True)
os.makedirs(MMD_DIR, exist_ok=True)


def load_arff(filepath):
    """Loads an ARFF file and returns a Pandas DataFrame."""
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Convert all `bytes` columns to `str`
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    # Ensure nominal values are in the correct format (true or false)
    if 'problems' in df.columns:
        df['problems'] = df['problems'].map({'yes': 'true', 'no': 'false'})

    return df


def save_arff(dataframe, filepath):
    """Saves a Pandas DataFrame to an ARFF file."""
    with open(filepath, 'w') as f:
        f.write("@relation aligned_dataset\n\n")
        for col in dataframe.columns[:-1]:  # Exclude the target column
            f.write(f"@attribute {col} numeric\n")
        f.write(f"@attribute {dataframe.columns[-1]} {{false,true}}\n\n")  # Adjust target labels as needed
        f.write("@data\n")
        dataframe.to_csv(f, index=False, header=False)


def validate_data(dataframe, dataset_name):
    """Validate dataset and handle missing or invalid values."""
    print(f"Validating {dataset_name}...")

    # Identify numeric columns
    numeric_columns = dataframe.select_dtypes(include=['number']).columns

    # Fill missing values only for numeric columns with column means
    dataframe[numeric_columns] = dataframe[numeric_columns].fillna(dataframe[numeric_columns].mean())

    # Handle non-numeric columns separately (e.g., categorical, boolean)
    non_numeric_columns = dataframe.select_dtypes(exclude=['number']).columns
    for col in non_numeric_columns:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])  # Use mode for non-numeric columns

    return dataframe


def coral(source_df, target_df):
    """
    Implements the CORAL algorithm to align the source dataset
    with the target dataset.
    """

    # Validate datasets
    source_df = validate_data(source_df, "Source Dataset")
    target_df = validate_data(target_df, "Target Dataset")

    source_data = source_df.iloc[:, :-1].to_numpy()  # Exclude target column
    target_data = target_df.iloc[:, :-1].to_numpy()  # Exclude target column

    # Compute covariance matrices
    source_cov = np.cov(source_data, rowvar=False) + np.eye(source_data.shape[1])
    target_cov = np.cov(target_data, rowvar=False) + np.eye(target_data.shape[1])

    # Compute whitening and coloring transformation
    source_whiten = fractional_matrix_power(source_cov, -0.5)
    target_color = fractional_matrix_power(target_cov, 0.5)

    # Align source data
    aligned_data = source_data @ source_whiten @ target_color

    # Replace source data with aligned data
    aligned_df = pd.DataFrame(aligned_data, columns=source_df.columns[:-1])
    aligned_df[source_df.columns[-1]] = source_df.iloc[:, -1].values  # Add target column back

    return aligned_df


# MMD
def rbf_kernel(x, y, sigma=1.0):
    """
    Compute the RBF (Gaussian) kernel matrix between two datasets.

    Parameters:
    - X (np.ndarray): Source dataset (N x D)
    - Y (np.ndarray): Target dataset (M x D)
    - sigma (float): Bandwidth for RBF kernel

    Returns:
    - np.ndarray: Kernel matrix of shape (N, M)
    """
    pairwise_sq_dists = np.sum(x**2, axis=1)[:, None] + np.sum(y**2, axis=1) - 2 * np.dot(x, y.T)
    return np.exp(-pairwise_sq_dists / (2 * sigma**2))


def compute_mmd(X_s, X_t, kernel=rbf_kernel, sigma=1.0):
    """
    Computes the MMD metric between source and target datasets.

    X_s: np.ndarray (n_samples_s, n_features) - Source dataset
    X_t: np.ndarray (n_samples_t, n_features) - Target dataset
    kernel: callable - Kernel function to compute pairwise similarities
    sigma: float - Bandwidth for RBF kernel
    """
    K_ss = rbf(X_s, X_s, gamma=1 / (2 * sigma**2)).mean()
    K_tt = rbf(X_t, X_t, gamma=1 / (2 * sigma**2)).mean()
    K_st = rbf(X_s, X_t, gamma=1 / (2 * sigma**2)).mean()
    return K_ss + K_tt - 2 * K_st


def compute_grad(X_s, X_t, kernel, sigma):
    grad = np.zeros_like(X_s)
    K_ss = kernel(X_s, X_s, sigma)  # Source-to-source kernel matrix
    K_st = kernel(X_s, X_t, sigma)  # Source-to-target kernel matrix
    for i in range(X_s.shape[0]):
        grad[i] = 2 * (np.sum(K_ss[i][:, None] * (X_s[i] - X_s), axis=0) / X_s.shape[0] -
                       np.sum(K_st[i][:, None] * (X_s[i] - X_t), axis=0) / X_t.shape[0])
    return grad



def mmd_alignment(source_ds, target_ds, kernel=rbf_kernel, sigma=1.0, learning_rate=0.1, epochs=50):
    """
    Align the source dataset to the target dataset using MMD minimization.

    source_ds: np.ndarray - Source dataset
    target_ds: np.ndarray - Target dataset
    kernel: callable - Kernel function
    sigma: float - Bandwidth for RBF kernel
    learning_rate: float - Step size for alignment
    epochs: int - Number of optimization steps
    """
    # Initialize aligned source as a copy of X_s
    X_s_aligned = source_ds.copy()

    print(f"Starting MMD alignment...")
    print(f"Source shape: {source_ds.shape}, Target shape: {target_ds.shape}")
    print(f"Initial MMD: {compute_mmd(source_ds, target_ds, kernel, sigma):.6f}")

    for epoch in range(epochs):
        # Compute gradient of MMD w.r.t. source data
        #grad = np.zeros_like(source_ds)
        grad = compute_grad(X_s_aligned, target_ds, kernel, sigma)

        # Normalize gradient
        grad /= source_ds.shape[0]

        # Check for NaN or divergent values
        if np.isnan(grad).any() or np.isinf(grad).any():
            print("NaN or Inf detected in gradient. Exiting...")
            break
        if np.linalg.norm(grad, ord=2) > 1e6:
            print("Gradient norm too large. Exiting...")
            break

        # Update source data
        X_s_aligned -= learning_rate * grad

        # Print MMD value every few epochs
        if epoch % 10 == 0:
            mmd_value = compute_mmd(X_s_aligned, target_ds, kernel, sigma)
            print(f"Epoch {epoch}: MMD = {mmd_value:.6f}")

    return X_s_aligned


def handle_missing_values(dataset):
    """
    Handles missing values in a mixed-type dataset (both numeric and non-numeric).
    """
    # Check if dataset is a numpy array; if so, assume numeric
    if isinstance(dataset, np.ndarray):
        nan_count = np.isnan(dataset).sum()
        print(f"Handling {nan_count} NaN values in the numeric dataset...")
        col_means = np.nanmean(dataset, axis=0)
        inds = np.where(np.isnan(dataset))
        dataset[inds] = np.take(col_means, inds[1])
        return dataset

    # If dataset is a pandas DataFrame
    elif isinstance(dataset, pd.DataFrame):
        print("Handling NaN values in DataFrame...")

        # Handle numeric columns
        numeric_columns = dataset.select_dtypes(include=['number']).columns
        dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())

        # Handle non-numeric columns
        non_numeric_columns = dataset.select_dtypes(exclude=['number']).columns
        for col in non_numeric_columns:
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])  # Replace with mode (most common value)

        return dataset

    else:
        raise TypeError("Dataset must be a numpy array or pandas DataFrame.")


def process_datasets(datasets):
    """Processes all combinations of datasets using CORAL."""
    for i, source_name in enumerate(datasets):
        for j, target_name in enumerate(datasets):
            if i != j:  # Skip aligning dataset to itself
                print(f"Aligning {source_name} to {target_name}...")

                source_path = os.path.join(DATASETS_DIR, source_name)
                target_path = os.path.join(DATASETS_DIR, target_name)

                source_df = handle_missing_values(load_arff(source_path))
                target_df = handle_missing_values(load_arff(target_path))

                # Exclude target column for MMD alignment
                x_s = source_df.iloc[:, :-1].values
                x_t = target_df.iloc[:, :-1].values

                # Perform MMD alignment
                x_s_aligned = mmd_alignment(x_s, x_t)

                # Convert the aligned data back to a DataFrame and preserve the target column
                aligned_df = pd.DataFrame(x_s_aligned, columns=source_df.columns[:-1])
                aligned_df[source_df.columns[-1]] = source_df.iloc[:, -1].values

                # CORAL
                #aligned_df = coral(source_df, target_df)
                #output_path = os.path.join(CORAL_DIR, f"{source_name[:-5]}-{target_name[:-5]}.arff")
                #save_arff(aligned_df, output_path)

                output_path = os.path.join(MMD_DIR, f"{source_name[:-5]}-{target_name[:-5]}.arff")

                # source_path = os.path.join(DATASETS_DIR, source_name)
                # target_path = os.path.join(DATASETS_DIR, target_name)

                # source_df = load_arff(source_path)
                # target_df = load_arff(target_path)

                #aligned_df = coral(source_df, target_df)
                #output_path = os.path.join(CORAL_DIR, f"{source_name[:-5]}-{target_name[:-5]}.arff")

                save_arff(aligned_df, output_path)


if __name__ == "__main__":
    # List of dataset files in the directory
    datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".arff")]

    # Process each source-target pair
    process_datasets(datasets)
<<<<<<< HEAD
    print("Alignment complete. Files saved to:", MMD_DIR)
=======
    print("Alignment complete. Files saved to:", CORAL_DIR)
>>>>>>> 1808f16409febb1f0c9d26bf36212ef4fd2fbd4e
