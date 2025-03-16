import os
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.io import arff
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler


# Path to the datasets directory
DATASETS_DIR = "datasets/promise"
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


def median_heuristic_sigma(X):
    """
    Compute the median of the pairwise distances of samples in X,
    which can be used as the bandwidth (σ) for the RBF kernel.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)

    Returns:
    - sigma: float, median heuristic value for σ
    """
    # Compute all pairwise distances
    dists = pairwise_distances(X)

    # Extract the upper triangular portion (excluding the diagonal)
    upper_tri_indices = np.triu_indices_from(dists, k=1)
    median_distance = np.median(dists[upper_tri_indices])

    return median_distance


def compute_sigma_from_datasets(X_source, X_target):
    """
    Compute σ using the median heuristic on the combined data.

    Parameters:
    - X_source: np.ndarray, source dataset features
    - X_target: np.ndarray, target dataset features

    Returns:
    - sigma: float, computed σ value
    """
    combined = np.vstack([X_source, X_target])
    return median_heuristic_sigma(combined)


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
    """
    Computes the gradient of the MMD with respect to the source data.

    For each source sample x_i:
      grad_i = - (2/(n * sigma**2)) * sum_j k(x_i, x_j) (x_i - x_j)
               + (2/(m * sigma**2)) * sum_j k(x_i, y_j) (x_i - y_j)

    Parameters:
    - X_s: np.ndarray, source data of shape (n, d)
    - X_t: np.ndarray, target data of shape (m, d)
    - kernel: callable, kernel function accepting (X_s, X, sigma)
    - sigma: float, RBF kernel bandwidth

    Returns:
    - grad: np.ndarray, gradient with same shape as X_s
    """
    n = X_s.shape[0]
    m = X_t.shape[0]
    grad = np.zeros_like(X_s)

    # Compute kernel matrices for source-source and source-target
    K_ss = kernel(X_s, X_s, sigma)  # shape: (n, n)
    K_st = kernel(X_s, X_t, sigma)  # shape: (n, m)

    for i in range(n):
        # Source-to-source term: note the negative sign
        term_ss = np.sum(K_ss[i][:, None] * (X_s[i] - X_s), axis=0)
        # Source-to-target term: positive contribution
        term_st = np.sum(K_st[i][:, None] * (X_s[i] - X_t), axis=0)

        grad[i] = - (2 / (n * sigma**2)) * term_ss + (2 / (m * sigma**2)) * term_st

    return grad


def mmd_alignment_adam(source_ds, target_ds, kernel=rbf_kernel, sigma=2.0, learning_rate=0.1,
                       min_epochs=50, max_epochs=100, improvement_threshold=1e-5,
                       beta1=0.9, beta2=0.999, epsilon=1e-8, log_file="mmd_alignment_log.txt"):
    """
    Align the source dataset to the target dataset by minimizing the MMD using the Adam optimizer.

    Parameters:
      - source_ds: np.ndarray, source data (n_samples, n_features)
      - target_ds: np.ndarray, target data (m_samples, n_features)
      - kernel: callable, kernel function
      - sigma: float, bandwidth for the RBF kernel
      - learning_rate: float, base step size for the Adam update
      - min_epochs: int, minimum number of epochs to run
      - max_epochs: int, maximum number of epochs to run
      - improvement_threshold: float, minimum decrease in MMD required to continue after min_epochs
      - beta1: float, exponential decay rate for the first moment estimates
      - beta2: float, exponential decay rate for the second moment estimates
      - epsilon: float, small constant to prevent division by zero
      - log_file: str, file path to log the initial MMD, final MMD, and total epochs run.

    Returns:
      - X_s_aligned: np.ndarray, the aligned source data
    """
    # Initialize aligned source as a copy of the source dataset
    X_s_aligned = source_ds.copy()

    initial_mmd = compute_mmd(source_ds, target_ds, kernel, sigma)
    print(f"Starting MMD alignment with Adam optimizer...")
    print(f"Source shape: {source_ds.shape}, Target shape: {target_ds.shape}")
    print(f"Initial MMD: {initial_mmd:.6f}")

    previous_mmd = initial_mmd
    total_epochs_run = 0

    # Initialize Adam's first and second moment variables (same shape as source_ds)
    m = np.zeros_like(source_ds)
    v = np.zeros_like(source_ds)
    t = 0  # timestep

    for epoch in range(max_epochs):
        t += 1  # increment time step
        # Compute gradient of MMD with respect to source data
        grad = compute_grad(X_s_aligned, target_ds, kernel, sigma)

        # Check for NaN or divergent values in the gradient
        if np.isnan(grad).any() or np.isinf(grad).any():
            print("NaN or Inf detected in gradient. Exiting...")
            break
        if np.linalg.norm(grad, ord=2) > 1e6:
            print("Gradient norm too large. Exiting...")
            break

        # Update first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Compute bias-corrected estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Adam update: compute step and update aligned data
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        X_s_aligned -= update

        total_epochs_run += 1

        # Compute current MMD value after the update
        current_mmd = compute_mmd(X_s_aligned, target_ds, kernel, sigma)

        # Print every 10 epochs for monitoring
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: MMD = {current_mmd:.6f}")

        # After min_epochs, check if improvement is significant
        if epoch >= min_epochs:
            improvement = previous_mmd - current_mmd
            if improvement < improvement_threshold:
                print(f"Improvement {improvement:.6f} is less than threshold {improvement_threshold}, stopping at epoch {epoch}")
                break

        previous_mmd = current_mmd

    final_mmd = compute_mmd(X_s_aligned, target_ds, kernel, sigma)

    # Log initial and final MMD values and total epochs run to a file
    with open(log_file, "a") as f:
        f.write(f"Sigma: {sigma:.6f}, Initial MMD: {initial_mmd:.6f}, Final MMD: {final_mmd:.6f}, Epochs run: {total_epochs_run}\n")

    print(f"Final MMD: {final_mmd:.6f}, Total epochs run: {total_epochs_run}")
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


def standardize_data(source_df, target_df):
    """
    Standardizes the numeric columns of source and target DataFrames using a StandardScaler
    fitted on the combined data of only the common numeric columns.

    Parameters:
    - source_df: pandas DataFrame for the source dataset.
    - target_df: pandas DataFrame for the target dataset.

    Returns:
    - source_df, target_df: DataFrames with the common numeric columns standardized.
    """
    # Get numeric columns from each DataFrame
    source_numeric = source_df.select_dtypes(include=['number']).columns
    target_numeric = target_df.select_dtypes(include=['number']).columns

    # Find the common numeric columns between source and target
    common_numeric = source_numeric.intersection(target_numeric)

    if len(common_numeric) == 0:
        print("No common numeric columns found for standardization.")
        return source_df, target_df

    # Combine the numeric columns from both datasets
    combined = pd.concat([source_df[common_numeric], target_df[common_numeric]])

    scaler = StandardScaler()
    scaler.fit(combined)

    # Transform only the common numeric columns
    source_df[common_numeric] = scaler.transform(source_df[common_numeric])
    target_df[common_numeric] = scaler.transform(target_df[common_numeric])

    return source_df, target_df


def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df


def process_datasets(datasets, da_type):
    """Processes all combinations of datasets using CORAL."""
    for i, source_name in enumerate(datasets):
        for j, target_name in enumerate(datasets):
            if i != j:  # Skip aligning dataset to itself
                print(f"Aligning {source_name} to {target_name}...")

                source_path = os.path.join(DATASETS_DIR, source_name)
                target_path = os.path.join(DATASETS_DIR, target_name)

                # Load datasets and handling missing values
                source_df = handle_missing_values(load_arff(source_path))
                target_df = handle_missing_values(load_arff(target_path))

                # Clean the column names to allow easier matching
                source_df = clean_columns(source_df)
                target_df = clean_columns(target_df)

                # Standardize numeric features before alignment
                source_df, target_df = standardize_data(source_df, target_df)

                if da_type.lower() == "coral":
                    # CORAL
                    aligned_df = coral(source_df, target_df)
                    output_path = os.path.join(CORAL_DIR, f"{source_name[:-5]}-{target_name[:-5]}.arff")
                elif da_type.lower() == "mmd":
                    # Exclude target column for MMD alignment
                    x_s = source_df.iloc[:, :-1].values
                    x_t = target_df.iloc[:, :-1].values

                    sigma = compute_sigma_from_datasets(x_s, x_t)
                    print("Sigma: ", sigma)

                    # Perform MMD alignment
                    x_s_aligned = mmd_alignment_adam(x_s, x_t, sigma=sigma)

                    # Convert the aligned data back to a DataFrame and preserve the target column
                    aligned_df = pd.DataFrame(x_s_aligned, columns=source_df.columns[:-1])
                    aligned_df[source_df.columns[-1]] = source_df.iloc[:, -1].values

                    output_path = os.path.join(MMD_DIR, f"{source_name[:-5]}-{target_name[:-5]}.arff")
                else:
                    print("Invalid DA type, exiting process...")
                    return

                save_arff(aligned_df, output_path)
                print("Alignment complete. File saved to:", output_path)


if __name__ == "__main__":
    # List of dataset files in the directory
    datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".arff")]

    print(f"{len(datasets)} datasets to process.")
    print("Process starting...")
    # Process each source-target pair
    process_datasets(datasets, "mmd")
