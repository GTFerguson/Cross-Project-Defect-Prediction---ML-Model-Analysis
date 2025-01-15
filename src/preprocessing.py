import os
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.io import arff
import pandas as pd

# Path to the datasets directory
DATASETS_DIR = "src/datasets/promise"
CORAL_DIR = os.path.join(DATASETS_DIR, "coral")

# Ensure the coral directory exists
os.makedirs(CORAL_DIR, exist_ok=True)


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


def process_datasets(datasets):
    """Processes all combinations of datasets using CORAL."""
    for i, source_name in enumerate(datasets):
        for j, target_name in enumerate(datasets):
            if i != j:  # Skip aligning dataset to itself
                print(f"Aligning {source_name} to {target_name}...")
                source_path = os.path.join(DATASETS_DIR, source_name)
                target_path = os.path.join(DATASETS_DIR, target_name)

                source_df = load_arff(source_path)
                target_df = load_arff(target_path)

                aligned_df = coral(source_df, target_df)
                output_path = os.path.join(CORAL_DIR, f"{source_name[:-5]}-{target_name[:-5]}.arff")
                save_arff(aligned_df, output_path)


if __name__ == "__main__":
    # List of dataset files in the directory
    datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".arff")]

    # Process each source-target pair
    process_datasets(datasets)
    print("Alignment complete. Files saved to:", CORAL_DIR)
