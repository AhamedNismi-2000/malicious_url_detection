import pandas as pd

# -------------------------------
# Pandas display settings for console
# -------------------------------
pd.set_option('display.max_columns', None)        # Show all columns
pd.set_option('display.expand_frame_repr', False) # Prevent wrapping to multiple lines
pd.set_option('display.max_colwidth', None)       # Show full content of each cell

# -------------------------------
# Path to your CSV
# -------------------------------
file_path = r"D:\Research Project\Research_Working _dir\malicious_url_detection\features\heuristic\heuristic_features.csv"

# -------------------------------
# Function to preview CSV
# -------------------------------
def preview_csv(file_path, n_rows=5):
    df = pd.read_csv(file_path, nrows=n_rows)
    print(f"\nPreviewing first {n_rows} rows:\n")
    print(df.to_string(index=False))  # Use to_string for nice console formatting

# -------------------------------
# Function to get CSV info
# -------------------------------
def csv_info(file_path):
    df = pd.read_csv(file_path, nrows=1)
    print("\nColumns:", df.columns.tolist())
    
    total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1
    print("Approximate number of rows:", total_rows)
    
    print("\nData types (first row only):")
    print(df.dtypes.to_string())

# -------------------------------
# Function to sample rows
# -------------------------------
def sample_csv(file_path, chunk_size=100_000, n_samples=5):
    samples = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        sampled_rows = chunk.sample(n=min(n_samples, len(chunk)))
        samples.append(sampled_rows)
        if len(samples) * n_samples >= n_samples:
            break
    result = pd.concat(samples)
    print(f"\nRandomly sampled {len(result)} rows:")
    print(result.to_string(index=False))

# -------------------------------
# Example usage
# -------------------------------
preview_csv(file_path, n_rows=1)
csv_info(file_path)
sample_csv(file_path, chunk_size=100_000, n_samples=5)
