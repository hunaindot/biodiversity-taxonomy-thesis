import os
import pandas as pd

def load_data(
    csv_path: str = "data/curated_data.csv",
    sample_n: int = 1000,
    sample: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:

    if os.path.exists(csv_path):
        print("Loading dataset from local file...")
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        print("Local file not found. Downloading from huggingface/hunain505/biodiversity-research-text...")
        url = "https://huggingface.co/datasets/Hunain505/biodiversity-research-text/resolve/main/dataset/curated_data/data.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.read_csv(url, low_memory=False)
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")

    count_summary = {
        "row_count": len(df),
        "web_of_science_record_count": df["UT (Unique WOS ID)"].count(),
        "web_of_science_record_distinct_count": df["UT (Unique WOS ID)"].nunique(),
        "duplicated_row_count": df.duplicated().sum()
    }

    print(f"Dataset shape: {df.shape}")
    print(f" Count Summary : {count_summary}")
    df = df.drop_duplicates()
    print(f"Dropped {count_summary['row_count'] - len(df)} duplicated rows. Remaining rows: {len(df)}")

    before_count = len(df)
    df = df.dropna(subset=["Abstract"])
    after_count = len(df)
    print(f"Dropped {before_count - after_count} rows without Abstract. Remaining rows: {after_count}")

    df_sample = None
    if sample:
        print(f"Sampling {sample_n} rows from the dataset of length {len(df)}.")
        df_sample = df.sample(n=sample_n, random_state=42)

    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
    print(f" {len(null_cols)} Columns with null values:\n{null_cols}")

    return df, df_sample