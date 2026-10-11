import pandas as pd

input_file = "data/processed/clean_dataset.csv"
output_file = "data/processed/clean_dataset_v2.csv"

# Read the dataset
df = pd.read_csv(input_file)

print("Original rows:", len(df))

# Randomly keep 20,421 rows
df_v2 = df.sample(n=20421, random_state=42)

# Save Dataset v2
df_v2.to_csv(output_file, index=False)

print("Dataset v2 rows:", len(df_v2))
print("Saved to:", output_file)