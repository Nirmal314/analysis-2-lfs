import pandas as pd

# File paths
tsv_file = "data/indian_movies_with_plots_parallel.tsv"
csv_file = "data/top_rated_indian_movies_since_2015.csv"
tsv_to_csv_file = "data/indian_movies_with_plots_parallel.csv"
output_file = "data/merged_indian_movies.csv"

# Step 1: Read and convert TSV to CSV
df_tsv = pd.read_csv(tsv_file, sep="\t")
df_tsv.to_csv(tsv_to_csv_file, index=False)

# Step 2: Read the CSV files
df_converted = pd.read_csv(tsv_to_csv_file)
df_csv = pd.read_csv(csv_file, na_values=['\\N'])

# Step 3: Merge on 'tconst' with a full outer join
merged_df = pd.merge(df_converted, df_csv, on='tconst', how='outer')

# Step 4: Write the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)