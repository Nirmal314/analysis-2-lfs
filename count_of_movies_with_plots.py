import pandas as pd

# Read the TSV file that you previously saved
df = pd.read_csv("indian_movies_with_plots.tsv", sep="\t")

# Total number of movies
total_movies = df.shape[0]

# Number of movies with a valid plot (assuming "N/A" indicates a missing plot)
# movies_with_plot = df[df["plot"] != "N/A"].shape[0]

# Count movies with a valid plot (non-null)
movies_with_plot = df["plot"].notna().sum()

print(f"Movies with plots: {movies_with_plot} out of {total_movies} total movies.")
