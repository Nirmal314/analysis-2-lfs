import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("netflix_titles.csv")
df_india = df[df['country'].str.contains('India', na=False)]
df_india_clean = df_india.dropna(subset=['title', 'director', 'country'])
str_cols = df_india_clean.select_dtypes(include='object').columns
df_india_clean[str_cols] = df_india_clean[str_cols].apply(lambda x: x.str.strip())
# Export only the 'description' column to a new CSV file
df_india_clean[['description']].to_csv("indian_movies_descriptions.csv", index=False)

print("Descriptions exported to india_descriptions.csv")