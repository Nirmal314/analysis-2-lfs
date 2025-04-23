import pandas as pd

df = pd.read_csv("netflix_titles.csv")

df_india = df[df['country'].str.contains('India', na=False)]

df_india_clean = df_india.dropna(subset=['title', 'director', 'country'])

str_cols = df_india_clean.select_dtypes(include='object').columns
df_india_clean[str_cols] = df_india_clean[str_cols].apply(lambda x: x.str.strip())

df_india_clean = df_india_clean.drop(columns=['country','director','cast','listed_in'])

df_india_clean.to_csv("indian_movies.csv", index=False)

print("Filtered and cleaned dataset saved as india_filtered_cleaned.csv")
