import pandas as pd

df = pd.read_csv('data/merged_indian_movies.csv')

# Remove rows where 'plot' is NaN or empty string
df_cleaned = df[df['plot'].notna() & (df['plot'].str.strip() != '')]

# Save the cleaned dataframe to a new CSV if needed
df_cleaned.to_csv('data/cleaned_ds.csv', index=False)

print(df_cleaned)
