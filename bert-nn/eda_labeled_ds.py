import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the categorized dataset
df = pd.read_csv("indian_movies_descriptions_ollama_categorized.csv")

# --- Basic EDA ---

print("Dataset info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nCareer category distribution:")
print(df['career_category'].value_counts())

# Plot career category distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='career_category', order=df['career_category'].value_counts().index)
plt.title('Distribution of Career Categories in Movie Descriptions')
plt.xlabel('Career Category')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Text length analysis ---

# Add a column for description length (number of words)
df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))

print("\nDescription word count statistics:")
print(df['description_word_count'].describe())

# Plot description length distribution by career category
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='career_category', y='description_word_count',
            order=df['career_category'].value_counts().index)
plt.title('Description Length by Career Category')
plt.xlabel('Career Category')
plt.ylabel('Number of Words in Description')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Optional: Save summary statistics to CSV ---

summary = df.groupby('career_category')['description_word_count'].describe()
summary.to_csv("career_category_description_length_summary.csv")
print("\nSummary statistics saved to career_category_description_length_summary.csv")
