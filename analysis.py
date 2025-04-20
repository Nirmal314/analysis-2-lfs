import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/merged_indian_movies.csv')

# Data Cleaning
# Handle missing values
num_cols = ['startYear', 'endYear', 'runtimeMinutes', 'averageRating', 'numVotes']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['tconst', 'plot', 'titleType', 'primaryTitle', 'originalTitle', 'isAdult', 'genres']
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# Convert data types
df['isAdult'] = df['isAdult'].astype(bool)
df['startYear'] = df['startYear'].astype(int)
df['endYear'] = df['endYear'].fillna(pd.NA).astype('Int64')

# Exploratory Data Analysis (EDA)
print('Dataset Info:')
print(df.info())
print('\nDescriptive Statistics:')
print(df.describe())

# 1. Title Type Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='titleType', data=df)
plt.title('Title Type Distribution')
plt.xticks(rotation=45)
plt.savefig('images/title_type_distribution.png')
plt.close()

# 2. Average Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['averageRating'], bins=20)
plt.title('Average Rating Distribution')
plt.savefig('images/rating_distribution.png')
plt.close()

# 3. Top Genres
genres = df['genres'].str.split(',', expand=True).stack().value_counts()
plt.figure(figsize=(8, 5))
genres.head(10).plot(kind='bar')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.savefig('images/genre_distribution.png')
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png')
plt.close()

print('EDA plots saved as PNG files.')