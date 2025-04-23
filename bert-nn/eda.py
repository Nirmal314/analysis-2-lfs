import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("netflix_titles.csv")

# Filter rows where 'country' contains 'India'
df_india = df[df['country'].str.contains('India', na=False)]

# Drop rows with missing values in important columns
df_india_clean = df_india.dropna(subset=['title', 'director', 'country'])

# Strip whitespace from string columns
str_cols = df_india_clean.select_dtypes(include='object').columns
df_india_clean[str_cols] = df_india_clean[str_cols].apply(lambda x: x.str.strip())

# Drop the specified columns
df_india_clean = df_india_clean.drop(columns=['country', 'director', 'cast', 'listed_in'])

# Convert 'date_added' to datetime
df_india_clean['date_added'] = pd.to_datetime(df_india_clean['date_added'], errors='coerce')

# --- Data Analysis ---

# 1. Basic overview
print("Data Info:")
print(df_india_clean.info())
print("\nData Description:")
print(df_india_clean.describe(include='all'))

# 2. Distribution of Shows by Type
type_counts = df_india_clean['type'].value_counts()
print("\nCount of Shows by Type:")
print(type_counts)
type_counts.plot(kind='bar', title='Count of Shows by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# 3. Number of Shows Added Over Time
shows_per_month = df_india_clean['date_added'].dt.to_period('M').value_counts().sort_index()
print("\nNumber of Shows Added Over Time (Year-Month):")
print(shows_per_month)
shows_per_month.plot(kind='line', title='Number of Shows Added Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Shows Added')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Distribution of Ratings
rating_counts = df_india_clean['rating'].value_counts()
print("\nDistribution of Ratings:")
print(rating_counts)
rating_counts.plot(kind='bar', title='Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 5. Duration Analysis
# Extract numeric duration
df_india_clean['duration_num'] = df_india_clean['duration'].str.extract('(\d+)').astype(int)

# Separate movies and TV shows
movies = df_india_clean[df_india_clean['type'] == 'Movie']
tv_shows = df_india_clean[df_india_clean['type'] == 'TV Show']

print("\nAverage movie duration (minutes):", movies['duration_num'].mean())
print("Average number of seasons for TV shows:", tv_shows['duration_num'].mean())

# 6. Top 10 Longest Movies
top_longest_movies = movies.sort_values(by='duration_num', ascending=False).head(10)
print("\nTop 10 Longest Movies:")
print(top_longest_movies[['title', 'duration_num']])
