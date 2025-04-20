import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

years = [2020, 2021, 2022, 2023, 2024]
trending_movies = {}

def get_trending_movies_for_year(year):
    url = f"https://www.boxofficeindia.com/yearly-collections-{year}"
    
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        movies = []
        
        return movies
    except Exception as e:
        print(f"Error fetching data for {year}: {e}")
        return []

for year in years:
    trending_movies[year] = get_trending_movies_for_year(year)
    time.sleep(1)

imdb_df = pd.read_csv('top_rated_indian_movies_since_2005.csv')

for year in years:
    if trending_movies[year]:
        year_df = pd.DataFrame(trending_movies[year])
        
        enhanced_df = pd.merge(
            year_df, 
            imdb_df[imdb_df['startYear'] == year],
            on='primaryTitle',
            how='left'
        )
        
        enhanced_df.to_csv(f'india_domestic_trending_{year}.csv', index=False)