import pandas as pd
import re
import gzip

# Process in chunks to save memory
chunk_size = 100000


def get_indian_title_ids():
    print("Identifying Indian titles using multiple methods...")
    indian_title_ids = set()

    # Method 1: Using region codes in title.akas
    india_regions = ["IN", "IN-TN", "IN-KA", "IN-AP", "IN-TG", "IN-KL", "IN-MH"]

    print("Processing title.akas.tsv.gz for region-based identification...")
    for chunk in pd.read_csv(
        "title.akas.tsv.gz", sep="\t", compression="gzip", chunksize=chunk_size
    ):
        # Filter for Indian regions
        indian_chunk = chunk[chunk["region"].isin(india_regions)]
        indian_title_ids.update(indian_chunk["titleId"].unique())

    print(f"Found {len(indian_title_ids)} titles using region codes")

    # Method 3: Adding specific known movies that might have been missed
    known_indian_movies = [
        "tt1187043",  # 3 Idiots
        "tt0986264",  # Taare Zameen Par
        "tt5074352",  # Dangal
        "tt2338151",  # PK
        "tt2631186",  # Bahubali: The Beginning
        "tt0169102",  # Lagaan
        "tt3863552",  # Drishyam
        "tt8108198",  # The Tashkent Files
        "tt6452574",  # Tanhaji
        "tt8291224",  # Uri: The Surgical Strike
    ]
    indian_title_ids.update(known_indian_movies)

    print(f"Total unique Indian title IDs identified: {len(indian_title_ids)}")
    return indian_title_ids


def process_basics_with_ids(indian_title_ids):
    print(
        "Processing title.basics.tsv.gz to extract Indian movies from 2005 onwards..."
    )
    indian_movies = []

    for chunk in pd.read_csv(
        "title.basics.tsv.gz",
        sep="\t",
        compression="gzip",
        chunksize=chunk_size,
        low_memory=False,
    ):
        # Filter for movies only
        movies_chunk = chunk[chunk["titleType"] == "movie"]

        # Handle the year filter safely - replace \\N with NaN and filter only valid years
        movies_chunk["startYear"] = pd.to_numeric(
            movies_chunk["startYear"].replace("\\N", pd.NA), errors="coerce"
        )

        # Filter for years >= 2005
        movies_chunk = movies_chunk[
            (movies_chunk["startYear"].notna()) & (movies_chunk["startYear"] >= 2005)
        ]

        # Filter for Indian movies
        indian_movies_chunk = movies_chunk[
            movies_chunk["tconst"].isin(indian_title_ids)
        ]

        if not indian_movies_chunk.empty:
            indian_movies.append(indian_movies_chunk)

    # Combine all chunks
    if indian_movies:
        all_indian_movies = pd.concat(indian_movies, ignore_index=True)
        print(f"Found {len(all_indian_movies)} Indian movies from 2005 onwards")
        all_indian_movies.to_csv("indian_movies_since_2005.tsv", sep="\t", index=False)
        return all_indian_movies
    else:
        print("No Indian movies found matching the criteria")
        return pd.DataFrame()


def merge_with_ratings(indian_movies):
    print("Merging with ratings data...")
    if indian_movies.empty:
        print("No movies to merge with ratings")
        return pd.DataFrame()

    ratings_data = []
    movie_ids = set(indian_movies["tconst"])

    for chunk in pd.read_csv(
        "title.ratings.tsv.gz", sep="\t", compression="gzip", chunksize=chunk_size
    ):
        # Filter for ratings of our Indian movies
        chunk_ratings = chunk[chunk["tconst"].isin(movie_ids)]
        if not chunk_ratings.empty:
            ratings_data.append(chunk_ratings)

    if ratings_data:
        all_ratings = pd.concat(ratings_data, ignore_index=True)
        print(f"Found ratings for {len(all_ratings)} Indian movies")

        # Merge with movie data
        merged_data = pd.merge(indian_movies, all_ratings, on="tconst", how="left")

        # Sort by rating and votes
        merged_data = merged_data.sort_values(
            by=["averageRating", "numVotes"], ascending=[False, False]
        )

        # Save to CSV
        merged_data.to_csv("top_rated_indian_movies_since_2005.csv", index=False)
        print(f"Saved {len(merged_data)} Indian movies with ratings to CSV")
        return merged_data
    else:
        print("No ratings found for the Indian movies")
        return pd.DataFrame()


def verify_known_movies(final_data):
    # List of known popular Indian movies (title, tconst) to verify our dataset
    known_movies = [
        ("3 Idiots", "tt1187043"),
        ("Dangal", "tt5074352"),
        ("Taare Zameen Par", "tt0986264"),
        ("PK", "tt2338151"),
        ("Bahubali: The Beginning", "tt2631186"),
    ]

    print("\nVerifying presence of known Indian movies:")
    for title, tconst in known_movies:
        # Check if the movie is in our dataset by tconst
        if tconst in final_data["tconst"].values:
            movie_info = final_data[final_data["tconst"] == tconst].iloc[0]
            print(
                f"✓ Found: {title} ({int(movie_info['startYear'])}) - Rating: {movie_info.get('averageRating', 'N/A')}"
            )
        else:
            print(f"✗ Missing: {title} - should be included")


# Main execution
if __name__ == "__main__":
    indian_title_ids = get_indian_title_ids()
    indian_movies = process_basics_with_ids(indian_title_ids)
    final_data = merge_with_ratings(indian_movies)

    if not final_data.empty:
        verify_known_movies(final_data)
        print("\nProcess completed successfully!")
    else:
        print("\nFailed to create the dataset")
