import pandas as pd
from imdb import IMDb
from multiprocessing import Pool, cpu_count, Lock
import csv
import os

ia = IMDb()
lock = Lock()

INPUT_FILE = "top_rated_indian_movies_since_2015.csv"
OUTPUT_FILE = "indian_movies_with_plots_parallel_2.tsv"


def get_plot(tconst):
    try:
        movie = ia.get_movie(tconst[2:])
        plot = movie.get("plot", ["N/A"])[0]
        if plot != "N/A":
            print(f"SUCCESS: Plot found for movie {tconst}: '{plot[:60]}...'")
        else:
            print(f"FAILURE: No plot available for movie {tconst}.")
        return tconst, plot
    except Exception as e:
        print(f"FAILURE: Unable to fetch plot for {tconst}. Reason: {str(e)}")
        return tconst, "N/A"


def write_result(tconst, plot):
    with lock:
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([tconst, plot])


def process_movie(tconst):
    tconst, plot = get_plot(tconst)
    write_result(tconst, plot)
    return tconst


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    movie_ids = df["tconst"].tolist()

    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["tconst", "plot"])

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_movie, movie_ids)
