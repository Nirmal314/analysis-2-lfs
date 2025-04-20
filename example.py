from imdb import IMDb
import pandas as pd

ia = IMDb()


def get_plot(tconst):
    try:
        movie = ia.get_movie(tconst[2:])  # Remove 'tt' prefix if present
        plot = movie.get("plot", ["N/A"])[0]
        if plot != "N/A":
            print(f"SUCCESS: Plot found for movie {tconst}: '{plot[:60]}...'")
        else:
            print(f"FAILURE: No plot available for movie {tconst}.")
        return plot
    except Exception as e:
        error_message = str(e)
        print(
            f"FAILURE: Unable to fetch plot for movie {tconst}. Reason: {error_message}"
        )
        return "N/A"


df = pd.read_csv("top_rated_indian_movies_since_2015.csv")

df["plot"] = df["tconst"].apply(get_plot)
df.to_csv("indian_movies_with_plots.tsv", sep="\t", index=False)
