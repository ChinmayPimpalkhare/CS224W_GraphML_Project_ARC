import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

# TMDb API Configuration
TMDB_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your TMDb API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# File lock for thread-safe CSV writing
file_lock = Lock()


def download_movielens_1m(data_dir="data"):
    """Download and extract MovieLens 1M dataset"""
    Path(data_dir).mkdir(exist_ok=True)

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print("Downloading MovieLens 1M dataset...")

    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(data_dir)

    print("Download complete!")
    return f"{data_dir}/ml-1m"


def load_and_save_movielens_data(ml_dir, output_dir="output"):
    """Load MovieLens 1M data and save to CSV files"""
    Path(output_dir).mkdir(exist_ok=True)

    print("\nLoading MovieLens data...")

    # Load movies
    movies = pd.read_csv(
        f"{ml_dir}/movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    # Load ratings
    ratings = pd.read_csv(
        f"{ml_dir}/ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    # Load users
    users = pd.read_csv(
        f"{ml_dir}/users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )

    # Save to CSV
    print(f"Saving MovieLens data to {output_dir}/")
    movies.to_csv(f"{output_dir}/movies.csv", index=False)
    ratings.to_csv(f"{output_dir}/ratings.csv", index=False)
    users.to_csv(f"{output_dir}/users.csv", index=False)

    print(f"Saved {len(movies)} movies to movies.csv")
    print(f"Saved {len(ratings)} ratings to ratings.csv")
    print(f"Saved {len(users)} users to users.csv")

    return movies, ratings, users


def extract_year_from_title(title):
    """Extract year from movie title (format: 'Movie Name (YYYY)')"""
    match = re.search(r"\((\d{4})\)$", title)
    if match:
        year = match.group(1)
        clean_title = title[: match.start()].strip()
        return clean_title, year
    return title, None


def search_tmdb_by_title(title, year, api_key):
    """Search TMDb by movie title and year"""
    try:
        url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": api_key, "query": title, "year": year}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]
        return None
    except Exception as e:
        print(f"Error searching for '{title}': {e}")
        return None


def get_tmdb_movie_details(tmdb_id, api_key):
    """Fetch movie details from TMDb including cast and crew"""
    try:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        params = {"api_key": api_key, "append_to_response": "credits,external_ids"}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching TMDb ID {tmdb_id}: {e}")
        return None


def process_single_movie(row, api_key):
    """Process a single movie and return enriched data"""
    movie_id = row["movie_id"]
    title = row["title"]

    clean_title, year = extract_year_from_title(title)

    # Search for TMDb ID
    tmdb_id = search_tmdb_by_title(clean_title, year, api_key)

    result = {
        "movie_id": movie_id,
        "title": title,
        "tmdb_id": tmdb_id,
        "imdb_id": None,
        "directors": [],
        "actors": [],
        "genres": [],
    }

    if not tmdb_id:
        return result

    # Get detailed information
    details = get_tmdb_movie_details(tmdb_id, api_key)

    if not details:
        return result

    # Extract IMDb ID
    result["imdb_id"] = details.get("external_ids", {}).get("imdb_id")

    # Extract directors
    crew = details.get("credits", {}).get("crew", [])
    directors = [c for c in crew if c["job"] == "Director"]
    result["directors"] = [{"tmdb_id": d["id"], "name": d["name"]} for d in directors]

    # Extract top 5 actors
    cast = details.get("credits", {}).get("cast", [])[:5]
    result["actors"] = [
        {"tmdb_id": a["id"], "name": a["name"], "character": a.get("character", "")}
        for a in cast
    ]

    # Extract genres
    genres = details.get("genres", [])
    result["genres"] = [{"tmdb_id": g["id"], "name": g["name"]} for g in genres]

    return result


def append_to_csv(filepath, data, write_header=False):
    """Thread-safe append to CSV file"""
    with file_lock:
        df = pd.DataFrame([data])
        df.to_csv(filepath, mode="a", header=write_header, index=False)


def augment_with_tmdb_parallel(
    movies, api_key, output_dir="output", max_workers=10, chunk_size=50
):
    """
    Augment MovieLens movies with TMDb data using parallel processing.
    Saves progress incrementally to avoid data loss.
    """

    # Prepare output files
    movies_enriched_file = f"{output_dir}/movies_enriched.csv"
    directors_file = f"{output_dir}/directors.csv"
    actors_file = f"{output_dir}/actors.csv"
    genres_file = f"{output_dir}/genres.csv"
    movie_director_file = f"{output_dir}/movie_director_edges.csv"
    movie_actor_file = f"{output_dir}/movie_actor_edges.csv"
    movie_genre_file = f"{output_dir}/movie_genre_edges.csv"

    # Initialize files with headers
    pd.DataFrame(columns=["movie_id", "title", "tmdb_id", "imdb_id"]).to_csv(
        movies_enriched_file, index=False
    )
    pd.DataFrame(columns=["director_id", "tmdb_id", "name"]).to_csv(
        directors_file, index=False
    )
    pd.DataFrame(columns=["actor_id", "tmdb_id", "name"]).to_csv(
        actors_file, index=False
    )
    pd.DataFrame(columns=["genre_id", "tmdb_id", "name"]).to_csv(
        genres_file, index=False
    )
    pd.DataFrame(columns=["movie_id", "director_id"]).to_csv(
        movie_director_file, index=False
    )
    pd.DataFrame(columns=["movie_id", "actor_id", "character"]).to_csv(
        movie_actor_file, index=False
    )
    pd.DataFrame(columns=["movie_id", "genre_id"]).to_csv(movie_genre_file, index=False)

    # Track unique entities
    director_map = {}  # tmdb_id -> internal_id
    actor_map = {}
    genre_map = {}

    director_counter = 0
    actor_counter = 0
    genre_counter = 0

    processed_count = 0
    total_movies = len(movies)

    print(f"\n{'=' * 60}")
    print(f"Starting parallel TMDb augmentation for {total_movies} movies")
    print(f"Workers: {max_workers} | Chunk size: {chunk_size}")
    print(f"{'=' * 60}\n")

    # Process movies in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_movie = {
            executor.submit(process_single_movie, row, api_key): row
            for _, row in movies.iterrows()
        }

        # Process completed tasks
        for future in as_completed(future_to_movie):
            try:
                result = future.result()

                # Save movie enriched data
                movie_data = {
                    "movie_id": result["movie_id"],
                    "title": result["title"],
                    "tmdb_id": result["tmdb_id"],
                    "imdb_id": result["imdb_id"],
                }
                append_to_csv(movies_enriched_file, movie_data)

                # Process directors
                for director in result["directors"]:
                    director_tmdb_id = director["tmdb_id"]

                    if director_tmdb_id not in director_map:
                        director_map[director_tmdb_id] = director_counter
                        director_data = {
                            "director_id": director_counter,
                            "tmdb_id": director_tmdb_id,
                            "name": director["name"],
                        }
                        append_to_csv(directors_file, director_data)
                        director_counter += 1

                    # Add edge
                    edge_data = {
                        "movie_id": result["movie_id"],
                        "director_id": director_map[director_tmdb_id],
                    }
                    append_to_csv(movie_director_file, edge_data)

                # Process actors
                for actor in result["actors"]:
                    actor_tmdb_id = actor["tmdb_id"]

                    if actor_tmdb_id not in actor_map:
                        actor_map[actor_tmdb_id] = actor_counter
                        actor_data = {
                            "actor_id": actor_counter,
                            "tmdb_id": actor_tmdb_id,
                            "name": actor["name"],
                        }
                        append_to_csv(actors_file, actor_data)
                        actor_counter += 1

                    # Add edge
                    edge_data = {
                        "movie_id": result["movie_id"],
                        "actor_id": actor_map[actor_tmdb_id],
                        "character": actor.get("character", ""),
                    }
                    append_to_csv(movie_actor_file, edge_data)

                # Process genres
                for genre in result["genres"]:
                    genre_tmdb_id = genre["tmdb_id"]

                    if genre_tmdb_id not in genre_map:
                        genre_map[genre_tmdb_id] = genre_counter
                        genre_data = {
                            "genre_id": genre_counter,
                            "tmdb_id": genre_tmdb_id,
                            "name": genre["name"],
                        }
                        append_to_csv(genres_file, genre_data)
                        genre_counter += 1

                    # Add edge
                    edge_data = {
                        "movie_id": result["movie_id"],
                        "genre_id": genre_map[genre_tmdb_id],
                    }
                    append_to_csv(movie_genre_file, edge_data)

                processed_count += 1

                # Progress update
                if processed_count % chunk_size == 0:
                    progress = (processed_count / total_movies) * 100
                    print(
                        f"Progress: {processed_count}/{total_movies} ({progress:.1f}%) | "
                        f"Directors: {len(director_map)} | "
                        f"Actors: {len(actor_map)} | "
                        f"Genres: {len(genre_map)}"
                    )

            except Exception as e:
                print(f"Error processing movie: {e}")

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"{'=' * 60}\n")

    # Print final statistics
    print("=== Final Statistics ===")
    print(f"Movies processed: {processed_count}/{total_movies}")
    print(f"Unique directors: {len(director_map)}")
    print(f"Unique actors: {len(actor_map)}")
    print(f"Unique genres: {len(genre_map)}")
    print(f"\nAll data saved to: {output_dir}/")


def main():
    output_dir = "output"

    # Step 1: Download and save MovieLens data
    ml_dir = download_movielens_1m()
    movies, ratings, users = load_and_save_movielens_data(ml_dir, output_dir)

    # Step 2: Augment with TMDb data (parallel processing)
    print("\n" + "=" * 60)
    print("Starting TMDb augmentation...")
    print("=" * 60)

    augment_with_tmdb_parallel(
        movies,
        TMDB_API_KEY,
        output_dir=output_dir,
        max_workers=10,  # Adjust based on your needs
        chunk_size=50,  # Progress update frequency
    )

    print("\nAll processing complete!")
    print(f"\nOutput files in '{output_dir}/':")
    print("  - movies.csv (original MovieLens)")
    print("  - ratings.csv")
    print("  - users.csv")
    print("  - movies_enriched.csv (with TMDb/IMDb IDs)")
    print("  - directors.csv")
    print("  - actors.csv")
    print("  - genres.csv")
    print("  - movie_director_edges.csv")
    print("  - movie_actor_edges.csv")
    print("  - movie_genre_edges.csv")


if __name__ == "__main__":
    main()
