# scripts/check_env.py
import os

from dotenv import load_dotenv


def main() -> None:
    # Load variables from .env if present (repo root)
    load_dotenv()

    key = os.getenv("TMDB_API_KEY")
    if key:
        print("TMDB_API_KEY is set. Length:", len(key))
        print("First 6 chars:", key[:6])
    else:
        print("TMDB_API_KEY is NOT set. Create a .env or export the variable.")


if __name__ == "__main__":
    main()
