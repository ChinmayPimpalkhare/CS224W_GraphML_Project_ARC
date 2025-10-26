from dotenv import load_dotenv
load_dotenv()  # load .env if present

import os
key = os.getenv("TMDB_API_KEY")
if key:
    print("TMDB_API_KEY is set. Length:", len(key))
    print("First 6 chars:", key[:6])
else:
    print("TMDB_API_KEY is NOT set. Create a .env or export the variable.")