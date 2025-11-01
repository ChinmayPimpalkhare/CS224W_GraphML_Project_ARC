import argparse
import io
import pathlib
import zipfile

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=str, default="data/raw")
    args = p.parse_args()
    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print("Downloading MovieLens 1M…")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest)
    print(f"✓ Extracted to {dest}")


if __name__ == "__main__":
    main()
