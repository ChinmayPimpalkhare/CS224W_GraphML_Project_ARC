"""Construct a heterogeneous graph (users, movies, directors, actors, genres) from CSVs.
This is a stub—hook in your existing enrichment code in src/graphflix/data/movielens.py.
"""
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=False, help='path to data config')
    args = p.parse_args()
    print("TODO: build PyG HeteroData from processed CSVs.")

if __name__ == "__main__":
    main()