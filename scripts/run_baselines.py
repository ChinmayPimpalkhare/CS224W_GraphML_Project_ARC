import argparse
from src.graphflix.training.train import train_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    args = p.parse_args()
    train_model(args.config)

if __name__ == "__main__":
    main()