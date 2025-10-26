import yaml, argparse

def load_config():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    with open(args.config) as f:
        return yaml.safe_load(f)