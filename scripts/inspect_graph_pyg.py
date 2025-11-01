from pathlib import Path

import torch

P = Path("data/processed/ml1m") / "graph_pyg.pt"
data = torch.load(P)
print("Node types:", data.node_types)
print("Edge types:", data.edge_types)
for nt in data.node_types:
    print(f"{nt:>10} num_nodes:", data[nt].num_nodes)
for et in data.edge_types:
    ei = data[et].edge_index
    print(f"{et} edges:", ei.size(1), "  dtype:", ei.dtype)
print("Metadata:", data.metadata())
