from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

import torch

Cell = Tuple[int, int, int]


def _to_cell(coords: torch.Tensor, cell_size: float) -> Cell:
    cell = torch.floor(coords / cell_size).to(torch.int64)
    return int(cell[0].item()), int(cell[1].item()), int(cell[2].item())


def build_spatial_blocks(points: torch.Tensor, cell_size: float) -> Dict[Cell, List[int]]:
    blocks: Dict[Cell, List[int]] = defaultdict(list)
    for idx, point in enumerate(points):
        blocks[_to_cell(point, cell_size)].append(idx)
    return dict(blocks)


def query_candidate_neighbors(points: torch.Tensor, blocks: Dict[Cell, List[int]], cell_size: float = 0.25) -> List[List[int]]:
    offsets = list(product([-1, 0, 1], repeat=3))
    candidates: List[List[int]] = []

    for idx, point in enumerate(points):
        cell = _to_cell(point, cell_size)
        neighbor_ids: List[int] = []
        for dx, dy, dz in offsets:
            neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            neighbor_ids.extend(blocks.get(neighbor_cell, []))

        unique_ids = sorted(set(i for i in neighbor_ids if i != idx))
        candidates.append(unique_ids)

    return candidates
