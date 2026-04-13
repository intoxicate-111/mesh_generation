# 🧩 Geometry-Aware Mesh Generation

A differentiable mesh generation pipeline based on point primitives + anisotropic covariance + rendering supervision.

## 🚀 Overview

This project implements a geometry-aware pipeline that:

- Represents geometry using points with anisotropic covariance (3D Gaussian style)
- Uses spatial blocking to efficiently find local neighbors
- Defines edge connectivity via symmetric anisotropic compatibility
- Constructs a mesh from points
- Optimizes geometry using differentiable rendering (PyTorch3D)

## 🎯 Core Idea

Traditional methods define connectivity using distance (`kNN` / radius). This project replaces it with a covariance-based anisotropic compatibility criterion.

Each point encodes a local surface patch, and edges are formed only when:

- Two points are spatially close
- Their local geometric structures are mutually compatible

## 🧠 Key Insight

- Covariance ≠ plane
- Covariance ≈ local surface distribution

We use covariance to approximate:

- Tangent directions (large eigenvalues)
- Normal direction (small eigenvalue)

## 📦 Pipeline

`Points → Spatial Blocks → Candidate Neighbors → Covariance Compatibility → Edge Weights → Mesh Construction → Rendering → Loss`

## 🏗️ Project Structure

```text
project/
├── models/
│   ├── point_representation.py
│   ├── covariance.py
│
├── geometry/
│   ├── spatial_blocks.py
│   ├── edge_weights.py
│   ├── mesh_builder.py
│
├── rendering/
│   ├── renderer.py
│
├── training/
│   ├── train.py
│
├── utils/
│   ├── math_utils.py
│
└── README.md
```

## 🔧 Installation

Install PyTorch3D:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Requirements:

- Python ≥ 3.9
- PyTorch ≥ 2.x
- CUDA (recommended)

Run training:

```bash
cd project
python -m training.train
```

Saved outputs (auto-generated):

- `project/outputs/target.pt`
- `project/outputs/checkpoint_final.pt`
- `project/outputs/masks_final.pt`
- `project/outputs/loss_history.csv`

## 📌 Modules

### 1. Point Representation

Each point stores:

- `points: [N, 3]`
- `quat: [N, 4]`
- `log_scale: [N, 3]`

Covariance:

$$
\Sigma = R \cdot \mathrm{diag}(s^2) \cdot R^T
$$

Where:

- $R$ comes from normalized quaternion
- $s = \exp(\text{log\_scale})$

### 2. Spatial Blocking (Neighbor Acceleration)

Used to reduce neighbor search complexity.

API:

```python
def build_spatial_blocks(points, cell_size):
		...

def query_candidate_neighbors(points, blocks):
		...
```

Behavior:

- Divide space into a 3D grid
- Assign each point to a voxel
- Query neighbors only from:
	- same voxel
	- adjacent voxels

Important:

- Blocking only selects candidates, **not** final edges.

### 3. Edge Compatibility (Core)

Define symmetric anisotropic compatibility:

$$
d_{ij} = (x_j - x_i)^T \Sigma_i^{-1}(x_j - x_i)
$$

$$
S_{ij} = \exp(-\alpha d_{ij}) \cdot \exp(-\alpha d_{ji})
$$

API:

```python
def compute_edge_weights(points, Sigma, candidate_idx, alpha):
		...
```

Properties:

- symmetric
- differentiable
- anisotropic
- geometry-aware

### 4. Mesh Construction

Stage 1 (required):

- Fixed topology (`icosphere`)
- Optimize vertex positions

Stage 2 (optional):

- Build edges from candidate graph
- Implemented helper: `build_edges_from_candidate_graph(...)`

Triangle score:

$$
S_{ijk} = S_{ij}S_{jk}S_{ki}
$$

- Implemented helper: `triangle_scores_from_edge_matrix(...)`

### 5. Rendering (PyTorch3D)

Use silhouette rendering:

```python
mesh = Meshes(verts=[verts], faces=[faces])
pred = renderer(mesh, cameras=cameras)
mask = pred[..., 3]
```

### 6. Loss

Rendering:

$$
L_{\text{render}} = \mathrm{MSE}(\text{mask}_{\text{pred}}, \text{mask}_{\text{gt}})
$$

Regularization:

- Laplacian smoothing
- Edge length loss

## ⚙️ Training Stages

- Stage 1: Basic Rendering Loop
	- Fixed mesh topology
	- Optimize vertices
	- Verify rendering + gradients
- Stage 2: Covariance Learning
	- Add quaternion + scale
	- Ensure stability
- Stage 3: Spatial Blocking
	- Replace brute-force neighbor search
- Stage 4: Edge Weights
	- Implement anisotropic compatibility
- Stage 5: Mesh from Graph
	- Build mesh from weighted edges

## 🚫 Constraints

Do **not**:

- Use full pairwise distance matrix
- Use hard thresholds during training
- Optimize raw covariance matrices
- Start with dynamic topology
- Add RGB loss at early stage

## 📊 Success Criteria

- Loss decreases
- Mesh converges to target shape
- Stable training (no NaN)
- Neighbor search efficient
- Edge weights meaningful

## 🧠 Summary

Build a mesh pipeline where:

- covariance encodes local geometry
- spatial blocks provide efficient neighbor search
- anisotropic compatibility defines connectivity
- rendering supervision drives optimization

## 🔥 Core Contribution (Conceptual)

Replace distance-based connectivity with covariance-driven anisotropic geometric compatibility.