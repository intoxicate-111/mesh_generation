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
.
├── main.py
├── render_obj_views.py
├── data/
│   ├── multiview_dataset.py
│   └── views.json (generated or user-provided)
├── models/
│   ├── point_representation.py
│   ├── covariance.py
├── geometry/
│   ├── spatial_blocks.py
│   ├── edge_weights.py
│   ├── mesh_builder.py
├── rendering/
│   └── renderer.py
├── training/
│   └── train.py
├── utils/
│   └── math_utils.py
└── ReadMe.md
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
- trimesh (for STL support)

Run training:

```bash
python main.py --views_json ./data/views.json
```

Generate training views from an OBJ/STL mesh:

```bash
python render_obj_views.py --mesh_path ./assets/mesh.obj --output_dir ./data --num_views 24
```

STL example:

```bash
python render_obj_views.py --mesh_path ./assets/mesh.stl --output_dir ./data --num_views 24
```

`views.json` format (N views with camera poses):

```json
{
	"data_root": "./data/images",
	"fov": 60.0,
	"views": [
		{
			"image": "view_000.png",
			"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
			"T": [0.0, 0.0, 2.7]
		},
		{
			"image": "view_001.png",
			"R": [[0.9848, 0, 0.1736], [0, 1, 0], [-0.1736, 0, 0.9848]],
			"T": [0.0, 0.0, 2.7]
		}
	]
}
```

Notes:

- `image` supports relative paths under `data_root`
- `R` is a 3x3 world-to-view rotation matrix
- `T` is a 3D translation vector
- Image will be loaded as grayscale silhouette target

Saved outputs (auto-generated):

- `outputs/target.pt`
- `outputs/checkpoint_final.pt`
- `outputs/masks_final.pt`
- `outputs/mesh_final.obj`
- `outputs/loss_history.csv`

Additional CLI options:

- `--steps` (default: 400)
- `--num_points` (default: 64)
- `--image_size` (default: 128)
- `--lr` (default: 1e-2)

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