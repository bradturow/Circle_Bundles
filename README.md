# circle_bundles

`circle_bundles` is a Python package providing computational tools for constructing and analyzing
**discrete approximate circle bundles** from data. The package implements an end-to-end pipeline
for detecting circle-bundle structure in point clouds, computing local trivializations and
transition functions, and extracting global topological invariants such as orientability and
Euler-type characteristic classes.

The software accompanies the methods introduced in *Discrete Approximate Circle Bundles* (Turow and Perea) and
supports both synthetic and real-world datasets, including applications to optical flow patch
spaces.

---

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/YOUR_USERNAME/circle_bundles.git
cd circle_bundles
pip install -e .
```

After installation, you should be able to import the package from Python:

```python
import circle_bundles
```

---

## Quick example

The following example constructs a discrete approximate circle bundle from synthetic data
exhibiting circle-bundle structure (a torus-based example), builds a cover of the base space,
and checks whether the resulting bundle is orientable.

```python
import numpy as np

from circle_bundles.api import build_bundle
from circle_bundles.synthetic.tori_and_kb import sample_C2_torus
from circle_bundles.base_covers import MetricBallCover

# Sample synthetic data exhibiting circle-bundle structure (torus example)
rng = np.random.default_rng(0)
data, base_points, _ = sample_C2_torus(n_points=2000, rng=rng)

# Build a metric-ball cover of the base using evenly spaced landmarks
n_landmarks = 12
angs = np.linspace(0.0, 2.0 * np.pi, n_landmarks, endpoint=False)
landmarks = np.column_stack([np.cos(angs), np.sin(angs)])
radius = 1.4 * np.pi / n_landmarks

cover = MetricBallCover(base_points, landmarks, radius)

# Construct the discrete approximate circle bundle (quiet, README-safe)
bundle = build_bundle(data, cover, show=False, verbose=False)

# Check triviality via orientability
print("Bundle orientable:", bundle.classes.orientable)
```

---

## Notebooks

- `notebooks/tutorials/`  
  Minimal tutorial demonstrating the core pipeline.

- `notebooks/demos/` 
  Application-oriented and advanced demonstrations (not required for software review).

- `notebooks/paper_circle_bundles/` and `notebooks/paper_optical_flow/`  
  Notebooks reproducing figures and experiments from the accompanying papers.


---

## Citation

If you use this software in academic work, please cite it using the metadata provided in
`CITATION.cff`.

---

## License

This project is released under the MIT License.
