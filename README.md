# circle_bundles

`circle_bundles` is a Python package providing computational tools for constructing and analyzing  
**discrete approximate circle bundles** in data.

The package implements an end-to-end pipeline for detecting circle-bundle structure in point
clouds, computing local trivializations and transition functions, and extracting global
topological invariants such as orientability and Euler-type characteristic classes.

This software accompanies the methods introduced in *Discrete Approximate Circle Bundles*
(Turow & Perea) and supports both synthetic and real-world datasets, including applications
to optical flow patch spaces.

---

## Documentation

Full documentation, including tutorials and worked examples, is available at:

**https://circle-bundles.readthedocs.io**

---

## Installation

`circle_bundles` requires Python 3.9 or newer.

Install directly from GitHub using `pip`:

```bash
pip install git+https://github.com/bradturow/Circle_Bundles.git
```

After installation, you should be able to import the package from Python:

```python
import circle_bundles
```

---

## Quick example

A minimal end-to-end example using a classical circle bundle: the Hopf fibration  
\( S^3 \to S^2 \). The Hopf fibration is a non-trivial orientable circle bundle with Euler
number \( \pm 1 \).

```python
import numpy as np
import circle_bundles as cb

# Sample points on S^3
n_samples = 3000
rng = np.random.default_rng(0)
s3_data = cb.sample_sphere(n_samples, dim=3, rng=rng)

# Hopf projection S^3 -> S^2
base_points = cb.hopf_projection(s3_data)

# Build an open cover of S^2
n_landmarks = 60
s2_cover = cb.get_s2_fibonacci_cover(
    base_points,
    n_vertices=n_landmarks,
)

bundle = cb.Bundle(X=s3_data, U=s2_cover.U)
bundle.get_local_trivs(verbose=False)
class_result = bundle.get_classes()

print(class_result.summary_text)
```

---

## Tutorials

Tutorials and example workflows are provided as part of the documentation and rendered
via Read the Docs.

The source notebooks live in:

- `notebooks/tutorials/`  
  Core tutorials demonstrating the main analysis pipeline and example datasets.

For narrative explanations and rendered outputs, see the documentation site.

---

## Citation

If you use this software in academic work, please cite:

Brad Turow and José A. Perea.  
**circle_bundles: Discrete Approximate Circle Bundles for Data Analysis** (version 0.1.0), 2026.  
Northeastern University.  
Available at: https://github.com/bradturow/circle_bundles

A `CITATION.cff` file is also provided for convenience.

---

## License

This project is released under the MIT License.
