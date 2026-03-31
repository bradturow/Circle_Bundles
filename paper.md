---
title: "circle_bundles: Computational tools for discrete approximate circle bundles"
tags:
  - topological data analysis
  - fiber bundles
  - applied topology
  - computational geometry
  - dimensionality reduction
authors:
  - name: Brad Turow
    orcid: 0000-0002-0694-4411
    affiliation: 1
  - name: José A. Perea
    orcid: 0000-0002-6440-5096
    affiliation: 1
affiliations:
  - name: Northeastern University, Department of Mathematics
    index: 1
date: 2026-01-23
bibliography: paper.bib
---

## Summary

`circle_bundles` is a Python package implementing algorithms for constructing and analyzing *discrete approximate circle bundles* from point cloud data. A circle bundle is a continuously-varying family of circles parameterized by a base space — classical examples include the torus and Klein bottle as total spaces of circle bundles over the circle, and $SO(3)$ as a circle bundle over $S^2$. Such structures arise naturally in computer vision, computational chemistry, cryo-electron microscopy, and equivariant representation learning, where data lie near low-dimensional manifolds with nontrivial fiber structure.

The package provides an end-to-end pipeline for: constructing metric covers of a base space and computing discrete approximate local trivializations; extracting $O(2)$-valued transition functions between local circular coordinate systems; computing global topological invariants — specifically the first Stiefel-Whitney class and twisted Euler class — via Čech cocycle methods; assessing the persistence of these characteristic class representatives using a weights filtration on the nerve complex; and producing topologically-consistent coordinatization maps from the data into $V(2,d) \times_{O(2)} S^1$, a low-dimensional model for the universal circle bundle.

The software accompanies and implements the methods introduced in [@turow2025discrete] and supports both synthetic and real-world datasets, including applications to optical flow patch spaces [@opt_flow_torus; @Sintel] and 3D density functions. The repository includes tutorial notebooks demonstrating the full pipeline on worked examples, as well as reproducibility notebooks for all figures and experiments in the accompanying research paper.

## Statement of Need

Datasets arising in computer vision, imaging, and motion tracking often lie near low-dimensional manifolds with nontrivial global topology. When this topology takes the form of a circle bundle — as is the case for optical flow patches [@opt_flow_torus], natural image patches [@Klein_bottle], and projection images in cryo-EM [@cryoem_singer_sigworth] — standard methods based on persistent homology are insufficient: they can fail to detect the bundle structure, and they cannot produce the fiberwise coordinates or global classifications needed for downstream tasks.

The target audience is researchers in topological data analysis (TDA), computational geometry, and geometric machine learning who need to identify, classify, and coordinatize circle bundle structure in point cloud data. More broadly, the package is useful to practitioners in any domain where data are parameterized by a base space with nontrivial monodromy or reflection symmetry, such as molecular dynamics simulations, robotics, and representation learning on non-Euclidean spaces.

Existing TDA software packages — including `ripser` [@Ripser], `giotto-tda`, and `scikit-tda` — provide powerful tools for persistent homology but do not support construction or analysis of bundle structure. The `dreimac` package [@DREiMac] computes circular coordinates from persistent cohomology classes and serves as a complementary tool: `circle_bundles` uses `dreimac`-style circular coordinates [@Sparse_CC] as local trivializations, then analyzes how those local coordinates glue together to recover global bundle topology. The `FibeRed` approach [@scoccola2022fibered] addresses fiberwise dimensionality reduction using vector bundles, but does not handle $O(2)$-structure, orientability, or twisted characteristic classes. To our knowledge, `circle_bundles` is the first open-source implementation of a general pipeline for computing characteristic classes of approximate circle bundles directly from data.

## Software Design

The package is organized around a `Bundle` class that encapsulates a point cloud `X`, a feature map to a base space, and an open cover `U`. This design reflects the mathematical structure of the problem: a bundle is a triple $(X, \pi, B)$, and the cover is the primary computational object through which local and global information are linked.

The two core methods — `get_local_trivs()` and `get_classes()` — correspond to the two main algorithmic stages: (1) computing discrete approximate local trivializations and their $O(2)$-valued witness cocycles, and (2) extracting characteristic classes and persistence information from those cocycles. This separation allows users to inspect and intervene at the intermediate cocycle level, which is important for practitioners who may want to substitute custom local coordinate methods or adjust cover parameters.

A key design decision was to use $O(2)$ (rather than the full homeomorphism group of $S^1$) as the structure group throughout. This is mathematically justified by the fact that any circle bundle over a paracompact space admits $O(2)$-valued transition functions [@husemoller], and it makes the algorithms computationally tractable: transition matrices are computed as solutions to a minimax problem over a compact Lie group, and characteristic classes reduce to integer-valued Čech cocycles computable via Smith normal form. The coordinatization pipeline integrates Principal Stiefel Coordinates [@LeeEtAl2025] for dimensionality reduction of the resulting point cloud in the Stiefel manifold. The weights filtration and persistence of characteristic classes are implemented as a simplex-wise filtration on the nerve complex, making the pipeline compatible with multi-scale analysis of the data.

## Research Impact Statement

The algorithms implemented in this package were used to confirm the torus model for high-contrast optical flow patches proposed by @opt_flow_torus, to detect Klein bottle structure in a synthetic manifold dataset, and to infer the global topology of a 3-manifold of 3D prism densities — a setting where persistent homology is computationally intractable. These experiments are fully reproducible using the notebooks provided in the repository.

The software directly accompanies two preprints submitted to arXiv [@turow2025discrete], and the methods form the computational basis of a PhD dissertation at Northeastern University. An upcoming companion paper on optical flow will contain further applications enabled by this codebase. The package has been tested by colleagues in the Perea research group, and the public release is timed to support reproducibility of the results in the accompanying papers.

## AI Usage Disclosure

GitHub Copilot was used for code completion assistance during software development. No generative AI tools were used in the writing of this paper or the accompanying research articles.

## Acknowledgements

This work was partially supported by the National Science Foundation through CAREER award DMS-2415445.

## References