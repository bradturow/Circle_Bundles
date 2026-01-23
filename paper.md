---
title: "circle_bundles: Computational tools for discrete approximate circle bundles"
tags:
  - topological data analysis
  - fiber bundles
  - applied topology
  - computational geometry
authors:
  - name: Brad Turow
    orcid: 0000-0002-0694-4411
    affiliation: 1
  - name: José A. Perea
    orcid: 0000-0002-6440-5096
    affiliation: 1
affiliations:
  - name: Northeastern University
    index: 1
date: 2026-01-23

bibliography: paper.bib
---

## Summary

`circle_bundles` is a Python software package implementing algorithms for the construction and analysis of *discrete approximate circle bundles* from data. The package provides computational tools for identifying circle-bundle–like structure in point clouds, estimating local trivializations, computing transition functions, and extracting global topological invariants such as (twisted) Euler classes.

The software accompanies and implements the methods introduced in *Discrete Approximate Circle Bundles* \cite{turow2025discrete}, and supports both synthetic and real-world datasets, including applications to optical flow patch spaces. In addition to the core algorithms, the repository includes tutorial notebooks demonstrating the pipeline in simple geometric settings, as well as notebooks reproducing figures and experiments from the associated research papers.

## Statement of need

Circle bundles arise naturally in geometry, topology, and applied data analysis, including settings such as dynamical systems, imaging, and equivariant representation learning. In practice, however, data are often available only as finite point clouds or sampled measurements, and classical bundle-theoretic tools are not directly applicable.

While existing topological data analysis libraries provide powerful methods for computing homology and persistence, there is limited software support for *constructing and analyzing bundle-like structure* in sampled data, particularly in the presence of nontrivial monodromy or reflection symmetry. The `circle_bundles` package addresses this gap by providing a computational framework for detecting, modeling, and analyzing approximate circle bundles using metric covers, local circular coordinates, and Čech cocycle methods.

## Functionality

The package implements the full pipeline described in \cite{turow2025discrete}, including:

- Construction of discrete approximate circle bundles from point cloud data using metric covers and local circular coordinates.
- Computation of transition functions between local trivializations, with support for both oriented circle bundles and non-orientable O(2)-bundle structures.
- Extraction of global topological invariants, including Euler classes and twisted Euler classes, via Čech cocycle representations.
- Support for both synthetic datasets and real data applications, including optical flow patch spaces.
- Visualization utilities for inspecting covers, nerves, local trivializations, transition functions, and global bundle structure.

In addition to the core library, the repository contains tutorial notebooks illustrating the methods in simple geometric settings, as well as reproducibility notebooks corresponding to figures and experiments in the associated research papers.

## Related work

The software builds on ideas from topological data analysis, fiber bundle theory, and circular coordinate methods. It complements existing TDA libraries by focusing on bundle structure rather than homology alone, and by supporting nontrivial monodromy and twisted coefficient systems. To our knowledge, `circle_bundles` is the first open-source implementation of a general pipeline for computing characteristic classes of approximate circle bundles directly from data.

## Availability and reproducibility

The `circle_bundles` package is open source and publicly available on GitHub. All experiments reported in the accompanying research papers can be reproduced using the notebooks provided in the repository. The software is written in Python and is designed to integrate with standard scientific computing tools.

## Acknowledgements

This work was partially supported by the National Science Foundation through CAREER award DMS-2415445.
