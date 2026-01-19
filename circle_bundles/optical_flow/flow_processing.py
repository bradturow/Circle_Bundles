# optical_flow/flow_processing.py
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from .contrast import get_contrast_norms

PathLike = Union[str, Path]

TAG_FLOAT = 202021.25

__all__ = [
    "read_flo",
    "sample_from_frame",
    "get_patch_sample",
    "preprocess_flow_patches",
]


def read_flo(file: PathLike) -> np.ndarray:
    """
    Read a Middlebury/Sintel .flo optical flow file.

    Returns
    -------
    flow : (H, W, 2) float32 array
    """
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(str(file))
    if file.suffix.lower() != ".flo":
        raise ValueError(f"Expected a .flo file, got: {file.name}")

    with file.open("rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if float(magic) != float(TAG_FLOAT):
            raise ValueError(f"Invalid .flo file (bad magic {magic}) for {file}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)

    return data.reshape((h, w, 2))


def sample_from_frame(
    flo_path: PathLike,
    n_patches: int,
    *,
    d: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample n_patches random dxd optical flow patches from a .flo file.

    Returns
    -------
    samples : (n_patches, 2*d*d + 2) array
        [0 : d*d)       : u components (flattened Fortran order)
        [d*d : 2*d*d)   : v components
        [-2], [-1]      : (row, col) top-left corner in the frame
    """
    n_patches = int(n_patches)
    d = int(d)
    if n_patches <= 0:
        raise ValueError("n_patches must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    flow = read_flo(flo_path)
    H, W = flow.shape[:2]
    if d <= 0 or d > H or d > W:
        raise ValueError(f"Invalid patch size d={d} for flow of shape {(H, W)}")

    patchrows = rng.integers(0, H - d + 1, size=n_patches, endpoint=False)
    patchcols = rng.integers(0, W - d + 1, size=n_patches, endpoint=False)

    n_patch_cols = 2 * (d * d)
    patches = np.zeros((n_patches, n_patch_cols), dtype=np.float32)

    for i, (r, c) in enumerate(zip(patchrows, patchcols)):
        p = flow[r : r + d, c : c + d, :]  # (d,d,2)
        patches[i] = p.reshape((n_patch_cols,), order="F")

    corners = np.column_stack([patchrows, patchcols]).astype(np.int32)
    return np.concatenate([patches, corners], axis=1)


def get_patch_sample(
    flow_root: PathLike,
    *,
    patches_per_frame: int = 385,
    d: int = 3,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, List[List[Path]]]:
    """
    Sample patches_per_frame from every .flo file under flow_root/*/*.flo.

    Returns
    -------
    patch_df : DataFrame with columns ['patch','row','column','scene','frame']
              where 'patch' stores a 1D np.ndarray of length 2*d*d.
    file_paths : list of lists of .flo Paths, grouped by scene folder order
    """
    flow_root = Path(flow_root)
    if not flow_root.is_dir():
        raise NotADirectoryError(str(flow_root))

    patches_per_frame = int(patches_per_frame)
    d = int(d)
    rng = np.random.default_rng(int(random_state))

    file_paths: List[List[Path]] = []
    sample_list: List[np.ndarray] = []

    scene_num = 1
    scene_folders = sorted([p for p in flow_root.iterdir() if p.is_dir()])

    for subfolder in scene_folders:
        frame_num = 1
        scene_paths: List[Path] = []

        flo_files = sorted([p for p in subfolder.iterdir() if p.suffix.lower() == ".flo"])
        for flo_path in flo_files:
            scene_paths.append(flo_path)
            print(f"\rCollecting samples from scene {scene_num}, frame {frame_num}", end="")

            new_samples = sample_from_frame(flo_path, patches_per_frame, d=d, rng=rng)

            # append scene/frame cols
            scene_col = np.full((new_samples.shape[0], 1), scene_num, dtype=np.int32)
            frame_col = np.full((new_samples.shape[0], 1), frame_num, dtype=np.int32)
            new_samples = np.column_stack([new_samples, scene_col, frame_col])

            sample_list.append(new_samples)
            frame_num += 1

        file_paths.append(scene_paths)
        scene_num += 1

    print("\nFinalizing dataframe...", end="")

    all_patches = np.concatenate(sample_list, axis=0)
    n_patch_cols = 2 * (d * d)

    patches = all_patches[:, :n_patch_cols].astype(np.float32)
    rows = all_patches[:, n_patch_cols].astype(np.int32)
    cols = all_patches[:, n_patch_cols + 1].astype(np.int32)
    scenes = all_patches[:, n_patch_cols + 2].astype(np.int32)
    frames = all_patches[:, n_patch_cols + 3].astype(np.int32)

    patch_df = pd.DataFrame(
        {
            "patch": list(patches),
            "row": rows,
            "column": cols,
            "scene": scenes,
            "frame": frames,
        }
    )

    print(" Done")
    return patch_df, file_paths


def preprocess_flow_patches(
    patch_df: pd.DataFrame,
    *,
    hc_frac: float = 0.2,
    max_samples: int = 50_000,
    k_list: Sequence[int] = (300,),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Preprocess optical flow patches in patch_df.

    Requires patch_df['patch'] to contain length-2*n^2 vectors.

    Steps:
    - infer n
    - compute x/y mean
    - compute contrast norm + keep top hc_frac
    - downsample to max_samples
    - mean-center + contrast-normalize
    - compute density estimates 1 / dist_to_kNN for each k in k_list
    - sort by largest k density (descending)
    """
    if "patch" not in patch_df.columns:
        raise ValueError("patch_df must contain a 'patch' column.")
    if not (0 < float(hc_frac) <= 1):
        raise ValueError("hc_frac must be in (0,1].")

    patch_df = patch_df.copy().reset_index(drop=True)

    # infer n
    sample_patch = patch_df["patch"].iloc[0]
    total_len = int(len(sample_patch))
    if total_len % 2 != 0:
        raise ValueError(f"Patch length {total_len} is not even; expected 2*n^2.")
    d2 = total_len // 2
    n = int(np.sqrt(d2))
    if 2 * n * n != total_len:
        raise ValueError(f"Patch length {total_len} is not of the form 2*n^2.")

    patches = np.vstack(patch_df["patch"].values)  # (N, 2*n^2)

    # means
    patch_df["x mean"] = patches[:, :d2].mean(axis=1)
    patch_df["y mean"] = patches[:, d2:].mean(axis=1)

    # contrast norms
    patch_df["norm"] = get_contrast_norms(patches, patch_type="opt_flow")

    # keep top hc_frac
    keep = int(np.ceil(float(hc_frac) * len(patch_df)))
    patch_df = patch_df.sort_values("norm", ascending=False).head(keep).reset_index(drop=True)

    # downsample
    max_samples = int(max_samples)
    if len(patch_df) > max_samples:
        patch_df = patch_df.sample(n=max_samples, random_state=int(random_state)).reset_index(drop=True)

    if len(patch_df) == 0:
        raise ValueError("After filtering/downsampling, patch_df is empty.")

    # mean-center + normalize
    patches = np.vstack(patch_df["patch"].values)
    x_centered = patches[:, :d2] - patches[:, :d2].mean(axis=1, keepdims=True)
    y_centered = patches[:, d2:] - patches[:, d2:].mean(axis=1, keepdims=True)
    centered = np.hstack([x_centered, y_centered])

    norms = patch_df["norm"].to_numpy(dtype=float)
    safe_norms = np.where(norms == 0, 1e-8, norms)
    normalized = centered / safe_norms[:, None]
    patch_df["patch"] = list(normalized.astype(np.float32))

    # densities via kNN distances
    k_list = sorted(set(int(k) for k in k_list))
    if len(k_list) == 0:
        return patch_df

    K = k_list[-1]
    if K >= len(patch_df):
        raise ValueError(
            f"max(k_list)={K} must be < number of samples after preprocessing ({len(patch_df)})."
        )

    kdt = KDTree(normalized, leaf_size=30, metric="euclidean")
    dist, _ = kdt.query(normalized, k=K + 1, return_distance=True)  # dist[:,0]=0 self

    for k in k_list:
        patch_df[f"density_{k}"] = 1.0 / np.maximum(dist[:, k], 1e-12)

    patch_df = patch_df.sort_values(by=f"density_{K}", ascending=False).reset_index(drop=True)
    return patch_df
