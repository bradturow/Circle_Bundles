# optical_flow/flow_frames.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .flow_processing import read_flo  # unified .flo reader

PathLike = Union[str, Path]

__all__ = [
    "change_path",
    "get_labeled_frame",
    "write_video_from_frames",
    "get_labeled_video",
    "annotate_optical_flow",
    "get_sintel_scene_folders",
]


def change_path(flow_path: PathLike, old_base: PathLike, new_base: PathLike) -> Path:
    """
    Map a Sintel .flo path to its corresponding .png frame path.

    Example:
      old_base/.../flow/<scene>/frame_0001.flo
    -> new_base/.../clean/<scene>/frame_0001.png
    """
    flow_path = Path(flow_path)
    old_base = Path(old_base)
    new_base = Path(new_base)

    try:
        rel = flow_path.relative_to(old_base)
    except Exception as e:
        raise ValueError(
            f"flow_path is not under old_base.\nflow_path={flow_path}\nold_base={old_base}"
        ) from e

    if rel.suffix.lower() != ".flo":
        raise ValueError(f"Expected a .flo file, got: {flow_path}")

    return (new_base / rel).with_suffix(".png")


def get_labeled_frame(
    frame_path: PathLike,
    save_path: Optional[PathLike],
    patch_df: pd.DataFrame,
    *,
    dot_radius: int = 3,
    custom_colors: Sequence[str] = ("#4B0082", "#FF4500", "#ADD8E6"),
    show: bool = False,
):
    """
    Draw colored dots at patch locations on a Sintel frame.

    patch_df must have columns: ['row', 'column', 'color'].
    'color' can be any hashable ID (ints/strings). IDs are mapped to custom_colors.
    """
    # Lazy imports (keeps package lightweight)
    try:
        import matplotlib.colors as mcolors
    except Exception as e:  # pragma: no cover
        raise ImportError("get_labeled_frame requires matplotlib.") from e

    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # pragma: no cover
        raise ImportError("get_labeled_frame requires Pillow (PIL).") from e

    required = {"row", "column", "color"}
    missing = required - set(patch_df.columns)
    if missing:
        raise ValueError(f"patch_df missing columns: {missing}")

    rows = patch_df["row"].to_numpy()
    cols = patch_df["column"].to_numpy()
    cids = patch_df["color"].to_numpy()

    unique_ids = list(pd.unique(cids))
    color_map = {
        cid: tuple(int(255 * c) for c in mcolors.to_rgb(custom_colors[i % len(custom_colors)]))
        for i, cid in enumerate(unique_ids)
    }

    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    r = int(dot_radius)
    for x, y, cid in zip(cols, rows, cids):
        rgb = color_map[cid]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rgb)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

    if show:
        img.show()

    return img


def write_video_from_frames(
    image_files: Sequence[PathLike],
    output_video: PathLike,
    *,
    fps: float = 10.0,
    codec: str = "MJPG",
) -> None:
    """Write an AVI video from a sequence of image paths using OpenCV."""
    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise ImportError("write_video_from_frames requires opencv-python (cv2).") from e

    if len(image_files) == 0:
        raise ValueError("image_files is empty")

    output_video = str(output_video)
    first = cv2.imread(str(image_files[0]))
    if first is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, float(fps), (w, h))

    for p in image_files:
        frame = cv2.imread(str(p))
        if frame is None:
            raise ValueError(f"Could not read image: {p}")
        out.write(frame)

    out.release()


def get_labeled_video(
    flo_paths: Sequence[PathLike],
    *,
    scene_num: int,
    patch_df: pd.DataFrame,
    inds_list: Sequence[np.ndarray],
    old_flow_base: PathLike,
    new_frame_base: PathLike,
    out_dir: PathLike,
    fps: float = 10.0,
) -> Path:
    """
    Create a labeled-frame video for a single scene.

    patch_df must contain at least columns: ['scene','frame','row','column'].
    inds_list is a list of boolean masks (length == len(patch_df)) selecting patches.
    Each mask becomes one color group (0,1,2,...).

    Notes
    -----
    Assumes flo_paths are ordered by frame number, corresponding to frames 1..T.
    """
    patch_df = patch_df.reset_index(drop=True)

    out_dir = Path(out_dir)
    folder_path = out_dir / f"Scene_{scene_num}_labeled"
    folder_path.mkdir(parents=True, exist_ok=True)

    # Overwrite frames if they exist
    for p in folder_path.glob("Labeled_frame_*.png"):
        p.unlink()

    save_paths: list[Path] = []

    flo_paths = list(flo_paths)
    for n, flo_path in enumerate(flo_paths):
        frame_num = n + 1

        selected_frames: list[pd.DataFrame] = []
        for i, inds in enumerate(inds_list):
            inds = np.asarray(inds, dtype=bool)
            if inds.shape[0] != len(patch_df):
                raise ValueError(f"inds_list[{i}] has wrong length: {inds.shape[0]} vs {len(patch_df)}")

            frame_patches = patch_df[
                (patch_df["scene"] == scene_num)
                & (patch_df["frame"] == frame_num)
                & inds
            ].copy()

            if len(frame_patches) == 0:
                continue

            frame_patches["color"] = i
            selected_frames.append(frame_patches)

        selected_patches = (
            pd.concat(selected_frames, ignore_index=True) if selected_frames else
            pd.DataFrame(columns=["row", "column", "color"])
        )

        png_path = change_path(flo_path, old_base=old_flow_base, new_base=new_frame_base)
        save_path = folder_path / f"Labeled_frame_{frame_num:04d}.png"

        get_labeled_frame(png_path, save_path, selected_patches)
        save_paths.append(save_path)

    video_path = folder_path / "Labeled_video.avi"
    write_video_from_frames(save_paths, video_path, fps=fps)
    return video_path


def annotate_optical_flow(
    image_dir: PathLike,
    flow_dir: PathLike,
    save_dir: PathLike,
    *,
    lattice_res: Tuple[int, int] = (20, 30),
    scale: float = 0.2,
    width: float = 0.0025,
    arrow_color: str = "red",
) -> None:
    """
    Overlay optical flow vectors at lattice points on each frame and save as PDFs.
    Skips the last image in image_dir (since it may not have a .flo match).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError("annotate_optical_flow requires matplotlib.") from e

    try:
        import imageio.v3 as iio
    except Exception as e:  # pragma: no cover
        raise ImportError("annotate_optical_flow requires imageio.") from e

    image_dir = Path(image_dir)
    flow_dir = Path(flow_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() == ".png"])
    for img_path in image_files[:-1]:
        flow_path = flow_dir / (img_path.stem + ".flo")

        try:
            img = iio.imread(img_path)
            flow = read_flo(flow_path)
        except Exception as e:
            print(f"❌ Failed to process {img_path.name}: {e}")
            continue

        H, W = flow.shape[:2]
        n_rows, n_cols = lattice_res

        ys = np.linspace(0, H - 1, n_rows).astype(int)
        xs = np.linspace(0, W - 1, n_cols).astype(int)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

        coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        u = flow[grid_y, grid_x, 0].reshape(-1)
        v = -flow[grid_y, grid_x, 1].reshape(-1)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img)
        ax.quiver(
            coords[:, 0], coords[:, 1], u, v,
            color=arrow_color, angles="xy",
            scale_units="xy", scale=1 / float(scale),
            width=float(width),
        )
        ax.axis("off")

        save_path = save_dir / (img_path.stem + ".pdf")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def get_sintel_scene_folders(
    base_path: PathLike,
    scene_type: str = "clean",
    scene: str = "all",
) -> list[Path]:
    """
    Return folder paths for each scene in Sintel.

    base_path should contain subfolders like: 'clean', 'final', 'flow'
    """
    base_path = Path(base_path)
    if scene_type not in {"clean", "final", "flow"}:
        raise ValueError("scene_type must be one of {'clean','final','flow'}")

    root = base_path / scene_type
    if not root.exists():
        raise ValueError(f"Folder does not exist: {root}")

    if scene == "all":
        scene_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    else:
        scene_names = [scene]

    return [root / s for s in scene_names]
