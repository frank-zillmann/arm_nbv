"""Mesh rendering using OpenCV rasterization (no OpenGL dependency).

Uses a painter's-algorithm triangle rasterizer built on cv2.fillConvexPoly.
This avoids all OpenGL/EGL/Filament context conflicts with MuJoCo.
"""

from typing import Tuple

import cv2
import numpy as np


def render_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    resolution: Tuple[int, int] = (64, 64),
    grayscale: bool = True,
    lighting: str = "ambient",
) -> np.ndarray:
    """
    Renders mesh via CPU rasterization (no OpenGL required).

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        extrinsic: (4, 4) camera-to-world pose matrix
        intrinsic: (3, 3) camera intrinsic matrix
        resolution: (H, W) output resolution
        grayscale: If True, return (H, W); else (H, W, 3)
        lighting: "unlit" (flat 1.0) or "ambient" (simple diffuse)

    Returns:
        Image array in [0, 1], shape (H, W) or (H, W, 3)
    """
    H, W = resolution
    empty = np.zeros((H, W) if grayscale else (H, W, 3), dtype=np.float32)

    if len(vertices) == 0 or len(faces) == 0:
        return empty

    # --- world → camera ---------------------------------------------------
    w2c = np.linalg.inv(extrinsic)
    verts_cam = (w2c[:3, :3] @ vertices.T + w2c[:3, 3:4]).T  # (N, 3)

    # --- project to pixel (u, v) -----------------------------------------
    proj = (intrinsic @ verts_cam.T).T  # (N, 3)
    z = proj[:, 2]
    safe_z = np.where(z > 1e-6, z, 1e-6)
    px = proj[:, :2] / safe_z[:, None]  # (N, 2)

    # --- cull & sort faces ------------------------------------------------
    face_z = z[faces]  # (M, 3)
    valid = face_z.min(axis=1) > 0.01
    valid_faces = faces[valid]
    if len(valid_faces) == 0:
        return empty

    mean_depth = face_z[valid].mean(axis=1)
    order = np.argsort(-mean_depth)  # back-to-front
    sorted_faces = valid_faces[order]

    # --- shading ----------------------------------------------------------
    fv = verts_cam[sorted_faces]  # (M', 3, 3)
    if lighting == "ambient":
        normals = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)
        shade = (np.abs(normals[:, 2]) * 0.6 + 0.3).astype(np.float32)
    else:
        shade = np.full(len(sorted_faces), 1.0, dtype=np.float32)

    # --- rasterize (painter's algorithm) ----------------------------------
    img = np.zeros((H, W), dtype=np.float32)
    tri_px = px[sorted_faces].astype(np.int32)  # (M', 3, 2)

    for i in range(len(sorted_faces)):
        cv2.fillConvexPoly(img, tri_px[i], float(shade[i]))

    if not grayscale:
        img = np.stack([img, img, img], axis=-1)
    return img
