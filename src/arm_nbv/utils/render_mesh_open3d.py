"""DEPRECATED: Mesh rendering using Open3D's offscreen renderer.

Causes segfaults in headless environments due to Filament engine bugs.
Use render_mesh.py (cv2-based) instead.
"""

import contextlib
import os
import sys
from typing import Tuple

import numpy as np
import open3d as o3d


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Temporarily suppress stdout and stderr at the C level (for Filament engine spam)."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(old_stdout, stdout_fd)
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

_renderer_cache: dict = {}


def _get_or_create_renderer(
    width: int, height: int
) -> o3d.visualization.rendering.OffscreenRenderer:
    """Get cached renderer or create new one for given resolution."""
    key = (width, height)
    if key not in _renderer_cache:
        with _suppress_stdout_stderr():
            _renderer_cache[key] = o3d.visualization.rendering.OffscreenRenderer(
                width, height
            )
    return _renderer_cache[key]


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
    DEPRECATED: Renders mesh using Open3D's offscreen renderer.

    Known to segfault on headless servers. Use render_mesh.render_mesh() instead.

    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        extrinsic: (4, 4) camera extrinsic matrix (camera-to-world pose)
        intrinsic: (3, 3) camera intrinsic matrix
        resolution: (H, W) output resolution
        grayscale: If True, return grayscale image (H, W), else RGB (H, W, 3)
        lighting: "unlit" or "ambient"

    Returns:
        Image array with values in [0, 1], shape (H, W) or (H, W, 3)
    """
    H, W = resolution

    if len(vertices) == 0 or len(faces) == 0:
        if grayscale:
            return np.zeros((H, W), dtype=np.float32)
        return np.zeros((H, W, 3), dtype=np.float32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    renderer = _get_or_create_renderer(W, H)
    renderer.scene.clear_geometry()
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    material = o3d.visualization.rendering.MaterialRecord()
    if lighting == "unlit":
        material.shader = "defaultUnlit"
    elif lighting == "ambient":
        material.shader = "defaultLit"
    else:
        raise ValueError(
            f"Unknown lighting mode: {lighting}. Use 'unlit' or 'ambient'."
        )
    renderer.scene.add_geometry("mesh", mesh, material)

    if lighting == "ambient":
        renderer.scene.scene.enable_sun_light(False)
        renderer.scene.set_lighting(
            renderer.scene.LightingProfile.SOFT_SHADOWS, (0, 0, 0)
        )

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    extrinsic_w2c = np.linalg.inv(extrinsic)
    renderer.setup_camera(intrinsic_o3d, extrinsic_w2c)

    img = np.asarray(renderer.render_to_image()).copy()
    result = img.astype(np.float32) / 255.0

    if grayscale:
        return np.mean(result, axis=-1)
    return result
