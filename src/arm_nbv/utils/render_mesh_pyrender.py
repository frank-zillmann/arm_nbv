"""DEPRECATED: Mesh rendering using pyrender's offscreen renderer (EGL/OSMesa).

Causes GLError (invalid operation on glGenVertexArrays) when MuJoCo's EGL
context is active. Use render_mesh.py (cv2-based) instead.
"""

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import Tuple

import numpy as np
import pyrender
import trimesh

_renderer_cache: dict = {}


def _get_or_create_renderer(width: int, height: int) -> pyrender.OffscreenRenderer:
    key = (width, height)
    if key not in _renderer_cache:
        _renderer_cache[key] = pyrender.OffscreenRenderer(width, height)
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
    DEPRECATED: Renders mesh using pyrender's offscreen renderer.

    Conflicts with MuJoCo's EGL context. Use render_mesh.render_mesh() instead.

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
        return np.zeros((H, W) if grayscale else (H, W, 3), dtype=np.float32)

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)

    ambient = [0.3, 0.3, 0.3] if lighting == "ambient" else [1.0, 1.0, 1.0]
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0], ambient_light=ambient)
    scene.add(mesh)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0
    )

    cv2gl = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_pose = extrinsic.astype(np.float64) @ cv2gl
    scene.add(camera, pose=camera_pose)

    if lighting == "ambient":
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

    flags = (
        pyrender.RenderFlags.FLAT if lighting == "unlit" else pyrender.RenderFlags.NONE
    )
    r = _get_or_create_renderer(W, H)
    color, _ = r.render(scene, flags=flags)

    result = color.astype(np.float32) / 255.0
    return np.mean(result, axis=-1) if grayscale else result
