"""Tests for the point-cloud reconstruction policy and the coverage error."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering, needed to build the env

import numpy as np
import pytest

from arm_nbv.reconstruction_policies.pointcloud_reconstruction_policy import (
    PointCloudReconstructionPolicy,
)


def _make_policy():
    return PointCloudReconstructionPolicy(
        point_min_dist=0.01, depth_max=2.0, device="cpu"
    )


def _frontal_plane_depth(h=32, w=32, z=0.5):
    return np.full((h, w), z, np.float32)


def test_backprojection_recovers_plane_depth():
    """A frontal plane at depth z (identity extrinsic) back-projects to points at z."""
    p = _make_policy()
    K = np.array([[30.0, 0, 16.0], [0, 30.0, 16.0], [0, 0, 1.0]])
    p.add_obs(K, np.eye(4), depth_image=_frontal_plane_depth(z=0.5))

    pts = p.reconstruct("point_cloud")
    assert pts.shape[1] == 3
    assert len(pts) > 0
    np.testing.assert_allclose(pts[:, 2], 0.5, atol=1e-5)


def test_dedup_discards_close_points():
    """Integrating the same frame twice adds no new points (dedup)."""
    p = _make_policy()
    K = np.array([[30.0, 0, 16.0], [0, 30.0, 16.0], [0, 0, 1.0]])
    depth = _frontal_plane_depth(z=0.5)

    p.add_obs(K, np.eye(4), depth_image=depth)
    n1 = len(p.reconstruct("point_cloud"))
    p.add_obs(K, np.eye(4), depth_image=depth)
    n2 = len(p.reconstruct("point_cloud"))
    assert n2 == n1


def test_grid_is_uint8_and_marks_surface():
    """The derived grid is uint8 (1,G,G,G) with high values near observed points."""
    p = _make_policy()
    K = np.array([[30.0, 0, 16.0], [0, 30.0, 16.0], [0, 0, 1.0]])
    p.add_obs(K, np.eye(4), depth_image=_frontal_plane_depth(z=0.5))
    pts = p.reconstruct("point_cloud")

    grid = p.reconstruct(
        "grid", grid_size=16, bbox_center=pts.mean(0), bbox_size=0.5
    )
    assert grid.shape == (1, 16, 16, 16)
    assert grid.dtype == np.uint8
    assert grid.max() == 255  # surface voxels
    assert (grid > 0).sum() > 0


def test_empty_grid_is_zero():
    p = _make_policy()
    grid = p.reconstruct(
        "grid", grid_size=16, bbox_center=np.zeros(3), bbox_size=0.5
    )
    assert grid.shape == (1, 16, 16, 16)
    assert grid.max() == 0


def _fake_env(gt_points):
    """A minimal stand-in for the robosuite env carrying only the coverage state."""
    from scipy.spatial import cKDTree

    fake = type("F", (), {})()
    fake.gt_surface_points = gt_points  # pre-seeded, so no mesh sampling is needed
    fake.gt_covered = np.zeros(len(gt_points), dtype=bool)
    fake.gt_tree = cKDTree(gt_points)
    fake.n_recon_consumed = 0
    return fake


def _coverage_error(env, recon_points, coverage_distance=0.01):
    from robosuite.environments.manipulation.reconstruct3D import Reconstruct3D

    return Reconstruct3D.compute_point_cloud_coverage_error(
        env, recon_points, coverage_distance=coverage_distance
    )


def test_coverage_error_full_and_empty():
    gt = np.random.default_rng(0).random((500, 3))

    # Reconstruction == GT -> fully covered -> error ~0
    err_full, info_full = _coverage_error(_fake_env(gt), gt.copy())
    assert err_full == pytest.approx(0.0)
    assert info_full["n_covered"] == len(gt)
    assert info_full["n_recon_points"] == len(gt)

    # Empty reconstruction -> nothing covered -> error 1
    err_empty, info_empty = _coverage_error(_fake_env(gt), np.zeros((0, 3)))
    assert err_empty == pytest.approx(1.0)
    assert info_empty["n_recon_points"] == 0


def test_coverage_error_partial():
    gt = np.random.default_rng(1).random((400, 3))
    half = gt[:200]  # only cover half of the GT points
    err, info = _coverage_error(_fake_env(gt), half, coverage_distance=1e-4)
    assert 0.4 < err < 0.6
    assert info["n_covered"] == pytest.approx(200, abs=5)


def test_coverage_incremental_matches_full_recompute():
    """Feeding the cloud in growing chunks must equal a one-shot computation."""
    rng = np.random.default_rng(2)
    gt = rng.random((2000, 3))
    recon = rng.random((3000, 3))

    incremental = _fake_env(gt)
    for n in (500, 1200, 2100, 3000):
        err_inc, info_inc = _coverage_error(incremental, recon[:n])

    err_full, info_full = _coverage_error(_fake_env(gt), recon)
    assert err_inc == err_full
    assert info_inc["n_covered"] == info_full["n_covered"]
    assert info_inc["n_recon_points"] == len(recon)


def test_coverage_raises_when_cloud_shrinks():
    """A shrinking cloud breaks the append-only assumption and must fail loudly."""
    rng = np.random.default_rng(3)
    gt = rng.random((1000, 3))
    recon = rng.random((2000, 3))

    env = _fake_env(gt)
    _coverage_error(env, recon)
    with pytest.raises(RuntimeError):
        _coverage_error(env, recon[:200])


# --- observable-surface ground truth (semantic exclusion at mesh build time) ---


@pytest.fixture(scope="module")
def robot_env():
    """The raw robosuite Reconstruct3D env, with its meshes computed."""
    from arm_nbv.config import TrainConfig
    from arm_nbv.utils.env_factory import create_env

    cfg = TrainConfig.load(
        ["configs/pointcloud_coverage.yaml", "configs/scripted_obs.yaml"]
    )
    env = create_env(cfg, seed=0, wrap_monitor=False)
    yield env.robot_env
    env.close()


def _face_normals(vertices, faces):
    """Per-face normals from the winding as stored (no re-orientation)."""
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    n = np.cross(v1 - v0, v2 - v0)
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)


def test_gt_mesh_has_no_downward_faces(robot_env):
    """Every face of the GT mesh can be seen from above (winding is outward already)."""
    v, f = robot_env.compute_gt_mesh()
    assert len(f) > 0
    nz = _face_normals(v, f)[:, 2]
    assert nz.min() > -0.1
    assert (nz > 0.9).any()  # the table top survived


def test_table_contributes_only_its_top(robot_env):
    """Nothing in the GT mesh lies below the table top: its sides/underside are dropped."""
    import mujoco

    v, _ = robot_env.compute_gt_mesh()
    model, data = robot_env.sim.model, robot_env.sim.data
    for gid in range(model.ngeom):
        if model.body(model.geom_bodyid[gid]).name != "table" or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        tv, _, _ = robot_env._geom_mesh(gid, mujoco.mjtGeom.mjGEOM_BOX)
        tv = tv @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        assert v[:, 2].min() == pytest.approx(tv[:, 2].max())
        return
    pytest.skip("no table box geom")


def test_box_keeps_five_of_six_faces(robot_env):
    """A box resting on the table keeps 10 of its 12 triangles (5 of 6 faces)."""
    import mujoco

    model, data = robot_env.sim.model, robot_env.sim.data
    object_bodies = {obj.root_body for obj in robot_env.primitives_on_table}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if model.body(model.geom_bodyid[gid]).name not in object_bodies:
            continue
        _, faces, normals = robot_env._geom_mesh(gid, mujoco.mjtGeom.mjGEOM_BOX)
        normal_z = (normals @ data.geom_xmat[gid].reshape(3, 3).T)[:, 2]
        assert len(faces) == 12
        assert (normal_z > -0.1).sum() == 10
        return
    pytest.skip("no box objects in this scene")
