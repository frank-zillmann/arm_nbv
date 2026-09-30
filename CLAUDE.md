# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Guidelines (added by me, not Claude)
- keep commit messages short and human-like, e.g. "fix bug in chamfer distance computation" or "add new config for nvblox TSDF", do not do Co-Author commit or giant commit messages
- keep code as short and simple as possible, if you see a possibility to refactor such that it can become simpler/shorter/more readable, do it
- keep doc strings short and precise, do not write long explanations in doc strings, if necessary comment in the code itself
- rather use (professional) libraries than writing your own implementation, do not hesitate from installing a new library if it makes the code simpler/shorter/more readable
- do not ask for permission unless for really safety-critical changes

## Project

Learning next-best-view policies for a wrist-mounted camera on a robot arm (Panda, via robosuite/MuJoCo)
to efficiently reconstruct novel scenes with reinforcement learning (PPO via stable-baselines3), combining
RL with multi-view geometry / 3D reconstruction (Open3D or NVIDIA nvblox TSDF).

## Commands

### Setup
```bash
source ./install/setup.sh   # creates conda env, installs torch/nvblox (if CUDA), robosuite (editable), package (editable)
```
`external/robosuite` is a git submodule (`git clone --recursive`) — a fork with a custom `Reconstruct3D` manipulation
environment at `external/robosuite/robosuite/environments/manipulation/reconstruct3D.py`.

### Training
```bash
# Default: SAC (off-policy, replay buffer) + point-cloud reconstruction + coverage reward
python scripts/train.py --config configs/pointcloud_coverage.yaml

# Legacy configs (open3d/nvblox reconstruction) are kept for reference only
python scripts/train.py --config configs/open3d_chamfer_distance.yaml

# Combine configs (later files override earlier); configs/debug.yaml and configs/demo.yaml are overlays, not standalone configs
python scripts/train.py --config configs/pointcloud_coverage.yaml configs/debug.yaml

# Resume from checkpoint
python scripts/train.py --config configs/pointcloud_coverage.yaml --checkpoint path/to/checkpoint.zip
```

### Evaluation
```bash
# Hard-coded baseline: TCP moves along table-edge corners, camera stays pointed at table center. Useful for comparing against learned policies.
python scripts/eval_scripted.py --config configs/open3d_chamfer_distance.yaml --n_episodes 3

# Render evaluation episode videos from data/logs/.../eval_data/episode_xxxx
python scripts/create_episode_videos.py path/to/run/eval_data/episode_xxxx
```

### Tests
```bash
pytest                                    # testpaths = tests/ (see pyproject.toml)
pytest tests/reconstruct3d/test_chamfer_distance.py
pytest tests/reconstruct3d/test_chamfer_distance.py::test_name
pytest tests/reconstruction_policies/
```

## Architecture

### RL environment stack
`Reconstruct3DGymWrapper` (`src/arm_nbv/reconstruct3d_gym_wrapper.py`) is the top-level `gym.Env`. It wraps a
robosuite `Reconstruct3D` env (Panda arm, OSC_POSE delta control, wrist camera `robot0_eye_in_hand`) and drives a
pluggable **reconstruction policy** each step:

1. `robot_env.step(action)` → raw obs dict with RGB/depth from the wrist camera (and `birdview`/`frontview` when in
   eval mode, for logging).
2. Depth is converted to a real depth map and fed to `reconstruction_policy.add_obs(...)` along with camera
   intrinsics/extrinsics.
3. `reconstruction_policy.reconstruct(type="point_cloud" | "grid" | "mesh" | "tsdf", ...)` produces the current
   reconstruction (point cloud for the coverage reward, a derived 3D grid for the observation, mesh/tsdf for legacy
   metrics and eval visualization).
4. `robot_env.reward(...)` scores reconstruction quality against a ground-truth mesh/SDF/point-cloud computed at
   `reset()` (`compute_gt_mesh`, `compute_static_env_sdf` — defined in the robosuite `Reconstruct3D` env, not
   this repo).
5. Observation dict is assembled from whatever is listed in `config.observations` (`camera_pose`,
   `camera_pose_history`, `camera_rotation_matrix`, `recon_grid`, `wrist_image`, `sdf_grid`).

`_integrate_and_reconstruct` is the shared pipeline used by both `step()` and `reset()`. `TimingStats` (same file)
instruments each stage when `config.benchmark=True` — only works with `n_envs=1`/`DummyVecEnv`, not `SubprocVecEnv`.

### Reconstruction policies (`src/arm_nbv/reconstruction_policies/`)
All implement `BaseReconstructionPolicy` (`add_obs`, `reconstruct`, `reset`):
- `PointCloudReconstructionPolicy` — **default.** Accumulates a point cloud by back-projecting each depth image
  (torch), deduped via fine-voxel hashing (`point_min_dist`) so memory stays bounded — no nvblox-style VRAM leak.
  `reconstruct("point_cloud")` → points for the reward; `reconstruct("grid", ...)` → a uint8 truncated
  distance-field grid (`scipy.ndimage.distance_transform_edt`) for the NN observation; `reconstruct("mesh", ...)` →
  marching-cubes mesh for eval only.
- `Open3DReconstructionPolicy` — legacy. Open3D tensor `VoxelBlockGrid` TSDF, CPU or CUDA.
- `NvbloxReconstructionPolicy` — legacy, deprecated (unbounded VRAM growth / segfaults after many steps; latest
  nvblox v0.0.10 does not fix it). `nvblox_torch` GPU mapper.

Selected in config via `reconstruction_policy: pointcloud|open3d|nvblox` and
`reconstruction_metric: point_cloud_coverage|chamfer_distance|voxelwise_tsdf_error` — see `create_reconstruction_policy`
in `src/arm_nbv/utils/env_factory.py`. The default pairs `pointcloud` with `point_cloud_coverage` (fraction of
GT surface points with no reconstructed point within `coverage_distance`).

### Coverage reward (`compute_point_cloud_coverage_error`)
Both clouds are compared directly. Three things make it cheap and well-posed:
- **The GT mesh contains only observable surfaces.** `compute_gt_mesh` walks the **collision geoms only**
  (`COLLISION_GEOM_GROUP = 0`) — every body also carries a visual duplicate in group 1, so taking both would double
  every surface; this also drops the table legs, which exist as visual geoms only. It then emits the table's top face
  and each object's faces except the (near-)downward one it rests on (the 5 visible faces of a cube) — nothing else can
  ever be reconstructed, and including it would cap the coverage reward. Selection happens per geom *while the mesh is built*,
  from world-space normals that are outward **by construction**: `_geom_mesh` returns `(vertices, faces, face_normals)`
  built by `trimesh.creation`. Do **not** re-derive this by filtering the merged mesh — MuJoCo geom windings are
  arbitrary, so a post-hoc normal filter is meaningless and fails silently when the scene changes. Unreferenced
  vertices are dropped, so `bbox_center`/`bbox_size` cover the observable region only (the recon grid no longer wastes
  z-range below the table). The GT cloud is then sampled from it with `trimesh.sample.sample_surface` (area-weighted).
- **Incremental evaluation:** coverage is monotone and the cloud is append-only, so each step only queries the points
  added since the last call against a `cKDTree` built over the (fixed) GT cloud **once per episode**. Bit-identical to
  a full recomputation, but ~8x faster than rebuilding a tree over the growing cloud each step (which was 65% of step
  time). `gt_covered` / `gt_tree` / `n_recon_consumed` are the per-episode state, reset in `_reset_internal`;
  a shrinking cloud violates the append-only assumption and raises.
- `coverage_n_samples` (default 50k) sets the GT resolution; the info dict reports `n_covered`, `n_gt`,
  `n_recon_points`.

Because the GT mesh is open (undersides removed), `compute_static_env_sdf` (mesh2sdf) has no reliable inside/outside
and the legacy `voxelwise_tsdf_error` metric should not be trusted; the chamfer metric and eval rendering are fine.

A GPU path was evaluated and rejected: brute-force distances are O(Q·R) (~1.5e10 pairs late in an episode) and at
best match the KD-tree, while contending with `SubprocVecEnv` workers. The algorithmic fix above is the real win.

### Env/vec-env construction
`src/arm_nbv/utils/env_factory.py`: `create_env(config, ...)` builds one `Reconstruct3DGymWrapper` (optionally
`Monitor`-wrapped); `make_env_fn(...)` returns a thunk for SB3's `DummyVecEnv`/`SubprocVecEnv`. `scripts/train.py`
uses `SubprocVecEnv` when `n_envs > 1`, else `DummyVecEnv`; the eval env used during training is always a single
`DummyVecEnv` to bound memory.

### RL algorithm
`scripts/train.py` trains SB3 `SAC` (`config.algorithm="sac"`) — off-policy with a replay buffer for sample
efficiency. Recurrent policies don't combine cleanly with off-policy replay in SB3, so the observation is made
Markov by feeding the accumulated reconstruction directly (the `recon_grid` 3D grid), plus the pose-history
Transformer (a feed-forward net over a fixed observation buffer, fully replay-compatible). The `recon_grid` obs is
stored as `uint8` to keep replay RAM bounded (~32 KB/obs at 32³).

### Policy network (`src/arm_nbv/robot_policies/`)
SB3 `MultiInputPolicy` feature extractors, one per observation key, combined by `CombinedExtractor`
(`combined_extractor.py`) and driven from `scripts/train.py` based on `config.observations`:
- `CameraPoseExtractor` — MLP over the 7D pose.
- `CameraPoseHistoryExtractor` — small transformer over the pose history buffer.
- `Grid3DExtractor` — 3D-CNN over the `recon_grid` distance-field grid (the policy's direct 3D view of its
  reconstruction).
- `ImageExtractor` — CNN over `wrist_image` (optional live wrist RGB).
- `ScriptedPolicy` — non-learned baseline (table-edge policy), used by `scripts/eval_scripted.py`, not part of the
  SB3 policy stack.
- No extractor yet exists for `sdf_grid` (`scripts/train.py` prints a warning and skips it if configured).

### Config system (`src/arm_nbv/config.py`)
Single `TrainConfig` dataclass with defaults; `TrainConfig.load([...])` merges one or more YAML files in order
(later files override earlier keys), matching the `--config a.yaml b.yaml` CLI pattern. `reconstruction_policy` and
`reconstruction_metric` have no defaults and must be set by at least one config file. `configs/debug.yaml` and
`configs/demo.yaml` are override-only overlays meant to be layered on top of a base config, not run standalone.

### Test layout
`tests/reconstruct3d/` covers reconstruction math/rendering (chamfer distance, SDF computation, mesh rendering,
mesh export, full env dummy-step runs); `tests/reconstruction_policies/` covers the point-cloud policy + coverage
error and the Open3D TSDF generator directly.

## Environment / rendering notes
- `MUJOCO_GL=egl` is required for headless GPU rendering; see README "Known issues and fixes" for the full list of
  system-level fixes (libGL, Mesa/NVIDIA EGL vendor selection, `/dev/dri` permissions, robosuite's
  `IMAGE_CONVENTION` macro for image orientation with EGL).
- Long training runs are memory-hungry (host RSS especially via the SAC replay buffer). The point-cloud policy has
  bounded reconstruction memory by design (fine-voxel dedup + fixed grid), unlike the deprecated nvblox path.
  `config.benchmark=True` (single env only) exists specifically to profile where time/memory goes per step.
