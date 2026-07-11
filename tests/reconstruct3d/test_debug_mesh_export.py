"""
Debug script to verify mesh export and coordinate transformations.
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def debug_mesh_export():
    """Debug mesh extraction to verify bodies and transformations."""

    # Create environment with default Panda controller (OSC_POSE)
    controller_config = load_composite_controller_config(
        robot="Panda",
    )

    env = suite.make(
        env_name="Reconstruct3D",
        robots="Panda",
        controller_configs=controller_config,
    )

    # Reset environment
    env.reset()

    print("=" * 60)
    print("DEBUG: Mesh Export Analysis")
    print("=" * 60)

    # Print all body names in the scene
    print("\nAll bodies in scene:")
    for body_id in range(env.sim.model.nbody):
        body_name = env.sim.model.body(body_id).name
        print(f"  {body_id}: {body_name}")

    # Print object body names we're looking for
    print("\nObject bodies we're looking for:")
    object_body_names = set()
    object_body_names.add("table")

    for obj in env.primitives_on_table:
        object_body_names.add(obj.root_body)

    for name in object_body_names:
        print(f"  {name}")

    # Check each geom
    print(f"\nAll geoms in scene (total: {env.sim.model.ngeom}):")
    for geom_id in range(env.sim.model.ngeom):
        body_id = env.sim.model.geom_bodyid[geom_id]
        body_name = env.sim.model.body(body_id).name
        geom_type = env.sim.model.geom_type[geom_id]
        geom_pos = env.sim.data.geom_xpos[geom_id]
        geom_group = env.sim.model.geom_group[geom_id]
        geom_contype = env.sim.model.geom_contype[geom_id]
        geom_conaffinity = env.sim.model.geom_conaffinity[geom_id]

        # Check if included
        included = body_name in object_body_names

        print(
            f"  {geom_id}: body={body_name:30s} type={geom_type} group={geom_group} contype={geom_contype} geom_conaffinity={geom_conaffinity} pos={geom_pos} {'[INCLUDED]' if included else ''}"
        )

    # Extract mesh and analyze
    print("\n" + "=" * 60)
    print("Extracting mesh...")
    # Export only collision geoms (group 1+)
    vertices, faces = env.compute_static_env_mesh(geom_groups=[1])

    print(f"\nExtracted mesh statistics:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    print(f"  Vertex bounds:")
    print(f"    X: [{vertices[:, 0].min():.4f}, {vertices[:, 0].max():.4f}]")
    print(f"    Y: [{vertices[:, 1].min():.4f}, {vertices[:, 1].max():.4f}]")
    print(f"    Z: [{vertices[:, 2].min():.4f}, {vertices[:, 2].max():.4f}]")

    # Compare with camera position to check consistency
    print("\nCamera positions for reference:")
    for cam_name in ["frontview", "birdview", "sideview", "robot0_eye_in_hand"]:
        try:
            cam_id = env.sim.model.camera_name2id(cam_name)
            cam_pos = env.sim.data.cam_xpos[cam_id]
            print(f"  {cam_name}: {cam_pos}")
        except:
            pass

    # Check table specifically
    print("\nTable geom details:")
    for geom_id in range(env.sim.model.ngeom):
        body_id = env.sim.model.geom_bodyid[geom_id]
        body_name = env.sim.model.body(body_id).name
        if "table" in body_name.lower():
            geom_pos = env.sim.data.geom_xpos[geom_id]
            geom_size = env.sim.model.geom_size[geom_id]
            print(f"  Body: {body_name}")
            print(f"    Position: {geom_pos}")
            print(f"    Size: {geom_size}")

    env.close()
    print("\nDebug complete!")


if __name__ == "__main__":
    debug_mesh_export()
