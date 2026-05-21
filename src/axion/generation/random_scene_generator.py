"""Careless random scene generator.

`RandomSceneGenerator` scatters bodies and builds randomized articulated
trees with no overlap checks. Per-call bounds + density let it generate
diverse, dense, and potentially-interpenetrating starts — useful for
training data, stress tests, or any scenario where physical plausibility
on frame zero is not required.

For collision-aware placement (free / ground-resting / touching / chains),
see ``placement_scene_generator.PlacementSceneGenerator``.
"""
import math
import random

import newton
import numpy as np
import warp as wp


class RandomSceneGenerator:
    def __init__(self, builder: newton.ModelBuilder, seed=42):
        """
        Initialize the scene generator.

        Args:
            builder: The newton.ModelBuilder to add objects to.
            seed: Random seed for reproducibility.
        """
        self.builder = builder

        random.seed(seed)
        np.random.seed(seed)

    def _random_params(self, shape_type: str, size_bounds: tuple):
        if shape_type == "box":
            return {
                "hx": random.uniform(size_bounds[0], size_bounds[1]),
                "hy": random.uniform(size_bounds[0], size_bounds[1]),
                "hz": random.uniform(size_bounds[0], size_bounds[1]),
            }
        elif shape_type == "sphere":
            return {"radius": random.uniform(size_bounds[0], size_bounds[1])}
        elif shape_type == "capsule":
            return {
                "radius": random.uniform(size_bounds[0], size_bounds[1]),
                "half_height": random.uniform(size_bounds[0], size_bounds[1]),
            }
        return {}

    def _random_xform(self, pos_bounds: tuple):
        p_min, p_max = pos_bounds
        pos = wp.vec3(
            random.uniform(p_min[0], p_max[0]),
            random.uniform(p_min[1], p_max[1]),
            random.uniform(p_min[2], p_max[2]),
        )

        # Random rotation
        axis_vec = [random.uniform(-1, 1) for _ in range(3)]
        if all(v == 0 for v in axis_vec):
            axis_vec = [0, 0, 1]
        axis = wp.normalize(wp.vec3(*axis_vec))
        angle = random.uniform(0, 2 * math.pi)
        rot = wp.quat_from_axis_angle(axis, angle)

        return wp.transform(pos, rot)

    def _random_density(self, density_bounds: tuple):
        return random.uniform(density_bounds[0], density_bounds[1])

    def _add_to_builder(
        self, shape_type: str, params: dict, xform: wp.transform, density: float
    ) -> int:
        body = self.builder.add_body(xform=xform)
        cfg = self.builder.ShapeConfig(density=density)
        if shape_type == "box":
            self.builder.add_shape_box(
                body, hx=params["hx"], hy=params["hy"], hz=params["hz"], cfg=cfg
            )
        elif shape_type == "sphere":
            self.builder.add_shape_sphere(body, radius=params["radius"], cfg=cfg)
        elif shape_type == "capsule":
            self.builder.add_shape_capsule(
                body, radius=params["radius"], half_height=params["half_height"], cfg=cfg
            )

        return body

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate_random_object(
        self,
        pos_bounds: tuple,
        density_bounds: tuple,
        size_bounds: tuple,
        shape_type: str = None,
    ) -> int:
        """Generates a random object that doesn't touch anything."""
        if shape_type is None:
            shape_type = random.choice(["box", "sphere", "capsule"])

        params = self._random_params(shape_type, size_bounds)
        xform = self._random_xform(pos_bounds)
        density = self._random_density(density_bounds)
        self._add_to_builder(shape_type, params, xform, density)

    def _add_link_to_builder(
        self, shape_type: str, params: dict, xform: wp.transform, density: float
    ) -> int:
        """Helper to add links (required for joints) instead of standard bodies."""
        body = self.builder.add_link(xform=xform)
        cfg = self.builder.ShapeConfig(density=density)
        if shape_type == "box":
            self.builder.add_shape_box(
                body, hx=params["hx"], hy=params["hy"], hz=params["hz"], cfg=cfg
            )
        elif shape_type == "sphere":
            self.builder.add_shape_sphere(body, radius=params["radius"], cfg=cfg)
        elif shape_type == "capsule":
            self.builder.add_shape_capsule(
                body, radius=params["radius"], half_height=params["half_height"], cfg=cfg
            )
        return body

    def _random_quat(self) -> wp.quat:
        """Generates a random quaternion for orientations."""
        axis_vec = [random.uniform(-1, 1) for _ in range(3)]
        if all(v == 0 for v in axis_vec):
            axis_vec = [0, 0, 1]
        axis = wp.normalize(wp.vec3(*axis_vec))
        angle = random.uniform(0, 2 * math.pi)
        return wp.quat_from_axis_angle(axis, angle)

    def generate_chaotic_tree(
        self,
        num_objects: int,
        pos_bounds: tuple,
        density_bounds: tuple,
        size_bounds: tuple,
        joint_types: list[str] = ["revolute", "ball", "fixed"],
    ) -> list[int]:
        """Option 3: Generates a dense, random tree of interconnected objects.
        Guarantees a valid articulation structure (no multiple parents/loops).
        """
        link_indices = []
        joint_indices = []
        object_data = []

        # 1. Spawn a dense cloud of random objects
        for i in range(num_objects):
            shape_type = random.choice(["box", "sphere", "capsule"])
            params = self._random_params(shape_type, size_bounds)
            density = self._random_density(density_bounds)

            # Spawn randomly within the bounds
            pos = wp.vec3(
                random.uniform(pos_bounds[0][0], pos_bounds[1][0]),
                random.uniform(pos_bounds[0][1], pos_bounds[1][1]),
                random.uniform(pos_bounds[0][2], pos_bounds[1][2]),
            )
            rot = self._random_quat()
            xform = wp.transform(pos, rot)

            link_id = self._add_link_to_builder(shape_type, params, xform, density)
            link_indices.append(link_id)

            # Store data for joint creation
            object_data.append({"id": link_id, "pos": pos, "rot": rot})

        # 2. Build a Random Spanning Tree
        # Start with the first object as the "root" of the tangled web
        connected_indices = [0]
        unconnected_indices = list(range(1, num_objects))

        # Keep connecting random unattached objects to random already-attached objects
        while unconnected_indices:
            # Pick a random valid parent from the cluster
            parent_idx = random.choice(connected_indices)

            # Pick a random child to pull into the cluster
            child_idx = random.choice(unconnected_indices)

            obj_parent = object_data[parent_idx]
            obj_child = object_data[child_idx]

            # Place the joint anchor exactly halfway between the two scattered objects.
            anchor_world = (obj_parent["pos"] + obj_child["pos"]) * 0.5

            # Calculate local transforms relative to the anchor
            p_parent_local = wp.quat_rotate_inv(obj_parent["rot"], anchor_world - obj_parent["pos"])
            parent_xform = wp.transform(p_parent_local, wp.quat_identity())

            p_child_local = wp.quat_rotate_inv(obj_child["rot"], anchor_world - obj_child["pos"])
            child_xform = wp.transform(p_child_local, wp.quat_identity())

            # Create the joint
            joint_type = random.choice(joint_types)
            j_idx = -1

            if joint_type == "revolute":
                axis = wp.normalize(
                    wp.vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
                )
                j_idx = self.builder.add_joint_revolute(
                    parent=obj_parent["id"],
                    child=obj_child["id"],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    axis=axis,
                )
            elif joint_type == "ball":
                j_idx = self.builder.add_joint_ball(
                    parent=obj_parent["id"],
                    child=obj_child["id"],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                )
            elif joint_type == "fixed":
                j_idx = self.builder.add_joint_fixed(
                    parent=obj_parent["id"],
                    child=obj_child["id"],
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                )

            if j_idx != -1:
                joint_indices.append(j_idx)

            # Mark the child as connected so it can now become a parent to future objects
            unconnected_indices.remove(child_idx)
            connected_indices.append(child_idx)

        # 3. Add all joints to the articulation
        if joint_indices:
            self.builder.add_articulation(joint_indices)

        return link_indices

    def generate_random_velocities(self, lin_vel_bounds: tuple, ang_vel_bounds: tuple):
        pass
