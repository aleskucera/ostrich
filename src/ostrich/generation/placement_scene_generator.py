"""Collision-aware scene generator.

`PlacementSceneGenerator` tracks every body it creates and AABB-checks each
placement against existing objects (and the ground). Use it when scenes
need physically-plausible non-overlapping starts: free-floating objects,
ground-resting objects, contact-touching pairs, and kinematic chains.

For random scattering or randomized articulated trees with no overlap
checks, see ``random_scene_generator.RandomSceneGenerator``.
"""
import math
import random
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import newton
import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# Geometric Utilities (module-private; not re-exported from the package)
# ---------------------------------------------------------------------------


class Geometry:
    @staticmethod
    def get_support(shape_type: str, params: Dict, direction: wp.vec3) -> wp.vec3:
        """
        Computes the support point of a shape in a given local direction.
        Returns the point on the surface furthest in that direction.
        """
        d = wp.normalize(direction)

        if shape_type == "box":
            # params: hx, hy, hz
            # support is corner matching signs of direction
            return wp.vec3(
                math.copysign(params["hx"], d[0]),
                math.copysign(params["hy"], d[1]),
                math.copysign(params["hz"], d[2]),
            )

        elif shape_type == "sphere":
            # params: radius
            return d * params["radius"]

        elif shape_type == "capsule" or shape_type == "cylinder":
            # params: radius, half_height. Axis is Z.
            # Cylinder support is slightly different from capsule, but for "touching"
            # we can approximate cylinder with capsule logic for the "rim" or faces.
            # Exact cylinder support:
            # if d.z > 0, top cap. if d.z < 0, bottom cap.
            # radial component direction

            r = params["radius"]
            h = params["half_height"]

            if shape_type == "capsule":
                # Sphere at top (0,0,h) and bottom (0,0,-h)
                # Support is support of one of these spheres
                # dot(center_top, d) vs dot(center_btm, d)
                # center_top is (0,0,h), center_btm is (0,0,-h)
                # dot is h*d.z vs -h*d.z

                sign_z = 1.0 if d[2] >= 0 else -1.0
                center = wp.vec3(0.0, 0.0, sign_z * h)
                return center + d * r

            else:  # cylinder
                # Top/Bottom circle check
                # Horizontal component
                rad_len = math.sqrt(d[0] * d[0] + d[1] * d[1])
                if rad_len > 1e-6:
                    radial = wp.vec3(d[0] / rad_len * r, d[1] / rad_len * r, 0.0)
                else:
                    radial = wp.vec3(0.0, 0.0, 0.0)

                sign_z = 1.0 if d[2] >= 0 else -1.0
                return radial + wp.vec3(0.0, 0.0, sign_z * h)

        return wp.vec3(0, 0, 0)

    @staticmethod
    def transform_point(xform: wp.transform, p: wp.vec3) -> wp.vec3:
        return wp.transform_point(xform, p)

    @staticmethod
    def rotate_vector(quat: wp.quat, v: wp.vec3) -> wp.vec3:
        return wp.quat_rotate(quat, v)

    @staticmethod
    def inverse_rotate_vector(quat: wp.quat, v: wp.vec3) -> wp.vec3:
        return wp.quat_rotate_inv(quat, v)

    @staticmethod
    def get_aabb(shape_type: str, params: Dict, xform: wp.transform) -> Tuple[wp.vec3, wp.vec3]:
        """Computes world-space AABB (min, max) for a shape."""
        # Get 8 corners for box, or extrema for others.
        # Simplification: Use a "local AABB" and transform all 8 corners

        local_min, local_max = wp.vec3(0.0), wp.vec3(0.0)

        if shape_type == "box":
            local_min = wp.vec3(-params["hx"], -params["hy"], -params["hz"])
            local_max = wp.vec3(params["hx"], params["hy"], params["hz"])
        elif shape_type == "sphere":
            r = params["radius"]
            local_min = wp.vec3(-r, -r, -r)
            local_max = wp.vec3(r, r, r)
        elif shape_type in ["capsule", "cylinder"]:
            r = params["radius"]
            h = params["half_height"]
            # Z-axis aligned
            local_min = wp.vec3(-r, -r, -h - (r if shape_type == "capsule" else 0))
            local_max = wp.vec3(r, r, h + (r if shape_type == "capsule" else 0))

        # Get all 8 corners of local AABB
        corners = []
        for x in [local_min[0], local_max[0]]:
            for y in [local_min[1], local_max[1]]:
                for z in [local_min[2], local_max[2]]:
                    corners.append(wp.vec3(x, y, z))

        # Transform corners
        world_corners = [wp.transform_point(xform, c) for c in corners]

        # Find min/max
        inf = float("inf")
        w_min = [inf, inf, inf]
        w_max = [-inf, -inf, -inf]

        for c in world_corners:
            for i in range(3):
                w_min[i] = min(w_min[i], c[i])
                w_max[i] = max(w_max[i], c[i])

        # Add a small margin for safety
        margin = 0.05
        return (
            wp.vec3(w_min[0] - margin, w_min[1] - margin, w_min[2] - margin),
            wp.vec3(w_max[0] + margin, w_max[1] + margin, w_max[2] + margin),
        )

    @staticmethod
    def aabb_overlap(aabb1, aabb2):
        min1, max1 = aabb1
        min2, max2 = aabb2
        return (
            min1[0] < max2[0]
            and max1[0] > min2[0]
            and min1[1] < max2[1]
            and max1[1] > min2[1]
            and min1[2] < max2[2]
            and max1[2] > min2[2]
        )


# ---------------------------------------------------------------------------
# Scene Generator
# ---------------------------------------------------------------------------


@dataclass
class GeneratedObject:
    index: int  # Index in our tracking list
    body_id: int
    shape_type: str
    params: Dict
    xform: wp.transform
    aabb: Tuple[wp.vec3, wp.vec3]


class PlacementSceneGenerator:
    def __init__(self, builder: newton.ModelBuilder, seed=42):
        """
        Initialize the scene generator.

        Args:
            builder: The newton.ModelBuilder to add objects to.
            seed: Random seed for reproducibility.
        """
        self.builder = builder
        self.objects: List[GeneratedObject] = []

        random.seed(seed)
        np.random.seed(seed)

        # Configuration ranges
        self.size_min = 0.2
        self.size_max = 0.6
        self.area_bounds = ((-5, -5, 0), (5, 5, 5))  # min, max

        # Ground plane is assumed to be added by the user of this class or we can add it here if needed.
        # But to be more flexible (as a library), we operate on the passed builder.

    def _random_params(self, shape_type):
        if shape_type == "box":
            return {
                "hx": random.uniform(self.size_min, self.size_max),
                "hy": random.uniform(self.size_min, self.size_max),
                "hz": random.uniform(self.size_min, self.size_max),
            }
        elif shape_type == "sphere":
            return {"radius": random.uniform(self.size_min, self.size_max)}
        elif shape_type in ["capsule", "cylinder"]:
            return {
                "radius": random.uniform(self.size_min, self.size_max),
                "half_height": random.uniform(self.size_min, self.size_max),
            }
        return {}

    def _random_xform(self, pos_bounds=None):
        if pos_bounds:
            p_min, p_max = pos_bounds
            pos = wp.vec3(
                random.uniform(p_min[0], p_max[0]),
                random.uniform(p_min[1], p_max[1]),
                random.uniform(p_min[2], p_max[2]),
            )
        else:
            pos = wp.vec3(0.0)

        # Random rotation
        axis_vec = [random.uniform(-1, 1) for _ in range(3)]
        if all(v == 0 for v in axis_vec):
            axis_vec = [0, 0, 1]
        axis = wp.normalize(wp.vec3(*axis_vec))
        angle = random.uniform(0, 2 * math.pi)
        rot = wp.quat_from_axis_angle(axis, angle)

        return wp.transform(pos, rot)

    def _add_to_builder(self, shape_type, params, xform, mass=1.0) -> int:
        body = self.builder.add_body(xform=xform, mass=mass)

        if shape_type == "box":
            self.builder.add_shape_box(body, hx=params["hx"], hy=params["hy"], hz=params["hz"])
        elif shape_type == "sphere":
            self.builder.add_shape_sphere(body, radius=params["radius"])
        elif shape_type == "capsule":
            self.builder.add_shape_capsule(
                body, radius=params["radius"], half_height=params["half_height"]
            )
        elif shape_type == "cylinder":
            self.builder.add_shape_cylinder(
                body, radius=params["radius"], half_height=params["half_height"]
            )

        return body

    def _add_link_to_builder(self, shape_type, params, xform, mass=1.0) -> int:
        body = self.builder.add_link(xform=xform, mass=mass)

        if shape_type == "box":
            self.builder.add_shape_box(body, hx=params["hx"], hy=params["hy"], hz=params["hz"])
        elif shape_type == "sphere":
            self.builder.add_shape_sphere(body, radius=params["radius"])
        elif shape_type == "capsule":
            self.builder.add_shape_capsule(
                body, radius=params["radius"], half_height=params["half_height"]
            )
        elif shape_type == "cylinder":
            self.builder.add_shape_cylinder(
                body, radius=params["radius"], half_height=params["half_height"]
            )

        return body

    def check_collisions(self, test_aabb, ignore_indices=None):
        if ignore_indices is None:
            ignore_indices = []

        for obj in self.objects:
            if obj.index in ignore_indices:
                continue
            if Geometry.aabb_overlap(test_aabb, obj.aabb):
                return True
        return False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate_random_free(self, shape_type=None, max_attempts=100) -> Optional[int]:
        """Generates a random object that doesn't touch anything."""
        if shape_type is None:
            shape_type = random.choice(["box", "sphere", "capsule", "cylinder"])

        for _ in range(max_attempts):
            params = self._random_params(shape_type)
            xform = self._random_xform(self.area_bounds)

            # Ensure Z is high enough to not hit ground (Z=0)
            aabb = Geometry.get_aabb(shape_type, params, xform)
            if aabb[0][2] < 0.05:  # Hits ground
                continue

            if not self.check_collisions(aabb):
                # Success
                body_id = self._add_to_builder(shape_type, params, xform)
                obj = GeneratedObject(len(self.objects), body_id, shape_type, params, xform, aabb)
                self.objects.append(obj)
                return obj.index

        print("Failed to place free object")
        return None

    def generate_random_ground_touching(
        self, shape_type=None, max_attempts=100, margin=1e-3
    ) -> Optional[int]:
        """Generates a random object that touches the ground."""
        if shape_type is None:
            shape_type = random.choice(["box", "sphere", "capsule", "cylinder"])

        for _ in range(max_attempts):
            params = self._random_params(shape_type)

            # 1. Generate random rotation
            dummy_xform = self._random_xform()  # pos=0
            rot = dummy_xform.q

            # 2. Find lowest point in local space relative to center (0,0,0)
            # We want point furthest in world -Z direction.
            # Local direction is InvRot * (0,0,-1)
            down_world = wp.vec3(0, 0, -1)
            down_local = Geometry.inverse_rotate_vector(rot, down_world)

            support_local = Geometry.get_support(shape_type, params, down_local)

            # 3. Calculate Z offset
            # Lowest point in world: Rot * support_local
            lowest_world = Geometry.rotate_vector(rot, support_local)
            z_offset = -lowest_world[2] + margin

            # 4. Pick random XY
            x = random.uniform(self.area_bounds[0][0], self.area_bounds[1][0])
            y = random.uniform(self.area_bounds[0][1], self.area_bounds[1][1])

            pos = wp.vec3(x, y, z_offset)
            xform = wp.transform(pos, rot)

            aabb = Geometry.get_aabb(shape_type, params, xform)
            if not self.check_collisions(aabb):
                body_id = self._add_to_builder(shape_type, params, xform)
                obj = GeneratedObject(len(self.objects), body_id, shape_type, params, xform, aabb)
                self.objects.append(obj)
                return obj.index

        print("Failed to place grounded object")
        return None

    def generate_random_touching(
        self, target_idx: int, shape_type=None, max_attempts=100, margin=1e-3
    ) -> Optional[int]:
        """Generates a random object that touches an existing object."""
        if target_idx >= len(self.objects):
            return None
        target = self.objects[target_idx]

        if shape_type is None:
            shape_type = random.choice(["box", "sphere", "capsule", "cylinder"])

        for _ in range(max_attempts):
            params = self._random_params(shape_type)

            # 1. Pick random direction for contact normal (from target to new obj)
            n_vec = [random.gauss(0, 1) for _ in range(3)]
            normal = wp.normalize(wp.vec3(*n_vec))

            # 2. Find contact point on Target in World Space
            # normal in target local space
            n_local_target = Geometry.inverse_rotate_vector(target.xform.q, normal)
            p_support_target = Geometry.get_support(
                target.shape_type, target.params, n_local_target
            )
            p_contact_world = Geometry.transform_point(target.xform, p_support_target)

            # 3. Orient new object (Random or aligned? Random is better)
            rot = self._random_xform().q

            # 4. Find support of New Object in direction -normal
            n_local_new = Geometry.inverse_rotate_vector(rot, -normal)
            p_support_new = Geometry.get_support(shape_type, params, n_local_new)

            # 5. Position new object so its support point is at p_contact_world
            # p_contact_world = pos_new + Rot * p_support_new
            # pos_new = p_contact_world - Rot * p_support_new
            p_support_new_world_offset = Geometry.rotate_vector(rot, p_support_new)

            # Apply margin along the normal direction
            pos_new = p_contact_world - p_support_new_world_offset + normal * margin

            xform = wp.transform(pos_new, rot)

            # 6. Check collisions (ignore target)
            aabb = Geometry.get_aabb(shape_type, params, xform)

            # Also check ground.
            # NOTE: We use 0.0 + margin/2 as safety to ensure it doesn't clip through floor
            # if the target is sitting low.
            if aabb[0][2] < 0.0:
                continue

            if not self.check_collisions(aabb, ignore_indices=[target_idx]):
                body_id = self._add_to_builder(shape_type, params, xform)
                obj = GeneratedObject(len(self.objects), body_id, shape_type, params, xform, aabb)
                self.objects.append(obj)
                return obj.index

        print(f"Failed to place touching object on {target_idx}")
        return None

    def generate_chain(
        self,
        length=5,
        start_pos=(0, 0, 2),
        shape_type="box",
        joint_type="revolute",
        root_type="fixed",
        max_attempts=100,
    ) -> List[int]:
        """
        Generates a chain of connected bodies.

        Args:
            length: Number of links in the chain.
            start_pos: Starting position of the first link.
            shape_type: Type of shape for links ("box", "sphere", "capsule", "cylinder").
            joint_type: Type of joint connecting links ("revolute", "fixed", "prismatic", "ball").
            root_type: Type of root connection ("fixed" to world, "free" floating, "revolute" to world).
            max_attempts: Max attempts to place each subsequent link.

        Returns:
            List of body indices in the chain.
        """
        chain_indices = []
        joint_indices = []

        # 1. Place Root Link (Link 0)
        params = self._random_params(shape_type)
        pos = wp.vec3(*start_pos)
        rot = wp.quat_identity()
        xform = wp.transform(pos, rot)

        aabb = Geometry.get_aabb(shape_type, params, xform)
        if self.check_collisions(aabb):
            print("Chain root collision")
            return []

        # Use add_link for chain components
        body_id = self._add_link_to_builder(shape_type, params, xform)
        obj = GeneratedObject(len(self.objects), body_id, shape_type, params, xform, aabb)
        self.objects.append(obj)
        chain_indices.append(obj.index)

        # Add Root Joint
        root_joint_idx = -1
        if root_type == "fixed":
            root_joint_idx = self.builder.add_joint_fixed(
                parent=-1, child=body_id, parent_xform=xform, child_xform=wp.transform()
            )
        elif root_type == "revolute":
            root_joint_idx = self.builder.add_joint_revolute(
                parent=-1,
                child=body_id,
                parent_xform=xform,
                child_xform=wp.transform(),
                axis=wp.vec3(0, 1, 0),
            )
        elif root_type == "free":
            root_joint_idx = self.builder.add_joint_free(child=body_id)

        if root_joint_idx != -1:
            joint_indices.append(root_joint_idx)

        prev_obj = obj

        # 2. Place subsequent links
        for i in range(1, length):
            params = self._random_params(shape_type)

            success = False
            for _ in range(max_attempts):
                # Pick a connection direction on previous object
                n_vec = [random.gauss(0, 1) for _ in range(3)]
                dir_from_prev = wp.normalize(wp.vec3(*n_vec))

                # Connection point on Prev (World)
                dir_local_prev = Geometry.inverse_rotate_vector(prev_obj.xform.q, dir_from_prev)
                support_prev = Geometry.get_support(
                    prev_obj.shape_type, prev_obj.params, dir_local_prev
                )
                anchor_world = Geometry.transform_point(prev_obj.xform, support_prev)

                # New Object Orientation
                rot_new = self._random_xform().q

                # Connection point on New (Local)
                dir_local_new = Geometry.inverse_rotate_vector(rot_new, -dir_from_prev)
                support_new = Geometry.get_support(shape_type, params, dir_local_new)

                # Calculate Position with gap
                support_new_world_offset = Geometry.rotate_vector(rot_new, support_new)
                gap = 0.1
                pos_new = anchor_world - support_new_world_offset + dir_from_prev * gap

                xform_new = wp.transform(pos_new, rot_new)

                # Check collision (ignore previous link)
                aabb_new = Geometry.get_aabb(shape_type, params, xform_new)

                if aabb_new[0][2] < 0.05:  # Ground check
                    continue

                if not self.check_collisions(aabb_new, ignore_indices=[prev_obj.index]):
                    # Valid placement -> Add Link
                    body_id = self._add_link_to_builder(shape_type, params, xform_new)
                    new_obj = GeneratedObject(
                        len(self.objects), body_id, shape_type, params, xform_new, aabb_new
                    )
                    self.objects.append(new_obj)
                    chain_indices.append(new_obj.index)

                    # Add Joint
                    # Parent Frame (Prev): anchor_world in Prev local
                    p_parent_local = Geometry.inverse_rotate_vector(
                        prev_obj.xform.q, anchor_world - prev_obj.xform.p
                    )
                    parent_xform = wp.transform(p_parent_local, wp.quat_identity())

                    # Child Frame (New): anchor_world in New local
                    p_child_local = Geometry.inverse_rotate_vector(rot_new, anchor_world - pos_new)
                    child_xform = wp.transform(p_child_local, wp.quat_identity())

                    axis = wp.vec3(0, 1, 0)  # Local Y axis

                    joint_idx = -1
                    if joint_type == "revolute":
                        joint_idx = self.builder.add_joint_revolute(
                            parent=prev_obj.body_id,
                            child=new_obj.body_id,
                            parent_xform=parent_xform,
                            child_xform=child_xform,
                            axis=axis,
                            # limit_lower=-1.5,
                            # limit_upper=1.5
                        )
                    elif joint_type == "fixed":
                        joint_idx = self.builder.add_joint_fixed(
                            parent=prev_obj.body_id,
                            child=new_obj.body_id,
                            parent_xform=parent_xform,
                            child_xform=child_xform,
                        )
                    elif joint_type == "ball":
                        joint_idx = self.builder.add_joint_ball(
                            parent=prev_obj.body_id,
                            child=new_obj.body_id,
                            parent_xform=parent_xform,
                            child_xform=child_xform,
                        )
                    elif joint_type == "prismatic":
                        joint_idx = self.builder.add_joint_prismatic(
                            parent=prev_obj.body_id,
                            child=new_obj.body_id,
                            parent_xform=parent_xform,
                            child_xform=child_xform,
                            axis=wp.vec3(1, 0, 0),
                            limit_lower=-0.5,
                            limit_upper=0.5,
                        )

                    if joint_idx != -1:
                        joint_indices.append(joint_idx)

                    prev_obj = new_obj
                    success = True
                    break

            if not success:
                print(f"Failed to extend chain at link {i}")
                break

        # Create Articulation
        if joint_indices:
            self.builder.add_articulation(joint_indices)

        return chain_indices

    def generate_joint_pair(
        self, start_pos=(0, 0, 2), shape_type="box", joint_type="revolute"
    ) -> List[int]:
        """Generates two bodies connected by a joint."""
        return self.generate_chain(
            length=2, start_pos=start_pos, shape_type=shape_type, joint_type=joint_type
        )

