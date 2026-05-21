import newton
import warp as wp
from axion.core.types import JointMode
from newton import Model


class AxionModelBuilder(newton.ModelBuilder):
    """
    A custom ModelBuilder for Axion that adds necessary attributes for PID control and joint modes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_axion_custom_attributes()

    def _add_axion_custom_attributes(self):
        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_dof_mode",
                frequency=Model.AttributeFrequency.JOINT_DOF,
                dtype=wp.int32,
                default=JointMode.NONE,
                assignment=Model.AttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="joint_compliance",
                frequency=Model.AttributeFrequency.JOINT,
                dtype=wp.float32,
                default=-1.0,
                assignment=Model.AttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="track_u_offset",
                frequency=Model.AttributeFrequency.JOINT,
                dtype=wp.float32,
                default=0.0,
                assignment=Model.AttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="is_track_joint",
                frequency=Model.AttributeFrequency.JOINT,
                dtype=wp.int32,
                default=0,
                assignment=Model.AttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="track_velocity",
                frequency=Model.AttributeFrequency.JOINT,
                dtype=wp.float32,
                default=0.0,
                assignment=Model.AttributeAssignment.CONTROL,
            )
        )

        # Anisotropic friction: per-shape body-local axis defining the friction
        # frame. Zero vector => isotropic (use shape_material_mu only). When set,
        # `shape_material_mu` is the coefficient along the projected axis and
        # `mu_perp` is the coefficient perpendicular to it in the tangent plane.
        # `mu_perp` < 0 is a sentinel meaning "same as shape_material_mu".
        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="friction_axis_local",
                frequency=Model.AttributeFrequency.SHAPE,
                dtype=wp.vec3,
                default=wp.vec3(0.0, 0.0, 0.0),
                assignment=Model.AttributeAssignment.MODEL,
            )
        )

        self.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="mu_perp",
                frequency=Model.AttributeFrequency.SHAPE,
                dtype=wp.float32,
                default=-1.0,
                assignment=Model.AttributeAssignment.MODEL,
            )
        )

    def add_track(
        self,
        parent_body: int,
        num_elements: int,
        element_radius: float,
        element_half_width: float,
        shape_config: newton.ModelBuilder.ShapeConfig,
        track_helper,
        track_center: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
        track_rotation: wp.quat = wp.quat_identity(),
        parent_world_xform: wp.transform = wp.transform_identity(),
        name_prefix: str = "track",
    ):
        """
        Adds a sequence of capsules constrained to a track path.

        Args:
            parent_body: The body index to attach the track elements to (e.g., base/world).
            num_elements: Number of elements to place on the track.
            element_radius: Radius of the capsule elements.
            element_half_width: Half-width (half-height along Z) of the capsule elements.
            shape_config: Configuration for the visual/collision shape.
            track_helper: An object with properties `total_len` and method `get_frame(u)`.
            track_center: Offset for the entire track system.
            track_rotation: Rotation for the entire track system.
            parent_world_xform: The initial world transform of the parent body.
                                Used to initialize track links at the correct world location.
            name_prefix: Prefix for the track link keys.
        """
        import numpy as np

        spacing = track_helper.total_len / num_elements

        # Update shape config to use negative collision group so tracks don't collide with themselves
        track_shape_config = shape_config.copy()
        track_shape_config.collision_group = -1

        # Transform for the track base
        X_track = wp.transform(track_center, track_rotation)

        created_joints = []
        for i in range(num_elements):
            u = i * spacing

            # Get track frame (assuming 2D track in XY plane for now)
            # track_helper.get_frame(u) returns pos (2D), tan (2D)
            pos_2d, tan_2d = track_helper.get_frame(u)

            # Convert to 3D local frame
            tangent = np.array([tan_2d[0], tan_2d[1], 0.0])
            normal = np.array([-tan_2d[1], tan_2d[0], 0.0])
            binormal = np.array([0.0, 0.0, 1.0])

            pos_local = np.array([pos_2d[0], pos_2d[1], 0.0])

            # Orientation matrix to quaternion
            # Frame: X=Tangent, Y=Normal, Z=Binormal
            rot_matrix = np.column_stack((tangent, normal, binormal))
            q_local = wp.quat_from_matrix(
                wp.matrix_from_cols(wp.vec3(tangent), wp.vec3(normal), wp.vec3(binormal))
            )

            # Compute world transform of the anchor point on the track
            X_anchor_local = wp.transform(wp.vec3(pos_local), q_local)

            # X_anchor_relative is the pose of the link relative to the parent body
            X_anchor_relative = wp.transform_multiply(X_track, X_anchor_local)

            # X_link_world is the initial global pose of the link
            X_link_world = wp.transform_multiply(parent_world_xform, X_anchor_relative)

            # Create the link body
            # We position it exactly at the anchor point initially
            link = self.add_link(
                label=f"{name_prefix}_link_{i}",
                mass=0.0,  # Kinematic / infinite mass effectively if fixed?
                # Actually, if it's attached via FIXED joint to parent, its mass matters less for statics,
                # but for dynamics, if parent is static, this is static.
                xform=X_link_world,
            )

            self.add_shape_capsule(
                body=link,
                radius=element_radius,
                half_height=element_half_width,
                cfg=track_shape_config,
            )

            # Add FIXED Joint
            # Connect parent to link.
            # parent_xform is the location of the joint on the parent body (track path).
            # child_xform is identity (joint is at the center of the link).

            joint_idx = self.add_joint(
                newton.JointType.FIXED,
                parent_body,
                link,
                parent_xform=X_anchor_relative,
                child_xform=wp.transform_identity(),
                custom_attributes={"track_u_offset": u, "is_track_joint": 1},
            )
            created_joints.append(joint_idx)

        return created_joints

    def finalize_replicated(
        self,
        num_worlds: int,
        gravity: float = -9.81,
        requires_grad: bool = False,
        global_builder: "newton.ModelBuilder | None" = None,
        **kwargs,
    ) -> newton.Model:
        """
        Creates a new newton.ModelBuilder, replicates the content of this builder into it
        for the specified number of worlds, and finalizes it to return the Model.

        If ``global_builder`` is provided, its contents are added once with
        ``shape_world = -1`` (Newton's "global" sentinel) before replication, so the
        resulting Model contains a single instance of those shapes that collides
        against shapes from every world via Newton's broadphase.
        """
        final_builder = newton.ModelBuilder(gravity=gravity)
        for k, v in kwargs.items():
            setattr(final_builder, k, v)
        if global_builder is not None:
            # current_world is -1 by default; add_builder copies entities with that
            # tag so they get shape_world=-1 in the final model.
            final_builder.add_builder(global_builder)
        final_builder.replicate(self, world_count=num_worlds)
        model = final_builder.finalize(requires_grad=requires_grad)
        self._backfill_empty_custom_attributes(final_builder, model, requires_grad=requires_grad)
        return model

    @staticmethod
    def _backfill_empty_custom_attributes(
        builder: newton.ModelBuilder, model: Model, requires_grad: bool
    ) -> None:
        """Ensure every registered MODEL custom attribute exists on the model.

        Why: Newton's ``ModelBuilder.finalize()`` skips custom-attribute arrays
        whose frequency count is 0 (see ``builder.py`` — ``if count == 0: continue``).
        Axion downstream code assumes registered attributes (e.g. ``joint_dof_mode``)
        always exist on the model, which fails for valid models that happen to have
        zero entities at a given frequency (e.g. a model containing only fixed
        joints has ``joint_dof_count == 0``).
        """
        for custom_attr in builder.custom_attributes.values():
            if custom_attr.assignment != Model.AttributeAssignment.MODEL:
                continue

            namespace = custom_attr.namespace
            name = custom_attr.name
            if namespace:
                ns_obj = getattr(model, namespace, None)
                if ns_obj is not None and hasattr(ns_obj, name):
                    continue
            elif hasattr(model, name):
                continue

            empty = custom_attr.build_array(0, device=model.device, requires_grad=requires_grad)
            model.add_attribute(name, empty, custom_attr.frequency, custom_attr.assignment, namespace)
