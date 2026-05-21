import warp as wp
from newton import Contacts
from newton import Model

from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


@wp.func
def _shape_to_local_idx(
    shape_idx: wp.int32,
    num_globals: wp.int32,
    num_shapes_per_world: wp.int32,
) -> wp.int32:
    """Map Newton's flat shape index to the column of AxionModel's (W, S+G)
    per-world layout. Per-world shapes go to columns [0, S); globals to
    columns [S, S+G)."""
    if shape_idx < num_globals:
        # Global shape: column num_shapes_per_world + global_idx
        return num_shapes_per_world + shape_idx
    return (shape_idx - num_globals) % num_shapes_per_world


@wp.kernel
def batch_contact_data_kernel(
    shape_world: wp.array(dtype=wp.int32),
    num_globals: wp.int32,
    num_shapes_per_world: wp.int32,
    # Contact info
    contact_count: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_thickness0: wp.array(dtype=wp.float32),
    contact_thickness1: wp.array(dtype=wp.float32),
    # OUTPUT: This array holds the assigned column index for every contact
    batched_contact_count: wp.array(dtype=wp.int32),
    batched_contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    batched_contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    batched_contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    batched_contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    batched_contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    batched_contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    batched_contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
):
    contact_idx = wp.tid()

    shape_0 = contact_shape0[contact_idx]
    shape_1 = contact_shape1[contact_idx]

    if contact_idx >= contact_count[0] or shape_0 == shape_1:
        return

    # Pick world_idx from the per-world side (globals have shape_world == -1).
    # If both sides are global the contact has no world to live in; drop it.
    w0 = wp.int32(-1)
    w1 = wp.int32(-1)
    if shape_0 >= 0:
        w0 = shape_world[shape_0]
    if shape_1 >= 0:
        w1 = shape_world[shape_1]
    world_idx = wp.int32(-1)
    if w0 >= 0:
        world_idx = w0
    elif w1 >= 0:
        world_idx = w1
    else:
        return

    slot = wp.atomic_add(batched_contact_count, world_idx, 1)

    if slot >= batched_contact_point0.shape[1]:
        return

    batched_contact_point0[world_idx, slot] = contact_point0[contact_idx]
    batched_contact_point1[world_idx, slot] = contact_point1[contact_idx]
    # Newton (since PR #2069) emits rigid_contact_normal pointing from shape0
    # toward shape1 (A-to-B). Axion's contact/friction kernels were written for
    # the older B-to-A convention, so we flip here at the boundary.
    batched_contact_normal[world_idx, slot] = -contact_normal[contact_idx]
    batched_contact_thickness0[world_idx, slot] = contact_thickness0[contact_idx]
    batched_contact_thickness1[world_idx, slot] = contact_thickness1[contact_idx]

    if shape_0 >= 0:
        batched_contact_shape0[world_idx, slot] = _shape_to_local_idx(
            shape_0, num_globals, num_shapes_per_world
        )
    else:
        batched_contact_shape0[world_idx, slot] = shape_0

    if shape_1 >= 0:
        batched_contact_shape1[world_idx, slot] = _shape_to_local_idx(
            shape_1, num_globals, num_shapes_per_world
        )
    else:
        batched_contact_shape1[world_idx, slot] = shape_1


class AxionContacts:
    def __init__(self, model: Model, max_contacts_per_world: int) -> None:
        # Newton's flat layout: [globals | world_0 | world_1 | ...].
        # shape_world_start[0] is the count of globals (where world 0 starts).
        shape_starts_np = model.shape_world_start.numpy()
        self.num_global_shapes = int(shape_starts_np[0])
        per_world_total = int(shape_starts_np[-1]) - self.num_global_shapes
        assert per_world_total % model.world_count == 0, (
            f"Per-world shape count {per_world_total} not divisible by "
            f"world_count {model.world_count}; worlds must be uniform."
        )

        self.model = model
        self.device = model.device
        self.num_worlds = model.world_count
        self.num_shapes_per_world = per_world_total // model.world_count
        self.max_contacts = max_contacts_per_world

        with wp.ScopedDevice(self.device):
            self.contact_count = wp.zeros(model.world_count, dtype=wp.int32)
            self.contact_point0 = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.vec3
            )
            self.contact_point1 = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.vec3
            )
            self.contact_normal = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.vec3
            )
            self.contact_shape0 = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.int32
            )
            self.contact_shape1 = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.int32
            )
            self.contact_thickness0 = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.float32
            )
            self.contact_thickness1 = wp.zeros(
                (model.world_count, max_contacts_per_world), dtype=wp.float32
            )

    def load_contact_data(
        self, contacts: Contacts, axion_model: AxionModel, data: EngineData, dims: EngineDimensions
    ):
        self.contact_count.zero_()

        wp.launch(
            kernel=batch_contact_data_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                self.model.shape_world,
                self.num_global_shapes,
                self.num_shapes_per_world,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
            ],
            outputs=[
                self.contact_count,
                self.contact_point0,
                self.contact_point1,
                self.contact_normal,
                self.contact_shape0,
                self.contact_shape1,
                self.contact_thickness0,
                self.contact_thickness1,
            ],
            device=self.device,
        )
