from typing import Any

import warp as wp


@wp.kernel
def batched_argmin_kernel(
    values: wp.array(dtype=wp.float32, ndim=2),
    # Outputs
    min_indices: wp.array(dtype=wp.int32),
):
    """Finds the index of the minimum value along the entire batch dimension (axis 0)."""
    world_idx = wp.tid()
    count = values.shape[0]

    min_idx = wp.int32(0)
    min_value = values[0, world_idx]

    for i in range(1, count):
        value = values[i, world_idx]
        if value < min_value:
            min_idx = wp.int32(i)
            min_value = value

    min_indices[world_idx] = min_idx


@wp.kernel
def batched_argmin_dynamic_kernel(
    values: wp.array(dtype=wp.float32, ndim=2),
    start_idx: wp.int32,
    end_idx_arr: wp.array(dtype=wp.int32),
    # Outputs
    min_indices: wp.array(dtype=wp.int32),
):
    """
    Finds the index of the minimum value along axis 0,
    bounded by [start_idx, end_idx_arr[0]).
    """
    world_idx = wp.tid()
    end_idx = end_idx_arr[0]

    # Guard against empty or invalid ranges
    if end_idx <= 0:
        min_indices[world_idx] = wp.int32(0)
        return

    # If the end index is at or behind the start index, default to the last valid index
    if end_idx <= start_idx:
        min_indices[world_idx] = end_idx - wp.int32(1)
        return

    min_idx = wp.int32(start_idx)
    min_value = values[min_idx, world_idx]

    for i in range(start_idx + 1, end_idx):
        value = values[i, world_idx]
        if value < min_value:
            min_idx = wp.int32(i)
            min_value = value

    min_indices[world_idx] = min_idx


@wp.kernel
def gather_2d_kernel(
    source: wp.array(dtype=Any, ndim=3),
    indices: wp.array(dtype=wp.int32),
    # Outputs
    dest: wp.array(dtype=Any, ndim=2),
):
    """Gathers 2D slices from a 3D buffer based on a 1D array of batch indices."""
    world_idx, element_idx = wp.tid()
    dest[world_idx, element_idx] = source[indices[world_idx], world_idx, element_idx]


@wp.kernel
def gather_1d_kernel(
    source: wp.array(dtype=Any, ndim=2),
    indices: wp.array(dtype=wp.int32),
    # Outputs
    dest: wp.array(dtype=Any, ndim=1),
):
    """Gathers 1D elements from a 2D buffer based on a 1D array of batch indices."""
    world_idx = wp.tid()
    dest[world_idx] = source[indices[world_idx], world_idx]
