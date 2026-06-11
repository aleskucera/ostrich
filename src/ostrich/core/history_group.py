from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import warp as wp

# Global cache to store compiled kernels by (ndim, dtype)
_SAVE_KERNEL_CACHE = {}


def create_save_to_buffer_kernel(arr: wp.array):
    x_ndim = len(arr.shape)
    dtype = arr.dtype

    if x_ndim == 1:

        @wp.kernel
        def save_to_buffer_kernel_1d(
            x: wp.array(dtype=dtype, ndim=1),
            iter_idx: wp.array(dtype=wp.int32),
            max_iter: wp.int32,
            x_buffer: wp.array(dtype=dtype, ndim=2),
        ):
            tid = wp.tid()
            idx = iter_idx[0]

            if idx >= max_iter:
                return

            if tid < x.shape[0]:
                x_buffer[idx, tid] = x[tid]

        return save_to_buffer_kernel_1d

    elif x_ndim == 2:

        @wp.kernel
        def save_to_buffer_kernel_2d(
            x: wp.array(dtype=dtype, ndim=2),
            iter_idx: wp.array(dtype=wp.int32),
            max_iter: wp.int32,
            x_buffer: wp.array(dtype=dtype, ndim=3),
        ):
            i, j = wp.tid()
            idx = iter_idx[0]

            if idx >= max_iter:
                return

            if i < x.shape[0] and j < x.shape[1]:
                x_buffer[idx, i, j] = x[i, j]

        return save_to_buffer_kernel_2d

    elif x_ndim == 3:

        @wp.kernel
        def save_to_buffer_kernel_3d(
            x: wp.array(dtype=dtype, ndim=3),
            iter_idx: wp.array(dtype=wp.int32),
            max_iter: wp.int32,
            x_buffer: wp.array(dtype=dtype, ndim=4),
        ):
            i, j, k = wp.tid()
            idx = iter_idx[0]

            if idx >= max_iter:
                return

            if i < x.shape[0] and j < x.shape[1] and k < x.shape[2]:
                x_buffer[idx, i, j, k] = x[i, j, k]

        return save_to_buffer_kernel_3d

    else:
        raise NotImplementedError(
            f"Shape with {x_ndim} dimensions is not supported for history logging."
        )


def get_or_create_save_kernel(arr: wp.array):
    key = (arr.ndim, arr.dtype)

    if key not in _SAVE_KERNEL_CACHE:
        _SAVE_KERNEL_CACHE[key] = create_save_to_buffer_kernel(arr)

    return _SAVE_KERNEL_CACHE[key]


class HistoryGroup:
    """
    Manages a collection of arrays that need to be logged at the same frequency
    (e.g., every Newton iteration).

    Allocates buffers automatically and manages the kernel launches for capturing data.
    """

    def __init__(self, capacity: int, index_array: wp.array, device: wp.Device):
        self.capacity = capacity
        self.index_array = index_array  # The GPU-resident counter (e.g., iter_count)
        self.device = device
        self.buffers: Dict[str, wp.array] = {}
        self._ops: List[Tuple[wp.array, wp.array, Any]] = []  # (source, buffer, kernel)

    def register(self, name: str, source_array: wp.array) -> wp.array:
        # 1. Allocate Buffer
        # Shape becomes (capacity, d1, d2, ...)
        buffer_shape = (self.capacity,) + source_array.shape
        buffer = wp.zeros(buffer_shape, dtype=source_array.dtype, device=self.device)

        # 2. Get/Compile Kernel (Cached)
        kernel = get_or_create_save_kernel(source_array)

        # 3. Store
        self.buffers[name] = buffer
        self._ops.append((source_array, buffer, kernel))

        return buffer

    def capture(self):
        for source, buffer, kernel in self._ops:
            wp.launch(
                kernel=kernel,
                dim=source.shape,
                inputs=[
                    source,
                    self.index_array,
                    self.capacity,
                ],
                outputs=[buffer],
                device=self.device,
            )
