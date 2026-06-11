import math
from typing import Optional

import warp as wp


def create_tiled_sum_kernel(tile_size: int, dtype: type):
    @wp.kernel
    def tiled_sum_kernel(
        inp: wp.array(dtype=dtype, ndim=2),
        out: wp.array(dtype=dtype, ndim=2),
    ):
        row, block = wp.tid()
        col_start = block * tile_size

        # Load tile of shape (1, tile_size)
        tile = wp.tile_load(inp, shape=(1, tile_size), offset=(row, col_start), bounds_check=True)

        # Sum the tile -> (1, 1)
        sum_tile = wp.tile_sum(tile)
        sum_tile = wp.tile_reshape(sum_tile, shape=(1, 1))

        # Atomically add to output at (row, 0)
        # We treat output as 2D (Rows, 1) to match tile dimensionality
        wp.tile_atomic_add(out, sum_tile, offset=(row, 0))

    return tiled_sum_kernel


def create_tiled_dot_kernel(tile_size: int, dtype: type):
    @wp.kernel
    def tiled_dot_kernel(
        a: wp.array(dtype=dtype, ndim=2),
        b: wp.array(dtype=dtype, ndim=2),
        out: wp.array(dtype=dtype, ndim=2),
    ):
        row, block = wp.tid()
        col_start = block * tile_size

        tile_a = wp.tile_load(a, shape=(1, tile_size), offset=(row, col_start), bounds_check=True)
        tile_b = wp.tile_load(b, shape=(1, tile_size), offset=(row, col_start), bounds_check=True)

        # Element-wise multiply
        prod = wp.tile_map(wp.mul, tile_a, tile_b)

        # Sum the result -> (1, 1)
        sum_tile = wp.tile_sum(prod)
        sum_tile = wp.tile_reshape(sum_tile, shape=(1, 1))

        # Atomically add to output
        wp.tile_atomic_add(out, sum_tile, offset=(row, 0))

    return tiled_dot_kernel


def create_tiled_argmin_kernel(tile_size: int, dtype: type):
    @wp.kernel
    def tiled_argmin_kernel(
        inp: wp.array(dtype=dtype, ndim=2),
        out_idx: wp.array(dtype=wp.int32, ndim=2),
        out_val: wp.array(dtype=dtype, ndim=2),
    ):
        row, block = wp.tid()
        col_start = block * tile_size

        tile = wp.tile_load(inp, shape=(1, tile_size), offset=(row, col_start))

        # Find min value and index
        min_val = wp.tile_min(tile)
        min_idx = wp.tile_argmin(tile)

        # Extract values
        out_idx[row, 0] = min_idx[0] + col_start
        out_val[row, 0] = min_val[0]

    return tiled_argmin_kernel


class TiledSum:
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 256,
        block_threads: int = 128,
        device: wp.Device | str = "cuda",
    ):
        """
        Tiled sum computation for arbitrary N-dimensional arrays.
        Sums the elements along the last dimension.

        Internally flattens the input to (ROWS, COLS) where COLS is the last dimension.

        Args:
            shape: Shape of the input array (d1, d2, ..., dN).
            dtype: Data type (default wp.float32).
            tile_size: Width of the tile to sum in one go.
            block_threads: CUDA threads per block.
            device: Computing device.
        """
        self.shape = tuple(shape)
        self.dtype = dtype
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        if len(self.shape) < 1:
            raise ValueError("Input shape must have at least 1 dimension.")

        # Flatten logic: (..., COLS) -> (ROWS, COLS)
        self.cols = self.shape[-1]
        self.rows = 1
        for d in self.shape[:-1]:
            self.rows *= d

        self.num_blocks = (self.cols + self.tile_size - 1) // self.tile_size

        # Create Kernel
        self.kernel = create_tiled_sum_kernel(self.tile_size, self.dtype)

        # Create dummy tensors for graph recording
        # Input view: (ROWS, COLS)
        # Output view: (ROWS, 1)
        a_dummy = wp.empty((self.rows, self.cols), dtype=self.dtype, device=self.device)
        out_dummy = wp.empty((self.rows, 1), dtype=self.dtype, device=self.device)

        self.launch = wp.launch_tiled(
            kernel=self.kernel,
            dim=(self.rows, self.num_blocks),
            inputs=[a_dummy],
            outputs=[out_dummy],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )

    def compute(self, a: wp.array, out: wp.array):
        """
        Computes out = sum(a, axis=-1).

        Args:
            a: Input array.
            out: Output array.
        """
        assert self.dtype == a.dtype == out.dtype, "Data types do not match."
        expected_out_shape = self.shape[:-1]
        # Warp doesn't support 0-d arrays, so we treat scalar output as (1,)
        if expected_out_shape == ():
            expected_out_shape = (1,)

        assert (
            out.shape == expected_out_shape
        ), f"Output shape {out.shape} mismatch. Expected {expected_out_shape}"

        # Zero output before accumulation
        out.zero_()

        # Create flattened views (Zero-copy)
        a_view = a.reshape((self.rows, self.cols))
        out_view = out.reshape((self.rows, 1))

        self.launch.set_param_at_index(0, a_view)
        self.launch.set_param_at_index(1, out_view)
        self.launch.launch()


class TiledDot:
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 256,
        block_threads: int = 128,
        device: wp.Device | str = "cuda",
    ):
        """
        Tiled dot product computation for arbitrary N-dimensional arrays.
        Computes sum(a * b) along the last dimension.

        Args:
            shape: Shape of the input arrays (d1, d2, ..., dN).
            dtype: Data type.
            tile_size: Width of the tile.
            block_threads: CUDA threads per block.
            device: Computing device.
        """
        self.shape = tuple(shape)
        self.dtype = dtype
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        if len(self.shape) < 1:
            raise ValueError("Input shape must have at least 1 dimension.")

        self.cols = self.shape[-1]
        self.rows = 1
        for d in self.shape[:-1]:
            self.rows *= d

        self.num_blocks = (self.cols + self.tile_size - 1) // self.tile_size

        self.kernel = create_tiled_dot_kernel(self.tile_size, self.dtype)

        a_dummy = wp.empty((self.rows, self.cols), dtype=self.dtype, device=self.device)
        b_dummy = wp.empty((self.rows, self.cols), dtype=self.dtype, device=self.device)
        out_dummy = wp.empty((self.rows, 1), dtype=self.dtype, device=self.device)

        self.launch = wp.launch_tiled(
            kernel=self.kernel,
            dim=(self.rows, self.num_blocks),
            inputs=[a_dummy, b_dummy],
            outputs=[out_dummy],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )

    def compute(self, a: wp.array, b: wp.array, out: wp.array):
        """
        Computes out = sum(a * b, axis=-1).
        """
        assert self.dtype == a.dtype == b.dtype == out.dtype, "Data types do not match."
        assert (
            a.shape == b.shape == self.shape
        ), f"Input shapes {a.shape}, {b.shape} mismatch. Expected {self.shape}"
        expected_out_shape = self.shape[:-1]
        if expected_out_shape == ():
            expected_out_shape = (1,)

        assert (
            out.shape == expected_out_shape
        ), f"Output shape {out.shape} mismatch. Expected {expected_out_shape}"

        out.zero_()

        a_view = a.reshape((self.rows, self.cols))
        b_view = b.reshape((self.rows, self.cols))
        out_view = out.reshape((self.rows, 1))

        self.launch.set_param_at_index(0, a_view)
        self.launch.set_param_at_index(1, b_view)
        self.launch.set_param_at_index(2, out_view)
        self.launch.launch()


class TiledSqNorm(TiledDot):
    """
    Tiled squared norm computation (sum(a^2)).
    Inherits from TiledDot and computes dot(a, a).
    """

    def compute(self, a: wp.array, out: wp.array):
        super().compute(a, a, out)


class TiledArgMin:
    def __init__(
        self,
        shape: tuple | list,
        dtype: type = wp.float32,
        tile_size: int = 256,
        block_threads: int = 128,
        device: wp.Device | str = "cuda",
    ):
        """
        Tiled argmin computation.
        Finds min value and index along the last dimension.
        """
        self.shape = tuple(shape)
        self.dtype = dtype
        self.tile_size = tile_size
        self.block_threads = block_threads
        self.device = device

        if len(self.shape) < 1:
            raise ValueError("Input shape must have at least 1 dimension.")

        self.cols = self.shape[-1]
        self.rows = 1
        for d in self.shape[:-1]:
            self.rows *= d

        self.num_blocks = (self.cols + self.tile_size - 1) // self.tile_size

        # Currently only supports single block reduction
        if self.num_blocks > 1:
            raise NotImplementedError(
                "TiledArgMin currently supports only inputs that fit in a single tile (last dim <= tile_size)."
            )

        self.kernel = create_tiled_argmin_kernel(self.tile_size, self.dtype)

        a_dummy = wp.empty((self.rows, self.cols), dtype=self.dtype, device=self.device)
        out_idx_dummy = wp.empty((self.rows, 1), dtype=wp.int32, device=self.device)
        out_val_dummy = wp.empty((self.rows, 1), dtype=self.dtype, device=self.device)

        self.launch = wp.launch_tiled(
            kernel=self.kernel,
            dim=(self.rows, self.num_blocks),
            inputs=[a_dummy, out_idx_dummy, out_val_dummy],
            block_dim=self.block_threads,
            device=self.device,
            record_cmd=True,
        )

    def compute(self, a: wp.array, out_idx: wp.array, out_val: Optional[wp.array] = None):
        """
        Computes index and value of minimum along last axis.
        """
        assert a.dtype == self.dtype, "Data types do not match."

        # Output shapes expected: shape[:-1]
        expected_out_shape = self.shape[:-1]
        if expected_out_shape == ():
            expected_out_shape = (1,)

        # Check out_idx shape
        assert out_idx.shape == expected_out_shape, f"Idx shape {out_idx.shape} mismatch."

        # If out_val provided, check shape
        if out_val is not None:
            assert out_val.shape == expected_out_shape, f"Val shape {out_val.shape} mismatch."
        else:
            # Create temp buffer if not provided (though kernel writes to it)
            # Actually we need to provide a buffer to the kernel.
            # If user didn't provide out_val, we use a temporary one or the dummy?
            # We can't use dummy because of race conditions if threaded?
            # But here we are sequential.
            # However, Warp launch needs valid arrays.
            # We can construct a temporary one.
            out_val = wp.empty(expected_out_shape, dtype=self.dtype, device=self.device)

        a_view = a.reshape((self.rows, self.cols))
        out_idx_view = out_idx.reshape((self.rows, 1))
        out_val_view = out_val.reshape((self.rows, 1))

        self.launch.set_param_at_index(0, a_view)
        self.launch.set_param_at_index(1, out_idx_view)
        self.launch.set_param_at_index(2, out_val_view)
        self.launch.launch()
