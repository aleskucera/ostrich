from dataclasses import dataclass
from typing import Generic
from typing import Optional
from typing import Tuple
from typing import TypeVar

import warp as wp

from .engine_dims import EngineDimensions

# Generic type var to indicate this works for float arrays, vector arrays, etc.
T = TypeVar("T")


@wp.kernel
def _cast_float_to_spatial_2d(
    src: wp.array(dtype=float, ndim=2),
    dst: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    # i: world index, j: body index
    i, j = wp.tid()
    offset = j * 6
    dst[i, j] = wp.spatial_vector(
        src[i, offset + 0],
        src[i, offset + 1],
        src[i, offset + 2],
        src[i, offset + 3],
        src[i, offset + 4],
        src[i, offset + 5],
    )


@wp.kernel
def _cast_float_to_spatial_3d(
    src: wp.array(dtype=float, ndim=3),
    dst: wp.array(dtype=wp.spatial_vector, ndim=3),
):
    # i: step index, j: world index, k: body index
    i, j, k = wp.tid()
    offset = k * 6
    dst[i, j, k] = wp.spatial_vector(
        src[i, j, offset + 0],
        src[i, j, offset + 1],
        src[i, j, offset + 2],
        src[i, j, offset + 3],
        src[i, j, offset + 4],
        src[i, j, offset + 5],
    )


@wp.kernel
def _cast_spatial_to_float_2d(
    src: wp.array(dtype=wp.spatial_vector, ndim=2),
    dst: wp.array(dtype=float, ndim=2),
):
    i, j = wp.tid()
    vec = src[i, j]
    offset = j * 6
    dst[i, offset + 0] = vec[0]
    dst[i, offset + 1] = vec[1]
    dst[i, offset + 2] = vec[2]
    dst[i, offset + 3] = vec[3]
    dst[i, offset + 4] = vec[4]
    dst[i, offset + 5] = vec[5]


@wp.kernel
def _cast_spatial_to_float_3d(
    src: wp.array(dtype=wp.spatial_vector, ndim=3),
    dst: wp.array(dtype=float, ndim=3),
):
    i, j, k = wp.tid()
    vec = src[i, j, k]
    offset = k * 6
    dst[i, j, offset + 0] = vec[0]
    dst[i, j, offset + 1] = vec[1]
    dst[i, j, offset + 2] = vec[2]
    dst[i, j, offset + 3] = vec[3]
    dst[i, j, offset + 4] = vec[4]
    dst[i, j, offset + 5] = vec[5]


@dataclass(frozen=True)
class ConstraintView(Generic[T]):
    data: wp.array
    dims: EngineDimensions
    axis: int = -1

    def _slice_at_axis(self, s: slice) -> Optional[wp.array]:
        if s.start == s.stop:
            return None
        ndim = self.data.ndim
        target_axis = self.axis % ndim
        pre_slice = (slice(None),) * target_axis
        post_slice = (slice(None),) * (ndim - target_axis - 1)
        return self.data[pre_slice + (s,) + post_slice]

    @property
    def full(self) -> wp.array:
        return self.data

    @property
    def j(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_j)

    @property
    def ctrl(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_ctrl)

    @property
    def eq(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_eq)

    @property
    def n(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_n)

    @property
    def f(self) -> Optional[wp.array]:
        return self._slice_at_axis(self.dims.slice_f)

    def zero_(self):
        self.data.zero_()


@dataclass(frozen=True)
class SystemView(Generic[T]):
    data: wp.array
    dims: EngineDimensions
    _d_spatial: Optional[wp.array] = None

    def _get_split_slice(self, start, stop) -> Tuple[slice]:
        ndim = self.data.ndim
        return (slice(None),) * (ndim - 1) + (slice(start, stop),)

    @property
    def full(self) -> wp.array:
        return self.data

    @property
    def d(self) -> wp.array:
        """Dynamics part (float view)."""
        indexer = self._get_split_slice(None, self.dims.N_u)
        return self.data[indexer]

    @property
    def d_spatial(self) -> wp.array:
        """
        Dynamics part reinterpreted as spatial vectors.
        Returns the persistent auxiliary buffer.
        """
        if self._d_spatial is not None:
            return self._d_spatial

        # Fallback (Slow path)
        print("WARNING: This should not happen.")
        d_data = self.d.contiguous()
        base_shape = d_data.shape[:-1]
        new_shape = base_shape + (self.dims.N_b,)
        return wp.array(d_data, shape=new_shape, dtype=wp.spatial_vector)

    def sync_to_float(self):
        """Copies data from the spatial buffer to the main float array."""
        if self._d_spatial is not None:
            if self._d_spatial.ndim == 2:
                kernel = _cast_spatial_to_float_2d
            elif self._d_spatial.ndim == 3:
                kernel = _cast_spatial_to_float_3d
            else:
                raise ValueError(f"Unsupported spatial buffer dimensionality: {self._d_spatial.ndim}")

            wp.launch(
                kernel=kernel,
                dim=self._d_spatial.shape,
                inputs=[self._d_spatial, self.d],
                device=self.data.device,
            )

    def sync_to_spatial(self):
        """Copies data from the main float array to the spatial buffer."""
        if self._d_spatial is not None:
            if self._d_spatial.ndim == 2:
                kernel = _cast_float_to_spatial_2d
            elif self._d_spatial.ndim == 3:
                kernel = _cast_float_to_spatial_3d
            else:
                raise ValueError(f"Unsupported spatial buffer dimensionality: {self._d_spatial.ndim}")

            wp.launch(
                kernel=kernel,
                dim=self._d_spatial.shape,
                inputs=[self.d, self._d_spatial],
                device=self.data.device,
            )

    @property
    def c(self) -> ConstraintView[T]:
        indexer = self._get_split_slice(self.dims.N_u, None)
        return ConstraintView(self.data[indexer], self.dims, axis=-1)

    def zero_(self):
        self.data.zero_()
