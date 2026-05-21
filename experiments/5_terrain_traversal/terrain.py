"""Procedural terrain generation for terrain traversal experiments.

Generates random heightmap meshes using sum-of-sinusoids with randomized
amplitudes, frequencies, and phases.
"""

import numpy as np
import newton


def generate_heightmap(
    seed: int,
    grid_res: int = 80,
    extent: tuple[float, float] = (24.0, 24.0),
    num_octaves: int = 4,
    amplitude_range: tuple[float, float] = (0.05, 0.25),
    freq_range: tuple[float, float] = (0.5, 2.5),
    roughness: float = 1.0,
    terrain_freq: float = 1.0,
) -> np.ndarray:
    """Generate a random heightmap using sum-of-sinusoids.

    Args:
        seed: Random seed for reproducibility.
        grid_res: Number of grid points per dimension.
        extent: Physical size (x, y) in meters.
        num_octaves: Number of sinusoidal components to sum.
        amplitude_range: (min, max) amplitude per octave in meters.
        freq_range: (min, max) spatial frequency per octave.
        roughness: Multiplier for amplitude (0.5=gentle, 1.0=default, 2.0=rugged).
        terrain_freq: Multiplier for frequency (0.5=broad hills, 1.0=default, 2.0=tight ripples).

    Returns:
        Heightfield array of shape (grid_res, grid_res).
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(-extent[0] / 2, extent[0] / 2, grid_res)
    y = np.linspace(-extent[1] / 2, extent[1] / 2, grid_res)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Z = np.zeros_like(X)
    for _ in range(num_octaves):
        amp = rng.uniform(*amplitude_range) * roughness
        fx = rng.uniform(*freq_range) * terrain_freq
        fy = rng.uniform(*freq_range) * terrain_freq
        px = rng.uniform(0, 2 * np.pi)
        py = rng.uniform(0, 2 * np.pi)
        Z += amp * np.sin(2 * np.pi * fx * X / extent[0] + px) * np.sin(
            2 * np.pi * fy * Y / extent[1] + py
        )

    return Z.astype(np.float32)


def heightmap_to_mesh(
    heightfield: np.ndarray,
    extent: tuple[float, float] = (24.0, 24.0),
    center: tuple[float, float] = (0.0, 0.0),
    z_offset: float = 0.05,
) -> newton.Mesh:
    """Convert a 2D heightfield array to a Newton triangle mesh.

    Args:
        heightfield: (grid_res_x, grid_res_y) array of Z heights.
        extent: Physical size (x, y) in meters.
        center: Center (x, y) of the mesh.
        z_offset: Vertical offset added to all heights.

    Returns:
        A newton.Mesh instance.
    """
    nx, ny = heightfield.shape
    x = np.linspace(-extent[0] / 2, extent[0] / 2, nx) + center[0]
    y = np.linspace(-extent[1] / 2, extent[1] / 2, ny) + center[1]
    X, Y = np.meshgrid(x, y, indexing="ij")

    vertices = np.column_stack(
        [X.ravel(), Y.ravel(), heightfield.ravel() + z_offset]
    ).astype(np.float32)

    # Two triangles per grid cell
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    v0 = ii * ny + jj
    v1 = ii * ny + (jj + 1)
    v2 = (ii + 1) * ny + jj
    v3 = (ii + 1) * ny + (jj + 1)

    faces = np.column_stack([v0, v2, v1, v1, v2, v3]).reshape(-1, 3).astype(np.int32)
    indices = faces.flatten()

    return newton.Mesh(vertices, indices)


def sample_height_at(
    heightfield: np.ndarray,
    extent: tuple[float, float],
    x: float,
    y: float,
    z_offset: float = 0.05,
) -> float:
    """Sample the heightfield at a world (x, y) position via bilinear interpolation.

    Args:
        heightfield: (grid_res_x, grid_res_y) array of Z heights.
        extent: Physical size (x, y) in meters.
        x: World x coordinate.
        y: World y coordinate.
        z_offset: Vertical offset applied to the mesh.

    Returns:
        Interpolated terrain height at (x, y).
    """
    nx, ny = heightfield.shape
    # Map world coords to grid indices
    gi = (x + extent[0] / 2) / extent[0] * (nx - 1)
    gj = (y + extent[1] / 2) / extent[1] * (ny - 1)
    gi = np.clip(gi, 0, nx - 1)
    gj = np.clip(gj, 0, ny - 1)
    i0, j0 = int(gi), int(gj)
    i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
    di, dj = gi - i0, gj - j0
    h = (
        heightfield[i0, j0] * (1 - di) * (1 - dj)
        + heightfield[i1, j0] * di * (1 - dj)
        + heightfield[i0, j1] * (1 - di) * dj
        + heightfield[i1, j1] * di * dj
    )
    return float(h) + z_offset


def generate_terrain_mesh(
    seed: int,
    grid_res: int = 80,
    extent: tuple[float, float] = (24.0, 24.0),
    z_offset: float = 0.05,
    spawn_xy: tuple[float, float] = (-8.0, 0.0),
    **heightmap_kwargs,
) -> tuple[newton.Mesh, float]:
    """Generate a random terrain mesh from a seed.

    Args:
        seed: Random seed.
        grid_res: Grid resolution per dimension.
        extent: Physical size (x, y) in meters.
        z_offset: Vertical offset for the mesh.
        spawn_xy: (x, y) position to sample the spawn height at.
        **heightmap_kwargs: Passed to generate_heightmap.

    Returns:
        Tuple of (newton.Mesh, spawn_height) where spawn_height is the
        terrain height at spawn_xy.
    """
    hmap = generate_heightmap(seed=seed, grid_res=grid_res, extent=extent, **heightmap_kwargs)
    mesh = heightmap_to_mesh(hmap, extent=extent, z_offset=z_offset)
    spawn_height = sample_height_at(hmap, extent, x=spawn_xy[0], y=spawn_xy[1], z_offset=z_offset)
    return mesh, spawn_height
