"""Scene generators for tests, examples, and dataset creation.

Two generators with different invariants — pick by the placement guarantee
you need:

- :class:`PlacementSceneGenerator` — collision-aware. Tracks created bodies
  and rejects placements that would overlap. Supports free, ground-resting,
  contact-touching, and kinematic-chain construction.
- :class:`RandomSceneGenerator` — careless. Scatters bodies and builds
  random articulated trees with no overlap checks; per-call bounds and
  density. Useful for training data and stress tests.
"""
from .placement_scene_generator import PlacementSceneGenerator
from .random_scene_generator import RandomSceneGenerator

__all__ = ["PlacementSceneGenerator", "RandomSceneGenerator"]
