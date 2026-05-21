"""Contact-pipeline preprocessing.

This subpackage sits between Newton's narrow-phase output and Axion's
constraint kernels. Today it provides per-pair contact reduction policies;
in the future it can host other contact-pipeline preprocessing (warm
starting, contact-graph analysis, etc.).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ContactReducer, NoOpReducer
from .config import ContactReductionConfig, ContactReductionPolicy

if TYPE_CHECKING:
    from axion.core.engine_data import EngineData
    from axion.core.engine_dims import EngineDimensions
    from axion.core.model import AxionModel
    import warp as wp


def build_reducer(
    cfg: ContactReductionConfig,
    axion_model: "AxionModel",
    data: "EngineData",
    dims: "EngineDimensions",
    device: "wp.Device",
) -> ContactReducer:
    """Construct a reducer instance from a config.

    Args:
        cfg: User-provided reduction settings. ``cfg.policy="none"``
            yields a no-op reducer.
        axion_model: Source of ``shape_body`` (per-world shape→body table).
        data: Source of ``body_pose_prev`` (current step's body pose,
            updated by ``load_data`` before reduction runs).
        dims: Engine dimensions; reducers allocate persistent
            ``(num_worlds, contact_count)`` scratch buffers from these.
        device: Warp device for kernel launches and scratch allocation.
    """
    policy = cfg.policy
    if policy == "none":
        return NoOpReducer()
    if policy == "top_k":
        from .top_k import TopKReducer

        return TopKReducer(cfg, axion_model, data, dims, device)
    if policy == "fps":
        from .fps import FPSReducer

        return FPSReducer(cfg, axion_model, data, dims, device)
    if policy == "cluster":
        from .cluster import ClusterReducer

        return ClusterReducer(cfg, axion_model, data, dims, device)
    raise NotImplementedError(
        f"Contact reduction policy {policy!r} is not implemented yet. "
        "Subsequent phases add 'hull'."
    )


__all__ = [
    "ContactReducer",
    "NoOpReducer",
    "ContactReductionConfig",
    "ContactReductionPolicy",
    "build_reducer",
]
