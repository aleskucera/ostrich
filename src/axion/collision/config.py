"""Configuration for contact reduction policies.

The reducer runs once per simulation step, between Newton's narrow-phase
output and the construction of the constraint linear system. It groups
active contacts by the (body0, body1) pair they touch and prunes the set
to at most ``max_per_pair`` representatives per pair.
"""
from dataclasses import dataclass, fields
from typing import Literal


ContactReductionPolicy = Literal["none", "top_k", "fps", "cluster", "hull"]


@dataclass(frozen=True)
class ContactReductionConfig:
    """Per-pair contact reduction settings.

    Attributes:
        policy: Reduction strategy.
            - ``"none"``: pass-through (no reduction); the engine behaves as
              before this module existed.
            - ``"top_k"``: keep the K deepest contacts per (b0, b1) pair.
            - ``"fps"``: greedy farthest-point sampling per pair, seeded
              with the deepest contact.
            - ``"cluster"``: collapse contacts whose normals and positions
              agree within thresholds; keep the deepest member of each
              cluster, up to K clusters.
            - ``"hull"``: 2D convex hull on the contact plane, keep
              boundary points, up to K.
        max_per_pair: K — upper bound on contacts kept per (b0, b1) pair.
            Ignored when ``policy="none"``. K=4 matches Bullet's default.
        cluster_normal_dot_thresh: Two contact normals are considered
            "the same direction" when their dot product exceeds this value
            (~0.996 ≈ 5°). Only consumed by ``policy="cluster"``.
        cluster_pos_thresh: World-space distance below which two contacts
            on the same body pair are considered the "same point" [m].
            Only consumed by ``policy="cluster"``.
    """

    policy: ContactReductionPolicy = "none"
    max_per_pair: int = 4
    cluster_normal_dot_thresh: float = 0.996
    cluster_pos_thresh: float = 5e-3

    def __post_init__(self) -> None:
        if self.max_per_pair < 1:
            raise ValueError(
                f"max_per_pair must be >= 1, got {self.max_per_pair}"
            )
        if not (-1.0 <= self.cluster_normal_dot_thresh <= 1.0):
            raise ValueError(
                f"cluster_normal_dot_thresh must be in [-1, 1], "
                f"got {self.cluster_normal_dot_thresh}"
            )
        if self.cluster_pos_thresh < 0.0:
            raise ValueError(
                f"cluster_pos_thresh must be >= 0, got {self.cluster_pos_thresh}"
            )

    @classmethod
    def coerce(cls, obj) -> "ContactReductionConfig":
        """Best-effort conversion from a dict-like object (e.g. a Hydra
        ``DictConfig`` left in place by partial overrides) into a real
        ``ContactReductionConfig`` with proper defaults filled in.

        Already-instantiated configs are returned unchanged.
        """
        if isinstance(obj, cls):
            return obj
        try:
            items = dict(obj)
        except Exception:
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in items.items() if k in valid})
