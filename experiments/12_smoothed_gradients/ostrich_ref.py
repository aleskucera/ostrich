"""Ostrich reference runs (CPU) for comparing against sim2d.

A 3D box / sphere on the ground plane, moving in the x-z plane. Mapping to 2D:
x -> x, z -> y, rotation about +y by phi -> theta = -phi.
"""

import dataclasses

import newton
import numpy as np
import warp as wp
from ostrich.core.engine import OstrichEngine
from ostrich.core.engine_config import OstrichEngineConfig
from ostrich.core.logging_config import LoggingConfig
from ostrich.core.model_builder import OstrichModelBuilder

wp.config.quiet = True


def tight_config(max_iters=100, atol=1e-6) -> OstrichEngineConfig:
    """Many Newton / linear iterations and no warm start, so Ostrich gets as
    close to the root as float32 allows."""
    cfg = OstrichEngineConfig()
    return dataclasses.replace(
        cfg,
        nr=dataclasses.replace(cfg.nr, max_iters=max_iters, atol=atol),
        linear=dataclasses.replace(cfg.linear, max_iters=100, tol=1e-10, atol=1e-10),
        warm_start=dataclasses.replace(cfg.warm_start, enabled=False),
    )


def build(shape: str, mu=0.5, size=(0.2, 0.2, 0.05), radius=0.05, z0=None):
    """Returns the model; the body mass includes the shape's density."""
    b = OstrichModelBuilder()
    b.rigid_gap = 0.01
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    half_height = size[2] if shape == "box" else radius
    z0 = half_height + 0.0005 if z0 is None else z0
    body = b.add_body(mass=1.0, xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()))
    if shape == "box":
        b.add_shape_box(body=body, hx=size[0], hy=size[1], hz=size[2],
                        cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    else:
        b.add_shape_sphere(body=body, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    return b.finalize_replicated(num_worlds=1, gravity=-9.81)


def mass_and_inertia(model):
    """Mass and the inertia about y (the axis of planar rotation)."""
    return float(model.body_mass.numpy()[0]), float(model.body_inertia.numpy()[0, 1, 1])


def to_2d(state):
    """(q, v) in sim2d coordinates from an Ostrich state."""
    x, y, z, qx, qy, qz, qw = state.body_q.numpy().reshape(-1, 7)[0]
    vx, vy, vz, wx, wy, wz = state.body_qd.numpy().reshape(-1, 6)[0]
    theta = -2.0 * np.arctan2(qy, qw)
    return np.array([x, z, theta]), np.array([vx, vz, -wy])


def run(model, n_steps, dt=0.01, config=None, kick=None, kick_at=0):
    """Steps the model; at step `kick_at` the 2D velocity is set to `kick`.
    Returns arrays q, v of shape (n_steps + 1, 3) in sim2d coordinates."""
    engine = OstrichEngine(model=model, sim_steps=n_steps,
                           config=config or tight_config(), logging_config=LoggingConfig())
    s_in, s_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)
    ctrl = model.control()
    qs, vs = [], []
    for k in range(n_steps + 1):
        if kick is not None and k == kick_at:
            qd = np.zeros((1, 6), dtype=np.float32)
            qd[0, [0, 2, 4]] = kick[0], kick[1], -kick[2]
            wp.copy(s_in.body_qd, wp.array(qd, dtype=wp.spatial_vector).reshape(s_in.body_qd.shape))
        q, v = to_2d(s_in)
        qs.append(q)
        vs.append(v)
        if k == n_steps:
            break
        contacts = model.collide(s_in)
        engine.step(s_in, s_out, ctrl, contacts, dt)
        s_in, s_out = s_out, s_in
    return np.array(qs), np.array(vs)
