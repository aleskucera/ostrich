"""Ostrich: one sliding step of a box, d v1 / d v0 from step_backward vs central FD (CPU).

Run from the repo root. ADJ_C sets adjoint.contact_compliance (default 1e-6);
OSTRICH_FRICTION_ADJOINT=frozen selects the legacy friction mode.
"""
import dataclasses, os, sys
sys.path.insert(0, "experiments/12_smoothed_gradients")
import numpy as np, warp as wp, newton
import ostrich_ref as ref
from ostrich.core.engine import OstrichEngine
from ostrich.core.logging_config import LoggingConfig
from ostrich.simulation.trajectory_buffer import TrajectoryBuffer

DT, V_KICK = 0.01, 1.5
with wp.ScopedDevice("cpu"):
    model = ref.build("box")
    engine = OstrichEngine(model=model, sim_steps=2, config=dataclasses.replace(ref.tight_config(max_iters=100, atol=1e-7), adjoint=dataclasses.replace(ref.tight_config().adjoint, contact_compliance=float(os.environ.get("ADJ_C", "1e-6")))),
                           logging_config=LoggingConfig(), differentiable_simulation=True)
    s_in, s_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s_in)
    for _ in range(30):
        engine.step(s_in, s_out, model.control(), model.collide(s_in), DT); s_in, s_out = s_out, s_in
    q_rest = s_in.body_q.numpy().copy()
    qd0 = np.zeros((1, 6), np.float32); qd0[0, 0] = V_KICK

    def forward(qd):
        st, so = model.state(), model.state()
        wp.copy(st.body_q, wp.array(q_rest, dtype=wp.transform).reshape(st.body_q.shape))
        wp.copy(st.body_qd, wp.array(qd.astype(np.float32), dtype=wp.spatial_vector).reshape(st.body_qd.shape))
        engine.step(st, so, model.control(), model.collide(st), DT)
        return so.body_qd.numpy().reshape(-1)

    v1 = forward(qd0)
    buf = TrajectoryBuffer(data=engine.data, contacts=engine.ostrich_contacts, dims=engine.dims, num_steps=1, device=model.device)
    buf.save_step(0, engine.data, engine.ostrich_contacts)
    cf = engine.data._constr_force.numpy()[0]; on, of = engine.dims.offset_n, engine.dims.offset_f
    print("v1", v1.round(5), " lam_n", cf[on:on + 4].round(2), " |lam_t|/(mu lam_n)", (np.linalg.norm(cf[of:of + 8].reshape(4, 2), axis=1) / (0.5 * cf[on:on + 4])).round(4))

    rows = {}
    for out_idx, name in ((0, "vx1"), (2, "vz1")):
        buf.zero_grad()
        g = np.zeros((1, 6), np.float32); g[0, out_idx] = 1.0
        wp.copy(buf.body_vel.grad[1], wp.array(g, dtype=wp.spatial_vector).reshape(engine.data.body_vel_grad.shape))
        buf.load_step(0, engine.data, engine.ostrich_contacts)
        engine.data.zero_gradients(); engine.step_backward()
        adj = engine.data.body_vel_prev.grad.numpy().reshape(-1).copy()
        eps = 1e-3; fd = np.zeros(6)
        for j in range(6):
            d = np.zeros((1, 6)); d[0, j] = eps
            fd[j] = (forward(qd0 + d)[out_idx] - forward(qd0 - d)[out_idx]) / (2 * eps)
        rows[name] = (adj, fd)
    mode = os.environ.get("OSTRICH_FRICTION_ADJOINT", "true (default)")
    print(f"friction_linearization = {mode}; columns: d/d(vx0, vy0, vz0, wx0, wy0, wz0)")
    for name, (adj, fd) in rows.items():
        print(f"  d{name}: adjoint {adj.round(4)}")
        print(f"  FD{'':5s}  {fd.round(4)}")
