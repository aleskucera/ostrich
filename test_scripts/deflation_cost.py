"""Step 2 — wall-time estimate for projection deflation in the engine.

Fewer iterations is not a speed-up by itself: deflation adds a per-solve
setup (AZ = k products with A, E = ZᵀAZ, a k×k Cholesky), one projection
per iteration, a final correction, and the cost of producing Z.

(A) ITERATIONS under the engine's own settings (offline, CPU): float32,
    library default tol 1e-5 / cap 16 (the config that produced the
    dumped systems) and examples/conf/engine/ostrich.yaml tol 1e-3 / cap
    26. Capped solves also report the accuracy actually reached.

(B) GPU TIMING with the engine's REAL PCRSolver + SystemOperator +
    JacobiPreconditioner, captured as one CUDA graph (as deployed),
    convergence disabled to time a fixed iteration count. Deflation is
    written as Warp kernels that mirror the matrix-free operator
    (multi-column A·Z scatter/gather, block-reduction dot products, the
    projection fused into the matvec — +1 launch per iteration, see the
    kernel section), captured in the same graph. Z production (GCRO-DR
    harmonic Ritz) has no Warp implementation; it is timed with eager
    torch on the GPU — an upper bound at small W. A correctness check
    runs the Warp deflated solve against the offline deflated_pcr.

(C) ESTIMATE per solve = fixed+setup + Z build + iterations × per-iteration,
    all three measured. All worlds iterate until the slowest converges
    (capture_while), so for W > 1 the iteration count is the expected MAX
    over W systems drawn from the held-out set. "ceiling" = zero overhead
    (only the iteration savings count): if it does not win, no
    implementation can.

Run:
    python test_scripts/deflation_cost.py
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import tempfile
import time
from types import SimpleNamespace as NS

import numpy as np
import torch
import warp as wp

from ostrich.mechanics import compute_spatial_momentum
from ostrich.optim import PCRSolver
from ostrich.optim.preconditioner import JacobiPreconditioner
from ostrich.optim.system_operator import (SystemLinearData, SystemOperator,
                                           kernel_invM_Jt_matvec_direct)
from test_scripts.precond_lab import jacobi_apply, load_systems
from test_scripts.recycle_probe import NC, embed, pct
from test_scripts.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation_ab import z_sources

DEV = "cuda:0"
REG = 1e-6
SETTINGS = {"default 1e-5/cap16": (1e-5, 16), "yaml 1e-3/cap26": (1e-3, 26)}
WORLDS = (1, 64, 1024)
KS = (8, 16, 18)
GCRODR_M = 16


# ---------------------------------------------------------------- Warp kernels
#
# Layout: Z and AZ are stored transposed, (W, k, N_c), so threads over the
# constraint slots c read consecutive memory. 402-long dot products are
# block reductions (one thread block per world, Warp tiles), never serial
# loops. Per PCR iteration the deflated operator adds ONE launch: since A
# is symmetric, Zᵀ(Ax) = (AZ)ᵀx, so s = E⁻¹(AZ)ᵀx is computed from the
# matvec's input, and z = Ax − AZ·s is fused into the matvec's gather pass.

BLOCK = 128


@wp.kernel
def _az_scatter(Zt: wp.array(dtype=wp.float32, ndim=3),
                J: wp.array(dtype=wp.spatial_vector, ndim=3),
                bidx: wp.array(dtype=wp.int32, ndim=3),
                mask: wp.array(dtype=wp.float32, ndim=2),
                minv: wp.array(dtype=wp.float32, ndim=2),
                Iinv: wp.array(dtype=wp.mat33, ndim=2),
                tmp: wp.array(dtype=wp.spatial_vector, ndim=3)):
    """tmp[w, j, body] += M⁻¹ Jᵀ Z[:, j]   (SystemOperator pass 1, k columns)."""
    w, j, c = wp.tid()
    if mask[w, c] == 0.0:
        return
    x = Zt[w, j, c]
    if x == 0.0:
        return
    b1 = bidx[w, c, 0]
    b2 = bidx[w, c, 1]
    if b1 >= 0:
        wp.atomic_add(tmp, w, j, b1, compute_spatial_momentum(minv[w, b1], Iinv[w, b1], x * J[w, c, 0]))
    if b2 >= 0:
        wp.atomic_add(tmp, w, j, b2, compute_spatial_momentum(minv[w, b2], Iinv[w, b2], x * J[w, c, 1]))


@wp.kernel
def _az_gather(tmp: wp.array(dtype=wp.spatial_vector, ndim=3),
               Zt: wp.array(dtype=wp.float32, ndim=3),
               J: wp.array(dtype=wp.spatial_vector, ndim=3),
               bidx: wp.array(dtype=wp.int32, ndim=3),
               mask: wp.array(dtype=wp.float32, ndim=2),
               C: wp.array(dtype=wp.float32, ndim=2),
               reg: float,
               AZt: wp.array(dtype=wp.float32, ndim=3)):
    """AZ[:, j] = J tmp[j] + (C + reg) Z[:, j]   (pass 2)."""
    w, j, c = wp.tid()
    v = float(0.0)
    if mask[w, c] > 0.0:
        b1 = bidx[w, c, 0]
        b2 = bidx[w, c, 1]
        if b1 >= 0:
            v += wp.dot(J[w, c, 0], tmp[w, j, b1])
        if b2 >= 0:
            v += wp.dot(J[w, c, 1], tmp[w, j, b2])
        v += (C[w, c] + reg) * Zt[w, j, c]
    AZt[w, j, c] = v


@functools.cache
def k_kernels(k):
    """Kernels whose inner k-loops must be unrolled (k is a compile-time constant)."""
    vk = wp.types.vector(length=k, dtype=wp.float32)
    opts = {"max_unroll": 64}

    @wp.kernel(enable_backward=False, module="unique", module_options=opts)
    def reduce_solve(Mt: wp.array3d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32),
                     base: wp.array2d(dtype=wp.float32), sign: float,
                     Einv: wp.array3d(dtype=wp.float32),
                     t_out: wp.array2d(dtype=wp.float32), s_out: wp.array2d(dtype=wp.float32)):
        """t = base + sign·Mᵀv (block reduction over slots), s = E⁻¹ t."""
        w, lane = wp.tid()
        acc = vk()
        for c in range(lane, v.shape[1], BLOCK):
            vc = v[w, c]
            for j in range(k):
                acc[j] += Mt[w, j, c] * vc
        t = wp.tile_sum(wp.tile(acc, preserve_type=True))[0]
        if lane < k:
            tj = base[w, lane] + sign * t[lane]
            t_out[w, lane] = tj
            sj = float(0.0)
            for l in range(k):
                sj += Einv[w, lane, l] * (base[w, l] + sign * t[l])
            s_out[w, lane] = sj

    @wp.kernel(enable_backward=False, module="unique", module_options=opts)
    def gram(Zt: wp.array3d(dtype=wp.float32), AZt: wp.array3d(dtype=wp.float32),
             E: wp.array3d(dtype=wp.float32)):
        """E[i, :] = Z[:, i]ᵀ AZ   (one block per (world, row))."""
        w, i, lane = wp.tid()
        acc = vk()
        for c in range(lane, Zt.shape[2], BLOCK):
            zc = Zt[w, i, c]
            for j in range(k):
                acc[j] += zc * AZt[w, j, c]
        e = wp.tile_sum(wp.tile(acc, preserve_type=True))[0]
        if lane < k:
            E[w, i, lane] = e[lane]

    @wp.kernel(enable_backward=False, module="unique", module_options=opts)
    def gather_project(dyn: wp.array(dtype=wp.spatial_vector, ndim=2),
                       J: wp.array(dtype=wp.spatial_vector, ndim=3),
                       bidx: wp.array(dtype=wp.int32, ndim=3),
                       mask: wp.array(dtype=wp.float32, ndim=2),
                       C: wp.array(dtype=wp.float32, ndim=2),
                       x: wp.array(dtype=wp.float32, ndim=2),
                       y: wp.array(dtype=wp.float32, ndim=2),
                       alpha: float, beta: float, reg: float,
                       AZt: wp.array3d(dtype=wp.float32), s: wp.array2d(dtype=wp.float32),
                       z: wp.array(dtype=wp.float32, ndim=2)):
        """SystemOperator's gather pass fused with the projection:
        z = beta·y + alpha·(Ax − AZ·s)."""
        w, c = wp.tid()
        Ax = float(0.0)
        if mask[w, c] > 0.0:
            b1 = bidx[w, c, 0]
            b2 = bidx[w, c, 1]
            if b1 >= 0:
                Ax += wp.dot(J[w, c, 0], dyn[w, b1])
            if b2 >= 0:
                Ax += wp.dot(J[w, c, 1], dyn[w, b2])
            Ax += (C[w, c] + reg) * x[w, c]
            for j in range(k):
                Ax -= AZt[w, j, c] * s[w, j]
        res = alpha * Ax
        if beta != 0.0:
            res += beta * y[w, c]
        z[w, c] = res

    @wp.kernel(enable_backward=False, module="unique", module_options=opts)
    def axpy_cols(v: wp.array2d(dtype=wp.float32), Mt: wp.array3d(dtype=wp.float32),
                  s: wp.array2d(dtype=wp.float32), sign: float, out: wp.array2d(dtype=wp.float32)):
        """out = v + sign·M s."""
        w, c = wp.tid()
        acc = v[w, c]
        for j in range(k):
            acc += sign * Mt[w, j, c] * s[w, j]
        out[w, c] = acc

    return NS(reduce_solve=reduce_solve, gram=gram, gather_project=gather_project, axpy_cols=axpy_cols)


@functools.cache
def chol_inv_kernel(k):
    """Kernel computing Einv = E⁻¹ for SPD k×k E, one thread per world.

    Generated as fully unrolled straight-line code with one scalar per
    matrix entry, so the factor stays in registers: Cholesky E = L Lᵀ,
    M = L⁻¹, Einv = Mᵀ M. Only the lower triangle of E is read. (A loop
    version keeps L in local memory: 1.2 ms at k = 18, W = 1; this one
    ~55 µs.) Warp needs kernel source in a file, so the module is written
    to a temp dir and imported.

    Untried opportunity: Warp's cooperative wp.tile_cholesky /
    wp.tile_cholesky_solve (one thread block per world, backed by MathDx
    cuSolverDx) could beat this single-thread kernel. It could not be
    tested here: MathDx's cuSOLVER backend errors on this GTX 1050
    (sm_61), and Warp offers no non-MathDx fallback on GPU while MathDx
    is enabled. Try it on a machine with a newer GPU (sm_70+).
    """
    lines = ["@wp.kernel(enable_backward=False)",
             f"def chol_inv_{k}(E: wp.array3d(dtype=wp.float32), Einv: wp.array3d(dtype=wp.float32)):",
             "    w = wp.tid()"]
    for i in range(k):
        for j in range(i + 1):
            lines.append(f"    a{i}_{j} = E[w, {i}, {j}]")
    for j in range(k):                                   # Cholesky, lower
        s = " + ".join(f"l{j}_{p} * l{j}_{p}" for p in range(j))
        lines.append(f"    l{j}_{j} = wp.sqrt(wp.max(a{j}_{j}{' - (' + s + ')' if s else ''}, 1.0e-30))")
        lines.append(f"    r{j} = 1.0 / l{j}_{j}")
        for i in range(j + 1, k):
            s = " + ".join(f"l{i}_{p} * l{j}_{p}" for p in range(j))
            lines.append(f"    l{i}_{j} = (a{i}_{j}{' - (' + s + ')' if s else ''}) * r{j}")
    for j in range(k):                                   # M = L⁻¹, lower
        lines.append(f"    m{j}_{j} = r{j}")
        for i in range(j + 1, k):
            s = " + ".join(f"l{i}_{p} * m{p}_{j}" for p in range(j, i))
            lines.append(f"    m{i}_{j} = -({s}) * r{i}")
    for i in range(k):                                   # Einv = Mᵀ M
        for j in range(i, k):
            lines.append(f"    e = {' + '.join(f'm{p}_{i} * m{p}_{j}' for p in range(j, k))}")
            lines.append(f"    Einv[w, {i}, {j}] = e")
            if i != j:
                lines.append(f"    Einv[w, {j}, {i}] = e")
    path = pathlib.Path(tempfile.gettempdir()) / "deflation_chol_gen" / f"chol_inv_{k}.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text("import warp as wp\n\n\n" + "\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location(f"chol_inv_{k}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, f"chol_inv_{k}")


class Deflation:
    """PA operator + setup + final correction, mirroring deflated_pcr.
    Zt: (W, k, N_c) — Z transposed, zero on inactive slots."""

    def __init__(self, A: SystemOperator, Zt: wp.array):
        self.A, self.Zt = A, Zt
        W, k, n = Zt.shape
        Nb = A.data.N_b
        self.W, self.n, self.k = W, n, k
        self.kk = k_kernels(k)
        f = lambda *shape: wp.zeros(shape, dtype=wp.float32, device=DEV)
        self.AZt, self.E, self.Einv = f(W, k, n), f(W, k, k), f(W, k, k)
        self.t, self.s, self.tb, self.zero_k = f(W, k), f(W, k), f(W, k), f(W, k)
        self.Pb = f(W, n)
        self.tmp_b = wp.zeros((W, k, Nb), dtype=wp.spatial_vector, device=DEV)
        self.dyn = wp.zeros((W, Nb), dtype=wp.spatial_vector, device=DEV)
        self.op = NS(matvec=self.matvec)

    def compute_AZ(self):
        d, W, n, k = self.A.data, self.W, self.n, self.k
        self.tmp_b.zero_()
        wp.launch(_az_scatter, dim=(W, k, n), inputs=[self.Zt, d.J_values, d.constraint_body_idx,
                  d.constraint_active_mask, d.body_inv_mass, d.world_inv_inertia, self.tmp_b], device=DEV)
        wp.launch(_az_gather, dim=(W, k, n), inputs=[self.tmp_b, self.Zt, d.J_values, d.constraint_body_idx,
                  d.constraint_active_mask, d.C_values, self.A.regularization, self.AZt], device=DEV)

    def _reduce_solve(self, Mt, v, base, sign, t_out):
        wp.launch_tiled(self.kk.reduce_solve, dim=[self.W], block_dim=BLOCK, device=DEV,
                        inputs=[Mt, v, base, sign, self.Einv, t_out, self.s])

    def setup(self, b):
        W, n, k = self.W, self.n, self.k
        self.compute_AZ()
        wp.launch_tiled(self.kk.gram, dim=[W, k], block_dim=BLOCK, device=DEV,
                        inputs=[self.Zt, self.AZt, self.E])
        wp.launch(chol_inv_kernel(k), dim=W, inputs=[self.E, self.Einv], device=DEV)
        self._reduce_solve(self.Zt, b, self.zero_k, 1.0, self.tb)        # s = E⁻¹ Zᵀb, tb = Zᵀb
        wp.launch(self.kk.axpy_cols, dim=(W, n), inputs=[b, self.AZt, self.s, -1.0, self.Pb], device=DEV)

    def matvec(self, x, y, z, alpha, beta):                                # z = beta·y + alpha·PA x
        d = self.A.data
        self._reduce_solve(self.AZt, x, self.zero_k, 1.0, self.t)         # s = E⁻¹ (AZ)ᵀ x = E⁻¹ Zᵀ A x
        self.dyn.zero_()
        wp.launch(kernel_invM_Jt_matvec_direct, dim=(self.W, self.n),
                  inputs=[x, d.J_values, d.constraint_body_idx, d.constraint_active_mask,
                          d.body_inv_mass, d.world_inv_inertia], outputs=[self.dyn], device=DEV)
        wp.launch(self.kk.gather_project, dim=(self.W, self.n), device=DEV,
                  inputs=[self.dyn, d.J_values, d.constraint_body_idx, d.constraint_active_mask,
                          d.C_values, x, y, alpha, beta, self.A.regularization, self.AZt, self.s, z])

    def finish(self, x):                                                   # x ← x + Z E⁻¹ (Zᵀb − AZᵀx)
        self._reduce_solve(self.AZt, x, self.tb, -1.0, self.t)
        wp.launch(self.kk.axpy_cols, dim=(self.W, self.n), inputs=[x, self.Zt, self.s, 1.0, x], device=DEV)


# ---------------------------------------------------------------- engine objects

def engine_objects(d, idx):
    """Real SystemOperator + Jacobi for worlds = dumped systems `idx`."""
    W = len(idx)
    arr = lambda key, dt, cast=None: wp.array(
        np.ascontiguousarray(d[key][idx, 0] if cast is None else d[key][idx, 0].astype(cast)),
        dtype=dt, device=DEV)
    J = arr("J_values", wp.spatial_vector)
    bidx = arr("constr_body_idx", wp.int32, np.int32)
    mask, C = arr("constr_active_mask", wp.float32), arr("C_values", wp.float32)
    minv, Iinv = arr("body_inv_mass", wp.float32), arr("world_inv_inertia", wp.mat33)
    b = arr("b", wp.float32)
    Nb = int(d["meta"][0]["N_b"])
    data = SystemLinearData(N_w=W, N_c=NC, N_b=Nb, J_values=J, constraint_body_idx=bidx,
                            constraint_active_mask=mask, C_values=C, body_inv_mass=minv,
                            world_inv_inertia=Iinv)
    A = SystemOperator(data, device=DEV, regularization=REG)
    eng = NS(dims=NS(N_w=W, N_c=NC, num_worlds=W, num_constraints=NC), device=DEV,
             ostrich_model=NS(body_inv_mass=minv),
             data=NS(world_inv_inertia=Iinv, J_values=NS(full=J), C_values=NS(full=C),
                     constr_body_idx=NS(full=bidx), constr_active_mask=NS(full=mask)))
    M = JacobiPreconditioner(eng, REG)
    M.update()
    return A, M, b


def time_graph(fn, warm_s=0.5, meas_s=1.0):
    """ms per replay of the captured graph. Time-based warm-up: the GPU only
    reaches full clocks under sustained load (a short warm-up overstated
    PCR's per-iteration cost by ~60% on the GTX 1050)."""
    with wp.ScopedCapture(device=DEV) as cap:
        fn()

    def run(seconds):
        n, t0 = 0, time.perf_counter()
        while True:
            for _ in range(10):
                wp.capture_launch(cap.graph)
            wp.synchronize()
            n += 10
            if time.perf_counter() - t0 >= seconds:
                return (time.perf_counter() - t0) / n * 1e3
    run(warm_s)
    return run(meas_s)


def time_torch(fn, warm_s=0.5, meas_s=1.0):
    """ms per call (eager torch), same time-based warm-up as time_graph."""
    def run(seconds):
        n, t0 = 0, time.perf_counter()
        while True:
            fn()
            torch.cuda.synchronize()
            n += 1
            if time.perf_counter() - t0 >= seconds:
                return (time.perf_counter() - t0) / n * 1e3
    run(warm_s)
    return run(meas_s)


# ---------------------------------------------------------------- (A) iterations

def iterations(S, d, va):
    s16, s8 = z_sources(S, d, va, 16), z_sources(S, d, va, 8)
    srcs = {"jacobi": ({}, 0), "joint/control": (s16["joint/control"], 18),
            "gcrodr k=16": (s16["gcrodr accumulated"], 16),
            "gcrodr k=8": (s8["gcrodr accumulated"], 8)}
    out = {}
    for sname, (tol, cap) in SETTINGS.items():
        for name, (Zs, k) in srcs.items():
            its, rr = [], []
            for i in va:
                A, b = S[i].A.float(), S[i].b.float()
                Z = Zs.get(i, torch.zeros(S[i].n, 0, dtype=DT)).float()
                x, it, _ = deflated_pcr(A, b, Z, jacobi_apply(A), max_iters=cap, tol=tol, atol=tol)
                its.append(it)
                rr.append(float(torch.linalg.norm(S[i].b - S[i].A @ x.double()) / torch.linalg.norm(S[i].b)))
            out[(sname, name)] = (np.array(its), np.array(rr), k)
    return out, s16


# ---------------------------------------------------------------- (B) timing

def correctness(d, S, va, s16):
    """Warp deflated solve vs offline deflated_pcr (float32, cap 16)."""
    acts = [np.nonzero(d["constr_active_mask"][s, 0] > 0.0)[0] for s in range(len(S))]
    bad = 0
    for i in va[::6]:
        Zl = s16["gcrodr accumulated"].get(i)
        if Zl is None or Zl.shape[1] == 0:
            continue
        A, M, b = engine_objects(d, np.array([i]))
        Zt = np.ascontiguousarray(embed(Zl, acts[i]).float().numpy().T[None])
        D = Deflation(A, wp.array(Zt, dtype=wp.float32, device=DEV))
        D.setup(b)
        bn, pbn = np.linalg.norm(b.numpy()), np.linalg.norm(D.Pb.numpy())
        x = wp.zeros((1, NC), dtype=wp.float32, device=DEV)
        solver = PCRSolver(max_iters=16, batch_dim=1, vec_dim=NC, device=DEV)
        solver.solve(D.op, D.Pb, x, M, iters=16, tol=1e-5 * bn / pbn, atol=1e-5)
        D.finish(x)
        it_w = int(solver.iter_count.numpy()[0])
        Af, bf = S[i].A.float(), S[i].b.float()
        xo, it_o, _ = deflated_pcr(Af, bf, Zl.float(), jacobi_apply(Af), max_iters=16)
        xw = torch.from_numpy(x.numpy()[0, acts[i]]).double()
        rw = float(torch.linalg.norm(S[i].b - S[i].A @ xw) / torch.linalg.norm(S[i].b))
        ro = float(torch.linalg.norm(S[i].b - S[i].A @ xo.double()) / torch.linalg.norm(S[i].b))
        bad += abs(it_w - it_o) > 1 or rw > 3 * max(ro, 1e-5)    # float32: both stop near the 1e-5 target
        print(f"    sys {i:3d}: iters warp {it_w:2d} / offline {it_o:2d}   rel res warp {rw:.1e} / offline {ro:.1e}")
    print(f"    → {'OK' if bad == 0 else f'{bad} MISMATCHES'}")


@wp.kernel
def _noop(a: wp.array(dtype=wp.float32, ndim=2)):
    w, c = wp.tid()


def timings(d):
    """ms: base[W] = (fixed, per_iter); defl[W][k] = (fixed + setup, per_iter);
    prim[W] = (A matvec, empty launch, A·Z, Cholesky-inverse; k = 18)."""
    base, defl, prim = {}, {}, {}
    for W in WORLDS:
        idx = np.arange(W) % len(d["b"])
        A, M, b = engine_objects(d, idx)
        solver = PCRSolver(max_iters=32, batch_dim=W, vec_dim=NC, device=DEV)
        x = wp.zeros((W, NC), dtype=wp.float32, device=DEV)

        def plain(n):
            x.zero_()
            solver.solve(A, b, x, M, iters=n, tol=-1.0, atol=-1.0)
        t4, t32 = time_graph(lambda: plain(4)), time_graph(lambda: plain(32))
        per = (t32 - t4) / 28
        base[W] = (t4 - 4 * per, per)
        out = wp.zeros((W, NC), dtype=wp.float32, device=DEV)
        rep20 = lambda f: time_graph(lambda: [f() for _ in range(20)], 0.2, 0.4) / 20
        D18 = Deflation(A, wp.array(np.zeros((W, 18, NC), np.float32), dtype=wp.float32, device=DEV))
        prim[W] = (rep20(lambda: A.matvec(b, b, out, 1.0, 0.0)),
                   rep20(lambda: wp.launch(_noop, dim=(W, NC), inputs=[out], device=DEV)),
                   rep20(D18.compute_AZ),
                   rep20(lambda: wp.launch(chol_inv_kernel(18), dim=W, inputs=[D18.E, D18.Einv], device=DEV)))
        defl[W] = {}
        rng = np.random.default_rng(0)
        for k in KS:
            Zt = np.linalg.qr(rng.standard_normal((W, NC, k)))[0].transpose(0, 2, 1)
            D = Deflation(A, wp.array(np.ascontiguousarray(Zt, dtype=np.float32), dtype=wp.float32, device=DEV))

            def deflated(n):
                x.zero_()
                D.setup(b)
                solver.solve(D.op, D.Pb, x, M, iters=n, tol=-1.0, atol=-1.0)
                D.finish(x)
            d4, d32 = time_graph(lambda: deflated(4)), time_graph(lambda: deflated(32))
            dper = (d32 - d4) / 28
            defl[W][k] = (d4 - 4 * dper, dper)
        print(f"  W={W:5d}  jacobi: fixed {base[W][0]:.3f} ms, {1e3*per:6.1f} µs/iter   deflation: "
              + "  ".join(f"k={k}: fixed+setup {defl[W][k][0]:.3f} ms, {1e3*defl[W][k][1]:6.1f} µs/iter"
                          for k in KS)
              + f"\n           primitives: A matvec {1e3*prim[W][0]:.1f} µs, empty launch {1e3*prim[W][1]:.1f} µs,"
              f" A·Z (k=18) {1e3*prim[W][2]:.1f} µs, Cholesky-inverse (k=18) {1e3*prim[W][3]:.1f} µs")
    return base, defl, prim


def gcrodr_zprod(d):
    """Per-solve cost of the GCRO-DR update (torch eager, GPU, float32):
    QR of [U ‖ 16 Krylov vecs], As·Cand, harmonic-Ritz eigenproblem, QR.
    The As·Cand product (k+16 columns) uses the Warp multi-column A·Z."""
    out = {}
    for W in WORLDS:
        idx = np.arange(W) % len(d["b"])
        A, _, _ = engine_objects(d, idx)
        out[W] = {}
        for k in (8, 16):
            m = k + GCRODR_M
            Cand = torch.randn(W, NC, m, device=DEV)
            Dg = torch.rand(W, NC, device=DEV) + 0.5
            D = Deflation(A, wp.array(np.ones((W, m, NC), np.float32), dtype=wp.float32, device=DEV))
            t_az = time_graph(D.compute_AZ)

            def harmonic():
                V = torch.linalg.qr(Cand)[0]
                Wm = V * Dg.unsqueeze(2)                                 # stands in for As·V (timed above)
                M1 = Wm.transpose(1, 2) @ Wm
                B = Wm.transpose(1, 2) @ V
                B = 0.5 * (B + B.transpose(1, 2)) + 1e-6 * torch.eye(m, device=DEV)
                L = torch.linalg.cholesky_ex(B)[0]
                Li = torch.linalg.solve_triangular(L, torch.eye(m, device=DEV).expand(W, m, m), upper=False)
                Cs = Li @ M1 @ Li.transpose(1, 2)
                _, U = torch.linalg.eigh(0.5 * (Cs + Cs.transpose(1, 2)))
                Y = V @ (Li.transpose(1, 2) @ U[:, :, :k])
                torch.linalg.qr(Dg.unsqueeze(2) * torch.linalg.qr(Y)[0])
            out[W][k] = t_az + time_torch(harmonic)
    return out


# ---------------------------------------------------------------- (C) estimate

def batch_iters(its, W, rng, draws=400):
    if W == 1:
        return float(its.mean())
    return float(np.mean([its[rng.integers(0, len(its), W)].max() for _ in range(draws)]))


def main():
    wp.init()
    print("(B) GPU timing — real engine PCR graph + Warp deflation kernels (same graph), float32")
    d0 = np.load(SCENES["obstacle"], allow_pickle=True)
    base, defl, _ = timings(d0)
    zprod = gcrodr_zprod(d0)
    print("  GCRO-DR Z update per solve (torch eager + Warp A·Cand): "
          + "  ".join(f"W={W}: k=8 {zprod[W][8]:.2f} ms, k=16 {zprod[W][16]:.2f} ms" for W in WORLDS))

    for label, path in SCENES.items():
        S = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        va = list(range(int(0.8 * len(S)), len(S)))
        print(f"\n===== {label}: held-out {len(va)} systems =====")
        it, s16 = iterations(S, d, va)
        print("  correctness, Warp vs offline (GCRO-DR Z, float32, cap 16):")
        correctness(d, S, va, s16)

        for sname, (tol, cap) in SETTINGS.items():
            print(f"\n  [{sname}]  (A) iterations: median p90 max #capped | rel-res p90")
            for name in ("jacobi", "joint/control", "gcrodr k=16", "gcrodr k=8"):
                its, rr, k = it[(sname, name)]
                print(f"    {name:14s} {pct(its,50):3.0f} {pct(its,90):3.0f} {its.max():3d} {int((its >= cap).sum()):3d}"
                      f" | {pct(rr,90):.0e}")
            print("    (C) time per solve, ms:  fixed+setup + Z build + iterations × per-iteration = total"
                  "  (Δ vs Jacobi; ceiling = zero overhead)")
            jits = it[(sname, "jacobi")][0]
            for W in WORLDS:
                rng = np.random.default_rng(1)
                nj = batch_iters(jits, W, rng)
                fx, per = base[W]
                tj = fx + nj * per
                print(f"    W={W:<5d} {'jacobi':14s} {fx:6.3f} + {0:6.3f} + {nj:4.1f} × {per:6.4f} = {tj:7.3f}")
                for name in ("joint/control", "gcrodr k=16", "gcrodr k=8"):
                    its, _, k = it[(sname, name)]
                    n = batch_iters(its, W, np.random.default_rng(1))
                    fs, dper = defl[W][k]
                    zb = zprod[W][k] if name.startswith("gcrodr") else 0.0
                    tot = fs + zb + n * dper
                    print(f"    {'':7s} {name:14s} {fs:6.3f} + {zb:6.3f} + {n:4.1f} × {dper:6.4f} = {tot:7.3f}"
                          f"  ({100 * (tot / tj - 1):+5.0f}%; ceiling {100 * ((fx + n * per) / tj - 1):+4.0f}%)")


if __name__ == "__main__":
    main()
