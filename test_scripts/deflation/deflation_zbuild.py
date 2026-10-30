"""Warp kernels that build Z for the spectral sources, and their GPU timing (W = 1).

Deflation only depends on span(Z), so no kernel computes individual eigenvectors.

bottom_k        refines a block X (NP×K, scaled space) towards the bottom-K invariant
                subspace of a symmetric M: Chebyshev-filtered subspace iteration
                (degree DEG on [a, b], a = λ_max of the Rayleigh quotient XᵀMX by
                warm-started power iteration, clamped to b/2; b = Gershgorin bound),
                followed by shifted CholeskyQR3. Products with M are chunked over RB
                rows, so M stays in global memory. One thread block per world.
scale_orth      harmonic m = K: Z = CholeskyQR(D·V). With m = K the harmonic-Ritz
                vectors span the whole Krylov space, so span(Z) = span(D·V) and no
                eigenproblem is needed.
harmonic_*      harmonic m > K: B = VᵀW, M1 = WᵀW (W = As·V, reused from the previous
                solve), L = chol(B), C = L⁻¹ M1 L⁻ᵀ; bottom_k on C (to tolerance);
                G = L⁻ᵀ X, Z = CholeskyQR(D·V G).

The exact sources need the bottom-16 eigenvectors, but the 16/17 boundary lies in the
bulk of the scaled spectrum (λ16 ≈ 0.3–1), where no cheap method converges and where
the choice of vectors does not affect DPCR. They are therefore built by CHAINED
refinement: each update applies s filter steps to the previous Z (recycle-exact: per
system, from the previous system's Z; cross-step: once per time step, on the last
system of the previous step, from the previous cross-step Z). s is the smallest value
in STEP_CHOICES whose DPCR iterations over the held-out systems are no worse than with
the exact eigenvectors (median no worse, 90th percentile within 10%).

Run (selection, DPCR checks against float64 references, GPU timing):
    python -m test_scripts.deflation.deflation_zbuild
"""


import functools

import numpy as np
import torch
import warp as wp

from test_scripts.deflation.deflation import DT, SCENES, deflated_pcr
from test_scripts.deflation.deflation_cost import time_graph
from test_scripts.ideal_target_stability import bad_subspace
from test_scripts.precond_lab import jacobi_apply, load_systems
from test_scripts.recycle_approx import harmonic_ritz, krylov_residual_basis
from test_scripts.recycle_probe import pct, restrict

DEV = "cuda:0"
wp.set_module_options({"max_unroll": 1})   # unrolled loops would give every iteration its own shared tiles
K = 16
RB = 8             # rows per chunk of a product with M (shared memory: 4 chunks live at once)
DEG = 8            # Chebyshev filter degree
MAXIT = 60         # outer iterations of bottom_k
TOL = 1e-3         # bottom_k stops at ‖MX − XH‖_F ≤ TOL·b, or when the residual stagnates
POWER_ITS = 4      # power-iteration steps per outer iteration (warm-started)
TOL_SMALL = 3e-4   # tolerance for the m×m harmonic problem: loosest value whose DPCR iterations match
BLOCK = 128


def padded_size(n):
    return max(RB, -(-n // RB) * RB)


# ---------------------------------------------------------------- kernels

@functools.cache
def bottom_k_kernel(NP):
    NCH = NP // RB

    @wp.func
    def product(M: wp.array2d(dtype=wp.float32), x: wp.tile(dtype=wp.float32, shape=(NP, K), storage="shared"),
                out: wp.tile(dtype=wp.float32, shape=(NP, K), storage="shared")):
        for r in range(NCH):
            mc = wp.tile_load(M, shape=(RB, NP), offset=(r * RB, 0), storage="shared")
            wp.tile_assign(out, wp.tile_matmul(mc, x), offset=(r * RB, 0))

    @wp.func
    def orthonormalise(y: wp.tile(dtype=wp.float32, shape=(NP, K), storage="shared"), shift: float):
        """One Cholesky-QR pass, y ← y L⁻ᵀ with L Lᵀ = yᵀy + shift·‖y‖_F² I."""
        g = wp.tile_matmul(wp.tile_transpose(y), y)
        sh = shift * wp.tile_extract(wp.tile_sum(wp.tile_map(wp.mul, y, y)), 0)
        L = wp.tile_cholesky(wp.tile_diag_add(g, wp.tile_full(shape=K, value=sh, dtype=wp.float32)))
        for r in range(NCH):
            yv = wp.tile_view(y, offset=(r * RB, 0), shape=(RB, K))
            wp.tile_assign(y, wp.tile_transpose(wp.tile_lower_solve(L, wp.tile_transpose(yv))), offset=(r * RB, 0))

    @wp.kernel(enable_backward=False)
    def bottom_k(M: wp.array3d(dtype=wp.float32), X0: wp.array3d(dtype=wp.float32), tol: float, power_its: int, maxit: int,
                 X: wp.array3d(dtype=wp.float32), its: wp.array(dtype=wp.int32), res: wp.array(dtype=wp.float32)):
        w = wp.tid()
        b = float(0.0)                                   # Gershgorin bound of M
        for r in range(NCH):
            rows = wp.tile_sum(wp.tile_map(wp.abs, wp.tile_load(M[w], shape=(RB, NP), offset=(r * RB, 0))), axis=1)
            b = wp.max(b, wp.tile_extract(wp.tile_max(rows), 0))
        x = wp.tile_zeros(shape=(NP, K), dtype=wp.float32, storage="shared")
        y = wp.tile_zeros(shape=(NP, K), dtype=wp.float32, storage="shared")
        my = wp.tile_zeros(shape=(NP, K), dtype=wp.float32, storage="shared")
        yp = wp.tile_zeros(shape=(NP, K), dtype=wp.float32, storage="register")
        wp.tile_assign(x, wp.tile_load(X0[w], shape=(NP, K)), offset=(0, 0))
        v = wp.tile_ones(shape=(K, 1), dtype=wp.float32, storage="shared")
        n_it = maxit
        rel = float(0.0)
        prev = float(1e30)
        for it in range(maxit):
            product(M[w], x, my)
            h = wp.tile_matmul(wp.tile_transpose(x), my)
            rs = float(0.0)                              # ‖MX − XH‖_F, chunk by chunk
            for r in range(NCH):
                d = wp.tile_view(my, offset=(r * RB, 0), shape=(RB, K)) - wp.tile_matmul(
                    wp.tile_view(x, offset=(r * RB, 0), shape=(RB, K)), h)
                rs += wp.tile_extract(wp.tile_sum(wp.tile_map(wp.mul, d, d)), 0)
            rel = wp.sqrt(rs) / b
            if rel <= tol or (it >= 3 and rel > 0.9 * prev and rel < 20.0 * tol):  # converged / at the float32 floor
                n_it = it
                break
            prev = rel
            for p in range(power_its):                   # a = λ_max(H), warm-started power iteration
                wp.tile_assign(v, wp.tile_matmul(h, v), offset=(0, 0))
                nv = wp.sqrt(wp.tile_extract(wp.tile_sum(wp.tile_map(wp.mul, v, v)), 0))
                wp.tile_assign(v, v * (1.0 / nv), offset=(0, 0))
            a = wp.min(wp.tile_extract(wp.tile_matmul(wp.tile_transpose(v), wp.tile_matmul(h, v)), 0, 0),
                       0.5 * b)                    # a wide interval keeps T_DEG within float32
            c = 0.5 * (a + b)
            e = 0.5 * (b - a) + 1e-20
            wp.tile_assign(yp, x, offset=(0, 0))         # Chebyshev filter of degree DEG on [a, b]
            wp.tile_assign(y, (my - x * c) * (1.0 / e), offset=(0, 0))
            for j in range(DEG - 1):
                product(M[w], y, my)
                wp.tile_assign(my, (my - y * c) * (2.0 / e) - yp, offset=(0, 0))
                wp.tile_assign(yp, y, offset=(0, 0))
                wp.tile_assign(y, my, offset=(0, 0))
            orthonormalise(y, 1e-3)                      # shifted CholeskyQR3: the filter can make the
            orthonormalise(y, 0.0)                       # columns nearly dependent (Gram condition ~1e8,
            orthonormalise(y, 0.0)                       # beyond float32); the shift handles that
            wp.tile_assign(x, y, offset=(0, 0))
        wp.tile_store(X[w], x)
        its[w] = n_it
        res[w] = rel
    return bottom_k


@functools.cache
def scale_orth_kernel(NP):
    @wp.kernel(enable_backward=False)
    def scale_orth(V: wp.array3d(dtype=wp.float32), D: wp.array3d(dtype=wp.float32), Z: wp.array3d(dtype=wp.float32)):
        w = wp.tid()
        z = wp.tile_zeros(shape=(NP, K), dtype=wp.float32, storage="shared")
        wp.tile_assign(z, wp.tile_map(wp.mul, wp.tile_load(V[w], shape=(NP, K)),
                                      wp.tile_broadcast(wp.tile_load(D[w], shape=(NP, 1)), shape=(NP, K))),
                       offset=(0, 0))
        for q in range(2):
            L = wp.tile_cholesky(wp.tile_matmul(wp.tile_transpose(z), z))
            wp.tile_assign(z, wp.tile_transpose(wp.tile_lower_solve(L, wp.tile_transpose(z))), offset=(0, 0))
        wp.tile_store(Z[w], z)
    return scale_orth


@functools.cache
def harmonic_kernels(NP, m):
    @wp.kernel(enable_backward=False)
    def prep(V: wp.array3d(dtype=wp.float32), Wm: wp.array3d(dtype=wp.float32),
             Lout: wp.array3d(dtype=wp.float32), Cout: wp.array3d(dtype=wp.float32)):
        w = wp.tid()
        B = wp.tile_zeros(shape=(m, m), dtype=wp.float32)
        M1 = wp.tile_zeros(shape=(m, m), dtype=wp.float32)
        for r in range(NP // RB):
            vc = wp.tile_load(V[w], shape=(RB, m), offset=(r * RB, 0))
            wc = wp.tile_load(Wm[w], shape=(RB, m), offset=(r * RB, 0))
            wp.tile_matmul(wp.tile_transpose(wc), vc, B)
            wp.tile_matmul(wp.tile_transpose(wc), wc, M1)
        Bs = (B + wp.tile_transpose(B)) * 0.5
        L = wp.tile_cholesky(wp.tile_diag_add(Bs, wp.tile_full(shape=m, value=1e-7, dtype=wp.float32)))
        T = wp.tile_lower_solve(L, M1)                    # L⁻¹ M1
        C = wp.tile_transpose(wp.tile_lower_solve(L, wp.tile_transpose(T)))
        wp.tile_store(Lout[w], L)
        wp.tile_store(Cout[w], (C + wp.tile_transpose(C)) * 0.5)

    @wp.kernel(enable_backward=False)
    def finish(V: wp.array3d(dtype=wp.float32), D: wp.array3d(dtype=wp.float32), L: wp.array3d(dtype=wp.float32),
               X: wp.array3d(dtype=wp.float32), Z: wp.array3d(dtype=wp.float32)):
        w = wp.tid()
        G = wp.tile_upper_solve(wp.tile_transpose(wp.tile_load(L[w], shape=(m, m))), wp.tile_load(X[w], shape=(m, K)))
        z = wp.tile_zeros(shape=(NP, K), dtype=wp.float32, storage="shared")
        for r in range(NP // RB):
            vc = wp.tile_load(V[w], shape=(RB, m), offset=(r * RB, 0))
            dc = wp.tile_broadcast(wp.tile_load(D[w], shape=(RB, 1), offset=(r * RB, 0)), shape=(RB, K))
            wp.tile_assign(z, wp.tile_map(wp.mul, wp.tile_matmul(vc, G), dc), offset=(r * RB, 0))
        for q in range(2):
            Lz = wp.tile_cholesky(wp.tile_matmul(wp.tile_transpose(z), z))
            wp.tile_assign(z, wp.tile_transpose(wp.tile_lower_solve(Lz, wp.tile_transpose(z))), offset=(0, 0))
        wp.tile_store(Z[w], z)
    return prep, finish


# ---------------------------------------------------------------- host side

def scaled(A):
    D = torch.diagonal(A).rsqrt()
    return D, D.unsqueeze(1) * A * D.unsqueeze(0)


def pad_matrix(As, NP):
    """NP×NP float32: As, then padding rows with diagonal = the Gershgorin bound of As
    (eigenvalue at the top of the spectrum, so it never enters the bottom-K span)."""
    n = As.shape[0]
    out = np.zeros((NP, NP), np.float32)
    out[:n, :n] = As.numpy()
    out[np.arange(n, NP), np.arange(n, NP)] = float(As.abs().sum(1).max())
    return out


def pad_block(X, NP):
    out = np.zeros((NP, X.shape[1]), np.float32)
    out[:X.shape[0]] = X.numpy()
    return out


def warm_start(X, n, NP, seed=0):
    """Previous Z (scaled space, n×k' with k' ≤ K after restriction to the current
    rows) as an orthonormal NP×K block; missing columns are filled with random vectors."""
    out = np.zeros((NP, K))
    out[:n, :X.shape[1]] = X.numpy()
    rng = np.random.default_rng(seed)
    out[:n, X.shape[1]:] = rng.standard_normal((n, K - X.shape[1]))
    out[:n] = np.linalg.qr(out[:n])[0]
    return out.astype(np.float32)


def run_bottom_k(Mpad, X0pad, NP, tol=TOL, maxit=MAXIT, timed=False):
    kern = bottom_k_kernel(NP)
    M = wp.array(Mpad[None], dtype=wp.float32, device=DEV)
    X0 = wp.array(X0pad[None], dtype=wp.float32, device=DEV)
    X = wp.zeros((1, NP, K), dtype=wp.float32, device=DEV)
    its = wp.zeros(1, dtype=wp.int32, device=DEV)
    res = wp.zeros(1, dtype=wp.float32, device=DEV)
    launch = lambda: wp.launch_tiled(kern, dim=[1], inputs=[M, X0, tol, POWER_ITS, maxit, X, its, res],
                                     block_dim=BLOCK, device=DEV)
    launch()
    out = (X.numpy()[0], (int(its.numpy()[0]), float(res.numpy()[0])))
    if timed:
        return out + (time_graph(launch, warm_s=0.1, meas_s=0.25),)
    return out


def run_harmonic(V, Wm, D, NP, m, timed=False):
    """V, Wm: NP×m (scaled space), D: NP. Returns Z (NP×K, original space) [, ms]."""
    vw = wp.array(V[None], dtype=wp.float32, device=DEV)
    dw = wp.array(D.reshape(1, NP, 1), dtype=wp.float32, device=DEV)
    Z = wp.zeros((1, NP, K), dtype=wp.float32, device=DEV)
    if m == K:
        kern = scale_orth_kernel(NP)
        launch = lambda: wp.launch_tiled(kern, dim=[1], inputs=[vw, dw, Z], block_dim=BLOCK, device=DEV)
    else:
        prep, finish = harmonic_kernels(NP, m)
        ww = wp.array(Wm[None], dtype=wp.float32, device=DEV)
        L = wp.zeros((1, m, m), dtype=wp.float32, device=DEV)
        C = wp.zeros((1, m, m), dtype=wp.float32, device=DEV)
        X0 = wp.array(np.linalg.qr(np.random.default_rng(1).standard_normal((m, K)))[0][None].astype(np.float32), device=DEV)
        X = wp.zeros((1, m, K), dtype=wp.float32, device=DEV)
        its = wp.zeros(1, dtype=wp.int32, device=DEV)
        res = wp.zeros(1, dtype=wp.float32, device=DEV)
        small = bottom_k_kernel(m)

        def launch():
            wp.launch_tiled(prep, dim=[1], inputs=[vw, ww, L, C], block_dim=BLOCK, device=DEV)
            wp.launch_tiled(small, dim=[1], inputs=[C, X0, TOL_SMALL, POWER_ITS, MAXIT, X, its, res],
                            block_dim=BLOCK, device=DEV)
            wp.launch_tiled(finish, dim=[1], inputs=[vw, dw, L, X, Z], block_dim=BLOCK, device=DEV)
    launch()
    out = Z.numpy()[0]
    if timed:
        return out, time_graph(launch, warm_s=0.1, meas_s=0.25)
    return out


# ---------------------------------------------------------------- per-scene measurement

STEP_CHOICES = (1, 2, 3)
TIME_EVERY = 5     # time every 5th refinement of a chain (timing is the slow part)


def step_structure(d, n_sys):
    step_idx = d["step_idx"]
    last_of_step = {}
    for i in range(n_sys):
        last_of_step[int(step_idx[i])] = i
    return step_idx, [last_of_step[s] for s in sorted(last_of_step)]


class Scene:
    def __init__(self, path):
        self.S = load_systems(path=path, dtype=DT)
        d = np.load(path, allow_pickle=True)
        self.va = list(range(int(0.8 * len(self.S)), len(self.S)))
        self.acts = [np.nonzero(d["constr_active_mask"][s, 0] > 0.0)[0] for s in range(len(self.S))]
        self.NP = padded_size(max(s.n for s in self.S))
        self.step_idx, self.lasts = step_structure(d, len(self.S))
        self._exact = {}

    def exact(self, i):
        """Exact bottom-K basis of system i: (scaled space, original space)."""
        if i not in self._exact:
            D, _ = scaled(self.S[i].A)
            Zo = bad_subspace(self.S[i].A, min(K, self.S[i].n))
            self._exact[i] = (torch.linalg.qr(Zo / D.unsqueeze(1))[0], Zo)
        return self._exact[i]

    def carry(self, Z, src, dst):
        """Z on system src's rows → system dst's rows."""
        return restrict(_embed(Z, self.acts[src]), self.acts[dst])

    def refine(self, X, src, dst, steps, timed):
        """s filter steps on system dst, warm-started from X (scaled space of src)."""
        D, As = scaled(self.S[dst].A)
        X0 = warm_start(self.carry(X, src, dst), self.S[dst].n, self.NP)
        r = run_bottom_k(pad_matrix(As, self.NP), X0, self.NP, tol=-1.0, maxit=steps, timed=timed)
        X = torch.from_numpy(r[0][:self.S[dst].n]).double()
        return X, D.unsqueeze(1) * X, (r[2] if timed else None)


def recycle_chain(sc, steps):
    """Per system i: refine the previous system's Z. Returns (DPCR iters, ms samples)."""
    p = sc.va[0] - 1
    X, prev = sc.exact(p)[0], p
    its, ms = [], []
    for t, i in enumerate(sc.va):
        X, Z, m = sc.refine(X, prev, i, steps, timed=(t % TIME_EVERY == 0))
        prev = i
        its.append(_iters(sc.S[i], Z))
        if m is not None:
            ms.append(m)
    return its, ms


def cross_chain(sc, steps):
    """Once per time step N: refine the previous cross-step Z on the last system of step
    N-1; used by every system of step N. Returns (DPCR iters, ms per system samples)."""
    first = int(sc.step_idx[sc.va[0]])
    chain = [l for l in sc.lasts if int(sc.step_idx[l]) < first][-2:] + \
            [l for l in sc.lasts if int(sc.step_idx[l]) >= first]
    per_step = dict(zip(*np.unique(sc.step_idx, return_counts=True)))
    X, prev = sc.exact(chain[0])[0], chain[0]
    Zstep, ms = {}, []
    for l in chain[1:]:
        X, Z, m = sc.refine(X, prev, l, steps, timed=True)
        prev = l
        Zstep[int(sc.step_idx[l]) + 1] = (l, Z)
        ms.append(m / per_step[int(sc.step_idx[l]) + 1] if int(sc.step_idx[l]) + 1 in per_step else m)
    its = [_iters(sc.S[i], sc.carry(*reversed(Zstep[int(sc.step_idx[i])]), i)) for i in sc.va]
    return its, ms


def _reference(sc, source):
    if source == "recycle-exact (prev)":
        return [_iters(sc.S[i], sc.carry(sc.exact(i - 1)[1], i - 1, i)) for i in sc.va]
    out = []
    for i in sc.va:
        prev = [l for l in sc.lasts if int(sc.step_idx[l]) == int(sc.step_idx[i]) - 1][0]
        out.append(_iters(sc.S[i], sc.carry(sc.exact(prev)[1], prev, i)))
    return out


def harmonic_build(sc, m, sample=12):
    """Harmonic m as the Figure 2 source: built on the previous system p from its Krylov
    basis V and W = As_p·V (reused from p's solve), carried to system i."""
    picks = [sc.va[j] for j in np.linspace(0, len(sc.va) - 1, sample).astype(int)]
    its, its_ref, ms = [], [], []
    for i in picks:
        p = i - 1
        Dp, Asp = scaled(sc.S[p].A)
        torch.manual_seed(0)
        V = krylov_residual_basis(Asp, torch.randn(sc.S[p].n, dtype=DT), m)
        Zw, t = run_harmonic(pad_block(V, sc.NP), pad_block(Asp @ V, sc.NP),
                             pad_block(Dp.unsqueeze(1), sc.NP)[:, 0], sc.NP, m, timed=True)
        Zw = torch.from_numpy(Zw[:sc.S[p].n]).double()
        Zref = Dp.unsqueeze(1) * harmonic_ritz(Asp, V, K)
        its.append(_iters(sc.S[i], sc.carry(Zw, p, i)))
        its_ref.append(_iters(sc.S[i], sc.carry(Zref, p, i)))
        ms.append(t)
    return its, its_ref, ms


def measure_scene(path, verbose=True):
    """GPU ms per system to build Z with the Warp kernels: {source: ms}."""
    sc = Scene(path)
    out = {}
    for source, chain in (("recycle-exact (prev)", recycle_chain), ("cross-step (N-1)", cross_chain)):
        ref = _reference(sc, source)
        for steps in STEP_CHOICES:
            its, ms = chain(sc, steps)
            ok = pct(its, 50) <= pct(ref, 50) and pct(its, 90) <= 1.1 * pct(ref, 90)   # GPU reductions are
            # not bitwise reproducible and the chain carries that forward, so the p90 moves by ~1 system
            if ok or steps == STEP_CHOICES[-1]:
                break
        out[source] = float(np.median(ms))
        if verbose:
            print(f"    {source:22s} {steps} filter step(s): DPCR p50/p90 {pct(its, 50):.0f}/{pct(its, 90):.0f}"
                  f" (exact {pct(ref, 50):.0f}/{pct(ref, 90):.0f}){'' if ok else '  NOT MATCHED'}"
                  f" | {out[source]:.3f} ms per system")
    for m in (16, 32):
        its, ref, ms = harmonic_build(sc, m)
        out[f"harmonic m={m}"] = float(np.median(ms))
        if verbose:
            print(f"    harmonic m={m:<12d} DPCR p50/p90 {pct(its, 50):.0f}/{pct(its, 90):.0f}"
                  f" (float64 reference {pct(ref, 50):.0f}/{pct(ref, 90):.0f}) | {out[f'harmonic m={m}']:.3f} ms per system")
    return out


def _embed(X, act):
    out = torch.zeros(402, X.shape[1], dtype=X.dtype)
    out[torch.as_tensor(act)] = X
    return out


def _iters(s, Z):
    return deflated_pcr(s.A, s.b, Z.to(DT), jacobi_apply(s.A))[1]


def main():
    wp.init()
    for label, path in SCENES.items():
        print(f"===== {label}")
        measure_scene(path)


if __name__ == "__main__":
    main()
