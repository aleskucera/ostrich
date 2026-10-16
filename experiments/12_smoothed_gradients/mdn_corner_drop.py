"""Mixture density model of the corner drop's outcome.

Learns p(x(T) | theta) for drops from x0 ~ N(theta, SIGMA) as a mixture of K
Gaussians whose weights, means and widths are outputs of a small MLP of
theta. This is an outcome-level model (the noisy simulation end to end), not
yet a step-by-step simulator. Under the model the expected loss is closed
form,

    E[L](theta) = sum_k w_k [(mu_k - X_TARGET)^2 + s_k^2],

so its gradient includes the mode-weight term sum_k L_k dw_k/dtheta
explicitly; that is the term the simulator's adjoint cannot see.

Training data: theta ~ U(THETA_RANGE), x0 = theta + SIGMA eps, and x(T) from
the dense grid of simulated drops in results/corner_drop.npz (nearest grid
point, within half the 1e-4 m spacing), so any training-set size is free.
References (true E[L], its gradient, the right-mode weight) are quadratures
on the same grid.

Run: PYTHONPATH=. python mdn_corner_drop.py   (after corner_drop_study.py)
Writes results/mdn_corner_drop.npz and results/mdn_corner_drop.png.
"""

import math
import time

import numpy as np
import torch

import ball_landscape as bl
import corner_drop_study as cs

K = 3
THETA_RANGE = (-0.10, 0.15)
SIZES = (500, 2000, 8000, 32000)
EVAL_THETAS = np.linspace(-0.08, 0.12, 81)


class MDN(torch.nn.Module):
    def __init__(self, k=K, width=64):
        super().__init__()
        self.k = k
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 3 * k),
        )

    def forward(self, theta):
        """theta (B,) in metres -> weights, means, stds (B, K)."""
        out = self.net((theta[:, None] - 0.025) / 0.07)
        logits, mu, log_s = out.split(self.k, dim=1)
        return torch.softmax(logits, dim=1), 2.0 * mu, torch.exp(log_s).clamp(min=1e-3)

    def nll(self, theta, x):
        w, mu, s = self(theta)
        logp = -0.5 * ((x[:, None] - mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
        return -torch.logsumexp(torch.log(w) + logp, dim=1).mean()

    def expected_loss(self, theta, target):
        w, mu, s = self(theta)
        L_k = (mu - target) ** 2 + s**2
        return (w * L_k).sum(dim=1), w, mu, s, L_k


class ModeMDN(torch.nn.Module):
    """Mode-structured mixture: a classifier for the right-mode weight w_R(theta)
    and, per mode, K_MODE Gaussians whose means are kept on that mode's side
    (right: mu > 0, left: mu < 0). Trained with the mode label of each sample
    (the side x(T) ends on): -log w_mode(theta) - log p_mode(x | theta). The
    mode-weight term of the gradient then comes only from the classifier."""

    K_MODE = 2

    def __init__(self, width=64):
        super().__init__()
        k = self.K_MODE
        self.k = 2 * k
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1 + 2 * 3 * k),
        )

    def parts(self, theta):
        out = self.net((theta[:, None] - 0.025) / 0.07)
        k = self.K_MODE
        w_right = torch.sigmoid(out[:, 0])
        sides = []
        for j, sign in ((0, -1.0), (1, 1.0)):  # left, right
            o = out[:, 1 + j * 3 * k : 1 + (j + 1) * 3 * k]
            logits, mu, log_s = o.split(k, dim=1)
            sides.append((torch.softmax(logits, dim=1), sign * torch.nn.functional.softplus(2.0 * mu + 1.5),
                          torch.exp(log_s - 2.0).clamp(min=1e-3)))
        return w_right, sides

    def forward(self, theta):
        """As MDN.forward: weights, means, stds over all 2 K_MODE components."""
        w_right, ((wl, ml, sl), (wr, mr, sr)) = self.parts(theta)
        w = torch.cat([(1 - w_right)[:, None] * wl, w_right[:, None] * wr], dim=1)
        return w, torch.cat([ml, mr], dim=1), torch.cat([sl, sr], dim=1)

    def nll(self, theta, x):
        w_right, sides = self.parts(theta)
        right = x > 0
        nll = -torch.where(right, torch.log(w_right), torch.log1p(-w_right))
        for j, (w, mu, s) in enumerate(sides):
            logp = -0.5 * ((x[:, None] - mu) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
            side_nll = -torch.logsumexp(torch.log(w) + logp, dim=1)
            nll = nll + torch.where(right if j == 1 else ~right, side_nll, 0.0)
        return nll.mean()

    expected_loss = MDN.expected_loss


def sample_data(grid, n, rng):
    x, xT = grid
    theta = rng.uniform(*THETA_RANGE, n)
    x0 = theta + cs.SIGMA * rng.standard_normal(n)
    idx = np.clip(np.rint((x0 - x[0]) / cs.GRID_STEP).astype(int), 0, len(x) - 1)
    return torch.tensor(theta), torch.tensor(xT[idx])


def train(theta, xT, steps=6000, seed=0, structured=False):
    torch.manual_seed(seed)
    model = (ModeMDN() if structured else MDN()).double()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(theta), (min(1024, len(theta)),), generator=gen)
        loss = model.nll(theta[i], xT[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def model_curves(model, thetas):
    """E[L], its gradient, its two parts and the right-mode weight on thetas."""
    th = torch.tensor(thetas, requires_grad=True)
    E, w, mu, s, L_k = model.expected_loss(th, cs.X_TARGET)
    grad = torch.autograd.grad(E.sum(), th, retain_graph=True)[0]
    # Split: sum_k w_k dL_k/dtheta (within) + sum_k L_k dw_k/dtheta (mode weights)
    nk = w.shape[1]
    dL_k = torch.stack([torch.autograd.grad(L_k[:, k].sum(), th, retain_graph=True)[0] for k in range(nk)], dim=1)
    dw_k = torch.stack([torch.autograd.grad(w[:, k].sum(), th, retain_graph=True)[0] for k in range(nk)], dim=1)
    within = (w * dL_k).sum(dim=1)
    weights = (L_k * dw_k).sum(dim=1)
    # Right-mode weight: mixture mass on x > 0.
    w_right = (w * 0.5 * (1 + torch.erf(mu / (s * math.sqrt(2))))).sum(dim=1)
    det = lambda t: t.detach().numpy()
    return {"E": det(E), "grad": det(grad), "within": det(within), "weights": det(weights), "w_right": det(w_right)}


def references(grid, thetas):
    x, xT = grid
    d = np.load(bl.RESULTS / "corner_drop.npz")
    L = d["L"]
    out = {"E": [], "grad": [], "w_right": []}
    for th in thetas:
        p = cs.gaussian(x, th, cs.SIGMA) * cs.GRID_STEP
        out["E"].append((L * p).sum())
        out["grad"].append((L * (x - th) / cs.SIGMA**2 * p).sum())
        out["w_right"].append((p * (xT > 0)).sum())
    return {k: np.array(v) for k, v in out.items()}


def zeroth_order_error(grid, n_sims, rng, thetas, ref_grad, repeats=100):
    """Median relative error of zeroth-order (antithetic) at each theta with
    n_sims simulations per estimate (n_sims / 2 pairs)."""
    x, _ = grid
    L = np.load(bl.RESULTS / "corner_drop.npz")["L"]
    errs = []
    for th, true in zip(thetas, ref_grad):
        eps = rng.standard_normal((repeats, max(n_sims // 2, 1)))
        ip = np.clip(np.rint((th + cs.SIGMA * eps - x[0]) / cs.GRID_STEP).astype(int), 0, len(x) - 1)
        im = np.clip(np.rint((th - cs.SIGMA * eps - x[0]) / cs.GRID_STEP).astype(int), 0, len(x) - 1)
        est = ((L[ip] - L[im]) * eps).mean(axis=1) / (2 * cs.SIGMA)
        errs.append(np.median(np.abs(est - true)) / max(abs(true), 1.0))
    return np.array(errs)


def main():
    d = np.load(bl.RESULTS / "corner_drop.npz")
    grid = (d["x"], d["xT"])
    rng = np.random.default_rng(0)
    ref = references(grid, EVAL_THETAS)
    scale = np.abs(ref["grad"]).max()
    results = {"thetas": EVAL_THETAS, **{f"ref_{k}": v for k, v in ref.items()}}
    print(f"{'model':>10} {'train sims':>10} {'NLL':>7} {'|w_right err|':>14} {'E[L] rel err':>13} "
          f"{'grad err median':>16} {'grad err max':>13}   (gradient errors relative to max|grad|)")
    for tag, structured in (("mdn", False), ("smdn", True)):
        for n in SIZES:
            theta, xT = sample_data(grid, n, rng)
            t0 = time.time()
            model = train(theta, xT, structured=structured)
            th_val, x_val = sample_data(grid, 20000, np.random.default_rng(99))
            with torch.no_grad():
                nll = model.nll(th_val, x_val).item()
            c = model_curves(model, EVAL_THETAS)
            for k, v in c.items():
                results[f"{tag}{n}_{k}"] = v
            err = np.abs(c["grad"] - ref["grad"]) / scale
            print(f"{'plain' if tag == 'mdn' else 'by mode':>10} {n:10d} {nll:7.3f} "
                  f"{np.abs(c['w_right'] - ref['w_right']).max():14.3f} "
                  f"{np.median(np.abs(c['E'] - ref['E']) / ref['E']):13.3f} {np.median(err):16.3f} {err.max():13.3f}"
                  f"   ({time.time() - t0:.0f} s)", flush=True)
            if n == SIZES[-1]:
                th_t = torch.tensor(np.linspace(*THETA_RANGE, 200))
                with torch.no_grad():
                    w, mu, s = model(th_t)
                results.update({f"{tag}_fit_theta": th_t.numpy(), f"{tag}_fit_w": w.numpy(),
                                f"{tag}_fit_mu": mu.numpy(), f"{tag}_fit_s": s.numpy()})
                results.update(data_theta=theta[:4000].numpy(), data_xT=xT[:4000].numpy())
    # Zeroth-order with the same total number of simulations, spent at one theta.
    for n in SIZES:
        results[f"zo{n}_err"] = zeroth_order_error(grid, n, rng, EVAL_THETAS, ref["grad"])
    print("\nmedian gradient error / max|grad| over theta: MDN (amortised over all theta) vs zeroth-order "
          "(same simulations spent at each single theta)")
    for n in SIZES:
        mdn_err = np.median(np.abs(results[f"mdn{n}_grad"] - ref["grad"])) / scale
        smdn_err = np.median(np.abs(results[f"smdn{n}_grad"] - ref["grad"])) / scale
        zo_err = np.median(results[f"zo{n}_err"] * np.maximum(np.abs(ref["grad"]), 1.0)) / scale
        print(f"   {n:6d} sims: plain mixture {mdn_err:.3f}   by-mode mixture {smdn_err:.3f}   zeroth-order {zo_err:.3f}")
    np.savez(bl.RESULTS / "mdn_corner_drop.npz", **results)
    plot()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.load(bl.RESULTS / "mdn_corner_drop.npz")
    ink, muted = "#0b0b0b", "#52514e"
    c_plain, c_mode, c_zo = "#eb6834", "#1baf7a", "#2a78d6"
    th = r["thetas"]
    n = SIZES[-1]
    scale = np.abs(r["ref_grad"]).max()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), constrained_layout=True)

    def style(ax, title, xlabel, ylabel):
        ax.set_title(title, color=ink, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=muted)
        ax.set_ylabel(ylabel, color=muted)
        ax.tick_params(colors=muted)
        ax.grid(alpha=0.3, lw=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # (a), (b) fitted components on the data, plain vs by mode
    for ax, tag, title in ((axes[0, 0], "mdn", "Plain mixture: components cross modes"),
                           (axes[0, 1], "smdn", "By-mode mixture: components stay put")):
        ax.scatter(r["data_theta"], r["data_xT"], s=2, color=muted, alpha=0.25, linewidths=0)
        mu, w = r[f"{tag}_fit_mu"], r[f"{tag}_fit_w"]
        for k in range(mu.shape[1]):
            right = np.median(mu[:, k]) > 0
            ax.scatter(r[f"{tag}_fit_theta"], mu[:, k], s=4 + 30 * w[:, k], color=c_plain if right else c_zo,
                       alpha=np.clip(w[:, k], 0.05, 1), linewidths=0)
        style(ax, title, "theta = nominal drop position [m]", "x(T) [m]")

    # (c) right-mode weight
    ax = axes[0, 2]
    ax.plot(th, r["ref_w_right"], color=ink, lw=2.2, label="true (simulator)")
    ax.plot(th, r[f"mdn{n}_w_right"], color=c_plain, lw=1.6, label="plain mixture")
    ax.plot(th, r[f"smdn{n}_w_right"], color=c_mode, lw=1.6, ls="--", label="by-mode mixture")
    style(ax, f"Weight of the right mode ({n} sims)", "theta [m]", "P(x(T) > 0)")
    ax.legend(frameon=False, fontsize=9)

    # (d) gradient: truth, both models, by-mode split
    ax = axes[1, 0]
    ax.plot(th, r["ref_grad"], color=ink, lw=2.2, label="true dE[L]/dtheta")
    ax.plot(th, r[f"mdn{n}_grad"], color=c_plain, lw=1.5, label="plain mixture")
    ax.plot(th, r[f"smdn{n}_grad"], color=c_mode, lw=1.8, ls="--", label="by-mode mixture")
    ax.plot(th, r[f"smdn{n}_weights"], color=c_mode, lw=1, alpha=0.6, label="  by-mode: mode-weight term")
    ax.plot(th, r[f"smdn{n}_within"], color=c_zo, lw=1, alpha=0.6, label="  by-mode: within-mode term")
    style(ax, f"Gradient of the expected loss ({n} sims)", "theta [m]", "gradient [m]")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # (e) gradient error vs simulations
    ax = axes[1, 1]
    for tag, col, mk, lab in (("mdn", c_plain, "s", "plain mixture (one fit, all theta)"),
                              ("smdn", c_mode, "^", "by-mode mixture (one fit, all theta)")):
        ax.plot(SIZES, [np.median(np.abs(r[f"{tag}{m}_grad"] - r["ref_grad"])) / scale for m in SIZES],
                color=col, lw=2, marker=mk, label=lab)
    ax.plot(SIZES, [np.median(r[f"zo{m}_err"] * np.maximum(np.abs(r["ref_grad"]), 1.0)) / scale for m in SIZES],
            color=c_zo, lw=2, marker="o", label="zeroth-order (all sims at each theta)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "Median gradient error vs simulations", "simulations", "median |error| / max|true gradient|")
    ax.legend(frameon=False, fontsize=9)

    # (f) gradient error along theta
    ax = axes[1, 2]
    ax.plot(th, np.abs(r[f"mdn{n}_grad"] - r["ref_grad"]) / scale, color=c_plain, lw=1.5, label="plain mixture")
    ax.plot(th, np.abs(r[f"smdn{n}_grad"] - r["ref_grad"]) / scale, color=c_mode, lw=1.8, label="by-mode mixture")
    ax.plot(th, r[f"zo{n}_err"] * np.maximum(np.abs(r["ref_grad"]), 1.0) / scale, color=c_zo, lw=1.3,
            label=f"zeroth-order, {n} sims at each theta")
    ax.set_yscale("log")
    style(ax, f"Gradient error along theta ({n} sims)", "theta [m]", "|error| / max|true gradient|")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Corner drop: mixture density models of the outcome and their gradients "
                 f"(top left: component means, dot size = weight; {SIZES[-1]} training simulations)",
                 color=muted, fontsize=10, x=0.01, ha="left")
    fig.savefig(bl.RESULTS / "mdn_corner_drop.png", dpi=150)
    print(f"wrote {bl.RESULTS / 'mdn_corner_drop.png'}")


if __name__ == "__main__":
    main()
