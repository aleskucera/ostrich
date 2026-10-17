"""M1: one step of mixture propagation through a contact event vs Monte Carlo.

Case A, ball hitting the ground: branches "impact this step" (d_free < 0)
and "free flight". The boundary is linear in the state and, frictionless,
both branch maps are affine, so the split must match Monte Carlo up to
sampling error.

Case B, ball hitting the wedge apex: branches left face / apex vertex /
right face (which contact feature applies at the start of the step). The
boundaries are linear, the vertex branch map is not (its normal depends on
the position): this measures the cost of linearising within a branch.

For each branch: predicted probability, mean and covariance after the step
vs Monte Carlo (N_MC samples through the real step, labelled by the same
boundary functions). Errors are in units of the Monte Carlo standard error.

Run: PYTHONPATH=. python m1_mixture_step.py
"""

import torch

import corner_drop as cd
import mixture as mx
import sim2d

N_MC = 40_000


def monte_carlo(sim, mu, Sigma, boundaries, labels, gen):
    s = torch.distributions.MultivariateNormal(mu, Sigma).sample((N_MC,))
    q1, v1, info = sim.step(s[:, :3], s[:, 3:])
    out = torch.cat([q1, v1], dim=1)
    sample_labels = mx.label_samples(s, boundaries)
    res = {}
    for label in labels:
        mask = torch.tensor([lab == label for lab in sample_labels])
        x = out[mask]
        res[label] = (mask.float().mean(), x.mean(dim=0), torch.cov(x.T), len(x))
    return res, out


def report(name, pred, mc, names):
    print(f"\n{name}")
    print(f"  {'branch':>10} {'p pred':>8} {'p MC':>8} {'(se)':>7} {'mean err / se (max)':>20} {'cov rel err':>12}")
    for label, p, m, S in pred:
        p_mc, m_mc, S_mc, n = mc[label]
        if n < 50:
            print(f"  {names[label]:>10} {p.item():8.4f} {p_mc.item():8.4f}   (too few Monte Carlo samples to compare moments)")
            continue
        se_p = torch.sqrt(p_mc * (1 - p_mc) / N_MC)
        se_m = torch.sqrt(torch.diagonal(S_mc).clamp(min=1e-30) / max(n, 1))
        mean_err = ((m - m_mc).abs() / se_m.clamp(min=1e-15)).max()
        cov_err = (S - S_mc).norm() / S_mc.norm()
        print(f"  {names[label]:>10} {p.item():8.4f} {p_mc.item():8.4f} {se_p.item():7.4f} "
              f"{mean_err.item():20.2f} {cov_err.item():12.3e}")


def total(name, pred, out):
    m, S = mx.moments(pred)
    se = torch.sqrt(torch.diagonal(torch.cov(out.T)) / N_MC)
    print(f"  whole mixture: mean err / se (max) {((m - out.mean(dim=0)).abs() / se).max().item():.2f}, "
          f"cov rel err {((S - torch.cov(out.T)).norm() / torch.cov(out.T).norm()).item():.3e}")


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    # Case A: ball above the ground, falling; impact probability ~ 0.5.
    sim = sim2d.Sim2D(sim2d.ball(1.0, 0.05), sim2d.SimConfig(restitution=0.8, mu=0.0))
    h, vy = sim.cfg.dt, -2.0
    y_half = 0.05 - h * (vy - 9.81 * h)  # d_free = 0 at the mean
    mu = torch.tensor([0.0, y_half, 0.0, 0.3, vy, 0.0], dtype=torch.float64)
    Sigma = torch.diag(torch.tensor([1e-2, 5e-3, 1e-2, 5e-2, 2e-1, 1e-1], dtype=torch.float64) ** 2)
    f0 = torch.zeros(3, dtype=torch.float64)
    g_impact = lambda s: sim.free_gap(s[:3], s[3:], f0)[1][0]
    pred = mx.propagate(sim, mx.partition(mu, Sigma, [g_impact]))
    mc, out = monte_carlo(sim, mu, Sigma, [g_impact], [(-1,), (1,)], gen)
    report("Case A: ball onto the ground (e = 0.8, frictionless)", pred, mc, {(-1,): "impact", (1,): "free"})
    total("A", pred, out)

    # Case B: ball onto the wedge apex; x spread over all three features.
    sim = cd.make_sim()
    mu = torch.tensor([0.0, cd.H + cd.R + 0.002, 0.0, 0.0, -1.5, 0.0], dtype=torch.float64)
    Sigma = torch.diag(torch.tensor([2e-2, 1e-3, 1e-2, 5e-2, 1e-1, 1e-1], dtype=torch.float64) ** 2)
    g_left = lambda s: -((s[:2] - sim.apex) @ sim.dir_left)  # < 0: the left face applies
    g_right = lambda s: -((s[:2] - sim.apex) @ sim.dir_right)  # < 0: the right face applies
    f0 = torch.zeros(3, dtype=torch.float64)
    impact = lambda k: (lambda s: sim.free_gap(s[:3], s[3:], f0)[1][k])  # < 0: contact k is hit this step
    feature = {(-1, 1): 1, (1, -1): 2, (1, 1): 3}  # left face, right face, apex vertex
    tree = lambda label: (g_left if len(label) == 0 else g_right if len(label) == 1
                          else impact(feature[label[:2]]) if len(label) == 2 and label[:2] in feature else None)
    labels = {}
    for lab, nm in (((-1, 1), "left"), ((1, 1), "apex"), ((1, -1), "right")):
        labels[lab + (-1,)] = f"{nm} hit"
        labels[lab + (1,)] = f"{nm} free"
    pred = [c for c in mx.propagate(sim, mx.partition(mu, Sigma, tree)) if c[0] in labels]
    mc, out = monte_carlo(sim, mu, Sigma, tree, list(labels), gen)
    report("Case B: ball onto the wedge apex (e = 0.5, frictionless)", pred, mc, labels)
    total("B", pred, out)


if __name__ == "__main__":
    main()
