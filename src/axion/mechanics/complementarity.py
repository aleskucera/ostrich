import warp as wp


@wp.func
def scaled_fisher_burmeister(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32 = 1.0,
    beta: wp.float32 = 1.0,
):
    """
    Computes the Scaled Fisher-Burmeister NCP function value.
    Phi(a, b) = alpha*a + beta*b - sqrt((alpha*a)^2 + (beta*b)^2)
    """
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)

    return scaled_a + scaled_b - norm


@wp.func
def scaled_fisher_burmeister_diff(
    a: wp.float32, b: wp.float32, alpha: wp.float32, beta: wp.float32,
    eps_sq: wp.float32,
):
    """Scaled FB with smoothing: φ_ε(a, b) = α·a + β·b − √((α·a)² + (β·b)² + eps_sq).

    The ``eps_sq`` term removes the corner degeneracy of the standard
    FB function at (a=0, b>0) and (a>0, b=0) — at the cost of slightly
    soft complementarity (λ·g ≈ eps_sq at convergence). For our problem
    the touching boundary (λ_n>0, signed_dist=0) is where seeded warm-
    starts land, and standard FB has ∂φ/∂a → 0 there, slowing PCR.
    Setting eps_sq > 0 keeps that derivative non-zero.
    """
    scaled_a = alpha * a
    scaled_b = beta * b

    norm = wp.sqrt(scaled_a * scaled_a + scaled_b * scaled_b + eps_sq)

    value = scaled_a + scaled_b - norm

    # Analytic derivatives (valid everywhere thanks to eps_sq > 0).
    dvalue_da = alpha * (1.0 - scaled_a / norm)
    dvalue_db = beta * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db

