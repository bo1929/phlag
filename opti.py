import jax
import jax.numpy as jnp
from jax import grad


# ================================================================
# 1️⃣ KL-Repulsion M-step
# ================================================================
def mstep_kl_repulsion(
    p: jnp.ndarray, c: jnp.ndarray, lambda_: float, max_iter=100, tol=1e-6
):
    """
    KL-repulsion M-step for a two-state HMM emission distribution.

    Objective:
        maximize_Q [ Σ_i c_i log q_i + λ Σ_i q_i log(q_i / p_i) ],
        s.t. Σ_i q_i = 1, q_i ≥ 0.

    - The first term is the expected complete-data log-likelihood.
    - The second term is a *repulsion regularizer* encouraging Q to be
      dissimilar from the known distribution P via KL(Q || P).

    Interpretation:
        • Equivalent to adding an energy term proportional to the information
          distance between Q and P.
        • Large λ values push Q away from P while maintaining normalization.
        • Convex in Q; solved via projected gradient ascent.

    Args:
        p: known categorical distribution P (shape K, sums to 1).
        c: expected emission counts from E-step.
        lambda_: repulsion strength.
        max_iter: maximum gradient iterations.
        tol: stopping tolerance for updates.

    Returns:
        q: updated emission distribution for the unknown state.
    """
    K = p.shape[0]
    q = jnp.ones(K) / K  # initialize uniform

    def objective(q):
        q = jnp.clip(q, 1e-9, 1.0)
        return jnp.sum(c * jnp.log(q)) + lambda_ * jnp.sum(
            q * (jnp.log(q) - jnp.log(p))
        )

    grad_obj = grad(objective)

    for _ in range(max_iter):
        q_new = q + 0.1 * grad_obj(q)
        q_new = jnp.clip(q_new, 1e-9, None)
        q_new = q_new / jnp.sum(q_new)
        if jnp.linalg.norm(q_new - q) < tol:
            break
        q = q_new
    return q


# ================================================================
# 2️⃣ Hellinger-Repulsion M-step
# ================================================================
def mstep_hellinger_repulsion(
    p: jnp.ndarray, c: jnp.ndarray, lambda_: float, max_iter=100, tol=1e-6
):
    """
    Hellinger-repulsion M-step for a two-state HMM.

    Objective:
        maximize_Q [ Σ_i c_i log q_i + λ Σ_i (√(p_i q_i) - q_i) ],
        s.t. Σ_i q_i = 1, q_i ≥ 0.

    - The first term: expected log-likelihood.
    - The second term: a *Hellinger-distance repulsion* encouraging
      Q to have a large metric distance from P in the probability simplex.

    Interpretation:
        • The Hellinger distance H^2(P,Q) = ½ Σ_i (√p_i - √q_i)^2
          defines a symmetric, bounded dissimilarity measure.
        • Maximizing Σ_i (√(p_i q_i) - q_i) ≈ -H^2(P,Q) increases
          the separation between P and Q while remaining metric-based.
        • Solved via projected gradient ascent; convex for small λ.

    Args:
        p: known categorical distribution P.
        c: expected emission counts from E-step.
        lambda_: repulsion strength.
        max_iter: number of gradient steps.
        tol: stopping tolerance.

    Returns:
        q: updated emission distribution.
    """
    K = p.shape[0]
    q = jnp.ones(K) / K

    def objective(q):
        q = jnp.clip(q, 1e-9, 1.0)
        return jnp.sum(c * jnp.log(q)) + lambda_ * jnp.sum(jnp.sqrt(p * q) - q)

    grad_obj = grad(objective)

    for _ in range(max_iter):
        q_new = q + 0.1 * grad_obj(q)
        q_new = jnp.clip(q_new, 1e-9, None)
        q_new = q_new / jnp.sum(q_new)
        if jnp.linalg.norm(q_new - q) < tol:
            break
        q = q_new
    return q


# ================================================================
# 3️⃣ Repelled Dirichlet Prior M-step
# ================================================================
def mstep_repeldirichlet(
    p: jnp.ndarray, c: jnp.ndarray, eta: float, epsilon: float = 1e-3
):
    """
    Repelled-Dirichlet M-step for the unknown emission distribution.

    Prior:
        Q ~ Dirichlet(β), where β_i = η(1 - p_i) + ε

    Objective:
        maximize_Q [ Σ_i c_i log q_i + (β_i - 1) log q_i ],
        s.t. Σ_i q_i = 1.

    - Closed-form solution via Dirichlet posterior mode (MAP estimate).
    - Acts as a Bayesian prior that reduces q_i where p_i is large.

    Interpretation:
        • Encourages Q to be probabilistically complementary to P.
        • Simple and stable; dissimilarity strength controlled by η.
        • Reduces to standard MLE when η → 0.

    Args:
        p: known categorical distribution P.
        c: expected counts from E-step.
        eta: repulsion strength parameter.
        epsilon: small constant to avoid zero prior mass.

    Returns:
        q: updated emission distribution for the unknown state.
    """
    alpha = eta * (1.0 - p) + epsilon
    q = c + alpha - 1.0
    q = jnp.clip(q, 1e-9, None)
    q = q / jnp.sum(q)
    return q


# ================================================================
# 4️⃣ Dot-Product (Overlap) Penalty M-step
# ================================================================
def mstep_dotproduct(
    p: jnp.ndarray, c: jnp.ndarray, lambda_: float, max_iter=50, tol=1e-9
):
    """
    Dot-product (overlap) penalty M-step.

    Objective:
        maximize_Q [ Σ_i c_i log q_i - λ Σ_i p_i q_i ],
        s.t. Σ_i q_i = 1, q_i ≥ 0.

    Analytical form:
        q_i ∝ c_i / (λ p_i + ν), where ν enforces normalization Σ_i q_i = 1.

    - Penalizes alignment (dot product) between Q and P, promoting
      low overlap regions in the simplex.
    - Smoothly trades off likelihood fit vs. dissimilarity with λ.

    Interpretation:
        • The term -λ Σ_i p_i q_i acts as a linear “repulsive potential”.
        • Simple to compute, numerically stable, and preserves convexity.
        • Similar to entropy-regularized optimal transport repulsion.

    Args:
        p: known categorical distribution P.
        c: expected emission counts from E-step.
        lambda_: repulsion strength.
        max_iter: max iterations for root finding ν.
        tol: tolerance for normalization constraint.

    Returns:
        q: updated emission distribution (normalized).
    """

    def f(nu):
        q = c / (lambda_ * p + nu)
        return jnp.sum(q) - 1.0

    nu_lo, nu_hi = -lambda_ * jnp.max(p) + 1e-6, 1e3
    for _ in range(max_iter):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        val = f(nu_mid)
        nu_lo, nu_hi = jax.lax.cond(
            val > 0, lambda _: (nu_mid, nu_hi), lambda _: (nu_lo, nu_mid), None
        )
        # if jnp.abs(val) < tol:
        #     break

    nu = 0.5 * (nu_lo + nu_hi)
    q = c / (lambda_ * p + nu)
    q = jnp.clip(q, 1e-9, None)
    q = q / jnp.sum(q)
    return q


# ================================================================
# ✅ Example Usage
# ================================================================
if __name__ == "__main__":
    p = jnp.array([0.7, 0.2, 0.1])
    c = jnp.array([15.0, 10.0, 5.0])

    print("P:", p)
    print("KL Prior Q:", mstep_kl_repulsion(p, c, lambda_=2.0))
    print("Hellinger Prior Q:", mstep_hellinger_repulsion(p, c, lambda_=2.0))
    print("Repelled Dirichlet Prior Q:", mstep_repeldirichlet(p, c, eta=10.0))
    print("Dot-product Prior Q:", mstep_dotproduct(p, c, lambda_=2.0))


import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize


# ---------------------------------------------
# 2️⃣ KL penalty (MAP as Dirichlet prior)
# ---------------------------------------------
def mstep_kl_penalty(counts: jnp.ndarray, p: jnp.ndarray, lam: float):
    """
    MAP update with KL(P||Q) penalty interpreted as Dirichlet prior
    counts: expected counts
    p: known emission distribution
    lam: penalty strength (prior pseudo-count)
    q_i = (c_i + lam*p_i) / sum_j (c_j + lam*p_j)
    """
    q = counts + lam * p
    q /= q.sum()
    return q


# ---------------------------------------------
# 3️⃣ Dot-product penalty (MAP-like update via Lagrange)
# ---------------------------------------------
def mstep_dot_product(
    counts: jnp.ndarray, p: jnp.ndarray, lam: float, tol: float = 1e-10
):
    """
    MAP-like update for dot-product penalty:
        maximize sum_i c_i log q_i - lam * sum_i p_i q_i
        s.t. sum_i q_i = 1, q_i >= 0
    Uses Lagrange multiplier to enforce normalization.
    """
    K = counts.shape[0]

    def q_lagrange(nu):
        q = counts / (nu + lam * p)
        return q.sum() - 1.0

    # Solve for nu using scalar root-finding
    # bounds: nu must be > -lam*min(p_i) to keep q_i > 0
    nu_min = -lam * p.min() + 1e-12
    nu_max = counts.sum() * 10.0  # arbitrary upper bound
    sol = minimize(
        lambda nu: jnp.square(q_lagrange(nu)),
        bounds=(nu_min, nu_max),
        method="bounded",
        options={"xatol": tol},
    )
    nu_opt = sol.x
    q = counts / (nu_opt + lam * p)
    q /= q.sum()  # normalize for numerical stability
    return q


# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    counts = jnp.array([10.0, 5.0, 3.0])
    p = jnp.array([0.6, 0.3, 0.1])
    lam = 5.0
    eta = 2.0

    q_dirichlet = mstep_repelled_dirichlet(counts, eta)
    q_kl = mstep_kl_penalty(counts, p, lam)
    q_dot = mstep_dot_product(counts, p, lam)

    print("Repelled Dirichlet:", q_dirichlet)
    print("KL penalty MAP:", q_kl)
    print("Dot-product MAP:", q_dot)
