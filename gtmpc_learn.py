"""
gtmpc_learn: Learn strongly monotone game-theoretic MPC (GT-MPC) laws from data.

Given input/output trajectories from multiple experiments, the goal is to learn the stage-cost weights and
slack penalties of a strongly monotone GT-MPC controller that imitates the observed behavior.

The approach first identifies a linear prediction model

    x(t+1) = A x(t) + B u(t)
    y(t) = C x(t)

from the collected data, which is shared by N agents.

Agent i commands the inputs u_i = u[I_i] and, at the parameter p = (x_hat, u_prev, r), solves

    min_{x_i, eps_i}  sum_{t=0}^{T-1} (y(t+1) - r)' Qy_i (y(t+1) - r)
                      + sum_{t=0}^{Tu-1} du(t)' Qdu_i du(t) + qeps_i eps_i + Qeps_i eps_i^2
    s.t.  umin <= u_i(t) <= umax,  dumin <= du_i(t) <= dumax,  t = 0, ..., Tu-1
          ymin - (sum_j eps_j) 1 <= y(t) <= ymax + (sum_j eps_j) 1,  t = 1, ..., Tc
          eps_i >= 0

where x_i = (du_i(0), ..., du_i(Tu-1)) collects agent i's input increments and
the inputs are held constant after Tu steps. The GT-MPC law applies the first
move of the variational generalized Nash equilibrium (GNE) of the resulting
game. The input increments x = (x_1, ..., x_N) are ordered by agent; the decision
vector z = (x, eps) of the paper is stored by GNEProblem in the agent ordering
(x_1, eps_1, ..., x_N, eps_N).

The weights are learned from closed-loop data by imitation (learn_bilevel),
interpreting the measured moves of the agents as the first moves of the
variational GNE of the game at the measured parameters, while making the game
strongly monotone. The learning problem is a bilevel problem whose inner
problems are the GNE problems of the game, solved by L-BFGS-B. The GNEs are
computed by DR-DAQP (nashopt) and differentiated through the reduced QP of
Bemporad and Tatarenko (2026) at the active set. Strong monotonicity holds for
every value of the parameters by an exact shift of the own input-increment
weights; optionally (force_stability=True), closed-loop stability of the
unconstrained Nash equilibrium feedback is enforced by a penalty with a
jointly learned quadratic Lyapunov function.

Reference: A. Bemporad and T. Tatarenko, "Learning monotone game-theoretic linear MPC", 2026.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from nashopt import GNEP_LQ

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------
# Identification and state estimation
# ----------------------------------------------------------------------------
def identify_model(Ys, Us, nx, seed=0, rho_x0=1e-6, rho_th=1e-8, adam_epochs=1000, lbfgs_epochs=4000):
    """Identify a linear model (A, B, C) and the initial state of each experiment with jax-sysid.

    Ys, Us: lists of output and input sequences (one array per experiment).
    Returns the jax-sysid model, the matrices (A, B, C), and the list of initial states.
    """
    from jax_sysid.models import LinearModel
    ny, nu = Ys[0].shape[1], Us[0].shape[1]
    model = LinearModel(nx, ny, nu, feedthrough=False, seed=seed)
    model.loss(rho_x0=rho_x0, rho_th=rho_th)
    model.optimization(adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, iprint=-1)
    model.fit([Y.copy() for Y in Ys], [U.copy() for U in Us])
    A, B, C, _ = model.ssdata()
    return model, (np.array(A), np.array(B), np.array(C)), [np.array(x0) for x0 in model.x0]


def equilibrium(A, B, C):
    """Matrices (Px, Pu) such that (x, u) = (Px r, Pu r) satisfies x = A x + B u, C x = r."""
    n, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    M = np.block([[A - np.eye(n), B], [C, np.zeros((ny, nu))]])
    sol = np.linalg.solve(M, np.vstack((np.zeros((n, ny)), np.eye(ny))))
    return sol[:n], sol[n:]


class KalmanFilter:
    """Kalman filter for x(t+1) = A x(t) + B u(t) + w, y(t) = C x(t) + v, cov(w) = Q, cov(v) = R."""

    def __init__(self, A, B, C, Q, R, P0):
        self.A, self.B, self.C, self.Q, self.R, self.P0 = A, B, C, Q, R, P0

    def reset(self, x0):
        self.x, self.P = np.array(x0, dtype=float), self.P0.copy()

    def update(self, y):
        """Measurement update; returns the filtered estimate x(t|t)."""
        S = self.C @ self.P @ self.C.T + self.R
        L = np.linalg.solve(S, self.C @ self.P).T
        self.x = self.x + L @ (y - self.C @ self.x)
        self.P = (np.eye(len(self.x)) - L @ self.C) @ self.P
        return self.x

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def filter(self, x0, Y, U):
        """Filtered estimates x(t|t), t = 0, ..., len(Y)-1, starting from the prior x0."""
        self.reset(x0)
        Xh = []
        for y, u in zip(Y, U):
            Xh.append(self.update(y).copy())
            self.predict(u)
        return np.array(Xh)


# ----------------------------------------------------------------------------
# GT-MPC game
# ----------------------------------------------------------------------------
def _as_vector(v, m):
    return np.broadcast_to(np.asarray(v, dtype=float), (m,)).copy()


class GTMPCGame:
    """Structure of the GT-MPC game: prediction model, agents, horizons, and constraints.

    agent_inputs: list of N lists, the indices of the inputs commanded by each agent.
    T, Tu, Tc: prediction, control, and output-constraint horizons (Tc <= T).
    umin, umax, dumin, dumax, ymin, ymax: bounds (scalars or vectors, possibly infinite).
    mu: monotonicity constant imposed on the learned game.
    eta: lower bound on the own input-increment weights, Qdu_i^{ii} >= eta I.
    """

    def __init__(self, A, B, C, agent_inputs, T, Tu, umin, umax, dumax, ymin, ymax, dumin=None, Tc=None,
                 mu=0.1, eta=1e-3):
        self.A, self.B, self.C = np.asarray(A), np.asarray(B), np.asarray(C)
        self.n, self.nu, self.ny = A.shape[0], B.shape[1], C.shape[0]
        self.agents = [list(I) for I in agent_inputs]
        self.N = len(self.agents)
        self.T, self.Tu, self.Tc = T, Tu, T if Tc is None else Tc
        self.umin, self.umax = _as_vector(umin, self.nu), _as_vector(umax, self.nu)
        self.dumax = _as_vector(dumax, self.nu)
        self.dumin = -self.dumax if dumin is None else _as_vector(dumin, self.nu)
        self.ymin, self.ymax = _as_vector(ymin, self.ny), _as_vector(ymax, self.ny)
        self.mu, self.eta = mu, eta

        # agent-ordered decision vector x = (x_1, ..., x_N), x_i = (du_i(0), ..., du_i(Tu-1))
        self.ni = [len(I) for I in self.agents]
        self.mi = [n_i * Tu for n_i in self.ni]
        off = np.cumsum([0] + self.mi)
        self.blk = [np.arange(off[i], off[i + 1]) for i in range(self.N)]
        self.nx = int(off[-1])
        # Pi maps x to the time-ordered increments (du(0), ..., du(Tu-1))
        self.Pi = np.zeros((self.nu * Tu, self.nx))
        for i, I in enumerate(self.agents):
            for t in range(Tu):
                for a, j in enumerate(I):
                    self.Pi[t * self.nu + j, off[i] + t * self.ni[i] + a] = 1.
        self.Sel = [np.eye(self.nx)[b] for b in self.blk]
        # cumulative sums of agent i's increments: u_i(t) - u_i(-1), t < Tu
        self.Scum = [np.kron(np.tril(np.ones((Tu, Tu))), np.eye(n_i)) for n_i in self.ni]

        # outputs Y = (y(1), ..., y(T)) = Hy p + Gx x, tracking errors Y - 1 kron r = Hp p + Gx x
        Ad, Bd, Cd = self.A, self.B, self.C
        Phi = np.vstack([Cd @ np.linalg.matrix_power(Ad, t + 1) for t in range(T)])
        Gam = np.zeros((self.ny * T, self.nu * T))
        for t in range(T):
            for j in range(t + 1):
                Gam[t * self.ny:(t + 1) * self.ny, j * self.nu:(j + 1) * self.nu] = \
                    Cd @ np.linalg.matrix_power(Ad, t - j) @ Bd
        hold = np.kron(np.array([[1. if l <= t else 0. for l in range(Tu)] for t in range(T)]), np.eye(self.nu))
        Gu = Gam @ np.kron(np.ones((T, 1)), np.eye(self.nu))
        self.Gx = Gam @ hold @ self.Pi
        self.Gi = [self.Gx[:, b] for b in self.blk]
        self.Hy = np.hstack((Phi, Gu, np.zeros((self.ny * T, self.ny))))
        self.Hp = np.hstack((Phi, Gu, -np.kron(np.ones((T, 1)), np.eye(self.ny))))

    # ------------------------------------------------------------------------
    def pseudogradient(self, Qy, Qdu, kron=np.kron, vstack=np.vstack):
        """Matrices (A, Q) of the pseudogradient A x + q(p), q(p) = Q p, with respect to the input increments x,
        for the weights Qy_i, Qdu_i.

        kron and vstack select the backend (numpy or jax.numpy).
        """
        rows, q = [], []
        for i in range(self.N):
            Qb = kron(np.eye(self.T), Qy[i])
            Rdu = self.Sel[i] @ self.Pi.T @ kron(np.eye(self.Tu), Qdu[i]) @ self.Pi
            rows.append(2 * self.Gi[i].T @ Qb @ self.Gx + 2 * Rdu)
            q.append(2 * self.Gi[i].T @ Qb @ self.Hp)
        return vstack(rows), vstack(q)

    def increment_weight(self, i, Qii, S):
        """Qdu_i with own block Qii, cross block S (n_i x (nu - n_i)), and zero remaining blocks."""
        I = self.agents[i]
        others = [j for j in range(self.nu) if j not in I]
        Ei, Eo = np.eye(self.nu)[:, I], np.eye(self.nu)[:, others]
        return Ei @ Qii @ Ei.T + Ei @ S @ Eo.T + Eo @ S.T @ Ei.T

    def complete_increment_weight(self, i, Qdu):
        """Qdu_i with the block of u_{-i} set to S' Qii^{-1} S, the smallest choice for which Qdu_i >= 0.

        Only the block row i of Qdu_i (Qii and S) enters the game, so the completion leaves A and Q
        unchanged.
        """
        I = self.agents[i]
        others = [j for j in range(self.nu) if j not in I]
        Q = np.array(Qdu, dtype=float)
        Q = 0.5 * (Q + Q.T)
        Qii, S = Q[np.ix_(I, I)], Q[np.ix_(I, others)]
        Q[np.ix_(others, others)] = S.T @ np.linalg.solve(Qii, S)
        return 0.5 * (Q + Q.T)

    def closed_loop_matrix(self, A, Q, xp=np):
        """Closed-loop matrix of the unconstrained Nash equilibrium feedback on the identified model.

        Without constraints, the GNE is x = -A^{-1} q(p), p = (xi_hat, u_prev, r), and its first move
        du(0) = K_xi xi_hat + K_u u_prev + K_r r gives, for zeta = (xi, u_prev) with xi_hat = xi and the
        identified model (A_m, B_m),
            zeta(k+1) = M zeta(k) + (terms in r),  M = [[A_m + B_m K_xi, B_m (I + K_u)], [K_xi, I + K_u]].
        xp selects the backend (numpy or jax.numpy).
        """
        n, nu = self.n, self.nu
        K = -self.Pi[:nu] @ xp.linalg.solve(A, Q[:, :n + nu])
        Kx, Ku = K[:, :n], K[:, n:]
        return xp.block([[self.A + self.B @ Kx, self.B @ (xp.eye(nu) + Ku)], [Kx, xp.eye(nu) + Ku]])

    def closed_loop_spectral_radius(self, weights):
        """Spectral radius of the closed-loop matrix of the unconstrained Nash equilibrium feedback."""
        A, Q = self.pseudogradient(weights["Qy"], weights["Qdu"])
        return float(np.max(np.abs(np.linalg.eigvals(self.closed_loop_matrix(A, Q)))))

    # ------------------------------------------------------------------------
    def _parameter(self, U, Xh, R, k):
        """Parameter p_k = (x_hat(k), u(k-1), r(k)) of sample k."""
        r = self.C @ Xh[k]
        tracked = ~np.isnan(R[k])
        r[tracked] = R[k][tracked]
        return np.hstack((Xh[k], U[k - 1], r))

    def gne_samples(self, experiments):
        """Samples of the bilevel problem from a list of experiments.

        Each experiment is a dict with U (K x nu, applied inputs), Xh (K x n, state estimates), and R
        (K x ny, references, NaN on the outputs that are not tracked, which are replaced by the
        current output estimates C x_hat). A sample is taken at every step k >= 1, with parameter
        p_k = (x_hat(k), u(k-1), r(k)), previous input u(k-1), and measured increment u(k) - u(k-1).
        """
        P, Up, dU = [], [], []
        for e in experiments:
            U, Xh, R = e["U"], e["Xh"], e["R"]
            for k in range(1, U.shape[0]):
                P.append(self._parameter(U, Xh, R, k))
                Up.append(U[k - 1])
                dU.append(U[k] - U[k - 1])
        return dict(P=np.array(P), u_prev=np.array(Up), du=np.array(dU))


# ----------------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------------
def _weights_from_theta(game, th):
    Qy = [Cy.T @ Cy for Cy in th["Cy"]]
    Qdu = []
    for i in range(game.N):
        Qii = th["C"][i].T @ th["C"][i] + game.eta * jnp.eye(game.ni[i])
        I = game.agents[i]
        others = [j for j in range(game.nu) if j not in I]
        Ei, Eo = jnp.eye(game.nu)[:, I], jnp.eye(game.nu)[:, others]
        Qdu.append(Ei @ Qii @ Ei.T + Ei @ th["S"][i] @ Eo.T + Eo @ th["S"][i].T @ Ei.T)
    return Qy, Qdu


def _sqnorm(mats):
    return sum(jnp.sum(M ** 2) for M in mats)


def _shifted_game(game, th, scale_y=1.):
    """Weights and game matrices with the exact monotonicity shift of the own increment weights.

    The output weights are Qy_i = scale_y Cy_i' Cy_i. With Atilde the pseudogradient matrix of the
    weights defined by th, c = max{0, mu - lambda_min(sym(Atilde))} / 2 is added to every own
    increment weight Qdu_i^{ii}, so that A = Atilde + 2 c I and lambda_min(sym(A)) >= mu for every th.
    """
    Qy, Qdu = _weights_from_theta(game, th)
    Qy = [scale_y * Q for Q in Qy]
    A0, Q = game.pseudogradient(Qy, Qdu, jnp.kron, jnp.vstack)
    c = 0.5 * jnp.maximum(0., game.mu - jnp.min(jnp.linalg.eigvalsh(0.5 * (A0 + A0.T))))
    E = jnp.eye(game.nu)
    Qdu = [Qdu[i] + c * E[:, game.agents[i]] @ E[game.agents[i], :] for i in range(game.N)]
    return Qy, Qdu, A0 + 2. * c * jnp.eye(game.nx), Q


def _lyapunov_basis(M, gamma):
    """Upper-triangular L with L' L = P, where M' P M - gamma^2 P = -I (identity if rho(M) >= gamma)."""
    from scipy.linalg import solve_discrete_lyapunov
    M = np.asarray(M)
    if np.max(np.abs(np.linalg.eigvals(M))) >= gamma:
        return np.eye(M.shape[0])
    P = solve_discrete_lyapunov(M.T / gamma, np.eye(M.shape[0]))
    return np.linalg.cholesky(0.5 * (P + P.T)).T


def learn_bilevel(game, samples, rho=1e-5, delta=0.01, qeps0=1e-3, Qeps0=0.1, force_stability=True,
                  gamma=0.9, rho_s=10., max_restarts=4, grid=np.arange(-10., -1.99, 0.5), maxiter=2000,
                  active_tol=1e-9, primal_tol=1e-10, verbose=True):
    """Learn the weights by the bilevel problem whose inner problems are the GNE problems, solved by L-BFGS-B.

    The weights are parametrized as Qy_i = s Cy_i' Cy_i, Qdu_i^{ii} = C_i' C_i + (eta + c) I, and cross
    weights S_i, where the shift c = max{0, mu - lambda_min(sym(Atilde))} / 2 makes the game strongly
    monotone with constant mu for every value of the parameters. The slack penalties are learned
    subject to qeps_i >= 0 and Qeps_i >= mu / 2, as qeps_i = qeps0 a_i, Qeps_i = Qeps0 b_i, with a_i
    and b_i of order one. The objective is
        [(1/(N K)) sum_k ||u_bar_k(theta) - u_k||^2
         + (rho/2) sum_i (||Qy_i||_F^2 + ||Qdu_i^{ii}||_F^2 + 2 ||S_i||_F^2)] / f0
    where u_bar_k(theta) is the first move of the variational GNE of the game at the measured parameter
    p_k and previous input u_{k-1}, u_k is the measured input, f0 is the imitation loss at the
    initialization, so that rho_s weights the stability penalty relative to it, and, if
    force_stability is True,
        + rho_s max{0, ||L M(theta) L^{-1}||_2^2 - gamma^2}^2,
    where M(theta) is the closed-loop matrix of the unconstrained Nash equilibrium feedback on the
    identified model (GTMPCGame.closed_loop_matrix) and L, upper triangular with positive diagonal,
    is learned jointly, so that ||L M L^{-1}||_2 < 1 certifies the quadratic Lyapunov function
    z' L' L z. If the certificate fails at the solution, rho_s is multiplied by 10 and the problem is
    solved again from the current point, at most max_restarts times.

    For a given set of active constraints, the GNE solves a linear system obtained from the reduced QP
    of Bemporad and Tatarenko (2026) in the multipliers of the active constraints, which is
    differentiated by JAX. At each evaluation, the active set of each sample at the previous
    evaluation is kept if it still gives nonnegative multipliers and a feasible z, which certifies
    the GNE; otherwise, the GNE is computed by DR-DAQP (GNEProblem.solve) and its active set is
    formed by the constraints whose (normalized) multipliers exceed active_tol.

    The parameters are initialized at Qy_i = s I, C_i = sqrt((mu + delta)/2 - eta) I, S_i = 0,
    qeps_i = qeps0, Qeps_i = Qeps0, with log10(s) in grid minimizing the imitation loss. The problem
    is subject only to the bounds on the slack penalties.

    Returns the weights (dict with Qy, Qdu, qeps, Qeps) and a dict with the selected scale s, the
    closed-loop spectral radius, and, if force_stability, the Lyapunov certificate. The block of
    u_{-i} of Qdu_i, which does not affect the game, is set to S_i' (Qdu_i^{ii})^{-1} S_i (see
    GTMPCGame.complete_increment_weight), so that Qdu_i >= 0.
    """
    from scipy.optimize import minimize
    N, mu = game.N, game.mu
    d = (mu + delta) / 2
    assert game.eta <= d, "eta must not exceed (mu + delta) / 2"
    assert qeps0 > 0. and Qeps0 >= mu / 2, "the initial slack penalties must satisfy qeps0 > 0 and Qeps0 >= mu / 2"

    gp = GNEProblem(game, primal_tol)
    P, Up, dU = samples["P"], samples["u_prev"], samples["du"]
    K = P.shape[0]
    # constraints with normalized rows, which leave the GNE unchanged and scale the multipliers
    nr = np.linalg.norm(gp.G, axis=1)
    Gn = jnp.asarray(gp.G / nr[:, None])
    Bn = jnp.asarray(np.array([gp.rhs(p, u) for p, u in zip(P, Up)]) / nr)
    Pj, dUj = jnp.asarray(P), jnp.asarray(dU)

    # active sets: indices of at most nz active constraints per sample, padded, and their mask
    amax = gp.nz
    idx, msk = np.zeros((K, amax), dtype=int), np.zeros((K, amax))

    def update_active_sets(F, Fp, fe, ks):
        # GNEs of the samples ks by DR-DAQP; the active constraints are those with the largest multipliers
        for k in ks:
            _, lam, ok, _ = gp.solve(F, Fp, fe, P[k], Up[k])
            if not ok:
                raise RuntimeError("GNE not found")
            lam = lam * nr
            a = np.argsort(-lam)[:amax]
            a = a[lam[a] > active_tol]
            idx[k], msk[k] = 0, 0.
            idx[k, :len(a)], msk[k, :len(a)] = a, 1.

    def gne_active(F, Fp, fe, idx, msk):
        # GNEs and multipliers with the given active sets: the rows of the reduced QP (H lambda + q0 = 0)
        # of the active constraints, with lambda = 0 on the other ones
        Fi = jnp.linalg.inv(F)

        def one(p, b, ik, mk):
            x0 = -Fi @ (Fp @ p + fe)
            Ga = Gn[ik] * mk[:, None]
            lam = jnp.linalg.solve(Ga @ Fi @ Ga.T + jnp.diag(1. - mk + 1e-12 * mk), (Ga @ x0 - b[ik]) * mk)
            return x0 - Fi @ (Ga.T @ lam), lam
        return jax.vmap(one)(Pj, Bn, idx, msk)

    @jax.jit
    def certified(F, Fp, fe, idx, msk):
        Z, lam = gne_active(F, Fp, fe, idx, msk)
        return jnp.all(lam * msk >= -active_tol, axis=1) & jnp.all(Z @ Gn.T <= Bn + active_tol, axis=1)

    def imitation(F, Fp, fe, idx, msk):
        Z = gne_active(F, Fp, fe, idx, msk)[0]
        return jnp.sum((Z @ gp.Fm.T - dUj) ** 2) / (N * K)

    # The output weights are Qy_i = s Cy_i' Cy_i, with Cy_i initialized at the identity, so that all
    # the optimization variables are of order one.
    def theta():
        return dict(Cy=[jnp.eye(game.ny) for _ in range(N)],
                    C=[jnp.sqrt(d - game.eta) * jnp.eye(n_i) for n_i in game.ni],
                    S=[jnp.zeros((n_i, game.nu - n_i)) for n_i in game.ni], a=jnp.ones(N), b=jnp.ones(N))

    def slack(th):
        return qeps0 * th["a"], Qeps0 * th["b"]

    def matrices(th, s):
        _, _, A, Q = _shifted_game(game, th, s)
        return gp.matrices(A, Q, *slack(th), xp=jnp)

    # scale of the initial output weights
    vals, imit_jit = [], jax.jit(imitation)
    for g_ in grid:
        F, Fp, fe = (np.asarray(a) for a in matrices(theta(), 10. ** g_))
        update_active_sets(F, Fp, fe, range(K))
        vals.append(float(imit_jit(F, Fp, fe, idx, msk)))
    s = 10. ** grid[int(np.argmin(vals))]
    f0 = min(vals)
    if verbose:
        print(f"Initialization: s = {s:.1e}, imitation loss {f0:.2e}")
    th0 = theta()

    # Lyapunov factor L = Lhat L0, with L0 from the closed loop of the initial game and Lhat upper
    # triangular with positive diagonal, initialized at the identity
    if force_stability:
        _, _, A, Q = _shifted_game(game, th0, s)
        L0 = jnp.asarray(_lyapunov_basis(game.closed_loop_matrix(np.asarray(A), np.asarray(Q)), gamma))
        nz = game.n + game.nu
        th0 = dict(th0, W=jnp.zeros((nz, nz)), d=jnp.zeros(nz))

    def lyap_norm(th):
        _, _, A, Q = _shifted_game(game, th, s)
        L = (jnp.triu(th["W"], 1) + jnp.diag(jnp.exp(th["d"]))) @ L0
        M = game.closed_loop_matrix(A, Q, jnp)
        return jnp.linalg.norm(L @ M @ jnp.linalg.inv(L), 2)

    def loss(th, rho_s, idx, msk):
        Qy, Qdu, A, Q = _shifted_game(game, th, s)
        F, Fp, fe = gp.matrices(A, Q, *slack(th), xp=jnp)
        v = (imitation(F, Fp, fe, idx, msk) + rho / 2 * _sqnorm(Qy + Qdu)) / f0
        if force_stability:
            v = v + rho_s * jnp.maximum(0., lyap_norm(th) ** 2 - gamma ** 2) ** 2
        return v

    z, unravel = ravel_pytree(th0)
    # bounds qeps_i >= 0, Qeps_i >= mu / 2
    lb = jax.tree_util.tree_map(lambda v: -jnp.inf * jnp.ones_like(v), th0)
    lb.update(a=jnp.zeros(N), b=mu / 2 / Qeps0 * jnp.ones(N))
    bounds = [(float(l) if np.isfinite(l) else None, None) for l in np.asarray(ravel_pytree(lb)[0])]
    mats = jax.jit(lambda z: matrices(unravel(z), s))
    for _ in range(max_restarts + 1 if force_stability else 1):
        fg = jax.jit(jax.value_and_grad(lambda z, r, i, m: loss(unravel(z), r, i, m)))

        def fun(z):
            zj = jnp.asarray(z)
            F, Fp, fe = mats(zj)
            update_active_sets(*(np.asarray(a) for a in (F, Fp, fe)),
                               np.flatnonzero(~np.asarray(certified(F, Fp, fe, idx, msk))))
            v, g = fg(zj, rho_s, idx, msk)
            return float(v), np.asarray(g, dtype=float)

        res = minimize(fun, np.asarray(z), jac=True, method="L-BFGS-B", bounds=bounds, tol=1e-10,
                       options=dict(maxiter=maxiter, maxfun=2 * maxiter))
        z = res.x
        cert = float(lyap_norm(unravel(jnp.asarray(z)))) if force_stability else None
        if verbose:
            print(f"L-BFGS-B: {res.nit} iterations, {res.message}"
                  + (f"; ||L M L^-1||_2 = {cert:.4f} (rho_s = {rho_s:g})" if force_stability else ""))
        if not force_stability or cert < 1.:
            break
        rho_s *= 10.
    th = unravel(jnp.asarray(z))
    Qy, Qdu, _, _ = _shifted_game(game, th, s)
    qe, Qe = slack(th)
    weights = dict(Qy=[np.asarray(Q) for Q in Qy],
                   Qdu=[game.complete_increment_weight(i, np.asarray(Q)) for i, Q in enumerate(Qdu)],
                   qeps=np.asarray(qe), Qeps=np.asarray(Qe))
    info = dict(s=s, spectral_radius=game.closed_loop_spectral_radius(weights))
    if force_stability:
        info.update(lyapunov_norm=cert, rho_s=rho_s, stable_certified=cert < 1.)
    return weights, info


# ----------------------------------------------------------------------------
# GT-MPC law
# ----------------------------------------------------------------------------
class GNEProblem:
    """Variational GNE problem of the GT-MPC game, solved by DR-DAQP through nashopt.

    The decision vector is ordered by agent, z = (x_1, eps_1, ..., x_N, eps_N). The pseudogradient
    of the game is Abar z + qbar(p), where Abar = F and qbar(p) = Fp p + fe are obtained from the matrices
    (A, Q) of GTMPCGame.pseudogradient and the slack penalties, and the shared constraints are
    G z <= rhs(p, u_prev), where G does not depend on p.
    """

    def __init__(self, game, primal_tol=1e-8):
        g = game
        self.game, self.primal_tol = game, primal_tol
        self.dims = [m + 1 for m in g.mi]
        off = np.cumsum([0] + self.dims)
        self.nz = int(off[-1])
        self.xz = [np.arange(off[i], off[i] + g.mi[i]) for i in range(g.N)]
        self.ez = np.array([off[i] + g.mi[i] for i in range(g.N)])
        self.Xz = np.zeros((g.nx, self.nz))
        for i in range(g.N):
            self.Xz[np.ix_(g.blk[i], self.xz[i])] = np.eye(g.mi[i])
        self.E = np.zeros((self.nz, g.N))
        self.E[self.ez, np.arange(g.N)] = 1.
        # Fm z = E_f x = first input increments du(0), in the input ordering
        self.Fm = np.zeros((g.nu, self.nz))
        for i, I in enumerate(g.agents):
            self.Fm[I, self.xz[i][:g.ni[i]]] = 1.
        # shared constraints
        rows = slice(0, g.ny * g.Tc)
        Gx = g.Gx[rows] @ self.Xz
        Ge = np.zeros((g.ny * g.Tc, self.nz))
        Ge[:, self.ez] = 1.
        Gs = []
        for i, I in enumerate(g.agents):
            S = np.zeros((g.mi[i], self.nz))
            S[:, self.xz[i]] = g.Scum[i]
            D = np.zeros((g.mi[i], self.nz))
            D[:, self.xz[i]] = np.eye(g.mi[i])
            Gs += [S, -S, D, -D]
        Gpos = np.zeros((g.N, self.nz))
        Gpos[np.arange(g.N), self.ez] = -1.
        G = np.vstack(Gs + [Gx - Ge, -Gx - Ge, Gpos])
        self.keep = np.isfinite(self._rhs(np.zeros(g.n + g.nu + g.ny), np.zeros(g.nu)))
        self.G = G[self.keep]

    def _rhs(self, p, u_prev):
        g = self.game
        Y0 = g.Hy[:g.ny * g.Tc] @ p
        bs = []
        for I in g.agents:
            bs += [np.tile(g.umax[I] - u_prev[I], g.Tu), np.tile(u_prev[I] - g.umin[I], g.Tu),
                   np.tile(g.dumax[I], g.Tu), -np.tile(g.dumin[I], g.Tu)]
        return np.hstack(bs + [np.tile(g.ymax, g.Tc) - Y0, Y0 - np.tile(g.ymin, g.Tc), np.zeros(g.N)])

    def rhs(self, p, u_prev):
        return self._rhs(p, u_prev)[self.keep]

    def matrices(self, A, Q, qeps, Qeps, xp=np):
        """Abar = F, with Abar_ii = blkdiag(A_ii, 2 Qeps_i), Abar_ij = blkdiag(A_ij, 0), and qbar(p) = Fp p + fe,
        qbar_i(p) = (q_i(p), qeps_i), in the ordering of z; xp selects the backend (numpy or jax.numpy)."""
        F = self.Xz.T @ A @ self.Xz + self.E @ xp.diag(2. * Qeps) @ self.E.T
        return F, self.Xz.T @ Q, self.E @ qeps

    def solve(self, F, Fp, fe, p, u_prev):
        """Variational GNE z, multipliers of the shared constraints, success flag, and solution time."""
        f = Fp @ p + fe
        Qs, cs = [], []
        for i in range(self.game.N):
            zi = np.append(self.xz[i], self.ez[i])
            Qi = np.zeros((self.nz, self.nz))
            Qi[zi, :] = F[zi, :]
            Qi[:, zi] = F[zi, :].T
            Qi[np.ix_(zi, zi)] = 0.5 * (F[np.ix_(zi, zi)] + F[np.ix_(zi, zi)].T)
            ci = np.zeros(self.nz)
            ci[zi] = f[zi]
            Qs.append(Qi)
            cs.append(ci)
        gnep = GNEP_LQ(self.dims, Qs, cs, A=self.G, b=self.rhs(p, u_prev), variational=True, solver="dr_daqp")
        sol = gnep.solve(solver_options=dict(primal_tol=self.primal_tol))
        z = np.asarray(sol.x)
        ok = int(sol.status_str) > 0 and np.all(np.isfinite(z))
        return z, np.asarray(sol.lam), ok, sol.elapsed_time


class GTMPCController:
    """GT-MPC law: first move of the variational GNE of the learned game, computed by DR-DAQP."""

    def __init__(self, game, weights, primal_tol=1e-8):
        self.gne = GNEProblem(game, primal_tol)
        A, Q = game.pseudogradient(weights["Qy"], weights["Qdu"])
        self.data = self.gne.matrices(A, Q, np.asarray(weights["qeps"]), np.asarray(weights["Qeps"]))

    def __call__(self, x_hat, u_prev, r):
        """Return the input u, a success flag, and the solution time."""
        p = np.hstack((x_hat, u_prev, r))
        z, _, ok, t = self.gne.solve(*self.data, p, u_prev)
        return np.asarray(u_prev, dtype=float) + self.gne.Fm @ z, ok, t
