"""
Reference generator based on the learned GT-MPC game (Section 4.1 and Figure 1 of the paper).

Same process, agents, data, and learned game (Tu = 4) as in example_paper.py. The agents keep operating
the process by playing their own (unknown) GT-MPC game, with the true model and state. A reference
generator placed on top of them receives the desired references r_des(k) and sends to the agents, at
each step k, the reference

    r(k) = argmin_r (y_1(r) - r_des,1(k))^2 + omega sum_{l=2,3} (y_l(r) - r_des,l(k))^2
                    + sigma ||r - r_des(k)||_2^2

    y(r) = C (A x_hat(k) + B u_bar(r))

where u_bar(r) is the first move of the variational GNE of the learned game at the parameter
p = (x_hat(k), u(k-1), r), (A, B, C) is the identified model, and x_hat(k) is the Kalman-filter
estimate of its state. The problem is solved by a Gauss-Newton method: for the active set of the
current GNE, the GNE is affine in r, with Jacobian obtained from the active rows of the reduced QP in
the multipliers.

The script saves Figure 1 of the paper (learn_gtmpc_example.pdf): the agents and the learned GT-MPC law
tracking r_des, and the agents tracking the references r(k) of the generator, with omega = 0.02.

Requirements: numpy, scipy, matplotlib, jax, jax-sysid, nashopt.

Reference: A. Bemporad and T. Tatarenko, "Learning monotone game-theoretic linear MPC", 2026.
"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gtmpc_learn import GNEProblem, GTMPCController
import example_paper as ex

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
omega = 0.02                # weight of the tracking errors of y_2, y_3
sigma = 1e-2                # regularization of r - r_des


# ----------------------------------------------------------------------------
# Reference generator
# ----------------------------------------------------------------------------
class ReferenceGenerator:
    """r(k) = argmin_r sum_l w_l (y_l(r) - r_des,l)^2 + sigma ||r - r_des||^2, w = (1, omega, ..., omega).

    model = (A, B, C) is the identified model, y(r) = C (A x_hat + B u_bar(r)), where u_bar(r) is the first
    move of the variational GNE of the game with the given weights.
    """

    def __init__(self, game, weights, model, omega, sigma, maxit=30, tol=1e-9, active_tol=1e-9):
        self.gp = GNEProblem(game, primal_tol=1e-10)
        A, Q = game.pseudogradient(weights["Qy"], weights["Qdu"])
        self.F, self.Fp, self.fe = self.gp.matrices(A, Q, np.asarray(weights["qeps"]), np.asarray(weights["Qeps"]))
        self.Fi = np.linalg.inv(self.F)
        self.ny = game.ny
        self.dx0 = -self.Fi @ self.Fp[:, -game.ny:]  # derivative of the unconstrained GNE with respect to r
        self.nrm = np.linalg.norm(self.gp.G, axis=1)
        Am, Bm, Cm = model
        self.CA, self.CB = Cm @ Am, Cm @ Bm
        self.sw = np.sqrt(np.hstack((1., omega * np.ones(game.ny - 1))))  # square roots of the output weights
        self.sigma, self.maxit, self.tol, self.active_tol = sigma, maxit, tol, active_tol

    def gne(self, x_hat, u_prev, r):
        """First input increments of the GNE and their Jacobian with respect to r (current active set)."""
        z, lam, ok, _ = self.gp.solve(self.F, self.Fp, self.fe, np.hstack((x_hat, u_prev, r)), u_prev)
        if not ok:
            raise RuntimeError("GNE not found")
        # For a fixed active set: z = x0 - Fi Ga' lam, lam = (Ga Fi Ga')^{-1} (Ga x0 - ba), where only x0
        # depends on r, as the constraints do not depend on the reference
        Ga = self.gp.G[lam * self.nrm > self.active_tol]
        dz = self.dx0
        if Ga.shape[0] > 0:
            dz = dz - self.Fi @ Ga.T @ np.linalg.lstsq(Ga @ self.Fi @ Ga.T, Ga @ self.dx0, rcond=None)[0]
        return self.gp.Fm @ z, self.gp.Fm @ dz

    def residual(self, x_hat, u_prev, r, r_des):
        """Weighted prediction errors e and their Jacobian with respect to r."""
        du, D = self.gne(x_hat, u_prev, r)
        e = self.sw * (self.CA @ x_hat + self.CB @ (u_prev + du) - r_des)
        return e, self.sw[:, None] * (self.CB @ D)

    def cost(self, r, r_des, e):
        return float(np.sum(e ** 2) + self.sigma * np.sum((r - r_des) ** 2))

    def __call__(self, x_hat, u_prev, r_des, r_prev):
        """Reference r(k), by Gauss-Newton with backtracking, warm-started from the better of r(k-1), r_des(k)."""
        cands = []
        for r in (r_prev, r_des):
            e, E = self.residual(x_hat, u_prev, r, r_des)
            cands.append((self.cost(r, r_des, e), r.copy(), e, E))
        J, r, e, E = min(cands, key=lambda c: c[0])
        for _ in range(self.maxit):
            g = E.T @ e + self.sigma * (r - r_des)
            s = -np.linalg.solve(E.T @ E + self.sigma * np.eye(self.ny), g)
            if np.linalg.norm(s) <= self.tol * (1. + np.linalg.norm(r)):
                break
            t = 1.
            while t > 1e-8:
                e_n, E_n = self.residual(x_hat, u_prev, r + t * s, r_des)
                J_n = self.cost(r + t * s, r_des, e_n)
                if J_n <= J + 2e-4 * t * g @ s:
                    break
                t *= 0.5
            if t <= 1e-8:
                break
            r, J, e, E = r + t * s, J_n, e_n, E_n
        return r


def run(data, gen, Rdes):
    """Closed loop of the agents (true game and state) with the references r(k) set by gen, which uses the
    Kalman-filter estimate on the identified model. Returns the measured outputs, inputs, and references."""
    A, B, C = data["plant"]
    rng = np.random.default_rng([ex.seed, 0])
    x, u = data["Px"] @ Rdes[0], data["Pu"] @ Rdes[0]
    kf = data["kf"]
    kf.reset(data["Pxm"] @ Rdes[0])
    r = Rdes[0].copy()
    Y, U, R = [], [], []
    for k in range(Rdes.shape[0]):
        y = C @ x + ex.noise * rng.uniform(-1, 1, ex.N)
        r = gen(kf.update(y).copy(), u, Rdes[k], r)
        u, ok, _ = data["agents"](x, u, r)
        if not ok:
            raise RuntimeError("GNE not found")
        kf.predict(u)
        Y.append(y)
        U.append(u)
        R.append(r)
        x = A @ x + B @ u
    return np.array(Y), np.array(U), np.array(R)


# ----------------------------------------------------------------------------
# Figure 1
# ----------------------------------------------------------------------------
def figure_reference():
    # Desired references r_des of Figure 1: from the equilibrium of the initial reference, each reference
    # changes twice
    k = np.arange(ex.N_cl)
    R = np.tile([2.15, -2.53, 1.92], (ex.N_cl, 1))
    for j, (t1, r1, t2, r2) in enumerate([(20, 2.8, 40, -2.5), (30, 2.7, 50, -2.8), (10, 2.6, 35, -2.3)]):
        R[k >= t1, j] = r1
        R[k >= t2, j] = r2
    return R


def plot(data, game, w, Rdes, Yg, Ug, Rg, filename="learn_gtmpc_example.pdf"):
    # Outputs (top row) and inputs (bottom row) of the agents and of the learned GT-MPC law tracking r_des,
    # and of the agents tracking the references Rg of the generator; the red circles mark the dip and the bump
    # of y_1 removed by the generator
    plant, Px, Pu, Pxm = data["plant"], data["Px"], data["Pu"], data["Pxm"]
    N, ymax, umax = ex.N, ex.ymax, ex.umax
    k = np.arange(Rdes.shape[0])
    Yag, Uag = ex.run_agents(np.random.default_rng([ex.seed, 0]), plant, data["agents"], Px @ Rdes[0], Pu @ Rdes[0],
                             Rdes)
    Y, U, _ = ex.run_gtmpc(np.random.default_rng([ex.seed, 0]), plant, GTMPCController(game, w), data["kf"],
                           Px @ Rdes[0], Pxm @ Rdes[0], Pu @ Rdes[0], Rdes)
    plt.rcParams.update({"axes.labelsize": 26, "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22,
                         "text.usetex": shutil.which("latex") is not None, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{times}"})
    fig, ax = plt.subplots(2, N, figsize=(15, 6.), sharex=True)
    ylim = 1.05 * max(1.2 * ymax, np.max(np.abs(Rg)))
    for j in range(N):
        ax[0, j].step(k, Rg[:, j], where="post", color="C1", lw=1.5, label="generated $r$")
        ax[0, j].plot(k, Yg[:, j], color="C2", lw=3.5, label="agents with generator")
        ax[0, j].plot(k, Yag[:, j], "k", lw=3.5, label="agents")
        ax[0, j].plot(k, Y[:, j], color="C0", ls="--", lw=3.5, label="GT-MPC")
        ax[0, j].step(k, Rdes[:, j], where="post", color="gray", ls=":", lw=2.5, label="$r^\\mathrm{des}$")
        ax[0, j].set_ylabel(f"$y_{j + 1}$")
        ax[0, j].set_yticks([-ymax, 0., ymax])
        ax[0, j].set_ylim(-ylim, ylim)
        ax[1, j].step(k, Ug[:, j], color="C2", lw=3.5, where="post")
        ax[1, j].step(k, Uag[:, j], "k", lw=3.5, where="post")
        ax[1, j].step(k, U[:, j], color="C0", ls="--", lw=3.5, where="post")
        ax[1, j].set_ylabel(f"$u_{j + 1}$")
        ax[1, j].set_yticks([-0.1, 0., 0.1])
        ax[1, j].set_ylim(-1.2 * umax, 1.2 * umax)
        ax[1, j].set_xlabel("time step $k$")
        for a, bound in ((ax[0, j], ymax), (ax[1, j], umax)):
            a.axhline(bound, color="r", ls="--", lw=2)
            a.axhline(-bound, color="r", ls="--", lw=2)
            a.grid(True)
    for center in ((34., 1.95), (53., -1.95)):
        ax[0, 0].add_patch(Ellipse(center, 15.6, 3.24, fill=False, color="r", lw=2.5, zorder=5))
    handles, labels = ax[0, 0].get_legend_handles_labels()
    idx = [labels.index(n) for n in ("agents", "GT-MPC", "$r^\\mathrm{des}$", "agents with generator",
                                     "generated $r$")]
    fig.legend([handles[i] for i in idx], [labels[i] for i in idx], loc="upper center", ncol=len(idx),
               handlelength=2., frameon=False, columnspacing=1.2)
    fig.tight_layout(rect=(0., 0., 1., 0.91))
    fig.savefig(filename)
    print(f"Saved {filename}")


def main():
    data = ex.prepare_data()
    game, w, _ = ex.learn(data, ex.Tu)
    Rdes = figure_reference()
    Yg, Ug, Rg = run(data, ReferenceGenerator(game, w, data["model"], omega, sigma), Rdes)
    plot(data, game, w, Rdes, Yg, Ug, Rg)


if __name__ == "__main__":
    main()
