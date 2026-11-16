"""
Numerical example of the paper "Learning monotone game-theoretic linear MPC".

A randomly generated process with 3 inputs, 3 outputs, and 4 states is operated
by 3 agents, each commanding one input. The agents play a non-cooperative
GT-MPC game with conflicting, agent-specific weights, the true process model
and state, and soft output constraints. Data are collected in closed-loop
experiments in which the agents track piecewise-constant references, with no
additional excitation. A linear model is identified from these data with
jax-sysid, the states are reconstructed by a Kalman filter, and the weights of
a strongly monotone GT-MPC game that imitates the agents are learned by the
bilevel problem (L-BFGS-B). The learned GT-MPC law is then validated in closed
loop against the agents (Section 4 and Table 1 of the paper; Figure 1 is
produced by example_reference_generator.py).

Requirements: numpy, scipy, matplotlib, jax, jax-sysid, nashopt.

Reference: A. Bemporad and T. Tatarenko, "Learning monotone game-theoretic linear MPC", arXiv 2611.17504, 2026.
"""

import sys
import time
import numpy as np
from jax_sysid.utils import compute_scores
from gtmpc_learn import (identify_model, equilibrium, KalmanFilter, GTMPCGame, learn_bilevel,
                         GTMPCController)


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
seed = 10
n, N = 4, 3                 # states, agents (one input and one output each)
umin, umax = -0.15, 0.15    # input bounds
dumax = 0.1                 # input-increment bounds |u(k) - u(k-1)| <= dumax
ymax = 3.                   # output bounds |y| <= ymax
noise = 0.05                # amplitude of the uniform output measurement noise

# process generation
rho_ol = 0.95               # spectral radius of the open-loop state matrix
cpl = 0.5                   # coupling strength
g0 = 25.                    # diagonal DC gains
max_cpl_dc = 0.25           # off-diagonal DC gains are at most max_cpl_dc * g0

# GT-MPC horizons, used by the agents and by the learned GT-MPC law
T, Tu = 10, 4               # prediction and control horizons (output-constraint horizon Tc = T)

# agents: GT-MPC game with weights Qy_i = wy diag(roll(qy_pattern, i)), Qdu_i = qdu_true[i] e_i e_i'
wy = 1e-4
qy_pattern = [1.0, 0.6, 0.1]   # weights on the own output, the next output, and the last output
qdu_true = [0.05, 0.10, 0.20]  # own input-increment weights
qeps_true, Qeps_true = 3e-3, 0.5  # slack penalties

# data collection
M_exp = 7                   # experiments
Nsim = 200                  # steps per experiment
r_hold, r_amp = 15, 5.      # the references change every r_hold steps, uniform in [-r_amp, r_amp]^N
r0_amp = 1.5                # initial equilibria of references uniform in [-r0_amp, r0_amp]^N

# learning
mu, eta = 0.1, 1e-3         # monotonicity constant, lower bound on the own increment weights
rho = 1e-5                  # regularization
delta = 0.01                # monotonicity margin of the initial game
gamma, rho_s = 0.9, 10.     # contraction rate of the learned Lyapunov function, weight of the stability penalty
qeps0, Qeps0 = 1e-3, 0.1    # initial slack penalties

# closed-loop validation
M_cl, N_cl = 10, 80         # reference steps and their length

# Kalman filter
Q_kf, R_kf, P0_kf = 1e-6 * np.eye(n), noise ** 2 / 3. * np.eye(N), 1e-4 * np.eye(n)


# ----------------------------------------------------------------------------
# Process and agents
# ----------------------------------------------------------------------------
def random_process(rng):
    # Weakly coupled process: input i mainly drives state i, which mainly drives output i
    off = 1. - np.eye(n)
    A = np.diag(rng.uniform(0.6, 0.9, n)) + cpl * rng.uniform(-1, 1, (n, n)) * off
    A = rho_ol * A / np.max(np.abs(np.linalg.eigvals(A)))
    mask = np.vstack((1. - np.eye(N), np.ones((n - N, N))))
    B = np.vstack((np.diag(rng.uniform(0.5, 1., N)), np.zeros((n - N, N)))) + cpl * rng.uniform(-1, 1, (n, N)) * mask
    C = np.hstack((np.diag(rng.uniform(1., 2., N)), np.zeros((N, n - N)))) + cpl * rng.uniform(-1, 1, (N, n)) * mask.T
    G0 = C @ np.linalg.solve(np.eye(n) - A, B)
    return A, B * (g0 / np.diag(G0))[None, :], C


def weakly_coupled_process(rng):
    # Random process whose off-diagonal DC gains are at most max_cpl_dc * g0
    while True:
        A, B, C = random_process(rng)
        G0 = C @ np.linalg.solve(np.eye(n) - A, B)
        if np.max(np.abs(G0 - np.diag(np.diag(G0)))) <= max_cpl_dc * g0:
            return A, B, C


def agents_game(plant):
    # GT-MPC game played by the agents: true model and state, soft output constraints
    game = GTMPCGame(*plant, [[i] for i in range(N)], T, Tu, umin, umax, dumax, -ymax, ymax, mu=mu, eta=eta)
    Qy = [wy * np.diag(np.roll(qy_pattern, i)) for i in range(N)]
    Qdu = [game.complete_increment_weight(i, game.increment_weight(i, np.array([[qdu_true[i]]]), np.zeros((1, N - 1))))
           for i in range(N)]
    return game, dict(Qy=Qy, Qdu=Qdu, qeps=qeps_true * np.ones(N), Qeps=Qeps_true * np.ones(N))


def run_agents(rng, plant, controller, x0, u0, R):
    # Closed loop with the agents tracking the references R (one row per step). Returns measured
    # outputs and inputs.
    A, B, C = plant
    x, u = x0, u0.copy()
    Y, U = [], []
    for t in range(R.shape[0]):
        y = C @ x + noise * rng.uniform(-1, 1, N)
        u, ok, _ = controller(x, u, R[t])
        if not ok:
            raise RuntimeError("GNE not found")
        Y.append(y)
        U.append(u)
        x = A @ x + B @ u
    return np.array(Y), np.array(U)


def collect(rng, plant, controller, m):
    # m experiments, each from the equilibrium of a random reference, with piecewise-constant references
    Px, Pu = equilibrium(*plant)
    assert r0_amp * np.max(np.sum(np.abs(Pu), axis=1)) <= umax
    data = []
    for _ in range(m):
        R = np.repeat(rng.uniform(-r_amp, r_amp, (-(-Nsim // r_hold), N)), r_hold, axis=0)[:Nsim]
        r0 = rng.uniform(-r0_amp, r0_amp, N)
        Y, U = run_agents(rng, plant, controller, Px @ r0, Pu @ r0, R)
        data.append(dict(Y=Y, U=U, R=R))
    return data


def run_gtmpc(rng, plant, controller, kf, x0, xh0, u0, R):
    # Closed loop with the GT-MPC law and the Kalman filter on the identified model
    A, B, C = plant
    x, u = x0, u0.copy()
    kf.reset(xh0)
    Y, U, times = [], [], []
    for t in range(R.shape[0]):
        y = C @ x + noise * rng.uniform(-1, 1, N)
        u, ok, dt = controller(kf.update(y), u, R[t])
        if not ok:
            raise RuntimeError("GNE not found")
        kf.predict(u)
        Y.append(y)
        U.append(u)
        times.append(dt)
        x = A @ x + B @ u
    return np.array(Y), np.array(U), times


# ----------------------------------------------------------------------------
# Data, learning, and validation
# ----------------------------------------------------------------------------
def prepare_data():
    # Process, agents, training data, identified model, state estimates, and closed-loop validation runs
    rng = np.random.default_rng(seed)
    plant = weakly_coupled_process(rng)
    Px, Pu = equilibrium(*plant)
    game_ag, w_ag = agents_game(plant)
    agents = GTMPCController(game_ag, w_ag)
    A = game_ag.pseudogradient(w_ag["Qy"], w_ag["Qdu"])[0]
    print(f"Agents' game: lambda_min(sym(A)) = {np.min(np.linalg.eigvalsh((A + A.T) / 2)):.3f}, "
          f"||A - A'||_F / ||A||_F = {np.linalg.norm(A - A.T) / np.linalg.norm(A):.2f}")

    data = collect(rng, plant, agents, M_exp)

    t0 = time.perf_counter()
    model, (Am, Bm, Cm), x0s = identify_model([e["Y"] for e in data], [e["U"] for e in data], n)
    print(f"Identification: {time.perf_counter() - t0:.1f} s")
    val = collect(np.random.default_rng(seed + 1), plant, agents, 3)
    Yhat = [np.array(model.predict(np.array(model.learn_x0(e["U"], e["Y"], verbosity=False)), e["U"])[0])
            for e in val]
    _, bfr, _ = compute_scores(None, None, np.vstack([e["Y"] for e in val]), np.vstack(Yhat), fit="BFR")
    print("Best fit rates on 3 further experiments: " + ", ".join(f"y{i + 1} {b:.1f}%" for i, b in enumerate(bfr)))
    kf = KalmanFilter(Am, Bm, Cm, Q_kf, R_kf, P0_kf)
    train = [dict(U=e["U"], Xh=kf.filter(x0, e["Y"], e["U"]), R=e["R"]) for e, x0 in zip(data, x0s)]

    # Reference steps r0 -> r1 for which the measured outputs of the agents reach the output bounds. Run c
    # uses the noise generator seeded by (seed, c) in the closed loops with the agents and with the GT-MPC laws.
    rng_cl = np.random.default_rng(seed + 2)
    runs, c = [], 0
    while len(runs) < M_cl:
        c += 1
        r1 = rng_cl.uniform(-0.9 * ymax, 0.9 * ymax, N)
        r0 = r1 - np.sign(r1) * rng_cl.uniform(0., 3., N)
        if max(np.max(np.abs(Pu @ r0)), np.max(np.abs(Pu @ r1))) > 0.9 * umax:
            continue
        Y, U = run_agents(np.random.default_rng([seed, c]), plant, agents, Px @ r0, Pu @ r0, np.tile(r1, (N_cl, 1)))
        if np.max(np.abs(Y)) >= ymax:
            runs.append((c, r0, r1, Y, U))
    return dict(plant=plant, agents=agents, w_agents=w_ag, Px=Px, Pu=Pu, train=train, model=(Am, Bm, Cm), kf=kf,
                Pxm=equilibrium(Am, Bm, Cm)[0], runs=runs)


def learn(data, Tu_):
    # GT-MPC law learned by the bilevel problem with control horizon Tu_
    game = GTMPCGame(*data["model"], [[i] for i in range(N)], T, Tu_, umin, umax, dumax, -ymax, ymax, mu=mu,
                     eta=eta)
    t0 = time.perf_counter()
    w, _ = learn_bilevel(game, game.gne_samples(data["train"]), rho=rho, delta=delta, qeps0=qeps0, Qeps0=Qeps0,
                         gamma=gamma, rho_s=rho_s, verbose=False)
    return game, w, time.perf_counter() - t0


def validate(data, game, w):
    # Monotonicity constant, relative imitation errors RMS_u and RMS_y, GNE solution time per step, and
    # spectral radius of the closed loop with the unconstrained Nash equilibrium feedback on the identified model
    ctrl = GTMPCController(game, w)
    Px, Pu, Pxm = data["Px"], data["Pu"], data["Pxm"]
    du = nu = dy = ny = 0.
    tms = []
    for c, r0, r1, Yag, Uag in data["runs"]:
        Y, U, dt = run_gtmpc(np.random.default_rng([seed, c]), data["plant"], ctrl, data["kf"], Px @ r0, Pxm @ r0,
                             Pu @ r0, np.tile(r1, (N_cl, 1)))
        du += np.sum((U - Uag) ** 2)
        nu += np.sum(Uag ** 2)
        dy += np.sum((Y - Yag) ** 2)
        ny += np.sum(Yag ** 2)
        tms += dt
    A = game.pseudogradient(w["Qy"], w["Qdu"])[0]
    return dict(mineig=np.min(np.linalg.eigvalsh((A + A.T) / 2)), rms_u=np.sqrt(du / nu), rms_y=np.sqrt(dy / ny),
                us=1e6 * np.mean(tms), rho_cl=game.closed_loop_spectral_radius(w))


def weights_table(w, wa):
    # Table 1 (left): weights of the agents and learned weights, written as alpha * (w / alpha), where alpha
    # minimizes sum_j ((w_j - alpha wa_j) / wa_j)^2 over the listed weights
    def listed(v):
        return np.hstack([np.diag(Q) for Q in v["Qy"]] + [[v["Qdu"][i][i, i] for i in range(N)], v["qeps"][:1],
                          v["Qeps"][:1]])
    alpha = round(float(np.mean(listed(w) / listed(wa))), 2)
    print(f"\nLearned weights = alpha * (values below), alpha = {alpha:.2f}  [agents' weights in brackets]")
    for i in range(N):
        print(f"  diag(Qy_{i + 1}) = 1e-4 * {np.round(np.diag(w['Qy'][i]) / alpha * 1e4, 2)}  "
              f"[1e-4 * {np.diag(wa['Qy'][i]) * 1e4}]")
    for i in range(N):
        print(f"  Qdu_{i + 1}^{i + 1}{i + 1} = {w['Qdu'][i][i, i] / alpha:.3f}  [{wa['Qdu'][i][i, i]:g}]")
    print(f"  qeps_i = {w['qeps'][0] / alpha:.4f}  [{qeps_true:g}],  "
          f"Qeps_i = {w['Qeps'][0] / alpha:.2f}  [{Qeps_true:g}]")
    offdiag = max(np.max(np.abs(Q - np.diag(np.diag(Q)))) for Q in w["Qy"])
    cross = max(np.max(np.abs(np.delete(w["Qdu"][i][i], i))) for i in range(N))
    print(f"  max |off-diagonal entries of Qy_i| = {offdiag:.1e}, max |cross weights S_i| = {cross:.1e}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(sweep=False):
    data = prepare_data()
    print(f"\n{'Tu':>3} {'time [s]':>9} {'lambda_min':>10} {'RMS_u':>7} {'RMS_y':>7} {'us/step':>8} {'rho_cl':>7}")
    for Tu_ in (range(1, T + 1) if sweep else [Tu]):
        game, w, t = learn(data, Tu_)
        m = validate(data, game, w)
        print(f"{Tu_:3d} {t:9.1f} {m['mineig']:10.4f} {m['rms_u']:7.4f} {m['rms_y']:7.4f} {m['us']:8.0f} "
              f"{m['rho_cl']:7.4f}", flush=True)
    if not sweep:
        weights_table(w, data["w_agents"])


if __name__ == "__main__":
    # python example_paper.py          Tu = 4: Table 1 (left) and row Tu = 4 of Table 1 (right)
    # python example_paper.py --sweep  Table 1 (right): Tu = 1, ..., T
    main(sweep="--sweep" in sys.argv[1:])
