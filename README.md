# gtmpc_learn

Learning strongly monotone game-theoretic MPC (GT-MPC) laws from data.

Several agents operate a common linear plant, each pursuing its own objective
under input and output constraints. From the agents' closed-loop input/output
data, `gtmpc_learn` learns a GT-MPC game whose closed-loop behavior imitates
theirs, e.g., to reproduce it in simulation on a digital twin of the plant. The
learned game is strongly monotone by construction, so the GT-MPC law is well
defined: at every time step, the variational generalized Nash equilibrium (GNE)
of the game exists and is unique, and it is computed efficiently online.

This repository contains the code of the paper

> A. Bemporad and T. Tatarenko, "Learning monotone game-theoretic linear MPC", 2026.
> [arXiv 2611.17504](https://arxiv.org/abs/2611.17504), 2026.


## Method

1. Identify a linear model `x(t+1) = A x(t) + B u(t)`, `y(t) = C x(t)` from
   input/output data using the [jax-sysid](https://github.com/bemporad/jax-sysid)
   package, and reconstruct the state trajectories with a Kalman filter.
2. Learn the weights `Qy_i`, `Qdu_i` and the slack penalties `qeps_i`,
   `Qeps_i` of the GT-MPC game in which agent `i`, given the parameter
   `p = (x_hat, u_prev, r)`, solves

   ```
   min_{x_i, eps_i}  sum_{t=0}^{T-1} (y(t+1) - r)' Qy_i (y(t+1) - r)
                     + sum_{t=0}^{Tu-1} du(t)' Qdu_i du(t) + qeps_i eps_i + Qeps_i eps_i^2
   s.t.  umin <= u_i(t) <= umax,  dumin <= du_i(t) <= dumax,  t = 0, ..., Tu-1
         ymin - (sum_j eps_j) 1 <= y(t) <= ymax + (sum_j eps_j) 1,  t = 1, ..., Tc
         eps_i >= 0
   ```

   where `x_i = (du_i(0), ..., du_i(Tu-1))` are agent `i`'s input increments.
   The measured moves of the agents at each step are interpreted as the first
   moves of the variational GNE of the game at the measured parameter. The
   slack penalties are learned subject to `qeps_i >= 0` and `Qeps_i >= mu/2`.
   The game is constrained to be strongly monotone with a given constant
   `mu`, i.e., `sym(A) >= mu I`, where `A` is the pseudogradient matrix.
3. Apply the first move of the variational GNE of the learned game at each step.

The weights are learned by `learn_bilevel`, a bilevel problem whose inner
problems are the GNE problems of the game, with all the input,
input-increment, and soft output constraints, solved by L-BFGS-B. The
imitation loss compares the first moves of the GNEs with the measured moves.
The GNEs are computed by DR-DAQP and differentiated through the reduced QP in
the multipliers of Bemporad and Tatarenko (2026), with the active constraints
fixed; a differentiable QP solver could be used instead. Strong monotonicity
holds for every value of the parameters, thanks to an exact shift of the
agents' own input-increment weights. With `force_stability=True` (default),
closed-loop stability of the unconstrained Nash equilibrium feedback is also
enforced, by a penalty involving a jointly learned quadratic Lyapunov
function, and certified at the solution. Closed-loop stability can also be
checked with `game.closed_loop_spectral_radius(weights)`.

## Requirements

- Python >= 3.11
- numpy, scipy, matplotlib
- [jax](https://github.com/jax-ml/jax) and [jax-sysid](https://github.com/bemporad/jax-sysid) (system identification)
- [nashopt](https://github.com/bemporad/nashopt) (DR-DAQP solver for the variational GNE)

```
pip install numpy scipy matplotlib jax jax-sysid nashopt
```

The results in the paper were obtained with Python 3.11.15, numpy 1.26.4,
scipy 1.14.1, jax 0.4.31, jax-sysid 1.1.0, and nashopt 1.3.9.
A LaTeX installation is optional; it is used for the figure labels if found.

## Reproducing the paper

```
python example_paper.py
```

runs the numerical example of the paper (Section 4) with control horizon
`Tu = 4`. Three agents play a GT-MPC game with conflicting objectives and soft
output constraints; only their closed-loop operating data are recorded, with
no additional excitation. The script performs data generation,
identification, learning, and closed-loop validation against the agents. It
prints the learning time, the monotonicity constant, the imitation errors
`RMS_u`, `RMS_y`, the solution time per step, and the closed-loop spectral
radius (row `Tu = 4` of Table 1, right), and the learned weights compared with
those of the agents (Table 1, left).

```
python example_paper.py --sweep
```

repeats learning and validation for `Tu = 1, ..., 10` on the same data, which
gives Table 1 (right). Computation times depend on the machine.

```
python example_reference_generator.py
```

runs the reference-optimization example of Section 4.1. With the same data and
the game learned with `Tu = 4`, a reference generator placed on top of the
agents, who keep playing their own GT-MPC game, sends them at each step the
reference

```
r(k) = argmin_r (y_1(r) - r_des,1(k))^2 + omega ||y_{2,3}(r) - r_des,{2,3}(k)||_2^2
                + sigma ||r - r_des(k)||_2^2
```

where `y(r) = C (A x_hat(k) + B u_bar(r))` is the output predicted by the
identified model when the agents apply the GT-MPC action `u_bar(r)` of the
learned game, `omega = 0.02`, and `sigma = 0.01`. The problem is solved by a
Gauss-Newton method, in which the Jacobian of the GNE with respect to `r` is
obtained at the current active set. The script saves Figure 1 of the paper
(`learn_gtmpc_example.pdf`).

## Using the library

```python
from gtmpc_learn import (identify_model, KalmanFilter, GTMPCGame,
                         learn_bilevel, GTMPCController)

# 1. Model identification and state estimation
model, (A, B, C), x0s = identify_model(Ys, Us, nx)
kf = KalmanFilter(A, B, C, Q_kf, R_kf, P0_kf)

# 2. Training data: one dict per experiment
experiments = [dict(U=U, Xh=kf.filter(x0, Y, U), R=R), ...]

# 3. Game structure and learning
game = GTMPCGame(A, B, C, agent_inputs=[[0], [1], [2]], T=10, Tu=4,
                 umin=-0.15, umax=0.15, dumax=0.1, ymin=-3., ymax=3., mu=0.1, eta=1e-3)
weights, info = learn_bilevel(game, game.gne_samples(experiments), force_stability=True, gamma=0.9)
print(info["spectral_radius"], info["stable_certified"])

# 4. Closed loop
controller = GTMPCController(game, weights)
u, ok, solve_time = controller(kf.update(y), u_prev, r)
```

Each experiment is a dictionary with:

- `U` (K x nu): applied inputs;
- `Xh` (K x n): state estimates;
- `R` (K x ny): references, with `NaN` on the outputs that are not tracked.

Bounds can be scalars or vectors and may be infinite. An agent can command
several inputs (`agent_inputs` lists the input indices of each agent).

The learned `weights` are a dictionary with the output weights `Qy`, the
input-increment weights `Qdu`, and the slack penalties `qeps`, `Qeps` of all
the agents.

## Files

- `gtmpc_learn.py`: the library.
- `example_paper.py`: the numerical example of the paper (Section 4, Table 1).
- `example_reference_generator.py`: the reference-optimization example of the paper (Section 4.1, Figure 1).
