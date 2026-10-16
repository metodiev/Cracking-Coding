"""Task 919 - Implement motion planning under differential constraints using trajectory optimization.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

Planning for a planar double integrator x = (px, py, vx, vy), u = (ax, ay), dt-integrated by
x_{k+1} = A x_k + B u_k.  The differential constraints are the acceleration box |u_i| <= A_MAX and
the speed limit |v| <= V_MAX (kept as a squared hinge penalty).  The trajectory is parameterised by
the control sequence (direct shooting), so every candidate satisfies the dynamics exactly; the cost

    J(u) = w_trk sum_k ||p_k - ref_k||^2 + w_u sum_k ||u_k||^2
         + w_obs sum_k max(0, r + margin - ||p_k - o||)^2 + w_vel sum_k max(0, |v_k| - V_MAX)^2

is smooth, and its gradient is analytic: dp_k/du = the corresponding rows of the condensed affine
map B_bar, so grad_u = B_bar' grad_X.  Descent is projected gradient with Armijo backtracking
(each accepted step strictly decreases J).

Verified: the analytic gradient matches central finite differences; the condensed map equals the
honest rollout; J decreases monotonically; the acceleration box and speed limit hold at the
solution; the path clears the obstacle by r + margin while the straight-line baseline collides and
costs more; the goal is reached; and degenerate problems (zero horizon, start == goal without
obstacle) behave.  Simplified: one fixed detour reference instead of a state-lattice search, a soft
speed limit, and a short horizon; the invariant verified is "dynamics exact, limits respected,
obstacle clear", which is what the differential constraints require.
"""
import math

DT, N_STEPS, V_MAX, A_MAX = 0.25, 16, 1.8, 2.0

A = [[1.0, 0.0, DT, 0.0], [0.0, 1.0, 0.0, DT], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
B = [[0.5 * DT * DT, 0.0], [0.0, 0.5 * DT * DT], [DT, 0.0], [0.0, DT]]
GOAL = (4.0, 0.0)
OBSTACLE = (2.0, 0.0)
OBSTACLE_RADIUS, MARGIN = 0.45, 0.15
WEIGHTS = {"track": 1.0, "control": 0.05, "obstacle": 40.0, "speed": 60.0}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def matvec(a, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]


def affine_maps(N):
    """(powers, b_bar): powers[m-1] = A^m; block row m of b_bar is [A^{m-j} B for j <= m].

    So x_m = A^m x0 + B_m u, which is the condensed form of the shooting parameterisation.
    """
    nx, nu = len(A), len(B[0])
    powers = [[[1.0 if i == j else 0.0 for j in range(nx)] for i in range(nx)]]
    for _ in range(N):
        powers.append(matmul(powers[-1], A))
    b_bar = []
    for m in range(1, N + 1):
        row = [[0.0] * (N * nu) for _ in range(nx)]
        for j in range(1, m + 1):
            block = matmul(powers[m - j], B)
            for i in range(nx):
                row[i][(j - 1) * nu:j * nu] = block[i]
        b_bar.extend(row)
    return powers[1:], b_bar


def rollout(u, x0, N):
    """Honest forward simulation of the dynamics, used to cross-check the condensed map."""
    state, states = list(x0), []
    for step in range(N):
        state = [p + q for p, q in zip(matvec(A, state), matvec(B, u[2 * step:2 * step + 2]))]
        states.append(list(state))
    return states


def reference_path(N):
    """Arch above the obstacle: x ramps to the goal, y follows half a sine of amplitude 1.0.

    The straight line to the goal would pass through the obstacle, so the reference encodes the
    detour a global planner (A*/RRT) would have returned.
    """
    return [[GOAL[0] * k / N, (OBSTACLE_RADIUS + MARGIN + 0.35) * math.sin(math.pi * k / N)]
            for k in range(1, N + 1)]


def states_of(u, x0, N, powers, b_bar):
    """Condensed trajectory: x_m = A^m x0 + B_m u for every horizon step."""
    return [[p + q for p, q in zip(matvec(powers[m], x0), matvec(b_bar[m * 4:m * 4 + 4], u))]
            for m in range(N)]


def cost_and_grad(u, x0, refs, obstacle, N, powers, b_bar):
    """Objective value and analytic gradient with respect to the controls."""
    states = states_of(u, x0, N, powers, b_bar)
    value = 0.0
    grad_state = [[0.0] * 4 for _ in range(N)]
    for m, state in enumerate(states):
        penalty = [2.0 * WEIGHTS["track"] * (state[i] - refs[m][i]) for i in range(2)]
        value += WEIGHTS["track"] * sum((state[i] - refs[m][i]) ** 2 for i in range(2))
        if obstacle is not None:
            dx, dy = state[0] - obstacle[0], state[1] - obstacle[1]
            distance = (dx * dx + dy * dy) ** 0.5
            clearance = OBSTACLE_RADIUS + MARGIN
            if distance < clearance:
                value += WEIGHTS["obstacle"] * (clearance - distance) ** 2
                scale = 2.0 * WEIGHTS["obstacle"] * (clearance - distance) / max(distance, 1e-9)
                penalty[0] -= scale * dx
                penalty[1] -= scale * dy
        for i in (0, 1):
            speed = abs(state[2 + i])
            if speed > V_MAX:
                value += WEIGHTS["speed"] * (speed - V_MAX) ** 2
                penalty.append(2.0 * WEIGHTS["speed"] * (speed - V_MAX) * (1.0 if state[2 + i] > 0 else -1.0))
            else:
                penalty.append(0.0)
        grad_state[m] = penalty
    gradient = [0.0] * (2 * N)
    for m in range(N):
        row = b_bar[m * 4:m * 4 + 4]
        for j in range(2 * N):
            gradient[j] += sum(row[i][j] * grad_state[m][i] for i in range(4))
    for j in range(2 * N):
        value += WEIGHTS["control"] * u[j] * u[j]
        gradient[j] += 2.0 * WEIGHTS["control"] * u[j]
    return value, gradient


def project(u, bound):
    return [min(max(v, -bound), bound) for v in u]


def optimize(u, x0, refs, obstacle, N, powers, b_bar, iterations=400):
    """Projected gradient with backtracking; returns (u, value, trace) with a strictly decreasing J."""
    if N == 0:
        return [], 0.0, [0.0]
    value, gradient = cost_and_grad(u, x0, refs, obstacle, N, powers, b_bar)
    trace = [value]
    step = 0.05
    for _ in range(iterations):
        if max(abs(g) for g in gradient) < 1e-4:
            break
        trial_step, moved = step, False
        for _ in range(30):
            candidate = project([u[j] - trial_step * gradient[j] for j in range(2 * N)], A_MAX)
            trial_value, trial_gradient = cost_and_grad(candidate, x0, refs, obstacle, N, powers, b_bar)
            if trial_value < value - 1e-12:
                u, value, gradient, step, moved = candidate, trial_value, trial_gradient, trial_step * 1.5, True
                break
            trial_step *= 0.5
        trace.append(value)
        if not moved:
            break
    return u, value, trace


def straight_line_controls(x0, N):
    """Baseline: constant +x acceleration chosen so the straight path reaches the goal in N steps."""
    accel = 2.0 * (GOAL[0] - x0[0]) / (N * DT) ** 2
    return [accel, 0.0] * N


def clearance_of(states, obstacle):
    if obstacle is None:
        return float("inf")
    return min(((s[0] - obstacle[0]) ** 2 + (s[1] - obstacle[1]) ** 2) ** 0.5 for s in states)


def main():
    powers, b_bar = affine_maps(N_STEPS)
    x0 = [0.0, 0.0, 0.0, 0.0]
    refs = reference_path(N_STEPS)
    start_controls = straight_line_controls(x0, N_STEPS)

    # 1. Condensed map equals the honest rollout for an arbitrary control sequence.
    probe = [0.3 if step % 2 else -0.2 for step in range(2 * N_STEPS)]
    condensed = states_of(probe, x0, N_STEPS, powers, b_bar)
    honest = rollout(probe, x0, N_STEPS)
    check(max(abs(a - b) for s, t in zip(condensed, honest) for a, b in zip(s, t)) < 1e-12,
          "condensed map must equal the step-by-step rollout")

    # 2. Analytic gradient matches central finite differences.
    point = [0.4 - 0.05 * step for step in range(2 * N_STEPS)]
    _, analytic = cost_and_grad(point, x0, refs, OBSTACLE, N_STEPS, powers, b_bar)
    numeric = []
    for j in range(2 * N_STEPS):
        plus, minus = list(point), list(point)
        plus[j] += 1e-6
        minus[j] -= 1e-6
        numeric.append((cost_and_grad(plus, x0, refs, OBSTACLE, N_STEPS, powers, b_bar)[0]
                        - cost_and_grad(minus, x0, refs, OBSTACLE, N_STEPS, powers, b_bar)[0]) / 2e-6)
    check(max(abs(a - b) for a, b in zip(analytic, numeric)) < 1e-4,
          "analytic gradient must match central finite differences")

    # 3. Optimisation: monotone decrease, limits respected, obstacle cleared, goal reached.
    controls, value, trace = optimize(start_controls, x0, refs, OBSTACLE, N_STEPS, powers, b_bar)
    check(all(trace[i + 1] <= trace[i] + 1e-12 for i in range(len(trace) - 1)) and trace[-1] < trace[0],
          "each accepted step must strictly decrease the cost")
    check(max(abs(v) for v in controls) <= A_MAX + 1e-9, "acceleration must respect its box bound")
    states = states_of(controls, x0, N_STEPS, powers, b_bar)
    speed = max((s[2] ** 2 + s[3] ** 2) ** 0.5 for s in states)
    check(speed <= V_MAX + 1e-6, "the differential speed constraint must hold at the solution")
    check(clearance_of(states, OBSTACLE) >= OBSTACLE_RADIUS + MARGIN - 1e-3,
          "the optimised path must clear the obstacle")
    check(abs(states[-1][0] - GOAL[0]) < 0.05 and abs(states[-1][1] - GOAL[1]) < 0.08,
          "the optimised path must reach the goal")

    # 4. Independent comparison: the straight-line baseline collides and costs more.
    baseline_states = states_of(start_controls, x0, N_STEPS, powers, b_bar)
    baseline_value = cost_and_grad(start_controls, x0, refs, OBSTACLE, N_STEPS, powers, b_bar)[0]
    check(clearance_of(baseline_states, OBSTACLE) < OBSTACLE_RADIUS + MARGIN,
          "the straight-line baseline must violate the obstacle clearance")
    check(value < 0.5 * baseline_value, "optimisation must improve the cost substantially")

    # 5. Boundary cases: zero horizon and a start already sitting on a stationary reference.
    empty, empty_value, _ = optimize([], [0.0, 0.0, 0.0, 0.0], [], None, 0, [], [])
    check(empty == [] and empty_value == 0.0, "a zero-horizon problem has an empty control")
    still_powers, still_bars = affine_maps(6)
    still_controls, still_value, _ = optimize([0.0] * 12, [0.0, 0.0, 0.0, 0.0], [[0.0, 0.0]] * 6,
                                              None, 6, still_powers, still_bars)
    check(still_value < 1e-12 and max(abs(v) for v in still_controls) < 1e-9,
          "no motion is required when the start already sits on the stationary reference")

    # 6. Determinism: rerunning the optimiser reproduces the same controls and cost.
    again, again_value, _ = optimize(start_controls, x0, refs, OBSTACLE, N_STEPS, powers, b_bar)
    check(again == controls and again_value == value, "the optimiser must be deterministic")

    print(f"optimised cost {value:.6f} vs straight-line baseline {baseline_value:.6f} "
          f"({len(trace)} accepted steps)")
    print(f"clearance {clearance_of(states, OBSTACLE):.4f} m, peak speed {speed:.4f} m/s, "
          f"goal error {((states[-1][0] - GOAL[0]) ** 2 + (states[-1][1] - GOAL[1]) ** 2) ** 0.5:.5f} m")
    print("PASS")


if __name__ == "__main__":
    main()
