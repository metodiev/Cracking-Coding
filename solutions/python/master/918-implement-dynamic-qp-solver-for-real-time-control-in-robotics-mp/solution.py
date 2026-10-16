"""Task 918 - Implement dynamic QP solver for real-time control in robotics (MPC small demo).  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

MPC of a double integrator x_{k+1} = A x_k + B u_k.  Condensing the horizon makes the future states
affine in the controls, so the quadratic tracking cost becomes the box QP min 0.5 u'Hu + g'u with
H = 2 (B_bar' Q_bar B_bar + R_bar) > 0 and the Riccati solution as the terminal weight (the
unconstrained problem is then the infinite-horizon LQR, which is what lets the loop settle).  The
solver is projected gradient with the Gershgorin Lipschitz bound; optimality is the box KKT system.
Verified: P solves the Riccati equation, condensation equals an honest rollout, the free solve
matches -H^{-1}g, the constrained solve is KKT-optimal, no worse than the clipped free optimum and
equal to an exhaustive active-set oracle, the objective never rises, degenerate sizes behave, and a
closed loop respects the actuator limits and settles.  Simplified: dense condensation, no terminal
set, state constraints or warm start.
"""
import math
from itertools import product

DT = 0.1

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def transpose(a):
    return [list(col) for col in zip(*a)]

def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]

def matvec(a, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]

def solve_linear(matrix, rhs):
    """Gauss-Jordan with partial pivoting on [matrix | rhs]; None when (numerically) singular."""
    n = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-13:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        a[col] = [v / a[col][col] for v in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][n] for i in range(n)]

def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

def qp_value(H, g, u):
    return 0.5 * sum(u[i] * sum(H[i][j] * u[j] for j in range(len(u))) for i in range(len(u))) \
        + sum(g[i] * u[i] for i in range(len(u)))

def solve_box_qp(H, g, lo, hi, max_iter=20000, tol=1e-14, trace=None):
    """Projected gradient on 0.5 u'Hu + g'u over [lo, hi]; returns (u, sweeps).  Every iterate is
    clamped into the box (feasible throughout); `trace` collects the objective per sweep.
    """
    n = len(g)
    if n == 0:
        return [], 0
    step = 1.0 / max(sum(abs(x) for x in row) for row in H)          # Gershgorin bound of H
    u = [min(max(0.0, lo[i]), hi[i]) for i in range(n)]
    for sweep in range(1, max_iter + 1):
        grad = [v + gi for v, gi in zip(matvec(H, u), g)]
        moved = 0.0
        for i in range(n):
            candidate = min(max(u[i] - step * grad[i], lo[i]), hi[i])
            moved = max(moved, abs(candidate - u[i]))
            u[i] = candidate
        if trace is not None:
            trace.append(qp_value(H, g, u))
        if moved < tol:
            return u, sweep
    return u, max_iter

def kkt_gap(H, g, lo, hi, u):
    """Max violation of the box KKT system (grad = 0 free, grad >= 0 at lo, grad <= 0 at hi)."""
    grad = [v + gi for v, gi in zip(matvec(H, u), g)]
    return max((max(-grad[i], 0.0) if u[i] <= lo[i] + 1e-9 else
                max(grad[i], 0.0) if u[i] >= hi[i] - 1e-9 else abs(grad[i]))
               for i in range(len(u)))

def qp_bruteforce(H, g, lo, hi):
    """Independent oracle: enumerate the active set (each coordinate free, at lo, or at hi), solve
    the free block exactly and keep the feasible KKT points.  Costly in n, so n <= 4 only.
    """
    n = len(g)
    best = None
    for choice in product(*[(None, lo[i], hi[i]) for i in range(n)]):
        u = [0.0 if c is None else c for c in choice]
        free = [i for i in range(n) if choice[i] is None]
        if free:
            sub = [[H[i][j] for j in free] for i in free]
            rhs = [-(g[i] + sum(H[i][j] * u[j] for j in range(n) if j not in free)) for i in free]
            solution = solve_linear(sub, rhs)
            if solution is None or any(not lo[i] - 1e-9 <= v <= hi[i] + 1e-9
                                       for i, v in zip(free, solution)):
                continue
            for i, v in zip(free, solution):
                u[i] = v
        value = qp_value(H, g, u)
        if kkt_gap(H, g, lo, hi, u) < 1e-7 and (best is None or value < best[0]):
            best = (value, u)
    return best

def condensed_terms(A, B, Q, R, N, terminal):
    """H, M (g = 2 M x0) and the affine maps: powers[m-1] = A^m and block row m of B_bar is
    [A^{m-j} B for j <= m], so x_m = A^m x0 + B_m u and is weighted by Q (Riccati P at m = N).
    """
    nx, nu = len(A), len(B[0])
    powers = [[[1.0 if i == j else 0.0 for j in range(nx)] for i in range(nx)]]
    for _ in range(N):
        powers.append(matmul(powers[-1], A))
    dim = N * nu
    H = [[0.0] * dim for _ in range(dim)]
    M = [[0.0] * nx for _ in range(dim)]
    b_bar = []
    for m in range(1, N + 1):
        b_row = [[0.0] * dim for _ in range(nx)]           # nx x dim: controls -> x_m
        for j in range(1, m + 1):
            block = matmul(powers[m - j], B)
            for i in range(nx):
                b_row[i][(j - 1) * nu:j * nu] = block[i]
        b_bar.extend(b_row)
        weight = terminal if m == N else Q
        wb, wa = matmul(weight, b_row), matmul(weight, powers[m])
        for i in range(dim):
            for j in range(dim):
                H[i][j] += 2.0 * sum(b_row[k][i] * wb[k][j] for k in range(nx))
            for j in range(nx):
                M[i][j] += sum(b_row[k][i] * wa[k][j] for k in range(nx))
    for step in range(N):
        for i in range(nu):
            for j in range(nu):
                H[step * nu + i][step * nu + j] += 2.0 * R[i][j]
    return H, M, powers[1:], b_bar

def dare_solution(A, B, Q, R, iterations=300):
    """Iterate the Riccati recursion P <- Q + A'PA - A'PB (R + B'PB)^{-1} B'PA to its fixed point.
    Returns (P, max|change| of the last sweep), i.e. the DARE residual, for checking.
    """
    nx, nu = len(A), len(B[0])
    at = transpose(A)
    p = [row[:] for row in Q]
    gap = float("inf")
    for _ in range(iterations):
        at_p = matmul(at, p)
        bt_p = matmul(transpose(B), p)
        s_plus = [[matmul(bt_p, B)[i][j] + R[i][j] for j in range(nu)] for i in range(nu)]
        bt_pa = matmul(bt_p, A)
        solved = [solve_linear(s_plus, [bt_pa[i][j] for i in range(nu)]) for j in range(nx)]
        if any(column is None for column in solved):
            break
        at_pa, at_pb = matmul(at_p, A), matmul(at_p, B)
        nxt = [[Q[i][j] + at_pa[i][j] - sum(at_pb[i][k] * solved[j][k] for k in range(nu))
                for j in range(nx)] for i in range(nx)]
        gap = max(abs(p[i][j] - nxt[i][j]) for i in range(nx) for j in range(nx))
        p = nxt
        if gap < 1e-15:
            break
    return p, gap

def simulate_mpc(A, B, H, M, bound, x0, reference, steps):
    """Closed loop: solve the QP each sample and apply the first control only.  The reference enters
    as an error-state offset, so one time-invariant condensed problem serves every sample.
    """
    lo, hi = [-bound] * len(H), [bound] * len(H)
    state = [x0[0] - reference, x0[1]]
    positions, applied = [], []
    for _ in range(steps):
        control, _ = solve_box_qp(H, [2.0 * v for v in matvec(M, state)], lo, hi)
        applied.append(control[0])
        state = [p + q for p, q in zip(matvec(A, state), matvec(B, [control[0]]))]
        positions.append(state[0] + reference)
    return positions, applied, state

def main():
    A, B = [[1.0, DT], [0.0, 1.0]], [[0.5 * DT * DT], [DT]]
    Q, R, horizon = [[1.0, 0.0], [0.0, 0.1]], [[0.5]], 5
    terminal, residual = dare_solution(A, B, Q, R)
    H, M, powers, b_bar = condensed_terms(A, B, Q, R, horizon, terminal)

    # 1. Riccati weight solved; condensed Hessian symmetric positive definite.
    check(residual < 1e-12, "terminal weight must solve the Riccati equation")
    dirs = [[1.0 if i == k else 0.0 for i in range(horizon)] for k in range(horizon)]
    dirs += [[1.0] * horizon, [1.0 if i % 2 else -1.0 for i in range(horizon)]]
    zero = [0.0] * horizon
    check(all(abs(H[i][j] - H[j][i]) < 1e-12 for i in range(horizon) for j in range(horizon))
          and all(qp_value(H, zero, d) > 0.0 for d in dirs),
          "condensed Hessian must be symmetric positive definite")

    # 2. Condensation reproduces the honest rollout for an arbitrary control sequence.
    x0, probe = [0.7, -0.4], [0.3, -0.55, 0.11, 0.9, -1.0]
    stacked, honest, state = [], [], list(x0)
    for m, control in enumerate(probe, start=1):
        state = [p + q for p, q in zip(matvec(A, state), matvec(B, [control]))]
        honest.extend(state)
        stacked.extend([p + q for p, q in zip(matvec(powers[m - 1], x0),
                                                matvec(b_bar[(m - 1) * 2:m * 2], probe))])
    check(max_abs(stacked, honest) < 1e-12, "condensation must equal the step-by-step rollout")

    # 3. Wide bounds reproduce the unconstrained closed form -H^{-1} g.
    g = [2.0 * v for v in matvec(M, x0)]
    free_solution, sweeps = solve_box_qp(H, g, [-50.0] * horizon, [50.0] * horizon)
    closed_form = solve_linear(H, [-v for v in g])
    check(max_abs(free_solution, closed_form) < 1e-6 and sweeps < 20000,
          "unconstrained solve must match -H^{-1} g")

    # 4. Tight bounds: KKT optimal, and no worse than the clipped free optimum.
    lo, hi = [-0.35] * horizon, [0.35] * horizon
    constrained, _ = solve_box_qp(H, g, lo, hi)
    clipped = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(closed_form)]
    check(kkt_gap(H, g, lo, hi, constrained) < 1e-6
          and qp_value(H, g, constrained) <= qp_value(H, g, clipped) + 1e-9,
          "solver output must be KKT-optimal and no worse than the clipped optimum")

    # 5. Independent oracle: exhaustive active-set enumeration on four small instances.
    hs, ms, _, _ = condensed_terms(A, B, Q, R, 4, terminal)
    for start, lower, upper in [([0.0, 0.0], -0.2, 0.2), ([1.2, 0.5], -0.6, 0.4),
                                ([0.3, -0.9], -0.05, 0.9), ([-0.8, 0.2], -1.0, 1.0)]:
        gs, bounds = [2.0 * v for v in matvec(ms, start)], ([lower] * 4, [upper] * 4)
        fast, _ = solve_box_qp(hs, gs, *bounds)
        oracle = qp_bruteforce(hs, gs, *bounds)
        check(oracle is not None and max_abs(fast, oracle[1]) < 1e-5
              and abs(qp_value(hs, gs, fast) - oracle[0]) < 1e-6,
              "projected gradient must match the active-set oracle")

    # 6. The objective never increases along the sweeps; 7. degenerate sizes.
    trace = []
    solve_box_qp(hs, [2.0 * v for v in matvec(ms, [1.2, 0.5])], [-0.6] * 4, [0.4] * 4, trace=trace)
    pinned, _ = solve_box_qp(H, g, [0.25] * horizon, [0.25] * horizon)
    empty_control, empty_sweeps = solve_box_qp([], [], [], [])
    check(len(trace) > 1 and all(trace[i + 1] <= trace[i] + 1e-12 for i in range(len(trace) - 1)),
          "projected gradient must never increase the objective")
    check(max_abs(pinned, [0.25] * horizon) < 1e-9 and empty_control == [] and empty_sweeps == 0,
          "equal bounds pin the control and a zero-horizon QP is empty")

    # 8. Closed loop: limits respected, setpoint reached, held and not overshot; 9. deterministic.
    reference = 1.0
    positions, applied, final_state = simulate_mpc(A, B, H, M, 1.0, [0.0, 0.0], reference, 120)
    check(max(applied) <= 1.0 + 1e-9 and min(applied) >= -1.0 - 1e-9,
          "applied acceleration must stay inside the actuator limits")
    check(abs(positions[-1] - reference) < 1e-3 and max(positions) < reference + 0.06
          and min(positions[50:]) > reference - 0.06
          and all(abs(p - reference) < 1e-3 for p in positions[-5:]),
          "closed loop must reach, hold and not overshoot the setpoint")
    again, again_applied, _ = simulate_mpc(A, B, H, M, 1.0, [0.0, 0.0], reference, 120)
    check(again_applied == applied and again == positions, "the controller must be deterministic")
    print(f"box QP: {len(H)} controls, {sweeps} sweeps to the free optimum, DARE residual {residual:.1e}")
    print(f"final position {positions[-1]:.6f}, peak position {max(positions):.6f}, "
          f"final velocity {final_state[1]:.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
