"""Task 920 - Implement end-to-end SLAM with loop closure detection and pose graph optimization.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

A scaled-down but complete SLAM pipeline for a planar robot on a closed loop of ten poses.  Each
pose carries a landmark-range descriptor, so a loop closure is a pair whose signatures agree while
being far apart in time; odometry and loop closures both deliver relative transforms T_i^-1 T_j,
corrupted by a deterministic noise stream.  The back end minimises the pose graph residual
r = [R_i^T (t_j - t_i) - t_ij ; wrap(theta_j - theta_i - theta_ij)] by Gauss-Newton on the 3(N-1)
free variables (pose 0 is the gauge), with an analytic Jacobian.  Verified: the Jacobian matches
central finite differences, Gauss-Newton decreases the objective every iteration, the loop closure
is found with no false positives, the trajectory error drops far below odometry-only drift, a
three-pose translation chain matches its closed-form optimum, and degenerate graphs are handled.
Simplified: landmark-range descriptors, one loop edge, dense linear algebra.
"""
import math

MASK64 = (1 << 64) - 1
TWO_POW_53 = 9007199254740992.0
MATCH_TOL, MIN_SEPARATION = 0.25, 3
GROUND_TRUTH = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, math.pi / 2),
                (2.0, 2.0, math.pi / 2), (1.0, 2.0, math.pi), (0.0, 2.0, math.pi),
                (0.0, 1.4, -math.pi / 2), (0.0, 0.7, -math.pi / 2), (0.0, 0.0, 0.0)]
LANDMARKS = [(-1.0, 1.0), (3.0, 1.0), (1.0, -1.0), (1.0, 3.0)]
ODOM_SIGMA = (0.05, 0.05, 0.025)
LOOP_SIGMA = (0.012, 0.012, 0.006)
ODOM_BIAS = (0.05, -0.03, 0.02)      # wheel-calibration bias: pure dead reckoning diverges

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)

def gaussian(state):
    """Sum of three uniforms, rescaled: a cheap deterministic approximation of N(0, 1)."""
    total = 0.0
    for _ in range(3):
        state, raw = splitmix64(state)
        total += (raw >> 11) / TWO_POW_53
    return state, (total - 1.5) * 2.0

def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def rotate(theta, vector):
    c, s = math.cos(theta), math.sin(theta)
    return [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]]

def transpose(theta, vector):
    c, s = math.cos(theta), math.sin(theta)
    return [c * vector[0] + s * vector[1], -s * vector[0] + c * vector[1]]

def relative(pose_a, pose_b):
    """Measured transform T_a^-1 T_b as (dx, dy, dtheta) in the frame of pose a."""
    delta = [pose_b[0] - pose_a[0], pose_b[1] - pose_a[1]]
    local = transpose(pose_a[2], delta)
    return (local[0], local[1], wrap(pose_b[2] - pose_a[2]))

def noisy_relative(pose_a, pose_b, sigma, state, bias=(0.0, 0.0, 0.0)):
    base = relative(pose_a, pose_b)
    out = []
    for value, scale, offset in zip(base, sigma, bias):
        state, noise = gaussian(state)
        out.append(value + scale * noise + offset)
    return state, tuple(out)

def descriptor(pose):
    return [round(math.dist(pose[:2], landmark), 2) for landmark in LANDMARKS]

def detect_loop_closures(poses):
    """Pairs (i, j) whose landmark-range signatures agree and that are far apart in time."""
    signatures = [descriptor(pose) for pose in poses]
    found = []
    for i in range(len(poses)):
        for j in range(i + MIN_SEPARATION, len(poses)):
            if sum(abs(a - b) for a, b in zip(signatures[i], signatures[j])) < MATCH_TOL:
                found.append((i, j))
    return found

def build_graph():
    """Simulate the front end: ground-truth poses -> odometry + loop-closure measurements.

    Each edge carries the information weight (sigma_odom / sigma_edge)^2, so the tighter loop
    measurement is trusted more, exactly as a whittening by the inverse covariance would do.
    """
    state = 0x5EED5EED
    poses = [list(pose) for pose in GROUND_TRUTH]
    edges = []
    for i in range(len(poses) - 1):
        state, measurement = noisy_relative(poses[i], poses[i + 1], ODOM_SIGMA, state, ODOM_BIAS)
        edges.append((i, i + 1, measurement, 1.0))
    closures = detect_loop_closures(poses)
    weight = (ODOM_SIGMA[0] / LOOP_SIGMA[0]) ** 2
    for i, j in closures:
        state, measurement = noisy_relative(poses[i], poses[j], LOOP_SIGMA, state)
        edges.append((i, j, measurement, weight))
    return poses, edges, closures

def residuals(poses, edges):
    """Stacked whitened residuals sqrt(w) * [R_i^T (t_j - t_i) - t_ij ; wrapped angle error]."""
    out = []
    for i, j, (dx, dy, dtheta), weight in edges:
        local = transpose(poses[i][2], [poses[j][0] - poses[i][0], poses[j][1] - poses[i][1]])
        scale = weight ** 0.5
        out.extend([scale * (local[0] - dx), scale * (local[1] - dy),
                    scale * wrap(poses[j][2] - poses[i][2] - dtheta)])
    return out

def jacobian(poses, edges):
    """Dense analytic Jacobian of the residuals w.r.t. the free poses 1..n-1.

    With R = R_i^T and t = t_j - t_i the 3x3 block of edge (i, j) w.r.t. pose i is
    [[-R00, -R01, spin_x], [-R10, -R11, spin_y], [0, 0, -1]] with spin = J R t, J = [[0,1],[-1,0]];
    the block w.r.t. pose j is the same without the spin column and with +1 in the angle slot.
    """
    count = len(poses) - 1
    rows = []
    for i, j, _measurement, weight in edges:
        c, s = math.cos(poses[i][2]), math.sin(poses[i][2])
        rot = [[c, s], [-s, c]]
        spin = rotate(-math.pi / 2, transpose(poses[i][2], [poses[j][0] - poses[i][0],
                                                           poses[j][1] - poses[i][1]]))
        scale = weight ** 0.5
        block_i = [[-rot[0][0], -rot[0][1], spin[0]],
                   [-rot[1][0], -rot[1][1], spin[1]],
                   [0.0, 0.0, -1.0]]
        block_j = [[rot[0][0], rot[0][1], 0.0],
                   [rot[1][0], rot[1][1], 0.0],
                   [0.0, 0.0, 1.0]]
        for k in range(3):
            row = [0.0] * (3 * count)
            if i > 0:
                row[3 * (i - 1):3 * i] = [scale * v for v in block_i[k]]
            if j > 0:
                row[3 * (j - 1):3 * j] = [scale * v for v in block_j[k]]
            rows.append(row)
    return rows

def solve_linear(matrix, rhs):
    """Gauss-Jordan with partial pivoting; None when (numerically) singular."""
    n = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        a[col] = [v / a[col][col] for v in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[i][n] for i in range(n)]

def objective(poses, edges):
    return sum(r * r for r in residuals(poses, edges))

def gauss_newton(poses, edges, iterations=8):
    """Damped Gauss-Newton on the pose graph; returns (poses, trace) with a decreasing objective."""
    poses = [list(pose) for pose in poses]
    count = len(poses) - 1
    trace = [objective(poses, edges)]
    if count == 0 or not edges:
        return poses, trace
    for _ in range(iterations):
        r = residuals(poses, edges)
        jac = jacobian(poses, edges)
        normal = [[sum(row[a] * row[b] for row in jac) for b in range(3 * count)]
                  for a in range(3 * count)]
        for a in range(3 * count):
            normal[a][a] += 1e-9                     # tiny damping for the fixed gauge direction
        gradient = [sum(jac[k][a] * r[k] for k in range(len(r))) for a in range(3 * count)]
        delta = solve_linear(normal, [-g for g in gradient])
        if delta is None:
            break
        for index in range(1, len(poses)):
            for axis in range(3):
                poses[index][axis] += delta[3 * (index - 1) + axis]
        trace.append(objective(poses, edges))
    return poses, trace

def ate(poses, truth):
    return sum(math.dist(p[:2], t[:2]) for p, t in zip(poses, truth)) / len(poses)

def odometry_chain(poses, edges):
    """Dead-reckoning integration of the consecutive odometry edges (the 'before' baseline)."""
    out = [list(poses[0])]
    for i, _j, (dx, dy, dtheta), _weight in [e for e in edges if e[1] == e[0] + 1]:
        step = rotate(out[i][2], [dx, dy])
        out.append([out[i][0] + step[0], out[i][1] + step[1], wrap(out[i][2] + dtheta)])
    return out

def main():
    truth, edges, closures = build_graph()

    # 1. The front end finds exactly the true loop closure (pose 9 revisits pose 0).
    check(closures == [(0, 9)], "descriptor matching must find the single true loop closure")
    check(len(edges) == 10, "nine odometry edges plus the loop edge must be present")

    # 2. Independent oracle: numeric Jacobian (central differences) versus the analytic one.
    probe = [list(pose) for pose in truth]
    probe[3][2] += 0.1
    analytic = jacobian(probe, edges)
    worst = 0.0
    for column in range(len(analytic[0])):
        kept, step = probe[column // 3 + 1][column % 3], 1e-6
        probe[column // 3 + 1][column % 3] = kept + step
        plus = residuals(probe, edges)
        probe[column // 3 + 1][column % 3] = kept - step
        minus = residuals(probe, edges)
        probe[column // 3 + 1][column % 3] = kept
        worst = max(worst, max(abs((p - m) / (2 * step) - analytic[row][column])
                               for row, (p, m) in enumerate(zip(plus, minus))))
    check(worst < 1e-5, "analytic Jacobian must match central finite differences")

    # 3. Optimisation: the objective decreases every iteration, the gauge pose stays fixed.
    fixed_pose = list(truth[0])
    optimised, trace = gauss_newton(truth, edges)
    check(all(trace[i + 1] <= trace[i] + 1e-12 for i in range(len(trace) - 1)) and trace[-1] < trace[0],
          "Gauss-Newton must decrease the pose graph objective")
    check(optimised[0] == fixed_pose, "pose 0 is the gauge and must not move")

    # 4. Drift correction: the optimised trajectory is far closer to the ground truth.
    drifted = odometry_chain(truth, edges)
    before, after = ate(drifted, GROUND_TRUTH), ate(optimised, GROUND_TRUTH)
    drift_end = math.dist(drifted[-1][:2], GROUND_TRUTH[-1][:2])
    optimised_end = math.dist(optimised[-1][:2], GROUND_TRUTH[-1][:2])
    check(before > 0.08, "odometry-only integration must drift noticeably")
    check(after < 0.8 * before, "pose graph optimisation must reduce the trajectory error")
    check(optimised_end < 0.4 * drift_end,
          "the loop closure must pull the final pose back onto the true loop")
    check(max(abs(r) for r in residuals(optimised, edges)) < 0.05,
          "the optimised graph must be nearly consistent with its measurements")

    # 5. Independent closed form: a three-pose unit-weight translation chain with an inconsistent
    #    loop edge; minimising (a-1)^2 + (b-a-1)^2 + (b-2.4)^2 gives a = 3.4/3, b = 6.8/3.
    chain = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.4, 0.0, 0.0]]
    chain_edges = [(0, 1, (1.0, 0.0, 0.0), 1.0), (1, 2, (1.0, 0.0, 0.0), 1.0),
                   (0, 2, (2.4, 0.0, 0.0), 1.0)]
    solved, _ = gauss_newton(chain, chain_edges)
    check(abs(solved[1][0] - 3.4 / 3) < 1e-6 and abs(solved[2][0] - 6.8 / 3) < 1e-6,
          "the chain optimum must match the hand-derived least-squares solution")

    # 6. Boundary cases: a single pose and an edge-free graph both stay put with a zero objective.
    lone, lone_trace = gauss_newton([[1.0, 2.0, 0.5]], [])
    check(lone == [[1.0, 2.0, 0.5]] and lone_trace == [0.0], "a one-pose graph is already optimal")
    free, free_trace = gauss_newton(truth, [])
    check(free == [list(pose) for pose in truth] and free_trace == [0.0],
          "a graph without edges must not be modified")

    # 7. Determinism: rebuilding and re-solving reproduces the same numbers.
    again_truth, again_edges, _ = build_graph()
    again, _ = gauss_newton(again_truth, again_edges)
    check(again == optimised and again_edges == edges, "the pipeline must be deterministic")

    print(f"loop closures {closures}, edges {len(edges)}, objective {trace[0]:.4f} -> {trace[-1]:.6f} "
          f"in {len(trace) - 1} Gauss-Newton steps")
    print(f"ATE odometry {before:.4f} m -> optimised {after:.4f} m, Jacobian error {worst:.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
