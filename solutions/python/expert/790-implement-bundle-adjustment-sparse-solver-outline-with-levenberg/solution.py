"""Task 790 - Implement bundle adjustment sparse solver outline with Levenberg-Marquardt.  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)

Bundle adjustment is the joint nonlinear least-squares refinement of camera poses and 3D
structure from image observations.  With a pinhole camera (focal f, principal point
(cu, cv)) and camera i described by a world-frame rotation R_i and centre C_i,

    q_i = R_i^T (X_j - C_i)          u = f q_x / q_z + cu,   v = f q_y / q_z + cv

and the residual of the observation of point j in camera i is pred_ij - obs_ij.  The damping
step solves

    (J^T J + lambda * diag(J^T J)) delta = -J^T r

with J the 2N x M Jacobian of the residual vector, assembled densely and factored by
Gaussian elimination with partial pivoting.  LM shrinks lambda by 3 when the step lowers the
cost and multiplies it by 10 when it does not, so the accepted cost sequence is strictly
decreasing by construction - asserted at every iteration, together with the residual of the
linear system that produced the step.

Each iteration re-linearises around the current estimate: the rotation increment enters as
R_i <- exp([delta_omega_i]_x) R_i, so evaluated at delta = 0 the orientation block of the
Jacobian is the exact first-order expression dq/d(delta_omega_k) = -R_i^T (e_k x d) with
d = X_j - C_i, and the point block is du/dX = (f/q_z)(row_x(R^T) - (q_x/q_z) row_z(R^T)) with
du/dC = -du/dX.  All three blocks are cross-checked against central differences of the
projection itself, so a sign or index error cannot pass unnoticed.

Outline-level simplifications (documented, not faked): the normal equations are dense - no
Schur complement, no sparse Cholesky, no variable ordering - there is no robust loss or
outlier rejection, no distortion model, camera 0 is fixed at the origin with identity
rotation (removing the 6 rigid degrees of freedom) and the 1-parameter scale gauge of a
monocular system is left in place, so convergence to the truth is verified *after*
normalising that scale.  Verified: analytic vs finite-difference Jacobian, LM cost
monotonicity and the damping rule, the LM linear-system residual, convergence to the true
structure up to scale from a perturbed start with sub-pixel noise, and the noiseless limit
where the reprojection error reaches machine precision.
"""
import math

FOCAL = 800.0
PRINCIPAL = (320.0, 240.0)
IDENTITY3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, r = splitmix64(state)
    return state, r / 18446744073709551616.0


# ------------------------------------------------------------------- rotations
def matrix_from_vector(vector):
    """Rodrigues: exp([vector]_x) as a 3x3 tuple."""
    theta = math.sqrt(sum(v * v for v in vector))
    if theta < 1e-12:
        k = [[0.0] * 3 for _ in range(3)]
    else:
        k = [[0.0, -vector[2] / theta, vector[1] / theta],
             [vector[2] / theta, 0.0, -vector[0] / theta],
             [-vector[1] / theta, vector[0] / theta, 0.0]]
    s, c = math.sin(theta), math.cos(theta)
    k2 = [[sum(k[i][m] * k[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    return tuple(tuple((1.0 if i == j else 0.0) + s * k[i][j] + (1.0 - c) * k2[i][j]
                       for j in range(3)) for i in range(3))


def matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def transpose(a):
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


# ------------------------------------------------------------------ projection
def project(rotation, center, point):
    """Pinhole projection of a world point; returns (u, v) and the camera-frame point."""
    d = [point[i] - center[i] for i in range(3)]
    q = [sum(rotation[k][i] * d[k] for k in range(3)) for i in range(3)]     # R^T d
    if q[2] <= 1e-6:
        raise ValueError("point behind the camera")
    return (FOCAL * q[0] / q[2] + PRINCIPAL[0], FOCAL * q[1] / q[2] + PRINCIPAL[1]), q


def projection_derivatives(rotation, center, point):
    """Analytic du/dX, dv/dX, du/dC, dv/dC, du/domega, dv/domega for the projection.

    The omega derivatives are the first-order (increment) ones, valid at delta_omega = 0.
    """
    (_u, _v), q = project(rotation, center, point)
    rt = transpose(rotation)
    row_x, row_y, row_z = rt[0], rt[1], rt[2]
    scale = FOCAL / q[2]
    grad_u_q = (scale, 0.0, -scale * q[0] / q[2])
    grad_v_q = (0.0, scale, -scale * q[1] / q[2])
    d = [point[i] - center[i] for i in range(3)]
    du_dx = [scale * (row_x[i] - q[0] / q[2] * row_z[i]) for i in range(3)]
    dv_dx = [scale * (row_y[i] - q[1] / q[2] * row_z[i]) for i in range(3)]
    du_dc = [-value for value in du_dx]
    dv_dc = [-value for value in dv_dx]
    du_dw = [0.0, 0.0, 0.0]
    dv_dw = [0.0, 0.0, 0.0]
    for k in range(3):
        if k == 0:
            cross = [0.0, -d[2], d[1]]                 # e_x x d
        elif k == 1:
            cross = [d[2], 0.0, -d[0]]                 # e_y x d
        else:
            cross = [-d[1], d[0], 0.0]                 # e_z x d
        dq = [-sum(rt[i][m] * cross[m] for m in range(3)) for i in range(3)]   # -R^T (e_k x d)
        du_dw[k] = sum(grad_u_q[i] * dq[i] for i in range(3))
        dv_dw[k] = sum(grad_v_q[i] * dq[i] for i in range(3))
    return du_dx, dv_dx, du_dc, dv_dc, du_dw, dv_dw


# --------------------------------------------------------------- bundle problem
class BundleProblem:
    """Cameras (camera 0 fixed) plus 3D points plus 2D observations of point/camera pairs."""

    def __init__(self, cameras, points, observations):
        self.rotations = [tuple(tuple(row) for row in camera[0]) for camera in cameras]
        self.centers = [list(camera[1]) for camera in cameras]
        self.points = [list(p) for p in points]
        self.observations = dict(observations)

    def free_cameras(self):
        return range(1, len(self.rotations))

    # ------------------------------------------------------------------- state
    def parameter_offsets(self):
        """Column offset of each parameter block plus the total parameter count."""
        offsets = {}
        position = 0
        for index in self.free_cameras():
            offsets[("omega", index)] = position
            position += 3
            offsets[("center", index)] = position
            position += 3
        for j in range(len(self.points)):
            offsets[("point", j)] = position
            position += 3
        return offsets, position

    def snapshot(self):
        return (list(self.rotations), [list(c) for c in self.centers],
                [list(p) for p in self.points])

    def restore(self, state):
        self.rotations = list(state[0])
        self.centers = [list(c) for c in state[1]]
        self.points = [list(p) for p in state[2]]

    def apply_delta(self, delta):
        """Apply an increment: R <- exp([delta_omega]_x) R, C <- C + delta_C, X <- X + delta_X."""
        offsets, _columns = self.parameter_offsets()
        for index in list(self.free_cameras()):
            base_w = offsets[("omega", index)]
            increment = matrix_from_vector(delta[base_w:base_w + 3])
            self.rotations[index] = matmul(increment, self.rotations[index])
            base_c = offsets[("center", index)]
            for k in range(3):
                self.centers[index][k] += delta[base_c + k]
        for j in range(len(self.points)):
            base_p = offsets[("point", j)]
            for k in range(3):
                self.points[j][k] += delta[base_p + k]

    # -------------------------------------------------------------- evaluation
    def residuals(self):
        out = []
        for (i, j), (u_obs, v_obs) in sorted(self.observations.items()):
            (u, v), _q = project(self.rotations[i], self.centers[i], self.points[j])
            out.append(u - u_obs)
            out.append(v - v_obs)
        return out

    def cost(self):
        return math.fsum(r * r for r in self.residuals())

    def analytic_jacobian(self):
        """Rows in observation order, columns in parameter-block order."""
        offsets, columns = self.parameter_offsets()
        free = set(self.free_cameras())
        rows = []
        for (i, j) in sorted(self.observations.keys()):
            row_u = [0.0] * columns
            row_v = [0.0] * columns
            du_dx, dv_dx, du_dc, dv_dc, du_dw, dv_dw = projection_derivatives(
                self.rotations[i], self.centers[i], self.points[j])
            if i in free:
                base_w = offsets[("omega", i)]
                base_c = offsets[("center", i)]
                for k in range(3):
                    row_u[base_w + k] = du_dw[k]
                    row_v[base_w + k] = dv_dw[k]
                    row_u[base_c + k] = du_dc[k]
                    row_v[base_c + k] = dv_dc[k]
            base_p = offsets[("point", j)]
            for k in range(3):
                row_u[base_p + k] = du_dx[k]
                row_v[base_p + k] = dv_dx[k]
            rows.append(row_u)
            rows.append(row_v)
        return rows

    def numeric_jacobian(self, step=1e-6):
        """Central-difference Jacobian of the increment map; validates the analytic one."""
        _offsets, columns = self.parameter_offsets()
        base_state = self.snapshot()
        rows = [[0.0] * columns for _ in range(2 * len(self.observations))]
        for column in range(columns):
            plus = [0.0] * columns
            plus[column] = step
            self.apply_delta(plus)
            values_plus = self.residuals()
            self.restore(base_state)
            self.apply_delta([-value for value in plus])
            values_minus = self.residuals()
            self.restore(base_state)
            for row in range(len(rows)):
                rows[row][column] = (values_plus[row] - values_minus[row]) / (2.0 * step)
        return rows


def normal_equations(jacobian, residuals, damping):
    """Return (A, g) with A = J^T J + lambda diag(J^T J) and g = J^T r."""
    rows = len(jacobian)
    columns = len(jacobian[0])
    a = [[0.0] * columns for _ in range(columns)]
    g = [0.0] * columns
    for row in range(rows):
        values = jacobian[row]
        for i in range(columns):
            if values[i] == 0.0:
                continue
            g[i] += values[i] * residuals[row]
            for k in range(i, columns):
                a[i][k] += values[i] * values[k]
    for i in range(columns):
        for k in range(i):
            a[i][k] = a[k][i]
        a[i][i] = a[i][i] * (1.0 + damping) + 1e-12
    return a, g


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting (dense; see the header for the caveat)."""
    n = len(rhs)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular normal equations")
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inv
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))
        x[row] = acc / a[row][row]
    return x


def levenberg_marquardt(problem, iterations=60, damping=1e-3, tolerance=1e-14):
    """Damped Gauss-Newton; returns a trace of (cost, damping, accepted, step residual)."""
    damping = float(damping)
    cost = problem.cost()
    trace = []
    for _ in range(iterations):
        residuals = problem.residuals()
        jacobian = problem.analytic_jacobian()
        a, g = normal_equations(jacobian, residuals, damping)
        delta = solve_linear(a, [-value for value in g])
        step_residual = max(abs(sum(a[i][k] * delta[k] for k in range(len(delta))) + g[i])
                            for i in range(len(delta)))
        problem.apply_delta(delta)
        new_cost = problem.cost()
        if new_cost < cost:
            accepted = True
            improvement = cost - new_cost
            cost = new_cost
            damping = max(damping / 3.0, 1e-12)
        else:
            accepted = False
            improvement = 0.0
            problem.apply_delta([-value for value in delta])
            damping = min(damping * 10.0, 1e12)
        trace.append((cost, damping, accepted, step_residual, improvement))
        if improvement <= tolerance * max(1.0, cost):
            break
    return trace


# --------------------------------------------------------------- scene builder
CAMERA_CENTERS = [(0.0, 0.0, 0.0), (1.2, 0.15, 0.05), (0.85, 1.05, -0.08), (1.05, -1.0, 0.06)]
POINT_COUNT = 8


def build_scene(seed, noise=0.0, rotation_perturbation=0.0, point_perturbation=0.0,
                center_perturbation=0.05):
    """Four cameras looking down +z at eight points; returns truth, initial guess and layout."""
    state = seed
    true_cameras = []
    for index in range(len(CAMERA_CENTERS)):
        angles = []
        for _ in range(3):
            state, r = uniform(state)
            angles.append((r - 0.5) * 0.12)
        true_cameras.append((matrix_from_vector(angles), list(CAMERA_CENTERS[index])))
    truth_points = []
    for _ in range(POINT_COUNT):
        coords = []
        for _c in range(2):
            state, r = uniform(state)
            coords.append((r - 0.5) * 1.6)
        state, r = uniform(state)
        truth_points.append([coords[0], coords[1], 3.0 + r * 1.5])
    observations = {}
    for i in range(len(true_cameras)):
        rotation, center = true_cameras[i]
        for j in range(POINT_COUNT):
            (u, v), _q = project(rotation, center, truth_points[j])
            state, r1 = uniform(state)
            state, r2 = uniform(state)
            observations[(i, j)] = (u + (r1 - 0.5) * 2.0 * noise,
                                    v + (r2 - 0.5) * 2.0 * noise)
    state = seed ^ 0x5EED
    initial_cameras = [true_cameras[0]]
    for i in range(1, len(true_cameras)):
        rotation, center = true_cameras[i]
        perturb = []
        for _k in range(3):
            state, r = uniform(state)
            perturb.append((r - 0.5) * 2.0 * rotation_perturbation)
        moved = []
        for k in range(3):
            state, r = uniform(state)
            moved.append(center[k] + (r - 0.5) * 2.0 * center_perturbation)
        initial_cameras.append((matmul(matrix_from_vector(perturb), rotation), moved))
    initial_points = []
    for point in truth_points:
        shifted = []
        for k in range(3):
            state, r = uniform(state)
            shifted.append(point[k] + (r - 0.5) * 2.0 * point_perturbation)
        initial_points.append(shifted)
    truth = BundleProblem(true_cameras, truth_points, observations)
    initial = BundleProblem(initial_cameras, initial_points, observations)
    return truth, initial, true_cameras, truth_points


def normalised_comparison(problem, true_cameras, truth_points):
    """Compare structure after dividing out the monocular scale gauge (camera 0 at origin)."""
    true_baseline = math.sqrt(sum(c * c for c in true_cameras[1][1]))
    estimate_baseline = math.sqrt(sum(c * c for c in problem.centers[1]))
    scale = estimate_baseline / true_baseline
    center_error = max(abs(problem.centers[1][k] / scale - true_cameras[1][1][k])
                       for k in range(3))
    point_error = max(abs(problem.points[j][k] / scale - truth_points[j][k])
                      for j in range(len(truth_points)) for k in range(3))
    return scale, center_error, point_error


def main():
    # --- 1. the analytic Jacobian must match central differences --------------
    _truth, initial, true_cameras, truth_points = build_scene(
        790001, noise=0.3, rotation_perturbation=0.03, point_perturbation=0.05)
    analytic = initial.analytic_jacobian()
    numeric = initial.numeric_jacobian()
    worst = 0.0
    scale_ref = 0.0
    for row_a, row_n in zip(analytic, numeric):
        for value_a, value_n in zip(row_a, row_n):
            worst = max(worst, abs(value_a - value_n))
            scale_ref = max(scale_ref, abs(value_n))
    check(worst < 1e-4 * max(1.0, scale_ref),
          "analytic Jacobian must match finite differences, worst error %.3e" % worst)
    check(len(analytic) == 2 * len(initial.observations),
          "the Jacobian must have two rows per observation")
    check(len(analytic[0]) == 6 * (len(CAMERA_CENTERS) - 1) + 3 * POINT_COUNT,
          "the Jacobian must have one column per parameter")

    # --- 2. LM drives the reprojection error down to the noise level ----------
    start_rms = math.sqrt(initial.cost() / len(initial.observations))
    trace = levenberg_marquardt(initial, iterations=60)
    final_rms = math.sqrt(initial.cost() / len(initial.observations))
    check(final_rms < 0.6,
          "LM must reach sub-pixel RMS reprojection error, got %.4f px" % final_rms)
    check(final_rms < start_rms / 5.0,
          "LM must cut the RMS error by at least 5x: %.4f -> %.4f" % (start_rms, final_rms))
    accepted_costs = [cost for cost, _d, accepted, _s, _i in trace if accepted]
    check(all(accepted_costs[i] > accepted_costs[i + 1] for i in range(len(accepted_costs) - 1)),
          "the cost must strictly decrease on every accepted LM step")
    check(all(step < 1e-6 for _c, _d, _a, step, _i in trace),
          "each LM step must solve its damped normal equations accurately")
    damping_rule_ok = True
    for index, current in enumerate(trace):
        previous_damping = 1e-3 if index == 0 else trace[index - 1][1]
        expected = (max(previous_damping / 3.0, 1e-12) if current[2]
                    else min(previous_damping * 10.0, 1e12))
        damping_rule_ok &= abs(current[1] - expected) < 1e-18
    check(damping_rule_ok, "the LM controller must shrink damping by 3 and grow it by 10")
    check(sum(1 for entry in trace if entry[2]) >= 3,
          "LM must accept at least three steps on this scene")

    # --- 3. convergence to the true structure up to the scale gauge ----------
    scale, center_error, point_error = normalised_comparison(
        initial, true_cameras, truth_points)
    check(center_error < 0.05,
          "the recovered baseline must match the truth within 0.05, got %.4f" % center_error)
    check(point_error < 0.05,
          "the recovered structure must match the truth within 0.05, got %.4f" % point_error)
    check(0.98 < scale < 1.02,
          "the scale gauge must stay near one for a well-conditioned scene, got %.6f" % scale)

    # --- 4. zero-noise limit: machine-precision reprojection -----------------
    _truth2, exact_init, true_cameras2, truth_points2 = build_scene(
        790002, noise=0.0, rotation_perturbation=0.02, point_perturbation=0.04)
    exact_start_rms = math.sqrt(exact_init.cost() / len(exact_init.observations))
    levenberg_marquardt(exact_init, iterations=80)
    exact_final_rms = math.sqrt(exact_init.cost() / len(exact_init.observations))
    check(exact_final_rms < 1e-6,
          "a noiseless scene must be fitted to machine precision, RMS %.3e px"
          % exact_final_rms)
    check(exact_final_rms < exact_start_rms / 1000.0,
          "the noiseless fit must improve by orders of magnitude: %.3e -> %.3e"
          % (exact_start_rms, exact_final_rms))
    _scale2, _center_error2, point_error2 = normalised_comparison(
        exact_init, true_cameras2, truth_points2)
    check(point_error2 < 1e-5,
          "the noiseless structure must be recovered exactly up to scale, got %.3e"
          % point_error2)

    # --- 5. boundary behaviour and invalid projections ----------------------
    _t3, tiny, tiny_cameras, _tp3 = build_scene(790003, noise=0.1, rotation_perturbation=0.01,
                                                point_perturbation=0.02)
    single = BundleProblem(tiny_cameras, tiny.points, {(0, 0): tiny.observations[(0, 0)]})
    single_trace = levenberg_marquardt(single, iterations=5)
    check(len(single_trace) >= 1, "a single-observation problem must still take LM steps")
    behind = list(tiny.points[0])
    behind[2] = -1.0
    raised = False
    try:
        project(IDENTITY3, [0.0, 0.0, 0.0], behind)
    except ValueError:
        raised = True
    check(raised, "a point behind the camera must be rejected")
    zero_cost = BundleProblem(tiny_cameras, tiny.points, {})
    check(zero_cost.cost() == 0.0, "an observation-free problem must have zero cost")
    check(tiny.parameter_offsets()[1] == 6 * (len(CAMERA_CENTERS) - 1) + 3 * POINT_COUNT,
          "the parameter count must match the free camera and point blocks")

    print("Jacobian check: worst |analytic - numeric| = %.3e (column scale %.3e)"
          % (worst, scale_ref))
    print("LM: RMS %.4f -> %.4f px in %d iterations, %d accepted"
          % (start_rms, final_rms, len(trace), sum(1 for entry in trace if entry[2])))
    print("structure error after normalising scale %.6f: centre %.4f, points %.4f"
          % (scale, center_error, point_error))
    print("noiseless run: RMS %.3e -> %.3e px, structure error %.3e"
          % (exact_start_rms, exact_final_rms, point_error2))
    print("PASS")


if __name__ == "__main__":
    main()
