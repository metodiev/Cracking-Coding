"""Task 789 - Implement shape-matching registration (ICP) iterative closest point.  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)

Rigid registration of two point clouds alternates two exact subproblems:

  1. correspondence: for every (transformed) source point find the nearest target point -
     here a brute-force O(n_s * n_t) scan, optionally dropping the worst fraction of pairs
     (trimmed ICP) so that outliers cannot pull the transform away;
  2. transform: solve min_R,t sum_i |R p_i + t - q_i|^2 with R a proper rotation.  The
     translation is exact, t = q_bar - R p_bar, and the rotation comes from Horn's
     unit-quaternion method: the 4x4 symmetric matrix N built from the covariance
     S = sum_i (p_i - p_bar)(q_i - q_bar)^T has the property that the unit eigenvector
     (w, x, y, z) of its largest eigenvalue is the optimal rotation quaternion.  N is
     diagonalised here with cyclic Jacobi sweeps, so no linear-algebra package is needed.

Each iteration applies the increment to a cumulative transform.  The objective (mean squared
nearest-neighbour distance, or the trimmed version of it) is then non-increasing: the
transform step is optimal for the current correspondences and re-matching can only lower the
objective again - that monotonicity is asserted for every iteration.

Verified: the recovered transform reproduces a known rotation+translation to ~1e-9 with known
correspondences; it is orthonormal with determinant +1; on a planar cloud its residual is
compared against an independent dense search over the rotation angle refined by ternary
search; ICP decreases its objective at every step and converges to the known pose from a
perturbed start; trimmed ICP survives outliers in the target; forward-then-backward
registration composes to the identity; the degenerate (all-points-coincident) configuration
is rejected instead of returning garbage; and repeated runs are bit-identical.

Simplified versus production ICP: no k-d tree (brute-force matching, so clouds stay small),
no point-to-plane metric, no normal estimation, and outliers are handled by a fixed trim
fraction rather than a robust kernel with a scale estimate.
"""
import math

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


# ------------------------------------------------------------------ linear algebra
def jacobi_eigen(matrix, sweeps=60):
    """Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations."""
    n = len(matrix)
    a = [list(row) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < 1e-18:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], [[v[i][j] for i in range(n)] for j in range(n)]


def quaternion_to_matrix(q):
    """Rotation matrix from a (possibly unnormalised) quaternion (w, x, y, z)."""
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def apply_transform(rotation, translation, point):
    return [sum(rotation[i][j] * point[j] for j in range(3)) + translation[i] for i in range(3)]


def compose(rotation_a, translation_a, rotation_b, translation_b):
    """Transform equivalent to applying B first and then A."""
    rotation = tuple(tuple(sum(rotation_a[i][k] * rotation_b[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))
    translation = [sum(rotation_a[i][k] * translation_b[k] for k in range(3)) + translation_a[i]
                   for i in range(3)]
    return rotation, translation


def centroid(points):
    n = float(len(points))
    return [sum(p[i] for p in points) / n for i in range(3)]


def determinant(rotation):
    return (rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]))


def kabsch(points, targets):
    """Optimal proper rigid transform mapping `points` onto `targets` (Horn's quaternion).

    Returns (rotation, translation, residual) with residual the sum of squared errors.
    """
    if len(points) != len(targets) or len(points) < 2:
        raise ValueError("at least two matched pairs are required")
    pc = centroid(points)
    tc = centroid(targets)
    source_spread = math.fsum(sum((p[i] - pc[i]) ** 2 for i in range(3)) for p in points)
    if source_spread < 1e-18:
        raise ValueError("degenerate configuration: the source points coincide")
    s = [[0.0] * 3 for _ in range(3)]
    for p, q in zip(points, targets):
        dp = [p[i] - pc[i] for i in range(3)]
        dq = [q[i] - tc[i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                s[i][j] += dp[i] * dq[j]
    if math.fsum(abs(s[i][j]) for i in range(3) for j in range(3)) < 1e-15:
        raise ValueError("degenerate configuration: the target points coincide")
    sxx, sxy, sxz = s[0]
    syx, syy, syz = s[1]
    szx, szy, szz = s[2]
    n = ((sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
         (syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
         (szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
         (sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz))
    eigenvalues, eigenvectors = jacobi_eigen(n)
    best = max(range(4), key=lambda i: eigenvalues[i])
    rotation = quaternion_to_matrix(tuple(eigenvectors[best]))
    rotated_centroid = [sum(rotation[i][j] * pc[j] for j in range(3)) for i in range(3)]
    translation = [tc[i] - rotated_centroid[i] for i in range(3)]
    residual = 0.0
    for p, q in zip(points, targets):
        moved = apply_transform(rotation, translation, p)
        residual += sum((moved[i] - q[i]) ** 2 for i in range(3))
    return rotation, translation, residual


# -------------------------------------------------------------------------- ICP
def nearest_index(point, targets):
    best_index = 0
    best_distance = float("inf")
    for index, target in enumerate(targets):
        distance = ((point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2
                    + (point[2] - target[2]) ** 2)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index, best_distance


class ICP:
    """Point-to-point ICP with optional trimming of the worst correspondences."""

    def __init__(self, max_iterations=60, tolerance=1e-13, trim_fraction=1.0):
        if not 0.0 < trim_fraction <= 1.0:
            raise ValueError("trim_fraction must lie in (0, 1]")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.trim_fraction = trim_fraction

    def _pairs(self, source, target, rotation, translation):
        """(squared distance, source index, target index) for every source point."""
        pairs = []
        for index, p in enumerate(source):
            moved = apply_transform(rotation, translation, p)
            target_index, distance = nearest_index(moved, target)
            pairs.append((distance, index, target_index))
        pairs.sort()
        return pairs

    def objective(self, source, target, rotation, translation):
        """Mean squared nearest-neighbour distance over the kept fraction."""
        pairs = self._pairs(source, target, rotation, translation)
        keep = max(2, int(round(self.trim_fraction * len(source))))
        used = pairs[:keep]
        return math.fsum(d for d, _i, _k in used) / len(used)

    def align(self, source, target, rotation=None, translation=None):
        if len(source) < 3 or len(target) < 3:
            raise ValueError("registration needs at least three points per cloud")
        rotation = IDENTITY3 if rotation is None else rotation
        translation = [0.0, 0.0, 0.0] if translation is None else list(translation)
        keep = max(2, int(round(self.trim_fraction * len(source))))
        history = [self.objective(source, target, rotation, translation)]
        for _ in range(self.max_iterations):
            moved = [apply_transform(rotation, translation, p) for p in source]
            pairs = []
            for index, point in enumerate(moved):
                target_index, distance = nearest_index(point, target)
                pairs.append((distance, index, target_index))
            pairs.sort()
            used = pairs[:keep]
            # the increment maps the *currently transformed* points onto their neighbours
            src = [moved[i] for _d, i, _k in used]
            dst = [target[k] for _d, _i, k in used]
            step_rotation, step_translation, _residual = kabsch(src, dst)
            rotation, translation = compose(step_rotation, step_translation, rotation, translation)
            history.append(self.objective(source, target, rotation, translation))
            if history[-2] - history[-1] <= self.tolerance * max(1.0, history[-2]):
                break
        return {"rotation": rotation, "translation": translation, "history": history,
                "iterations": len(history) - 1,
                "objective": self.objective(source, target, rotation, translation)}


# ------------------------------------------------------------------ test helpers
def rotation_from_axes(angles):
    """Rotation built by composing the three elementary axis rotations."""
    rotation = IDENTITY3
    for axis, angle in zip(range(3), angles):
        c, s = math.cos(angle), math.sin(angle)
        if axis == 0:
            step = ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
        elif axis == 1:
            step = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
        else:
            step = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
        rotation = compose(step, [0.0, 0.0, 0.0], rotation, [0.0, 0.0, 0.0])[0]
    return rotation


def rotation_angle(rotation):
    """Rotation angle of a proper rotation, from its trace."""
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def make_cloud(count, seed, radius=1.0, planar=True):
    """Deterministic pseudo-random cloud in [-radius, radius]^2 (or ^3)."""
    state = seed
    points = []
    for _ in range(count):
        state, r1 = uniform(state)
        state, r2 = uniform(state)
        state, r3 = uniform(state)
        z = 0.0 if planar else (r3 - 0.5) * 2.0 * radius
        points.append([(r1 - 0.5) * 2.0 * radius, (r2 - 0.5) * 2.0 * radius, z])
    return points, state


def planar_rss(angle, points, targets):
    """Sum of squared errors for a planar rotation by `angle` with optimal translation.

    The optimal translation for a fixed angle is the centroid difference (in all three
    axes), so it is applied analytically and does not need to be searched.
    """
    pc = centroid(points)
    tc = centroid(targets)
    dz = tc[2] - pc[2]
    c, s = math.cos(angle), math.sin(angle)
    total = 0.0
    for p, q in zip(points, targets):
        x = c * (p[0] - pc[0]) - s * (p[1] - pc[1])
        y = s * (p[0] - pc[0]) + c * (p[1] - pc[1])
        total += (x + tc[0] - q[0]) ** 2 + (y + tc[1] - q[1]) ** 2 + (p[2] + dz - q[2]) ** 2
    return total


def dense_planar_search(points, targets, samples=7200):
    """Dense angular scan; returns (best angle, best value, angles examined)."""
    best_angle, best_value = 0.0, float("inf")
    for k in range(samples):
        angle = 2.0 * math.pi * (k + 0.5) / samples
        value = planar_rss(angle, points, targets)
        if value < best_value:
            best_angle, best_value = angle, value
    return best_angle, best_value, samples


def ternary_refine(points, targets, center, half_width, iterations=90):
    """Ternary search for the best rotation angle inside a bracket (unimodal there)."""
    lo, hi = center - half_width, center + half_width
    for _ in range(iterations):
        third = (hi - lo) / 3.0
        left, right = lo + third, hi - third
        if planar_rss(left, points, targets) < planar_rss(right, points, targets):
            hi = right
        else:
            lo = left
    angle = 0.5 * (lo + hi)
    return angle, planar_rss(angle, points, targets)


def main():
    # --- 1. Kabsch recovers a known rigid transform exactly --------------------
    cloud, _ = make_cloud(10, 789001)
    truth_rotation = rotation_from_axes([0.4, -0.7, 0.25])
    truth_translation = [0.7, -1.3, 0.25]
    moved = [apply_transform(truth_rotation, truth_translation, p) for p in cloud]
    rotation, translation, residual = kabsch(cloud, moved)
    rotation_error = max(abs(rotation[i][j] - truth_rotation[i][j])
                         for i in range(3) for j in range(3))
    check(rotation_error < 1e-9,
          "Kabsch must recover the rotation to 1e-9, max element error %.3e" % rotation_error)
    check(max(abs(translation[i] - truth_translation[i]) for i in range(3)) < 1e-9,
          "Kabsch must recover the translation to 1e-9")
    check(residual < 1e-18,
          "a noiseless correspondence must leave a residual of ~0, got %.3e" % residual)

    # --- 2. the rotation is orthonormal with determinant +1 -------------------
    gram = tuple(tuple(sum(rotation[i][k] * rotation[j][k] for k in range(3)) for j in range(3))
                 for i in range(3))
    check(max(abs(gram[i][j] - IDENTITY3[i][j]) for i in range(3) for j in range(3)) < 1e-12,
          "R^T R must be the identity")
    determinant_value = determinant(rotation)
    check(abs(determinant_value - 1.0) < 1e-12,
          "the rotation must be proper (det = +1), got %.12f" % determinant_value)

    # --- 3. independent oracle: dense angular search + ternary refinement -----
    planar, _ = make_cloud(9, 789002, planar=True)
    planar_truth = rotation_from_axes([0.0, 0.0, 0.83])
    planar_target = [apply_transform(planar_truth, [1.1, -0.6, 0.35], p) for p in planar]
    _r, _t, planar_residual = kabsch(planar, planar_target)
    grid_angle, grid_value, samples = dense_planar_search(planar, planar_target)
    refined_angle, refined_value = ternary_refine(
        planar, planar_target, grid_angle, 2.0 * math.pi / samples + 1e-6)
    check(planar_residual <= refined_value + 1e-12,
          "Kabsch residual %.12f must not exceed the refined search optimum %.12f"
          % (planar_residual, refined_value))
    check(refined_value - planar_residual < 1e-9,
          "on an exact planar motion the analytic and search optima must coincide, gap %.3e"
          % (refined_value - planar_residual))
    noisy_planar = [[p[0] + (0.002 if i % 2 else -0.0015), p[1] + (0.001 if i % 3 else -0.002),
                     0.0] for i, p in enumerate(planar_target)]
    _rn, _tn, noisy_residual = kabsch(planar, noisy_planar)
    noisy_grid_angle, noisy_grid_value, _ = dense_planar_search(planar, noisy_planar)
    _noisy_angle, noisy_refined = ternary_refine(
        planar, noisy_planar, noisy_grid_angle, 2.0 * math.pi / samples + 1e-6)
    check(noisy_residual <= noisy_refined + 1e-12,
          "the 3D optimum %.12f must not be beaten by the planar search %.12f"
          % (noisy_residual, noisy_refined))
    check(abs(rotation_angle(kabsch(planar, planar_target)[0])
              - rotation_angle(planar_truth)) < 1e-9,
          "the recovered planar-motion angle must equal the applied angle")

    # --- 4. identical clouds: identity transform, zero objective --------------
    result = ICP().align(cloud, cloud)
    check(max(abs(result["translation"][i]) for i in range(3)) < 1e-9,
          "registering a cloud onto itself must leave the translation at zero")
    check(result["objective"] < 1e-18,
          "registering a cloud onto itself must reach zero objective, got %.3e"
          % result["objective"])

    # --- 5. ICP converges to the known pose from a perturbed start -----------
    reg_cloud, _ = make_cloud(20, 789100)
    reg_rotation = rotation_from_axes([0.15, -0.3, 0.2])
    reg_translation = [0.25, -0.22, 0.1]
    reg_target = []
    for index, p in enumerate(apply_transform(reg_rotation, reg_translation, q)
                              for q in reg_cloud):
        reg_target.append([p[0] + (0.002 if index % 2 else -0.0025),
                           p[1] + (0.0015 if index % 3 else -0.001), p[2]])
    start_rotation = rotation_from_axes([0.05, -0.1, 0.4])
    start_translation = [0.15, -0.12, 0.05]
    result = ICP(max_iterations=80, tolerance=1e-13).align(
        reg_cloud, reg_target, start_rotation, start_translation)
    pose_error = max(abs(result["rotation"][i][j] - reg_rotation[i][j])
                     for i in range(3) for j in range(3))
    check(pose_error < 0.02,
          "ICP must recover the pose to 0.02, max element error %.4f" % pose_error)
    check(abs(rotation_angle(result["rotation"]) - rotation_angle(reg_rotation)) < 0.02,
          "the recovered rotation angle must match the truth within 0.02 rad")
    check(result["objective"] < 5e-5,
          "ICP must reach a small objective on lightly noisy data, got %.3e" % result["objective"])
    check(all(result["history"][i] >= result["history"][i + 1] - 1e-18
              for i in range(len(result["history"]) - 1)),
          "the ICP objective must decrease monotonically: %r" % (result["history"],))
    check(result["history"][0] > 10.0 * result["history"][-1],
          "the objective must really improve from the perturbed start: %.3e -> %.3e"
          % (result["history"][0], result["history"][-1]))
    check(2 <= result["iterations"] <= 80, "ICP must terminate within the iteration budget")

    # --- 6. trimmed ICP survives outliers in the target ----------------------
    outlier_state = 789123
    outliers = []
    for _ in range(7):
        outlier_state, r1 = uniform(outlier_state)
        outlier_state, r2 = uniform(outlier_state)
        outliers.append([(r1 - 0.5) * 2.0, (r2 - 0.5) * 2.0, 0.0])
    contaminated = reg_target[:13] + outliers
    plain = ICP(max_iterations=80, tolerance=1e-13).align(
        reg_cloud, contaminated, start_rotation, start_translation)
    plain_error = max(abs(plain["rotation"][i][j] - reg_rotation[i][j])
                      for i in range(3) for j in range(3))
    trimmed = ICP(max_iterations=80, tolerance=1e-13, trim_fraction=0.6)
    trimmed_result = trimmed.align(reg_cloud, contaminated, start_rotation, start_translation)
    trimmed_error = max(abs(trimmed_result["rotation"][i][j] - reg_rotation[i][j])
                        for i in range(3) for j in range(3))
    check(trimmed_error < 0.02,
          "trimmed ICP must stay close to the truth under 35%% outliers, error %.4f"
          % trimmed_error)
    check(trimmed_error < plain_error,
          "trimming must beat the untrimmed fit on contaminated data: %.4f vs %.4f"
          % (trimmed_error, plain_error))
    check(all(trimmed_result["history"][i] >= trimmed_result["history"][i + 1] - 1e-18
              for i in range(len(trimmed_result["history"]) - 1)),
          "the trimmed objective must also decrease monotonically")
    check(trimmed_result["objective"] < 1e-5,
          "the trimmed inlier objective must go to zero, got %.3e" % trimmed_result["objective"])

    # --- 7. inverse consistency: forward then backward is the identity -------
    forward = ICP(max_iterations=80, tolerance=1e-13).align(
        reg_cloud, reg_target, start_rotation, start_translation)
    backward = ICP(max_iterations=80, tolerance=1e-13).align(
        reg_target, reg_cloud, forward["rotation"], forward["translation"])
    composed = compose(backward["rotation"], backward["translation"],
                       forward["rotation"], forward["translation"])
    identity_error = max(abs(composed[0][i][j] - IDENTITY3[i][j])
                         for i in range(3) for j in range(3))
    check(identity_error < 1e-6,
          "forward then backward registration must compose to the identity, error %.3e"
          % identity_error)
    check(max(abs(composed[1][i]) for i in range(3)) < 1e-6,
          "forward then backward registration must cancel the translation too")

    # --- 8. degeneracy, invalid input, reproducibility -----------------------
    raised = False
    try:
        kabsch([[0.3, 0.4, 0.0]] * 5, cloud[:5])
    except ValueError:
        raised = True
    check(raised, "a zero-spread point set must be rejected as degenerate")
    raised = False
    try:
        ICP().align([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cloud[:2])
    except ValueError:
        raised = True
    check(raised, "fewer than three points per cloud must be rejected")
    raised = False
    try:
        ICP(trim_fraction=1.5)
    except ValueError:
        raised = True
    check(raised, "an out-of-range trim fraction must be rejected")
    repeat = ICP(max_iterations=80, tolerance=1e-13).align(
        reg_cloud, reg_target, start_rotation, start_translation)
    check(repeat["rotation"] == result["rotation"]
          and repeat["iterations"] == result["iterations"],
          "ICP must be deterministic for identical input")

    print("Kabsch: rotation error %.3e, det %.12f, residual %.3e"
          % (rotation_error, determinant_value, residual))
    print("planar: analytic residual %.3e vs refined search %.3e (grid %.3e over %d angles)"
          % (planar_residual, refined_value, grid_value, samples))
    print("planar with noise: analytic %.9f vs planar-search %.9f"
          % (noisy_residual, noisy_refined))
    print("ICP: iterations=%d objective=%.3e pose error=%.3e (start %.3e)"
          % (result["iterations"], result["objective"], pose_error, result["history"][0]))
    print("outlier target: untrimmed pose error %.4f, trimmed %.4f (iterations %d, objective %.3e)"
          % (plain_error, trimmed_error, trimmed_result["iterations"],
             trimmed_result["objective"]))
    print("forward/backward composition error: %.3e" % identity_error)
    print("PASS")


if __name__ == "__main__":
    main()
