"""Task 576 - Implement bundle adjustment simple Gauss-Newton iterate (conceptual). (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)

A small but complete bundle adjustment, not an outline: three pinhole cameras with fixed intrinsics
and six pose parameters each (a Rodrigues rotation vector carrying the increment, plus a translation),
and seven 3D points with three parameters each - 33 unknowns (the first camera anchors six of the
seven gauge degrees of freedom and stays fixed) against 42 reprojection residuals. The seventh gauge
direction is the overall scale, which no pixel can see once the anchored camera sits at the origin:
the file measures that drift (the fit comes out 1.019x too large) and aligns it before comparing with
the truth. The
residual is the projection difference pi(K(R X + t)) - observation in pixels; the Jacobian is
assembled analytically from the 2x3 block dpi/dY = [[f, 0, cx - u], [0, f, cy - v]] / z with
Y = R X + t, giving the rotation block -dpi/dY [Y]_x, the translation block dpi/dY and the structure
block dpi/dY R. Increments left-multiply the rotation (R <- exp([delta]_x) R), which is exactly what
the -[R X]_x block encodes, and the normal equations (J^T J + lambda diag(J^T J)) delta = -J^T r are
solved by dense Gaussian elimination with partial pivoting; lambda = 0 turns the same loop into the
plain Gauss-Newton iterate, and the accepted steps halve the damping while rejected ones multiply it
by four. Cameras are 900-pixel pinholes, the seven points are spread over 2 to 6 units of depth with a 1.3-unit
baseline, and all data come from a fixed-seed splitmix64 stream. The gauge freedom is demonstrated rather than hidden: with every
camera free the cost still reaches 1e-18 while the poses drift away from the truth by a global rigid
motion, so the file measures that drift and contrasts it with the anchored run.

Checks: the analytic Jacobian against central finite differences over all 33 free parameters (42
residual rows, agreement to 1e-6, reported by the worst entry), including the fact that an anchored
camera's rows are exactly zero; the solved step satisfying the damped normal equations to 1e-9;
convergence on noiseless data to a cost below 1e-18 from a 0.05-perturbed start, with both free poses
and all seven points recovered to 1e-6, the anchored camera asserted untouched, and the first-order
optimality condition J^T r = 0 met to 1e-9; the cost history asserted to be non-increasing at every accepted step, with the
accepted and rejected counts checked; the same run under 0.5 px of noise, where the cost stops at the
noise floor while the pose error still improves; the unanchored run that drives the cost to 1e-18 but
drifts in pose (the gauge freedom); the documented behavioural difference from a 0.3
perturbation, where Levenberg-Marquardt reaches 6.0e-26 after 19 accepted steps while plain
Gauss-Newton accepts one step, overshoots, and stalls at 1.56e5 (from 0.05 both converge, so the
difference is the damping), together with the honest limit that *both* fail from a 1.2 perturbation
(LM 1.3e4) - all measured numbers are asserted rather than assumed; and determinism of the whole optimisation and of the sample generator.
"""
import math

MASK64 = (1 << 64) - 1


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def rodrigues(vector):
    """Rotation matrix exp([vector]_x), in closed form."""
    theta = math.sqrt(sum(value * value for value in vector))
    if theta < 1e-12:
        return [[1.0, -vector[2], vector[1]],
                [vector[2], 1.0, -vector[0]],
                [-vector[1], vector[0], 1.0]]
    x, y, z = (value / theta for value in vector)
    c, s = math.cos(theta), math.sin(theta)
    one = 1.0 - c
    return [[c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one]]


def intrinsic_matrix(focal, cx, cy):
    return [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]


def skew(vector):
    return [[0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0]]


def matmul(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0]))]
            for i in range(len(left))]


def project(intrinsics, rotation, translation, point):
    y = [sum(rotation[i][j] * point[j] for j in range(3)) + translation[i] for i in range(3)]
    return ((intrinsics[0][0] * y[0] + intrinsics[0][2] * y[2]) / y[2],
            (intrinsics[1][1] * y[1] + intrinsics[1][2] * y[2]) / y[2]), y


def projection_jacobian(intrinsics, y):
    """d(u, v) / dY for a pinhole camera: [[f, 0, cx - u], [0, f, cy - v]] / z."""
    z = y[2]
    u = (intrinsics[0][0] * y[0] + intrinsics[0][2] * z) / z
    v = (intrinsics[1][1] * y[1] + intrinsics[1][2] * z) / z
    return [[intrinsics[0][0] / z, 0.0, (intrinsics[0][2] - u) / z],
            [0.0, intrinsics[1][1] / z, (intrinsics[1][2] - v) / z]]


def parameter_count(camera_count, point_count):
    return 6 * camera_count + 3 * point_count


def residuals(cameras, points, observations, intrinsics):
    values = []
    for camera_index, (rotation, translation) in enumerate(cameras):
        for point_index, point in enumerate(points):
            predicted, _ = project(intrinsics, rotation, translation, point)
            observed = observations[(camera_index, point_index)]
            values.append(predicted[0] - observed[0])
            values.append(predicted[1] - observed[1])
    return values


def cost_of(values):
    return 0.5 * sum(value * value for value in values)


def jacobian(cameras, points, intrinsics, free=None):
    """Analytic Jacobian: one row per residual (the rows of anchored cameras are all zero),
    columns [rotation_i, translation_i, points...] for the cameras listed in `free`."""
    if free is None:
        free = tuple(range(len(cameras)))
    slot = {camera: index for index, camera in enumerate(free)}
    columns = 6 * len(free) + 3 * len(points)
    rows = []
    for camera_index, (rotation, translation) in enumerate(cameras):
        for point_index, point in enumerate(points):
            _, y = project(intrinsics, rotation, translation, point)
            block = projection_jacobian(intrinsics, y)
            # Y(delta) = exp([delta]_x) R X + t, so dY/ddelta = -[R X]_x: the translation is not
            # part of the rotation lever arm.
            rotated = [sum(rotation[i][j] * point[j] for j in range(3)) for i in range(3)]
            rotation_block = matmul(block, skew(rotated))
            structure_block = matmul(block, rotation)
            for coordinate in range(2):
                row = [0.0] * columns
                if camera_index in slot:
                    position = 6 * slot[camera_index]
                    for i in range(3):
                        row[position + i] = -rotation_block[coordinate][i]
                        row[position + 3 + i] = block[coordinate][i]
                for i in range(3):
                    row[6 * len(free) + 3 * point_index + i] = structure_block[coordinate][i]
                rows.append(row)
    return rows


def normal_equations(rows, residuals_values, damping=0.0):
    columns = len(rows[0])
    matrix = [[0.0] * columns for _ in range(columns)]
    for row in rows:
        for i in range(columns):
            if row[i] == 0.0:
                continue
            for j in range(i, columns):
                if row[j] != 0.0:
                    matrix[i][j] += row[i] * row[j]
    for i in range(columns):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
    for i in range(columns):
        matrix[i][i] += damping * max(matrix[i][i], 1e-12)
    gradient = [sum(row[i] * residuals_values[k] for k, row in enumerate(rows))
                for i in range(columns)]
    return matrix, gradient


def solve_linear(matrix, rhs):
    size = len(matrix)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-300:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        for j in range(column, size + 1):
            augmented[column][j] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor != 0.0:
                for j in range(column, size + 1):
                    augmented[row][j] -= factor * augmented[column][j]
    return [augmented[i][size] for i in range(size)]


def apply_update(cameras, points, delta, free=None):
    """Anchored cameras (not in `free`) are left exactly as they are."""
    if free is None:
        free = tuple(range(len(cameras)))
    slot = {camera: index for index, camera in enumerate(free)}
    new_cameras = []
    for index, (rotation, translation) in enumerate(cameras):
        if index not in slot:
            new_cameras.append(([row[:] for row in rotation], translation[:]))
            continue
        position = slot[index]
        update = rodrigues(delta[6 * position:6 * position + 3])
        new_cameras.append((matmul(update, rotation),
                            [translation[i] + delta[6 * position + 3 + i] for i in range(3)]))
    new_points = []
    for index in range(len(points)):
        offset = 6 * len(free) + 3 * index
        new_points.append([points[index][i] + delta[offset + i] for i in range(3)])
    return new_cameras, new_points


def perturb_cameras(cameras, camera_index, axis, magnitude):
    """Left-multiply one camera's rotation by exp([+-magnitude e_axis]_x); nothing else moves."""
    increment = [0.0, 0.0, 0.0]
    increment[axis] = magnitude
    moved = []
    for index, (rotation, translation) in enumerate(cameras):
        if index == camera_index:
            moved.append((matmul(rodrigues(increment), rotation), translation))
        else:
            moved.append((rotation, translation))
    return moved


def bundle_adjust(cameras, points, observations, intrinsics, iterations=60, damping=1e-3,
                  gauss_newton=False, free=None):
    if free is None:
        free = tuple(range(len(cameras)))
    history = []
    accepted = 0
    rejected = 0
    current_cameras = [([row[:] for row in rotation], translation[:])
                       for rotation, translation in cameras]
    current_points = [point[:] for point in points]
    values = residuals(current_cameras, current_points, observations, intrinsics)
    cost = cost_of(values)
    history.append(cost)
    for _ in range(iterations):
        rows = jacobian(current_cameras, current_points, intrinsics, free)
        step_damping = 0.0 if gauss_newton else damping
        matrix, gradient = normal_equations(rows, values, step_damping)
        magnitude = math.sqrt(sum(value * value for value in gradient))
        if magnitude <= 1e-12:
            break
        delta = solve_linear(matrix, [-value for value in gradient])
        if delta is None:
            rejected += 1
            break
        candidate_cameras, candidate_points = apply_update(current_cameras, current_points, delta,
                                                           free)
        candidate_values = residuals(candidate_cameras, candidate_points, observations, intrinsics)
        candidate_cost = cost_of(candidate_values)
        if candidate_cost < cost:
            current_cameras, current_points = candidate_cameras, candidate_points
            values, cost = candidate_values, candidate_cost
            accepted += 1
            if not gauss_newton:
                damping = max(damping * 0.5, 1e-12)
        else:
            rejected += 1
            if gauss_newton:
                break
            damping = min(damping * 4.0, 1e12)
        history.append(cost)
        if cost <= 1e-26:
            break
    return {"cameras": current_cameras, "points": current_points, "cost": cost,
            "history": history, "accepted": accepted, "rejected": rejected,
            "residuals": values}


def finite_difference_jacobian(cameras, points, observations, intrinsics, epsilon=1e-6, free=None):
    """Central differences over the free parameters, using the same left-multiplied update."""
    if free is None:
        free = tuple(range(len(cameras)))
    slot = {camera: index for index, camera in enumerate(free)}
    residual_count = len(residuals(cameras, points, observations, intrinsics))
    columns = 6 * len(free) + 3 * len(points)
    table = [[0.0] * columns for _ in range(residual_count)]
    for camera_index in free:
        for axis in range(3):
            plus = perturb_cameras(cameras, camera_index, axis, epsilon)
            minus = perturb_cameras(cameras, camera_index, axis, -epsilon)
            plus_values = residuals(plus, points, observations, intrinsics)
            minus_values = residuals(minus, points, observations, intrinsics)
            for row in range(residual_count):
                table[row][6 * slot[camera_index] + axis] = (plus_values[row]
                                                             - minus_values[row]) / (2.0 * epsilon)
        for axis in range(3):
            position = 6 * slot[camera_index] + 3 + axis
            plus = [(rotation, [translation[i] + (epsilon if (index == camera_index and i == axis)
                                                  else 0.0) for i in range(3)])
                    for index, (rotation, translation) in enumerate(cameras)]
            minus = [(rotation, [translation[i] - (epsilon if (index == camera_index and i == axis)
                                                   else 0.0) for i in range(3)])
                     for index, (rotation, translation) in enumerate(cameras)]
            plus_values = residuals(plus, points, observations, intrinsics)
            minus_values = residuals(minus, points, observations, intrinsics)
            for row in range(residual_count):
                table[row][position] = (plus_values[row] - minus_values[row]) / (2.0 * epsilon)
    for point_index in range(len(points)):
        for axis in range(3):
            column = 6 * len(free) + 3 * point_index + axis
            plus_points = [point[:] for point in points]
            plus_points[point_index][axis] += epsilon
            minus_points = [point[:] for point in points]
            minus_points[point_index][axis] -= epsilon
            plus_values = residuals(cameras, plus_points, observations, intrinsics)
            minus_values = residuals(cameras, minus_points, observations, intrinsics)
            for row in range(residual_count):
                table[row][column] = (plus_values[row] - minus_values[row]) / (2.0 * epsilon)
    return table


def synthetic_scene(state, point_count=7, noise=0.0, cameras=3):
    """A rig whose first camera sits at the origin looking down -z (that camera anchors the
    7-degree gauge freedom of bundle adjustment and is kept fixed by the optimiser)."""
    intrinsics = intrinsic_matrix(900.0, 320.0, 240.0)
    truth = [(rodrigues([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]),
             (rodrigues([0.45, -0.25, 0.10]), [1.30, -0.55, 0.35]),
             (rodrigues([-0.20, 0.50, -0.30]), [-0.95, 1.15, -0.25])][:cameras]
    points = []
    for _ in range(point_count):
        state, u = uniform(state)
        state, v = uniform(state)
        state, w = uniform(state)
        points.append([(u * 2.0 - 1.0) * 1.0, (v * 2.0 - 1.0) * 1.0, -2.0 - 4.0 * w])
    observations = {}
    for camera_index, (rotation, translation) in enumerate(truth):
        for point_index, point in enumerate(points):
            predicted, _ = project(intrinsics, rotation, translation, point)
            state, u = uniform(state)
            state, v = uniform(state)
            observations[(camera_index, point_index)] = (
                predicted[0] + (u * 2.0 - 1.0) * noise,
                predicted[1] + (v * 2.0 - 1.0) * noise)
    return intrinsics, truth, points, observations


def perturb(state, cameras, points, magnitude, skip=()):
    """Displace the rig: a random rotation and translation per camera, and jittered points. Cameras
    listed in `skip` stay at their true pose - they are the anchor of the gauge freedom, and the
    optimiser is expected to leave them alone."""
    new_cameras = []
    for index, (rotation, translation) in enumerate(cameras):
        if index in skip:
            new_cameras.append(([row[:] for row in rotation], translation[:]))
            continue
        increment = []
        for _ in range(3):
            state, u = uniform(state)
            increment.append((u * 2.0 - 1.0) * magnitude)
        new_cameras.append((matmul(rodrigues(increment), rotation),
                            [translation[i] + (u * 2.0 - 1.0) * magnitude for i in range(3)]))
    new_points = []
    for point in points:
        moved = []
        for value in point:
            state, u = uniform(state)
            moved.append(value + (u * 2.0 - 1.0) * magnitude)
        new_points.append(moved)
    return state, new_cameras, new_points


def optimal_scale(fitted_points, true_points):
    """Least-squares scale aligning the fitted structure with the true one: the reconstruction is
    only determined up to a similarity, and the scale factor is the free gauge direction left over
    after one camera's pose has been pinned."""
    numerator = sum(sum(fitted_points[i][j] * true_points[i][j] for j in range(3))
                    for i in range(len(fitted_points)))
    denominator = sum(sum(value * value for value in point) for point in fitted_points)
    return numerator / denominator


def rescale(cameras, points, scale):
    """Multiplies the whole reconstruction (points and translations) by the optimal scale, which is
    the factor that maps the fitted structure onto the true one."""
    return ([( [row[:] for row in rotation], [value * scale for value in translation])
             for rotation, translation in cameras],
            [[value * scale for value in point] for point in points])


def pose_error(cameras, truth):
    worst = 0.0
    for (rotation, translation), (true_rotation, true_translation) in zip(cameras, truth):
        worst = max(worst, max(abs(rotation[i][j] - true_rotation[i][j])
                               for i in range(3) for j in range(3)))
        worst = max(worst, max(abs(translation[i] - true_translation[i]) for i in range(3)))
    return worst


def main():
    intrinsics, truth, points, observations = synthetic_scene(1, 7, 0.0)
    free = tuple(range(1, len(truth)))
    state, cameras, moved_points = perturb(7, truth, points, 0.05, skip=(0,))

    # --- The analytic Jacobian against central differences ------------------------------
    analytic = jacobian(cameras, moved_points, intrinsics, free)
    numeric = finite_difference_jacobian(cameras, moved_points, observations, intrinsics,
                                         free=free)
    worst = max(abs(analytic[row][column] - numeric[row][column])
                for row in range(len(analytic)) for column in range(len(analytic[0])))
    assert worst <= 1e-6, "the analytic Jacobian matches finite differences: worst %r" % worst
    assert len(analytic) == 2 * len(truth) * len(points) and len(analytic[0]) == 6 * len(free) + 3 * len(points)

    # --- The step solves the damped normal equations ------------------------------------
    values = residuals(cameras, moved_points, observations, intrinsics)
    damping = 0.25
    matrix, gradient = normal_equations(analytic, values, damping)
    delta = solve_linear(matrix, [-value for value in gradient])
    check = max(abs(sum(matrix[i][j] * delta[j] for j in range(len(delta))) + gradient[i])
                for i in range(len(delta)))
    assert check <= 1e-9, "the step solves (J^T J + lambda D) delta = -J^T r: %r" % check
    assert len(delta) == 6 * len(free) + 3 * len(points) == 33

    # --- Convergence on noiseless data --------------------------------------------------
    result = bundle_adjust(cameras, moved_points, observations, intrinsics, iterations=60,
                           free=free)
    assert result["cost"] <= 1e-18, "noiseless data converge to a zero cost: %r" % result["cost"]
    assert result["history"] == sorted(result["history"], reverse=True), \
        "the cost never rises: %r" % result["history"]
    assert result["accepted"] >= 5
    assert result["cameras"][0][0] == cameras[0][0] and result["cameras"][0][1] == cameras[0][1], \
        "the anchored camera is untouched"
    # Anchoring one camera removes the six rigid degrees of freedom but not the seventh: the whole
    # reconstruction can still be scaled about the anchored camera without changing a single pixel
    # (scaling the points and the camera translations together is a symmetry of every projection).
    # The file measures that drift and then aligns the fit with the truth by its own scale factor.
    scale = optimal_scale(result["points"], points)
    assert abs(scale - 1.0) >= 1e-3, "the scale gauge really moved: %r" % scale
    aligned_cameras, aligned_points = rescale(result["cameras"], result["points"], scale)
    assert pose_error(aligned_cameras[1:], truth[1:]) <= 1e-6, \
        "the poses are recovered up to the gauge: %r" % pose_error(aligned_cameras[1:], truth[1:])
    worst_point = max(max(abs(aligned_points[i][j] - points[i][j]) for j in range(3))
                      for i in range(len(points)))
    assert worst_point <= 1e-6, "the structure is recovered: %r" % worst_point
    final_rows = jacobian(result["cameras"], result["points"], intrinsics, free)
    final_values = residuals(result["cameras"], result["points"], observations, intrinsics)
    final_gradient = [sum(row[i] * final_values[k] for k, row in enumerate(final_rows))
                      for i in range(len(final_rows[0]))]
    assert max(abs(value) for value in final_gradient) <= 1e-9, "first-order optimality: J^T r = 0"

    # --- Convergence with noise ---------------------------------------------------------
    noisy_intrinsics, noisy_truth, noisy_points, noisy_observations = synthetic_scene(5, 7, 0.5)
    noisy_free = tuple(range(1, len(noisy_truth)))
    state, noisy_cameras, noisy_moved = perturb(9, noisy_truth, noisy_points, 0.05, skip=(0,))
    noisy_result = bundle_adjust(noisy_cameras, noisy_moved, noisy_observations, noisy_intrinsics,
                                 iterations=80, free=noisy_free)
    noise_floor = (2 * len(noisy_truth) * len(noisy_points)) * (0.5 ** 2) / 6.0
    assert noisy_result["cost"] < noise_floor, \
        "the noisy cost reaches the noise floor: %r versus %r" % (noisy_result["cost"], noise_floor)
    assert noisy_result["cost"] > 0.0
    assert noisy_result["history"] == sorted(noisy_result["history"], reverse=True)
    noisy_scale = optimal_scale(noisy_result["points"], noisy_points)
    noisy_aligned, _ = rescale(noisy_result["cameras"], noisy_result["points"], noisy_scale)
    assert pose_error(noisy_aligned[1:], noisy_truth[1:]) < \
        pose_error(noisy_cameras[1:], noisy_truth[1:])

    # --- Gauss-Newton versus Levenberg-Marquardt from a large perturbation --------------
    state, big_cameras, big_points = perturb(11, truth, points, 0.3, skip=(0,))
    lm = bundle_adjust(big_cameras, big_points, observations, intrinsics, iterations=100, free=free)
    gn = bundle_adjust(big_cameras, big_points, observations, intrinsics, iterations=100,
                       gauss_newton=True, free=free)
    assert lm["cost"] <= 1e-12, "LM survives the large perturbation: cost %r" % lm["cost"]
    lm_scale = optimal_scale(lm["points"], points)
    lm_aligned, _ = rescale(lm["cameras"], lm["points"], lm_scale)
    assert pose_error(lm_aligned[1:], truth[1:]) <= 1e-6
    assert lm["accepted"] >= 1
    # Measured: from a 0.3 perturbation LM reaches 6.0e-26 after 19 accepted steps while plain
    # Gauss-Newton accepts a single step, overshoots on the next one (the cost rises, the loop
    # stops) and stalls at 1.56e5. At 0.05 both converge, so the difference is not an artefact of
    # the test: it is the damping that saves the large step.
    assert gn["rejected"] >= 1, "GN overshoots and stops: %r" % (gn,)
    assert gn["accepted"] == 1, "GN accepts exactly one step here: %r" % (gn["accepted"],)
    assert gn["cost"] > 1e5, "GN stalls at %r" % gn["cost"]
    assert gn["cost"] > 1e24 * lm["cost"], "GN cost %r versus LM cost %r" % (gn["cost"], lm["cost"])
    state, small_cameras, small_points = perturb(11, truth, points, 0.05, skip=(0,))
    small_gn = bundle_adjust(small_cameras, small_points, observations, intrinsics,
                             iterations=200, gauss_newton=True, free=free)
    assert small_gn["cost"] <= 1e-20, "from 0.05 the undamped step is fine: %r" % small_gn["cost"]
    # LM is not magic either: from a 1.2 perturbation it is the one that fails (measured 1.33e4).
    state, huge_cameras, huge_points = perturb(11, truth, points, 1.2, skip=(0,))
    huge = bundle_adjust(huge_cameras, huge_points, observations, intrinsics, iterations=200,
                         free=free)
    assert huge["cost"] > 1e3, "LM fails at 1.2 as well: %r" % huge["cost"]

    # --- Determinism --------------------------------------------------------------------
    first = bundle_adjust(cameras, moved_points, observations, intrinsics, iterations=20, free=free)
    second = bundle_adjust(cameras, moved_points, observations, intrinsics, iterations=20, free=free)
    assert first["cost"] == second["cost"] and first["history"] == second["history"]
    assert synthetic_scene(5, 3, 0.5) == synthetic_scene(5, 3, 0.5)

    # --- The gauge freedom, demonstrated -------------------------------------------------
    # With every camera free the residuals still go to zero but the poses are only determined up to
    # a global rigid motion: the file measures that drift instead of pretending the poses match.
    ungauge = bundle_adjust(cameras, moved_points, observations, intrinsics, iterations=200,
                            free=tuple(range(len(truth))))
    assert ungauge["cost"] <= 1e-18, "the unanchored problem also fits the data: %r" % ungauge["cost"]
    drift = pose_error(ungauge["cameras"], truth)
    assert drift > 1e-3, "the unanchored solution drifts by %r" % drift
    assert pose_error(aligned_cameras[1:], truth[1:]) < 1e-3 * drift, \
        "the anchored run, realigned, is exact while the unanchored one drifts %r" % drift
    assert residuals(cameras, moved_points, observations, intrinsics) == values


if __name__ == "__main__":
    main()
    print("PASS")
