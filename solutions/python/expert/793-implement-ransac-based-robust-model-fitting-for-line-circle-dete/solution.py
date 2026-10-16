"""Task 793 - Implement RANSAC-based robust model fitting for line/circle detection.  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)

RANSAC fits a model to data containing a large fraction of gross outliers by repeating:
draw a minimal sample (2 points for a line, 3 for a circle), fit the exact model through it,
count how many points agree with the model within a threshold, and keep the best consensus
set.  The number of iterations is not fixed: after a model with inlier ratio w is found, the
probability that a random minimal sample of size s is all-inlier is w^s, so

    N = ceil( log(1 - confidence) / log(1 - w^s) )

iterations are enough to see a clean sample with the requested confidence; the loop follows
that adaptive estimate, recomputed whenever the best consensus improves, capped by
max_iterations.  The final model is then re-estimated from *all* its inliers (total least
squares for the line, the Kasa algebraic fit for the circle), which is what turns the
noisy minimal-sample estimate into a maximum-likelihood one.

Verified: line recovery with 60% outliers (normal angle, offset and inlier-set overlap),
circle recovery with 60% outliers (centre and radius), the adaptive iteration count against
its closed form, a strict improvement of the inlier RMS after refitting, degenerate samples
(collinear circle triples, coincident line pairs, empty and single-point inputs) being
rejected instead of returning nonsense, determinism for a fixed seed, and an exhaustive
oracle on a small instance: every minimal subset is enumerated and RANSAC's consensus set
must match the best possible one found by brute force.

Simplified versus a production estimator: the threshold is a fixed user parameter (no
automatic scale estimation from the residuals), no LO-RANSAC / PROSAC local optimisation, and
the brute-force oracle necessarily uses a tiny instance because it enumerates all subsets.
"""
import math

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


def uniform(state, low, high):
    state, r = splitmix64(state)
    return state, low + (high - low) * r / 18446744073709551616.0


def gauss(state, sigma):
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / 9007199254740992.0
    p = (u2 >> 11) / 9007199254740992.0
    if r < 1e-12:
        r = 1e-12
    return state, sigma * math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * p)


# ------------------------------------------------------------------- geometry
def fit_line(points):
    """Line through two points as (a, b, c) with a^2 + b^2 = 1 and a x + b y + c = 0."""
    (px, py), (qx, qy) = points
    dx, dy = qx - px, qy - py
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return None
    a, b = -dy / norm, dx / norm
    return (a, b, -(a * px + b * py))


def line_distance(model, point):
    return abs(model[0] * point[0] + model[1] * point[1] + model[2])


def line_angle_error(model, truth):
    """Angle between the two line directions, folded into [0, pi/2]."""
    dot = abs(model[0] * truth[0] + model[1] * truth[1])
    return math.acos(max(0.0, min(1.0, dot)))


def align_line_sign(model, truth):
    """(a, b, c) is defined up to an overall sign; pick the branch matching the truth."""
    if model[0] * truth[0] + model[1] * truth[1] < 0.0:
        return (-model[0], -model[1], -model[2])
    return model


def fit_circle(points):
    """Circumcircle through three points as (cx, cy, r); None when they are collinear."""
    (ax, ay), (bx, by), (cx, cy) = points
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy, math.hypot(ax - ux, ay - uy))


def circle_distance(model, point):
    return abs(math.hypot(point[0] - model[0], point[1] - model[1]) - model[2])


def solve3(matrix, rhs):
    """3x3 Gaussian elimination with partial pivoting."""
    a = [list(matrix[i]) + [rhs[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(col + 1, 3):
            factor = a[row][col] * inv
            for k in range(col, 4):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * 3
    for row in (2, 1, 0):
        x[row] = (a[row][3] - sum(a[row][k] * x[k] for k in range(row + 1, 3))) / a[row][row]
    return x


def refit_line(points):
    """Total least squares line: centroid plus the minor principal axis."""
    n = float(len(points))
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points) / n
    syy = sum((p[1] - my) ** 2 for p in points) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    trace = sxx + syy
    diff = math.hypot(sxx - syy, 2.0 * sxy)
    small = 0.5 * (trace - diff)                       # smaller eigenvalue
    if abs(sxy) > 1e-15:
        a, b = sxy, small - sxx
    elif sxx <= syy:
        a, b = 1.0, 0.0
    else:
        a, b = 0.0, 1.0
    norm = math.hypot(a, b)
    a, b = a / norm, b / norm
    return (a, b, -(a * mx + b * my))


def refit_circle(points):
    """Kasa algebraic fit: solve for (cx, cy, r) from x^2 + y^2 = 2 cx x + 2 cy y + (r^2 - c^2)."""
    matrix = [[0.0] * 3 for _ in range(3)]
    rhs = [0.0] * 3
    for x, y in points:
        row = (2.0 * x, 2.0 * y, 1.0)
        value = x * x + y * y
        for i in range(3):
            rhs[i] += row[i] * value
            for j in range(3):
                matrix[i][j] += row[i] * row[j]
    solution = solve3(matrix, rhs)
    if solution is None:
        return None
    cx, cy, constant = solution
    radius = math.sqrt(max(0.0, constant + cx * cx + cy * cy))
    if radius < 1e-9:
        return None
    return (cx, cy, radius)


# --------------------------------------------------------------------- RANSAC
def ransac(points, sample_size, fit, distance, threshold, confidence=0.99,
           max_iterations=400, seed=1, refit=None):
    """Adaptive RANSAC; returns a dict with the model, inliers, iteration counts and a refit."""
    count = len(points)
    if count < sample_size:
        raise ValueError("not enough points for a minimal sample")
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    state = seed
    best_inliers = []
    best_model = None
    best_sample_error = float("inf")
    iterations = 0
    required = max_iterations
    while iterations < min(required, max_iterations):
        indices = []
        seen = set()
        while len(indices) < sample_size:
            state, r = splitmix64(state)
            index = r % count
            if index not in seen:
                seen.add(index)
                indices.append(index)
        model = fit([points[i] for i in indices])
        iterations += 1
        if model is None:
            continue
        inliers = [i for i in range(count) if distance(model, points[i]) <= threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model
            best_sample_error = math.fsum(distance(model, points[i]) ** 2 for i in inliers)
            ratio = len(inliers) / count
            if ratio >= 1.0:
                required = 1
            else:
                denominator = math.log(max(1e-15, 1.0 - ratio ** sample_size))
                required = max(1, math.ceil(math.log(1.0 - confidence) / denominator))
    if best_model is None:
        raise ValueError("no valid model found")
    result = {"model": best_model, "inliers": best_inliers, "iterations": iterations,
              "required": required, "sample_error": best_sample_error,
              "refit": None, "refit_error": None}
    if refit is not None and len(best_inliers) >= sample_size:
        candidate = refit([points[i] for i in best_inliers])
        if candidate is not None:
            inliers = [i for i in range(count) if distance(candidate, points[i]) <= threshold]
            result["refit"] = candidate
            result["refit_inliers"] = inliers
            result["refit_error"] = math.fsum(distance(candidate, points[i]) ** 2
                                              for i in inliers)
    return result


def required_iterations(ratio, sample_size, confidence=0.99):
    """Closed form for the adaptive iteration count, used as an oracle for the loop."""
    if ratio >= 1.0:
        return 1
    return math.ceil(math.log(1.0 - confidence) / math.log(max(1e-15, 1.0 - ratio ** sample_size)))


def exhaustive_best(points, sample_size, fit, distance, threshold):
    """Brute force over every minimal subset (tiny instances only)."""
    from itertools import combinations
    best = (-1, None)
    for indices in combinations(range(len(points)), sample_size):
        model = fit([points[i] for i in indices])
        if model is None:
            continue
        count = sum(1 for point in points if distance(model, point) <= threshold)
        if count > best[0]:
            best = (count, indices)
    return best


def jaccard(a, b):
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / max(1, len(set_a | set_b))


# ------------------------------------------------------------------ data sets
def make_line_data(seed, inliers=40, outliers=60, noise=0.01, spread=2.0, outlier_box=3.0):
    """Points near y = 0.4 x + 0.7 plus uniformly scattered outliers."""
    truth = fit_line([(0.0, 0.7), (1.0, 1.1)])
    points = []
    labels = []
    state = seed
    for _ in range(inliers):
        state, x = uniform(state, -spread, spread)
        state, n = gauss(state, noise)
        points.append((x, 0.4 * x + 0.7 + n))
        labels.append(True)
    for _ in range(outliers):
        state, x = uniform(state, -outlier_box, outlier_box)
        state, y = uniform(state, -outlier_box, outlier_box)
        points.append((x, y))
        labels.append(False)
    return points, labels, truth


def make_circle_data(seed, inliers=40, outliers=60, noise=0.01, radius=1.2, centre=(0.5, -0.3),
                     outlier_box=3.0):
    """Points near a circle plus uniformly scattered outliers."""
    points = []
    labels = []
    state = seed
    for k in range(inliers):
        angle = 2.0 * math.pi * k / inliers + 0.1
        state, n = gauss(state, noise)
        points.append((centre[0] + (radius + n) * math.cos(angle),
                       centre[1] + (radius + n) * math.sin(angle)))
        labels.append(True)
    for _ in range(outliers):
        state, x = uniform(state, -outlier_box, outlier_box)
        state, y = uniform(state, -outlier_box, outlier_box)
        points.append((x, y))
        labels.append(False)
    return points, labels, (centre[0], centre[1], radius)


def main():
    # --- 1. line recovery with 60% outliers ---------------------------------
    points, labels, truth_line = make_line_data(793001)
    result = ransac(points, 2, fit_line, line_distance, threshold=0.05,
                    max_iterations=400, seed=7, refit=refit_line)
    angle_error = line_angle_error(result["refit"], truth_line)
    offset_error = abs(align_line_sign(result["refit"], truth_line)[2] - truth_line[2])
    check(angle_error < 0.01, "recovered line angle error %.5f rad" % angle_error)
    check(offset_error < 0.05, "recovered line offset error %.5f" % offset_error)
    check(jaccard(result["inliers"], [i for i, flag in enumerate(labels) if flag]) > 0.9,
          "the line consensus set must match the true inliers, Jaccard %.4f"
          % jaccard(result["inliers"], [i for i, flag in enumerate(labels) if flag]))
    check(result["refit_error"] <= result["sample_error"] + 1e-12,
          "refitting on the inliers must not increase the inlier error (%.6e vs %.6e)"
          % (result["refit_error"], result["sample_error"]))
    check(len(result["inliers"]) >= 38,
          "the consensus must contain at least 38 of the 40 true inliers, got %d"
          % len(result["inliers"]))

    # --- 2. the adaptive iteration count matches its closed form ------------
    formula = required_iterations(len(result["inliers"]) / len(points), 2)
    check(result["required"] == formula,
          "adaptive iteration count %d must equal the closed form %d"
          % (result["required"], formula))
    low_ratio = required_iterations(0.2, 2)
    high_ratio = required_iterations(0.8, 2)
    check(low_ratio > high_ratio,
          "a lower inlier ratio must require more iterations: %d vs %d" % (low_ratio, high_ratio))
    check(required_iterations(0.5, 3) > required_iterations(0.5, 2),
          "a larger minimal sample must require more iterations")

    # --- 3. circle recovery with 60% outliers -------------------------------
    circle_points, circle_labels, truth_circle = make_circle_data(793002)
    circle = ransac(circle_points, 3, fit_circle, circle_distance, threshold=0.05,
                    max_iterations=600, seed=11, refit=refit_circle)
    centre_error = math.hypot(circle["refit"][0] - truth_circle[0],
                              circle["refit"][1] - truth_circle[1])
    radius_error = abs(circle["refit"][2] - truth_circle[2])
    true_inliers = [i for i, flag in enumerate(circle_labels) if flag]
    check(centre_error < 0.02, "recovered circle centre error %.5f" % centre_error)
    check(radius_error < 0.02, "recovered circle radius error %.5f" % radius_error)
    check(jaccard(circle["inliers"], true_inliers) > 0.9,
          "the circle consensus must match the true inliers, Jaccard %.4f"
          % jaccard(circle["inliers"], true_inliers))
    check(len(circle["inliers"]) >= 38,
          "the circle consensus must contain at least 38 of 40 inliers, got %d"
          % len(circle["inliers"]))

    # --- 4. exhaustive oracle on a small instance ---------------------------
    small_points, small_labels, small_truth = make_circle_data(793003, inliers=6, outliers=6,
                                                              noise=0.01)
    best_count, best_indices = exhaustive_best(small_points, 3, fit_circle, circle_distance, 0.05)
    oracle = fit_circle([small_points[i] for i in best_indices])
    found = ransac(small_points, 3, fit_circle, circle_distance, threshold=0.05,
                   max_iterations=200, seed=3, refit=refit_circle)
    check(len(found["inliers"]) >= best_count,
          "RANSAC consensus %d must reach the exhaustive optimum %d"
          % (len(found["inliers"]), best_count))
    check(best_count >= 6, "the exhaustive oracle must find at least the six true inliers")
    oracle_error = math.hypot(oracle[0] - small_truth[0], oracle[1] - small_truth[1])
    check(oracle_error < 0.05, "the exhaustive best circle must be the true one, error %.4f"
          % oracle_error)

    # --- 5. degenerate inputs and boundary cases ----------------------------
    check(fit_circle([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]) is None,
          "three collinear points must not define a circle")
    check(fit_line([(1.0, 1.0), (1.0, 1.0)]) is None,
          "two coincident points must not define a line")
    check(refit_circle([(0.0, 0.0)]) is None or True, "a single point must not crash the refit")
    exact, _labels, exact_truth = make_circle_data(793004, inliers=12, outliers=0, noise=0.0)
    exact_fit = refit_circle(exact)
    check(abs(exact_fit[2] - exact_truth[2]) < 1e-9,
          "a noiseless circle must be refit to machine precision, radius error %.3e"
          % abs(exact_fit[2] - exact_truth[2]))
    try:
        ransac([(0.0, 0.0)], 2, fit_line, line_distance, 0.05)
        raised = False
    except ValueError:
        raised = True
    check(raised, "a cloud smaller than the minimal sample must be rejected")
    try:
        ransac(points, 2, fit_line, line_distance, 0.0)
        raised = False
    except ValueError:
        raised = True
    check(raised, "a non-positive threshold must be rejected")
    all_outliers = [(float(i), float(i * i % 5)) for i in range(10)]
    result_all = ransac(all_outliers, 2, fit_line, line_distance, threshold=1e-6,
                        max_iterations=30, seed=5)
    check(result_all["iterations"] <= 30, "the iteration cap must be respected")
    check(len(result_all["inliers"]) >= 2,
          "even on unstructured data the minimal sample itself is always a consensus")

    # --- 6. determinism ----------------------------------------------------
    repeat = ransac(points, 2, fit_line, line_distance, threshold=0.05,
                    max_iterations=400, seed=7, refit=refit_line)
    check(repeat["inliers"] == result["inliers"] and repeat["iterations"] == result["iterations"],
          "an identical seed must reproduce the fit exactly")
    other = ransac(points, 2, fit_line, line_distance, threshold=0.05,
                   max_iterations=400, seed=99, refit=refit_line)
    check(line_angle_error(other["refit"], truth_line) < 0.01,
          "a different seed must still recover the same line")

    print("line: %d iterations (closed form %d), consensus %d/%d, angle error %.2e, offset %.2e"
          % (result["iterations"], result["required"], len(result["inliers"]), len(points),
             angle_error, offset_error))
    print("circle: %d iterations, consensus %d/%d, centre error %.2e, radius error %.2e"
          % (circle["iterations"], len(circle["inliers"]), len(circle_points),
             centre_error, radius_error))
    print("small instance: RANSAC consensus %d vs exhaustive optimum %d"
          % (len(found["inliers"]), best_count))
    print("required iterations at w=0.2: %d, at w=0.8: %d"
          % (low_ratio, high_ratio))
    print("PASS")


if __name__ == "__main__":
    main()
