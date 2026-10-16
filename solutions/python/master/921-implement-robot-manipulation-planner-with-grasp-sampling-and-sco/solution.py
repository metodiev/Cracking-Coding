"""Task 921 - Implement robot manipulation planner with grasp sampling and scoring.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

Grasp planning for a rectangular 2D object with a two-finger gripper, plus the pick-and-place motion
that follows.  The boundary is sampled by arc length, a candidate grasp is a pair of contacts on
opposite edges far enough apart (either side may be the approach), and the score comes from force
closure: with the four extreme wrenches w = (f_x, f_y, c_x f_y - c_y f_x) for f = n +- mu t, the
grasp is force-closure exactly when the origin lies inside their convex hull, i.e. when the
barycentric system sum(lambda_i w_i) = 0, sum(lambda_i) = 1 has every lambda_i > 0; the smallest
lambda is the closure quality.  Ranking multiplies quality by the contact separation and the first
feasible grasp (reachable annulus, approach from above, no obstacle) yields the waypoint sequence.
Verified: a centred opposite-edge grasp is force closure while a same-face pair is not, quality
grows with mu, a 3x finer grid confirms the sampling is near-optimal, the blocked best grasp is
skipped, the placed contacts land on the target object's boundary, and infeasible sets report
failure.  Simplified: planar grasps, a rectangular object, one gripper, disc obstacles.
"""
import math

FRICTION, K_SAMPLES, MIN_SEPARATION = 0.5, 40, 0.15
HALF_EXTENTS = (0.3, 0.2)
OBJECT_CENTRE = (0.9, 0.0)
TARGET = (0.6, 0.4, math.pi / 6)
BASE, REACH = (0.0, 0.0), (0.25, 1.1)
MIN_APPROACH_Y = -0.4
OBSTACLES = [((0.5, 0.0), 0.25)]

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def scale(a, k):
    return (a[0] * k, a[1] * k)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def norm(a):
    return math.hypot(a[0], a[1])

def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])

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

def boundary_samples(half, count):
    """Points and outward normals around a rectangle, sampled by edge length (deterministic)."""
    a, b = half
    edges = [((a, -b), (a, b), (1.0, 0.0)), ((-a, b), (a, b), (0.0, 1.0)),
             ((-a, -b), (-a, b), (-1.0, 0.0)), ((-a, -b), (a, -b), (0.0, -1.0))]
    perimeter = 4.0 * (a + b)
    if perimeter <= 0.0:
        return []
    samples = []
    for start, end, normal in edges:
        length = norm(sub(end, start))
        steps = max(1, int(round(count * length / perimeter)))
        for k in range(steps):
            samples.append((add(start, scale(sub(end, start), k / steps)), normal))
    return samples

def arc_parameters(half, count):
    a, b = half
    perimeter = 4.0 * (a + b)
    return [index * perimeter / count for index in range(count)]

def sample_grasps(half, count, min_separation):
    """Antipodal candidate pairs: opposite normals along different edges, far enough apart."""
    samples = boundary_samples(half, count)
    arcs = arc_parameters(half, count)
    grasps = []
    for i in range(len(samples)):
        for j in range(len(samples)):
            point_i, normal_i = samples[i]
            point_j, normal_j = samples[j]
            if dot(normal_i, normal_j) > -0.999 or i == j:
                continue
            if min(arcs[i], arcs[j]) + 4.0 * sum(half) - max(arcs[i], arcs[j]) < min_separation:
                continue
            if (i, j) >= (j, i):                      # keep one ordering per unordered pair
                continue
            grasps.append((point_i, normal_i, point_j, normal_j))
    return grasps

def wrenches(contact_a, normal_a, contact_b, normal_b, mu):
    """Four extreme wrenches of the two friction cones, as (force_x, force_y, torque)."""
    out = []
    for contact, normal in ((contact_a, normal_a), (contact_b, normal_b)):
        tangent = (-normal[1], normal[0])
        for sign in (1.0, -1.0):
            force = add(normal, scale(tangent, sign * mu))
            torque = contact[0] * force[1] - contact[1] * force[0]
            out.append((force[0], force[1], torque))
    return out

def closure_quality(grasp, mu):
    """Smallest barycentric weight of the origin in the wrench hull; <= 0 means no force closure."""
    contact_a, normal_a, contact_b, normal_b = grasp
    points = wrenches(sub(contact_a, OBJECT_CENTRE), normal_a,
                      sub(contact_b, OBJECT_CENTRE), normal_b, mu)
    matrix = [[point[0] for point in points], [point[1] for point in points],
              [point[2] for point in points], [1.0, 1.0, 1.0, 1.0]]
    weights = solve_linear(matrix, [0.0, 0.0, 0.0, 1.0])
    if weights is None:
        return -1.0
    return min(weights)

def score(grasp, mu):
    quality = closure_quality(grasp, mu)
    if quality <= 1e-9:
        return None
    return quality * norm(sub(grasp[0], grasp[2]))

def world(point):
    """Map a point given in object-local coordinates into world coordinates."""
    return add(point, OBJECT_CENTRE)

def rank_grasps(grasps, mu):
    """Feasible-scoring grasps sorted by descending score; ties prefer the most centred grasp.

    Grasps are local to the object, so "centred" means the midpoint closest to the object's centre
    of mass, which is what keeps the payload balanced during the transfer.
    """
    scored = [item for item in
              ((score(grasp, mu), norm(scale(add(grasp[0], grasp[2]), 0.5)), index, grasp)
               for index, grasp in enumerate(grasps)) if item[0] is not None]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(score, index, grasp) for score, _centred, index, grasp in scored]

def blocked(point):
    return any(norm(sub(point, centre)) < radius for centre, radius in OBSTACLES)

def reachable(point):
    distance = norm(sub(point, BASE))
    return REACH[0] <= distance <= REACH[1]

def feasible(grasp, approach_side):
    """A grasp is executable when the approach comes from above and starts in reach, collision free."""
    contact = world(grasp[0] if approach_side == 0 else grasp[2])
    normal = grasp[1] if approach_side == 0 else grasp[3]
    pregrasp = add(contact, scale(normal, 0.1))
    clear = all(not blocked(add(contact, scale(sub(pregrasp, contact), t / 8.0))) for t in range(9))
    return normal[1] >= MIN_APPROACH_Y and reachable(pregrasp) and clear, pregrasp

def plan(grasps, mu):
    """Pick and place: returns (waypoints, chosen grasp index, score) or None when nothing works."""
    for grasp_score, index, grasp in rank_grasps(grasps, mu):
        for side in (0, 1):
            ok, pregrasp = feasible(grasp, side)
            if not ok:
                continue
            middle = world(scale(add(grasp[0], grasp[2]), 0.5))
            placed = (TARGET[0], TARGET[1])
            waypoints = [pregrasp, middle, add(middle, (0.0, 0.15)), add(placed, (0.0, 0.15)),
                         placed, add(placed, (0.0, 0.15))]
            return waypoints, index, grasp_score
    return None

def main():
    # 1. Force-closure detection on hand-built grasps (a positive and a negative case).
    closure = ((( 0.0, 0.2), (0.0, 1.0), (0.0, -0.2), (0.0, -1.0)))
    same_face = (((0.0, 0.2), (0.0, 1.0), (0.1, 0.2), (0.0, 1.0)))
    check(abs(closure_quality(closure, FRICTION) - 0.25) < 1e-9,
          "a symmetric two-finger grasp must be force closure with quality 1/4")
    check(closure_quality(same_face, FRICTION) <= 1e-9,
          "two contacts on the same face with parallel normals must not be force closure")

    # 2. Quality grows with friction; a lateral offset needs enough friction for force closure.
    offset = (((0.1, 0.2), (0.0, 1.0), (-0.1, -0.2), (0.0, -1.0)))
    ladder = [closure_quality(offset, mu) for mu in (0.1, 0.3, 0.7, 1.0)]
    check(all(a < b for a, b in zip(ladder, ladder[1:])),
          "closure quality must increase with the friction coefficient")
    check(ladder[1] <= 0.0 < ladder[2],
          "a laterally offset grasp needs enough friction to be force closure")
    check(closure_quality(closure, 0.0) <= 1e-9,
          "without friction a two-contact antipodal grasp cannot be force closure")

    # 3. Sampling: deterministic, antipodal, and every pair spans more than the minimum separation.
    grasps = sample_grasps(HALF_EXTENTS, K_SAMPLES, MIN_SEPARATION)
    check(len(grasps) == len(sample_grasps(HALF_EXTENTS, K_SAMPLES, MIN_SEPARATION)) and grasps,
          "grasp sampling must be deterministic and non-empty")
    check(all(dot(g[1], g[3]) < -0.999 and norm(sub(g[0], g[2])) >= MIN_SEPARATION - 1e-9
              for g in grasps), "every sample must be an antipodal pair with enough separation")

    # 4. Independent oracle: a 3x finer grid cannot beat the sampled ranking by much.
    ranked = rank_grasps(grasps, FRICTION)
    fine = rank_grasps(sample_grasps(HALF_EXTENTS, K_SAMPLES * 3, MIN_SEPARATION), FRICTION)
    check(fine and ranked[0][0] >= 0.8 * fine[0][0],
          "the coarse sampling must be near-optimal against the finer grid")

    # 5. Planning: the best-scoring grasp is blocked, so the next feasible one is chosen.
    planned = plan(grasps, FRICTION)
    check(planned is not None, "a feasible grasp must exist in this scene")
    waypoints, chosen, chosen_score = planned
    check(chosen != ranked[0][1] and chosen_score > 0.0,
          "the planner must skip the blocked best grasp and take the next feasible one")
    check(all(reachable(point) and not blocked(point) for point in waypoints),
          "every planned waypoint must be reachable and collision free")

    # 6. Placement precision: the rigid transform of the grasp carries the object onto the target,
    #    which requires the chosen grasp to be centred on the object's centre of mass.
    a, b = HALF_EXTENTS
    grasp = grasps[chosen]
    midpoint = world(scale(add(grasp[0], grasp[2]), 0.5))
    move = lambda point: add(rotate(sub(point, midpoint), TARGET[2]), TARGET[:2])
    check(norm(sub(move(midpoint), TARGET[:2])) < 1e-9,
          "the grasp midpoint must be carried exactly onto the target")
    for contact in (grasp[0], grasp[2]):
        local = rotate(sub(move(world(contact)), TARGET[:2]), -TARGET[2])
        check(abs(max(abs(local[0]) / a, abs(local[1]) / b) - 1.0) < 1e-9,
              "a placed contact must still lie on the target object's boundary")
    check(abs(norm(sub(move(world(grasp[0])), move(world(grasp[2]))))
            - norm(sub(grasp[0], grasp[2]))) < 1e-9,
          "the placement transform must preserve the grasp width")

    # 7. Boundary cases: empty samples and a scene with every approach blocked.
    check(sample_grasps((0.0, 0.0), 8, 0.05) == [], "a degenerate object has no boundary samples")
    check(plan([], FRICTION) is None, "an empty sample set must report failure, not crash")
    OBSTACLES.append(((0.75, 0.1), 1.6))
    try:
        check(plan(grasps, FRICTION) is None, "a fully blocked scene must report failure")
    finally:
        OBSTACLES.pop()

    # 8. Determinism: rerunning the sampler and planner reproduces the same decision.
    again = plan(sample_grasps(HALF_EXTENTS, K_SAMPLES, MIN_SEPARATION), FRICTION)
    check(again == planned, "the planner must be deterministic")

    print(f"sampled {len(grasps)} antipodal grasps, best score {ranked[0][0]:.4f} (blocked), "
          f"chosen score {chosen_score:.4f}")
    print(f"waypoints {len(waypoints)}, closure quality of the chosen grasp "
          f"{closure_quality(grasps[chosen], FRICTION):.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
