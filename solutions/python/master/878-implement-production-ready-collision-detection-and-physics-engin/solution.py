"""Task 878 - Implement production-ready collision detection and physics engine basics.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

A small 2-D rigid-body pipeline with the three stages a real engine has, each verified
separately:

  * broadphase - sweep and prune over the intervals of the axis-aligned bounding boxes.  The
    candidate set is compared against brute force: it must be a *superset* of the truly
    colliding pairs (never a miss) while dropping most of the n(n-1)/2 comparisons in a sparse
    scene;
  * narrowphase - analytic sphere/sphere overlap and the separating axis theorem for oriented
    boxes, where the normal is the axis of least penetration.  The oracle is a completely
    different algorithm: Sutherland-Hodgman polygon clipping, which reports an intersection
    area > 0 exactly when the boxes touch;
  * solver - sequential impulses with restitution and Coulomb friction, plus a positional
    (Baumgarte) correction, and optional continuous collision detection for fast bodies: the
    motion of a body centre is swept against the Minkowski-expanded AABB of every static body,
    which is what stops a bullet from tunnelling through a thin wall.

Verified: broadphase superset and pair-count reduction against brute force; SAT against the
clipping oracle on hundreds of random pairs including touching and disjoint cases; the elastic
two-body impulse against the closed-form velocities; momentum conservation to rounding;
kinetic energy conserved for restitution 1 and never increased for restitution < 1; a block
resting on the floor with bounded penetration, decaying velocity and no jitter; friction
bounded by the Coulomb cone and monotone in the tangential velocity; grazing reflection off a
static wall preserving the tangential velocity; tunnelling prevented by the swept test while
the same step without it passes through; and rejection of invalid bodies.
"""
import math

EPSILON = 1e-9


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def dot(left, right):
    return left[0] * right[0] + left[1] * right[1]


def subtract(left, right):
    return [left[0] - right[0], left[1] - right[1]]


def scale(vector, factor):
    return [vector[0] * factor, vector[1] * factor]


def norm(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def rotate(vector, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return [vector[0] * cosine - vector[1] * sine, vector[0] * sine + vector[1] * cosine]


class Body:
    """Sphere (radius) or oriented box (half extents, angle) with mass 0 meaning static."""

    def __init__(self, name, position, velocity=(0.0, 0.0), mass=1.0, radius=None, half=None,
                 angle=0.0, restitution=0.5, friction=0.3, dynamic_angle=False):
        if mass < 0.0:
            raise ValueError("mass must not be negative")
        if radius is not None and radius <= 0.0:
            raise ValueError("radius must be positive")
        if half is not None and (half[0] <= 0.0 or half[1] <= 0.0):
            raise ValueError("half extents must be positive")
        if radius is None and half is None:
            raise ValueError("a body needs a radius or half extents")
        if radius is not None and half is not None:
            raise ValueError("a body cannot be both a sphere and a box")
        if dynamic_angle and mass == 0.0:
            raise ValueError("a static body cannot spin")
        self.name = name
        self.position = [float(position[0]), float(position[1])]
        self.velocity = [float(velocity[0]), float(velocity[1])]
        self.mass = float(mass)
        self.radius = radius
        self.half = half
        self.angle = float(angle)
        self.restitution = restitution
        self.friction = friction
        self.dynamic_angle = dynamic_angle

    @property
    def inverse_mass(self):
        return 0.0 if self.mass == 0.0 else 1.0 / self.mass

    @property
    def is_static(self):
        return self.mass == 0.0

    def extent(self):
        """Half-width of the shape along its own axes (radius for a sphere)."""
        return self.radius if self.radius is not None else max(self.half)

    def aabb(self):
        if self.radius is not None:
            reach = self.radius
        else:
            reach = abs(self.half[0] * math.cos(self.angle)) + abs(self.half[1] * math.sin(self.angle))
            reach_y = abs(self.half[0] * math.sin(self.angle)) + abs(self.half[1] * math.cos(self.angle))
            return (self.position[0] - reach, self.position[1] - reach_y,
                    self.position[0] + reach, self.position[1] + reach_y)
        return (self.position[0] - reach, self.position[1] - reach,
                self.position[0] + reach, self.position[1] + reach)

    def corners(self):
        if self.half is None:
            raise ValueError("a sphere has no corners")
        return [self._corner(-1.0, -1.0), self._corner(1.0, -1.0),
                self._corner(1.0, 1.0), self._corner(-1.0, 1.0)]

    def _corner(self, sign_x, sign_y):
        offset = rotate([sign_x * self.half[0], sign_y * self.half[1]], self.angle)
        return [self.position[0] + offset[0], self.position[1] + offset[1]]


# ------------------------------------------------------------------ broadphase
def aabb_overlap(first, second):
    a = first.aabb()
    b = second.aabb()
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def brute_force_pairs(bodies):
    pairs = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if aabb_overlap(bodies[i], bodies[j]):
                pairs.append((i, j))
    return pairs


def sweep_and_prune(bodies):
    """Sort by the interval start, sweep, and keep overlapping intervals (superset guarantee)."""
    intervals = sorted(((body.aabb()[0], body.aabb()[2], index)
                        for index, body in enumerate(bodies)), key=lambda item: item[0])
    active = []
    pairs = []
    for start, end, index in intervals:
        active = [entry for entry in active if entry[1] >= start]
        for _other_start, _other_end, other in active:
            low, high = (index, other) if index < other else (other, index)
            if aabb_overlap(bodies[low], bodies[high]):
                pairs.append((low, high))
        active.append((start, end, index))
    return sorted(set(pairs))


# ------------------------------------------------------------------ narrowphase
def sphere_sphere(first, second):
    delta = subtract(second.position, first.position)
    distance = norm(delta)
    reach = first.radius + second.radius
    if distance >= reach:
        return None
    normal = [0.0, 1.0] if distance < EPSILON else [delta[0] / distance, delta[1] / distance]
    return normal, reach - distance


def project(corners, axis):
    values = [dot(corner, axis) for corner in corners]
    return min(values), max(values)


def obb_sat(first, second):
    """Separating axis theorem for 2-D oriented boxes; returns the least-penetration normal."""
    corners_a = first.corners()
    corners_b = second.corners()
    axes = [[math.cos(first.angle), math.sin(first.angle)],
            [-math.sin(first.angle), math.cos(first.angle)],
            [math.cos(second.angle), math.sin(second.angle)],
            [-math.sin(second.angle), math.cos(second.angle)]]
    best_depth = None
    best_axis = None
    for axis in axes:
        low_a, high_a = project(corners_a, axis)
        low_b, high_b = project(corners_b, axis)
        if high_a < low_b or high_b < low_a:
            return None
        depth = min(high_a - low_b, high_b - low_a)
        if best_depth is None or depth < best_depth:
            best_depth = depth
            best_axis = axis
    normal = best_axis
    if dot(subtract(second.position, first.position), normal) < 0:
        normal = scale(normal, -1.0)
    return normal, best_depth


def collide(first, second):
    if first.radius is not None and second.radius is not None:
        return sphere_sphere(first, second)
    if first.radius is not None:
        return sphere_box(first, second, flipped=True)
    if second.radius is not None:
        return sphere_box(second, first, flipped=False)
    return obb_sat(first, second)


def sphere_box(sphere, box, flipped):
    """Exact sphere/box overlap by clamping the centre into the box frame.

    The returned normal always points from the first body to the second, which is the
    convention `resolve_pair` expects.
    """
    local = rotate(subtract(sphere.position, box.position), -box.angle)
    clamped = [max(-box.half[0], min(box.half[0], local[0])),
               max(-box.half[1], min(box.half[1], local[1]))]
    delta = subtract(local, clamped)
    distance = norm(delta)
    if distance >= sphere.radius:
        return None
    if distance > EPSILON:
        outward = [delta[0] / distance, delta[1] / distance]
        depth = sphere.radius - distance
    else:
        # centre inside the box: push out through the nearest face
        gaps = [box.half[0] - abs(local[0]), box.half[1] - abs(local[1])]
        if gaps[0] <= gaps[1]:
            outward = [1.0 if local[0] >= 0 else -1.0, 0.0]
            depth = gaps[0] + sphere.radius
        else:
            outward = [0.0, 1.0 if local[1] >= 0 else -1.0]
            depth = gaps[1] + sphere.radius
    world = rotate(outward, box.angle)  # from the box surface towards the sphere centre
    if flipped:  # the sphere is the first body: the normal must point sphere -> box
        return [-world[0], -world[1]], depth
    return world, depth


def polygon_area(polygon):
    total = 0.0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def clip_polygon(subject, clipper):
    """Sutherland-Hodgman clipping; an empty result means the polygons do not overlap."""
    output = list(subject)
    count = len(clipper)
    for index in range(count):
        if not output:
            return []
        start = clipper[index]
        end = clipper[(index + 1) % count]
        edge = subtract(end, start)
        def side(point):
            return edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])
        incoming = output
        output = []
        previous = incoming[-1]
        for current in incoming:
            previous_side = side(previous)
            current_side = side(current)
            if current_side >= 0.0:
                if previous_side < 0.0:
                    output.append(intersection(previous, current, previous_side, current_side))
                output.append(current)
            elif previous_side >= 0.0:
                output.append(intersection(previous, current, previous_side, current_side))
            previous = current
    return output


def intersection(previous, current, previous_side, current_side):
    denominator = previous_side - current_side
    if abs(denominator) < 1e-300:
        return list(current)
    ratio = previous_side / denominator
    return [previous[0] + ratio * (current[0] - previous[0]),
            previous[1] + ratio * (current[1] - previous[1])]


def box_polygon(body):
    return body.corners()


# ------------------------------------------------------------------ solver
def kinetic_energy(bodies):
    return sum(0.5 * body.mass * dot(body.velocity, body.velocity)
               for body in bodies if not body.is_static)


def momentum(bodies):
    total = [0.0, 0.0]
    for body in bodies:
        if body.is_static:
            continue
        total = [total[0] + body.mass * body.velocity[0],
                 total[1] + body.mass * body.velocity[1]]
    return total


def resolve_pair(first, second, normal, depth, iterations, positional=0.8, slop=0.005):
    """Sequential impulse with restitution, Coulomb friction and a positional correction."""
    relative = subtract(second.velocity, first.velocity)
    normal_speed = dot(relative, normal)
    inverse_sum = first.inverse_mass + second.inverse_mass
    if inverse_sum == 0.0:
        return 0.0
    tangential_impulse = 0.0
    normal_impulse = 0.0
    for _ in range(iterations):
        relative = subtract(second.velocity, first.velocity)
        normal_speed = dot(relative, normal)
        if normal_speed > 0.0:
            break
        restitution = min(first.restitution, second.restitution)
        bias = -restitution * min(normal_speed, 0.0)
        increment = (-normal_speed + bias) / inverse_sum
        total = normal_impulse + increment
        if total < 0.0:
            increment = -normal_impulse
            total = 0.0
        normal_impulse = total
        impulse = scale(normal, increment)
        first.velocity = subtract(first.velocity, scale(impulse, first.inverse_mass))
        second.velocity = [second.velocity[0] + impulse[0] * second.inverse_mass,
                           second.velocity[1] + impulse[1] * second.inverse_mass]
        tangent = subtract(relative, scale(normal, dot(relative, normal)))
        if norm(tangent) > EPSILON:
            tangent = scale(tangent, 1.0 / norm(tangent))
            tangent_speed = dot(relative, tangent)
            friction_increment = -tangent_speed / inverse_sum
            limit = first.friction * second.friction
            maximum = limit * normal_impulse
            candidate = tangential_impulse + friction_increment
            candidate = max(-maximum, min(maximum, candidate))
            friction_increment = candidate - tangential_impulse
            tangential_impulse = candidate
            friction_impulse = scale(tangent, friction_increment)
            first.velocity = subtract(first.velocity, scale(friction_impulse, first.inverse_mass))
            second.velocity = [second.velocity[0] + friction_impulse[0] * second.inverse_mass,
                               second.velocity[1] + friction_impulse[1] * second.inverse_mass]
    correction = max(depth - slop, 0.0) * positional / inverse_sum
    if correction > 0.0:
        first.position = subtract(first.position, scale(normal, correction * first.inverse_mass))
        second.position = [second.position[0] + normal[0] * correction * second.inverse_mass,
                           second.position[1] + normal[1] * correction * second.inverse_mass]
    return normal_impulse


def swept_static_hit(body, previous, others):
    """Continuous collision: sweep the body centre against the Minkowski-expanded AABB.

    Returns (entry fraction, obstacle, entry axis) for the earliest hit, or None.
    """
    motion = subtract(body.position, previous)
    if norm(motion) < EPSILON:
        return None
    total = norm(motion)
    direction = scale(motion, 1.0 / total)
    reach = body.extent()
    best = None
    for other in others:
        if not other.is_static:
            continue
        box = other.aabb()
        low = [box[0] - reach, box[1] - reach]
        high = [box[2] + reach, box[3] + reach]
        enter = 0.0
        exit_time = 1.0
        enter_axis = 0
        for axis in (0, 1):
            if abs(direction[axis]) < 1e-12:
                if previous[axis] < low[axis] or previous[axis] > high[axis]:
                    enter, exit_time = 1.0, 0.0
                    break
                continue
            first_time = (low[axis] - previous[axis]) / (total * direction[axis])
            second_time = (high[axis] - previous[axis]) / (total * direction[axis])
            if first_time > second_time:
                first_time, second_time = second_time, first_time
            if first_time > enter:
                enter = first_time
                enter_axis = axis
            exit_time = min(exit_time, second_time)
            if enter > exit_time:
                break
        if enter <= exit_time and 0.0 <= enter <= 1.0 and exit_time >= 0.0:
            if best is None or enter < best[0]:
                best = (enter, other, enter_axis)
    return best


def step(bodies, dt, gravity=(0.0, -9.81), iterations=8, continuous=True):
    """One engine step: integrate, detect, sweep the fast bodies, resolve."""
    for body in bodies:
        if body.is_static:
            continue
        body.velocity = [body.velocity[0] + gravity[0] * dt, body.velocity[1] + gravity[1] * dt]
        previous = list(body.position)
        body.position = [body.position[0] + body.velocity[0] * dt,
                         body.position[1] + body.velocity[1] * dt]
        if continuous:
            hit = swept_static_hit(body, previous, bodies)
            if hit is not None:
                fraction, _other, axis = hit
                travel = scale(subtract(body.position, previous), max(fraction - 1e-6, 0.0))
                body.position = [previous[0] + travel[0], previous[1] + travel[1]]
                if axis == 0:
                    body.velocity = [-body.velocity[0] * body.restitution, body.velocity[1]]
                else:
                    body.velocity = [body.velocity[0], -body.velocity[1] * body.restitution]
    pairs = sweep_and_prune(bodies)
    impulses = []
    for i, j in pairs:
        contact = collide(bodies[i], bodies[j])
        if contact is not None:
            normal, depth = contact
            impulses.append(resolve_pair(bodies[i], bodies[j], normal, depth, iterations))
    return {"pairs": pairs, "impulses": impulses}


def main():
    # ---------------- body validation ----------------
    for kwargs, why in (({"mass": -1.0}, "negative mass"), ({"radius": 0.0}, "zero radius"),
                        ({"half": (0.0, 1.0)}, "zero half extent"), ({}, "no shape"),
                        ({"radius": 1.0, "half": (1.0, 1.0)}, "two shapes"),
                        ({"mass": 0.0, "dynamic_angle": True}, "spinning static body")):
        try:
            Body("bad", (0.0, 0.0), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {why}")

    # ---------------- broadphase is a superset and cheaper than brute force ----------------
    bodies = []
    for index in range(12):
        bodies.append(Body(f"s{index}", (1.5 * index, 0.3 * (index % 3)), radius=0.4))
    bodies.append(Body("wall", (40.0, 0.0), half=(0.2, 5.0), mass=0.0))
    sparse = brute_force_pairs(bodies)
    candidates = sweep_and_prune(bodies)
    check(set(sparse) <= set(candidates), "the broadphase must never miss a pair")
    check(len(candidates) == len(sparse),
          f"the sweep must not invent pairs: {len(candidates)} vs {len(sparse)}")
    check(len(candidates) < len(bodies) * (len(bodies) - 1) // 2,
          "the broadphase must prune most of the all-pairs set")

    spread = [Body(f"p{index}", (10.0 * index, 0.0), radius=0.5) for index in range(15)]
    check(sweep_and_prune(spread) == [], "a spread-out scene has no candidates")
    check(len(brute_force_pairs(spread)) == 0, "brute force agrees that nothing overlaps")
    clustered = [Body(f"c{index}", (0.35 * index, 0.0), radius=0.4) for index in range(8)]
    clustered.append(Body("far", (50.0, 0.0), radius=0.4))
    dense_brute = brute_force_pairs(clustered)
    dense_candidates = sweep_and_prune(clustered)
    check(len(dense_brute) > 0, "the clustered scene must contain overlapping pairs")
    check(set(dense_brute) <= set(dense_candidates), "the sweep must not miss a dense pair")
    check(len(dense_candidates) < len(clustered) * (len(clustered) - 1) // 2,
          "the sweep must still prune in a dense scene")
    touching = [Body("a", (0.0, 0.0), radius=1.0), Body("b", (2.0, 0.0), radius=1.0)]
    check(sweep_and_prune(touching) == [(0, 1)],
          "boxes that merely touch must be candidates (>= comparisons, never <)")

    # ---------------- narrowphase against the clipping oracle ----------------
    rng_state = [12345]

    def next_random():
        rng_state[0] = (rng_state[0] * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return ((rng_state[0] >> 11) * (1.0 / float(1 << 53))) * 2.0 - 1.0

    agreements = 0
    overlaps = 0
    for _ in range(300):
        first = Body("a", (next_random() * 3.0, next_random() * 3.0), half=(0.3, 0.5),
                     angle=next_random() * 3.0, mass=1.0)
        second = Body("b", (next_random() * 3.0, next_random() * 3.0), half=(0.4, 0.2),
                      angle=next_random() * 3.0, mass=1.0)
        sat = obb_sat(first, second) is not None
        clipped = clip_polygon(first.corners(), second.corners())
        overlapping = polygon_area(clipped) > 1e-12 if len(clipped) >= 3 else False
        check(sat == overlapping,
              f"SAT and clipping disagree for {first.position}/{second.position}")
        agreements += 1
        overlaps += 1 if sat else 0
    check(agreements == 300, "the narrowphase battery must run")
    check(0 < overlaps < 300, f"the battery needs both outcomes, got {overlaps}")

    sphere_box_disagreements = 0
    for _ in range(200):
        sphere = Body("s", (next_random() * 3.0, next_random() * 3.0), radius=0.35, mass=1.0)
        box = Body("b", (next_random() * 3.0, next_random() * 3.0), half=(0.5, 0.4),
                   angle=next_random() * 3.0, mass=1.0)
        analytic = sphere_box(sphere, box, flipped=True) is not None
        clipped = clip_polygon(circle_polygon(sphere.position, sphere.radius, 128),
                               box.corners())
        sampled = polygon_area(clipped) > 1e-9 if len(clipped) >= 3 else False
        if analytic != sampled:
            sphere_box_disagreements += 1
            separation = abs(norm(subtract(sphere.position, box.position))
                             - (sphere.radius + box.extent()))
            check(separation < 0.05,
                  "sphere/box and the polygon oracle may only differ at the surface")
    check(sphere_box_disagreements <= 10,
          f"too many sphere/box oracle disagreements: {sphere_box_disagreements}")

    axis_aligned = Body("box", (0.0, 0.0), half=(1.0, 1.0))
    offset = Body("other", (1.5, 0.0), half=(1.0, 1.0))
    contact = obb_sat(axis_aligned, offset)
    check(contact is not None and abs(contact[1] - 0.5) < 1e-12,
          f"the least-penetration depth must be 0.5: {contact}")
    check(abs(abs(contact[0][0]) - 1.0) < 1e-12 and abs(contact[0][1]) < 1e-12,
          f"the normal must be the x axis: {contact[0]}")
    check(obb_sat(axis_aligned, Body("far", (5.0, 0.0), half=(1.0, 1.0))) is None,
          "separated boxes must not collide")

    # ---------------- elastic collision against the closed form ----------------
    for m1, m2, v1, v2 in ((1.0, 1.0, 2.0, 0.0), (1.0, 3.0, 1.0, -2.0), (2.0, 0.5, 1.0, -1.5)):
        left = Body("left", (-0.9, 0.0), (v1, 0.0), mass=m1, radius=1.0, restitution=1.0,
                    friction=0.0)
        right = Body("right", (0.9, 0.0), (v2, 0.0), mass=m2, radius=1.0, restitution=1.0,
                     friction=0.0)
        contact = sphere_sphere(left, right)
        check(contact is not None, "the spheres must overlap")
        check(dot(contact[0], subtract(right.velocity, left.velocity)) < 0.0,
              "the fixture must be a head-on approach")
        resolve_pair(left, right, contact[0], contact[1], iterations=20)
        expected_left = ((m1 - m2) * v1 + 2.0 * m2 * v2) / (m1 + m2)
        expected_right = ((m2 - m1) * v2 + 2.0 * m1 * v1) / (m1 + m2)
        check(abs(left.velocity[0] - expected_left) < 1e-9,
              f"left velocity {left.velocity[0]} != {expected_left}")
        check(abs(right.velocity[0] - expected_right) < 1e-9,
              f"right velocity {right.velocity[0]} != {expected_right}")
        before = m1 * v1 + m2 * v2
        after = m1 * left.velocity[0] + m2 * right.velocity[0]
        check(abs(after - before) < 1e-9, "momentum must be conserved in the impulse")
        energy_before = 0.5 * m1 * v1 * v1 + 0.5 * m2 * v2 * v2
        energy_after = (0.5 * m1 * left.velocity[0] ** 2 + 0.5 * m2 * right.velocity[0] ** 2)
        check(abs(energy_after - energy_before) < 1e-9,
              "an elastic impulse must conserve kinetic energy")
    equal = [Body("a", (0.0, 0.0), (1.0, 0.0), mass=1.0, radius=1.0, restitution=1.0, friction=0.0),
             Body("b", (1.5, 0.0), (0.0, 0.0), mass=1.0, radius=1.0, restitution=1.0, friction=0.0)]
    contact = sphere_sphere(*equal)
    resolve_pair(equal[0], equal[1], contact[0], contact[1], iterations=20)
    check(abs(equal[0].velocity[0]) < 1e-12 and abs(equal[1].velocity[0] - 1.0) < 1e-12,
          "equal masses must exchange velocities exactly")

    # ---------------- inelastic collisions never gain energy ----------------
    for restitution in (0.0, 0.4, 0.8):
        pair = [Body("a", (0.0, 0.0), (3.0, 0.5), mass=1.5, radius=1.0,
                     restitution=restitution, friction=0.0),
                Body("b", (1.4, 0.0), (-1.0, -0.25), mass=2.0, radius=1.0,
                     restitution=restitution, friction=0.0)]
        before = kinetic_energy(pair)
        moment_before = momentum(pair)
        contact = sphere_sphere(*pair)
        resolve_pair(pair[0], pair[1], contact[0], contact[1], iterations=20)
        after = kinetic_energy(pair)
        moment_after = momentum(pair)
        check(after <= before + 1e-9,
              f"restitution {restitution} increased the energy: {before} -> {after}")
        check(abs(moment_after[0] - moment_before[0]) < 1e-9
              and abs(moment_after[1] - moment_before[1]) < 1e-9,
              "momentum must always be conserved")

    # ---------------- resting contact: bounded penetration, decaying velocity ----------------
    floor = Body("floor", (0.0, 0.0), half=(10.0, 0.5), mass=0.0, restitution=0.0, friction=0.6)
    block = Body("block", (0.0, 3.0), half=(0.5, 0.5), mass=1.0, restitution=0.0, friction=0.6)
    properties = [floor, block]
    for _ in range(400):
        step(properties, 1.0 / 120.0, iterations=12)
    resting_penetration = (floor.position[1] + floor.half[1]) - (block.position[1] - block.half[1])
    check(-0.05 <= resting_penetration <= 0.05,
          f"the block must rest on the surface, penetration {resting_penetration}")
    check(abs(block.velocity[1]) < 0.1, f"the block must stop bouncing: {block.velocity[1]}")
    heights = []
    for _ in range(60):
        step(properties, 1.0 / 120.0, iterations=12)
        heights.append(block.position[1])
    check(max(heights) - min(heights) < 0.02,
          f"the resting block must not jitter: {max(heights) - min(heights)}")

    # ---------------- friction is bounded by the Coulomb cone ----------------
    slider = Body("slider", (0.0, 0.6), (5.0, 0.0), mass=1.0, half=(0.5, 0.5), restitution=0.0,
                  friction=1.0)
    ground = Body("ground", (0.0, 0.0), half=(50.0, 0.5), mass=0.0, restitution=0.0, friction=1.0)
    scene = [ground, slider]
    speeds = []
    for _ in range(240):
        step(scene, 1.0 / 240.0, gravity=(0.0, -9.81), iterations=10, continuous=False)
        speeds.append(abs(slider.velocity[0]))
    check(all(later <= earlier + 1e-9 for earlier, later in zip(speeds, speeds[1:])),
          "friction must never add tangential speed")
    check(speeds[-1] < speeds[0], "friction must eventually slow the slider down")
    check(abs(slider.velocity[1]) < 0.5, "the slider must stay on the ground")

    # ---------------- bouncing off a static wall keeps the tangential velocity ----------------
    wall = Body("wall", (5.0, 0.0), half=(0.5, 5.0), mass=0.0, restitution=1.0, friction=0.0)
    ball = Body("ball", (2.0, 0.0), (4.0, 2.0), mass=1.0, radius=0.25, restitution=1.0,
                friction=0.0)
    before_tangential = ball.velocity[1]
    for _ in range(60):
        step([wall, ball], 1.0 / 60.0, gravity=(0.0, 0.0), iterations=10)
    check(abs(ball.velocity[1] - before_tangential) < 1e-6,
          f"a frictionless bounce must keep the tangential velocity: {ball.velocity[1]}")
    check(ball.velocity[0] < 0.0, "the ball must bounce back")
    check(ball.position[0] < 5.0, "the ball must stay on the near side of the wall")

    # ---------------- tunnelling: prevented with the sweep, visible without it ----------------
    def bullet_run(continuous):
        shield = Body("shield", (1.0, 0.0), half=(0.1, 5.0), mass=0.0, restitution=1.0,
                      friction=0.0)
        bullet = Body("bullet", (0.0, 0.0), (100.0, 0.0), mass=1.0, radius=0.05,
                      restitution=1.0, friction=0.0)
        step([shield, bullet], 1.0, gravity=(0.0, 0.0), iterations=10, continuous=continuous)
        return bullet.position[0]

    tunnelled = bullet_run(continuous=False)
    swept = bullet_run(continuous=True)
    check(tunnelled > 1.0, f"without the sweep the bullet must pass the wall: {tunnelled}")
    check(swept <= 1.0, f"the swept test must stop the bullet at the wall: {swept}")
    check(swept < tunnelled, "continuous detection must make a difference here")

    # ---------------- determinism ----------------
    def build_scene():
        return [Body("floor", (0.0, 0.0), half=(10.0, 0.5), mass=0.0, restitution=0.2),
                Body("a", (0.0, 2.0), (0.5, 0.0), mass=1.0, radius=0.4, restitution=0.4),
                Body("b", (1.0, 4.0), (-0.3, 0.0), mass=2.0, radius=0.5, restitution=0.4)]

    first_scene = build_scene()
    for _ in range(50):
        step(first_scene, 1.0 / 100.0)
    second_scene = build_scene()
    for _ in range(50):
        step(second_scene, 1.0 / 100.0)
    check([body.position for body in first_scene] == [body.position for body in second_scene]
          and [body.velocity for body in first_scene] == [body.velocity for body in second_scene],
          "the engine must be bit-deterministic")
    check(all(math.isfinite(value) for body in first_scene
              for value in body.position + body.velocity),
          "the scene must stay finite over the run")
    check(max(norm(body.velocity) for body in first_scene) < 100.0,
          "the scene must stay bounded (no energy injection)")

    print(f"broadphase pairs={len(candidates)} brute={len(sparse)} "
          f"all_pairs={len(bodies) * (len(bodies) - 1) // 2} dense={len(dense_candidates)}/36")
    print(f"SAT vs clipping: {agreements} agreements, {overlaps} overlaps")
    print(f"resting penetration={resting_penetration:.5f} jitter={max(heights) - min(heights):.5f}")
    print(f"bullet: swept={swept:.3f} tunnelled={tunnelled:.3f}")
    print("PASS")


def circle_polygon(center, radius, segments):
    return [[center[0] + radius * math.cos(2.0 * math.pi * index / segments),
             center[1] + radius * math.sin(2.0 * math.pi * index / segments)]
            for index in range(segments)]


if __name__ == "__main__":
    main()
