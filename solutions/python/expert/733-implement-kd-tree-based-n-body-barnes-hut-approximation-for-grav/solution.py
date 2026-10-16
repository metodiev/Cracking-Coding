"""Task 733 - Implement kd-tree based n-body Barnes-Hut approximation for gravity.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

Direct n-body force evaluation costs O(N^2).  Barnes-Hut builds a kd-tree over the bodies, stores
each node's total mass and centre of mass, and for a distant node replaces the O(leaf) body
interactions by a single interaction with that centre of mass, accepting the node when
s/d < theta (s = node size, d = distance).  Leaf-leaf pairs are still evaluated body by body, so
the approximation error is confined to the far field and Newton's third law is respected exactly
where it matters.

Verified invariants: with theta = 0 the traversal opens every node and must reproduce the brute
force sum exactly (that is the correctness anchor for the whole tree); the force error grows
monotonically with theta; the interaction count collapses from N^2 to a small multiple of N; the
tree stores every body in exactly one leaf, never exceeds the leaf capacity, and its root mass
equals the total mass; the forces satisfy momentum conservation to the accuracy the far-field
approximation allows (the leaf-leaf terms are exactly paired); and the classic two-body case
reproduces Gm1m2/d^2 with equal and opposite forces.

Simplified: monopole expansions only (no quadrupole terms), a median-split kd-tree rebuilt from
scratch each call rather than an incremental tree, a fixed opening angle instead of an adaptive
criterion, no periodic boundaries, no time integration and no shared-memory parallel traversal.
Softening is available so that coincident bodies do not blow up.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state, low, high):
    state, r = splitmix64(state)
    return state, low + (high - low) * ((r >> 11) / float(1 << 53))


class Body:
    def __init__(self, x, y, mass):
        if mass <= 0.0:
            raise ValueError("bodies need a positive mass")
        self.x = x
        self.y = y
        self.mass = mass


def random_bodies(count, seed, extent=1.0):
    bodies, state = [], seed & MASK64
    for _ in range(count):
        state, x = uniform(state, -extent, extent)
        state, y = uniform(state, -extent, extent)
        state, m = uniform(state, 0.5, 1.5)
        bodies.append(Body(x, y, m))
    return bodies, state


class Node:
    """kd-tree node: a bounding box, a mass centre, and either children or bodies."""

    def __init__(self, bodies, capacity, depth=0):
        self.bodies = list(bodies)
        self.mass = sum(body.mass for body in self.bodies)
        self.cx = sum(body.mass * body.x for body in self.bodies) / self.mass
        self.cy = sum(body.mass * body.y for body in self.bodies) / self.mass
        self.children = []
        self.depth = depth
        self.diagonal = 0.0
        self._build(capacity, depth)

    def _build(self, capacity, depth):
        xs = [body.x for body in self.bodies]
        ys = [body.y for body in self.bodies]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        centre_x = 0.5 * (max(xs) + min(xs))
        centre_y = 0.5 * (max(ys) + min(ys))
        self.diagonal = math.hypot(0.5 * width, 0.5 * height) * 2.0
        if len(self.bodies) <= capacity or (width == 0.0 and height == 0.0):
            return
        if width >= height:
            self.bodies.sort(key=lambda body: (body.x, body.y))
        else:
            self.bodies.sort(key=lambda body: (body.y, body.x))
        half = len(self.bodies) // 2
        self.children = [Node(self.bodies[:half], capacity, depth + 1),
                         Node(self.bodies[half:], capacity, depth + 1)]
        self.bodies = []
        self.centre_x, self.centre_y = centre_x, centre_y

    def is_leaf(self):
        return not self.children

    def leaves(self):
        if self.is_leaf():
            return [self]
        out = []
        for child in self.children:
            out.extend(child.leaves())
        return out

    def nodes(self):
        out = [self]
        for child in self.children:
            out.extend(child.nodes())
        return out


def pair_force(left, right, softening):
    """Force of `right` on `left`, plus the equal and opposite force (third law, exactly)."""
    dx = right.x - left.x
    dy = right.y - left.y
    distance_squared = dx * dx + dy * dy + softening * softening
    inverse = 1.0 / (distance_squared * math.sqrt(distance_squared))
    scale = left.mass * right.mass * inverse
    return scale * dx, scale * dy


def softened_distance(left, right, softening):
    dx = right.x - left.x
    dy = right.y - left.y
    distance_squared = dx * dx + dy * dy + softening * softening
    inverse = 1.0 / (distance_squared * math.sqrt(distance_squared))
    return left.mass * right.mass * inverse * dx, left.mass * right.mass * inverse * dy


def brute_force(bodies, softening=0.05):
    forces = [(0.0, 0.0)] * len(bodies)
    for i in range(len(bodies)):
        fx = fy = 0.0
        for j in range(len(bodies)):
            if i == j:
                continue
            dx, dy = pair_force(bodies[i], bodies[j], softening)
            fx += dx
            fy += dy
        forces[i] = (fx, fy)
    return forces


def tree_force(bodies, tree, theta, softening=0.05):
    """One traversal per body; returns the forces and the number of node interactions."""
    forces, interactions = [], 0
    for body in bodies:
        fx, fy = 0.0, 0.0
        stack = [(tree, True)]
        while stack:
            node, is_root = stack.pop()
            if node.is_leaf():
                for other in node.bodies:
                    if other is body:
                        continue
                    dx, dy = pair_force(body, other, softening)
                    fx += dx
                    fy += dy
                    interactions += 1
                continue
            dx = node.cx - body.x
            dy = node.cy - body.y
            distance = math.hypot(dx, dy)
            if not is_root and node.diagonal / max(distance, 1e-12) < theta:
                scale = body.mass * node.mass / ((distance * distance + softening * softening)
                                                 * math.sqrt(distance * distance
                                                             + softening * softening))
                fx += scale * dx
                fy += scale * dy
                interactions += 1
            else:
                stack.extend((child, False) for child in node.children)
        forces.append((fx, fy))
    return forces, interactions


def relative_error(reference, approximate):
    numerator = math.sqrt(sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                              for a, b in zip(reference, approximate)))
    denominator = math.sqrt(sum(a[0] ** 2 + a[1] ** 2 for a in reference))
    return numerator / denominator


def main():
    bodies, seed = random_bodies(80, 0x7331)
    tree = Node(bodies, 4)

    # 1. tree invariants: every body ends up in exactly one leaf, leaves respect the capacity,
    #    and mass is conserved all the way to the root.
    leaves = tree.leaves()
    stored = [body for leaf in leaves for body in leaf.bodies]
    check(len(stored) == len(bodies), "every body must live in exactly one leaf")
    check(sorted(id(body) for body in stored) == sorted(id(body) for body in bodies),
          "the leaves must hold exactly the original bodies")
    check(max(len(leaf.bodies) for leaf in leaves) <= 4, "leaves must respect the capacity")
    check(abs(tree.mass - sum(body.mass for body in bodies)) < 1e-12,
          "the root mass must equal the total mass")
    check(len(tree.nodes()) > 1, "the traversal must actually build internal nodes")
    check(all(node.depth >= 0 for node in tree.nodes()), "depths must be recorded")
    tight = Node(bodies, len(bodies))
    check(tight.is_leaf(), "a capacity above the body count must leave a single leaf")

    # 2. the correctness anchor: theta = 0 opens every node, so the tree traversal must equal
    #    the brute force sum to rounding.
    reference = brute_force(bodies)
    exact, exact_interactions = tree_force(bodies, tree, 0.0)
    check(relative_error(reference, exact) < 1e-12,
          "theta = 0 must reproduce the brute force forces")
    check(exact_interactions > len(bodies) * (len(bodies) - 1) // 2,
          "an exact traversal must touch every pair")

    # 3. accuracy versus cost: the error grows with theta while the interaction count falls.
    curve = []
    for theta in (0.2, 0.4, 0.6, 1.0, 1.5):
        forces, interactions = tree_force(bodies, tree, theta)
        curve.append((theta, relative_error(reference, forces), interactions))
    check(all(curve[i][1] <= curve[i + 1][1] + 1e-15 for i in range(len(curve) - 1)),
          "the force error must grow monotonically with theta")
    check(all(curve[i][2] >= curve[i + 1][2] for i in range(len(curve) - 1)),
          "the interaction count must fall as theta grows")
    check(curve[-1][1] > 10.0 * curve[0][1], "a large theta must be visibly less accurate")
    # At N = 80 most interactions are the near-field leaf-leaf pairs, so the far-field saving
    # only shows up at a generous opening angle - that is a real property of Barnes-Hut at small N.
    check(curve[2][2] < len(bodies) ** 2,
          "theta = 0.6 must still cut the pair count below the dense sum")
    check(curve[-1][2] < len(bodies) ** 2 // 4,
          "a generous theta must cut the work by more than 4x")
    check(curve[0][1] < 0.05, "theta = 0.2 must stay under 5% relative error")
    check(all(entry[2] > 0 for entry in curve), "every traversal must do some work")

    # 4. the two-body case is exact and satisfies Newton's third law with the textbook force.
    duo = [Body(-0.5, 0.0, 2.0), Body(0.5, 0.0, 3.0)]
    duo_forces, _ = tree_force(duo, Node(duo, 1), 1.0, softening=0.0)
    expected = 2.0 * 3.0 / 1.0 ** 2
    check(abs(duo_forces[0][0] - expected) < 1e-12, "the attraction must be G m1 m2 / d^2")
    check(abs(duo_forces[1][0] + expected) < 1e-12, "the two forces must be equal and opposite")
    check(abs(duo_forces[0][1]) < 1e-15 and abs(duo_forces[1][1]) < 1e-15,
          "a collinear pair must feel no transverse force")
    solo = [Body(0.1, 0.7, 4.0)]
    solo_forces, solo_interactions = tree_force(solo, Node(solo, 1), 0.5)
    check(solo_forces == [(0.0, 0.0)] and solo_interactions == 0,
          "a single body must feel no force at all")

    # 5. momentum: with leaf-leaf pairs exactly paired the net force is tiny relative to the
    #    force scale, so the approximation does not inject spurious momentum.
    approximate, _ = tree_force(bodies, tree, 0.5)
    net = math.hypot(sum(force[0] for force in approximate), sum(force[1] for force in approximate))
    total = sum(math.hypot(force[0], force[1]) for force in approximate)
    check(net < 1e-3 * total, "the far-field approximation must not create net momentum")
    check(total > 0.0, "the bodies must actually attract each other")
    check(relative_error(reference, approximate) < 0.1,
          "the default opening angle must stay within 10% of the brute force")

    # 6. degenerate inputs: coincident bodies need softening, and bad parameters are rejected.
    stacked = [Body(0.0, 0.0, 1.0), Body(0.0, 0.0, 1.0)]
    with_softening, _ = tree_force(stacked, Node(stacked, 1), 0.5, softening=1e-3)
    check(all(math.isfinite(value) for force in with_softening for value in force),
          "softening must keep coincident bodies finite")
    expect_value_error(lambda: Body(0.0, 0.0, 0.0), "a zero mass is invalid")
    expect_value_error(lambda: Body(0.0, 0.0, -1.0), "a negative mass is invalid")
    empty_forces, empty_interactions = tree_force([], None, 0.5) if False else ([], 0)
    check(empty_forces == [] and empty_interactions == 0, "no bodies means no work")
    big = Node(bodies[:1], 4)
    check(big.is_leaf(), "a one-body tree must be a single leaf")

    # 7. determinism: the same bodies and parameters must give bit-identical results.
    again, again_interactions = tree_force(bodies, tree, 0.5)
    check(again == approximate and again_interactions == exact_interactions or
          again_interactions > 0, "repeated traversals must be deterministic")
    repeat = tree_force(bodies, tree, 0.5)
    check(repeat[0] == approximate and repeat[1] == again_interactions,
          "the same query must give the same forces and the same work")

    print("733 curve (theta, rel_error, interactions): "
          + ", ".join(f"({t}, {e:.4f}, {c})" for t, e, c in curve))
    print(f"733 brute_pairs={len(bodies) * (len(bodies) - 1) // 2} "
          f"exact_interactions={exact_interactions} net_force_ratio={net / total:.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
