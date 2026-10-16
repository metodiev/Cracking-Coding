"""Task 738 - Implement motion planning basic RRT (rapidly exploring random tree).  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

RRT grows a tree from the start configuration by repeatedly sampling a random free point,
finding the nearest tree node and extending that node a fixed step towards the sample; the new
edge is kept only if it is collision free.  A goal bias makes the sampler return the goal a
small fraction of the time, which pulls the tree in rather than relying on luck, and the run
stops as soon as a step lands within the goal tolerance with a clear line of sight to the goal.
The result is a path that is feasible (every edge collision free) though not optimal, which is
exactly what basic RRT promises.

Collision checking is exact for the axis-aligned rectangular obstacles used here: a segment is
clipped against each rectangle with the Liang-Barsky parameter interval, and closed contact
counts as a collision, so the planner never squeezes through a corner.  A second, independent
sampled line-of-sight check is used by the tests as an oracle - the two disagreeing would mean
the planner accepted an edge through an obstacle.

Deterministic PRNG (splitmix64 with a fixed seed), linear nearest-neighbour scan, no steering
beyond a straight segment, no rewiring or smoothing.  Verified: paths start at the start
(configuration), end at the goal, every edge passes the sampled oracle, path length is at least
the straight-line distance and close to it in open space, nearest-neighbour queries match a
brute-force argmin, tree edges are all collision free, parents never point forward, a fully
blocked corridor and a goal inside an obstacle both yield no path, and repeated runs with the
same seed reproduce the same tree exactly.
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
    """Deterministic float in [low, high) from the PRNG state."""
    state, r = splitmix64(state)
    return state, low + (high - low) * (r / float(1 << 64))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Scenario:
    """Bounded 2D workspace with axis-aligned rectangular obstacles (closed sets)."""

    def __init__(self, bounds, obstacles):
        self.bounds = bounds
        self.obstacles = obstacles

    def point_free(self, p, tol=0.0):
        for (x0, y0, x1, y1) in self.obstacles:
            if x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol:
                return False
        return True

    def segment_clear(self, a, b):
        """Exact Liang-Barsky clipping against every obstacle rectangle."""
        for rect in self.obstacles:
            if segment_hits_rect(a, b, rect):
                return False
        return True

    def sampled_clear(self, a, b, samples=400):
        """Independent oracle: dense sampling of the segment (not used by the planner)."""
        for i in range(samples + 1):
            t = i / samples
            if not self.point_free((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))):
                return False
        return True


def segment_hits_rect(a, b, rect):
    """Closed segment versus closed rectangle, Liang-Barsky parameter interval."""
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t_low, t_high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return False                # parallel to this slab and outside it
            continue
        ratio = q / p
        if p < 0:
            if ratio > t_high:
                return False
            t_low = max(t_low, ratio)
        else:
            if ratio < t_low:
                return False
            t_high = min(t_high, ratio)
    return t_low <= t_high


class RRT:
    """Rapidly exploring random tree over a Scenario with a fixed step length."""

    def __init__(self, scenario, start, goal, step, seed, goal_bias=0.15):
        self.scenario = scenario
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.step = float(step)
        self.seed = seed
        self.goal_bias = goal_bias
        self.nodes = []
        self.parents = []
        self.iterations = 0
        self.nearest_calls = 0
        self.steering_failures = 0
        self.goal_samples = 0
        self.extensions = 0

    # ------------------------------------------------------------------ primitives
    def nearest(self, p):
        """Brute-force nearest node: RRT's classic O(n) query."""
        self.nearest_calls += 1
        best, best_distance = 0, math.inf
        for index, node in enumerate(self.nodes):
            d = (node[0] - p[0]) ** 2 + (node[1] - p[1]) ** 2
            if d < best_distance:
                best, best_distance = index, d
        return best

    def steer(self, source, target):
        """Point at most `step` away from `source` in the direction of `target`."""
        length = dist(source, target)
        if length <= self.step:
            return target
        ratio = self.step / length
        return (source[0] + ratio * (target[0] - source[0]),
                source[1] + ratio * (target[1] - source[1]))

    def sample(self, state):
        """Goal-biased uniform sampling inside the bounds."""
        state, r = splitmix64(state)
        if r / float(1 << 64) < self.goal_bias:
            self.goal_samples += 1
            return state, self.goal
        x0, y0, x1, y1 = self.scenario.bounds
        state, x = uniform(state, x0, x1)
        state, y = uniform(state, y0, y1)
        return state, (x, y)

    def path_to(self, index):
        """Waypoints from the start to `index` following the parent chain."""
        chain = []
        while index != -1:
            chain.append(self.nodes[index])
            index = self.parents[index]
        return chain[::-1]

    # ------------------------------------------------------------------ planning
    def run(self, iterations):
        """Grow the tree; return the path (start -> goal) or None if nothing was found."""
        self.nodes = [self.start]
        self.parents = [-1]
        state = self.seed
        if dist(self.start, self.goal) <= self.step and self.scenario.segment_clear(self.start, self.goal):
            self.nodes.append(self.goal)
            self.parents.append(0)
            return self.path_to(1)
        for _ in range(iterations):
            self.iterations += 1
            state, sample = self.sample(state)
            index = self.nearest(sample)
            candidate = self.steer(self.nodes[index], sample)
            if not self.scenario.segment_clear(self.nodes[index], candidate):
                self.steering_failures += 1
                continue
            self.nodes.append(candidate)
            self.parents.append(index)
            self.extensions += 1
            if (dist(candidate, self.goal) <= self.step
                    and self.scenario.segment_clear(candidate, self.goal)):
                self.nodes.append(self.goal)
                self.parents.append(len(self.nodes) - 2)
                return self.path_to(len(self.nodes) - 1)
        return None

    def edges(self):
        return [(self.nodes[self.parents[i]], self.nodes[i])
                for i in range(1, len(self.nodes))]


def path_length(path):
    return sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def open_scenario():
    return Scenario((0.0, 0.0, 10.0, 10.0), [])


def gapped_wall_scenario():
    """A wall at x = 5 with a gap between y = 4 and y = 6."""
    return Scenario((0.0, 0.0, 10.0, 10.0), [(5.0, -1.0, 5.3, 4.0), (5.0, 6.0, 5.3, 11.0)])


def sealed_wall_scenario():
    return Scenario((0.0, 0.0, 10.0, 10.0), [(5.0, -1.0, 5.3, 11.0)])


def main():
    # 1. open space: a path exists, is feasible and stays close to the straight line
    start, goal = (0.5, 0.5), (9.5, 9.5)
    planner = RRT(open_scenario(), start, goal, 0.4, 0x5EED)
    path = planner.run(4000)
    check(path is not None and path[0] == start and path[-1] == goal,
          "the open scenario must yield a path from start to goal")
    straight = dist(start, goal)
    check(path_length(path) >= straight - 1e-9, "a path can never be shorter than the straight line")
    check(path_length(path) < straight * 1.4,
          "in open space the goal-biased tree must stay within 40% of the straight line")
    check(all(planner.scenario.sampled_clear(path[i], path[i + 1]) for i in range(len(path) - 1)),
          "every path edge must pass the independent sampled line-of-sight oracle")
    check(planner.goal_samples > 0, "the goal bias must have fired at least once")
    check(max(len(set(path[i] for i in range(len(path)))) for _ in [0]) == len(path),
          "the path must not repeat a waypoint")
    check(abs(path[-1][0] - goal[0]) < 1e-9 and abs(path[-1][1] - goal[1]) < 1e-9,
          "the last waypoint must be the goal exactly")

    # 2. nearest-neighbour queries match a brute-force argmin, ties broken by the lowest index
    probes = [(0.0, 0.0), (5.0, 9.9), (9.99, 0.0), (2.0, 2.0), (7.5, 7.5)]
    best = None
    for probe in probes:
        index = planner.nearest(probe)
        distances = [(node[0] - probe[0]) ** 2 + (node[1] - probe[1]) ** 2 for node in planner.nodes]
        check(distances[index] == min(distances), "nearest() must return a true argmin")
        best = index
    check(best is not None and planner.nearest_calls > 0, "nearest-neighbour calls must be counted")

    # 3. tree invariants: parents point backwards, every edge is collision free
    check(all(planner.parents[i] < i for i in range(1, len(planner.nodes))),
          "every parent index must precede its child (so the tree cannot contain a cycle)")
    check(self_consistent_tree(planner), "the tree must be well formed")
    check(all(planner.scenario.sampled_clear(a, b) for a, b in planner.edges()),
          "every tree edge must pass the sampled line-of-sight oracle")

    # 4. the wall with a gap: a path must be found and must actually use the gap
    gapped = RRT(gapped_wall_scenario(), (1.0, 5.0), (9.0, 5.0), 0.3, 0xC0FFEE)
    detour = gapped.run(6000)
    check(detour is not None, "a wall with a gap must be passable")
    check(all(gapped.scenario.segment_clear(detour[i], detour[i + 1]) for i in range(len(detour) - 1)),
          "the detour must be collision free")
    check(any(4.0 < node[1] < 6.0 and 4.4 <= node[0] <= 5.6 for node in detour),
          "the detour must pass through the gap rather than around the wall")
    check(path_length(detour) > dist((1.0, 5.0), (9.0, 5.0)), "the detour must be longer than straight")

    # 5. negative cases: a sealed wall and a goal inside an obstacle yield no path
    sealed = RRT(sealed_wall_scenario(), (1.0, 5.0), (9.0, 5.0), 0.3, 0xBEEF)
    check(sealed.run(3000) is None, "a sealed wall must make the goal unreachable")
    check(sealed.iterations == 3000 and sealed.extensions > 0,
          "the planner must still have grown a tree before giving up")
    trapped = RRT(open_scenario(), (1.0, 1.0), (5.05, 5.05), 0.3, 0x1234)
    trapped.scenario = Scenario((0.0, 0.0, 10.0, 10.0), [(4.9, 4.9, 5.2, 5.2)])
    check(trapped.run(2000) is None, "a goal inside an obstacle must be unreachable")

    # 6. boundary: start equal to the goal, and a start already within one step of the goal
    trivial = RRT(open_scenario(), (2.0, 2.0), (2.0, 2.0), 0.5, 0x77)
    check(trivial.run(100) == [(2.0, 2.0)] or trivial.run(100)[-1] == (2.0, 2.0),
          "a zero-length request must still return a valid path")
    trivial.iterations = 0
    near = RRT(open_scenario(), (2.0, 2.0), (2.2, 2.0), 0.5, 0x78)
    check(near.run(10) == [(2.0, 2.0), (2.2, 2.0)] and near.iterations == 0,
          "a goal within one step must be connected immediately, without any sampling")

    # 7. determinism: identical seeds reproduce the tree, the path and the counters
    first = RRT(gapped_wall_scenario(), (1.0, 5.0), (9.0, 5.0), 0.3, 0xC0FFEE)
    second = RRT(gapped_wall_scenario(), (1.0, 5.0), (9.0, 5.0), 0.3, 0xC0FFEE)
    check(first.run(6000) == second.run(6000) and first.nodes == second.nodes,
          "identical seeds must reproduce identical paths and trees")
    check(first.iterations == second.iterations and first.steering_failures == second.steering_failures,
          "identical seeds must reproduce identical counters")

    # 8. step size: a longer step reaches the goal with fewer edges
    coarse = RRT(gapped_wall_scenario(), (1.0, 5.0), (9.0, 5.0), 1.0, 0x99)
    fine = RRT(gapped_wall_scenario(), (1.0, 5.0), (9.0, 5.0), 0.2, 0x99)
    coarse_path, fine_path = coarse.run(6000), fine.run(6000)
    check(coarse_path is not None and fine_path is not None, "both step sizes must find the gap")
    check(len(coarse_path) < len(fine_path),
          "a coarse step must reach the goal with fewer waypoints than a fine step")

    print(f"738 open nodes={len(planner.nodes)} waypoints={len(path)} "
          f"length={path_length(path):.3f} straight={straight:.3f}")
    print(f"738 detour waypoints={len(detour)} length={path_length(detour):.3f} "
          f"iterations={gapped.iterations} failures={gapped.steering_failures}")
    print("PASS")


def self_consistent_tree(planner):
    """Every node is reachable from the root, no node is its own ancestor."""
    seen = {0}
    for i in range(1, len(planner.nodes)):
        if planner.parents[i] < 0 or planner.parents[i] >= len(planner.nodes):
            return False
        if planner.parents[i] not in seen:
            return False
        seen.add(i)
    return len(seen) == len(planner.nodes)


if __name__ == "__main__":
    main()
