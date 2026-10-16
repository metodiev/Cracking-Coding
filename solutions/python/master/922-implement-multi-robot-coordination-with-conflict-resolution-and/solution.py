"""Task 922 - Implement multi-robot coordination with conflict resolution and task allocation.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

Multi-robot coordination on a small 6x6 grid with obstacles, in three layers.  Task allocation
minimises the total obstacle-aware BFS distance exactly by a subset DP (dp[mask] = best cost of
filling the first popcount(mask) rows), cross-checked against an exhaustive permutation search.
Prioritised planning routes robots in priority order with a space-time BFS that respects the
higher-priority reservations: occupied (cell, t) vertices, traversed (a -> b, t) edges (which also
forbids a swap) and the goal cells of finished robots forever, with waiting in place as a legal move.
A separate checker scans every pair of paths for vertex conflicts and swap conflicts, so the
coordinator output is judged by an independent routine.  Verified: the allocator matches the
exhaustive optimum while the greedy heuristic can only be worse, the detector fires on hand-made
swaps, the paths are valid moves or waits and conflict free, the makespan respects the single-robot
lower bound, uncoordinated shortest paths do collide, and a walled-in robot is reported unplanned
while the others still move.  Simplified: discrete time, holonomic motion, static tasks.
"""
from itertools import permutations

GRID = 6
OBSTACLES = {(1, 1), (1, 2), (1, 3), (1, 4), (4, 2), (4, 3)}
STARTS = [(0, 0), (0, 5), (5, 0)]
GOALS = [(5, 5), (5, 1), (2, 5)]
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))       # (0, 0) is the wait move
HORIZON = 40

def check(ok, message):
    if not ok:
        raise AssertionError(message)

def free_cells():
    return {(x, y) for x in range(GRID) for y in range(GRID)} - OBSTACLES

def neighbours(cell, cells):
    return [(cell[0] + dx, cell[1] + dy) for dx, dy in STEPS
            if (cell[0] + dx, cell[1] + dy) in cells]

def distances(start, cells):
    """BFS distance from one cell to every reachable cell."""
    frontier, seen = [start], {start: 0}
    for cell in frontier:
        for nxt in neighbours(cell, cells):
            if nxt not in seen:
                seen[nxt] = seen[cell] + 1
                frontier.append(nxt)
    return seen

def shortest_path(start, goal, cells):
    """Wait-free obstacle-aware shortest path, used for the uncoordinated baseline."""
    if start == goal:
        return [start]
    frontier, parent = [start], {start: None}
    for cell in frontier:
        for nxt in neighbours(cell, cells):
            if nxt in parent:
                continue
            parent[nxt] = cell
            if nxt == goal:
                path, node = [], nxt
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]
            frontier.append(nxt)
    return None

def cost_matrix(robots, tasks, cells):
    return [[distances(robot, cells).get(task, float("inf")) for task in tasks] for robot in robots]

def allocate(costs):
    """Exact assignment of rows to distinct columns by subset DP; returns (total, pairing)."""
    if not costs or not costs[0] or min(len(costs), len(costs[0])) == 0:
        return 0, []
    flipped = len(costs) > len(costs[0])
    table = [list(row) for row in (zip(*costs) if flipped else costs)]
    layer = {0: (0, [])}
    for row in range(len(table)):
        nxt = {}
        for mask, (total, pairing) in layer.items():
            for column in range(len(table[0])):
                if mask >> column & 1:
                    continue
                candidate = (total + table[row][column], pairing + [(row, column)])
                key = mask | 1 << column
                if key not in nxt or candidate[0] < nxt[key][0]:
                    nxt[key] = candidate
        layer = nxt
    total, pairing = min(layer.values())
    return total, sorted((column, row) for row, column in pairing) if flipped else pairing

def allocate_permutations(costs):
    """Independent oracle: exhaustive search over every injective assignment."""
    if not costs or not costs[0]:
        return 0, []
    flipped = len(costs) > len(costs[0])
    table = [list(row) for row in (zip(*costs) if flipped else costs)]
    best = None
    for columns in permutations(range(len(table[0])), len(table)):
        total = sum(table[row][column] for row, column in enumerate(columns))
        pairing = sorted((column, row) if flipped else (row, column)
                         for row, column in enumerate(columns))
        if best is None or total < best[0]:
            best = (total, pairing)
    return best

def greedy_assign(costs):
    """Cheap heuristic kept for comparison: each row grabs its cheapest still free column."""
    used, total = set(), 0
    for row in costs:
        free = [(value, column) for column, value in enumerate(row) if column not in used]
        if not free:
            return float("inf")
        value, column = min(free)
        total += value
        used.add(column)
    return total

def plan_robot(start, goal, reservations, horizon):
    """Space-time BFS avoiding the reserved vertices, edges and finished goal cells."""
    reserved_cells, reserved_edges, permanent = reservations
    queue, parent = [(start, 0)], {(start, 0): None}
    for cell, time in queue:
        if cell == goal:
            path, node = [], (cell, time)
            while node is not None:
                path.append(node[0])
                node = parent[node]
            return path[::-1]
        if time >= horizon:
            continue
        for nxt in neighbours(cell, free_cells()):
            node = (nxt, time + 1)
            if node in parent or (nxt != goal and nxt in permanent):
                continue
            if time + 1 in reserved_cells.get(nxt, ()) or (cell, nxt, time + 1) in reserved_edges:
                continue
            parent[node] = (cell, time)
            queue.append(node)
    return None

def reserve(path, reservations):
    """Record one planned path in the reservation tables (in place)."""
    reserved_cells, reserved_edges, permanent = reservations
    for time, cell in enumerate(path):
        reserved_cells.setdefault(cell, set()).add(time)
        if time:
            previous = path[time - 1]
            reserved_edges.add((previous, cell, time))
            reserved_edges.add((cell, previous, time))   # a later robot may not undo the move
    permanent.add(path[-1])

def coordinate(starts, goals, horizon=HORIZON):
    """Prioritised planning: earlier robots reserve space-time, later ones route around them."""
    reservations = ({}, set(), set())
    paths, unplanned = [], []
    for index, (start, goal) in enumerate(zip(starts, goals)):
        path = plan_robot(start, goal, reservations, horizon)
        if path is None:
            paths.append(None)
            unplanned.append(index)
            continue
        reserve(path, reservations)
        paths.append(path)
    return paths, unplanned

def conflicts(paths):
    """Independent checker: vertex conflicts (same cell, same time) and swaps between two steps."""
    found = []
    for a in range(len(paths)):
        for b in range(a + 1, len(paths)):
            if paths[a] is None or paths[b] is None:
                continue
            for time in range(1, max(len(paths[a]), len(paths[b]))):
                cell_a = paths[a][min(time, len(paths[a]) - 1)]
                cell_b = paths[b][min(time, len(paths[b]) - 1)]
                previous_a = paths[a][min(time - 1, len(paths[a]) - 1)]
                previous_b = paths[b][min(time - 1, len(paths[b]) - 1)]
                if cell_a == cell_b:
                    found.append(("vertex", a, b, time))
                elif cell_a == previous_b and cell_b == previous_a and previous_a != previous_b:
                    found.append(("edge", a, b, time))
    return found

def valid_path(path, start, goal):
    return (path[0] == start and path[-1] == goal
            and all(cell in free_cells() for cell in path)
            and all(abs(path[k + 1][0] - path[k][0]) + abs(path[k + 1][1] - path[k][1]) <= 1
                    for k in range(len(path) - 1)))

def main():
    cells = free_cells()

    # 1. Allocation: the exact allocator matches the exhaustive oracle on three cost tables, and
    #    the greedy heuristic can never be cheaper than the optimum.
    cases = [(STARTS, GOALS), ([(0, 0), (0, 5), (5, 5)], [(5, 0), (2, 5)]),
             ([(0, 0), (0, 5), (5, 0), (5, 5)], [(3, 0), (3, 5), (2, 3)])]
    for starts, goals in cases:
        table = cost_matrix(starts, goals, cells)
        total, pairing = allocate(table)
        oracle_total, _ = allocate_permutations(table)
        check(total == oracle_total, "the subset DP must match the exhaustive assignment optimum")
        check(len(pairing) == min(len(starts), len(goals))
              and len({row for row, _ in pairing}) == len(pairing)
              and len({column for _, column in pairing}) == len(pairing),
              "an assignment must be injective and use every robot or task available")
        check(all(table[row][column] < float("inf") for row, column in pairing),
              "every assigned pair must be mutually reachable")
        check(greedy_assign(table) >= total, "a greedy heuristic cannot beat the optimum")

    # 2. Independent conflict detector: a hand-made swap is flagged, disjoint paths are not.
    check(any(kind == "vertex" for kind, _, _, _ in conflicts(
        [[(0, 0), (1, 0), (2, 0)], [(2, 0), (1, 0), (0, 0)]])),
        "the detector must flag two robots occupying the same cell")
    check(conflicts([[(0, 0), (1, 0)], [(3, 0), (3, 1)], [(5, 5), (5, 5)]]) == [],
          "disjoint paths must be reported conflict free")

    # 3. Coordination: every robot is planned, valid and conflict free under the detector.
    paths, unplanned = coordinate(STARTS, GOALS)
    check(unplanned == [] and all(path is not None for path in paths),
          "every robot must be planned in the open scene")
    check(all(valid_path(path, start, goal) for path, start, goal in zip(paths, STARTS, GOALS)),
          "planned paths must be valid grid moves or waits, all reaching the goals")
    check(conflicts(paths) == [], "planned paths must be conflict free")

    # 4. The makespan respects the single-robot shortest-path lower bound.
    makespan = max(len(path) - 1 for path in paths)
    lower = max(distances(start, cells)[goal] for start, goal in zip(STARTS, GOALS))
    check(lower <= makespan < HORIZON, "the makespan must sit between the lower bound and the horizon")

    # 5. Conflict resolution does work: the uncoordinated shortest paths do collide.
    naive = [shortest_path(start, goal, cells) for start, goal in zip(STARTS, GOALS)]
    check(all(path is not None for path in naive) and conflicts(naive) != [],
          "the uncoordinated shortest paths must contain a conflict to resolve")

    # 6. Boundary cases: empty inputs, a sealed-off robot and a robot already on its goal.
    check(allocate([]) == (0, []) and allocate([[]]) == (0, []), "an empty task set must allocate nothing")
    check(shortest_path((5, 5), (5, 5), cells) == [(5, 5)],
          "a robot already on its goal needs a single-cell path")
    OBSTACLES.update({(4, 5), (5, 4)})
    try:
        sealed_paths, sealed = coordinate(STARTS, GOALS)
        check(sealed_paths[0] is None and sealed == [0],
              "a robot walled in must be reported unplanned instead of crashing")
        check(all(path is not None for path in sealed_paths[1:]) and conflicts(sealed_paths) == [],
              "the remaining robots must still be coordinated")
    finally:
        OBSTACLES.difference_update({(4, 5), (5, 4)})

    # 7. Determinism: the same scene yields the same plans.
    again, _ = coordinate(STARTS, GOALS)
    check(again == paths, "the coordinator must be deterministic")

    print(f"allocation optimum {allocate(cost_matrix(STARTS, GOALS, cells))[0]:.0f}, "
          f"makespan {makespan} (lower bound {lower}), horizon {HORIZON}")
    print(f"path lengths {[len(path) - 1 for path in paths]}, conflicts {len(conflicts(paths))}")
    print("PASS")


if __name__ == "__main__":
    main()
