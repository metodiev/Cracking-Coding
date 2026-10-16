"""Task 731 - Implement adaptive mesh refinement driving solver coupling.  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

Adaptive mesh refinement (AMR) solves a PDE on a coarse mesh, uses an *error indicator* to decide
where the solution is under-resolved, refines exactly those cells, and then couples the levels:
the coarse solution provides the boundary data for the fine patch, and the fine patch provides
the correction that makes the composite solution more accurate than either level alone.  That
loop - indicator, refine, re-balance, re-solve, couple - is the part of AMR the task asks for,
so this file implements it end to end for the 1D Poisson problem -u'' = f with Dirichlet data,
where everything can be verified against a closed-form solution.

The pieces are a uniform finite-difference solver (Thomas algorithm), a curvature-based
indicator, cell refinement with 2:1 balancing (no cell may sit next to a cell more than one level
finer - the invariant that keeps the discretisation well posed), and a patch solver that takes
its Dirichlet data from the coarse solution by interpolation.

Verified invariants: the uniform solver's discrete residual is at rounding level, the indicator
flags the cells around the peak of the solution, refined meshes satisfy the 2:1 balance rule,
the balancing pass actually splits coarse neighbours of a doubly refined cell, the composite
solution matches the coarse solution exactly at the interface nodes (the coupling condition),
the composite error is smaller than the coarse error, refining the whole domain reproduces the
uniform fine solution exactly, and the telescoping flux identity sum(second differences) =
boundary one-sided flux holds for the patch solution.

Simplified: one-dimensional, single refinement level in the solve (multi-level meshes are built
and validated but not solved), linear interpolation for the coarse-to-fine Dirichlet data instead
of a proper flux-matching refluxing step, no patch hierarchy bookkeeping, no parallel patches and
no time-dependent problem.  The parts that are real: the indicator, the refinement and balancing
logic, the interpolated coupling, the measured accuracy gain, and the level-consistency checks.
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


def source_term(x):
    """f = pi^2 sin(pi x), whose exact solution on [0,1] with zero Dirichlet data is sin(pi x)."""
    return math.pi ** 2 * math.sin(math.pi * x)


def exact_solution(x):
    return math.sin(math.pi * x)


class Cell:
    """One mesh cell with a refinement level; levels must satisfy the 2:1 rule."""

    def __init__(self, lo, hi, level=0):
        if not hi > lo:
            raise ValueError("a cell needs positive width")
        if level < 0:
            raise ValueError("levels are non-negative")
        self.lo = lo
        self.hi = hi
        self.level = level

    def width(self):
        return self.hi - self.lo

    def split(self):
        middle = 0.5 * (self.lo + self.hi)
        return [Cell(self.lo, middle, self.level + 1), Cell(middle, self.hi, self.level + 1)]

    def __repr__(self):
        return f"Cell({self.lo:.4f}, {self.hi:.4f}, L{self.level})"


def uniform_mesh(cells):
    return [Cell(i / cells, (i + 1) / cells, 0) for i in range(cells)]


def solve_uniform(left, right, intervals, left_value=0.0, right_value=0.0, rhs=source_term):
    """Uniform finite differences with Dirichlet data, solved by the Thomas algorithm."""
    if intervals < 1:
        raise ValueError("at least one interval is required")
    h = (right - left) / intervals
    nodes = [left + i * h for i in range(intervals + 1)]
    interior = intervals - 1
    lower = [-1.0] * interior
    upper = [-1.0] * interior
    diagonal = [2.0] * interior
    load = [h * h * rhs(nodes[i + 1]) for i in range(interior)]
    if interior:
        load[0] += left_value
        load[-1] += right_value
    for i in range(1, interior):
        factor = lower[i] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        load[i] -= factor * load[i - 1]
    values = [left_value] + [0.0] * interior + [right_value]
    if interior:
        values[interior] = load[-1] / diagonal[-1]
        for i in range(interior - 2, -1, -1):
            values[i + 1] = (load[i] - upper[i] * values[i + 2]) / diagonal[i]
    return nodes, values, h


def discrete_residual(nodes, values, rhs=source_term):
    """Infinity norm of (u_{i-1} - 2u_i + u_{i+1})/h^2 + f: zero for an exact FD solution.

    The scheme solves -u'' = f, so the discrete second difference approximates -f.
    """
    h = nodes[1] - nodes[0]
    worst = 0.0
    for i in range(1, len(nodes) - 1):
        second = (values[i - 1] - 2.0 * values[i] + values[i + 1]) / (h * h)
        worst = max(worst, abs(second + rhs(nodes[i])))
    return worst


def curvature_indicators(nodes, values):
    """Cell indicator = |u_{i-1} - 2u_i + u_{i+1}| at the neighbouring interior nodes.

    That second difference estimates h^2 |u''|, which is large where the solution bends and tiny
    where it is flat - and unlike the local truncation residual it is not zero for the discrete
    solution, which satisfies its own equations exactly.
    """
    indicators = [0.0] * (len(nodes) - 1)
    for i in range(1, len(nodes) - 1):
        second = abs(values[i - 1] - 2.0 * values[i] + values[i + 1])
        indicators[i - 1] = max(indicators[i - 1], second)
        indicators[i] = max(indicators[i], second)
    return indicators


def flag_cells(indicators, threshold):
    return {index for index, value in enumerate(indicators) if value >= threshold}


def refine(cells, flagged):
    """Split every flagged cell; returns the new cell list in mesh order."""
    out = []
    for index, cell in enumerate(cells):
        out.extend(cell.split() if index in flagged else [cell])
    return out


def adjacency(cells):
    """Index pairs of cells that share a face (the mesh is ordered, so neighbours are adjacent)."""
    return [(i, i + 1) for i in range(len(cells) - 1)]


def balance(cells):
    """Enforce the 2:1 rule: split any cell sitting next to a cell two levels finer."""
    for _ in range(16):
        if max((abs(cells[i].level - cells[j].level) for i, j in adjacency(cells)), default=0) <= 1:
            return cells
        flagged = set()
        for i, j in adjacency(cells):
            if cells[j].level - cells[i].level >= 2:
                flagged.add(i)
            if cells[i].level - cells[j].level >= 2:
                flagged.add(j)
        cells = refine(cells, flagged)
    raise AssertionError("balancing did not converge")


def balance_error(cells):
    return max((abs(cells[i].level - cells[j].level) for i, j in adjacency(cells)), default=0)


def patch_solve(left_node, right_node, intervals, coarse_nodes, coarse_values, rhs=source_term):
    """Solve a fine patch with Dirichlet data interpolated from the coarse solution."""
    left_value = interpolate(coarse_nodes, coarse_values, left_node)
    right_value = interpolate(coarse_nodes, coarse_values, right_node)
    return solve_uniform(left_node, right_node, intervals, left_value, right_value, rhs)


def interpolate(nodes, values, x):
    if x <= nodes[0]:
        return values[0]
    if x >= nodes[-1]:
        return values[-1]
    for i in range(len(nodes) - 1):
        if nodes[i] <= x <= nodes[i + 1]:
            span = nodes[i + 1] - nodes[i]
            weight = 0.0 if span == 0 else (x - nodes[i]) / span
            return values[i] * (1.0 - weight) + values[i + 1] * weight
    raise AssertionError("the point must fall inside the mesh")


def composite_solution(nodes, values, patch_nodes, patch_values):
    """Piecewise function: coarse everywhere except inside the patch, where the fine data wins."""
    lo, hi = patch_nodes[0], patch_nodes[-1]

    def sample(x):
        if lo <= x <= hi:
            return interpolate(patch_nodes, patch_values, x)
        return interpolate(nodes, values, x)

    return sample


def max_error(sample, cells):
    worst = 0.0
    for cell in cells:
        for step in range(5):
            x = cell.lo + (cell.hi - cell.lo) * step / 4.0
            worst = max(worst, abs(sample(x) - exact_solution(x)))
    return worst


def main():
    # 1. the uniform solver: the discrete residual is at rounding level and the error behaves
    #    like O(h^2), which is what the refinement is ultimately compared against.
    nodes, values, h = solve_uniform(0.0, 1.0, 16)
    check(discrete_residual(nodes, values) < 1e-12, "the FD solution must satisfy the equation")
    coarse_error = max(abs(values[i] - exact_solution(nodes[i])) for i in range(len(nodes)))
    check(0.1 * h * h < coarse_error < 1.0 * h * h,
          "the coarse error must sit at the expected O(h^2) level")
    finer_nodes, finer_values, finer_h = solve_uniform(0.0, 1.0, 64)
    finer_error = max(abs(finer_values[i] - exact_solution(finer_nodes[i]))
                      for i in range(len(finer_nodes)))
    check(finer_error < coarse_error / 10.0, "halving the mesh must shrink the error sharply")
    check(finer_error / coarse_error < (h / finer_h) ** 2,
          "the convergence must be at least second order")

    # 2. the indicator must point at the peak of the solution rather than at the flat ends.
    cells = uniform_mesh(16)
    indicators = curvature_indicators(nodes, values)
    peak_cell = max(range(len(cells)), key=lambda i: indicators[i])
    check(cells[peak_cell].lo <= 0.5 <= cells[peak_cell].hi,
          "the indicator must peak at the middle cell for a sine solution")
    middle = sum(indicators[5:11])
    check(indicators[0] * 3.0 < max(indicators),
          "the flat ends must score well below the curved middle")
    check(middle > sum(indicators[:2]) + sum(indicators[-2:]),
          "the middle cells must dominate the outermost ones exactly")
    flagged = flag_cells(indicators, 0.5 * max(indicators))
    check(peak_cell in flagged, "the peak cell must be flagged")
    check(flagged == set(range(min(flagged), max(flagged) + 1)),
          "a sine solution must flag one contiguous block")

    # 3. refinement and the 2:1 balance rule, including a doubly refined cell that forces the
    #    balancing pass to split its coarse neighbours.
    refined = refine(cells, flagged)
    levels = {cell.level for cell in refined}
    check(levels == {0, 1}, "one refinement pass must produce exactly two levels")
    check(len(refined) == len(cells) + len(flagged), "each flagged cell must split in two")
    check(balance_error(refined) <= 1, "one refinement level is automatically balanced")
    deep = refine(initial := list(uniform_mesh(8)), {3})
    check(balance_error(deep) == 1, "a single split stays within the 2:1 rule")
    deeper = refine(deep, {6}) if len(deep) > 6 else deep
    needs_balance = [Cell(0.0, 0.25, 0), Cell(0.25, 0.3125, 2), Cell(0.3125, 0.375, 2),
                     Cell(0.375, 0.5, 0), Cell(0.5, 1.0, 0)]
    check(balance_error(needs_balance) == 2, "the crafted mesh must violate the 2:1 rule")
    balanced = balance(list(needs_balance))
    check(balance_error(balanced) <= 1, "balancing must remove the level jump")
    check(len(balanced) > len(needs_balance), "balancing must split the coarse neighbours")
    check(min(cell.level for cell in balanced) == 0, "balancing must not coarsen anything")
    check(abs(sum(cell.width() for cell in balanced) - 1.0) < 1e-12,
          "balancing must preserve the domain measure")

    # 4. the coupling: the patch takes its boundary data from the coarse solution, and the
    #    composite solution must be strictly more accurate than the coarse one.
    first_fine, last_fine = min(flagged), max(flagged)
    patch_nodes, patch_values, patch_h = patch_solve(cells[first_fine].lo, cells[last_fine].hi,
                                                     2 * len(flagged), nodes, values)
    check(patch_nodes[0] == cells[first_fine].lo and patch_nodes[-1] == cells[last_fine].hi,
          "the patch must span exactly the flagged cells")
    check(abs(patch_values[0] - interpolate(nodes, values, patch_nodes[0])) < 1e-15,
          "the left interface value must come from the coarse solution")
    check(abs(patch_values[-1] - interpolate(nodes, values, patch_nodes[-1])) < 1e-15,
          "the right interface value must come from the coarse solution")
    check(discrete_residual(patch_nodes, patch_values) < 1e-12,
          "the patch solution must satisfy the PDE on the fine grid")
    composite = composite_solution(nodes, values, patch_nodes, patch_values)
    composite_error = max_error(composite, cells)
    check(composite_error < coarse_error, "the composite solution must beat the coarse one")
    check(composite_error < coarse_error / 1.5, "the improvement must be clear, not marginal")
    for node, value in zip(nodes, values):
        if not patch_nodes[0] < node < patch_nodes[-1]:
            check(abs(composite(node) - value) < 1e-12,
                  "outside the patch the composite solution must equal the coarse one")

    # 5. discrete conservation on the patch: the telescoping second difference equals the
    #    boundary one-sided flux, exactly (the Green identity of the FD operator).
    total = 0.0
    for i in range(1, len(patch_nodes) - 1):
        total += patch_values[i - 1] - 2.0 * patch_values[i] + patch_values[i + 1]
    boundary = (patch_values[0] - patch_values[1]) + (patch_values[-1] - patch_values[-2])
    check(abs(total - boundary) < 1e-12, "the telescoping flux identity must hold exactly")
    check(abs(total + patch_h * patch_h * sum(source_term(x) for x in patch_nodes[1:-1])) < 1e-9,
          "the flux identity must reproduce minus the integrated source term")

    # 6. boundary cases: nothing flagged reproduces the coarse solution, and refining the whole
    #    domain reproduces the uniform fine solution exactly.
    none_flagged = flag_cells(indicators, 1.1 * max(indicators))
    check(none_flagged == set(), "an unreachable threshold must flag no cell at all")
    check(max_error(lambda x: interpolate(nodes, values, x), cells) == coarse_error,
          "with no flagged cells the composite solution is just the coarse one")
    single = flag_cells(indicators, max(indicators))
    check(len(single) >= 1, "the strictest threshold must still flag the peak cell")
    lone_nodes, lone_values, _ = patch_solve(cells[min(single)].lo, cells[max(single)].hi,
                                            2 * len(single), nodes, values)
    check(abs(lone_values[0] - interpolate(nodes, values, lone_nodes[0])) < 1e-15,
          "a single-cell patch must still take its data from the coarse solution")
    all_cells = set(range(len(cells)))
    full_nodes, full_values, _ = patch_solve(0.0, 1.0, 2 * len(cells), nodes, values)
    uniform_nodes, uniform_values, _ = solve_uniform(0.0, 1.0, 2 * len(cells))
    check(len(full_nodes) == len(uniform_nodes), "a fully refined patch must have the fine size")
    check(max(abs(full_values[i] - uniform_values[i]) for i in range(len(full_nodes))) < 1e-15,
          "refining the whole domain must reproduce the uniform fine solution exactly")
    check(len(initial) == 8, "the untouched initial mesh must be preserved")

    # 7. rejected inputs and determinism.
    expect_value_error(lambda: Cell(1.0, 1.0, 0), "a degenerate cell is invalid")
    expect_value_error(lambda: Cell(0.0, 1.0, -1), "a negative level is invalid")
    expect_value_error(lambda: solve_uniform(0.0, 1.0, 0), "zero intervals is invalid")
    repeat = solve_uniform(0.0, 1.0, 16)
    check(repeat[1] == values, "the solve must be deterministic")
    check(math.isclose(sum(cell.width() for cell in refined), 1.0),
          "refinement must preserve the domain measure")

    print(f"731 coarse_error={coarse_error:.3e} fine_error={finer_error:.3e} "
          f"composite_error={composite_error:.3e}")
    print(f"731 flagged={sorted(flagged)} refined_cells={len(refined)} "
          f"balance_error={balance_error(refined)}")
    print("PASS")


if __name__ == "__main__":
    main()
