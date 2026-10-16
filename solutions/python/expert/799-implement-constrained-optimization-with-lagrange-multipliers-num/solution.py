"""Task 799 - Implement constrained optimization with Lagrange multipliers numeric solver.  (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)

For min f(x) subject to g_i(x) = 0 and h_j(x) <= 0 the first-order optimality conditions are

    grad f + sum_i lambda_i grad g_i + sum_j mu_j grad h_j = 0,
    g = 0,   h <= 0,   mu_j >= 0,   mu_j h_j = 0.

With only equalities a Newton step on that system is

    [ H   J^T ] [dx ]   [ grad f + J^T lambda ]
    [ J    0  ] [dlam] = -[ g                 ]

where H is the Hessian of the Lagrangian L = f + lambda^T g (here built by central second
differences of L, so the solver works for any twice-differentiable input) and J the constraint
Jacobian.  Inequalities are handled with a classical active-set loop: solve the equality-
constrained problem on the current active set, drop any active multiplier that came out
negative (the constraint is not really binding) and add any inactive constraint that is
violated, until the KKT system is satisfied.  The second-order condition is checked
independently by projecting H onto the null space of J (Gram-Schmidt) and testing
Z^T H Z positive definite.

Verified against closed forms with measured tolerances: min x^2+y^2 s.t. x+y=1 gives
(0.5, 0.5) with lambda = -1, min x^2+2y^2 s.t. x+y=1 gives (2/3, 1/3) with lambda = -4/3, and
the three-variable problem with two constraints gives (1, 1, 1); the multiplier sign and
complementary slackness of an active inequality are checked explicitly, an inactive inequality
returns mu = 0; each test is cross-checked against an independent one-dimensional search on the
constraint manifold and against a solver run on finite-difference gradients; and infeasible or
degenerate constraint sets are reported instead of returning a wrong answer.

Simplified versus a production NLP solver: dense linear algebra, finite-difference Hessians
(no exact second derivatives and no quasi-Newton updates), a plain active-set loop instead of
an interior-point method, and no line search or trust region - the Newton step is accepted
directly, which is adequate for the smooth small problems used here.
"""
import math

IDENTITY = None


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------- linear algebra
def solve_linear(matrix, rhs):
    """Dense Gaussian elimination with partial pivoting; None when singular."""
    n = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inv
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = (a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
    return x


# ------------------------------------------------------------- differentiation
def numeric_gradient(f, x, step=1e-6):
    gradient = []
    for i in range(len(x)):
        plus = list(x)
        minus = list(x)
        plus[i] += step
        minus[i] -= step
        gradient.append((f(plus) - f(minus)) / (2.0 * step))
    return gradient


def numeric_jacobian(constraints, x, step=1e-6):
    rows = []
    for constraint in constraints:
        row = []
        for i in range(len(x)):
            plus = list(x)
            minus = list(x)
            plus[i] += step
            minus[i] -= step
            row.append((constraint(plus) - constraint(minus)) / (2.0 * step))
        rows.append(row)
    return rows


def lagrangian_gradient(f, constraints, multipliers, x):
    gradient = numeric_gradient(f, x)
    jacobian = numeric_jacobian(constraints, x)
    for row, multiplier in zip(jacobian, multipliers):
        for i in range(len(x)):
            gradient[i] += multiplier * row[i]
    return gradient


def lagrangian_hessian(f, constraints, multipliers, x, step=1e-5):
    """Central second differences of L = f + sum mu g."""
    n = len(x)
    hessian = [[0.0] * n for _ in range(n)]

    def lagrangian(point):
        value = f(point)
        for index, constraint in enumerate(constraints):
            value += multipliers[index] * constraint(point)
        return value

    for i in range(n):
        for j in range(n):
            if i == j:
                plus = list(x)
                minus = list(x)
                plus[i] += step
                minus[i] -= step
                hessian[i][j] = (lagrangian(plus) - 2.0 * lagrangian(x)
                                 + lagrangian(minus)) / (step * step)
            elif j > i:
                pp = list(x)
                pm = list(x)
                mp = list(x)
                mm = list(x)
                pp[i] += step
                pp[j] += step
                pm[i] += step
                pm[j] -= step
                mp[i] -= step
                mp[j] += step
                mm[i] -= step
                mm[j] -= step
                value = (lagrangian(pp) - lagrangian(pm) - lagrangian(mp)
                         + lagrangian(mm)) / (4.0 * step * step)
                hessian[i][j] = hessian[j][i] = value
    return hessian


def null_space_basis(jacobian, dimension):
    """Orthonormal basis of the null space of the constraint Jacobian (Gram-Schmidt)."""
    rows = [list(row) for row in jacobian]
    basis = []
    for row in rows:
        vector = row[:]
        for existing in basis:
            projection = math.fsum(a * b for a, b in zip(vector, existing))
            vector = [a - projection * b for a, b in zip(vector, existing)]
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm > 1e-8:
            basis.append([value / norm for value in vector])
    free = []
    for axis in range(dimension):
        vector = [1.0 if i == axis else 0.0 for i in range(dimension)]
        for existing in basis + free:
            projection = math.fsum(a * b for a, b in zip(vector, existing))
            vector = [a - projection * b for a, b in zip(vector, existing)]
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm > 1e-8:
            free.append([value / norm for value in vector])
    return free


# -------------------------------------------------------------------- solver
def kkt_solve(f, equality, inequality, x, active, multipliers, max_iter=120, tol=1e-9):
    """Newton on the KKT system for the current active set."""
    constraints = list(equality) + [inequality[j] for j in active]
    count = len(constraints)
    lam = list(multipliers) + [0.0] * (count - len(multipliers))
    lam = lam[:count]
    point = list(x)
    history = []
    stalled = 0
    for _ in range(max_iter):
        gradient = numeric_gradient(f, point)
        jacobian = numeric_jacobian(constraints, point)
        values = [constraint(point) for constraint in constraints]
        stationarity = [gradient[i] + math.fsum(jacobian[j][i] * lam[j]
                                                for j in range(count))
                        for i in range(len(point))]
        residual = max([abs(value) for value in stationarity + values] or [0.0])
        history.append(residual)
        if residual < tol:
            break
        if len(history) > 2 and history[-2] - residual <= 1e-14 * max(1.0, history[-2]):
            stalled += 1
            if stalled >= 2:                     # finite-difference noise floor reached
                break
        else:
            stalled = 0
        hessian = lagrangian_hessian(f, constraints, lam, point)
        size = len(point) + count
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(len(point)):
            for j in range(len(point)):
                matrix[i][j] = hessian[i][j]
            for j in range(count):
                matrix[i][len(point) + j] = jacobian[j][i]
        for j in range(count):
            for i in range(len(point)):
                matrix[len(point) + j][i] = jacobian[j][i]
        rhs = [-value for value in stationarity] + [-value for value in values]
        step = solve_linear(matrix, rhs)
        if step is None:
            raise ValueError("singular KKT system (degenerate or infeasible constraints)")
        for i in range(len(point)):
            point[i] += step[i]
        for j in range(count):
            lam[j] += step[len(point) + j]
    else:
        raise ValueError("the KKT iteration did not converge")
    return point, lam, history


def solve_constrained(f, equality, inequality, x0, max_iter=80):
    """Active-set loop around the equality-constrained Newton solve."""
    inequality = list(inequality)
    active = []
    multipliers = []
    point = list(x0)
    for _ in range(max_iter):
        point, lam, history = kkt_solve(f, equality, inequality, point, active, multipliers)
        if not inequality:
            return {"x": point, "lam_eq": lam[:len(equality)], "mu": [], "active": [],
                    "residual": history[-1]}
        active_multipliers = lam[len(equality):]
        feasible = [inequality[j](point) for j in range(len(inequality))]
        violated = [j for j in range(len(inequality))
                    if j not in active and feasible[j] > 1e-9]
        negative = [index for index, value in enumerate(active_multipliers) if value < -1e-9]
        if violated:
            active.append(violated[0])
            multipliers = []
            continue
        if negative:
            del active[negative[0]]
            multipliers = []
            continue
        return {"x": point, "lam_eq": lam[:len(equality)], "mu": active_multipliers,
                "active": list(active), "residual": history[-1]}
    raise ValueError("the active-set loop did not settle")


def kkt_diagnostics(f, equality, inequality, solution):
    """Residual of the KKT conditions plus the projected-Hessian second-order test."""
    point = solution["x"]
    gradient = numeric_gradient(f, point)
    jacobian = numeric_jacobian(list(equality) + list(inequality), point)
    stationarity = list(gradient)
    for index, row in enumerate(jacobian):
        multiplier = (solution["lam_eq"][index] if index < len(equality)
                      else solution["mu"][index - len(equality)])
        for i in range(len(point)):
            stationarity[i] += multiplier * row[i]
    equality_values = [constraint(point) for constraint in equality]
    inequality_values = [constraint(point) for constraint in inequality]
    multipliers = list(solution["lam_eq"]) + list(solution["mu"])
    hessian = lagrangian_hessian(f, list(equality) + list(inequality), multipliers, point)
    projected_curvature = None
    projected = null_space_basis(jacobian[:len(equality)], len(point)) if equality else []
    if projected:
        curvature = []
        for z_i in projected:
            row = []
            for z_j in projected:
                total = 0.0
                for i in range(len(point)):
                    inner = math.fsum(hessian[i][j] * z_j[j] for j in range(len(point)))
                    total += z_i[i] * inner
                row.append(total)
            curvature.append(row)
        projected_curvature = [curvature[i][i] for i in range(len(curvature))]
    return {"stationarity": max(abs(value) for value in stationarity),
            "feasibility": max([abs(value) for value in equality_values]
                               + [max(0.0, value) for value in inequality_values] + [0.0]),
            "complementarity": max([abs(multiplier * value) for multiplier, value
                                    in zip(solution["mu"], inequality_values)] + [0.0]),
            "curvature": projected_curvature}


# ------------------------------------------------------------------ problems
def f_circle(x):
    return x[0] ** 2 + x[1] ** 2


def f_ellipse(x):
    return x[0] ** 2 + 2.0 * x[1] ** 2


def f_three(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


def g_line(x):
    return x[0] + x[1] - 1.0


def g_plane(x):
    return x[0] + x[1] + x[2] - 3.0


def g_symmetry(x):
    return x[0] - x[1]


def h_above(x):
    return 1.0 - x[0] - x[1]                     # x + y >= 1  <=>  h <= 0


def h_above_loose(x):
    return -1.0 - x[0] - x[1]                    # x + y >= -1, inactive at the optimum


def ternary_search(f, lo, hi, iterations=200):
    """One-dimensional reference minimisation (independent of the KKT machinery)."""
    for _ in range(iterations):
        third = (hi - lo) / 3.0
        left, right = lo + third, hi - third
        if f(left) < f(right):
            hi = right
        else:
            lo = left
    point = 0.5 * (lo + hi)
    return point, f(point)


def main():
    # --- 1. equality-constrained closed forms -------------------------------
    circle = solve_constrained(f_circle, [g_line], [], [0.0, 0.0])
    check(abs(circle["x"][0] - 0.5) < 1e-8 and abs(circle["x"][1] - 0.5) < 1e-8,
          "min x^2+y^2 s.t. x+y=1 must give (0.5, 0.5), got %r" % (circle["x"],))
    check(abs(circle["lam_eq"][0] + 1.0) < 1e-6,
          "the multiplier of the circle problem must be -1, got %.8f" % circle["lam_eq"][0])

    ellipse = solve_constrained(f_ellipse, [g_line], [], [0.0, 0.0])
    check(abs(ellipse["x"][0] - 2.0 / 3.0) < 1e-8 and abs(ellipse["x"][1] - 1.0 / 3.0) < 1e-8,
          "min x^2+2y^2 s.t. x+y=1 must give (2/3, 1/3), got %r" % (ellipse["x"],))
    check(abs(ellipse["lam_eq"][0] + 4.0 / 3.0) < 1e-6,
          "the ellipse multiplier must be -4/3, got %.8f" % ellipse["lam_eq"][0])

    three = solve_constrained(f_three, [g_plane, g_symmetry], [], [0.0, 0.0, 0.0])
    check(max(abs(value - 1.0) for value in three["x"]) < 1e-8,
          "the three-variable problem must give (1, 1, 1), got %r" % (three["x"],))

    # --- 2. independent one-dimensional oracles ----------------------------
    line_point, line_value = ternary_search(lambda t: f_circle([t, 1.0 - t]), -3.0, 3.0)
    check(abs(line_value - f_circle(circle["x"])) < 1e-10,
          "the manifold search %.10f must match the KKT optimum %.10f"
          % (line_value, f_circle(circle["x"])))
    check(abs(line_point - 0.5) < 1e-6, "the manifold search must locate x = 0.5")
    reduced = ternary_search(lambda t: f_three([t, t, 3.0 - 2.0 * t]), -2.0, 2.0)
    check(abs(reduced[1] - f_three(three["x"])) < 1e-9,
          "the two-constraint manifold search %.10f must match the KKT optimum %.10f"
          % (reduced[1], f_three(three["x"])))

    # --- 3. KKT residuals and the second-order condition -------------------
    for solution, f, equality, name in ((circle, f_circle, [g_line], "circle"),
                                        (ellipse, f_ellipse, [g_line], "ellipse"),
                                        (three, f_three, [g_plane, g_symmetry], "three")):
        diagnostics = kkt_diagnostics(f, equality, [], solution)
        check(diagnostics["stationarity"] < 1e-5,
              "%s: stationarity residual %.3e" % (name, diagnostics["stationarity"]))
        check(diagnostics["feasibility"] < 1e-9,
              "%s: feasibility residual %.3e" % (name, diagnostics["feasibility"]))
        check(diagnostics["curvature"] is not None
              and all(value > 0.0 for value in diagnostics["curvature"]),
              "%s: the projected Lagrangian Hessian must be positive definite" % name)

    # --- 4. inequality constraints and active sets -------------------------
    active_problem = solve_constrained(f_circle, [], [h_above], [2.0, 2.0])
    check(abs(active_problem["x"][0] - 0.5) < 1e-7
          and abs(active_problem["x"][1] - 0.5) < 1e-7,
          "min x^2+y^2 s.t. x+y>=1 must give (0.5, 0.5), got %r" % (active_problem["x"],))
    check(active_problem["active"] == [0], "the inequality must be detected as active")
    check(abs(active_problem["mu"][0] - 1.0) < 1e-5,
          "the active inequality multiplier must be +1, got %.8f" % active_problem["mu"][0])
    check(active_problem["mu"][0] > 0.0, "an active inequality must have a positive multiplier")
    inactive_problem = solve_constrained(f_circle, [], [h_above_loose], [2.0, 2.0])
    check(abs(inactive_problem["x"][0]) < 1e-8 and abs(inactive_problem["x"][1]) < 1e-8,
          "the unconstrained minimum (0, 0) must be returned when x+y>=-1 is inactive, got %r"
          % (inactive_problem["x"],))
    check(inactive_problem["active"] == [],
          "an inactive constraint must not enter the active set")
    diagnostics = kkt_diagnostics(f_circle, [], [h_above], active_problem)
    check(diagnostics["complementarity"] < 1e-8,
          "complementary slackness residual %.3e" % diagnostics["complementarity"])
    check(diagnostics["feasibility"] < 1e-9, "the inequality must hold at the solution")

    # --- 5. a nonlinear objective on a nonlinear constraint -----------------
    nonlinear = solve_constrained(lambda x: -x[0] * x[1],
                                  [lambda x: x[0] ** 2 + x[1] ** 2 - 1.0], [], [0.9, 0.9])
    check(abs(nonlinear["x"][0] - math.sqrt(0.5)) < 1e-6
          and abs(nonlinear["x"][1] - math.sqrt(0.5)) < 1e-6,
          "min -xy on the unit circle must give (1/sqrt2, 1/sqrt2), got %r"
          % (nonlinear["x"],))
    check(abs(-nonlinear["x"][0] * nonlinear["x"][1] + 0.5) < 1e-9,
          "the circle objective at the optimum must be -0.5")
    nonlinear_angle = ternary_search(
        lambda t: -math.cos(t) * math.sin(t), 0.3, 1.2)
    check(abs(nonlinear_angle[1] + 0.5) < 1e-9,
          "the polar search must reproduce the objective -0.5, got %.10f" % nonlinear_angle[1])
    repeated = solve_constrained(lambda x: -x[0] * x[1],
                                 [lambda x: x[0] ** 2 + x[1] ** 2 - 1.0], [], [0.9, 0.9])
    check(repeated["x"] == nonlinear["x"], "the nonlinear solve must be deterministic")

    # --- 6. unconstrained, infeasible and degenerate cases -----------------
    free = solve_constrained(lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2, [], [], [0.0, 0.0])
    check(abs(free["x"][0] - 3.0) < 1e-8 and abs(free["x"][1] + 1.0) < 1e-8,
          "an unconstrained problem must converge to (3, -1), got %r" % (free["x"],))
    raised = False
    try:
        solve_constrained(f_circle, [g_line, lambda x: x[0] + x[1] - 2.0], [], [0.0, 0.0])
    except ValueError:
        raised = True
    check(raised, "an infeasible equality system must be reported, not silently solved")
    raised = False
    try:
        solve_constrained(f_circle, [g_line, g_line], [], [0.0, 0.0])
    except ValueError:
        raised = True
    check(raised, "a duplicated (redundant) constraint must be reported as degenerate")
    repeat = solve_constrained(f_ellipse, [g_line], [], [0.0, 0.0])
    check(repeat["x"] == ellipse["x"], "the solver must be deterministic")

    print("circle: x=%r lambda=%.8f, manifold search %.10f vs optimum %.10f"
          % ([round(value, 10) for value in circle["x"]], circle["lam_eq"][0],
             line_value, f_circle(circle["x"])))
    print("ellipse: x=%r lambda=%.8f" % ([round(value, 10) for value in ellipse["x"]],
                                         ellipse["lam_eq"][0]))
    print("three variables: x=%r" % ([round(value, 10) for value in three["x"]],))
    print("active inequality: x=%r mu=%.8f active=%r"
          % ([round(value, 8) for value in active_problem["x"]],
             active_problem["mu"][0], active_problem["active"]))
    print("KKT residuals: stationarity %.3e, complementarity %.3e"
          % (diagnostics["stationarity"], diagnostics["complementarity"]))
    print("PASS")


if __name__ == "__main__":
    main()
