"""Task 788 - Implement stiff ODE solver L-stable method concept.  (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)

A method is A-stable when its linear stability function R(z) satisfies |R(z)| <= 1 for every
z in the closed left half-plane, and L-stable when in addition R(z) -> 0 as z -> -infinity.
That extra decay is what kills the artefacts of stiff problems (the fast, decaying modes
must be annihilated by the step, not merely kept bounded).  Two L-stable methods are
implemented here through one generic diagonally/implicit Runge-Kutta driver:

  * implicit Euler, A = [[1]], b = [1]:      R(z) = 1 / (1 - z),
  * Alexander's 2-stage SDIRK of order 2 with gamma = 1 - 1/sqrt(2),
        A = [[g, 0], [1 - 2g, g]],  b = [1/2, 1/2],  c = [g, 1 - g],
    whose stability function is R(z) = 1 + z (1 - g z + (1 - 2g) z / 2) / (1 - g z)^2 and,
    because 1 - 2g = 2 g (1 - g), satisfies R(z) -> 0 exactly as z -> -infinity.

Each step solves the stage equations Y_i = y + h * sum_j a_ij f(t + c_j h, Y_j) with a dense
Newton iteration: the residual and its Jacobian (d f/d y by central differences) are built
for the flattened stage vector, the linear system is solved with Gaussian elimination with
partial pivoting, and the residual history is returned so the quadratic convergence can be
checked rather than assumed.  The stability function is evaluated independently of the
Newton code by solving (I - z A) w = e in complex arithmetic and forming 1 + z b^T w, which
is then compared against the amplification the integrator actually produces.

Concept-level simplifications: scalar or small vector first-order ODEs, a dense Newton solve
with a finite-difference Jacobian (no analytic Jacobians, no sparse LU, no Newton-Raphson
with reuse across steps), no adaptive error control, and only the two tableaus above.
Verified against closed forms (y = cos t for the stiff smooth test, y = 1/(1 + 1000 t) for
the nonlinear stiff test) and against an independent fixed-step RK4 solution; the explicit
methods are shown to blow up where the L-stable ones succeed with the same step size.
"""
import math

GAMMA = 1.0 - math.sqrt(0.5)                     # 1 - 1/sqrt(2), 1 - 2*GAMMA = 2*GAMMA^2
IMPLICIT_EULER = ("implicit Euler", ((1.0,),), (1.0,), (1.0,))
ALEXANDER_SDIRK2 = ("Alexander SDIRK2", ((GAMMA, 0.0), (1.0 - 2.0 * GAMMA, GAMMA)),
                    (GAMMA, 1.0 - GAMMA), (0.5, 0.5))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def solve_linear(matrix, rhs):
    """Dense Gaussian elimination with partial pivoting; works for floats and complex."""
    n = len(rhs)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular system")
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inv
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))
        x[row] = acc / a[row][row]
    return x


class ImplicitRungeKutta:
    """Generic implicit RK step with a dense Newton solve on the stage vector."""

    def __init__(self, tableau):
        name, a_rows, c, b = tableau
        self.name = name
        self.A = a_rows
        self.C = c
        self.B = b
        self.stages = len(b)
        if len(a_rows) != self.stages or any(len(row) != self.stages for row in a_rows):
            raise ValueError("tableau must be square")
        if len(c) != self.stages:
            raise ValueError("the node vector must match the stage count")
        if any(abs(sum(row) - c[i]) > 1e-14 for i, row in enumerate(a_rows)):
            raise ValueError("row sums must reproduce the nodes c")

    # ------------------------------------------------------------------ stages
    def _residual(self, f, t, y, h, stages):
        """Residual of the stage equations plus the stage derivatives it needs."""
        m = len(y)
        deriv = [f(t + self.C[j] * h, stages[j * m:(j + 1) * m]) for j in range(self.stages)]
        out = [0.0] * (self.stages * m)
        for i in range(self.stages):
            for k in range(m):
                acc = stages[i * m + k] - y[k]
                for j in range(i + 1):
                    acc -= h * self.A[i][j] * deriv[j][k]
                out[i * m + k] = acc
        return out, deriv

    def _jacobian(self, f, t, y, h, stages, deriv, delta=1e-7):
        m = len(y)
        size = self.stages * m
        jac = [[0.0] * size for _ in range(size)]
        partials = []
        for j in range(self.stages):
            base = stages[j * m:(j + 1) * m]
            block = [[0.0] * m for _ in range(m)]        # partials[l][k] = df_k/dy_l
            for l in range(m):
                step = delta * max(1.0, abs(base[l]))
                plus = list(base)
                minus = list(base)
                plus[l] += step
                minus[l] -= step
                fp = f(t + self.C[j] * h, plus)
                fm = f(t + self.C[j] * h, minus)
                for k in range(m):
                    block[l][k] = (fp[k] - fm[k]) / (2.0 * step)
            partials.append(block)
        for i in range(self.stages):
            for j in range(self.stages):
                factor = -h * self.A[i][j]
                for k in range(m):
                    if i == j:
                        jac[i * m + k][i * m + k] = 1.0
                    if factor != 0.0:
                        for l in range(m):
                            jac[i * m + k][j * m + l] += factor * partials[j][l][k]
        return jac

    def step(self, f, t, y, h, max_iter=40, tol=1e-13):
        """Advance one step; returns (y_new, residual_history, iterations, stage_vector)."""
        m = len(y)
        if h < 0.0:
            raise ValueError("h must be non-negative")
        derivative = f(t, y)
        stages = [y[k] + h * self.C[i] * derivative[k]
                  for i in range(self.stages) for k in range(m)]
        history = []
        for iteration in range(1, max_iter + 1):
            residual, deriv = self._residual(f, t, y, h, stages)
            norm = max(abs(v) for v in residual)
            history.append(norm)
            if norm <= tol:
                break
            jac = self._jacobian(f, t, y, h, stages, deriv)
            delta_vec = solve_linear(jac, [-v for v in residual])
            for index in range(len(stages)):
                stages[index] += delta_vec[index]
        else:
            raise RuntimeError("Newton did not converge in %d iterations" % max_iter)
        deriv = [f(t + self.C[j] * h, stages[j * m:(j + 1) * m]) for j in range(self.stages)]
        y_new = [y[k] + h * sum(self.B[j] * deriv[j][k] for j in range(self.stages))
                 for k in range(m)]
        return y_new, history, iteration, stages

    # -------------------------------------------------------- stability function
    def amplification(self, z):
        """R(z) = 1 + z * b^T (I - z A)^{-1} e, solved in complex arithmetic."""
        n = self.stages
        matrix = [[complex(1.0, 0.0) if i == j else complex(0.0, 0.0) for j in range(n)]
                  for i in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] -= z * self.A[i][j]
        w = solve_linear(matrix, [complex(1.0, 0.0)] * n)
        return complex(1.0, 0.0) + z * sum(self.B[i] * w[i] for i in range(n))


def explicit_euler_step(f, t, y, h):
    derivative = f(t, y)
    return [a + h * b for a, b in zip(y, derivative)]


def run(method, f, t0, y0, t_end, h):
    """Fixed-step run; returns (final value, total Newton iterations)."""
    y = list(y0)
    t = t0
    steps = int(round((t_end - t0) / h))
    iterations = 0
    for _ in range(steps):
        y, history, used, _stages = method.step(f, t, y, h)
        iterations += used
        t += h
    return y, iterations


def run_explicit(f, t0, y0, t_end, h):
    y = list(y0)
    t = t0
    for _ in range(int(round((t_end - t0) / h))):
        y = explicit_euler_step(f, t, y, h)
        t += h
    return y


def rk4_reference(f, t0, y0, t_end, steps):
    """Independent fixed-step RK4 used as the high-resolution reference."""
    y = list(y0)
    h = (t_end - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, [a + h / 2 * b for a, b in zip(y, k1)])
        k3 = f(t + h / 2, [a + h / 2 * b for a, b in zip(y, k2)])
        k4 = f(t + h, [a + h * b for a, b in zip(y, k3)])
        y = [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(y, k1, k2, k3, k4)]
        t += h
    return y


# --------------------------------------------------------------- test problems
def f_stiff_linear(t, y):
    """y' = -1000 (y - cos t) - sin t, exact solution y = cos t."""
    return [-1000.0 * (y[0] - math.cos(t)) - math.sin(t)]


def f_stiff_nonlinear(t, y):
    """y' = -1000 y^2, exact solution y = 1 / (1 + 1000 t)."""
    return [-1000.0 * y[0] * y[0]]


def f_rotation(a, b):
    """Real 2x2 form of the complex test equation y' = (a + i b) y."""
    return lambda t, y: [a * y[0] - b * y[1], b * y[0] + a * y[1]]


def main():
    euler = ImplicitRungeKutta(IMPLICIT_EULER)
    sdirk2 = ImplicitRungeKutta(ALEXANDER_SDIRK2)
    check(abs(1.0 - 2.0 * GAMMA - 2.0 * GAMMA * (1.0 - GAMMA)) < 1e-15,
          "the L-stability identity 1 - 2*gamma = 2*gamma*(1-gamma) must hold exactly")
    check(abs(sum(ALEXANDER_SDIRK2[3]) - 1.0) < 1e-15,
          "the SDIRK2 weights must sum to one")

    # --- 1. the closed-form stability functions -------------------------------
    for x in (0.5, 1.0, 5.0, 100.0):
        z = complex(-x, 0.0)
        check(abs(euler.amplification(z) - 1.0 / (1.0 - z)) < 1e-13,
              "implicit Euler R(z) must equal 1/(1-z) at z=%s" % (z,))
        d = 1.0 - GAMMA * z
        expected = 1.0 + z * (1.0 - GAMMA * z + (1.0 - 2.0 * GAMMA) * z / 2.0) / d ** 2
        check(abs(sdirk2.amplification(z) - expected) < 1e-13,
              "SDIRK2 R(z) must equal the closed form at z=%s" % (z,))
    check(abs(sdirk2.amplification(complex(-1e8, 0.0))) < 1e-7
          and abs(euler.amplification(complex(-1e8, 0.0))) < 1e-7,
          "L-stability: R(z) must vanish as z -> -infinity for both methods")

    # --- 2. A-stability over a grid of the left half-plane --------------------
    grid_violations = {"euler_method": 0, "sdirk2": 0, "explicit": 0}
    for i in range(-10, 1):
        for j in range(-10, 11):
            z = complex(i / 2.0, j / 2.0)
            if i > 0:
                continue
            grid_violations["euler_method"] += abs(euler.amplification(z)) > 1.0 + 1e-12
            grid_violations["sdirk2"] += abs(sdirk2.amplification(z)) > 1.0 + 1e-12
            grid_violations["explicit"] += abs(1.0 + z) > 1.0 + 1e-12
    check(grid_violations["euler_method"] == 0 and grid_violations["sdirk2"] == 0,
          "both methods must be A-stable on the grid, violations %r" % (grid_violations,))
    check(grid_violations["explicit"] > 0,
          "explicit Euler must violate A-stability somewhere on the grid")

    # --- 3. the empirical one-step amplification equals R(h*lambda) -----------
    lam = -50.0
    for h in (0.005, 0.02, 0.1):
        for method in (euler, sdirk2):
            y_new, history, used, _ = method.step(lambda t, y: [lam * y[0]], 0.0, [1.0], h)
            predicted = method.amplification(complex(h * lam, 0.0)).real
            check(abs(y_new[0] - predicted) < 1e-11,
                  "%s amplification mismatch at h=%g: %.14f vs %.14f"
                  % (method.name, h, y_new[0], predicted))
    rotation = f_rotation(-2.0, 6.0)
    y_rot, _h, _u, _s = sdirk2.step(rotation, 0.0, [1.0, 0.0], 0.05)
    predicted = sdirk2.amplification(complex(-0.1, 0.3))
    check(abs(math.hypot(y_rot[0], y_rot[1]) - abs(predicted)) < 1e-12,
          "the 2x2 real rotation system must reproduce |R(h*lambda)| for complex lambda")
    check(abs(math.atan2(y_rot[1], y_rot[0]) - math.atan2(predicted.imag, predicted.real)) < 1e-12,
          "the rotation angle must equal arg R(h*lambda)")

    # --- 4. Newton: residual, iteration count, quadratic convergence ----------
    y_new, history, used, _ = sdirk2.step(f_stiff_linear, 0.0, [1.0], 0.05)
    check(history[-1] < 1e-13, "the converged Newton residual must be below 1e-13, got %.3e"
          % history[-1])
    check(used <= 8, "Newton must converge in a handful of iterations, used %d" % used)
    check(len(history) >= 3, "the residual history must expose the Newton iterations")
    quadratic = [history[i + 1] / max(history[i] ** 2, 1e-300)
                 for i in range(1, len(history) - 1) if history[i] > 1e-12]
    check(all(r < 1e6 for r in quadratic),
          "the residual must decrease at least quadratically: %r" % (history,))
    y_euler, hist_euler, used_euler, stage_euler = euler.step(f_stiff_linear, 0.0, [1.0], 0.05)
    defining_equation = stage_euler[0] - 1.0 - 0.05 * f_stiff_linear(0.05, [stage_euler[0]])[0]
    check(abs(defining_equation) < 1e-14,
          "implicit Euler's stage must satisfy its own defining equation exactly, residual %.3e"
          % defining_equation)

    # --- 5. stiff smooth problem: accuracy and the explicit failure ----------
    t_end = 1.0
    inexact = {}
    for h in (0.02, 0.005, 0.001):
        y_e, _ = run(euler, f_stiff_linear, 0.0, [1.0], t_end, h)
        y_s, _ = run(sdirk2, f_stiff_linear, 0.0, [1.0], t_end, h)
        inexact[h] = (abs(y_e[0] - math.cos(t_end)), abs(y_s[0] - math.cos(t_end)))
    check(all(err < 0.05 for _e, err in inexact.values()),
          "SDIRK2 must stay close to cos t on the stiff problem: %r" % (inexact,))
    check(inexact[0.02][0] > inexact[0.005][0] > inexact[0.001][0],
          "implicit Euler must improve with smaller h: %r" % (inexact,))
    euler_ratio = inexact[0.005][0] / inexact[0.001][0]
    sdirk_ratio = inexact[0.005][1] / inexact[0.001][1]
    check(4.0 < euler_ratio < 6.5,
          "implicit Euler must converge at order 1 (error ratio ~5), got %.2f" % euler_ratio)
    check(20.0 < sdirk_ratio < 35.0,
          "SDIRK2 must converge at order 2 (error ratio ~25), got %.2f" % sdirk_ratio)
    explicit_blowup = run_explicit(f_stiff_linear, 0.0, [1.0], t_end, 0.02)
    check(abs(explicit_blowup[0]) > 1e3,
          "explicit Euler must blow up at h=0.02 on the stiff problem, got %.3e"
          % explicit_blowup[0])
    check(abs(explicit_blowup[0]) > 1e6 * inexact[0.02][0],
          "the L-stable step must succeed where the explicit step is unusable")

    # --- 6. nonlinear stiff problem against the closed form and RK4 ----------
    t_nl = 0.01
    exact_nl = 1.0 / (1.0 + 1000.0 * t_nl)
    y_nl, _ = run(sdirk2, f_stiff_nonlinear, 0.0, [1.0], t_nl, 2.5e-4)
    reference_nl = rk4_reference(f_stiff_nonlinear, 0.0, [1.0], t_nl, 20000)
    nonlinear_errors = []
    for h in (1e-3, 5e-4, 2.5e-4):
        y_e, _ = run(euler, f_stiff_nonlinear, 0.0, [1.0], t_nl, h)
        y_s, _ = run(sdirk2, f_stiff_nonlinear, 0.0, [1.0], t_nl, h)
        nonlinear_errors.append((h, abs(y_e[0] - exact_nl), abs(y_s[0] - exact_nl)))
    check(abs(reference_nl[0] - exact_nl) < 1e-9,
          "the RK4 reference must match the closed form, err %.3e"
          % abs(reference_nl[0] - exact_nl))
    check(abs(y_nl[0] - reference_nl[0]) < 1e-3,
          "SDIRK2 must agree with the RK4 reference on the nonlinear stiff problem, err %.3e"
          % abs(y_nl[0] - reference_nl[0]))
    check(all(err < 0.02 for _h, _e, err in nonlinear_errors),
          "SDIRK2 nonlinear errors must stay small: %r" % (nonlinear_errors,))
    check(all(nonlinear_errors[i][2] < nonlinear_errors[i][1]
              for i in range(len(nonlinear_errors))),
          "SDIRK2 must beat implicit Euler on the nonlinear stiff problem too: %r"
          % (nonlinear_errors,))
    explicit_nonlinear = run_explicit(f_stiff_nonlinear, 0.0, [1.0], t_nl, 1e-3)
    check(abs(explicit_nonlinear[0] - exact_nl) > abs(nonlinear_errors[0][1]),
          "explicit Euler must be less accurate than implicit Euler at h=1e-3 on y'=-1000y^2")

    # --- 7. boundary cases and rejected tableaus -----------------------------
    y_zero, _h0, _u0, _s0 = euler.step(f_stiff_linear, 0.3, [0.7], 0.0)
    check(y_zero == [0.7], "a zero-length step must leave the state untouched")
    tiny_h = 1e-12
    y_small, _hs, _us, _ss = sdirk2.step(f_stiff_linear, 0.3, [0.7], tiny_h)
    movement = abs(y_small[0] - 0.7)
    slope = abs(f_stiff_linear(0.3, [0.7])[0])
    check(movement < 2.0 * tiny_h * slope,
          "a tiny step must move the state by about h*f, measured %.3e vs %.3e"
          % (movement, tiny_h * slope))
    rejections = 0
    for bad in (lambda: ImplicitRungeKutta(("bad", ((1.0,),), (0.5,), (1.0,))),
                lambda: ImplicitRungeKutta(("bad", ((1.0, 0.0),), (1.0, 1.0), (1.0, 1.0))),
                lambda: euler.step(f_stiff_linear, 0.0, [1.0], -0.01)):
        try:
            bad()
        except ValueError:
            rejections += 1
    check(rejections == 3,
          "inconsistent tableaus and negative step sizes must be rejected, got %d" % rejections)

    print("implicit Euler R(-1e8)=%.3e, SDIRK2 R(-1e8)=%.3e"
          % (abs(euler.amplification(complex(-1e8, 0.0))),
             abs(sdirk2.amplification(complex(-1e8, 0.0)))))
    print("A-stability grid violations: %r" % (grid_violations,))
    print("stiff smooth errors (h, implicit Euler, SDIRK2): %r"
          % [(h, format(e, '.2e'), format(s, '.2e')) for h, (e, s) in sorted(inexact.items())])
    print("observed error ratios for h: 0.005 -> 0.001 : implicit Euler %.2f, SDIRK2 %.2f"
          % (euler_ratio, sdirk_ratio))
    print("explicit Euler at h=0.02 blows up to %.3e" % explicit_blowup[0])
    print("nonlinear stiff errors (h, implicit Euler, SDIRK2) vs exact %.6f: %r"
          % (exact_nl, [(h, format(a, '.2e'), format(b, '.2e')) for h, a, b in nonlinear_errors]))
    print("SDIRK2 vs RK4 reference on y'=-1000y^2: %.3e" % abs(y_nl[0] - reference_nl[0]))
    print("PASS")


if __name__ == "__main__":
    main()
