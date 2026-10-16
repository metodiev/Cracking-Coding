"""Task 787 - Implement adaptive step-size ODE integrator (Dormand-Prince outline).  (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)

The Dormand-Prince pair DP5(4) advances dy/dt = f(t, y) with two embedded formulas that
share all seven stage evaluations: a fifth-order solution y5 (whose weights are exactly the
seventh row of the tableau) and a fourth-order solution y4.  The local error estimate is
e = h * sum_i (b5_i - b4_i) k_i, and because b5 equals the last tableau row the seventh
stage is evaluated at (t + h, y5), which is *First Same As Last*: that evaluation is reused
as k1 of the next step, so an accepted step costs six function evaluations, not seven.

Step-size control uses the standard scaled RMS norm
    err_norm = sqrt( mean_i ( e_i / (atol + rtol * |y_i|) )^2 )
with the PI-free elementary controller h_new = h * clip(safety * err_norm^(-1/5), 1/5, 5).
Rejected steps redo the step with the reduced h and must recompute k1 (the reuse is only
valid for the h that produced it), which the function-evaluation accounting in this file
tracks exactly.

Verified against closed-form solutions with measured tolerances: the textbook problem
y' = y - t^2 + 1 with y = (t+1)^2 - 0.5 e^t, a pure decay y' = -lambda y, and the two-body
circular orbit y'' = -y/|y|^3 integrated over a full period against a high-resolution
(rtol = 1e-13) reference.  The fixed-step variant is checked for fifth order, the step-size
sequence is checked to grow once the solution decays under atol, and the first step on a
stiff problem is shown to be limited by the explicit stability boundary (h * lambda <= 3.4,
the real-axis limit of DP5).

Outline-level simplifications: no dense output/interpolation, no PI or Gustafsson control,
no dense or sparse linear algebra (the methods are explicit), and no event detection.
"""
import math

C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
A = ((), (1.0 / 5.0,),
     (3.0 / 40.0, 9.0 / 40.0),
     (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
     (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
     (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
     (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0))
B5 = A[6] + (0.0,)                                       # the 7th weight is zero (FSAL stage)
B4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0,
      187.0 / 2100.0, 1.0 / 40.0)
E = tuple(b5 - b4 for b5, b4 in zip(B5, B4))
DP5_REAL_STABILITY_LIMIT = 3.306                            # |R(-x)| = 1 for the 5th-order weights


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def scaled_error_norm(err, y, y_new, rtol, atol):
    """RMS norm of err / (atol + rtol * max(|y|, |y_new|)), the standard DP5 criterion."""
    total = 0.0
    for e, a, b in zip(err, y, y_new):
        scale = atol + rtol * max(abs(a), abs(b))
        total += (e / scale) ** 2
    return math.sqrt(total / len(err))


def dopri5_step(f, t, y, h, k1):
    """One DP5(4) step with k1 = f(t, y) supplied by the caller (FSAL).

    Returns (y5, y4, err, last_stage, evaluations).
    """
    stages = [k1]
    evaluations = 0
    for i in range(1, 7):
        row = A[i]
        yi = [y[j] + h * math.fsum(row[m] * stages[m][j] for m in range(len(row)))
              for j in range(len(y))]
        stages.append(f(t + C[i] * h, yi))
        evaluations += 1
    y5 = [y[j] + h * math.fsum(B5[i] * stages[i][j] for i in range(7)) for j in range(len(y))]
    y4 = [y[j] + h * math.fsum(B4[i] * stages[i][j] for i in range(7)) for j in range(len(y))]
    err = [h * math.fsum(E[i] * stages[i][j] for i in range(7)) for j in range(len(y))]
    return y5, y4, err, stages[6], evaluations


class DormandPrince:
    """Adaptive DP5(4) driver with FSAL reuse and elementary step-size control."""

    def __init__(self, f, rtol=1e-8, atol=1e-10, safety=0.9, h_init=1e-3,
                 h_min=1e-12, h_max=0.5, max_factor=5.0, max_steps=200000):
        if rtol <= 0.0 or atol < 0.0:
            raise ValueError("rtol must be positive and atol non-negative")
        if not 0.0 < safety < 1.0:
            raise ValueError("safety must lie in (0, 1)")
        if h_init <= 0.0 or h_min <= 0.0 or h_max < h_min:
            raise ValueError("inconsistent step-size bounds")
        self.f = f
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        self.h_init = h_init
        self.h_min = h_min
        self.h_max = h_max
        self.max_factor = max_factor
        self.max_steps = max_steps

    def integrate(self, t0, y0, t_end, fixed_h=None):
        """Integrate to t_end; returns (t, y, stats).  fixed_h disables the controller."""
        if t_end < t0:
            raise ValueError("t_end must not precede t0")
        y = list(y0)
        t = float(t0)
        if t_end == t0:
            return t, y, {"accepted": 0, "rejected": 0, "evaluations": 0, "steps": [], "errors": []}
        stats = {"accepted": 0, "rejected": 0, "evaluations": 0, "steps": [], "errors": []}
        h = min(fixed_h if fixed_h is not None else self.h_init, t_end - t0)
        k1 = self.f(t, y)
        stats["evaluations"] += 1
        while t < t_end:
            h = min(h, t_end - t)                     # never step past the end point
            y5, _y4, err, last, evals = dopri5_step(self.f, t, y, h, k1)
            stats["evaluations"] += evals
            norm = scaled_error_norm(err, y, y5, self.rtol, self.atol)
            if norm <= 1.0 or fixed_h is not None:
                if fixed_h is not None:
                    factor = 1.0                       # fixed-step mode disables the controller
                else:
                    factor = min(self.max_factor,
                                 max(1.0, self.safety * norm ** (-1.0 / 5.0)))
                y, t = y5, t + h
                k1 = last                            # FSAL: reuse the last stage
                stats["accepted"] += 1
                stats["steps"].append((t, h))
                stats["errors"].append(norm)
                h = h * factor
            else:
                factor = max(0.2, self.safety * norm ** (-1.0 / 5.0))
                h = h * factor
                k1 = self.f(t, y)                    # rejected h invalidates the reused stage
                stats["evaluations"] += 1
                stats["rejected"] += 1
                if h < self.h_min:
                    raise RuntimeError("step size fell below h_min")
            if stats["accepted"] + stats["rejected"] > self.max_steps:
                raise RuntimeError("step budget exhausted")
        return t, y, stats


# --------------------------------------------------------------- test problems
def f_textbook(t, y):
    return [y[0] - t * t + 1.0]


def exact_textbook(t):
    return (t + 1.0) ** 2 - 0.5 * math.exp(t)


def f_decay(lam):
    return lambda t, y: [-lam * y[0]]


def f_stiff(t, y):
    """Exact solution cos(t) - a stiff, smooth problem for the stability check."""
    return [-1000.0 * (y[0] - math.cos(t)) - math.sin(t)]


def f_orbit(t, y):
    """Circular two-body orbit: y = (x, vx, y, vy) with |y_pos| = 1."""
    r2 = y[0] * y[0] + y[2] * y[2]
    r3 = r2 ** 1.5
    return [y[1], -y[0] / r3, y[3], -y[2] / r3]


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


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    TOL_LADDER = (1e-4, 1e-6, 1e-8, 1e-10)

    # --- 1. accuracy on the textbook problem, and tolerance responsiveness -----
    errors = []
    accepted_counts = []
    for tol in TOL_LADDER:
        solver = DormandPrince(f_textbook, rtol=tol, atol=tol * 1e-4, h_init=1e-3)
        t, y, stats = solver.integrate(0.0, [0.5], 2.0)
        errors.append(abs(y[0] - exact_textbook(t)))
        accepted_counts.append(stats["accepted"])
    check(all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)),
          "max error must shrink monotonically with the tolerance: %r" % (errors,))
    check(all(e < 60.0 * tol for e, tol in zip(errors, TOL_LADDER)),
          "achieved error must track the requested tolerance: %r" % (errors,))
    check(all(accepted_counts[i] < accepted_counts[i + 1] for i in range(len(accepted_counts) - 1)),
          "looser tolerances must need fewer steps: %r" % (accepted_counts,))
    check(errors[-1] < 1e-8, "rtol=1e-10 must reach an error below 1e-8, got %.3e" % errors[-1])

    # --- 2. fifth order in the fixed-step limit --------------------------------
    fixed_errors = []
    for steps in (10, 20, 40):
        solver = DormandPrince(f_textbook, rtol=1e-3, h_init=1.0)
        t, y, _ = solver.integrate(0.0, [0.5], 2.0, fixed_h=2.0 / steps)
        fixed_errors.append(abs(y[0] - exact_textbook(t)))
    orders = [math.log(fixed_errors[i] / fixed_errors[i + 1], 2.0) for i in range(2)]
    check(all(abs(o - 5.0) < 0.4 for o in orders),
          "DP5 fixed-step order must be ~5, got %r" % (orders,))

    # --- 3. FSAL bookkeeping: evaluations = 7 + 6*accepted + 7*rejected --------
    solver = DormandPrince(f_textbook, rtol=1e-7, atol=1e-12, h_init=0.4, h_max=0.3)
    t, y, fsal_stats = solver.integrate(0.0, [0.5], 4.0)
    expected = 7 + 6 * (fsal_stats["accepted"] - 1) + 7 * fsal_stats["rejected"]
    check(fsal_stats["evaluations"] == expected,
          "function evaluations must follow the FSAL rule: %d vs %d"
          % (fsal_stats["evaluations"], expected))
    check(fsal_stats["rejected"] > 0,
          "the deliberately large initial step must be rejected at least once")

    # --- 4. the step sequence grows once atol dominates ------------------------
    solver = DormandPrince(f_decay(300.0), rtol=1e-6, atol=1e-10, h_init=0.01, h_max=0.5)
    t, y, stats = solver.integrate(0.0, [1.0], 2.0)
    step_sizes = [h for _t, h in stats["steps"]]
    check(step_sizes[-1] > 10.0 * step_sizes[0],
          "steps must grow when the decaying solution falls under atol: %.3e -> %.3e"
          % (step_sizes[0], step_sizes[-1]))
    check(step_sizes[0] * 300.0 <= 3.4,
          "the first step must respect the explicit stability limit, h*lambda=%.3f"
          % (step_sizes[0] * 300.0))
    check(abs(y[0]) < 1e-3, "the decayed solution must be numerically zero at t=2, got %.3e" % y[0])

    # --- 5. a high-resolution reference for the same decay ---------------------
    reference = rk4_reference(f_decay(300.0), 0.0, [1.0], 0.02, 20000)
    solver = DormandPrince(f_decay(300.0), rtol=1e-8, atol=1e-12, h_init=1e-4)
    t, y, _ = solver.integrate(0.0, [1.0], 0.02)
    decay_error = abs(y[0] - reference[0])
    check(decay_error < 1e-8,
          "adaptive solution must match the RK4 reference on the fast decay, err %.3e" % decay_error)
    check(abs(reference[0] - math.exp(-6.0)) < 1e-5,
          "the RK4 reference itself must match the closed form e^-6, got %.6f" % reference[0])

    # --- 6. stiff problem: stability-limited stepping, no blow-up --------------
    solver = DormandPrince(f_stiff, rtol=1e-6, atol=1e-9, h_init=0.05, h_max=0.2)
    t, y, stiff_stats = solver.integrate(0.0, [1.0], 1.0)
    stiff_error = abs(y[0] - math.cos(t))
    check(stiff_error < 1e-4,
          "stiff smooth problem error %.3e must stay below 1e-4" % stiff_error)
    check(stiff_stats["rejected"] > 0,
          "an over-large initial step on the stiff problem must be rejected")
    check(stiff_stats["steps"][0][1] * 1000.0 <= 3.4,
          "the accepted stiff step must satisfy h*lambda <= 3.4, measured %.4f"
          % (stiff_stats["steps"][0][1] * 1000.0))

    # --- 7. orbital problem against a very tight reference --------------------
    tight = DormandPrince(f_orbit, rtol=1e-13, atol=1e-14, h_init=1e-4, h_max=0.1)
    t_ref, y_ref, _ = tight.integrate(0.0, [1.0, 0.0, 0.0, 1.0], 2.0 * math.pi)
    loose = DormandPrince(f_orbit, rtol=1e-9, atol=1e-12, h_init=1e-4, h_max=0.1)
    t_orb, y_orb, stats_orb = loose.integrate(0.0, [1.0, 0.0, 0.0, 1.0], 2.0 * math.pi)
    orbit_error = max_abs_diff(y_orb, y_ref)
    check(orbit_error < 1e-6,
          "one orbital period must match the tight reference to 1e-6, err %.3e" % orbit_error)
    check(abs(math.hypot(y_orb[0], y_orb[2]) - 1.0) < 1e-8,
          "the orbit must stay on the unit circle, radius %.12f" % math.hypot(y_orb[0], y_orb[2]))
    check(max_abs_diff(y_ref, [1.0, 0.0, 0.0, 1.0]) < 1e-9,
          "the tight reference must return to the initial state after one period")
    check(abs(t_ref - 2.0 * math.pi) < 1e-15 and abs(t_orb - 2.0 * math.pi) < 1e-15,
          "both runs must stop exactly at the end time")

    # --- 8. boundary cases and rejected input ---------------------------------
    solver = DormandPrince(f_textbook, rtol=1e-8, atol=1e-12)
    t, y, stats = solver.integrate(1.5, [0.5], 1.5)
    check(t == 1.5 and y == [0.5] and stats["accepted"] == 0 and stats["evaluations"] == 0,
          "a zero-length interval must be a no-op")
    short = DormandPrince(f_textbook, rtol=1e-8, atol=1e-12, h_init=0.5)
    t, y, short_stats = short.integrate(0.0, [0.5], 0.05)
    check(short_stats["accepted"] == 1 and abs(t - 0.05) < 1e-15,
          "an interval shorter than h_init must be covered by a single step")
    rejections = 0
    for bad in (lambda: DormandPrince(f_textbook, rtol=0.0),
                lambda: DormandPrince(f_textbook, safety=1.5),
                lambda: DormandPrince(f_textbook, h_init=-1.0),
                lambda: DormandPrince(f_textbook).integrate(1.0, [0.0], 0.0 - 0.5)):
        try:
            bad()
        except ValueError:
            rejections += 1
    check(rejections == 4, "invalid tolerances, safety factors and intervals must be rejected")

    print("textbook errors by tolerance: %s" % [format(e, '.2e') for e in errors])
    print("fixed-step DP5 errors: %s -> orders %s"
          % ([format(e, '.2e') for e in fixed_errors], [round(o, 3) for o in orders]))
    print("FSAL run: accepted=%d rejected=%d evaluations=%d"
          % (fsal_stats["accepted"], fsal_stats["rejected"], fsal_stats["evaluations"]))
    print("decay step sizes: first=%.4e last=%.4e" % (step_sizes[0], step_sizes[-1]))
    print("stiff error=%.3e accepted=%d rejected=%d first h*lambda=%.4f"
          % (stiff_error, stiff_stats["accepted"], stiff_stats["rejected"],
             stiff_stats["steps"][0][1] * 1000.0))
    print("orbit error vs tight reference: %.3e" % orbit_error)
    print("PASS")


if __name__ == "__main__":
    main()
