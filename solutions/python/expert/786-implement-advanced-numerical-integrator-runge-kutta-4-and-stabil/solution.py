"""Task 786 - Implement advanced numerical integrator (Runge-Kutta 4) and stability tests.  (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)

Classical fourth-order Runge-Kutta advances dy/dt = f(t, y) (y a float vector) by

    k1 = f(t, y)                     k2 = f(t + h/2, y + h/2 k1)
    k3 = f(t + h/2, y + h/2 k2)      k4 = f(t + h,   y + h   k3)
    y(t + h) = y + h/6 (k1 + 2 k2 + 2 k3 + k4)

Applying one step to the linear test equation y' = lambda*y shows that the numerical
amplification factor is the stability polynomial

    R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24,   z = h * lambda,

so the *linear stability region* is {z : |R(z)| <= 1}.  Two exact facts about that region
are used as reference values: on the negative real axis |R(-x)| crosses 1 at
x* = 2.7852935634056964 (the classical RK4 step-size limit) and on the imaginary axis the
region ends exactly at 2*sqrt(2) = 2.8284271247461903.  The integration error of a
y' = lambda*y run is compared against exp(z) for the same run, which links the empirically
measured amplification to the closed-form polynomial instead of trusting the code.

Verification is against closed-form solutions with realistic tolerances: y' = -lambda*y,
y' = 2t (a cubic/quartic polynomial that RK4 integrates exactly), the textbook
y' = y - t^2 + 1 with y = (t+1)^2 - 0.5 e^t, and the harmonic oscillator y'' = -omega^2 y
whose energy 0.5(y^2 + (y'/omega)^2) is conserved exactly.  Simplifications: scalar-or-vector
first-order ODEs only (no implicit/stiff handling, no dense output, no error control - those
live in tasks 787/788), and the stability limit is found by bisection on R itself rather
than by tracing the boundary of the region in the complex plane.
"""
import math

RK4_STABILITY_LIMIT_REAL = 2.7852935634056964      # |R(-x)| = 1 on the negative real axis
RK4_STABILITY_LIMIT_IMAG = 2.8284271247461903      # |R(i*2*sqrt(2))| = 1 exactly


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------ integrators
def rk4_step(f, t, y, h):
    """One classical RK4 step for a vector state y (list of floats)."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, [yi + 0.5 * h * ki for yi, ki in zip(y, k1)])
    k3 = f(t + 0.5 * h, [yi + 0.5 * h * ki for yi, ki in zip(y, k2)])
    k4 = f(t + h, [yi + h * ki for yi, ki in zip(y, k3)])
    return [yi + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)
            for yi, a, b, c, d in zip(y, k1, k2, k3, k4)]


def euler_step(f, t, y, h):
    return [yi + h * ki for yi, ki in zip(y, f(t, y))]


def midpoint_step(f, t, y, h):
    """Explicit midpoint rule (order 2), used only to contrast the observed order."""
    k1 = f(t, y)
    half = [yi + 0.5 * h * ki for yi, ki in zip(y, k1)]
    k2 = f(t + 0.5 * h, half)
    return [yi + h * ki for yi, ki in zip(y, k2)]


def integrate(step, f, t0, y0, t_end, steps):
    """Fixed-step driver; returns the final state and the trajectory."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if t_end < t0:
        raise ValueError("t_end must not precede t0")
    y = list(y0)
    t = float(t0)
    trajectory = [(t, list(y))]
    if steps == 0:
        return y, trajectory
    h = (t_end - t0) / steps
    for _ in range(steps):
        y = step(f, t, y, h)
        t += h
        trajectory.append((t, list(y)))
    return y, trajectory


def rk4_amplification(z):
    """R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 for complex z (the exact one-step factor)."""
    return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def rk4_real_amplification(x):
    """R(-x) on the negative real axis, real arithmetic only."""
    return 1.0 - x + x * x / 2.0 - x ** 3 / 6.0 + x ** 4 / 24.0


def bisect_limit(lo, hi, f, tol=1e-13):
    """Bisection for the root of f (f(lo) and f(hi) must straddle zero)."""
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fmid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------- test problems
def f_decay(lam):
    return lambda t, y: [-lam * y[0]]


def f_poly(t, y):
    return [2.0 * t]                                     # exact y = t^2


def f_textbook(t, y):
    return [y[0] - t * t + 1.0]                          # exact y = (t+1)^2 - 0.5 e^t


def f_oscillator(omega):
    """y'' = -omega^2 y written as a first-order system."""
    return lambda t, y: [y[1], -(omega ** 2) * y[0]]


def exact_textbook(t):
    return (t + 1.0) ** 2 - 0.5 * math.exp(t)


def oscillator_energy(y, omega):
    return 0.5 * (y[0] ** 2 + (y[1] / omega) ** 2)


def max_abs_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def observed_order(errors, ratio):
    """log2-free order estimate: error ratio between successive refinements."""
    return [math.log(errors[i] / errors[i + 1], ratio) for i in range(len(errors) - 1)]


def main():
    # --- 1. consistency of the code with its stability polynomial ---------------
    lam = 3.7
    for h in (0.05, 0.2, 0.5):
        y1, _ = integrate(rk4_step, f_decay(lam), 0.0, [1.0], h, 1)
        predicted = rk4_amplification(complex(-h * lam, 0.0))
        check(abs(y1[0] - predicted.real) < 1e-13,
              "RK4 amplification must equal R(-h*lambda): measured %.16f vs R=%.16f"
              % (y1[0], predicted.real))
    # R(z) is the order-4 truncation of exp(z), so (R(z) - exp(z)) / z^5 -> -1/120.
    # Richardson on z and z/2 removes the linear term of that limit, leaving O(z^2).
    def truncation_ratio(z):
        return (rk4_amplification(complex(z, 0.0)).real - math.exp(z)) / z ** 5

    coefficient = 2.0 * truncation_ratio(0.025) - truncation_ratio(0.05)
    check(abs(coefficient + 1.0 / 120.0) < 2e-6,
          "R(z) - exp(z) must be -z^5/120 + O(z^6), got z^5 coefficient %.9f" % coefficient)

    # --- 2. fourth-order convergence on y' = -lambda y -------------------------
    exact = math.exp(-1.0)
    errors_rk4 = [abs(integrate(rk4_step, f_decay(1.0), 0.0, [1.0], 1.0, n)[0][0] - exact)
                  for n in (10, 20, 40)]
    orders_rk4 = observed_order(errors_rk4, 2.0)
    check(all(abs(o - 4.0) < 0.25 for o in orders_rk4),
          "RK4 observed order must be ~4, got %r" % (orders_rk4,))
    errors_euler = [abs(integrate(euler_step, f_decay(1.0), 0.0, [1.0], 1.0, n)[0][0] - exact)
                    for n in (10, 20, 40)]
    orders_euler = observed_order(errors_euler, 2.0)
    check(all(abs(o - 1.0) < 0.15 for o in orders_euler),
          "explicit Euler observed order must be ~1, got %r" % (orders_euler,))
    errors_mid = [abs(integrate(midpoint_step, f_decay(1.0), 0.0, [1.0], 1.0, n)[0][0] - exact)
                  for n in (10, 20, 40)]
    check(all(abs(o - 2.0) < 0.2 for o in observed_order(errors_mid, 2.0)),
          "explicit midpoint observed order must be ~2, got %r" % (observed_order(errors_mid, 2.0),))

    # --- 3. exactness on polynomials and the textbook problem ------------------
    y_poly, _ = integrate(rk4_step, f_poly, 0.0, [0.0], 3.0, 3)
    check(abs(y_poly[0] - 9.0) < 1e-13, "RK4 must integrate y' = 2t exactly, got %r" % (y_poly[0],))
    y_text, _ = integrate(rk4_step, f_textbook, 0.0, [0.5], 2.0, 40)
    textbook_err = abs(y_text[0] - exact_textbook(2.0))
    check(textbook_err < 1e-6,
          "textbook problem error %.3e must be below 1e-6" % textbook_err)
    y_text_coarse, _ = integrate(rk4_step, f_textbook, 0.0, [0.5], 2.0, 20)
    textbook_err_coarse = abs(y_text_coarse[0] - exact_textbook(2.0))
    check(textbook_err_coarse > 8.0 * textbook_err,
          "halving h must cut the error by roughly 16x: %.3e -> %.3e"
          % (textbook_err_coarse, textbook_err))

    # --- 4. energy behaviour of the harmonic oscillator ------------------------
    omega = 2.0
    traj_euler = integrate(euler_step, f_oscillator(omega), 0.0, [1.0, 0.0], 20.0, 2000)[1]
    euler_drift = abs(oscillator_energy(traj_euler[-1][1], omega) / oscillator_energy([1.0, 0.0], omega) - 1.0)
    traj_rk4 = integrate(rk4_step, f_oscillator(omega), 0.0, [1.0, 0.0], 20.0, 2000)[1]
    rk4_drift = abs(oscillator_energy(traj_rk4[-1][1], omega) / oscillator_energy([1.0, 0.0], omega) - 1.0)
    check(rk4_drift < 1e-6,
          "RK4 energy drift %.3e over t=20 must stay below 1e-6" % rk4_drift)
    check(euler_drift > 0.1,
          "explicit Euler must visibly gain energy, measured %.3e" % euler_drift)
    check(rk4_drift < euler_drift / 100.0,
          "RK4 must beat Euler on energy conservation by orders of magnitude")

    # --- 5. the real-axis stability limit, against the published constant ------
    real_limit = bisect_limit(2.6, 3.0, lambda x: rk4_real_amplification(x) - 1.0)
    check(abs(real_limit - RK4_STABILITY_LIMIT_REAL) < 1e-9,
          "bisected limit %.13f must match %.13f" % (real_limit, RK4_STABILITY_LIMIT_REAL))
    check(abs(abs(rk4_real_amplification(real_limit)) - 1.0) < 1e-12,
          "|R| must equal 1 exactly at the step-size limit")
    check(rk4_real_amplification(real_limit + 0.5) > 1.0
          and rk4_real_amplification(real_limit - 0.5) < 1.0,
          "|R| must exceed 1 above the limit and stay below it underneath")

    # --- 6. imaginary-axis limit: 2*sqrt(2), and the matching oscillator run ---
    imag_limit = bisect_limit(2.7, 2.9, lambda x: abs(rk4_amplification(complex(0.0, x))) - 1.0)
    check(abs(imag_limit - RK4_STABILITY_LIMIT_IMAG) < 1e-9,
          "bisected imaginary limit %.13f must match 2*sqrt(2)=%.13f"
          % (imag_limit, RK4_STABILITY_LIMIT_IMAG))
    below = integrate(rk4_step, f_oscillator(1.0), 0.0, [1.0, 0.0], 280.0, 100)[1]
    above = integrate(rk4_step, f_oscillator(1.0), 0.0, [1.0, 0.0], 280.0, 98)[1]
    ratio_below = oscillator_energy(below[-1][1], 1.0) / oscillator_energy([1.0, 0.0], 1.0)
    ratio_above = oscillator_energy(above[-1][1], 1.0) / oscillator_energy([1.0, 0.0], 1.0)
    check(ratio_below < 1.0,
          "energy must decay when h*omega=280/100=%.3f sits inside the region, got %.3e"
          % (280.0 / 100.0, ratio_below))
    check(ratio_above > 1e5,
          "energy must blow up when h*omega=280/98=%.4f sits outside the region, got %.3e"
          % (280.0 / 98.0, ratio_above))

    # --- 7. stiffness: the step-size limit forbids a large explicit step -------
    stiff_lam = 1000.0
    h_safe = 2.7 / stiff_lam
    h_too_big = 2.9 / stiff_lam
    steps_safe = int(round(0.5 / h_safe))
    steps_big = int(round(0.5 / h_too_big))
    y_safe, _ = integrate(rk4_step, f_decay(stiff_lam), 0.0, [1.0], 0.5, steps_safe)
    y_big, _ = integrate(rk4_step, f_decay(stiff_lam), 0.0, [1.0], 0.5, steps_big)
    check(abs(y_safe[0]) < 1e-3,
          "h*lambda=2.7 must decay (exact value ~0), got %.3e" % abs(y_safe[0]))
    check(abs(y_big[0]) > 1e3,
          "h*lambda=2.9 must be unstable, got %.3e" % abs(y_big[0]))
    check(abs(rk4_real_amplification(2.9)) ** steps_big > 1e3,
          "the polynomial must predict the blow-up observed on the stiff problem")

    # --- 8. boundary and rejection behaviour ----------------------------------
    y_zero, traj_zero = integrate(rk4_step, f_decay(1.0), 0.0, [1.0], 1.0, 0)
    check(y_zero == [1.0] and len(traj_zero) == 1,
          "a zero-step integration must return the initial state unchanged")
    y_one, traj_one = integrate(rk4_step, f_decay(1.0), 0.0, [1.0], 0.25, 1)
    check(len(traj_one) == 2 and abs(traj_one[-1][0] - 0.25) < 1e-15,
          "a single step must land exactly on t_end")
    rejections = 0
    for bad in (lambda: integrate(rk4_step, f_decay(1.0), 0.0, [1.0], 1.0, -1),
                lambda: integrate(rk4_step, f_decay(1.0), 0.0, [1.0], -1.0, 10)):
        try:
            bad()
        except ValueError:
            rejections += 1
    check(rejections == 2, "negative step counts and reversed intervals must be rejected")

    print("RK4 order (errors %s): %s" % ([format(e, '.2e') for e in errors_rk4],
                                         [round(o, 3) for o in orders_rk4]))
    print("stability limits: real=%.13f (ref %.13f), imaginary=%.13f (ref %.13f)"
          % (real_limit, RK4_STABILITY_LIMIT_REAL, imag_limit, RK4_STABILITY_LIMIT_IMAG))
    print("textbook problem max error at n=40: %.3e" % textbook_err)
    print("oscillator energy drift after t=20: RK4=%.3e Euler=%.3e" % (rk4_drift, euler_drift))
    print("PASS")


if __name__ == "__main__":
    main()
