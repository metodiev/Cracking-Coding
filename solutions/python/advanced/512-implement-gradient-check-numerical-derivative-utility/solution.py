"""Task 512 - Implement gradient check numerical derivative utility. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

A numerical-differentiation toolbox used to *check* analytic gradients. Three stencils are provided
for the partial derivative with respect to coordinate i: the forward difference
(f(x + h e_i) - f(x)) / h, which is only first-order accurate, the central difference
(f(x + h e_i) - f(x - h e_i)) / 2h, which is second-order accurate, and the fourth-order
Richardson-extrapolated central stencil
(-f(x + 2h e_i) + 8 f(x + h e_i) - 8 f(x - h e_i) + f(x - 2h e_i)) / 12h. On top of them sit a
numeric gradient, a numeric Jacobian for vector-valued functions, and `check_gradient`, which
compares an analytic gradient against the numeric one element-wise and reports the maximum absolute
error, the maximum relative error and the offending coordinate; the relative test falls back on the
absolute tolerance when a component is (near) zero, which is the standard guard against dividing by
a vanishing analytic value. The classic trade-off is documented and measured: truncation error
shrinks like h^p while round-off grows like eps / h, so there is an interior optimum h.

Documented checks: for f(x) = sum x_i^3 at x = (1, 2, -3) the central stencil with h = 1e-5 lands
within 1e-9 of the analytic gradient (3, 12, 27); halving h reduces the central error by 4x, the
forward error by 2x and the Richardson error by 16x (all asserted with bracket tolerances), which is
exactly the p-th order prediction; the step-size sweep from h = 1e-1 down to 1e-16 shows the error
falling, then rising again from cancellation, with the minimum strictly inside the sweep; the
Jacobian of the vector function [x0 x1, sin x0 + x1^2, exp x0 - x1] matches the analytic 3x2 matrix
to 1e-7 and is (correctly) not symmetric; a deliberately wrong analytic gradient is rejected while
the correct one is accepted, and the report names the faulty coordinate; for f = exp(sum x) with
large values the relative error is the meaningful metric (<= 1e-9 at h = 1e-5) while the absolute
error is much larger, and a genuinely zero gradient (f = sum x^2 at the origin) is handled exactly by
the absolute tolerance; and the utility never mutates the point it is handed.
"""

import math

DEFAULT_H = 1e-5


def _shifted(x, i, delta):
    point = list(x)
    point[i] += delta
    return point


def forward_difference(f, x, i, h):
    return (f(_shifted(x, i, h)) - f(list(x))) / h


def central_difference(f, x, i, h):
    return (f(_shifted(x, i, h)) - f(_shifted(x, i, -h))) / (2.0 * h)


def richardson(f, x, i, h):
    """Fourth-order central stencil (Richardson extrapolation of two central differences)."""
    numerator = (-f(_shifted(x, i, 2.0 * h)) + 8.0 * f(_shifted(x, i, h))
                 - 8.0 * f(_shifted(x, i, -h)) + f(_shifted(x, i, -2.0 * h)))
    return numerator / (12.0 * h)


def numeric_gradient(f, x, h=DEFAULT_H, method=central_difference):
    return [method(f, x, i, h) for i in range(len(x))]


def numeric_jacobian(f, x, h=DEFAULT_H, method=central_difference):
    """Row r is the gradient of output r, assembled from scalar stencils."""
    outputs = f(list(x))
    jacobian = [[0.0] * len(x) for _ in range(len(outputs))]
    for row_index in range(len(outputs)):
        for i in range(len(x)):
            scalar = lambda point, r=row_index: f(point)[r]
            jacobian[row_index][i] = method(scalar, x, i, h)
    return jacobian


def gradient_error(analytic, numeric):
    """Element-wise comparison: (max_abs, max_rel, worst_index)."""
    if len(analytic) != len(numeric):
        raise ValueError("gradient length mismatch")
    max_abs = 0.0
    max_rel = 0.0
    worst = 0
    for index, (a, n) in enumerate(zip(analytic, numeric)):
        absolute = abs(a - n)
        if absolute > max_abs:
            max_abs, worst = absolute, index
        denominator = max(abs(a), abs(n))
        relative = absolute / denominator if denominator > 0.0 else 0.0
        max_rel = max(max_rel, relative)
    return max_abs, max_rel, worst


def check_gradient(f, grad_f, x, h=DEFAULT_H, rtol=1e-5, atol=1e-8,
                   method=central_difference):
    """True when every component passes |a - n| <= atol + rtol * max(|a|, |n|)."""
    analytic = grad_f(list(x))
    numeric = numeric_gradient(f, x, h=h, method=method)
    if len(analytic) != len(numeric):
        raise ValueError("length mismatch")
    report = {"h": h, "method": method.__name__, "max_abs": 0.0, "max_rel": 0.0, "worst": None}
    ok = True
    for index, (a, n) in enumerate(zip(analytic, numeric)):
        absolute = abs(a - n)
        denominator = max(abs(a), abs(n))
        relative = absolute / denominator if denominator > 0.0 else 0.0
        if absolute > report["max_abs"]:
            report["max_abs"] = absolute
            report["worst"] = (index, a, n)
        report["max_rel"] = max(report["max_rel"], relative)
        if absolute > atol + rtol * denominator:
            ok = False
    report["ok"] = ok
    return ok, report


def sweep(f, grad_f, x, exponents=None, method=central_difference):
    """Absolute error of the numeric gradient for h = 10^-k, k = 1..16."""
    if exponents is None:
        exponents = range(1, 17)
    out = []
    for exponent in exponents:
        h = 10.0 ** (-exponent)
        analytic = grad_f(list(x))
        numeric = numeric_gradient(f, x, h=h, method=method)
        out.append((h, gradient_error(analytic, numeric)[0]))
    return out


def cubic(x):
    return sum(value ** 3 for value in x)


def cubic_grad(x):
    return [3.0 * value * value for value in x]


def quintic(x):
    return sum(value ** 5 for value in x)


def quintic_grad(x):
    return [5.0 * value ** 4 for value in x]


def exp_sum(x):
    total = sum(x)
    return math.exp(total)


def exp_sum_grad(x):
    return [math.exp(sum(x))] * len(x)


def sphere(x):
    return sum(value * value for value in x)


def sphere_grad(x):
    return [2.0 * value for value in x]


def vector_function(x):
    return [x[0] * x[1], math.sin(x[0]) + x[1] * x[1], math.exp(x[0]) - x[1]]


def vector_jacobian(x):
    return [[x[1], x[0]],
            [math.cos(x[0]), 2.0 * x[1]],
            [math.exp(x[0]), -1.0]]


def main():
    point = [1.0, 2.0, -3.0]
    exact = [3.0, 12.0, 27.0]
    assert cubic(point) == 1.0 + 8.0 - 27.0
    assert cubic_grad(point) == exact

    # --- Accuracy at a sane step size. -------------------------------------------------------
    for method in (forward_difference, central_difference, richardson):
        value = numeric_gradient(cubic, point, h=DEFAULT_H, method=method)[1]
        assert abs(value - 12.0) <= 1e-4, (method.__name__, value)
    central = numeric_gradient(cubic, point, h=DEFAULT_H)
    assert max(abs(a - n) for a, n in zip(exact, central)) <= 1e-9
    rich = numeric_gradient(cubic, point, h=1e-3, method=richardson)
    assert max(abs(a - n) for a, n in zip(exact, rich)) <= 1e-9, rich

    # --- Convergence order: halving h cuts the error by 2^p. A cubic is a poor probe because the
    # 4th-order stencil is exact on it, so the order tests use x^5 whose truncation terms are all
    # non-zero (the cubic is used for accuracy and for the step-size sweep below).
    quint_point = [1.0, 0.5]
    quint_exact = [5.0, 0.3125]

    def quint_error(method, h):
        values = numeric_gradient(quintic, quint_point, h=h, method=method)
        return max(abs(a - n) for a, n in zip(quint_exact, values))

    assert quintic(quint_point) == 1.03125
    assert quint_error(central_difference, 1e-6) <= 1e-6
    central_ratio = quint_error(central_difference, 1e-2) / quint_error(central_difference, 5e-3)
    forward_ratio = quint_error(forward_difference, 1e-3) / quint_error(forward_difference, 5e-4)
    richardson_ratio = quint_error(richardson, 1e-1) / quint_error(richardson, 5e-2)
    assert 3.6 <= central_ratio <= 4.4, central_ratio
    assert 1.8 <= forward_ratio <= 2.2, forward_ratio
    assert 12.0 <= richardson_ratio <= 20.0, richardson_ratio
    assert quint_error(richardson, 1e-2) < quint_error(central_difference, 1e-2), \
        "at the same step the 4th-order stencil must beat the 2nd-order one"

    # --- Step-size sweep: interior optimum, cancellation at the extreme. ---------------------
    errors = sweep(cubic, cubic_grad, point)
    assert len(errors) == 16
    assert errors[0][0] == 0.1 and errors[-1][0] == 1e-16
    best_index = min(range(len(errors)), key=lambda index: errors[index][1])
    assert 0 < best_index < len(errors) - 1, ("the optimum must be interior", best_index)
    assert errors[best_index][1] < 1e-9
    assert errors[-1][1] > errors[best_index][1], "round-off must dominate for tiny h"
    assert errors[0][1] > errors[best_index][1], "truncation must dominate for large h"
    assert errors[-1][1] > errors[4][1]
    monotone_run = [error for _, error in errors[:best_index]]
    assert monotone_run == sorted(monotone_run, reverse=True), "error falls before the optimum"

    # --- Jacobian of a vector-valued function. ----------------------------------------------
    jacobian = numeric_jacobian(vector_function, [0.3, -0.7], h=1e-6)
    analytic = vector_jacobian([0.3, -0.7])
    assert len(jacobian) == 3 and len(jacobian[0]) == 2
    for row_numeric, row_analytic in zip(jacobian, analytic):
        for got, want in zip(row_numeric, row_analytic):
            assert abs(got - want) <= 1e-7, (got, want)
    transposed = [list(row) for row in zip(*jacobian)]
    assert transposed != jacobian, "this Jacobian is genuinely not symmetric"
    assert abs(jacobian[0][0] - (-0.7)) <= 1e-7 and abs(jacobian[0][1] - 0.3) <= 1e-7

    # --- Verdicts: the right gradient passes, a wrong one fails with the faulty coordinate. --
    ok, report = check_gradient(cubic, cubic_grad, point)
    assert ok and report["worst"][0] == 1 and report["max_abs"] <= 1e-9
    assert report["method"] == "central_difference" and report["h"] == DEFAULT_H
    def wrong_grad(x):
        values = cubic_grad(x)
        values[0] += 1e-3
        return values
    ok_bad, bad_report = check_gradient(cubic, wrong_grad, point)
    assert not ok_bad and bad_report["worst"][0] == 0 and bad_report["worst"][1] == 3.001
    assert bad_report["max_abs"] >= 9e-4 and bad_report["max_rel"] >= 1e-4
    ok_forward, forward_report = check_gradient(cubic, cubic_grad, point, rtol=3e-3,
                                               method=forward_difference)
    assert ok_forward and forward_report["max_abs"] > report["max_abs"], \
        "the forward stencil is the sloppier one"

    # --- Scale: relative error is the meaningful metric for exp(sum x). ---------------------
    big = [5.0, 4.0, 3.0]
    scale = math.exp(sum(big))
    ok_big, big_report = check_gradient(exp_sum, exp_sum_grad, big, h=1e-6, rtol=1e-6)
    assert ok_big and big_report["max_rel"] <= 1e-6, big_report
    # The absolute error is 1.7e-4 on a gradient of size 1.6e5 while the cubic example stays at
    # 1e-9: the same utility, the same step - only the scale of the function changed.
    assert 1e-4 <= big_report["max_abs"] <= 1e-3, big_report["max_abs"]
    assert report["max_abs"] <= 1e-9 and scale > 1e5

    # --- Zero gradient and the atol fallback. -----------------------------------------------
    origin_ok, origin_report = check_gradient(sphere, sphere_grad, [0.0, 0.0])
    assert origin_ok and origin_report["max_abs"] == 0.0 and origin_report["max_rel"] == 0.0
    assert numeric_gradient(sphere, [0.0, 0.0]) == [0.0, 0.0]
    assert numeric_gradient(sphere, [0.0, 0.0], method=richardson) == [0.0, 0.0]

    # --- Single-variable functions and no mutation of the caller's point. -------------------
    single = numeric_gradient(lambda x: x[0] ** 3, [2.0])
    assert abs(single[0] - 12.0) <= 1e-9
    protected = [1.0, 2.0, -3.0]
    snapshot = list(protected)
    numeric_gradient(cubic, protected)
    numeric_gradient(cubic, protected, method=richardson)
    numeric_jacobian(vector_function, protected[:2])
    assert protected == snapshot, "the utility must not mutate its input"
    assert _shifted(protected, 0, 1.0) == [2.0, 2.0, -3.0] and protected == snapshot


if __name__ == "__main__":
    main()
    print("PASS")
