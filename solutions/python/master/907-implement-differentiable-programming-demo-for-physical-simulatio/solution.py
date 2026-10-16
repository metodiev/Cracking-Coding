"""Task 907 - Implement differentiable programming demo for physical simulation parameter fitting.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; no wall clock, no PRNG needed - the observations
come from the same simulator run with the true parameters):

  * `Jet` - a first-order "jet" number: a value plus the partial derivatives w.r.t. a fixed list
    of parameters.  Forward-mode automatic differentiation then costs ONE simulation sweep for
    the whole gradient instead of one per parameter, and the same sweep also yields the
    trajectory Jacobian.
  * `rollout` - a physical simulation (projectile with quadratic drag) written once, generically
    over Jet numbers:  v <- v + dt * (-g - k v |v|),  x <- x + dt * v.  Because the arithmetic is
    overloaded, no hand-derived adjoint is needed and gravity and the drag coefficient are
    differentiated exactly.
  * `loss_grad` - the least-squares mismatch of the simulated trajectory and the observations,
    with its exact gradient; `fit` - gradient descent with a diagonal preconditioner built from
    the same Jacobian.
  * Oracles: central finite differences for the gradient, the closed form d x_i / d g = -t_i^2/2
    in the drag-free limit, and the first-order optimality condition grad L = 0 at the truth.

Simplified versus production: one scalar parameter per physical constant, explicit Euler instead
of a stiff integrator, no minibatch stochasticity.  Verified below: Jet arithmetic against
analytic derivatives, the trajectory gradient against finite differences and the closed form, the
zero-gradient optimality at the true parameters, loss monotonicity and convergence of the fit,
and the zero-step / single-step / identifiability boundaries.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Jet:
    """Value plus the partial derivatives w.r.t. the parameter vector (fixed length)."""

    __slots__ = ("value", "grad")

    def __init__(self, value, grad):
        self.value = value
        self.grad = tuple(grad)

    @staticmethod
    def constant(value, width):
        return Jet(value, (0.0,) * width)

    @staticmethod
    def variable(value, index, width):
        return Jet(value, tuple(1.0 if i == index else 0.0 for i in range(width)))

    def _other(self, other):
        return other if isinstance(other, Jet) else Jet.constant(other, len(self.grad))

    def __add__(self, other):
        other = self._other(other)
        return Jet(self.value + other.value, [a + b for a, b in zip(self.grad, other.grad)])

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, [-a for a in self.grad])

    def __sub__(self, other):
        return self + (-self._other(other))

    def __rsub__(self, other):
        return self._other(other) - self

    def __mul__(self, other):
        other = self._other(other)
        return Jet(self.value * other.value,
                   [a * other.value + b * self.value for a, b in zip(self.grad, other.grad)])

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._other(other)
        return Jet(self.value / other.value,
                   [(a * other.value - self.value * b) / (other.value ** 2)
                    for a, b in zip(self.grad, other.grad)])

    def abs(self):
        return Jet(abs(self.value), [a if self.value > 0 else -a for a in self.grad])

    def exp(self):
        factor = __import__("math").exp(self.value)
        return Jet(factor, [a * factor for a in self.grad])


WIDTH = 2                     # the parameter vector is (gravity, drag)


def rollout(g, k, dt, steps, height):
    """Projectile with quadratic drag, integrated by explicit Euler over Jet numbers."""
    x = Jet.constant(height, WIDTH)
    v = Jet.constant(0.0, WIDTH)
    xs, vs = [x], [v]
    for _ in range(steps):
        acceleration = -g - k * v * v.abs()
        v = v + acceleration * dt
        x = x + v * dt
        xs.append(x)
        vs.append(v)
    return xs, vs


def true_trajectory(g_true, k_true, dt, steps, height):
    xs, _ = rollout(Jet.constant(g_true, WIDTH), Jet.constant(k_true, WIDTH), dt, steps, height)
    return [x.value for x in xs]


def loss_grad(observed, g_value, k_value, dt, steps, height, with_jacobian=False):
    """Least-squares loss and its exact gradient; optionally the per-step Jacobian row."""
    g = Jet.variable(g_value, 0, WIDTH)
    k = Jet.variable(k_value, 1, WIDTH)
    xs, _ = rollout(g, k, dt, steps, height)
    total = Jet.constant(0.0, WIDTH)
    rows = []
    for x, obs in zip(xs[1:], observed):
        residual = x - obs
        total = total + residual * residual
        if with_jacobian:
            rows.append(x.grad)
    count = len(observed)
    # L = sum(r^2) / (2 N)  =>  the accumulated sum of squares divides its gradient by 2N too
    if with_jacobian:
        return total.value / (2 * count), [t / (2 * count) for t in total.grad], rows
    return total.value / (2 * count), [t / (2 * count) for t in total.grad]


def finite_difference(observed, g_value, k_value, dt, steps, height, h=1e-5):
    """Central differences - the independent oracle for the gradient."""
    grads = []
    for index, base in enumerate((g_value, k_value)):
        step = [0.0, 0.0]
        step[index] = h
        plus = loss_grad(observed, g_value + step[0], k_value + step[1], dt, steps, height)[0]
        minus = loss_grad(observed, g_value - step[0], k_value - step[1], dt, steps, height)[0]
        grads.append((plus - minus) / (2 * h))
    return grads


def fit(observed, dt, steps, height, start, iterations, learning_rate=0.9, free=(True, True)):
    """Preconditioned gradient descent over the free parameters; returns (params, history)."""
    g_value, k_value = start
    history = [loss_grad(observed, g_value, k_value, dt, steps, height)[0]]
    for _ in range(iterations):
        loss, gradient, rows = loss_grad(observed, g_value, k_value, dt, steps, height, True)
        diagonal = [sum(row[i] * row[i] for row in rows) / len(rows) + 1e-6 for i in (0, 1)]
        if free[0]:
            g_value -= learning_rate * gradient[0] / diagonal[0]
        if free[1]:
            k_value -= learning_rate * gradient[1] / diagonal[1]
        history.append(loss_grad(observed, g_value, k_value, dt, steps, height)[0])
    return (g_value, k_value), history


def main():
    g_true, k_true = 9.81, 0.05
    height, dt, steps = 20.0, 0.02, 40
    observed = true_trajectory(g_true, k_true, dt, steps, height)[1:]
    check(len(observed) == steps, "one observation per integration step")

    # ---- 1. Jet arithmetic against analytic derivatives
    w = 3
    a = Jet.variable(1.5, 0, w)
    b = Jet.variable(-0.75, 1, w)
    f = (a * b + a * a) / (b - 2.0)
    # f(a, b) = (ab + a^2)/(b - 2);  df/da = (b + 2a)/(b-2), df/db = -(ab + a^2)/(b-2)^2
    da = (b.value + 2 * a.value) / (b.value - 2.0)
    db = -(2 * a.value + a.value ** 2) / (b.value - 2.0) ** 2
    check(abs(f.grad[0] - da) < 1e-12 and abs(f.grad[1] - db) < 1e-12,
          "product/quotient rules must match the analytic derivative")
    e = (a * 2.0).exp()
    check(abs(e.grad[0] - 2.0 * __import__("math").exp(2.0 * 1.5)) < 1e-9,
          "the chain rule through exp must hold")
    check(abs(a.grad[0] - 1.0) < 1e-15 and a.grad[1] == 0.0, "variables carry a unit basis")

    # ---- 2. the trajectory gradient: finite differences and the drag-free closed form
    for g_value, k_value in ((9.81, 0.05), (11.0, 0.2), (8.0, 0.0), (20.0, 0.5)):
        exact = loss_grad(observed, g_value, k_value, dt, steps, height)[1]
        approx = finite_difference(observed, g_value, k_value, dt, steps, height)
        for index, (left, right) in enumerate(zip(exact, approx)):
            check(abs(left - right) < 1e-6 * max(1.0, abs(left)),
                  "autodiff vs finite differences at (%s, %s) component %d: %.9f vs %.9f"
                  % (g_value, k_value, index, left, right))
    dragless = true_trajectory(g_true, 0.0, dt, steps, height)[1:]
    probe_g = 10.5
    exact = loss_grad(dragless, probe_g, 0.0, dt, steps, height)[1]
    # explicit Euler with k = 0 gives x_n = h - g dt^2 n(n+1)/2 exactly, so dx_n/dg is closed form
    total = 0.0
    for index, obs in enumerate(dragless):
        step_count = index + 1
        slope = -dt * dt * step_count * (step_count + 1) / 2.0
        total += (height + probe_g * slope - obs) * slope
    closed = total / len(dragless)
    check(abs(exact[0] - closed) < 1e-9,
          "the drag-free gradient must match the closed form (%.9f vs %.9f)" % (exact[0], closed))
    check(loss_grad(dragless, g_true, 0.0, dt, steps, height)[1][0] == 0.0,
          "with exact drag-free data the gravity gradient must vanish at the truth")

    # ---- 3. first-order optimality at the true parameters
    loss, gradient = loss_grad(observed, g_true, k_true, dt, steps, height)
    check(loss < 1e-24, "the exact trajectory must have (near) zero residual loss")
    check(max(abs(v) for v in gradient) < 1e-12,
          "the gradient must vanish at the true parameters: %r" % gradient)

    # ---- 4. fitting from a perturbed start
    start = (12.5, 0.35)
    start_loss = loss_grad(observed, start[0], start[1], dt, steps, height)[0]
    fitted, history = fit(observed, dt, steps, height, start, 400)
    check(start_loss > 1e-3, "the perturbed start must be genuinely wrong")
    check(history[-1] < start_loss * 1e-6,
          "the fit must cut the loss by six orders of magnitude (%.3e -> %.3e)"
          % (start_loss, history[-1]))
    check(all(history[i + 1] <= history[i] + 1e-12 for i in range(len(history) - 1)),
          "the preconditioned descent must not increase the loss")
    error_before = abs(start[0] - g_true) + abs(start[1] - k_true)
    error_after = abs(fitted[0] - g_true) + abs(fitted[1] - k_true)
    check(error_after < error_before / 100.0,
          "the parameters must be recovered: %.5f -> %.5f" % (error_before, error_after))
    check(fitted[0] > 0 and fitted[1] >= 0, "the fitted constants must stay physical")

    # ---- 5. identifiability: a frozen parameter provably stays put
    only_g, _ = fit(observed, dt, steps, height, (12.5, k_true), 400, free=(True, False))
    check(abs(only_g[0] - g_true) < 1e-3,
          "with the drag frozen, gravity must be recovered exactly (%.6f)" % only_g[0])
    check(only_g[1] == k_true, "a frozen parameter must not move")
    both, _ = fit(observed, dt, steps, height, (12.5, k_true), 400)
    check(abs(both[1] - k_true) < 1e-6, "the drag coefficient must also be identifiable")

    # ---- 6. boundaries: no steps, one step, and the loss at the truth
    flat = [height] * 5
    zero_loss, zero_grad = loss_grad(flat, 9.81, 0.05, dt, 0, height)
    check(zero_loss == 0.0 and zero_grad == [0.0, 0.0],
          "a zero-step rollout cannot depend on the parameters")
    single = [height - 9.81 * dt * dt]
    one_loss, one_grad = loss_grad(single, 9.81, 0.05, dt, 1, height)
    check(abs(one_loss) < 1e-18 and max(abs(v) for v in one_grad) < 1e-12,
          "a single exact step must also have a vanishing gradient")
    check(loss_grad(observed, g_true, k_true, dt, steps, height)[0]
          < loss_grad(observed, g_true + 0.5, k_true, dt, steps, height)[0],
          "the truth must be a strict minimum in gravity")

    # ---- 7. determinism
    again = loss_grad(observed, 10.0, 0.1, dt, steps, height)[1]
    check(again == loss_grad(observed, 10.0, 0.1, dt, steps, height)[1],
          "the gradient must be reproducible")
    check(true_trajectory(g_true, k_true, dt, steps, height) == observed[:0] + [height] + observed,
          "the simulator must reproduce the observations it generated")

    print("fitted (g, k) = (%.6f, %.6f) from (%.2f, %.2f)" % (fitted + start))
    print("loss %.3e -> %.3e in %d iterations" % (start_loss, history[-1], len(history) - 1))
    print("truth: g=%.2f k=%.2f over %d steps of dt=%.3f" % (g_true, k_true, steps, dt))
    print("PASS")


if __name__ == "__main__":
    main()
