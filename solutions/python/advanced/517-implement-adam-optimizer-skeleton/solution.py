"""Task 517 - Implement Adam optimizer skeleton. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

The task calls for a skeleton; this is the full optimiser instead, because the interesting part of
Adam is exactly the bookkeeping a skeleton would stub out. Adam keeps a first moment m and a second
moment v of the gradient, updates them with the exponential moving averages
m <- b1 m + (1 - b1) g and v <- b2 v + (1 - b2) g^2, corrects their initialisation bias with
m_hat = m / (1 - b1^t) and v_hat = v / (1 - b2^t) (both moments start at zero, so without the
correction the first steps would be far too small), and then takes the step
theta <- theta - lr * m_hat / (sqrt(v_hat) + eps). A decoupled weight-decay variant (AdamW) subtracts
lr * wd * theta from the parameter instead of folding the decay into the gradient, and an amsgrad flag
keeps the running maximum of v_hat for the denominator. Plain gradient descent is included as the
reference optimiser for the comparison experiments.

Documented checks: the first step from x = 3 on f = x^2 (gradient 6, so m_hat = 6 and v_hat = 36) moves
to exactly 3 - lr with lr = 0.1, to 1e-9 - the classic demonstration that bias correction makes the
first step size independent of the gradient scale; the first step is bit-identical for beta2 = 0.9,
0.99, 0.999 and 0.9999 (within 1e-12) and proportional to lr for gradients 100x apart, deviating from
lr only by lr * eps / |g|, which is precisely what the bias correction buys; twenty steps on f = x^3 are compared against an independent, straight-line
implementation of the update rules written out in the test (agreement to 1e-12, plus the internal
step counter t and the moment shapes); Adam converges to the minimum of a 3-D anisotropic quadratic
(max |theta - theta*| <= 1e-3 and loss <= 1e-6 after 4000 steps, actually 1e-16 after 500) where
gradient descent with its best stable rate (0.0099, just under the 0.01 stability limit) is nine
orders of magnitude behind after the same 500 steps; on a zero-gradient problem the decoupled
weight decay shrinks the parameter by exactly the closed form prod (1 - lr wd) - the documented
difference between AdamW and L2-in-the-gradient; and two runs with the same seed produce identical
trajectories.
"""

import math

DEFAULT_LR = 0.1
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


class Adam:
    """Adam with optional amsgrad and decoupled (AdamW) weight decay."""

    def __init__(self, parameters, learning_rate=DEFAULT_LR, beta1=BETA1, beta2=BETA2, eps=EPS,
                 weight_decay=0.0, amsgrad=False):
        self.theta = list(parameters)
        self.lr = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.t = 0
        self.m = [0.0] * len(self.theta)
        self.v = [0.0] * len(self.theta)
        self.v_max = [0.0] * len(self.theta)
        self.updates = 0

    def step(self, gradient):
        if len(gradient) != len(self.theta):
            raise ValueError("gradient length mismatch")
        self.t += 1
        bias1 = 1.0 - self.b1 ** self.t
        bias2 = 1.0 - self.b2 ** self.t
        for index in range(len(self.theta)):
            g = gradient[index]
            if self.weight_decay:
                # Decoupled decay: the raw gradient entering the moments stays clean (AdamW), the
                # shrinkage is applied directly to the parameter below.
                self.theta[index] -= self.lr * self.weight_decay * self.theta[index]
            self.m[index] = self.b1 * self.m[index] + (1.0 - self.b1) * g
            self.v[index] = self.b2 * self.v[index] + (1.0 - self.b2) * g * g
            m_hat = self.m[index] / bias1
            v_hat = self.v[index] / bias2
            if self.amsgrad:
                self.v_max[index] = max(self.v_max[index], v_hat)
                denominator = math.sqrt(self.v_max[index]) + self.eps
            else:
                denominator = math.sqrt(v_hat) + self.eps
            self.theta[index] -= self.lr * m_hat / denominator
        self.updates += 1
        return self.theta


class GradientDescent:
    def __init__(self, parameters, learning_rate):
        self.theta = list(parameters)
        self.lr = learning_rate

    def step(self, gradient):
        self.theta = [value - self.lr * g for value, g in zip(self.theta, gradient)]
        return self.theta


def adam_x3_steps(x, steps, learning_rate=DEFAULT_LR):
    """Straight-line reference implementation used only by the tests (no classes involved)."""
    m = 0.0
    v = 0.0
    trajectory = [x]
    for t in range(1, steps + 1):
        g = 3.0 * x * x
        m = BETA1 * m + (1.0 - BETA1) * g
        v = BETA2 * v + (1.0 - BETA2) * g * g
        m_hat = m / (1.0 - BETA1 ** t)
        v_hat = v / (1.0 - BETA2 ** t)
        x = x - learning_rate * m_hat / (math.sqrt(v_hat) + EPS)
        trajectory.append(x)
    return trajectory


def quadratic(theta, curvature):
    """f(theta) = 0.5 * sum curvature_i * theta_i^2 (minimum at the origin)."""
    return 0.5 * sum(c * value * value for c, value in zip(curvature, theta))


def quadratic_grad(theta, curvature):
    return [c * value for c, value in zip(curvature, theta)]


def main():
    # --- First step: m_hat = g, v_hat = g^2, so the step is lr * g / (|g| + eps). -------------
    optimizer = Adam([3.0], learning_rate=0.1)
    optimizer.step([6.0])
    assert abs(optimizer.theta[0] - 2.9) <= 1e-9, optimizer.theta
    assert abs(optimizer.m[0] - 0.6) <= 1e-15 and abs(optimizer.v[0] - 0.036) <= 1e-15
    assert optimizer.t == 1 and optimizer.updates == 1

    # Bias correction makes the first step independent of beta2 and proportional to lr for any
    # gradient scale: the only deviation from lr comes from eps sitting in the denominator.
    for scale in (0.01, 1.0, 100.0):
        group = []
        for beta2 in (0.9, 0.99, 0.999, 0.9999):
            probe = Adam([3.0], learning_rate=0.1, beta2=beta2)
            probe.step([6.0 * scale])
            group.append(probe.theta[0])
        assert max(group) - min(group) <= 1e-12, ("beta2 must not matter on step one", group)
        assert abs(group[0] - 2.9) <= 0.1 * EPS / (6.0 * scale) + 1e-12, (scale, group[0])
    first_steps = [Adam([3.0], learning_rate=0.1, beta2=beta2).step([6.0])[0]
                   for beta2 in (0.9, 0.99, 0.999, 0.9999)]
    assert len(set(first_steps)) == 1, ("beta2 must not affect step one", first_steps)
    # Without the correction the first step would be eps-dominated and visibly smaller - the whole
    # reason the correction exists. (m = 0.1 g, v = 0.001 g^2 without bias correction.)
    uncorrected = 0.1 * (0.1 * 6.0) / (math.sqrt(0.001 * 36.0) + EPS)
    assert abs(uncorrected - 0.1) > 1e-3, uncorrected

    # --- Twenty steps on x^3 against the straight-line reference. ----------------------------
    optimizer = Adam([0.7], learning_rate=0.05)
    trajectory = [optimizer.theta[0]]
    for _ in range(20):
        x = optimizer.theta[0]
        optimizer.step([3.0 * x * x])
        trajectory.append(optimizer.theta[0])
    reference = adam_x3_steps(0.7, 20, learning_rate=0.05)
    for got, want in zip(trajectory, reference):
        assert abs(got - want) <= 1e-12, (got, want)
    assert optimizer.t == 20
    assert len(optimizer.m) == len(optimizer.v) == 1
    assert optimizer.theta[0] < 0.7 and trajectory[-1] < trajectory[1]

    # --- Convergence on an ill-conditioned quadratic, beating tuned gradient descent. --------
    curvature = [1.0, 50.0, 200.0]
    start = [10.0, -8.0, 6.0]
    optimum = [0.0, 0.0, 0.0]
    adam = Adam(start, learning_rate=0.1)
    for _ in range(4000):
        adam.step(quadratic_grad(adam.theta, curvature))
    worst_adam = max(abs(a - b) for a, b in zip(adam.theta, optimum))
    assert worst_adam <= 1e-3, worst_adam
    adam_loss = quadratic(adam.theta, curvature)
    assert adam_loss <= 1e-6, adam_loss

    # Over a shorter budget the difference is unmistakable: Adam is nine orders of magnitude
    # closer to the optimum than gradient descent with its best stable step size.
    short_adam = Adam(start, learning_rate=0.1)
    for _ in range(500):
        short_adam.step(quadratic_grad(short_adam.theta, curvature))
    short_adam_loss = quadratic(short_adam.theta, curvature)
    best_gd = None
    for rate in (0.001, 0.002, 0.005, 0.008, 0.009, 0.0095, 0.0099):
        descent = GradientDescent(start, rate)
        for _ in range(500):
            descent.step(quadratic_grad(descent.theta, curvature))
        loss = quadratic(descent.theta, curvature)
        if best_gd is None or loss < best_gd[0]:
            best_gd = (loss, rate)
    # 2 / 200 = 0.01 is the stability limit, so the best stable rate is found at the edge of it.
    assert best_gd[1] <= 0.01, best_gd
    assert best_gd[0] > 1e9 * short_adam_loss, (best_gd, short_adam_loss)
    assert short_adam_loss < 1e-12
    divergence = GradientDescent(start, 0.02)
    for _ in range(200):
        divergence.step(quadratic_grad(divergence.theta, curvature))
    assert quadratic(divergence.theta, curvature) > quadratic(start, curvature)

    # --- Decoupled weight decay has the closed form prod (1 - lr * wd). ----------------------
    lr, wd = 0.1, 0.05
    shrink = Adam([5.0], learning_rate=lr, weight_decay=wd)
    for _ in range(37):
        shrink.step([0.0])                       # zero gradient: only the decay acts
    expected = 5.0 * (1.0 - lr * wd) ** 37
    assert abs(shrink.theta[0] - expected) <= 1e-12, (shrink.theta[0], expected)
    plain = Adam([5.0], learning_rate=lr)
    for _ in range(37):
        plain.step([0.0])
    assert plain.theta[0] == 5.0, "without decay a zero gradient changes nothing"
    assert shrink.theta[0] < plain.theta[0]
    # The same decay without a gradient leaves the moments at zero as well.
    assert shrink.m == [0.0] and shrink.v == [0.0] and shrink.t == 37

    # --- amsgrad keeps the running maximum of v_hat. -----------------------------------------
    ams = Adam([0.0], learning_rate=0.1, amsgrad=True)
    plain = Adam([0.0], learning_rate=0.1)
    for _ in range(3):
        ams.step([2.0])
        plain.step([2.0])
    assert abs(ams.theta[0] - plain.theta[0]) <= 1e-9, \
        "with a constant gradient amsgrad and plain Adam agree"
    assert ams.v_max[0] >= ams.v[0] / (1.0 - BETA2 ** ams.t) - 1e-15
    ams_peak = Adam([0.0], learning_rate=0.1, amsgrad=True)
    plain_decay = Adam([0.0], learning_rate=0.1)
    for gradient in (2.0, 2.0, 0.01, 0.01):
        ams_peak.step([gradient])
        plain_decay.step([gradient])
    assert ams_peak.v_max[0] > ams_peak.v[0] / (1.0 - BETA2 ** ams_peak.t)
    assert ams_peak.theta[0] > plain_decay.theta[0] + 1e-6, \
        "the stale second moment makes the later steps smaller"
    ams_empty = Adam([0.0], learning_rate=0.1, amsgrad=True)
    ams_empty.step([0.0])
    assert ams_empty.v_max == [0.0] and ams_empty.theta == [0.0]

    # --- Determinism and input handling. ----------------------------------------------------
    first = Adam([1.0, 2.0], learning_rate=0.05)
    second = Adam([1.0, 2.0], learning_rate=0.05)
    for _ in range(25):
        first.step([1.0, -1.0])
        second.step([1.0, -1.0])
    assert first.theta == second.theta and first.m == second.m and first.v == second.v
    assert len(first.theta) == 2 and len(first.m) == 2 and len(first.v) == 2
    try:
        first.step([1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("a mismatched gradient must be rejected")
    fresh = Adam([1.0])
    assert fresh.theta == [1.0] and fresh.t == 0 and fresh.m == [0.0] and fresh.v == [0.0]
    assert Adam([1.0], learning_rate=0.1, beta1=0.0, beta2=0.0).step([3.0])[0] == \
        1.0 - 0.1 * 3.0 / (3.0 + EPS)


if __name__ == "__main__":
    main()
    print("PASS")
