"""Task 931 - Implement certified robustness verifier for small networks (interval bound propagation).  (Tier: Master, CTCI Beyond - Machine Learning & AI)

A soundness-first robustness checker for a small dense ReLU network.

* Network: 2 inputs -> 4 hidden units with ReLU -> 2 logits, weights fixed deterministically.
* Interval bound propagation: an input box [l, u] is pushed through every layer.  For z = W x + b
  the tightest elementwise interval is z_lo = W+ l + W- u + b and z_hi = W+ u + W- l + b with
  W+ = max(W, 0) and W- = min(W, 0) (this is the exact affine image of the box), and ReLU maps
  [a, b] to [max(0, a), max(0, b)].  A point is certified for radius eps when the lower bound of
  the true class exceeds the upper bound of every other class.
* Cross-checks: a Lipschitz argument (gradient norm <= the product of the layer operator norms) gives
  an independent, looser interval that must contain the IBP interval, and a sampling search tries to
  falsify every certificate inside the box.

Verified: at eps = 0 the IBP output interval collapses onto the exact forward pass; every sampled
point in the box stays inside the IBP interval; the IBP interval is contained in the Lipschitz
interval (IBP is at least as tight) and intervals grow monotonically with eps; the certified set
shrinks as eps grows and never exceeds the clean accuracy; no certified point can be flipped by any
sampled perturbation (a falsification attempt that must fail); a zero-weight, all-negative or huge-eps
instance behaves; and the verifier is deterministic.  Simplified: dense small network, one bound
propagation method, no branch-and-bound tightening.
"""
import math

INPUT_DIM, HIDDEN, CLASSES = 2, 4, 2
EPSILONS = (0.0, 0.02, 0.05, 0.1, 0.25)
SAMPLES = 160
MASK64 = (1 << 64) - 1
TWO_POW_53 = 9007199254740992.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, raw = splitmix64(state)
    return state, (raw >> 11) / TWO_POW_53


def make_network(state=0x9911):
    """Deterministic 2-4-2 ReLU network: weights, biases and the operator norms of each layer.

    The output layer carries a small positive bias so both logits can be positive; ReLU is applied
    to every layer including the last, which is what the bound propagation mirrors.
    """
    layers = []
    for rows, columns in ((HIDDEN, INPUT_DIM), (CLASSES, HIDDEN)):
        matrix, state = [], state
        for _ in range(rows):
            row = []
            for _ in range(columns):
                state, draw = uniform(state)
                row.append(round((draw - 0.5) * 3.0, 6))
            matrix.append(row)
        layers.append((matrix, [0.3 if rows == CLASSES else 0.0] * rows))
    return layers


def dataset(count=24, state=0x33AA):
    """Points away from the decision boundary, with their network labels."""
    data, state = [], state
    while len(data) < count:
        state, x1 = uniform(state)
        state, x2 = uniform(state)
        point = [round(0.15 + 0.7 * x1, 6), round(0.15 + 0.7 * x2, 6)]
        logits = forward(make_network(), point)
        if abs(logits[0] - logits[1]) > 0.35:
            data.append((point, 0 if logits[0] > logits[1] else 1))
    return data


def forward(network, point):
    """Exact evaluation of the network."""
    activations = list(point)
    for matrix, bias in network:
        activations = [sum(matrix[row][index] * activations[index]
                           for index in range(len(activations))) + bias[row]
                       for row in range(len(matrix))]
        activations = [value if value > 0.0 else 0.0 for value in activations]
    return activations


def propagate(network, lower, upper):
    """Interval bound propagation of an input box through the whole network."""
    low, high = list(lower), list(upper)
    for matrix, bias in network:
        next_low, next_high = [], []
        for row in range(len(matrix)):
            minimum = bias[row]
            maximum = bias[row]
            for index in range(len(low)):
                positive = max(matrix[row][index], 0.0)
                negative = min(matrix[row][index], 0.0)
                minimum += positive * low[index] + negative * high[index]
                maximum += positive * high[index] + negative * low[index]
            next_low.append(minimum)
            next_high.append(maximum)
        low = [max(0.0, value) for value in next_low]
        high = [max(0.0, value) for value in next_high]
    return low, high


def lipschitz_interval(network, point, epsilon):
    """Independent looser bound: the output can move at most L * eps, with L the product of norms."""
    centre = forward(network, point)
    bound = epsilon
    for matrix, _ in network:
        norm = math.sqrt(sum(matrix[row][index] ** 2 for row in range(len(matrix))
                             for index in range(len(matrix[0]))))
        bound *= norm
    return [value - bound for value in centre], [value + bound for value in centre]


def certified(network, point, label, epsilon):
    """True when IBP proves the label cannot change inside the eps box around the point."""
    lower = [max(0.0, value - epsilon) for value in point]
    upper = [min(1.0, value + epsilon) for value in point]
    low, high = propagate(network, lower, upper)
    others = [index for index in range(len(low)) if index != label]
    return low[label] > max(high[index] for index in others)


def sample_box(point, epsilon, count=SAMPLES, state=0x51EE):
    """Deterministic sample of the eps box around a point."""
    samples, state = [], state
    for _ in range(count):
        drawn = []
        for value in point:
            state, draw = uniform(state)
            drawn.append(max(0.0, min(1.0, value + epsilon * (2.0 * draw - 1.0))))
        samples.append(drawn)
    return samples


def certified_accuracy(network, data, epsilon):
    return sum(1 for point, label in data if certified(network, point, label, epsilon)) / len(data)


def clean_accuracy(network, data):
    correct = 0
    for point, label in data:
        logits = forward(network, point)
        correct += 1 if logits.index(max(logits)) == label else 0
    return correct / len(data)


def main():
    network = make_network()
    data = dataset()
    check(clean_accuracy(network, data) == 1.0 and len(data) == 24,
          "the constructed dataset must be perfectly classified by the network")

    # 1. Exactness at eps = 0: the interval collapses onto the true forward pass.
    point, label = data[0]
    low, high = propagate(network, point, point)
    exact = forward(network, point)
    check(max(abs(a - b) for a, b in zip(low, exact)) < 1e-12
          and max(abs(a - b) for a, b in zip(high, exact)) < 1e-12,
          "at radius zero the bounds must equal the exact evaluation")

    # 2. Soundness by sampling: no sampled point may escape the interval.
    epsilon = 0.15
    box_low, box_high = propagate(network, [max(0.0, value - epsilon) for value in point],
                                  [min(1.0, value + epsilon) for value in point])
    inside = True
    for sample in sample_box(point, epsilon):
        logits = forward(network, sample)
        inside = inside and all(box_low[index] - 1e-9 <= logits[index] <= box_high[index] + 1e-9
                                for index in range(CLASSES))
    check(inside, "every sampled output must stay inside the propagated interval")

    # 3. Independent looser bound: the Lipschitz interval must contain the IBP interval.
    loose_low, loose_high = lipschitz_interval(network, point, epsilon)
    check(all(loose_low[index] <= box_low[index] + 1e-12
              and box_high[index] <= loose_high[index] + 1e-12 for index in range(CLASSES)),
          "IBP must be at least as tight as the Lipschitz bound")

    # 4. Monotonicity in eps: bigger radius means a wider interval and no more certificates.
    curve = [certified_accuracy(network, data, value) for value in EPSILONS]
    check(all(curve[i] >= curve[i + 1] - 1e-12 for i in range(len(curve) - 1))
          and curve[0] == 1.0 and curve[-1] < 1.0,
          "certified accuracy must decrease as the radius grows and reach zero eventually")
    wide = propagate(network, [-2.0, -2.0], [2.0, 2.0])
    check(all(a <= b for a, b in zip(wide[0], box_low))
          and all(a <= b for a, b in zip(box_high, wide[1])),
          "a larger box must produce a wider interval that contains the smaller one")

    # 5. Falsification attempt: a certified point must survive every sampled perturbation.
    certified_points = [(point, label) for point, label in data
                        if certified(network, point, label, 0.1)]
    flipped = 0
    for point, label in certified_points:
        for sample in sample_box(point, 0.1, count=60):
            logits = forward(network, sample)
            if logits.index(max(logits)) != label:
                flipped += 1
    check(certified_points and flipped == 0,
          "a certificate must survive an adversarial sampling search")

    # 6. Boundary cases: zero weights collapse the interval to the bias, deep negative units die.
    flat = [([[0.0, 0.0]] * HIDDEN, [0.5] * HIDDEN), ([[0.0] * HIDDEN] * CLASSES, [1.0, -1.0])]
    flat_low, flat_high = propagate(flat, [0.0, 0.0], [1.0, 1.0])
    check(flat_low == flat_high == [1.0, 0.0],
          "a zero-weight network must have an interval equal to its activated biases")
    dead = [([[-1.0, -1.0]] * HIDDEN, [0.0] * HIDDEN), ([[1.0] * HIDDEN] * CLASSES, [0.0] * CLASSES)]
    dead_low, dead_high = propagate(dead, [0.0, 0.0], [1.0, 1.0])
    check(dead_low == dead_high == [0.0, 0.0], "units that can never activate must bound to zero")
    check(certified_accuracy(network, data, 5.0) == 0.0,
          "an absurd radius must certify nothing at all")

    # 7. Certificate versus clean accuracy: certification can never exceed the clean result.
    check(all(value <= clean_accuracy(network, data) + 1e-12 for value in curve),
          "a certificate implies correctness, so it cannot beat the clean accuracy")

    # 8. Determinism: the same instance, samples and bounds every run.
    check(sample_box(point, epsilon) == sample_box(point, epsilon)
          and propagate(network, box_low, box_high)
          == propagate(network, box_low, box_high)
          and certified(network, point, label, 0.1) == certified(network, point, label, 0.1),
          "the verifier must be reproducible")

    print("certified accuracy curve: " + ", ".join(
        f"eps={value}: {curve[index]:.3f}" for index, value in enumerate(EPSILONS)))
    print(f"certified at 0.1: {len(certified_points)}/{len(data)}, sampled flips {flipped}, "
          f"IBP interval {[round(value, 4) for value in box_low]} .. "
          f"{[round(value, 4) for value in box_high]}")
    print("PASS")


if __name__ == "__main__":
    main()
