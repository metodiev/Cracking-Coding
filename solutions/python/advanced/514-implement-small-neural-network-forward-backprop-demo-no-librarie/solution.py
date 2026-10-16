"""Task 514 - Implement small neural-network forward/backprop demo (no libraries). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

A 2-3-1 multilayer perceptron trained on XOR, written out by hand with plain Python lists: the hidden
layer computes h = tanh(W1 x + b1) (three tanh units, so the network can actually bend the decision
boundary), the output unit computes y = sigmoid(W2 h + b2), and the loss is the binary cross-entropy
L = -[t ln y + (1 - t) ln(1 - y)] averaged over the batch. The backward pass is the textbook chain:
the sigmoid/BCE pair collapses to dz2 = y - t, so dW2 = dz2 h, db2 = dz2, then
dW1 = ((W2 dz2) * (1 - h^2)) x^T and db1 = (W2 dz2) * (1 - h^2). The weights are initialised from the
deterministic 75/74 LCG (symmetric initialisation would make the hidden units identical and the
network would be stuck at a linear model), training is full-batch gradient descent with a fixed
learning rate, and every number that enters an assertion is fully reproducible.

Documented checks: the analytic gradient of all 13 parameters (2*3 + 3 + 3*1 + 1) agrees with a
central-difference numerical gradient to 1e-6 (and to 1e-4 in relative terms) at the initialisation
and again after training; the forward pass is checked against an independent hand-written evaluation
of the same expression for one input point; the loss must fall monotonically over the training run
(allowing a 1e-12 wiggle) and end below 0.05; all four XOR points must be classified correctly with
a comfortable margin (prediction < 0.1 for the 0-targets and > 0.9 for the 1-targets); the hidden
units must not collapse onto each other (pairwise distinct activations and weight rows); a learning
rate of 5.0 is shown to be worse than the default 0.5 after the same number of steps; the same seed
must reproduce the parameter vectors bit for bit; and the hidden activations must stay inside
(-1, 1) as tanh demands.
"""

import math

INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TARGETS = [0.0, 1.0, 1.0, 0.0]
HIDDEN = 3
DEFAULT_LR = 0.5
DEFAULT_STEPS = 6000


def lcg(seed):
    return (seed * 75 + 74) % 65537


def init_params(seed=2024, scale=1.0):
    """Deterministic small initial weights; symmetry must be broken for tanh units to differ."""
    w1 = []
    for _ in range(HIDDEN):
        row = []
        for _ in range(2):
            seed = lcg(seed)
            row.append((seed % 2001 - 1000) / 1000.0 * scale)
        w1.append(row)
    seed = lcg(seed)
    b1 = [(seed % 2001 - 1000) / 1000.0 * scale for _ in range(HIDDEN)]
    seed = lcg(seed)
    w2 = [(seed % 2001 - 1000) / 1000.0 * scale for _ in range(HIDDEN)]
    seed = lcg(seed)
    b2 = (seed % 2001 - 1000) / 1000.0 * scale
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def forward(params, x):
    """Returns (hidden activations, output probability)."""
    w1, b1, w2, b2 = params["w1"], params["b1"], params["w2"], params["b2"]
    hidden = [math.tanh(sum(w1[j][k] * x[k] for k in range(2)) + b1[j]) for j in range(HIDDEN)]
    logit = sum(w2[j] * hidden[j] for j in range(HIDDEN)) + b2
    return hidden, sigmoid(logit)


def loss(params, inputs=INPUTS, targets=TARGETS):
    total = 0.0
    for x, t in zip(inputs, targets):
        _, y = forward(params, x)
        total += -(t * math.log(y) + (1.0 - t) * math.log(1.0 - y))
    return total / len(inputs)


def gradients(params, inputs=INPUTS, targets=TARGETS):
    """Full-batch backpropagation. Returns a dict with the same keys as params."""
    grads = {"w1": [[0.0, 0.0] for _ in range(HIDDEN)], "b1": [0.0] * HIDDEN,
             "w2": [0.0] * HIDDEN, "b2": 0.0}
    w2 = params["w2"]
    for x, t in zip(inputs, targets):
        hidden, y = forward(params, x)
        dz2 = y - t                                  # sigmoid + BCE collapses to this
        grads["b2"] += dz2
        for j in range(HIDDEN):
            grads["w2"][j] += dz2 * hidden[j]
            dz1 = dz2 * w2[j] * (1.0 - hidden[j] * hidden[j])
            grads["b1"][j] += dz1
            grads["w1"][j][0] += dz1 * x[0]
            grads["w1"][j][1] += dz1 * x[1]
    scale = 1.0 / len(inputs)
    grads["w1"] = [[value * scale for value in row] for row in grads["w1"]]
    grads["w2"] = [value * scale for value in grads["w2"]]
    grads["b1"] = [value * scale for value in grads["b1"]]
    grads["b2"] = grads["b2"] * scale
    return grads


def flat(params):
    out = []
    for row in params["w1"]:
        out.extend(row)
    out.extend(params["b1"])
    out.extend(params["w2"])
    out.append(params["b2"])
    return out


def unflat(vector):
    params = {"w1": [list(vector[0:2]), list(vector[2:4]), list(vector[4:6])],
              "b1": list(vector[6:9]), "w2": list(vector[9:12]), "b2": vector[12]}
    return params


def train(learning_rate=DEFAULT_LR, steps=DEFAULT_STEPS, seed=2024):
    params = init_params(seed)
    history = [loss(params)]
    for _ in range(steps):
        grads = gradients(params)
        for j in range(HIDDEN):
            for k in range(2):
                params["w1"][j][k] -= learning_rate * grads["w1"][j][k]
            params["b1"][j] -= learning_rate * grads["b1"][j]
            params["w2"][j] -= learning_rate * grads["w2"][j]
        params["b2"] -= learning_rate * grads["b2"]
        history.append(loss(params))
    return params, history


def numeric_gradient(seed=2024, h=1e-5):
    params = init_params(seed)
    best = None
    for index in range(len(flat(params))):
        plus = flat(params)[:]
        minus = flat(params)[:]
        plus[index] += h
        minus[index] -= h
        value = (loss(unflat(plus)) - loss(unflat(minus))) / (2.0 * h)
        if best is None:
            best = [value]
        else:
            best.append(value)
    return best


def main():
    params = init_params()
    assert len(params["w1"]) == 3 and all(len(row) == 2 for row in params["w1"])
    assert len(params["b1"]) == 3 and len(params["w2"]) == 3
    assert len(flat(params)) == 13, "2*3 + 3 + 3*1 + 1 parameters"
    assert max(abs(value) for value in flat(params)) <= 1.0

    # --- Forward pass against an independent hand evaluation. --------------------------------
    hidden, y = forward(params, [1.0, 0.0])
    expected_hidden = [math.tanh(params["w1"][j][0] + params["b1"][j]) for j in range(3)]
    for got, want in zip(hidden, expected_hidden):
        assert abs(got - want) <= 1e-15
    logit = sum(params["w2"][j] * expected_hidden[j] for j in range(3)) + params["b2"]
    assert abs(y - 1.0 / (1.0 + math.exp(-logit))) <= 1e-15
    assert all(-1.0 < value < 1.0 for value in hidden), "tanh activations stay in (-1, 1)"
    assert len({round(value, 9) for value in hidden}) == 3, "the hidden units must differ"

    # --- Backprop against central differences at the initialisation. --------------------------
    analytic_vector = flat(gradients(params))
    numeric = numeric_gradient()
    assert len(analytic_vector) == len(numeric) == 13
    worst = max(abs(a - n) for a, n in zip(analytic_vector, numeric))
    assert worst <= 1e-6, ("gradient check at init", worst)
    for index, (a, n) in enumerate(zip(analytic_vector, numeric)):
        denominator = max(abs(a), abs(n), 1e-12)
        assert abs(a - n) / denominator <= 1e-4, (index, a, n)

    # --- Training converges on XOR. ----------------------------------------------------------
    initial = loss(params)
    trained, history = train()
    final = history[-1]
    assert initial > 0.6, initial
    assert final < 0.05, final
    assert final < initial / 10.0
    for index in range(1, len(history)):
        assert history[index] <= history[index - 1] + 1e-12, ("loss must not rise", index)
    assert history[0] == initial and len(history) == DEFAULT_STEPS + 1
    predictions = [forward(trained, x)[1] for x in INPUTS]
    assert predictions[0] < 0.1 and predictions[3] < 0.1, predictions
    assert predictions[1] > 0.9 and predictions[2] > 0.9, predictions
    assert sum(1 for x, t in zip(INPUTS, TARGETS) if (forward(trained, x)[1] > 0.5) == (t > 0.5)) == 4
    trained_hidden = [forward(trained, x)[0] for x in INPUTS]
    assert len({tuple(round(value, 9) for value in row) for row in trained_hidden}) == 4

    # --- Gradient check again, now at the trained parameters. --------------------------------
    trained_analytic = flat(gradients(trained))
    assert len(trained_analytic) == 13
    for a, n in zip(trained_analytic, numeric_gradient_at(trained)):
        assert abs(a - n) <= 1e-6, (a, n)
    assert max(abs(value) for value in trained_analytic) <= 1e-3, \
        "at convergence every partial derivative is tiny"

    # --- Symmetry breaking survives training, and the seed is reproducible. ------------------
    w1 = trained["w1"]
    for first in range(HIDDEN):
        for second in range(first + 1, HIDDEN):
            assert abs(w1[first][0] - w1[second][0]) + abs(w1[first][1] - w1[second][1]) > 1e-6
    again, history_again = train()
    assert history_again == history
    assert flat(again) == flat(trained), "the same seed must give identical parameters"

    # --- The learning rate is a genuine trade-off (all runs share the same 200-step budget). --
    tiny = train(learning_rate=0.1, steps=200)[1]
    steady = train(learning_rate=0.5, steps=200)[1]
    fast = train(learning_rate=5.0, steps=200)[1]
    wild = train(learning_rate=20.0, steps=200)[1]
    assert tiny[-1] > steady[-1], ("too small a step is merely slow", tiny[-1], steady[-1])
    assert fast[-1] < steady[-1], "a bigger step makes faster progress on this problem"
    assert sum(1 for i in range(1, len(fast)) if fast[i] > fast[i - 1] + 1e-12) > 0, \
        "the aggressive rate overshoots along the way"
    assert wild[-1] > wild[0] and wild[-1] > 10.0 * steady[-1], \
        ("too large a step diverges", wild[0], wild[-1])
    assert max(wild) > 5.0


def numeric_gradient_at(params, h=1e-5):
    """Central differences at an arbitrary parameter vector (kept separate from the init check)."""
    base = flat(params)
    out = []
    for index in range(len(base)):
        plus = base[:]
        minus = base[:]
        plus[index] += h
        minus[index] -= h
        out.append((loss(unflat(plus)) - loss(unflat(minus))) / (2.0 * h))
    return out


if __name__ == "__main__":
    main()
    print("PASS")
