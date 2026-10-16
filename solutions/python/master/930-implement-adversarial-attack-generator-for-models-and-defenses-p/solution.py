"""Task 930 - Implement adversarial attack generator for models and defenses (PGD).  (Tier: Master, CTCI Beyond - Machine Learning & AI)

A small hand-rolled neural network, the gradient-based attacks against it, and adversarial training
as the defence.

* Model: 2 -> 8 (ReLU) -> 2 softmax layer with backpropagation written out by hand, trained by
  deterministic SGD on a 2-D two-class dataset, so the clean accuracy climbs above 95%.
* Attacks: FGSM is a single signed-gradient step, x + eps sign(grad_x loss); PGD iterates it with a
  step size eps/4 and projects every iterate back into the eps ball around the input and into the
  valid input range [0, 1].  Both ascend the loss and are computed with the analytic input gradient.
* Defence: adversarial training, which augments each mini-batch with freshly generated FGSM images,
  and is then scored against PGD (the standard way to show that a defence transfers).

Verified: the analytic input gradient matches central finite differences; on a linear model FGSM
reduces to the closed form x - eps sign(w) for a class-1 point, exactly what the theory says; the PGD
loss is non-decreasing over iterations and never below the FGSM loss; every perturbed image stays
inside the eps ball and inside [0, 1]; eps = 0 reproduces the clean input bit for bit; the clean
accuracy is high while the attacked accuracy collapses, with a high attack success rate; adversarial
training measurably raises the robust accuracy; and the pipeline is deterministic.  Simplified: a
tiny network, a 2-D toy dataset, no momentum/random restarts in PGD.
"""
import math

HIDDEN, INPUT_DIM, CLASSES = 8, 2, 2
SAMPLES, EPOCHS, LEARNING_RATE = 120, 90, 0.35
EPSILON, PGD_STEPS, PGD_ALPHA = 0.3, 10, 0.075
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

def make_dataset(samples=SAMPLES, state=0xC0DE):
    """Two classes separated by x1 + 0.6 x2 = 0.8, in the unit square."""
    data, state = [], state
    while len(data) < samples:
        state, x1 = uniform(state)
        state, x2 = uniform(state)
        label = 1 if x1 + 0.6 * x2 > 0.8 else 0
        if 0.72 < x1 + 0.6 * x2 < 0.88:
            continue                              # drop the ambiguous band for a clean task
        data.append([[x1, x2], label])
    return data

def initial_parameters(state=0x1234):
    weights = []
    for rows in ((HIDDEN, INPUT_DIM), (CLASSES, HIDDEN)):
        matrix, state = [], state
        for _ in range(rows[0]):
            row = []
            for _ in range(rows[1]):
                state, draw = uniform(state)
                row.append((draw - 0.5) * 1.2)
            matrix.append(row)
        weights.append(matrix)
    return {"w1": weights[0], "b1": [0.0] * HIDDEN, "w2": weights[1], "b2": [0.0] * CLASSES}

def forward(parameters, point):
    pre = [sum(parameters["w1"][index][feature] * point[feature] for feature in range(INPUT_DIM))
           + parameters["b1"][index] for index in range(HIDDEN)]
    hidden = [value if value > 0.0 else 0.0 for value in pre]
    logits = [sum(parameters["w2"][label][index] * hidden[index] for index in range(HIDDEN))
              + parameters["b2"][label] for label in range(CLASSES)]
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    total = sum(exponentials)
    return hidden, pre, [value / total for value in exponentials]

def loss(parameters, point, label):
    return -math.log(max(forward(parameters, point)[2][label], 1e-12))

def input_gradient(parameters, point, label):
    """d loss / d x, by hand, for gradient-based attacks."""
    hidden, pre, probabilities = forward(parameters, point)
    delta = [probabilities[class_index] - (1.0 if class_index == label else 0.0)
             for class_index in range(CLASSES)]
    hidden_gradient = [sum(delta[class_index] * parameters["w2"][class_index][index]
                           for class_index in range(CLASSES)) for index in range(HIDDEN)]
    return [sum(hidden_gradient[index] * parameters["w1"][index][feature] for index in range(HIDDEN)
                if pre[index] > 0.0) for feature in range(INPUT_DIM)]

def train(data, parameters, epochs=EPOCHS, rate=LEARNING_RATE, epsilon=0.0, state=0x77):
    """SGD training; when epsilon > 0 each sample is augmented with a fresh FGSM image."""
    for _ in range(epochs):
        for point, label in data:
            sample = point
            if epsilon > 0.0:
                sample = fgsm(parameters, point, label, epsilon)
            hidden, pre, probabilities = forward(parameters, sample)
            delta = [probabilities[class_index] - (1.0 if class_index == label else 0.0)
                     for class_index in range(CLASSES)]
            hidden_gradient = [sum(delta[class_index] * parameters["w2"][class_index][index]
                                   for class_index in range(CLASSES)) for index in range(HIDDEN)]
            for class_index in range(CLASSES):
                for index in range(HIDDEN):
                    parameters["w2"][class_index][index] -= rate * delta[class_index] * hidden[index]
                parameters["b2"][class_index] -= rate * delta[class_index]
            for index in range(HIDDEN):
                if pre[index] <= 0.0:
                    continue
                for feature in range(INPUT_DIM):
                    parameters["w1"][index][feature] -= (rate * hidden_gradient[index]
                                                         * sample[feature])
                parameters["b1"][index] -= rate * hidden_gradient[index]
    return parameters

def project(point, original, epsilon):
    """Clamp into the eps ball around the original image and into the valid input range."""
    return [min(max(point[index], original[index] - epsilon, 0.0), original[index] + epsilon, 1.0)
            for index in range(len(point))]

def fgsm(parameters, point, label, epsilon):
    gradient = input_gradient(parameters, point, label)
    stepped = [point[index] + epsilon * (1.0 if gradient[index] > 0 else -1.0)
               if gradient[index] != 0 else point[index] for index in range(len(point))]
    return project(stepped, point, epsilon)

def pgd(parameters, point, label, epsilon, steps=PGD_STEPS, alpha=PGD_ALPHA):
    """Projected gradient descent; returns (adversarial point, loss trace)."""
    current, trace = list(point), [loss(parameters, point, label)]
    for _ in range(steps):
        gradient = input_gradient(parameters, current, label)
        stepped = [current[index] + alpha * (1.0 if gradient[index] > 0 else -1.0)
                   if gradient[index] != 0 else current[index] for index in range(len(current))]
        current = project(stepped, point, min(alpha * steps, epsilon))
        trace.append(loss(parameters, current, label))
    return current, trace

def accuracy(parameters, data, attack=None):
    correct = 0
    for point, label in data:
        sample = point if attack is None else attack(parameters, point, label)
        _, _, probabilities = forward(parameters, sample)
        correct += 1 if probabilities.index(max(probabilities)) == label else 0
    return correct / len(data)

def success_rate(parameters, data, attack):
    """Fraction of points whose prediction changes, or whose confidence drops, under the attack."""
    changed = 0
    for point, label in data:
        _, _, clean = forward(parameters, point)
        _, _, attacked = forward(parameters, attack(parameters, point, label))
        predicted_clean = clean.index(max(clean))
        predicted_attacked = attacked.index(max(attacked))
        if predicted_clean != predicted_attacked or attacked[label] < clean[label] - 1e-9:
            changed += 1
    return changed / len(data)

def linear_model_gradient(weights, point, label):
    score = weights[0] * point[0] + weights[1] * point[1]
    probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
    scale = probability - label
    return [scale * weights[0], scale * weights[1]]

def fgsm_linear(weights, point, label, epsilon):
    """FGSM for the two-feature logistic model, used as the closed-form oracle."""
    gradient = linear_model_gradient(weights, point, label)
    return [min(max(point[index] + epsilon * (1.0 if gradient[index] > 0 else -1.0), 0.0), 1.0)
            for index in range(2)]

def main():
    data = make_dataset()
    check(len(data) == SAMPLES and all(0.0 <= x <= 1.0 for point, _ in data for x in point),
          "the dataset must be deterministic and inside the unit square")
    trained = train(data, initial_parameters())
    clean = accuracy(trained, data)
    check(clean > 0.95, "the trained network must fit the separable task")

    # 1. Analytic input gradient against central finite differences (the independent oracle).
    point, label = data[3]
    analytic = input_gradient(trained, point, label)
    numeric = []
    for feature in range(INPUT_DIM):
        plus, minus = list(point), list(point)
        plus[feature] += 1e-6
        minus[feature] -= 1e-6
        numeric.append((loss(trained, plus, label) - loss(trained, minus, label)) / 2e-6)
    check(max(abs(a - b) for a, b in zip(analytic, numeric)) < 1e-5,
          "the hand-written input gradient must match finite differences")

    # 2. Closed-form oracle: FGSM on a linear model is exactly x - eps sign(w) for a class-1 point.
    weights = [1.5, -1.0]
    linear_point = [0.4, 0.6]
    signed = fgsm_linear(weights, linear_point, 1, EPSILON)
    expected = [linear_point[0] - EPSILON * (1.0 if weights[0] > 0 else -1.0),
                linear_point[1] - EPSILON * (1.0 if weights[1] > 0 else -1.0)]
    check(signed == expected, "FGSM on a linear model must equal the closed-form solution")

    # 3. PGD: non-decreasing loss, at least as strong as FGSM, always inside the eps ball.
    attacked, trace = pgd(trained, point, label, EPSILON)
    check(all(trace[i + 1] >= trace[i] - 1e-9 for i in range(len(trace) - 1)),
          "PGD must never decrease the loss it is ascending")
    check(trace[-1] >= loss(trained, fgsm(trained, point, label, EPSILON), label) - 1e-9
          and trace[-1] >= trace[0],
          "PGD must be at least as strong as the single-step FGSM attack")
    check(all(abs(attacked[i] - point[i]) <= EPSILON + 1e-12 and 0.0 <= attacked[i] <= 1.0
              for i in range(INPUT_DIM)),
          "every adversarial image must stay inside the eps ball and the valid input range")

    # 4. eps = 0 is a no-op, and a large eps is still a valid image but a stronger attack.
    untouched = fgsm(trained, point, label, 0.0)
    check(untouched == point == pgd(trained, point, label, 0.0)[0],
          "a zero budget may not change a single pixel")
    wild = fgsm(trained, point, label, 0.5)
    check(all(0.0 <= value <= 1.0 for value in wild),
          "even an extreme budget must leave the image inside the valid range")

    # 5. Attack effectiveness: clean accuracy is high, attacked accuracy collapses, successes abound.
    attack = lambda parameters, point, label: pgd(parameters, point, label, EPSILON)[0]
    robust = accuracy(trained, data, attack)
    check(clean > 0.95 and robust < 0.3, "a 0.2 budget must be enough to break the undefended classifier")
    check(success_rate(trained, data, attack) > 0.9,
          "the attack must change the model's behaviour on almost every sample")

    # 6. Defence: adversarial training raises the robust accuracy against the very same attack.
    defended = train(data, initial_parameters(), epochs=EPOCHS, epsilon=0.15)
    defended_robust = accuracy(defended, data, attack)
    check(accuracy(defended, data) > 0.9, "the defended model must still classify clean inputs well")
    check(defended_robust > robust + 0.2, "adversarial training must materially improve robustness")

    # 7. Boundary cases: a single sample, and an attack against a degenerate all-zero gradient.
    solo = accuracy(trained, data[:1], attack)
    check(solo in (0.0, 1.0), "a single-sample evaluation must return a clean 0 or 1")
    flat = {"w1": [[0.0, 0.0]] * HIDDEN, "b1": [0.0] * HIDDEN,
            "w2": [[0.0] * HIDDEN] * CLASSES, "b2": [0.0] * CLASSES}
    check(fgsm(flat, point, label, EPSILON) == list(point),
          "a model with no gradient cannot move an input")

    # 8. Determinism: identical training runs and identical attacks.
    check(train(data, initial_parameters()) == trained
          and pgd(trained, point, label, EPSILON)[0] == attacked,
          "training and attacks must be reproducible")

    print(f"clean accuracy {clean:.3f}, PGD accuracy {robust:.3f} at eps={EPSILON}, "
          f"attack success {success_rate(trained, data, attack):.3f}")
    print(f"defended: clean {accuracy(defended, data):.3f}, robust {defended_robust:.3f}, "
          f"loss trace {trace[0]:.3f} -> {trace[-1]:.3f}")
    print("PASS")

def fgsm_linear(weights, point, label, epsilon):
    """FGSM for the two-feature logistic model, used as the closed-form oracle."""
    gradient = linear_model_gradient(weights, point, label)
    return [min(max(point[index] + epsilon * (1.0 if gradient[index] > 0 else -1.0), 0.0), 1.0)
            for index in range(2)]


if __name__ == "__main__":
    main()
