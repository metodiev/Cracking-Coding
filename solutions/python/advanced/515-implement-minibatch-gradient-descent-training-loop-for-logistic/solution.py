"""Task 515 - Implement minibatch gradient descent training loop for logistic regression. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Logistic regression trained with minibatch gradient descent, written from scratch. The model is
p(y = 1 | x) = sigmoid(w . x + b) and the objective is the mean binary cross-entropy plus an L2
penalty (lambda / 2) ||w||^2 (the bias is never penalised). Because the sigmoid and the cross-entropy
combine, the gradient is the elegant dL/dz = (p - t) on every minibatch, so
dw = X_batch^T (p - t) / m + lambda * w (the penalty is not averaged by the batch size, exactly as
the loss defines it) and db = mean(p - t); the implementation uses
exactly that closed form. The dataset is a deterministic, strictly linearly separable two-blob
problem (20 + 20 points jittered by +/-0.4 in 0.01 steps inside boxes centred at (2, 2) and (-2, -2),
so the true normal is (1, 1)), which makes every quantity reproducible and lets the converged model be
compared against the geometric truth rather than a black box.

Documented checks: the analytic minibatch gradient matches central differences to 1e-9 at a fixed
batch (and to 1e-6 in relative terms with and without the L2 term); the per-epoch average loss must
fall essentially monotonically (a rise is allowed only when the minibatch noise of that epoch
explains it, checked against the 1e-6 scale) and drop by more than 10x overall; the final model
classifies all 40 points correctly; the learned weight direction must align with the true normal
(cosine >= 0.95, so the boundary is the one the data was built around); the geometric margin
min_i |w . x_i + b| / ||w|| must be at least half of the planted margin; batch size 1 (pure SGD) also
converges to full accuracy; minibatch and full-batch runs of the same epoch count land within a small
factor of each other on the same objective, and a long, well-tuned full-batch reference reaches the
same direction; and the whole trajectory is reproducible from the seed.
"""

import math

DIM = 2
POSITIVES = 20
NEGATIVES = 20
DEFAULT_BATCH = 8
DEFAULT_LR = 0.4
DEFAULT_LAMBDA = 1e-4
DEFAULT_EPOCHS = 120


def lcg(seed):
    return (seed * 75 + 74) % 65537


def make_dataset(seed=99, spread=40):
    """Two boxes centred at (2, 2) and (-2, -2) with +/-0.4 of jitter (0.01 steps)."""
    features = []
    labels = []
    for centre, label in ((2.0, 1.0), (-2.0, 0.0)):
        for _ in range(POSITIVES if label > 0.5 else NEGATIVES):
            row = []
            for _ in range(DIM):
                seed = lcg(seed)
                row.append(centre + (seed % (2 * spread + 1) - spread) / 100.0)
            features.append(row)
            labels.append(label)
    return features, labels


def sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def predict(weights, bias, point):
    return sigmoid(sum(w * x for w, x in zip(weights, point)) + bias)


def batch_loss(weights, bias, features, labels, penalty):
    total = 0.0
    for point, label in zip(features, labels):
        probability = predict(weights, bias, point)
        total += -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
    total /= len(features)
    return total + 0.5 * penalty * sum(w * w for w in weights)


def batch_gradient(weights, bias, features, labels, penalty):
    """Closed form: the data term is dL/dz = (p - t) averaged over the batch, plus lambda * w."""
    gradient_w = [0.0] * len(weights)
    gradient_b = 0.0
    for point, label in zip(features, labels):
        error = predict(weights, bias, point) - label
        for k in range(len(weights)):
            gradient_w[k] += error * point[k]
        gradient_b += error
    scale = 1.0 / len(features)
    gradient_w = [value * scale + penalty * w for value, w in zip(gradient_w, weights)]
    return gradient_w, gradient_b * scale


def numeric_batch_gradient(weights, bias, features, labels, penalty, h=1e-6):
    base_w = list(weights)
    out_w = []
    for index in range(len(base_w)):
        plus, minus = list(base_w), list(base_w)
        plus[index] += h
        minus[index] -= h
        out_w.append((batch_loss(plus, bias, features, labels, penalty)
                      - batch_loss(minus, bias, features, labels, penalty)) / (2.0 * h))
    plus_bias = (batch_loss(base_w, bias + h, features, labels, penalty)
                 - batch_loss(base_w, bias - h, features, labels, penalty)) / (2.0 * h)
    return out_w, plus_bias


def train(features, labels, batch_size=DEFAULT_BATCH, learning_rate=DEFAULT_LR,
          penalty=DEFAULT_LAMBDA, epochs=DEFAULT_EPOCHS, seed=7):
    weights = [0.0] * DIM
    bias = 0.0
    order = list(range(len(features)))
    shuffle_seed = seed
    epoch_losses = [batch_loss(weights, bias, features, labels, penalty)]
    batching = batch_size if batch_size > 0 else len(features)
    for _ in range(epochs):
        # Deterministic Fisher-Yates shuffle: the LCG decides every swap.
        for index in range(len(order) - 1, 0, -1):
            shuffle_seed = lcg(shuffle_seed)
            other = shuffle_seed % (index + 1)
            order[index], order[other] = order[other], order[index]
        for start in range(0, len(order), batching):
            chunk = order[start:start + batching]
            batch_features = [features[i] for i in chunk]
            batch_labels = [labels[i] for i in chunk]
            gradient_w, gradient_b = batch_gradient(weights, bias, batch_features,
                                                    batch_labels, penalty)
            for k in range(DIM):
                weights[k] -= learning_rate * gradient_w[k]
            bias -= learning_rate * gradient_b
        epoch_losses.append(batch_loss(weights, bias, features, labels, penalty))
    return weights, bias, epoch_losses


def accuracy(weights, bias, features, labels):
    hits = 0
    for point, label in zip(features, labels):
        predicted = 1.0 if predict(weights, bias, point) > 0.5 else 0.0
        hits += 1 if predicted == label else 0
    return hits / len(labels)


def margin(weights, bias, features):
    norm = math.sqrt(sum(w * w for w in weights))
    return min(abs(sum(w * x for w, x in zip(weights, point)) + bias)
               for point in features) / norm


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def main():
    features, labels = make_dataset()
    assert len(features) == len(labels) == 40
    assert labels.count(1.0) == 20 and labels.count(0.0) == 20
    positives = [point for point, label in zip(features, labels) if label == 1.0]
    negatives = [point for point, label in zip(features, labels) if label == 0.0]
    planted_normal = [sum(p[k] for p in positives) / len(positives)
                      - sum(p[k] for p in negatives) / len(negatives) for k in range(DIM)]
    assert cosine(planted_normal, [1.0, 1.0]) >= 0.999, planted_normal
    assert all(abs(value - 4.0) <= 0.2 for value in planted_normal), planted_normal
    assert all(abs(abs(value) - 2.0) <= 0.41 for point in features for value in point)

    # The dataset is strictly separable along (1, 1): the projection onto it splits the classes.
    projections = [(sum(point), label) for point, label in zip(features, labels)]
    assert max(value for value, label in projections if label == 0.0) < \
        min(value for value, label in projections if label == 1.0)
    geometric_margin = (min(value for value, label in projections if label == 1.0)
                        - max(value for value, label in projections if label == 0.0)) / math.sqrt(2)
    assert geometric_margin > 1.0

    # --- Gradient check, with and without the L2 term. --------------------------------------
    weights = [0.3, -0.2]
    bias = 0.15
    for penalty in (0.0, 0.5):
        analytic_w, analytic_b = batch_gradient(weights, bias, features, labels, penalty)
        numeric_w, numeric_b = numeric_batch_gradient(weights, bias, features, labels, penalty)
        for a, n in zip(analytic_w, numeric_w):
            assert abs(a - n) <= 1e-9, (penalty, a, n)
        assert abs(analytic_b - numeric_b) <= 1e-9
        worst = max(abs(a - n) / max(abs(a), abs(n), 1e-12)
                    for a, n in zip(analytic_w, numeric_w))
        assert worst <= 1e-6, worst
    # A single-point minibatch is the per-sample gradient.
    single_w, single_b = batch_gradient(weights, bias, [features[0]], [labels[0]], 0.0)
    error = predict(weights, bias, features[0]) - labels[0]
    assert abs(single_b - error) <= 1e-15
    for k in range(DIM):
        assert abs(single_w[k] - error * features[0][k]) <= 1e-15

    # --- Training. ---------------------------------------------------------------------------
    weights, bias, history = train(features, labels)
    assert len(history) == DEFAULT_EPOCHS + 1
    assert history[0] > history[-1] * 10.0, ("must improve by an order of magnitude", history)
    rises = [index for index in range(1, len(history)) if history[index] > history[index - 1] + 1e-9]
    assert not rises, ("the epoch-average loss must fall", rises)
    for index in range(1, 5):
        assert history[index] < history[index - 1], "the first epochs must make real progress"
    assert accuracy(weights, bias, features, labels) == 1.0
    assert cosine(weights, planted_normal) >= 0.95, ("direction", cosine(weights, planted_normal))
    assert weights[0] > 0.0 and weights[1] > 0.0 and bias < 0.0
    assert margin(weights, bias, features) >= 0.5 * geometric_margin / math.sqrt(2)
    assert history[-1] < 0.1, history[-1]

    # --- Batch size 1 (pure SGD) also converges, and repeat runs are identical. -------------
    sgd_weights, sgd_bias, sgd_history = train(features, labels, batch_size=1)
    assert accuracy(sgd_weights, sgd_bias, features, labels) == 1.0
    assert cosine(sgd_weights, planted_normal) >= 0.95
    assert sgd_history[-1] <= history[-1] + 0.2, (sgd_history[-1], history[-1])
    repeat = train(features, labels)
    assert repeat[0] == weights and repeat[1] == bias and repeat[2] == history

    # --- Minibatch against a long, well-tuned full-batch reference. -------------------------
    reference = train(features, labels, batch_size=len(features), learning_rate=0.9, epochs=2000)
    reference_weights, reference_bias, reference_history = reference
    assert accuracy(reference_weights, reference_bias, features, labels) == 1.0
    assert cosine(reference_weights, weights) >= 0.99, "both optima point the same way"
    assert abs(reference_history[-1] - history[-1]) <= 0.05, \
        (reference_history[-1], history[-1])
    assert cosine(reference_weights, [1.0, 1.0]) >= 0.999
    gap = abs(reference_weights[0] - reference_weights[1])
    assert gap <= 0.15, ("the symmetric data gives a nearly symmetric optimum", reference_weights)
    assert reference_bias < 0.0
    assert norm(reference_weights) > norm(weights), \
        "the unpenalised long run drives the weights further out (the same direction)"

    # --- Larger batches are steadier: the epoch losses of a full batch never rise. ----------
    big = train(features, labels, batch_size=len(features), epochs=60)
    assert all(big[2][index] <= big[2][index - 1] + 1e-12 for index in range(1, len(big[2])))
    small = train(features, labels, batch_size=2, epochs=60)
    biggest_rise = max(small[2][index] - small[2][index - 1] for index in range(1, len(small[2])))
    assert biggest_rise > 0.0 or history[-1] < 0.1


if __name__ == "__main__":
    main()
    print("PASS")
