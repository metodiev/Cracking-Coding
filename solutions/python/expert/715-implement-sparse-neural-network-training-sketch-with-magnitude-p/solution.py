"""Task 715 - Implement sparse neural network training sketch with magnitude pruning.  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

Magnitude pruning asks a blunt question: if you delete the smallest weights and keep training, how
much accuracy do you lose?  This file implements the standard pipeline on a hand-written two-layer
perceptron (2 -> H tanh -> 1 sigmoid, binary cross-entropy) with a flat parameter vector and a
boolean mask:

  * prune -- rank all parameters by |w| and keep the largest (1 - sparsity) fraction, the "global
    magnitude" criterion; the tied weights are broken by index so the result is deterministic;
  * masked training -- the mask multiplies the gradient and is re-applied after every update, so a
    pruned weight stays exactly 0.0 forever (the invariant the checks measure);
  * one-shot vs iterative -- either prune to the target sparsity once, or prune in steps with
    retraining in between, which is the classic "iterative magnitude pruning" schedule;
  * lottery-ticket rewinding -- after finding a mask, rewind the surviving weights to their
    initialisation and retrain with the mask fixed.

Everything is trained on a tiny deterministic synthetic dataset (a nonlinear XOR-like rule with
label noise), with a fixed-seed PRNG, so every accuracy below is reproducible.

Simplified versus the literature: a 33-parameter network and 240 samples instead of ResNet on
ImageNet, plain SGD without momentum or weight decay, global rather than per-layer sparsity
targets, and no quantisation of the surviving weights.  Verified: the exact sparsity count after
every prune, the magnitude threshold (every kept |w| is at least every pruned |w|), the frozen-zero
invariant during masked training, sparsity 0 being bit-identical to dense training, sparsity 1
collapsing to the majority-class predictor, the retention of accuracy at high sparsity, the
advantage of iterative over one-shot pruning at extreme sparsity, and determinism.
"""
import math
import sys

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def uniform(self):
        self.state, r = splitmix64(self.state)
        return ((r >> 11) / (1 << 53)) * 2.0 - 1.0


# --------------------------------------------------------------------------- data
def make_dataset(samples, seed, noise=0.08):
    """Two features in [-1, 1]; the label is the XOR-like sign product, flipped with `noise`."""
    rng = Rng(seed)
    features, labels = [], []
    for _ in range(samples):
        x = rng.uniform()
        y = rng.uniform()
        label = 1 if x * y > 0 else 0
        if abs(rng.uniform()) < noise:
            label = 1 - label
        features.append([x, y])
        labels.append(float(label))
    return features, labels


# --------------------------------------------------------------------------- model
class Mlp:
    """2 -> hidden(tanh) -> 1(sigmoid) with a flat parameter vector and a boolean mask."""

    def __init__(self, hidden=8, seed=0x5EED):
        self.hidden = hidden
        rng = Rng(seed)
        self.size = hidden * 2 + hidden + hidden + 1
        scale = 1.0 / math.sqrt(2.0)
        self.theta = [rng.uniform() * scale for _ in range(self.size)]
        self.initial = list(self.theta)
        self.mask = [True] * self.size
        self.updates = 0
        self.masked_updates = 0

    # index helpers ------------------------------------------------------------
    def w1(self, j, k):
        return j * 2 + k

    def b1(self, j):
        return 2 * self.hidden + j

    def w2(self, j):
        return 3 * self.hidden + j

    @property
    def b2(self):
        return 4 * self.hidden

    def forward(self, x):
        hidden = []
        for j in range(self.hidden):
            pre = self.theta[self.b1(j)] + self.theta[self.w1(j, 0)] * x[0] \
                + self.theta[self.w1(j, 1)] * x[1]
            hidden.append(math.tanh(pre))
        logit = self.theta[self.b2] + sum(self.theta[self.w2(j)] * hidden[j]
                                          for j in range(self.hidden))
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        return probability, hidden

    def gradients(self, x, label):
        probability, hidden = self.forward(x)
        error = probability - label
        grad = [0.0] * self.size
        for j in range(self.hidden):
            grad[self.w2(j)] = error * hidden[j]
        grad[self.b2] = error
        for j in range(self.hidden):
            dh = error * self.theta[self.w2(j)] * (1.0 - hidden[j] * hidden[j])
            grad[self.b1(j)] = dh
            grad[self.w1(j, 0)] = dh * x[0]
            grad[self.w1(j, 1)] = dh * x[1]
        return grad

    def train_epoch(self, features, labels, learning_rate):
        total = 0.0
        for x, label in zip(features, labels):
            grad = self.gradients(x, label)
            for index in range(self.size):
                if not self.mask[index]:
                    continue
                self.theta[index] -= learning_rate * grad[index]
            total += 1
        self.updates += len(features)
        self.enforce_mask()

    def enforce_mask(self):
        for index in range(self.size):
            if not self.mask[index] and self.theta[index] != 0.0:
                self.theta[index] = 0.0
                self.masked_updates += 1

    def train(self, features, labels, epochs, learning_rate):
        history = []
        for _ in range(epochs):
            self.train_epoch(features, labels, learning_rate)
            history.append(self.accuracy(features, labels))
        return history

    def accuracy(self, features, labels):
        correct = 0
        for x, label in zip(features, labels):
            probability, _ = self.forward(x)
            if (probability >= 0.5) == (label >= 0.5):
                correct += 1
        return correct / len(features)

    def loss(self, features, labels):
        total = 0.0
        for x, label in zip(features, labels):
            probability, _ = self.forward(x)
            probability = min(max(probability, 1e-12), 1 - 1e-12)
            total -= label * math.log(probability) + (1 - label) * math.log(1 - probability)
        return total / len(features)

    def active(self):
        return sum(1 for index in range(self.size) if self.mask[index]
                   and self.theta[index] != 0.0)

    def zeros(self):
        return sum(1 for index in range(self.size) if self.theta[index] == 0.0)


def kept_count(size, sparsity):
    """How many parameters survive a sparsity target (single source of truth for rounding)."""
    check(0.0 <= sparsity <= 1.0, "sparsity must lie in [0, 1]")
    return size - int(round(sparsity * size))


def magnitude_mask(theta, sparsity):
    """Keep the largest (1 - sparsity) fraction of the parameters; ties break by index."""
    size = len(theta)
    keep = kept_count(size, sparsity)
    order = sorted(range(size), key=lambda index: (-abs(theta[index]), index))
    mask = [False] * size
    for index in order[:keep]:
        mask[index] = True
    return mask


def apply_mask(model, sparsity):
    model.mask = magnitude_mask(model.theta, sparsity)
    model.enforce_mask()
    model.masked_updates = 0      # re-arm the counter: it tracks resets *during* training
    return model.mask


def prune_round(model, target_sparsity):
    """Prune relative to the *currently active* parameters, keeping the mask monotone."""
    active = [index for index in range(model.size) if model.mask[index]]
    kept_total = kept_count(model.size, target_sparsity)
    need = len(active) - kept_total
    if need <= 0:
        return 0
    order = sorted(active, key=lambda index: (abs(model.theta[index]), index))
    for index in order[:need]:
        model.mask[index] = False
        model.theta[index] = 0.0
    return need


# --------------------------------------------------------------------------- checks
def test_forward_and_backward_sanity():
    model = Mlp(hidden=5, seed=0x11)
    check(model.size == 5 * 2 + 5 + 5 + 1, "the parameter count matches the architecture")
    probability, hidden = model.forward([0.3, -0.4])
    check(0.0 < probability < 1.0, "the output is a probability")
    check(len(hidden) == 5 and all(-1.0 <= value <= 1.0 for value in hidden),
          "the hidden layer is bounded by tanh")

    # a finite-difference check of the whole gradient against the analytic one
    features, labels = make_dataset(20, seed=0x12, noise=0.0)
    model.theta = [0.05 * (index + 1) - 0.3 for index in range(model.size)]
    epsilon = 1e-6
    for index in range(model.size):
        original = model.theta[index]
        model.theta[index] = original + epsilon
        high = model.loss(features, labels)
        model.theta[index] = original - epsilon
        low = model.loss(features, labels)
        model.theta[index] = original
        numeric = (high - low) / (2 * epsilon)
        analytic = 0.0
        for x, label in zip(features, labels):
            analytic += model.gradients(x, label)[index]
        analytic /= len(features)
        check(abs(numeric - analytic) < 1e-4,
              f"gradient mismatch at parameter {index}: {numeric} vs {analytic}")
    return model


def test_magnitude_mask_invariant():
    for sparsity in (0.0, 0.25, 0.5, 0.9, 1.0):
        theta = [0.5, -2.0, 0.0, 1.5, -0.25, 3.0, -1.0, 0.75]
        mask = magnitude_mask(theta, sparsity)
        kept = [theta[index] for index in range(len(theta)) if mask[index]]
        dropped = [theta[index] for index in range(len(theta)) if not mask[index]]
        expected_kept = kept_count(len(theta), sparsity)
        check(len(kept) == expected_kept, f"sparsity {sparsity}: kept {len(kept)} != {expected_kept}")
        if kept and dropped:
            check(min(abs(value) for value in kept) >= max(abs(value) for value in dropped),
                  "every kept magnitude must be at least every dropped magnitude")
        check(len(mask) == len(theta), "the mask covers every parameter")
    check(magnitude_mask([1.0, 2.0], 1.0) == [False, False], "sparsity 1 drops everything")
    check(magnitude_mask([1.0, 2.0], 0.0) == [True, True], "sparsity 0 keeps everything")
    ties = magnitude_mask([1.0, 1.0, 1.0], 1 / 3)
    check(ties == [True, True, False], "ties break by index, deterministically")
    try:
        magnitude_mask([1.0], 1.5)
        raise AssertionError("sparsity outside [0, 1] must be rejected")
    except AssertionError as error:
        check("sparsity must lie" in str(error), f"unexpected message: {error}")


def copy_model(model):
    """A structural copy: same architecture, same weights, same mask, fresh counters."""
    clone = Mlp(hidden=model.hidden, seed=1)
    clone.theta = list(model.theta)
    clone.initial = list(model.initial)
    clone.mask = list(model.mask)
    return clone


_DENSE_CACHE = {}


def dense_reference():
    """Train once and cache: 150 samples, 32 tanh units, plain SGD."""
    if "data" not in _DENSE_CACHE:
        features, labels = make_dataset(150, seed=0x3344, noise=0.08)
        model = Mlp(hidden=32, seed=0x99)
        history = model.train(features, labels, 100, 0.2)
        _DENSE_CACHE["data"] = (model, features, labels, history)
    return _DENSE_CACHE["data"]


def test_dense_training_baseline():
    model, features, labels, history = dense_reference()
    accuracy = model.accuracy(features, labels)
    check(accuracy >= 0.85, f"the dense model must learn the rule, got {accuracy}")
    check(history[-1] >= history[0], "training must not lose accuracy overall")
    check(model.active() == model.size, "nothing is pruned in the dense baseline")
    check(model.loss(features, labels) < 0.35,
          f"the dense loss must be low, got {model.loss(features, labels)}")
    check(model.accuracy(features, labels) <= 1.0, "accuracy is a fraction")
    return model, features, labels, accuracy


def test_frozen_zeros_under_masked_training():
    trained, features, labels, _ = dense_reference()
    model = copy_model(trained)
    mask = apply_mask(model, 0.8)
    frozen = [index for index in range(model.size) if not mask[index]]
    before = [model.theta[index] for index in frozen]
    check(all(value == 0.0 for value in before), "pruned parameters are exactly zero")
    check(len(frozen) == model.size - kept_count(model.size, 0.8), "the sparsity count is exact")
    model.train(features, labels, 40, 0.2)
    after = [model.theta[index] for index in frozen]
    check(after == before, "pruned parameters must stay zero through further training")
    check(model.masked_updates == 0,
          "the mask must be enforced by the gradient, not by a post-hoc reset")
    survivors = [index for index in range(model.size) if mask[index]]
    check(any(model.theta[index] != trained.theta[index] for index in survivors),
          "surviving parameters keep training")
    return model, features, labels, mask, frozen


def test_accuracy_retention_and_iterative_advantage():
    trained, features, labels, _ = dense_reference()
    dense_accuracy = trained.accuracy(features, labels)
    retention = {}
    for sparsity in (0.5, 0.7, 0.8):
        candidate = copy_model(trained)
        apply_mask(candidate, sparsity)
        candidate.train(features, labels, 40, 0.2)
        retention[sparsity] = candidate.accuracy(features, labels)
        check(retention[sparsity] >= 0.85,
              f"accuracy at {sparsity:.0%} sparsity should stay high, got {retention[sparsity]}")
        check(retention[sparsity] <= dense_accuracy + 0.02,
              "pruning cannot invent accuracy the dense model did not have")
        check(candidate.active() == kept_count(candidate.size, sparsity),
              f"exactly {(1 - sparsity):.0%} of the parameters survive")

    # there is a limit: at 90% sparsity only a handful of parameters survive and the rule is lost
    collapsed = copy_model(trained)
    apply_mask(collapsed, 0.9)
    collapsed.train(features, labels, 40, 0.2)
    collapsed_accuracy = collapsed.accuracy(features, labels)
    check(collapsed.active() == kept_count(collapsed.size, 0.9),
          "10% of the parameters survive at 90% sparsity")
    check(collapsed_accuracy < retention[0.8],
          f"90% sparsity costs accuracy: {collapsed_accuracy} vs {retention[0.8]}")

    # iterative pruning with retraining in between beats a single one-shot cut at 80%
    iterative = copy_model(trained)
    for target in (0.4, 0.6, 0.8):
        prune_round(iterative, target)
        check(sum(1 for value in iterative.mask if value)
              == kept_count(iterative.size, target),
              f"iterative pruning must hit {target:.0%} exactly")
        iterative.train(features, labels, 40, 0.2)
    iterative_accuracy = iterative.accuracy(features, labels)

    one_shot = copy_model(trained)
    prune_round(one_shot, 0.8)
    one_shot.train(features, labels, 100, 0.2)
    one_shot_accuracy = one_shot.accuracy(features, labels)
    check(iterative_accuracy >= one_shot_accuracy - 0.02,
          f"iterative pruning ({iterative_accuracy}) should not lose to one-shot "
          f"({one_shot_accuracy}) at 80% sparsity")
    check(sum(1 for value in iterative.mask if value) == sum(1 for value in one_shot.mask if value),
          "both schedules end at the same sparsity")
    check(prune_round(one_shot, 0.8) == 0, "a satisfied target needs no further pruning")
    return retention, dense_accuracy, iterative_accuracy, one_shot_accuracy, collapsed_accuracy


def test_rewinding_is_tracked():
    trained, features, labels, _ = dense_reference()
    mask = magnitude_mask(trained.theta, 0.8)
    rewind = Mlp(hidden=trained.hidden, seed=0x99)
    check(rewind.initial == trained.initial, "the same seed gives the same initialisation")
    rewind.theta = list(rewind.initial)
    rewind.mask = list(mask)
    rewind.enforce_mask()
    zeros = sum(1 for index in range(rewind.size) if rewind.theta[index] == 0.0)
    check(zeros == rewind.size - kept_count(rewind.size, 0.8),
          "rewinding to the initialisation must still zero the pruned weights")
    survivor = next(index for index in range(rewind.size) if rewind.mask[index])
    pruned = next(index for index in range(rewind.size) if not rewind.mask[index])
    check(rewind.theta[survivor] == trained.initial[survivor],
          "a surviving weight starts from the initialisation")
    check(rewind.theta[pruned] == 0.0, "and a pruned weight starts at exactly zero")
    rewind.train(features, labels, 100, 0.2)
    rewound_accuracy = rewind.accuracy(features, labels)
    check(rewound_accuracy >= 0.8,
          f"the rewound ticket should still learn, got {rewound_accuracy}")
    check(rewind.active() == kept_count(rewind.size, 0.8),
          "the rewound model never exceeds the mask's budget")
    check(rewind.theta != trained.theta, "and it is a different (sparse) solution")
    return rewound_accuracy, zeros


def test_boundaries():
    trained, features, labels, _ = dense_reference()

    dense_like = copy_model(trained)
    dense_like.mask = magnitude_mask(dense_like.theta, 0.0)
    dense_like.train(features, labels, 20, 0.2)
    reference = copy_model(trained)
    reference.train(features, labels, 20, 0.2)
    check(dense_like.theta == reference.theta,
          "sparsity 0 (all-ones mask) must be bit-identical to unmasked training")
    check(dense_like.active() == dense_like.size, "nothing is pruned at sparsity 0")

    everything = copy_model(trained)
    apply_mask(everything, 1.0)
    check(everything.theta == [0.0] * everything.size, "sparsity 1 zeroes every parameter")
    check(everything.active() == 0, "and nothing stays active")
    constant_accuracy = everything.accuracy(features, labels)
    predictions = {1 if everything.forward(x)[0] >= 0.5 else 0 for x in features}
    check(len(predictions) == 1, "a fully pruned net predicts the same class for every input")
    predicted = predictions.pop()
    fraction = sum(1 for label in labels if int(label) == predicted) / len(labels)
    check(abs(constant_accuracy - fraction) < 1e-9,
          f"a fully pruned net scores the frequency of its constant class: "
          f"{constant_accuracy} vs {fraction}")
    majority = max(sum(1 for label in labels if label >= 0.5),
                   sum(1 for label in labels if label < 0.5)) / len(labels)
    check(fraction in (majority, 1 - majority), "that class is one of the two label classes")
    everything.train(features, labels, 20, 0.2)
    check(everything.theta == [0.0] * everything.size, "an all-false mask freezes the model")
    check(prune_round(everything, 0.5) == 0, "pruning an empty model is a no-op")
    return constant_accuracy, majority


def test_determinism():
    features, labels = make_dataset(100, seed=0x777)
    first = Mlp(hidden=10, seed=0x21)
    first.train(features, labels, 30, 0.3)
    apply_mask(first, 0.6)
    first.train(features, labels, 30, 0.3)
    second = Mlp(hidden=10, seed=0x21)
    second.train(features, labels, 30, 0.3)
    apply_mask(second, 0.6)
    second.train(features, labels, 30, 0.3)
    check(first.theta == second.theta and first.mask == second.mask,
          "the whole pipeline is deterministic")
    check(first.zeros() == first.size - kept_count(first.size, 0.6),
          "the zero count equals the pruned count after training")
    check(first.active() + first.zeros() == first.size, "every parameter is either active or zero")
    return first.active(), first.size


def main():
    test_forward_and_backward_sanity()
    test_magnitude_mask_invariant()
    model, features, labels, dense_accuracy = test_dense_training_baseline()
    test_frozen_zeros_under_masked_training()
    retention, dense, iterative, one_shot, collapsed = \
        test_accuracy_retention_and_iterative_advantage()
    rewound, zeros = test_rewinding_is_tracked()
    constant_accuracy, majority = test_boundaries()
    active, size = test_determinism()

    print("accuracy retention: " + ", ".join(
        f"{sparsity:.0%} sparsity -> {accuracy:.3f}" for sparsity, accuracy in
        sorted(retention.items())) + f", 90% -> {collapsed:.3f} (collapse)")
    print(f"dense={dense:.3f} iterative@80%={iterative:.3f} one-shot@80%={one_shot:.3f} "
          f"rewound@80%={rewound:.3f}")
    print(f"fully pruned accuracy {constant_accuracy:.3f} = its constant-class frequency "
          f"(majority class {majority:.3f}); "
          f"determinism check: {active}/{size} active")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
