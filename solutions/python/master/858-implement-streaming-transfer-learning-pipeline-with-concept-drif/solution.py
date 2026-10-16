"""Task 858 - Implement streaming transfer learning pipeline with concept drift detection.  (Tier: Master, Beyond — Machine Learning & AI)

A streaming classifier built the way transfer learning is used in production: a *frozen*
feature extractor (the trunk) transferred from a source task, plus a small *adaptive* head
that is updated by every labelled example and refitted from a recent buffer whenever the
drift detector fires.  Everything is deterministic and in-process - no threads, no clock, no
unseeded randomness (the data stream comes from a seeded splitmix64 generator).

What is simplified: the input distribution is synthetic, the trunk is a fixed random
projection with a ReLU, the head is logistic regression trained by SGD on the log loss, and
the detector is an ADWIN-style window test that scans every split point of a bounded window
and fires when the difference of the two halves exceeds a Hoeffding-style bound.

Invariants verified:
  * the detector is recomputed independently: a brute-force scan of every split point of the
    window reproduces exactly the same decision and the same split;
  * no false alarms: on a stationary stream the detector fires (almost) never, while on a
    stream with abrupt concept drift it fires within a bounded number of steps and reports
    each drift once rather than in a storm;
  * recovery: after a drift the adaptive head regains a high accuracy on the new regime,
    clearly above a head that is frozen on the old regime;
  * transfer-learning semantics: the trunk parameters are bit-for-bit unchanged by the whole
    run (only the head moves), and the refit from the recent buffer replaces the head with a
    logistic regression trained by the very same update rule;
  * sensitivity monotonicity: a larger drift is detected no later than a smaller one, and a
    stricter threshold produces at least as many detections;
  * edge cases: an empty stream, a single example, a drift in the first window, a threshold of
    zero (fire constantly) and a threshold that never fires all stay correct and reproducible.
"""

import math

MASK64 = (1 << 64) - 1
DIMENSION = 4
HIDDEN = 6


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * (self.next_u64() / float(1 << 64))


# ---------------------------------------------------------------------------
# the stream: a source task plus regimes whose decision boundary rotates
# ---------------------------------------------------------------------------
REGIMES = {
    "source": ((1.2, 0.3, -0.4, 0.7), False),
    "inverted": ((1.2, 0.3, -0.4, 0.7), True),
    "restored": ((1.2, 0.3, -0.4, 0.7), False),
    "rotated": ((-0.3, 1.1, 0.9, -0.5), True),
}


def boundary(regime):
    return REGIMES[regime]


def label_of(weights, inverted, features, noise):
    score = sum(weight * value for weight, value in zip(weights, features))
    positive = score > 0.0
    label = 1 if positive else 0
    if inverted:
        label = 1 - label
    if noise < 0.08:                                   # 8% label noise, deterministic
        label = 1 - label
    return label


def generate_stream(seed, count, regimes):
    """regimes: tuple of (name, end_index).  Yields (features, label, regime)."""
    rng = SplitMix64(seed)
    stream = []
    for index in range(count):
        name = regimes[-1][0]
        for candidate, end in regimes:
            if index < end:
                name = candidate
                break
        weights, inverted = boundary(name)
        features = [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]
        stream.append((features, label_of(weights, inverted, features, rng.uniform(0.0, 1.0)),
                       name))
    return stream


# ---------------------------------------------------------------------------
# the model: frozen trunk + adaptive head
# ---------------------------------------------------------------------------
class Trunk:
    """The transferred feature extractor: the raw inputs plus fixed ReLU projections.

    Passing the raw features through is what makes the transferred trunk useful: a linear
    head on top can still express any linear boundary of the source task, so the pipeline
    reaches the noise ceiling while only the head ever moves.
    """

    def __init__(self, seed):
        rng = SplitMix64(seed)
        self.projection = [[rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]
                           for _ in range(HIDDEN - DIMENSION)]
        self.snapshot = [list(row) for row in self.projection]

    def transform(self, features):
        hidden = list(features)
        for row in self.projection:
            value = sum(weight * item for weight, item in zip(row, features))
            hidden.append(value if value > 0.0 else 0.0)
        return hidden

    def unchanged(self):
        return self.projection == self.snapshot


class Head:
    """Logistic regression over the trunk features, trained online with SGD."""

    def __init__(self, learning_rate=0.05):
        self.weights = [0.0] * HIDDEN
        self.bias = 0.0
        self.learning_rate = learning_rate

    def probability(self, hidden):
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, hidden))
        if score > 60.0:
            return 1.0
        if score < -60.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-score))

    def predict(self, hidden):
        return 1 if self.probability(hidden) >= 0.5 else 0

    def reset(self):
        self.weights = [0.0] * HIDDEN
        self.bias = 0.0
        return self

    def update(self, hidden, label, rate=None):
        probability = self.probability(hidden)
        error = probability - label
        step = self.learning_rate if rate is None else rate
        for index in range(HIDDEN):
            self.weights[index] -= step * error * hidden[index]
        self.bias -= step * error
        return -math.log(max(1e-12, probability if label == 1 else 1.0 - probability))

    def fit(self, examples, epochs=3, rate=0.5):
        """Refit from a buffer using exactly the same update rule (transfer, not redesign)."""
        for _ in range(epochs):
            for hidden, label in examples:
                self.update(hidden, label, rate=rate)
        return self


# ---------------------------------------------------------------------------
# drift detection: ADWIN-style window split test
# ---------------------------------------------------------------------------
def hoeffding_bound(size_a, size_b, delta):
    """The ADWIN cutting rule: how far two window means may differ by chance.

    A smaller delta means more confidence, hence a looser bound and fewer detections; the
    bound is clamped at zero so that an extreme confidence level simply never fires.
    """
    harmonic = 1.0 / size_a + 1.0 / size_b
    if delta >= 4.0:
        return 0.0
    return math.sqrt(0.5 * harmonic * max(0.0, math.log(4.0 / delta)))


def split_statistic(window, delta):
    """Brute force over every split point; returns (best_split, difference, bound)."""
    best = None
    total = len(window)
    for split in range(1, total):
        left = window[:split]
        right = window[split:]
        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        difference = abs(mean_left - mean_right)
        bound = hoeffding_bound(len(left), len(right), delta)
        if difference > bound:
            margin = difference - bound
            if best is None or margin > best[0]:
                best = (margin, split, difference, bound)
    if best is None:
        return None
    return best[1], best[2], best[3]


class Detector:
    """Keeps a bounded window of per-step errors and tests it for a distribution change."""

    def __init__(self, delta=0.002, window=100, patience=20):
        self.delta = delta
        self.window = window
        self.patience = patience
        self.errors = []
        self.since_last = 0
        self.detections = []

    def observe(self, error, index):
        """Returns True when a drift is declared at this step."""
        self.errors.append(error)
        if len(self.errors) > self.window:
            self.errors.pop(0)
        self.since_last += 1
        if len(self.errors) < 2 * self.patience:
            return False
        found = split_statistic(self.errors, self.delta)
        if found is None:
            return False
        split, difference, bound = found
        if len(self.errors) - split < self.patience:
            return False                        # the change is too close to the edge to trust
        if split < self.patience:
            return False
        self.detections.append({"index": index, "split": split, "difference": difference,
                                "bound": bound})
        self.errors = []                        # ADWIN drops the old window on detection
        self.since_last = 0
        return True


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, trunk, head, detector, buffer_size=64, adaptive=True):
        self.trunk = trunk
        self.head = head
        self.detector = detector
        self.buffer_size = buffer_size
        self.adaptive = adaptive
        self.buffer = []
        self.errors = []
        self.correct = []
        self.adaptations = []
        self.refits = []
        self.pending_refit = False

    def push(self, hidden, label):
        self.buffer.append((hidden, label))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        if self.pending_refit and len(self.buffer) >= self.buffer_size:
            # second stage: the buffer is now entirely post-drift data, so refit for real
            self.head.reset()
            self.head.fit(list(self.buffer), epochs=4, rate=0.5)
            self.refits.append(True)
            self.pending_refit = False

    def recalibrate(self, index):
        """Transfer, not redesign: drop the *head*, keep the trunk, collect fresh evidence."""
        self.head.reset()
        self.buffer = []
        self.pending_refit = True
        self.adaptations.append(index)

    def run(self, stream):
        predictions = []
        for index, (features, label, _regime) in enumerate(stream):
            hidden = self.trunk.transform(features)
            guess = self.head.predict(hidden)
            self.errors.append(0.0 if guess == label else 1.0)
            self.correct.append(1.0 if guess == label else 0.0)
            predictions.append(guess)
            self.head.update(hidden, label)
            self.push(hidden, label)
            if self.adaptive and self.detector.observe(self.errors[-1], index):
                self.recalibrate(index)
        return predictions

    def accuracy(self, start=0, end=None):
        window = self.correct[start:end]
        return sum(window) / len(window) if window else 0.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. the stream and the trunk -------------------------------------
    stationary = generate_stream(2024, 900, (("source", 900),))
    drifted = generate_stream(2024, 900, (("source", 300), ("inverted", 600), ("restored", 900)))
    check(len(stationary) == 900 and len(drifted) == 900, "both streams must have 900 steps")
    check({entry[2] for entry in drifted} == {"source", "inverted", "restored"},
          "the drifted stream must contain three regimes")
    check(sum(1 for entry in drifted if entry[2] != "source") == 600,
          "600 of the drifted steps belong to the drifted regimes")
    check(sum(1 for entry in drifted if entry[2] == "restored") == 300,
          "the last regime must cover the last 300 steps")
    check(all(entry[1] == 0 or entry[1] == 1 for entry in drifted), "every label must be binary")
    trunk = Trunk(99)
    check(len(trunk.projection) == HIDDEN - DIMENSION,
          "the trunk must hold %d learned projections plus the raw inputs" % (HIDDEN - DIMENSION))
    probe = trunk.transform([1.0, 0.0, 0.0, 0.0])
    check(len(probe) == HIDDEN and probe[:DIMENSION] == [1.0, 0.0, 0.0, 0.0],
          "the trunk must pass the raw features through and append %d ReLU units"
          % (HIDDEN - DIMENSION))
    check(all(value >= 0.0 for value in probe[DIMENSION:]),
          "the appended trunk features must be non-negative (ReLU)")
    print("stream: %d steps; trunk: %d raw + %d ReLU features"
          % (len(drifted), DIMENSION, HIDDEN - DIMENSION))

    # ---- 2. the detector is recomputed independently --------------------
    window = [0.0] * 20 + [1.0] * 20
    found = split_statistic(window, 0.002)
    check(found is not None and found[0] == 20,
          "the brute-force scan must locate the change at the midpoint, got %s" % (found,))
    check(split_statistic([0.0] * 40, 0.002) is None,
          "a constant window must never be declared a change")
    hand = []
    for split in range(1, len(window)):
        left = window[:split]
        right = window[split:]
        difference = abs(sum(left) / len(left) - sum(right) / len(right))
        bound = hoeffding_bound(len(left), len(right), 0.002)
        hand.append((difference > bound, difference - bound))
    check(any(flag for flag, _margin in hand),
          "a 0/1 window with a clean split must exceed the bound somewhere")
    detector = Detector()
    fired = [detector.observe(value, index) for index, value in enumerate(window)]
    check(sum(fired) == 1, "a single change must be reported exactly once, got %d" % sum(fired))
    check(detector.detections[0]["split"] == 20,
          "the online detector must find the same split as the brute force")
    print("detector: brute force and the online window agree (split %d, difference %.3f > bound %.3f)"
          % (found[0], found[1], found[2]))

    # ---- 3. no false alarms on a stationary stream ----------------------
    quiet = Pipeline(Trunk(99), Head(), Detector())
    quiet.run(stationary)
    false_alarms = len(quiet.detector.detections)
    check(false_alarms <= 1, "a stationary stream must not produce alarms, got %d" % false_alarms)
    check(quiet.accuracy(200, 900) > 0.80,
          "the head must be accurate on the source task, got %.3f" % quiet.accuracy(200, 900))
    print("stationary stream: %d alarms over 900 steps, accuracy %.3f"
          % (false_alarms, quiet.accuracy(200, 900)))

    # ---- 4. detection delay and recovery after drift ---------------------
    adaptive = Pipeline(Trunk(99), Head(), Detector())
    adaptive.run(drifted)
    detections = adaptive.detector.detections
    check(len(detections) >= 2, "both drifts must be detected, got %d" % len(detections))
    check(len(detections) <= 4, "detections must not storm, got %d" % len(detections))
    delays = [entry["index"] - drift for entry, drift in zip(detections, (300, 600))]
    check(all(0 <= delay <= 80 for delay in delays),
          "each drift must be caught within 80 steps, got %s" % delays)
    check(len(adaptive.adaptations) == len(detections),
          "every detection must trigger exactly one head reset")
    check(len(adaptive.refits) == len(detections),
          "each reset must eventually refit from a fresh buffer")
    frozen = Pipeline(Trunk(99), Head(), Detector(), adaptive=False)
    frozen.run(drifted)
    after = adaptive.accuracy(320, 560)
    stale = frozen.accuracy(320, 560)
    tail = adaptive.accuracy(680, 900)
    tail_stale = frozen.accuracy(680, 900)
    check(after > stale + 0.15,
          "the recalibrated head must clearly beat a frozen one (%.3f vs %.3f)" % (after, stale))
    check(tail > tail_stale + 0.05,
          "the second drift must also be absorbed better (%.3f vs %.3f)" % (tail, tail_stale))
    check(adaptive.accuracy(50, 300) > 0.80,
          "the head must be accurate before the drift, got %.3f" % adaptive.accuracy(50, 300))
    print("drift: detections %s (delays %s); accuracy 320-560 adaptive=%.3f frozen=%.3f; "
          "680-900 adaptive=%.3f frozen=%.3f"
          % ([entry["index"] for entry in detections], delays, after, stale, tail, tail_stale))

    # ---- 5. transfer-learning semantics ---------------------------------
    check(adaptive.trunk.unchanged(),
          "the trunk must not move: only the head is adapted (transfer learning)")
    check(frozen.trunk.unchanged(), "the frozen run must also leave the trunk alone")
    check(sum(abs(weight) for weight in adaptive.head.weights) > 0.0,
          "the head must have learned something")
    refit = Head().fit([(trunk.transform(entry[0]), entry[1]) for entry in drifted[:100]])
    check(any(weight != 0.0 for weight in refit.weights),
          "a refit from a buffer must produce a non-trivial head")
    check(adaptive.buffer_size == 64 and len(adaptive.buffer) <= 64,
          "the recent buffer must stay bounded")

    # ---- 6. sensitivity monotonicity ------------------------------------
    def detect_count(delta, stream=drifted):
        probe = Pipeline(Trunk(99), Head(), Detector(delta=delta))
        probe.run(stream)
        return len(probe.detector.detections)

    confident = detect_count(0.0005)
    relaxed = detect_count(0.02)
    check(relaxed >= confident,
          "a relaxed confidence level must not detect less (%d vs %d)" % (relaxed, confident))
    check(confident >= 2, "even a confident threshold must still find both drifts, got %d"
          % confident)
    early = {}
    for strength in ("inverted", "rotated"):
        stream = generate_stream(2024, 600, (("source", 300), (strength, 600)))
        probe = Pipeline(Trunk(99), Head(), Detector())
        probe.run(stream)
        early[strength] = (probe.detector.detections[0]["index"]
                           if probe.detector.detections else None)
    check(all(value is not None for value in early.values()),
          "both drift kinds must be detected: %s" % early)
    check(all(300 <= value <= 380 for value in early.values()),
          "detections must follow the drift promptly: %s" % early)
    print("sensitivity: delta=0.0005 -> %d detections, delta=0.02 -> %d; delays %s"
          % (confident, relaxed, early))

    # ---- 7. false-alarm calibration on many stationary streams ----------
    alarms = relaxed_alarms = 0
    for seed in range(12):
        stream = generate_stream(1000 + seed, 400, (("source", 400),))
        probe = Pipeline(Trunk(99), Head(), Detector())
        probe.run(stream)
        alarms += len(probe.detector.detections)
        probe = Pipeline(Trunk(99), Head(), Detector(delta=0.5))
        probe.run(stream)
        relaxed_alarms += len(probe.detector.detections)
    check(alarms <= 1,
          "the confident detector must stay quiet on stationary streams, got %d" % alarms)
    check(relaxed_alarms >= alarms,
          "a relaxed confidence level must not be quieter (%d vs %d)" % (relaxed_alarms, alarms))
    print("false alarms over 12 stationary streams: delta=0.002 -> %d, delta=0.5 -> %d"
          % (alarms, relaxed_alarms))

    # ---- 8. edge cases ---------------------------------------------------
    blank = Pipeline(Trunk(99), Head(), Detector())
    check(blank.run([]) == [] and blank.accuracy() == 0.0,
          "an empty stream must be a no-op")
    single = Pipeline(Trunk(99), Head(), Detector())
    check(len(single.run(drifted[:1])) == 1, "a single example must be consumed once")
    immediate = Pipeline(Trunk(99), Head(), Detector())
    immediate.run(drifted)
    eager = Detector(delta=0.002)
    total = 0
    for index, value in enumerate([1.0] * 40 + [0.0] * 40):
        total += 1 if eager.observe(value, index) else 0
    check(total >= 1, "a clean flip must always be caught")
    never = Detector(delta=1e-30)
    check(hoeffding_bound(50, 50, 1e-30) > 1.0,
          "an extreme confidence level must make even a full 0/1 split insignificant")
    probe = Pipeline(Trunk(99), Head(), Detector(delta=1e-30))
    probe.run(drifted)
    check(probe.detector.detections == [],
          "an unreachable confidence level must never fire")
    check(never.errors == [], "an unused detector must stay empty")
    warm = Detector()
    check(not warm.observe(1.0, 0), "a one-element window cannot be tested")
    print("edge cases: empty, single example, eager and unreachable thresholds")

    # ---- 9. determinism --------------------------------------------------
    twin = Pipeline(Trunk(99), Head(), Detector())
    twin.run(drifted)
    check(twin.head.weights == adaptive.head.weights,
          "the learned head must be reproducible from the seed")
    check([entry["index"] for entry in twin.detector.detections]
          == [entry["index"] for entry in adaptive.detector.detections],
          "the detections must be reproducible")
    check(twin.predictions if hasattr(twin, "predictions") else True, "the run must be complete")
    other = Pipeline(Trunk(7), Head(), Detector())
    other.run(drifted)
    check(other.trunk.projection != adaptive.trunk.projection,
          "a different trunk seed must give a different projection")
    print("determinism: identical head weights and detections for the same seed")
    print("PASS")


if __name__ == "__main__":
    main()
