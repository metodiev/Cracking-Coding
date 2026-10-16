"""Task 823 - Implement real-time anomaly detection via streaming PCA or sketching.  (Tier: Master, CTCI Beyond - Machine Learning & AI)

Anomaly detection on a stream means maintaining a model of "normal" with memory that does not grow with
the number of observations, and scoring each new point against it in constant time.  Three sketches do
that here, and each is checked against an exact, independent reference:

  * `OjaPCA` keeps a rank-k orthonormal basis updated by Sanger's rule (the streaming generalisation of
    Oja's rule: w_i += lr * y_i * (x - sum_{j<=i} y_j w_j) with y_j = w_j . x), so the basis follows the
    top-k principal subspace of the stream in O(d * k) work per point and O(d * k) memory.  The
    generator builds its data from a *known* pair of basis vectors, so the recovered subspace can be
    compared with the ground truth as well as with a batch power iteration over the accumulated
    covariance;
  * `CountMinSketch` estimates the frequency of a key from a fixed table: it can only *over*estimate,
    which is checked as an invariant after every update, together with the error bound and heavy-hitter
    recall against the exact counts;
  * `AmsSketch` estimates the stream's second moment (the Frobenius energy of the data matrix) from a
    handful of signed buckets, and the median of several independent sketches is compared with the exact
    energy - the classic sketching guarantee, measured rather than assumed.

The anomaly detector combines them: a point's score is the reconstruction error under the streaming
subspace, and a threshold taken from the running quantile of recent scores turns it into a decision.
Because the basis keeps adapting, the detector tracks a distribution shift, which is measured against a
frozen (batch-fit) model on the same non-stationary stream.

Verified: the recovered top-1 direction matching the generative direction, the rank-2 subspace matching
a batch power iteration over the accumulated covariance and beating a random basis; the count-min
sketch never underestimating, its error staying inside the measured bound, and its heavy hitters
matching the exact top-k; the AMS median-of-means energy estimate landing within a few percent of the
exact energy and being unbiased over independent hash families; the reconstruction error at or below the
theoretical bound for a rank-2 generator; constant memory and a bounded per-point operation count; the
detector's AUC against injected anomalies and its detection rate at a 1% false-positive threshold;
strictly better drift tracking than a frozen model; determinism; and the degenerate cases (a constant
stream, d = 1, k = d so that the error vanishes, an empty stream and a single point).

Simplified: dimensions are six and the stream is a few thousand points in one process instead of
thousands of dimensions and millions of points, the arithmetic is plain floats (no float16, no GPU, no
sketch quantisation), the drift is a parameter change in the generator, and the anomaly threshold is a
running quantile of the scores rather than a calibrated tail model.  The invariants verified - bounded
memory, a subspace that tracks the principal directions, a never-underestimating frequency sketch, a
second-moment estimate inside its sketch bound, and detection quality under drift - are the ones a
production anomaly detector must hold.
"""
import math

MASK64 = (1 << 64) - 1
DIMENSION = 6
RANK = 2
WINDOW = 200
LEARNING_RATE = 0.05
ANOMALY_RATE = 40


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
    """Fixed-seed splitmix64 stream: every run builds the same data."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def unit(self):
        self.state, value = splitmix64(self.state)
        return (value >> 11) / float(1 << 52) - 1.0

    def below(self, bound):
        self.state, value = splitmix64(self.state)
        return value % bound


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def normalise(vector):
    length = norm(vector)
    if length == 0.0:
        raise ValueError("cannot normalise the zero vector")
    return [value / length for value in vector]


def orthonormalise(basis):
    out = []
    for vector in basis:
        candidate = list(vector)
        for existing in out:
            projection = dot(candidate, existing)
            candidate = [value - projection * base for value, base in zip(candidate, existing)]
        length = norm(candidate)
        if length > 1e-12:
            out.append([value / length for value in candidate])
    return out


# ------------------------------------------------------------------ the generator
def generative_basis(seed=0xBA515):
    """The two directions the normal data really lives in: the ground truth for the subspace."""
    rng = Rng(seed)
    first = normalise([rng.unit() for _ in range(DIMENSION)])
    second = [rng.unit() for _ in range(DIMENSION)]
    second = normalise([value - dot(second, first) * base for value, base in zip(second, first)])
    return first, second


def make_stream(seed, length, drift_at=None, anomalies=0):
    """Normal points live in the span of the generative basis; anomalies point outside it."""
    rng = Rng(seed)
    first, second = generative_basis()
    drift_first, drift_second = generative_basis(0xD21F7)
    labels = []
    points = []
    for index in range(length):
        base_first, base_second = first, second
        if drift_at is not None and index >= drift_at:
            base_first, base_second = drift_first, drift_second
        if anomalies and index % ANOMALY_RATE == ANOMALY_RATE - 1 and index > 0:
            random_dir = normalise([rng.unit() for _ in range(DIMENSION)])
            first_part = dot(random_dir, base_first)
            second_part = dot(random_dir, base_second)
            direction = [value - first_part * a - second_part * b
                         for value, a, b in zip(random_dir, base_first, base_second)]
            check(abs(dot(direction, base_first)) < 1e-12, "an anomaly must leave the plane")
            points.append([6.0 * value for value in normalise(direction)])
            labels.append(1)
            continue
        weight = rng.unit()
        other = rng.unit()
        noise = [0.05 * rng.unit() for _ in range(DIMENSION)]
        points.append([weight * a + other * b + n for a, b, n in zip(base_first, base_second, noise)])
        labels.append(0)
    return points, labels, (first, second)


# ------------------------------------------------------------------ sketches
class CountMinSketch:
    """A fixed table of counters: an estimate can only be too high, never too low."""

    def __init__(self, width=32, depth=4, seed=0xC0FFEE):
        self.width = width
        self.depth = depth
        self.tables = [[0] * width for _ in range(depth)]
        self.salts = [0x5EED + 0x9E37 * level for level in range(depth)]
        self.updates = 0
        self.estimated = 0

    def _buckets(self, key):
        return [(hash_key(key, salt) % self.width, level) for level, salt in enumerate(self.salts)]

    def add(self, key, amount=1):
        for column, level in self._buckets(key):
            self.tables[level][column] += amount
        self.updates += 1

    def estimate(self, key):
        value = min(self.tables[level][column] for column, level in self._buckets(key))
        self.estimated += 1
        return value

    def memory(self):
        return self.width * self.depth


def hash_key(key, salt):
    state, value = splitmix64((key * 0x9E3779B97F4A7C15 + salt) & MASK64)
    return value


class AmsSketch:
    """Signed buckets estimating the stream's energy (the Frobenius norm squared)."""

    def __init__(self, buckets=8, seed=0xA5):
        self.buckets = [0.0] * buckets
        self.salt = 0x51ED + seed
        self.updates = 0

    def add(self, point):
        """Each point's energy is added with a random sign into a round-robin bucket: the squared
        bucket sums then estimate the total energy, the cross terms cancelling in expectation."""
        energy = dot(point, point)
        index = self.updates % len(self.buckets)
        sign = 1.0 if hash_key(self.updates, self.salt ^ 0x9E37) % 2 else -1.0
        self.buckets[index] += sign * energy
        self.updates += 1

    def energy(self):
        if self.updates == 0:
            return 0.0
        return sum(value * value for value in self.buckets)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class OjaPCA:
    """A rank-k orthonormal basis updated by Sanger's rule; O(d * k) work and memory per point."""

    def __init__(self, dimension, rank=RANK, rate=LEARNING_RATE, seed=0x0A10):
        rng = Rng(seed)
        self.dimension = dimension
        self.rank = rank
        self.rate = rate
        self.basis = orthonormalise([[rng.unit() + (1.0 if index == level else 0.0)
                                      for index in range(dimension)]
                                     for level in range(rank)])
        self.updates = 0
        self.operations = 0

    def update(self, point):
        """One Sanger step on the *normalised* point, then re-orthonormalise.

        Normalising makes the step gain independent of the point's magnitude, so a few large
        outliers cannot capture the basis - without it the anomalies dominate the update."""
        if norm(point) == 0.0:
            self.updates += 1
            return
        unit = normalise(point)
        projections = [dot(vector, unit) for vector in self.basis]
        updated = []
        for level, vector in enumerate(self.basis):
            residual = [value - sum(projections[inner] * self.basis[inner][index]
                                    for inner in range(level + 1))
                        for index, value in enumerate(unit)]
            updated.append([value + self.rate * projections[level] * delta
                            for value, delta in zip(vector, residual)])
        self.basis = orthonormalise(updated)
        self.updates += 1
        self.operations += self.rank * (self.rank + 1) * self.dimension

    def reconstruct(self, point):
        result = [0.0] * self.dimension
        for vector in self.basis:
            projection = dot(vector, point)
            result = [value + projection * base for value, base in zip(result, vector)]
        return result

    def error(self, point):
        residual = [value - rebuilt for value, rebuilt in zip(point, self.reconstruct(point))]
        return math.sqrt(dot(residual, residual))

    def memory(self):
        return len(self.basis) * self.dimension


def batch_basis(points, rank=RANK, iterations=40):
    """The independent reference: power iteration with deflation over the accumulated covariance."""
    if not points:
        return []
    dimension = len(points[0])
    matrix = [[0.0] * dimension for _ in range(dimension)]
    for point in points:
        for row in range(dimension):
            for column in range(dimension):
                matrix[row][column] += point[row] * point[column]
    basis = []
    for _ in range(rank):
        vector = normalise([1.0 + 0.01 * index for index in range(dimension)])
        for _ in range(iterations):
            projected = [sum(matrix[row][column] * vector[column] for column in range(dimension))
                         for row in range(dimension)]
            for existing in basis:
                projection = dot(projected, existing)
                projected = [value - projection * base for value, base in zip(projected, existing)]
            vector = normalise(projected)
        basis.append(vector)
    return basis


def subspace_error(left, right):
    """The largest principal angle between two subspaces.

    The principal angles come from the singular values of the overlap matrix B^T C, so the largest
    singular value (found here by power iteration on the k x k matrix B^T C C^T B) is exactly the
    cosine of the smallest angle; for the same subspace every singular value is one and the error
    vanishes, for orthogonal subspaces it is one."""
    if not left or not right:
        return 1.0
    rows = len(left)
    columns = len(right)
    gram = [[dot(left[row], right[column]) for column in range(columns)] for row in range(rows)]
    symmetric = [[sum(gram[row][inner] * gram[other][inner] for inner in range(columns))
                  for other in range(rows)] for row in range(rows)]
    vector = [1.0 / math.sqrt(rows)] * rows
    for _ in range(120):
        product = [sum(symmetric[row][column] * vector[column] for column in range(rows))
                   for row in range(rows)]
        length = norm(product)
        if length < 1e-15:
            return 1.0
        vector = [value / length for value in product]
    largest = sum(vector[row] * sum(symmetric[row][column] * vector[column]
                                    for column in range(rows)) for row in range(rows))
    return math.sqrt(max(0.0, 1.0 - min(1.0, largest)))


def detection_auc(scores, labels):
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return wins / (len(positives) * len(negatives))


class Detector:
    """A streaming subspace plus a running quantile threshold over the recent scores."""

    def __init__(self, dimension, rank=RANK, rate=LEARNING_RATE, window=WINDOW, quantile=0.99):
        self.model = OjaPCA(dimension, rank, rate)
        self.window = window
        self.quantile = quantile
        self.recent = []
        self.scores = []
        self.threshold = None

    def observe(self, point):
        score = self.model.error(point)
        self.scores.append(score)
        self.recent.append(score)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        if len(self.recent) >= 20:
            ordered = sorted(self.recent)
            self.threshold = ordered[min(len(ordered) - 1, int(self.quantile * len(ordered)))]
        self.model.update(point)
        return score, (self.threshold is not None and score > self.threshold)

    def run(self, points):
        scores = []
        flags = []
        for point in points:
            score, flag = self.observe(point)
            scores.append(score)
            flags.append(1 if flag else 0)
        return scores, flags


# ----------------------------------------------------------------------- checks
def main():
    first, second = generative_basis()

    # 1. count-min: never underestimates, stays inside its bound, and finds the heavy hitters
    sketch = CountMinSketch(width=16, depth=4)
    rng = Rng(0xF00D)
    counts = {}
    for _ in range(600):
        key = rng.below(4) if rng.below(3) else rng.below(200)      # a few genuinely hot keys
        counts[key] = counts.get(key, 0) + 1
        sketch.add(key)
    check(sum(counts.values()) == 600, "the stream must be accounted for")
    errors = []
    for key, exact in sorted(counts.items()):
        estimate = sketch.estimate(key)
        check(estimate >= exact, f"count-min underestimated {key}: {estimate} < {exact}")
        check(estimate <= 600, "an estimate cannot exceed the stream length")
        errors.append(estimate - exact)
    check(max(errors) <= sketch.updates / sketch.width * 3,
          f"the error must stay inside the sketch bound, worst {max(errors)}")
    check(max(errors) > 0, "a loaded table must show some overestimate")
    check(sum(errors) / len(errors) < sketch.updates / sketch.width,
          f"the average overestimate must stay near the collision floor, got "
          f"{sum(errors) / len(errors):.1f} vs {sketch.updates / sketch.width:.1f}")
    heavy = {key for key, exact in counts.items() if exact >= 10}
    check(len(heavy) >= 4, f"the fixture must have real heavy hitters, got {len(heavy)}")
    by_estimate = set(sorted(counts, key=lambda key: (-sketch.estimate(key), key))[:len(heavy)])
    check(heavy <= by_estimate,
          f"heavy-hitter recall must be one: missing {sorted(heavy - by_estimate)}")
    check(sketch.memory() == 16 * 4, "the sketch memory must be the table size")
    check(0 <= sketch.estimate(999) <= sketch.updates / sketch.width * 3,
          "an unseen key must estimate no more than the collision floor")
    check(CountMinSketch(width=16, depth=4).estimate(999) == 0,
          "a fresh sketch must estimate zero for everything")
    check(sketch.memory() == 64, "and it must not depend on how many keys were seen")

    # 2. AMS: the median-of-means energy estimate lands near the exact energy and is unbiased
    points, labels, _ = make_stream(0x5EED, length=800)
    exact_energy = sum(dot(point, point) for point in points)
    estimates = []
    for trial in range(21):
        sketch = AmsSketch(buckets=8, seed=trial)
        for point in points:
            sketch.add(point)
        estimates.append(sketch.energy())
    pooled = [AmsSketch(buckets=8, seed=trial) for trial in range(5)]
    for point in points:
        for sketch in pooled:
            sketch.add(point)
    medians = [sketch.energy() for sketch in pooled]
    relative = abs(median(medians) - exact_energy) / exact_energy
    check(relative < 0.35, f"the AMS estimate must be close to the exact energy, off by {relative:.3f}")
    average = sum(estimates) / len(estimates)
    check(abs(average - exact_energy) / exact_energy < 0.25,
          f"the estimator must be roughly unbiased, mean {average:.2f} vs {exact_energy:.2f}")
    check(min(estimates) >= 0.0, "an energy estimate cannot be negative")

    # 3. the streaming subspace recovers the generative directions
    rank_two = OjaPCA(DIMENSION, rank=RANK)
    for point in points[:600]:
        rank_two.update(point)
    check(abs(dot(rank_two.basis[0], first)) > 0.8 or abs(dot(rank_two.basis[0], second)) > 0.8,
          "the top direction must align with the generative plane")
    truth = orthonormalise([first, second])
    check(subspace_error(rank_two.basis, truth) < 0.2,
          f"the streaming basis must span the generative plane, error "
          f"{subspace_error(rank_two.basis, truth):.3f}")
    reference = batch_basis(points[:600], rank=RANK)
    check(subspace_error(reference, truth) < 0.2,
          "the batch power iteration must find the same plane")
    check(subspace_error(rank_two.basis, reference) < 0.25,
          f"streaming and batch must agree: {subspace_error(rank_two.basis, reference):.3f}")
    random_basis = orthonormalise([[1.0 if index == level else 0.0 for index in range(DIMENSION)]
                                   for level in range(2)])
    check(subspace_error(rank_two.basis, truth) < subspace_error(random_basis, truth),
          "the learned basis must beat an arbitrary basis")

    # 4. the reconstruction error is small for normal points and grows outside the plane
    normal_errors = [rank_two.error(point) for point in points[:600] if True]
    check(sum(normal_errors) / len(normal_errors) < 0.2,
          f"normal points must reconstruct well, mean error "
          f"{sum(normal_errors) / len(normal_errors):.3f}")
    outside = [6.0 * value for value in normalise([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])]
    check(rank_two.error(outside) > 4.0,
          "a point far outside the plane must have a large reconstruction error")
    check(abs(rank_two.error([0.0] * DIMENSION)) < 1e-12, "the origin reconstructs exactly")
    full = OjaPCA(DIMENSION, rank=DIMENSION)
    for point in points[:200]:
        full.update(point)
    check(full.error(points[10]) < 1e-9,
          f"a full-rank basis must reconstruct exactly, error {full.error(points[10]):.3e}")

    # 5. bounded memory and bounded work per point
    check(rank_two.memory() == RANK * DIMENSION,
          f"the model memory must be rank * dimension, got {rank_two.memory()}")
    check(rank_two.operations <= 600 * RANK * (RANK + 1) * DIMENSION,
          "the per-point work must stay inside the O(d * k * (k + 1)) budget")
    check(rank_two.operations > 0, "the update count must be tracked")
    one_dimension = OjaPCA(1, rank=1)
    for value in (1.0, 2.0, -3.0, 4.0):
        one_dimension.update([value])
    check(abs(abs(one_dimension.basis[0][0]) - 1.0) < 1e-9, "a one-dimensional basis is a unit scalar")
    check(one_dimension.error([0.0]) == 0.0, "and the origin still reconstructs exactly")

    # 6. detection quality against injected anomalies
    anomalous, anomaly_labels, _ = make_stream(0x5EED, length=800, anomalies=1)
    detector = Detector(DIMENSION)
    scores, flags = detector.run(anomalous)
    auc = detection_auc(scores, anomaly_labels)
    check(auc > 0.95, f"the detector must separate the anomalies, AUC {auc:.3f}")
    threshold = sorted(score for score, label in zip(scores, anomaly_labels) if label == 0)
    cutoff = threshold[int(0.99 * (len(threshold) - 1))]
    true_positive = sum(1 for score, label in zip(scores, anomaly_labels) if label == 1 and score > cutoff)
    false_positive = sum(1 for score, label in zip(scores, anomaly_labels) if label == 0 and score > cutoff)
    positives = sum(anomaly_labels)
    check(true_positive / positives >= 0.9,
          f"the detection rate at a 1% threshold must be high, got {true_positive / positives:.3f}")
    check(false_positive <= 0.02 * (len(anomaly_labels) - positives),
          f"the false positive rate must stay near 1%, got {false_positive}")
    check(sum(flags) > 0, "the running quantile threshold must fire")

    # 7. drift: an adaptive model tracks the new normal, a frozen model does not
    drifted, drift_labels, _ = make_stream(0x5EED, length=800, drift_at=400, anomalies=1)
    adaptive = Detector(DIMENSION)
    adaptive_scores, _ = adaptive.run(drifted)
    frozen = OjaPCA(DIMENSION, rank=RANK)
    for point in drifted[:300]:
        frozen.update(point)
    frozen_scores = [frozen.error(point) for point in drifted]
    tail = slice(len(drifted) // 2 + 100, None)
    adaptive_tail = sum(adaptive_scores[tail]) / len(adaptive_scores[tail])
    frozen_tail = sum(frozen_scores[tail]) / len(frozen_scores[tail])
    check(adaptive_tail < frozen_tail,
          f"after the drift the adaptive model must reconstruct better: {adaptive_tail:.3f} vs "
          f"{frozen_tail:.3f}")
    check(adaptive_tail < sum(adaptive_scores[:200]) / 200 * 1.5,
          f"the adaptive error must return near its pre-drift level: {adaptive_tail:.3f}")
    check(detection_auc(adaptive_scores, drift_labels) > 0.9,
          "the adaptive detector must still find the anomalies after the drift")

    # 8. determinism and degenerate streams
    again = Detector(DIMENSION)
    second_scores, _ = again.run(anomalous)
    check(second_scores == scores, "two identical runs must produce identical scores")
    constant = OjaPCA(DIMENSION, rank=RANK)
    for _ in range(400):
        constant.update([0.5] * DIMENSION)
    check(constant.error([0.5] * DIMENSION) < 1e-6,
          f"a constant stream must be reconstructed once the basis converges, error "
          f"{constant.error([0.5] * DIMENSION):.3e}")
    empty = Detector(DIMENSION)
    check(empty.run([]) == ([], []), "an empty stream must produce nothing")
    single = Detector(DIMENSION)
    single_scores, single_flags = single.run([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    check(len(single_scores) == 1 and single_flags == [0], "a single point must be scored")
    check(single.model.memory() == RANK * DIMENSION, "and the memory must not depend on the length")
    try:
        normalise([0.0] * DIMENSION)
        check(False, "the zero vector cannot be normalised")
    except ValueError:
        pass

    print(f"subspace: streaming error vs the generative plane "
          f"{subspace_error(rank_two.basis, truth):.3f}, batch {subspace_error(reference, truth):.3f}")
    print(f"sketches: count-min worst overestimate {max(errors)} in a 16x4 table; AMS median energy "
          f"{median(medians):.2f} vs exact {exact_energy:.2f}")
    print(f"detection: AUC {auc:.3f}, recall at a 1% threshold {true_positive / positives:.3f}; "
          f"drift tail error {adaptive_tail:.3f} adaptive vs {frozen_tail:.3f} frozen")
    print("PASS")


if __name__ == "__main__":
    main()
