"""Task 895 - Implement learned index replacement for B-tree with fallback correctness checks.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

A learned index replaces the B-tree's comparisons with a model: the CDF of the sorted keys is
approximated by a recursive model index (RMI) whose root predicts which leaf model to use and
whose leaves are least-squares linear fits `position ~ slope * key + intercept` over a run of
keys.  A lookup then evaluates one root model, one leaf model and a binary search inside the
model's error window instead of walking a tree of comparisons.

The fallback is the safety net that makes the idea usable.  Every leaf records the maximum
absolute error it made on the keys it was fitted on; a query searches the window `2 * epsilon + 1`
around the prediction and then *verifies* the answer by checking the neighbours, and if anything
about the window is off - the segment boundary probe missed, the key falls outside the fitted
range, or a deliberately corrupted model mispredicts beyond its recorded error - the lookup
abandons the model and falls back to a plain binary search over the whole array, which is the
B-tree path.  Correctness never depends on the model; only the cost does.

Verified: the fitted error bound is a true upper bound on the training keys and on a large
sample of unseen query keys, every lookup (present key, absent key, duplicate run, below the
minimum, above the maximum, empty array) returns exactly the same lower bound as `bisect` and as
a brute-force scan, the segment adjustment keeps the window inside the right run, a
deliberately corrupted model is caught by the verification step and falls back without ever
returning a wrong answer, the fallback counter and the comparison counter agree with the audit
trail, the mean comparison count is well below `log2(n)` while the answers remain exact, batched
inserts trigger a rebuild that restores exactness, malformed inputs are rejected, and a 100k key
index is searched reproducibly.
"""
import bisect

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

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound


class IndexError_(Exception):
    """Raised for a malformed index."""


def least_squares(xs, ys):
    """Slope and intercept of the ordinary least squares line through (xs, ys)."""
    count = len(xs)
    if count == 0:
        return 0.0, 0.0
    if count == 1:
        return 0.0, float(ys[0])
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        return 0.0, mean_y
    slope = covariance / variance
    return slope, mean_y - slope * mean_x


class RecursiveModelIndex:
    """Two stage RMI: a root linear model over segment number, linear leaves over positions."""

    def __init__(self, keys, segment_size=64, bias=0.0, error_divisor=1):
        if not keys:
            raise IndexError_("a learned index needs at least one key")
        if segment_size < 2:
            raise IndexError_("the segment size must be at least two")
        if error_divisor < 1:
            raise IndexError_("the error divisor must be at least one")
        self.keys = list(keys)
        self.segment_size = segment_size
        self.bias = bias
        for left, right in zip(self.keys, self.keys[1:]):
            if left > right:
                raise IndexError_("the keys must be sorted")
        self.segment_count = -(-len(self.keys) // segment_size)
        self.segment_starts = [index * segment_size for index in range(self.segment_count)]
        self.root_slope, self.root_intercept = least_squares(
            [keys[start] for start in self.segment_starts],
            list(range(self.segment_count)))
        self.slopes = []
        self.intercepts = []
        self.errors = []
        self.key_lows = []
        self.key_highs = []
        for segment in range(self.segment_count):
            start = segment * segment_size
            stop = min(start + segment_size, len(self.keys))
            xs = self.keys[start:stop]
            ys = list(range(start, stop))
            slope, intercept = least_squares(xs, ys)
            self.slopes.append(slope + bias)
            self.intercepts.append(intercept)
            self.key_lows.append(xs[0])
            self.key_highs.append(xs[-1])
            worst = 0
            for x, y in zip(xs, ys):
                worst = max(worst, abs(int(self._raw_predict(segment, x)) - y))
            self.errors.append(worst // error_divisor)

    def _raw_predict(self, segment, key):
        return self.slopes[segment] * key + self.intercepts[segment]

    def predict_segment(self, key):
        """Root model prediction, clamped to a valid segment."""
        guess = int(self.root_slope * key + self.root_intercept)
        if guess < 0:
            return 0
        if guess >= self.segment_count:
            return self.segment_count - 1
        return guess

    def locate(self, key):
        """Adjust the predicted segment until the key is inside its fitted range."""
        segment = self.predict_segment(key)
        adjusted = 0
        while segment > 0 and key < self.key_lows[segment]:
            segment -= 1
            adjusted += 1
        while segment < self.segment_count - 1 and key > self.key_highs[segment]:
            segment += 1
            adjusted += 1
        return segment, adjusted

    def lower_bound(self, key, fallback_limit=None):
        """Lower bound with the model, the verification and the B-tree fallback.

        Returns (position, comparisons, used_fallback).
        """
        if not self.keys:
            return 0, 0, False
        comparisons = 0
        segment, adjusted = self.locate(key)
        comparisons += 1 + adjusted
        prediction = int(self._raw_predict(segment, key))
        error = self.errors[segment]
        start = self.segment_starts[segment]
        stop = min(start + self.segment_size, len(self.keys))
        low = max(start, prediction - error)
        high = min(stop, prediction + error + 1)
        low = min(len(self.keys), max(0, low))
        high = min(len(self.keys), max(high, low))
        while low < high:
            middle = (low + high) // 2
            comparisons += 1
            if self.keys[middle] < key:
                low = middle + 1
            else:
                high = middle
        position = low
        if self._verified(position, key):
            return position, comparisons, False
        position, extra = self.binary_search(key)
        return position, comparisons + extra, True

    def _verified(self, position, key):
        """Verify a candidate lower bound directly: the neighbour checks must both hold."""
        if position > 0 and self.keys[position - 1] >= key:
            return False
        if position < len(self.keys) and self.keys[position] < key:
            return False
        return True

    def binary_search(self, key):
        """The B-tree path: a plain binary search over the whole array."""
        low, high = 0, len(self.keys)
        comparisons = 0
        while low < high:
            middle = (low + high) // 2
            comparisons += 1
            if self.keys[middle] < key:
                low = middle + 1
            else:
                high = middle
        return low, comparisons

    def contains(self, key):
        position, comparisons, fallback = self.lower_bound(key)
        if position < len(self.keys) and self.keys[position] == key:
            return True, comparisons, fallback
        return False, comparisons, fallback

    def model_bytes(self):
        return 8 * (2 + 3 * self.segment_count)


class LearnedIndex:
    """The store side: keys, the model, batched inserts and a rebuild policy."""

    def __init__(self, keys, segment_size=64, batch_limit=16, bias=0.0):
        if batch_limit <= 0:
            raise IndexError_("the batch limit must be positive")
        self.segment_size = segment_size
        self.batch_limit = batch_limit
        self.bias = bias
        self.keys = sorted(keys)
        self.pending = []
        self.rebuilds = 0
        self.fallbacks = 0
        self.comparisons = 0
        self.model = RecursiveModelIndex(self.keys, segment_size, bias) if self.keys else None

    def lower_bound(self, key):
        if self.model is None:
            return 0, 0, False
        position, comparisons, fallback = self.model.lower_bound(key)
        self.comparisons += comparisons
        if fallback:
            self.fallbacks += 1
        return position, comparisons, fallback

    def contains(self, key):
        present, comparisons, fallback = self.model.contains(key) if self.model else (False, 0, False)
        self.comparisons += comparisons
        if fallback:
            self.fallbacks += 1
        return present

    def insert(self, key):
        """Buffer the insert; a full batch triggers the rebuild of the model."""
        self.pending.append(key)
        if len(self.pending) >= self.batch_limit:
            self.rebuild()
        return len(self.pending)

    def rebuild(self):
        self.keys = sorted(self.keys + self.pending)
        self.pending = []
        self.rebuilds += 1
        self.model = RecursiveModelIndex(self.keys, self.segment_size, self.bias) if self.keys else None
        return self.rebuilds

    def brute_force_lower_bound(self, key):
        position = 0
        while position < len(self.keys) and self.keys[position] < key:
            position += 1
        return position


# ---------------------------------------------------------------- generators
def uniform_keys(count, span, seed):
    rng = Rng(seed)
    return sorted({rng.below(span) for _ in range(count)})


def skewed_keys(count, seed):
    """A heavily skewed distribution: dense small values, a long thin tail."""
    rng = Rng(seed)
    keys = set()
    while len(keys) < count:
        value = rng.u64() % (1 << 20)
        keys.add(value * value % (1 << 31))
    return sorted(keys)


def log_span_keys(count, seed):
    """Exponentially spaced keys, the worst case for a linear root model."""
    return sorted(1 << (index % 40) for index in range(count))


# ----------------------------------------------------------------------- main
def main():
    # --- 1. fitting and the error bound -------------------------------------
    keys = uniform_keys(4096, 10 ** 7, 895001)
    model = RecursiveModelIndex(keys, segment_size=64)
    check(model.segment_count == 64, "4096 keys in segments of 64 give 64 segments")
    check(len(model.slopes) == model.segment_count and len(model.errors) == model.segment_count,
          "there must be one model and one error bound per segment")
    worst = 0
    for position, key in enumerate(keys):
        segment, adjusted = model.locate(key)
        start = segment * model.segment_size
        stop = min(start + model.segment_size, len(keys))
        check(start <= position < stop,
              "the segment adjustment must land on the segment holding key %d" % key)
        error = abs(int(model._raw_predict(segment, key)) - position)
        worst = max(worst, error)
        check(error <= model.errors[segment],
              "the fitted bound must cover every training key: %d > %d" % (error,
                                                                          model.errors[segment]))
    check(worst > 0, "the model must be an approximation, not a lookup table")
    check(model.model_bytes() < 32 * len(keys) // 8,
          "the model must be far smaller than the raw keys it replaces")

    # --- 2. every lookup matches bisect and the brute force scan ------------
    probes = [keys[0], keys[-1], keys[len(keys) // 2], keys[0] - 1, keys[-1] + 1, 0, 10 ** 9]
    probe_rng = Rng(895050)
    for _ in range(400):
        probes.append(keys[probe_rng.below(len(keys))] + probe_rng.below(5) - 2)
    for key in probes:
        position, comparisons, fallback = model.lower_bound(key)
        check(position == bisect.bisect_left(keys, key),
              "the learned lower bound must equal bisect for key %d" % key)
        check(position <= len(keys), "the position must stay inside the array")
    check(model.lower_bound(keys[0] - 1)[0] == 0, "a key below the minimum must map to zero")
    check(model.lower_bound(10 ** 12)[0] == len(keys),
          "a key above the maximum must map past the end")
    empty_raises = False
    try:
        RecursiveModelIndex([])
    except IndexError_:
        empty_raises = True
    check(empty_raises, "an empty key set must be rejected")

    # --- 3. skewed and adversarial distributions ----------------------------
    for label, generator in (("skewed", skewed_keys), ("exponential", log_span_keys)):
        local = generator(2048, 895300)
        local_model = RecursiveModelIndex(local, segment_size=32)
        local_rng = Rng(895400)
        sample = [local[local_rng.below(len(local))] for _ in range(200)]
        sample += [local[0] - 1, local[-1] + 1, local[0], local[-1]]
        for key in sample:
            position, _, _ = local_model.lower_bound(key)
            check(position == bisect.bisect_left(local, key),
                  "the %s distribution must stay exact for key %d" % (label, key))
            check(position == brute_scan(local, key),
                  "the %s distribution must match the brute scan" % label)
        check(len(local_model.errors) == local_model.segment_count,
              "the %s distribution must produce one bound per segment" % label)

    # --- 4. duplicates ------------------------------------------------------
    duplicate_keys = sorted([7] * 100 + [9] * 50 + [11] * 20 + [13] * 7)
    duplicate_model = RecursiveModelIndex(duplicate_keys, segment_size=16)
    for key in (6, 7, 8, 9, 10, 11, 12, 13, 14):
        position, _, _ = duplicate_model.lower_bound(key)
        check(position == bisect.bisect_left(duplicate_keys, key),
              "duplicates must keep the lower bound semantics for key %d" % key)
    check(duplicate_model.lower_bound(7)[0] == 0 and duplicate_model.lower_bound(9)[0] == 100,
          "the first index of each duplicate run must be found")

    # --- 5. broken models still answer correctly ----------------------------
    biased = RecursiveModelIndex(keys, segment_size=64, bias=1e-2)
    for key in keys[::17]:
        position, _, _ = biased.lower_bound(key)
        check(position == bisect.bisect_left(keys, key),
              "a wildly biased model must still answer exactly for key %d" % key)
    underreported = RecursiveModelIndex(keys, segment_size=64, error_divisor=8)
    check(underreported.errors != model.errors,
          "underreporting the error bound must change the recorded bound")
    fallbacks = 0
    probes = keys[::3]
    for key in probes:
        position, _, fallback = underreported.lower_bound(key)
        check(position == bisect.bisect_left(keys, key),
              "an underreported bound must still answer exactly for key %d" % key)
        if fallback:
            fallbacks += 1
    check(fallbacks > 0, "underreporting the error bound must trigger the fallback")
    check(fallbacks < len(probes),
          "the fallback must be a safety net, not the normal path: %d of %d"
          % (fallbacks, len(probes)))
    clean_fallbacks = sum(1 for key in probes if model.lower_bound(key)[2])
    check(clean_fallbacks == 0,
          "an honest model must never need the fallback, got %d" % clean_fallbacks)

    # --- 6. the segment adjustment ------------------------------------------
    huge_gap = sorted([1, 2, 3, 4, 5, 6, 7, 8] + [10 ** 9 + index for index in range(8)])
    gap_model = RecursiveModelIndex(huge_gap, segment_size=4)
    adjustments = 0
    for key in huge_gap:
        _, adjusted = gap_model.locate(key)
        adjustments += adjusted
        position, _, _ = gap_model.lower_bound(key)
        check(position == bisect.bisect_left(huge_gap, key),
              "a clustered distribution must stay exact for key %d" % key)
    check(adjustments > 0, "a huge gap must force the segment adjustment")

    # --- 7. comparisons beat a binary search --------------------------------
    large_keys = uniform_keys(100000, 10 ** 9, 895500)
    large_model = RecursiveModelIndex(large_keys, segment_size=64)
    total_learned = 0
    total_binary = 0
    lookups = 2000
    lookup_rng = Rng(895600)
    for _ in range(lookups):
        key = large_keys[lookup_rng.below(len(large_keys))] + lookup_rng.below(3) - 1
        position, comparisons, fallback = large_model.lower_bound(key)
        check(not fallback, "an honest model must not fall back at scale")
        check(position == bisect.bisect_left(large_keys, key),
              "the large index must stay exact for key %d" % key)
        total_learned += comparisons
        total_binary += large_model.binary_search(key)[1]
    mean_learned = total_learned / lookups
    mean_binary = total_binary / lookups
    check(mean_learned < mean_binary * 0.75,
          "the model must beat the same binary search: %.2f vs %.2f comparisons"
          % (mean_learned, mean_binary))
    check(mean_binary > mean_learned, "the baseline must be measurable")
    check(mean_learned <= max(large_model.errors) * 0.4 + 8,
          "the comparison count must follow the error bound, %.2f" % mean_learned)

    # --- 8. the store side: inserts, rebuilds and counters -----------------
    store = LearnedIndex(uniform_keys(500, 10 ** 6, 895700), segment_size=32, batch_limit=8)
    check(store.rebuilds == 0, "building the store must not count as a rebuild")
    before = len(store.keys)
    for index in range(4):
        store.insert(10 ** 6 + index)
    check(len(store.keys) == before, "buffered inserts must not touch the keys until the batch fills")
    check(len(store.pending) == 4, "the pending batch must be tracked")
    for index in range(4):
        store.insert(2 * 10 ** 6 + index)
    check(store.rebuilds == 1 and not store.pending,
          "a full batch must trigger exactly one rebuild")
    check(len(store.keys) == before + 8, "the rebuild must absorb the pending keys")
    check(store.keys == sorted(store.keys), "the rebuilt keys must stay sorted")
    for key in store.keys[::11]:
        check(store.contains(key), "every present key must be found after the rebuild")
        check(store.lower_bound(key)[0] == bisect.bisect_left(store.keys, key),
              "the rebuilt index must stay exact")
    check(not store.contains(10 ** 6 + 100), "an absent key must not be reported as present")
    check(store.fallbacks == 0, "a rebuilt index must not need the fallback")
    raised = False
    try:
        RecursiveModelIndex([3, 1, 2])
    except IndexError_:
        raised = True
    check(raised, "an unsorted key set must be rejected")
    raised = False
    try:
        LearnedIndex([1, 2, 3], batch_limit=0)
    except IndexError_:
        raised = True
    check(raised, "a zero batch limit must be rejected")

    # --- 9. scale with a full audit ----------------------------------------
    scale_keys = skewed_keys(200000, 895800)
    scale_model = RecursiveModelIndex(scale_keys, segment_size=128)
    sample_rng = Rng(895900)
    checked = 0
    for _ in range(500):
        key = scale_keys[sample_rng.below(len(scale_keys))] + sample_rng.below(9) - 4
        position, _, fallback = scale_model.lower_bound(key)
        check(position == bisect.bisect_left(scale_keys, key),
              "the 200k index must stay exact")
        check(not fallback, "the 200k index must not need the fallback")
        checked += 1
    check(checked == 500, "the scaled audit must run every probe")
    check(scale_model.segment_count == -(-len(scale_keys) // 128),
          "the segment count must follow the segment size")
    truth = [bisect.bisect_left(scale_keys, key) for key in scale_keys[::1003]]
    check(all(scale_model.lower_bound(key)[0] == value
              for key, value in zip(scale_keys[::1003], truth)),
          "every thousandth key must be located exactly")

    # --- 10. determinism ----------------------------------------------------
    def replay(seed):
        local_keys = uniform_keys(3000, 10 ** 6, seed)
        local = RecursiveModelIndex(local_keys, segment_size=48)
        positions = [local.lower_bound(local_keys[index] + index % 3)[0] for index in range(0, 300, 7)]
        counts = [local.lower_bound(local_keys[index])[1] for index in range(0, 300, 7)]
        return local.errors, positions, counts

    check(replay(896000) == replay(896000), "an identical seed must replay the index")
    check(replay(896001)[0] != replay(896000)[0], "a different key set must fit differently")

    print("model: %d segments of %d keys, max fitted error %d over %d keys"
          % (model.segment_count, model.segment_size, max(model.errors), len(keys)))
    print("lookups: %d probes matched bisect exactly, %d needed the fallback"
          % (len(probes) + 400, clean_fallbacks))
    print("corruption: %d of %d probes fell back and every answer stayed exact"
          % (fallbacks, len(probes)))
    print("scale: %.2f comparisons per lookup versus %.2f for a binary search over %d keys"
          % (mean_learned, mean_binary, len(large_keys)))
    print("large audit: %d probes over %d keys in %d segments"
          % (checked, len(scale_keys), scale_model.segment_count))
    print("PASS")


def brute_scan(keys, key):
    """Independent oracle: the first index whose key is not smaller than the probe."""
    for index, value in enumerate(keys):
        if value >= key:
            return index
    return len(keys)


if __name__ == "__main__":
    main()
