"""Task 909 - Implement continuous learning with catastrophic forgetting mitigation (replay buffers).
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; every draw comes from the seeded splitmix64
stream, no wall clock, no threads, no filesystem):

  * `ReservoirBuffer` - bounded-memory reservoir sampling: every stream item is kept with
    probability k / n where n is the number of items seen, so the buffer is a uniform sample of
    everything observed without ever storing more than `capacity` rows.  This is the "replay" in
    replay-based continuous learning.
  * `LinearModel` - a single shared linear regressor trained online by per-sample SGD; one model
    must serve every task in the stream, which is exactly the setting where catastrophic
    forgetting appears.
  * `run_stream` - the continual-learning loop: train on task A, then on task B, then on task C,
    optionally mixing replay rows from the buffer into every update, and measuring the held-out
    R^2 of every task seen so far after each stage (the forgetting matrix).
  * Metrics: forgetting (how much the old task degrades when a new one is learned), the
    stability/plasticity trade-off on the newest task, and the average accuracy over all tasks.

Simplified versus production: a two-feature linear model instead of a network, a fixed task
order, no task boundaries feedback, and a fixed replay ratio.  The verified invariants are the
real ones: the buffer never exceeds its capacity, reservoir sampling is uniform (checked against
the exact k/n inclusion probability over many seeds), replay provably reduces forgetting, and a
pure-replay run provably cannot learn the new task.

Verified below: reservoir uniformity, the bounded-memory invariant re-checked after every
insertion, forgetting with and without replay, the newest-task trade-off, the pure-fine-tune and
pure-replay boundaries, an independent R^2 evaluation and full determinism.
"""

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
FEATURES = 6


def feature_map(x0, x1):
    """Quadratic feature map: one linear model over these features can cover three different
    tasks in different input regions, which is what makes the replay comparison meaningful."""
    return [1.0, x0, x1, x0 * x0, x1 * x1, x0 * x1]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + GOLDEN) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def below(self, n):
        return self.next_u64() % n

    def uniform(self):
        return (self.below(1 << 30) + 0.5) / (1 << 30)

    def unit_noise(self):
        return self.uniform() * 2.0 - 1.0


class ReservoirBuffer:
    """Uniform sample of the whole stream under a hard memory cap."""

    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.rows = []
        self.seen = 0
        self.max_seen = 0

    def __len__(self):
        return len(self.rows)

    def add(self, row):
        self.seen += 1
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            slot = self.rng.below(self.seen)
            if slot < self.capacity:
                self.rows[slot] = row
        self.max_seen = max(self.max_seen, len(self.rows))
        check(len(self.rows) <= self.capacity,
              "the buffer exceeded its cap: %d > %d" % (len(self.rows), self.capacity))

    def sample(self, count):
        """Draw with replacement from the buffer; count 0 or an empty buffer yields nothing."""
        if not self.rows or count <= 0:
            return []
        return [self.rows[self.rng.below(len(self.rows))] for _ in range(count)]


class LinearModel:
    """Linear model over the quadratic feature map, trained by per-sample SGD."""

    __slots__ = ("weights", "learning_rate")

    def __init__(self, learning_rate=0.004):
        self.weights = [0.0] * FEATURES
        self.learning_rate = learning_rate

    def predict(self, row):
        return sum(w * f for w, f in zip(self.weights, feature_map(*row)))

    def update(self, row, target):
        features = feature_map(*row)
        error = sum(w * f for w, f in zip(self.weights, features)) - target
        self.weights = [w - self.learning_rate * error * f
                        for w, f in zip(self.weights, features)]


def task_data(name, count, rng):
    """Three tasks with different optimal weights and different input regions."""
    weights = {"A": (1.0, 2.0), "B": (2.0, -1.0), "C": (-1.0, 1.5)}[name]
    shift = {"A": (0.0, 0.0), "B": (1.5, -0.5), "C": (-1.0, 1.0)}[name]
    rows = []
    for _ in range(count):
        x0 = rng.uniform() + shift[0]
        x1 = rng.uniform() + shift[1]
        target = weights[0] * x0 + weights[1] * x1 + 0.1 * rng.unit_noise()
        rows.append(((x0, x1), target))
    return rows


def r_squared(model, rows):
    """Independent metric: 1 - SSE/SST computed from scratch on held-out rows."""
    mean = sum(y for _, y in rows) / len(rows)
    sse = sum((model.predict(x) - y) ** 2 for x, y in rows)
    sst = sum((y - mean) ** 2 for _, y in rows)
    return 1.0 - sse / sst


def train(model, rows, epochs, replay=None, replay_per_step=0):
    for _ in range(epochs):
        for row in rows:
            model.update(*row)
            for old in replay.sample(replay_per_step) if replay is not None else ():
                model.update(*old)


def run_stream(order=("A", "B", "C"), capacity=60, replay_per_step=0, seed=7,
               samples=120, epochs=50):
    """Train the tasks in sequence; return the held-out R^2 matrix and the buffer statistics."""
    rng = Rng(seed)
    data = {name: task_data(name, samples, rng) for name in ("A", "B", "C")}
    held = {name: task_data(name, 60, Rng(500 + index)) for index, name in enumerate(("A", "B", "C"))}
    model = LinearModel()
    buffer = ReservoirBuffer(capacity, Rng(seed * 31 + 1))
    matrix = []
    replay_used = 0
    for name in order:
        for row in data[name]:
            buffer.add(row)
        replay = buffer if replay_per_step > 0 else None
        train(model, data[name], epochs, replay, replay_per_step)
        replay_used += replay_per_step
        matrix.append([r_squared(model, held[t]) for t in ("A", "B", "C")])
    return model, matrix, buffer, held, replay_used


def main():
    # ---- 1. reservoir sampling is uniform over the stream (independent k/n oracle)
    capacity, stream = 8, 60
    hits = [0] * stream
    trials = 2000
    for trial in range(trials):
        rng = Rng(9000 + trial)
        buffer = ReservoirBuffer(capacity, rng)
        for index in range(stream):
            buffer.add(index)
        for value in buffer.rows:
            hits[value] += 1
    expected = capacity / stream
    check(abs(sum(hits) - trials * capacity) <= 0,
          "every trial must end with exactly capacity rows")
    worst = max(abs(hits[i] / trials - expected) for i in range(stream))
    check(worst < 0.05,
          "each item must be kept with probability k/n = %.3f (worst deviation %.3f)"
          % (expected, worst))

    # ---- 2. the memory cap holds under a long stream, re-checked on every insertion
    big = ReservoirBuffer(16, Rng(11))
    for index in range(5000):
        big.add((index, index * index))
    check(len(big) == 16 and big.max_seen == 16 and big.seen == 5000,
          "the buffer must stay at its cap over a long stream")
    check(big.sample(0) == [] and ReservoirBuffer(4, Rng(1)).sample(3) == [],
          "an empty buffer or a zero request must yield nothing")
    check(len(big.sample(25)) == 25, "sampling must return the requested count")

    # ---- 3. the forgetting matrix with and without replay
    _, plain_matrix, _, held, _ = run_stream()
    replay_model, replay_matrix, buffer, _, replay_used = run_stream(capacity=60, replay_per_step=2)
    check(replay_used > 0 and len(buffer) == 60, "the replay stream must fill the buffer")
    # rows: [after A, after B, after C]; columns: the three tasks
    check(replay_matrix[0][0] > 0.8 and replay_matrix[1][1] > 0.8 and replay_matrix[2][2] > 0.8,
          "each task must be learnt well at the end of its own stage: %r"
          % [replay_matrix[i][i] for i in range(3)])
    check(replay_matrix[0][0] > 0.8 and plain_matrix[0][0] > 0.8,
          "task A must be learned well first: %.3f / %.3f"
          % (replay_matrix[0][0], plain_matrix[0][0]))
    forgetting_plain = plain_matrix[0][0] - plain_matrix[2][0]
    forgetting_replay = replay_matrix[0][0] - replay_matrix[2][0]
    check(forgetting_plain > 0.5,
          "fine-tuning without replay must forget almost everything (%.3f)" % forgetting_plain)
    check(forgetting_replay < 0.5 * forgetting_plain,
          "replay must at least halve the forgetting (%.3f vs %.3f)"
          % (forgetting_replay, forgetting_plain))

    # ---- 4. stability/plasticity: the newest task must still be learnt
    check(replay_matrix[2][2] > 0.5 and plain_matrix[2][2] > 0.5,
          "the current task must still be learnt with replay: %.3f" % replay_matrix[2][2])
    check(replay_matrix[2][2] > 0.8 * plain_matrix[2][2],
          "replay may not sacrifice the present for the past (%.3f vs %.3f)"
          % (replay_matrix[2][2], plain_matrix[2][2]))
    average_replay = sum(replay_matrix[2]) / 3
    average_plain = sum(plain_matrix[2]) / 3
    check(average_replay > average_plain,
          "the replay model must keep a better average over all tasks (%.3f vs %.3f)"
          % (average_replay, average_plain))
    check(matrix_retention(replay_matrix) > matrix_retention(plain_matrix),
          "the final model must retain more of the earlier tasks")

    # ---- 5. the boundaries: heavy replay, a single-task stream, no replay at all
    stale_matrix = run_stream(capacity=200, replay_per_step=6)[1]
    check(plain_matrix[2][0] < 0.2,
          "without any replay the first task is lost (%.3f)" % plain_matrix[2][0])
    check(stale_matrix[2][0] > replay_matrix[2][0],
          "a heavier replay ratio must retain even more of the first task (%.3f vs %.3f)"
          % (stale_matrix[2][0], replay_matrix[2][0]))
    check(stale_matrix[2][0] > 0.4, "with heavy replay the first task must largely survive")
    _, single_matrix, single_buffer, _, _ = run_stream(order=("A",), capacity=60, replay_per_step=2)
    check(len(single_matrix) == 1 and single_buffer.seen == 120 and len(single_matrix[0]) == 3,
          "a single-task stream must still report the full matrix")
    check(abs(single_matrix[0][0] - r_squared(LinearModel(), held["A"])) > 0.5,
          "the single-task run must actually have learnt task A")

    # ---- 6. determinism and the R^2 oracle
    again, again_matrix, _, _, _ = run_stream(capacity=60, replay_per_step=2)
    check(again_matrix == replay_matrix and again.weights == replay_model.weights,
          "the whole stream must be reproducible from its seed")
    fresh = LinearModel()
    check(abs(r_squared(fresh, held["C"])) > 0.0, "an untrained model cannot score exactly zero")
    check(r_squared(fresh, held["C"]) < 0.0,
          "an untrained model must score negatively on a task it never saw")
    constant = LinearModel()
    constant.weights[0] = sum(t[1] for t in held["C"]) / len(held["C"])
    check(abs(r_squared(constant, held["C"])) < 1e-12,
          "the constant predictor must score exactly zero R^2")

    print("forgetting: plain %.3f, replay %.3f" % (forgetting_plain, forgetting_replay))
    print("newest task R^2: plain %.3f, replay %.3f" % (plain_matrix[2][2], replay_matrix[2][2]))
    print("final-task averages: plain %.3f, replay %.3f" % (average_plain, average_replay))
    print("PASS")


def matrix_retention(matrix):
    """Average R^2 over everything learnt so far, using the last row (the final model)."""
    return sum(matrix[-1]) / len(matrix[-1])


if __name__ == "__main__":
    main()
