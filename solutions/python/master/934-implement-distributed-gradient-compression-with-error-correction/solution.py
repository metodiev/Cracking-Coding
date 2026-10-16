"""Task 934 - Implement distributed gradient compression with error correction under packet loss.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

Data-parallel SGD where every worker compresses its gradient to the top-k coordinates, packets can
be lost, and error feedback repairs the damage.

* Model: a separable quadratic f(w) = 0.5 sum_j c_j (w_j - t_j)^2 over 8 coordinates, 4 workers.
  Worker i sees a shard whose gradient differs from the true one by a fixed bias, and the biases sum
  to zero across shards, so the exact average is still the true gradient.  The learning rate obeys
  the 1 / max(c_j) stability bound, which makes plain descent provably non-increasing.
* Compression: each worker keeps a residual r; `topk(r)` sends the k largest-magnitude coordinates.
  The error-feedback (EF) variant subtracts what it sent and keeps the rest, so nothing is forgotten;
  the naive variant instead scales the top-k by d/k and drops the tail entirely, which is exactly
  what leaves a variance floor.
* Packet loss: each worker's packet is dropped with probability p; the coordinator averages only the
  packets it received, and the worker keeps its residual (so a lost packet is deferred, not lost).

Verified: only k entries per worker travel per round; the EF bookkeeping invariant
(transmitted + resident residual == the gradient that was produced) holds after every round, with and
without loss; the exact trajectory matches the closed-form geometric recursion per coordinate; exact
and EF runs converge to the optimum while the naive sparsifier plateaus well above it; EF still
converges with a 30% loss rate; k = 0 transmits nothing and makes no progress, k = d with no loss is
bit-identical to the exact run, and a 100% loss rate freezes the model; and every run is
deterministic.  Simplified: one compression scheme, a quadratic objective and a single coordinator.
"""
import math

DIM, WORKERS, K_TOP, ROUNDS = 8, 4, 2, 40
CURVATURE = (1.0, 1.4, 1.8, 2.2, 1.2, 1.6, 2.0, 1.1)
TARGET = (1.0, -0.5, 0.75, 0.25, -1.25, 0.5, -0.75, 1.5)
START = (0.0,) * DIM
LEARNING_RATE = 0.5 / max(CURVATURE)
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


def shard_biases():
    """Per-worker gradient offsets that sum to zero per coordinate (shard heterogeneity)."""
    biases = [[0.05 * ((worker + 2 * index) % 5 - 2) for index in range(DIM)]
              for worker in range(WORKERS)]
    for index in range(DIM):
        mean = sum(biases[worker][index] for worker in range(WORKERS)) / WORKERS
        for worker in range(WORKERS):
            biases[worker][index] -= mean
    return biases


BIASES = shard_biases()


def objective(weights):
    return 0.5 * sum(CURVATURE[index] * (weights[index] - TARGET[index]) ** 2
                     for index in range(DIM))


def true_gradient(weights):
    return [CURVATURE[index] * (weights[index] - TARGET[index]) for index in range(DIM)]


def worker_gradients(weights):
    base = true_gradient(weights)
    return [[base[index] + BIASES[worker][index] for index in range(DIM)]
            for worker in range(WORKERS)]


def top_k(values, k):
    """Indices and values of the k largest-magnitude coordinates, ties broken by index."""
    order = sorted(range(len(values)), key=lambda index: (-abs(values[index]), index))
    return [(index, values[index]) for index in order[:k]]


def run(variant="ef", loss_rate=0.0, k=K_TOP, rounds=ROUNDS, state=0x5EED):
    """Train with one compression variant; returns the loss curve and the bookkeeping report."""
    weights = list(START)
    residuals = [[0.0] * DIM for _ in range(WORKERS)]
    losses, transmitted_entries, payload_bytes = [], 0, 0
    produced_total = [0.0] * DIM
    transmitted_total = [0.0] * DIM
    variant_invariant_ok = True
    for _ in range(rounds):
        losses.append(objective(weights))
        payloads = []
        for worker in range(WORKERS):
            gradient = worker_gradients(weights)[worker]
            produced_total = [produced_total[index] + gradient[index] for index in range(DIM)]
            state, draw = uniform(state)
            dropped = loss_rate > 0.0 and draw < loss_rate
            if variant == "exact":
                sent = [(index, gradient[index]) for index in range(DIM)]
            elif variant == "ef":
                for index in range(DIM):                       # r <- r + g, then r <- r - sent
                    residuals[worker][index] += gradient[index]
                sent = top_k(residuals[worker], k)
                if not dropped:
                    for index, value in sent:
                        residuals[worker][index] -= value
            else:                                              # naive: scale the top-k, drop the tail
                sent = [(index, value * DIM / max(k, 1)) for index, value in top_k(gradient, k)]
            if dropped:
                continue
            transmitted_entries += len(sent)
            payload_bytes += 4 * len(sent)
            for index, value in sent:
                transmitted_total[index] += value
            payloads.append(sent)
        if variant == "ef":
            resident = [sum(residuals[worker][index] for worker in range(WORKERS))
                        for index in range(DIM)]
            variant_invariant_ok = variant_invariant_ok and all(
                abs(resident[index] + transmitted_total[index] - produced_total[index]) < 1e-9
                for index in range(DIM))
        if payloads:
            aggregate = [0.0] * DIM
            for payload in payloads:
                for index, value in payload:
                    aggregate[index] += value
            weights = [weights[index] - LEARNING_RATE * aggregate[index] / len(payloads)
                       for index in range(DIM)]
    losses.append(objective(weights))
    return {"weights": weights, "losses": losses, "distance": math.dist(weights, TARGET),
            "transmitted": transmitted_entries, "bytes": payload_bytes,
            "invariant": variant_invariant_ok}


def closed_form(rounds=ROUNDS):
    """Oracle: the exact per-coordinate recursion is w_n - t = (1 - eta c)^n (w_0 - t)."""
    return [TARGET[index] + (START[index] - TARGET[index])
            * (1.0 - LEARNING_RATE * CURVATURE[index]) ** rounds for index in range(DIM)]


def main():
    check(len(BIASES) == WORKERS and all(
        abs(sum(BIASES[worker][index] for worker in range(WORKERS))) < 1e-12
        for index in range(DIM)),
        "shard biases must cancel exactly so the average gradient is the true gradient")
    check(all(abs(sum(worker_gradients(START)[worker][index] for worker in range(WORKERS)) / WORKERS
                  - true_gradient(START)[index]) < 1e-12 for index in range(DIM)),
          "the averaged worker gradient must equal the true gradient")

    # 1. The exact run reproduces the closed-form geometric recursion coordinate by coordinate.
    exact = run("ef", k=DIM)
    oracle = closed_form()
    check(max(abs(a - b) for a, b in zip(exact["weights"], oracle)) < 1e-9,
          "the exact trajectory must match the closed-form recursion")
    check(exact["losses"][-1] < 1e-8 and all(
        exact["losses"][index + 1] <= exact["losses"][index] + 1e-12
        for index in range(len(exact["losses"]) - 1)),
        "exact descent must be monotone and reach the optimum")

    # 2. Compression: exactly k entries per received worker travel every round.
    ef = run("ef", loss_rate=0.0)
    received = ROUNDS * WORKERS
    check(ef["transmitted"] == received * K_TOP and ef["bytes"] == 4 * received * K_TOP,
          "each worker must transmit exactly k coordinates per round")
    check(received * DIM == (DIM // K_TOP) * ef["transmitted"],
          "the uncompressed protocol must transmit d / k times more entries")

    # 3. Bookkeeping invariant: transmitted plus resident residual equals what was produced,
    #    re-checked after every round, with and without loss.
    lossy = run("ef", loss_rate=0.3)
    check(ef["invariant"] and lossy["invariant"],
          "error feedback must never lose or duplicate a gradient coordinate")

    # 4. Convergence: error feedback reaches a tight neighbourhood of the optimum, with and without
    #    packet loss, whereas the exact protocol is the theoretical best case.
    check(ef["distance"] < 0.05 and ef["losses"][-1] < exact["losses"][-1] + 0.01,
          "error feedback must converge near the optimum without packet loss")
    check(lossy["distance"] < 0.15 and lossy["losses"][-1] < 1e-2,
          "error feedback must still converge with 30% of the packets lost")

    # 5. The naive sparsifier plateaus: dropping the tail leaves a variance floor far above zero.
    naive = run("naive", loss_rate=0.0)
    check(naive["losses"][-1] > 100 * max(ef["losses"][-1], 1e-12),
          "dropping the tail without error feedback must leave a large residual loss")
    check(naive["losses"][-1] > ef["losses"][-1],
          "error feedback must beat naive sparsification on the same budget")
    check(lossy["losses"][-1] < naive["losses"][-1],
          "even a lossy error-feedback run must beat the naive sparsifier")

    # 6. Boundary cases: k = 0, k = d and a 100% loss rate.
    empty = run("ef", k=0)
    check(empty["transmitted"] == 0 and empty["distance"] == math.dist(START, TARGET)
          and empty["invariant"],
          "with k = 0 nothing is transmitted, nothing moves and the residual keeps everything")
    check(run("ef", k=DIM)["weights"] == exact["weights"],
          "k = d without loss must reduce to the exact run")
    frozen = run("ef", loss_rate=1.0)
    check(frozen["transmitted"] == 0 and frozen["distance"] == math.dist(START, TARGET),
          "losing every packet must freeze the model instead of corrupting it")
    single = run("ef", rounds=1)
    check(len(single["losses"]) == 2 and single["losses"][1] <= single["losses"][0],
          "a single round must still decrease the objective")

    # 7. Determinism: identical runs, and the loss curves are reproducible bit for bit.
    check(run("ef", loss_rate=0.3)["losses"] == lossy["losses"]
          and run("naive")["weights"] == naive["weights"],
          "the simulation must be reproducible")

    print(f"exact final loss {exact['losses'][-1]:.3e}, ef {ef['losses'][-1]:.3e}, "
          f"ef+30% loss {lossy['losses'][-1]:.3e}, naive {naive['losses'][-1]:.3e}")
    print(f"transmitted entries {ef['transmitted']} ({ef['bytes']} B) for "
          f"{ROUNDS * WORKERS} worker rounds, compression ratio d/k = {DIM // K_TOP}")
    print("PASS")


if __name__ == "__main__":
    main()
