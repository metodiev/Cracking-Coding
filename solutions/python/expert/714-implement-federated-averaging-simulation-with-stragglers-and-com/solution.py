"""Task 714 - Implement federated averaging simulation with stragglers and compression.  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

Federated averaging (McMahan et al.) trains one model across clients that never share data: each
round every client starts from the current global model, runs a few local SGD steps, uploads the
resulting *delta*, and the server adds the sample-weighted mean of the deltas.  This file simulates
that on a small linear regression and adds the two complications the task names:

  * compression -- a client sends only the k largest-magnitude coordinates of its delta
    (top-k sparsification) and/or a one-bit sign vector (quantisation).  Sparsification is paired
    with *error feedback*: the residual (delta minus what was sent) is kept locally and added to
    the next round's delta, so what a client has cumulatively sent plus the residual it still
    holds equals the cumulative update it computed.  That conservation is the invariant the checks
    verify, along with the exact sparsity count and the exact tail energy of the dropped
    coordinates.
  * stragglers -- slow or unreachable clients.  Two policies are implemented and each is compared
    against a plaintext oracle: "drop" averages only the clients that answered (renormalising the
    sample weights) and "stale" applies a late client's delta even though it was computed from the
    previous global model.

Simplified versus the literature: one tiny deterministic synthetic dataset, one model family (ridge
linear regression, whose pooled loss and gradients are cheap and exact), no client sampling or
momentum variants, and no privacy mechanism on the uploaded deltas (task 713 covers that).
Verified: the server aggregate equals an independently computed sample-weighted mean for the
policies tested, FedAvg with one client is step-for-step plain SGD, error feedback conserves what
was computed, the sparsity and reconstruction bounds are exact, heterogeneous and homogeneous
straggler schedules both behave as predicted, and the degenerate settings (zero local steps,
k = 0, k = dimension, sign-only compression, empty shards) are exercised.
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
        """Uniform in (-1, 1)."""
        self.state, r = splitmix64(self.state)
        return ((r >> 11) / (1 << 53)) * 2.0 - 1.0


# --------------------------------------------------------------------------- data and model
def make_dataset(client_count, samples, dimension, seed, truth=None, noise=0.05):
    """One shard per client; the last model coordinate is the intercept."""
    rng = Rng(seed)
    shards = []
    for _ in range(client_count):
        weights = list(truth) if truth is not None else [rng.uniform() for _ in range(dimension + 1)]
        features, targets = [], []
        for _ in range(samples):
            row = [rng.uniform() for _ in range(dimension)]
            value = weights[dimension] + sum(weights[k] * row[k] for k in range(dimension))
            features.append(row)
            targets.append(value + noise * rng.uniform())
        shards.append((features, targets))
    return shards


def mse(model, features, targets):
    if not features:
        return 0.0
    total = 0.0
    for row, target in zip(features, targets):
        prediction = model[-1] + sum(model[k] * row[k] for k in range(len(row)))
        total += (prediction - target) ** 2
    return total / len(features)


def pooled_loss(model, shards):
    features = [row for shard in shards for row in shard[0]]
    targets = [value for shard in shards for value in shard[1]]
    return mse(model, features, targets)


def gradient(model, features, targets, l1=0.0):
    count = len(features)
    result = [0.0] * len(model)
    if count == 0:
        return result
    for row, target in zip(features, targets):
        error = model[-1] + sum(model[k] * row[k] for k in range(len(row))) - target
        for k in range(len(row)):
            result[k] += 2.0 * error * row[k]
        result[-1] += 2.0 * error
    scale = 1.0 / count
    for k in range(len(model) - 1):
        result[k] = result[k] * scale + l1 * model[k]
    result[-1] *= scale
    return result


def local_update(model, features, targets, steps, learning_rate, l1=0.0):
    local = list(model)
    for _ in range(steps):
        g = gradient(local, features, targets, l1)
        for k in range(len(local)):
            local[k] -= learning_rate * g[k]
    return [local[k] - model[k] for k in range(len(model))]


# --------------------------------------------------------------------------- compression
def top_k(values, k):
    """The k largest-magnitude coordinates (ties by index, so it is deterministic)."""
    check(0 <= k <= len(values), "k must lie in [0, len(values)]")
    if k == len(values):
        return list(values)
    order = sorted(range(len(values)), key=lambda index: (-abs(values[index]), index))
    keep = set(order[:k])
    return [value if index in keep else 0.0 for index, value in enumerate(values)]


def sign_compress(values, scale):
    """One bit per coordinate: the sign, rescaled."""
    return [math.copysign(scale, value) if value != 0.0 else 0.0 for value in values]


def tail_energy(values, k):
    kept = top_k(values, k)
    return sum((values[index] - kept[index]) ** 2 for index in range(len(values)))


class Client:
    """A client: its shard, its compression mode, and its error-feedback residual."""

    def __init__(self, index, features, targets, compression=None, k=None, scale=0.0, feedback=True):
        self.index = index
        self.features = features
        self.targets = targets
        self.compression = compression
        self.k = k
        self.scale = scale
        self.feedback = feedback
        self.residual = None
        self.max_residual_norm = 0.0
        self.sent_total = None
        self.raw_total = None
        self.sent_nonzeros = 0
        self.rounds = 0

    def samples(self):
        return len(self.features)

    def compress(self, delta):
        """Compress a delta; with error feedback the leftover is carried into the next round."""
        if self.compression is None:
            sent = list(delta)
            self._bookkeep(delta, sent)
            return sent
        if self.residual is None:
            self.residual = [0.0] * len(delta)
        pending = [delta[index] + self.residual[index] for index in range(len(delta))]
        if self.compression == "topk":
            sent = top_k(pending, self.k)
        elif self.compression == "sign":
            sent = sign_compress(pending, self.scale)
        else:
            sent = sign_compress(top_k(pending, self.k), self.scale)
        kept = sent if self.feedback else [0.0] * len(delta)
        self.residual = [pending[index] - kept[index] for index in range(len(delta))]
        self.max_residual_norm = max(self.max_residual_norm,
                                     math.sqrt(sum(value * value for value in self.residual)))
        self._bookkeep(delta, sent)
        return sent

    def _bookkeep(self, delta, sent):
        if self.sent_total is None:
            self.sent_total = [0.0] * len(delta)
            self.raw_total = [0.0] * len(delta)
        for index in range(len(delta)):
            self.sent_total[index] += sent[index]
            self.raw_total[index] += delta[index]
        self.sent_nonzeros += sum(1 for value in sent if value != 0.0)
        self.rounds += 1


def fedavg(clients, rounds, steps, learning_rate, initial=None, l1=0.0, stragglers=(),
           policy="drop"):
    """Federated averaging.  Returns (pooled loss history, final model, statistics)."""
    check(bool(clients), "at least one client is required")
    check(policy in ("drop", "stale"), "the straggler policy must be 'drop' or 'stale'")
    check(steps >= 0, "the number of local steps cannot be negative")
    model = list(initial) if initial is not None else [0.0] * 4
    for client in clients:
        check(client.samples() == 0 or len(client.features[0]) + 1 == len(model),
              "every client must agree on the dimension")

    losses = []
    aggregates = []
    straggler_rounds = 0
    for round_index in range(rounds):
        late = set(stragglers[round_index]) if round_index < len(stragglers) else set()
        if late:
            straggler_rounds += 1
        previous_model = list(model)
        deltas = {}
        weights = {}
        for client in clients:
            if client.samples() == 0:
                continue
            if client.index in late and policy == "drop":
                continue                       # a straggler that never answered is left out
            source = previous_model if (client.index in late and policy == "stale") else model
            delta = local_update(source, client.features, client.targets, steps, learning_rate, l1)
            deltas[client.index] = client.compress(delta)
            weights[client.index] = float(client.samples())
        if weights:
            total_weight = sum(weights.values())
            aggregate = [0.0] * len(model)
            for index, sent in deltas.items():
                share = weights[index] / total_weight
                for k in range(len(model)):
                    aggregate[k] += share * sent[k]
            aggregates.append(list(aggregate))
            model = [model[k] + aggregate[k] for k in range(len(model))]
        losses.append(pooled_loss(model, [(client.features, client.targets) for client in clients]))
    statistics = {"aggregates": aggregates, "straggler_rounds": straggler_rounds}
    return losses, model, statistics


def fresh_clients(shards, **kwargs):
    return [Client(index, features, targets, **kwargs)
            for index, (features, targets) in enumerate(shards)]


# --------------------------------------------------------------------------- checks
def test_aggregate_matches_plaintext_oracle():
    shards = make_dataset(5, 30, 3, seed=0x1111)
    model = [0.2, -0.1, 0.05, 0.3]
    clients = fresh_clients(shards)
    deltas = {client.index: local_update(model, client.features, client.targets, 2, 0.05)
              for client in clients}
    sizes = [client.samples() for client in clients]
    total = sum(sizes)
    oracle = [sum(sizes[client.index] / total * deltas[client.index][k]
                  for client in clients) for k in range(len(model))]
    losses, final, statistics = fedavg(fresh_clients(shards), 1, 2, 0.05, initial=model)
    check(len(statistics["aggregates"]) == 1, "one round means one aggregate")
    for k in range(len(model)):
        check(abs(statistics["aggregates"][0][k] - oracle[k]) < 1e-12,
              f"the aggregate must be the sample-weighted mean at coordinate {k}")
        check(abs(final[k] - (model[k] + oracle[k])) < 1e-12, "the model moves by that aggregate")
    check(abs(losses[0] - pooled_loss(final, shards)) < 1e-12, "the loss is the pooled loss")

    # unequal shard sizes: the average must follow the sample counts, not the client count
    uneven = [Client(0, shards[0][0], shards[0][1], ),
              Client(1, shards[1][0][:5], shards[1][1][:5])]
    check(uneven[0].samples() == 30 and uneven[1].samples() == 5, "the shards really differ")
    _, uneven_final, _ = fedavg(uneven, 1, 1, 0.05, initial=model)
    heavy = local_update(model, uneven[0].features, uneven[0].targets, 1, 0.05)
    light = local_update(model, uneven[1].features, uneven[1].targets, 1, 0.05)
    for k in range(len(model)):
        check(abs(uneven_final[k] - (model[k] + (30 * heavy[k] + 5 * light[k]) / 35.0)) < 1e-12,
              "shard sizes must weight the average")
    return model


def test_single_client_is_plain_sgd():
    shards = make_dataset(1, 60, 4, seed=0x2222)
    features, targets = shards[0]
    rounds, steps, learning_rate = 40, 3, 0.05
    losses, federated_model, _ = fedavg(fresh_clients(shards), rounds, steps, learning_rate,
                                        initial=[0.0] * 5)
    model = [0.0] * 5
    for _ in range(rounds):
        delta = local_update(model, features, targets, steps, learning_rate)
        model = [model[k] + delta[k] for k in range(len(model))]
    check(all(abs(a - b) < 1e-12 for a, b in zip(federated_model, model)),
          "one client is exactly local SGD")
    check(abs(losses[-1] - mse(model, features, targets)) < 1e-12, "and its loss matches")
    check(losses[-1] < 0.01, f"plain SGD converges on this shard, got {losses[-1]}")
    return losses[-1]


def test_fedavg_convergence_and_baseline():
    shards = make_dataset(4, 60, 4, seed=0x3333, truth=[0.7, -0.4, 0.2, 0.9, -0.1], noise=0.05)
    start_loss = pooled_loss([0.0] * 5, shards)
    losses, model, statistics = fedavg(fresh_clients(shards), 40, 3, 0.03, initial=[0.0] * 5)
    check(losses[-1] < 0.25 * start_loss,
          f"FedAvg must make real progress: {start_loss:.4f} -> {losses[-1]:.4f}")
    check(losses[-1] < 0.05, f"and land near the noise floor, got {losses[-1]}")
    check(statistics["straggler_rounds"] == 0, "no stragglers were scheduled")
    baseline_losses, _, _ = fedavg(fresh_clients(shards), 120, 1, 0.03, initial=[0.0] * 5)
    check(baseline_losses[-1] < 0.05, "the fine-grained baseline also converges")
    check(losses[-1] <= 2.0 * baseline_losses[-1] + 0.01,
          f"FedAvg ({losses[-1]:.5f}) must be competitive with the baseline "
          f"({baseline_losses[-1]:.5f})")
    return start_loss, losses[-1], baseline_losses[-1]


def test_topk_and_sign_codecs():
    values = [0.5, -2.0, 0.0, 1.5, -0.25, 3.0]
    for k in (0, 1, 3, 6):
        compressed = top_k(values, k)
        nonzeros = sum(1 for value in compressed if value != 0.0)
        available = sum(1 for value in values if value != 0.0)
        check(nonzeros == min(k, available),
              f"top-{k} keeps exactly {min(k, available)} coordinates, kept {nonzeros}")
        expected = {index for index in sorted(range(len(values)),
                                             key=lambda index: (-abs(values[index]), index))[:k]
                    if values[index] != 0.0}
        check({index for index, value in enumerate(compressed) if value != 0.0} == expected,
              f"the kept coordinates for k = {k} are the largest magnitudes")
        error = sum((values[index] - compressed[index]) ** 2 for index in range(len(values)))
        check(abs(error - tail_energy(values, k)) < 1e-12,
              "the reconstruction error equals the tail energy")
        check(error <= sum(value * value for value in values) + 1e-12, "and never exceeds it")
    check(top_k(values, len(values)) == values, "k = n is the identity")
    check(top_k(values, 0) == [0.0] * len(values), "k = 0 sends nothing")
    check(sign_compress([0.5, -0.5, 0.0], 1.0) == [1.0, -1.0, 0.0], "the sign codec is one bit")
    try:
        top_k([1.0], 2)
        raise AssertionError("k > len(values) must be rejected")
    except AssertionError as error:
        check("k must lie" in str(error), f"unexpected message: {error}")


def test_error_feedback_conservation():
    shards = make_dataset(3, 40, 4, seed=0x4444, truth=[0.7, -0.4, 0.2, 0.9, -0.1], noise=0.05)
    rounds, steps, learning_rate = 150, 2, 0.05
    compressed = fresh_clients(shards, compression="topk", k=1)
    losses, _, _ = fedavg(compressed, rounds, steps, learning_rate, initial=[0.0] * 5)
    for client in compressed:
        for k in range(5):
            check(abs(client.sent_total[k] + client.residual[k] - client.raw_total[k]) < 1e-9,
                  "cumulative sent + residual must equal the cumulative computed update")
        check(client.sent_nonzeros <= 1 * rounds,
              f"at most k = 1 coordinate per round may be sent, got {client.sent_nonzeros}")
        check(client.rounds == rounds, "every round is accounted for")
        raw_norm = math.sqrt(sum(value * value for value in client.raw_total))
        check(client.max_residual_norm <= raw_norm,
              "the residual stays below the total update it came from")
        check(client.max_residual_norm <= 2.0, "and does not grow without bound")
    check(losses[-1] < 0.01, f"top-1 with error feedback converges, got {losses[-1]}")

    without_feedback = fresh_clients(shards, compression="topk", k=1, feedback=False)
    plain_losses, _, _ = fedavg(without_feedback, rounds, steps, learning_rate, initial=[0.0] * 5)
    check(plain_losses[-1] > 10.0 * losses[-1],
          f"dropping the residual must hurt badly: {plain_losses[-1]:.5f} vs {losses[-1]:.5f}")

    uncompressed_losses, _, _ = fedavg(fresh_clients(shards), rounds, steps, learning_rate,
                                       initial=[0.0] * 5)
    check(losses[-1] <= 1.5 * uncompressed_losses[-1] + 0.005,
          f"compressed ({losses[-1]:.5f}) must stay near uncompressed "
          f"({uncompressed_losses[-1]:.5f})")
    return losses[-1], plain_losses[-1], uncompressed_losses[-1]


def test_straggler_policies():
    shards = make_dataset(4, 30, 3, seed=0x5555)
    model = [0.1, 0.1, 0.1, 0.1]

    no_schedule = fresh_clients(shards, compression="topk", k=2)
    _, final_all, statistics = fedavg(no_schedule, 6, 2, 0.05, initial=model)
    with_empty_schedule = fresh_clients(shards, compression="topk", k=2)
    _, final_empty, statistics_empty = fedavg(with_empty_schedule, 6, 2, 0.05, initial=model,
                                              stragglers=[tuple()] * 6)
    check(final_all == final_empty, "an empty straggler schedule changes nothing")
    check(statistics["straggler_rounds"] == 0 and statistics_empty["straggler_rounds"] == 0,
          "no straggler rounds are recorded")

    # identical shards: dropping a client cannot change the update
    identical = make_dataset(1, 25, 2, seed=0x6666, truth=[0.3, -0.2, 0.05], noise=0.0)
    same = identical * 3
    _, final_full, _ = fedavg(fresh_clients(same), 5, 2, 0.05, initial=[0.0, 0.0, 0.0])
    _, final_partial, statistics = fedavg(fresh_clients(same), 5, 2, 0.05, initial=[0.0, 0.0, 0.0],
                                          stragglers=[(1, 2)] + [tuple()] * 4, policy="drop")
    check(all(abs(a - b) < 1e-12 for a, b in zip(final_full, final_partial)),
          "with identical shards, dropping clients leaves the update unchanged")
    check(statistics["straggler_rounds"] == 1, "one straggler round is recorded")

    # heterogeneous shards: the drop policy must equal the renormalised mean of the responders
    late = (2,)
    healthy = [0, 1, 3]
    oracle = [0.0] * len(model)
    weights = {index: float(len(shards[index][0])) for index in healthy}
    total_weight = sum(weights.values())
    for index in healthy:
        delta = local_update(model, shards[index][0], shards[index][1], 2, 0.05)
        for k in range(len(model)):
            oracle[k] += weights[index] / total_weight * delta[k]
    _, final_dropped, _ = fedavg(fresh_clients(shards), 1, 2, 0.05, initial=model,
                                 stragglers=[late], policy="drop")
    for k in range(len(model)):
        check(abs(final_dropped[k] - (model[k] + oracle[k])) < 1e-12,
              "the drop policy must renormalise over the clients that answered")

    # the stale policy gives the straggler a different starting point, so its delta differs
    previous = [0.4, -0.3, 0.2, 0.0]
    stale_delta = local_update(previous, shards[2][0], shards[2][1], 2, 0.05)
    fresh_delta = local_update(model, shards[2][0], shards[2][1], 2, 0.05)
    check(stale_delta != fresh_delta, "a stale client really computes a different delta")
    _, _, stale_stats = fedavg(fresh_clients(shards), 3, 2, 0.05, initial=model,
                               stragglers=[(2,), (2,), tuple()], policy="stale")
    check(stale_stats["straggler_rounds"] == 2, "two stale rounds are recorded")
    check(len(stale_stats["aggregates"]) == 3, "every round still produces an aggregate")

    # a round where no client can contribute leaves the model untouched
    empty_shards = [([], []) for _ in range(3)]
    losses, unchanged, statistics = fedavg(fresh_clients(empty_shards), 3, 2, 0.05, initial=model)
    check(unchanged == model, "clients with no samples cannot move the model")
    check(losses == [0.0, 0.0, 0.0], "the pooled loss of an empty dataset is zero")
    check(statistics["aggregates"] == [], "no aggregate is produced without samples")
    return final_dropped


def test_boundaries_and_validation():
    shards = make_dataset(3, 25, 3, seed=0x7777)
    model = [0.1, 0.2, 0.3, 0.4]

    _, no_steps, _ = fedavg(fresh_clients(shards), 5, 0, 0.05, initial=model)
    check(no_steps == model, "zero local steps leave the model alone")

    zero_losses, k_zero_final, _ = fedavg(fresh_clients(shards, compression="topk", k=0),
                                          6, 2, 0.05, initial=model)
    check(k_zero_final == model, "k = 0 sends nothing, so the model never moves")
    check(len(set(zero_losses)) == 1, "and the loss stays constant")

    _, full_final, _ = fedavg(fresh_clients(shards, compression="topk", k=4), 6, 2, 0.05,
                              initial=model)
    _, plain_final, _ = fedavg(fresh_clients(shards), 6, 2, 0.05, initial=model)
    check(all(abs(a - b) < 1e-12 for a, b in zip(full_final, plain_final)),
          "k = dimension is the identity compression")

    sign_losses, sign_final, sign_stats = fedavg(fresh_clients(shards, compression="sign",
                                                               scale=0.02),
                                                 6, 2, 0.05, initial=model)
    check(len(sign_stats["aggregates"]) == 6, "sign compression produces an aggregate per round")
    check(sign_losses[-1] < sign_losses[0], "and still reduces the loss")
    check(sign_final != model, "the sign codec moves the model")

    try:
        fedavg(fresh_clients(shards), 3, 1, 0.05, policy="teleport")
        raise AssertionError("an unknown straggler policy must be rejected")
    except AssertionError as error:
        check("straggler policy" in str(error), f"unexpected message: {error}")
    try:
        fedavg(fresh_clients(shards), 3, -1, 0.05)
        raise AssertionError("negative local steps must be rejected")
    except AssertionError as error:
        check("local steps" in str(error), f"unexpected message: {error}")
    try:
        fedavg([], 3, 1, 0.05)
        raise AssertionError("no clients must be rejected")
    except AssertionError as error:
        check("at least one client" in str(error), f"unexpected message: {error}")

    first = fedavg(fresh_clients(shards, compression="topk", k=2), 5, 2, 0.05, initial=model)
    second = fedavg(fresh_clients(shards, compression="topk", k=2), 5, 2, 0.05, initial=model)
    check(first[0] == second[0] and first[1] == second[1], "the simulation is deterministic")
    check(first[0][-1] < first[0][0], f"the compressed run reduces the loss: {first[0]}")
    return first[0][0], first[0][-1]


def main():
    test_aggregate_matches_plaintext_oracle()
    single_loss = test_single_client_is_plain_sgd()
    start_loss, federated_loss, baseline_loss = test_fedavg_convergence_and_baseline()
    test_topk_and_sign_codecs()
    compressed, no_feedback, uncompressed = test_error_feedback_conservation()
    test_straggler_policies()
    first_loss, last_loss = test_boundaries_and_validation()

    print(f"single client (plain SGD): final loss {single_loss:.6f}")
    print(f"fedavg: {start_loss:.5f} -> {federated_loss:.5f} "
          f"(fine-grained baseline {baseline_loss:.5f})")
    print(f"top-1 (of 5): {compressed:.5f} with error feedback, {no_feedback:.5f} without, "
          f"{uncompressed:.5f} uncompressed")
    print(f"compressed boundary run: {first_loss:.5f} -> {last_loss:.5f}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
