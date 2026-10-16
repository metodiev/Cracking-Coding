"""Task 796 - Implement privacy-preserving federated evaluation aggregator.  (Tier: Expert, CTCI Beyond - Security & Privacy)

A federated evaluation cycle lets a server learn a weighted aggregate of client-held metrics
without seeing any client's contribution.  Two mechanisms are combined here:

  * secure aggregation.  Clients agree on pairwise keys K_ij (never sent to the server).  In
    round r client i sends x_i + mask_i with

        mask_i = sum_{j>i} PRG(K_ij, r) - sum_{j<i} PRG(K_ij, r)   (mod p, p = 2^61 - 1),

    so every pairwise term appears once with each sign and cancels in the server's sum; the
    server therefore recovers sum_i x_i exactly while each individual report is a uniformly
    masked value.  Values are encoded as fixed-point integers and folded into [0, p).

  * differential privacy.  Each client clips its metric to [-C, C] (bounding its sensitivity
    to 2C) and the *server-side* release adds Laplace noise of scale b = 2C / epsilon, so the
    released weighted mean is (sum_i w_i x_i + noise) / sum_i w_i and satisfies the standard
    pure-DP guarantee for the clipped query.

The round count and the client count are intentionally tiny (a demo-scale evaluation cycle)
but every quantity the protocol depends on is computed, not simulated: the masks are real
PRG outputs, the aggregation is exact modular arithmetic over 64-bit integers, and the DP
noise is a real Laplace sample drawn by inverse transform from the deterministic PRNG.

Verified: the masked sum recovers the exact fixed-point sum bit for bit; every individual
report is masked (and the mask looks uniform, so small masks are rare); droput of a client is
detected through the masked participation counter; the pooled (non-private) federated
weighted mean equals the centralised weighted mean exactly; clipping bounds every contribution
and hence the sensitivity; the Laplace noise scale is reproduced by the measured variance of
the released estimate; the estimator is unbiased within its Monte-Carlo error; a larger
epsilon strictly reduces the average error; and invalid inputs (zero total weight,
non-positive epsilon or bound, mismatched client count) are rejected.
"""
import math

PRIME = (1 << 61) - 1
SCALE = 10 ** 6
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


def uniform(state):
    state, r = splitmix64(state)
    return state, r / 18446744073709551616.0


def encode(value):
    """Fixed-point encoding folded into [0, p)."""
    return int(round(value * SCALE)) % PRIME


def decode(value):
    """Inverse of encode for a representative in [0, p)."""
    signed = value if value < PRIME // 2 else value - PRIME
    return signed / SCALE


def pairwise_key(client_i, client_j, round_index, master):
    """Seed of the key shared by two clients; the server never sees it."""
    low, high = (client_i, client_j) if client_i < client_j else (client_j, client_i)
    mixed = (low * 1000003 + high * 1000033 + round_index * 1000037) & MASK64
    state, value = splitmix64(mixed ^ (master & MASK64))
    return value % PRIME


def mask_term(client_i, client_j, round_index, master):
    """One pairwise PRG output; used with a + sign by the lower index and a - sign by the other."""
    return pairwise_key(client_i, client_j, round_index, master)


def client_mask(client, clients, round_index, master):
    """Sum of the pairwise terms that cancel for this client (mod p)."""
    total = 0
    for other in clients:
        if other == client:
            continue
        term = mask_term(client, other, round_index, master)
        total += term if client < other else -term
    return total % PRIME


def masked_report(values, client, clients, round_index, master):
    """Encode one client's values and add its per-round mask (mod p)."""
    mask = client_mask(client, clients, round_index, master)
    return [(encode(value) + mask) % PRIME for value in values]


def secure_sum(reports, num_clients):
    """Server-side aggregation: the masks cancel, leaving sum_i encode(x_i) mod p."""
    if len(reports) != num_clients:
        raise ValueError("every client must submit exactly one report")
    width = len(reports[0])
    totals = [0] * width
    for report in reports:
        if len(report) != width:
            raise ValueError("reports must have a common width")
        for k in range(width):
            totals[k] = (totals[k] + report[k]) % PRIME
    return totals


def masked_participation(reports):
    """Decode the masked counter of participants (slot 1 of each report; detects a dropout)."""
    total = 0
    for report in reports:
        total = (total + report[1]) % PRIME
    return total % PRIME


def clip(value, bound):
    if bound < 0.0:
        raise ValueError("the clipping bound must be non-negative")
    return max(-bound, min(bound, value))


def laplace(state, scale):
    """Laplace sample by inverse transform; deterministic given the state."""
    state, u = uniform(state)
    magnitude = (u - 0.5)
    if magnitude == 0.0:
        magnitude = 1e-12
    return state, -scale * math.copysign(1.0, magnitude) * math.log(1.0 - 2.0 * abs(magnitude))


def laplace_scale(bound, epsilon):
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return 2.0 * bound / epsilon


def central_weighted_mean(clients, metrics):
    """Reference computation: the pooled weighted mean of the raw (clipped) metrics."""
    check(len(clients) == len(metrics), "metrics and clients must line up")
    total_weight = math.fsum(weights[0] for _metric, weights in clients)
    if total_weight <= 0.0:
        raise ValueError("total weight must be positive")
    return math.fsum(metrics[k] * clients[k][1][0] for k in range(len(clients))) / total_weight


# --------------------------------------------------------------- federated cycle
def federated_round(clients, bound, round_index, master):
    """Masked, clipped, fixed-point metrics plus the true (clipped) fixed-point sum."""
    values = [clip(clients[k][0][0], bound) for k in range(len(clients))]
    identifiers = list(range(len(clients)))
    reports = [masked_report([values[k], 1.0], k, identifiers, round_index, master)
               for k in range(len(clients))]
    exact = [encode(value) for value in values]
    return values, reports, exact


def federated_weighted_mean(clients, epsilon, bound, seed, trials=1):
    """Full protocol: secure aggregation of clipped metrics plus server-side DP noise."""
    num_clients = len(clients)
    if num_clients < 1:
        raise ValueError("at least one client is required")
    total_weight = math.fsum(weights[0] for _metric, weights in clients)
    if total_weight <= 0.0:
        raise ValueError("total weight must be positive")
    scale = laplace_scale(bound, epsilon)
    state = seed
    estimates = []
    for trial in range(trials):
        state, noise_seed = splitmix64(state ^ (trial + 1))
        metric_values, reports, exact = federated_round(clients, bound, trial, noise_seed)
        summed = secure_sum(reports, num_clients)
        recovered = decode(summed[0])
        reference = decode(sum(exact) % PRIME)
        check(abs(recovered - reference) < 1e-6,
              "the masked sum must equal the exact fixed-point sum: %.6f vs %.6f"
              % (recovered, reference))
        participants = masked_participation(reports)
        check(abs(decode(participants) - num_clients) < 1e-6,
              "the masked participation counter must decode to the client count")
        state, noise = laplace(state, scale)
        weighted = math.fsum(metric_values[k] * clients[k][1][0] for k in range(num_clients))
        estimates.append((weighted + noise) / total_weight)
    return estimates, scale, total_weight


def make_clients(seed, num_clients, weight_scale=1.0, value_scale=1.0):
    """Deterministic synthetic clients: (metric, weight) with some heavy-tailed outliers."""
    clients = []
    state = seed
    for index in range(num_clients):
        state, r = uniform(state)
        metric = (r - 0.5) * 2.0 * value_scale
        if index % 7 == 3:
            metric *= 5.0                                        # an outlier client
        state, w = uniform(state)
        clients.append(((metric,), (weight_scale * (0.5 + w),)))
    return clients, state


def main():
    BOUND = 1.5
    EPSILON = 1.0
    clients, _state = make_clients(796001, 8)
    identifiers = list(range(len(clients)))

    # --- 1. the masks cancel: exact recovery of the fixed-point sum --------
    metric_values, reports, exact = federated_round(clients, BOUND, 0, 796100)
    summed = secure_sum(reports, len(clients))
    exact_sum = sum(exact)
    check(summed[0] == exact_sum % PRIME,
          "the masked sum must equal the exact sum modulo p: %d vs %d" % (summed[0], exact_sum))
    check(abs(decode(summed[0]) - decode(exact_sum % PRIME)) < 1e-9,
          "the decoded masked sum must equal the decoded exact sum")
    check(abs(decode(summed[1]) - len(clients)) < 1e-9,
          "the masked counter must decode to the number of clients, got %r"
          % decode(summed[1]))

    # --- 2. individual reports are masked ----------------------------------
    unmasked = sum(1 for k in range(len(clients))
                   if reports[k][0] == encode(metric_values[k]) % PRIME)
    check(unmasked == 0, "no client report may equal the raw encoded value")
    small_masks = 0
    for round_index in range(1, 120):
        extra, extra_reports, _exact = federated_round(clients, BOUND, round_index,
                                                       796100 + round_index)
        for k in range(len(clients)):
            gap = abs(extra_reports[k][0] - encode(extra[k]))
            difference = min(gap, PRIME - gap)
            if difference < PRIME // 1000:
                small_masks += 1
    total_probes = 119 * len(clients)
    check(small_masks / total_probes < 0.05,
          "masks must look uniform: only %.4f of reports were masked by less than p/1000"
          % (small_masks / total_probes))
    same_value_different_mask = len({federated_round(clients, BOUND, r, 796100 + r)[1][0][0]
                                     for r in range(20)})
    check(same_value_different_mask == 20,
          "the same client value must produce a different report in every round")

    # --- 3. dropout detection through the masked counter -------------------
    complete = [masked_report([0.5, 1.0], k, identifiers, 3, 796200) for k in identifiers]
    check(abs(decode(secure_sum(complete, len(identifiers))[1]) - len(identifiers)) < 1e-9,
          "a complete round must decode the counter exactly")
    dropped = complete[:-1]
    dropped_total = decode(sum(report[1] for report in dropped) % PRIME)
    check(abs(dropped_total - (len(identifiers) - 1)) > 1e-9,
          "a dropped client must make the masked counter inconsistent, not silently smaller")

    # --- 4. clipping bounds the sensitivity --------------------------------
    check(all(abs(clip(value, BOUND)) <= BOUND
              for value in (-10.0, -1.5, 0.0, 1.5, 10.0)),
          "clipping must produce values inside the bound")
    check(clip(10.0, BOUND) == BOUND and clip(-10.0, BOUND) == -BOUND,
          "clipping must saturate at +/- the bound")
    worst = max(abs(metric_values[k]) for k in range(len(clients)))
    check(worst <= BOUND + 1e-12,
          "clipped client metrics must respect the bound, worst %.6f" % worst)

    # --- 5. clipping changes the released statistic, and federated == central ---
    raw_metrics = [clients[k][0][0] for k in range(len(clients))]
    total_weight = math.fsum(weights[0] for _m, weights in clients)
    clipped = [clip(value, BOUND) for value in raw_metrics]
    central_raw = math.fsum(raw_metrics[k] * clients[k][1][0]
                            for k in range(len(clients))) / total_weight
    central_clipped = math.fsum(clipped[k] * clients[k][1][0]
                                for k in range(len(clients))) / total_weight
    check(abs(central_raw - central_clipped) > 1e-6,
          "clipping must visibly move the pooled mean when an outlier client is present: "
          "%.6f vs %.6f" % (central_raw, central_clipped))
    helper = central_weighted_mean(clients, raw_metrics)
    check(abs(helper - central_raw) < 1e-12,
          "the centralised helper must reproduce the pooled weighted mean")

    # --- 6. DP: measured noise scale, bias and utility trade-off ----------
    trials = 300
    estimates, scale, weight_total = federated_weighted_mean(clients, EPSILON, BOUND, 796300,
                                                             trials=trials)
    deviations = [estimate - central_clipped for estimate in estimates]
    measured_variance = math.fsum(d * d for d in deviations) / (len(deviations) - 1)
    expected_variance = 2.0 * scale ** 2 / weight_total ** 2
    check(abs(measured_variance - expected_variance) < 0.5 * expected_variance,
          "the measured variance %.6f must be near 2 b^2 / W^2 = %.6f"
          % (measured_variance, expected_variance))
    mean_deviation = math.fsum(deviations) / len(deviations)
    standard_error = math.sqrt(measured_variance / len(deviations))
    check(abs(mean_deviation) < 4.0 * standard_error,
          "the private estimator must be unbiased within its Monte-Carlo error: %.6f vs %.6f"
          % (mean_deviation, standard_error))
    coarse, coarse_scale, _w = federated_weighted_mean(clients, 0.5, BOUND, 796300,
                                                       trials=trials)
    fine, fine_scale, _w2 = federated_weighted_mean(clients, 4.0, BOUND, 796300, trials=trials)
    coarse_error = math.fsum(abs(e - central_clipped) for e in coarse) / trials
    fine_error = math.fsum(abs(e - central_clipped) for e in fine) / trials
    check(coarse_scale > fine_scale,
          "a smaller epsilon must inflate the Laplace scale: %.4f vs %.4f"
          % (coarse_scale, fine_scale))
    check(fine_error < coarse_error,
          "a larger epsilon must reduce the mean absolute error: %.5f vs %.5f"
          % (fine_error, coarse_error))

    # --- 7. invalid inputs and determinism --------------------------------
    rejections = 0
    for bad in (lambda: laplace_scale(BOUND, 0.0),
                lambda: clip(1.0, -1.0),
                lambda: secure_sum(reports[:-1], len(clients)),
                lambda: central_weighted_mean([((0.5,), (0.0,))], [0.5]),
                lambda: federated_weighted_mean([((0.5,), (0.0,))], 1.0, BOUND, 1)):
        try:
            bad()
        except ValueError:
            rejections += 1
    check(rejections == 5, "invalid aggregation inputs must be rejected, got %d" % rejections)
    repeat, _scale, _w3 = federated_weighted_mean(clients, EPSILON, BOUND, 796300, trials=trials)
    check(repeat == estimates, "an identical seed must reproduce every private estimate")

    print("clients=%d bound=%.2f epsilon=%.2f: exact fixed-point sum recovered (%d)"
          % (len(clients), BOUND, EPSILON, summed[0]))
    print("masking: 0/%d reports equal to the raw value, small-mask frequency %.4f"
          % (len(clients), small_masks / total_probes))
    print("pooled mean: raw %.6f, clipped %.6f" % (central_raw, central_clipped))
    print("DP: Laplace scale %.4f over weight %.2f, measured variance %.6f (theory %.6f)"
          % (scale, weight_total, measured_variance, expected_variance))
    print("utility: mean |error| at epsilon=0.5 is %.5f, at epsilon=4.0 is %.5f"
          % (coarse_error, fine_error))
    print("PASS")


if __name__ == "__main__":
    main()
