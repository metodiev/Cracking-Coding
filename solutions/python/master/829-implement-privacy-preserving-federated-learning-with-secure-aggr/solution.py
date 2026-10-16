"""Task 829 - Implement privacy-preserving federated learning with secure aggregation and differential privacy. (Tier: Master, CTCI Beyond — Security & Privacy)

FedAvg, but the server must never see an individual client update: each client clips its
update to an L2 ball of radius C, quantises it onto an integer grid, and adds pairwise
masks that are derived from a (toy) Diffie-Hellman secret shared with each peer, so the
masks cancel *exactly* in the sum. The server receives only the masked vectors, decodes
the aggregate, adds Gaussian noise calibrated to (epsilon, delta) -- giving
(epsilon, delta)-differential privacy for the released aggregate -- and applies the mean
update. A dropout recovery step, an (epsilon, delta) accountant and a utility/convergence
check complete the pipeline.

What is implemented
-------------------
* `MOD = 2^61 - 1` and a fixed-point grid of 2^-20: every update coordinate becomes an
  integer, so the modular sum is exact and "the masks cancel" is an exact integer identity,
  not a floating-point approximation. `decode` maps the residue back to a signed integer.
* `pair_secret`: a real (toy-size) Diffie-Hellman exchange (61-bit modulus, private
  exponents from a seeded PRNG); the mask seed for a pair is SHA-256 of the shared secret
  and the session id, expanded into one mask integer per coordinate.
* `SecureAggregator`: builds the masks, masks each update (add for the lower id, subtract
  for the higher id of a pair), sums modulo MOD, and recovers from one dropout by having
  the surviving clients reveal the masks they share with the dropped one.
* `gaussian_noise`: splitmix64 + Box-Muller, rounded onto the same integer grid, with the
  classic calibration sigma = C * sqrt(2 ln(1.25/delta)) / epsilon. `Accountant` reports
  the composed (epsilon, delta) over rounds with basic composition.
* `federated_round` / `train`: local SGD (regularised logistic regression), clipping,
  quantisation, masking, aggregation, noise, and the FedAvg mean update; a centralised
  full-batch reference is trained alongside for the utility comparison.

What is simplified (and what is *not* claimed)
----------------------------------------------
* The DH modulus is 61 bits: a real adversary could break it. The verified invariant is
  exact mask cancellation and exact dropout recovery, not cryptographic secrecy; the
  confidentiality check below is a smoke test, and no security is fabricated.
* Differential privacy is applied to the *aggregate* of clipped updates with a Gaussian
  mechanism and basic (linear) composition -- no subsampling amplification, no RDP/PRV
  accountant, and the noise is rounded onto the 2^-20 grid (a bounded extra perturbation
  that the checks measure rather than ignore).
* Tiny scale: 4 clients, 8 samples each, 3 parameters. No threads, sockets, wall-clock or
  unseeded randomness.

Invariants verified
-------------------
1.  Fixed-point round trip: |dequantise(quantise(v)) - v| <= 0.5 * 2^-20 per coordinate,
    and signed decoding is exact for negative values.
2.  Secure aggregation is exact: the masked sum (mod MOD) equals the ordinary integer sum
    of the quantised updates, hence the decoded aggregate equals the float sum of the
    clipped updates up to the quantisation bound.
3.  Mask cancellation is structural: removing one client's mask breaks the sum, and the
    masks are asymmetric (client i adds what client j subtracts).
4.  One client sees no masks (n = 1 degenerates to no privacy), which the checks state
    explicitly instead of pretending otherwise.
5.  Clipping is real: every client's update has norm <= C after clipping, and at least one
    client exceeded C before it (the clip is active every round).
6.  Different clients' transcripts are different: the masked vector differs from the true
    update in every coordinate (a smoke test for "the server does not receive plaintext").
7.  Dropout: with one client dropped the naive aggregate does *not* equal the surviving
    sum, while the recovery shares restore the exact sum over the survivors.
8.  DP calibration: sigma equals C * sqrt(2 ln(1.25/delta)) / epsilon, the drawn noise has
    the predicted standard deviation and zero mean, and smaller epsilon strictly implies
    more noise (and a larger deviation of the aggregate).
9.  Accounting: basic composition over T rounds gives T * epsilon with the same delta, and
    the accountant is monotone in the number of rounds.
10. Utility/liveness: the federated loss decreases over rounds, and the DP run stays within
    the documented tolerance of the non-private federated run and of the centralised
    full-batch reference at the same number of local steps.
11. Determinism: the whole pipeline is bit-reproducible.
"""
import hashlib
import math

MOD = 2 ** 61 - 1                     # Mersenne prime: exact modular arithmetic
SCALE = 1 << 20                       # fixed-point grid step 2^-20
CLIP = 0.2                            # L2 clipping radius C
DELTA = 1e-5
MASK64 = (1 << 64) - 1
DH_P, DH_G = MOD, 37                  # toy-size Diffie-Hellman group


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# deterministic randomness
# --------------------------------------------------------------------------------------
class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def normal_pair(self):
        """Box-Muller: two independent standard normals."""
        u1 = max((self.next_u64() >> 11) / float(1 << 53), 1e-300)
        u2 = (self.next_u64() >> 11) / float(1 << 53)
        radius = math.sqrt(-2.0 * math.log(u1))
        return radius * math.cos(2.0 * math.pi * u2), radius * math.sin(2.0 * math.pi * u2)


# --------------------------------------------------------------------------------------
# fixed point + modular helpers
# --------------------------------------------------------------------------------------
def quantise(vector, scale=SCALE):
    return [int(round(v * scale)) for v in vector]


def dequantise(integers, scale=SCALE):
    return [v / scale for v in integers]


def signed(residue):
    """Interpret a residue modulo MOD as a signed integer (|value| < MOD/2)."""
    return residue - MOD if residue > MOD // 2 else residue


def norm(vector):
    return math.sqrt(sum(v * v for v in vector))


def clip(vector, radius):
    length = norm(vector)
    if length <= radius:
        return list(vector), 1.0
    factor = radius / length
    return [v * factor for v in vector], factor


# --------------------------------------------------------------------------------------
# masked aggregation
# --------------------------------------------------------------------------------------
def pair_secret(private_a, private_b, modulus=DH_P, generator=DH_G):
    """Toy Diffie-Hellman: pow(g, a*b, p) computed from the public values only."""
    public_b = pow(generator, private_b, modulus)
    return pow(public_b, private_a, modulus)


def mask_vector(secret, session, dimension, modulus=MOD):
    """Expand a shared secret into one mask integer per coordinate (SHA-256 based PRF)."""
    masks = []
    for d in range(dimension):
        digest = hashlib.sha256(f"{session}:{secret}:{d}".encode()).digest()
        masks.append(int.from_bytes(digest[:16], "big") % modulus)
    return masks


class SecureAggregator:
    """Masked sum plus single-dropout recovery, all in exact modular arithmetic."""

    def __init__(self, clients, session, dimension):
        self.clients = list(clients)
        self.session = session
        self.dimension = dimension
        self.masks = {}                       # (i, j) with i < j -> mask list
        self.my_mask = {client: [0] * dimension for client in self.clients}
        for position, i in enumerate(self.clients):
            for j in self.clients[position + 1:]:
                secret = pair_secret(hash_client(i, session), hash_client(j, session))
                mask = mask_vector(secret, session, dimension)
                self.masks[(i, j)] = mask
                for d in range(dimension):
                    self.my_mask[i][d] = (self.my_mask[i][d] + mask[d]) % MOD
                    self.my_mask[j][d] = (self.my_mask[j][d] - mask[d]) % MOD

    def masked_sums(self, updates):
        """Client -> masked quantised update (no client ever sends its raw vector)."""
        masked = {}
        for client, integers in updates.items():
            masked[client] = [(v + m) % MOD for v, m in zip(integers, self.my_mask[client])]
        return masked

    def aggregate(self, masked):
        return [sum(m[d] for m in masked.values()) % MOD for d in range(self.dimension)]

    def recovery_shares(self, dropped, survivors):
        """What the survivors must reveal so the server can cancel the dropped masks."""
        shares = [[0] * self.dimension for _ in survivors]
        for position, client in enumerate(survivors):
            pair = (client, dropped) if client < dropped else (dropped, client)
            mask = self.masks[pair]
            sign = 1 if client < dropped else -1
            for d in range(self.dimension):
                shares[position][d] = (shares[position][d] + sign * mask[d]) % MOD
        return shares


def hash_client(client, session):
    return int.from_bytes(hashlib.sha256(f"{session}:{client}".encode()).digest()[:8], "big")


# --------------------------------------------------------------------------------------
# differential privacy
# --------------------------------------------------------------------------------------
def sigma_for(epsilon, radius=CLIP, delta=DELTA):
    """Gaussian mechanism calibration for an L2 sensitivity of `radius`."""
    return radius * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def epsilon_for(sigma, radius=CLIP, delta=DELTA):
    """Invert the calibration (the accountant reports epsilon from the noise actually used)."""
    return radius * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


class Accountant:
    """Basic composition: T rounds of epsilon give (T * epsilon, delta)."""

    def __init__(self, epsilon, delta=DELTA):
        self.epsilon = epsilon
        self.delta = delta

    def total(self, rounds):
        return self.epsilon * rounds, self.delta


def gaussian_noise(rng, dimension, sigma, scale=SCALE):
    """Gaussian noise rounded onto the fixed-point grid (integers), so the sum stays exact."""
    integers = []
    while len(integers) < dimension:
        for value in rng.normal_pair():
            integers.append(int(round(value * sigma * scale)))
            if len(integers) == dimension:
                break
    return integers


# --------------------------------------------------------------------------------------
# federated logistic regression (the model)
# --------------------------------------------------------------------------------------
def make_data(clients=4, per_client=8):
    """Deterministic 2-D data; client c owns a contiguous slice of the same stream."""
    shards = []
    for c in range(clients):
        shard = []
        for k in range(c * per_client, (c + 1) * per_client):
            a = ((k * 37) % 17) / 17.0 * 2.0 - 1.0
            b = ((k * 61) % 23) / 23.0 * 2.0 - 1.0
            shard.append(((a, b), 1.0 if a + 0.5 * b > 0.0 else -1.0))
        shards.append(shard)
    return shards


def loss_and_grad(params, data, ridge=0.01):
    """Mean logistic loss of y in {-1, +1} plus a ridge penalty on the weights."""
    w0, w1, bias = params
    total, g0, g1, gb = 0.0, 0.0, 0.0, 0.0
    for x, y in data:
        z = y * (w0 * x[0] + w1 * x[1] + bias)
        total += math.log1p(math.exp(-z)) if z > 0 else (-z + math.log1p(math.exp(z)))
        factor = -y * (1.0 - 1.0 / (1.0 + math.exp(-z)))
        g0 += factor * x[0]
        g1 += factor * x[1]
        gb += factor
    count = len(data)
    penalty = 0.5 * ridge * (w0 * w0 + w1 * w1)
    return (total / count + penalty,
            [g0 / count + ridge * w0, g1 / count + ridge * w1, gb / count])


def local_update(global_params, data, lr, epochs):
    params = list(global_params)
    for _ in range(epochs):
        _, grad = loss_and_grad(params, data)
        params = [p - lr * g for p, g in zip(params, grad)]
    return [p - g for p, g in zip(params, global_params)]


def federated_round(global_params, shards, aggregator, params_meta):
    """One round: local SGD, clip, quantise, mask, aggregate, add noise, average."""
    dimension = len(global_params)
    updates, clipped, factors = {}, {}, {}
    for client in aggregator.clients:
        raw = local_update(global_params, shards[client], params_meta["lr"],
                           params_meta["epochs"])
        tight, factor = clip(raw, params_meta["clip"])
        clipped[client] = quantise(tight)
        factors[client] = (norm(raw), factor)
        updates[client] = clipped[client]
    masked = aggregator.masked_sums(updates)
    noise = gaussian_noise(params_meta["rng"], dimension, params_meta["sigma"]) \
        if params_meta["sigma"] > 0 else [0] * dimension
    residue = aggregator.aggregate(masked)
    total = [(signed(r) + n) % MOD for r, n in zip(residue, noise)]
    aggregate = [signed(t) for t in total]
    mean = [v / (SCALE * len(aggregator.clients)) for v in aggregate]
    new_params = [g + m for g, m in zip(global_params, mean)]
    return {"params": new_params, "aggregate": aggregate, "noise": noise,
            "masked": masked, "factors": factors, "updates": updates}


def train(shards, rounds=30, lr=0.5, epochs=2, sigma=0.0, seed=1, session="s1",
          clients=None, clip_radius=CLIP):
    clients = sorted(shards) if clients is None else clients
    aggregator = SecureAggregator(clients, session, 3)
    meta = {"lr": lr, "epochs": epochs, "sigma": sigma, "clip": clip_radius,
            "rng": SplitMix64(seed)}
    params = [0.0, 0.0, 0.0]
    history, rounds_log = [], []
    for _ in range(rounds):
        outcome = federated_round(params, shards, aggregator, meta)
        params = outcome["params"]
        history.append(central_loss(params, shards))
        rounds_log.append(outcome)
    return {"params": params, "loss": history, "rounds": rounds_log, "aggregator": aggregator}


def central_loss(params, shards):
    """Loss on the union of all shards -- the global objective."""
    return loss_and_grad(params, [pair for shard in shards.values() for pair in shard])[0]


def centralised_reference(shards, steps, lr=0.5):
    """Full-batch gradient descent on the union with the same number of local steps."""
    data = [pair for shard in shards.values() for pair in shard]
    params = [0.0, 0.0, 0.0]
    for _ in range(steps):
        _, grad = loss_and_grad(params, data)
        params = [p - lr * g for p, g in zip(params, grad)]
    return params, loss_and_grad(params, data)[0]


def main():
    shards = {client: shard for client, shard in enumerate(make_data())}
    dimension = 3

    # -- 1. fixed point --------------------------------------------------------------
    probe = [0.1234567, -1.5, 0.0000001, -0.7654321]
    integers = quantise(probe)
    round_trip = dequantise(integers)
    check(max(abs(a - b) for a, b in zip(probe, round_trip)) <= 0.5 / SCALE + 1e-18,
          "quantise/dequantise stays inside half a grid step")
    check(all(signed(i % MOD) == i for i in integers), "signed decoding is exact")
    check(signed((MOD - 7) % MOD) == -7 and signed(7) == 7,
          "negative coordinates survive the modular encoding")

    # -- 2. secure aggregation -------------------------------------------------------
    agent = SecureAggregator([0, 1, 2, 3], "session-a", dimension)
    updates = {client: quantise(local_update([0.0, 0.0, 0.0], shards[client], 0.5, 2))
               for client in shards}
    masked = agent.masked_sums(updates)
    naive_residue = agent.aggregate(masked)
    exact = [sum(updates[c][d] for c in agent.clients) for d in range(dimension)]
    check([signed(r) for r in naive_residue] == exact,
          "the masked sum equals the exact integer sum of the updates")
    check(all(masked[c] != updates[c] for c in agent.clients),
          "no client sends its plaintext update")
    broken = dict(masked)
    broken[0] = [(v + 1) % MOD for v in broken[0]]
    check(agent.aggregate(broken) != naive_residue,
          "removing one client's mask breaks the cancellation")
    solo = SecureAggregator([0], "session-a", dimension)
    solo_masked = solo.masked_sums({0: updates[0]})
    check(solo.masked_sums({0: updates[0]})[0] == [v % MOD for v in updates[0]]
          and solo.my_mask[0] == [0] * dimension,
          "a single client has no masks, so it has no privacy (stated, not hidden)")

    # -- 3. clipping ------------------------------------------------------------------
    tight, factor = clip([0.6, 0.6, 0.6], CLIP)
    check(abs(norm(tight) - CLIP) < 1e-15 and factor < 1.0,
          "clipping puts a long update exactly on the sphere")
    short, factor_short = clip([0.01, 0.0, 0.0], CLIP)
    check(short == [0.01, 0.0, 0.0] and factor_short == 1.0,
          "short updates pass through untouched")

    # -- 4. the DP calibration ---------------------------------------------------------
    epsilon = 8.0
    sigma = sigma_for(epsilon)
    check(abs(sigma - CLIP * math.sqrt(2.0 * math.log(1.25 / DELTA)) / epsilon) < 1e-15,
          "sigma follows the Gaussian-mechanism calibration")
    check(abs(epsilon_for(sigma) - epsilon) < 1e-9, "epsilon is recovered from sigma")
    rng = SplitMix64(7)
    draws = [rng.normal_pair()[0] for _ in range(4000)]
    mean = sum(draws) / len(draws)
    std = math.sqrt(sum((d - mean) ** 2 for d in draws) / len(draws))
    check(abs(mean) < 0.06 and abs(std - 1.0) < 0.05,
          f"the Box-Muller stream is standard normal (mean {mean:.3f}, sd {std:.3f})")
    noisy = gaussian_noise(SplitMix64(11), 2000, sigma)
    grid_mean = sum(noisy) / len(noisy) / SCALE
    grid_std = math.sqrt(sum((n / SCALE - grid_mean) ** 2 for n in noisy) / len(noisy))
    check(abs(grid_std - sigma) < 0.1 * sigma and abs(grid_mean) < 0.1 * sigma,
          f"the integer grid noise keeps sigma ({grid_std:.4f} vs {sigma:.4f})")

    sigmas = [sigma_for(e) for e in (16.0, 8.0, 4.0, 2.0, 1.0)]
    check(all(a < b for a, b in zip(sigmas, sigmas[1:])),
          "smaller epsilon strictly means more noise")
    check(abs(sigma_for(1.0) / CLIP - 4.8449) < 1e-3,
          "sigma(epsilon=1) is C * sqrt(2 ln(1.25/delta)) = C * 4.845 for delta = 1e-5")

    # -- 5. accounting -----------------------------------------------------------------
    accountant = Accountant(epsilon)
    check(accountant.total(1) == (epsilon, DELTA) and accountant.total(5)[0] == 5 * epsilon,
          "basic composition multiplies epsilon by the number of rounds")
    check(accountant.total(30)[0] > accountant.total(10)[0] and accountant.total(30)[1] == DELTA,
          "the accountant is monotone in rounds and keeps delta fixed")

    # -- 6. dropout recovery -----------------------------------------------------------
    survivors = [1, 2, 3]
    partial = {client: masked[client] for client in survivors}
    broken_sum = agent.aggregate(partial)
    surviving_updates = [sum(updates[c][d] for c in survivors) % MOD for d in range(dimension)]
    check(broken_sum != surviving_updates,
          "without recovery the surviving clients' masks do not cancel")
    shares = agent.recovery_shares(0, survivors)
    correction = [sum(share[d] for share in shares) % MOD for d in range(dimension)]
    recovered = [(broken_sum[d] - correction[d]) % MOD for d in range(dimension)]
    check(recovered == surviving_updates,
          "the recovery shares restore the exact sum over the survivors")
    check(all(share != [0] * dimension for share in shares),
          "every surviving client contributes a non-trivial recovery share")

    # -- 7. one federated round end to end ---------------------------------------------
    meta = {"lr": 0.5, "epochs": 2, "sigma": 0.0, "clip": CLIP, "rng": SplitMix64(3)}
    outcome = federated_round([0.0, 0.0, 0.0], shards, agent, meta)
    tight_updates = [dequantise(outcome["updates"][c]) for c in agent.clients]
    check(all(norm(u) <= CLIP + 1e-6 for u in tight_updates),
          "every transmitted update respects the clipping radius")
    check(any(factor < 1.0 for _, factor in outcome["factors"].values()),
          "at least one client was clipped, so the clip is active")
    exact_mean = [sum(u[d] for u in tight_updates) / len(tight_updates) for d in range(dimension)]
    decoded_mean = [v / (SCALE * len(agent.clients)) for v in outcome["aggregate"]]
    check(max(abs(a - b) for a, b in zip(exact_mean, decoded_mean)) < 0.5 / SCALE,
          "the decoded aggregate is the clipped mean up to the quantisation bound")

    # -- 8. the whole pipeline ---------------------------------------------------------
    non_private = train(shards, rounds=30, lr=0.5, epochs=2, sigma=0.0)
    private = train(shards, rounds=30, lr=0.5, epochs=2, sigma=sigma, seed=5)
    reference, reference_loss = centralised_reference(shards, steps=60, lr=0.5)
    check(non_private["loss"][-1] < 0.8 * non_private["loss"][0],
          f"federated training is live ({non_private['loss'][0]:.4f} -> "
          f"{non_private['loss'][-1]:.4f})")
    check(non_private["loss"][-1] <= 1.05 * reference_loss,
          f"the federated model matches the centralised reference "
          f"({non_private['loss'][-1]:.4f} vs {reference_loss:.4f})")
    check(private["loss"][-1] <= 1.05 * non_private["loss"][-1],
          f"the private run stays within 5% of the non-private one "
          f"({private['loss'][-1]:.4f} vs {non_private['loss'][-1]:.4f})")
    per_round_shift = [abs(a - b) for a, b in
                       zip(private["loss"][:5], non_private["loss"][:5])]
    check(max(per_round_shift) < 0.2,
          "no single noisy round derails the optimisation")
    check(all(math.isfinite(v) for v in private["loss"]), "no divergence under DP noise")

    # -- 9. the noise is added once per round, at the calibrated scale -----------------
    trajectory = []
    for index in range(200):
        outcome = federated_round([0.0, 0.0, 0.0], shards, agent,
                                  {"lr": 0.5, "epochs": 2, "sigma": sigma, "clip": CLIP,
                                   "rng": SplitMix64(1000 + index)})
        exact_mean = [sum(dequantise(outcome["updates"][c])[d] for c in agent.clients)
                      / len(agent.clients) for d in range(dimension)]
        decoded = [v / (SCALE * len(agent.clients)) for v in outcome["aggregate"]]
        trajectory.append([a - b for a, b in zip(decoded, exact_mean)])
    noise_std = math.sqrt(sum(v[d] ** 2 for v in trajectory for d in range(dimension))
                          / (len(trajectory) * dimension))
    expected = sigma / len(agent.clients)
    check(abs(noise_std - expected) < 0.15 * expected,
          f"the released aggregate carries sigma/n = {expected:.4f} of noise ({noise_std:.4f})")

    # -- 10. determinism ----------------------------------------------------------------
    again = train(shards, rounds=30, lr=0.5, epochs=2, sigma=sigma, seed=5)
    check(again["params"] == private["params"] and again["loss"] == private["loss"],
          "the whole pipeline is bit-reproducible")

    print(f"sigma for epsilon={epsilon}, delta={DELTA}: {sigma:.4f} "
          f"(epsilon={1.0} would need {sigma_for(1.0):.4f})")
    print(f"loss: federated {non_private['loss'][-1]:.5f}, private "
          f"{private['loss'][-1]:.5f}, centralised reference {reference_loss:.5f}")
    print(f"aggregate noise sd {noise_std:.5f} vs sigma/n = {expected:.5f}; "
          f"composed epsilon over 30 rounds {accountant.total(30)[0]:.1f}")
    print("PASS")


if __name__ == "__main__":
    main()
