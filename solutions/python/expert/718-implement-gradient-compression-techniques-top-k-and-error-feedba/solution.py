"""Task 718 - Implement gradient compression techniques (top-k) and error feedback.  (Tier: Expert, CTCI Beyond — Machine Learning & AI)

Sparsifying gradients: instead of all-reducing a dense gradient vector, a worker keeps only
the k entries of largest magnitude and ships those indices plus values.  Plain top-k throws
the discarded mass away every step, which permanently biases the update direction.  Error
feedback removes that bias by carrying a residual vector between steps:

    residual_t = grad_t + error_{t-1};  send_t = top_k(residual_t);  error_t = residual_t - send_t

so a coefficient that was dropped is delayed, never lost.  The invariant verified below is
the telescoping identity  sum_{t<=T} send_t + error_T == sum_{t<=T} grad_t  (componentwise):
the accumulated traffic plus the outstanding residual reconstructs the exact gradient mass
seen so far.  A second compressor samples k coordinates uniformly and rescales the kept
entries by dim/k, which is unbiased in expectation (also checked statistically).

Simplified versus production systems (DeepSpeed top-k, PowerSGD): the objective is a small
deterministic quadratic whose optimum is known in closed form through Sherman-Morrison (so
convergence is measured against ground truth, not guessed), training is plain SGD with no
momentum, sent values are not quantized, and there is one worker, so no all-reduce is
involved.  The compression itself is real: top-k selection, the error-feedback recurrence
and its conservation identity, the sparsity budget, unbiasedness of random-k, the residual
contraction needed when rescaling is combined with feedback, and the convergence benefit
over a no-feedback baseline.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    """A rejected invalid input is a pass: refuse it, do not silently adapt."""
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    """Deterministic 64-bit PRNG step (same generator used across the expert tier)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state, low, high):
    """Deterministic float in [low, high)."""
    state, r = splitmix64(state)
    return state, low + (high - low) * ((r >> 11) / float(1 << 53))


def top_k_indices(values, k):
    """Indices of the k largest magnitudes; ties broken by index so the result is unique."""
    order = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    return order[:k]


class Quadratic:
    """f(w) = 0.5 w^T A w - b^T w with A = diag(d) + u u^T (SPD, deterministic).

    Sherman-Morrison gives the exact minimiser, which the checks use as an independent
    oracle: grad(w*) must vanish and loss(w*) must be the smallest value reachable.
    """

    def __init__(self, dim, seed):
        self.dim = dim
        self.d = []
        self.u = []
        self.b = []
        for _ in range(dim):
            seed, d_i = uniform(seed, 0.6, 1.4)
            seed, u_i = uniform(seed, -0.35, 0.35)
            seed, b_i = uniform(seed, -1.0, 1.0)
            self.d.append(d_i)
            self.u.append(u_i)
            self.b.append(b_i)
        dinv_b = [self.b[i] / self.d[i] for i in range(dim)]
        dinv_u = [self.u[i] / self.d[i] for i in range(dim)]
        denom = 1.0 + sum(self.u[i] * dinv_u[i] for i in range(dim))
        scale = sum(self.u[i] * dinv_b[i] for i in range(dim)) / denom
        self.optimum = [dinv_b[i] - scale * dinv_u[i] for i in range(dim)]
        self.optimum_loss = self.loss(self.optimum)

    def gradient(self, w):
        uw = sum(self.u[i] * w[i] for i in range(self.dim))
        return [self.d[i] * w[i] + self.u[i] * uw - self.b[i] for i in range(self.dim)]

    def loss(self, w):
        quad = sum(self.d[i] * w[i] * w[i] for i in range(self.dim))
        uw = sum(self.u[i] * w[i] for i in range(self.dim))
        linear = sum(self.b[i] * w[i] for i in range(self.dim))
        return 0.5 * quad + 0.5 * uw * uw - linear


class TopKErrorFeedback:
    """top-k sparsifier with the error-feedback residual.

    Keeping the residual makes the accumulated sent vectors plus the outstanding error
    exactly equal to the accumulated raw gradients; disabling feedback makes that identity
    fail, which is precisely the bias error feedback is designed to remove.
    """

    def __init__(self, dim, k, use_error_feedback=True):
        if dim < 1:
            raise ValueError("gradient must have at least one coordinate")
        if k < 1:
            raise ValueError("k must be at least 1 (a zero budget sends nothing)")
        self.dim = dim
        self.k = min(k, dim)                      # a budget above dim is clamped, not an error
        self.use_error_feedback = use_error_feedback
        self.error = [0.0] * dim
        self.grad_total = [0.0] * dim
        self.sent_total = [0.0] * dim
        self.sent_nonzeros = 0
        self.steps = 0
        self.max_error = 0.0

    def compress(self, grad):
        """Return the dense vector that would go on the wire (k non-zeros)."""
        if self.use_error_feedback:
            residual = [grad[i] + self.error[i] for i in range(self.dim)]
        else:
            residual = list(grad)
        chosen = top_k_indices(residual, self.k)
        sent = [0.0] * self.dim
        for i in chosen:
            sent[i] = residual[i]
        if self.use_error_feedback:
            self.error = [residual[i] - sent[i] for i in range(self.dim)]
        for i in range(self.dim):
            self.grad_total[i] += grad[i]
            self.sent_total[i] += sent[i]
        self.steps += 1
        self.sent_nonzeros += len(chosen)
        self.max_error = max(self.max_error, max(abs(v) for v in self.error))
        return sent

    def conservation_gap(self):
        """max_i |sum(grad) - (sum(sent) + error)| - zero for perfect error feedback."""
        return max(abs(self.grad_total[i] - (self.sent_total[i] + self.error[i]))
                   for i in range(self.dim))


class RandomKErrorFeedback:
    """Unbiased random-k sparsifier: sample k coordinates and rescale them by dim/k.

    The rescaling makes E[sent] equal the residual, which the checks confirm statistically
    over a fixed seed.  `error_decay` scales the residual update: the textbook raw recursion
    (decay = 1) magnifies the residual of a kept coordinate by dim/k - 1 every round, so
    combining it with this estimator explodes; decay = k/dim contracts the residual and
    restores convergence - both behaviours are measured below.
    """

    def __init__(self, dim, k, seed, use_error_feedback=True):
        if not 1 <= k <= dim:
            raise ValueError("random-k needs 1 <= k <= dim")
        self.dim = dim
        self.k = k
        self.seed = seed & MASK64
        self.use_error_feedback = use_error_feedback
        self.error = [0.0] * dim
        self.scale = dim / k
        self.error_decay = 1.0
        self.last_indices = []
        self.grad_total = [0.0] * dim
        self.sent_total = [0.0] * dim

    def _sample(self):
        """Partial Fisher-Yates: k distinct coordinates, deterministic for a given seed."""
        index = list(range(self.dim))
        for pos in range(self.k):
            self.seed, r = splitmix64(self.seed)
            swap = pos + r % (self.dim - pos)
            index[pos], index[swap] = index[swap], index[pos]
        return index[:self.k]

    def compress(self, grad):
        if self.use_error_feedback:
            residual = [grad[i] + self.error[i] for i in range(self.dim)]
        else:
            residual = list(grad)
        sent = [0.0] * self.dim
        chosen = self._sample()
        for i in chosen:
            sent[i] = residual[i] * self.scale
        self.last_indices = chosen
        if self.use_error_feedback:
            self.error = [self.error_decay * (residual[i] - sent[i]) for i in range(self.dim)]
        for i in range(self.dim):
            self.grad_total[i] += grad[i]
            self.sent_total[i] += sent[i]
        return sent

    def conservation_gap(self):
        """Same telescoping identity as the top-k compressor, for the sampled variant."""
        return max(abs(self.grad_total[i] - (self.sent_total[i] + self.error[i]))
                   for i in range(self.dim))


def train(quad, compressor, steps, learning_rate):
    """Plain SGD driven by the compressor's output; returns the final weights and losses."""
    w = [0.0] * quad.dim
    losses = []
    for _ in range(steps):
        grad = quad.gradient(w)
        sent = compressor.compress(grad)
        w = [w[i] - learning_rate * sent[i] for i in range(quad.dim)]
        losses.append(quad.loss(w))
    return w, losses


def main():
    # 1. top-k selection itself: compare against a brute-force magnitude sort and check the
    #    tie-breaking rule keeps the result deterministic.
    seed, values = 0x7181, []
    for _ in range(40):
        seed, v = uniform(seed, -1.0, 1.0)
        values.append(round(v, 3))
    chosen = top_k_indices(values, 5)
    brute = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:5]
    check(sorted(chosen) == sorted(brute), "top-k must pick the k largest magnitudes")
    check(len(chosen) == 5 and len(set(chosen)) == 5, "top-k must return 5 distinct indices")
    smallest_kept = min(abs(values[i]) for i in chosen)
    largest_dropped = max(abs(values[i]) for i in range(len(values)) if i not in chosen)
    check(smallest_kept >= largest_dropped, "no dropped entry may outweigh a kept entry")
    ties = [2.0] * 6 + [1.0]
    check(top_k_indices(ties, 3) == [0, 1, 2], "ties must resolve to the lowest indices")

    # 2. the error-feedback invariant after many steps of a real training run, plus
    #    convergence measured against the closed-form optimum of the quadratic.
    dim, k, steps, lr = 64, 8, 300, 0.1
    quad = Quadratic(dim, 0x7182)
    check(math.dist(quad.gradient(quad.optimum), [0.0] * dim) < 1e-9, "the optimum has no gradient")
    feedback = TopKErrorFeedback(dim, k)
    ef_w, ef_losses = train(quad, feedback, steps, lr)
    check(feedback.steps == steps, "every step must be compressed")
    check(feedback.sent_nonzeros == steps * k, "each step must send exactly k values")
    check(feedback.conservation_gap() < 1e-9, "sum(sent) + error must rebuild sum(grad)")
    ef_gap = ef_losses[-1] - quad.optimum_loss
    ef_dist = math.dist(ef_w, quad.optimum)
    check(ef_gap < 1e-9, "error-feedback SGD must reach the quadratic's optimum")
    check(ef_dist < 1e-6, "the error-feedback iterates must converge to the optimum point")

    # 3. negative check: the same budget without error feedback loses gradient mass every
    #    step and stalls at a biased point far from the optimum.
    naive = TopKErrorFeedback(dim, k, use_error_feedback=False)
    naive_w, naive_losses = train(quad, naive, steps, lr)
    naive_gap = naive_losses[-1] - quad.optimum_loss
    naive_dist = math.dist(naive_w, quad.optimum)
    check(naive.conservation_gap() > 1e-3, "plain top-k loses gradient mass (biased baseline)")
    check(ef_gap * 1000.0 < naive_gap, "error feedback must win by orders of magnitude")
    check(ef_dist * 100.0 < naive_dist, "error feedback must land far closer to the optimum")
    check(ef_losses[-1] < naive_losses[-1], "the feedback run must end below the plain run")

    # 4. the outstanding residual stays bounded by the gradient scale (it is a delay of one
    #    compression round, not a growing leak).
    grad_scale = max(abs(v) for v in feedback.grad_total)
    check(feedback.max_error < 10.0 * grad_scale, "the residual must stay bounded")

    # 5. boundary cases: k == dim is loss-free, k above dim clamps, dim == 1 is exact, and an
    #    empty budget is rejected rather than silently doing nothing.
    exact = TopKErrorFeedback(9, 12)
    check(exact.k == 9, "a budget larger than the vector must clamp to the vector length")
    dense_grad = [0.5 * i - 2.0 for i in range(9)]
    sent = exact.compress(dense_grad)
    check(max(abs(sent[i] - dense_grad[i]) for i in range(9)) < 1e-12, "k == dim is loss-free")
    check(exact.conservation_gap() < 1e-12, "an exact compressor must not accumulate error")
    single = TopKErrorFeedback(1, 1)
    check(single.compress([3.5]) == [3.5], "a single coordinate must always be sent")
    check(single.error == [0.0], "a single coordinate leaves no residual behind")
    for bad in (0, -3): expect_value_error(lambda b=bad: TopKErrorFeedback(8, b), "k < 1 rejected")
    expect_value_error(lambda: TopKErrorFeedback(0, 1), "an empty gradient must be rejected")
    expect_value_error(lambda: RandomKErrorFeedback(8, 9, 1), "random-k needs k <= dim")

    # 6. zero gradient is a fixed point: nothing is sent and the residual stays untouched.
    frozen = TopKErrorFeedback(16, 4)
    check(frozen.compress([0.0] * 16) == [0.0] * 16, "a zero gradient sends nothing")
    check(frozen.error == [0.0] * 16, "a zero gradient leaves a zero residual")
    check(frozen.max_error == 0.0, "the residual norm must remain zero")

    # 7. statistical unbiasedness of random-k: the sample mean over many draws must track
    #    the residual it is estimating, and each draw must keep exactly k non-zeros.
    probe_dim = 50
    residual = [0.25 * i - 6.1 for i in range(probe_dim)]
    sampler = RandomKErrorFeedback(probe_dim, 5, 0x7183, use_error_feedback=False)
    total = [0.0] * probe_dim
    counts = [0] * probe_dim
    draws = 20000
    for _ in range(draws):
        sent = sampler.compress(residual)
        check(len(set(sampler.last_indices)) == 5, "random-k must draw 5 distinct coordinates")
        check(sum(1 for v in sent if v != 0.0) <= 5, "random-k must not touch other coordinates")
        for i in sampler.last_indices:
            counts[i] += 1
        for i in range(probe_dim):
            total[i] += sent[i] / draws
    expected = draws * 5 / probe_dim
    spread = max(abs(c - expected) for c in counts) / expected
    check(all(c > 0 for c in counts), "every coordinate must be sampled eventually")
    check(spread < 0.12, "the sampler must pick coordinates uniformly (counts must be even)")
    worst = max(abs(total[i] - residual[i]) / abs(residual[i])
                for i in range(probe_dim) if abs(residual[i]) >= 1.0)
    check(worst < 0.08, "the dim/k rescaling must make random-k unbiased up to noise")

    # 8. the sampled compressor: one round conserves mass exactly, the stream is
    #    deterministic, and the residual recursion must be contracted because the dim/k
    #    rescaling magnifies the residual of every kept coordinate by dim/k - 1.
    single = RandomKErrorFeedback(dim, k, 0x7185)
    probe_grad = quad.gradient([0.05 * i for i in range(dim)])
    single.compress(probe_grad)
    check(single.conservation_gap() < 1e-12, "one compression round must conserve mass")
    raw = RandomKErrorFeedback(dim, k, 0x7185)
    _, raw_losses = train(quad, raw, steps, 0.01)
    check(raw_losses[-1] - quad.optimum_loss > 1e30, "raw EF must explode when rescaled")
    contracted = RandomKErrorFeedback(dim, k, 0x7185)
    contracted.error_decay = k / dim
    contract_w, contract_losses = train(quad, contracted, steps, lr)
    check(contract_losses[-1] - quad.optimum_loss < 1e-9, "contracting the residual must fix it")
    check(math.dist(contract_w, quad.optimum) < 1e-6, "the contracted run must hit the optimum")
    rerun = RandomKErrorFeedback(dim, k, 0x7185)
    rerun.error_decay = k / dim
    rerun_w, _ = train(quad, rerun, steps, lr)
    check(rerun_w == contract_w, "the same seed and problem must reproduce the same weights")

    # 9. learning-rate limit of error feedback: the residual acts like momentum, so the
    #    same 2-entry budget converges at lr = 0.05 and diverges at lr = 0.2.
    _, gentle_losses = train(quad, TopKErrorFeedback(dim, 2), steps, 0.05)
    _, hard_losses = train(quad, TopKErrorFeedback(dim, 2), steps, 0.2)
    check(gentle_losses[-1] - quad.optimum_loss < 1e-6, "a small step makes a 2-entry budget work")
    check(hard_losses[-1] - quad.optimum_loss > 1.0, "a 4x larger step must diverge (EF momentum)")

    # 10. the error-feedback run really is sparse on the wire.
    nonzeros = sum(1 for v in feedback.sent_total if v != 0.0)
    check(1 <= nonzeros <= dim, "the accumulated traffic touches a subset of the coordinates")
    print(f"718 top-k dim={dim} k={k} gap ef={ef_gap:.3e} naive={naive_gap:.3e} "
          f"dist ef={ef_dist:.3e} naive={naive_dist:.3e}")
    print(f"718 random-k unbiased rel-error={worst:.3e} sampled-coord-count spread={spread:.1%}")
    print(f"718 random-k raw-EF gap={raw_losses[-1] - quad.optimum_loss:.1e} "
          f"contracted-EF gap={contract_losses[-1] - quad.optimum_loss:.1e}")
    print("PASS")


if __name__ == "__main__":
    main()
