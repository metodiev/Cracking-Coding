"""Task 785 - Implement collapsed Gibbs for LDA and perplexity evaluation.  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

Latent Dirichlet allocation describes a document as a mixture over K topics, with a
Dirichlet(alpha) document-topic distribution and, per topic, a Dirichlet(beta) word
distribution.  Collapsed Gibbs sampling integrates the multinomial parameters out in
closed form, so a single token's topic is resampled from

    p(z = k | rest)  proportional to  (n_dk + alpha) * (n_kw + beta) / (n_k + V * beta)

with n_dk the number of tokens of topic k in document d, n_kw the number of occurrences of
word w in topic k, n_k the total number of tokens of topic k and V the vocabulary size.
The token under consideration is removed from all three counts before the conditional is
formed (the "-1" of the collapsed conditional), which keeps every numerator positive and
the sampler well defined.

After S sweeps the count tables are the sufficient statistics and give the plug-in
estimates

    phi[k][w]   = (n_kw + beta)  / (n_k + V * beta)        topic -> word
    theta[d][k] = (n_dk + alpha) / (N_d + K * alpha)       document -> topic

Perplexity of a token stream is exp(-sum_tokens log sum_k theta[d][k] * phi[k][w] / #tokens).
For held-out documents theta is re-inferred by running the same conditional with phi
frozen (the standard "document completion" protocol), so held-out perplexity is never
fitted on its own counts.

Simplified versus a production topic model: one single-threaded chain, dense count tables,
no burn-in/restart diagnostics, and the predictive quantities use plug-in (MAP) parameters
instead of averaging over the full posterior.

What is verified: the count bookkeeping is rebuilt from the topic assignment after *every*
sweep and compared (exact invariant), phi/theta normalisation, that the sampler recovers a
synthetic two-topic corpus with disjoint vocabularies, that trained perplexity beats both
the uniform model (perplexity = V) and the unigram K = 1 limit, that the K = 1 perplexity
equals a hand-derived closed form, bit-identical reproducibility for a fixed seed, that the
chain's empirical state distribution matches the *exhaustively enumerated* collapsed
posterior on a 4-token corpus (an independent oracle for the sampler kernel), and the
documented rejection of invalid hyper-parameters.
"""
import itertools
import math

MASK64 = (1 << 64) - 1
TWO_POW_64 = 18446744073709551616.0


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class SplitMix64:
    """Deterministic 64-bit PRNG: same stream on every platform, no global state."""

    __slots__ = ("state",)

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        """Uniform in [0, 1)."""
        return self.next_u64() / TWO_POW_64


class LDAGibbs:
    """Collapsed Gibbs sampler for LDA plus the perplexity machinery."""

    def __init__(self, vocab_size, num_topics, alpha=0.1, beta=0.01, seed=20240785):
        if vocab_size < 1 or num_topics < 1:
            raise ValueError("vocab_size and num_topics must be positive")
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        self.V = vocab_size
        self.K = num_topics
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = SplitMix64(seed)
        self.docs = []
        self.z = []
        self.n_dk = []
        self.n_kw = []
        self.n_k = []
        self.sweeps = 0

    # ------------------------------------------------------------ construction
    def initialise(self, docs):
        """Load a corpus and seed the assignment deterministically (token i -> topic i % K)."""
        for doc in docs:
            for w in doc:
                if not 0 <= w < self.V:
                    raise ValueError("word id %r outside vocabulary" % (w,))
        self.docs = [list(doc) for doc in docs]
        self.n_dk = [[0] * self.K for _ in self.docs]
        self.n_kw = [[0] * self.V for _ in range(self.K)]
        self.n_k = [0] * self.K
        self.z = []
        for d, doc in enumerate(self.docs):
            row = []
            for i, w in enumerate(doc):
                k = i % self.K
                row.append(k)
                self.n_dk[d][k] += 1
                self.n_kw[k][w] += 1
                self.n_k[k] += 1
            self.z.append(row)
        self.sweeps = 0
        return self

    # ------------------------------------------------------------------ kernel
    def _conditional(self, d, w):
        """Unnormalised collapsed conditional for a token already removed from the counts."""
        alpha, beta, V = self.alpha, self.beta, self.V
        n_dk, n_kw, n_k = self.n_dk[d], self.n_kw, self.n_k
        return [(n_dk[k] + alpha) * (n_kw[k][w] + beta) / (n_k[k] + V * beta)
                for k in range(self.K)]

    def _draw(self, weights):
        total = math.fsum(weights)
        r = self.rng.uniform() * total
        acc = 0.0
        for index, p in enumerate(weights):
            acc += p
            if r < acc:
                return index
        return len(weights) - 1

    def sweep(self):
        """One pass over every token; the assignment is updated in place."""
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                k = self.z[d][i]
                self.n_dk[d][k] -= 1
                self.n_kw[k][w] -= 1
                self.n_k[k] -= 1
                new_k = self._draw(self._conditional(d, w))
                self.n_dk[d][new_k] += 1
                self.n_kw[new_k][w] += 1
                self.n_k[new_k] += 1
                self.z[d][i] = new_k
        self.sweeps += 1
        return self

    def fit(self, docs, sweeps):
        self.initialise(docs)
        for _ in range(sweeps):
            self.sweep()
        return self

    # -------------------------------------------------------------- invariants
    def counts_consistent(self):
        """Rebuild every count table from z; equality is the sampler's core invariant."""
        if sum(self.n_k) != sum(len(doc) for doc in self.docs):
            return False
        rebuilt_kw = [[0] * self.V for _ in range(self.K)]
        rebuilt_dk = [[0] * self.K for _ in self.docs]
        rebuilt_k = [0] * self.K
        for d, doc in enumerate(self.docs):
            if len(self.z[d]) != len(doc):
                return False
            for i, w in enumerate(doc):
                k = self.z[d][i]
                rebuilt_kw[k][w] += 1
                rebuilt_dk[d][k] += 1
                rebuilt_k[k] += 1
        return rebuilt_kw == self.n_kw and rebuilt_dk == self.n_dk and rebuilt_k == self.n_k

    def phi(self):
        """Topic -> word distributions, rows summing to one."""
        return [[(self.n_kw[k][w] + self.beta) / (self.n_k[k] + self.V * self.beta)
                 for w in range(self.V)] for k in range(self.K)]

    def theta(self, d):
        """Document -> topic distribution of training document d."""
        n_d = len(self.docs[d])
        return [(self.n_dk[d][k] + self.alpha) / (n_d + self.K * self.alpha)
                for k in range(self.K)]

    # -------------------------------------------------------------- perplexity
    @staticmethod
    def _token_log_prob(theta, phi, w):
        return math.log(math.fsum(theta[k] * phi[k][w] for k in range(len(theta))))

    def perplexity(self):
        """Perplexity of the training corpus under the plug-in parameters."""
        phi = self.phi()
        total_log = 0.0
        tokens = 0
        for d, doc in enumerate(self.docs):
            theta = self.theta(d)
            for w in doc:
                total_log += self._token_log_prob(theta, phi, w)
                tokens += 1
        return math.exp(-total_log / tokens) if tokens else float("inf")

    def infer_theta(self, doc, phi, sweeps=30):
        """Document completion: sample a held-out document's topics with phi frozen."""
        if not doc:
            return [1.0 / self.K] * self.K
        counts = [0] * self.K
        assign = [0] * len(doc)
        for i in range(len(doc)):
            assign[i] = i % self.K
            counts[assign[i]] += 1
        for _ in range(sweeps):
            for i, w in enumerate(doc):
                counts[assign[i]] -= 1
                pick = self._draw([(counts[kk] + self.alpha) * phi[kk][w]
                                   for kk in range(self.K)])
                counts[pick] += 1
                assign[i] = pick
        n_d = len(doc)
        return [(counts[k] + self.alpha) / (n_d + self.K * self.alpha) for k in range(self.K)]

    def heldout_perplexity(self, docs, sweeps=30):
        """Perplexity of unseen documents: theta re-inferred, phi never updated."""
        phi = self.phi()
        total_log = 0.0
        tokens = 0
        for doc in docs:
            theta = self.infer_theta(doc, phi, sweeps)
            for w in doc:
                total_log += self._token_log_prob(theta, phi, w)
                tokens += 1
        return math.exp(-total_log / tokens) if tokens else float("inf")


# --------------------------------------------------------------------- oracles
def log_collapsed_joint(state, docs, V, K, alpha, beta):
    """log p(z, w | alpha, beta) with the topic parameters integrated out."""
    n_kw = [[0] * V for _ in range(K)]
    n_dk = [[0] * K for _ in docs]
    n_k = [0] * K
    flat = 0
    for d, doc in enumerate(docs):
        for w in doc:
            k = state[flat]
            flat += 1
            n_kw[k][w] += 1
            n_dk[d][k] += 1
            n_k[k] += 1
    total = 0.0
    for d, doc in enumerate(docs):
        total += math.lgamma(K * alpha) - math.lgamma(len(doc) + K * alpha)
        for k in range(K):
            total += math.lgamma(n_dk[d][k] + alpha) - math.lgamma(alpha)
    for k in range(K):
        total += math.lgamma(V * beta) - math.lgamma(n_k[k] + V * beta)
        for w in range(V):
            total += math.lgamma(n_kw[k][w] + beta) - math.lgamma(beta)
    return total


def exact_posterior_over_states(docs, V, K, alpha, beta):
    """Exhaustive posterior p(z | w) for a tiny corpus (the sampler's reference)."""
    tokens = sum(len(doc) for doc in docs)
    logs = []
    states = []
    for state in itertools.product(range(K), repeat=tokens):
        states.append(state)
        logs.append(log_collapsed_joint(state, docs, V, K, alpha, beta))
    top = max(logs)
    weights = [math.exp(x - top) for x in logs]
    norm = math.fsum(weights)
    return {state: w / norm for state, w in zip(states, weights)}


def state_histogram(model, sweeps):
    """Empirical distribution of the chain's state, one sample per sweep."""
    counts = {}
    for _ in range(sweeps):
        model.sweep()
        flat = tuple(model.z[d][i] for d, doc in enumerate(model.docs) for i in range(len(doc)))
        counts[flat] = counts.get(flat, 0) + 1
    return {state: c / sweeps for state, c in counts.items()}


def make_corpus(groups, num_docs, doc_len, rng, purity=0.9):
    """Two-group synthetic corpus: each document has a topic; noise crosses groups."""
    docs = []
    doc_group = []
    for d in range(num_docs):
        topic = d % len(groups)
        doc_group.append(topic)
        doc = []
        for _ in range(doc_len):
            g = topic if rng.uniform() < purity else (topic + 1) % len(groups)
            words = groups[g]
            doc.append(words[rng.next_u64() % len(words)])
        docs.append(doc)
    return docs, doc_group


def group_mass(phi_row, groups):
    """Fraction of a topic's probability mass falling in each word group."""
    return [math.fsum(phi_row[w] for w in group) for group in groups]


def main():
    V, K, ALPHA, BETA = 24, 2, 0.1, 0.05
    groups = [list(range(12)), list(range(12, 24))]
    docs, _ = make_corpus(groups, 40, 20, SplitMix64(785001), purity=0.9)

    model = LDAGibbs(V, K, ALPHA, BETA, seed=785123)
    model.initialise(docs)
    check(model.counts_consistent(), "count tables must match the initial assignment")

    init_perplexity = model.perplexity()
    for step in range(150):
        model.sweep()
        check(model.counts_consistent(),
              "count invariant rebuilt from z must hold after every sweep (step %d)" % step)

    # --- normalisation of the plug-in estimates ---------------------------------
    phi = model.phi()
    check(all(abs(math.fsum(row) - 1.0) < 1e-12 for row in phi),
          "every topic->word row must be a normalised distribution")
    check(all(abs(math.fsum(model.theta(d)) - 1.0) < 1e-12 for d in range(len(docs))),
          "every document->topic row must be a normalised distribution")

    # --- topic recovery on the synthetic corpus ---------------------------------
    masses = [group_mass(row, groups) for row in phi]
    dominant = [0 if m[0] > m[1] else 1 for m in masses]
    check(dominant[0] != dominant[1],
          "the two topics must specialise on different word groups, got %r" % (dominant,))
    check(min(max(m) for m in masses) > 0.8,
          "each topic must concentrate > 0.8 of its mass on one group, got %r" % (masses,))

    # --- perplexity: beats the uniform model and the unigram limit --------------
    trained_perplexity = model.perplexity()
    unigram = LDAGibbs(V, 1, ALPHA, BETA, seed=785123).fit(docs, 150)
    unigram_perplexity = unigram.perplexity()
    check(trained_perplexity < unigram_perplexity,
          "K=2 perplexity %.4f must beat unigram perplexity %.4f"
          % (trained_perplexity, unigram_perplexity))
    check(trained_perplexity < 0.8 * V,
          "trained perplexity %.4f must stay well below V=%d" % (trained_perplexity, V))
    check(init_perplexity > trained_perplexity,
          "training must lower perplexity: init %.4f -> trained %.4f"
          % (init_perplexity, trained_perplexity))

    # --- K = 1 has a hand-derivable closed form ---------------------------------
    n_w = [0] * V
    for doc in docs:
        for w in doc:
            n_w[w] += 1
    total_tokens = sum(n_w)
    closed_form = math.exp(-math.fsum(n_w[w] * math.log((n_w[w] + BETA) / (total_tokens + V * BETA))
                                      for w in range(V)) / total_tokens)
    check(abs(unigram_perplexity - closed_form) < 1e-9,
          "K=1 perplexity %.12f must equal closed form %.12f" % (unigram_perplexity, closed_form))

    # --- held-out documents from the same generative process --------------------
    heldout, _ = make_corpus(groups, 8, 20, SplitMix64(785999), purity=0.9)
    heldout_ppl = model.heldout_perplexity(heldout)
    uniform_ppl = float(V)
    check(0.0 < heldout_ppl < uniform_ppl,
          "held-out perplexity %.4f must improve on the uniform baseline %.1f"
          % (heldout_ppl, uniform_ppl))
    theta_empty = model.infer_theta([], phi)
    check(all(abs(x - 1.0 / K) < 1e-12 for x in theta_empty),
          "an empty held-out document must fall back to the uniform topic prior")
    check(model.heldout_perplexity([], sweeps=5) == float("inf"),
          "perplexity of a token-free corpus must be reported as infinity")

    # --- reproducibility and negative checks ------------------------------------
    twin = LDAGibbs(V, K, ALPHA, BETA, seed=785123).fit(docs, 150)
    check(twin.z == model.z and twin.perplexity() == trained_perplexity,
          "an identical seed must reproduce the assignment and perplexity bit for bit")
    other = LDAGibbs(V, K, ALPHA, BETA, seed=4242).fit(docs, 150)
    check(other.z != model.z, "a different seed must produce a different chain")
    for bad in (lambda: LDAGibbs(V, K, alpha=0.0, beta=BETA),
                lambda: LDAGibbs(V, K, alpha=ALPHA, beta=-1.0),
                lambda: LDAGibbs(V, 0),
                lambda: LDAGibbs(V, K, ALPHA, BETA).initialise([[V]])):
        rejected = False
        try:
            bad()
        except ValueError:
            rejected = True
        check(rejected, "invalid hyper-parameters / out-of-vocabulary words must be rejected")

    # --- independent oracle: exhaustive collapsed posterior on 4 tokens ---------
    tiny_docs = [[0, 1], [1, 0]]
    tiny = LDAGibbs(2, 2, alpha=0.5, beta=0.5, seed=99).initialise(tiny_docs)
    reference = exact_posterior_over_states(tiny_docs, 2, 2, 0.5, 0.5)
    observed = state_histogram(tiny, 20000)
    distance = 0.5 * math.fsum(abs(observed.get(s, 0.0) - p) for s, p in reference.items())
    check(distance < 0.05,
          "Gibbs occupancy must match the exhaustive posterior, TV distance %.4f" % distance)
    check(abs(math.fsum(reference.values()) - 1.0) < 1e-12,
          "the enumerated posterior must be a probability distribution")

    print("corpus: V=%d K=%d docs=%d sweeps=%d" % (V, K, len(docs), model.sweeps))
    print("perplexity: init=%.4f trained=%.4f unigram=%.4f heldout=%.4f (uniform=%.1f)"
          % (init_perplexity, trained_perplexity, unigram_perplexity, heldout_ppl, uniform_ppl))
    print("topic group mass: %s" % ([[round(x, 4) for x in m] for m in masses],))
    print("posterior TV distance vs exhaustive enumeration: %.4f" % distance)
    print("PASS")


if __name__ == "__main__":
    main()
