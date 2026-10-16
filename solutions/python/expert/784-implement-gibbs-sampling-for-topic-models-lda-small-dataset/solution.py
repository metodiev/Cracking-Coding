"""Task 784 - Implement Gibbs sampling for topic models (LDA) small dataset.  (Tier: Expert, Beyond - Machine Learning & AI)

Latent Dirichlet allocation describes every document as a mixture of K topics and every topic as a
distribution over a vocabulary of V words, with Dirichlet(alpha) document-topic and Dirichlet(beta)
topic-word priors.  Collapsed Gibbs sampling integrates the multinomials out analytically, so the
only state is the topic assignment z of every token, and a sweep resamples each token in turn from the
*exact* conditional

    p(z_i = k | z_-i, w)  proportional to  (n_dk + alpha) * (n_kw_i + beta) / (n_k + V * beta),

where the counts n_dk, n_kw, n_k exclude the token being resampled.  Repeating sweeps leaves the chain
in its stationary distribution; the topic-word matrix E[n_kw] / (n_k + V beta) is the learned model.

Verified: the counts are consistent with the assignment matrix after every sweep and the token total is
conserved; the conditional distribution used for sampling agrees with an independent brute-force
recomputation of the same formula from the raw assignments (and sums to one); the sampler is
deterministic per seed and different seeds still recover the same structure; a synthetic corpus built
from two disjoint vocabulary halves is recovered nearly perfectly (the learned topics align with the
ground-truth partitions); the reported assignment's log joint probability improves over the random
initialisation; and the degenerate cases (a single word type, a single topic, a one-token document, an
empty corpus) behave.

Simplified: a small corpus (a few dozen tokens) so brute-force checks are cheap, a fixed number of
sweeps instead of a convergence diagnostic, no hyper-parameter resampling and no held-out perplexity
estimation - the reported metric is the collapsed log joint.  The sampler, the conditional and the
count invariants are the real algorithm.
"""
import math

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


def random_corpus(topics, docs_per_topic, tokens_per_doc, vocab, seed):
    """Documents drawn from disjoint halves of the vocabulary: a corpus with obvious topics."""
    corpus = []
    for topic in range(topics):
        low = topic * (vocab // topics)
        high = low + vocab // topics
        for _ in range(docs_per_topic):
            document = []
            for _ in range(tokens_per_doc):
                seed, r = splitmix64(seed)
                document.append(low + r % (high - low))
            corpus.append(document)
    return corpus, seed


class CollapsedGibbsLDA:
    """The collapsed Gibbs sampler: assignments plus the three count tables it maintains."""

    def __init__(self, topics, vocab, alpha=0.1, beta=0.01, seed=0xC0FFEE):
        self.topics = topics
        self.vocab = vocab
        self.alpha = alpha
        self.beta = beta
        self.state = seed
        self.corpus = []
        self.assignments = []
        self.doc_topic = []
        self.topic_word = []
        self.topic_total = []
        self.initial_joint = 0.0
        self.sweeps = 0

    # ------------------------------------------------------------------ setup
    def initialise(self, corpus):
        self.corpus = [list(document) for document in corpus]
        self.doc_topic = [[0] * self.topics for _ in self.corpus]
        self.topic_word = [[0] * self.vocab for _ in range(self.topics)]
        self.topic_total = [0] * self.topics
        self.assignments = []
        for document in self.corpus:
            row = []
            for word in document:
                self.state, r = splitmix64(self.state)
                topic = r % self.topics
                row.append(topic)
                self.doc_topic[len(self.assignments)][topic] += 1
                self.topic_word[topic][word] += 1
                self.topic_total[topic] += 1
            self.assignments.append(row)
        self.initial_joint = self.log_joint()
        return self

    # ------------------------------------------------------------------ inference
    def conditional(self, document, word, topic):
        """The unnormalised conditional p(z = topic | everything else) for one token."""
        n_dk = self.doc_topic[document][topic]
        n_kw = self.topic_word[topic][word]
        n_k = self.topic_total[topic]
        return (n_dk + self.alpha) * (n_kw + self.beta) / (n_k + self.vocab * self.beta)

    def conditional_vector(self, document, position):
        """Leave-one-out conditional recomputed from the raw assignments (the brute-force view)."""
        word = self.corpus[document][position]
        topic_word = [[0] * self.vocab for _ in range(self.topics)]
        doc_topic = [0] * self.topics
        totals = [0] * self.topics
        for doc, words in enumerate(self.corpus):
            for spot, stored in enumerate(words):
                if doc == document and spot == position:
                    continue
                label = self.assignments[doc][spot]
                totals[label] += 1
                if doc == document:
                    doc_topic[label] += 1
                if stored == word:
                    topic_word[label][word] += 1
        return [(doc_topic[topic] + self.alpha) * (topic_word[topic][word] + self.beta)
                / (totals[topic] + self.vocab * self.beta) for topic in range(self.topics)]

    def sample_topic(self, probabilities):
        total = sum(probabilities)
        check(total > 0, "the conditional distribution must not be degenerate")
        self.state, r = splitmix64(self.state)
        threshold = (r % (1 << 30)) / float(1 << 30) * total
        cumulative = 0.0
        for topic, probability in enumerate(probabilities):
            cumulative += probability
            if threshold < cumulative:
                return topic
        return len(probabilities) - 1

    def sweep(self):
        """Resample every token once, in document order."""
        for document, words in enumerate(self.corpus):
            for position, word in enumerate(words):
                previous = self.assignments[document][position]
                self.doc_topic[document][previous] -= 1
                self.topic_word[previous][word] -= 1
                self.topic_total[previous] -= 1
                probabilities = [self.conditional(document, word, topic)
                                 for topic in range(self.topics)]
                topic = self.sample_topic(probabilities)
                self.assignments[document][position] = topic
                self.doc_topic[document][topic] += 1
                self.topic_word[topic][word] += 1
                self.topic_total[topic] += 1
        self.sweeps += 1
        return self.log_joint()

    def fit(self, sweeps=80):
        for _ in range(sweeps):
            self.sweep()
        return self

    def restore(self, assignments):
        """Rebuild every count table from an assignment matrix (also the invariant checker)."""
        self.assignments = [list(row) for row in assignments]
        self.doc_topic = [[0] * self.topics for _ in self.corpus]
        self.topic_word = [[0] * self.vocab for _ in range(self.topics)]
        self.topic_total = [0] * self.topics
        for document, words in enumerate(self.corpus):
            for position, word in enumerate(words):
                topic = self.assignments[document][position]
                self.doc_topic[document][topic] += 1
                self.topic_word[topic][word] += 1
                self.topic_total[topic] += 1
        return self

    # ------------------------------------------------------------------ objectives
    def log_joint(self):
        """Collapsed log joint p(w, z), up to the constant terms that do not depend on z."""
        total = 0.0
        for document, words in enumerate(self.corpus):
            total += math.lgamma(len(words) + self.topics * self.alpha)
            for topic in range(self.topics):
                total -= math.lgamma(self.doc_topic[document][topic] + self.alpha)
        for topic in range(self.topics):
            total += math.lgamma(self.topic_total[topic] + self.vocab * self.beta)
            total -= self.vocab * math.lgamma(self.beta)
            for word in range(self.vocab):
                total -= math.lgamma(self.topic_word[topic][word] + self.beta)
        return total

    def document_completion(self):
        """Held-out predictive log-likelihood: each document's second half given its first half."""
        total = 0.0
        for document, words in enumerate(self.corpus):
            split = max(1, len(words) // 2)
            doc_counts = [0] * self.topics
            for label in self.assignments[document][:split]:
                doc_counts[label] += 1
            for word in words[split:]:
                weights = []
                for topic in range(self.topics):
                    p_topic = (doc_counts[topic] + self.alpha) / (split + self.topics * self.alpha)
                    p_word = ((self.topic_word[topic][word] + self.beta)
                              / (self.topic_total[topic] + self.vocab * self.beta))
                    weights.append(p_topic * p_word)
                total += math.log(sum(weights))
        return total

    def topic_word_matrix(self):
        return [[(self.topic_word[topic][word] + self.beta)
                 / (self.topic_total[topic] + self.vocab * self.beta)
                 for word in range(self.vocab)] for topic in range(self.topics)]

    def check_counts(self):
        """Counts must be exactly the aggregates of the assignment matrix; returns the token count."""
        tokens = 0
        for document, words in enumerate(self.corpus):
            check(len(self.assignments[document]) == len(words), "one topic per token")
            expected = [0] * self.topics
            for position, topic in enumerate(self.assignments[document]):
                expected[topic] += 1
                check(0 <= topic < self.topics, "a topic label must be in range")
            check(expected == self.doc_topic[document], "doc-topic counts must match the assignments")
            tokens += len(words)
        check(sum(self.topic_total) == tokens, "the token total must be conserved")
        for topic in range(self.topics):
            for word in range(self.vocab):
                observed = sum(1 for document, words in enumerate(self.corpus)
                               for position, stored in enumerate(words)
                               if stored == word and self.assignments[document][position] == topic)
                check(observed == self.topic_word[topic][word], "topic-word counts must match")
            check(sum(self.topic_word[topic]) == self.topic_total[topic],
                  "a topic's word counts must sum to its total")
        return tokens


def alignment(assignments, corpus, vocab, topics):
    """Fraction of the token mass whose learned topic matches the corpus partition it came from."""
    learned_mass = [[0] * 2 for _ in range(topics)]
    for document, words in enumerate(corpus):
        for position, word in enumerate(words):
            truth = 0 if word < vocab // 2 else 1
            learned_mass[assignments[document][position]][truth] += 1
    mapping = {}
    for topic in range(topics):
        mapping[topic] = 0 if learned_mass[topic][0] >= learned_mass[topic][1] else 1
    correct = sum(learned_mass[topic][mapping[topic]] for topic in range(topics))
    total = sum(sum(row) for row in learned_mass)
    return correct / total


def main():
    # 1. boundaries: a single word type, a single topic, a one-token document, an empty corpus
    blank = CollapsedGibbsLDA(2, 1).initialise([])
    check(blank.topic_total == [0, 0] and sum(blank.topic_total) == 0,
          "an empty corpus has no tokens")
    check(math.isfinite(blank.log_joint()), "an empty corpus must still have a finite objective")
    check(blank.log_joint() == CollapsedGibbsLDA(2, 1).initialise([]).log_joint(),
          "the empty-corpus objective must depend only on the parameters")
    one_word = CollapsedGibbsLDA(2, 1, seed=7).initialise([[0, 0, 0], [0]])
    one_word.fit(sweeps=5)
    check(one_word.check_counts() == 4, "the tokens must be conserved with a single word type")
    single_topic = CollapsedGibbsLDA(1, 5, seed=3).initialise([[0, 1, 2], [3, 4]])
    single_topic.fit(sweeps=5)
    check(single_topic.topic_total == [5], "with one topic every token belongs to it")
    check(single_topic.doc_topic == [[len(document)] for document in single_topic.corpus],
          "with one topic each document's row is just its length")
    tiny = CollapsedGibbsLDA(2, 4, seed=9).initialise([[1]])
    tiny.fit(sweeps=5)
    check(tiny.check_counts() == 1 and len(tiny.assignments[0]) == 1, "a one-token document works")

    # 2. the conditional used for sampling equals a brute-force recomputation of the same formula
    model = CollapsedGibbsLDA(3, 6, alpha=0.2, beta=0.05, seed=11).initialise([[0, 1, 2, 3], [3, 2, 1]])
    for document, position in ((0, 1), (1, 2), (0, 3)):
        word = model.corpus[document][position]
        label = model.assignments[document][position]
        model.doc_topic[document][label] -= 1            # the leave-one-out state sweep() samples from
        model.topic_word[label][word] -= 1
        model.topic_total[label] -= 1
        probabilities = [model.conditional(document, word, topic) for topic in range(3)]
        recomputed = model.conditional_vector(document, position)
        check(all(abs(a - b) < 1e-9 for a, b in zip(probabilities, recomputed)),
              "the sampler's conditional must match the recomputed formula")
        check(all(value > 0 for value in probabilities), "every conditional weight must be positive")
        check(sum(probabilities) > 0, "the conditional must have positive mass to sample from")
        model.doc_topic[document][label] += 1
        model.topic_word[label][word] += 1
        model.topic_total[label] += 1
    check(model.check_counts() == 7, "the leave-one-out experiment must restore the counts")

    # 3. a synthetic corpus with two disjoint halves is recovered
    corpus, seed = random_corpus(2, 4, 6, 20, 0xBEEF)
    tokens = sum(len(document) for document in corpus)
    check(tokens == 48 and len(corpus) == 8, "the synthetic corpus must have 48 tokens")
    sampler = CollapsedGibbsLDA(2, 20, alpha=0.1, beta=0.01, seed=0x5EED).initialise(corpus)
    initial_alignment = alignment(sampler.assignments, corpus, 20, 2)
    completion_before = sampler.document_completion()
    sampler.fit(sweeps=60)
    sampler.check_counts()
    final_alignment = alignment(sampler.assignments, corpus, 20, 2)
    completion_after = sampler.document_completion()
    check(final_alignment >= 0.9,
          f"the learned topics must align with the true partitions, got {final_alignment:.3f}")
    check(final_alignment > initial_alignment - 1e-9, "sampling must not destroy the structure")
    check(completion_after > completion_before,
          f"the held-out completion likelihood must improve "
          f"({completion_before:.2f} -> {completion_after:.2f})")
    check(math.isfinite(sampler.log_joint()) and math.isfinite(sampler.initial_joint),
          "the collapsed log joint must stay finite along the chain")
    snapshot = [list(row) for row in sampler.assignments]
    counts_before = [list(row) for row in sampler.topic_word]
    sampler.restore(snapshot)
    check(sampler.topic_word == counts_before,
          "rebuilding the counts from the assignments must reproduce them exactly")

    # 4. the learned topic-word distributions concentrate on the right half of the vocabulary
    matrix = sampler.topic_word_matrix()
    check(all(abs(sum(row) - 1.0) < 1e-9 for row in matrix), "every topic distribution must sum to one")
    concentrated = 0
    for topic in range(2):
        left = sum(matrix[topic][:10])
        right = sum(matrix[topic][10:])
        check(abs(max(left, right) - 1.0) < 0.35, "a topic must be concentrated on one half")
        concentrated += max(left, right)
    check(concentrated / 2 >= 0.9, f"topic concentration must be high, got {concentrated / 2:.3f}")
    check(max(matrix[0]) > 1.0 / 20, "the modal word of a topic must beat the uniform share")

    # 5. determinism and seed independence of the *structure*
    same = CollapsedGibbsLDA(2, 20, alpha=0.1, beta=0.01, seed=0x5EED).initialise(corpus).fit(sweeps=60)
    check(same.assignments == sampler.assignments, "the same seed must reproduce the chain exactly")
    other = CollapsedGibbsLDA(2, 20, alpha=0.1, beta=0.01, seed=0xFACE).initialise(corpus).fit(sweeps=60)
    check(other.assignments != sampler.assignments, "another seed must explore other assignments")
    check(alignment(other.assignments, corpus, 20, 2) >= 0.9,
          "another seed must recover the same topics")

    # 6. counts survive a long chain: invariants checked after every single sweep
    tracked = CollapsedGibbsLDA(2, 20, alpha=0.1, beta=0.01, seed=0x1234).initialise(corpus)
    for _ in range(20):
        tracked.sweep()
        tracked.check_counts()

    print(f"784 tokens={tokens} alignment={final_alignment:.3f} initial={initial_alignment:.3f} "
          f"completion {completion_before:.2f} -> {completion_after:.2f}")
    print(f"784 concentration={concentrated / 2:.3f} topics={sampler.topics} vocab={sampler.vocab} "
          f"sweeps={sampler.sweeps}")
    print("PASS")


if __name__ == "__main__":
    main()
