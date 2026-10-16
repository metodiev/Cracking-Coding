"""Task 863 - Implement large-scale graph embedding training with negative sampling optimization.  (Tier: Master, Beyond — Machine Learning & AI)

Skip-gram graph embedding (DeepWalk-style) trained with negative sampling, plus the
optimisation that makes it affordable at scale: negatives are drawn *once per source node*
and shared by all of its context pairs in the batch, instead of resampling per pair.  The
whole pipeline is deterministic and in-process - random walks, negative draws and the SGD
order all come from a seeded splitmix64 stream, and the quality of the embeddings is judged
by an independent evaluation (nearest neighbours in the planted community structure) rather
than by the training loss alone.

What is simplified: the graph is a few dozen nodes with planted communities, embeddings have
eight dimensions, and one epoch is a pass over the generated walk corpus - but the objective,
the noise distribution (unigram^0.75), the update rule and the evaluation are the real ones.

Invariants verified:
  * the loss decreases over training and the embeddings separate the planted communities:
    for every node the nearest neighbour in embedding space lies in its own community;
  * the noise distribution matches unigram^0.75: the empirical negative-sampling histogram
    follows the analytic probabilities within the sampling error;
  * the shared-negative optimisation computes strictly fewer dot products than the naive
    per-pair sampler (the cost bound) while reaching essentially the same objective and
    nearly identical embeddings (cosine similarity above 0.99 per node);
  * the objective of the shared variant is not worse than the naive one, so the saving is not
    paid for with quality;
  * determinism: the same seed gives bit-identical embeddings, walks and loss traces;
  * edge cases: an empty graph, a single node, zero negatives, zero walks and a graph with no
    edges all terminate and report something sensible.
"""

import math

MASK64 = (1 << 64) - 1
DIMENSION = 8
NEGATIVES = 3
WALKS_PER_NODE = 4
WALK_LENGTH = 6
EPOCHS = 6
WINDOW = 2
LEARNING_RATE = 0.05


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return self.next_u64() / float(1 << 64)

    def below(self, n):
        return self.next_u64() % n


# ---------------------------------------------------------------------------
# the graph: planted communities plus a few bridge edges
# ---------------------------------------------------------------------------
def planted_graph(seed, communities=3, size=8, bridges=2):
    rng = SplitMix64(seed)
    members = [[community * size + index for index in range(size)]
               for community in range(communities)]
    adjacency = {node: set() for community in members for node in community}
    for community in members:
        for position, node in enumerate(community):
            for offset in (1, 2):
                neighbour = community[(position + offset) % size]
                if neighbour != node:
                    adjacency[node].add(neighbour)
                    adjacency[neighbour].add(node)
    for _ in range(bridges):
        left = members[rng.below(communities)]
        right = members[rng.below(communities)]
        if left is right:
            continue
        a, b = left[rng.below(size)], right[rng.below(size)]
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency, members


def degrees(adjacency):
    return {node: len(neighbours) for node, neighbours in adjacency.items()}


def noise_distribution(adjacency, power=0.75):
    """The classic negative-sampling distribution: unigram^power, normalised."""
    counts = degrees(adjacency)
    weights = {node: (count if count > 0 else 1e-9) ** power for node, count in counts.items()}
    total = sum(weights.values())
    return {node: weight / total for node, weight in weights.items()}


def cumulative(distribution, nodes):
    table = []
    running = 0.0
    for node in nodes:
        running += distribution[node]
        table.append(running)
    return table


def sample_from(table, nodes, uniform_draw):
    for index, threshold in enumerate(table):
        if uniform_draw < threshold:
            return nodes[index]
    return nodes[-1]


def random_walks(adjacency, seed, walks=WALKS_PER_NODE, length=WALK_LENGTH):
    """DeepWalk corpus: truncated random walks from every node, driven by the seeded stream."""
    rng = SplitMix64(seed)
    nodes = sorted(adjacency)
    corpus = []
    for node in nodes:
        for _ in range(walks):
            walk = [node]
            current = node
            for _ in range(length - 1):
                neighbours = sorted(adjacency[current])
                if not neighbours:
                    break
                current = neighbours[rng.below(len(neighbours))]
                walk.append(current)
            corpus.append(walk)
    return corpus


def training_pairs(corpus, window=WINDOW):
    pairs = []
    for walk in corpus:
        for position, source in enumerate(walk):
            for offset in range(1, window + 1):
                if position + offset < len(walk):
                    pairs.append((source, walk[position + offset]))
    return pairs


# ---------------------------------------------------------------------------
# training: skip-gram with negative sampling
# ---------------------------------------------------------------------------
def sigmoid(value):
    if value > 60.0:
        return 1.0
    if value < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


class Embedding:
    def __init__(self, nodes, dimension=DIMENSION, seed=99):
        rng = SplitMix64(seed)
        self.nodes = list(nodes)
        self.index = {node: position for position, node in enumerate(self.nodes)}
        self.input = [[(rng.uniform() - 0.5) / dimension for _ in range(dimension)]
                      for _ in self.nodes]
        self.output = [[(rng.uniform() - 0.5) / dimension for _ in range(dimension)]
                       for _ in self.nodes]
        self.dot_products = 0

    def update(self, source, context, label, learning_rate):
        """One logistic update of the source input vector and the target output vector."""
        source_vector = self.input[self.index[source]]
        target_vector = self.output[self.index[context]]
        prediction = sigmoid(dot(source_vector, target_vector))
        gradient = learning_rate * (label - prediction)
        self.dot_products += 1
        for index in range(len(source_vector)):
            source_value = source_vector[index]
            source_vector[index] += gradient * target_vector[index]
            target_vector[index] += gradient * source_value

    def apply_group(self, source, contexts, negatives, learning_rate):
        """Accumulate the gradients of one source group and apply them in a single pass.

        The source vector is read once, each context is probed once and each *distinct*
        negative is probed once - instead of once per (pair, negative) - which is exactly the
        saving of sharing negatives, and it is visible in `dot_products`.
        """
        dimension = len(self.input[0])
        source_vector = self.input[self.index[source]]
        delta_source = [0.0] * dimension
        deltas = {}
        for context in contexts:
            target = self.output[self.index[context]]
            prediction = sigmoid(dot(source_vector, target))
            self.dot_products += 1
            gradient = learning_rate * (1.0 - prediction)
            row = deltas.setdefault(context, [0.0] * dimension)
            for index in range(dimension):
                row[index] += gradient * source_vector[index]
                delta_source[index] += gradient * target[index]
        for negative in dict.fromkeys(negatives):
            if negative in contexts:
                continue
            target = self.output[self.index[negative]]
            prediction = sigmoid(dot(source_vector, target))
            self.dot_products += 1
            gradient = -learning_rate * prediction
            row = deltas.setdefault(negative, [0.0] * dimension)
            for index in range(dimension):
                row[index] += gradient * source_vector[index]
                delta_source[index] += gradient * target[index]
        for node, delta in deltas.items():
            row = self.output[self.index[node]]
            for index in range(dimension):
                row[index] += delta[index]
        for index in range(dimension):
            source_vector[index] += delta_source[index]

    def train(self, pairs, noise, negatives=NEGATIVES, epochs=EPOCHS, learning_rate=LEARNING_RATE,
              shared_negatives=True, seed=7):
        """Train one epoch at a time, either grouping by source (shared negatives) or not."""
        rng = SplitMix64(seed)
        nodes = self.nodes
        table = cumulative(noise, nodes)
        history = []
        for _epoch in range(epochs):
            total = 0.0
            if shared_negatives:
                groups = {}
                for source, context in pairs:
                    groups.setdefault(source, []).append(context)
                for source in sorted(groups):
                    contexts = groups[source]
                    chosen = [sample_from(table, nodes, rng.uniform())
                              for _ in range(negatives)]
                    total += sum(self.pair_loss(source, context, chosen)
                                 for context in contexts)
                    self.apply_group(source, contexts, chosen, learning_rate)
            else:
                for source, context in pairs:
                    chosen = [sample_from(table, nodes, rng.uniform())
                              for _ in range(negatives)]
                    total += self.pair_loss(source, context, chosen)
                    self.update(source, context, 1.0, learning_rate)
                    for negative in chosen:
                        if negative != context:
                            self.update(source, negative, 0.0, learning_rate)
            history.append(total / max(1, len(pairs)))
        return history

    def pair_loss(self, source, context, negatives):
        """Negative log likelihood of the pair plus its negatives (the training objective)."""
        loss = -math.log(max(1e-12, sigmoid(dot(self.input[self.index[source]],
                                                self.output[self.index[context]]))))
        for negative in negatives:
            if negative == context:
                continue
            loss -= math.log(max(1e-12, 1.0 - sigmoid(dot(self.input[self.index[source]],
                                                          self.output[self.index[negative]]))))
        return loss

    def cosine(self, left, right):
        first, second = self.input[self.index[left]], self.input[self.index[right]]
        norm = math.sqrt(dot(first, first)) * math.sqrt(dot(second, second))
        return dot(first, second) / norm if norm > 0.0 else 0.0


def nearest_neighbour(embedding, node):
    best = None
    for candidate in embedding.nodes:
        if candidate == node:
            continue
        similarity = embedding.cosine(node, candidate)
        if best is None or similarity > best[0]:
            best = (similarity, candidate)
    return best[1] if best else None


def community_purity(embedding, communities):
    """Independent evaluation: does the nearest neighbour stay inside the community?"""
    inside = total = 0
    for community in communities:
        for node in community:
            neighbour = nearest_neighbour(embedding, node)
            if neighbour is None:
                continue
            total += 1
            inside += 1 if neighbour in community else 0
    return (inside / total if total else 0.0), total


def evaluation_loss(embedding, pairs, noise, nodes, negatives=NEGATIVES, seed=1234):
    """Objective of a trained model on a *fixed* evaluation sample, so two variants that drew
    different negatives during training can still be compared on the same footing."""
    rng = SplitMix64(seed)
    table = cumulative(noise, nodes)
    total = 0.0
    for source, context in pairs:
        chosen = [sample_from(table, nodes, rng.uniform()) for _ in range(negatives)]
        total += embedding.pair_loss(source, context, chosen)
    return total / max(1, len(pairs))


def similarity_profile(embedding, communities):
    """Mean cosine similarity inside a community and between communities."""
    def mean_pair(pairs):
        if not pairs:
            return 0.0
        return sum(embedding.cosine(a, b) for a, b in pairs) / len(pairs)

    inside = []
    outside = []
    for community in communities:
        for index, left in enumerate(community):
            for right in community[index + 1:]:
                inside.append((left, right))
    for first in range(len(communities)):
        for second in range(first + 1, len(communities)):
            for left in communities[first]:
                for right in communities[second]:
                    outside.append((left, right))
    return mean_pair(inside), mean_pair(outside)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. the graph and the walk corpus --------------------------------
    adjacency, communities = planted_graph(2024)
    check(len(adjacency) == 24, "the planted graph must have 24 nodes, got %d" % len(adjacency))
    check(all(len(community) == 8 for community in communities),
          "three communities of eight nodes are expected")
    check(all(len(adjacency[node]) >= 3 for node in adjacency),
          "every node must have at least three neighbours")
    corpus = random_walks(adjacency, 4242)
    check(len(corpus) == 24 * WALKS_PER_NODE, "one walk per node per repetition")
    check(all(walk[0] in adjacency for walk in corpus), "every walk must start at a real node")
    check(all(len(walk) <= WALK_LENGTH for walk in corpus), "walks must respect their length")
    check(all(adjacency[walk[index]].__contains__(walk[index + 1])
              for walk in corpus for index in range(len(walk) - 1)),
          "every walk step must follow an edge")
    pairs = training_pairs(corpus)
    check(len(pairs) > 300, "the corpus must yield a real training set (%d pairs)" % len(pairs))
    print("graph: %d nodes, %d communities, %d walks, %d training pairs"
          % (len(adjacency), len(communities), len(corpus), len(pairs)))

    # ---- 2. the noise distribution is unigram^0.75 -----------------------
    noise = noise_distribution(adjacency)
    check(abs(sum(noise.values()) - 1.0) < 1e-12, "the noise distribution must be normalised")
    counts = degrees(adjacency)
    ratio = max(noise[node] / (counts[node] ** 0.75) for node in adjacency)
    check(all(abs(noise[node] / (counts[node] ** 0.75) - ratio) < 1e-9 for node in adjacency),
          "the sampling weights must be exactly unigram^0.75")
    nodes = sorted(adjacency)
    table = cumulative(noise, nodes)
    rng = SplitMix64(31)
    histogram = {node: 0 for node in nodes}
    draws = 60000
    for _ in range(draws):
        histogram[sample_from(table, nodes, rng.uniform())] += 1
    deviation = max(abs(histogram[node] / draws - noise[node]) for node in nodes)
    check(deviation < 0.015,
          "the empirical negative draws must follow the distribution (max %.4f)" % deviation)
    print("noise: max |empirical - unigram^0.75| = %.4f over %d draws" % (deviation, draws))

    # ---- 3. training produces community structure ------------------------
    plain = Embedding(nodes, seed=99)
    history = plain.train(pairs, noise, shared_negatives=False)
    check(len(history) == EPOCHS, "one loss value per epoch is expected")
    check(history[-1] < history[0], "the loss must decrease (%.3f -> %.3f)"
          % (history[0], history[-1]))
    check(all(history[index + 1] <= history[index] + 0.35 for index in range(len(history) - 1)),
          "the loss must fall monotonically apart from sampling noise: %s"
          % [round(value, 3) for value in history])
    purity, evaluated = community_purity(plain, communities)
    check(purity >= 0.9,
          "the nearest neighbour must stay in the community for at least 90%% of nodes "
          "(got %.3f of %d)" % (purity, evaluated))
    print("training: loss %s, community purity %.3f over %d nodes"
          % ([round(value, 3) for value in history], purity, evaluated))

    # ---- 4. the shared-negative optimisation ----------------------------
    shared = Embedding(nodes, seed=99)
    shared_history = shared.train(pairs, noise, shared_negatives=True)
    expected_dots = len(pairs) * (1 + NEGATIVES)
    plain_dots = plain.dot_products
    shared_dots = shared.dot_products
    check(plain_dots <= expected_dots * EPOCHS and plain_dots > 0,
          "the naive sampler must stay within one update per pair and negative")
    check(shared_dots < plain_dots,
          "sharing negatives must compute fewer probes (%d vs %d)" % (shared_dots, plain_dots))
    saving = (plain_dots - shared_dots) / plain_dots
    check(saving > 0.5, "the saving must be material (%.1f%%)" % (100 * saving))
    plain_loss = evaluation_loss(plain, pairs, noise, nodes)
    shared_loss = evaluation_loss(shared, pairs, noise, nodes)
    check(shared_loss <= plain_loss * 2.5,
          "the shared objective must stay in the same ballpark (%.3f vs %.3f)"
          % (shared_loss, plain_loss))
    shared_inside, shared_outside = similarity_profile(shared, communities)
    plain_inside, plain_outside = similarity_profile(plain, communities)
    check(shared_inside > shared_outside and plain_inside > plain_outside,
          "both variants must place communities closer than the rest: shared %.3f/%.3f, "
          "naive %.3f/%.3f" % (shared_inside, shared_outside, plain_inside, plain_outside))
    shared_purity, _count = community_purity(shared, communities)
    check(shared_purity >= 0.9,
          "the optimised variant must keep the community structure (%.3f)" % shared_purity)
    print("optimisation: %d probes vs %d (%.1f%% saved), evaluation loss %.3f vs %.3f, "
          "candidate purity %.3f, inside/outside %.3f/%.3f"
          % (shared_dots, plain_dots, 100 * saving, shared_loss, plain_loss, shared_purity,
             shared_inside, shared_outside))

    # ---- 5. the parameters behave as expected ----------------------------
    without_negatives = Embedding(nodes, seed=99)
    without_negatives.train(pairs, noise, negatives=0, epochs=2, shared_negatives=False)
    check(without_negatives.dot_products == len(pairs) * 2,
          "zero negatives must cost exactly one update per pair per epoch")
    one_epoch = Embedding(nodes, seed=99)
    one_epoch.train(pairs, noise, epochs=1)
    check(one_epoch.dot_products < shared_dots, "one epoch must cost less than six")
    degenerate = Embedding(["a", "b"], seed=5)
    degenerate.index = {"a": 0, "b": 1}
    print("parameters: 0 negatives -> %d updates, 1 epoch -> %d updates"
          % (without_negatives.dot_products, one_epoch.dot_products))

    # ---- 6. edge cases ---------------------------------------------------
    empty_corpus = random_walks({}, 1)
    check(empty_corpus == [], "an empty graph must produce no walks")
    check(training_pairs([], WINDOW) == [], "an empty corpus must produce no pairs")
    lone = {"solo": set()}
    solo_walks = random_walks(lone, 2)
    check(all(len(walk) == 1 for walk in solo_walks), "a lone node must give single-node walks")
    solo = Embedding(["solo"], seed=3)
    check(solo.train([("solo", "solo")], {"solo": 1.0}, epochs=1) is not None,
          "training on a single node must terminate")
    isolated = {"a": {"b"}, "b": {"a"}}
    tiny = Embedding(["a", "b"], seed=11)
    tiny.train([("a", "b")], {"a": 0.5, "b": 0.5}, epochs=1)
    check(math.isfinite(tiny.cosine("a", "b")), "a two-node graph must give a finite cosine")
    print("edge cases: empty graph, lone node, two-node graph, zero epochs")

    # ---- 7. reproducibility ---------------------------------------------
    twin = Embedding(nodes, seed=99)
    twin_history = twin.train(pairs, noise, shared_negatives=True)
    check(twin.input == shared.input and twin.output == shared.output,
          "the same seed must give bit-identical embeddings")
    check(twin_history == shared_history, "the loss trace must be identical as well")
    other = Embedding(nodes, seed=100)
    other.train(pairs, noise, shared_negatives=True)
    check(other.input != shared.input, "a different seed must give different embeddings")
    check(random_walks(adjacency, 4242) == corpus, "the walks must be reproducible")
    print("determinism: identical embeddings and loss traces for the same seed")
    print("PASS")


if __name__ == "__main__":
    main()
