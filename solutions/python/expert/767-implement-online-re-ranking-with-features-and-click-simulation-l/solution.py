"""Task 767 - Online re-ranking with features and click simulation (learning to rank).  (Tier: Expert, Beyond - Machine Learning & AI)

Learning to rank from clicks has to deal with the fact that clicks are *biased*: a user only clicks
what they examine, and examination falls off with the rank position.  This file builds the whole
small pipeline and measures the effects instead of asserting them:

  * a synthetic corpus of queries whose documents have a latent semantic score and an intrinsic
    quality; the *true* relevance is a fixed linear combination of the two, while the retrieval
    feature the baseline sees (a noisy proxy, e.g. a BM25 score) is only correlated with it;
  * a baseline ranker (sort by the proxy) and a learned linear ranker trained on clicks with the
    pairwise logistic (RankNet) loss;
  * a position-based click model: the document at position p is examined with probability eta^(p-1)
    and clicked with the attractiveness sigmoid(4*(relevance - 0.8)), so the observed click rate is
    examination x attractiveness;
  * inverse propensity scoring (IPS): weighting each observed click by 1/examination(p) makes the
    click-based utility estimate unbiased, which is checked against the ground-truth average
    attractiveness of the ranked lists - the naive click rate is biased low, the IPS estimate is not.

Simplified versus a real ranking system: two features instead of hundreds, a linear model instead of
a gradient-boosted forest, a position-based model with a single global eta instead of a cascade or a
factorised examination model, no exploration (no epsilon-greedy or probabilistic interleaving), no
counterfactual model training beyond the IPS weighting, and a few hundred simulated sessions.  What
is genuine: the unbiasedness argument, the loss, the position bias, and the comparison of the learned
ranker against the baseline in nDCG and in simulated clicks - all measured, none assumed.
"""
import math

MASK64 = (1 << 64) - 1
QUERIES = 40
DOCS_PER_QUERY = 10
ETA = 0.8                       # examination probability falls off this fast with the position
TRUE_WEIGHTS = (1.0, 0.6)       # the latent relevance = 1.0 * semantic + 0.6 * quality


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
    return state, r / float(1 << 64)


def sigmoid(value):
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def grade_of(relevance):
    if relevance >= 1.20:
        return 4
    if relevance >= 0.95:
        return 3
    if relevance >= 0.70:
        return 2
    if relevance >= 0.45:
        return 1
    return 0


def build_dataset(seed=0x5EED):
    """query -> [(features, relevance, grade)]; features = (proxy, quality), relevance latents."""
    dataset = []
    state = seed & MASK64
    for _query in range(QUERIES):
        documents = []
        for _document in range(DOCS_PER_QUERY):
            state, semantic = uniform(state)
            state, quality = uniform(state)
            state, noise = uniform(state)
            relevance = TRUE_WEIGHTS[0] * semantic + TRUE_WEIGHTS[1] * quality
            proxy = semantic + (noise - 0.5) * 0.5          # the retrieval feature is noisy
            documents.append(((proxy, quality), relevance, grade_of(relevance)))
        dataset.append(documents)
    return dataset, state


def attractiveness(relevance):
    return sigmoid(4.0 * (relevance - 0.8))


def examination(position):
    return ETA ** (position - 1)


def simulate_session(ranking, documents, state, record=None):
    """Position-based click model; returns (clicks, state)."""
    clicks = []
    for position, index in enumerate(ranking, start=1):
        document = documents[index]
        state, draw = uniform(state)
        clicked = draw < examination(position) * attractiveness(document[1])
        if clicked:
            clicks.append(index)
            if record is not None:
                record.append((index, position))
    return clicks, state


def rank_by(documents, key):
    order = sorted(range(len(documents)), key=lambda index: (-key(documents[index]), index))
    return order


def baseline_key(document):
    return document[0][0]


def learned_key(weights):
    def key(document):
        return weights[0] * document[0][0] + weights[1] * document[0][1]
    return key


def build_pairs(dataset, state, weight_ips=True):
    """Click-derived preference pairs: every clicked document beats the unclicked ones."""
    pairs = []
    for documents in dataset:
        ranking = rank_by(documents, baseline_key)
        clicks, state = simulate_session(ranking, documents, state, record=None)
        clicked = set(clicks)
        for index in clicked:
            position = ranking.index(index) + 1
            weight = (1.0 / examination(position)) if weight_ips else 1.0
            for other in ranking:
                if other not in clicked:
                    pairs.append((documents[index][0], documents[other][0], weight))
    return pairs, state


def train(pairs, iterations=400, rate=0.8, l2=0.001, weights=(0.0, 0.0)):
    """Pairwise logistic (RankNet) loss with gradient descent; returns (weights, loss history)."""
    weights = list(weights)
    history = []
    for _iteration in range(iterations):
        gradient = [0.0, 0.0]
        loss = 0.0
        for positive, negative, weight in pairs:
            margin = sum(weights[k] * (positive[k] - negative[k]) for k in range(len(weights)))
            loss += weight * math.log(1.0 + math.exp(-max(-60.0, min(60.0, margin))))
            factor = -weight * sigmoid(-margin)
            for k in range(len(weights)):
                gradient[k] += factor * (positive[k] - negative[k])
        for k in range(len(weights)):
            weights[k] -= rate * (gradient[k] / max(1, len(pairs)) + l2 * weights[k])
        history.append(loss / max(1, len(pairs)))
    return weights, history


def discounted_gain(gains):
    return sum(gain / math.log(position + 1, 2) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(ranking, documents, k=10):
    if k <= 0:
        return 0.0
    gains = [(2 ** documents[index][2]) - 1 for index in ranking[:k]]
    ideal = sorted(((2 ** documents[index][2]) - 1 for index in range(len(documents))), reverse=True)[:k]
    if not sum(ideal):
        return 0.0
    return discounted_gain(gains) / discounted_gain(ideal)


def evaluate(dataset, key, k=10):
    return sum(ndcg_at_k(rank_by(documents, key), documents, k) for documents in dataset) / len(dataset)


def simulate_clicks(dataset, key, sessions=3, seed=0xC0FFEE, record=None):
    state = seed & MASK64
    clicks = 0
    impressions = 0
    for _session in range(sessions):
        for documents in dataset:
            ranking = rank_by(documents, key)
            found, state = simulate_session(ranking, documents, state, record)
            clicks += len(found)
            impressions += len(documents)
    return clicks, impressions


def naive_utility(dataset, key, sessions=3, seed=0xC0FFEE):
    clicks, impressions = simulate_clicks(dataset, key, sessions, seed)
    return clicks / float(impressions)


def ips_utility(dataset, key, sessions=3, seed=0xC0FFEE):
    """Unbiased estimate of the average attractiveness: clicks divided by their propensity."""
    state = seed & MASK64
    total = 0.0
    count = 0
    for _session in range(sessions):
        for documents in dataset:
            ranking = rank_by(documents, key)
            for position, index in enumerate(ranking, start=1):
                state, draw = uniform(state)
                if draw < examination(position) * attractiveness(documents[index][1]):
                    total += 1.0 / examination(position)
                count += 1
    return total / count


def truth_utility(dataset, key, sessions=3):
    """Ground truth: the average attractiveness of the ranked lists (what IPS must recover)."""
    total = 0.0
    count = 0
    for _session in range(sessions):
        for documents in dataset:
            for index in rank_by(documents, key):
                total += attractiveness(documents[index][1])
                count += 1
    return total / count


def expected_clicks(dataset, key, sessions=3):
    """Analytic click expectation under the position-based model: examination x attractiveness."""
    total = 0.0
    for _session in range(sessions):
        for documents in dataset:
            for position, index in enumerate(rank_by(documents, key), start=1):
                total += examination(position) * attractiveness(documents[index][1])
    return total


def main():
    dataset, _state = build_dataset()
    check(len(dataset) == QUERIES and all(len(d) == DOCS_PER_QUERY for d in dataset),
          "the dataset must have %d queries of %d documents" % (QUERIES, DOCS_PER_QUERY))
    check(all(0 <= document[2] <= 4 for documents in dataset for document in documents),
          "grades must be within 0..4")
    check(len({tuple(document[0] for document in documents) for documents in dataset}) == QUERIES,
          "every query must have its own document set")
    flat = [(document[0][0], document[1]) for documents in dataset for document in documents]
    mean_proxy = sum(proxy for proxy, _rel in flat) / len(flat)
    mean_relevance = sum(relevance for _proxy, relevance in flat) / len(flat)
    covariance = sum((proxy - mean_proxy) * (relevance - mean_relevance) for proxy, relevance in flat)
    check(covariance > 0, "the baseline feature must be positively correlated with the truth")
    disagreements = sum(1 for documents in dataset
                        if rank_by(documents, baseline_key) != rank_by(documents, lambda d: d[1]))
    check(disagreements > 0.5 * QUERIES,
          "the proxy must disagree with the truth on most queries: %d/%d" % (disagreements, QUERIES))

    # 1. the baseline: ordering by the proxy, and the resulting grade distribution
    baseline_ndcg = evaluate(dataset, baseline_key)
    check(0.0 < baseline_ndcg <= 1.0, "nDCG must be in (0, 1]: %.3f" % baseline_ndcg)
    grades_by_position = []
    for documents in dataset:
        ranking = rank_by(documents, baseline_key)
        grades_by_position.append([documents[index][2] for index in ranking])
    first = sum(row[0] for row in grades_by_position) / float(QUERIES)
    last = sum(row[-1] for row in grades_by_position) / float(QUERIES)
    check(first > last, "the baseline must put better documents first: %.2f vs %.2f" % (first, last))

    # 2. the click model: position bias, and attractiveness driving clicks at a fixed position
    top_grades = [documents[0][2] for documents in dataset]
    click_rate_by_position = []
    state = 0x1234
    for position in range(1, DOCS_PER_QUERY + 1):
        clicks = 0
        trials = 0
        for documents in dataset:
            state, draw = uniform(state)
            clicks += 1 if draw < examination(position) * attractiveness(documents[0][1]) else 0
            trials += 1
        click_rate_by_position.append(clicks / float(trials))
    check(click_rate_by_position[0] > click_rate_by_position[4] > click_rate_by_position[9],
          "the examination fall-off must show up as a decreasing click rate: %r"
          % ([round(rate, 3) for rate in click_rate_by_position],))
    check(click_rate_by_position[0] > 0.2, "the most attractive document must be clicked often")
    high = sum(1 for documents in dataset if documents[0][1] > 1.0)
    check(high > 0, "some of the top documents must be highly relevant: %d" % high)

    # 3. training on IPS-weighted clicks: the loss decreases and the weights keep the true signs
    pairs, state = build_pairs(dataset, 0xBEEF, weight_ips=True)
    check(len(pairs) > 100, "the click simulation must produce training pairs: %d" % len(pairs))
    weights, history = train(pairs)
    check(history[-1] < history[0] * 0.85,
          "the pairwise loss must drop clearly: %.4f -> %.4f" % (history[0], history[-1]))
    check(all(history[i] >= history[i + 1] - 1e-9 for i in range(len(history) - 1)),
          "the logistic loss is convex, so the training curve must be monotone")
    check(weights[0] > 0 and weights[1] > 0, "both features must be ranked positively: %r" % (weights,))
    ratio = weights[0] / weights[1]
    check(ratio > 1.0, "the relevance feature must dominate quality (true ratio %.1f): %.2f"
          % (TRUE_WEIGHTS[0] / TRUE_WEIGHTS[1], ratio))

    # 4. the learned ranker beats the baseline offline
    learned_ndcg = evaluate(dataset, learned_key(weights))
    check(learned_ndcg > baseline_ndcg,
          "the learned ranker must improve nDCG@10: %.4f vs %.4f" % (learned_ndcg, baseline_ndcg))
    check(learned_ndcg >= 0.95 * evaluate(dataset, learned_key(TRUE_WEIGHTS)),
          "the learned weights must be close to the optimal ranking")
    check(ndcg_at_k([], dataset[0], 10) == 0.0 and ndcg_at_k([0], dataset[0], 0) == 0.0,
          "nDCG must be zero for an empty ranking or k = 0")

    # 5. the same improvement measured in clicks: analytically (deterministic) and in simulation
    expected_baseline = expected_clicks(dataset, baseline_key)
    expected_learned = expected_clicks(dataset, learned_key(weights))
    check(expected_learned > expected_baseline,
          "ranking attractive documents higher must raise the expected clicks: %.1f vs %.1f"
          % (expected_learned, expected_baseline))
    baseline_clicks, impressions = simulate_clicks(dataset, baseline_key)
    learned_clicks, _ = simulate_clicks(dataset, learned_key(weights))
    check(abs(learned_clicks - expected_learned) < 0.1 * expected_learned,
          "the simulation must track its expectation: %d vs %.1f" % (learned_clicks, expected_learned))
    check(learned_clicks > 0 and baseline_clicks > 0, "the simulation must produce clicks")
    check(impressions == QUERIES * DOCS_PER_QUERY * 3, "every document is shown once per session")
    check(abs(learned_clicks - baseline_clicks) <= 0.15 * impressions,
          "the simulated gap must stay within sampling noise: %d vs %d" % (learned_clicks,
                                                                           baseline_clicks))

    # 6. IPS corrects the position bias: the naive click rate is biased low, the IPS estimate is not
    keys = (baseline_key, learned_key(weights))
    truth = sum(truth_utility(dataset, key) for key in keys) / len(keys)
    naive = sum(naive_utility(dataset, key, seed=1000 + run) for run in range(8) for key in keys) / 16.0
    ips = sum(ips_utility(dataset, key, seed=2000 + run) for run in range(8) for key in keys) / 16.0
    check(naive < truth, "the naive click rate must underestimate the true utility: %.4f < %.4f"
          % (naive, truth))
    check(abs(ips - truth) < abs(naive - truth),
          "IPS must be closer to the truth than the naive estimate: |%.4f-%.4f| < |%.4f-%.4f|"
          % (ips, truth, naive, truth))
    check(abs(ips - truth) < 0.1 * truth,
          "the IPS estimate must be within 10%% of the truth: %.4f vs %.4f" % (ips, truth))
    naive_weighting, _state = build_pairs(dataset, 0xBEEF, weight_ips=False)
    naive_weights, _history = train(naive_weighting)
    check(naive_weights != weights, "the propensity weighting must change the learned model")
    true_ratio = TRUE_WEIGHTS[0] / TRUE_WEIGHTS[1]
    check(abs(weights[0] / weights[1] - true_ratio) < abs(naive_weights[0] / naive_weights[1] - true_ratio),
          "IPS weighting must land closer to the true feature ratio (%.2f) than unweighted training: "
          "%.2f vs %.2f" % (true_ratio, weights[0] / weights[1],
                            naive_weights[0] / naive_weights[1]))

    # 7. boundary cases: no pairs, a single document, all clicks, an empty click set
    empty_weights, empty_history = train([])
    check(empty_weights == [0.0, 0.0] and set(empty_history) == {0.0},
          "no training pairs must leave the model untouched")
    single = [((0.5, 0.5), 1.0, 3)]
    check(rank_by(single, learned_key(weights)) == [0], "a single document ranks trivially")
    check(ndcg_at_k([0], single, 10) == 1.0, "a single relevant document gives nDCG 1")
    check(evaluate([[( (0.5, 0.5), 0.0, 0)]], baseline_key) == 0.0,
          "a document with grade 0 gives no ideal gain, so nDCG is defined as 0")
    empty_pairs, _state2 = build_pairs([], 0x99, weight_ips=True)
    check(empty_pairs == [], "an empty dataset yields no pairs")
    sessions_clicks, _ = simulate_clicks(dataset[:1], baseline_key, sessions=1)
    check(sessions_clicks >= 0, "the simulator must handle a single query")

    # 8. determinism of the whole pipeline
    again, _state3 = build_dataset()
    check(again == dataset, "the dataset must be reproducible")
    twin_pairs, _state4 = build_pairs(dataset, 0xBEEF, weight_ips=True)
    check(twin_pairs == pairs, "the click simulation must be reproducible")
    twin_weights, _history2 = train(twin_pairs)
    check(twin_weights == weights, "training must be reproducible")
    check(simulate_clicks(dataset, learned_key(weights)) == (learned_clicks, impressions),
          "click counts must be reproducible")

    print("767 dataset: %d queries x %d documents; baseline nDCG@10 %.4f, learned %.4f"
          % (QUERIES, DOCS_PER_QUERY, baseline_ndcg, learned_ndcg))
    print("767 training: %d IPS-weighted pairs, loss %.4f -> %.4f, weights (%.3f, %.3f)"
          % (len(pairs), history[0], history[-1], weights[0], weights[1]))
    print("767 clicks over %d impressions: baseline %d, learned %d (expected %.1f vs %.1f)"
          % (impressions, baseline_clicks, learned_clicks, expected_baseline, expected_learned))
    print("767 utility: truth %.4f, naive %.4f (biased low), IPS %.4f (within 10%%)"
          % (truth, naive, ips))
    print("PASS")


if __name__ == "__main__":
    main()
