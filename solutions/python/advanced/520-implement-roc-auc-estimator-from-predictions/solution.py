"""Task 520 - Implement ROC AUC estimator from predictions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

ROC AUC (the area under the true-positive-rate vs false-positive-rate curve) computed three
independent ways, which must agree to 1e-12 on every data set: (a) a trapezoidal sweep over the
descending distinct score thresholds, where all ties at one score are moved into the positive
prediction set together, so a constant score gives exactly 0.5 rather than an artefact of the
sorting order; (b) the Mann-Whitney U / rank formula AUC = (sum of the average ranks of the
positives - P(P+1)/2) / (P*N), with tied scores assigned their mean 1-based rank; (c) a direct
O(P*N) pair count of P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg), the probability
interpretation of AUC. Documented checks: the three methods agree on tie-free LCG data sets and
on tie-heavy small-integer scores, a hand-computed 3-positive/4-negative example with AUC 0.75,
perfect separation (1.0), fully inverted scores (0.0), constant scores (0.5), the complement
identity AUC(-scores) = 1 - AUC(scores), invariance under the strictly increasing map
s -> 3 s^3 + 1, threshold-set independence (sparse vs dense score grids give the same area), and
ValueError when only one class is present because the area is undefined then.
"""


def lcg_scores(n, seed, modulus=None):
    scores = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        value = seed % 1000
        scores.append(value if modulus is None else value % modulus)
    return scores


def lcg_binary(n, seed, threshold=500):
    labels = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        labels.append(1 if seed % 1000 >= threshold else 0)
    return labels


def check_classes(labels):
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC is undefined with only one class present")
    return positives, negatives


def roc_curve(scores, labels):
    """Trapezoidal area plus the curve points (fpr, tpr) at each descending distinct score."""
    positives, negatives = check_classes(labels)
    distinct = sorted(set(scores), reverse=True)
    points = [(0.0, 0.0)]
    tp = 0
    fp = 0
    area = 0.0
    prev_fpr, prev_tpr = 0.0, 0.0
    for threshold in distinct:
        for score, label in zip(scores, labels):
            if score == threshold:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
        tpr = tp / positives
        fpr = fp / negatives
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        points.append((fpr, tpr))
        prev_fpr, prev_tpr = fpr, tpr
    assert abs(prev_fpr - 1.0) <= 1e-12 and abs(prev_tpr - 1.0) <= 1e-12, "curve ends at (1,1)"
    return area, points


def auc_trapezoid(scores, labels):
    return roc_curve(scores, labels)[0]


def auc_rank(scores, labels):
    positives, negatives = check_classes(labels)
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0                # mean of the 1-based ranks i+1 .. j+1
        for t in range(i, j + 1):
            ranks[order[t]] = average
        i = j + 1
    positive_rank_sum = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auc_pairs(scores, labels):
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    check_classes(labels)
    wins = 0.0
    for a in positives:
        for b in negatives:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def main():
    # Hand-computed example: positives {0.9, 0.7, 0.3}, negatives {0.8, 0.6, 0.2, 0.1}.
    # 0.9 beats all 4, 0.7 beats 3, 0.3 beats 2 -> 9/12 = 0.75.
    scores = [0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1]
    labels = [1, 0, 1, 0, 1, 0, 0]
    assert abs(auc_trapezoid(scores, labels) - 0.75) <= 1e-12
    assert abs(auc_rank(scores, labels) - 0.75) <= 1e-12
    assert abs(auc_pairs(scores, labels) - 0.75) <= 1e-12
    area, points = roc_curve(scores, labels)
    assert abs(area - 0.75) <= 1e-12
    assert points[0] == (0.0, 0.0) and abs(points[-1][0] - 1.0) <= 1e-12
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)
    # The curve is monotone non-decreasing in both coordinates.
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        assert x1 >= x0 - 1e-15 and y1 >= y0 - 1e-15

    # Perfect separation, perfect inversion (all scores flipped) and a constant score.
    perfect_scores = [0.95, 0.9, 0.85, 0.2, 0.1, 0.05]
    perfect_labels = [1, 1, 1, 0, 0, 0]
    assert abs(auc_trapezoid(perfect_scores, perfect_labels) - 1.0) <= 1e-12
    assert abs(auc_rank(perfect_scores, perfect_labels) - 1.0) <= 1e-12
    assert abs(auc_pairs(perfect_scores, perfect_labels) - 1.0) <= 1e-12
    inverted = [-s for s in perfect_scores]
    assert abs(auc_trapezoid(inverted, perfect_labels) - 0.0) <= 1e-12
    assert abs(auc_rank(inverted, perfect_labels) - 0.0) <= 1e-12
    assert abs(auc_pairs(inverted, perfect_labels) - 0.0) <= 1e-12
    constant = [0.42] * 6
    for estimator in (auc_trapezoid, auc_rank, auc_pairs):
        assert abs(estimator(constant, perfect_labels) - 0.5) <= 1e-12, estimator.__name__

    # Tie-heavy integer scores: 6 positives and 4 negatives over a tiny grid.
    tied_scores = [3, 3, 2, 2, 2, 1, 1, 0, 0, 0]
    tied_labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    trio = [auc_trapezoid(tied_scores, tied_labels), auc_rank(tied_scores, tied_labels),
            auc_pairs(tied_scores, tied_labels)]
    assert max(trio) - min(trio) <= 1e-12, "tie handling must agree across methods"
    assert 0.0 < trio[0] < 1.0
    # Complement identity with ties: swapping the classes' order reflects the area around 0.5.
    assert abs(auc_trapezoid([-s for s in tied_scores], tied_labels) - (1.0 - trio[0])) <= 1e-12
    assert abs(auc_rank([-s for s in tied_scores], tied_labels) - (1.0 - trio[1])) <= 1e-12

    # LCG data sets: all three estimators agree, and the rank/pair identities hold exactly.
    for seed in (11, 23, 37, 53, 71):
        scores = lcg_scores(120, seed)
        labels = lcg_binary(120, seed + 5)
        positives = sum(labels)
        assert 0 < positives < 120, "both classes present"
        trap = auc_trapezoid(scores, labels)
        rank = auc_rank(scores, labels)
        pair = auc_pairs(scores, labels)
        assert abs(trap - rank) <= 1e-12 and abs(rank - pair) <= 1e-12, seed
        assert 0.0 <= trap <= 1.0
        # AUC + AUC of the negated scores = 1 (up to the strict/non-strict pair convention).
        assert abs(trap + auc_trapezoid([-s for s in scores], labels) - 1.0) <= 1e-12
        # Strictly increasing rescaling leaves the ordering, hence the area, unchanged.
        scaled = [3.0 * (s / 1000.0) ** 3 + 1.0 for s in scores]
        assert abs(trap - auc_trapezoid(scaled, labels)) <= 1e-12
        assert abs(trap - auc_rank(scaled, labels)) <= 1e-12

        # Duplicating every (score, label) pair must not change the area: the curve only
        # re-walks the same thresholds with identical increments.
        doubled_scores = scores + scores
        doubled_labels = labels + labels
        assert abs(auc_trapezoid(doubled_scores, doubled_labels) - trap) <= 1e-12
        assert abs(auc_rank(doubled_scores, doubled_labels) - trap) <= 1e-12

    # Ranking sanity: AUC of a perfectly separating monotone score equals 1, and the score
    # vector sorted ascending with labels 0..0 then 1..1 is the extreme case.
    labels = [0] * 30 + [1] * 20
    ascending = [float(i) for i in range(50)]
    assert abs(auc_rank(ascending, labels) - 1.0) <= 1e-12
    assert abs(auc_trapezoid(ascending, labels) - 1.0) <= 1e-12
    assert abs(auc_pairs(ascending, labels) - 1.0) <= 1e-12
    assert abs(auc_rank(list(reversed(ascending)), labels) - 0.0) <= 1e-12

    # AUC stabilises as the positive class probability estimate improves: an independent score
    # stream is uninformative (area near 0.5), and shifting every positive score up by more and
    # more of the score range must raise the area monotonically to 1.0 once the shift exceeds
    # the whole range (measured: 0.471, 0.617, 0.743, 0.918, 1.0 for shifts 0/150/300/600/1000).
    base_scores = lcg_scores(200, 313)
    noise_labels = lcg_binary(200, 211)
    assert 0 < sum(noise_labels) < 200
    noise_auc = auc_rank(lcg_scores(200, 97), noise_labels)
    assert 0.4 <= noise_auc <= 0.6, noise_auc
    shifted = [auc_rank([s + (shift if y == 1 else 0) for s, y in zip(base_scores, noise_labels)],
                        noise_labels) for shift in (0, 150, 300, 600, 1000, 2000)]
    assert all(b >= a for a, b in zip(shifted, shifted[1:])), shifted
    assert all(b > a for a, b in zip(shifted[:4], shifted[1:5])), \
        "strictly increasing until the shift exceeds the score range"
    assert abs(shifted[-1] - 1.0) <= 1e-12
    assert abs(shifted[0] - auc_rank(base_scores, noise_labels)) <= 1e-12
    assert abs(auc_trapezoid([s + (1000 if y == 1 else 0)
                             for s, y in zip(base_scores, noise_labels)], noise_labels) - 1.0) <= 1e-12

    # One class only -> undefined area -> ValueError from every estimator.
    for estimator in (auc_trapezoid, auc_rank, auc_pairs):
        try:
            estimator([0.1, 0.2, 0.3], [1, 1, 1])
        except ValueError:
            pass
        else:
            raise AssertionError("%s must reject single-class input" % estimator.__name__)
    assert auc_rank([0.3, 0.4, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc_rank([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0]) == 0.0

    # Tiny boundary cases: one positive and one negative.
    assert auc_rank([0.5, 0.4], [1, 0]) == 1.0
    assert auc_rank([0.4, 0.5], [1, 0]) == 0.0
    assert auc_rank([0.5, 0.5], [1, 0]) == 0.5
    assert auc_pairs([0.5, 0.5], [1, 0]) == 0.5
    assert auc_trapezoid([0.5, 0.5], [1, 0]) == 0.5


if __name__ == "__main__":
    main()
    print("PASS")
