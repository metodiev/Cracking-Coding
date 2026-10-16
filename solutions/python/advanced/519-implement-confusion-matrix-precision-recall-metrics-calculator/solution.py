"""Task 519 - Implement confusion matrix + precision/recall metrics calculator. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Multi-class evaluation from a labelled confusion matrix. cm[t][p] counts the pairs whose true
label is labels[t] and whose predicted label is labels[p] (labels is supplied explicitly, so a
class that never appears in either vector still gets a zero row/column). For every class the
calculator derives tp = cm[i][i], fp = column sum - tp, fn = row sum - tp, support = row sum and
then precision, recall and F1 with the documented 0/0 convention (a metric whose denominator is
zero is reported as 0.0, never a division error). The aggregates are macro (unweighted mean over
all supplied classes, so absent classes dilute the macro average), weighted (support-weighted
mean) and micro (global tp/fp/fn, which for single-label classification equals accuracy). Checks:
a hand-computed binary table, a hand-computed 3-class table where one class is never predicted,
row/column marginal identities, micro == accuracy on LCG data, macro/weighted recomputed by hand
from the per-class rows, invariance of the whole report under a permutation of the label order
(the matrix must permute the same way), perfect and fully-reversed predictions, zero-support
classes, and ValueError on unknown labels or length mismatches.
"""


def confusion_matrix(y_true, y_pred, labels):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    index = {}
    for i, label in enumerate(labels):
        if label in index:
            raise ValueError("duplicate label in labels: %r" % (label,))
        index[label] = i
    cm = [[0] * len(labels) for _ in labels]
    for actual, predicted in zip(y_true, y_pred):
        if actual not in index:
            raise ValueError("unknown true label: %r" % (actual,))
        if predicted not in index:
            raise ValueError("unknown predicted label: %r" % (predicted,))
        cm[index[actual]][index[predicted]] += 1
    return cm


def per_class_report(cm, labels):
    n = len(cm)
    rows = []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i]) - tp
        support = sum(cm[i])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        rows.append({"label": labels[i], "tp": tp, "fp": fp, "fn": fn, "support": support,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def micro_scores(cm):
    n = len(cm)
    tp = sum(cm[i][i] for i in range(n))
    total = sum(sum(row) for row in cm)
    fp = total - tp
    fn = total - tp                              # every non-diagonal cell is one fp and one fn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_scores(rows):
    n = len(rows)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"precision": sum(row["precision"] for row in rows) / n,
            "recall": sum(row["recall"] for row in rows) / n,
            "f1": sum(row["f1"] for row in rows) / n}


def weighted_scores(rows):
    total = sum(row["support"] for row in rows)
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {key: sum(row[key] * row["support"] for row in rows) / total
            for key in ("precision", "recall", "f1")}


def accuracy(cm):
    total = sum(sum(row) for row in cm)
    if total == 0:
        return 0.0
    return sum(cm[i][i] for i in range(len(cm))) / total


def lcg_labels(n, seed, classes):
    out = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        out.append(seed % classes)
    return out


def near(a, b, tol=1e-12):
    return abs(a - b) <= tol


def main():
    # Hand-computed binary table: y_true = [1,1,0,0,1,0], y_pred = [1,0,0,0,1,1].
    # cm = [[2, 1], [1, 2]] -> accuracy 4/6, precision 2/3, recall 2/3, F1 2/3.
    cm = confusion_matrix([1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 1, 1], [0, 1])
    assert cm == [[2, 1], [1, 2]], cm
    rows = per_class_report(cm, [0, 1])
    assert all(near(row["precision"], 2.0 / 3.0) for row in rows)
    assert all(near(row["recall"], 2.0 / 3.0) for row in rows)
    assert all(near(row["f1"], 2.0 / 3.0) for row in rows)
    assert [row["support"] for row in rows] == [3, 3]
    assert [row["fp"] for row in rows] == [1, 1]
    assert near(accuracy(cm), 4.0 / 6.0)
    assert near(micro_scores(cm)["f1"], 4.0 / 6.0), "micro F1 equals accuracy for one label/row"
    assert near(macro_scores(rows)["f1"], 2.0 / 3.0)
    assert near(weighted_scores(rows)["f1"], 2.0 / 3.0)

    # Hand-computed 3-class table where class 2 is never predicted:
    # y_true = [0,1,2], y_pred = [0,0,0] -> cm = [[1,0,0],[1,0,0],[1,0,0]].
    cm = confusion_matrix([0, 1, 2], [0, 0, 0], [0, 1, 2])
    assert cm == [[1, 0, 0], [1, 0, 0], [1, 0, 0]], cm
    rows = per_class_report(cm, [0, 1, 2])
    assert near(rows[0]["precision"], 1.0 / 3.0) and near(rows[0]["recall"], 1.0)
    assert rows[1]["precision"] == 0.0 and rows[1]["recall"] == 0.0 and rows[1]["f1"] == 0.0
    assert rows[2]["precision"] == 0.0 and rows[2]["recall"] == 0.0
    assert near(accuracy(cm), 1.0 / 3.0)
    assert near(macro_scores(rows)["precision"], (1.0 / 3.0) / 3.0)
    assert near(weighted_scores(rows)["precision"], (1.0 / 3.0) / 3.0)
    assert near(macro_scores(rows)["recall"], 1.0 / 3.0)

    # A supplied label that never occurs keeps its zero row/column and dilutes macro scores.
    cm = confusion_matrix([0, 1], [0, 1], [0, 1, 7])
    assert cm == [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    rows = per_class_report(cm, [0, 1, 7])
    assert rows[2]["support"] == 0 and rows[2]["f1"] == 0.0
    assert near(macro_scores(rows)["f1"], 2.0 / 3.0), "absent class counts in the macro mean"
    assert near(weighted_scores(rows)["f1"], 1.0), "absent class has zero weight"

    # Perfect and fully-reversed binary predictions.
    truth = lcg_labels(120, 11, 2)
    perfect = confusion_matrix(truth, list(truth), [0, 1])
    rows = per_class_report(perfect, [0, 1])
    assert near(accuracy(perfect), 1.0) and near(micro_scores(perfect)["precision"], 1.0)
    assert all(row["f1"] == 1.0 for row in rows)
    reversed_cm = confusion_matrix(truth, [1 - y for y in truth], [0, 1])
    assert near(accuracy(reversed_cm), 0.0)
    assert all(row["f1"] == 0.0 for row in per_class_report(reversed_cm, [0, 1]))

    # LCG 3-class data: marginals, micro == accuracy, macro/weighted recomputed by brute force.
    for seed in (23, 37, 53):
        truth = lcg_labels(150, seed, 3)
        predicted = lcg_labels(150, seed + 7, 3)
        cm = confusion_matrix(truth, predicted, [0, 1, 2])
        assert sum(sum(row) for row in cm) == 150
        for i in range(3):
            assert sum(cm[i]) == sum(1 for t in truth if t == i), "row sums are true counts"
            assert sum(cm[r][i] for r in range(3)) == \
                sum(1 for p in predicted if p == i), "column sums are predicted counts"
        rows = per_class_report(cm, [0, 1, 2])
        micro = micro_scores(cm)
        assert near(micro["precision"], micro["recall"])
        assert near(micro["precision"], accuracy(cm))
        expected_precision = sum(row["precision"] for row in rows) / 3.0
        assert near(macro_scores(rows)["precision"], expected_precision)
        expected_weighted = sum(row["precision"] * row["support"] for row in rows) / 150.0
        assert near(weighted_scores(rows)["precision"], expected_weighted)
        # Independent brute-force recount of one cell and of the diagonal.
        assert cm[1][2] == sum(1 for t, p in zip(truth, predicted) if t == 1 and p == 2)
        assert accuracy(cm) == sum(1 for t, p in zip(truth, predicted) if t == p) / 150.0

    # Re-labelling invariance: permuting `labels` permutes the matrix identically.
    truth = lcg_labels(60, 71, 4)
    predicted = lcg_labels(60, 97, 4)
    base = confusion_matrix(truth, predicted, [0, 1, 2, 3])
    order = [2, 0, 3, 1]
    permuted = confusion_matrix(truth, predicted, order)
    assert permuted == [[base[order[i]][order[j]] for j in range(4)] for i in range(4)]
    base_rows = {row["label"]: row for row in per_class_report(base, [0, 1, 2, 3])}
    perm_rows = {row["label"]: row for row in per_class_report(permuted, order)}
    for label in range(4):
        for key in ("tp", "fp", "fn", "support", "precision", "recall", "f1"):
            assert near(base_rows[label][key], perm_rows[label][key]), (label, key)
    assert near(macro_scores(list(base_rows.values()))["f1"],
                macro_scores(list(perm_rows.values()))["f1"])

    # Errors: unknown labels and length mismatch.
    for bad_true, bad_pred in (([0, 9], [0, 1]), ([0, 1], [0, 9])):
        try:
            confusion_matrix(bad_true, bad_pred, [0, 1])
        except ValueError:
            pass
        else:
            raise AssertionError("unknown label must raise ValueError")
    try:
        confusion_matrix([0, 1], [0], [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise ValueError")
    try:
        confusion_matrix([0, 1], [0, 1], [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate labels must raise ValueError")

    # Empty input is legal and everything degenerates to 0.0 without a division error.
    cm = confusion_matrix([], [], [0, 1])
    assert cm == [[0, 0], [0, 0]]
    assert accuracy(cm) == 0.0
    assert micro_scores(cm) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert all(row["f1"] == 0.0 for row in per_class_report(cm, [0, 1]))


if __name__ == "__main__":
    main()
    print("PASS")
