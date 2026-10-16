/**
 * Task 520 - Implement ROC AUC estimator from predictions. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * ROC/AUC estimation computed four independent ways, all of which must agree. (1) The empirical
 * ROC curve is built by sorting the (score, label) pairs in descending score order and sweeping
 * them in blocks of equal score, so every point of the curve is a threshold that actually occurs
 * in the data; the curve starts at (FPR, TPR) = (0, 0) with the implicit +infinity threshold and
 * ends at (1, 1). (2) The AUC is the trapezoidal area under that staircase. (3) The same number is
 * produced by the rank/Mann-Whitney U identity AUC = (sum of mid-ranks of the positives -
 * nPos(nPos+1)/2) / (nPos * nNeg) with ties receiving averaged ranks. (4) A brute-force double
 * loop over positive/negative pairs counts P(s_pos > s_neg) + 0.5 * P(s_pos == s_neg), which is
 * the probabilistic definition of AUC.
 *
 * Documented checks: all three estimators agree to <= 1e-12 on tie-heavy integer score sets
 * (LCG-generated, including a fully tied set), on hand-checked 4-sample examples (AUC 1, 0, 0.75
 * and 0.5), and on jittered floating-point scores; the curve is non-decreasing in TPR and
 * non-increasing in FPR, its endpoints are exactly (0,0) and (1,1), and each of its points is
 * re-derived from a direct counting of the samples at or above that threshold; AUC(1 - s) =
 * 1 - AUC(s) and AUC is invariant under strictly increasing score transforms; the number of curve
 * points equals the number of distinct scores plus one; Youden's J = max(TPR - FPR) is compared
 * with a brute-force search over every candidate threshold; and single-class inputs are rejected
 * with a documented error instead of returning a silent NaN.
 */
"use strict";

/** Deterministic LCG: integer scores in [0, limit) plus labels, with one-in-`noise` flips. */
function lcgScores(n, seed, limit, noise = 5) {
  const scores = [];
  const labels = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    const score = seed % limit;
    seed = (seed * 75 + 74) % 65537;
    const base = score >= Math.floor(limit / 2) ? 1 : 0;
    const label = seed % noise === 0 ? 1 - base : base;
    scores.push(score);
    labels.push(label);
  }
  return { scores, labels };
}

function checkInputs(scores, labels) {
  if (scores.length !== labels.length) throw new Error("length mismatch");
  if (scores.length === 0) throw new Error("empty input");
  let positives = 0;
  for (const label of labels) {
    if (label !== 0 && label !== 1) throw new Error(`label must be 0 or 1, got ${label}`);
    positives += label;
  }
  if (positives === 0 || positives === labels.length) {
    throw new Error("AUC is undefined when only one class is present");
  }
  return positives;
}

/**
 * Empirical ROC curve: points (fpr, tpr, threshold) in descending threshold order, starting at
 * (0, 0) for the +infinity threshold and ending at (1, 1).
 */
function rocCurve(scores, labels) {
  const positives = checkInputs(scores, labels);
  const negatives = scores.length - positives;
  const order = scores.map((score, i) => i)
    .sort((a, b) => (scores[b] - scores[a]) || (a - b));
  const points = [{ threshold: Infinity, fpr: 0.0, tpr: 0.0 }];
  let tp = 0;
  let fp = 0;
  let i = 0;
  while (i < order.length) {
    const threshold = scores[order[i]];
    let j = i;
    while (j < order.length && scores[order[j]] === threshold) {
      if (labels[order[j]] === 1) tp += 1;
      else fp += 1;
      j += 1;
    }
    points.push({ threshold, tpr: tp / positives, fpr: fp / negatives });
    i = j;
  }
  return points;
}

function aucTrapezoid(scores, labels) {
  const points = rocCurve(scores, labels);
  let area = 0.0;
  for (let i = 1; i < points.length; i++) {
    area += (points[i].fpr - points[i - 1].fpr) * (points[i].tpr + points[i - 1].tpr) / 2.0;
  }
  return area;
}

/** Mann-Whitney U with mid-ranks for ties. */
function aucRank(scores, labels) {
  const positives = checkInputs(scores, labels);
  const negatives = scores.length - positives;
  const order = scores.map((score, i) => i)
    .sort((a, b) => (scores[a] - scores[b]) || (a - b));
  const ranks = new Array(scores.length).fill(0.0);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j < order.length && scores[order[j]] === scores[order[i]]) j += 1;
    const midrank = (i + 1 + j) / 2.0;
    for (let k = i; k < j; k++) ranks[order[k]] = midrank;
    i = j;
  }
  let rankSum = 0.0;
  for (let k = 0; k < scores.length; k++) if (labels[k] === 1) rankSum += ranks[k];
  return (rankSum - (positives * (positives + 1)) / 2.0) / (positives * negatives);
}

/** Probabilistic definition: P(score+ > score-) + 0.5 P(score+ == score-). */
function aucPairCount(scores, labels) {
  const positives = checkInputs(scores, labels);
  let wins = 0.0;
  let total = 0;
  for (let i = 0; i < scores.length; i++) {
    if (labels[i] !== 1) continue;
    for (let j = 0; j < scores.length; j++) {
      if (labels[j] !== 0) continue;
      total += 1;
      if (scores[i] > scores[j]) wins += 1.0;
      else if (scores[i] === scores[j]) wins += 0.5;
    }
  }
  if (total !== positives * (scores.length - positives)) throw new Error("pair count mismatch");
  return wins / total;
}

/** Youden's J = max(TPR - FPR) and the threshold attaining it (first in descending order). */
function youdenJ(scores, labels) {
  const points = rocCurve(scores, labels);
  let best = points[0];
  for (const point of points) {
    if (point.tpr - point.fpr > best.tpr - best.fpr) best = point;
  }
  return { j: best.tpr - best.fpr, threshold: best.threshold };
}

/** Brute-force best threshold by scanning every midpoint between distinct scores. */
function bruteForceYouden(scores, labels) {
  const distinct = [...new Set(scores)].sort((a, b) => a - b);
  const candidates = [Infinity, -Infinity];
  for (let i = 0; i < distinct.length; i++) candidates.push(distinct[i]);
  let bestJ = -Infinity;
  for (const threshold of candidates) {
    let tp = 0;
    let fp = 0;
    let positives = 0;
    let negatives = 0;
    for (let i = 0; i < scores.length; i++) {
      const above = scores[i] >= threshold;
      if (labels[i] === 1) {
        positives += 1;
        if (above) tp += 1;
      } else {
        negatives += 1;
        if (above) fp += 1;
      }
    }
    const j = tp / positives - fp / negatives;
    if (j > bestJ) bestJ = j;
  }
  return bestJ;
}

function main() {
  const assert = require("node:assert");

  const hand = [
    [[1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1], 1.0],
    [[1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0], 0.0],
    [[1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1], 0.75],
    [[5.0, 5.0, 5.0, 5.0], [0, 1, 0, 1], 0.5],
    [[0.1, 0.2, 0.3, 0.4], [1, 0, 1, 0], 0.25],
  ];
  for (const [scores, labels, expected] of hand) {
    assert.ok(Math.abs(aucTrapezoid(scores, labels) - expected) <= 1e-12,
      `trapezoid AUC of ${JSON.stringify(scores)}`);
    assert.ok(Math.abs(aucRank(scores, labels) - expected) <= 1e-12, "rank AUC");
    assert.ok(Math.abs(aucPairCount(scores, labels) - expected) <= 1e-12, "pair-count AUC");
  }

  const curve = rocCurve([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1]);
  assert.deepStrictEqual(curve.map((p) => [p.tpr, p.fpr]),
    [[0, 0], [0.5, 0], [0.5, 0.5], [1, 0.5], [1, 1]], "hand-checked staircase");
  assert.strictEqual(curve[0].threshold, Infinity);
  assert.strictEqual(curve[curve.length - 1].threshold, 1.0);

  const sets = [
    ["wide-ties", lcgScores(300, 11, 10)],
    ["narrow-ties", lcgScores(240, 29, 4)],
    ["risky", lcgScores(200, 37, 32, 3)],
    ["mostly-separated", lcgScores(150, 41, 100, 21)],
  ];
  const report = [];
  for (const [name, { scores, labels }] of sets) {
    const trapezoid = aucTrapezoid(scores, labels);
    const rank = aucRank(scores, labels);
    const pairs = aucPairCount(scores, labels);
    assert.ok(Math.abs(trapezoid - rank) <= 1e-12, `${name}: trapezoid vs rank`);
    assert.ok(Math.abs(trapezoid - pairs) <= 1e-12, `${name}: trapezoid vs pair count`);
    assert.ok(trapezoid >= -1e-15 && trapezoid <= 1.0 + 1e-15, `${name}: AUC is a probability`);

    const points = rocCurve(scores, labels);
    for (let i = 1; i < points.length; i++) {
      assert.ok(points[i].tpr >= points[i - 1].tpr - 1e-15, `${name}: TPR is non-decreasing`);
      assert.ok(points[i].fpr >= points[i - 1].fpr - 1e-15, `${name}: FPR is non-decreasing`);
      assert.ok(points[i].threshold < points[i - 1].threshold, `${name}: thresholds decrease`);
    }
    assert.strictEqual(points[0].tpr, 0.0);
    assert.strictEqual(points[0].fpr, 0.0);
    assert.strictEqual(points[points.length - 1].tpr, 1.0);
    assert.strictEqual(points[points.length - 1].fpr, 1.0);
    assert.strictEqual(points.length, new Set(scores).size + 1,
      `${name}: one curve point per distinct score plus the origin`);

    // Every curve point re-derived by direct counting at that threshold.
    const afterOrigin = points.slice(1);
    const shuffled = afterOrigin.slice().sort((a, b) => a.tpr - b.tpr || a.fpr - b.fpr);
    for (let i = 0; i < shuffled.length; i++) {
      const threshold = shuffled[i].threshold;
      let tp = 0;
      let fp = 0;
      let positives = 0;
      let negatives = 0;
      for (let k = 0; k < scores.length; k++) {
        if (labels[k] === 1) positives += 1;
        else negatives += 1;
        const above = scores[k] >= threshold;
        if (above && labels[k] === 1) tp += 1;
        if (above && labels[k] === 0) fp += 1;
      }
      assert.ok(Math.abs(shuffled[i].tpr - tp / positives) <= 1e-15, `${name}: TPR re-derived`);
      assert.ok(Math.abs(shuffled[i].fpr - fp / negatives) <= 1e-15, `${name}: FPR re-derived`);
    }

    const flipped = scores.map((s) => -s);
    assert.ok(Math.abs(aucTrapezoid(flipped, labels) - (1.0 - trapezoid)) <= 1e-12,
      `${name}: negating the scores mirrors the AUC`);
    const scaled = scores.map((s) => s * 3.0 + 7.0);
    assert.ok(Math.abs(aucRank(scaled, labels) - rank) <= 1e-12,
      `${name}: strictly increasing transforms preserve the AUC`);

    const youden = youdenJ(scores, labels);
    assert.ok(Math.abs(youden.j - bruteForceYouden(scores, labels)) <= 1e-12,
      `${name}: Youden's J matches the brute-force threshold search`);
    report.push([name, trapezoid, points.length]);
  }

  assert.ok(report[0][1] > 0.5, "informative coarse scores stay better than chance");
  assert.ok(report[3][1] > report[0][1], "finer score resolution separates better");

  // Float scores, no ties at all: the three estimators must still coincide.
  const floatScores = [];
  const floatLabels = [];
  let seed = 7;
  for (let i = 0; i < 120; i++) {
    seed = (seed * 75 + 74) % 65537;
    const u = seed / 65537.0;
    seed = (seed * 75 + 74) % 65537;
    const label = seed % 2;
    floatScores.push(u + (label === 1 ? 0.15 : 0.0));
    floatLabels.push(label);
  }
  assert.strictEqual(new Set(floatScores).size, floatScores.length, "scores are all distinct");
  const floatAuc = aucTrapezoid(floatScores, floatLabels);
  assert.ok(Math.abs(floatAuc - aucRank(floatScores, floatLabels)) <= 1e-12);
  assert.ok(Math.abs(floatAuc - aucPairCount(floatScores, floatLabels)) <= 1e-12);
  assert.ok(floatAuc > 0.6 && floatAuc < 0.9, "shifted mixture lands in a sensible band");

  assert.throws(() => aucTrapezoid([1.0, 2.0], [1, 1]), /one class/);
  assert.throws(() => aucTrapezoid([1.0, 2.0], [0, 0]), /one class/);
  assert.throws(() => aucTrapezoid([1.0], [2]), /label must be 0 or 1/);
  assert.throws(() => aucTrapezoid([1.0, 2.0], [1]), /length mismatch/);
  assert.throws(() => aucTrapezoid([], []), /empty input/);
}

module.exports = { lcgScores, checkInputs, rocCurve, aucTrapezoid, aucRank, aucPairCount, youdenJ,
  bruteForceYouden };

if (require.main === module) {
  main();
  console.log("PASS");
}
