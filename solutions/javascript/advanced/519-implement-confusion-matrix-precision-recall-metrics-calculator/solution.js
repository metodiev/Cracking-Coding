/**
 * Task 519 - Implement confusion matrix + precision/recall metrics calculator. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Multiclass evaluation metrics built on one primitive: the confusion matrix M[true][pred], with
 * row sums equal to class supports and column sums equal to the predicted totals. From M the file
 * derives, per class, TP = M[c][c], FP = column(c) - TP, FN = row(c) - TP, support = row(c), and
 * then precision, recall, F1 = 2PR/(P+R), specificity and balanced accuracy; the degenerate
 * denominators (a class that is never predicted or never present) are defined as 0.0 rather than
 * NaN, which is documented and asserted explicitly. Averaging follows the textbook definitions of
 * macro (unweighted mean over the classes present in the union of the label sets), weighted
 * (support-weighted mean) and micro/Sokolova-Zechner aggregation (sum the TP/FP/FN counters and
 * divide once). For single-label multiclass problems micro-precision, micro-recall and micro-F1
 * must all equal plain accuracy; that identity is asserted rather than assumed, and each average
 * is additionally recomputed by an independent loop over the matrix.
 *
 * Documented checks: the matrix is built twice - once by index lookup and once by a brute-force
 * double loop over the predictions - and the two agree on an LCG-generated 3-class set; a
 * hand-computed 10-sample binary example fixes every metric to its known rational value; a perfect
 * classifier yields 1.0 everywhere; always predicting one class gives precision 0 for the never
 * predicted classes and recall 0 for the never produced class; the row/column totals of the
 * matrix reproduce the class counts of the raw label lists; unknown prediction labels are
 * rejected; and metric definitions obey min(P, R) <= F1 <= max(P, R), balanced accuracy =
 * (recall_0 + recall_1)/2 in the binary case, and macro-F1 = mean of the per-class F1 scores.
 */
"use strict";

/** Deterministic LCG producing a class label in [0, classes) from a small seed. */
function lcgLabels(n, seed, classes) {
  const labels = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    labels.push(seed % classes);
  }
  return labels;
}

/** Noisy copy of a label list: with probability ~1/5 the label is shifted by 1 modulo classes. */
function corrupt(labels, seed, classes, oneIn = 5) {
  const out = [];
  for (const label of labels) {
    seed = (seed * 75 + 74) % 65537;
    out.push(seed % oneIn === 0 ? (label + 1) % classes : label);
  }
  return out;
}

class ConfusionMatrix {
  constructor(yTrue, yPred, labels = null) {
    if (yTrue.length !== yPred.length) throw new Error("length mismatch");
    if (labels === null) {
      const seen = new Set();
      for (const v of yTrue) seen.add(v);
      for (const v of yPred) seen.add(v);
      labels = [...seen].sort((a, b) => a - b);
    }
    this.labels = labels.slice();
    this.index = new Map();
    this.labels.forEach((label, i) => this.index.set(label, i));
    const k = this.labels.length;
    this.matrix = [];
    for (let i = 0; i < k; i++) this.matrix.push(new Array(k).fill(0));
    for (let i = 0; i < yTrue.length; i++) {
      const row = this.index.get(yTrue[i]);
      const col = this.index.get(yPred[i]);
      if (row === undefined) throw new Error(`unknown true label ${yTrue[i]}`);
      if (col === undefined) throw new Error(`unknown predicted label ${yPred[i]}`);
      this.matrix[row][col] += 1;
    }
  }

  /** Same counts, produced without the index map: a naive double loop over the predictions. */
  static bruteMatrix(yTrue, yPred, labels) {
    const k = labels.length;
    const matrix = [];
    for (let i = 0; i < k; i++) matrix.push(new Array(k).fill(0));
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        let count = 0;
        for (let i = 0; i < yTrue.length; i++) {
          if (yTrue[i] === labels[r] && yPred[i] === labels[c]) count += 1;
        }
        matrix[r][c] = count;
      }
    }
    return matrix;
  }

  total() {
    return this.matrix.reduce((acc, row) => acc + row.reduce((a, b) => a + b, 0), 0);
  }

  rowTotals() {
    return this.matrix.map((row) => row.reduce((a, b) => a + b, 0));
  }

  colTotals() {
    const k = this.labels.length;
    const out = new Array(k).fill(0);
    for (const row of this.matrix) {
      row.forEach((value, c) => {
        out[c] += value;
      });
    }
    return out;
  }

  accuracy() {
    let correct = 0;
    for (let i = 0; i < this.labels.length; i++) correct += this.matrix[i][i];
    const total = this.total();
    return total === 0 ? 0.0 : correct / total;
  }

  /** Per-class TP/FP/FN/support plus the derived rates; zero denominators give 0.0. */
  perClass() {
    const k = this.labels.length;
    const rows = this.rowTotals();
    const cols = this.colTotals();
    const out = [];
    for (let c = 0; c < k; c++) {
      const tp = this.matrix[c][c];
      const fp = cols[c] - tp;
      const fn = rows[c] - tp;
      const support = rows[c];
      const precision = tp + fp === 0 ? 0.0 : tp / (tp + fp);
      const recall = tp + fn === 0 ? 0.0 : tp / (tp + fn);
      const f1 = precision + recall === 0.0 ? 0.0 : (2.0 * precision * recall) / (precision + recall);
      const specificity = (() => {
        // TN = every sample that is neither this class nor predicted as this class.
        const tn = this.total() - rows[c] - cols[c] + tp;
        const denom = tn + fp;
        return denom === 0 ? 0.0 : tn / denom;
      })();
      out.push({ label: this.labels[c], tp, fp, fn, support, precision, recall, f1,
        specificity, balanced: specificity + recall === 0.0 ? 0.0 : (recall + specificity) / 2.0 });
    }
    return out;
  }

  macro(key) {
    const rows = this.perClass();
    if (rows.length === 0) return 0.0;
    return rows.reduce((acc, row) => acc + row[key], 0.0) / rows.length;
  }

  weighted(key) {
    const rows = this.perClass();
    const total = this.total();
    if (total === 0) return 0.0;
    return rows.reduce((acc, row) => acc + row[key] * row.support, 0.0) / total;
  }

  micro() {
    const rows = this.perClass();
    const tp = rows.reduce((a, r) => a + r.tp, 0);
    const fp = rows.reduce((a, r) => a + r.fp, 0);
    const fn = rows.reduce((a, r) => a + r.fn, 0);
    const precision = tp + fp === 0 ? 0.0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0.0 : tp / (tp + fn);
    const f1 = precision + recall === 0.0 ? 0.0 : (2.0 * precision * recall) / (precision + recall);
    return { tp, fp, fn, precision, recall, f1 };
  }

  /** Per-class one-vs-rest counters for any class label, from the raw lists (spot check helper). */
  static oneVsRestCounts(yTrue, yPred, positive) {
    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const actual = yTrue[i] === positive;
      const predicted = yPred[i] === positive;
      if (actual && predicted) tp += 1;
      else if (!actual && predicted) fp += 1;
      else if (actual && !predicted) fn += 1;
      else tn += 1;
    }
    return { tp, fp, tn, fn };
  }
}

const TOL = 1e-12;

function main() {
  const assert = require("node:assert");

  // Hand-computed binary example: 10 samples, 7 positives / 3 negatives.
  const yTrue = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0];
  const yPred = [1, 1, 1, 1, 0, 0, 1, 0, 1, 0];
  const binary = new ConfusionMatrix(yTrue, yPred, [0, 1]);
  assert.deepStrictEqual(binary.matrix, [[2, 1], [2, 5]], "rows are truth, columns prediction");
  const binaryRows = binary.perClass();
  assert.deepStrictEqual(binaryRows.map((r) => [r.label, r.tp, r.fp, r.fn, r.support]),
    [[0, 2, 2, 1, 3], [1, 5, 1, 2, 7]]);
  assert.ok(Math.abs(binaryRows[1].precision - 5 / 6) <= TOL);
  assert.ok(Math.abs(binaryRows[1].recall - 5 / 7) <= TOL);
  assert.ok(Math.abs(binaryRows[1].f1 - (2 * 5) / (6 + 7)) <= TOL, "F1 = 2TP/(2TP+FP+FN)");
  assert.ok(Math.abs(binary.accuracy() - 0.7) <= TOL);
  assert.ok(Math.abs(binary.macro("precision") - (2 / 4 + 5 / 6) / 2) <= TOL);
  assert.ok(Math.abs(binary.weighted("recall") - (3 * (2 / 3) + 7 * (5 / 7)) / 10) <= TOL);
  const binaryMicro = binary.micro();
  assert.strictEqual(binaryMicro.tp, 7);
  assert.ok(Math.abs(binaryMicro.precision - binary.accuracy()) <= TOL,
    "single-label: micro precision is accuracy");
  assert.ok(Math.abs(binaryMicro.recall - binary.accuracy()) <= TOL);
  assert.ok(Math.abs(binaryMicro.f1 - binary.accuracy()) <= TOL);
  assert.ok(Math.abs(binaryRows[0].balanced - (2 / 3 + 5 / 7) / 2) <= TOL,
    "balanced accuracy = (recall + specificity)/2 for class 0");

  // Brute-force one-vs-rest counters must reproduce the matrix cells for both classes.
  for (const positive of [0, 1]) {
    const counts = ConfusionMatrix.oneVsRestCounts(yTrue, yPred, positive);
    const row = binaryRows[positive];
    assert.strictEqual(counts.tp, row.tp);
    assert.strictEqual(counts.fp, row.fp);
    assert.strictEqual(counts.fn, row.fn);
    assert.ok(Math.abs(counts.tp + counts.tn - binary.accuracy() * yTrue.length) <= 1e-9);
  }

  // LCG-generated 3-class data: index-map construction vs naive double-loop construction.
  const classes = 3;
  const truth = lcgLabels(400, 31, classes);
  const predicted = corrupt(truth, 97, classes, 5);
  const cm = new ConfusionMatrix(truth, predicted);
  assert.deepStrictEqual(cm.labels, [0, 1, 2]);
  assert.deepStrictEqual(cm.matrix, ConfusionMatrix.bruteMatrix(truth, predicted, cm.labels),
    "two independent matrix constructions must agree");

  const rows = cm.perClass();
  const rowTotals = cm.rowTotals();
  const colTotals = cm.colTotals();
  assert.strictEqual(cm.total(), 400);
  for (let r = 0; r < classes; r++) {
    const expectedSupport = truth.filter((v) => v === cm.labels[r]).length;
    assert.strictEqual(rowTotals[r], expectedSupport, "row total is the class support");
    const expectedPredicted = predicted.filter((v) => v === cm.labels[r]).length;
    assert.strictEqual(colTotals[r], expectedPredicted, "column total is the predicted count");
    assert.strictEqual(rows[r].tp + rows[r].fn, rows[r].support);
    assert.ok(Math.abs(rows[r].recall - rows[r].tp / rows[r].support) <= TOL);
    // The harmonic mean sits between the two inputs: min(P, R) <= F1 <= max(P, R).
    assert.ok(rows[r].f1 >= Math.min(rows[r].precision, rows[r].recall) - TOL, "F1 >= min(P, R)");
    assert.ok(rows[r].f1 <= Math.max(rows[r].precision, rows[r].recall) + TOL, "F1 <= max(P, R)");
    assert.ok(Math.abs(2 * (rows[r].precision * rows[r].recall) -
      rows[r].f1 * (rows[r].precision + rows[r].recall)) <= 1e-9, "harmonic-mean identity");
  }
  assert.ok(Math.abs(cm.macro("f1") -
    rows.reduce((a, r) => a + r.f1, 0.0) / classes) <= TOL);
  assert.ok(Math.abs(cm.weighted("f1") -
    rows.reduce((a, r) => a + r.f1 * r.support, 0.0) / 400) <= TOL);
  assert.ok(cm.accuracy() >= 0.75 && cm.accuracy() <= 0.85, "one-in-five corruption");

  const micro = cm.micro();
  for (const key of ["precision", "recall", "f1"]) {
    assert.ok(Math.abs(micro[key] - cm.accuracy()) <= 1e-12, `micro ${key} equals accuracy`);
  }
  for (const key of ["precision", "recall"]) {
    // Micro precision/recall are weighted averages of the per-class values (weights tp+fp / tp+fn).
    assert.ok(micro[key] >= Math.min(...rows.map((r) => r[key])) - TOL,
      `micro ${key} lies inside the per-class range`);
    assert.ok(micro[key] <= Math.max(...rows.map((r) => r[key])) + TOL);
  }
  assert.ok(Math.abs(micro.f1 - (2 * micro.precision * micro.recall) /
    (micro.precision + micro.recall)) <= TOL, "micro F1 is the harmonic mean of micro P and R");
  assert.strictEqual(micro.fp, micro.fn, "single-label multiclass: FP and FN totals are equal");
  assert.strictEqual(micro.fp, 400 - micro.tp, "FP = N - TP in the single-label case");
  assert.strictEqual(micro.fn, 400 - micro.tp);
  assert.strictEqual(micro.tp + micro.fn, 400);

  // Perfect classifier.
  const perfect = new ConfusionMatrix(truth, truth.slice(), [0, 1, 2]);
  assert.ok(Math.abs(perfect.accuracy() - 1.0) <= TOL);
  assert.strictEqual(perfect.micro().fp, 0);
  assert.strictEqual(perfect.micro().fn, 0);
  for (const row of perfect.perClass()) assert.ok(Math.abs(row.f1 - 1.0) <= TOL);

  // Constant predictor: recall 0 for the unseen classes, precision 0 for the never predicted one.
  const constant = new ConfusionMatrix([0, 0, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1], [0, 1, 2]);
  assert.deepStrictEqual(constant.matrix, [[0, 2, 0], [0, 2, 0], [0, 2, 0]]);
  const constantRows = constant.perClass();
  assert.deepStrictEqual(constantRows.map((r) => r.precision), [0.0, 2 / 6, 0.0]);
  assert.deepStrictEqual(constantRows.map((r) => r.recall), [0.0, 1.0, 0.0]);
  assert.ok(Math.abs(constant.accuracy() - 2 / 6) <= TOL);
  assert.ok(Math.abs(constantRows[1].balanced - (1.0 + 0.0) / 2) <= TOL,
    "balanced accuracy of the always-positive predictor: (recall + specificity)/2");
  assert.ok(Math.abs(constant.macro("recall") - 1 / 3) <= TOL);

  // Degenerate labels: a class present in the spec but absent from the data.
  const sparse = new ConfusionMatrix([0, 0], [0, 0], [0, 1, 2]);
  assert.deepStrictEqual(sparse.matrix, [[2, 0, 0], [0, 0, 0], [0, 0, 0]]);
  const sparseRows = sparse.perClass();
  assert.strictEqual(sparseRows[1].precision, 0.0, "never predicted -> precision defined as 0");
  assert.strictEqual(sparseRows[1].recall, 0.0, "never present -> recall defined as 0 (not NaN)");
  assert.strictEqual(sparseRows[1].f1, 0.0);
  assert.ok(Number.isFinite(sparse.macro("precision")));

  // Empty input is allowed and yields 0.0 metrics rather than NaN.
  const empty = new ConfusionMatrix([], [], [0, 1]);
  assert.deepStrictEqual(empty.matrix, [[0, 0], [0, 0]]);
  assert.strictEqual(empty.accuracy(), 0.0);
  assert.strictEqual(empty.macro("f1"), 0.0);
  assert.strictEqual(empty.weighted("f1"), 0.0);

  assert.throws(() => new ConfusionMatrix([0, 1], [0], [0, 1]), /length mismatch/);
  assert.throws(() => new ConfusionMatrix([0, 1], [0, 2], [0, 1]), /unknown predicted label 2/);
  assert.throws(() => new ConfusionMatrix([0, 3], [0, 1], [0, 1]), /unknown true label 3/);
}

module.exports = { lcgLabels, corrupt, ConfusionMatrix, TOL };

if (require.main === module) {
  main();
  console.log("PASS");
}
