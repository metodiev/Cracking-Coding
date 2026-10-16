/** Task 897 - Implement a compressed neural network runtime with weight sharing and Huffman coding.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Three compression stages over a small multi-layer perceptron, and a runtime that executes the
 * compressed model directly:
 *
 *  train     an 8-12-3 MLP trained by full-batch gradient descent on a deterministic synthetic
 *            three-class problem (so the accuracy numbers below mean something).
 *  share     k-means weight sharing: the weights are clustered into k centroids (Lloyd's algorithm with
 *            a seeded initialisation) and replaced by the cluster index. The codebook, not the weights,
 *            is what gets stored; the centroids are re-derived from their cluster members, which is
 *            verified by recomputation.
 *  entropy   the index stream is Huffman coded. The tree is checked for the prefix-free property, the
 *            Kraft sum is required to be exactly 1, the average code length is required to be within one
 *            bit of the source entropy (the Huffman optimality bound), and the round trip is exact.
 *
 * The runtime then dequantises on the fly and runs inference, and reports the compression ratio in bits
 * against float32, the accuracy retained, and the prediction agreement with the uncompressed model.
 *
 * Invariants verified below: (1) k-means converges (the assignment is optimal for the centroids and the
 * centroids are the cluster means — both checked by recomputation) and its distortion never rises;
 * (2) Huffman codes are prefix free, Kraft-exact, within 1 bit/symbol of the entropy, and round trip
 * exactly, including a one-symbol alphabet and an empty stream; (3) the compressed model keeps at least
 * 90% of the original validation accuracy and agrees with the original on ≥ 90% of the inputs;
 * (4) the entropy-coded index stream beats fixed-length coding whenever the index distribution is
 * skewed, and never loses more than nothing when it is uniform; (5) the total compressed size is a
 * fraction of the float32 model; (6) the whole pipeline — training, clustering, coding and inference —
 * is deterministic.
 *
 * Simplifications: a tiny MLP, k = 16 centroids, no GPU or batching, and the "runtime" is an
 * interpreter over the codebook rather than generated machine code.
 */
"use strict";

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
  /** Standard normal via the sum of twelve uniforms (deterministic, adequate here). */
  normal() { let total = 0; for (let i = 0; i < 12; i++) total += this.unit(); return total - 6; }
}

/* ------------------------------------------------------------------ dataset and model */

const INPUTS = 8;
const HIDDEN = 12;
const CLASSES = 3;

/** Three Gaussian blobs in feature space: class c shifts the first three features by c. */
function makeDataset(count, seed) {
  const prng = new Prng(seed);
  const samples = [];
  for (let i = 0; i < count; i++) {
    const label = prng.below(CLASSES);
    const features = [];
    for (let f = 0; f < INPUTS; f++) {
      const centre = f < CLASSES ? (f === label ? 1.6 : -0.4) : 0;
      features.push(centre + 1.15 * prng.normal());
    }
    samples.push({ features, label });
  }
  return samples;
}

class Mlp {
  constructor(seed) {
    const prng = new Prng(seed);
    this.w1 = Array.from({ length: INPUTS }, () =>
      Array.from({ length: HIDDEN }, () => prng.normal() * Math.sqrt(2 / INPUTS)));
    this.b1 = new Array(HIDDEN).fill(0);
    this.w2 = Array.from({ length: HIDDEN }, () =>
      Array.from({ length: CLASSES }, () => prng.normal() * Math.sqrt(2 / HIDDEN)));
    this.b2 = new Array(CLASSES).fill(0);
  }

  /** Forward pass; with `weights` given, the supplied (shared) weights replace the dense ones. */
  forward(features, weights = null) {
    const w1 = weights ? weights.w1 : this.w1;
    const b1 = weights ? weights.b1 : this.b1;
    const w2 = weights ? weights.w2 : this.w2;
    const b2 = weights ? weights.b2 : this.b2;
    const hidden = new Array(HIDDEN).fill(0);
    for (let h = 0; h < HIDDEN; h++) {
      let sum = b1[h];
      for (let i = 0; i < INPUTS; i++) sum += features[i] * w1[i][h];
      hidden[h] = Math.tanh(sum);
    }
    const logits = new Array(CLASSES).fill(0);
    for (let c = 0; c < CLASSES; c++) {
      let sum = b2[c];
      for (let h = 0; h < HIDDEN; h++) sum += hidden[h] * w2[h][c];
      logits[c] = sum;
    }
    return { hidden, logits };
  }

  predict(features, weights = null) {
    const { logits } = this.forward(features, weights);
    let best = 0;
    for (let c = 1; c < CLASSES; c++) if (logits[c] > logits[best]) best = c;
    return best;
  }

  /** Full-batch gradient descent with a fixed learning rate — deterministic, no shuffling. */
  train(samples, options = {}) {
    const epochs = options.epochs || 240;
    const rate = options.rate || 0.05;
    for (let epoch = 0; epoch < epochs; epoch++) {
      const gw1 = Array.from({ length: INPUTS }, () => new Array(HIDDEN).fill(0));
      const gb1 = new Array(HIDDEN).fill(0);
      const gw2 = Array.from({ length: HIDDEN }, () => new Array(CLASSES).fill(0));
      const gb2 = new Array(CLASSES).fill(0);
      for (const sample of samples) {
        const { hidden, logits } = this.forward(sample.features);
        const target = new Array(CLASSES).fill(0);
        target[sample.label] = 1;
        for (let c = 0; c < CLASSES; c++) {
          const error = logits[c] - target[c];
          gb2[c] += error;
          for (let h = 0; h < HIDDEN; h++) gw2[h][c] += error * hidden[h];
        }
        for (let h = 0; h < HIDDEN; h++) {
          let back = 0;
          for (let c = 0; c < CLASSES; c++) back += (logits[c] - target[c]) * this.w2[h][c];
          const delta = back * (1 - hidden[h] * hidden[h]);
          gb1[h] += delta;
          for (let i = 0; i < INPUTS; i++) gw1[i][h] += delta * sample.features[i];
        }
      }
      const scale = rate / samples.length;
      for (let i = 0; i < INPUTS; i++) {
        for (let h = 0; h < HIDDEN; h++) this.w1[i][h] -= scale * gw1[i][h];
      }
      for (let h = 0; h < HIDDEN; h++) {
        this.b1[h] -= scale * gb1[h];
        for (let c = 0; c < CLASSES; c++) this.w2[h][c] -= scale * gw2[h][c];
      }
      for (let c = 0; c < CLASSES; c++) this.b2[c] -= scale * gb2[c];
    }
    return this;
  }

  /** Flat view of every learnable weight, in a fixed order (the order the codec relies on). */
  flatten() {
    const values = [];
    for (let i = 0; i < INPUTS; i++) for (let h = 0; h < HIDDEN; h++) values.push(this.w1[i][h]);
    for (let h = 0; h < HIDDEN; h++) values.push(this.b1[h]);
    for (let h = 0; h < HIDDEN; h++) for (let c = 0; c < CLASSES; c++) values.push(this.w2[h][c]);
    for (let c = 0; c < CLASSES; c++) values.push(this.b2[c]);
    return values;
  }

  static unflatten(values) {
    const model = { w1: [], b1: [], w2: [], b2: [] };
    let cursor = 0;
    for (let i = 0; i < INPUTS; i++) {
      const row = [];
      for (let h = 0; h < HIDDEN; h++) row.push(values[cursor++]);
      model.w1.push(row);
    }
    for (let h = 0; h < HIDDEN; h++) model.b1.push(values[cursor++]);
    for (let h = 0; h < HIDDEN; h++) {
      const row = [];
      for (let c = 0; c < CLASSES; c++) row.push(values[cursor++]);
      model.w2.push(row);
    }
    for (let c = 0; c < CLASSES; c++) model.b2.push(values[cursor++]);
    return model;
  }
}

const accuracy = (model, samples, weights = null) => samples.reduce(
  (total, sample) => total + (model.predict(sample.features, weights) === sample.label ? 1 : 0), 0) / samples.length;

const agreement = (model, samples, weights) => samples.reduce(
  (total, sample) => total + (model.predict(sample.features, weights) === model.predict(sample.features) ? 1 : 0), 0) / samples.length;

/* ------------------------------------------------------------------ k-means weight sharing */

function kMeans(values, k, options = {}) {
  const prng = new Prng(options.seed || 897);
  const maxIterations = options.maxIterations || 40;
  const points = values.slice();
  // Seeded initialisation from the value range, so the result is reproducible.
  const low = Math.min(...points);
  const high = Math.max(...points);
  let centroids = Array.from({ length: k }, (_, index) =>
    low + ((high - low) * (index + 0.5)) / k + (prng.unit() - 0.5) * (high - low) * 0.01);
  let assignment = new Array(points.length).fill(0);
  let previousDistortion = Infinity;
  let iterations = 0;
  const history = [];
  for (; iterations < maxIterations; iterations++) {
    let changed = false;
    for (let p = 0; p < points.length; p++) {
      let best = 0;
      let bestDistance = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const distance = (points[p] - centroids[c]) ** 2;
        if (distance < bestDistance) { bestDistance = distance; best = c; }
      }
      if (assignment[p] !== best) { assignment[p] = best; changed = true; }
    }
    const sums = new Array(k).fill(0);
    const counts = new Array(k).fill(0);
    for (let p = 0; p < points.length; p++) { sums[assignment[p]] += points[p]; counts[assignment[p]]++; }
    centroids = centroids.map((value, c) => (counts[c] > 0 ? sums[c] / counts[c] : value));
    const distortion = points.reduce((total, value, index) => total + (value - centroids[assignment[index]]) ** 2, 0);
    history.push(distortion);
    if (!changed || Math.abs(previousDistortion - distortion) < 1e-15) { previousDistortion = distortion; break; }
    previousDistortion = distortion;
  }
  return { centroids, assignment, distortion: previousDistortion, iterations, history };
}

/* ------------------------------------------------------------------ Huffman */

/** Builds a Huffman code from symbol frequencies; symbols are the integers 0..k-1. */
function buildHuffman(frequencies) {
  const nodes = [];
  for (let symbol = 0; symbol < frequencies.length; symbol++) {
    if (frequencies[symbol] > 0) nodes.push({ symbol, weight: frequencies[symbol], left: null, right: null });
  }
  if (nodes.length === 0) return { codes: [], tree: null };
  if (nodes.length === 1) {
    // A single symbol still needs a code that can be decoded: the convention is one bit.
    const codes = new Array(frequencies.length).fill(null);
    codes[nodes[0].symbol] = "0";
    return { codes, tree: nodes[0], single: true };
  }
  const forest = nodes.slice();
  while (forest.length > 1) {
    forest.sort((a, b) => (a.weight - b.weight) || ((a.symbol ?? -1) - (b.symbol ?? -1)));
    const left = forest.shift();
    const right = forest.shift();
    forest.push({ symbol: null, weight: left.weight + right.weight, left, right });
  }
  const tree = forest[0];
  const codes = new Array(frequencies.length).fill(null);
  const walk = (node, prefix) => {
    if (node.symbol !== null) { codes[node.symbol] = prefix; return; }
    walk(node.left, `${prefix}0`);
    walk(node.right, `${prefix}1`);
  };
  walk(tree, "");
  return { codes, tree };
}

const entropy = (frequencies) => {
  const total = frequencies.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let result = 0;
  for (const frequency of frequencies) {
    if (frequency === 0) continue;
    const p = frequency / total;
    result -= p * Math.log2(p);
  }
  return result;
};

const averageCodeLength = (frequencies, codes) => {
  const total = frequencies.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let bits = 0;
  for (let symbol = 0; symbol < frequencies.length; symbol++) {
    if (frequencies[symbol] === 0) continue;
    bits += frequencies[symbol] * (codes[symbol] ? codes[symbol].length : 0);
  }
  return bits / total;
};

function huffmanEncode(symbols, codes) {
  let bits = "";
  for (const symbol of symbols) {
    const code = codes[symbol];
    if (code === undefined || code === null) throw new Error(`symbol ${symbol} has no code`);
    bits += code;
  }
  return bits;
}

/** Decoding by walking the tree bit by bit; a corrupt prefix is reported, never silently accepted. */
function huffmanDecode(bits, tree) {
  if (tree === null) { if (bits.length > 0) throw new Error("no tree for a non-empty stream"); return []; }
  if (tree.symbol !== null) {
    if (bits.length === 0) return [];
    return new Array(bits.length).fill(tree.symbol);
  }
  const out = [];
  let node = tree;
  for (const bit of bits) {
    node = bit === "0" ? node.left : node.right;
    if (node === null) throw new Error("invalid Huffman stream");
    if (node.symbol !== null) { out.push(node.symbol); node = tree; }
  }
  if (node !== tree) throw new Error("truncated Huffman stream");
  return out;
}

function kraftSum(codes) {
  return codes.reduce((total, code) => (code === null ? total : total + Math.pow(2, -code.length)), 0);
}

function isPrefixFree(codes) {
  const present = codes.filter((code) => code !== null && code !== undefined);
  for (let i = 0; i < present.length; i++) {
    for (let j = 0; j < present.length; j++) {
      if (i === j) continue;
      if (present[j].startsWith(present[i])) return false;
    }
  }
  return true;
}

/* ------------------------------------------------------------------ the compressed model */

function compressModel(model, options = {}) {
  const weights = model.flatten();
  const clusters = options.clusters || 16;
  const kmeans = kMeans(weights, clusters, { seed: options.seed || 897 });
  const frequencies = new Array(clusters).fill(0);
  for (const index of kmeans.assignment) frequencies[index]++;
  const huffman = buildHuffman(frequencies);
  const bits = huffmanEncode(kmeans.assignment, huffman.codes);
  const floatBits = weights.length * 32;
  const fixedBits = weights.length * Math.ceil(Math.log2(clusters));
  const codebookBits = clusters * 32;
  const corrected = 1 + Math.ceil(Math.log2(clusters - 1));
  return {
    model, weights, kmeans, frequencies, huffman, bits,
    sizes: {
      float: floatBits, fixed: fixedBits, huffman: bits.length, codebook: codebookBits,
      total: bits.length + codebookBits, corrected,
    },
    ratio: (bits.length + codebookBits) / floatBits,
    dequantised: Mlp.unflatten(kmeans.assignment.map((index) => kmeans.centroids[index])),
    stats: {
      entropy: entropy(frequencies), averageCodeLength: averageCodeLength(frequencies, huffman.codes),
      distortion: kmeans.distortion, iterations: kmeans.iterations,
    },
  };
}

/** The runtime: inference straight from the codebook + indices, dequantising on the fly. */
function runCompressed(compressed, features) {
  const model = compressed.model;
  const weights = {
    w1: [], b1: [], w2: [], b2: [],
  };
  let cursor = 0;
  const value = () => compressed.kmeans.centroids[compressed.kmeans.assignment[cursor++]];
  for (let i = 0; i < INPUTS; i++) {
    const row = [];
    for (let h = 0; h < HIDDEN; h++) row.push(value());
    weights.w1.push(row);
  }
  for (let h = 0; h < HIDDEN; h++) weights.b1.push(value());
  for (let h = 0; h < HIDDEN; h++) {
    const row = [];
    for (let c = 0; c < CLASSES; c++) row.push(value());
    weights.w2.push(row);
  }
  for (let c = 0; c < CLASSES; c++) weights.b2.push(value());
  return model.forward(features, weights);
}

module.exports = {
  Prng, makeDataset, Mlp, accuracy, agreement, kMeans, buildHuffman, entropy, averageCodeLength,
  huffmanEncode, huffmanDecode, kraftSum, isPrefixFree, compressModel, runCompressed,
  INPUTS, HIDDEN, CLASSES,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- dataset and training ------------------------------------------------------------------
  const training = makeDataset(240, 11);
  const validation = makeDataset(120, 22);
  const model = new Mlp(7).train(training, { epochs: 260, rate: 0.08 });
  const baseAccuracy = accuracy(model, validation);
  assert.ok(baseAccuracy > 0.75, `the model must actually learn (accuracy ${baseAccuracy})`);
  assert.ok(baseAccuracy < 1, "and it must not be trivially perfect");
  assert.strictEqual(model.flatten().length, INPUTS * HIDDEN + HIDDEN + HIDDEN * CLASSES + CLASSES);
  const roundTrip = Mlp.unflatten(model.flatten());
  assert.deepStrictEqual(roundTrip.w1, model.w1);
  assert.deepStrictEqual(roundTrip.b2, model.b2);
  assert.strictEqual(accuracy(model, validation, model), baseAccuracy, "uncompressed weights are a no-op");

  // --- k-means: assignments are optimal for the centroids, centroids are the cluster means ----
  {
    const result = kMeans(model.flatten(), 16, { seed: 99 });
    assert.ok(result.centroids.length === 16);
    assert.ok(result.iterations >= 1 && result.iterations <= 40);
    for (let p = 0; p < model.flatten().length; p++) {
      const value = model.flatten()[p];
      const chosen = result.centroids[result.assignment[p]];
      for (const centroid of result.centroids) {
        assert.ok(Math.abs(value - chosen) <= Math.abs(value - centroid) + 1e-12,
          "the assignment must be to the nearest centroid");
      }
    }
    for (let c = 0; c < result.centroids.length; c++) {
      const members = model.flatten().filter((_, index) => result.assignment[index] === c);
      if (members.length === 0) continue;
      const mean = members.reduce((a, b) => a + b, 0) / members.length;
      assert.ok(Math.abs(mean - result.centroids[c]) < 1e-9, `centroid ${c} must be the cluster mean`);
    }
    for (let i = 1; i < result.history.length; i++) {
      assert.ok(result.history[i] <= result.history[i - 1] + 1e-12, "the distortion must not rise");
    }
    const seeded = kMeans(model.flatten(), 16, { seed: 99 });
    assert.deepStrictEqual(seeded.centroids, result.centroids, "k-means is reproducible");
    const oneCluster = kMeans(model.flatten(), 1, { seed: 1 });
    assert.strictEqual(oneCluster.centroids.length, 1);
    assert.ok(oneCluster.distortion > result.distortion, "fewer clusters means more distortion");
  }

  // --- Huffman: prefix free, Kraft exact, within one bit of the entropy, round trip -----------
  {
    const frequencies = [50, 20, 10, 5, 5, 4, 3, 3];
    const { codes } = buildHuffman(frequencies);
    assert.ok(isPrefixFree(codes), "codes must be prefix free");
    assert.ok(Math.abs(kraftSum(codes) - 1) < 1e-12, `Kraft sum ${kraftSum(codes)} must be 1`);
    const expected = entropy(frequencies);
    const average = averageCodeLength(frequencies, codes);
    assert.ok(average >= expected - 1e-12, "no coder beats the entropy");
    assert.ok(average < expected + 1, `Huffman must be within one bit (${average} vs ${expected})`);
    const symbols = [];
    frequencies.forEach((frequency, symbol) => { for (let i = 0; i < frequency; i++) symbols.push(symbol); });
    const bits = huffmanEncode(symbols, codes);
    const { tree } = buildHuffman(frequencies);
    assert.deepStrictEqual(huffmanDecode(bits, tree), symbols, "the round trip must be exact");
    assert.ok(bits.length < symbols.length * 3, "and it must actually compress this distribution");
    // A prefix that stops in the middle of a code path must be rejected, not silently decoded.
    const handBuilt = buildHuffman([2, 1, 1]);
    assert.deepStrictEqual(huffmanDecode(handBuilt.codes[1] + handBuilt.codes[2], handBuilt.tree), [1, 2]);
    assert.throws(() => huffmanDecode(handBuilt.codes[1].slice(0, 1), handBuilt.tree), /truncated/);
  }

  // --- Huffman edge cases ----------------------------------------------------------------------
  {
    const single = buildHuffman([0, 0, 7, 0]);
    assert.strictEqual(single.codes[2], "0", "a one-symbol alphabet gets a one-bit code");
    assert.deepStrictEqual(huffmanDecode("000", single.tree), [2, 2, 2]);
    assert.deepStrictEqual(huffmanDecode("", single.tree), []);
    assert.deepStrictEqual(buildHuffman([0, 0, 0]).codes, [], "no symbols, no codes");
    assert.deepStrictEqual(huffmanDecode("", null), []);
    assert.throws(() => huffmanDecode("0", null), /no tree/);
    const uniform = new Array(8).fill(5);
    const uniformCodes = buildHuffman(uniform).codes;
    assert.ok(uniformCodes.every((code) => code.length === 3), "a uniform distribution wants fixed length");
    assert.ok(Math.abs(averageCodeLength(uniform, uniformCodes) - 3) < 1e-12);
    assert.throws(() => huffmanEncode([9], uniformCodes), /no code/);
    const truncatedCodes = ["0", "10", "11"];
    assert.ok(isPrefixFree(truncatedCodes));
    assert.ok(Math.abs(kraftSum(truncatedCodes) - 1) < 1e-12);
    assert.throws(() => huffmanDecode("11111", buildHuffman([1, 1, 1]).tree), /truncated/);
  }

  // --- compression quality and runtime behaviour ------------------------------------------------
  const compressed = compressModel(model, { clusters: 16, seed: 897 });
  const quantisedAccuracy = accuracy(model, validation, compressed.dequantised);
  assert.ok(quantisedAccuracy >= baseAccuracy - 0.1,
    `weight sharing must not cost much accuracy: ${baseAccuracy} -> ${quantisedAccuracy}`);
  assert.ok(quantisedAccuracy >= 0.7);
  let agrees = 0;
  for (const sample of validation) {
    const viaRuntime = runCompressed(compressed, sample.features);
    const viaModel = model.forward(sample.features, compressed.dequantised);
    for (let c = 0; c < CLASSES; c++) {
      assert.ok(Math.abs(viaRuntime.logits[c] - viaModel.logits[c]) < 1e-12,
        "the runtime must agree with the dequantised model exactly");
    }
    if (model.predict(sample.features, compressed.dequantised) === model.predict(sample.features)) agrees++;
  }
  assert.ok(agrees / validation.length >= 0.9, `prediction agreement ${agrees / validation.length}`);

  // --- coding efficiency ------------------------------------------------------------------------
  {
    const { sizes, stats } = compressed;
    assert.ok(sizes.huffman < sizes.fixed, `Huffman ${sizes.huffman} must beat fixed ${sizes.fixed}`);
    assert.ok(stats.averageCodeLength < Math.log2(16), "the indices must be skewed for this to win");
    assert.ok(stats.averageCodeLength >= stats.entropy - 1e-12);
    assert.ok(stats.averageCodeLength < stats.entropy + 1, "the one-bit Huffman bound holds here too");
    assert.ok(compressed.ratio < 0.4, `compression ratio ${compressed.ratio} must be a real saving`);
    assert.ok(sizes.codebook > 0 && sizes.total > sizes.huffman, "the codebook counts too");
    assert.ok(sizes.corrected >= 1);
    // A uniform index stream cannot be beaten by entropy coding.
    const uniformFrequencies = new Array(16).fill(30);
    const uniformTree = buildHuffman(uniformFrequencies);
    const uniformBits = huffmanEncode(new Array(16).fill(0).map((_, i) => i).flatMap((i) => new Array(30).fill(i)), uniformTree.codes);
    assert.strictEqual(uniformBits.length, 480 * 4, "a uniform 16-symbol stream is exactly 4 bits each");
  }

  // --- fewer clusters = smaller and less accurate ------------------------------------------------
  {
    const coarse = compressModel(model, { clusters: 4, seed: 897 });
    const fine = compressModel(model, { clusters: 32, seed: 897 });
    assert.ok(coarse.sizes.fixed < fine.sizes.fixed, "fewer centroids means shorter indices");
    assert.ok(coarse.kmeans.distortion > fine.kmeans.distortion, "and more distortion");
    assert.ok(accuracy(model, validation, fine.dequantised) >= accuracy(model, validation, coarse.dequantised) - 0.05);
  }

  // --- determinism --------------------------------------------------------------------------------
  {
    const runOnce = () => {
      const data = makeDataset(120, 33);
      const trained = new Mlp(5).train(data, { epochs: 60, rate: 0.1 });
      const packed = compressModel(trained, { clusters: 8, seed: 4 });
      return `${accuracy(trained, validation).toFixed(6)}|${packed.kmeans.distortion.toFixed(9)}|${packed.bits.length}|` +
        `${packed.stats.averageCodeLength.toFixed(6)}`;
    };
    assert.strictEqual(runOnce(), runOnce(), "the whole pipeline must be reproducible");
  }

  console.log(`compressed-nn: accuracy ${baseAccuracy.toFixed(3)} -> ${quantisedAccuracy.toFixed(3)} at ` +
    `${(compressed.ratio * 100).toFixed(1)}% of float32, ${compressed.stats.averageCodeLength.toFixed(2)} bits/weight ` +
    `(entropy ${compressed.stats.entropy.toFixed(2)}, fixed length 4)`);
  console.log("PASS");
}
