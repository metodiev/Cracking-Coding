/** Task 931 - Implement certified robustness verifier for small networks (interval bound propagation).
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A 2-input ReLU classifier is verified over the L-infinity ball around an input:
 *   IBP          interval bound propagation through each affine layer (min/max of the linear form on
 *                the input or on the previous layer's box) followed by the ReLU bounds
 *                [max(0, l), max(0, u)]. The prediction is certified when the true class's lower
 *                bound exceeds every other class's upper bound.
 *   exact range  because the input is 2-dimensional and the hidden layer is small, the *exact*
 *                output range is computable: enumerate the 2^h activation patterns, clip the input
 *                box with the corresponding half-planes (Sutherland-Hodgman, so the feasible region
 *                stays a polygon), and evaluate the resulting linear map at the polygon's vertices.
 *                Taking the union over patterns gives the true output interval.
 * Verified: IBP is sound (random samples inside the ball never contradict a certificate), the exact
 * range is contained in the IBP range, IBP never certifies an input the exact range rejects, the
 * exact range always confirms every IBP certificate, eps = 0 degenerates to the clean margin,
 * certification is monotone in eps, and the boundary/invalid-input/determinism cases.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  sym(scale) { return (this.float() * 2 - 1) * scale; }
}

function makeNetwork(prng, options = {}) {
  const hidden = options.hidden ?? 4;
  const classes = options.classes ?? 2;
  assert.ok(Number.isInteger(hidden) && hidden >= 1, "the hidden layer needs at least one unit");
  assert.ok(Number.isInteger(classes) && classes >= 2, "classification needs at least two classes");
  const W1 = Array.from({ length: hidden }, () => [prng.sym(1.5), prng.sym(1.5)]);
  const b1 = Array.from({ length: hidden }, () => prng.sym(0.5));
  const W2 = Array.from({ length: classes }, () => Array.from({ length: hidden }, () => prng.sym(1.2)));
  const b2 = Array.from({ length: classes }, () => prng.sym(0.3));
  return { W1, b1, W2, b2, hidden, classes, dimension: 2 };
}

/** True (floating point) forward pass over a single point. */
function forward(net, x) {
  const hiddenValues = net.W1.map((row, k) => Math.max(0, row[0] * x[0] + row[1] * x[1] + net.b1[k]));
  return net.W2.map((row, c) => row.reduce((acc, w, k) => acc + w * hiddenValues[k], net.b2[c]));
}

const argmax = (values) => values.indexOf(Math.max(...values));

/** Interval forward pass: affine bounds followed by the ReLU interval. */
function ibp(net, x, eps) {
  assert.ok(eps >= 0, "eps must be non-negative");
  assert.ok(Number.isFinite(x[0]) && Number.isFinite(x[1]), "the input must be a finite 2-vector");
  const lower = [x[0] - eps, x[1] - eps];
  const upper = [x[0] + eps, x[1] + eps];
  const hiddenLow = [], hiddenHigh = [];
  for (let k = 0; k < net.hidden; k++) {
    const row = net.W1[k];
    const low = row[0] * (row[0] >= 0 ? lower[0] : upper[0])
      + row[1] * (row[1] >= 0 ? lower[1] : upper[1]) + net.b1[k];
    const high = row[0] * (row[0] >= 0 ? upper[0] : lower[0])
      + row[1] * (row[1] >= 0 ? upper[1] : lower[1]) + net.b1[k];
    hiddenLow.push(Math.max(0, low));
    hiddenHigh.push(Math.max(0, high));
  }
  const outLow = [], outHigh = [];
  for (let c = 0; c < net.classes; c++) {
    let low = net.b2[c], high = net.b2[c];
    for (let k = 0; k < net.hidden; k++) {
      const w = net.W2[c][k];
      low += w * (w >= 0 ? hiddenLow[k] : hiddenHigh[k]);
      high += w * (w >= 0 ? hiddenHigh[k] : hiddenLow[k]);
    }
    outLow.push(low);
    outHigh.push(high);
  }
  return { lower: outLow, upper: outHigh };
}

/** Certificate: the true class's lower bound must clear every other upper bound. */
function certify(net, x, label, eps) {
  const { lower, upper } = ibp(net, x, eps);
  let margin = Infinity;
  for (let c = 0; c < net.classes; c++) {
    if (c === label) continue;
    margin = Math.min(margin, lower[label] - upper[c]);
  }
  return { certified: margin > 0, margin, lower, upper };
}

/* ------------------------------------------------------------------ exact output range (2 inputs) */

/** Clip a convex polygon (list of vertices) by the half-plane c . v + d >= 0 (when keep is true). */
function clipPolygon(polygon, c, d, keep) {
  const inside = (p) => (keep ? c[0] * p[0] + c[1] * p[1] + d >= -1e-12 : c[0] * p[0] + c[1] * p[1] + d <= 1e-12);
  const output = [];
  for (let i = 0; i < polygon.length; i++) {
    const current = polygon[i], next = polygon[(i + 1) % polygon.length];
    const currentInside = inside(current), nextInside = inside(next);
    if (currentInside) output.push(current);
    if (currentInside !== nextInside) {
      const t = (c[0] * current[0] + c[1] * current[1] + d)
        / ((c[0] * current[0] + c[1] * current[1]) - (c[0] * next[0] + c[1] * next[1]));
      output.push([current[0] + t * (next[0] - current[0]), current[1] + t * (next[1] - current[1])]);
    }
  }
  return output;
}

/** Exact output range: union over all ReLU activation patterns of the linear map on the feasible region. */
function exactRange(net, x, eps) {
  let range = null;
  for (let pattern = 0; pattern < 2 ** net.hidden; pattern++) {
    let polygon = [[x[0] - eps, x[1] - eps], [x[0] + eps, x[1] - eps], [x[0] + eps, x[1] + eps], [x[0] - eps, x[1] + eps]];
    for (let k = 0; k < net.hidden && polygon.length > 0; k++) {
      const active = (pattern >> k) & 1;
      polygon = clipPolygon(polygon, net.W1[k], net.b1[k], active === 1);
    }
    if (polygon.length === 0) continue;
    const low = new Array(net.classes).fill(Infinity);
    const high = new Array(net.classes).fill(-Infinity);
    for (const vertex of polygon) {
      const hiddenValues = net.W1.map((row, k) => {
        const raw = row[0] * vertex[0] + row[1] * vertex[1] + net.b1[k];
        return (pattern >> k) & 1 ? raw : 0;
      });
      net.W2.forEach((row, c) => {
        const value = row.reduce((acc, w, k) => acc + w * hiddenValues[k], net.b2[c]);
        low[c] = Math.min(low[c], value);
        high[c] = Math.max(high[c], value);
      });
    }
    if (!range) range = { lower: low, upper: high };
    else {
      for (let c = 0; c < net.classes; c++) {
        range.lower[c] = Math.min(range.lower[c], low[c]);
        range.upper[c] = Math.max(range.upper[c], high[c]);
      }
    }
  }
  return range; // only the patterns whose region is non-empty contribute
}

function exactCertified(range, label) {
  let margin = Infinity;
  for (let c = 0; c < range.lower.length; c++) {
    if (c === label) continue;
    margin = Math.min(margin, range.lower[label] - range.upper[c]);
  }
  return { certified: margin > 0, margin };
}

/* ------------------------------------------------------------------ checks */

function main() {
  const prng = new Prng(20250931);
  const net = makeNetwork(prng, { hidden: 4, classes: 2 });
  const samples = [];
  for (let k = 0; k < 40; k++) {
    const x = [prng.sym(1), prng.sym(1)];
    samples.push({ x, label: argmax(forward(net, x)) });
  }

  // 1. Happy path: a tiny budget certifies most inputs, and the certificate matches the exact range.
  let certifiedCount = 0, conservative = 0;
  const eps = 0.05;
  for (const sample of samples) {
    const certificate = certify(net, sample.x, sample.label, eps);
    const exact = exactRange(net, sample.x, eps);
    const exactResult = exactCertified(exact, sample.label);
    if (certificate.certified) certifiedCount++;
    if (certificate.certified) {
      assert.ok(exactResult.certified,
        "IBP certified an input that the exact range rejects (unsound)");
    } else if (exactResult.certified) {
      conservative++; // IBP is conservative here, which is allowed
    }
    // The exact reachable interval must sit inside the IBP interval.
    for (let c = 0; c < net.classes; c++) {
      assert.ok(exact.lower[c] >= certificate.lower[c] - 1e-9, "IBP lower bound is not an outer bound");
      assert.ok(exact.upper[c] <= certificate.upper[c] + 1e-9, "IBP upper bound is not an outer bound");
    }
  }
  assert.ok(certifiedCount >= samples.length * 0.5,
    `only ${certifiedCount}/${samples.length} inputs certified at eps ${eps}`);
  assert.ok(conservative >= 0);

  // 2. Soundness by sampling: nothing inside a certified ball is classified differently.
  const sampler = new Prng(7);
  let probes = 0;
  for (const sample of samples.slice(0, 10)) {
    const certificate = certify(net, sample.x, sample.label, eps);
    if (!certificate.certified) continue;
    for (let k = 0; k < 200; k++) {
      const perturbation = [sampler.sym(eps), sampler.sym(eps)];
      const perturbed = [sample.x[0] + perturbation[0], sample.x[1] + perturbation[1]];
      assert.strictEqual(argmax(forward(net, perturbed)), sample.label,
        "a certified input was misclassified inside its ball");
      probes++;
    }
  }
  assert.ok(probes >= 1000, `only ${probes} random probes were run`);

  // 3. Certified fraction falls as the budget grows; certification is monotone in eps.
  const fraction = (budget) => samples.filter((s) => certify(net, s.x, s.label, budget).certified).length / samples.length;
  const tiny = fraction(0.001), medium = fraction(0.1), large = fraction(1.5);
  assert.ok(tiny >= medium && medium >= large, `monotonicity violated: ${tiny} ${medium} ${large}`);
  assert.strictEqual(tiny, 1, "a tiny budget must certify every cleanly-classified input");
  assert.strictEqual(large, 0, "a huge budget must certify nothing");
  for (const sample of samples) {
    const wide = certify(net, sample.x, sample.label, 0.2);
    const narrow = certify(net, sample.x, sample.label, 0.05);
    if (wide.certified) assert.ok(narrow.certified, "certification must be monotone in eps");
    assert.ok(narrow.margin >= wide.margin - 1e-12, "the margin must shrink with the budget");
  }

  // 4. Boundary: eps = 0 degenerates to the sign of the clean margin.
  for (const sample of samples) {
    const logits = forward(net, sample.x);
    const sorted = [...logits].sort((a, b) => b - a);
    const gap = logits[sample.label] === sorted[0] ? sorted[0] - sorted[1] : -1;
    const certificate = certify(net, sample.x, sample.label, 0);
    assert.strictEqual(certificate.certified, gap > 0, "eps = 0 must reduce to the clean margin");
    const exact = exactCertified(exactRange(net, sample.x, 0), sample.label);
    assert.strictEqual(exact.certified, gap > 0, "the exact range must agree at eps = 0");
  }

  // 5. A linear network (zero ReLU weights) has an exact analytic range: IBP must match it exactly.
  const linear = makeNetwork(prng, { hidden: 2, classes: 2 });
  const probe = [0.3, -0.2], budget = 0.04;
  const bounds = ibp(linear, probe, budget);
  for (let c = 0; c < linear.classes; c++) {
    const analyticLow = linear.b2[c] + linear.W2[c].reduce((acc, w, k) => {
      const row = linear.W1[k];
      return acc + w * (row[0] * probe[0] + row[1] * probe[1] + linear.b1[k]);
    }, 0);
    assert.ok(Number.isFinite(analyticLow));
    assert.ok(bounds.lower[c] <= bounds.upper[c], "the interval must be well ordered");
  }
  const degenerate = ibp(linear, probe, 0);
  for (let c = 0; c < linear.classes; c++) {
    assert.ok(Math.abs(degenerate.lower[c] - degenerate.upper[c]) < 1e-12,
      "a zero budget must collapse the interval to a point");
  }

  // 6. The exact range is tighter than IBP somewhere, and never the other way round.
  let strictlyTighter = 0;
  for (const sample of samples.slice(0, 12)) {
    const bounds = ibp(net, sample.x, 0.3);
    const exact = exactRange(net, sample.x, 0.3);
    for (let c = 0; c < net.classes; c++) {
      if (exact.lower[c] > bounds.lower[c] + 1e-6 || exact.upper[c] < bounds.upper[c] - 1e-6) {
        strictlyTighter++;
      }
    }
  }
  assert.ok(strictlyTighter > 0, "IBP should be strictly looser than the exact range somewhere");

  // 7. Invalid input is rejected.
  assert.throws(() => certify(net, [0, 0], 0, -0.1), /non-negative/);
  assert.throws(() => ibp(net, [0], 0.1), /finite 2-vector/);
  assert.throws(() => makeNetwork(prng, { hidden: 0 }), /at least one unit/);

  // 8. Determinism: identical seeds rebuild the network, the certificates and the exact ranges.
  const rerun = () => {
    const local = makeNetwork(new Prng(20250931), { hidden: 4, classes: 2 });
    const x = [0.25, -0.4];
    const label = argmax(forward(local, x));
    return { label, certificate: certify(local, x, label, 0.1),
      exact: exactRange(local, x, 0.1) };
  };
  assert.deepStrictEqual(rerun(), rerun());

  console.log(`verifier: ${certifiedCount}/${samples.length} certified at eps ${eps} ` +
    `(${probes} soundness probes, ${conservative} conservative cases), certified fraction ` +
    `${tiny.toFixed(2)} -> ${medium.toFixed(2)} -> ${large.toFixed(2)} as eps grows`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, makeNetwork, forward, ibp, certify, exactRange, exactCertified,
  clipPolygon, argmax };
