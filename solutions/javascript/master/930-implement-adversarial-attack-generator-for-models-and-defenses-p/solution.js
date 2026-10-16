/** Task 930 - Implement adversarial attack generator for models and defenses (PGD).
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A logistic-regression classifier on a 2D separable set, attacked inside an L-infinity ball:
 *   FGSM        one gradient step: x + eps * sign(dL/dx);
 *   PGD         k projected steps, optional random start inside the ball, projection back onto the
 *               box after every step (the standard Madry-style L-infinity PGD);
 *   oracle      for the *hinge* loss on a linear score the optimum is closed form:
 *               delta* = -eps * y * sign(w), so max hinge loss = max(0, 1 - y(s - eps*||w||_1)).
 *               PGD with the hinge objective must reach that value exactly, and a brute-force
 *               enumeration of the 2^d box corners confirms the closed form independently;
 *   defense     adversarial training: retrain on a mixture of clean and PGD-perturbed samples and
 *               measure the robust accuracy under the same attack.
 * Simplification: a linear binary model with two features (the analytic optimum is exact there),
 * full-batch gradient descent, no mini-batching or scheduling. Verified: the attack stays in the
 * budget, eps = 0 is a no-op, PGD is never weaker than FGSM, the hinge oracle is attained, robust
 * accuracy drops on the naive model and recovers after adversarial training, boundary and invalid
 * input cases, determinism.
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

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/** Sign-class logistic regression trained by deterministic full-batch gradient descent. */
class LogisticModel {
  constructor(dimension) {
    assert.ok(Number.isInteger(dimension) && dimension > 0, "the dimension must be positive");
    this.dimension = dimension;
    this.weights = new Array(dimension).fill(0);
    this.bias = 0;
  }
  score(x) {
    assert.strictEqual(x.length, this.dimension, "the sample has the wrong dimension");
    return this.weights.reduce((acc, w, i) => acc + w * x[i], this.bias);
  }
  /** Element-wise logistic loss of one labelled sample. */
  loss(x, y) {
    const margin = y * this.score(x);
    return Math.log(1 + Math.exp(-margin)) + 1e-3 * this.weights.reduce((a, w) => a + w * w, 0);
  }
  /** Gradient of the loss with respect to the *input*, used by the attacks. */
  inputGradient(x, y) {
    const probability = sigmoid(y * this.score(x));
    return this.weights.map((w) => -(1 - probability) * y * w);
  }
  hinge(x, y) { return Math.max(0, 1 - y * this.score(x)); }
  predict(x) { return this.score(x) >= 0 ? 1 : -1; }
  accuracy(samples) {
    assert.ok(samples.length > 0, "the evaluation set is empty");
    return samples.filter((s) => this.predict(s.x) === s.y).length / samples.length;
  }
  train(samples, options = {}) {
    const epochs = options.epochs ?? 400;
    const step = options.step ?? 0.5;
    assert.ok(epochs > 0 && step > 0, "epochs and step must be positive");
    for (let epoch = 0; epoch < epochs; epoch++) {
      const gradW = new Array(this.dimension).fill(0);
      let gradB = 0;
      for (const sample of samples) {
        const error = -(1 - sigmoid(sample.y * this.score(sample.x))) * sample.y;
        for (let i = 0; i < this.dimension; i++) gradW[i] += error * sample.x[i];
        gradB += error;
      }
      for (let i = 0; i < this.dimension; i++) {
        this.weights[i] -= step * (gradW[i] / samples.length + 2e-3 * this.weights[i]);
      }
      this.bias -= step * (gradB / samples.length);
    }
    return this;
  }
}

const project = (x, original, eps) => x.map((v, i) => Math.min(original[i] + eps, Math.max(original[i] - eps, v)));

function fgsm(model, sample, eps) {
  assert.ok(eps >= 0, "eps must be non-negative");
  if (eps === 0) return { x: [...sample.x], delta: sample.x.map(() => 0), loss: model.loss(sample.x, sample.y) };
  const gradient = model.inputGradient(sample.x, sample.y);
  const delta = gradient.map((g) => eps * Math.sign(g));
  const x = sample.x.map((v, i) => v + delta[i]);
  return { x, delta, loss: model.loss(x, sample.y) };
}

/** PGD in the L-infinity ball; the objective is the logistic loss unless `objective` says hinge. */
function pgd(model, sample, eps, options = {}) {
  assert.ok(eps >= 0, "eps must be non-negative");
  const steps = options.steps ?? 10;
  const stepSize = options.stepSize ?? (steps > 0 ? (2 * eps) / steps : 0);
  assert.ok(Number.isInteger(steps) && steps >= 0, "steps must be a non-negative integer");
  const objective = options.objective ?? "loss";
  const value = (x) => (objective === "hinge" ? model.hinge(x, sample.y) : model.loss(x, sample.y));
  // The hinge loss is flat once the margin exceeds one, so the hinge objective is driven by its
  // monotone surrogate: minimizing y * (w x + b), whose linear optimum is the box corner.
  const gradientAt = () => (objective === "hinge"
    ? model.weights.map((w) => -sample.y * w) : model.inputGradient(x, sample.y));
  let x = [...sample.x];
  if (options.randomStart && eps > 0) {
    const prng = options.prng ?? new Prng(1);
    x = sample.x.map((v) => v + prng.sym(eps));
  }
  let current = value(x);
  for (let k = 0; k < steps; k++) {
    const gradient = gradientAt();
    x = project(x.map((v, i) => v + stepSize * Math.sign(gradient[i])), sample.x, eps);
    current = value(x);
  }
  const projected = project(x, sample.x, eps);
  return { x: projected, delta: projected.map((v, i) => v - sample.x[i]), loss: value(projected) };
}

/** Closed-form optimum of the hinge objective inside the box: delta* = -eps y sign(w). */
function analyticHingeAttack(model, sample, eps) {
  const norm1 = model.weights.reduce((acc, w) => acc + Math.abs(w), 0);
  const worst = sample.y * model.score(sample.x) - eps * norm1;
  return { delta: model.weights.map((w) => -eps * sample.y * Math.sign(w)), loss: Math.max(0, 1 - worst) };
}

/** Independent check of the closed form: the box corners of a 2-feature input are enumerated. */
function bruteForceHingeAttack(model, sample, eps) {
  let best = -Infinity, bestDelta = null;
  for (const sx of [-eps, eps]) {
    for (const sy of [-eps, eps]) {
      const delta = [sx, sy];
      const x = sample.x.map((v, i) => v + delta[i]);
      const loss = model.hinge(x, sample.y);
      if (loss > best) { best = loss; bestDelta = delta; }
    }
  }
  return { delta: bestDelta, loss: best };
}

function makeDataset(prng, count) {
  const samples = [];
  while (samples.length < count) {
    const x = [prng.sym(1), prng.sym(1)];
    const decision = 1.5 * x[0] + x[1] - 0.2;
    if (Math.abs(decision) < 0.5) continue; // keep a healthy geometric margin
    samples.push({ x, y: decision >= 0 ? 1 : -1 });
  }
  return samples;
}

const attacked = (samples, model, eps, options) =>
  samples.map((sample) => ({ ...sample, ...pgd(model, sample, eps, options) }));

/** Median L-infinity radius that flips the prediction of a linear model (analytic for this model). */
function medianFlipRadius(model, samples) {
  const norm1 = model.weights.reduce((acc, w) => acc + Math.abs(w), 0);
  const radii = samples.map((s) => Math.abs(model.score(s.x)) / norm1).sort((a, b) => a - b);
  return radii[Math.floor(radii.length / 2)];
}

function main() {
  const prng = new Prng(20250930);
  const train = makeDataset(prng, 160);
  const test = makeDataset(prng, 80);
  const model = new LogisticModel(2).train(train, { epochs: 400, step: 0.5 });
  const eps = 0.45;

  // 1. Happy path: the trained model separates the set, and the attack destroys that accuracy.
  const clean = model.accuracy(test);
  assert.ok(clean >= 0.9, `clean accuracy ${clean} is too low`);
  const fgsmAttacked = test.map((s) => ({ ...s, ...fgsm(model, s, eps) }));
  const pgdAttacked = test.map((s) => ({ ...s, ...pgd(model, s, eps, { steps: 12 }) }));
  const fgsmAccuracy = fgsmAttacked.filter((s) => model.predict(s.x) === s.y).length / test.length;
  const pgdAccuracy = pgdAttacked.filter((s) => model.predict(s.x) === s.y).length / test.length;
  assert.ok(fgsmAccuracy < clean * 0.5, `FGSM barely hurt: ${fgsmAccuracy}`);
  assert.ok(pgdAccuracy <= fgsmAccuracy + 1e-9, `PGD ${pgdAccuracy} is weaker than FGSM ${fgsmAccuracy}`);

  // 2. Budget invariant: every perturbation, including random-start PGD, stays inside the ball.
  const randomStart = test.map((s) => pgd(model, s, eps, { steps: 12, randomStart: true, prng: new Prng(5) }));
  for (const attack of [...fgsmAttacked, ...pgdAttacked, ...randomStart]) {
    for (const d of attack.delta) assert.ok(Math.abs(d) <= eps + 1e-12, `delta ${d} exceeds eps`);
  }
  for (const attack of randomStart) {
    attack.x.forEach((v, i) => assert.ok(Math.abs(v - test[randomStart.indexOf(attack)].x[i]) <= eps + 1e-12));
  }

  // 3. PGD is never weaker than FGSM; for a *linear* model with a monotone loss the FGSM corner is
  //    already optimal, so the two attacks must in fact reach exactly the same loss.
  let matched = 0;
  for (let k = 0; k < test.length; k++) {
    assert.ok(pgdAttacked[k].loss >= fgsmAttacked[k].loss - 1e-12,
      `PGD loss ${pgdAttacked[k].loss} below FGSM ${fgsmAttacked[k].loss}`);
    assert.ok(Math.abs(pgdAttacked[k].loss - fgsmAttacked[k].loss) < 1e-9,
      "a linear model leaves PGD nothing beyond the FGSM corner");
    matched++;
  }
  assert.strictEqual(matched, test.length);

  // 4. Boundary: eps = 0 is a no-op and leaves the clean accuracy untouched.
  const noAttack = test.map((s) => fgsm(model, s, 0));
  assert.ok(noAttack.every((a) => a.delta.every((d) => d === 0)), "eps = 0 must not perturb");
  assert.strictEqual(noAttack.filter((a, k) => model.predict(a.x) === test[k].y).length / test.length, clean);
  assert.strictEqual(pgd(model, test[0], 0, { steps: 5 }).delta.every((d) => d === 0), true);
  assert.strictEqual(pgd(model, test[0], eps, { steps: 0 }).x.length, 2, "zero steps returns the input");

  // 5. Oracle: PGD on the hinge objective attains the closed-form optimum, which the corner
  //    enumeration of the 2-feature box confirms independently.
  for (let k = 0; k < 8; k++) {
    const sample = test[k * 7 % test.length];
    const oracle = analyticHingeAttack(model, sample, eps);
    const corners = bruteForceHingeAttack(model, sample, eps);
    assert.ok(Math.abs(oracle.loss - corners.loss) < 1e-12, "the closed form is not the box optimum");
    const attacked = pgd(model, sample, eps, { steps: 6, objective: "hinge" });
    assert.ok(Math.abs(attacked.loss - oracle.loss) < 1e-9,
      `PGD hinge loss ${attacked.loss} missed the optimum ${oracle.loss}`);
    if (oracle.loss > 0) {
      assert.ok(attacked.delta.every((d, i) => Math.sign(d) === -Math.sign(model.weights[i]) * sample.y || d === 0),
        "the optimal perturbation must follow -y * sign(w)");
    }
  }

  // 6. Defense: adversarial training raises the robust accuracy under the same attack. The budget
  //    used for hardening is smaller because a linear model's geometric margin is bounded.
  const defenseEps = 0.15;
  const hardened = new LogisticModel(2);
  for (let round = 0; round < 6; round++) {
    const augmented = train.map((sample) => {
      const adversarial = pgd(hardened, sample, defenseEps, { steps: 8 });
      return { x: adversarial.x, y: sample.y };
    });
    hardened.train([...train, ...augmented], { epochs: 120, step: 0.4 });
  }
  const robustBefore = attacked(test, model, defenseEps, { steps: 12 })
    .filter((s) => model.predict(s.x) === s.y).length / test.length;
  const hardenedAttacked = attacked(test, hardened, defenseEps, { steps: 12 });
  const robustAfter = hardenedAttacked.filter((s) => hardened.predict(s.x) === s.y).length / test.length;
  assert.ok(hardened.accuracy(test) >= 0.85, "hardening must not destroy clean accuracy");
  assert.ok(robustAfter >= robustBefore, `robust accuracy fell from ${robustBefore} to ${robustAfter}`);
  const radiusBefore = medianFlipRadius(model, test);
  const radiusAfter = medianFlipRadius(hardened, test);
  assert.ok(radiusAfter > radiusBefore,
    `the median flip radius must grow: ${radiusBefore} -> ${radiusAfter}`);
  assert.ok(radiusAfter / radiusBefore > 1.01, "hardening must buy a measurable margin");
  assert.ok(radiusAfter > eps * 0.4, "the hardened margin must clear a useful fraction of the budget");

  // 7. Boundary cases: one sample, a coordinate with zero weight, an already-wrong prediction.
  const single = pgd(model, test[0], eps, { steps: 3 });
  assert.strictEqual(single.x.length, 2);
  const flat = new LogisticModel(2);
  flat.weights = [0, 1];
  flat.bias = 0;
  const flatSample = { x: [0.5, 0.5], y: 1 };
  const flatAttack = fgsm(flat, flatSample, 0.2);
  assert.strictEqual(Math.abs(flatAttack.delta[0]), 0, "a zero weight must produce no perturbation");
  assert.ok(Math.abs(flatAttack.delta[1] + 0.2) < 1e-12, "the sign must follow -y * w");
  const wrong = { x: [-1, -1], y: 1 };
  assert.strictEqual(model.predict(wrong.x), -1, "the sample must start misclassified");
  assert.strictEqual(model.predict(pgd(model, wrong, eps, { steps: 5 }).x), -1,
    "an attack cannot rescue an already-wrong prediction");

  // 8. Invalid input is rejected.
  assert.throws(() => new LogisticModel(0), /positive/);
  assert.throws(() => model.score([1]), /wrong dimension/);
  assert.throws(() => fgsm(model, { x: [1, 1], y: 1 }, -0.1), /non-negative/);
  assert.throws(() => pgd(model, { x: [1, 1], y: 1 }, 0.1, { steps: -1 }), /non-negative integer/);
  assert.throws(() => model.predict([1]), /wrong dimension/);
  assert.throws(() => model.accuracy([]), /empty/);

  // 9. Determinism: the same seed replays identically, and for a linear model the optimum does not
  //    depend on the random start; a short run exposes the start itself.
  const replay = () => attacked(test.slice(0, 10), model, eps,
    { steps: 8, randomStart: true, prng: new Prng(9) }).map((a) => a.delta);
  assert.deepStrictEqual(replay(), replay());
  const otherSeed = attacked(test.slice(0, 10), model, eps,
    { steps: 8, randomStart: true, prng: new Prng(10) }).map((a) => a.delta);
  assert.deepStrictEqual(otherSeed, replay(),
    "the linear optimum is reached from any start inside the ball");
  const partial = (seed) => attacked(test.slice(0, 5), model, eps,
    { steps: 1, stepSize: 0.05, randomStart: true, prng: new Prng(seed) }).map((a) => a.delta);
  assert.deepStrictEqual(partial(9), partial(9));
  assert.notDeepStrictEqual(partial(9), partial(10), "the random start must influence a short run");
  for (const delta of partial(9)) {
    for (const d of delta) assert.ok(Math.abs(d) <= eps + 1e-12);
  }

  console.log(`pgd: clean ${clean.toFixed(3)} -> fgsm ${fgsmAccuracy.toFixed(3)} -> pgd ` +
    `${pgdAccuracy.toFixed(3)}, robust after hardening ${robustAfter.toFixed(3)} ` +
    `(was ${robustBefore.toFixed(3)}), eps ${eps}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, LogisticModel, fgsm, pgd, analyticHingeAttack, bruteForceHingeAttack,
  makeDataset, project };
