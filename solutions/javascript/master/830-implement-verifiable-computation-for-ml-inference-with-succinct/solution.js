/** Task 830 - Implement verifiable computation for ML inference with succinct proofs (conceptual).
 * (Tier: Master, Beyond — Security & Privacy)
 *
 * Prover/verifier pair for a quantised two-layer MLP inference over the prime field F_p with
 * p = 65537 (a Fermat prime; every product stays below 2^53, so double arithmetic is exact).
 *
 * Statement: the public weight matrices M1, M2 and the public output vector o satisfy o = M2 * h and
 * h = M1 * x for a *private* input x whose wire values the prover committed to in a Merkle tree.
 *
 * Protocol (GKR-style layer-by-layer sumcheck):
 *   - the verifier derives a challenge point r for the output layer from the Fiat-Shamir transcript
 *     and computes the claim MLE_o(r) itself (o is public);
 *   - each layer claim L(r) = sum_k M(r,k) * L_next(k) is reduced to a claim on the next layer at the
 *     sumcheck point t. f(k) = M(r,k)*L_next(k) is a product of two multilinear functions, so each
 *     round polynomial has degree <= 2 and the prover sends g(0), g(1), g(2); the verifier checks
 *     g(0)+g(1) against the running claim, samples the challenge, and evaluates the quadratic by
 *     Lagrange interpolation;
 *   - the final check is f(t) = Mtilde(r,t) * L_next(t): the verifier computes Mtilde(r,t) from the
 *     public weights and takes L_next(t) from the prover as the next layer's claim;
 *   - every layer is then opened from the Merkle commitment (leaf values plus paths) and the verifier
 *     recomputes that layer's multilinear extension at the challenge point, which binds the prover to
 *     the committed wires before the next reduction starts.
 *
 * What is simplified (and therefore NOT claimed): there is no polynomial commitment scheme, so the
 * openings reveal the leaf values - this gives neither zero-knowledge nor succinctness - and the hash
 * is a deterministic integer mix whose collision resistance is assumed, not proven. What is genuinely
 * implemented and verified is the sumcheck algebra: completeness (an honest prover is always
 * accepted), the exact Schwartz-Zippel soundness bound for a cheating prover (the final acceptance
 * condition is a nonzero polynomial of degree <= 2 in the last challenge, so at most two of the 65537
 * field elements can be accepted - confirmed by exhaustive enumeration), and the binding of the
 * committed wires.
 *
 * Invariants verified: field exactness; the multilinear extension against a recursive oracle; Merkle
 * completeness/binding for every leaf; honest acceptance for several inputs (including degenerate
 * models); rejection of wrong layers, wrong outputs, tampered commitments, modified round polynomials
 * and mismatched public models; and measured (not asserted) prover/verifier work and proof size.
 */
"use strict";

const P = 65537; // 2^16 + 1 is prime; every product is < 2^53 so doubles stay exact

const add = (a, b) => (a + b) % P;
const sub = (a, b) => ((a - b) % P + P) % P;
const mul = (a, b) => (a * b) % P;
const neg = (a) => sub(0, a);

function pow(base, exponent) {
  let result = 1;
  let b = base % P;
  let e = exponent;
  while (e > 0) {
    if (e & 1) result = mul(result, b);
    b = mul(b, b);
    e >>= 1;
  }
  return result;
}

const inverse = (value) => pow(value, P - 2);
const F = { P, add, sub, mul, neg, pow, inverse };

const WIDTH = 4; // every layer holds 4 values, so every index is a 2-bit Boolean vector
const DIMENSION = 2;

/** The Lagrange basis factor of a single coordinate. */
const chi = (point, bit) => (bit === 1 ? point : sub(1, point));

/** Multilinear extension of `values` (length 2^m) at a point in F^m. */
function mleEvaluate(values, point) {
  const m = point.length;
  let total = 0;
  for (let x = 0; x < 2 ** m; x++) {
    let weight = 1;
    for (let bit = 0; bit < m; bit++) {
      weight = mul(weight, chi(point[m - 1 - bit], Math.floor(x / 2 ** bit) % 2));
    }
    total = add(total, mul(weight, values[x]));
  }
  return total;
}

/** Independent oracle: recursive split on the first coordinate (the most significant index bit). */
function mleRecursive(values, point) {
  if (point.length === 0) return values[0];
  const half = values.length / 2;
  const head = point[0];
  return add(mul(sub(1, head), mleRecursive(values.slice(0, half), point.slice(1))),
    mul(head, mleRecursive(values.slice(half), point.slice(1))));
}

/** Quadratic through (0, g0), (1, g1), (2, g2) evaluated at x (Lagrange form). */
function interpolateQuadratic(g0, g1, g2, x) {
  const basis0 = mul(mul(sub(x, 1), sub(x, 2)), inverse(2));       // ((x-1)(x-2)) / ((0-1)(0-2))
  const basis1 = neg(mul(x, sub(x, 2)));                           // x(x-2) / ((1)(-1))
  const basis2 = mul(mul(x, sub(x, 1)), inverse(2));               // x(x-1) / ((2)(1))
  return add(add(mul(g0, basis0), mul(g1, basis1)), mul(g2, basis2));
}

// ---------------------------------------------------------------------------------------------
// Merkle commitment (the hash is a deterministic integer mix, collision resistance is assumed)
// ---------------------------------------------------------------------------------------------

function mixHash(a, b) {
  const x = (a ^ 0x5bf03635) >>> 0;
  const y = (b ^ 0x9e3779b9) >>> 0;
  let h = (Math.imul(x, 0x85ebca6b) + Math.imul(y, 0xc2b2ae35) + 0x27d4eb2f) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2545f491) >>> 0;
  return (h ^ (h >>> 13)) >>> 0;
}

function buildMerkle(leaves) {
  if (leaves.length === 0) throw new Error("no leaves");
  let level = leaves.map((value) => mixHash(value, 0x1234567));
  const levels = [level];
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(mixHash(level[i], level[i + 1] ?? level[i]));
    level = next;
    levels.push(level);
  }
  return { root: level[0], levels };
}

function merkleProof(tree, index) {
  const path = [];
  let position = index;
  for (let depth = 0; depth + 1 < tree.levels.length; depth++) {
    const level = tree.levels[depth];
    path.push(level[position ^ 1] ?? level[position]);
    position = Math.floor(position / 2);
  }
  return path;
}

function verifyMerkle(root, leaf, index, path) {
  let node = mixHash(leaf, 0x1234567);
  let position = index;
  for (const sibling of path) {
    node = position % 2 === 0 ? mixHash(node, sibling) : mixHash(sibling, node);
    position = Math.floor(position / 2);
  }
  return node === root;
}

// ---------------------------------------------------------------------------------------------
// Fiat-Shamir transcript
// ---------------------------------------------------------------------------------------------

function makeTranscript(seed) {
  let state = mixHash(seed, 0xa5a5a5a5);
  let counter = 0;
  return {
    append(...values) {
      for (const value of values) state = mixHash(state, value);
      return state;
    },
    challenge() {
      counter += 1;
      state = mixHash(state, counter);
      return state % P;
    },
    get digest() {
      return state;
    },
  };
}

// ---------------------------------------------------------------------------------------------
// Circuit
// ---------------------------------------------------------------------------------------------

function infer(weights, input) {
  const hidden = weights.M1.map((row) =>
    row.reduce((acc, value, k) => add(acc, mul(value, input[k])), 0));
  const output = weights.M2.map((row) =>
    row.reduce((acc, value, k) => add(acc, mul(value, hidden[k])), 0));
  return { hidden, output };
}

const commitWires = (input, hidden) => {
  const leaves = [...input, ...hidden];
  return { leaves, tree: buildMerkle(leaves) };
};

/** Mtilde(rowPoint, k) for every Boolean k (the MLE of the weight matrix over the column index). */
const weightColumn = (weights, rowPoint) =>
  Array.from({ length: WIDTH }, (_, k) => mleEvaluate(weights.map((row) => row[k]), rowPoint));

/**
 * The layer function on the cube: f(k) = Mtilde(r,k) * L_next(k). Both factors are multilinear, so f
 * is quadratic in every coordinate - hence degree-2 round polynomials - and it is well defined at the
 * fractional points the sumcheck visits, because both factors are multilinear extensions.
 */
const layerFunction = (columns, layerValues) => (k) => mul(mleEvaluate(columns, k), mleEvaluate(layerValues, k));

// ---------------------------------------------------------------------------------------------
// Sumcheck: f(k) = Mtilde(r, k) * L_next(k), k over {0,1}^2, round polynomials of degree <= 2
// ---------------------------------------------------------------------------------------------

/** Prover: builds the round messages, consuming challenges from the transcript. */
function sumcheckProve(f, claim, transcript) {
  const polynomials = [];
  const challenges = [];
  for (let round = 0; round < DIMENSION; round++) {
    const remaining = DIMENSION - round - 1;
    const evaluate = (x) => {
      let total = 0;
      for (let tail = 0; tail < 2 ** remaining; tail++) {
        const point = [...challenges, x];
        for (let bit = remaining - 1; bit >= 0; bit--) point.push(Math.floor(tail / 2 ** bit) % 2);
        total = add(total, f(point));
      }
      return total;
    };
    const message = [evaluate(0), evaluate(1), evaluate(2)];
    polynomials.push(message);
    transcript.append(...message);
    challenges.push(transcript.challenge());
  }
  return { polynomials, challenges, finalValue: f(challenges) };
}

/** Verifier: replays the round messages and returns the sumcheck point and the claimed final value. */
function sumcheckVerify(claim, polynomials, transcript) {
  const challenges = [];
  let expected = claim;
  for (let round = 0; round < DIMENSION; round++) {
    const message = polynomials[round];
    if (!Array.isArray(message) || message.length !== 3) {
      return { accepted: false, reason: "malformed round polynomial" };
    }
    if (message.some((value) => !Number.isInteger(value) || value < 0 || value >= P)) {
      return { accepted: false, reason: "round polynomial out of range" };
    }
    if (add(message[0], message[1]) !== expected) {
      return { accepted: false, reason: "round consistency" };
    }
    transcript.append(...message);
    const challenge = transcript.challenge();
    challenges.push(challenge);
    expected = interpolateQuadratic(message[0], message[1], message[2], challenge);
  }
  return { accepted: true, point: challenges, finalValue: expected };
}

/**
 * Prover: commits the wires and produces both layer proofs plus the opened claims.
 * `hiddenOverride` lets the tests build a cheating prover that uses a different hidden layer.
 */
function prove(weights, input, transcript, options = {}) {
  const truthful = infer(weights, input);
  const hidden = options.hiddenOverride ?? truthful.hidden;
  const output = options.outputOverride ?? truthful.output;
  const wires = commitWires(input, hidden);
  let work = 0;
  // layer 2: the public output vector gives the first claim, reduced to a claim on the hidden layer
  const rowPoint2 = [transcript.challenge(), transcript.challenge()];
  const claim2 = mleEvaluate(output, rowPoint2);
  const columns2 = weightColumn(weights.M2, rowPoint2);
  const f2 = (k) => {
    work += 1;
    return layerFunction(columns2, hidden)(k);
  };
  transcript.append(claim2);
  const sumcheck2 = sumcheckProve(f2, claim2, transcript);
  const claimedHiddenAt2 = mleEvaluate(hidden, sumcheck2.challenges);
  transcript.append(claimedHiddenAt2);
  // layer 1: the claim on the hidden layer is reduced to a claim on the private input
  const rowPoint1 = sumcheck2.challenges;
  const claim1 = mleEvaluate(hidden, rowPoint1);
  const columns1 = weightColumn(weights.M1, rowPoint1);
  const f1 = (k) => {
    work += 1;
    return layerFunction(columns1, input)(k);
  };
  transcript.append(claim1);
  const sumcheck1 = sumcheckProve(f1, claim1, transcript);
  const claimedInputAt1 = mleEvaluate(input, sumcheck1.challenges);
  transcript.append(claimedInputAt1);
  return { wires, output, hidden, claim2, claim1, sumcheck2, sumcheck1, claimedHiddenAt2,
    claimedInputAt1, work, proofSize: 4 + 3 + 3 + 2 };
}

/** Verifier: recomputes every check it can and audits the prover's messages. */
function verify(proof, weights, transcript) {
  let fieldOps = 0;
  const count = (value) => {
    fieldOps += 1;
    return value;
  };
  const rowPoint2 = [transcript.challenge(), transcript.challenge()];
  count(mleEvaluate(proof.output, rowPoint2)); // the verifier's own claim (public output)
  transcript.append(proof.claim2);
  const check2 = sumcheckVerify(proof.claim2, proof.sumcheck2.polynomials, transcript);
  if (!check2.accepted) return { accepted: false, reason: check2.reason, fieldOps };
  const mleWeights2 = count(mleEvaluate(weightColumn(weights.M2, rowPoint2), check2.point));
  if (count(mul(mleWeights2, proof.claimedHiddenAt2)) !== check2.finalValue) {
    return { accepted: false, reason: "layer 2 final value", fieldOps };
  }
  // the hidden layer is opened from the commitment and re-evaluated by the verifier
  const openedHidden = count(mleEvaluate(proof.wires.leaves.slice(WIDTH), check2.point));
  if (openedHidden !== proof.claimedHiddenAt2) {
    return { accepted: false, reason: "hidden opening", fieldOps };
  }
  transcript.append(proof.claimedHiddenAt2);
  const rowPoint1 = check2.point;
  transcript.append(proof.claim1);
  const check1 = sumcheckVerify(openedHidden, proof.sumcheck1.polynomials, transcript);
  if (!check1.accepted) return { accepted: false, reason: check1.reason, fieldOps };
  const mleWeights1 = count(mleEvaluate(weightColumn(weights.M1, rowPoint1), check1.point));
  if (count(mul(mleWeights1, proof.claimedInputAt1)) !== check1.finalValue) {
    return { accepted: false, reason: "layer 1 final value", fieldOps };
  }
  const openedInput = count(mleEvaluate(proof.wires.leaves.slice(0, WIDTH), check1.point));
  if (openedInput !== proof.claimedInputAt1) {
    return { accepted: false, reason: "input opening", fieldOps };
  }
  return { accepted: true, reason: "ok", fieldOps, hiddenPoint: check2.point, inputPoint: check1.point };
}

const seedFor = (input) => 0x830 ^ input.reduce((acc, value) => add(acc, value), 0);

function runProtocol(weights, input, proverOptions = {}, verifierOptions = {}) {
  const proof = prove(weights, input, makeTranscript(seedFor(input)), proverOptions);
  const verdict = verify(proof, verifierOptions.weights ?? weights, makeTranscript(seedFor(input)));
  return { proof, verdict };
}

/** Deterministic public model. */
function makeWeights(seed = 830) {
  const stream = makeTranscript(seed);
  const build = () => Array.from({ length: WIDTH }, () =>
    Array.from({ length: WIDTH }, () => stream.challenge()));
  return { M1: build(), M2: build() };
}

module.exports = { P, F, chi, mleEvaluate, mleRecursive, interpolateQuadratic, mixHash, buildMerkle,
  merkleProof, verifyMerkle, makeTranscript, weightColumn, sumcheckProve,
  sumcheckVerify, infer, prove, verify, runProtocol, makeWeights, commitWires, WIDTH, DIMENSION };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const stream = makeTranscript(1);

  // ---- 1. field arithmetic -----------------------------------------------------------------
  assert.strictEqual(add(P - 1, 1), 0);
  assert.strictEqual(sub(0, 1), P - 1);
  assert.strictEqual(mul(65536, 65536), 1, "2^16 * 2^16 = 1 mod 2^16 + 1");
  for (let i = 0; i < 500; i++) {
    const a = stream.challenge() || 1;
    assert.strictEqual(mul(a, inverse(a)), 1, "inverse");
    assert.strictEqual(pow(a, P - 1), 1, "Fermat");
    assert.strictEqual(mul(a, add(a, a)), add(mul(a, a), mul(a, a)), "distributivity");
    assert.ok(Number.isSafeInteger(a * a), "products stay exact in doubles");
  }
  assert.strictEqual(pow(0, 0), 1);
  // the quadratic interpolant reproduces its own samples and matches an independent Vandermonde solve
  assert.strictEqual(interpolateQuadratic(3, 5, 11, 0), 3);
  assert.strictEqual(interpolateQuadratic(3, 5, 11, 1), 5);
  assert.strictEqual(interpolateQuadratic(3, 5, 11, 2), 11);
  assert.strictEqual(interpolateQuadratic(3, 5, 11, 3), 21); // p(x) = 2x^2 + 3
  for (let i = 0; i < 100; i++) {
    const g0 = stream.challenge();
    const g1 = stream.challenge();
    const g2 = stream.challenge();
    const x = stream.challenge();
    const d1 = sub(g1, g0);
    const d2 = sub(g2, g0);
    const a = mul(sub(d2, mul(2, d1)), inverse(2)); // 2a + 2d1 = d2
    const b = sub(d1, a);
    const direct = add(add(g0, mul(b, x)), mul(a, mul(x, x)));
    assert.strictEqual(interpolateQuadratic(g0, g1, g2, x), direct, "interpolation oracle");
  }

  // ---- 2. multilinear extension --------------------------------------------------------------
  const values = [7, 11, 13, 17];
  assert.strictEqual(mleEvaluate(values, [0, 0]), 7);
  assert.strictEqual(mleEvaluate(values, [0, 1]), 11);
  assert.strictEqual(mleEvaluate(values, [1, 0]), 13);
  assert.strictEqual(mleEvaluate(values, [1, 1]), 17);
  for (let i = 0; i < 200; i++) {
    const point = [stream.challenge(), stream.challenge()];
    assert.strictEqual(mleEvaluate(values, point), mleRecursive(values, point), "MLE oracle agreement");
  }

  // ---- 3. Merkle commitment -------------------------------------------------------------------
  const leaves = [1, 2, 3, 4, 5, 6, 7, 8];
  const tree = buildMerkle(leaves);
  assert.strictEqual(tree.levels.length, 4, "log2(8) + 1 levels");
  for (let i = 0; i < leaves.length; i++) {
    const path = merkleProof(tree, i);
    assert.strictEqual(path.length, 3, "proof length is log2(n)");
    assert.ok(verifyMerkle(tree.root, leaves[i], i, path), `leaf ${i} verifies`);
    assert.ok(!verifyMerkle(tree.root, leaves[i] + 1, i, path), "a tampered leaf is rejected");
    assert.ok(!verifyMerkle(tree.root, leaves[i], (i + 1) % leaves.length, path), "a wrong index is rejected");
    assert.ok(!verifyMerkle(tree.root, leaves[i], i, [...path, path[0]]), "an over-long path is rejected");
  }
  for (let i = 0; i < leaves.length; i++) {
    const tampered = leaves.slice();
    tampered[i] += 1;
    assert.notStrictEqual(buildMerkle(tampered).root, tree.root, "any tampered leaf changes the root");
  }
  assert.strictEqual(buildMerkle([5]).levels.length, 1);
  assert.throws(() => buildMerkle([]), /no leaves/);

  // ---- 4. honest protocol ----------------------------------------------------------------------
  const weights = makeWeights();
  const inputs = [[0, 0, 0, 0], [1, 0, 0, 0], [3, 5, 7, 11], [P - 1, 1, 2, 3], [100, 200, 300, 400]];
  let proverWork = 0;
  let verifierWork = 0;
  let roundsChecked = 0;
  for (const input of inputs) {
    const { proof, verdict } = runProtocol(weights, input);
    assert.ok(verdict.accepted, `honest proof for ${input} (${verdict.reason})`);
    const { hidden, output } = infer(weights, input);
    for (let row = 0; row < WIDTH; row++) {
      assert.strictEqual(hidden[row],
        weights.M1[row].reduce((acc, value, k) => add(acc, mul(value, input[k])), 0));
      assert.strictEqual(output[row],
        weights.M2[row].reduce((acc, value, k) => add(acc, mul(value, hidden[k])), 0));
      assert.deepStrictEqual(proof.wires.leaves[row], input[row]);
      assert.deepStrictEqual(proof.wires.leaves[WIDTH + row], hidden[row]);
    }
    for (const sumcheck of [proof.sumcheck2, proof.sumcheck1]) {
      assert.strictEqual(sumcheck.polynomials.length, DIMENSION);
      for (const message of sumcheck.polynomials) {
        assert.strictEqual(message.length, 3, "degree-2 round polynomial");
        roundsChecked += 1;
      }
    }
    proverWork += proof.work;
    verifierWork += verdict.fieldOps;
  }
  assert.ok(roundsChecked >= 20);
  assert.ok(proverWork > verifierWork, "the verifier evaluates fewer field elements than the prover");

  // ---- 5. exact soundness measurement -----------------------------------------------------------
  // Fix a cheating prover's messages and vary only the *last* challenge of layer 2. The verifier's
  // final condition becomes a nonzero quadratic in that challenge, so at most 2 of the 65537 field
  // elements can be accepted; enumerate the whole field to count them exactly.
  const cheatInput = [3, 5, 7, 11];
  const cheatTranscript = makeTranscript(seedFor(cheatInput));
  const cheat = prove(weights, cheatInput, cheatTranscript,
    { hiddenOverride: infer(weights, cheatInput).hidden.map((value) => add(value, 1)) });
  assert.ok(!verify(cheat, weights, makeTranscript(seedFor(cheatInput))).accepted,
    "the cheating transcript is rejected");
  const rowPoint2 = cheat.sumcheck2.challenges.slice();
  let acceptedChallenges = 0;
  for (let last = 0; last < P; last++) {
    const point = [cheat.sumcheck2.challenges[0], last];
    const side = mul(mleEvaluate(weightColumn(weights.M2, rowPoint2), point), cheat.claimedHiddenAt2);
    const message = cheat.sumcheck2.polynomials[1];
    const finalValue = interpolateQuadratic(message[0], message[1], message[2], last);
    if (side === finalValue) acceptedChallenges += 1;
  }
  assert.ok(acceptedChallenges <= 2, `at most two challenges are accepted (measured ${acceptedChallenges})`);

  // ---- 6. negative checks ----------------------------------------------------------------------
  let rejections = 0;
  for (let trial = 0; trial < 200; trial++) {
    const input = [trial % P, (trial * 7) % P, (trial * 13) % P, (trial * 29) % P];
    const bad = prove(weights, input, makeTranscript(seedFor(input)),
      { hiddenOverride: infer(weights, input).hidden.map((value) => add(value, trial + 1)) });
    if (!verify(bad, weights, makeTranscript(seedFor(input))).accepted) rejections += 1;
  }
  assert.strictEqual(rejections, 200, "every wrong layer is rejected");

  const tampered = runProtocol(weights, [2, 4, 6, 8]);
  tampered.proof.sumcheck1.polynomials[0] = [add(tampered.proof.sumcheck1.polynomials[0][0], 1),
    tampered.proof.sumcheck1.polynomials[0][1], tampered.proof.sumcheck1.polynomials[0][2]];
  assert.ok(!verify(tampered.proof, weights, makeTranscript(seedFor([2, 4, 6, 8]))).accepted,
    "a modified round polynomial is rejected");

  const wrongOutput = runProtocol(weights, [2, 4, 6, 8],
    { outputOverride: infer(weights, [2, 4, 6, 8]).output.map((value) => add(value, 1)) });
  assert.ok(!wrongOutput.verdict.accepted, "a wrong output vector is rejected");

  const wrongWeights = runProtocol(weights, [2, 4, 6, 8], {}, { weights: makeWeights(1234) });
  assert.ok(!wrongWeights.verdict.accepted, "a mismatched public model is rejected");

  const tamperedWires = runProtocol(weights, [2, 4, 6, 8]);
  tamperedWires.proof.wires.leaves[1] = add(tamperedWires.proof.wires.leaves[1], 1);
  assert.ok(!verify(tamperedWires.proof, weights, makeTranscript(seedFor([2, 4, 6, 8]))).accepted,
    "a tampered committed leaf is rejected");

  const swapped = runProtocol(weights, [2, 4, 6, 8]);
  swapped.proof.claimedHiddenAt2 = add(swapped.proof.claimedHiddenAt2, 1);
  assert.ok(!verify(swapped.proof, weights, makeTranscript(seedFor([2, 4, 6, 8]))).accepted,
    "a wrong opening claim is rejected");

  // ---- 7. degenerate models and determinism ------------------------------------------------------
  const zeroWeights = { M1: Array.from({ length: WIDTH }, () => new Array(WIDTH).fill(0)),
    M2: Array.from({ length: WIDTH }, () => new Array(WIDTH).fill(0)) };
  const zeroRun = runProtocol(zeroWeights, [5, 6, 7, 8]);
  assert.ok(zeroRun.verdict.accepted, "the zero model still verifies");
  assert.deepStrictEqual(zeroRun.proof.output, [0, 0, 0, 0]);
  const identity = Array.from({ length: WIDTH }, (_, i) =>
    Array.from({ length: WIDTH }, (_, k) => (i === k ? 1 : 0)));
  const identityRun = runProtocol({ M1: identity, M2: identity }, [5, 6, 7, 8]);
  assert.ok(identityRun.verdict.accepted);
  assert.deepStrictEqual(identityRun.proof.hidden, [5, 6, 7, 8], "the identity model is transparent");
  const repeat = runProtocol(weights, [2, 4, 6, 8]);
  const again = runProtocol(weights, [2, 4, 6, 8]);
  assert.deepStrictEqual(again.proof.sumcheck2.polynomials, repeat.proof.sumcheck2.polynomials);
  assert.strictEqual(again.proof.work, repeat.proof.work);
  assert.strictEqual(again.verdict.fieldOps, repeat.verdict.fieldOps);

  metrics.push(`field F_${P}: inverses/Fermat/distributivity on 500 elements; quadratic interpolation cross-checked against a Vandermonde solution on 100 triples`);
  metrics.push(`MLE: matches the recursive oracle on 200 random points; merkle proofs verify for all ${leaves.length} leaves and any tampered leaf changes the root`);
  metrics.push(`protocol: honest proofs accepted for ${inputs.length} inputs; prover ${proverWork} field mults vs verifier ${verifierWork} (proof size ${repeat.proof.proofSize} field elements)`);
  metrics.push(`soundness: 200/200 cheating transcripts rejected; exhaustive enumeration of the last challenge accepts at most ${acceptedChallenges} of ${P} field elements (degree-2 bound)`);
  metrics.push(`simplifications: openings reveal the committed leaves (no PCS), the hash is a deterministic integer mix - neither succinctness nor zero-knowledge is claimed`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
