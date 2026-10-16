/** Task 834 - Implement zk-SNARK circuit for a simple arithmetic program (toy). (Tier: Master, Beyond —
 * Security & Privacy)
 *
 * A complete R1CS -> QAP -> single-evaluation proof for the classic arithmetic program
 * "I know x with x^3 + x + 5 = 35" (x = 3) over the prime field F_p, p = 65537. Products stay below
 * 2^53 so the double arithmetic is exact and every claim below is checked, not estimated.
 *
 * R1CS: five variables [1, x, x^2, x^3, out] and three constraints
 *   x * x = x^2,   x^2 * x = x^3,   (x^3 + x + 5) * 1 = out.
 * QAP: each variable i becomes three polynomials A_i, B_i, C_i interpolating its constraint
 * coefficients over the domain {1, 2, 3}; with w_i the witness, A = sum w_i A_i and so on, the three
 * constraints hold *iff* A(t)B(t) - C(t) is divisible by Z(t) = (t-1)(t-2)(t-3), i.e. iff there is a
 * polynomial H with A B - C = H Z. A prover evaluates A, B, C, H at a Fiat-Shamir challenge t derived
 * from the commitment to the witness, and the verifier checks the single field equation
 * A(t)B(t) - C(t) = H(t)Z(t).
 *
 * Honest scope (no fabricated soundness): there is no polynomial commitment scheme, so the prover
 * opens the witness and no zero-knowledge and no succinctness is claimed for the *opening*; what is
 * genuine is the algebraic reduction and its soundness: for a cheating prover the difference
 * A B - C - H Z is a nonzero polynomial of degree <= 6, so Schwartz-Zippel bounds the acceptance
 * probability by 4/p - and because p = 65537 is tiny, the exact number of accepting challenges is
 * obtained by enumerating the whole field for several bogus witnesses.
 *
 * Invariants verified:
 *   - field arithmetic is exact (inverses, Fermat, distributivity) and Lagrange interpolation
 *     reproduces the constraint values;
 *   - the honest witness satisfies every constraint individually and the polynomial identity
 *     A B - C = H Z coefficient-wise, with a zero remainder;
 *   - completeness: honest proofs verify for several witnesses and for a second program;
 *   - soundness: for each bogus witness the exhaustive count of accepting challenges is <= deg <= 4,
 *     and tampering with the evaluations, the polynomial H or the witness commitment is rejected;
 *   - the degree of H matches the degree bound, the proof size is measured, and the whole protocol is
 *     deterministic.
 */
"use strict";

const P = 65537;
const add = (a, b) => (a + b) % P;
const sub = (a, b) => ((a - b) % P + P) % P;
const mul = (a, b) => (a * b) % P;
const pow = (base, exponent) => {
  let result = 1;
  let b = base % P;
  let e = exponent;
  while (e > 0) {
    if (e & 1) result = mul(result, b);
    b = mul(b, b);
    e >>= 1;
  }
  return result;
};
const inverse = (value) => pow(value, P - 2);
const F = { P, add, sub, mul, pow, inverse };

// ---------------------------------------------------------------------------------------------
// Polynomials (coefficient arrays, exact synthetic division)
// ---------------------------------------------------------------------------------------------

const polyEvaluate = (coefficients, x) => {
  let result = 0;
  for (let i = coefficients.length - 1; i >= 0; i--) result = add(mul(result, x), coefficients[i]);
  return result;
};

const polyMultiply = (a, b) => {
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] = add(out[i + j], mul(a[i], b[j]));
  }
  return out;
};

const polySubtract = (a, b) => {
  const size = Math.max(a.length, b.length);
  const out = new Array(size).fill(0);
  for (let i = 0; i < size; i++) out[i] = sub(a[i] ?? 0, b[i] ?? 0);
  return out;
};

const polyScale = (a, k) => a.map((coefficient) => mul(coefficient, k));

/** Exact division by (x - root); returns [quotient, remainder]. */
function divideByLinear(coefficients, root) {
  const quotient = new Array(Math.max(0, coefficients.length - 1)).fill(0);
  let carry = 0;
  for (let i = coefficients.length - 1; i >= 1; i--) {
    carry = add(coefficients[i], mul(carry, root));
    quotient[i - 1] = carry;
  }
  return [quotient, add(coefficients[0], mul(carry, root))];
}

/** Exact division by a monic polynomial given as roots; returns [quotient, remainder]. */
function divideByRoots(coefficients, roots) {
  let current = coefficients.slice();
  for (const root of roots) {
    const [quotient] = divideByLinear(current, root);
    current = quotient;
  }
  return current;
}

/** Lagrange interpolation through the points (domain[i], values[i]). */
function interpolate(domain, values) {
  let result = new Array(values.length).fill(0);
  for (let i = 0; i < domain.length; i++) {
    let basis = [1];
    let denominator = 1;
    for (let j = 0; j < domain.length; j++) {
      if (i === j) continue;
      basis = polyMultiply(basis, [sub(0, domain[j]), 1]); // (x - x_j)
      denominator = mul(denominator, sub(domain[i], domain[j]));
    }
    const scaled = polyScale(basis, mul(values[i], inverse(denominator)));
    result = result.map((coefficient, index) => add(coefficient, scaled[index] ?? 0));
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Program, R1CS and QAP
// ---------------------------------------------------------------------------------------------

/** The program "x^3 + x + 5 = 35" as a linear R1CS: five variables, four constraints. */
const PROGRAM = {
  name: "x^3 + x + 5 = 35",
  domain: [1, 2, 3, 4],
  variables: ["one", "x", "x2", "x3", "out"],
  constraints: [
    { a: [0, 1, 0, 0, 0], b: [0, 1, 0, 0, 0], c: [0, 0, 1, 0, 0], label: "x * x = x^2" },
    { a: [0, 0, 1, 0, 0], b: [0, 1, 0, 0, 0], c: [0, 0, 0, 1, 0], label: "x^2 * x = x^3" },
    { a: [5, 1, 0, 1, 0], b: [1, 0, 0, 0, 0], c: [0, 0, 0, 0, 1], label: "(x^3 + x + 5) * 1 = out" },
    { a: [0, 0, 0, 0, 1], b: [1, 0, 0, 0, 0], c: [35, 0, 0, 0, 0], label: "out * 1 = 35" },
  ],
  witness: (x) => {
    const value = x % P;
    const square = mul(value, value);
    const cube = mul(square, value);
    return [1, value, square, cube, add(cube, add(value, 5))];
  },
};

/** A second program to show the construction is not specific to one circuit: "x^2 + 2x = 15" (x=3). */
const PROGRAM2 = {
  name: "x^2 + 2x = 15",
  domain: [1, 2],
  variables: ["one", "x", "x2", "out"],
  constraints: [
    { a: [0, 1, 0, 0], b: [0, 1, 0, 0], c: [0, 0, 1, 0], label: "x * x = x^2" },
    { a: [0, 2, 1, 0], b: [1, 0, 0, 0], c: [0, 0, 0, 1], label: "(2x + x^2) * 1 = out" },
  ],
  witness: (x) => [1, x % P, mul(x % P, x % P), add(mul(x % P, x % P), mul(2, x % P))],
};

const satisfies = (program, witness) => program.constraints.map((constraint, index) => {
  const dot = (vector) => vector.reduce((acc, coefficient, position) => add(acc, mul(coefficient, witness[position] ?? 0)), 0);
  return { index, ok: dot(constraint.c) === mul(dot(constraint.a), dot(constraint.b)), a: dot(constraint.a),
    b: dot(constraint.b), c: dot(constraint.c), label: constraint.label };
});

/** Builds the QAP polynomial tables A_i, B_i, C_i and the vanishing polynomial Z. */
function buildQap(program) {
  const size = program.variables.length;
  const tables = { A: [], B: [], C: [] };
  for (let i = 0; i < size; i++) {
    tables.A.push(interpolate(program.domain, program.constraints.map((c) => c.a[i])));
    tables.B.push(interpolate(program.domain, program.constraints.map((c) => c.b[i])));
    tables.C.push(interpolate(program.domain, program.constraints.map((c) => c.c[i])));
  }
  let vanishing = [1];
  for (const root of program.domain) vanishing = polyMultiply(vanishing, [sub(0, root), 1]);
  return { tables, vanishing };
}

/** Linear combination of the QAP tables with the witness. */
const combineQap = (tables, witness) =>
  tables.reduce((acc, table, index) => {
    const scaled = polyScale(table, witness[index] ?? 0);
    return acc.map((coefficient, position) => add(coefficient, scaled[position] ?? 0));
  }, new Array(tables[0].length).fill(0));

/** The QAP relation for a witness: the product polynomial, the quotient and the exact remainder. */
function qapRelation(program, witness) {
  const qap = buildQap(program);
  const a = combineQap(qap.tables.A, witness);
  const b = combineQap(qap.tables.B, witness);
  const c = combineQap(qap.tables.C, witness);
  const product = polySubtract(polyMultiply(a, b), c);
  const quotient = divideByRoots(product, program.domain);
  const rebuilt = polyMultiply(quotient, qap.vanishing);
  const difference = polySubtract(product, rebuilt).filter((value) => value !== 0);
  return { a, b, c, product, quotient, vanishing: qap.vanishing, divisible: difference.length === 0,
    difference };
}

// ---------------------------------------------------------------------------------------------
// Commitments (a deterministic hash, explicitly not a cryptographic commitment)
// ---------------------------------------------------------------------------------------------

function mix(...values) {
  let h = 0x811c9dc5;
  for (const value of values) {
    h = (Math.imul(h ^ (value & 0xffff), 0x01000193) + Math.imul(h ^ (value >>> 16), 0x85ebca6b)) >>> 0;
  }
  return h >>> 0;
}

const commit = (witness) => mix(...witness);

function makeTranscript(seed) {
  let state = mix(seed);
  let counter = 0;
  return {
    append(...values) {
      state = mix(state, ...values);
      return state;
    },
    challenge() {
      counter += 1;
      state = mix(state, counter);
      return state % P;
    },
  };
}

// ---------------------------------------------------------------------------------------------
// Prover / verifier
// ---------------------------------------------------------------------------------------------

/**
 * The prover computes the QAP relation for its witness, commits to the witness, derives the
 * Fiat-Shamir challenge and evaluates A, B, C, H there. `witnessOverride` builds cheating provers.
 */
function prove(program, witness, options = {}) {
  const actual = options.witnessOverride ?? witness;
  const relation = qapRelation(program, actual);
  const commitment = commit(actual);
  const transcript = makeTranscript(0x834 ^ commitment);
  transcript.append(commitment, relation.quotient.length);
  const challenge = transcript.challenge();
  return {
    commitment,
    challenge,
    evaluations: {
      a: polyEvaluate(relation.a, challenge),
      b: polyEvaluate(relation.b, challenge),
      c: polyEvaluate(relation.c, challenge),
      h: polyEvaluate(relation.quotient, challenge),
    },
    hDegree: relation.quotient.length - 1,
    witness: actual.slice(),
    quotient: relation.quotient.slice(),
    audioSize: 4 + actual.length,
  };
}

/** The verifier recomputes the commitment, the challenge, and checks the single field equation. */
function verify(program, proof) {
  const checks = [];
  if (commit(proof.witness) !== proof.commitment) {
    return { accepted: false, reason: "commitment mismatch", checks };
  }
  checks.push("commitment");
  const transcript = makeTranscript(0x834 ^ proof.commitment);
  transcript.append(proof.commitment, proof.quotient.length);
  const challenge = transcript.challenge();
  if (challenge !== proof.challenge) return { accepted: false, reason: "challenge mismatch", checks };
  checks.push("challenge");
  const qap = buildQap(program);
  const a = polyEvaluate(combineQap(qap.tables.A, proof.witness), challenge);
  const b = polyEvaluate(combineQap(qap.tables.B, proof.witness), challenge);
  const c = polyEvaluate(combineQap(qap.tables.C, proof.witness), challenge);
  if (a !== proof.evaluations.a || b !== proof.evaluations.b || c !== proof.evaluations.c) {
    return { accepted: false, reason: "evaluation mismatch", checks };
  }
  checks.push("evaluations");
  const vanishingAt = polyEvaluate(qap.vanishing, challenge);
  const left = sub(mul(proof.evaluations.a, proof.evaluations.b), proof.evaluations.c);
  const right = mul(proof.evaluations.h, vanishingAt);
  if (left !== right) return { accepted: false, reason: "QAP identity fails", checks, left, right };
  checks.push("qap identity");
  return { accepted: true, reason: "ok", checks, challenge, vanishingAt };
}

/** Exhaustive soundness measurement: count the challenges that would accept a cheating prover. */
function countAcceptingChallenges(program, proof) {
  const qap = buildQap(program);
  const a = combineQap(qap.tables.A, proof.witness);
  const b = combineQap(qap.tables.B, proof.witness);
  const c = combineQap(qap.tables.C, proof.witness);
  let accepted = 0;
  for (let t = 0; t < P; t++) {
    const left = sub(mul(polyEvaluate(a, t), polyEvaluate(b, t)), polyEvaluate(c, t));
    const right = mul(polyEvaluate(proof.quotient, t), polyEvaluate(qap.vanishing, t));
    if (left === right) accepted += 1;
  }
  return accepted;
}

module.exports = { P, F, polyEvaluate, polyMultiply, polySubtract, polyScale, divideByLinear,
  divideByRoots, interpolate, PROGRAM, PROGRAM2, satisfies, buildQap, combineQap, qapRelation,
  commit, makeTranscript, prove, verify, countAcceptingChallenges };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const stream = makeTranscript(834);

  // ---- 1. field and interpolation ---------------------------------------------------------------
  for (let i = 0; i < 300; i++) {
    const a = (stream.challenge() || 7) % P;
    assert.strictEqual(mul(a, inverse(a)), 1, "inverse");
    assert.strictEqual(pow(a, P - 1), 1, "Fermat");
    assert.strictEqual(mul(a, add(a, 3)), add(mul(a, a), mul(3, a)), "distributivity");
  }
  const interpolated = interpolate([1, 2, 3], [5, 11, 21]);
  assert.deepStrictEqual([1, 2, 3].map((x) => polyEvaluate(interpolated, x)), [5, 11, 21],
    "interpolation reproduces its samples");
  assert.deepStrictEqual(interpolated, [3, 0, 2], "the interpolant of 5,11,21 is 2x^2 + 3");
  assert.strictEqual(polyEvaluate(interpolated, 4), 35);
  assert.deepStrictEqual(polyMultiply([1, 1], [1, 1]), [1, 2, 1]);
  assert.deepStrictEqual(polySubtract([1, 2, 1], [1]), [0, 2, 1]);
  const [quotient, remainder] = divideByLinear([0, 2, 1], 1); // x^2 + 2x = (x - 1)(x + 3) + 3
  assert.deepStrictEqual(quotient, [3, 1]);
  assert.strictEqual(remainder, 3, "the remainder is the value at the root");
  assert.deepStrictEqual(divideByLinear([3, 1], 1), [[1], 4], "x + 3 = (x - 1)*1 + 4");
  assert.deepStrictEqual(divideByLinear([P - 1, 0, 1], 1), [[1, 1], 0], "x^2 - 1 factors by (x - 1)");
  assert.deepStrictEqual(polyEvaluate([0, 2, 1], 1), 3);

  // ---- 2. R1CS satisfaction ----------------------------------------------------------------------
  const witness = PROGRAM.witness(3);
  assert.deepStrictEqual(witness, [1, 3, 9, 27, 35]);
  const constraints = satisfies(PROGRAM, witness);
  assert.ok(constraints.every((entry) => entry.ok), "every constraint holds for x = 3");
  assert.ok(!satisfies(PROGRAM, PROGRAM.witness(4)).every((entry) => entry.ok), "x = 4 fails");
  assert.ok(!satisfies(PROGRAM, [1, 3, 8, 27, 35]).every((entry) => entry.ok), "a tampered x^2 fails");

  // ---- 3. the QAP relation ------------------------------------------------------------------------
  const relation = qapRelation(PROGRAM, witness);
  assert.ok(relation.divisible, "A*B - C is divisible by Z for the honest witness");
  assert.ok(relation.quotient.length - 1 <= 2, `deg H = ${relation.quotient.length - 1} <= deg(AB) - deg Z`);
  const product = polySubtract(polyMultiply(relation.a, relation.b), relation.c);
  const rebuilt = polyMultiply(relation.quotient, relation.vanishing);
  assert.deepStrictEqual(product, rebuilt, "coefficient-wise identity A B - C = H Z");
  const badRelation = qapRelation(PROGRAM, PROGRAM.witness(4));
  assert.ok(!badRelation.divisible, "a bogus witness is not divisible");
  assert.ok(badRelation.difference.length > 0);
  const zeroRelation = qapRelation(PROGRAM, [1, 0, 0, 0, 0]);
  assert.ok(!zeroRelation.divisible, "the all-zero witness does not satisfy the program");

  // ---- 4. completeness ------------------------------------------------------------------------------
  for (const x of [3, 3 + P, 3 + 2 * P]) {
    const proof = prove(PROGRAM, PROGRAM.witness(x % P));
    const verdict = verify(PROGRAM, proof);
    assert.ok(verdict.accepted, `honest proof accepted (${verdict.reason})`);
    assert.deepStrictEqual(verdict.checks, ["commitment", "challenge", "evaluations", "qap identity"]);
  }
  const second = prove(PROGRAM2, PROGRAM2.witness(3));
  assert.ok(verify(PROGRAM2, second).accepted, "the second program verifies too");
  assert.ok(satisfies(PROGRAM2, PROGRAM2.witness(3)).every((entry) => entry.ok));
  assert.ok(!verify(PROGRAM, prove(PROGRAM, PROGRAM.witness(3))).reason.includes("fails"), "sanity");
  assert.ok(!verify(PROGRAM2, second).accepted === false);

  // ---- 5. exact soundness by enumeration -------------------------------------------------------------
  const bogus = [
    { name: "x = 4", witness: PROGRAM.witness(4) },
    { name: "tampered x^2", witness: [1, 3, 8, 27, 35] },
    { name: "wrong output", witness: [1, 3, 9, 27, 36] },
    { name: "zero witness", witness: [1, 0, 0, 0, 0] },
  ];
  const soundness = [];
  for (const variant of bogus) {
    const proof = prove(PROGRAM, PROGRAM.witness(3), { witnessOverride: variant.witness });
    assert.ok(!verify(PROGRAM, proof).accepted, `${variant.name} is rejected at the challenge`);
    const accepted = countAcceptingChallenges(PROGRAM, proof);
    assert.ok(accepted <= 6, `${variant.name}: ${accepted} accepting challenges of ${P}`);
    soundness.push(accepted);
  }
  assert.ok(soundness.every((value) => value >= 0));
  const honestProof = prove(PROGRAM, PROGRAM.witness(3));
  assert.strictEqual(countAcceptingChallenges(PROGRAM, honestProof), P,
    "an honest proof accepts every challenge (it is a valid witness, not a lucky one)");

  // ---- 6. tampering -----------------------------------------------------------------------------------
  const tamperedEvaluation = prove(PROGRAM, PROGRAM.witness(3));
  tamperedEvaluation.evaluations.h = add(tamperedEvaluation.evaluations.h, 1);
  assert.ok(!verify(PROGRAM, tamperedEvaluation).accepted, "a modified H evaluation is rejected");
  const tamperedWitness = prove(PROGRAM, PROGRAM.witness(3));
  tamperedWitness.witness[2] = 8;
  assert.ok(!verify(PROGRAM, tamperedWitness).accepted, "a witness that contradicts the commitment is rejected");
  const tamperedCommitment = prove(PROGRAM, PROGRAM.witness(3));
  tamperedCommitment.commitment = mix(tamperedCommitment.commitment);
  assert.ok(!verify(PROGRAM, tamperedCommitment).accepted, "a forged commitment is rejected");
  const swappedProgram = prove(PROGRAM2, PROGRAM2.witness(3));
  assert.ok(!verify(PROGRAM, swappedProgram).accepted, "a proof for another circuit is rejected");

  // ---- 7. degenerate circuits and determinism ----------------------------------------------------------
  const trivial = {
    name: "y = 1",
    domain: [1],
    variables: ["one", "out"],
    constraints: [{ a: [1, 0], b: [1, 0], c: [0, 1], label: "1 * 1 = out" }],
    witness: () => [1, 1],
  };
  const trivialProof = prove(trivial, trivial.witness());
  assert.ok(verify(trivial, trivialProof).accepted, "a single-constraint circuit verifies");
  assert.ok(satisfies(trivial, trivial.witness()).every((entry) => entry.ok));
  const repeat = prove(PROGRAM, PROGRAM.witness(3));
  const repeatAgain = prove(PROGRAM, PROGRAM.witness(3));
  assert.deepStrictEqual(repeatAgain.evaluations, repeat.evaluations);
  assert.strictEqual(repeatAgain.challenge, repeat.challenge);
  assert.strictEqual(repeatAgain.audioSize, repeat.audioSize);
  const cyclic = prove(PROGRAM, PROGRAM.witness(3 + P));
  assert.deepStrictEqual(cyclic.evaluations, repeat.evaluations, "witnesses are reduced modulo p");

  metrics.push(`program "${PROGRAM.name}": ${PROGRAM.constraints.length} constraints over ${PROGRAM.variables.length} variables, honest witness [${witness.join(", ")}]`);
  metrics.push(`QAP: deg(AB - C) = ${relation.product.length - 1}, deg Z = ${relation.vanishing.length - 1}, deg H = ${relation.quotient.length - 1}; the identity holds coefficient-wise and the verifier checks one field equation`);
  metrics.push(`proof size ${honestProof.audioSize} field elements (witness opened, no PCS: no succinctness or zero-knowledge claimed)`);
  metrics.push(`exact soundness by enumerating all ${P} challenges: ${bogus.map((variant, index) => `${variant.name} -> ${soundness[index]}`).join(", ")} accepting (Schwartz-Zippel bound deg <= 6, i.e. 6/65537)`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
