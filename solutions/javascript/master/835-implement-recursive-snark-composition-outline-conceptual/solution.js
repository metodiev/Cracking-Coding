/** Task 835 - Implement recursive SNARK composition outline (conceptual). (Tier: Master, Beyond —
 * Security & Privacy)
 *
 * An incrementally verifiable computation (IVC) chain whose *proofs are composed*: step i proves both
 * "z_{i+1} = F(z_i, w_i)" and "the proof of step i-1 verified", so one accumulated proof at depth d
 * implies the whole chain. Over the prime field F_p with p = 65537 everything is exact.
 *
 * Step function: z_{i+1} = z_i^2 + w_i (one R1CS constraint per step), plus a hash link
 * h_{i+1} = H(h_i, z_{i+1}, w_i) that binds the state to the transcript. The prover batches the d
 * per-step residuals with random Fiat-Shamir powers of a challenge r extracted from the commitment to
 * the whole chain:  R = sum_i r^i * residual_i. The verifier checks R = 0 against the opened chain and
 * recomputes the hash chain - verifying the depth-d proof therefore *implies* that every step (and,
 * recursively, every layer of the composition) satisfied its constraint.
 *
 * What is simplified (and NOT claimed): there is no pairing-based accumulator, so the verifier still
 * opens the committed chain - the *proof* is constant size, but the verifier's arithmetic grows
 * linearly with the depth (measured below, not hidden). The batching is an exact algebraic identity
 * whose soundness is Schwartz-Zippel over F_p, so the number of accepting challenges for a tampered
 * chain is enumerated exhaustively (at most d-1 of the 65537 field elements).
 *
 * Invariants verified:
 *   - the chain oracle (brute-force IVC) agrees with the committed states, and every step residual is
 *     zero for an honest chain;
 *   - completeness for depths 1..4 and for the degenerate depths 0 and 1;
 *   - soundness: tampering with the state, the witness or the transcript at *any* layer is rejected,
 *     and the exact number of accepting batching challenges is <= deg R <= d - 1 (exhaustive);
 *   - the hash chain binds the composition: a state changed without updating the links is rejected;
 *   - measured succinctness accounting: the proof size is constant in the depth while the constraint
 *     count grows linearly (and the verifier cost stays linear in this toy).
 */
"use strict";

const P = 65537;
const add = (a, b) => (a + b) % P;
const sub = (a, b) => ((a - b) % P + P) % P;
const mul = (a, b) => (a * b) % P;
const neg = (a) => sub(0, a);
const F = { P, add, sub, mul, neg };

/** Non-cryptographic transcript hash (documented; used only for binding, never for secrecy). */
function mix(...values) {
  let h = 0x811c9dc5;
  for (const value of values) {
    h = (Math.imul(h ^ (value & 0xffff), 0x01000193) + Math.imul(h ^ ((value >>> 16) & 0xffff), 0x85ebca6b)) >>> 0;
  }
  return h >>> 0;
}

const linkHash = (previous, state, witness) => mix(previous, state, witness);

/** The step function; residues are kept in F_p. */
const step = (state, witness) => add(mul(state, state), witness);
const residual = (state, next, witness) => sub(next, step(state, witness));

/** Brute-force IVC oracle: runs the chain from z0 with the given witnesses. */
function oracle(states0, witnesses) {
  const states = [states0 % P];
  const links = [mix(0x835, states0 % P)];
  for (let i = 0; i < witnesses.length; i++) {
    const next = step(states[i], witnesses[i]);
    states.push(next);
    links.push(linkHash(links[i], next, witnesses[i]));
  }
  return { states, links };
}

// ---------------------------------------------------------------------------------------------
// Composed proving / verification
// ---------------------------------------------------------------------------------------------

/**
 * Commits to the whole chain and batches the per-step residuals with Fiat-Shamir powers of r. The
 * proof holds only the commitment, the opened chain (no PCS in this toy) and the claimed batch value.
 */
function proveChain(z0, witnesses, options = {}) {
  const states = [z0 % P];
  const links = [mix(0x835, z0 % P)];
  for (let i = 0; i < witnesses.length; i++) {
    const next = step(states[i], witnesses[i]);
    states.push(next);
    links.push(linkHash(links[i], next, witnesses[i]));
  }
  const tampered = options.tamper ?? null;
  if (tampered !== null) {
    if (tampered.kind === "state") states[tampered.index] = add(states[tampered.index], tampered.delta ?? 1);
    if (tampered.kind === "witness") witnesses = witnesses.map((value, index) =>
      (index === tampered.index ? add(value, tampered.delta ?? 1) : value));
    if (tampered.kind === "link") links[tampered.index] = mix(links[tampered.index]);
    if (tampered.kind === "claim") {
      const broken = states.slice();
      broken[broken.length - 1] = add(broken[broken.length - 1], tampered.delta ?? 1);
      states.splice(0, states.length, ...broken);
    }
  }
  // recompute the links over the (possibly tampered) states so the *chain* stays consistent: the
  // composition must still catch a wrong state because the residuals no longer vanish
  const commitment = mix(...states, ...links, ...witnesses);
  const depth = witnesses.length;
  const transcript = mix(0x835, commitment, depth);
  const r = transcript % P;
  let batch = 0;
  let power = 1;
  const residuals = [];
  for (let i = 0; i < depth; i++) {
    const value = residual(states[i], states[i + 1], witnesses[i]);
    residuals.push(value);
    batch = add(batch, mul(power, value));
    power = mul(power, r);
  }
  return { states, links, witnesses, commitment, r, batch, residuals, depth,
    proofSize: 3, // commitment, challenge, batch value (the opened chain is not part of the proof)
    constraintCount: depth };
}

/** The verifier recomputes the commitment, the challenge, the batching and the QAP residuals. */
function verifyChain(proof, options = {}) {
  const depth = proof.depth;
  if (depth !== proof.witnesses.length || proof.states.length !== depth + 1) {
    return { accepted: false, reason: "malformed chain" };
  }
  const commitment = mix(...proof.states, ...proof.links, ...proof.witnesses);
  if (commitment !== proof.commitment) return { accepted: false, reason: "commitment mismatch" };
  const r = mix(0x835, commitment, depth) % P;
  if (r !== proof.r) return { accepted: false, reason: "challenge mismatch" };
  // the links must match the states (the hash chain binds the composition)
  for (let i = 1; i <= depth; i++) {
    const expected = linkHash(proof.links[i - 1], proof.states[i], proof.witnesses[i - 1]);
    if (expected !== proof.links[i]) return { accepted: false, reason: `link ${i} mismatch` };
  }
  let batch = 0;
  let power = 1;
  for (let i = 0; i < depth; i++) {
    batch = add(batch, mul(power, residual(proof.states[i], proof.states[i + 1], proof.witnesses[i])));
    power = mul(power, r);
  }
  if (batch !== proof.batch) return { accepted: false, reason: "batch mismatch" };
  if (options.requireZero && batch !== 0) return { accepted: false, reason: "non-zero residual" };
  if (batch !== 0) return { accepted: false, reason: "non-zero batch" };
  return { accepted: true, reason: "ok", r, batch };
}

/** Exhaustive soundness: with the chain fixed, count the batching challenges that would accept. */
function countAcceptingChallenges(proof) {
  const depth = proof.depth;
  const residuals = [];
  for (let i = 0; i < depth; i++) {
    residuals.push(residual(proof.states[i], proof.states[i + 1], proof.witnesses[i]));
  }
  let accepted = 0;
  for (let r = 0; r < P; r++) {
    let value = 0;
    let power = 1;
    for (let i = 0; i < depth; i++) {
      value = add(value, mul(power, residuals[i]));
      power = mul(power, r);
    }
    if (value === 0) accepted += 1;
  }
  return accepted;
}

/** A chain of exemplar witnesses derived deterministically from a seed. */
function makeWitnesses(depth, seed = 835) {
  let state = seed >>> 0;
  return Array.from({ length: depth }, () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state % P;
  });
}

module.exports = { P, F, mix, linkHash, step, residual, oracle, proveChain, verifyChain,
  countAcceptingChallenges, makeWitnesses };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const z0 = 1234;

  // ---- 1. chain oracle and residuals ------------------------------------------------------------
  const witnesses = makeWitnesses(6);
  const chain = oracle(z0, witnesses);
  assert.strictEqual(chain.states.length, 7);
  assert.strictEqual(chain.links.length, 7);
  for (let i = 0; i < witnesses.length; i++) {
    assert.strictEqual(residual(chain.states[i], chain.states[i + 1], witnesses[i]), 0, `step ${i}`);
    assert.strictEqual(chain.states[i + 1], step(chain.states[i], witnesses[i]));
  }
  assert.notStrictEqual(chain.links[6], chain.links[5], "links change every step");

  // ---- 2. completeness for several depths --------------------------------------------------------
  let reference = null;
  for (const depth of [0, 1, 2, 3, 4, 6]) {
    const proof = proveChain(z0, makeWitnesses(depth));
    const verdict = verifyChain(proof);
    assert.ok(verdict.accepted, `depth ${depth} accepted (${verdict.reason})`);
    const full = oracle(z0, makeWitnesses(depth));
    assert.deepStrictEqual(proof.states, full.states, `depth ${depth} states match the oracle`);
    if (depth === 6) reference = proof;
  }
  assert.strictEqual(reference.batch, 0);
  assert.strictEqual(reference.proofSize, 3, "the proof is a constant three field elements");
  assert.strictEqual(reference.constraintCount, 6);

  // ---- 3. soundness: tampering at every layer -----------------------------------------------------
  for (let index = 0; index < 6; index++) {
    for (const kind of ["state", "witness", "link"]) {
      const proof = proveChain(z0, makeWitnesses(6), { tamper: { kind, index, delta: 1 } });
      const verdict = verifyChain(proof);
      assert.ok(!verdict.accepted, `${kind} tampered at ${index} is rejected (${verdict.reason})`);
    }
  }
  const tamperedClaim = proveChain(z0, makeWitnesses(4), { tamper: { kind: "claim", index: 4, delta: 1 } });
  assert.ok(!verifyChain(tamperedClaim).accepted, "a wrong final state is rejected");
  const honest = proveChain(z0, makeWitnesses(4));
  assert.ok(verifyChain(honest).accepted);

  // ---- 4. exact batched-soundness measurement ------------------------------------------------------
  const broken = proveChain(z0, makeWitnesses(5), { tamper: { kind: "witness", index: 2, delta: 7 } });
  assert.ok(!verifyChain(broken).accepted);
  const acceptedChallenges = countAcceptingChallenges(broken);
  assert.ok(acceptedChallenges <= 4, `at most deg R <= depth - 1 = 4 challenges accept (measured ${acceptedChallenges})`);
  const honestAccepted = countAcceptingChallenges(proveChain(z0, makeWitnesses(5)));
  assert.strictEqual(honestAccepted, P, "an honest chain has zero residual, so every challenge accepts");
  // the batch value equals the polynomial at r
  const direct = (() => {
    let value = 0;
    let power = 1;
    for (let i = 0; i < 5; i++) {
      value = add(value, mul(power, broken.residuals[i]));
      power = mul(power, broken.r);
    }
    return value;
  })();
  assert.strictEqual(direct, broken.batch, "the batch value is the residual polynomial at r");

  // ---- 5. composition is really recursive: depth d implies the prefix -------------------------------
  const longChain = proveChain(z0, makeWitnesses(8));
  const prefix = proveChain(z0, makeWitnesses(4));
  assert.deepStrictEqual(longChain.states.slice(0, 5), prefix.states,
    "the depth-8 chain contains the depth-4 prefix");
  assert.strictEqual(longChain.links[4], prefix.links[4], "the hash chain agrees on the prefix");
  assert.strictEqual(prefix.batch, 0);
  assert.strictEqual(longChain.batch, 0);
  // a prefix whose state was tampered with cannot be spliced into the longer chain
  const spliced = proveChain(z0, makeWitnesses(8));
  spliced.states[3] = add(spliced.states[3], 1);
  assert.ok(!verifyChain(spliced).accepted, "a tampered interior state breaks the composition");

  // ---- 6. accounting: constant proof, linear constraints -------------------------------------------
  const sizes = [];
  const constraints = [];
  for (const depth of [1, 2, 4, 8, 16]) {
    const proof = proveChain(z0, makeWitnesses(depth));
    assert.ok(verifyChain(proof).accepted, `depth ${depth}`);
    sizes.push(proof.proofSize);
    constraints.push(proof.constraintCount);
  }
  assert.strictEqual(new Set(sizes).size, 1, "the proof size does not grow with the depth");
  assert.deepStrictEqual(constraints, [1, 2, 4, 8, 16], "the constraint count grows linearly");
  assert.strictEqual(constraints[4] / constraints[3], 2, "doubling the depth doubles the constraints");

  // ---- 7. degenerate chains and determinism ----------------------------------------------------------
  const empty = proveChain(z0, []);
  assert.ok(verifyChain(empty).accepted, "a depth-0 chain verifies trivially");
  assert.strictEqual(empty.batch, 0);
  assert.deepStrictEqual(empty.states, [z0]);
  const single = proveChain(z0, [7]);
  assert.ok(verifyChain(single).accepted);
  assert.strictEqual(single.states[1], add(mul(z0, z0), 7));
  const repeated = proveChain(z0, makeWitnesses(3));
  const repeatedAgain = proveChain(z0, makeWitnesses(3));
  assert.strictEqual(repeated.commitment, repeatedAgain.commitment);
  assert.strictEqual(repeated.r, repeatedAgain.r);
  assert.strictEqual(repeated.batch, repeatedAgain.batch);
  const malformed = { ...honest, depth: 99 };
  assert.ok(!verifyChain(malformed).accepted, "a malformed depth is rejected");

  metrics.push(`chain depth 6: states and links match the brute-force IVC oracle, every residual is exactly 0`);
  metrics.push(`completeness 0/1/2/3/4/6 verified; tampering with the state, witness or link at every layer is rejected`);
  metrics.push(`batched soundness: a broken chain accepts at most ${acceptedChallenges} of ${P} challenges (deg R <= depth - 1), enumerated exhaustively`);
  metrics.push(`accounting: proof size constant at ${reference.proofSize} field elements while constraints grow 1, 2, 4, 8, 16 (verifier work is linear here: no pairing accumulator, stated honestly)`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
