/** Task 758 - Implement secure multiparty computation basic XOR-sharing demo. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Two layers, both testable. (1) *XOR sharing* over wide values: a secret is split into
 * `parties` shares whose XOR is the secret, so linear functions are computed locally and for
 * free (the shares of `a ^ b` are the component-wise XOR of the shares of `a` and `b`).
 * (2) *GF(2) circuit evaluation* with the standard 2-/n-party Beaver-triple protocol for the
 * one non-linear gate: for `z = x & y` the parties hold shares of a random triple (a, b, c = ab),
 * open `d = x ^ a` and `e = y ^ b` (the only communication), and compute
 * `z_i = c_i ^ (d & b_i) ^ (e & a_i) ^ [i == 0] (d & e)` locally. XOR-wise summing the z_i
 * gives `ab ^ d*b ^ e*a ^ de = xy`, so the protocol is correct; and `d`/`e` are one-time pads
 * that make the transcript independent of the secret inputs.
 * Privacy is not asserted but *enumerated*: because the randomness space is tiny, the test
 * collects the complete multiset of party 0's views (its shares of every wire plus every opened
 * value) over all randomnesses, for every input of the other party, and requires those multisets
 * to be identical — the exhaustive form of "the view is simulatable"/"any n-1 shares reveal
 * nothing".
 *
 * What is simplified: only the GF(2) gate set XOR/NOT/AND for circuits, only 2 and 3 parties are
 * exercised, shares are host numbers (not field elements of a large prime field), the toy triples
 * are drawn from a seeded PRNG instead of being precomputed by an OT protocol, and there is no
 * honest-majority or malicious-security model — this models the *shape* of additive-sharing MPC,
 * never real cryptography. What is verified: share/reconstruct round-trips over 1..8 bit secrets
 * and 1..5 parties; the linearity of XOR sharing; exhaustive perfect secrecy of both the sharing
 * and the AND protocol; full-adder and 4-bit ripple-carry-adder circuits evaluated in MPC agreeing
 * with plaintext evaluation (all 256 operand pairs); the transcript contains exactly two bits per
 * AND gate; AND-free circuits keep the transcript empty; and input validation.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

const BIG = 1n;

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x758758;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** XOR-share a wide value: the last share absorbs the parity of the random masks. */
function xorShare(secret, parties, rng, width = 8) {
  if (!Number.isInteger(parties) || parties < 1) throw new RangeError("party count must be a positive integer");
  if (!Number.isInteger(width) || width < 1) throw new RangeError("width must be a positive integer");
  const mask = (BIG << BigInt(width)) - 1n;
  const shares = [];
  let parity = 0n;
  for (let party = 0; party < parties - 1; party++) {
    const value = BigInt(rng(Number(1n << BigInt(width))));
    shares.push(value);
    parity ^= value;
  }
  shares.push((secret & mask) ^ parity);
  return shares;
}

function xorReconstruct(shares) {
  if (!Array.isArray(shares) || shares.length === 0) throw new RangeError("at least one share is required");
  return shares.reduce((accumulator, share) => accumulator ^ share, 0n);
}

const xorShares = (left, right) => left.map((share, i) => share ^ right[i]);

/** One-bit sharing, the shape the gate protocol works with. */
function bitShare(secret, parties, rng) {
  const shares = [];
  let parity = 0;
  for (let party = 0; party < parties - 1; party++) {
    const bit = rng(2);
    shares.push(bit);
    parity ^= bit;
  }
  shares.push(secret ^ parity);
  return shares;
}

const reconstructBits = (shares) => shares.reduce((accumulator, bit) => accumulator ^ bit, 0);

/** Beaver triple for `AND`: shared random a, b and their product. */
function beaverTriple(parties, rng) {
  const a = bitShare(rng(2), parties, rng);
  const b = bitShare(rng(2), parties, rng);
  return { a, b, c: bitShare(reconstructBits(a) & reconstructBits(b), parties, rng) };
}

/** The one communication round of the AND gate: open d = x ^ a and e = y ^ b. */
function andProtocol(xShares, yShares, triple) {
  const d = reconstructBits(xShares.map((share, i) => share ^ triple.a[i]));
  const e = reconstructBits(yShares.map((share, i) => share ^ triple.b[i]));
  const z = triple.c.map((share, i) =>
    share ^ (d & triple.b[i]) ^ (e & triple.a[i]) ^ (i === 0 ? d & e : 0));
  return { z, opened: [d, e] };
}

/** Plaintext evaluation of a gate list, used as the reference oracle. */
function plaintextEvaluate(circuit, inputs) {
  const values = { ...inputs };
  for (const gate of circuit.gates) {
    if (gate.op === "xor") values[gate.out] = values[gate.left] ^ values[gate.right];
    else if (gate.op === "not") values[gate.out] = values[gate.left] ^ 1;
    else if (gate.op === "and") values[gate.out] = values[gate.left] & values[gate.right];
    else throw new TypeError(`unknown gate '${gate.op}'`);
  }
  return Object.fromEntries(circuit.outputs.map((wire) => [wire, values[wire]]));
}
/**
 * MPC evaluation: every wire carries one share per party, XOR/NOT are local, AND consumes a
 * fresh Beaver triple. Returns the per-party output shares and the public transcript.
 */
function runMpc(circuit, sharesByWire, parties, rng) {
  const values = {};
  for (const [wire, shares] of Object.entries(sharesByWire)) {
    if (shares.length !== parties) throw new RangeError(`wire '${wire}' has ${shares.length} shares, expected ${parties}`);
    values[wire] = shares.slice();
  }
  const transcript = [];
  for (const gate of circuit.gates) {
    if (gate.op === "xor") values[gate.out] = values[gate.left].map((share, i) => share ^ values[gate.right][i]);
    else if (gate.op === "not") values[gate.out] = values[gate.left].map((share) => share ^ 1);
    else if (gate.op === "and") {
      const { z, opened } = andProtocol(values[gate.left], values[gate.right], beaverTriple(parties, rng));
      transcript.push({ gate: gate.out, opened });
      values[gate.out] = z;
    } else {
      throw new TypeError(`unknown gate '${gate.op}'`);
    }
  }
  return {
    outputs: Object.fromEntries(circuit.outputs.map((wire) => [wire, values[wire]])),
    transcript,
    andGates: circuit.gates.filter((gate) => gate.op === "and").length,
  };
}
/** One-bit full adder: sum = x ^ y ^ cin, carry-out = (x & y) ^ (cin & (x ^ y)). */
function fullAdderCircuit() {
  return {
    inputs: ["x", "y", "cin"],
    gates: [
      { op: "xor", left: "x", right: "y", out: "t1" },
      { op: "xor", left: "t1", right: "cin", out: "sum" },
      { op: "and", left: "x", right: "y", out: "t2" },
      { op: "and", left: "cin", right: "t1", out: "t3" },
      { op: "xor", left: "t2", right: "t3", out: "carry" },
    ],
    outputs: ["sum", "carry"],
  };
}
/** Ripple-carry adder over `bits` width, built from full adders. */
function rippleAdderCircuit(bits) {
  const gates = [];
  let carry = "cin";
  for (let i = 0; i < bits; i++) {
    gates.push({ op: "xor", left: `x${i}`, right: `y${i}`, out: `t1_${i}` });
    gates.push({ op: "xor", left: `t1_${i}`, right: carry, out: `s${i}` });
    gates.push({ op: "and", left: `x${i}`, right: `y${i}`, out: `t2_${i}` });
    gates.push({ op: "and", left: carry, right: `t1_${i}`, out: `t3_${i}` });
    gates.push({ op: "xor", left: `t2_${i}`, right: `t3_${i}`, out: `c${i}` });
    carry = `c${i}`;
  }
  return {
    inputs: ["cin", ...Array.from({ length: bits }, (_, i) => `x${i}`), ...Array.from({ length: bits }, (_, i) => `y${i}`)],
    gates,
    outputs: [...Array.from({ length: bits }, (_, i) => `s${i}`), carry],
  };
}

/** Shares every circuit input bit across the parties. */
function shareInputs(circuit, assignment, parties, rng) {
  const sharesByWire = {};
  for (const wire of circuit.inputs) sharesByWire[wire] = bitShare(assignment[wire], parties, rng);
  return sharesByWire;
}
const bitsToNumber = (bits) => bits.reduce((accumulator, bit, i) => accumulator | (bit << i), 0);

module.exports = {
  makeRng, xorShare, xorReconstruct, xorShares, bitShare, reconstructBits, beaverTriple,
  andProtocol, plaintextEvaluate, runMpc, fullAdderCircuit, rippleAdderCircuit, shareInputs,
};

if (require.main === module) {
  const rng = makeRng(0x758758);

  // 1. share/reconstruct round-trip over widths and party counts
  let roundTrips = 0;
  for (let width = 1; width <= 8; width++) {
    for (let parties = 1; parties <= 5; parties++) {
      for (let secret = 0; secret < (1 << width); secret += Math.max(1, (1 << width) >> 3)) {
        const shares = xorShare(BigInt(secret), parties, rng, width);
        strictEqual(shares.length, parties);
        strictEqual(xorReconstruct(shares), BigInt(secret), `${secret} with ${parties} parties of width ${width}`);
        roundTrips++;
      }
    }
  }
  // 2. XOR sharing is linear: shares of a^b are the component-wise XOR of the shares
  for (let trial = 0; trial < 20; trial++) {
    const a = BigInt(rng(256));
    const b = BigInt(rng(256));
    const sharedA = xorShare(a, 3, rng, 8);
    const sharedB = xorShare(b, 3, rng, 8);
    const summed = xorShares(sharedA, sharedB);
    strictEqual(xorReconstruct(summed), a ^ b, "local addition is correct in MPC");
    strictEqual(xorReconstruct(xorShare(a ^ b, 3, makeRng(trial + 1), 8)), xorReconstruct(summed),
      "an independently drawn sharing reconstructs the same sum");
  }

  // 3. exhaustive perfect secrecy of the sharing: party 0's share is uniform, whatever the secret
  const shareViews = (secret, parties, width) => {
    const views = new Set();
    for (let seed = 0; seed < 64; seed++) {
      const source = makeRng(seed * 2654435761 + 1);
      views.add(xorShare(BigInt(secret), parties, source, width)[0].toString());
    }
    return [...views].sort().join(",");
  };
  strictEqual(shareViews(0, 2, 3), shareViews(7, 2, 3), "party 0's view is identical for every secret (2 parties, 3 bits)");
  strictEqual(shareViews(0, 4, 2), shareViews(3, 4, 2), "and for 4 parties over 2 bits");
  ok(shareViews(0, 2, 3).split(",").length === 8, "the view covers the whole share space");

  // 4. exhaustive perfect secrecy of the AND protocol over the full randomness space
  const partyZeroViews = (x0, y0, x1, y1) => {
    const views = [];
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 2; b++) {
        for (let ra = 0; ra < 2; ra++) {
          for (let rb = 0; rb < 2; rb++) {
            for (let rc = 0; rc < 2; rc++) {
              for (let rx = 0; rx < 2; rx++) {
                for (let ry = 0; ry < 2; ry++) {
                  const xShares = [rx, x0 ^ x1 ^ rx];
                  const yShares = [ry, y0 ^ y1 ^ ry];
                  const triple = { a: [ra, a ^ ra], b: [rb, b ^ rb], c: [rc, (a & b) ^ rc] };
                  const { z, opened } = andProtocol(xShares, yShares, triple);
                  views.push([xShares[0], yShares[0], opened[0], opened[1], z[0]].join(""));
                }
              }
            }
          }
        }
      }
    }
    return views.sort().join("|");
  };
  const baselineView = partyZeroViews(0, 0, 0, 0);
  for (const [x1, y1] of [[0, 1], [1, 0], [1, 1]]) {
    strictEqual(partyZeroViews(0, 0, x1, y1), baselineView, `party 0's view must not depend on party 1's input (${x1},${y1})`);
  }
  strictEqual(baselineView.split("|").length, 128, "the whole randomness space is enumerated");
  ok(new Set(baselineView.split("|")).size >= 16, "the view is far from degenerate");

  // 5. the AND protocol is correct for every input pair with freshly drawn triples
  let andChecks = 0;
  for (let x = 0; x < 2; x++) {
    for (let y = 0; y < 2; y++) {
      for (let trial = 0; trial < 8; trial++) {
        const triple = beaverTriple(2, makeRng(trial * 31 + 7));
        const xShares = bitShare(x, 2, makeRng(trial + 1));
        const yShares = bitShare(y, 2, makeRng(trial + 101));
        strictEqual(reconstructBits(andProtocol(xShares, yShares, triple).z), x & y, `${x} & ${y}, trial ${trial}`);
        andChecks++;
      }
    }
  }

  // 6. the full adder agrees with plaintext evaluation for every input, with 2 and 3 parties
  const adder = fullAdderCircuit();
  let adderRuns = 0;
  for (let parties = 2; parties <= 3; parties++) {
    for (let assignment = 0; assignment < 8; assignment++) {
      const bits = { x: assignment & 1, y: (assignment >> 1) & 1, cin: (assignment >> 2) & 1 };
      const expected = plaintextEvaluate(adder, bits);
      const result = runMpc(adder, shareInputs(adder, bits, parties, makeRng(assignment + parties * 31)), parties, makeRng(99));
      for (const wire of adder.outputs) {
        strictEqual(reconstructBits(result.outputs[wire]), expected[wire], `${wire} with ${parties} parties, input ${assignment}`);
      }
      strictEqual(result.transcript.length, result.andGates, "one transcript entry per AND gate");
      for (const entry of result.transcript) ok(entry.opened.every((bit) => bit === 0 || bit === 1), "only bits are opened");
      adderRuns++;
    }
  }

  // 7. the 4-bit ripple-carry adder agrees with integer addition for all 256 operand pairs
  const ripple = rippleAdderCircuit(4);
  let sums = 0;
  for (let left = 0; left < 16; left++) {
    for (let right = 0; right < 16; right++) {
      const assignment = { cin: 0 };
      for (let bit = 0; bit < 4; bit++) {
        assignment[`x${bit}`] = (left >> bit) & 1;
        assignment[`y${bit}`] = (right >> bit) & 1;
      }
      const shares = shareInputs(ripple, assignment, 2, makeRng(left * 16 + right + 1));
      const result = runMpc(ripple, shares, 2, makeRng(left * 16 + right + 7));
      const plain = plaintextEvaluate(ripple, assignment);
      const sumBits = [];
      for (let bit = 0; bit < 4; bit++) {
        strictEqual(reconstructBits(result.outputs[`s${bit}`]), plain[`s${bit}`], `bit ${bit} of ${left}+${right}`);
        sumBits.push(reconstructBits(result.outputs[`s${bit}`]));
      }
      sumBits.push(reconstructBits(result.outputs.c3));
      strictEqual(sumBits[4], plain.c3, `carry-out of ${left}+${right} agrees with the plaintext adder`);
      strictEqual(bitsToNumber(sumBits), left + right, `${left} + ${right} computed in MPC`);
      sums++;
    }
  }

  // 8. an AND-free circuit never communicates: the transcript is empty
  const xonly = {
    inputs: ["x", "y"],
    gates: [
      { op: "xor", left: "x", right: "y", out: "t" },
      { op: "not", left: "t", out: "z" },
    ],
    outputs: ["t", "z"],
  };
  for (let assignment = 0; assignment < 4; assignment++) {
    const bits = { x: assignment & 1, y: (assignment >> 1) & 1 };
    const result = runMpc(xonly, shareInputs(xonly, bits, 3, makeRng(assignment + 5)), 3, makeRng(1));
    const expected = plaintextEvaluate(xonly, bits);
    for (const wire of xonly.outputs) strictEqual(reconstructBits(result.outputs[wire]), expected[wire]);
    deepStrictEqual(result.transcript, [], "XOR and NOT are free: nothing is opened");
  }

  // 9. boundaries: a single party, secret zero, all-ones secret, a constant-triple edge case
  deepStrictEqual(xorShare(0n, 1, makeRng(1), 8), [0n], "one party holds the secret itself");
  deepStrictEqual(xorShare(255n, 1, makeRng(1), 8), [255n]);
  const ones = xorShare(255n, 3, makeRng(2), 8);
  strictEqual(xorReconstruct(ones), 255n);
  ok(ones.every((share) => share >= 0n && share < 256n), "shares stay inside the width");
  const constantTriple = { a: [0, 0], b: [0, 0], c: [0, 0] };
  strictEqual(reconstructBits(andProtocol([1, 0], [0, 1], constantTriple).z), 1, "a degenerate triple is still correct");
  // 10. validation
  throws(() => xorShare(1n, 0, makeRng(1)), RangeError);
  throws(() => xorShare(1n, 2, makeRng(1), 0), RangeError);
  throws(() => xorReconstruct([]), RangeError);
  throws(() => runMpc(adder, { x: [0] }, 2, makeRng(1)), RangeError);
  throws(() => plaintextEvaluate({ gates: [{ op: "nand", out: "q" }], outputs: ["q"] }, {}), TypeError);

  console.log(
    `roundTrips=${roundTrips} andChecks=${andChecks} adderRuns=${adderRuns} rippleSums=${sums} ` +
    `andGatesPerRun=${runMpc(adder, shareInputs(adder, { x: 1, y: 1, cin: 0 }, 2, makeRng(3)), 2, makeRng(4)).andGates}`
  );
  console.log("PASS");
}
