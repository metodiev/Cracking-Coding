/** Task 889 - Implement secure multiparty computation for private set intersection at scale.
 *  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * Two-party semi-honest PSI from arithmetic secret sharing over the 61-bit Mersenne prime
 * p = 2^61 - 1, following the standard cuckoo-hashing construction:
 *
 *  offline      a trusted dealer produces Beaver triples ([α], [β], [γ] with γ = αβ) — documented
 *               simplification, exactly the SPDZ preprocessing assumption; the online phase is real.
 *  hashing      both parties map each element to a field value v(e) = (FNV-1a(e) mod p-1) + 1 and use
 *               three public hash functions h1,h2,h3 into m >= 4n slots. Party A cuckoo-hashes its
 *               elements (one slot per element); party B places each element in all three buckets.
 *               A common element therefore shares a slot/bucket, so comparing per slot is exact.
 *  sharing      each party additively shares every value it owns: [x]_owner + [x]_peer = x mod p,
 *               with the peer's share = x - r for a uniform r (so A's message is perfectly hidden).
 *  comparison   per occupied slot A holds [a]; B's bucket holds [b_1..b_k]. The parties compute
 *               [d_i] = [a] - [b_i] (free, local), then the products Π[d_i] and finally Π[d_i]·[u]
 *               for a fresh shared random [u], using one Beaver triple per multiplication, and reveal
 *               the result z. Because a prime field has no zero divisors, z = 0 exactly when a equals
 *               some b_i (a match); for every non-match z = d·u is uniform over F_p^*, so the reveal
 *               leaks nothing but the output. Neither party ever sees an unshifted value.
 *
 * Invariants verified below: (1) correctness — the intersection equals a brute-force oracle on
 * overlapping, disjoint, empty, duplicate-carrying and 1000-element inputs; (2) information-theoretic
 * hiding — for a small prime the distributions of the masked shares and of the revealed z are
 * evaluated exhaustively and found identical for every secret (perfect secrecy of the offline masks),
 * and the non-match reveal is a bijection of F_p^*; (3) exactness of the zero test — a prime field
 * has no zero divisors, checked exhaustively; (4) data-obliviousness — the transcript carries only
 * shifted values (no element or raw field value appears on the wire); (5) collision handling — the
 * cuckoo table never silently drops or duplicates an element, and an unsatisfiable insertion is
 * reported as a reseed instead of returning a wrong answer; (6) cost — O(n) triples and O(n) field
 * elements on the wire rather than the O(n^2) of naive comparison; (7) determinism.
 *
 * Simplifications: one process models both parties plus an explicit in-memory network (message log),
 * the offline triples come from a dealer, and "at scale" is n = 1000 elements with a 61-bit field
 * instead of millions of records over a real socket.
 */
"use strict";

/* ------------------------------------------------------------------ determinism */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  /** Uniform field element in [0, prime). */
  field(prime) {
    let value = 0n;
    for (let i = 0; i < 4; i++) value = (value << 16n) | BigInt(this.next() & 0xffff);
    return value % prime;
  }
  /** Uniform nonzero field element. */
  unit(prime) { let value = this.field(prime); return value === 0n ? 1n : value; }
}

/* ------------------------------------------------------------------ field helpers */

const MERSENNE61 = (1n << 61n) - 1n;
const GENERATOR = 37n;
const PRIME_FACTORS_OF_P_MINUS_1 = [2n, 3n, 5n, 7n, 11n, 13n, 31n, 41n, 61n, 151n, 331n, 1321n];

const mod = (x, p) => ((x % p) + p) % p;

function modPow(base, exponent, prime) {
  let result = 1n;
  let b = mod(base, prime);
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % prime;
    b = (b * b) % prime;
    e >>= 1n;
  }
  return result;
}

/** 64-bit FNV-1a over the string, as a BigInt. */
function fnv64(text) {
  let hash = 0xcbf29ce484222325n;
  for (let i = 0; i < text.length; i++) {
    hash ^= BigInt(text.charCodeAt(i));
    hash = (hash * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return hash;
}

/** Maps an element into F_p^*, keeping 0 free to mean "empty slot". */
const fieldOf = (element, prime) => (fnv64(element) % (prime - 1n)) + 1n;

const slotOf = (element, prime, salt, slots) => Number(fnv64(`${salt}:${element}`) % BigInt(slots));

/* ------------------------------------------------------------------ network */

class Network {
  constructor() { this.log = []; this.cost = 0; }
  send(from, to, kind, values) {
    this.log.push({ from, to, kind, values: values.slice(), count: values.length });
    this.cost += values.length;
  }
  receivedBy(party) { return this.log.filter((message) => message.to === party); }
  /** Everything a network eavesdropper could see: kinds and payloads, never the labels. */
  transcript() { return this.log; }
}

/* ------------------------------------------------------------------ Beaver triples */

/** Trusted dealer: shares of α, β and γ = αβ. */
function dealTriples(count, prng, prime) {
  const triples = [];
  for (let i = 0; i < count; i++) {
    const alphaA = prng.field(prime);
    const alphaB = prng.field(prime);
    const betaA = prng.field(prime);
    const betaB = prng.field(prime);
    const alpha = mod(alphaA + alphaB, prime);
    const beta = mod(betaA + betaB, prime);
    const gamma = (alpha * beta) % prime;
    const gammaA = prng.field(prime);
    triples.push({ alphaA, alphaB, betaA, betaB, gammaA, gammaB: mod(gamma - gammaA, prime) });
  }
  return triples;
}

/**
 * Shared values are explicit pairs [A's share, B's share]; a reveal appends both the values that go
 * on the wire and the reconstructed value to the transcript.
 */
class SecureChannel {
  constructor(network, triples, prime) {
    this.network = network;
    this.triples = triples;
    this.prime = prime;
    this.nextTriple = 0;
    this.multiplications = 0;
    this.reveals = [];
  }

  /** The owner keeps r, the peer receives value - r; only the shifted value goes on the wire. */
  shareWith(owner, value, prng) {
    const mask = prng.field(this.prime);
    const peer = mod(value - mask, this.prime);
    if (owner === "A") this.network.send("A", "B", "share", [peer]);
    else this.network.send("B", "A", "share", [peer]);
    return owner === "A" ? [mask, peer] : [peer, mask];
  }

  /** Reveal a shared value to both parties (this is what the protocol output is derived from). */
  reveal(shared) {
    const value = mod(shared[0] + shared[1], this.prime);
    this.reveals.push(value);
    return value;
  }

  /** [x] * [y] via one Beaver triple: reveal e and f, then combine locally. */
  multiply(x, y) {
    const triple = this.triples[this.nextTriple++];
    if (!triple) throw new Error("offline triples exhausted");
    this.multiplications++;
    const e = this.reveal([mod(x[0] - triple.alphaA, this.prime), mod(x[1] - triple.alphaB, this.prime)]);
    const f = this.reveal([mod(y[0] - triple.betaA, this.prime), mod(y[1] - triple.betaB, this.prime)]);
    this.network.send("A", "B", "reveal", [e]);
    this.network.send("B", "A", "reveal", [f]);
    const gammaA = mod(triple.gammaA + e * triple.betaA + f * triple.alphaA + e * f, this.prime);
    const gammaB = mod(triple.gammaB + e * triple.betaB + f * triple.alphaB, this.prime);
    return [gammaA, gammaB];
  }

  /** Reveal to both parties: the value itself is public, the shares are not. */
  constant(value) { return [value, 0n]; }
}

/* ------------------------------------------------------------------ cuckoo hashing */

/**
 * Deterministic cuckoo hashing of A's elements into `slots` slots using three hash functions.
 * Returns { table, failures } — a failure is reported to the caller so it can reseed instead of
 * silently producing a wrong intersection.
 */
function cuckooPlace(elements, prime, seeds, slots, maxEvictions = 64) {
  const table = new Array(slots).fill(null);
  const placements = new Map();
  let failures = 0;
  let evictions = 0;
  const place = (element) => {
    let current = element;
    let saltIndex = 0;
    for (let step = 0; step < maxEvictions; step++) {
      const slot = slotOf(current, prime, seeds[saltIndex % seeds.length], slots);
      const displaced = table[slot];
      table[slot] = current;
      placements.set(current, slot);
      if (displaced === null) return true;
      current = displaced;
      saltIndex++;
      evictions++;
    }
    failures++;
    return false;
  };
  for (const element of elements) if (!place(element)) break;
  return { table, placements, failures, evictions };
}

/* ------------------------------------------------------------------ protocol */

/**
 * Semi-honest PSI. Returns { intersection, network, channel, stats }.
 * Messages only ever carry shifted field values or revealed masks.
 */
function runPsi(setA, setB, options = {}) {
  const prime = options.prime || MERSENNE61;
  const prng = options.prng || new Prng(options.seed || 889);
  const uniqueA = [...new Set(setA)];
  const uniqueB = [...new Set(setB)];
  const slots = options.slots || nextPowerOfTwo(4 * Math.max(uniqueA.length, 1));
  const seeds = options.seeds || ["h1", "h2", "h3"];
  const network = new Network();

  // ---- public hashing (performed independently by both parties)
  const cuckoo = cuckooPlace(uniqueA, prime, seeds, slots);
  if (cuckoo.failures > 0) {
    return { intersection: null, reseed: true, failures: cuckoo.failures, network, stats: { slots } };
  }
  const buckets = new Array(slots).fill(null).map(() => []);
  for (const element of uniqueB) {
    const value = fieldOf(element, prime);
    for (const seed of seeds) buckets[slotOf(element, prime, seed, slots)].push(value);
  }
  const occupied = cuckoo.table
    .map((value, slot) => ({ slot, value }))
    .filter((entry) => entry.value !== null);

  // ---- offline phase (trusted dealer)
  const placementsB = uniqueB.length * seeds.length;
  const channel = new SecureChannel(network, dealTriples(placementsB + occupied.length + 8, prng, prime), prime);

  // ---- A shares one value per occupied slot, B shares every element placement
  const sharedSlots = occupied.map((entry) => ({
    slot: entry.slot,
    label: entry.value,
    share: channel.shareWith("A", fieldOf(entry.value, prime), prng),
  }));
  const sharedBuckets = buckets.map((bucket) => bucket.map((value) => channel.shareWith("B", value, prng)));

  // ---- online phase: oblivious equality of A's slot value against B's bucket
  const matchedLabels = [];
  let pairsTested = 0;
  let slotsTested = 0;
  for (const entry of sharedSlots) {
    const bucket = sharedBuckets[entry.slot];
    if (bucket.length === 0) continue;   // structurally impossible to match: no reveal needed
    slotsTested++;
    let product = null;
    for (const candidate of bucket) {
      const difference = [
        mod(entry.share[0] - candidate[0], prime),
        mod(entry.share[1] - candidate[1], prime),
      ];
      product = product === null ? difference : channel.multiply(product, difference);
    }
    const blind = [prng.field(prime), prng.field(prime)];
    const masked = channel.multiply(product, blind);
    const z = channel.reveal(masked);
    pairsTested += bucket.length;
    if (z === 0n) matchedLabels.push(entry.label);
  }

  // Because a match means a == b in the field, the matched elements are exactly the intersection.
  const fieldToElements = new Map();
  for (const element of [...uniqueA, ...uniqueB]) fieldToElements.set(fieldOf(element, prime), element);
  const intersection = matchedLabels
    .map((label) => fieldToElements.get(fieldOf(label, prime)))
    .filter((element) => element !== undefined);
  return {
    intersection: [...new Set(intersection)].sort(),
    reseed: false,
    network,
    channel,
    stats: { slots, occupied: occupied.length, pairsTested, slotsTested, multiplications: channel.multiplications, communication: network.cost },
  };
}

function nextPowerOfTwo(value) {
  let size = 1;
  while (size < value) size *= 2;
  return size;
}

/* ------------------------------------------------------------------ oracle */

function bruteForceIntersection(a, b) {
  const right = new Set(b);
  return [...new Set(a)].filter((element) => right.has(element)).sort();
}

function makeUniverse(count, prefix = "user") {
  return Array.from({ length: count }, (_, i) => `${prefix}-${i}`);
}

module.exports = {
  Prng, Network, SecureChannel, dealTriples, runPsi, cuckooPlace, bruteForceIntersection,
  makeUniverse, fieldOf, slotOf, modPow, fnv64, mod, nextPowerOfTwo, MERSENNE61, GENERATOR,
  PRIME_FACTORS_OF_P_MINUS_1,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- field: 37 generates F_p^* (checked against every prime factor of p-1) ---------------
  for (const factor of PRIME_FACTORS_OF_P_MINUS_1) {
    assert.notStrictEqual(modPow(GENERATOR, (MERSENNE61 - 1n) / factor, MERSENNE61), 1n,
      `37 must not be a ${factor}-th power`);
  }
  assert.strictEqual(modPow(GENERATOR, MERSENNE61 - 1n, MERSENNE61), 1n, "Fermat");
  assert.strictEqual(modPow(2n, 61n, MERSENNE61), 1n, "2^61 = p + 1");
  assert.strictEqual(mod(-3n, 7n), 4n);

  // --- primitive roots cannot collide: distinct exponents give distinct group elements ------
  {
    const seen = new Set();
    for (let e = 1n; e < 200n; e++) seen.add(modPow(GENERATOR, e, MERSENNE61).toString());
    assert.strictEqual(seen.size, 199, "g^e must be injective for distinct exponents");
  }

  // --- perfect hiding of a shared value, verified exhaustively on a small prime -------------
  {
    const small = 101n;
    for (let value = 0n; value < small; value++) {
      const image = new Set();
      for (let mask = 0n; mask < small; mask++) image.add(mod(value - mask, small).toString());
      assert.strictEqual(image.size, Number(small), "x - r must cover the whole field exactly once");
    }
  }

  // --- the blinded zero test is exact: a prime field has no zero divisors -------------------
  {
    const small = 101n;
    for (let a = 0n; a < small; a++) {
      for (let b = 0n; b < small; b++) {
        if ((a * b) % small === 0n) assert.ok(a === 0n || b === 0n, `${a}*${b} vanished`);
      }
    }
    // For d != 0 the reveal z = d*u is a bijection of the nonzero field elements.
    for (let d = 1n; d < small; d++) {
      const image = new Set();
      for (let u = 1n; u < small; u++) image.add(((d * u) % small).toString());
      assert.strictEqual(image.size, Number(small) - 1, `multiplication by ${d} must be a bijection`);
    }
  }

  // --- cuckoo hashing places every element exactly once ------------------------------------
  {
    const elements = makeUniverse(200, "k");
    const placed = cuckooPlace(elements, MERSENNE61, ["h1", "h2", "h3"], 1024);
    assert.strictEqual(placed.failures, 0);
    const inTable = placed.table.filter((value) => value !== null);
    assert.strictEqual(inTable.length, elements.length, "one slot per element");
    assert.strictEqual(new Set(inTable).size, elements.length, "no element stored twice");
    for (const element of elements) {
      const slot = placed.placements.get(element);
      const candidates = ["h1", "h2", "h3"].map((seed) => slotOf(element, MERSENNE61, seed, 1024));
      assert.ok(candidates.includes(slot), "an element only ever sits in one of its own slots");
    }
  }

  // --- hashing is injective on a realistic universe (so no false positives) ----------------
  {
    const universe = makeUniverse(2000);
    const values = new Set(universe.map((element) => fieldOf(element, MERSENNE61).toString()));
    assert.strictEqual(values.size, universe.length, "distinct elements must have distinct field values");
    const small = makeUniverse(50);
    for (let i = 0; i < small.length; i++) {
      for (let j = i + 1; j < small.length; j++) {
        assert.notStrictEqual(fieldOf(small[i], MERSENNE61), fieldOf(small[j], MERSENNE61));
        assert.notStrictEqual(fieldOf(small[i], MERSENNE61), 0n, "0 is reserved for an empty slot");
      }
    }
  }

  // --- protocol correctness against the brute-force oracle ---------------------------------
  const universe = makeUniverse(600);
  const scenarios = [
    { name: "identical", a: universe.slice(0, 200), b: universe.slice(0, 200) },
    { name: "disjoint", a: universe.slice(0, 200), b: universe.slice(200, 400) },
    { name: "half", a: universe.slice(0, 200), b: universe.slice(100, 300) },
    { name: "empty-a", a: [], b: universe.slice(0, 50) },
    { name: "empty-b", a: universe.slice(0, 50), b: [] },
    { name: "both-empty", a: [], b: [] },
    { name: "singleton", a: ["user-7"], b: ["user-7"] },
    { name: "no-overlap-singleton", a: ["user-7"], b: ["user-8"] },
    { name: "duplicates", a: ["user-1", "user-1", "user-2", "user-3"], b: ["user-2", "user-2", "user-3"] },
  ];
  for (const scenario of scenarios) {
    const result = runPsi(scenario.a, scenario.b, { seed: 889 });
    assert.ok(!result.reseed, `${scenario.name} must not need a reseed`);
    assert.deepStrictEqual(result.intersection, bruteForceIntersection(scenario.a, scenario.b),
      `${scenario.name} intersection`);
  }

  // --- at scale: 1000 vs 1000 with a 300-element overlap ------------------------------------
  const prng = new Prng(20260889);
  const left = [];
  const right = [];
  const used = new Set();
  while (left.length < 700) { const value = `rec-${prng.next() % 100000}`; if (!used.has(value)) { used.add(value); left.push(value); } }
  for (const element of left.slice(0, 300)) right.push(element);
  while (right.length < 1000) {
    const value = `rec-${prng.next() % 100000}`;
    if (!used.has(value) && !right.includes(value)) right.push(value);
  }
  const large = runPsi(left, right, { seed: 4711 });
  assert.ok(!large.reseed);
  assert.deepStrictEqual(large.intersection, bruteForceIntersection(left, right), "large intersection");
  assert.strictEqual(large.intersection.length, 300);
  assert.ok(large.stats.slots >= 4 * left.length, "the table must stay sparsely loaded");
  assert.strictEqual(large.stats.occupied, left.length, "cuckoo places each of A's elements once");

  // --- cost: linear in n, and far cheaper than all-pairs comparison -------------------------
  const naivePairs = left.length * right.length;
  assert.ok(large.stats.communication <= 6 * (left.length + right.length),
    `communication ${large.stats.communication} must be linear`);
  assert.ok(large.stats.multiplications <= 4 * right.length,
    `multiplications ${large.stats.multiplications} must be linear`);
  assert.ok(naivePairs / large.stats.multiplications > 100, "the protocol must beat all-pairs by orders");

  // --- data obliviousness: no element and no raw field value ever appears on the wire ------
  {
    const transcript = JSON.stringify(large.network.transcript().map((message) => message.values.map((v) => v.toString())));
    for (const element of [...left.slice(0, 20), ...right.slice(0, 20)]) {
      assert.ok(!transcript.includes(element), `element ${element} leaked`);
    }
    const smallSet = ["user-3", "user-4"];
    const probe = runPsi(smallSet, ["user-4", "user-9"], { seed: 5 });
    const wire = new Set(probe.network.transcript().flatMap((message) => message.values.map((v) => v.toString())));
    for (const element of [...smallSet, "user-9"]) {
      assert.ok(!wire.has(fieldOf(element, MERSENNE61).toString()), "raw field values must never be sent");
    }
    assert.strictEqual(probe.channel.reveals.length, 2 * probe.stats.multiplications + probe.stats.slotsTested,
      "two mask reveals per multiplication plus one output reveal per tested slot");
  }

  // --- the reveal leaks only the output: matched slots reveal 0, others reveal uniformly ----
  {
    const small = 101n;
    const hits = [];
    const misses = [];
    const noise = new Prng(99);
    for (let trial = 0; trial < 400; trial++) {
      const match = trial % 2 === 0;
      const a = 7n + BigInt(trial % 30);
      const b = match ? a : a + 1n + BigInt(trial % 5);
      const d = mod(a - b, small);
      const u = noise.unit(small);
      const z = (d * u) % small;
      (match ? hits : misses).push(z);
    }
    assert.ok(hits.every((z) => z === 0n), "a match always reveals exactly zero");
    assert.ok(misses.every((z) => z !== 0n), "a mismatch never reveals zero");
    const missSet = new Set(misses.map((z) => z.toString()));
    assert.ok(missSet.size > 20, "the mismatch reveal spans many field values (no structure)");
  }

  // --- collision handling: an unsatisfiable placement is reported, not silently wrong ------
  {
    // Force a degenerate hash family (a single slot) so the cuckoo table cannot absorb 3 elements.
    const forced = cuckooPlace(["a", "b", "c"], MERSENNE61, ["same", "same", "same"], 1);
    assert.ok(forced.failures > 0, "a full table must report a failure");
    const retried = runPsi(["a", "b", "c"], ["b"], { seed: 1, seeds: ["same"], slots: 1 });
    assert.strictEqual(retried.reseed, true, "the protocol must ask for a reseed instead of lying");
    assert.strictEqual(retried.intersection, null);
    const proper = runPsi(["a", "b", "c"], ["b"], { seed: 1, slots: 16 });
    assert.deepStrictEqual(proper.intersection, ["b"]);
  }

  // --- triples are consumed exactly once and none are left dangling ------------------------
  {
    const probe = runPsi(universe.slice(0, 60), universe.slice(30, 90), { seed: 11 });
    assert.strictEqual(probe.channel.nextTriple, probe.stats.multiplications,
      "every multiplication consumes exactly one triple");
    assert.ok(probe.channel.reveals.every((value) => value >= 0n && value < MERSENNE61));
  }

  // --- determinism ------------------------------------------------------------------------
  const digest = () => JSON.stringify(runPsi(left.slice(0, 50), right.slice(0, 50), { seed: 7 }).intersection);
  assert.strictEqual(digest(), digest());

  console.log(`psi: ${left.length}v${right.length} elements, ${large.stats.slots} slots, ` +
    `${large.stats.multiplications} multiplies, ${large.stats.communication} field elements on the wire ` +
    `(all-pairs would need ${naivePairs})`);
  console.log("PASS");
}
