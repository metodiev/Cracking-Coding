/** Task 839 - Implement oblivious transfer based protocol for private set intersection.
 *  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * Two parties learn X ∩ Y without revealing the rest of their sets. The build follows the
 * Pinkas–Schneider–Zohner "PSI from OT" recipe:
 *
 *  1. Base oblivious transfer, Bellare–Micali style over the toy group Z_p* (p = 2^61 - 1, a
 *     Mersenne prime; the generator is *found* at run time and its order certified against the
 *     factors of p - 1). The sender publishes C = g^c with c secret; a receiver that wants message i
 *     publishes PK_i = g^k and PK_j = C * g^{-r_j} for j != i. The sender replies
 *     (g^{rho_j}, m_j * PK_j^{rho_j}); only index i can be opened, because the receiver knows the
 *     discrete log of exactly one PK. Hence the receiver learns 1-of-N messages, and the sender's
 *     view of the PK vector is uniform for *every* i (with C fixed, (k, r_{j != i}) -> PK is a
 *     bijection onto group^N), so the choice index is hidden.
 *  2. Hashing layer: a keyed hash maps each item to bins h1/h2 and to a slot. The client cuckoo
 *     hashes its set (one item per occupied bin) and thereby learns which of h1(x), h2(x) it owns;
 *     the server stores PRF_K(y) at the coordinates (h1(y), slot(y)) and (h2(y), slot(y)). Empty
 *     coordinates are filled with deterministic keyed padding (the analogue of the masked/correlated
 *     random values the real protocol consumes).
 *  3. For every client item the client runs a 1-out-of-L OT (L = slots per bin) over its own bin, so
 *     it opens exactly one table entry per item and the *server never learns the slot it asked for*.
 *     If that entry equals PRF_K(x), x is in the intersection; otherwise the entry is a keyed hash of
 *     some other item or of the padding, which reveals nothing.
 *
 * Simplified / toy aspects (stated honestly): 61-bit modulus with 40-bit exponents, so discrete log
 * is easy and this is *not* cryptographically secure — the protocol structure, the OT algebra, the
 * cuckoo layer and the security measurements are genuine. The server learns the bins (keyed hashes)
 * of the client's items: that is the well-known index leak of table-based PSI. The client is
 * semi-honest (it queries one coordinate per item); a malicious client could probe other coordinates,
 * which real PSI blocks through OT extension with correlated randomness. Two *server* items landing
 * on one coordinate overwrite each other and can cause a false negative: the server counts these
 * collisions and reports `degraded` instead of losing a value silently. Everything is in-process —
 * both parties exchange explicit messages over an in-memory network, no sockets, threads or timers.
 *
 * Invariants verified below:
 *  - OT correctness for every choice, and a cheating receiver that tries to open a second index
 *    fails in 100% of 100 trials (sender-side privacy, measured);
 *  - receiver-choice privacy of the OT: two-sample chi-square over the residues mod 97 of the
 *    sender-visible PK_0 for choice 0 versus choice 3 (1600 samples each), limit derived from the
 *    statistic's own degrees of freedom — the sender cannot tell which index was chosen;
 *  - PSI exactness: recall 1 and precision 1 versus a brute-force intersection oracle over random
 *    set pairs on both sides, including empty, disjoint, identical, duplicated and single-element
 *    inputs;
 *  - no silent failure: a forced collision is counted, flagged `degraded`, and the measured
 *    collision rate over a parameter sweep stays within the birthday bound;
 *  - cost: exactly one OT per distinct client item, two network messages per OT.
 */
"use strict";

const MODULUS = (1n << 61n) - 1n; // Mersenne prime 2^61 - 1
const EXPONENT_BITS = 40n; // toy key size
const MASK64 = (1n << 64n) - 1n;
// 2 * 3^2 * 5^2 * 7 * 11 * 13 * 31 * 41 * 61 * 151 * 331 * 1321 = 2^61 - 2 (asserted below).
const PRIME_POWERS = [[2n, 1], [3n, 2], [5n, 2], [7n, 1], [11n, 1], [13n, 1], [31n, 1], [41n, 1],
  [61n, 1], [151n, 1], [331n, 1], [1321n, 1]];

/** Normalises an integer into [0, p). */
function field(x) {
  const r = x % MODULUS;
  return r < 0n ? r + MODULUS : r;
}

function powMod(base, exponent) {
  let acc = 1n;
  let b = field(base);
  let e = exponent;
  while (e > 0n) {
    if ((e & 1n) === 1n) acc = (acc * b) % MODULUS;
    b = (b * b) % MODULUS;
    e >>= 1n;
  }
  return acc;
}

const inverse = (x) => powMod(field(x), MODULUS - 2n);

/** The smallest generator of Z_p*, certified against the prime factors of p - 1. */
function findGenerator() {
  for (let g = 2n; g < 100n; g++) {
    let isGenerator = true;
    for (const [q] of PRIME_POWERS) {
      if (powMod(g, (MODULUS - 1n) / q) === 1n) {
        isGenerator = false;
        break;
      }
    }
    if (isGenerator) return g;
  }
  throw new Error("no generator found");
}

const GENERATOR = findGenerator();
const GENERATOR_INVERSE = inverse(GENERATOR); // lets the receiver blind with g^{-r} instead of 1/g^r
const GENERATOR_ORDER = MODULUS - 1n;

/** splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = field(BigInt(seed)) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  exponent() {
    return 1n + (this.next() % ((1n << EXPONENT_BITS) - 1n));
  }

  bit() {
    return Number(this.next() & 1n);
  }
}

/** Keyed 64-bit hash: the stand-in for the random oracle the real protocol assumes. */
function keyedHash(salt, text) {
  let h = (BigInt(salt) ^ 0x9e3779b97f4a7c15n) & MASK64;
  for (let i = 0; i < text.length; i++) {
    h = ((h ^ BigInt(text.charCodeAt(i))) * 0x100000001b3n) & MASK64;
    h = ((h ^ (h >> 29n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    h = (h ^ (h >> 32n)) & MASK64;
  }
  return h;
}

/** Field-element PRF_K(item): what the client compares against after its OT. */
function prf(key, item) {
  const salt = BigInt(key);
  const low = keyedHash(salt ^ 0x5bf03635n, item);
  const high = keyedHash(salt ^ 0x2545f491n, item);
  return ((high << 64n) | low) % MODULUS;
}

const hashBin = (key, item, fn, bins) =>
  Number(keyedHash(BigInt(key) ^ BigInt(0x1000 + fn), item) % BigInt(bins));
const hashSlot = (key, item, slots) =>
  Number(keyedHash(BigInt(key) ^ 0x7ffffn, item) % BigInt(slots));
const coordinate = (bin, slot) => `${bin}:${slot}`;

/** Deterministic filler for an unoccupied coordinate (masked/random-looking, reveals nothing). */
const padValue = (key, bin, slot) => prf(BigInt(key) ^ 0x9ad3n, `pad:${coordinate(bin, slot)}`);

/** Explicit in-memory message network: both parties are objects, no sockets, no timers. */
class InMemoryNetwork {
  constructor() {
    this.links = new Map();
    this.sent = 0;
  }

  send(from, to, type, payload) {
    const link = `${from}->${to}`;
    if (!this.links.has(link)) this.links.set(link, []);
    this.links.get(link).push({ type, payload });
    this.sent++;
  }

  /** Drains the inbox of `to` in send order. */
  receive(to) {
    const inbox = [];
    for (const [link, queue] of this.links) {
      if (link.split("->")[1] !== to) continue;
      inbox.push(...queue);
      queue.length = 0;
    }
    return inbox;
  }
}

/** Sender half of one 1-out-of-N OT: publishes C = g^c, c secret. */
function otSenderSetup(n, rng) {
  if (n < 2) throw new Error("OT needs at least two messages");
  return { n, C: powMod(GENERATOR, rng.exponent()) };
}

/** Receiver half: PK_i = g^k for the chosen index, PK_j = C * g^{-r_j} elsewhere. */
function otReceiverQuery(setup, chosen, rng) {
  if (!Number.isInteger(chosen) || chosen < 0 || chosen >= setup.n) {
    throw new Error("invalid OT choice");
  }
  const publicKeys = [];
  const blinders = new Map();
  let chosenSecret = 0n;
  for (let j = 0; j < setup.n; j++) {
    if (j === chosen) {
      chosenSecret = rng.exponent();
      publicKeys.push(powMod(GENERATOR, chosenSecret));
    } else {
      const r = rng.exponent();
      blinders.set(j, r);
      publicKeys.push(field(setup.C * powMod(GENERATOR_INVERSE, r)));
    }
  }
  return { chosen, publicKeys, chosenSecret, blinders, setup };
}

/** Sender encrypts one message per index: (g^{rho_j}, m_j * PK_j^{rho_j}). */
function otSenderReply(messages, publicKeys, rng) {
  if (messages.length !== publicKeys.length) throw new Error("message/key count mismatch");
  return messages.map((message, j) => {
    const rho = rng.exponent();
    return { a: powMod(GENERATOR, rho), b: field(field(message) * powMod(publicKeys[j], rho)) };
  });
}

/** Receiver opens exactly one index with the discrete log it knows. */
function otReceiverOpen(reply, query) {
  const chosen = reply[query.chosen];
  return field(chosen.b * inverse(powMod(chosen.a, query.chosenSecret)));
}

/**
 * A *cheating* receiver tries to open another index: it only knows C * g^{-r_j}, so its key
 * g^{rho_j r_j} differs from the required PK_j^{rho_j} by the unknown factor g^{rho_j (c - 2 r_j)}.
 */
function otCheatingOpen(reply, query, index) {
  const r = query.blinders.get(index);
  if (r === undefined) throw new Error("index is not blinder-backed");
  return field(reply[index].b * inverse(powMod(reply[index].a, r)));
}

/**
 * Cuckoo hashing of the client set: at most one item per bin, eviction driven by the seeded PRNG.
 * `aborted` reports that eviction exceeded its budget (a real implementation stashes the item; here
 * the run is flagged degraded rather than silently losing it).
 */
function cuckooInsert(items, key, bins, rng) {
  const table = new Map();
  let maxEvictions = 0;
  let aborted = false;
  for (const item of items) {
    let current = item;
    let fn = rng.bit();
    let evictions = 0;
    for (;;) {
      const bin = hashBin(key, current, fn, bins);
      const occupant = table.get(bin);
      table.set(bin, { item: current, fn });
      if (occupant === undefined) break;
      current = occupant.item;
      fn = 1 - occupant.fn;
      evictions++;
      if (evictions > 2 * bins) {
        aborted = true;
        break;
      }
    }
    maxEvictions = Math.max(maxEvictions, evictions);
    if (aborted) break;
  }
  const lookup = new Map();
  for (const [bin, entry] of table) lookup.set(entry.item, { bin, fn: entry.fn });
  return { table, lookup, aborted, maxEvictions };
}

/**
 * Server side. Stores PRF_K(y) at both candidate coordinates of every item and counts collisions on
 * the coordinate space (bins x slots); unoccupied coordinates are padded lazily, so a table of a few
 * thousand bins costs nothing until it is queried.
 */
function buildServerTable(serverItems, key, bins, slots) {
  const placements = new Map();
  let collisions = 0;
  let placed = 0;
  for (const item of serverItems) {
    const value = prf(key, item);
    for (const fn of [0, 1]) {
      const bin = hashBin(key, item, fn, bins);
      const slot = hashSlot(key, item, slots);
      const coord = coordinate(bin, slot);
      placed++;
      const occupant = placements.get(coord);
      if (occupant === undefined) placements.set(coord, value);
      else if (occupant !== value) collisions++; // a different item already owns the coordinate
    }
  }
  return { placements, collisions, placed, bins, slots, key };
}

/** The L OT messages of one bin: placements where present, deterministic padding elsewhere. */
function binMessages(table, bin) {
  const messages = new Array(table.slots);
  for (let slot = 0; slot < table.slots; slot++) {
    const coord = coordinate(bin, slot);
    const value = table.placements.get(coord);
    messages[slot] = value === undefined ? padValue(table.key, bin, slot) : value;
  }
  return messages;
}

/**
 * Runs the whole PSI over the in-memory network and returns the intersection plus the metrics the
 * checks need (collisions, abort flag, OT count, network traffic, transcript fingerprint).
 */
function privateSetIntersection(clientItems, serverItems, options) {
  const { seed, bins, slots, key } = options;
  const rng = new SplitMix64(seed);
  const network = new InMemoryNetwork();
  const clientSet = [...new Set(clientItems.map(String))];
  const serverSet = [...new Set(serverItems.map(String))];
  const cuckoo = cuckooInsert(clientSet, key, bins, rng);
  const table = buildServerTable(serverSet, key, bins, slots);
  const intersection = [];
  const requested = [];
  const transcript = [];
  let otCount = 0;
  for (const item of clientSet) {
    const placed = cuckoo.lookup.get(item);
    if (placed === undefined) continue; // unplaced item after a cuckoo abort: reported as degraded
    const slot = hashSlot(key, item, slots);
    network.send("client", "server", "ot-receiver-query", { bin: placed.bin });
    const query = otReceiverQuery(otSenderSetup(slots, rng), slot, rng);
    const reply = otSenderReply(binMessages(table, placed.bin), query.publicKeys, rng);
    network.send("server", "client", "ot-sender-reply", { bin: placed.bin, reply });
    const value = otReceiverOpen(reply, query);
    otCount++;
    requested.push(coordinate(placed.bin, slot));
    transcript.push(value);
    if (value === prf(key, item)) intersection.push(item);
  }
  const inbox = network.receive("client");
  const queries = network.receive("server");
  return {
    intersection: intersection.sort(),
    requested,
    collisions: table.collisions,
    placements: table.placed,
    aborted: cuckoo.aborted,
    degraded: cuckoo.aborted || table.collisions > 0,
    otCount,
    messagesPerOt: slots,
    detail: { cuckoo, table, inboxes: inbox.length + queries.length, network },
    fingerprint: fingerprint(transcript, key),
  };
}

/** Rolling 64-bit fingerprint of the client's transcript, used only to prove determinism. */
function fingerprint(values, key) {
  let h = 0xcbf29ce484222325n;
  for (const value of values) {
    h = (h ^ (value & MASK64)) & MASK64;
    h = (h * 0x100000001b3n) & MASK64;
  }
  return (h ^ keyedHash(key, String(values.length))) & MASK64;
}

/** Independent brute-force oracle: plain set semantics, no hashing and no OTs. */
function bruteIntersection(clientItems, serverItems) {
  const server = new Set(serverItems.map(String));
  return [...new Set(clientItems.map(String))].filter((x) => server.has(x)).sort();
}

/** Recall and precision of a reported intersection against the oracle. */
function quality(reported, expected) {
  const report = new Set(reported);
  const truth = new Set(expected);
  const hit = expected.filter((x) => report.has(x)).length;
  const extra = reported.filter((x) => !truth.has(x)).length;
  return { recall: expected.length === 0 ? 1 : hit / expected.length,
    precision: reported.length === 0 ? 1 : (reported.length - extra) / reported.length };
}

module.exports = { MODULUS, EXPONENT_BITS, PRIME_POWERS, GENERATOR, GENERATOR_INVERSE,
  GENERATOR_ORDER, field, powMod, inverse, findGenerator, SplitMix64, keyedHash, prf, hashBin,
  hashSlot, coordinate, padValue, InMemoryNetwork, otSenderSetup, otReceiverQuery, otSenderReply,
  otReceiverOpen, otCheatingOpen, cuckooInsert, buildServerTable, binMessages,
  privateSetIntersection, fingerprint, bruteIntersection, quality };

if (require.main === module) {
  const assert = require("node:assert");
  const PARAMS = { bins: 4096, slots: 16 }; // 65536 coordinates: birthday-safe for <= 20 items

  // --- toy group: certified generator and order ------------------------------
  const factorProduct = PRIME_POWERS.reduce((acc, [q, e]) => acc * q ** BigInt(e), 1n);
  assert.strictEqual(factorProduct, MODULUS - 1n, "the known prime factors multiply to p - 1");
  assert.strictEqual(GENERATOR, 37n, "the first certified generator is 37");
  assert.strictEqual(powMod(GENERATOR, GENERATOR_ORDER), 1n, "generator order divides p - 1");
  for (const [q] of PRIME_POWERS) {
    assert.notStrictEqual(powMod(GENERATOR, GENERATOR_ORDER / q), 1n, `order not divisible by ${q}`);
  }
  assert.strictEqual(field(field(7n) * inverse(7n)), 1n, "field inverses are exact");
  assert.strictEqual(field(GENERATOR * GENERATOR_INVERSE), 1n, "g * g^-1 = 1");

  // --- base OT correctness ---------------------------------------------------
  const runBaseOt = (n, chosen, seedVal) => {
    const rng = new SplitMix64(seedVal);
    const setup = otSenderSetup(n, rng);
    const messages = Array.from({ length: n }, (_, j) => field(BigInt(1000 + j) * 7919n + 13n));
    const query = otReceiverQuery(setup, chosen, rng);
    const reply = otSenderReply(messages, query.publicKeys, rng);
    return { messages, query, reply, opened: otReceiverOpen(reply, query) };
  };
  let otCorrect = 0;
  for (let trial = 0; trial < 80; trial++) {
    const n = 2 + (trial % 7); // 2..8
    const chosen = trial % n;
    const { messages, opened } = runBaseOt(n, chosen, 600 + trial);
    assert.strictEqual(opened, messages[chosen], `base OT trial ${trial}, choice ${chosen}`);
    otCorrect++;
  }
  assert.strictEqual(otCorrect, 80);
  for (let chosen = 0; chosen < 16; chosen++) {
    const { messages, opened } = runBaseOt(16, chosen, 7000 + chosen);
    assert.strictEqual(opened, messages[chosen], `1-of-16 OT choice ${chosen}`);
  }
  assert.throws(() => runBaseOt(16, 16, 1), /choice/, "out-of-range choice is rejected");
  assert.throws(() => otSenderSetup(1, new SplitMix64(1)), /two messages/, "1-of-1 is not an OT");

  // --- sender privacy: a cheating receiver cannot open a second index --------
  let cheatSuccesses = 0;
  const cheatResidues = new Set();
  for (let trial = 0; trial < 100; trial++) {
    const n = 5;
    const { messages, query, reply } = runBaseOt(n, trial % n, 9100 + trial);
    const stolenIndex = (query.chosen + 1) % n;
    const stolen = otCheatingOpen(reply, query, stolenIndex);
    cheatResidues.add(Number(stolen % 97n));
    if (stolen === messages[stolenIndex]) cheatSuccesses++;
  }
  assert.strictEqual(cheatSuccesses, 0, "a cheating receiver never opens a second message");
  assert.ok(cheatResidues.size >= 60, `cheated openings spread out: ${cheatResidues.size}/97`);
  assert.throws(() => otCheatingOpen([], { blinders: new Map() }, 0), /blinder/, "bad cheat index");

  // --- receiver-choice privacy, measured ------------------------------------
  // With C fixed, (k, r_{j != i}) -> PK is a bijection onto group^N, so the sender's view of PK_0 is
  // distributed identically whatever index was chosen. Two-sample chi-square, 1600 samples per side.
  const residueCounts = (chosen, samples) => {
    const counts = new Array(97).fill(0);
    for (let trial = 0; trial < samples; trial++) {
      const rng = new SplitMix64(13000 + trial);
      const setup = otSenderSetup(4, rng);
      const query = otReceiverQuery(setup, chosen, rng);
      counts[Number(query.publicKeys[0] % 97n)]++;
    }
    return counts;
  };
  const samples = 1600;
  const choiceZero = residueCounts(0, samples);
  const choiceLast = residueCounts(3, samples);
  let chiSquare = 0;
  let cells = 0;
  for (let r = 0; r < 97; r++) {
    const total = choiceZero[r] + choiceLast[r];
    if (total === 0) continue;
    chiSquare += (choiceZero[r] - choiceLast[r]) ** 2 / total;
    cells++;
  }
  const chiDf = cells - 1;
  const chiLimit = chiDf + 4.5 * Math.sqrt(2 * chiDf);
  const occupied = (counts) => counts.filter((c) => c > 0).length;
  assert.ok(chiSquare < chiLimit,
    `sender cannot tell the choice (chi2=${chiSquare.toFixed(1)} < ${chiLimit.toFixed(1)}, df=${chiDf})`);
  assert.ok(occupied(choiceZero) >= 90 && occupied(choiceLast) >= 90,
    `PK_0 covers residues: ${occupied(choiceZero)}/${occupied(choiceLast)} of 97`);
  assert.ok(Math.max(...choiceZero) < samples / 5, "no residue dominates the sender's view");

  // --- cuckoo layer ----------------------------------------------------------
  const cuckooRng = new SplitMix64(31337);
  const clientItems = Array.from({ length: 48 }, (_, i) => `user-${i}`);
  const cuckoo = cuckooInsert(clientItems, 0xabcde, 4096, cuckooRng);
  assert.strictEqual(cuckoo.aborted, false, "cuckoo insertion succeeds");
  assert.strictEqual(cuckoo.table.size, clientItems.length, "one item per occupied bin");
  for (const item of clientItems) {
    const placed = cuckoo.lookup.get(item);
    assert.ok(placed !== undefined, `${item} is retrievable`);
    assert.strictEqual(cuckoo.table.get(placed.bin).item, item, "lookup points at the placed item");
  }
  assert.ok(cuckoo.maxEvictions <= 8, `bounded eviction chain: ${cuckoo.maxEvictions}`);
  const reloaded = cuckooInsert(clientItems, 0xabcde, 4096, new SplitMix64(31337));
  assert.deepStrictEqual([...reloaded.lookup].map(([k, v]) => `${k}@${v.bin}`),
    [...cuckoo.lookup].map(([k, v]) => `${k}@${v.bin}`), "cuckoo layout is deterministic");

  // --- PSI against the brute-force oracle ------------------------------------
  const setRng = new SplitMix64(2024);
  let psiTrials = 0;
  let totalCollisions = 0;
  let totalOts = 0;
  let degradedTrials = 0;
  for (let trial = 0; trial < 24; trial++) {
    const size = 8 + (trial % 7) * 2; // 8..20
    const client = Array.from({ length: size }, (_, i) => `c${Number(setRng.next() % 60n)}-${i}`);
    const server = Array.from({ length: size }, (_, i) => `s${Number(setRng.next() % 60n)}-${i}`);
    for (let k = 0; k < 5; k++) server.push(client[(trial + k) % client.length]); // controlled overlap
    const result = privateSetIntersection(client, server,
      { ...PARAMS, seed: 100 + trial, key: 0x51ab + trial * 7919 });
    const expected = bruteIntersection(client, server);
    const q = quality(result.intersection, expected);
    if (result.degraded) degradedTrials++;
    assert.strictEqual(result.degraded, false, `trial ${trial} is not degraded`);
    assert.strictEqual(result.collisions, 0, `trial ${trial} has no coordinate collision`);
    assert.strictEqual(q.recall, 1, `trial ${trial} recall is 1`);
    assert.strictEqual(q.precision, 1, `trial ${trial} precision is 1`);
    assert.strictEqual(result.otCount, new Set(client.map(String)).size,
      `trial ${trial} uses one OT per distinct client item`);
    assert.strictEqual(result.detail.network.sent, 2 * result.otCount,
      `trial ${trial} moves exactly two network messages per OT`);
    assert.strictEqual(result.messagesPerOt, PARAMS.slots, "bin tables have `slots` messages");
    assert.ok(expected.length >= 5, "the controlled overlap is really exercised");
    psiTrials++;
    totalCollisions += result.collisions;
    totalOts += result.otCount;
  }
  assert.strictEqual(psiTrials, 24);
  assert.strictEqual(totalCollisions, 0, "no collisions across all trials");
  assert.strictEqual(degradedTrials, 0, "no trial is degraded");

  // --- boundary inputs -------------------------------------------------------
  const run = (client, server, seedVal, key) =>
    privateSetIntersection(client, server, { ...PARAMS, seed: seedVal, key });
  assert.deepStrictEqual(run([], ["a", "b"], 5, 1).intersection, [], "empty client set");
  assert.strictEqual(run([], ["a", "b"], 5, 1).otCount, 0, "empty client set performs no OT");
  assert.deepStrictEqual(run(["a", "b"], [], 6, 2).intersection, [], "empty server set");
  assert.deepStrictEqual(run(["x", "y"], ["a", "b"], 7, 3).intersection, [], "disjoint sets");
  assert.deepStrictEqual(run(["k1", "k2", "k3"], ["k3", "k2", "k1"], 8, 4).intersection,
    ["k1", "k2", "k3"], "identical sets intersect fully");
  const duplicated = run(["dup", "dup", "other"], ["dup", "dup"], 9, 5);
  assert.deepStrictEqual(duplicated.intersection, ["dup"], "duplicates collapse to one result");
  assert.strictEqual(duplicated.otCount, 2, "duplicates cost one OT, not three");
  assert.deepStrictEqual(run(["solo"], ["solo"], 10, 6).intersection, ["solo"], "single element");
  assert.deepStrictEqual(run(["solo"], ["other"], 11, 7).intersection, [], "near-miss single element");

  // --- no silent failure: collisions are counted, reported and bounded --------
  // Pigeonhole a colliding pair of items under a one-bin table, then show the loss and the flag.
  const collidingPair = (key, slots) => {
    const seen = new Map();
    for (let i = 0; i < 400; i++) {
      const item = `x${i}`;
      const coord = coordinate(hashBin(key, item, 0, 1), hashSlot(key, item, slots));
      if (seen.has(coord)) return [seen.get(coord), item];
      seen.set(coord, item);
    }
    throw new Error("no colliding pair found");
  };
  const [firstOfPair, secondOfPair] = collidingPair(0x51ab, 16);
  const forced = buildServerTable([firstOfPair, secondOfPair], 0x51ab, 1, 16);
  assert.strictEqual(forced.placed, 4, "two items occupy two coordinates each");
  assert.strictEqual(forced.collisions, 2, "the shared coordinate is detected twice");
  // The lossy run: the second item overwrites the coordinate the client asks for.
  const lossy = privateSetIntersection([firstOfPair], [secondOfPair, firstOfPair],
    { bins: 1, slots: 16, seed: 12, key: 0x51ab });
  assert.strictEqual(lossy.degraded, true, "the lossy run is flagged degraded");
  assert.ok(lossy.collisions >= 1, "and the collision count is exposed");
  assert.deepStrictEqual(lossy.intersection, [],
    "the overwritten value really is lost — which is why the flag matters");
  assert.deepStrictEqual(bruteIntersection([firstOfPair], [secondOfPair, firstOfPair]),
    [firstOfPair], "the oracle disagrees: the failure is never silent");
  // Parameter sweep: more slots per bin must buy fewer collisions (birthday-bound check).
  const sweepItems = Array.from({ length: 20 }, (_, i) => `sweep-${i}`);
  const sweepRuns = 8;
  const losses = new Map();
  for (const slots of [2, 4, 8]) {
    let collisions = 0;
    for (let trial = 0; trial < sweepRuns; trial++) {
      collisions += privateSetIntersection(sweepItems, sweepItems,
        { bins: 256, slots, seed: 200 + trial, key: 0x900 + trial }).collisions;
    }
    losses.set(slots, collisions);
  }
  // Birthday mean per sweep plus a 4-sigma allowance (clashes are approximately Poisson).
  const birthdayBound = (slots) => sweepRuns * (40 * 39 / 2) / (256 * slots);
  const boundLimit = (slots) => birthdayBound(slots) + 4 * Math.sqrt(birthdayBound(slots));
  assert.ok(losses.get(2) > losses.get(8),
    `more slots buy fewer collisions (slots=2: ${losses.get(2)} vs slots=8: ${losses.get(8)})`);
  assert.ok(losses.get(8) <= boundLimit(8),
    `measured collisions ${losses.get(8)} within the birthday bound ` +
    `~${birthdayBound(8).toFixed(1)} +- ${(4 * Math.sqrt(birthdayBound(8))).toFixed(1)}`);
  const abortedRun = privateSetIntersection(["a", "b"], ["a", "b"],
    { bins: 1, slots: 16, seed: 13, key: 0x51ab });
  assert.strictEqual(abortedRun.aborted, true, "one bin cannot host two items: insertion aborts");
  assert.strictEqual(abortedRun.degraded, true, "an aborted cuckoo run is flagged degraded");

  // --- key-dependence of the server's view, and determinism ------------------
  const coordinateSets = new Set();
  const slotCoverage = new Set();
  const keyedItems = ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"];
  for (let trial = 0; trial < 30; trial++) {
    const result = privateSetIntersection(keyedItems, keyedItems,
      { ...PARAMS, seed: 20 + trial, key: 0x1000 + trial * 131 });
    coordinateSets.add(result.requested.join("|"));
    for (const coord of result.requested) slotCoverage.add(coord.split(":")[1]);
    assert.deepStrictEqual(result.intersection, keyedItems, "self-intersection finds everything");
  }
  assert.strictEqual(coordinateSets.size, 30,
    "the server's view depends on the shared PRF key, never on a fixed coordinate set");
  assert.strictEqual(slotCoverage.size, PARAMS.slots, `all ${PARAMS.slots} slots appear across keys`);

  const target = clientItems.slice(0, 10);
  const partner = clientItems.slice(5, 20);
  const first = privateSetIntersection(target, partner, { ...PARAMS, seed: 55, key: 0x33 });
  const replay = privateSetIntersection(target, partner, { ...PARAMS, seed: 55, key: 0x33 });
  assert.deepStrictEqual(first.intersection, replay.intersection, "replay yields the same result");
  assert.strictEqual(first.fingerprint, replay.fingerprint, "replay yields the same transcript");
  assert.deepStrictEqual(first.intersection, bruteIntersection(target, partner),
    "overlapping slices intersect exactly like the oracle");

  console.log(`group=Z_2^61-1 generator=${GENERATOR} base-OT-trials=80 cheating-openings-failed=100`);
  console.log(`privacy chi2=${chiSquare.toFixed(1)} < ${chiLimit.toFixed(1)} (df=${chiDf}) psi-trials=${psiTrials}`);
  console.log(`cost: ots=${totalOts} slots-per-bin=${PARAMS.slots} bins=${PARAMS.bins} collisions=${totalCollisions} degraded=${degradedTrials}`);
  console.log("PASS");
}
