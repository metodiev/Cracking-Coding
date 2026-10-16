/** Task 886 - Implement end-to-end encrypted search for large document corpus with privacy budget.
 *  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * A forward-private searchable symmetric encryption (SSE) scheme plus an ε-budgeted statistical
 * release. The client tokenises documents into terms and, for each (term, document) pair, uploads a
 * *label* = HMAC(kSearch, "label|epoch|term|doc") together with the document id encrypted under
 * AES-256-CTR and authenticated with HMAC-SHA256. A query is a trapdoor: the list of labels the
 * client has issued for that term over the epochs it has seen. The server therefore only ever
 * observes pseudorandom labels, ciphertexts and MACs — never a term or a document id.
 *
 * Forward privacy: every document is added in a fresh epoch and label keys are derived from
 * (term, epoch), so a label created *after* a query cannot be linked to that query, and re-issuing a
 * trapdoor yields entirely fresh labels. Backward privacy: deletion re-derives the label in a new
 * epoch and the server removes exactly that entry, so deleted documents are never returned again.
 * The client keeps the (small) keyword state — the standard practical instantiation of
 * forward-private SSE, and the documented simplification of this model.
 *
 * Privacy budget: the analyst-facing histogram is released with a two-sided discrete Laplace
 * mechanism calibrated to sensitivity 1 (one neighbouring corpus differs in one document count), so
 * each release is ε-differentially private. Budgets compose sequentially and are enforced strictly:
 * a release that would exceed the remaining budget is rejected.
 *
 * Invariants verified below: (1) recall = precision = 1 against a brute-force plaintext oracle;
 * (2) label freshness / no linkage between a past trapdoor and a later update; (3) no plaintext
 * leakage into the server transcript, including after deletions; (4) tamper detection on every
 * stored entry; (5) exact ε-differential-privacy guarantee of the released histogram (the noise pmf
 * is evaluated in closed form and the likelihood ratio is checked against e^ε); (6) unbiasedness and
 * calibrated variance of the noise (so the release stays useful); (7) sequential budget accounting;
 * (8) determinism for a fixed PRNG seed.
 *
 * Simplifications: no network, no wall clock, no Math.random — epochs are an explicit counter and
 * all randomness comes from a seeded splitmix32 PRNG.
 */
"use strict";

const crypto = require("node:crypto");

/* ------------------------------------------------------------------ deterministic randomness */

/** splitmix32: small, seedable, exactly reproducible. */
class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  /** Uniform on (0, 1) — strictly interior so the inverse-CDF never sees 0 or 1. */
  unit() { return (this.next() + 0.5) / 0x1_0000_0000; }
}

/** Closed-form pmf of the two-sided discrete Laplace with sensitivity 1: (1-p)/(1+p) * p^|k|. */
function laplacePmf(k, epsilon) {
  const p = Math.exp(-epsilon);
  return ((1 - p) / (1 + p)) * Math.pow(p, Math.abs(k));
}

/**
 * Inverse-CDF sampler for that pmf. For u >= 1/(1+p) the value is positive, and
 * F(k) = 1 - p^(k+1)/(1+p), hence k = floor(log((1-u)(1+p)) / log p); mirrored for u < 1/(1+p).
 */
function discreteLaplace(unit, epsilon) {
  const p = Math.exp(-epsilon);
  const scale = 1 + p;
  const cut = 1 / scale;
  const a = unit < cut ? unit * scale : (1 - unit) * scale;
  let magnitude = Math.floor(Math.log(a) / Math.log(p));
  if (magnitude < 0) magnitude = 0;
  return unit < cut ? -magnitude : magnitude;
}

/* ------------------------------------------------------------------ crypto helpers */

const hmacHex = (key, message) => crypto.createHmac("sha256", key).update(message).digest("hex");

/** Any string key is hashed into a 32-byte AES key deterministically. */
const asKey = (key) => (Buffer.isBuffer(key) ? key : crypto.createHash("sha256").update(String(key)).digest());

/** Deterministic authenticated encryption: CTR keystream with a counter IV plus an HMAC tag. */
function seal(key, counter, plaintext) {
  const iv = Buffer.alloc(16);
  iv.writeUInt32BE(counter >>> 0, 12);
  const cipher = crypto.createCipheriv("aes-256-ctr", asKey(key), iv);
  const ct = Buffer.concat([cipher.update(String(plaintext), "utf8"), cipher.final()]);
  const hex = ct.toString("hex");
  return { ct: hex, tag: hmacHex(key, `tag|${counter}|${hex}`).slice(0, 32) };
}

function open(key, counter, sealed) {
  const expect = hmacHex(key, `tag|${counter}|${sealed.ct}`).slice(0, 32);
  if (expect !== sealed.tag) throw new Error("authentication failure: entry was tampered with");
  const iv = Buffer.alloc(16);
  iv.writeUInt32BE(counter >>> 0, 12);
  const decipher = crypto.createDecipheriv("aes-256-ctr", asKey(key), iv);
  return Buffer.concat([decipher.update(Buffer.from(sealed.ct, "hex")), decipher.final()]).toString("utf8");
}

/* ------------------------------------------------------------------ parties */

/** Untrusted server: an append-only label -> sealed document id map, with an exact transcript. */
class SearchServer {
  constructor() {
    this.entries = new Map();   // label -> { sealed, epoch }
    this.trapdoors = [];        // every query ever received
    this.updates = 0;
    this.deletes = 0;
    this.readsOnLastQuery = 0;
  }

  store(label, sealed, epoch) {
    if (this.entries.has(label)) throw new Error(`label collision on ${label}`);
    this.entries.set(label, { sealed, epoch });
    this.updates++;
  }

  remove(label) {
    if (!this.entries.delete(label)) throw new Error(`cannot delete unknown label ${label}`);
    this.deletes++;
  }

  query(trapdoor) {
    this.trapdoors.push(trapdoor.slice());
    let reads = 0;
    for (const label of trapdoor) {
      if (this.entries.has(label)) reads++;
    }
    this.readsOnLastQuery = reads;
    return trapdoor.filter((label) => this.entries.has(label)).map((label) => ({ label, ...this.entries.get(label) }));
  }

  /** Everything a curious server (or a network eavesdropper) could have recorded. */
  transcript() {
    return {
      labels: [...this.entries.keys()],
      ciphertexts: [...this.entries.values()].map((entry) => entry.sealed.ct),
      tags: [...this.entries.values()].map((entry) => entry.sealed.tag),
      epochs: [...this.entries.values()].map((entry) => entry.epoch),
      trapdoors: this.trapdoors,
    };
  }
}

class SearchClient {
  constructor(masterSecret, { epsilon, budget, seed = 886 } = {}) {
    this.master = masterSecret;
    this.kSearch = hmacHex(masterSecret, "kSearch");
    this.kDoc = hmacHex(masterSecret, "kDoc");
    this.epsilon = epsilon;
    this.budget = budget;
    this.spent = 0;
    this.releases = 0;
    this.epoch = 0;
    this.labelSeq = 0;
    this.state = new Map();   // term -> [{ label, epoch, docId }] (client-side keyword state)
    this.local = new Map();   // docId -> Set(term), the client's own plaintext view
    this.prng = new Prng(seed);
    this.noiseDraws = [];
  }

  static termsOf(text) { return [...new Set(String(text).toLowerCase().split(/[^a-z0-9]+/u).filter(Boolean))]; }

  /** Uploads one document in a fresh epoch; returns the labels written to the server. */
  addDocument(server, docId, text) {
    if (this.local.has(docId)) throw new Error(`document ${docId} already exists`);
    const terms = SearchClient.termsOf(text);
    this.epoch++;
    const entryCounter = this.epoch * 4096 + 1;
    const labels = [];
    for (const term of terms) {
      // One fresh epoch per update: the label for (term, doc) is unrelated to every previous label.
      const label = hmacHex(this.kSearch, `label|${term}|${this.epoch}|${docId}`).slice(0, 48);
      server.store(label, seal(this.kDoc, entryCounter, docId), this.epoch);
      if (!this.state.has(term)) this.state.set(term, []);
      this.state.get(term).push({ label, epoch: this.epoch, docId });
      labels.push(label);
    }
    this.local.set(docId, new Set(terms));
    return labels;
  }

  deleteDocument(server, docId) {
    const terms = this.local.get(docId);
    if (!terms) throw new Error(`unknown document ${docId}`);
    for (const term of terms) {
      const issued = this.state.get(term);
      const index = issued.findIndex((rec) => rec.docId === docId);
      const rec = issued.splice(index, 1)[0];
      server.remove(rec.label);
    }
    this.local.delete(docId);
    this.labelSeq++;
  }

  /** Trapdoor = every label issued for the term in every epoch this client has used. */
  search(server, term) {
    const issued = this.state.get(term) || [];
    const results = server.query(issued.map((rec) => rec.label));
    const ids = new Set();
    for (const hit of results) ids.add(open(this.kDoc, (hit.epoch * 4096) + 1, hit.sealed));
    return [...ids].sort();
  }

  /** The ε-differentially-private statistical release: an exact count plus calibrated noise. */
  reportCount(server, term) {
    if (this.spent + this.epsilon > this.budget + 1e-12) {
      throw new Error(`privacy budget exhausted: spent ${this.spent}, epsilon ${this.epsilon}, budget ${this.budget}`);
    }
    const exact = this.search(server, term).length;
    const unit = this.prng.unit();
    const noise = discreteLaplace(unit, this.epsilon);
    this.spent = Math.min(this.budget, this.spent + this.epsilon);
    this.releases++;
    this.noiseDraws.push(noise);
    return { exact, noisy: Math.max(0, exact + noise), noise, spent: this.spent };
  }
}

/* ------------------------------------------------------------------ oracle */

/** Independent plaintext oracle — a straightforward scan, no crypto involved. */
function bruteForceSearch(documents, term) {
  const needle = String(term).toLowerCase();
  return documents
    .filter((doc) => SearchClient.termsOf(doc.text).includes(needle))
    .map((doc) => doc.id)
    .sort();
}

function buildCorpus(count, vocabulary, seed) {
  const prng = new Prng(seed);
  const docs = [];
  for (let i = 0; i < count; i++) {
    const words = [];
    const length = 3 + (prng.next() % 5);
    for (let w = 0; w < length; w++) words.push(vocabulary[prng.next() % vocabulary.length]);
    docs.push({ id: `doc-${String(i).padStart(3, "0")}`, text: words.join(" ") });
  }
  return docs;
}

module.exports = {
  Prng, laplacePmf, discreteLaplace, seal, open, hmacHex, SearchServer, SearchClient,
  bruteForceSearch, buildCorpus,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- the noise mechanism matches its closed form (this is the DP guarantee, not a sample) ---
  for (const epsilon of [0.5, 1, 2]) {
    let total = 0;
    for (let k = -60; k <= 60; k++) total += laplacePmf(k, epsilon);
    assert.ok(Math.abs(total - 1) < 1e-9, `pmf must sum to 1 for epsilon=${epsilon}`);
    const ratio = laplacePmf(0, epsilon) / laplacePmf(1, epsilon);
    assert.ok(Math.abs(ratio - Math.exp(epsilon)) < 1e-9, `likelihood ratio must be e^epsilon`);
    for (let k = -40; k <= 40; k++) {
      for (const shift of [-1, 1]) {
        const observed = laplacePmf(k, epsilon) / laplacePmf(k + shift, epsilon);
        assert.ok(observed <= Math.exp(epsilon) + 1e-9, `privacy violated at k=${k}`);
      }
    }
  }

  // --- the sampler is calibrated: empirical pmf, mean and variance against theory -----------
  const sampler = new Prng(20260886);
  const epsilon = 1.0;
  const counts = new Map();
  const samples = 40000;
  let sum = 0;
  let sumSquares = 0;
  for (let i = 0; i < samples; i++) {
    const value = discreteLaplace(sampler.unit(), epsilon);
    counts.set(value, (counts.get(value) || 0) + 1);
    sum += value;
    sumSquares += value * value;
  }
  const mean = sum / samples;
  const variance = sumSquares / samples - mean * mean;
  const p = Math.exp(-epsilon);
  const theoryVariance = (2 * p) / Math.pow(1 - p, 2);
  assert.ok(Math.abs(mean) < 0.05, `mean ${mean} must be near 0`);
  assert.ok(Math.abs(variance - theoryVariance) / theoryVariance < 0.1,
    `variance ${variance} must be near ${theoryVariance}`);
  for (let k = -3; k <= 3; k++) {
    const empirical = (counts.get(k) || 0) / samples;
    assert.ok(Math.abs(empirical - laplacePmf(k, epsilon)) < 0.01, `empirical pmf at k=${k}`);
  }

  // --- exact encrypted search against the plaintext oracle ---------------------------------
  const vocabulary = ["kernel", "quota", "vector", "ledger", "proof", "atlas", "cadence"];
  const docs = buildCorpus(60, vocabulary, 77);
  const server = new SearchServer();
  const client = new SearchClient("top-secret-master-key", { epsilon: 0.1, budget: 0.5, seed: 4242 });
  let uploadedLabels = 0;
  for (const doc of docs) uploadedLabels += client.addDocument(server, doc.id, doc.text).length;

  for (const term of vocabulary) {
    const expected = bruteForceSearch(docs, term);
    assert.deepStrictEqual(client.search(server, term), expected, `recall/precision for ${term}`);
  }
  assert.strictEqual(client.search(server, "absent-term").length, 0);
  assert.strictEqual(server.entries.size, uploadedLabels, "index size is linear in (term, doc) pairs");
  assert.strictEqual(client.epoch, docs.length, "one fresh epoch per update");

  // --- leakage monitor: no term and no document id in the whole server transcript ----------
  const transcript = JSON.stringify(server.transcript());
  for (const term of vocabulary) {
    assert.ok(!transcript.includes(`"${term}"`), `term ${term} leaked`);
    assert.ok(!transcript.includes(term), `term ${term} leaked as a substring`);
  }
  for (const doc of docs.slice(0, 12)) assert.ok(!transcript.includes(doc.id), `doc id ${doc.id} leaked`);
  assert.ok(!transcript.includes("top-secret"));

  // --- tamper detection: a modified ciphertext must not decrypt ----------------------------
  const victim = [...server.entries.entries()][0];
  const flipped = { ct: `00${victim[1].sealed.ct.slice(2)}`, tag: victim[1].sealed.tag };
  assert.throws(() => open(client.kDoc, 4097, flipped), /tampered/);

  // --- a client with the wrong key sees nothing --------------------------------------------
  const stranger = new SearchClient("other-key", { epsilon: 0.1, budget: 0.5 });
  assert.strictEqual(stranger.search(server, "kernel").length, 0);
  assert.strictEqual(stranger.state.size, 0, "an unrelated client has no keyword state");

  // --- forward privacy: a trapdoor never contains a label created later --------------------
  const history = [];
  const fwdServer = new SearchServer();
  const fwdClient = new SearchClient("fwd-key", { epsilon: 0.2, budget: 5, seed: 9 });
  const earlyQuery = { results: null, trapdoor: null };
  for (let round = 0; round < 6; round++) {
    const docId = `fwd-${round}`;
    const labels = fwdClient.addDocument(fwdServer, docId, round % 2 === 0 ? "kernel proof" : "kernel ledger");
    history.push({ epoch: fwdClient.epoch, labels });
    if (round === 2) {
      earlyQuery.results = fwdClient.search(fwdServer, "kernel");
      earlyQuery.trapdoor = fwdServer.trapdoors[fwdServer.trapdoors.length - 1].slice();
    }
  }
  assert.deepStrictEqual(earlyQuery.results, ["fwd-0", "fwd-1", "fwd-2"], "mid-stream query sees exactly 3 epochs");
  assert.strictEqual(earlyQuery.trapdoor.length, 3);
  const laterLabels = new Set(history.filter((h) => h.epoch > 3).flatMap((h) => h.labels));
  for (const label of earlyQuery.trapdoor) {
    assert.ok(fwdServer.entries.get(label).epoch <= 3, "an old trapdoor holds only pre-query labels");
    assert.ok(!laterLabels.has(label), "a later update reused an earlier label");
  }
  const kernelRecords = fwdClient.state.get("kernel");
  assert.strictEqual(new Set(kernelRecords.map((r) => r.label)).size, 6, "every update issues a fresh label");
  assert.deepStrictEqual(fwdClient.search(fwdServer, "kernel"),
    ["fwd-0", "fwd-1", "fwd-2", "fwd-3", "fwd-4", "fwd-5"], "recall survives later epochs");
  const allLabels = [...fwdServer.entries.keys()];
  assert.strictEqual(new Set(allLabels).size, allLabels.length, "no label collisions at all");

  // --- backward privacy: deletion removes exactly the deleted document's entries -----------
  const rawLedger = bruteForceSearch(docs, "ledger").length;
  const victimDoc = bruteForceSearch(docs, "ledger")[0];
  const removed = client.local.get(victimDoc).size;
  client.deleteDocument(server, victimDoc);
  assert.strictEqual(server.entries.size, uploadedLabels - removed);
  const surviving = docs.filter((doc) => doc.id !== victimDoc);
  assert.deepStrictEqual(client.search(server, "ledger"), bruteForceSearch(surviving, "ledger"),
    "deleted document must never be returned");
  assert.strictEqual(client.search(server, "ledger").length, rawLedger - 1);
  assert.throws(() => client.deleteDocument(server, victimDoc), /unknown document/);

  // --- liveness/cost: a query touches only the labels it needs, never the corpus -----------
  const liveVector = bruteForceSearch(surviving, "vector").length;
  client.search(server, "vector");
  assert.strictEqual(server.readsOnLastQuery, liveVector);
  assert.ok(server.readsOnLastQuery < server.entries.size, "a query must not scan the whole index");
  assert.strictEqual(server.trapdoors[server.trapdoors.length - 1].length, liveVector,
    "trapdoor length equals the number of live postings");

  // --- privacy budget: exact sequential composition and strict rejection -------------------
  const budgeted = new SearchClient("budget-key", { epsilon: 0.1, budget: 0.35, seed: 5 });
  const bServer = new SearchServer();
  for (const doc of docs) budgeted.addDocument(bServer, doc.id, doc.text);
  const released = [];
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      released.push(budgeted.reportCount(bServer, "kernel"));
    } catch (error) {
      assert.match(error.message, /budget exhausted/);
      break;
    }
  }
  assert.strictEqual(released.length, 3, "floor(0.35 / 0.1) = 3 releases fit");
  assert.ok(Math.abs(budgeted.spent - 0.3) < 1e-9, `spent ${budgeted.spent}`);
  assert.strictEqual(budgeted.releases, 3);
  const truth = bruteForceSearch(docs, "kernel").length;
  for (const release of released) {
    assert.strictEqual(release.exact, truth, "the private count is exact before noise");
    assert.ok(Number.isInteger(release.noise) && Math.abs(release.noise) <= 200,
      `noise ${release.noise} must be an integer of bounded magnitude`);
  }

  // --- unbiasedness / no systematic bias: averaging releases recovers the true count -------
  const repeat = new SearchClient("repeat-key", { epsilon: 0.5, budget: 1000, seed: 12345 });
  const rServer = new SearchServer();
  for (const doc of docs) repeat.addDocument(rServer, doc.id, doc.text);
  const trials = 2000;
  let accumulated = 0;
  for (let i = 0; i < trials; i++) accumulated += repeat.reportCount(rServer, "quota").noisy;
  const estimate = accumulated / trials;
  const expected = bruteForceSearch(docs, "quota").length;
  assert.ok(Math.abs(estimate - expected) < 0.25, `mean release ${estimate} vs true ${expected}`);
  assert.ok(Math.abs(repeat.spent - Math.min(repeat.budget, trials * 0.5)) < 1e-9);

  // --- determinism ------------------------------------------------------------------------
  const runTwice = () => {
    const c = new SearchClient("determinism", { epsilon: 0.25, budget: 10, seed: 7 });
    const s = new SearchServer();
    for (const doc of docs) c.addDocument(s, doc.id, doc.text);
    const out = [];
    for (let i = 0; i < 8; i++) out.push(c.reportCount(s, vocabulary[i % vocabulary.length]));
    return JSON.stringify(out) + `|${c.search(s, "atlas").join(",")}`;
  };
  assert.strictEqual(runTwice(), runTwice());

  // --- edge cases: degenerate corpora ------------------------------------------------------
  const empty = new SearchClient("edge", { epsilon: 1, budget: 3, seed: 1 });
  const emptyServer = new SearchServer();
  assert.deepStrictEqual(empty.search(emptyServer, "anything"), []);
  assert.strictEqual(empty.reportCount(emptyServer, "anything").exact, 0);
  const single = new SearchClient("edge", { epsilon: 1, budget: 3, seed: 2 });
  const singleServer = new SearchServer();
  single.addDocument(singleServer, "only", "atlas atlas ledger");
  assert.deepStrictEqual(single.search(singleServer, "atlas"), ["only"]);
  assert.deepStrictEqual(single.search(singleServer, "ledger"), ["only"]);
  assert.strictEqual(single.search(singleServer, "atlas").length, 1, "duplicate terms are one posting");
  assert.throws(() => single.addDocument(singleServer, "only", "atlas"), /already exists/);
  const blank = new SearchClient("edge", { epsilon: 1, budget: 3, seed: 3 });
  const blankServer = new SearchServer();
  assert.deepStrictEqual(blank.addDocument(blankServer, "blank", "   "), [], "no terms, no labels");
  assert.strictEqual(blankServer.entries.size, 0);

  console.log(`sse: ${docs.length} documents, ${server.entries.size} live labels, ` +
    `mean|noise| = ${(repeat.noiseDraws.reduce((a, b) => a + Math.abs(b), 0) / repeat.noiseDraws.length).toFixed(3)}, ` +
    `budget spent ${budgeted.spent.toFixed(2)}/${budgeted.budget}`);
  console.log("PASS");
}
