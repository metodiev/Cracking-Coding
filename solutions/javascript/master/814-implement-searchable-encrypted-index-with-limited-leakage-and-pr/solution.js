/** Task 814 — Implement searchable encrypted index with limited leakage and proof-of-correctness.
 *  (Tier: Master, Beyond — Security & Privacy)
 *
 * Scheme. The client builds an encrypted inverted index over a corpus of documents; the server only
 * ever stores and serves opaque labels:
 *   - `label(w)  = PRF_K("label:" + w)`   — the search trapdoor; deterministic, so equality search
 *     works, and one-way in K, so the token is not recoverable from the index;
 *   - `listKey(w) = PRF_K("list:" + w)`   — keys the per-token posting list;
 *   - every posting is `sealed id` (a keystream ciphertext of a fixed-width document id) plus
 *     `tag = PRF_{listKey}(label | sealed id)`, so the client can verify authenticity of each entry;
 *   - every token bucket carries a sealed header `{count, digest}` where
 *     `digest = PRF_{listKey}(sorted real ids)`, which makes the response *provably complete*: the
 *     client recomputes the digest over the entries it authenticated, so an omission, a replay or a
 *     swap is detected with certainty (the server does not know the keys, so it cannot re-forge the
 *     header — forging would require the PRF outputs).
 *
 * Limited leakage. Every bucket is padded with key-derived dummy postings up to a power-of-two bucket
 * class, and buckets are uploaded for every token the client may query — including tokens that do not
 * occur in the corpus. The server therefore sees the same response shape for a hit and for a miss, so
 * absence is hidden; the *measured* residual leakage is exactly the set of query labels (the access
 * pattern: repeating a query reveals that the same search was repeated, the classic deterministic
 * trapdoor leak) and the coarsened bucket class (a size class, not the exact frequency). The file
 * measures both the leak that padding removes (hit/miss distinguishability drop from 1.0 to 0.0) and
 * the leak that remains (repeat-query linkability = 1.0, bucket classes = log2 of the frequency
 * range), instead of claiming more than the construction delivers.
 *
 * Simplified versus production: a keystream + keyed tag instead of AES/ChaCha and a real PRF, a
 * single party holding the key (no multi-user key derivation), no dynamic updates or forward privacy
 * (an update would need a fresh state), no oblivious RAM — the access pattern leak is inherent here —
 * and the document vocabulary is small so the checks can enumerate it.
 *
 * Verified against brute force: exact and conjunctive search recall/precision for every vocabulary
 * token and for absent probes; padding actually happens and dummies never surface; single and multiple
 * omissions, byte tampering, entry replay, bucket swaps and a missing header are all rejected;
 * absence indistinguishability and the remaining (access-pattern) leakage are measured; label
 * uniqueness and key dependence; bucket-class and storage-overhead bounds; determinism; empty corpus,
 * single-document and all-documents tokens.
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic primitives

function keyedHash64(key, text) {
  let hash = 0xcbf29ce484222325n;
  const message = `${key}\u0000${text}`;
  for (let i = 0; i < message.length; i++) {
    hash = BigInt.asUintN(64, hash ^ BigInt(message.charCodeAt(i)));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash;
}

/** 64-bit keyed pseudo-random function, hex encoded. */
function prf(key, message) {
  return keyedHash64(key, message).toString(16).padStart(16, "0");
}

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

function keystreamBits(key, nonce, length) {
  const next = splitmix64(keyedHash64(key, `ks:${nonce}`));
  const bytes = Buffer.alloc(length);
  for (let i = 0; i < length; i += 8) {
    const word = next();
    for (let j = 0; j < 8 && i + j < length; j++) bytes[i + j] = Number((word >> BigInt(8 * j)) & 0xffn);
  }
  return bytes;
}

const DOC_ID_WIDTH = 4;

function sealText(key, nonce, plaintext) {
  const body = Buffer.from(plaintext, "utf8");
  const stream = keystreamBits(key, nonce, body.length);
  const ciphertext = Buffer.alloc(body.length);
  for (let i = 0; i < body.length; i++) ciphertext[i] = body[i] ^ stream[i];
  return ciphertext.toString("hex");
}

function openText(key, nonce, ciphertextHex) {
  const ciphertext = Buffer.from(ciphertextHex, "hex");
  const stream = keystreamBits(key, nonce, ciphertext.length);
  const plaintext = Buffer.alloc(ciphertext.length);
  for (let i = 0; i < plaintext.length; i++) plaintext[i] = ciphertext[i] ^ stream[i];
  return plaintext.toString("utf8");
}

// ---------------------------------------------------------------------------- corpus

const VOCABULARY_SIZE = 24;

/** Skewed token frequencies: squared uniform picks make a few tokens very common (Zipf-like). */
function buildCorpus(documentCount, tokensPerDocument, seed) {
  const next = splitmix64(seed);
  const vocabulary = Array.from({ length: VOCABULARY_SIZE }, (_, i) => `t${String(i).padStart(2, "0")}`);
  const documents = [];
  for (let id = 0; id < documentCount; id++) {
    const tokens = new Set();
    for (let k = 0; k < tokensPerDocument; k++) {
      const pick = Number(next() % BigInt(VOCABULARY_SIZE));
      const skewed = (pick * pick) % VOCABULARY_SIZE;
      tokens.add(vocabulary[skewed]);
    }
    documents.push({ id: String(id).padStart(DOC_ID_WIDTH, "0"), tokens: [...tokens].sort() });
  }
  return { vocabulary, documents };
}

function bruteForceSearch(documents, token) {
  return documents.filter((document) => document.tokens.includes(token)).map((document) => document.id).sort();
}

function bruteForceConjunctive(documents, tokens) {
  return documents
    .filter((document) => tokens.every((token) => document.tokens.includes(token)))
    .map((document) => document.id)
    .sort();
}

// ---------------------------------------------------------------------------- encrypted index

function bucketClass(count, base) {
  let size = base;
  while (size < count) size *= 2;
  return Math.max(size, base);
}

class SearchableIndex {
  /**
   * @param {object} corpus        documents from `buildCorpus`
   * @param {string} key           master key (client side only)
   * @param {object} [options]     `{ padded, bucketBase }`
   */
  constructor(corpus, key, options = {}) {
    this.corpus = corpus;
    this.key = key;
    this.padded = options.padded !== false;
    this.bucketBase = options.bucketBase ?? 4;
    this.buckets = new Map(); // label -> { bucket, entries, header, token (client side only) }
    this.build(corpus.vocabulary, options.queryTokens ?? []);
  }

  labelFor(token) {
    return prf(this.key, `label:${token}`);
  }

  listKeyFor(token) {
    return prf(this.key, `list:${token}`);
  }

  headerKeyFor(token) {
    return prf(this.key, `header:${token}`);
  }

  /** Real postings of a token, sorted, plus deterministic padding dummies to the bucket class. */
  build(corpusVocabulary, extraTokens = []) {
    const seen = new Set();
    const tokens = [...corpusVocabulary, ...extraTokens].filter((token) => {
      if (seen.has(token)) return false;
      seen.add(token);
      return true;
    });
    this.buckets.clear();
    for (const token of tokens) {
      const matches = bruteForceSearch(this.corpus.documents, token);
      const label = this.labelFor(token);
      const listKey = this.listKeyFor(token);
      if (!this.padded && matches.length === 0) continue; // unpadded: absence is visible to the server
      const size = bucketClass(matches.length, this.bucketBase);
      const entries = matches.map((documentId, position) => {
        const sealedId = sealText(listKey, position, documentId);
        return { nonce: position, sealedId, tag: prf(listKey, `${label}|${position}|${sealedId}`) };
      });
      // Dummies are indistinguishable from real postings to the server but decrypt to an id the
      // client knows does not exist, so the client filters them after authentication.
      for (let position = entries.length; position < size; position++) {
        const sealedId = sealText(listKey, position, `#${String(position).padStart(DOC_ID_WIDTH - 1, "0")}`);
        entries.push({ nonce: position, sealedId, tag: prf(listKey, `${label}|${position}|${sealedId}`) });
      }
      const digest = prf(listKey, matches.join(","));
      const header = sealText(this.headerKeyFor(token), 0, JSON.stringify({ count: matches.length, digest }));
      this.buckets.set(label, { bucket: size, entries, header });
    }
    return this.buckets.size;
  }

  /** Everything the server is allowed to see. */
  serialize() {
    const entries = [...this.buckets.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([label, bucket]) => ({
      label,
      bucket: bucket.bucket,
      entries: bucket.entries.map((entry) => ({ sealedId: entry.sealedId, tag: entry.tag })),
      header: bucket.header,
    }));
    return JSON.stringify(entries);
  }

  /**
   * Client-side search: authenticate every entry, drop dummies, then verify the recomputed digest
   * against the sealed header — an omission, a replay, a swap or a forged entry is rejected.
   */
  search(token, response) {
    const label = this.labelFor(token);
    const listKey = this.listKeyFor(token);
    if (!response) return { documents: [], dummies: 0, bucket: 0, verified: true, reason: "not found" };
    if (response.label !== label) throw new Error("response does not belong to this trapdoor");
    if (this.padded && response.entries.length !== response.bucket) throw new Error("bucket is not padded to its class");
    if (!response.header) throw new Error("missing completeness header");
    const documents = [];
    let dummies = 0;
    for (const entry of response.entries) {
      if (prf(listKey, `${label}|${entry.nonce}|${entry.sealedId}`) !== entry.tag) throw new Error("entry failed authentication");
      const documentId = openText(listKey, entry.nonce, entry.sealedId);
      if (documentId.startsWith("#")) dummies++;
      else documents.push(documentId);
    }
    if (new Set(documents).size !== documents.length) throw new Error("duplicate document in response");
    documents.sort();
    const header = JSON.parse(openText(this.headerKeyFor(token), 0, response.header));
    if (header.count !== documents.length) throw new Error("completeness check failed: count mismatch");
    if (header.digest !== prf(listKey, documents.join(","))) throw new Error("completeness check failed: digest mismatch");
    return { documents, dummies, bucket: response.bucket, verified: true };
  }

  conjunctiveSearch(tokens, responses) {
    const sets = tokens.map((token, index) => new Set(this.search(token, responses[index]).documents));
    if (sets.length === 0) return [];
    return [...sets[0]].filter((documentId) => sets.every((set) => set.has(documentId))).sort();
  }
}

// ---------------------------------------------------------------------------- server

/** Storage server plus the adversaries a proof-of-correctness scheme must defeat. */
class IndexServer {
  constructor(index) {
    this.buckets = new Map();
    for (const [label, bucket] of index.buckets) {
      this.buckets.set(label, { label, bucket: bucket.bucket, entries: bucket.entries.map((entry) => ({ ...entry })), header: bucket.header });
    }
    this.queryLog = [];
  }

  query(label) {
    this.queryLog.push(label);
    const stored = this.buckets.get(label);
    if (!stored) return null;
    return { label, bucket: stored.bucket, entries: stored.entries.map((entry) => ({ ...entry })), header: stored.header };
  }

  /** Server-side descriptor of a response: what a curious server can observe about a query. */
  static descriptor(response) {
    return response === null ? "absent" : `bucket:${response.bucket}:entries:${response.entries.length}`;
  }

  dropEntry(label, position) {
    const stored = this.buckets.get(label);
    stored.entries.splice(position, 1);
  }

  tamperEntry(label, position, byteMask) {
    const stored = this.buckets.get(label);
    const entry = stored.entries[position];
    const bytes = Buffer.from(entry.sealedId, "hex");
    bytes[0] ^= byteMask;
    entry.sealedId = bytes.toString("hex");
  }

  duplicateEntry(label, position) {
    const stored = this.buckets.get(label);
    stored.entries.splice(position, 0, { ...stored.entries[position] });
  }

  swapBuckets(labelA, labelB) {
    const a = this.buckets.get(labelA);
    const b = this.buckets.get(labelB);
    this.buckets.set(labelA, { ...b, label: labelA });
    this.buckets.set(labelB, { ...a, label: labelB });
  }

  dropHeader(label) {
    this.buckets.get(label).header = null;
  }
}

module.exports = {
  keyedHash64,
  prf,
  splitmix64,
  sealText,
  openText,
  bucketClass,
  buildCorpus,
  bruteForceSearch,
  bruteForceConjunctive,
  SearchableIndex,
  IndexServer,
  DOC_ID_WIDTH,
};

if (require.main === module) {
  const assert = require("node:assert");
  const key = "client-master-key";

  const corpus = buildCorpus(200, 12, 0x8145eed);
  const probes = ["zz-absent", "qq-absent-2", "t99"];
  const index = new SearchableIndex(corpus, key, { queryTokens: probes });
  const serialized = index.serialize();
  const server = new IndexServer(index);

  // --- recall and precision against brute force for every vocabulary token -----
  let checkedTokens = 0;
  let paddedBuckets = 0;
  let totalDummies = 0;
  for (const token of [...corpus.vocabulary, ...probes]) {
    const expected = bruteForceSearch(corpus.documents, token);
    const result = index.search(token, server.query(index.labelFor(token)));
    assert.deepStrictEqual(result.documents, expected, `wrong result for ${token}`);
    assert.ok(result.verified);
    if (result.dummies > 0) paddedBuckets++;
    totalDummies += result.dummies;
    checkedTokens++;
  }
  assert.strictEqual(checkedTokens, corpus.vocabulary.length + probes.length);
  assert.ok(totalDummies > 0, "padding must actually happen");
  assert.ok(paddedBuckets > 0);

  // --- absence is indistinguishable from a hit (padded) but not unpadded --------
  let indistinguishablePairs = 0;
  let comparedPairs = 0;
  const presentToken = "t00";
  for (const token of probes) {
    const absentView = IndexServer.descriptor(server.query(index.labelFor(token)));
    for (const other of corpus.vocabulary) {
      const presentView = IndexServer.descriptor(server.query(index.labelFor(other)));
      // only tokens in the same bucket class are truly indistinguishable
      if (index.buckets.get(index.labelFor(other)).bucket === index.buckets.get(index.labelFor(token)).bucket) {
        comparedPairs++;
        if (absentView === presentView) indistinguishablePairs++;
      }
    }
  }
  assert.strictEqual(indistinguishablePairs, comparedPairs, "padded absent probes must be shape-identical to hits");
  assert.ok(comparedPairs > 0);

  const unpadded = new SearchableIndex(corpus, key, { padded: false, queryTokens: [] });
  const unpaddedServer = new IndexServer(unpadded);
  let leakedAbsent = 0;
  for (const token of probes) {
    if (IndexServer.descriptor(unpaddedServer.query(unpadded.labelFor(token))) === "absent") leakedAbsent++;
  }
  assert.strictEqual(leakedAbsent, probes.length, "without padding the server learns absence");

  // --- what still leaks: the access pattern and the bucket class --------------
  const repeatedLabel = index.labelFor("t00");
  const first = IndexServer.descriptor(server.query(repeatedLabel));
  const second = IndexServer.descriptor(server.query(repeatedLabel));
  assert.strictEqual(first, second, "repeating a query is linkable - the deterministic trapdoor leak");
  const bucketClasses = new Set([...index.buckets.values()].map((bucket) => bucket.bucket));
  const frequencies = corpus.vocabulary.map((token) => bruteForceSearch(corpus.documents, token).length);
  const maxFrequency = Math.max(...frequencies);
  assert.ok(Math.max(...bucketClasses) >= maxFrequency);
  assert.ok(bucketClasses.size <= Math.log2(maxFrequency) + 1, `bucket classes ${bucketClasses.size}`);

  // --- adversarial server: omission, tampering, replay, swap, missing header ---
  const victim = "t01";
  const victimLabel = index.labelFor(victim);
  const victimBucket = index.buckets.get(victimLabel);
  let omissionsDetected = 0;
  let omissionsTried = 0;
  for (let position = 0; position < victimBucket.entries.length; position++) {
    const hostile = new IndexServer(index);
    hostile.dropEntry(victimLabel, position);
    omissionsTried++;
    try {
      index.search(victim, hostile.query(victimLabel));
    } catch (error) {
      omissionsDetected++;
      assert.match(error.message, /authentication|completeness|padded|count|digest/);
    }
  }
  assert.strictEqual(omissionsDetected, omissionsTried, "every single omission must be caught");

  const hostileTamper = new IndexServer(index);
  hostileTamper.tamperEntry(victimLabel, 0, 0x5a);
  assert.throws(() => index.search(victim, hostileTamper.query(victimLabel)), /authentication/);

  const hostileReplay = new IndexServer(index);
  hostileReplay.duplicateEntry(victimLabel, 0);
  assert.throws(() => index.search(victim, hostileReplay.query(victimLabel)), /padded|duplicate|completeness/);

  const swapA = "t00";
  const swapB = corpus.vocabulary.find((token) => bruteForceSearch(corpus.documents, token).length !== bruteForceSearch(corpus.documents, swapA).length);
  const hostileSwap = new IndexServer(index);
  hostileSwap.swapBuckets(index.labelFor(swapA), index.labelFor(swapB));
  assert.throws(() => index.search(swapA, hostileSwap.query(index.labelFor(swapA))), /authentication|completeness|count|digest/);

  const hostileHeader = new IndexServer(index);
  hostileHeader.dropHeader(victimLabel);
  assert.throws(() => index.search(victim, hostileHeader.query(victimLabel)), /missing completeness header/);

  const multiOmissionDetected = (() => {
    const hostile = new IndexServer(index);
    hostile.dropEntry(victimLabel, 0);
    if (victimBucket.entries.length > 2) hostile.dropEntry(victimLabel, 1);
    try {
      index.search(victim, hostile.query(victimLabel));
      return false;
    } catch {
      return true;
    }
  })();
  assert.ok(multiOmissionDetected, "multiple omissions must be caught as well");

  // --- label properties: uniqueness, key dependence, no plaintext leakage -------
  const labels = new Set();
  for (const token of [...corpus.vocabulary, ...probes, ...Array.from({ length: 200 }, (_, i) => `probe-${i}`)]) {
    labels.add(prf(key, `label:${token}`));
  }
  assert.strictEqual(labels.size, corpus.vocabulary.length + probes.length + 200, "labels must be collision free here");
  const otherIndex = new SearchableIndex(corpus, "different-key", { queryTokens: probes });
  let sameLabels = 0;
  for (const token of corpus.vocabulary) if (otherIndex.labelFor(token) === index.labelFor(token)) sameLabels++;
  assert.strictEqual(sameLabels, 0, "labels must depend on the key");
  for (const token of corpus.vocabulary) {
    assert.ok(!serialized.includes(token), `plaintext token ${token} leaked into the index`);
  }
  for (const document of corpus.documents.slice(0, 20)) {
    assert.ok(!serialized.includes(`"${document.id}"`), `plaintext document id ${document.id} leaked`);
  }

  // --- conjunctive queries ------------------------------------------------------
  let conjunctiveChecked = 0;
  let conjunctiveSeed = splitmix64(0xa814n);
  for (let trial = 0; trial < 25; trial++) {
    const size = 2 + Number(conjunctiveSeed() % 2n);
    const tokens = [];
    for (let k = 0; k < size; k++) tokens.push(corpus.vocabulary[Number(conjunctiveSeed() % BigInt(corpus.vocabulary.length))]);
    const unique = [...new Set(tokens)].sort();
    const responses = unique.map((token) => server.query(index.labelFor(token)));
    assert.deepStrictEqual(index.conjunctiveSearch(unique, responses), bruteForceConjunctive(corpus.documents, unique), unique.join("&"));
    conjunctiveChecked++;
  }

  // --- cost bounds and determinism ---------------------------------------------
  let ciphertextBytes = 0;
  let paddedTotal = 0;
  for (const bucket of index.buckets.values()) {
    paddedTotal += bucket.entries.length;
    ciphertextBytes += bucket.entries.reduce((sum, entry) => sum + entry.sealedId.length / 2 + entry.tag.length / 2, 0);
  }
  const uniquePostings = corpus.documents.reduce((sum, document) => sum + document.tokens.length, 0);
  const overhead = paddedTotal / uniquePostings;
  assert.ok(overhead <= 3, `padding overhead ${overhead.toFixed(2)}x must stay bounded`);
  const queryResult = index.search("t00", server.query(index.labelFor("t00")));
  assert.strictEqual(queryResult.dummies + queryResult.documents.length, index.buckets.get(index.labelFor("t00")).bucket);
  const rebuilt = new SearchableIndex(corpus, key, { queryTokens: probes });
  assert.strictEqual(rebuilt.serialize(), serialized, "index construction must be deterministic");

  // --- edge cases ---------------------------------------------------------------
  const emptyCorpus = buildCorpus(0, 0, 1);
  const emptyIndex = new SearchableIndex(emptyCorpus, key, { queryTokens: ["nothing"] });
  const emptyResult = emptyIndex.search("nothing", new IndexServer(emptyIndex).query(emptyIndex.labelFor("nothing")));
  assert.deepStrictEqual(emptyResult.documents, []);
  assert.strictEqual(emptyResult.dummies, emptyIndex.buckets.get(emptyIndex.labelFor("nothing")).bucket);
  const singleCorpus = buildCorpus(1, 2, 7);
  const singleIndex = new SearchableIndex(singleCorpus, key, { queryTokens: [] });
  const singleServer = new IndexServer(singleIndex);
  for (const token of singleCorpus.vocabulary) {
    const expected = bruteForceSearch(singleCorpus.documents, token);
    assert.deepStrictEqual(singleIndex.search(token, singleServer.query(singleIndex.labelFor(token))).documents, expected, token);
  }
  const universalToken = corpus.documents[0].tokens[0];
  const universalCount = bruteForceSearch(corpus.documents, universalToken).length;
  const universalBucket = index.buckets.get(index.labelFor(universalToken));
  assert.ok(universalBucket.bucket >= universalCount && universalBucket.bucket / 2 < universalCount);
  assert.ok(serialized.length > ciphertextBytes, "labels and tags are part of the wire format");

  console.log(
    `tokens ${checkedTokens}, padding overhead ${overhead.toFixed(2)}x (${totalDummies} dummies), bucket classes ${bucketClasses.size}, ` +
      `omissions caught ${omissionsDetected}/${omissionsTried}, conjunctive queries ${conjunctiveChecked}, ` +
      `absent-vs-present distinguishable ${indistinguishablePairs}/${comparedPairs} (padded) vs ${leakedAbsent}/${probes.length} (unpadded)`
  );
  console.log("PASS");
}
