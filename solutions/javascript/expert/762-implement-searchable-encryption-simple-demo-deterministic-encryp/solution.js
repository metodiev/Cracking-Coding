/** Task 762 - Implement searchable encryption simple demo (deterministic encryption with leakage analysis). (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * A deterministic searchable-encryption scheme and a *quantified* analysis of what it leaks. The
 * scheme is the textbook "searchable symmetric encryption by keyword token": every keyword is
 * replaced by `token = PRF(key, word)` (here a toy 32-bit keyed checksum, not cryptography), the
 * per-document ciphertext is the list of tokens of that document's words, and a search is simply
 * the token of the query word matched against the index — no key material ever reaches the server.
 * Because the PRF is deterministic, the server learns exactly the plaintext *equality pattern*: two
 * ciphertext tokens are equal iff the two plaintext words are equal. The tests do not take that on
 * faith; they build the equality matrix and the frequency vector over ciphertexts and over
 * plaintexts and require them to be identical, and they run a dictionary attack that recovers every
 * keyword whose token appears in the attacker's vocabulary.
 * The contrast makes the trade-off explicit: a randomized (per-occurrence salted) encryption
 * destroys the equality pattern completely — at the price of a search recall of exactly zero.
 *
 * What is simplified: a 32-bit toy PRF instead of HMAC/AES, no encrypted index layers, no
 * access-pattern hiding, no forward privacy, no range or fuzzy queries, and the leakage analysis is
 * per-corpus and exact rather than statistical. What is verified: search results match an
 * independent plaintext map for every word in the corpus; the equality pattern, the frequency
 * vector and the per-document length leak are identical between the plaintext and ciphertext views;
 * a dictionary covering the whole vocabulary recovers every keyword while partial coverage recovers
 * exactly the covered fraction; a randomized scheme leaks no equality but cannot be searched; a
 * wrong key yields no matches and no false positives; and empty/degenerate inputs behave.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** 32-bit avalanche, used as the toy PRF's compression step. */
function mix32(value) {
  let hash = value >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

function hashText(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) hash = (Math.imul(hash, 0x01000193) ^ text.charCodeAt(i)) >>> 0;
  return hash >>> 0;
}

/** Deterministic keyword token: the whole scheme hangs on this being a (toy) PRF. */
const tokenOf = (key, word) => mix32(mix32(key) ^ hashText(word)).toString(16).padStart(8, "0");

/** Randomized encryption of one occurrence: the same word gives a fresh token every time. */
const saltedTokenOf = (key, word, salt) => mix32(mix32(key) ^ hashText(`${word}#${salt}`)).toString(16).padStart(8, "0");

function normalise(corpus) {
  if (!Array.isArray(corpus)) throw new TypeError("the corpus must be an array of documents");
  return corpus.map((document, id) => {
    if (!Array.isArray(document)) throw new TypeError(`document ${id} must be an array of words`);
    return document.map((word) => {
      if (typeof word !== "string") throw new TypeError(`document ${id} contains a non-string word`);
      return word.toLowerCase();
    });
  });
}

/** The encrypted index: token -> document ids, plus the per-document token lists. */
function buildIndex(key, corpus, { randomised = false } = {}) {
  const documents = normalise(corpus);
  const index = new Map();
  const ciphertexts = documents.map((document, id) => document.map((word, position) => {
    const token = randomised ? saltedTokenOf(key, word, `${id}:${position}`) : tokenOf(key, word);
    if (!index.has(token)) index.set(token, new Set());
    index.get(token).add(id);
    return token;
  }));
  return { index, ciphertexts, documents };
}

const search = (key, corpus, word, options) => {
  const { index } = buildIndex(key, corpus, options);
  return [...(index.get(tokenOf(key, word.toLowerCase())) ?? [])].sort((a, b) => a - b);
};

/** Querying an existing index with the wrong key must match nothing at all. */
const searchWithKey = (index, key, word) => [...(index.get(tokenOf(key, word.toLowerCase())) ?? [])].sort((a, b) => a - b);

/** Independent oracle: the plaintext inverted index. */
function plaintextIndex(corpus) {
  const inverted = new Map();
  normalise(corpus).forEach((document, id) => {
    for (const word of document) {
      if (!inverted.has(word)) inverted.set(word, new Set());
      inverted.get(word).add(id);
    }
  });
  return inverted;
}

/** Equality pattern of a sequence of symbols: which positions carry the same symbol. */
function equalityPattern(symbols) {
  const pattern = [];
  for (let i = 0; i < symbols.length; i++) {
    for (let j = i + 1; j < symbols.length; j++) if (symbols[i] === symbols[j]) pattern.push(`${i}=${j}`);
  }
  return pattern.join(",");
}

/** Frequency vector: the counts of each distinct symbol, sorted descending (the server sees it). */
function frequencyVector(symbols) {
  const counts = new Map();
  for (const symbol of symbols) counts.set(symbol, (counts.get(symbol) ?? 0) + 1);
  return [...counts.values()].sort((a, b) => b - a);
}

/**
 * Dictionary attack: every keyword whose token the attacker can compute is recovered. Returns how
 * many of the corpus's distinct words were recovered and how many decoy tokens matched by accident.
 */
function dictionaryAttack(key, corpus, vocabulary) {
  const { index } = buildIndex(key, corpus);
  const plain = plaintextIndex(corpus);
  const recovered = new Set();
  let collisions = 0;
  for (const word of vocabulary) {
    const candidate = (word ?? "").toLowerCase();
    const ids = index.get(tokenOf(key, candidate));
    if (ids === undefined) continue;
    const truth = [...(plain.get(candidate) ?? [])].sort((a, b) => a - b);
    if (equalityPattern([...ids].sort((a, b) => a - b)) === equalityPattern(truth)) recovered.add(candidate);
    if (!plain.has(candidate)) collisions++;
  }
  return { recovered: recovered.size, vocabularySize: plain.size, collisions, words: recovered };
}

module.exports = { mix32, hashText, tokenOf, saltedTokenOf, buildIndex, search, searchWithKey, plaintextIndex, equalityPattern, frequencyVector, dictionaryAttack };

if (require.main === module) {
  const KEY = 0xc0ffee11;
  const CORPUS = [
    ["alpha", "beta", "gamma"],
    ["beta", "delta", "beta"],
    ["alpha", "alpha", "delta"],
    [],
    ["epsilon"],
  ];
  const plain = plaintextIndex(CORPUS);
  const { index, ciphertexts } = buildIndex(KEY, CORPUS);

  // 1. search agreement with the independent plaintext index for every distinct word
  for (const word of plain.keys()) {
    deepStrictEqual(search(KEY, CORPUS, word), [...plain.get(word)].sort((a, b) => a - b), `search(${word})`);
  }
  // 2. tokens are deterministic and separate words never collide on this corpus
  strictEqual(tokenOf(KEY, "alpha"), tokenOf(KEY, "alpha"), "the same word always has the same token");
  ok(tokenOf(KEY, "alpha") !== tokenOf(KEY, "beta"), "different words get different tokens");
  strictEqual(index.size, plain.size, "the index has one entry per distinct keyword");
  // 3. leakage 1: the equality pattern of the ciphertext equals the plaintext's
  const flatPlain = CORPUS.flat().map((word) => word.toLowerCase());
  const flatTokens = ciphertexts.flat();
  strictEqual(equalityPattern(flatTokens), equalityPattern(flatPlain), "deterministic encryption leaks equality exactly");
  strictEqual(new Set(flatTokens).size, new Set(flatPlain).size, "and therefore leaks the number of distinct words");
  // 4. leakage 2: the frequency vector is identical (the server sees the keyword frequencies)
  deepStrictEqual(frequencyVector(flatTokens), frequencyVector(flatPlain));
  // 5. leakage 3: the per-document length, and hence the whole document size distribution, leaks
  deepStrictEqual(ciphertexts.map((tokens) => tokens.length), CORPUS.map((document) => document.length));
  // 6. dictionary attack: full coverage recovers everything, partial coverage recovers exactly that
  const full = dictionaryAttack(KEY, CORPUS, ["alpha", "beta", "gamma", "delta", "epsilon"]);
  strictEqual(full.recovered, full.vocabularySize, "a full dictionary recovers every keyword");
  strictEqual(full.collisions, 0);
  const covered = ["alpha", "beta", "gamma", "zeta"];
  const partial = dictionaryAttack(KEY, CORPUS, covered);
  strictEqual(partial.recovered, covered.filter((word) => plain.has(word)).length, "coverage limits the attack exactly");
  ok(partial.recovered < full.vocabularySize, "an incomplete dictionary leaves words hidden");
  // 7. the randomised scheme destroys the equality pattern but also destroys search
  const { ciphertexts: salted } = buildIndex(KEY, CORPUS, { randomised: true });
  const flatSalted = salted.flat();
  strictEqual(new Set(flatSalted).size, flatSalted.length, "no two salted ciphertexts are equal");
  ok(equalityPattern(flatSalted) !== equalityPattern(flatPlain), "the equality leak is gone");
  deepStrictEqual(search(KEY, CORPUS, "beta", { randomised: true }), [], "and so is the ability to search");
  ok(flatPlain.length > new Set(flatPlain).size, "the plaintext really does contain repeats");
  // 8. a wrong key finds nothing, and never a false positive
  const wrongKey = 0xdeadbeef;
  for (const word of plain.keys()) deepStrictEqual(searchWithKey(index, wrongKey, word), [], `wrong key must not match ${word}`);
  ok(tokenOf(wrongKey, "alpha") !== tokenOf(KEY, "alpha"), "a different key gives a different token");
  // 9. boundaries: unknown word, empty corpus, empty document, repeated word in one document
  deepStrictEqual(search(KEY, CORPUS, "zeta"), [], "a word outside the corpus matches nothing");
  strictEqual(buildIndex(KEY, []).index.size, 0, "an empty corpus has an empty index");
  deepStrictEqual(search(KEY, [[]], "alpha"), [], "an empty document contributes nothing");
  deepStrictEqual(search(KEY, [["word", "word"]], "word"), [0], "a repeated word still yields one document id");
  deepStrictEqual(frequencyVector(["a", "a", "b"]), [2, 1]);
  throws(() => buildIndex(KEY, "not a corpus"), TypeError);
  throws(() => buildIndex(KEY, [["ok", 7]]), TypeError);
  // 10. determinism: identical inputs give identical indexes and ciphertexts
  deepStrictEqual(buildIndex(KEY, CORPUS), buildIndex(KEY, CORPUS));

  console.log(
    `docs=${CORPUS.length} distinctWords=${plain.size} tokens=${index.size} equalityPairs=${equalityPattern(flatTokens).split(",").filter(Boolean).length} ` +
    `dictionaryRecovered=${full.recovered}/${full.vocabularySize} saltedEqualityPairs=${equalityPattern(flatSalted).split(",").filter(Boolean).length}`
  );
  console.log("PASS");
}
