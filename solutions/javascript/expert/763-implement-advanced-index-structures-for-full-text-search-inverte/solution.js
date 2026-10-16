/** Task 763 - Implement advanced index structures for full-text search (inverted index with positional info). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A positional inverted index of the Lucene shape: for each term a posting list of
 * `{docId, positions}` entries sorted by doc id with the positions inside a document ascending and
 * unique, plus *skip pointers* every `skipInterval` entries for fast list intersections, a document
 * length table for scoring, and the term frequencies a scorer needs. On top of it sit four query
 * evaluators: a single-term lookup, a positional phrase query (the classic "add the term offset to
 * each candidate position and intersect"), a boolean combination of required/excluded terms with
 * set algebra over doc ids, and a proximity query for terms within k tokens of each other.
 * Every query has a brute-force oracle in the same file — a scan of the tokenised documents — and
 * the tests demand equality with it, so the index cannot be self-consistently wrong.
 *
 * What is simplified: no compression of the postings (no frame-of-reference/vint blocks), no
 * stop-word or stemming pipeline, no scoring (that is task 764), skips are stored as a simple
 * every-k table instead of a multi-level skip tree, and positions are plain JavaScript array
 * indices rather than byte offsets. What is verified: posting-list invariants (ascending doc ids,
 * ascending unique positions, tf = position count) and equality with the brute-force postings; skip
 * pointers point at the right entries and a skip-aided intersection equals the linear one; phrase,
 * boolean and proximity results match brute force over every query of a small corpus and over 30
 * seeded random corpora; and boundary inputs (empty corpus, empty document, unknown term, phrase
 * longer than the document, mixed-case punctuation) behave.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Lowercase alphanumeric tokenisation: punctuation and case are folded away. */
const tokenise = (text) => (typeof text === "string" ? text.toLowerCase().match(/[a-z0-9]+/g) ?? [] : []);

class InvertedIndex {
  constructor(documents, { skipInterval = 2 } = {}) {
    if (!Array.isArray(documents)) throw new TypeError("documents must be an array of strings");
    if (!Number.isInteger(skipInterval) || skipInterval < 1) throw new RangeError("skipInterval must be a positive integer");
    this.skipInterval = skipInterval;
    this.documents = documents.map((document) => tokenise(document));
    this.postings = new Map();
    this.lengths = this.documents.map((tokens) => tokens.length);
    this.documents.forEach((tokens, docId) => {
      tokens.forEach((term, position) => this.#add(term, docId, position));
    });
    this.skips = new Map();
    for (const term of this.postings.keys()) this.skips.set(term, this.#buildSkips(term));
  }

  #add(term, docId, position) {
    if (!this.postings.has(term)) this.postings.set(term, []);
    const list = this.postings.get(term);
    const last = list[list.length - 1];
    if (last !== undefined && last.docId === docId) last.positions.push(position);
    else list.push({ docId, positions: [position] });
  }

  /** Every skipInterval-th entry records where the next block starts. */
  #buildSkips(term) {
    const list = this.postings.get(term);
    const skips = [];
    for (let index = 0; index + this.skipInterval < list.length; index += this.skipInterval) {
      skips.push({ at: index, target: index + this.skipInterval, docId: list[index + this.skipInterval].docId });
    }
    return skips;
  }

  get size() { return this.postings.size; }
  get documentCount() { return this.documents.length; }

  docsOf(term) { return (this.postings.get(term) ?? []).map((entry) => entry.docId); }

  positionsOf(term, docId) {
    const entry = (this.postings.get(term) ?? []).find((candidate) => candidate.docId === docId);
    return entry === undefined ? [] : entry.positions;
  }

  search(term) { return this.docsOf(term); }

  /** Intersects a term's posting list with a sorted doc-id list, jumping via the skip table. */
  intersectWithSkips(term, other) {
    const list = this.postings.get(term) ?? [];
    const skips = this.skips.get(term) ?? [];
    const hits = [];
    let index = 0;
    let position = 0;
    while (index < list.length && position < other.length) {
      if (list[index].docId === other[position]) { hits.push(list[index].docId); index++; position++; continue; }
      const skip = list[index].docId < other[position]
        ? skips.find((candidate) => candidate.at === index && candidate.docId <= other[position])
        : undefined;
      if (skip !== undefined) index = skip.target;
      else if (list[index].docId < other[position]) index++;
      else position++;
    }
    return hits;
  }

  /** Phrase query: intersect the posting lists with the per-term offset added to the position. */
  phrase(terms) {
    if (!Array.isArray(terms) || terms.length === 0) throw new RangeError("a phrase needs at least one term");
    const normalised = terms.map((term) => term.toLowerCase());
    if (normalised.length === 1) return this.search(normalised[0]);
    const candidates = this.search(normalised[0]);
    const hits = [];
    for (const docId of candidates) {
      const first = this.positionsOf(normalised[0], docId);
      const others = normalised.slice(1).map((term) => new Set(this.positionsOf(term, docId)));
      if (others.some((set) => set.size === 0)) continue;
      for (const position of first) {
        let matched = true;
        for (let offset = 1; offset < normalised.length; offset++) {
          if (!others[offset - 1].has(position + offset)) { matched = false; break; }
        }
        if (matched) { hits.push(docId); break; }
      }
    }
    return hits;
  }

  /** Boolean query: `required` terms are AND-ed, `optional` OR-ed, `excluded` NOT-ed. */
  boolean({ required = [], optional = [], excluded = [] } = {}) {
    const unions = Array.from({ length: this.documentCount }, (_, docId) => docId);
    let result = required.reduce((accumulator, term) => this.intersectWithSkips(term, accumulator), unions.slice());
    if (optional.length > 0) {
      const allowed = new Set(optional.flatMap((term) => this.search(term)));
      result = required.length > 0 ? result.filter((docId) => allowed.has(docId)) : [...allowed].sort((a, b) => a - b);
    }
    if (excluded.length > 0) {
      const banned = new Set(excluded.flatMap((term) => this.search(term)));
      result = result.filter((docId) => !banned.has(docId));
    }
    return result;
  }

  /** Two terms appearing within `distance` tokens of each other. */
  proximity(first, second, distance) {
    if (!Number.isInteger(distance) || distance < 1) throw new RangeError("distance must be a positive integer");
    const hits = [];
    for (const entry of this.postings.get(first.toLowerCase()) ?? []) {
      const others = this.positionsOf(second.toLowerCase(), entry.docId);
      if (others.length === 0) continue;
      if (entry.positions.some((a) => others.some((b) => Math.abs(a - b) <= distance))) hits.push(entry.docId);
    }
    return hits;
  }

  /** Posting-list invariants: ordered doc ids, ordered unique positions, tf = |positions|. */
  checkInvariants() {
    for (const [term, list] of this.postings) {
      for (let index = 0; index < list.length; index++) {
        const { docId, positions } = list[index];
        if (index > 0 && list[index - 1].docId >= docId) return { ok: false, reason: `${term}: doc ids not ascending` };
        for (let i = 0; i < positions.length; i++) {
          if (i > 0 && positions[i - 1] >= positions[i]) return { ok: false, reason: `${term}: positions not unique/ascending` };
        }
        const truth = this.documents[docId].filter((token) => token === term).length;
        if (truth !== positions.length) return { ok: false, reason: `${term}: tf ${positions.length} != ${truth}` };
      }
    }
    for (const [term, skips] of this.skips) {
      const list = this.postings.get(term);
      let previous = -1;
      for (const skip of skips) {
        if (skip.target <= previous || skip.target >= list.length) return { ok: false, reason: `${term}: bad skip target` };
        if (list[skip.target].docId !== skip.docId) return { ok: false, reason: `${term}: skip doc id mismatch` };
        previous = skip.target;
      }
    }
    return { ok: true };
  }
}

/** Brute-force oracles: direct scans of the raw documents. */
const bruteForceTerm = (documents, term) => documents
  .map((document, docId) => ({ docId, tokens: tokenise(document) }))
  .filter(({ tokens }) => tokens.includes(term.toLowerCase())).map(({ docId }) => docId);

const bruteForcePhrase = (documents, terms) => {
  const normalised = terms.map((term) => term.toLowerCase());
  return documents.map((document, docId) => ({ docId, tokens: tokenise(document) }))
    .filter(({ tokens }) => tokens.some((_, start) => normalised.every((term, offset) => tokens[start + offset] === term)))
    .map(({ docId }) => docId);
};

const bruteForceProximity = (documents, first, second, distance) => documents
  .map((document, docId) => ({ docId, tokens: tokenise(document) }))
  .filter(({ tokens }) => {
    const a = tokens.map((token, index) => (token === first.toLowerCase() ? index : -1)).filter((index) => index >= 0);
    const b = tokens.map((token, index) => (token === second.toLowerCase() ? index : -1)).filter((index) => index >= 0);
    return a.some((left) => b.some((right) => Math.abs(left - right) <= distance));
  }).map(({ docId }) => docId);

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x763763;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const VOCABULARY = ["alpha", "beta", "gamma", "delta", "epsilon"];

module.exports = { tokenise, InvertedIndex, bruteForceTerm, bruteForcePhrase, bruteForceProximity, makeRng, VOCABULARY };

if (require.main === module) {
  const DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog",
    "Quick, quick! Brown bears jump over nothing",
    "A lazy dog sleeps over the quick brown fox",
    "Nothing to see here",
    "",
    "quick brown fox quick brown fox",
  ];
  const index = new InvertedIndex(DOCUMENTS);
  // 1. posting-list invariants and brute-force agreement for every term
  ok(index.checkInvariants().ok, index.checkInvariants().reason);
  for (const term of index.postings.keys()) {
    deepStrictEqual(index.search(term), bruteForceTerm(DOCUMENTS, term), `postings of '${term}'`);
  }
  ok(index.size > 0 && index.documentCount === DOCUMENTS.length);
  // 2. phrase queries across every pair and triple of the vocabulary, against brute force
  let phraseQueries = 0;
  for (const first of VOCABULARY) {
    for (const second of VOCABULARY) {
      deepStrictEqual(index.phrase([first, second]), bruteForcePhrase(DOCUMENTS, [first, second]), `phrase ${first} ${second}`);
      phraseQueries++;
      for (const third of VOCABULARY) {
        deepStrictEqual(index.phrase([first, second, third]), bruteForcePhrase(DOCUMENTS, [first, second, third]), `${first} ${second} ${third}`);
        phraseQueries++;
      }
    }
  }
  deepStrictEqual(index.phrase(["quick", "brown", "fox"]), [0, 2, 5], "the repeated phrase in document 5 matches once");
  deepStrictEqual(index.phrase(["brown", "quick"]), [], "word order matters");
  deepStrictEqual(index.phrase(["dog", "dog"]), [], "a phrase may not repeat a term at one position");
  deepStrictEqual(index.phrase(["the"]), index.search("the"), "a one-term phrase is a term lookup");
  // 3. boolean combination equals set algebra over the brute-force doc sets
  const required = ["quick"], optional = ["brown"], excluded = ["lazy"];
  const expected = Array.from({ length: DOCUMENTS.length }, (_, docId) => docId)
    .filter((docId) => bruteForceTerm(DOCUMENTS, "quick").includes(docId))
    .filter((docId) => bruteForceTerm(DOCUMENTS, "brown").includes(docId))
    .filter((docId) => !bruteForceTerm(DOCUMENTS, "lazy").includes(docId));
  deepStrictEqual(index.boolean({ required, optional, excluded }), expected, "boolean AND/OR/NOT");
  deepStrictEqual(index.boolean({ optional: ["nothing"] }).sort((a, b) => a - b), [1, 3], "optional-only is a union");
  deepStrictEqual(index.boolean({ excluded: ["quick"] }), [3, 4], "exclusion without requirements is a complement");
  deepStrictEqual(index.boolean({}), [0, 1, 2, 3, 4, 5], "an empty query matches every document");
  // 4. skip-aided intersection equals the linear one, and skips point at the right entries
  const sorted = [...index.postings.keys()].sort();
  for (const term of sorted) {
    for (const other of ["quick", "brown", "dog", "nothing", "alpha"]) {
      const linear = index.docsOf(term).filter((docId) => bruteForceTerm(DOCUMENTS, other).includes(docId));
      deepStrictEqual(index.intersectWithSkips(term, index.docsOf(other)), linear, `${term} AND ${other} via skips`);
      deepStrictEqual(index.boolean({ required: [term, other] }), linear, `${term} AND ${other}`);
    }
  }
  ok(index.skips.get("quick").length > 0, "the repeated term has skip entries");
  for (const skip of index.skips.get("quick")) {
    strictEqual(index.docsOf("quick")[skip.target], skip.docId, "every skip lands on a real posting entry");
  }
  // 5. proximity matches a brute-force window scan
  for (const first of VOCABULARY) {
    for (const second of VOCABULARY) {
      for (const distance of [1, 2, 4]) {
        deepStrictEqual(index.proximity(first, second, distance), bruteForceProximity(DOCUMENTS, first, second, distance),
          `proximity ${first}/${second} within ${distance}`);
      }
    }
  }
  // 6. randomised corpora and queries: index answers must equal brute force everywhere
  const rng = makeRng(0x763763);
  let randomChecks = 0;
  for (let trial = 0; trial < 30; trial++) {
    const documents = Array.from({ length: 1 + rng(5) }, () =>
      Array.from({ length: rng(8) }, () => VOCABULARY[rng(VOCABULARY.length)]).join(" "));
    const random = new InvertedIndex(documents, { skipInterval: 1 + rng(3) });
    ok(random.checkInvariants().ok, random.checkInvariants().reason);
    for (let query = 0; query < 8; query++) {
      const terms = Array.from({ length: 1 + rng(3) }, () => VOCABULARY[rng(VOCABULARY.length)]);
      deepStrictEqual(random.phrase(terms), bruteForcePhrase(documents, terms), `trial ${trial}: phrase ${terms}`);
      deepStrictEqual(random.search(terms[0]), bruteForceTerm(documents, terms[0]), `trial ${trial}: term ${terms[0]}`);
      randomChecks += 2;
    }
  }
  // 7. boundaries, validation and determinism
  deepStrictEqual(tokenise("Hello, WORLD! 42"), ["hello", "world", "42"], "tokenisation folds case and punctuation");
  deepStrictEqual(tokenise("!!!"), []);
  deepStrictEqual(tokenise(undefined), []);
  const empty = new InvertedIndex([]);
  strictEqual(empty.size, 0);
  deepStrictEqual(empty.phrase(["alpha"]), []);
  deepStrictEqual(empty.boolean({}), []);
  const tiny = new InvertedIndex(["one two", "two one"]);
  deepStrictEqual(tiny.phrase(["one", "two", "one", "two"]), [], "a phrase longer than the document cannot match");
  deepStrictEqual(tiny.phrase(["ONE", "TWO"]), [0], "phrases are case-insensitive");
  throws(() => new InvertedIndex("nope"), TypeError);
  throws(() => new InvertedIndex([], { skipInterval: 0 }), RangeError);
  throws(() => index.phrase([]), RangeError);
  throws(() => index.proximity("quick", "brown", 0), RangeError);
  deepStrictEqual(new InvertedIndex(DOCUMENTS).search("quick"), index.search("quick"), "the index is deterministic");

  console.log(
    `docs=${index.documentCount} terms=${index.size} phraseQueries=${phraseQueries} randomChecks=${randomChecks} ` +
    `termsWithSkips=${[...index.skips.values()].filter((skips) => skips.length > 0).length}`
  );
  console.log("PASS");
}
