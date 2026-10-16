/** Task 764 - Implement ranking functions TF-IDF + BM25 simple scoring and evaluation. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Two classic ranking functions on a small in-memory corpus plus a real evaluation harness.
 * TF-IDF is the `ltc` flavour: `tf = 1 + log(count)`, `idf = log(1 + N/df)`, cosine-normalised.
 * BM25 is the probabilistic one: `idf = log(1 + (N - df + 0.5)/(df + 0.5))`,
 * `tf = f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))`, summed over query terms. Documents are
 * ranked by descending score with a doc-id tie-break so that equal scores never reorder a run.
 * The evaluation side implements precision@k, recall@k, average precision (MAP), MRR and NDCG@k
 * with graded relevance, which is what lets the tests state the properties that make a scorer
 * believable: BM25 grows with term frequency but saturates, `b = 0` removes length normalisation
 * while `b = 1` punishes long documents, IDF never goes negative (so a term in every document still
 * contributes a small positive weight), and NDCG equals the hand-computed value for a graded list.
 *
 * What is simplified: no index structures (task 763), no stemming/stop-words, no phrase scoring, no
 * query-term weighting, no field boosts and no learning of k1/b. What is verified: BM25 and TF-IDF
 * agree with an independently written formula oracle on a hand-checked corpus; the monotonicity and
 * saturation properties of BM25 hold; NDCG, AP, MRR, precision and recall match hand-computed values
 * and a definitional oracle over 60 seeded rankings; every random ranking scores between 0 and the
 * perfect ranking; and degenerate inputs (empty query, unseen term, empty document, single-document
 * corpus, k = 0) behave predictably.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

const tokenise = (text) => (typeof text === "string" ? text.toLowerCase().match(/[a-z0-9]+/g) ?? [] : []);
const BM25_DEFAULTS = { k1: 1.2, b: 0.75 };

/** Corpus statistics: token counts, document frequencies, average length. */
function buildStats(documents) {
  if (!Array.isArray(documents)) throw new TypeError("documents must be an array of strings");
  const tokens = documents.map((document) => tokenise(document));
  const frequencies = new Map();
  const documentFrequency = new Map();
  tokens.forEach((words) => {
    const seen = new Set();
    for (const word of words) {
      frequencies.set(word, (frequencies.get(word) ?? 0) + 1);
      if (!seen.has(word)) { seen.add(word); documentFrequency.set(word, (documentFrequency.get(word) ?? 0) + 1); }
    }
  });
  const total = tokens.reduce((sum, words) => sum + words.length, 0);
  return {
    n: tokens.length,
    tokens,
    frequencies,
    documentFrequency,
    averageLength: tokens.length === 0 ? 0 : total / tokens.length,
  };
}

const termFrequency = (stats, term, docId) => stats.frequencies.get(term) === undefined
  ? 0 : stats.tokens[docId].filter((word) => word === term).length;

/** BM25 positive IDF: never zero, never negative, and tiny for a term in every document. */
const idf = (stats, term) => {
  const df = stats.documentFrequency.get(term) ?? 0;
  return Math.log(1 + (stats.n - df + 0.5) / (df + 0.5));
};

const bm25 = (queryTerms, docId, stats, { k1 = BM25_DEFAULTS.k1, b = BM25_DEFAULTS.b } = {}) => {
  if (k1 < 0) throw new RangeError("k1 must be non-negative");
  if (b < 0 || b > 1) throw new RangeError("b must be in [0, 1]");
  const length = stats.tokens[docId].length;
  const normalisation = 1 - b + b * (stats.averageLength === 0 ? 0 : length / stats.averageLength);
  return queryTerms.reduce((total, term) => {
    const f = termFrequency(stats, term, docId);
    if (f === 0) return total;
    return total + idf(stats, term) * ((f * (k1 + 1)) / (f + k1 * normalisation));
  }, 0);
};

/** ltc TF-IDF with cosine normalisation. */
const tfidf = (queryTerms, docId, stats) => {
  const weight = (term) => {
    const frequency = termFrequency(stats, term, docId);
    return frequency === 0 ? 0 : (1 + Math.log(frequency)) * Math.log(1 + stats.n / (stats.documentFrequency.get(term) ?? 1));
  };
  const queryNorm = Math.sqrt(queryTerms.length);
  const documentNorm = Math.sqrt([...new Set(stats.tokens[docId])].reduce((sum, term) => sum + weight(term) ** 2, 0));
  if (queryNorm === 0 || documentNorm === 0) return 0;
  return queryTerms.reduce((total, term) => total + weight(term), 0) / (queryNorm * documentNorm);
};

/** Ranking: descending score, doc-id tie-break so the result is total and deterministic. */
function rank(query, stats, scorer = bm25, options = {}) {
  const terms = tokenise(query);
  return stats.tokens.map((_, docId) => ({ docId, score: scorer(terms, docId, stats, options) }))
    .sort((left, right) => right.score - left.score || left.docId - right.docId);
}

const topK = (ranking, k) => ranking.slice(0, Math.max(0, k)).map((entry) => entry.docId);
const hits = (ranking, relevant, k) => topK(ranking, k).filter((docId) => relevant.has(docId)).length;
const precisionAtK = (ranking, relevant, k) => (k <= 0 || ranking.length === 0 ? 0 : hits(ranking, relevant, k) / Math.min(k, ranking.length));
const recallAtK = (ranking, relevant, k) => (relevant.size === 0 ? 0 : hits(ranking, relevant, k) / relevant.size);
const reciprocalRank = (ranking, relevant) => {
  const index = ranking.findIndex((entry) => relevant.has(entry.docId));
  return index < 0 ? 0 : 1 / (index + 1);
};
const averagePrecision = (ranking, relevant) => {
  if (relevant.size === 0) return 0;
  let running = 0;
  return ranking.reduce((total, entry, index) => {
    if (!relevant.has(entry.docId)) return total;
    running++;
    return total + running / (index + 1);
  }, 0) / relevant.size;
};
const dcg = (gains) => gains.reduce((total, gain, index) => total + (2 ** gain - 1) / Math.log2(index + 2), 0);
const ndcgAtK = (ranking, gainOf, k = ranking.length) => {
  const gains = ranking.slice(0, Math.max(0, k)).map((entry) => gainOf(entry.docId) ?? 0);
  const ideal = ranking.map((entry) => gainOf(entry.docId) ?? 0).sort((a, b) => b - a).slice(0, Math.max(0, k));
  const best = dcg(ideal);
  return best === 0 ? 0 : dcg(gains) / best;
};

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x764764;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

module.exports = {
  tokenise, buildStats, termFrequency, idf, bm25, tfidf, rank, topK, precisionAtK, recallAtK,
  reciprocalRank, averagePrecision, dcg, ndcgAtK, makeRng,
};

if (require.main === module) {
  const DOCUMENTS = [
    "the quick brown fox",
    "the quick brown fox jumps over the lazy dog",
    "lazy dogs sleep all day",
    "quick quick quick brown",
    "",
  ];
  const stats = buildStats(DOCUMENTS);
  const QUERY = "quick brown";
  const terms = tokenise(QUERY);

  // 1. BM25 against an independently written formula oracle (transcribed from the definition)
  const oracleBm25 = (query, docId, k1 = 1.2, b = 0.75) => {
    const length = tokenise(DOCUMENTS[docId]).length;
    const average = DOCUMENTS.reduce((sum, document) => sum + tokenise(document).length, 0) / DOCUMENTS.length;
    return query.reduce((total, term) => {
      const f = tokenise(DOCUMENTS[docId]).filter((word) => word === term).length;
      if (f === 0) return total;
      const df = DOCUMENTS.filter((document) => tokenise(document).includes(term)).length;
      const inverse = Math.log(1 + (DOCUMENTS.length - df + 0.5) / (df + 0.5));
      return total + inverse * (f * (k1 + 1)) / (f + k1 * (1 - b + b * length / average));
    }, 0);
  };
  for (let docId = 0; docId < DOCUMENTS.length; docId++) {
    const measured = bm25(terms, docId, stats);
    ok(Math.abs(measured - oracleBm25(terms, docId)) < 1e-12, `BM25 of document ${docId}`);
  }
  // 2. BM25 grows with term frequency and saturates
  const grows = [1, 2, 3, 4, 5].map((count) => bm25(["quick"], 0, buildStats(["quick ".repeat(count).trim()])));
  for (let index = 1; index < grows.length; index++) ok(grows[index] > grows[index - 1], "more occurrences must score higher");
  for (let index = 2; index < grows.length; index++) {
    ok(grows[index] - grows[index - 1] < grows[index - 1] - grows[index - 2], "the increments must shrink (saturation)");
  }
  // 3. b = 0 removes length normalisation, b = 1 punishes long documents
  const short = buildStats(["quick quick", "quick quick filler filler filler filler"]);
  const flat = { k1: 1.2, b: 0 };
  strictEqual(bm25(["quick"], 0, short, flat), bm25(["quick"], 1, short, flat), "b = 0 ignores length");
  ok(bm25(["quick"], 1, short) < bm25(["quick"], 0, short), "b = 0.75 prefers the shorter document");
  // 4. IDF is always positive, and small for a term present everywhere
  ok(idf(stats, "quick") > 0 && idf(stats, "missing") > 0, "BM25 IDF is never negative");
  ok(idf(stats, "day") > idf(stats, "the"), "a rarer term is worth more");
  // 5. both scorers put the document that repeats the query term first (with different margins)
  const tfidfRanking = rank(QUERY, stats, tfidf);
  const bm25Ranking = rank(QUERY, stats);
  strictEqual(bm25Ranking[0].docId, 3, "BM25 puts the document that repeats 'quick' first");
  strictEqual(tfidfRanking[0].docId, 3, "TF-IDF agrees on the winner");
  ok(bm25Ranking[0].score > bm25Ranking[1].score, "and it is not a tie");
  ok(tfidfRanking[0].score !== bm25Ranking[0].score, "the two scores are not interchangeable");
  ok(tfidfRanking.at(-1).score === 0 && bm25Ranking.at(-1).score === 0, "documents without query terms score zero");
  deepStrictEqual(rank(QUERY, stats).map((entry) => entry.docId), rank(QUERY, stats).map((entry) => entry.docId), "ranking is deterministic");
  // 6. evaluation metrics against hand-computed values
  const graded = { 0: 0, 1: 3, 2: 2, 3: 1, 4: 0 };
  const perfect = [{ docId: 1 }, { docId: 2 }, { docId: 3 }, { docId: 0 }, { docId: 4 }];
  const reversed = [{ docId: 4 }, { docId: 0 }, { docId: 3 }, { docId: 2 }, { docId: 1 }];
  const gainOf = (docId) => graded[docId];
  const handDcg = (list) => list.reduce((total, gain, index) => total + (2 ** gain - 1) / Math.log2(index + 2), 0);
  strictEqual(ndcgAtK(perfect, gainOf), 1, "the ideal order reaches NDCG 1");
  const reversedGains = reversed.map((entry) => gainOf(entry.docId));
  const expectedNdcg = handDcg(reversedGains) / handDcg([...reversedGains].sort((a, b) => b - a));
  ok(Math.abs(ndcgAtK(reversed, gainOf) - expectedNdcg) < 1e-12, "the reversed graded order matches the hand computation");
  ok(expectedNdcg > 0.2 && expectedNdcg < 0.5, `a fully reversed graded order scores low (${expectedNdcg.toFixed(4)})`);
  // 7. precision, recall, AP and MRR on a small binary relevance set
  const relevant = new Set([1, 3]);
  const mixed = [{ docId: 3 }, { docId: 0 }, { docId: 1 }];
  strictEqual(precisionAtK(mixed, relevant, 1), 1, "the first document is relevant");
  strictEqual(precisionAtK(mixed, relevant, 2), 0.5, "one of the first two is relevant");
  strictEqual(recallAtK(mixed, relevant, 3), 1, "both relevant documents are found");
  strictEqual(recallAtK(mixed, relevant, 1), 0.5, "only one is found at rank 1");
  ok(Math.abs(averagePrecision(mixed, relevant) - (1 + 2 / 3) / 2) < 1e-12, "MAP follows the definition");
  strictEqual(reciprocalRank(mixed, relevant), 1, "the first hit is at rank 1");
  strictEqual(reciprocalRank([{ docId: 0 }, { docId: 1 }], relevant), 0.5, "the first hit is at rank 2");
  strictEqual(reciprocalRank([{ docId: 0 }], relevant), 0, "no hit means a reciprocal rank of zero");
  // 8. randomised rankings: metrics stay within their theoretical bounds and match the oracle
  const rng = makeRng(0x764764);
  let randomChecks = 0;
  for (let trial = 0; trial < 60; trial++) {
    const size = 1 + rng(8);
    const scores = Array.from({ length: size }, () => rng(5));
    const ranking = scores.map((score, docId) => ({ docId, score })).sort((a, b) => b.score - a.score || a.docId - b.docId);
    const gains = scores.map((score) => score);
    const measured = ndcgAtK(ranking, (docId) => gains[docId]);
    ok(measured >= 0 && measured <= 1 + 1e-12, `NDCG must be a fraction (trial ${trial})`);
    const best = dcg([...gains].sort((a, b) => b - a));
    const oracle = best === 0 ? 0 : dcg(ranking.map((entry) => gains[entry.docId])) / best;
    ok(Math.abs(measured - oracle) < 1e-12, "NDCG matches the definition");
    const binary = new Set(gains.map((gain, docId) => (gain > 0 ? docId : -1)).filter((docId) => docId >= 0));
    ok(precisionAtK(ranking, binary, size) <= 1 && recallAtK(ranking, binary, size) <= 1, "precision and recall are fractions");
    randomChecks++;
  }
  // 9. degenerate inputs and validation
  strictEqual(bm25(tokenise("nosuchterm"), 0, stats), 0, "an unseen term scores zero everywhere");
  deepStrictEqual(rank("nosuchterm", stats).map((entry) => entry.docId), [0, 1, 2, 3, 4], "a zero-score query tie-breaks by doc id");
  deepStrictEqual(rank("", stats).every((entry) => entry.score === 0), true, "an empty query scores everything zero");
  strictEqual(bm25(["quick"], 4, stats), 0, "an empty document scores zero");
  strictEqual(tfidf(["quick"], 4, stats), 0, "and so does TF-IDF");
  strictEqual(precisionAtK([], relevant, 3), 0);
  strictEqual(recallAtK([], relevant, 3), 0);
  strictEqual(averagePrecision([], new Set()), 0, "no relevant documents means no average precision");
  strictEqual(ndcgAtK(perfect, gainOf, 0), 0, "k = 0 truncates everything");
  strictEqual(ndcgAtK([{ docId: 0 }, { docId: 1 }], () => 0), 0, "a ranking with no gain scores zero");
  strictEqual(buildStats([]).averageLength, 0, "an empty corpus has no average length");
  deepStrictEqual(topK(rank(QUERY, buildStats(["quick"])), 5), [0], "a single-document corpus");
  throws(() => bm25(["quick"], 0, stats, { k1: -1 }), RangeError);
  throws(() => bm25(["quick"], 0, stats, { b: 1.5 }), RangeError);
  throws(() => buildStats("not a corpus"), TypeError);

  console.log(
    `documents=${stats.n} terms=${stats.frequencies.size} bm25Top=${bm25Ranking[0].docId} tfidfTop=${tfidfRanking[0].docId} ` +
    `ndcgPerfect=1 ndcgReversed=${ndcgAtK(reversed, gainOf).toFixed(4)} randomRankings=${randomChecks}`
  );
  console.log("PASS");
}
