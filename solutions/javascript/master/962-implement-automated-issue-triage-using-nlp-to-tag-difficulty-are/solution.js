/** Task 962 - Implement automated issue triage using NLP to tag difficulty/area.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A small bag-of-words triage pipeline. Text is lower-cased, split on non-alphanumerics and stripped of
 * stopwords; a multinomial Naive Bayes model with Laplace smoothing is trained on a labelled issue
 * corpus and produces a normalised posterior over areas; a keyword rule set derives the difficulty tag;
 * and a confidence threshold routes weak evidence to a human instead of guessing. Scoring happens in log
 * space (so probabilities are a softmax of summed log-likelihoods) and ties are broken by area id, which
 * keeps everything deterministic.
 *
 * Simplified: real NLP feature engineering (embeddings, n-grams beyond unigrams, issue templates) is out
 * of scope -- unigrams with Laplace smoothing stand in for it, and the difficulty tag is a keyword rule
 * rather than a model. Verified: hand-written holdout accuracy plus the confusion matrix and its
 * precision/recall, a hand-computed posterior oracle on a 2x2 corpus, normalisation, unseen-word
 * smoothing arithmetic, the difficulty rules, threshold routing at both ends, the empty-text boundary,
 * training-order independence and deterministic replay.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const STOPWORDS = new Set(["the", "a", "an", "of", "in", "on", "to", "with", "and", "for", "is", "it", "when", "that", "this", "we", "our"]);
const tokenize = (text) => text.toLowerCase().split(/[^a-z0-9+#]+/).filter((token) => token && !STOPWORDS.has(token));
const AREAS = ["parser", "scheduler", "network", "storage", "graphics"];
const TRAINING = [
  { text: "parser fails on nested grammar with wrong precedence token", area: "parser" },
  { text: "token stream tokenizer reports grammar ambiguity precedence", area: "parser" },
  { text: "grammar precedence parse tree token stream is broken", area: "parser" },
  { text: "scheduler skips queue entries and preempts the wrong deadline task", area: "scheduler" },
  { text: "queue preempt deadline scheduler starvation under load", area: "scheduler" },
  { text: "deadline queue preempt policy in scheduler is unfair", area: "scheduler" },
  { text: "network packet retry loop drops latency measurements", area: "network" },
  { text: "latency spikes and packet retry storm on network", area: "network" },
  { text: "packet loss retry storm latency budget on network", area: "network" },
  { text: "storage compaction breaks index durability guarantee", area: "storage" },
  { text: "index compaction durability test fails on storage", area: "storage" },
  { text: "durability compaction index corruption in storage", area: "storage" },
  { text: "graphics shader renders wrong pixel texture", area: "graphics" },
  { text: "texture pixel shader artifacts in graphics", area: "graphics" },
  { text: "pixel shader texture blending is wrong in graphics", area: "graphics" },
];
const TEST_SET = [
  { text: "token precedence grammar error", area: "parser" },
  { text: "parse tree token stream mismatch", area: "parser" },
  { text: "queue preempt starvation on deadline", area: "scheduler" },
  { text: "scheduler deadline queue unfair", area: "scheduler" },
  { text: "packet retry latency regression", area: "network" },
  { text: "network packet loss retry", area: "network" },
  { text: "index compaction durability bug", area: "storage" },
  { text: "storage compaction index corruption", area: "storage" },
  { text: "shader texture pixel artifacts", area: "graphics" },
  { text: "graphics pixel shader wrong", area: "graphics" },
];
const DIFFICULTY_RULES = [
  { level: "advanced", keywords: ["distributed", "consensus", "scalability", "race", "complexity", "corruption"] },
  { level: "intermediate", keywords: ["performance", "memory", "concurrency", "cache", "deadline", "latency"] },
];
const DEFAULT_DIFFICULTY = "beginner";
const CONFIDENCE_THRESHOLD = 0.6;

/** Train counts plus the smoothed vocabulary; counts are order independent by construction. */
function train(corpus) {
  const documents = new Map(AREAS.map((area) => [area, 0]));
  const totals = new Map(AREAS.map((area) => [area, 0]));
  const counts = new Map();
  const vocabulary = new Set();
  for (const doc of corpus) {
    documents.set(doc.area, (documents.get(doc.area) || 0) + 1);
    for (const token of tokenize(doc.text)) {
      vocabulary.add(token);
      totals.set(doc.area, totals.get(doc.area) + 1);
      const key = `${doc.area}\u0001${token}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  }
  const model = { documents, totals, counts, vocabulary, size: corpus.length };
  /** Laplace-smoothed likelihood P(token | area) = (count + 1) / (totals + vocabulary size). */
  model.wordLogProbability = (token, area) => Math.log(((counts.get(`${area}\u0001${token}`) || 0) + 1) / ((totals.get(area) || 0) + vocabulary.size));
  model.priorLogProbability = (area) => Math.log(((documents.get(area) || 0) + 1) / (corpus.length + AREAS.length));
  return model;
}
/** Posterior over areas: log-prior plus summed log-likelihoods, normalised in probability space. */
function posterior(model, text) {
  const tokens = tokenize(text);
  const scored = AREAS.map((area) => ({ area, log: model.priorLogProbability(area)
    + tokens.reduce((a, token) => a + model.wordLogProbability(token, area), 0) }));
  const best = Math.max(...scored.map((entry) => entry.log));
  const weights = scored.map((entry) => ({ area: entry.area, weight: Math.exp(entry.log - best) }));
  const total = weights.reduce((a, entry) => a + entry.weight, 0);
  return weights.map((entry) => ({ area: entry.area, probability: entry.weight / total }))
    .sort((a, b) => (b.probability - a.probability) || (a.area < b.area ? -1 : 1));
}
function difficultyOf(text) {
  const tokens = new Set(tokenize(text));
  for (const rule of DIFFICULTY_RULES) {
    if (rule.keywords.some((keyword) => tokens.has(keyword))) return rule.level;
  }
  return DEFAULT_DIFFICULTY;
}
/** Full triage verdict: area, difficulty, confidence and the human-routing decision. */
function triage(model, text, { threshold = CONFIDENCE_THRESHOLD } = {}) {
  const probabilities = posterior(model, text);
  const [top] = probabilities;
  const needsHuman = top.probability < threshold;
  return { area: needsHuman ? null : top.area, difficulty: difficultyOf(text), confidence: top.probability,
    probabilities, needsHuman, reason: needsHuman ? `confidence ${top.probability.toFixed(3)} below ${threshold}` : "confident" };
}
function evaluate(model, testSet, options) {
  const matrix = Object.fromEntries(AREAS.map((truth) => [truth, Object.fromEntries(AREAS.map((pred) => [pred, 0]))]));
  const escalated = [];
  let correct = 0;
  for (const item of testSet) {
    const verdict = triage(model, item.text, options);
    if (verdict.area === null) { escalated.push(item.text); continue; }
    matrix[item.area][verdict.area] += 1;
    if (verdict.area === item.area) correct += 1;
  }
  const precision = Object.fromEntries(AREAS.map((area) => {
    const predicted = AREAS.reduce((a, truth) => a + matrix[truth][area], 0);
    return [area, predicted === 0 ? 0 : matrix[area][area] / predicted];
  }));
  const recall = Object.fromEntries(AREAS.map((area) => {
    const actual = AREAS.reduce((a, pred) => a + matrix[area][pred], 0);
    return [area, actual === 0 ? 0 : matrix[area][area] / actual];
  }));
  return { matrix, correct, accuracy: correct / testSet.length, precision, recall, escalated };
}

function main() {
  const rng = makeRng(20240962);
  const model = train(TRAINING);
  const results = evaluate(model, TEST_SET);
  // 1. Happy path: every holdout issue is labelled with its own area, one confusion-matrix row each.
  assert.strictEqual(results.accuracy, 1, `accuracy ${results.accuracy}`);
  assert(AREAS.every((area) => results.precision[area] === 1 && results.recall[area] === 1), `precision/recall ${JSON.stringify(results.precision)}`);
  assert(AREAS.every((area) => AREAS.reduce((a, pred) => a + results.matrix[area][pred], 0) === 2), "two held-out issues per area");
  // 2. Independent posterior oracle: hand-derived Naive Bayes on a 2x2 corpus with Laplace smoothing.
  const tiny = train([{ text: "alpha alpha beta", area: "parser" }, { text: "alpha gamma", area: "scheduler" },
    { text: "beta delta", area: "parser" }, { text: "gamma gamma", area: "scheduler" }]);
  // Hand counts: 4 documents over 5 labels, vocabulary {alpha,beta,gamma,delta} = 4 words;
  // parser totals 5 tokens (alpha x2, beta x2, delta x1) and scheduler totals 4 (alpha x1, gamma x3),
  // so the priors are (2+1)/(4+5) for those two and (0+1)/(4+5) for the three empty areas.
  const like = (count, total) => (count + 1) / (total + 4);
  const handParser = (3 / 9) * like(2, 5) * like(2, 5) * like(1, 5);
  const handScheduler = (3 / 9) * like(1, 4) * like(0, 4) * like(0, 4);
  const handEmpty = (1 / 9) * like(0, 0) * like(0, 0) * like(0, 0);
  const expectedParser = handParser / (handParser + handScheduler + 3 * handEmpty);
  const probabilities = posterior(tiny, "alpha beta delta");
  const oracleTop = probabilities.find((entry) => entry.area === "parser").probability;
  assert(tiny.vocabulary.size === 4 && Math.abs(oracleTop - expectedParser) < 1e-12,
    `posterior oracle ${oracleTop} != ${expectedParser}`);
  assert(oracleTop > 0.5 && probabilities[0].area === "parser", `the hand arithmetic favours parser (${oracleTop})`);
  // 3. Normalisation and smoothing: posteriors sum to one, and an unseen word keeps every area alive.
  const parserTotal = tiny.totals.get("parser");
  assert(Math.abs(probabilities.reduce((a, entry) => a + entry.probability, 0) - 1) < 1e-12, "posteriors sum to 1");
  const unseen = tiny.wordLogProbability("zzz-unseen", "parser");
  assert(Math.abs(unseen - Math.log(1 / (parserTotal + 4))) < 1e-12, `unseen word smoothed to ${Math.exp(unseen)}`);
  assert(Number.isFinite(unseen) && unseen < 0, "unseen words are penalised, never zero");
  assert(posterior(tiny, "zzz-unseen").every((entry) => Number.isFinite(entry.probability) && entry.probability > 0),
    "an unknown token still yields a usable posterior");
  // 4. Difficulty rules: keyword hit or the documented default, checked by hand.
  assert.strictEqual(difficultyOf("race between distributed workers"), "advanced", "advanced keyword wins");
  assert.strictEqual(difficultyOf("latency spike with cache miss"), "intermediate", "intermediate keyword");
  assert.strictEqual(difficultyOf("typo in the docs"), "beginner", "no keyword falls back to beginner");
  assert.strictEqual(difficultyOf("distributed cache latency"), "advanced", "advanced outranks intermediate");
  // 5. Threshold routing at both ends, plus the exact-boundary predicate.
  const strong = triage(model, "token precedence grammar error");
  assert(strong.needsHuman === false && strong.area === "parser" && strong.confidence > CONFIDENCE_THRESHOLD, `strong verdict ${JSON.stringify(strong.area)}`);
  assert(triage(model, "queue preempt deadline", { threshold: 1.01 }).needsHuman, "a threshold above 1 routes everything to a human");
  assert(triage(model, "queue preempt deadline", { threshold: 0 }).needsHuman === false, "a zero threshold never routes away");
  const weak = triage(model, "something feels off today");
  assert(weak.needsHuman === true && weak.area === null && weak.reason.startsWith("confidence"), `weak verdict ${JSON.stringify(weak)}`);
  // 6. Boundary: empty text carries no evidence at all and must always be escalated.
  const blank = triage(model, "");
  assert(blank.needsHuman && blank.probabilities.every((entry) => Math.abs(entry.probability - 1 / AREAS.length) < 1e-12),
    `empty text is uniform (${blank.confidence})`);
  assert(blank.difficulty === "beginner" && tokenize("").length === 0, "empty text has no tokens and no difficulty signal");
  // 7. Training order independence: shuffling the corpus cannot change counts or predictions.
  const shuffled = TRAINING.slice();
  for (let i = shuffled.length - 1; i > 0; i -= 1) { const j = rng.int(i + 1); [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]; }
  assert.deepStrictEqual(evaluate(train(shuffled), TEST_SET, { threshold: 0.6 }), results, "training order does not matter");
  assert.deepStrictEqual(triage(train(shuffled), "shader texture pixel artifacts"), triage(model, "shader texture pixel artifacts"), "same verdict");
  // 8. Determinism: retraining rebuilds the same counts, and verdicts replay byte for byte.
  const shape = (m) => ({ documents: [...m.documents].sort(), totals: [...m.totals].sort(),
    counts: [...m.counts].sort(), vocabulary: [...m.vocabulary].sort() });
  assert.deepStrictEqual(shape(train(TRAINING)), shape(model), "the model replays identically");
  assert.deepStrictEqual(TEST_SET.map((item) => triage(model, item.text)), TEST_SET.map((item) => triage(model, item.text)), "verdicts replay");
  console.log(`triage: vocab=${model.vocabulary.size} docs=${model.size} accuracy=${results.accuracy} `
    + `confidence=${strong.confidence.toFixed(3)} escalated=${[weak, blank].filter((v) => v.needsHuman).length}/2`);
  console.log("PASS");
}

module.exports = { makeRng, STOPWORDS, tokenize, AREAS, TRAINING, TEST_SET, DIFFICULTY_RULES,
  DEFAULT_DIFFICULTY, CONFIDENCE_THRESHOLD, train, posterior, difficultyOf, triage, evaluate };
if (require.main === module) main();
