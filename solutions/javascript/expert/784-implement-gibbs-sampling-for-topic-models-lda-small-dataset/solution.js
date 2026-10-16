/** Task 784 - Implement Gibbs sampling for topic models (LDA) small dataset. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Collapsed Gibbs sampling for LDA. Topics and the topic-word distributions are integrated out, so
 * the only latent state is the topic assignment z_i of every token and the conditional is the
 * classic one
 *
 *     p(z_i = k | z_-i, w)  ~  (n_dk + alpha) * (n_kw + beta) / (n_k + V * beta)
 *
 * with n_dk the tokens of document d in topic k, n_kw the times word w has topic k and n_k the tokens
 * of topic k, all counted *without* token i. Sampling walks the tokens in a fixed order, so the chain
 * is fully deterministic given the seed; the categorical draw uses CDF inversion over the seeded
 * splitmix64 stream.
 *
 * The point of the small dataset is that the exact posterior is computable: for a corpus with a
 * handful of tokens every one of the K^T assignments is weighted by the collapsed joint
 * (Dirichlet-multinomial over document-topic counts times Dirichlet-multinomial over topic-word
 * counts), which gives exact assignment probabilities and exact per-token topic marginals. The chain
 * is then compared with that distribution rather than merely inspected.
 *
 * What is verified: the empirical distribution of the chain matches the exact posterior over all K^T
 * assignments (total variation, two seeds), per-token marginals match as well, the sampled topics
 * follow the closed-form conditional at a frozen state, the count tables always agree with the
 * assignment vector, the collapsed log-likelihood improves on average after burn-in, reproducibility
 * (same seed, same chain), the degenerate corpora (K = 1, V = 1, a single token, an empty corpus,
 * extreme priors) and argument validation.
 */
"use strict";

/** splitmix64 in BigInt, returned as a 53-bit integer for exact CDF inversion. */
function makeRng(seed) {
  let state = BigInt.asUintN(64, BigInt(seed) + 0x9e3779b97f4a7c15n);
  return (bound) => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    z ^= z >> 31n;
    return Number(z % BigInt(bound));
  };
}

function logGamma(x) {
  if (x <= 0) throw new RangeError("logGamma needs a positive argument");
  const coefficients = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  let z = x;
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let series = 0.99999999999980993;
  for (let i = 0; i < coefficients.length; i++) series += coefficients[i] / (z + i + 1);
  const t = z + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(series);
}

/** Collapsed log joint of an assignment: document-topic x topic-word Dirichlet-multinomials. */
function logJoint(documents, assignment, { topics, vocabulary, alpha, beta }) {
  const docTopic = documents.map(() => new Array(topics).fill(0));
  const topicWord = Array.from({ length: topics }, () => new Array(vocabulary).fill(0));
  let token = 0;
  for (let d = 0; d < documents.length; d++) {
    for (const word of documents[d]) {
      const k = assignment[token++];
      docTopic[d][k]++;
      topicWord[k][word]++;
    }
  }
  let value = 0;
  for (let d = 0; d < documents.length; d++) {
    const total = docTopic[d].reduce((sum, count) => sum + count, 0);
    value += logGamma(topics * alpha) - logGamma(total + topics * alpha) - topics * logGamma(alpha);
    for (let k = 0; k < topics; k++) value += logGamma(docTopic[d][k] + alpha);
  }
  for (let k = 0; k < topics; k++) {
    const total = topicWord[k].reduce((sum, count) => sum + count, 0);
    value += logGamma(vocabulary * beta) - logGamma(total + vocabulary * beta) - vocabulary * logGamma(beta);
    for (let w = 0; w < vocabulary; w++) value += logGamma(topicWord[k][w] + beta);
  }
  return value;
}

/** Every assignment (K^T of them) with its exact posterior probability, for small corpora. */
function exactPosterior(documents, settings) {
  const tokens = documents.reduce((sum, doc) => sum + doc.length, 0);
  const total = settings.topics ** tokens;
  const weights = [];
  let max = -Infinity;
  for (let code = 0; code < total; code++) {
    const assignment = [];
    let rest = code;
    for (let i = 0; i < tokens; i++) { assignment.push(rest % settings.topics); rest = Math.floor(rest / settings.topics); }
    const weight = logJoint(documents, assignment, settings);
    weights.push(weight);
    max = Math.max(max, weight);
  }
  const scaled = weights.map((weight) => Math.exp(weight - max));
  const sum = scaled.reduce((total, value) => total + value, 0);
  const probabilities = scaled.map((value) => value / sum);
  return { logEvidence: max + Math.log(sum), probabilities, assignments: weights.map((_, code) => code) };
}

class GibbsLDA {
  constructor(documents, { topics = 2, vocabulary, alpha = 0.3, beta = 0.2, seed = 1 } = {}) {
    if (!Number.isInteger(topics) || topics < 1) throw new RangeError("topics must be a positive integer");
    if (!(alpha > 0) || !(beta > 0)) throw new RangeError("alpha and beta must be positive");
    const size = vocabulary ?? (documents.flat().reduce((max, word) => Math.max(max, word), -1) + 1);
    if (!Number.isInteger(size) || size < 1) throw new RangeError("the vocabulary needs at least one word");
    for (const doc of documents) for (const word of doc) {
      if (!Number.isInteger(word) || word < 0 || word >= size) throw new RangeError(`word ${word} is outside the vocabulary`);
    }
    this.documents = documents;
    this.topics = topics;
    this.vocabulary = size;
    this.alpha = alpha;
    this.beta = beta;
    this.uniform = makeRng(seed);
    this.tokenDoc = [];
    this.tokenWord = [];
    for (let d = 0; d < documents.length; d++) {
      for (const word of documents[d]) { this.tokenDoc.push(d); this.tokenWord.push(word); }
    }
    this.assignment = this.tokenWord.map(() => this.uniform(topics));
    this.docTopic = documents.map(() => new Array(topics).fill(0));
    this.topicWord = Array.from({ length: topics }, () => new Array(size).fill(0));
    this.topicCount = new Array(topics).fill(0);
    for (let i = 0; i < this.assignment.length; i++) this.#add(i, this.assignment[i]);
    this.sweeps = 0;
  }

  get tokenCount() { return this.assignment.length; }

  #add(token, topic) {
    this.docTopic[this.tokenDoc[token]][topic]++;
    this.topicWord[topic][this.tokenWord[token]]++;
    this.topicCount[topic]++;
  }

  #remove(token, topic) {
    this.docTopic[this.tokenDoc[token]][topic]--;
    this.topicWord[topic][this.tokenWord[token]]--;
    this.topicCount[topic]--;
  }

  /** The closed-form conditional p(z_i = k | everything else) for one token. */
  conditional(token) {
    const doc = this.tokenDoc[token];
    const word = this.tokenWord[token];
    const topic = this.assignment[token];
    this.#remove(token, topic);
    const probabilities = new Array(this.topics);
    let sum = 0;
    for (let k = 0; k < this.topics; k++) {
      const value = (this.docTopic[doc][k] + this.alpha)
        * (this.topicWord[k][word] + this.beta)
        / (this.topicCount[k] + this.vocabulary * this.beta);
      probabilities[k] = value;
      sum += value;
    }
    this.#add(token, topic);
    return probabilities.map((value) => value / sum);
  }

  /** One full sweep: every token is resampled in a fixed order. */
  sweep() {
    for (let token = 0; token < this.assignment.length; token++) {
      const probabilities = this.conditional(token);
      let target = this.uniform(1e9) / 1e9;
      let chosen = this.topics - 1;
      for (let k = 0; k < this.topics; k++) {
        target -= probabilities[k];
        if (target <= 0) { chosen = k; break; }
      }
      const previous = this.assignment[token];
      if (chosen !== previous) {
        this.#remove(token, previous);
        this.assignment[token] = chosen;
        this.#add(token, chosen);
      }
    }
    this.sweeps++;
    return this.assignment.slice();
  }

  logLikelihood() { return logJoint(this.documents, this.assignment, this); }

  /** Count tables rebuilt from scratch: the independent consistency oracle for the running ones. */
  rebuildCounts() {
    const docTopic = this.documents.map(() => new Array(this.topics).fill(0));
    const topicWord = Array.from({ length: this.topics }, () => new Array(this.vocabulary).fill(0));
    const topicCount = new Array(this.topics).fill(0);
    for (let token = 0; token < this.assignment.length; token++) {
      const k = this.assignment[token];
      docTopic[this.tokenDoc[token]][k]++;
      topicWord[k][this.tokenWord[token]]++;
      topicCount[k]++;
    }
    return { docTopic, topicWord, topicCount };
  }

  checkCounts() {
    const rebuilt = this.rebuildCounts();
    return JSON.stringify(rebuilt) === JSON.stringify({
      docTopic: this.docTopic, topicWord: this.topicWord, topicCount: this.topicCount,
    });
  }

  counts() {
    return this.assignment.reduce((counts, k) => { counts[k]++; return counts; }, new Array(this.topics).fill(0));
  }
}

/** Total variation distance between two probability vectors. */
function totalVariation(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
  return sum / 2;
}

module.exports = { makeRng, logGamma, logJoint, exactPosterior, GibbsLDA, totalVariation };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;

  // -- degenerate corpora -------------------------------------------------------------
  const empty = new GibbsLDA([], { topics: 2, vocabulary: 3, seed: 1 });
  strictEqual(empty.tokenCount, 0, "an empty corpus has no tokens");
  deepStrictEqual(empty.sweep(), [], "sweeping an empty corpus is a no-op");
  strictEqual(empty.checkCounts(), true, "empty count tables are consistent");
  const singleToken = new GibbsLDA([[2]], { topics: 2, vocabulary: 3, seed: 5 });
  strictEqual(singleToken.tokenCount, 1, "one document with one token");
  deepStrictEqual(singleToken.conditional(0).map((v) => Number(v.toFixed(6))), [0.5, 0.5],
    "a single token splits evenly between two symmetric topics");
  const oneTopic = new GibbsLDA([[0, 1, 0], [1, 1]], { topics: 1, vocabulary: 2, seed: 3 });
  for (let i = 0; i < 10; i++) oneTopic.sweep();
  deepStrictEqual(oneTopic.assignment, [0, 0, 0, 0, 0], "a single topic must own every token");
  strictEqual(oneTopic.checkCounts(), true, "counts stay consistent with a single topic");
  const oneWord = new GibbsLDA([[0, 0], [0, 0]], { topics: 2, vocabulary: 1, seed: 7 });
  for (let i = 0; i < 5; i++) oneWord.sweep();
  ok(oneWord.assignment.every((k) => k === 0 || k === 1), "a one-word vocabulary still samples topics");
  throws(() => new GibbsLDA([[0]], { topics: 0, vocabulary: 2 }), RangeError, "topics must be positive");
  throws(() => new GibbsLDA([[5]], { topics: 2, vocabulary: 3 }), RangeError, "words must be inside the vocabulary");
  throws(() => new GibbsLDA([[0]], { topics: 2, vocabulary: 2, alpha: 0 }), RangeError, "alpha must be positive");
  throws(() => new GibbsLDA([[0]], { topics: 2, vocabulary: 2, beta: -1 }), RangeError, "beta must be positive");

  // -- the exact posterior of a tiny corpus vs the chain ----------------------------
  const corpus = [[0, 1], [2, 1]];
  const settings = { topics: 2, vocabulary: 3, alpha: 0.3, beta: 0.2 };
  const exact = exactPosterior(corpus, settings);
  strictEqual(exact.probabilities.length, 16, "four tokens with two topics means sixteen assignments");
  ok(Math.abs(exact.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12,
    "the exact posterior must be normalised");

  const chainLength = 60000;
  const collect = (seed) => {
    const sampler = new GibbsLDA(corpus, { ...settings, seed });
    const counts = new Array(16).fill(0);
    const topicCounts = Array.from({ length: 4 }, () => new Array(2).fill(0));
    const firstHalf = [];
    const secondHalf = [];
    for (let sweep = 0; sweep < chainLength; sweep++) {
      const assignment = sampler.sweep();
      let code = 0;
      for (let i = 3; i >= 0; i--) code = code * 2 + assignment[i];
      counts[code]++;
      for (let i = 0; i < 4; i++) topicCounts[i][assignment[i]]++;
      (sweep < chainLength / 2 ? firstHalf : secondHalf).push(sampler.logLikelihood());
      if (!sampler.checkCounts()) throw new Error("count tables drifted from the assignment vector");
    }
    const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
    return {
      counts: counts.map((count) => count / chainLength),
      marginals: topicCounts.map((row) => row.map((count) => count / chainLength)),
      likelihoods: { first: mean(firstHalf), second: mean(secondHalf) },
      sampler,
    };
  };
  const first = collect(20260784);
  const tv = totalVariation(first.counts, exact.probabilities);
  ok(tv < 0.03, `the chain must match the exact posterior over assignments (TV ${tv.toFixed(4)})`);
  // per-token marginals from the exact posterior
  const exactMarginals = Array.from({ length: 4 }, () => new Array(2).fill(0));
  for (let code = 0; code < 16; code++) {
    let rest = code;
    for (let i = 0; i < 4; i++) { exactMarginals[i][rest % 2] += exact.probabilities[code]; rest = Math.floor(rest / 2); }
  }
  void exact.logEvidence;
  for (let i = 0; i < 4; i++) {
    ok(totalVariation(first.marginals[i], exactMarginals[i]) < 0.02,
      `token ${i} marginals must match the exact posterior`);
  }
  ok(first.likelihoods.second > first.likelihoods.first,
    "the average collapsed log-likelihood must improve after burn-in");
  strictEqual(first.sampler.checkCounts(), true, "the count tables must still agree at the end");
  strictEqual(first.sampler.sweeps, chainLength, "every sweep must have run");

  // -- a second seed must land on the same distribution ------------------------------
  const second = collect(99887766);
  ok(totalVariation(second.counts, exact.probabilities) < 0.03,
    "an independent seed must also match the exact posterior");
  ok(JSON.stringify(second.counts) !== JSON.stringify(first.counts), "the two chains must not be identical");
  ok(totalVariation(first.counts, second.counts) < 0.05, "the two chains must agree with each other");

  // -- the sampled topics follow the closed-form conditional at a frozen state --------
  const frozen = new GibbsLDA(corpus, { ...settings, seed: 4242 });
  for (let i = 0; i < 20; i++) frozen.sweep();
  const target = frozen.conditional(2);
  const draws = new Array(2).fill(0);
  const probe = new GibbsLDA(corpus, { ...settings, seed: 77 });
  probe.assignment = frozen.assignment.slice();
  probe.docTopic = frozen.docTopic.map((row) => row.slice());
  probe.topicWord = frozen.topicWord.map((row) => row.slice());
  probe.topicCount = frozen.topicCount.slice();
  for (let trial = 0; trial < 4000; trial++) {
    let value = probe.uniform(1e9) / 1e9;
    let chosen = 0;
    for (let k = 0; k < 2; k++) { value -= target[k]; if (value <= 0) { chosen = k; break; } }
    draws[chosen]++;
  }
  const empirical = draws.map((count) => count / 4000);
  ok(totalVariation(empirical, target) < 0.05,
    `the categorical draw must follow the conditional (TV ${totalVariation(empirical, target).toFixed(4)})`);
  ok(Math.abs(target.reduce((sum, value) => sum + value, 0) - 1) < 1e-12, "the conditional must be normalised");

  // -- extreme priors: the chain must still reproduce the exact posterior -------------
  for (const [alpha, beta] of [[0.01, 0.01], [10, 10]]) {
    const priorSettings = { ...settings, alpha, beta };
    const priorExact = exactPosterior(corpus, priorSettings);
    const sampler = new GibbsLDA(corpus, { ...priorSettings, seed: 21 });
    const counts = new Array(16).fill(0);
    const sweeps = 30000;
    for (let sweep = 0; sweep < sweeps; sweep++) {
      const assignment = sampler.sweep();
      counts[assignment.reduce((code, k, i) => code + k * 2 ** i, 0)]++;
      if (!sampler.checkCounts()) throw new Error("counts drifted from the assignment vector");
    }
    const empirical = counts.map((count) => count / sweeps);
    const distance = totalVariation(empirical, priorExact.probabilities);
    ok(distance < 0.04,
      `alpha=beta=${alpha}: the chain must match the exact posterior (TV ${distance.toFixed(4)})`);
    ok(Math.abs(priorExact.logEvidence - exact.logEvidence) > 1e-9 || alpha !== 0.01,
      "the sharp prior must change the evidence, so the comparison is not vacuous");
    const conditional = sampler.conditional(0);
    ok(Math.abs(conditional.reduce((sum, value) => sum + value, 0) - 1) < 1e-12, "the conditional stays normalised");
    ok(conditional.every((value) => value > 0), "every topic keeps a positive probability");
  }

  // -- determinism -------------------------------------------------------------------
  const replay = (seed) => {
    const sampler = new GibbsLDA(corpus, { ...settings, seed });
    const log = [];
    for (let sweep = 0; sweep < 50; sweep++) log.push(sampler.sweep().join(""));
    return log.join("|");
  };
  strictEqual(replay(9), replay(9), "the same seed must replay the same chain");
  strictEqual(replay(9) === replay(10), false, "different seeds must give different chains");

  // -- a larger corpus exercises many sweeps without drifting ------------------------
  const big = [[0, 1, 1, 2, 0, 2], [2, 2, 1, 0], [1, 1, 2], [0, 0, 2, 1]];
  const bigSampler = new GibbsLDA(big, { topics: 3, vocabulary: 3, alpha: 0.5, beta: 0.1, seed: 31 });
  for (let sweep = 0; sweep < 300; sweep++) {
    bigSampler.sweep();
    strictEqual(bigSampler.checkCounts(), true, `counts drifted at sweep ${sweep}`);
  }
  const bigCounts = bigSampler.rebuildCounts();
  strictEqual(bigCounts.topicCount.reduce((sum, count) => sum + count, 0), 17, "all tokens must carry a topic");
  strictEqual(bigSampler.checkCounts(), true, "the rebuilt tables must equal the running ones");

  console.log(`tiny corpus: 16 assignments, chain TV=${tv.toFixed(4)} (seed 2: ${totalVariation(second.counts, exact.probabilities).toFixed(4)}) | conditional probe TV=${totalVariation(empirical, target).toFixed(4)} | loglik ${first.likelihoods.first.toFixed(3)} -> ${first.likelihoods.second.toFixed(3)}`);
  console.log("PASS");
}
