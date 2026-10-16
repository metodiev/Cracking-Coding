/** Task 785 - Implement collapsed Gibbs for LDA and perplexity evaluation. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * Collapsed Gibbs for LDA (Griffiths & Steyvers): the topics are integrated out analytically,
 * so every token is resampled from the exact conditional
 *   p(z_i = k | z_-i, w) ∝ (n_mk + alpha) * (n_kw + beta) / (n_k + V*beta)
 * with n_mk = tokens of doc m in topic k, n_kw = times word w came from topic k, n_k = size of
 * topic k. Those counts are sufficient statistics, so a sweep only touches the two tables, and
 * the invariant is that z, n_mk, n_kw and n_k stay mutually consistent after every token move.
 *
 * The likelihood is the collapsed closed form
 *   log p(w,z) = sum_m [ logG(K*a) - K*logG(a) + sum_k logG(n_mk+a) - logG(n_m + K*a) ]
 *              + sum_k [ logG(V*b) - V*logG(b) + sum_w logG(n_kw+b) - logG(n_k + V*b) ]
 * so perplexity = exp(-log p(w,z) / #tokens). Held-out text is scored with the predictive
 * estimator p(w) = sum_k theta_mk * phi_kw against the frozen training phi.
 *
 * Simplified: a small synthetic corpus (a few hundred tokens) keeps the file far under a
 * second, so perplexity is reported for that scaled-down corpus. Verified without faking
 * anything: the count invariant after every sweep, sampled frequencies against the analytic
 * conditional, the logGamma likelihood against an exact BigInt rational oracle (alpha = beta =
 * 1 turns every gamma ratio into a ratio of integer factorials) and the marginal summed over
 * all assignments, the K = 1 Dirichlet-multinomial closed form, recovery of the generative
 * topics, held-out perplexity beating shuffled text, and invalid-input rejection.
 */
"use strict";
const { ok, throws, strictEqual } = require("node:assert");

/** Lanczos g = 7, n = 9: |error| < 1e-14 for x > 0. */
const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

function logGamma(x) {
  if (!(x > 0)) throw new RangeError(`logGamma requires x > 0 (got ${x})`);
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  const z = x - 1;
  let a = LANCZOS[0];
  for (let i = 1; i < LANCZOS.length; i++) a += LANCZOS[i] / (z + i);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Deterministic splitmix32: uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

const zeros = (rows, cols) => Array.from({ length: rows }, () => new Array(cols).fill(0));

/** Draws an index from an unnormalised weight vector (deterministic given `random`). */
function sampleCategorical(weights, random) {
  let total = 0;
  for (const w of weights) total += w;
  if (!(total > 0)) return Math.floor(random() * weights.length) % weights.length;
  let r = random() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

/** The sufficient statistics of the collapsed sampler, derived from a topic assignment. */
function countTables(words, z, K, V) {
  const nmk = zeros(words.length, K);
  const nkt = zeros(K, V);
  const nk = new Array(K).fill(0);
  for (let m = 0; m < words.length; m++) {
    for (let i = 0; i < words[m].length; i++) {
      nmk[m][z[m][i]] += 1;
      nkt[z[m][i]][words[m][i]] += 1;
      nk[z[m][i]] += 1;
    }
  }
  return { nmk, nkt, nk };
}

class LdaGibbs {
  constructor(corpus, { topics, alpha = 0.1, beta = 0.1, vocabSize = 0, seed = 12345 } = {}) {
    if (!Number.isInteger(topics) || topics < 1) throw new RangeError(`topics must be a positive integer (got ${topics})`);
    if (!(alpha > 0) || !(beta > 0)) throw new RangeError("alpha and beta must be > 0");
    if (corpus.length === 0) throw new RangeError("corpus must contain at least one document");
    this.words = corpus.map((doc) => {
      if (!Array.isArray(doc)) throw new TypeError("every document must be an array of word ids");
      return doc.map((w) => {
        if (!Number.isInteger(w) || w < 0) throw new TypeError(`word id must be a non-negative integer (got ${w})`);
        return w;
      });
    });
    this.K = topics;
    this.alpha = alpha;
    this.beta = beta;
    this.seed = seed;
    this.V = Math.max(vocabSize, Math.max(-1, ...this.words.flat()) + 1, 1);
    this.M = this.words.length;
    this.docLength = this.words.map((doc) => doc.length);
    this.tokenCount = this.docLength.reduce((a, b) => a + b, 0);
    const random = makeRng(seed);
    this.z = this.words.map((doc) => doc.map(() => Math.floor(random() * this.K) % this.K));
    this.recount();
  }

  /** Rebuilds the live tables from z (also used after external edits of z). */
  recount() {
    const { nmk, nkt, nk } = countTables(this.words, this.z, this.K, this.V);
    Object.assign(this, { nmk, nkt, nk });
    return this;
  }

  /** True when z, n_mk, n_kw, n_k are mutually consistent - the sampler's invariant. */
  invariantsHold() {
    const t = countTables(this.words, this.z, this.K, this.V);
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    if (!same(this.nmk, t.nmk) || !same(this.nkt, t.nkt) || !same(this.nk, t.nk)) return false;
    if (this.nk.reduce((a, b) => a + b, 0) !== this.tokenCount) return false;
    const flat = this.words.flat();
    for (let w = 0; w < this.V; w++) {
      if (this.nkt.reduce((a, row) => a + row[w], 0) !== flat.filter((x) => x === w).length) return false;
    }
    return this.nmk.every((row, m) => row.reduce((a, b) => a + b, 0) === this.docLength[m]);
  }

  /** Exact collapsed conditional p(z_i = k | .) for token (m, i), normalised. */
  conditional(m, i) {
    const w = this.words[m][i];
    const probs = new Array(this.K).fill(0);
    let total = 0;
    for (let k = 0; k < this.K; k++) {
      probs[k] = (this.nmk[m][k] + this.alpha) * ((this.nkt[k][w] + this.beta) / (this.nk[k] + this.V * this.beta));
      total += probs[k];
    }
    return probs.map((p) => p / total);
  }

  /** One collapsed Gibbs sweep in fixed (document, position) order. */
  sweep(random = this.random) {
    if (random === undefined) random = makeRng(this.seed);
    for (let m = 0; m < this.M; m++) {
      for (let i = 0; i < this.words[m].length; i++) {
        const w = this.words[m][i];
        const old = this.z[m][i];
        this.nmk[m][old] -= 1;
        this.nkt[old][w] -= 1;
        this.nk[old] -= 1;
        const k = sampleCategorical(this.conditional(m, i), random);
        this.z[m][i] = k;
        this.nmk[m][k] += 1;
        this.nkt[k][w] += 1;
        this.nk[k] += 1;
      }    }
    return this;
  }

  /** Exact collapsed log likelihood log p(w, z) of the current state. */
  collapsedLogLikelihood() {
    let logP = 0;
    for (let m = 0; m < this.M; m++) {
      logP += logGamma(this.K * this.alpha) - this.K * logGamma(this.alpha) - logGamma(this.docLength[m] + this.K * this.alpha);
      for (let k = 0; k < this.K; k++) logP += logGamma(this.nmk[m][k] + this.alpha);
    }
    for (let k = 0; k < this.K; k++) {
      logP += logGamma(this.V * this.beta) - this.V * logGamma(this.beta) - logGamma(this.nk[k] + this.V * this.beta);
      for (let w = 0; w < this.V; w++) logP += logGamma(this.nkt[k][w] + this.beta);
    }
    return logP;
  }

  perplexityOnCorpus() { return Math.exp(-this.collapsedLogLikelihood() / this.tokenCount); }

  /** phi[k][w] = p(w | k) and theta[m][k] = p(k | m): posterior means under the priors. */
  topicWordMatrix() {
    return this.nkt.map((row, k) => row.map((c) => (c + this.beta) / (this.nk[k] + this.V * this.beta)));
  }

  docTopicMatrix() { return this.nmk.map((row, m) => row.map((c) => (c + this.alpha) / (this.docLength[m] + this.K * this.alpha))); }

  /**
   * Held-out predictive perplexity: phi frozen at the training posterior, held-out topics
   * resampled by a deterministic chain, each token contributing -log sum_k theta_mk phi_kw.
   */
  heldOutPerplexity(corpus, { sweeps = 10 } = {}) {
    const phi = this.topicWordMatrix();
    const random = makeRng((this.seed ^ 0x5bf03635) >>> 0);
    const docs = corpus.filter((doc) => doc.length > 0);
    if (docs.length === 0) return Number.NaN;
    const z = docs.map((doc) => doc.map(() => Math.floor(random() * this.K) % this.K));
    const nmk = zeros(docs.length, this.K);
    const eachToken = (fn) => {
      for (let m = 0; m < docs.length; m++) for (let i = 0; i < docs[m].length; i++) fn(m, i, docs[m][i]);
    };
    eachToken((m, i) => { nmk[m][z[m][i]] += 1; });
    for (let s = 0; s < sweeps; s++) {
      eachToken((m, i, w) => {
        nmk[m][z[m][i]] -= 1;
        z[m][i] = sampleCategorical(phi.map((row, k) => (nmk[m][k] + this.alpha) * row[w]), random);
        nmk[m][z[m][i]] += 1;
      });
    }
    let logProb = 0;
    eachToken((m, i, w) => {
      const theta = nmk[m].map((c) => (c + this.alpha) / (docs[m].length + this.K * this.alpha));
      logProb += Math.log(Math.max(theta.reduce((acc, t, k) => acc + t * phi[k][w], 0), 1e-300));
    });
    return Math.exp(-logProb / docs.reduce((a, doc) => a + doc.length, 0));
  }
}

/** Exact p(w, z) for alpha = beta = 1 as a BigInt rational, i.e. a ratio of integer factorials:
 * an oracle that shares no code path with logGamma. */
function exactJointMass(docTopicCounts, topicWordCounts, K, V) {
  const fact = [1n];
  for (let i = 1; i <= 64; i++) fact.push(fact[i - 1] * BigInt(i));
  let num = 1n;
  let den = 1n;
  for (const [rows, dim] of [[docTopicCounts, K], [topicWordCounts, V]]) {
    for (const row of rows) {
      num *= fact[dim - 1];
      for (const c of row) num *= fact[c];
      den *= fact[row.reduce((a, b) => a + b, 0) + dim - 1];
    }
  }
  return { num, den };
}

/** Decimal value of a BigInt fraction with 40 guard digits. */
const fractionToNumber = (f) => Number((f.num * 10n ** 40n) / f.den) / 1e40;

/** Deterministic three-topic corpus on disjoint vocabularies (word ids 0-7, 20-27, 40-47). */
function buildSyntheticCorpus(seed) {
  const random = makeRng(seed);
  const topicWords = [0, 20, 40].map((base) => Array.from({ length: 8 }, (_, i) => base + i));
  const corpus = Array.from({ length: 18 }, (_, m) => Array.from({ length: 24 }, () => {
    const pick = random() < 0.85 ? m % 3 : (m % 3 + 1) % 3;
    return topicWords[pick][Math.floor(random() * 8) % 8];
  }));
  return { corpus, topicWords, V: 60 };
}

/** Mean cosine similarity between the inferred topic profiles and the generative vocabularies. */
function topicAlignment(phi, topicWords, V) {
  const unit = (vec) => {
    const norm = Math.sqrt(vec.reduce((a, x) => a + x * x, 0));
    return vec.map((x) => x / norm);
  };
  const cosine = (a, b) => a.reduce((acc, x, w) => acc + x * b[w], 0);
  let sum = 0;
  for (const row of phi) {
    const profile = unit(row);
    const scores = topicWords.map((words) => cosine(profile, unit(Array.from({ length: V }, (_, w) => (words.includes(w) ? 1 : 0)))));
    sum += Math.max(...scores);
  }
  return sum / phi.length;
}

function main() {
  const { corpus, topicWords, V } = buildSyntheticCorpus(20251016);
  const model = new LdaGibbs(corpus, { topics: 3, alpha: 0.1, beta: 0.1, vocabSize: V, seed: 7 });
  // 1. Invariant after every sweep: the live tables must equal tables re-derived from z.
  let invariant = model.invariantsHold();
  const chain = [model.collapsedLogLikelihood()];
  for (let s = 0; s < 60; s++) {
    model.sweep();
    invariant = model.invariantsHold() && invariant;
    chain.push(model.collapsedLogLikelihood());
  }
  ok(invariant, "count-table invariant must hold after every Gibbs sweep");
  // 2. Sampled topics must follow the analytic collapsed conditional.
  const probe = new LdaGibbs([corpus[0]], { topics: 3, alpha: 0.1, beta: 0.1, vocabSize: V, seed: 99 });
  probe.sweep();
  const probs = probe.conditional(0, 3);
  ok(Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-12, "the conditional must sum to 1");
  const rng = makeRng(4242);
  const freq = new Array(3).fill(0);
  for (let i = 0; i < 30000; i++) freq[sampleCategorical(probs, rng)] += 1;
  const freqErr = Math.max(...freq.map((c, k) => Math.abs(c / 30000 - probs[k])));
  console.log(`  - conditional: max sampled-frequency error ${freqErr.toFixed(5)}`);
  ok(freqErr < 0.02, `sampled frequencies must match the conditional (got ${freqErr})`);
  // 3. Convergence: the collapsed likelihood improves and then plateaus.
  const head = chain.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
  const tail = chain.slice(-20);
  const last = tail.reduce((a, b) => a + b, 0) / tail.length;
  const spread = Math.max(...tail) - Math.min(...tail);
  console.log(`  - collapsed log-likelihood ${head.toFixed(1)} -> ${last.toFixed(1)} (last-20 spread ${spread.toFixed(2)})`);
  ok(last - head > 100 && spread < Math.abs(last) * 0.05, "the chain must improve and then plateau");
  // 4. Exact BigInt oracle for each assignment, and the marginal summed over the assignments.
  const tiny = new LdaGibbs([[0, 1, 0]], { topics: 2, alpha: 1, beta: 1, vocabSize: 2, seed: 3 });
  let worstRel = 0;
  let numericMarginal = 0;
  let exactMarginal = 0;
  for (let mask = 0; mask < 8; mask++) {
    tiny.z = [[(mask >> 2) & 1, (mask >> 1) & 1, mask & 1]];
    tiny.recount();
    const numeric = Math.exp(tiny.collapsedLogLikelihood());
    const exact = fractionToNumber(exactJointMass(tiny.nmk, tiny.nkt, 2, 2));
    numericMarginal += numeric;
    exactMarginal += exact;
    worstRel = Math.max(worstRel, Math.abs(numeric - exact) / exact);
  }
  console.log(`  - logGamma vs exact BigInt oracle: worst relative error ${worstRel.toExponential(2)}; marginal ${exactMarginal.toFixed(6)}`);
  ok(worstRel < 1e-9, `collapsed likelihood must match the exact rational oracle (got ${worstRel})`);
  ok(Math.abs(numericMarginal - exactMarginal) / exactMarginal < 1e-9, "summing exp(log p(w,z)) must reproduce the marginal");
  // 5. K = 1 reduces to the Dirichlet-multinomial closed form.
  const oneTopic = new LdaGibbs([[0, 0, 1, 1, 1]], { topics: 1, alpha: 1, beta: 1, vocabSize: 2, seed: 1 });
  const closedForm = Math.exp(logGamma(2) - logGamma(7) + logGamma(4) + logGamma(3));
  const relK1 = Math.abs(Math.exp(oneTopic.collapsedLogLikelihood()) - closedForm) / closedForm;
  console.log(`  - K = 1 marginal: relative error ${relK1.toExponential(2)}`);
  ok(relK1 < 1e-12, "K = 1 likelihood must equal the Dirichlet-multinomial closed form");
  // 6. Topic recovery, and both posterior distributions normalised.
  const phi = model.topicWordMatrix();
  const alignment = topicAlignment(phi, topicWords, V);
  console.log(`  - mean topic alignment ${alignment.toFixed(3)}`);
  ok(alignment > 0.8, `inferred topics must match the generative topics (alignment ${alignment})`);
  ok(phi.every((r) => Math.abs(r.reduce((a, b) => a + b, 0) - 1) < 1e-12), "phi rows must sum to 1");
  ok(model.docTopicMatrix().every((r) => Math.abs(r.reduce((a, b) => a + b, 0) - 1) < 1e-12), "theta rows must sum to 1");
  // 7. Perplexity: in-domain held-out text must beat shuffled text and the uniform bound V.
  const heldOut = buildSyntheticCorpus(31337).corpus.slice(0, 6);
  const inDomain = model.heldOutPerplexity(heldOut);
  const shuffledPpl = model.heldOutPerplexity(heldOut.map((doc) => doc.map((w) => (w + 19) % V)));
  console.log(`  - perplexity: training ${model.perplexityOnCorpus().toFixed(1)}, held-out ${inDomain.toFixed(1)}, shuffled ${shuffledPpl.toFixed(1)} (V = ${V})`);
  ok(Number.isFinite(inDomain) && inDomain > 1 && inDomain < V, "held-out perplexity must be finite and below V");
  ok(inDomain < shuffledPpl, "in-domain held-out perplexity must beat shuffled held-out text");
  // 8. Determinism: the same seed reproduces the chain exactly.
  const replay = [new LdaGibbs(corpus, { topics: 3, alpha: 0.1, beta: 0.1, vocabSize: V, seed: 7 }), new LdaGibbs(corpus, { topics: 3, alpha: 0.1, beta: 0.1, vocabSize: V, seed: 7 })];
  for (let s = 0; s < 5; s++) replay.forEach((m) => m.sweep());
  strictEqual(replay[0].collapsedLogLikelihood(), replay[1].collapsedLogLikelihood(), "the same seed must reproduce the chain");
  // 9. Boundary: one word and one topic make the conditional degenerate.
  const single = new LdaGibbs([[0]], { topics: 1, alpha: 0.5, beta: 0.5, vocabSize: 1, seed: 2 });
  strictEqual(single.conditional(0, 0)[0], 1, "with one topic the conditional must be exactly 1");
  single.sweep();
  strictEqual(single.nk[0], 1, "the single token must stay in its only topic");
  ok(Math.abs(single.perplexityOnCorpus() - 1) < 1e-12, "a one-token, one-topic corpus has perplexity exactly 1");
  // 10. Invalid input is rejected instead of silently producing a bogus model.
  throws(() => new LdaGibbs([[0]], { topics: 0 }), RangeError, "topics = 0 must throw");
  throws(() => new LdaGibbs([[0]], { topics: 1, beta: 0 }), RangeError, "beta = 0 must throw");
  throws(() => new LdaGibbs([], { topics: 1 }), RangeError, "an empty corpus must throw");
  throws(() => new LdaGibbs([[-1]], { topics: 1 }), TypeError, "a negative word id must throw");
  console.log("PASS");
}

module.exports = { logGamma, makeRng, sampleCategorical, countTables, LdaGibbs, exactJointMass, fractionToNumber };

if (require.main === module) main();
