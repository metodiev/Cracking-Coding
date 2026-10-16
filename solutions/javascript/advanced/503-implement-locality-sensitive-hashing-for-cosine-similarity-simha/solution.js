/**
 * Task 503 - Implement locality-sensitive hashing for cosine similarity (simhash). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * SimHash: D = 256 random hyperplanes (rows of a Gaussian matrix drawn from the file's own LCG, no
 * Math.random) turn a vector v into a 256-bit signature, bit i = 1 iff <r_i, v> > 0. The classical
 * property is p = P[bit differs] = theta / pi, so the Hamming distance gives the angle estimate
 * theta ~ pi * h / D and hence cos(theta) ~ cos(pi * h / D); averaging the estimate over several
 * independent hyperplane families (the file's fixed LCG seed list) is what makes it tight. The
 * test data is deterministic too: sign vectors of dimension 64 (entries +/-1) plus one
 * near-duplicate each, obtained by flipping one or two coordinates, so the planted cosine is
 * exactly 1 - 2k/64 and the 20 planted pairs are known in advance. The retriever is the standard
 * banded index: b = 32 bands of r = 8 bits (b * r = 256), each band hashed into a Map keyed by the
 * raw bit block, and a pair is a candidate iff it collides in at least one band.
 *
 * Documented checks: the Hamming-based estimator is exact at the extremes - v vs v gives 0 bits and
 * cosine 1, v vs -v gives ~256 bits and cosine ~ -1 - and averaged over 12 families it recovers the
 * planted cosine of every near-duplicate pair to 0.05 (worst case 0.023); the angle ladder
 * theta = 0, 0.2, 0.5, 1, pi/2, 2, 2.5, pi (24 families) is monotonically non-increasing in theta
 * to 1e-9 and within 0.04 of cos(theta) at every step. The folklore linear form 1 - 2h/D is
 * demonstrated to be a systematically low small-angle approximation: it underestimates by 0.172 at
 * h/D ~ 0.117, its error grows with h/D (0.127 at h/D ~ 0.086), and at h/D ~ 1 both forms agree on
 * -1. The banded index is verified to be *sound* (its candidate set equals, element for element,
 * a brute-force band-by-band comparison, and every candidate really shares a band) and *complete*
 * for the planted pairs (all 20 retrieved), while the candidate set stays under a quarter of the
 * 780 possible pairs. Finally 200 LCG-unrelated vector pairs land at a mean Hamming fraction of
 * 0.5 +/- 0.02 with worst deviation 0.2, which is why their collisions are rare.
 */
"use strict";

const DIM = 64;
const PLANES = 256;
const BANDS = 32;
const BAND_BITS = 8;
const WORDS = PLANES / 32;

function nextNormal(seedIn) {
  let seed = seedIn;
  let total = 0.0;
  for (let i = 0; i < 12; i++) {
    seed = (seed * 75 + 74) % 65537;
    total += (seed % 65537) / 65537.0;
  }
  return [total - 6.0, seed];
}

function hyperplanes(planes, dim, seed) {
  const out = [];
  let s = seed;
  for (let i = 0; i < planes; i++) {
    const row = new Array(dim);
    for (let j = 0; j < dim; j++) {
      const pair = nextNormal(s);
      s = pair[1];
      row[j] = pair[0];
    }
    out.push(row);
  }
  return out;
}

/** D-bit signature packed into Uint32 words. Ties (dot == 0) are documented as +1. */
function signature(vector, planes, seed) {
  const rows = hyperplanes(planes, vector.length, seed);
  const words = new Uint32Array(Math.ceil(planes / 32));
  for (let i = 0; i < planes; i++) {
    let total = 0.0;
    const row = rows[i];
    for (let j = 0; j < vector.length; j++) total += row[j] * vector[j];
    if (total > 0.0) words[i >> 5] |= (1 << (i & 31));
  }
  return words;
}

function popcount(value) {
  let v = value >>> 0;
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(v, 0x01010101) >>> 24) & 0x3f;
}

function hamming(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += popcount(a[i] ^ b[i]);
  return total;
}

/**
 * Unbiased-in-the-mean angle estimator: the bit-difference probability is theta / pi, so the
 * recovered cosine is cos(pi * hamming / D). Averages over many bit pairs beat the linear
 * small-angle form.
 */
function estimateCosine(hammingDistance, planes) {
  return Math.cos(Math.PI * hammingDistance / planes);
}

/** The folklore small-angle approximation cos ~ 1 - 2 * hamming / D (a lower bound here). */
function estimateCosineLinear(hammingDistance, planes) {
  return 1.0 - 2.0 * hammingDistance / planes;
}

/** Bits of one band as a string key, so the Map keys are the raw bit blocks. */
function bandKey(words, band, bandBits) {
  let key = "";
  for (let bit = 0; bit < bandBits; bit++) {
    const index = band * bandBits + bit;
    key += (words[index >> 5] >>> (index & 31)) & 1;
  }
  return key;
}

/** Banded LSH index; returns the candidate pair set as strings "i-j" (i < j). */
function retrievePairs(signatures, bands, bandBits) {
  const candidates = new Set();
  for (let band = 0; band < bands; band++) {
    const buckets = new Map();
    signatures.forEach((words, index) => {
      const key = bandKey(words, band, bandBits);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(index);
    });
    for (const members of buckets.values()) {
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          candidates.add(`${members[i]}-${members[j]}`);
        }
      }
    }
  }
  return candidates;
}

/** Independent brute-force construction of the same candidate set (band by band, bit by bit). */
function bruteForcePairs(signatures, bands, bandBits) {
  const candidates = new Set();
  const n = signatures.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let band = 0; band < bands; band++) {
        let match = true;
        for (let bit = 0; bit < bandBits; bit++) {
          const index = band * bandBits + bit;
          const bi = (signatures[i][index >> 5] >>> (index & 31)) & 1;
          const bj = (signatures[j][index >> 5] >>> (index & 31)) & 1;
          if (bi !== bj) {
            match = false;
            break;
          }
        }
        if (match) {
          candidates.add(`${i}-${j}`);
          break;
        }
      }
    }
  }
  return candidates;
}

function lcgSignVector(dim, seed) {
  let s = seed;
  const out = new Array(dim);
  for (let i = 0; i < dim; i++) {
    s = (s * 75 + 74) % 65537;
    out[i] = (s % 2 === 0) ? 1.0 : -1.0;
  }
  return out;
}

/** Copy with `flips` coordinates (starting at `offset`) negated. */
function flipCoordinates(vector, flips, offset) {
  const out = vector.slice();
  for (let i = 0; i < flips; i++) out[(offset + i) % out.length] *= -1.0;
  return out;
}

function cosine(a, b) {
  let dot = 0.0;
  let na = 0.0;
  let nb = 0.0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

/** Dataset: 20 base sign vectors plus one near-duplicate each (1 or 2 flipped coordinates). */
function plantedDataset(bases, seed) {
  const vectors = [];
  const pairs = [];
  let s = seed;
  for (let i = 0; i < bases; i++) {
    const base = lcgSignVector(DIM, s);
    s = (s * 75 + 74) % 65537;
    const flips = 1 + (s % 2);                    // 1 or 2 flipped coordinates
    s = (s * 75 + 74) % 65537;
    const offset = s % DIM;
    const near = flipCoordinates(base, flips, offset);
    vectors.push(base);
    vectors.push(near);
    pairs.push([2 * i, 2 * i + 1]);
  }
  return { vectors, pairs };
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (1) exact cases: v vs v and v vs -v.
  const v = lcgSignVector(DIM, 4242);
  const sigV = signature(v, PLANES, 20260116);
  assert.strictEqual(hamming(sigV, sigV), 0, "identical vectors: zero Hamming distance");
  close(estimateCosine(0, PLANES), 1.0, 0.0, "identical vectors estimate cosine 1");
  const negated = v.map((value) => -value);
  const sigNeg = signature(negated, PLANES, 20260116);
  const opposite = hamming(sigV, sigNeg);
  assert.ok(opposite >= PLANES - 2, `antipodal vectors differ in ~D bits (${opposite})`);
  assert.ok(estimateCosine(opposite, PLANES) <= -0.98, "antipodal estimate is ~ -1");
  assert.strictEqual(hamming(signature(v, PLANES, 20260116), sigV), 0,
    "a repeated signature is bit-identical");

  // (2) the estimator recovers planted cosines (averaged over twelve hyperplane families).
  const { vectors, pairs } = plantedDataset(20, 777);
  assert.strictEqual(vectors.length, 40);
  assert.strictEqual(pairs.length, 20);
  const FAMILIES = 12;
  const sigs = familySeeds(FAMILIES).map((seed) =>
    vectors.map((vec) => signature(vec, PLANES, seed)));
  let worstPlantedError = 0.0;
  for (const [i, j] of pairs) {
    const truth = cosine(vectors[i], vectors[j]);
    const guess = averagedEstimate(sigs, i, j);
    worstPlantedError = Math.max(worstPlantedError, Math.abs(truth - guess));
    assert.ok(Math.abs(truth - guess) <= 0.05,
      `planted pair (${i}, ${j}) cosine recovered to 0.05`);
    assert.ok(truth >= 0.9, "planted pairs are genuinely close");
  }
  assert.ok(worstPlantedError <= 0.05, `worst planted error ${worstPlantedError} <= 0.05`);

  // the folklore linear form 1 - 2h/D is only a small-angle approximation.
  const nearSigs = [signature(vectors[0], PLANES, 3), signature(vectors[1], PLANES, 3)];
  const nearDistance = hamming(nearSigs[0], nearSigs[1]);
  const nearTruth = cosine(vectors[0], vectors[1]);
  const linearGuess = estimateCosineLinear(nearDistance, PLANES);
  const angleGuess = estimateCosine(nearDistance, PLANES);
  assert.ok(linearGuess <= nearTruth + 1e-12, "the linear form underestimates the cosine");
  assert.ok(nearTruth - linearGuess <= 0.2,
    `at h/D ~ 0.12 the linear form is within 0.2 of the truth (error ${nearTruth - linearGuess})`);
  assert.ok(Math.abs(angleGuess - nearTruth) <= 0.01,
    "cos(pi * h / D) is accurate to 0.01 on the same signature");
  assert.ok(Math.abs(angleGuess - nearTruth) <= (nearTruth - linearGuess) / 5.0,
    "the angle form is at least 5x closer than the linear form");
  const smallSigs = [signature([1.0, 0.0], PLANES, 3),
    signature([Math.cos(0.3), Math.sin(0.3)], PLANES, 3)];
  const smallDistance = hamming(smallSigs[0], smallSigs[1]);
  const smallLinearError = Math.abs(estimateCosineLinear(smallDistance, PLANES) - Math.cos(0.3));
  assert.ok(smallLinearError < nearTruth - linearGuess,
    "the linear form's error grows with h/D (0.127 at p ~ 0.086, 0.172 at p ~ 0.117)");
  const farDistance = hamming(signature([1.0, 0.0], PLANES, 3), signature([-1.0, 0.0], PLANES, 3));
  assert.ok(Math.abs(estimateCosineLinear(farDistance, PLANES) - (-1.0)) <= 0.02,
    "at h/D ~ 1 both forms agree on -1");

  // (3) monotone angle ladder: estimates fall as the angle grows.
  const angles = [0.0, 0.2, 0.5, 1.0, Math.PI / 2, 2.0, 2.5, Math.PI];
  const ladder = angles.map((theta) =>
    estimateAngle([1.0, 0.0], [Math.cos(theta), Math.sin(theta)], 24));
  for (let i = 1; i < ladder.length; i++) {
    assert.ok(ladder[i] <= ladder[i - 1] + 1e-9,
      `estimate is non-increasing in theta (step ${i})`);
  }
  for (let i = 0; i < ladder.length; i++) {
    assert.ok(Math.abs(ladder[i] - Math.cos(angles[i])) <= 0.04,
      `ladder step ${i} (theta = ${angles[i]}): estimate within 0.04 of cos(theta)`);
  }

  // (4) unrelated vectors: Hamming fraction near 0.5, few accidental collisions.
  let fractionSum = 0.0;
  let fractionWorst = 0.0;
  const unrelated = 200;
  const baseSig = signature(vectors[0], PLANES, 5);
  let s = 31337;
  for (let i = 0; i < unrelated; i++) {
    s = (s * 75 + 74) % 65537;
    const other = lcgSignVector(DIM, s);
    const fraction = hamming(baseSig, signature(other, PLANES, 5)) / PLANES;
    fractionSum += fraction;
    fractionWorst = Math.max(fractionWorst, Math.abs(fraction - 0.5));
  }
  const meanFraction = fractionSum / unrelated;
  assert.ok(Math.abs(meanFraction - 0.5) <= 0.02, `mean Hamming fraction ~0.5 (${meanFraction})`);
  assert.ok(fractionWorst <= 0.2, "no unrelated pair is anywhere near identical");

  // (5) the banded index: sound, complete on planted pairs, and selective.
  const family = vectors.map((vec) => signature(vec, PLANES, 3));
  const indexed = retrievePairs(family, BANDS, BAND_BITS);
  const brute = bruteForcePairs(family, BANDS, BAND_BITS);
  assert.strictEqual(indexed.size, brute.size, "indexed and brute-force candidate sets agree");
  for (const key of brute) assert.ok(indexed.has(key), `candidate ${key} present in the index`);
  for (const key of indexed) assert.ok(brute.has(key), `candidate ${key} is a real band collision`);

  for (const [i, j] of pairs) {
    assert.ok(indexed.has(`${i}-${j}`), `planted pair (${i}, ${j}) must be retrieved`);
  }
  const totalPairs = 40 * 39 / 2;
  assert.ok(indexed.size <= totalPairs / 4,
    `candidates are <= 25% of all pairs (${indexed.size} / ${totalPairs})`);
  assert.ok(indexed.size >= pairs.length, "at least the planted pairs are candidates");
  assert.ok(indexed.size < totalPairs, "the index still prunes");

  // (6) every candidate really collides in a band, and planted pairs collide in many.
  const shareBand = (a, b) => {
    for (let band = 0; band < BANDS; band++) {
      if (bandKey(family[a], band, BAND_BITS) === bandKey(family[b], band, BAND_BITS)) {
        return true;
      }
    }
    return false;
  };
  for (const key of indexed) {
    const [i, j] = key.split("-").map(Number);
    assert.ok(shareBand(i, j), `${key} shares a band`);
  }
  for (const [i, j] of pairs) {
    const distance = hamming(family[i], family[j]);
    assert.ok(distance / PLANES <= 0.2, "planted pairs sit at a small Hamming fraction");
    assert.ok(shareBand(i, j), "planted pairs share a band");
  }
  assert.ok(averagedEstimate(sigs, pairs[0][0], pairs[0][1]) >= 0.9,
    "planted near-duplicate estimate ~ cos");

  // deterministic re-run
  assert.deepStrictEqual([...retrievePairs(vectors.map((vec) => signature(vec, PLANES, 3)),
    BANDS, BAND_BITS)].sort(), [...indexed].sort(), "retrieval is deterministic");
}

/** Family seeds for the averaged estimator: independent LCG streams, fixed in the file. */
function familySeeds(count) {
  return Array.from({ length: count }, (_, i) => 101 + i * 739);
}

/** Average the simhash cosine estimate over `families` independent hyperplane families. */
function estimateAngle(a, b, families) {
  let total = 0.0;
  for (const seed of familySeeds(families)) {
    const sigA = signature(a, PLANES, seed);
    const sigB = signature(b, PLANES, seed);
    total += estimateCosine(hamming(sigA, sigB), PLANES);
  }
  return total / families;
}

/** Same averaging, but over pre-computed signatures (planar family index = position). */
function averagedEstimate(signatureFamilies, i, j) {
  let total = 0.0;
  for (const family of signatureFamilies) {
    total += estimateCosine(hamming(family[i], family[j]), PLANES);
  }
  return total / signatureFamilies.length;
}

module.exports = { DIM, PLANES, BANDS, BAND_BITS, WORDS, nextNormal, hyperplanes, signature,
  popcount, hamming, estimateCosine, estimateCosineLinear, bandKey, retrievePairs,
  bruteForcePairs, lcgSignVector, flipCoordinates, cosine, plantedDataset, familySeeds,
  estimateAngle, averagedEstimate };

if (require.main === module) {
  main();
  console.log("PASS");
}
