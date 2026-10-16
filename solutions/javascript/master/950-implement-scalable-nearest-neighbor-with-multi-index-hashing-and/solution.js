/** Task 950 - Implement scalable nearest neighbor with multi-index hashing and rerank pipeline.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Binary vectors of 128 bits are split into eight 16-bit blocks, and each block is indexed in a hash
 * table. A query with radius r probes the tables of every subset of m - r blocks, so any vector that
 * differs from the query in at most r blocks is guaranteed to be retrieved - the soundness property
 * multi-index hashing is built on - and the candidate list is then reranked with the exact Hamming
 * distance. Because a bit flip always changes its own block, "at most r differing bits" implies "at
 * most r differing blocks", which is why the recall for true within-radius neighbours is exactly 1.
 *
 * Simplified: the vectors are 128-bit (8 blocks of 16 bits) instead of thousands of bits, the sub-key
 * tables are built lazily per probed subset instead of eagerly (a space/time trade-off that real
 * systems tune), and there is no quantization or graph layer above the index. Verified: the block
 * partition invariant, the Hamming distance against a bit-by-bit oracle, exact recall at r = m,
 * guaranteed recall for all within-radius neighbours, approximate recall for the nearest neighbours
 * with a small candidate fraction, rerank ordering, duplicate/self/empty/single boundaries and
 * determinism.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

const BLOCKS = 8;
const BLOCK_BITS = 16;

const popcount16 = (x) => {
  let value = x - ((x >> 1) & 0x5555);
  value = (value & 0x3333) + ((value >> 2) & 0x3333);
  value = (value + (value >> 4)) & 0x0f0f;
  return (value + (value >> 8)) & 0x1f;
};
const hamming = (a, b) => {
  let distance = 0;
  for (let i = 0; i < a.length; i += 1) distance += popcount16(a[i] ^ b[i]);
  return distance;
};
/** Oracle: the same distance counted bit by bit. */
const hammingSlow = (a, b) => {
  let distance = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ^ b[i];
    for (let bit = 0; bit < BLOCK_BITS; bit += 1) if (x & (1 << bit)) distance += 1;
  }
  return distance;
};

function randomVector(rng) {
  const vector = new Uint16Array(BLOCKS);
  for (let i = 0; i < BLOCKS; i += 1) vector[i] = rng.int(1 << BLOCK_BITS);
  return vector;
}
/** Flip `count` random bits, spreading them over distinct blocks so the radius is exact. */
function flipBits(base, count, rng) {
  const vector = Uint16Array.from(base);
  const blocks = new Set();
  while (blocks.size < count) blocks.add(rng.int(BLOCKS));
  for (const block of blocks) vector[block] ^= 1 << rng.int(BLOCK_BITS);
  return vector;
}
/** The eight 16-bit words are the 128-bit vector, so re-encoding is lossless. */
const vectorOf = (words) => Uint16Array.from(words);
const blocksOf = (vector) => Array.from(vector);

function combinations(items, size) {
  if (size === 0) return [[]];
  if (items.length < size) return [];
  const [head, ...rest] = items;
  return [...combinations(rest, size - 1).map((tail) => [head, ...tail]), ...combinations(rest, size)];
}
const subsetKey = (vector, subset) => subset.map((i) => vector[i]).join(",");

/** The index: lazily built sub-key tables per probed block subset. */
function buildIndex(vectors) {
  return { vectors, tables: new Map() };
}
function candidatesFor(index, query, radius) {
  const m = query.length;
  const all = Array.from({ length: m }, (_, i) => i);
  if (radius >= m) return Array.from({ length: index.vectors.length }, (_, i) => i);
  const subsetSize = Math.max(1, m - radius);
  const found = new Set();
  for (const subset of combinations(all, subsetSize)) {
    const signature = subset.join(",");
    let table = index.tables.get(signature);
    if (!table) {
      table = new Map();
      index.vectors.forEach((vector, id) => {
        const key = subsetKey(vector, subset);
        table.set(key, [...(table.get(key) ?? []), id]);
      });
      index.tables.set(signature, table);
    }
    for (const id of table.get(subsetKey(query, subset)) ?? []) found.add(id);
  }
  return [...found];
}
/** Rerank pipeline: exact distances over the candidate list, then the top-k. */
function search(index, query, { radius = 2, topK = 5 } = {}) {
  const candidates = candidatesFor(index, query, radius);
  const scored = candidates.map((id) => ({ id, distance: hamming(index.vectors[id], query) }))
    .sort((a, b) => a.distance - b.distance || a.id - b.id);
  return { candidates, results: scored.slice(0, topK) };
}
/** Ground truth: the exact top-k over the whole dataset. */
function bruteForce(vectors, query, topK = 5) {
  return vectors.map((vector, id) => ({ id, distance: hamming(vector, query) }))
    .sort((a, b) => a.distance - b.distance || a.id - b.id)
    .slice(0, topK);
}
const sameResults = (a, b) => a.length === b.length && a.every((r, i) => r.id === b[i].id && r.distance === b[i].distance);

function main() {
  const rng = splitmix64(950);
  const dataset = Array.from({ length: 2000 }, () => randomVector(rng));
  const index = buildIndex(dataset);
  const queries = Array.from({ length: 20 }, () => flipBits(dataset[rng.int(dataset.length)], 2, rng));

  // 1. The block partition is lossless and covers all 128 bits exactly once.
  const sample = dataset[0];
  assert(sample.length * BLOCK_BITS === 128, "eight 16-bit blocks make a 128-bit vector");
  const recombined = vectorOf(blocksOf(sample));
  assert(hamming(recombined, sample) === 0 && [...recombined].every((w, i) => w === sample[i]), "blocks reconstruct the vector exactly");

  // 2. Oracle: the fast popcount distance equals the bit-by-bit count.
  for (let i = 0; i < 200; i += 1) {
    const a = randomVector(rng);
    const b = randomVector(rng);
    assert(hamming(a, b) === hammingSlow(a, b), "popcount and bit-by-bit distances agree");
  }
  assert(hamming(sample, sample) === 0, "a vector is at distance zero from itself");

  // 3. Exact mode: radius = m probes everything, so recall is perfect and the rerank matches brute force.
  const exact = search(index, queries[0], { radius: BLOCKS, topK: 5 });
  assert(exact.candidates.length === dataset.length, "radius m probes the whole dataset");
  assert(sameResults(exact.results, bruteForce(dataset, queries[0], 5)), "exact mode reproduces the brute-force ranking");

  // 4. Guaranteed recall: every neighbour within the radius is in the candidate set.
  let checked = 0;
  for (const query of queries) {
    const truth = bruteForce(dataset, query, dataset.length).filter((r) => r.distance <= 2);
    const found = new Set(search(index, query, { radius: 2, topK: dataset.length }).candidates);
    assert(truth.every((r) => found.has(r.id)), `a within-radius neighbour was missed for query ${query.join("-")}`);
    checked += truth.length;
  }
  assert(checked > 0, `the guarantee was exercised on ${checked} within-radius neighbours`);

  // 5. Approximate recall for the true nearest neighbour and a small candidate fraction.
  let hits = 0;
  let candidateTotal = 0;
  for (const query of queries) {
    const truth = bruteForce(dataset, query, 1)[0];
    const run = search(index, query, { radius: 2, topK: 1 });
    if (run.results[0] && run.results[0].id === truth.id) hits += 1;
    candidateTotal += run.candidates.length;
  }
  const recall = hits / queries.length;
  const candidateFraction = candidateTotal / queries.length / dataset.length;
  assert(recall === 1, `radius-two search finds every near neighbour, recall ${recall}`);
  assert(candidateFraction < 0.1, `candidates must be a small slice of the data, got ${candidateFraction}`);

  // 6. Monotonicity: a larger radius can only add candidates and never lose a hit.
  const small = search(index, queries[1], { radius: 1, topK: 5 });
  const large = search(index, queries[1], { radius: 2, topK: 5 });
  assert(small.candidates.length <= large.candidates.length, `radius 1 probes ${small.candidates.length} <= ${large.candidates.length}`);
  assert(large.candidates.length <= dataset.length, "the candidate list never exceeds the dataset");
  assert(sameResults(search(index, queries[1], { radius: 2, topK: 5 }).results, large.results), "search is deterministic");

  // 7. Rerank ordering: distances are sorted and the reported distances are exact.
  for (const query of queries.slice(0, 5)) {
    const run = search(index, query, { radius: 2, topK: 10 });
    run.results.forEach((result, i) => {
      assert(result.distance === hamming(dataset[result.id], query), "rerank reports exact distances");
      assert(i === 0 || run.results[i - 1].distance <= result.distance, "rerank output is sorted");
    });
  }

  // 8. Duplicates: a self-query returns distance zero, and exact copies are all retrievable.
  const duplicated = [...dataset, ...dataset.slice(0, 50)];
  const dupIndex = buildIndex(duplicated);
  const selfRun = search(dupIndex, dataset[7], { radius: 2, topK: 5 });
  assert(selfRun.results[0].distance === 0 && selfRun.results[0].id === 7, "a self-query ranks itself first");
  assert(selfRun.candidates.includes(2000 + 7), "the duplicate copy is retrieved as well");

  // 9. Boundary: an empty index, a single-vector index and topK larger than the dataset.
  const empty = search(buildIndex([]), randomVector(rng), { radius: 2, topK: 5 });
  assert(empty.candidates.length === 0 && empty.results.length === 0, "an empty index returns nothing");
  const singleIndex = buildIndex([sample]);
  const single = search(singleIndex, sample, { radius: 2, topK: 5 });
  assert(single.results.length === 1 && single.results[0].distance === 0, "a single-vector index returns it");
  assert(search(index, queries[0], { radius: 2, topK: 5000 }).results.length === search(index, queries[0], { radius: 2, topK: 5000 }).candidates.length, "topK is capped by the candidate count");

  // 10. Scaling: with the same radius the candidate fraction stays small as the dataset grows.
  const bigger = buildIndex(Array.from({ length: 4000 }, () => randomVector(rng)));
  const bigRun = search(bigger, flipBits(bigger.vectors[9], 2, rng), { radius: 2, topK: 3 });
  assert(bigRun.candidates.length / 4000 < 0.1, `the candidate fraction stays small at scale, got ${bigRun.candidates.length / 4000}`);
  assert(bigRun.results[0].distance <= 2, "the near neighbour survives the rerank");

  console.log(`multi-index: n=${dataset.length} recall@1=${recall} candidates=${(candidateFraction * 100).toFixed(2)}% radius1=${small.candidates.length} radius2=${large.candidates.length}`);
  console.log("PASS");
}
module.exports = { splitmix64, randomVector, flipBits, hamming, hammingSlow, buildIndex, candidatesFor, search, bruteForce, combinations, BLOCKS, BLOCK_BITS };
if (require.main === module) main();
