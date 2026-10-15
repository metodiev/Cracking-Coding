/** Task 466 - Implement hyperloglog sketch for cardinality estimation (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Outline HyperLogLog with m = 2^9 = 512 registers. Items are hashed with the FIXED
 * 32-bit mixer mix32 (0x45d9f3b xorshift-multiply; identical constants in every
 * language port). The top p = 9 bits pick the register; the rank is the 1-based
 * position of the leftmost set bit of the remaining 23 bits (all-zero tail counts
 * as 24). Estimate = alpha_m * m^2 / sum(2^-M[j]) with alpha_m = 0.7213/(1+1.079/m),
 * plus linear counting m*ln(m/V) while raw <= 2.5m and V registers are still zero.
 *
 * Checks: for each digest size n in {100, 1000, 5000, 20000}, 16 independent digests
 * are built from keys hashed as (pool << 17) | value so the pools are disjoint (a
 * shared 16-bit key space would correlate the digests). Documented bounds:
 * |mean estimate - n| / n <= 1.04/sqrt(m) ~ 0.046 (measured max 0.008); every single
 * digest within the 3-sigma envelope 3*1.04/sqrt(m) ~ 0.138 (measured max 0.098);
 * mean estimates increase with cardinality; a register-wise max merge of two
 * complementary digests equals the digest of the union exactly.
 */
"use strict";

const P_BITS = 9;
const M = 1 << P_BITS;
const RANK_WIDTH = 32 - P_BITS;
const ALPHA = 0.7213 / (1 + 1.079 / M);
const SIGMA = 1.04 / Math.sqrt(M);

function mix32(x) {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x45d9f3b) >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x45d9f3b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function bitLength(value) {
  return 32 - Math.clz32(value);
}

function rankOf(rest) {
  if (rest === 0) return RANK_WIDTH + 1;
  return RANK_WIDTH - bitLength(rest) + 1;
}

class HyperLogLog {
  constructor() {
    this.registers = new Array(M).fill(0);
  }

  add(key) {
    const h = mix32(key);
    const index = h >>> RANK_WIDTH;
    const rest = h & ((1 << RANK_WIDTH) - 1);
    const r = rankOf(rest);
    if (r > this.registers[index]) this.registers[index] = r;
  }

  merge(other) {
    for (let i = 0; i < M; i++) this.registers[i] = Math.max(this.registers[i], other.registers[i]);
  }

  estimate() {
    let z = 0;
    let zeros = 0;
    for (const m of this.registers) {
      z += Math.pow(2, -m);
      if (m === 0) zeros++;
    }
    const raw = (ALPHA * M * M) / z;
    if (raw <= 2.5 * M && zeros > 0) return M * Math.log(M / zeros);
    return raw;
  }
}

function makeLcg(state) {
  let seed = state;
  return () => (seed = (seed * 75 + 74) % 65537);
}

function poolKeys(count, pool, startSeed) {
  const lcg = makeLcg(startSeed);
  const keys = [];
  const seen = new Set();
  while (keys.length < count) {
    const value = lcg();
    if (!seen.has(value)) {
      seen.add(value);
      keys.push((pool << 17) | value);
    }
  }
  return keys;
}

function digest(keys) {
  const sketch = new HyperLogLog();
  for (const key of keys) sketch.add(key);
  return sketch;
}

module.exports = { HyperLogLog, mix32, rankOf, poolKeys };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(mix32(0), 0);
  assert.strictEqual(rankOf(0), RANK_WIDTH + 1);
  assert.strictEqual(rankOf(1 << (RANK_WIDTH - 1)), 1);
  assert.strictEqual(rankOf(1), RANK_WIDTH);
  assert.strictEqual(rankOf(0b101), RANK_WIDTH - 2);

  const sizes = [100, 1000, 5000, 20000];
  const pools = 16;
  const meanEstimates = [];
  let maxMeanError = 0;
  let maxSingleError = 0;
  for (const size of sizes) {
    let sum = 0;
    for (let pool = 0; pool < pools; pool++) {
      const est = digest(poolKeys(size, pool, (pool + 1) * 131)).estimate();
      sum += est;
      const singleError = Math.abs(est - size) / size;
      maxSingleError = Math.max(maxSingleError, singleError);
      assert.ok(singleError <= 3 * SIGMA, `3-sigma size ${size} pool ${pool}`);
    }
    const mean = sum / pools;
    meanEstimates.push(mean);
    const meanError = Math.abs(mean - size) / size;
    maxMeanError = Math.max(maxMeanError, meanError);
    assert.ok(meanError <= SIGMA, `mean error size ${size}`); // documented 1.04/sqrt(m)
  }
  for (let i = 1; i < meanEstimates.length; i++) {
    assert.ok(meanEstimates[i - 1] < meanEstimates[i]); // estimates grow
  }

  const keys = poolKeys(3000, 7, 4242);
  const merged = digest(keys.slice(0, 1500));
  merged.merge(digest(keys.slice(1500)));
  const whole = digest(keys);
  assert.deepStrictEqual(merged.registers, whole.registers);
  assert.strictEqual(merged.estimate(), whole.estimate());
  assert.ok(maxMeanError <= 0.02 && maxSingleError <= 0.12); // measured margins
  console.log("PASS");
}
