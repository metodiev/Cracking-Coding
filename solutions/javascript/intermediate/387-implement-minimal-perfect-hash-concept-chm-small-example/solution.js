/** Task 387 - Implement minimal perfect hash concept (CHM) small example. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const HASH_MOD = 2147483647;           // 2^31 - 1; products stay below 2^53, so Number is exact
const BASE1 = 131;
const BASE2 = 1000003;

const KEYS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan",
  "judy", "karl", "laura"];
const MORE_KEYS = ["kiwi", "mango", "melon", "olive", "peach", "plum", "quince", "raspberry"];

function polyHash(text, base) {
  let value = 0;
  for (let i = 0; i < text.length; i++) {
    value = (value * base + text.charCodeAt(i)) % HASH_MOD;
  }
  return value;
}

/**
 * Hash-displace build: buckets by h1 mod n; each bucket gets the smallest shift whose
 * sigmas (h2 + shift) mod n are distinct and unused; final slots are consecutive in
 * increasing sigma order. Retries the whole build with base2 + attempt on failure.
 * Returns { n, table, g, displacement, base2 } with -1 sentinels where applicable.
 */
function buildMphf(keys) {
  const n = keys.length;
  for (let attempt = 0; attempt < 100; attempt++) {
    const base2 = BASE2 + attempt;
    const buckets = Array.from({ length: n }, () => []);
    for (const key of keys) buckets[polyHash(key, BASE1) % n].push(key);
    const order = [...Array(n).keys()]
      .sort((x, y) => buckets[y].length - buckets[x].length || x - y);
    const table = new Array(n).fill(null);
    const g = new Array(n).fill(-1);
    const displacement = new Array(n).fill(-1);
    let nextSlot = 0;
    let feasible = true;
    for (const bucket of order) {
      const members = buckets[bucket];
      if (!members.length) continue;
      let chosen = -1;
      let sigmas = [];
      for (let shift = 0; shift < n && chosen < 0; shift++) {
        const used = new Array(n).fill(false);
        let good = true;
        const candidate = [];
        for (const key of members) {
          const sigma = (polyHash(key, base2) + shift) % n;
          if (used[sigma] || g[sigma] !== -1) { good = false; break; }
          used[sigma] = true;
          candidate.push(sigma);
        }
        if (good) {
          chosen = shift;
          sigmas = candidate;
        }
      }
      if (chosen < 0) {
        feasible = false;
        break;
      }
      displacement[bucket] = chosen;
      const paired = members.map((key, i) => [sigmas[i], key])
        .sort((x, y) => x[0] - y[0] || (x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0));
      paired.forEach(([sigma, key], rank) => {
        g[sigma] = nextSlot + rank;
        table[nextSlot + rank] = key;
      });
      nextSlot += members.length;
    }
    if (feasible) return { n, table, g, displacement, base2 };
  }
  throw new Error("no displacement assignment found");
}

/** Slot of a member key in [0, n); -1 when the first-level bucket stayed empty. */
function mphfLookup(mphf, key) {
  const bucket = polyHash(key, BASE1) % mphf.n;
  const shift = mphf.displacement[bucket];
  if (shift < 0) return -1;
  const sigma = (polyHash(key, mphf.base2) + shift) % mphf.n;
  if (mphf.g[sigma] < 0) return -1;
  return mphf.g[sigma];
}

module.exports = { HASH_MOD, BASE1, BASE2, KEYS, MORE_KEYS, polyHash, buildMphf, mphfLookup };

if (require.main === module) {
  const assert = require("node:assert");

  const n = KEYS.length;
  assert.strictEqual(n, 12);
  assert.strictEqual(new Set(KEYS).size, n);

  // deterministic hash sanity (values pin the polynomial hash)
  assert.strictEqual(polyHash("alice", BASE1), 893813729);
  assert.strictEqual(polyHash("alice", BASE2), 76280623);
  const bucketOf = (key) => polyHash(key, BASE1) % n;
  assert.strictEqual(bucketOf("alice"), 5);
  assert.strictEqual(bucketOf("bob"), 1);
  assert.strictEqual(bucketOf("laura"), 1);
  assert.strictEqual(bucketOf("frank"), 8);
  assert.strictEqual(bucketOf("judy"), 8);
  assert.strictEqual(bucketOf("karl"), 8);
  let largest = 0;
  for (let b = 0; b < n; b++) {
    largest = Math.max(largest, KEYS.filter((key) => bucketOf(key) === b).length);
  }
  assert.strictEqual(largest, 3);

  const mphf = buildMphf(KEYS);
  assert.strictEqual(mphf.base2, BASE2, "first attempt already succeeds");
  assert.deepStrictEqual(mphf.table,
    ["frank", "judy", "karl", "laura", "bob", "ivan", "grace", "heidi", "dave", "alice",
      "erin", "carol"]);
  assert.deepStrictEqual(mphf.g, [7, 0, 5, 9, 10, 11, 1, 2, 3, 4, 6, 8]);
  assert.deepStrictEqual(mphf.displacement, [-1, 1, 7, 0, 5, 8, 5, -1, 0, -1, 6, -1]);

  // minimal perfect hash: every key maps to its own slot in [0, n), no collisions
  const slots = new Set();
  for (const key of KEYS) {
    const slot = mphfLookup(mphf, key);
    assert.ok(slot >= 0 && slot < n, `slot in range ${key}`);
    assert.strictEqual(mphf.table[slot], key, `table[slot] = ${key}`);
    assert.ok(!slots.has(slot), `slot not reused ${key}`);
    slots.add(slot);
  }
  assert.strictEqual(slots.size, n);

  // determinism: rebuilding yields the identical structure
  const again = buildMphf(KEYS);
  assert.deepStrictEqual(again.table, mphf.table);
  assert.deepStrictEqual(again.g, mphf.g);
  assert.deepStrictEqual(again.displacement, mphf.displacement);
  assert.strictEqual(again.base2, mphf.base2);

  // non-member keys are out of scope: any in-range answer is meaningless by construction
  const stranger = mphfLookup(mphf, "stranger");
  assert.ok(stranger === -1 || (stranger >= 0 && stranger < n));

  // a second key set needs one hash retry (documented) and still builds a bijection
  const second = buildMphf(MORE_KEYS);
  assert.strictEqual(second.base2, BASE2 + 2, "attempts 0 and 1 fail, attempt 2 succeeds");
  assert.deepStrictEqual(second.table,
    ["plum", "kiwi", "peach", "melon", "olive", "mango", "raspberry", "quince"]);
  assert.deepStrictEqual(second.displacement, [0, 0, 0, -1, 1, 4, -1, -1]);
  const slots3 = new Set();
  for (const key of MORE_KEYS) {
    const slot = mphfLookup(second, key);
    assert.ok(slot >= 0 && slot < MORE_KEYS.length, `second slot in range ${key}`);
    assert.strictEqual(second.table[slot], key, `second table[slot] = ${key}`);
    assert.ok(!slots3.has(slot), `second slot not reused ${key}`);
    slots3.add(slot);
  }
  assert.strictEqual(slots3.size, MORE_KEYS.length);

  console.log("PASS");
}
