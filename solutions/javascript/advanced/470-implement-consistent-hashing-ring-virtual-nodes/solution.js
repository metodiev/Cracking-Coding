/** Task 470 - Implement consistent hashing ring (virtual nodes). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Consistent-hashing ring with VNODES = 64 virtual nodes per physical node. Ring
 * positions come from the FIXED LCG hash (documented, deterministic): fold over
 * bytes with h = (h*75 + 74 + ch) mod 65537, then two LCG avalanche steps
 * h = (h*75 + 74) mod 65537 (the plain fold alone clusters badly - measured shares
 * 0.02..0.48; the avalanched hash balances). Points are sorted by (position, node
 * label); a key hashes with the same function and is owned by the first vnode
 * clockwise at or after its position (wrap-around).
 *
 * Checks on 4 physical nodes x 64 vnodes and 4096 keys:
 *   determinism (identical rebuild), bisect lookup equals a brute-force clockwise
 *   scan on a key sample, balance within the documented [0.18, 0.32] band (measured
 *   0.220..0.270), adding "node-e" moves exactly 716 keys (0.175 <= 0.30 documented)
 *   with every moved key going to "node-e", and removing "node-b" moves exactly its
 *   1034 keys (0.252 <= 0.30) with no other key changing owner.
 */
"use strict";

const VNODES = 64;
const MODULUS = 65537;

function lcgHash(text) {
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 75 + 74 + text.charCodeAt(i)) % MODULUS;
  h = (h * 75 + 74) % MODULUS;
  h = (h * 75 + 74) % MODULUS;
  return h;
}

function comparePoints(a, b) {
  if (a[0] !== b[0]) return a[0] - b[0];
  if (a[1] !== b[1]) return a[1] < b[1] ? -1 : 1;
  return a[2] - b[2];
}

class ConsistentHashRing {
  constructor(nodes) {
    this.points = [];
    for (const node of nodes) {
      for (let i = 0; i < VNODES; i++) this.points.push([lcgHash(`${node}#${i}`), node, i]);
    }
    this.points.sort(comparePoints);
  }

  distinctPositions() {
    return new Set(this.points.map((p) => p[0])).size;
  }

  owner(key) {
    const pos = lcgHash(key);
    let lo = 0;
    let hi = this.points.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.points[mid][0] < pos) lo = mid + 1;
      else hi = mid;
    }
    if (lo === this.points.length) lo = 0; // wrap around
    return this.points[lo][1];
  }

  ownerBrute(key) {
    const pos = lcgHash(key);
    let best = null;
    for (const point of this.points) {
      if (point[0] >= pos && (best === null || comparePoints(point, best) < 0)) best = point;
    }
    if (best === null) best = this.points[0];
    return best[1];
  }
}

function assign(ring, keys) {
  const owners = new Map();
  for (const key of keys) owners.set(key, ring.owner(key));
  return owners;
}

module.exports = { lcgHash, ConsistentHashRing, assign };

if (require.main === module) {
  const assert = require("node:assert");
  const nodes = ["node-a", "node-b", "node-c", "node-d"];
  const keys = Array.from({ length: 4096 }, (_, i) => `k${i}`);

  const ring = new ConsistentHashRing(nodes);
  const first = assign(ring, keys);
  const again = assign(new ConsistentHashRing(nodes), keys);
  assert.strictEqual(again.size, first.size);
  for (const [key, owner] of first) assert.strictEqual(again.get(key), owner); // deterministic
  assert.ok(ring.distinctPositions() > (nodes.length * VNODES) / 2);

  // Bisect lookup equals the brute-force clockwise scan on a fixed sample.
  for (let i = 0; i < keys.length; i += 79) {
    assert.strictEqual(ring.owner(keys[i]), ring.ownerBrute(keys[i]), `brute ${i}`);
  }

  // Balance within the documented band.
  const shares = {};
  for (const node of nodes) shares[node] = 0;
  for (const owner of first.values()) shares[owner]++;
  for (const node of nodes) {
    const share = shares[node] / keys.length;
    assert.ok(share >= 0.18 && share <= 0.32, `share ${node} ${share}`);
  }

  // Adding a node: bounded movement, all movers go to the new node.
  const afterAdd = assign(new ConsistentHashRing([...nodes, "node-e"]), keys);
  const moved = keys.filter((key) => afterAdd.get(key) !== first.get(key));
  assert.ok(moved.length <= 0.3 * keys.length, `add movement ${moved.length}`);
  assert.ok(moved.every((key) => afterAdd.get(key) === "node-e"));

  // Removing a node: only that node's keys move.
  const afterRemove = assign(new ConsistentHashRing(nodes.filter((n) => n !== "node-b")), keys);
  const movedRemove = keys.filter((key) => afterRemove.get(key) !== first.get(key));
  assert.ok(movedRemove.length <= 0.3 * keys.length, `remove movement ${movedRemove.length}`);
  for (const key of keys) {
    if (first.get(key) !== "node-b") assert.strictEqual(afterRemove.get(key), first.get(key), `untouched ${key}`);
  }
  assert.ok(movedRemove.every((key) => first.get(key) === "node-b"));

  assert.strictEqual(moved.length, 716);
  assert.strictEqual(movedRemove.length, 1034);
  console.log("PASS");
}
