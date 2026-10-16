/**
 * Task 683 - Implement compacted trie (Patricia trie) for IP routing table simulation.
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * A path-compressed binary trie over 32-bit IPv4 addresses. Every node stores a real prefix
 * (address masked to its length) and, when a route ends there, a next-hop value; a node's children are
 * reached through the bit just after its own prefix, so chains of single-child nodes are collapsed away.
 * Insertion either replaces the value of an existing prefix, splits a node when the new prefix diverges
 * from it (an intermediate node is created at the first differing bit), or inserts the new prefix as an
 * ancestor when it is shorter than the node it meets. Lookup walks down while the node's stored prefix
 * still matches the address, remembering the last node with a value, which is the longest prefix match.
 * Removal drops the value and splices a node that is left with no value and at most one child.
 *
 * Verified against a brute-force linear scan over the routing table (the independent reference): every
 * lookup agrees for a table with a default route, nested /8, /16, /24 and /32 host routes, and for a
 * table of 30 random prefixes probed with 200 random addresses; the compression invariants hold (each
 * node's prefix is masked to its length, children extend their parent and have longer prefixes, and the
 * node count stays below 2 * prefixes); a duplicate insert replaces instead of growing the table; the
 * longest match always wins (/32 over /24 over /0); deleting every prefix keeps the lookups equal to the
 * brute force after each deletion while the node count shrinks; and missing prefixes, an empty table and
 * the reproducibility of the whole structure are covered as boundary cases.
 */
"use strict";
const assert = require("node:assert");
const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed683n;
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }
  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}
// ---------------------------------------------------------------------------------- addresses
const ip = (a, b, c, d) => ((a * 0x1000000 + b * 0x10000 + c * 0x100 + d) >>> 0);
const mask = (len) => (len === 0 ? 0 : (0xffffffff << (32 - len)) >>> 0);
const bitAt = (addr, bit) => (addr >>> (31 - bit)) & 1;
const format = (addr) => [24, 16, 8, 0].map((shift) => (addr >>> shift) & 0xff).join(".");
/** Number of leading bits the two prefixes agree on. */
function commonBits(a, aLen, b, bLen) {
  const limit = Math.min(aLen, bLen);
  let len = 0;
  while (len < limit && bitAt(a, len) === bitAt(b, len)) len += 1;
  return len;
}
const makeTrie = () => ({ root: null, prefixes: 0 });
function attach(trie, parent, side, node) {
  if (parent === null) trie.root = node;
  else if (side === 1) parent.right = node;
  else parent.left = node;
}
function insert(trie, addr, len, value) {
  const target = (addr & mask(len)) >>> 0;
  const makeLeaf = () => ({ prefix: target, len, value, left: null, right: null });
  if (trie.root === null) {
    trie.root = makeLeaf();
    trie.prefixes += 1;
    return { inserted: true, replaced: false };
  }
  let node = trie.root;
  let parent = null;
  let side = 0;
  for (;;) {
    const common = commonBits(target, len, node.prefix, node.len);
    if (common === len && len === node.len) {
      const had = node.value !== null;
      node.value = value;
      if (!had) trie.prefixes += 1;
      return { inserted: !had, replaced: had };
    }
    if (common < Math.min(len, node.len)) {
      // The paths diverge: create an intermediate node at the first differing bit.
      const middle = { prefix: (target & mask(common)) >>> 0, len: common, value: null, left: null, right: null };
      const leaf = makeLeaf();
      if (bitAt(target, common) === 1) {
        middle.left = node;
        middle.right = leaf;
      } else {
        middle.left = leaf;
        middle.right = node;
      }
      attach(trie, parent, side, middle);
      trie.prefixes += 1;
      return { inserted: true, replaced: false };
    }
    if (len < node.len) {
      // The new prefix is a strict ancestor of this node: insert it above.
      const ancestor = makeLeaf();
      if (bitAt(node.prefix, len) === 1) ancestor.right = node;
      else ancestor.left = node;
      attach(trie, parent, side, ancestor);
      trie.prefixes += 1;
      return { inserted: true, replaced: false };
    }
    const bit = bitAt(target, node.len);
    const child = bit === 1 ? node.right : node.left;
    if (child === null) {
      const leaf = makeLeaf();
      if (bit === 1) node.right = leaf;
      else node.left = leaf;
      trie.prefixes += 1;
      return { inserted: true, replaced: false };
    }
    parent = node;
    side = bit;
    node = child;
  }
}
/** Longest prefix match: descend while the compressed path still matches the address. */
function lookup(trie, addr) {
  const address = addr >>> 0;
  let node = trie.root;
  let best = null;
  while (node !== null) {
    if (node.prefix !== ((address & mask(node.len)) >>> 0)) break; // the skipped bits disagree
    if (node.value !== null) best = { len: node.len, value: node.value };
    if (node.len >= 32) break;
    node = bitAt(address, node.len) === 1 ? node.right : node.left;
  }
  return best;
}
function remove(trie, addr, len) {
  const target = (addr & mask(len)) >>> 0;
  const path = [];
  let node = trie.root;
  let parent = null;
  let side = 0;
  while (node !== null) {
    if (node.len === len) break;
    if (node.len > len) {
      node = null; // a longer prefix cannot contain this one
      break;
    }
    const bit = bitAt(target, node.len);
    path.push({ node, side: bit });
    parent = node;
    side = bit;
    node = bit === 1 ? node.right : node.left;
  }
  if (node === null || node.prefix !== target || node.len !== len || node.value === null) {
    return { removed: false };
  }
  node.value = null;
  trie.prefixes -= 1;
  if (node.left === null || node.right === null) {
    attach(trie, parent, side, node.left !== null ? node.left : node.right);
  }
  // A node left with no value and one child is redundant: splice it and keep walking up.
  for (let i = path.length - 1; i >= 0; i--) {
    const current = path[i].node;
    if (current.value !== null || (current.left !== null && current.right !== null)) continue;
    const above = i === 0 ? null : path[i - 1].node;
    const aboveSide = i === 0 ? 0 : path[i - 1].side;
    attach(trie, above, aboveSide, current.left !== null ? current.left : current.right);
  }
  return { removed: true };
}
/** Independent reference: the longest matching prefix by linear scan. */
function bruteForceLpm(table, addr) {
  let best = null;
  for (const entry of table) {
    if ((addr & mask(entry.len)) >>> 0 !== entry.prefix) continue;
    if (best === null || entry.len > best.len) best = { len: entry.len, value: entry.value };
  }
  return best === null ? null : { len: best.len, value: best.value };
}
const tableEntries = (table) => table.map((entry) => ({ prefix: (entry.addr & mask(entry.len)) >>> 0, len: entry.len, value: entry.value }));
/** Walk the trie and check the compression invariants. */
function trieProblems(trie) {
  const bad = [];
  let nodes = 0;
  const walk = (node, parent) => {
    nodes += 1;
    if (node.prefix !== ((node.prefix & mask(node.len)) >>> 0)) bad.push(`${format(node.prefix)} is not masked to /${node.len}`);
    if (node.len < 0 || node.len > 32) bad.push(`bad length ${node.len}`);
    if (parent !== null) {
      if (!(parent.len < node.len)) bad.push(`child /${node.len} does not extend parent /${parent.len}`);
      if (node.prefix !== ((parent.prefix | (node.prefix & ~mask(parent.len))) >>> 0) && parent.len > 0) {
        if ((node.prefix & mask(parent.len)) >>> 0 !== parent.prefix) bad.push(`child of ${format(parent.prefix)} does not nest`);
      }
    }
    let children = 0;
    for (const side of ["left", "right"]) {
      const child = node[side];
      if (child === null) continue;
      children += 1;
      if (bitAt(child.prefix, node.len) !== (side === "right" ? 1 : 0)) bad.push(`child on the wrong side of /${node.len}`);
      walk(child, node);
    }
    if (children === 0 && node.value === null) bad.push(`a leaf without a value at /${node.len}`);
    if (children === 1 && node.value === null && node.len > 0) bad.push(`uncompressed single-child node at /${node.len}`);
  };
  if (trie.root !== null) walk(trie.root, null);
  return { bad, nodes };
}
// ------------------------------------------------------------------------------------ test data
function routingTable() {
  return [
    { addr: ip(0, 0, 0, 0), len: 0, value: "default" },
    { addr: ip(10, 0, 0, 0), len: 8, value: "ten" },
    { addr: ip(10, 1, 0, 0), len: 16, value: "ten-one" },
    { addr: ip(10, 1, 2, 0), len: 24, value: "ten-one-two" },
    { addr: ip(10, 1, 2, 3), len: 32, value: "host" },
    { addr: ip(10, 1, 3, 0), len: 24, value: "sibling" },
    { addr: ip(192, 168, 0, 0), len: 16, value: "lan" },
    { addr: ip(192, 168, 1, 0), len: 24, value: "lan-one" },
    { addr: ip(172, 16, 0, 0), len: 12, value: "private" },
  ];
}
function probeAddresses() {
  const base = [
    ip(0, 0, 0, 0), ip(10, 0, 0, 0), ip(10, 1, 0, 0), ip(10, 1, 2, 0), ip(10, 1, 2, 3), ip(10, 1, 2, 4),
    ip(10, 1, 3, 0), ip(10, 1, 255, 255), ip(192, 168, 0, 0), ip(192, 168, 1, 0), ip(192, 168, 1, 255),
    ip(172, 16, 0, 0), ip(172, 31, 255, 255), ip(172, 32, 0, 0), ip(255, 255, 255, 255), ip(8, 8, 8, 8),
  ];
  const neighbours = [];
  for (const addr of base) {
    neighbours.push((addr + 1) >>> 0, (addr - 1) >>> 0);
  }
  return [...base, ...neighbours];
}
function main() {
  // (1) Building the table and checking every probe against the brute-force scan.
  const trie = makeTrie();
  const table = routingTable();
  for (const entry of table) {
    const result = insert(trie, entry.addr, entry.len, entry.value);
    assert.strictEqual(result.inserted, true, `${format(entry.addr)}/${entry.len} is new`);
  }
  const reference = tableEntries(table);
  assert.strictEqual(trie.prefixes, table.length, "the trie holds one value per route");
  let probes = 0;
  for (const addr of probeAddresses()) {
    probes += 1;
    assert.deepStrictEqual(lookup(trie, addr), bruteForceLpm(reference, addr), `lookup of ${format(addr)} must match the linear scan`);
  }
  assert.ok(probes > 40, `${probes} addresses were cross-checked`);
  const structure = trieProblems(trie);
  assert.deepStrictEqual(structure.bad, [], "the compression invariants hold");
  assert.ok(structure.nodes < 2 * table.length, `documented: ${structure.nodes} nodes for ${table.length} prefixes, i.e. the trie is compressed`);

  // (2) Longest prefix wins, and each route answers for its own network address.
  assert.deepStrictEqual([lookup(trie, ip(10, 1, 2, 3)).value, lookup(trie, ip(10, 1, 2, 3)).len], ["host", 32], "a /32 host route beats every covering prefix");
  assert.deepStrictEqual([lookup(trie, ip(10, 1, 2, 9)).value, lookup(trie, ip(10, 1, 2, 9)).len], ["ten-one-two", 24], "a neighbour of the host route falls back to the /24");
  assert.deepStrictEqual([lookup(trie, ip(10, 1, 9, 9)).value, lookup(trie, ip(10, 1, 9, 9)).len], ["ten-one", 16], "and a further neighbour to the /16");
  assert.deepStrictEqual([lookup(trie, ip(10, 9, 9, 9)).value, lookup(trie, ip(10, 9, 9, 9)).len], ["ten", 8], "while 10.9.9.9 only matches the /8");
  assert.deepStrictEqual([lookup(trie, ip(4, 4, 4, 4)).value, lookup(trie, ip(4, 4, 4, 4)).len], ["default", 0], "everything else falls through to the default route");
  for (const entry of table) {
    const found = lookup(trie, entry.addr);
    assert.ok(found !== null && found.len >= entry.len && found.value !== null, `documented: ${format(entry.addr)}/${entry.len} answers for its own network address`);
  }

  // (3) Duplicate inserts replace instead of growing the trie.
  const before = trie.prefixes;
  const replacement = insert(trie, ip(10, 1, 2, 0), 24, "ten-one-two-v2");
  assert.deepStrictEqual([replacement.inserted, replacement.replaced, trie.prefixes - before], [false, true, 0], "an existing prefix is replaced in place");
  assert.strictEqual(lookup(trie, ip(10, 1, 2, 200)).value, "ten-one-two-v2", "and the new value is live");
  assert.deepStrictEqual(insert(trie, ip(10, 1, 2, 0), 24, "ten-one-two").replaced, true, "putting the old value back also replaces");
  const updated = trieProblems(trie);
  assert.ok(updated.nodes <= structure.nodes, "replacement never adds nodes");

  // (4) Random tables: 30 prefixes probed with 200 random addresses.
  const rng = new SplitMix64(DEFAULT_SEED);
  const randomTrie = makeTrie();
  const randomTable = [];
  for (let i = 0; i < 30; i++) {
    const len = [8, 12, 16, 20, 24, 28, 32][rng.nextInt(7)];
    const addr = (rng.nextInt(0x100000000) & mask(len)) >>> 0;
    const value = `r${i}`;
    insert(randomTrie, addr, len, value);
    const existing = randomTable.findIndex((entry) => entry.len === len && entry.prefix === addr);
    if (existing >= 0) randomTable[existing] = { prefix: addr, len, value };
    else randomTable.push({ prefix: addr, len, value });
  }
  assert.strictEqual(randomTrie.prefixes, randomTable.length, "the random trie has no duplicate prefixes");
  let randomProbes = 0;
  for (let i = 0; i < 200; i++) {
    const addr = rng.nextInt(0x100000000);
    randomProbes += 1;
    assert.deepStrictEqual(lookup(randomTrie, addr), bruteForceLpm(randomTable, addr), `random probe ${format(addr)} matches the linear scan`);
  }
  assert.strictEqual(randomProbes, 200, "two hundred random probes were compared");
  assert.deepStrictEqual(trieProblems(randomTrie).bad, [], "the random trie is compressed and consistent too");
  assert.ok(trieProblems(randomTrie).nodes < 2 * randomTable.length, "and its node count stays compact");

  // (5) Deletion: the lookups follow the brute force after every single removal.
  const shrinking = makeTrie();
  const shrinkTable = [];
  for (const entry of routingTable()) {
    insert(shrinking, entry.addr, entry.len, entry.value);
    shrinkTable.push({ prefix: (entry.addr & mask(entry.len)) >>> 0, len: entry.len, value: entry.value });
  }
  const deletionOrder = [...routingTable()].reverse();
  let deletions = 0;
  for (const entry of deletionOrder) {
    const result = remove(shrinking, entry.addr, entry.len);
    assert.strictEqual(result.removed, true, `${format(entry.addr)}/${entry.len} can be deleted`);
    deletions += 1;
    const index = shrinkTable.findIndex((item) => item.len === entry.len && item.prefix === ((entry.addr & mask(entry.len)) >>> 0));
    shrinkTable.splice(index, 1);
    assert.strictEqual(shrinking.prefixes, shrinkTable.length, "the prefix count follows the deletions");
    for (const addr of probeAddresses()) {
      assert.deepStrictEqual(lookup(shrinking, addr), bruteForceLpm(shrinkTable, addr), `after deleting ${format(entry.addr)}/${entry.len}, ${format(addr)} must still match`);
    }
    assert.deepStrictEqual(trieProblems(shrinking).bad, [], "and the trie stays consistent while shrinking");
  }
  assert.strictEqual(deletions, routingTable().length, "every route was deleted once");
  assert.deepStrictEqual([shrinking.prefixes, shrinking.root], [0, null], "the empty trie has no root left");

  // (6) Boundary cases: an empty trie, a missing prefix and unsorted insertion orders.
  const empty = makeTrie();
  assert.deepStrictEqual(lookup(empty, ip(1, 2, 3, 4)), null, "an empty routing table matches nothing");
  assert.deepStrictEqual(remove(empty, ip(1, 2, 3, 4), 32).removed, false, "deleting from an empty trie does nothing");
  assert.deepStrictEqual(remove(trie, ip(203, 0, 113, 0), 24).removed, false, "deleting a prefix that was never added fails");
  const reversed = makeTrie();
  for (const entry of [...routingTable()].reverse()) insert(reversed, entry.addr, entry.len, entry.value);
  assert.strictEqual(lookup(reversed, ip(10, 1, 2, 3)).len, 32, "insertion order does not change the longest match");
  assert.strictEqual(lookup(reversed, ip(10, 1, 2, 3)).value, "host", "nor the value");
  assert.deepStrictEqual(lookup(makeTrie(), ip(0, 0, 0, 0)), null, "a /0 route must be inserted explicitly");
  const onlyDefault = makeTrie();
  insert(onlyDefault, 0, 0, "default");
  assert.deepStrictEqual([lookup(onlyDefault, ip(9, 9, 9, 9)).value, trieProblems(onlyDefault).bad], ["default", []], "a table with only the default route matches everything");

  // (7) Determinism of the whole structure.
  const first = makeTrie();
  const second = makeTrie();
  for (const entry of routingTable()) {
    insert(first, entry.addr, entry.len, entry.value);
    insert(second, entry.addr, entry.len, entry.value);
  }
  assert.deepStrictEqual(JSON.stringify(first), JSON.stringify(second), "identical insertions build identical tries");
  assert.strictEqual(lookup(first, ip(172, 16, 0, 1)).value, "private", "the /12 route is found");
  assert.strictEqual(lookup(first, ip(172, 15, 0, 1)).value, "default", "and 172.15.0.1 falls back to the default");
}

module.exports = { SplitMix64, DEFAULT_SEED, ip, mask, bitAt, format, commonBits, makeTrie, attach,
  insert, lookup, remove, bruteForceLpm, tableEntries, trieProblems, routingTable, probeAddresses };
if (require.main === module) {
  main();
  console.log("PASS");
}
