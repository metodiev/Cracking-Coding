/** Task 684 - Implement longest-prefix-match lookup optimized (binary trie). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: an IPv4 forwarding table (FIB) stored in a *binary trie*. One node per bit of prefix,
 * most significant bit first, so the prefix /p of network N is exactly the path formed by the p
 * leading bits of N. Every node carries
 *     value       - the next hop of a prefix that *ends* at this node (null if none),
 *     prefixLen   - the length of that prefix (-1 if none),
 *     bestDepth   - the longest prefix length anywhere in this node's subtree (-1 if the whole
 *                   subtree holds no prefix).
 * Lookup walks the address bits, remembering the last node that carried a value. The
 * optimization over a plain bit walk is `bestDepth`: as soon as the current node's subtree holds
 * no prefix at all the walk stops, and a `lastVisitedNodes` counter records how many nodes were
 * actually touched (the naive walk would always touch up to 32). Deletion recomputes `bestDepth`
 * bottom-up and unlinks childless leaves, so the trie stays canonical (no empty subtrees).
 *
 * Verified here: strict parser round-trips, hand-derived LPM decisions (including /0, /32),
 * exhaustive agreement with a brute-force linear scan of the prefix list on deterministically
 * generated tables, the longest-match invariant for every inserted prefix, node pruning and
 * node counts after insertion/deletion, and in-order enumeration of the table.
 */
"use strict";

const ADDRESS_BITS = 32;
const ALL_ONES = 0xffffffff;

/** Strict dotted-quad parser. Rejects wrong arity, non-digits and leading zeros. */
function parseIPv4(text) {
  const parts = String(text).split(".");
  if (parts.length !== 4) throw new RangeError(`not an IPv4 address: ${text}`);
  let value = 0;
  for (const part of parts) {
    if (!/^(0|[1-9]\d{0,2})$/.test(part)) throw new RangeError(`bad octet "${part}" in ${text}`);
    const octet = Number(part);
    if (octet > 255) throw new RangeError(`octet out of range in ${text}`);
    value = ((value << 8) | octet) >>> 0;
  }
  return value >>> 0;
}

function formatIPv4(value) {
  const v = value >>> 0;
  return [(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff].join(".");
}

/** Prefix mask of length `len`: `len` leading ones followed by 32-len zeros. */
function prefixMask(len) {
  if (!Number.isInteger(len) || len < 0 || len > ADDRESS_BITS) {
    throw new RangeError(`prefix length out of range: ${len}`);
  }
  return len === 0 ? 0 : (ALL_ONES << (ADDRESS_BITS - len)) >>> 0;
}

/** "10.1.0.0/16" -> { network, prefixLen } with the host bits cleared. */
function parseCIDR(text) {
  const [addr, lenText] = String(text).split("/");
  const prefixLen = Number(lenText);
  const mask = prefixMask(prefixLen);
  return { network: (parseIPv4(addr) & mask) >>> 0, prefixLen };
}

function makeNode() {
  return { child: [null, null], value: null, prefixLen: -1, prefix: 0, bestDepth: -1 };
}

/** bestDepth = max(own prefix length, bestDepth of both children). */
function recompute(node) {
  let best = node.prefixLen;
  for (const child of node.child) {
    if (child !== null && child.bestDepth > best) best = child.bestDepth;
  }
  node.bestDepth = best;
}

class PrefixTrie {
  constructor() {
    this.root = makeNode();
    this.size = 0;                 // number of stored prefixes
    this.nodeCount = 1;            // nodes including the root
    this.lastVisitedNodes = 0;     // diagnostics for the pruning optimization
  }

  /** Insert or replace the next hop for `network/prefixLen` (host bits are masked off). */
  insert(network, prefixLen, value) {
    const mask = prefixMask(prefixLen);
    const net = (network & mask) >>> 0;
    const path = [this.root];
    let node = this.root;
    for (let depth = 0; depth < prefixLen; depth++) {
      const bit = (net >>> (ADDRESS_BITS - 1 - depth)) & 1;
      if (node.child[bit] === null) {
        node.child[bit] = makeNode();
        this.nodeCount++;
      }
      node = node.child[bit];
      path.push(node);
    }
    if (node.value === null) this.size++;
    node.value = value;
    node.prefixLen = prefixLen;
    node.prefix = net;

    for (let i = path.length - 1; i >= 0; i--) recompute(path[i]);
    return node;
  }

  /** Longest-prefix match for an address; null when nothing matches. */
  lookup(ip) {
    const addr = ip >>> 0;
    let node = this.root;
    let best = null;
    this.lastVisitedNodes = 0;
    if (node.value !== null) {
      best = { nextHop: node.value, prefixLen: node.prefixLen, prefix: node.prefix };
    }
    for (let depth = 0; depth < ADDRESS_BITS; depth++) {
      const bit = (addr >>> (ADDRESS_BITS - 1 - depth)) & 1;
      const child = node.child[bit];
      if (child === null) break;
      this.lastVisitedNodes++;
      if (child.bestDepth === -1) break;    // subtree is prefix-less: nothing longer below
      node = child;
      if (node.value !== null) {
        best = { nextHop: node.value, prefixLen: node.prefixLen, prefix: node.prefix };
      }
    }
    return best;
  }

  /** Remove `network/prefixLen`; false when it was not present. */
  delete(network, prefixLen) {
    const mask = prefixMask(prefixLen);
    const net = (network & mask) >>> 0;
    const path = [this.root];
    let node = this.root;
    for (let depth = 0; depth < prefixLen; depth++) {
      const bit = (net >>> (ADDRESS_BITS - 1 - depth)) & 1;
      const next = node.child[bit];
      if (next === null) return false;
      path.push(next);
      node = next;
    }
    if (node.value === null) return false;
    node.value = null;
    node.prefixLen = -1;

    // Path indices: path[i] sits at depth i, its on-path child uses bit (31 - i).
    for (let i = path.length - 1; i >= 0; i--) {
      const parent = path[i];
      const bit = (net >>> (ADDRESS_BITS - 1 - i)) & 1;
      const child = parent.child[bit];
      if (child !== null && child.value === null && child.child[0] === null && child.child[1] === null) {
        parent.child[bit] = null;     // prune: keep the trie canonical
        this.nodeCount--;
      }
      recompute(parent);
    }
    this.size--;
    return true;
  }

  /** All stored prefixes, in bit order (0-bit branch first) - i.e. by network then length. */
  entries() {
    const out = [];
    const walk = (node) => {
      if (node.value !== null) {
        out.push({ prefix: node.prefix, prefixLen: node.prefixLen, nextHop: node.value });
      }
      if (node.child[0] !== null) walk(node.child[0]);
      if (node.child[1] !== null) walk(node.child[1]);
    };
    walk(this.root);
    return out;
  }

  /** True when every node that carries no prefix has at least one child (invariant). */
  isCanonical() {
    const stack = [this.root];
    while (stack.length > 0) {
      const node = stack.pop();
      // the root may be an empty leaf (empty table); any other prefix-less leaf would be dead wood
      if (node !== this.root && node.value === null && node.child[0] === null && node.child[1] === null) return false;
      const expected = (() => {
        let best = node.prefixLen;
        for (const child of node.child) if (child !== null && child.bestDepth > best) best = child.bestDepth;
        return best;
      })();
      if (expected !== node.bestDepth) return false;
      for (const child of node.child) if (child !== null) stack.push(child);
    }
    return true;
  }
}

/** Oracle: linear scan of a flat prefix list, longest length wins. */
function lpmBruteForce(table, ip) {
  const addr = ip >>> 0;
  let best = null;
  for (const entry of table) {
    const mask = prefixMask(entry.prefixLen);
    if ((addr & mask) >>> 0 !== ((entry.network & mask) >>> 0)) continue;
    if (best === null || entry.prefixLen > best.prefixLen) best = entry;
  }
  if (best === null) return null;
  return { nextHop: best.nextHop, prefixLen: best.prefixLen, prefix: (best.network & prefixMask(best.prefixLen)) >>> 0 };
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  int(bound) {
    return this.next() % bound;
  }
}

module.exports = { parseIPv4, formatIPv4, prefixMask, parseCIDR, PrefixTrie, lpmBruteForce, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  // --- parser round-trips and strictness -------------------------------------------------
  for (const text of ["0.0.0.0", "10.1.2.3", "128.0.0.1", "192.168.1.1", "255.255.255.255"]) {
    assert.strictEqual(formatIPv4(parseIPv4(text)), text, `round-trip ${text}`);
  }
  assert.strictEqual(parseIPv4("255.255.255.255"), ALL_ONES >>> 0);
  for (const bad of ["10.1.2", "10.1.2.3.4", "10.1.2.256", "10.1.2.03", "a.b.c.d"]) {
    assert.throws(() => parseIPv4(bad), /not an IPv4 address|bad octet|out of range/, `reject ${bad}`);
  }
  assert.deepStrictEqual(parseCIDR("10.1.2.3/16"), { network: parseIPv4("10.1.0.0"), prefixLen: 16 });
  assert.strictEqual(prefixMask(0), 0);
  assert.strictEqual(prefixMask(8), 0xff000000 >>> 0);
  assert.strictEqual(prefixMask(32), ALL_ONES >>> 0);
  assert.throws(() => prefixMask(33), RangeError);

  // --- hand-derived LPM decisions --------------------------------------------------------
  const trie = new PrefixTrie();
  const table = [
    { network: parseIPv4("0.0.0.0"), prefixLen: 0, nextHop: "DEFAULT" },
    { network: parseIPv4("10.0.0.0"), prefixLen: 8, nextHop: "A" },
    { network: parseIPv4("10.1.0.0"), prefixLen: 16, nextHop: "B" },
    { network: parseIPv4("10.1.2.0"), prefixLen: 24, nextHop: "C" },
    { network: parseIPv4("10.1.2.7"), prefixLen: 32, nextHop: "HOST" },
  ];
  for (const entry of table) trie.insert(entry.network, entry.prefixLen, entry.nextHop);
  assert.strictEqual(trie.size, 5);
  assert.strictEqual(trie.lookup(parseIPv4("10.1.2.7")).nextHop, "HOST");   // /32 beats /24
  assert.strictEqual(trie.lookup(parseIPv4("10.1.2.8")).nextHop, "C");      // /24
  assert.strictEqual(trie.lookup(parseIPv4("10.1.3.1")).nextHop, "B");      // /16
  assert.strictEqual(trie.lookup(parseIPv4("10.2.0.1")).nextHop, "A");      // /8
  assert.strictEqual(trie.lookup(parseIPv4("192.0.2.1")).nextHop, "DEFAULT"); // /0
  assert.strictEqual(trie.lookup(parseIPv4("10.1.2.7")).prefixLen, 32);
  assert.strictEqual(trie.lookup(10).nextHop, "DEFAULT");
  assert.ok(trie.isCanonical(), "canonical after inserts");

  // empty trie and a table consisting only of a default route
  const empty = new PrefixTrie();
  assert.strictEqual(empty.lookup(parseIPv4("8.8.8.8")), null);
  assert.deepStrictEqual(empty.entries(), []);
  empty.insert(0, 0, "D");
  assert.strictEqual(empty.lookup(parseIPv4("8.8.8.8")).nextHop, "D");
  assert.strictEqual(empty.lookup(parseIPv4("255.255.255.255")).nextHop, "D");

  // --- enumeration order -----------------------------------------------------------------
  const prefixes = trie.entries().map((e) => `${formatIPv4(e.prefix)}/${e.prefixLen}`);
  assert.deepStrictEqual(prefixes, [
    "0.0.0.0/0", "10.0.0.0/8", "10.1.0.0/16", "10.1.2.0/24", "10.1.2.7/32",
  ]);

  // --- replacement and deletion ----------------------------------------------------------
  assert.strictEqual(trie.insert(parseIPv4("10.1.0.0"), 16, "B2").value, "B2");
  assert.strictEqual(trie.size, 5, "replace must not grow the table");
  assert.strictEqual(trie.lookup(parseIPv4("10.1.3.1")).nextHop, "B2");
  assert.strictEqual(trie.insert(parseIPv4("10.1.0.0"), 16, "B").value, "B");

  const nodesBefore = trie.nodeCount;
  assert.strictEqual(trie.delete(parseIPv4("10.1.2.7"), 32), true);
  assert.strictEqual(trie.lookup(parseIPv4("10.1.2.7")).nextHop, "C", "falls back to /24");
  assert.ok(trie.nodeCount < nodesBefore, "leaf pruned");
  assert.strictEqual(trie.delete(parseIPv4("10.1.2.7"), 32), false, "double delete");
  assert.strictEqual(trie.delete(parseIPv4("172.16.0.0"), 12), false, "unknown prefix");
  assert.strictEqual(trie.size, 4);

  const pruned = new PrefixTrie();
  pruned.insert(parseIPv4("192.168.5.0"), 24, "P");
  const nodesWithLeaf = pruned.nodeCount;
  pruned.delete(parseIPv4("192.168.5.0"), 24);
  assert.strictEqual(pruned.nodeCount, 1, "only the root stays");
  assert.ok(nodesWithLeaf > 1 && pruned.isCanonical());
  assert.strictEqual(pruned.lookup(parseIPv4("192.168.5.1")), null);

  // --- pruning is observable: a table living only in 128/1 never walks far for 10/8 ------
  const half = new PrefixTrie();
  half.insert(parseIPv4("128.0.0.0"), 8, "HI");
  half.lookup(parseIPv4("10.0.0.1"));
  assert.strictEqual(half.lastVisitedNodes, 0, "left subtree absent -> no node visited");
  half.lookup(parseIPv4("200.0.0.1"));
  assert.strictEqual(half.lastVisitedNodes, 1, "walk stops where the subtree dies");

  // deeper ancestor with no prefix of its own prunes the walk of a /32-only table
  const deep = new PrefixTrie();
  deep.insert(parseIPv4("1.2.3.4"), 32, "X");
  assert.strictEqual(deep.lookup(parseIPv4("1.2.3.4")).nextHop, "X");
  assert.strictEqual(deep.lastVisitedNodes, 32);
  assert.strictEqual(deep.lookup(parseIPv4("1.2.3.5")), null);
  assert.ok(deep.lastVisitedNodes <= 32, "never touches more than one node per bit");

  // --- oracle agreement on deterministic tables -----------------------------------------
  const lcg = new Lcg(684);
  for (let round = 0; round < 12; round++) {
    const size = 20 + lcg.int(60);
    const local = new PrefixTrie();
    const localTable = [];
    const seen = new Set();
    for (let i = 0; i < size; i++) {
      const prefixLen = 1 + lcg.int(32);
      const mask = prefixMask(prefixLen);
      const network = (lcg.next() & mask) >>> 0;
      const key = `${network}/${prefixLen}`;
      if (seen.has(key)) continue;           // keep the table free of duplicates
      seen.add(key);
      const nextHop = `nh${lcg.int(7)}`;
      const entry = { network, prefixLen, nextHop };
      local.insert(network, prefixLen, nextHop);
      localTable.push(entry);
    }
    assert.strictEqual(local.size, localTable.length, "size equals distinct prefixes");
    assert.ok(local.isCanonical(), `canonical round ${round}`);

    // every stored prefix must at least match itself
    for (const entry of localTable) {
      const hit = local.lookup(entry.network);
      assert.ok(hit !== null && hit.prefixLen >= entry.prefixLen, `self match round ${round}`);
      assert.strictEqual((entry.network & prefixMask(hit.prefixLen)) >>> 0, hit.prefix, "returned prefix is masked");
    }

    for (let probe = 0; probe < 120; probe++) {
      const ip = lcg.next() >>> 0;
      const fast = local.lookup(ip);
      const slow = lpmBruteForce(localTable, ip);
      assert.deepStrictEqual(fast, slow, `oracle round ${round} ip ${formatIPv4(ip)}`);
    }
  }

  // --- larger table, mixed insert/delete, brute-force cross-check -------------------------
  const big = new PrefixTrie();
  const bigTable = [];
  const lcg2 = new Lcg(20241016);
  const bigSeen = new Set();
  while (bigTable.length < 200) {
    const prefixLen = [8, 12, 16, 20, 24, 28, 30, 32][lcg2.int(8)];
    const network = (lcg2.next() & prefixMask(prefixLen)) >>> 0;
    const key = `${network}/${prefixLen}`;
    if (bigSeen.has(key)) continue;
    bigSeen.add(key);
    const entry = { network, prefixLen, nextHop: `hop-${lcg2.int(16)}` };
    big.insert(network, prefixLen, entry.nextHop);
    bigTable.push(entry);
  }
  assert.strictEqual(big.size, 200);
  for (let probe = 0; probe < 400; probe++) {
    const ip = lcg2.next() >>> 0;
    assert.deepStrictEqual(big.lookup(ip), lpmBruteForce(bigTable, ip), `big oracle ${probe}`);
  }

  for (let i = 0; i < 60; i++) {
    const entry = bigTable[i];
    assert.strictEqual(big.delete(entry.network, entry.prefixLen), true, "delete known");
    bigTable.splice(bigTable.indexOf(entry), 1);
  }
  assert.strictEqual(big.size, 140);
  assert.ok(big.isCanonical(), "canonical after deletions");
  for (let probe = 0; probe < 200; probe++) {
    const ip = lcg2.next() >>> 0;
    assert.deepStrictEqual(big.lookup(ip), lpmBruteForce(bigTable, ip), `post-delete oracle ${probe}`);
  }

  // distribution of /32 host routes: exactly the hosts, nothing else
  const hosts = new PrefixTrie();
  const hostIps = [];
  for (let i = 0; i < 50; i++) {
    const ip = parseIPv4(`203.0.113.${i}`);
    hosts.insert(ip, 32, `h${i}`);
    hostIps.push(ip);
  }
  assert.strictEqual(hosts.size, 50);
  for (let i = 0; i < 50; i++) {
    assert.strictEqual(hosts.lookup(hostIps[i]).nextHop, `h${i}`, `host ${i}`);
  }
  assert.strictEqual(hosts.lookup(parseIPv4("203.0.113.200")), null, "no covering /24 in the table");

  console.log(`prefixes=${big.size} nodes=${big.nodeCount} lookups=${big.lastVisitedNodes}`);
  console.log("PASS");
}
