/** Task 685 - Implement fast IP routing simulation with TCAM-like behavior (bitwise). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: a Ternary Content-Addressable Memory (TCAM) for IPv4 forwarding. A TCAM entry is a
 * (value, mask) pair and matches a 32-bit word w bitwise when `(w & mask) === value`; the mask
 * may contain *holes*, so an entry is not restricted to a prefix (e.g. value 0x0A000009 with mask
 * 0xFFFFFF0F matches every 10.0.x.9 address). Lookup is inherently parallel in hardware, so the
 * software model scans the table and returns the *first* matching entry: array order is priority
 * order. A real router therefore programs prefix rules sorted by descending prefix length, which
 * turns first-match into longest-prefix-match. Entries are kept canonical (`value & ~mask == 0`)
 * and the table has a fixed capacity, exactly like silicon.
 *
 * Cross-checked here: first-match priority, canonical-form rejection, capacity refusal, per-entry
 * hit counters, ternary (non-prefix) rules, a hex-pattern compiler with wildcard nibbles, and - the
 * important one - exhaustive bitwise agreement between the TCAM programmed with prefixes sorted by
 * length and a brute-force longest-prefix-match scan of the same route list.
 */
"use strict";

const WORD_BITS = 32;
const ALL_ONES = 0xffffffff;

/** Host-order 32-bit AND that never returns a negative number. */
function band(a, b) {
  return (a & b) >>> 0;
}

function matchesTernary(value, mask, word) {
  return band(word >>> 0, mask) === band(value, mask);
}

function popcount32(value) {
  let v = value >>> 0;
  let count = 0;
  while (v !== 0) {
    v = (v & (v - 1)) >>> 0;
    count++;
  }
  return count;
}

/** Number of masked bits: the "specificity" of a ternary rule. Prefix /p has width p. */
function maskWidth(mask) {
  return popcount32(mask);
}

/** mask of `len` leading ones; len = 0 means the wildcard rule that matches everything. */
function prefixMask(len) {
  if (!Number.isInteger(len) || len < 0 || len > WORD_BITS) {
    throw new RangeError(`prefix length out of range: ${len}`);
  }
  return len === 0 ? 0 : (ALL_ONES << (WORD_BITS - len)) >>> 0;
}

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

/** "a.b.c.d/len" -> { network, prefixLen } value/mask pair for a TCAM prefix entry. */
function prefixRule(cidr, nextHop) {
  const [addr, lenText] = String(cidr).split("/");
  const prefixLen = Number(lenText);
  const mask = prefixMask(prefixLen);
  return { value: band(parseIPv4(addr), mask), mask, nextHop, label: cidr, prefixLen };
}

/** Compile a hex/wildcard pattern such as "0A00xx00" (x or ? = wildcard nibble). */
function patternRule(pattern, nextHop) {
  const text = String(pattern).toUpperCase();
  if (!/^[0-9A-FX?]{8}$/.test(text)) throw new RangeError(`bad ternary pattern: ${pattern}`);
  let value = 0;
  let mask = 0;
  for (const ch of text) {
    mask = (mask << 4) >>> 0;
    value = (value << 4) >>> 0;
    if (ch === "X" || ch === "?") continue;
    mask = (mask | 0xf) >>> 0;
    value = (value | parseInt(ch, 16)) >>> 0;
  }
  return { value: value >>> 0, mask: mask >>> 0, nextHop, label: text };
}

class TcamTable {
  constructor(capacity) {
    if (!Number.isInteger(capacity) || capacity <= 0) throw new RangeError("capacity must be positive");
    this.capacity = capacity;
    this.entries = [];
    this.hits = [];
    this.lookups = 0;
    this.matched = 0;
  }

  /** Add one entry at the *end* (lowest priority). Returns false when the table is full. */
  add(value, mask, nextHop, label) {
    if (this.entries.length >= this.capacity) return false;
    const canonicalValue = band(value, mask);
    if (canonicalValue !== band(value, ALL_ONES)) {
      throw new RangeError(`entry not canonical: value has bits outside the mask (${label})`);
    }
    this.entries.push({ value: canonicalValue, mask: mask >>> 0, nextHop, label });
    this.hits.push(0);
    return true;
  }

  addRule(rule) {
    return this.add(rule.value, rule.mask, rule.nextHop, rule.label);
  }

  /** Program prefix routes in longest-prefix-first order, ties broken by insertion order. */
  programPrefixRoutes(routes) {
    const ordered = routes
      .map((route, index) => ({ route, index }))
      .sort((a, b) => (b.route.prefixLen - a.route.prefixLen) || (a.index - b.index));
    let programmed = 0;
    for (const { route } of ordered) {
      if (this.add(route.value, route.mask, route.nextHop, route.label)) programmed++;
    }
    return programmed;
  }

  /** First match wins - the software stand-in for TCAM's parallel compare. */
  lookup(word) {
    this.lookups++;
    for (let i = 0; i < this.entries.length; i++) {
      const entry = this.entries[i];
      if (matchesTernary(entry.value, entry.mask, word)) {
        this.hits[i]++;
        this.matched++;
        return { nextHop: entry.nextHop, index: i, label: entry.label, width: maskWidth(entry.mask) };
      }
    }
    return null;
  }

  /** Remove by label; returns the number of entries removed. */
  removeLabel(label) {
    let removed = 0;
    for (let i = this.entries.length - 1; i >= 0; i--) {
      if (this.entries[i].label === label) {
        this.entries.splice(i, 1);
        this.hits.splice(i, 1);
        removed++;
      }
    }
    return removed;
  }

  utilization() {
    return this.entries.length / this.capacity;
  }

  /** Sum of hit counters must equal the number of lookups that matched, at every instant. */
  hitsConsistent() {
    let total = 0;
    for (const h of this.hits) total += h;
    return this.hits.length === this.entries.length && total === this.matched;
  }
}

/** Brute-force longest-prefix match over a flat route list (the oracle). */
function bruteForceLpm(routes, word) {
  let best = null;
  for (const route of routes) {
    if (matchesTernary(route.value, route.mask, word)) {
      if (best === null || route.prefixLen > best.prefixLen) best = route;
    }
  }
  return best === null ? null : best.nextHop;
}

/** Forward a deterministic packet stream; returns per-next-hop and per-entry counters. */
function forwardPackets(table, packets) {
  const perHop = new Map();
  const perEntry = new Array(table.entries.length).fill(0);
  const drops = [];
  for (const packet of packets) {
    const hit = table.lookup(packet);
    if (hit === null) {
      drops.push(packet);
      continue;
    }
    perHop.set(hit.nextHop, (perHop.get(hit.nextHop) || 0) + 1);
    perEntry[hit.index]++;
  }
  return { perHop, perEntry, drops };
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

module.exports = {
  matchesTernary, popcount32, maskWidth, prefixMask, parseIPv4, formatIPv4,
  prefixRule, patternRule, TcamTable, bruteForceLpm, forwardPackets, Lcg,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- bitwise match semantics -----------------------------------------------------------
  assert.strictEqual(matchesTernary(0x0a000001, ALL_ONES, 0x0a000001), true);
  assert.strictEqual(matchesTernary(0x0a000001, ALL_ONES, 0x0a000002), false);
  assert.strictEqual(matchesTernary(0, 0, 0xdeadbeef), true, "mask 0 matches everything");
  assert.strictEqual(matchesTernary(0x0a000000, 0xff000000, 0x0affffff), true);
  assert.strictEqual(matchesTernary(0x0a000000, 0xff000000, 0x0bffffff), false);
  assert.strictEqual(matchesTernary(0x0a000009, 0xffffff0f, 0x0a0000f9), true, "hole in the mask");
  assert.strictEqual(matchesTernary(0x0a000009, 0xffffff0f, 0x0a0000f8), false);
  assert.strictEqual(popcount32(0xffffffff), 32);
  assert.strictEqual(maskWidth(0xf0f0f0f0), 16, "ternary width is the Hamming weight");

  // --- canonical-form rejection and capacity ---------------------------------------------
  const limited = new TcamTable(3);
  assert.throws(() => limited.add(0x0a000001, 0xffffff00, "X", "dirty"), RangeError,
    "value bits outside the mask must be rejected");
  assert.strictEqual(limited.add(0x0a000000, 0xffffff00, "A", "10.0.0.0/24"), true);
  assert.strictEqual(limited.addRule(prefixRule("10.0.0.0/16", "B")), true);
  assert.strictEqual(limited.add(0, 0, "DEFAULT", "0.0.0.0/0"), true);
  assert.strictEqual(limited.add(0, 0, "EXTRA", "overflow"), false, "table full");
  assert.strictEqual(limited.entries.length, 3);
  assert.strictEqual(limited.utilization(), 1);
  assert.strictEqual(limited.removeLabel("nope"), 0);
  assert.strictEqual(limited.removeLabel("10.0.0.0/24"), 1);
  assert.strictEqual(limited.entries.length, 2, "hits array stays aligned");

  // --- priority order: the first entry wins, no matter how specific the later ones are ----
  const priority = new TcamTable(8);
  priority.addRule(prefixRule("0.0.0.0/0", "DEFAULT"));         // index 0 -> always wins first
  priority.addRule(prefixRule("10.0.0.0/8", "TEN"));
  assert.strictEqual(priority.lookup(parseIPv4("10.1.2.3")).nextHop, "DEFAULT",
    "unsorted programming yields a default-route black hole");
  assert.strictEqual(priority.lookup(parseIPv4("10.1.2.3")).width, 0);

  const sorted = new TcamTable(8);
  sorted.programPrefixRoutes([
    prefixRule("0.0.0.0/0", "DEFAULT"),
    prefixRule("10.0.0.0/8", "TEN"),
    prefixRule("10.1.0.0/16", "TEN1"),
  ]);
  assert.deepStrictEqual(sorted.entries.map((e) => e.label), ["10.1.0.0/16", "10.0.0.0/8", "0.0.0.0/0"],
    "longest prefix first");
  assert.strictEqual(sorted.lookup(parseIPv4("10.1.2.3")).nextHop, "TEN1");
  assert.strictEqual(sorted.lookup(parseIPv4("10.9.9.9")).nextHop, "TEN");
  assert.strictEqual(sorted.lookup(parseIPv4("192.0.2.1")).nextHop, "DEFAULT");

  // --- ternary rules that a prefix trie cannot express -----------------------------------
  const acl = new TcamTable(8);
  acl.addRule(patternRule("0A0000XX", "SMALL"));        // 10.0.0.0/24 written as a nibble pattern
  acl.addRule(patternRule("0A000009", "EXACT9"));
  assert.strictEqual(acl.lookup(parseIPv4("10.0.0.0")).nextHop, "SMALL");
  assert.strictEqual(acl.lookup(parseIPv4("10.0.0.9")).nextHop, "SMALL", "earlier rule wins");
  assert.strictEqual(acl.lookup(parseIPv4("10.0.1.0")), null, "0X nibbles do not cross octets");
  const reversed = new TcamTable(8);
  reversed.addRule(patternRule("0A000009", "EXACT9"));
  reversed.addRule(patternRule("0A0000XX", "SMALL"));
  assert.strictEqual(reversed.lookup(parseIPv4("10.0.0.9")).nextHop, "EXACT9", "priority flipped");
  assert.strictEqual(reversed.lookup(parseIPv4("10.0.0.8")).nextHop, "SMALL");
  const odd = new TcamTable(4);
  odd.addRule(patternRule("0A00xx09", "LAST-OCT-9"));
  assert.strictEqual(odd.lookup(parseIPv4("10.0.77.9")).nextHop, "LAST-OCT-9");
  assert.strictEqual(odd.lookup(parseIPv4("10.0.77.8")), null);
  assert.strictEqual(maskWidth(odd.entries[0].mask), 24, "10.0.x.9 masks 24 of 32 bits");

  // --- forwarding simulation: drops and counters ------------------------------------------
  const edge = new TcamTable(16);
  edge.programPrefixRoutes([
    prefixRule("0.0.0.0/0", "core"),
    prefixRule("10.10.0.0/16", "lan-a"),
    prefixRule("10.20.0.0/16", "lan-b"),
  ]);
  const lcg = new Lcg(685);
  const packets = [];
  for (let i = 0; i < 60; i++) {
    packets.push(parseIPv4(`10.${lcg.int(3) === 0 ? 10 : 20}.${lcg.int(256)}.${lcg.int(256)}`));
  }
  const fwd = forwardPackets(edge, packets);
  let forwarded = 0;
  for (const count of fwd.perHop.values()) forwarded += count;
  assert.strictEqual(forwarded + fwd.drops.length, packets.length, "every packet is accounted for");
  assert.strictEqual(edge.lookups, packets.length);
  assert.ok(edge.hitsConsistent(), "hit counters match the matched lookups");
  assert.strictEqual(fwd.perEntry.length, 3);
  assert.strictEqual(fwd.perEntry[2], 0, "default route (lowest priority) never hit by 10.x traffic");
  assert.strictEqual(fwd.perEntry.reduce((a, b) => a + b, 0), forwarded, "per-entry counters sum to forwarded");
  assert.strictEqual(fwd.perHop.get("core"), undefined);
  assert.strictEqual(edge.lookup(parseIPv4("10.10.5.5")).nextHop, "lan-a");
  assert.strictEqual(edge.lookup(parseIPv4("10.20.5.5")).nextHop, "lan-b");

  // --- the key property: TCAM sorted by prefix length == brute-force LPM ------------------
  const routes = [
    prefixRule("0.0.0.0/0", "DEFAULT"),
    prefixRule("10.0.0.0/8", "A"),
    prefixRule("10.1.0.0/16", "B"),
    prefixRule("10.1.2.0/24", "C"),
    prefixRule("10.1.2.7/32", "HOST"),
    prefixRule("172.16.0.0/12", "V"),
    prefixRule("192.168.1.128/25", "S"),
    prefixRule("203.0.113.0/24", "T"),
  ];
  const big = new TcamTable(routes.length);
  assert.strictEqual(big.programPrefixRoutes(routes), routes.length);
  assert.strictEqual(big.entries.length, routes.length, "no truncation at exactly capacity");

  const lcg2 = new Lcg(20241016);
  const probes = [
    parseIPv4("10.1.2.7"), parseIPv4("10.1.2.8"), parseIPv4("10.1.9.9"), parseIPv4("10.9.9.9"),
    parseIPv4("11.0.0.1"), parseIPv4("172.31.255.255"), parseIPv4("192.168.1.200"),
    parseIPv4("192.168.1.100"), parseIPv4("203.0.113.44"), parseIPv4("8.8.8.8"),
  ];
  for (let i = 0; i < 300; i++) probes.push(lcg2.next() >>> 0);
  for (const probe of probes) {
    const fast = big.lookup(probe);
    const slow = bruteForceLpm(routes, probe);
    assert.strictEqual(fast === null ? null : fast.nextHop, slow, `lpm agreement ${formatIPv4(probe)}`);
  }
  assert.strictEqual(big.lookup(parseIPv4("10.1.2.7")).nextHop, "HOST");
  assert.strictEqual(big.lookup(parseIPv4("10.1.2.7")).width, 32);
  assert.strictEqual(big.lookup(parseIPv4("172.20.1.1")).nextHop, "V");
  assert.strictEqual(big.lookup(parseIPv4("192.168.1.100")).nextHop, "DEFAULT", "/25 does not cover .100");
  assert.ok(big.hitsConsistent(), "counters consistent after the sweep");

  // --- edge cases: empty table, wildcard-only table, multi-octet holes --------------------
  const emptyTable = new TcamTable(2);
  assert.strictEqual(emptyTable.lookup(parseIPv4("10.0.0.1")), null);
  assert.strictEqual(emptyTable.utilization(), 0);
  assert.ok(emptyTable.hitsConsistent(), "no entries, no hits");

  const wildcard = new TcamTable(2);
  wildcard.addRule(patternRule("xxxxxxxx", "ANY"));
  for (const ip of ["0.0.0.0", "10.0.0.1", "255.255.255.255"]) {
    assert.strictEqual(wildcard.lookup(parseIPv4(ip)).nextHop, "ANY", ip);
  }

  const holey = new TcamTable(4);
  holey.add(0x0a000900, 0xff00ff00, "SRC10-DST9", "10.x.9.x");
  assert.strictEqual(holey.lookup(parseIPv4("10.7.9.200")).nextHop, "SRC10-DST9");
  assert.strictEqual(holey.lookup(parseIPv4("10.7.8.200")), null);
  assert.strictEqual(maskWidth(0xff00ff00), 16);
  assert.ok(maskWidth(0xff00ff00) < maskWidth(prefixMask(24)), "wildcarded rule is coarser than /24");

  // programPrefixRoutes sorts, so the identical route list yields the longest match instead of /0
  const programmed = new TcamTable(4);
  programmed.programPrefixRoutes([
    { value: 0, mask: 0, nextHop: "DEFAULT", label: "0.0.0.0/0", prefixLen: 0 },
    prefixRule("10.0.0.0/8", "TEN"),
  ]);
  assert.strictEqual(programmed.lookup(parseIPv4("10.1.1.1")).nextHop, "TEN", "priority order after sorting");

  console.log(`entries=${big.entries.length} probes=${probes.length} util=${big.utilization().toFixed(2)}`);
  console.log("PASS");
}
