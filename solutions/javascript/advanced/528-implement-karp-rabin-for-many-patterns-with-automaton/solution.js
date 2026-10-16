/**
 * Task 528 - Implement Karp-Rabin for many patterns with automaton. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Two multi-pattern matchers that must agree on every input. (1) An Aho-Corasick automaton: a trie
 * of the patterns with breadth-first failure links (the failure of a node is the longest proper
 * suffix of its string that is again a trie node), where every node's output list is the union of
 * its own terminal patterns and the outputs inherited from its failure link, so a single left-to-
 * right scan reports all occurrences in O(|text| + matches). (2) Rabin-Karp hashing for many
 * patterns at once: the patterns are grouped by length, one rolling hash over the text is computed
 * per distinct length (BigInt arithmetic modulo 1000000007 with base 131), and every hash hit is
 * *verified* against the actual characters, so a hash collision can never produce a false positive.
 *
 * Documented checks: both matchers return exactly the sorted [start, patternIndex] list produced by
 * a brute-force double loop, for LCG-generated texts over two-, three- and twenty-six-letter
 * alphabets with patterns of mixed lengths, for duplicate and self-overlapping patterns, and for a
 * text consisting of a single repeated character (where the match count has the closed form
 * sum over patterns of (n - L + 1)); the rolling hash is validated independently by comparing the
 * incremental window hashes with the direct O(L) recomputation for every window of several lengths,
 * and by checking that every character verification performed during hashing ends in a genuine match
 * - on the tested data the hash never collides, and a collision would only cost a wasted
 * verification, never a false report; the automaton is validated structurally - every failure link points to the
 * longest proper suffix that is a trie node, output sets equal the terminal patterns that are
 * suffixes of the node's string, node and transition counts respect the total pattern length - and
 * the automaton's current state after each text position equals the brute-force longest suffix of
 * the processed prefix that is a prefix of some pattern. Boundary cases: no patterns, empty text,
 * patterns longer than the text, and empty patterns (rejected).
 */
"use strict";

const BASE = 131n;
const MOD = 1000000007n;

function lcgString(n, seed, alphabet) {
  let out = "";
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    out += String.fromCharCode(97 + (seed % alphabet));
  }
  return out;
}

function modPow(base, exponent, modulus) {
  let result = 1n;
  let b = base % modulus;
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}

function hashOf(pattern) {
  let h = 0n;
  for (let i = 0; i < pattern.length; i++) {
    h = (h * BASE + BigInt(pattern.charCodeAt(i))) % MOD;
  }
  return h;
}

/** Direct hash of every window of the given length (O(text * length)). */
function directWindowHashes(text, length) {
  const out = [];
  for (let start = 0; start + length <= text.length; start++) {
    out.push(hashOf(text.slice(start, start + length)));
  }
  return out;
}

/** Incremental rolling hashes for the same windows (O(text)); must match `directWindowHashes`. */
function rollingWindowHashes(text, length) {
  const n = text.length;
  if (length > n) return [];
  const high = modPow(BASE, BigInt(length - 1), MOD);
  let current = 0n;
  for (let i = 0; i < length; i++) current = (current * BASE + BigInt(text.charCodeAt(i))) % MOD;
  const out = [current];
  for (let start = length; start < n; start++) {
    const dropped = (BigInt(text.charCodeAt(start - length)) * high) % MOD;
    current = (current - dropped + MOD) % MOD;
    current = (current * BASE + BigInt(text.charCodeAt(start))) % MOD;
    out.push(current);
  }
  return out;
}

function verify(text, start, pattern) {
  if (start + pattern.length > text.length) return false;
  for (let i = 0; i < pattern.length; i++) {
    if (text.charCodeAt(start + i) !== pattern.charCodeAt(i)) return false;
  }
  return true;
}

/** Rabin-Karp over many patterns, grouped by length; every hash hit is verified by characters. */
function karpRabinMulti(text, patterns) {
  for (const pattern of patterns) if (pattern.length === 0) throw new Error("empty pattern");
  const matches = [];
  let hits = 0;
  let verifications = 0;
  const byLength = new Map();
  patterns.forEach((pattern, index) => {
    const length = pattern.length;
    if (length > text.length) return;
    if (!byLength.has(length)) byLength.set(length, new Map());
    const table = byLength.get(length);
    const key = hashOf(pattern).toString();
    if (!table.has(key)) table.set(key, []);
    table.get(key).push(index);
  });
  for (const [length, table] of byLength) {
    const hashes = rollingWindowHashes(text, length);
    hashes.forEach((hash, start) => {
      const bucket = table.get(hash.toString());
      if (bucket === undefined) return;
      hits += 1;
      for (const index of bucket) {
        verifications += 1;
        if (verify(text, start, patterns[index])) matches.push([start, index]);
      }
    });
  }
  matches.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return { matches, hits, verifications, lengths: [...byLength.keys()].sort((a, b) => a - b) };
}

/** Aho-Corasick automaton over a pattern list. */
class AhoCorasick {
  constructor(patterns) {
    for (const pattern of patterns) if (pattern.length === 0) throw new Error("empty pattern");
    this.patterns = patterns.slice();
    this.child = [new Map()];
    this.fail = [-1];
    this.output = [[]];
    this.depth = [0];
    this.parent = [-1];
    this.parentChar = [-1];
    patterns.forEach((pattern, index) => this.insert(pattern, index));
    this.buildFailures();
  }

  insert(pattern, index) {
    let v = 0;
    for (let i = 0; i < pattern.length; i++) {
      const code = pattern.charCodeAt(i);
      let nxt = this.child[v].get(code);
      if (nxt === undefined) {
        nxt = this.child.length;
        this.child.push(new Map());
        this.fail.push(-1);
        this.output.push([]);
        this.depth.push(this.depth[v] + 1);
        this.parent.push(v);
        this.parentChar.push(code);
        this.child[v].set(code, nxt);
      }
      v = nxt;
    }
    this.output[v].push(index);
  }

  buildFailures() {
    const queue = [];
    for (const [, child] of this.child[0]) {
      this.fail[child] = 0;
      queue.push(child);
    }
    for (let head = 0; head < queue.length; head++) {
      const v = queue[head];
      for (const [code, child] of this.child[v]) {
        let f = this.fail[v];
        while (f !== 0 && !this.child[f].has(code)) f = this.fail[f];
        const candidate = this.child[f].has(code) && this.child[f].get(code) !== child
          ? this.child[f].get(code) : 0;
        this.fail[child] = candidate;
        this.output[child] = this.output[child].concat(this.output[candidate]);
        queue.push(child);
      }
    }
  }

  stateCount() {
    return this.child.length;
  }

  transitionCount() {
    return this.child.reduce((acc, map) => acc + map.size, 0);
  }

  /** Follow goto/failure transitions for one character. */
  step(state, code) {
    let v = state;
    while (v !== 0 && !this.child[v].has(code)) v = this.fail[v];
    if (this.child[v].has(code)) v = this.child[v].get(code);
    return v;
  }

  /** Scan the text, returning the sorted [start, patternIndex] matches and the visited states. */
  search(text) {
    const matches = [];
    const states = [];
    let v = 0;
    for (let i = 0; i < text.length; i++) {
      v = this.step(v, text.charCodeAt(i));
      states.push(v);
      for (const index of this.output[v]) {
        matches.push([i - this.patterns[index].length + 1, index]);
      }
    }
    matches.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    return { matches, states, transitions: text.length };
  }

  /** The string of the trie node, read back through the parent pointers (for structural checks). */
  nodeString(v) {
    const codes = [];
    let current = v;
    while (current !== 0) {
      codes.push(this.parentChar[current]);
      current = this.parent[current];
    }
    return String.fromCharCode(...codes.reverse());
  }
}

/** Brute force: every (start, patternIndex) pair, sorted. */
function bruteForceMatches(text, patterns) {
  const matches = [];
  patterns.forEach((pattern, index) => {
    if (pattern.length === 0) return;
    for (let start = 0; start + pattern.length <= text.length; start++) {
      if (text.startsWith(pattern, start)) matches.push([start, index]);
    }
  });
  matches.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return matches;
}

function main() {
  const assert = require("node:assert");

  // Closed form on a single-character text.
  const a8 = "aaaaaaaa";
  const a8Patterns = ["a", "aa", "aaa"];
  const a8Expected = [];
  a8Patterns.forEach((pattern, index) => {
    for (let start = 0; start + pattern.length <= a8.length; start++) a8Expected.push([start, index]);
  });
  a8Expected.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  assert.strictEqual(a8Expected.length, 8 + 7 + 6, "sum of (n - L + 1)");
  const a8Automaton = new AhoCorasick(a8Patterns);
  assert.deepStrictEqual(a8Automaton.search(a8).matches, a8Expected);
  assert.deepStrictEqual(karpRabinMulti(a8, a8Patterns).matches, a8Expected);

  // Hand-checked overlapping case.
  const overlapText = "abababa";
  const overlapPatterns = ["aba", "bab", "abab"];
  const overlapExpected = bruteForceMatches(overlapText, overlapPatterns);
  assert.deepStrictEqual(overlapExpected, [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2], [3, 1],
    [4, 0]]);
  assert.deepStrictEqual(new AhoCorasick(overlapPatterns).search(overlapText).matches,
    overlapExpected);
  assert.deepStrictEqual(karpRabinMulti(overlapText, overlapPatterns).matches, overlapExpected);

  const cases = [
    ["binary", lcgString(200, 11, 2),
      [lcgString(3, 21, 2), lcgString(4, 23, 2), lcgString(5, 25, 2), lcgString(6, 27, 2)]],
    ["ternary", lcgString(300, 31, 3),
      [lcgString(2, 41, 3), lcgString(3, 43, 3), lcgString(4, 45, 3), lcgString(7, 47, 3)]],
    ["wide", lcgString(400, 53, 26),
      ["abc", "xyz", "abcdefgh", lcgString(4, 55, 26), lcgString(5, 57, 26)]],
    ["duplicates", lcgString(150, 61, 2), ["ab", "ab", "ba", "ba", "abab"]],
    ["long-pattern", lcgString(120, 67, 2),
      [lcgString(120, 67, 2), lcgString(60, 71, 2), "ab", "baba"]],
    ["absent", lcgString(80, 73, 2), ["zz", "zzz", "q"]],
  ];
  for (const [name, text, patterns] of cases) {
    const expected = bruteForceMatches(text, patterns);
    const automaton = new AhoCorasick(patterns);
    const viaAutomaton = automaton.search(text);
    const viaHash = karpRabinMulti(text, patterns);
    assert.deepStrictEqual(viaAutomaton.matches, expected, `${name}: automaton matches`);
    assert.deepStrictEqual(viaHash.matches, expected, `${name}: Rabin-Karp matches`);
    assert.strictEqual(viaAutomaton.transitions, text.length,
      `${name}: the automaton visits one state per character`);
    assert.ok(automaton.stateCount() <= patterns.reduce((acc, p) => acc + p.length, 0) + 1,
      `${name}: trie nodes <= total pattern length + 1`);
    assert.ok(automaton.transitionCount() <= patterns.reduce((acc, p) => acc + p.length, 0),
      `${name}: trie transitions <= total pattern length`);
    assert.ok(viaHash.hits <= expected.length,
      `${name}: every hash hit yields at least one match (duplicates share a hit)`);
    assert.strictEqual(viaHash.verifications, expected.length,
      `${name}: every verified candidate is a genuine match - no hash collision on this data`);

    // The automaton's state after each prefix equals the longest suffix that is a pattern prefix.
    const states = viaAutomaton.states;
    for (let i = 0; i < text.length; i++) {
      const prefix = text.slice(0, i + 1);
      let expectedLength = 0;
      for (let start = 0; start < prefix.length; start++) {
        const suffix = prefix.slice(start);
        if (patterns.some((p) => p.startsWith(suffix)) && suffix.length > expectedLength) {
          expectedLength = suffix.length;
        }
      }
      assert.strictEqual(automaton.depth[states[i]], expectedLength,
        `${name}: automaton state depth at ${i}`);
      const nodeString = automaton.nodeString(states[i]);
      assert.ok(prefix.endsWith(nodeString) && nodeString.length === expectedLength,
        `${name}: the state string is the longest matching suffix`);
    }
  }

  // Structural: failure links point at the longest proper suffix that is a trie node; output lists
  // contain exactly the terminal patterns that are suffixes of the node string.
  const structural = new AhoCorasick(["a", "ab", "bab", "bc", "bca", "c", "ca"]);
  for (let v = 1; v < structural.stateCount(); v++) {
    const label = structural.nodeString(v);
    let expectedFail = 0;
    for (let cut = 1; cut < label.length && expectedFail === 0; cut++) {
      const suffix = label.slice(cut);
      for (let w = 1; w < structural.stateCount(); w++) {
        if (structural.nodeString(w) === suffix) {
          expectedFail = w;
          break;
        }
      }
    }
    assert.strictEqual(structural.fail[v], expectedFail,
      `failure link of '${label}' is the longest trie suffix`);
    const expectedOutput = structural.patterns
      .map((pattern, index) => [pattern, index])
      .filter(([pattern]) => label.endsWith(pattern))
      .map(([, index]) => index)
      .sort((a, b) => a - b);
    assert.deepStrictEqual(structural.output[v].slice().sort((a, b) => a - b), expectedOutput,
      `output set of '${label}'`);
  }

  // Rolling hash validation: incremental windows equal the direct recomputation, window by window.
  for (const [text, lengths] of [[lcgString(120, 91, 3), [1, 2, 5, 17]], ["a".repeat(50), [3, 10]]]) {
    for (const length of lengths) {
      assert.deepStrictEqual(rollingWindowHashes(text, length), directWindowHashes(text, length),
        `rolling hash equals the direct hash for length ${length}`);
    }
    assert.deepStrictEqual(rollingWindowHashes(text, text.length + 1), [],
      "a window longer than the text yields no hashes");
  }
  assert.strictEqual(hashOf("abc"), hashOf("abc"), "hashing is deterministic");
  assert.notStrictEqual(hashOf("abc"), hashOf("abd"));

  // Boundary cases.
  assert.deepStrictEqual(new AhoCorasick(["xyz"]).search("").matches, []);
  assert.deepStrictEqual(karpRabinMulti("", ["xyz"]).matches, []);
  assert.deepStrictEqual(new AhoCorasick(["abcdef"]).search("abc").matches,
    [], "patterns longer than the text never match");
  assert.deepStrictEqual(karpRabinMulti("abc", ["abcdef"]).matches, []);
  assert.deepStrictEqual(new AhoCorasick([]).search("anything").matches, []);
  assert.deepStrictEqual(karpRabinMulti("anything", []).matches, []);
  const single = new AhoCorasick(["ab"]);
  assert.deepStrictEqual(single.search("ababab").matches, [[0, 0], [2, 0], [4, 0]]);
  assert.strictEqual(single.stateCount(), 3);

  assert.throws(() => new AhoCorasick([""]), /empty pattern/);
  assert.throws(() => karpRabinMulti("abc", [""]), /empty pattern/);
}

module.exports = { BASE, MOD, lcgString, modPow, hashOf, directWindowHashes, rollingWindowHashes,
  verify, karpRabinMulti, AhoCorasick, bruteForceMatches };

if (require.main === module) {
  main();
  console.log("PASS");
}
