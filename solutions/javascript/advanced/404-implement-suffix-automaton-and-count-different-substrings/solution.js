/** Task 404 - Implement a suffix automaton and count different substrings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A suffix automaton is the minimal DFA recognizing all substrings of s. Each state stands for
 * an equivalence class of substrings sharing the same end-position set, with `length` the
 * longest member of the class and `link` the state for the longest proper suffix outside it.
 * Every substring in the class is a suffix of the state's longest string with length in
 * (length(link), length], so each class contributes length - length(link) distinct substrings
 * and the total is their sum over all states except the initial one.
 *
 * Anchors: "abab" -> 7, "aaaa" -> 4, "abc" -> 6. Cross-checks: the count matches a brute-force
 * substring set, every substring is accepted by walking transitions from the initial state,
 * generated non-substrings are rejected, and `occurrenceCount` (end-position propagation along
 * suffix links in decreasing-length order) matches a naive scan.
 */
"use strict";

class SuffixAutomaton {
  constructor() {
    this.next = [new Map()];
    this.link = [-1];
    this.length = [0];
    this.isClone = [false];
    this.last = 0;
  }

  extend(ch) {
    const cur = this.length.length;
    this.next.push(new Map());
    this.link.push(0);
    this.length.push(this.length[this.last] + 1);
    this.isClone.push(false);
    let p = this.last;
    while (p !== -1 && !this.next[p].has(ch)) {
      this.next[p].set(ch, cur);
      p = this.link[p];
    }
    if (p === -1) {
      this.link[cur] = 0;
    } else {
      const q = this.next[p].get(ch);
      if (this.length[p] + 1 === this.length[q]) {
        this.link[cur] = q;
      } else {
        const clone = this.length.length;
        this.next.push(new Map(this.next[q]));
        this.link.push(this.link[q]);
        this.length.push(this.length[p] + 1);
        this.isClone.push(true);
        while (p !== -1 && this.next[p].get(ch) === q) {
          this.next[p].set(ch, clone);
          p = this.link[p];
        }
        this.link[q] = clone;
        this.link[cur] = clone;
      }
    }
    this.last = cur;
  }

  stateFor(t) {
    let state = 0;
    for (const ch of t) {
      const nxt = this.next[state].get(ch);
      if (nxt === undefined) return -1;
      state = nxt;
    }
    return state;
  }

  contains(t) {
    return this.stateFor(t) !== -1;
  }

  distinctSubstrings() {
    let total = 0;
    for (let v = 1; v < this.length.length; v++) total += this.length[v] - this.length[this.link[v]];
    return total;
  }

  endposSizes() {
    const states = this.length.length;
    const order = Array.from({ length: states - 1 }, (_, i) => i + 1);
    order.sort((a, b) => this.length[b] - this.length[a]);
    const count = new Array(states).fill(0);
    for (let v = 1; v < states; v++) count[v] = this.isClone[v] ? 0 : 1;
    for (const v of order) count[this.link[v]] += count[v];
    return count;
  }

  occurrenceCount(t) {
    const state = this.stateFor(t);
    return state === -1 ? 0 : this.endposSizes()[state];
  }

  static build(s) {
    const automaton = new SuffixAutomaton();
    for (const ch of s) automaton.extend(ch);
    return automaton;
  }
}

function allSubstrings(s) {
  const seen = new Set();
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) seen.add(s.slice(i, j));
  }
  return seen;
}

function countOccurrences(s, t) {
  if (!t) return 0;
  let count = 0;
  for (let i = 0; i + t.length <= s.length; i++) {
    if (s.startsWith(t, i)) count++;
  }
  return count;
}

function compareSuffixes(s, i, j) {
  const n = s.length;
  while (i < n && j < n) {
    if (s[i] !== s[j]) return s[i] < s[j] ? -1 : 1;
    i++;
    j++;
  }
  if (i === j) return 0;
  return i === n ? -1 : 1;
}

function kasaiLcp(s, sa) {
  const n = s.length;
  const lcp = new Array(n).fill(0);
  const rank = new Array(n).fill(0);
  for (let i = 0; i < n; i++) rank[sa[i]] = i;
  let h = 0;
  for (let i = 0; i < n; i++) {
    const r = rank[i];
    if (r === 0) {
      h = 0;
      continue;
    }
    const j = sa[r - 1];
    while (i + h < n && j + h < n && s[i + h] === s[j + h]) h++;
    lcp[r] = h;
    if (h > 0) h--;
  }
  return lcp;
}

function lcgString(length, alphabet, seed) {
  const chars = [];
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    chars.push(alphabet[seed % alphabet.length]);
  }
  return [chars.join(""), seed];
}

module.exports = { SuffixAutomaton, allSubstrings, countOccurrences, kasaiLcp, lcgString };

if (require.main === module) {
  const assert = require("node:assert");
  const abab = SuffixAutomaton.build("abab");
  assert.strictEqual(abab.distinctSubstrings(), 7);
  assert.strictEqual(allSubstrings("abab").size, 7);

  const aaaa = SuffixAutomaton.build("aaaa");
  assert.strictEqual(aaaa.distinctSubstrings(), 4);

  assert.strictEqual(SuffixAutomaton.build("").distinctSubstrings(), 0);
  assert.strictEqual(SuffixAutomaton.build("a").distinctSubstrings(), 1);
  assert.strictEqual(SuffixAutomaton.build("abc").distinctSubstrings(), 6);
  assert.strictEqual(SuffixAutomaton.build("banana").distinctSubstrings(), 15);
  assert.ok(abab.contains(""));
  assert.ok(!abab.contains("baba"));
  assert.notStrictEqual(abab.stateFor("abab"), -1);
  assert.strictEqual(abab.stateFor("ababa"), -1);

  for (const s of ["", "a", "aa", "ab", "banana", "mississippi", "abababab", "aabbaabb"]) {
    const automaton = SuffixAutomaton.build(s);
    const substrings = allSubstrings(s);
    assert.strictEqual(automaton.distinctSubstrings(), substrings.size, s);
    const sa = Array.from({ length: s.length }, (_, i) => i).sort((a, b) => compareSuffixes(s, a, b));
    const lcp = kasaiLcp(s, sa);
    assert.strictEqual(automaton.distinctSubstrings(),
      (s.length * (s.length + 1)) / 2 - lcp.reduce((a, b) => a + b, 0), s);
  }

  const ababSubstrings = allSubstrings("abab");
  for (const t of ababSubstrings) assert.ok(abab.contains(t), t);
  for (const t of ["c", "bb", "abb", "baa", "ababa", "babab", "xyz"]) assert.ok(!abab.contains(t), t);
  for (const a of ababSubstrings) {
    for (const b of ababSubstrings) {
      const joined = a + b;
      assert.strictEqual(abab.contains(joined), ababSubstrings.has(joined), joined);
    }
  }

  for (const s of ["banana", "abab", "aaaa", "mississippi", "abababab"]) {
    const automaton = SuffixAutomaton.build(s);
    for (const t of allSubstrings(s)) {
      assert.strictEqual(automaton.occurrenceCount(t), countOccurrences(s, t), `${s}/${t}`);
    }
    assert.strictEqual(automaton.occurrenceCount("xx"), 0, s);
  }
  assert.strictEqual(SuffixAutomaton.build("banana").occurrenceCount("ana"), 2);
  assert.strictEqual(SuffixAutomaton.build("banana").occurrenceCount("a"), 3);
  assert.strictEqual(SuffixAutomaton.build("mississippi").occurrenceCount("issi"), 2);

  let seed = 2718;
  for (const alphabet of ["ab", "abc"]) {
    for (let trial = 0; trial < 30; trial++) {
      let s;
      [s, seed] = lcgString(20, alphabet, seed);
      const automaton = SuffixAutomaton.build(s);
      const substrings = allSubstrings(s);
      assert.strictEqual(automaton.distinctSubstrings(), substrings.size, s);
      for (const t of substrings) {
        assert.ok(automaton.contains(t), t);
        assert.strictEqual(automaton.occurrenceCount(t), countOccurrences(s, t), t);
      }
      for (const t of [alphabet[0] + alphabet[alphabet.length - 1] + alphabet[0], "zzz", "q", s + alphabet[0]]) {
        if (!substrings.has(t)) assert.ok(!automaton.contains(t), t);
      }
    }
  }
  console.log("PASS");
}
