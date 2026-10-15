/** Task 431 - Implement suffix automaton for distinct substring queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Query implemented: the distinct-substring count of the automaton's own string, i.e. the sum
 * of len[state] - len[link[state]] over all non-initial states ("abab" -> 7, "aaaa" -> 4).
 * Two further suffix-automaton queries back the accept/reject checks: isSubstring(t) walks the
 * transition function from the initial state, and occurrences(t) reads the end-position count of
 * the reached state (terminal counts are propagated along suffix links in decreasing-length
 * order). Patterns passed to occurrences are non-empty; every answer is cross-checked against a
 * brute-force set of all substrings on LCG-generated strings.
 */
"use strict";

class SuffixAutomaton {
  constructor(text) {
    this.length = [0];
    this.link = [-1];
    this.ends = [0];
    this.next = [new Map()];
    let last = 0;
    for (const ch of text) {
      const cur = this.length.length;
      this.length.push(this.length[last] + 1);
      this.link.push(-1);
      this.ends.push(1);
      this.next.push(new Map());
      let p = last;
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
          this.length.push(this.length[p] + 1);
          this.link.push(this.link[q]);
          this.ends.push(0);
          this.next.push(new Map(this.next[q]));
          while (p !== -1 && this.next[p].get(ch) === q) {
            this.next[p].set(ch, clone);
            p = this.link[p];
          }
          this.link[q] = clone;
          this.link[cur] = clone;
        }
      }
      last = cur;
    }
    const order = [];
    for (let v = 1; v < this.length.length; v++) order.push(v);
    order.sort((a, b) => this.length[b] - this.length[a]);
    for (const state of order) this.ends[this.link[state]] += this.ends[state];
  }

  distinctSubstrings() {
    let total = 0;
    for (let v = 1; v < this.length.length; v++) total += this.length[v] - this.length[this.link[v]];
    return total;
  }

  walk(pattern) {
    let state = 0;
    for (const ch of pattern) {
      const nxt = this.next[state].get(ch);
      if (nxt === undefined) return -1;
      state = nxt;
    }
    return state;
  }

  isSubstring(pattern) {
    return this.walk(pattern) !== -1;
  }

  occurrences(pattern) {
    const state = this.walk(pattern);
    return state === -1 ? 0 : this.ends[state];
  }
}

function bruteDistinctSubstrings(text) {
  const found = new Set();
  for (let i = 0; i < text.length; i++) {
    for (let j = i + 1; j <= text.length; j++) found.add(text.slice(i, j));
  }
  return found.size;
}

function bruteOccurrences(text, pattern) {
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) count++;
  }
  return count;
}

function lcgStrings(count, seed, maxLen = 40, alphabet = "abc") {
  const strings = [];
  for (let s = 0; s < count; s++) {
    const length = 1 + (seed % maxLen);
    const chars = [];
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      chars.push(alphabet[seed % alphabet.length]);
    }
    strings.push(chars.join(""));
  }
  return strings;
}

module.exports = { SuffixAutomaton, bruteDistinctSubstrings, bruteOccurrences, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(new SuffixAutomaton("").distinctSubstrings(), 0);
  assert.strictEqual(new SuffixAutomaton("a").distinctSubstrings(), 1);
  assert.strictEqual(new SuffixAutomaton("abab").distinctSubstrings(), 7);
  assert.strictEqual(new SuffixAutomaton("aaaa").distinctSubstrings(), 4);
  assert.strictEqual(new SuffixAutomaton("abc").distinctSubstrings(), 6);
  assert.strictEqual(new SuffixAutomaton("ababa").distinctSubstrings(), 9);

  const banana = new SuffixAutomaton("banana");
  assert.strictEqual(banana.distinctSubstrings(), 15);
  assert.ok(banana.isSubstring("ban") && banana.isSubstring("ana") && banana.isSubstring("nan"));
  assert.ok(!banana.isSubstring("bn") && !banana.isSubstring("bananas"));
  assert.strictEqual(banana.occurrences("an"), 2);
  assert.strictEqual(banana.occurrences("ana"), 2);
  assert.strictEqual(banana.occurrences("x"), 0);

  const abab = new SuffixAutomaton("abab");
  assert.strictEqual(abab.occurrences("ab"), 2);
  assert.strictEqual(abab.occurrences("ba"), 1);
  assert.strictEqual(abab.occurrences("abab"), 1);

  const aaaa = new SuffixAutomaton("aaaa");
  assert.strictEqual(aaaa.occurrences("a"), 4);
  assert.strictEqual(aaaa.occurrences("aa"), 3);
  assert.strictEqual(aaaa.occurrences("aaa"), 2);
  assert.strictEqual(aaaa.occurrences("aaaa"), 1);

  let rejected = 0;
  let accepted = 0;
  for (const text of lcgStrings(40, 431)) {
    const auto = new SuffixAutomaton(text);
    assert.strictEqual(auto.distinctSubstrings(), bruteDistinctSubstrings(text), text);
    assert.ok(auto.isSubstring(text), text);
    assert.strictEqual(auto.occurrences(text), 1, text);
    for (let start = 0; start < text.length; start++) {
      for (let end = start + 1; end <= Math.min(start + 6, text.length); end++) {
        const pattern = text.slice(start, end);
        assert.ok(auto.isSubstring(pattern), `${text} ${pattern}`);
        assert.strictEqual(auto.occurrences(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
        accepted++;
      }
    }
    for (const mutation of ["d", "bd", "abz", text + "d"]) {
      if (auto.isSubstring(mutation)) {
        assert.strictEqual(auto.occurrences(mutation), bruteOccurrences(text, mutation));
      } else {
        assert.strictEqual(bruteOccurrences(text, mutation), 0);
        rejected++;
      }
    }
  }
  assert.ok(rejected > 0 && accepted > 0);
  console.log("PASS");
}
