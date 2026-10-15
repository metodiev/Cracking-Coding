/** Task 311 - Suffix automaton (basic conceptual, small cases). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

class SuffixAutomaton {
  constructor() {
    this.length = [0];               // len[v]
    this.link = [-1];                // suffix link; -1 marks the initial state
    this.transitions = [new Map()];
    this.last = 0;
  }

  extend(char) {
    const cur = this.length.length;
    this.length.push(this.length[this.last] + 1);
    this.link.push(0);
    this.transitions.push(new Map());
    let p = this.last;
    while (p !== -1 && !this.transitions[p].has(char)) {
      this.transitions[p].set(char, cur);
      p = this.link[p];
    }
    if (p === -1) {
      this.link[cur] = 0;
    } else {
      const q = this.transitions[p].get(char);
      if (this.length[p] + 1 === this.length[q]) {
        this.link[cur] = q;
      } else {
        const clone = this.length.length;
        this.length.push(this.length[p] + 1);
        this.link.push(this.link[q]);
        this.transitions.push(new Map(this.transitions[q]));
        while (p !== -1 && this.transitions[p].get(char) === q) {
          this.transitions[p].set(char, clone);
          p = this.link[p];
        }
        this.link[q] = clone;
        this.link[cur] = clone;
      }
    }
    this.last = cur;
  }

  /** DFA walk from the initial state; accepts "" vacuously (the empty substring). */
  accepts(word) {
    let state = 0;
    for (const char of word) {
      if (!this.transitions[state].has(char)) return false;
      state = this.transitions[state].get(char);
    }
    return true;
  }

  distinctSubstrings() {
    let total = 0;
    for (let v = 1; v < this.length.length; v++) total += this.length[v] - this.length[this.link[v]];
    return total;
  }
}

function buildSuffixAutomaton(s) {
  const automaton = new SuffixAutomaton();
  for (const char of s) automaton.extend(char);
  return automaton;
}

/** The independent oracle: every substring of s, collected in a set. */
function bruteSubstrings(s) {
  const out = new Set();
  for (let i = 0; i < s.length; i++) {
    for (let j = i + 1; j <= s.length; j++) out.add(s.slice(i, j));
  }
  return out;
}

function allWords(alphabet, maxLength) {
  const words = [""];
  let frontier = [""];
  for (let depth = 0; depth < maxLength; depth++) {
    const next = [];
    for (const word of frontier) for (const ch of alphabet) next.push(word + ch);
    words.push(...next);
    frontier = next;
  }
  return words;
}

function lcgStrings(length, count, alphabet = "ab", seed = 3110) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += alphabet[seed % alphabet.length];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { SuffixAutomaton, buildSuffixAutomaton, bruteSubstrings, allWords, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  // "abab" has exactly the 7 distinct substrings a, b, ab, ba, aba, bab, abab
  const abab = buildSuffixAutomaton("abab");
  assert.deepStrictEqual([...bruteSubstrings("abab")].sort(), ["a", "ab", "aba", "abab", "b", "ba", "bab"]);
  assert.strictEqual(abab.distinctSubstrings(), 7);

  // "aaaa" has 4: a, aa, aaa, aaaa
  const aaaa = buildSuffixAutomaton("aaaa");
  assert.strictEqual(bruteSubstrings("aaaa").size, 4);
  assert.strictEqual(aaaa.distinctSubstrings(), 4);

  const banana = buildSuffixAutomaton("banana");
  assert.strictEqual(banana.distinctSubstrings(), 15);
  assert.strictEqual(bruteSubstrings("banana").size, 15);

  const empty = buildSuffixAutomaton("");
  assert.strictEqual(empty.distinctSubstrings(), 0);
  assert.ok(empty.accepts(""));

  assert.ok(!abab.accepts("bb") && !abab.accepts("baa") && !abab.accepts("baab"));
  assert.ok(abab.accepts("bab"));

  function verify(s, alphabet) {
    const automaton = buildSuffixAutomaton(s);
    const substrings = bruteSubstrings(s);
    // 1) the len/link formula agrees with the brute-force substring set
    assert.strictEqual(automaton.distinctSubstrings(), substrings.size, s);
    // 2) every substring of s is accepted by walking the automaton
    for (const word of substrings) assert.ok(automaton.accepts(word), `${s} ${word}`);
    // 3) every short word over the alphabet is accepted iff it really is a substring
    //    (the empty word is accepted vacuously; the brute set only holds non-empty ones)
    for (const word of allWords(alphabet, 4)) {
      if (word === "") assert.ok(automaton.accepts(word), s);
      else assert.strictEqual(automaton.accepts(word), substrings.has(word), `${s} ${word}`);
    }
    // 4) the state count stays within the textbook 2n bound
    assert.ok(automaton.length.length <= 2 * s.length + 1, s);
  }

  const fixed = [["a", "ab"], ["ab", "ab"], ["abab", "ab"], ["banana", "ban"], ["mississippi", "mips"]];
  for (const [s, alphabet] of fixed) verify(s, alphabet);
  for (const s of lcgStrings(10, 12, "ab")) verify(s, "ab");
  for (const s of lcgStrings(9, 8, "abc")) verify(s, "abc");

  console.log("PASS");
}
