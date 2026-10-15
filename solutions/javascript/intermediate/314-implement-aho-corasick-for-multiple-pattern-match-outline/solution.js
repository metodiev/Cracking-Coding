/** Task 314 - Aho-Corasick multi-pattern matching (outline). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

class AhoCorasick {
  /** Patterns are assumed non-empty. */
  constructor(patterns) {
    this.patterns = [...patterns];
    this.goto = [new Map()];
    this.fail = [0];
    this.output = [new Set()];
    for (const pattern of this.patterns) {
      let node = 0;
      for (const char of pattern) {
        if (!this.goto[node].has(char)) {
          this.goto[node].set(char, this.goto.length);
          this.goto.push(new Map());
          this.fail.push(0);
          this.output.push(new Set());
        }
        node = this.goto[node].get(char);
      }
      this.output[node].add(pattern);
    }
    // BFS over the trie: depth-1 states already fail to the root
    const queue = [...this.goto[0].values()];
    while (queue.length > 0) {
      const node = queue.shift();
      for (const pattern of this.output[this.fail[node]]) this.output[node].add(pattern);
      for (const [char, child] of this.goto[node]) {
        let f = this.fail[node];
        while (f !== 0 && !this.goto[f].has(char)) f = this.fail[f];
        this.fail[child] = this.goto[f].has(char) ? this.goto[f].get(char) : 0;
        queue.push(child);
      }
    }
  }

  /** pattern -> ascending list of start indices (overlapping matches included). */
  search(text) {
    const found = new Map();
    for (const pattern of this.patterns) found.set(pattern, []);
    let node = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      while (node !== 0 && !this.goto[node].has(char)) node = this.fail[node];
      node = this.goto[node].has(char) ? this.goto[node].get(char) : 0;
      for (const pattern of this.output[node]) found.get(pattern).push(i - pattern.length + 1);
    }
    return found;
  }
}

/** Independent oracle: scan each pattern on its own. */
function naiveSearchAll(text, patterns) {
  const found = new Map();
  for (const pattern of patterns) {
    const hits = [];
    for (let i = 0; i + pattern.length <= text.length; i++) {
      if (text.slice(i, i + pattern.length) === pattern) hits.push(i);
    }
    found.set(pattern, hits);
  }
  return found;
}

function allWords(alphabet, maxLength) {
  const words = [];
  let frontier = [""];
  for (let depth = 0; depth < maxLength; depth++) {
    const next = [];
    for (const word of frontier) for (const ch of alphabet) next.push(word + ch);
    words.push(...next);
    frontier = next;
  }
  return words;
}

function formatResults(result) {
  return JSON.stringify([...result.entries()]);
}

function lcgStrings(length, count, alphabet = "ab", seed = 3140) {
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

module.exports = { AhoCorasick, naiveSearchAll, allWords, formatResults, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  // Classic example: hand-checked against "ushers" (u0 s1 h2 e3 r4 s5)
  const classic = ["he", "she", "his", "hers"];
  const automaton = new AhoCorasick(classic);
  assert.strictEqual(formatResults(automaton.search("ushers")), '[["he",[2]],["she",[1]],["his",[]],["hers",[2]]]');
  assert.strictEqual(formatResults(automaton.search("ushers")), formatResults(naiveSearchAll("ushers", classic)));

  // Overlapping matches are all reported
  assert.strictEqual(formatResults(new AhoCorasick(["aa", "aaa"]).search("aaaa")), '[["aa",[0,1,2]],["aaa",[0,1]]]');
  const single = new AhoCorasick(["a", "ab"]);
  assert.strictEqual(formatResults(single.search("ab")), '[["a",[0]],["ab",[0]]]');
  assert.strictEqual(formatResults(single.search("")), '[["a",[]],["ab",[]]]');

  for (const alphabet of ["ab", "abc"]) {
    const patternSets = [
      allWords(alphabet, 3),
      [1, 2, 3].map((k) => alphabet[0].repeat(k)),
      allWords(alphabet, 2),
    ];
    const texts = ["", alphabet, alphabet.repeat(4), alphabet[0].repeat(6), ...lcgStrings(18, 20, alphabet)];
    for (const patterns of patternSets) {
      const ac = new AhoCorasick(patterns);
      for (const text of texts) {
        const result = ac.search(text);
        assert.strictEqual(formatResults(result), formatResults(naiveSearchAll(text, patterns)),
          `${patterns} / ${text}`);
        for (const [pattern, hits] of result) {
          for (const start of hits) {                        // every report is a real match
            assert.strictEqual(text.slice(start, start + pattern.length), pattern, `${text} ${pattern}`);
          }
        }
      }
    }
  }

  console.log("PASS");
}
