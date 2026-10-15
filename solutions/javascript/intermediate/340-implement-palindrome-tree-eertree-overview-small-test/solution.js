/** Task 340 - Implement palindrome tree (Eertree) overview & small test. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

/**
 * Eertree over a string: two roots (length -1 imaginary, length 0 empty), one node per
 * distinct palindromic substring, each with a suffix link to its longest proper palindromic
 * suffix. Every position finds (or creates) its longest palindromic suffix and counts one
 * occurrence; propagating counts along suffix links from the longest node down turns them
 * into full occurrence counts. Cross-checked against brute-force palindrome sets/counts.
 */

class Eertree {
  constructor(text) {
    this.text = text;
    // Node 0 = imaginary root (length -1), node 1 = empty root (length 0).
    this.length = [-1, 0];
    this.link = [0, 0];
    this.next = [new Map(), new Map()];
    this.occurrences = [0, 0];
    this.palindrome = [null, ""];
    this.last = 1;
    for (let position = 0; position < text.length; position++) {
      this.extend(position, text[position]);
    }
    // Longest nodes first: a node's stored count (times it was the longest palindromic
    // suffix) is added into its suffix link to become the true occurrence count.
    const nodes = Array.from({ length: this.length.length - 2 }, (_, i) => i + 2);
    nodes.sort((a, b) => this.length[b] - this.length[a]);
    for (const node of nodes) {
      this.occurrences[this.link[node]] += this.occurrences[node];
    }
  }

  extend(position, ch) {
    let current = this.last;
    for (;;) {
      const index = position - 1 - this.length[current];
      if (index >= 0 && this.text[index] === ch) break;
      current = this.link[current];
    }

    if (this.next[current].has(ch)) {
      this.last = this.next[current].get(ch);
      this.occurrences[this.last] += 1;
      return;
    }

    const node = this.length.length;
    this.length.push(this.length[current] + 2);
    this.next.push(new Map());
    this.palindrome.push(this.length[node] === 1
      ? ch : ch + this.palindrome[current] + ch);
    if (this.length[node] === 1) {
      this.link.push(1);
    } else {
      let candidate = this.link[current];
      for (;;) {
        const index = position - 1 - this.length[candidate];
        if (index >= 0 && this.text[index] === ch) break;
        candidate = this.link[candidate];
      }
      this.link.push(this.next[candidate].get(ch));
    }
    this.next[current].set(ch, node);
    this.occurrences.push(1);
    this.last = node;
  }

  distinctCount() {
    return this.length.length - 2;
  }

  /** Map palindrome -> full occurrence count for every distinct palindromic substring. */
  counts() {
    const result = new Map();
    for (let node = 2; node < this.length.length; node++) {
      result.set(this.palindrome[node], this.occurrences[node]);
    }
    return result;
  }
}

function bruteDistinct(text) {
  const found = new Set();
  for (let start = 0; start < text.length; start++) {
    for (let end = start + 1; end <= text.length; end++) {
      const piece = text.slice(start, end);
      if (piece === [...piece].reverse().join("")) found.add(piece);
    }
  }
  return found;
}

function bruteCounts(text) {
  const counts = new Map();
  for (const piece of bruteDistinct(text)) {
    let total = 0;
    for (let start = 0; start + piece.length <= text.length; start++) {
      if (text.slice(start, start + piece.length) === piece) total += 1;
    }
    counts.set(piece, total);
  }
  return counts;
}

/** Stable "piece=count;" rendering used to compare maps with deepStrictEqual. */
function render(counts) {
  return [...counts.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([piece, count]) => `${piece}=${count};`).join("");
}

module.exports = { Eertree, bruteDistinct, bruteCounts, render };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked: "abba" has distinct palindromes {a, b, bb, abba}.
  const tree = new Eertree("abba");
  assert.strictEqual(tree.distinctCount(), 4);
  assert.strictEqual(render(tree.counts()), "a=2;abba=1;b=2;bb=1;");
  assert.deepStrictEqual([...tree.counts().keys()].sort(), ["a", "abba", "b", "bb"]);
  assert.strictEqual(render(tree.counts()), render(bruteCounts("abba")));
  assert.strictEqual(tree.distinctCount(), bruteDistinct("abba").size);

  // "aaaa": four distinct palindromes with counts 4, 3, 2, 1.
  const four = new Eertree("aaaa");
  assert.strictEqual(four.distinctCount(), 4);
  assert.strictEqual(render(four.counts()), "a=4;aa=3;aaa=2;aaaa=1;");
  assert.strictEqual(render(four.counts()), render(bruteCounts("aaaa")));

  // Edge cases.
  assert.strictEqual(new Eertree("").distinctCount(), 0);
  assert.strictEqual(new Eertree("").counts().size, 0);
  const single = new Eertree("x");
  assert.strictEqual(single.distinctCount(), 1);
  assert.strictEqual(render(single.counts()), "x=1;");
  assert.strictEqual(new Eertree("abcd").distinctCount(), 4);
  assert.strictEqual(render(new Eertree("abcd").counts()), "a=1;b=1;c=1;d=1;");
  assert.strictEqual(render(new Eertree("aba").counts()), "a=2;aba=1;b=1;");
  assert.strictEqual(render(new Eertree("abccba").counts()), "a=2;abccba=1;b=2;bccb=1;c=2;cc=1;");

  // LCG strings: distinct palindromes and every occurrence count must match brute force.
  let seed = 1717;
  for (let trial = 0; trial < 60; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const size = seed % 40;
    seed = (seed * 75 + 74) % 65537;
    const alphabet = seed % 2 === 0 ? "ab" : "abc";
    let text = "";
    for (let i = 0; i < size; i++) {
      seed = (seed * 75 + 74) % 65537;
      text += alphabet[seed % alphabet.length];
    }
    const random = new Eertree(text);
    const expected = bruteCounts(text);
    assert.strictEqual(render(random.counts()), render(expected), `random counts ${trial} ${text}`);
    assert.strictEqual(random.distinctCount(), expected.size, `random distinct ${trial} ${text}`);
    for (const [piece, count] of random.counts()) {
      assert.strictEqual(piece, [...piece].reverse().join(""), "palindrome check");
      assert.ok(count >= 1 && piece.length <= text.length, "count bounds");
    }
  }
  console.log("PASS");
}
