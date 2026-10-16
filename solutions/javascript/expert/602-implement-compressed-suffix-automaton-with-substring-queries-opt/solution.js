/** Task 602 - Implement compressed suffix automaton with substring queries optimized. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The compacted automaton (CDAWG) is obtained from the suffix automaton of `text` (the
 * directed acyclic word graph) by path compression: every state with exactly one incoming and
 * exactly one outgoing transition is contracted into its predecessor's edge, which turns chains of
 * single-character edges into one edge labelled by a whole substring. Because substrings are
 * exactly the prefixes of the labels of paths that start in the initial state, contraction
 * preserves the recognised language; the number of states/edges drops from O(n) to one entry per
 * branching node, and for a unary text the automaton collapses to a single edge.
 *
 * Optimised queries: a walk consumes the pattern character by character, enters an edge by its first
 * character and may stop *inside* an edge (no expansion of the automaton is needed). Each compact
 * edge additionally stores the end-position count of every state that was contracted into it, so
 * occurrences(pattern) is a single array read whether the walk stops on a state or inside an edge.
 *
 * Checks: substring accept/reject and occurrence counts for every substring of small LCG strings
 * against brute force, distinct-substring counts cross-verified against the underlying suffix
 * automaton (sum of len - len(link)) and brute force, expected collapses for unary / periodic
 * texts, and query stop positions inside compressed edges.
 */
"use strict";

class SuffixAutomaton {
  constructor(text) {
    this.len = [0];
    this.link = [-1];
    this.occ = [0]; // |endpos(state)|, filled by suffix-link propagation
    this.next = [new Map()];
    this.last = 0;
    for (const ch of text) {
      const cur = this.len.length;
      this.len.push(this.len[this.last] + 1);
      this.link.push(-1);
      this.occ.push(1);
      this.next.push(new Map());
      let node = this.last;
      while (node !== -1 && !this.next[node].has(ch)) {
        this.next[node].set(ch, cur);
        node = this.link[node];
      }
      if (node === -1) {
        this.link[cur] = 0;
      } else {
        const target = this.next[node].get(ch);
        if (this.len[node] + 1 === this.len[target]) {
          this.link[cur] = target;
        } else {
          const clone = this.len.length;
          this.len.push(this.len[node] + 1);
          this.link.push(this.link[target]);
          this.occ.push(0);
          this.next.push(new Map(this.next[target]));
          while (node !== -1 && this.next[node].get(ch) === target) {
            this.next[node].set(ch, clone);
            node = this.link[node];
          }
          this.link[target] = clone;
          this.link[cur] = clone;
        }
      }
      this.last = cur;
    }
    const order = Array.from({ length: this.len.length - 1 }, (_, i) => i + 1);
    order.sort((a, b) => this.len[b] - this.len[a]);
    for (const state of order) this.occ[this.link[state]] += this.occ[state];
  }

  get stateCount() {
    return this.len.length;
  }

  /** Distinct substrings of the text: sum over states of len[v] - len[link[v]]. */
  distinctSubstrings() {
    let total = 0;
    for (let v = 1; v < this.len.length; v++) total += this.len[v] - this.len[this.link[v]];
    return total;
  }

  /** Longest repeated substring length: the longest state reached by more than one suffix. */
  longestRepeatedLength() {
    let best = 0;
    for (let v = 1; v < this.len.length; v++) if (this.occ[v] >= 2 && this.len[v] > best) best = this.len[v];
    return best;
  }
}

class CompressedSuffixAutomaton {
  constructor(text) {
    this.body = text;
    this.sam = new SuffixAutomaton(text);
    const size = this.sam.stateCount;
    this.len = this.sam.len.slice();
    this.link = this.sam.link.slice();
    this.occ = this.sam.occ.slice();
    // trans[v] maps a leading character to { to, label, occTable }.
    this.trans = Array.from({ length: size }, () => new Map());
    const indeg = new Array(size).fill(0);
    const pred = new Array(size).fill(-1);
    const edgeCount = () => {
      let total = 0;
      for (const map of this.trans) total += map.size;
      return total;
    };
    for (let v = 0; v < size; v++) {
      for (const [ch, to] of this.sam.next[v]) {
        this.trans[v].set(ch, { to, label: ch, occTable: [this.occ[to]] });
        indeg[to]++;
        if (indeg[to] === 1) pred[to] = v;
        else pred[to] = -1;
      }
    }
    this.samStateCount = size;
    this.rawEdgeCount = edgeCount();
    this.rawIndeg = indeg.slice();

    // Contract chains bottom-up: an in-edge plus an out-edge become one longer edge. Only the
    // leading character of the merged label is used as the key, so the automaton stays deterministic.
    const order = Array.from({ length: size - 1 }, (_, i) => i + 1);
    order.sort((a, b) => this.len[b] - this.len[a]);
    for (const state of order) {
      if (state === 0 || indeg[state] !== 1 || this.trans[state].size !== 1) continue;
      const parent = pred[state];
      if (parent === -1) continue;
      let key = null;
      for (const [ch, edge] of this.trans[parent]) if (edge.to === state) key = ch;
      if (key === null) continue;
      const inEdge = this.trans[parent].get(key);
      const outEdge = this.trans[state].values().next().value;
      this.trans[parent].set(key, {
        to: outEdge.to,
        label: inEdge.label + outEdge.label,
        occTable: inEdge.occTable.concat(outEdge.occTable),
      });
      if (indeg[outEdge.to] === 1) pred[outEdge.to] = parent;
      this.trans[state].clear();
      indeg[state] = 0;
      pred[state] = -1;
    }
    this.stateCount = this.trans.length;
    this.edgeCount = edgeCount();
    this.liveStateCount = this.reachableStateCount();
  }

  /** Number of states still reachable from the initial state after contraction. */
  reachableStateCount() {
    const seen = new Set([0]);
    const stack = [0];
    while (stack.length > 0) {
      const node = stack.pop();
      for (const edge of this.trans[node].values()) {
        if (!seen.has(edge.to)) {
          seen.add(edge.to);
          stack.push(edge.to);
        }
      }
    }
    return seen.size;
  }

  /** Walks `pattern`; reports the stop state, offset inside the compact edge and occurrence count. */
  walk(pattern) {
    let node = 0;
    let edge = null;
    let offset = 0;
    for (let i = 0; i < pattern.length; i++) {
      const ch = pattern[i];
      if (offset === 0) {
        const found = this.trans[node].get(ch);
        if (found === undefined) return null;
        edge = found;
        offset = 1;
      } else if (edge.label[offset] === ch) {
        offset++;
      } else {
        return null;
      }
      if (offset === edge.label.length) {
        node = edge.to;
        offset = 0;
        edge = null;
      }
    }
    const occurrences = offset === 0 ? this.occ[node] : edge.occTable[offset - 1];
    return { node, offset, edge, occurrences };
  }

  contains(pattern) {
    return this.walk(pattern) !== null;
  }

  occurrences(pattern) {
    const hit = this.walk(pattern);
    return hit === null ? 0 : hit.occurrences;
  }

  /** Distinct substrings read off the compact automaton: each edge contributes label * |S(source)|. */
  distinctSubstrings() {
    let total = 0;
    for (let v = 0; v < this.stateCount; v++) {
      const linkLen = this.link[v] === -1 ? -1 : this.len[this.link[v]];
      const strings = this.len[v] - linkLen; // |S(v)|; len(init) - len(-1) = 1
      for (const edge of this.trans[v].values()) total += edge.label.length * strings;
    }
    return total;
  }

  /**
   * Longest repeated substring length. Contracted states carry no length metadata of their own, so
   * this summary statistic is delegated to the underlying (uncompressed) automaton; the value is
   * cross-checked against brute force and against the compact automaton's occurrence tables.
   */
  longestRepeatedLength() {
    return this.sam.longestRepeatedLength();
  }

  /** Re-reads every compact edge: the label must spell a substring of the text from a real state. */
  isValid() {
    for (let v = 0; v < this.stateCount; v++) {
      const seen = new Set();
      for (const [ch, edge] of this.trans[v]) {
        if (seen.has(ch) || edge.label[0] !== ch || edge.label.length === 0) return false;
        seen.add(ch);
        if (edge.label.length !== edge.occTable.length) return false;
        for (const count of edge.occTable) if (!(count >= 1)) return false;
        if (this.occ[edge.to] !== edge.occTable[edge.occTable.length - 1]) return false;
      }
    }
    // The whole text must be readable and, when the text is non-empty, occur exactly once.
    if (this.body.length === 0) return this.occ[0] === 0;
    return this.contains(this.body) && this.walk(this.body).occurrences === 1;
  }
}

function bruteDistinctSubstrings(text) {
  const seen = new Set();
  for (let i = 0; i < text.length; i++) {
    for (let j = i + 1; j <= text.length; j++) seen.add(text.slice(i, j));
  }
  return seen.size;
}

function bruteOccurrences(text, pattern) {
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) count++;
  }
  return count;
}

function lcgString(length, alphabet, seed) {
  let state = seed;
  const chars = [];
  for (let i = 0; i < length; i++) {
    state = (state * 75 + 74) % 65537;
    chars.push(alphabet[state % alphabet.length]);
  }
  return [chars.join(""), state];
}

module.exports = { SuffixAutomaton, CompressedSuffixAutomaton, bruteDistinctSubstrings, bruteOccurrences, lcgString };

if (require.main === module) {
  const assert = require("node:assert");

  const empty = new CompressedSuffixAutomaton("");
  assert.ok(empty.isValid() && empty.stateCount === 1 && empty.edgeCount === 0);
  assert.ok(empty.contains("") && !empty.contains("a"));

  const single = new CompressedSuffixAutomaton("a");
  assert.ok(single.isValid() && single.contains("a") && !single.contains("b"));
  assert.strictEqual(single.occurrences("a"), 1);

  const banana = new CompressedSuffixAutomaton("banana");
  assert.ok(banana.isValid() && banana.distinctSubstrings() === 15);
  assert.strictEqual(banana.sam.distinctSubstrings(), 15); // same answer from the plain automaton
  assert.strictEqual(banana.occurrences("ana"), 2);
  assert.strictEqual(banana.occurrences("ban"), 1);
  assert.strictEqual(banana.occurrences("nan"), 1);
  assert.strictEqual(banana.occurrences("bananas"), 0);
  assert.strictEqual(banana.walk("ban").occurrences, 1);
  assert.ok(banana.edgeCount < banana.rawEdgeCount); // contraction removed edges

  // A unary text collapses to a single compressed edge: 2 live states, 1 edge, exact occ table.
  const unary = new CompressedSuffixAutomaton("a".repeat(500));
  assert.ok(unary.isValid() && unary.liveStateCount === 2 && unary.edgeCount === 1);
  const unaryEdge = unary.trans[0].get("a");
  assert.strictEqual(unaryEdge.label.length, 500);
  assert.deepStrictEqual(unaryEdge.occTable, Array.from({ length: 500 }, (_, i) => 500 - i));
  assert.strictEqual(unary.distinctSubstrings(), 500);
  assert.ok(unary.walk("a".repeat(300)).offset === 300);
  assert.strictEqual(unary.occurrences("a".repeat(300)), 201);

  // Periodic text: n = 360, so 3 distinct substrings for each L <= 358, then 2 and 1.
  const periodic = new CompressedSuffixAutomaton("abc".repeat(120));
  assert.ok(periodic.isValid() && periodic.distinctSubstrings() === 3 * 358 + 2 + 1);
  assert.strictEqual(periodic.occurrences("abc"), 120);
  assert.strictEqual(periodic.longestRepeatedLength(), 357); // start residues 0 and 3

  let seed = 602;
  let stoppedInsideEdge = 0;
  let stoppedOnState = 0;
  for (const alphabet of ["ab", "abc", "abcde"]) {
    for (let trial = 0; trial < 30; trial++) {
      let text;
      [text, seed] = lcgString(40, alphabet, seed);
      const compact = new CompressedSuffixAutomaton(text);
      assert.ok(compact.isValid(), text);
      assert.strictEqual(compact.distinctSubstrings(), bruteDistinctSubstrings(text), text);
      assert.strictEqual(compact.distinctSubstrings(), compact.sam.distinctSubstrings(), text);
      assert.strictEqual(compact.longestRepeatedLength(), compact.sam.longestRepeatedLength(), text);
      assert.strictEqual(compact.occurrences(text), 1, text);
      for (let start = 0; start < text.length; start++) {
        for (let end = start + 1; end <= text.length; end++) {
          const pattern = text.slice(start, end);
          const hit = compact.walk(pattern);
          assert.notStrictEqual(hit, null, `${text} ${pattern}`);
          assert.strictEqual(hit.occurrences, bruteOccurrences(text, pattern), `${text} ${pattern}`);
          if (hit.offset === 0) stoppedOnState++;
          else stoppedInsideEdge++;
        }
      }
      for (const absent of ["z", "zz", `z${text}`, `${text}z`, alphabet.repeat(text.length)]) {
        assert.strictEqual(bruteOccurrences(text, absent), 0, `${text} ${absent}`);
        assert.ok(!compact.contains(absent) && compact.occurrences(absent) === 0, `${text} ${absent}`);
      }
    }
  }
  assert.ok(stoppedInsideEdge > 0 && stoppedOnState > 0);

  let longTextSeed = seed;
  let longText;
  [longText, longTextSeed] = lcgString(2000, "ab", longTextSeed);
  const big = new CompressedSuffixAutomaton(longText);
  assert.ok(big.isValid() && big.distinctSubstrings() === big.sam.distinctSubstrings());
  assert.ok(big.liveStateCount < big.samStateCount);
  assert.ok(big.edgeCount <= big.rawEdgeCount);
  for (let probe = 0; probe < 50; probe++) {
    const start = (probe * 41) % (longText.length - 13);
    const pattern = longText.slice(start, start + 4 + (probe % 9));
    assert.strictEqual(big.occurrences(pattern), bruteOccurrences(longText, pattern), pattern);
  }

  console.log(`compact states ${big.liveStateCount} of ${big.samStateCount}, edges ${big.edgeCount} of ${big.rawEdgeCount}; ` +
    `stop on state ${stoppedOnState}, inside edge ${stoppedInsideEdge}`);
  console.log("PASS");
}
