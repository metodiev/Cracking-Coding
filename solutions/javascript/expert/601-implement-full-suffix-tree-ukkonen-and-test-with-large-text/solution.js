/** Task 601 - Implement full suffix tree (Ukkonen) and test with large text. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Ukkonen's algorithm builds the explicit suffix tree of `text` in O(n) amortised time. It keeps an
 * implicit tree plus the active point (activeNode, activeEdge, activeLength) and the number of
 * suffixes still to insert (`remaining`); rule 1 opens a new leaf, rule 2 extends the active edge,
 * rule 3 splits an edge, and suffix links keep the walkdown amortised constant. A unique sentinel
 * '\u0000' is appended so that no suffix is a prefix of another: each of the n+1 suffixes then ends
 * in a leaf and the tree is a genuine full suffix tree.
 *
 * Invariant verified below: after processing character i the tree holds exactly the i+1 suffixes of
 * text[0..i]. Because the sentinel occurs once, an edge label never contains it twice, so
 * (a) occurrences(pattern) is the number of leaves below the node where the walk of `pattern` ends
 * and (b) distinct substrings of the text equal the sum of all edge label lengths minus n + 1, the
 * n+1 substrings ending at the sentinel being exactly the ones to discard.
 *
 * Checks: structural invariants (distinct first child labels, depth = parent depth + edge length,
 * leaf count = n+1, suffix links), brute-force cross-verification of occurrences / hasSubstring /
 * distinct substrings on small LCG strings, and a suffix-array + Kasai-LCP oracle on a 4000-character
 * text.
 */
"use strict";

const SENTINEL = "\u0000";

class SuffixTree {
  /** Builds the suffix tree of `text`; throws if the text already contains the sentinel. */
  constructor(text) {
    if (text.includes(SENTINEL)) throw new Error("text must not contain the sentinel character");
    this.body = text;
    this.text = text + SENTINEL;
    const n = this.text.length;
    this.start = [];
    this.end = [];
    this.link = [];
    this.child = [];
    const newNode = (edgeStart, edgeEnd) => {
      this.start.push(edgeStart);
      this.end.push(edgeEnd); // -1 marks "leaf edge: runs to the end of the text"
      this.link.push(0);
      this.child.push(new Map());
      return this.start.length - 1;
    };
    const root = newNode(-1, -1);
    this.link[root] = -1;

    let leafEnd = -1;
    let remaining = 0;
    let activeNode = root;
    let activeEdge = -1;
    let activeLength = 0;
    let lastNewNode = -1;
    for (let i = 0; i < n; i++) {
      leafEnd = i;
      remaining++;
      lastNewNode = -1;
      while (remaining > 0) {
        if (activeLength === 0) activeEdge = i;
        const edgeChar = this.text[activeEdge];
        const existing = this.child[activeNode].get(edgeChar);
        if (existing === undefined) {
          this.child[activeNode].set(edgeChar, newNode(i, -1));
          if (lastNewNode !== -1) {
            this.link[lastNewNode] = activeNode;
            lastNewNode = -1;
          }
        } else {
          const edgeLength = this.edgeLength(existing, leafEnd);
          if (activeLength >= edgeLength) {
            activeEdge += edgeLength; // walk down the whole edge and retry from the node below
            activeLength -= edgeLength;
            activeNode = existing;
            continue;
          }
          if (this.text[this.start[existing] + activeLength] === this.text[i]) {
            if (lastNewNode !== -1 && activeNode !== root) {
              this.link[lastNewNode] = activeNode;
              lastNewNode = -1;
            }
            activeLength++;
            break; // rule 3: the suffix is already represented
          }
          const split = newNode(this.start[existing], this.start[existing] + activeLength - 1);
          this.child[activeNode].set(edgeChar, split);
          this.child[split].set(this.text[i], newNode(i, -1));
          this.start[existing] += activeLength;
          this.child[split].set(this.text[this.start[existing]], existing);
          if (lastNewNode !== -1) this.link[lastNewNode] = split;
          lastNewNode = split;
        }
        remaining--;
        if (activeNode === root && activeLength > 0) {
          activeLength--;
          activeEdge = i - remaining + 1;
        } else if (activeNode !== root) activeNode = this.link[activeNode];
      }
    }
    this.lastIndex = leafEnd;
    this.computeStructure();
  }

  /** Label length of the edge leading into `node`; leaf edges grow with the text. */
  edgeLength(node, leafEnd = this.lastIndex) {
    const stop = this.end[node] === -1 ? leafEnd : this.end[node];
    return stop - this.start[node] + 1;
  }

  /** Precomputes parent/depth/leaf counts in one DFS pass so queries are plain array reads. */
  computeStructure() {
    const nodeCount = this.start.length;
    this.parent = new Array(nodeCount).fill(-1);
    this.edgeLen = new Array(nodeCount).fill(0);
    this.depth = new Array(nodeCount).fill(0);
    this.leafCount = new Array(nodeCount).fill(0);
    const order = [];
    const stack = [0];
    while (stack.length > 0) {
      const node = stack.pop();
      order.push(node);
      for (const next of this.child[node].values()) {
        this.parent[next] = node;
        this.edgeLen[next] = this.edgeLength(next);
        this.depth[next] = this.depth[node] + this.edgeLen[next];
        stack.push(next);
      }
    }
    // `order` lists a parent before its children, so the reverse pass counts leaves bottom-up.
    for (let i = order.length - 1; i >= 0; i--) {
      const node = order[i];
      if (this.child[node].size === 0) this.leafCount[node] = 1;
      else for (const next of this.child[node].values()) this.leafCount[node] += this.leafCount[next];
    }
    this.nodeCount = nodeCount;
    this.edgeCount = nodeCount - 1;
  }

  /** Walks `pattern`; returns the node reached plus the offset inside its incoming edge (0 = at it). */
  match(pattern) {
    let node = 0;
    let offset = 0;
    for (let i = 0; i < pattern.length; i++) {
      const ch = pattern[i];
      if (offset === 0) {
        const next = this.child[node].get(ch);
        if (next === undefined) return null;
        node = next;
        offset = 1;
      } else {
        if (this.text[this.start[node] + offset] !== ch) return null;
        offset++;
      }
      if (offset === this.edgeLen[node]) offset = 0; // pattern ends exactly on a node
    }
    return { node, offset };
  }

  hasSubstring(pattern) {
    return this.match(pattern) !== null;
  }

  /** Occurrences of `pattern`; the leaves below the stop node are the matching suffix starts. */
  occurrences(pattern) {
    if (pattern.length === 0) return this.body.length;
    const hit = this.match(pattern);
    return hit === null ? 0 : this.leafCount[hit.node];
  }

  distinctSubstrings() {
    let total = 0;
    for (let node = 1; node < this.nodeCount; node++) total += this.edgeLen[node];
    return total - (this.body.length + 1);
  }

  /** Deepest node whose label repeats (leaf count >= 2); such a label cannot contain the sentinel. */
  longestRepeatedSubstring() {
    let best = 0;
    for (let node = 1; node < this.nodeCount; node++) {
      if (this.leafCount[node] >= 2 && this.depth[node] > best) best = this.depth[node];
    }
    return best;
  }

  /** Re-reads the edges from the text and checks the structural invariants of the tree. */
  isValid() {
    if (this.child[0].size === 0 && this.body.length > 0) return false;
    let leaves = 0;
    for (let node = 0; node < this.nodeCount; node++) {
      const first = new Set();
      for (const [ch, next] of this.child[node]) {
        if (first.has(ch) || this.parent[next] !== node || this.leafCount[next] <= 0) return false;
        first.add(ch);
        if (this.depth[next] !== this.depth[node] + this.edgeLen[next]) return false;
        if (this.text[this.start[next]] !== ch) return false; // a child edge starts with its key
      }
      const isLeaf = this.child[node].size === 0;
      if (isLeaf) leaves++;
      if (isLeaf && this.text.length - this.start[node] !== this.edgeLength(node)) return false;
      if (this.link[node] > 0 && this.depth[this.link[node]] >= this.depth[node]) return false;
    }
    return leaves === this.body.length + 1 && this.leafCount[0] === this.body.length + 1;
  }
}

function bruteOccurrences(text, pattern) {
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) if (text.startsWith(pattern, i)) count++;
  return count;
}

function bruteDistinctSubstrings(text) {
  const seen = new Set();
  for (let i = 0; i < text.length; i++) {
    for (let j = i + 1; j <= text.length; j++) seen.add(text.slice(i, j));
  }
  return seen.size;
}

/** Independent oracle: prefix-doubling suffix array plus Kasai's LCP array. */
function suffixArrayDoubling(s) {
  const n = s.length;
  if (n === 0) return [];
  const sa = Array.from({ length: n }, (_, i) => i);
  let rank = Array.from(s, (ch) => ch.charCodeAt(0));
  const key = (i, shift) => (i + shift < n ? rank[i + shift] : -1);
  for (let shift = 1; ; shift <<= 1) {
    sa.sort((a, b) => rank[a] - rank[b] || key(a, shift) - key(b, shift));
    const next = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const prev = sa[i - 1];
      const cur = sa[i];
      next[cur] = next[prev] + (rank[prev] !== rank[cur] || key(prev, shift) !== key(cur, shift) ? 1 : 0);
    }
    if (next[sa[n - 1]] === n - 1) return sa;
    rank = next;
  }
}

function lcpKasai(s, sa) {
  const n = s.length;
  const pos = new Array(n).fill(0);
  for (let i = 0; i < n; i++) pos[sa[i]] = i;
  const lcp = new Array(n).fill(0);
  let k = 0;
  for (let i = 0; i < n; i++) {
    if (pos[i] === n - 1) {
      k = 0;
      continue;
    }
    const j = sa[pos[i] + 1];
    while (i + k < n && j + k < n && s[i + k] === s[j + k]) k++;
    lcp[pos[i]] = k;
    if (k > 0) k--;
  }
  return lcp;
}

const oracleDistinct = (s, lcp) => lcp.reduce((acc, value, i) => acc + s.length - i - value, 0);

function lcgString(length, alphabet, seed) {
  let state = seed;
  const chars = [];
  for (let i = 0; i < length; i++) {
    state = (state * 75 + 74) % 65537;
    chars.push(alphabet[state % alphabet.length]);
  }
  return [chars.join(""), state];
}

module.exports = { SuffixTree, bruteOccurrences, bruteDistinctSubstrings, suffixArrayDoubling,
  lcpKasai, oracleDistinct, lcgString, SENTINEL };

if (require.main === module) {
  const assert = require("node:assert");

  const empty = new SuffixTree("");
  assert.ok(empty.isValid() && empty.distinctSubstrings() === 0 && empty.occurrences("a") === 0);
  assert.deepStrictEqual(empty.child.map((m) => m.size), [1, 0]);

  const single = new SuffixTree("a");
  assert.ok(single.isValid() && single.occurrences("a") === 1 && single.occurrences("aa") === 0);
  assert.strictEqual(single.distinctSubstrings(), 1);

  const banana = new SuffixTree("banana");
  assert.ok(banana.isValid() && banana.distinctSubstrings() === 15);
  assert.strictEqual(banana.occurrences("ana"), 2);
  assert.strictEqual(banana.occurrences("a"), 3);
  assert.strictEqual(banana.occurrences("bananas"), 0);
  assert.strictEqual(banana.longestRepeatedSubstring(), 3); // "ana"
  assert.ok(banana.hasSubstring("nan") && !banana.hasSubstring("bn"));

  const repeated = new SuffixTree("a".repeat(200));
  assert.ok(repeated.isValid() && repeated.distinctSubstrings() === 200);
  assert.strictEqual(repeated.longestRepeatedSubstring(), 199);
  assert.strictEqual(repeated.occurrences("a"), 200);
  assert.strictEqual(repeated.occurrences("a".repeat(201)), 0);
  // Unary text: an explicit chain, so 2n + 1 nodes and n + 1 leaves.
  assert.deepStrictEqual([repeated.leafCount[0], repeated.nodeCount, repeated.edgeCount], [201, 401, 400]);

  const mississippi = new SuffixTree("mississippi");
  assert.ok(mississippi.isValid() && mississippi.occurrences("issi") === 2);
  assert.strictEqual(mississippi.distinctSubstrings(), bruteDistinctSubstrings("mississippi"));
  assert.throws(() => new SuffixTree("bad\u0000text"), /sentinel/);

  let seed = 601;
  for (const alphabet of ["ab", "abc", "abcd"]) {
    for (let trial = 0; trial < 10; trial++) {
      let text;
      [text, seed] = lcgString(30, alphabet, seed);
      const tree = new SuffixTree(text);
      const lcp = lcpKasai(text, suffixArrayDoubling(text));
      assert.ok(tree.isValid(), `structure ${text}`);
      assert.strictEqual(tree.distinctSubstrings(), bruteDistinctSubstrings(text), text);
      assert.strictEqual(tree.distinctSubstrings(), oracleDistinct(text, lcp), text);
      assert.strictEqual(tree.longestRepeatedSubstring(), Math.max(...lcp), text);
      for (let i = 0; i < text.length; i++) {
        for (let j = i + 1; j <= Math.min(i + 5, text.length); j++) {
          const pattern = text.slice(i, j);
          assert.ok(tree.hasSubstring(pattern), `${text} ${pattern}`);
          assert.strictEqual(tree.occurrences(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
        }
      }
      for (const absent of ["z", `z${text}`, `${text}z`, alphabet.repeat(text.length + 1)]) {
        assert.strictEqual(bruteOccurrences(text, absent), 0, `${text} ${absent}`); // truly absent
        assert.ok(!tree.hasSubstring(absent) && tree.occurrences(absent) === 0, `${text} ${absent}`);
      }
      assert.strictEqual(tree.occurrences(""), text.length);
    }
  }

  // Large text checked against the suffix array + Kasai LCP oracle instead of brute force.
  const large = [];
  let big;
  [big, seed] = lcgString(4000, "ab", seed);
  large.push(big, "abcab".repeat(700));  for (const text of large) {
    const tree = new SuffixTree(text);
    const lcp = lcpKasai(text, suffixArrayDoubling(text));
    assert.ok(tree.isValid() && tree.leafCount[0] === text.length + 1);
    assert.strictEqual(tree.distinctSubstrings(), oracleDistinct(text, lcp));
    assert.strictEqual(tree.longestRepeatedSubstring(), Math.max(...lcp));
    for (let probe = 0; probe < 40; probe++) {
      const from = (probe * 37) % text.length;
      const pattern = text.slice(from, from + 1 + (probe % 12));
      assert.strictEqual(tree.occurrences(pattern), bruteOccurrences(text, pattern), pattern);
    }
    assert.strictEqual(tree.occurrences(text.slice(0, 11) + "\u0001"), 0);
  }

  console.log(`large text: ${large.map((t) => t.length).join(", ")} characters`);
  console.log("PASS");
}
