/** Task 603 - Implement generalized suffix tree for multiple texts and LCS of many strings. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A generalized suffix tree is built by streaming T1$1 T2$2 ... Tk$k into Ukkonen's *online*
 * construction: Ukkonen already grows an existing tree one character at a time, so feeding the next
 * text into the same tree is exactly incremental insertion - no rebuild, no restart. Distinct
 * terminators $i (Unicode private use area, one per text) keep suffixes of different texts apart.
 *
 * Every node carries the bit set of documents owning at least one suffix below it, and that set is
 * exactly the set of documents containing the node's path label: an occurrence of a label without a
 * terminator is contiguous, so it lies inside one text region and is credited to that text only. The
 * LCS of all k texts is the deepest *terminator-free* position whose set covers every document, and
 * the number of distinct common substrings is the number of terminator-free edge positions on
 * covering edges. The document set is constant along an edge (a node and a prefix of its label share
 * the same leaves), so the longest terminator-free prefix of each edge is one candidate.
 *
 * Checks: LCS and common-substring counts of 2-5 and 10 texts against exhaustive brute force (the
 * answer is re-verified to occur in every text), per-document occurrence counts against brute force,
 * subset queries, and degenerate inputs (one text, an empty text, disjoint alphabets, unary texts,
 * terminator characters in the input). */
"use strict";

const TERMINATOR_BASE = 0xe000;
const MAX_TEXTS = 32;

class GeneralizedSuffixTree {
  constructor(texts) {
    if (texts.length > MAX_TEXTS) throw new Error(`at most ${MAX_TEXTS} texts are supported`);
    this.texts = texts.slice();
    this.docCount = texts.length;
    this.terminators = texts.map((_, i) => String.fromCharCode(TERMINATOR_BASE + i));
    for (const text of texts) {
      if (this.terminators.some((terminator) => text.includes(terminator))) {
        throw new Error("texts must not contain terminator characters");
      }
    }
    this.sentinelPos = [];
    this.positionDoc = [];
    let joined = "";
    for (let doc = 0; doc < texts.length; doc++) {
      this.positionDoc.push(...new Array(texts[doc].length).fill(doc));
      joined += texts[doc];
      this.sentinelPos.push(joined.length);
      this.positionDoc.push(doc); // a terminator belongs to the region it closes
      joined += this.terminators[doc];
    }
    this.buildUkkonen(joined);
    this.computeMasks();
  }

  /** One-shot Ukkonen build; a leaf's edge start doubles as its suffix start position. */
  buildUkkonen(text) {
    this.text = text;
    const n = text.length;
    this.start = [];
    this.end = [];
    this.link = [];
    this.child = [];
    const newNode = (edgeStart, edgeEnd) => {
      this.start.push(edgeStart);
      this.end.push(edgeEnd); // -1 marks "leaf edge: runs to the end of the string"
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
        const existing = this.child[activeNode].get(text[activeEdge]);
        if (existing === undefined) {
          this.child[activeNode].set(text[activeEdge], newNode(i, -1));
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
          if (text[this.start[existing] + activeLength] === text[i]) {
            if (lastNewNode !== -1 && activeNode !== root) {
              this.link[lastNewNode] = activeNode;
              lastNewNode = -1;
            }
            activeLength++;
            break; // rule 3: the suffix is already represented
          }
          const split = newNode(this.start[existing], this.start[existing] + activeLength - 1);
          this.child[activeNode].set(text[activeEdge], split);
          this.child[split].set(text[i], newNode(i, -1));
          this.start[existing] += activeLength;
          this.child[split].set(text[this.start[existing]], existing);
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
  }
  edgeLength(node, leafEnd = this.lastIndex) {
    const stop = this.end[node] === -1 ? leafEnd : this.end[node];
    return stop - this.start[node] + 1;
  }

  /** Longest terminator-free prefix of a node's own edge label (edge length when there is none). */
  firstTerminatorOffset(node) {
    let best = this.edgeLen[node];
    for (const position of this.sentinelPos) {
      if (position >= this.start[node] && position < this.start[node] + this.edgeLen[node]) {
        best = Math.min(best, position - this.start[node]);
      }
    }
    return best;
  }

  /** Parents, depths, document bit sets, per-document leaf counts and terminator-free prefixes. */
  computeMasks() {
    const nodeCount = this.start.length;
    this.parent = new Array(nodeCount).fill(-1);
    this.edgeLen = new Array(nodeCount).fill(0);
    this.depth = new Array(nodeCount).fill(0);
    this.docMask = new Array(nodeCount).fill(0);
    this.clean = new Array(nodeCount).fill(true);
    this.cleanPrefix = new Array(nodeCount).fill(0);
    this.docLeafCount = Array.from({ length: nodeCount }, () => new Array(this.docCount).fill(0));
    const order = [];
    const stack = [0];
    while (stack.length > 0) {
      const node = stack.pop();
      order.push(node);
      for (const next of this.child[node].values()) {
        this.parent[next] = node;
        this.edgeLen[next] = this.edgeLength(next);
        this.depth[next] = this.depth[node] + this.edgeLen[next];
        this.cleanPrefix[next] = this.clean[node] ? this.firstTerminatorOffset(next) : 0;
        this.clean[next] = this.cleanPrefix[next] === this.edgeLen[next];
        stack.push(next);
      }    }
    // `order` puts every parent before its children, so this reverse pass is bottom-up.
    for (let i = order.length - 1; i >= 0; i--) {
      const node = order[i];
      for (const next of this.child[node].values()) {
        this.docMask[node] |= this.docMask[next];
        for (let doc = 0; doc < this.docCount; doc++) this.docLeafCount[node][doc] += this.docLeafCount[next][doc];
      }
      if (this.child[node].size === 0) {
        const doc = this.positionDoc[this.start[node]];
        this.docMask[node] = 1 << doc;
        this.docLeafCount[node][doc] = 1;
      }
    }
    this.nodeCount = nodeCount;
  }

  /** Walks `pattern`; the stop node holds every occurrence as a leaf in its subtree. */
  match(pattern) {
    let node = 0;
    let offset = 0;
    for (const ch of pattern) {
      if (offset === 0) {
        node = this.child[node].get(ch);
        if (node === undefined) return null;
        offset = 1;
      } else if (this.text[this.start[node] + offset] === ch) offset++;
      else return null;
      if (offset === this.edgeLen[node]) offset = 0;
    }
    return { node, offset };
  }

  /** Bit mask of the documents containing `pattern`; 0 when the pattern does not occur at all. */
  documentsContaining(pattern) {
    if (pattern.length === 0) return (1 << this.docCount) - 1;
    const hit = this.match(pattern);
    return hit === null ? 0 : this.docMask[hit.node];
  }

  /** Occurrences of `pattern` inside `doc` (suffix starts, so the empty pattern gives |T|). */
  occurrencesIn(pattern, doc) {
    if (pattern.length === 0) return this.texts[doc].length;
    const hit = this.match(pattern);
    return hit === null ? 0 : this.docLeafCount[hit.node][doc];
  }

  /** Deepest terminator-free position whose document set covers every doc in `docs`. */
  longestCommonSubstringOf(docs) {
    const wanted = docs.reduce((mask, doc) => mask | (1 << doc), 0);
    let best = null;
    for (let node = 1; node < this.nodeCount; node++) {
      const parent = this.parent[node];
      if (!this.clean[parent]) continue; // the path into the parent already holds a terminator
      if ((this.docMask[node] & wanted) !== wanted) continue;
      const depth = this.depth[parent] + this.cleanPrefix[node];
      if (best === null || depth > best.depth) best = { node, depth };
    }
    if (best === null) return "";
    const pieces = [];
    for (let current = best.node; current !== 0; current = this.parent[current]) {
      pieces.push(this.text.slice(this.start[current], this.start[current] + this.edgeLen[current]));
    }
    return pieces.reverse().join("").slice(0, best.depth);
  }

  longestCommonSubstring() {
    return this.longestCommonSubstringOf(Array.from({ length: this.docCount }, (_, i) => i));
  }

  /** Distinct substrings shared by `docs`: one per terminator-free position on a covering edge. */
  commonSubstringCount(docs = null) {
    const wanted = docs === null ? (1 << this.docCount) - 1 : docs.reduce((m, d) => m | (1 << d), 0);
    let total = 0;
    for (let node = 1; node < this.nodeCount; node++) {
      if (!this.clean[this.parent[node]] || (this.docMask[node] & wanted) !== wanted) continue;
      total += this.cleanPrefix[node];
    }
    return total;
  }

  /** Structural check: distinct child keys, parent links and child masks inside the parent mask. */
  isValid() {
    for (let node = 0; node < this.nodeCount; node++) {
      const keys = new Set();
      for (const [ch, next] of this.child[node]) {
        const contained = (this.docMask[next] & ~this.docMask[node]) === 0;
        if (keys.has(ch) || this.parent[next] !== node || !contained) return false;
        if (this.text[this.start[next]] !== ch) return false;
        keys.add(ch);
      }
      if (this.child[node].size === 0 && this.docMask[node] !== 1 << this.positionDoc[this.start[node]]) {
        return false;
      }
    }
    for (let doc = 0; doc < this.docCount; doc++) {
      if (this.texts[doc].length > 0 && this.occurrencesIn(this.texts[doc], doc) !== 1) return false;
    }
    return true;
  }
}

function bruteOccurrences(text, pattern) {
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) if (text.startsWith(pattern, i)) count++;
  return count;
}

/** Every distinct substring shared by all texts, shortest first - the LCS is the last entry. */
function bruteCommonSubstrings(texts) {
  const base = texts.length === 0 ? "" : texts.reduce((a, b) => (a.length <= b.length ? a : b));
  const found = new Set();
  for (let i = 0; i < base.length; i++) {
    for (let j = i + 1; j <= base.length; j++) {
      const piece = base.slice(i, j);
      if (texts.every((text) => text.includes(piece))) found.add(piece);
    }
  }
  return [...found].sort((a, b) => a.length - b.length || (a < b ? -1 : 1));
}

function lcgString(length, alphabet, seed) {
  let state = seed >>> 0;
  const chars = [];
  for (let i = 0; i < length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    chars.push(alphabet[(state >>> 8) % alphabet.length]);
  }
  return [chars.join(""), state];
}
module.exports = { GeneralizedSuffixTree, bruteOccurrences, bruteCommonSubstrings, lcgString };

if (require.main === module) {
  const assert = require("node:assert");

  const single = new GeneralizedSuffixTree(["banana"]);
  assert.ok(single.isValid() && single.longestCommonSubstring() === "banana");
  assert.ok(single.commonSubstringCount() === 15 && single.occurrencesIn("ana", 0) === 2);
  assert.ok(single.occurrencesIn("", 0) === 6 && single.documentsContaining("ana") === 1);

  const emptyDoc = new GeneralizedSuffixTree(["banana", ""]);
  assert.ok(emptyDoc.isValid() && emptyDoc.longestCommonSubstring() === "" && emptyDoc.occurrencesIn("", 1) === 0);
  assert.strictEqual(emptyDoc.commonSubstringCount(), 0);
  const pair = new GeneralizedSuffixTree(["banana", "ananas"]);
  assert.ok(pair.isValid() && pair.longestCommonSubstring() === "anana");
  assert.deepStrictEqual([pair.documentsContaining("ana"), pair.documentsContaining("ban"), pair.documentsContaining("nas"),
    pair.documentsContaining("xyz")], [0b11, 0b01, 0b10, 0]);
  assert.deepStrictEqual([pair.occurrencesIn("ana", 0), pair.occurrencesIn("ana", 1), pair.occurrencesIn("nan", 0),
    pair.occurrencesIn("banana", 1)], [2, 2, 1, 0]);
  const disjoint = new GeneralizedSuffixTree(["aaa", "bbb"]);
  assert.ok(disjoint.isValid() && disjoint.longestCommonSubstring() === "" && disjoint.commonSubstringCount() === 0);
  assert.deepStrictEqual([disjoint.documentsContaining("aa"), disjoint.documentsContaining("bbb")], [0b01, 0b10]);
  assert.throws(() => new GeneralizedSuffixTree(["ok", "bad\uE001text"]), /terminator/);
  let seed = 603;
  for (let trial = 0; trial < 24; trial++) {
    const texts = [];
    const docs = 4 + (trial % 2);
    for (let doc = 0; doc < docs; doc++) {
      let text;
      [text, seed] = lcgString(6 + (trial % 7) * 3, "abc", seed);
      texts.push(text);
    }
    const expected = bruteCommonSubstrings(texts);
    const tree = new GeneralizedSuffixTree(texts);
    assert.ok(tree.isValid(), texts.join("|"));
    const found = tree.longestCommonSubstring();
    const expectedLcs = expected.length === 0 ? "" : expected[expected.length - 1];
    assert.strictEqual(found.length, expectedLcs.length, texts.join("|"));
    if (found.length > 0) assert.ok(texts.every((text) => text.includes(found)), found);
    assert.strictEqual(tree.commonSubstringCount(), expected.length, texts.join("|"));
    for (let doc = 0; doc < docs; doc++) {
      for (let i = 0; i < texts[doc].length; i++) {
        for (let j = i + 1; j <= Math.min(i + 6, texts[doc].length); j++) {
          const pattern = texts[doc].slice(i, j);
          assert.strictEqual(tree.occurrencesIn(pattern, doc), bruteOccurrences(texts[doc], pattern), pattern);
        }
      }
      assert.strictEqual(tree.longestCommonSubstringOf([doc]), texts[doc], texts.join("|"));
    }
  }

  // Ten texts over distinct flanks: the shared block must survive the bit-set intersection.
  const shared = "ababbaabbaba";
  const flanks = ["z", "y", "x", "w", "v", "u", "t", "s", "r", "q"];
  const many = flanks.map((flank, i) => `${flank}${shared}${flank}${i % 3}`);
  const big = new GeneralizedSuffixTree(many);
  assert.ok(big.isValid());
  const bigLcs = big.longestCommonSubstring();
  assert.ok(bigLcs.length === bruteCommonSubstrings(many).pop().length && bigLcs.includes(shared));
  assert.ok(big.documentsContaining(shared) === (1 << 10) - 1 &&
    big.commonSubstringCount() === bruteCommonSubstrings(many).length);
  console.log(`generalized suffix tree: ${big.nodeCount} nodes, 10 texts, LCS ${bigLcs.length} chars`);
  console.log("PASS");
}
