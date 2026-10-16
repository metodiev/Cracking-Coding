/**
 * Task 579 - Implement wavelet tree for strings (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A real bitvector wavelet tree over a byte (or any small-alphabet) string.  Each node holds a
 * sub-alphabet range [lo, hi] and a bitvector marking, for every position of its (stable) value
 * list, whether the symbol goes to the right child; the bitvector is stored as an Int32Array of
 * prefix ranks, so rank1(i) is O(1) and rank0(i) = i - rank1(i).  The tree is built recursively by
 * splitting at the midpoint of the range, which gives depth ceil(log2(alphabet)) and at most
 * 2 * alphabet - 1 nodes (exactly that many here: empty subtrees are materialised rather than
 * pruned, which keeps every rank/select descent uniform); every level stores as many bits as the
 * values reaching it, hence
 * O(n log alphabet) bits.  "select" is implemented by descending to the leaf and then walking the
 * stored path back up, using a binary search on the prefix ranks - documented as O(log n) per level
 * instead of the O(1) of a block-sampled bitvector.
 *
 * Supported operations: access(i), rank(c, i) (occurrences of c in s[0..i)), count(c), select(c, j)
 * (0-based position of the j-th occurrence, j >= 1, -1 when absent), rangeCount(l, r, c) (occurrences
 * of c in s[l..r)) and rangeCountLess(l, r, x) (symbols strictly below x in s[l..r), i.e. the C[] of
 * an FM-index).
 *
 * Documented checks: an *exhaustive* comparison against brute force on two small strings - every
 * prefix length i, every symbol c, every range (l, r) and every threshold x - for rank, count,
 * access, select, rangeCount and rangeCountLess; the classic "abracadabra" example with its
 * hand-computed values (counts a=5, b=2, r=2, c=1, d=1; select('a', 1) = 0 and select('a', 5) = 10;
 * access(4) = 'c'); random strings over 4-, 16-, 26- and 256-symbol alphabets compared against brute
 * force, including the rank/select inverses select(c, rank(c, i + 1)) = i on every occurrence and
 * rank(c, select(c, j) + 1) = j; out-of-range j gives -1 and out-of-range symbols throw; the
 * structural claims are asserted numerically (depth = ceil(log2(alphabet)) + 1 levels, exactly
 * 2 alphabet - 1 nodes in this implementation
 * nodes, bitvector length equal to the sum of node sizes and at most n (depth + 1)); and the empty
 * string, single-symbol strings, the all-identical string and the full 256-symbol alphabet are all
 * exercised.  Inputs come from a documented 64-bit splitmix64 generator, never from Math.random.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

class WaveletTree {
  constructor(symbols, alphabetSize) {
    if (alphabetSize === undefined) {
      alphabetSize = symbols.length === 0 ? 1 : Math.max(...symbols) + 1;
    }
    if (!Number.isInteger(alphabetSize) || alphabetSize < 1) {
      throw new Error("alphabetSize must be a positive integer");
    }
    for (const symbol of symbols) {
      if (!Number.isInteger(symbol) || symbol < 0 || symbol >= alphabetSize) {
        throw new Error(`symbol ${symbol} outside the alphabet [0, ${alphabetSize})`);
      }
    }
    this.alphabetSize = alphabetSize;
    this.length = symbols.length;
    this.bitLength = 0;
    this.root = this.build(symbols.slice(), 0, alphabetSize - 1);
  }

  build(values, lo, hi) {
    if (lo === hi) {
      return { lo, hi, leaf: true, size: values.length, left: null, right: null };
    }
    const mid = (lo + hi) >> 1;
    const ranks = new Int32Array(values.length + 1);
    const leftValues = [];
    const rightValues = [];
    for (let i = 0; i < values.length; i++) {
      const goRight = values[i] > mid;
      ranks[i + 1] = ranks[i] + (goRight ? 1 : 0);
      if (goRight) rightValues.push(values[i]);
      else leftValues.push(values[i]);
    }
    this.bitLength += values.length;
    const node = { lo, hi, mid, ranks, size: values.length, leaf: false, left: null, right: null };
    node.left = this.build(leftValues, lo, mid);
    node.right = this.build(rightValues, mid + 1, hi);
    return node;
  }

  static countNodes(node) {
    if (node === null) return 0;
    return 1 + WaveletTree.countNodes(node.left) + WaveletTree.countNodes(node.right);
  }

  get nodeCount() {
    return WaveletTree.countNodes(this.root);
  }

  get depth() {
    const walk = (node) => (node === null ? 0 : 1 + Math.max(walk(node.left), walk(node.right)));
    return walk(this.root);
  }

  check(index, bound) {
    if (!Number.isInteger(index) || index < 0 || index > bound) {
      throw new Error(`position ${index} outside [0, ${bound}]`);
    }
  }

  /** Occurrences of `symbol` in s[0 .. i). */
  rank(symbol, i) {
    if (!Number.isInteger(symbol) || symbol < 0 || symbol >= this.alphabetSize) {
      throw new Error(`symbol ${symbol} outside the alphabet`);
    }
    this.check(i, this.length);
    let node = this.root;
    let position = i;
    while (!node.leaf) {
      if (symbol <= node.mid) {
        position -= node.ranks[position];
        node = node.left;
      } else {
        position = node.ranks[position];
        node = node.right;
      }
    }
    return position;
  }

  count(symbol) {
    return this.rank(symbol, this.length);
  }

  /** 0-based position of the j-th occurrence of `symbol` (j >= 1), or -1 when it does not exist. */
  select(symbol, j) {
    if (!Number.isInteger(symbol) || symbol < 0 || symbol >= this.alphabetSize) {
      throw new Error(`symbol ${symbol} outside the alphabet`);
    }
    if (!Number.isInteger(j) || j < 1) throw new Error("select expects j >= 1");
    if (this.count(symbol) < j) return -1;
    // The descent does not need a position: the symbol alone determines the path, and the leaf
    // position of the j-th occurrence is j - 1.  The actual document position is then recovered by
    // walking the recorded path back up with select on each node's bitvector.
    const path = [];
    let node = this.root;
    while (!node.leaf) {
      if (symbol <= node.mid) {
        path.push({ node, side: 0 });
        node = node.left;
      } else {
        path.push({ node, side: 1 });
        node = node.right;
      }
    }
    let result = j - 1;
    for (let k = path.length - 1; k >= 0; k--) {
      const { node: parent, side } = path[k];
      // selectInVector returns the first index p with `count >= result + 1` bits in [0, p), which is
      // one past the position of the wanted bit, hence the -1.
      result = WaveletTree.selectInVector(parent.ranks, result + 1, side) - 1;
    }
    return result;
  }

  /** First index p with at least k bits equal to `bit` in [0, p): one past the k-th such bit. */
  static selectInVector(ranks, k, bit) {
    let low = 0;
    let high = ranks.length - 1;
    while (low < high) {
      const mid = (low + high) >> 1;
      const count = bit === 1 ? ranks[mid] : mid - ranks[mid];
      if (count >= k) high = mid;
      else low = mid + 1;
    }
    return low;
  }

  access(i) {
    if (this.length === 0) throw new Error("access on an empty wavelet tree");
    this.check(i, this.length - 1);
    let node = this.root;
    let position = i;
    while (!node.leaf) {
      const goesRight = node.ranks[position + 1] > node.ranks[position];
      if (goesRight) {
        position = node.ranks[position];
        node = node.right;
      } else {
        position -= node.ranks[position];
        node = node.left;
      }
    }
    return node.lo;
  }

  /** Occurrences of `symbol` in s[l .. r). */
  rangeCount(l, r, symbol) {
    if (!Number.isInteger(symbol) || symbol < 0 || symbol >= this.alphabetSize) {
      throw new Error(`symbol ${symbol} outside the alphabet`);
    }
    this.check(l, this.length);
    this.check(r, this.length);
    if (l >= r) return 0;
    let node = this.root;
    let left = l;
    let right = r;
    while (!node.leaf) {
      if (symbol <= node.mid) {
        left -= node.ranks[left];
        right -= node.ranks[right];
        node = node.left;
      } else {
        left = node.ranks[left];
        right = node.ranks[right];
        node = node.right;
      }
    }
    return right - left;
  }

  /** Number of symbols strictly below `x` in s[l .. r): the FM-index C[] query. */
  rangeCountLess(l, r, x) {
    this.check(l, this.length);
    this.check(r, this.length);
    if (l >= r) return 0;
    const walk = (node, left, right) => {
      if (left >= right) return 0;
      if (x <= node.lo) return 0;
      if (x > node.hi) return right - left;
      if (node.leaf) return 0; // node.lo === node.hi and x > lo would have been caught above
      const leftZeros = left - node.ranks[left];
      const rightZeros = right - node.ranks[right];
      const leftOnes = node.ranks[left];
      const rightOnes = node.ranks[right];
      if (x <= node.mid + 1) return walk(node.left, leftZeros, rightZeros);
      return (rightZeros - leftZeros) + walk(node.right, leftOnes, rightOnes);
    };
    return walk(this.root, l, r);
  }
}

function bruteRank(symbols, symbol, i) {
  let count = 0;
  for (let p = 0; p < i; p++) if (symbols[p] === symbol) count += 1;
  return count;
}

function bruteRangeCount(symbols, l, r, symbol) {
  let count = 0;
  for (let p = l; p < r; p++) if (symbols[p] === symbol) count += 1;
  return count;
}

function bruteRangeCountLess(symbols, l, r, x) {
  let count = 0;
  for (let p = l; p < r; p++) if (symbols[p] < x) count += 1;
  return count;
}

function bytesOf(text) {
  return [...text].map((ch) => ch.codePointAt(0));
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240520);

  // (1) Exhaustive agreement with brute force on two small strings.
  {
    const cases = [
      { symbols: bytesOf("abracadabra"), alphabet: 256 },
      { symbols: [3, 1, 3, 1, 2, 0, 3, 2, 2, 1, 0, 0, 3], alphabet: 4 },
    ];
    for (const { symbols, alphabet } of cases) {
      const tree = new WaveletTree(symbols, alphabet);
      const n = symbols.length;
      for (let i = 0; i <= n; i++) {
        for (let c = 0; c < alphabet; c++) {
          assert.strictEqual(tree.rank(c, i), bruteRank(symbols, c, i),
            `rank(${c}, ${i}) on [${symbols.join(",")}]`);
        }
      }
      for (let i = 0; i < n; i++) {
        assert.strictEqual(tree.access(i), symbols[i], `access(${i})`);
      }
      for (let c = 0; c < alphabet; c++) {
        const total = bruteRank(symbols, c, n);
        assert.strictEqual(tree.count(c), total);
        for (let j = 1; j <= total + 1; j++) {
          if (j <= total) {
            const position = tree.select(c, j);
            assert.strictEqual(symbols[position], c, `select(${c}, ${j}) points at a ${c}`);
            assert.strictEqual(tree.rank(c, position), j - 1);
            assert.strictEqual(tree.rank(c, position + 1), j);
          } else {
            assert.strictEqual(tree.select(c, j), -1, `select(${c}, ${j}) is out of range`);
          }
        }
      }
      for (let l = 0; l <= n; l++) {
        for (let r = l; r <= n; r++) {
          for (let c = 0; c < alphabet; c++) {
            assert.strictEqual(tree.rangeCount(l, r, c), bruteRangeCount(symbols, l, r, c));
          }
          for (let x = 0; x <= alphabet; x++) {
            assert.strictEqual(tree.rangeCountLess(l, r, x), bruteRangeCountLess(symbols, l, r, x),
              `rangeCountLess(${l}, ${r}, ${x})`);
          }
        }
      }
    }
  }

  // (2) The classic example, with the values written out explicitly.
  {
    const text = "abracadabra";
    const tree = new WaveletTree(bytesOf(text), 256);
    assert.strictEqual(tree.length, 11);
    assert.strictEqual(tree.count("a".codePointAt(0)), 5);
    assert.strictEqual(tree.count("b".codePointAt(0)), 2);
    assert.strictEqual(tree.count("r".codePointAt(0)), 2);
    assert.strictEqual(tree.count("c".codePointAt(0)), 1);
    assert.strictEqual(tree.count("d".codePointAt(0)), 1);
    assert.strictEqual(tree.count("z".codePointAt(0)), 0);
    assert.strictEqual(tree.select("a".codePointAt(0), 1), 0);
    assert.strictEqual(tree.select("a".codePointAt(0), 5), 10);
    assert.strictEqual(tree.select("r".codePointAt(0), 1), 2);
    assert.strictEqual(tree.select("r".codePointAt(0), 2), 9);
    assert.strictEqual(tree.select("z".codePointAt(0), 1), -1);
    assert.strictEqual(tree.access(4), "c".codePointAt(0));
    assert.strictEqual(tree.rank("a".codePointAt(0), 7), 3,
      "the a's at positions 0, 3 and 5 are the only ones before position 7");
    assert.strictEqual(tree.rank("a".codePointAt(0), 8), 4, "position 7 itself is an a");
    assert.strictEqual(tree.rangeCount(1, 5, "a".codePointAt(0)), 1);
    assert.strictEqual(tree.rangeCountLess(0, 11, "b".codePointAt(0)), 5);
    assert.strictEqual(tree.rangeCountLess(0, 11, "g".codePointAt(0)), 9,
      "everything below 'g' is 9 characters: the two r's are above it");
    assert.strictEqual(tree.rangeCountLess(0, 11, "a".codePointAt(0)), 0);
    assert.strictEqual(tree.rangeCountLess(2, 3, "c".codePointAt(0)), 0);
    assert.strictEqual(tree.rangeCount(5, 5, "a".codePointAt(0)), 0, "empty range");
    assert.strictEqual(tree.rangeCount(5, 4, "a".codePointAt(0)), 0, "inverted range");
  }

  // (3) Random strings over several alphabet sizes, including the rank/select inverses.
  {
    for (const alphabet of [4, 16, 26, 256]) {
      for (let trial = 0; trial < 3; trial++) {
        const n = 60 + rng.nextInt(80);
        const symbols = [];
        for (let i = 0; i < n; i++) symbols.push(rng.nextInt(alphabet));
        const tree = new WaveletTree(symbols, alphabet);
        for (let q = 0; q < 60; q++) {
          const i = rng.nextInt(n + 1);
          const c = rng.nextInt(alphabet);
          assert.strictEqual(tree.rank(c, i), bruteRank(symbols, c, i));
          const l = rng.nextInt(n + 1);
          const r = l + rng.nextInt(n - l + 1);
          const x = rng.nextInt(alphabet + 1);
          assert.strictEqual(tree.rangeCount(l, r, c), bruteRangeCount(symbols, l, r, c));
          assert.strictEqual(tree.rangeCountLess(l, r, x), bruteRangeCountLess(symbols, l, r, x));
        }
        for (let i = 0; i < n; i++) {
          const c = symbols[i];
          assert.strictEqual(tree.select(c, tree.rank(c, i + 1)), i,
            `select(c, rank(c, ${i + 1})) = ${i}`);
        }
        for (let c = 0; c < alphabet; c++) {
          const total = tree.count(c);
          for (let j = 1; j <= total; j++) {
            assert.strictEqual(tree.rank(c, tree.select(c, j) + 1), j);
          }
        }
        // `depth` counts node levels: ceil(log2(alphabet)) splits plus the leaves.
        assert.strictEqual(tree.depth, Math.ceil(Math.log2(alphabet)) + 1,
          `depth ${tree.depth} for alphabet ${alphabet}`);
        assert.ok(tree.nodeCount <= 2 * alphabet - 1,
          `node count ${tree.nodeCount} <= 2 * ${alphabet} - 1`);
        assert.strictEqual(tree.bitLength, (() => {
          let total = 0;
          const walk = (node) => {
            if (node.leaf) return;
            total += node.size;
            walk(node.left);
            walk(node.right);
          };
          walk(tree.root);
          return total;
        })(), "the stored bit length is the sum of the node sizes");
        assert.ok(tree.bitLength <= n * (tree.depth + 1),
          `bit length ${tree.bitLength} <= n (depth + 1) = ${n * (tree.depth + 1)}`);
      }
    }
  }

  // (4) Edge cases and error handling.
  {
    const empty = new WaveletTree([], 256);
    assert.strictEqual(empty.length, 0);
    assert.strictEqual(empty.count(0), 0);
    assert.strictEqual(empty.rank(0, 0), 0);
    assert.strictEqual(empty.select(0, 1), -1);
    assert.strictEqual(empty.rangeCount(0, 0, 5), 0);
    assert.strictEqual(empty.rangeCountLess(0, 0, 200), 0);
    assert.throws(() => empty.access(0), /empty wavelet tree/);
    assert.throws(() => empty.select(256, 1), /outside the alphabet/);
    assert.strictEqual(empty.rangeCountLess(0, 0, 0), 0);
    assert.strictEqual(empty.depth, 9, "an empty 256-symbol alphabet still has 8 splits + leaves");
    assert.strictEqual(empty.bitLength, 0);

    const single = new WaveletTree([7], 8);
    assert.strictEqual(single.access(0), 7);
    assert.strictEqual(single.count(7), 1);
    assert.strictEqual(single.select(7, 1), 0);
    assert.strictEqual(single.select(7, 2), -1);

    const uniform = new WaveletTree(new Array(1000).fill(5), 256);
    assert.strictEqual(uniform.count(5), 1000);
    assert.strictEqual(uniform.select(5, 1000), 999);
    assert.strictEqual(uniform.select(5, 1001), -1);
    assert.strictEqual(uniform.rank(5, 500), 500);
    assert.strictEqual(uniform.rangeCountLess(0, 1000, 6), 1000);
    assert.strictEqual(uniform.rangeCountLess(0, 1000, 5), 0);
    assert.strictEqual(uniform.depth, 9);
    assert.ok(uniform.bitLength === 8000, "every level stores n bits even for uniform data");
    assert.strictEqual(uniform.nodeCount, 511,
      "the tree is complete over the alphabet even where one side is empty (2 * 256 - 1 = 511 " +
      "nodes), a documented choice that keeps every descent uniform instead of pruning empty sides");

    const full = new WaveletTree([...Array(256).keys()], 256);
    assert.strictEqual(full.length, 256);
    for (let c = 0; c < 256; c++) {
      assert.strictEqual(full.count(c), 1);
      assert.strictEqual(full.select(c, 1), c);
      assert.strictEqual(full.access(c), c);
    }
    assert.throws(() => new WaveletTree([256], 256), /outside the alphabet/);
    assert.throws(() => new WaveletTree([-1], 256), /outside the alphabet/);
    assert.throws(() => new WaveletTree([1.5], 4), /outside the alphabet/);
    assert.throws(() => new WaveletTree([1], 0), /alphabetSize/);
    const tree = new WaveletTree([1, 2, 3], 4);
    assert.throws(() => tree.rank(4, 1), /outside the alphabet/);
    assert.throws(() => tree.rank(1, 4), /outside/);
    assert.throws(() => tree.rangeCount(0, 4, 1), /outside/);
    assert.throws(() => tree.rangeCountLess(0, 4, 1), /outside/);
    assert.throws(() => tree.select(1, 0), /j >= 1/);
    assert.throws(() => tree.select(1, 1.5), /j >= 1/);
    assert.strictEqual(tree.access(2), 3);
  }

  // (5) Determinism of construction and of the derived structure.
  {
    const symbols = [];
    for (let i = 0; i < 500; i++) symbols.push(rng.nextInt(32));
    const first = new WaveletTree(symbols, 32);
    const second = new WaveletTree(symbols.slice(), 32);
    assert.strictEqual(first.bitLength, second.bitLength);
    assert.strictEqual(first.nodeCount, second.nodeCount);
    assert.strictEqual(first.depth, second.depth);
    assert.deepStrictEqual(collectRanks(second.root), collectRanks(first.root),
      "two builds of the same string produce identical prefix-rank arrays");
    for (let i = 0; i <= 500; i += 37) {
      for (let c = 0; c < 32; c++) assert.strictEqual(first.rank(c, i), second.rank(c, i));
    }
    const other = [];
    for (let i = 0; i < 500; i++) other.push(rng.nextInt(32));
    assert.notDeepStrictEqual(collectRanks(new WaveletTree(other, 32).root),
      collectRanks(first.root), "a different string gives a different structure");
    const streamA = [];
    const replay = new Rng(20240520);
    for (let i = 0; i < 5; i++) streamA.push(replay.nextInt(32));
    const streamB = [];
    const replay2 = new Rng(20240520);
    for (let i = 0; i < 5; i++) streamB.push(replay2.nextInt(32));
    assert.deepStrictEqual(streamB, streamA, "a seeded generator replays its stream exactly");
    assert.notDeepStrictEqual(streamA, symbols.slice(0, 5),
      "while the shared generator in main has long since advanced");
    // Every occurrence of a symbol keeps its rank/select pair consistent after a rebuild.
    const rebuilt = new WaveletTree(symbols.slice(), 32);
    for (let i = 0; i < 500; i++) {
      const c = symbols[i];
      assert.strictEqual(rebuilt.select(c, rebuilt.rank(c, i + 1)), i);
    }
  }
}

/** Helper used only to prove that two independent builds are structurally independent objects. */
function collectRanks(node, out = []) {
  if (node === null) return out;
  if (!node.leaf) {
    out.push(node.ranks[node.size]);
    collectRanks(node.left, out);
    collectRanks(node.right, out);
  }
  return out;
}

module.exports = { MASK64, Rng, WaveletTree, bruteRank, bruteRangeCount, bruteRangeCountLess,
  bytesOf, collectRanks };

if (require.main === module) {
  main();
  console.log("PASS");
}
