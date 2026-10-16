/** Task 608 - Implement BWT-based full-text index with fast locate. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The index is the run-length encoded BWT of text$ (the sentinel makes the suffix array equal to the
 * sorted rotations, so the position walk "+1 mod N" is exact), the cumulative table C, and sampled
 * suffix values. Two mappings drive everything:
 *   LF(row)  = C[BWT[row]] + rank(BWT[row], row)   - the row whose suffix starts one position earlier
 *   psi      = LF^-1                               - the row whose suffix starts one position later
 *
 * rank is answered run-wise: inside a run every row has the same character, so
 * rank(c, i) = (occurrences of c in all runs before the run holding i) + (i - runStart) when the run
 * character is c; the run prefix counts are checkpointed every 32 runs.
 *
 * Fast locate works in both directions with the same sampling (rows whose suffix position is a
 * multiple of `rate`): row -> position walks LF (decrementing the position by one) and position -> row
 * steps back to the previous sampled position and applies psi there. Both stay within rate - 1 steps,
 * and the bound is measured, not assumed. `extract` reads any substring from the BWT by walking LF
 * from the row of the suffix that starts after it.
 *
 * Checks: the run encoding reproduces the BWT exactly (adjacent runs differ, lengths sum to N),
 * run-wise rank against brute force, LF/psi inverse, count/locate against brute force,
 * rowToPosition/positionToRow against the suffix array on every row, extract against the text on many
 * ranges, step bounds, run compression on periodic text, and LCG plus 4000-character texts. */"use strict";

const SENTINEL = "\u0000";
const RUN_CHECKPOINT = 32;

class BWTIndex {
  constructor(text, sampleRate = 32) {
    if (text.includes(SENTINEL)) throw new Error("text must not contain the sentinel character");
    this.body = text;
    this.sampleRate = Math.max(1, sampleRate);
    const terminated = text + SENTINEL;
    this.N = terminated.length;
    const sa = suffixArrayDoubling(terminated);
    this.bwt = new Array(this.N);
    for (let row = 0; row < this.N; row++) this.bwt[row] = terminated[(sa[row] - 1 + this.N) % this.N];

    this.chars = Array.from(new Set([...terminated])).sort();
    this.charIndex = new Map(this.chars.map((ch, i) => [ch, i]));

    // Run-length encoding of the BWT, a row -> run map and checkpointed counts over whole runs.
    this.runChar = [];
    this.runStart = [];
    this.runLength = [];
    this.runOfRow = new Int32Array(this.N);
    for (let row = 0; row < this.N; row++) {
      if (row === 0 || this.bwt[row] !== this.bwt[row - 1]) {
        this.runChar.push(this.bwt[row]);
        this.runStart.push(row);
        this.runLength.push(1);
      } else {
        this.runLength[this.runLength.length - 1]++;
      }
      this.runOfRow[row] = this.runChar.length - 1;
    }
    this.runCount = this.runChar.length;
    this.runCheckpoints = this.chars.map(() => new Int32Array(Math.ceil(this.runCount / RUN_CHECKPOINT) + 1));
    this.runCharacterCounts = new Array(this.chars.length).fill(0);
    for (let run = 0; run < this.runCount; run++) {
      if (run % RUN_CHECKPOINT === 0) {
        for (let c = 0; c < this.chars.length; c++) {
          this.runCheckpoints[c][run / RUN_CHECKPOINT] = this.runCharacterCounts[c];
        }
      }
      this.runCharacterCounts[this.charIndex.get(this.runChar[run])] += this.runLength[run];
    }
    for (let c = 0; c < this.chars.length; c++) {
      this.runCheckpoints[c][Math.ceil(this.runCount / RUN_CHECKPOINT)] = this.runCharacterCounts[c];
    }
    // C[c] is the number of characters smaller than c, i.e. the totals of the preceding characters.
    this.C = this.chars.map((_, i) => this.runCharacterCounts.slice(0, i).reduce((a, b) => a + b, 0));

    this.lfRow = new Int32Array(this.N);
    this.psiRow = new Int32Array(this.N);
    for (let row = 0; row < this.N; row++) {
      const c = this.charIndex.get(this.bwt[row]);
      const value = this.C[c] + this.rankIndex(c, row);
      this.lfRow[row] = value;
      this.psiRow[value] = row;
    }

    // Suffix sampling: a sampled row stores the suffix position it stands for, -1 when unsampled.
    this.marked = new Int32Array(this.N).fill(-1);
    this.rowOfValue = new Int32Array(this.N).fill(-1);
    this.sampleCount = 0;
    for (let row = 0; row < this.N; row++) {
      if (sa[row] % this.sampleRate !== 0) continue;
      this.marked[row] = sa[row];
      this.rowOfValue[sa[row]] = row;
      this.sampleCount++;
    }
  }

  /** Occurrences of character `c` in bwt[0..i) using the run-length encoding. */
  rankIndex(c, i) {
    if (i <= 0) return 0;
    if (i >= this.N) return this.runCharacterCounts[c]; // whole-BWT count, valid for i == N too
    const run = this.runOfRow[i];
    const block = (run / RUN_CHECKPOINT) | 0;
    let total = this.runCheckpoints[c][block];
    for (let r = block * RUN_CHECKPOINT; r < run; r++) {
      if (this.charIndex.get(this.runChar[r]) === c) total += this.runLength[r];
    }
    if (this.charIndex.get(this.runChar[run]) === c) total += i - this.runStart[run];
    return total;
  }

  /** Backward search: SA interval [l, r) of `pattern`, or null. */
  search(pattern) {
    let l = 0;
    let r = this.N;
    for (let i = pattern.length - 1; i >= 0; i--) {
      const c = this.charIndex.get(pattern[i]);
      if (c === undefined) return null;
      l = this.C[c] + this.rankIndex(c, l);
      r = this.C[c] + this.rankIndex(c, r);
      if (l >= r) return null;
    }
    return { l, r };
  }

  count(pattern) {
    const interval = this.search(pattern);
    return interval === null ? 0 : interval.r - interval.l;
  }

  /** Walks LF down to the next sampled row; returns that row and the number of steps taken. */
  walkToSample(row) {
    let steps = 0;
    let current = row;
    while (this.marked[current] < 0) {
      current = this.lfRow[current];
      steps++;
    }
    return { row: current, steps };
  }

  /** Suffix position of `row`: walk LF until a sampled row (at most sampleRate - 1 steps). */
  rowToPosition(row) {
    const hit = this.walkToSample(row);
    return (this.marked[hit.row] + hit.steps) % this.N;
  }

  /** Row of the suffix starting at `position`: step back to the sampled position, then apply psi. */
  positionToRow(position) {
    const steps = position % this.sampleRate;
    let row = this.rowOfValue[position - steps];
    for (let step = 0; step < steps; step++) row = this.psiRow[row];
    return row;
  }

  locate(pattern, limit = Infinity) {
    const interval = this.search(pattern);
    if (interval === null) return [];
    const positions = [];
    for (let row = interval.l; row < interval.r; row++) {
      const start = this.rowToPosition(row);
      if (start < this.body.length) positions.push(start);
    }
    positions.sort((a, b) => a - b);
    return positions.slice(0, limit);
  }
  /** Worst-case LF steps used to locate the interval of `pattern`; must stay below the rate. */
  locateStepBound(pattern) {
    const interval = this.search(pattern);
    if (interval === null) return 0;
    let worst = 0;
    for (let row = interval.l; row < interval.r; row++) worst = Math.max(worst, this.walkToSample(row).steps);
    return worst;
  }

  /** Recovers text[start .. start + length) from the BWT, one LF step per character. */
  extract(start, length) {
    if (length <= 0) return "";
    if (start < 0 || start + length > this.body.length) throw new Error("range outside the text");
    let row = this.positionToRow(start + length); // the suffix starting right after the range
    const out = new Array(length);
    for (let k = length - 1; k >= 0; k--) {
      out[k] = this.bwt[row];
      row = this.lfRow[row];
    }
    return out.join("");
  }

  /** The run-length encoding expanded back into the BWT; used to verify the encoding. */
  expandRuns() {
    const out = [];
    for (let run = 0; run < this.runCount; run++) out.push(this.runChar[run].repeat(this.runLength[run]));
    return out.join("").split("");
  }

  indexSize() {
    return this.runCount * 12 + this.chars.length * this.runCheckpoints[0].length * 4 +
      4 * this.N + this.N + 4 * this.sampleCount; // runs, checkpoints, psi, marked values, samples
  }
}

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

function bruteOccurrences(text, pattern) {
  const positions = [];
  for (let i = 0; i + pattern.length <= text.length; i++) if (text.startsWith(pattern, i)) positions.push(i);
  return positions;
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
/** Brute-force rank of a character in a packed array prefix, mirroring rankIndex. */
function bruteRank(sequence, ch, i) {
  let count = 0;
  for (let k = 0; k < i; k++) if (sequence[k] === ch) count++;
  return count;
}

module.exports = { BWTIndex, suffixArrayDoubling, bruteOccurrences, bruteRank, lcgString, SENTINEL };

if (require.main === module) {
  const assert = require("node:assert");

  const empty = new BWTIndex("");
  const single = new BWTIndex("a");
  assert.deepStrictEqual([empty.N, empty.runCount, empty.count("a"), empty.locate("a").length], [1, 1, 0, 0]);
  assert.deepStrictEqual([empty.rowToPosition(0), empty.positionToRow(0), empty.extract(0, 0)], [0, 0, ""]);
  assert.deepStrictEqual([single.count("a"), single.extract(0, 1), single.positionToRow(0), single.locate("a")[0]],
    [1, "a", 1, 0]); // the suffix array of "a\0" is [1, 0]
  assert.deepStrictEqual(single.expandRuns(), [...single.bwt]);

  const unary = new BWTIndex("a".repeat(200), 8);
  assert.ok(unary.count("a") === 200 && unary.runCount === 2 && unary.extract(100, 50) === "a".repeat(50));
  assert.deepStrictEqual(unary.locate("a", 2), [0, 1]);
  assert.deepStrictEqual(unary.expandRuns(), [...unary.bwt]);

  for (const text of ["banana", "mississippi", "abracadabra", "abab", "abcabc", "aabaaabb"]) {
    const rate = 4;
    const index = new BWTIndex(text, rate);
    const sa = suffixArrayDoubling(text + SENTINEL);
    assert.deepStrictEqual(index.expandRuns(), [...index.bwt], `${text} runs`);
    assert.strictEqual(index.runLength.reduce((a, b) => a + b, 0), index.N);
    for (let run = 1; run < index.runCount; run++) {
      assert.notStrictEqual(index.runChar[run], index.runChar[run - 1], `${text} adjacent runs`);
    }
    for (let c = 0; c < index.chars.length; c++) {
      const ch = index.chars[c];
      for (let i = 0; i <= index.N; i++) {
        assert.strictEqual(index.rankIndex(c, i), bruteRank(index.bwt, ch, i), `${text} rank ${ch}`);
      }
    }
    for (let row = 0; row < index.N; row++) {
      assert.ok(index.psiRow[index.lfRow[row]] === row, `${text} lf/psi ${row}`);
      assert.strictEqual(index.rowToPosition(row), sa[row], `${text} rowToPosition ${row}`);
    }
    for (let position = 0; position <= text.length; position++) {
      assert.strictEqual(index.positionToRow(position), sa.indexOf(position), `${text} positionToRow`);
    }
    for (let start = 0; start <= text.length; start++) {
      for (const length of [0, Math.min(1, text.length - start), Math.min(3, text.length - start), text.length - start]) {
        assert.strictEqual(index.extract(start, length), text.slice(start, start + length), `${text} extract`);
      }
    }
    for (let start = 0; start < text.length; start++) {
      for (let end = start + 1; end <= text.length; end++) {
        const pattern = text.slice(start, end);
        assert.deepStrictEqual(index.locate(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
        assert.strictEqual(index.count(pattern), bruteOccurrences(text, pattern).length, `${text} count`);
        assert.ok(index.locateStepBound(pattern) < rate, `${text} ${pattern} steps`);
      }
    }
  }

  const banana = new BWTIndex("banana", 2);
  assert.deepStrictEqual([banana.locate("ana"), banana.locate("a")], [[1, 3], [1, 3, 5]]);
  assert.ok(banana.count("bananas") === 0 && banana.count("") === banana.N && banana.extract(2, 4) === "nana");
  assert.throws(() => banana.extract(4, 3), /range/);
  assert.throws(() => new BWTIndex("bad\u0000text"), /sentinel/);

  let seed = 608;
  for (const alphabet of ["ab", "abc", "abcdefghij"]) {
    for (let trial = 0; trial < 10; trial++) {
      let text;
      [text, seed] = lcgString(50, alphabet, seed);
      const rate = 1 + (trial % 7);
      const index = new BWTIndex(text, rate);
      const sa = suffixArrayDoubling(text + SENTINEL);
      assert.deepStrictEqual(index.expandRuns(), [...index.bwt], text);
      for (let row = 0; row < index.N; row += 3) {
        assert.ok(index.rowToPosition(row) === sa[row] && index.psiRow[index.lfRow[row]] === row, text);
      }
      for (let position = 0; position <= text.length; position += 2) {
        assert.strictEqual(index.positionToRow(position), sa.indexOf(position), `${text} pos ${position}`);
      }
      for (let start = 0; start < text.length; start += 5) {
        for (let end = start + 1; end <= Math.min(start + 9, text.length); end++) {
          const pattern = text.slice(start, end);
          assert.deepStrictEqual(index.locate(pattern), bruteOccurrences(text, pattern), `${text} ${pattern}`);
          assert.ok(index.locateStepBound(pattern) < rate, `${text} ${pattern} steps`);
        }
      }
      assert.strictEqual(index.extract(0, text.length), text);
      assert.ok([`z${text}`, text + "z", alphabet.repeat(text.length)].every((p) => index.count(p) === 0), text);
    }
  }

  const repetitive = new BWTIndex("ab".repeat(1000), 64);
  assert.ok(repetitive.count("ab") === 1000 && repetitive.runCount < repetitive.N / 10);
  assert.strictEqual(repetitive.extract(1000, 500), "ab".repeat(250)); // few runs on periodic text

  let bigText;
  [bigText, seed] = lcgString(4000, "ab", seed);
  const big = new BWTIndex(bigText, 64);
  assert.ok(big.N === 4001 && big.sampleCount === Math.ceil(4001 / 64));
  for (let probe = 0; probe < 60; probe++) {
    const start = (probe * 67) % (bigText.length - 30);
    const pattern = bigText.slice(start, start + 7 + (probe % 9));
    assert.deepStrictEqual(big.locate(pattern), bruteOccurrences(bigText, pattern), pattern);
    assert.strictEqual(big.count(pattern), bruteOccurrences(bigText, pattern).length, pattern);
  }
  assert.ok(big.extract(0, 4000) === bigText && big.extract(3990, 10) === bigText.slice(3990));
  assert.deepStrictEqual(big.expandRuns(), [...big.bwt]);
  assert.ok(big.locateStepBound("a") < 64);

  console.log(`bwt index: N=${big.N}, ${big.runCount} runs (vs ${big.N} rows), ` +
    `${big.sampleCount} sampled suffixes, ${big.indexSize()} bytes`);
  console.log("PASS");
}
