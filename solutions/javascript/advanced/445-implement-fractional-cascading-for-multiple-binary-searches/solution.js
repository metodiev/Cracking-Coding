/** Task 445 - Implement fractional cascading for multiple binary searches. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function bisectLeft(values, x) {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

class FractionalCascade {
  constructor(lists) {
    const k = lists.length;
    this.k = k;
    this.augmented = new Array(k);
    this.native = new Array(k);
    this.augmented[k - 1] = lists[k - 1].slice();
    this.native[k - 1] = new Array(this.augmented[k - 1].length).fill(true);
    for (let i = k - 2; i >= 0; i--) {
      const copied = this.augmented[i + 1].filter((_, index) => index % 2 === 0);
      const merged = FractionalCascade.merge(lists[i], copied);
      this.augmented[i] = merged[0];
      this.native[i] = merged[1];
    }
    this.bridge = new Array(Math.max(0, k - 1));
    for (let i = 0; i < k - 1; i++) {
      const upper = this.augmented[i + 1];
      const bridge = new Array(this.augmented[i].length);
      let j = 0;
      for (let p = 0; p < this.augmented[i].length; p++) {
        while (j < upper.length && upper[j] < this.augmented[i][p]) j++;
        bridge[p] = j;
      }
      this.bridge[i] = bridge;
    }
    this.nextNative = new Array(k);
    for (let i = 0; i < k; i++) {
      const size = this.augmented[i].length;
      const nxt = new Array(size + 1).fill(size);
      for (let p = size - 1; p >= 0; p--) nxt[p] = this.native[i][p] ? p : nxt[p + 1];
      this.nextNative[i] = nxt;
    }
  }

  static merge(nativeList, copies) {
    const values = [];
    const flags = [];
    let p = 0;
    let q = 0;
    while (p < nativeList.length || q < copies.length) {
      if (q === copies.length || (p < nativeList.length && nativeList[p] <= copies[q])) {
        values.push(nativeList[p]);
        flags.push(true);
        p++;
      } else {
        values.push(copies[q]);
        flags.push(false);
        q++;
      }
    }
    return [values, flags];
  }

  successors(x) {
    const results = new Array(this.k).fill(null);
    let pos = bisectLeft(this.augmented[0], x);
    for (let i = 0; i < this.k; i++) {
      if (i > 0) {
        if (pos === this.augmented[i - 1].length) {
          const m = this.augmented[i].length;
          pos = m > 0 && this.augmented[i][m - 1] >= x ? m - 1 : m;
        } else {
          let q = this.bridge[i - 1][pos];
          if (q > 0 && this.augmented[i][q - 1] >= x) q--;
          pos = q;
        }
      }
      const nxt = this.nextNative[i][pos];
      results[i] = nxt < this.augmented[i].length ? this.augmented[i][nxt] : null;
    }
    return results;
  }
}

function lcgLists(count, size, seed, stepMax = 9) {
  const lists = [];
  for (let i = 0; i < count; i++) {
    const values = [];
    let current = -40;
    for (let j = 0; j < size; j++) {
      seed = (seed * 75 + 74) % 65537;
      current += 1 + (seed % stepMax);
      values.push(current);
    }
    lists.push(values);
  }
  return lists;
}

function independentSuccessors(lists, x) {
  return lists.map((values) => {
    const pos = bisectLeft(values, x);
    return pos < values.length ? values[pos] : null;
  });
}

module.exports = { FractionalCascade, lcgLists, independentSuccessors };

if (require.main === module) {
  const assert = require("node:assert");

  let cascade = new FractionalCascade([[1, 10], [5, 20], [3, 7, 30]]);
  assert.deepStrictEqual(cascade.augmented[0], [1, 3, 10, 20]);
  assert.deepStrictEqual(cascade.augmented[1], [3, 5, 20, 30]);
  assert.deepStrictEqual(cascade.augmented[2], [3, 7, 30]);
  assert.deepStrictEqual(cascade.successors(6), [10, 20, 7]);
  assert.deepStrictEqual(cascade.successors(3), [10, 5, 3]);
  assert.deepStrictEqual(cascade.successors(0), [1, 5, 3]);
  assert.deepStrictEqual(cascade.successors(10), [10, 20, 30]);
  assert.deepStrictEqual(cascade.successors(31), [null, null, null]);
  assert.deepStrictEqual(cascade.successors(20), [null, 20, 30]);

  const lists = lcgLists(5, 12, 445);
  cascade = new FractionalCascade(lists);
  for (let i = 0; i < 4; i++) {
    assert.strictEqual(cascade.augmented[i].length, lists[i].length + Math.ceil(cascade.augmented[i + 1].length / 2));
  }
  let seed = 555;
  for (let step = 0; step < 200; step++) {
    seed = (seed * 75 + 74) % 65537;
    const x = seed % 200 - 90;
    assert.deepStrictEqual(cascade.successors(x), independentSuccessors(lists, x), `x=${x}`);
  }

  for (const values of lcgLists(3, 5, 616)) {
    cascade = new FractionalCascade([values]);
    seed = 77;
    for (let step = 0; step < 40; step++) {
      seed = (seed * 75 + 74) % 65537;
      const x = seed % 120 - 60;
      assert.deepStrictEqual(cascade.successors(x), independentSuccessors([values], x), `single x=${x}`);
    }
  }
  console.log("PASS");
}
