/** Task 128 - Implement basic memoization wrapper. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function memoize(fn) {
  const cache = new Map();
  const wrapper = (...args) => {
    const key = JSON.stringify(args);
    if (!cache.has(key)) cache.set(key, fn(...args));
    return cache.get(key);
  };
  wrapper.cache = cache;
  return wrapper;
}

module.exports = { memoize };

if (require.main === module) {
  const assert = require("node:assert");
  let calls = 0;

  const fib = memoize((n) => {
    calls++;
    return n <= 1 ? n : fib(n - 1) + fib(n - 2);
  });

  assert.strictEqual(fib(30), 832040);
  assert.strictEqual(calls, 31);
  assert.strictEqual(fib.cache.get(JSON.stringify([10])), 55);

  const gridPaths = memoize((rows, cols) => {
    if (rows < 0 || cols < 0) return 0;
    if (rows === 0 || cols === 0) return 1;
    return gridPaths(rows - 1, cols) + gridPaths(rows, cols - 1);
  });
  assert.strictEqual(gridPaths(3, 3), 20);
  assert.strictEqual(gridPaths(3, 3), 20);
  console.log("PASS");
}
