/** Task 102 - Flood fill algorithm (DFS/BFS). (Tier: Beginner, CTCI Ch 4) */
"use strict";

function floodFill(image, sr, sc, newColor) {
  if (!image || image.length === 0 || image[0].length === 0) return image;
  const rows = image.length;
  const cols = image[0].length;
  const startColor = image[sr][sc];
  if (startColor === newColor) return image;
  image[sr][sc] = newColor;
  const queue = [[sr, sc]];
  for (let head = 0; head < queue.length; head++) {
    const [r, c] = queue[head];
    for (const [nr, nc] of [[r + 1, c], [r - 1, c], [r, c + 1], [r, c - 1]]) {
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && image[nr][nc] === startColor) {
        image[nr][nc] = newColor;
        queue.push([nr, nc]);
      }
    }
  }
  return image;
}

module.exports = { floodFill };

if (require.main === module) {
  const assert = require("node:assert");
  const classic = [[1, 1, 1], [1, 1, 0], [1, 0, 1]];
  assert.deepStrictEqual(floodFill(classic, 1, 1, 2), [[2, 2, 2], [2, 2, 0], [2, 0, 1]]);
  const noop = [[0, 0], [0, 0]];
  assert.deepStrictEqual(floodFill(noop, 0, 0, 0), [[0, 0], [0, 0]]);
  const patch = [[1, 1, 0], [0, 1, 0]];
  assert.deepStrictEqual(floodFill(patch, 0, 0, 5), [[5, 5, 0], [0, 5, 0]]);
  const single = [[7]];
  assert.deepStrictEqual(floodFill(single, 0, 0, 9), [[9]]);
  assert.deepStrictEqual(floodFill([], 0, 0, 1), []);
  const mix = [[1, 1], [1, 0]];
  assert.deepStrictEqual(floodFill(mix, 0, 0, 0), [[0, 0], [0, 0]]);
  console.log("PASS");
}
