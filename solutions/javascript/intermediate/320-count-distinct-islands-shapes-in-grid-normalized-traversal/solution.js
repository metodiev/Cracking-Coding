/** Task 320 - Count distinct island shapes in a grid (normalised traversal). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

const STEPS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/** Number of 4-connected islands of 1s (shapes merged or not). */
function countIslands(grid) {
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  const seen = Array.from({ length: rows }, () => new Array(cols).fill(false));
  let total = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === 1 && !seen[r][c]) {
        total++;
        const stack = [[r, c]];
        seen[r][c] = true;
        while (stack.length > 0) {
          const [cr, cc] = stack.pop();
          for (const [dr, dc] of STEPS) {
            const nr = cr + dr;
            const nc = cc + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === 1 && !seen[nr][nc]) {
              seen[nr][nc] = true;
              stack.push([nr, nc]);
            }
          }
        }
      }
    }
  }
  return total;
}

/**
 * One sorted array of normalised "dr,dc" cells per island (translation only; rotations and
 * reflections are different shapes).
 */
function collectIslandShapes(grid) {
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  const seen = Array.from({ length: rows }, () => new Array(cols).fill(false));
  const shapes = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === 1 && !seen[r][c]) {
        const stack = [[r, c]];
        seen[r][c] = true;
        const cells = [];
        while (stack.length > 0) {
          const [cr, cc] = stack.pop();
          cells.push([cr, cc]);
          for (const [dr, dc] of STEPS) {
            const nr = cr + dr;
            const nc = cc + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === 1 && !seen[nr][nc]) {
              seen[nr][nc] = true;
              stack.push([nr, nc]);
            }
          }
        }
        const minR = Math.min(...cells.map((cell) => cell[0]));
        const minC = Math.min(...cells.map((cell) => cell[1]));
        shapes.push(cells.map((cell) => `${cell[0] - minR},${cell[1] - minC}`).sort());
      }
    }
  }
  return shapes;
}

function countDistinctIslands(grid) {
  return new Set(collectIslandShapes(grid).map((shape) => shape.join(";"))).size;
}

/** Independent oracle: recursive flood fill, returning a set of canonical joined keys. */
function islandShapesSortedTuples(grid) {
  const rows = grid.length;
  const cols = rows > 0 ? grid[0].length : 0;
  const seen = Array.from({ length: rows }, () => new Array(cols).fill(false));
  const shapes = new Set();

  function flood(r, c, cells) {
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] !== 1 || seen[r][c]) return;
    seen[r][c] = true;
    cells.push([r, c]);
    flood(r + 1, c, cells);
    flood(r - 1, c, cells);
    flood(r, c + 1, cells);
    flood(r, c - 1, cells);
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === 1 && !seen[r][c]) {
        const cells = [];
        flood(r, c, cells);
        const minR = Math.min(...cells.map((cell) => cell[0]));
        const minC = Math.min(...cells.map((cell) => cell[1]));
        const keys = cells.map((cell) => `${cell[0] - minR},${cell[1] - minC}`).sort();
        shapes.add(keys.join(";"));
      }
    }
  }
  return shapes;
}

/** Canonical key set of the shape arrays, for a direct set comparison. */
function canonicalize(shapes) {
  return new Set(shapes.map((shape) => shape.join(";")));
}

function isConnectedShape(shape) {
  if (shape.length === 0) return false;
  const cellSet = new Set(shape);
  const stack = [shape[0]];
  const reached = new Set([shape[0]]);
  while (stack.length > 0) {
    const [r, c] = stack.pop().split(",").map(Number);
    for (const [dr, dc] of STEPS) {
      const next = `${r + dr},${c + dc}`;
      if (cellSet.has(next) && !reached.has(next)) {
        reached.add(next);
        stack.push(next);
      }
    }
  }
  return reached.size === cellSet.size;
}

function lcgGrids(rows, cols, count, seed = 3200) {
  const grids = [];
  for (let k = 0; k < count; k++) {
    const grid = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        seed = (seed * 75 + 74) % 65537;
        row.push(seed % 3 === 0 ? 1 : 0);
      }
      grid.push(row);
    }
    grids.push(grid);
  }
  return grids;
}

module.exports = { STEPS, countIslands, collectIslandShapes, countDistinctIslands, islandShapesSortedTuples, canonicalize, isConnectedShape, lcgGrids };

if (require.main === module) {
  const assert = require("node:assert");

  const classic = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
  ];
  assert.strictEqual(countDistinctIslands(classic), 3);
  assert.deepStrictEqual([...canonicalize(collectIslandShapes(classic))].sort(),
    ["0,0;0,1", "0,0;0,1;0,2;0,3;0,4", "0,0;0,1;1,0;1,1"]);

  // Rotations are NOT identified: a horizontal bar and a vertical bar are different shapes
  const rotated = [
    [1, 1, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
  ];
  assert.strictEqual(countDistinctIslands(rotated), 2);
  assert.deepStrictEqual([...canonicalize(collectIslandShapes(rotated))].sort(), ["0,0;0,1", "0,0;1,0"]);

  // Identical islands (even far apart) count once
  assert.strictEqual(countDistinctIslands([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]), 1);
  // ... but an L-shape and its mirror are two shapes
  assert.strictEqual(countDistinctIslands([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]]), 2);
  assert.strictEqual(countDistinctIslands([[1]]), 1);
  assert.strictEqual(countDistinctIslands([[0, 0], [0, 0]]), 0);
  assert.strictEqual(countDistinctIslands([]), 0);

  // Diagonal cells are not 4-connected: two separate islands, but both are single cells,
  // so they still share one shape. The 3x3 grid below distinguishes the two situations:
  // 4-connectivity gives {(0,0)} and {(0,0),(0,1)} -> 2 shapes, diagonal merging only 1.
  assert.strictEqual(countIslands([[1, 0], [0, 1]]), 2);
  assert.strictEqual(countDistinctIslands([[1, 0], [0, 1]]), 1);
  assert.strictEqual(countDistinctIslands([[1, 0, 0], [0, 1, 1], [0, 0, 0]]), 2);
  assert.strictEqual(countDistinctIslands([[1, 1, 1, 1]]), 1);
  assert.strictEqual(countDistinctIslands([[1], [1], [1]]), 1);

  // Zero padding on the top/left must not change the answer (shapes are translation invariant)
  const padded = Array.from({ length: 9 }, () => new Array(7).fill(0));
  for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) padded[r + 2][c + 2] = classic[r][c];
  assert.strictEqual(countDistinctIslands(padded), 3);

  const grids = [...lcgGrids(8, 8, 40), ...lcgGrids(6, 11, 20), ...lcgGrids(4, 4, 20)];
  for (const grid of grids) {
    const shapes = collectIslandShapes(grid);
    const oracle = islandShapesSortedTuples(grid);
    assert.deepStrictEqual([...canonicalize(shapes)].sort(), [...oracle].sort(), JSON.stringify(grid));
    assert.strictEqual(countDistinctIslands(grid), oracle.size, JSON.stringify(grid));
    for (const shape of shapes) {
      const minR = Math.min(...shape.map((key) => Number(key.split(",")[0])));
      const minC = Math.min(...shape.map((key) => Number(key.split(",")[1])));
      assert.strictEqual(minR, 0, JSON.stringify(grid));                 // keys are normalised ...
      assert.strictEqual(minC, 0, JSON.stringify(grid));
      assert.ok(isConnectedShape(shape), JSON.stringify(grid));          // ... and 4-connected
    }
  }

  console.log("PASS");
}
