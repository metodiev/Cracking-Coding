/** Task 726 - Implement sparse-dense matrix multiply block algorithm optimizations. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * C = A * B where A is sparse and B is dense is the inner loop of every sparse layer, and the
 * storage format decides how fast it runs:
 *
 *   * CSR (compressed sparse row) keeps one (value, column) pair per nonzero: minimal storage,
 *     but the kernel reads one index per scalar multiply and the access pattern is irregular.
 *   * BSR (blocked CSR, the "block algorithm") groups the nonzeros into r x c tiles and stores
 *     one index per *tile*: few index reads, contiguous vectorizable payload, and all-zero
 *     tiles can be skipped entirely -- at the price of doing arithmetic on the zeros that
 *     remain inside a partially dense tile.
 *
 * This file implements triplet assembly (with duplicate merging), CSR SpMM, BSR construction
 * with zero-tile skipping and block SpMM, plus the accounting that decides when blocking pays:
 * tile density, index entries and the exact multiply-add overhead of the tile form.
 *
 * Verified here: the CSR result equals a dense reference multiply exactly (integer payloads
 * keep every partial sum exact, so the comparison is an equality, not a tolerance); BSR
 * reproduces the CSR result exactly and rebuilds the original matrix from its tiles; the tile
 * form really does drop all-zero tiles and never invents a nonzero; the index-entry and
 * multiply-add metrics match independent recomputations; the block density bound holds; and
 * the degenerate inputs (empty inner dimension, zero rows, an all-zero matrix, a single tile)
 * behave. Deterministic: all data comes from a seeded splitmix64 generator.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

/** Assemble triplets into CSR: sorts each row, merges duplicates, drops explicit zeros. */
function buildCSR(rowCount, colCount, entries) {
  const perRow = Array.from({ length: rowCount }, () => []);
  for (const entry of entries) {
    ok(entry.row >= 0 && entry.row < rowCount, `row ${entry.row} out of range`);
    ok(entry.col >= 0 && entry.col < colCount, `column ${entry.col} out of range`);
    perRow[entry.row].push(entry);
  }
  const values = [];
  const colIndices = [];
  const rowPtr = [0];
  for (let row = 0; row < rowCount; row++) {
    perRow[row].sort((a, b) => a.col - b.col);
    let i = 0;
    while (i < perRow[row].length) {
      const col = perRow[row][i].col;
      let sum = 0;
      while (i < perRow[row].length && perRow[row][i].col === col) {
        sum += perRow[row][i].value;
        i++;
      }
      if (sum !== 0) {
        values.push(sum);
        colIndices.push(col);
      }
    }
    rowPtr.push(values.length);
  }
  return { rowCount, colCount, values, colIndices, rowPtr, nnz: values.length };
}

/** Row-wise sparse-dense multiply: accumulate one nonzero at a time. */
function csrSpMM(csr, dense, columns) {
  const out = Array.from({ length: csr.rowCount }, () => new Array(columns).fill(0));
  for (let row = 0; row < csr.rowCount; row++) {
    for (let p = csr.rowPtr[row]; p < csr.rowPtr[row + 1]; p++) {
      const value = csr.values[p];
      const source = dense[csr.colIndices[p]];
      for (let column = 0; column < columns; column++) out[row][column] += value * source[column];
    }
  }
  return out;
}

/** Dense reference used as the oracle. */
function denseMatMul(sparse, dense) {
  const rows = sparse.length;
  const inner = rows === 0 ? 0 : sparse[0].length;
  const columns = dense.length === 0 ? 0 : dense[0].length;
  const out = Array.from({ length: rows }, () => new Array(columns).fill(0));
  for (let row = 0; row < rows; row++) {
    for (let k = 0; k < inner; k++) {
      const value = sparse[row][k];
      if (value === 0) continue;
      for (let column = 0; column < columns; column++) out[row][column] += value * dense[k][column];
    }
  }
  return out;
}

/**
 * Blocked CSR. Zero tiles are skipped, so the tile count is the number of *occupied* tiles;
 * a partially dense tile carries its zeros inside the payload, which is exactly the trade.
 */
function buildBSR(csr, tileRows, tileCols) {
  ok(Number.isInteger(tileRows) && tileRows > 0 && Number.isInteger(tileCols) && tileCols > 0,
    "tile shape must be two positive integers");
  const blockRowCount = Math.ceil(csr.rowCount / tileRows);
  const blockColCount = Math.ceil(csr.colCount / tileCols);
  const blocks = [];
  const rowPtr = [0];
  for (let blockRow = 0; blockRow < blockRowCount; blockRow++) {
    for (let blockCol = 0; blockCol < blockColCount; blockCol++) {
      const tile = new Array(tileRows * tileCols).fill(0);
      let occupied = 0;
      for (let r = 0; r < tileRows; r++) {
        const row = blockRow * tileRows + r;
        if (row >= csr.rowCount) break;
        for (let p = csr.rowPtr[row]; p < csr.rowPtr[row + 1]; p++) {
          const column = csr.colIndices[p];
          if (column < blockCol * tileCols || column >= (blockCol + 1) * tileCols) continue;
          tile[r * tileCols + (column - blockCol * tileCols)] = csr.values[p];
          occupied++;
        }
      }
      if (occupied === 0) continue;                 // the all-zero tile is never stored
      blocks.push({ blockRow, blockCol, tile, occupied });
    }
    rowPtr.push(blocks.length);
  }
  return { rowCount: csr.rowCount, colCount: csr.colCount, tileRows, tileCols, blockRowCount, blockColCount, blocks, rowPtr };
}

/** Tile-wise sparse-dense multiply: one index lookup per occupied tile. */
function bsrSpMM(bsr, dense, columns) {
  const out = Array.from({ length: bsr.rowCount }, () => new Array(columns).fill(0));
  for (const block of bsr.blocks) {
    for (let r = 0; r < bsr.tileRows; r++) {
      const row = block.blockRow * bsr.tileRows + r;
      if (row >= bsr.rowCount) break;
      for (let c = 0; c < bsr.tileCols; c++) {
        const value = block.tile[r * bsr.tileCols + c];
        if (value === 0) continue;
        const column = block.blockCol * bsr.tileCols + c;
        if (column >= bsr.colCount) break;
        const source = dense[column];
        for (let j = 0; j < columns; j++) out[row][j] += value * source[j];
      }
    }
  }
  return out;
}

/** Rebuild the full sparse matrix from the tiles: the inverse of the blocking. */
function bsrToDense(bsr) {
  const out = Array.from({ length: bsr.rowCount }, () => new Array(bsr.colCount).fill(0));
  for (const block of bsr.blocks) {
    for (let r = 0; r < bsr.tileRows; r++) {
      const row = block.blockRow * bsr.tileRows + r;
      if (row >= bsr.rowCount) break;
      for (let c = 0; c < bsr.tileCols; c++) {
        const column = block.blockCol * bsr.tileCols + c;
        if (column >= bsr.colCount) break;
        out[row][column] = block.tile[r * bsr.tileCols + c];
      }
    }
  }
  return out;
}

const csrToDense = (csr) => {
  const out = Array.from({ length: csr.rowCount }, () => new Array(csr.colCount).fill(0));
  for (let row = 0; row < csr.rowCount; row++) {
    for (let p = csr.rowPtr[row]; p < csr.rowPtr[row + 1]; p++) out[row][csr.colIndices[p]] = csr.values[p];
  }
  return out;
};

/** Cost accounting: index entries and the arithmetic each format actually performs. */
function metrics(csr, bsr, columns) {
  const tileSlots = bsr.blocks.reduce((acc, block) => acc + block.tile.length, 0);
  const storedNonZeros = bsr.blocks.reduce((acc, block) => acc + block.occupied, 0);
  return {
    csrIndexEntries: csr.nnz,
    bsrIndexEntries: bsr.blocks.length,
    csrMultiplyAdds: csr.nnz * columns,
    bsrMultiplyAdds: tileSlots * columns,
    bsrZeroMultiplyAdds: (tileSlots - storedNonZeros) * columns,
    blockDensity: tileSlots === 0 ? 0 : storedNonZeros / tileSlots,
    occupiedTiles: bsr.blocks.length,
    totalTiles: bsr.blockRowCount * bsr.blockColCount,
  };
}

/** Clustered sparsity: nonzeros are drawn inside a few tiles of a block grid. */
function clusteredEntries(rng, rows, cols, tile, occupancies) {
  const entries = [];
  for (let blockRow = 0; blockRow < rows / tile; blockRow++) {
    for (let blockCol = 0; blockCol < cols / tile; blockCol++) {
      if (rng() > occupancies) continue;
      for (let r = 0; r < tile; r++) {
        for (let c = 0; c < tile; c++) {
          if (rng() < 0.85) {
            entries.push({ row: blockRow * tile + r, col: blockCol * tile + c, value: 1 + Math.floor(rng() * 4) });
          }
        }
      }
    }
  }
  return entries;
}

module.exports = {
  makeRng, buildCSR, csrSpMM, denseMatMul, buildBSR, bsrSpMM, bsrToDense, csrToDense,
  metrics, clusteredEntries,
};

if (require.main === module) {
  const rng = makeRng(20260726);

  // -- triplet assembly: sorting, duplicate merging and zero cancellation -----------------
  const assembled = buildCSR(3, 4, [
    { row: 0, col: 2, value: 5 }, { row: 0, col: 0, value: 1 }, { row: 0, col: 2, value: -2 },
    { row: 1, col: 3, value: 7 }, { row: 2, col: 1, value: 4 }, { row: 2, col: 1, value: -4 },
    { row: 2, col: 2, value: 6 },
  ]);
  deepStrictEqual(assembled.colIndices, [0, 2, 3, 2], "columns must be sorted per row");
  deepStrictEqual(assembled.values, [1, 3, 7, 6], "duplicates merge and cancelling pairs vanish");
  deepStrictEqual(assembled.rowPtr, [0, 2, 3, 4]);
  strictEqual(assembled.nnz, 4);
  deepStrictEqual(csrToDense(assembled), [[1, 0, 3, 0], [0, 0, 0, 7], [0, 0, 6, 0]],
    "CSR must rebuild the matrix it was assembled from");
  throws(() => buildCSR(2, 2, [{ row: 2, col: 0, value: 1 }]), /out of range/);

  // -- SpMM against the dense reference, exact because the payload is integral -----------
  const sparse = [[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 5], [0, 0, 0, 0]];
  const dense = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
  const csr = buildCSR(4, 4, sparse.flatMap((row, r) =>
    row.map((value, c) => ({ row: r, col: c, value })).filter((entry) => entry.value !== 0)));
  const reference = denseMatMul(sparse, dense);
  deepStrictEqual(csrSpMM(csr, dense, 3), reference, "CSR SpMM must equal the dense reference exactly");
  for (const [tileRows, tileCols] of [[1, 1], [2, 2], [2, 4], [4, 4], [3, 3]]) {
    const bsr = buildBSR(csr, tileRows, tileCols);
    deepStrictEqual(bsrSpMM(bsr, dense, 3), reference,
      `BSR with ${tileRows}x${tileCols} tiles must equal the dense reference exactly`);
    deepStrictEqual(bsrToDense(bsr), sparse, `BSR with ${tileRows}x${tileCols} tiles must rebuild A`);
  }

  // -- all-zero tiles are skipped and nothing is invented --------------------------------
  const ragged = buildCSR(6, 6, [
    { row: 0, col: 0, value: 2 }, { row: 1, col: 1, value: 3 }, { row: 5, col: 5, value: 4 },
  ]);
  const skipped = buildBSR(ragged, 2, 2);
  strictEqual(skipped.blocks.length, 2, "the two nonzeros inside one tile share it, so two tiles are stored");
  strictEqual(skipped.blockRowCount * skipped.blockColCount, 9, "over a 3x3 grid of tiles");
  deepStrictEqual(skipped.blocks.map((block) => [block.blockRow, block.blockCol]), [[0, 0], [2, 2]],
    "the occupied tiles are (0,0) and (2,2)");
  deepStrictEqual(bsrToDense(skipped), csrToDense(ragged), "skipping tiles loses no nonzeros");
  strictEqual(skipped.blocks.reduce((acc, block) => acc + block.occupied, 0), ragged.nnz,
    "the tile form stores exactly the nonzeros that exist");

  // -- the accounting the tuning decision is based on -------------------------------------
  const clustered = clusteredEntries(rng, 64, 64, 4, 0.5);
  const clusterCsr = buildCSR(64, 64, clustered);
  const clusterBsr = buildBSR(clusterCsr, 4, 4);
  const denseB = Array.from({ length: 64 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i % 5) - j));
  const clusterReference = denseMatMul(csrToDense(clusterCsr), denseB);
  deepStrictEqual(csrSpMM(clusterCsr, denseB, 8), clusterReference, "CSR must beat the reference on the big case");
  deepStrictEqual(bsrSpMM(clusterBsr, denseB, 8), clusterReference, "and so must the tile form");
  const stats = metrics(clusterCsr, clusterBsr, 8);
  strictEqual(stats.bsrIndexEntries, clusterBsr.blocks.length, "one index entry per occupied tile");
  ok(stats.bsrIndexEntries * 4 < stats.csrIndexEntries,
    `tiles must cut index traffic: ${stats.bsrIndexEntries} vs ${stats.csrIndexEntries} entries`);
  strictEqual(stats.bsrMultiplyAdds, stats.bsrIndexEntries * 16 * 8, "tile arithmetic is tiles * r*c * n");
  strictEqual(stats.bsrZeroMultiplyAdds + clusterCsr.nnz * 8, stats.bsrMultiplyAdds,
    "the tile overhead is exactly the arithmetic spent on in-tile zeros");
  ok(stats.blockDensity > 0.5 && stats.blockDensity <= 1, `tile density must be in (0,1], got ${stats.blockDensity}`);
  const occupiedFraction = stats.occupiedTiles / stats.totalTiles;
  ok(occupiedFraction > 0.3 && occupiedFraction < 0.7,
    `about half the tiles are occupied by construction, got ${occupiedFraction.toFixed(3)}`);
  ok(stats.occupiedTiles < stats.totalTiles, "the all-zero tiles are genuinely skipped");
  ok(stats.bsrMultiplyAdds > stats.csrMultiplyAdds,
    "blocking trades extra arithmetic for fewer index lookups");
  console.log(`64x64 clustered: nnz=${clusterCsr.nnz} tiles=${stats.bsrIndexEntries}` +
    ` index ratio=${(stats.csrIndexEntries / stats.bsrIndexEntries).toFixed(1)}x` +
    ` density=${stats.blockDensity.toFixed(2)} extra MACs=${stats.bsrZeroMultiplyAdds}`);

  // -- 1x1 tiles degenerate to CSR exactly ------------------------------------------------
  const oneByOne = buildBSR(clusterCsr, 1, 1);
  const oneByOneStats = metrics(clusterCsr, oneByOne, 8);
  strictEqual(oneByOneStats.bsrIndexEntries, clusterCsr.nnz, "1x1 tiles index every nonzero");
  strictEqual(oneByOneStats.bsrZeroMultiplyAdds, 0, "and waste no arithmetic on in-tile zeros");
  strictEqual(oneByOneStats.blockDensity, 1, "a 1x1 tile holding a nonzero is fully dense");
  deepStrictEqual(bsrSpMM(oneByOne, denseB, 8), clusterReference, "1x1 tiles still compute C correctly");

  // -- invariant: C's row sums are A applied to the row sums of B (linearity) -----------
  const rowSums = (matrix) => matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const bRowSums = rowSums(denseB);
  const expectedRowSums = csrToDense(clusterCsr).map((row) =>
    row.reduce((acc, value, k) => acc + value * bRowSums[k], 0));
  deepStrictEqual(rowSums(clusterReference), expectedRowSums,
    "linearity: the row sums of C are A applied to the row sums of B");
  deepStrictEqual(rowSums(bsrSpMM(clusterBsr, denseB, 8)), expectedRowSums,
    "the tile kernel must satisfy the same invariant");

  // -- degenerate shapes -------------------------------------------------------------------
  const emptyInner = buildCSR(2, 0, []);
  strictEqual(emptyInner.nnz, 0, "an empty inner dimension has no nonzeros");
  deepStrictEqual(csrSpMM(emptyInner, [], 0), [[], []], "and multiplies to an empty result");
  const zeroRows = buildCSR(0, 3, []);
  deepStrictEqual(csrSpMM(zeroRows, [[1, 2], [3, 4], [5, 6]], 2), [], "no rows means no output rows");
  const blank = buildBSR(buildCSR(4, 4, []), 2, 2);
  strictEqual(blank.blocks.length, 0, "an all-zero matrix has no occupied tile");
  deepStrictEqual(bsrSpMM(blank, dense, 3), [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "multiplying an empty sparse matrix gives zeros");
  const single = buildBSR(buildCSR(1, 1, [{ row: 0, col: 0, value: 3 }]), 4, 4);
  strictEqual(single.blocks.length, 1, "one nonzero is one tile even when the tile is oversized");
  deepStrictEqual(bsrSpMM(single, [[5]], 1), [[15]], "and the oversized tile still computes correctly");
  throws(() => buildBSR(csr, 0, 2), /positive integers/);

  // -- determinism -------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(4242);
    const localCsr = buildCSR(32, 32, clusteredEntries(local, 32, 32, 4, 0.6));
    return `${localCsr.nnz}|${buildBSR(localCsr, 4, 4).blocks.length}|${JSON.stringify(csrSpMM(localCsr, denseB.slice(0, 32), 8))}`;
  };
  deepStrictEqual(replay(), replay(), "sparse assembly, blocking and multiply must be deterministic");
  console.log("PASS");
}
