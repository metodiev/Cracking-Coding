// Task 726 - Implement sparse-dense matrix multiply block algorithm optimizations. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// C = A * B with A sparse (CSR) and B dense is the workhorse of pruned neural networks and of iterative
// solvers. The arithmetic is fixed - every nonzero of A must scale a whole row of B - so the only thing
// a "block algorithm" can optimise is memory traffic, and this file counts that traffic instead of
// timing it:
//
//   CsrNaive     for each output column, sweep the row's nonzeros accumulating straight into C: every
//                update becomes a read-modify-write of C and the row's indices are re-read once per
//                column, so indexReads = nnz * n and cUpdates = nnz * n.
//   CsrBlocked   block the output columns: one pass over the nonzeros per block, accumulating in a
//                register buffer and writing each C element once: indexReads = nnz * ceil(n / block),
//                cUpdates = 0, cWrites <= m * n. Columns of B that are identically zero are skipped.
//   BcsrBlocked  store A as dense br x bc sub-blocks: one index covers br * bc values, so the index
//                traffic drops to the block count at the price of padding slots.
//
// Documented simplification: values are 32-bit integers so every comparison is exact and the counters
// describe layout effects rather than floating-point behaviour. Verified: all three variants agree with
// the dense reference and with a coordinate-wise brute force; the traffic identities above hold with the
// block count (not the nonzero count) driving the index traffic; empty rows and empty columns are
// skipped and counted; the degenerate shapes (no nonzeros, a single row, a single column, a zero-row
// matrix) behave; and the CSR layout is smaller than a dense image at low density.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal sealed class Traffic
{
    internal long DenseReads;           // elements of B read
    internal long CReadModifyWrites;    // in-place updates of C
    internal long CWrites;              // stores of C elements
    internal long IndexReads;           // column indices (or block indices) pulled from A
    internal long ValueSlots;           // A value slots visited, padded slots included
    internal long EmptyRowsSkipped;
    internal long EmptyColumnsSkipped;
}

internal sealed class CsrMatrix
{
    internal readonly int Rows;
    internal readonly int Cols;
    internal readonly int[] RowStart;
    internal readonly int[] Column;
    internal readonly int[] Value;

    internal CsrMatrix(int rows, int cols, List<int> rowStart, List<int> column, List<int> value)
    {
        Rows = rows;
        Cols = cols;
        RowStart = rowStart.ToArray();
        Column = column.ToArray();
        Value = value.ToArray();
    }

    internal int Nonzeros => Column.Length;
    internal int RowNonzeros(int row) => RowStart[row + 1] - RowStart[row];
    internal bool IsEmptyRow(int row) => RowNonzeros(row) == 0;

    /// <summary>Byte cost of the CSR layout (int index + int value + row pointers).</summary>
    internal long CsrBytes => 8L * Nonzeros + 4L * (Rows + 1);
}

internal sealed class DenseMatrix
{
    internal readonly int Rows;
    internal readonly int Cols;
    internal readonly int[] Data;

    internal DenseMatrix(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new int[rows * cols];
    }

    internal int Get(int i, int j) => Data[i * Cols + j];
    internal void Set(int i, int j, int value) => Data[i * Cols + j] = value;

    internal bool Same(DenseMatrix other)
    {
        if (Rows != other.Rows || Cols != other.Cols) return false;
        for (int i = 0; i < Data.Length; i++) if (Data[i] != other.Data[i]) return false;
        return true;
    }
}

internal static class Spmm
{
    /// <summary>Dense triple loop over the expanded A: the reference, deliberately not optimised.</summary>
    internal static DenseMatrix DenseReference(CsrMatrix a, DenseMatrix b)
    {
        var c = new DenseMatrix(a.Rows, b.Cols);
        if (a.Rows == 0 || a.Cols == 0) return c;
        var dense = new int[a.Rows * a.Cols];
        for (int i = 0; i < a.Rows; i++)
            for (int p = a.RowStart[i]; p < a.RowStart[i + 1]; p++)
                dense[i * a.Cols + a.Column[p]] = a.Value[p];
        for (int i = 0; i < a.Rows; i++)
            for (int k = 0; k < a.Cols; k++)
            {
                int av = dense[i * a.Cols + k];
                if (av == 0) continue;
                for (int j = 0; j < b.Cols; j++) c.Data[i * c.Cols + j] += av * b.Data[k * b.Cols + j];
            }
        return c;
    }

    /// <summary>Independent oracle: every output element sums its row's contributions directly.</summary>
    internal static DenseMatrix CoordinateOracle(CsrMatrix a, DenseMatrix b)
    {
        var c = new DenseMatrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
            for (int j = 0; j < b.Cols; j++)
            {
                int sum = 0;
                for (int p = a.RowStart[i]; p < a.RowStart[i + 1]; p++)
                    sum += a.Value[p] * b.Data[a.Column[p] * b.Cols + j];
                c.Data[i * c.Cols + j] = sum;
            }
        return c;
    }

    internal static DenseMatrix CsrNaive(CsrMatrix a, DenseMatrix b, Traffic traffic)
    {
        var c = new DenseMatrix(a.Rows, b.Cols);
        for (int i = 0; i < a.Rows; i++)
        {
            if (a.IsEmptyRow(i)) { traffic.EmptyRowsSkipped++; continue; }
            for (int j = 0; j < b.Cols; j++)
                for (int p = a.RowStart[i]; p < a.RowStart[i + 1]; p++)
                {
                    traffic.IndexReads++;
                    traffic.DenseReads++;
                    traffic.ValueSlots++;
                    traffic.CReadModifyWrites++;
                    c.Data[i * c.Cols + j] += a.Value[p] * b.Data[a.Column[p] * b.Cols + j];
                }
        }
        return c;
    }

    /// <summary>Column-blocked sweep: register accumulator, one store per C element, zero columns skipped.</summary>
    internal static DenseMatrix CsrBlocked(CsrMatrix a, DenseMatrix b, int blockCols, Traffic traffic)
    {
        if (blockCols < 1) throw new ArgumentOutOfRangeException(nameof(blockCols));
        var c = new DenseMatrix(a.Rows, b.Cols);
        var liveColumn = new bool[b.Cols];
        for (int j = 0; j < b.Cols; j++)
        {
            for (int k = 0; k < b.Rows; k++) if (b.Data[k * b.Cols + j] != 0) { liveColumn[j] = true; break; }
            if (!liveColumn[j]) traffic.EmptyColumnsSkipped++;
        }
        var accumulator = new int[blockCols];
        for (int i = 0; i < a.Rows; i++)
        {
            if (a.IsEmptyRow(i)) { traffic.EmptyRowsSkipped++; continue; }
            for (int j0 = 0; j0 < b.Cols; j0 += blockCols)
            {
                int width = Math.Min(blockCols, b.Cols - j0);
                for (int t = 0; t < width; t++) accumulator[t] = 0;
                for (int p = a.RowStart[i]; p < a.RowStart[i + 1]; p++)
                {
                    traffic.IndexReads++;
                    int av = a.Value[p];
                    int row = a.Column[p] * b.Cols;
                    for (int t = 0; t < width; t++)
                    {
                        traffic.ValueSlots++;
                        if (!liveColumn[j0 + t]) continue;
                        traffic.DenseReads++;
                        accumulator[t] += av * b.Data[row + j0 + t];
                    }
                }
                for (int t = 0; t < width; t++)
                {
                    if (!liveColumn[j0 + t]) continue;
                    traffic.CWrites++;
                    c.Data[i * c.Cols + j0 + t] = accumulator[t];
                }
            }
        }
        return c;
    }

    /// <summary>Blocked CSR: dense br x bc sub-blocks, one index per block, padded holes visited but skipped.</summary>
    internal static DenseMatrix BcsrBlocked(CsrMatrix a, DenseMatrix b, int blockRows, int blockCols, Traffic traffic, out int blockCount)
    {
        if (blockRows < 1 || blockCols < 1) throw new ArgumentOutOfRangeException(nameof(blockRows));
        var blocksPerRow = (a.Cols + blockCols - 1) / blockCols;
        var keys = new List<long>();
        var seen = new HashSet<long>();
        for (int i = 0; i < a.Rows; i++)
            for (int p = a.RowStart[i]; p < a.RowStart[i + 1]; p++)
            {
                long key = (long)(i / blockRows) * (blocksPerRow + 1) + a.Column[p] / blockCols;
                if (seen.Add(key)) keys.Add(key);
            }
        keys.Sort();
        blockCount = keys.Count;

        var c = new DenseMatrix(a.Rows, b.Cols);
        var sub = new int[blockRows * blockCols];
        foreach (long key in keys)
        {
            int blockRow = (int)(key / (blocksPerRow + 1));
            int blockCol = (int)(key % (blocksPerRow + 1));
            traffic.IndexReads++;
            for (int t = 0; t < sub.Length; t++) sub[t] = 0;
            for (int di = 0; di < blockRows; di++)
            {
                int row = blockRow * blockRows + di;
                if (row >= a.Rows) continue;
                for (int p = a.RowStart[row]; p < a.RowStart[row + 1]; p++)
                {
                    if (a.Column[p] / blockCols != blockCol) continue;
                    sub[di * blockCols + a.Column[p] % blockCols] = a.Value[p];
                }
            }
            for (int di = 0; di < blockRows; di++)
            {
                int row = blockRow * blockRows + di;
                if (row >= a.Rows) continue;
                for (int dj = 0; dj < blockCols; dj++)
                {
                    traffic.ValueSlots++;
                    int av = sub[di * blockCols + dj];
                    if (av == 0) continue;                       // a padded hole: visited, contributes nothing
                    int k = blockCol * blockCols + dj;
                    if (k >= b.Rows) continue;
                    for (int j = 0; j < b.Cols; j++)
                    {
                        traffic.DenseReads++;
                        traffic.CWrites++;
                        c.Data[row * c.Cols + j] += av * b.Data[k * b.Cols + j];
                    }
                }
            }
        }
        return c;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- every variant agrees with the independent oracle --------------------------------------
        {
            CsrMatrix a = RandomCsr(24, 20, 250, 0x726AUL, out _);
            DenseMatrix b = RandomDense(20, 18, 0x726BUL);
            DenseMatrix reference = Spmm.DenseReference(a, b);
            DenseMatrix oracle = Spmm.CoordinateOracle(a, b);
            Check(reference.Same(oracle), "the dense reference and the coordinate oracle agree");
            var naive = new Traffic();
            var blocked = new Traffic();
            var bcsr = new Traffic();
            Check(Spmm.CsrNaive(a, b, naive).Same(oracle), "the naive CSR multiply matches the oracle");
            Check(Spmm.CsrBlocked(a, b, 4, blocked).Same(oracle), "the column-blocked multiply matches the oracle");
            Check(Spmm.BcsrBlocked(a, b, 2, 2, bcsr, out int blocks).Same(oracle), "the blocked-CSR multiply matches the oracle");
            Console.WriteLine("m=" + a.Rows + " k=" + a.Cols + " n=" + b.Cols + " nnz=" + a.Nonzeros + " blocks=" + blocks
                + " naiveIndexReads=" + naive.IndexReads + " blockedIndexReads=" + blocked.IndexReads
                + " bcsrIndexReads=" + bcsr.IndexReads + " naiveCUpdates=" + naive.CReadModifyWrites
                + " blockedCWrites=" + blocked.CWrites);

            Check(naive.IndexReads == (long)a.Nonzeros * b.Cols, "the naive variant reads one index per column sweep");
            Check(naive.CReadModifyWrites == (long)a.Nonzeros * b.Cols, "the naive variant read-modify-writes C");
            Check(naive.ValueSlots == (long)a.Nonzeros * b.Cols, "the naive variant visits every nonzero per column");
            Check(blocked.CReadModifyWrites == 0 && blocked.CWrites > 0, "the blocked variant never read-modify-writes C");
            Check(blocked.CWrites == (long)a.Rows * b.Cols, "the blocked variant stores each C element exactly once");
            Check(blocked.IndexReads == (long)a.Nonzeros * ((b.Cols + 3) / 4), "blocking divides the index traffic by the block width");
            Check(blocked.IndexReads < naive.IndexReads / 3, "column blocking cuts the index traffic ("
                + blocked.IndexReads + " vs " + naive.IndexReads + ")");
            Check(blocked.DenseReads == naive.DenseReads, "blocking does not change how often B is read");
            Check(bcsr.IndexReads == blocks && blocks < a.Nonzeros,
                "one block index covers several nonzeros (" + blocks + " blocks for " + a.Nonzeros + " nonzeros)");
            Check(bcsr.ValueSlots == 4L * blocks, "the blocked format visits the padded sub-blocks");
            Check(bcsr.CWrites > 0 && bcsr.DenseReads == bcsr.CWrites, "each visited sub-block slot scales a row of B");
        }

        // ---- a 2x2-aligned matrix needs no padding at all -------------------------------------------
        {
            var rowStart = new List<int> { 0 };
            var column = new List<int>();
            var value = new List<int>();
            const int size = 32;
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++) if (i / 2 == j / 2) { column.Add(j); value.Add(1 + (i * 7 + j) % 5); }
                rowStart.Add(column.Count);
            }
            var structured = new CsrMatrix(size, size, rowStart, column, value);
            DenseMatrix b = RandomDense(size, size, 0x7265C7UL);
            var traffic = new Traffic();
            DenseMatrix c = Spmm.BcsrBlocked(structured, b, 2, 2, traffic, out int blocks);
            Check(c.Same(Spmm.CoordinateOracle(structured, b)), "the structured matrix multiplies correctly");
            Check(blocks * 4 == structured.Nonzeros, "a 2x2-aligned matrix needs no padding (" + blocks + " blocks)");
            Check(traffic.IndexReads == blocks, "one block index per block");
            Check(traffic.ValueSlots == structured.Nonzeros, "no padded slots for an aligned matrix");
            Check(traffic.IndexReads * 4 == structured.Nonzeros, "the index traffic is a quarter of the nonzero traffic");
        }

        // ---- empty rows and empty columns are skipped ------------------------------------------------
        {
            var rowStart = new List<int> { 0 };
            var column = new List<int>();
            var value = new List<int>();
            const int rows = 6, cols = 6;
            for (int i = 0; i < rows; i++)
            {
                if (i != 3) { column.Add(i); value.Add(i + 1); }
                rowStart.Add(column.Count);
            }
            var a = new CsrMatrix(rows, cols, rowStart, column, value);
            var b = new DenseMatrix(cols, 5);
            for (int i = 0; i < cols; i++) for (int j = 0; j < 5; j++) b.Set(i, j, j == 2 ? 0 : 1);
            var traffic = new Traffic();
            DenseMatrix c = Spmm.CsrBlocked(a, b, 4, traffic);
            Check(c.Same(Spmm.CoordinateOracle(a, b)), "the skip optimisations preserve the result");
            Check(traffic.EmptyRowsSkipped == 1, "the single empty row was skipped");
            Check(traffic.EmptyColumnsSkipped == 1, "the single zero column of B was skipped");
            Check(traffic.CWrites == 5L * 4, "only the live columns of the five non-empty rows are written ("
                + traffic.CWrites + ")");
            Check(a.IsEmptyRow(3) && !a.IsEmptyRow(0) && a.RowNonzeros(0) == 1, "row emptiness is reported correctly");
        }

        // ---- degenerate shapes -------------------------------------------------------------------------
        {
            var empty = new CsrMatrix(0, 0, new List<int> { 0 }, new List<int>(), new List<int>());
            var b0 = new DenseMatrix(0, 4);
            var traffic = new Traffic();
            Check(Spmm.CsrNaive(empty, b0, traffic).Rows == 0, "a zero-row matrix produces zero rows of output");
            Check(Spmm.CoordinateOracle(empty, b0).Same(new DenseMatrix(0, 4)), "and no values");
            Check(Spmm.DenseReference(empty, b0).Same(new DenseMatrix(0, 4)), "the dense reference too");

            var rowStart = new List<int> { 0, 0, 0, 0, 0 };
            var allZero = new CsrMatrix(4, 4, rowStart, new List<int>(), new List<int>());
            DenseMatrix b1 = RandomDense(4, 3, 0x7260FFUL);
            var zeroTraffic = new Traffic();
            Check(Spmm.CsrBlocked(allZero, b1, 2, zeroTraffic).Same(new DenseMatrix(4, 3)),
                "a matrix with no nonzeros multiplies to zero");
            Check(zeroTraffic.IndexReads == 0 && zeroTraffic.DenseReads == 0, "and does no work at all");
            Check(zeroTraffic.EmptyRowsSkipped == 4, "all four empty rows are counted");

            var single = new CsrMatrix(1, 1, new List<int> { 0, 1 }, new List<int> { 0 }, new List<int> { 5 });
            var b2 = new DenseMatrix(1, 1);
            b2.Set(0, 0, -3);
            var oneTraffic = new Traffic();
            Check(Spmm.CsrNaive(single, b2, oneTraffic).Get(0, 0) == -15, "a 1x1 sparse multiply is exact");
            Check(oneTraffic.IndexReads == 1 && oneTraffic.ValueSlots == 1, "one nonzero, one access");
            Check(Spmm.CsrBlocked(single, b2, 7, oneTraffic).Get(0, 0) == -15, "a block wider than the matrix still works");

            CsrMatrix tall = RandomCsr(10, 8, 400, 0x726C01UL, out _);
            DenseMatrix b3 = RandomDense(8, 1, 0x726DUL);
            Check(Spmm.CsrBlocked(tall, b3, 8, new Traffic()).Same(Spmm.CoordinateOracle(tall, b3)),
                "a single-column dense operand works with any block width");
            Check(Spmm.CsrNaive(tall, b3, new Traffic()).Same(Spmm.CoordinateOracle(tall, b3)),
                "the single-column case also matches the naive variant");

            bool threw = false;
            try { Spmm.CsrBlocked(tall, b3, 0, new Traffic()); } catch (ArgumentOutOfRangeException) { threw = true; }
            Check(threw, "a zero-width column block is rejected");
        }

        // ---- layout cost and reproducibility -------------------------------------------------------------
        {
            CsrMatrix sparse = RandomCsr(64, 64, 50, 0x7265ABUL, out int nonzeros);
            Check(sparse.CsrBytes < 64L * 64 * 8, "the sparse layout is smaller than a dense image at 5% density ("
                + sparse.CsrBytes + " vs " + (64L * 64 * 8) + ")");
            Check(nonzeros == sparse.Nonzeros && nonzeros > 0, "the generated matrix really has nonzeros");

            DenseMatrix b = RandomDense(64, 48, 0x726B1UL);
            var naive = new Traffic();
            var blocked = new Traffic();
            var bcsr = new Traffic();
            DenseMatrix c1 = Spmm.CsrNaive(sparse, b, naive);
            DenseMatrix c2 = Spmm.CsrBlocked(sparse, b, 8, blocked);
            DenseMatrix c3 = Spmm.BcsrBlocked(sparse, b, 2, 2, bcsr, out _);
            Check(c1.Same(c2) && c2.Same(c3), "all variants agree on the larger instance");
            Check(c1.Same(Spmm.CoordinateOracle(sparse, b)), "the larger instance also matches the oracle");
            var again = new Traffic();
            Check(Spmm.CsrBlocked(sparse, b, 8, again).Same(c2) && again.CWrites == blocked.CWrites,
                "the same input reproduces the same result and the same counters");
            Check(blocked.CWrites < naive.CReadModifyWrites, "the blocked variant writes far less often ("
                + blocked.CWrites + " vs " + naive.CReadModifyWrites + ")");
            Check(blocked.IndexReads < naive.IndexReads, "and reads far fewer indices ("
                + blocked.IndexReads + " vs " + naive.IndexReads + ")");
            Console.WriteLine("large: nnz=" + sparse.Nonzeros + " naiveIndexReads=" + naive.IndexReads
                + " blockedIndexReads=" + blocked.IndexReads + " bcsrIndexReads=" + bcsr.IndexReads
                + " blockedCWrites=" + blocked.CWrites + " naiveCUpdates=" + naive.CReadModifyWrites);
        }

        Console.WriteLine("PASS");
    }

    /// <summary>Random CSR matrix; density is in per-mille to stay exact in integer arithmetic.</summary>
    private static CsrMatrix RandomCsr(int rows, int cols, int permille, ulong seed, out int nonzeros)
    {
        var rng = new SplitMix64(seed);
        var rowStart = new List<int> { 0 };
        var column = new List<int>();
        var value = new List<int>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                if (rng.NextInt(1000) < permille) { column.Add(j); value.Add(1 + rng.NextInt(9)); }
            rowStart.Add(column.Count);
        }
        nonzeros = column.Count;
        return new CsrMatrix(rows, cols, rowStart, column, value);
    }

    private static DenseMatrix RandomDense(int rows, int cols, ulong seed)
    {
        var rng = new SplitMix64(seed);
        var matrix = new DenseMatrix(rows, cols);
        for (int i = 0; i < matrix.Data.Length; i++) matrix.Data[i] = rng.NextInt(7) - 3;
        return matrix;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
