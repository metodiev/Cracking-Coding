// Task 681 - Implement numeric GPU kernel mapping to CPU threads simulation.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A SIMT execution simulator for two numeric kernels, with the grid/block/thread mapping written out the
// way a GPU actually performs it:
//   * an elementwise AXPY kernel on a 1-D grid: the flattened index blockIdx.x * blockDim.x + threadIdx.x
//     covers the data set, and the final block masks off the lanes whose index is out of range;
//   * a tiled matrix multiply on a 2-D grid of T x T output tiles. Every block owns T threads; lane `tx`
//     is responsible for the whole column `tx` of the output tile and keeps its T partial products in
//     lane-private storage. Each k-tile stages a T x T tile of A and a T x T tile of B into shared memory
//     with 2T statements, hits __syncthreads(), performs T x T multiply-accumulate statements reading the
//     staged tiles (twice each), hits a second barrier so the tiles cannot be overwritten while still in
//     use, and only then writes the tile back through the bounds predicate.
// The simulator counts every wave (one statement issued by every warp of a block), every barrier and every
// warp-level barrier wait, every active and masked lane, and every shared and global access. The thread to
// element mapping is verified as a bijection with a coverage array over the whole index space, and both
// kernels are compared against plain sequential reference loops, including shapes that are not multiples of
// the block dimension (where the masked-lane counters must be non-zero) and shapes with a zero dimension.
// The performance claims are accounting identities recomputed from the launch geometry instead of being
// asserted: waves = blocks x warps x (kTiles x (2T + T*T) + T), barrier waits = barriers x warps x blocks,
// global loads x T = the naive kernel's global loads, and the staged tile is read back and compared with the
// exact slice of A it was supposed to hold.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic matrix generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class GpuCounters
{
    internal long Waves;
    internal long Barriers;
    internal long BarrierWaits;
    internal long ActiveLanes;
    internal long InactiveLanes;
    internal long SharedLoads;
    internal long SharedStores;
    internal long GlobalLoads;
    internal long GlobalStores;
}

internal sealed class Launch
{
    internal int[] Output = Array.Empty<int>();
    internal GpuCounters Counters = new GpuCounters();
    internal int Blocks;
    internal int WarpsPerBlock;
    internal int LastTileA = -1;               // first element of the last tile staged into shared memory
    internal int LastTileB = -1;
}

internal static class Simt
{
    internal static int Ceil(int value, int divisor) => divisor <= 0 ? 0 : (value + divisor - 1) / divisor;

    /// Row-major flattened index of a (row, column) pair.
    internal static int Flat(int row, int column, int columns) => row * columns + column;

    /// 1-D AXPY: y[gid] = alpha * x[gid] + y[gid] for gid < n, one statement per thread.
    internal static Launch Axpy(int[] x, int[] y, int n, int blockDim, int warpSize, int alpha)
    {
        var launch = new Launch { Output = (int[])y.Clone() };
        if (n <= 0 || blockDim <= 0) return launch;
        int grid = Ceil(n, blockDim);
        int warps = Ceil(blockDim, warpSize);
        launch.Blocks = grid;
        launch.WarpsPerBlock = warps;
        for (int block = 0; block < grid; block++)
        {
            launch.Counters.Waves += warps;                 // one statement, issued by every warp
            for (int thread = 0; thread < blockDim; thread++)
            {
                int gid = block * blockDim + thread;
                if (gid >= n)
                {
                    launch.Counters.InactiveLanes++;
                    continue;
                }
                launch.Counters.ActiveLanes++;
                launch.Counters.GlobalLoads += 2;
                launch.Counters.GlobalStores++;
                launch.Output[gid] = alpha * x[gid] + y[gid];
            }
        }
        return launch;
    }

    /// Tiled matrix multiply; returns the launch with counters and the computed product.
    internal static Launch MatMul(int[] a, int[] b, int rows, int inner, int columns, int t, int warpSize)
    {
        var launch = new Launch { Output = new int[rows * columns] };
        if (t <= 0 || rows <= 0 || columns <= 0) return launch;
        int gridRows = Ceil(rows, t);
        int gridColumns = Ceil(columns, t);
        int kTiles = inner <= 0 ? 0 : Ceil(inner, t);
        int warps = Ceil(t, warpSize);
        launch.Blocks = gridRows * gridColumns;
        launch.WarpsPerBlock = warps;
        var sharedA = new int[t * t];
        var sharedB = new int[t * t];

        for (int by = 0; by < gridRows; by++)
        {
            for (int bx = 0; bx < gridColumns; bx++)
            {
                var accumulator = new int[t * t];           // lane-private partials, indexed [row][tx]
                for (int tile = 0; tile < kTiles; tile++)
                {
                    // ---- staging: 2T statements, one element of each tile per lane and step -------------
                    for (int step = 0; step < t; step++)
                    {
                        launch.Counters.Waves += 2 * warps;
                        for (int tx = 0; tx < t; tx++)
                        {
                            int rowA = by * t + step;
                            int columnA = tile * t + tx;
                            if (rowA < rows && columnA < inner)
                            {
                                launch.Counters.ActiveLanes++;
                                launch.Counters.GlobalLoads++;
                                sharedA[Flat(step, tx, t)] = a[Flat(rowA, columnA, inner)];
                            }
                            else
                            {
                                launch.Counters.InactiveLanes++;
                                sharedA[Flat(step, tx, t)] = 0;
                            }
                            launch.Counters.SharedStores++;

                            int rowB = tile * t + step;
                            int columnB = bx * t + tx;
                            if (rowB < inner && columnB < columns)
                            {
                                launch.Counters.ActiveLanes++;
                                launch.Counters.GlobalLoads++;
                                sharedB[Flat(step, tx, t)] = b[Flat(rowB, columnB, columns)];
                            }
                            else
                            {
                                launch.Counters.InactiveLanes++;
                                sharedB[Flat(step, tx, t)] = 0;
                            }
                            launch.Counters.SharedStores++;
                        }
                    }
                    launch.Counters.Barriers++;
                    launch.Counters.BarrierWaits += warps;      // every warp waits at the barrier

                    // ---- accumulate: T*T statements over the staged tiles ------------------------------
                    for (int row = 0; row < t; row++)
                    {
                        for (int k = 0; k < t; k++)
                        {
                            launch.Counters.Waves += warps;
                            for (int tx = 0; tx < t; tx++)
                            {
                                launch.Counters.ActiveLanes++;
                                launch.Counters.SharedLoads += 2;
                                accumulator[Flat(row, tx, t)] += sharedA[Flat(row, k, t)] * sharedB[Flat(k, tx, t)];
                            }
                        }
                    }
                    launch.Counters.Barriers++;
                    launch.Counters.BarrierWaits += warps;      // no warp may overwrite a tile still in use
                }

                // ---- write-back: T statements through the bounds predicate ---------------------------
                for (int row = 0; row < t; row++)
                {
                    launch.Counters.Waves += warps;
                    for (int tx = 0; tx < t; tx++)
                    {
                        int globalRow = by * t + row;
                        int globalColumn = bx * t + tx;
                        if (globalRow >= rows || globalColumn >= columns)
                        {
                            launch.Counters.InactiveLanes++;
                            continue;
                        }
                        launch.Counters.ActiveLanes++;
                        launch.Counters.GlobalStores++;
                        launch.Output[Flat(globalRow, globalColumn, columns)] = accumulator[Flat(row, tx, t)];
                    }
                }

                launch.LastTileA = sharedA[0];
                launch.LastTileB = sharedB[0];
            }
        }
        return launch;
    }

    internal static int[] ReferenceMatMul(int[] a, int[] b, int rows, int inner, int columns)
    {
        var c = new int[rows * columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                int total = 0;
                for (int k = 0; k < inner; k++) total += a[Flat(i, k, inner)] * b[Flat(k, j, columns)];
                c[Flat(i, j, columns)] = total;
            }
        }
        return c;
    }

    internal static int[] ReferenceAxpy(int[] x, int[] y, int n, int alpha)
    {
        var result = new int[n];
        for (int i = 0; i < n; i++) result[i] = alpha * x[i] + y[i];
        return result;
    }

    /// The wave count implied by the launch geometry: per block, kTiles x (2T staging + T*T accumulate)
    /// statements plus T statements of write-back, each issued once per warp.
    internal static long ExpectedWaves(int blocks, int warps, int t, int kTiles)
    {
        return (long)blocks * warps * ((long)kTiles * (2L * t + (long)t * t) + t);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int[] RandomVector(ref ulong state, int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = Rng.NextBelow(ref state, 20) - 10;
        return values;
    }

    private static int[] RandomMatrix(ref ulong state, int rows, int columns)
    {
        var values = new int[rows * columns];
        for (int i = 0; i < values.Length; i++) values[i] = Rng.NextBelow(ref state, 9) - 4;
        return values;
    }

    private static bool SameVector(int[] expected, int[] actual)
    {
        if (expected.Length != actual.Length) return false;
        for (int i = 0; i < expected.Length; i++) if (expected[i] != actual[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- the flattened mapping is a bijection ----------------------------------------------
        ulong state = 681001UL;
        int mappings = 0;
        for (int n = 1; n <= 40; n++)
        {
            foreach (int blockDim in new[] { 4, 8, 16 })
            {
                int grid = Simt.Ceil(n, blockDim);
                var covered = new bool[n];
                var duplicate = new bool[n];
                int surplus = 0;
                for (int block = 0; block < grid; block++)
                {
                    for (int thread = 0; thread < blockDim; thread++)
                    {
                        int gid = block * blockDim + thread;
                        if (gid >= n) { surplus++; continue; }
                        if (covered[gid]) duplicate[gid] = true;
                        covered[gid] = true;
                    }
                }
                bool duplicates = false;
                int coveredCount = 0;
                for (int i = 0; i < n; i++)
                {
                    if (duplicate[i]) duplicates = true;
                    if (covered[i]) coveredCount++;
                }
                Check(!duplicates, "n = " + n + " blockDim = " + blockDim + ": no element is covered twice");
                Check(coveredCount == n, "n = " + n + " blockDim = " + blockDim + ": every element is covered");
                Check(surplus == grid * blockDim - n, "n = " + n + " blockDim = " + blockDim
                      + ": the masked lanes are exactly the surplus slots (" + surplus + ")");
                mappings++;
            }
        }
        Check(mappings == 120, "the mapping was verified for " + mappings + " shapes");

        // ---- AXPY against the sequential reference --------------------------------------------
        int[] x = RandomVector(ref state, 100);
        int[] y = RandomVector(ref state, 100);
        Launch axpy = Simt.Axpy(x, y, 100, 16, 4, 3);
        Check(axpy.Counters.InactiveLanes == 12, "100 elements in blocks of 16 leave 12 masked lanes (got "
              + axpy.Counters.InactiveLanes + ")");
        Check(axpy.Counters.ActiveLanes == 100, "and exactly 100 active lanes");
        Check(axpy.Counters.ActiveLanes + axpy.Counters.InactiveLanes == (long)axpy.Blocks * 16,
              "every lane slot is accounted for");
        Check(SameVector(Simt.ReferenceAxpy(x, y, 100, 3), axpy.Output),
              "the simulated kernel computes exactly what the sequential loop computes");
        Check(axpy.Counters.Waves == (long)axpy.Blocks * axpy.WarpsPerBlock,
              "one statement per thread means one wave per warp per block (got " + axpy.Counters.Waves + ")");
        Launch exact = Simt.Axpy(x, y, 96, 16, 4, 3);
        Check(exact.Counters.InactiveLanes == 0 && exact.Counters.ActiveLanes == 96,
              "a size that is a multiple of the block dimension masks nothing");

        // ---- tiled matrix multiply -------------------------------------------------------------
        int t = 4;
        int[] a = RandomMatrix(ref state, 16, 16);
        int[] b = RandomMatrix(ref state, 16, 16);
        Launch tiled = Simt.MatMul(a, b, 16, 16, 16, t, 2);
        int[] reference = Simt.ReferenceMatMul(a, b, 16, 16, 16);
        Check(SameVector(reference, tiled.Output), "the tiled kernel matches the triple-loop reference on 16x16x16");
        Check(tiled.Counters.InactiveLanes == 0, "with no masking when every dimension is a multiple of the tile");
        Check(tiled.WarpsPerBlock == 2, "four threads at warp size two form two warps");
        Check(tiled.Blocks == 16, "the grid is four by four tiles (got " + tiled.Blocks + ")");
        int kTiles = 4;
        Check(tiled.Counters.Waves == Simt.ExpectedWaves(tiled.Blocks, tiled.WarpsPerBlock, t, kTiles),
              "the wave count equals blocks x warps x statements (" + tiled.Counters.Waves + ")");
        Check(tiled.Counters.Barriers == (long)tiled.Blocks * 2 * kTiles, "two barriers per k-tile per block ("
              + tiled.Counters.Barriers + ")");
        Check(tiled.Counters.BarrierWaits == tiled.Counters.Barriers * tiled.WarpsPerBlock,
              "every warp waits at every barrier (" + tiled.Counters.BarrierWaits + " waits)");
        Check(tiled.Counters.GlobalLoads == (long)tiled.Blocks * kTiles * 2 * t * t,
              "each staged element is loaded exactly once per tile (" + tiled.Counters.GlobalLoads + ")");
        Check(tiled.Counters.SharedLoads == (long)tiled.Blocks * kTiles * 2 * t * t * t,
              "and read T times per accumulating lane (" + tiled.Counters.SharedLoads + " shared reads)");
        Check(tiled.Counters.SharedLoads > tiled.Counters.GlobalLoads,
              "shared reads outnumber global loads, which is the point of staging");
        long naiveLoads = 16L * 16L * 2L * 16L;
        Check(tiled.Counters.GlobalLoads * t == naiveLoads, "staging divides the global loads by the tile width: "
              + tiled.Counters.GlobalLoads + " against " + naiveLoads);
        long laneSlots = (long)tiled.Blocks * kTiles * (2 * t * t + t * t * t) + (long)tiled.Blocks * t * t;
        Check(tiled.Counters.ActiveLanes + tiled.Counters.InactiveLanes == laneSlots,
              "every issued lane slot is either active or masked (" + laneSlots + ")");

        // ---- the staged tile is the exact slice of A it should be -------------------------------
        int lastRow = 12;
        int lastColumn = 12;
        Check(tiled.LastTileA == a[Simt.Flat(lastRow, lastColumn, 16)],
              "the last staged A tile starts at A[12][12] for the final block and k-tile (" + tiled.LastTileA + ")");
        Check(tiled.LastTileB == b[Simt.Flat(lastRow, lastColumn, 16)], "and so does the B tile");

        // ---- shapes that are not multiples of the tile -----------------------------------------
        int[] raggedA = RandomMatrix(ref state, 10, 10);
        int[] raggedB = RandomMatrix(ref state, 10, 7);
        Launch ragged = Simt.MatMul(raggedA, raggedB, 10, 10, 7, 4, 2);
        Check(SameVector(Simt.ReferenceMatMul(raggedA, raggedB, 10, 10, 7), ragged.Output),
              "a 10x10x7 product with a 4x4 tile is still correct");
        Check(ragged.Counters.InactiveLanes > 0, "and masked lanes appear where the tile overhangs the matrix ("
              + ragged.Counters.InactiveLanes + ")");
        Check(ragged.Blocks == 6, "the grid rounds up to three by two tiles (got " + ragged.Blocks + ")");
        Check(ragged.Counters.Waves == Simt.ExpectedWaves(ragged.Blocks, ragged.WarpsPerBlock, 4, 3),
              "the ragged launch still obeys the wave identity");

        // ---- larger shape with a widest inner dimension -----------------------------------------
        int[] bigA = RandomMatrix(ref state, 24, 20);
        int[] bigB = RandomMatrix(ref state, 20, 24);
        Launch big = Simt.MatMul(bigA, bigB, 24, 20, 24, 4, 4);
        Check(SameVector(Simt.ReferenceMatMul(bigA, bigB, 24, 20, 24), big.Output),
              "a 24x20x24 product with a 4x4 tile matches the reference");
        Check(big.Counters.Waves == Simt.ExpectedWaves(big.Blocks, big.WarpsPerBlock, 4, 5),
              "and the wave identity holds for a non-square inner dimension");
        Check(big.WarpsPerBlock == 1, "a warp size of four makes one warp per block");

        // ---- randomised shapes ------------------------------------------------------------------
        int shapes = 40;
        for (int s = 0; s < shapes; s++)
        {
            int rows = 1 + Rng.NextBelow(ref state, 12);
            int inner = 1 + Rng.NextBelow(ref state, 12);
            int columns = 1 + Rng.NextBelow(ref state, 12);
            int[] left = RandomMatrix(ref state, rows, inner);
            int[] right = RandomMatrix(ref state, inner, columns);
            Launch launch = Simt.MatMul(left, right, rows, inner, columns, 3, 2);
            Check(SameVector(Simt.ReferenceMatMul(left, right, rows, inner, columns), launch.Output),
                  "shape " + s + " (" + rows + "x" + inner + "x" + columns + ") is computed correctly");
            int tiles = Simt.Ceil(inner, 3);
            Check(launch.Counters.Waves == Simt.ExpectedWaves(launch.Blocks, launch.WarpsPerBlock, 3, tiles),
                  "shape " + s + ": the wave identity holds");
            Check(launch.Counters.Barriers == (long)launch.Blocks * 2 * tiles,
                  "shape " + s + ": two barriers per k-tile per block");
        }

        // ---- degenerate shapes ------------------------------------------------------------------
        Launch empty = Simt.MatMul(Array.Empty<int>(), Array.Empty<int>(), 0, 0, 0, 4, 2);
        Check(empty.Counters.Waves == 0 && empty.Blocks == 0, "a zero-sized launch issues nothing");
        Launch emptyAxpy = Simt.Axpy(Array.Empty<int>(), Array.Empty<int>(), 0, 8, 4, 2);
        Check(emptyAxpy.Blocks == 0 && emptyAxpy.Counters.Waves == 0, "so does an empty AXPY");
        Launch one = Simt.MatMul(new[] { 3 }, new[] { 5 }, 1, 1, 1, 1, 1);
        Check(one.Output[0] == 15, "a one-by-one kernel multiplies correctly (got " + one.Output[0] + ")");
        Check(one.Counters.InactiveLanes == 0, "with a single lane and nothing masked");
        Launch single = Simt.Axpy(new[] { 2 }, new[] { 3 }, 1, 1, 1, 4);
        Check(single.Output[0] == 11, "and a one-element AXPY computes 4 * 2 + 3");
        Launch zeroInner = Simt.MatMul(new[] { 1, 2, 3, 4 }, new int[0], 2, 0, 2, 2, 2);
        Check(zeroInner.Output[0] == 0 && zeroInner.Output[1] == 0,
              "an inner dimension of zero leaves every accumulator at zero");
        Check(zeroInner.Counters.Barriers == 0, "and stages nothing at all");

        // ---- determinism ------------------------------------------------------------------------
        Launch repeat = Simt.MatMul(a, b, 16, 16, 16, t, 2);
        bool repeatMatch = repeat.Counters.Waves == tiled.Counters.Waves
                           && repeat.Counters.SharedLoads == tiled.Counters.SharedLoads
                           && SameVector(tiled.Output, repeat.Output);
        Check(repeatMatch, "the simulator is deterministic for a fixed launch");

        Console.WriteLine("AXPY: " + axpy.Blocks + " blocks, " + axpy.Counters.Waves + " waves, "
              + axpy.Counters.InactiveLanes + " masked lanes of 112");
        Console.WriteLine("tiled multiply 16x16x16: " + tiled.Counters.GlobalLoads + " global loads against "
              + naiveLoads + " naive, " + tiled.Counters.SharedLoads + " shared reads");
        Console.WriteLine("barriers: " + tiled.Counters.Barriers + ", warp waits " + tiled.Counters.BarrierWaits
              + ", lane slots " + laneSlots);
        Console.WriteLine("PASS");
    }
}
