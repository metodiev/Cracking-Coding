// Task 720 - Implement ring-allreduce simulation and communication cost modeling. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// Ring all-reduce (Baidu/UBER Horovod style) turns "every rank needs the elementwise sum of everyone's
// buffer" into two ring sweeps of p - 1 steps each:
//
//   1. reduce-scatter: at step s rank r sends the chunk it currently holds complete, chunk (r - s) mod p,
//      to rank r + 1 and adds the chunk it receives, chunk (r - s - 1) mod p, into its own buffer. After
//      p - 1 steps rank r holds the *complete* sum of chunk (r + 1) mod p and nothing else.
//   2. all-gather: at step s rank r forwards chunk (r - s + 1) mod p to rank r + 1 and copies chunk
//      (r - s) mod p from rank r - 1, so after p - 1 more steps every rank holds every chunk's sum.
//
// Each of the 2(p - 1) messages carries only n / p elements, so the per-rank traffic is 2n(p - 1) / p
// which is provably bandwidth-optimal, and the total traffic over all ranks is exactly 2n(p - 1) -
// both are asserted against counters the simulation keeps. Chunks are split with the remainder given
// to the lowest chunks, so ragged n works without padding.
//
// Cost model (alpha = per-message latency, beta = per-byte cost): T_ring = 2(p - 1)(alpha + beta n / p),
// compared against a star (gather + broadcast through one server) and recursive doubling (a butterfly
// with log2(p) rounds): the star is p times worse per rank in bandwidth, the butterfly is worse in
// bandwidth for p > 2 but better when latency dominates. Integer payloads make every result exact.
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

/// <summary>Ring all-reduce with chunk accounting: every message that crosses the ring is counted.</summary>
internal sealed class RingAllReduce
{
    private readonly int _p;
    private readonly int[] _chunkStart;   // p + 1 entries; chunk j covers [_chunkStart[j], _chunkStart[j + 1])

    internal RingAllReduce(int ranks, int elements)
    {
        if (ranks < 1) throw new ArgumentOutOfRangeException(nameof(ranks));
        if (elements < 0) throw new ArgumentOutOfRangeException(nameof(elements));
        _p = ranks;
        Elements = elements;
        _chunkStart = new int[ranks + 1];
        int baseSize = elements / ranks;
        int remainder = elements % ranks;
        for (int j = 0; j < ranks; j++) _chunkStart[j + 1] = _chunkStart[j] + baseSize + (j < remainder ? 1 : 0);
        SentByRank = new long[ranks];
        ReceivedByRank = new long[ranks];
    }

    internal int Ranks => _p;
    internal int Elements { get; }
    internal long Messages { get; private set; }
    internal long ElementsTransferred { get; private set; }
    internal long[] SentByRank { get; }
    internal long[] ReceivedByRank { get; }

    internal int ChunkStart(int j) => _chunkStart[j];
    internal int ChunkSize(int j) => _chunkStart[j + 1] - _chunkStart[j];

    internal int[][] ReduceScatter(int[][] rankData)
    {
        int p = _p;
        var buffers = new int[p][];
        for (int r = 0; r < p; r++)
        {
            if (rankData[r].Length != Elements) throw new ArgumentException("rank buffer length mismatch");
            buffers[r] = (int[])rankData[r].Clone();
        }
        for (int step = 0; step < p - 1; step++)
        {
            var messages = new List<(int Target, int Chunk, int[] Values)>(p);
            for (int r = 0; r < p; r++)
            {
                int chunk = Mod(r - step, p);
                messages.Add(((r + 1) % p, chunk, Slice(buffers[r], _chunkStart[chunk], _chunkStart[chunk + 1])));
            }
            foreach ((int target, int chunk, int[] values) in messages)
            {
                int from = _chunkStart[chunk];
                for (int i = 0; i < values.Length; i++) buffers[target][from + i] += values[i];
                Record(r: Mod(target - 1, p), target, values.Length);
            }
        }
        return buffers;
    }

    internal int[][] AllGather(int[][] reduced)
    {
        int p = _p;
        var buffers = new int[p][];
        for (int r = 0; r < p; r++) buffers[r] = (int[])reduced[r].Clone();
        for (int step = 0; step < p - 1; step++)
        {
            var messages = new List<(int Target, int Chunk, int[] Values)>(p);
            for (int r = 0; r < p; r++)
            {
                int chunk = Mod(r - step + 1, p);
                messages.Add(((r + 1) % p, chunk, Slice(buffers[r], _chunkStart[chunk], _chunkStart[chunk + 1])));
            }
            foreach ((int target, int chunk, int[] values) in messages)
            {
                int from = _chunkStart[chunk];
                for (int i = 0; i < values.Length; i++) buffers[target][from + i] = values[i];
                Record(r: Mod(target - 1, p), target, values.Length);
            }
        }
        return buffers;
    }

    internal int[][] Run(int[][] rankData) => AllGather(ReduceScatter(rankData));

    private void Record(int r, int target, int elements)
    {
        Messages++;
        ElementsTransferred += elements;
        SentByRank[r] += elements;
        ReceivedByRank[target] += elements;
    }

    private static int Mod(int value, int p)
    {
        int m = value % p;
        return m < 0 ? m + p : m;
    }

    private static int[] Slice(int[] source, int from, int to)
    {
        var slice = new int[to - from];
        for (int i = from; i < to; i++) slice[i - from] = source[i];
        return slice;
    }

    /// <summary>Ideally balanced ring all-reduce: 2(p - 1) messages of n / p elements per rank.</summary>
    internal static double RingTime(int p, int n, double alpha, double beta) =>
        2.0 * (p - 1) * (alpha + beta * (double)n / p);

    /// <summary>Star all-reduce: gather n elements per worker into a server, reduce, broadcast back.</summary>
    internal static double StarTime(int p, int n, double alpha, double beta) =>
        2.0 * (alpha + beta * n);

    /// <summary>Recursive doubling: log2(p) rounds, each rank sends the whole buffer (only for powers of two).</summary>
    internal static double ButterflyTime(int p, int n, double alpha, double beta) =>
        Log2(p) * (alpha + beta * n);

    internal static double RingVolumePerRank(int p, int n) => 2.0 * n * (p - 1) / p;

    internal static double StarVolumePerRank(int p, int n) => 2.0 * n;

    internal static double ButterflyVolumePerRank(int p, int n) => n * Log2(p);

    private static int Log2(int p)
    {
        int k = 0;
        while (1 << k < p) k++;
        return k;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- p = 1 and p = 2 boundaries -----------------------------------------------------------
        {
            var single = new RingAllReduce(1, 9);
            int[][] input = Bufs(1, 9, 11);
            int[][] result = single.Run(Clone(input));
            Check(single.Messages == 0 && single.ElementsTransferred == 0, "one rank needs no communication at all");
            Check(Same(result, input), "the lone rank keeps its own buffer");

            var pair = new RingAllReduce(2, 8);
            int[][] two = Bufs(2, 8, 5);
            int[][] sums = pair.Run(Clone(two));
            Check(Same(sums, EqualArrays(BruteForceSum(two), 2)), "two ranks end with the exact elementwise sum");
            Check(pair.Messages == 4 && pair.ElementsTransferred == 16,
                "p = 2 exchanges four half-buffers (" + pair.Messages + " messages, " + pair.ElementsTransferred + " elements)");
        }

        // ---- the general case: every rank ends with the sum, and traffic is bandwidth-optimal -------
        foreach ((int p, int n) in new[] { (4, 8), (3, 7), (5, 3), (5, 0), (8, 1000), (6, 61) })
        {
            var ring = new RingAllReduce(p, n);
            int[][] input = Bufs(p, n, 1000 + p);
            int[] expected = BruteForceSum(input);
            int[][] result = ring.Run(Clone(input));
            Check(Same(result, EqualArrays(expected, p)), "p=" + p + " n=" + n + ": every rank holds the exact sum");
            Check(ring.Messages == 2L * (p - 1) * p, "p=" + p + ": exactly 2(p-1)p messages cross the ring");
            Check(ring.ElementsTransferred == 2L * n * (p - 1),
                "p=" + p + " n=" + n + ": total traffic is 2n(p-1) elements (bandwidth-optimal)");

            double expectedVolume = RingAllReduce.RingVolumePerRank(p, n);
            bool volumeMatches = true, volumeBounded = true, formulaMatches = true;
            for (int r = 0; r < p; r++)
            {
                // The ideal 2n(p-1)/p per rank only holds when p divides n; ragged chunks are checked
                // by the exact per-rank formula below instead.
                if (n % p == 0 && Math.Abs(ring.SentByRank[r] + ring.ReceivedByRank[r] - 2.0 * expectedVolume) > 1e-9)
                    volumeMatches = false;
                if (ring.SentByRank[r] > 2L * n) volumeBounded = false;
                // reduce-scatter never sends chunk (r+1) mod p, all-gather never sends chunk (r+2) mod p
                double formula = 2.0 * n - ring.ChunkSize((r + 1) % p) - ring.ChunkSize((r + 2) % p);
                if (Math.Abs(ring.SentByRank[r] - formula) > 1e-9) formulaMatches = false;
            }
            Check(volumeMatches, "p=" + p + " n=" + n + ": an even split gives every rank exactly 2n(p-1)/p elements");
            Check(volumeBounded, "p=" + p + ": no rank moves more than twice the buffer");
            Check(formulaMatches, "p=" + p + ": each rank sends the whole buffer minus the two chunks it keeps");
            Check(ring.ChunkSize(0) >= ring.ChunkSize(p - 1) && SumChunks(ring) == n,
                "p=" + p + " n=" + n + ": chunks partition the buffer with the remainder first");
        }

        // ---- reduce-scatter leaves the complete sum of exactly one chunk per rank ------------------
        {
            var ring = new RingAllReduce(5, 23);
            int[][] input = Bufs(5, 23, 77);
            int[] expected = BruteForceSum(input);
            int[][] scattered = ring.ReduceScatter(Clone(input));
            bool invariant = true;
            for (int r = 0; r < 5; r++)
            {
                int chunk = (r + 1) % 5;                       // rank r owns chunk (r+1) mod p completely
                for (int i = ring.ChunkStart(chunk); i < ring.ChunkStart(chunk + 1); i++)
                    if (scattered[r][i] != expected[i]) invariant = false;
            }
            Check(invariant, "after reduce-scatter rank r holds the complete sum of chunk (r+1) mod p");
            Check(ring.Messages == 20, "reduce-scatter alone sends (p-1)p messages");
            int[][] gathered = ring.AllGather(Clone(scattered));
            Check(Same(gathered, EqualArrays(expected, 5)), "all-gather completes the sums on every rank");
        }

        // ---- cost model: ring vs star vs butterfly --------------------------------------------------
        {
            const double alpha = 1e-6, beta = 1e-9;
            int n = 1 << 20;
            double ring4 = RingAllReduce.RingTime(4, n, alpha, beta);
            double star4 = RingAllReduce.StarTime(4, n, alpha, beta);
            double ring16 = RingAllReduce.RingTime(16, n, alpha, beta);
            double ringSmall = RingAllReduce.RingTime(16, 8, alpha, beta);
            double butterflySmall = RingAllReduce.ButterflyTime(16, 8, alpha, beta);
            Console.WriteLine("ring(4,1MiB)=" + (ring4 * 1e6).ToString("G4") + "us star(4)=" + (star4 * 1e6).ToString("G4")
                + "us ring(16)=" + (ring16 * 1e6).ToString("G4") + "us butterflySmall(16,8B)="
                + (butterflySmall * 1e6).ToString("G4") + "us");

            Check(ring4 < star4, "for p = 4 the ring beats a star for a large buffer");
            Check(Math.Abs(RingAllReduce.RingVolumePerRank(4, n) / RingAllReduce.StarVolumePerRank(4, n) - 0.75) < 1e-12,
                "per-rank ring traffic is (p-1)/p of the star traffic");
            Check(ring16 > ring4, "more ranks cost more ring latency for a fixed buffer");
            Check(ringSmall < RingAllReduce.RingTime(16, 16, alpha, beta), "larger messages cost more than smaller ones");
            Check(butterflySmall < ringSmall, "when latency dominates a log2(p)-round butterfly beats the 2(p-1)-round ring");
            Check(RingAllReduce.RingTime(2, n, alpha, beta) < star4, "even p = 2 beats the star on a large buffer");
            for (int p = 2; p <= 16; p *= 2)
            {
                Check(RingAllReduce.RingVolumePerRank(p, n) < RingAllReduce.StarVolumePerRank(p, n),
                    "p=" + p + ": the ring never moves more per rank than the star");
                if (p > 2)
                    Check(RingAllReduce.RingVolumePerRank(p, n) < RingAllReduce.ButterflyVolumePerRank(p, n),
                        "p=" + p + ": the ring moves less per rank than recursive doubling");
            }
            Check(Math.Abs(RingAllReduce.ButterflyVolumePerRank(4, n) - RingAllReduce.StarVolumePerRank(2, n)) < 1e-9,
                "recursive doubling at p = 4 moves per rank what a two-rank star moves");
        }

        // ---- determinism -----------------------------------------------------------------------------
        {
            var first = new RingAllReduce(7, 29);
            var second = new RingAllReduce(7, 29);
            int[][] input = Bufs(7, 29, 4242);
            int[][] a = first.Run(Clone(input));
            int[][] b = second.Run(Clone(input));
            Check(Same(a, b), "the same input produces the same result twice");
            Check(first.Messages == second.Messages && first.ElementsTransferred == second.ElementsTransferred,
                "the accounting is reproducible as well");
            Check(first.Messages == 2L * 6 * 7 && first.ElementsTransferred == 2L * 29 * 6,
                "p=7 n=29 matches the closed-form traffic");
            bool inputUntouched = true;
            for (int r = 0; r < 7; r++) if (!SameArray(input[r], Bufs(7, 29, 4242)[r])) inputUntouched = false;
            Check(inputUntouched, "the caller's buffers are never mutated in place");
        }

        Console.WriteLine("PASS");
    }

    private static int[][] Bufs(int p, int n, int salt)
    {
        var rng = new SplitMix64((ulong)(0x5EED0000UL + (ulong)salt));
        var bufs = new int[p][];
        for (int r = 0; r < p; r++)
        {
            bufs[r] = new int[n];
            for (int i = 0; i < n; i++) bufs[r][i] = rng.NextInt(1000) - 500;
        }
        return bufs;
    }

    private static int[][] Clone(int[][] data)
    {
        var copy = new int[data.Length][];
        for (int r = 0; r < data.Length; r++) copy[r] = (int[])data[r].Clone();
        return copy;
    }

    private static int[] BruteForceSum(int[][] data)
    {
        int n = data[0].Length;
        var sum = new int[n];
        foreach (int[] buf in data)
            for (int i = 0; i < n; i++) sum[i] += buf[i];
        return sum;
    }

    private static int[][] EqualArrays(int[] values, int count)
    {
        var result = new int[count][];
        for (int r = 0; r < count; r++) result[r] = (int[])values.Clone();
        return result;
    }

    private static bool Same(int[][] a, int[][] b)
    {
        if (a.Length != b.Length) return false;
        for (int r = 0; r < a.Length; r++) if (!SameArray(a[r], b[r])) return false;
        return true;
    }

    private static bool SameArray(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static int SumChunks(RingAllReduce ring)
    {
        int total = 0;
        for (int j = 0; j < ring.Ranks; j++) total += ring.ChunkSize(j);
        return total;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
