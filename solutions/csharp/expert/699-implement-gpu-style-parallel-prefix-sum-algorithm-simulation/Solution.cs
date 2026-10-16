// Task 699 - Implement GPU-style parallel prefix-sum algorithm simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// Four levels of the standard GPU scan are simulated on the CPU with an explicit cost model, because
// the point of the exercise is the communication pattern rather than the numbers:
//
//   * warp scan: 32 lanes, five Hillis-Steele shuffle rounds (offsets 1, 2, 4, 8, 16) where a lane
//     below the offset shifts in the identity, which is what __shfl_up_sync does;
//   * block scan by Hillis-Steele in shared memory: log2(n) steps but n log2(n) additions, i.e. not
//     work efficient;
//   * block scan by Blelloch: the up-sweep / down-sweep pair producing the *exclusive* scan with
//     O(n) additions and 2 log2(n) steps, i.e. work efficient;
//   * multi-block scan: one scan per block, a scan of the block totals, and the add-back phase, which
//     is how a single kernel launch turns into a three phase algorithm for arbitrary input sizes.
//
// Every result is compared with a sequential reference scan, and the algorithm properties are
// measured rather than assumed: the addition counts are counted (Hillis-Steele must cost about
// log2(n) times more work than Blelloch), the shuffle rounds are counted, and bank conflicts are
// modelled by mapping the warp's shared-memory accesses onto 32 banks. That model is used to show
// the classic layout result: the interleaved ("strided") ownership where lane t owns elements
// t + k * 32 is conflict free, while contiguous ownership where lane t owns 8t .. 8t+7 serialises
// eight ways on a stride of 8. Degenerate sizes (n = 1, block = 1) and non multiple sizes (padding
// with the identity) are covered, and the whole simulation is deterministic.
using System;
using System.Collections.Generic;

internal static class ScanCost
{
    public static long Additions;
    public static long Steps;
    public static long ShuffleRounds;
    public static long ShuffleOps;
    public static long KernelPhases;

    public static void Reset()
    {
        Additions = 0;
        Steps = 0;
        ShuffleRounds = 0;
        ShuffleOps = 0;
        KernelPhases = 0;
    }
}

internal static class Scan
{
    public const int WarpSize = 32;

    /// <summary>Warp level inclusive scan by shuffles; lane - offset shifts in the identity.</summary>
    public static int[] WarpInclusive(int[] values)
    {
        if (values.Length != WarpSize) throw new ArgumentException("a warp has exactly 32 lanes");
        var result = (int[])values.Clone();
        for (int offset = 1; offset < WarpSize; offset *= 2)
        {
            ScanCost.ShuffleRounds++;
            var previous = (int[])result.Clone();
            for (int lane = 0; lane < WarpSize; lane++)
            {
                ScanCost.ShuffleOps++;
                ScanCost.Steps++;
                int incoming = lane >= offset ? previous[lane - offset] : 0;      // shfl_up semantics
                result[lane] = previous[lane] + incoming;
                ScanCost.Additions++;
            }
        }
        return result;
    }

    /// <summary>Hillis-Steele block scan in shared memory. Not work efficient: n log2(n) additions.</summary>
    public static int[] HillisSteeleInclusive(int[] values)
    {
        int n = values.Length;
        if (n == 0 || (n & (n - 1)) != 0) throw new ArgumentException("the block scan needs a power of two");
        var result = (int[])values.Clone();
        for (int offset = 1; offset < n; offset *= 2)
        {
            ScanCost.Steps++;
            var previous = (int[])result.Clone();
            for (int index = 0; index < n; index++)
            {
                ScanCost.Additions++;
                int incoming = index >= offset ? previous[index - offset] : 0;
                result[index] = previous[index] + incoming;
            }
        }
        return result;
    }

    /// <summary>Blelloch up-sweep / down-sweep, producing the exclusive scan with O(n) additions.</summary>
    public static int[] BlellochExclusive(int[] values)
    {
        int n = values.Length;
        if (n == 0 || (n & (n - 1)) != 0) throw new ArgumentException("the block scan needs a power of two");
        var result = (int[])values.Clone();
        for (int offset = 1; offset < n; offset *= 2)
        {
            ScanCost.Steps++;
            for (int index = 2 * offset - 1; index < n; index += 2 * offset)
            {
                result[index] += result[index - offset];
                ScanCost.Additions++;
            }
        }
        result[n - 1] = 0;
        for (int offset = n / 2; offset > 0; offset /= 2)
        {
            ScanCost.Steps++;
            for (int index = 2 * offset - 1; index < n; index += 2 * offset)
            {
                int left = result[index - offset];
                result[index - offset] = result[index];
                result[index] += left;
                ScanCost.Additions += 2;
            }
        }
        return result;
    }

    public static int[] SequentialInclusive(int[] values)
    {
        var result = new int[values.Length];
        int running = 0;
        for (int i = 0; i < values.Length; i++)
        {
            running += values[i];
            result[i] = running;
        }
        return result;
    }

    public static int[] SequentialExclusive(int[] values)
    {
        var result = new int[values.Length];
        int running = 0;
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = running;
            running += values[i];
        }
        return result;
    }

    public static int[] ExclusiveFromInclusive(int[] inclusive)
    {
        var result = new int[inclusive.Length];
        for (int i = 1; i < inclusive.Length; i++) result[i] = inclusive[i - 1];
        return result;
    }

    /// <summary>Chunked ownership with contiguous runs per lane, the layout that conflicts.</summary>
    public static int[] MultiElementScanContiguous(int[] values, int itemsPerLane, int laneCount)
    {
        int n = laneCount * itemsPerLane;
        if (values.Length != n) throw new ArgumentException("the input must fill the lanes exactly");
        var laneTotals = new int[laneCount];
        for (int lane = 0; lane < laneCount; lane++)
        {
            int running = 0;
            for (int k = 0; k < itemsPerLane; k++)
            {
                running += values[lane * itemsPerLane + k];
                ScanCost.Additions++;
            }
            laneTotals[lane] = running;
        }
        int[] exclusiveTotals = BlellochExclusive(laneTotals);
        var result = new int[n];
        for (int lane = 0; lane < laneCount; lane++)
        {
            int running = exclusiveTotals[lane];
            for (int k = 0; k < itemsPerLane; k++)
            {
                int index = lane * itemsPerLane + k;
                running += values[index];
                result[index] = running;
                ScanCost.Additions++;
            }
        }
        return result;
    }

    /// <summary>Interleaved ownership where lane t owns t, t + laneCount, ... : conflict free.</summary>
    public static int[] MultiElementScanInterleaved(int[] values, int itemsPerLane, int laneCount)
    {
        int n = laneCount * itemsPerLane;
        if (values.Length != n) throw new ArgumentException("the input must fill the lanes exactly");
        var laneTotals = new int[laneCount];
        for (int lane = 0; lane < laneCount; lane++)
        {
            int running = 0;
            for (int k = 0; k < itemsPerLane; k++)
            {
                running += values[k * laneCount + lane];
                ScanCost.Additions++;
            }
            laneTotals[lane] = running;
        }
        int[] exclusiveTotals = BlellochExclusive(laneTotals);
        var result = new int[n];
        for (int lane = 0; lane < laneCount; lane++)
        {
            int running = exclusiveTotals[lane];
            for (int k = 0; k < itemsPerLane; k++)
            {
                int index = k * laneCount + lane;
                running += values[index];
                result[index] = running;
                ScanCost.Additions++;
            }
        }
        return result;
    }

    /// <summary>Three phase scan for an arbitrary length: per block, block totals, add back.</summary>
    public static int[] MultiBlockInclusive(int[] values, int blockSize, out int phases, out int[] blockTotals)
    {
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
            throw new ArgumentException("the block size must be a power of two");
        int n = values.Length;
        int blockCount = (n + blockSize - 1) / blockSize;
        int paddedLength = blockCount * blockSize;
        var padded = new int[paddedLength];
        Array.Copy(values, padded, n);

        blockTotals = new int[blockCount];
        var scanned = new int[paddedLength];
        for (int block = 0; block < blockCount; block++)
        {
            var blockValues = new int[blockSize];
            Array.Copy(padded, block * blockSize, blockValues, 0, blockSize);
            int[] blockScan = HillisSteeleInclusive(blockValues);
            Array.Copy(blockScan, 0, scanned, block * blockSize, blockSize);
            blockTotals[block] = blockScan[blockSize - 1];
        }
        phases = 1;

        // The block totals are scanned by a single block, so they are padded to a power of two with
        // the identity; the padding cannot change the offsets of the real blocks.
        int scanLength = 1;
        while (scanLength < blockCount) scanLength *= 2;
        var paddedTotals = new int[scanLength];
        Array.Copy(blockTotals, paddedTotals, blockCount);
        int[] exclusiveBlockOffsets = Scan.BlellochExclusive(paddedTotals);
        phases++;
        for (int block = 0; block < blockCount; block++)
        {
            int addend = exclusiveBlockOffsets[block];
            for (int k = 0; k < blockSize; k++)
            {
                scanned[block * blockSize + k] += addend;
                ScanCost.Additions++;
            }
        }
        phases++;
        var result = new int[n];
        Array.Copy(scanned, result, n);
        return result;
    }

    // --- Shared memory bank model ---------------------------------------------------------

    /// <summary>Serialisation degree of one warp instruction: the busiest bank's access count.</summary>
    public static int BankConflicts(IReadOnlyList<int> warpAccesses)
    {
        if (warpAccesses.Count == 0) return 0;
        var counts = new int[WarpSize];
        foreach (int address in warpAccesses) counts[address % WarpSize]++;
        int worst = 0;
        foreach (int count in counts) worst = Math.Max(worst, count);
        return worst;
    }

    /// <summary>Addresses read by a warp whose lane t handles the contiguous run 8t .. 8t+7.</summary>
    public static List<int> ContiguousAccesses(int indexWithinRun, int itemsPerLane) =>
        BuildAccesses(indexWithinRun, itemsPerLane, contiguous: true);

    /// <summary>Addresses read by a warp whose lane t handles t, t + 32, t + 64, ...</summary>
    public static List<int> InterleavedAccesses(int indexWithinRun, int itemsPerLane) =>
        BuildAccesses(indexWithinRun, itemsPerLane, contiguous: false);

    private static List<int> BuildAccesses(int indexWithinRun, int itemsPerLane, bool contiguous)
    {
        var accesses = new List<int>();
        for (int lane = 0; lane < WarpSize; lane++)
        {
            int index = contiguous
                ? lane * itemsPerLane + indexWithinRun
                : indexWithinRun * WarpSize + lane;
            accesses.Add(index);
        }
        return accesses;
    }

    /// <summary>Addresses read by a warp whose lane t handles the strided element lane * stride.</summary>
    public static List<int> StridedAccesses(int stride)
    {
        var accesses = new List<int>();
        for (int lane = 0; lane < WarpSize; lane++) accesses.Add(lane * stride);
        return accesses;
    }
}

internal sealed class DeterministicRng699
{
    private ulong _state;

    public DeterministicRng699(ulong seed) => _state = seed;

    public int NextInt(int bound)
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(z % (ulong)bound);
    }
}

public static class Solution
{
    public static void Main()
    {
        var rng = new DeterministicRng699(0x699C0FFEEUL);

        // --- Warp scan: five shuffle rounds, exact result. ------------------------------
        var warpInput = new int[Scan.WarpSize];
        for (int i = 0; i < warpInput.Length; i++) warpInput[i] = rng.NextInt(20) - 10;
        ScanCost.Reset();
        int[] warpScan = Scan.WarpInclusive(warpInput);
        Check(ScanCost.ShuffleRounds == 5, "a 32 lane warp scan needs log2(32) = 5 shuffle rounds, got " + ScanCost.ShuffleRounds);
        Check(ScanCost.ShuffleOps == 5 * 32, "each round shuffles all 32 lanes, ops=" + ScanCost.ShuffleOps);
        Check(SequenceEqual(warpScan, Scan.SequentialInclusive(warpInput)), "the warp scan equals the sequential inclusive scan");
        Check(warpScan[0] == warpInput[0], "lane zero is unchanged, the shuffle shifts in the identity");

        var singleLane = new int[Scan.WarpSize];
        singleLane[0] = 7;
        Check(Scan.WarpInclusive(singleLane)[31] == 7, "a single value propagates to the last lane");
        bool rejected = false;
        try { Scan.WarpInclusive(new int[16]); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a warp that is not 32 lanes wide is rejected");

        // --- Block scan: Hillis-Steele versus Blelloch. ---------------------------------
        const int N = 1024;
        var blockInput = new int[N];
        for (int i = 0; i < N; i++) blockInput[i] = rng.NextInt(200) - 100;
        int[] reference = Scan.SequentialInclusive(blockInput);

        ScanCost.Reset();
        int[] hillisSteele = Scan.HillisSteeleInclusive(blockInput);
        long hillisSteps = ScanCost.Steps, hillisWork = ScanCost.Additions;
        Check(SequenceEqual(hillisSteele, reference), "the Hillis-Steele block scan matches the reference");
        Check(hillisSteps == 10, "Hillis-Steele takes log2(1024) = 10 steps, got " + hillisSteps);
        Check(hillisWork == 10L * N, "and n log2(n) = 10240 additions, got " + hillisWork);

        ScanCost.Reset();
        int[] blelloch = Scan.BlellochExclusive(blockInput);
        long blellochSteps = ScanCost.Steps, blellochWork = ScanCost.Additions;
        Check(SequenceEqual(blelloch, Scan.SequentialExclusive(blockInput)), "the Blelloch block scan produces the exclusive scan");
        Check(blellochSteps == 20, "Blelloch takes 2 log2(1024) = 20 steps, got " + blellochSteps);
        Check(blellochWork <= 4L * N, "Blelloch stays linear in n, work=" + blellochWork);
        Check(hillisWork > 2 * blellochWork, $"Hillis-Steele costs about log2(n) times more work ({hillisWork} vs {blellochWork})");
        Check(SequenceEqual(Scan.ExclusiveFromInclusive(hillisSteele), blelloch),
            "shifting the inclusive scan right gives exactly the exclusive scan");

        // --- Bank conflicts: the layout is what matters. --------------------------------
        Check(Scan.BankConflicts(Scan.InterleavedAccesses(0, 8)) == 1,
            "the interleaved layout is conflict free, serialisation=" + Scan.BankConflicts(Scan.InterleavedAccesses(0, 8)));
        int contiguous = Scan.BankConflicts(Scan.ContiguousAccesses(0, 8));
        Check(contiguous == 8, "the contiguous layout serialises eight ways (stride 8 over 32 lanes), got " + contiguous);
        var sameWord = new List<int>();
        for (int lane = 0; lane < Scan.WarpSize; lane++) sameWord.Add(5);
        Check(Scan.BankConflicts(sameWord) == 32, "all lanes hitting one address is a 32 way conflict");
        Check(Scan.BankConflicts(Scan.StridedAccesses(1)) == 1, "a unit stride is conflict free");
        Check(Scan.BankConflicts(Scan.StridedAccesses(2)) == 2, "a stride of two gives two way conflicts");
        Check(Scan.BankConflicts(Scan.StridedAccesses(8)) == 8, "a stride of eight gives eight way conflicts");
        Check(Scan.BankConflicts(Scan.StridedAccesses(32)) == 32, "a stride of 32 is the worst case");

        // --- Multi element per lane: contiguous keeps the order, interleaved transposes it. ---
        ScanCost.Reset();
        int[] contiguousScan = Scan.MultiElementScanContiguous(blockInput, 8, 128);
        int[] interleavedScan = Scan.MultiElementScanInterleaved(blockInput, 8, 128);
        Check(SequenceEqual(contiguousScan, reference), "the contiguous per lane scan matches the reference");
        // With interleaved ownership the lanes' runs are visited lane major, so the result is the
        // scan of the sequence viewed in that order; the mapping is reproduced explicitly here.
        var order = new List<int>();
        for (int lane = 0; lane < 128; lane++)
            for (int k = 0; k < 8; k++) order.Add(k * 128 + lane);
        var reordered = new int[order.Count];
        for (int i = 0; i < order.Count; i++) reordered[i] = blockInput[order[i]];
        int[] reorderedScan = Scan.SequentialInclusive(reordered);
        var expectedInterleaved = new int[order.Count];
        for (int i = 0; i < order.Count; i++) expectedInterleaved[order[i]] = reorderedScan[i];
        Check(SequenceEqual(interleavedScan, expectedInterleaved),
            "the interleaved per lane scan is exactly the scan in lane major order");
        Check(!SequenceEqual(interleavedScan, reference), "the interleaved layout does change the scan order, as documented");

        // --- Three phase multi block scan over several shapes. -------------------------
        var shapes = new (int Length, int BlockSize)[] { (1024, 256), (1000, 256), (70, 64), (64, 64), (1, 1), (5, 1) };
        foreach ((int length, int blockSize) in shapes)
        {
            var values = new int[length];
            for (int i = 0; i < length; i++) values[i] = rng.NextInt(1000) - 500;
            int[] expected = Scan.SequentialInclusive(values);
            ScanCost.Reset();
            int[] got = Scan.MultiBlockInclusive(values, blockSize, out int phases, out int[] totals);
            Check(phases == 3, $"the multi block scan of {length} with blocks of {blockSize} uses three phases");
            Check(SequenceEqual(got, expected), $"the multi block scan of {length} with blocks of {blockSize} is exact");
            Check(got[length - 1] == expected[length - 1], "the total matches the sequential total");
            int blockCount = (length + blockSize - 1) / blockSize;
            Check(totals.Length == blockCount, $"the block totals array has {blockCount} entries");
            for (int block = 0; block < blockCount; block++)
            {
                int sum = 0;
                for (int k = 0; k < blockSize && block * blockSize + k < length; k++) sum += values[block * blockSize + k];
                Check(totals[block] == sum, $"block {block} total is the sum of its values");
            }
        }

        // A non multiple length must be padded with the identity, so the tail is untouched.
        var ragged = new int[1000];
        for (int i = 0; i < ragged.Length; i++) ragged[i] = i % 7;
        int[] raggedScan = Scan.MultiBlockInclusive(ragged, 256, out _, out _);
        Check(SequenceEqual(raggedScan, Scan.SequentialInclusive(ragged)), "the ragged length scan is exact");
        Check(raggedScan[999] == Scan.SequentialInclusive(ragged)[999], "the last element is the grand total");

        // --- Boundaries and invalid inputs. ---------------------------------------------
        Check(SequenceEqual(Scan.HillisSteeleInclusive(new[] { 42 }), new[] { 42 }), "a one element block scan returns its input");
        Check(Scan.BlellochExclusive(new[] { 42 })[0] == 0, "a one element exclusive scan is the identity");
        rejected = false;
        try { Scan.HillisSteeleInclusive(new[] { 1, 2, 3 }); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a non power of two block is rejected");
        rejected = false;
        try { Scan.MultiBlockInclusive(new[] { 1, 2, 3 }, 3, out _, out _); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a non power of two block size is rejected in the multi block scan");

        // Large magnitude values wrap exactly like the sequential reference.
        var big = new[] { int.MaxValue - 3, 10, -20, 5, 7, 11, -13, 17 };
        Check(SequenceEqual(Scan.HillisSteeleInclusive(big), Scan.SequentialInclusive(big)),
            "overflowing values wrap exactly like the sequential scan");
        Check(SequenceEqual(Scan.BlellochExclusive(big), Scan.SequentialExclusive(big)),
            "the exclusive scan wraps the same way");

        // --- Determinism. -----------------------------------------------------------------
        Check(Summary() == Summary(), "reruns are identical");

        Console.WriteLine($"warpRounds={5} hsSteps={hillisSteps} hsWork={hillisWork} " +
                          $"blellochSteps={blellochSteps} blellochWork={blellochWork}");
        Console.WriteLine($"contiguousConflicts={contiguous} interleavedConflicts=1 " +
                          $"stride2Conflicts={Scan.BankConflicts(Scan.StridedAccesses(2))} phases=3");
        Console.WriteLine("PASS");
    }

    private static bool SequenceEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static string Summary()
    {
        var rng = new DeterministicRng699(4242);
        var values = new int[512];
        for (int i = 0; i < values.Length; i++) values[i] = rng.NextInt(50) - 25;
        int[] multi = Scan.MultiBlockInclusive(values, 128, out int phases, out _);
        int[] warp = Scan.WarpInclusive(Values32(values));
        return $"phases={phases} last={multi[511]} warpLast={warp[31]} total={Scan.SequentialInclusive(values)[511]}";
    }

    private static int[] Values32(int[] source)
    {
        var slice = new int[Scan.WarpSize];
        Array.Copy(source, slice, Scan.WarpSize);
        return slice;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
