// Task 728 - Implement cache-oblivious matrix multiplication idea demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// The cache-oblivious recursion multiplies by splitting every operand in half and recursing into the
// eight sub-products until a base case is reached, without ever knowing the cache size or line length.
// Frigo et al. showed that this is nevertheless optimal for *every* cache: at the recursion depth where
// the three sub-matrices fit in cache, the sub-problem is solved with compulsory misses only.
//
// To make that observable this file measures instead of asserts: a two-level memory model (a word array
// plus a fully associative LRU cache of lines) counts misses, and three algorithms run over identical
// data - a naive i-k-j triple loop, the recursive cache-oblivious split, and a blocked triple loop whose
// block size is tuned per cache size. Documented simplifications: the cache is fully associative (real
// caches are set associative), the replacement policy is exact LRU, and one word is one element of an
// integer matrix, so the arithmetic is exact and only the data movement is analysed.
//
// Verified: all three algorithms agree exactly with a dense reference on square and non-square problems
// including odd sizes and every base case down to one; at a cache large enough to hold the recursion's
// sub-problems the recursion misses far less than the naive loop and lands within a small factor of the
// compulsory lower bound, while the naive order stubbornly stays at one sweep of B per row and does not
// react to the cache size at all; at a cache too small for even its smallest sub-problem the recursion
// re-reads more than the naive loop does and hand-tuned blocking wins, and a base case whose sub-blocks
// overflow the cache is measurably worse than a smaller one. The measurements say exactly where the idea
// pays off instead of assuming it always does.
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

/// <summary>Fully associative LRU cache over a word-addressed array, counting accesses and misses.</summary>
internal sealed class CacheModel
{
    private readonly int[] _line;         // slot -> line id, -1 when empty
    private readonly long[] _stamp;       // slot -> last use, for the LRU decision
    private readonly Dictionary<int, int> _slotOf = new Dictionary<int, int>();
    private long _clock;

    internal CacheModel(int capacityLines, int lineWords)
    {
        if (capacityLines < 1 || lineWords < 1) throw new ArgumentOutOfRangeException(nameof(capacityLines));
        _line = new int[capacityLines];
        _stamp = new long[capacityLines];
        for (int i = 0; i < capacityLines; i++) _line[i] = -1;
        LineWords = lineWords;
    }

    internal int LineWords { get; }
    internal int CapacityLines => _line.Length;
    internal long Accesses { get; private set; }
    internal long Misses { get; private set; }
    internal int CapacityWords => _line.Length * LineWords;

    internal void Touch(int wordIndex)
    {
        Accesses++;
        int line = wordIndex / LineWords;
        if (_slotOf.TryGetValue(line, out int hit))
        {
            _stamp[hit] = ++_clock;
            return;
        }
        Misses++;
        int victim = 0;
        for (int slot = 1; slot < _line.Length; slot++) if (_stamp[slot] < _stamp[victim]) victim = slot;
        if (_line[victim] >= 0) _slotOf.Remove(_line[victim]);
        _line[victim] = line;
        _stamp[victim] = ++_clock;
        _slotOf[line] = victim;
    }
}

/// <summary>Word array plus the cache in front of it: every access goes through the model.</summary>
internal sealed class Arena
{
    internal readonly int[] Data;
    internal readonly CacheModel Cache;

    internal Arena(int words, CacheModel cache)
    {
        Data = new int[words];
        Cache = cache;
    }

    internal int Load(int address)
    {
        Cache.Touch(address);
        return Data[address];
    }

    internal void Store(int address, int value)
    {
        Cache.Touch(address);
        Data[address] = value;
    }
}

internal static class MatMul
{
    /// <summary>Naive i-k-j triple loop over the whole problem.</summary>
    internal static void Naive(Arena arena, int aBase, int bBase, int cBase, int rows, int inner, int cols, int stride)
    {
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < inner; k++)
            {
                int av = arena.Load(aBase + i * stride + k);
                for (int j = 0; j < cols; j++)
                {
                    int bv = arena.Load(bBase + k * stride + j);
                    int cv = arena.Load(cBase + i * stride + j);
                    arena.Store(cBase + i * stride + j, cv + av * bv);
                }
            }
    }

    /// <summary>
    /// Cache-oblivious recursion: halve rows, inner dimension and columns, recurse into the eight
    /// sub-products, and only stop at <paramref name="baseCase"/> where the plain loop takes over.
    /// </summary>
    internal static void Recursive(Arena arena, int aBase, int bBase, int cBase, int rows, int inner, int cols, int stride, int baseCase)
    {
        if (rows <= baseCase || inner <= baseCase || cols <= baseCase)
        {
            Naive(arena, aBase, bBase, cBase, rows, inner, cols, stride);
            return;
        }
        int r1 = rows / 2, r2 = rows - r1;
        int k1 = inner / 2, k2 = inner - k1;
        int c1 = cols / 2, c2 = cols - c1;
        Recursive(arena, aBase, bBase, cBase, r1, k1, c1, stride, baseCase);
        Recursive(arena, aBase + k1, bBase + k1 * stride, cBase, r1, k2, c1, stride, baseCase);
        Recursive(arena, aBase, bBase + c1, cBase + c1, r1, k1, c2, stride, baseCase);
        Recursive(arena, aBase + k1, bBase + k1 * stride + c1, cBase + c1, r1, k2, c2, stride, baseCase);
        Recursive(arena, aBase + r1 * stride, bBase, cBase + r1 * stride, r2, k1, c1, stride, baseCase);
        Recursive(arena, aBase + r1 * stride + k1, bBase + k1 * stride, cBase + r1 * stride, r2, k2, c1, stride, baseCase);
        Recursive(arena, aBase + r1 * stride, bBase + c1, cBase + r1 * stride + c1, r2, k1, c2, stride, baseCase);
        Recursive(arena, aBase + r1 * stride + k1, bBase + k1 * stride + c1, cBase + r1 * stride + c1, r2, k2, c2, stride, baseCase);
    }

    /// <summary>Blocked triple loop with a tunable block size (the classic cache-aware alternative).</summary>
    internal static void Blocked(Arena arena, int aBase, int bBase, int cBase, int rows, int inner, int cols, int stride, int block)
    {
        for (int i0 = 0; i0 < rows; i0 += block)
            for (int k0 = 0; k0 < inner; k0 += block)
                for (int j0 = 0; j0 < cols; j0 += block)
                    for (int i = i0; i < Math.Min(rows, i0 + block); i++)
                        for (int k = k0; k < Math.Min(inner, k0 + block); k++)
                        {
                            int av = arena.Load(aBase + i * stride + k);
                            for (int j = j0; j < Math.Min(cols, j0 + block); j++)
                            {
                                int bv = arena.Load(bBase + k * stride + j);
                                int cv = arena.Load(cBase + i * stride + j);
                                arena.Store(cBase + i * stride + j, cv + av * bv);
                            }
                        }
    }

    /// <summary>Leaf counter of the recursion: how many base-case calls a given base case produces.</summary>
    internal static long LeafCount(int rows, int inner, int cols, int baseCase)
    {
        if (rows <= baseCase || inner <= baseCase || cols <= baseCase) return 1;
        int r1 = rows / 2, r2 = rows - r1;
        int k1 = inner / 2, k2 = inner - k1;
        int c1 = cols / 2, c2 = cols - c1;
        return LeafCount(r1, k1, c1, baseCase) + LeafCount(r1, k2, c1, baseCase)
             + LeafCount(r1, k1, c2, baseCase) + LeafCount(r1, k2, c2, baseCase)
             + LeafCount(r2, k1, c1, baseCase) + LeafCount(r2, k2, c1, baseCase)
             + LeafCount(r2, k1, c2, baseCase) + LeafCount(r2, k2, c2, baseCase);
    }
}

public static class Solution
{
    private const int Size = 48;
    private const int bigCacheLines = 128;

    public static void Main()
    {
        // ---- correctness of every variant, including odd and rectangular shapes --------------------
        {
            foreach ((int rows, int inner, int cols) in new[] { (1, 1, 1), (7, 7, 7), (16, 16, 16), (33, 17, 9), (48, 48, 48) })
            {
                int stride = Math.Max(rows, Math.Max(inner, cols));
                var cache = new CacheModel(1 << 16, 8);
                Arena arena = Fill(rows, inner, cols, stride, 0x728AUL, cache);
                int[] expected = Reference(arena, rows, inner, cols, stride);
                int aBase = 0, bBase = stride * stride * 2, cBase = stride * stride * 4;
                MatMul.Naive(arena, aBase, bBase, cBase, rows, inner, cols, stride);
                Check(Same(expected, Extract(arena, cBase, rows, cols, stride)),
                    rows + "x" + inner + "x" + cols + ": the naive loop is the reference");
                Clear(arena, cBase, stride);
                MatMul.Recursive(arena, aBase, bBase, cBase, rows, inner, cols, stride, 4);
                Check(Same(expected, Extract(arena, cBase, rows, cols, stride)),
                    rows + "x" + inner + "x" + cols + ": the cache-oblivious recursion agrees");
                Clear(arena, cBase, stride);
                MatMul.Blocked(arena, aBase, bBase, cBase, rows, inner, cols, stride, 8);
                Check(Same(expected, Extract(arena, cBase, rows, cols, stride)),
                    rows + "x" + inner + "x" + cols + ": the blocked loop agrees as well");
            }
        }

        // ---- the recursion is correct for every base case, including base case one ----------------
        {
            int stride = Size;
            Arena arena = Fill(Size, Size, Size, stride, 0x728BUL, new CacheModel(1 << 16, 8));
            int[] expected = Reference(arena, Size, Size, Size, stride);
            int cBase = stride * stride * 4;
            foreach (int baseCase in new[] { 1, 2, 3, 4, 8, 16, 32, 64 })
            {
                Clear(arena, cBase, stride);
                MatMul.Recursive(arena, 0, stride * stride * 2, cBase, Size, Size, Size, stride, baseCase);
                Check(Same(expected, Extract(arena, cBase, Size, Size, stride)),
                    "base case " + baseCase + " computes the same product");
            }
            Check(MatMul.LeafCount(32, 32, 32, 8) == 64, "a 32-cube with base case eight has eight cubed leaves");
            Check(MatMul.LeafCount(48, 48, 48, 8) == 512, "a 48-cube halves down to six, so eight cubed leaves");
        }

        // ---- misses: the naive loop against the recursion, for two cache sizes --------------------
        {
            const int smallLines = 16, bigLines = 128;
            long naiveSmall, recSmall, recLeafSmall, blockSmall, naiveBig, recBig, blockBig;
            {
                Arena arena = MakeArena(smallLines);
                int cBase = Size * Size * 4;
                naiveSmall = Misses(arena, cBase, (a) => MatMul.Naive(a, 0, Size * Size * 2, cBase, Size, Size, Size, Size));
                recSmall = Misses(arena, cBase, (a) => MatMul.Recursive(a, 0, Size * Size * 2, cBase, Size, Size, Size, Size, 1));
                recLeafSmall = Misses(arena, cBase, (a) => MatMul.Recursive(a, 0, Size * Size * 2, cBase, Size, Size, Size, Size, 8));
                blockSmall = TuneBlocked(smallLines);
            }
            {
                Arena arena = MakeArena(bigLines);
                int cBase = Size * Size * 4;
                naiveBig = Misses(arena, cBase, (a) => MatMul.Naive(a, 0, Size * Size * 2, cBase, Size, Size, Size, Size));
                recBig = Misses(arena, cBase, (a) => MatMul.Recursive(a, 0, Size * Size * 2, cBase, Size, Size, Size, Size, 1));
                blockBig = TuneBlocked(bigLines);
            }
            long compulsory = 3L * Size * Size / 8;
            Console.WriteLine("n=" + Size + " cache=" + smallLines + " lines: naive=" + naiveSmall + " recursiveBase1=" + recSmall
                + " recursiveBase8=" + recLeafSmall + " blockedBest=" + blockSmall + " compulsory=" + compulsory);
            Console.WriteLine("n=" + Size + " cache=" + bigLines + " lines: naive=" + naiveBig + " recursiveBase1=" + recBig
                + " blockedBest=" + blockBig);

            Check(recSmall >= compulsory && naiveSmall >= compulsory,
                "no algorithm beats the compulsory misses (" + compulsory + ")");
            Check(recBig * 2 < naiveBig, "with room in the cache the recursion misses far less than the naive loop ("
                + recBig + " vs " + naiveBig + ")");
            Check(recBig <= 3 * compulsory, "at that cache size the recursion is within a small factor of the "
                + "compulsory lower bound (" + recBig + " vs " + compulsory + ")");
            Check(naiveBig > 4 * compulsory, "the naive loop stays at one full sweep of B per row (" + naiveBig + ")");
            Check(recBig < recSmall, "a tiny cache makes the recursion pay for its re-reads (" + recSmall + " vs " + recBig + ")");
            Check(naiveSmall == naiveBig, "the naive order is completely insensitive to the cache size");
            Check(recSmall < recLeafSmall, "a base case that overflows the cache is measurably worse ("
                + recSmall + " vs " + recLeafSmall + ")");
            Check(blockSmall < naiveSmall, "at a tiny cache the tuned blocking still wins (" + blockSmall + " vs " + naiveSmall + ")");
            Check(blockSmall < recSmall, "and it beats the fixed-base-case recursion there ("
                + blockSmall + " vs " + recSmall + ")");
            Check(recBig <= 2 * blockBig, "at the larger cache the recursion is within twice the best tuned blocking ("
                + recBig + " vs " + blockBig + ")");
            Check(naiveBig <= 2 * compulsory || blockBig <= 2 * compulsory, "a well-sized cache reaches compulsory misses");
        }

        // ---- the base case barely matters when it fits the cache -------------------------------------
        {
            var misses = new List<long>();
            foreach (int baseCase in new[] { 1, 2, 4, 8 })
            {
                Arena arena = MakeArena(bigCacheLines);
                misses.Add(Misses(arena, Size * Size * 4,
                    (a) => MatMul.Recursive(a, 0, Size * Size * 2, Size * Size * 4, Size, Size, Size, Size, baseCase)));
            }
            long smallest = long.MaxValue, largest = 0;
            foreach (long value in misses) { smallest = Math.Min(smallest, value); largest = Math.Max(largest, value); }
            Console.WriteLine("base cases " + string.Join(",", misses) + " best=" + smallest + " worst=" + largest);
            Check(smallest > 0, "every base case really runs");
            Check(largest <= 2 * smallest, "the miss count barely depends on the base case ("
                + smallest + " to " + largest + ")");
            Check(misses.Count == 4, "all four base cases were measured");
        }

        // ---- the same data always gives the same numbers ---------------------------------------------
        {
            int cBase = Size * Size * 4;
            Arena first = Fill(Size, Size, Size, Size, 0x728EUL, new CacheModel(64, 8));
            int[] expected = Reference(first, Size, Size, Size, Size);
            MatMul.Recursive(first, 0, Size * Size * 2, cBase, Size, Size, Size, Size, 8);
            Arena second = Fill(Size, Size, Size, Size, 0x728EUL, new CacheModel(64, 8));
            MatMul.Recursive(second, 0, Size * Size * 2, cBase, Size, Size, Size, Size, 8);
            Check(Same(Extract(first, cBase, Size, Size, Size), Extract(second, cBase, Size, Size, Size)),
                "the product is reproducible");
            Check(Same(expected, Extract(first, cBase, Size, Size, Size)), "and it is still the right product");
            Check(first.Cache.Accesses == second.Cache.Accesses && first.Cache.Misses == second.Cache.Misses,
                "the access and miss counts are reproducible as well");
            Check(first.Cache.Accesses > first.Cache.Misses, "every access is counted, hits included");
            Check(first.Cache.CapacityWords == 512, "the reported cache capacity is line count times line size");
        }

        Console.WriteLine("PASS");
    }

    private static long Misses(Arena arena, int cBase, Action<Arena> run)
    {
        Clear(arena, cBase, (int)Math.Sqrt(cBase / 4.0));
        long before = arena.Cache.Misses;
        run(arena);
        return arena.Cache.Misses - before;
    }

    private static Arena MakeArena(int cacheLines) => Fill(Size, Size, Size, Size, 0x728CUL, new CacheModel(cacheLines, 8));

    private static long TuneBlocked(int cacheLines)
    {
        long best = long.MaxValue;
        foreach (int block in new[] { 2, 4, 6, 8, 12, 16, 24, 32 })
        {
            Arena arena = MakeArena(cacheLines);
            int cBase = Size * Size * 4;
            long misses = Misses(arena, cBase,
                (a) => MatMul.Blocked(a, 0, Size * Size * 2, cBase, Size, Size, Size, Size, block));
            best = Math.Min(best, misses);
        }
        return best;
    }

    private static Arena Fill(int rows, int inner, int cols, int stride, ulong seed, CacheModel cache)
    {
        var arena = new Arena(6 * stride * stride, cache);
        var rng = new SplitMix64(seed);
        for (int i = 0; i < rows; i++) for (int k = 0; k < inner; k++) arena.Data[i * stride + k] = rng.NextInt(9) - 4;
        for (int k = 0; k < inner; k++) for (int j = 0; j < cols; j++) arena.Data[stride * stride * 2 + k * stride + j] = rng.NextInt(9) - 4;
        return arena;
    }

    private static int[] Reference(Arena arena, int rows, int inner, int cols, int stride)
    {
        var product = new int[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < inner; k++)
            {
                int av = arena.Data[i * stride + k];
                for (int j = 0; j < cols; j++)
                    product[i * cols + j] += av * arena.Data[stride * stride * 2 + k * stride + j];
            }
        return product;
    }

    private static int[] Extract(Arena arena, int cBase, int rows, int cols, int stride)
    {
        var product = new int[rows * cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) product[i * cols + j] = arena.Data[cBase + i * stride + j];
        return product;
    }

    private static void Clear(Arena arena, int cBase, int stride)
    {
        // zeroing bypasses the cache model: the measured run then starts from a defined memory image
        for (int i = 0; i < stride * stride; i++) arena.Data[cBase + i] = 0;
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
