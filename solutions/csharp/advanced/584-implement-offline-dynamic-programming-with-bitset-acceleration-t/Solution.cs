// Task 584 - Implement offline dynamic programming with bitset acceleration tricks. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
//
// Bitset-accelerated dynamic programming, plus an offline query layer. The workhorse is a plain
// `ulong[]` bitset of fixed capacity that supports the two operations every bitset DP needs:
// `OrShiftInto(dest, src, shift)` implements `dest |= src << shift` word by word (word order
// descending so that an aliased call `OrShiftInto(dp, dp, w)` is safe, i.e. the classic
// reachability update `reach |= reach << weight`), and `XorShiftInto` does the same over GF(2) for
// the parity polynomial product prod_i (1 + x^weight_i), whose coefficient of x^s is the number of
// subsets summing to s modulo two. Four DPs are built on top: (1) subset-sum reachability with a
// capacity mask, (2) parity of subset-sum counts, (3) 0/1 knapsack maximising value under a weight
// capacity in O(n * sum(values) * capacity / 64) by keeping one capacity bitset per achievable
// value level, (4) transitive closure of a DAG as `reach[v] = {v} union reach[u] for each edge
// v -> u` processed in reverse topological order, which is the bitset form of Warshall. A
// bit-parallel LCS (Crochemore et al.: `u = s & match[c]; s = (s + u) | (s - u)` on a single 64-bit
// word, LCS = number of zero bits) covers the Ch 8 recursion/DP side. "Offline" here means the
// batch of range queries is known up front: `AnswerRangeSubsetSumQueries` sorts queries by
// (left, right) and rebuilds the interval DP once per *distinct* interval, reusing it for every
// duplicate, and reports the number of builds. All the interesting inputs come from a documented
// 64-bit LCG (`state = state * 6364136223846793005 + 1442695040888963407`, MMIX constants) rather
// than the repo's 16-bit LCG, because these jobs need many independent draws; the bit-parallel LCS
// is single-word and therefore documented as limited to patterns of at most 62 characters.
//
// Documented checks: the shift-or word routine is compared against a bit-by-bit naive shift on a
// 200-bit set for every shift in [0, 130], and an aliased in-place update matches an out-of-place
// one; subset-sum reachability equals the exhaustive enumeration of all 2^n subsets for a 7-item
// hand instance and a 14-item LCG instance (16384 subsets), including an instance that contains a
// zero weight, and the popcount equals the number of distinct reachable sums; adding an item can
// only grow the reachable set (subset test); the GF(2) parity bitset equals the parity of the
// brute-force subset counts on the same instance, and on a zero-free copy of it the count of sum 0 is
// exactly 1 and hence parity-odd (the original list contains zero weights, which double that count
// and make it even -- so the file documents why the zero-free instance is the right one to pin the
// empty-subset contribution down); the level-wise knapsack bitset DP matches the
// classic O(n*capacity) DP on the hand instance (values 4,5,6 / weights 3,4,5 / capacity 7 gives 9),
// on a capacity-0 instance, on an empty item list and on four LCG instances; transitive closure of a
// 30-vertex LCG DAG equals per-source BFS for all 900 ordered pairs, contains its own diagonal, is
// idempotent, is monotone along edges, and on a 40-vertex chain reproduces the closed form
// {i..n-1}; the offline batch answers every query exactly as a per-query rebuild does, performs
// exactly one build per distinct interval (checked against a distinct-interval count) and is
// reproducible; the bit-parallel LCS matches both reference LCS implementations (O(nm) DP and
// memoised recursion) on LCG strings, reproduces the textbook values LCS("ABCDGH","AEDFHR") = 3 and
// LCS("AGGTAB","GXTXAYB") = 4, is bounded by min(|a|, |b|), rejects patterns longer than 62
// characters, and the O(nm) DP agrees with memoised recursion on 200-character strings, which the
// single-word bit-parallel routine cannot reach.
using System;
using System.Collections.Generic;

internal sealed class BitSet
{
    internal readonly int Length;
    internal readonly ulong[] Words;

    internal BitSet(int length)
    {
        Length = length;
        Words = new ulong[(length + 63) >> 6];
    }

    internal bool Get(int index)
    {
        if (index < 0 || index >= Length) return false;
        return ((Words[index >> 6] >> (index & 63)) & 1UL) != 0UL;
    }

    internal void Set(int index)
    {
        Words[index >> 6] |= 1UL << (index & 63);
    }

    internal bool IsEmpty()
    {
        foreach (ulong word in Words) if (word != 0UL) return false;
        return true;
    }

    /** Drops the padding bits above Length so that word comparisons are structural. */
    internal void Trim()
    {
        int used = Length & 63;
        if (used != 0) Words[Words.Length - 1] &= (1UL << used) - 1UL;
    }

    internal int PopCount()
    {
        int total = 0;
        foreach (ulong word in Words)
        {
            ulong x = word;
            while (x != 0UL)
            {
                x &= x - 1UL;
                total++;
            }
        }
        return total;
    }

    internal List<int> ToList()
    {
        var values = new List<int>();
        for (int i = 0; i < Length; i++) if (Get(i)) values.Add(i);
        return values;
    }

    internal BitSet Clone()
    {
        var copy = new BitSet(Length);
        Array.Copy(Words, copy.Words, Words.Length);
        return copy;
    }

    internal bool StructurallyEquals(BitSet other)
    {
        if (Length != other.Length) return false;
        for (int i = 0; i < Words.Length; i++) if (Words[i] != other.Words[i]) return false;
        return true;
    }

    internal bool IsSubsetOf(BitSet other)
    {
        for (int i = 0; i < Words.Length; i++) if ((Words[i] & ~other.Words[i]) != 0UL) return false;
        return true;
    }

    internal void OrWith(BitSet other)
    {
        for (int i = 0; i < Words.Length; i++) Words[i] |= other.Words[i];
        Trim();
    }

    /** dest |= source << shift. Word order is descending, so dest and source may alias. */
    internal static void OrShiftInto(BitSet destination, BitSet source, int shift)
    {
        ShiftInto(destination, source, shift, false);
    }

    /** dest ^= source << shift (the GF(2) counterpart of OrShiftInto). */
    internal static void XorShiftInto(BitSet destination, BitSet source, int shift)
    {
        ShiftInto(destination, source, shift, true);
    }

    private static void ShiftInto(BitSet destination, BitSet source, int shift, bool xor)
    {
        int wordShift = shift >> 6;
        int bitShift = shift & 63;
        for (int i = source.Words.Length - 1; i >= 0; i--)
        {
            ulong value = source.Words[i];
            if (value == 0UL) continue;
            int target = i + wordShift;
            if (target >= destination.Words.Length) continue;
            ulong low = value << bitShift;
            destination.Words[target] = xor ? destination.Words[target] ^ low : destination.Words[target] | low;
            if (bitShift != 0)
            {
                ulong high = value >> (64 - bitShift);
                if (high != 0UL && target + 1 < destination.Words.Length)
                {
                    destination.Words[target + 1] = xor
                            ? destination.Words[target + 1] ^ high
                            : destination.Words[target + 1] | high;
                }
            }
        }
        destination.Trim();
    }
}

internal static class BitsetDp
{
    /** Reachable subset sums of the first `count` weights, keeping only sums <= limit. */
    internal static BitSet SubsetSumReachable(int[] weights, int count, int limit)
    {
        var reach = new BitSet(limit + 1);
        reach.Set(0);
        for (int i = 0; i < count; i++) BitSet.OrShiftInto(reach, reach, weights[i]);
        return reach;
    }

    /** Exhaustive enumeration of all 2^count subsets, used as the brute-force reference. */
    internal static HashSet<int> BruteForceSums(int[] weights, int count, int limit)
    {
        var sums = new HashSet<int>();
        long subsets = 1L << count;
        for (long mask = 0; mask < subsets; mask++)
        {
            int sum = 0;
            for (int i = 0; i < count; i++) if ((mask >> i & 1L) != 0L) sum += weights[i];
            if (sum <= limit) sums.Add(sum);
        }
        return sums;
    }

    /** Parity of the number of subsets achieving each sum: prod_i (1 + x^weights[i]) over GF(2). */
    internal static BitSet ParitySubsetSums(int[] weights, int count, int limit)
    {
        var parity = new BitSet(limit + 1);
        parity.Set(0);
        for (int i = 0; i < count; i++) BitSet.XorShiftInto(parity, parity, weights[i]);
        return parity;
    }

    internal static int[] BruteForceCounts(int[] weights, int count, int limit)
    {
        var counts = new int[limit + 1];
        long subsets = 1L << count;
        for (long mask = 0; mask < subsets; mask++)
        {
            int sum = 0;
            for (int i = 0; i < count; i++) if ((mask >> i & 1L) != 0L) sum += weights[i];
            if (sum <= limit) counts[sum]++;
        }
        return counts;
    }

    /** 0/1 knapsack via one capacity bitset per total-value level. */
    internal static int MaxValueKnapsackBitset(int[] values, int[] weights, int capacity)
    {
        int total = 0;
        foreach (int value in values) total += value;
        var reach = new BitSet[total + 1];
        for (int t = 0; t <= total; t++) reach[t] = new BitSet(capacity + 1);
        reach[0].Set(0);
        for (int i = 0; i < values.Length; i++)
        {
            for (int t = total - values[i]; t >= 0; t--)
            {
                if (reach[t].IsEmpty()) continue;
                BitSet.OrShiftInto(reach[t + values[i]], reach[t], weights[i]);
            }
        }
        for (int t = total; t >= 0; t--) if (!reach[t].IsEmpty()) return t;
        return 0;
    }

    /** Classic O(n * capacity) value DP used as the independent reference. */
    internal static int MaxValueKnapsackClassic(int[] values, int[] weights, int capacity)
    {
        const int Unreachable = int.MinValue;
        var dp = new int[capacity + 1];
        for (int c = 0; c <= capacity; c++) dp[c] = Unreachable;
        dp[0] = 0;
        for (int i = 0; i < values.Length; i++)
        {
            for (int c = capacity; c >= weights[i]; c--)
            {
                if (dp[c - weights[i]] == Unreachable) continue;
                int candidate = dp[c - weights[i]] + values[i];
                if (candidate > dp[c]) dp[c] = candidate;
            }
        }
        int best = Unreachable;
        for (int c = 0; c <= capacity; c++) if (dp[c] > best) best = dp[c];
        return best;
    }

    /** Bitset transitive closure of a DAG given as an adjacency list; reach[v] contains v itself. */
    internal static BitSet[] TransitiveClosure(int n, List<int>[] adjacency)
    {
        int[] order = TopologicalOrder(n, adjacency);
        var reach = new BitSet[n];
        for (int i = 0; i < n; i++) reach[i] = new BitSet(n);
        foreach (int v in order)
        {
            reach[v].Set(v);
            foreach (int u in adjacency[v])
            {
                for (int w = 0; w < reach[u].Words.Length; w++) reach[v].Words[w] |= reach[u].Words[w];
            }
        }
        return reach;
    }

    /** Kahn's algorithm; the tested graphs are DAGs, so the order always consumes all vertices. */
    internal static int[] TopologicalOrder(int n, List<int>[] adjacency)
    {
        var indegree = new int[n];
        foreach (List<int> edges in adjacency) foreach (int u in edges) indegree[u]++;
        var queue = new Queue<int>();
        for (int v = 0; v < n; v++) if (indegree[v] == 0) queue.Enqueue(v);
        var order = new List<int>();
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            order.Add(v);
            foreach (int u in adjacency[v])
            {
                indegree[u]--;
                if (indegree[u] == 0) queue.Enqueue(u);
            }
        }
        order.Reverse();
        return order.ToArray();
    }

    /** Per-source BFS reachability (including the source itself) used as the reference. */
    internal static BitSet BfsReachability(int n, List<int>[] adjacency, int source)
    {
        var reach = new BitSet(n);
        var queue = new Queue<int>();
        reach.Set(source);
        queue.Enqueue(source);
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            foreach (int u in adjacency[v])
            {
                if (!reach.Get(u))
                {
                    reach.Set(u);
                    queue.Enqueue(u);
                }
            }
        }
        return reach;
    }

    internal sealed class OfflineResult
    {
        internal bool[] Answers = Array.Empty<bool>();
        internal int Builds;
    }

    /**
     * Offline batch of subset-sum range queries {left, right, target}: is there a subset of
     * weights[left..right) summing exactly to target? Queries are sorted by (left, right) and the
     * interval DP is rebuilt only when the interval changes, so the number of builds equals the
     * number of distinct intervals.
     */
    internal static OfflineResult AnswerRangeSubsetSumQueries(int[] weights, int[][] queries)
    {
        var order = new List<int>();
        for (int i = 0; i < queries.Length; i++) order.Add(i);
        order.Sort((x, y) =>
        {
            if (queries[x][0] != queries[y][0]) return queries[x][0] - queries[y][0];
            if (queries[x][1] != queries[y][1]) return queries[x][1] - queries[y][1];
            return x - y;
        });
        var answers = new bool[queries.Length];
        var dp = new BitSet(1);
        int builds = 0;
        int lastLeft = int.MinValue;
        int lastRight = int.MinValue;
        foreach (int qi in order)
        {
            int left = queries[qi][0];
            int right = queries[qi][1];
            int target = queries[qi][2];
            if (left != lastLeft || right != lastRight)
            {
                int sum = 0;
                for (int i = left; i < right; i++) sum += weights[i];
                dp = new BitSet(sum + 1);
                dp.Set(0);
                for (int i = left; i < right; i++) BitSet.OrShiftInto(dp, dp, weights[i]);
                builds++;
                lastLeft = left;
                lastRight = right;
            }
            answers[qi] = dp.Get(target);
        }
        return new OfflineResult { Answers = answers, Builds = builds };
    }

    /** Brute-force answer of one range query, for cross-checking the offline batch. */
    internal static bool BruteForceRangeQuery(int[] weights, int left, int right, int target)
    {
        int count = right - left;
        long subsets = 1L << count;
        for (long mask = 0; mask < subsets; mask++)
        {
            int sum = 0;
            for (int i = 0; i < count; i++) if ((mask >> i & 1L) != 0L) sum += weights[left + i];
            if (sum == target) return true;
        }
        return false;
    }

    /**
     * Bit-parallel LCS on single 64-bit words (Crochemore et al.). `s` keeps one bit per pattern
     * position; after processing each text character the number of zero bits inside the mask is the
     * LCS length. Limited to patterns of at most 62 characters.
     */
    internal static int LcsBitParallel(string pattern, string text)
    {
        int n = pattern.Length;
        if (n > 62) throw new ArgumentException("Bit-parallel LCS is single-word: pattern too long.");
        var match = new Dictionary<char, ulong>();
        for (int i = 0; i < n; i++)
        {
            match.TryGetValue(pattern[i], out ulong bits);
            match[pattern[i]] = bits | (1UL << i);
        }
        ulong mask = (1UL << n) - 1UL;
        ulong s = mask;
        foreach (char c in text)
        {
            match.TryGetValue(c, out ulong m);
            ulong u = s & m;
            s = ((s + u) | (s - u)) & mask;
        }
        ulong zeros = mask & ~s;
        int length = 0;
        while (zeros != 0UL)
        {
            zeros &= zeros - 1UL;
            length++;
        }
        return length;
    }

    internal static int LcsQuadratic(string a, string b)
    {
        var dp = new int[a.Length + 1, b.Length + 1];
        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                dp[i, j] = a[i - 1] == b[j - 1]
                        ? dp[i - 1, j - 1] + 1
                        : Math.Max(dp[i - 1, j], dp[i, j - 1]);
            }
        }
        return dp[a.Length, b.Length];
    }

    internal static int LcsMemoized(string a, string b)
    {
        int n = a.Length;
        int m = b.Length;
        var memo = new int[n + 1, m + 1];
        for (int i = 0; i <= n; i++) for (int j = 0; j <= m; j++) memo[i, j] = -1;

        int Solve(int i, int j)
        {
            if (i == 0 || j == 0) return 0;
            if (memo[i, j] >= 0) return memo[i, j];
            int value = a[i - 1] == b[j - 1]
                    ? Solve(i - 1, j - 1) + 1
                    : Math.Max(Solve(i - 1, j), Solve(i, j - 1));
            memo[i, j] = value;
            return value;
        }

        return Solve(n, m);
    }
}

internal static class Lcg64
{
    /** 64-bit LCG with the MMIX constants; documented replacement for the 16-bit repo LCG. */
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<int>[] RandomDag(int n, int edges, ulong seed)
    {
        var adjacency = new List<int>[n];
        for (int v = 0; v < n; v++) adjacency[v] = new List<int>();
        ulong state = seed;
        for (int e = 0; e < edges; e++)
        {
            int u = Lcg64.NextBelow(ref state, n);
            int v = Lcg64.NextBelow(ref state, n);
            if (u > v) { int t = u; u = v; v = t; }
            if (u == v || adjacency[u].Contains(v)) continue;
            adjacency[u].Add(v);
        }
        return adjacency;
    }

    private static string RandomWord(int length, ulong seed, int alphabet)
    {
        var chars = new char[length];
        ulong state = seed;
        for (int i = 0; i < length; i++) chars[i] = (char)('a' + Lcg64.NextBelow(ref state, alphabet));
        return new string(chars);
    }

    private static List<int>[] ClosureToAdjacency(BitSet[] closure, int n)
    {
        var adjacency = new List<int>[n];
        for (int v = 0; v < n; v++)
        {
            adjacency[v] = new List<int>();
            for (int w = 0; w < n; w++) if (w != v && closure[v].Get(w)) adjacency[v].Add(w);
        }
        return adjacency;
    }

    public static void Main()
    {
        // ---- the shift-or primitive itself ------------------------------------------------
        var source = new BitSet(200);
        ulong state = 584001UL;
        for (int i = 0; i < 200; i++) if (Lcg64.NextBelow(ref state, 2) == 1) source.Set(i);
        for (int shift = 0; shift <= 130; shift++)
        {
            var word = source.Clone();
            BitSet.OrShiftInto(word, source, shift);
            var naive = new BitSet(200);
            for (int i = 0; i < 200; i++) if (source.Get(i) && i + shift < 200) naive.Set(i + shift);
            naive.OrWith(source);
            Check(word.StructurallyEquals(naive), "shift-or matches the bit-by-bit reference at " + shift);
        }
        var aliased = source.Clone();
        BitSet.OrShiftInto(aliased, aliased, 7);
        var outOfPlace = source.Clone();
        BitSet.OrShiftInto(outOfPlace, source.Clone(), 7);
        Check(aliased.StructurallyEquals(outOfPlace), "aliased shift-or equals the out-of-place form");

        // ---- subset-sum reachability ------------------------------------------------------
        int[] hand = { 3, 5, 7, 11, 13, 17, 19 };
        BitSet handReach = BitsetDp.SubsetSumReachable(hand, hand.Length, 75);
        HashSet<int> handBrute = BitsetDp.BruteForceSums(hand, hand.Length, 75);
        Check(handReach.PopCount() == handBrute.Count, "hand popcount equals the brute-force sum count");
        foreach (int sum in handReach.ToList()) Check(handBrute.Contains(sum), "hand reachable sum " + sum);
        foreach (int sum in handBrute) Check(handReach.Get(sum), "hand brute-force sum " + sum);
        Check(handBrute.Contains(75) && handBrute.Contains(0), "the hand instance reaches 0 and 75");
        Check(!handBrute.Contains(1), "the hand instance cannot reach 1");

        ulong weightsSeed = 584123UL;
        var lcgWeights = new int[14];
        for (int i = 0; i < 14; i++) lcgWeights[i] = Lcg64.NextBelow(ref weightsSeed, 40);
        BitSet lcgReach = BitsetDp.SubsetSumReachable(lcgWeights, lcgWeights.Length, 200);
        HashSet<int> lcgBrute = BitsetDp.BruteForceSums(lcgWeights, lcgWeights.Length, 200);
        Check(lcgReach.PopCount() == lcgBrute.Count, "LCG popcount equals the 16384-subset enumeration");
        foreach (int sum in lcgBrute) Check(lcgReach.Get(sum), "LCG brute-force sum " + sum);
        Check(lcgReach.PopCount() > 100, "the LCG instance reaches many sums (" + lcgReach.PopCount() + ")");
        Check(lcgReach.Get(0), "the empty subset always reaches 0");

        var withZero = new int[15];
        Array.Copy(lcgWeights, withZero, 14);
        withZero[14] = 0;
        BitSet zeroReach = BitsetDp.SubsetSumReachable(withZero, withZero.Length, 200);
        Check(zeroReach.StructurallyEquals(lcgReach), "a zero weight leaves reachability unchanged");

        var extended = new int[15];
        Array.Copy(lcgWeights, extended, 14);
        extended[14] = 23;
        BitSet extendedReach = BitsetDp.SubsetSumReachable(extended, extended.Length, 200);
        Check(lcgReach.IsSubsetOf(extendedReach), "adding an item can only grow the reachable set");
        Check(extendedReach.Get(23) && !lcgReach.Get(23), "the new item adds the sum 23");

        BitSet truncated = BitsetDp.SubsetSumReachable(hand, hand.Length, 20);
        var maskedBrute = new HashSet<int>();
        foreach (int sum in handBrute) if (sum <= 20) maskedBrute.Add(sum);
        Check(truncated.PopCount() == maskedBrute.Count,
                "the capacity mask keeps exactly the " + maskedBrute.Count + " sums within the capacity");
        foreach (int sum in maskedBrute) Check(truncated.Get(sum), "the capacity mask keeps " + sum);
        Check(truncated.Get(20), "3 + 17 fits exactly into the capacity 20");
        Check(truncated.Get(19), "the item 19 fits into the capacity 20");
        Check(!truncated.Get(21), "the capacity 20 excludes 21");

        // ---- GF(2) parity of subset counts ------------------------------------------------
        var parityWeights = new int[16];
        ulong paritySeed = 584777UL;
        for (int i = 0; i < 16; i++) parityWeights[i] = Lcg64.NextBelow(ref paritySeed, 12);
        const int ParityLimit = 60;
        BitSet parity = BitsetDp.ParitySubsetSums(parityWeights, parityWeights.Length, ParityLimit);
        int[] counts = BitsetDp.BruteForceCounts(parityWeights, parityWeights.Length, ParityLimit);
        for (int s = 0; s <= ParityLimit; s++)
        {
            Check(parity.Get(s) == (counts[s] % 2 == 1), "parity of the subset count of sum " + s);
        }
        Check(parity.Get(0) == (counts[0] % 2 == 1), "sum 0 parity matches the brute-force count parity");
        // The LCG weight list above contains zeros (each zero doubles the number of subsets reaching
        // any sum it can be added to), so an even count at sum 0 is legitimate there. The zero-free
        // copy pins the empty-subset contribution down instead.
        var zeroFree = new int[16];
        for (int i = 0; i < 16; i++) zeroFree[i] = parityWeights[i] == 0 ? 12 : parityWeights[i];
        int[] zeroFreeCounts = BitsetDp.BruteForceCounts(zeroFree, zeroFree.Length, ParityLimit);
        BitSet zeroFreeParity = BitsetDp.ParitySubsetSums(zeroFree, zeroFree.Length, ParityLimit);
        Check(zeroFreeCounts[0] == 1, "a zero-free weight list reaches sum 0 exactly once");
        Check(zeroFreeParity.Get(0), "the empty subset gives an odd count at sum 0");
        for (int s = 0; s <= ParityLimit; s++)
        {
            Check(zeroFreeParity.Get(s) == (zeroFreeCounts[s] % 2 == 1), "zero-free parity at sum " + s);
        }

        // ---- knapsack --------------------------------------------------------------------
        Check(BitsetDp.MaxValueKnapsackBitset(new[] { 4, 5, 6 }, new[] { 3, 4, 5 }, 7) == 9,
                "hand knapsack optimum is 9");
        Check(BitsetDp.MaxValueKnapsackClassic(new[] { 4, 5, 6 }, new[] { 3, 4, 5 }, 7) == 9,
                "the classic reference also finds 9");
        Check(BitsetDp.MaxValueKnapsackBitset(new[] { 4, 5, 6 }, new[] { 3, 4, 5 }, 0) == 0,
                "capacity 0 admits only the empty knapsack");
        Check(BitsetDp.MaxValueKnapsackBitset(Array.Empty<int>(), Array.Empty<int>(), 25) == 0,
                "no items means value 0");
        for (int trial = 0; trial < 4; trial++)
        {
            ulong knapSeed = 584500UL + (ulong)trial;
            var values = new int[12];
            var weights = new int[12];
            for (int i = 0; i < 12; i++)
            {
                values[i] = 1 + Lcg64.NextBelow(ref knapSeed, 9);
                weights[i] = 1 + Lcg64.NextBelow(ref knapSeed, 16);
            }
            int capacity = 40 + Lcg64.NextBelow(ref knapSeed, 30);
            int bitsetAnswer = BitsetDp.MaxValueKnapsackBitset(values, weights, capacity);
            int classicAnswer = BitsetDp.MaxValueKnapsackClassic(values, weights, capacity);
            Check(bitsetAnswer == classicAnswer,
                    "knapsack trial " + trial + ": bitset " + bitsetAnswer + " vs classic " + classicAnswer);
        }

        // ---- transitive closure ----------------------------------------------------------
        const int N = 30;
        List<int>[] dag = RandomDag(N, 90, 584999UL);
        BitSet[] closure = BitsetDp.TransitiveClosure(N, dag);
        int edgeCount = 0;
        foreach (List<int> edges in dag) edgeCount += edges.Count;
        Check(edgeCount > 40, "the LCG DAG has a nontrivial number of edges (" + edgeCount + ")");
        for (int v = 0; v < N; v++)
        {
            Check(closure[v].Get(v), "the closure of vertex " + v + " contains itself");
            Check(closure[v].StructurallyEquals(BitsetDp.BfsReachability(N, dag, v)),
                    "closure of vertex " + v + " equals BFS reachability");
        }
        BitSet[] closureTwice = BitsetDp.TransitiveClosure(N, ClosureToAdjacency(closure, N));
        for (int v = 0; v < N; v++)
        {
            Check(closureTwice[v].StructurallyEquals(closure[v]), "transitive closure is idempotent at " + v);
        }
        for (int v = 0; v < N; v++)
        {
            foreach (int u in dag[v]) Check(closure[u].IsSubsetOf(closure[v]), "reachability is monotone along " + v + "->" + u);
        }
        const int Chain = 40;
        var chain = new List<int>[Chain];
        for (int v = 0; v < Chain; v++) chain[v] = new List<int>();
        for (int v = 0; v + 1 < Chain; v++) chain[v].Add(v + 1);
        BitSet[] chainClosure = BitsetDp.TransitiveClosure(Chain, chain);
        for (int v = 0; v < Chain; v++)
        {
            Check(chainClosure[v].PopCount() == Chain - v, "chain closure size at " + v);
            for (int w = 0; w < Chain; w++)
            {
                Check(chainClosure[v].Get(w) == (w >= v), "chain closed form at " + v + " -> " + w);
            }
        }

        // ---- offline range queries -------------------------------------------------------
        var queryWeights = new int[20];
        ulong querySeed = 584321UL;
        for (int i = 0; i < queryWeights.Length; i++) queryWeights[i] = 1 + Lcg64.NextBelow(ref querySeed, 25);
        var queries = new List<int[]>();
        var distinct = new HashSet<long>();
        for (int q = 0; q < 60; q++)
        {
            int left = Lcg64.NextBelow(ref querySeed, 18);
            int right = left + 1 + Lcg64.NextBelow(ref querySeed, 20 - left);
            int target = Lcg64.NextBelow(ref querySeed, 90);
            queries.Add(new[] { left, right, target });
            distinct.Add(((long)left << 32) | (uint)right);
        }
        for (int q = 0; q < 10; q++) queries.Add(new[] { queries[q][0], queries[q][1], 90 + q });
        BitsetDp.OfflineResult offline = BitsetDp.AnswerRangeSubsetSumQueries(queryWeights, queries.ToArray());
        Check(offline.Builds == distinct.Count, "one DP build per distinct interval (" + offline.Builds
                + " builds for " + distinct.Count + " distinct intervals)");
        for (int q = 0; q < queries.Count; q++)
        {
            bool expected = BitsetDp.BruteForceRangeQuery(queryWeights, queries[q][0], queries[q][1], queries[q][2]);
            Check(offline.Answers[q] == expected, "offline answer " + q + " matches the exhaustive rebuild");
        }
        BitsetDp.OfflineResult repeat = BitsetDp.AnswerRangeSubsetSumQueries(queryWeights, queries.ToArray());
        Check(repeat.Builds == offline.Builds, "the offline build count is reproducible");
        for (int q = 0; q < queries.Count; q++) Check(repeat.Answers[q] == offline.Answers[q], "offline answer " + q + " is stable");
        int positiveAnswers = 0;
        foreach (bool answer in offline.Answers) if (answer) positiveAnswers++;
        Check(positiveAnswers > 0 && positiveAnswers < queries.Count, "the query batch mixes positive and negative answers ("
                + positiveAnswers + " positive)");

        // ---- bit-parallel LCS ------------------------------------------------------------
        Check(BitsetDp.LcsBitParallel("ABCDGH", "AEDFHR") == 3, "textbook LCS(ABCDGH, AEDFHR) is 3");
        Check(BitsetDp.LcsBitParallel("AGGTAB", "GXTXAYB") == 4, "textbook LCS(AGGTAB, GXTXAYB) is 4");
        Check(BitsetDp.LcsBitParallel("abc", "") == 0, "an empty text has LCS 0");
        Check(BitsetDp.LcsBitParallel("", "abc") == 0, "an empty pattern has LCS 0");
        Check(BitsetDp.LcsBitParallel("xyz", "xyz") == 3, "identical strings are their own LCS");
        for (int trial = 0; trial < 6; trial++)
        {
            string a = RandomWord(1 + Lcg64.NextBelow(ref querySeed, 40), 584000UL + (ulong)trial, 4);
            string b = RandomWord(1 + Lcg64.NextBelow(ref querySeed, 40), 585000UL + (ulong)trial, 4);
            int bitParallel = BitsetDp.LcsBitParallel(a, b);
            Check(bitParallel == BitsetDp.LcsQuadratic(a, b), "bit-parallel LCS matches the O(nm) DP on trial " + trial);
            Check(bitParallel == BitsetDp.LcsMemoized(a, b), "bit-parallel LCS matches memoised recursion on trial " + trial);
            Check(bitParallel <= Math.Min(a.Length, b.Length), "LCS is bounded by min(|a|, |b|) on trial " + trial);
        }
        string longA = RandomWord(200, 586000UL, 6);
        string longB = RandomWord(200, 587000UL, 6);
        int longQuadratic = BitsetDp.LcsQuadratic(longA, longB);
        Check(longQuadratic == BitsetDp.LcsMemoized(longA, longB),
                "the O(nm) DP and memoised recursion agree on 200-character strings");
        Check(longQuadratic > 100, "long random strings over a 6-letter alphabet share a long subsequence ("
                + longQuadratic + ")");
        bool rejected = false;
        try
        {
            BitsetDp.LcsBitParallel(new string('a', 63), "aaa");
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        Check(rejected, "the single-word bit-parallel LCS rejects patterns longer than 62 characters");

        Console.WriteLine("PASS");
    }
}
