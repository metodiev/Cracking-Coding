// Task 558 - Implement boolean matrix multiplication optimization using bitsets. (Tier: Advanced, CTCI Ch 5 — Bit Manipulation)
//
// Boolean (OR-AND) matrix multiplication with 64-bit words: each row of the left operand is scanned for
// its set bits and the corresponding rows of the right operand are OR-ed together, so an n x n boolean
// product costs O(n^2 * n/64) word operations instead of the O(n^3) scalar trips of the textbook
// triple loop - and the file counts both to substantiate the constant-factor claim rather than
// asserting it. The same primitive gives: boolean matrix powers (A^k is the adjacency relation of
// "reachable in exactly k steps"), the reflexive-transitive closure by repeated squaring of I v A
// (log2 n squarings suffice because a shortest path has at most n-1 edges), and the O(n^2/64)
// row-propagation closure used by graph libraries. Every result is compared against an independent
// method: the naive triple loop for products, a count-based integer power (a boolean entry is set
// exactly when the number of walks is positive), BFS/DFS reachability for closures, Boolean algebra
// laws (associativity, identity) and the idempotence of a closed relation. Random matrices use
// splitmix64 (documented 64-bit generator); the word-boundary cases n = 64 and n = 70 are exercised
// explicitly because off-by-one errors in word counts only show up there.
//
// Documented checks: bitset products equal naive products entrywise for all 9 ordered pairs of the
// three 6-vertex test matrices (dense, sparse and a directed path, plus a diagonal-only saturated
// relation); Boolean associativity (AB)C = A(BC) and the identity A * I = A both hold; A^k has a set
// bit exactly when the integer walk count is positive, checked for k = 0..6 on six graphs with 5 to 9
// vertices (1841 cell comparisons); the closure computed by repeated squaring equals the bitset
// row-propagation closure and the BFS reachability relation, is idempotent (C * C = C), contains A and
// has a full diagonal; the closure of a 6-vertex graph (a 3-cycle plus a 3-path) matches an explicitly
// tabulated relation with 15 set pairs, and the closure of the 64-vertex directed path is exactly the
// upper triangle (2080 set bits, asserted exactly); n = 70 (two words) is checked against the naive
// product, whose square has 68 set pairs; the empty relation closes to the identity; the measured
// word-operation count of the bitset product is strictly below the naive n^3 scalar count for n = 20;
// and repeated runs are identical.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    /** splitmix64 (Steele/Lea/Flood): 64-bit state advanced by a golden-ratio increment. */
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int Below(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal static class BoolMatrix
{
    internal static long WordOps;

    internal static int Words(int n)
    {
        return (n + 63) / 64;
    }

    internal static ulong[][] FromBools(bool[,] a)
    {
        int n = a.GetLength(0);
        int w = Words(n);
        var m = new ulong[n][];
        for (int i = 0; i < n; i++)
        {
            m[i] = new ulong[w];
            for (int j = 0; j < n; j++)
            {
                if (a[i, j]) m[i][j >> 6] |= 1UL << (j & 63);
            }
        }
        return m;
    }

    internal static bool[,] ToBools(ulong[][] a, int n)
    {
        var r = new bool[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) r[i, j] = (a[i][j >> 6] & (1UL << (j & 63))) != 0UL;
        }
        return r;
    }

    internal static ulong[][] Identity(int n)
    {
        int w = Words(n);
        var m = new ulong[n][];
        for (int i = 0; i < n; i++)
        {
            m[i] = new ulong[w];
            m[i][i >> 6] |= 1UL << (i & 63);
        }
        return m;
    }

    internal static ulong[][] Multiply(ulong[][] a, ulong[][] b, int n, bool count)
    {
        int w = Words(n);
        var c = new ulong[n][];
        for (int i = 0; i < n; i++)
        {
            c[i] = new ulong[w];
            for (int t = 0; t < n; t++)
            {
                if ((a[i][t >> 6] & (1UL << (t & 63))) == 0UL) continue;
                ulong[] target = c[i];
                ulong[] source = b[t];
                for (int k = 0; k < w; k++) target[k] |= source[k];
                if (count) WordOps++;
            }
        }
        return c;
    }

    internal static bool[,] MultiplyNaive(bool[,] a, bool[,] b)
    {
        int n = a.GetLength(0);
        var c = new bool[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int t = 0; t < n; t++)
                {
                    if (a[i, t] && b[t, j]) { c[i, j] = true; break; }
                }
            }
        }
        return c;
    }

    internal static ulong[][] Power(ulong[][] a, int exponent, int n)
    {
        ulong[][] result = Identity(n);
        ulong[][] basis = a;
        int remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0) result = Multiply(result, basis, n, false);
            remaining >>= 1;
            if (remaining > 0) basis = Multiply(basis, basis, n, false);
        }
        return result;
    }

    internal static ulong[][] Or(ulong[][] a, ulong[][] b, int n)
    {
        int w = Words(n);
        var c = new ulong[n][];
        for (int i = 0; i < n; i++)
        {
            c[i] = new ulong[w];
            for (int k = 0; k < w; k++) c[i][k] = a[i][k] | b[i][k];
        }
        return c;
    }

    internal static bool Equal(ulong[][] a, ulong[][] b, int n)
    {
        int w = Words(n);
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < w; k++)
            {
                if (a[i][k] != b[i][k]) return false;
            }
        }
        return true;
    }

    /** Reflexive-transitive closure by log2(n) squarings of I v A. */
    internal static ulong[][] ClosureBySquaring(ulong[][] a, int n)
    {
        ulong[][] r = Or(Identity(n), a, n);
        int steps = 1;
        while (steps < n)
        {
            r = Multiply(r, r, n, false);
            steps *= 2;
        }
        return r;
    }

    /** Closure by row propagation: OR the row of every reachable neighbour into the source row. */
    internal static ulong[][] ClosureByPropagation(ulong[][] a, int n)
    {
        int w = Words(n);
        var r = new ulong[n][];
        for (int i = 0; i < n; i++)
        {
            r[i] = new ulong[w];
            Array.Copy(a[i], r[i], w);
            r[i][i >> 6] |= 1UL << (i & 63);
        }
        for (int i = 0; i < n; i++)
        {
            var queue = new Queue<int>();
            var seen = new bool[n];
            for (int t = 0; t < n; t++)
            {
                if ((r[i][t >> 6] & (1UL << (t & 63))) != 0UL) { seen[t] = true; queue.Enqueue(t); }
            }
            while (queue.Count > 0)
            {
                int at = queue.Dequeue();
                for (int v = 0; v < n; v++)
                {
                    if ((r[at][v >> 6] & (1UL << (v & 63))) == 0UL || seen[v]) continue;
                    seen[v] = true;
                    queue.Enqueue(v);
                    r[i][v >> 6] |= 1UL << (v & 63);
                }
            }
        }
        return r;
    }

    internal static bool[,] ReachabilityByDfs(bool[,] a)
    {
        int n = a.GetLength(0);
        var r = new bool[n, n];
        for (int s = 0; s < n; s++)
        {
            var stack = new Stack<int>();
            var seen = new bool[n];
            stack.Push(s);
            seen[s] = true;
            while (stack.Count > 0)
            {
                int at = stack.Pop();
                r[s, at] = true;
                for (int v = 0; v < n; v++)
                {
                    if (a[at, v] && !seen[v]) { seen[v] = true; stack.Push(v); }
                }
            }
        }
        return r;
    }

    internal static int SetBits(ulong[][] a, int n)
    {
        int total = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((a[i][j >> 6] & (1UL << (j & 63))) != 0UL) total++;
            }
        }
        return total;
    }

    internal static long[,] IntegerPower(bool[,] a, int exponent)
    {
        int n = a.GetLength(0);
        var current = new long[n, n];
        for (int i = 0; i < n; i++) current[i, i] = 1;
        for (int step = 0; step < exponent; step++)
        {
            var next = new long[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int t = 0; t < n; t++)
                {
                    if (current[i, t] == 0) continue;
                    for (int j = 0; j < n; j++)
                    {
                        if (a[t, j]) next[i, j] += current[i, t];
                    }
                }
            }
            current = next;
        }
        return current;
    }

    internal static bool[,] RandomBools(int n, int probabilityPpm, ref ulong state)
    {
        var m = new bool[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (SplitMix64.Below(ref state, 1000000) < probabilityPpm) m[i, j] = true;
            }
        }
        return m;
    }

    internal static bool[,] PathDigraph(int n)
    {
        var m = new bool[n, n];
        for (int i = 0; i + 1 < n; i++) m[i, i + 1] = true;
        return m;
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 558001UL;
        var graphs = new List<bool[,]>
        {
            BoolMatrix.RandomBools(5, 500000, ref rng),
            BoolMatrix.RandomBools(6, 200000, ref rng),
            BoolMatrix.RandomBools(7, 800000, ref rng),
            BoolMatrix.RandomBools(9, 400000, ref rng),
            BoolMatrix.PathDigraph(6),
        };
        // A cycle 0->1->2->0 plus a path 3->4->5, hand-tabulated below.
        var bipart = new bool[6, 6];
        bipart[0, 1] = bipart[1, 2] = bipart[2, 0] = true;
        bipart[3, 4] = bipart[4, 5] = true;
        graphs.Add(bipart);

        // Bitset products equal naive products on every ordered pair of the 6-vertex test matrices.
        var sixVertex = new List<bool[,]>
        {
            BoolMatrix.RandomBools(6, 450000, ref rng),
            BoolMatrix.RandomBools(6, 200000, ref rng),
            BoolMatrix.PathDigraph(6),
        };
        int products = 0;
        foreach (bool[,] g in sixVertex)
        {
            foreach (bool[,] h in sixVertex)
            {
                bool[,] naive = BoolMatrix.MultiplyNaive(g, h);
                bool[,] bitset = BoolMatrix.ToBools(
                        BoolMatrix.Multiply(BoolMatrix.FromBools(g), BoolMatrix.FromBools(h), 6, false), 6);
                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        Check(naive[i, j] == bitset[i, j], "bitset product != naive at " + i + "," + j);
                    }
                }
                products++;
            }
        }
        Check(products == 9, "9 ordered product comparisons, got " + products);

        // Associativity and the identity matrix.
        var m5 = BoolMatrix.RandomBools(6, 450000, ref rng);
        var m6 = BoolMatrix.RandomBools(6, 350000, ref rng);
        var m7 = BoolMatrix.RandomBools(6, 250000, ref rng);
        int n6 = 6;
        ulong[][] a6 = BoolMatrix.FromBools(m5);
        ulong[][] b6 = BoolMatrix.FromBools(m7);
        ulong[][] c6 = BoolMatrix.FromBools(m6);
        ulong[][] left = BoolMatrix.Multiply(BoolMatrix.Multiply(a6, c6, n6, false), b6, n6, false);
        ulong[][] right = BoolMatrix.Multiply(a6, BoolMatrix.Multiply(c6, b6, n6, false), n6, false);
        Check(BoolMatrix.Equal(left, right, n6), "boolean multiplication is associative");
        Check(BoolMatrix.Equal(BoolMatrix.Multiply(a6, BoolMatrix.Identity(n6), n6, false), a6, n6),
                "the identity matrix is neutral");
        Check(BoolMatrix.Equal(BoolMatrix.Multiply(BoolMatrix.Identity(n6), a6, n6, false), a6, n6),
                "the identity matrix is neutral on the left too");
        var diagonal = new bool[4, 4];
        for (int i = 0; i < 4; i++) diagonal[i, i] = true;
        Check(BoolMatrix.Equal(BoolMatrix.Multiply(BoolMatrix.FromBools(diagonal),
                BoolMatrix.FromBools(diagonal), 4, false), BoolMatrix.FromBools(diagonal), 4),
                "a diagonal relation is saturated");

        // A^k is set exactly where the integer walk count is positive.
        int cells = 0;
        foreach (bool[,] g in graphs)
        {
            int n = g.GetLength(0);
            ulong[][] bits = BoolMatrix.FromBools(g);
            for (int k = 0; k <= 6; k++)
            {
                ulong[][] power = BoolMatrix.Power(bits, k, n);
                long[,] counts = BoolMatrix.IntegerPower(g, k);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        bool set = (power[i][j >> 6] & (1UL << (j & 63))) != 0UL;
                        Check(set == (counts[i, j] > 0), "A^k bit disagrees with the walk count at k = " + k
                                + " cell " + i + "," + j);
                        cells++;
                    }
                }
            }
        }
        Check(cells == 7 * 263, "7 lengths x (25+36+49+81+36+36) cells = 1841, got " + cells);

        // Closures: squaring, propagation and DFS must all agree, and be idempotent.
        foreach (bool[,] g in graphs)
        {
            int n = g.GetLength(0);
            ulong[][] bits = BoolMatrix.FromBools(g);
            ulong[][] squared = BoolMatrix.ClosureBySquaring(bits, n);
            ulong[][] propagated = BoolMatrix.ClosureByPropagation(bits, n);
            bool[,] dfs = BoolMatrix.ReachabilityByDfs(g);
            Check(BoolMatrix.Equal(squared, propagated, n), "squaring closure != propagation closure");
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    bool set = (squared[i][j >> 6] & (1UL << (j & 63))) != 0UL;
                    Check(set == dfs[i, j], "closure != DFS reachability at " + i + "," + j);
                    if (i == j) Check(set, "the closure is reflexive at " + i);
                }
            }
            Check(BoolMatrix.Equal(BoolMatrix.Multiply(squared, squared, n, false), squared, n),
                    "a closed relation is idempotent");
            Check(BoolMatrix.Equal(BoolMatrix.Or(squared, bits, n), squared, n), "the closure contains A");
        }

        // The hand-tabulated example: {0,1,2} form a cycle, 3 -> 4 -> 5.
        var bipartClosure = BoolMatrix.ClosureBySquaring(BoolMatrix.FromBools(bipart), 6);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Check((bipartClosure[i][j >> 6] & (1UL << (j & 63))) != 0UL,
                        "the 3-cycle closes into a complete relation at " + i + "," + j);
            }
            Check((bipartClosure[i][3 >> 6] & (1UL << (3 & 63))) == 0UL
                    && (bipartClosure[3][i >> 6] & (1UL << (i & 63))) == 0UL,
                    "the cycle and the path stay disconnected at " + i);
        }
        Check((bipartClosure[3][5 >> 6] & (1UL << (5 & 63))) != 0UL, "3 reaches 5 through 4");
        Check((bipartClosure[5][3 >> 6] & (1UL << (3 & 63))) == 0UL, "5 cannot reach 3");
        Check(BoolMatrix.SetBits(bipartClosure, 6) == 9 + 6, "the example closure has 9 + 6 set pairs");

        // Word-boundary cases: the 64-vertex path closes to the upper triangle.
        var path64 = BoolMatrix.PathDigraph(64);
        ulong[][] closure64 = BoolMatrix.ClosureBySquaring(BoolMatrix.FromBools(path64), 64);
        Check(BoolMatrix.SetBits(closure64, 64) == 64 * 65 / 2,
                "the closure of the 64-path has 2080 set pairs, got " + BoolMatrix.SetBits(closure64, 64));
        bool triangular = true;
        for (int i = 0; i < 64 && triangular; i++)
        {
            for (int j = 0; j < 64; j++)
            {
                bool set = (closure64[i][j >> 6] & (1UL << (j & 63))) != 0UL;
                if (set != (i <= j)) { triangular = false; break; }
            }
        }
        Check(triangular, "the 64-path closure is exactly i <= j");

        var path70 = BoolMatrix.PathDigraph(70);
        bool[,] naive70 = BoolMatrix.MultiplyNaive(path70, path70);
        bool[,] bits70 = BoolMatrix.ToBools(BoolMatrix.Multiply(BoolMatrix.FromBools(path70),
                BoolMatrix.FromBools(path70), 70, false), 70);
        bool same = true;
        for (int i = 0; i < 70 && same; i++)
        {
            for (int j = 0; j < 70; j++)
            {
                if (naive70[i, j] != bits70[i, j]) { same = false; break; }
            }
        }
        Check(same, "n = 70 (two words) matches the naive product");
        Check(BoolMatrix.SetBits(BoolMatrix.Multiply(BoolMatrix.FromBools(path70),
                BoolMatrix.FromBools(path70), 70, false), 70) == 68,
                "the square of the 70-path has 68 set pairs");

        // The optimization really reduces the operation count.
        int big = 20;
        ulong beforeDense = rng;
        var dense = BoolMatrix.RandomBools(big, 500000, ref rng);
        ulong[][] denseBits = BoolMatrix.FromBools(dense);
        BoolMatrix.WordOps = 0;
        BoolMatrix.Multiply(denseBits, denseBits, big, true);
        long words = BoolMatrix.WordOps;
        Check(words < (long)big * big * big, "bitset product used " + words + " word-ops, naive needs "
                + (long)big * big * big);

        // The empty relation closes to the identity.
        var empty4 = new bool[4, 4];
        ulong[][] emptyClosure = BoolMatrix.ClosureBySquaring(BoolMatrix.FromBools(empty4), 4);
        Check(BoolMatrix.SetBits(emptyClosure, 4) == 4, "the empty relation closes to the identity");

        // Determinism.
        ulong sameRng = beforeDense;
        var repeatDense = BoolMatrix.RandomBools(big, 500000, ref sameRng);
        Check(BoolMatrix.Equal(BoolMatrix.FromBools(repeatDense), denseBits, big),
                "the same seed reproduces the random matrix");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
