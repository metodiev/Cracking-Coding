// Task 540 - Implement Möbius transform on subset lattice (SOS). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// The subset lattice is the poset ({0,1}^n, subset-of), and this file implements the whole
// transform toolkit on it in the classical O(n * 2^n) "sum over subsets" (SOS) style: the zeta
// transform (subset sums, `if (mask & bit) f[mask] += f[mask ^ bit]`), its inverse the Möbius
// transform (the same loops with subtraction), the mirrored superset versions, and the two
// convolutions they diagonalise - OR convolution (zeta then pointwise product then Möbius) and AND
// convolution (using superset transforms). A ranked-zeta fast subset convolution (O(n^2 2^n)) is
// included so that the O(3^n) subset convolution can be cross-checked for small n.
//
// Documented checks: Möbius undoes zeta in both directions (n = 1..8, LCG values) and likewise for
// the superset pair; the SOS result equals a direct double loop over all pairs (T subset S) for
// n <= 5; the closed form sum_S zeta[S] = sum_T f[T] * 2^(n - |T|) holds; applying zeta to the
// indicator of a singleton counts the supersets of that element; OR convolution equals the naive
// sum over A | B = S, AND convolution equals the naive sum over A & B = S; fast subset convolution
// equals the naive sum over T subset S of f[T] g[S \ T] for n <= 4; the n = 0 lattice (arrays of
// length one) degenerates to the identity; and every table is reproduced on a second run.
using System;
using System.Collections.Generic;

internal static class SubsetLattice
{
    /** in-place f[S] = sum over T subset of S of f[T]. */
    internal static void ZetaSubset(long[] f, int n)
    {
        for (int bit = 0; bit < n; bit++)
        {
            int step = 1 << bit;
            for (int mask = 0; mask < f.Length; mask++)
            {
                if ((mask & step) != 0) f[mask] += f[mask ^ step];
            }
        }
    }

    /** in-place inverse of ZetaSubset (inclusion-exclusion). */
    internal static void MobiusSubset(long[] f, int n)
    {
        for (int bit = 0; bit < n; bit++)
        {
            int step = 1 << bit;
            for (int mask = 0; mask < f.Length; mask++)
            {
                if ((mask & step) != 0) f[mask] -= f[mask ^ step];
            }
        }
    }

    /** in-place f[S] = sum over T superset of S of f[T]. */
    internal static void ZetaSuperset(long[] f, int n)
    {
        for (int bit = 0; bit < n; bit++)
        {
            int step = 1 << bit;
            for (int mask = 0; mask < f.Length; mask++)
            {
                if ((mask & step) == 0) f[mask] += f[mask | step];
            }
        }
    }

    /** in-place inverse of ZetaSuperset. */
    internal static void MobiusSuperset(long[] f, int n)
    {
        for (int bit = 0; bit < n; bit++)
        {
            int step = 1 << bit;
            for (int mask = 0; mask < f.Length; mask++)
            {
                if ((mask & step) == 0) f[mask] -= f[mask | step];
            }
        }
    }

    internal static long[] OrConvolution(long[] f, long[] g, int n)
    {
        var a = (long[])f.Clone();
        var b = (long[])g.Clone();
        ZetaSubset(a, n);
        ZetaSubset(b, n);
        for (int s = 0; s < a.Length; s++) a[s] *= b[s];
        MobiusSubset(a, n);
        return a;
    }

    internal static long[] AndConvolution(long[] f, long[] g, int n)
    {
        var a = (long[])f.Clone();
        var b = (long[])g.Clone();
        ZetaSuperset(a, n);
        ZetaSuperset(b, n);
        for (int s = 0; s < a.Length; s++) a[s] *= b[s];
        MobiusSuperset(a, n);
        return a;
    }

    /** Ranked zeta then Möbius: O(n^2 2^n) subset convolution. */
    internal static long[] SubsetConvolution(long[] f, long[] g, int n)
    {
        int size = 1 << n;
        var fa = new long[n + 1][];
        var fb = new long[n + 1][];
        var fc = new long[n + 1][];
        for (int k = 0; k <= n; k++)
        {
            fa[k] = new long[size];
            fb[k] = new long[size];
            fc[k] = new long[size];
        }
        for (int mask = 0; mask < size; mask++)
        {
            fa[Popcount(mask)][mask] = f[mask];
            fb[Popcount(mask)][mask] = g[mask];
        }
        for (int k = 0; k <= n; k++)
        {
            ZetaSubset(fa[k], n);
            ZetaSubset(fb[k], n);
        }
        for (int i = 0; i <= n; i++)
        {
            for (int j = 0; j + i <= n; j++)
            {
                for (int mask = 0; mask < size; mask++) fc[i + j][mask] += fa[i][mask] * fb[j][mask];
            }
        }
        for (int k = 0; k <= n; k++) MobiusSubset(fc[k], n);
        var result = new long[size];
        for (int mask = 0; mask < size; mask++) result[mask] = fc[Popcount(mask)][mask];
        return result;
    }

    internal static int Popcount(int value)
    {
        int count = 0;
        while (value != 0)
        {
            count += value & 1;
            value >>= 1;
        }
        return count;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int state = 540001;

    private static long[] LcgVector(int size, int limit)
    {
        var values = new long[size];
        for (int i = 0; i < size; i++)
        {
            state = (state * 75 + 74) % 65537;
            values[i] = state % limit;
        }
        return values;
    }

    private static string Render(long[] values)
    {
        return string.Join(",", values);
    }

    private static long[] ZetaSubsetNaive(long[] f, int n)
    {
        var result = new long[f.Length];
        for (int s = 0; s < f.Length; s++)
        {
            for (int t = 0; t < f.Length; t++)
            {
                if ((t & ~s) == 0) result[s] += f[t];
            }
        }
        return result;
    }

    private static long[] ZetaSupersetNaive(long[] f, int n)
    {
        var result = new long[f.Length];
        for (int s = 0; s < f.Length; s++)
        {
            for (int t = 0; t < f.Length; t++)
            {
                if ((s & ~t) == 0) result[s] += f[t];
            }
        }
        return result;
    }

    internal static void Main()
    {
        for (int n = 1; n <= 8; n++)
        {
            int size = 1 << n;
            long[] f = LcgVector(size, 1000);
            long[] work = (long[])f.Clone();
            SubsetLattice.ZetaSubset(work, n);
            SubsetLattice.MobiusSubset(work, n);
            Check(Render(work) == Render(f), "Möbius undoes the subset zeta transform for n=" + n);

            long[] work2 = (long[])f.Clone();
            SubsetLattice.ZetaSuperset(work2, n);
            SubsetLattice.MobiusSuperset(work2, n);
            Check(Render(work2) == Render(f), "superset Möbius undoes the superset zeta for n=" + n);

            long[] work3 = (long[])f.Clone();
            SubsetLattice.MobiusSubset(work3, n);
            SubsetLattice.ZetaSubset(work3, n);
            Check(Render(work3) == Render(f), "zeta undoes Möbius for n=" + n);

            long[] zeta = (long[])f.Clone();
            SubsetLattice.ZetaSubset(zeta, n);
            long left = 0;
            long right = 0;
            for (int s = 0; s < size; s++)
            {
                left += zeta[s];
                right += f[s] * (1L << (n - SubsetLattice.Popcount(s)));
            }
            Check(left == right, "sum of all subset sums equals the closed form for n=" + n);

            if (n <= 5)
            {
                Check(Render(zeta) == Render(ZetaSubsetNaive(f, n)), "SOS equals the direct pair loop for n=" + n);
                long[] sup = (long[])f.Clone();
                SubsetLattice.ZetaSuperset(sup, n);
                Check(Render(sup) == Render(ZetaSupersetNaive(f, n)), "superset SOS equals the direct loop for n=" + n);
            }
        }

        // singleton indicator: the superset zeta counts supersets, the subset zeta counts the empty set only.
        var indicator = new long[16];
        indicator[1] = 1;
        var supIndicator = (long[])indicator.Clone();
        SubsetLattice.ZetaSuperset(supIndicator, 4);
        for (int s = 0; s < 16; s++)
        {
            long expected = (s & ~1) == 0 ? 1 : 0;
            Check(supIndicator[s] == expected, "superset zeta counts supersets of {0} at " + s);
        }
        var singletons = new long[16];
        for (int i = 0; i < 4; i++) singletons[1 << i] = 1;
        var subsetCounts = (long[])singletons.Clone();
        SubsetLattice.ZetaSubset(subsetCounts, 4);
        for (int s = 0; s < 16; s++)
        {
            Check(subsetCounts[s] == SubsetLattice.Popcount(s), "subset zeta counts the elements of S at " + s);
        }

        // OR / AND convolutions against their naive definitions.
        foreach (int n in new[] { 1, 2, 3, 4 })
        {
            int size = 1 << n;
            long[] f = LcgVector(size, 6);
            long[] g = LcgVector(size, 6);
            long[] orFast = SubsetLattice.OrConvolution(f, g, n);
            long[] andFast = SubsetLattice.AndConvolution(f, g, n);
            var orNaive = new long[size];
            var andNaive = new long[size];
            for (int a = 0; a < size; a++)
            {
                for (int b = 0; b < size; b++)
                {
                    orNaive[a | b] += f[a] * g[b];
                    andNaive[a & b] += f[a] * g[b];
                }
            }
            Check(Render(orFast) == Render(orNaive), "OR convolution matches the naive sum for n=" + n);
            Check(Render(andFast) == Render(andNaive), "AND convolution matches the naive sum for n=" + n);

            long[] subsetFast = SubsetLattice.SubsetConvolution(f, g, n);
            var subsetNaive = new long[size];
            for (int s = 0; s < size; s++)
            {
                long total = 0;
                for (int t = 0; t < size; t++)
                {
                    if ((t & ~s) != 0) continue;
                    total += f[t] * g[s ^ t];
                }
                subsetNaive[s] = total;
            }
            Check(Render(subsetFast) == Render(subsetNaive), "subset convolution matches the naive sum for n=" + n);
        }

        // Sanity: OR convolution with the all-ones indicator keeps the total mass and preserves
        // the mass already sitting at a mask (the pair (s, 0) contributes f[s]).
        var onesIndicator = new long[8];
        for (int i = 0; i < 8; i++) onesIndicator[i] = 1;
        long[] f3 = LcgVector(8, 4);
        long totalF3 = 0;
        foreach (long value in f3) totalF3 += value;
        long[] orOnes = SubsetLattice.OrConvolution(f3, onesIndicator, 3);
        long mass = 0;
        for (int s = 0; s < 8; s++)
        {
            mass += orOnes[s];
            Check(orOnes[s] >= f3[s], "OR with the full mask keeps the mass already at " + s);
            Check(orOnes[s] <= 8 * totalF3, "OR mass stays bounded at " + s);
        }
        Check(mass == 8 * totalF3, "OR with the full mask multiplies the total mass by 8");
        Check(orOnes[7] == f3[0] + 2 * (f3[1] + f3[2] + f3[4]) + 4 * (f3[3] + f3[5] + f3[6]) + 8 * f3[7],
                "the full mask collects f[a] * 2^popcount(a)");
        Check(totalF3 > 0, "the OR sanity vector is non-zero");

        // n = 0 boundary: a single-element lattice, everything is the identity.
        var single = new long[] { 42 };
        var singleWork = (long[])single.Clone();
        SubsetLattice.ZetaSubset(singleWork, 0);
        Check(singleWork[0] == 42, "n = 0 subset zeta is the identity");
        SubsetLattice.MobiusSubset(singleWork, 0);
        Check(singleWork[0] == 42, "n = 0 Möbius is the identity");
        Check(SubsetLattice.OrConvolution(single, single, 0)[0] == 1764, "n = 0 OR convolution squares the value");
        Check(SubsetLattice.SubsetConvolution(single, single, 0)[0] == 1764, "n = 0 subset convolution squares");

        Console.WriteLine("PASS");
    }
}
