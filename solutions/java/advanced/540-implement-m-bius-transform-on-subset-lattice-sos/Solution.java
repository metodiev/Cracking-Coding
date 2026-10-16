/** Task 540 - Implement Möbius transform on subset lattice (SOS). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Subset-sum (SOS) dynamic programming over the subset lattice of {0..n-1}: the zeta transform
 * accumulates f[S] = sum over T subset of S of g[T] by looping over the n bit positions and
 * adding the value of the mask without that bit, giving O(n 2^n); the Möbius transform runs the
 * same loops with subtraction and inverts it exactly, so zeta followed by Möbius is the identity
 * on integer arrays. Superset variants (sum over T superset of S, and their inverse) are built by
 * replacing the "bit set" branch with the complementary branch.
 *
 * Three applications are implemented on top and each is checked against an independent brute
 * force: OR convolution h[S] = sum over T union U == S of f[T]g[U] (zeta, pointwise multiply,
 * Möbius), and ranked subset convolution h[S] = sum over T subset of S of f[T]g[S\T] (zeta each
 * rank slice, multiply the generating polynomials in the rank variable, Möbius, then keep the
 * rank-|S| slice). Documented checks: fast zeta equals the O(3^n) subset enumeration and Möbius
 * equals the inclusion-exclusion sum with signs (-1)^(|S|-|T|) for every mask, on LCG arrays up
 * to n = 8 (and n = 10 for zeta); the delta-function identities zeta(1_T)[S] = [T subset of S]
 * and zeta_superset(1_T)[S] = [S subset of T] pin the sign conventions; zeta(1)[S] = 2^|S| and
 * the mass identity sum_S zeta(f)[S] = sum_T f[T] 2^(n-|T|) are asserted; Möbius of the rank
 * function |S| is the indicator of the singleton masks; and both convolution formulas match
 * exhaustive enumeration for n <= 8, including indicator inputs.
 */
import java.util.*;

public class Solution {
    /** f[S] becomes sum over T subset of S of f[T]. */
    static long[] zetaSubset(long[] f, int n) {
        long[] a = f.clone();
        for (int bit = 0; bit < n; bit++) {
            for (int mask = 0; mask < a.length; mask++) {
                if ((mask & (1 << bit)) != 0) a[mask] += a[mask ^ (1 << bit)];
            }
        }
        return a;
    }

    /** Inverse of zetaSubset via the inclusion-exclusion recurrence. */
    static long[] mobiusSubset(long[] f, int n) {
        long[] a = f.clone();
        for (int bit = 0; bit < n; bit++) {
            for (int mask = 0; mask < a.length; mask++) {
                if ((mask & (1 << bit)) != 0) a[mask] -= a[mask ^ (1 << bit)];
            }
        }
        return a;
    }

    /** f[S] becomes sum over T superset of S of f[T]. */
    static long[] zetaSuperset(long[] f, int n) {
        long[] a = f.clone();
        for (int bit = 0; bit < n; bit++) {
            for (int mask = 0; mask < a.length; mask++) {
                if ((mask & (1 << bit)) == 0) a[mask] += a[mask | (1 << bit)];
            }
        }
        return a;
    }

    /** Inverse of zetaSuperset. */
    static long[] mobiusSuperset(long[] f, int n) {
        long[] a = f.clone();
        for (int bit = 0; bit < n; bit++) {
            for (int mask = 0; mask < a.length; mask++) {
                if ((mask & (1 << bit)) == 0) a[mask] -= a[mask | (1 << bit)];
            }
        }
        return a;
    }

    static long[] bruteZeta(long[] f, int n) {
        long[] out = new long[f.length];
        for (int s = 0; s < f.length; s++) {
            long total = 0;
            for (int t = s; ; t = (t - 1) & s) {
                total += f[t];
                if (t == 0) break;
            }
            out[s] = total;
        }
        return out;
    }

    static long[] bruteMobius(long[] f, int n) {
        long[] out = new long[f.length];
        for (int s = 0; s < f.length; s++) {
            long total = 0;
            for (int t = s; ; t = (t - 1) & s) {
                long sign = (Integer.bitCount(s) - Integer.bitCount(t)) % 2 == 0 ? 1 : -1;
                total += sign * f[t];
                if (t == 0) break;
            }
            out[s] = total;
        }
        return out;
    }

    static long[] bruteSupersetZeta(long[] f, int n) {
        int size = f.length;
        long[] out = new long[size];
        for (int s = 0; s < size; s++) {
            long total = 0;
            for (int t = 0; t < size; t++) {
                if ((t & s) == s) total += f[t];
            }
            out[s] = total;
        }
        return out;
    }

    static long[] orConvolution(long[] f, long[] g, int n) {
        long[] a = zetaSubset(f, n);
        long[] b = zetaSubset(g, n);
        for (int i = 0; i < a.length; i++) a[i] *= b[i];
        return mobiusSubset(a, n);
    }

    /** h[S] = sum over all T, U with T | U == S — overlapping pairs included. */
    static long[] bruteOrConvolution(long[] f, long[] g, int n) {
        long[] out = new long[f.length];
        for (int t = 0; t < f.length; t++) {
            for (int u = 0; u < g.length; u++) out[t | u] += f[t] * g[u];
        }
        return out;
    }

    /** Fast subset convolution via rank slices (zeta, rank-polynomial product, Möbius). */
    static long[] subsetConvolution(long[] f, long[] g, int n) {
        int size = 1 << n;
        long[][] byRankF = new long[n + 1][size];
        long[][] byRankG = new long[n + 1][size];
        long[][] product = new long[n + 1][size];
        for (int mask = 0; mask < size; mask++) {
            byRankF[Integer.bitCount(mask)][mask] = f[mask];
            byRankG[Integer.bitCount(mask)][mask] = g[mask];
        }
        for (int rank = 0; rank <= n; rank++) {
            byRankF[rank] = zetaSubset(byRankF[rank], n);
            byRankG[rank] = zetaSubset(byRankG[rank], n);
        }
        for (int mask = 0; mask < size; mask++) {
            for (int rank = 0; rank <= n; rank++) {
                long total = 0;
                for (int lower = 0; lower <= rank; lower++) {
                    total += byRankF[lower][mask] * byRankG[rank - lower][mask];
                }
                product[rank][mask] = total;
            }
        }
        for (int rank = 0; rank <= n; rank++) product[rank] = mobiusSubset(product[rank], n);
        long[] out = new long[size];
        for (int mask = 0; mask < size; mask++) out[mask] = product[Integer.bitCount(mask)][mask];
        return out;
    }

    /** h[S] = sum over *disjoint* T, U with T | U == S, enumerated over all mask pairs. */
    static long[] bruteSubsetConvolution(long[] f, long[] g, int n) {
        long[] out = new long[f.length];
        for (int t = 0; t < f.length; t++) {
            for (int u = 0; u < g.length; u++) {
                if ((t & u) == 0) out[t | u] += f[t] * g[u];
            }
        }
        return out;
    }

    static long[] lcgVector(int size, int seed) {
        long[] out = new long[size];
        for (int i = 0; i < size; i++) {
            seed = (seed * 75 + 74) % 65537;
            out[i] = seed % 9 - 4;
        }
        return out;
    }

    public static void main(String[] args) {
        check(zetaSubset(new long[]{7}, 0)[0] == 7 && mobiusSubset(new long[]{7}, 0)[0] == 7,
                "n = 0: the lattice is a single mask");
        check(Arrays.equals(zetaSubset(new long[]{3, 5}, 1), new long[]{3, 8}), "n = 1 zeta");
        check(Arrays.equals(mobiusSubset(new long[]{3, 8}, 1), new long[]{3, 5}), "n = 1 Möbius");

        int[] sizes = {4, 8, 16, 64, 256};
        int[] seeds = {7, 19, 31, 43, 55};
        for (int s = 0; s < sizes.length; s++) {
            int n = Integer.numberOfTrailingZeros(sizes[s]);
            long[] f = lcgVector(sizes[s], seeds[s]);
            long[] zeta = zetaSubset(f, n);
            check(Arrays.equals(zeta, bruteZeta(f, n)), "n = " + n + ": fast zeta equals subset enumeration");
            check(Arrays.equals(mobiusSubset(zeta, n), f), "n = " + n + ": Möbius inverts zeta exactly");
            check(Arrays.equals(mobiusSubset(f, n), bruteMobius(f, n)),
                    "n = " + n + ": Möbius equals inclusion-exclusion");
            check(Arrays.equals(zetaSuperset(f, n), bruteSupersetZeta(f, n)),
                    "n = " + n + ": superset zeta matches brute force");
            check(Arrays.equals(mobiusSuperset(zetaSuperset(f, n), n), f),
                    "n = " + n + ": superset Möbius inverts");

            long[] one = new long[sizes[s]];
            Arrays.fill(one, 1);
            long[] zetaOne = zetaSubset(one, n);
            for (int mask = 0; mask < sizes[s]; mask++) {
                check(zetaOne[mask] == (1L << Integer.bitCount(mask)),
                        "zeta(1)[S] = 2^|S| at mask " + mask);
            }
            long mass = 0;
            long weighted = 0;
            for (int mask = 0; mask < sizes[s]; mask++) {
                mass += zeta[mask];
                weighted += f[mask] * (1L << (n - Integer.bitCount(mask)));
            }
            check(mass == weighted, "mass identity sum zeta = sum f * 2^(n-|T|)");

            long[] rank = new long[sizes[s]];
            for (int mask = 0; mask < sizes[s]; mask++) rank[mask] = Integer.bitCount(mask);
            long[] rankMobius = mobiusSubset(rank, n);
            for (int mask = 0; mask < sizes[s]; mask++) {
                check(rankMobius[mask] == (Integer.bitCount(mask) == 1 ? 1 : 0),
                        "Möbius of the rank function is the singleton indicator at " + mask);
            }

            long[] g = lcgVector(sizes[s], seeds[s] + 500);
            check(Arrays.equals(orConvolution(f, g, n), bruteOrConvolution(f, g, n)),
                    "n = " + n + ": OR convolution matches the O(3^n) definition");
            check(Arrays.equals(subsetConvolution(f, g, n), bruteSubsetConvolution(f, g, n)),
                    "n = " + n + ": subset convolution matches the O(3^n) definition");
        }

        long[] oddSizes = lcgVector(1024, 91);
        check(Arrays.equals(zetaSubset(oddSizes, 10), bruteZeta(oddSizes, 10)), "n = 10 zeta");
        check(Arrays.equals(mobiusSubset(zetaSubset(oddSizes, 10), 10), oddSizes), "n = 10 round trip");

        long[] delta = new long[8];
        delta[3] = 5;
        long[] deltaZeta = zetaSubset(delta, 3);
        long[] deltaSuperset = zetaSuperset(delta, 3);
        for (int s = 0; s < 8; s++) {
            check(deltaZeta[s] == ((s & 3) == 3 ? 5 : 0), "zeta(delta_T)[S] = [T subset of S]");
            check(deltaSuperset[s] == ((s | 3) == 3 ? 5 : 0), "zeta_superset(delta_T)[S] = [S subset of T]");
        }

        long[] empty = new long[16];
        check(Arrays.equals(orConvolution(empty, lcgVector(16, 3), 4), empty), "zero annihilates OR");
        long[] full = new long[16];
        Arrays.fill(full, 1);
        long[] fullOr = orConvolution(full, full, 4);
        for (int s = 0; s < 16; s++) {
            long expected = 1;
            for (int k = 0; k < Integer.bitCount(s); k++) expected *= 3;
            check(fullOr[s] == expected, "OR of the full lattice is 3^|S| at mask " + s);
        }

        long[] setA = new long[8];
        setA[0] = 1;
        setA[3] = 1;
        long[] subsetConvIndicators = subsetConvolution(setA, setA, 3);
        for (int s = 0; s < 8; s++) {
            int ways = 0;
            for (int t = s; ; t = (t - 1) & s) {
                ways += (int) (setA[t] * setA[s ^ t]);
                if (t == 0) break;
            }
            check(subsetConvIndicators[s] == ways, "indicator subset convolution at " + s);
        }
        check(subsetConvIndicators[3] == 2 && subsetConvIndicators[0] == 1,
                "hand-checked: {0,3} splits as ({}, {0,3}) and ({0,3}, {}) only, {} splits once");
        check(subsetConvIndicators[7] == 0, "mask {0,1,2} cannot be split into two subsets of {0,3}");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
