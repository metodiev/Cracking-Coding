/** Task 379 - Binomial coefficients C(n, k) mod prime p via precomputed factorials. (Tier: Intermediate, CTCI Ch 6) */

/**
 * For a prime p = 1_000_000_007 factorials and inverse factorials are precomputed once:
 * invFact[n] = fact[n]^(p-2) mod p (Fermat), then downward invFact[i] = invFact[i+1]*(i+1).
 * Then C(n, k) = fact[n] * invFact[k] * invFact[n-k] mod p in O(1) per query, valid for
 * 0 <= k <= n < p (Lucas is not needed below p).
 *
 * Hand-verified assertions: C(10,3) = 120, C(20,10) = 184756, C(n,0) = C(n,n) = 1,
 * k > n -> 0 (the convention used everywhere in this repo), k < 0 -> 0.
 *
 * Independent verification paths:
 *   * a multiplicative loop C(n,k) = prod_{i=1..k} (n-k+i) * inv(i) mod p that never
 *     touches the factorial tables;
 *   * additive Pascal rows built by repeated addition mod p (no modular inverses at
 *     all) for n <= 140, cross-checked on LCG (n, k) pairs;
 *   * the known constants C(100,50) mod p = 538992043 and C(1000,500) mod p =
 *     159835829, each obtained independently (multiplicative loop + big-integer
 *     reduction of the exact value) and asserted here.
 *
 * The precomputation and all queries use exact long modular arithmetic (products stay
 * below 1.1e18, well inside the signed 64-bit range) - no floating point anywhere.
 */

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/** C(n, k) mod MOD with precomputed tables. */
class Binomial {
    final long[] fact;
    final long[] invFact;
    final int limit;

    Binomial(int limit) {
        this.limit = limit;
        fact = new long[limit + 1];
        fact[0] = 1;
        for (int i = 1; i <= limit; i++) fact[i] = fact[i - 1] * i % Solution.MOD;
        invFact = new long[limit + 1];
        invFact[limit] = Solution.modPow(fact[limit], Solution.MOD - 2, Solution.MOD);
        for (int i = limit; i > 0; i--) invFact[i - 1] = invFact[i] * i % Solution.MOD;
    }

    long choose(int n, int k) {
        if (k < 0 || n < 0 || k > n) return 0;
        if (n > limit) throw new IllegalArgumentException("n above the precomputed limit");
        return fact[n] * invFact[k] % Solution.MOD * invFact[n - k] % Solution.MOD;
    }
}

public class Solution {
    static final long MOD = 1_000_000_007L;

    /** Fast modular exponentiation (binary). */
    static long modPow(long base, long exponent, long modulus) {
        long result = 1 % modulus;
        base %= modulus;
        while (exponent > 0) {
            if ((exponent & 1) == 1) result = result * base % modulus;
            base = base * base % modulus;
            exponent >>= 1;
        }
        return result;
    }

    /** Independent path: running product with per-step modular inverses. */
    static long chooseMultiplicative(int n, int k) {
        if (k < 0 || n < 0 || k > n) return 0;
        if (k > n - k) k = n - k;
        long result = 1;
        for (int i = 1; i <= k; i++) {
            result = result * ((n - k + i) % MOD) % MOD;
            result = result * modPow(i, MOD - 2, MOD) % MOD;
        }
        return result;
    }

    /** Additive Pascal row mod MOD - no division or inversion used at all. */
    static long[] pascalRow(int n) {
        long[] row = {1};
        for (int step = 0; step < n; step++) {
            long[] next = new long[row.length + 1];
            next[0] = 1;
            next[next.length - 1] = 1;
            for (int i = 1; i < row.length; i++) next[i] = (row[i - 1] + row[i]) % MOD;
            row = next;
        }
        return row;
    }

    public static void main(String[] args) {
        Binomial table = new Binomial(1000);

        check(table.choose(10, 3) == 120, "C(10,3) = 120");
        check(table.choose(20, 10) == 184756, "C(20,10) = 184756");
        check(table.choose(5, 0) == 1 && table.choose(5, 5) == 1, "C(n,0) = C(n,n) = 1");
        check(table.choose(0, 0) == 1, "C(0,0) = 1");
        check(table.choose(3, 5) == 0, "k > n -> 0");
        check(table.choose(3, -1) == 0, "k < 0 -> 0");
        check(table.choose(0, 1) == 0, "C(0,1) = 0");

        // symmetry C(n, k) == C(n, n-k)
        int[] ns = {1, 2, 7, 40, 999, 1000};
        for (int n : ns) {
            for (int k : new int[]{0, 1, n / 2, n - 1, n}) {
                check(table.choose(n, k) == table.choose(n, n - k), "symmetry " + n + " " + k);
            }
        }

        // real (non-modular) checks where the exact value is known
        check(chooseMultiplicative(10, 3) == 120, "multiplicative C(10,3)");
        check(chooseMultiplicative(20, 10) == 184756, "multiplicative C(20,10)");
        check(chooseMultiplicative(52, 5) == 2598960, "poker hands C(52,5)");

        // known constants, each verified by the independent multiplicative path
        check(table.choose(100, 50) == 538992043L, "C(100,50) mod p");
        check(chooseMultiplicative(100, 50) == 538992043L, "multiplicative C(100,50)");
        check(table.choose(1000, 500) == 159835829L, "C(1000,500) mod p");
        check(chooseMultiplicative(1000, 500) == 159835829L, "multiplicative C(1000,500)");

        // additive Pascal rows: completely different arithmetic path for smaller n
        Lcg lcg = new Lcg(37913);
        for (int trial = 0; trial < 60; trial++) {
            int n = lcg.next(141);
            int k = lcg.next(n + 2) - 1;              // includes -1, 0, n and n+1
            long expected = (k < 0 || k > n) ? 0 : pascalRow(n)[k];
            check(table.choose(n, k) == expected, "pascal cross-check " + n + " " + k);
            check(chooseMultiplicative(n, k) == expected, "pascal multiplicative " + n + " " + k);
        }
        for (int n : new int[]{1, 2, 3, 10, 50, 140}) {
            long[] row = pascalRow(n);
            for (int k = 0; k <= n; k++) {
                check(table.choose(n, k) == row[k], "full pascal row " + n + " " + k);
                check(chooseMultiplicative(n, k) == row[k], "full pascal multiplicative " + n + " " + k);
            }
        }

        // Pascal identity C(n, k) = C(n-1, k-1) + C(n-1, k) on the modular tables
        for (int trial = 0; trial < 40; trial++) {
            int n = lcg.next(1001);
            int k = lcg.next(n + 1);
            if (k < 1 || k > n) continue;
            check(table.choose(n, k) == (table.choose(n - 1, k - 1) + table.choose(n - 1, k)) % MOD,
                    "pascal identity " + n + " " + k);
        }

        // binomial theorem sanity: sum_k C(n, k) = 2^n mod p
        for (int n : new int[]{5, 30, 200, 1000}) {
            long total = 0;
            for (int k = 0; k <= n; k++) total = (total + table.choose(n, k)) % MOD;
            check(total == modPow(2, n, MOD), "binomial theorem " + n);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
