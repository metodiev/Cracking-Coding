/** Task 381 - Chinese remainder theorem solver for pairwise coprime moduli. (Tier: Intermediate, CTCI Ch 6) */
import java.util.*;

public class Solution {
    /** Returns {g, x, y} with a*x + b*y == g == gcd(a, b). */
    static long[] extendedGcd(long a, long b) {
        if (b == 0) return new long[] { a, 1, 0 };
        long[] prev = extendedGcd(b, a % b);
        long g = prev[0], x = prev[1], y = prev[2];
        return new long[] { g, y, x - (a / b) * y };
    }

    static long modInverse(long a, long modulus) {
        long[] eg = extendedGcd(((a % modulus) + modulus) % modulus, modulus);
        if (eg[0] != 1) throw new IllegalArgumentException("modular inverse does not exist");
        return ((eg[1] % modulus) + modulus) % modulus;
    }

    /** Merge two congruences; throws IllegalArgumentException when moduli are not coprime. */
    static long[] crtTwo(long a1, long m1, long a2, long m2) {
        if (extendedGcd(m1, m2)[0] != 1)
            throw new IllegalArgumentException("moduli " + m1 + " and " + m2 + " are not coprime");
        long t = (((a2 - a1) % m2) * modInverse(m1, m2)) % m2;
        t = ((t % m2) + m2) % m2;
        long modulus = m1 * m2;
        return new long[] { ((a1 + m1 * t) % modulus + modulus) % modulus, modulus };
    }

    /**
     * Solve a system of congruences with pairwise coprime moduli.
     * Returns {x, m}: x the smallest non-negative solution, m the product of the moduli.
     * Throws IllegalArgumentException for an empty system or non-coprime moduli.
     */
    static long[] crt(long[][] congruences) {
        if (congruences.length == 0) throw new IllegalArgumentException("empty congruence system");
        long a = 0, m = 1;
        for (long[] c : congruences) {
            long residue = c[0], modulus = c[1];
            if (modulus <= 0) throw new IllegalArgumentException("modulus must be positive");
            long[] merged = crtTwo(a, m, ((residue % modulus) + modulus) % modulus, modulus);
            a = merged[0];
            m = merged[1];
        }
        return new long[] { a, m };
    }

    private static int seed = 4212;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        // classic worked example: 23 = 2 (3), 23 = 3 (5), 23 = 2 (7)
        check(Arrays.equals(crt(new long[][] { { 2, 3 }, { 3, 5 }, { 2, 7 } }), new long[] { 23, 105 }),
                "classic (23, 105)");
        long[] classic = crt(new long[][] { { 2, 3 }, { 3, 5 }, { 2, 7 } });
        check(classic[0] % 3 == 2 && classic[0] % 5 == 3 && classic[0] % 7 == 2, "substitution classic");
        check(classic[1] == 105, "modulus product");

        // single congruence is the identity; merge order must not matter
        check(Arrays.equals(crt(new long[][] { { 4, 9 } }), new long[] { 4, 9 }), "single congruence");
        check(Arrays.equals(crt(new long[][] { { 23, 105 } }), new long[] { 23, 105 }), "already combined");
        check(Arrays.equals(crt(new long[][] { { 2, 7 }, { 3, 5 }, { 2, 3 } }), new long[] { 23, 105 }),
                "merge order");
        check(Arrays.equals(crt(new long[][] { { 0, 2 }, { 1, 3 } }), new long[] { 4, 6 }), "small system");

        // non-coprime moduli are rejected (documented error), even when consistent
        long[][][] bad = {
            { { 1, 4 }, { 2, 6 } },
            { { 0, 2 }, { 0, 4 } },
            { { 1, 9 }, { 1, 3 } },
        };
        for (long[][] system : bad) {
            boolean threw = false;
            try {
                crt(system);
            } catch (IllegalArgumentException e) {
                threw = true;
            }
            check(threw, "non-coprime rejected");
        }

        // LCG-driven systems: cross-check by substitution and by brute force where feasible
        int[] primes = { 3, 5, 7, 11, 13, 17, 19, 23 };
        int checked = 0;
        for (int trial = 0; trial < 40; trial++) {
            int count = 2 + nextSeed() % 3;
            List<Integer> chosen = new ArrayList<>();
            while (chosen.size() < count) {
                int p = primes[nextSeed() % primes.length];
                if (!chosen.contains(p)) chosen.add(p);
            }
            long[][] system = new long[count][2];
            for (int i = 0; i < count; i++) {
                int p = chosen.get(i);
                system[i][0] = nextSeed() % p;
                system[i][1] = p;
            }
            long[] solved = crt(system);
            for (long[] c : system)
                check(solved[0] % c[1] == c[0], "substitution trial " + trial);
            long product = 1;
            for (long[] c : system) product *= c[1];
            check(solved[1] == product && solved[0] < solved[1], "product trial " + trial);
            if (solved[1] <= 100000) {
                long brute = 0;
                while (brute < solved[1]) {
                    boolean ok = true;
                    for (long[] c : system) if (brute % c[1] != c[0]) { ok = false; break; }
                    if (ok) break;
                    brute++;
                }
                check(brute == solved[0], "brute force trial " + trial);
                checked++;
            }
        }
        check(checked >= 10, "enough brute-force checks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
