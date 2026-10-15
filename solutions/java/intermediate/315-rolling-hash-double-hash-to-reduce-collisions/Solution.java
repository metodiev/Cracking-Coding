/** Task 315 - Rolling hash with two moduli to reduce collisions. (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

class DoubleRollingHash {
    static final long MOD1 = 1_000_000_007L;
    static final long MOD2 = 998_244_353L;
    static final long BASE = 131L;

    private final long[] h1;
    private final long[] h2;
    private final long[] p1;
    private final long[] p2;

    DoubleRollingHash(String s) {
        int n = s.length();
        h1 = new long[n + 1];
        h2 = new long[n + 1];
        p1 = new long[n + 1];
        p2 = new long[n + 1];
        p1[0] = 1;
        p2[0] = 1;
        for (int i = 0; i < n; i++) {
            long code = s.charAt(i);
            h1[i + 1] = (h1[i] * BASE + code) % MOD1;
            h2[i + 1] = (h2[i] * BASE + code) % MOD2;
            p1[i + 1] = p1[i] * BASE % MOD1;
            p2[i + 1] = p2[i] * BASE % MOD2;
        }
    }

    /** Hash of s[lo..hi) as the pair (mod MOD1, mod MOD2). */
    long[] hash(int lo, int hi) {
        long a = (h1[hi] - h1[lo] * p1[hi - lo] % MOD1 + MOD1) % MOD1;
        long b = (h2[hi] - h2[lo] * p2[hi - lo] % MOD2 + MOD2) % MOD2;
        return new long[]{a, b};
    }

    boolean equal(int lo1, int hi1, int lo2, int hi2) {
        return Arrays.equals(hash(lo1, hi1), hash(lo2, hi2));
    }
}

public class Solution {
    /** Rolling hash built from scratch, no prefix arrays; used as a cross-check. */
    static long[] directHash(String s) {
        long h1 = 0;
        long h2 = 0;
        for (int i = 0; i < s.length(); i++) {
            h1 = (h1 * DoubleRollingHash.BASE + s.charAt(i)) % DoubleRollingHash.MOD1;
            h2 = (h2 * DoubleRollingHash.BASE + s.charAt(i)) % DoubleRollingHash.MOD2;
        }
        return new long[]{h1, h2};
    }

    static final long MOD_WEAK = 65537L;
    static final long BASE_WEAK = 65536L;        // == -1 (mod 65537), a deliberately bad base

    /** Single-modulus hash with a base that is -1 modulo the modulus (alternating sum). */
    static long weakHash(String s) {
        long h = 0;
        for (int i = 0; i < s.length(); i++) h = (h * BASE_WEAK + s.charAt(i)) % MOD_WEAK;
        return h;
    }

    private static int seed = 3150;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check((DoubleRollingHash.MOD1 - 1) % 2 == 0 && (DoubleRollingHash.MOD2 - 1) % 2 == 0, "bases");
        check(DoubleRollingHash.MOD1 != DoubleRollingHash.MOD2, "distinct moduli");

        String s = "banana";
        DoubleRollingHash rolling = new DoubleRollingHash(s);
        check(Arrays.equals(rolling.hash(1, 3), directHash("an")), "an");
        check(Arrays.equals(rolling.hash(0, 2), directHash("ba")), "ba");
        check(rolling.equal(1, 3, 3, 5), "an == an");
        check(!rolling.equal(0, 2, 3, 5), "ba != an");
        check(rolling.equal(1, 6, 1, 6), "self equal");

        // Deliberate collision of a weak single-modulus hash, separated by the double hash
        check(weakHash("ab") == weakHash("bc") && weakHash("ab") == 1, "weak collision");  // 97*(-1)+98 == 98*(-1)+99
        check(weakHash("ab") != weakHash("ba"), "weak not universal");
        check(!Arrays.equals(new DoubleRollingHash("ab").hash(0, 2), new DoubleRollingHash("bc").hash(0, 2)), "double separates");
        check(!Arrays.equals(directHash("ab"), directHash("bc")), "double separates 2");
        check(directHash("ab")[0] != directHash("bc")[0], "single strong separator");

        // O(1) equality queries must agree with naive slicing, systematically and randomly
        List<String> texts = new ArrayList<>(Arrays.asList("a", "banana", "mississippi", "aaaaaaaa"));
        for (int i = 0; i < 20; i++) texts.add(lcgString(12, "abca"));
        for (int i = 0; i < 10; i++) texts.add(lcgString(16, "abcd"));

        long checked = 0;
        for (String text : texts) {
            DoubleRollingHash hashes = new DoubleRollingHash(text);
            int n = text.length();
            for (int length = 1; length <= n; length++) {
                for (int lo1 = 0; lo1 + length <= n; lo1++) {
                    for (int lo2 = lo1; lo2 + length <= n; lo2++) {
                        boolean expected = text.substring(lo1, lo1 + length).equals(text.substring(lo2, lo2 + length));
                        check(hashes.equal(lo1, lo1 + length, lo2, lo2 + length) == expected, "equal " + text);
                        checked++;
                    }
                }
            }
            for (int lo = 0; lo <= n; lo++) {
                for (int hi = lo; hi <= n; hi++) {
                    check(Arrays.equals(hashes.hash(lo, hi), directHash(text.substring(lo, hi))), "hash " + text);
                }
            }
        }
        check(checked > 1000, "coverage");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
