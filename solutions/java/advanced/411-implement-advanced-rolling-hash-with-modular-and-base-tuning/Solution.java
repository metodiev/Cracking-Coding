/** Task 411 - Implement advanced rolling hash with modular and base tuning. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Polynomial hashing: hash(s) = sum(s[i] * base^(n-1-i)) mod modulus. Tuning notes:
 *   * double hashing uses bases 131/137 with the large primes 1_000_000_007 / 1_000_000_009;
 *     with those sizes every cross product fits comfortably in a signed 64-bit long.
 *   * the demo pair modulus 65537 with base 2 is deliberately bad: 2^16 = -1 and 2^32 = 1
 *     (mod 65537), so same-length strings that only shift a character by 32 positions
 *     collide — that is the constructed single-hash collision asserted below.
 *   * base 1 degenerates the hash to the plain character sum, so anagrams collide.
 *   * prefix hashes make every substring fingerprint O(1); MultiHash combines several
 *     (base, modulus) pairs so an accidental equality must beat all of them at once.
 */
class RollingHash {
    final long base;
    final long mod;
    private final long[] pref;
    private final long[] pow;

    RollingHash(String s, long base, long mod) {
        this.base = base;
        this.mod = mod;
        pref = new long[s.length() + 1];
        for (int i = 0; i < s.length(); i++) {
            pref[i + 1] = (pref[i] * base + s.charAt(i)) % mod;
        }
        pow = new long[s.length() + 1];
        pow[0] = 1;
        for (int i = 1; i <= s.length(); i++) {
            pow[i] = pow[i - 1] * base % mod;
        }
    }

    /** Fingerprint of s[l:r]. */
    long get(int l, int r) {
        return ((pref[r] - pref[l] * pow[r - l]) % mod + mod) % mod;
    }
}

class MultiHash {
    private final RollingHash[] hashes;

    MultiHash(String s, long[] bases, long[] mods) {
        hashes = new RollingHash[bases.length];
        for (int i = 0; i < bases.length; i++) hashes[i] = new RollingHash(s, bases[i], mods[i]);
    }

    long[] fingerprint(int l, int r) {
        long[] out = new long[hashes.length];
        for (int i = 0; i < hashes.length; i++) out[i] = hashes[i].get(l, r);
        return out;
    }

    boolean equal(int l1, int r1, int l2, int r2) {
        if (r1 - l1 != r2 - l2) return false;
        long[] a = fingerprint(l1, r1);
        long[] b = fingerprint(l2, r2);
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}

public class Solution {
    static final long MOD1 = 1_000_000_007L;
    static final long MOD2 = 1_000_000_009L;
    static final long BASE1 = 131;
    static final long BASE2 = 137;
    static final long DEMO_MOD = 65537;
    static final long DEMO_BASE = 2;

    static long polyHash(String s, long base, long mod) {
        long h = 0;
        for (int i = 0; i < s.length(); i++) {
            h = (h * base + s.charAt(i)) % mod;
        }
        return h;
    }

    private static int seed = 2024;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // hand values: "ab" = 97*131 + 98, "abc" = 97*131^2 + 98*131 + 99
        check(polyHash("ab", BASE1, MOD1) == 12805, "hand ab");
        check(polyHash("abc", BASE1, MOD1) == 1677554, "hand abc");

        RollingHash h = new RollingHash("abc", BASE1, MOD1);
        check(h.get(0, 3) == 1677554, "prefix whole");
        check(h.get(0, 1) == 97, "prefix a");
        check(h.get(1, 2) == 98, "prefix b");
        check(h.get(2, 3) == 99, "prefix c");
        check(h.get(1, 3) == polyHash("bc", BASE1, MOD1), "prefix bc");
        check(h.get(0, 0) == 0 && h.get(3, 3) == 0, "empty substring");

        // constructed single-hash collision: base 2, modulus 65537, length 33,
        // characters 32 positions apart swap places — weights repeat because 2^32 = 1.
        StringBuilder sb = new StringBuilder("a"), tb = new StringBuilder("b");
        for (int i = 0; i < 31; i++) { sb.append('c'); tb.append('c'); }
        sb.append('b');
        tb.append('a');
        String s = sb.toString(), t = tb.toString();
        check(!s.equals(t) && s.length() == 33 && t.length() == 33, "collision strings differ");
        check(polyHash(s, DEMO_BASE, DEMO_MOD) == polyHash(t, DEMO_BASE, DEMO_MOD), "single-hash collision");
        check(new RollingHash(s, DEMO_BASE, DEMO_MOD).get(0, 33)
                == new RollingHash(t, DEMO_BASE, DEMO_MOD).get(0, 33), "rolling collision");

        // double hashing with tuned large primes separates them again
        long[] bases = {BASE1, BASE2};
        long[] mods = {MOD1, MOD2};
        MultiHash dblS = new MultiHash(s, bases, mods);
        MultiHash dblT = new MultiHash(t, bases, mods);
        long[] fs = dblS.fingerprint(0, 33);
        long[] ft = dblT.fingerprint(0, 33);
        boolean same = true;
        for (int i = 0; i < fs.length; i++) if (fs[i] != ft[i]) same = false;
        check(!same, "double hash distinguishes");

        // base 1 degenerates to the character sum
        check(polyHash("ab", 1, MOD1) == polyHash("ba", 1, MOD1), "base 1 anagram collision");
        check(polyHash("ab", BASE1, MOD1) != polyHash("ba", BASE1, MOD1), "base 131 separates");
        check(polyHash("A", 131, 256) == 'A' % 256, "small modulus trims");

        for (int round = 0; round < 6; round++) {
            String text = lcgString(61, "abcde");
            MultiHash multi = new MultiHash(text, bases, mods);
            for (int i = 0; i < text.length(); i++) {
                for (int j = i; j <= text.length(); j++) {
                    check(multi.equal(i, j, i, j), "reflexive " + i + " " + j);
                }
            }
            for (int q = 0; q < 60; q++) {
                seed = (seed * 75 + 74) % 65537;
                int l1 = seed % text.length();
                seed = (seed * 75 + 74) % 65537;
                int l2 = seed % text.length();
                seed = (seed * 75 + 74) % 65537;
                int maxSpan = Math.min(text.length() - l1, text.length() - l2);
                int span = seed % (maxSpan + 1);
                boolean oracle = text.substring(l1, l1 + span).equals(text.substring(l2, l2 + span));
                check(multi.equal(l1, l1 + span, l2, l2 + span) == oracle, "oracle " + q);
            }
        }

        // every substring fingerprint equals the direct polynomial hash
        String miss = "mississippi";
        for (long base : bases) {
            for (long mod : mods) {
                RollingHash rh = new RollingHash(miss, base, mod);
                for (int l = 0; l <= miss.length(); l++) {
                    for (int r = l; r <= miss.length(); r++) {
                        check(rh.get(l, r) == polyHash(miss.substring(l, r), base, mod), "direct " + l + " " + r);
                    }
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
