/** Task 317 - Longest duplicate substring via binary search + rolling hash. (Tier: Intermediate, CTCI Ch 1) */
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
    /**
     * Simplest duplicate substring of maximal length, found at the smallest starting index; ""
     * is the documented sentinel meaning "no duplicate substring exists".
     */
    static String longestDuplicateSubstring(String s) {
        int n = s.length();
        if (n <= 1) return "";
        DoubleRollingHash rolling = new DoubleRollingHash(s);
        int lo = 1, hi = n - 1, best = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (hasDuplicate(rolling, n, mid)) {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        if (best == 0) return "";
        for (int i = 0; i + best <= n; i++) {
            for (int j = i + 1; j + best <= n; j++) {
                if (rolling.equal(i, i + best, j, j + best)) return s.substring(i, i + best);
            }
        }
        throw new AssertionError("binary search and extraction disagree");
    }

    static boolean hasDuplicate(DoubleRollingHash rolling, int n, int length) {
        Set<Long> seen = new HashSet<>();
        for (int i = 0; i + length <= n; i++) {
            long[] key = rolling.hash(i, i + length);
            long packed = key[0] * DoubleRollingHash.MOD2 + key[1];
            if (!seen.add(packed)) return true;
        }
        return false;
    }

    /** All-pairs LCP scan; the independent oracle for the best length. */
    static int longestDuplicateNaiveLength(String s) {
        int best = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j < s.length(); j++) {
                int h = 0;
                while (j + h < s.length() && s.charAt(i + h) == s.charAt(j + h)) h++;
                best = Math.max(best, h);
            }
        }
        return best;
    }

    /** Occurrences counted with overlaps (non-overlapping counting would undercount repeats). */
    static int overlappingCount(String s, String word) {
        int count = 0;
        for (int i = 0; i + word.length() <= s.length(); i++) {
            if (s.startsWith(word, i)) count++;
        }
        return count;
    }

    private static int seed = 3170;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(longestDuplicateSubstring("banana").equals("ana"), "banana");          // "ana" at 1 and 3
        check(longestDuplicateSubstring("aaaa").equals("aaa"), "aaaa");              // "aaa" at 0 and 1
        check(longestDuplicateSubstring("abcd").isEmpty(), "abcd");                  // nothing repeats
        check(longestDuplicateSubstring("").isEmpty(), "empty");
        check(longestDuplicateSubstring("a").isEmpty(), "single");
        check(longestDuplicateSubstring("aa").equals("a"), "aa");
        check(longestDuplicateSubstring("abab").equals("ab"), "abab");
        check(longestDuplicateSubstring("abcabcabc").equals("abcabc"), "abcabcabc"); // at 0 and 3
        check(longestDuplicateSubstring("mississippi").equals("issi"), "mississippi"); // at 1 and 4

        List<String> samples = new ArrayList<>(Arrays.asList("banana", "aaaa", "abcd", "abab", "abcabcabc",
                "mississippi", "aabcaabxaaaz"));
        for (int i = 0; i < 30; i++) samples.add(lcgString(24, "ab"));
        for (int i = 0; i < 15; i++) samples.add(lcgString(18, "abc"));

        for (String s : samples) {
            String result = longestDuplicateSubstring(s);
            int naive = longestDuplicateNaiveLength(s);
            check(result.length() == naive, "length " + s);
            if (naive == 0) {
                check(result.isEmpty(), "empty " + s);
            } else {
                check(overlappingCount(s, result) > 1, "duplicate " + s);           // a real duplicate ...
                check(result.equals(longestDuplicateSubstring(s)), "deterministic " + s);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
