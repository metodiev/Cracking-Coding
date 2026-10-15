/** Task 184 - Implement prefix function (KMP) for pattern matching. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    /**
     * pi[i] is the length of the longest proper prefix of s[:i+1] that is also a suffix of
     * s[:i+1]. The fallback loop runs in O(n) overall because j only ever moves back via
     * earlier prefix-function values.
     */
    static int[] prefixFunction(String s) {
        int n = s.length();
        int[] pi = new int[n];
        for (int i = 1; i < n; i++) {
            int j = pi[i - 1];
            while (j > 0 && s.charAt(i) != s.charAt(j)) j = pi[j - 1];
            if (s.charAt(i) == s.charAt(j)) j++;
            pi[i] = j;
        }
        return pi;
    }

    /** O(n^2) reference: longest proper border of every prefix, brute-forced. */
    static int[] prefixFunctionNaive(String s) {
        int n = s.length();
        int[] pi = new int[n];
        for (int end = 0; end < n; end++) {
            int best = 0;
            for (int length = 1; length <= end; length++) {
                if (s.regionMatches(0, s, end - length + 1, length)) best = length;
            }
            pi[end] = best;
        }
        return pi;
    }

    public static void main(String[] args) {
        check(Arrays.equals(prefixFunction(""), new int[] {}), "empty string");
        check(Arrays.equals(prefixFunction("a"), new int[] {0}), "single char");
        check(Arrays.equals(prefixFunction("ab"), new int[] {0, 0}), "no border");
        check(Arrays.equals(prefixFunction("ababaca"), new int[] {0, 0, 1, 2, 3, 0, 1}), "ababaca");
        check(Arrays.equals(prefixFunction("aabaaab"), new int[] {0, 1, 0, 1, 2, 2, 3}), "aabaaab");
        check(Arrays.equals(prefixFunction("aaaa"), new int[] {0, 1, 2, 3}), "aaaa");
        check(Arrays.equals(prefixFunction("abcabcd"), new int[] {0, 0, 0, 1, 2, 3, 0}), "abcabcd");
        check(Arrays.equals(prefixFunction("abababab"), new int[] {0, 0, 1, 2, 3, 4, 5, 6}), "abababab");
        check(Arrays.equals(prefixFunction("aabaaabaaab"), new int[] {0, 1, 0, 1, 2, 2, 3, 4, 5, 6, 7}), "aabaaabaaab");

        // agreement with the plain definition on pseudo-random strings
        Random rng = new Random(5);
        for (int t = 0; t < 200; t++) {
            int length = rng.nextInt(41);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++) sb.append((char) ('a' + rng.nextInt(2)));
            String s = sb.toString();
            check(Arrays.equals(prefixFunction(s), prefixFunctionNaive(s)), "cross-check " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
