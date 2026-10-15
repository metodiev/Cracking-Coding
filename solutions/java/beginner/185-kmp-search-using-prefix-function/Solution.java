/** Task 185 - KMP search using prefix function. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    /** Prefix function (LPS array) of `s`, as in task 184. */
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

    /**
     * All start indices of `pattern` in `text`, in increasing order (overlapping matches
     * included), in O(n + m). An empty pattern returns an empty list by convention:
     * "no match" avoids an unbounded list of positions.
     */
    static List<Integer> kmpSearch(String text, String pattern) {
        List<Integer> matches = new ArrayList<>();
        if (pattern.isEmpty()) return matches;
        int[] pi = prefixFunction(pattern);
        int j = 0;
        for (int i = 0; i < text.length(); i++) {
            while (j > 0 && text.charAt(i) != pattern.charAt(j)) j = pi[j - 1];
            if (text.charAt(i) == pattern.charAt(j)) j++;
            if (j == pattern.length()) {
                matches.add(i - pattern.length() + 1);
                j = pi[j - 1];                // keep going, matches may overlap
            }
        }
        return matches;
    }

    /** Reference scan used by the self-check. */
    static List<Integer> kmpSearchNaive(String text, String pattern) {
        List<Integer> matches = new ArrayList<>();
        if (pattern.isEmpty()) return matches;
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) matches.add(i);
        }
        return matches;
    }

    public static void main(String[] args) {
        check(kmpSearch("aaaaa", "aaaa").equals(Arrays.asList(0, 1)), "overlapping matches");
        check(kmpSearch("abababa", "aba").equals(Arrays.asList(0, 2, 4)), "abababa");
        check(kmpSearch("abcabcabc", "abc").equals(Arrays.asList(0, 3, 6)), "repeated block");
        check(kmpSearch("abracadabra", "cad").equals(Arrays.asList(4)), "single match");
        check(kmpSearch("mississippi", "issi").equals(Arrays.asList(1, 4)), "mississippi");
        check(kmpSearch("aaaa", "b").isEmpty(), "no match");
        check(kmpSearch("ab", "abc").isEmpty(), "pattern longer than text");
        check(kmpSearch("", "a").isEmpty(), "empty text");
        check(kmpSearch("", "").isEmpty(), "empty text and pattern");
        check(kmpSearch("anything", "").isEmpty(), "empty pattern");
        check(kmpSearch("a", "a").equals(Arrays.asList(0)), "single char");
        check(kmpSearch("aaa", "aa").equals(Arrays.asList(0, 1)), "aaa / aa");
        check(kmpSearch("aaa", "aaa").equals(Arrays.asList(0)), "exact match");

        // agreement with the naive scan on pseudo-random strings
        Random rng = new Random(13);
        for (int t = 0; t < 300; t++) {
            StringBuilder textBuilder = new StringBuilder();
            int textLength = rng.nextInt(31);
            for (int i = 0; i < textLength; i++) textBuilder.append((char) ('a' + rng.nextInt(2)));
            String text = textBuilder.toString();
            StringBuilder patternBuilder = new StringBuilder();
            int patternLength = 1 + rng.nextInt(5);
            for (int i = 0; i < patternLength; i++) patternBuilder.append((char) ('a' + rng.nextInt(2)));
            String pattern = patternBuilder.toString();
            check(kmpSearch(text, pattern).equals(kmpSearchNaive(text, pattern)), "cross-check " + text + " / " + pattern);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
