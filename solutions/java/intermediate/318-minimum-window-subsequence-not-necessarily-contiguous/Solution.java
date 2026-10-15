/** Task 318 - Minimum window subsequence (not necessarily contiguous). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    /** Shortest window of s containing t as a subsequence, "" if none exists. */
    static String minWindowSubsequence(String s, String t) {
        int n = s.length(), m = t.length();
        if (m == 0 || m > n) return "";
        int[] bestStart = new int[m];                        // bestStart[j] = latest start covering t[0..j]
        Arrays.fill(bestStart, -1);
        int bestLength = n + 1, begin = -1, end = -1;
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            for (int j = m - 1; j >= 0; j--) {               // descending keeps row i-1 intact
                if (ch == t.charAt(j)) bestStart[j] = j == 0 ? i : bestStart[j - 1];
            }
            if (bestStart[m - 1] != -1) {
                int start = bestStart[m - 1];
                if (i - start + 1 < bestLength) {            // strictly better keeps the leftmost window
                    bestLength = i - start + 1;
                    begin = start;
                    end = i;
                }
            }
        }
        return begin == -1 ? "" : s.substring(begin, end + 1);
    }

    static boolean isSubsequence(String needle, String haystack) {
        int k = 0;
        for (int i = 0; i < haystack.length() && k < needle.length(); i++) {
            if (haystack.charAt(i) == needle.charAt(k)) k++;
        }
        return k == needle.length();
    }

    /** Tries every window; the independent oracle. Returns {window} or null when none exists. */
    static String[] bruteMinWindow(String s, String t) {
        if (t.isEmpty() || t.length() > s.length()) return null;
        String best = null;
        for (int left = 0; left < s.length(); left++) {
            for (int right = left; right < s.length(); right++) {
                String window = s.substring(left, right + 1);
                if (isSubsequence(t, window) && (best == null || window.length() < best.length())) best = window;
            }
        }
        return best == null ? null : new String[]{best};
    }

    private static int seed = 3180;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(minWindowSubsequence("abcdebdde", "bde").equals("bcde"), "classic");     // [1..4] beats [5..8]
        check(minWindowSubsequence("abc", "abc").equals("abc"), "identity");
        check(minWindowSubsequence("abc", "d").isEmpty(), "absent");
        check(minWindowSubsequence("bab", "ab").equals("ab"), "bab");
        check(minWindowSubsequence("a", "a").equals("a"), "singles");
        check(minWindowSubsequence("a", "aa").isEmpty(), "too long");
        check(minWindowSubsequence("", "a").isEmpty(), "empty text");
        check(minWindowSubsequence("", "").isEmpty(), "both empty");
        check(minWindowSubsequence("ab", "").isEmpty(), "empty pattern");

        // Hand-verified absent case: "ezu" needs e < z < u, but every u (indices 41, 42) precedes
        // every z (indices 43, 44), and no u follows a z.
        String longText = "jmeqksfrsdcmsiwvaovxtkwmamwhvlbajfmbfmecwuuzzaqq";
        check(minWindowSubsequence(longText, "ezu").isEmpty(), "long absent");
        check(bruteMinWindow(longText, "ezu") == null, "long absent oracle");

        List<String> texts = new ArrayList<>(Arrays.asList("abcdebdde", "bab", "abc", "aab", "zzz"));
        for (int i = 0; i < 20; i++) texts.add(lcgString(30, "ab"));
        for (int i = 0; i < 10; i++) texts.add(lcgString(24, "abc"));

        List<String> patterns = new ArrayList<>(Arrays.asList("", "ab", "ba", "z", "cc", "bca"));
        for (int i = 0; i < 20; i++) patterns.add(lcgString(1, "abc"));
        for (int i = 0; i < 20; i++) patterns.add(lcgString(2, "abc"));
        for (int i = 0; i < 20; i++) patterns.add(lcgString(3, "abc"));

        for (String text : texts) {
            for (String pattern : patterns) {
                String window = minWindowSubsequence(text, pattern);
                String[] oracle = bruteMinWindow(text, pattern);
                if (oracle == null) {
                    check(window.isEmpty(), "none " + text + " / " + pattern);
                } else {
                    check(text.contains(window), "in text " + text + " / " + pattern);
                    check(isSubsequence(pattern, window), "subsequence " + text + " / " + pattern);
                    check(window.length() == oracle[0].length(), "length " + text + " / " + pattern);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
