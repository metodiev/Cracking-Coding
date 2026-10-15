/** Task 313 - The Z-function for pattern matching. (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static final char SENTINEL = '#';

    /** z[i] = longest common prefix of s and s.substring(i); z[0] = 0 by convention. */
    static int[] zFunction(String s) {
        int n = s.length();
        int[] z = new int[n];
        int left = 0, right = 0;
        for (int i = 1; i < n; i++) {
            if (i < right) z[i] = Math.min(right - i, z[i - left]);
            while (i + z[i] < n && s.charAt(z[i]) == s.charAt(i + z[i])) z[i]++;
            if (i + z[i] > right) {
                left = i;
                right = i + z[i];
            }
        }
        return z;
    }

    /** Direct O(n^2) definition; the independent oracle. */
    static int[] zFunctionNaive(String s) {
        int n = s.length();
        int[] z = new int[n];
        for (int i = 1; i < n; i++) {
            int h = 0;
            while (i + h < n && s.charAt(h) == s.charAt(i + h)) h++;
            z[i] = h;
        }
        return z;
    }

    /** All start indices of pattern in text via the Z-function on pattern + "#" + text. */
    static int[] zSearch(String text, String pattern) {
        int m = pattern.length();
        if (m == 0) {
            int[] all = new int[text.length() + 1];
            for (int i = 0; i <= text.length(); i++) all[i] = i;
            return all;
        }
        String combined = pattern + SENTINEL + text;
        int[] z = zFunction(combined);
        List<Integer> result = new ArrayList<>();
        for (int i = m + 1; i < combined.length(); i++) {
            if (z[i] >= m) result.add(i - m - 1);
        }
        int[] out = new int[result.size()];
        for (int i = 0; i < out.length; i++) out[i] = result.get(i);
        return out;
    }

    /** Independent oracle: try every position. */
    static int[] naiveSearch(String text, String pattern) {
        if (pattern.isEmpty()) {
            int[] all = new int[text.length() + 1];
            for (int i = 0; i <= text.length(); i++) all[i] = i;
            return all;
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) result.add(i);
        }
        int[] out = new int[result.size()];
        for (int i = 0; i < out.length; i++) out[i] = result.get(i);
        return out;
    }

    private static int seed = 3130;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // Worked example, values checked by hand
        check(Arrays.equals(zFunction("aabcaabxaaaz"), new int[]{0, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0}), "worked example");
        check(Arrays.equals(zFunction("aaaa"), new int[]{0, 3, 2, 1}), "aaaa");
        check(Arrays.equals(zFunction("abab"), new int[]{0, 0, 2, 0}), "abab");
        check(Arrays.equals(zFunction("a"), new int[]{0}), "single");
        check(zFunction("").length == 0, "empty");

        String text = "aabcaabxaaaz";
        check(Arrays.equals(zSearch(text, "aab"), new int[]{0, 4}), "aab");
        check(Arrays.equals(zSearch(text, "aa"), new int[]{0, 4, 8, 9}), "aa");
        check(Arrays.equals(zSearch(text, "az"), new int[]{10}), "az");
        check(Arrays.equals(zSearch(text, "x"), new int[]{7}), "x");
        check(zSearch(text, "zz").length == 0, "zz absent");
        check(Arrays.equals(zSearch(text, "bcaab"), new int[]{2}), "bcaab");
        check(Arrays.equals(zSearch(text, text), new int[]{0}), "self");
        check(Arrays.equals(zSearch("abc", ""), new int[]{0, 1, 2, 3}), "empty pattern");
        check(zSearch("", "a").length == 0, "empty text");

        List<String> texts = new ArrayList<>(Arrays.asList("a", "aa", "aaaa", "abab", "aabcaabxaaaz", "mississippi"));
        for (int i = 0; i < 30; i++) texts.add(lcgString(24, "ab"));
        for (int i = 0; i < 12; i++) texts.add(lcgString(18, "abc"));

        for (String s : texts) {
            int[] z = zFunction(s);
            check(Arrays.equals(z, zFunctionNaive(s)), "z mismatch " + s);
            for (int i = 0; i < s.length(); i++) check(z[i] >= 0 && z[i] <= s.length() - i, "z bounds " + s + " " + i);
            List<String> patterns = new ArrayList<>();
            for (int i = 0; i < s.length(); i++) {
                for (int extra = 1; extra <= 3; extra++) {
                    if (i + extra <= s.length()) patterns.add(s.substring(i, i + extra));
                }
            }
            patterns.addAll(Arrays.asList("zz", "ababab", ""));
            for (String pattern : patterns) {
                check(Arrays.equals(zSearch(s, pattern), naiveSearch(s, pattern)), "search " + s + " / " + pattern);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
