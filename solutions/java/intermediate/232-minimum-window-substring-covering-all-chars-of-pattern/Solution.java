/** Task 232 - Minimum window substring covering all chars of pattern. (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    /**
     * Sliding window with a need/missing counter: grow with `right`; when the window
     * stops missing characters, shrink from the left while still valid, keeping the
     * smallest window seen. O(|s| + |t|) time. Leftmost window wins ties; "" if none.
     */
    static String minWindow(String s, String t) {
        if (s.isEmpty() || t.isEmpty() || t.length() > s.length()) return "";
        int[] need = new int[128];
        for (int i = 0; i < t.length(); i++) need[t.charAt(i)]++;
        int missing = t.length();
        int left = 0, bestStart = 0, bestLen = Integer.MAX_VALUE;
        for (int right = 0; right < s.length(); right++) {
            char ch = s.charAt(right);
            if (need[ch] > 0) missing--;
            need[ch]--;
            while (missing == 0) {
                if (right - left + 1 < bestLen) {   // strict: ties keep the leftmost window
                    bestStart = left;
                    bestLen = right - left + 1;
                }
                char lc = s.charAt(left);
                need[lc]++;
                if (need[lc] > 0) missing++;
                left++;
            }
        }
        return bestLen == Integer.MAX_VALUE ? "" : s.substring(bestStart, bestStart + bestLen);
    }

    /** O(n^2 * k) reference used only for cross-checks; same leftmost tie-break. */
    static String bruteMinWindow(String s, String t) {
        if (s.isEmpty() || t.isEmpty() || t.length() > s.length()) return "";
        int[] want = new int[128];
        for (int i = 0; i < t.length(); i++) want[t.charAt(i)]++;
        String best = "";
        for (int left = 0; left < s.length(); left++) {
            for (int right = left + t.length(); right <= s.length(); right++) {
                int[] have = new int[128];
                for (int i = left; i < right; i++) have[s.charAt(i)]++;
                boolean ok = true;
                for (int c = 0; c < 128 && ok; c++) if (have[c] < want[c]) ok = false;
                if (ok) {
                    String window = s.substring(left, right);
                    if (best.isEmpty() || window.length() < best.length()) best = window;
                    break;
                }
            }
        }
        return best;
    }

    public static void main(String[] args) {
        check(minWindow("ADOBECODEBANC", "ABC").equals("BANC"), "classic");
        check(minWindow("cabwefgewcwaefgcf", "cae").equals("cwae"), "second classic");
        check(minWindow("a", "a").equals("a"), "single char");
        check(minWindow("a", "aa").equals(""), "pattern longer than s");
        check(minWindow("abc", "abcd").equals(""), "pattern longer than s again");
        check(minWindow("ab", "cd").equals(""), "no window");
        check(minWindow("aa", "aa").equals("aa"), "duplicates in t");
        check(minWindow("abaab", "aa").equals("aa"), "duplicates in t again");
        check(minWindow("bba", "ab").equals("ba"), "shorter right side");
        check(minWindow("aabc", "aabc").equals("aabc"), "whole string");
        check(minWindow("abc", "").equals(""), "empty pattern");
        check(minWindow("", "a").equals(""), "empty s");

        java.util.Random rng = new java.util.Random(232);
        StringBuilder sb = new StringBuilder();
        for (int test = 0; test < 200; test++) {     // cross-check vs brute force
            int sLen = rng.nextInt(15);
            sb.setLength(0);
            for (int i = 0; i < sLen; i++) sb.append(rng.nextBoolean() ? 'a' : 'b');
            String s = sb.toString();
            int tLen = rng.nextInt(5);
            sb.setLength(0);
            for (int i = 0; i < tLen; i++) sb.append(rng.nextBoolean() ? 'a' : 'b');
            String t = sb.toString();
            check(minWindow(s, t).equals(bruteMinWindow(s, t)), "matches brute force");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
