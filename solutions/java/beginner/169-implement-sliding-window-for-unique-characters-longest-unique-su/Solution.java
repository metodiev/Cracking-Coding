/** Task 169 - Sliding window for unique characters (longest unique substring). (Tier: Beginner, CTCI Ch 1) */
import java.util.*;

public class Solution {
    /** Returns {length, substring}. */
    static Object[] longestUniqueSubstring(String s) {
        Map<Character, Integer> last = new HashMap<>();
        int bestLen = 0, bestStart = 0, lo = 0;
        for (int hi = 0; hi < s.length(); hi++) {
            char c = s.charAt(hi);
            Integer seen = last.get(c);
            if (seen != null && seen >= lo) lo = seen + 1;
            last.put(c, hi);
            if (hi - lo + 1 > bestLen) {
                bestLen = hi - lo + 1;
                bestStart = lo;
            }
        }
        return new Object[]{bestLen, s.substring(bestStart, bestStart + bestLen)};
    }

    public static void main(String[] args) {
        check(checkPair("abcabcbb", 3, "abc"), "abcabcbb");
        check(checkPair("bbbbb", 1, "b"), "bbbbb");
        check(checkPair("pwwkew", 3, "wke"), "pwwkew");
        check(checkPair("", 0, ""), "empty");
        check(checkPair("dvdf", 3, "vdf"), "dvdf");
        check(checkPair("abba", 2, "ab"), "abba");
        System.out.println("PASS");
    }

    private static boolean checkPair(String input, int expectedLen, String expectedSub) {
        Object[] r = longestUniqueSubstring(input);
        return (int) r[0] == expectedLen && r[1].equals(expectedSub);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
