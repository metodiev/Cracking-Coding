/** Task 319 - Sliding window with variable constraints (two pointers, advanced). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    /** Length of the longest substring with at most k distinct characters (0 if k <= 0). */
    static int longestAtMostKDistinct(String s, int k) {
        if (k <= 0) return 0;
        Map<Character, Integer> counts = new HashMap<>();
        int left = 0, best = 0;
        for (int right = 0; right < s.length(); right++) {
            char ch = s.charAt(right);
            counts.merge(ch, 1, Integer::sum);
            while (counts.size() > k) {
                char out = s.charAt(left++);
                counts.merge(out, -1, Integer::sum);
                if (counts.get(out) == 0) counts.remove(out);
            }
            best = Math.max(best, right - left + 1);
        }
        return best;
    }

    /** Length of the longest substring obtainable by replacing at most k characters. */
    static int characterReplacement(String s, int k) {
        Map<Character, Integer> counts = new HashMap<>();
        int left = 0, best = 0, maxCount = 0;
        for (int right = 0; right < s.length(); right++) {
            char ch = s.charAt(right);
            counts.merge(ch, 1, Integer::sum);
            maxCount = Math.max(maxCount, counts.get(ch));
            while (right - left + 1 - maxCount > k) {
                char out = s.charAt(left++);
                counts.merge(out, -1, Integer::sum);
                maxCount = Collections.max(counts.values());        // recomputed after every shrink
            }
            best = Math.max(best, right - left + 1);
        }
        return best;
    }

    /** Try every window; the independent oracle for the distinct-characters variant. */
    static int bruteAtMostKDistinct(String s, int k) {
        int best = 0;
        for (int i = 0; i < s.length(); i++) {
            Set<Character> seen = new HashSet<>();
            for (int j = i; j < s.length(); j++) {
                seen.add(s.charAt(j));
                if (seen.size() > k) break;
                best = Math.max(best, j - i + 1);
            }
        }
        return best;
    }

    /** Try every window and count its maximal frequency; the independent oracle. */
    static int bruteCharacterReplacement(String s, int k) {
        int best = 0;
        for (int i = 0; i < s.length(); i++) {
            Map<Character, Integer> counts = new HashMap<>();
            for (int j = i; j < s.length(); j++) {
                counts.merge(s.charAt(j), 1, Integer::sum);
                if (j - i + 1 - Collections.max(counts.values()) <= k) best = Math.max(best, j - i + 1);
            }
        }
        return best;
    }

    private static int seed = 3190;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(longestAtMostKDistinct("eceba", 2) == 3, "eceba k=2");              // "ece"
        check(longestAtMostKDistinct("aa", 1) == 2, "aa k=1");
        check(longestAtMostKDistinct("abcadcacacaca", 3) == 11, "13 chars k=3");  // "cadcacacaca"
        check(longestAtMostKDistinct("abcadcacacaca", 2) == 8, "13 chars k=2");   // "cacacaca"
        check(longestAtMostKDistinct("", 2) == 0, "empty");
        check(longestAtMostKDistinct("abc", 0) == 0, "k=0");
        check(longestAtMostKDistinct("abc", 5) == 3, "k large");
        check(longestAtMostKDistinct("abaccc", 1) == 3, "ccc");

        check(characterReplacement("ABAB", 2) == 4, "ABAB k=2");
        check(characterReplacement("AABABBA", 1) == 4, "AABABBA k=1");
        check(characterReplacement("AAAA", 2) == 4, "AAAA");
        check(characterReplacement("ABCDE", 0) == 1, "ABCDE k=0");
        check(characterReplacement("", 0) == 0, "empty");
        check(characterReplacement("A", 0) == 1, "single");
        check(characterReplacement("ABAB", 0) == 1, "ABAB k=0");

        List<String> samples = new ArrayList<>(Arrays.asList("eceba", "aa", "abcadcacacaca"));
        for (int i = 0; i < 30; i++) samples.add(lcgString(20, "abc"));
        for (int i = 0; i < 15; i++) samples.add(lcgString(14, "ab"));

        for (String s : samples) {
            for (int k = 0; k < 4; k++) {
                check(longestAtMostKDistinct(s, k) == bruteAtMostKDistinct(s, k), "distinct " + s + " " + k);
                check(characterReplacement(s, k) == bruteCharacterReplacement(s, k), "replace " + s + " " + k);
            }
            check(longestAtMostKDistinct(s, 4) == s.length(), "all distinct " + s);
            for (int k = 0; k < 3; k++) {                                       // monotone in k
                check(longestAtMostKDistinct(s, k) <= longestAtMostKDistinct(s, k + 1), "monotone distinct");
                check(characterReplacement(s, k) <= characterReplacement(s, k + 1), "monotone replace");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
