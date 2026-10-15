/** Task 016 - Count occurrences of each character in a string. (Tier: Beginner, CTCI Ch 1) */
import java.util.LinkedHashMap;
import java.util.Map;

public class Solution {
    static Map<Character, Integer> charCounts(String s) {
        Map<Character, Integer> counts = new LinkedHashMap<>();
        for (char c : s.toCharArray())
            counts.merge(c, 1, Integer::sum);
        return counts;
    }

    public static void main(String[] args) {
        Map<Character, Integer> c = charCounts("aabbbc");
        check(c.get('a') == 2 && c.get('b') == 3 && c.get('c') == 1, "aabbbc");
        check(charCounts("").isEmpty(), "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
