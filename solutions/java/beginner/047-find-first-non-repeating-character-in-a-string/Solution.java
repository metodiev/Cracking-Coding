/** Task 047 - Find first non-repeating character in a string. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static Character firstNonRepeating(String s) {
        int[] counts = new int[128];
        for (char c : s.toCharArray()) counts[c]++;
        for (char c : s.toCharArray()) if (counts[c] == 1) return c;
        return null;
    }

    public static void main(String[] args) {
        check(firstNonRepeating("aabbcde") == 'c', "c");
        check(firstNonRepeating("aabb") == null, "none");
        check(firstNonRepeating("x") == 'x', "single");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
