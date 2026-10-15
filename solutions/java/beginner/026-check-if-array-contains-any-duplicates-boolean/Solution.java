/** Task 026 - Check if array contains any duplicates (boolean). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static boolean containsDuplicates(int[] items) {
        java.util.Set<Integer> seen = new java.util.HashSet<>();
        for (int x : items) if (!seen.add(x)) return true;
        return false;
    }

    public static void main(String[] args) {
        check(containsDuplicates(new int[]{1, 2, 3, 1}), "has dup");
        check(!containsDuplicates(new int[]{1, 2, 3}), "unique");
        check(!containsDuplicates(new int[]{}), "empty");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
