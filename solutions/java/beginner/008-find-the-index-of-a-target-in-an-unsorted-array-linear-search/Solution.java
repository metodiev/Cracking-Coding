/** Task 008 - Find the index of a target in an unsorted array (linear search). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int linearSearch(int[] items, int target) {
        for (int i = 0; i < items.length; i++)
            if (items[i] == target) return i;
        return -1;
    }

    public static void main(String[] args) {
        check(linearSearch(new int[]{4, 2, 7, 1}, 7) == 2, "found");
        check(linearSearch(new int[]{4, 2, 7, 1}, 9) == -1, "missing");
        check(linearSearch(new int[]{}, 1) == -1, "empty");
        check(linearSearch(new int[]{5, 5, 5}, 5) == 0, "duplicates -> first");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
