/** Task 019 - Check if array is sorted (ascending). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static boolean isSorted(int[] items) {
        for (int i = 0; i + 1 < items.length; i++)
            if (items[i] > items[i + 1]) return false;
        return true;
    }

    public static void main(String[] args) {
        check(isSorted(new int[]{}), "empty");
        check(isSorted(new int[]{1}), "single");
        check(isSorted(new int[]{1, 2, 2, 3}), "sorted");
        check(!isSorted(new int[]{1, 3, 2}), "unsorted");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
