/** Task 024 - Remove element at index from array (in-place). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int removeAt(int[] items, int index) {
        if (index < 0 || index >= items.length)
            throw new IndexOutOfBoundsException("index out of range");
        for (int i = index; i + 1 < items.length; i++)
            items[i] = items[i + 1];
        return items.length - 1;
    }

    public static void main(String[] args) {
        int[] a = {10, 20, 30, 40};
        check(removeAt(a, 1) == 3 && a[0] == 10 && a[1] == 30 && a[2] == 40, "middle");
        check(removeAt(new int[]{1}, 0) == 0, "single");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
