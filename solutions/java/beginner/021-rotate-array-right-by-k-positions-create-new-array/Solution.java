/** Task 021 - Rotate array right by k positions (create new array). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int[] rotateRight(int[] items, int k) {
        if (items.length == 0) return new int[0];
        k %= items.length;
        int[] out = new int[items.length];
        for (int i = 0; i < items.length; i++)
            out[(i + k) % items.length] = items[i];
        return out;
    }

    public static void main(String[] args) {
        check(java.util.Arrays.equals(rotateRight(new int[]{1, 2, 3, 4, 5}, 2), new int[]{4, 5, 1, 2, 3}), "k=2");
        check(java.util.Arrays.equals(rotateRight(new int[]{1, 2}, 3), new int[]{2, 1}), "k wraps");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
