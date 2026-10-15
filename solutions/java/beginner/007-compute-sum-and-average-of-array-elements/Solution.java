/** Task 007 - Compute sum and average of array elements. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    /** Returns {sum, average}; average is 0 for an empty array. */
    static double[] sumAndAverage(int[] items) {
        double total = 0;
        for (int x : items) total += x;
        double avg = items.length == 0 ? 0 : total / items.length;
        return new double[]{total, avg};
    }

    public static void main(String[] args) {
        double[] r = sumAndAverage(new int[]{1, 2, 3, 4});
        check(r[0] == 10 && r[1] == 2.5, "basic");
        check(sumAndAverage(new int[]{})[1] == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
