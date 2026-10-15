/** Task 071 - Bubble sort (optimized). (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static void bubbleSort(int[] a) {
        int end = a.length;
        while (end > 1) {
            int lastSwap = 0;
            for (int i = 1; i < end; i++) {
                if (a[i - 1] > a[i]) {
                    int t = a[i - 1]; a[i - 1] = a[i]; a[i] = t;
                    lastSwap = i;
                }
            }
            if (lastSwap == 0) break;
            end = lastSwap;
        }
    }

    public static void main(String[] args) {
        int[] a = {5, 3, 8, 1, 9, 2};
        bubbleSort(a);
        check(Arrays.equals(a, new int[]{1, 2, 3, 5, 8, 9}), "basic");
        int[] b = {1, 2, 3};
        bubbleSort(b);
        check(Arrays.equals(b, new int[]{1, 2, 3}), "early exit");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
