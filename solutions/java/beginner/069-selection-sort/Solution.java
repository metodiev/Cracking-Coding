/** Task 069 - Selection sort. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static void selectionSort(int[] a) {
        int n = a.length;
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++)
                if (a[j] < a[minIdx]) minIdx = j;
            if (minIdx != i) {
                int t = a[i]; a[i] = a[minIdx]; a[minIdx] = t;
            }
        }
    }

    public static void main(String[] args) {
        int[] a = {5, 3, 8, 1, 9, 2};
        selectionSort(a);
        check(Arrays.equals(a, new int[]{1, 2, 3, 5, 8, 9}), "basic");
        int[] b = {2, 1};
        selectionSort(b);
        check(Arrays.equals(b, new int[]{1, 2}), "two");
        int[] c = {};
        selectionSort(c);
        check(c.length == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
