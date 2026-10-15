/** Task 070 - Insertion sort. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static void insertionSort(int[] a) {
        for (int i = 1; i < a.length; i++) {
            int key = a[i];
            int j = i - 1;
            while (j >= 0 && a[j] > key) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
    }

    public static void main(String[] args) {
        int[] a = {5, 3, 8, 1, 9, 2};
        insertionSort(a);
        check(Arrays.equals(a, new int[]{1, 2, 3, 5, 8, 9}), "basic");
        int[] b = {1, 2, 3};
        insertionSort(b);
        check(Arrays.equals(b, new int[]{1, 2, 3}), "already sorted");
        int[] c = {};
        insertionSort(c);
        check(c.length == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
