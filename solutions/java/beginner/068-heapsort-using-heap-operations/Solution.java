/** Task 068 - Heapsort using heap operations. (Tier: Beginner, CTCI Ch 10) */
import java.util.Arrays;

public class Solution {
    static void heapSort(int[] a) {
        int n = a.length;
        for (int i = n / 2 - 1; i >= 0; i--) siftDown(a, i, n);
        for (int end = n - 1; end > 0; end--) {
            int t = a[0]; a[0] = a[end]; a[end] = t;
            siftDown(a, 0, end);
        }
    }

    private static void siftDown(int[] a, int i, int size) {
        while (true) {
            int largest = i, left = 2 * i + 1, right = 2 * i + 2;
            if (left < size && a[left] > a[largest]) largest = left;
            if (right < size && a[right] > a[largest]) largest = right;
            if (largest == i) return;
            int t = a[i]; a[i] = a[largest]; a[largest] = t;
            i = largest;
        }
    }

    public static void main(String[] args) {
        int[] a = {5, 3, 8, 1, 9, 2};
        heapSort(a);
        check(Arrays.equals(a, new int[]{1, 2, 3, 5, 8, 9}), "basic");
        int[] b = {3, 3, 1};
        heapSort(b);
        check(Arrays.equals(b, new int[]{1, 3, 3}), "duplicates");
        int[] c = {};
        heapSort(c);
        check(c.length == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
