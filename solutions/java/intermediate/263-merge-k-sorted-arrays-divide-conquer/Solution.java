/** Task 263 - Merge K sorted arrays (divide & conquer). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

public class Solution {
    static int[] mergeTwo(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        int i = 0, j = 0, k = 0;
        while (i < a.length && j < b.length) out[k++] = a[i] <= b[j] ? a[i++] : b[j++];
        while (i < a.length) out[k++] = a[i++];
        while (j < b.length) out[k++] = b[j++];
        return out;
    }

    /**
     * Pairwise divide-and-conquer merge with a plain 2-way two-pointer merge (NO heap):
     * log k levels of O(n) work -> O(n log k). Index ranges avoid copying the halves.
     * k = 0 -> empty; k = 1 returns a fresh copy; inputs are never modified.
     */
    static int[] mergeKDivide(int[][] arrays) {
        if (arrays.length == 0) return new int[0];
        return divide(arrays, 0, arrays.length);
    }

    private static int[] divide(int[][] arrays, int lo, int hi) {
        if (hi - lo == 1) return Arrays.copyOf(arrays[lo], arrays[lo].length);
        int mid = (lo + hi) / 2;
        return mergeTwo(divide(arrays, lo, mid), divide(arrays, mid, hi));
    }

    /** Independent heap-based merge used only for cross-checks (same result expected). */
    static int[] mergeKHeap(int[][] arrays) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((p, q) -> {
            if (p[0] != q[0]) return Integer.compare(p[0], q[0]);
            if (p[1] != q[1]) return Integer.compare(p[1], q[1]);
            return Integer.compare(p[2], q[2]);
        });
        for (int i = 0; i < arrays.length; i++)
            if (arrays[i].length > 0) pq.add(new int[]{arrays[i][0], i, 0});
        List<Integer> out = new ArrayList<>();
        while (!pq.isEmpty()) {
            int[] e = pq.poll();
            out.add(e[0]);
            if (e[2] + 1 < arrays[e[1]].length) pq.add(new int[]{arrays[e[1]][e[2] + 1], e[1], e[2] + 1});
        }
        int[] result = new int[out.size()];
        for (int i = 0; i < result.length; i++) result[i] = out.get(i);
        return result;
    }

    /** Flat sort reference used only for cross-checks. */
    static int[] mergeOracle(int[][] arrays) {
        List<Integer> values = new ArrayList<>();
        for (int[] arr : arrays) for (int v : arr) values.add(v);
        Collections.sort(values);
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    static void checkArray(int[] got, int[] expected, String label) {
        check(Arrays.equals(got, expected), label + " got=" + Arrays.toString(got));
    }

    public static void main(String[] args) {
        int[][] classic = {{1, 4, 5}, {1, 3, 4}, {2, 6}};
        checkArray(mergeKDivide(classic), new int[]{1, 1, 2, 3, 4, 4, 5, 6}, "classic");
        checkArray(mergeKDivide(new int[0][]), new int[0], "k = 0");
        checkArray(mergeKDivide(new int[][]{{3, 4}}), new int[]{3, 4}, "k = 1");
        checkArray(mergeKDivide(new int[][]{{}}), new int[0], "single empty");
        checkArray(mergeKDivide(new int[][]{{1}, {2, 3, 4, 5}, {6}}), new int[]{1, 2, 3, 4, 5, 6}, "unequal lengths");
        checkArray(mergeKDivide(new int[][]{{2, 2}, {2, 2}, {2}}), new int[]{2, 2, 2, 2, 2}, "duplicates");
        checkArray(mergeKDivide(new int[][]{{}, {1}, {}, {2, 3}}), new int[]{1, 2, 3}, "empty arrays inside");
        checkArray(mergeKDivide(new int[][]{{-3, -1}, {-2, 0}}), new int[]{-3, -2, -1, 0}, "negatives");

        int[] original = {3, 4};
        int[] copy = mergeKDivide(new int[][]{original});
        check(Arrays.equals(copy, original) && copy != original, "k = 1 returns a fresh copy");

        int[][][] fixed = {classic, new int[0][], {{3, 4}}, {{}}, {{1}, {2, 3, 4, 5}, {6}},
                {{2, 2}, {2, 2}, {2}}, {{-3, -1}, {-2, 0}}};
        for (int[][] arrays : fixed)                        // divide & conquer == heap == flat sort
            checkArray(mergeKDivide(arrays), mergeKHeap(arrays), "divide == heap");
        for (int[][] arrays : fixed)
            checkArray(mergeKDivide(arrays), mergeOracle(arrays), "divide == oracle");

        Random rng = new Random(263);
        for (int test = 0; test < 300; test++) {
            int k = rng.nextInt(7);
            int[][] arrays = new int[k][];
            for (int i = 0; i < k; i++) {
                int[] arr = new int[rng.nextInt(7)];
                for (int j = 0; j < arr.length; j++) arr[j] = rng.nextInt(13) - 6;
                Arrays.sort(arr);
                arrays[i] = arr;
            }
            checkArray(mergeKDivide(arrays), mergeKHeap(arrays), "random divide == heap");
            checkArray(mergeKDivide(arrays), mergeOracle(arrays), "random divide == oracle");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
