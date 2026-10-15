/** Task 339 - Range minimum query with offline queries (Mo's algorithm simple). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class LazyMinMultiset {
    // Minimum of the swept window: a min-heap with lazily deleted values; each remove marks
    // one occurrence, and the mark is consumed when that value reaches the heap top.
    private final PriorityQueue<Integer> heap = new PriorityQueue<>();
    private final Map<Integer, Integer> pending = new HashMap<>();
    private int size = 0;

    void add(int value) {
        heap.add(value);
        size++;
    }

    void remove(int value) {
        pending.merge(value, 1, Integer::sum);
        size--;
    }

    int minimum() {
        while (!heap.isEmpty() && pending.getOrDefault(heap.peek(), 0) > 0) {
            int value = heap.poll();
            pending.merge(value, -1, Integer::sum);
        }
        if (size <= 0) throw new AssertionError("minimum of an empty multiset");
        return heap.peek();
    }
}

public class Solution {
    /**
     * Simplified Mo's algorithm: block-sort the queries by (l / B, r) with B = n / sqrt(q),
     * then sweep the window boundaries. The sweep performs O((n + q) * sqrt(n)) add/remove
     * moves (the classic Mo bound), each O(log n) amortised on the lazy min-heap.
     */
    static int[] moRangeMin(int[] values, int[][] queries) {
        int n = values.length;
        int q = queries.length;
        if (q == 0) return new int[0];
        int root = (int) Math.sqrt(q);
        int block = root > 0 ? Math.max(1, n / root) : n;

        Integer[] order = new Integer[q];
        for (int i = 0; i < q; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int blockA = queries[a][0] / block;
            int blockB = queries[b][0] / block;
            if (blockA != blockB) return blockA - blockB;
            int rightA = blockA % 2 == 0 ? queries[a][1] : -queries[a][1];
            int rightB = blockB % 2 == 0 ? queries[b][1] : -queries[b][1];
            return rightA - rightB;
        });

        LazyMinMultiset window = new LazyMinMultiset();
        int[] answers = new int[q];
        int currentLeft = 0, currentRight = -1;
        for (int index : order) {
            int left = queries[index][0];
            int right = queries[index][1];
            while (currentLeft > left) window.add(values[--currentLeft]);
            while (currentRight < right) window.add(values[++currentRight]);
            while (currentLeft < left) window.remove(values[currentLeft++]);
            while (currentRight > right) window.remove(values[currentRight--]);
            answers[index] = window.minimum();
        }
        return answers;
    }

    static int[] naiveRangeMin(int[] values, int[][] queries) {
        int[] answers = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            int best = values[queries[i][0]];
            for (int j = queries[i][0] + 1; j <= queries[i][1]; j++) best = Math.min(best, values[j]);
            answers[i] = best;
        }
        return answers;
    }

    private static int seed = 5150;

    private static String join(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        // Hand-checked minima on a small fixed array.
        int[] values = {1, 2, 1, 3, 2, 4};
        int[][] queries = {{0, 2}, {1, 4}, {0, 5}, {3, 3}, {2, 5}, {1, 1}, {5, 5}};
        check(join(moRangeMin(values, queries)).equals("1,1,1,3,1,2,4"), "hand answers");
        check(join(naiveRangeMin(values, queries)).equals("1,1,1,3,1,2,4"), "hand naive");
        check(join(moRangeMin(new int[]{7}, new int[][]{{0, 0}})).equals("7"), "single element");
        check(moRangeMin(values, new int[][]{}).length == 0, "no queries");

        // All-equal and strictly increasing arrays, including duplicate-heavy windows.
        int[] flat = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
        check(join(moRangeMin(flat, new int[][]{{0, 9}, {2, 3}, {4, 4}})).equals("5,5,5"), "flat");
        int[] rising = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        check(join(moRangeMin(rising, new int[][]{{0, 9}, {0, 0}, {3, 7}})).equals("0,0,3"), "rising");
        int[] twos = {2, 2, 3, 2, 3, 3};
        check(join(moRangeMin(twos, new int[][]{{0, 5}, {1, 4}, {2, 5}, {3, 5}})).equals("2,2,2,2"),
                "duplicate-heavy");

        // LCG cross-check with a spread of sizes and query counts.
        for (int trial = 0; trial < 12; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 120;
            seed = (seed * 75 + 74) % 65537;
            int q = 1 + seed % 200;
            seed = (seed * 75 + 74) % 65537;
            int alphabet = 1 + seed % 12;
            int[] randomValues = new int[n];
            for (int i = 0; i < n; i++) {
                seed = (seed * 75 + 74) % 65537;
                randomValues[i] = seed % alphabet;
            }
            int[][] randomQueries = new int[q][];
            for (int i = 0; i < q; i++) {
                seed = (seed * 75 + 74) % 65537;
                int left = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int right = left + seed % (n - left);
                randomQueries[i] = new int[]{left, right};
            }
            check(join(moRangeMin(randomValues, randomQueries))
                    .equals(join(naiveRangeMin(randomValues, randomQueries))), "random " + trial);
        }

        // Single-position queries, one full query, and an adversarially reversed query order.
        seed = 424242;
        int[] singlesValues = new int[50];
        for (int i = 0; i < 50; i++) {
            seed = (seed * 75 + 74) % 65537;
            singlesValues[i] = seed % 7;
        }
        int[][] singles = new int[50][];
        for (int i = 0; i < 50; i++) singles[i] = new int[]{i, i};
        check(join(moRangeMin(singlesValues, singles)).equals(join(singlesValues)), "single positions");
        check(join(moRangeMin(singlesValues, new int[][]{{0, 49}})).equals(join(naiveRangeMin(singlesValues, new int[][]{{0, 49}}))),
                "full range");
        int[][] reversed = new int[50][];
        for (int i = 0; i < 50; i++) reversed[i] = new int[]{49 - i, 49 - i};
        check(join(moRangeMin(singlesValues, reversed)).equals(join(naiveRangeMin(singlesValues, reversed))),
                "reversed order");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
