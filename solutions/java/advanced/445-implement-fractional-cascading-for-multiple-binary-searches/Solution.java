/** Task 445 - Implement fractional cascading for multiple binary searches. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class FractionalCascade {
    private final int k;
    private final int[][] augmented;
    private final boolean[][] nativeFlag;
    private final int[][] bridge;
    private final int[][] nextNative;

    FractionalCascade(int[][] lists) {
        k = lists.length;
        augmented = new int[k][];
        nativeFlag = new boolean[k][];
        augmented[k - 1] = lists[k - 1].clone();
        nativeFlag[k - 1] = new boolean[augmented[k - 1].length];
        Arrays.fill(nativeFlag[k - 1], true);
        for (int i = k - 2; i >= 0; i--) {
            int[] next = augmented[i + 1];
            int[] copied = new int[(next.length + 1) / 2];
            for (int j = 0; j < copied.length; j++) copied[j] = next[2 * j];
            int[] values = new int[lists[i].length + copied.length];
            boolean[] flags = new boolean[values.length];
            int p = 0, q = 0, idx = 0;
            while (p < lists[i].length || q < copied.length) {
                if (q == copied.length || (p < lists[i].length && lists[i][p] <= copied[q])) {
                    values[idx] = lists[i][p++];
                    flags[idx++] = true;
                } else {
                    values[idx] = copied[q++];
                    flags[idx++] = false;
                }
            }
            augmented[i] = values;
            nativeFlag[i] = flags;
        }
        bridge = new int[k][];
        for (int i = 0; i < k - 1; i++) {
            int[] upper = augmented[i + 1];
            int[] b = new int[augmented[i].length];
            int j = 0;
            for (int p = 0; p < augmented[i].length; p++) {
                while (j < upper.length && upper[j] < augmented[i][p]) j++;
                b[p] = j;
            }
            bridge[i] = b;
        }
        nextNative = new int[k][];
        for (int i = 0; i < k; i++) {
            int size = augmented[i].length;
            int[] nxt = new int[size + 1];
            nxt[size] = size;
            for (int p = size - 1; p >= 0; p--) nxt[p] = nativeFlag[i][p] ? p : nxt[p + 1];
            nextNative[i] = nxt;
        }
    }

    Integer[] successors(int x) {
        Integer[] results = new Integer[k];
        int pos = bisectLeft(augmented[0], x);
        for (int i = 0; i < k; i++) {
            if (i > 0) {
                if (pos == augmented[i - 1].length) {
                    int m = augmented[i].length;
                    pos = (m > 0 && augmented[i][m - 1] >= x) ? m - 1 : m;
                } else {
                    int q = bridge[i - 1][pos];
                    if (q > 0 && augmented[i][q - 1] >= x) q--;
                    pos = q;
                }
            }
            int nxt = nextNative[i][pos];
            results[i] = nxt < augmented[i].length ? Integer.valueOf(augmented[i][nxt]) : null;
        }
        return results;
    }

    private static int bisectLeft(int[] values, int x) {
        int lo = 0, hi = values.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (values[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public class Solution {
    private static int seed;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static int[][] lcgLists(int count, int size, int baseSeed) {
        seed = baseSeed;
        int[][] lists = new int[count][];
        for (int i = 0; i < count; i++) {
            int[] values = new int[size];
            int current = -40;
            for (int j = 0; j < size; j++) {
                current += 1 + nextSeed() % 9;
                values[j] = current;
            }
            lists[i] = values;
        }
        return lists;
    }

    private static Integer[] independentSuccessors(int[][] lists, int x) {
        Integer[] results = new Integer[lists.length];
        for (int i = 0; i < lists.length; i++) {
            int pos = lowerBound(lists[i], x);
            results[i] = pos < lists[i].length ? Integer.valueOf(lists[i][pos]) : null;
        }
        return results;
    }

    private static int lowerBound(int[] values, int x) {
        int lo = 0, hi = values.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (values[mid] < x) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static void main(String[] args) {
        FractionalCascade cascade = new FractionalCascade(new int[][]{{1, 10}, {5, 20}, {3, 7, 30}});
        check(Arrays.equals(cascade.successors(6), new Integer[]{10, 20, 7}), "hand query 6");
        check(Arrays.equals(cascade.successors(3), new Integer[]{10, 5, 3}), "hand query 3");
        check(Arrays.equals(cascade.successors(0), new Integer[]{1, 5, 3}), "hand query 0");
        check(Arrays.equals(cascade.successors(10), new Integer[]{10, 20, 30}), "hand query 10");
        check(Arrays.equals(cascade.successors(31), new Integer[]{null, null, null}), "hand query past end");
        check(Arrays.equals(cascade.successors(20), new Integer[]{null, 20, 30}), "hand query 20");

        int[][] lists = lcgLists(5, 12, 445);
        cascade = new FractionalCascade(lists);
        seed = 555;
        for (int step = 0; step < 200; step++) {
            int x = nextSeed() % 200 - 90;
            check(Arrays.equals(cascade.successors(x), independentSuccessors(lists, x)), "random query " + x);
        }

        int[][] single = lcgLists(1, 7, 616);
        cascade = new FractionalCascade(single);
        seed = 77;
        for (int step = 0; step < 40; step++) {
            int x = nextSeed() % 120 - 60;
            check(Arrays.equals(cascade.successors(x), independentSuccessors(single, x)), "single list query " + x);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
