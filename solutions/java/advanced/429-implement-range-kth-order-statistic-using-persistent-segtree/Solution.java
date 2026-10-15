/** Task 429 - Implement range kth order statistic using persistent segtree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class PersistentKthSmallest {
    final int[] sortedValues;
    final int domain;
    final List<Integer> left = new ArrayList<>();
    final List<Integer> right = new ArrayList<>();
    final List<Integer> count = new ArrayList<>();
    final List<Integer> roots = new ArrayList<>();

    PersistentKthSmallest(int[] values) {
        int[] sorted = values.clone();
        Arrays.sort(sorted);
        int distinct = 0;
        for (int value : sorted) {
            if (distinct == 0 || sorted[distinct - 1] != value) sorted[distinct++] = value;
        }
        sortedValues = Arrays.copyOf(sorted, distinct);
        domain = distinct;
        Map<Integer, Integer> rankOf = new HashMap<>();
        for (int i = 0; i < domain; i++) rankOf.put(sortedValues[i], i);
        left.add(0);
        right.add(0);
        count.add(0);
        int root = 0;
        roots.add(0);
        for (int value : values) {
            root = insert(root, 0, domain - 1, rankOf.get(value));
            roots.add(root);
        }
    }

    private int makeNode(int l, int r, int c) {
        left.add(l);
        right.add(r);
        count.add(c);
        return count.size() - 1;
    }

    private int insert(int node, int lo, int hi, int position) {
        if (lo == hi) return makeNode(0, 0, count.get(node) + 1);
        int mid = (lo + hi) / 2;
        if (position <= mid) {
            int l = insert(left.get(node), lo, mid, position);
            int r = right.get(node);
            return makeNode(l, r, count.get(l) + count.get(r));
        }
        int l = left.get(node);
        int r = insert(right.get(node), mid + 1, hi, position);
        return makeNode(l, r, count.get(l) + count.get(r));
    }

    int kth(int from, int to, int k) {
        int low = roots.get(from);
        int high = roots.get(to + 1);
        int lo = 0, hi = domain - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int leftCount = count.get(left.get(high)) - count.get(left.get(low));
            if (k <= leftCount) {
                low = left.get(low);
                high = left.get(high);
                hi = mid;
            } else {
                k -= leftCount;
                low = right.get(low);
                high = right.get(high);
                lo = mid + 1;
            }
        }
        return sortedValues[lo];
    }
}

public class Solution {
    private static int seed;

    private static int[] lcgArray(int n) {
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 13 - 6;
        }
        return values;
    }

    public static void main(String[] args) {
        PersistentKthSmallest tree = new PersistentKthSmallest(new int[]{5, 1, 4, 2, 3});
        check(tree.kth(0, 4, 1) == 1, "whole 1st");
        check(tree.kth(0, 4, 2) == 2, "whole 2nd");
        check(tree.kth(0, 4, 3) == 3, "whole 3rd");
        check(tree.kth(0, 4, 4) == 4, "whole 4th");
        check(tree.kth(0, 4, 5) == 5, "whole 5th");
        check(tree.kth(1, 3, 1) == 1, "middle 1st");
        check(tree.kth(1, 3, 2) == 2, "middle 2nd");
        check(tree.kth(1, 3, 3) == 4, "middle 3rd");
        check(tree.kth(2, 2, 1) == 4, "cell");
        check(tree.kth(4, 4, 1) == 3, "last cell");
        check(tree.kth(0, 0, 1) == 5, "first cell");

        PersistentKthSmallest dupes = new PersistentKthSmallest(new int[]{2, 2, 2});
        check(dupes.kth(0, 2, 1) == 2, "dupes 1st");
        check(dupes.kth(0, 2, 2) == 2, "dupes 2nd");
        check(dupes.kth(0, 2, 3) == 2, "dupes 3rd");
        check(dupes.kth(1, 1, 1) == 2, "dupes single");

        PersistentKthSmallest negatives = new PersistentKthSmallest(new int[]{-3, 7, -3, 0});
        check(negatives.kth(0, 3, 1) == -3, "neg 1st");
        check(negatives.kth(0, 3, 2) == -3, "neg 2nd");
        check(negatives.kth(0, 3, 3) == 0, "neg 3rd");
        check(negatives.kth(0, 3, 4) == 7, "neg 4th");
        check(negatives.kth(1, 2, 1) == -3, "neg slice 1st");
        check(negatives.kth(1, 2, 2) == 7, "neg slice 2nd");

        for (int config : new int[]{429, 12345, 54321}) {
            seed = config;
            int[] values = lcgArray(1 + seed % 40);
            PersistentKthSmallest structure = new PersistentKthSmallest(values);
            int local = config;
            for (int step = 0; step < 40; step++) {
                local = (local * 75 + 74) % 65537;
                int from = local % values.length;
                int to = (local / 7) % values.length;
                if (from > to) { int tmp = from; from = to; to = tmp; }
                int[] segment = Arrays.copyOfRange(values, from, to + 1);
                Arrays.sort(segment);
                local = (local * 75 + 74) % 65537;
                int k = 1 + local % segment.length;
                int got = structure.kth(from, to, k);
                check(got == segment[k - 1], "lcg query " + step + " got=" + got + " want=" + segment[k - 1]);
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
