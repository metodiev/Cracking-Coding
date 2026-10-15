/** Task 435 - Implement finger tree (sequence operations) concept. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Concept version of a finger tree: a sequence annotated by a monoid so that every subtree caches
 * the annotation of the elements below it, which is what lets a real finger tree answer prefix and
 * range measures in O(log n). The annotation monoid used here is M = (count, sum) with identity
 * (0, 0) and componentwise addition; measure() returns the cached annotation of the whole sequence
 * and measureRange(lo, hi) folds it over a range. The spine operations pushFront / pushBack /
 * popFront / popBack / front / back / concat / splitAt are simulated over a plain list while
 * keeping the annotation exact - documented honest O(n) internals, no worst-case finger-tree bound
 * claimed - and every scripted operation is asserted against a plain-list oracle (values, size,
 * cached measure and the monoid law combine(measure(a), measure(b)) == measure(concat(a, b))).
 */
import java.util.*;

class FingerTree {
    static final long[] IDENTITY = {0, 0};

    private final List<Integer> items;
    private long[] measure;

    FingerTree() {
        this.items = new ArrayList<>();
        this.measure = IDENTITY.clone();
    }

    FingerTree(List<Integer> initial) {
        this.items = new ArrayList<>(initial);
        long[] total = IDENTITY.clone();
        for (int value : this.items) total = combine(total, new long[]{1, value});
        this.measure = total;
    }

    static long[] combine(long[] left, long[] right) {
        return new long[]{left[0] + right[0], left[1] + right[1]};
    }

    static boolean sameMeasure(long[] actual, long count, long sum) {
        return actual[0] == count && actual[1] == sum;
    }

    int size() {
        return items.size();
    }

    long[] measure() {
        return measure.clone();
    }

    long[] measureRange(int lo, int hi) {
        if (lo < 0 || hi > items.size() || lo > hi) throw new IndexOutOfBoundsException("range out of bounds");
        long[] total = IDENTITY.clone();
        for (int i = lo; i < hi; i++) total = combine(total, new long[]{1, items.get(i)});
        return total;
    }

    Integer front() {
        return items.isEmpty() ? null : items.get(0);
    }

    Integer back() {
        return items.isEmpty() ? null : items.get(items.size() - 1);
    }

    FingerTree pushFront(int value) {
        items.add(0, value);
        measure = combine(new long[]{1, value}, measure);
        return this;
    }

    FingerTree pushBack(int value) {
        items.add(value);
        measure = combine(measure, new long[]{1, value});
        return this;
    }

    Integer popFront() {
        if (items.isEmpty()) return null;
        int value = items.remove(0);
        measure = combine(new long[]{-1, -value}, measure);
        return value;
    }

    Integer popBack() {
        if (items.isEmpty()) return null;
        int value = items.remove(items.size() - 1);
        measure = combine(new long[]{-1, -value}, measure);
        return value;
    }

    FingerTree concat(FingerTree other) {
        FingerTree joined = new FingerTree(items);
        joined.items.addAll(other.items);
        joined.measure = combine(measure, other.measure);
        return joined;
    }

    FingerTree[] splitAt(int index) {
        if (index < 0 || index > items.size()) throw new IndexOutOfBoundsException("split index out of bounds");
        FingerTree left = new FingerTree(items.subList(0, index));
        FingerTree right = new FingerTree(items.subList(index, items.size()));
        return new FingerTree[]{left, right};
    }

    List<Integer> toList() {
        return new ArrayList<>(items);
    }
}

public class Solution {
    static void checkMeasure(FingerTree tree, long count, long sum, String label) {
        check(FingerTree.sameMeasure(tree.measure(), count, sum), label);
    }

    public static void main(String[] args) {
        FingerTree empty = new FingerTree();
        check(empty.size() == 0, "empty size");
        checkMeasure(empty, 0, 0, "empty measure");
        check(empty.toList().isEmpty(), "empty list");
        check(empty.front() == null && empty.back() == null, "empty ends");
        check(empty.popFront() == null && empty.popBack() == null, "empty pops");

        FingerTree tree = new FingerTree();
        tree.pushBack(1).pushBack(2).pushBack(3).pushFront(0);
        check(tree.toList().toString().equals("[0, 1, 2, 3]"), "hand list");
        check(tree.size() == 4, "hand size");
        checkMeasure(tree, 4, 6, "hand measure");
        check(tree.front() == 0 && tree.back() == 3, "hand ends");

        FingerTree[] parts = tree.splitAt(0);
        check(parts[0].toList().isEmpty() && parts[1].toList().equals(tree.toList()), "split at 0");
        checkMeasure(parts[0], 0, 0, "split 0 left measure");
        parts = tree.splitAt(4);
        check(parts[0].toList().equals(tree.toList()) && parts[1].toList().isEmpty(), "split at len");
        checkMeasure(parts[1], 0, 0, "split len right measure");
        parts = tree.splitAt(2);
        check(parts[0].toList().toString().equals("[0, 1]"), "split middle left");
        check(parts[1].toList().toString().equals("[2, 3]"), "split middle right");
        check(java.util.Arrays.equals(FingerTree.combine(parts[0].measure(), parts[1].measure()), tree.measure()),
                "split measures combine");
        check(FingerTree.sameMeasure(tree.measureRange(1, 3), 2, 3), "range measure");
        check(FingerTree.sameMeasure(tree.measureRange(2, 2), 0, 0), "empty range measure");

        check(new FingerTree(Arrays.asList(9)).concat(new FingerTree()).toList().toString().equals("[9]"), "concat empty right");
        check(new FingerTree().concat(new FingerTree(Arrays.asList(7))).toList().toString().equals("[7]"), "concat empty left");
        checkMeasure(new FingerTree(Arrays.asList(1, 2)).concat(new FingerTree(Arrays.asList(3))), 3, 6, "concat measure");
        check(tree.concat(tree).toList().toString().equals("[0, 1, 2, 3, 0, 1, 2, 3]"), "self concat");

        tree = new FingerTree(Arrays.asList(5, 6, 7));
        check(tree.popFront() == 5, "pop front value");
        check(tree.popBack() == 7, "pop back value");
        check(tree.toList().toString().equals("[6]"), "single left");
        checkMeasure(tree, 1, 6, "single measure");
        check(tree.popFront() == 6, "pop last");
        check(tree.popFront() == null, "pop empty");
        check(tree.toList().isEmpty() && tree.measure()[0] == 0, "drained");

        tree = new FingerTree();
        List<Integer> oracle = new ArrayList<>();
        int seed = 4350;
        for (int step = 0; step < 400; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 5;
            if (op == 0) {
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 50 - 25;
                tree.pushBack(value);
                oracle.add(value);
            } else if (op == 1) {
                seed = (seed * 75 + 74) % 65537;
                int value = seed % 50 - 25;
                tree.pushFront(value);
                oracle.add(0, value);
            } else if (op == 2) {
                Integer expected = oracle.isEmpty() ? null : oracle.remove(0);
                check(Objects.equals(tree.popFront(), expected), "pop front step " + step);
            } else if (op == 3) {
                Integer expected = oracle.isEmpty() ? null : oracle.remove(oracle.size() - 1);
                check(Objects.equals(tree.popBack(), expected), "pop back step " + step);
            } else {
                seed = (seed * 75 + 74) % 65537;
                int index = seed % (oracle.size() + 1);
                FingerTree[] split = tree.splitAt(index);
                check(split[0].toList().equals(oracle.subList(0, index)), "split left step " + step);
                check(split[1].toList().equals(oracle.subList(index, oracle.size())), "split right step " + step);
                check(java.util.Arrays.equals(FingerTree.combine(split[0].measure(), split[1].measure()), tree.measure()),
                        "split combine step " + step);
                tree = split[0].concat(split[1]);
            }
            check(tree.toList().equals(oracle), "list step " + step);
            check(tree.size() == oracle.size(), "size step " + step);
            long sum = 0;
            for (int value : oracle) sum += value;
            checkMeasure(tree, oracle.size(), sum, "measure step " + step);
            if (!oracle.isEmpty()) {
                check(tree.front() == oracle.get(0) && tree.back() == oracle.get(oracle.size() - 1), "ends step " + step);
            }
            seed = (seed * 75 + 74) % 65537;
            int lo = seed % (oracle.size() + 1);
            seed = (seed * 75 + 74) % 65537;
            int hi = lo + seed % (oracle.size() - lo + 1);
            long rangeSum = 0;
            for (int i = lo; i < hi; i++) rangeSum += oracle.get(i);
            check(FingerTree.sameMeasure(tree.measureRange(lo, hi), hi - lo, rangeSum), "range step " + step);
        }
        check(!oracle.isEmpty(), "workload left values");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
