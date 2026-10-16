/** Task 642 - Implement join algorithms: nested-loop, hash join, sort-merge join. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Three real join implementations over in-memory tuples. A tuple is a two-slot int array {key, payload};
 * a joined tuple is the four-slot concatenation of the left and right input tuples.
 *
 *   nested loop  - the textbook double loop; supports an arbitrary predicate (this is the only one of
 *                  the three that can run a theta join such as |left.key - right.key| <= 1);
 *   hash join    - builds a HashMap from the smaller input, probes with the larger one, and therefore
 *                  handles duplicate keys on both sides (probe hits are appended, not overwritten);
 *   sort merge   - merge-sorts both inputs with a hand-written merge sort and then merges them,
 *                  expanding the full left-run x right-run block for equal keys, so it produces
 *                  sorted output and handles duplicates exactly like the hash join.
 *
 * Counters are kept per algorithm (pair comparisons for nested loop, build+probe touches for the hash
 * join, sort comparisons and merge advances for sort merge), because the point of a join *algorithm*
 * is the work it does, not just the answer.
 *
 * Verified: all three algorithms return exactly the same multiset of tuples on hand-worked cases,
 * on 60 generated tables (including negative keys, no-match, all-match and duplicate-heavy inputs)
 * and on empty inputs; the result size equals an independent aggregation oracle
 * sum over keys of count(left, key) * count(right, key); every produced tuple really has equal join
 * keys; sort merge emits its rows in key order while nested loop emits them in left-major order; and
 * the counters show the expected ordering of the work performed (hash join touches each input row
 * once, nested loop compares every pair, sort merge pays for sorting but then only walks both inputs).
 */
import java.util.*;

class JoinAlgorithms {

    static final class Counter {
        long pairComparisons;      // nested loop: one per (left, right) pair
        long keyComparisons;       // equality/order checks against the key
        long probes;               // hash join: build inserts + probes
        long sortComparisons;      // sort merge: comparisons inside the merge sorts
        long rowsScanned;

        void add(Counter other) {
            pairComparisons += other.pairComparisons;
            keyComparisons += other.keyComparisons;
            probes += other.probes;
            sortComparisons += other.sortComparisons;
            rowsScanned += other.rowsScanned;
        }
    }

    interface PairPredicate {
        boolean test(int[] left, int[] right);
    }

    static int[] concat(int[] left, int[] right) {
        int[] out = new int[left.length + right.length];
        System.arraycopy(left, 0, out, 0, left.length);
        System.arraycopy(right, 0, out, left.length, right.length);
        return out;
    }

    /** Double loop with an equality test; the only algorithm here that accepts a general predicate. */
    static List<int[]> nestedLoop(List<int[]> left, List<int[]> right, int leftKey, int rightKey, Counter c) {
        return nestedLoopTheta(left, right, (l, r) -> l[leftKey] == r[rightKey], c);
    }

    static List<int[]> nestedLoopTheta(List<int[]> left, List<int[]> right, PairPredicate predicate, Counter c) {
        List<int[]> out = new ArrayList<>();
        for (int[] l : left) {
            for (int[] r : right) {
                c.pairComparisons++;
                c.keyComparisons++;
                if (predicate.test(l, r)) out.add(concat(l, r));
            }
        }
        c.rowsScanned += (long) left.size() * right.size();
        return out;
    }

    /** Builds the smaller input into a bucket map, then probes it with the larger one. */
    static List<int[]> hashJoin(List<int[]> left, List<int[]> right, int leftKey, int rightKey, Counter c) {
        boolean leftIsBuild = left.size() <= right.size();
        List<int[]> build = leftIsBuild ? left : right;
        List<int[]> probe = leftIsBuild ? right : left;
        int buildKey = leftIsBuild ? leftKey : rightKey;
        int probeKey = leftIsBuild ? rightKey : leftKey;
        Map<Integer, List<int[]>> buckets = new HashMap<>();
        for (int[] row : build) {
            buckets.computeIfAbsent(row[buildKey], key -> new ArrayList<>()).add(row);
            c.probes++;
        }
        List<int[]> out = new ArrayList<>();
        for (int[] row : probe) {
            c.probes++;
            List<int[]> bucket = buckets.get(row[probeKey]);
            if (bucket == null) continue;
            for (int[] match : bucket) {
                c.keyComparisons++;
                out.add(leftIsBuild ? concat(match, row) : concat(row, match));
            }
        }
        return out;
    }

    static List<int[]> mergeSort(List<int[]> rows, int key, Counter c) {
        if (rows.size() < 2) return rows;
        int mid = rows.size() / 2;
        List<int[]> a = mergeSort(new ArrayList<>(rows.subList(0, mid)), key, c);
        List<int[]> b = mergeSort(new ArrayList<>(rows.subList(mid, rows.size())), key, c);
        List<int[]> out = new ArrayList<>(rows.size());
        int i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            c.sortComparisons++;
            if (a.get(i)[key] <= b.get(j)[key]) out.add(a.get(i++));
            else out.add(b.get(j++));
        }
        while (i < a.size()) out.add(a.get(i++));
        while (j < b.size()) out.add(b.get(j++));
        return out;
    }

    /** Sorts both inputs, then walks them once; equal-key runs are expanded as a full block. */
    static List<int[]> sortMergeJoin(List<int[]> left, List<int[]> right, int leftKey, int rightKey, Counter c) {
        List<int[]> sortedLeft = mergeSort(new ArrayList<>(left), leftKey, c);
        List<int[]> sortedRight = mergeSort(new ArrayList<>(right), rightKey, c);
        List<int[]> out = new ArrayList<>();
        int i = 0, j = 0;
        while (i < sortedLeft.size() && j < sortedRight.size()) {
            int a = sortedLeft.get(i)[leftKey];
            int b = sortedRight.get(j)[rightKey];
            c.keyComparisons++;
            if (a < b) {
                i++;
            } else if (a > b) {
                j++;
            } else {
                int iEnd = i;
                while (iEnd < sortedLeft.size() && sortedLeft.get(iEnd)[leftKey] == a) iEnd++;
                int jEnd = j;
                while (jEnd < sortedRight.size() && sortedRight.get(jEnd)[rightKey] == b) jEnd++;
                for (int x = i; x < iEnd; x++) {
                    for (int y = j; y < jEnd; y++) out.add(concat(sortedLeft.get(x), sortedRight.get(y)));
                }
                i = iEnd;
                j = jEnd;
            }
        }
        return out;
    }

    /** Independent oracle: sum over keys of count(left, key) * count(right, key). */
    static long expectedSize(List<int[]> left, List<int[]> right, int leftKey, int rightKey) {
        Map<Integer, Long> counts = new HashMap<>();
        for (int[] l : left) counts.merge(l[leftKey], 1L, Long::sum);
        long total = 0;
        for (int[] r : right) total += counts.getOrDefault(r[rightKey], 0L);
        return total;
    }

    static List<String> canonical(List<int[]> rows) {
        List<String> out = new ArrayList<>(rows.size());
        for (int[] row : rows) out.add(Arrays.toString(row));
        Collections.sort(out);
        return out;
    }

    static boolean hasEqualKeys(List<int[]> rows, int leftKey, int rightKey, int width) {
        for (int[] row : rows) if (row[leftKey] != row[width + rightKey]) return false;
        return true;
    }

    static boolean sortedByKey(List<int[]> rows, int key) {
        for (int i = 1; i < rows.size(); i++) if (rows.get(i - 1)[key] > rows.get(i)[key]) return false;
        return true;
    }

    static List<int[]> table(int[][] rows) {
        List<int[]> out = new ArrayList<>();
        for (int[] row : rows) out.add(row);
        return out;
    }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

public class Solution {

    /** {key, payload} tuples with keys drawn from a small domain, payloads unique per table. */
    static List<int[]> randomTable(SplitMix64 rng, int rows, int keyDomain, int payloadBase) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < rows; i++) out.add(new int[]{rng.nextInt(keyDomain), payloadBase + i});
        return out;
    }

    public static void main(String[] args) {
        List<int[]> left = JoinAlgorithms.table(new int[][]{{1, 10}, {2, 20}, {2, 21}});
        List<int[]> right = JoinAlgorithms.table(new int[][]{{2, 200}, {3, 300}, {2, 201}});

        JoinAlgorithms.Counter loopCounter = new JoinAlgorithms.Counter();
        List<int[]> loop = JoinAlgorithms.nestedLoop(left, right, 0, 0, loopCounter);
        JoinAlgorithms.Counter hashCounter = new JoinAlgorithms.Counter();
        List<int[]> hash = JoinAlgorithms.hashJoin(left, right, 0, 0, hashCounter);
        JoinAlgorithms.Counter mergeCounter = new JoinAlgorithms.Counter();
        List<int[]> merge = JoinAlgorithms.sortMergeJoin(left, right, 0, 0, mergeCounter);

        check(JoinAlgorithms.canonical(loop).equals(Arrays.asList(
                        "[2, 20, 2, 200]", "[2, 20, 2, 201]", "[2, 21, 2, 200]", "[2, 21, 2, 201]")),
                "hand-worked nested loop result: " + JoinAlgorithms.canonical(loop));
        check(JoinAlgorithms.canonical(loop).equals(JoinAlgorithms.canonical(hash)), "hash agrees with loop");
        check(JoinAlgorithms.canonical(loop).equals(JoinAlgorithms.canonical(merge)), "merge agrees with loop");
        check(loop.size() == JoinAlgorithms.expectedSize(left, right, 0, 0),
                "result size matches the aggregation oracle");
        check(JoinAlgorithms.hasEqualKeys(loop, 0, 0, 2), "every output tuple has equal keys");
        check(loopCounter.pairComparisons == 9, "nested loop compared every pair: "
                + loopCounter.pairComparisons);
        check(hashCounter.probes == 6, "hash join touched each row once: " + hashCounter.probes);
        check(JoinAlgorithms.sortedByKey(merge, 0), "sort merge emits sorted output");

        SplitMix64 rng = new SplitMix64(20240617L);
        int trials = 60, totalRows = 0;
        for (int trial = 0; trial < trials; trial++) {
            int leftRows = 1 + rng.nextInt(30);
            int rightRows = 1 + rng.nextInt(30);
            int domain = 1 + rng.nextInt(12);
            List<int[]> l = randomTable(rng, leftRows, domain, 1000);
            List<int[]> r = randomTable(rng, rightRows, domain, 2000);
            JoinAlgorithms.Counter c1 = new JoinAlgorithms.Counter();
            JoinAlgorithms.Counter c2 = new JoinAlgorithms.Counter();
            JoinAlgorithms.Counter c3 = new JoinAlgorithms.Counter();
            List<int[]> a = JoinAlgorithms.nestedLoop(l, r, 0, 0, c1);
            List<int[]> b = JoinAlgorithms.hashJoin(l, r, 0, 0, c2);
            List<int[]> c = JoinAlgorithms.sortMergeJoin(l, r, 0, 0, c3);
            check(JoinAlgorithms.canonical(a).equals(JoinAlgorithms.canonical(b)),
                    "loop vs hash in trial " + trial);
            check(JoinAlgorithms.canonical(a).equals(JoinAlgorithms.canonical(c)),
                    "loop vs merge in trial " + trial);
            check(a.size() == JoinAlgorithms.expectedSize(l, r, 0, 0),
                    "aggregation oracle in trial " + trial);
            check(JoinAlgorithms.hasEqualKeys(a, 0, 0, 2) && JoinAlgorithms.hasEqualKeys(c, 0, 0, 2),
                    "key equality in trial " + trial);
            check(JoinAlgorithms.sortedByKey(c, 0), "merge output sorted in trial " + trial);
            check(c1.pairComparisons == (long) l.size() * r.size(), "pair count in trial " + trial);
            check(c2.probes == l.size() + r.size(), "hash touches in trial " + trial);
            totalRows += a.size();
        }
        check(totalRows > 1000, "the trials really produced matches: " + totalRows + " rows");

        List<int[]> denseLeft = randomTable(rng, 60, 3, 0);
        List<int[]> denseRight = randomTable(rng, 60, 3, 100);
        JoinAlgorithms.Counter denseLoop = new JoinAlgorithms.Counter();
        JoinAlgorithms.Counter denseHash = new JoinAlgorithms.Counter();
        JoinAlgorithms.Counter denseMerge = new JoinAlgorithms.Counter();
        List<int[]> denseA = JoinAlgorithms.nestedLoop(denseLeft, denseRight, 0, 0, denseLoop);
        List<int[]> denseB = JoinAlgorithms.hashJoin(denseLeft, denseRight, 0, 0, denseHash);
        List<int[]> denseC = JoinAlgorithms.sortMergeJoin(denseLeft, denseRight, 0, 0, denseMerge);
        check(denseA.size() > 1000, "many-to-many expansion: " + denseA.size() + " rows");
        check(JoinAlgorithms.canonical(denseA).equals(JoinAlgorithms.canonical(denseB))
                        && JoinAlgorithms.canonical(denseA).equals(JoinAlgorithms.canonical(denseC)),
                "duplicate keys are handled identically by all three algorithms");
        check(denseA.size() == JoinAlgorithms.expectedSize(denseLeft, denseRight, 0, 0),
                "dense result size matches the oracle");

        List<int[]> bigLeft = randomTable(rng, 400, 200, 5000);
        List<int[]> bigRight = randomTable(rng, 400, 200, 9000);
        JoinAlgorithms.Counter bigLoop = new JoinAlgorithms.Counter();
        JoinAlgorithms.Counter bigHash = new JoinAlgorithms.Counter();
        JoinAlgorithms.Counter bigMerge = new JoinAlgorithms.Counter();
        List<int[]> bigA = JoinAlgorithms.nestedLoop(bigLeft, bigRight, 0, 0, bigLoop);
        List<int[]> bigB = JoinAlgorithms.hashJoin(bigLeft, bigRight, 0, 0, bigHash);
        List<int[]> bigC = JoinAlgorithms.sortMergeJoin(bigLeft, bigRight, 0, 0, bigMerge);
        check(bigLoop.pairComparisons == 160000, "nested loop did 400*400 pair comparisons: "
                + bigLoop.pairComparisons);
        check(bigHash.probes == 800, "hash join did one touch per row: " + bigHash.probes);
        check(bigHash.keyComparisons < bigLoop.keyComparisons / 40,
                "hash join compares far fewer keys: " + bigHash.keyComparisons + " vs " + bigLoop.keyComparisons);
        check(bigMerge.keyComparisons < bigLoop.keyComparisons / 10,
                "sort merge walks rather than pairs: " + bigMerge.keyComparisons);
        check(bigMerge.sortComparisons > 0 && bigMerge.sortComparisons < bigLookup(bigLeft),
                "sorting cost stays below n^2/2: " + bigMerge.sortComparisons);
        check(JoinAlgorithms.canonical(bigA).equals(JoinAlgorithms.canonical(bigB))
                        && JoinAlgorithms.canonical(bigA).equals(JoinAlgorithms.canonical(bigC)),
                "all three agree on 400x400 inputs");
        check(bigB.size() == JoinAlgorithms.expectedSize(bigLeft, bigRight, 0, 0), "big oracle check");

        List<int[]> empty = new ArrayList<>();
        JoinAlgorithms.Counter emptyCounter = new JoinAlgorithms.Counter();
        check(JoinAlgorithms.nestedLoop(empty, bigRight, 0, 0, emptyCounter).isEmpty(), "empty left");
        check(JoinAlgorithms.hashJoin(bigLeft, empty, 0, 0, emptyCounter).isEmpty(), "empty right");
        check(JoinAlgorithms.sortMergeJoin(empty, empty, 0, 0, emptyCounter).isEmpty(), "both empty");
        check(JoinAlgorithms.expectedSize(empty, bigRight, 0, 0) == 0, "oracle on empty input");

        List<int[]> singleLeft = JoinAlgorithms.table(new int[][]{{7, 1}});
        List<int[]> singleRight = JoinAlgorithms.table(new int[][]{{7, 2}});
        JoinAlgorithms.Counter singleCounter = new JoinAlgorithms.Counter();
        check(JoinAlgorithms.canonical(JoinAlgorithms.nestedLoop(singleLeft, singleRight, 0, 0, singleCounter))
                        .equals(List.of("[7, 1, 7, 2]")), "single row join");
        check(JoinAlgorithms.canonical(JoinAlgorithms.hashJoin(singleLeft, singleRight, 0, 0, singleCounter))
                        .equals(List.of("[7, 1, 7, 2]")), "single row hash join");
        check(JoinAlgorithms.canonical(JoinAlgorithms.sortMergeJoin(singleLeft, singleRight, 0, 0, singleCounter))
                        .equals(List.of("[7, 1, 7, 2]")), "single row sort merge");

        List<int[]> noMatchLeft = JoinAlgorithms.table(new int[][]{{1, 1}, {2, 2}});
        List<int[]> noMatchRight = JoinAlgorithms.table(new int[][]{{3, 3}, {4, 4}});
        JoinAlgorithms.Counter noMatch = new JoinAlgorithms.Counter();
        check(JoinAlgorithms.hashJoin(noMatchLeft, noMatchRight, 0, 0, noMatch).isEmpty(), "no matches");
        check(JoinAlgorithms.sortMergeJoin(noMatchLeft, noMatchRight, 0, 0, noMatch).isEmpty(),
                "sort merge finds no matches either");

        List<int[]> allLeft = JoinAlgorithms.table(new int[][]{{9, 1}, {9, 2}, {9, 3}});
        List<int[]> allRight = JoinAlgorithms.table(new int[][]{{9, 7}, {9, 8}});
        JoinAlgorithms.Counter allCounter = new JoinAlgorithms.Counter();
        check(JoinAlgorithms.hashJoin(allLeft, allRight, 0, 0, allCounter).size() == 6, "full cross product");
        check(JoinAlgorithms.sortMergeJoin(allLeft, allRight, 0, 0, allCounter).size() == 6, "merge cross product");

        List<int[]> negative = JoinAlgorithms.table(new int[][]{{-3, 1}, {-3, 2}, {0, 3}, {5, 4}});
        List<int[]> negativeOther = JoinAlgorithms.table(new int[][]{{-3, 9}, {0, 8}, {4, 7}});
        JoinAlgorithms.Counter negativeCounter = new JoinAlgorithms.Counter();
        List<int[]> negativeResult = JoinAlgorithms.hashJoin(negative, negativeOther, 0, 0, negativeCounter);
        check(negativeResult.size() == 3, "negative and zero keys: " + negativeResult.size());
        check(JoinAlgorithms.canonical(negativeResult).equals(JoinAlgorithms.canonical(
                        JoinAlgorithms.sortMergeJoin(negative, negativeOther, 0, 0, negativeCounter))),
                "negative keys agree between hash and merge");

        JoinAlgorithms.Counter thetaCounter = new JoinAlgorithms.Counter();
        List<int[]> theta = JoinAlgorithms.nestedLoopTheta(
                JoinAlgorithms.table(new int[][]{{1, 0}, {4, 0}, {7, 0}}),
                JoinAlgorithms.table(new int[][]{{2, 0}, {6, 0}, {8, 0}}),
                (l, r) -> Math.abs(l[0] - r[0]) <= 1, thetaCounter);
        check(theta.size() == 3, "theta join |delta| <= 1 matched pairs: " + theta.size());
        check(thetaCounter.pairComparisons == 9, "theta join still scans every pair");

        List<int[]> leftMajor = JoinAlgorithms.nestedLoop(
                JoinAlgorithms.table(new int[][]{{2, 1}, {1, 2}}),
                JoinAlgorithms.table(new int[][]{{1, 3}, {2, 4}}), 0, 0, new JoinAlgorithms.Counter());
        check(leftMajor.size() == 2 && leftMajor.get(0)[1] == 1 && leftMajor.get(1)[1] == 2,
                "nested loop preserves left-major order");

        System.out.println("hand case: 4 rows, all three algorithms agree; nested loop " 
                + loopCounter.pairComparisons + " pairs, hash " + hashCounter.probes + " touches, merge sorted");
        System.out.println("400x400: loop pairs=" + bigLoop.pairComparisons + " hashTouches=" + bigHash.probes
                + " hashKeyChecks=" + bigHash.keyComparisons + " mergeKeyChecks=" + bigMerge.keyComparisons
                + " mergeSortComparisons=" + bigMerge.sortComparisons + " rows=" + bigA.size());
        System.out.println("60 generated trials agreed, " + totalRows + " rows produced in total");
        System.out.println("PASS");
    }

    static long bigLookup(List<int[]> table) {
        return (long) table.size() * table.size() / 2;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
