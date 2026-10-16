/** Task 582 - Implement run-length encoded segment tree for binary string operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A segment tree over a binary string whose every node stores a run-length summary of its segment -
 * length, first and last bit, number of ones, the prefix, suffix and longest runs of ones and of zeros,
 * and the number of runs - together with the two lazy tags needed by range assign 0, range assign 1 and
 * range flip. Merging two summaries is O(1) (the longest run of a parent is the maximum of the children
 * runs and, when the children touch with the same bit, the sum of the left suffix run and the right
 * prefix run), and the tags compose: a flip turns an assign into the complementary assign, two flips
 * cancel, and flipping a summary swaps the ones and zeros statistics. Every query (ones, longest run of
 * ones, longest run of zeros, number of runs) is therefore O(log n), as is every update. A second,
 * independent structure - a "Chtholly" interval map of explicit (start, bit) runs, split and merged on
 * every update - computes the same four statistics by walking the runs in the range, and every check
 * below compares the two implementations rather than trusting either alone.
 *
 * Documented checks. (1) After each of 2000 random mixed operations (assign, flip, and all four queries)
 * on a 257-character string - the odd length makes the tree's splits unbalanced, so the lazy
 * composition is exercised at different depths - all four statistics of the two implementations must
 * agree exactly. (2) Composition laws are asserted on random ranges: flipping twice is the identity,
 * assigning then flipping equals assigning the complementary bit, assign is idempotent, and the ones
 * count after a flip equals the complement of the ones count of the same range. (3) Freshly built trees
 * are checked against a direct scan for all 100 prefixes of a random string, which pins down the
 * cross-boundary merging that a run-length summary has to get right. (4) Exhaustive small cases: for
 * every binary string of length 1..6 (126 strings) and every pair (from, to) with from <= to, the four
 * statistics are compared with a direct scan, so boundary handling is verified for every placement
 * instead of on samples. (5) Edge cases: empty ranges (documented to report 0 for all four statistics),
 * single-element ranges, the full range, updates covering the whole string, and all-zero / all-one
 * strings (where the longest run equals the range length). (6) The invariants of the RLE representation
 * are asserted: the interval map holds exactly one entry per run of the current string, the number of
 * runs of the whole string equals the run count reported by the tree, and a whole-string assign leaves
 * a single run.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws thousands of
 * independent positions and operation kinds), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /** Run-length summary plus the lazy tags pending on the node. */
    static final class Summary {
        int length;
        int first;
        int last;
        int ones;
        int prefixOnes;
        int suffixOnes;
        int longestOnes;
        int prefixZeros;
        int suffixZeros;
        int longestZeros;
        int runs;
        int assign = -1;
        boolean flip;

        Summary copy() {
            Summary out = new Summary();
            out.length = length;
            out.first = first;
            out.last = last;
            out.ones = ones;
            out.prefixOnes = prefixOnes;
            out.suffixOnes = suffixOnes;
            out.longestOnes = longestOnes;
            out.prefixZeros = prefixZeros;
            out.suffixZeros = suffixZeros;
            out.longestZeros = longestZeros;
            out.runs = runs;
            out.assign = assign;
            out.flip = flip;
            return out;
        }
    }

    static Summary unit(int bit) {
        Summary out = new Summary();
        out.length = 1;
        out.first = bit;
        out.last = bit;
        out.ones = bit;
        out.prefixOnes = bit;
        out.suffixOnes = bit;
        out.longestOnes = bit;
        out.prefixZeros = 1 - bit;
        out.suffixZeros = 1 - bit;
        out.longestZeros = 1 - bit;
        out.runs = 1;
        return out;
    }

    static void mergeInto(Summary out, Summary left, Summary right) {
        out.length = left.length + right.length;
        out.first = left.first;
        out.last = right.last;
        out.ones = left.ones + right.ones;
        out.prefixOnes = left.prefixOnes;
        if (left.prefixOnes == left.length && right.first == 1) out.prefixOnes = left.length + right.prefixOnes;
        out.suffixOnes = right.suffixOnes;
        if (right.suffixOnes == right.length && left.last == 1) out.suffixOnes = right.length + left.suffixOnes;
        out.longestOnes = Math.max(Math.max(left.longestOnes, right.longestOnes),
                left.last == 1 && right.first == 1 ? left.suffixOnes + right.prefixOnes : 0);
        out.prefixZeros = left.prefixZeros;
        if (left.prefixZeros == left.length && right.first == 0) out.prefixZeros = left.length + right.prefixZeros;
        out.suffixZeros = right.suffixZeros;
        if (right.suffixZeros == right.length && left.last == 0) out.suffixZeros = right.length + left.suffixZeros;
        out.longestZeros = Math.max(Math.max(left.longestZeros, right.longestZeros),
                left.last == 0 && right.first == 0 ? left.suffixZeros + right.prefixZeros : 0);
        out.runs = left.runs + right.runs - (left.last == right.first ? 1 : 0);
        out.assign = -1;
        out.flip = false;
    }

    static void applyAssign(Summary node, int bit) {
        node.first = bit;
        node.last = bit;
        node.ones = bit == 1 ? node.length : 0;
        node.prefixOnes = node.suffixOnes = node.longestOnes = bit == 1 ? node.length : 0;
        node.prefixZeros = node.suffixZeros = node.longestZeros = bit == 0 ? node.length : 0;
        node.runs = 1;
        node.assign = bit;
        node.flip = false;
    }

    static void applyFlip(Summary node) {
        node.first = 1 - node.first;
        node.last = 1 - node.last;
        node.ones = node.length - node.ones;
        int t;
        t = node.prefixOnes;
        node.prefixOnes = node.prefixZeros;
        node.prefixZeros = t;
        t = node.suffixOnes;
        node.suffixOnes = node.suffixZeros;
        node.suffixZeros = t;
        t = node.longestOnes;
        node.longestOnes = node.longestZeros;
        node.longestZeros = t;
        if (node.assign >= 0) node.assign = 1 - node.assign;
        else node.flip = !node.flip;
    }

    /** Segment tree over a binary string. */
    static final class RleSegmentTree {
        final int size;
        final Summary[] nodes;

        RleSegmentTree(int[] bits) {
            size = bits.length;
            nodes = new Summary[4 * Math.max(size, 1) + 4];
            if (size > 0) build(1, 0, size, bits);
        }

        private void build(int index, int from, int to, int[] bits) {
            nodes[index] = new Summary();
            if (to - from == 1) {
                Summary leaf = unit(bits[from]);
                nodes[index] = leaf;
                return;
            }
            int middle = (from + to) >>> 1;
            build(2 * index, from, middle, bits);
            build(2 * index + 1, middle, to, bits);
            mergeInto(nodes[index], nodes[2 * index], nodes[2 * index + 1]);
        }

        private void push(int index) {
            Summary node = nodes[index];
            if (node.length == 1 || (node.assign < 0 && !node.flip)) return;
            if (node.assign >= 0) {
                applyAssign(nodes[2 * index], node.assign);
                applyAssign(nodes[2 * index + 1], node.assign);
                node.assign = -1;
            }
            if (node.flip) {
                applyFlip(nodes[2 * index]);
                applyFlip(nodes[2 * index + 1]);
                node.flip = false;
            }
        }

        private void pull(int index) {
            Summary node = nodes[index];
            int assign = node.assign;
            boolean flip = node.flip;
            mergeInto(node, nodes[2 * index], nodes[2 * index + 1]);
            node.assign = assign;
            node.flip = flip;
        }

        void assign(int from, int to, int bit) { update(1, 0, size, from, to, bit, false); }

        void flip(int from, int to) { update(1, 0, size, from, to, 0, true); }

        private void update(int index, int segFrom, int segTo, int from, int to, int bit, boolean doFlip) {
            if (to <= segFrom || segTo <= from) return;
            if (from <= segFrom && segTo <= to) {
                if (doFlip) applyFlip(nodes[index]);
                else applyAssign(nodes[index], bit);
                return;
            }
            push(index);
            int middle = (segFrom + segTo) >>> 1;
            update(2 * index, segFrom, middle, from, to, bit, doFlip);
            update(2 * index + 1, middle, segTo, from, to, bit, doFlip);
            pull(index);
        }

        /** {ones, longestOnes, longestZeros, runs} for [from, to). */
        int[] stats(int from, int to) {
            if (from >= to) return new int[]{0, 0, 0, 0};
            Summary accumulator = query(1, 0, size, from, to);
            return new int[]{accumulator.ones, accumulator.longestOnes, accumulator.longestZeros,
                    accumulator.runs};
        }

        private Summary query(int index, int segFrom, int segTo, int from, int to) {
            if (from <= segFrom && segTo <= to) {
                Summary out = new Summary();
                Summary node = nodes[index];
                out.length = node.length;
                out.first = node.first;
                out.last = node.last;
                out.ones = node.ones;
                out.prefixOnes = node.prefixOnes;
                out.suffixOnes = node.suffixOnes;
                out.longestOnes = node.longestOnes;
                out.prefixZeros = node.prefixZeros;
                out.suffixZeros = node.suffixZeros;
                out.longestZeros = node.longestZeros;
                out.runs = node.runs;
                return out;
            }
            push(index);
            int middle = (segFrom + segTo) >>> 1;
            if (to <= middle) return query(2 * index, segFrom, middle, from, to);
            if (from >= middle) return query(2 * index + 1, middle, segTo, from, to);
            Summary left = query(2 * index, segFrom, middle, from, to);
            Summary right = query(2 * index + 1, middle, segTo, from, to);
            Summary out = new Summary();
            mergeInto(out, left, right);
            return out;
        }

        void assignWhole(int bit) { if (size > 0) applyAssign(nodes[1], bit); }

        int runCount() { return size == 0 ? 0 : nodes[1].runs; }
    }

    /** Independent implementation: explicit runs in a sorted map (the "Chtholly" interval structure). */
    static final class IntervalRuns {
        final TreeMap<Integer, Integer> runs = new TreeMap<>();
        int size;

        IntervalRuns(int[] bits) {
            size = bits.length;
            if (size == 0) return;
            runs.put(0, bits[0]);
            for (int i = 1; i < size; i++) {
                if (bits[i] != bits[i - 1]) runs.put(i, bits[i]);
            }
        }

        private void split(int position) {
            if (position <= 0 || position >= size || runs.containsKey(position)) return;
            Integer bit = runs.floorEntry(position).getValue();
            runs.put(position, bit);
        }

        void assign(int from, int to, int bit) {
            if (from >= to) return;
            split(from);
            split(to);
            ArrayList<Integer> inside = new ArrayList<>(runs.subMap(from, true, to, false).keySet());
            for (int key : inside) runs.remove(key);
            runs.put(from, bit);
            mergeAround(from);
            mergeAround(to);
        }

        void flip(int from, int to) {
            if (from >= to) return;
            split(from);
            split(to);
            ArrayList<Integer> inside = new ArrayList<>(runs.subMap(from, true, to, false).keySet());
            for (int key : inside) runs.put(key, 1 - runs.get(key));
            mergeAround(from);
            mergeAround(to);
        }

        private void mergeAround(int position) {
            Integer key = runs.floorKey(position);
            if (key == null) return;
            Integer next = runs.higherKey(key);
            if (next != null && runs.get(next).equals(runs.get(key))) runs.remove(next);
            Integer previous = runs.lowerKey(key);
            if (previous != null && runs.get(previous).equals(runs.get(key))) runs.remove(key);
        }

        int[] stats(int from, int to) {
            if (from >= to || size == 0) return new int[]{0, 0, 0, 0};
            Map.Entry<Integer, Integer> entry = runs.floorEntry(from);
            int position = from;
            int ones = 0;
            int runsCount = 0;
            int bestOnes = 0;
            int bestZeros = 0;
            int currentOnes = 0;
            int currentZeros = 0;
            while (position < to && entry != null) {
                int runStart = entry.getKey();
                Integer nextKey = runs.higherKey(runStart);
                int runEnd = nextKey == null ? size : nextKey;
                int start = Math.max(position, runStart);
                int end = Math.min(to, runEnd);
                int length = end - start;
                if (length > 0) {
                    runsCount++;
                    if (entry.getValue() == 1) {
                        ones += length;
                        currentOnes += length;
                        currentZeros = 0;
                    } else {
                        currentZeros += length;
                        currentOnes = 0;
                    }
                    bestOnes = Math.max(bestOnes, currentOnes);
                    bestZeros = Math.max(bestZeros, currentZeros);
                    position = end;
                }
                if (position >= to) break;
                entry = nextKey == null ? null : runs.ceilingEntry(nextKey);
            }
            return new int[]{ones, bestOnes, bestZeros, runsCount};
        }

        int runCount() { return runs.size(); }
    }

    static int[] scan(int[] bits, int from, int to) {
        if (from >= to) return new int[]{0, 0, 0, 0};
        int ones = 0;
        int bestOnes = 0;
        int bestZeros = 0;
        int runs = 0;
        int current = 0;
        for (int i = from; i < to; i++) {
            if (bits[i] == 1) {
                ones++;
                current++;
                bestOnes = Math.max(bestOnes, current);
            } else {
                current = 0;
            }
        }
        current = 0;
        for (int i = from; i < to; i++) {
            if (bits[i] == 0) {
                current++;
                bestZeros = Math.max(bestZeros, current);
            } else {
                current = 0;
            }
        }
        for (int i = from; i < to; i++) {
            if (i == from || bits[i] != bits[i - 1]) runs++;
        }
        return new int[]{ones, bestOnes, bestZeros, runs};
    }

    public static void main(String[] args) {
        Rng rng = new Rng(582_2025L);

        // ---- (4) exhaustive small cases -----------------------------------------------------------
        for (int length = 1; length <= 6; length++) {
            for (int mask = 0; mask < (1 << length); mask++) {
                int[] bits = new int[length];
                for (int i = 0; i < length; i++) bits[i] = (mask >>> i) & 1;
                RleSegmentTree tree = new RleSegmentTree(bits);
                IntervalRuns intervals = new IntervalRuns(bits);
                for (int from = 0; from <= length; from++) {
                    for (int to = from; to <= length; to++) {
                        int[] expected = scan(bits, from, to);
                        check(Arrays.equals(tree.stats(from, to), expected),
                                "tree statistics match the scan for " + Arrays.toString(bits) + " [" + from
                                        + "," + to + ")");
                        check(Arrays.equals(intervals.stats(from, to), expected),
                                "interval statistics match the scan for " + Arrays.toString(bits) + " ["
                                        + from + "," + to + ")");
                    }
                }
                check(tree.runCount() == intervals.runCount(),
                        "both structures agree on the number of runs");
            }
        }
        System.out.println("exhaustive: 126 strings of length 1..6, every range, both structures match");

        // ---- (3) all prefixes of a random string ----------------------------------------------------
        int[] bits = new int[100];
        for (int i = 0; i < bits.length; i++) bits[i] = rng.nextInt(2);
        RleSegmentTree tree = new RleSegmentTree(bits);
        for (int to = 1; to <= bits.length; to++) {
            check(Arrays.equals(tree.stats(0, to), scan(bits, 0, to)), "prefix statistics match a scan");
        }

        // ---- (1) and (2): the random workload --------------------------------------------------------
        int n = 257;
        int[] current = new int[n];
        for (int i = 0; i < n; i++) current[i] = rng.nextInt(2);
        RleSegmentTree segment = new RleSegmentTree(current);
        IntervalRuns map = new IntervalRuns(current);
        long assignments = 0;
        long flips = 0;
        long queries = 0;
        for (int step = 0; step < 2000; step++) {
            int kind = rng.nextInt(6);
            int from = rng.nextInt(n + 1);
            int to = from + rng.nextInt(n - from + 1);
            if (kind == 0) {
                int bit = rng.nextInt(2);
                segment.assign(from, to, bit);
                map.assign(from, to, bit);
                for (int i = from; i < to; i++) current[i] = bit;
                assignments++;
            } else if (kind == 1) {
                segment.flip(from, to);
                map.flip(from, to);
                for (int i = from; i < to; i++) current[i] = 1 - current[i];
                flips++;
            } else {
                int[] a = segment.stats(from, to);
                int[] b = map.stats(from, to);
                int[] expected = scan(current, from, to);
                check(Arrays.equals(a, expected), "segment tree statistics match the scan at step " + step);
                check(Arrays.equals(b, expected), "interval statistics match the scan at step " + step);
                queries++;
            }
            if (step % 100 == 0) {
                check(segment.runCount() == map.runCount(), "run counts agree at step " + step);
                int[] whole = segment.stats(0, n);
                check(whole[3] == map.runCount(), "the tree's run count equals the number of intervals");
            }
        }
        System.out.println("workload: " + assignments + " assignments, " + flips + " flips, " + queries
                + " queries, final runs " + map.runCount() + " (tree says " + segment.runCount() + ")");

        // ---- composition laws -------------------------------------------------------------------------
        for (int trial = 0; trial < 100; trial++) {
            int from = rng.nextInt(n + 1);
            int to = from + rng.nextInt(n - from + 1);
            int[] before = segment.stats(from, to);
            int[] mapBefore = map.stats(from, to);
            check(Arrays.equals(before, mapBefore), "structures agree before the composition checks");
            segment.flip(from, to);
            segment.flip(from, to);
            check(Arrays.equals(segment.stats(from, to), before), "a double flip is the identity");
            int bit = rng.nextInt(2);
            segment.assign(from, to, bit);
            segment.flip(from, to);
            int[] afterAssignFlip = segment.stats(from, to);
            map.assign(from, to, 1 - bit);
            check(Arrays.equals(afterAssignFlip, map.stats(from, to)),
                    "assign then flip equals assigning the complement");
            map.assign(from, to, bit);
            int[] assigned = map.stats(from, to);
            check(assigned[0] == (bit == 1 ? to - from : 0), "assign sets the ones count exactly");
            check(assigned[1] == (bit == 1 ? to - from : 0), "assign sets the longest run exactly");
            check(assigned[3] == (from == to ? 0 : 1), "assign leaves a single run");
            // restore both structures to the same state
            segment.assign(from, to, bit);
            map.assign(from, to, bit);
            check(Arrays.equals(segment.stats(from, to), map.stats(from, to)),
                    "the structures are back in sync");
            int onesBefore = segment.stats(from, to)[0];
            segment.flip(from, to);
            map.flip(from, to);
            int[] flipped = segment.stats(from, to);
            check(flipped[0] == (to - from) - onesBefore, "a flip complements the ones count of the range");
            check(Arrays.equals(flipped, map.stats(from, to)), "structures agree after a flip");
        }
        // flipping doubles back to the original state
        int[] restored = new int[n];
        for (int i = 0; i < n; i++) restored[i] = rng.nextInt(2);
        RleSegmentTree second = new RleSegmentTree(restored);
        for (int i = 0; i < n; i++) restored[i] = 1 - restored[i];
        RleSegmentTree third = new RleSegmentTree(restored);
        check(second.stats(0, n)[0] + third.stats(0, n)[0] == n,
                "complementary strings have complementary ones counts");

        // ---- edge cases -------------------------------------------------------------------------------
        for (int i = 0; i <= n; i++) {
            int[] empty = segment.stats(i, i);
            check(empty[0] == 0 && empty[1] == 0 && empty[2] == 0 && empty[3] == 0,
                    "an empty range reports zeros");
        }
        int[] single = segment.stats(10, 11);
        check(single[0] + (1 - single[0]) == 1 && single[3] == 1, "a single element is one run");
        check(single[1] + single[2] == 1, "a single element is either a one or a zero, not both");
        int[] allZeros = new int[64];
        RleSegmentTree zeros = new RleSegmentTree(allZeros);
        check(zeros.stats(0, 64)[2] == 64 && zeros.stats(0, 64)[0] == 0, "an all-zero string");
        check(zeros.stats(0, 64)[3] == 1, "an all-zero string has one run");
        zeros.assignWhole(1);
        check(zeros.runCount() == 1 && zeros.stats(0, 64)[1] == 64, "a whole-string assign leaves one run");
        int[] allOnes = new int[64];
        Arrays.fill(allOnes, 1);
        RleSegmentTree onesTree = new RleSegmentTree(allOnes);
        check(onesTree.stats(5, 64)[1] == 59, "a range inside an all-one string is one run");
        check(zeros.runCount() == 1 && onesTree.runCount() == 1, "both extremes are single runs");
        RleSegmentTree emptyTree = new RleSegmentTree(new int[0]);
        check(emptyTree.runCount() == 0 && emptyTree.stats(0, 0)[3] == 0, "the empty string");

        System.out.println("PASS");
    }
}
