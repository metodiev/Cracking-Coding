/** Task 644 - Implement query execution engine operator pipelining simulation. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A Volcano-style pull engine. Every operator implements open() / next() / close(); next() returns one
 * tuple or null at end of stream, and the parent pulls from the child. The operators are:
 *
 *   Scan           - streams a stored table, one tuple per next();
 *   Filter         - pulls until the predicate passes, so it can emit several input tuples per output;
 *   Map            - streams a computed column;
 *   Limit          - stops after k tuples and never pulls again, which is what makes demand-driven
 *                    execution visible in the counters;
 *   Sort           - a pipeline *breaker*: open() drains the whole child before the first tuple;
 *   NestedLoopJoin - streams, but re-opens (rewinds) the inner side once per outer tuple;
 *   HashJoin       - a hybrid: the build side is drained in open() (blocking), the probe side streams.
 *
 * Every operator keeps counters (next() calls, tuples in, tuples out, rewinds, build tuples), because
 * the pipeline/blocking distinction is a statement about work done, not about output.
 *
 * Simplified (deliberately): no buffer pool, no spilling, no parallelism, no iterator invalidation and
 * no cost accounting per tuple - the counters are call counts.
 *
 * Verified: for five different query trees the pipelined result equals an independent materialised
 * evaluator's result (which builds plain lists); a Limit above a streaming pipeline stops the scan
 * after five of a thousand rows, while the identical query over a Sort drains all one thousand (the
 * blocking operator really blocks) and a Limit above a HashJoin still drains the build side but not
 * the probe side; the nested-loop join re-opens its inner side exactly once per outer tuple; re-running
 * a query reproduces the counters exactly; and empty input, Limit(0), and a Limit larger than the input
 * behave correctly.
 */
import java.util.*;
import java.util.function.IntPredicate;
import java.util.function.IntUnaryOperator;

class Volcano {

    abstract static class Operator {
        long nextCalls, tuplesIn, tuplesOut, rewinds, buildTuples;

        abstract void open();
        abstract int[] next();
        abstract void close();
        abstract String describe();

        List<int[]> drain() {
            List<int[]> out = new ArrayList<>();
            open();
            int[] tuple;
            while ((tuple = next()) != null) out.add(tuple);
            close();
            return out;
        }
    }

    static int[] concat(int[] left, int[] right) {
        int[] out = new int[left.length + right.length];
        System.arraycopy(left, 0, out, 0, left.length);
        System.arraycopy(right, 0, out, left.length, right.length);
        return out;
    }

    static final class Scan extends Operator {
        final List<int[]> rows;
        final String name;
        int position;

        Scan(String name, List<int[]> rows) { this.name = name; this.rows = rows; }

        void open() { position = 0; }

        int[] next() {
            nextCalls++;
            if (position >= rows.size()) return null;
            tuplesOut++;
            return rows.get(position++);
        }

        void close() { }

        String describe() { return "Scan(" + name + ")"; }
    }

    static final class Filter extends Operator {
        final Operator child;
        final IntPredicate predicate;

        Filter(Operator child, IntPredicate predicate) { this.child = child; this.predicate = predicate; }

        void open() { child.open(); }

        int[] next() {
            while (true) {
                nextCalls++;
                int[] tuple = child.next();
                if (tuple == null) return null;
                tuplesIn++;
                if (predicate.test(tuple[1])) {
                    tuplesOut++;
                    return tuple;
                }
            }
        }

        void close() { child.close(); }

        String describe() { return "Filter(" + child.describe() + ")"; }
    }

    static final class MapOp extends Operator {
        final Operator child;
        final IntUnaryOperator mapper;

        MapOp(Operator child, IntUnaryOperator mapper) { this.child = child; this.mapper = mapper; }

        void open() { child.open(); }

        int[] next() {
            nextCalls++;
            int[] tuple = child.next();
            if (tuple == null) return null;
            tuplesIn++;
            tuplesOut++;
            return new int[]{tuple[0], mapper.applyAsInt(tuple[1])};
        }

        void close() { child.close(); }

        String describe() { return "Map(" + child.describe() + ")"; }
    }

    static final class Limit extends Operator {
        final Operator child;
        final int limit;
        long produced;

        Limit(Operator child, int limit) { this.child = child; this.limit = limit; }

        void open() { child.open(); produced = 0; }

        int[] next() {
            nextCalls++;
            if (produced >= limit) return null;
            int[] tuple = child.next();
            if (tuple == null) return null;
            produced++;
            tuplesOut++;
            return tuple;
        }

        void close() { child.close(); }

        String describe() { return "Limit(" + limit + ", " + child.describe() + ")"; }
    }

    /** Pipeline breaker: open() pulls the entire child into a buffer and merge-sorts it. */
    static final class SortOp extends Operator {
        final Operator child;
        List<int[]> buffer;
        int position;

        SortOp(Operator child) { this.child = child; }

        void open() {
            child.open();
            buffer = new ArrayList<>();
            int[] tuple;
            while ((tuple = child.next()) != null) buffer.add(tuple);
            child.close();
            buffer = mergeSort(buffer);
            position = 0;
        }

        private static List<int[]> mergeSort(List<int[]> rows) {
            if (rows.size() < 2) return rows;
            int mid = rows.size() / 2;
            List<int[]> a = mergeSort(new ArrayList<>(rows.subList(0, mid)));
            List<int[]> b = mergeSort(new ArrayList<>(rows.subList(mid, rows.size())));
            List<int[]> out = new ArrayList<>(rows.size());
            int i = 0, j = 0;
            while (i < a.size() && j < b.size()) out.add(a.get(i)[0] <= b.get(j)[0] ? a.get(i++) : b.get(j++));
            while (i < a.size()) out.add(a.get(i++));
            while (j < b.size()) out.add(b.get(j++));
            return out;
        }

        int[] next() {
            nextCalls++;
            if (position >= buffer.size()) return null;
            tuplesOut++;
            return buffer.get(position++);
        }

        void close() { }

        String describe() { return "Sort(" + child.describe() + ")"; }
    }

    /** Streaming join that rewinds its inner side for every outer tuple. */
    static final class NestedLoopJoin extends Operator {
        final Operator outer, inner;
        int[] currentOuter;
        boolean innerOpen;

        NestedLoopJoin(Operator outer, Operator inner) { this.outer = outer; this.inner = inner; }

        void open() {
            outer.open();
            currentOuter = null;
            innerOpen = false;
        }

        int[] next() {
            while (true) {
                nextCalls++;
                if (currentOuter == null) {
                    currentOuter = outer.next();
                    if (currentOuter == null) return null;
                    if (innerOpen) {
                        inner.close();
                        innerOpen = false;
                    }
                    inner.open();
                    innerOpen = true;
                    rewinds++;
                }
                int[] candidate;
                while ((candidate = inner.next()) != null) {
                    if (candidate[0] == currentOuter[0]) {
                        tuplesOut++;
                        return concat(currentOuter, candidate);
                    }
                }
                inner.close();
                innerOpen = false;
                currentOuter = null;
            }
        }

        void close() {
            outer.close();
            if (innerOpen) {
                inner.close();
                innerOpen = false;
            }
        }

        String describe() { return "NLJ(" + outer.describe() + ", " + inner.describe() + ")"; }
    }

    /** Hybrid: the build side is drained in open(), the probe side streams per next(). */
    static final class HashJoin extends Operator {
        final Operator build, probe;
        final Map<Integer, List<int[]>> buckets = new HashMap<>();
        int[] currentProbe;
        Iterator<int[]> pending;

        HashJoin(Operator build, Operator probe) { this.build = build; this.probe = probe; }

        void open() {
            buckets.clear();
            build.open();
            int[] tuple;
            while ((tuple = build.next()) != null) {
                buckets.computeIfAbsent(tuple[0], key -> new ArrayList<>()).add(tuple);
                buildTuples++;
            }
            build.close();
            probe.open();
            currentProbe = null;
            pending = null;
        }

        int[] next() {
            while (true) {
                nextCalls++;
                if (pending != null && pending.hasNext()) {
                    tuplesOut++;
                    return concat(currentProbe, pending.next());
                }
                currentProbe = probe.next();
                if (currentProbe == null) return null;
                pending = buckets.getOrDefault(currentProbe[0], List.of()).iterator();
            }
        }

        void close() { probe.close(); }

        String describe() { return "HashJoin(" + build.describe() + ", " + probe.describe() + ")"; }
    }

    // ---------- independent materialised evaluator, used as the oracle ----------

    static List<int[]> materialisedFilter(List<int[]> rows, IntPredicate predicate) {
        List<int[]> out = new ArrayList<>();
        for (int[] row : rows) if (predicate.test(row[1])) out.add(row);
        return out;
    }

    static List<int[]> materialisedMap(List<int[]> rows, IntUnaryOperator mapper) {
        List<int[]> out = new ArrayList<>();
        for (int[] row : rows) out.add(new int[]{row[0], mapper.applyAsInt(row[1])});
        return out;
    }

    static List<int[]> materialisedNestedLoop(List<int[]> outer, List<int[]> inner) {
        List<int[]> out = new ArrayList<>();
        for (int[] l : outer) for (int[] r : inner) if (l[0] == r[0]) out.add(concat(l, r));
        return out;
    }

    static List<int[]> materialisedSort(List<int[]> rows) {
        List<int[]> copy = new ArrayList<>(rows);
        copy.sort((a, b) -> Integer.compare(a[0], b[0]));
        return copy;
    }

    static List<int[]> first(List<int[]> rows, int limit) {
        return new ArrayList<>(rows.subList(0, Math.min(limit, rows.size())));
    }

    static List<String> canonical(List<int[]> rows) {
        List<String> out = new ArrayList<>();
        for (int[] row : rows) out.add(Arrays.toString(row));
        Collections.sort(out);
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

    static List<int[]> generate(SplitMix64 rng, int rows, int keyDomain) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < rows; i++) out.add(new int[]{rng.nextInt(keyDomain), i});
        return out;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(998877L);
        List<int[]> left = generate(rng, 1000, 50);
        List<int[]> right = generate(rng, 300, 50);

        Volcano.Scan leftScan = new Volcano.Scan("left", left);
        Volcano.Scan rightScan = new Volcano.Scan("right", right);
        Volcano.Operator filtered = new Volcano.Filter(leftScan, value -> value % 100 == 0);
        List<int[]> pipelined = filtered.drain();
        List<int[]> materialised = Volcano.materialisedFilter(left, value -> value % 100 == 0);
        check(Volcano.canonical(pipelined).equals(Volcano.canonical(materialised)),
                "filter pipeline equals the materialised evaluator");
        check(leftScan.tuplesOut == 1000, "a full drain consumes every row: " + leftScan.tuplesOut);
        check(filtered.tuplesIn == leftScan.tuplesOut, "the filter saw every produced row");
        check(pipelined.size() == 10, "expected ten qualifying rows: " + pipelined.size());

        Volcano.Scan smallLeft = new Volcano.Scan("l", left);
        Volcano.Operator joined = new Volcano.NestedLoopJoin(smallLeft, new Volcano.Scan("r", right));
        List<int[]> loopResult = joined.drain();
        List<int[]> loopOracle = Volcano.materialisedNestedLoop(left, right);
        check(Volcano.canonical(loopResult).equals(Volcano.canonical(loopOracle)),
                "blocking nested loop equals the oracle");
        check(joined.rewinds == smallLeft.tuplesOut,
                "the inner side is rewound once per outer tuple: " + joined.rewinds + " vs " + smallLeft.tuplesOut);
        check(!loopResult.isEmpty(), "the join produced rows: " + loopResult.size());

        Volcano.Operator hashJoined = new Volcano.HashJoin(new Volcano.Scan("b", right), new Volcano.Scan("p", left));
        List<int[]> hashResult = hashJoined.drain();
        check(Volcano.canonical(hashResult).equals(Volcano.canonical(loopOracle)),
                "hash join and nested loop agree");
        check(hashJoined.buildTuples == 300, "the build side was drained completely: " + hashJoined.buildTuples);

        Volcano.Operator mapping = new Volcano.MapOp(new Volcano.Filter(new Volcano.Scan("left", left),
                value -> value % 7 == 0), value -> value * 2 + 1);
        List<int[]> mapPipeline = mapping.drain();
        List<int[]> mapOracle = Volcano.materialisedMap(
                Volcano.materialisedFilter(left, value -> value % 7 == 0), value -> value * 2 + 1);
        check(Volcano.canonical(mapPipeline).equals(Volcano.canonical(mapOracle)),
                "map over filter equals the oracle");
        check(mapPipeline.size() == mapOracle.size(), "row counts agree");

        Volcano.Scan sortedScan = new Volcano.Scan("left", left);
        Volcano.Operator sorted = new Volcano.SortOp(sortedScan);
        List<int[]> sortedRows = sorted.drain();
        boolean sortedOk = true;
        for (int i = 1; i < sortedRows.size(); i++) if (sortedRows.get(i - 1)[0] > sortedRows.get(i)[0]) sortedOk = false;
        check(sortedOk, "sort output is ordered by key");
        check(sortedRows.size() == 1000, "sort does not lose rows: " + sortedRows.size());
        check(sortedScan.tuplesOut == 1000, "the sort drained its child");

        Volcano.Scan limitScan = new Volcano.Scan("left", left);
        Volcano.Operator limited = new Volcano.Limit(
                new Volcano.Filter(limitScan, value -> value >= 0), 5);
        List<int[]> early = limited.drain();
        check(early.size() == 5, "limit returned five rows: " + early.size());
        check(limitScan.tuplesOut == 5, "demand-driven limit stopped the scan after five rows: "
                + limitScan.tuplesOut);
        check(limitScan.nextCalls == 5, "the scan was never asked for a sixth row: " + limitScan.nextCalls);
        check(Volcano.canonical(early).equals(Volcano.canonical(Volcano.first(left, 5))),
                "limit returns the first rows in order");

        Volcano.Scan breakerScan = new Volcano.Scan("left", left);
        Volcano.Operator overSort = new Volcano.Limit(new Volcano.SortOp(breakerScan), 5);
        List<int[]> sortedTop = overSort.drain();
        check(sortedTop.size() == 5, "limit over sort returns five rows");
        check(breakerScan.tuplesOut == 1000, "the blocking sort drained all 1000 rows before the limit applied: "
                + breakerScan.tuplesOut);
        check(Volcano.canonical(sortedTop).equals(
                        Volcano.canonical(Volcano.first(Volcano.materialisedSort(left), 5))),
                "top-5 by key matches the oracle");

        Volcano.Scan buildScan = new Volcano.Scan("b", right);
        Volcano.Scan probeScan = new Volcano.Scan("p", left);
        Volcano.Operator limitedJoin = new Volcano.Limit(new Volcano.HashJoin(buildScan, probeScan), 5);
        List<int[]> joinTop = limitedJoin.drain();
        check(joinTop.size() == 5, "limit over hash join returned five rows");
        check(buildScan.tuplesOut == 300, "the build side is blocking and drained fully: " + buildScan.tuplesOut);
        check(probeScan.tuplesOut < 1000 && probeScan.tuplesOut > 0,
                "the probe side streamed and stopped early: " + probeScan.tuplesOut + " of 1000 rows");

        Volcano.Scan repeatScan = new Volcano.Scan("left", left);
        Volcano.Operator repeatQuery = new Volcano.Limit(new Volcano.Filter(repeatScan, value -> value >= 0), 5);
        repeatQuery.drain();
        check(repeatScan.tuplesOut == limitScan.tuplesOut && repeatScan.nextCalls == limitScan.nextCalls,
                "a repeat run reproduces the counters exactly");

        Volcano.Scan emptyScan = new Volcano.Scan("empty", List.of());
        Volcano.Operator overEmpty = new Volcano.Limit(new Volcano.Filter(emptyScan, value -> true), 5);
        check(overEmpty.drain().isEmpty(), "empty input produces no tuples");
        check(emptyScan.nextCalls == 1, "an empty scan is asked once: " + emptyScan.nextCalls);

        Volcano.Scan zeroScan = new Volcano.Scan("left", left);
        Volcano.Operator zeroLimit = new Volcano.Limit(zeroScan, 0);
        check(zeroLimit.drain().isEmpty(), "limit 0 produces nothing");
        check(zeroScan.nextCalls == 0 && zeroScan.tuplesOut == 0, "limit 0 never pulls a tuple: "
                + zeroScan.nextCalls);

        Volcano.Scan bigLimitScan = new Volcano.Scan("left", left);
        Volcano.Operator bigLimit = new Volcano.Limit(bigLimitScan, 5000);
        check(bigLimit.drain().size() == 1000, "a limit larger than the input is transparent");
        check(bigLimitScan.tuplesOut == 1000, "and the scan still produced everything");

        Volcano.Operator emptyJoin = new Volcano.HashJoin(new Volcano.Scan("b", List.of()), new Volcano.Scan("p", left));
        check(emptyJoin.drain().isEmpty(), "an empty build side joins to nothing");
        check(emptyJoin.buildTuples == 0, "no build tuples for an empty build side");

        System.out.println("pipelined trees matched the materialised oracle: filter, map, " 
                + "nested loop, hash join, sort, limit");
        System.out.println("limit(5) over a streaming pipeline scanned " + limitScan.tuplesOut
                + " rows; over a sort it scanned " + breakerScan.tuplesOut
                + "; over a hash join the build drained " + buildScan.tuplesOut
                + " and the probe streamed " + probeScan.tuplesOut);
        System.out.println("nested loop rewound its inner side " + joined.rewinds + " times");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
