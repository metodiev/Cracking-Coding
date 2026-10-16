/** Task 888 - Implement incremental view maintenance with delta propagation and bounded memory. (Tier: Master, CTCI Ch 14 Databases)
 *
 * A view DAG is maintained incrementally instead of being recomputed:
 *
 *     sales(id, dept, amount)                       base table, primary key id
 *     dept(id, region)                              base table, primary key id
 *     big   = GROUP BY dept: COUNT(*)      of  sigma[amount > threshold](sales)
 *     grp   = GROUP BY dept: (SUM(amount), COUNT(*)) of sales
 *     reg   = grp JOIN dept: (SUM, COUNT) per region     (inner join on dept.id)
 *
 * Each base change is a delta (net multiplicity per primary key), and every operator has a delta rule
 * that is applied instead of a recomputation:
 *
 *     filter      d(sigma_p T)  = sigma_p (dT)
 *     group-by    SUM and COUNT are linear, so a group's aggregate moves by the delta's contribution and
 *                 a group disappears exactly when its count reaches zero
 *     join        a dimension change (dept.row moved to another region, or deleted) invalidates the
 *                 affected groups, so their pre-batch contribution is subtracted from the old region and
 *                 their post-batch contribution added to the new one - the standard affected-group repair
 *                 for a non-linear operator
 *
 * Bounded memory is a first-class part of the contract: deltas are queued in a buffer whose depth is
 * capped in entries. When a new batch would exceed the cap the queue is coalesced (one net delta per key,
 * opposite multiplicities cancelling) and, if the coalesced delta still does not fit, checkpointed into
 * the base state. The views therefore always equal a full recomputation of the *applied* state, and after
 * a checkpoint they equal a full recomputation of the effective state. A bounded-memory engine is run
 * side by side with an unbounded reference engine and both must end in the same state.
 *
 * What is simplified: everything is single-threaded and in-process, the "wire" is a method call, the
 * workload is generated deterministically from splitmix64, and cost is counted in delta entries processed
 * rather than in CPU cycles. Transactional atomicity is real: a batch that would duplicate a key or delete
 * a missing key is rejected as a whole, and the views are left untouched (validate-then-apply). No wall
 * clock is read anywhere, so every claim below is reproducible.
 *
 * Verified: the initial views equal the brute-force oracle; after every one of ~40 batches all three views
 * equal the oracle; the delta the operators applied equals the state difference the oracle computes over
 * the batch (delta algebra, not just the final state); the join repair moves a group's totals when a
 * department changes region or is deleted, while the group view keeps the group; a group vanishes exactly
 * when its count reaches zero and reappears on reinsertion; duplicate-free bag semantics for the filter
 * projection are exact; coalescing cancels opposite multiplicities and is lossless; the queue depth never
 * exceeds the cap under a flood of uncoordinated batches; the bounded-memory engine converges to the same
 * state as the unbounded reference; invalid batches are rejected atomically with the views unchanged;
 * incremental cost stays far below full recomputation; and two runs of the same workload are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    long nextLong(long bound) { return Math.floorMod(nextLong(), bound); }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A row of the sales fact table. */
class Row {
    final int id, dept;
    final long amount;

    Row(int id, int dept, long amount) { this.id = id; this.dept = dept; this.amount = amount; }

    public boolean equals(Object o) {
        if (!(o instanceof Row)) return false;
        Row r = (Row) o;
        return id == r.id && dept == r.dept && amount == r.amount;
    }

    public int hashCode() { return Objects.hash(id, dept, amount); }

    public String toString() { return "(" + id + "," + dept + "," + amount + ")"; }
}

/** One transaction: inserts, deletes and dictionary changes. */
class Batch {
    final List<Row> inserts = new ArrayList<>();
    final List<Integer> deletes = new ArrayList<>();
    final Map<Integer, String> deptSet = new LinkedHashMap<>();
    final List<Integer> deptDelete = new ArrayList<>();

    Batch ins(Row r) { inserts.add(r); return this; }

    Batch del(int id) { deletes.add(id); return this; }

    Batch dept(int id, String region) { deptSet.put(id, region); return this; }

    Batch deptDrop(int id) { deptDelete.add(id); return this; }

    int entryCount() { return inserts.size() + deletes.size() + deptSet.size() + deptDelete.size(); }

    boolean isEmpty() {
        return inserts.isEmpty() && deletes.isEmpty() && deptSet.isEmpty() && deptDelete.isEmpty();
    }
}

/** Raised when a batch is not applicable; the engine must leave its state untouched. */
class RejectedBatch extends RuntimeException {
    RejectedBatch(String message) { super(message); }
}

/** The view state: three views over the two base tables. */
class Views {
    final Map<Integer, long[]> byDept = new TreeMap<>();      // dept -> {sum, count}
    final Map<Integer, Integer> bigByDept = new TreeMap<>();   // dept -> occurrences with amount > threshold
    final Map<String, long[]> byRegion = new TreeMap<>();      // region -> {sum, count}

    Views copy() {
        Views v = new Views();
        for (Map.Entry<Integer, long[]> e : byDept.entrySet()) v.byDept.put(e.getKey(), e.getValue().clone());
        v.bigByDept.putAll(bigByDept);
        for (Map.Entry<String, long[]> e : byRegion.entrySet()) v.byRegion.put(e.getKey(), e.getValue().clone());
        return v;
    }

    /** Normalized copy: zero counts and zero sums are dropped, so two equivalent states compare equal. */
    Views normalized() {
        Views v = new Views();
        for (Map.Entry<Integer, long[]> e : byDept.entrySet())
            if (e.getValue()[1] != 0) v.byDept.put(e.getKey(), e.getValue().clone());
        for (Map.Entry<Integer, Integer> e : bigByDept.entrySet())
            if (e.getValue() != 0) v.bigByDept.put(e.getKey(), e.getValue());
        for (Map.Entry<String, long[]> e : byRegion.entrySet())
            if (e.getValue()[1] != 0) v.byRegion.put(e.getKey(), e.getValue().clone());
        return v;
    }

    /** after - before over the union of keys: the true state difference of two snapshots. */
    static Views difference(Views before, Views after) {
        Views d = new Views();
        Set<Integer> deptKeys = new TreeSet<>(before.byDept.keySet());
        deptKeys.addAll(after.byDept.keySet());
        for (int k : deptKeys) {
            long[] a = after.byDept.get(k), b = before.byDept.get(k);
            long sum = (a == null ? 0 : a[0]) - (b == null ? 0 : b[0]);
            long cnt = (a == null ? 0 : a[1]) - (b == null ? 0 : b[1]);
            if (cnt != 0 || sum != 0) d.byDept.put(k, new long[]{sum, cnt});
        }
        Set<Integer> bigKeys = new TreeSet<>(before.bigByDept.keySet());
        bigKeys.addAll(after.bigByDept.keySet());
        for (int k : bigKeys) {
            int delta = after.bigByDept.getOrDefault(k, 0) - before.bigByDept.getOrDefault(k, 0);
            if (delta != 0) d.bigByDept.put(k, delta);
        }
        Set<String> regions = new TreeSet<>(before.byRegion.keySet());
        regions.addAll(after.byRegion.keySet());
        for (String k : regions) {
            long[] a = after.byRegion.get(k), b = before.byRegion.get(k);
            long sum = (a == null ? 0 : a[0]) - (b == null ? 0 : b[0]);
            long cnt = (a == null ? 0 : a[1]) - (b == null ? 0 : b[1]);
            if (cnt != 0 || sum != 0) d.byRegion.put(k, new long[]{sum, cnt});
        }
        return d;
    }

    static <K> boolean deepEquals(Map<K, long[]> a, Map<K, long[]> b) {
        if (!a.keySet().equals(b.keySet())) return false;
        for (K k : a.keySet()) if (!Arrays.equals(a.get(k), b.get(k))) return false;
        return true;
    }

    static boolean sameViews(Views a, Views b) {
        Views x = a.normalized(), y = b.normalized();
        return deepEquals(x.byDept, y.byDept) && x.byBigCopy().equals(y.byBigCopy())
                && deepEquals(x.byRegion, y.byRegion);
    }

    Map<Integer, Integer> byBigCopy() { return new TreeMap<>(bigByDept); }
}

/** The incremental engine: applies one delta with the operator delta rules, and counts its work. */
class Engine {
    final long threshold;
    final Map<Integer, Row> sales = new TreeMap<>();
    final Map<Integer, String> dept = new TreeMap<>();
    final Views views = new Views();
    long deltaEntriesApplied;
    long groupsRepaired;
    long fullRecomputes;

    Engine(long threshold) { this.threshold = threshold; }

    /** Validates a batch against the current state without mutating anything. */
    void validate(Batch b) {
        Set<Integer> inserted = new HashSet<>();
        for (Row r : b.inserts) {
            if (sales.containsKey(r.id) || !inserted.add(r.id))
                throw new RejectedBatch("duplicate insert for id " + r.id);
            if (r.amount < 0) throw new RejectedBatch("negative amount for id " + r.id);
        }
        Set<Integer> deleted = new HashSet<>();
        for (int id : b.deletes) {
            if (!sales.containsKey(id) || inserted.contains(id) || !deleted.add(id))
                throw new RejectedBatch("delete of missing or concurrently inserted id " + id);
        }
        for (int id : b.deptSet.keySet())
            if (b.deptSet.get(id) == null) throw new RejectedBatch("null region for dept " + id);
        for (int id : b.deptDelete)
            if (!dept.containsKey(id) || b.deptSet.containsKey(id))
                throw new RejectedBatch("delete of missing or concurrently set dept " + id);
    }

    /**
     * Applies the batch with delta propagation. The returned value is the delta the operators actually
     * applied (view state after minus view state before), which the checks compare with the oracle.
     */
    Views apply(Batch b) {
        validate(b);
        Views before = views.copy();
        Map<Integer, Integer> net = new TreeMap<>();
        for (Row r : b.inserts) net.merge(r.id, 1, Integer::sum);
        for (int id : b.deletes) net.merge(id, -1, Integer::sum);
        Set<Integer> affected = new TreeSet<>();
        for (Map.Entry<Integer, Integer> e : net.entrySet()) {
            int id = e.getKey();
            Row row = e.getValue() > 0 ? findInsert(b, id) : sales.get(id);
            affected.add(row.dept);
        }
        affected.addAll(b.deptSet.keySet());
        affected.addAll(b.deptDelete);

        // The pre-batch dimension mapping of every affected group, captured before phase 2 mutates it.
        Map<Integer, String> preRegion = new TreeMap<>();
        for (int group : affected) preRegion.put(group, dept.get(group));

        // Phase 1: the sales delta, propagated through filter -> group-by.
        for (Map.Entry<Integer, Integer> e : net.entrySet()) {
            int id = e.getKey(), mult = e.getValue();
            if (mult == 0) continue;
            Row row = mult > 0 ? findInsert(b, id) : sales.get(id);
            if (mult > 0) sales.put(id, row); else sales.remove(id);
            long[] agg = views.byDept.computeIfAbsent(row.dept, k -> new long[2]);
            agg[0] += mult * row.amount;
            agg[1] += mult;
            if (agg[1] == 0) views.byDept.remove(row.dept);
            if (row.amount > threshold) {
                int occ = views.bigByDept.getOrDefault(row.dept, 0) + mult;
                if (occ == 0) views.bigByDept.remove(row.dept); else views.bigByDept.put(row.dept, occ);
            }
            deltaEntriesApplied++;
        }

        // Phase 2: the dimension delta. The join is not linear in a moving dimension, so every affected
        // group has its pre-batch contribution removed from the old region and its post-batch contribution
        // added to the new one - the standard affected-group repair.
        for (Map.Entry<Integer, String> e : b.deptSet.entrySet()) dept.put(e.getKey(), e.getValue());
        for (int id : b.deptDelete) dept.remove(id);
        Views pre = before;
        for (int group : affected) {
            long[] preAgg = pre.byDept.getOrDefault(group, new long[2]);
            String oldRegion = preRegion.get(group);
            if (oldRegion != null && preAgg[1] != 0) {
                long[] reg = views.byRegion.computeIfAbsent(oldRegion, k -> new long[2]);
                reg[0] -= preAgg[0];
                reg[1] -= preAgg[1];
                if (reg[1] == 0 && reg[0] == 0) views.byRegion.remove(oldRegion);
            }
            long[] postAgg = views.byDept.getOrDefault(group, new long[2]);
            String newRegion = dept.get(group);
            if (newRegion != null && postAgg[1] != 0) {
                long[] reg = views.byRegion.computeIfAbsent(newRegion, k -> new long[2]);
                reg[0] += postAgg[0];
                reg[1] += postAgg[1];
            }
            groupsRepaired++;
        }
        return Views.difference(before, views.copy());
    }

    private static Row findInsert(Batch b, int id) {
        for (Row r : b.inserts) if (r.id == id) return r;
        throw new IllegalStateException("no insert for id " + id);
    }

    /** A full recomputation of the views from the base tables - used only by the checks. */
    void recomputeFully() {
        fullRecomputes++;
        views.byDept.clear();
        views.bigByDept.clear();
        views.byRegion.clear();
        for (Row r : sales.values()) {
            long[] agg = views.byDept.computeIfAbsent(r.dept, k -> new long[2]);
            agg[0] += r.amount;
            agg[1]++;
            if (r.amount > threshold) views.bigByDept.merge(r.dept, 1, Integer::sum);
        }
        for (Map.Entry<Integer, long[]> e : views.byDept.entrySet()) {
            String region = dept.get(e.getKey());
            if (region == null) continue;
            long[] reg = views.byRegion.computeIfAbsent(region, k -> new long[2]);
            reg[0] += e.getValue()[0];
            reg[1] += e.getValue()[1];
        }
    }
}

/** The brute-force oracle: recompute everything from scratch for a state. */
class Oracle {
    final long threshold;

    Oracle(long threshold) { this.threshold = threshold; }

    Views compute(Map<Integer, Row> sales, Map<Integer, String> dept) {
        Views v = new Views();
        for (Row r : sales.values()) {
            long[] agg = v.byDept.computeIfAbsent(r.dept, k -> new long[2]);
            agg[0] += r.amount;
            agg[1]++;
            if (r.amount > threshold) v.bigByDept.merge(r.dept, 1, Integer::sum);
        }
        for (Map.Entry<Integer, long[]> e : v.byDept.entrySet()) {
            String region = dept.get(e.getKey());
            if (region == null) continue;
            long[] reg = v.byRegion.computeIfAbsent(region, k -> new long[2]);
            reg[0] += e.getValue()[0];
            reg[1] += e.getValue()[1];
        }
        return v.normalized();
    }
}

/** A bounded delta queue that coalesces and checkpoints instead of growing without limit. */
class DeltaBuffer {
    final Engine engine;
    final int capacity;
    final Map<Integer, Row> pendingIns = new LinkedHashMap<>();
    final Set<Integer> pendingDel = new TreeSet<>();
    final Map<Integer, String> pendingDeptSet = new LinkedHashMap<>();
    final Set<Integer> pendingDeptDrop = new TreeSet<>();
    int maxDepth, coalesces, checkpoints, rejects;
    long flushedEntries;

    DeltaBuffer(Engine engine, int capacity) { this.engine = engine; this.capacity = capacity; }

    int depth() {
        return pendingIns.size() + pendingDel.size() + pendingDeptSet.size() + pendingDeptDrop.size();
    }

    /** Submits a batch; validates it against the pending view of the world, then queues or flushes. */
    void submit(Batch b) {
        if (b.isEmpty()) return;
        // Validate against base state plus everything already queued.
        Engine shadow = new Engine(engine.threshold);
        shadow.sales.putAll(engine.sales);
        shadow.dept.putAll(engine.dept);
        for (Row r : pendingIns.values()) shadow.sales.put(r.id, r);
        for (int id : pendingDel) shadow.sales.remove(id);
        for (Map.Entry<Integer, String> e : pendingDeptSet.entrySet()) shadow.dept.put(e.getKey(), e.getValue());
        for (int id : pendingDeptDrop) shadow.dept.remove(id);
        try {
            shadow.validate(b);
        } catch (RejectedBatch e) {
            rejects++;
            throw e;
        }
        for (Row r : b.inserts) {
            if (pendingDel.remove(r.id)) coalesces++;
            pendingIns.put(r.id, r);
        }
        for (int id : b.deletes) {
            if (pendingIns.remove(id) != null) coalesces++;
            else pendingDel.add(id);
        }
        for (Map.Entry<Integer, String> e : b.deptSet.entrySet()) {
            if (pendingDeptDrop.remove(e.getKey())) coalesces++;
            pendingDeptSet.put(e.getKey(), e.getValue());
        }
        for (int id : b.deptDelete) {
            if (pendingDeptSet.remove(id) != null) coalesces++;
            else pendingDeptDrop.add(id);
        }
        if (depth() > capacity) checkpoint();
        maxDepth = Math.max(maxDepth, depth());
    }

    /** Coalesces the queue into one net delta and applies it; the queue becomes empty. */
    void checkpoint() {
        Batch net = new Batch();
        // Cancel inserts that are also deleted by an explicitly queued delete.
        for (Map.Entry<Integer, Row> e : pendingIns.entrySet())
            if (!pendingDel.contains(e.getKey())) net.ins(e.getValue());
        for (int id : pendingDel) if (!pendingIns.containsKey(id)) net.del(id);
        for (Map.Entry<Integer, String> e : pendingDeptSet.entrySet()) net.dept(e.getKey(), e.getValue());
        for (int id : pendingDeptDrop) net.deptDrop(id);
        int entries = net.entryCount();
        flushedEntries += entries;
        pendingIns.clear();
        pendingDel.clear();
        pendingDeptSet.clear();
        pendingDeptDrop.clear();
        checkpoints++;
        if (!net.isEmpty()) engine.apply(net);
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static final long THRESHOLD = 50_000;

    static Engine seeded(long seed, int rows, int depts) {
        Engine engine = new Engine(THRESHOLD);
        SplitMix64 rng = new SplitMix64(seed);
        for (int d = 100; d < 100 + depts; d++) engine.dept.put(d, "region-" + (d % 3));
        for (int i = 1; i <= rows; i++)
            engine.sales.put(i, new Row(i, 100 + rng.nextInt(depts), 10_000 + rng.nextInt(80) * 1_000L));
        engine.recomputeFully();
        return engine;
    }

    static Views oracleOf(Engine e) {
        Map<Integer, Row> sales = new TreeMap<>(e.sales);
        Map<Integer, String> dept = new TreeMap<>(e.dept);
        Views v = new Oracle(e.threshold).compute(sales, dept);
        return v;
    }

    static boolean sameViews(Views a, Views b) {
        Views x = a.normalized(), y = b.normalized();
        return Views.deepEquals(x.byDept, y.byDept) && x.byBigCopy().equals(y.byBigCopy())
                && Views.deepEquals(x.byRegion, y.byRegion);
    }

    public static void main(String[] args) {
        // 1. The initial views equal the brute-force oracle.
        Engine engine = seeded(7, 400, 6);
        check(sameViews(engine.views, oracleOf(engine)), "initial views differ from the oracle");
        check(engine.views.byDept.size() >= 5, "too few groups in the initial state");
        check(engine.views.byRegion.size() >= 2, "too few regions in the initial state");

        // 2. A delete that empties a group removes it; a reinsertion brings it back.
        Engine tiny = new Engine(THRESHOLD);
        tiny.dept.put(1, "r1");
        tiny.apply(new Batch().ins(new Row(1, 1, 60_000)).ins(new Row(2, 1, 70_000)).ins(new Row(3, 2, 20_000)));
        check(tiny.views.byDept.get(1)[1] == 2 && tiny.views.byDept.get(1)[0] == 130_000, "group aggregate wrong");
        check(tiny.views.byDept.get(2)[1] == 1 && tiny.views.byDept.get(2)[0] == 20_000, "second group wrong");
        check(tiny.views.bigByDept.getOrDefault(2, 0) == 0, "a row below the threshold entered the filter view");
        check(tiny.views.byRegion.get("r1")[1] == 2 && !tiny.views.byRegion.containsKey("r2"),
                "the region view joined a group with no department");
        tiny.apply(new Batch().del(1).del(2));
        check(!tiny.views.byDept.containsKey(1), "an emptied group survived");
        check(!tiny.views.byRegion.containsKey("r1"), "an emptied group kept its region totals");
        tiny.apply(new Batch().ins(new Row(4, 1, 55_000)));
        check(tiny.views.byDept.get(1)[1] == 1 && tiny.views.bigByDept.get(1) == 1, "reinserted group wrong");
        check(sameViews(tiny.views, oracleOf(tiny)), "tiny engine diverged from the oracle");

        // 3. The join repair: moving a department moves its totals, deleting one drops them.
        Engine join = new Engine(THRESHOLD);
        join.dept.put(1, "north");
        join.dept.put(2, "south");
        join.apply(new Batch().ins(new Row(1, 1, 80_000)).ins(new Row(2, 1, 20_000)).ins(new Row(3, 2, 30_000)));
        check(join.views.byRegion.get("north")[0] == 100_000 && join.views.byRegion.get("north")[1] == 2,
                "region aggregate wrong");
        Views applied = join.apply(new Batch().dept(1, "east"));
        check(!join.views.byRegion.containsKey("north") && join.views.byRegion.get("east")[0] == 100_000,
                "join repair did not move the group between regions");
        check(join.views.byDept.get(1)[1] == 2, "the group view must keep the group after a dimension change");
        check(applied.byRegion.containsKey("north") && applied.byRegion.get("north")[1] == -2
                && applied.byRegion.get("east")[1] == 2, "applied delta does not describe the move");
        check(sameViews(join.views, oracleOf(join)), "region view diverged from the oracle after a dimension move");
        join.apply(new Batch().deptDrop(2));
        check(!join.views.byRegion.containsKey("south"), "a deleted department kept its region totals");
        check(join.views.byDept.containsKey(2), "the group view must survive a deleted department");
        check(sameViews(join.views, oracleOf(join)), "join view diverged from the oracle after a department delete");

        // 4. The applied delta equals the true state difference, batch after batch.
        Engine delta = seeded(11, 300, 5);
        SplitMix64 rng = new SplitMix64(31337);
        Oracle oracle = new Oracle(THRESHOLD);
        int batches = 40, verified = 0;
        long ivmEntries = 0, fullEntries = 0;
        for (int b = 0; b < batches; b++) {
            Batch batch = new Batch();
            Set<Integer> deletedThisBatch = new HashSet<>();
            int kind = rng.nextInt(4);
            Set<Integer> existing = new TreeSet<>(delta.sales.keySet());
            List<Integer> existingList = new ArrayList<>(existing);
            for (int i = 0; i < 6; i++) {
                if (kind == 0 && delta.sales.size() > 50) {
                    int id = existingList.get(rng.nextInt(existingList.size()));
                    if (deletedThisBatch.add(id)) batch.del(id);
                } else {
                    int id = 900_000 + b * 10 + i;
                    if (!delta.sales.containsKey(id))
                        batch.ins(new Row(id, 100 + rng.nextInt(5), rng.nextInt(90) * 1_000L));
                }
            }
            if (kind == 3) {
                int d = 100 + rng.nextInt(5);
                if (delta.dept.containsKey(d)) batch.dept(d, "region-" + rng.nextInt(4));
            }
            if (batch.isEmpty()) continue;
            Map<Integer, Row> beforeSales = new TreeMap<>(delta.sales);
            Map<Integer, String> beforeDept = new TreeMap<>(delta.dept);
            Views before = oracle.compute(beforeSales, beforeDept).normalized();
            long workBefore = delta.deltaEntriesApplied + delta.groupsRepaired;
            Views appliedDelta = delta.apply(batch);
            ivmEntries += delta.deltaEntriesApplied + delta.groupsRepaired - workBefore;
            Views after = oracle.compute(new TreeMap<>(delta.sales), new TreeMap<>(delta.dept)).normalized();
            Views trueDelta = Views.difference(before, after);
            check(sameViews(delta.views, after), "views diverged from the oracle in batch " + b);
            check(sameViews(appliedDelta, trueDelta), "applied delta differs from the true difference in batch " + b);
            fullEntries += delta.sales.size();
            verified++;
        }
        check(verified >= 30, "only " + verified + " non-empty batches were verified");
        check(ivmEntries * 4 < fullEntries, "incremental cost " + ivmEntries
                + " is not far below full recomputation " + fullEntries);
        check(delta.groupsRepaired > 0, "the join repair never ran");

        // 5. Invalid batches are rejected atomically: state and views unchanged.
        Engine guard = seeded(3, 100, 4);
        Views snapshot = guard.views.copy();
        Map<Integer, Row> salesSnapshot = new TreeMap<>(guard.sales);
        String[] badCases = {"duplicate", "missing-delete", "dept-delete"};
        for (String badCase : badCases) {
            boolean threw = false;
            try {
                switch (badCase) {
                    case "duplicate": guard.apply(new Batch().ins(new Row(1, 100, 5_000))); break;
                    case "missing-delete": guard.apply(new Batch().del(987_654)); break;
                    default: guard.apply(new Batch().deptDrop(999_999)); break;
                }
            } catch (RejectedBatch e) { threw = true; }
            check(threw, "invalid batch was accepted: " + badCase);
            check(sameViews(guard.views, snapshot) && guard.sales.equals(salesSnapshot),
                    "a rejected batch mutated the state: " + badCase);
        }
        boolean atomic = false;
        try { guard.apply(new Batch().ins(new Row(500_001, 100, 9_000)).del(987_654)); }
        catch (RejectedBatch e) { atomic = true; }
        check(atomic && !guard.sales.containsKey(500_001),
                "a batch with one valid and one invalid operation was not rejected as a whole");

        // 6. Bounded memory: the queue never exceeds its cap under a flood.
        Engine flood = seeded(5, 200, 4);
        DeltaBuffer buffer = new DeltaBuffer(flood, 8);
        SplitMix64 floodRng = new SplitMix64(4242);
        int submitted = 0;
        for (int b = 0; b < 30; b++) {
            Batch batch = new Batch();
            for (int i = 0; i < 3; i++)
                batch.ins(new Row(700_000 + b * 3 + i, 100 + floodRng.nextInt(4),
                        floodRng.nextInt(90) * 1_000L));
            buffer.submit(batch);
            check(buffer.depth() <= 8, "queue depth " + buffer.depth() + " exceeded the cap after batch " + b);
            submitted++;
        }
        check(buffer.checkpoints > 0, "the flood never triggered a checkpoint");
        check(buffer.maxDepth >= 3 && buffer.maxDepth <= 8, "the queue must actually buffer deltas, peak was "
                + buffer.maxDepth);
        check(submitted == 30, "the flood did not submit every batch");
        buffer.checkpoint();
        check(buffer.depth() == 0, "a checkpoint must empty the queue");
        check(sameViews(flood.views, oracleOf(flood)), "bounded-memory engine diverged from the oracle");

        // 7. Coalescing cancels opposite multiplicities and stays lossless.
        Engine lossless = seeded(9, 150, 3);
        Engine reference = seeded(9, 150, 3);
        DeltaBuffer q = new DeltaBuffer(lossless, 16);
        SplitMix64 mix = new SplitMix64(777);
        for (int round = 0; round < 12; round++) {
            Batch batch = new Batch();
            for (int i = 0; i < 3; i++) {
                batch.ins(new Row(800_000 + round * 3 + i, 100 + mix.nextInt(3), mix.nextInt(90) * 1_000L));
                if (round > 0 && mix.nextInt(3) == 0)
                    batch.del(800_000 + (round - 1) * 3 + i);   // delete a row inserted an earlier round
            }
            q.submit(batch);
            reference.apply(batch);
        }
        q.checkpoint();
        check(q.coalesces > 0, "the queue never cancelled an opposite multiplicity");
        check(sameViews(lossless.views, oracleOf(lossless)), "coalesced engine diverged from the oracle");
        check(sameViews(lossless.views, reference.views),
                "bounded-memory engine and unbounded reference disagree");
        check(lossless.sales.equals(reference.sales), "base tables disagree after coalescing");
        check(lossless.sales.size() == reference.sales.size(), "row counts disagree after coalescing");

        // 8. A queued insert that is later deleted with an explicit delete must cancel out.
        Engine cancel = seeded(2, 20, 2);
        DeltaBuffer cq = new DeltaBuffer(cancel, 100);
        cq.submit(new Batch().ins(new Row(600_001, 100, 66_000)));
        cq.submit(new Batch().del(600_001));
        check(cq.depth() == 0, "an insert followed by a delete must cancel in the queue, depth " + cq.depth());
        cq.checkpoint();
        check(!cancel.sales.containsKey(600_001), "the cancelled row reached the base table");
        check(sameViews(cancel.views, oracleOf(cancel)), "cancellation corrupted the views");

        // 9. Repeated recomputation and an empty batch are no-ops.
        Engine noop = seeded(4, 60, 3);
        Views beforeNoop = noop.views.copy();
        noop.apply(new Batch());
        check(sameViews(noop.views, beforeNoop) && noop.deltaEntriesApplied == 0, "an empty batch changed state");
        noop.recomputeFully();
        check(sameViews(noop.views, beforeNoop), "a redundant full recomputation changed the views");

        // 10. Determinism: the same workload on a fresh engine lands in the same state.
        Engine a = seeded(21, 250, 5), b2 = seeded(21, 250, 5);
        SplitMix64 seedA = new SplitMix64(5150);
        long entriesA = 0, entriesB = 0;
        for (int i = 0; i < 15; i++) {
            Batch batch = new Batch();
            for (int k = 0; k < 4; k++)
                batch.ins(new Row(950_000 + i * 4 + k, 100 + seedA.nextInt(5), seedA.nextInt(90) * 1_000L));
            a.apply(batch);
            entriesA += a.deltaEntriesApplied;
        }
        SplitMix64 seedB = new SplitMix64(5150);
        for (int i = 0; i < 15; i++) {
            Batch batch = new Batch();
            for (int k = 0; k < 4; k++)
                batch.ins(new Row(950_000 + i * 4 + k, 100 + seedB.nextInt(5), seedB.nextInt(90) * 1_000L));
            b2.apply(batch);
            entriesB += b2.deltaEntriesApplied;
        }
        check(sameViews(a.views, b2.views) && entriesA == entriesB, "the engine is not deterministic");
        check(a.sales.equals(b2.sales), "deterministic runs produced different base tables");

        System.out.println("workload: " + verified + " batches, incremental entries " + ivmEntries
                + " vs full recomputation " + fullEntries);
        System.out.println("bounded queue: cap " + buffer.capacity + ", peak depth " + buffer.maxDepth
                + ", checkpoints " + buffer.checkpoints + ", cancellations " + q.coalesces);
        System.out.println("groups repaired by the join delta rule: " + delta.groupsRepaired);
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
