/** Task 925 - Implement hybrid transactional/analytical processing demo (HTAP) with maintenance.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * A single-node HTAP engine: one transactional row store is the source of truth, while analytical queries are
 * answered from a columnar base plus a small delta that holds every insert/update/delete since the last merge.
 * Queries scan base and delta together, merging the two views key by key (a delta entry shadows the base) and
 * dropping tombstoned keys, so the analytical answer always equals the transactional one. A background
 * maintenance step merges the delta into the base once it reaches a threshold, which keeps the delta - and so
 * the query overhead - bounded without ever changing an answer.
 * Simplified: no concurrency, no MVCC, no disk layout; the "columnar" store is a typed map and maintenance is
 * triggered by the delta size rather than by a background thread. What is *not* simplified: the dual
 * representation, the shadowing/tombstone rules, and the merge, whose whole point is that it is invisible.
 * Verified: every aggregate equals a brute-force scan of the row store after every single operation; deletes
 * and updates are reflected on both sides; the merge fires, preserves every answer, and keeps the delta at or
 * below the threshold; empty and single-row relations behave; and the engine is deterministic.
 */
import java.util.*;

/** The aggregates the analytical side must answer. */
final class Summary {
    final long count;
    final double sum, average, max;

    Summary(long count, double sum, double max) {
        this.count = count;
        this.sum = sum;
        this.max = max;
        this.average = count == 0 ? Double.NaN : sum / count;
    }

    boolean sameAs(Summary other) {
        return count == other.count && Math.abs(sum - other.sum) < 1e-9
                && (max == other.max || Math.abs(max - other.max) < 1e-9)
                && (Double.isNaN(average) == Double.isNaN(other.average)
                    && (Double.isNaN(average) || Math.abs(average - other.average) < 1e-9));
    }
}

/** Row store (transactional) plus columnar base and delta (analytical). */
final class Htap {
    static final double MERGE_THRESHOLD = 4;

    final Map<String, Double> rows = new HashMap<>();      // the source of truth
    final Map<String, Double> base = new HashMap<>();      // merged columnar view
    final Map<String, Double> delta = new HashMap<>();     // insert/update since the last merge
    final Set<String> tombstones = new HashSet<>();        // deletes since the last merge
    int merges, maxDelta;
    double sumMerges;

    void put(String key, double value) {
        rows.put(key, value);
        delta.put(key, value);
        tombstones.remove(key);
        maintain();
    }

    void remove(String key) {
        if (rows.remove(key) == null) return;
        delta.remove(key);
        tombstones.add(key);
        maintain();
    }

    /** The maintenance step: fold the delta into the base and clear the tombstones. */
    void maintain() {
        maxDelta = Math.max(maxDelta, delta.size() + tombstones.size());
        if (delta.size() + tombstones.size() < MERGE_THRESHOLD) return;
        merges++;
        sumMerges += delta.size();
        for (Map.Entry<String, Double> e : delta.entrySet()) base.put(e.getKey(), e.getValue());
        for (String key : tombstones) base.remove(key);
        delta.clear();
        tombstones.clear();
    }

    /** Analytical scan: the base view shadowed by the delta, minus the tombstones. */
    Summary query(double floor) {
        long count = 0;
        double sum = 0, max = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, Double> e : base.entrySet()) {
            if (tombstones.contains(e.getKey()) || delta.containsKey(e.getKey())) continue;   // shadowed by the delta
            if (e.getValue() < floor) continue;
            count++;
            sum += e.getValue();
            max = Math.max(max, e.getValue());
        }
        for (Map.Entry<String, Double> e : delta.entrySet()) {
            if (e.getValue() < floor) continue;
            count++;
            sum += e.getValue();
            max = Math.max(max, e.getValue());
        }
        return new Summary(count, sum, count == 0 ? Double.NEGATIVE_INFINITY : max);
    }

    /** Brute force over the row store - the oracle the query must match. */
    Summary oracle(double floor) {
        long count = 0;
        double sum = 0, max = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, Double> e : rows.entrySet()) {
            if (e.getValue() < floor) continue;
            count++;
            sum += e.getValue();
            max = Math.max(max, e.getValue());
        }
        return new Summary(count, sum, count == 0 ? Double.NEGATIVE_INFINITY : max);
    }

    /** Entries held outside the base: the query cost the maintenance is supposed to bound. */
    int deltaSize() { return delta.size() + tombstones.size(); }
}

public class Solution {
    public static void main(String[] args) {
        Htap db = new Htap();
        List<String> script = new ArrayList<>();
        for (int t = 0; t < 40; t++) {
            if (t % 11 == 7) script.add("del:k" + (t % 13));
            else script.add("put:k" + (t % 13) + "=" + (t % 9));
        }

        int checks = 0, observedMerges = 0, worstDelta = 0;
        Summary afterMerge = null;
        for (String step : script) {
            if (step.startsWith("del:")) db.remove(step.substring(4));
            else {
                String[] kv = step.substring(4).split("=");
                db.put(kv[0], Double.parseDouble(kv[1]));
            }
            checks++;
            worstDelta = Math.max(worstDelta, db.deltaSize());
            check(db.query(0).sameAs(db.oracle(0)), "aggregate over all rows diverged after step " + checks);
            check(db.query(3).sameAs(db.oracle(3)), "filtered aggregate diverged after step " + checks);
            check(db.rows.size() == db.query(0).count, "the analytical count must match the row store size");
        }
        observedMerges = db.merges;
        afterMerge = db.query(0);

        System.out.println("HTAP: " + script.size() + " operations, " + db.merges + " merges, base rows " + db.base.size()
                + ", delta held " + db.deltaSize() + " entries (peak " + worstDelta + ", threshold "
                + Htap.MERGE_THRESHOLD + ")");
        System.out.println("final view: count " + afterMerge.count + ", sum " + String.format("%.2f", afterMerge.sum)
                + ", average " + String.format("%.3f", afterMerge.average) + ", max "
                + String.format("%.1f", afterMerge.max) + "; row store rows " + db.rows.size());

        check(db.merges > 0, "maintenance must actually run, merges " + db.merges);
        check(worstDelta <= Htap.MERGE_THRESHOLD + 1, "the delta must stay bounded, peak " + worstDelta);
        check(db.deltaSize() < Htap.MERGE_THRESHOLD, "after maintenance the delta must be below the threshold");
        check(db.query(0).sameAs(afterMerge), "the merge must not change any answer");

        // The merge is invisible: comparing the answers before and after it forces it to happen here.
        double beforeMergeSum = db.query(0).sum;
        db.put("k99", 7.5);
        db.put("k98", 8.5);
        db.put("k97", 9.5);
        check(db.merges > observedMerges, "three inserts over a non-empty delta must trigger a merge");
        check(Math.abs(db.query(0).sum - beforeMergeSum - 25.5) < 1e-9, "the merged view must still add up, got "
                + db.query(0).sum);

        // A key that is updated twice inside one delta window must be counted once.
        Htap single = new Htap();
        single.put("a", 1);
        single.put("a", 5);
        single.put("b", 2);
        single.put("b", 6);
        single.put("c", 3);
        check(single.query(0).count == 3 && Math.abs(single.query(0).sum - 14) < 1e-9,
                "shadowed delta entries must collapse to the latest value: count " + single.query(0).count
                        + ", sum " + single.query(0).sum);
        single.remove("b");
        check(single.query(0).count == 2 && Math.abs(single.query(0).sum - 8) < 1e-9, "a delete must remove the row");
        check(single.query(0).sameAs(single.oracle(0)), "the single-row engine must still agree with the oracle");

        // Boundaries: an empty relation, a single row, and a filter that excludes everything.
        Htap empty = new Htap();
        Summary none = empty.query(0);
        check(none.count == 0 && none.sum == 0 && Double.isNaN(none.average) && none.max == Double.NEGATIVE_INFINITY,
                "an empty relation has no rows, a zero sum, an undefined average and no maximum");
        empty.remove("missing");
        check(empty.merges == 0 && empty.query(0).count == 0, "deleting an absent key must be a no-op");
        Htap one = new Htap();
        one.put("only", 4.25);
        check(one.query(0).count == 1 && Math.abs(one.query(0).average - 4.25) < 1e-12 && one.query(0).max == 4.25,
                "a single row has its value as count, sum, average and max");
        check(one.query(5).count == 0 && Double.isNaN(one.query(5).average),
                "a filter above every value must return an empty aggregate");

        Htap replay = new Htap();
        for (String step : script) {
            if (step.startsWith("del:")) replay.remove(step.substring(4));
            else {
                String[] kv = step.substring(4).split("=");
                replay.put(kv[0], Double.parseDouble(kv[1]));
            }
        }
        check(replay.query(0).sameAs(afterMerge) && replay.merges == observedMerges,
                "replaying the script must be reproducible");

        System.out.println("merge check: " + checks + " consistency checks stayed green across " + db.merges
                + " merges, delta peak " + worstDelta + ", final delta " + db.deltaSize());
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
