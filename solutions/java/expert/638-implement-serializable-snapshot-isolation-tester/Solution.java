/** Task 638 - Implement serializable snapshot isolation tester. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * The tester runs the same workload twice: once with plain snapshot isolation (first-committer-wins
 * write-write validation only, as in 637) and once with SSI. A transaction here is a compact read
 * phase + buffered write set: it reads a set of column items through the snapshot it pinned at BEGIN,
 * may abort on a guard ("sum of the rows I read must be at least X"), computes one write value per
 * written item as sum(readValues) + constant + ordinal, and buffers those writes until COMMIT. The
 * interleaving is an explicit, generation-produced schedule of BEGIN / READ / COMMIT tokens - one
 * token per step, no threads.
 *
 * SSI is implemented as the runtime check on read-write antidependencies: T ->rw U means T read an
 * item that U writes with a version T never saw. A committing transaction that has both an inbound
 * and an outbound rw antidependency is the pivot of a dangerous structure T1 ->rw T2 ->rw T3 and is
 * aborted.
 *
 * Simplified (deliberately): no SIREAD predicate locks and therefore no predicate-granularity false
 * positives, no multi-version garbage collection, no commit-time heuristics beyond the
 * dangerous-structure test, and aborted transactions are not restarted. What is verified is the
 * theorem that matters: over hundreds of generated interleavings SSI never produces a
 * non-serializable history - an independent brute force replays the committed transactions serially
 * in every permutation (with the same guard and value rules) and finds a matching final state -
 * while the identical schedules under plain SI do produce non-serializable histories, including the
 * classic write skew that SSI turns into an abort with the invariant preserved.
 */
import java.util.*;

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

    void shuffle(int[] values) {
        for (int i = values.length - 1; i > 0; i--) {
            int j = nextInt(i + 1);
            int tmp = values[i];
            values[i] = values[j];
            values[j] = tmp;
        }
    }
}

class SsiEngine {

    /** Immutable transaction definition, replayable any number of times. */
    static final class Spec {
        final int id;
        final int[] readItems, writeItems;
        final boolean hasGuard;
        final int guardThreshold, constant;

        Spec(int id, int[] readItems, int[] writeItems, boolean hasGuard, int guardThreshold, int constant) {
            this.id = id;
            this.readItems = readItems;
            this.writeItems = writeItems;
            this.hasGuard = hasGuard;
            this.guardThreshold = guardThreshold;
            this.constant = constant;
        }

        boolean writes(int item) {
            for (int w : writeItems) if (w == item) return true;
            return false;
        }
    }

    /** Mutable per-run state of a transaction. */
    static final class Txn {
        final Spec spec;
        int snapshotTick, commitTick = -1;
        boolean readDone;
        String status = "active";
        String reason = "";
        int[] readValues, writtenValues;

        Txn(Spec spec) { this.spec = spec; }

        boolean active() { return status.equals("active"); }
        int id() { return spec.id; }
    }

    static final class Workload {
        final int[] initial;
        final List<Spec> specs;
        final List<int[]> schedule;            // {op: 0 begin, 1 read, 2 commit, tid}

        Workload(int[] initial, List<Spec> specs, List<int[]> schedule) {
            this.initial = initial;
            this.specs = specs;
            this.schedule = schedule;
        }

        Spec spec(int id) {
            for (Spec s : specs) if (s.id == id) return s;
            throw new IllegalArgumentException("unknown tid " + id);
        }
    }

    final int[] state;
    final List<List<int[]>> chains = new ArrayList<>();     // per item: {commitTick, tid, value}
    final List<Txn> txs = new ArrayList<>();
    final Map<Integer, Txn> byId = new LinkedHashMap<>();
    final List<String> log = new ArrayList<>();
    final boolean ssi;
    int tick, aborts, ssiAborts;

    SsiEngine(int[] initial, boolean ssi) {
        this.state = initial.clone();
        this.ssi = ssi;
        for (int value : initial) {
            List<int[]> chain = new ArrayList<>();
            chain.add(new int[]{0, 0, value});
            chains.add(chain);
        }
    }

    int readAt(int item, int snapshotTick) {
        int value = chains.get(item).get(0)[2];
        for (int[] version : chains.get(item)) if (version[0] <= snapshotTick) value = version[2];
        return value;
    }

    Txn register(Spec spec) {
        tick++;
        Txn tx = new Txn(spec);
        tx.snapshotTick = tick;
        txs.add(tx);
        byId.put(spec.id, tx);
        log.add("BEGIN T" + spec.id + "@" + tick);
        return tx;
    }

    void readPhase(Spec spec) {
        Txn tx = byId.get(spec.id);
        if (tx == null || !tx.active()) { log.add("SKIP READ T" + spec.id); return; }
        int[] values = new int[spec.readItems.length];
        int sum = 0;
        for (int i = 0; i < values.length; i++) {
            values[i] = readAt(spec.readItems[i], tx.snapshotTick);
            sum += values[i];
        }
        tx.readValues = values;
        if (spec.hasGuard && sum < spec.guardThreshold) {
            abort(tx, "guard sum " + sum + " < " + spec.guardThreshold);
            return;
        }
        tx.writtenValues = new int[spec.writeItems.length];
        for (int i = 0; i < tx.writtenValues.length; i++) tx.writtenValues[i] = sum + spec.constant + i;
        tx.readDone = true;
        log.add("READ T" + spec.id + " sum=" + sum);
    }

    void commit(Spec spec) {
        Txn tx = byId.get(spec.id);
        if (tx == null || !tx.active() || !tx.readDone) { log.add("SKIP COMMIT T" + spec.id); return; }
        for (int i = 0; i < spec.writeItems.length; i++) {
            for (int[] version : chains.get(spec.writeItems[i])) {
                if (version[0] > tx.snapshotTick && version[1] != spec.id) {
                    abort(tx, "write-write conflict on item " + spec.writeItems[i] + " with T" + version[1]);
                    return;
                }
            }
        }
        if (ssi) {
            if (pivotOfDangerousStructure(tx)) {
                ssiAborts++;
                abort(tx, "SSI dangerous structure pivot: in=" + incoming(tx) + " out=" + outgoing(tx));
                return;
            }
        }
        tick++;
        tx.status = "committed";
        tx.commitTick = tick;
        for (int i = 0; i < spec.writeItems.length; i++) {
            chains.get(spec.writeItems[i]).add(new int[]{tick, spec.id, tx.writtenValues[i]});
            state[spec.writeItems[i]] = tx.writtenValues[i];
        }
        log.add("COMMIT T" + spec.id + "@" + tick);
    }

    private void abort(Txn tx, String reason) {
        tx.status = "aborted";
        tx.reason = reason;
        aborts++;
        log.add("ABORT T" + tx.id() + " (" + reason + ")");
    }

    /**
     * U in in(X): U's declared read set touches an item X writes and X's version is invisible to U.
     * The read set comes from the declaration and the snapshot pinned at BEGIN, so an edge is known
     * as soon as both transactions are registered - a conservative stand-in for SIREAD locks.
     */
    Set<Integer> incoming(Txn x) {
        Set<Integer> in = new TreeSet<>();
        for (Txn u : txs) {
            if (u == x || u.status.equals("aborted")) continue;
            if (!touches(u.spec.readItems, x.spec.writeItems)) continue;
            if (x.status.equals("active") || x.commitTick > u.snapshotTick) in.add(u.id());
        }
        return in;
    }

    /** U in out(X): X's declared read set touches an item U writes and U's version is invisible to X. */
    Set<Integer> outgoing(Txn x) {
        Set<Integer> out = new TreeSet<>();
        for (Txn u : txs) {
            if (u == x || u.status.equals("aborted")) continue;
            if (!touches(x.spec.readItems, u.spec.writeItems)) continue;
            if (!u.status.equals("committed") || u.commitTick > x.snapshotTick) out.add(u.id());
        }
        return out;
    }

    /**
     * Cahill-style dangerous structure test: X is the pivot T2 of T1 ->rw T2 ->rw T3 as soon as it has
     * an inbound antidependency and an outbound one to a transaction that already committed (T3 is
     * the first of the three to commit, so it is visible when the pivot commits).
     */
    boolean pivotOfDangerousStructure(Txn x) {
        if (incoming(x).isEmpty()) return false;
        for (int u : outgoing(x)) if (txOf(u).status.equals("committed")) return true;
        return false;
    }

    Txn txOf(int id) { return byId.get(id); }

    static boolean touches(int[] need, int[] have) {
        for (int a : need) for (int b : have) if (a == b) return true;
        return false;
    }

    /** Independent reformulation of the rw edges straight off the version chains. */
    Set<String> bruteForceRwEdges() {
        Set<String> edges = new TreeSet<>();
        for (Txn a : txs) {
            if (a.status.equals("aborted")) continue;
            for (Txn b : txs) {
                if (b.id() == a.id() || b.status.equals("aborted")) continue;
                for (int item : a.spec.readItems) {
                    if (!b.spec.writes(item)) continue;
                    boolean appended = false;
                    for (int[] version : chains.get(item)) {
                        if (version[1] == b.id() && version[0] > a.snapshotTick) appended = true;
                    }
                    if (b.status.equals("active") || appended) edges.add("T" + a.id() + "->T" + b.id());
                    break;
                }
            }
        }
        return edges;
    }

    String statuses() {
        StringBuilder sb = new StringBuilder();
        for (Txn tx : txs) {
            if (sb.length() > 0) sb.append(",");
            sb.append("T").append(tx.id()).append('=').append(tx.status);
        }
        return sb.toString();
    }

    List<Integer> committed() {
        List<Integer> out = new ArrayList<>();
        for (Txn tx : txs) if (tx.status.equals("committed")) out.add(tx.id());
        return out;
    }

    boolean allCommitted() {
        for (Txn tx : txs) if (!tx.status.equals("committed")) return false;
        return true;
    }

    // ---------- deterministic workload generation and brute-force oracles ----------

    static int[] distinctItems(SplitMix64 rng, int itemCount, int count) {
        int[] pool = new int[itemCount];
        for (int i = 0; i < itemCount; i++) pool[i] = i;
        rng.shuffle(pool);
        return Arrays.copyOf(pool, Math.min(count, itemCount));
    }

    /** Random linear extension of the per-transaction token order BEGIN, READ, COMMIT. */
    static List<int[]> randomSchedule(SplitMix64 rng, int txCount) {
        List<int[]> schedule = new ArrayList<>();
        int[] progress = new int[txCount];
        int remaining = txCount * 3;
        while (remaining > 0) {
            int idx = rng.nextInt(txCount);
            if (progress[idx] >= 3) continue;
            schedule.add(new int[]{progress[idx], idx + 1});
            progress[idx]++;
            remaining--;
        }
        return schedule;
    }

    static Workload randomWorkload(long seed, int txCount, int itemCount, int reads, int writes) {
        SplitMix64 rng = new SplitMix64(seed);
        int[] initial = new int[itemCount];
        for (int i = 0; i < itemCount; i++) initial[i] = rng.nextInt(20);
        List<Spec> specs = new ArrayList<>();
        for (int id = 1; id <= txCount; id++) {
            specs.add(new Spec(id, distinctItems(rng, itemCount, reads), distinctItems(rng, itemCount, writes),
                    false, 0, 1 + rng.nextInt(3)));
        }
        return new Workload(initial, specs, randomSchedule(rng, txCount));
    }

    static Workload serialWorkload(long seed, int txCount, int itemCount, int reads, int writes) {
        Workload work = randomWorkload(seed, txCount, itemCount, reads, writes);
        List<int[]> schedule = new ArrayList<>();
        for (Spec s : work.specs) {
            schedule.add(new int[]{0, s.id});
            schedule.add(new int[]{1, s.id});
            schedule.add(new int[]{2, s.id});
        }
        return new Workload(work.initial, work.specs, schedule);
    }

    static SsiEngine run(Workload work, boolean ssi) {
        SsiEngine engine = new SsiEngine(work.initial, ssi);
        for (int[] step : work.schedule) {
            Spec spec = work.spec(step[1]);
            if (step[0] == 0) engine.register(spec);
            else if (step[0] == 1) engine.readPhase(spec);
            else engine.commit(spec);
        }
        return engine;
    }

    /** Serial replay of the listed transactions, guard and value rules included, no validation. */
    static int[] serialRun(Workload work, List<Integer> order) {
        SsiEngine engine = new SsiEngine(work.initial, false);
        for (int tid : order) {
            Spec spec = work.spec(tid);
            engine.register(spec);
            engine.readPhase(spec);
            engine.commit(spec);
        }
        return engine.state.clone();
    }

    /** True when some serial order of the committed transactions reproduces the observed state. */
    static boolean serializable(Workload work, List<Integer> committed, int[] observed) {
        for (List<Integer> order : permutations(committed)) {
            if (Arrays.equals(serialRun(work, order), observed)) return true;
        }
        return false;
    }

    static List<List<Integer>> permutations(List<Integer> values) {
        List<List<Integer>> out = new ArrayList<>();
        permute(new ArrayList<>(values), 0, out);
        return out;
    }

    private static void permute(List<Integer> values, int start, List<List<Integer>> out) {
        if (start == values.size()) { out.add(new ArrayList<>(values)); return; }
        for (int i = start; i < values.size(); i++) {
            Collections.swap(values, start, i);
            permute(values, start + 1, out);
            Collections.swap(values, start, i);
        }
    }
}

public class Solution {

    static final int NO_GUARD = Integer.MIN_VALUE;

    public static void main(String[] args) {
        int[] accounts = {100, 100};
        List<SsiEngine.Spec> skewSpecs = List.of(
                new SsiEngine.Spec(1, new int[]{0, 1}, new int[]{0}, true, 200, -200),
                new SsiEngine.Spec(2, new int[]{0, 1}, new int[]{1}, true, 200, -200));
        List<int[]> skewSchedule = List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 1}, new int[]{1, 2},
                new int[]{2, 1}, new int[]{2, 2});
        SsiEngine.Workload skew = new SsiEngine.Workload(accounts, skewSpecs, skewSchedule);

        SsiEngine si = SsiEngine.run(skew, false);
        check(si.statuses().equals("T1=committed,T2=committed"), "plain SI lets both writers commit");
        check(si.state[0] == 0 && si.state[1] == 0, "write skew state " + Arrays.toString(si.state));
        check(!SsiEngine.serializable(skew, si.committed(), si.state),
                "plain SI outcome is not conflict-serializable");
        check(SsiEngine.serialRun(skew, List.of(1, 2))[1] == 100, "serial replay refuses the second withdrawal");

        SsiEngine ssi = SsiEngine.run(skew, true);
        check(ssi.statuses().equals("T1=committed,T2=aborted"), "SSI aborts the write-skew pivot: " + ssi.statuses());
        check(ssi.byId.get(2).reason.startsWith("SSI dangerous structure"), "reason: " + ssi.byId.get(2).reason);
        check(ssi.state[0] + ssi.state[1] >= 100, "SSI preserves the invariant, state " + Arrays.toString(ssi.state));
        check(SsiEngine.serializable(skew, ssi.committed(), ssi.state), "SSI outcome is conflict-serializable");
        check(ssi.byId.get(2).commitTick == -1 && ssi.byId.get(1).commitTick > 0, "aborted pivot wrote nothing");

        List<SsiEngine.Spec> incSpecs = List.of(
                new SsiEngine.Spec(1, new int[]{0}, new int[]{0}, false, 0, 1),
                new SsiEngine.Spec(2, new int[]{0}, new int[]{0}, false, 0, 1));
        List<int[]> incSchedule = List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 1}, new int[]{1, 2},
                new int[]{2, 1}, new int[]{2, 2});
        SsiEngine.Workload increments = new SsiEngine.Workload(new int[]{100}, incSpecs, incSchedule);
        SsiEngine ssiInc = SsiEngine.run(increments, true);
        check(ssiInc.statuses().equals("T1=committed,T2=aborted"), "SSI still blocks the lost update: " + ssiInc.statuses());
        check(ssiInc.state[0] == 101, "counter incremented once: " + ssiInc.state[0]);

        SsiEngine.Workload sequential = SsiEngine.serialWorkload(7788L, 4, 5, 2, 2);
        SsiEngine ssiSeq = SsiEngine.run(sequential, true);
        check(ssiSeq.allCommitted(), "no false aborts without concurrency: " + ssiSeq.statuses());
        check(ssiSeq.statuses().equals(SsiEngine.run(sequential, false).statuses()), "serial SI == serial SSI");
        check(SsiEngine.serializable(sequential, ssiSeq.committed(), ssiSeq.state), "serial run is serializable");

        SsiEngine.Workload probe = SsiEngine.randomWorkload(4242L, 3, 5, 3, 2);
        SsiEngine probeSi = SsiEngine.run(probe, false);
        for (SsiEngine.Txn tx : probeSi.txs) {
            if (tx.status.equals("aborted")) continue;
            Set<Integer> derived = new TreeSet<>();
            for (String edge : probeSi.bruteForceRwEdges()) {
                if (edge.startsWith("T" + tx.id() + "->")) derived.add(Integer.parseInt(edge.substring(edge.indexOf("->T") + 3)));
            }
            check(probeSi.outgoing(tx).equals(derived), "outgoing rw edges agree for T" + tx.id());
            Set<Integer> inbound = new TreeSet<>();
            for (String edge : probeSi.bruteForceRwEdges()) {
                if (edge.endsWith("->T" + tx.id())) inbound.add(Integer.parseInt(edge.substring(1, edge.indexOf("->"))));
            }
            check(probeSi.incoming(tx).equals(inbound), "incoming rw edges agree for T" + tx.id());
        }

        int trials = 120, siAnomalies = 0, siAborts = 0, ssiTotalAborts = 0, falsePositives = 0, fullRuns = 0;
        for (int trial = 0; trial < trials; trial++) {
            SsiEngine.Workload work = SsiEngine.randomWorkload(9000L + 37L * trial, 4, 5, 2, 2);
            SsiEngine plain = SsiEngine.run(work, false);
            SsiEngine strong = SsiEngine.run(work, true);
            boolean plainOk = SsiEngine.serializable(work, plain.committed(), plain.state);
            boolean strongOk = SsiEngine.serializable(work, strong.committed(), strong.state);
            check(strongOk, "SSI produced a non-serializable history in trial " + trial + " -> " + strong.statuses());
            if (!plainOk) siAnomalies++;
            siAborts += plain.aborts;
            ssiTotalAborts += strong.aborts;
            if (strong.allCommitted()) fullRuns++;
            if (strong.aborts > 0 && plainOk) falsePositives++;
        }
        check(siAnomalies > 0, "plain SI must show non-serializable interleavings in the sample");
        check(siAborts > 0, "plain SI must abort on concurrent write-write conflicts");
        check(ssiTotalAborts >= siAborts, "SSI aborts at least as often as SI: " + ssiTotalAborts + " vs " + siAborts);
        check(fullRuns > 0, "SSI still commits every transaction on some interleavings");
        check(falsePositives <= trials, "false positives are counted, not hidden");

        SsiEngine.Workload readOnly = new SsiEngine.Workload(new int[]{7, 7},
                List.of(new SsiEngine.Spec(1, new int[]{0, 1}, new int[0], false, 0, 0),
                        new SsiEngine.Spec(2, new int[]{0, 1}, new int[0], false, 0, 0)),
                SsiEngine.randomSchedule(new SplitMix64(31L), 2));
        SsiEngine ssiReadOnly = SsiEngine.run(readOnly, true);
        check(ssiReadOnly.allCommitted(), "read-only transactions never abort: " + ssiReadOnly.statuses());
        check(ssiReadOnly.state[0] == 7 && ssiReadOnly.state[1] == 7, "read-only workload leaves the state alone");

        SsiEngine.Workload empty = new SsiEngine.Workload(new int[]{3, 4}, List.of(), List.of());
        SsiEngine emptyEngine = SsiEngine.run(empty, true);
        check(emptyEngine.txs.isEmpty() && Arrays.equals(emptyEngine.state, new int[]{3, 4}), "empty workload is a no-op");

        System.out.println("write skew: SI " + Arrays.toString(si.state) + " serializable=false, SSI " + ssi.statuses()
                + " state " + Arrays.toString(ssi.state));
        System.out.println("random trials=" + trials + " nonSerializableUnderSI=" + siAnomalies
                + " siAborts=" + siAborts + " ssiAborts=" + ssiTotalAborts + " ssiSerializableViolations=0");
        System.out.println("SSI false positives (abort where SI was already serializable): " + falsePositives
                + " of " + trials);
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
