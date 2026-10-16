/** Task 640 - Implement optimistic concurrency control simulation with validation phase. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Kung-Robinson optimistic concurrency control over integer column items. A transaction is a
 * declaration of a read item set, a write item set, and one constant; its three phases are explicit
 * steps of a generated schedule:
 *   read phase     - every read item is copied out of the *currently committed* state (classic OCC
 *                    reads the live database, not a snapshot) and the values to be written are
 *                    computed into a private workspace: value_i = sum(read values) + constant + i;
 *   validate phase - backward validation against every transaction that already validated, in
 *                    timestamp order: the transaction's read set must not overlap an earlier
 *                    validator's write set, and neither may its write set (Kung-Robinson plus the
 *                    write-write clause that makes the timestamp order the serialization order);
 *   write phase    - atomic with validation (this simulator has no preemption there): the buffered
 *                    values are installed under a fresh timestamp.
 * A transaction that fails validation is aborted with the conflicting transaction named and applies
 * nothing; the retry re-runs the read phase against the newer state and validates again.
 *
 * Simplified (deliberately): no preemption between validation and writing, no dynamic timestamp
 * adjustment, no read-only optimisation, and the write values are a fixed arithmetic expression of
 * the read values instead of an arbitrary program.
 *
 * Verified: the engine re-checks its own central invariant after *every* successful validation - the
 * committed state must equal an independent serial replay of the validated transactions in
 * timestamp order - so a schedule that is not serialisable cannot pass; a failed validation leaves
 * the state byte-identical; validation (not read) order picks the winner between two conflicting
 * transactions; retries converge to a state with no aborts outstanding; and high contention really
 * aborts more often than low contention on the same generator.
 */
import java.util.*;

class OccEngine {

    static final class Spec {
        final int id;
        final int[] readItems, writeItems;
        final int constant;

        Spec(int id, int[] readItems, int[] writeItems, int constant) {
            this.id = id;
            this.readItems = readItems;
            this.writeItems = writeItems;
            this.constant = constant;
        }

        boolean writes(int item) {
            for (int w : writeItems) if (w == item) return true;
            return false;
        }
    }

    static final class Workload {
        final int[] initial;
        final List<Spec> specs;
        final List<int[]> schedule;            // {op: 0 read, 1 validate; tid}

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

    static final class Txn {
        final Spec spec;
        int[] readValues, writtenValues;
        boolean readDone;
        int attempts, timestamp = -1;
        int readStartStep, validateStep = -1;
        String status = "active";
        String reason = "";

        Txn(Spec spec) { this.spec = spec; }

        int id() { return spec.id; }
    }

    final int[] initial;
    final int[] state;
    final Map<Integer, Txn> txs = new LinkedHashMap<>();
    final Map<Integer, Spec> specById = new LinkedHashMap<>();
    final List<Txn> validationOrder = new ArrayList<>();
    final List<String> log = new ArrayList<>();
    int steps, tsCounter, aborts, validations, invariantChecks, invariantViolations;

    OccEngine(int[] initial, List<Spec> specs) {
        this.initial = initial.clone();
        this.state = initial.clone();
        for (Spec spec : specs) {
            specById.put(spec.id, spec);
            txs.put(spec.id, new Txn(spec));
        }
    }

    Txn txOf(int id) { return txs.get(id); }

    /** Read phase: the live committed state is copied out and the private write set is computed. */
    void readPhase(int tid) {
        Txn tx = txOf(tid);
        tx.attempts++;
        tx.readStartStep = ++steps;
        int sum = 0;
        tx.readValues = new int[tx.spec.readItems.length];
        for (int i = 0; i < tx.readValues.length; i++) {
            tx.readValues[i] = state[tx.spec.readItems[i]];
            sum += tx.readValues[i];
        }
        tx.writtenValues = new int[tx.spec.writeItems.length];
        for (int i = 0; i < tx.writtenValues.length; i++) tx.writtenValues[i] = sum + tx.spec.constant + i;
        tx.readDone = true;
        tx.status = "active";
        tx.reason = "";
        log.add("READ T" + tid + " " + Arrays.toString(tx.readValues));
    }

    /** Validate + write phase; returns true when the transaction installed its workspace. */
    boolean validate(int tid) {
        Txn tx = txOf(tid);
        if (!tx.readDone) { log.add("SKIP VALIDATE T" + tid); return false; }
        tx.validateStep = ++steps;
        tx.timestamp = ++tsCounter;
        for (Txn earlier : validationOrder) {
            // A validator that finished its write phase before this read phase began is already
            // visible in the values that were copied out, so it poses no conflict.
            if (earlier.validateStep < tx.readStartStep) continue;
            if (intersects(tx.spec.readItems, earlier.spec.writeItems)) {
                abort(tx, "read set overlaps the write set of T" + earlier.id());
                return false;
            }
            if (intersects(tx.spec.writeItems, earlier.spec.writeItems)) {
                abort(tx, "write set overlaps the write set of T" + earlier.id());
                return false;
            }
        }
        for (int i = 0; i < tx.spec.writeItems.length; i++) {
            state[tx.spec.writeItems[i]] = tx.writtenValues[i];
        }
        tx.status = "validated";
        validationOrder.add(tx);
        validations++;
        log.add("VALIDATE T" + tid + "@ts" + tx.timestamp);
        invariantChecks++;
        if (!Arrays.equals(state, serialReplay(validationIds()))) invariantViolations++;
        return true;
    }

    private void abort(Txn tx, String reason) {
        tx.status = "aborted";
        tx.reason = reason;
        aborts++;
        log.add("ABORT T" + tx.id() + " (" + reason + ")");
    }

    static boolean intersects(int[] left, int[] right) {
        for (int a : left) for (int b : right) if (a == b) return true;
        return false;
    }

    List<Integer> validationIds() {
        List<Integer> out = new ArrayList<>();
        for (Txn tx : validationOrder) out.add(tx.id());
        return out;
    }

    /** Independent serial oracle: replay the given transactions in order, applying writes last. */
    int[] serialReplay(List<Integer> order) {
        int[] serial = initial.clone();
        for (int tid : order) {
            Spec spec = specById.get(tid);
            int sum = 0;
            for (int item : spec.readItems) sum += serial[item];
            for (int i = 0; i < spec.writeItems.length; i++) {
                serial[spec.writeItems[i]] = sum + spec.constant + i;
            }
        }
        return serial;
    }

    boolean allValidated() {
        for (Txn tx : txs.values()) if (!tx.status.equals("validated")) return false;
        return true;
    }

    // ---------- deterministic workload generation ----------

    static int[] distinctItems(SplitMix64 rng, int itemCount, int count) {
        int[] pool = new int[itemCount];
        for (int i = 0; i < itemCount; i++) pool[i] = i;
        rng.shuffle(pool);
        return Arrays.copyOf(pool, Math.min(count, itemCount));
    }

    /** Canonical OCC worst case: every transaction reads before any of them validates. */
    static List<int[]> waveSchedule(List<Spec> specs) {
        List<int[]> schedule = new ArrayList<>();
        for (Spec s : specs) schedule.add(new int[]{0, s.id});
        for (Spec s : specs) schedule.add(new int[]{1, s.id});
        return schedule;
    }

    /** Random linear extension of the per-transaction token order READ, VALIDATE. */
    static List<int[]> mixedSchedule(SplitMix64 rng, List<Spec> specs) {
        List<int[]> schedule = new ArrayList<>();
        int[] progress = new int[specs.size()];
        int remaining = specs.size() * 2;
        while (remaining > 0) {
            int idx = rng.nextInt(specs.size());
            if (progress[idx] >= 2) continue;
            schedule.add(new int[]{progress[idx], specs.get(idx).id});
            progress[idx]++;
            remaining--;
        }
        return schedule;
    }

    static Workload randomWorkload(long seed, int txCount, int itemCount, int reads, int writes, boolean wave) {
        SplitMix64 rng = new SplitMix64(seed);
        int[] initial = new int[itemCount];
        for (int i = 0; i < itemCount; i++) initial[i] = rng.nextInt(20);
        List<Spec> specs = new ArrayList<>();
        for (int id = 1; id <= txCount; id++) {
            specs.add(new Spec(id, distinctItems(rng, itemCount, reads), distinctItems(rng, itemCount, writes),
                    1 + rng.nextInt(5)));
        }
        return new Workload(initial, specs, wave ? waveSchedule(specs) : mixedSchedule(rng, specs));
    }

    /** Each transaction owns its own slice of the item space: no conflicts are possible at all. */
    static Workload partitionedWorkload(long seed, int txCount, int itemsPerTx, int reads, int writes) {
        SplitMix64 rng = new SplitMix64(seed);
        int[] initial = new int[txCount * itemsPerTx];
        for (int i = 0; i < initial.length; i++) initial[i] = rng.nextInt(20);
        List<Spec> specs = new ArrayList<>();
        for (int id = 1; id <= txCount; id++) {
            int base = (id - 1) * itemsPerTx;
            int[] readItems = new int[Math.min(reads, itemsPerTx)];
            int[] writeItems = new int[Math.min(writes, itemsPerTx)];
            for (int i = 0; i < readItems.length; i++) readItems[i] = base + i;
            for (int i = 0; i < writeItems.length; i++) writeItems[i] = base + i;
            specs.add(new Spec(id, readItems, writeItems, 1 + rng.nextInt(5)));
        }
        return new Workload(initial, specs, waveSchedule(specs));
    }

    /** Runs the schedule, then retries aborted transactions until they validate or attempts run out. */
    static OccEngine execute(Workload work, int maxAttempts) {
        OccEngine engine = new OccEngine(work.initial, work.specs);
        for (int[] step : work.schedule) {
            if (step[0] == 0) engine.readPhase(step[1]);
            else engine.validate(step[1]);
        }
        for (int pass = 0; pass < maxAttempts; pass++) {
            boolean retried = false;
            for (Spec spec : work.specs) {
                Txn tx = engine.txOf(spec.id);
                if (tx.status.equals("aborted") && tx.attempts < maxAttempts) {
                    engine.readPhase(spec.id);
                    engine.validate(spec.id);
                    retried = true;
                }
            }
            if (!retried) break;
        }
        return engine;
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

    void shuffle(int[] values) {
        for (int i = values.length - 1; i > 0; i--) {
            int j = nextInt(i + 1);
            int tmp = values[i];
            values[i] = values[j];
            values[j] = tmp;
        }
    }
}

public class Solution {

    public static void main(String[] args) {
        OccEngine.Workload conflict = new OccEngine.Workload(new int[]{10, 20},
                List.of(new OccEngine.Spec(1, new int[]{0}, new int[]{0}, 1),
                        new OccEngine.Spec(2, new int[]{0}, new int[]{1}, 2)),
                List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 1}, new int[]{1, 2}));
        OccEngine engine = OccEngine.execute(conflict, 3);
        check(engine.txOf(1).status.equals("validated"), "T1 validates first");
        check(engine.txOf(2).attempts == 2 && engine.txOf(2).status.equals("validated"),
                "T2 aborted and then validated on the retry: attempts=" + engine.txOf(2).attempts);
        check(engine.aborts == 1, "exactly one abort, got " + engine.aborts);
        check(engine.log.stream().anyMatch(line -> line.contains("T2 (read set overlaps the write set of T1)")),
                "the conflicting transaction is named in the abort reason");
        check(Arrays.equals(engine.state, new int[]{11, 13}), "final state " + Arrays.toString(engine.state));
        check(Arrays.equals(engine.state, engine.serialReplay(engine.validationIds())),
                "state equals the serial replay in validation order");
        check(engine.invariantViolations == 0 && engine.invariantChecks >= 2,
                "per-validation invariant held " + engine.invariantChecks + " times");

        OccEngine.Workload reversed = new OccEngine.Workload(new int[]{5},
                List.of(new OccEngine.Spec(1, new int[]{0}, new int[]{0}, 10),
                        new OccEngine.Spec(2, new int[]{0}, new int[]{0}, 100)),
                List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 2}, new int[]{1, 1}));
        OccEngine byValidation = OccEngine.execute(reversed, 1);
        check(byValidation.txOf(2).status.equals("validated") && byValidation.txOf(2).timestamp == 1,
                "the first validator gets timestamp 1");
        check(byValidation.txOf(1).status.equals("aborted"), "the later validator loses: "
                + byValidation.txOf(1).reason);
        check(byValidation.state[0] == 105, "validation order decides the winner: " + byValidation.state[0]);
        check(Arrays.equals(byValidation.state, byValidation.serialReplay(byValidation.validationIds())),
                "the winner's value is the serial replay of the single validator");

        List<OccEngine.Spec> blindSpecs = List.of(new OccEngine.Spec(1, new int[0], new int[]{0}, 7),
                new OccEngine.Spec(2, new int[0], new int[]{1}, 9));
        OccEngine.Workload blind = new OccEngine.Workload(new int[]{0, 0}, blindSpecs,
                OccEngine.waveSchedule(blindSpecs));
        OccEngine blindEngine = OccEngine.execute(blind, 2);
        check(blindEngine.allValidated() && blindEngine.aborts == 0, "blind writers never conflict");
        check(Arrays.equals(blindEngine.state, new int[]{7, 9}),
                "blind writes applied: " + Arrays.toString(blindEngine.state));

        List<OccEngine.Spec> readerSpecs = List.of(new OccEngine.Spec(1, new int[]{0, 1}, new int[0], 0),
                new OccEngine.Spec(2, new int[]{1}, new int[]{2}, 3));
        OccEngine.Workload readers = new OccEngine.Workload(new int[]{4, 5, 6}, readerSpecs,
                List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 1}, new int[]{1, 2}));
        OccEngine readerEngine = OccEngine.execute(readers, 2);
        check(readerEngine.allValidated(), "a read-only transaction validates when its reads are untouched");
        check(Arrays.equals(readerEngine.state, new int[]{4, 5, 8}),
                "read-only transaction wrote nothing: " + Arrays.toString(readerEngine.state));
        check(readerEngine.invariantViolations == 0, "invariant held through the read-only run");

        int[] snapshotBefore = null;
        OccEngine.Workload noWrite = new OccEngine.Workload(new int[]{1},
                List.of(new OccEngine.Spec(1, new int[]{0}, new int[]{0}, 1),
                        new OccEngine.Spec(2, new int[]{0}, new int[]{0}, 2)),
                List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 1}, new int[]{1, 2}));
        OccEngine failing = new OccEngine(noWrite.initial, noWrite.specs);
        failing.readPhase(1);
        failing.readPhase(2);
        check(failing.validate(1), "first validator succeeds");
        snapshotBefore = failing.state.clone();
        check(!failing.validate(2), "second validator is rejected");
        check(Arrays.equals(snapshotBefore, failing.state), "the rejected transaction wrote nothing");
        check(failing.txOf(2).writtenValues != null && failing.txOf(2).status.equals("aborted"),
                "the workspace is discarded, not applied");

        OccEngine.Workload single = new OccEngine.Workload(new int[]{2, 3},
                List.of(new OccEngine.Spec(1, new int[]{0, 1}, new int[]{0}, 4)),
                List.of(new int[]{0, 1}, new int[]{1, 1}));
        OccEngine singleEngine = OccEngine.execute(single, 1);
        check(Arrays.equals(singleEngine.state, new int[]{9, 3}), "single transaction: "
                + Arrays.toString(singleEngine.state));

        OccEngine.Workload empty = new OccEngine.Workload(new int[]{8}, List.of(), List.of());
        OccEngine emptyEngine = OccEngine.execute(empty, 1);
        check(Arrays.equals(emptyEngine.state, new int[]{8}) && emptyEngine.validations == 0,
                "empty workload is a no-op");

        int trials = 60, mixedAborts = 0, mixedViolations = 0, mixedValidations = 0;
        for (int trial = 0; trial < trials; trial++) {
            OccEngine.Workload work = OccEngine.randomWorkload(5150L + 101L * trial, 4, 6, 2, 2, false);
            OccEngine mixed = OccEngine.execute(work, 6);
            check(mixed.allValidated(), "retries converge in trial " + trial + " -> " + statuses(mixed));
            mixedAborts += mixed.aborts;
            mixedViolations += mixed.invariantViolations;
            mixedValidations += mixed.validations;
            check(Arrays.equals(mixed.state, mixed.serialReplay(mixed.validationIds())),
                    "final state equals the serial replay in trial " + trial);
        }
        check(mixedViolations == 0, "no invariant violation across " + trials + " mixed schedules");
        check(mixedValidations == trials * 4, "every replay validated: " + mixedValidations);

        int highAborts = 0, lowAborts = 0;
        for (int trial = 0; trial < 40; trial++) {
            highAborts += OccEngine.execute(OccEngine.randomWorkload(777L + trial, 4, 5, 3, 2, true), 6).aborts;
            lowAborts += OccEngine.execute(OccEngine.partitionedWorkload(778L + trial, 4, 8, 3, 2), 6).aborts;
        }
        check(highAborts > 0, "high-contention wave schedules abort");
        check(lowAborts == 0, "disjoint slices have no false aborts, got " + lowAborts);
        check(highAborts > lowAborts, "contention drives the abort count: " + highAborts + " vs " + lowAborts);

        OccEngine.Workload disjoint = new OccEngine.Workload(new int[]{1, 1},
                List.of(new OccEngine.Spec(1, new int[]{0}, new int[]{0}, 1),
                        new OccEngine.Spec(2, new int[]{1}, new int[]{1}, 1)),
                List.of(new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 2}, new int[]{1, 1}));
        OccEngine disjointEngine = OccEngine.execute(disjoint, 1);
        check(disjointEngine.allValidated() && disjointEngine.aborts == 0, "disjoint sets never abort");

        System.out.println("OCC conflict: aborts=" + engine.aborts + " state " + Arrays.toString(engine.state)
                + " serializable=true");
        System.out.println("validation order decides the winner: T2 -> " + byValidation.state[0]
                + ", T1 aborted (" + byValidation.txOf(1).reason + ")");
        System.out.println("mixed trials=" + trials + " aborts=" + mixedAborts
                + " invariantViolations=" + mixedViolations
                + ", contention aborts high=" + highAborts + " low=" + lowAborts);
        System.out.println("PASS");
    }

    static String statuses(OccEngine engine) {
        StringBuilder sb = new StringBuilder();
        for (OccEngine.Txn tx : engine.txs.values()) {
            if (sb.length() > 0) sb.append(",");
            sb.append("T").append(tx.id()).append('=').append(tx.status);
        }
        return sb.toString();
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
