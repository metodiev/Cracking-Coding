/** Task 637 - Implement SNAPSHOT isolation anomalies demonstrator (write skew). (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * Model: a multi-version store in which a transaction is a small declarative program (READ item into
 * a slot, arithmetic on slots, GUARD slot >= bound, WRITE slot into item) split into phases, and the
 * interleaving is an explicit schedule of BEGIN / EXEC(tid, phase) / COMMIT steps - no threads, no
 * timing. BEGIN assigns a snapshot tick, every point read resolves the newest version whose commit
 * tick is <= the transaction's snapshot tick, writes are buffered in a private map, and COMMIT applies
 * first-committer-wins validation: a transaction aborts if an item it buffered was committed by
 * another transaction after its own snapshot. That is textbook SI: read-write conflicts never abort,
 * only concurrent write-write conflicts do.
 *
 * Simplified (deliberately): versions are stamped with a logical tick instead of an MVCC xmin/xmax
 * chain, there are no predicate reads, no garbage collection and aborting transactions are never
 * retried. What is verified nevertheless: a snapshot read is stable and equal to the store state at
 * the snapshot tick; the lost-update anomaly is prevented by first-committer-wins; the classic
 * doctors/overdraft write skew is really reachable under this engine and really is not serializable
 * (brute force over every serial order of the committed transactions, replaying their program bodies,
 * finds no matching final state) and really does violate the check x + y >= 100, while the same two
 * transactions executed serially refuse the second withdrawal and keep the invariant.
 */
import java.util.*;

class MvccEngine {

    static final class Abort extends RuntimeException {
        Abort(String message) { super(message); }
    }

    /** One interpreted instruction of a transaction body. Slots are transaction-local variables. */
    static final class Instr {
        final String op;
        final String[] args;
        final int n;

        private Instr(String op, String[] args, int n) {
            this.op = op; this.args = args; this.n = n;
        }
        static Instr read(String item, String slot) {
            return new Instr("READ", new String[]{item, slot}, 0);
        }
        static Instr write(String slot, String item) {
            return new Instr("WRITE", new String[]{slot, item}, 0);
        }
        static Instr constant(String slot, int value) {
            return new Instr("CONST", new String[]{slot}, value);
        }
        static Instr subtract(String slot, int value, String out) {
            return new Instr("SUB", new String[]{slot, out}, value);
        }
        static Instr sum(String x, String y, String out) {
            return new Instr("SUM", new String[]{x, y, out}, 0);
        }
        static Instr guardAtLeast(String slot, int bound) {
            return new Instr("GUARD", new String[]{slot}, bound);
        }
    }

    static final class Version {
        final int value, commitTick, tid;
        Version(int value, int commitTick, int tid) {
            this.value = value; this.commitTick = commitTick; this.tid = tid;
        }
    }

    static final class Tx {
        final int id;
        final int snapshotTick;
        final Map<String, Integer> locals = new LinkedHashMap<>();
        final LinkedHashMap<String, Integer> reads = new LinkedHashMap<>();
        final LinkedHashMap<String, Integer> writes = new LinkedHashMap<>();
        int pc;
        int commitTick = -1;
        String status = "active";
        String reason = "";

        Tx(int id, int snapshotTick) { this.id = id; this.snapshotTick = snapshotTick; }

        boolean active() { return status.equals("active"); }
    }

    final Map<String, List<Version>> chains = new LinkedHashMap<>();
    final Map<Integer, Tx> txs = new LinkedHashMap<>();
    final List<String> log = new ArrayList<>();
    int tick;

    MvccEngine(Map<String, Integer> initial) {
        for (Map.Entry<String, Integer> e : initial.entrySet()) {
            List<Version> chain = new ArrayList<>();
            chain.add(new Version(e.getValue(), 0, 0));
            chains.put(e.getKey(), chain);
        }
    }

    /** Newest version visible to a transaction that started at {@code snapshotTick}. */
    int readAt(String item, int snapshotTick) {
        Version best = chains.get(item).get(0);
        for (Version v : chains.get(item)) if (v.commitTick <= snapshotTick) best = v;
        return best.value;
    }

    int latest(String item) {
        List<Version> chain = chains.get(item);
        return chain.get(chain.size() - 1).value;
    }

    Map<String, Integer> state() {
        Map<String, Integer> out = new TreeMap<>();
        for (String item : chains.keySet()) out.put(item, latest(item));
        return out;
    }

    Map<String, Integer> snapshotState(int snapshotTick) {
        Map<String, Integer> out = new TreeMap<>();
        for (String item : chains.keySet()) out.put(item, readAt(item, snapshotTick));
        return out;
    }

    void begin(int tid) {
        tick++;
        txs.put(tid, new Tx(tid, tick));
        log.add("BEGIN T" + tid + " snapshot@" + tick);
    }

    boolean execute(int tid, int phase, Map<Integer, List<List<Instr>>> program) {
        Tx tx = txs.get(tid);
        if (tx == null || !tx.active()) {
            log.add("SKIP EXEC T" + tid + " phase " + phase);
            return false;
        }
        try {
            for (Instr ins : program.get(tid).get(phase)) apply(ins, tx);
        } catch (Abort abort) {
            tx.status = "aborted";
            tx.reason = abort.getMessage();
            tx.writes.clear();
            log.add("ABORT T" + tid + " (" + tx.reason + ")");
            return false;
        }
        tx.pc = phase + 1;
        log.add("EXEC T" + tid + " phase " + phase);
        return true;
    }

    private void apply(Instr ins, Tx tx) {
        switch (ins.op) {
            case "READ": {
                int value = readAt(ins.args[0], tx.snapshotTick);
                tx.locals.put(ins.args[1], value);
                tx.reads.put(ins.args[0], value);
                break;
            }
            case "CONST":
                tx.locals.put(ins.args[0], ins.n);
                break;
            case "SUB":
                tx.locals.put(ins.args[1], tx.locals.get(ins.args[0]) - ins.n);
                break;
            case "SUM":
                tx.locals.put(ins.args[2], tx.locals.get(ins.args[0]) + tx.locals.get(ins.args[1]));
                break;
            case "GUARD": {
                int have = tx.locals.get(ins.args[0]);
                if (have < ins.n) throw new Abort("guard " + ins.args[0] + "=" + have + " < " + ins.n);
                break;
            }
            case "WRITE":
                tx.writes.put(ins.args[1], tx.locals.get(ins.args[0]));
                break;
            default:
                throw new IllegalStateException("unknown op " + ins.op);
        }
    }

    /** First-committer-wins: only concurrent write-write conflicts abort. */
    boolean commit(int tid) {
        Tx tx = txs.get(tid);
        if (tx == null || !tx.active()) {
            log.add("SKIP COMMIT T" + tid);
            return false;
        }
        for (String item : tx.writes.keySet()) {
            for (Version v : chains.get(item)) {
                if (v.commitTick > tx.snapshotTick && v.tid != tid) {
                    tx.status = "aborted";
                    tx.reason = "write-write conflict on " + item + " with T" + v.tid;
                    tx.writes.clear();
                    log.add("ABORT T" + tid + " (" + tx.reason + ")");
                    return false;
                }
            }
        }
        tick++;
        tx.status = "committed";
        tx.commitTick = tick;
        for (Map.Entry<String, Integer> w : tx.writes.entrySet()) {
            chains.get(w.getKey()).add(new Version(w.getValue(), tick, tid));
        }
        log.add("COMMIT T" + tid + "@" + tick);
        return true;
    }

    String statuses() {
        StringBuilder sb = new StringBuilder();
        for (Tx tx : txs.values()) {
            if (sb.length() > 0) sb.append(",");
            sb.append("T").append(tx.id).append('=').append(tx.status);
        }
        return sb.toString();
    }

    List<Integer> committedIds() {
        List<Integer> out = new ArrayList<>();
        for (Tx tx : txs.values()) if (tx.status.equals("committed")) out.add(tx.id);
        return out;
    }

    /** Serial replay: no MVCC, each transaction sees everything committed before it, no validation. */
    static Map<String, Integer> serialRun(Map<String, Integer> initial,
                                          Map<Integer, List<List<Instr>>> program,
                                          List<Integer> order) {
        MvccEngine engine = new MvccEngine(initial);
        for (int tid : order) {
            engine.tick++;
            MvccEngine.Tx tx = new MvccEngine.Tx(tid, engine.tick);
            engine.txs.put(tid, tx);
            boolean alive = true;
            for (int phase = 0; phase < program.get(tid).size() && alive; phase++) {
                alive = engine.execute(tid, phase, program);
            }
            if (alive) engine.commit(tid);
        }
        return engine.state();
    }

    /** True when the SI outcome is equivalent to some serial order of the committed transactions. */
    static boolean serializable(Map<String, Integer> initial,
                                Map<Integer, List<List<Instr>>> program,
                                List<Integer> committed, Map<String, Integer> siState) {
        for (List<Integer> order : permutations(committed)) {
            if (serialRun(initial, program, order).equals(siState)) return true;
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

    /** Doctors-on-call / overdraft write skew: withdraw 100 if the pair still holds the reserve. */
    static Map<Integer, List<List<MvccEngine.Instr>>> writeSkewProgram() {
        Map<Integer, List<List<MvccEngine.Instr>>> program = new LinkedHashMap<>();
        program.put(1, List.of(
                List.of(MvccEngine.Instr.read("x", "a"), MvccEngine.Instr.read("y", "b"),
                        MvccEngine.Instr.sum("a", "b", "total"), MvccEngine.Instr.guardAtLeast("total", 200)),
                List.of(MvccEngine.Instr.subtract("a", 100, "nx"), MvccEngine.Instr.write("nx", "x"))));
        program.put(2, List.of(
                List.of(MvccEngine.Instr.read("x", "a"), MvccEngine.Instr.read("y", "b"),
                        MvccEngine.Instr.sum("a", "b", "total"), MvccEngine.Instr.guardAtLeast("total", 200)),
                List.of(MvccEngine.Instr.subtract("b", 100, "ny"), MvccEngine.Instr.write("ny", "y"))));
        return program;
    }

    static Map<Integer, List<List<MvccEngine.Instr>>> incrementProgram() {
        Map<Integer, List<List<MvccEngine.Instr>>> program = new LinkedHashMap<>();
        List<MvccEngine.Instr> body = List.of(MvccEngine.Instr.read("c", "cur"),
                MvccEngine.Instr.constant("one", 1), MvccEngine.Instr.sum("cur", "one", "next"),
                MvccEngine.Instr.write("next", "c"));
        program.put(1, List.of(body));
        program.put(2, List.of(body));
        return program;
    }

    public static void main(String[] args) {
        Map<String, Integer> initial = new LinkedHashMap<>();
        initial.put("x", 100);
        initial.put("y", 100);
        Map<Integer, List<List<MvccEngine.Instr>>> skew = writeSkewProgram();

        MvccEngine engine = new MvccEngine(initial);
        engine.begin(1);
        engine.begin(2);
        engine.execute(1, 0, skew);
        engine.execute(2, 0, skew);
        engine.execute(1, 1, skew);
        engine.commit(1);
        engine.execute(2, 1, skew);
        engine.commit(2);

        check(engine.statuses().equals("T1=committed,T2=committed"), "both skew writers commit under SI");
        check(engine.state().toString().equals("{x=0, y=0}"), "SI final state: " + engine.state());
        int total = engine.state().get("x") + engine.state().get("y");
        check(total < 100, "write skew broke the invariant x + y >= 100 (got " + total + ")");
        check(engine.txs.get(2).reads.get("x") == 100 && engine.txs.get(2).reads.get("y") == 100,
                "T2 read the pre-T1 snapshot (100, 100)");
        check(engine.latest("x") == 0, "T1's withdrawn value is visible after commit");
        check(engine.snapshotState(engine.txs.get(1).snapshotTick).toString().equals("{x=100, y=100}"),
                "T1 snapshot state");
        check(engine.serializable(initial, skew, engine.committedIds(), engine.state()) == false,
                "SI outcome has no equivalent serial order");

        Map<String, Integer> serial = MvccEngine.serialRun(initial, skew, List.of(1, 2));
        check(serial.toString().equals("{x=0, y=100}"), "serial run: " + serial);
        check(serial.get("x") + serial.get("y") >= 100, "serial run keeps the invariant");
        check(MvccEngine.serialRun(initial, skew, List.of(2, 1)).get("x") == 100,
                "the second withdrawal is refused in every serial order");
        check(MvccEngine.serialRun(initial, skew, List.of(2, 1)).get("y") == 0,
                "the first serial withdrawal does go through");
        check(engine.log.contains("EXEC T1 phase 0") && engine.log.contains("COMMIT T2@4"),
                "schedule log records the interleaving");

        MvccEngine consistency = new MvccEngine(initial);
        Map<Integer, List<List<MvccEngine.Instr>>> readerPlus = new LinkedHashMap<>(skew);
        readerPlus.put(3, List.of(List.of(MvccEngine.Instr.read("x", "a"),
                MvccEngine.Instr.read("y", "b"), MvccEngine.Instr.sum("a", "b", "t"))));
        consistency.begin(1);
        consistency.begin(2);
        consistency.execute(1, 0, readerPlus);
        consistency.execute(2, 0, readerPlus);
        consistency.execute(1, 1, readerPlus);
        consistency.commit(1);
        consistency.begin(3);
        consistency.execute(2, 1, readerPlus);
        consistency.commit(2);
        consistency.execute(3, 0, readerPlus);
        consistency.commit(3);
        check(consistency.txs.get(3).reads.get("x") == 0 && consistency.txs.get(3).reads.get("y") == 100,
                "T3 sees the prefix state (x=0, y=100)");
        check(consistency.snapshotState(consistency.txs.get(3).snapshotTick).toString().equals("{x=0, y=100}"),
                "T3's snapshot is the committed prefix at its begin tick");
        check(consistency.readAt("y", consistency.txs.get(3).snapshotTick) != consistency.latest("y"),
                "T3's repeatable read is stale relative to the final state");
        check(consistency.statuses().equals("T1=committed,T2=committed,T3=committed"), "T3 read only, never aborts");

        Map<String, Integer> counter = new LinkedHashMap<>();
        counter.put("c", 100);
        Map<Integer, List<List<MvccEngine.Instr>>> increments = incrementProgram();
        MvccEngine lost = new MvccEngine(counter);
        lost.begin(1);
        lost.begin(2);
        lost.execute(1, 0, increments);
        lost.execute(2, 0, increments);
        lost.commit(1);
        lost.execute(2, 0, increments);
        lost.commit(2);
        check(lost.statuses().equals("T1=committed,T2=aborted"), "lost update blocked: " + lost.statuses());
        check(lost.txs.get(2).reason.equals("write-write conflict on c with T1"), "abort reason: " + lost.txs.get(2).reason);
        check(lost.state().get("c") == 101, "counter incremented exactly once");
        check(lost.serializable(counter, increments, lost.committedIds(), lost.state()),
                "retrying nothing leaves the single-transaction serial history");
        check(MvccEngine.serialRun(counter, increments, List.of(1, 2)).get("c") == 102,
                "a serial run of both increments reaches 102");
        check(MvccEngine.serialRun(counter, increments, List.of(1)).get("c") == 101, "single increment is 101");

        Map<String, Integer> disjoint = new LinkedHashMap<>();
        disjoint.put("p", 5);
        disjoint.put("q", 7);
        Map<Integer, List<List<MvccEngine.Instr>>> independent = new LinkedHashMap<>();
        independent.put(1, List.of(List.of(MvccEngine.Instr.constant("v", 9), MvccEngine.Instr.write("v", "p"))));
        independent.put(2, List.of(List.of(MvccEngine.Instr.constant("v", 11), MvccEngine.Instr.write("v", "q"))));
        MvccEngine parallel = new MvccEngine(disjoint);
        parallel.begin(1);
        parallel.begin(2);
        parallel.execute(1, 0, independent);
        parallel.execute(2, 0, independent);
        parallel.commit(2);
        parallel.commit(1);
        check(parallel.statuses().equals("T1=committed,T2=committed"), "no false aborts on disjoint writes");
        check(parallel.state().toString().equals("{p=9, q=11}"), "disjoint state: " + parallel.state());

        Map<String, Integer> guarded = new LinkedHashMap<>();
        guarded.put("z", 1);
        Map<Integer, List<List<MvccEngine.Instr>>> guardedProgram = new LinkedHashMap<>();
        guardedProgram.put(1, List.of(
                List.of(MvccEngine.Instr.constant("v", 5), MvccEngine.Instr.write("v", "z")),
                List.of(MvccEngine.Instr.constant("m", 0), MvccEngine.Instr.guardAtLeast("m", 10))));
        MvccEngine aborted = new MvccEngine(guarded);
        aborted.begin(1);
        aborted.execute(1, 0, guardedProgram);
        aborted.execute(1, 1, guardedProgram);
        aborted.commit(1);
        check(aborted.statuses().equals("T1=aborted"), "guard failure aborts the transaction");
        check(aborted.state().get("z") == 1, "buffered writes of an aborted transaction are discarded");
        check(aborted.chains.get("z").size() == 1, "no version was appended for the aborted transaction");

        MvccEngine empty = new MvccEngine(initial);
        check(empty.state().equals(initial), "empty schedule leaves the store untouched");
        check(empty.committedIds().isEmpty(), "nothing committed in an empty schedule");

        System.out.println("SI write skew: final " + engine.state() + ", x+y=" + total
                + " < 100 and serializable=false");
        System.out.println("first-committer-wins: " + lost.statuses() + " on c -> " + lost.state());
        System.out.println("pass checks: SI snapshots, lost update, disjoint writes, guard abort, empty schedule");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
