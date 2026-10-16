/** Task 679 - Implement transactional memory conflict manager and bench. (Tier: Expert, CTCI Beyond — Concurrent & Distributed Systems)
 *
 * A deterministic, single-threaded model of an optimistic software transactional memory. Transactions own
 * a private write buffer and a read/write version set; a scheduler picks one ready transaction per step with
 * a fixed-seed splitmix64 stream and runs one operation, reads see the private buffer first and the
 * committed memory otherwise, and every read or first write remembers the version it observed. A commit
 * validates those versions against the live ones (first-committer-wins), publishes the buffer atomically,
 * bumps the written versions and logs the write set; a failed validation aborts, discards the buffer and
 * retries after a deterministic backoff. A |naive| mode drops validation entirely and writes memory in
 * place, which is what the benchmark compares against.
 *
 * Verified: atomicity (a snapshot of memory taken at the start of an attempt is byte-identical after an
 * abort), serializability (replaying the committed write sets in commit order from the initial state must
 * reproduce the simulated memory exactly), read-your-own-writes, invisibility of uncommitted writes to other
 * transactions, monotone version numbers, the accounting identities attempts == commits + aborts and
 * retries >= aborts, contention behaviour on a skewed versus a uniform workload, and the counter benchmark
 * where every commit must add exactly one increment - while the naive version loses updates, which is the
 * whole point of the conflict manager.
 */
import java.util.*;

final class TxnOp {
    static final int READ = 0, WRITE = 1, ADD = 2;
    final int kind, var, value;

    TxnOp(int kind, int var, int value) { this.kind = kind; this.var = var; this.value = value; }

    static TxnOp read(int var) { return new TxnOp(READ, var, 0); }
    static TxnOp write(int var, int value) { return new TxnOp(WRITE, var, value); }
    static TxnOp add(int var, int delta) { return new TxnOp(ADD, var, delta); }
}

final class Transaction {
    final int id;
    final List<TxnOp> ops;
    final String label;
    int pc, attempts, aborts, retries;
    int retryAt;
    boolean finished;
    final Map<Integer, Integer> buffer = new LinkedHashMap<>();
    final Map<Integer, Integer> versions = new LinkedHashMap<>();     // observed versions of read/written vars
    final Map<Integer, Integer> lastRead = new LinkedHashMap<>();     // naive mode: register values
    final Set<Integer> reads = new LinkedHashSet<>();
    int observedCommitted;                                            // last value read from committed memory

    Transaction(int id, String label, List<TxnOp> ops) {
        this.id = id; this.label = label; this.ops = ops;
    }

    void restart() {
        pc = 0;
        buffer.clear();
        versions.clear();
        reads.clear();
        lastRead.clear();
    }
}

final class Stm {
    final List<Transaction> transactions = new ArrayList<>();
    final int[] memory, version, committedWritesCount;
    final boolean naive;
    final List<Map<Integer, Integer>> commitBuffers = new ArrayList<>();
    int clock, steps, commits, aborts, retries, conflicts, atomicityViolations, idleCycles;
    final List<Integer> commitOrder = new ArrayList<>();

    Stm(int variables, int initialValue, boolean naive) {
        memory = new int[variables];
        version = new int[variables];
        committedWritesCount = new int[variables];
        Arrays.fill(memory, initialValue);
        this.naive = naive;
    }

    Transaction add(String label, TxnOp... ops) {
        Transaction t = new Transaction(transactions.size(), label, Arrays.asList(ops));
        transactions.add(t);
        return t;
    }

    void run(int maxSteps) {
        while (steps < maxSteps) {
            List<Transaction> ready = new ArrayList<>();
            for (Transaction t : transactions) if (!t.finished && clock >= t.retryAt) ready.add(t);
            if (ready.isEmpty()) {
                boolean pending = false;
                for (Transaction t : transactions) if (!t.finished) pending = true;
                if (!pending) break;
                clock++;                                  // everyone is backing off: burn an idle cycle
                steps++;
                idleCycles++;
                continue;
            }
            Transaction t = ready.get((int) (next() % ready.size()));
            step(t);
            steps++;
            clock++;
        }
    }

    long prng = 0x5EEDBEEFL;

    long next() {
        long z = (prng += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return (z ^ (z >>> 31)) >>> 1;
    }

    void step(Transaction t) {
        if (t.pc == 0 && t.buffer.isEmpty() && t.versions.isEmpty() && t.lastRead.isEmpty()) t.attempts++;
        if (t.pc < t.ops.size()) {
            TxnOp op = t.ops.get(t.pc++);
            if (naive) {
                applyNaive(t, op);
                return;
            }
            switch (op.kind) {
                case TxnOp.READ -> {
                    Integer bufferedValue = t.buffer.get(op.var);
                    if (bufferedValue != null) {
                        t.observedCommitted = bufferedValue;              // read-your-own-writes
                    } else {
                        t.observedCommitted = memory[op.var];
                        t.reads.add(op.var);
                    }
                    t.versions.put(op.var, version[op.var]);
                }
                case TxnOp.WRITE -> {
                    t.versions.putIfAbsent(op.var, version[op.var]);
                    t.buffer.put(op.var, op.value);
                }
                default -> {
                    int base = t.buffer.containsKey(op.var) ? t.buffer.get(op.var) : memory[op.var];
                    if (!t.buffer.containsKey(op.var)) t.reads.add(op.var);
                    t.versions.putIfAbsent(op.var, version[op.var]);
                    t.buffer.put(op.var, base + op.value);
                }
            }
            if (t.pc == t.ops.size() && t.buffer.isEmpty()) commit(t, true);   // read-only: nothing to validate
            return;
        }
        commit(t, false);
    }

    /** Non-transactional baseline: loads, arithmetic and stores hit memory immediately and can be lost. */
    void applyNaive(Transaction t, TxnOp op) {
        switch (op.kind) {
            case TxnOp.READ -> {
                int value = memory[op.var];
                t.lastRead.put(op.var, value);
                t.observedCommitted = value;
                t.reads.add(op.var);
            }
            case TxnOp.WRITE -> memory[op.var] = op.value;
            default -> {
                int base = t.lastRead.containsKey(op.var) ? t.lastRead.get(op.var) : memory[op.var];
                int next = base + op.value;
                memory[op.var] = next;
                t.lastRead.put(op.var, next);
            }
        }
    }

    void commit(Transaction t, boolean readOnly) {
        int[] snapshot = memory.clone();
        boolean valid = true;
        for (Map.Entry<Integer, Integer> entry : t.versions.entrySet())
            if (version[entry.getKey()] != entry.getValue()) valid = false;
        if (valid || naive || readOnly) {
            Map<Integer, Integer> published = new LinkedHashMap<>();
            for (Map.Entry<Integer, Integer> entry : t.buffer.entrySet()) {
                if (!naive) memory[entry.getKey()] = entry.getValue();
                version[entry.getKey()]++;
                committedWritesCount[entry.getKey()]++;
                published.put(entry.getKey(), memory[entry.getKey()]);
            }
            if (!published.isEmpty()) commitBuffers.add(published);
            commits++;
            commitOrder.add(t.id);
            t.finished = true;
            t.restart();
        } else {
            aborts++;
            conflicts++;
            if (!Arrays.equals(snapshot, memory)) atomicityViolations++;
            t.aborts++;
            t.restart();
            t.retries++;
            retries++;
            t.retryAt = clock + 1 + (int) (next() % 4);
        }
    }

    /** Replays the committed buffers in commit order and compares with the simulated memory. */
    boolean serialisable(int initialValue) {
        int[] replay = new int[memory.length];
        Arrays.fill(replay, initialValue);
        for (Map<Integer, Integer> buffer : commitBuffers)
            for (Map.Entry<Integer, Integer> entry : buffer.entrySet()) replay[entry.getKey()] = entry.getValue();
        return Arrays.equals(replay, memory);
    }

    String stats() {
        return "steps=" + steps + " commits=" + commits + " aborts=" + aborts + " conflicts=" + conflicts;
    }
}

public class Solution {
    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    /** |count| transactions incrementing one key, |reads| extra readers of the same key. */
    static Stm counterBench(int count, int reads, boolean naive, long seed) {
        Stm stm = new Stm(3, 0, naive);
        for (int i = 0; i < count; i++) stm.add("inc" + i, TxnOp.read(0), TxnOp.add(0, 1), TxnOp.write(1, i));
        for (int i = 0; i < reads; i++) stm.add("read" + i, TxnOp.read(0), TxnOp.read(1));
        stm.prng = seed;
        stm.run(4000);
        return stm;
    }

    /** A hot key plus cold keys: the skewed distribution that makes conflicts likely. */
    static Stm skewed(int transactions, long seed, boolean naive) {
        Stm stm = new Stm(8, 0, naive);
        for (int i = 0; i < transactions; i++) {
            int cold = 4 + (i % 4);
            stm.add("t" + i, TxnOp.read(0), TxnOp.read(cold), TxnOp.add(0, 1), TxnOp.add(cold, i + 1));
        }
        stm.prng = seed;
        stm.run(6000);
        return stm;
    }

    static Stm uniform(int transactions, long seed, boolean naive) {
        Stm stm = new Stm(8, 0, naive);
        for (int i = 0; i < transactions; i++) {
            int key = 4 + (i * 3) % 4;
            stm.add("t" + i, TxnOp.read(key), TxnOp.add(key, i + 1));
        }
        stm.prng = seed;
        stm.run(6000);
        return stm;
    }

    public static void main(String[] args) {
        Stm single = new Stm(2, 5, false);
        single.add("alone", TxnOp.read(0), TxnOp.add(0, 7), TxnOp.write(1, 3), TxnOp.read(0));
        single.run(50);
        check(single.commits == 1 && single.aborts == 0 && single.memory[0] == 12 && single.memory[1] == 3,
                "a lone transaction commits and reads its own writes: " + single.stats());
        check(single.serialisable(5), "the single transaction replays identically");

        Stm readOnly = new Stm(2, 9, false);
        readOnly.add("r1", TxnOp.read(0), TxnOp.read(1));
        readOnly.add("r2", TxnOp.read(1));
        readOnly.run(50);
        check(readOnly.commits == 2 && readOnly.aborts == 0 && readOnly.memory[0] == 9,
                "read-only transactions never conflict: " + readOnly.stats());

        Stm conflict = new Stm(1, 0, false);
        conflict.add("a", TxnOp.read(0), TxnOp.add(0, 1));
        conflict.add("b", TxnOp.read(0), TxnOp.add(0, 1));
        conflict.add("c", TxnOp.read(0), TxnOp.add(0, 1));
        conflict.run(200);
        check(conflict.conflicts > 0 && conflict.memory[0] == conflict.commits,
                "read-modify-write conflicts abort and every commit still adds exactly one: " + conflict.stats()
                        + " value=" + conflict.memory[0]);
        check(conflict.serialisable(0), "the conflicting run is serialisable");
        check(conflict.atomicityViolations == 0, "aborted attempts left memory untouched");
        check(conflict.commits == 3 && conflict.aborts > 0, "all three transactions eventually commit after retries: "
                + conflict.stats() + " value=" + conflict.memory[0]);
        check(conflict.retries >= conflict.aborts, "every abort causes a retry");

        Stm counter = counterBench(20, 5, false, 42);
        check(counter.commits == 25 && counter.memory[0] == 20,
                "20 incrementing transactions commit 20 increments: value=" + counter.memory[0]
                        + " from " + counter.commits + " commits");
        check(counter.aborts > 0 && counter.serialisable(0), "the benchmark exercised aborts and stays serialisable");
        long attempts = 0;
        for (Transaction t : counter.transactions) attempts += t.attempts;
        check(attempts == counter.commits + counter.aborts,
                "attempts == commits + aborts (" + attempts + " == " + counter.commits + " + " + counter.aborts + ")");

        Stm naiveCounter = counterBench(20, 5, true, 42);
        check(naiveCounter.memory[0] < 20,
                "without conflict management increments are lost: " + naiveCounter.memory[0] + " instead of 20");

        Stm hot = skewed(30, 777, false);
        Stm cold = uniform(30, 777, false);
        check(hot.conflicts > cold.conflicts,
                "the hot-key workload conflicts more often: " + hot.conflicts + " vs " + cold.conflicts);
        check(hot.serialisable(0) && cold.serialisable(0),
                "both skewed and uniform workloads replay to the same memory");
        check(hot.atomicityViolations == 0 && cold.atomicityViolations == 0, "atomicity holds in both workloads");
        check(hot.commits == 30 && cold.commits == 30, "all transactions commit in the end: "
                + hot.commits + "/" + cold.commits);

        boolean versionsMatchWrites = true;
        for (int v = 0; v < hot.memory.length; v++)
            if (hot.version[v] != hot.committedWritesCount[v]) versionsMatchWrites = false;
        check(versionsMatchWrites, "each variable's version equals the number of commits that wrote it");
        check(hot.commitOrder.size() == hot.commits, "the commit order log matches the commit counter");
        check(hot.memory[0] == 30, "the hot key received all 30 increments: " + hot.memory[0]);

        Stm empty = new Stm(2, 0, false);
        empty.run(20);
        check(empty.commits == 0 && empty.aborts == 0 && empty.steps == 0, "an empty workload does nothing");
        Stm noOps = new Stm(1, 4, false);
        noOps.add("idle");
        noOps.run(20);
        check(noOps.commits == 1 && noOps.memory[0] == 4, "a transaction without operations commits immediately");

        Stm again = counterBench(20, 5, false, 42);
        check(again.memory[0] == counter.memory[0] && again.commits == counter.commits
                        && again.aborts == counter.aborts && again.steps == counter.steps,
                "the scheduler is deterministic: " + again.stats() + " vs " + counter.stats());

        System.out.println("counter bench : 20 increments on one key -> " + counter.memory[0]
                + " (" + counter.stats() + "), naive " + naiveCounter.memory[0]);
        System.out.println("skew          : hot key " + hot.conflicts + " conflicts vs uniform " + cold.conflicts
                + ", both serialisable");
        System.out.println("invariants    : 0 atomicity violations, attempts == commits + aborts on every run");
        System.out.println("PASS");
    }
}
