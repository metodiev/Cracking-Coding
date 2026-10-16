/** Task 615 - Implement transactional memory simulator basics. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A software transactional memory is simulated on a single thread by driving every transaction from an
 * explicit schedule: a schedule is a list of thread ids and one "schedule step" executes exactly one
 * action of that thread's transaction, so the interleavings are reproduced faithfully instead of being
 * hoped for. Each transaction is a straight-line program of read/write actions over shared TVars. Reads
 * record the version stamp of the cell they saw, writes are buffered in a write set, and a commit
 * validates every recorded read version and then installs the write set with a fresh version stamp.
 * A transaction that fails validation aborts, drops its write set and is restarted at its first action,
 * which is what makes the simulation exhibit the classic lost-update and read-skew cases.
 *
 * Simplified: commit is a single sequential step (no lock stealing, no livelock control) and a retried
 * transaction restarts from the beginning, because the simulator only needs the memory semantics, not
 * hardware behaviour. Verified: conflicting schedules produce exactly the serial outcome of the
 * successful transactions in commit order (replayed independently on a plain copy of the cells), the
 * total of a set of transfers is conserved, aborted transactions leave no trace in any cell, retried
 * transactions eventually commit and the non-conflicting schedule aborts nothing; turning the read-set
 * validation off makes the lost update reappear, which is checked as the counterexample showing what
 * the validation prevents.
 */
import java.util.*;

class TVar {
    int value;
    long version;                       // bumped by every committed write, so a read can be validated

    TVar(int value) {
        this.value = value;
        this.version = 1;
    }
}

class Transaction {
    static final int READ = 0, WRITE = 1;

    final int id;
    final int[][] ops;                  // {kind, varIndex, delta}; a write stores view(var) + delta
    final int[] readVersion;            // version seen by the last read, -1 when untouched
    final int[] written;                // buffered value per variable
    final boolean[] dirty;
    int pc;
    boolean done;
    int aborts;

    Transaction(int id, int varCount, int[][] ops) {
        this.id = id;
        this.ops = ops;
        this.readVersion = new int[varCount];
        this.written = new int[varCount];
        this.dirty = new boolean[varCount];
        restart();
    }

    void restart() {
        pc = 0;
        Arrays.fill(readVersion, -1);
        Arrays.fill(dirty, false);
    }

    boolean pending() { return !done; }
}

class StmSimulator {
    final TVar[] cells;
    final Transaction[] transactions;
    final List<Integer> commitOrder = new ArrayList<>();
    final List<int[]> commitLog = new ArrayList<>();      // {transactionId, varIndex, value} per installed write
    boolean validateReads = true;
    int aborts = 0;

    StmSimulator(TVar[] cells, Transaction[] transactions) {
        this.cells = cells;
        this.transactions = transactions;
    }

    /** One action of one transaction: either the next program action or the commit attempt. */
    boolean step(int txnId) {
        Transaction t = transactions[txnId];
        if (t.done) return true;
        if (t.pc < t.ops.length) {
            int[] op = t.ops[t.pc++];
            int var = op[1];
            if (op[0] == Transaction.READ) {
                read(t, var);
            } else {
                t.written[var] = view(t, var) + op[2];
                t.dirty[var] = true;
            }
            return false;
        }
        return commit(t);
    }

    /** The transaction's own view: its buffered write, otherwise the cell (whose version it records). */
    int view(Transaction t, int var) {
        if (t.dirty[var]) return t.written[var];
        read(t, var);
        return cells[var].value;
    }

    private void read(Transaction t, int var) {
        if (t.dirty[var]) return;
        t.readVersion[var] = (int) cells[var].version;
    }

    /** Validates every read and installs the write set atomically; returns true when committed. */
    boolean commit(Transaction t) {
        if (validateReads) {
            for (int var = 0; var < cells.length; var++) {
                if (t.readVersion[var] >= 0 && t.readVersion[var] != cells[var].version) {
                    aborts++;
                    t.aborts++;
                    t.restart();
                    return false;
                }
            }
        }
        for (int var = 0; var < cells.length; var++) {
            if (t.dirty[var]) {
                cells[var].value = t.written[var];
                cells[var].version++;
                commitLog.add(new int[]{t.id, var, t.written[var]});
            }
        }
        t.done = true;
        commitOrder.add(t.id);
        return true;
    }

    boolean allDone() {
        for (Transaction t : transactions) if (!t.done) return false;
        return true;
    }

    /** Repeats the schedule until every transaction has committed; returns true when it did. */
    boolean run(int[] schedule, int maxRounds) {
        int rounds = 0;
        while (!allDone() && rounds < maxRounds) {
            for (int txnId : schedule) {
                if (transactions[txnId].done) continue;
                step(txnId);
            }
            rounds++;
        }
        return allDone();
    }

    int totalValue() {
        int sum = 0;
        for (TVar cell : cells) sum += cell.value;
        return sum;
    }
}

public class Solution {
    private static long state = 0x452821E638D01377L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static Transaction increment(int id, int varCount, int var, int delta) {
        return new Transaction(id, varCount, new int[][]{
                {Transaction.READ, var, 0}, {Transaction.WRITE, var, delta}});
    }

    static Transaction transfer(int id, int varCount, int from, int to, int amount) {
        return new Transaction(id, varCount, new int[][]{
                {Transaction.READ, from, 0}, {Transaction.READ, to, 0},
                {Transaction.WRITE, from, -amount}, {Transaction.WRITE, to, amount}});
    }

    static int[] repeat(int[] schedule, int times) {
        int[] out = new int[schedule.length * times];
        for (int i = 0; i < out.length; i++) out[i] = schedule[i % schedule.length];
        return out;
    }

    /** A plain sequential replay of the committed transactions, used as the serializability oracle. */
    static int[] serialOracle(int[] initial, List<int[]> commitLog) {
        int[] values = initial.clone();
        for (int[] write : commitLog) values[write[1]] = write[2];
        return values;
    }

    /** Reads the transaction's own ops against a plain model, in commit order. */
    static int[] replayInCommitOrder(int[] initial, List<Transaction> txns, List<Integer> commitOrder) {
        int[] values = initial.clone();
        for (int txnId : commitOrder) {
            Transaction t = txns.get(txnId);
            int[] view = values.clone();                  // per-variable view of this transaction
            boolean[] dirty = new boolean[values.length];
            for (int[] op : t.ops) {
                int var = op[1];
                if (op[0] == Transaction.READ) continue;
                view[var] = view[var] + op[2];
                values[var] = view[var];
                dirty[var] = true;
            }
        }
        return values;
    }

    static void checkSerialEquivalence(int[] initial, StmSimulator sim, List<Transaction> txns, String label) {
        int[] expected = replayInCommitOrder(initial, txns, sim.commitOrder);
        int[] actual = new int[initial.length];
        for (int i = 0; i < initial.length; i++) actual[i] = sim.cells[i].value;
        check(Arrays.equals(actual, expected), label + ": final state differs from the commit order replay");
        check(sim.commitOrder.size() == sim.transactions.length, label + ": every transaction committed");
        Set<Integer> seen = new HashSet<>(sim.commitOrder);
        check(seen.size() == sim.commitOrder.size(), label + ": each transaction commits exactly once");
    }

    public static void main(String[] args) {
        // 1. independent increments never conflict: nothing aborts
        TVar[] counters = new TVar[]{new TVar(0), new TVar(0), new TVar(0), new TVar(0)};
        List<Transaction> tasks = new ArrayList<>();
        for (int i = 0; i < 4; i++) tasks.add(increment(i, 4, i, 5));
        Transaction[] array = tasks.toArray(new Transaction[0]);
        StmSimulator independent = new StmSimulator(counters, array);
        check(independent.run(repeat(new int[]{0, 1, 2, 3}, 4), 20), "the independent schedule converges");
        check(independent.aborts == 0, "no false aborts when the transactions are disjoint");
        check(independent.totalValue() == 20, "every increment was applied");
        checkSerialEquivalence(new int[]{0, 0, 0, 0}, independent, tasks, "independent increments");

        // 2. the same cell from three transactions: the interleaving forces aborts, but no update is lost
        TVar[] counter = new TVar[]{new TVar(100)};
        List<Transaction> racers = new ArrayList<>();
        for (int i = 0; i < 3; i++) racers.add(increment(i, 1, 0, 1));
        StmSimulator contended = new StmSimulator(counter,
                racers.toArray(new Transaction[0]));
        check(contended.run(new int[]{0, 1, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2}, 20), "the contended schedule converges");
        check(counter[0].value == 103, "three increments land, none is lost");
        check(contended.aborts > 0, "the conflicting interleaving really aborts");
        checkSerialEquivalence(new int[]{100}, contended, racers, "contended increments");

        // 3. a conflicting transfer keeps the total and matches the commit order
        TVar[] accounts = new TVar[]{new TVar(100), new TVar(50), new TVar(25)};
        List<Transaction> transfers = new ArrayList<>();
        transfers.add(transfer(0, 3, 0, 1, 30));
        transfers.add(transfer(1, 3, 1, 2, 30));
        transfers.add(transfer(2, 3, 2, 0, 30));
        StmSimulator bank = new StmSimulator(accounts, transfers.toArray(new Transaction[0]));
        int[] roundRobin = new int[]{0, 1, 2};
        check(bank.run(repeat(roundRobin, 6), 20), "the transfer schedule converges");
        check(accounts[0].value + accounts[1].value + accounts[2].value == 175, "the total is conserved");
        checkSerialEquivalence(new int[]{100, 50, 25}, bank, transfers, "transfer round robin");

        // every interleaving must still match the serial outcome of its own commit order
        for (int trial = 0; trial < 40; trial++) {
            TVar[] cells = new TVar[]{new TVar(70), new TVar(30), new TVar(0)};
            List<Transaction> txns = new ArrayList<>();
            txns.add(transfer(0, 3, 0, 1, 10));
            txns.add(transfer(1, 3, 1, 2, 20));
            txns.add(transfer(2, 3, 2, 0, 5));
            txns.add(transfer(3, 3, 0, 2, 7));
            StmSimulator sim = new StmSimulator(cells, txns.toArray(new Transaction[0]));
            int[] schedule = new int[18];
            int rounds = 0;
            while (!sim.allDone() && rounds < 60) {
                for (int i = 0; i < 14; i++) schedule[i] = (int) ((nextLong() >>> 33) % 4);
                for (int i = 0; i < 4; i++) schedule[14 + i] = i;      // every transaction gets a slot
                sim.run(schedule, 1);
                rounds++;
            }
            check(sim.allDone(), "random schedule " + trial + " converges");
            int total = cells[0].value + cells[1].value + cells[2].value;
            check(total == 100, "random schedule " + trial + " conserves the total");
            checkSerialEquivalence(new int[]{70, 30, 0}, sim, txns, "random schedule " + trial);
        }

        // 4. an aborted transaction leaves nothing behind
        TVar[] one = new TVar[]{new TVar(0)};
        List<Transaction> pair = new ArrayList<>();
        pair.add(increment(0, 1, 0, 100));
        pair.add(increment(1, 1, 0, 1));
        StmSimulator aborts = new StmSimulator(one, pair.toArray(new Transaction[0]));
        // both read version 1, the first commits (value 100, version 2), the second must abort and retry
        check(aborts.run(new int[]{0, 1, 0, 1, 0, 1}, 20), "the aborting schedule converges");
        check(one[0].value == 101, "the retried transaction adds exactly one");
        check(aborts.aborts >= 1, "the second transaction aborted at least once");
        check(Arrays.equals(serialOracle(new int[]{0}, aborts.commitLog), new int[]{101}),
                "the commit log replays to the final value");

        // 5. counterexample: with the read-set validation switched off the update is lost
        TVar[] naive = new TVar[]{new TVar(0)};
        List<Transaction> naiveTxns = new ArrayList<>();
        naiveTxns.add(increment(0, 1, 0, 1));
        naiveTxns.add(increment(1, 1, 0, 1));
        StmSimulator buggy = new StmSimulator(naive, naiveTxns.toArray(new Transaction[0]));
        buggy.validateReads = false;
        check(buggy.run(new int[]{0, 1, 0, 1, 0, 1}, 20), "the unchecked schedule converges");
        check(naive[0].value == 1, "without validation one increment is lost");
        check(buggy.aborts == 0, "no validation means no aborts");

        // 6. a transaction sees its own writes and a read-only transaction always commits
        TVar[] solo = new TVar[]{new TVar(5), new TVar(7)};
        Transaction both = new Transaction(0, 2, new int[][]{
                {Transaction.READ, 0, 0}, {Transaction.WRITE, 0, 5},
                {Transaction.READ, 0, 0}, {Transaction.WRITE, 0, 5}});
        StmSimulator own = new StmSimulator(solo, new Transaction[]{both});
        check(own.run(new int[]{0}, 10), "the single transaction commits");
        check(solo[0].value == 15, "the second write builds on the transaction's own first write");
        check(solo[1].value == 7, "untouched cells keep their value");
        check(both.readVersion[0] == 1, "the read version is the one observed before the writes");

        Transaction reader = new Transaction(1, 2, new int[][]{{Transaction.READ, 0, 0}, {Transaction.READ, 1, 0}});
        StmSimulator reads = new StmSimulator(solo, new Transaction[]{reader});
        check(reads.run(new int[]{0}, 10) && reader.readVersion[1] == 1, "read-only transactions commit");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
