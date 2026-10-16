/** Task 618 - Implement linearizability checker for concurrent histories. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A history is a list of operations with an invocation and a response time, a thread, an argument and
 * the observed response. The checker searches for a linearization: it keeps the set of pending
 * operations (invoked but not yet answered) and tries every pending operation whose response is legal
 * in the current abstract state, recursing after applying it. Only "minimal" candidates are tried - an
 * operation that is completely preceded by another pending operation cannot be linearized first - and
 * failures are memoized on (state, pending set), which is what keeps the search small enough to run
 * exhaustively on the small histories used here.
 *
 * Two abstract models are provided: a read/write register and a FIFO queue. Verified: sequential
 * histories are accepted with the trivial witness; histories whose only legal order is not the real-time
 * order force the search to backtrack (the checker counts backtracks, so the check is that it really
 * searched); the classic non-linearizable register and queue histories are rejected; a history built by
 * interleaving a genuinely sequential execution is always accepted; every accepted witness is
 * independently replayed against the model and must reproduce every response, and every rejected
 * history explores a bounded number of states.
 */
import java.util.*;

class HistoryOp {
    static final int WRITE = 0, READ = 1;

    final int thread;
    final int kind;                 // WRITE (argument) or READ (observed response)
    final int value;
    final int invoke, response;

    HistoryOp(int thread, int kind, int value, int invoke, int response) {
        this.thread = thread;
        this.kind = kind;
        this.value = value;
        this.invoke = invoke;
        this.response = response;
    }

    String label() {
        return "t" + thread + (kind == WRITE ? ":write(" + value + ")" : ":read=" + value)
                + "[" + invoke + "," + response + "]";
    }

    boolean isWrite() { return kind == WRITE; }
}

interface MemoryModel {
    String name();

    Object initialState();

    /** Returns the state after the operation, or null when the observed response is impossible here. */
    Object apply(Object state, HistoryOp op);
}

class RegisterModel implements MemoryModel {
    private final int initial;

    RegisterModel(int initial) {
        this.initial = initial;
    }

    public String name() { return "register(initial=" + initial + ")"; }

    public Object initialState() { return Integer.valueOf(initial); }

    public Object apply(Object state, HistoryOp op) {
        if (op.kind == HistoryOp.WRITE) return Integer.valueOf(op.value);
        return ((Integer) state).intValue() == op.value ? state : null;
    }
}

class QueueModel implements MemoryModel {
    public String name() { return "queue"; }

    public Object initialState() { return ""; }

    public Object apply(Object state, HistoryOp op) {
        String queue = (String) state;
        if (op.kind == HistoryOp.WRITE) {
            return queue.isEmpty() ? String.valueOf(op.value) : queue + "," + op.value;
        }
        if (queue.isEmpty()) return null;
        int comma = queue.indexOf(',');
        int head = Integer.parseInt(comma < 0 ? queue : queue.substring(0, comma));
        if (head != op.value) return null;
        return comma < 0 ? "" : queue.substring(comma + 1);
    }
}

class LinearizabilityChecker {
    final MemoryModel model;
    private final Map<String, Boolean> failed = new HashMap<>();
    long explored;
    long backtracks;
    List<HistoryOp> witness = new ArrayList<>();

    LinearizabilityChecker(MemoryModel model) {
        this.model = model;
    }

    boolean check(List<HistoryOp> history) {
        failed.clear();
        explored = 0;
        backtracks = 0;
        witness = new ArrayList<>();
        List<HistoryOp> order = new ArrayList<>();
        long pending = (1L << history.size()) - 1;
        boolean ok = search(history, model.initialState(), pending, order);
        if (ok) witness = new ArrayList<>(order);
        return ok;
    }

    private boolean search(List<HistoryOp> history, Object state, long pending, List<HistoryOp> order) {
        explored++;
        if (pending == 0) return true;
        String key = state + "|" + pending;
        Boolean alreadyFailed = failed.get(key);
        if (alreadyFailed != null) {
            backtracks++;
            return false;
        }
        int earliestResponse = Integer.MAX_VALUE;
        for (int i = 0; i < history.size(); i++) {
            if ((pending & (1L << i)) != 0) earliestResponse = Math.min(earliestResponse, history.get(i).response);
        }
        for (int i = 0; i < history.size(); i++) {
            if ((pending & (1L << i)) == 0) continue;
            HistoryOp op = history.get(i);
            if (op.invoke > earliestResponse) continue;          // another pending op must come first
            Object next = model.apply(state, op);
            if (next == null) continue;
            order.add(op);
            if (search(history, next, pending & ~(1L << i), order)) return true;
            order.remove(order.size() - 1);
            backtracks++;
        }
        failed.put(key, Boolean.FALSE);
        return false;
    }
}

public class Solution {
    private static long state = 0xA4093822299F31D0L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static HistoryOp write(int thread, int value, int invoke, int response) {
        return new HistoryOp(thread, HistoryOp.WRITE, value, invoke, response);
    }

    static HistoryOp read(int thread, int value, int invoke, int response) {
        return new HistoryOp(thread, HistoryOp.READ, value, invoke, response);
    }

    /** An accepted witness must be a permutation of the history that reproduces every response. */
    static void verifyWitness(MemoryModel model, List<HistoryOp> history, List<HistoryOp> witness,
                              String label) {
        check(witness.size() == history.size(), label + ": the witness has every operation");
        check(new HashSet<>(witness).size() == history.size(), label + ": no operation is repeated");
        check(new HashSet<>(witness).containsAll(history), label + ": the witness uses the given operations");
        Object current = model.initialState();
        for (HistoryOp op : witness) {
            Object next = model.apply(current, op);
            check(next != null, label + ": the witness step " + op.label() + " is legal");
            current = next;
        }
    }

    static boolean accepts(LinearizabilityChecker checker, List<HistoryOp> history, String label) {
        boolean ok = checker.check(history);
        if (ok) verifyWitness(checker.model, history, checker.witness, label);
        check(checker.explored < 20000, label + ": the search stays bounded (" + checker.explored + " states)");
        return ok;
    }

    public static void main(String[] args) {
        LinearizabilityChecker register = new LinearizabilityChecker(new RegisterModel(0));

        // 1. a purely sequential history
        List<HistoryOp> sequential = Arrays.asList(
                write(0, 1, 1, 2), read(1, 1, 3, 4), write(0, 2, 5, 6), read(1, 2, 7, 8));
        check(accepts(register, sequential, "sequential"), "a sequential history is linearizable");
        check(register.witness.equals(sequential), "the witness is the real-time order");

        // 2. a sequential execution whose operations overlap: only one order is legal
        List<HistoryOp> needsBacktracking = Arrays.asList(
                write(0, 1, 1, 6), read(1, 0, 2, 3), read(2, 1, 4, 5));
        check(accepts(register, needsBacktracking, "overlapping"), "the overlapping history is linearizable");
        check(register.witness.equals(Arrays.asList(
                        needsBacktracking.get(1), needsBacktracking.get(0), needsBacktracking.get(2))),
                "the only legal linearization puts the read of 0 first");
        check(register.backtracks > 0, "the search had to backtrack");

        // 3. a completed write followed by a read that ignores it
        List<HistoryOp> stale = Arrays.asList(write(0, 1, 1, 2), read(1, 0, 3, 4));
        check(!accepts(register, stale, "stale read"), "a read after the write must not see the old value");

        // 4. two reads that contradict each other in real-time order
        List<HistoryOp> contradiction = Arrays.asList(
                write(0, 1, 1, 10), read(1, 1, 2, 4), read(2, 0, 5, 6));
        check(!accepts(register, contradiction, "contradiction"),
                "a later read cannot see the value before the write");

        // 5. reads that both see the initial value while a write is not yet ordered
        List<HistoryOp> concurrentReads = Arrays.asList(
                read(0, 0, 1, 4), read(1, 0, 2, 5), write(2, 7, 3, 6));
        check(accepts(register, concurrentReads, "concurrent reads"),
                "reads that both return 0 are fine before the write");

        // 6. the empty history and a single operation
        check(accepts(register, new ArrayList<>(), "empty"), "the empty history is linearizable");
        check(accepts(register, Arrays.asList(read(0, 0, 1, 2)), "single read"), "a single read of 0");
        check(!accepts(register, Arrays.asList(read(0, 5, 1, 2)), "single bad read"), "a single read of 5");

        // 7. FIFO queue histories
        LinearizabilityChecker queue = new LinearizabilityChecker(new QueueModel());
        List<HistoryOp> enqDeq = Arrays.asList(
                write(0, 1, 1, 2), write(1, 2, 3, 4), read(2, 1, 5, 6), read(3, 2, 7, 8));
        check(accepts(queue, enqDeq, "queue sequential"), "the FIFO execution is linearizable");
        List<HistoryOp> lifo = Arrays.asList(
                write(0, 1, 1, 2), write(1, 2, 3, 4), read(2, 2, 5, 6));
        check(!accepts(queue, lifo, "queue lifo"), "a queue cannot return the newest element first");
        List<HistoryOp> overlapEnq = Arrays.asList(
                write(0, 1, 1, 4), write(1, 2, 2, 5), read(2, 1, 3, 6));
        check(accepts(queue, overlapEnq, "queue overlap"), "the dequeue may be ordered before enqueue 2");
        List<HistoryOp> emptyDeq = Arrays.asList(read(0, 0, 1, 2));
        check(!accepts(queue, emptyDeq, "dequeue from empty"), "a dequeue from an empty queue is impossible");

        // 8. random interleavings of genuinely sequential executions are always linearizable
        int rejectedAfterFlip = 0;
        for (int trial = 0; trial < 60; trial++) {
            int ops = 4 + (int) ((nextLong() >>> 33) % 4);
            List<HistoryOp> history = new ArrayList<>();
            int clock = 1;
            int expected = 3;
            for (int k = 0; k < ops; k++) {
                boolean doesWrite = ((nextLong() >>> 33) & 1L) == 0 || k == 0;
                if (doesWrite) {
                    expected = 1 + (int) ((nextLong() >>> 33) % 5);
                    history.add(write(k % 3, expected, clock, clock + 1));
                } else {
                    history.add(read(k % 3, expected, clock, clock + 1));
                }
                clock += 2;
            }
            check(accepts(register, history, "sequential trial " + trial),
                    "a sequential execution must be linearizable");
            // stretch the intervals so that operations overlap while the order stays realizable
            List<HistoryOp> stretched = new ArrayList<>();
            for (HistoryOp op : history) {
                int extra = (int) ((nextLong() >>> 33) % 3);
                stretched.add(new HistoryOp(op.thread, op.kind, op.value, op.invoke, op.response + extra));
            }
            check(accepts(register, stretched, "stretched trial " + trial),
                    "stretching cannot destroy linearizability");
            // now flip one response: the result is either still linearizable (with a valid witness) or not
            int flip = (int) ((nextLong() >>> 33) % stretched.size());
            HistoryOp chosen = stretched.get(flip);
            HistoryOp broken = new HistoryOp(chosen.thread, HistoryOp.READ, 1 - chosen.value,
                    chosen.invoke, chosen.response);
            List<HistoryOp> candidate = new ArrayList<>(stretched);
            candidate.set(flip, broken);
            if (!accepts(register, candidate, "flipped trial " + trial)) rejectedAfterFlip++;
        }
        check(rejectedAfterFlip > 0, "flipping a response usually makes the history non-linearizable");
        System.out.println("flipped histories rejected: " + rejectedAfterFlip + " of 60, last search explored "
                + register.explored + " states");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
