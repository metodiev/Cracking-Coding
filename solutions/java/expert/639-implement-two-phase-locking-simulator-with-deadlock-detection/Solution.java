/** Task 639 - Implement two-phase locking simulator with deadlock detection. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A deterministic strict-2PL simulator. Transactions are flat operation lists (LOCK_S / LOCK_X /
 * READ / ADD / WRITE / COMMIT / ABORT) over named integer items, and the interleaving is produced by
 * the lock manager itself: the scheduler gives every active transaction exactly one operation per
 * round in ascending transaction-id order, an operation that needs an incompatible lock leaves the
 * transaction waiting (the lock manager records the denied request), and when a full round makes no
 * progress the wait-for graph is built from the granted holder sets, searched for a cycle with a
 * depth-first search, and the highest-id transaction in the cycle is aborted as the victim.
 * Writes are buffered in the transaction and applied at COMMIT, which makes undo trivial and does not
 * change the locking behaviour: locks are only ever released in COMMIT and ABORT, so the protocol is
 * strict (well-formed, two-phase, no lock acquired after a release).
 *
 * Simplified (deliberately): lock granularity is a whole item, there are no timeouts, no lock
 * escalation, no intent/shared-hierarchy modes, only one waiter is woken per release (it simply
 * retries next round) and an aborted transaction is restarted as a fresh attempt instead of being
 * compensated.
 *
 * Verified: shared locks co-exist and exclusive locks exclude (the lock table's mutual-exclusion
 * invariant is re-checked after every round); the classic two-transaction and three-transaction
 * deadlocks are really detected, the cycle members are reported and the victim is the highest id in
 * the cycle; every transaction commits after a bounded number of restarts and the final state equals
 * the serial replay of the committed transactions in commit order (the serializability guarantee of
 * strict 2PL, checked by an independent oracle); the same schedule run with lock enforcement switched
 * off loses updates and is not serializable - the negative control that shows the locks are doing the
 * work.
 */
import java.util.*;

class LockTable {
    final Map<String, Set<Integer>> sharers = new TreeMap<>();
    final Map<String, Integer> exclusive = new TreeMap<>();
    int denied;
    int maxSharers;

    boolean granted(String item, int tid, char mode) {
        Integer holder = exclusive.get(item);
        if (holder != null && holder != tid) return false;
        if (mode == 'S') return true;
        for (int sharer : sharers.getOrDefault(item, Collections.emptySet())) if (sharer != tid) return false;
        return true;
    }

    void acquire(String item, int tid, char mode) {
        if (mode == 'X') {
            exclusive.put(item, tid);
        } else {
            Set<Integer> holders = sharers.computeIfAbsent(item, key -> new TreeSet<>());
            holders.add(tid);
            maxSharers = Math.max(maxSharers, holders.size());
        }
    }

    void release(String item, int tid) {
        if (exclusive.getOrDefault(item, -1) == tid) exclusive.remove(item);
        Set<Integer> holders = sharers.get(item);
        if (holders != null) {
            holders.remove(tid);
            if (holders.isEmpty()) sharers.remove(item);
        }
    }

    /** Transactions that currently block a request for {@code mode} on {@code item}. */
    List<Integer> blockers(String item, int tid, char mode) {
        List<Integer> out = new ArrayList<>();
        Integer holder = exclusive.get(item);
        if (holder != null && holder != tid) {
            out.add(holder);
            return out;
        }
        if (mode == 'X') {
            for (int sharer : sharers.getOrDefault(item, Collections.emptySet())) if (sharer != tid) out.add(sharer);
        }
        return out;
    }

    /** Mutual exclusion: an item is either exclusively held or shared, never both. */
    boolean invariantOk() {
        for (String item : exclusive.keySet()) {
            Set<Integer> holders = sharers.get(item);
            if (holders != null && !holders.isEmpty()) return false;
        }
        return true;
    }
}

class TwoPlScheduler {

    static final class Op {
        final String kind, item, slot;
        final int n;

        private Op(String kind, String item, String slot, int n) {
            this.kind = kind; this.item = item; this.slot = slot; this.n = n;
        }
        static Op lockS(String item) { return new Op("LOCK_S", item, null, 0); }
        static Op lockX(String item) { return new Op("LOCK_X", item, null, 0); }
        static Op read(String item, String slot) { return new Op("READ", item, slot, 0); }
        static Op add(String slot, int n) { return new Op("ADD", null, slot, n); }
        static Op write(String slot, String item) { return new Op("WRITE", item, slot, 0); }
        static Op commit() { return new Op("COMMIT", null, null, 0); }
        static Op abort() { return new Op("ABORT", null, null, 0); }
    }

    static final class Tx {
        final int id;
        final List<Op> ops;
        int pc;
        int attempts = 1;
        int commitTick = -1;
        String status = "active";
        String reason = "";
        String waitingItem;
        char waitingMode;
        final Map<String, Integer> locals = new LinkedHashMap<>();
        final Map<String, Integer> writes = new LinkedHashMap<>();
        final LinkedHashSet<String> held = new LinkedHashSet<>();

        Tx(int id, List<Op> ops) { this.id = id; this.ops = ops; }
    }

    final LockTable locks = new LockTable();
    final Map<String, Integer> state = new TreeMap<>();
    final List<Tx> txs = new ArrayList<>();
    final Map<Integer, Tx> byId = new LinkedHashMap<>();
    final List<String> log = new ArrayList<>();
    final List<List<String>> deadlockCycles = new ArrayList<>();
    final boolean locking;
    int tick, deadlocks, aborts, deniedRequests, executedOps, rounds;

    TwoPlScheduler(Map<String, Integer> initial, boolean locking) {
        state.putAll(initial);
        this.locking = locking;
    }

    Tx add(int id, List<Op> ops) {
        Tx tx = new Tx(id, ops);
        txs.add(tx);
        byId.put(id, tx);
        return tx;
    }

    void releaseAll(Tx tx) {
        for (String item : tx.held) locks.release(item, tx.id);
        tx.held.clear();
    }

    boolean anyActive() {
        for (Tx tx : txs) if (tx.status.equals("active")) return true;
        return false;
    }

    List<Integer> commitOrder() {
        List<Tx> ordered = new ArrayList<>(txs);
        ordered.sort(Comparator.comparingInt(tx -> tx.commitTick));
        List<Integer> out = new ArrayList<>();
        for (Tx tx : ordered) if (tx.status.equals("committed")) out.add(tx.id);
        return out;
    }

    void run(int maxRounds) {
        for (int round = 0; round < maxRounds; round++) {
            rounds++;
            boolean moved = false;
            for (Tx tx : txs) {
                if (!tx.status.equals("active")) continue;
                if (stepOne(tx)) moved = true;
            }
            if (!anyActive()) return;
            if (!moved) {
                if (!locking || !breakDeadlock()) {
                    log.add("STALL with no cycle at round " + rounds);
                    return;
                }
            }
        }
    }

    /** Executes exactly one operation; returns false when the transaction had to wait for a lock. */
    boolean stepOne(Tx tx) {
        if (tx.pc >= tx.ops.size()) {
            tx.status = "done";
            log.add("T" + tx.id + " ran out of operations");
            return false;
        }
        Op op = tx.ops.get(tx.pc);
        if (op.kind.equals("LOCK_S") || op.kind.equals("LOCK_X")) {
            char mode = op.kind.equals("LOCK_S") ? 'S' : 'X';
            if (locking && !locks.granted(op.item, tx.id, mode)) {
                locks.denied++;
                deniedRequests++;
                tx.waitingItem = op.item;
                tx.waitingMode = mode;
                log.add("T" + tx.id + " waits for " + mode + " " + op.item);
                return false;
            }
            if (locking && tx.held.add(op.item)) locks.acquire(op.item, tx.id, mode);
            tx.waitingItem = null;
            tx.pc++;
            executedOps++;
            return true;
        }
        if (op.kind.equals("COMMIT")) {
            tick++;
            state.putAll(tx.writes);
            tx.status = "committed";
            tx.commitTick = tick;
            if (locking) releaseAll(tx);
            tx.pc++;
            executedOps++;
            log.add("COMMIT T" + tx.id + "@" + tick);
            return true;
        }
        if (op.kind.equals("ABORT")) {
            tx.writes.clear();
            tx.status = "aborted";
            tx.reason = "explicit abort";
            aborts++;
            if (locking) releaseAll(tx);
            tx.pc++;
            executedOps++;
            log.add("ABORT T" + tx.id + " (explicit)");
            return true;
        }
        dataOp(op, tx.locals, tx.writes, state);
        tx.pc++;
        executedOps++;
        return true;
    }

    /** Data operations are shared with the serial oracle, so both use identical arithmetic. */
    static void dataOp(Op op, Map<String, Integer> locals, Map<String, Integer> writes,
                       Map<String, Integer> state) {
        switch (op.kind) {
            case "READ":
                locals.put(op.slot, state.get(op.item));
                break;
            case "ADD":
                locals.put(op.slot, locals.get(op.slot) + op.n);
                break;
            case "WRITE":
                writes.put(op.item, locals.get(op.slot));
                break;
            default:
                throw new IllegalStateException("not a data operation: " + op.kind);
        }
    }

    Map<Integer, Set<Integer>> waitForGraph() {
        Map<Integer, Set<Integer>> graph = new TreeMap<>();
        for (Tx tx : txs) {
            if (!tx.status.equals("active") || tx.waitingItem == null) continue;
            graph.put(tx.id, new TreeSet<>(locks.blockers(tx.waitingItem, tx.id, tx.waitingMode)));
        }
        return graph;
    }

    static List<Integer> findCycle(Map<Integer, Set<Integer>> graph) {
        Map<Integer, Integer> colour = new TreeMap<>();
        for (int node : graph.keySet()) {
            if (colour.getOrDefault(node, 0) != 0) continue;
            List<Integer> cycle = new ArrayList<>();
            if (dfs(node, graph, colour, new ArrayList<>(), cycle)) return cycle;
        }
        return List.of();
    }

    private static boolean dfs(int node, Map<Integer, Set<Integer>> graph, Map<Integer, Integer> colour,
                               List<Integer> stack, List<Integer> cycle) {
        colour.put(node, 1);
        stack.add(node);
        for (int next : graph.getOrDefault(node, Collections.emptySet())) {
            if (colour.getOrDefault(next, 0) == 1) {
                cycle.addAll(stack.subList(stack.indexOf(next), stack.size()));
                cycle.add(next);
                return true;
            }
            if (colour.getOrDefault(next, 0) == 0 && dfs(next, graph, colour, stack, cycle)) return true;
        }
        stack.remove(stack.size() - 1);
        colour.put(node, 2);
        return false;
    }

    /** Aborts the highest transaction id in the detected cycle. */
    boolean breakDeadlock() {
        Map<Integer, Set<Integer>> graph = waitForGraph();
        List<Integer> cycle = findCycle(graph);
        if (cycle.isEmpty()) return false;
        deadlocks++;
        List<String> names = new ArrayList<>();
        for (int id : cycle) names.add("T" + id);
        deadlockCycles.add(names);
        TreeSet<Integer> members = new TreeSet<>(cycle);
        int victim = members.last();
        Tx tx = byId.get(victim);
        releaseAll(tx);
        tx.status = "aborted";
        tx.reason = "deadlock victim of cycle " + names;
        tx.writes.clear();
        tx.waitingItem = null;
        aborts++;
        log.add("DEADLOCK " + names + " victim T" + victim);
        return true;
    }

    boolean restartAborted(int maxAttempts) {
        boolean any = false;
        for (Tx tx : txs) {
            if (tx.status.equals("aborted") && tx.attempts < maxAttempts) {
                tx.attempts++;
                tx.pc = 0;
                tx.locals.clear();
                tx.writes.clear();
                tx.held.clear();
                tx.waitingItem = null;
                tx.status = "active";
                tx.reason = "";
                any = true;
            }
        }
        return any;
    }

    void runToCompletion(int maxRounds, int maxAttempts) {
        for (int attempt = 0; attempt < maxAttempts; attempt++) {
            run(maxRounds);
            if (!restartAborted(maxAttempts)) return;
        }
    }

    /** Serial oracle: replay the oplists in the given order, locks ignored, writes applied at COMMIT. */
    Map<String, Integer> serialReplay(List<Integer> order, Map<String, Integer> initial) {
        Map<String, Integer> serial = new TreeMap<>(initial);
        for (int tid : order) {
            Map<String, Integer> locals = new LinkedHashMap<>();
            Map<String, Integer> writes = new LinkedHashMap<>();
            for (Op op : byId.get(tid).ops) {
                if (op.kind.equals("COMMIT")) serial.putAll(writes);
                else if (op.kind.equals("ABORT")) break;
                else if (!op.kind.startsWith("LOCK")) dataOp(op, locals, writes, serial);
            }
        }
        return serial;
    }

    boolean serializable(Map<String, Integer> initial) {
        for (List<Integer> order : permutations(commitOrder())) {
            if (serialReplay(order, initial).equals(state)) return true;
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

    /** T1 updates a then b, T2 updates b then a: classic lock-order inversion. */
    static List<List<TwoPlScheduler.Op>> crossedProgram() {
        List<TwoPlScheduler.Op> first = List.of(
                TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.read("a", "v"),
                TwoPlScheduler.Op.add("v", 1), TwoPlScheduler.Op.write("v", "a"),
                TwoPlScheduler.Op.lockX("b"), TwoPlScheduler.Op.read("b", "v"),
                TwoPlScheduler.Op.add("v", 10), TwoPlScheduler.Op.write("v", "b"),
                TwoPlScheduler.Op.commit());
        List<TwoPlScheduler.Op> second = List.of(
                TwoPlScheduler.Op.lockX("b"), TwoPlScheduler.Op.read("b", "v"),
                TwoPlScheduler.Op.add("v", 2), TwoPlScheduler.Op.write("v", "b"),
                TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.read("a", "v"),
                TwoPlScheduler.Op.add("v", 20), TwoPlScheduler.Op.write("v", "a"),
                TwoPlScheduler.Op.commit());
        return List.of(first, second);
    }

    /** Three transactions locking a->b, b->c, c->a: a cycle of length three. */
    static List<List<TwoPlScheduler.Op>> ringProgram() {
        String[][] order = {{"a", "b"}, {"b", "c"}, {"c", "a"}};
        List<List<TwoPlScheduler.Op>> program = new ArrayList<>();
        for (int i = 0; i < order.length; i++) {
            List<TwoPlScheduler.Op> ops = new ArrayList<>();
            for (String item : order[i]) {
                ops.add(TwoPlScheduler.Op.lockX(item));
                ops.add(TwoPlScheduler.Op.read(item, "v"));
                ops.add(TwoPlScheduler.Op.add("v", 1 + i));
                ops.add(TwoPlScheduler.Op.write("v", item));
            }
            ops.add(TwoPlScheduler.Op.commit());
            program.add(ops);
        }
        return program;
    }

    static TwoPlScheduler build(Map<String, Integer> initial, boolean locking,
                                List<List<TwoPlScheduler.Op>> program) {
        TwoPlScheduler scheduler = new TwoPlScheduler(initial, locking);
        for (int i = 0; i < program.size(); i++) scheduler.add(i + 1, program.get(i));
        return scheduler;
    }

    public static void main(String[] args) {
        Map<String, Integer> twoItems = new LinkedHashMap<>();
        twoItems.put("a", 100);
        twoItems.put("b", 100);

        TwoPlScheduler locked = build(twoItems, true, crossedProgram());
        locked.runToCompletion(200, 3);
        check(locked.deadlocks == 1, "one deadlock detected, got " + locked.deadlocks);
        check(locked.deadlockCycles.get(0).equals(List.of("T1", "T2", "T1")),
                "wait-for cycle 1 -> 2 -> 1, got " + locked.deadlockCycles.get(0));
        check(locked.byId.get(2).attempts == 2, "T2 was restarted after being the victim");
        check(locked.byId.get(2).reason.isEmpty(), "the successful retry has no abort reason");
        check(!locked.anyActive(), "every transaction terminated");
        check(locked.state.toString().equals("{a=121, b=112}"), "locked state " + locked.state);
        check(locked.commitOrder().equals(List.of(1, 2)), "commit order " + locked.commitOrder());
        check(locked.serialReplay(List.of(1, 2), twoItems).equals(locked.state),
                "final state equals the serial replay in commit order");
        check(locked.serializable(twoItems), "strict 2PL schedule is serializable");
        check(locked.deniedRequests > 0, "lock requests were really denied (blocking happened)");
        check(locked.locks.invariantOk(), "lock table invariant after the run");
        check(locked.log.stream().noneMatch(line -> line.startsWith("STALL")), "no stall without a cycle");
        check(locked.log.stream().anyMatch(line -> line.startsWith("DEADLOCK")), "deadlock was logged");

        TwoPlScheduler unprotected = build(twoItems, false, crossedProgram());
        unprotected.runToCompletion(200, 3);
        check(!unprotected.anyActive(), "unlocked run terminates too");
        check(unprotected.deadlocks == 0, "no deadlock detection without locks");
        check(unprotected.deniedRequests == 0, "no blocking without locks");
        check(unprotected.state.toString().equals("{a=120, b=102}"), "unlocked state " + unprotected.state);
        check(!unprotected.serializable(twoItems), "unlocked schedule loses updates and is not serializable");
        check(unprotected.serialReplay(List.of(1, 2), twoItems).toString().equals("{a=121, b=112}"),
                "the serial oracle still produces the consistent state");

        Map<String, Integer> items = new LinkedHashMap<>();
        items.put("a", 0);
        items.put("b", 0);
        items.put("c", 0);
        TwoPlScheduler ring = build(items, true, ringProgram());
        ring.runToCompletion(400, 5);
        check(ring.deadlocks == 1, "one three-way deadlock, got " + ring.deadlocks);
        check(new TreeSet<>(ring.deadlockCycles.get(0)).equals(new TreeSet<>(List.of("T1", "T2", "T3"))),
                "the cycle contains all three transactions: " + ring.deadlockCycles.get(0));
        check(ring.byId.get(3).attempts == 2 && ring.byId.get(3).commitTick > 0,
                "T3 was the victim and committed after the restart");
        check(!ring.anyActive(), "ring transactions all terminate");
        check(ring.serializable(items), "ring schedule is serializable after the restart");
        check(ring.serialReplay(ring.commitOrder(), items).equals(ring.state),
                "ring state equals its serial replay");

        Map<String, Integer> shared = new LinkedHashMap<>();
        shared.put("a", 42);
        List<List<TwoPlScheduler.Op>> readers = List.of(
                List.of(TwoPlScheduler.Op.lockS("a"), TwoPlScheduler.Op.read("a", "v"),
                        TwoPlScheduler.Op.commit()),
                List.of(TwoPlScheduler.Op.lockS("a"), TwoPlScheduler.Op.read("a", "v"),
                        TwoPlScheduler.Op.commit()));
        TwoPlScheduler concurrentReaders = build(shared, true, readers);
        concurrentReaders.runToCompletion(50, 1);
        check(concurrentReaders.state.get("a") == 42, "readers do not modify the item");
        check(concurrentReaders.deadlocks == 0 && concurrentReaders.deniedRequests == 0,
                "shared locks are compatible");
        check(concurrentReaders.locks.maxSharers == 2, "two readers held the share lock at once, max "
                + concurrentReaders.locks.maxSharers);
        check(concurrentReaders.locks.invariantOk(), "shared holders leave the invariant intact");

        List<List<TwoPlScheduler.Op>> exclusivePair = List.of(
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.commit()),
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.commit()));
        TwoPlScheduler writers = build(shared, true, exclusivePair);
        writers.runToCompletion(50, 1);
        check(writers.deniedRequests > 0, "exclusive locks exclude");
        check(writers.locks.maxSharers == 0, "no shared holder appears under exclusive locks");

        List<List<TwoPlScheduler.Op>> sameOrder = List.of(
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.read("a", "v"),
                        TwoPlScheduler.Op.add("v", 5), TwoPlScheduler.Op.write("v", "a"),
                        TwoPlScheduler.Op.commit()),
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.read("a", "v"),
                        TwoPlScheduler.Op.add("v", 7), TwoPlScheduler.Op.write("v", "a"),
                        TwoPlScheduler.Op.commit()));
        Map<String, Integer> single = new LinkedHashMap<>();
        single.put("a", 100);
        TwoPlScheduler ordered = build(single, true, sameOrder);
        ordered.runToCompletion(50, 1);
        check(ordered.deadlocks == 0, "a consistent lock order never deadlocks");
        check(ordered.state.get("a") == 112, "ordered increments: " + ordered.state.get("a"));
        check(ordered.deniedRequests > 0, "the second writer was blocked but not deadlocked");
        check(ordered.serializable(single), "ordered concurrent increments are serializable");

        List<List<TwoPlScheduler.Op>> reentrant = List.of(
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.lockX("a"),
                        TwoPlScheduler.Op.read("a", "v"), TwoPlScheduler.Op.add("v", 3),
                        TwoPlScheduler.Op.write("v", "a"), TwoPlScheduler.Op.commit()));
        TwoPlScheduler twice = build(single, true, reentrant);
        twice.runToCompletion(20, 1);
        check(twice.state.get("a") == 103, "re-acquiring a held lock is a no-op: " + twice.state.get("a"));
        check(twice.deniedRequests == 0, "a transaction never blocks itself");

        List<List<TwoPlScheduler.Op>> withAbort = List.of(
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.read("a", "v"),
                        TwoPlScheduler.Op.add("v", 50), TwoPlScheduler.Op.write("v", "a"),
                        TwoPlScheduler.Op.abort()),
                List.of(TwoPlScheduler.Op.lockX("a"), TwoPlScheduler.Op.read("a", "v"),
                        TwoPlScheduler.Op.add("v", 1), TwoPlScheduler.Op.write("v", "a"),
                        TwoPlScheduler.Op.commit()));
        TwoPlScheduler aborted = build(single, true, withAbort);
        aborted.run(50);
        check(aborted.state.get("a") == 101, "the aborted writer left no trace: " + aborted.state.get("a"));
        check(aborted.byId.get(1).status.equals("aborted") && aborted.byId.get(2).status.equals("committed"),
                "abort then commit: " + aborted.byId.get(1).status + "/" + aborted.byId.get(2).status);
        check(aborted.byId.get(1).held.isEmpty(), "the abort released every lock");

        TwoPlScheduler none = build(single, true, List.of());
        none.run(10);
        check(none.state.get("a") == 100 && !none.anyActive(), "empty workload is a no-op");
        check(TwoPlScheduler.findCycle(Map.of()).isEmpty(), "no cycle in an empty wait-for graph");

        System.out.println("strict 2PL: 2-way deadlock " + locked.deadlockCycles.get(0) + " victim T2, "
                + "3-way deadlock " + ring.deadlockCycles.get(0) + " victim T3");
        System.out.println("locked state " + locked.state + " (serializable) vs unlocked state "
                + unprotected.state + " (not serializable)");
        System.out.println("denied lock requests=" + (locked.deniedRequests + ring.deniedRequests
                + ordered.deniedRequests + writers.deniedRequests) + ", executed ops="
                + (locked.executedOps + ring.executedOps + ordered.executedOps));
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
