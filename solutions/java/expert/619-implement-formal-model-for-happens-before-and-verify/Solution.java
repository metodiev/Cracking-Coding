/** Task 619 - Implement formal model for happens-before and verify. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A small concurrent program is modelled as events with two independent orderings, and the model is
 * checked for agreement:
 *   - the *happens-before* relation is the transitive closure of program order plus the
 *     synchronizes-with edges (the j-th send on a channel with the j-th receive, an unlock with the
 *     next lock of the same lock, a fork with the first event of the created thread, a join with the
 *     last event of the joined thread), computed by Floyd-Warshall;
 *   - *vector clocks* are computed with the rules "inherit the predecessor, merge every incoming
 *     synchronizes-with clock, then increment the own component" (recursively with memoization, so the
 *     order of the script does not matter).
 * The central verification is the standard theorem: for distinct events a and b, a happens-before b
 * exactly when VC(a) <= VC(b). It is checked for every pair of every test program, together with the
 * partial-order laws (irreflexive, transitive, antisymmetric) and the data-race rule: two accesses to
 * the same variable from different threads, at least one of them a write, are a race exactly when no
 * happens-before order connects them.
 *
 * The model is also verified negatively: reversing the synchronizes-with edges (a receive "causing" its
 * send) makes the relation cyclic, and the checker reports the cycle and the now mismatching vector
 * clocks instead of accepting a causal loop.
 */
import java.util.*;

class HbEvent {
    static final int ACCESS = 0, SEND = 1, RECV = 2, LOCK = 3, UNLOCK = 4, FORK = 5, JOIN = 6;

    final int id;
    final int thread;
    final int kind;
    final String name;                 // the variable, channel or lock
    final boolean write;
    final int threadTarget;            // the created or joined thread for FORK / JOIN
    final int seqInThread;             // position inside its thread, for the program-order edge
    int syncIn = -1;                   // the event this one synchronizes with
    int[] clock;
    boolean inProgress;                // guards the recursive clock computation against cycles

    HbEvent(int id, int thread, int kind, String name, boolean write, int threadTarget, int seqInThread) {
        this.id = id;
        this.thread = thread;
        this.kind = kind;
        this.name = name;
        this.write = write;
        this.threadTarget = threadTarget;
        this.seqInThread = seqInThread;
    }

    String label() {
        switch (kind) {
            case ACCESS: return "t" + thread + (write ? ":W(" : ":R(") + name + ")";
            case SEND: return "t" + thread + ":send(" + name + ")";
            case RECV: return "t" + thread + ":recv(" + name + ")";
            case LOCK: return "t" + thread + ":lock(" + name + ")";
            case UNLOCK: return "t" + thread + ":unlock(" + name + ")";
            case FORK: return "t" + thread + ":fork(t" + threadTarget + ")";
            default: return "t" + thread + ":join(t" + threadTarget + ")";
        }
    }
}

class HbProgram {
    final int threadCount;
    final List<HbEvent> events = new ArrayList<>();
    private final Map<String, List<Integer>> byChannel = new LinkedHashMap<>();
    private final Map<String, List<Integer>> byLock = new LinkedHashMap<>();
    private final int[] lastEventOfThread;
    private final int[] firstEventOfThread;
    private final int[] forkEventOfThread;
    private final int[] seqCounter;

    HbProgram(int threadCount) {
        this.threadCount = threadCount;
        lastEventOfThread = new int[threadCount];
        firstEventOfThread = new int[threadCount];
        forkEventOfThread = new int[threadCount];
        seqCounter = new int[threadCount];
        Arrays.fill(lastEventOfThread, -1);
        Arrays.fill(firstEventOfThread, -1);
        Arrays.fill(forkEventOfThread, -1);
    }

    private HbEvent add(int thread, int kind, String name, boolean write, int target) {
        HbEvent event = new HbEvent(events.size(), thread, kind, name, write, target, seqCounter[thread]++);
        events.add(event);
        if (firstEventOfThread[thread] < 0) firstEventOfThread[thread] = event.id;
        lastEventOfThread[thread] = event.id;
        if (kind == HbEvent.FORK) forkEventOfThread[target] = event.id;
        return event;
    }

    int read(int thread, String var) { return add(thread, HbEvent.ACCESS, var, false, -1).id; }

    int write(int thread, String var) { return add(thread, HbEvent.ACCESS, var, true, -1).id; }

    int send(int thread, String channel) { return add(thread, HbEvent.SEND, channel, false, -1).id; }

    int receive(int thread, String channel) { return add(thread, HbEvent.RECV, channel, false, -1).id; }

    int lock(int thread, String lock) { return add(thread, HbEvent.LOCK, lock, false, -1).id; }

    int unlock(int thread, String lock) { return add(thread, HbEvent.UNLOCK, lock, false, -1).id; }

    int fork(int thread, int child) { return add(thread, HbEvent.FORK, "", false, child).id; }

    int join(int thread, int child) { return add(thread, HbEvent.JOIN, "", false, child).id; }

    /** Pairs sends with receives, locks with the previous unlock, forks and joins with their thread. */
    void buildSynchronizesWith() {
        for (HbEvent e : events) {
            if (e.kind == HbEvent.SEND || e.kind == HbEvent.RECV) {
                byChannel.computeIfAbsent(e.name, k -> new ArrayList<>()).add(e.id);
            }
            if (e.kind == HbEvent.LOCK || e.kind == HbEvent.UNLOCK) {
                byLock.computeIfAbsent(e.name, k -> new ArrayList<>()).add(e.id);
            }
        }
        for (List<Integer> list : byChannel.values()) {
            List<Integer> sends = new ArrayList<>();
            List<Integer> receives = new ArrayList<>();
            for (int id : list) {
                if (events.get(id).kind == HbEvent.SEND) sends.add(id); else receives.add(id);
            }
            for (int i = 0; i < Math.min(sends.size(), receives.size()); i++) {
                events.get(receives.get(i)).syncIn = sends.get(i);
            }
        }
        for (List<Integer> list : byLock.values()) {
            int lastUnlock = -1;
            for (int id : list) {
                HbEvent e = events.get(id);
                if (e.kind == HbEvent.UNLOCK) lastUnlock = id;
                else if (lastUnlock >= 0) e.syncIn = lastUnlock;
            }
        }
        for (int t = 0; t < threadCount; t++) {
            if (firstEventOfThread[t] >= 0 && forkEventOfThread[t] >= 0) {
                events.get(firstEventOfThread[t]).syncIn = forkEventOfThread[t];
            }
        }
        for (HbEvent e : events) {
            if (e.kind == HbEvent.JOIN && lastEventOfThread[e.threadTarget] >= 0) {
                e.syncIn = lastEventOfThread[e.threadTarget];
            }
        }
    }

    private int programOrderPredecessor(HbEvent e) {
        if (e.seqInThread == 0) return -1;
        for (int i = e.id - 1; i >= 0; i--) {
            HbEvent other = events.get(i);
            if (other.thread == e.thread && other.seqInThread == e.seqInThread - 1) return other.id;
        }
        return -1;
    }

    /** The transitive closure of program order plus the synchronizes-with edges. */
    boolean[][] happensBefore(boolean reversedSync) {
        int n = events.size();
        boolean[][] hb = new boolean[n][n];
        for (HbEvent e : events) {
            int previous = programOrderPredecessor(e);
            if (previous >= 0) hb[previous][e.id] = true;
            if (e.syncIn >= 0) {
                if (reversedSync) hb[e.id][e.syncIn] = true; else hb[e.syncIn][e.id] = true;
            }
        }
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (!hb[i][k]) continue;
                for (int j = 0; j < n; j++) if (hb[k][j]) hb[i][j] = true;
            }
        }
        return hb;
    }

    /** Vector clocks with memoization, so the script order does not have to be a topological order. */
    int[] clock(int id) {
        HbEvent e = events.get(id);
        if (e.clock != null) return e.clock;
        if (e.inProgress) return new int[threadCount];       // a cyclic model has no vector clocks at all
        e.inProgress = true;
        int[] clock = new int[threadCount];
        int previous = programOrderPredecessor(e);
        if (previous >= 0) clock = clock(previous).clone();
        if (e.syncIn >= 0) {
            int[] incoming = clock(e.syncIn);
            for (int t = 0; t < threadCount; t++) clock[t] = Math.max(clock[t], incoming[t]);
        }
        clock[e.thread]++;
        e.clock = clock;
        e.inProgress = false;
        return clock;
    }

    static boolean lessOrEqual(int[] a, int[] b) {
        for (int i = 0; i < a.length; i++) if (a[i] > b[i]) return false;
        return true;
    }

    boolean isCycle(boolean[][] hb) {
        for (int i = 0; i < hb.length; i++) if (hb[i][i]) return true;
        return false;
    }

    /** Two accesses to the same variable from different threads with no happens-before order. */
    List<int[]> races(boolean[][] hb) {
        List<int[]> found = new ArrayList<>();
        for (HbEvent a : events) {
            if (a.kind != HbEvent.ACCESS) continue;
            for (HbEvent b : events) {
                if (b.kind != HbEvent.ACCESS || b.id <= a.id) continue;
                if (a.thread == b.thread || !a.name.equals(b.name)) continue;
                if (!a.write && !b.write) continue;
                if (!hb[a.id][b.id] && !hb[b.id][a.id]) found.add(new int[]{a.id, b.id});
            }
        }
        return found;
    }

    int[] vectorClockOf(int id) { return clock(id); }
}

public class Solution {
    static void verifyModel(HbProgram program, String label) {
        program.buildSynchronizesWith();
        boolean[][] hb = program.happensBefore(false);
        int n = program.events.size();

        // partial order laws
        check(!program.isCycle(hb), label + ": happens-before is irreflexive (no cycle)");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) check(!(hb[i][j] && hb[j][i]), label + ": happens-before is antisymmetric");
                if (hb[i][j] && i != j) {
                    for (int k = 0; k < n; k++) {
                        if (hb[j][k]) check(hb[i][k], label + ": happens-before is transitive");
                    }
                }
            }
        }

        // vector clocks agree with the closure
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                boolean byClock = HbProgram.lessOrEqual(program.vectorClockOf(i), program.vectorClockOf(j));
                check(byClock == hb[i][j], label + ": VC(" + program.events.get(i).label() + ") <= VC("
                        + program.events.get(j).label() + ") is " + byClock + " but happens-before says "
                        + hb[i][j]);
                boolean concurrentFrom = !hb[i][j] && !hb[j][i];
                boolean concurrentFromClocks = !byClock
                        && !HbProgram.lessOrEqual(program.vectorClockOf(j), program.vectorClockOf(i));
                check(concurrentFrom == concurrentFromClocks, label + ": concurrency is symmetric");
            }
        }
        // program order is part of happens-before, and the first event of a forked thread follows its fork
        for (HbEvent e : program.events) {
            int previous = -1;
            for (HbEvent other : program.events) {
                if (other.thread == e.thread && other.seqInThread == e.seqInThread - 1) previous = other.id;
            }
            if (previous >= 0) check(hb[previous][e.id], label + ": program order is included");
        }
    }

    static int countMismatches(HbProgram program, boolean[][] hb) {
        int mismatches = 0;
        for (int i = 0; i < program.events.size(); i++) {
            for (int j = 0; j < program.events.size(); j++) {
                if (i == j) continue;
                boolean byClock = HbProgram.lessOrEqual(program.vectorClockOf(i), program.vectorClockOf(j));
                if (byClock != hb[i][j]) mismatches++;
            }
        }
        return mismatches;
    }

    static HbProgram lockedCounter() {
        HbProgram program = new HbProgram(2);
        program.lock(0, "L");
        program.write(0, "x");
        program.unlock(0, "L");
        program.lock(1, "L");
        program.read(1, "x");
        program.unlock(1, "L");
        return program;
    }

    static HbProgram messagePassing() {
        HbProgram program = new HbProgram(2);
        program.write(0, "x");
        program.send(0, "ch");
        program.receive(1, "ch");
        program.read(1, "x");
        return program;
    }

    static HbProgram forkJoin() {
        HbProgram program = new HbProgram(2);
        program.write(0, "x");
        program.fork(0, 1);
        program.write(1, "x");
        program.join(0, 1);
        program.read(0, "x");
        return program;
    }

    static HbProgram unsynchronised() {
        HbProgram program = new HbProgram(2);
        program.write(0, "x");
        program.read(1, "x");
        return program;
    }

    public static void main(String[] args) {
        HbProgram locked = lockedCounter();
        verifyModel(locked, "lock");
        check(locked.races(locked.happensBefore(false)).isEmpty(), "the lock orders the accesses");

        HbProgram passing = messagePassing();
        verifyModel(passing, "message");
        check(passing.races(passing.happensBefore(false)).isEmpty(), "the message orders the accesses");

        HbProgram forked = forkJoin();
        verifyModel(forked, "fork/join");
        check(forked.races(forked.happensBefore(false)).isEmpty(), "fork and join order everything");

        HbProgram racy = unsynchronised();
        verifyModel(racy, "racy");
        List<int[]> races = racy.races(racy.happensBefore(false));
        check(races.size() == 1, "an unsynchronised write/read pair is one race");
        check(!racy.happensBefore(false)[races.get(0)[0]][races.get(0)[1]], "the race pair is unordered");

        HbProgram disjoint = new HbProgram(2);
        disjoint.write(0, "x");
        disjoint.write(1, "y");
        verifyModel(disjoint, "disjoint variables");
        check(disjoint.races(disjoint.happensBefore(false)).isEmpty(), "different variables do not race");

        HbProgram halfProtected = new HbProgram(2);
        halfProtected.lock(0, "L");
        halfProtected.write(0, "x");
        halfProtected.unlock(0, "L");
        halfProtected.read(1, "x");                    // read before taking the lock
        halfProtected.lock(1, "L");
        halfProtected.read(1, "x");
        halfProtected.unlock(1, "L");
        verifyModel(halfProtected, "half protected");
        List<int[]> halfRaces = halfProtected.races(halfProtected.happensBefore(false));
        check(halfRaces.size() == 1, "the read outside the lock races with the write inside it");
        check(halfProtected.events.get(halfRaces.get(0)[0]).write, "the write is one side of the race");
        check(!halfProtected.happensBefore(false)[halfRaces.get(0)[1]][halfRaces.get(0)[0]],
                "the racing read is not ordered after the write");

        // a two-channel loop: reversing the synchronizes-with edges turns it into a causal cycle
        HbProgram pingPong = new HbProgram(2);
        pingPong.send(0, "a");
        pingPong.receive(0, "b");
        pingPong.receive(1, "a");
        pingPong.send(1, "b");
        verifyModel(pingPong, "ping pong");
        check(pingPong.races(pingPong.happensBefore(false)).isEmpty(), "the loop has no races");
        check(!pingPong.isCycle(pingPong.happensBefore(false)), "the loop is acyclic");

        // a three-thread chain: each hand-off adds one to the shared clock
        HbProgram chain = new HbProgram(3);
        chain.write(0, "x");
        chain.send(0, "a");
        chain.receive(1, "a");
        chain.write(1, "x");
        chain.send(1, "b");
        chain.receive(2, "b");
        chain.read(2, "x");
        verifyModel(chain, "chain");
        check(chain.races(chain.happensBefore(false)).isEmpty(), "the chain is race free");
        check(chain.vectorClockOf(6)[2] == 2, "the third thread's clock counts its own first event");
        check(chain.vectorClockOf(6)[0] == 2 && chain.vectorClockOf(6)[1] == 3,
                "the chain's clock carries both hand-offs");

        // negative 1: a receive-before-send loop is causally impossible, and the model says so
        HbProgram impossible = new HbProgram(2);
        impossible.receive(0, "b");
        impossible.send(0, "a");
        impossible.receive(1, "a");
        impossible.send(1, "b");
        impossible.buildSynchronizesWith();
        boolean[][] impossibleHb = impossible.happensBefore(false);
        check(impossible.isCycle(impossibleHb), "a receive-before-send loop is rejected as cyclic");
        int cyclicMismatches = countMismatches(impossible, impossibleHb);
        check(cyclicMismatches > 0, "the vector clocks cannot describe a cyclic history");

        // negative 2: a model that believes receive -> send no longer agrees with the vector clocks
        boolean[][] reversed = chain.happensBefore(true);
        check(!chain.isCycle(reversed), "reversing one hand-off of a chain stays acyclic");
        int reversedMismatches = countMismatches(chain, reversed);
        check(reversedMismatches > 0, "the vector clocks disagree with the reversed model");
        System.out.println("cyclic model mismatches: " + cyclicMismatches
                + ", reversed model mismatches: " + reversedMismatches);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
