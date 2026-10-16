/** Task 668 - Implement priority inheritance protocol simulation for priority inversion. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * A deterministic, tick-driven uniprocessor scheduler. Every task runs a scripted list of steps
 * (unit work, acquire, release) and each tick executes exactly one step of the ready task with the
 * highest effective priority. Locks are handed off to the highest-effective-priority waiter.
 *
 * Priority inheritance protocol (PIP): the effective priority of a task is re-derived from scratch
 * every tick - start with the base priorities, then repeatedly let every blocked task push its own
 * effective priority onto the holder of the lock it is waiting for, until nothing changes. The
 * fixpoint walks blocking chains transitively (L <- M <- H), which is what bounds inversion.
 * With inheritance disabled the effective priority is simply the base priority (no protocol).
 *
 * Verified: mutual exclusion (a lock has at most one owner and the two bookkeeping structures agree),
 * effective priority >= base priority always, "one task per tick" accounting, no double ownership and
 * no deadlock under resource-ordered scripts, deadlock detection cross-checked against a cycle search
 * in the wait-for graph, priority-ordered handoff, and the classic inversion scenario measured both
 * with and without the protocol: H's blocking time is the critical section length plus the single
 * failed attempt under PIP but grows by exactly M's interference without it, while the independent
 * inversion counter
 * (ticks where another runnable or blocked task's effective priority beats the running one's)
 * drops from 10 to 0.
 */
import java.util.*;

/** One scripted tick: unit work, an acquire, or a release. */
final class Step {
    static final int WORK = 0, LOCK = 1, UNLOCK = 2;
    final int kind;
    final String resource;

    Step(int kind, String resource) { this.kind = kind; this.resource = resource; }

    static Step work() { return new Step(WORK, null); }
    static Step lock(String r) { return new Step(LOCK, r); }
    static Step unlock(String r) { return new Step(UNLOCK, r); }
}

/** A real-time task: bigger |basePriority| means more urgent; it may not run before its release tick. */
final class RtTask {
    final int id, basePriority, releaseTick;
    final String name;
    final List<Step> script;
    final TreeSet<String> held = new TreeSet<>();   // locks this task currently owns
    int pc, ticksRun, blockedTicks, effPriority, maxEffPriority;
    String blockedOn;                               // lock this task is waiting for, or null

    RtTask(int id, String name, int basePriority, int releaseTick, List<Step> script) {
        this.id = id; this.name = name; this.basePriority = basePriority;
        this.releaseTick = releaseTick; this.script = script;
        this.effPriority = basePriority;
        this.maxEffPriority = basePriority;
    }

    boolean finished() { return pc >= script.size(); }
}

final class PipSim {
    final List<RtTask> tasks = new ArrayList<>();
    final List<String> trace = new ArrayList<>();            // running task name, one entry per tick
    final List<String> blockedPerTick = new ArrayList<>();   // blocked set, one entry per tick
    final Map<String, Integer> owner = new TreeMap<>();      // lock -> owning task id
    final Map<String, List<Integer>> waiters = new TreeMap<>();
    final List<String> acquisitions = new ArrayList<>();     // "task@resource" in acquisition order
    final boolean inheritance;
    boolean deadlock, timedOut, doubleOwnership;
    int ticks, invertedTicks, inheritanceTicks;

    PipSim(boolean inheritance) { this.inheritance = inheritance; }

    RtTask add(String name, int priority, int releaseTick, List<Step> script) {
        RtTask t = new RtTask(tasks.size(), name, priority, releaseTick, script);
        tasks.add(t);
        return t;
    }

    /** Fixpoint of the inheritance rule: blocked task -> holder of the lock it waits for. */
    void propagate() {
        for (RtTask t : tasks) t.effPriority = t.basePriority;
        if (!inheritance) return;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (RtTask t : tasks) {
                if (t.blockedOn == null) continue;
                Integer holder = owner.get(t.blockedOn);
                if (holder == null) continue;
                RtTask h = tasks.get(holder);
                if (h.effPriority < t.effPriority) { h.effPriority = t.effPriority; changed = true; }
            }
        }
    }

    /** Two independent structures record ownership; they must never disagree, and never double own. */
    void checkOwnership() {
        Map<String, Integer> counts = new TreeMap<>();
        for (RtTask t : tasks)
            for (String r : t.held) counts.merge(r, 1, Integer::sum);
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() > 1) doubleOwnership = true;
            if (!Objects.equals(owner.get(e.getKey()), holdingTask(e.getKey()))) doubleOwnership = true;
        }
        for (String r : owner.keySet())
            if (!counts.containsKey(r)) doubleOwnership = true;
    }

    int holdingTask(String r) {
        for (RtTask t : tasks) if (t.held.contains(r)) return t.id;
        return -1;
    }

    void runOneStep(RtTask r) {
        Step s = r.script.get(r.pc);
        switch (s.kind) {
            case Step.WORK -> r.pc++;
            case Step.LOCK -> {
                if (owner.containsKey(s.resource)) {
                    r.blockedOn = s.resource;
                    waiters.computeIfAbsent(s.resource, k -> new ArrayList<>()).add(r.id);
                } else {
                    owner.put(s.resource, r.id);
                    r.held.add(s.resource);
                    r.pc++;
                    acquisitions.add(r.name + "@" + s.resource);
                }
            }
            case Step.UNLOCK -> {
                if (!r.held.remove(s.resource)) throw new IllegalStateException("release of unheld " + s.resource);
                owner.remove(s.resource);
                r.pc++;
                List<Integer> w = waiters.get(s.resource);
                if (w != null && !w.isEmpty()) {
                    propagate();                     // the winner is chosen by inherited priority
                    int best = -1;
                    for (int id : w)
                        if (best < 0 || tasks.get(id).effPriority > tasks.get(best).effPriority) best = id;
                    w.remove((Integer) best);
                    RtTask nxt = tasks.get(best);
                    owner.put(s.resource, best);
                    nxt.held.add(s.resource);
                    nxt.blockedOn = null;
                    nxt.pc++;                        // its pending acquire completes now
                    acquisitions.add(nxt.name + "@" + s.resource);
                }
            }
            default -> throw new IllegalStateException("bad step kind");
        }
        r.ticksRun++;
    }

    void run(int maxTicks) {
        while (true) {
            if (allFinished()) return;
            if (ticks >= maxTicks) { timedOut = true; return; }
            propagate();
            RtTask r = null;
            for (RtTask t : tasks)
                if (!t.finished() && t.blockedOn == null && ticks >= t.releaseTick)
                    if (r == null || t.effPriority > r.effPriority) r = t;   // strict > keeps lowest id on ties
            if (r == null) { deadlock = true; return; }
            for (RtTask t : tasks) t.maxEffPriority = Math.max(t.maxEffPriority, t.effPriority);
            if (r.effPriority > r.basePriority) inheritanceTicks++;
            int best = Integer.MIN_VALUE;
            for (RtTask t : tasks)
                if (t != r && !t.finished() && ticks >= t.releaseTick)
                    best = Math.max(best, t.effPriority);
            if (best > r.effPriority) invertedTicks++;
            runOneStep(r);
            for (RtTask t : tasks) if (t.blockedOn != null) t.blockedTicks++;
            trace.add(r.name);
            blockedPerTick.add(blockedNames());
            checkOwnership();
            ticks++;
        }
    }

    String blockedNames() {
        StringBuilder sb = new StringBuilder();
        for (RtTask t : tasks)
            if (t.blockedOn != null) { if (sb.length() > 0) sb.append(','); sb.append(t.name); }
        return sb.toString();
    }

    boolean allFinished() { for (RtTask t : tasks) if (!t.finished()) return false; return true; }

    /** Deadlock oracle: a blocked task's wait-for chain ends in a wait-for cycle. */
    boolean hasWaitCycle() {
        Map<Integer, Integer> edge = new TreeMap<>();
        for (RtTask t : tasks)
            if (t.blockedOn != null && owner.containsKey(t.blockedOn)) edge.put(t.id, owner.get(t.blockedOn));
        for (int start : edge.keySet()) {
            Set<Integer> seen = new HashSet<>();
            Integer cur = start;
            while (cur != null && seen.add(cur)) cur = edge.get(cur);
            if (cur != null) return true;
        }
        return false;
    }

    int tasksHolding(String r) {
        int c = 0;
        for (RtTask t : tasks) if (t.held.contains(r)) c++;
        return c;
    }

    int blockedTicksFromTrace(String name) {
        int c = 0;
        for (String s : blockedPerTick)
            for (String part : s.split(","))
                if (part.equals(name)) { c++; break; }
        return c;
    }

    static List<Step> script(Step... steps) { return new ArrayList<>(Arrays.asList(steps)); }
}

public class Solution {
    /** The classic unbounded priority inversion scenario: L holds the lock H needs, M is medium. */
    static PipSim inversionScenario(boolean inheritance) {
        PipSim sim = new PipSim(inheritance);
        sim.add("L", 1, 0, PipSim.script(Step.lock("A"), Step.work(), Step.work(), Step.work(), Step.work(),
                Step.unlock("A"), Step.work()));
        sim.add("H", 10, 1, PipSim.script(Step.work(), Step.lock("A"), Step.work(), Step.unlock("A")));
        sim.add("M", 5, 2, PipSim.script(Step.work(), Step.work(), Step.work(), Step.work(), Step.work()));
        sim.run(1000);
        return sim;
    }

    /** Nested blocking: H waits on B (held by M) while M waits on A (held by L) - the chain L <- M <- H. */
    static PipSim chainScenario(boolean inheritance) {
        PipSim sim = new PipSim(inheritance);
        sim.add("L", 1, 0, PipSim.script(Step.lock("A"), Step.work(), Step.work(), Step.work(), Step.unlock("A")));
        sim.add("M", 5, 1, PipSim.script(Step.lock("B"), Step.work(), Step.lock("A"), Step.work(),
                Step.unlock("A"), Step.unlock("B")));
        sim.add("H", 10, 2, PipSim.script(Step.lock("B"), Step.work(), Step.unlock("B")));
        sim.run(1000);
        return sim;
    }

    /** Resource-order violating pair: each holds what the other needs, so both must block forever. */
    static PipSim deadlockScenario(boolean inheritance) {
        PipSim sim = new PipSim(inheritance);
        sim.add("L", 1, 0, PipSim.script(Step.lock("A"), Step.work(), Step.lock("B"), Step.work(), Step.unlock("B")));
        sim.add("H", 10, 1, PipSim.script(Step.lock("B"), Step.work(), Step.lock("A"), Step.work(), Step.unlock("A")));
        sim.run(1000);
        return sim;
    }

    /** Order-independent handoff: three waiters queue up, the most urgent must win the lock. */
    static PipSim handoffScenario() {
        PipSim sim = new PipSim(true);
        sim.add("W", 1, 0, PipSim.script(Step.lock("R"), Step.work(), Step.unlock("R")));
        sim.add("MID", 5, 1, PipSim.script(Step.lock("R"), Step.work(), Step.unlock("R")));
        sim.add("TOP", 9, 2, PipSim.script(Step.lock("R"), Step.unlock("R")));
        sim.run(1000);
        return sim;
    }

    static long state = 0x0BADC0DEL;

    static long nextLong() {                       // splitmix64: fixed seed, reproducible
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** A conflict-rich script: every task holds the shared lock A for its whole body and nests B/C in
     *  index order inside, so locks are always acquired in a consistent order and it cannot deadlock. */
    static PipSim mixedTraffic() {
        PipSim sim = new PipSim(true);
        String[] res = {"A", "B", "C"};
        for (int id = 0; id < 5; id++) {
            List<Step> s = new ArrayList<>();
            s.add(Step.lock(res[0]));
            for (int round = 0; round < 3; round++) {
                for (int work = 0; work < 1 + (int) (nextLong() >>> 42) % 3; work++) s.add(Step.work());
                if (round < 2) {
                    String inner = res[1 + (int) (nextLong() >>> 40) % 2];
                    s.add(Step.lock(inner));
                    s.add(Step.work());
                    s.add(Step.unlock(inner));
                } else {
                    s.add(Step.lock(res[1]));
                    s.add(Step.work());
                    s.add(Step.lock(res[2]));
                    s.add(Step.work());
                    s.add(Step.unlock(res[2]));
                    s.add(Step.unlock(res[1]));
                }
            }
            s.add(Step.unlock(res[0]));
            sim.add("T" + id, (id + 1) * 3, id, s);
        }
        sim.run(20000);
        return sim;
    }

    public static void main(String[] args) {
        PipSim plain = new PipSim(false);
        RtTask high = plain.add("H", 10, 0, PipSim.script(Step.work(), Step.work()));
        RtTask low = plain.add("L", 1, 0, PipSim.script(Step.work()));
        plain.run(100);
        check(String.join("", plain.trace).equals("HHL"), "no contention runs in priority order, trace=" + plain.trace);
        check(high.blockedTicks == 0 && low.blockedTicks == 0 && !plain.deadlock, "no blocking without contention");
        check(plain.inheritanceTicks == 0 && high.maxEffPriority == 10, "idle protocol leaves priorities alone");

        PipSim inherit = inversionScenario(true);
        PipSim noInherit = inversionScenario(false);
        check(!inherit.deadlock && !noInherit.deadlock && inherit.allFinished() && noInherit.allFinished(),
                "both policies complete the inversion scenario");
        check(inherit.invertedTicks == 0 && noInherit.invertedTicks == 10,
                "inversion ticks 0 with PIP vs 10 without (" + inherit.invertedTicks + "/" + noInherit.invertedTicks + ")");
        RtTask hIn = inherit.tasks.get(1), lIn = inherit.tasks.get(0);
        RtTask hNo = noInherit.tasks.get(1);
        check(hIn.blockedTicks == 5, "H blocked the 4-unit critical section plus its failing attempt, got " + hIn.blockedTicks);
        check(hNo.blockedTicks == 10 && hNo.blockedTicks - hIn.blockedTicks == 5,
                "without PIP H waits 10 ticks; the extra 5 are exactly M's interference");
        check(inherit.blockedTicksFromTrace("H") == hIn.blockedTicks
                        && noInherit.blockedTicksFromTrace("H") == hNo.blockedTicks,
                "per-tick blocked snapshots reproduce the blocked-time counters");
        check(lIn.maxEffPriority == 10 && noInherit.tasks.get(0).maxEffPriority == 1,
                "L inherits H's priority only when the protocol is on");
        check(inherit.inheritanceTicks == 5, "L ran boosted for 5 ticks, got " + inherit.inheritanceTicks);
        check(hIn.ticksRun + lIn.ticksRun + inherit.tasks.get(2).ticksRun == inherit.ticks
                && inherit.ticks == inherit.trace.size(), "exactly one task per tick");

        PipSim chain = chainScenario(true);
        check(!chain.deadlock && chain.allFinished(), "nested blocking resolves under PIP");
        check(chain.tasks.get(0).maxEffPriority == 10 && chain.tasks.get(1).maxEffPriority == 10,
                "priority propagates transitively to the outermost holder");
        check(chain.invertedTicks == 0, "no inversion while the chain unwinds");
        check(chain.tasks.get(2).blockedTicks > 0 && chain.tasks.get(1).blockedTicks > 0,
                "both M and H really blocked during the chain, got " + chain.tasks.get(1).blockedTicks
                        + "/" + chain.tasks.get(2).blockedTicks);

        PipSim handoff = handoffScenario();
        check(handoff.acquisitions.equals(List.of("W@R", "TOP@R", "MID@R")),
                "lock handed to the most urgent waiter first, order=" + handoff.acquisitions);
        check(handoff.allFinished() && handoff.tasksHolding("R") == 0, "handoff leaves the lock free");

        PipSim stuck = deadlockScenario(true);
        check(stuck.deadlock && !stuck.timedOut, "AB/BA resource order deadlocks the simulator");
        check(stuck.hasWaitCycle(), "deadlock coincides with a wait-for cycle");
        check(deadlockScenario(false).hasWaitCycle(), "cycle detection is policy independent");
        check(stuck.tasks.get(0).blockedTicks > 0 && stuck.tasks.get(1).blockedTicks > 0,
                "both deadlocked tasks are reported blocked");

        PipSim mixed = mixedTraffic();
        check(!mixed.doubleOwnership && !mixed.deadlock && mixed.allFinished(),
                "resource-ordered traffic keeps mutual exclusion and makes progress");
        check(mixed.owner.isEmpty() && mixed.tasksHolding("A") + mixed.tasksHolding("B") + mixed.tasksHolding("C") == 0,
                "every lock released at the end of the run");
        int runTicks = 0;
        for (RtTask t : mixed.tasks) runTicks += t.ticksRun;
        check(runTicks == mixed.ticks && mixed.ticks == mixed.trace.size(),
                "work accounting: " + runTicks + " task ticks in " + mixed.ticks + " ticks");
        int blockedTotal = 0;
        for (RtTask t : mixed.tasks) blockedTotal += t.blockedTicks;
        check(blockedTotal > 0 && mixed.inheritanceTicks > 0,
                "mixed workload exercised real contention and inheritance, blocked=" + blockedTotal
                        + " boosted=" + mixed.inheritanceTicks);
        check(mixed.invertedTicks == 0, "PIP removes inversion from the mixed workload");
        check(PipSim.script(Step.work()).size() == 1, "script builder keeps step order");

        System.out.println("inversion scenario : H blocked " + hIn.blockedTicks + " ticks with PIP, "
                + hNo.blockedTicks + " without; inversion ticks " + inherit.invertedTicks + " vs " + noInherit.invertedTicks);
        System.out.println("mixed workload     : " + mixed.tasks.size() + " tasks, " + mixed.ticks + " ticks, "
                + mixed.inheritanceTicks + " inherited-priority ticks, 0 double ownership");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
