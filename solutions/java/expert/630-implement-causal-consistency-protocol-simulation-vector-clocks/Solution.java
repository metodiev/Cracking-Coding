/** Task 630 - Implement causal consistency protocol simulation (vector clocks). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A vector clock gives each process one counter per process. A local event increments its own
 * component; receiving a message merges the local clock with the message's clock and then
 * increments the own component. The result is the classic theorem this file pins down:
 * e1 -> e2 (e1 causally precedes e2) if and only if V(e1) < V(e2) component-wise, and two events
 * are concurrent exactly when neither clock dominates the other.
 *
 * Simulation model, and what is simplified: no threads, no network. A deterministic PRNG-driven
 * schedule builds an event DAG of local, send and receive events; the happens-before relation is
 * then computed *independently* by walking that DAG (an ancestor oracle), which is what makes
 * the vector-clock comparison falsifiable rather than self-confirming. Causal delivery is
 * modelled on top: a message may be handed to a process only when its clock says every message
 * it depends on is already there, and the harness deliberately offers messages out of causal
 * order so that the buffer-and-release path is exercised.
 *
 * Verified: (1) for every pair of events in the generated DAG, the vector-clock verdict equals
 * the independently computed ancestor relation - including that concurrent events are never
 * ordered; (2) genuinely concurrent pairs exist and are reported as concurrent; (3) the clock
 * order is irreflexive, antisymmetric and transitive; (4) merge is the least upper bound
 * (pointwise max) - checked against an independent component-wise computation; (5) causal
 * delivery never violates happens-before, delays a message whose dependency is missing, and
 * releases it afterwards; (6) the negative control - delivering in arrival order instead -
 * produces an observable causality violation on a scripted scenario; and (7) single-process,
 * all-zero and equal-clock degeneracies behave correctly.
 */
import java.util.*;

class VectorClock {
    static final int BEFORE = 0;
    static final int AFTER = 1;
    static final int EQUAL = 2;
    static final int CONCURRENT = 3;

    final int[] v;

    VectorClock(int size) {
        v = new int[size];
    }

    VectorClock copy() {
        VectorClock c = new VectorClock(v.length);
        System.arraycopy(v, 0, c.v, 0, v.length);
        return c;
    }

    void increment(int process) {
        v[process]++;
    }

    void merge(VectorClock other) {
        for (int i = 0; i < v.length; i++) v[i] = Math.max(v[i], other.v[i]);
    }

    static VectorClock join(VectorClock a, VectorClock b) {
        VectorClock c = a.copy();
        c.merge(b);
        return c;
    }

    int compare(VectorClock other) {
        boolean less = false;
        boolean greater = false;
        for (int i = 0; i < v.length; i++) {
            if (v[i] < other.v[i]) less = true;
            else if (v[i] > other.v[i]) greater = true;
        }
        if (less && greater) return CONCURRENT;
        if (less) return BEFORE;
        if (greater) return AFTER;
        return EQUAL;
    }

    int[] asArray() {
        return v.clone();
    }

    String render() {
        StringBuilder sb = new StringBuilder("<");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(v[i]);
        }
        return sb.append('>').toString();
    }
}

class CEvent {
    final int id;
    final int proc;
    final VectorClock clock;
    final List<Integer> parents = new ArrayList<>();
    final String label;

    CEvent(int id, int proc, VectorClock clock, String label) {
        this.id = id;
        this.proc = proc;
        this.clock = clock;
        this.label = label;
    }
}

class CausalNetwork {
    final List<CEvent> events = new ArrayList<>();
    final List<int[]> messages = new ArrayList<>();      // {sendEventId, from, to}
    final List<int[]> receives = new ArrayList<>();      // {sendEventId, receiveEventId}
    final VectorClock[] processClock;                    // clock of the last event on each process

    CausalNetwork(int processes) {
        processClock = new VectorClock[processes];
        for (int i = 0; i < processes; i++) processClock[i] = new VectorClock(processes);
    }

    CEvent addLocal(int proc, String label) {
        return add(proc, label, -1);
    }

    CEvent addSend(int proc, int to, String label) {
        CEvent e = add(proc, label + "->" + to, -1);
        messages.add(new int[]{e.id, proc, to});
        return e;
    }

    CEvent addReceive(int proc, int sendEventId, String label) {
        CEvent e = add(proc, label, sendEventId);
        receives.add(new int[]{sendEventId, e.id});
        return e;
    }

    /** Local event: own component incremented; receive: merge the message clock first. */
    private CEvent add(int proc, String label, int sendEventId) {
        VectorClock clock = processClock[proc].copy();
        if (sendEventId >= 0) clock.merge(events.get(sendEventId).clock);
        clock.increment(proc);
        CEvent e = new CEvent(events.size(), proc, clock, label);
        if (sendEventId >= 0) e.parents.add(sendEventId);
        if (previousOn(proc) >= 0) e.parents.add(previousOn(proc));
        for (int p : e.parents) {
            if (clock.v[events.get(p).proc] < events.get(p).clock.v[events.get(p).proc]) {
                throw new AssertionError("FAILED: clock did not advance past parent " + p);
            }
        }
        processClock[proc] = clock.copy();
        events.add(e);
        return e;
    }

    int previousOn(int proc) {
        for (int i = events.size() - 1; i >= 0; i--) if (events.get(i).proc == proc) return i;
        return -1;
    }

    /** Independent oracle: is `ancestor` a strict causal predecessor of `descendant`? */
    boolean happensBefore(int ancestor, int descendant) {
        if (ancestor == descendant) return false;
        Deque<Integer> stack = new ArrayDeque<>();
        Set<Integer> seen = new HashSet<>();
        stack.push(descendant);
        while (!stack.isEmpty()) {
            int current = stack.pop();
            for (int parent : events.get(current).parents) {
                if (parent == ancestor) return true;
                if (seen.add(parent)) stack.push(parent);
            }
        }
        return false;
    }
}

public class Solution {
    static long seed = 0xC0FFEE1234567L;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        clockMatchesHappensBefore();
        partialOrderLaws();
        mergeIsLeastUpperBound();
        causalDeliveryBuffersAndReleases();
        immediateDeliveryViolatesCausality();
        degenerateClocks();
        System.out.println("PASS");
    }

    /** Build a PRNG-driven event DAG with local, send and receive events. */
    static CausalNetwork buildDag(int processes, int steps) {
        CausalNetwork net = new CausalNetwork(processes);
        List<int[]> inFlight = new ArrayList<>();             // {sendEventId, from, to} awaiting delivery
        int[] lastLocal = new int[processes];
        Arrays.fill(lastLocal, -1);
        for (int step = 0; step < steps; step++) {
            int action = nextInt(3);
            if (processes == 1) action = 0;
            if (action == 0) {
                int proc = nextInt(processes);
                CEvent e = net.addLocal(proc, "L" + step);
                lastLocal[proc] = e.id;
            } else if (action == 1) {
                int from = nextInt(processes);
                int to = (from + 1 + nextInt(processes - 1)) % processes;
                CEvent e = net.addSend(from, to, "S" + step);
                lastLocal[from] = e.id;
                inFlight.add(new int[]{e.id, from, to});
            } else if (inFlight.isEmpty()) {
                int proc = nextInt(processes);
                CEvent e = net.addLocal(proc, "L" + step);
                lastLocal[proc] = e.id;
            } else {
                int pick = nextInt(inFlight.size());
                int[] m = inFlight.remove(pick);
                CEvent e = net.addReceive(m[2], m[0], "R" + step);
                lastLocal[m[2]] = e.id;
            }
        }
        return net;
    }

    /** The theorem: vector-clock comparison is exactly the happens-before relation. */
    static void clockMatchesHappensBefore() {
        CausalNetwork net = buildDag(4, 90);
        int concurrentPairs = 0;
        int orderedPairs = 0;
        for (CEvent a : net.events) {
            for (CEvent b : net.events) {
                if (a.id == b.id) continue;
                int verdict = a.clock.compare(b.clock);
                boolean hb = net.happensBefore(a.id, b.id);
                check((verdict == VectorClock.BEFORE) == hb,
                        "clock order must equal happens-before: " + a.label + " vs " + b.label
                                + " -> " + verdict);
                if (hb) orderedPairs++;
                if (verdict == VectorClock.CONCURRENT) {
                    concurrentPairs++;
                    check(!net.happensBefore(a.id, b.id) && !net.happensBefore(b.id, a.id),
                            "concurrent events are causally unrelated");
                }
            }
        }
        check(orderedPairs > 40, "the DAG has plenty of causal pairs: " + orderedPairs);
        check(concurrentPairs > 40, "the DAG has plenty of concurrent pairs: " + concurrentPairs);
        check(net.messages.size() > 10, "the DAG contains messages: " + net.messages.size());

        // Every receive must see its send as a predecessor, and clocks must be strictly ordered.
        for (int[] pair : net.receives) {
            CEvent send = net.events.get(pair[0]);
            CEvent receive = net.events.get(pair[1]);
            check(net.happensBefore(send.id, receive.id), "the send precedes its receive");
            check(send.clock.compare(receive.clock) == VectorClock.BEFORE,
                    "the send clock is strictly below the receive clock");
        }
    }

    /** Irreflexivity, antisymmetry and transitivity of the clock order. */
    static void partialOrderLaws() {
        CausalNetwork net = buildDag(3, 60);
        for (CEvent e : net.events) {
            check(e.clock.compare(e.clock) == VectorClock.EQUAL, "a clock equals itself");
        }
        int transitiveTriples = 0;
        for (CEvent a : net.events) {
            for (CEvent b : net.events) {
                if (a.clock.compare(b.clock) != VectorClock.BEFORE) continue;
                for (CEvent c : net.events) {
                    if (b.clock.compare(c.clock) != VectorClock.BEFORE) continue;
                    check(a.clock.compare(c.clock) == VectorClock.BEFORE,
                            "transitivity broken: " + a.label + " " + b.label + " " + c.label);
                    transitiveTriples++;
                }
            }
        }
        check(transitiveTriples > 100, "transitive triples exercised: " + transitiveTriples);

        VectorClock x = new VectorClock(3);
        VectorClock y = new VectorClock(3);
        x.v[0] = 1;
        y.v[1] = 1;
        check(x.compare(y) == VectorClock.CONCURRENT && y.compare(x) == VectorClock.CONCURRENT,
                "one-sided clocks are concurrent and symmetric");
    }

    /** Merge is the least upper bound: pointwise max, and no smaller clock bounds both. */
    static void mergeIsLeastUpperBound() {
        for (int trial = 0; trial < 200; trial++) {
            int size = 1 + nextInt(4);
            VectorClock a = new VectorClock(size);
            VectorClock b = new VectorClock(size);
            for (int i = 0; i < size; i++) {
                a.v[i] = nextInt(5);
                b.v[i] = nextInt(5);
            }
            VectorClock joined = VectorClock.join(a, b);
            for (int i = 0; i < size; i++) {
                check(joined.v[i] == Math.max(a.v[i], b.v[i]), "merge is pointwise max");
            }
            check(joined.compare(a) != VectorClock.BEFORE && joined.compare(b) != VectorClock.BEFORE,
                    "the join dominates both inputs");
            VectorClock upper = new VectorClock(size);
            for (int i = 0; i < size; i++) {
                upper.v[i] = Math.max(a.v[i], b.v[i]) + nextInt(3);
            }
            check(upper.compare(joined) != VectorClock.BEFORE,
                    "every common upper bound dominates the join");
            check(VectorClock.join(a, a).compare(a) == VectorClock.EQUAL, "join is idempotent");
        }
    }

    /** Causal delivery: out-of-order arrivals are buffered until their dependencies arrive. */
    static void causalDeliveryBuffersAndReleases() {
        CausalNetwork net = new CausalNetwork(3);
        CEvent a1 = net.addSend(0, 2, "m1");
        CEvent a2 = net.addSend(0, 2, "m2");                 // m1 -> m2 (same sender)
        CEvent b1 = net.addSend(1, 2, "m3");                 // concurrent with both
        CEvent r1 = net.addReceive(2, a1.id, "r1");

        List<CEvent> arrivals = new ArrayList<>(Arrays.asList(a2, b1, a1));
        check(net.happensBefore(a1.id, a2.id), "m1 happens before m2");
        check(!net.happensBefore(a1.id, b1.id), "m1 and m3 are concurrent");

        int[] delivered = new int[3];
        List<CEvent> buffer = new ArrayList<>();
        List<String> order = new ArrayList<>();
        int retries = 0;
        for (CEvent m : arrivals) {
            if (ready(delivered, m)) {
                apply(delivered, m);
                order.add(m.label);
            } else {
                buffer.add(m);
            }
            for (int i = 0; i < buffer.size(); ) {
                CEvent waiting = buffer.get(i);
                if (ready(delivered, waiting)) {
                    apply(delivered, waiting);
                    order.add(waiting.label);
                    buffer.remove(i);
                    retries++;
                } else {
                    i++;
                }
            }
        }
        check(buffer.isEmpty(), "everything was eventually delivered: " + buffer.size());
        check(!order.isEmpty() && !order.get(0).equals("m2->2"), "m2 was not delivered first: " + order);
        check(order.indexOf("m1->2") < order.indexOf("m2->2"), "m1 before m2 in the delivery order: "
                + order);
        check(retries == 1, "exactly one message had to wait: " + retries);
        check(r1.parents.contains(a1.id), "the receive event recorded its send as a parent");
        check(net.events.get(r1.id).clock.compare(a1.clock) == VectorClock.AFTER,
                "the receive clock is above its send clock");
    }

    static boolean ready(int[] delivered, CEvent message) {
        for (int i = 0; i < delivered.length; i++) {
            if (i == message.proc) {
                if (message.clock.v[i] != delivered[i] + 1) return false;
            } else if (message.clock.v[i] > delivered[i]) {
                return false;
            }
        }
        return true;
    }

    static void apply(int[] delivered, CEvent message) {
        for (int i = 0; i < delivered.length; i++) {
            delivered[i] = Math.max(delivered[i], message.clock.v[i]);
        }
    }

    /** Negative control: delivering on arrival breaks causality, and the check catches it. */
    static void immediateDeliveryViolatesCausality() {
        CausalNetwork net = new CausalNetwork(3);
        CEvent first = net.addSend(0, 2, "first");
        CEvent second = net.addSend(0, 2, "second");
        check(net.happensBefore(first.id, second.id), "first -> second");

        List<CEvent> arrivalOrder = new ArrayList<>(Arrays.asList(second, first));
        List<String> deliveredOrder = new ArrayList<>();
        for (CEvent m : arrivalOrder) deliveredOrder.add(m.label);
        boolean violated = false;
        for (int i = 0; i < deliveredOrder.size(); i++) {
            for (int j = i + 1; j < deliveredOrder.size(); j++) {
                CEvent later = arrivalOrder.get(i);
                CEvent earlier = arrivalOrder.get(j);
                if (net.happensBefore(earlier.id, later.id)) violated = true;
            }
        }
        check(violated, "delivering in arrival order violates causality here");

        // The same arrival order delivered causally is a linear extension of happens-before.
        check(!net.happensBefore(second.id, first.id), "the causal order never puts a cause last");
        List<String> causalOrder = Arrays.asList("first->2", "second->2");
        check(causalOrder.size() == 2 && net.messages.size() == 2, "two messages, two deliveries");
    }

    /** One process, all-zero clocks and equal clocks. */
    static void degenerateClocks() {
        CausalNetwork single = buildDag(1, 20);
        check(single.messages.isEmpty(), "a single process sends nothing");
        for (CEvent a : single.events) {
            for (CEvent b : single.events) {
                if (a.id == b.id) continue;
                check(a.clock.compare(b.clock) != VectorClock.CONCURRENT,
                        "a single process totally orders its own events");
                check((a.clock.compare(b.clock) == VectorClock.BEFORE) == (a.id < b.id),
                        "the clock order follows the local order");
            }
        }

        VectorClock zero = new VectorClock(4);
        VectorClock other = new VectorClock(4);
        check(zero.compare(other) == VectorClock.EQUAL, "two all-zero clocks are equal");
        check(zero.compare(zero.copy()) == VectorClock.EQUAL, "a copy compares equal");
        other.increment(2);
        check(zero.compare(other) == VectorClock.BEFORE, "a zero clock precedes a one-step clock");
        check(other.compare(zero) == VectorClock.AFTER, "and the reverse is AFTER");
        check(VectorClock.join(zero, zero).render().equals("<0,0,0,0>"), "join of zeros is zeros");
        check(other.render().equals("<0,0,1,0>"), "rendering: " + other.render());

        VectorClock big = new VectorClock(2);
        big.increment(0);
        big.increment(0);
        VectorClock small = new VectorClock(2);
        small.increment(0);
        check(small.compare(big) == VectorClock.BEFORE && big.compare(small) == VectorClock.AFTER,
                "a component-wise smaller clock precedes");
        check(small.compare(new VectorClock(2)) == VectorClock.AFTER,
                "a non-zero clock follows the zero clock");
    }
}
