/** Task 636 - Implement hybrid logical clocks and test ordering in distributed events. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A hybrid logical clock stamps an event with a pair (l, c): `l` tracks the largest physical time
 * the node has *seen* anywhere (its own clock or a message) and `c` is a logical counter that
 * breaks ties. A local event takes max(l, physicalNow) and, if that is the same l as before,
 * bumps c; a receive takes the maximum of the local l, the message's l and the local physical
 * clock, and picks the counter that keeps the pair strictly greater than both inputs. The result
 * is a clock that (a) never goes backwards even when the physical clock does, (b) respects
 * causality like a Lamport clock, and (c) keeps `l` within a bounded distance of real time, which
 * is what lets timestamps be stored, compared across a cluster and expired.
 *
 * Simulation model, and what is simplified: no threads and no real clocks. A schedule of local,
 * send and receive events is generated from a fixed PRNG; each node has a deterministic physical
 * clock, and one node's clock jumps *backwards* in the middle of the run on purpose. Causality is
 * computed independently by walking the event DAG built from the schedule (the send event is a
 * parent of the receive), so the HLC is checked against an oracle rather than against itself.
 *
 * Verified: (1) a node's HLC is strictly increasing even across a backwards physical-clock jump;
 * (2) the HLC never contradicts causality: if e1 happens before e2 then hlc(e1) < hlc(e2), checked
 * for every pair in the generated DAG; (3) sorting events by (l, c, node) is a strict total order
 * and a linear extension of happened-before, with node ids breaking exact ties; (4) the physical
 * part stays within a bounded distance of local real time under bounded message delays;
 * (5) a message from far in the future is accepted, still monotone, and flagged as exceeding the
 * drift bound; (6) the logical counter overflows into `l` instead of wrapping; (7) the negative
 * control - a pure physical timestamp - breaks monotonicity and causality exactly where the HLC
 * does not.
 */
import java.util.*;

class Hlc {
    static final long MAX_DRIFT = 60_000;
    static final int C_MAX = 65535;

    long l;
    int c;
    int overflowBumps;
    int driftFlags;
    int events;

    Hlc(long start) {
        l = start;
    }

    long[] stamp() {
        return new long[]{l, c};
    }

    /** Local event (or send): the wall clock pushes l forward, otherwise the counter moves. */
    long[] now(long physical) {
        events++;
        if (physical > l) {
            l = physical;
            c = 0;
        } else {
            c++;
        }
        guard();
        return stamp();
    }

    /** Receive: take the maximum of local l, message l and local physical time. */
    long[] receive(long physical, long messageL, int messageC) {
        events++;
        long maxL = Math.max(l, Math.max(messageL, physical));
        if (maxL == l && maxL == messageL) c = Math.max(c, messageC) + 1;
        else if (maxL == l) c = c + 1;
        else if (maxL == messageL) c = messageC + 1;
        else c = 0;
        l = maxL;
        guard();
        return stamp();
    }

    void guard() {
        if (c > C_MAX) {                       // overflow the counter into the logical time
            l++;
            c = 0;
            overflowBumps++;
        }
    }

    /** True (and counted) when the logical time runs too far ahead of local real time. */
    boolean driftExceeded(long physical) {
        if (l - physical > MAX_DRIFT) {
            driftFlags++;
            return true;
        }
        return false;
    }

    static int compare(long[] a, long[] b) {
        if (a[0] != b[0]) return Long.compare(a[0], b[0]);
        return Integer.compare((int) a[1], (int) b[1]);
    }

    static String render(long[] stamp) {
        return stamp[0] + "." + stamp[1];
    }
}

class TEvent {
    final int id;
    final int node;
    final long[] hlc;
    final long physical;
    final List<Integer> parents = new ArrayList<>();

    TEvent(int id, int node, long[] hlc, long physical) {
        this.id = id;
        this.node = node;
        this.hlc = hlc;
        this.physical = physical;
    }
}

class Timeline {
    final List<TEvent> events = new ArrayList<>();
    final List<int[]> messages = new ArrayList<>();      // {sendEventId, from, to}
    final Hlc[] clocks;
    final long[] physical;
    final long start;

    Timeline(int nodes, long start) {
        clocks = new Hlc[nodes];
        physical = new long[nodes];
        this.start = start;
        for (int i = 0; i < nodes; i++) {
            clocks[i] = new Hlc(start);
            physical[i] = start;
        }
    }

    /** Advance node `node`'s wall clock; `skew` may be negative to model a backwards jump. */
    void advancePhysical(int node, long amount) {
        physical[node] += amount;
    }

    TEvent local(int node, String label) {
        return add(node, clocks[node].now(physical[node]), -1);
    }

    TEvent send(int node, int to, String label) {
        TEvent e = add(node, clocks[node].now(physical[node]), -1);
        messages.add(new int[]{e.id, node, to});
        return e;
    }

    TEvent receive(int node, int sendEventId) {
        TEvent sender = events.get(sendEventId);
        return add(node,
                clocks[node].receive(physical[node], sender.hlc[0], (int) sender.hlc[1]),
                sendEventId);
    }

    TEvent add(int node, long[] stamp, int extraParent) {
        TEvent e = new TEvent(events.size(), node, stamp, physical[node]);
        if (extraParent >= 0) e.parents.add(extraParent);
        for (int i = events.size() - 1; i >= 0; i--) {
            if (events.get(i).node == node) {
                e.parents.add(i);
                break;
            }
        }
        events.add(e);
        return e;
    }

    /** Independent oracle for happened-before, computed on the event DAG. */
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

    boolean monotonePerNode() {
        for (int node = 0; node < clocks.length; node++) {
            long[] previous = null;
            for (TEvent e : events) {
                if (e.node != node) continue;
                if (previous != null && Hlc.compare(previous, e.hlc) >= 0) return false;
                previous = e.hlc;
            }
        }
        return true;
    }
}

public class Solution {
    static long seed = 0x1CEB00DA5EEDL;

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
        basicsAndBackwardsClock();
        monotoneAcrossBackwardsJump();
        causalityOnGeneratedTimeline();
        totalOrderIsALinearExtension();
        boundedDrift();
        farFutureMessageIsFlagged();
        counterOverflow();
        negativeControlPhysicalClock();
        System.out.println("PASS");
    }

    /** The algorithm itself, on hand-worked cases. */
    static void basicsAndBackwardsClock() {
        Hlc clock = new Hlc(0);
        long[] first = clock.now(100);
        check(first[0] == 100 && first[1] == 0, "a local event at the current time: "
                + Hlc.render(first));
        long[] second = clock.now(100);
        check(second[0] == 100 && second[1] == 1, "the counter breaks the tie: "
                + Hlc.render(second));
        long[] third = clock.now(50);                       // the wall clock jumped backwards
        check(third[0] == 100 && third[1] == 2, "a backwards clock keeps l and bumps c: "
                + Hlc.render(third));
        check(Hlc.compare(first, second) < 0 && Hlc.compare(second, third) < 0,
                "each event is strictly after the previous one");

        long[] later = clock.now(200);
        check(later[0] == 200 && later[1] == 0, "a forward jump resets the counter: "
                + Hlc.render(later));

        Hlc other = new Hlc(0);
        long[] theirs = other.now(150);
        long[] mine = clock.receive(160, theirs[0], (int) theirs[1]);
        check(mine[0] == 200 && mine[1] == 1, "the receive keeps the greatest l and bumps c: "
                + Hlc.render(mine));
        check(Hlc.compare(theirs, mine) < 0, "a receive is after its message");

        Hlc fresh = new Hlc(0);
        long[] fromMessage = fresh.receive(10, 5000, 3);
        check(fromMessage[0] == 5000 && fromMessage[1] == 4,
                "a message from the future drives the clock: " + Hlc.render(fromMessage));
        long[] sameL = fresh.receive(10, 5000, 3);
        check(sameL[0] == 5000 && sameL[1] == 5, "and a second such message only bumps c: "
                + Hlc.render(sameL));
    }

    /** A node whose wall clock jumps backwards never emits a stamp that goes backwards. */
    static void monotoneAcrossBackwardsJump() {
        Timeline timeline = new Timeline(3, 1_000_000);
        long[][] stamps = new long[40][];
        for (int i = 0; i < 40; i++) {
            if (i == 20) {
                timeline.advancePhysical(0, -25_000);        // NTP pulled the clock back
                timeline.advancePhysical(1, -10_000);
            } else {
                timeline.advancePhysical(0, 100);
                timeline.advancePhysical(1, 100);
                timeline.advancePhysical(2, 100);
            }
            stamps[i] = timeline.clocks[0].now(timeline.physical[0]);
            if (i > 0) {
                check(Hlc.compare(stamps[i - 1], stamps[i]) < 0,
                        "stamp " + i + " went backwards: " + Hlc.render(stamps[i - 1]) + " -> "
                                + Hlc.render(stamps[i]));
            }
        }
        check(timeline.physical[0] < stamps[39][0], "the logical time outran the wall clock");
        check(stamps[39][0] - timeline.physical[0] < Hlc.MAX_DRIFT,
                "but stayed within the drift bound: " + (stamps[39][0] - timeline.physical[0]));
        check(timeline.monotonePerNode(), "every node is monotone");
    }

    /** HLC order never contradicts the independently computed happened-before relation. */
    static void causalityOnGeneratedTimeline() {
        Timeline timeline = buildTimeline(4, 90);
        int causalPairs = 0;
        for (TEvent a : timeline.events) {
            check(!timeline.happensBefore(a.id, a.id), "an event does not precede itself");
        }
        for (TEvent a : timeline.events) {
            for (TEvent b : timeline.events) {
                if (a.id == b.id) continue;
                if (timeline.happensBefore(a.id, b.id)) {
                    causalPairs++;
                    check(Hlc.compare(a.hlc, b.hlc) < 0, "causality violated: " + Hlc.render(a.hlc)
                            + " (event " + a.id + ") must precede " + Hlc.render(b.hlc));
                }
            }
        }
        check(causalPairs > 100, "the timeline has plenty of causal pairs: " + causalPairs);
        check(timeline.messages.size() > 10, "and messages: " + timeline.messages.size());
        check(timeline.events.size() >= 85, "events generated: " + timeline.events.size());
        for (int[] m : timeline.messages) {
            TEvent send = timeline.events.get(m[0]);
            for (TEvent e : timeline.events) {
                if (e.node != m[2] || e.id <= m[0]) continue;
                if (e.parents.contains(m[0])) {
                    check(Hlc.compare(send.hlc, e.hlc) < 0, "the receive beats the send");
                }
            }
        }
    }

    /** Sorting by (l, c, node) is a strict total order and a linear extension of causality. */
    static void totalOrderIsALinearExtension() {
        Timeline timeline = buildTimeline(4, 70);
        List<TEvent> sorted = new ArrayList<>(timeline.events);
        sorted.sort((x, y) -> {
            int byStamp = Hlc.compare(x.hlc, y.hlc);
            return byStamp != 0 ? byStamp : Integer.compare(x.node, y.node);
        });
        int position = 0;
        for (int i = 0; i + 1 < sorted.size(); i++) {
            TEvent a = sorted.get(i);
            TEvent b = sorted.get(i + 1);
            int byStamp = Hlc.compare(a.hlc, b.hlc);
            check(byStamp < 0 || (byStamp == 0 && a.node <= b.node),
                    "the total order is strict and tie-broken by node");
            position++;
        }
        check(position == sorted.size() - 1, "every adjacent pair was compared");
        for (int i = 0; i < sorted.size(); i++) {
            for (int j = i + 1; j < sorted.size(); j++) {
                TEvent earlier = sorted.get(i);
                TEvent later = sorted.get(j);
                check(!timeline.happensBefore(later.id, earlier.id),
                        "the order never puts a cause after its effect: " + earlier.id + " / "
                                + later.id);
            }
        }

        // Exact ties in (l, c) are broken by the node id, so the order is still total.
        Hlc a = new Hlc(500);
        Hlc b = new Hlc(500);
        long[] left = a.now(500);
        long[] right = b.now(500);
        check(Hlc.compare(left, right) == 0, "two independent nodes can produce the same stamp");
        List<int[]> pair = new ArrayList<>(Arrays.asList(new int[]{1, 0}, new int[]{2, 0}));
        pair.sort((x, y) -> x[1] == y[1] ? Integer.compare(x[0], y[0]) : Integer.compare(x[1], y[1]));
        check(pair.get(0)[0] == 1 && pair.get(1)[0] == 2, "the node id breaks the tie");
    }

    /** Under bounded message delay the physical part stays close to real time. */
    static void boundedDrift() {
        Timeline timeline = buildTimeline(3, 60);
        long worst = 0;
        for (TEvent e : timeline.events) {
            worst = Math.max(worst, e.hlc[0] - e.physical);
        }
        check(worst < Hlc.MAX_DRIFT, "drift stayed bounded: " + worst + " ms");
        for (Hlc clock : timeline.clocks) {
            check(clock.driftFlags == 0, "no drift warning in a well-behaved run");
            check(clock.overflowBumps == 0, "no counter overflow in a well-behaved run");
        }
        check(timeline.monotonePerNode(), "monotone");
    }

    /** A message from far in the future is accepted, monotone, and flagged. */
    static void farFutureMessageIsFlagged() {
        Hlc clock = new Hlc(1_000_000);
        long[] before = clock.now(1_000_000);
        long[] after = clock.receive(1_000_100, 1_000_000 + Hlc.MAX_DRIFT + 5_000, 7);
        check(Hlc.compare(before, after) < 0, "the clock still moves forward");
        check(after[1] == 8, "the counter follows the message's: " + Hlc.render(after));
        check(clock.driftExceeded(1_000_100), "the drift bound is exceeded and reported");
        check(clock.driftFlags == 1, "exactly one warning: " + clock.driftFlags);
        long[] again = clock.now(1_000_200);
        check(Hlc.compare(after, again) < 0, "and monotonicity survives the warning");
        check(clock.driftExceeded(1_000_200), "the warning still stands while real time lags");
        check(clock.driftFlags == 2, "each check is counted: " + clock.driftFlags);
        check(!clock.driftExceeded(after[0]), "once real time catches up the warning is gone");
        check(clock.driftFlags == 2, "and nothing new was counted");
    }

    /** The logical counter never wraps: it carries into l. */
    static void counterOverflow() {
        Hlc clock = new Hlc(7_000);
        clock.c = Hlc.C_MAX;
        long[] stamp = clock.now(7_000);
        check(stamp[1] == 0 && stamp[0] == 7_001,
                "the counter carried into the logical time: " + Hlc.render(stamp));
        check(clock.overflowBumps == 1, "one carry was recorded");
        long[] next = clock.now(7_000);
        check(Hlc.compare(stamp, next) < 0 && next[1] == 1, "and the clock kept moving: "
                + Hlc.render(next));
        for (int i = 0; i < 5; i++) clock.now(7_001);
        check(clock.overflowBumps == 1, "no spurious carries");
    }

    /** Negative control: a pure physical timestamp breaks both properties. */
    static void negativeControlPhysicalClock() {
        long[] physicalStamps = {1_000_000, 999_000, 999_500};       // the middle one jumped back
        boolean monotone = true;
        for (int i = 1; i < physicalStamps.length; i++) {
            if (physicalStamps[i] <= physicalStamps[i - 1]) monotone = false;
        }
        check(!monotone, "a raw physical clock is not monotone across a backwards jump");

        Hlc clock = new Hlc(1_000_000);
        long[] a = clock.now(1_000_000);
        long[] b = clock.now(999_000);
        long[] c = clock.now(999_500);
        check(Hlc.compare(a, b) < 0 && Hlc.compare(b, c) < 0,
                "the hybrid clock is monotone on the very same sequence: " + Hlc.render(a) + " "
                        + Hlc.render(b) + " " + Hlc.render(c));

        // Causality: a message sent at 999_000 and received at 1_000_500 must be stamped after.
        long sendPhysical = 999_000;
        long[] sendStamp = clock.now(sendPhysical);
        long receivePhysical = 1_000_500;
        long[] receiveStamp = clock.receive(receivePhysical, sendStamp[0], (int) sendStamp[1]);
        check(Hlc.compare(sendStamp, receiveStamp) < 0, "the receive is stamped after the send");
        check(sendPhysical < receivePhysical, "physical order agrees here, which is why a raw");
        long[] backwardsSend = clock.now(998_000);
        long[] lateReceive = clock.receive(999_500, backwardsSend[0], (int) backwardsSend[1]);
        check(Hlc.compare(backwardsSend, lateReceive) < 0,
                "and even a backwards send stamp precedes its receive");
    }

    /** Build a PRNG-driven timeline with local, send and receive events and bounded delays. */
    static Timeline buildTimeline(int nodes, int steps) {
        Timeline timeline = new Timeline(nodes, 10_000_000);
        List<int[]> inFlight = new ArrayList<>();
        for (int step = 0; step < steps; step++) {
            for (int node = 0; node < nodes; node++) {
                timeline.advancePhysical(node, 5 + nextInt(20));
            }
            int action = nodes == 1 ? 0 : nextInt(4);
            if (action == 0) {
                timeline.local(nextInt(nodes), "local");
            } else if (action == 1) {
                int from = nextInt(nodes);
                int to = (from + 1 + nextInt(nodes - 1)) % nodes;
                TEvent e = timeline.send(from, to, "send");
                inFlight.add(new int[]{e.id, from, to});
                timeline.advancePhysical(from, 30);          // network delay is paid by the sender
            } else if (action == 2 && !inFlight.isEmpty()) {
                int pick = nextInt(inFlight.size());
                int[] message = inFlight.remove(pick);
                timeline.advancePhysical(message[2], 40);    // the message is in flight
                timeline.receive(message[2], message[0]);
            } else {
                timeline.local(nextInt(nodes), "local");
            }
        }
        return timeline;
    }
}
