/** Task 616 - Implement distributed snapshot algorithm (Chandy-Lamport). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * The classic algorithm is simulated over a fixed directed communication graph with FIFO channels. Each
 * process holds a balance; a transfer debits the sender and enqueues a message, a delivery credits the
 * receiver. A snapshot is started by one process, which records its own state and then sends a marker on
 * every outgoing channel. A process that receives its first marker records its state, starts recording
 * every incoming channel except the one the marker arrived on (which is empty by definition) and sends
 * markers on its outgoing channels; a marker on a channel that is already being recorded closes that
 * channel. The recorded channel contents are exactly the messages that were in flight at the cut.
 *
 * Simplified: the schedule of sends, deliveries and the snapshot trigger is an explicit list of events
 * (single-threaded simulation, no threads or sockets) and every process sends its markers once, when it
 * records. Verified: the algorithm's recorded channel contents equal the messages that the sender had
 * sent before its own recording and the receiver received after its own recording (the FIFO argument
 * behind the algorithm), the recorded balances equal an independently replayed cut, the total of all
 * balances plus all in-flight money is invariant after every event and the snapshot reproduces that
 * total ("sum of balances + transit == total money"), and an uncoordinated snapshot is shown to break
 * that invariant.
 */
import java.util.*;

class SnapshotProcess {
    final int id;
    final int initialBalance;
    int balance;
    boolean recorded;
    int recordedBalance = -1;                 // the snapshot value of this process
    int recordedAtEvent = -1;
    int receivedWhenRecorded = -1;
    boolean markersSent;

    SnapshotProcess(int id, int initialBalance) {
        this.id = id;
        this.initialBalance = initialBalance;
        this.balance = initialBalance;
    }
}

class ChannelState {
    final int from, to;
    final Deque<Integer> queue = new ArrayDeque<>();     // messages and -1 for a marker
    final List<Integer> transit = new ArrayList<>();     // what the algorithm records
    boolean recording;
    boolean closed;
    int markersSeen;

    ChannelState(int from, int to) {
        this.from = from;
        this.to = to;
    }
}

class SnapshotSimulator {
    static final int TRANSFER = 0, DELIVER = 1, INITIATE = 2, MARKER = -1;

    final SnapshotProcess[] processes;
    final List<ChannelState> channels = new ArrayList<>();
    final List<int[]> transferLog = new ArrayList<>();   // {from, to, value, sendEvent, deliverEvent}
    final List<ChannelState> outgoing = new ArrayList<>();
    final List<ChannelState> incoming = new ArrayList<>();
    int eventIndex;
    int totalTransferred;

    SnapshotSimulator(int[] initialBalances, int[][] graph) {
        processes = new SnapshotProcess[initialBalances.length];
        for (int i = 0; i < initialBalances.length; i++) {
            processes[i] = new SnapshotProcess(i, initialBalances[i]);
            totalTransferred += initialBalances[i];
        }
        for (int[] edge : graph) channels.add(new ChannelState(edge[0], edge[1]));
    }

    private ChannelState channel(int from, int to) {
        for (ChannelState c : channels) if (c.from == from && c.to == to) return c;
        throw new IllegalArgumentException("no channel " + from + " -> " + to);
    }

    List<ChannelState> outgoingOf(int id) {
        outgoing.clear();
        for (ChannelState c : channels) if (c.from == id) outgoing.add(c);
        return new ArrayList<>(outgoing);
    }

    List<ChannelState> incomingOf(int id) {
        incoming.clear();
        for (ChannelState c : channels) if (c.to == id) incoming.add(c);
        return new ArrayList<>(incoming);
    }

    /** Money in flight, i.e. the value of the messages that are still queued somewhere. */
    int inFlight() {
        int sum = 0;
        for (ChannelState c : channels) {
            for (int message : c.queue) if (message != MARKER) sum += message;
        }
        return sum;
    }

    /** Sum of all balances plus everything still in flight: must never change. */
    int conservedTotal() {
        int sum = inFlight();
        for (SnapshotProcess p : processes) sum += p.balance;
        return sum;
    }

    void record(SnapshotProcess p, ChannelState markerChannel) {
        p.recorded = true;
        p.recordedBalance = p.balance;
        p.recordedAtEvent = eventIndex;
        p.receivedWhenRecorded = deliveredPerProcess[p.id];
        List<ChannelState> in = incomingOf(p.id);
        for (ChannelState c : in) {
            if (c == markerChannel) {
                c.recording = false;
                c.closed = true;                             // the marker arrived before recording started
            } else if (!c.closed) {
                c.recording = true;                          // record this channel from now on
            }
        }
        if (!p.markersSent) {
            p.markersSent = true;
            for (ChannelState c : outgoingOf(p.id)) c.queue.addLast(MARKER);
        }
    }

    private int deliveredCount;
    private final int[] deliveredPerProcess = new int[32];

    void transfer(int from, int to, int value) {
        ChannelState c = channel(from, to);
        processes[from].balance -= value;
        c.queue.addLast(value);
        transferLog.add(new int[]{from, to, value, eventIndex, -1});
    }

    void deliver(int from, int to) {
        ChannelState c = channel(from, to);
        Integer message = c.queue.pollFirst();
        if (message == null) throw new IllegalStateException("empty channel " + from + " -> " + to);
        if (message == MARKER) {
            c.markersSeen++;
            SnapshotProcess p = processes[to];
            if (!p.recorded) {
                record(p, c);
            } else {
                c.recording = false;
                c.closed = true;
            }
            return;
        }
        processes[to].balance += message;
        deliveredCount++;
        deliveredPerProcess[to]++;
        if (c.recording) c.transit.add(message);
        for (int i = transferLog.size() - 1; i >= 0; i--) {
            int[] entry = transferLog.get(i);
            if (entry[0] == from && entry[1] == to && entry[2] == message && entry[4] < 0) {
                entry[4] = eventIndex;
                break;
            }
        }
    }

    void initiate(int id) {
        SnapshotProcess p = processes[id];
        if (p.recorded) throw new IllegalStateException("already recorded");
        record(p, null);
    }

    /** Runs a script of {kind, from, to, value} events and then drains every channel. */
    List<int[]> runScript(List<int[]> script) {
        for (int[] event : script) {
            eventIndex++;
            switch (event[0]) {
                case TRANSFER: transfer(event[1], event[2], event[3]); break;
                case DELIVER: deliver(event[1], event[2]); break;
                case INITIATE: initiate(event[1]); break;
                default: throw new IllegalStateException("unknown event " + event[0]);
            }
        }
        boolean progress = true;
        while (progress) {
            progress = false;
            for (ChannelState c : channels) {
                if (!c.queue.isEmpty()) {
                    eventIndex++;
                    deliver(c.from, c.to);
                    progress = true;
                }
            }
        }
        return transferLog;
    }

    boolean everyoneRecorded() {
        for (SnapshotProcess p : processes) if (!p.recorded) return false;
        return true;
    }

    boolean everyChannelClosed() {
        for (ChannelState c : channels) if (!c.closed) return false;
        return true;
    }
}

public class Solution {
    private static long state = 0xBE5466CF34E90C6CL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int[][] graph() {
        return new int[][]{{0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 1}, {3, 0}};
    }

    /** A deterministic script that keeps every channel queue non-empty when it delivers. */
    static List<int[]> buildScript(int processes, int[][] edges, int events) {
        List<Deque<Integer>> shadow = new ArrayList<>();
        for (int i = 0; i < edges.length; i++) shadow.add(new ArrayDeque<>());
        List<int[]> script = new ArrayList<>();
        boolean initiated = false;
        int initiateAt = 5 + (int) ((nextLong() >>> 33) % 4);
        for (int e = 0; e < events; e++) {
            if (!initiated && e == initiateAt) {
                int initiator = (int) ((nextLong() >>> 33) % processes);
                script.add(new int[]{SnapshotSimulator.INITIATE, initiator, 0, 0});
                for (int c = 0; c < edges.length; c++) {
                    if (edges[c][0] == initiator) shadow.get(c).addLast(SnapshotSimulator.MARKER);
                }
                initiated = true;
                continue;
            }
            List<Integer> ready = new ArrayList<>();
            for (int c = 0; c < shadow.size(); c++) if (!shadow.get(c).isEmpty()) ready.add(c);
            if (ready.isEmpty() || (nextLong() >>> 33) % 100 < 55) {
                int c = (int) ((nextLong() >>> 33) % edges.length);
                int value = 1 + (int) ((nextLong() >>> 33) % 20);
                shadow.get(c).addLast(value);
                script.add(new int[]{SnapshotSimulator.TRANSFER, edges[c][0], edges[c][1], value});
            } else {
                int c = ready.get((int) ((nextLong() >>> 33) % ready.size()));
                shadow.get(c).pollFirst();
                script.add(new int[]{SnapshotSimulator.DELIVER, edges[c][0], edges[c][1], 0});
            }
        }
        return script;
    }

    /** The cut that the log implies: debits before the sender recorded, credits before the receiver did. */
    static int[] replayCut(List<int[]> log, SnapshotSimulator sim, int[] recordedAt, int[] initial) {
        int[] balances = initial.clone();
        for (int[] entry : log) {
            int from = entry[0], to = entry[1], value = entry[2], sent = entry[3], delivered = entry[4];
            if (sent >= 0 && sent < recordedAt[from]) balances[from] -= value;
            if (delivered >= 0 && delivered < recordedAt[to]) balances[to] += value;
        }
        return balances;
    }

    static List<Integer> expectedTransit(List<int[]> log, int channelFrom, int channelTo, int[] recordedAt) {
        List<Integer> expected = new ArrayList<>();
        for (int[] entry : log) {
            if (entry[0] != channelFrom || entry[1] != channelTo) continue;
            boolean sentBeforeCut = entry[3] >= 0 && entry[3] < recordedAt[channelFrom];
            boolean receivedBeforeCut = entry[4] >= 0 && entry[4] < recordedAt[channelTo];
            if (sentBeforeCut && !receivedBeforeCut) expected.add(entry[2]);
        }
        return expected;
    }

    static void verifySnapshot(SnapshotSimulator sim, String label) {
        int[] initial = new int[sim.processes.length];
        int[] recordedAt = new int[sim.processes.length];
        for (int i = 0; i < initial.length; i++) {
            initial[i] = sim.processes[i].initialBalance;
            recordedAt[i] = sim.processes[i].recordedAtEvent;
        }
        check(sim.everyoneRecorded(), label + ": every process recorded a state");
        check(sim.everyChannelClosed(), label + ": every channel was closed by a marker");
        // 1. the recorded balances are the balances of the cut
        int[] cutBalances = replayCut(sim.transferLog, sim, recordedAt, initial);
        for (int i = 0; i < initial.length; i++) {
            check(sim.processes[i].recordedBalance == cutBalances[i],
                    label + ": recorded balance of " + i + " matches the replayed cut");
            int delivered = 0;
            for (int[] entry : sim.transferLog) {
                if (entry[1] == i && entry[4] >= 0 && entry[4] < recordedAt[i]) delivered++;
            }
            check(sim.processes[i].receivedWhenRecorded == delivered,
                    label + ": delivery count of " + i + " at the cut");
        }
        // 2. the algorithm's channel contents are the messages in flight at the cut
        for (ChannelState c : sim.channels) {
            List<Integer> expected = expectedTransit(sim.transferLog, c.from, c.to, recordedAt);
            check(c.transit.equals(expected), label + ": transit on " + c.from + "->" + c.to + " is "
                    + c.transit + " but the cut says " + expected);
            check(c.markersSeen == 1, label + ": exactly one marker on " + c.from + "->" + c.to);
        }
        // 3. the conservation invariant of the snapshot
        int balanceSum = 0, transitSum = 0;
        for (SnapshotProcess p : sim.processes) balanceSum += p.recordedBalance;
        for (ChannelState c : sim.channels) for (int value : c.transit) transitSum += value;
        check(balanceSum + transitSum == sim.totalTransferred,
                label + ": snapshot balances + transit == total money");
        check(sim.conservedTotal() == sim.totalTransferred, label + ": the live system conserves the total");
    }

    public static void main(String[] args) {
        // the small textbook trace: three processes in a chain
        SnapshotSimulator chain = new SnapshotSimulator(new int[]{100, 50, 25},
                new int[][]{{0, 1}, {1, 2}, {2, 0}});
        List<int[]> script = new ArrayList<>();
        script.add(new int[]{SnapshotSimulator.TRANSFER, 0, 1, 10});
        script.add(new int[]{SnapshotSimulator.TRANSFER, 1, 2, 5});
        script.add(new int[]{SnapshotSimulator.INITIATE, 0, 0, 0});
        script.add(new int[]{SnapshotSimulator.TRANSFER, 2, 0, 7});
        script.add(new int[]{SnapshotSimulator.DELIVER, 0, 1});
        script.add(new int[]{SnapshotSimulator.TRANSFER, 1, 2, 3});
        script.add(new int[]{SnapshotSimulator.DELIVER, 1, 2});
        chain.runScript(script);
        verifySnapshot(chain, "chain");
        check(chain.processes[0].balance == 97 && chain.processes[1].balance == 52
                && chain.processes[2].balance == 26, "the chain run ends at the expected balances");
        check(chain.processes[0].recordedBalance == 90 && chain.processes[1].recordedBalance == 52
                && chain.processes[2].recordedBalance == 26, "the snapshot values are the cut values");
        check(chain.channels.get(0).transit.isEmpty() && chain.channels.get(1).transit.isEmpty(),
                "0->1 and 1->2 were closed by their own marker and hold nothing");
        check(chain.channels.get(2).transit.equals(Arrays.asList(7)),
                "the transfer 2->0 is the only message in flight at the cut");

        // randomized graphs and schedules
        int[][] edges = graph();
        SnapshotSimulator first = null;
        for (int trial = 0; trial < 25; trial++) {
            SnapshotSimulator sim = new SnapshotSimulator(new int[]{200, 150, 100, 50}, edges);
            List<int[]> events = buildScript(4, edges, 24);
            sim.runScript(events);
            verifySnapshot(sim, "random " + trial);
            if (first == null) first = sim;
        }

        // a marker that arrives at a process that already recorded closes the channel
        SnapshotSimulator slow = new SnapshotSimulator(new int[]{40, 60, 20}, new int[][]{{0, 1}, {1, 2}, {2, 1}});
        List<int[]> slowScript = new ArrayList<>();
        slowScript.add(new int[]{SnapshotSimulator.TRANSFER, 2, 1, 9});
        slowScript.add(new int[]{SnapshotSimulator.INITIATE, 0, 0, 0});
        slowScript.add(new int[]{SnapshotSimulator.DELIVER, 0, 1});
        slow.runScript(slowScript);
        verifySnapshot(slow, "late marker");
        check(slow.channels.get(2).transit.equals(Arrays.asList(9)),
                "the transfer 2->1 is recorded as in transit");

        // an uncoordinated snapshot at different times does not preserve the total
        int breaks = 0;
        for (int delta = -3; delta <= 3; delta++) {
            if (delta == 0) continue;
            int[] recordedAt = new int[first.processes.length];
            for (int i = 0; i < recordedAt.length; i++) {
                recordedAt[i] = Math.max(1, first.processes[i].recordedAtEvent + delta * (i + 1));
            }
            int sum = 0;
            for (int value : replayCut(first.transferLog, first, recordedAt,
                    new int[]{200, 150, 100, 50})) {
                sum += value;
            }
            if (sum != first.totalTransferred) breaks++;
        }
        check(breaks > 0, "an uncoordinated cut loses or duplicates money");
        System.out.println("uncoordinated cuts that break the invariant: " + breaks + " of 6");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
