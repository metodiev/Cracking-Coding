/** Task 852 - Implement global snapshotting for distributed streaming pipeline (Chandy-Lamport at scale).
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A pipeline of N processes connected by a strongly connected, FIFO, directed channel set. Application
 *   events are: a local event (bumps a per-process counter), and a money transfer, which deducts the amount
 *   from the sender's recorded state at send time and enqueues a DATA message that credits the receiver on
 *   delivery. The Chandy-Lamport rule is implemented exactly:
 *
 *       initiate(p): record p's state, then send one MARKER on every out-channel of p;
 *       on MARKER over c:  if the state is not recorded yet, record it and send markers on every out-channel,
 *                          then close channel c;
 *       on DATA over c:    if the state is already recorded and c is still open, append the message to c's
 *                          recorded channel state before applying it.
 *
 *   Messages travel over explicit in-memory FIFO queues and a deterministic round-robin delivery loop picks
 *   which channel advances next, so any interleaving can be replayed bit-for-bit. All randomness comes from a
 *   fixed-seed splitmix64 stream.
 *
 * WHAT IS SIMPLIFIED
 *   No real processes, threads, sockets or clocks: the "network" is a set of queues plus a scheduler, and
 *   time is the number of deliveries. Message loss, reordering, re-transmission and failure injection are out
 *   of scope; the model keeps the property that makes Chandy-Lamport non-trivial, namely arbitrary
 *   interleaving of local events, sends and receives across the whole system.
 *
 * INVARIANTS VERIFIED (safety, liveness, cost and application consistency)
 *   Safety: the recorded cut is consistent (no receive of a DATA message lies in a recorded state unless its
 *   send also does - checked against the full event log, i.e. against the definition rather than the code);
 *   each recorded channel state equals exactly the set of messages sent before the upstream state was
 *   recorded and received after the downstream state was recorded, and the recorded process states equal an
 *   independent prefix replay of the event log, so no state is torn.
 *   Liveness: the snapshot completes at every process and on every channel, using no more deliveries than
 *   (data in flight at initiation) + (data sent afterwards) + |E| markers, with exactly one control marker
 *   per channel - the Chandy-Lamport overhead bound.
 *   Application consistency (the transactional invariant): money is conserved by the snapshot,
 *   sum(recorded states) + sum(recorded channel states) == the initial total, and it stays conserved with two
 *   concurrent initiators and under repeated snapshots whose cuts are monotone. A deliberately broken variant
 *   that drops in-transit messages is rejected by these same checks, and a doctored cut is flagged by the
 *   consistency checker, so the verifiers have teeth.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every interleaving is reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A network message: DATA carries money, MARKER is the Chandy-Lamport control message. */
final class Msg {
    final int id;
    final int from;
    final int to;
    final long amount;
    final boolean marker;
    int sendIndex = -1;
    int recvIndex = -1;

    Msg(int id, int from, int to, long amount, boolean marker) {
        this.id = id;
        this.from = from;
        this.to = to;
        this.amount = amount;
        this.marker = marker;
    }
}

/** FIFO channel with an explicit in-flight queue. */
final class Channel {
    final int id;
    final int from;
    final int to;
    final ArrayDeque<Msg> queue = new ArrayDeque<>();

    Channel(int id, int from, int to) {
        this.id = id;
        this.from = from;
        this.to = to;
    }
}

/** One application event of one process, with its per-process index used by the verification oracles. */
final class Ev {
    static final int LOCAL = 0, SEND = 1, RECV = 2;
    final int proc;
    final int kind;
    final int channel;
    final int index;
    final long moneyDelta;
    final int counterDelta;

    Ev(int proc, int kind, int channel, int index, long moneyDelta, int counterDelta) {
        this.proc = proc;
        this.kind = kind;
        this.channel = channel;
        this.index = index;
        this.moneyDelta = moneyDelta;
        this.counterDelta = counterDelta;
    }
}

/** Bookkeeping of one snapshot round: what each process recorded, and on which channel states. */
final class SnapshotRound {
    final int processes;
    final int[] recordedAt;
    final long[] recordedMoney;
    final int[] recordedCounter;
    final List<List<Integer>> channelMessages;
    final boolean[] channelDone;
    int markersSent;
    int initiatorCount;
    boolean complete;
    int completionStep = -1;

    SnapshotRound(int processes, int channels) {
        this.processes = processes;
        this.recordedAt = new int[processes];
        this.recordedMoney = new long[processes];
        this.recordedCounter = new int[processes];
        this.channelMessages = new ArrayList<>();
        for (int i = 0; i < channels; i++) channelMessages.add(new ArrayList<>());
        this.channelDone = new boolean[channels];
        Arrays.fill(this.recordedAt, -1);
    }

    boolean isRecorded(int p) { return recordedAt[p] >= 0; }

    boolean allRecorded() {
        for (int p = 0; p < processes; p++) if (recordedAt[p] < 0) return false;
        return true;
    }

    boolean allChannelsClosed() {
        for (boolean b : channelDone) if (!b) return false;
        return true;
    }
}

/** A process: application state plus the pointer to the snapshot round currently in flight. */
final class Proc {
    final int id;
    long money;
    int counter;
    int appEvents;
    final int[] outChannels;
    final int[] inChannels;
    SnapshotRound round;

    Proc(int id, int[] inChannels, int[] outChannels) {
        this.id = id;
        this.inChannels = inChannels;
        this.outChannels = outChannels;
    }
}

/** The in-memory network plus the deterministic scheduler and the Chandy-Lamport instrumentation. */
final class Cluster {
    final int n;
    final List<Proc> procs = new ArrayList<>();
    final List<Channel> channels = new ArrayList<>();
    final List<Ev> events = new ArrayList<>();
    final List<Msg> messages = new ArrayList<>();
    final List<SnapshotRound> rounds = new ArrayList<>();
    final long initialMoney;
    final boolean recordInTransit;
    final SplitMix64 rng;
    SnapshotRound round;
    int cursor;
    long controlMarkers;
    int dataSent;
    int deliveries;

    Cluster(int n, long seed, boolean recordInTransit, long moneyPerProcess) {
        this.n = n;
        this.recordInTransit = recordInTransit;
        this.rng = new SplitMix64(seed);
        this.initialMoney = moneyPerProcess;
        List<List<Integer>> in = new ArrayList<>();
        List<List<Integer>> out = new ArrayList<>();
        for (int i = 0; i < n; i++) { in.add(new ArrayList<>()); out.add(new ArrayList<>()); }
        // strongly connected topology: a ring plus a chord, so every process is reachable from every other.
        if (n > 1) {
            for (int i = 0; i < n; i++) addChannel(i, (i + 1) % n, in, out);
        }
        for (int i = 0; i < n; i++) if (n > 2) addChannel(i, (i + 2) % n, in, out);
        for (int i = 0; i < n; i++) {
            procs.add(new Proc(i, toArray(in.get(i)), toArray(out.get(i))));
            procs.get(i).money = moneyPerProcess;
        }
    }

    static int[] toArray(List<Integer> list) {
        int[] a = new int[list.size()];
        for (int i = 0; i < a.length; i++) a[i] = list.get(i);
        return a;
    }

    private void addChannel(int from, int to, List<List<Integer>> in, List<List<Integer>> out) {
        int id = channels.size();
        channels.add(new Channel(id, from, to));
        out.get(from).add(id);
        in.get(to).add(id);
    }

    // ------------------------------------------------------------------ application events

    void localEvent(int p) {
        Proc proc = procs.get(p);
        events.add(new Ev(p, Ev.LOCAL, -1, proc.appEvents, 0L, 1));
        proc.appEvents++;
        proc.counter++;
    }

    boolean sendData(int p, int channel, long amount) {
        Proc proc = procs.get(p);
        if (proc.money < amount) return false;
        Channel c = channels.get(channel);
        Msg m = new Msg(messages.size(), p, c.to, amount, false);
        m.sendIndex = proc.appEvents;
        messages.add(m);
        c.queue.addLast(m);
        events.add(new Ev(p, Ev.SEND, channel, proc.appEvents, -amount, 0));
        proc.appEvents++;
        proc.money -= amount;
        dataSent++;
        return true;
    }

    // ------------------------------------------------------------------ deliveries

    int deliverOne() {
        int size = channels.size();
        for (int offset = 0; offset < size; offset++) {
            int id = (cursor + offset) % size;
            if (!channels.get(id).queue.isEmpty()) {
                cursor = id + 1;
                deliver(id);
                return id;
            }
        }
        return -1;
    }

    /** One delivery advance, counted so liveness bounds can be checked. */
    void deliverStep() {
        if (deliverOne() >= 0) deliveries++;
    }

    void drain() {
        while (deliverOne() >= 0) deliveries++;
    }

    private void deliver(int channelId) {
        Channel c = channels.get(channelId);
        Msg m = c.queue.pollFirst();
        if (m.marker) {
            handleMarker(c, m);
            controlMarkers++;
            return;
        }
        Proc q = procs.get(m.to);
        m.recvIndex = q.appEvents;
        if (recordInTransit && q.round != null && q.round.isRecorded(q.id) && !q.round.channelDone[channelId]) {
            q.round.channelMessages.get(channelId).add(m.id);
        }
        events.add(new Ev(q.id, Ev.RECV, channelId, q.appEvents, m.amount, 0));
        q.appEvents++;
        q.money += m.amount;
    }

    private void handleMarker(Channel c, Msg m) {
        Proc q = procs.get(m.to);
        if (q.round == null || q.round.channelDone[c.id]) return;
        if (!q.round.isRecorded(q.id)) {
            recordState(q);
            sendMarkers(q);
        }
        q.round.channelDone[c.id] = true;
        maybeComplete(q.round);
    }

    // ------------------------------------------------------------------ Chandy-Lamport rules

    void recordState(Proc p) {
        p.round.recordedAt[p.id] = p.appEvents;
        p.round.recordedMoney[p.id] = p.money;
        p.round.recordedCounter[p.id] = p.counter;
    }

    void sendMarkers(Proc p) {
        for (int channel : p.outChannels) {
            Channel c = channels.get(channel);
            c.queue.addLast(new Msg(-1, p.id, c.to, 0L, true));
            p.round.markersSent++;
        }
    }

    SnapshotRound initiate(int[] initiators) {
        SnapshotRound r = new SnapshotRound(n, channels.size());
        r.initiatorCount = initiators.length;
        round = r;
        for (Proc p : procs) p.round = r;
        for (int id : initiators) {
            Proc p = procs.get(id);
            if (!r.isRecorded(id)) {
                recordState(p);
                sendMarkers(p);
            }
        }
        rounds.add(r);
        maybeComplete(r);
        return r;
    }

    private void maybeComplete(SnapshotRound r) {
        if (!r.complete && r.allRecorded() && r.allChannelsClosed()) {
            r.complete = true;
            r.completionStep = deliveries;
        }
    }

    // ------------------------------------------------------------------ scheduler

    /** One application step: either a local event or a transfer, chosen deterministically. */
    void appStep() {
        int p = rng.nextInt(n);
        int choice = rng.nextInt(100);
        Proc proc = procs.get(p);
        if (choice < 35 || proc.outChannels.length == 0) {
            localEvent(p);
        } else {
            int channel = proc.outChannels[rng.nextInt(proc.outChannels.length)];
            long amount = 1 + rng.nextInt(5);
            if (!sendData(p, channel, amount)) localEvent(p);
        }
        // deliveries lag sends, so a backlog of in-flight messages builds up across the pipeline
        if (rng.nextInt(100) < 30) deliverStep();
    }

    /** Data messages still in flight, used for the liveness bound. */
    int inFlightData() {
        int count = 0;
        for (Channel c : channels) for (Msg m : c.queue) if (!m.marker) count++;
        return count;
    }

    int markersInFlight() {
        int count = 0;
        for (Channel c : channels) for (Msg m : c.queue) if (m.marker) count++;
        return count;
    }

    long snapshotTotal(SnapshotRound r) {
        long total = 0;
        for (int p = 0; p < n; p++) if (r.isRecorded(p)) total += r.recordedMoney[p];
        for (int c = 0; c < channels.size(); c++) {
            for (int id : r.channelMessages.get(c)) total += messages.get(id).amount;
        }
        return total;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ verification oracles

    /** Independent replay: state of p built from the log's first k[p] events only. */
    static long[] replayMoney(Cluster cluster, int[] k) {
        long[] money = new long[cluster.n];
        int[] counter = new int[cluster.n];
        Arrays.fill(money, cluster.initialMoney);
        for (Ev e : cluster.events) {
            if (e.index < k[e.proc]) {
                money[e.proc] += e.moneyDelta;
                counter[e.proc] += e.counterDelta;
            }
        }
        long[] packed = new long[2 * cluster.n];
        for (int p = 0; p < cluster.n; p++) { packed[p] = money[p]; packed[cluster.n + p] = counter[p]; }
        return packed;
    }

    /**
     * Definitional consistency test of a cut: a DATA message may only appear inside a recorded process state
     * if its send event is inside the cut too. Returns the number of violating messages.
     */
    static int cutViolations(Cluster cluster, int[] k) {
        int violations = 0;
        for (Msg m : cluster.messages) {
            if (m.recvIndex < 0 || m.sendIndex < 0) continue;
            if (k[m.to] >= 0 && m.recvIndex < k[m.to]) {
                if (k[m.from] < 0 || m.sendIndex >= k[m.from]) violations++;
            }
        }
        return violations;
    }

    /** Exact channel-state oracle: sent before the upstream recording AND received after the downstream one. */
    static Set<Integer> expectedChannelState(Cluster cluster, SnapshotRound r, Channel c) {
        Set<Integer> expected = new HashSet<>();
        if (!r.isRecorded(c.from) || !r.isRecorded(c.to)) return expected;
        int kFrom = r.recordedAt[c.from];
        int kTo = r.recordedAt[c.to];
        for (Msg m : cluster.messages) {
            if (m.from != c.from || m.to != c.to) continue;
            if (m.sendIndex < kFrom && m.recvIndex >= kTo) expected.add(m.id);
        }
        return expected;
    }

    static Set<Integer> recordedChannelState(SnapshotRound r, int channel) {
        return new HashSet<>(r.channelMessages.get(channel));
    }

    /** Runs the scripted mid-run scenario (application events interleaved with deliveries). */
    static Cluster midRun(int n, long seed, boolean recordInTransit, int appSteps, int[] initiators,
            int appStepsAfter) {
        Cluster cluster = new Cluster(n, seed, recordInTransit, 100L);
        for (int i = 0; i < appSteps; i++) cluster.appStep();
        cluster.initiate(initiators);
        for (int i = 0; i < appStepsAfter; i++) cluster.appStep();
        cluster.drain();
        return cluster;
    }

    /** All safety/consistency checks for one snapshot round of one cluster; returns the money total. */
    static long auditRound(Cluster cluster, SnapshotRound r, String label) {
        check(r.complete, label + ": snapshot completes on every process and channel");
        check(r.allRecorded(), label + ": every process recorded a state");
        check(r.allChannelsClosed(), label + ": every channel state was closed");
        check(r.markersSent == cluster.channels.size(),
                label + ": exactly one marker per channel, got " + r.markersSent);

        // recorded process states == independent prefix replay.
        long[] replayed = replayMoney(cluster, r.recordedAt);
        for (int p = 0; p < cluster.n; p++) {
            check(replayed[p] == r.recordedMoney[p],
                    label + ": recorded money of p" + p + " equals the prefix replay");
            check(replayed[cluster.n + p] == r.recordedCounter[p],
                    label + ": recorded counter of p" + p + " equals the prefix replay");
        }

        // consistent cut, and exact channel states.
        check(cutViolations(cluster, r.recordedAt) == 0, label + ": recorded cut is consistent");
        Set<Integer> inChannelStates = new HashSet<>();
        for (Channel c : cluster.channels) {
            Set<Integer> expected = expectedChannelState(cluster, r, c);
            Set<Integer> recorded = recordedChannelState(r, c.id);
            check(expected.equals(recorded), label + ": channel " + c.id + " state matches the oracle "
                    + "(expected " + expected.size() + ", recorded " + recorded.size() + ")");
            for (int id : recorded) check(inChannelStates.add(id), label + ": message in one channel only");
        }
        // a message counted in a channel state must not also be part of a recorded process state.
        for (Msg m : cluster.messages) {
            if (inChannelStates.contains(m.id)) {
                check(m.recvIndex >= r.recordedAt[m.to],
                        label + ": in-transit message " + m.id + " not consumed by a recorded state");
                check(m.sendIndex < r.recordedAt[m.from],
                        label + ": in-transit message " + m.id + " was already sent in the cut");
            }
        }
        // transactional invariant: no money created or destroyed by the snapshot.
        long total = cluster.snapshotTotal(r);
        check(total == cluster.initialMoney * cluster.n,
                label + ": money conserved by the snapshot (" + total + " != "
                        + (cluster.initialMoney * cluster.n) + ")");
        for (Proc p : cluster.procs) check(r.recordedMoney[p.id] >= 0, label + ": no negative recorded balance");
        return total;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final int n = 5;
        final long seed = 852_2024L;

        // --- 1. well-formed network: strongly connected, no self loops, FIFO queues per channel.
        Cluster topology = new Cluster(n, seed, true, 100L);
        check(topology.channels.size() == 2 * n, "ring plus chord gives 2N channels");
        for (Channel c : topology.channels) {
            check(c.from != c.to, "channel " + c.id + " is not a self loop");
            check(c.queue.isEmpty(), "channel " + c.id + " starts empty");
        }
        for (Proc p : topology.procs) {
            check(p.inChannels.length == p.outChannels.length && p.inChannels.length == 2,
                    "each process has two in- and out-channels");
        }

        // --- 2. single snapshot in the middle of a running pipeline.
        int snapshotStep = 120;
        Cluster mid = new Cluster(n, seed, true, 100L);
        for (int i = 0; i < snapshotStep; i++) mid.appStep();
        System.out.printf("pre-snapshot: messages=%d in-flight=%d deliveries=%d%n",
                mid.messages.size(), mid.inFlightData(), mid.deliveries);
        check(mid.inFlightData() > 0, "the pipeline has messages in flight before the snapshot");
        int beforeInitiation = mid.deliveries;
        int inFlight = mid.inFlightData();
        int dataBefore = mid.dataSent;
        SnapshotRound r1 = mid.initiate(new int[]{0});
        for (int i = 0; i < 60; i++) mid.appStep();
        mid.drain();
        int dataAfter = mid.dataSent - dataBefore;
        int deliveriesToComplete = r1.completionStep - beforeInitiation;
        System.out.printf("mid-run snapshot: in-flight=%d, data sent after initiation=%d, deliveries to complete=%d%n",
                inFlight, dataAfter, deliveriesToComplete);
        check(deliveriesToComplete <= inFlight + dataAfter + mid.channels.size(),
                "snapshot completes within the Chandy-Lamport message bound");
        check(deliveriesToComplete >= 0, "completion step is recorded");
        auditRound(mid, r1, "mid-run");
        check(mid.messages.size() > 20, "the workload actually transferred money (" + mid.messages.size() + ")");
        check(mid.controlMarkers == mid.channels.size(),
                "exactly |E| control markers traversed the network, got " + mid.controlMarkers);
        check(mid.inFlightData() == 0 && mid.markersInFlight() == 0, "network quiesces after the drain");
        int totalEvents = 0;
        for (Proc p : mid.procs) totalEvents += p.appEvents;
        int recordedEvents = 0;
        for (int p = 0; p < n; p++) recordedEvents += r1.recordedAt[p];
        check(recordedEvents > 0 && recordedEvents < totalEvents,
                "the cut is proper: neither empty nor the whole history ("
                        + recordedEvents + " of " + totalEvents + " events)");
        check(mid.messages.size() > 0 && r1.channelMessages.stream().anyMatch(l -> !l.isEmpty()),
                "at least one in-transit message crossed the snapshot");

        // --- 3. two concurrent initiators: the cut stays consistent and the overhead bound holds.
        Cluster concurrent = midRun(n, seed + 1, true, 90, new int[]{1, 2}, 40);
        SnapshotRound r2 = concurrent.rounds.get(0);
        check(r2.initiatorCount == 2, "two initiators recorded");
        auditRound(concurrent, r2, "concurrent initiators");
        check(concurrent.controlMarkers == concurrent.channels.size(),
                "concurrent initiators still send exactly one marker per channel");

        // --- 4. repeated snapshots: the second cut dominates the first.
        Cluster repeated = new Cluster(n, seed + 2, true, 100L);
        for (int i = 0; i < 80; i++) repeated.appStep();
        SnapshotRound first = repeated.initiate(new int[]{0});
        repeated.drain();
        auditRound(repeated, first, "first snapshot");
        for (int i = 0; i < 80; i++) repeated.appStep();
        SnapshotRound second = repeated.initiate(new int[]{0});
        repeated.drain();
        auditRound(repeated, second, "second snapshot");
        for (int p = 0; p < n; p++) {
            check(second.recordedAt[p] >= first.recordedAt[p], "cut of the later snapshot dominates p" + p);
        }
        check(second.recordedAt[0] > first.recordedAt[0], "the initiator reached a strictly later cut");
        check(repeated.controlMarkers == 2 * repeated.channels.size(),
                "two sequential snapshots cost two markers per channel");

        // --- 5. canonical crossing message: p0 -> p1 with p1 initiating while the transfer is in flight.
        Cluster canonical = new Cluster(2, seed + 3, true, 100L);
        int ch01 = -1, ch10 = -1;
        for (Channel c : canonical.channels) {
            if (c.from == 0 && c.to == 1) ch01 = c.id;
            if (c.from == 1 && c.to == 0) ch10 = c.id;
        }
        check(ch01 >= 0 && ch10 >= 0, "canonical scenario has channels in both directions");
        check(canonical.sendData(0, ch01, 7L), "transfer of 7 is accepted");
        SnapshotRound rCanon = canonical.initiate(new int[]{1});
        canonical.drain();
        auditRound(canonical, rCanon, "crossing message");
        check(canonical.rounds.size() == 1, "canonical scenario has exactly one snapshot");
        check(rCanon.channelMessages.get(ch01).size() == 1
                        && canonical.messages.get(rCanon.channelMessages.get(ch01).get(0)).amount == 7L,
                "the crossing 7-unit message is captured in the p0->p1 channel state");
        check(canonical.snapshotTotal(rCanon) == 200L, "crossing message keeps the money invariant");
        check(rCanon.recordedMoney[1] == 100L, "receiver's recorded state excludes the in-flight credit");

        // --- 6. the verifiers have teeth: a variant that drops in-transit messages is rejected.
        Cluster brokenCanonical = new Cluster(2, seed + 3, false, 100L);
        check(brokenCanonical.sendData(0, ch01, 7L), "broken scenario sends the same 7-unit transfer");
        SnapshotRound brokenRound = brokenCanonical.initiate(new int[]{1});
        brokenCanonical.drain();
        check(expectedChannelState(brokenCanonical, brokenRound,
                        brokenCanonical.channels.get(ch01)).size() == 1,
                "the oracle still expects the crossing message");
        check(recordedChannelState(brokenRound, ch01).isEmpty(),
                "the broken variant fails to record the in-transit message");
        long brokenTotal = brokenCanonical.snapshotTotal(brokenRound);
        check(brokenTotal == 200L - 7L, "the broken variant loses the in-flight 7 units: " + brokenTotal);
        Cluster brokenRandom = midRun(n, seed, false, snapshotStep, new int[]{0}, 60);
        int brokenRecorded = 0;
        for (int c = 0; c < brokenRandom.channels.size(); c++) {
            brokenRecorded += recordedChannelState(brokenRandom.rounds.get(0), c).size();
        }
        check(brokenRecorded == 0, "the broken variant never records a channel state");
        int correctRecorded = 0;
        for (int c = 0; c < mid.channels.size(); c++) {
            correctRecorded += recordedChannelState(r1, c).size();
        }
        check(correctRecorded > 0, "the correct algorithm records in-transit messages (" + correctRecorded + ")");
        check(brokenRandom.snapshotTotal(brokenRandom.rounds.get(0)) < brokenRandom.initialMoney * n,
                "the broken variant loses money on a random schedule too");

        // --- 7. the cut checker itself is validated by tampering with a correct cut.
        Msg crossing = null;
        for (Msg m : mid.messages) {
            if (m.recvIndex >= r1.recordedAt[m.to] && m.sendIndex < r1.recordedAt[m.from]) crossing = m;
        }
        check(crossing != null, "the mid-run snapshot contains an in-transit message");
        check(cutViolations(mid, r1.recordedAt) == 0, "the genuine cut is consistent");
        int[] tampered = r1.recordedAt.clone();
        tampered[crossing.to] = crossing.recvIndex + 1;   // pull the receive into the cut ...
        tampered[crossing.from] = crossing.sendIndex;     // ... while pushing its send out of it
        check(cutViolations(mid, tampered) >= 1, "the checker flags the doctored cut");
        check(cutViolations(mid, tampered) > cutViolations(mid, r1.recordedAt),
                "the doctored cut is strictly worse than the genuine one");

        // --- 8. determinism, different interleavings and degenerate topologies.
        Cluster again = new Cluster(n, seed, true, 100L);
        for (int i = 0; i < snapshotStep; i++) again.appStep();
        SnapshotRound againRound = again.initiate(new int[]{0});
        for (int i = 0; i < 60; i++) again.appStep();
        again.drain();
        boolean same = againRound.markersSent == r1.markersSent && again.messages.size() == mid.messages.size();
        for (int p = 0; p < n; p++) same &= againRound.recordedAt[p] == r1.recordedAt[p]
                && againRound.recordedMoney[p] == r1.recordedMoney[p];
        check(same, "identical seeds reproduce the snapshot exactly");
        for (int c = 0; c < mid.channels.size(); c++) {
            check(recordedChannelState(againRound, c).equals(recordedChannelState(r1, c)),
                    "channel states are reproduced for channel " + c);
        }

        Cluster other = new Cluster(n, seed + 99, true, 100L);
        for (int i = 0; i < snapshotStep; i++) other.appStep();
        SnapshotRound otherRound = other.initiate(new int[]{0});
        other.drain();
        auditRound(other, otherRound, "different schedule");
        boolean differs = false;
        for (int p = 0; p < n; p++) differs |= otherRound.recordedAt[p] != r1.recordedAt[p];
        check(differs, "a different interleaving yields a different (still consistent) cut");

        Cluster single = new Cluster(1, seed, true, 100L);
        check(single.channels.isEmpty(), "a single process has no channels");
        for (int i = 0; i < 5; i++) single.localEvent(0);
        SnapshotRound singleRound = single.initiate(new int[]{0});
        single.drain();
        check(singleRound.complete && singleRound.markersSent == 0,
                "single-process snapshot completes with no markers");
        check(singleRound.recordedCounter[0] == 5 && singleRound.recordedMoney[0] == 100L,
                "single-process state is captured exactly");
        check(single.snapshotTotal(singleRound) == 100L, "single-process money invariant");

        System.out.println("rounds audited: " + (mid.rounds.size() + concurrent.rounds.size()
                + repeated.rounds.size() + canonical.rounds.size() + other.rounds.size()
                + single.rounds.size()));
        System.out.println("checks=" + checks);
        System.out.println("PASS");
    }
}
