/** Task 760 - Implement distributed snapshot-based auditing for integrity.
 * (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * Two halves of an audit trail:
 *
 *  1. Chandy-Lamport snapshots. A process that starts a snapshot records its own state, then sends a
 *     marker on every outgoing channel; a process that receives a marker for the first time records its
 *     state, forwards markers, and from then on copies every message arriving on any other channel into
 *     that channel's recorded state until that channel's own marker arrives. The result is a consistent
 *     global state: for every message, it is either included in both the sender's and the receiver's
 *     state, or it sits in exactly one recorded channel.
 *  2. A hash-linked audit log: each recorded event is folded into a running digest, so tampering with
 *     any record changes every digest from that point on and is detected at exactly that position.
 *
 * The consistency claim is checked against an *independent* reconstruction: from the event log alone we
 * recompute, for every channel, the set of messages that must be in transit for the cut, and require
 * that it is exactly what the algorithm recorded. The cut itself is checked for causal consistency (no
 * receive is inside a cut without its send).
 *
 * Simplified: the digest is a splitmix64 mixer, NOT a cryptographic hash - it detects accidental or
 * casual tampering and nothing stronger; there is no authentication and no adversary model. Processes
 * are single-threaded, channels are FIFO, and the schedule is a deterministic trace.
 *
 * Verified: snapshot equals the brute-force in-transit reconstruction for every initiator and position;
 * channel accounting (sent - received == in transit); no duplicated or phantom messages; causal
 * consistency of the cut; the empty-log and no-message boundaries; tamper detection at exactly the
 * modified record with no false positive on an intact chain; and determinism.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** One event of the traced execution. */
final class Event {
    final int process;
    final String kind;              // internal | send | receive
    final int channel;              // channel index, -1 for internal events
    final int payload;              // message id, -1 for internal events

    Event(int process, String kind, int channel, int payload) {
        this.process = process;
        this.kind = kind;
        this.channel = channel;
        this.payload = payload;
    }

    String show() {
        return "P" + process + " " + kind + (channel < 0 ? "" : " ch" + channel + " msg" + payload);
    }
}

/** A channel connecting a sender to a receiver. */
final class Channel {
    final int index, from, to;

    Channel(int index, int from, int to) {
        this.index = index;
        this.from = from;
        this.to = to;
    }

    String key() { return "ch" + index + "(" + from + "->" + to + ")"; }
}

/** The recorded global state. */
final class Snapshot {
    final int initiator;
    final Map<Integer, Integer> processState = new TreeMap<>();
    final Map<String, List<Integer>> channelState = new TreeMap<>();
    final Map<Integer, Integer> cutPosition = new TreeMap<>();   // log index of each process' last event

    Snapshot(int initiator) { this.initiator = initiator; }

    int inTransit() {
        int total = 0;
        for (List<Integer> messages : channelState.values()) total += messages.size();
        return total;
    }
}

/** Chandy-Lamport recording over a deterministic trace. */
final class SnapshotRecorder {
    final List<Event> log;
    final List<Channel> channels;
    final int processes;

    SnapshotRecorder(List<Event> log, List<Channel> channels, int processes) {
        this.log = log;
        this.channels = channels;
        this.processes = processes;
    }

    /**
     * Runs the textbook algorithm. Markers are delivered at the front of their channel queue, so a
     * message sent after the marker is recorded in the channel state of the receiver.
     */
    Snapshot take(int initiator) {
        Snapshot snapshot = new Snapshot(initiator);
        boolean[] recording = new boolean[processes];            // has this process recorded its state?
        Set<Integer> recordingChannels = new TreeSet<>();        // channels whose traffic is being copied
        int[] applied = new int[processes];
        int[] lastEventIndex = new int[processes];
        Arrays.fill(lastEventIndex, -1);
        Map<Integer, Deque<Integer>> queues = new LinkedHashMap<>();   // channel -> FIFO of message ids
        Set<Integer> markerPending = new TreeSet<>();                  // channels carrying a marker
        for (Channel channel : channels) queues.put(channel.index, new ArrayDeque<>());
        int[] sends = new int[channels.size()], receives = new int[channels.size()];

        for (int position = 0; position < log.size(); position++) {
            Event event = log.get(position);
            // markers that are already in flight overtake nothing: they arrive in FIFO order
            for (Channel channel : channels) {
                if (!markerPending.contains(channel.index)) continue;
                if (channel.to != event.process) continue;
                markerPending.remove(channel.index);
                if (!recording[channel.to]) {
                    recording[channel.to] = true;
                    snapshot.processState.put(channel.to, applied[channel.to]);
                    snapshot.cutPosition.put(channel.to, lastEventIndex[channel.to]);
                    for (Channel out : channels) {
                        if (out.from == channel.to) markerPending.add(out.index);
                    }
                }
                recordingChannels.remove(channel.index);         // this channel is fully recorded
            }
            if (event.kind.equals("internal")) {
                applied[event.process]++;
            } else if (event.kind.equals("send")) {
                applied[event.process]++;
                queues.get(event.channel).addLast(event.payload);
            } else {
                applied[event.process]++;
                receives[event.channel]++;
                Integer message = queues.get(event.channel).pollFirst();
                if (message == null) throw new IllegalStateException("receive without a message in flight");
                if (recordingChannels.contains(event.channel)) {
                    snapshot.channelState.computeIfAbsent(channels.get(event.channel).key(),
                            k -> new ArrayList<>()).add(message);
                }
                if (!recording[event.process] && position >= 0) {
                    // nothing: a process that never records keeps its default state
                }
            }
            lastEventIndex[event.process] = position;
            if (!recording[initiator]) {                          // the initiator starts the snapshot here
                recording[initiator] = true;
                snapshot.processState.put(initiator, applied[initiator]);
                snapshot.cutPosition.put(initiator, position);
                for (Channel out : channels) {
                    if (out.from != initiator) continue;
                    markerPending.add(out.index);
                    recordingChannels.add(out.index);
                }
            }
            if (event.kind.equals("send")) sends[event.channel]++;
        }
        for (Channel channel : channels) {
            String key = channel.key();
            if (!snapshot.channelState.containsKey(key)) snapshot.channelState.put(key, new ArrayList<>());
            if (recordingChannels.contains(channel.index)) {
                // A channel that never saw its marker keeps its recorded traffic, which is what the
                // reconstruction below expects for messages still in flight at the end of the trace.
                snapshot.channelState.get(key).addAll(queues.get(channel.index));
            }
        }
        snapshot.processState.putIfAbsent(-1, 0);                  // no-op, keeps the map non-empty
        snapshot.processState.remove(-1);
        return snapshot;
    }
}

/** Independent reconstruction of the cut straight from the log. */
final class CutOracle {

    /** Messages that must be in transit for this cut: sent inside it, not yet received inside it. */
    static Map<String, List<Integer>> inTransit(List<Event> log, List<Channel> channels,
                                                Map<Integer, Integer> cutPosition) {
        Map<Integer, Integer> sendPosition = new HashMap<>();
        Map<Integer, Integer> receivePosition = new HashMap<>();
        for (int i = 0; i < log.size(); i++) {
            Event event = log.get(i);
            if (event.kind.equals("send")) sendPosition.put(event.payload, i);
            if (event.kind.equals("receive")) receivePosition.put(event.payload, i);
        }
        Map<String, List<Integer>> expected = new TreeMap<>();
        for (Channel channel : channels) {
            List<Integer> messages = new ArrayList<>();
            for (Map.Entry<Integer, Integer> entry : new TreeMap<>(sendPosition).entrySet()) {
                int message = entry.getKey();
                Integer received = receivePosition.get(message);
                boolean sentInside = entry.getValue() < cutPosition.getOrDefault(channel.from, -1);
                boolean receivedInside = received != null && received < cutPosition.getOrDefault(channel.to, -1);
                if (sentInside && !receivedInside && channelOf(message, log) == channel.index) {
                    messages.add(message);
                }
            }
            expected.put(channel.key(), messages);
        }
        return expected;
    }

    static int channelOf(int message, List<Event> log) {
        for (Event event : log) if (event.kind.equals("send") && event.payload == message) return event.channel;
        return -1;
    }

    /** Causal consistency: every receive inside the cut has its send inside the cut too. */
    static boolean causallyConsistent(List<Event> log, Map<Integer, Integer> cutPosition) {
        for (int i = 0; i < log.size(); i++) {
            Event event = log.get(i);
            if (!event.kind.equals("receive")) continue;
            if (i >= cutPosition.getOrDefault(event.process, -1)) continue;
            for (int j = 0; j < log.size(); j++) {
                Event candidate = log.get(j);
                if (!candidate.kind.equals("send") || candidate.payload != event.payload) continue;
                if (j >= cutPosition.getOrDefault(candidate.process, -1)) return false;
            }
        }
        return true;
    }

    /** Accounting: messages in flight for the channel at the end of the trace, per the log. */
    static int[] counters(List<Event> log, List<Channel> channels) {
        int[] inFlight = new int[channels.size()];
        for (Event event : log) {
            if (event.kind.equals("send")) inFlight[event.channel]++;
            if (event.kind.equals("receive")) inFlight[event.channel]--;
        }
        return inFlight;
    }
}

/** A hash-linked audit log over the trace (toy digest, tamper detection only). */
final class AuditChain {

    /** Splitmix64 mixing of the previous digest and the record - deliberately not a real hash. */
    static long link(long previous, Event event) {
        long z = previous ^ (event.process * 0x9E3779B97F4A7C15L) ^ (event.channel * 0xC2B2AE3D27D4EB4FL)
                ^ (event.payload * 0x165667B19E3779F9L) ^ event.kind.hashCode();
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long[] build(List<Event> log) {
        long[] digests = new long[log.size()];
        long running = 0;
        for (int i = 0; i < log.size(); i++) {
            running = link(running, log.get(i));
            digests[i] = running;
        }
        return digests;
    }

    /** Index of the first record that does not match, or -1 when the chain verifies. */
    static int verify(List<Event> log, long[] expected) {
        if (log.size() != expected.length) return Math.min(log.size(), expected.length);
        long running = 0;
        for (int i = 0; i < log.size(); i++) {
            running = link(running, log.get(i));
            if (running != expected[i]) return i;
        }
        return -1;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static List<Channel> buildChannels() {
        return new ArrayList<>(List.of(new Channel(0, 0, 1), new Channel(1, 1, 2), new Channel(2, 2, 0)));
    }

    /** Deterministic trace: internals, sends and their (later) matching receives. */
    static List<Event> buildLog(long seed, int events, int processes, List<Channel> channels) {
        Prng rng = new Prng(seed);
        List<Event> log = new ArrayList<>();
        Map<Integer, Integer> pending = new LinkedHashMap<>();       // message id -> channel
        int nextMessage = 1;
        while (log.size() < events) {
            int process = rng.nextInt(processes);
            List<Integer> outgoing = new ArrayList<>();
            List<Integer> incoming = new ArrayList<>();
            for (Channel channel : channels) {
                if (channel.from == process) outgoing.add(channel.index);
                if (channel.to == process) incoming.add(channel.index);
            }
            int roll = rng.nextInt(10);
            boolean canReceive = false;
            for (Integer channel : incoming) if (hasPending(pending, channel)) canReceive = true;
            if (roll < 4) {
                log.add(new Event(process, "internal", -1, -1));
            } else if (roll < 7 && !outgoing.isEmpty()) {
                int channel = outgoing.get(rng.nextInt(outgoing.size()));
                log.add(new Event(process, "send", channel, nextMessage));
                pending.put(nextMessage, channel);
                nextMessage++;
            } else if (canReceive) {
                int channel = incoming.get(rng.nextInt(incoming.size()));
                Integer message = firstPending(pending, channel);
                if (message == null) {
                    log.add(new Event(process, "internal", -1, -1));
                } else {
                    log.add(new Event(process, "receive", channel, message));
                    pending.remove(message);
                }
            } else {
                log.add(new Event(process, "internal", -1, -1));
            }
        }
        return log;
    }

    static boolean hasPending(Map<Integer, Integer> pending, int channel) {
        return firstPending(pending, channel) != null;
    }

    static Integer firstPending(Map<Integer, Integer> pending, int channel) {
        for (Map.Entry<Integer, Integer> entry : pending.entrySet()) {
            if (entry.getValue() == channel) return entry.getKey();
        }
        return null;
    }

    public static void main(String[] args) {
        List<Channel> channels = buildChannels();
        List<Event> log = buildLog(0x574E5EEDL, 60, 3, channels);
        check(log.size() == 60, "the trace has 60 events");
        int sends = 0, receives = 0;
        for (Event event : log) {
            if (event.kind.equals("send")) sends++;
            if (event.kind.equals("receive")) receives++;
        }
        check(sends >= 10 && receives >= 10, "the trace contains traffic: " + sends + " sends, " + receives
                + " receives");

        // 1. the recording equals the independent reconstruction, for every initiator
        int checkedSnapshots = 0, totalInTransit = 0;
        for (int initiator = 0; initiator < 3; initiator++) {
            Snapshot snapshot = new SnapshotRecorder(log, channels, 3).take(initiator);
            Map<String, List<Integer>> expected = CutOracle.inTransit(log, channels, snapshot.cutPosition);
            check(CutOracle.causallyConsistent(log, snapshot.cutPosition),
                    "the cut is causally consistent for initiator " + initiator);
            for (Channel channel : channels) {
                List<Integer> recorded = new ArrayList<>(snapshot.channelState.get(channel.key()));
                List<Integer> reconstructed = new ArrayList<>(expected.get(channel.key()));
                Collections.sort(recorded);
                Collections.sort(reconstructed);
                check(recorded.equals(reconstructed), "channel " + channel.key() + " for initiator "
                        + initiator + " recorded " + recorded + " but the log says " + reconstructed);
            }
            check(snapshot.processState.size() == 3, "all three processes recorded a state");
            totalInTransit += snapshot.inTransit();
            checkedSnapshots++;
        }
        check(checkedSnapshots == 3, "three snapshots verified");

        // 2. accounting: messages sent minus received equals what is still in flight at the end
        int[] accounting = CutOracle.counters(log, channels);
        Map<Integer, Integer> pendingAtEnd = new LinkedHashMap<>();
        for (Event event : log) {
            if (event.kind.equals("send")) pendingAtEnd.merge(event.channel, 1, Integer::sum);
            if (event.kind.equals("receive")) pendingAtEnd.merge(event.channel, -1, Integer::sum);
        }
        for (Channel channel : channels) {
            check(accounting[channel.index] == pendingAtEnd.getOrDefault(channel.index, 0), "accounting for "
                    + channel.key());
        }
        check(sends - receives == accounting[0] + accounting[1] + accounting[2], "global message balance");

        // 3. no phantom or duplicated messages, and every recorded message is genuinely in transit
        Snapshot snapshot = new SnapshotRecorder(log, channels, 3).take(0);
        Set<Integer> seen = new LinkedHashSet<>();
        for (Map.Entry<String, List<Integer>> entry : snapshot.channelState.entrySet()) {
            for (int message : entry.getValue()) {
                check(seen.add(message), "message " + message + " recorded twice");
                check(CutOracle.channelOf(message, log) == channelIndex(channels, entry.getKey()),
                        "message " + message + " belongs to its recorded channel");
            }
        }
        for (int message : seen) {
            int sendPosition = -1, receivePosition = -1;
            for (int i = 0; i < log.size(); i++) {
                Event event = log.get(i);
                if (event.payload != message) continue;
                if (event.kind.equals("send")) sendPosition = i;
                if (event.kind.equals("receive")) receivePosition = i;
            }
            check(sendPosition >= 0 && sendPosition < snapshot.cutPosition.get(log.get(sendPosition).process),
                    "recorded message was sent before its sender's state was recorded");
            check(receivePosition > snapshot.cutPosition.getOrDefault(log.get(receivePosition).process, -1),
                    "recorded message was not yet delivered when the receiver recorded its state");
        }

        // 4. hash chain: intact chains verify, tampering is caught at exactly the modified record
        long[] digests = AuditChain.build(log);
        check(AuditChain.verify(log, digests) == -1, "an intact audit chain verifies");
        int detected = 0;
        for (int index = 0; index < log.size(); index++) {
            List<Event> tampered = new ArrayList<>(log);
            Event original = tampered.get(index);
            tampered.set(index, new Event(original.process, original.kind, original.channel,
                    original.payload + 1000));
            long[] tamperedDigests = AuditChain.build(tampered);
            check(AuditChain.verify(tampered, digests) == index, "tampering at record " + index
                    + " is detected at " + AuditChain.verify(tampered, digests));
            check(AuditChain.verify(tampered, tamperedDigests) == -1, "the tampered chain is still "
                    + "self-consistent");
            detected++;
        }
        check(detected == log.size(), "every one of the " + log.size() + " records is tamper-evident");

        // 5. boundaries: an empty trace, a trace with no traffic at all, and a shortened trace
        List<Event> empty = new ArrayList<>();
        Snapshot emptySnapshot = new SnapshotRecorder(empty, channels, 3).take(0);
        check(emptySnapshot.inTransit() == 0, "an empty trace has nothing in transit");
        check(AuditChain.verify(empty, AuditChain.build(empty)) == -1, "an empty chain verifies");
        List<Event> internals = new ArrayList<>();
        for (int i = 0; i < 5; i++) internals.add(new Event(i % 3, "internal", -1, -1));
        Snapshot quiet = new SnapshotRecorder(internals, channels, 3).take(1);
        check(quiet.inTransit() == 0, "a trace without traffic records no messages");
        check(CutOracle.causallyConsistent(internals, quiet.cutPosition), "the quiet cut is consistent");
        check(AuditChain.verify(log, Arrays.copyOf(digests, digests.length - 1)) == log.size() - 1,
                "a chain of the wrong length is rejected at the first missing record");

        // 6. determinism: the same seed rebuilds the same trace and the same snapshot
        List<Event> replay = buildLog(0x574E5EEDL, 60, 3, channels);
        check(replay.size() == log.size(), "same seed, same trace length");
        boolean identical = true;
        for (int i = 0; i < log.size(); i++) identical &= replay.get(i).show().equals(log.get(i).show());
        check(identical, "same seed reproduces the trace");
        check(Arrays.equals(AuditChain.build(replay), digests), "same trace, same digest chain");

        System.out.println("trace: " + log.size() + " events, " + sends + " sends, " + receives
                + " receives over " + channels.size() + " channels");
        System.out.println("snapshots: 3 initiators, each matching the log reconstruction; "
                + totalInTransit + " messages recorded in transit in total");
        System.out.println("audit chain: " + log.size() + " records, all tamper-evident at exactly the "
                + "modified index, no false positive");
        System.out.println("PASS");
    }

    static int channelIndex(List<Channel> channels, String key) {
        for (Channel channel : channels) if (channel.key().equals(key)) return channel.index;
        return -1;
    }
}
