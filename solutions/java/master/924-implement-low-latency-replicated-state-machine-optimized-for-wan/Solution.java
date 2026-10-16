/** Task 924 - Implement low-latency replicated state machine optimized for WANs with batching.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A five-replica replicated state machine over a wide-area network, driven by an explicit tick schedule. The
 * client proposes operations to the leader; the leader buffers them and flushes a batch at the end of every
 * `window` ticks, paying a serialisation cost that grows with the batch size, one WAN round trip (delay D each
 * way) and then a 3-of-5 quorum of acknowledgements. The batch commits atomically at the tick the quorum ack
 * lands, and every live replica appends then applies the batch in the leader's order.
 * Simplified: an ideal network (fixed delay, no loss or reordering), equally fast followers, no elections or
 * log repair, and atomic batch commits rather than entry-by-entry. Not simplified: the quorum rule, the
 * batched round trip with its serialisation cost, per-replica logs, and the two invariants that make it safe -
 * a replica only ever applies a prefix of the committed log, and any two logs agree on every index they share.
 * Verified: every live replica converges to the oracle state machine; both safety invariants hold after every
 * commit; batching cuts the round trips and messages by more than the batch width; the latency bound of a
 * batched round trip is respected and the batching delay is visible against the unbatched run; a partitioned
 * fifth replica changes neither the commits nor the latency; and idle, single-operation and replay runs behave.
 */
import java.util.*;

/** One replica: the entries it has received and how many of them it has applied to its state machine. */
final class Replica {
    final int id;
    final List<String> log = new ArrayList<>();
    int applied;
    TreeMap<String, Integer> state = new TreeMap<>();

    Replica(int id) { this.id = id; }
}

/** A trivial key/value state machine: "k3=2" style operations, applied strictly in log order. */
final class StateMachine {
    void apply(TreeMap<String, Integer> state, String op) {
        String[] parts = op.split("=");
        state.merge(parts[0], Integer.parseInt(parts[1]), Integer::sum);
    }

    TreeMap<String, Integer> from(List<String> ops) {
        TreeMap<String, Integer> state = new TreeMap<>();
        for (String op : ops) apply(state, op);
        return state;
    }
}

/** The outcome of one simulated configuration. */
final class Run {
    final List<String> committed = new ArrayList<>();
    final List<Replica> replicas = new ArrayList<>();
    int messages, batches, safetyViolations, logMismatches;
}

public class Solution {
    static final int REPLICAS = 5, QUORUM = 3, WAN = 4, HEADER_COST = 1, TICKS = 40;

    static int serialize(int ops) { return HEADER_COST + (ops + 3) / 4; }

    /**
     * @param window      ticks between two flushes (1 = one round trip per tick, i.e. unbatched)
     * @param downReplica index of a partitioned replica, or -1
     * @param opsPerTick  client load
     * @param totalOps    how many operations the client proposes in total
     */
    static Run simulate(int window, int downReplica, int opsPerTick, int totalOps) {
        Run run = new Run();
        for (int r = 0; r < REPLICAS; r++) run.replicas.add(new Replica(r));
        StateMachine machine = new StateMachine();
        int ticks = totalOps == 0 ? 0 : (totalOps + opsPerTick - 1) / opsPerTick;
        int windowStart = 0, proposed = 0;
        while (proposed < totalOps) {
            int windowEnd = Math.min(ticks, windowStart + window);
            int upTo = Math.min(totalOps, windowEnd * opsPerTick);
            List<String> batch = new ArrayList<>();
            while (proposed < upTo) {
                batch.add("k" + (proposed % 7) + "=" + (1 + proposed % 5));
                proposed++;
            }
            if (batch.isEmpty()) { windowStart = windowEnd; continue; }
            int acks = REPLICAS - 1 - (downReplica > 0 ? 1 : 0);      // the leader always acks for itself
            run.messages += (REPLICAS - 1) + acks;                    // AppendEntries fan-out plus the acks
            run.batches++;
            if (acks < QUORUM - 1) break;                             // no quorum: nothing can commit
            run.committed.addAll(batch);
            for (int r = 0; r < REPLICAS; r++) {
                Replica replica = run.replicas.get(r);
                if (r == downReplica) continue;                       // the partition never receives it
                replica.log.addAll(batch);
                int before = replica.applied;
                while (replica.applied < run.committed.size()) machine.apply(replica.state, replica.log.get(replica.applied++));
                if (replica.applied != before + batch.size()) run.safetyViolations++;
                if (replica.applied > run.committed.size()) run.safetyViolations++;
            }
            for (int a = 0; a < REPLICAS; a++) {                      // log matching on every shared index
                for (int b = a + 1; b < REPLICAS; b++) {
                    List<String> x = run.replicas.get(a).log, y = run.replicas.get(b).log;
                    for (int i = 0; i < Math.min(x.size(), y.size()); i++) if (!x.get(i).equals(y.get(i))) run.logMismatches++;
                }
            }
            windowStart = windowEnd;
        }
        return run;
    }

    /** Commit latency of every operation, measured from the tick the client proposed it. */
    static List<Double> latencies(int window, int opsPerTick, int totalOps) {
        List<Double> out = new ArrayList<>();
        int ticks = (totalOps + opsPerTick - 1) / opsPerTick, windowStart = 0, proposed = 0;
        while (proposed < totalOps) {
            int windowEnd = Math.min(ticks, windowStart + window);
            int upTo = Math.min(totalOps, windowEnd * opsPerTick);
            if (upTo == proposed) { windowStart = windowEnd; continue; }
            double commit = windowEnd + serialize(upTo - proposed) + 2.0 * WAN;
            for (int t = proposed; t < upTo; t++) out.add(commit - t / (double) opsPerTick);
            proposed = upTo;
            windowStart = windowEnd;
        }
        return out;
    }

    static double mean(List<Double> values) {
        double sum = 0;
        for (double v : values) sum += v;
        return values.isEmpty() ? 0 : sum / values.size();
    }

    public static void main(String[] args) {
        Run batched = simulate(3, -1, 2, 80);
        Run unbatched = simulate(1, -1, 2, 80);
        Run partitioned = simulate(3, 4, 2, 80);
        List<String> everything = batched.replicas.get(0).log;
        TreeMap<String, Integer> oracle = new StateMachine().from(everything);
        List<Double> batchedLatency = latencies(3, 2, 80);
        List<Double> unbatchedLatency = latencies(1, 2, 80);

        System.out.println("replicated log: " + everything.size() + " ops, window 3 -> " + batched.batches + " batches / "
                + batched.messages + " messages; unbatched -> " + unbatched.batches + " batches / " + unbatched.messages
                + " messages");
        System.out.println("latency (one-way WAN " + WAN + "): batched mean " + String.format("%.2f", mean(batchedLatency))
                + " (max " + String.format("%.2f", Collections.max(batchedLatency)) + "), unbatched mean "
                + String.format("%.2f", mean(unbatchedLatency)) + ", safety violations " + batched.safetyViolations
                + ", log mismatches " + batched.logMismatches);

        for (Replica r : batched.replicas) {
            check(r.state.equals(oracle), "replica " + r.id + " must converge to the oracle state machine");
            check(r.applied == everything.size(), "replica " + r.id + " must apply every committed entry, got " + r.applied);
        }
        check(batched.safetyViolations == 0, "a replica must never apply past the commit index");
        check(batched.logMismatches == 0, "logs must agree on every shared index");
        check(batched.committed.size() == 80, "every proposed operation must commit: " + batched.committed.size());

        check(batched.messages > 0 && batched.messages * 2 <= unbatched.messages,
                "batching must more than halve the messages: " + batched.messages + " vs " + unbatched.messages);
        check(batched.batches * 2 <= unbatched.batches, "batching must cut the round trips: " + batched.batches
                + " vs " + unbatched.batches);

        check(mean(unbatchedLatency) >= 2.0 * WAN, "an unbatched round trip cannot beat two WAN delays: "
                + mean(unbatchedLatency));
        check(mean(batchedLatency) > mean(unbatchedLatency), "batching pays with latency: " + mean(batchedLatency)
                + " vs " + mean(unbatchedLatency));
        check(Collections.max(batchedLatency) <= 2.0 * WAN + 3 + serialize(6) + 0.5, "a batched round trip must stay "
                + "inside the bound 2D + window + serialise(6), got " + Collections.max(batchedLatency));

        check(partitioned.committed.size() == batched.committed.size(), "a 4/5 quorum must still commit everything");
        check(partitioned.replicas.get(4).log.size() < everything.size(), "the partitioned replica must lag behind");
        for (int r = 0; r < 4; r++) check(partitioned.replicas.get(r).state.equals(oracle),
                "live replica " + r + " must converge with the fifth partitioned away");
        check(mean(latencies(3, 2, 80)) == mean(batchedLatency), "the commit latency ignores the partitioned fifth");

        Run idle = simulate(3, -1, 2, 0);
        check(idle.messages == 0 && idle.committed.isEmpty() && idle.batches == 0,
                "an idle client must produce no messages and no commits");
        Run single = simulate(3, -1, 2, 1);
        check(single.batches == 1 && single.committed.size() == 1, "a single operation is a single batch: "
                + single.batches + " batches, " + single.committed.size() + " ops");
        check(single.replicas.get(3).state.equals(new StateMachine().from(single.replicas.get(0).log)),
                "a single-operation run must still converge on every replica");

        Run replay = simulate(3, -1, 2, 80);
        check(replay.messages == batched.messages && replay.committed.equals(batched.committed)
                && replay.replicas.get(2).state.equals(oracle), "the schedule must be reproducible");

        System.out.println("partitioned replica applied " + partitioned.replicas.get(4).applied + " of "
                + everything.size() + " entries; live replicas " + partitioned.replicas.get(0).applied + "/"
                + partitioned.replicas.get(3).applied + "; batched latency gain "
                + String.format("%.2f", mean(batchedLatency) - mean(unbatchedLatency)) + " ticks for "
                + (unbatched.messages - batched.messages) + " fewer messages");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
