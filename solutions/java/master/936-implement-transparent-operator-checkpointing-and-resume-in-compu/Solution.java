/** Task 936 - Implement transparent operator checkpointing and resume in compute graph.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A linear streaming compute graph (source -> three stateful operators -> sink) driven by
 *  Chandy-Lamport barriers: every `interval` records the source injects a barrier that travels one hop
 *  per processed record; only when it reaches the sink is the checkpoint published, and only the
 *  outputs produced *before* the injection are committed. On a crash the executor restores the newest
 *  published checkpoint, rewinds the source to the checkpointed offset and replays the tail. A barrier
 *  still in flight at the crash is invisible, which is what makes the checkpointing transparent.
 *  Simplified: single-threaded mock of the barrier transport (one hop per record instead of channel
 *  queues) and a scripted crash point instead of a random failure injector.
 *  Verified: every record is committed exactly once even across rollbacks, the recovered output and
 *  state are identical to the failure-free run, each published snapshot equals a from-scratch prefix
 *  computation, a mid-barrier crash rolls back to the *previous* published checkpoint, and the final
 *  aggregate matches an independent closed-form oracle.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static final int OPS = 3;          // stateful operators between the source and the sink
    static final int HOPS = OPS + 1;   // barrier hops: one per operator, plus the sink ack

    static int value(int record) { return 3 * record + 1; }

    /** A published checkpoint: the source offset it covers and the operator state at that offset. */
    static final class Checkpoint {
        final int offset, committed, completedAt;
        final int[] state;
        Checkpoint(int offset, int[] state, int committed, int completedAt) {
            this.offset = offset;
            this.state = state;
            this.committed = committed;
            this.completedAt = completedAt;
        }
    }

    /** A barrier travelling downstream; several can be in flight when the interval is short. */
    static final class Barrier {
        final int mark;      // records produced when the barrier was injected
        final int[] state;   // operator state at that moment
        int hopsLeft;
        Barrier(int mark, int[] state, int hopsLeft) {
            this.mark = mark;
            this.state = state;
            this.hopsLeft = hopsLeft;
        }
    }

    static final class Pipeline {
        final int records, interval;
        final int[] state = new int[OPS];
        final List<int[]> committed = new ArrayList<>();
        final List<int[]> pending = new ArrayList<>();      // produced but not yet committed output
        final List<Checkpoint> checkpoints = new ArrayList<>();
        final List<Barrier> inFlight = new ArrayList<>();   // FIFO: the oldest completes first
        int srcOffset, processed, recordsSinceBarrier;

        Pipeline(int records, int interval) { this.records = records; this.interval = interval; }

        void step() {
            for (Iterator<Barrier> it = inFlight.iterator(); it.hasNext(); ) {
                Barrier b = it.next();
                if (--b.hopsLeft == 0) {
                    publish(b);
                    it.remove();
                }
            }
            int r = srcOffset++;
            state[0] = state[0] * 31 + value(r);
            state[1] ^= value(r) << (r % 7);
            state[2] += value(r) * (r + 1);
            pending.add(new int[] {r, state[2]});
            processed++;
            if (++recordsSinceBarrier == interval && srcOffset < records) inject();
        }

        void inject() {
            recordsSinceBarrier = 0;
            inFlight.add(new Barrier(srcOffset, state.clone(), HOPS));
        }

        /** A barrier reached the sink: publish its snapshot and commit exactly the outputs before it. */
        void publish(Barrier b) {
            int n = b.mark - committed.size();
            committed.addAll(new ArrayList<>(pending.subList(0, n)));
            pending.subList(0, n).clear();
            checkpoints.add(new Checkpoint(b.mark, b.state, committed.size(), processed));
        }

        /** End of stream: complete every barrier still in flight, then flush the tail of the output. */
        void drain() {
            while (!inFlight.isEmpty()) publish(inFlight.remove(0));
            if (!pending.isEmpty()) {
                inject();
                publish(inFlight.remove(0));
            }
        }

        /** Roll back to the newest published checkpoint; in-flight barriers are discarded. */
        int restore() {
            Checkpoint cp = checkpoints.isEmpty() ? null : checkpoints.get(checkpoints.size() - 1);
            int offset = cp == null ? 0 : cp.offset;
            Arrays.fill(state, 0);
            if (cp != null) System.arraycopy(cp.state, 0, state, 0, OPS);
            while (committed.size() > (cp == null ? 0 : cp.committed)) committed.remove(committed.size() - 1);
            pending.clear();
            inFlight.clear();
            srcOffset = offset;
            recordsSinceBarrier = offset % interval;
            return offset;
        }
    }

    static final class Run {
        final List<int[]> output;
        final int[] state;
        final List<Checkpoint> checkpoints;
        final int processed;
        int restores, restoredOffset = -1, inFlightAtFail = -1;
        Run(Pipeline p) {
            this.output = p.committed;
            this.state = p.state.clone();
            this.checkpoints = p.checkpoints;
            this.processed = p.processed;
        }
    }

    /** Execute the graph; failAt >= 0 injects a crash after that many records were consumed. */
    static Run execute(int records, int interval, int failAt) {
        Pipeline p = new Pipeline(records, interval);
        boolean crash = failAt >= 0 && failAt < records;
        int inFlightAtFail = -1, restoredOffset = -1, restores = 0;
        for (int i = 0; i < (crash ? failAt : records); i++) p.step();
        if (crash) {
            inFlightAtFail = p.inFlight.isEmpty() ? -1 : p.inFlight.get(0).mark;
            restoredOffset = p.restore();
            restores = 1;
            while (p.srcOffset < records) p.step();
        }
        p.drain();
        Run run = new Run(p);                      // snapshots the state after any recovery
        run.inFlightAtFail = inFlightAtFail;
        run.restoredOffset = restoredOffset;
        run.restores = restores;
        return run;
    }

    static boolean sameOutput(List<int[]> a, List<int[]> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) if (!Arrays.equals(a.get(i), b.get(i))) return false;
        return true;
    }

    /** Exactly-once witness: output i carries record index i, so this proves no loss and no duplicate. */
    static boolean ordered(List<int[]> out) {
        for (int i = 0; i < out.size(); i++) if (out.get(i)[0] != i) return false;
        return true;
    }

    static int[] stateFromScratch(int n) {
        int[] st = new int[OPS];
        for (int r = 0; r < n; r++) {
            int v = value(r);
            st[0] = st[0] * 31 + v;
            st[1] ^= v << (r % 7);
            st[2] += v * (r + 1);
        }
        return st;
    }

    /** Closed form of sum over r of (3r+1)(r+1) - an oracle that shares no code with the pipeline. */
    static long weightedSumClosedForm(int n) {
        long m = n;
        return 3 * (m - 1) * m * (2 * m - 1) / 6 + 4 * m * (m - 1) / 2 + m;
    }

    /** Newest checkpoint that was already published when `failAt` records had been consumed. */
    static int expectedRestore(List<Checkpoint> clean, int failAt) {
        int expected = 0;
        for (Checkpoint cp : clean) if (cp.completedAt <= failAt) expected = Math.max(expected, cp.offset);
        return expected;
    }

    public static void main(String[] args) {
        final int records = 600, interval = 8;
        Run clean = execute(records, interval, -1);

        check(clean.restores == 0 && clean.output.size() == records && ordered(clean.output),
                "a failure-free run must commit every record exactly once, got " + clean.output.size());
        check(clean.checkpoints.size() >= 60
                        && clean.checkpoints.get(clean.checkpoints.size() - 1).offset == records,
                "the failure-free run must publish one checkpoint per barrier plus a final flush, got "
                        + clean.checkpoints.size());

        int badSnapshots = 0;
        for (Checkpoint cp : clean.checkpoints) {
            if (!Arrays.equals(cp.state, stateFromScratch(cp.offset))) badSnapshots++;
        }
        check(badSnapshots == 0, badSnapshots + " of " + clean.checkpoints.size()
                + " snapshots disagree with a from-scratch prefix computation");

        final int failAt = 98;                                  // barrier from offset 96 is still in flight
        Run recovered = execute(records, interval, failAt);
        int expected = expectedRestore(clean.checkpoints, failAt);
        check(recovered.restoredOffset == expected && recovered.inFlightAtFail > recovered.restoredOffset,
                "a mid-barrier crash must roll back to the newest published checkpoint: restored "
                        + recovered.restoredOffset + ", in flight " + recovered.inFlightAtFail
                        + ", expected " + expected);
        check(sameOutput(clean.output, recovered.output) && Arrays.equals(clean.state, recovered.state),
                "recovery must be exactly-once: " + recovered.output.size() + " vs " + clean.output.size()
                        + " outputs, state " + Arrays.toString(clean.state) + " vs "
                        + Arrays.toString(recovered.state));
        check(recovered.processed > records,
                "the replay must re-consume the tail, processed " + recovered.processed + " records");

        Run early = execute(records, interval, 3);              // boundary: nothing published yet
        check(early.restoredOffset == 0 && early.restores == 1
                        && sameOutput(clean.output, early.output) && Arrays.equals(clean.state, early.state),
                "a crash before the first barrier completes must replay from genesis");

        Run perRecord = execute(50, 1, -1);
        Run perRecordFailed = execute(50, 1, 27);
        Run oneBarrier = execute(50, 64, -1);
        Run oneBarrierFailed = execute(50, 64, 27);
        check(sameOutput(perRecord.output, perRecordFailed.output) && perRecordFailed.restoredOffset > 0
                        && sameOutput(oneBarrier.output, oneBarrierFailed.output)
                        && oneBarrier.checkpoints.size() == 1 && oneBarrierFailed.restoredOffset == 0,
                "interval 1 must resume from a recent checkpoint while a single end-of-stream barrier falls back to genesis");

        Run empty = execute(0, interval, -1);
        check(empty.output.isEmpty() && empty.checkpoints.isEmpty()
                        && empty.state[0] == 0 && empty.state[1] == 0 && empty.state[2] == 0
                        && execute(0, interval, -1).restores == 0,
                "an empty stream must produce no output and no checkpoints");

        check(clean.state[2] == (int) weightedSumClosedForm(records)
                        && Arrays.equals(clean.state, stateFromScratch(records))
                        && clean.state[2] == clean.output.get(records - 1)[1],
                "the aggregate must match the closed-form oracle: " + clean.state[2] + " vs "
                        + weightedSumClosedForm(records));

        int bad = 0;
        int[] crashPoints = {1, 7, 12, 39, 97, 100, 260, 597};
        for (int f : crashPoints) {
            Run r = execute(records, interval, f);
            if (!sameOutput(clean.output, r.output) || !ordered(r.output) || !Arrays.equals(clean.state, r.state)) bad++;
        }
        check(bad == 0, bad + " of " + crashPoints.length + " crash points broke exactly-once semantics");

        System.out.println("checkpointing: " + records + " records, barrier every " + interval
                + " (hops " + HOPS + "), " + clean.checkpoints.size() + " checkpoints, "
                + recovered.processed + " physical records after one crash");
        System.out.println("restored offset " + recovered.restoredOffset + " (in flight "
                + recovered.inFlightAtFail + "), aggregate " + clean.state[2] + ", crash points tested "
                + crashPoints.length);
        System.out.println("PASS");
    }
}
