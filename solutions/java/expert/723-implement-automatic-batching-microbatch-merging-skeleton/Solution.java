/** Task 723 - Implement automatic batching (microbatch merging) skeleton. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A serving loop receives one inference request at a time, but accelerating hardware is only efficient when
 * it is fed a batch: a kernel launch costs the same whether it processes one item or thirty-two. Automatic
 * batching merges whatever has arrived into one microbatch. The scheduler here is the standard one: it opens
 * a window at the first queued request, absorbs requests while the batch is not full and while they arrive
 * no later than maxDelay after the window opened, then launches. Batches therefore trade latency (up to
 * maxDelay of waiting) for throughput (up to maxBatchSize requests per launch).
 *
 * Invariants verified:
 *   - partition    every request is in exactly one batch, batches keep arrival order, and batch size never
 *                  exceeds maxBatchSize
 *   - latency      no request waits longer than maxDelay, a request never joins a batch that opened before
 *                  it arrived, and launch times are non-decreasing
 *   - greedy       each batch is maximal: it is full, or the next request arrived after the deadline
 *   - economy      the modelled cost, which includes the padding waste of ragged batches, is lower than
 *                  serving every request alone, with the launch count reduced by the measured factor
 *
 * Simplified: arrival times and sequence lengths come from a deterministic generator instead of a network,
 * and the cost model is a fixed launch overhead plus the total padded token count, so a "kernel" is a unit
 * of modelled work rather than a real device call. The verified properties are the scheduling ones: the
 * partition, the latency bound, greedy maximality, and the throughput accounting.
 */
import java.util.*;

public class Solution {
    /** One inference request: when it arrived and how much work it carries. */
    static final class Request {
        final int id;
        final int arrival;
        final int length;
        Request(int id, int arrival, int length) {
            this.id = id;
            this.arrival = arrival;
            this.length = length;
        }
    }

    /** A launched microbatch and the numbers the cost model needs. */
    static final class Batch {
        final List<Integer> ids = new ArrayList<>();
        int openedAt;
        int launchedAt;
        int maxLength;
        int sumLength;

        int size() { return ids.size(); }
        /** Ragged work: the shorter sequences are padded to the longest one in the batch. */
        int padding() { return size() * maxLength - sumLength; }
        int cost(int launchOverhead) { return launchOverhead + sumLength + padding(); }
    }

    /** The scheduler: window of maxDelay, at most maxBatchSize requests per launch. */
    static final class Batcher {
        final int maxBatchSize;
        final int maxDelay;
        final int launchOverhead;

        Batcher(int maxBatchSize, int maxDelay, int launchOverhead) {
            if (maxBatchSize <= 0) throw new IllegalArgumentException("batch size must be positive");
            if (maxDelay < 0) throw new IllegalArgumentException("delay must not be negative");
            if (launchOverhead < 0) throw new IllegalArgumentException("overhead must not be negative");
            this.maxBatchSize = maxBatchSize;
            this.maxDelay = maxDelay;
            this.launchOverhead = launchOverhead;
        }

        /** Plans the schedule; `requests` must be sorted by arrival time. */
        List<Batch> plan(List<Request> requests) {
            List<Batch> batches = new ArrayList<>();
            int i = 0;
            while (i < requests.size()) {
                Batch batch = new Batch();
                batch.openedAt = requests.get(i).arrival;
                int deadline = batch.openedAt + maxDelay;
                int j = i;
                while (j < requests.size() && j - i < maxBatchSize && requests.get(j).arrival <= deadline) {
                    Request request = requests.get(j);
                    batch.ids.add(request.id);
                    batch.sumLength += request.length;
                    batch.maxLength = Math.max(batch.maxLength, request.length);
                    batch.launchedAt = request.arrival;
                    j++;
                }
                batches.add(batch);
                i = j;
            }
            return batches;
        }
    }

    private static long rngState = 0xB5297A4D2E9F3C15L;

    /** splitmix64 - deterministic, no Math.random. */
    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** A stream with a given mean gap between arrivals and a bounded sequence length. */
    static List<Request> stream(int count, int meanGap, int maxLength) {
        List<Request> requests = new ArrayList<>();
        int time = 0;
        for (int id = 0; id < count; id++) {
            time += nextInt(2 * meanGap + 1);
            requests.add(new Request(id, time, 1 + nextInt(maxLength)));
        }
        return requests;
    }

    /** Cost of serving each request on its own: one launch each. */
    static long soloCost(List<Request> requests, int launchOverhead) {
        long cost = 0;
        for (Request request : requests) cost += launchOverhead + request.length;
        return cost;
    }

    static long batchedCost(List<Batch> batches, int launchOverhead) {
        long cost = 0;
        for (Batch batch : batches) cost += batch.cost(launchOverhead);
        return cost;
    }

    /** Returns a description of the first scheduling violation, or null when the plan is sound. */
    static String verify(List<Request> requests, List<Batch> batches, Batcher batcher) {
        Map<Integer, Integer> seen = new HashMap<>();
        int previousLaunch = Integer.MIN_VALUE;
        for (Batch batch : batches) {
            if (batch.size() == 0 || batch.size() > batcher.maxBatchSize) return "bad batch size " + batch.size();
            if (batch.launchedAt < previousLaunch) return "batches launched out of order";
            previousLaunch = batch.launchedAt;
            for (int id : batch.ids) {
                if (seen.put(id, 1) != null) return "request " + id + " in two batches";
                Request request = requests.get(id);
                if (request.arrival > batch.launchedAt) return "request " + id + " joined before arriving";
                if (batch.launchedAt - request.arrival > batcher.maxDelay) {
                    return "request " + id + " waited " + (batch.launchedAt - request.arrival);
                }
            }
        }
        if (seen.size() != requests.size()) return "lost requests: " + seen.size() + " of " + requests.size();
        return null;
    }

    /** True when a batch absorbed everything it legally could. */
    static boolean maximal(List<Request> requests, List<Batch> batches, Batcher batcher) {
        int at = 0;
        for (Batch batch : batches) {
            int deadline = requests.get(at).arrival + batcher.maxDelay;
            if (batch.size() < batcher.maxBatchSize && at + batch.size() < requests.size()) {
                if (requests.get(at + batch.size()).arrival <= deadline) return false;
            }
            at += batch.size();
        }
        return true;
    }

    public static void main(String[] args) {
        final int overhead = 12;

        // --- rejected parameters (negative checks) -----------------------------
        boolean rejectedSize = false;
        boolean rejectedDelay = false;
        try { new Batcher(0, 5, overhead); } catch (IllegalArgumentException e) { rejectedSize = true; }
        try { new Batcher(4, -1, overhead); } catch (IllegalArgumentException e) { rejectedDelay = true; }
        check(rejectedSize && rejectedDelay,
                "a non-positive batch size and a negative delay must both be rejected");

        // --- degenerate streams ------------------------------------------------
        Batcher batcher = new Batcher(8, 6, overhead);
        check(batcher.plan(new ArrayList<>()).isEmpty(), "an empty stream needs no batches");
        check(batchedCost(batcher.plan(new ArrayList<>()), overhead) == 0, "an empty stream costs nothing");

        List<Request> lone = Collections.singletonList(new Request(0, 5, 7));
        List<Batch> loneBatches = batcher.plan(lone);
        check(loneBatches.size() == 1 && loneBatches.get(0).size() == 1, "a single request forms one batch");
        check(loneBatches.get(0).launchedAt == 5, "the batch launches as soon as the request arrives");
        check(loneBatches.get(0).cost(overhead) == overhead + 7, "one request costs one launch and its work");

        // --- a dense stream: the case batching exists for ----------------------
        rngState = 0xB5297A4D2E9F3C15L;
        List<Request> dense = stream(400, 2, 5);
        List<Batch> denseBatches = batcher.plan(dense);
        check(verify(dense, denseBatches, batcher) == null, "dense plan: " + verify(dense, denseBatches, batcher));
        check(maximal(dense, denseBatches, batcher), "every dense batch must be maximal");
        long denseSoloCost = soloCost(dense, overhead);
        long denseBatchedCost = batchedCost(denseBatches, overhead);
        check(denseBatchedCost < denseSoloCost, "batching must reduce the modelled cost: " + denseBatchedCost
                + " vs " + denseSoloCost);
        check(denseBatchedCost * 3 < denseSoloCost * 2, "batching must win by more than 1.5x");
        long paddingTotal = 0;
        for (Batch batch : denseBatches) paddingTotal += batch.padding();
        check(denseBatchedCost == denseSoloCost - (dense.size() - denseBatches.size()) * (long) overhead
                        + paddingTotal,
                "the saving is exactly the launches removed minus the padding added");
        check(paddingTotal > 0, "ragged sequence lengths must produce padding waste");
        check(denseBatches.size() * 4 < dense.size(), "the launch count must drop by at least 4x: "
                + denseBatches.size() + " vs " + dense.size());
        double averageBatch = dense.size() / (double) denseBatches.size();
        check(averageBatch > 3.5, "a dense stream should fill its window, average " + averageBatch);
        Batcher wide = new Batcher(8, 20, overhead);
        List<Batch> wideBatches = wide.plan(dense);
        double wideAverage = dense.size() / (double) wideBatches.size();
        check(verify(dense, wideBatches, wide) == null && wideAverage > 0.9 * wide.maxBatchSize,
                "a wider window must fill the batches up to the size limit, average " + wideAverage);

        // --- latency accounting on the dense stream ---------------------------
        int worstWait = 0;
        int filled = 0;
        for (Batch batch : denseBatches) {
            if (batch.size() == batcher.maxBatchSize) filled++;
            for (int id : batch.ids) worstWait = Math.max(worstWait, batch.launchedAt - dense.get(id).arrival);
        }
        check(worstWait <= batcher.maxDelay, "no request may wait longer than maxDelay, worst " + worstWait);
        check(worstWait > 0, "some request must actually wait for a batch to fill");
        check(filled > 0, "some batches must fill up completely");

        // --- padding: ragged batches must be accounted for --------------------
        List<Request> ragged = new ArrayList<>();
        ragged.add(new Request(0, 0, 1));
        ragged.add(new Request(1, 1, 4));
        ragged.add(new Request(2, 2, 9));
        Batcher small = new Batcher(3, 10, overhead);
        List<Batch> raggedBatches = small.plan(ragged);
        check(raggedBatches.size() == 1 && raggedBatches.get(0).size() == 3, "all three merge into one batch");
        check(raggedBatches.get(0).padding() == 3 * 9 - 14, "padding pads each row to the longest one: "
                + raggedBatches.get(0).padding());
        check(raggedBatches.get(0).cost(overhead) == overhead + 3 * 9, "padded cost is launch plus the pad");
        List<Request> equal = new ArrayList<>();
        for (int i = 0; i < 3; i++) equal.add(new Request(i, i, 5));
        check(small.plan(equal).get(0).padding() == 0, "equal-length requests pad nothing");

        // --- boundaries: no merging possible or not allowed -------------------
        List<Batch> solo = batcher.plan(dense);
        Batcher noMerging = new Batcher(1, 6, overhead);
        List<Batch> soloBatches = noMerging.plan(dense);
        check(soloBatches.size() == dense.size(), "a batch size of one means one launch per request");
        check(batchedCost(soloBatches, overhead) == soloCost(dense, overhead),
                "with a batch size of one, batching and serving alone cost the same");
        check(solo.size() > 0 && soloBatches.size() > solo.size(), "merging must strictly reduce launches");

        Batcher noWaiting = new Batcher(8, 0, overhead);
        List<Batch> impatient = noWaiting.plan(dense);
        check(verify(dense, impatient, noWaiting) == null, "zero delay still yields a valid plan");
        int impatientWorst = 0;
        for (Batch batch : impatient) {
            for (int id : batch.ids) impatientWorst = Math.max(impatientWorst, batch.launchedAt - dense.get(id).arrival);
        }
        check(impatientWorst == 0, "with zero delay nobody waits, worst " + impatientWorst);
        check(maximal(dense, impatient, noWaiting), "the zero-delay plan is still greedy");

        rngState = 0x77F1B2C3D4E5A690L;
        List<Request> idle = stream(40, 40, 5);
        List<Batch> idleBatches = batcher.plan(idle);
        check(verify(idle, idleBatches, batcher) == null, "idle stream plan: "
                + verify(idle, idleBatches, batcher));
        check(idleBatches.size() > 3 * idle.size() / 4 && idle.size() / (double) idleBatches.size() < 1.2,
                "requests further apart than the window barely merge: " + idleBatches.size() + " batches for "
                        + idle.size() + " requests");

        // --- determinism -------------------------------------------------------
        rngState = 0xB5297A4D2E9F3C15L;
        List<Request> replay = stream(400, 2, 5);
        List<Batch> replayBatches = batcher.plan(replay);
        check(replayBatches.size() == denseBatches.size(), "the same stream must plan the same batches");
        check(batchedCost(replayBatches, overhead) == denseBatchedCost, "and the same cost");
        check(dense.get(dense.size() - 1).id == replay.get(replay.size() - 1).id
                        && dense.get(dense.size() - 1).arrival == replay.get(replay.size() - 1).arrival,
                "the generator must be repeatable");

        System.out.println("automatic batching: " + dense.size() + " requests, batch size " + batcher.maxBatchSize
                + ", window " + batcher.maxDelay + ", launch overhead " + overhead);
        System.out.println("batches " + denseBatches.size() + " (average " + averageBatch + " requests), cost "
                + denseBatchedCost + " against " + denseSoloCost + " unbatched = "
                + (denseSoloCost / (double) denseBatchedCost) + "x");
        System.out.println("worst wait " + worstWait + " < window " + batcher.maxDelay + ", fully filled batches "
                + filled);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
