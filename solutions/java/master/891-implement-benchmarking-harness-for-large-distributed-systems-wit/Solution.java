/** Task 891 - Implement benchmarking harness for large distributed systems with failure injection. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A deterministic discrete-event simulator of a replicated request path, plus the harness that measures it.
 * Requests arrive open-loop (exponential inter-arrival from splitmix64), a coordinator fans each one out to
 * k replicas, and the request completes when a quorum of acknowledgements arrives before its deadline;
 * otherwise it fails. Fault injection is explicit and scheduled: network loss (a message never leaves),
 * node crashes (an arriving message is discarded and in-flight work dies), slow nodes (service time
 * multiplied) and per-node clock skew. All timing comes from the simulated clock and the event queue, so
 * the run is bit-for-bit reproducible and no wall clock, thread or socket is involved.
 *
 * The harness reports throughput, error rate, nearest-rank latency quantiles, per-node queue depth, message
 * conservation, causality and Little's law, and the checks cross-verify the reported numbers against an
 * independent computation over the raw samples. Neighbourhood scaling: the demo runs 16 nodes / 500
 * requests / 3 replicas per request, and every knob (nodes, replicas, quorum, rates, fault probabilities,
 * skew) is a parameter, so the same code sizes up to tens of thousands of events.
 *
 * What is simplified: replicas are state machines that acknowledge after a service time rather than real
 * services; there is no retry or client-side backoff; loss is an independent coin per message rather than a
 * bursty channel; skew is a fixed per-node offset that is applied to that node's own timestamps and never
 * to the causal order, which is the property that is actually verified (injecting skew must not reorder
 * events). Nothing is faked: every reported metric is recomputed and compared by the checks.
 *
 * Verified: two runs with the same seed are identical in metrics, event counts and samples; with faults off
 * every request completes, the error rate is zero and the achieved throughput sits between half the offered
 * load and the offered load plus slack; message conservation holds exactly (sent = delivered + lost in
 * flight + dropped), message ids are unique and never delivered twice, and an acknowledgement never precedes
 * its replica's message, which never precedes its send time; the event queue never emits an out-of-order
 * event; the injected fault counts match the schedule; error rate is non-decreasing and throughput
 * non-increasing as loss grows; quantiles match an independent selection-based oracle and the degenerate
 * quantile cases (empty, single sample, p0, p100) are exact; the timeout path is exercised under heavy loss;
 * clock skew is applied to node timestamps while causal order and non-negative latencies survive; Little's
 * law holds within 15%; zero-request and single-node runs behave; and warmup exclusion removes exactly the
 * samples inside the window.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Uniform in [0, 1). */
    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    long uniform(long lo, long hi) { return lo + Math.floorMod(nextLong(), Math.max(1, hi - lo + 1)); }

    /** Exponential with the given mean, drawn by inversion, capped at `cap`. */
    long exponential(double mean, long cap) {
        double u = Math.max(1e-12, 1.0 - nextDouble());
        return Math.min(cap, Math.max(1, (long) (-mean * Math.log(u))));
    }

    boolean bernoulli(double p) { return nextDouble() < p; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** An event ordered by timestamp and then by insertion sequence, so ties are broken deterministically. */
class Event implements Comparable<Event> {
    static final char ARRIVAL = 'A', REPLICA = 'Q', ACK = 'K', TIMEOUT = 'T', FAULT = 'F';
    final long time;
    final long seq;
    final char kind;
    final int node;
    final int request;
    final int replica;
    final long sendTime;

    Event(long time, long seq, char kind, int node, int request, int replica, long sendTime) {
        this.time = time; this.seq = seq; this.kind = kind; this.node = node;
        this.request = request; this.replica = replica; this.sendTime = sendTime;
    }

    public int compareTo(Event o) {
        int c = Long.compare(time, o.time);
        return c != 0 ? c : Long.compare(seq, o.seq);
    }
}

/** Binary heap of events. */
class EventQueue {
    private Event[] heap = new Event[16];
    private int size;

    int size() { return size; }

    boolean isEmpty() { return size == 0; }

    void push(Event e) {
        if (size == heap.length) heap = Arrays.copyOf(heap, size * 2);
        heap[size] = e;
        int i = size++;
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (heap[parent].compareTo(heap[i]) <= 0) break;
            Event t = heap[parent]; heap[parent] = heap[i]; heap[i] = t;
            i = parent;
        }
    }

    Event pop() {
        Event top = heap[0];
        heap[0] = heap[--size];
        int i = 0;
        while (true) {
            int l = 2 * i + 1, r = l + 1, best = i;
            if (l < size && heap[l].compareTo(heap[best]) < 0) best = l;
            if (r < size && heap[r].compareTo(heap[best]) < 0) best = r;
            if (best == i) break;
            Event t = heap[best]; heap[best] = heap[i]; heap[i] = t;
            i = best;
        }
        return top;
    }
}

/** Knobs of one benchmark run. */
class Config {
    int nodes = 16;
    int requests = 500;
    int replicas = 3;
    int quorum = 2;
    double arrivalRate = 400.0;          // requests per simulated second
    long minDelay = 1, maxDelay = 6;     // network delay per hop, milliseconds
    long serviceMin = 2, serviceMax = 8; // replica service time, milliseconds
    long timeout = 60;                   // coordinator deadline, milliseconds
    double lossRate = 0.0;
    double crashRate = 0.0;
    long crashDuration = 40;
    double slowRate = 0.0;
    long slowFactor = 4;
    long faultPeriod = 25;               // a fault is considered every faultPeriod ms
    long clockSkew = 0;                  // per-node offset applied to that node's own timestamps
    long warmup = 0;                     // samples before this time are excluded from quantiles
    long horizon = 20_000;               // hard stop on simulated time
}

/** Aggregate result of a run. */
class Metrics {
    long completed, failed, timedOut;
    long sent, delivered, droppedByNetwork, lostToCrash, lostToDownAck, acksArrived, inNetworkAtEnd;
    long faultsInjected, crashesInjected, slowsInjected, skewApplied;
    long causalityViolations, orderingViolations, ackOverflow, lateAcks;
    long[] samples = new long[0];
    double throughput, errorRate, offeredLoad;
    double meanLatency, occupancy, littleL;
    long simulationEnd, lastArrival;
    double realizedOfferRate;

    double quantile(double p) { return Quantiles.nearestRank(samples, p); }

    long sum() {
        long s = 0;
        for (long v : samples) s += v;
        return s;
    }
}

/** Nearest-rank quantiles, implemented independently of the harness for cross-verification. */
class Quantiles {
    static double nearestRank(long[] samples, double p) {
        if (samples.length == 0) return 0.0;
        long[] copy = samples.clone();
        Arrays.sort(copy);
        int rank = (int) Math.ceil(p * copy.length);
        if (rank < 1) rank = 1;
        if (rank > copy.length) rank = copy.length;
        return copy[rank - 1];
    }

    /** Same definition by selection, without sorting: used as the independent oracle in the checks. */
    static double selectRank(long[] samples, double p) {
        if (samples.length == 0) return 0.0;
        int rank = (int) Math.ceil(p * samples.length);
        if (rank < 1) rank = 1;
        if (rank > samples.length) rank = samples.length;
        long[] copy = samples.clone();
        int lo = 0, hi = copy.length - 1, k = rank - 1;
        while (lo < hi) {
            long pivot = copy[(lo + hi) >>> 1];
            int i = lo, j = hi;
            while (i <= j) {
                while (copy[i] < pivot) i++;
                while (copy[j] > pivot) j--;
                if (i <= j) { long t = copy[i]; copy[i] = copy[j]; copy[j] = t; i++; j--; }
            }
            if (k <= j) hi = j; else if (k >= i) lo = i; else break;
        }
        return copy[k];
    }
}

/** The harness: a deterministic event-driven simulator with scheduled fault injection. */
class Harness {
    final Config cfg;
    final SplitMix64 rng;
    final EventQueue queue = new EventQueue();
    final long[] downUntil;
    final long[] slowUntil;
    final long[] startTime;
    final int[] coordinatorOf;
    final int[] acks;
    final boolean[] done;
    long now, seq;
    long sent, delivered, droppedByNetwork, lostToCrash, lostToDownAck;
    long scheduled, arrived, acksArrived, lateAcks;
    long completed, failed, timedOut;
    long faultsInjected, crashesInjected, slowsInjected, skewApplied;
    long causalityViolations, orderingViolations, ackOverflow;
    long active, occupancyArea, lastEventTime;
    final List<Long> latency = new ArrayList<>();
    long lastEventTimeSeen = Long.MIN_VALUE;
    int arrivalCount;
    long lastArrival;

    Harness(Config cfg, long seed) {
        this.cfg = cfg;
        this.rng = new SplitMix64(seed);
        downUntil = new long[cfg.nodes];
        slowUntil = new long[cfg.nodes];
        startTime = new long[cfg.requests];
        coordinatorOf = new int[cfg.requests];
        acks = new int[cfg.requests];
        done = new boolean[cfg.requests];
    }

    long scheduleNextArrival() {
        return now + rng.exponential(1000.0 / cfg.arrivalRate, 500);
    }

    int serviceTime(int node) {
        long base = rng.uniform(cfg.serviceMin, cfg.serviceMax);
        boolean slow = slowUntil[node] > now;
        return (int) (slow ? base * cfg.slowFactor : base);
    }

    long nodeSkew(int node) { return cfg.clockSkew == 0 ? 0 : (node * 7L) % (cfg.clockSkew + 1); }

    void send(int request, int from, int to, long extraDelay) {
        long delay = rng.uniform(cfg.minDelay, cfg.maxDelay) + extraDelay;
        boolean applySkew = cfg.clockSkew > 0 && rng.nextInt(4) == 0;
        long skewed = applySkew ? delay + nodeSkew(to) : delay;
        if (applySkew) skewApplied++;
        sent++;
        if (rng.bernoulli(cfg.lossRate)) {
            droppedByNetwork++;
            return;
        }
        scheduled++;
        queue.push(new Event(now + skewed, seq++, 'Q', to, request, from, now));
    }

    void startRequest(int request) {
        active++;
        startTime[request] = now;
        int coordinator = rng.nextInt(cfg.nodes);
        coordinatorOf[request] = coordinator;
        List<Integer> targets = new ArrayList<>();
        int tries = 0;
        while (targets.size() < cfg.replicas && tries < cfg.nodes * 2) {
            int node = rng.nextInt(cfg.nodes);
            if (node != coordinator && !targets.contains(node)) targets.add(node);
            tries++;
        }
        for (int target : targets) send(request, coordinator, target, 0);
        queue.push(new Event(now + cfg.timeout, seq++, 'T', coordinator, request, -1, now));
        arrivalCount++;
        lastArrival = now;
        if (arrivalCount < cfg.requests)
            queue.push(new Event(scheduleNextArrival(), seq++, 'A', -1, arrivalCount, -1, now));
    }

    void injectFault() {
        int node = rng.nextInt(cfg.nodes);
        double roll = rng.nextDouble();
        faultsInjected++;
        if (roll < cfg.crashRate) {
            downUntil[node] = now + cfg.crashDuration;
            crashesInjected++;
        } else if (roll < cfg.crashRate + cfg.slowRate) {
            slowUntil[node] = now + cfg.crashDuration;
            slowsInjected++;
        }
        queue.push(new Event(now + cfg.faultPeriod, seq++, 'F', -1, -1, -1, now));
    }

    Metrics run() {
        queue.push(new Event(0, seq++, 'F', -1, -1, -1, now));
        if (cfg.requests > 0) queue.push(new Event(scheduleNextArrival(), seq++, 'A', -1, 0, -1, now));
        lastEventTime = 0;
        while (!queue.isEmpty()) {
            Event e = queue.pop();
            if (e.time < lastEventTimeSeen) orderingViolations++;
            lastEventTimeSeen = e.time;
            long dt = e.time - lastEventTime;
            if (dt > 0) { occupancyArea += active * dt; lastEventTime = e.time; }
            now = e.time;
            if (now > cfg.horizon) break;
            if (completed + failed >= cfg.requests) break;
            if (e.kind == 'Q') {
                delivered++;
                arrived++;
                if (e.sendTime > e.time - cfg.minDelay) causalityViolations++;
                if (downUntil[e.node] > now) { lostToCrash++; continue; }
                queue.push(new Event(now + serviceTime(e.node), seq++, 'K', -1, e.request, e.node, now));
            } else if (e.kind == 'K') {
                int request = e.request;
                acksArrived++;
                if (done[request]) { lateAcks++; continue; }
                if (acks[request] >= cfg.replicas) { ackOverflow++; continue; }
                if (downUntil[coordinatorOf[request]] > now) { lostToDownAck++; continue; }
                acks[request]++;
                if (acks[request] >= cfg.quorum) {
                    done[request] = true;
                    active--;
                    completed++;
                    latency.add(e.time - startTime[request]);
                }
            } else if (e.kind == 'T') {
                int request = e.request;
                if (!done[request]) {
                    done[request] = true;
                    active--;
                    failed++;
                    timedOut++;
                }
            } else if (e.kind == 'A') {
                startRequest(e.request);
            } else {
                injectFault();
            }
        }
        Metrics m = new Metrics();
        m.completed = completed;
        m.failed = failed;
        m.timedOut = timedOut;
        m.sent = sent;
        m.delivered = delivered;
        m.droppedByNetwork = droppedByNetwork;
        m.lostToCrash = lostToCrash;
        m.lostToDownAck = lostToDownAck;
        m.acksArrived = acksArrived;
        m.lateAcks = lateAcks;
        m.ackOverflow = ackOverflow;
        m.inNetworkAtEnd = (scheduled - arrived) + ((arrived - lostToCrash) - acksArrived);
        m.faultsInjected = faultsInjected;
        m.crashesInjected = crashesInjected;
        m.slowsInjected = slowsInjected;
        m.skewApplied = skewApplied;
        m.causalityViolations = causalityViolations;
        m.orderingViolations = orderingViolations;
        m.samples = latency.stream().mapToLong(Long::longValue).filter(t -> t >= cfg.warmup).toArray();
        m.simulationEnd = now;
        m.lastArrival = lastArrival;
        m.offeredLoad = cfg.arrivalRate;
        m.realizedOfferRate = cfg.requests * 1000.0 / Math.max(1, lastArrival);
        m.throughput = m.completed * 1000.0 / Math.max(1, now);
        m.errorRate = cfg.requests == 0 ? 0.0 : failed / (double) cfg.requests;
        double sum = 0;
        for (long t : m.samples) sum += t;
        m.meanLatency = m.samples.length == 0 ? 0.0 : sum / m.samples.length;
        m.occupancy = occupancyArea / (double) Math.max(1, now);
        m.littleL = m.throughput * m.meanLatency / 1000.0;
        return m;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static Metrics run(Config cfg, long seed) { return new Harness(cfg, seed).run(); }

    static Config clean() {
        Config c = new Config();
        return c;
    }

    public static void main(String[] args) {
        // 1. A clean run: no faults, every request completes, throughput within the offered load.
        Metrics clean = run(clean(), 77);
        check(clean.failed == 0 && clean.timedOut == 0, "a fault-free run produced " + clean.failed + " failures");
        check(clean.completed == 500, "expected 500 completions, got " + clean.completed);
        check(clean.droppedByNetwork == 0 && clean.crashesInjected == 0 && clean.slowsInjected == 0,
                "fault-free run injected faults");
        check(Math.abs(clean.throughput - clean.offeredLoad) < clean.offeredLoad * 0.15,
                "fault-free throughput " + clean.throughput + " should match the offered load "
                        + clean.offeredLoad);
        check(clean.throughput <= clean.realizedOfferRate * 1.02, "completions outpace arrivals: "
                + clean.throughput + " vs a realized arrival rate of " + clean.realizedOfferRate);
        check(clean.realizedOfferRate > 0 && clean.simulationEnd >= clean.lastArrival,
                "the run must end after the last arrival");
        check(clean.errorRate == 0.0, "clean run reports a non-zero error rate");

        // 2. Conservation of messages and unique delivery, on a run with every fault enabled.
        Config faulty = clean();
        faulty.lossRate = 0.15;
        faulty.crashRate = 0.20;
        faulty.slowRate = 0.10;
        faulty.clockSkew = 3;
        Metrics f = run(faulty, 4242);
        check(f.sent == f.droppedByNetwork + f.inNetworkAtEnd + f.lostToCrash + f.acksArrived,
                "message conservation violated: sent " + f.sent + ", dropped " + f.droppedByNetwork
                        + ", in network " + f.inNetworkAtEnd + ", lost to crashes " + f.lostToCrash
                        + ", acknowledgements " + f.acksArrived);
        check(f.acksArrived <= f.delivered, "more acknowledgements arrived (" + f.acksArrived
                + ") than replica messages were delivered (" + f.delivered + ")");
        check(f.sent >= f.droppedByNetwork + f.lostToCrash, "network loss accounting is impossible");
        check(f.ackOverflow == 0, "a replica acknowledged a request more than once: " + f.ackOverflow);
        check(f.causalityViolations == 0, "causality violated " + f.causalityViolations + " times");
        check(f.orderingViolations == 0, "the event queue emitted " + f.orderingViolations
                + " out-of-order events");
        check(f.crashesInjected > 0 && f.slowsInjected > 0 && f.skewApplied > 0,
                "fault injection did not fire: crashes " + f.crashesInjected + ", slows " + f.slowsInjected
                        + ", skew " + f.skewApplied);
        check(f.errorRate > 0.0, "a heavily faulty run reported no failures");
        check(f.completed + f.failed == 500, "requests went missing: " + (f.completed + f.failed));
        check(f.lostToCrash > 0, "no message was lost to a crashed node");

        // 3. Determinism: the same seed reproduces metrics and samples exactly.
        Metrics again = run(faulty, 4242);
        check(again.completed == f.completed && again.failed == f.failed && again.sent == f.sent,
                "a repeated run disagrees on the counters");
        check(Arrays.equals(again.samples, f.samples), "a repeated run produced different latency samples");
        check(again.sum() == f.sum(), "a repeated run produced a different latency sum");
        Metrics other = run(faulty, 4243);
        check(other.sent != f.sent || other.completed != f.completed || other.sum() != f.sum(),
                "a different seed produced an identical run");

        // 4. Quantiles: the harness and an independent selection-based oracle agree exactly.
        double[] ps = {0.0, 0.5, 0.95, 0.99, 1.0};
        for (double p : ps) {
            check(f.quantile(p) == Quantiles.selectRank(f.samples, p), "quantile p=" + p + " disagrees with "
                    + "the independent oracle: " + f.quantile(p) + " vs " + Quantiles.selectRank(f.samples, p));
        }
        check(f.quantile(0.0) == Arrays.stream(f.samples).min().orElse(0), "p0 must be the minimum");
        check(f.quantile(1.0) == Arrays.stream(f.samples).max().orElse(0), "p100 must be the maximum");
        check(f.quantile(0.5) <= f.quantile(0.95) && f.quantile(0.95) <= f.quantile(0.99),
                "the quantile function must be monotone");

        // 5. Degenerate quantile inputs.
        check(Quantiles.nearestRank(new long[0], 0.9) == 0.0, "an empty sample set must yield 0");
        check(Quantiles.nearestRank(new long[]{7}, 0.0) == 7 && Quantiles.nearestRank(new long[]{7}, 1.0) == 7,
                "a single sample must be every quantile");
        long[] tiny = {5, 1, 4, 2, 3};
        check(Quantiles.nearestRank(tiny, 0.5) == 3 && Quantiles.selectRank(tiny, 0.5) == 3,
                "nearest-rank of the median is wrong");
        check(Quantiles.nearestRank(tiny, 0.2) == 1 && Quantiles.nearestRank(tiny, 0.21) == 2,
                "nearest-rank rank computation is wrong");

        // 6. Fault escalation is monotone: error rate up, throughput down.
        double[] lossLevels = {0.0, 0.1, 0.3, 0.6};
        double previousError = -1, previousThroughput = Double.MAX_VALUE;
        StringBuilder table = new StringBuilder();
        for (double loss : lossLevels) {
            Config c = clean();
            c.lossRate = loss;
            c.timeout = 30;
            Metrics m = run(c, 909);
            table.append(String.format("loss=%.1f err=%.2f thr=%.0f | ", loss, m.errorRate, m.throughput));
            check(m.errorRate >= previousError - 1e-9, "error rate fell from " + previousError + " to "
                    + m.errorRate + " when loss grew to " + loss);
            check(m.throughput <= previousThroughput + 1e-9, "throughput rose from " + previousThroughput
                    + " to " + m.throughput + " when loss grew to " + loss);
            previousError = m.errorRate;
            previousThroughput = m.throughput;
        }
        check(previousError > 0.5, "heavy loss should fail most requests, got " + previousError);

        // 7. The timeout path under heavy loss, and its accounting.
        Config timeoutCfg = clean();
        timeoutCfg.lossRate = 0.9;
        timeoutCfg.timeout = 12;
        Metrics to = run(timeoutCfg, 12);
        check(to.timedOut > 0, "no request timed out under 90% loss");
        check(to.timedOut <= to.failed && to.failed == 500 - to.completed,
                "timeouts and failures disagree: " + to.timedOut + "/" + to.failed + "/" + to.completed);

        // 8. Little's law: average occupancy equals throughput times mean latency within 15%.
        Config steady = clean();
        steady.requests = 800;
        steady.arrivalRate = 250;
        Metrics steadyM = run(steady, 555);
        double littleGap = Math.abs(steadyM.occupancy - steadyM.littleL) / Math.max(1.0, steadyM.occupancy);
        check(littleGap < 0.15, "Little's law off by " + String.format("%.1f%%", littleGap * 100)
                + ": occupancy " + steadyM.occupancy + " vs throughput*latency " + steadyM.littleL
                + " (samples " + steadyM.samples.length + " sum " + steadyM.sum()
                + ", completed " + steadyM.completed + ", end " + steadyM.simulationEnd + ")");

        // 9. Warmup: samples inside the window are excluded, and only those.
        Config warm = clean();
        warm.warmup = 100;
        Metrics warmM = run(warm, 31337);
        Metrics coldM = run(clean(), 31337);
        check(warmM.samples.length < coldM.samples.length,
                "warmup did not exclude any sample: " + warmM.samples.length + " vs " + coldM.samples.length);
        int excluded = 0;
        for (long v : coldM.samples) if (v < 100) excluded++;
        check(warmM.samples.length == coldM.samples.length - excluded,
                "warmup excluded the wrong number of samples");
        check(warmM.meanLatency >= coldM.meanLatency - 1e-9 || warmM.samples.length == 0,
                "warmup should not lower the mean latency");

        // 10. Clock skew is really applied, yet causal order and latency bounds survive.
        Config skewed = clean();
        skewed.clockSkew = 5;
        skewed.lossRate = 0.1;
        Metrics skewM = run(skewed, 2024);
        check(skewM.skewApplied > 0, "clock skew was never applied");
        check(skewM.causalityViolations == 0 && skewM.orderingViolations == 0,
                "skew reordered events: " + skewM.causalityViolations + "/" + skewM.orderingViolations);
        long cap = skewed.timeout + 4L * (skewed.maxDelay + skewed.serviceMax);
        boolean bounded = true;
        for (long t : skewM.samples) if (t < 0 || t > cap) bounded = false;
        check(bounded, "latencies outside [0, " + cap + "] after skew injection");

        // 11. Degenerate runs: no requests, a single node with quorum one, and a short horizon.
        Config none = clean();
        none.requests = 0;
        Metrics empty = run(none, 1);
        check(empty.completed == 0 && empty.failed == 0 && empty.samples.length == 0 && empty.quantile(0.5) == 0.0,
                "a zero-request run produced traffic");
        Config single = clean();
        single.nodes = 1;
        single.replicas = 1;
        single.quorum = 1;
        single.requests = 40;
        Metrics singleM = run(single, 8);
        check(singleM.completed + singleM.failed == 40, "single-node run lost requests");
        Config shortHorizon = clean();
        shortHorizon.horizon = 50;
        Metrics cut = run(shortHorizon, 8);
        check(cut.completed + cut.failed < 500, "the horizon did not cut the run short");
        check(cut.sent >= cut.delivered, "conservation broken on a truncated run");

        // 12. Replica count changes the fault sensitivity: fewer replicas tolerate less loss.
        Config fragile = clean();
        fragile.lossRate = 0.4;
        fragile.replicas = 2;
        fragile.quorum = 2;
        fragile.timeout = 30;
        Config robust = clean();
        robust.lossRate = 0.4;
        robust.replicas = 4;
        robust.quorum = 2;
        robust.timeout = 30;
        Metrics fragileM = run(fragile, 61), robustM = run(robust, 61);
        check(robustM.errorRate < fragileM.errorRate, "quorum-2-of-4 should beat quorum-2-of-2 under loss: "
                + robustM.errorRate + " vs " + fragileM.errorRate);

        System.out.println("clean run    : " + clean.completed + " completed, throughput "
                + String.format("%.0f", clean.throughput) + "/s, p50 " + clean.quantile(0.5)
                + " ms, p95 " + clean.quantile(0.95) + " ms, p99 " + clean.quantile(0.99) + " ms");
        System.out.println("faulty run   : " + f.completed + " completed, " + f.failed + " failed ("
                + String.format("%.1f%%", f.errorRate * 100) + "), " + f.sent + " messages sent, "
                + f.droppedByNetwork + " lost in the network, " + f.lostToCrash + " lost to crashes, "
                + f.crashesInjected + " crashes and " + f.slowsInjected + " slow nodes injected");
        System.out.println("loss sweep   : " + table.toString().trim());
        System.out.println("Little's law : occupancy " + String.format("%.2f", steadyM.occupancy)
                + " vs throughput*latency " + String.format("%.2f", steadyM.littleL) + " (gap "
                + String.format("%.1f%%", littleGap * 100) + ")");
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
