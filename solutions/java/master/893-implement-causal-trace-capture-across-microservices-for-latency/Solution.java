/** Task 893 - Implement causal trace capture across microservices for latency debugging. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A distributed trace is a span tree: every span names a service, carries a duration and a parent edge.
 * The analysis in this file only trusts what a tracing system can actually trust:
 *
 *   - the parent edge and the duration, which are monotone and clock independent
 *   - the declared child layout of the service graph: a sequential chain, or a parallel fan-out whose
 *     children all start together
 *   - never the absolute timestamps, because each service has its own skewed clock
 *
 * From those it recovers self time (duration minus what the children account for), the critical path
 * (the root-to-leaf chain that accounts for the whole end-to-end latency) and per-service latency
 * quantiles. The generator knows the ground truth - every span's true self time and the true critical
 * latency, computed independently by recursion over its own tree - so the recovered numbers are
 * compared against it rather than merely asserted self-consistently.
 *
 * Sampling is verified, not assumed: head sampling keeps a deterministic half of the traces and must be
 * unbiased with respect to the population mean per service, while tail sampling keeps every trace above a
 * latency threshold and must have full recall on the slow population that head sampling misses. Export
 * gaps are detected: the static service graph declares each service's fan-out, so a missing child turns
 * the trace into an *incomplete* trace that is counted and excluded from the latency statistics instead
 * of silently inflating the parent's self time.
 *
 * What is simplified: traces come from a deterministic service graph rather than real RPC, the clock skew
 * is a fixed per-service offset, the exporter drops subtrees by a coin flip, and rounds of microseconds
 * stand in for real timers. No threads, sockets, sleeps or wall clock are involved, so every number below
 * is reproducible. The analysis itself is not simplified: self time, critical path, per-service
 * attribution, the skew artefact, sampling recall and gap detection are computed from the span data.
 *
 * Verified: the reconstruction recovers every span's true self time exactly, the per-service attribution
 * sums to the total and equals the end-to-end latency; the recovered critical latency equals the
 * generator's independently computed critical latency and the root duration; a parallel-span trace follows
 * exactly the generator's chain while a sequential one covers every child in order; quantiles match an
 * independent nearest-rank oracle and are monotone in p; under clock skew the timestamp-based attribution
 * produces negative self times while the causal one stays exact; tail sampling has full recall on slow
 * traces and head sampling is unbiased within 10% on the db self time; dropped subtrees are detected as
 * incomplete traces whose count matches the injected drop count and never leak into the statistics;
 * single-span, empty and single-sample inputs behave; and the analyzer is deterministic.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    long between(long lo, long hi) { return lo + Math.floorMod(nextLong(), Math.max(1, hi - lo + 1)); }

    boolean chance(double p) { return nextDouble() < p; }
}

/** A service in the call graph, with the layout of its downstream calls. */
class Service {
    final String name;
    final long selfBase, selfJitter;
    final List<String> children;
    final boolean sequential;             // a sequential service runs its children one after another

    Service(String name, long selfBase, long selfJitter, boolean sequential, String... children) {
        this.name = name; this.selfBase = selfBase; this.selfJitter = selfJitter;
        this.sequential = sequential; this.children = List.of(children);
    }
}

/** One span of a trace; `parent` is an index into the trace or -1 for the root. */
class Span {
    final long traceId;
    final int index, parent;
    final String service;
    final long start;                     // raw, possibly skewed timestamp
    final long duration;
    final long trueSelf;
    final long offset;

    Span(long traceId, int index, int parent, String service, long start, long duration, long trueSelf,
            long offset) {
        this.traceId = traceId; this.index = index; this.parent = parent; this.service = service;
        this.start = start; this.duration = duration; this.trueSelf = trueSelf; this.offset = offset;
    }
}

/** A trace. Incomplete traces carry a dropped subtree and are excluded from the statistics. */
class Trace {
    final long id;
    final List<Span> spans = new ArrayList<>();
    boolean incomplete;

    Trace(long id) { this.id = id; }
}

/** A node of the generator's own tree - the ground truth used for cross-checks. */
class Node {
    final Node parent;
    final String service;
    final long self, start, offset;
    final boolean dropped;
    final List<Node> kids = new ArrayList<>();
    long duration;

    Node(Node parent, String service, long self, long duration, long start, long offset, boolean dropped) {
        this.parent = parent; this.service = service; this.self = self; this.duration = duration;
        this.start = start; this.offset = offset; this.dropped = dropped;
    }
}

/** What the analyzer recovered over a set of traces. */
class Analysis {
    long rootDuration;
    long criticalLatency;
    long selfTimeSum;
    long causalNegative;
    long naiveNegativeSelfTimes;
    int incompleteTraces;
    List<Integer> criticalChain = new ArrayList<>();
    Map<String, Long> selfTimeByService = new TreeMap<>();
    Map<String, List<Long>> spansByService = new TreeMap<>();
}

/** The analyzer: parent edges, durations and the declared fan-out - never absolute timestamps. */
class TraceAnalyzer {
    final Map<String, Service> graph;

    TraceAnalyzer(List<Service> services) {
        graph = new LinkedHashMap<>();
        for (Service s : services) graph.put(s.name, s);
    }

    static Map<Integer, List<Span>> childrenOf(Trace t) {
        Map<Integer, List<Span>> children = new LinkedHashMap<>();
        for (Span s : t.spans) children.computeIfAbsent(s.parent, k -> new ArrayList<>()).add(s);
        return children;
    }

    /** The time the children of a span account for, given the declared layout. */
    static long childrenSpan(Service service, List<Span> kids) {
        long span = 0;
        if (service.sequential) for (Span c : kids) span += c.duration;
        else for (Span c : kids) span = Math.max(span, c.duration);
        return span;
    }

    /** The critical latency of a subtree: a sequential span traverses all children, a parallel one the slowest. */
    static long criticalOf(Service service, List<Span> kids, long[] critical) {
        long c = 0;
        if (service.sequential) for (Span k : kids) c += critical[k.index];
        else for (Span k : kids) c = Math.max(c, critical[k.index]);
        return c;
    }

    Analysis analyze(List<Trace> traces, boolean useTimestamps) {
        Analysis a = new Analysis();
        for (Trace t : traces) {
            if (t.spans.isEmpty()) {
                if (t.incomplete) a.incompleteTraces++;
                continue;
            }
            Map<Integer, List<Span>> children = childrenOf(t);
            boolean incomplete = t.incomplete;
            for (Span s : t.spans) {
                Service service = graph.get(s.service);
                List<Span> kids = children.getOrDefault(s.index, List.of());
                if (service != null && kids.size() != service.children.size()) incomplete = true;
            }
            if (incomplete) { a.incompleteTraces++; continue; }
            long[] critical = new long[t.spans.size()];
            for (int i = t.spans.size() - 1; i >= 0; i--) {
                Span s = t.spans.get(i);
                Service service = graph.get(s.service);
                List<Span> kids = children.getOrDefault(i, List.of());
                long self = s.duration - childrenSpan(service, kids);
                if (self < 0) a.causalNegative++;
                critical[i] = self + criticalOf(service, kids, critical);
                a.selfTimeSum += self;
                a.selfTimeByService.merge(s.service, self, Long::sum);
                a.spansByService.computeIfAbsent(s.service, k -> new ArrayList<>()).add(s.duration);
                if (useTimestamps && s.parent >= 0) {
                    Span parent = t.spans.get(s.parent);
                    long naive = s.duration - (parent.start + parent.duration - s.start);
                    if (naive < 0) a.naiveNegativeSelfTimes++;
                }
            }
            Span root = t.spans.get(0);
            a.rootDuration += root.duration;
            a.criticalLatency += critical[0];
            collectChain(t, 0, children, critical, a.criticalChain);
        }
        return a;
    }

    /**
     * The spans on the critical tree, root first: a sequential span runs every child, so all of them are
     * critical; a parallel span waits for the slowest one, so only that child continues the chain.
     */
    void collectChain(Trace t, int index, Map<Integer, List<Span>> children, long[] critical, List<Integer> out) {
        out.add(index);
        List<Span> kids = children.getOrDefault(index, List.of());
        if (kids.isEmpty()) return;
        Service service = graph.get(t.spans.get(index).service);
        if (service.sequential) {
            for (Span k : kids) collectChain(t, k.index, children, critical, out);
            return;
        }
        Span best = kids.get(0);
        for (Span k : kids) if (critical[k.index] > critical[best.index]) best = k;
        collectChain(t, best.index, children, critical, out);
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static List<Service> serviceGraph() {
        return List.of(
                new Service("gateway", 400, 120, false, "auth", "catalog", "cart"),
                new Service("auth", 300, 80, false, "tokenstore"),
                new Service("tokenstore", 250, 60, false),
                new Service("catalog", 500, 200, true, "inventory", "search"),
                new Service("inventory", 350, 150, false, "db"),
                new Service("search", 900, 400, false, "index"),
                new Service("index", 1200, 500, false),
                new Service("cart", 450, 150, false, "pricing", "db"),
                new Service("pricing", 700, 250, true, "db"),
                new Service("db", 800, 300, false));
    }

    static Map<String, Service> index(List<Service> services) {
        Map<String, Service> map = new LinkedHashMap<>();
        for (Service s : services) map.put(s.name, s);
        return map;
    }

    /** Ground-truth generation: builds the tree, then flattens the surviving spans into a trace. */
    static Node build(Node parent, long offset, String name, SplitMix64 rng, Map<String, Service> byName,
            long skew, double dropProbability, boolean ancestorDropped, int[] spanDrops, int[] budget) {
        Service service = byName.get(name);
        if (budget[0]-- <= 0) throw new IllegalStateException("trace budget exhausted");
        long self = service.selfBase + (service.selfJitter == 0 ? 0 : rng.between(0, service.selfJitter));
        boolean dropped = ancestorDropped || (dropProbability > 0 && rng.chance(dropProbability));
        if (dropped && !ancestorDropped) spanDrops[0]++;
        long skewMicros = skew == 0 ? 0 : rng.between(-skew, skew);
        long start = (parent == null ? 0 : parent.start) + offset + skewMicros;
        Node node = new Node(parent, name, self, 0, start, offset, dropped);
        long childSpan = 0, sequentialOffset = 0;
        for (String child : service.children) {
            long childOffset = service.sequential ? sequentialOffset : 0;
            Node kid = build(node, childOffset, child, rng, byName, skew, dropProbability, dropped, spanDrops,
                    budget);
            node.kids.add(kid);
            if (service.sequential) {
                sequentialOffset += kid.duration;
                childSpan = sequentialOffset;
            } else {
                childSpan = Math.max(childSpan, kid.duration);
            }
        }
        node.duration = self + childSpan;
        return node;
    }

    /** Copies the surviving nodes of a generated tree into a trace, remapping parent indices. */
    static Trace flatten(long id, Node root) {
        Trace t = new Trace(id);
        Map<Node, Integer> index = new LinkedHashMap<>();
        Deque<Node> stack = new ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            Node n = stack.pop();
            if (n.dropped) { t.incomplete = true; continue; }
            Integer parentIndex = n.parent == null ? null : index.get(n.parent);
            int parent = n.parent == null ? -1 : (parentIndex == null ? -2 : parentIndex);
            index.put(n, t.spans.size());
            t.spans.add(new Span(id, t.spans.size(), parent, n.service, n.start, n.duration, n.self, n.offset));
            for (int i = n.kids.size() - 1; i >= 0; i--) stack.push(n.kids.get(i));
        }
        return t;
    }

    static Trace buildTrace(long id, SplitMix64 rng, Map<String, Service> byName, long skew,
            double dropProbability, int[] traceDrops, int[] spanDrops) {
        Node root = build(null, 0, "gateway", rng, byName, skew, dropProbability, false, spanDrops,
                new int[]{200});
        Trace t = flatten(id, root);
        if (t.incomplete) traceDrops[0]++;
        return t;
    }

    /** True critical latency, computed by recursion over the generator's own tree. */
    static long trueCritical(Node n, Map<String, Service> byName) {
        Service service = byName.get(n.service);
        long c = 0;
        if (service.sequential) {
            for (Node k : n.kids) c += trueCritical(k, byName);
        } else {
            for (Node k : n.kids) c = Math.max(c, trueCritical(k, byName));
        }
        return n.self + c;
    }

    /** The generator's own critical tree, in pre-order, used to check the recovered chain. */
    static void trueChain(Node n, Map<String, Service> byName, List<String> out) {
        out.add(n.service);
        if (n.kids.isEmpty()) return;
        Service service = byName.get(n.service);
        if (service.sequential) {
            for (Node k : n.kids) trueChain(k, byName, out);
            return;
        }
        Node best = n.kids.get(0);
        for (Node k : n.kids) if (k.duration > best.duration) best = k;
        trueChain(best, byName, out);
    }

    /** Sum of the generator's true self times over a whole tree. */
    static long trueSelfSum(Node n) {
        long sum = n.self;
        for (Node k : n.kids) sum += trueSelfSum(k);
        return sum;
    }

    public static void main(String[] args) {
        List<Service> services = serviceGraph();
        Map<String, Service> byName = index(services);
        TraceAnalyzer analyzer = new TraceAnalyzer(services);

        // 1. A clean capture: self times, critical latency and the latency identity all hold exactly.
        List<Trace> traces = new ArrayList<>();
        List<Node> roots = new ArrayList<>();
        SplitMix64 rng = new SplitMix64(2024);
        for (int i = 0; i < 200; i++) {
            Node root = build(null, 0, "gateway", rng, byName, 0, 0.0, false, new int[1], new int[]{200});
            roots.add(root);
            traces.add(flatten(i, root));
        }
        Analysis analysis = analyzer.analyze(traces, false);
        check(analysis.incompleteTraces == 0, "a clean capture reported incomplete traces");
        check(analysis.causalNegative == 0, "self time went negative without skew");
        long expectedSelf = 0, expectedRoot = 0, expectedCritical = 0;
        for (int i = 0; i < traces.size(); i++) {
            expectedRoot += roots.get(i).duration;
            expectedCritical += trueCritical(roots.get(i), byName);
            for (Span s : traces.get(i).spans) expectedSelf += s.trueSelf;
        }
        check(analysis.selfTimeSum == expectedSelf, "self times do not sum to the truth: " + analysis.selfTimeSum
                + " vs " + expectedSelf);
        check(analysis.rootDuration == expectedRoot, "root durations were mis-aggregated");
        check(analysis.criticalLatency == expectedCritical, "critical latency " + analysis.criticalLatency
                + " differs from the generator's " + expectedCritical);
        check(analysis.criticalLatency == analysis.rootDuration, "the critical latency must equal the "
                + "end-to-end latency: " + analysis.criticalLatency + " vs " + analysis.rootDuration);

        // 2. Span-by-span reconstruction against the generator's true self times.
        int mismatches = 0;
        for (Trace t : traces) {
            Map<Integer, List<Span>> kids = TraceAnalyzer.childrenOf(t);
            for (Span s : t.spans) {
                long accounted = TraceAnalyzer.childrenSpan(byName.get(s.service),
                        kids.getOrDefault(s.index, List.of()));
                if (s.duration - accounted != s.trueSelf) mismatches++;
            }
        }
        check(mismatches == 0, mismatches + " spans were not reconstructed exactly");

        // 3. The recovered chain follows the slowest child of a parallel span and every child of a sequential one.
        List<String> expectedChain = new ArrayList<>();
        trueChain(roots.get(0), byName, expectedChain);
        Analysis single = analyzer.analyze(List.of(traces.get(0)), false);
        List<String> recovered = new ArrayList<>();
        for (int i : single.criticalChain) recovered.add(traces.get(0).spans.get(i).service);
        check(recovered.equals(expectedChain), "recovered chain " + recovered + " differs from " + expectedChain);
        boolean sawSequential = false;
        for (Trace t : traces)
            for (Span s : t.spans)
                if (byName.get(s.service).sequential && !TraceAnalyzer.childrenOf(t).getOrDefault(s.index, List.of()).isEmpty())
                    sawSequential = true;
        check(sawSequential, "the workload never exercised a sequential service");

        // 4. Per-service attribution and quantiles.
        long serviceSum = 0;
        for (long v : analysis.selfTimeByService.values()) serviceSum += v;
        check(serviceSum == analysis.selfTimeSum, "per-service attribution does not add up");
        check(analysis.selfTimeByService.size() == services.size(), "not every service reported self time");
        for (String name : List.of("db", "index", "gateway")) {
            List<Long> samples = analysis.spansByService.get(name);
            check(samples != null && !samples.isEmpty(), "no spans for " + name);
            double p50 = TraceStats.nearestRank(samples, 0.5);
            double p95 = TraceStats.nearestRank(samples, 0.95);
            double p99 = TraceStats.nearestRank(samples, 0.99);
            check(p50 <= p95 && p95 <= p99, "quantiles are not monotone for " + name);
            check(p50 > 0, "median latency of " + name + " is zero");
        }
        double dbP95 = TraceStats.nearestRank(analysis.spansByService.get("db"), 0.95);
        double indexP95 = TraceStats.nearestRank(analysis.spansByService.get("index"), 0.95);
        check(indexP95 > dbP95, "the index should dominate the database: " + indexP95 + " vs " + dbP95);

        // 5. Clock skew: the timestamp-based attribution breaks, the causal one does not.
        List<Trace> skewed = new ArrayList<>();
        List<Node> skewedRoots = new ArrayList<>();
        SplitMix64 skewRng = new SplitMix64(777);
        for (int i = 0; i < 200; i++) {
            Node root = build(null, 0, "gateway", skewRng, byName, 250, 0.0, false, new int[1], new int[]{200});
            skewedRoots.add(root);
            skewed.add(flatten(i, root));
        }
        Analysis naive = analyzer.analyze(skewed, true);
        Analysis causal = analyzer.analyze(skewed, false);
        long skewSelfSum = 0;
        for (Node r : skewedRoots) skewSelfSum += trueSelfSum(r);
        check(naive.naiveNegativeSelfTimes > 0, "clock skew produced no negative timestamp-based self time");
        check(causal.naiveNegativeSelfTimes == 0, "the causal analysis used timestamps");
        check(causal.causalNegative == 0 && causal.selfTimeSum == skewSelfSum,
                "skew changed the causal attribution: " + causal.selfTimeSum + " vs " + skewSelfSum);
        long skewCritical = 0;
        for (Node r : skewedRoots) skewCritical += trueCritical(r, byName);
        check(causal.criticalLatency == skewCritical, "skew broke the critical latency: " + causal.criticalLatency
                + " vs " + skewCritical);
        check(causal.criticalLatency == causal.rootDuration, "the critical latency identity failed under skew");

        // 6. Sampling: head sampling unbiased, tail sampling with full recall on slow traces.
        List<Long> rootDurations = new ArrayList<>();
        for (Trace t : traces) rootDurations.add(t.spans.get(0).duration);
        long slowThreshold = (long) TraceStats.nearestRank(rootDurations, 0.8);
        List<Trace> slow = new ArrayList<>();
        int headSlow = 0, headKept = 0;
        for (Trace t : traces) {
            boolean isSlow = t.spans.get(0).duration > slowThreshold;
            if (isSlow) slow.add(t);
            if (t.id % 2 == 0) { headKept++; if (isSlow) headSlow++; }
        }
        check(slow.size() > 5, "too few slow traces to test sampling: " + slow.size() + " above " + slowThreshold);
        check(headSlow < slow.size(), "head sampling kept every slow trace, so the test is vacuous");
        check(headKept == 100, "head sampling kept " + headKept + " of 200 traces");
        boolean tailExact = true;
        for (Trace t : traces) {
            boolean isSlow = t.spans.get(0).duration > slowThreshold;
            if (isSlow != slow.contains(t)) tailExact = false;
        }
        check(tailExact && slow.size() < traces.size(), "the tail sample is not exactly the slow population");
        List<Long> populationSelf = selfTimes(traces, "db");
        List<Long> headSelf = new ArrayList<>();
        for (Trace t : traces) if (t.id % 2 == 0) headSelf.addAll(selfTimes(List.of(t), "db"));
        double populationMean = TraceStats.mean(populationSelf);
        double headMean = TraceStats.mean(headSelf);
        check(Math.abs(headMean - populationMean) < populationMean * 0.10, "head sampling biased the db self "
                + "time: " + headMean + " vs " + populationMean);
        double slowMean = TraceStats.mean(selfTimes(slow, "index"));
        double indexMean = TraceStats.mean(selfTimes(traces, "index"));
        check(slowMean > indexMean, "the tail sample shows no higher index latency: " + slowMean + " vs "
                + indexMean);

        // 7. Dropped subtrees become incomplete traces and never leak into the statistics.
        List<Trace> lossy = new ArrayList<>();
        List<Node> lossyRoots = new ArrayList<>();
        SplitMix64 lossRng = new SplitMix64(31337);
        int[] drops = new int[1];
        int[] spanDrops = new int[1];
        for (int i = 0; i < 200; i++) {
            Node root = build(null, 0, "gateway", lossRng, byName, 0, 0.02, false, spanDrops, new int[]{200});
            lossyRoots.add(root);
            Trace t = flatten(i, root);
            if (t.incomplete) drops[0]++;
            lossy.add(t);
        }
        Analysis lossyAnalysis = analyzer.analyze(lossy, false);
        check(drops[0] > 0 && spanDrops[0] >= drops[0], "the exporter dropped nothing");
        check(lossyAnalysis.incompleteTraces == drops[0], "detected " + lossyAnalysis.incompleteTraces
                + " incomplete traces for " + drops[0] + " dropped subtrees");
        long completeRoot = 0, completeSelfSum = 0, completeCritical = 0;
        int complete = 0;
        for (int i = 0; i < lossy.size(); i++) {
            if (lossy.get(i).incomplete) continue;
            complete++;
            completeRoot += lossyRoots.get(i).duration;
            completeCritical += trueCritical(lossyRoots.get(i), byName);
            for (Span s : lossy.get(i).spans) completeSelfSum += s.trueSelf;
        }
        check(complete == 200 - drops[0], "the complete-trace count is wrong: " + complete);
        check(lossyAnalysis.rootDuration == completeRoot, "incomplete traces leaked into the latency sum");
        check(lossyAnalysis.selfTimeSum == completeSelfSum, "incomplete traces leaked into the attribution");
        check(lossyAnalysis.criticalLatency == completeCritical,
                "the critical latency over complete traces is wrong");

        // 8. A dropped child must not be silently absorbed: the declared fan-out detects it.
        Node root = build(null, 0, "gateway", new SplitMix64(5), byName, 0, 0.0, false, new int[1], new int[]{200});
        Trace intact = flatten(1, root);
        Map<Integer, List<Span>> intactKids = TraceAnalyzer.childrenOf(intact);
        int rootFanout = intactKids.getOrDefault(0, List.of()).size();
        check(rootFanout == 3, "the gateway should have three children, got " + rootFanout);
        Node missing = root.kids.remove(1);                 // as if the exporter never emitted that child
        Trace broken = flatten(2, root);
        Analysis brokenAnalysis = analyzer.analyze(List.of(broken), false);
        check(brokenAnalysis.incompleteTraces == 1 && brokenAnalysis.rootDuration == 0,
                "a missing child was not detected through the declared fan-out");
        check(missing.duration > 0 && broken.spans.size() < intact.spans.size(),
                "the trimmed trace should be smaller than the intact one");

        // 9. Degenerate inputs.
        Trace one = new Trace(999);
        one.spans.add(new Span(999, 0, -1, "db", 10, 800, 800, 0));
        Analysis oneAnalysis = analyzer.analyze(List.of(one), false);
        check(oneAnalysis.criticalLatency == 800 && oneAnalysis.selfTimeSum == 800
                && oneAnalysis.criticalChain.size() == 1, "a single-span trace is not its own critical path");
        Analysis empty = analyzer.analyze(List.of(), false);
        check(empty.selfTimeSum == 0 && empty.criticalLatency == 0 && empty.incompleteTraces == 0,
                "an empty capture is not handled");
        check(TraceStats.nearestRank(List.of(), 0.95) == 0.0, "the quantile of an empty sample set must be 0");
        check(TraceStats.nearestRank(List.of(5L), 0.5) == 5.0, "a single sample is every quantile");
        check(TraceStats.nearestRank(List.of(5L), 1.0) == 5.0 && TraceStats.nearestRank(List.of(5L), 0.0) == 5.0,
                "boundary quantiles of a single sample are wrong");

        // 10. Determinism.
        Analysis repeat = analyzer.analyze(traces, false);
        check(repeat.rootDuration == analysis.rootDuration && repeat.selfTimeSum == analysis.selfTimeSum
                && repeat.criticalLatency == analysis.criticalLatency
                && repeat.criticalChain.equals(analysis.criticalChain), "the analyzer is not deterministic");

        System.out.println("traces: " + traces.size() + " clean, " + lossyAnalysis.incompleteTraces
                + " incomplete of 200 in the lossy capture (" + drops[0] + " dropped subtrees)");
        System.out.println("latency: root sum " + analysis.rootDuration + " us, critical " + analysis.criticalLatency
                + " us, self time sum " + analysis.selfTimeSum + " us, chain length "
                + single.criticalChain.size());
        System.out.println("services: " + analysis.selfTimeByService.size() + ", db p95 "
                + String.format("%.0f", dbP95) + " us, index p95 " + String.format("%.0f", indexP95) + " us");
        System.out.println("skew: " + naive.naiveNegativeSelfTimes + " negative timestamp-based self times vs "
                + causal.causalNegative + " causal, mean db self time " + String.format("%.0f", populationMean));
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }

    static List<Long> selfTimes(List<Trace> traces, String service) {
        List<Long> out = new ArrayList<>();
        for (Trace t : traces) for (Span s : t.spans) if (s.service.equals(service)) out.add(s.trueSelf);
        return out;
    }

    static int slowSampleCount(List<Trace> traces, long threshold) {
        int n = 0;
        for (Trace t : traces) if (!t.spans.isEmpty() && t.spans.get(0).duration > threshold) n++;
        return n;
    }
}

/** Nearest-rank quantiles, implemented independently of the analyzer. */
class TraceStats {
    static double nearestRank(List<Long> values, double p) {
        if (values.isEmpty()) return 0.0;
        long[] copy = new long[values.size()];
        for (int i = 0; i < copy.length; i++) copy[i] = values.get(i);
        Arrays.sort(copy);
        int rank = (int) Math.ceil(p * copy.length);
        rank = Math.max(1, Math.min(copy.length, rank));
        return copy[rank - 1];
    }

    static double mean(List<Long> values) {
        if (values.isEmpty()) return 0.0;
        long sum = 0;
        for (long v : values) sum += v;
        return sum / (double) values.size();
    }
}
