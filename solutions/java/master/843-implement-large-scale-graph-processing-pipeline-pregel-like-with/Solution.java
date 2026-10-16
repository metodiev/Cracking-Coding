/** Task 843 - Implement large-scale graph processing pipeline (Pregel-like) with fault tolerance. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The engine is a vertex-centric BSP (Pregel) runtime:
 *
 *   superstep   - every active vertex runs the same program, reads the messages that arrived in the
 *                 previous superstep, may change its own value, sends messages to its neighbours and may
 *                 contribute to an aggregator. A vertex is active in the next superstep exactly when it
 *                 received at least one message, so the computation terminates when no messages are left.
 *   combiner    - an optional associative, commutative merge for messages targeted at the same vertex
 *                 (sum for PageRank, min for BFS/SSSP), applied before delivery.
 *   aggregator  - a global reduce over the values contributed during a superstep, visible to the driver.
 *   checkpoints - the engine snapshots the vertex values, the pending mailboxes and the aggregator every
 *                 `checkpointInterval` supersteps. A worker failure at superstep k discards everything and
 *                 replays from the newest checkpoint, which is deterministic because the message stream is
 *                 a pure function of the restored state.
 *
 * Four algorithms are implemented on top of it - BFS, weighted SSSP, connected components and PageRank -
 * and each is compared against an independent reference implementation (a plain queue BFS, Dijkstra, a
 * union-find, and a dense power iteration).
 *
 * Simplifications, stated honestly: one in-process process instead of a cluster of workers; "machines" are
 * not modelled beyond the failure/rollback schedule, so there is no message serialisation and no network.
 * Vertices are numbered 0..n-1 and messages carry a single double. Wall clock, threads and randomness are
 * absent; the graphs are small (tens of vertices) because the point is the engine's semantics.
 *
 * Invariants verified below. Correctness: all four algorithms match their reference implementation, on
 * connected and disconnected graphs. Termination and liveness: BFS finishes in eccentricity + 1 rounds,
 * the message count is bounded, no vertex is executed twice per superstep, and every vertex that halts
 * produces no further messages. Fault tolerance: a failure at superstep k with rollback to the newest
 * checkpoint reproduces the exact same result, and the replayed work is counted. Cost: supersteps,
 * delivered messages and the combiner's saving are measured.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** Compressed adjacency: neighbours of v are head[offset[v] .. offset[v+1]). */
final class Graph {
    final int n;
    final int[] offset;
    final int[] head;
    final double[] weight;

    Graph(int n, int[] offset, int[] head, double[] weight) {
        this.n = n;
        this.offset = offset;
        this.head = head;
        this.weight = weight;
    }

    static Graph of(int n, int[][] edges) {
        int[] degree = new int[n];
        for (int[] e : edges) {
            degree[e[0]]++;
            if (e[1] != e[0]) degree[e[1]]++;
        }
        int[] offset = new int[n + 1];
        for (int v = 0; v < n; v++) offset[v + 1] = offset[v] + degree[v];
        int[] head = new int[offset[n]];
        double[] weight = new double[offset[n]];
        int[] cursor = offset.clone();
        for (int[] e : edges) {
            head[cursor[e[0]]] = e[1];
            weight[cursor[e[0]]++] = e.length > 2 ? e[2] : 1;
            if (e[1] != e[0]) {
                head[cursor[e[1]]] = e[0];
                weight[cursor[e[1]]++] = e.length > 2 ? e[2] : 1;
            }
        }
        return new Graph(n, offset, head, weight);
    }

    int degree(int v) { return offset[v + 1] - offset[v]; }

    int neighbour(int v, int index) { return head[offset[v] + index]; }

    double edgeWeight(int v, int index) { return weight[offset[v] + index]; }

    int edgeCount() { return offset[n] / 2; }
}

/** What a vertex program may do in one superstep. */
final class Outbox {
    double newValue;
    boolean valueChanged;
    final List<int[]> targets = new ArrayList<>();
    final List<Double> values = new ArrayList<>();
    double aggregateContribution;
    boolean contributes;

    void setValue(double value) {
        newValue = value;
        valueChanged = true;
    }

    void send(int target, double value) {
        targets.add(new int[]{target});
        values.add(value);
    }

    void contribute(double value) {
        aggregateContribution = value;
        contributes = true;
    }
}

/** The vertex contract: read the inbox, update the value, send messages, contribute to the aggregator. */
interface VertexProgram {
    void compute(int vertex, double value, List<Double> inbox, Graph graph, Outbox outbox);
}

/** Merge for messages aimed at the same vertex (associative and commutative). */
interface MessageCombiner {
    double combine(double first, double second);

    static MessageCombiner sum() { return (a, b) -> a + b; }

    static MessageCombiner min() { return Math::min; }
}

/** A snapshot of the whole computation. */
final class Checkpoint {
    final int superstep;
    final double[] values;
    final List<List<Double>> mailboxes;
    final boolean[] active;
    final double aggregator;

    Checkpoint(int superstep, double[] values, List<List<Double>> mailboxes, boolean[] active,
            double aggregator) {
        this.superstep = superstep;
        this.values = values;
        this.mailboxes = mailboxes;
        this.active = active;
        this.aggregator = aggregator;
    }
}

/** The result of one run, with the measured cost and the recovery bookkeeping. */
final class PregelResult {
    final double[] values;
    final int supersteps;
    final int sentMessages;
    final int deliveredMessages;
    final double aggregator;
    final int vertexExecutions;
    final int replayedExecutions;
    final int checkpointCount;
    final int recoveredAt;

    PregelResult(double[] values, int supersteps, int sentMessages, int deliveredMessages,
            double aggregator, int vertexExecutions, int replayedExecutions, int checkpointCount,
            int recoveredAt) {
        this.values = values;
        this.supersteps = supersteps;
        this.sentMessages = sentMessages;
        this.deliveredMessages = deliveredMessages;
        this.aggregator = aggregator;
        this.vertexExecutions = vertexExecutions;
        this.replayedExecutions = replayedExecutions;
        this.checkpointCount = checkpointCount;
        this.recoveredAt = recoveredAt;
    }
}

/** The BSP runtime: mailboxes, supersteps, combiner, aggregator, checkpoints and rollback. */
final class PregelEngine {
    final Graph graph;
    final double[] values;
    List<List<Double>> mailboxes = new ArrayList<>();
    boolean[] active;
    final int checkpointInterval;
    final int failAtSuperstep;
    MessageCombiner combiner;
    double aggregator;
    List<Checkpoint> checkpoints = new ArrayList<>();
    int supersteps, sent, delivered, executions, replayed, recoveredAt;
    final int maxSupersteps;
    boolean failureInjected;

    PregelEngine(Graph graph, double initialValue, int checkpointInterval, int failAtSuperstep) {
        this(graph, initialValue, checkpointInterval, failAtSuperstep, 0);
    }

    PregelEngine(Graph graph, double initialValue, int checkpointInterval, int failAtSuperstep,
            int maxSupersteps) {
        this.maxSupersteps = maxSupersteps;
        this.graph = graph;
        this.values = new double[graph.n];
        Arrays.fill(values, initialValue);
        this.checkpointInterval = Math.max(1, checkpointInterval);
        this.failAtSuperstep = failAtSuperstep;
        this.active = new boolean[graph.n];
        for (int v = 0; v < graph.n; v++) mailboxes.add(new ArrayList<>());
    }

    void activate(int vertex, double seed) {
        active[vertex] = true;
        mailboxes.get(vertex).add(seed);
    }

    PregelResult run(VertexProgram program) {
        int step = 0;
        while (hasWork() && (maxSupersteps == 0 || step < maxSupersteps)) {
            step++;
            List<List<Double>> next = new ArrayList<>();
            for (int v = 0; v < graph.n; v++) next.add(new ArrayList<>());
            boolean[] nextActive = new boolean[graph.n];
            double aggregateValue = 0;
            boolean anyContribution = false;
            for (int v = 0; v < graph.n; v++) {
                if (!active[v]) continue;
                executions++;
                Outbox outbox = new Outbox();
                program.compute(v, values[v], mailboxes.get(v), graph, outbox);
                if (outbox.valueChanged) values[v] = outbox.newValue;
                for (int i = 0; i < outbox.targets.size(); i++) {
                    int target = outbox.targets.get(i)[0];
                    double value = outbox.values.get(i);
                    sent++;
                    List<Double> inbox = next.get(target);
                    if (combiner != null) {
                        boolean merged = false;
                        for (int k = 0; k < inbox.size(); k++) {
                            inbox.set(k, combiner.combine(inbox.get(k), value));
                            merged = true;
                            break;
                        }
                        if (!merged) inbox.add(value);
                    } else {
                        inbox.add(value);
                    }
                    nextActive[target] = true;
                }
                if (outbox.contributes) {
                    aggregateValue += outbox.aggregateContribution;
                    anyContribution = true;
                }
            }
            if (anyContribution) aggregator = aggregateValue;
            for (List<Double> inbox : next) delivered += inbox.size();
            mailboxes = next;
            active = nextActive;
            supersteps = step;
            if (step == failAtSuperstep && !failureInjected) {
                failureInjected = true;
                rollback();
                step = checkpoints.isEmpty() ? 0 : checkpoints.get(checkpoints.size() - 1).superstep;
                recoveredAt = step;
                continue;
            }
            if (step % checkpointInterval == 0) checkpoints.add(snapshot(step));
        }
        return new PregelResult(values, supersteps, sent, delivered, aggregator, executions, replayed,
                checkpoints.size(), recoveredAt);
    }

    boolean hasWork() {
        for (boolean flag : active) if (flag) return true;
        return false;
    }

    Checkpoint snapshot(int step) {
        double[] copy = values.clone();
        List<List<Double>> boxes = new ArrayList<>();
        for (List<Double> box : mailboxes) boxes.add(new ArrayList<>(box));
        return new Checkpoint(step, copy, boxes, active.clone(), aggregator);
    }

    /** Discards the state of the failed superstep and the partially done ones and restores the snapshot. */
    void rollback() {
        if (checkpoints.isEmpty()) return;
        Checkpoint last = checkpoints.get(checkpoints.size() - 1);
        System.arraycopy(last.values, 0, values, 0, values.length);
        mailboxes = new ArrayList<>();
        for (List<Double> box : last.mailboxes) mailboxes.add(new ArrayList<>(box));
        active = last.active.clone();
        aggregator = last.aggregator;
        replayed += Math.max(0, supersteps - last.superstep);
    }

    void setCombiner(MessageCombiner combiner) { this.combiner = combiner; }
}

/** Reference implementations used as oracles. */
final class Oracles {
    static double[] bfs(Graph graph, int source) {
        double[] distance = new double[graph.n];
        Arrays.fill(distance, Double.POSITIVE_INFINITY);
        distance[source] = 0;
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            int v = queue.poll();
            for (int i = 0; i < graph.degree(v); i++) {
                int w = graph.neighbour(v, i);
                if (distance[w] == Double.POSITIVE_INFINITY) {
                    distance[w] = distance[v] + 1;
                    queue.add(w);
                }
            }
        }
        return distance;
    }

    static double[] dijkstra(Graph graph, int source) {
        double[] distance = new double[graph.n];
        Arrays.fill(distance, Double.POSITIVE_INFINITY);
        distance[source] = 0;
        boolean[] done = new boolean[graph.n];
        for (int round = 0; round < graph.n; round++) {
            int best = -1;
            for (int v = 0; v < graph.n; v++) {
                if (!done[v] && (best == -1 || distance[v] < distance[best])) best = v;
            }
            if (best == -1 || distance[best] == Double.POSITIVE_INFINITY) break;
            done[best] = true;
            for (int i = 0; i < graph.degree(best); i++) {
                int w = graph.neighbour(best, i);
                double candidate = distance[best] + graph.edgeWeight(best, i);
                if (candidate < distance[w]) distance[w] = candidate;
            }
        }
        return distance;
    }

    static int[] components(Graph graph) {
        int[] parent = new int[graph.n];
        for (int v = 0; v < graph.n; v++) parent[v] = v;
        for (int v = 0; v < graph.n; v++) {
            for (int i = 0; i < graph.degree(v); i++) union(parent, v, graph.neighbour(v, i));
        }
        int[] label = new int[graph.n];
        for (int v = 0; v < graph.n; v++) label[v] = find(parent, v);
        return label;
    }

    static int find(int[] parent, int v) {
        while (parent[v] != v) {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    }

    static void union(int[] parent, int a, int b) {
        int ra = find(parent, a), rb = find(parent, b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
    }

    static double[] pageRank(Graph graph, double damping, int iterations) {
        double[] rank = new double[graph.n];
        Arrays.fill(rank, 1.0 / graph.n);
        for (int step = 0; step < iterations; step++) {
            double dangling = 0;
            for (int v = 0; v < graph.n; v++) {
                if (graph.degree(v) == 0) dangling += rank[v];
            }
            double base = (1 - damping) / graph.n + damping * dangling / graph.n;
            double[] next = new double[graph.n];
            for (int v = 0; v < graph.n; v++) next[v] = base;
            for (int v = 0; v < graph.n; v++) {
                if (graph.degree(v) == 0) continue;
                double share = damping * rank[v] / graph.degree(v);
                for (int i = 0; i < graph.degree(v); i++) next[graph.neighbour(v, i)] += share;
            }
            rank = next;
        }
        return rank;
    }
}

public class Solution {
    public static void main(String[] args) {
        // ---- 1. a small graph with a known structure --------------------------------------------
        int[][] rawEdges = {
            {0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {2, 6}, {7, 8}, {9, 9},
        };
        Graph graph = Graph.of(10, rawEdges);
        check(graph.n == 10 && graph.edgeCount() == 9, "the graph has nine distinct edges, got "
                + graph.edgeCount());
        check(graph.degree(9) == 1, "a self loop keeps degree one for the vertex that owns it");

        PregelResult bfs = bfsRun(graph, 0, null);
        double[] expectedBfs = Oracles.bfs(graph, 0);
        check(Arrays.equals(bfs.values, expectedBfs), "BFS must match the reference, got "
                + Arrays.toString(bfs.values));
        check(Double.isInfinite(bfs.values[7]) && Double.isInfinite(bfs.values[8]),
                "the disconnected component stays unreachable");
        int eccentricity = 0;
        for (double d : expectedBfs) if (!Double.isInfinite(d)) eccentricity = (int) Math.max(eccentricity, d);
        check(bfs.supersteps <= 2 * (eccentricity + 1),
                "improvement-forwarding BFS needs at most twice the eccentricity, got " + bfs.supersteps
                        + " for " + eccentricity);
        check(bfs.deliveredMessages <= graph.edgeCount() * 2 * 3,
                "the message count must stay bounded, delivered " + bfs.deliveredMessages);

        // ---- 2. the same engine for weighted shortest paths --------------------------------------
        int[][] weightedEdges = {
            {0, 1, 4}, {0, 2, 1}, {2, 1, 2}, {1, 3, 5}, {2, 3, 8}, {3, 4, 3}, {4, 5, 1},
        };
        Graph weighted = Graph.of(6, weightedEdges);
        PregelResult sssp = shortestPathRun(weighted, 0, -1);
        double[] expected = Oracles.dijkstra(weighted, 0);
        for (int v = 0; v < weighted.n; v++) {
            check(Math.abs(sssp.values[v] - expected[v]) < 1e-9,
                    "SSSP mismatch at " + v + ": " + sssp.values[v] + " vs " + expected[v]);
        }
        check(sssp.deliveredMessages > 0 && sssp.vertexExecutions >= weighted.n,
                "every vertex must run at least once");

        // ---- 3. connected components and PageRank ----------------------------------------------
        PregelResult cc = componentsRun(graph, -1);
        int[] expectedLabels = Oracles.components(graph);
        check(Arrays.equals(labelsOf(cc.values), expectedLabels),
                "connected components must match union-find, got " + Arrays.toString(labelsOf(cc.values)));
        check(cc.values[0] == 0 && cc.values[7] == 7 && cc.values[8] == 7,
                "the labels are the smallest vertex of each component");

        int iterations = 12;
        PregelResult pr = pageRankRun(graph, 0.85, iterations, null);
        double[] expectedRank = Oracles.pageRank(graph, 0.85, iterations);
        double maxError = 0;
        for (int v = 0; v < graph.n; v++) maxError = Math.max(maxError, Math.abs(pr.values[v]
                - expectedRank[v]));
        check(maxError < 1e-9, "PageRank must match the dense power iteration, error " + maxError
                + " mine " + Arrays.toString(pr.values) + " oracle " + Arrays.toString(expectedRank)
                + " steps " + pr.supersteps);
        check(pr.supersteps == iterations, "a fixed iteration count must run exactly that many supersteps");
        check(Math.abs(sum(pr.values) - 1) < 1e-9, "PageRank mass must be conserved, sum "
                + sum(pr.values));

        // ---- 4. combiner reduces traffic without changing the answer -----------------------------
        PregelResult combined = bfsRun(graph, 0, MessageCombiner.min());
        PregelResult plain = bfsRun(graph, 0, null);
        check(Arrays.equals(combined.values, plain.values), "the combiner must not change the result");
        check(combined.deliveredMessages <= plain.deliveredMessages,
                "the combiner must not increase traffic: " + combined.deliveredMessages + " vs "
                        + plain.deliveredMessages);
        PregelResult rankCombined = pageRankRun(graph, 0.85, iterations, MessageCombiner.sum());
        check(rankCombined.deliveredMessages < pr.deliveredMessages,
                "summing ranks before delivery must save messages: " + rankCombined.deliveredMessages
                        + " vs " + pr.deliveredMessages);

        // ---- 5. aggregator ------------------------------------------------------------------------
        PregelResult aggregated = bfsRun(graph, 0, null);
        check(aggregated.aggregator > 0, "the BFS program contributes the frontier size to the aggregator, "
                + "got " + aggregated.aggregator);

        // ---- 6. fault tolerance: a failure rolls back and replays to the same answer -------------
        int failAt = 3;
        PregelResult recovered = bfsRun(graph, 0, null, 2, failAt);
        check(Arrays.equals(recovered.values, expectedBfs),
                "recovery must reproduce the BFS result exactly");
        check(recovered.recoveredAt >= 2 && recovered.replayedExecutions > 0,
                "the rollback must replay at least one superstep, replayed "
                        + recovered.replayedExecutions);
        check(recovered.checkpointCount >= recovered.recoveredAt / 2,
                "checkpoints must be taken every two supersteps, saw " + recovered.checkpointCount);
        check(recovered.supersteps == plain.supersteps,
                "recovery must not change the number of supersteps, got " + recovered.supersteps);
        PregelResult ssspRecovered = shortestPathRun(weighted, 0, failAt);
        for (int v = 0; v < weighted.n; v++) {
            check(Math.abs(ssspRecovered.values[v] - expected[v]) < 1e-9,
                    "recovered SSSP mismatch at " + v);
        }
        PregelResult lateFailure = bfsRun(graph, 0, null, 2, 4);
        check(Arrays.equals(lateFailure.values, expectedBfs), "a failure later in the run recovers too");
        check(lateFailure.replayedExecutions >= 2,
                "replaying from a checkpoint two steps back must redo work, got "
                        + lateFailure.replayedExecutions);

        // ---- 7. determinism, degenerate inputs and halting -------------------------------------
        PregelResult again = bfsRun(graph, 0, null);
        check(Arrays.equals(again.values, bfs.values) && again.supersteps == bfs.supersteps
                && again.deliveredMessages == bfs.deliveredMessages,
                "two identical runs must agree in every counter");
        Graph empty = Graph.of(4, new int[0][]);
        PregelResult emptyRun = bfsRun(empty, 0, null);
        check(emptyRun.values[0] == 0 && emptyRun.supersteps == 1
                && Double.isInfinite(emptyRun.values[3]),
                "a graph without edges still runs the source exactly once");
        Graph single = Graph.of(1, new int[0][]);
        PregelResult singleRun = bfsRun(single, 0, null);
        check(singleRun.supersteps == 1 && singleRun.values[0] == 0 && singleRun.sentMessages == 0,
                "a single vertex terminates immediately and sends nothing");

        System.out.println("bfs: " + bfs.supersteps + " supersteps, " + bfs.sentMessages + " sent, "
                + bfs.deliveredMessages + " delivered, " + bfs.vertexExecutions + " executions");
        System.out.println("sssp: " + Arrays.toString(sssp.values) + " in " + sssp.supersteps
                + " supersteps; components match union-find; pagerank error " + maxError);
        System.out.println("combiner: min saves " + (plain.deliveredMessages - combined.deliveredMessages)
                + " deliveries, sum saves " + (pr.deliveredMessages - rankCombined.deliveredMessages));
        System.out.println("fault tolerance: failure at superstep " + failAt + " rolled back to "
                + recovered.recoveredAt + " and replayed " + recovered.replayedExecutions
                + " executions over " + recovered.checkpointCount + " checkpoints");
        System.out.println("PASS");
    }

    /** BFS as a vertex program: the value is the distance, only improvements are forwarded. */
    static PregelResult bfsRun(Graph graph, int source, MessageCombiner combiner) {
        return bfsRun(graph, source, combiner, 2, -1);
    }

    static PregelResult bfsRun(Graph graph, int source, MessageCombiner combiner, int interval,
            int failAt) {
        PregelEngine engine = new PregelEngine(graph, Double.POSITIVE_INFINITY, interval, failAt);
        engine.setCombiner(combiner);
        engine.activate(source, 0.0);
        return engine.run((vertex, value, inbox, g, outbox) -> {
            double best = Double.POSITIVE_INFINITY;
            for (double message : inbox) best = Math.min(best, message);
            outbox.contribute(inbox.size());
            if (best < value) {
                outbox.setValue(best);
                for (int i = 0; i < g.degree(vertex); i++) {
                    outbox.send(g.neighbour(vertex, i), best + 1);
                }
            }
        });
    }

    /** SSSP: identical shape, but the edge weights are added on the way out. */
    static PregelResult shortestPathRun(Graph graph, int source, int failAt) {
        PregelEngine engine = new PregelEngine(graph, Double.POSITIVE_INFINITY, 2, failAt);
        engine.setCombiner(MessageCombiner.min());
        engine.activate(source, 0.0);
        return engine.run((vertex, value, inbox, g, outbox) -> {
            double best = Double.POSITIVE_INFINITY;
            for (double message : inbox) best = Math.min(best, message);
            if (best < value) {
                outbox.setValue(best);
                for (int i = 0; i < g.degree(vertex); i++) {
                    outbox.send(g.neighbour(vertex, i), best + g.edgeWeight(vertex, i));
                }
            }
        });
    }

    /** Label propagation: the seed is the vertex's own id, the value is the smallest label seen. */
    static PregelResult componentsRun(Graph graph, int failAt) {
        PregelEngine engine = new PregelEngine(graph, Double.POSITIVE_INFINITY, 2, failAt);
        engine.setCombiner(MessageCombiner.min());
        for (int v = 0; v < graph.n; v++) engine.activate(v, v);
        return engine.run((vertex, value, inbox, g, outbox) -> {
            double best = value;
            for (double message : inbox) best = Math.min(best, message);
            if (best < value) {
                outbox.setValue(best);
                for (int i = 0; i < g.degree(vertex); i++) outbox.send(g.neighbour(vertex, i), best);
            }
        });
    }

    /** PageRank with a fixed number of iterations; dangling mass is spread evenly, as in the oracle. */
    static PregelResult pageRankRun(Graph graph, double damping, int iterations, MessageCombiner combiner) {
        PregelEngine engine = new PregelEngine(graph, 1.0 / graph.n, 2, -1, iterations);
        engine.setCombiner(combiner);
        // The first superstep must already see the ranks of the uniform start vector, so seed each mailbox
        // with the incoming share SUM_{u -> v} rank_0(u)/deg(u) exactly as the dense oracle does.
        double[] initialIncoming = new double[graph.n];
        for (int u = 0; u < graph.n; u++) {
            if (graph.degree(u) == 0) continue;
            double share = (1.0 / graph.n) / graph.degree(u);
            for (int i = 0; i < graph.degree(u); i++) initialIncoming[graph.neighbour(u, i)] += share;
        }
        for (int v = 0; v < graph.n; v++) engine.activate(v, initialIncoming[v]);
        return engine.run((vertex, value, inbox, g, outbox) -> {
            double incoming = 0;
            for (double message : inbox) incoming += message;
            double teleport = (1 - damping) / g.n + damping * engine.aggregator / g.n;
            double updated = teleport + damping * incoming;
            outbox.setValue(updated);
            if (g.degree(vertex) == 0) outbox.contribute(updated);
            for (int i = 0; i < g.degree(vertex); i++) {
                outbox.send(g.neighbour(vertex, i), updated / g.degree(vertex));
            }
        });
    }

    static int[] labelsOf(double[] values) {
        int[] labels = new int[values.length];
        for (int v = 0; v < values.length; v++) labels[v] = (int) values[v];
        return labels;
    }

    static double sum(double[] values) {
        double total = 0;
        for (double value : values) total += value;
        return total;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
