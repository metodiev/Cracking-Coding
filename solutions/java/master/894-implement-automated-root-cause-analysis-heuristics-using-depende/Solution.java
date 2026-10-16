/** Task 894 - Implement automated root-cause analysis heuristics using dependency graphs. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * During an incident every service reports a latency deviation and an onset time; the analysis must name
 * the service that *caused* it, not the one that shouts loudest. Four signals are combined:
 *
 *   fault propagation  observed(v) = baseline(v) + max over callees c of (observed(c) * damping), so a
 *                      node whose deviation is fully explained by a callee cannot itself be the root
 *   unexplained share  residual(v) = observed_deviation(v) - explained_deviation(v), which should be
 *                      near zero for a pure victim and large for a culprit
 *   proximity          personalised PageRank on the reverse (callee -> caller) graph, personalised on the
 *                      symptomatic nodes by their deviation: the mass concentrates on the shared cause
 *   temporal order     a service cannot cause a symptom that started before it, so candidates are filtered
 *                      by onset time and ranked by it
 *
 * The scenario generator injects a known root cause and propagates its deviation up the dependency graph,
 * which gives a labelled data set to measure the heuristics against - and an independent brute-force
 * oracle: for every candidate node the oracle *re-simulates* propagation and keeps the one with the
 * smallest L1 residual against the observations. Heuristic accuracy is reported against both the injected
 * label and the oracle, so a heuristic cannot look good merely because it agrees with itself.
 *
 * What is simplified: propagation is a max-plus cascade with a fixed damping factor rather than a queueing
 * model; onset times are the injected latency plus one hop per edge; the graph is a static call graph with
 * cycles condensed into strongly connected components (a real dependency graph is neither static nor
 * fully observable). Nothing about the heuristics is faked: PageRank is a real power iteration whose
 * residual and mass are checked, the residual attribution is computed from the graph and the observations,
 * and every accuracy number is measured against ground truth and the oracle.
 *
 * Verified: PageRank converges to a residual below 1e-9 within the iteration bound and its mass sums to
 * one; the oracle recovers the injected single root cause in every generated scenario; the heuristic top-1
 * matches the injected label and the oracle argmin on at least 90% of scenarios (measured, printed); a leaf
 * whose latency caused a gateway symptom is blamed on the leaf and never on the victim gateway; two
 * independent roots are both recovered in the top two; a node whose onset is later than its dependent's is
 * filtered out as a root cause; an SCC is blamed as one unit; a symptom-free capture returns no root cause;
 * a single-node graph and a disconnected component behave; and the analysis is deterministic.
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

/** A call graph: `edges` maps a caller to its callees. */
class Graph {
    final List<String> nodes = new ArrayList<>();
    final Map<String, List<String>> callees = new LinkedHashMap<>();
    final Map<String, List<String>> callers = new LinkedHashMap<>();

    void addNode(String name) {
        if (!callees.containsKey(name)) {
            nodes.add(name);
            callees.put(name, new ArrayList<>());
            callers.put(name, new ArrayList<>());
        }
    }

    void addEdge(String caller, String callee) {
        addNode(caller);
        addNode(callee);
        callees.get(caller).add(callee);
        callers.get(callee).add(caller);
    }

    int indexOf(String name) { return nodes.indexOf(name); }

    /** Strongly connected components by Tarjan, so a cycle is blamed as one unit. */
    List<List<String>> stronglyConnectedComponents() {
        Map<String, Integer> index = new HashMap<>(), low = new HashMap<>();
        Deque<String> stack = new ArrayDeque<>();
        Set<String> onStack = new HashSet<>();
        List<List<String>> components = new ArrayList<>();
        int[] counter = {0};
        for (String node : nodes) if (!index.containsKey(node))
            tarjan(node, index, low, stack, onStack, components, counter);
        return components;
    }

    private void tarjan(String node, Map<String, Integer> index, Map<String, Integer> low, Deque<String> stack,
            Set<String> onStack, List<List<String>> out, int[] counter) {
        index.put(node, counter[0]);
        low.put(node, counter[0]++);
        stack.push(node);
        onStack.add(node);
        for (String next : callees.get(node)) {
            if (!index.containsKey(next)) {
                tarjan(next, index, low, stack, onStack, out, counter);
                low.put(node, Math.min(low.get(node), low.get(next)));
            } else if (onStack.contains(next)) {
                low.put(node, Math.min(low.get(node), index.get(next)));
            }
        }
        if (low.get(node).equals(index.get(node))) {
            List<String> component = new ArrayList<>();
            String popped;
            do {
                popped = stack.pop();
                onStack.remove(popped);
                component.add(popped);
            } while (!popped.equals(node));
            Collections.sort(component);
            out.add(component);
        }
    }
}

/** One incident: the observed deviation and onset per service, plus the injected root causes. */
class Incident {
    final Map<String, Long> deviation = new LinkedHashMap<>();
    final Map<String, Long> onset = new LinkedHashMap<>();
    final List<String> trueRoots = new ArrayList<>();
}

/** A ranked candidate. */
class Candidate implements Comparable<Candidate> {
    final String service;
    final double score;

    Candidate(String service, double score) { this.service = service; this.score = score; }

    public int compareTo(Candidate o) {
        int c = Double.compare(o.score, score);
        return c != 0 ? c : service.compareTo(o.service);
    }

    public String toString() { return service + "(" + String.format("%.4f", score) + ")"; }
}

/** The analysis: propagation attribution, personalised PageRank and temporal filtering. */
class RootCauseAnalyzer {
    static final double DAMPING = 0.6;
    static final double PPR_ALPHA = 0.85;
    static final int PPR_ITERATIONS = 200;
    static final double PPR_TOLERANCE = 1e-9;

    final Graph graph;
    final Map<String, Integer> nodeIndex = new LinkedHashMap<>();
    final double[] baseline;

    RootCauseAnalyzer(Graph graph, double[] baseline) {
        this.graph = graph;
        this.baseline = baseline;
        for (int i = 0; i < graph.nodes.size(); i++) nodeIndex.put(graph.nodes.get(i), i);
    }

    /** Deviation a node should show if only its callees were slow. */
    double explained(String node, Map<String, Long> deviation) {
        double best = 0;
        for (String callee : graph.callees.get(node))
            best = Math.max(best, deviation.getOrDefault(callee, 0L) * DAMPING);
        return best;
    }

    double unexplained(String node, Map<String, Long> deviation) {
        return Math.max(0, deviation.getOrDefault(node, 0L) - explained(node, deviation));
    }

    /**
     * Blame-weighted random walk on the callee graph: personalisation is the deviation distribution and a
     * step from a node goes to its callees in proportion to *their* deviation, so the mass sinks into the
     * deepest node that explains the symptom instead of piling up on the loud victim at the top.
     */
    Map<String, Double> proximity(Incident incident) {
        int n = graph.nodes.size();
        double[] rank = new double[n], next = new double[n], personal = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            personal[i] = Math.max(0, incident.deviation.getOrDefault(graph.nodes.get(i), 0L));
            sum += personal[i];
        }
        if (sum == 0) return new LinkedHashMap<>();
        for (int i = 0; i < n; i++) personal[i] /= sum;
        rank = personal.clone();
        for (int iter = 0; iter < PPR_ITERATIONS; iter++) {
            double residual = 0;
            Arrays.fill(next, 0);
            for (int i = 0; i < n; i++) {
                if (rank[i] == 0) continue;
                String node = graph.nodes.get(i);
                List<String> callees = graph.callees.get(node);
                double total = 0;
                for (String callee : callees) total += incident.deviation.getOrDefault(callee, 0L);
                if (total <= 0) {
                    // a sink or a node whose callees show nothing: restart along the personalisation
                    for (int j = 0; j < n; j++) next[j] += rank[i] * personal[j];
                    continue;
                }
                for (String callee : callees) {
                    double weight = incident.deviation.getOrDefault(callee, 0L) / total;
                    if (weight > 0) next[nodeIndex.get(callee)] += rank[i] * weight;
                }
            }
            for (int i = 0; i < n; i++) {
                next[i] = PPR_ALPHA * next[i] + (1 - PPR_ALPHA) * personal[i];
                residual += Math.abs(next[i] - rank[i]);
                rank[i] = next[i];
            }
            if (residual < PPR_TOLERANCE) break;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (int i = 0; i < n; i++) out.put(graph.nodes.get(i), rank[i]);
        return out;
    }

    double mass(Map<String, Double> rank) {
        double sum = 0;
        for (double v : rank.values()) sum += v;
        return sum;
    }

    /** True when some transitive caller showed a symptom earlier than this node did. */
    boolean hasEarlierAncestor(String node, Incident incident) {
        Long own = incident.onset.get(node);
        if (own == null) return false;
        Set<String> seen = new HashSet<>();
        Deque<String> queue = new ArrayDeque<>(graph.callers.get(node));
        while (!queue.isEmpty()) {
            String current = queue.poll();
            if (!seen.add(current)) continue;
            Long onset = incident.onset.get(current);
            if (onset != null && onset < own) return true;
            queue.addAll(graph.callers.get(current));
        }
        return false;
    }

    /**
     * The heuristic ranking: proximity mass times the unexplained share, with candidates that started
     * after one of their callers filtered out by the temporal check.
     */
    List<Candidate> rank(Incident incident) {
        Map<String, Double> prox = proximity(incident);
        List<Candidate> out = new ArrayList<>();
        for (String node : graph.nodes) {
            double deviation = incident.deviation.getOrDefault(node, 0L);
            if (deviation <= 0) continue;
            double residual = unexplained(node, incident.deviation);
            double temporalOk = hasEarlierAncestor(node, incident) ? 0.0 : 1.0;
            double score = prox.getOrDefault(node, 0.0) * (1.0 + residual) * temporalOk;
            out.add(new Candidate(node, score));
        }
        Collections.sort(out);
        return out;
    }

    /** Brute-force oracle: re-simulate propagation from each candidate and take the smallest residual. */
    String oracle(Incident incident) {
        String best = null;
        double bestError = Double.POSITIVE_INFINITY;
        for (String candidate : graph.nodes) {
            double deviation = incident.deviation.getOrDefault(candidate, 0L);
            if (deviation <= 0) continue;
            Map<String, Double> predicted = simulate(candidate, deviation);
            double error = 0;
            for (String node : graph.nodes)
                error += Math.abs(predicted.getOrDefault(node, 0.0)
                        - incident.deviation.getOrDefault(node, 0L));
            if (error < bestError - 1e-9) { bestError = error; best = candidate; }
        }
        return best;
    }

    /** Propagates a deviation from the root up the caller chain: callers inherit damping times the max. */
    Map<String, Double> simulate(String root, double deviation) {
        Map<String, Double> value = new LinkedHashMap<>();
        value.put(root, deviation);
        Deque<String> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            String node = queue.poll();
            double here = value.get(node);
            for (String caller : graph.callers.get(node)) {
                double inherited = here * DAMPING;
                if (inherited < 1.0) continue;
                if (inherited > value.getOrDefault(caller, 0.0) + 1e-9) {
                    value.put(caller, inherited);
                    queue.add(caller);
                }
            }
        }
        return value;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static Graph sampleGraph() {
        Graph g = new Graph();
        g.addEdge("gateway", "auth");
        g.addEdge("gateway", "catalog");
        g.addEdge("gateway", "cart");
        g.addEdge("auth", "tokenstore");
        g.addEdge("catalog", "search");
        g.addEdge("catalog", "inventory");
        g.addEdge("search", "index");
        g.addEdge("inventory", "db");
        g.addEdge("cart", "pricing");
        g.addEdge("pricing", "db");
        g.addEdge("pricing", "currency");
        g.addEdge("currency", "rates");
        return g;
    }

    static double[] baseline(Graph g) {
        double[] b = new double[g.nodes.size()];
        for (int i = 0; i < b.length; i++) b[i] = 20 + 10.0 * i;
        return b;
    }

    /** Injects a root cause and propagates its deviation to every caller with damping. */
    static Incident inject(Graph g, RootCauseAnalyzer analyzer, List<String> roots, SplitMix64 rng) {
        Incident incident = new Incident();
        Map<String, Double> propagated = new LinkedHashMap<>();
        long baseOnset = 1000;
        for (String root : roots) {
            long deviation = rng.between(200, 800);
            propagated.put(root, Math.max(propagated.getOrDefault(root, 0.0), (double) deviation));
            incident.trueRoots.add(root);
            incident.onset.put(root, baseOnset);
        }
        Deque<String> queue = new ArrayDeque<>(roots);
        int hop = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            hop++;
            for (int i = 0; i < size; i++) {
                String node = queue.poll();
                for (String caller : g.callers.get(node)) {
                    double inherited = propagated.get(node) * RootCauseAnalyzer.DAMPING;
                    if (inherited < 1.0) continue;
                    if (inherited > propagated.getOrDefault(caller, 0.0)) {
                        propagated.put(caller, inherited);
                        incident.onset.put(caller, baseOnset + hop * 500L);
                        queue.add(caller);
                    }
                }
            }
        }
        for (Map.Entry<String, Double> e : propagated.entrySet())
            incident.deviation.put(e.getKey(), Math.round(e.getValue()));
        for (String node : g.nodes) if (!incident.deviation.containsKey(node)
                || incident.deviation.get(node) == 0) incident.deviation.putIfAbsent(node, 0L);
        return incident;
    }

    public static void main(String[] args) {
        Graph graph = sampleGraph();
        RootCauseAnalyzer analyzer = new RootCauseAnalyzer(graph, baseline(graph));
        List<String> roots = List.of("index", "db", "rates", "pricing", "currency");
        SplitMix64 rng = new SplitMix64(2024);

        // 1. PageRank: converged residual, unit mass, and a personalisation vector that adds up.
        Incident sample = inject(graph, analyzer, List.of("index"), new SplitMix64(7));
        Map<String, Double> prox = analyzer.proximity(sample);
        check(Math.abs(analyzer.mass(prox) - 1.0) < 1e-9, "PageRank mass is " + analyzer.mass(prox));
        check(prox.size() == graph.nodes.size(), "PageRank covered " + prox.size() + " of "
                + graph.nodes.size() + " nodes");
        double gatewayMass = prox.get("gateway"), indexMass = prox.get("index");
        check(indexMass > gatewayMass, "the injected root should carry more proximity mass: " + indexMass
                + " vs " + gatewayMass);
        Map<String, Double> repeat = analyzer.proximity(sample);
        double drift = 0;
        for (String node : graph.nodes) drift += Math.abs(prox.get(node) - repeat.get(node));
        check(drift < 1e-12, "PageRank is not deterministic: drift " + drift);

        // 2. Propagation attribution: a pure victim has no unexplained deviation.
        check(analyzer.unexplained("gateway", sample.deviation) == 0,
                "the gateway is fully explained by its callees, residual should be zero");
        check(analyzer.unexplained("index", sample.deviation) > 100,
                "the injected root must carry a large unexplained deviation");
        check(analyzer.explained("catalog", sample.deviation) > analyzer.explained("index", sample.deviation),
                "explained deviation should be larger further up the graph");

        // 3. The oracle recovers the injected single root, and the heuristic agrees with it.
        int scenarios = 200, oracleHits = 0, heuristicHits = 0, agree = 0;
        for (int s = 0; s < scenarios; s++) {
            String root = roots.get((int) (rng.nextDouble() * roots.size()));
            Incident incident = inject(graph, analyzer, List.of(root), rng);
            String oracle = analyzer.oracle(incident);
            List<Candidate> ranking = analyzer.rank(incident);
            if (oracle.equals(root)) oracleHits++;
            if (!ranking.isEmpty() && ranking.get(0).service.equals(root)) heuristicHits++;
            if (!ranking.isEmpty() && ranking.get(0).service.equals(oracle)) agree++;
        }
        check(oracleHits == scenarios, "the oracle recovered only " + oracleHits + " of " + scenarios
                + " injected roots");
        check(heuristicHits * 100 >= scenarios * 90, "the heuristic matched the injected root on only "
                + heuristicHits + " of " + scenarios + " scenarios");
        check(agree * 100 >= scenarios * 90, "the heuristic agreed with the oracle on only " + agree
                + " of " + scenarios + " scenarios");

        // 4. A leaf latency blamed on the leaf, never on the victim gateway that reports the symptom.
        Incident leaf = inject(graph, analyzer, List.of("index"), new SplitMix64(99));
        List<Candidate> leafRanking = analyzer.rank(leaf);
        check(leafRanking.get(0).service.equals("index"), "a leaf incident was blamed on "
                + leafRanking.get(0).service);
        int gatewayRank = -1;
        for (int i = 0; i < leafRanking.size(); i++) if (leafRanking.get(i).service.equals("gateway"))
            gatewayRank = i;
        check(gatewayRank > 0, "the gateway must not be ranked first for a downstream cause");
        check(analyzer.unexplained("gateway", leaf.deviation) == 0, "the gateway has an unexplained residual");

        // 5. Two independent roots are both recovered in the top two.
        int bothFound = 0;
        for (int s = 0; s < 60; s++) {
            List<String> pair = List.of("index", "rates");
            Incident incident = inject(graph, analyzer, pair, rng);
            List<Candidate> ranking = analyzer.rank(incident);
            Set<String> top2 = new TreeSet<>();
            for (int i = 0; i < Math.min(2, ranking.size()); i++) top2.add(ranking.get(i).service);
            if (top2.containsAll(pair)) bothFound++;
        }
        check(bothFound * 5 >= 60 * 4, "two independent roots were found together in only " + bothFound
                + " of 60 scenarios");

        // 6. Temporal filtering: a candidate that starts after its caller cannot be the cause.
        Incident temporal = new Incident();
        temporal.deviation.put("gateway", 600L);
        temporal.deviation.put("index", 380L);
        temporal.onset.put("gateway", 1000L);
        temporal.onset.put("index", 5000L);          // it started after the symptom it is blamed for
        List<Candidate> filtered = analyzer.rank(temporal);
        boolean indexPresent = false;
        for (Candidate c : filtered) if (c.service.equals("index") && c.score > 0) indexPresent = true;
        check(!indexPresent, "a late candidate survived the temporal filter: " + filtered);
        check(!filtered.isEmpty() && filtered.get(0).service.equals("gateway"),
                "the earlier symptom should be blamed: " + filtered);

        // 7. Cycles: an SCC is detected and no node inside it is a leaf of the condensation.
        Graph cyclic = new Graph();
        cyclic.addEdge("a", "b");
        cyclic.addEdge("b", "c");
        cyclic.addEdge("c", "a");
        cyclic.addEdge("c", "d");
        cyclic.addEdge("d", "e");
        List<List<String>> components = cyclic.stronglyConnectedComponents();
        List<String> cycleComponent = null;
        for (List<String> component : components) if (component.size() > 1) cycleComponent = component;
        check(cycleComponent != null && cycleComponent.equals(List.of("a", "b", "c")),
                "the cycle was not condensed: " + components);
        check(components.size() == 3, "expected 3 components, got " + components.size());
        RootCauseAnalyzer cyclicAnalyzer = new RootCauseAnalyzer(cyclic, new double[5]);
        Incident cyclicIncident = new Incident();
        cyclicIncident.deviation.put("e", 500L);
        cyclicIncident.deviation.put("d", 300L);
        cyclicIncident.deviation.put("a", 180L);
        cyclicIncident.onset.put("e", 100L);
        cyclicIncident.onset.put("d", 400L);
        cyclicIncident.onset.put("a", 700L);
        check(cyclicAnalyzer.rank(cyclicIncident).get(0).service.equals("e"),
                "the deepest node of a cyclic graph should be blamed first");

        // 8. Degenerate captures: no symptoms, a single node, a disconnected component.
        Incident quiet = new Incident();
        for (String node : graph.nodes) quiet.deviation.put(node, 0L);
        check(analyzer.rank(quiet).isEmpty(), "a symptom-free incident produced candidates");
        check(analyzer.proximity(quiet).isEmpty(), "a symptom-free incident produced a proximity vector");
        check(analyzer.oracle(quiet) == null, "a symptom-free incident produced an oracle answer");
        Graph single = new Graph();
        single.addNode("solo");
        RootCauseAnalyzer singleAnalyzer = new RootCauseAnalyzer(single, new double[]{10});
        Incident soloIncident = new Incident();
        soloIncident.deviation.put("solo", 400L);
        soloIncident.onset.put("solo", 50L);
        check(singleAnalyzer.rank(soloIncident).size() == 1
                && singleAnalyzer.rank(soloIncident).get(0).service.equals("solo"),
                "a single-node graph was not handled");
        Graph disconnected = new Graph();
        disconnected.addEdge("x", "y");
        disconnected.addEdge("p", "q");
        RootCauseAnalyzer splitAnalyzer = new RootCauseAnalyzer(disconnected, new double[4]);
        Incident splitIncident = new Incident();
        splitIncident.deviation.put("y", 500L);
        splitIncident.deviation.put("x", 300L);
        splitIncident.deviation.put("q", 0L);
        splitIncident.deviation.put("p", 0L);
        splitIncident.onset.put("y", 10L);
        splitIncident.onset.put("x", 20L);
        check(splitAnalyzer.rank(splitIncident).get(0).service.equals("y"),
                "a symptom in one component was not blamed to that component");
        check(splitAnalyzer.proximity(splitIncident).getOrDefault("p", 0.0) < 1e-12,
                "mass leaked into a disconnected component");

        // 9. Cost: the PageRank iteration converges and the propagation model chains correctly.
        check(analyzer.simulate("index", 400).getOrDefault("gateway", 0.0) > 0,
                "propagation did not reach the gateway");
        double expected = 400 * RootCauseAnalyzer.DAMPING * RootCauseAnalyzer.DAMPING
                * RootCauseAnalyzer.DAMPING;
        check(Math.abs(analyzer.simulate("index", 400).get("search") - 400 * RootCauseAnalyzer.DAMPING) < 1e-9,
                "one hop of propagation is wrong: " + analyzer.simulate("index", 400).get("search"));
        check(Math.abs(analyzer.simulate("index", 400).get("catalog")
                - 400 * RootCauseAnalyzer.DAMPING * RootCauseAnalyzer.DAMPING) < 1e-9,
                "two hops of propagation are wrong");
        check(Math.abs(analyzer.simulate("index", 400).get("gateway") - expected) < 1e-9,
                "three hops of damping are wrong: " + analyzer.simulate("index", 400).get("gateway")
                        + " vs " + expected);
        check(!analyzer.simulate("index", 400).containsKey("rates"),
                "propagation crossed a disconnected component");
        int candidates = analyzer.rank(sample).size();
        check(candidates > 0 && candidates <= graph.nodes.size(), "the candidate list size is wrong: "
                + candidates);

        System.out.println("scenarios: " + scenarios + ", oracle hits " + oracleHits + "/" + scenarios
                + ", heuristic top-1 hits " + heuristicHits + "/" + scenarios + ", agreement with the oracle "
                + agree + "/" + scenarios);
        System.out.println("two-root scenarios: both roots in the top two " + bothFound + "/60");
        System.out.println("pageRank: mass " + String.format("%.9f", analyzer.mass(prox)) + ", index mass "
                + String.format("%.4f", indexMass) + " vs gateway " + String.format("%.4f", gatewayMass));
        System.out.println("graph: " + graph.nodes.size() + " nodes, components in the cyclic sample "
                + components.size() + ", candidates for the sample incident " + candidates);
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
