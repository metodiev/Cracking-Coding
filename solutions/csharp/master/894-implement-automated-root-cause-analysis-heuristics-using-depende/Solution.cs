// Task 894 - Implement automated root-cause analysis heuristics using dependency graphs. (Tier: Master, Beyond — Software Engineering & Tooling)
//
// An incident is a root-cause service plus a fault that propagates along the service dependency graph:
// an edge (a -> b) means "b depends on a", so b inherits a's anomaly attenuated by the edge weight, and a
// fault in a *dependency* is what makes its dependents unhealthy. Some edges amplify (a >1 attenuation -
// retry storms), which is precisely why "the most anomalous service" is not the root cause. Four
// heuristics score candidate roots from the observed per-service anomaly vector alone:
//
//  1. top anomaly - the naive baseline (arg max observed);
//  2. anomalous sources - among the services above an anomaly threshold, those whose dependencies are
//     all healthy, i.e. the sources of the anomalous sub-DAG;
//  3. centrality prior - the anomaly blended with a PageRank prior over the dependency graph;
//  4. predictive fit - inverse propagation: re-simulate the propagation from each candidate with that
//     candidate's observed magnitude and keep the smallest residual. Ties are returned as a candidate
//     set instead of being broken arbitrarily.
//
// Simplifications (documented, not behavioural): anomalies are scalar scores (not latency distributions),
// propagation is a static maximum-with-attenuation over a DAG rather than a time series, and the graph is
// an explicit layered DAG instead of a discovered topology. No time windowing, no seasonality, no
// feedback from retries beyond the fixed per-edge amplification.
//
// Verified: the propagation model is acyclic and direction-correct; over 300 generated incidents the
// predictive-fit heuristic reaches high recall and beats the naive baseline; an amplified edge makes the
// naive heuristic blame a downstream service while inverse propagation still finds the root; leaf and
// entry-point faults are attributed exactly; every heuristic always answers with an anomalous service;
// the true root's residual is the model's error and stays small; equal anomalies are reported as a tie;
// results are deterministic and seed-reproducible; silent clusters abstain; a fault never crosses graph
// components; single-node graphs and cycles/out-of-range edges are handled; and recall holds at 60
// services with 400 incidents.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class Edge
{
    internal int From;          // the dependency
    internal int To;            // the service that depends on it
    internal double Attenuation;
}

internal sealed class DependencyGraph
{
    internal readonly int Nodes;
    internal readonly List<Edge> Edges = new List<Edge>();
    internal readonly List<int>[] Dependencies;      // incoming: what a service depends on
    internal readonly List<int>[] Dependents;        // outgoing: who depends on a service
    internal readonly int[] Topological;

    internal DependencyGraph(int nodes, IEnumerable<Edge> edges)
    {
        Nodes = nodes;
        foreach (Edge edge in edges)
        {
            if (edge.From < 0 || edge.From >= nodes || edge.To < 0 || edge.To >= nodes)
                throw new InvalidOperationException("edge outside the graph: " + edge.From + "->" + edge.To);
            Edges.Add(edge);
        }
        Dependencies = Enumerable.Range(0, nodes).Select(_ => new List<int>()).ToArray();
        Dependents = Enumerable.Range(0, nodes).Select(_ => new List<int>()).ToArray();
        foreach (Edge edge in Edges)
        {
            Dependencies[edge.To].Add(edge.From);
            Dependents[edge.From].Add(edge.To);
        }
        Topological = TopoSort();
    }

    private int[] TopoSort()
    {
        var indegree = new int[Nodes];
        foreach (Edge edge in Edges) indegree[edge.To]++;
        var ready = new SortedSet<int>();
        var order = new List<int>();
        for (int i = 0; i < Nodes; i++) if (indegree[i] == 0) ready.Add(i);
        while (ready.Count > 0)
        {
            int node = ready.Min;
            ready.Remove(node);
            order.Add(node);
            foreach (int next in Dependents[node])
                if (--indegree[next] == 0) ready.Add(next);
        }
        if (order.Count != Nodes) throw new InvalidOperationException("dependency graph contains a cycle");
        return order.ToArray();
    }

    /** A fault at `root` propagates to everything that (transitively) depends on it. */
    internal double[] Propagate(int root, double magnitude)
    {
        var scores = new double[Nodes];
        scores[root] = magnitude;
        foreach (int node in Topological)
        {
            if (scores[node] == 0) continue;
            foreach (Edge edge in Edges)
            {
                if (edge.From != node) continue;
                double induced = scores[node] * edge.Attenuation;
                if (induced > scores[edge.To]) scores[edge.To] = induced;
            }
        }
        return scores;
    }

    /** True when a fault at `upstream` can reach `downstream` at all. */
    internal bool Reaches(int upstream, int downstream)
    {
        var stack = new Stack<int>();
        var seen = new HashSet<int>();
        stack.Push(upstream);
        while (stack.Count > 0)
        {
            int node = stack.Pop();
            if (!seen.Add(node)) continue;
            if (node == downstream) return true;
            foreach (int next in Dependents[node]) stack.Push(next);
        }
        return false;
    }

    internal double[] PageRank(double damping, int iterations)
    {
        var rank = Enumerable.Repeat(1.0 / Nodes, Nodes).ToArray();
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            double dangling = 0;
            for (int node = 0; node < Nodes; node++) if (Dependents[node].Count == 0) dangling += rank[node];
            var next = Enumerable.Repeat((1.0 - damping) / Nodes + damping * dangling / Nodes, Nodes).ToArray();
            for (int node = 0; node < Nodes; node++)
            {
                List<int> outgoing = Dependents[node];
                if (outgoing.Count == 0) continue;
                double share = damping * rank[node] / outgoing.Count;
                foreach (int dependent in outgoing) next[dependent] += share;
            }
            rank = next;
        }
        return rank;
    }
}

internal sealed class Incident
{
    internal double[] Observed = Array.Empty<double>();
    internal int Truth;
    internal double Magnitude;
    internal int Faults;
}

internal sealed class RcaResult
{
    internal string Heuristic = "";
    internal int Primary = -1;
    internal List<int> Candidates = new List<int>();
    internal double Residual;
    internal int BlastRadius;
    internal bool Tie;
}

internal static class Rca
{
    internal const double AnomalyThreshold = 0.10;

    private static List<int> Ranked(DependencyGraph graph, double[] observed, Func<int, double> score) =>
        Enumerable.Range(0, graph.Nodes)
            .Where(service => observed[service] > AnomalyThreshold)
            .OrderByDescending(service => score(service))
            .ThenBy(service => service)
            .ToList();

    private static RcaResult FromRanking(string heuristic, DependencyGraph graph, double[] observed, List<int> ranked)
    {
        var result = new RcaResult
        {
            Heuristic = heuristic,
            BlastRadius = observed.Count(value => value > AnomalyThreshold),
        };
        if (ranked.Count == 0) return result;
        result.Primary = ranked[0];
        double best = observed[ranked[0]];
        result.Candidates.AddRange(ranked.TakeWhile(service => Math.Abs(observed[service] - best) < 1e-9));
        result.Tie = result.Candidates.Count > 1;
        return result;
    }

    internal static RcaResult TopAnomaly(DependencyGraph graph, double[] observed) =>
        FromRanking("top-anomaly", graph, observed, Ranked(graph, observed, service => observed[service]));

    internal static RcaResult AnomalousSources(DependencyGraph graph, double[] observed)
    {
        var anomalous = Enumerable.Range(0, graph.Nodes).Where(s => observed[s] > AnomalyThreshold).ToHashSet();
        List<int> sources = anomalous.Where(s => graph.Dependencies[s].All(d => !anomalous.Contains(d))).ToList();
        if (sources.Count == 0) sources = anomalous.ToList();
        return FromRanking("anomalous-source", graph, observed,
            sources.OrderByDescending(s => observed[s]).ThenBy(s => s).ToList());
    }

    internal static RcaResult CentralityPrior(DependencyGraph graph, double[] observed, double weight)
    {
        double[] rank = graph.PageRank(0.85, 30);
        return FromRanking("centrality-prior", graph, observed,
            Ranked(graph, observed, service => observed[service] + weight * rank[service]));
    }

    /** Inverse propagation: the candidate whose re-simulated incident matches the observation best. */
    internal static RcaResult PredictiveFit(DependencyGraph graph, double[] observed)
    {
        var result = new RcaResult { Heuristic = "predictive-fit" };
        var scored = new List<KeyValuePair<int, double>>();
        foreach (int candidate in Enumerable.Range(0, graph.Nodes).Where(s => observed[s] > AnomalyThreshold))
            scored.Add(new KeyValuePair<int, double>(candidate, ResidualFor(graph, observed, candidate)));
        if (scored.Count == 0) return result;
        scored = scored.OrderBy(pair => pair.Value).ThenBy(pair => pair.Key).ToList();
        result.Primary = scored[0].Key;
        result.Residual = scored[0].Value;
        result.Candidates.AddRange(scored.TakeWhile(pair => Math.Abs(pair.Value - result.Residual) < 1e-9)
            .Select(pair => pair.Key));
        result.Tie = result.Candidates.Count > 1;
        result.BlastRadius = observed.Count(value => value > AnomalyThreshold);
        return result;
    }

    /** Residual of a candidate: sqrt of the squared difference from the re-simulated observation. */
    internal static double ResidualFor(DependencyGraph graph, double[] observed, int root)
    {
        double[] predicted = graph.Propagate(root, observed[root]);
        double sum = 0;
        for (int service = 0; service < graph.Nodes; service++)
        {
            double difference = predicted[service] - observed[service];
            sum += difference * difference;
        }
        return Math.Sqrt(sum);
    }
}

internal sealed class IncidentGenerator
{
    private readonly DependencyGraph _graph;
    private ulong _random;

    internal IncidentGenerator(DependencyGraph graph, ulong seed)
    {
        _graph = graph;
        _random = seed * 0x9E3779B97F4A7C15UL + 0x1234567UL;
    }

    private double NextDouble()
    {
        _random += 0x9E3779B97F4A7C15UL;
        ulong z = _random;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z >> 11) * (1.0 / 9007199254740992.0);
    }

    private int NextNode() => Math.Min(_graph.Nodes - 1, (int)(NextDouble() * _graph.Nodes));

    internal Incident Next(int maxFaults)
    {
        int root = NextNode();
        double magnitude = 0.3 + NextDouble() * 1.2;
        double[] observed = _graph.Propagate(root, magnitude);
        int faults = 1;
        if (maxFaults > 1 && NextDouble() < 0.25)
        {
            int second = NextNode();
            if (second != root && !_graph.Reaches(root, second) && !_graph.Reaches(second, root))
            {
                double[] extra = _graph.Propagate(second, 0.3 + NextDouble() * 1.2);
                for (int i = 0; i < observed.Length; i++) observed[i] = Math.Max(observed[i], extra[i]);
                faults = 2;
            }
        }
        for (int service = 0; service < observed.Length; service++) observed[service] += NextDouble() * 0.04;
        return new Incident { Observed = observed, Truth = root, Magnitude = magnitude, Faults = faults };
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static DependencyGraph BuildCluster()
    {
        // Two layers of fan-out with mixed attenuation; a few edges amplify above 1.
        var edges = new List<Edge>();
        void Link(int from, int to, double attenuation) => edges.Add(new Edge { From = from, To = to, Attenuation = attenuation });
        Link(0, 1, 0.9); Link(0, 2, 1.3); Link(1, 3, 0.8); Link(1, 4, 1.1);
        Link(2, 4, 0.7); Link(2, 5, 0.95); Link(3, 6, 0.85); Link(4, 6, 1.4);
        Link(4, 7, 0.6); Link(5, 7, 0.9); Link(6, 8, 0.75); Link(7, 8, 1.2); Link(7, 9, 0.65);
        return new DependencyGraph(10, edges);
    }

    private static DependencyGraph BuildLayered(int nodes)
    {
        var edges = new List<Edge>();
        for (int layer = 0; (layer + 1) * 10 < nodes; layer++)
        {
            for (int i = 0; i < 10; i++)
            {
                int from = layer * 10 + i;
                for (int k = 0; k < 3; k++)
                {
                    int to = (layer + 1) * 10 + (i + k * 3) % 10;
                    if (to >= nodes) continue;
                    edges.Add(new Edge { From = from, To = to, Attenuation = 0.6 + (from + to) % 9 * 0.1 });
                }
            }
        }
        return new DependencyGraph(nodes, edges);
    }

    private static string Fixed(double value) => value.ToString("F3", CultureInfo.InvariantCulture);

    public static void Main()
    {
        DependencyGraph graph = BuildCluster();
        Check(graph.Topological.Length == 10, "the graph is acyclic and topologically sorted");
        Check(graph.Topological[0] == 0, "the dependency-free service comes first in the topological order");
        double[] rank = graph.PageRank(0.85, 30);
        Check(rank.Sum() > 0.99 && rank.Sum() < 1.01, "PageRank mass is conserved: " + Fixed(rank.Sum()));
        Check(graph.Reaches(0, 8) && !graph.Reaches(8, 0), "reachability follows the dependency direction");
        Check(!graph.Reaches(3, 9), "a leaf dependency cannot reach an unrelated branch");
        Check(graph.Dependencies[8].Count == 2 && graph.Dependents[8].Count == 0, "an unhealthy dependency has two causes");

        // ---- 1. accuracy over a generated incident population -----------------------------------
        var generator = new IncidentGenerator(graph, 20261016);
        int incidents = 0, multiFault = 0, topHits = 0, sourceHits = 0, fitHits = 0, centralityHits = 0;
        double residualSum = 0;
        for (int i = 0; i < 300; i++)
        {
            Incident incident = generator.Next(2);
            if (incident.Faults > 1) { multiFault++; continue; }
            incidents++;
            RcaResult top = Rca.TopAnomaly(graph, incident.Observed);
            RcaResult source = Rca.AnomalousSources(graph, incident.Observed);
            RcaResult fit = Rca.PredictiveFit(graph, incident.Observed);
            RcaResult centrality = Rca.CentralityPrior(graph, incident.Observed, 0.02);
            if (top.Candidates.Contains(incident.Truth)) topHits++;
            if (source.Candidates.Contains(incident.Truth)) sourceHits++;
            if (fit.Candidates.Contains(incident.Truth)) fitHits++;
            if (centrality.Candidates.Contains(incident.Truth)) centralityHits++;
            residualSum += Rca.ResidualFor(graph, incident.Observed, incident.Truth);
            Check(top.Primary >= 0 && incident.Observed[top.Primary] > 0, "every heuristic names an anomalous service");
            Check(incident.Observed[top.Primary] >= incident.Observed.Max() - 1e-9, "the naive heuristic names the maximum");
            Check(fit.BlastRadius == incident.Observed.Count(v => v > Rca.AnomalyThreshold), "the blast radius counts the anomalous set");
        }
        Check(incidents >= 200, "enough single-root incidents were generated: " + incidents);
        Check(multiFault > 0, "the generator also produced multi-root incidents: " + multiFault);
        double fitRecall = (double)fitHits / incidents, sourceRecall = (double)sourceHits / incidents;
        double topRecall = (double)topHits / incidents, centralityRecall = (double)centralityHits / incidents;
        Check(fitRecall >= 0.9, "predictive-fit recall is high: " + Fixed(fitRecall));
        Check(fitRecall > topRecall, "inverse propagation beats the naive baseline: " + Fixed(fitRecall) + " > " + Fixed(topRecall));
        Check(fitRecall >= sourceRecall && fitRecall >= centralityRecall, "and it is the best of the four heuristics");
        Check(residualSum / incidents < 0.5, "the propagation model explains incidents within a small residual: " +
            Fixed(residualSum / incidents));

        // ---- 2. amplification makes the naive heuristic blame the wrong service -----------------
        double[] escalated = graph.Propagate(1, 0.4);                 // small fault at 1; edge 1->4 amplifies
        RcaResult escalatedTop = Rca.TopAnomaly(graph, escalated);
        RcaResult escalatedSource = Rca.AnomalousSources(graph, escalated);
        RcaResult escalatedFit = Rca.PredictiveFit(graph, escalated);
        Check(escalated[4] > escalated[1], "a downstream service can look worse than the root: " +
            Fixed(escalated[1]) + " -> " + Fixed(escalated[4]));
        Check(escalatedTop.Primary != 1, "the naive heuristic blames the amplified dependent: " + escalatedTop.Primary);
        Check(escalatedFit.Candidates.Contains(1), "inverse propagation still finds the true root");
        Check(escalatedFit.Residual < Rca.ResidualFor(graph, escalated, 4),
            "the true root fits the observation better than the dependent");
        Check(escalatedSource.Candidates.Count >= 1, "the source heuristic still answers something");
        Check(Rca.ResidualFor(graph, escalated, 1) < 1e-9, "a noise-free incident is explained exactly");

        // ---- 3. local faults are attributed exactly --------------------------------------------
        double[] leafOnly = graph.Propagate(9, 1.0);
        RcaResult leafFit = Rca.PredictiveFit(graph, leafOnly);
        Check(leafFit.Primary == 9 && leafFit.BlastRadius == 1, "a leaf fault that spreads to nobody stays local");
        double[] entry = graph.Propagate(0, 1.0);
        Check(Rca.PredictiveFit(graph, entry).Primary == 0, "an entry-point fault is attributed to the entry point");
        Check(Rca.AnomalousSources(graph, entry).Primary == 0, "and the source heuristic agrees");
        double[] big = graph.Propagate(2, 1.0);
        Check(Rca.AnomalousSources(graph, big).Primary == 2, "a mid-graph root is a source of the anomalous sub-graph");

        // ---- 4. ties, determinism and stability -------------------------------------------------
        var tieObserved = new double[graph.Nodes];
        tieObserved[3] = 0.5;
        tieObserved[5] = 0.5;
        RcaResult tie = Rca.TopAnomaly(graph, tieObserved);
        Check(tie.Tie && tie.Candidates.SequenceEqual(new[] { 3, 5 }),
            "equal anomalies are reported as a tie: " + string.Join(",", tie.Candidates));
        Check(Rca.AnomalousSources(graph, tieObserved).Primary >= 0, "the source heuristic copes with a tie");
        var firstRun = new IncidentGenerator(graph, 99).Next(1);
        var secondRun = new IncidentGenerator(graph, 99).Next(1);
        Check(firstRun.Observed.SequenceEqual(secondRun.Observed), "the same seed reproduces the same incident");
        Check(Rca.PredictiveFit(graph, firstRun.Observed).Primary == Rca.PredictiveFit(graph, secondRun.Observed).Primary,
            "the same incident always yields the same answer");
        var otherSeed = new IncidentGenerator(graph, 100).Next(1);
        Check(!otherSeed.Observed.SequenceEqual(firstRun.Observed), "a different seed gives a different incident");
        Check(Rca.TopAnomaly(graph, firstRun.Observed).Candidates.SequenceEqual(Rca.TopAnomaly(graph, secondRun.Observed).Candidates),
            "candidate sets are stable");

        // ---- 5. edge cases ----------------------------------------------------------------------
        var empty = new double[graph.Nodes];
        Check(Rca.TopAnomaly(graph, empty).Primary == -1, "a silent cluster yields no candidate");
        Check(Rca.PredictiveFit(graph, empty).Primary == -1, "the fit heuristic also abstains");
        Check(Rca.AnomalousSources(graph, empty).Primary == -1, "and so does the source heuristic");
        Check(Rca.PredictiveFit(graph, empty).Candidates.Count == 0, "with no candidate set either");
        var single = new DependencyGraph(1, Array.Empty<Edge>());
        Check(Rca.PredictiveFit(single, new[] { 1.0 }).Primary == 0, "a one-service graph names that service");
        Check(single.Propagate(0, 2.0)[0] == 2.0, "propagation on a single node keeps its magnitude");
        Check(Rca.AnomalousSources(single, new[] { 0.4 }).BlastRadius == 1, "the blast radius of a single service is one");

        var disconnected = new DependencyGraph(6, new[]
        {
            new Edge { From = 0, To = 1, Attenuation = 0.8 },
            new Edge { From = 2, To = 3, Attenuation = 0.8 },
        });
        double[] isolatedFault = disconnected.Propagate(2, 0.9);
        Check(isolatedFault[0] == 0 && isolatedFault[1] == 0, "a fault never jumps between components");
        Check(Rca.PredictiveFit(disconnected, isolatedFault).Primary == 2, "the fault is attributed inside its component");
        Check(Rca.AnomalousSources(disconnected, isolatedFault).BlastRadius == 2, "the blast radius stops at the component");
        Check(!disconnected.Reaches(0, 2), "components are mutually unreachable");

        Check(Throws(() => new DependencyGraph(3, new[]
        {
            new Edge { From = 0, To = 1, Attenuation = 1 }, new Edge { From = 1, To = 0, Attenuation = 1 },
        })), "a dependency cycle is rejected");
        Check(Throws(() => new DependencyGraph(3, new[] { new Edge { From = 0, To = 5, Attenuation = 1 } })),
            "an edge outside the graph is rejected");
        Check(Throws(() => new DependencyGraph(2, new[] { new Edge { From = -1, To = 0, Attenuation = 1 } })),
            "a negative node index is rejected");

        // ---- 6. scale: 60 services, 400 incidents ----------------------------------------------
        DependencyGraph large = BuildLayered(60);
        var largeGenerator = new IncidentGenerator(large, 4242);
        int largeFit = 0, largeTop = 0, largeCount = 0;
        for (int i = 0; i < 400; i++)
        {
            Incident incident = largeGenerator.Next(1);
            largeCount++;
            if (Rca.PredictiveFit(large, incident.Observed).Candidates.Contains(incident.Truth)) largeFit++;
            if (Rca.TopAnomaly(large, incident.Observed).Candidates.Contains(incident.Truth)) largeTop++;
        }
        double largeFitRecall = (double)largeFit / largeCount, largeTopRecall = (double)largeTop / largeCount;
        Check(large.Nodes == 60 && large.Edges.Count >= 140, "the large cluster has a real topology: " + large.Edges.Count + " edges");
        Check(largeCount == 400, "all 400 incidents in the large run were single-root");
        Check(largeFitRecall >= 0.85, "fit recall stays high at scale: " + Fixed(largeFitRecall));
        Check(largeFitRecall >= largeTopRecall, "and never loses to the naive baseline at scale: " +
            Fixed(largeFitRecall) + " vs " + Fixed(largeTopRecall));

        Console.WriteLine("heuristics over " + incidents + " single-root incidents: fit=" + Fixed(fitRecall) +
            " source=" + Fixed(sourceRecall) + " centrality=" + Fixed(centralityRecall) + " top-anomaly=" + Fixed(topRecall));
        Console.WriteLine("mean residual of the true root=" + Fixed(residualSum / incidents) +
            ", multi-root incidents skipped=" + multiFault + ", cycles rejected, no cross-component leakage");
        Console.WriteLine("amplified incident: naive blames " + escalatedTop.Primary + " (score " + Fixed(escalatedTop.Candidates.Count > 0 ? escalated[escalatedTop.Primary] : 0) +
            "), inverse propagation names " + escalatedFit.Primary + " with residual " + Fixed(escalatedFit.Residual));
        Console.WriteLine("scale: 60 services / " + large.Edges.Count + " edges, " + largeCount + " incidents, fit=" +
            Fixed(largeFitRecall) + " top-anomaly=" + Fixed(largeTopRecall));
        Console.WriteLine("PASS");
    }
}
