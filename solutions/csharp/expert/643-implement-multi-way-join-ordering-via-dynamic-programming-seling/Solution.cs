// Task 643 - Implement multi-way join ordering via dynamic programming (Selinger-style). (Tier: Expert, CTCI Ch 14 - Databases)
//
// The Selinger/System-R dynamic program over subsets: a state is a subset of the relations and the
// cheapest left-deep plan that joins exactly that subset. Extending a state by one more relation is
// only allowed when at least one join predicate connects that relation to the subset (so no step is
// a cross product). Estimated cardinality of a subset is the textbook independence formula
//     card(S) = Prod_{r in S} |r| * Prod_{p: both endpoints in S} selectivity(p),
// which depends on the *set* and not on the order, and that is exactly what makes pruning by cost
// per subset safe - the same invariant is checked below against a keep-every-plan variant and
// against exhaustive enumeration of all valid left-deep orders.
//
// Simplified: one join method (a hash join whose build side is the newly added relation) with fixed
// CPU constants, no sort-merge alternative, no interesting orders / sort avoidance, no bushy plans
// and no physical property tracking. Selectivities are pre-computed constants rather than derived
// from histograms (task 641 does that part).
//
// Verified: hand-computed subset cardinalities, the DP optimum equals the exhaustive optimum over
// every connected permutation, the pruning variant equals the unpruned variant, every emitted step
// is connected by a predicate, and the DP touches far fewer candidates than exhaustive enumeration.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class Relation
{
    public readonly string Name;
    public readonly double Rows;

    public Relation(string name, double rows)
    {
        Name = name;
        Rows = rows;
    }
}

internal sealed class JoinEdge
{
    public readonly int A;
    public readonly int B;
    public readonly double Selectivity;

    public JoinEdge(int a, int b, double selectivity)
    {
        A = a;
        B = b;
        Selectivity = selectivity;
    }
}

internal sealed class JoinQuery
{
    public readonly string Name;
    public readonly List<Relation> Relations;
    public readonly List<JoinEdge> Edges;

    public JoinQuery(string name, List<Relation> relations, List<JoinEdge> edges)
    {
        Name = name;
        Relations = relations;
        Edges = edges;
    }

    // Product of the selectivities of every predicate whose other endpoint is already inside `mask`.
    public double ConnectingSelectivity(int mask, int relation)
    {
        double selectivity = 1.0;
        bool found = false;
        foreach (JoinEdge edge in Edges)
        {
            if (edge.A == relation && (mask & (1 << edge.B)) != 0) { selectivity *= edge.Selectivity; found = true; }
            else if (edge.B == relation && (mask & (1 << edge.A)) != 0) { selectivity *= edge.Selectivity; found = true; }
        }
        return found ? selectivity : 0.0;
    }

    public bool IsConnected(int mask, int relation) => ConnectingSelectivity(mask, relation) > 0.0;

    // card(S) = Prod |r| * Prod selectivity of the predicates internal to S.
    public double Cardinality(int mask)
    {
        double rows = 1.0;
        for (int i = 0; i < Relations.Count; i++)
        {
            if ((mask & (1 << i)) != 0) rows *= Relations[i].Rows;
        }
        foreach (JoinEdge edge in Edges)
        {
            if ((mask & (1 << edge.A)) != 0 && (mask & (1 << edge.B)) != 0) rows *= edge.Selectivity;
        }
        return Math.Max(1.0, rows);
    }

    public bool IsConnectedSet(int mask)
    {
        int start = -1;
        for (int i = 0; i < Relations.Count; i++)
        {
            if ((mask & (1 << i)) != 0) { start = i; break; }
        }
        if (start < 0) return false;
        int reached = 1 << start;
        var frontier = new Queue<int>();
        frontier.Enqueue(start);
        while (frontier.Count > 0)
        {
            int current = frontier.Dequeue();
            foreach (JoinEdge edge in Edges)
            {
                int other = edge.A == current ? edge.B : edge.B == current ? edge.A : -1;
                if (other < 0) continue;
                if ((mask & (1 << other)) == 0) continue;
                if ((reached & (1 << other)) != 0) continue;
                reached |= 1 << other;
                frontier.Enqueue(other);
            }
        }
        return (reached & mask) == mask;
    }
}

internal sealed class JoinPlan
{
    public readonly string Description;
    public readonly double Cost;
    public readonly double Rows;
    public readonly List<int> Order;
    public readonly List<double> StepRows;

    public JoinPlan(string description, double cost, double rows, List<int> order, List<double> stepRows)
    {
        Description = description;
        Cost = cost;
        Rows = rows;
        Order = order;
        StepRows = stepRows;
    }
}

internal sealed class PlannerMetrics
{
    public int Transitions;
    public int KeptStates;
    public int ValidPermutations;
    public int TotalPermutations;
}

internal static class SelingerPlanner
{
    public const double BuildCost = 0.02;
    public const double ProbeCost = 0.01;
    public const double CpuCost = 0.01;

    // The single-step cost used by every enumeration strategy, so the optima are comparable.
    public static double StepCost(double currentRows, double newRelationRows, double outputRows)
        => newRelationRows * BuildCost + currentRows * ProbeCost + outputRows * CpuCost;

    public static JoinPlan Best(JoinQuery query) => DynamicProgram(query, new PlannerMetrics());

    public static JoinPlan DynamicProgram(JoinQuery query, PlannerMetrics metrics)
    {
        int n = query.Relations.Count;
        var best = new JoinPlan?[1 << n];
        for (int i = 0; i < n; i++)
        {
            double rows = query.Relations[i].Rows;
            best[1 << i] = new JoinPlan(query.Relations[i].Name, 0.0, rows, new List<int> { i }, new List<double> { rows });
            metrics.KeptStates++;
        }

        for (int mask = 1; mask < (1 << n); mask++)
        {
            JoinPlan? current = best[mask];
            if (current == null) continue;
            for (int r = 0; r < n; r++)
            {
                if ((mask & (1 << r)) != 0) continue;
                if (!query.IsConnected(mask, r)) continue;
                metrics.Transitions++;
                int next = mask | (1 << r);
                double rows = query.Cardinality(next);
                double cost = current.Cost + StepCost(current.Rows, query.Relations[r].Rows, rows);
                JoinPlan? existing = best[next];
                if (existing != null && cost >= existing.Cost - 1e-12) continue;
                var order = new List<int>(current.Order) { r };
                var stepRows = new List<double>(current.StepRows) { rows };
                best[next] = new JoinPlan(current.Description + " | " + query.Relations[r].Name, cost, rows, order, stepRows);
                metrics.KeptStates++;
            }
        }
        return best[(1 << n) - 1] ?? throw new Exception("no connected left-deep plan exists for " + query.Name);
    }

    // Same DP but every plan for every subset is retained, i.e. no pruning at all: the cheapest full
    // plan must coincide with the pruned search.
    public static JoinPlan BestWithoutPruning(JoinQuery query, PlannerMetrics metrics)
    {
        int n = query.Relations.Count;
        var all = new List<JoinPlan>[1 << n];
        for (int i = 0; i < n; i++)
        {
            all[1 << i] = new List<JoinPlan>
            {
                new JoinPlan(query.Relations[i].Name, 0.0, query.Relations[i].Rows, new List<int> { i },
                    new List<double> { query.Relations[i].Rows }),
            };
        }
        for (int mask = 1; mask < (1 << n); mask++)
        {
            if (all[mask] == null || all[mask].Count == 0) continue;
            for (int r = 0; r < n; r++)
            {
                if ((mask & (1 << r)) != 0 || !query.IsConnected(mask, r)) continue;
                int next = mask | (1 << r);
                if (all[next] == null) all[next] = new List<JoinPlan>();
                double rows = query.Cardinality(next);
                foreach (JoinPlan plan in all[mask])
                {
                    metrics.Transitions++;
                    double cost = plan.Cost + StepCost(plan.Rows, query.Relations[r].Rows, rows);
                    all[next].Add(new JoinPlan(plan.Description + " | " + query.Relations[r].Name, cost, rows,
                        new List<int>(plan.Order) { r }, new List<double>(plan.StepRows) { rows }));
                }
            }
        }
        List<JoinPlan> full = all[(1 << n) - 1] ?? throw new Exception("no plan");
        metrics.KeptStates = all.Where(list => list != null).Sum(list => list!.Count);
        return full.OrderBy(p => p.Cost).ThenBy(p => p.Description, StringComparer.Ordinal).First();
    }

    // Every permutation of the relations, scored left-deep, keeping only the orders in which each
    // added relation is connected to the relations already joined.
    public static JoinPlan BestExhaustive(JoinQuery query, PlannerMetrics metrics)
    {
        int n = query.Relations.Count;
        JoinPlan? best = null;
        foreach (int[] permutation in Permutations(n))
        {
            metrics.TotalPermutations++;
            int mask = 0;
            double cost = 0.0;
            double rows = 1.0;
            bool valid = true;
            var description = new List<string>();
            var stepRows = new List<double>();
            for (int i = 0; i < permutation.Length; i++)
            {
                int relation = permutation[i];
                if (i > 0 && !query.IsConnected(mask, relation)) { valid = false; break; }
                double next = query.Cardinality(mask | (1 << relation));
                if (i > 0) cost += StepCost(rows, query.Relations[relation].Rows, next);
                mask |= 1 << relation;
                rows = next;
                description.Add(query.Relations[relation].Name);
                stepRows.Add(rows);
            }
            if (!valid) continue;
            metrics.ValidPermutations++;
            var plan = new JoinPlan(string.Join(" | ", description), cost, rows, permutation.ToList(), stepRows);
            if (best == null || plan.Cost < best.Cost - 1e-12
                || (Math.Abs(plan.Cost - best.Cost) < 1e-12 && string.CompareOrdinal(plan.Description, best.Description) < 0))
            {
                best = plan;
            }
        }
        return best ?? throw new Exception("no valid left-deep order");
    }

    public static List<int[]> Permutations(int n)
    {
        var result = new List<int[]>();
        Walk(0, n, new int[n], new bool[n], result);
        return result;
    }

    private static void Walk(int depth, int n, int[] current, bool[] used, List<int[]> result)
    {
        if (depth == n)
        {
            result.Add((int[])current.Clone());
            return;
        }
        for (int i = 0; i < n; i++)
        {
            if (used[i]) continue;
            used[i] = true;
            current[depth] = i;
            Walk(depth + 1, n, current, used, result);
            used[i] = false;
        }
    }
}

public static class Solution
{
    private static string F(double value) => value.ToString("0.####", CultureInfo.InvariantCulture);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void Close(double actual, double expected, string label)
        => Check(Math.Abs(actual - expected) < 1e-6, label + ": expected " + F(expected) + " got " + F(actual));

    private static JoinQuery Star()
    {
        var relations = new List<Relation>
        {
            new Relation("fact", 1000), new Relation("dim1", 100), new Relation("dim2", 50), new Relation("dim3", 20),
        };
        var edges = new List<JoinEdge>
        {
            new JoinEdge(0, 1, 1.0 / 100), new JoinEdge(0, 2, 1.0 / 50), new JoinEdge(0, 3, 1.0 / 20),
        };
        return new JoinQuery("star", relations, edges);
    }

    private static JoinQuery Chain()
    {
        var relations = new List<Relation>();
        for (int i = 0; i < 7; i++) relations.Add(new Relation("r" + i, 200 + 50 * i));
        var edges = new List<JoinEdge>();
        for (int i = 0; i + 1 < relations.Count; i++) edges.Add(new JoinEdge(i, i + 1, 1.0 / (100 + 10 * i)));
        return new JoinQuery("chain", relations, edges);
    }

    private static JoinQuery Small()
    {
        return new JoinQuery("small",
            new List<Relation> { new Relation("R", 1000), new Relation("S", 100), new Relation("T", 50) },
            new List<JoinEdge> { new JoinEdge(0, 1, 1.0 / 100), new JoinEdge(1, 2, 1.0 / 10) });
    }

    private static void CardinalityFormula()
    {
        JoinQuery query = Small();
        Close(query.Cardinality(0b001), 1000, "card({R}) = |R|");
        Close(query.Cardinality(0b011), 1000 * 100 / 100.0, "card({R,S}) = |R||S| / ndv(R.s)");
        Close(query.Cardinality(0b101), 1000 * 50, "card({R,T}) with no predicate is the cross product");
        Close(query.Cardinality(0b111), 1000 * 100 * 50 / 100.0 / 10.0, "card({R,S,T}) multiplies both selectivities");
        Check(query.Cardinality(0b111) < query.Cardinality(0b101), "an extra selective predicate shrinks the estimate");
        Check(query.IsConnectedSet(0b111) && query.IsConnectedSet(0b011) && !query.IsConnectedSet(0b101),
            "connectivity: {R,T} alone is not connected");
        Console.WriteLine("cardinalities: {R}=1000 {R,S}=1000 {R,S,T}=5000");
    }

    private static void DynamicProgramMatchesExhaustive()
    {
        foreach (JoinQuery query in new[] { Small(), Star(), Chain() })
        {
            var dpMetrics = new PlannerMetrics();
            JoinPlan dp = SelingerPlanner.DynamicProgram(query, dpMetrics);
            var exhaustiveMetrics = new PlannerMetrics();
            JoinPlan exhaustive = SelingerPlanner.BestExhaustive(query, exhaustiveMetrics);

            Close(dp.Cost, exhaustive.Cost, "DP optimum equals the exhaustive optimum for " + query.Name);
            Check(dp.Order.Count == query.Relations.Count, "the DP plan joins every relation once");
            Check(dp.Order.Distinct().Count() == dp.Order.Count, "the DP plan has no repeated relation");
            Check(exhaustiveMetrics.ValidPermutations > 0 && exhaustiveMetrics.ValidPermutations <= exhaustiveMetrics.TotalPermutations,
                "exhaustive enumeration counted its valid orders");
            Check(dpMetrics.Transitions <= exhaustiveMetrics.TotalPermutations,
                "the DP considers no more candidates than exhaustive enumeration for " + query.Name);
            Console.WriteLine(query.Name + ": dp cost=" + F(dp.Cost) + " exhaustive cost=" + F(exhaustive.Cost)
                + " (transitions " + dpMetrics.Transitions + " vs " + exhaustiveMetrics.TotalPermutations + " permutations, "
                + exhaustiveMetrics.ValidPermutations + " valid)");
        }

        var chainMetrics = new PlannerMetrics();
        SelingerPlanner.DynamicProgram(Chain(), chainMetrics);
        Check(chainMetrics.Transitions < 5040,
            "the DP explores far fewer candidates than the 5040 orders of a 7-relation chain, got " + chainMetrics.Transitions);
        Check(chainMetrics.KeptStates <= 1 + 6 * 6, "the DP keeps at most one plan per subset");
        Console.WriteLine("chain detail: " + chainMetrics.Transitions + " transitions and " + chainMetrics.KeptStates + " states vs 5040 orders");
    }

    private static void PruningIsSafe()
    {
        foreach (JoinQuery query in new[] { Small(), Star() })
        {
            var prunedMetrics = new PlannerMetrics();
            var unprunedMetrics = new PlannerMetrics();
            JoinPlan pruned = SelingerPlanner.DynamicProgram(query, prunedMetrics);
            JoinPlan unpruned = SelingerPlanner.BestWithoutPruning(query, unprunedMetrics);
            Close(pruned.Cost, unpruned.Cost, "keeping only the best plan per subset matches keeping all of them (" + query.Name + ")");
            Check(prunedMetrics.KeptStates <= unprunedMetrics.KeptStates,
                "pruning keeps fewer states than the unpruned search (" + query.Name + ")");
            Check(prunedMetrics.Transitions <= unprunedMetrics.Transitions,
                "pruning explores no more transitions than the unpruned search (" + query.Name + ")");
            Console.WriteLine(query.Name + ": pruned states=" + prunedMetrics.KeptStates
                + " unpruned=" + unprunedMetrics.KeptStates);
        }
    }

    private static void PlanValidity()
    {
        JoinQuery query = Star();
        var metrics = new PlannerMetrics();
        JoinPlan plan = SelingerPlanner.DynamicProgram(query, metrics);
        int mask = 0;
        for (int i = 0; i < plan.Order.Count; i++)
        {
            int relation = plan.Order[i];
            if (i > 0)
            {
                Check(query.IsConnected(mask, relation), "every step of the plan is connected by a predicate");
            }
            mask |= 1 << relation;
            Close(plan.StepRows[i], query.Cardinality(mask), "step " + i + " reports the cardinality of its prefix subset");
            Check(plan.StepRows[i] >= 1, "no step estimates fewer than one row");
        }
        Check(mask == (1 << query.Relations.Count) - 1, "the plan covers the whole query");

        double rebuilt = 0.0;
        double rows = 0.0;
        for (int i = 0; i < plan.Order.Count; i++)
        {
            if (i == 0)
            {
                rows = plan.StepRows[0];
                continue;
            }
            rebuilt += SelingerPlanner.StepCost(rows, query.Relations[plan.Order[i]].Rows, plan.StepRows[i]);
            rows = plan.StepRows[i];
        }
        Close(plan.Cost, rebuilt, "the reported plan cost is the sum of its step costs");

        JoinPlan again = SelingerPlanner.DynamicProgram(Star(), new PlannerMetrics());
        Check(again.Description == plan.Description && Math.Abs(again.Cost - plan.Cost) < 1e-12, "planning is deterministic");
    }

    private static void Boundaries()
    {
        var single = new JoinQuery("single", new List<Relation> { new Relation("only", 42) }, new List<JoinEdge>());
        JoinPlan one = SelingerPlanner.Best(single);
        Close(one.Cost, 0.0, "a single relation needs no join and costs nothing");
        Close(one.Rows, 42, "a single relation keeps its cardinality");

        var pair = new JoinQuery("pair",
            new List<Relation> { new Relation("a", 10), new Relation("b", 20) },
            new List<JoinEdge> { new JoinEdge(0, 1, 0.5) });
        JoinPlan two = SelingerPlanner.Best(pair);
        Close(two.Rows, 100, "two relations join to 10 * 20 * 0.5 = 100 rows");
        double expected = Math.Min(SelingerPlanner.StepCost(10, 20, 100), SelingerPlanner.StepCost(20, 10, 100));
        Close(two.Cost, expected, "the two-relation plan costs exactly one join step");

        bool threw = false;
        try
        {
            var disconnected = new JoinQuery("disconnected",
                new List<Relation> { new Relation("x", 10), new Relation("y", 10) }, new List<JoinEdge>());
            SelingerPlanner.Best(disconnected);
        }
        catch (Exception) { threw = true; }
        Check(threw, "a query with no connecting predicate is rejected instead of silently using a cross product");

        var chain3 = new JoinQuery("chain3",
            new List<Relation> { new Relation("a", 100), new Relation("b", 100), new Relation("c", 100) },
            new List<JoinEdge> { new JoinEdge(0, 1, 0.01), new JoinEdge(1, 2, 0.01) });
        var metrics = new PlannerMetrics();
        JoinPlan plan = SelingerPlanner.DynamicProgram(chain3, metrics);
        Check(!(plan.Order[0] == 0 && plan.Order[1] == 2), "the middle relation is joined before the two ends");
        Console.WriteLine("boundaries: 1 relation, 2 relations, disconnected query rejected");
    }

    public static void Main()
    {
        CardinalityFormula();
        DynamicProgramMatchesExhaustive();
        PruningIsSafe();
        PlanValidity();
        Boundaries();
        Console.WriteLine("PASS");
    }
}
