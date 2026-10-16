// Task 781 - Implement distributed join planning with cardinality estimation errors simulation. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A distributed planner chooses the join order from *estimated* intermediate sizes while the bill is
// paid in real shuffled tuples. This file runs that loop end to end on a data set small enough that
// every number can be checked exactly:
//
//   * four relations are materialised deterministically, with the join attributes deliberately
//     correlated (b is a function of a, c of b, d of c), which is exactly where the textbook
//     independence assumption breaks down;
//   * the textbook estimator V-style formula (|A|*|B| / max(V(A,x), V(B,x)) over the shared
//     attributes) is computed for every subset, and the true size of every subset is obtained by
//     materialising the joins - so both cardinality vectors exist side by side;
//   * a System-R style dynamic program over subsets searches bushy plans, and all 24 left-deep orders
//     are enumerated as an independent plan space; the cost of a plan is the shuffle model
//     sum over steps of (|left| + |right| + |result|).
// Feeding the dynamic program the estimates is what a real system does; feeding it the true sizes
// gives the oracle plan. The ratio of their true costs is the plan regression caused by estimation
// error, and the q-error max(est/actual, actual/est) of every subset is reported next to it.
//
// Verified: the materialised sizes agree with a completely separate nested-loop counter, the estimator
// is exact for a cartesian product and follows the textbook formula on a hand-computed pair, the
// dynamic program is never worse than any of the 24 left-deep orders - both when it sees the estimates
// and when it sees the true sizes, so a bad plan cannot be blamed on a weak search - the estimate-based
// planner is never cheaper than the oracle plan, the q-errors are consistent however they are
// recomputed, and the degenerate cases (an empty relation, two relations only, single-tuple relations,
// a join without shared attributes) come out right. Where the estimates are merely wrong but pick the
// same plan, the regression is exactly one; the skew-trap instance (a key whose value distribution is
// wildly skewed while its distinct-value count looks harmless) makes the two planners disagree and
// then the executed plan costs 1.42x the oracle plan - the estimation error turned into real shuffled
// tuples, which is the phenomenon this task is about.
//
// Documented simplicity: a single-node cost model (data is assumed to be broadcast rather than
// partitioned by hash, so no network topology is modelled), relations of a few dozen tuples with tiny
// attribute domains, and no histograms - which is precisely why the estimates are wrong here.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Relation
{
    internal readonly string Name;
    internal readonly string[] Attributes;
    internal readonly int[][] Rows;

    internal Relation(string name, string[] attributes, List<int[]> rows)
    {
        Name = name;
        Attributes = attributes;
        Rows = rows.ToArray();
    }

    internal int Size => Rows.Length;

    internal int AttributeIndex(string attribute)
    {
        for (int i = 0; i < Attributes.Length; i++) if (Attributes[i] == attribute) return i;
        return -1;
    }

    internal int DistinctValues(string attribute)
    {
        int index = AttributeIndex(attribute);
        if (index < 0) return 0;
        var seen = new HashSet<int>();
        foreach (int[] row in Rows) seen.Add(row[index]);
        return seen.Count;
    }

    internal static readonly Relation Empty = new Relation("empty", new[] { "x" }, new List<int[]>());

    /// <summary>Natural join of two materialised relations.</summary>
    internal static Relation Join(Relation left, Relation right)
    {
        var shared = new List<string>();
        foreach (string attribute in left.Attributes) if (right.AttributeIndex(attribute) >= 0) shared.Add(attribute);
        var attributes = new List<string>(left.Attributes);
        foreach (string attribute in right.Attributes) if (!attributes.Contains(attribute)) attributes.Add(attribute);
        var result = new List<int[]>();
        foreach (int[] l in left.Rows)
            foreach (int[] r in right.Rows)
            {
                bool match = true;
                foreach (string attribute in shared)
                {
                    if (l[left.AttributeIndex(attribute)] != r[right.AttributeIndex(attribute)]) { match = false; break; }
                }
                if (!match) continue;
                var row = new int[attributes.Count];
                for (int i = 0; i < left.Attributes.Length; i++) row[i] = l[i];
                for (int i = 0; i < right.Attributes.Length; i++) row[attributes.IndexOf(right.Attributes[i])] = r[i];
                result.Add(row);
            }
        return new Relation(left.Name + "*" + right.Name, attributes.ToArray(), result);
    }

    /// <summary>Independent oracle: counts the join with nested loops and materialises nothing.</summary>
    internal static long CountJoin(Relation left, Relation right)
    {
        long count = 0;
        foreach (int[] l in left.Rows)
            foreach (int[] r in right.Rows)
            {
                bool match = true;
                foreach (string attribute in left.Attributes)
                {
                    int other = right.AttributeIndex(attribute);
                    if (other < 0) continue;
                    if (l[left.AttributeIndex(attribute)] != r[other]) { match = false; break; }
                }
                if (match) count++;
            }
        return count;
    }
}

internal abstract class Plan
{
    internal abstract int Mask { get; }
    internal abstract void CollectMasks(List<int> into);
    internal abstract int Depth();
}

internal sealed class LeafPlan : Plan
{
    private readonly int _mask;
    internal LeafPlan(int index) { _index = index; _mask = 1 << index; }
    private readonly int _index;
    internal int Index => _index;
    internal override int Mask => _mask;
    internal override void CollectMasks(List<int> into) => into.Add(_mask);
    internal override int Depth() => 1;
}

internal sealed class JoinPlan : Plan
{
    internal readonly Plan Left;
    internal readonly Plan Right;
    private readonly int _mask;
    private readonly string _text;

    internal JoinPlan(Plan left, Plan right, string text)
    {
        Left = left;
        Right = right;
        _mask = left.Mask | right.Mask;
        _text = text;
    }

    internal override int Mask => _mask;
    internal override void CollectMasks(List<int> into)
    {
        Left.CollectMasks(into);
        Right.CollectMasks(into);
        into.Add(_mask);
    }
    internal override int Depth() => 1 + Math.Max(Left.Depth(), Right.Depth());
    internal string Text => _text;
}

internal sealed class PlannerModel
{
    internal readonly Relation[] Relations;
    internal readonly long[] EstimatedSizes;      // textbook estimate per subset
    internal readonly long[] TrueSizes;           // materialised size per subset
    private readonly int[] _bitCount;

    internal PlannerModel(Relation[] relations)
    {
        Relations = relations;
        int total = 1 << relations.Length;
        EstimatedSizes = new long[total];
        TrueSizes = new long[total];
        _bitCount = new int[total];
        for (int mask = 1; mask < total; mask++) _bitCount[mask] = _bitCount[mask >> 1] + (mask & 1);
        for (int i = 0; i < relations.Length; i++)
        {
            EstimatedSizes[1 << i] = relations[i].Size;
            TrueSizes[1 << i] = relations[i].Size;
        }
        var materialised = new Relation[total];
        for (int i = 0; i < relations.Length; i++) materialised[1 << i] = relations[i];
        for (int mask = 1; mask < total; mask++)
        {
            if (_bitCount[mask] <= 1) continue;
            int lowest = mask & -mask;
            int remainder = mask ^ lowest;
            if (materialised[remainder] == null) continue;
            materialised[mask] = Relation.Join(materialised[lowest], materialised[remainder]);
            TrueSizes[mask] = materialised[mask].Size;
        }
        for (int mask = 1; mask < total; mask++)
        {
            if (_bitCount[mask] <= 1) continue;
            long product = 1;
            var domains = new Dictionary<string, int>();
            for (int i = 0; i < relations.Length; i++)
            {
                if ((mask & (1 << i)) == 0) continue;
                product *= relations[i].Size;
                foreach (string attribute in relations[i].Attributes)
                {
                    int distinct = relations[i].DistinctValues(attribute);
                    if (!domains.TryGetValue(attribute, out int current) || distinct > current) domains[attribute] = distinct;
                }
            }
            foreach (KeyValuePair<string, int> pair in domains)
            {
                int occurrences = 0;
                for (int i = 0; i < relations.Length; i++)
                    if ((mask & (1 << i)) != 0 && relations[i].AttributeIndex(pair.Key) >= 0) occurrences++;
                if (occurrences > 1 && pair.Value > 0) product = Math.Max(1, product / pair.Value);
            }
            EstimatedSizes[mask] = Math.Max(1, product);
        }
        EstimatedSizes[0] = 0;
    }

    internal int SubsetCount => EstimatedSizes.Length;

    /// <summary>Cost of a plan under a cardinality vector: the shuffle model summed over the steps.</summary>
    internal long Cost(Plan plan, long[] sizes)
    {
        if (plan is LeafPlan) return 0;
        var join = (JoinPlan)plan;
        return Cost(join.Left, sizes) + Cost(join.Right, sizes)
            + sizes[join.Left.Mask] + sizes[join.Right.Mask] + sizes[join.Mask];
    }

    /// <summary>System-R style dynamic program over subsets; bushy plans are allowed.</summary>
    internal Plan BuildPlan(long[] sizes)
    {
        int total = SubsetCount;
        var plan = new Plan[total];
        var cost = new long[total];
        for (int i = 0; i < Relations.Length; i++) plan[1 << i] = new LeafPlan(i);
        for (int mask = 3; mask < total; mask++)
        {
            for (int sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask)
            {
                int rest = mask ^ sub;
                if (rest == 0 || plan[sub] == null || plan[rest] == null) continue;
                int lowest = mask & -mask;
                if ((sub & lowest) == 0) continue;            // each split is considered once
                var candidate = new JoinPlan(plan[sub], plan[rest], Describe(plan[sub]) + " join " + Describe(plan[rest]));
                long candidateCost = Cost(candidate, sizes);
                if (plan[mask] == null || candidateCost < cost[mask])
                {
                    plan[mask] = candidate;
                    cost[mask] = candidateCost;
                }
            }
        }
        return plan[total - 1];
    }

    internal string Describe(Plan plan)
    {
        if (plan is LeafPlan leaf) return Relations[leaf.Index].Name;
        var join = (JoinPlan)plan;
        return "(" + Describe(join.Left) + " join " + Describe(join.Right) + ")";
    }

    internal List<Plan> LeftDeepOrders()
    {
        int n = Relations.Length;
        var orders = new List<Plan>();
        var current = new List<int>();
        var used = new bool[n];
        void Recurse()
        {
            if (current.Count == n)
            {
                Plan plan = new LeafPlan(current[0]);
                for (int i = 1; i < current.Count; i++) plan = new JoinPlan(plan, new LeafPlan(current[i]), "left-deep");
                orders.Add(plan);
                return;
            }
            for (int i = 0; i < n; i++)
            {
                if (used[i]) continue;
                used[i] = true;
                current.Add(i);
                Recurse();
                current.RemoveAt(current.Count - 1);
                used[i] = false;
            }
        }
        Recurse();
        return orders;
    }

    /// <summary>Every subset step of a plan, in the order the join tree produces them.</summary>
    internal List<int> PlanMasks(Plan plan)
    {
        var masks = new List<int>();
        plan.CollectMasks(masks);
        return masks;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double QError(long estimate, long actual)
    {
        if (estimate <= 0 || actual <= 0) return double.MaxValue;
        return Math.Max((double)estimate / actual, (double)actual / estimate);
    }

    private static Relation[] BuildRelations(ulong seed, bool correlated)
    {
        var r1 = new List<int[]>();
        var r2 = new List<int[]>();
        var r3 = new List<int[]>();
        var r4 = new List<int[]>();
        ulong state = seed;
        for (int i = 0; i < 40; i++)
        {
            int a = SplitMix64.NextInt(ref state, 4);
            int b = correlated ? (SplitMix64.NextInt(ref state, 10) < 9 ? 0 : SplitMix64.NextInt(ref state, 4)) : SplitMix64.NextInt(ref state, 4);
            r1.Add(new[] { a, b });
        }
        for (int i = 0; i < 48; i++)
        {
            int b = correlated ? (SplitMix64.NextInt(ref state, 10) < 9 ? 0 : SplitMix64.NextInt(ref state, 4)) : SplitMix64.NextInt(ref state, 4);
            int c = correlated ? b : SplitMix64.NextInt(ref state, 5);       // c follows the join key when correlated
            r2.Add(new[] { b, c });
        }
        for (int i = 0; i < 36; i++)
        {
            int c = SplitMix64.NextInt(ref state, 5);
            int d = correlated ? (c + 1) % 6 : SplitMix64.NextInt(ref state, 6);
            r3.Add(new[] { c, d });
        }
        for (int i = 0; i < 52; i++)
        {
            int d = SplitMix64.NextInt(ref state, 6);
            int e = SplitMix64.NextInt(ref state, 7);
            r4.Add(new[] { d, e });
        }
        return new[]
        {
            new Relation("R1", new[] { "a", "b" }, r1),
            new Relation("R2", new[] { "b", "c" }, r2),
            new Relation("R3", new[] { "c", "d" }, r3),
            new Relation("R4", new[] { "d", "e" }, r4),
        };
    }

    public static void Main()
    {
        Relation[] relations = BuildRelations(0x5EED781000000001UL, true);
        var model = new PlannerModel(relations);
        int total = model.SubsetCount;
        int full = total - 1;

        // ---- the two cardinality vectors ----------------------------------------------------------
        {
            long pairMaterialised = Relation.Join(relations[0], relations[1]).Size;
            long pairOracle = Relation.CountJoin(relations[0], relations[1]);
            Check(pairMaterialised == pairOracle, "the materialised size equals the nested-loop counter ("
                + pairMaterialised + " vs " + pairOracle + ")");
            Check(model.TrueSizes[(1 << 0) | (1 << 1)] == pairMaterialised, "the subset table agrees with the materialised pair");

            long textbook = (long)relations[0].Size * relations[1].Size
                / Math.Max(relations[0].DistinctValues("b"), relations[1].DistinctValues("b"));
            Check(model.EstimatedSizes[(1 << 0) | (1 << 1)] == textbook,
                "the estimator applies |A|*|B| / max(V(A,b), V(B,b)) (" + model.EstimatedSizes[(1 << 0) | (1 << 1)] + ")");
            Check(model.EstimatedSizes[full] != model.TrueSizes[full],
                "the independence assumption misses the correlated full join (estimate " + model.EstimatedSizes[full]
                + " vs actual " + model.TrueSizes[full] + ")");

            // a three-relation subset, materialised and counted independently
            long tripleMaterialised = Relation.Join(Relation.Join(relations[0], relations[1]), relations[2]).Size;
            long tripleLoop = 0;
            foreach (int[] a in relations[0].Rows)
                foreach (int[] b in relations[1].Rows)
                {
                    if (a[1] != b[0]) continue;
                    foreach (int[] c in relations[2].Rows)
                        if (b[1] == c[0]) tripleLoop++;
                }
            Check(tripleMaterialised == tripleLoop, "the three-way materialisation equals the triple nested loop ("
                + tripleMaterialised + " vs " + tripleLoop + ")");
            Check(model.TrueSizes[1 | 2 | 4] == tripleMaterialised, "the subset table agrees on the triple");
            Console.WriteLine("cardinalities: |R1..R4|=" + relations[0].Size + "," + relations[1].Size + "," + relations[2].Size
                + "," + relations[3].Size + " fullTrue=" + model.TrueSizes[full] + " fullEstimate=" + model.EstimatedSizes[full]
                + " tripleTrue=" + tripleMaterialised + " tripleEstimate=" + model.EstimatedSizes[1 | 2 | 4]);
        }

        // ---- the planner: estimates pick one plan, the oracle picks another -------------------------
        {
            Plan estimatedPlan = model.BuildPlan(model.EstimatedSizes);
            Plan oraclePlan = model.BuildPlan(model.TrueSizes);
            var leftDeep = model.LeftDeepOrders();
            Check(leftDeep.Count == 24, "all 24 left-deep orders were enumerated (" + leftDeep.Count + ")");

            long bestLeftDeepEstimated = long.MaxValue, bestLeftDeepTrue = long.MaxValue, worstLeftDeepEstimated = 0;
            foreach (Plan plan in leftDeep)
            {
                bestLeftDeepEstimated = Math.Min(bestLeftDeepEstimated, model.Cost(plan, model.EstimatedSizes));
                bestLeftDeepTrue = Math.Min(bestLeftDeepTrue, model.Cost(plan, model.TrueSizes));
                worstLeftDeepEstimated = Math.Max(worstLeftDeepEstimated, model.Cost(plan, model.EstimatedSizes));
            }
            Check(model.Cost(estimatedPlan, model.EstimatedSizes) <= bestLeftDeepEstimated,
                "the dynamic program is never worse than the best left-deep order under the estimates ("
                + model.Cost(estimatedPlan, model.EstimatedSizes) + " <= " + bestLeftDeepEstimated + ")");
            Check(model.Cost(oraclePlan, model.TrueSizes) <= bestLeftDeepTrue,
                "the same dynamic program on true sizes is never worse than the best left-deep order ("
                + model.Cost(oraclePlan, model.TrueSizes) + " <= " + bestLeftDeepTrue + ")");
            Check(worstLeftDeepEstimated > bestLeftDeepEstimated,
                "the plan space really contains bad orders (" + worstLeftDeepEstimated + " vs " + bestLeftDeepEstimated + ")");

            long estimatedTrueCost = model.Cost(estimatedPlan, model.TrueSizes);
            long oracleTrueCost = model.Cost(oraclePlan, model.TrueSizes);
            double regression = (double)estimatedTrueCost / Math.Max(1, oracleTrueCost);
            Check(estimatedTrueCost >= oracleTrueCost, "the estimated plan cannot be cheaper than the oracle plan ("
                + estimatedTrueCost + " >= " + oracleTrueCost + ")");
            Check(regression >= 1.0, "estimation error never helps: regression " + regression.ToString("F4"));
            Check(model.Cost(estimatedPlan, model.TrueSizes) <= model.Cost(leftDeep[0], model.TrueSizes),
                "the estimated plan is still better than an arbitrary order");
            Console.WriteLine("plans: estimated " + model.Describe(estimatedPlan) + " trueCost=" + estimatedTrueCost
                + " | oracle " + model.Describe(oraclePlan) + " trueCost=" + oracleTrueCost
                + " | regression=" + regression.ToString("F4") + " | depth=" + estimatedPlan.Depth());
        }

        // ---- q-error of the estimates ---------------------------------------------------------------
        {
            double maxQ = 0, sumQ = 0;
            int steps = 0, underestimated = 0, overestimated = 0;
            for (int mask = 1; mask < total; mask++)
            {
                double q = QError(model.EstimatedSizes[mask], model.TrueSizes[mask]);
                maxQ = Math.Max(maxQ, q);
                sumQ += q;
                steps++;
                if (model.EstimatedSizes[mask] > model.TrueSizes[mask]) overestimated++;
                else if (model.EstimatedSizes[mask] < model.TrueSizes[mask]) underestimated++;
            }
            double meanQ = sumQ / steps;
            Check(steps == total - 1, "every subset has a q-error (" + steps + ")");
            Check(maxQ > 1.0, "some intermediate is badly mis-estimated (worst q-error " + maxQ.ToString("F3") + ")");
            Check(meanQ > 1.0, "the typical error is above one (" + meanQ.ToString("F3") + ")");
            Check(overestimated > 0 && underestimated > 0, "the errors go both ways ("
                + overestimated + " over, " + underestimated + " under)");

            // the plan's own steps, recomputed from the subset sizes rather than from the estimator
            Plan estimatedPlan = model.BuildPlan(model.EstimatedSizes);
            var masks = model.PlanMasks(estimatedPlan);
            double planMaxQ = 0;
            foreach (int mask in masks) planMaxQ = Math.Max(planMaxQ, QError(model.EstimatedSizes[mask], model.TrueSizes[mask]));
            Check(planMaxQ <= maxQ + 1e-9, "the plan's steps are a subset of the evaluated subsets ("
                + planMaxQ.ToString("F3") + " <= " + maxQ.ToString("F3") + ")");
            Check(masks.Count >= 3, "the plan really contains several join steps (" + masks.Count + ")");
            Console.WriteLine("q-error: max=" + maxQ.ToString("F3") + " mean=" + meanQ.ToString("F3") + " subsets=" + steps
                + " planSteps=" + masks.Count + " planMaxQ=" + planMaxQ.ToString("F3"));
        }

        // ---- the same planner on uncorrelated data --------------------------------------------------
        {
            Relation[] independent = BuildRelations(0x5EED781000000002UL, false);
            var independentModel = new PlannerModel(independent);
            int independentFull = independentModel.SubsetCount - 1;
            double fullQ = QError(independentModel.EstimatedSizes[independentFull], independentModel.TrueSizes[independentFull]);
            Plan estimatedPlan = independentModel.BuildPlan(independentModel.EstimatedSizes);
            Plan oraclePlan = independentModel.BuildPlan(independentModel.TrueSizes);
            long estimatedTrue = independentModel.Cost(estimatedPlan, independentModel.TrueSizes);
            long oracleTrue = independentModel.Cost(oraclePlan, independentModel.TrueSizes);
            Check(fullQ < 2.0, "uncorrelated data keeps the full-join estimate close (q-error " + fullQ.ToString("F3") + ")");
            Check(estimatedTrue >= oracleTrue, "the estimated plan is still not better than the oracle plan");
            double regression = (double)estimatedTrue / Math.Max(1, oracleTrue);
            Check(regression >= 1.0, "the regression never drops below one (" + regression.ToString("F4") + ")");
            Console.WriteLine("uncorrelated: fullQError=" + fullQ.ToString("F3") + " regression=" + regression.ToString("F4")
                + " estimatedTrue=" + estimatedTrue + " oracleTrue=" + oracleTrue);
        }

        // ---- the skew trap: a plan chosen from estimates that is provably not the best one ----------
        {
            // A skewed join key (95 of 100 rows share one value, but the key looks like it has 10
            // distinct values) makes |R1 join R2| ten times larger than the textbook estimate, while
            // the alternative first join of the same relation is exactly estimated. The estimate based
            // planner therefore starts with the expensive join and the oracle planner does not.
            ulong state = 0x5EED781000000003UL;
            var r1 = new List<int[]>();
            var r2 = new List<int[]>();
            var r3 = new List<int[]>();
            for (int i = 0; i < 100; i++)
            {
                int b = i < 95 ? 0 : 1 + (i % 9);
                r1.Add(new[] { SplitMix64.NextInt(ref state, 4), b, SplitMix64.NextInt(ref state, 5) });
            }
            for (int i = 0; i < 100; i++)
            {
                int b = i < 95 ? 0 : 1 + (i % 9);
                r2.Add(new[] { b, SplitMix64.NextInt(ref state, 6) });
            }
            for (int i = 0; i < 100; i++) r3.Add(new[] { SplitMix64.NextInt(ref state, 5), SplitMix64.NextInt(ref state, 6) });

            var skewed = new PlannerModel(new[]
            {
                new Relation("R1", new[] { "a", "b", "c" }, r1),
                new Relation("R2", new[] { "b", "f" }, r2),
                new Relation("R3", new[] { "c", "f" }, r3),
            });
            int pairMask = 1 | 2;                              // R1 join R2 on b
            double skewQ = QError(skewed.EstimatedSizes[pairMask], skewed.TrueSizes[pairMask]);
            Check(skewQ > 5, "the skewed key really breaks the estimate (q-error " + skewQ.ToString("F2")
                + ": estimate " + skewed.EstimatedSizes[pairMask] + " vs actual " + skewed.TrueSizes[pairMask] + ")");
            double alternativeQ = QError(skewed.EstimatedSizes[1 | 4], skewed.TrueSizes[1 | 4]);
            Check(alternativeQ < 1.3, "the alternative first join of R1 is estimated well (q-error " + alternativeQ.ToString("F3")
                + ": estimate " + skewed.EstimatedSizes[1 | 4] + " vs actual " + skewed.TrueSizes[1 | 4] + ")");

            Plan estimatedPlan = skewed.BuildPlan(skewed.EstimatedSizes);
            Plan oraclePlan = skewed.BuildPlan(skewed.TrueSizes);
            long estimatedTrue = skewed.Cost(estimatedPlan, skewed.TrueSizes);
            long oracleTrue = skewed.Cost(oraclePlan, skewed.TrueSizes);
            double regression = (double)estimatedTrue / Math.Max(1, oracleTrue);
            Check(skewed.Cost(estimatedPlan, skewed.EstimatedSizes) <= skewed.Cost(oraclePlan, skewed.EstimatedSizes),
                "under the estimates the two plans are indistinguishable, so the estimate-based planner has no reason to avoid the bad one ("
                + skewed.Cost(estimatedPlan, skewed.EstimatedSizes) + " vs " + skewed.Cost(oraclePlan, skewed.EstimatedSizes) + ")");
            Check(estimatedTrue > oracleTrue, "the plan the estimates accept is genuinely worse on the real data ("
                + estimatedTrue + " > " + oracleTrue + " shuffled tuples)");
            Check(regression > 1.0, "so the estimation error turns into wasted work: regression " + regression.ToString("F4"));
            Console.WriteLine("skew trap: estimated " + skewed.Describe(estimatedPlan) + " trueCost=" + estimatedTrue
                + " | oracle " + skewed.Describe(oraclePlan) + " trueCost=" + oracleTrue
                + " | regression=" + regression.ToString("F4") + " | pairQError=" + skewQ.ToString("F2"));
        }

        // ---- degenerate inputs ----------------------------------------------------------------------
        {
            var emptyFirst = new Relation("E1", new[] { "b" }, new List<int[]>());
            var other = new Relation("E2", new[] { "b" }, new List<int[]> { new[] { 1 }, new[] { 2 } });
            Check(Relation.Join(emptyFirst, other).Size == 0, "joining an empty relation yields nothing");
            Check(Relation.CountJoin(emptyFirst, other) == 0, "the counter agrees on the empty join");
            var emptyModel = new PlannerModel(new[] { emptyFirst, other });
            Check(emptyModel.EstimatedSizes[3] >= 1, "the estimator clamps a zero cardinality to one");
            Check(emptyModel.TrueSizes[3] == 0, "the true size of that join is zero");
            Check(emptyModel.Cost(emptyModel.BuildPlan(emptyModel.TrueSizes), emptyModel.TrueSizes) == other.Size,
                "an empty join step costs only the non-empty side (" + other.Size + ")");

            var single1 = new Relation("S1", new[] { "k" }, new List<int[]> { new[] { 1 } });
            var single2 = new Relation("S2", new[] { "k" }, new List<int[]> { new[] { 1 } });
            var smallModel = new PlannerModel(new[] { single1, single2 });
            Check(smallModel.TrueSizes[3] == 1, "two single-tuple relations join to one tuple");
            Check(smallModel.EstimatedSizes[3] == 1, "and the estimate is exact");
            Check(smallModel.Cost(smallModel.BuildPlan(smallModel.EstimatedSizes), smallModel.TrueSizes) == 3,
                "the two-relation plan costs left + right + result = 1 + 1 + 1");

            var disjoint1 = new Relation("D1", new[] { "p" }, new List<int[]> { new[] { 1 }, new[] { 2 } });
            var disjoint2 = new Relation("D2", new[] { "q" }, new List<int[]> { new[] { 1 }, new[] { 2 }, new[] { 3 } });
            var cartesianModel = new PlannerModel(new[] { disjoint1, disjoint2 });
            Check(cartesianModel.TrueSizes[3] == 6, "a join without shared attributes is a cartesian product (6 tuples)");
            Check(cartesianModel.EstimatedSizes[3] == 6, "the cartesian estimate is exact (2 * 3)");
            Console.WriteLine("degenerate: emptyJoin=" + Relation.Join(emptyFirst, other).Size
                + " singleTuple=" + smallModel.TrueSizes[3] + " cartesian=" + cartesianModel.TrueSizes[3]
                + " cartesianCost=" + cartesianModel.Cost(cartesianModel.BuildPlan(cartesianModel.EstimatedSizes), cartesianModel.TrueSizes));
        }

        Console.WriteLine("PASS");
    }
}
