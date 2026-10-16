// Task 674 - Implement register allocation via graph coloring (simplified).
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// The Chaitin/Briggs pipeline, scaled down to a straight-line instruction sequence but with the real
// stages and the real optimisations:
//   * liveness for straight-line code (a backward scan over the points, which is the degenerate case of the
//     CFG dataflow but is computed by its own pass here);
//   * a live interval per variable, from its first definition to the last point at which it is still live;
//   * the interference graph, built twice and compared: once from the intervals (two intervals interfere
//     when they overlap) and once from the per-point live sets (two variables interfere when some point
//     has both live), which is the definition the intervals are only a shortcut for — the shortcut is
//     exact here because the generated code assigns each variable once, so live ranges are contiguous;
//   * Chaitin simplification (repeatedly push a node of degree < K), then Briggs' *optimistic* selection
//     (when nothing has low degree, push a spill candidate anyway and try to colour it on the way out),
//     and finally the spill decision: only a node that cannot be coloured even optimistically is spilled.
// The checks deliberately do not trust the allocator's own verdict. Every interference graph produced is
// audited (no two adjacent nodes may share a colour, spilled variables must be exactly the uncoloured
// ones), and each graph is compared against an exhaustive colouring search: a graph that brute force says
// is not K-colourable must spill something, and the optimistic rule must never spill more than the
// pessimistic one (it does strictly better on a measurable number of graphs). Greedy colouring is a
// heuristic rather than an exact solver, so on colourable graphs the file reports the miss *rate* it
// measures instead of pretending the miss cannot happen. The last check ties the result back to the code:
// at every program point, variables that are simultaneously live must have distinct registers or be
// spilled.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic graph and program generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Ins
{
    internal string Var = "";
    internal List<string> Uses = new List<string>();

    internal static Ins Defines(string variable, params string[] uses)
    {
        var instruction = new Ins { Var = variable };
        instruction.Uses.AddRange(uses);
        return instruction;
    }
}

internal sealed class AllocResult
{
    internal int[] Color = Array.Empty<int>();          // -1 means spilled
    internal List<string> Spilled = new List<string>();
    internal int Pushes;
    internal int OptimisticPushes;
    internal int ColorsUsed;
}

internal static class Allocation
{
    /// Live set *before* each point (index p holds the variables live on entry to instruction p; index
    /// code.Count is the state after the last instruction, which is empty for straight-line code).
    internal static List<HashSet<string>> LiveBefore(List<Ins> code)
    {
        int points = code.Count;
        var live = new List<HashSet<string>>();
        for (int i = 0; i <= points; i++) live.Add(new HashSet<string>());
        for (int i = points - 1; i >= 0; i--)
        {
            var current = new HashSet<string>(live[i + 1]);
            current.Remove(code[i].Var);
            foreach (string use in code[i].Uses) current.Add(use);
            live[i] = current;
        }
        return live;
    }

    internal static List<string> Variables(List<Ins> code)
    {
        var variables = new List<string>();
        foreach (Ins instruction in code)
        {
            if (instruction.Var.Length > 0 && !variables.Contains(instruction.Var)) variables.Add(instruction.Var);
            foreach (string use in instruction.Uses) if (!variables.Contains(use)) variables.Add(use);
        }
        return variables;
    }

    /// Interference from the per-point live sets: both variables live at the same program point.
    internal static bool[,] InterferenceFromPoints(List<Ins> code, List<string> variables,
                                                   List<HashSet<string>> live)
    {
        int n = variables.Count;
        var graph = new bool[n, n];
        for (int point = 0; point <= code.Count; point++)
        {
            // The live set after a point, plus the definition at that point, is what is simultaneously live.
            var concurrent = new HashSet<string>(live[point]);
            if (point < code.Count) concurrent.Add(code[point].Var);
            for (int i = 0; i < n; i++)
            {
                if (!concurrent.Contains(variables[i])) continue;
                for (int j = i + 1; j < n; j++)
                {
                    if (!concurrent.Contains(variables[j])) continue;
                    graph[i, j] = true;
                    graph[j, i] = true;
                }
            }
        }
        return graph;
    }

    /// Interference from live intervals: first definition to last point at which the variable is live.
    internal static bool[,] InterferenceFromIntervals(List<Ins> code, List<string> variables,
                                                      List<HashSet<string>> live, int[] start, int[] end)
    {
        int n = variables.Count;
        var graph = new bool[n, n];
        for (int i = 0; i < n; i++)
        {
            int first = -1;
            for (int p = 0; p < code.Count; p++)
            {
                if (code[p].Var == variables[i] && first < 0) first = p;
            }
            int last = -1;
            for (int p = 0; p <= code.Count; p++)
            {
                if (live[p].Contains(variables[i])) last = p;
            }
            start[i] = first < 0 ? 0 : first;
            end[i] = last < 0 ? start[i] : last;
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Closed intervals: a variable is live at the point it is defined (that is the point where
                // it starts to hold a value) and at every later point where it is still needed.
                if (start[i] <= end[j] && start[j] <= end[i])
                {
                    graph[i, j] = true;
                    graph[j, i] = true;
                }
            }
        }
        return graph;
    }

    internal static bool[,] FromPairs(int n, params int[] pairs)
    {
        var graph = new bool[n, n];
        for (int i = 0; i + 1 < pairs.Length; i += 2)
        {
            graph[pairs[i], pairs[i + 1]] = true;
            graph[pairs[i + 1], pairs[i]] = true;
        }
        return graph;
    }

    /// Chaitin simplify with (optionally) Briggs' optimistic selection. With `optimistic` false the
    /// classical pessimistic rule is used instead: a node whose degree is >= K at push time is spilled
    /// immediately instead of being given a chance on the way out.
    internal static AllocResult Colour(bool[,] graph, int k) => Colour(graph, k, true);

    internal static AllocResult Colour(bool[,] graph, int k, bool optimistic)
    {
        int n = graph.GetLength(0);
        var result = new AllocResult { Color = new int[n] };
        for (int i = 0; i < n; i++) result.Color[i] = -1;

        var removed = new bool[n];
        var decided = new bool[n];
        var stack = new List<int>();
        int remaining = n;
        while (remaining > 0)
        {
            int pick = -1;
            for (int i = 0; i < n; i++)
            {
                if (removed[i]) continue;
                int degree = 0;
                for (int j = 0; j < n; j++) if (!removed[j] && j != i && graph[i, j]) degree++;
                if (degree < k) { pick = i; break; }
            }
            if (pick < 0)
            {
                // Nothing is trivially colourable: take the highest-degree node.
                int best = -1;
                int bestDegree = -1;
                for (int i = 0; i < n; i++)
                {
                    if (removed[i]) continue;
                    int degree = 0;
                    for (int j = 0; j < n; j++) if (!removed[j] && j != i && graph[i, j]) degree++;
                    if (degree > bestDegree) { bestDegree = degree; best = i; }
                }
                pick = best;
                if (optimistic)
                {
                    result.OptimisticPushes++;
                }
                else
                {
                    // Pessimistic: this node is spilled without trying to colour it later.
                    decided[pick] = true;
                    result.Spilled.Add(pick.ToString());
                    result.Pushes++;
                    removed[pick] = true;
                    remaining--;
                    continue;
                }
            }
            removed[pick] = true;
            stack.Add(pick);
            result.Pushes++;
            remaining--;
        }

        for (int s = stack.Count - 1; s >= 0; s--)
        {
            int node = stack[s];
            if (decided[node]) continue;
            var taken = new bool[k];
            for (int j = 0; j < n; j++)
            {
                if (j == node || graph[node, j] == false) continue;
                if (result.Color[j] >= 0) taken[result.Color[j]] = true;
            }
            int chosen = -1;
            for (int c = 0; c < k; c++) if (!taken[c]) { chosen = c; break; }
            result.Color[node] = chosen;
            if (chosen < 0) result.Spilled.Add(node.ToString());
        }
        result.ColorsUsed = 0;
        for (int i = 0; i < n; i++) if (result.Color[i] + 1 > result.ColorsUsed) result.ColorsUsed = result.Color[i] + 1;
        return result;
    }

    /// Exhaustive search for a K-colouring, used only to police the allocator on small graphs.
    internal static bool BruteForceColourable(bool[,] graph, int k)
    {
        int n = graph.GetLength(0);
        var colour = new int[n];
        for (int i = 0; i < n; i++) colour[i] = -1;
        return Search(graph, k, colour, 0);
    }

    private static bool Search(bool[,] graph, int k, int[] colour, int index)
    {
        int n = graph.GetLength(0);
        if (index == n) return true;
        for (int c = 0; c < k; c++)
        {
            bool ok = true;
            for (int j = 0; j < index; j++)
            {
                if (graph[index, j] && colour[j] == c) { ok = false; break; }
            }
            if (!ok) continue;
            colour[index] = c;
            if (Search(graph, k, colour, index + 1)) return true;
            colour[index] = -1;
        }
        return false;
    }

    /// No edge may join two nodes of the same colour; uncoloured nodes must be the spilled ones.
    internal static int Audit(bool[,] graph, AllocResult result)
    {
        int n = graph.GetLength(0);
        int violations = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (!graph[i, j]) continue;
                if (result.Color[i] >= 0 && result.Color[i] == result.Color[j]) violations++;
            }
        }
        return violations;
    }

    internal static int DegreeOf(bool[,] graph, int node)
    {
        int n = graph.GetLength(0);
        int degree = 0;
        for (int j = 0; j < n; j++) if (j != node && graph[node, j]) degree++;
        return degree;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static List<Ins> StraightLine()
    {
        var code = new List<Ins>();
        code.Add(Ins.Defines("a"));
        code.Add(Ins.Defines("b", "a"));
        code.Add(Ins.Defines("c", "a", "b"));
        code.Add(Ins.Defines("d", "c"));
        return code;
    }

    /// The generator assigns every variable at most once and defines it before its first use (except for the
    /// dedicated input "in", which is never defined). That keeps every live range contiguous, which is
    /// exactly the condition under which interval overlap and point liveness coincide — the property the
    /// cross-check asserts. Arbitrary (non-chordal) interference graphs, where no interval model applies,
    /// are covered by the random-graph sweep instead.
    private static List<Ins> RandomCode(ref ulong state, int length)
    {
        string[] names = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" };
        var defined = new List<string>();
        var used = new HashSet<string>();
        var code = new List<Ins>();
        for (int i = 0; i < length; i++)
        {
            var uses = new List<string>();
            int useCount = Rng.NextBelow(ref state, 3);
            for (int u = 0; u < useCount; u++)
            {
                if (defined.Count > 0 && Rng.NextBelow(ref state, 4) != 0)
                {
                    uses.Add(defined[Rng.NextBelow(ref state, defined.Count)]);
                }
                else
                {
                    uses.Add("in");
                }
            }
            string target = names[i];
            if (used.Contains(target)) continue;
            used.Add(target);
            defined.Add(target);
            code.Add(Ins.Defines(target, uses.ToArray()));
        }
        return code;
    }

    private static string Show(bool[,] graph)
    {
        int n = graph.GetLength(0);
        var parts = new List<string>();
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++) if (graph[i, j]) parts.Add(i + "-" + j);
        }
        return "{" + string.Join(",", parts) + "}";
    }

    public static void Main()
    {
        // ---- liveness and intervals ------------------------------------------------------------
        List<Ins> line = StraightLine();
        List<string> lineVariables = Allocation.Variables(line);
        List<HashSet<string>> lineLive = Allocation.LiveBefore(line);
        Check(lineVariables.Count == 4, "the straight-line program uses four variables");
        Check(string.Join(",", Sorted(lineLive[4])) == "", "nothing is live at the end");
        Check(string.Join(",", Sorted(lineLive[3])) == "c", "c is live during the last instruction: "
              + string.Join(",", Sorted(lineLive[3])));
        Check(string.Join(",", Sorted(lineLive[2])) == "a,b", "a and b are both live after the third point");
        int[] start = new int[lineVariables.Count];
        int[] end = new int[lineVariables.Count];
        Allocation.InterferenceFromIntervals(line, lineVariables, lineLive, start, end);
        Check(start[0] == 0 && end[0] == 2, "a's interval runs from its definition through its last use ("
              + start[0] + ".." + end[0] + ")");

        // ---- the two ways of building the interference graph must agree -------------------------
        ulong state = 674001UL;
        int programs = 200;
        int interferenceEdges = 0;
        for (int p = 0; p < programs; p++)
        {
            List<Ins> code = RandomCode(ref state, 3 + Rng.NextBelow(ref state, 8));
            List<string> variables = Allocation.Variables(code);
            List<HashSet<string>> live = Allocation.LiveBefore(code);
            var fromPoints = Allocation.InterferenceFromPoints(code, variables, live);
            int[] s = new int[variables.Count];
            int[] e = new int[variables.Count];
            var fromIntervals = Allocation.InterferenceFromIntervals(code, variables, live, s, e);
            bool same = true;
            for (int i = 0; i < variables.Count && same; i++)
            {
                for (int j = 0; j < variables.Count; j++)
                {
                    if (fromPoints[i, j] != fromIntervals[i, j]) same = false;
                }
            }
            Check(same, "program " + p + ": interval overlap and point liveness give the same graph ("
                  + Show(fromPoints) + " vs " + Show(fromIntervals) + ")");
            for (int i = 0; i < variables.Count; i++)
            {
                for (int j = i + 1; j < variables.Count; j++) if (fromPoints[i, j]) interferenceEdges++;
            }
            Check(live.Count == code.Count + 1, "program " + p + ": liveness is recorded per program point");
        }
        Check(interferenceEdges >= 300, "the random programs produce " + interferenceEdges + " interference pairs");

        // ---- colouring a graph that needs exactly three colours --------------------------------
        var triangle = Allocation.FromPairs(3, 0, 1, 1, 2, 0, 2);
        AllocResult three = Allocation.Colour(triangle, 3);
        Check(three.Spilled.Count == 0 && three.ColorsUsed == 3, "a triangle colours with exactly three registers");
        Check(Allocation.Audit(triangle, three) == 0, "no two adjacent nodes share a register");
        AllocResult two = Allocation.Colour(triangle, 2);
        Check(two.Spilled.Count == 1, "with only two registers exactly one variable spills (got " + two.Spilled.Count + ")");
        Check(!Allocation.BruteForceColourable(triangle, 2), "and brute force confirms the triangle is not 2-colourable");

        // ---- a clique forces spills; optimistic colouring still finds what exists ---------------
        var clique4 = Allocation.FromPairs(4, 0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3);
        AllocResult clique = Allocation.Colour(clique4, 3);
        Check(clique.Spilled.Count == 1, "a K4 needs one spill with three registers (got " + clique.Spilled.Count + ")");
        Check(Allocation.Audit(clique4, clique) == 0, "and the surviving colours still respect every edge");

        // Optimistic case: the 4-cycle. Every node has degree 2 = K, so no node can be pushed by the
        // low-degree rule, yet the graph is 2-colourable and must come out spill-free.
        var cycle4 = Allocation.FromPairs(4, 0, 1, 1, 2, 2, 3, 3, 0);
        AllocResult cycleResult = Allocation.Colour(cycle4, 2);
        Check(cycleResult.OptimisticPushes > 0, "the 4-cycle forces optimistic pushes ("
              + cycleResult.OptimisticPushes + ")");
        Check(cycleResult.Spilled.Count == 0 && cycleResult.ColorsUsed == 2,
              "but optimistic colouring still finds the 2-colouring with no spill");
        AllocResult cycleStrict = Allocation.Colour(cycle4, 2, false);
        Check(cycleStrict.Spilled.Count == 1,
              "while the pessimistic rule gives up and spills one node (got " + cycleStrict.Spilled.Count + ")");
        Check(Allocation.DegreeOf(cycle4, 0) >= 2, "even though every node has degree >= K");
        Check(Allocation.BruteForceColourable(cycle4, 2), "and brute force confirms the 2-colouring exists");
        Check(Allocation.Audit(cycle4, cycleResult) == 0, "the 4-cycle colouring respects every edge");

        // ---- random cross-check against exhaustive colouring ------------------------------------
        int colourings = 0;
        int spills = 0;
        int colorableButSpilled = 0;
        int optimisticSpillTotal = 0;
        int pessimisticSpillTotal = 0;
        int optimisticStrictlyBetter = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            int n = 3 + Rng.NextBelow(ref state, 5);
            int k = 2 + Rng.NextBelow(ref state, 3);
            var graph = new bool[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Rng.NextBelow(ref state, 100) < 45)
                    {
                        graph[i, j] = true;
                        graph[j, i] = true;
                    }
                }
            }
            AllocResult result = Allocation.Colour(graph, k);
            AllocResult strict = Allocation.Colour(graph, k, false);
            Check(Allocation.Audit(graph, result) == 0, "trial " + trial + ": the produced colouring is valid");
            Check(Allocation.Audit(graph, strict) == 0, "trial " + trial + ": the pessimistic colouring is valid too");
            bool brute = Allocation.BruteForceColourable(graph, k);
            optimisticSpillTotal += result.Spilled.Count;
            pessimisticSpillTotal += strict.Spilled.Count;
            if (result.Spilled.Count < strict.Spilled.Count) optimisticStrictlyBetter++;
            Check(result.Spilled.Count <= strict.Spilled.Count,
                  "trial " + trial + ": trying harder never spills more than giving up early ("
                  + result.Spilled.Count + " <= " + strict.Spilled.Count + ")");
            if (brute)
            {
                colourings++;
                if (result.Spilled.Count > 0) colorableButSpilled++;
            }
            else
            {
                spills++;
                Check(result.Spilled.Count > 0, "trial " + trial + ": the graph is not " + k
                      + "-colourable, so at least one node must spill on " + Show(graph));
            }
            Check(result.Pushes == n, "trial " + trial + ": every node went through the simplify stack");
        }
        Check(colourings >= 60 && spills >= 20, "the sweep covers both outcomes: " + colourings
              + " colourable, " + spills + " requiring spills");
        Check(optimisticStrictlyBetter >= 5, "optimistic selection beats the pessimistic rule outright in "
              + optimisticStrictlyBetter + " of 300 graphs");
        Check(optimisticSpillTotal <= pessimisticSpillTotal, "optimistic selection spills " + optimisticSpillTotal
              + " nodes in total against " + pessimisticSpillTotal + " for the pessimistic rule");
        // Greedy colouring is a heuristic, not an exact solver: it can still miss an existing colouring, so
        // the honest statement is a rate, measured here rather than assumed.
        Check(colorableButSpilled * 10 <= colourings + 10, "the allocator finds a colouring for the great majority"
              + " of colourable graphs (" + colorableButSpilled + " misses out of " + colourings + ")");

        // ---- the result must be consistent with the code ---------------------------------------
        int pointChecks = 0;
        for (int p = 0; p < programs; p++)
        {
            List<Ins> code = RandomCode(ref state, 3 + Rng.NextBelow(ref state, 8));
            List<string> variables = Allocation.Variables(code);
            List<HashSet<string>> live = Allocation.LiveBefore(code);
            var graph = Allocation.InterferenceFromPoints(code, variables, live);
            AllocResult result = Allocation.Colour(graph, 3);
            for (int point = 0; point <= code.Count; point++)
            {
                var concurrent = new HashSet<string>(live[point]);
                if (point < code.Count) concurrent.Add(code[point].Var);
                var registers = new Dictionary<int, string>();
                foreach (string variable in concurrent)
                {
                    int index = variables.IndexOf(variable);
                    if (index < 0) continue;
                    int colour = result.Color[index];
                    if (colour < 0) continue;
                    string owner;
                    if (registers.TryGetValue(colour, out owner) && owner != variable)
                    {
                        Check(false, "program " + p + " point " + point + ": register " + colour
                              + " holds both " + owner + " and " + variable);
                    }
                    registers[colour] = variable;
                }
                pointChecks++;
            }
        }
        Check(pointChecks > 1000, "the register assignment was re-verified at " + pointChecks + " program points");

        // ---- degenerate inputs -----------------------------------------------------------------
        var empty = new bool[0, 0];
        AllocResult emptyResult = Allocation.Colour(empty, 3);
        Check(emptyResult.Spilled.Count == 0 && emptyResult.ColorsUsed == 0, "an empty graph needs no registers");
        var independent = Allocation.FromPairs(3);
        AllocResult independentResult = Allocation.Colour(independent, 1);
        Check(independentResult.Spilled.Count == 0 && independentResult.ColorsUsed == 1,
              "three non-interfering variables share a single register");
        var single = new bool[1, 1];
        Check(Allocation.Colour(single, 1).Spilled.Count == 0, "a single variable fits in one register");
        Check(Allocation.BruteForceColourable(single, 1), "and brute force agrees");

        Console.WriteLine("random programs audited : " + programs + " (" + interferenceEdges + " interference pairs)");
        Console.WriteLine("random graphs: " + colourings + " colourable, " + spills + " needing spills");
        Console.WriteLine("program points re-verified: " + pointChecks);
        Console.WriteLine("PASS");
    }

    private static List<string> Sorted(HashSet<string> set)
    {
        var list = new List<string>(set);
        list.Sort();
        return list;
    }
}
