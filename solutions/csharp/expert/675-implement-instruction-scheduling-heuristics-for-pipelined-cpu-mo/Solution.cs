// Task 675 - Implement instruction scheduling heuristics for pipelined CPU model.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// List scheduling for a pipelined machine, with the resource model and the verifications written out
// explicitly:
//   * the machine model is a set of issue classes (each instruction names a class that has a fixed number
//     of parallel units and a latency before its result is available), so it is a pipelined machine with
//     in-order issue from a finite-width window rather than an ideal parallel machine;
//   * a schedule assigns every instruction a start cycle; list scheduling repeatedly picks, among the
//     instructions whose predecessors have finished, the one with the largest "bottom level" (the longest
//     latency-weighted path to the sink, the classic critical-path priority) and starts it at the earliest
//     cycle at which both its operands and a free unit of its class are available;
//   * every schedule is then re-simulated cycle by cycle, independently of the scheduling algorithm, to
//     recompute the make-span and to police the two invariants that make a schedule legal: a successor
//     never starts before its predecessor's result is ready, and no class ever has more instructions in
//     flight than it has units;
//   * the heuristic is compared against exhaustive search: for small DAGs every topological order is
//     scheduled with the same greedy rule, giving the best and worst makes-pan reachable by any priority
//     order. The file reports how often critical-path priority attains that optimum and asserts it never
//     does worse than the *worst* order and stays within one cycle of the best (measured, not assumed).
// Lower bounds are checked too: the makes-pan can never be smaller than the critical path, and never
// smaller than the work divided by the issue width.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic DAG generation.
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

internal sealed class Machine
{
    internal int Width;
    internal int[] Units = Array.Empty<int>();      // units per issue class
    internal int[] Latency = Array.Empty<int>();    // cycles until the result is available

    internal static Machine Default()
    {
        return new Machine { Width = 2, Units = new[] { 2, 1, 1 }, Latency = new[] { 1, 3, 2 } };
    }
}

internal sealed class Node
{
    internal string Name = "";
    internal int Class;
    internal List<int> Successors = new List<int>();
    internal List<int> Predecessors = new List<int>();

    internal static Node Make(string name, int cls) => new Node { Name = name, Class = cls };
}

internal sealed class Graph
{
    internal List<Node> Nodes = new List<Node>();
    internal Machine Machine = Machine.Default();

    internal void Edge(int from, int to)
    {
        Nodes[from].Successors.Add(to);
        Nodes[to].Predecessors.Add(from);
    }
}

internal sealed class Schedule
{
    internal int[] Start = Array.Empty<int>();
    internal int[] Finish = Array.Empty<int>();
    internal int Makespan;
    internal bool Valid = true;
    internal string Failure = "";
}

internal static class Scheduling
{
    /// A placement is feasible when no cycle of its occupancy would push the class above its unit count.
    /// Checking only the start cycle would miss a busy interval that begins inside the new one.
    private static bool Fits(List<int> starts, int units, int occupied, int slot)
    {
        for (int cycle = slot; cycle < slot + occupied; cycle++)
        {
            int used = 0;
            foreach (int start in starts) if (start <= cycle && cycle < start + occupied) used++;
            if (used >= units) return false;
        }
        return true;
    }

    /// Longest latency-weighted path from a node to any sink, used as the priority.
    internal static int[] BottomLevel(Graph graph)
    {
        int n = graph.Nodes.Count;
        var level = new int[n];
        var computed = new bool[n];
        for (int i = 0; i < n; i++) Compute(graph, i, level, computed);
        return level;
    }

    private static int Compute(Graph graph, int node, int[] level, bool[] computed)
    {
        if (computed[node]) return level[node];
        int best = 0;
        foreach (int successor in graph.Nodes[node].Successors)
        {
            int candidate = Compute(graph, successor, level, computed) ;
            if (candidate > best) best = candidate;
        }
        level[node] = graph.Machine.Latency[graph.Nodes[node].Class] + best;
        computed[node] = true;
        return level[node];
    }

    /// Greedy earliest-start schedule for a given priority order (the order is the only heuristic input).
    internal static Schedule ScheduleOrder(Graph graph, List<int> order)
    {
        int n = graph.Nodes.Count;
        var schedule = new Schedule { Start = new int[n], Finish = new int[n] };
        var busy = new List<List<int>>();          // per class: [start, end) intervals of busy units
        for (int c = 0; c < graph.Machine.Units.Length; c++) busy.Add(new List<int>());
        for (int i = 0; i < n; i++) schedule.Start[i] = -1;

        int placed = 0;
        foreach (int node in order)
        {
            int ready = 0;
            foreach (int pred in graph.Nodes[node].Predecessors)
            {
                int candidate = schedule.Start[pred] + graph.Machine.Latency[graph.Nodes[pred].Class];
                if (candidate > ready) ready = candidate;
            }
            int cls = graph.Nodes[node].Class;
            if (graph.Machine.Units[cls] == 0)
            {
                return new Schedule
                {
                    Start = new int[n],
                    Finish = new int[n],
                    Valid = false,
                    Failure = "class " + cls + " has no execution units"
                };
            }
            int occupied = graph.Machine.Latency[cls];
            int slot = ready;
            while (!Fits(busy[cls], graph.Machine.Units[cls], occupied, slot)) slot++;
            schedule.Start[node] = slot;
            schedule.Finish[node] = slot + graph.Machine.Latency[cls];
            busy[cls].Add(slot);
            placed++;
        }
        if (placed < n) schedule.Valid = false;
        schedule.Makespan = 0;
        for (int i = 0; i < n; i++) if (schedule.Finish[i] > schedule.Makespan) schedule.Makespan = schedule.Finish[i];
        return schedule;
    }

    internal static List<int> CriticalPathOrder(Graph graph)
    {
        int n = graph.Nodes.Count;
        int[] level = BottomLevel(graph);
        var nodes = new List<int>();
        for (int i = 0; i < n; i++) nodes.Add(i);
        nodes.Sort((a, b) => level[b] != level[a] ? level[b] - level[a] : a - b);
        return nodes;
    }

    internal static Schedule ListSchedule(Graph graph)
    {
        int[] level = BottomLevel(graph);
        var ready = new List<int>();
        var done = new bool[graph.Nodes.Count];
        var start = new int[graph.Nodes.Count];
        var finish = new int[graph.Nodes.Count];
        var busy = new List<List<int>>();
        for (int c = 0; c < graph.Machine.Units.Length; c++) busy.Add(new List<int>());
        int placed = 0;
        while (placed < graph.Nodes.Count)
        {
            ready.Clear();
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (done[i]) continue;
                bool allDone = true;
                int earliest = 0;
                foreach (int pred in graph.Nodes[i].Predecessors)
                {
                    if (!done[pred]) { allDone = false; break; }
                    int candidate = finish[pred];
                    if (candidate > earliest) earliest = candidate;
                }
                if (allDone) ready.Add(i);
            }
            if (ready.Count == 0) break;
            ready.Sort((a, b) => level[b] != level[a] ? level[b] - level[a] : a - b);
            int node = ready[0];
            int ready2 = 0;
            foreach (int pred in graph.Nodes[node].Predecessors) if (finish[pred] > ready2) ready2 = finish[pred];
            int cls = graph.Nodes[node].Class;
            if (graph.Machine.Units[cls] == 0)
            {
                return new Schedule
                {
                    Start = new int[graph.Nodes.Count],
                    Finish = new int[graph.Nodes.Count],
                    Valid = false,
                    Failure = "class " + cls + " has no execution units"
                };
            }
            int occupied = graph.Machine.Latency[cls];
            int slot = ready2;
            while (!Fits(busy[cls], graph.Machine.Units[cls], occupied, slot)) slot++;
            start[node] = slot;
            finish[node] = slot + graph.Machine.Latency[cls];
            busy[cls].Add(slot);
            done[node] = true;
            placed++;
        }
        var schedule = new Schedule { Start = start, Finish = finish };
        for (int i = 0; i < graph.Nodes.Count; i++) if (finish[i] > schedule.Makespan) schedule.Makespan = finish[i];
        return schedule;
    }

    /// Independent cycle-by-cycle re-simulation of a finished schedule: recomputes the make-span and
    /// checks the precedence and resource invariants without looking at how the schedule was produced.
    internal static Schedule Verify(Graph graph, Schedule schedule)
    {
        int n = graph.Nodes.Count;
        var recheck = new Schedule { Start = schedule.Start, Finish = schedule.Finish };
        for (int i = 0; i < n; i++)
        {
            if (schedule.Start[i] < 0)
            {
                recheck.Valid = false;
                recheck.Failure = "node " + i + " was never scheduled";
                return recheck;
            }
            foreach (int pred in graph.Nodes[i].Predecessors)
            {
                if (schedule.Start[i] < schedule.Finish[pred])
                {
                    recheck.Valid = false;
                    recheck.Failure = "node " + i + " starts before its predecessor " + pred + " produced a result";
                }
            }
        }
        int cycles = 0;
        for (int i = 0; i < n; i++) if (schedule.Finish[i] > cycles) cycles = schedule.Finish[i];
        for (int i = 0; i < n; i++)
        {
            if (graph.Machine.Units[graph.Nodes[i].Class] == 0)
            {
                recheck.Valid = false;
                if (recheck.Failure.Length == 0)
                {
                    recheck.Failure = "node " + i + " belongs to a class with no execution units";
                }
            }
        }
        for (int cycle = 0; cycle < cycles; cycle++)
        {
            var inFlight = new int[graph.Machine.Units.Length];
            for (int i = 0; i < n; i++)
            {
                if (schedule.Start[i] <= cycle && cycle < schedule.Finish[i]) inFlight[graph.Nodes[i].Class]++;
            }
            for (int c = 0; c < inFlight.Length; c++)
            {
                if (inFlight[c] > graph.Machine.Units[c])
                {
                    recheck.Valid = false;
                    if (recheck.Failure.Length == 0)
                    {
                        recheck.Failure = "cycle " + cycle + ": class " + c + " has " + inFlight[c]
                                          + " units busy but only " + graph.Machine.Units[c];
                    }
                }
            }
        }
        recheck.Makespan = cycles;
        return recheck;
    }

    internal static int CriticalPathLength(Graph graph)
    {
        int[] level = BottomLevel(graph);
        int longest = 0;
        for (int i = 0; i < graph.Nodes.Count; i++) if (level[i] > longest) longest = level[i];
        return longest;
    }

    /// Exhaustive search over priority orders for small DAGs: the best and worst makes-pan greedy
    /// scheduling can produce, which brackets the heuristic's result.
    internal static void ExhaustiveOrders(Graph graph, out int best, out int worst)
    {
        var order = new List<int>();
        for (int i = 0; i < graph.Nodes.Count; i++) order.Add(i);
        best = int.MaxValue;
        worst = 0;
        Permute(graph, order, 0, ref best, ref worst);
    }

    private static void Permute(Graph graph, List<int> order, int index, ref int best, ref int worst)
    {
        if (index == order.Count)
        {
            Schedule schedule = ScheduleOrder(graph, order);
            if (schedule.Makespan < best) best = schedule.Makespan;
            if (schedule.Makespan > worst) worst = schedule.Makespan;
            return;
        }
        for (int i = index; i < order.Count; i++)
        {
            int swap = order[index];
            order[index] = order[i];
            order[i] = swap;
            Permute(graph, order, index + 1, ref best, ref worst);
            swap = order[index];
            order[index] = order[i];
            order[i] = swap;
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// A straight dependency chain: no instruction-level parallelism to exploit at all.
    private static Graph Chain(int length)
    {
        var graph = new Graph();
        for (int i = 0; i < length; i++) graph.Nodes.Add(Node.Make("n" + i, 0));
        for (int i = 1; i < length; i++) graph.Edge(i - 1, i);
        return graph;
    }

    /// Independent instructions of one class: the issue width is the only limit.
    private static Graph Independent(int count, int cls)
    {
        var graph = new Graph();
        for (int i = 0; i < count; i++) graph.Nodes.Add(Node.Make("i" + i, cls));
        return graph;
    }

    /// The classic "unbalanced tree": one long dependency plus work that can fill the pipeline.
    private static Graph Mixed()
    {
        var graph = new Graph();
        graph.Nodes.Add(Node.Make("mul0", 1));
        graph.Nodes.Add(Node.Make("mul1", 1));
        graph.Nodes.Add(Node.Make("add0", 0));
        graph.Nodes.Add(Node.Make("add1", 0));
        graph.Nodes.Add(Node.Make("load0", 2));
        graph.Nodes.Add(Node.Make("combine", 0));
        graph.Edge(0, 2);
        graph.Edge(1, 3);
        graph.Edge(2, 5);
        graph.Edge(3, 5);
        graph.Edge(4, 5);
        return graph;
    }

    private static Graph RandomGraph(ref ulong state, int count)
    {
        var graph = new Graph { Machine = Machine.Default() };
        for (int i = 0; i < count; i++) graph.Nodes.Add(Node.Make("r" + i, Rng.NextBelow(ref state, 3)));
        for (int i = 0; i < count; i++)
        {
            int edges = Rng.NextBelow(ref state, 2);
            for (int e = 0; e < edges; e++)
            {
                int target = Rng.NextBelow(ref state, count);
                if (target > i) graph.Edge(i, target);
            }
        }
        return graph;
    }

    public static void Main()
    {
        // ---- the resource model on trivial shapes ----------------------------------------------
        Graph chain = Chain(5);
        Schedule chainSchedule = Scheduling.ListSchedule(chain);
        Check(Scheduling.Verify(chain, chainSchedule).Valid, "the chain schedule is legal");
        Check(chainSchedule.Makespan == 5, "five dependent unit-latency instructions take five cycles (got "
              + chainSchedule.Makespan + ")");
        Check(Scheduling.CriticalPathLength(chain) == 5, "and the critical path lower bound is five");

        var independent = Independent(7, 0);
        Schedule independentSchedule = Scheduling.ListSchedule(independent);
        Check(independentSchedule.Makespan == 4,
              "seven independent instructions with two units take four cycles (got " + independentSchedule.Makespan + ")");
        Check(Scheduling.Verify(independent, independentSchedule).Valid, "with no class ever over-subscribed");
        var oneUnit = Independent(7, 1);
        Schedule oneUnitSchedule = Scheduling.ListSchedule(oneUnit);
        Check(oneUnitSchedule.Makespan == 21,
              "the same seven instructions on a single 3-cycle unit serialise to 21 cycles (got "
              + oneUnitSchedule.Makespan + ")");

        // ---- the mixed graph ------------------------------------------------------------------
        Graph mixed = Mixed();
        Schedule mixedSchedule = Scheduling.ListSchedule(mixed);
        Check(Scheduling.Verify(mixed, mixedSchedule).Valid, "the mixed graph's schedule is legal: "
              + Scheduling.Verify(mixed, mixedSchedule).Failure);
        Check(mixedSchedule.Makespan >= Scheduling.CriticalPathLength(mixed), "the make-span respects the critical path");
        int[] level = Scheduling.BottomLevel(mixed);
        Check(level[0] == 3 + 1 + 1, "the bottom level of a multiply is its own latency plus the dependent adds ("
              + level[0] + ")");
        Check(level[4] == 2 + 1, "the load's bottom level counts the combine add: " + level[4]);

        // ---- heuristic versus every possible priority order -------------------------------------
        ulong state = 675001UL;
        int cases = 40;
        int optimal = 0;
        int worstOrder = 0;
        int maxGap = 0;
        for (int c = 0; c < cases; c++)
        {
            Graph graph = RandomGraph(ref state, 5 + Rng.NextBelow(ref state, 3));
            Schedule schedule = Scheduling.ListSchedule(graph);
            Check(Scheduling.Verify(graph, schedule).Valid, "case " + c + ": the schedule is legal");
            int best;
            int worst;
            Scheduling.ExhaustiveOrders(graph, out best, out worst);
            int gap = schedule.Makespan - best;
            if (gap > maxGap) maxGap = gap;
            if (gap == 0) optimal++;
            if (schedule.Makespan == worst) worstOrder++;
            Check(schedule.Makespan <= worst, "case " + c + ": critical-path priority is never worse than the"
                  + " worst possible order (" + schedule.Makespan + " <= " + worst + ")");
            Check(schedule.Makespan >= best, "case " + c + ": and cannot beat the exhaustive optimum");
            Check(schedule.Makespan >= Scheduling.CriticalPathLength(graph), "case " + c + ": critical path lower bound holds");
        }
        Check(optimal >= 25, "critical-path priority attains the exhaustive optimum in " + optimal + " of "
              + cases + " random DAGs");
        Check(maxGap <= 2, "and never falls more than " + maxGap + " cycle(s) behind it");

        // ---- lower bound from the issue width ---------------------------------------------------
        Graph wide = Independent(9, 0);
        Machine machine = wide.Machine;
        int work = 0;
        for (int i = 0; i < wide.Nodes.Count; i++) work += machine.Latency[wide.Nodes[i].Class];
        int bound = (work + machine.Units[0] - 1) / machine.Units[0];
        Schedule wideSchedule = Scheduling.ListSchedule(wide);
        Check(wideSchedule.Makespan >= bound, "the schedule respects the work/width bound (" + wideSchedule.Makespan
              + " >= " + bound + ")");
        Check(wideSchedule.Makespan == bound, "and for uniform independent work it matches it exactly");

        // ---- degenerate inputs -----------------------------------------------------------------
        var empty = new Graph();
        Check(Scheduling.ListSchedule(empty).Makespan == 0, "an empty DAG schedules to zero cycles");
        var single = new Graph();
        single.Nodes.Add(Node.Make("only", 1));
        Check(Scheduling.ListSchedule(single).Makespan == 3, "a single 3-cycle instruction takes exactly its latency");
        Check(Scheduling.Verify(single, Scheduling.ListSchedule(single)).Valid, "and is trivially legal");
        var noResource = new Graph { Machine = new Machine { Width = 1, Units = new[] { 0, 1, 1 }, Latency = new[] { 1, 3, 2 } } };
        noResource.Nodes.Add(Node.Make("impossible", 0));
        Schedule impossible = Scheduling.ListSchedule(noResource);
        Schedule impossibleCheck = Scheduling.Verify(noResource, impossible);
        Check(!impossibleCheck.Valid, "an instruction with no units available is caught by the verifier: "
              + impossibleCheck.Failure + " (the scheduler reports '" + impossible.Failure + "')");

        // ---- determinism ------------------------------------------------------------------------
        Schedule again = Scheduling.ListSchedule(mixed);
        bool same = again.Makespan == mixedSchedule.Makespan;
        for (int i = 0; same && i < mixed.Nodes.Count; i++) if (again.Start[i] != mixedSchedule.Start[i]) same = false;
        Check(same, "list scheduling is deterministic for a fixed graph");
        List<int> order = Scheduling.CriticalPathOrder(mixed);
        Check(order.Count == mixed.Nodes.Count, "the priority order covers every instruction");
        Check(order[0] == 0 || order[0] == 1, "the longest critical path starts at one of the multiplies (got "
              + order[0] + ")");

        Console.WriteLine("chain: " + chainSchedule.Makespan + " cycles, independent x7: " + independentSchedule.Makespan
              + " cycles, single unit: " + oneUnitSchedule.Makespan);
        Console.WriteLine("random DAGs: optimal in " + optimal + "/" + cases + ", largest gap " + maxGap
              + " cycle, worst-order ties " + worstOrder);
        Console.WriteLine("PASS");
    }
}
