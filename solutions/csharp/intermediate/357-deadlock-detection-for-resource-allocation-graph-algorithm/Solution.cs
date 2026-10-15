// Task 357 - Deadlock detection for resource allocation graph (algorithm). (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private const int ScriptTrials = 300;

    private static List<int> CanonicalCycle(List<int> cycle)
    {
        int minIndex = 0;
        for (int i = 1; i < cycle.Count; i++) if (cycle[i] < cycle[minIndex]) minIndex = i;
        var result = new List<int>();
        for (int i = 0; i < cycle.Count; i++) result.Add(cycle[(minIndex + i) % cycle.Count]);
        return result;
    }

    /// <summary>
    /// Collapse the resource allocation graph into the wait-for graph: a request (p, r)
    /// with r held by q becomes the edge p -> q; a request for a free resource adds nothing.
    /// </summary>
    private static List<List<int>> BuildWaitFor(int processes, int resources, int[][] holds, int[][] requests)
    {
        var holder = new int[resources];
        for (int i = 0; i < resources; i++) holder[i] = -1;
        foreach (var pair in holds)
        {
            if (holder[pair[1]] != -1 && holder[pair[1]] != pair[0])
            {
                throw new ArgumentException("resource held by more than one process: " + pair[1]);
            }
            holder[pair[1]] = pair[0];
        }
        var adjacency = new List<List<int>>();
        for (int i = 0; i < processes; i++) adjacency.Add(new List<int>());
        foreach (var pair in requests)
        {
            int currentHolder = holder[pair[1]];
            if (currentHolder != -1) adjacency[pair[0]].Add(currentHolder);
        }
        return adjacency;
    }

    /// <summary>Returns a canonical cycle or null. 0 = white, 1 = gray (on the DFS path), 2 = black.</summary>
    private static List<int>? FindCycle(int processes, List<List<int>> adjacency)
    {
        var color = new int[processes];
        var path = new List<int>();
        for (int start = 0; start < processes; start++)
        {
            if (color[start] == 0)
            {
                var cycle = DfsCycle(start, adjacency, color, path);
                if (cycle != null) return CanonicalCycle(cycle);
            }
        }
        return null;
    }

    private static List<int>? DfsCycle(int node, List<List<int>> adjacency, int[] color, List<int> path)
    {
        color[node] = 1;
        path.Add(node);
        foreach (int next in adjacency[node])
        {
            if (color[next] == 1)                          // back edge: the path slice is a cycle
            {
                int start = path.IndexOf(next);
                return path.GetRange(start, path.Count - start);
            }
            if (color[next] == 0)
            {
                var cycle = DfsCycle(next, adjacency, color, path);
                if (cycle != null) return cycle;
            }
        }
        path.RemoveAt(path.Count - 1);
        color[node] = 2;
        return null;
    }

    /// <summary>Independent check: a cycle exists iff some Kosaraju SCC has size > 1 or a self-loop exists.</summary>
    private static bool HasCycleViaScc(int processes, List<List<int>> adjacency)
    {
        var reverse = new List<List<int>>();
        for (int i = 0; i < processes; i++) reverse.Add(new List<int>());
        for (int u = 0; u < processes; u++)
        {
            foreach (int v in adjacency[u]) reverse[v].Add(u);
        }

        var visited = new bool[processes];
        var order = new List<int>();
        for (int u = 0; u < processes; u++) if (!visited[u]) DfsFinish(u, adjacency, visited, order);

        var component = new int[processes];
        for (int i = 0; i < processes; i++) component[i] = -1;
        var sizes = new List<int>();
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int u = order[i];
            if (component[u] != -1) continue;
            int groupIndex = sizes.Count;
            component[u] = groupIndex;
            int size = 0;
            var stack = new Stack<int>();
            stack.Push(u);
            while (stack.Count > 0)
            {
                int x = stack.Pop();
                size++;
                foreach (int y in reverse[x])
                {
                    if (component[y] == -1)
                    {
                        component[y] = groupIndex;
                        stack.Push(y);
                    }
                }
            }
            sizes.Add(size);
        }

        if (sizes.Any(size => size > 1)) return true;
        for (int u = 0; u < processes; u++) if (adjacency[u].Contains(u)) return true;
        return false;
    }

    private static void DfsFinish(int u, List<List<int>> adjacency, bool[] visited, List<int> order)
    {
        visited[u] = true;
        foreach (int v in adjacency[u]) if (!visited[v]) DfsFinish(v, adjacency, visited, order);
        order.Add(u);
    }

    private static void AssertValidCycle(List<List<int>> adjacency, List<int> cycle)
    {
        Check(ListsEqual(CanonicalCycle(cycle), cycle), "cycle returned canonicalised");
        Check(cycle.Distinct().Count() == cycle.Count, "cycle vertices are distinct");
        for (int i = 0; i < cycle.Count; i++)
        {
            int next = cycle[(i + 1) % cycle.Count];
            Check(adjacency[cycle[i]].Contains(next), "cycle edges exist");
        }
    }

    private class Rag
    {
        public int[][] Holds = Array.Empty<int[]>();
        public int[][] Requests = Array.Empty<int[]>();
    }

    private static long seed = 13579;

    /// <summary>LCG RAG: every resource gets one holder or stays free; processes issue 0..2 requests.</summary>
    private static Rag RandomRag(int processes, int resources)
    {
        var holder = new int[resources];
        for (int i = 0; i < resources; i++) holder[i] = -1;
        var holds = new List<int[]>();
        for (int resource = 0; resource < resources; resource++)
        {
            seed = (seed * 75 + 74) % 65537;
            if (seed % 3 != 0)
            {
                int process = (int)(seed % processes);
                holder[resource] = process;
                holds.Add(new[] { process, resource });
            }
        }
        var requests = new List<int[]>();
        for (int process = 0; process < processes; process++)
        {
            seed = (seed * 75 + 74) % 65537;
            int count = (int)(seed % 3);
            for (int i = 0; i < count; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                requests.Add(new[] { process, (int)(seed % resources) });
            }
        }
        return new Rag { Holds = holds.ToArray(), Requests = requests.ToArray() };
    }

    public static void Main()
    {
        // Hand-built deadlock: two processes wait for each other; canonical cycle [0, 1].
        var adjacency = BuildWaitFor(2, 2, new[] { new[] { 0, 0 }, new[] { 1, 1 } },
                                     new[] { new[] { 0, 1 }, new[] { 1, 0 } });
        Check(adjacency[0].SequenceEqual(new[] { 1 }) && adjacency[1].SequenceEqual(new[] { 0 }),
              "wait-for edges");
        Check(ListsEqual(FindCycle(2, adjacency)!, new List<int> { 0, 1 }), "classic two-process deadlock");

        // Acyclic: process 0 waits for process 1, which waits for nothing.
        adjacency = BuildWaitFor(2, 2, new[] { new[] { 0, 0 }, new[] { 1, 1 } }, new[] { new[] { 0, 1 } });
        Check(adjacency[0].SequenceEqual(new[] { 1 }) && adjacency[1].Count == 0, "one-way wait edge");
        Check(FindCycle(2, adjacency) == null, "no cycle without a wait cycle");

        // Self-loop: a process requests a resource it already holds.
        adjacency = BuildWaitFor(1, 1, new[] { new[] { 0, 0 } }, new[] { new[] { 0, 0 } });
        Check(adjacency[0].SequenceEqual(new[] { 0 }), "self-loop edge");
        Check(ListsEqual(FindCycle(1, adjacency)!, new List<int> { 0 }), "self-loop is detected");

        // Three-process cycle; the returned cycle is the hand-checked canonical rotation.
        adjacency = BuildWaitFor(3, 3, new[] { new[] { 1, 0 }, new[] { 2, 1 }, new[] { 0, 2 } },
                                 new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 2, 2 } });
        Check(adjacency[0].SequenceEqual(new[] { 1 }) && adjacency[1].SequenceEqual(new[] { 2 })
              && adjacency[2].SequenceEqual(new[] { 0 }), "three-cycle edges");
        Check(ListsEqual(FindCycle(3, adjacency)!, new List<int> { 0, 1, 2 }), "three-process deadlock");

        // Requests for free resources create no edges.
        adjacency = BuildWaitFor(2, 2, new[] { new[] { 0, 0 } }, new[] { new[] { 1, 1 } });
        Check(adjacency[0].Count == 0 && adjacency[1].Count == 0, "free resource: no edge");
        Check(FindCycle(2, adjacency) == null, "waiting for a free resource is not a deadlock");

        // All processes idle, and an extra process outside the cycle.
        adjacency = BuildWaitFor(3, 3, new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 2, 2 } }, Array.Empty<int[]>());
        Check(FindCycle(3, adjacency) == null, "idle processes never deadlock");
        adjacency = BuildWaitFor(3, 3, new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 2, 2 } },
                                 new[] { new[] { 0, 1 }, new[] { 1, 0 } });
        Check(ListsEqual(FindCycle(3, adjacency)!, new List<int> { 0, 1 }), "cycle independent of an idle process");

        // A resource may have only one holder.
        bool rejected = false;
        try
        {
            BuildWaitFor(2, 1, new[] { new[] { 0, 0 }, new[] { 1, 0 } }, Array.Empty<int[]>());
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        Check(rejected, "a double-held resource must be rejected");

        // Cross-check cycle existence against the independent SCC check on LCG RAGs.
        seed = 13579;
        int cyclesSeen = 0;
        for (int trial = 0; trial < ScriptTrials; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int processes = 2 + (int)(seed % 6);
            seed = (seed * 75 + 74) % 65537;
            int resources = 1 + (int)(seed % 6);
            var rag = RandomRag(processes, resources);
            var waitFor = BuildWaitFor(processes, resources, rag.Holds, rag.Requests);
            var cycle = FindCycle(processes, waitFor);
            bool sccSaysCycle = HasCycleViaScc(processes, waitFor);
            Check((cycle != null) == sccSaysCycle, "SCC cross-check at trial " + trial);
            if (cycle != null)
            {
                cyclesSeen++;
                AssertValidCycle(waitFor, cycle);
            }
        }
        Check(cyclesSeen >= 20, "LCG graphs should contain deadlocks (saw " + cyclesSeen + ")");

        Console.WriteLine("PASS");
    }

    private static bool ListsEqual(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
