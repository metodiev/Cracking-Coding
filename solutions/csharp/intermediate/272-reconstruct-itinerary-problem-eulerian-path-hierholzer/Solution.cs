// Task 272 - Reconstruct itinerary (Eulerian path, Hierholzer). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Returns the airport sequence, or null when the tickets cannot all be used.</summary>
    public static List<string>? FindItinerary(string[][] tickets, string start)
    {
        var adjacency = new SortedDictionary<string, List<string>>();
        foreach (var ticket in tickets)
        {
            if (!adjacency.TryGetValue(ticket[0], out var list))
            {
                list = new List<string>();
                adjacency[ticket[0]] = list;
            }
            list.Add(ticket[1]);
        }
        foreach (var list in adjacency.Values) list.Sort();

        var route = new List<string>();
        var stack = new Stack<string>();
        stack.Push(start);
        while (stack.Count > 0)
        {
            string airport = stack.Peek();
            var outgoing = adjacency.TryGetValue(airport, out var list) ? list : null;
            if (outgoing != null && outgoing.Count > 0)
            {
                string next = outgoing[0];
                outgoing.RemoveAt(0);
                stack.Push(next);
            }
            else
            {
                route.Add(stack.Pop());
            }
        }
        route.Reverse();
        return route.Count == tickets.Length + 1 ? route : null;
    }

    /// <summary>Independent check: consecutive pairs of `route` must consume the ticket multiset.</summary>
    public static bool UsesAllTickets(List<string>? route, string[][] tickets)
    {
        if (route == null) return false;
        var used = new SortedDictionary<string, int>();
        for (int i = 0; i + 1 < route.Count; i++)
        {
            string key = route[i] + "->" + route[i + 1];
            used[key] = used.TryGetValue(key, out int c) ? c + 1 : 1;
        }
        var wanted = new SortedDictionary<string, int>();
        foreach (var ticket in tickets)
        {
            string key = ticket[0] + "->" + ticket[1];
            wanted[key] = wanted.TryGetValue(key, out int c) ? c + 1 : 1;
        }
        return used.Count == wanted.Count && wanted.All(kv => used.TryGetValue(kv.Key, out int v) && v == kv.Value);
    }

    private static string Join(List<string>? values) => values == null ? "null" : string.Join(",", values);

    public static void Main()
    {
        var first = new[] { new[] { "MUC", "LHR" }, new[] { "JFK", "MUC" }, new[] { "SFO", "SJC" }, new[] { "LHR", "SFO" } };
        var route = FindItinerary(first, "JFK");
        Check(Join(route) == "JFK,MUC,LHR,SFO,SJC", "classic itinerary");
        Check(UsesAllTickets(route, first), "all tickets used");

        var second = new[] { new[] { "JFK", "SFO" }, new[] { "JFK", "ATL" }, new[] { "SFO", "ATL" }, new[] { "ATL", "JFK" }, new[] { "ATL", "SFO" } };
        route = FindItinerary(second, "JFK");
        Check(Join(route) == "JFK,ATL,JFK,SFO,ATL,SFO", "reconstruction itinerary");
        Check(UsesAllTickets(route, second), "all tickets used (2)");

        Check(Join(FindItinerary(new[] { new[] { "A", "B" } }, "A")) == "A,B", "single ticket");
        Check(Join(FindItinerary(Array.Empty<string[]>(), "JFK")) == "JFK", "no tickets");
        Check(FindItinerary(new[] { new[] { "A", "B" }, new[] { "C", "D" } }, "A") == null, "disconnected tickets");
        Check(FindItinerary(new[] { new[] { "A", "B" } }, "C") == null, "nothing leaves the start");

        var dupes = new[] { new[] { "JFK", "B" }, new[] { "JFK", "A" }, new[] { "A", "JFK" } };
        Check(Join(FindItinerary(dupes, "JFK")) == "JFK,A,JFK,B", "smallest destination first");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
