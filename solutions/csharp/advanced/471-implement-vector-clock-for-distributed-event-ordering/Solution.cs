// Task 471 - Implement vector clock for distributed event ordering. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
using System;
using System.Collections.Generic;

internal class Event
{
    public readonly int Node;
    public readonly char Kind; // 'L' local, 'S' send, 'R' receive
    public readonly int Msg;
    public readonly int[] Clock;

    public Event(int node, char kind, int msg, int[] clock)
    {
        Node = node;
        Kind = kind;
        Msg = msg;
        Clock = clock;
    }
}

internal class SentMsg
{
    public readonly int Id;
    public readonly int[] Clock;

    public SentMsg(int id, int[] clock)
    {
        Id = id;
        Clock = clock;
    }
}

public static class Solution
{
    private const int Nodes = 3;

    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static string Classify(int[] a, int[] b)
    {
        bool le = true, ge = true;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] > b[i]) le = false;
            if (a[i] < b[i]) ge = false;
        }
        if (le && ge) return "equal";
        if (le) return "before";
        if (ge) return "after";
        return "concurrent";
    }

    private static void Emit(List<Event> outList, int[][] c, int node, char kind, int msg)
    {
        outList.Add(new Event(node, kind, msg, (int[])c[node].Clone()));
    }

    private static List<Event> FeaturedEvents()
    {
        var c = new int[Nodes][];
        for (int i = 0; i < Nodes; i++) c[i] = new int[Nodes];
        var events = new List<Event>();
        c[0][0]++;
        Emit(events, c, 0, 'L', -1);          // 0: A has 1 local event
        c[0][0]++;
        Emit(events, c, 0, 'L', -1);          // 1: A has 2
        c[0][0]++;
        Emit(events, c, 0, 'S', 0);           // 2: A sends {A:3} to B
        c[1][1]++;
        Emit(events, c, 1, 'L', -1);          // 3: B local tick, unaware of A
        c[1][0] = Math.Max(c[1][0], 3);
        c[1][1]++;
        Emit(events, c, 1, 'R', 0);           // 4: {A:3, B:2}
        c[1][1]++;
        Emit(events, c, 1, 'S', 1);           // 5: B sends {A:3, B:3} to C
        c[2][0] = Math.Max(c[2][0], 3);
        c[2][1] = Math.Max(c[2][1], 3);
        c[2][2]++;
        Emit(events, c, 2, 'R', 1);           // 6: C merges -> {3,3,1}
        return events;
    }

    private static List<Event> GenScript(int count, int seed)
    {
        var events = new List<Event>();
        var pending = new List<SentMsg>[Nodes];
        for (int i = 0; i < Nodes; i++) pending[i] = new List<SentMsg>();
        var clocks = new int[Nodes][];
        for (int i = 0; i < Nodes; i++) clocks[i] = new int[Nodes];
        int nextMsg = 0;
        for (int k = 0; k < count; k++)
        {
            seed = Lcg(seed);
            int node = seed % Nodes;
            seed = Lcg(seed);
            int kind = seed % 3;
            if (kind == 2 && pending[node].Count == 0) kind = 0;
            if (kind == 1)
            {
                seed = Lcg(seed);
                int target = (node + 1 + seed % (Nodes - 1)) % Nodes;
                clocks[node][node]++;
                pending[target].Add(new SentMsg(nextMsg, (int[])clocks[node].Clone()));
                events.Add(new Event(node, 'S', nextMsg, (int[])clocks[node].Clone()));
                nextMsg++;
            }
            else if (kind == 2)
            {
                SentMsg m = pending[node][0];
                pending[node].RemoveAt(0);
                for (int i = 0; i < Nodes; i++) clocks[node][i] = Math.Max(clocks[node][i], m.Clock[i]);
                clocks[node][node]++;
                events.Add(new Event(node, 'R', m.Id, (int[])clocks[node].Clone()));
            }
            else
            {
                clocks[node][node]++;
                events.Add(new Event(node, 'L', -1, (int[])clocks[node].Clone()));
            }
        }
        return events;
    }

    private static bool[][] CausalReachability(List<Event> events)
    {
        int n = events.Count;
        var reach = new bool[n][];
        for (int i = 0; i < n; i++) reach[i] = new bool[n];
        var last = new int[Nodes];
        for (int i = 0; i < Nodes; i++) last[i] = -1;
        var sentAt = new int[n];
        for (int i = 0; i < n; i++) sentAt[i] = -1;
        for (int i = 0; i < n; i++)
        {
            Event e = events[i];
            if (last[e.Node] != -1) reach[last[e.Node]][i] = true;
            last[e.Node] = i;
            if (e.Kind == 'S') sentAt[e.Msg] = i;
            else if (e.Kind == 'R') reach[sentAt[e.Msg]][i] = true;
        }
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (reach[i][k])
                    for (int j = 0; j < n; j++)
                        if (reach[k][j]) reach[i][j] = true;
        return reach;
    }

    public static void Main()
    {
        var fe = FeaturedEvents();
        var cl = new int[fe.Count][];
        for (int i = 0; i < fe.Count; i++) cl[i] = fe[i].Clock;
        Check(Same(cl[0], 1, 0, 0), "clock 0");
        Check(Same(cl[1], 2, 0, 0), "clock 1");
        Check(Same(cl[2], 3, 0, 0), "clock 2");
        Check(Same(cl[3], 0, 1, 0), "clock 3");
        Check(Same(cl[4], 3, 2, 0), "clock 4");
        Check(Same(cl[5], 3, 3, 0), "clock 5");
        Check(Same(cl[6], 3, 3, 1), "clock 6");
        Check(Classify(cl[0], cl[1]) == "before", "0->1");
        Check(Classify(cl[1], cl[2]) == "before", "1->2");
        Check(Classify(cl[0], cl[2]) == "before", "0->2");
        Check(Classify(cl[3], cl[0]) == "concurrent", "3 vs 0");
        Check(Classify(cl[3], cl[1]) == "concurrent", "3 vs 1");
        Check(Classify(cl[3], cl[2]) == "concurrent", "3 vs 2");
        Check(Classify(cl[3], cl[4]) == "before", "3->4");
        Check(Classify(cl[2], cl[4]) == "before", "2->4");
        Check(Classify(cl[4], cl[5]) == "before", "4->5");
        Check(Classify(cl[5], cl[6]) == "before", "5->6");
        Check(Classify(cl[1], cl[6]) == "before", "1->6");
        Check(Classify(cl[6], cl[4]) == "after", "6 after 4");
        Check(Classify(cl[6], cl[0]) == "after", "6 after 0");
        for (int i = 0; i < fe.Count; i++)
            for (int j = i + 1; j < fe.Count; j++)
                Check(Classify(cl[i], cl[j]) != "equal", "no equal clocks");

        var events = GenScript(26, 1234);
        int l = 0, s = 0, r = 0;
        foreach (Event e in events)
        {
            if (e.Kind == 'L') l++;
            else if (e.Kind == 'S') s++;
            else r++;
        }
        Check(l == 20 && s == 4 && r == 2, "script shape");
        var reach = CausalReachability(events);
        int order = 0, concurrent = 0;
        for (int i = 0; i < events.Count; i++)
        {
            for (int j = i + 1; j < events.Count; j++)
            {
                string got = Classify(events[i].Clock, events[j].Clock);
                string expected = reach[i][j] ? "before" : reach[j][i] ? "after" : "concurrent";
                Check(got == expected, "pair " + i + "," + j + ": " + got + " vs " + expected);
                if (expected == "concurrent") concurrent++;
                else order++;
            }
        }
        Check(order == 140 && concurrent == 185, "pair counts");
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] clock, int a, int b, int c) => clock[0] == a && clock[1] == b && clock[2] == c;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
