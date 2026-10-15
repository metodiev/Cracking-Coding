// Task 472 - Implement basic Raft consensus pseudo-simulation (leader election outline). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
using System;
using System.Collections.Generic;

internal class Entry
{
    public readonly int Term;
    public readonly string Cmd;

    public Entry(int term, string cmd)
    {
        Term = term;
        Cmd = cmd;
    }

    public bool Same(Entry o) => o != null && Term == o.Term && Cmd == o.Cmd;
}

internal class Node
{
    public const int N = 5;
    public readonly int Id;
    public int Term;
    public int VotedFor = -1;
    public int Commit;
    public string State = "F"; // F follower, C candidate, L leader
    public readonly List<Entry> Log = new List<Entry>();
    public readonly HashSet<int> Voters = new HashSet<int>();
    public readonly int[] NextIndex = new int[N];
    public readonly int[] MatchIndex = new int[N];

    public Node(int id)
    {
        Id = id;
        for (int i = 0; i < N; i++) MatchIndex[i] = -1;
    }
}

internal abstract class Msg
{
    public readonly int Term;

    protected Msg(int term)
    {
        Term = term;
    }
}

internal class RvMsg : Msg
{
    public readonly int Cand, LastIdx, LastTerm;

    public RvMsg(int term, int cand, int lastIdx, int lastTerm) : base(term)
    {
        Cand = cand;
        LastIdx = lastIdx;
        LastTerm = lastTerm;
    }
}

internal class RvRespMsg : Msg
{
    public readonly int Voter;
    public readonly bool Granted;

    public RvRespMsg(int term, int voter, bool granted) : base(term)
    {
        Voter = voter;
        Granted = granted;
    }
}

internal class AeMsg : Msg
{
    public readonly int Leader, PrevIdx, PrevTerm, Commit;
    public readonly List<Entry> Entries;

    public AeMsg(int term, int leader, int prevIdx, int prevTerm, List<Entry> entries, int commit) : base(term)
    {
        Leader = leader;
        PrevIdx = prevIdx;
        PrevTerm = prevTerm;
        Entries = entries;
        Commit = commit;
    }
}

internal class AeRespMsg : Msg
{
    public readonly int Follower;
    public readonly bool Success;
    public readonly int Hint;

    public AeRespMsg(int term, int follower, bool success, int hint) : base(term)
    {
        Follower = follower;
        Success = success;
        Hint = hint;
    }
}

internal class InFlight
{
    public readonly int From, To;
    public readonly Msg Message;

    public InFlight(int from, int to, Msg message)
    {
        From = from;
        To = to;
        Message = message;
    }
}

internal class RaftSim
{
    public const int N = Node.N;
    private const int Majority = 3;

    public readonly List<Node> Nodes = new List<Node>();
    public readonly bool[] Alive = new bool[N];
    private List<int[]> partition;
    private readonly Queue<InFlight> queue = new Queue<InFlight>();
    public readonly SortedDictionary<int, int> LeadersByTerm = new SortedDictionary<int, int>();
    public readonly SortedDictionary<int, Entry> Committed = new SortedDictionary<int, Entry>();
    public int CommittedUpTo;
    private readonly int[] lastTerm = new int[N];
    private readonly int[] lastCommit = new int[N];

    public RaftSim()
    {
        for (int i = 0; i < N; i++)
        {
            Nodes.Add(new Node(i));
            Alive[i] = true;
        }
    }

    private bool Connected(int a, int b)
    {
        if (partition == null) return true;
        foreach (int[] g in partition)
        {
            bool hasA = false, hasB = false;
            foreach (int x in g)
            {
                if (x == a) hasA = true;
                if (x == b) hasB = true;
            }
            if (hasA && hasB) return true;
        }
        return false;
    }

    private void Send(int frm, int to, Msg msg)
    {
        if (!Alive[frm] || !Alive[to]) return;
        if (!Connected(frm, to)) return;
        queue.Enqueue(new InFlight(frm, to, msg));
    }

    public void Drain()
    {
        while (queue.Count > 0)
        {
            InFlight f = queue.Dequeue();
            if (!Alive[f.From] || !Alive[f.To] || !Connected(f.From, f.To)) continue;
            Deliver(f.From, f.To, f.Message);
        }
    }

    public void Crash(int i) => Alive[i] = false;

    public void RecoverNode(int i)
    {
        Alive[i] = true;
        Nodes[i].State = "F";
    }

    public void PartitionOff(int[][] groups) => partition = new List<int[]>(groups);

    public void Heal() => partition = null;

    public void Timeout(int i)
    {
        if (!Alive[i]) return;
        Node n = Nodes[i];
        n.Term += 1;
        n.VotedFor = i;
        n.State = "C";
        n.Voters.Clear();
        n.Voters.Add(i);
        int lastIdx = n.Log.Count - 1;
        int lastLogTerm = n.Log.Count == 0 ? 0 : n.Log[n.Log.Count - 1].Term;
        for (int j = 0; j < N; j++)
        {
            if (j != i) Send(i, j, new RvMsg(n.Term, i, lastIdx, lastLogTerm));
        }
    }

    public bool Client(string cmd)
    {
        Node best = null;
        foreach (Node n in Nodes)
        {
            if (n.State == "L" && Alive[n.Id] && (best == null || n.Term > best.Term)) best = n;
        }
        if (best == null) return false;
        best.Log.Add(new Entry(best.Term, cmd));
        return true;
    }

    public void Replicate()
    {
        foreach (Node n in Nodes)
        {
            if (n.State != "L" || !Alive[n.Id]) continue;
            for (int f = 0; f < N; f++)
            {
                if (f == n.Id) continue;
                int nn = Math.Min(n.NextIndex[f], n.Log.Count);
                int prevIdx = nn - 1;
                int prevTerm = prevIdx >= 0 ? n.Log[prevIdx].Term : 0;
                var entries = new List<Entry>();
                for (int k = nn; k < n.Log.Count; k++) entries.Add(n.Log[k]);
                Send(n.Id, f, new AeMsg(n.Term, n.Id, prevIdx, prevTerm, entries, n.Commit));
            }
        }
    }

    private void Deliver(int frm, int to, Msg msg)
    {
        Node n = Nodes[to];
        if (msg is RvMsg rv)
        {
            if (rv.Term < n.Term)
            {
                Send(to, frm, new RvRespMsg(n.Term, to, false));
                return;
            }
            if (rv.Term > n.Term)
            {
                n.Term = rv.Term;
                n.VotedFor = -1;
                if (n.State != "F") n.State = "F";
            }
            int ownIdx = n.Log.Count - 1;
            int ownTerm = n.Log.Count == 0 ? 0 : n.Log[n.Log.Count - 1].Term;
            bool upToDate = rv.LastTerm > ownTerm || (rv.LastTerm == ownTerm && rv.LastIdx >= ownIdx);
            bool granted = (n.VotedFor == -1 || n.VotedFor == rv.Cand) && upToDate;
            if (granted) n.VotedFor = rv.Cand;
            Send(to, frm, new RvRespMsg(n.Term, to, granted));
        }
        else if (msg is RvRespMsg resp)
        {
            if (resp.Term > n.Term)
            {
                n.Term = resp.Term;
                n.VotedFor = -1;
                n.State = "F";
                return;
            }
            if (n.State == "C" && resp.Term == n.Term && resp.Granted && !n.Voters.Contains(resp.Voter))
            {
                n.Voters.Add(resp.Voter);
                if (n.Voters.Count >= Majority) BecomeLeader(n);
            }
        }
        else if (msg is AeMsg ae)
        {
            if (ae.Term < n.Term)
            {
                Send(to, frm, new AeRespMsg(n.Term, to, false, n.Log.Count));
                return;
            }
            if (ae.Term > n.Term)
            {
                n.Term = ae.Term;
                n.VotedFor = -1;
            }
            n.State = "F";
            if (ae.PrevIdx >= n.Log.Count || (ae.PrevIdx >= 0 && n.Log[ae.PrevIdx].Term != ae.PrevTerm))
            {
                Send(to, frm, new AeRespMsg(n.Term, to, false, n.Log.Count));
                return;
            }
            int idx = ae.PrevIdx + 1;
            foreach (Entry entry in ae.Entries)
            {
                if (idx < n.Log.Count)
                {
                    if (n.Log[idx].Term != entry.Term)
                    {
                        n.Log.RemoveRange(idx, n.Log.Count - idx);
                        n.Log.Add(entry);
                    }
                }
                else
                {
                    n.Log.Add(entry);
                }
                idx++;
            }
            int lastNew = ae.PrevIdx + ae.Entries.Count + 1;
            if (ae.Commit > n.Commit)
            {
                int newCommit = Math.Min(ae.Commit, lastNew);
                for (int i = n.Commit; i < newCommit; i++) RecordCommit(i, n.Log[i]);
                n.Commit = Math.Max(n.Commit, newCommit);
            }
            Send(to, frm, new AeRespMsg(n.Term, to, true, lastNew));
        }
        else if (msg is AeRespMsg aeResp)
        {
            if (aeResp.Term > n.Term)
            {
                n.Term = aeResp.Term;
                n.VotedFor = -1;
                n.State = "F";
                return;
            }
            if (n.State != "L" || aeResp.Term != n.Term) return;
            if (aeResp.Success)
            {
                if (aeResp.Hint > n.MatchIndex[aeResp.Follower])
                {
                    n.MatchIndex[aeResp.Follower] = aeResp.Hint;
                    n.NextIndex[aeResp.Follower] = aeResp.Hint + 1;
                }
                int newCommit = n.Commit;
                for (int cand = n.Log.Count; cand > n.Commit; cand--)
                {
                    if (n.Log[cand - 1].Term != n.Term) continue;
                    int votes = 0;
                    for (int j = 0; j < N; j++)
                    {
                        if ((j == n.Id && n.Log.Count >= cand) || n.MatchIndex[j] >= cand) votes++;
                    }
                    if (votes >= Majority)
                    {
                        newCommit = cand;
                        break;
                    }
                }
                if (newCommit > n.Commit)
                {
                    for (int i = n.Commit; i < newCommit; i++) RecordCommit(i, n.Log[i]);
                    n.Commit = newCommit;
                }
            }
            else
            {
                n.NextIndex[aeResp.Follower] = Math.Max(0, Math.Min(n.NextIndex[aeResp.Follower] - 1, aeResp.Hint));
            }
        }
    }

    private void BecomeLeader(Node n)
    {
        n.State = "L";
        if (LeadersByTerm.TryGetValue(n.Term, out int prev) && prev != n.Id)
        {
            throw new Exception("FAILED: two leaders in same term");
        }
        LeadersByTerm[n.Term] = n.Id;
        for (int i = 0; i < N; i++)
        {
            n.NextIndex[i] = n.Log.Count;
            n.MatchIndex[i] = -1;
        }
        for (int i = 0; i < CommittedUpTo; i++)
        {
            if (!(i < n.Log.Count && n.Log[i].Same(Committed[i])))
            {
                throw new Exception("FAILED: elected leader lacks committed entry " + i);
            }
        }
    }

    private void RecordCommit(int i, Entry entry)
    {
        if (Committed.TryGetValue(i, out Entry prev) && !prev.Same(entry))
        {
            throw new Exception("FAILED: committed entry changed at " + i);
        }
        Committed[i] = entry;
        while (Committed.ContainsKey(CommittedUpTo)) CommittedUpTo++;
    }

    public void CheckInvariants()
    {
        for (int a = 0; a < N; a++)
        {
            for (int b = a + 1; b < N; b++)
            {
                List<Entry> la = Nodes[a].Log, lb = Nodes[b].Log;
                int m = Math.Min(la.Count, lb.Count);
                for (int i = 0; i < m; i++)
                {
                    if (la[i].Term == lb[i].Term)
                    {
                        for (int k = 0; k <= i; k++)
                        {
                            if (!la[k].Same(lb[k]))
                            {
                                throw new Exception("FAILED: log matching violated " + a + "," + b + "@" + i);
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < N; i++)
        {
            Node n = Nodes[i];
            if (n.Commit > n.Log.Count) throw new Exception("FAILED: commit beyond log");
            if (n.Term < lastTerm[i]) throw new Exception("FAILED: term regression");
            if (n.Commit < lastCommit[i]) throw new Exception("FAILED: commit regression");
            lastTerm[i] = n.Term;
            lastCommit[i] = n.Commit;
        }
    }

    public void Run(List<object[]> actions)
    {
        foreach (object[] action in actions)
        {
            string kind = (string)action[0];
            if (kind == "timeout") Timeout((int)action[1]);
            else if (kind == "client") Client((string)action[1]);
            else if (kind == "replicate") Replicate();
            else if (kind == "drain") Drain();
            else if (kind == "partition") PartitionOff((int[][])action[1]);
            else if (kind == "heal") Heal();
            else if (kind == "crash") Crash((int)action[1]);
            else if (kind == "recover") RecoverNode((int)action[1]);
            CheckInvariants();
        }
    }
}

public static class Solution
{
    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static List<object[]> RandomActions(int count, int seed)
    {
        var actions = new List<object[]>();
        int counter = 0;
        for (int k = 0; k < count; k++)
        {
            seed = Lcg(seed);
            int a = seed % 8;
            seed = Lcg(seed);
            int node = seed % RaftSim.N;
            if (a == 0)
            {
                actions.Add(new object[] { "timeout", node });
            }
            else if (a == 1)
            {
                counter++;
                actions.Add(new object[] { "client", "c" + counter });
            }
            else if (a == 2)
            {
                actions.Add(new object[] { "replicate" });
            }
            else if (a == 3)
            {
                actions.Add(new object[] { "drain" });
            }
            else if (a == 4)
            {
                seed = Lcg(seed);
                int mask = seed;
                var g0 = new List<int>();
                var g1 = new List<int>();
                for (int i = 0; i < RaftSim.N; i++)
                {
                    if (((mask >> i) & 1) == 1) g0.Add(i);
                    else g1.Add(i);
                }
                actions.Add(new object[] { "partition", new[] { g0.ToArray(), g1.ToArray() } });
            }
            else if (a == 5)
            {
                actions.Add(new object[] { "heal" });
            }
            else if (a == 6)
            {
                actions.Add(new object[] { "crash", node });
            }
            else
            {
                actions.Add(new object[] { "recover", node });
            }
            if (a != 3) actions.Add(new object[] { "drain" });
        }
        return actions;
    }

    private static bool SameLog(List<Entry> a, List<Entry> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (!a[i].Same(b[i])) return false;
        return true;
    }

    public static void Main()
    {
        var sim = new RaftSim();
        sim.Run(new List<object[]> { new object[] { "timeout", 0 }, new object[] { "drain" } });
        Check(sim.LeadersByTerm.Count == 1 && sim.LeadersByTerm[1] == 0, "term 1 leader");
        sim.Run(new List<object[]>
        {
            new object[] { "client", "x=1" }, new object[] { "replicate" }, new object[] { "drain" },
            new object[] { "replicate" }, new object[] { "drain" },
        });
        Check(sim.CommittedUpTo == 1, "x committed");
        Check(sim.Committed[0].Same(new Entry(1, "x=1")), "x entry");

        sim.Run(new List<object[]>
        {
            new object[] { "crash", 0 }, new object[] { "timeout", 1 }, new object[] { "drain" },
        });
        Check(sim.LeadersByTerm.TryGetValue(2, out int l2) && l2 == 1, "term 2 leader");
        sim.Run(new List<object[]>
        {
            new object[] { "client", "y=2" }, new object[] { "replicate" }, new object[] { "drain" },
            new object[] { "replicate" }, new object[] { "drain" },
        });
        Check(sim.CommittedUpTo == 2 && sim.Committed[1].Same(new Entry(2, "y=2")), "y committed");

        sim.Run(new List<object[]>
        {
            new object[] { "partition", new[] { new[] { 0, 1, 2 }, new[] { 3, 4 } } },
            new object[] { "client", "z=3" }, new object[] { "replicate" }, new object[] { "drain" },
            new object[] { "replicate" }, new object[] { "drain" },
        });
        Check(sim.CommittedUpTo == 2, "z must not commit without quorum");
        Node n1 = sim.Nodes[1];
        Node n2 = sim.Nodes[2];
        Check(n1.Log[n1.Log.Count - 1].Same(new Entry(2, "z=3")), "z appended at leader");
        Check(n2.Log[n2.Log.Count - 1].Same(new Entry(2, "z=3")), "z replicated to peer");

        sim.Run(new List<object[]> { new object[] { "timeout", 3 }, new object[] { "drain" } });
        Check(!sim.LeadersByTerm.ContainsKey(3), "minority cannot elect");
        sim.Run(new List<object[]> { new object[] { "heal" }, new object[] { "drain" } });
        sim.Run(new List<object[]> { new object[] { "timeout", 1 }, new object[] { "drain" } });
        Check(!sim.LeadersByTerm.ContainsKey(3), "stale votes block term 3");
        sim.Run(new List<object[]> { new object[] { "timeout", 1 }, new object[] { "drain" } });
        Check(sim.LeadersByTerm.TryGetValue(4, out int l4) && l4 == 1, "term 4 leader");

        sim.Run(new List<object[]>
        {
            new object[] { "client", "w=4" }, new object[] { "replicate" }, new object[] { "drain" },
            new object[] { "replicate" }, new object[] { "drain" }, new object[] { "replicate" }, new object[] { "drain" },
        });
        Check(sim.CommittedUpTo == 4, "w commits z");
        Check(sim.Committed[2].Same(new Entry(2, "z=3")), "z committed");
        Check(sim.Committed[3].Same(new Entry(4, "w=4")), "w committed");
        var expected = new List<Entry>
        {
            new Entry(1, "x=1"), new Entry(2, "y=2"), new Entry(2, "z=3"), new Entry(4, "w=4"),
        };
        for (int i = 1; i <= 4; i++) Check(SameLog(sim.Nodes[i].Log, expected), "log mismatch node " + i);

        sim.Run(RandomActions(64, 24680));

        var finale = new List<object[]> { new object[] { "heal" } };
        for (int i = 0; i < RaftSim.N; i++) finale.Add(new object[] { "recover", i });
        finale.Add(new object[] { "drain" });
        finale.Add(new object[] { "drain" });
        sim.Run(finale);
        int best = 0;
        for (int i = 1; i < RaftSim.N; i++)
        {
            Node a = sim.Nodes[i], b = sim.Nodes[best];
            int ta = a.Log.Count == 0 ? 0 : a.Log[a.Log.Count - 1].Term;
            int tb = b.Log.Count == 0 ? 0 : b.Log[b.Log.Count - 1].Term;
            if (ta > tb || (ta == tb && a.Log.Count > b.Log.Count)) best = i;
        }
        for (int attempt = 0; attempt < 40; attempt++)
        {
            sim.Run(new List<object[]> { new object[] { "timeout", best }, new object[] { "drain" } });
            if (sim.Nodes[best].State == "L") break;
        }
        Check(sim.Nodes[best].State == "L", "most up-to-date node must win");
        sim.Run(new List<object[]> { new object[] { "client", "end" } });
        for (int attempt = 0; attempt < 12; attempt++)
        {
            sim.Run(new List<object[]>
            {
                new object[] { "replicate" }, new object[] { "drain" },
                new object[] { "replicate" }, new object[] { "drain" },
            });
            if (sim.CommittedUpTo == sim.Nodes[best].Log.Count) break;
        }
        Check(sim.CommittedUpTo == sim.Nodes[best].Log.Count, "final commit caught up");
        bool sawEnd = false;
        foreach (Entry e in sim.Committed.Values)
        {
            if (e.Same(new Entry(sim.Nodes[best].Term, "end"))) sawEnd = true;
        }
        Check(sawEnd, "end command committed");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
