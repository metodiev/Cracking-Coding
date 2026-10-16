// Task 628 - Implement distributed consensus simulation with network partitions (Raft test harness). (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// This is a compact but faithful Raft: leader election with terms and one vote per term,
// log replication with the prevLogIndex/prevLogTerm consistency check, the commit rule that only
// entries from the leader's OWN term may be committed directly (older entries commit indirectly),
// and the up-to-date vote rule that makes the leader-completeness property hold.
//
//   * A follower whose election timer expires becomes a candidate, bumps the term, votes for
//     itself and requests votes; a majority of grants makes it leader.
//   * The leader heartbeats every tick; a follower truncates its log where it conflicts with the
//     leader's entries, so divergent uncommitted suffixes are repaired.
//   * The leader advances commitIndex to the highest index that a majority has acknowledged,
//     provided that entry belongs to the current term.
// Quorums are computed over the whole configured membership, so a minority island can never
// elect a leader or commit anything - that is the property that makes consensus safe under
// partitions.
//
// What is simplified (documented per the authoring contract): no threads, no sockets, no timers -
// the cluster is an explicit in-memory message queue driven by a logical tick clock, message
// latency is one tick, elections use staggered *deterministic* timeouts (node id based) instead of
// randomized ones, and there is no persistence, snapshots or cluster membership change. Verified
// invariants, re-checked after EVERY tick: at most one leader per term, the log matching property
// (same index+term implies identical prefixes), committed entries never change, and every new
// leader contains all previously committed entries.
using System;
using System.Collections.Generic;

#nullable enable

internal enum RaftRole { Follower, Candidate, Leader }

internal sealed class LogEntry
{
    public int Term;
    public int Index;
    public string Command = "";

    public LogEntry(int term, int index, string command)
    {
        Term = term;
        Index = index;
        Command = command;
    }
}

internal sealed class RaftMessage
{
    public int Seq;
    public int From;
    public int To;
    public string Kind = "";
    public int Term;
    public int LastLogIndex;
    public int LastLogTerm;
    public int PrevLogIndex;
    public int PrevLogTerm;
    public List<LogEntry> Entries = new List<LogEntry>();
    public int LeaderCommit;
    public bool Granted;
    public int MatchIndex;
    public int DeliverTick;
}

internal sealed class RaftNode
{
    public readonly int Id;
    public readonly int ElectionTimeout;
    public int CurrentTerm;
    public int VotedFor = -1;
    public RaftRole Role = RaftRole.Follower;
    public readonly List<LogEntry> Log = new List<LogEntry> { new LogEntry(0, 0, "") };
    public int CommitIndex;
    public int ElectionElapsed;
    public readonly HashSet<int> Grants = new HashSet<int>();
    public readonly Dictionary<int, int> NextIndex = new Dictionary<int, int>();
    public readonly Dictionary<int, int> MatchIndex = new Dictionary<int, int>();
    public readonly List<string> Applied = new List<string>();
    public int LeaderId = -1;

    public RaftNode(int id, int electionTimeout)
    {
        Id = id;
        ElectionTimeout = electionTimeout;
    }

    public int LastLogIndex => Log.Count - 1;
    public int LastLogTerm => Log[Log.Count - 1].Term;
    public int TermAt(int index) => index <= LastLogIndex ? Log[index].Term : 0;
    public string CommandAt(int index) => Log[index].Command;
}

internal sealed class RaftCluster
{
    public readonly List<RaftNode> Nodes = new List<RaftNode>();
    private readonly List<RaftMessage> queue = new List<RaftMessage>();
    private readonly HashSet<string> blocked = new HashSet<string>();
    private int seq;

    public int Now;
    public int MessagesSent;
    public int MessagesDropped;
    public int ElectionsStarted;
    public int LeaderChanges;
    public int LeaderPerTermViolations;
    public int LogMatchingViolations;
    public int CommittedChangedViolations;
    public int LeaderCompletenessViolations;
    public readonly Dictionary<int, string> CommittedCommand = new Dictionary<int, string>();
    public readonly Dictionary<int, int> CommittedTerm = new Dictionary<int, int>();

    public RaftCluster(int size, int baseTimeout, int timeoutStep)
    {
        for (int i = 1; i <= size; i++) Nodes.Add(new RaftNode(i, baseTimeout + i * timeoutStep));
    }

    public int Majority => Nodes.Count / 2 + 1;
    public RaftNode Node(int id) => Nodes[id - 1];

    public void Block(int a, int b)
    {
        blocked.Add(a + "->" + b);
        blocked.Add(b + "->" + a);
    }

    public void Heal(int a, int b)
    {
        blocked.Remove(a + "->" + b);
        blocked.Remove(b + "->" + a);
    }

    public void Partition(int[] side, int[] other)
    {
        foreach (int a in side) foreach (int b in other) Block(a, b);
    }

    public void HealPartition(int[] side, int[] other)
    {
        foreach (int a in side) foreach (int b in other) Heal(a, b);
    }

    public void Send(int from, int to, RaftMessage message)
    {
        MessagesSent++;
        if (blocked.Contains(from + "->" + to)) { MessagesDropped++; return; }
        message.From = from;
        message.To = to;
        message.Seq = seq++;
        message.DeliverTick = Now + 1;
        queue.Add(message);
    }

    public bool Propose(int nodeId, string command)
    {
        RaftNode node = Node(nodeId);
        if (node.Role != RaftRole.Leader) return false;
        node.Log.Add(new LogEntry(node.CurrentTerm, node.LastLogIndex + 1, command));
        return true;
    }

    public List<RaftNode> Leaders()
    {
        var leaders = new List<RaftNode>();
        foreach (RaftNode node in Nodes) if (node.Role == RaftRole.Leader) leaders.Add(node);
        return leaders;
    }

    public void Tick()
    {
        Now++;
        foreach (RaftNode node in Nodes)
        {
            if (node.Role == RaftRole.Leader)
            {
                foreach (RaftNode peer in Nodes) if (peer.Id != node.Id) SendAppend(node, peer);
            }
            else
            {
                node.ElectionElapsed++;
                if (node.ElectionElapsed >= node.ElectionTimeout) StartElection(node);
            }
        }
        DeliverDue();
        CheckInvariants();
    }

    public void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Tick();
    }

    private void StartElection(RaftNode node)
    {
        node.Role = RaftRole.Candidate;
        node.CurrentTerm++;
        node.VotedFor = node.Id;
        node.Grants.Clear();
        node.Grants.Add(node.Id);
        node.ElectionElapsed = 0;
        ElectionsStarted++;
        foreach (RaftNode peer in Nodes)
        {
            if (peer.Id == node.Id) continue;
            Send(node.Id, peer.Id, new RaftMessage
            {
                Kind = "vote_req",
                Term = node.CurrentTerm,
                LastLogIndex = node.LastLogIndex,
                LastLogTerm = node.LastLogTerm,
            });
        }
    }

    private void SendAppend(RaftNode leader, RaftNode peer)
    {
        int next = leader.NextIndex.TryGetValue(peer.Id, out int n) ? n : leader.LastLogIndex + 1;
        if (next < 1) next = 1;
        var message = new RaftMessage
        {
            Kind = "append",
            Term = leader.CurrentTerm,
            PrevLogIndex = next - 1,
            PrevLogTerm = leader.TermAt(next - 1),
            LeaderCommit = leader.CommitIndex,
        };
        for (int i = next; i <= leader.LastLogIndex; i++) message.Entries.Add(leader.Log[i]);
        Send(leader.Id, peer.Id, message);
    }

    private void DeliverDue()
    {
        while (true)
        {
            int best = -1;
            for (int i = 0; i < queue.Count; i++)
            {
                if (queue[i].DeliverTick > Now) continue;
                if (best < 0 || queue[i].DeliverTick < queue[best].DeliverTick
                    || (queue[i].DeliverTick == queue[best].DeliverTick && queue[i].Seq < queue[best].Seq)) best = i;
            }
            if (best < 0) return;
            RaftMessage message = queue[best];
            queue.RemoveAt(best);
            Handle(message);
        }
    }

    private void Handle(RaftMessage message)
    {
        RaftNode node = Node(message.To);
        if (message.Term > node.CurrentTerm)
        {
            node.CurrentTerm = message.Term;
            node.Role = RaftRole.Follower;
            node.VotedFor = -1;
        }
        switch (message.Kind)
        {
            case "vote_req": HandleVoteRequest(node, message); break;
            case "vote_resp": HandleVoteResponse(node, message); break;
            case "append": HandleAppend(node, message); break;
            case "append_resp": HandleAppendResponse(node, message); break;
        }
    }

    private void HandleVoteRequest(RaftNode node, RaftMessage message)
    {
        bool granted = false;
        if (message.Term == node.CurrentTerm && (node.VotedFor == -1 || node.VotedFor == message.From))
        {
            bool upToDate = message.LastLogTerm > node.LastLogTerm
                || (message.LastLogTerm == node.LastLogTerm && message.LastLogIndex >= node.LastLogIndex);
            if (upToDate)
            {
                granted = true;
                node.VotedFor = message.From;
                node.ElectionElapsed = 0;
            }
        }
        Send(node.Id, message.From, new RaftMessage { Kind = "vote_resp", Term = node.CurrentTerm, Granted = granted });
    }

    private void HandleVoteResponse(RaftNode node, RaftMessage message)
    {
        if (node.Role != RaftRole.Candidate || message.Term != node.CurrentTerm || !message.Granted) return;
        node.Grants.Add(message.From);
        if (node.Grants.Count < Majority) return;
        node.Role = RaftRole.Leader;
        node.LeaderId = node.Id;
        LeaderChanges++;
        foreach (RaftNode peer in Nodes) if (peer.Id != node.Id)
        {
            node.NextIndex[peer.Id] = node.LastLogIndex + 1;
            node.MatchIndex[peer.Id] = 0;
        }
    }

    private void HandleAppend(RaftNode node, RaftMessage message)
    {
        if (message.Term < node.CurrentTerm)
        {
            Send(node.Id, message.From, new RaftMessage { Kind = "append_resp", Term = node.CurrentTerm, Granted = false });
            return;
        }
        node.Role = RaftRole.Follower;
        node.LeaderId = message.From;
        node.ElectionElapsed = 0;
        bool consistent = message.PrevLogIndex <= node.LastLogIndex
            && node.TermAt(message.PrevLogIndex) == message.PrevLogTerm;
        if (!consistent)
        {
            Send(node.Id, message.From, new RaftMessage
            {
                Kind = "append_resp", Term = node.CurrentTerm, Granted = false, MatchIndex = 0,
            });
            return;
        }
        int index = message.PrevLogIndex;
        foreach (LogEntry entry in message.Entries)
        {
            index++;
            if (index <= node.LastLogIndex)
            {
                if (node.TermAt(index) != entry.Term)      // conflicting suffix: truncate and take the leader's
                {
                    node.Log.RemoveRange(index, node.Log.Count - index);
                    node.Log.Add(entry);
                }
            }
            else
            {
                node.Log.Add(entry);
            }
        }
        if (message.LeaderCommit > node.CommitIndex) node.CommitIndex = Math.Min(message.LeaderCommit, node.LastLogIndex);
        ApplyCommitted(node);
        Send(node.Id, message.From, new RaftMessage
        {
            Kind = "append_resp", Term = node.CurrentTerm, Granted = true, MatchIndex = node.LastLogIndex,
        });
    }

    private void HandleAppendResponse(RaftNode node, RaftMessage message)
    {
        if (node.Role != RaftRole.Leader || message.Term != node.CurrentTerm) return;
        if (message.Granted)
        {
            node.MatchIndex[message.From] = message.MatchIndex;
            node.NextIndex[message.From] = message.MatchIndex + 1;
            AdvanceCommit(node);
        }
        else
        {
            int next = node.NextIndex.TryGetValue(message.From, out int n) ? n : node.LastLogIndex + 1;
            node.NextIndex[message.From] = Math.Max(1, next - 1);
        }
    }

    /// <summary>The commit rule: highest majority-acknowledged index whose entry is from this term.</summary>
    private void AdvanceCommit(RaftNode leader)
    {
        var acked = new List<int> { leader.LastLogIndex };
        foreach (RaftNode peer in Nodes) if (peer.Id != leader.Id)
        {
            acked.Add(leader.MatchIndex.TryGetValue(peer.Id, out int m) ? m : 0);
        }
        acked.Sort();
        int majorityIndex = acked[acked.Count - Majority];
        if (majorityIndex > leader.CommitIndex && leader.TermAt(majorityIndex) == leader.CurrentTerm)
        {
            leader.CommitIndex = majorityIndex;
            ApplyCommitted(leader);
        }
    }

    private void ApplyCommitted(RaftNode node)
    {
        while (node.Applied.Count < node.CommitIndex)
        {
            int index = node.Applied.Count + 1;
            node.Applied.Add(node.Log[index].Command);
        }
    }

    /// <summary>Re-check every Raft safety property; called after each tick.</summary>
    private void CheckInvariants()
    {
        var byTerm = new Dictionary<int, int>();
        foreach (RaftNode node in Nodes)
        {
            if (node.Role != RaftRole.Leader) continue;
            if (byTerm.TryGetValue(node.CurrentTerm, out int other) && other != node.Id) LeaderPerTermViolations++;
            else byTerm[node.CurrentTerm] = node.Id;
        }

        foreach (RaftNode node in Nodes)
        {
            for (int index = 1; index <= node.CommitIndex; index++)
            {
                string command = node.CommandAt(index);
                int term = node.TermAt(index);
                if (CommittedCommand.TryGetValue(index, out string? known))
                {
                    if (known != command || CommittedTerm[index] != term) CommittedChangedViolations++;
                }
                else
                {
                    CommittedCommand[index] = command;
                    CommittedTerm[index] = term;
                }
            }
        }

        for (int i = 0; i < Nodes.Count; i++)
        {
            for (int j = i + 1; j < Nodes.Count; j++)
            {
                RaftNode a = Nodes[i], b = Nodes[j];
                int shared = Math.Min(a.LastLogIndex, b.LastLogIndex);
                for (int index = 1; index <= shared; index++)
                {
                    if (a.TermAt(index) != b.TermAt(index)) break;
                    if (a.CommandAt(index) != b.CommandAt(index)) { LogMatchingViolations++; break; }
                }
            }
        }

        foreach (RaftNode node in Nodes)
        {
            if (node.Role != RaftRole.Leader) continue;
            foreach (KeyValuePair<int, string> committed in CommittedCommand)
            {
                // A leader is only answerable for entries committed in its own term or earlier; a
                // stale leader from an older term is exempt (it will be forced to step down).
                if (node.CurrentTerm < CommittedTerm[committed.Key]) continue;
                if (committed.Key > node.LastLogIndex) { LeaderCompletenessViolations++; continue; }
                if (node.CommandAt(committed.Key) != committed.Value) LeaderCompletenessViolations++;
            }
        }
    }

    public bool LogsEqual()
    {
        RaftNode first = Nodes[0];
        foreach (RaftNode node in Nodes)
        {
            if (node.LastLogIndex != first.LastLogIndex) return false;
            for (int index = 1; index <= first.LastLogIndex; index++)
                if (node.CommandAt(index) != first.CommandAt(index) || node.TermAt(index) != first.TermAt(index)) return false;
        }
        return true;
    }

    public bool AppliedEqual()
    {
        List<string> first = Nodes[0].Applied;
        foreach (RaftNode node in Nodes)
        {
            if (node.Applied.Count != first.Count) return false;
            for (int i = 0; i < first.Count; i++) if (node.Applied[i] != first[i]) return false;
        }
        return true;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckSafe(RaftCluster cluster, string where)
    {
        Check(cluster.LeaderPerTermViolations == 0, "two leaders in one term at " + where);
        Check(cluster.LogMatchingViolations == 0, "log matching violated at " + where);
        Check(cluster.CommittedChangedViolations == 0, "a committed entry changed at " + where);
        Check(cluster.LeaderCompletenessViolations == 0, "a new leader is missing a committed entry at " + where);
    }

    private static void ElectionAndReplication()
    {
        var cluster = new RaftCluster(5, 12, 4);
        cluster.Run(30);
        List<RaftNode> leaders = cluster.Leaders();
        Check(leaders.Count == 1, "exactly one leader, got " + leaders.Count);
        Check(leaders[0].Id == 1, "the shortest timeout wins deterministically, got node " + leaders[0].Id);
        Check(leaders[0].CurrentTerm == 1, "elected in term 1, got " + leaders[0].CurrentTerm);
        Check(cluster.ElectionsStarted == 1, "only one election was needed, got " + cluster.ElectionsStarted);

        Check(cluster.Propose(1, "set a=1"), "the leader accepts a proposal");
        Check(cluster.Propose(3, "set b=2") == false, "a follower refuses to propose");
        cluster.Run(3);
        Check(cluster.Propose(1, "set c=3"), "proposal in a later tick");
        cluster.Run(20);

        foreach (RaftNode node in cluster.Nodes)
        {
            Check(node.CommitIndex == 2, "node " + node.Id + " committed both entries, got " + node.CommitIndex
                + " (log=" + node.LastLogIndex + ", applied=" + node.Applied.Count + ", role=" + node.Role + ")");
        }
        Check(cluster.LogsEqual(), "all five logs are identical");
        Check(cluster.AppliedEqual(), "all five state machines applied the same commands");
        Check(string.Join(",", cluster.Node(1).Applied) == "set a=1,set c=3", "applied in log order");
        Check(cluster.Node(1).LastLogIndex == 2, "the refused proposal never entered anyone's log");
        CheckSafe(cluster, "after replication");
    }

    private static void MinorityCannotElectOrCommit()
    {
        var cluster = new RaftCluster(5, 12, 4);
        cluster.Run(30);
        Check(cluster.Leaders()[0].Id == 1, "node 1 leads before the partition");
        cluster.Propose(1, "pre-split");
        cluster.Run(10);
        Check(cluster.Node(5).Applied.Count == 1, "everything is committed before the split");

        cluster.Partition(new[] { 1, 2 }, new[] { 3, 4, 5 });
        Check(cluster.Propose(1, "island-entry"), "the stale leader still accepts work while isolated");
        for (int i = 0; i < 60; i++)
        {
            cluster.Tick();
            CheckSafe(cluster, "tick " + i + " of the partition");
        }
        var island = new List<RaftNode>();
        for (int id = 1; id <= 2; id++) island.Add(cluster.Node(id));
        bool islandLeader = false;
        foreach (RaftNode node in island) if (node.Role == RaftRole.Leader && node.CurrentTerm > 1) islandLeader = true;
        Check(!islandLeader, "the two-node island never elects a new leader");
        RaftNode majorityLeader = cluster.Node(3);
        Check(majorityLeader.Role == RaftRole.Leader, "the majority side elected a leader");
        Check(majorityLeader.CurrentTerm >= 2, "in a newer term, got " + majorityLeader.CurrentTerm);
        Check(cluster.Propose(3, "majority-1"), "the majority leader accepts work");
        cluster.Run(10);
        Check(cluster.Node(3).CommitIndex == 2 && cluster.Node(4).CommitIndex == 2 && cluster.Node(5).CommitIndex == 2,
            "the majority commits the entry");
        Check(cluster.Node(1).CommitIndex == 1, "the island cannot commit anything new");
        Check(cluster.Node(1).LastLogIndex == 2, "the stale leader appended locally, index=" + cluster.Node(1).LastLogIndex);
        CheckSafe(cluster, "majority commit during the partition");
        Console.WriteLine("partition: island-term=" + cluster.Node(1).CurrentTerm + " majority-term=" + majorityLeader.CurrentTerm
            + " dropped=" + cluster.MessagesDropped);
    }

    private static void HealingRestoresASingleLeader()
    {
        var cluster = new RaftCluster(5, 12, 4);
        cluster.Run(30);
        cluster.Propose(1, "before");
        cluster.Run(10);
        cluster.Partition(new[] { 1, 2 }, new[] { 3, 4, 5 });
        cluster.Run(40);
        cluster.Propose(1, "stale-entry");      // appended on the island, never committed
        cluster.Run(5);
        cluster.Propose(3, "majority-entry");
        cluster.Run(10);
        int committedBeforeHeal = cluster.Node(3).CommitIndex;
        Check(cluster.Node(1).CommitIndex == 1, "the island's entry is uncommitted before the heal");

        cluster.HealPartition(new[] { 1, 2 }, new[] { 3, 4, 5 });
        cluster.Run(60);

        List<RaftNode> leaders = cluster.Leaders();
        Check(leaders.Count == 1, "one leader after healing, got " + leaders.Count);
        Check(cluster.Node(1).Role != RaftRole.Leader, "the stale leader stepped down");
        Check(cluster.LogsEqual(), "all logs converged");
        Check(cluster.AppliedEqual(), "all state machines converged");
        Check(cluster.Node(1).CommitIndex >= committedBeforeHeal, "the healed nodes caught up to the committed prefix");
        Check(cluster.Node(1).Applied.Contains("majority-entry"), "the majority's entry survived the heal");
        foreach (RaftNode node in cluster.Nodes)
            Check(node.Applied[0] == "before", "the originally committed entry is still first everywhere");
        CheckSafe(cluster, "after healing");
    }

    private static void RandomPartitionsKeepSafety()
    {
        var cluster = new RaftCluster(5, 12, 4);
        cluster.Run(25);
        int seed = 808080;
        int committedAtStart = 0;
        for (int round = 0; round < 40; round++)
        {
            int[] side = (round % 3 == 0) ? new[] { 1, 2 } : (round % 2 == 0 ? new[] { 4, 5 } : new[] { 1, 2, 3 });
            var other = new List<int>();
            for (int id = 1; id <= 5; id++) if (Array.IndexOf(side, id) < 0) other.Add(id);
            cluster.Partition(side, other.ToArray());
            cluster.Run(6);
            foreach (RaftNode node in cluster.Nodes)
                if (node.Role == RaftRole.Leader)
                {
                    seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                    cluster.Propose(node.Id, "r" + round + "-" + (seed % 97));
                    committedAtStart = Math.Max(committedAtStart, node.CommitIndex);
                }
            cluster.Run(6);
            cluster.HealPartition(side, other.ToArray());
            cluster.Run(10);
        }
        cluster.Run(120);
        CheckSafe(cluster, "after the random partition schedule");
        Check(cluster.Leaders().Count == 1, "exactly one leader once every link is healthy");
        Check(cluster.LogsEqual(), "logs converge after the random schedule");
        Check(cluster.AppliedEqual(), "state machines converge after the random schedule");
        Check(cluster.Node(1).CommitIndex >= committedAtStart, "the committed prefix never shrinks");
        Check(cluster.CommittedChangedViolations == 0 && cluster.LeaderPerTermViolations == 0,
            "no safety violation across " + cluster.ElectionsStarted + " elections");
        Console.WriteLine("random schedule: elections=" + cluster.ElectionsStarted + " leader-changes=" + cluster.LeaderChanges
            + " messages=" + cluster.MessagesSent + " dropped=" + cluster.MessagesDropped
            + " committed=" + cluster.CommittedCommand.Count);
    }

    private static void CommittedPrefixSurvivesReelections()
    {
        var cluster = new RaftCluster(3, 10, 6);
        cluster.Run(20);
        Check(cluster.Propose(1, "c1"), "leader proposes c1");
        cluster.Run(10);
        Check(cluster.Node(2).Applied.Count == 1, "c1 is committed");
        // Isolate the leader: the two remaining nodes form a majority and must elect a new leader
        // whose log already contains c1 (leader completeness).
        cluster.Partition(new[] { 1 }, new[] { 2, 3 });
        cluster.Run(40);
        Check(cluster.Node(2).Role == RaftRole.Leader || cluster.Node(3).Role == RaftRole.Leader, "a new leader took over");
        Check(cluster.Node(1).CurrentTerm < cluster.Node(2).CurrentTerm, "the isolated leader is behind in terms");
        RaftNode newLeader = cluster.Node(2).Role == RaftRole.Leader ? cluster.Node(2) : cluster.Node(3);
        Check(newLeader.CommandAt(1) == "c1", "the new leader carries the committed entry");
        Check(newLeader.Applied.Count == 1 && newLeader.Applied[0] == "c1", "the new leader applied it");
        cluster.Propose(newLeader.Id, "c2");
        cluster.Run(10);
        Check(cluster.Node(2).Applied.Count == 2 && cluster.Node(3).Applied.Count == 2, "the new term commits c2");
        Check(cluster.Node(1).Applied.Count == 1, "the isolated leader cannot commit c2");
        cluster.HealPartition(new[] { 1 }, new[] { 2, 3 });
        cluster.Run(60);
        Check(cluster.LogsEqual() && cluster.AppliedEqual(), "the isolated node converges after healing");
        Check(string.Join(",", cluster.Node(1).Applied) == "c1,c2", "and it applied both entries in order");
        CheckSafe(cluster, "after the isolated leader rejoined");
    }

    public static void Main()
    {
        ElectionAndReplication();
        MinorityCannotElectOrCommit();
        HealingRestoresASingleLeader();
        RandomPartitionsKeepSafety();
        CommittedPrefixSurvivesReelections();
        Console.WriteLine("PASS");
    }
}
