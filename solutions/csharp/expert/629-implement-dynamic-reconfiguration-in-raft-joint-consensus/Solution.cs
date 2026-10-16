// Task 629 - Implement dynamic reconfiguration in Raft (joint consensus). (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// A configuration is a list of quorum GROUPS. To be a quorum a set of servers must contain a
// majority of every group, so
//   * C_old         = one group (the old membership),
//   * C_new         = one group (the new membership),
//   * C_old,new     = two groups, i.e. a majority of the old AND a majority of the new.
// The transition is driven through the replicated log: the leader appends the joint configuration
// C_old,new, and because a quorum now needs both majorities, any election or commit during the
// transition must overlap the old and the new memberships - that overlap is exactly what makes
// the reconfiguration safe. Once C_old,new is committed the leader appends C_new; C_new is
// committed by a majority of the new configuration alone, and a leader that is no longer a voter
// steps down only after C_new is committed (it keeps leading through its own removal).
//
// Every node evaluates quorums against the LATEST configuration in its own log (the Raft rule,
// even before that entry is committed), which is what keeps the two phases from overlapping.
//
// What is simplified (documented per the authoring contract): deterministic in-memory network
// driven by a logical tick, one-tick message latency, staggered deterministic election timeouts
// instead of randomized ones, no snapshots (a new member is caught up entry by entry through the
// normal AppendEntries back-off), and the "leader steps down" rule is applied on commit of C_new
// rather than on append. Verified invariants: any two quorums intersect under every configuration
// (enumerated exhaustively for small configs), at most one leader per term, log matching,
// committed entries never change, and no leader exists outside its configuration's voter set.
using System;
using System.Collections.Generic;

#nullable enable

internal sealed class Config
{
    public readonly string Label;
    public readonly List<int[]> Groups = new List<int[]>();

    private Config(string label) { Label = label; }

    public static Config Single(params int[] voters) => new Config("C{" + Join(voters) + "}") { Groups = { voters } };

    public static Config Joint(Config oldConfig, Config newConfig)
    {
        var config = new Config(oldConfig.Label + "+" + newConfig.Label);
        foreach (int[] group in oldConfig.Groups) config.Groups.Add(group);
        foreach (int[] group in newConfig.Groups) config.Groups.Add(group);
        return config;
    }

    public bool IsJoint => Groups.Count > 1;

    public HashSet<int> VoterIds()
    {
        var ids = new HashSet<int>();
        foreach (int[] group in Groups) foreach (int id in group) ids.Add(id);
        return ids;
    }

    public bool IsVoter(int id) => VoterIds().Contains(id);

    /// <summary>A quorum needs a majority inside every group - both majorities of a joint config.</summary>
    public bool IsQuorum(IEnumerable<int> voters)
    {
        var set = voters as HashSet<int> ?? new HashSet<int>(voters);
        foreach (int[] group in Groups)
        {
            int have = 0;
            foreach (int id in group) if (set.Contains(id)) have++;
            if (have * 2 <= group.Length) return false;
        }
        return Groups.Count > 0;
    }

    public string Describe()
    {
        var parts = new List<string>();
        foreach (int[] group in Groups) parts.Add("{" + Join(group) + "}");
        return string.Join(" & ", parts);
    }

    private static string Join(int[] ids)
    {
        var parts = new List<string>();
        foreach (int id in ids) parts.Add(id.ToString());
        return string.Join(",", parts);
    }
}

internal enum RaftRole { Follower, Candidate, Leader }

internal sealed class LogEntry
{
    public int Term;
    public int Index;
    public string Command = "";
    public Config? Configuration;

    public LogEntry(int term, int index, string command) { Term = term; Index = index; Command = command; }
    public bool IsConfig => Configuration != null;
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
    public int StepDowns;

    public RaftNode(int id, int timeout) { Id = id; ElectionTimeout = timeout; }

    public int LastLogIndex => Log.Count - 1;
    public int LastLogTerm => Log[Log.Count - 1].Term;
    public int TermAt(int index) => index <= LastLogIndex ? Log[index].Term : 0;
    public string CommandAt(int index) => Log[index].Command;
    public bool IsConfigAt(int index) => Log[index].IsConfig;

    /// <summary>The latest configuration anywhere in the log wins, committed or not.</summary>
    public Config ActiveConfig(Config initial)
    {
        for (int index = LastLogIndex; index >= 0; index--) if (Log[index].IsConfig) return Log[index].Configuration!;
        return initial;
    }

    /// <summary>The latest COMMITTED configuration - what a departing leader steps down on.</summary>
    public Config CommittedConfig(Config initial)
    {
        for (int index = Math.Min(CommitIndex, LastLogIndex); index >= 0; index--)
            if (Log[index].IsConfig) return Log[index].Configuration!;
        return initial;
    }
}

internal sealed class RaftCluster
{
    public readonly Config InitialConfig;
    public readonly List<RaftNode> Nodes = new List<RaftNode>();
    private readonly List<RaftMessage> queue = new List<RaftMessage>();
    private readonly HashSet<string> blocked = new HashSet<string>();
    private int seq;

    public int Now;
    public int MessagesSent;
    public int MessagesDropped;
    public int ElectionsStarted;
    public int LeaderChanges;
    public int StepDowns;
    public int LeaderPerTermViolations;
    public int LogMatchingViolations;
    public int CommittedChangedViolations;
    public int LeaderOutsideConfigViolations;
    public int JointCommitChecks;
    public readonly Dictionary<int, string> CommittedCommand = new Dictionary<int, string>();
    public readonly Dictionary<int, int> CommittedTerm = new Dictionary<int, int>();

    public RaftCluster(Config initial, int baseTimeout, int timeoutStep)
    {
        InitialConfig = initial;
        foreach (int id in Sorted(initial.VoterIds())) Nodes.Add(new RaftNode(id, baseTimeout + id * timeoutStep));
    }

    /// <summary>Add a node object for a member that is not part of the initial configuration yet.</summary>
    public void AddNode(int id, int baseTimeout, int timeoutStep)
    {
        Nodes.Add(new RaftNode(id, baseTimeout + id * timeoutStep));
        Nodes.Sort((a, b) => a.Id.CompareTo(b.Id));
    }

    public int MajorityOld => InitialConfig.Groups[0].Length / 2 + 1;
    public RaftNode Node(int id) => Nodes[id - 1];

    private static List<int> Sorted(IEnumerable<int> ids)
    {
        var list = new List<int>(ids);
        list.Sort();
        return list;
    }

    public void Block(int a, int b) { blocked.Add(a + "->" + b); blocked.Add(b + "->" + a); }
    public void Heal(int a, int b) { blocked.Remove(a + "->" + b); blocked.Remove(b + "->" + a); }

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
        Config config = node.ActiveConfig(InitialConfig);
        if (node.Role != RaftRole.Leader || !config.IsVoter(nodeId)) return false;
        node.Log.Add(new LogEntry(node.CurrentTerm, node.LastLogIndex + 1, command));
        return true;
    }

    /// <summary>The leader appends a configuration entry, exactly like any other log entry.</summary>
    public bool AppendConfig(int nodeId, Config config)
    {
        RaftNode node = Node(nodeId);
        if (node.Role != RaftRole.Leader) return false;
        node.Log.Add(new LogEntry(node.CurrentTerm, node.LastLogIndex + 1, "config:" + config.Label)
        {
            Configuration = config,
        });
        return true;
    }

    public List<RaftNode> Leaders()
    {
        var leaders = new List<RaftNode>();
        foreach (RaftNode node in Nodes) if (node.Role == RaftRole.Leader) leaders.Add(node);
        return leaders;
    }

    /// <summary>The configuration each node currently believes in.</summary>
    public Dictionary<int, string> ActiveConfigs()
    {
        var map = new Dictionary<int, string>();
        foreach (RaftNode node in Nodes) map[node.Id] = node.ActiveConfig(InitialConfig).Describe();
        return map;
    }

    public void Tick()
    {
        Now++;
        foreach (RaftNode node in Nodes)
        {
            if (node.Role == RaftRole.Leader)
            {
                foreach (RaftNode peer in Nodes) if (peer.Id != node.Id) SendAppend(node, peer);
                MaybeStepDown(node);
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

    /// <summary>A leader outside the committed configuration steps down once C_new is committed.</summary>
    private void MaybeStepDown(RaftNode leader)
    {
        Config committed = leader.CommittedConfig(InitialConfig);
        if (committed.IsVoter(leader.Id)) return;
        leader.Role = RaftRole.Follower;
        leader.LeaderId = -1;
        leader.StepDowns++;
        StepDowns++;
    }

    private void StartElection(RaftNode node)
    {
        Config config = node.ActiveConfig(InitialConfig);
        if (!config.IsVoter(node.Id)) { node.ElectionElapsed = 0; return; }   // a non-voter never campaigns
        node.Role = RaftRole.Candidate;
        node.CurrentTerm++;
        node.VotedFor = node.Id;
        node.Grants.Clear();
        node.Grants.Add(node.Id);
        node.ElectionElapsed = 0;
        ElectionsStarted++;
        foreach (int voter in config.VoterIds())
        {
            if (voter == node.Id) continue;
            Send(node.Id, voter, new RaftMessage
            {
                Kind = "vote_req", Term = node.CurrentTerm,
                LastLogIndex = node.LastLogIndex, LastLogTerm = node.LastLogTerm,
            });
        }
    }

    private void SendAppend(RaftNode leader, RaftNode peer)
    {
        int next = leader.NextIndex.TryGetValue(peer.Id, out int n) ? n : leader.LastLogIndex + 1;
        if (next < 1) next = 1;
        var message = new RaftMessage
        {
            Kind = "append", Term = leader.CurrentTerm,
            PrevLogIndex = next - 1, PrevLogTerm = leader.TermAt(next - 1),
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
            if (upToDate) { granted = true; node.VotedFor = message.From; node.ElectionElapsed = 0; }
        }
        Send(node.Id, message.From, new RaftMessage { Kind = "vote_resp", Term = node.CurrentTerm, Granted = granted });
    }

    private void HandleVoteResponse(RaftNode node, RaftMessage message)
    {
        if (node.Role != RaftRole.Candidate || message.Term != node.CurrentTerm || !message.Granted) return;
        node.Grants.Add(message.From);
        Config config = node.ActiveConfig(InitialConfig);
        if (!config.IsQuorum(node.Grants)) return;      // joint configs need both majorities here
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
        if (message.PrevLogIndex > node.LastLogIndex || node.TermAt(message.PrevLogIndex) != message.PrevLogTerm)
        {
            Send(node.Id, message.From, new RaftMessage { Kind = "append_resp", Term = node.CurrentTerm, Granted = false, MatchIndex = 0 });
            return;
        }
        int index = message.PrevLogIndex;
        foreach (LogEntry entry in message.Entries)
        {
            index++;
            if (index <= node.LastLogIndex)
            {
                if (node.TermAt(index) != entry.Term)
                {
                    node.Log.RemoveRange(index, node.Log.Count - index);
                    node.Log.Add(entry);
                }
            }
            else node.Log.Add(entry);
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
            MaybeStepDown(node);        // C_new may just have committed this leader out of the configuration
        }
        else
        {
            int next = node.NextIndex.TryGetValue(message.From, out int n) ? n : node.LastLogIndex + 1;
            node.NextIndex[message.From] = Math.Max(1, next - 1);
        }
    }

    private void AdvanceCommit(RaftNode leader)
    {
        Config config = leader.ActiveConfig(InitialConfig);
        if (config.IsJoint) JointCommitChecks++;
        for (int index = leader.LastLogIndex; index > leader.CommitIndex; index--)
        {
            if (leader.TermAt(index) != leader.CurrentTerm) break;     // older terms commit indirectly
            var voters = new HashSet<int>();
            if (config.IsVoter(leader.Id)) voters.Add(leader.Id);
            foreach (RaftNode peer in Nodes)
            {
                if (peer.Id == leader.Id) continue;
                if ((leader.MatchIndex.TryGetValue(peer.Id, out int m) ? m : 0) >= index) voters.Add(peer.Id);
            }
            if (!config.IsQuorum(voters)) continue;
            leader.CommitIndex = index;
            ApplyCommitted(leader);
            break;
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

    private void CheckInvariants()
    {
        var byTerm = new Dictionary<int, int>();
        foreach (RaftNode node in Nodes)
        {
            if (node.Role != RaftRole.Leader) continue;
            Config committed = node.CommittedConfig(InitialConfig);
            if (!committed.IsVoter(node.Id)) LeaderOutsideConfigViolations++;
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
        Check(cluster.LeaderOutsideConfigViolations == 0, "a leader outside its configuration at " + where);
    }

    private static IEnumerable<int[]> Subsets(int[] ids)
    {
        int total = 1 << ids.Length;
        for (int mask = 1; mask < total; mask++)
        {
            var subset = new List<int>();
            for (int bit = 0; bit < ids.Length; bit++) if ((mask & (1 << bit)) != 0) subset.Add(ids[bit]);
            yield return subset.ToArray();
        }
    }

    /// <summary>Exhaustive quorum-intersection proof for the small configurations.</summary>
    private static void ExhaustiveQuorumOverlap()
    {
        var three = new[] { 1, 2, 3 };
        var five = new[] { 1, 2, 3, 4, 5 };
        var disjoint = new[] { 3, 4, 5 };
        var configs = new List<(Config Config, int[] Pool)>
        {
            (Config.Single(1, 2, 3), three),
            (Config.Single(1, 2, 3, 4, 5), five),
            (Config.Joint(Config.Single(1, 2, 3), Config.Single(1, 2, 3, 4, 5)), five),
            (Config.Joint(Config.Single(1, 2, 3), Config.Single(3, 4, 5)), new[] { 1, 2, 3, 4, 5 }),
        };

        foreach ((Config config, int[] pool) in configs)
        {
            var quorums = new List<int[]>();
            foreach (int[] subset in Subsets(pool)) if (config.IsQuorum(subset)) quorums.Add(subset);
            Check(quorums.Count > 0, "the configuration " + config.Describe() + " has quorums");
            for (int i = 0; i < quorums.Count; i++)
            {
                for (int j = i + 1; j < quorums.Count; j++)
                {
                    bool overlaps = false;
                    foreach (int id in quorums[i]) if (Array.IndexOf(quorums[j], id) >= 0) overlaps = true;
                    Check(overlaps, "quorums {" + string.Join(",", quorums[i]) + "} and {" + string.Join(",", quorums[j])
                        + "} of " + config.Describe() + " are disjoint");
                }
            }
        }

        Config joint = Config.Joint(Config.Single(1, 2, 3), Config.Single(3, 4, 5));
        Check(!joint.IsQuorum(new[] { 1, 2 }), "{1,2} is not a joint quorum: it misses the new majority");
        Check(!joint.IsQuorum(new[] { 4, 5 }), "{4,5} is not a joint quorum: it misses the old majority");
        Check(joint.IsQuorum(new[] { 1, 2, 3, 4 }), "{1,2,3,4} is a joint quorum here: old 3/3, new 2/3");
        Check(joint.IsQuorum(new[] { 2, 3, 4 }), "{2,3,4} is a quorum too, and the two overlap in {2,3,4}");
        Check(Config.Single(3, 4, 5).IsQuorum(new[] { 3, 4 }), "a majority of C_new is a quorum of C_new");
        Check(!Config.Single().IsQuorum(new[] { 1 }), "an empty configuration has no quorum");
        Check(Config.Single(7).IsQuorum(new[] { 7 }), "a single-node configuration is its own quorum");
        Console.WriteLine("quorum overlap: exhaustive check over " + configs.Count + " configurations");
    }

    private static void JointConsensusGatesProgress()
    {
        var oldConfig = Config.Single(1, 2, 3);
        var cluster = new RaftCluster(oldConfig, 10, 4);
        cluster.AddNode(4, 10, 4);
        cluster.AddNode(5, 10, 4);
        cluster.Run(25);
        Check(cluster.Leaders().Count == 1 && cluster.Leaders()[0].Id == 1, "node 1 leads the old configuration");
        Check(cluster.Propose(1, "before-reconfig"), "a proposal before the change");
        cluster.Run(10);
        Check(cluster.Node(2).Applied.Count == 1, "the entry is committed in C_old");

        // New members 4 and 5 are unreachable, so only {1,2,3} can respond.
        cluster.Partition(new[] { 1, 2 }, new[] { 4, 5 });
        var joint = Config.Joint(oldConfig, Config.Single(3, 4, 5));
        Check(cluster.AppendConfig(1, joint), "the leader appends the joint configuration");
        cluster.Run(20);
        Check(cluster.Node(1).CommitIndex == 1, "the joint configuration cannot commit without the new majority");
        Check(cluster.Node(1).IsConfigAt(2), "it sits uncommitted at index 2");
        Check(cluster.Node(1).ActiveConfig(oldConfig).IsJoint, "the active configuration is already joint");
        Check(cluster.Propose(1, "blocked"), "the joint leader still accepts work");
        cluster.Run(10);
        Check(cluster.Node(1).CommitIndex == 1, "no command commits while the joint quorum is unavailable");
        CheckSafe(cluster, "joint phase without the new majority");

        cluster.HealPartition(new[] { 1, 2 }, new[] { 4, 5 });
        cluster.Run(30);
        Check(cluster.Node(1).CommitIndex >= 2, "the joint configuration commits once the new majority answers");
        Check(cluster.Node(4).LastLogIndex >= 2, "the new members were caught up");
        Check(cluster.Node(4).ActiveConfig(oldConfig).IsJoint, "and they adopted the joint configuration");

        var finalConfig = Config.Single(3, 4, 5);
        Check(cluster.AppendConfig(1, finalConfig), "the leader appends C_new");
        cluster.Run(40);
        Check(cluster.Node(1).Role != RaftRole.Leader, "the removed leader stepped down");
        Check(cluster.Node(1).StepDowns == 1, "exactly one step-down");
        Check(cluster.Node(3).CommitIndex >= 3, "the new configuration commits its own entries");
        List<RaftNode> leaders = cluster.Leaders();
        Check(leaders.Count == 1, "one leader after the reconfiguration, got " + leaders.Count);
        Check(leaders[0].Id == 3 || leaders[0].Id == 4 || leaders[0].Id == 5,
            "the new leader is a member of C_new, got " + leaders[0].Id);
        foreach (int id in new[] { 1, 2 })
        {
            Check(!cluster.Node(id).ActiveConfig(oldConfig).IsVoter(id), "node " + id + " knows it is not a voter");
            Check(cluster.Node(id).Role != RaftRole.Leader, "node " + id + " never leads again");
        }
        Check(cluster.Node(3).ActiveConfig(oldConfig).Describe() == finalConfig.Describe(), "C_new is active on node 3");
        Check(cluster.Node(5).ActiveConfig(oldConfig).Describe() == finalConfig.Describe(), "C_new is active on node 5");
        CheckSafe(cluster, "after the full reconfiguration");
        Check(cluster.JointCommitChecks > 0, "commit decisions were evaluated against the joint configuration");
        Console.WriteLine("joint consensus: step-downs=" + cluster.StepDowns + " elections=" + cluster.ElectionsStarted
            + " joint-commit-checks=" + cluster.JointCommitChecks + " dropped=" + cluster.MessagesDropped);
    }

    private static void GrowingConfigurationKeepsQuorum()
    {
        var oldConfig = Config.Single(1, 2, 3);
        var cluster = new RaftCluster(oldConfig, 10, 3);
        cluster.AddNode(4, 10, 3);
        cluster.Run(20);
        Check(cluster.Leaders()[0].Id == 1, "node 1 leads");
        Check(cluster.Propose(1, "a1"), "commit a1");
        cluster.Run(8);

        Check(cluster.AppendConfig(1, Config.Joint(oldConfig, Config.Single(1, 2, 3, 4))), "append C_old,new");
        cluster.Run(20);
        Check(cluster.Node(4).LastLogIndex >= 2, "node 4 caught up through the back-off");
        Check(cluster.Node(1).CommitIndex >= 2, "the joint configuration is committed");
        Check(cluster.AppendConfig(1, Config.Single(1, 2, 3, 4)), "append C_new");
        cluster.Run(20);
        Check(cluster.Node(1).CommitIndex >= 3, "C_new is committed");

        Check(cluster.Propose(1, "a2"), "the new configuration still accepts work");
        cluster.Run(10);
        foreach (RaftNode node in cluster.Nodes) Check(node.Applied.Count >= 4, "node " + node.Id + " applied everything");

        // Now the majority really is 3 of 4. Isolating one node must not stop the other three.
        int committedBefore = cluster.Node(1).CommitIndex;
        cluster.Partition(new[] { 4 }, new[] { 1, 2, 3 });
        Check(cluster.Propose(1, "after-split"), "the 3-node side keeps leading");
        cluster.Run(15);
        Check(cluster.Node(1).CommitIndex > committedBefore, "three of four is a majority of the new configuration");
        Check(cluster.Node(4).CommitIndex == committedBefore, "the isolated member commits nothing");
        Check(cluster.Node(1).Applied.Count > cluster.Node(4).Applied.Count, "the isolated member lagged behind");
        CheckSafe(cluster, "3/1 partition after the reconfiguration");

        // A 2/2 split has no majority at all: neither side may commit, and the stale leader stays put.
        cluster.HealPartition(new[] { 4 }, new[] { 1, 2, 3 });
        cluster.Run(15);
        int committedAfterHeal = cluster.Node(4).CommitIndex;
        cluster.Partition(new[] { 1, 2 }, new[] { 3, 4 });
        cluster.Run(50);
        Check(cluster.Node(1).CommitIndex == committedAfterHeal, "a 2-node island cannot commit in a 4-node configuration");
        Check(cluster.Node(3).CommitIndex == committedAfterHeal, "and neither can the other side");
        Check(cluster.Leaders().Count == 1, "only the stale leader remains, got " + cluster.Leaders().Count);
        Check(cluster.Node(3).Role != RaftRole.Leader && cluster.Node(4).Role != RaftRole.Leader,
            "the minority side never elects");
        CheckSafe(cluster, "2/2 split");

        cluster.HealPartition(new[] { 1, 2 }, new[] { 3, 4 });
        cluster.Run(80);
        Check(cluster.Leaders().Count == 1, "converged back to one leader");
        Check(cluster.Node(1).Applied.Count == cluster.Node(3).Applied.Count, "all members converged on the log");
        CheckSafe(cluster, "after healing");
    }

    private static void RandomReconfigurationsStaySafe()
    {
        var oldConfig = Config.Single(1, 2, 3);
        var cluster = new RaftCluster(oldConfig, 10, 3);
        cluster.AddNode(4, 10, 3);
        cluster.Run(20);
        int seed = 24680;
        var steps = new[]
        {
            "joint-old-new", "command", "joint-wider", "command", "final-new", "command",
        };
        foreach (string step in steps)
        {
            var leaders = cluster.Leaders();
            if (leaders.Count == 0) { cluster.Run(20); leaders = cluster.Leaders(); }
            if (leaders.Count > 0)
            {
                RaftNode leader = leaders[0];
                Config current = leader.ActiveConfig(oldConfig);
                if (step == "command") cluster.Propose(leader.Id, "cmd-" + leader.Id);
                else if (step == "joint-old-new") cluster.AppendConfig(leader.Id, Config.Joint(current, Config.Single(1, 2, 3, 4)));
                else if (step == "joint-wider") cluster.AppendConfig(leader.Id, Config.Joint(current, Config.Single(2, 3, 4)));
                else cluster.AppendConfig(leader.Id, Config.Single(2, 3, 4));
            }
            for (int i = 0; i < 15; i++)
            {
                cluster.Tick();
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                if (seed % 11 == 0) cluster.Block(1, 4);
                if (seed % 13 == 0) cluster.Heal(1, 4);
                CheckSafe(cluster, "random step " + step + " tick " + i);
            }
        }
        cluster.Heal(1, 4);
        cluster.Run(150);
        CheckSafe(cluster, "after the random reconfiguration schedule");
        Check(cluster.Leaders().Count <= 1, "at most one leader survives");
        Check(cluster.MessagesSent > 0 && cluster.MessagesDropped > 0, "the schedule exercised partitions");
        Console.WriteLine("random reconfig: leader-changes=" + cluster.LeaderChanges + " messages=" + cluster.MessagesSent
            + " dropped=" + cluster.MessagesDropped + " committed=" + cluster.CommittedCommand.Count);
    }

    public static void Main()
    {
        ExhaustiveQuorumOverlap();
        JointConsensusGatesProgress();
        GrowingConfigurationKeepsQuorum();
        RandomReconfigurationsStaySafe();
        Console.WriteLine("PASS");
    }
}
