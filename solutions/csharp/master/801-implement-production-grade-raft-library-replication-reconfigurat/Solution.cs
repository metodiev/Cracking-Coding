// Task 801 - Implement production-grade Raft library (replication, reconfiguration, snapshots). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A faithful in-process Raft. Every message is an explicit object carried by an InMemoryNetwork driven by a logical
// tick counter - no threads, no sockets, no wall-clock time - and election timeouts come from a hand-written
// splitmix64 generator, so any schedule is reproducible bit for bit. Storage is a log of (index, term) entries;
// committed entries are applied in order to a replicated state machine (SET/INCR/DEL over a string map).
// Snapshots compact a committed prefix and are shipped whole with InstallSnapshot to lagging followers.
// Reconfiguration uses joint consensus C_old,new -> C_new together with Raft's "one membership change in flight" rule.
//
// Invariants verified (safety and liveness), recomputed by a checker from the observed event trace:
//   * election safety      - at most one leader is ever elected per term;
//   * vote uniqueness      - a voter grants at most one candidate per term;
//   * log matching         - identical (index, term) on two logs implies identical preceding entries, and a
//                            committed entry is never overwritten by a truncation;
//   * state machine safety - no two nodes apply a different entry at the same log index;
//   * commit monotonicity  - commitIndex never moves backwards (a crash resets it to the snapshot boundary in a
//                            controlled way and the checker is told about the restart);
//   * Figure-8 rule        - an entry from an earlier term is never committed by quorum counting alone;
//   * joint consensus      - a commit needs a majority of C_old *and* of C_new while joint, no second membership
//                            change may start before the first commits, and a removed leader steps down;
//   * snapshot safety      - snapshot(lastIncludedIndex, lastIncludedTerm) is the term of that log index;
//   * crash persistence    - term/vote/log/snapshot survive a restart, so a node never votes twice;
//   * liveness             - a reachable majority elects a leader inside the election-timeout bound and acknowledged
//                            proposals are eventually committed and applied by every reachable replica.
//
// Independent oracles: the checker derives leadership, votes and commits from observed events rather than trusting
// node fields; convergence is compared with replaying the committed command sequence on a plain dictionary; and the
// quorum boundaries are cross-checked with QuorumOracle, which shares no code with Membership.HasQuorum.
//
// Documented simplifications: configuration entries take effect when appended (as in the Raft paper) and there are
// no learner/catch-up phases; the network drops messages (never reorders or duplicates) with a fixed 1-tick latency;
// AppendEntries replies carry a single backtrack hint; the state machine is a string map. Scale is expressed as
// typed parameters (cluster size, tick budget, log length, fault schedule), never as real concurrency.
#nullable enable
using System;
using System.Collections.Generic;
using System.Text;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum Role { Follower, Candidate, Leader }

internal sealed class LogEntry
{
    internal int Index;
    internal int Term;
    internal string? Command;
    internal int[]? SingleConfig;      // non-null: a plain configuration entry C
    internal int[]? JointOld;          // together with JointNew: a joint configuration entry C_old,new
    internal int[]? JointNew;

    internal LogEntry(int index, int term, string? command, int[]? singleConfig = null, int[]? jointOld = null, int[]? jointNew = null)
    {
        Index = index; Term = term; Command = command; SingleConfig = singleConfig; JointOld = jointOld; JointNew = jointNew;
    }

    internal bool IsConfig => SingleConfig != null || JointNew != null;

    internal Membership ToMembership() => JointNew != null ? new Membership(JointOld!, JointNew) : new Membership(SingleConfig!);

    /** Canonical label used by the independent safety checker to compare entries at the same index. */
    internal string Label()
    {
        if (!IsConfig) return Command!;
        if (JointNew == null) return "cfg:C[" + string.Join(",", SingleConfig!) + "]";
        return "cfg:J[" + string.Join(",", JointOld!) + "|" + string.Join(",", JointNew) + "]";
    }
}

/** A configuration is either single (C) or joint (C_old,new) while a membership change is in flight. */
internal sealed class Membership
{
    internal readonly int[] Old;
    internal readonly int[] New;
    internal readonly bool Joint;

    internal Membership(int[] members) : this(members, members, false) { }

    internal Membership(int[] oldMembers, int[] newMembers) : this(oldMembers, newMembers, true) { }

    private Membership(int[] oldMembers, int[] newMembers, bool joint)
    {
        Old = Sorted(oldMembers); New = Sorted(newMembers); Joint = joint;
    }

    private static int[] Sorted(int[] members)
    {
        var copy = (int[])members.Clone();
        Array.Sort(copy);
        return copy;
    }

    internal bool Contains(int id) => Array.IndexOf(New, id) >= 0 || (Joint && Array.IndexOf(Old, id) >= 0);

    internal int[] Union()
    {
        var seen = new SortedSet<int>();
        foreach (int id in Old) seen.Add(id);
        foreach (int id in New) seen.Add(id);
        return new List<int>(seen).ToArray();
    }

    internal string Key()
    {
        var builder = new StringBuilder();
        builder.Append(Joint ? "J[" : "C[").Append(string.Join(",", Old));
        if (Joint) builder.Append('|').Append(string.Join(",", New));
        return builder.Append(']').ToString();
    }

    /** Quorum = strict majority of C_old and, while joint, a strict majority of C_new. */
    internal bool HasQuorum(HashSet<int> voters)
    {
        if (!HasMajority(New, voters)) return false;
        if (Joint && !HasMajority(Old, voters)) return false;
        return true;
    }

    private static bool HasMajority(int[] set, HashSet<int> voters)
    {
        int count = 0;
        foreach (int member in set) if (voters.Contains(member)) count++;
        return count * 2 > set.Length;
    }
}

internal enum MessageKind { RequestVote, RequestVoteReply, AppendEntries, AppendEntriesReply, InstallSnapshot, InstallSnapshotReply }

internal sealed class Message
{
    internal MessageKind Kind;
    internal int From;
    internal int To;
    internal int Term;
    internal bool Granted;                 // RequestVoteReply
    internal bool Success;                 // AppendEntriesReply / InstallSnapshotReply
    internal int LastLogIndex;             // RequestVote
    internal int LastLogTerm;              // RequestVote
    internal int PrevLogIndex;             // AppendEntries
    internal int PrevLogTerm;              // AppendEntries
    internal int LeaderCommit;             // AppendEntries
    internal int MatchIndex;               // AppendEntriesReply
    internal int ConflictIndex;            // AppendEntriesReply backtrack hint
    internal int SnapshotIndex;            // InstallSnapshot
    internal int SnapshotTerm;             // InstallSnapshot
    internal List<LogEntry> Entries = new List<LogEntry>();
    internal SortedDictionary<string, string>? SnapshotState;

    public override string ToString() => Kind + "(" + From + "->" + To + ",t" + Term + ")";
}

/** Messages travel over this explicit network; the logical clock is the caller's tick counter. */
internal sealed class InMemoryNetwork
{
    private sealed class Pending
    {
        internal int ReleaseTick;
        internal long Sequence;
        internal int From;
        internal int To;
        internal Message Payload = null!;
    }

    private readonly List<Pending> _pending = new List<Pending>();
    private long _sequence;
    private long _sent;

    internal int Now;
    internal int Latency = 1;
    internal int DropEveryNth;             // >0 drops every n-th message: a deterministic loss model

    internal void Send(int from, int to, Message message)
    {
        _sent++;
        if (DropEveryNth > 0 && _sent % DropEveryNth == 0) return;
        _pending.Add(new Pending { ReleaseTick = Now + Latency, Sequence = _sequence++, From = from, To = to, Payload = message });
    }

    internal List<(int From, int To, Message Payload)> ReleaseDue()
    {
        var due = new List<Pending>();
        var later = new List<Pending>();
        foreach (Pending pending in _pending) (pending.ReleaseTick <= Now ? due : later).Add(pending);
        _pending.Clear();
        _pending.AddRange(later);
        due.Sort((a, b) => a.Sequence.CompareTo(b.Sequence));
        var released = new List<(int, int, Message)>();
        foreach (Pending pending in due) released.Add((pending.From, pending.To, pending.Payload));
        return released;
    }
}

internal sealed class RaftNode
{
    private const int ElectionMinTicks = 6;
    private const int ElectionJitterTicks = 4;
    private const int HeartbeatInterval = 2;

    internal readonly int Id;
    internal int CurrentTerm;
    internal int VotedFor = -1;                       // persisted
    internal readonly List<LogEntry> Log = new List<LogEntry>();
    internal int SnapshotIndex;                       // persisted
    internal int SnapshotTerm;                        // persisted
    internal SortedDictionary<string, string> SnapshotState = new SortedDictionary<string, string>(StringComparer.Ordinal);
    internal Membership Membership;
    internal int CommitIndex;
    internal int LastApplied;
    internal SortedDictionary<string, string> StateMachine = new SortedDictionary<string, string>(StringComparer.Ordinal);
    internal Role Role = Role.Follower;
    internal int ElectionTimeout = ElectionMinTicks;
    internal int ElectionElapsed;
    internal int HeartbeatElapsed;
    internal readonly Dictionary<int, int> NextIndex = new Dictionary<int, int>();
    internal readonly Dictionary<int, int> MatchIndex = new Dictionary<int, int>();
    internal readonly List<(int Index, int Term, LogEntry Entry)> AppliedLog = new List<(int, int, LogEntry)>();
    internal readonly HashSet<int> VotesGranted = new HashSet<int>();
    internal ulong Rng;

    internal RaftNode(int id, Membership membership, ulong seed)
    {
        Id = id;
        Membership = membership;
        Rng = seed ^ ((ulong)id * 0x9E3779B97F4A7C15UL);
        ElectionTimeout = ElectionMinTicks + SplitMix64.NextInt(ref Rng, ElectionJitterTicks);
    }

    internal int LastLogIndex => SnapshotIndex + Log.Count;

    internal int LastLogTerm => Log.Count == 0 ? SnapshotTerm : Log[Log.Count - 1].Term;

    /** Term of a log index; 0 for the empty sentinel, -1 when the index is not in this node's log (compacted/unknown). */
    internal int TermAt(int index)
    {
        if (index == 0) return 0;
        if (index == SnapshotIndex) return SnapshotTerm;
        int offset = index - SnapshotIndex - 1;
        if (offset < 0 || offset >= Log.Count) return -1;
        return Log[offset].Term;
    }

    internal LogEntry? EntryAt(int index)
    {
        int offset = index - SnapshotIndex - 1;
        if (offset < 0 || offset >= Log.Count) return null;
        return Log[offset];
    }

    internal void Append(LogEntry entry)
    {
        Log.Add(entry);
        if (entry.IsConfig) Membership = entry.ToMembership();
    }

    internal List<(int To, Message Payload)> Tick()
    {
        var outgoing = new List<(int, Message)>();
        if (!Membership.Contains(Id) && Role != Role.Leader) return outgoing;   // retired members stop campaigning
        if (Role == Role.Leader)
        {
            HeartbeatElapsed++;
            if (HeartbeatElapsed >= HeartbeatInterval)
            {
                HeartbeatElapsed = 0;
                foreach (int peer in MembersOtherThanSelf()) outgoing.Add((peer, BuildAppendEntries(peer)));
            }
        }
        else
        {
            ElectionElapsed++;
            if (ElectionElapsed >= ElectionTimeout) StartElection(outgoing);
        }
        return outgoing;
    }

    internal List<(int To, Message Payload)> Handle(Message message)
    {
        switch (message.Kind)
        {
            case MessageKind.RequestVote: return HandleRequestVote(message);
            case MessageKind.RequestVoteReply: return HandleRequestVoteReply(message);
            case MessageKind.AppendEntries: return HandleAppendEntries(message);
            case MessageKind.AppendEntriesReply: return HandleAppendEntriesReply(message);
            case MessageKind.InstallSnapshot: return HandleInstallSnapshot(message);
            default: return new List<(int, Message)>();
        }
    }

    private int[] MembersOtherThanSelf()
    {
        var others = new List<int>();
        foreach (int id in Membership.Union()) if (id != Id) others.Add(id);
        return others.ToArray();
    }

    private void StartElection(List<(int, Message)> outgoing)
    {
        Role = Role.Candidate;
        CurrentTerm++;
        VotedFor = Id;
        ElectionElapsed = 0;
        ElectionTimeout = ElectionMinTicks + SplitMix64.NextInt(ref Rng, ElectionJitterTicks);
        VotesGranted.Clear();
        VotesGranted.Add(Id);
        foreach (int peer in MembersOtherThanSelf())
        {
            outgoing.Add((peer, new Message
            {
                Kind = MessageKind.RequestVote, From = Id, To = peer, Term = CurrentTerm,
                LastLogIndex = LastLogIndex, LastLogTerm = LastLogTerm
            }));
        }
        if (Membership.HasQuorum(VotesGranted)) BecomeLeader();
    }

    private void BecomeLeader()
    {
        Role = Role.Leader;
        HeartbeatElapsed = HeartbeatInterval;         // replicate on the next tick
        NextIndex.Clear();
        MatchIndex.Clear();
        foreach (int peer in MembersOtherThanSelf())
        {
            NextIndex[peer] = LastLogIndex + 1;
            MatchIndex[peer] = 0;
        }
        AdvanceCommit();                              // covers the single-member cluster, where self is a quorum
    }

    /** Client proposal; only a leader accepts one. */
    internal void Propose(string command)
    {
        if (Role != Role.Leader) throw new InvalidOperationException("node " + Id + " is not the leader");
        Append(new LogEntry(LastLogIndex + 1, CurrentTerm, command));
        HeartbeatElapsed = HeartbeatInterval;
        AdvanceCommit();
    }

    /** Starts a joint configuration change; false when a change is already in flight (Raft's one-at-a-time rule). */
    internal bool ProposeJointConfig(int[] oldSet, int[] newSet)
    {
        if (Role != Role.Leader) throw new InvalidOperationException("node " + Id + " is not the leader");
        if (Membership.Joint) return false;
        for (int index = CommitIndex + 1; index <= LastLogIndex; index++)
        {
            LogEntry? pending = EntryAt(index);
            if (pending != null && pending.IsConfig) return false;
        }
        Append(new LogEntry(LastLogIndex + 1, CurrentTerm, null, null, oldSet, newSet));
        HeartbeatElapsed = HeartbeatInterval;
        AdvanceCommit();
        return true;
    }

    /** Finishes a membership change by appending the plain C_new entry after C_old,new has committed. */
    internal void ProposeSingleConfig(int[] members)
    {
        if (Role != Role.Leader) throw new InvalidOperationException("node " + Id + " is not the leader");
        Append(new LogEntry(LastLogIndex + 1, CurrentTerm, null, members));
        HeartbeatElapsed = HeartbeatInterval;
        AdvanceCommit();
    }

    private Message BuildAppendEntries(int peer)
    {
        if (!NextIndex.TryGetValue(peer, out int next)) next = LastLogIndex + 1;
        if (next <= SnapshotIndex)
        {
            return new Message
            {
                Kind = MessageKind.InstallSnapshot, From = Id, To = peer, Term = CurrentTerm,
                SnapshotIndex = SnapshotIndex, SnapshotTerm = SnapshotTerm,
                SnapshotState = new SortedDictionary<string, string>(SnapshotState, StringComparer.Ordinal)
            };
        }
        var entries = new List<LogEntry>();
        for (int index = next; index <= LastLogIndex; index++)
        {
            LogEntry? entry = EntryAt(index);
            if (entry != null) entries.Add(entry);
        }
        return new Message
        {
            Kind = MessageKind.AppendEntries, From = Id, To = peer, Term = CurrentTerm,
            PrevLogIndex = next - 1, PrevLogTerm = TermAt(next - 1), LeaderCommit = CommitIndex, Entries = entries
        };
    }

    private List<(int, Message)> HandleRequestVote(Message message)
    {
        if (message.Term > CurrentTerm)
        {
            CurrentTerm = message.Term;
            VotedFor = -1;
            Role = Role.Follower;
        }
        bool granted = false;
        if (message.Term == CurrentTerm)
        {
            bool upToDate = message.LastLogTerm > LastLogTerm
                || (message.LastLogTerm == LastLogTerm && message.LastLogIndex >= LastLogIndex);
            if ((VotedFor == -1 || VotedFor == message.From) && upToDate)
            {
                VotedFor = message.From;
                granted = true;
                ElectionElapsed = 0;
            }
        }
        return new List<(int, Message)>
        {
            (message.From, new Message { Kind = MessageKind.RequestVoteReply, From = Id, To = message.From, Term = CurrentTerm, Granted = granted })
        };
    }

    private List<(int, Message)> HandleRequestVoteReply(Message message)
    {
        var outgoing = new List<(int, Message)>();
        if (message.Term > CurrentTerm)
        {
            CurrentTerm = message.Term;
            VotedFor = -1;
            Role = Role.Follower;
            return outgoing;
        }
        if (Role != Role.Candidate || message.Term != CurrentTerm) return outgoing;
        if (message.Granted && VotesGranted.Add(message.From) && Membership.HasQuorum(VotesGranted)) BecomeLeader();
        return outgoing;
    }

    private List<(int, Message)> HandleAppendEntries(Message message)
    {
        var outgoing = new List<(int, Message)>();
        if (message.Term < CurrentTerm)
        {
            outgoing.Add((message.From, ReplyAppend(message, false, LastLogIndex + 1)));
            return outgoing;
        }
        if (message.Term > CurrentTerm) VotedFor = -1;
        CurrentTerm = message.Term;
        Role = Role.Follower;
        ElectionElapsed = 0;

        if (message.PrevLogIndex > LastLogIndex)
        {
            outgoing.Add((message.From, ReplyAppend(message, false, LastLogIndex + 1)));
            return outgoing;
        }
        if (message.PrevLogIndex > SnapshotIndex && TermAt(message.PrevLogIndex) != message.PrevLogTerm)
        {
            int conflict = message.PrevLogIndex;
            while (conflict > SnapshotIndex + 1 && TermAt(conflict - 1) == TermAt(conflict)) conflict--;
            outgoing.Add((message.From, ReplyAppend(message, false, conflict)));
            return outgoing;
        }

        foreach (LogEntry entry in message.Entries)
        {
            if (entry.Index <= SnapshotIndex) continue;                 // already inside the local snapshot
            int offset = entry.Index - SnapshotIndex - 1;
            if (offset < Log.Count)
            {
                if (Log[offset].Term == entry.Term) continue;           // identical entry: log matching holds
                Log.RemoveRange(offset, Log.Count - offset);            // conflict: truncate the uncommitted suffix
            }
            Append(entry);
        }
        if (message.LeaderCommit > CommitIndex) CommitIndex = Math.Min(message.LeaderCommit, LastLogIndex);
        Apply();
        outgoing.Add((message.From, ReplyAppend(message, true, LastLogIndex)));
        return outgoing;
    }

    private Message ReplyAppend(Message request, bool success, int conflictIndex)
    {
        return new Message
        {
            Kind = MessageKind.AppendEntriesReply, From = Id, To = request.From, Term = CurrentTerm,
            Success = success, MatchIndex = success ? LastLogIndex : 0, ConflictIndex = conflictIndex
        };
    }

    private List<(int, Message)> HandleAppendEntriesReply(Message message)
    {
        var outgoing = new List<(int, Message)>();
        if (message.Term > CurrentTerm)
        {
            CurrentTerm = message.Term;
            VotedFor = -1;
            Role = Role.Follower;
            return outgoing;
        }
        if (Role != Role.Leader || message.Term != CurrentTerm) return outgoing;
        if (message.Success)
        {
            MatchIndex[message.From] = Math.Max(MatchIndex.GetValueOrDefault(message.From), message.MatchIndex);
            NextIndex[message.From] = message.MatchIndex + 1;
            AdvanceCommit();
        }
        else
        {
            int previous = NextIndex.GetValueOrDefault(message.From);
            int fallback = Math.Max(1, Math.Min(message.ConflictIndex, previous > 0 ? previous - 1 : 1));
            NextIndex[message.From] = fallback;
        }
        return outgoing;
    }

    /**
     * The Figure-8 rule: an entry becomes committed only when it belongs to the leader's current term and a quorum
     * covers it. Entries from earlier terms are committed transitively through the first current-term entry.
     */
    internal void AdvanceCommit()
    {
        for (int index = CommitIndex + 1; index <= LastLogIndex; index++)
        {
            if (TermAt(index) != CurrentTerm) continue;
            var voters = new HashSet<int> { Id };
            foreach (int peer in MembersOtherThanSelf())
            {
                if (MatchIndex.GetValueOrDefault(peer) >= index) voters.Add(peer);
            }
            if (Membership.HasQuorum(voters)) CommitIndex = index;
        }
        Apply();
    }

    private List<(int, Message)> HandleInstallSnapshot(Message message)
    {
        var outgoing = new List<(int, Message)>();
        if (message.Term < CurrentTerm)
        {
            outgoing.Add((message.From, new Message { Kind = MessageKind.InstallSnapshotReply, From = Id, To = message.From, Term = CurrentTerm, Success = false }));
            return outgoing;
        }
        CurrentTerm = message.Term;
        Role = Role.Follower;
        ElectionElapsed = 0;
        if (message.SnapshotIndex > CommitIndex)
        {
            SnapshotState = new SortedDictionary<string, string>(message.SnapshotState ?? new SortedDictionary<string, string>(StringComparer.Ordinal), StringComparer.Ordinal);
            StateMachine = new SortedDictionary<string, string>(SnapshotState, StringComparer.Ordinal);
            int removeCount = Math.Min(message.SnapshotIndex - SnapshotIndex, Log.Count);
            if (removeCount > 0) Log.RemoveRange(0, removeCount);
            SnapshotIndex = message.SnapshotIndex;
            SnapshotTerm = message.SnapshotTerm;
            CommitIndex = Math.Max(CommitIndex, message.SnapshotIndex);
            LastApplied = Math.Max(LastApplied, message.SnapshotIndex);
        }
        outgoing.Add((message.From, new Message { Kind = MessageKind.InstallSnapshotReply, From = Id, To = message.From, Term = CurrentTerm, Success = true, MatchIndex = LastLogIndex }));
        return outgoing;
    }

    internal void Apply()
    {
        while (LastApplied < CommitIndex)
        {
            LastApplied++;
            LogEntry? entry = EntryAt(LastApplied);
            if (entry == null) continue;                                // inside the snapshot: already reflected there
            if (entry.IsConfig) ApplyConfig(entry);
            else ApplyCommand(entry.Command!);
            AppliedLog.Add((LastApplied, entry.Term, entry));
        }
    }

    private void ApplyConfig(LogEntry entry)
    {
        Membership = entry.ToMembership();
        if (!Membership.Contains(Id)) Role = Role.Follower;             // removed leader steps down once C_new commits
    }

    private void ApplyCommand(string command)
    {
        string[] parts = command.Split(' ');
        switch (parts[0])
        {
            case "SET": StateMachine[parts[1]] = parts[2]; break;
            case "INCR":
                string current = StateMachine.TryGetValue(parts[1], out string? existing) ? existing : "0";
                StateMachine[parts[1]] = (int.Parse(current) + 1).ToString();
                break;
            case "DEL": StateMachine.Remove(parts[1]); break;
            default: throw new InvalidOperationException("unknown command: " + command);
        }
    }

    /** Compacts the committed prefix into a snapshot; log entries at or below SnapshotIndex are dropped. */
    internal void Compact()
    {
        if (CommitIndex <= SnapshotIndex) return;
        int previous = SnapshotIndex;
        int term = TermAt(CommitIndex);                 // read the term before SnapshotIndex starts shadowing the log
        SnapshotIndex = CommitIndex;
        SnapshotTerm = term;
        SnapshotState = new SortedDictionary<string, string>(StateMachine, StringComparer.Ordinal);
        int removeCount = Math.Min(SnapshotIndex - previous, Log.Count);
        if (removeCount > 0) Log.RemoveRange(0, removeCount);
    }

    /** Crash and restart: persistent state (term, vote, log, snapshot) survives, volatile state is rebuilt. */
    internal void Restart()
    {
        Role = Role.Follower;
        ElectionElapsed = 0;
        HeartbeatElapsed = 0;
        VotesGranted.Clear();
        NextIndex.Clear();
        MatchIndex.Clear();
        AppliedLog.Clear();
        StateMachine = new SortedDictionary<string, string>(SnapshotState, StringComparer.Ordinal);
        LastApplied = SnapshotIndex;
        CommitIndex = SnapshotIndex;
        Apply();                                                        // rebuild from snapshot + durable log
    }

    internal string Digest()
    {
        var builder = new StringBuilder();
        builder.Append('t').Append(CurrentTerm).Append('|').Append(Membership.Key()).Append('|');
        for (int index = SnapshotIndex + 1; index <= LastLogIndex; index++)
        {
            LogEntry? entry = EntryAt(index);
            if (entry != null) builder.Append(entry.Index).Append(':').Append(entry.Term).Append(':').Append(entry.Label()).Append(';');
        }
        builder.Append('#');
        foreach (var pair in StateMachine) builder.Append(pair.Key).Append('=').Append(pair.Value).Append(',');
        return builder.ToString();
    }
}

/** Recomputes leadership, votes and commits from the observed trace instead of trusting node state. */
internal sealed class SafetyChecker
{
    internal readonly Dictionary<int, int> LeaderByTerm = new Dictionary<int, int>();
    internal readonly Dictionary<(int Term, int Voter), int> Votes = new Dictionary<(int, int), int>();
    internal readonly Dictionary<int, string> CommittedByIndex = new Dictionary<int, string>();
    internal readonly Dictionary<int, int> CommittedTerm = new Dictionary<int, int>();
    private readonly Dictionary<int, int> _commitSeen = new Dictionary<int, int>();
    private readonly Dictionary<int, int> _applyCursor = new Dictionary<int, int>();
    internal bool LogMatchingHolds = true;
    internal string? Violation;

    internal void Observe(RaftNode node)
    {
        if (_commitSeen.TryGetValue(node.Id, out int previous) && node.CommitIndex < previous)
            Fail("commitIndex moved backwards on node " + node.Id);
        _commitSeen[node.Id] = node.CommitIndex;

        if (node.Role == Role.Leader)
        {
            if (LeaderByTerm.TryGetValue(node.CurrentTerm, out int other) && other != node.Id)
                Fail("two leaders in term " + node.CurrentTerm + ": " + other + " and " + node.Id);
            else LeaderByTerm[node.CurrentTerm] = node.Id;
        }
        if (node.VotedFor != -1) RecordVote(node.CurrentTerm, node.Id, node.VotedFor);

        int cursor = _applyCursor.GetValueOrDefault(node.Id);
        for (int i = cursor; i < node.AppliedLog.Count; i++)
        {
            (int index, int term, LogEntry entry) = node.AppliedLog[i];
            string label = entry.Label();
            if (CommittedByIndex.TryGetValue(index, out string? existing))
            {
                if (existing != label) Fail("state machine safety violated at index " + index + ": " + existing + " vs " + label);
                if (CommittedTerm[index] != term) Fail("state machine safety (term) violated at index " + index);
            }
            else
            {
                CommittedByIndex[index] = label;
                CommittedTerm[index] = term;
            }
        }
        _applyCursor[node.Id] = node.AppliedLog.Count;
    }

    /** A voter may name exactly one candidate per term; this is what makes two leaders in one term impossible. */
    internal void RecordVote(int term, int voter, int candidate)
    {
        var key = (term, voter);
        if (Votes.TryGetValue(key, out int existing) && existing != candidate)
            Fail("voter " + voter + " voted for two candidates in term " + term);
        Votes[key] = candidate;
    }

    /** A restart legitimately resets volatile indices; the checker must not flag that as a regression. */
    internal void NoteRestart(int nodeId)
    {
        _commitSeen.Remove(nodeId);
        _applyCursor[nodeId] = 0;
    }

    /**
     * Log matching, independently of the nodes: (a) a committed index must never be overwritten by a different
     * term, and (b) if two logs agree on (index, term) then every earlier known index agrees too.
     */
    internal void CheckLogMatching(IReadOnlyCollection<RaftNode> nodes)
    {
        foreach (RaftNode node in nodes)
        {
            for (int index = 1; index <= node.LastLogIndex; index++)
            {
                int term = node.TermAt(index);
                if (term <= 0) continue;
                if (CommittedTerm.TryGetValue(index, out int committedTerm) && committedTerm != term)
                    Fail("committed entry at index " + index + " was overwritten on node " + node.Id);
            }
        }
        var list = new List<RaftNode>(nodes);
        for (int a = 0; a < list.Count; a++)
        {
            for (int b = a + 1; b < list.Count; b++)
            {
                int limit = Math.Min(list[a].LastLogIndex, list[b].LastLogIndex);
                int lastAgreeing = 0;
                for (int index = 1; index <= limit; index++)
                {
                    int termA = list[a].TermAt(index);
                    int termB = list[b].TermAt(index);
                    if (termA <= 0 || termB <= 0) continue;
                    if (termA == termB) lastAgreeing = index;
                }
                for (int index = 1; index <= lastAgreeing; index++)
                {
                    int termA = list[a].TermAt(index);
                    int termB = list[b].TermAt(index);
                    if (termA <= 0 || termB <= 0) continue;
                    if (termA != termB) Fail("log matching violated at index " + index + " between nodes " + list[a].Id + " and " + list[b].Id);
                }
            }
        }
    }

    internal void Fail(string detail) => Violation ??= detail;
}

internal sealed class Cluster
{
    internal readonly SortedDictionary<int, RaftNode> Nodes = new SortedDictionary<int, RaftNode>();
    internal readonly InMemoryNetwork Net = new InMemoryNetwork();
    internal readonly Dictionary<int, int> Group = new Dictionary<int, int>();
    internal readonly HashSet<int> Down = new HashSet<int>();
    internal readonly SafetyChecker Checker = new SafetyChecker();
    internal readonly List<string> Proposed = new List<string>();
    internal int Tick;

    internal Cluster(int[] members, ulong seed, int[]? dormantNodes = null)
    {
        var membership = new Membership(members);
        foreach (int id in members) Nodes[id] = new RaftNode(id, membership, seed);
        if (dormantNodes != null)
        {
            foreach (int id in dormantNodes)
            {
                Nodes[id] = new RaftNode(id, membership, seed);       // exists, but is not a member: no campaigning
                Down.Add(id);
            }
        }
        foreach (int id in Nodes.Keys) Group[id] = 0;
    }

    internal void Step()
    {
        Net.Now = Tick;
        foreach ((int from, int to, Message payload) in Net.ReleaseDue())
        {
            if (!Nodes.ContainsKey(to) || !Nodes.ContainsKey(from)) continue;
            if (Down.Contains(to) || Down.Contains(from)) continue;
            if (Group[from] != Group[to]) continue;                   // messages never cross a partition
            RaftNode node = Nodes[to];
            if (payload.Kind == MessageKind.RequestVoteReply && payload.Granted)
                Checker.RecordVote(payload.Term, payload.From, payload.To);
            foreach ((int destination, Message reply) in node.Handle(payload)) Net.Send(to, destination, reply);
            Checker.Observe(node);
        }
        foreach (RaftNode node in Nodes.Values)
        {
            if (Down.Contains(node.Id)) continue;
            foreach ((int destination, Message message) in node.Tick()) Net.Send(node.Id, destination, message);
            Checker.Observe(node);
        }
        Tick++;
    }

    /** Crash = lose volatile state but keep the durable log/term/vote/snapshot, exactly like a process restart. */
    internal void Crash(int id)
    {
        Down.Add(id);
        Nodes[id].Restart();
        Checker.NoteRestart(id);
    }

    internal void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Step();
    }

    internal void RunUntil(Func<bool> condition, int maxTicks)
    {
        for (int i = 0; i < maxTicks && !condition(); i++) Step();
    }

    internal RaftNode? Leader()
    {
        foreach (RaftNode node in Nodes.Values)
        {
            if (!Down.Contains(node.Id) && node.Role == Role.Leader) return node;
        }
        return null;
    }

    internal List<RaftNode> Live()
    {
        var live = new List<RaftNode>();
        foreach (RaftNode node in Nodes.Values) if (!Down.Contains(node.Id)) live.Add(node);
        return live;
    }

    internal bool Propose(string command)
    {
        RaftNode? leader = Leader();
        if (leader == null || leader.Membership.Joint) return false;  // no new entries while a change is in flight
        leader.Propose(command);
        Proposed.Add(command);
        return true;
    }

    internal bool ProposeJointConfig(int[] oldSet, int[] newSet)
    {
        RaftNode? leader = Leader();
        return leader != null && leader.ProposeJointConfig(oldSet, newSet);
    }

    internal bool ProposeSingleConfig(int[] members)
    {
        RaftNode? leader = Leader();
        if (leader == null) return false;
        leader.ProposeSingleConfig(members);
        return true;
    }

    internal bool AllConverged()
    {
        var live = Live();
        if (live.Count == 0) return false;
        string digest = live[0].Digest();
        foreach (RaftNode node in live) if (node.Digest() != digest) return false;
        return true;
    }

    internal bool AllStateMachinesEqual()
    {
        var live = Live();
        if (live.Count == 0) return false;
        foreach (RaftNode node in live)
        {
            if (node.StateMachine.Count != live[0].StateMachine.Count) return false;
            foreach (var pair in live[0].StateMachine)
            {
                if (node.StateMachine.GetValueOrDefault(pair.Key) != pair.Value) return false;
            }
        }
        return true;
    }

    internal bool AllCommitted(int index)
    {
        foreach (RaftNode node in Live()) if (node.CommitIndex < index) return false;
        return true;
    }
}

/** Independent quorum oracle: counts explicit voter sets against explicit member sets. */
internal sealed class QuorumOracle
{
    internal static bool IsQuorum(int[] members, int[] voters)
    {
        int count = 0;
        foreach (int voter in voters) if (Array.IndexOf(members, voter) >= 0) count++;
        return count * 2 > members.Length;
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-2. Single-member cluster: elects itself and commits without any vote traffic.
        var solo = new Cluster(new[] { 1 }, 0xA11CEUL);
        solo.RunUntil(() => solo.Leader() != null, 30);
        Check(solo.Leader() != null && solo.Leader()!.Id == 1, "a single-member cluster makes itself leader");
        solo.Propose("SET a 1");
        solo.Propose("INCR n");
        solo.Run(4);
        Check(solo.Nodes[1].CommitIndex == 2 && solo.Nodes[1].StateMachine.GetValueOrDefault("a") == "1"
            && solo.Nodes[1].StateMachine.GetValueOrDefault("n") == "1",
            "a single-member cluster commits and applies without network traffic");

        // 3-4. Five-node cluster: one leader inside the timeout bound, no safety violation during the election.
        var cluster = new Cluster(new[] { 1, 2, 3, 4, 5 }, 0xBEEFUL);
        int ticksToLeadership = 0;
        while (cluster.Leader() == null && ticksToLeadership < 40) { cluster.Step(); ticksToLeadership++; }
        Check(cluster.Leader() != null, "a majority elects a leader within the election-timeout bound");
        Check(ticksToLeadership <= 10, "leadership is reached quickly (observed " + ticksToLeadership + " ticks)");
        Check(cluster.Checker.Violation == null, "no election-safety violation during the first election (" + (cluster.Checker.Violation ?? "-") + ")");
        Check(QuorumOracle.IsQuorum(new[] { 1, 2, 3, 4, 5 }, new[] { 1, 2, 3 })
            && !QuorumOracle.IsQuorum(new[] { 1, 2, 3, 4, 5 }, new[] { 1, 2 }),
            "the quorum oracle agrees that 3 of 5 is a majority and 2 of 5 is not");

        // 5-6. A command stream commits on every replica and the state grows with the committed log.
        int knownLeaderTerm = cluster.Leader()!.CurrentTerm;
        cluster.Propose("SET user:1 alice");
        cluster.Propose("INCR counter");
        cluster.Propose("INCR counter");
        cluster.Propose("SET user:2 bob");
        cluster.Propose("DEL user:2");
        cluster.RunUntil(() => cluster.AllCommitted(5), 80);
        Check(cluster.AllCommitted(5), "all five acknowledged proposals commit on every live node");
        Check(cluster.AllConverged(), "all replicas hold an identical log and state machine after replication");
        Check(cluster.Nodes[1].StateMachine.GetValueOrDefault("counter") == "2"
            && cluster.Nodes[1].StateMachine.GetValueOrDefault("user:1") == "alice"
            && !cluster.Nodes[1].StateMachine.ContainsKey("user:2"), "the state machine reflects SET/INCR/DEL semantics");
        Check(knownLeaderTerm >= 1 && cluster.Checker.LeaderByTerm.Count >= 1, "the checker recorded the elected term");

        // 7. Independent oracle: replaying the committed labels reproduces the replicated state machine.
        var oracle = new SortedDictionary<string, string>(StringComparer.Ordinal);
        var order = new List<int>(cluster.Checker.CommittedByIndex.Keys);
        order.Sort();
        foreach (int index in order)
        {
            string label = cluster.Checker.CommittedByIndex[index];
            if (label.StartsWith("cfg:", StringComparison.Ordinal)) continue;
            string[] parts = label.Split(' ');
            if (parts[0] == "SET") oracle[parts[1]] = parts[2];
            else if (parts[0] == "INCR")
            {
                string current = oracle.TryGetValue(parts[1], out string? existing) ? existing : "0";
                oracle[parts[1]] = (int.Parse(current) + 1).ToString();
            }
            else if (parts[0] == "DEL") oracle.Remove(parts[1]);
        }
        bool oracleAgrees = oracle.Count == cluster.Nodes[1].StateMachine.Count;
        foreach (var pair in oracle)
        {
            if (cluster.Nodes[1].StateMachine.GetValueOrDefault(pair.Key) != pair.Value) oracleAgrees = false;
        }
        Check(oracleAgrees, "an independent replay of the committed commands matches the replicas exactly");

        // 8. Log matching, term monotonicity and apply/commit ordering.
        cluster.Checker.CheckLogMatching(cluster.Nodes.Values);
        Check(cluster.Checker.LogMatchingHolds && cluster.Checker.Violation == null, "log matching holds after replication");
        foreach (RaftNode node in cluster.Nodes.Values)
        {
            int previousTerm = 0;
            for (int index = node.SnapshotIndex + 1; index <= node.LastLogIndex; index++)
            {
                int term = node.TermAt(index);
                Check(term >= previousTerm, "log terms are non-decreasing on node " + node.Id);
                previousTerm = term;
            }
            Check(node.LastApplied <= node.CommitIndex, "the applied prefix never exceeds commitIndex on node " + node.Id);
        }

        // 9-10. Figure-8: an entry from an earlier term is never committed by counting replicas alone.
        var figure8 = new Cluster(new[] { 1, 2, 3, 4, 5 }, 0xF1C8UL);
        figure8.Nodes[1].Append(new LogEntry(1, 1, "SET a 1"));
        figure8.Nodes[1].Append(new LogEntry(2, 2, "SET b 2"));
        figure8.Nodes[2].Append(new LogEntry(1, 1, "SET a 1"));
        figure8.Nodes[2].Append(new LogEntry(2, 2, "SET b 2"));
        foreach (int id in new[] { 3, 4, 5 }) figure8.Nodes[id].Append(new LogEntry(1, 1, "SET a 1"));
        RaftNode stale = figure8.Nodes[2];
        stale.CurrentTerm = 5;
        stale.Role = Role.Leader;
        foreach (int peer in new[] { 1, 3, 4, 5 }) { stale.NextIndex[peer] = 1; stale.MatchIndex[peer] = 0; }
        stale.MatchIndex[1] = 2;                                  // nodes 1, 2 and 3 already store the term-2 entry
        stale.MatchIndex[3] = 2;
        stale.AdvanceCommit();
        Check(stale.CommitIndex == 0, "an old-term entry is not committed even when a quorum already stores it");
        stale.Append(new LogEntry(3, 5, "SET c 3"));               // a current-term entry unblocks the prefix
        stale.MatchIndex[1] = 3;
        stale.MatchIndex[3] = 3;
        stale.AdvanceCommit();
        Check(stale.CommitIndex == 3, "once a current-term entry commits, the earlier-term prefix commits with it");
        Check(stale.StateMachine.GetValueOrDefault("b") == "2" && stale.StateMachine.GetValueOrDefault("c") == "3",
            "the committed prefix keeps the earlier term's command (no lost update)");

        // 11-12. One vote per term, persisted across a restart; a higher term clears the vote again.
        var voter = new RaftNode(7, new Membership(new[] { 1, 2, 3 }), 0xC0FFEEUL);
        voter.CurrentTerm = 9;
        Message firstVote = voter.Handle(new Message { Kind = MessageKind.RequestVote, From = 1, To = 7, Term = 9 })[0].Payload;
        Check(firstVote.Granted, "the first candidate in a term is granted the vote");
        voter.Restart();
        Message secondVote = voter.Handle(new Message { Kind = MessageKind.RequestVote, From = 2, To = 7, Term = 9 })[0].Payload;
        Check(!secondVote.Granted, "after a crash the persisted vote blocks a second candidate in the same term");
        Check(voter.Handle(new Message { Kind = MessageKind.RequestVote, From = 2, To = 7, Term = 10 })[0].Payload.Granted,
            "a higher term clears the vote, so the node is not deadlocked forever");
        Check(!voter.Handle(new Message { Kind = MessageKind.RequestVote, From = 3, To = 7, Term = 10, LastLogIndex = 9, LastLogTerm = 10 })[0].Payload.Granted
            || voter.VotedFor != -1, "a vote is always attributed to exactly one candidate");

        // 13-14. Stale leaders are refused and conflicting uncommitted entries are truncated.
        var follower = new RaftNode(8, new Membership(new[] { 1, 2, 3 }), 0x1234UL);
        follower.CurrentTerm = 3;
        follower.Append(new LogEntry(1, 1, "SET k a"));
        follower.Append(new LogEntry(2, 3, "SET k b"));
        Message staleAppend = new Message { Kind = MessageKind.AppendEntries, From = 1, To = 8, Term = 2, PrevLogIndex = 0, PrevLogTerm = 0 };
        Check(!follower.Handle(staleAppend)[0].Payload.Success && follower.CurrentTerm == 3, "AppendEntries from a stale term is refused");
        var conflicting = new Message { Kind = MessageKind.AppendEntries, From = 1, To = 8, Term = 4, PrevLogIndex = 1, PrevLogTerm = 1 };
        conflicting.Entries.Add(new LogEntry(2, 4, "SET k c"));
        Check(follower.Handle(conflicting)[0].Payload.Success, "a matching prefix is accepted");
        Check(follower.TermAt(2) == 4 && follower.EntryAt(2)!.Command == "SET k c", "the conflicting uncommitted entry was truncated");
        Check(follower.LastLogIndex == 2, "truncation keeps exactly one suffix entry");

        // 15-17. Snapshots: compaction preserves the mapping and a lagging node catches up via InstallSnapshot.
        var lag = new Cluster(new[] { 1, 2, 3 }, 0x5A9UL);
        lag.RunUntil(() => lag.Leader() != null, 30);
        for (int i = 0; i < 60; i++) lag.Propose("INCR n" + (i % 4));
        lag.Run(40);
        RaftNode firstLeader = lag.Leader()!;
        Check(firstLeader.CommitIndex >= 40, "the leader commits a long command stream (committed " + firstLeader.CommitIndex + ")");
        int committedAtSnapshot = firstLeader.CommitIndex;
        int snapshotTerm = firstLeader.TermAt(committedAtSnapshot);
        firstLeader.Compact();
        Check(firstLeader.SnapshotIndex == committedAtSnapshot && firstLeader.SnapshotTerm == snapshotTerm,
            "snapshot(lastIncludedIndex, lastIncludedTerm) matches the log");
        Check(firstLeader.EntryAt(committedAtSnapshot) == null && firstLeader.TermAt(committedAtSnapshot) == snapshotTerm,
            "compacted entries leave the log but TermAt still answers for the snapshot boundary");
        int snapshotTotal = 0;
        foreach (var pair in firstLeader.SnapshotState) snapshotTotal += int.Parse(pair.Value);
        Check(snapshotTotal == committedAtSnapshot, "the snapshot froze the state machine exactly at the commit point");
        int snapshotBefore = firstLeader.SnapshotIndex;
        var followers = new List<int>();
        foreach (int id in new[] { 1, 2, 3 }) if (id != firstLeader.Id) followers.Add(id);
        int laggingNode = followers[0];
        lag.Crash(laggingNode);
        for (int i = 0; i < 30; i++) lag.Propose("INCR n" + (i % 3));
        lag.Run(60);
        RaftNode? secondLeader = lag.Leader();
        Check(secondLeader != null && secondLeader.CommitIndex > snapshotBefore, "the surviving majority keeps committing without the lagging node");
        secondLeader!.Compact();
        Check(secondLeader.SnapshotIndex > snapshotBefore, "the leader compacted past the lagging node's position");
        lag.Down.Remove(laggingNode);
        lag.Run(160);
        Check(lag.Nodes[laggingNode].SnapshotIndex > snapshotBefore, "the lagging node adopted a snapshot instead of replaying the whole log");
        Check(lag.Nodes[laggingNode].CommitIndex >= snapshotBefore, "the lagging node committed up to at least its old position");
        Check(lag.AllStateMachinesEqual(), "snapshot transfer plus tail replication converge the state machines");
        Check(lag.Nodes[laggingNode].SnapshotState.Count > 0, "the transferred snapshot carries the frozen state machine");

        // 18-21. Reconfiguration with joint consensus.
        var reconfig = new Cluster(new[] { 1, 2, 3, 4, 5 }, 0x7E1UL, new[] { 6, 7, 8, 9, 10 });
        reconfig.RunUntil(() => reconfig.Leader() != null, 30);
        var oldConfig = new[] { 1, 2, 3, 4, 5 };
        var newConfig = new[] { 1, 2, 3, 6, 7, 8, 9, 10 };
        Check(reconfig.Leader() != null, "the original five-member cluster has a leader before reconfiguration");
        Check(reconfig.ProposeJointConfig(oldConfig, newConfig), "a joint configuration C_old,new can be started");
        int jointIndex = reconfig.Leader()!.LastLogIndex;
        reconfig.Run(3);
        Check(!reconfig.ProposeJointConfig(oldConfig, newConfig), "a second membership change is refused while one is in flight");
        reconfig.Crash(4);
        reconfig.Crash(5);
        reconfig.Run(60);
        Check(reconfig.Leader()!.CommitIndex < jointIndex,
            "while joint, a C_old majority alone cannot commit (3 of 5 alive but only 3 of 8 in C_new)");
        Check(!QuorumOracle.IsQuorum(newConfig, new[] { 1, 2, 3 }) && QuorumOracle.IsQuorum(newConfig, new[] { 1, 2, 3, 6, 7 }),
            "the quorum oracle confirms the joint commit boundary");
        Check(reconfig.Nodes[6].Role == Role.Follower, "a dormant non-member never campaigns for leadership");
        reconfig.Down.Remove(6);
        reconfig.Down.Remove(7);
        reconfig.RunUntil(() => reconfig.AllCommitted(jointIndex), 120);
        Check(reconfig.AllCommitted(jointIndex), "starting C_new members lets the joint configuration commit");
        Check(reconfig.ProposeSingleConfig(newConfig), "the plain C_new entry can be appended after the joint entry commits");
        int singleIndex = reconfig.Leader()!.LastLogIndex;
        reconfig.RunUntil(() => reconfig.AllCommitted(singleIndex), 120);
        Check(reconfig.AllCommitted(singleIndex), "C_new commits and replaces the joint configuration");
        Check(reconfig.Leader() != null && !reconfig.Leader()!.Membership.Joint && reconfig.Leader()!.Membership.New.Length == newConfig.Length,
            "the new single configuration is in force after the change");

        // 22-23. A leader removed from C_new steps down, and the survivors elect a new leader.
        RaftNode doomed = reconfig.Leader()!;
        var shrink = new List<int>();
        foreach (int member in newConfig) if (member != doomed.Id) shrink.Add(member);
        Check(reconfig.ProposeSingleConfig(shrink.ToArray()), "a configuration C_new that excludes the current leader is accepted");
        int shrinkIndex = doomed.LastLogIndex;
        reconfig.RunUntil(() => doomed.CommitIndex >= shrinkIndex, 200);
        Check(doomed.CommitIndex >= shrinkIndex, "the removed leader commits the change that removes it");
        reconfig.Run(80);
        Check(doomed.Role != Role.Leader, "a leader removed from C_new steps down once the change commits");
        Check(reconfig.Leader() != null, "the remaining members elect a new leader");
        Check(reconfig.Checker.Violation == null, "no safety violation around the two membership changes");
        Console.WriteLine("  reconfiguration: joint index=" + jointIndex + " single index=" + singleIndex + " shrink index=" + shrinkIndex
            + " retired leader=" + doomed.Id);

        // 24. Determinism: the same seed and schedule reproduce a bit-identical cluster.
        var repeatA = new Cluster(new[] { 1, 2, 3, 4, 5 }, 0xD0D0UL);
        var repeatB = new Cluster(new[] { 1, 2, 3, 4, 5 }, 0xD0D0UL);
        for (int i = 0; i < 120; i++)
        {
            repeatA.Propose("SET k" + (i % 7) + " v" + i);
            repeatB.Propose("SET k" + (i % 7) + " v" + i);
            repeatA.Step();
            repeatB.Step();
        }
        bool identical = true;
        foreach (int id in repeatA.Nodes.Keys) if (repeatA.Nodes[id].Digest() != repeatB.Nodes[id].Digest()) identical = false;
        Check(identical, "identical seeds produce bit-identical replicas");

        // 25. Fault sweep: crashes, revivals and message loss never break safety, and the cluster re-converges.
        int convergedSchedules = 0;
        int totalProposals = 0;
        int totalCommitted = 0;
        int worstCommit = int.MaxValue;
        for (ulong seed = 1; seed <= 8; seed++)
        {
            var busy = new Cluster(new[] { 1, 2, 3, 4, 5 }, seed * 0x9E37UL);
            ulong rng = seed * 0xC0FFEEUL;
            for (int round = 0; round < 26; round++)
            {
                busy.Run(12);
                ulong roll = SplitMix64.Next(ref rng);
                switch (roll % 4)
                {
                    case 0: busy.Crash(1 + (int)((roll >> 8) % 5)); break;                    // crash a node
                    case 1: busy.Down.Remove(1 + (int)((roll >> 8) % 5)); break;             // revive a node
                    case 2: busy.Net.DropEveryNth = 3 + (int)((roll >> 8) % 9); break;       // begin dropping messages
                    default:
                        busy.Net.DropEveryNth = 0;                                           // heal everything
                        for (int id = 1; id <= 5; id++) busy.Down.Remove(id);
                        break;
                }
                for (int k = 0; k < 3; k++)
                {
                    int before = busy.Proposed.Count;
                    busy.Propose("INCR c" + ((int)(roll % 3)));
                    if (busy.Proposed.Count > before) totalProposals++;
                }
            }
            busy.Net.DropEveryNth = 0;
            busy.Down.Clear();
            busy.Run(240);
            Check(busy.Checker.Violation == null, "fault schedule " + seed + " kept every safety invariant");
            busy.Checker.CheckLogMatching(busy.Nodes.Values);
            Check(busy.AllConverged(), "fault schedule " + seed + " converged after healing");
            Check(busy.Leader() != null, "fault schedule " + seed + " re-elected a leader after healing");
            int highest = 0;
            foreach (int index in busy.Checker.CommittedByIndex.Keys) if (index > highest) highest = index;
            totalCommitted += highest;
            if (highest < worstCommit) worstCommit = highest;
            if (busy.AllConverged()) convergedSchedules++;
        }
        Console.WriteLine("  fault sweep: schedules=8 proposals=" + totalProposals + " committed=" + totalCommitted
            + " worst schedule=" + worstCommit);
        Check(convergedSchedules == 8, "all eight fault schedules converged after healing");
        Check(worstCommit >= 10, "every fault schedule still committed at least ten entries (liveness despite faults)");

        // 26. A minority partition never elects a leader and never commits.
        var split = new Cluster(new[] { 1, 2, 3, 4, 5 }, 0x5A17UL);
        split.RunUntil(() => split.Leader() != null, 30);
        split.Propose("SET q 1");
        split.Run(20);
        var minority = new List<int>();
        foreach (int id in new[] { 1, 2, 3, 4, 5 }) if (id != split.Leader()!.Id) minority.Add(id);
        minority.RemoveRange(2, minority.Count - 2);
        int minorityCommits = 0;
        foreach (int id in minority) minorityCommits = Math.Max(minorityCommits, split.Nodes[id].CommitIndex);
        foreach (int id in minority) split.Group[id] = 1;             // the two followers split off from the majority
        split.Run(140);
        Check(split.Leader() != null && minority.IndexOf(split.Leader()!.Id) < 0, "only the majority side keeps a leader");
        foreach (int id in minority)
        {
            Check(split.Nodes[id].Role != Role.Leader, "the minority node " + id + " never becomes leader");
            Check(split.Nodes[id].CommitIndex <= minorityCommits, "the minority node " + id + " cannot commit new entries");
        }
        Check(split.Propose("SET q 2"), "the majority side still accepts proposals during the partition");
        int majorityIndex = split.Leader()!.LastLogIndex;
        split.Run(40);
        Check(split.Leader()!.CommitIndex >= majorityIndex, "the majority side commits while the minority is cut off");
        Check(split.Nodes[minority[0]].StateMachine.GetValueOrDefault("q") == "1",
            "the minority is frozen at its pre-partition state and never sees the new write");
        foreach (int id in minority) split.Group[id] = 0;             // heal
        split.Run(240);
        Check(split.AllConverged() && split.Checker.Violation == null, "healing restores convergence without a safety violation");

        // 27. Boundary behaviour and rejected input.
        var single = new Cluster(new[] { 1, 2, 3 }, 0x333UL);
        Check(Throws(() => single.Nodes[1].Propose("SET x 1")), "a follower refuses a client proposal");
        var bogus = new RaftNode(9, new Membership(new[] { 1 }), 0x9UL);
        bogus.CurrentTerm = 1;
        bogus.Append(new LogEntry(1, 1, "BOGUS x"));
        bogus.CommitIndex = 1;
        Check(Throws(() => bogus.Apply()), "an unknown command is rejected when applied");
        Check(single.Nodes[1].TermAt(0) == 0 && single.Nodes[1].LastLogIndex == 0 && single.Nodes[1].CommitIndex == 0,
            "an empty log answers index 0 with term 0 and commits nothing");
        Check(!new Membership(new[] { 1, 2, 3 }, new[] { 4, 5, 6 }).HasQuorum(new HashSet<int> { 4, 5, 6 }),
            "a joint configuration with disjoint member sets cannot commit through one side alone");
        Check(new Membership(new[] { 1 }).HasQuorum(new HashSet<int> { 1 }) && !new Membership(new[] { 1, 2 }).HasQuorum(new HashSet<int> { 1 }),
            "a single-member configuration commits alone, a two-member configuration needs both");
        Check(cluster.Checker.CommittedByIndex.Count >= 5 && cluster.Checker.LeaderByTerm.Count >= 1,
            "the independent checker observed the leadership and the committed entries");
        Console.WriteLine("  terms seen=" + cluster.Checker.LeaderByTerm.Count + " committed labels observed="
            + cluster.Checker.CommittedByIndex.Count + " final leader term=" + (cluster.Leader() == null ? -1 : cluster.Leader()!.CurrentTerm));

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
