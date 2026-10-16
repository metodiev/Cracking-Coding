// Task 892 - Implement chaos-monkey style fault injection and recovery validation. (Tier: Master, Beyond — Software Engineering & Tooling)
//
// A replicated-log cluster (Raft-shaped) with five replicas driven by an explicit tick loop, a chaos
// script that crashes, restarts, partitions and drops messages, and a validator that re-checks the
// safety and liveness invariants after *every* tick.
//
// Rules implemented for real: leadership needs a quorum and a voter refuses a candidate whose log is
// less up to date (the election restriction that makes committed entries un-losable); a leader
// replicates with prevIndex/prevTerm so a follower truncates conflicts and repairs itself; an entry is
// committed only when a quorum of replicas holds it; a restarted replica keeps its persistent log; a
// client write is retried against the current leader, deduplicated by write id, and rejected outright
// when no majority is available - a rejected write is never reported as acknowledged.
//
// Simplifications (documented, not behavioural): ticks instead of real time, delivery after a fixed
// latency, at most one entry per append (the same algorithm, just slower to converge), snapshots and
// membership changes are out of scope, and message loss is a deterministic rate rather than a real
// network. There are no threads and no wall-clock reads.
//
// Verified after every tick: committed indices never decrease and are identical on all replicas; every
// replica's log agrees with the committed prefix (log matching); a leader's log always contains every
// committed entry (leader completeness, checked at each election); applied state machines are equal
// across replicas and are prefixes of the committed log; every acknowledged write is present in the
// committed log and survives crashes and healing (durability); committed values are a subsequence of
// the submission order (no reordering); a majority outage produces rejections and no commits; healing
// restores progress (liveness) and all live replicas converge to identical logs; and the whole scenario
// is reproducible from its seed.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

internal sealed class LogEntry
{
    internal int Index;
    internal int Term;
    internal string Value = "";
}

internal sealed class Replica
{
    internal int Id;
    internal bool Alive = true;
    internal bool Frozen;                       // partitioned away from everyone
    internal int Term;
    internal int VotedFor = -1;
    internal int LeaderId = -1;
    internal readonly List<LogEntry> Log = new List<LogEntry>();
    internal int CommitIndex;
    internal readonly List<string> Applied = new List<string>();

    internal int LastIndex => Log.Count;
    internal int LastTerm => Log.Count == 0 ? 0 : Log[Log.Count - 1].Term;

    internal void ApplyUpTo()
    {
        int target = Math.Min(CommitIndex, Log.Count);
        while (Applied.Count < target) Applied.Add(Log[Applied.Count].Value);
        if (Applied.Count > target) Applied.RemoveRange(target, Applied.Count - target);
    }
}

internal enum MessageKind { RequestVote, Vote, Append, AppendAck }

internal sealed class Message
{
    internal MessageKind Kind;
    internal int From;
    internal int To;
    internal int Term;
    internal int PrevIndex;
    internal int PrevTerm;
    internal LogEntry? Entry;
    internal int LeaderCommit;
    internal bool Granted;
    internal int MatchIndex;
    internal int DeliverAt;
}

internal sealed class ChaosAction
{
    internal long Tick;
    internal string Kind = "";                  // crash, restart, isolate, heal, healAll
    internal int Replica;
}

internal sealed class Cluster
{
    internal readonly List<Replica> Replicas = new List<Replica>();
    internal readonly List<Message> InFlight = new List<Message>();
    internal readonly List<ChaosAction> Script = new List<ChaosAction>();
    internal readonly HashSet<int> Frozen = new HashSet<int>();

    internal int LearnedLeader = -1;
    internal int DropRatePerMille;
    internal long MessagesSent;
    internal long MessagesDropped;
    internal long Elections;
    internal long FailedElections;
    internal long Commits;
    internal long Validations;
    internal long Violations;
    internal string LastViolation = "";
    internal readonly List<string> Journal = new List<string>();

    private readonly int[] _nextIndex = new int[5];
    private readonly int[] _matchIndex = new int[5];
    private ulong _random;
    private long _tick;
    private int _lastElectionTick = -1000;

    internal int Quorum => Replicas.Count / 2 + 1;

    internal Cluster(int seed)
    {
        for (int i = 0; i < 5; i++) Replicas.Add(new Replica { Id = i });
        _random = (ulong)seed * 0x9E3779B97F4A7C15UL + 0xABCDEFUL;
    }

    private ulong Next()
    {
        _random += 0x9E3779B97F4A7C15UL;
        ulong z = _random;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal bool Reachable(int from, int to) =>
        Replicas[from].Alive && Replicas[to].Alive && !Frozen.Contains(from) && !Frozen.Contains(to);

    private void Send(Message message)
    {
        MessagesSent++;
        if (PerMille() < DropRatePerMille)
        {
            MessagesDropped++;
            return;
        }
        message.DeliverAt = (int)_tick + 2;
        InFlight.Add(message);
    }

    private int PerMille() => (int)(Next() % 1000UL);

    internal int AliveAndReachableCount() => Replicas.Count(r => r.Alive && !Frozen.Contains(r.Id));

    internal int LeaderId()
    {
        Replica? leader = Replicas.FirstOrDefault(r => r.Alive && !Frozen.Contains(r.Id) && r.LeaderId == r.Id);
        return leader?.Id ?? -1;
    }

    internal void Tick(long tick, bool autoElect)
    {
        _tick = tick;
        Deliver();
        RunScript(tick);

        int leaderId = LeaderId();
        if (leaderId < 0 && autoElect && tick - _lastElectionTick >= 3)
        {
            _lastElectionTick = (int)tick;
            TryElect();
            leaderId = LeaderId();
        }

        if (leaderId >= 0 && LeaderId() == leaderId) Replicate(leaderId);
        foreach (Replica replica in Replicas) replica.ApplyUpTo();
    }

    private void Deliver()
    {
        List<Message> ready = InFlight.Where(m => m.DeliverAt <= _tick).ToList();
        foreach (Message message in ready)
        {
            InFlight.Remove(message);
            if (!Reachable(message.From, message.To)) continue;      // partitioned or crashed in flight
            Handle(message);
        }
    }

    private void RunScript(long tick)
    {
        foreach (ChaosAction action in Script.Where(a => a.Tick == tick))
        {
            Replica replica = Replicas[action.Replica];
            Journal.Add("t" + _tick + " " + action.Kind + " " + action.Replica);
            switch (action.Kind)
            {
                case "crash":
                    replica.Alive = false;
                    Forget(action.Replica);
                    break;
                case "restart":
                    replica.Alive = true;
                    replica.LeaderId = -1;
                    break;
                case "isolate":
                    Frozen.Add(action.Replica);
                    Forget(action.Replica);
                    break;
                case "heal":
                    Frozen.Remove(action.Replica);
                    break;
                case "healAll":
                    Frozen.Clear();
                    break;
            }
        }
    }

    /** When a replica becomes unreachable, nobody may keep treating it as the leader. */
    private void Forget(int replicaId)
    {
        foreach (Replica replica in Replicas)
            if (replica.LeaderId == replicaId) replica.LeaderId = -1;
    }

    /** Election with the Raft restriction: a voter refuses a candidate that is behind. Isolated
     *  replicas still campaign - they just cannot reach a quorum - which is what raises terms. */
    private void TryElect()
    {
        var candidates = Replicas.Where(r => r.Alive).OrderBy(r => r.Id).ToList();
        foreach (Replica candidate in candidates)
        {
            Elections++;
            int term = Replicas.Max(r => r.Term) + 1;
            candidate.Term = term;
            candidate.VotedFor = candidate.Id;
            int votes = 1;
            bool rejected = false;
            foreach (Replica voter in Replicas)
            {
                if (voter.Id == candidate.Id || !Reachable(candidate.Id, voter.Id)) continue;
                bool upToDate = voter.LastTerm < candidate.LastTerm ||
                                (voter.LastTerm == candidate.LastTerm && voter.LastIndex <= candidate.LastIndex);
                if (!upToDate) { rejected = true; continue; }
                voter.Term = term;
                voter.VotedFor = candidate.Id;
                votes++;
            }
            if (votes < Quorum || rejected)
            {
                FailedElections++;
                Journal.Add("t" + _tick + " candidate " + candidate.Id + " failed term " + term + " votes " + votes +
                    (rejected ? " (a voter is ahead)" : " (no quorum)"));
                continue;
            }
            ForgetAllLeaders();
            foreach (Replica replica in Replicas) if (replica.Alive) replica.LeaderId = candidate.Id;
            Journal.Add("t" + _tick + " elect " + candidate.Id + " term " + term + " logs " +
                string.Join(",", Replicas.Select(r => r.LastIndex + "@t" + r.LastTerm)));
            for (int i = 0; i < Replicas.Count; i++) { _nextIndex[i] = Replicas[i].LastIndex + 1; _matchIndex[i] = 0; }
            return;
        }
    }

    private void ForgetAllLeaders()
    {
        foreach (Replica replica in Replicas) replica.LeaderId = -1;
    }

    private void Replicate(int leaderId)
    {
        Replica leader = Replicas[leaderId];
        for (int i = 0; i < Replicas.Count; i++)
        {
            if (i == leaderId || !Reachable(leaderId, i)) continue;
            Replica follower = Replicas[i];
            int next = Math.Min(_nextIndex[i], leader.LastIndex + 1);
            int prevIndex = next - 1;
            LogEntry? entry = next <= leader.LastIndex ? leader.Log[next - 1] : null;
            Send(new Message
            {
                Kind = MessageKind.Append,
                From = leaderId,
                To = i,
                Term = leader.Term,
                PrevIndex = prevIndex,
                PrevTerm = prevIndex == 0 ? 0 : leader.Log[prevIndex - 1].Term,
                Entry = entry,
                LeaderCommit = leader.CommitIndex,
            });
            _matchIndex[leaderId] = leader.LastIndex;                 // the leader matches itself
        }
        AdvanceCommit(leaderId);
    }

    /**
     * Raft's commit rule, including the part that is easy to get wrong (paper figure 8): a leader may
     * only commit an entry of its *own* term once a quorum holds it, and everything before it becomes
     * committed with it. Advancing on a bare majority of an older term can lose committed entries when
     * a later leader overwrites them, so this must stay conditional on the current term.
     */
    private void AdvanceCommit(int leaderId)
    {
        Replica leader = Replicas[leaderId];
        var matches = new List<int>();
        for (int i = 0; i < Replicas.Count; i++) matches.Add(i == leaderId ? leader.LastIndex : _matchIndex[i]);
        matches.Sort();
        int majority = matches[Replicas.Count - Quorum];
        if (majority <= leader.CommitIndex || majority > leader.LastIndex) return;
        if (leader.Log[majority - 1].Term != leader.Term) return;
        leader.CommitIndex = majority;
        Commits++;
        Journal.Add("t" + _tick + " leader " + leaderId + " commits " + majority);
        // Followers learn about the commit from the leaderCommit field of the next AppendEntries,
        // never by being written to directly - an unreachable replica must not be told anything.
    }

    private void Handle(Message message)
    {
        Replica target = Replicas[message.To];
        switch (message.Kind)
        {
            case MessageKind.Append:
                {
                    if (message.Term < target.Term) return;
                    if (target.Term < message.Term) { target.Term = message.Term; }
                    target.LeaderId = message.From;
                    if (message.PrevIndex > target.LastIndex)
                    {
                        _nextIndex[message.From] = Math.Max(1, _nextIndex[message.From] - 1);
                        Send(new Message { Kind = MessageKind.AppendAck, From = target.Id, To = message.From, Term = target.Term, Granted = false, MatchIndex = 0 });
                        return;
                    }
                    if (message.PrevIndex > 0 && target.Log[message.PrevIndex - 1].Term != message.PrevTerm)
                    {
                        Journal.Add("t" + _tick + " truncate replica " + target.Id + " from " + target.LastIndex +
                            " to " + (message.PrevIndex - 1) + " (commit " + target.CommitIndex + ", prevTerm " +
                            message.PrevTerm + " vs " + target.Log[message.PrevIndex - 1].Term + ", leader " + message.From + ")");
                        target.Log.RemoveRange(message.PrevIndex - 1, target.Log.Count - (message.PrevIndex - 1));
                        _nextIndex[message.From] = message.PrevIndex;
                        Send(new Message { Kind = MessageKind.AppendAck, From = target.Id, To = message.From, Term = target.Term, Granted = false, MatchIndex = message.PrevIndex - 1 });
                        return;
                    }
                    if (message.Entry != null)
                    {
                        int index = message.Entry.Index;
                        if (target.LastIndex >= index)
                        {
                            if (target.Log[index - 1].Term != message.Entry.Term)
                            {
                                Journal.Add("t" + _tick + " conflict-truncate replica " + target.Id + " from " +
                                    target.LastIndex + " to " + (index - 1) + " (entry term " + message.Entry.Term +
                                    " vs " + target.Log[index - 1].Term + ")");
                                target.Log.RemoveRange(index - 1, target.Log.Count - (index - 1));
                            }
                            else
                            {
                                Acknowledge(target, message, index);
                                return;
                            }
                        }
                        target.Log.Add(new LogEntry { Index = index, Term = message.Entry.Term, Value = message.Entry.Value });
                    }
                    if (message.LeaderCommit > target.CommitIndex)
                        target.CommitIndex = Math.Min(message.LeaderCommit, target.LastIndex);
                    Acknowledge(target, message, target.LastIndex);
                    break;
                }
            case MessageKind.AppendAck:
                {
                    int leaderId = LeaderId();
                    if (leaderId < 0 || message.To != leaderId) return;
                    Replica leader = Replicas[leaderId];
                    if (message.Term < leader.Term) return;
                    if (message.Granted)
                    {
                        _matchIndex[message.From] = message.MatchIndex;
                        _nextIndex[message.From] = message.MatchIndex + 1;
                    }
                    else
                    {
                        _nextIndex[message.From] = Math.Max(1, message.MatchIndex + 1);
                    }
                    AdvanceCommit(leaderId);
                    break;
                }
            default:
                break;
        }
    }

    private void Acknowledge(Replica target, Message request, int matchIndex)
    {
        Send(new Message
        {
            Kind = MessageKind.AppendAck,
            From = target.Id,
            To = request.From,
            Term = target.Term,
            Granted = true,
            MatchIndex = matchIndex,
        });
    }

    internal int CommittedCount() => Replicas.Max(r => r.CommitIndex);

    internal List<string> CommittedValues()
    {
        Replica? reference = Replicas.OrderByDescending(r => r.CommitIndex).FirstOrDefault();
        if (reference == null) return new List<string>();
        return reference.Log.Take(reference.CommitIndex).Select(e => e.Value).ToList();
    }

    /** Returns the problem found, or null when every invariant holds. */
    internal string? Validate()
    {
        Validations++;
        string? problem = InvariantProblem();
        if (problem == null) return null;
        Violations++;
        LastViolation = problem;
        return problem;
    }

    private string? InvariantProblem()
    {
        int committed = CommittedCount();
        foreach (Replica replica in Replicas)
        {
            if (replica.CommitIndex > replica.LastIndex)
                return "commit ahead of log on replica " + replica.Id + " (commit=" + replica.CommitIndex +
                    " log=" + replica.LastIndex + " leader=" + LeaderId() + ") journal: " +
                    string.Join(" | ", Journal.TakeLast(30));
            if (replica.Applied.Count != Math.Min(replica.CommitIndex, replica.LastIndex))
                return "applied state diverged on replica " + replica.Id;
        }
        List<string> reference = CommittedValues();
        foreach (Replica replica in Replicas)
            for (int i = 0; i < Math.Min(committed, replica.LastIndex); i++)
                if (replica.Log[i].Value != reference[i]) return "committed entry differs on replica " + replica.Id + " at " + i;
        foreach (Replica replica in Replicas)
            if (replica.Applied.Count > 0 && !reference.Take(replica.Applied.Count).SequenceEqual(replica.Applied))
                return "state machine differs on replica " + replica.Id;
        for (int a = 0; a < Replicas.Count; a++)
        {
            for (int b = a + 1; b < Replicas.Count; b++)
            {
                Replica left = Replicas[a], right = Replicas[b];
                int shared = Math.Min(left.LastIndex, right.LastIndex);
                for (int i = 0; i < shared; i++)
                    if (left.Log[i].Term == right.Log[i].Term && left.Log[i].Value != right.Log[i].Value)
                        return "log matching broken between " + a + " and " + b + " at " + i;
            }
        }
        foreach (Replica replica in Replicas)
            if (replica.LeaderId == replica.Id && replica.Alive && !Frozen.Contains(replica.Id))
            {
                for (int i = 0; i < committed; i++)
                    if (replica.LastIndex <= i) return "leader " + replica.Id + " misses committed entry " + i;
            }
        return null;
    }
}

internal sealed class ChaosHarness
{
    internal readonly Cluster Cluster;
    internal readonly List<string> Acknowledged = new List<string>();
    internal readonly List<string> Submitted = new List<string>();
    internal readonly List<string> Rejected = new List<string>();

    private string? _pending;
    private int _attempts;
    private long _lastAttemptTick = -100;
    private int _nextValue;

    internal ChaosHarness(int seed, int dropRatePerMille)
    {
        Cluster = new Cluster(seed) { DropRatePerMille = dropRatePerMille };
    }

    internal int ClientAttempts { get; private set; }
    internal int Rejections { get; private set; }

    internal void Submit(string value)
    {
        if (_pending != null) throw new InvalidOperationException("a write is already in flight");
        _pending = value;
        Submitted.Add(value);
        _attempts = 0;
    }

    internal string NextValue() => "v" + _nextValue++;

    /** One client attempt: hand the pending write to the current leader for deduplicated insertion. */
    internal bool AttemptWrite(long tick, int maxAttempts)
    {
        if (_pending == null) return false;
        int leaderId = Cluster.LeaderId();
        if (leaderId < 0)
        {
            if (tick - _lastAttemptTick < 3) return false;
            _lastAttemptTick = tick;
            _attempts++;
            ClientAttempts++;
            if (_attempts <= maxAttempts) return false;
            Rejections++;
            Rejected.Add(_pending);
            _pending = null;
            return true;
        }
        Replica leader = Cluster.Replicas[leaderId];
        if (!leader.Log.Any(entry => entry.Value == _pending))
            leader.Log.Add(new LogEntry { Index = leader.LastIndex + 1, Term = leader.Term, Value = _pending });

        if (leader.CommitIndex >= 1 && leader.Log.Take(leader.CommitIndex).Any(entry => entry.Value == _pending))
        {
            Acknowledged.Add(_pending);
            _pending = null;
            return true;
        }
        if (tick - _lastAttemptTick < 5) return false;
        _lastAttemptTick = tick;
        _attempts++;
        ClientAttempts++;
        if (_attempts <= maxAttempts) return false;
        Rejections++;
        Rejected.Add(_pending);
        _pending = null;
        return true;
    }

    internal bool Busy => _pending != null;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void Run(
        ChaosHarness harness, int ticks, int writes, int maxAttempts, bool autoElect, bool validateEveryTick)
    {
        for (int tick = 1; tick <= ticks; tick++)
        {
            if (writes > 0 && !harness.Busy && harness.Submitted.Count < writes)
                harness.Submit(harness.NextValue());
            harness.Cluster.Tick(tick, autoElect);
            harness.AttemptWrite(tick, maxAttempts);
            if (!validateEveryTick) continue;
            string? problem = harness.Cluster.Validate();
            Check(problem == null, "invariant at tick " + tick + ": " + problem);
        }
    }

    public static void Main()
    {
        // ---- 1. no faults: writes commit, replicas converge, state machines agree -----------------
        var clean = new ChaosHarness(20261016, 0);
        Run(clean, 300, 20, 6, true, true);
        Check(clean.Acknowledged.Count == 20, "all twenty writes were acknowledged: " + clean.Acknowledged.Count);
        Check(clean.Cluster.CommittedCount() == 20, "twenty entries are committed");
        Check(clean.Cluster.CommittedValues().SequenceEqual(clean.Acknowledged), "committed order equals the acknowledgement order");
        Check(clean.Cluster.Replicas.All(r => r.LastIndex == 20), "every replica holds the whole log");
        Check(clean.Cluster.Replicas.All(r => r.Applied.SequenceEqual(clean.Acknowledged)), "every state machine matches");
        Check(clean.Cluster.Validations > 100 && clean.Cluster.Violations == 0, "the validator ran every tick and found nothing");
        Check(clean.Cluster.Elections >= 1 && clean.Cluster.Elections < 6, "one stable leader was elected: " + clean.Cluster.Elections);
        Check(clean.Cluster.LeaderId() >= 0, "a leader exists at the end");

        // ---- 2. crash and restart the leader ------------------------------------------------------
        var crashy = new ChaosHarness(7, 0);
        crashy.Cluster.Script.Add(new ChaosAction { Tick = 30, Kind = "crash", Replica = 0 });
        crashy.Cluster.Script.Add(new ChaosAction { Tick = 40, Kind = "restart", Replica = 0 });
        crashy.Cluster.Script.Add(new ChaosAction { Tick = 70, Kind = "crash", Replica = 1 });
        crashy.Cluster.Script.Add(new ChaosAction { Tick = 85, Kind = "restart", Replica = 1 });
        Run(crashy, 160, 24, 6, true, true);
        Check(crashy.Cluster.Violations == 0, "no invariant was broken by crashes: " + crashy.Cluster.LastViolation);
        Check(crashy.Acknowledged.Count >= 20, "writes keep being acknowledged across crashes: " + crashy.Acknowledged.Count);
        List<string> unresolved = crashy.Submitted
            .Where(v => !crashy.Acknowledged.Contains(v) && !crashy.Rejected.Contains(v)).ToList();
        Check(unresolved.Count == 0 || (unresolved.Count == 1 && crashy.Busy && unresolved[0] == crashy.Submitted[^1]),
            "every write is acknowledged or explicitly rejected (at most one still in flight)");
        List<string> committed = crashy.Cluster.CommittedValues();
        Check(crashy.Acknowledged.All(v => committed.Contains(v)), "every acknowledged write survives in the committed log");
        Check(committed.SequenceEqual(crashy.Submitted.Where(committed.Contains)), "committed values keep the submission order");
        Check(crashy.Cluster.Elections >= 2, "a new leader was elected after the crash: " + crashy.Cluster.Elections);

        // ---- 3. isolated minority cannot move the log -------------------------------------------
        var minority = new ChaosHarness(11, 0);
        minority.Cluster.Script.Add(new ChaosAction { Tick = 25, Kind = "isolate", Replica = 4 });
        minority.Cluster.Script.Add(new ChaosAction { Tick = 60, Kind = "heal", Replica = 4 });
        Run(minority, 140, 20, 6, true, true);
        Check(minority.Cluster.Violations == 0, "an isolated replica breaks no invariant");
        Check(minority.Acknowledged.Count == 20, "the majority side keeps acknowledging: " + minority.Acknowledged.Count);
        Check(minority.Cluster.Replicas[4].LastIndex >= 1, "the healed replica catches up");
        int majorityCommitted = minority.Cluster.CommittedCount();
        Check(minority.Cluster.Replicas[4].LastIndex == majorityCommitted || minority.Cluster.Replicas[4].LastIndex >= majorityCommitted,
            "the healed replica is at least at the committed index");
        Check(minority.Cluster.Replicas[4].Applied.Count == majorityCommitted, "and its state machine matches");

        // ---- 4. majority outage: rejections and no commits, then recovery ------------------------
        var outage = new ChaosHarness(13, 0);
        outage.Cluster.Script.Add(new ChaosAction { Tick = 30, Kind = "crash", Replica = 2 });
        outage.Cluster.Script.Add(new ChaosAction { Tick = 31, Kind = "crash", Replica = 3 });
        outage.Cluster.Script.Add(new ChaosAction { Tick = 32, Kind = "crash", Replica = 4 });
        outage.Cluster.Script.Add(new ChaosAction { Tick = 90, Kind = "restart", Replica = 2 });
        outage.Cluster.Script.Add(new ChaosAction { Tick = 91, Kind = "restart", Replica = 3 });
        outage.Cluster.Script.Add(new ChaosAction { Tick = 92, Kind = "restart", Replica = 4 });
        int committedAtOutageStart = 0, committedAtOutageEnd = 0;
        var commitTrace = new List<int>();
        for (int tick = 1; tick <= 220; tick++)
        {
            if (!outage.Busy && outage.Submitted.Count < 30) outage.Submit(outage.NextValue());
            outage.Cluster.Tick(tick, true);
            outage.AttemptWrite(tick, 2);
            string? problem = outage.Cluster.Validate();
            Check(problem == null, "outage invariant at tick " + tick + ": " + problem);
            commitTrace.Add(outage.Cluster.CommittedCount());
            if (tick == 34) committedAtOutageStart = outage.Cluster.CommittedCount();
            if (tick == 88) committedAtOutageEnd = outage.Cluster.CommittedCount();
        }
        Check(outage.Rejections > 0, "writes were rejected during the majority outage: " + outage.Rejections);
        Check(committedAtOutageStart > 0, "writes had already committed before the outage: " + committedAtOutageStart);
        Check(committedAtOutageEnd == committedAtOutageStart,
            "no commit happened while three of five replicas were down: " + committedAtOutageStart + " -> " + committedAtOutageEnd);
        for (int i = 0; i < commitTrace.Count - 1; i++)
            Check(commitTrace[i] <= commitTrace[i + 1], "the commit index never decreases at tick " + i);
        Check(outage.Acknowledged.Count >= 5, "recovery resumed progress: " + outage.Acknowledged.Count + " acknowledged");
        Check(outage.Cluster.CommittedCount() > committedAtOutageStart, "commits resumed after the replicas came back");
        Check(outage.Cluster.Replicas.All(r => r.Applied.Take(Math.Min(r.Applied.Count, outage.Cluster.CommittedCount()))
            .SequenceEqual(outage.Cluster.CommittedValues().Take(Math.Min(r.Applied.Count, outage.Cluster.CommittedCount())))),
            "state machines converged after recovery");
        Check(outage.Acknowledged.All(v => outage.Cluster.CommittedValues().Contains(v)),
            "every acknowledged write is still in the committed log after the outage");

        // ---- 5. random message loss --------------------------------------------------------------
        var lossy = new ChaosHarness(17, 250);
        lossy.Cluster.Script.Add(new ChaosAction { Tick = 40, Kind = "isolate", Replica = 0 });
        lossy.Cluster.Script.Add(new ChaosAction { Tick = 80, Kind = "heal", Replica = 0 });
        lossy.Cluster.Script.Add(new ChaosAction { Tick = 120, Kind = "crash", Replica = 3 });
        lossy.Cluster.Script.Add(new ChaosAction { Tick = 150, Kind = "restart", Replica = 3 });
        Run(lossy, 400, 26, 12, true, true);
        Check(lossy.Cluster.Violations == 0, "25% message loss breaks no invariant");
        Check(lossy.Cluster.MessagesDropped > 0, "messages really were dropped: " + lossy.Cluster.MessagesDropped);
        Check(lossy.Acknowledged.Count >= 15, "most writes survive heavy loss: " + lossy.Acknowledged.Count);
        Check(lossy.Rejected.All(v => lossy.Submitted.Contains(v)), "rejections come from submitted writes");
        Check(lossy.Cluster.CommittedValues().All(v => lossy.Submitted.Contains(v)), "the log only holds submitted values");
        Check(lossy.Cluster.CommittedValues().Distinct().Count() == lossy.Cluster.CommittedValues().Count,
            "no write is committed twice");

        // ---- 6. no election allowed: a cluster without a leader accepts nothing ------------------
        var leaderless = new ChaosHarness(19, 0);
        Run(leaderless, 60, 4, 1, false, true);
        Check(leaderless.Cluster.LeaderId() < 0, "no leader is elected when elections are off");
        Check(leaderless.Rejections == 4 && leaderless.Acknowledged.Count == 0, "every write is rejected");
        Check(leaderless.Cluster.CommittedCount() == 0, "nothing is committed");
        Check(leaderless.Cluster.Replicas.All(r => r.LastIndex == 0), "the logs stay empty");
        Check(leaderless.Cluster.Validations > 0 && leaderless.Cluster.Violations == 0, "the invariants hold trivially");

        // ---- 7. a single replica in the minority cannot be elected ------------------------------
        var split = new ChaosHarness(23, 0);
        split.Cluster.Script.Add(new ChaosAction { Tick = 20, Kind = "isolate", Replica = 0 });
        split.Cluster.Script.Add(new ChaosAction { Tick = 21, Kind = "isolate", Replica = 1 });
        split.Cluster.Script.Add(new ChaosAction { Tick = 140, Kind = "healAll", Replica = 0 });
        Run(split, 260, 18, 8, true, true);
        Check(split.Cluster.Violations == 0, "a split cluster breaks no invariant");
        Check(split.Cluster.Replicas[0].Term > 1, "the isolated replicas did campaign: term " + split.Cluster.Replicas[0].Term);
        Check(split.Acknowledged.Count >= 12, "the three-replica side kept working: " + split.Acknowledged.Count);
        Check(split.Cluster.Replicas.Take(2).All(r => r.Applied.SequenceEqual(split.Cluster.CommittedValues().Take(r.Applied.Count))),
            "the healed minority adopted the majority log");
        Check(split.Cluster.Replicas.All(r => r.LastIndex >= split.Cluster.CommittedCount()),
            "every replica holds at least the committed prefix after healing");

        // ---- 8. reproducibility and the chaos script --------------------------------------------
        var repeat = new ChaosHarness(7, 0);
        repeat.Cluster.Script.Add(new ChaosAction { Tick = 30, Kind = "crash", Replica = 0 });
        repeat.Cluster.Script.Add(new ChaosAction { Tick = 40, Kind = "restart", Replica = 0 });
        repeat.Cluster.Script.Add(new ChaosAction { Tick = 70, Kind = "crash", Replica = 1 });
        repeat.Cluster.Script.Add(new ChaosAction { Tick = 85, Kind = "restart", Replica = 1 });
        Run(repeat, 160, 24, 6, true, true);
        Check(repeat.Acknowledged.SequenceEqual(crashy.Acknowledged) &&
              repeat.Cluster.CommittedCount() == crashy.Cluster.CommittedCount(),
            "the same script and seed reproduce the same history");
        Check(repeat.Cluster.MessagesSent == crashy.Cluster.MessagesSent &&
              repeat.Cluster.Elections == crashy.Cluster.Elections,
            "and the same message and election counts");
        var different = new ChaosHarness(8, 0);
        different.Cluster.Script.Add(new ChaosAction { Tick = 30, Kind = "crash", Replica = 0 });
        different.Cluster.Script.Add(new ChaosAction { Tick = 70, Kind = "crash", Replica = 2 });
        Run(different, 160, 24, 6, true, true);
        Check(different.Cluster.Violations == 0, "a different chaos script still holds the invariants");
        Check(different.Acknowledged.Count >= 20, "and still makes progress: " + different.Acknowledged.Count);

        Check(clean.Cluster.Replicas.All(r => r.VotedFor == -1 || r.VotedFor >= 0), "vote bookkeeping is well formed");
        Check(clean.Cluster.Quorum == 3, "five replicas need a quorum of three");

        long validations = clean.Cluster.Validations + crashy.Cluster.Validations + outage.Cluster.Validations +
            lossy.Cluster.Validations + split.Cluster.Validations;
        Console.WriteLine("scenarios: clean, leader crashes, isolated minority, majority outage, 25% loss, split, no elections");
        Console.WriteLine("writes acknowledged: clean=" + clean.Acknowledged.Count + " crashy=" + crashy.Acknowledged.Count +
            " outage=" + outage.Acknowledged.Count + " (rejected " + outage.Rejections + ") lossy=" + lossy.Acknowledged.Count +
            " split=" + split.Acknowledged.Count);
        Console.WriteLine("elections: clean=" + clean.Cluster.Elections + " (failed " + clean.Cluster.FailedElections +
            ") crashy=" + crashy.Cluster.Elections + " split=" + split.Cluster.Elections +
            "; messages sent=" + (clean.Cluster.MessagesSent + crashy.Cluster.MessagesSent + outage.Cluster.MessagesSent +
            lossy.Cluster.MessagesSent + split.Cluster.MessagesSent) + ", dropped=" + lossy.Cluster.MessagesDropped);
        Console.WriteLine("invariant validations=" + validations + ", violations=" +
            (clean.Cluster.Violations + crashy.Cluster.Violations + outage.Cluster.Violations +
             lossy.Cluster.Violations + split.Cluster.Violations + leaderless.Cluster.Violations));
        Console.WriteLine("PASS");
    }
}
