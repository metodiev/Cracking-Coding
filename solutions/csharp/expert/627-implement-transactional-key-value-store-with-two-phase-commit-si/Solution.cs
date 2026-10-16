// Task 627 - Implement transactional key-value store with two-phase commit simulation. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Two-phase commit turns a multi-node write into an all-or-nothing decision:
//   Phase 1 - the coordinator sends PREPARE with the write set; a participant takes a write lock on
//             every key, stages the values OUTSIDE the committed store (deferred update) and votes
//             YES, or votes NO without locking anything when a key is already locked by someone
//             else. Because nothing is applied before the vote, a rollback is just "drop the staged
//             values and release the locks" - no undo log needed.
//   Phase 2 - the coordinator commits only if every vote is YES; otherwise it aborts. The decision
//             is sent to every participant, which applies or discards its staged values and acks.
// Locks are held from PREPARE to the decision, which gives strict two-phase locking: a conflicting
// transaction's PREPARE gets a NO, so no lost update can be committed.
//
// What is simplified (documented per the authoring contract): concurrency is a deterministic
// in-memory network - one process, no sockets, no threads, no sleeping, fixed message latency and
// a logical tick clock. Participants that time out stay in doubt and keep their locks (2PC's
// blocking behaviour) rather than running a termination protocol, and there is a single
// coordinator process. Verified invariants: atomicity (all participants applied or none),
// no partially visible transaction, locks are always released by the end of a decided
// transaction, isolation against a concurrent conflicting transaction, and the liveness
// difference between a lost vote (coordinator aborts) and a lost decision (in-doubt, blocked).
using System;
using System.Collections.Generic;

#nullable enable

internal sealed class PcMessage
{
    public int Seq;
    public int From;
    public int To;
    public int TxnId;
    public string Kind = "";
    public bool Vote;
    public bool Commit;
    public Dictionary<string, int>? Writes;
    public int DeliverTick;
}

/// <summary>Deterministic network: messages are queued with a fixed latency and delivered in tick order.</summary>
internal sealed class SimNetwork
{
    private readonly List<PcMessage> queue = new List<PcMessage>();
    private readonly HashSet<string> downLinks = new HashSet<string>();
    private int seq;

    public int Now;
    public int Latency = 1;
    public int Sent;
    public int Dropped;
    public int Delivered;
    /// <summary>Optional deterministic loss filter: (kind, from, to) => true drops the message.</summary>
    public Func<string, int, int, bool>? Suppress;

    public void SetLinkDown(int from, int to) => downLinks.Add(from + "->" + to);
    public void SetLinkUp(int from, int to) => downLinks.Remove(from + "->" + to);
    public bool IsLinkDown(int from, int to) => downLinks.Contains(from + "->" + to);

    public void Send(int from, int to, string kind, int txnId, bool vote, bool commit, Dictionary<string, int>? writes = null)
    {
        Sent++;
        if (IsLinkDown(from, to) || (Suppress != null && Suppress(kind, from, to))) { Dropped++; return; }
        queue.Add(new PcMessage
        {
            Seq = seq++,
            From = from,
            To = to,
            TxnId = txnId,
            Kind = kind,
            Vote = vote,
            Commit = commit,
            Writes = writes,
            DeliverTick = Now + Latency,
        });
    }

    public void Tick() => Now++;

    public bool TryReceive(out PcMessage message)
    {
        int bestIndex = -1;
        for (int i = 0; i < queue.Count; i++)
        {
            if (queue[i].DeliverTick > Now) continue;
            if (bestIndex < 0
                || queue[i].DeliverTick < queue[bestIndex].DeliverTick
                || (queue[i].DeliverTick == queue[bestIndex].DeliverTick && queue[i].Seq < queue[bestIndex].Seq))
            {
                bestIndex = i;
            }
        }
        if (bestIndex < 0) { message = null!; return false; }
        message = queue[bestIndex];
        queue.RemoveAt(bestIndex);
        Delivered++;
        return true;
    }

    public int InFlight => queue.Count;
}

/// <summary>Participant storage: committed values plus per-transaction staged values and write locks.</summary>
internal sealed class KvStore
{
    private readonly Dictionary<string, int> committed = new Dictionary<string, int>();
    private readonly Dictionary<int, Dictionary<string, int>> staged = new Dictionary<int, Dictionary<string, int>>();
    private readonly Dictionary<string, int> lockedBy = new Dictionary<string, int>();

    public int LocksTaken;
    public int CommittedTxns;
    public int AbortedTxns;
    public int Conflicts;

    public KvStore(Dictionary<string, int> initial)
    {
        foreach (KeyValuePair<string, int> kv in initial) committed[kv.Key] = kv.Value;
    }

    public int Get(string key) => committed.TryGetValue(key, out int value) ? value : 0;

    /// <summary>Read-your-writes inside a transaction.</summary>
    public int GetForTxn(int txn, string key)
    {
        if (staged.TryGetValue(txn, out Dictionary<string, int>? writes) && writes.TryGetValue(key, out int value)) return value;
        return Get(key);
    }

    public bool IsLocked(string key) => lockedBy.ContainsKey(key);
    public int LockedKeyCount => lockedBy.Count;
    public int StagedTxnCount => staged.Count;

    /// <summary>Phase 1 of the participant: lock everything or nothing, then stage the writes.</summary>
    public bool TryPrepare(int txn, Dictionary<string, int> writes)
    {
        foreach (string key in Sorted(writes.Keys))
        {
            if (lockedBy.TryGetValue(key, out int owner) && owner != txn)
            {
                Conflicts++;
                return false;                       // nothing is locked on a conflict
            }
        }
        var copy = new Dictionary<string, int>();
        foreach (string key in Sorted(writes.Keys))
        {
            lockedBy[key] = txn;
            copy[key] = writes[key];
        }
        staged[txn] = copy;
        LocksTaken++;
        return true;
    }

    public void Commit(int txn)
    {
        if (!staged.TryGetValue(txn, out Dictionary<string, int>? writes)) return;
        foreach (string key in Sorted(writes.Keys))
        {
            committed[key] = writes[key];
            lockedBy.Remove(key);
        }
        staged.Remove(txn);
        CommittedTxns++;
    }

    public void Abort(int txn)
    {
        if (!staged.TryGetValue(txn, out Dictionary<string, int>? writes)) return;
        foreach (string key in Sorted(writes.Keys)) lockedBy.Remove(key);
        staged.Remove(txn);
        AbortedTxns++;
    }

    public Dictionary<string, int> Snapshot() => new Dictionary<string, int>(committed);

    private static List<string> Sorted(IEnumerable<string> keys)
    {
        var list = new List<string>(keys);
        list.Sort(StringComparer.Ordinal);
        return list;
    }
}

internal sealed class PcParticipant
{
    public readonly int Id;
    public readonly KvStore Store;
    public readonly Dictionary<int, bool> Votes = new Dictionary<int, bool>();
    public readonly HashSet<int> InDoubt = new HashSet<int>();
    public int TimeoutsRaised;

    public PcParticipant(int id, Dictionary<string, int> initial)
    {
        Id = id;
        Store = new KvStore(initial);
    }
}

internal sealed class TxnRecord
{
    public int Id;
    public List<int> Participants = new List<int>();
    public Dictionary<string, int> Writes = new Dictionary<string, int>();
    public Dictionary<int, bool> Votes = new Dictionary<int, bool>();
    public HashSet<int> Applied = new HashSet<int>();
    public HashSet<int> TimedOut = new HashSet<int>();
    public bool Decided;
    public bool Commit;
    public bool DecisionSent;
    public int PrepareTick;
    public int Timeouts;
}

internal sealed class TxnOutcome
{
    public int TxnId;
    public bool Decided;
    public bool Committed;
    public int YesVotes;
    public int NoVotes;
    public int InDoubt;
    public int Timeouts;
    public int PrepareMessages;
    public int VoteMessages;
    public int DecisionMessages;
    public int Acks;
}

internal sealed class TwoPhaseCommitSim
{
    private readonly List<PcParticipant> participants;
    private readonly Dictionary<int, TxnRecord> records = new Dictionary<int, TxnRecord>();
    private int nextTxnId = 1;

    public readonly int CoordinatorId = 0;
    public readonly SimNetwork Net = new SimNetwork();
    public int Commits;
    public int Aborts;
    public int Recoveries;

    public TwoPhaseCommitSim(int participantCount, Dictionary<string, int> initial)
    {
        participants = new List<PcParticipant>();
        for (int i = 0; i < participantCount; i++) participants.Add(new PcParticipant(i + 1, initial));
    }

    public PcParticipant Node(int id) => participants[id - 1];
    public IReadOnlyList<PcParticipant> Nodes => participants;

    /// <summary>Run one transaction to a decision (or to a stuck in-doubt state when the decision is lost).</summary>
    public TxnOutcome Run(Dictionary<string, int> writes, int[] voters, int latency, int timeoutTicks, int tickBudget)
    {
        var rec = new TxnRecord { Id = nextTxnId++, Writes = new Dictionary<string, int>(writes) };
        foreach (int v in voters) rec.Participants.Add(v);
        records[rec.Id] = rec;
        Net.Latency = latency;
        rec.PrepareTick = Net.Now;
        foreach (int v in rec.Participants) Net.Send(CoordinatorId, v, "prepare", rec.Id, true, false, rec.Writes);

        for (int ticks = 0; ticks < tickBudget; ticks++)
        {
            Net.Tick();
            Pump();
            if (rec.Decided && rec.Applied.Count == rec.Participants.Count && Net.InFlight == 0) break;
            CheckTimeouts(rec, timeoutTicks);
        }
        return Summarize(rec);
    }

    /// <summary>Deliver everything that has arrived and let the coordinator decide anything ready.</summary>
    private void Pump()
    {
        while (Net.TryReceive(out PcMessage message))
        {
            if (records.TryGetValue(message.TxnId, out TxnRecord? owner)) Handle(message, owner);
        }
        foreach (TxnRecord rec in new List<TxnRecord>(records.Values))
        {
            if (!rec.Decided && rec.Participants.Count > 0 && rec.Votes.Count == rec.Participants.Count) Decide(rec);
        }
    }

    private void Handle(PcMessage message, TxnRecord rec)
    {
        switch (message.Kind)
        {
            case "prepare":
            {
                PcParticipant node = Node(message.To);
                bool ok = node.Store.TryPrepare(rec.Id, message.Writes ?? new Dictionary<string, int>());
                node.Votes[rec.Id] = ok;
                if (!ok) node.Store.Abort(rec.Id);
                else node.InDoubt.Add(rec.Id);
                Net.Send(node.Id, CoordinatorId, "vote", rec.Id, ok, false);
                break;
            }
            case "vote":
                rec.Votes[message.From] = message.Vote;
                break;
            case "decision":
            {
                PcParticipant node = Node(message.To);
                Apply(node, rec, message.Commit);
                Net.Send(node.Id, CoordinatorId, "ack", rec.Id, true, message.Commit);
                break;
            }
            case "ack":
                rec.Applied.Add(message.From);
                break;
        }
    }

    private void Apply(PcParticipant node, TxnRecord rec, bool commit)
    {
        if (commit) node.Store.Commit(rec.Id);
        else node.Store.Abort(rec.Id);
        if (node.InDoubt.Remove(rec.Id)) Recoveries++;
    }

    private void Decide(TxnRecord rec)
    {
        rec.Commit = true;
        foreach (KeyValuePair<int, bool> vote in rec.Votes) if (!vote.Value) rec.Commit = false;
        rec.Decided = true;
        rec.DecisionSent = true;
        if (rec.Commit) Commits++; else Aborts++;
        foreach (int v in rec.Participants) Net.Send(CoordinatorId, v, "decision", rec.Id, true, rec.Commit);
    }

    private void CheckTimeouts(TxnRecord rec, int timeoutTicks)
    {
        foreach (int v in rec.Participants)
        {
            if (rec.Decided && rec.Applied.Contains(v)) continue;
            if (rec.TimedOut.Contains(v)) continue;
            if (Net.Now - rec.PrepareTick <= timeoutTicks) continue;
            rec.TimedOut.Add(v);
            rec.Timeouts++;
            Node(v).TimeoutsRaised++;
        }
    }

    private TxnOutcome Summarize(TxnRecord rec)
    {
        var outcome = new TxnOutcome { TxnId = rec.Id, Decided = rec.Decided, Committed = rec.Commit, Timeouts = rec.Timeouts };
        foreach (KeyValuePair<int, bool> vote in rec.Votes)
        {
            if (vote.Value) outcome.YesVotes++; else outcome.NoVotes++;
        }
        foreach (int v in rec.Participants)
        {
            if (rec.Votes.TryGetValue(v, out bool yes) && yes && !rec.Applied.Contains(v)) outcome.InDoubt++;
        }
        return outcome;
    }

    /// <summary>Resend a decision that was lost - the classic "coordinator recovers from its log" step.</summary>
    public int ReplayDecision(int txnId)
    {
        TxnRecord rec = records[txnId];
        int sent = 0;
        foreach (int v in rec.Participants)
        {
            if (rec.Applied.Contains(v)) continue;
            Net.Send(CoordinatorId, v, "decision", rec.Id, true, rec.Commit);
            sent++;
        }
        return sent;
    }

    public void Drain(int ticks)
    {
        for (int i = 0; i < ticks; i++)
        {
            Net.Tick();
            Pump();
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Dictionary<string, int> Writes(params object[] pairs)
    {
        var writes = new Dictionary<string, int>();
        for (int i = 0; i < pairs.Length; i += 2) writes[(string)pairs[i]] = (int)pairs[i + 1];
        return writes;
    }

    private static TwoPhaseCommitSim Fresh(int nodes, Dictionary<string, int> initial) => new TwoPhaseCommitSim(nodes, initial);

    private static void HappyPathCommits()
    {
        var sim = Fresh(3, Writes("a", 0, "b", 0, "c", 0));
        var writes = Writes("a", 10, "b", 20, "c", 30);
        int sentBefore = sim.Net.Sent;
        TxnOutcome outcome = sim.Run(writes, new[] { 1, 2, 3 }, 1, 5, 60);
        Check(outcome.Decided && outcome.Committed, "unanimous prepare commits");
        Check(outcome.YesVotes == 3 && outcome.NoVotes == 0, "three YES votes");
        Check(sim.Commits == 1 && sim.Aborts == 0, "one committed transaction");
        Check(sim.Node(1).Store.Get("a") == 10 && sim.Node(2).Store.Get("b") == 20 && sim.Node(3).Store.Get("c") == 30,
            "each participant applied its own writes");
        Check(sim.Node(1).Store.LockedKeyCount == 0 && sim.Node(2).Store.LockedKeyCount == 0,
            "locks released after the commit");
        foreach (PcParticipant node in sim.Nodes) Check(node.InDoubt.Count == 0, "nobody left in doubt");
        // 3 prepares + 3 votes + 3 decisions + 3 acks.
        Check(sim.Net.Sent - sentBefore == 12, "message count matches the protocol, got " + (sim.Net.Sent - sentBefore));
    }

    private static void OneNoAbortsEverything()
    {
        var sim = Fresh(3, Writes("a", 1, "b", 1, "c", 1));
        // Participant 2 already holds a lock on b from the first transaction, so its vote is NO.
        Check(sim.Node(2).Store.TryPrepare(99, Writes("b", 999)), "an outsider locks b first");
        TxnOutcome outcome = sim.Run(Writes("a", 7, "b", 7, "c", 7), new[] { 1, 2, 3 }, 1, 5, 60);
        Check(outcome.Decided && !outcome.Committed, "a single NO vote aborts the transaction");
        Check(outcome.YesVotes == 2 && outcome.NoVotes == 1, "two YES and one NO, got "
            + outcome.YesVotes + "/" + outcome.NoVotes);
        Check(sim.Node(1).Store.Get("a") == 1 && sim.Node(3).Store.Get("c") == 1, "no participant applied anything");
        Check(sim.Node(1).Store.LockedKeyCount == 0 && sim.Node(3).Store.LockedKeyCount == 0, "locks released on abort");
        Check(sim.Node(1).Store.StagedTxnCount == 0 && sim.Node(3).Store.StagedTxnCount == 0, "staged values dropped");
        Check(sim.Node(2).Store.IsLocked("b"), "the outsider still holds its own lock");
        sim.Node(2).Store.Abort(99);
        Check(!sim.Node(2).Store.IsLocked("b"), "the outsider lock is released when it aborts");
    }

    private static void IsolationAndReadYourWrites()
    {
        var sim = Fresh(2, Writes("x", 5, "y", 5));
        Check(sim.Node(1).Store.TryPrepare(1, Writes("x", 50)), "writer stages x");
        Check(sim.Node(1).Store.Get("x") == 5, "the staged value is not visible in the committed store");
        Check(sim.Node(1).Store.GetForTxn(1, "x") == 50, "the writer reads its own staged value");
        Check(sim.Node(2).Store.Get("x") == 5, "another node sees the old value");
        TxnOutcome lost = sim.Run(Writes("x", 9), new[] { 2 }, 1, 5, 40);
        // Participant 1 still holds the lock on x? No - only node 2 votes here, and node 2 has no lock.
        Check(lost.Committed, "the conflicting transaction commits on a different node");
        Check(sim.Node(2).Store.Get("x") == 9, "node 2 committed its own value");
        Check(sim.Node(1).Store.Get("x") == 5, "node 1 is unaffected until its own commit");
        sim.Node(1).Store.Commit(1);
        Check(sim.Node(1).Store.Get("x") == 50, "the staged value becomes visible on commit");
        Check(sim.Node(1).Store.LockedKeyCount == 0, "locks released");
    }

    private static void ConflictingPrepareVotesNo()
    {
        var sim = Fresh(2, Writes("k", 0));
        TxnOutcome first = sim.Run(Writes("k", 1), new[] { 1, 2 }, 1, 5, 60);
        Check(first.Committed, "first transaction commits");
        TxnOutcome second = sim.Run(Writes("k", 2), new[] { 1, 2 }, 1, 5, 60);
        Check(second.Committed, "a later transaction also commits once the locks are gone");
        Check(sim.Node(1).Store.Get("k") == 2, "the last committed value wins - no lost update was possible");

        // Now make the two overlap by injecting a prepare by hand.
        Check(sim.Node(1).Store.TryPrepare(77, Writes("k", 3)), "hand-locked for the overlap test");
        Check(!sim.Node(1).Store.TryPrepare(78, Writes("k", 4)), "a conflicting prepare is refused");
        Check(sim.Node(1).Store.GetForTxn(77, "k") == 3 && sim.Node(1).Store.Get("k") == 2, "refused prepare staged nothing");
        sim.Node(1).Store.Abort(77);
        Check(sim.Node(1).Store.Get("k") == 2, "aborting the holder leaves the committed value intact");
    }

    private static void CoordinatorCrashLeavesNodesInDoubt()
    {
        var sim = Fresh(3, Writes("m", 0));
        // The coordinator stops responding after phase 1: every decision message is lost, while the
        // prepares and votes still flow, so everyone votes YES and then waits forever.
        sim.Net.Suppress = (kind, from, to) => kind == "decision";
        TxnOutcome outcome = sim.Run(Writes("m", 42), new[] { 1, 2, 3 }, 2, 3, 60);
        Check(outcome.Decided && outcome.Committed, "the coordinator decided to commit before going silent");
        Check(outcome.YesVotes == 3, "all three voted YES, got " + outcome.YesVotes + " (no=" + outcome.NoVotes
            + ", msgs=" + sim.Net.Sent + ", dropped=" + sim.Net.Dropped + ", delivered=" + sim.Net.Delivered + ")");
        Check(outcome.InDoubt == 3, "all three participants are in doubt, got " + outcome.InDoubt);
        Check(outcome.Timeouts == 3, "each one raised its timeout, got " + outcome.Timeouts);
        Check(sim.Net.Dropped == 3, "three decision messages were lost, dropped=" + sim.Net.Dropped);
        foreach (PcParticipant node in sim.Nodes)
        {
            Check(node.Store.Get("m") == 0, "no participant applied anything");
            Check(node.Store.IsLocked("m"), "the lock is still held - 2PC blocks");
            Check(node.TimeoutsRaised == 1, "one timeout per participant");
        }
        // The coordinator comes back and replays its decision from the log: the block is resolved.
        sim.Net.Suppress = null;
        Check(sim.ReplayDecision(outcome.TxnId) == 3, "the recovery replays three decisions");
        sim.Drain(20);
        foreach (PcParticipant node in sim.Nodes)
        {
            Check(node.Store.Get("m") == 42, "the replayed commit applies everywhere");
            Check(!node.Store.IsLocked("m"), "the blocking lock is released");
        }
        Check(sim.Recoveries == 3, "three in-doubt participants recovered");
    }

    private static void DelayedDecisionResolvesAfterTimeout()
    {
        var sim = Fresh(2, Writes("p", 0));
        sim.Net.Latency = 6;                              // much slower than the timeout
        TxnOutcome outcome = sim.Run(Writes("p", 5), new[] { 1, 2 }, 6, 2, 80);
        Check(outcome.Decided && outcome.Committed, "the delayed decision still commits");
        Check(outcome.Timeouts > 0, "a timeout was raised while waiting, got " + outcome.Timeouts);
        Check(outcome.InDoubt == 0, "no participant is left in doubt");
        Check(sim.Node(1).Store.Get("p") == 5 && sim.Node(2).Store.Get("p") == 5, "both applied the commit");
        Check(sim.Node(1).TimeoutsRaised > 0, "the participant reported the timeout");
    }

    private static void LostVoteAbortsAndReleasesLocks()
    {
        var sim = Fresh(3, Writes("q", 0));
        sim.Net.SetLinkDown(2, sim.CoordinatorId);        // participant 2's vote never arrives
        TxnOutcome outcome = sim.Run(Writes("q", 1), new[] { 1, 2, 3 }, 1, 4, 80);
        Check(!outcome.Decided, "without every vote the coordinator cannot decide");
        Check(outcome.YesVotes == 2, "only two votes arrived, got " + outcome.YesVotes);
        Check(outcome.InDoubt == 2, "the coordinator knows of two in-doubt participants, got " + outcome.InDoubt);
        foreach (PcParticipant node in sim.Nodes)
        {
            // Node 2 also voted YES locally - its vote is simply lost - so all three hold locks.
            Check(node.InDoubt.Contains(outcome.TxnId), "node " + node.Id + " is in doubt locally");
            Check(node.Store.Get("q") == 0, "nothing was applied");
            Check(node.Store.IsLocked("q"), "node " + node.Id + " holds the lock");
        }
        sim.Net.SetLinkUp(2, sim.CoordinatorId);
        sim.Run(Writes("q", 1), new[] { 1, 2, 3 }, 1, 4, 80);
        // The stuck transaction is still holding node 1..3's locks, so the retry must fail.
        Check(sim.Node(1).Store.IsLocked("q"), "the abandoned transaction keeps its lock (2PC blocks)");
        foreach (PcParticipant node in sim.Nodes) node.Store.Abort(outcome.TxnId);
        Check(sim.Node(1).Store.IsLocked("q") == false, "an operator-forced abort frees the lock");
        TxnOutcome retry = sim.Run(Writes("q", 7), new[] { 1, 2, 3 }, 1, 4, 80);
        Check(retry.Committed, "the retry succeeds once the lock is gone");
        foreach (PcParticipant node in sim.Nodes) Check(node.Store.Get("q") == 7, "the retry applied everywhere");
    }

    private static void AtomicTransfer()
    {
        var sim = Fresh(2, Writes("alice", 1000, "bob", 0));
        TxnOutcome ok = sim.Run(Writes("alice", 900), new[] { 1 }, 1, 5, 60);
        TxnOutcome ok2 = sim.Run(Writes("bob", 100), new[] { 2 }, 1, 5, 60);
        Check(ok.Committed && ok2.Committed, "both halves commit in this simple two-node layout");
        int total = sim.Node(1).Store.Get("alice") + sim.Node(2).Store.Get("bob");
        Check(total == 1000, "the transfer conserves the total: " + total);

        // A two-node transfer whose second half is refused: the first half must be rolled back.
        var atomic = Fresh(2, Writes("alice", 500, "bob", 500));
        Check(atomic.Node(2).Store.TryPrepare(50, Writes("bob", 1)), "an outsider locks bob");
        TxnOutcome aborted = atomic.Run(Writes("alice", 400, "bob", 600), new[] { 1, 2 }, 1, 5, 60);
        Check(!aborted.Committed, "the transfer aborts when one node refuses");
        Check(atomic.Node(1).Store.Get("alice") == 500, "the debit was rolled back");
        Check(atomic.Node(2).Store.Get("bob") == 500, "the credit never happened");
        Check(atomic.Node(1).Store.LockedKeyCount == 0, "no lock leaked on the first node");
        atomic.Node(2).Store.Abort(50);
        Check(atomic.Node(2).Store.Get("bob") == 500, "the outsider's abort leaves the balance alone");
    }

    public static void Main()
    {
        HappyPathCommits();
        OneNoAbortsEverything();
        IsolationAndReadYourWrites();
        ConflictingPrepareVotesNo();
        CoordinatorCrashLeavesNodesInDoubt();
        DelayedDecisionResolvesAfterTimeout();
        LostVoteAbortsAndReleasesLocks();
        AtomicTransfer();
        Console.WriteLine("PASS");
    }
}
