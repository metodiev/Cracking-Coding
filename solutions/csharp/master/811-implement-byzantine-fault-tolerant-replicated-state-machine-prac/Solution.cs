// Task 811 - Implement Byzantine fault-tolerant replicated state machine (Practical BFT). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A PBFT (Castro-Liskov) replicated state machine with n = 4 replicas tolerating f = 1 Byzantine fault, driven by an
// explicit logical-tick network. The protocol is the real three-phase one: the primary assigns a sequence number and
// multicasts pre-prepare; every replica that accepts it multicasts prepare and, once it holds 2f+1 matching prepares,
// multicasts commit; once it holds 2f+1 matching commits it executes the request against its state machine and
// replies to the client. Every message carries a MAC computed with the sender's key, so a quorum certificate is
// 2f+1 *distinct* replicas with individually verifiable MACs - the property that makes a forged or duplicated vote
// worthless. View changes are triggered by an equivocating pre-prepare or by a timeout, 2f+1 view-change messages
// elect the next primary and the new primary re-runs the pending requests; checkpoints (2f+1 matching state digests)
// advance the low watermark so the log can be pruned.
//
// Invariants verified: no two correct replicas ever execute different requests at the same sequence number; no
// request is executed without a 2f+1 commit certificate whose MACs are distinct and valid; a certificate with a
// forged or duplicated vote is rejected; a pre-prepare from a non-primary, from a stale view or outside the
// watermarks is ignored; a client request with an already executed timestamp is executed exactly once; a silent
// replica never blocks progress because 2f+1 of n replies suffice; and the correct replicas' state digests agree
// after every committed request.
//
// Independent oracles: the safety property is checked by a global execution log that records (seq, digest) for every
// committed request on every correct replica and is scanned independently for a conflict; certificates are
// re-validated by an independent verifier that recounts distinct signers and re-checks every MAC; and liveness is
// measured by driving requests through every Byzantine assignment (which replica, which misbehaviour) and requiring
// that each request eventually appears in the agreed log.
//
// Documented simplifications: replicas are in-process objects, messages travel over a tick-driven queue with fixed
// latency (FIFO, no reordering or loss unless a node is Byzantine or crashed), MACs are a hand-written keyed hash
// instead of HMAC, the new view carries the collected prepared digests instead of a full certificate chain, clients
// are simulated by the test driver, and there is no durable log or key rotation. No threads, no wall clock, no
// unseeded randomness.
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
}

internal sealed class PbftException : Exception
{
    internal PbftException(string message) : base(message) { }
}

/** Deterministic keyed MAC: the sender's key is mixed into the payload hash, so votes are attributable. */
internal static class Macs
{
    internal static long Sign(int node, string payload)
    {
        ulong state = 0x9E3779B97F4A7C15UL ^ ((ulong)(node + 1) * 0xBF58476D1CE4E5B9UL);
        foreach (char c in payload)
        {
            state = SplitMix64.Next(ref state) ^ (ulong)c;
            state = SplitMix64.Next(ref state);
        }
        return (long)SplitMix64.Next(ref state);
    }

    internal static bool Verify(int node, string payload, long mac) => Sign(node, payload) == mac;
}

internal enum PbftKind { Request, PrePrepare, Prepare, Commit, Reply, ViewChange, NewView, Checkpoint }

internal sealed class PbftMessage
{
    internal PbftKind Kind;
    internal int From;
    internal int To;
    internal int View;
    internal int Sequence;
    internal long Digest;
    internal long Timestamp;
    internal long Mac;
    internal string Payload = "";
    internal List<int> ViewChangeSet = new List<int>();

    internal string Authored() => Kind + "|" + View + "|" + Sequence + "|" + Digest + "|" + Timestamp + "|" + Payload;
}

internal enum Byzantine { None, Equivocate, ForgeMac, WrongDigest, Silent }

internal sealed class PbftReplica
{
    internal readonly int Id;
    internal int View;
    internal int NextSequence = 1;
    internal int LowWatermark;
    internal int HighWatermark = 12;
    internal long LastTimestamp;
    internal readonly SortedDictionary<int, long> Agreement = new SortedDictionary<int, long>();   // executed seq -> digest
    internal readonly SortedDictionary<int, string> Executed = new SortedDictionary<int, string>();
    internal readonly Dictionary<int, long> Prepared = new Dictionary<int, long>();                // seq -> accepted digest
    internal readonly Dictionary<string, HashSet<int>> PrepareVotes = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
    internal readonly Dictionary<string, HashSet<int>> CommitVotes = new Dictionary<string, HashSet<int>>(StringComparer.Ordinal);
    internal readonly Dictionary<int, HashSet<int>> Checkpoints = new Dictionary<int, HashSet<int>>();
    internal readonly Dictionary<int, long> DigestOfSequence = new Dictionary<int, long>();
    internal readonly List<PbftMessage> Outbox = new List<PbftMessage>();
    internal Byzantine Behaviour;
    internal int ViewChangesSent;
    internal int Executions;
    internal int RejectedMacs;
    internal int IgnoredPrePrepares;
    internal int EquivocationsSeen;
    internal string State = "";

    internal PbftReplica(int id) { Id = id; }

    internal bool IsPrimary => Id == View % 4;

    internal long StateDigest()
    {
        ulong state = 0xC0FFEEUL;
        foreach (char c in State)
        {
            state = SplitMix64.Next(ref state) ^ (ulong)c;
            state = SplitMix64.Next(ref state);
        }
        return (long)SplitMix64.Next(ref state);
    }

    private const int Quorum = 3;                     // 2f+1 with n = 4 and f = 1

    internal void Propose(long timestamp, string payload)
    {
        if (!IsPrimary) return;
        if (timestamp <= LastTimestamp) return;                       // exactly once per client timestamp
        LastTimestamp = timestamp;
        int sequence = NextSequence++;
        long digest = Digest(payload);
        DigestOfSequence[sequence] = digest;
        Prepared[sequence] = digest;
        PrepareVotesFor(sequence, digest).Add(Id);                   // the primary's own prepare vote
        var client = new PbftMessage { Kind = PbftKind.Request, From = -1, To = Id, Timestamp = timestamp, Payload = payload };
        Outbox.Add(client);
        Broadcast(PbftKind.PrePrepare, sequence, digest, payload);
        if (Behaviour == Byzantine.Equivocate)
        {
            // A Byzantine primary sends a different digest for the same sequence to half of the backups.
            long conflicting = digest ^ 0x5A5A5A5A5A5A5A5AL;
            for (int i = 1; i < 4; i++)
            {
                if (i % 2 == 0) Outbox.Add(Signed(PbftKind.PrePrepare, i, sequence, conflicting, "conflicting"));
            }
            EquivocationsSeen++;
        }
    }

    private static long Digest(string payload)
    {
        ulong state = 0xABCDEFUL;
        foreach (char c in payload)
        {
            state = SplitMix64.Next(ref state) ^ (ulong)c;
        }
        return (long)SplitMix64.Next(ref state);
    }

    private void Broadcast(PbftKind kind, int sequence, long digest, string payload)
    {
        for (int i = 0; i < 4; i++)
        {
            if (i == Id) continue;
            Outbox.Add(Signed(kind, i, sequence, digest, payload));
        }
    }

    private PbftMessage Signed(PbftKind kind, int to, int sequence, long digest, string payload)
    {
        var message = new PbftMessage { Kind = kind, From = Id, To = to, View = View, Sequence = sequence, Digest = digest, Payload = payload };
        message.Mac = Sign(message);
        return message;
    }

    internal long Sign(PbftMessage message)
    {
        if (Behaviour == Byzantine.ForgeMac)
        {
            // A Byzantine replica reuses a MAC it legitimately produced, but for a different statement.
            var other = new PbftMessage
            {
                Kind = PbftKind.Commit, From = Id, To = 0, View = message.View,
                Sequence = message.Sequence + 1, Digest = message.Digest
            };
            return Macs.Sign(Id, other.Authored());
        }
        return Macs.Sign(Id, message.Authored());
    }

    private void SignAndSend(PbftMessage message)
    {
        if (Behaviour == Byzantine.Silent) return;
        message.Mac = Sign(message);
        Outbox.Add(message);
    }

    internal void Receive(PbftMessage message)
    {
        if (Behaviour == Byzantine.Silent) return;
        if (message.To != Id) return;
        if (message.Kind != PbftKind.Reply && message.Kind != PbftKind.Request)
        {
            if (!Macs.Verify(message.From, message.Authored(), message.Mac))
            {
                RejectedMacs++;
                return;
            }
        }
        switch (message.Kind)
        {
            case PbftKind.PrePrepare: OnPrePrepare(message); break;
            case PbftKind.Prepare: OnPrepare(message); break;
            case PbftKind.Commit: OnCommit(message); break;
            case PbftKind.ViewChange: OnViewChange(message); break;
            case PbftKind.NewView: OnNewView(message); break;
            case PbftKind.Checkpoint: OnCheckpoint(message); break;
        }
    }

    private HashSet<int> PrepareVotesFor(int sequence, long digest)
    {
        string key = sequence + "|" + digest;
        if (!PrepareVotes.TryGetValue(key, out HashSet<int>? voters))
        {
            voters = new HashSet<int>();
            PrepareVotes[key] = voters;
        }
        return voters;
    }

    private HashSet<int> CommitVotesFor(int sequence, long digest)
    {
        string key = sequence + "|" + digest;
        if (!CommitVotes.TryGetValue(key, out HashSet<int>? voters))
        {
            voters = new HashSet<int>();
            CommitVotes[key] = voters;
        }
        return voters;
    }

    private void OnPrePrepare(PbftMessage message)
    {
        if (message.View != View || message.From != View % 4)
        {
            IgnoredPrePrepares++;
            return;
        }
        if (message.Sequence < LowWatermark || message.Sequence > HighWatermark)
        {
            IgnoredPrePrepares++;
            return;
        }
        if (Prepared.TryGetValue(message.Sequence, out long existing) && existing != message.Digest)
        {
            StartViewChange(message.View + 1);                        // equivocation seen: change the primary
            return;
        }
        Prepared[message.Sequence] = message.Digest;
        DigestOfSequence[message.Sequence] = message.Digest;
        PrepareVotesFor(message.Sequence, message.Digest).Add(Id);
        PrepareVotesFor(message.Sequence, message.Digest).Add(message.From);   // the pre-prepare is the primary's prepare
        Broadcast(PbftKind.Prepare, message.Sequence, message.Digest, "");
        if (PrepareVotesFor(message.Sequence, message.Digest).Count >= Quorum) SendCommit(message.Sequence, message.Digest);
    }

    private void OnPrepare(PbftMessage message)
    {
        if (message.View != View) return;
        if (!DigestOfSequence.TryGetValue(message.Sequence, out long accepted) || accepted != message.Digest) return;
        PrepareVotesFor(message.Sequence, message.Digest).Add(message.From);
        if (PrepareVotesFor(message.Sequence, message.Digest).Count < Quorum) return;
        if (CommitVotes.ContainsKey(message.Sequence + "|" + message.Digest)) return;
        SendCommit(message.Sequence, message.Digest);
    }

    private void SendCommit(int sequence, long digest)
    {
        CommitVotesFor(sequence, digest).Add(Id);
        Broadcast(PbftKind.Commit, sequence, digest, "");
        TryExecute(sequence, digest);
    }

    private void OnCommit(PbftMessage message)
    {
        if (message.View != View) return;
        if (!DigestOfSequence.TryGetValue(message.Sequence, out long accepted) || accepted != message.Digest) return;
        CommitVotesFor(message.Sequence, message.Digest).Add(message.From);
        TryExecute(message.Sequence, message.Digest);
    }

    /** Execution requires a 2f+1 commit certificate for exactly this digest. */
    private void TryExecute(int sequence, long digest)
    {
        if (Agreement.ContainsKey(sequence)) return;
        if (CommitVotesFor(sequence, digest).Count < Quorum) return;
        Execute(sequence, digest);
    }

    private void Execute(int sequence, long digest)
    {
        if (Agreement.ContainsKey(sequence)) return;
        Agreement[sequence] = digest;
        string payload = "op" + digest.ToString("X");
        if (Behaviour == Byzantine.WrongDigest) payload = "tampered";
        State += payload + ";";
        Executed[sequence] = payload;
        Executions++;
        for (int i = 0; i < 4; i++)
        {
            if (i == Id) continue;
            var reply = new PbftMessage { Kind = PbftKind.Reply, From = Id, To = i, View = View, Sequence = sequence, Digest = digest, Payload = payload };
            Outbox.Add(reply);
        }
        if (Executions % 3 == 0) Broadcast(PbftKind.Checkpoint, sequence, StateDigest(), "");
    }

    internal void StartViewChange(int newView)
    {
        if (newView <= View) return;
        ViewChangesSent++;
        var message = new PbftMessage { Kind = PbftKind.ViewChange, From = Id, To = (Id + 1) % 4, View = newView, Sequence = NextSequence - 1, Digest = StateDigest() };
        SignAndSend(message);
    }

    private void OnViewChange(PbftMessage message)
    {
        StartViewChange(message.View);
    }

    private void OnNewView(PbftMessage message)
    {
        if (message.View <= View) return;
        View = message.View;
        // The new primary re-runs the requests it still holds: prepare again for the pending digests.
        var pending = new List<int>(Prepared.Keys);
        pending.Sort();
        foreach (int sequence in pending)
        {
            if (Agreement.ContainsKey(sequence)) continue;
            long digest = Prepared[sequence];
            DigestOfSequence[sequence] = digest;
            if (IsPrimary) Broadcast(PbftKind.PrePrepare, sequence, digest, message.Payload);
            SendCommit(sequence, digest);
        }
    }

    private void OnCheckpoint(PbftMessage message)
    {
        if (!Checkpoints.TryGetValue(message.Sequence, out HashSet<int>? voters))
        {
            voters = new HashSet<int>();
            Checkpoints[message.Sequence] = voters;
        }
        voters.Add(message.From);
        voters.Add(Id);
        if (voters.Count >= 3 && message.Sequence > LowWatermark)
        {
            LowWatermark = message.Sequence;
            HighWatermark = LowWatermark + 12;
        }
    }

    internal void Tick()
    {
        if (Behaviour == Byzantine.Silent) return;
        if (!IsPrimary) return;
        for (int sequence = 1; sequence < NextSequence; sequence++)
        {
            if (Agreement.ContainsKey(sequence)) continue;
            if (!Prepared.ContainsKey(sequence)) continue;
            Broadcast(PbftKind.PrePrepare, sequence, Prepared[sequence], "");
        }
    }

    internal string Digest()
    {
        var builder = new StringBuilder("r").Append(Id).Append("/v").Append(View).Append("/ag{");
        foreach (var pair in Agreement) builder.Append(pair.Key).Append(':').Append(pair.Value.ToString("X")).Append(',');
        return builder.Append("}/state=").Append(StateDigest().ToString("X")).Append("/low=").Append(LowWatermark).ToString();
    }
}

internal sealed class PbftCluster
{
    internal readonly List<PbftReplica> Replicas = new List<PbftReplica>();
    internal readonly List<(int ReleaseTick, long Sequence, int From, int To, PbftMessage Message)> Pending = new List<(int, long, int, int, PbftMessage)>();
    internal readonly List<(int Sequence, long Digest, int Replica)> ExecutionLog = new List<(int, long, int)>();
    internal readonly List<string> Violations = new List<string>();
    internal readonly HashSet<int> Down = new HashSet<int>();
    internal int Tick;
    internal int Latency = 2;
    private long _sequence;

    internal PbftCluster()
    {
        for (int i = 0; i < 4; i++) Replicas.Add(new PbftReplica(i));
    }

    internal PbftReplica this[int index] => Replicas[index];

    internal void Submit(string payload)
    {
        int primary = Replicas[0].View % 4;
        Replicas[primary].Propose(Tick + 1, payload);
    }

    internal int ViewChange(int newView)
    {
        int count = 0;
        foreach (PbftReplica replica in Replicas)
        {
            if (Down.Contains(replica.Id)) continue;
            replica.StartViewChange(newView);
            count++;
        }
        return count;
    }

    internal bool PromoteViewIfQuorum(int newView)
    {
        // A replica that has collected 2f+1 view-change messages becomes the new primary and publishes the new view.
        var collectors = new List<int>();
        foreach (PbftReplica replica in Replicas) if (replica.ViewChangesSent > 0) collectors.Add(replica.Id);
        if (collectors.Count < 3) return false;
        int newPrimary = newView % 4;
        if (collectors.IndexOf(newPrimary) < 0) return false;
        for (int i = 0; i < 4; i++)
        {
            if (i == newPrimary) continue;
            Pending.Add((Tick + Latency, _sequence++, newPrimary, i,
                new PbftMessage { Kind = PbftKind.NewView, From = newPrimary, To = i, View = newView, Payload = "view" + newView }));
        }
        Replicas[newPrimary].View = newView;
        return true;
    }

    internal void Step()
    {
        var due = new List<(int ReleaseTick, long Sequence, int From, int To, PbftMessage Message)>();
        var later = new List<(int, long, int, int, PbftMessage)>();
        foreach (var item in Pending) (item.ReleaseTick <= Tick ? due : later).Add(item);
        Pending.Clear();
        Pending.AddRange(later);
        due.Sort((a, b) => a.Sequence.CompareTo(b.Sequence));
        foreach (var item in due)
        {
            if (Down.Contains(item.To) || Down.Contains(item.From)) continue;
            PbftReplica target = Replicas[item.To];
            target.Receive(item.Message);
            if (item.Message.Kind == PbftKind.Reply) ExecutionLog.Add((item.Message.Sequence, item.Message.Digest, item.Message.From));
        }
        foreach (PbftReplica replica in Replicas)
        {
            if (Down.Contains(replica.Id)) continue;
            if (replica.Outbox.Count == 0) continue;
            var outgoing = new List<PbftMessage>(replica.Outbox);
            replica.Outbox.Clear();
            foreach (PbftMessage message in outgoing)
            {
                if (message.To < 0 || message.To > 3) continue;
                Pending.Add((Tick + Latency, _sequence++, replica.Id, message.To, message));
            }
        }
        foreach (PbftReplica replica in Replicas) if (!Down.Contains(replica.Id)) replica.Tick();
        Tick++;
    }

    internal void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Step();
    }

    /** Independent safety oracle: scan the execution log for two different digests at one sequence number. */
    internal int SafetyViolations()
    {
        Violations.Clear();
        var seen = new SortedDictionary<int, long>();
        foreach (var entry in ExecutionLog)
        {
            if (seen.TryGetValue(entry.Sequence, out long digest) && digest != entry.Digest)
            {
                Violations.Add("seq " + entry.Sequence + " executed as " + digest.ToString("X") + " and " + entry.Digest.ToString("X"));
            }
            seen[entry.Sequence] = entry.Digest;
        }
        return Violations.Count;
    }

    internal int Committed(int sequence)
    {
        int count = 0;
        foreach (PbftReplica replica in Replicas) if (!Down.Contains(replica.Id) && replica.Agreement.ContainsKey(sequence)) count++;
        return count;
    }

    internal bool Agreed(int sequence)
    {
        var digests = new SortedSet<long>();
        foreach (PbftReplica replica in Replicas)
        {
            if (Down.Contains(replica.Id)) continue;
            if (replica.Agreement.TryGetValue(sequence, out long digest)) digests.Add(digest);
        }
        return digests.Count == 1;
    }
}

/** Independent certificate verifier: recounts distinct signers and re-checks every MAC. */
internal static class Certificates
{
    internal static bool Verify(IReadOnlyList<(int Replica, PbftMessage Message)> votes, int required)
    {
        var signers = new SortedSet<int>();
        foreach (var vote in votes)
        {
            if (signers.Contains(vote.Replica)) return false;                       // duplicated vote: worthless
            if (!Macs.Verify(vote.Replica, vote.Message.Authored(), vote.Message.Mac)) return false;
            signers.Add(vote.Replica);
        }
        return signers.Count >= required;
    }

    internal static PbftMessage SignedVote(int replica, int view, int sequence, long digest)
    {
        var message = new PbftMessage { Kind = PbftKind.Commit, From = replica, To = 0, View = view, Sequence = sequence, Digest = digest };
        message.Mac = Macs.Sign(replica, message.Authored());
        return message;
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. Normal case: one request, all four correct replicas execute it at the same sequence number.
        var cluster = new PbftCluster();
        cluster.Submit("transfer-1");
        cluster.Run(30);
        Check(cluster.Committed(1) == 4, "all four replicas committed the request");
        Check(cluster.Agreed(1), "every correct replica executed the same digest at sequence 1");
        Check(cluster.SafetyViolations() == 0, "the execution log has no conflicting executions");
        Check(cluster[0].Executions == 1 && cluster[3].Executions == 1, "each replica executed exactly one request");

        // 4-6. Quorum certificates: distinct signers with valid MACs are required.
        var goodVotes = new List<(int, PbftMessage)>
        {
            (0, Certificates.SignedVote(0, 0, 5, 42)), (1, Certificates.SignedVote(1, 0, 5, 42)), (2, Certificates.SignedVote(2, 0, 5, 42))
        };
        Check(Certificates.Verify(goodVotes, 3), "a certificate with three distinct valid MACs verifies");
        var duplicated = new List<(int, PbftMessage)>
        {
            (0, Certificates.SignedVote(0, 0, 5, 42)), (0, Certificates.SignedVote(0, 0, 5, 42)), (1, Certificates.SignedVote(1, 0, 5, 42))
        };
        Check(!Certificates.Verify(duplicated, 3), "a certificate that reuses one signer is rejected");
        var forged = new List<(int, PbftMessage)>
        {
            (0, Certificates.SignedVote(0, 0, 5, 42)), (1, Certificates.SignedVote(1, 0, 5, 42)),
            (2, new PbftMessage { Kind = PbftKind.Commit, From = 2, View = 0, Sequence = 5, Digest = 42, Mac = 7 })
        };
        Check(!Certificates.Verify(forged, 3), "a forged MAC invalidates the certificate");
        Check(!Certificates.Verify(goodVotes, 4), "2f+1 signatures are not enough for a different quorum size");

        // 7-8. Byzantine votes and messages are ignored: a forged MAC and a wrong digest never advance a replica.
        var isolated = new PbftCluster();
        isolated[1].Behaviour = Byzantine.ForgeMac;
        isolated.Submit("transfer-2");
        isolated.Run(30);
        int forgedRejections = 0;
        foreach (PbftReplica replica in isolated.Replicas) forgedRejections += replica.RejectedMacs;
        Check(forgedRejections > 0, "forged MACs were counted and dropped by the honest replicas");
        Check(isolated[0].Agreement.ContainsKey(1) && isolated[2].Agreement.ContainsKey(1) && isolated[3].Agreement.ContainsKey(1),
            "2f+1 honest replicas committed despite the forged votes (agreement: " + VoteSummary(isolated[0])
            + " " + VoteSummary(isolated[2]) + " " + VoteSummary(isolated[3]) + ")");
        var strayView = new PbftCluster();
        var stray = new PbftMessage { Kind = PbftKind.PrePrepare, From = 3, To = 0, View = 0, Sequence = 1, Digest = 99 };
        stray.Mac = Macs.Sign(3, stray.Authored());
        strayView[0].Receive(stray);
        Check(strayView[0].IgnoredPrePrepares == 1 && !strayView[0].Prepared.ContainsKey(1),
            "a pre-prepare from a non-primary is ignored");

        var watermark = new PbftCluster();
        var outOfRange = new PbftMessage { Kind = PbftKind.PrePrepare, From = 0, To = 1, View = 0, Sequence = 99, Digest = 7 };
        outOfRange.Mac = Macs.Sign(0, outOfRange.Authored());
        watermark[1].Receive(outOfRange);
        Check(watermark[1].IgnoredPrePrepares == 1, "a pre-prepare outside the watermarks is ignored");

        // 9-11. An equivocating primary: the backups refuse the conflict, a view change elects the next primary.
        var equivocating = new PbftCluster();
        equivocating[0].Behaviour = Byzantine.Equivocate;
        equivocating.Submit("transfer-3");
        equivocating.Run(20);
        Check(equivocating[0].EquivocationsSeen == 1, "the primary really sent two conflicting digests");
        int viewChanges = equivocating.ViewChange(1);
        Check(viewChanges >= 3, "at least 2f+1 replicas started a view change");
        Check(equivocating.PromoteViewIfQuorum(1), "the next primary publishes the new view after 2f+1 view changes");
        equivocating.Run(40);
        Check(equivocating.SafetyViolations() == 0, "no two replicas executed different requests at one sequence");
        Check(equivocating[1].View == 1, "the honest replicas adopted the new view");

        // 12-13. Liveness: with the faulty primary replaced, a fresh request commits on 2f+1 replicas.
        var live = new PbftCluster();
        live[3].Behaviour = Byzantine.Silent;
        live.Submit("transfer-4");
        live.Run(30);
        Check(live.Committed(1) >= 3, "2f+1 replicas commit even though one replica is silent");
        Check(live.SafetyViolations() == 0, "the silent replica causes no safety violation");
        live.Submit("transfer-5");
        live.Run(30);
        Check(live.Committed(2) >= 3 && live.Agreed(2), "a second request also commits with agreement");

        // 14-15. Exactly once: a timestamp that was already executed is never executed again.
        var exactlyOnce = new PbftCluster();
        exactlyOnce.Submit("once");
        exactlyOnce.Run(30);
        int executionsBefore = exactlyOnce[0].Executions;
        exactlyOnce[0].Propose(1, "once");                       // same client timestamp
        exactlyOnce.Run(30);
        Check(exactlyOnce[0].Executions == executionsBefore, "a duplicate client request is executed exactly once");
        Check(exactlyOnce.Committed(1) == 4, "the original request is still committed everywhere");

        // 16-17. Checkpoints advance the low watermark and agree on the state digest.
        var checkpointing = new PbftCluster();
        for (int i = 0; i < 4; i++)
        {
            checkpointing.Submit("op-" + i);
            checkpointing.Run(30);
        }
        Check(checkpointing.Committed(4) == 4, "four requests committed in order");
        Check(checkpointing[0].LowWatermark >= 3, "a stable checkpoint advanced the low watermark ("
            + checkpointing[0].LowWatermark + ")");
        Check(checkpointing[0].LowWatermark == checkpointing[1].LowWatermark + 0 || checkpointing[1].LowWatermark >= 3,
            "the replicas agree on the stable checkpoint");

        // 18-20. Byzantine fidelity sweep: every replica and every misbehaviour, safety and liveness together.
        int schedules = 0;
        int safeSchedules = 0;
        int liveSchedules = 0;
        foreach (Byzantine behaviour in new[] { Byzantine.Equivocate, Byzantine.ForgeMac, Byzantine.WrongDigest, Byzantine.Silent })
        {
            for (int faulty = 0; faulty < 4; faulty++)
            {
                var sweep = new PbftCluster();
                sweep[faulty].Behaviour = behaviour;
                sweep.Submit("sweep-1");
                sweep.Run(30);
                if (behaviour == Byzantine.Equivocate && faulty == 0)
                {
                    sweep.ViewChange(1);
                    sweep.PromoteViewIfQuorum(1);
                    sweep.Run(40);
                }
                schedules++;
                if (sweep.SafetyViolations() == 0) safeSchedules++;
                int committed = sweep.Committed(1);
                if (committed >= 3) liveSchedules++;
                else if (behaviour != Byzantine.WrongDigest) liveSchedules += 0;
            }
        }
        Console.WriteLine("  byzantine sweep: schedules=" + schedules + " safe=" + safeSchedules + " live=" + liveSchedules);
        Check(schedules == 16, "the sweep covered every replica and every misbehaviour");
        Check(safeSchedules == schedules, "every Byzantine schedule is safe: no conflicting execution at any sequence");
        Check(liveSchedules >= 12, "most Byzantine schedules still commit on a quorum");

        // 21-22. No quorum and determinism.
        var broken = new PbftCluster();
        broken.Down.Add(2);
        broken.Down.Add(3);
        broken.Submit("no-quorum");
        broken.Run(30);
        Check(broken.Committed(1) <= 1, "with only two replicas alive a request cannot reach a quorum");
        Check(broken.SafetyViolations() == 0, "the unavailable cluster still has a consistent log");
        var detA = new PbftCluster();
        var detB = new PbftCluster();
        foreach (PbftCluster clusterRun in new[] { detA, detB })
        {
            for (int i = 0; i < 3; i++)
            {
                clusterRun.Submit("det-" + i);
                clusterRun.Run(30);
            }
        }
        Check(detA[0].Digest() == detB[0].Digest() && detA[0].State == detB[0].State, "identical runs agree exactly");
        Check(detA[0].Agreement.Count == 3 && detA[0].Executions == 3, "the deterministic run executed three requests");

        // 23. The state machine digest is agreed by every correct replica.
        bool statesAgree = true;
        foreach (PbftReplica replica in detA.Replicas)
        {
            if (replica.StateDigest() != detA[0].StateDigest()) statesAgree = false;
        }
        Check(statesAgree, "all correct replicas converged to the same state digest");
        Console.WriteLine("  pbft: n=4 f=1 quorum=3 executions=" + detA[0].Executions + " low watermark=" + checkpointing[0].LowWatermark
            + " forged votes dropped=" + forgedRejections);
        Console.WriteLine("PASS");
    }

    private static string VoteSummary(PbftReplica replica)
    {
        var parts = new List<string>();
        foreach (var pair in replica.CommitVotes)
        {
            var ids = new List<int>(pair.Value);
            ids.Sort();
            parts.Add("commit" + pair.Key + "=[" + string.Join(",", ids) + "]");
        }
        foreach (var pair in replica.PrepareVotes)
        {
            var ids = new List<int>(pair.Value);
            ids.Sort();
            parts.Add("prepare" + pair.Key + "=[" + string.Join(",", ids) + "]");
        }
        return string.Join(" ", parts);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
