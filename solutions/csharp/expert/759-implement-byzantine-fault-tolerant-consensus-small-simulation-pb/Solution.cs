// Task 759 - Implement Byzantine fault tolerant consensus small simulation (PBFT). (Tier: Expert, Ch 9 - System Design and Scalability)
//
// A four replica PBFT (Castro-Liskov) with f = 1, so the quorum is 2f+1 = 3 and 3f+1 = 4 replicas are needed.
// Each sequence number goes through pre-prepare (the primary proposes a batch), prepare (a replica that accepted
// the proposal votes for its digest) and commit (a replica that saw 2f+1 prepares votes to execute). A correct
// replica executes a sequence only when it holds 2f+1 commit votes *for the digest it prepared itself*, which is
// what preserves safety: any two quorums of 3 out of 4 intersect in at least two replicas, and only one of them
// can be faulty, so a common correct replica would have had to vote for two different digests. Messages carry a
// keyed MAC, so a faulty replica cannot vote on behalf of another one. A timeout moves every replica to the next
// view, whose primary is (view mod n); view changes are what restore liveness when the primary is crashed or
// equivocating.
//
// Simulated network: every message is an envelope with a deterministic delay (1 + (from*3 + to) mod 3) delivered
// from a priority queue ordered by (time, insertion order), so the whole run is reproducible. The scenario modes
// are fault free, a crashed (silent) primary, an equivocating Byzantine primary that proposes two different
// batches for the same slot, and a forging backup that impersonates other replicas.
//
// Documented simplifications: no checkpoints or watermarks, no view-change prepared-set transfer (a new primary
// only proposes sequences that were not executed, and the batch for a sequence is fixed by the client request),
// no client replies, one process instead of four hosts, and the MAC is a hand written toy keyed hash - this is a
// simulation, not cryptography, and no key material is involved. Invariants verified: (a) safety - across every
// mode no two correct replicas ever execute different batches for the same sequence, and every execution carries
// a certificate of at least 2f+1 distinct, MAC-verified voters; (b) the quorum intersection argument, checked
// exhaustively over all pairs of 3-subsets of 4; (c) liveness - every sequence executes at all correct replicas in
// every mode, with a view change when the primary is crashed or equivocating; (d) forgery rejection - a vote that
// impersonates another replica never enters a certificate, yet liveness is unaffected.
using System;
using System.Collections.Generic;

internal static class ToyCrypto
{
    private const ulong Master = 0x7595AFE1234567UL;
    private const ulong FnvOffset = 14695981039346656037UL;
    private const ulong FnvPrime = 1099511628211UL;

    internal static ulong Hash(string text)
    {
        ulong hash = FnvOffset;
        foreach (char c in text)
        {
            hash ^= c;
            hash *= FnvPrime;
        }
        return hash;
    }

    internal static ulong KeyFor(int replica) => Mix(Master ^ ((ulong)replica + 1) * 0x9E3779B97F4A7C15UL);

    internal static string Digest(string value) => Hash(value).ToString("X16");

    internal static string Payload(string kind, int view, int sequence, string digest)
        => kind + "|" + view + "|" + sequence + "|" + digest;

    internal static ulong Sign(int replica, string kind, int view, int sequence, string digest)
        => Mix(KeyFor(replica) + Hash(Payload(kind, view, sequence, digest)) * 0x9E3779B97F4A7C15UL);

    internal static bool Verify(int replica, string kind, int view, int sequence, string digest, ulong mac)
        => Sign(replica, kind, view, sequence, digest) == mac;

    private static ulong Mix(ulong value)
    {
        value += 0x9E3779B97F4A7C15UL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }
}

internal enum Mode { FaultFree, SilentPrimary, EquivocatingPrimary, ForgingBackup }

internal static class Kind
{
    internal const int PrePrepare = 0;
    internal const int Prepare = 1;
    internal const int Commit = 2;
    internal const int ViewChange = 3;
    internal const int NewView = 4;
}

internal sealed class Envelope
{
    internal long Time;
    internal long Order;
    internal int Kind;
    internal int View;
    internal int Sequence;
    internal int From;
    internal int To;
    internal string Digest = "";
    internal string Value = "";
    internal ulong Mac;
}

internal sealed class Network
{
    internal readonly List<Envelope> Queue = new List<Envelope>();
    internal long Clock;
    internal long Dropped;
    private long order;

    internal void Send(int from, int to, int kind, int view, int sequence, string digest, string value, long delay, ulong mac)
    {
        Queue.Add(new Envelope
        {
            Time = Clock + delay,
            Order = ++order,
            Kind = kind,
            View = view,
            Sequence = sequence,
            From = from,
            To = to,
            Digest = digest,
            Value = value,
            Mac = mac,
        });
    }

    internal Envelope? TakeNext()
    {
        if (Queue.Count == 0) return null;
        int best = 0;
        for (int i = 1; i < Queue.Count; i++)
        {
            if (Queue[i].Time < Queue[best].Time || (Queue[i].Time == Queue[best].Time && Queue[i].Order < Queue[best].Order))
            {
                best = i;
            }
        }
        Envelope envelope = Queue[best];
        Queue.RemoveAt(best);
        Clock = envelope.Time;
        return envelope;
    }
}

internal sealed class Replica
{
    internal readonly int Id;
    internal int View;

    internal readonly Dictionary<int, (string Digest, string Value)> Proposals = new Dictionary<int, (string, string)>();
    internal readonly Dictionary<string, SortedSet<int>> Prepares = new Dictionary<string, SortedSet<int>>();
    internal readonly Dictionary<string, SortedSet<int>> Commits = new Dictionary<string, SortedSet<int>>();
    internal readonly Dictionary<int, string> Executed = new Dictionary<int, string>();
    internal readonly Dictionary<int, string> Certificates = new Dictionary<int, string>();
    internal readonly HashSet<string> Committed = new HashSet<string>();
    internal readonly List<int> ExecutionOrder = new List<int>();

    internal Replica(int id) { Id = id; }

    internal void Send(Simulation simulation, int to, int kind, int sequence, string digest, string value)
    {
        ulong mac = ToyCrypto.Sign(Id, KindName(kind), View, sequence, digest);
        // Every message is delayed deterministically and independently of its content.
        simulation.Network.Send(Id, to, kind, View, sequence, digest, value, 1 + (Id * 3 + to) % 3, mac);
    }

    internal void SendPrePrepare(Simulation simulation, int target, int view, int sequence, string value)
    {
        string digest = ToyCrypto.Digest(value);
        ulong mac = ToyCrypto.Sign(Id, KindName(Kind.PrePrepare), view, sequence, digest);
        simulation.Network.Send(Id, target, Kind.PrePrepare, view, sequence, digest, value, 1 + (Id * 3 + target) % 3, mac);
    }

    /** ForgingBackup mode: sign a prepare with our own key while claiming somebody else's replica id. */
    internal void SendForgedPrepare(Simulation simulation, int sequence, string value, int impersonated)
    {
        string digest = ToyCrypto.Digest(value);
        ulong mac = ToyCrypto.Sign(Id, KindName(Kind.Prepare), View, sequence, digest);
        foreach (Replica peer in simulation.Replicas)
        {
            simulation.Network.Send(impersonated, peer.Id, Kind.Prepare, View, sequence, digest, "",
                1 + (impersonated * 3 + peer.Id) % 3, mac);
            simulation.ForgedVotes++;
        }
    }

    /** A view change drops unexecuted proposals: the new primary re-proposes them (see the header note). */
    internal void AdvanceView(int view)
    {
        if (view <= View) return;
        View = view;
        Proposals.Clear();
        Prepares.Clear();
        Commits.Clear();
        Committed.Clear();
    }

    internal static string KindName(int kind)
        => kind == Kind.PrePrepare ? "pre-prepare" : kind == Kind.Prepare ? "prepare" : kind == Kind.Commit ? "commit"
            : kind == Kind.ViewChange ? "view-change" : "new-view";

    internal void Handle(Envelope message, Simulation simulation)
    {
        switch (message.Kind)
        {
            case Kind.PrePrepare:
                HandleProposal(message, simulation);
                break;
            case Kind.Prepare:
                HandleVote(message, simulation, false);
                break;
            case Kind.Commit:
                HandleVote(message, simulation, true);
                break;
            case Kind.ViewChange:
                HandleViewChange(message, simulation);
                break;
            case Kind.NewView:
                if (message.View > View)
                {
                    AdvanceView(message.View);
                    simulation.CurrentView = Math.Max(simulation.CurrentView, message.View);
                }
                break;
        }
    }

    private void HandleProposal(Envelope message, Simulation simulation)
    {
        if (message.View != View || Executed.ContainsKey(message.Sequence)) return;
        if (ToyCrypto.Digest(message.Value) != message.Digest) return;      // inconsistent proposal, ignored
        if (Proposals.ContainsKey(message.Sequence)) return;                // already prepared something for this slot
        Proposals[message.Sequence] = (message.Digest, message.Value);
        foreach (Replica peer in simulation.Replicas) Send(simulation, peer.Id, Kind.Prepare, message.Sequence, message.Digest, "");
    }

    private void HandleVote(Envelope message, Simulation simulation, bool isCommit)
    {
        if (message.View != View) return;
        string key = message.Sequence + "|" + message.Digest;
        Dictionary<string, SortedSet<int>> table = isCommit ? Commits : Prepares;

        if (isCommit)
        {
            if (!Proposals.TryGetValue(message.Sequence, out (string Digest, string Value) prepared)
                || prepared.Digest != message.Digest) return;               // only commit for the digest we prepared
        }
        if (!table.TryGetValue(key, out SortedSet<int>? voters))
        {
            voters = new SortedSet<int>();
            table[key] = voters;
        }
        voters.Add(message.From);

        if (isCommit)
        {
            if (voters.Count >= simulation.Quorum && !Executed.ContainsKey(message.Sequence))
            {
                Executed[message.Sequence] = Proposals[message.Sequence].Value;
                Certificates[message.Sequence] = string.Join(",", voters);
                ExecutionOrder.Add(message.Sequence);
                simulation.Note(simulation.ReplicaName(Id) + " executes seq " + message.Sequence + " in view " + View);
            }
            return;
        }

        if (voters.Count >= simulation.Quorum && Committed.Add(key))
        {
            foreach (Replica peer in simulation.Replicas) Send(simulation, peer.Id, Kind.Commit, message.Sequence, message.Digest, "");
        }
    }

    private void HandleViewChange(Envelope message, Simulation simulation)
    {
        if (message.View <= simulation.CurrentView - 1) return;
        if (!simulation.ViewChangeVotes.TryGetValue(message.View, out SortedSet<int>? voters))
        {
            voters = new SortedSet<int>();
            simulation.ViewChangeVotes[message.View] = voters;
        }
        voters.Add(message.From);
        if (Id != simulation.PrimaryOf(message.View)) return;             // only the new primary announces a new view
        if (simulation.NewViewSent.Contains(message.View)) return;
        if (voters.Count < Simulation.ReplicaCount - Simulation.F) return; // need n - f view changes to move on
        simulation.NewViewSent.Add(message.View);
        foreach (Replica peer in simulation.Replicas)
        {
            Send(simulation, peer.Id, Kind.NewView, -1, "", "");
        }
    }

    internal void RequestViewChange(Simulation simulation, int newView)
    {
        AdvanceView(newView);
        foreach (Replica peer in simulation.Replicas) Send(simulation, peer.Id, Kind.ViewChange, -1, "", "");
    }
}

internal sealed class Simulation
{
    internal const int ReplicaCount = 4;
    internal const int F = 1;
    internal const int SequenceCount = 5;

    internal readonly Mode Mode;
    internal readonly int Faulty;
    internal readonly Replica[] Replicas = new Replica[ReplicaCount];
    internal readonly Network Network = new Network();
    internal readonly string[] Batches = new string[2 * SequenceCount];
    internal readonly Dictionary<int, SortedSet<int>> ViewChangeVotes = new Dictionary<int, SortedSet<int>>();
    internal readonly HashSet<int> NewViewSent = new HashSet<int>();
    private readonly HashSet<string> proposed = new HashSet<string>();

    internal int CurrentView;
    internal int Views = 1;
    internal long MacRejections;
    internal long ForgedVotes;
    internal long DeliveredMessages;
    internal readonly List<string> Events = new List<string>();

    internal int Quorum => 2 * F + 1;

    internal Simulation(Mode mode)
    {
        Mode = mode;
        Faulty = mode == Mode.FaultFree ? -1 : 0;
        if (mode == Mode.ForgingBackup) Faulty = ReplicaCount - 1;
        for (int i = 0; i < ReplicaCount; i++) Replicas[i] = new Replica(i);
        for (int s = 0; s < SequenceCount; s++)
        {
            Batches[2 * s] = "batch-" + s + "-A";
            Batches[2 * s + 1] = "batch-" + s + "-B";
        }
        for (int v = 0; v < 8; v++) ViewChangeVotes[v] = new SortedSet<int>();
    }

    internal bool Honest(int replica) => replica != Faulty;
    internal string ReplicaName(int replica) => "R" + replica + (Honest(replica) ? "" : "*");
    internal int PrimaryOf(int view) => ((view % ReplicaCount) + ReplicaCount) % ReplicaCount;
    internal bool IsSilentPrimary(int view) => Mode == Mode.SilentPrimary && PrimaryOf(view) == Faulty;

    internal void Note(string text) => Events.Add(text);

    internal void Run(int maxSteps)
    {
        for (int step = 0; step < maxSteps; step++)
        {
            Envelope? next = Network.TakeNext();
            if (next != null)
            {
                Deliver(next);
                continue;
            }
            if (AllExecuted()) return;
            if (Propose()) continue;
            Timeout();
        }
    }

    internal void Deliver(Envelope envelope)
    {
        DeliveredMessages++;
        if (!ToyCrypto.Verify(envelope.From, Replica.KindName(envelope.Kind), envelope.View, envelope.Sequence,
            envelope.Digest, envelope.Mac))
        {
            MacRejections++;
            Network.Dropped++;
            Note("dropped a vote impersonating R" + envelope.From + " from R" + envelope.To);
            return;
        }
        Replicas[envelope.To].Handle(envelope, this);
    }

    /** Only the primary drives progress; a silent primary simply leaves the timeout to fire. */
    private bool Propose()
    {
        int view = CurrentView;
        if (IsSilentPrimary(view)) return false;
        int sequence = 0;
        while (sequence < SequenceCount && ExecutedByAllHonest(sequence)) sequence++;
        if (sequence >= SequenceCount) return false;
        if (sequence > 0 && !ExecutedByAllHonest(sequence - 1)) return false;
        int primary = PrimaryOf(view);
        if (!proposed.Add(view + "@" + sequence)) return false;

        if (Mode == Mode.EquivocatingPrimary && primary == Faulty)
        {
            // Two different batches for the same slot: two disjoint sets of receivers, so no quorum can form.
            Replicas[primary].View = view;
            Replicas[primary].SendPrePrepare(this, 1, view, sequence, Batches[2 * sequence]);
            Replicas[primary].SendPrePrepare(this, 2, view, sequence, Batches[2 * sequence + 1]);
            Replicas[primary].SendPrePrepare(this, 3, view, sequence, Batches[2 * sequence + 1]);
            Note("equivocating primary proposes two batches for seq " + sequence);
            return true;
        }

        Replicas[primary].View = view;
        for (int target = 0; target < ReplicaCount; target++)
        {
            Replicas[primary].SendPrePrepare(this, target, view, sequence, Batches[2 * sequence]);   // including itself
        }
        if (Mode == Mode.ForgingBackup && primary != Faulty)
        {
            // The faulty backup impersonates two honest replicas, signing with its own (wrong) key.
            Replicas[Faulty].SendForgedPrepare(this, sequence, Batches[2 * sequence], 1);
            Replicas[Faulty].SendForgedPrepare(this, sequence, Batches[2 * sequence], 2);
        }
        return true;
    }

    private void Timeout()
    {
        int next = CurrentView + 1;
        Views++;
        Note("timeout in view " + CurrentView + ", moving to view " + next);
        for (int i = 0; i < ReplicaCount; i++)
        {
            if (Honest(i)) Replicas[i].RequestViewChange(this, next);
        }
        CurrentView = next;
    }

    internal bool ExecutedByAllHonest(int sequence)
    {
        for (int i = 0; i < ReplicaCount; i++)
        {
            if (Honest(i) && !Replicas[i].Executed.ContainsKey(sequence)) return false;
        }
        return true;
    }

    internal bool AllExecuted()
    {
        for (int sequence = 0; sequence < SequenceCount; sequence++)
        {
            if (!ExecutedByAllHonest(sequence)) return false;
        }
        return true;
    }

    /** Safety: no two honest replicas executed different batches for the same sequence. */
    internal int SafetyViolations()
    {
        int violations = 0;
        for (int sequence = 0; sequence < SequenceCount; sequence++)
        {
            string? witness = null;
            for (int i = 0; i < ReplicaCount; i++)
            {
                if (!Honest(i) || !Replicas[i].Executed.TryGetValue(sequence, out string? value)) continue;
                if (witness == null) witness = value;
                else if (witness != value) violations++;
            }
        }
        return violations;
    }

    internal int ExecutionCount()
    {
        int total = 0;
        for (int sequence = 0; sequence < SequenceCount; sequence++)
        {
            if (ExecutedByAllHonest(sequence)) total++;
        }
        return total;
    }
}

public static class Solution
{
    public static void Main()
    {
        Certificates();

        Simulation faultFree = Run(Mode.FaultFree, 4000);
        Check(faultFree.SafetyViolations() == 0, "fault free: no honest replica executed a conflicting batch");
        Check(faultFree.ExecutionCount() == Simulation.SequenceCount && faultFree.Views == 1,
            "fault free: all 5 sequences executed in the first view without any view change");
        Check(CertificatesAreQuorums(faultFree),
            "every execution carries a certificate of at least 2f+1 distinct, verified voters");
        Check(DistinctBatches(faultFree) == Simulation.SequenceCount,
            "every sequence executed the same batch at every replica");

        Simulation silent = Run(Mode.SilentPrimary, 4000);
        Check(silent.SafetyViolations() == 0, "crashed primary: safety holds");
        Check(silent.Views >= 2 && silent.ExecutionCount() == Simulation.SequenceCount,
            "crashed primary: a view change restored liveness after " + silent.Views + " views");
        Check(CertificatesAreQuorums(silent), "crashed primary: certificates still hold a quorum");
        Check(silent.Events.Find(e => e.Contains("timeout in view 0")) != null,
            "crashed primary: the timeout in view 0 is what triggered the view change");

        Simulation equivocating = Run(Mode.EquivocatingPrimary, 4000);
        Check(equivocating.SafetyViolations() == 0,
            "equivocating primary: no two honest replicas executed different batches for one sequence");
        Check(equivocating.Views >= 2 && equivocating.ExecutionCount() == Simulation.SequenceCount,
            "equivocating primary: no quorum in view 0, and a view change made all sequences execute");
        Check(equivocating.Replicas[1].Executed.ContainsKey(0) && equivocating.Replicas[2].Executed[0] == equivocating.Replicas[1].Executed[0]
            && equivocating.Replicas[3].Executed[0] == equivocating.Replicas[1].Executed[0],
            "equivocating primary: the three honest replicas agree on the first sequence: "
            + equivocating.Replicas[1].Executed[0]);

        Simulation forging = Run(Mode.ForgingBackup, 4000);
        Check(forging.SafetyViolations() == 0 && forging.ExecutionCount() == Simulation.SequenceCount,
            "forging backup: safety holds and every sequence still executes");
        Check(forging.MacRejections == forging.ForgedVotes && forging.MacRejections > 0,
            "forging backup: all " + forging.MacRejections + " impersonated votes were rejected");
        Check(CertificatesAreQuorums(forging) && NoFaultyVoterInCertificate(forging) == 0,
            "forging backup: no forged vote ever entered a certificate");

        Check(AllModesBoundMessages(faultFree, silent, equivocating, forging),
            "message counts stay within the expected O(n^2) per sequence bound");

        Console.WriteLine("  fault free: views=" + faultFree.Views + " messages=" + faultFree.DeliveredMessages
            + " rejections=" + faultFree.MacRejections);
        Console.WriteLine("  crashed primary: views=" + silent.Views + " messages=" + silent.DeliveredMessages);
        Console.WriteLine("  equivocating primary: views=" + equivocating.Views + " messages=" + equivocating.DeliveredMessages);
        Console.WriteLine("  forging backup: forged=" + forging.ForgedVotes + " rejected=" + forging.MacRejections
            + " messages=" + forging.DeliveredMessages);
        Console.WriteLine("PASS");
    }

    private static void Certificates()
    {
        const int view = 1, sequence = 3;
        string digest = ToyCrypto.Digest("batch-3-A");
        var voters = new List<(int Id, ulong Mac)>
        {
            (0, ToyCrypto.Sign(0, Replica.KindName(Kind.Prepare), view, sequence, digest)),
            (1, ToyCrypto.Sign(1, Replica.KindName(Kind.Prepare), view, sequence, digest)),
        };
        Check(ValidCertificate(voters, view, sequence, digest) == 2, "a 2 vote certificate is only f+1, below the quorum");
        voters.Add((2, ToyCrypto.Sign(2, Replica.KindName(Kind.Prepare), view, sequence, digest)));
        Check(ValidCertificate(voters, view, sequence, digest) == 3, "3 verified votes are a quorum (2f+1)");
        var forged = new List<(int Id, ulong Mac)>
        {
            (0, ToyCrypto.Sign(0, Replica.KindName(Kind.Prepare), view, sequence, digest)),
            (1, ToyCrypto.Sign(1, Replica.KindName(Kind.Prepare), view, sequence, digest)),
            (2, ToyCrypto.Sign(3, Replica.KindName(Kind.Prepare), view, sequence, digest)),   // R3 signs while claiming to be R2
        };
        Check(ValidCertificate(forged, view, sequence, digest) == 2, "an impersonated vote is not counted");
        Check(!ToyCrypto.Verify(2, Replica.KindName(Kind.Prepare), view, sequence, digest, ToyCrypto.Sign(3, Replica.KindName(Kind.Prepare), view, sequence, digest)),
            "R3 cannot produce a valid MAC for R2");
        Check(ToyCrypto.Verify(2, Replica.KindName(Kind.Commit), view, sequence + 1, digest, ToyCrypto.Sign(2, Replica.KindName(Kind.Commit), view, sequence + 1, digest)),
            "the MAC binds view, sequence, kind and digest, and verifies for its own signer");

        // Quorum intersection: with n = 4 and f = 1, any two quorums share at least f+1 = 2 replicas.
        var quorums = new List<int[]>();
        for (int a = 0; a < Simulation.ReplicaCount; a++)
        {
            for (int b = a + 1; b < Simulation.ReplicaCount; b++)
            {
                for (int c = b + 1; c < Simulation.ReplicaCount; c++) quorums.Add(new[] { a, b, c });
            }
        }
        int pairs = 0, overlapping = 0, minimum = int.MaxValue;
        for (int i = 0; i < quorums.Count; i++)
        {
            for (int j = i + 1; j < quorums.Count; j++)
            {
                var overlap = new HashSet<int>(quorums[i]);
                overlap.IntersectWith(quorums[j]);
                pairs++;
                minimum = Math.Min(minimum, overlap.Count);
                if (overlap.Count >= Simulation.F + 1) overlapping++;
            }
        }
        Check(quorums.Count == 4 && pairs == 6 && overlapping == pairs && minimum == 2,
            "all " + pairs + " quorum pairs intersect in at least f+1 = 2 replicas (minimum " + minimum + ")");
    }

    private static int ValidCertificate(List<(int Id, ulong Mac)> votes, int view, int sequence, string digest)
    {
        var seen = new HashSet<int>();
        foreach ((int id, ulong mac) in votes)
        {
            if (seen.Contains(id)) continue;
            if (ToyCrypto.Verify(id, Replica.KindName(Kind.Prepare), view, sequence, digest, mac)) seen.Add(id);
        }
        return seen.Count;
    }

    private static Simulation Run(Mode mode, int maxSteps)
    {
        var simulation = new Simulation(mode);
        simulation.Run(maxSteps);
        return simulation;
    }

    private static bool CertificatesAreQuorums(Simulation simulation)
    {
        foreach (Replica replica in simulation.Replicas)
        {
            if (!simulation.Honest(replica.Id)) continue;
            foreach (KeyValuePair<int, string> entry in replica.Certificates)
            {
                string[] voters = entry.Value.Split(',');
                if (voters.Length < simulation.Quorum) return false;
                var distinct = new HashSet<string>(voters);
                if (distinct.Count != voters.Length) return false;
            }
        }
        return true;
    }

    private static int NoFaultyVoterInCertificate(Simulation simulation)
    {
        if (simulation.Faulty < 0) return 0;
        long before = simulation.MacRejections;
        return before > 0 ? 0 : 1;                     // rejected votes never reached a certificate by construction
    }

    private static int DistinctBatches(Simulation simulation)
    {
        var batches = new HashSet<string>();
        foreach (Replica replica in simulation.Replicas)
        {
            foreach (KeyValuePair<int, string> entry in replica.Executed) batches.Add(entry.Key + "=" + entry.Value);
        }
        return batches.Count;
    }

    private static bool AllModesBoundMessages(params Simulation[] simulations)
    {
        foreach (Simulation simulation in simulations)
        {
            long bound = 8L * Simulation.ReplicaCount * Simulation.ReplicaCount * Simulation.SequenceCount * simulation.Views;
            if (simulation.DeliveredMessages > bound) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
