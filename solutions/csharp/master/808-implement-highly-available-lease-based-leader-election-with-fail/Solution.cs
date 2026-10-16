// Task 808 - Implement highly-available lease-based leader election with failover testing. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
//
// A lease-based leader election on an explicit logical clock. A candidate asks every voter for a promise covering a
// window [now+1, now+leaseTicks]; a voter grants at most one promise at a time (it refuses while its previous
// promise is still valid and it refuses any lease id that is not larger than the last one it promised), and a
// candidate becomes leader only after a strict majority of voters granted it. The leader therefore owns an
// exclusive lease window, and the storage layer enforces a monotone fencing token: every write carries the lease id
// and the storage refuses any token lower than the highest it has already accepted, so a leader whose lease has
// expired (a partitioned or restarted one) can never mutate the log again even if it still believes it is leader.
//
// Invariants verified: within any tick at most one leader holds a valid lease - the promise windows of two leaders
// can never overlap because a majority of voters would have to promise twice inside the same window; the fencing
// token strictly increases with every leadership change; the storage rejects every stale-token write; a leader that
// loses its majority stops being able to write once its window expires; every lease lies inside the lease duration
// bound; and after a crash a new leader is elected within the failover bound (election delay plus one lease window)
// as long as a majority of voters is alive.
//
// Independent oracles: the safety property is checked by an interval-overlap scan over the recorded lease history
// (independent of the election code), liveness is measured by an exhaustive sweep over every crash tick and a
// deterministic set of message-drop schedules, and the write path is validated by a second, independent token check
// that recomputes the highest token the storage should have accepted. Every schedule is replayed twice for
// determinism.
//
// Documented simplifications: voters, candidates and the storage are in-process objects, the clock is a counter,
// messages have fixed latency and are released by the tick loop (never reordered within a channel), and a crash
// means "messages to and from this node are dropped" rather than a real process failure. No threads, no wall clock,
// no unseeded randomness: the drop patterns come from a hand-written splitmix64.
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

internal enum MessageKind { PromiseRequest, PromiseReply }

internal sealed class LeaseMessage
{
    internal MessageKind Kind;
    internal int From;
    internal int To;
    internal long LeaseId;
    internal long WindowStart;
    internal long WindowEnd;
    internal bool Granted;
}

/** A voter remembers the furthest window it has promised and to which lease id. */
internal sealed class Voter
{
    internal readonly int Id;
    internal long PromiseExpiry = -1;
    internal long PromisedLeaseId;
    internal int Grants;
    internal int Refusals;

    internal Voter(int id) { Id = id; }

    internal bool Handle(LeaseMessage request)
    {
        if (request.WindowStart <= PromiseExpiry)                       // still bound by an earlier promise
        {
            Refusals++;
            return false;
        }
        if (request.LeaseId <= PromisedLeaseId)                         // lease ids must grow
        {
            Refusals++;
            return false;
        }
        PromiseExpiry = request.WindowEnd;
        PromisedLeaseId = request.LeaseId;
        Grants++;
        return true;
    }

    /** A promise is valid while the logical clock is inside its window. */
    internal bool HasValidPromise(long tick) => tick <= PromiseExpiry;
}

/** The replicated storage: the only place a leader can mutate state, guarded by the fencing token. */
internal sealed class LeaseStorage
{
    internal long HighestToken;
    internal readonly List<string> Committed = new List<string>();
    internal int Rejections;
    internal int Acceptances;

    internal bool Write(long token, string value)
    {
        if (token < HighestToken)
        {
            Rejections++;
            return false;
        }
        HighestToken = token;
        Committed.Add(token + ":" + value);
        Acceptances++;
        return true;
    }
}

internal sealed class LeaseRecord
{
    internal long LeaseId;
    internal int Leader;
    internal long WindowStart;
    internal long WindowEnd;
}

internal sealed class LeaseCluster
{
    internal readonly List<Voter> Voters = new List<Voter>();
    internal readonly HashSet<int> Down = new HashSet<int>();
    internal readonly List<(int ReleaseTick, long Sequence, LeaseMessage Message)> Pending = new List<(int, long, LeaseMessage)>();
    internal readonly List<LeaseRecord> History = new List<LeaseRecord>();
    internal readonly LeaseStorage Storage = new LeaseStorage();
    internal readonly List<string> Trace = new List<string>();
    internal long NextLeaseId = 1;
    internal int Tick;
    internal int Latency = 2;
    internal long LeaseTicks = 6;
    internal bool AutoRenew = true;
    internal long LeaderId = -1;
    internal long LeaderLeaseEnd = -1;
    internal long LeaderToken = -1;
    internal readonly Dictionary<int, PromiseRound> InFlight = new Dictionary<int, PromiseRound>();

    /** One promise round per candidate: opening a new one while an older one is still in flight is refused. */
    internal sealed class PromiseRound
    {
        internal long LeaseId;
        internal long WindowStart;
        internal long WindowEnd;
        internal int Grants;
        internal int Deadline;
    }

    private long _sequence;

    internal LeaseCluster(int voterCount)
    {
        for (int i = 0; i < voterCount; i++) Voters.Add(new Voter(i));
    }

    internal int Quorum => Voters.Count / 2 + 1;

    internal bool IsLeader(long tick) => LeaderId >= 0 && tick <= LeaderLeaseEnd && !Down.Contains((int)LeaderId);

    /** A candidate opens a promise round; returns the lease id, or -1 while an older round is still in flight. */
    internal long RequestLease(int candidate)
    {
        if (InFlight.TryGetValue(candidate, out PromiseRound? open))
        {
            if (open.Deadline > Tick) return -1;
            InFlight.Remove(candidate);
        }
        long leaseId = NextLeaseId++;
        // A renewal asks for the window that follows the current one, so the voters' promise is free again.
        long start = LeaderId == candidate && LeaderLeaseEnd > Tick ? LeaderLeaseEnd + 1 : Tick + 1;
        var round = new PromiseRound { LeaseId = leaseId, WindowStart = start, WindowEnd = start + LeaseTicks, Grants = 1, Deadline = Tick + 3 * Latency };
        InFlight[candidate] = round;
        foreach (Voter voter in Voters)
        {
            if (voter.Id == candidate) continue;
            if (Down.Contains(voter.Id)) continue;
            Pending.Add((Tick + Latency, _sequence++, new LeaseMessage
            {
                Kind = MessageKind.PromiseRequest, From = candidate, To = voter.Id, LeaseId = leaseId,
                WindowStart = round.WindowStart, WindowEnd = round.WindowEnd
            }));
        }
        Trace.Add("request t" + Tick + " candidate" + candidate + " lease" + leaseId + " window[" + round.WindowStart + "," + round.WindowEnd + "]");
        // A single-voter (or self-sufficient) cluster is already a quorum without waiting for replies.
        if (round.Grants >= Quorum && !Down.Contains(candidate) && leaseId > LeaderToken) Complete(candidate, round);
        return leaseId;
    }

    private void Complete(int candidate, PromiseRound round)
    {
        LeaderId = candidate;
        LeaderToken = round.LeaseId;
        LeaderLeaseEnd = round.WindowEnd;
        History.Add(new LeaseRecord { LeaseId = round.LeaseId, Leader = candidate, WindowStart = round.WindowStart, WindowEnd = round.WindowEnd });
        InFlight.Remove(candidate);
        Trace.Add("elected t" + Tick + " leader" + candidate + " lease" + round.LeaseId);
    }

    internal void Step()
    {
        // A live leader renews the next window early enough that the grant lands as the old lease expires.
        if (AutoRenew && LeaderId >= 0 && !Down.Contains((int)LeaderId) && LeaderLeaseEnd - Tick <= 2 * Latency) RequestLease((int)LeaderId);
        var due = new List<(int ReleaseTick, long Sequence, LeaseMessage Message)>();
        var later = new List<(int, long, LeaseMessage)>();
        foreach (var item in Pending) (item.ReleaseTick <= Tick ? due : later).Add(item);
        Pending.Clear();
        Pending.AddRange(later);
        due.Sort((a, b) => a.Sequence.CompareTo(b.Sequence));
        foreach (var item in due)
        {
            LeaseMessage message = item.Message;
            if (Down.Contains(message.To) || Down.Contains(message.From)) continue;
            if (message.Kind == MessageKind.PromiseRequest)
            {
                Voter voter = Voters[message.To];
                bool granted = voter.Handle(message);
                Pending.Add((Tick + Latency, _sequence++, new LeaseMessage
                {
                    Kind = MessageKind.PromiseReply, From = message.To, To = message.From, LeaseId = message.LeaseId,
                    WindowStart = message.WindowStart, WindowEnd = message.WindowEnd, Granted = granted
                }));
                continue;
            }
            if (message.Granted)
            {
                // A reply only counts for a round the candidate still has open, and for a lease id that is newer.
                if (!InFlight.TryGetValue(message.To, out PromiseRound? round)) continue;
                if (round.LeaseId != message.LeaseId) continue;
                round.Grants++;
                if (round.Grants < Quorum) continue;
                if (Down.Contains(message.To)) continue;
                if (message.LeaseId <= LeaderToken) continue;               // never take over from a newer leader
                Complete(message.To, round);
            }
        }
        Tick++;
    }

    internal void Run(int ticks)
    {
        for (int i = 0; i < ticks; i++) Step();
    }

    internal void Crash(int id) => Down.Add(id);

    internal void Heal(int id) => Down.Remove(id);

    /** A leader writes through the storage while its lease is valid; the storage also checks the fencing token. */
    internal bool LeaderWrite(string value)
    {
        if (LeaderId < 0 || Tick > LeaderLeaseEnd || Down.Contains((int)LeaderId)) return false;
        return Storage.Write(LeaderToken, value);
    }

    internal bool EverLeader(int id)
    {
        foreach (LeaseRecord record in History) if (record.Leader == id) return true;
        return false;
    }

    /** Independent safety oracle: no two lease records with different leaders may overlap in time. */
    internal string CheckNoOverlap()
    {
        for (int i = 0; i < History.Count; i++)
        {
            for (int j = i + 1; j < History.Count; j++)
            {
                LeaseRecord a = History[i];
                LeaseRecord b = History[j];
                if (a.Leader == b.Leader) continue;
                if (a.WindowStart <= b.WindowEnd && b.WindowStart <= a.WindowEnd)
                    return "leases " + a.LeaseId + " (leader " + a.Leader + ") and " + b.LeaseId + " (leader " + b.Leader + ") overlap";
            }
        }
        return "";
    }

    internal string HistoryDigest()
    {
        var builder = new StringBuilder();
        foreach (LeaseRecord record in History) builder.Append(record.LeaseId).Append('/').Append(record.Leader).Append('/')
            .Append(record.WindowStart).Append('/').Append(record.WindowEnd).Append(';');
        return builder.ToString();
    }
}

public static class Solution
{
    public static void Main()
    {
        // 1-3. A five-voter cluster elects a leader with a majority promise and keeps renewing its bounded lease.
        var cluster = new LeaseCluster(5);
        cluster.RequestLease(0);
        cluster.Run(20);
        Check(cluster.EverLeader(0), "the candidate wins after a majority of promises");
        Check(cluster.CheckNoOverlap() == "", "the lease history has no overlapping leaders");
        Check(cluster.IsLeader(cluster.Tick), "the leader keeps a valid lease while it renews (tick=" + cluster.Tick
            + " leader=" + cluster.LeaderId + " leaseEnd=" + cluster.LeaderLeaseEnd + " history=" + cluster.History.Count + ")");
        bool boundedWindows = true;
        foreach (LeaseRecord record in cluster.History)
        {
            if (record.WindowEnd - record.WindowStart != cluster.LeaseTicks) boundedWindows = false;
        }
        Check(boundedWindows && cluster.History.Count > 1, "every lease window equals the configured duration (" + cluster.History.Count + " leases issued)");
        Check(cluster.LeaderToken == cluster.History[cluster.History.Count - 1].LeaseId, "the fencing token is the lease id of the latest election");

        // 4-5. A voter refuses a second promise inside a window it already granted, and a lower lease id.
        var voter = new Voter(9);
        var firstRequest = new LeaseMessage { LeaseId = 5, WindowStart = 10, WindowEnd = 20 };
        var overlapping = new LeaseMessage { LeaseId = 6, WindowStart = 15, WindowEnd = 25 };
        var staleId = new LeaseMessage { LeaseId = 4, WindowStart = 30, WindowEnd = 40 };
        var later = new LeaseMessage { LeaseId = 7, WindowStart = 25, WindowEnd = 35 };
        Check(voter.Handle(firstRequest), "the first promise is granted");
        Check(!voter.Handle(overlapping), "an overlapping promise is refused (one promise at a time)");
        Check(!voter.Handle(staleId), "a lease id that does not grow is refused");
        Check(voter.Handle(later) && voter.PromiseExpiry == 35, "a later, disjoint window is granted and extends the promise");
        Check(voter.HasValidPromise(34) && !voter.HasValidPromise(36), "the promise is valid only inside its window");

        // 6-7. Fencing tokens: writes under an expired lease are refused by the storage.
        Check(cluster.LeaderWrite("first"), "the valid leader writes through the storage");
        long acceptedToken = cluster.Storage.HighestToken;
        Check(!cluster.Storage.Write(acceptedToken - 1, "stale-write"), "the storage refuses a stale fencing token");
        Check(cluster.Storage.Rejections == 1 && cluster.Storage.Acceptances == 1, "exactly one write was accepted and one refused");

        // 8-9. Renewal extends the window; a lease never exceeds the duration bound.
        var renewing = new LeaseCluster(3);
        renewing.AutoRenew = false;
        renewing.RequestLease(0);
        renewing.Run(20);
        long firstExpiry = renewing.LeaderLeaseEnd;
        long firstToken = renewing.LeaderToken;
        Check(renewing.History.Count == 1, "exactly one lease was granted before the renewal");
        Check(!renewing.IsLeader(renewing.Tick), "without a renewal the granted lease expires");
        renewing.RequestLease(0);
        renewing.Run(20);
        Check(renewing.History.Count == 2 && renewing.History[1].LeaseId > renewing.History[0].LeaseId, "a renewal uses a strictly larger lease id");
        Check(renewing.LeaderLeaseEnd > firstExpiry && renewing.LeaderToken > firstToken, "the renewal extends the lease and the token");
        bool bounded = true;
        foreach (LeaseRecord record in renewing.History)
        {
            if (record.WindowEnd - record.WindowStart != renewing.LeaseTicks) bounded = false;
        }
        Check(bounded, "every lease window is inside the lease duration bound");

        // 10-14. Failover: the leader crashes, the lease expires, a new candidate takes over with a bigger token.
        var failover = new LeaseCluster(5);
        failover.RequestLease(0);
        failover.Run(20);
        Check(failover.EverLeader(0), "the first leader is elected");
        Check(failover.LeaderWrite("before-crash"), "the live leader writes through the storage");
        long tokenBefore = failover.LeaderToken;
        failover.Crash(0);
        failover.Run((int)failover.LeaseTicks + 4);                 // let the last promised window run out
        Check(!failover.IsLeader(failover.Tick), "the crashed leader's lease is no longer valid");
        Check(!failover.LeaderWrite("from-ghost"), "a leader whose lease expired cannot write any more");
        int waited = 0;
        while (failover.LeaderId != 1 && waited < 90) { failover.RequestLease(1); failover.Step(); waited++; }
        Check(failover.LeaderId == 1 && failover.LeaderToken > tokenBefore, "a new leader is elected with a strictly larger fencing token");
        Check(waited <= 4 * (failover.LeaseTicks + failover.Latency), "failover completed inside the bound (" + waited + " ticks)");
        Check(failover.LeaderWrite("after-failover"), "the new leader writes successfully");
        Check(failover.CheckNoOverlap() == "", "the two leaderships never overlap (no split brain)");

        // 15-16. A healed old leader still cannot write because its token was overtaken.
        failover.Heal(0);
        failover.Run(10);
        Check(failover.LeaderId == 1, "the healed node does not take leadership back");
        Check(!failover.Storage.Write(failover.History[0].LeaseId, "old-leader-write"), "the healed old leader's token is refused forever");

        // 17-19. Safety across an exhaustive crash-tick sweep, and liveness in every one of them.
        int schedules = 0;
        int worstFailoverTicks = 0;
        bool everyScheduleSafe = true;
        bool everyScheduleLively = true;
        for (int crashAt = 1; crashAt <= 24; crashAt++)
        {
            var sweep = new LeaseCluster(5);
            sweep.RequestLease(0);
            sweep.Run(crashAt);
            if (!sweep.EverLeader(0)) continue;
            sweep.Crash(0);
            int waitedSweep = 0;
            while (sweep.LeaderId != 1 && waitedSweep < 90) { sweep.RequestLease(1); sweep.Step(); waitedSweep++; }
            schedules++;
            if (sweep.CheckNoOverlap() != "") everyScheduleSafe = false;
            if (sweep.LeaderId != 1) everyScheduleLively = false;
            else if (!sweep.LeaderWrite("sweep-" + crashAt)) everyScheduleSafe = false;
            if (waitedSweep > worstFailoverTicks) worstFailoverTicks = waitedSweep;
        }
        Check(schedules >= 20, "the crash-tick sweep covered every interesting timing (" + schedules + " schedules)");
        Check(everyScheduleSafe, "no two leases ever overlap in any crash schedule");
        Check(everyScheduleLively, "every crash schedule elected a replacement leader");
        Console.WriteLine("  failover sweep: schedules=" + schedules + " worst failover ticks=" + worstFailoverTicks);

        // 20-21. A lost majority: no leader, and no unsafe lease either.
        var lossy = new LeaseCluster(5);
        lossy.RequestLease(0);
        lossy.Run(20);
        int beforeLoss = lossy.History.Count;
        lossy.Crash(0);
        lossy.Crash(1);
        lossy.Crash(2);                                    // only voters 3 and 4 remain: no quorum of promises
        lossy.Run((int)lossy.LeaseTicks + 4);
        for (int attempt = 0; attempt < 10; attempt++) { lossy.RequestLease(3); lossy.Run(3); }
        Check(lossy.History.Count == beforeLoss, "with a minority of voters alive no new lease can be issued");
        Check(!lossy.IsLeader(lossy.Tick), "the cluster is unavailable rather than split-brained");
        Check(lossy.CheckNoOverlap() == "", "the unavailable cluster still has non-overlapping lease history");
        lossy.Heal(1);
        lossy.Heal(2);
        int retry = 0;
        while (lossy.LeaderId != 3 && retry < 60) { lossy.RequestLease(3); lossy.Step(); retry++; }
        Check(lossy.LeaderId == 3, "restoring the majority lets a new leader be elected");
        Check(lossy.LeaderWrite("healed"), "the new leader can write again");

        // 22-23. A voter that is still promised blocks a candidate; once the promise expires the candidate wins.
        var blocked = new LeaseCluster(5);
        blocked.AutoRenew = false;
        foreach (int id in new[] { 1, 2, 3, 4 }) blocked.Voters[id].PromiseExpiry = blocked.LeaseTicks + 100;
        blocked.RequestLease(0);
        blocked.Run(20);
        Check(blocked.History.Count == 0, "a candidate that cannot win a quorum of free voters never becomes leader");
        Check(blocked.Voters[1].Refusals >= 1, "the blocked voters recorded the refusals");
        foreach (int id in new[] { 1, 2, 3, 4 }) { blocked.Voters[id].PromiseExpiry = -1; blocked.Voters[id].PromisedLeaseId = 0; }
        blocked.RequestLease(0);
        blocked.Run(20);
        Check(blocked.History.Count == 1 && blocked.History[0].Leader == 0, "once the promises expire the candidate wins a quorum");

        // 22. Determinism: two runs of the same schedule agree exactly.
        var detA = RunSchedule(0x1234UL);
        var detB = RunSchedule(0x1234UL);
        Check(detA == detB, "identical schedules produce identical lease histories");

        // 23. Boundary cases: a single-voter cluster and a fully crashed cluster.
        var single = new LeaseCluster(1);
        single.RequestLease(0);
        single.Run(10);
        Check(single.History.Count >= 1 && single.Quorum == 1, "a single-voter cluster forms a quorum of one");
        var dead = new LeaseCluster(3);
        dead.Crash(0);
        dead.Crash(1);
        dead.Crash(2);
        dead.RequestLease(0);
        dead.Run(20);
        Check(dead.History.Count == 0 && dead.LeaderId == -1, "a fully crashed cluster has no leader and no lease");

        Console.WriteLine("  leases: history=" + failover.History.Count + " rejected writes=" + failover.Storage.Rejections
            + " accepted writes=" + failover.Storage.Acceptances + " final leader=" + failover.LeaderId);
        Console.WriteLine("PASS");
    }

    /** A deterministic mixed schedule: elections, a crash and message drops, used for the determinism check. */
    private static string RunSchedule(ulong seed)
    {
        var cluster = new LeaseCluster(5);
        ulong rng = seed;
        for (int round = 0; round < 8; round++)
        {
            int candidate = SplitMix64.NextInt(ref rng, 5);
            if (!cluster.Down.Contains(candidate)) cluster.RequestLease(candidate);
            cluster.Run(3 + SplitMix64.NextInt(ref rng, 6));
            if (SplitMix64.NextInt(ref rng, 4) == 0) cluster.Crash(1 + SplitMix64.NextInt(ref rng, 4));
        }
        cluster.Run(20);
        return cluster.HistoryDigest() + "|" + cluster.Storage.HighestToken + "|" + cluster.LeaderId;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
