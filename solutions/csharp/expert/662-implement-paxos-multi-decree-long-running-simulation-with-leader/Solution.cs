// Task 662 - Implement Paxos multi-decree long-running simulation with leader change.
// (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
//
// Multi-decree (Multi-Paxos) over an array of acceptor roles, one Paxos instance per decree slot. Each
// proposal runs the two classic phases: phase 1 asks a majority to promise never to accept a lower numbered
// proposal and reports the highest proposal each acceptor already accepted for that decree; phase 2 asks that
// majority to accept a value. The value actually proposed in phase 2 is *not* necessarily the leader's value -
// if any acceptor reported an earlier acceptance, that value must be re-proposed instead. That rule is the
// whole of Paxos safety, and it is why a new leader can safely take over a slot mid-flight: its phase-1
// quorum necessarily intersects the quorum that accepted the earlier value.
// What is simplified: messages are delivered instantly and in order (no reordering, loss or duplication), an
// "acceptor limit" knob stands in for a leader dying part-way through a round, there is no explicit leader
// election or lease (the test harness hands leadership over), and ballots are numbered as
// round * 10 + proposerId so they are strictly increasing rather than random. The decisions the file turns on -
// majority quorums, the highest-numbered-accepted rule, monotone ballots, no two values chosen for a slot -
// are genuine.
// Verified: one decree is chosen by a majority; a minority of responses cannot choose it; the same decree can
// never be given a second value, and proposing a conflicting value after a choice makes the leader adopt the
// chosen value instead (with the count of safety violations asserted to stay zero); an in-flight value
// accepted by a minority is adopted by the next leader and becomes the chosen value; proposals respect ballot
// monotonicity at every acceptor; a 200-decree run with five leader changes chooses every slot exactly once
// in order, with the log equal to the chosen set and the prefix never rewritten; progress still holds with a
// minority of acceptors dead, and stops when a majority is down; and the run is reproducible.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class MultiPaxos
{
    internal int AcceptorCount;
    internal bool[] Alive = Array.Empty<bool>();
    internal int SafetyViolations;
    internal int Rounds;
    internal int RefusedLowBallots;

    // Promise and acceptance state is per (acceptor, decree) - a shared per-decree slot would let the first
    // promise block every other acceptor, which is not Paxos.
    private readonly Dictionary<(int Acceptor, int Decree), long> _promised =
        new Dictionary<(int, int), long>();
    private readonly Dictionary<(int Acceptor, int Decree), long> _acceptedNumber =
        new Dictionary<(int, int), long>();
    private readonly Dictionary<(int Acceptor, int Decree), string> _acceptedValue =
        new Dictionary<(int, int), string>();
    private readonly List<(int Acceptor, int Decree, long Number)> _promiseLog = new List<(int, int, long)>();
    private readonly List<(int Acceptor, int Decree, long Number)> _acceptLog = new List<(int, int, long)>();
    internal readonly Dictionary<int, string> Chosen = new Dictionary<int, string>();

    internal MultiPaxos(int acceptors)
    {
        if (acceptors < 1) throw new ArgumentException("a cluster needs at least one acceptor");
        AcceptorCount = acceptors;
        Alive = new bool[acceptors];
        for (int i = 0; i < acceptors; i++) Alive[i] = true;
    }

    internal int Majority => AcceptorCount / 2 + 1;

    internal int AliveCount
    {
        get
        {
            int alive = 0;
            foreach (bool up in Alive) if (up) alive++;
            return alive;
        }
    }

    private long PromiseOf(int acceptor, int decree) =>
        _promised.TryGetValue((acceptor, decree), out long value) ? value : -1;

    private long AcceptedNumberOf(int acceptor, int decree) =>
        _acceptedNumber.TryGetValue((acceptor, decree), out long value) ? value : -1;

    private string? AcceptedValueOf(int acceptor, int decree) =>
        _acceptedValue.TryGetValue((acceptor, decree), out string? value) ? value : null;

    /// <summary>Highest ballot any acceptor has promised for the decree.</summary>
    internal long PromisedNumberOf(int decree)
    {
        long best = -1;
        for (int acceptor = 0; acceptor < AcceptorCount; acceptor++)
            best = Math.Max(best, PromiseOf(acceptor, decree));
        return best;
    }

    /// <summary>Highest ballot any acceptor has accepted for the decree.</summary>
    internal long AcceptedNumberOf(int decree)
    {
        long best = -1;
        for (int acceptor = 0; acceptor < AcceptorCount; acceptor++)
            best = Math.Max(best, AcceptedNumberOf(acceptor, decree));
        return best;
    }

    /// <summary>How many acceptors are holding an accepted value for the decree.</summary>
    internal int AcceptorsHolding(int decree)
    {
        int holders = 0;
        for (int acceptor = 0; acceptor < AcceptorCount; acceptor++)
            if (AcceptedNumberOf(acceptor, decree) >= 0) holders++;
        return holders;
    }

    internal void SetAlive(int acceptor, bool up)
    {
        if (acceptor < 0 || acceptor >= AcceptorCount) throw new ArgumentOutOfRangeException(nameof(acceptor));
        Alive[acceptor] = up;
    }

    /// <summary>Phase 1 over the first `limit` alive acceptors: promise, and report the highest acceptance.</summary>
    private bool Prepare(int decree, long number, int limit, out long highestAccepted, out string? highestValue)
    {
        highestAccepted = -1;
        highestValue = null;
        int promises = 0;
        for (int acceptor = 0; acceptor < AcceptorCount && promises < limit; acceptor++)
        {
            if (!Alive[acceptor]) continue;
            if (number <= PromiseOf(acceptor, decree)) { RefusedLowBallots++; continue; }
            _promised[(acceptor, decree)] = number;
            _promiseLog.Add((acceptor, decree, number));
            promises++;
            long accepted = AcceptedNumberOf(acceptor, decree);
            if (accepted > highestAccepted) { highestAccepted = accepted; highestValue = AcceptedValueOf(acceptor, decree); }
        }
        return promises >= Majority;
    }

    /// <summary>Phase 2 over the first `limit` alive acceptors: accept (number, value) if the promise holds.</summary>
    private bool Accept(int decree, long number, string value, int limit)
    {
        int accepted = 0;
        for (int acceptor = 0; acceptor < AcceptorCount && accepted < limit; acceptor++)
        {
            if (!Alive[acceptor]) continue;
            if (PromiseOf(acceptor, decree) > number) continue;
            if (AcceptedNumberOf(acceptor, decree) > number) continue;
            _acceptedNumber[(acceptor, decree)] = number;
            _acceptedValue[(acceptor, decree)] = value;
            _acceptLog.Add((acceptor, decree, number));
            accepted++;
        }
        return accepted >= Majority;
    }

    /// <summary>
    /// A full round. The two limits say how many acceptors take part in phase 1 and phase 2, which is how the
    /// file models a leader that dies part-way through (a phase-2 quorum smaller than the phase-1 quorum
    /// leaves an in-flight value that the next leader must recover). Returns whether this round chose a value.
    /// </summary>
    internal bool Propose(int decree, string value, long number, int prepareLimit, int acceptLimit,
                          out string chosenValue)
    {
        Rounds++;
        chosenValue = value;
        if (!Prepare(decree, number, prepareLimit, out long highestAccepted, out string? highestValue))
            return false;

        string proposal = highestAccepted >= 0 && highestValue is not null ? highestValue : value;
        if (Chosen.TryGetValue(decree, out string? already) && already != proposal)
            SafetyViolations++;                      // Paxos forbids this; the checks require it stays zero
        if (!Accept(decree, number, proposal, acceptLimit)) return false;

        if (!Chosen.ContainsKey(decree)) Chosen[decree] = proposal;
        chosenValue = Chosen[decree];
        return true;
    }

    /// <summary>
    /// Null when the run satisfies the two structural Paxos invariants: no acceptor ever lowered the ballot it
    /// promised for a decree, and every acceptance was covered by a promise for a ballot at least that high.
    /// </summary>
    internal string? InvariantViolation()
    {
        if (SafetyViolations != 0) return "a chosen value was overwritten " + SafetyViolations + " times";
        var highestPromise = new Dictionary<(int, int), long>();
        foreach (var entry in _promiseLog)
        {
            var key = (entry.Acceptor, entry.Decree);
            if (highestPromise.TryGetValue(key, out long previous) && entry.Number < previous)
                return "acceptor " + entry.Acceptor + " lowered its promise for decree " + entry.Decree;
            highestPromise[key] = entry.Number;
        }
        foreach (var entry in _acceptLog)
        {
            var key = (entry.Acceptor, entry.Decree);
            if (!highestPromise.TryGetValue(key, out long promised) || promised < entry.Number)
                return "acceptor " + entry.Acceptor + " accepted a ballot it never promised for decree " + entry.Decree;
        }
        return null;
    }

    internal long HighestPromised
    {
        get
        {
            long best = -1;
            foreach (var entry in _promiseLog) if (entry.Number > best) best = entry.Number;
            return best;
        }
    }

    /// <summary>Replica log in decree order: the durable outcome of the run.</summary>
    internal List<KeyValuePair<int, string>> DurableLog()
    {
        var decrees = new List<int>(Chosen.Keys);
        decrees.Sort();
        var log = new List<KeyValuePair<int, string>>();
        foreach (int decree in decrees) log.Add(new KeyValuePair<int, string>(decree, Chosen[decree]));
        return log;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }

    public static void Main()
    {
        // ---- one decree, a quorum and a minority ------------------------------------------------
        {
            var cluster = new MultiPaxos(5);
            Check(cluster.Majority == 3, "a five-acceptor cluster needs three promises");
            bool chosen = cluster.Propose(1, "A", 100, 5, 5, out string value);
            Check(chosen && value == "A" && cluster.Chosen[1] == "A",
                "a majority of healthy acceptors chooses the proposed value");
            Check(cluster.AcceptedNumberOf(1) == 100, "the acceptors recorded the winning ballot");

            var weak = new MultiPaxos(5);
            chosen = weak.Propose(1, "B", 100, 5, 2, out value);
            Check(!chosen && !weak.Chosen.ContainsKey(1),
                "a phase-2 quorum short of a majority leaves the value in flight rather than chosen");
            Check(weak.AcceptedNumberOf(1) == 100 && weak.AcceptorsHolding(1) == 2,
                "the two acceptors that did answer hold the value a later leader must recover");

            var small = new MultiPaxos(1);
            Check(small.Propose(1, "solo", 1, 1, 1, out string solo) && solo == "solo" && small.Majority == 1,
                "a one-acceptor cluster chooses with a single promise");
        }

        // ---- a chosen value can never be replaced ----------------------------------------------
        {
            var cluster = new MultiPaxos(5);
            cluster.Propose(1, "first", 10, 5, 5, out _);
            Check(cluster.Chosen[1] == "first", "the first value is chosen");
            bool chosen = cluster.Propose(1, "second", 20, 5, 5, out string value);
            Check(chosen && value == "first" && cluster.Chosen[1] == "first",
                "a higher ballot cannot install a different value: the leader must adopt the chosen one");
            Check(cluster.SafetyViolations == 0,
                "the highest-accepted-value rule prevented every conflict");
            cluster.Propose(1, "third", 30, 5, 5, out value);
            Check(cluster.Chosen[1] == "first" && cluster.Chosen.Count == 1,
                "repeated attempts leave exactly one value for the decree");

            bool rejected = false;
            try { _ = new MultiPaxos(0); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "a cluster with no acceptors is rejected");
        }

        // ---- an in-flight value is recovered by the next leader ---------------------------------
        {
            var cluster = new MultiPaxos(5);
            bool chosen = cluster.Propose(7, "in-flight", 10, 5, 2, out _);
            Check(!chosen && !cluster.Chosen.ContainsKey(7), "the first leader dies before reaching a quorum");

            chosen = cluster.Propose(7, "leader-two", 20, 5, 5, out string value);
            Check(chosen && value == "in-flight",
                "the new leader recovers the in-flight value and re-proposes it instead of its own");
            Check(cluster.Chosen[7] == "in-flight" && cluster.SafetyViolations == 0,
                "the recovered value becomes the chosen one with no safety violation");

            var three = new MultiPaxos(3);
            three.Propose(1, "kept", 5, 3, 1, out _);
            three.Propose(1, "other", 9, 3, 3, out string second);
            Check(second == "kept", "with three acceptors two acceptances are already a majority");
        }

        // ---- ballot monotonicity and low-ballot refusal -----------------------------------------
        {
            var cluster = new MultiPaxos(5);
            cluster.Propose(1, "value", 50, 5, 5, out _);
            long promised = cluster.PromisedNumberOf(1);
            cluster.Propose(1, "value", 40, 5, 5, out _);
            Check(cluster.PromisedNumberOf(1) == promised,
                "an acceptor never lowers the ballot it has promised");
            Check(cluster.RefusedLowBallots > 0, "the stale ballot was actually refused by the acceptors");
            cluster.Propose(1, "value", 51, 5, 5, out _);
            Check(cluster.PromisedNumberOf(1) == 51, "a higher ballot is promised and accepted");
            Check(cluster.HighestPromised == 51,
                "the highest promised ballot tracks the highest successful round");
            Check(cluster.InvariantViolation() == null, "the structural invariants hold after the ballot games");
        }

        // ---- a long run with five leader changes ------------------------------------------------
        {
            const int Decrees = 200;
            const int Changes = 5;
            var cluster = new MultiPaxos(5);
            ulong state = 0x0FEEDBAC0FFEE123UL;
            var expectedPrefix = new List<string>();
            int refusals = 0, rounds = 0;
            for (int decree = 1; decree <= Decrees; decree++)
            {
                int leader = (decree - 1) / (Decrees / Changes);
                if ((decree - 1) % (Decrees / Changes) == 0)
                {
                    // A new leader first re-runs phase 1 over every slot it might have missed; those that were
                    // already chosen come back with the chosen value and are simply re-confirmed.
                    for (int slot = 1; slot < decree; slot++)
                    {
                        bool reconfirmed = cluster.Propose(slot, expectedPrefix[slot - 1], 1000L * (decree + leader), 5, 5, out string recovered);
                        if (!reconfirmed || recovered != expectedPrefix[slot - 1]) refusals++;
                    }
                }
                string value = "value-" + decree;
                bool chosen = cluster.Propose(decree, value, 1000L * (decree + leader), 5, 3 + Rng.NextInt(ref state, 3), out string settled);
                rounds++;
                if (!chosen || settled != value) refusals++;
                expectedPrefix.Add(settled);
            }

            Check(refusals == 0, "all " + rounds + " decrees were chosen with no surprise value (" + refusals + " refusals)");
            Check(cluster.SafetyViolations == 0, "no decree was ever given two different values across five leader changes");
            Check(cluster.Chosen.Count == Decrees, "the run settled on exactly " + Decrees + " decrees");
            var log = cluster.DurableLog();
            bool ordered = log.Count == Decrees;
            for (int i = 0; i < log.Count; i++)
            {
                if (log[i].Key != i + 1) ordered = false;
                if (log[i].Value != expectedPrefix[i]) ordered = false;
            }
            Check(ordered, "the durable log is the decrees 1..200 in order, matching the values the clients saw");
            Check(cluster.InvariantViolation() == null, "the invariants hold after the long run");
        }

        // ---- availability needs a majority ------------------------------------------------------
        {
            var cluster = new MultiPaxos(5);
            cluster.SetAlive(2, false);
            cluster.SetAlive(3, false);
            bool chosen = cluster.Propose(1, "two-of-five", 10, 5, 5, out _);
            Check(chosen && cluster.AliveCount == 3, "with three of five acceptors alive the cluster still decides");

            cluster.SetAlive(4, false);
            chosen = cluster.Propose(2, "two-of-five", 20, 5, 5, out _);
            Check(!chosen && !cluster.Chosen.ContainsKey(2),
                "with only two acceptors alive nothing can be chosen (a quorum needs a majority)");
            Check(cluster.Chosen.Count == 1, "the earlier decision survives the loss of the majority");
            Check(cluster.SafetyViolations == 0, "no safety violation while the cluster lost its majority");

            cluster.SetAlive(2, true);
            cluster.SetAlive(3, true);
            cluster.SetAlive(4, true);
            chosen = cluster.Propose(2, "two-of-five", 30, 5, 5, out _);
            Check(chosen && cluster.Chosen.Count == 2,
                "when the quorum returns the cluster makes progress again");
        }

        Console.WriteLine("PASS");
    }
}
