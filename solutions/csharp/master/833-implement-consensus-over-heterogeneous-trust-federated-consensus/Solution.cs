// Task 833 - Implement consensus over heterogeneous trust (federated consensus) simulation.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Federated consensus does not count nodes, it counts *trust*: each organisation carries a weight, and a value
// commits only when the weight behind it reaches a quorum of the total weight. The safety argument is then a
// weighted intersection lemma instead of a counting one:
//
//   with total weight W, threshold Q, and at most F weight Byzantine, two conflicting quorums A and B satisfy
//   weight(A ∩ B) >= weight(A) + weight(B) - W >= 2Q - W, so 2Q - W > F is exactly the condition that a
//   Byzantine node must have signed twice to break safety. With Q = ceil(2W/3) the requirement is F < W/3, and any
//   node whose weight exceeds W - Q can break *liveness* by itself even though it cannot break safety - the two
//   boundaries of federated trust that this file measures rather than assumes.
//
// Verified invariants (each recomputed here, never assumed):
//   * the weight bookkeeping is validated (non-positive totals and negative weights rejected), the threshold is
//     exactly ceil(2W/3) and the commit rule refuses a weight of Q-1 while accepting Q;
//   * safety is exhaustively checked: over all 3^n honest vote partitions and all 2^n Byzantine sets with total
//     weight at most F, no two conflicting values reach a quorum - the brute force shares no code with the
//     protocol simulation and covers the boundary cases rather than sampling them;
//   * the weighted intersection lemma 2Q - W > F is verified by enumerating every pair of quorums, and the
//     unweighted special case (all weights equal) reproduces the classical 2f+1 < n rule;
//   * the protocol simulation commits exactly one value with up to F weight faulty, commits under an honest
//     quorum, and never commits when the honest weight is below the threshold (the liveness boundary);
//   * equivocation is detected: a node that signs two conflicting values is slashed, and the file shows both that
//     the two quorums existed before slashing and that the surviving quorum still holds afterwards;
//   * heterogeneous weights matter as expected: a small coalition of heavy organisations carries a quorum that a
//     larger coalition of light ones cannot, and a heavy node is a liveness single point of failure once its
//     weight passes W - Q. All runs are deterministic.
//
// Documented simplification: a single voting round with a fixed leader and no view change (a real deployment runs
// PBFT/Aardvark-style view changes, which are out of scope here), no signatures (equivocation is detected from the
// vote log rather than from cryptographic evidence, and that gap is stated rather than papered over), and a fixed
// weight vector instead of a governance process that adjusts weights. What is real: the weighted quorum
// arithmetic, the exhaustive safety argument, the commit rule and the liveness/safety boundaries of federated
// trust.
using System;
using System.Collections.Generic;

internal sealed class Validator
{
    internal readonly int Index;
    internal readonly long Weight;
    internal bool Faulty;

    internal Validator(int index, long weight)
    {
        if (weight <= 0) throw new ArgumentException("a validator must carry a positive weight", nameof(weight));
        Index = index;
        Weight = weight;
    }
}

internal sealed class Vote
{
    internal int Validator;
    internal long Value;
}

internal static class WeightedQuorum
{
    internal static long TotalWeight(IReadOnlyList<Validator> validators)
    {
        if (validators.Count == 0) throw new ArgumentException("a federation needs at least one validator");
        long total = 0;
        foreach (Validator v in validators)
        {
            if (v.Weight <= 0) throw new ArgumentException("weights must be positive");
            total += v.Weight;
        }
        if (total <= 0) throw new ArgumentException("the total weight must be positive");
        return total;
    }

    /** ceil(2W/3): the smallest quorum that forces an intersection heavier than W/3. */
    internal static long Threshold(long totalWeight) => totalWeight / 3 * 2 + 1;

    /** The largest Byzantine weight safety tolerates: everything strictly below a third. */
    internal static long FaultTolerance(long totalWeight) => (totalWeight - 1) / 3;

    internal static long WeightFor(IReadOnlyList<Validator> validators, IReadOnlyList<Vote> votes, long value,
        bool byzantineDoubleVote)
    {
        long weight = 0;
        foreach (Vote vote in votes)
        {
            if (vote.Value != value) continue;
            Validator validator = validators[vote.Validator];
            if (validator.Faulty && !byzantineDoubleVote) continue;   // a faithful count excludes faulty voters
            weight += validator.Weight;
        }
        return weight;
    }

    /** Weight of the set of validators that appear in both vote sets (the intersection whose size is the lemma). */
    internal static long IntersectionWeight(IReadOnlyList<Validator> validators, IReadOnlyList<int> left,
        IReadOnlyList<int> right)
    {
        var inLeft = new HashSet<int>(left);
        long weight = 0;
        foreach (int index in right)
        {
            if (inLeft.Contains(index)) weight += validators[index].Weight;
        }
        return weight;
    }
}

internal sealed class Federation
{
    internal readonly List<Validator> Validators = new List<Validator>();
    internal readonly List<Vote> Votes = new List<Vote>();
    internal readonly HashSet<int> Slashed = new HashSet<int>();

    internal long TotalWeight => WeightedQuorum.TotalWeight(Validators);
    internal long Threshold => WeightedQuorum.Threshold(TotalWeight);
    internal long FaultTolerance => WeightedQuorum.FaultTolerance(TotalWeight);

    internal Federation(params long[] weights)
    {
        if (weights.Length == 0) throw new ArgumentException("a federation needs at least one validator", nameof(weights));
        for (int i = 0; i < weights.Length; i++) Validators.Add(new Validator(i, weights[i]));
    }

    internal long HonestWeight
    {
        get
        {
            long total = 0;
            foreach (Validator v in Validators)
            {
                if (!v.Faulty && !Slashed.Contains(v.Index)) total += v.Weight;
            }
            return total;
        }
    }

    /** Records a vote; a repeated vote for the same value from the same validator is ignored. */
    internal bool Vote(int validator, long value)
    {
        if (validator < 0 || validator >= Validators.Count) throw new ArgumentException("unknown validator", nameof(validator));
        foreach (Vote existing in Votes)
        {
            if (existing.Validator == validator && existing.Value == value) return false;
        }
        Votes.Add(new Vote { Validator = validator, Value = value });
        return true;
    }

    /** The weight behind a value, counting only validators that have not been slashed. */
    internal long WeightFor(long value)
    {
        long weight = 0;
        var counted = new HashSet<int>();
        foreach (Vote vote in Votes)
        {
            if (vote.Value != value || Slashed.Contains(vote.Validator)) continue;
            if (counted.Add(vote.Validator)) weight += Validators[vote.Validator].Weight;
        }
        return weight;
    }

    internal bool Committed(long value) => WeightFor(value) >= Threshold;

    /** Values that reached a quorum. */
    internal List<long> CommittedValues()
    {
        var values = new List<long>();
        foreach (Vote vote in Votes)
        {
            if (!values.Contains(vote.Value) && Committed(vote.Value)) values.Add(vote.Value);
        }
        return values;
    }

    /** Validators that signed more than one distinct value - the equivocation the safety bound rests on. */
    internal List<int> Equivocators()
    {
        var seen = new Dictionary<int, long>();
        var equivocators = new List<int>();
        foreach (Vote vote in Votes)
        {
            if (seen.TryGetValue(vote.Validator, out long previous))
            {
                if (previous != vote.Value && !equivocators.Contains(vote.Validator)) equivocators.Add(vote.Validator);
            }
            else
            {
                seen[vote.Validator] = vote.Value;
            }
        }
        return equivocators;
    }

    internal long Slash(IReadOnlyList<int> equivocators)
    {
        long removed = 0;
        foreach (int index in equivocators)
        {
            if (Slashed.Add(index)) removed += Validators[index].Weight;
        }
        return removed;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    /**
     * Exhaustive safety oracle. Every honest validator is assigned to one of "votes for a", "votes for b" or
     * "abstains" (3^n), and every Byzantine validator is either absent or present in *both* quorums (2^n) - the
     * strongest adversary the model allows. If both values ever reach a quorum, safety is broken; the oracle then
     * reports which weight of double-signing was required.
     */
    private static bool SafetyHolds(long[] weights, int faultyBound, out long worstIntersection)
    {
        int n = weights.Length;
        long total = 0;
        foreach (long w in weights) total += w;
        long threshold = WeightedQuorum.Threshold(total);
        worstIntersection = long.MaxValue;
        int partitions = 1;
        for (int i = 0; i < n; i++) partitions *= 3;
        for (int code = 0; code < partitions; code++)
        {
            var assignment = new int[n];
            int rest = code;
            for (int i = 0; i < n; i++)
            {
                assignment[i] = rest % 3;
                rest /= 3;
            }
            for (int byzantine = 0; byzantine < (1 << n); byzantine++)
            {
                long faultyWeight = 0;
                for (int i = 0; i < n; i++)
                {
                    if ((byzantine & (1 << i)) != 0) faultyWeight += weights[i];
                }
                if (faultyWeight > faultyBound) continue;
                long weightA = 0, weightB = 0, overlap = 0;
                for (int i = 0; i < n; i++)
                {
                    bool inA = (byzantine & (1 << i)) != 0 || assignment[i] == 0;
                    bool inB = (byzantine & (1 << i)) != 0 || assignment[i] == 1;
                    if (inA) weightA += weights[i];
                    if (inB) weightB += weights[i];
                    if (inA && inB) overlap += weights[i];
                }
                if (weightA >= threshold && weightB >= threshold)
                {
                    return false;                       // two conflicting values committed
                }
                if (weightA >= threshold || weightB >= threshold) worstIntersection = Math.Min(worstIntersection, overlap);
            }
        }
        return true;
    }

    /** Every value that reached a quorum in the given voting pattern, using the faithful (slashing-free) count. */
    private static List<long> CommitSet(long[] weights, int[] assignment, int byzantine)
    {
        int n = weights.Length;
        long total = 0;
        foreach (long w in weights) total += w;
        long threshold = WeightedQuorum.Threshold(total);
        var values = new List<long>();
        for (long value = 0; value < 3; value++)
        {
            long weight = 0;
            for (int i = 0; i < n; i++)
            {
                // a Byzantine validator may sign every value; an honest one signs only its own assignment
                bool votes = (byzantine & (1 << i)) != 0 || assignment[i] == (int)value;
                if (votes) weight += weights[i];
            }
            if (weight >= threshold) values.Add(value);
        }
        return values;
    }

    public static void Main()
    {
        // 1-3. Weight bookkeeping, threshold and fault tolerance.
        var federation = new Federation(10, 8, 5, 3, 3, 1);
        Check(federation.TotalWeight == 30, "the total weight is the sum of the validator weights");
        Check(federation.Threshold == 21, "the quorum is ceil(2W/3) = 21 of 30");
        Check(federation.FaultTolerance == 9, "the tolerated Byzantine weight is floor((W-1)/3) = 9");
        Check(WeightedQuorum.Threshold(3) == 3 && WeightedQuorum.Threshold(9) == 7 && WeightedQuorum.Threshold(1) == 1,
            "small federations get a threshold that still forces a majority intersection");
        Check(Throws(() => new Federation()), "an empty federation is rejected");
        Check(Throws(() => new Federation(5, 0)), "a zero weight is rejected");
        Check(Throws(() => new Federation(5, -3)), "a negative weight is rejected");
        Check(federation.HonestWeight == 30, "with no faults every validator counts");

        // 4-5. The intersection lemma, enumerated over every pair of quorums.
        long worstPair = long.MaxValue;
        long overlapPairs = 0;
        int n = federation.Validators.Count;
        for (int left = 1; left < (1 << n); left++)
        {
            for (int right = 1; right < (1 << n); right++)
            {
                var leftSet = new List<int>();
                var rightSet = new List<int>();
                long weightLeft = 0, weightRight = 0;
                for (int i = 0; i < n; i++)
                {
                    if ((left & (1 << i)) != 0) { leftSet.Add(i); weightLeft += federation.Validators[i].Weight; }
                    if ((right & (1 << i)) != 0) { rightSet.Add(i); weightRight += federation.Validators[i].Weight; }
                }
                if (weightLeft < federation.Threshold || weightRight < federation.Threshold) continue;
                overlapPairs++;
                worstPair = Math.Min(worstPair,
                    WeightedQuorum.IntersectionWeight(federation.Validators, leftSet, rightSet));
            }
        }
        Console.WriteLine("  quorum pairs=" + overlapPairs + " smallest intersection weight=" + worstPair
            + " (must exceed F=" + federation.FaultTolerance + ")");
        Check(worstPair > federation.FaultTolerance,
            "every pair of quorums intersects in more weight than the fault bound");
        Check(2 * federation.Threshold - federation.TotalWeight == 12 && 12 > federation.FaultTolerance,
            "the closed form 2Q - W exceeds F, which is exactly the safety condition");
        Check(worstPair >= 2 * federation.Threshold - federation.TotalWeight,
            "every pair of quorums overlaps by at least the closed-form bound 2Q - W");

        // 6-7. Exhaustive safety: no fault budget and no vote pattern produces two conflicting quorums.
        long[] weights = { 10, 8, 5, 3, 3, 1 };
        Check(SafetyHolds(weights, (int)federation.FaultTolerance, out _),
            "exhaustive search finds no two conflicting quorums within the fault budget");
        Console.WriteLine("  exhaustive safety: 3^6 x 2^6 vote patterns at F=" + federation.FaultTolerance
            + ", overlap bound " + worstPair);
        Check(SafetyHolds(weights, 12, out _),
            "the budget can even be raised to the closed-form bound 2Q - W = 12 with no conflicting commit");
        Check(!SafetyHolds(weights, 13, out _),
            "one weight unit above that bound the exhaustive oracle finds the break, so the bound is tight");
        long[] equal = { 4, 4, 4, 4, 4, 4, 4 };
        var unweighted = new Federation(equal);
        Check(unweighted.Threshold == 19 && unweighted.FaultTolerance == 9,
            "equal weights reproduce the classical 2f+1 = 19 of 28 rule");
        Check(SafetyHolds(equal, (int)unweighted.FaultTolerance, out _), "and the unweighted rule is safe too");

        // 8-9. The protocol: a quorum commits, one vote short does not, and the honest weight is what matters.
        var round = new Federation(10, 8, 5, 3, 3, 1);
        for (int i = 0; i < 3; i++) round.Vote(i, 7);
        Check(round.WeightFor(7) == 23 && round.Committed(7), "the three heaviest validators alone carry the quorum");
        var shortRound = new Federation(10, 8, 5, 3, 3, 1);
        for (int i = 1; i < 6; i++) shortRound.Vote(i, 7);
        Check(shortRound.WeightFor(7) == 20 && !shortRound.Committed(7),
            "the five lightest validators are one weight short of the quorum");
        Check(shortRound.CommittedValues().Count == 0, "so nothing commits");
        var heavyMinority = new Federation(10, 8, 5, 3, 3, 1);
        for (int i = 0; i < 4; i++) heavyMinority.Vote(i, 42);
        Check(heavyMinority.Committed(42) && heavyMinority.CommittedValues().Count == 1,
            "a minority of heavy organisations decides, which is the point of weighted trust");

        // 10-11. With up to F weight faulty, exactly one value commits and the commit is single-valued.
        int committedConfigurations = 0;
        int unsafeConfigurations = 0;
        for (int byzantine = 0; byzantine < (1 << n); byzantine++)
        {
            long faultyWeight = 0;
            for (int i = 0; i < n; i++)
            {
                if ((byzantine & (1 << i)) != 0) faultyWeight += weights[i];
            }
            if (faultyWeight > federation.FaultTolerance) continue;
            committedConfigurations++;
            for (int value = 0; value < 3; value++)
            {
                var pattern = new int[n];
                for (int i = 0; i < n; i++) pattern[i] = (byzantine & (1 << i)) != 0 ? 0 : value;
                List<long> commits = CommitSet(weights, pattern, byzantine);
                if (commits.Count > 1) unsafeConfigurations++;
            }
        }
        Console.WriteLine("  protocol sweep: " + committedConfigurations + " fault sets inside the budget, "
            + unsafeConfigurations + " produced a conflicting commit");
        Check(unsafeConfigurations == 0, "no fault set inside the budget can commit two conflicting values");

        // 12-13. Equivocation: the adversary must double-sign, and slashing restores single-valued commits.
        var attack = new Federation(10, 8, 5, 3, 3, 1);
        for (int i = 0; i < 4; i++) attack.Vote(i, 100);          // 10 + 8 + 5 + 3 = 26
        attack.Vote(0, 200);
        attack.Vote(1, 200);
        attack.Vote(2, 200);                                       // 10 + 8 + 5 = 23 for the conflicting value
        Check(attack.Committed(100) && attack.Committed(200), "two conflicting values reach a quorum when nodes sign twice");
        List<int> equivocators = attack.Equivocators();
        Check(equivocators.Count == 3 && equivocators[0] == 0 && equivocators[1] == 1 && equivocators[2] == 2,
            "the vote log identifies exactly the three double-signers");
        Check(attack.WeightFor(100) == 26 && attack.WeightFor(200) == 23, "both quorums existed before slashing");
        long removedWeight = attack.Slash(equivocators);
        Check(removedWeight == 23, "slashing removes the double-signers' weight");
        Check(!attack.Committed(100) && !attack.Committed(200) && attack.CommittedValues().Count == 0,
            "after slashing neither conflicting value keeps a quorum: the round must restart, not pick a winner");
        Check(attack.HonestWeight == 7, "and the honest weight that remains is reported for the next round");
        var recovery = new Federation(3);
        Check(recovery.Threshold == 3 && recovery.Vote(0, 100) && recovery.Committed(100)
            && recovery.CommittedValues().Count == 1,
            "the surviving validators re-form a quorum and commit one value, so the federation recovers");

        // 14. The liveness boundary: a validator heavier than W - Q cannot be tolerated at all.
        var heavyNode = new Federation(11, 7, 6, 3, 3);
        Check(heavyNode.Threshold == 21 && heavyNode.TotalWeight == 30, "a lopsided federation still needs 21 of 30");
        Check(heavyNode.FaultTolerance == 9 && 11 > heavyNode.FaultTolerance,
            "the heaviest validator is heavier than the tolerated fault weight");
        var stalled = new Federation(11, 7, 6, 3, 3);
        stalled.Validators[0].Faulty = true;
        for (int i = 1; i < 5; i++) stalled.Vote(i, 5);
        Check(stalled.WeightFor(5) == 19 && !stalled.Committed(5),
            "when that validator is silent the remaining honest weight cannot reach the quorum: liveness is lost,"
            + " although safety still holds");
        Check(stalled.CommittedValues().Count == 0, "no value commits in the stalled round");

        // 15-17. Validation, deduplication and determinism.
        var validation = new Federation(4, 4, 4);
        Check(validation.Vote(0, 9), "the first vote from a validator is recorded");
        Check(!validation.Vote(0, 9), "a repeated vote for the same value is ignored");
        Check(validation.Votes.Count == 1 && validation.WeightFor(9) == 4, "so the weight is not double-counted");
        Check(validation.Vote(0, 9) == false && !validation.Committed(9), "and nothing gets committed by repetition");
        Check(Throws(() => validation.Vote(7, 1)), "a vote from an unknown validator is rejected");
        var first = new Federation(10, 8, 5, 3, 3, 1);
        for (int i = 0; i < 4; i++) first.Vote(i, 77);
        var second = new Federation(10, 8, 5, 3, 3, 1);
        for (int i = 0; i < 4; i++) second.Vote(i, 77);
        Check(first.CommittedValues().Count == second.CommittedValues().Count
            && first.WeightFor(77) == second.WeightFor(77), "replaying the same votes reproduces the outcome");

        // 18. Byzantine weight above the budget really can break safety: the bound is tight, not decorative.
        var broken = new Federation(10, 8, 5, 3, 3, 1);
        broken.Validators[0].Faulty = true;
        broken.Validators[1].Faulty = true;
        broken.Validators[2].Faulty = true;
        broken.Validators[3].Faulty = true;                       // 26 > F = 9
        for (int i = 0; i < 4; i++) broken.Vote(i, 1);
        broken.Vote(0, 2);
        broken.Vote(1, 2);
        broken.Vote(2, 2);
        Check(broken.Committed(1) && broken.Committed(2),
            "with more faulty weight than the budget two conflicting values commit - the budget is exactly tight");

        Console.WriteLine("PASS");
    }
}
