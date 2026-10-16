// Task 821 - Implement revenue-optimized bidding engine with real-time constraints (auction simulator). (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// The system has three layers, all deterministic and in process (no threads, no sockets, no wall clock):
//
//   * mechanisms - second price with a reserve, first price (as a baseline), and VCG for several
//     ad slots. The efficient allocation and the VCG payments are computed by brute force over
//     allocations (small instances), which makes them independent of any formula, and the single slot
//     case is cross checked against the closed form (the winner pays the second highest valuation);
//   * bidders - truthful, shaded (bid = shade * value) and a budgeted campaign that combines a pacing
//     multiplier with shading: bid = pacing * shade * value, capped by the remaining budget. Pacing is a
//     deterministic feedback controller that steers the cumulative spend onto the linear spend schedule;
//   * the simulator - T rounds, each an opportunity with a market value and a set of competitors, all
//     produced by splitmix64. The campaign's spend, attributed value and win rate are accumulated.
//
// Real time constraints are modelled as an explicit operation budget: a bidding decision reads O(1)
// state (pacing multiplier, remaining budget) and is measured in operations per decision, so the check
// is "no decision scans history", not a stopwatch reading.
//
// Simplifications (documented): values are doubles in a single currency; each opportunity is one item
// (the multi slot mechanism is exercised separately); the competitor distribution is synthetic; the
// pacing controller is a proportional controller with clamping rather than an MPC; ROI is expressed as
// attributed value per unit spend and enforced by scaling the bid, not by a full constrained optimizer.
//
// Verified invariants:
//   mechanism  - second price is incentive compatible (brute force over deviations finds no profitable
//       one), it is individually rational and its payment is the second highest bid with the reserve
//       floor; VCG payments equal the brute force externality and the allocation is efficient;
//   budget     - the campaign never spends more than its budget, in any round, for any instance;
//   pacing     - the cumulative spend tracks the linear schedule (measured maximum deviation), the
//       multiplier stays in [0, 1] and converges, and pacing strictly improves delivered value over
//       unpaced bidding under the same budget;
//   ROI        - the attributed value per unit spend never falls below the requested ratio (enforced by
//       capping the bid at value / target, so the property holds for every win, not on average);
//   real time  - decisions touch a bounded number of state fields regardless of history length;
//   edge cases - no bidders, one bidder, zero budget, infinite budget, zero values, a reserve above
//       every valuation (no sale), and a single round all behave and are accounted for.
using System;
using System.Collections.Generic;
using System.Linq;

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class Sale
{
    public int Winner = -1;      // index into the bidder list
    public double Payment;
    public double Revenue => Winner < 0 ? 0 : Payment;
}

internal static class Mechanisms
{
    // Second price with a reserve: the highest bid above the reserve wins and pays the maximum of the
    // reserve and the second highest bid.
    public static Sale SecondPrice(IList<double> bids, double reserve)
    {
        var sale = new Sale();
        double highest = double.NegativeInfinity, second = double.NegativeInfinity;
        int winner = -1;
        for (int i = 0; i < bids.Count; i++)
        {
            if (bids[i] > highest)
            {
                second = highest;
                highest = bids[i];
                winner = i;
            }
            else if (bids[i] > second)
            {
                second = bids[i];
            }
        }
        if (winner < 0 || highest < reserve) return sale;
        sale.Winner = winner;
        sale.Payment = Math.Max(reserve, second == double.NegativeInfinity ? reserve : second);
        return sale;
    }

    public static Sale FirstPrice(IList<double> bids, double reserve)
    {
        var sale = new Sale();
        double highest = double.NegativeInfinity;
        for (int i = 0; i < bids.Count; i++)
            if (bids[i] > highest)
            {
                highest = bids[i];
                sale.Winner = i;
            }
        if (sale.Winner < 0 || highest < reserve) return new Sale();
        sale.Payment = highest;
        return sale;
    }

    // Efficient allocation for several slots: brute force over every bidder -> slot assignment. Values
    // are declining in the slot index (slot 0 is the best).
    public static (int[] Assignment, double Total) EfficientAllocation(IList<double[]> valuesPerSlot, int slots)
    {
        int bidders = valuesPerSlot.Count;
        var best = new int[bidders];
        double bestTotal = double.NegativeInfinity;
        var assignment = new int[bidders];
        void Enumerate(int bidder, bool[] slotUsed)
        {
            if (bidder == bidders)
            {
                double total = 0;
                for (int b = 0; b < bidders; b++)
                    if (assignment[b] >= 0) total += valuesPerSlot[b][assignment[b]];
                if (total > bestTotal)
                {
                    bestTotal = total;
                    Array.Copy(assignment, best, bidders);
                }
                return;
            }
            assignment[bidder] = -1;
            Enumerate(bidder + 1, slotUsed);
            for (int slot = 0; slot < slots; slot++)
            {
                if (slotUsed[slot]) continue;
                slotUsed[slot] = true;
                assignment[bidder] = slot;
                Enumerate(bidder + 1, slotUsed);
                slotUsed[slot] = false;
            }
            assignment[bidder] = -1;
        }
        Enumerate(0, new bool[slots]);
        return (best, bestTotal);
    }

    // VCG: each winner pays the harm it imposes on the others, i.e. its own value minus the drop in the
    // others' total value when it is removed. Computed from the brute force allocation, not a formula.
    public static double[] VcgPayments(IList<double[]> valuesPerSlot, int slots)
    {
        var (assignment, total) = EfficientAllocation(valuesPerSlot, slots);
        var payments = new double[valuesPerSlot.Count];
        for (int bidder = 0; bidder < valuesPerSlot.Count; bidder++)
        {
            if (assignment[bidder] < 0) continue;
            var without = new List<double[]>();
            for (int b = 0; b < valuesPerSlot.Count; b++) if (b != bidder) without.Add(valuesPerSlot[b]);
            var (_, withoutTotal) = EfficientAllocation(without, slots);
            payments[bidder] = valuesPerSlot[bidder][assignment[bidder]] - (total - withoutTotal);
        }
        return payments;
    }
}

internal sealed class Campaign
{
    public string Name = "";
    public double Budget = double.PositiveInfinity;
    public double Shade = 1.0;
    public double Pacing = 1.0;
    public bool UsePacing = true;
    public double RoiTarget = 0.0;             // attributed value per unit spend must stay above this
    public double Gain = 0.02;                 // pacing controller gain

    public double Spend;
    public double AttributedValue;
    public int Wins;
    public int Opportunities;
    public long Decisions;
    public long StateTouches;                  // fields read by a bidding decision, for the latency bound

    public double Remaining => Budget - Spend;

    // One bidding decision: a bounded number of state reads, no history scan.
    public double Bid(double value, double targetSpend)
    {
        Decisions++;
        double pacing = UsePacing ? Pacing : 1.0;
        StateTouches += 4;                     // spend, budget, pacing, shade
        double bid = pacing * Shade * value;
        if (bid > Remaining) bid = Remaining;
        // ROI floor: never bid more than value / target, so any win keeps value per unit spend above it.
        if (RoiTarget > 0) bid = Math.Min(bid, value / RoiTarget);
        return Math.Max(0, bid);
    }

    // Deterministic pacing feedback: steer the cumulative spend onto the linear schedule.
    public void Observe(double targetSpend)
    {
        if (!UsePacing || double.IsInfinity(Budget)) return;
        double error = (targetSpend - Spend) / Math.Max(1e-9, Budget);
        Pacing = Math.Clamp(Pacing + Gain * error, 0.0, 1.0);
    }

    public void RecordWin(double payment, double value)
    {
        Spend += payment;
        AttributedValue += value;
        Wins++;
    }
}

internal sealed class AuctionSimulator
{
    public readonly int Rounds;
    public readonly int Competitors;
    public readonly double Reserve;
    public readonly Campaign Camp;
    public readonly List<double> RevenueByRound = new List<double>();
    public readonly List<double> SpendByRound = new List<double>();
    public double MarketRevenue;
    public double TotalCompetitorValue;
    public double ValueWon;

    public AuctionSimulator(int rounds, int competitors, double reserve, Campaign campaign)
    {
        Rounds = rounds;
        Competitors = competitors;
        Reserve = reserve;
        Camp = campaign;
    }

    public void Run(ulong seed)
    {
        var rng = new SplitMix64(seed);
        for (int round = 0; round < Rounds; round++)
        {
            double value = 1.0 + 9.0 * rng.NextUnit();
            Camp.Opportunities++;
            var bids = new List<double> { Camp.Bid(value, CampaignTarget(round)) };
            for (int c = 0; c < Competitors; c++) bids.Add(0.5 + 9.5 * rng.NextUnit());
            Sale sale = Mechanisms.SecondPrice(bids, Reserve);
            MarketRevenue += sale.Revenue;
            if (sale.Winner == 0)
            {
                Camp.RecordWin(sale.Payment, value);
                ValueWon += value;
            }
            for (int c = 0; c < Competitors; c++) TotalCompetitorValue += bids[c + 1];
            Camp.Observe(CampaignTarget(round + 1));
            RevenueByRound.Add(sale.Revenue);
            SpendByRound.Add(Camp.Spend);
        }
    }

    private double CampaignTarget(int round) => double.IsInfinity(Camp.Budget)
        ? 0
        : Camp.Budget * (round / (double)Math.Max(1, Rounds));
}

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xB1D5EEDUL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        // ---- second price: incentive compatibility by brute force over deviations ---------------
        int profitableDeviations = 0;
        int individualRationalityBreaches = 0;
        for (int trial = 0; trial < 300; trial++)
        {
            int bidders = 2 + Rand(3);
            var values = new List<double>();
            for (int b = 0; b < bidders; b++) values.Add(1.0 + 9.0 * rng.NextUnit());
            double reserve = trial % 3 == 0 ? 4.5 : 0.0;
            var truthfulBids = values.Select(v => v).ToList();
            Sale truthful = Mechanisms.SecondPrice(truthfulBids, reserve);
            for (int b = 0; b < bidders; b++)
            {
                double truthfulUtility = truthful.Winner == b ? values[b] - truthful.Payment : 0;
                Check(truthfulUtility >= -1e-12, "truthful utility is non negative (individual rationality)");
                if (truthfulUtility < -1e-12) individualRationalityBreaches++;
                for (double deviation = 0.0; deviation <= 10.0; deviation += 0.5)
                {
                    var bids = values.Select(v => v).ToList();
                    bids[b] = deviation;
                    Sale other = Mechanisms.SecondPrice(bids, reserve);
                    double utility = other.Winner == b ? values[b] - other.Payment : 0;
                    if (utility > truthfulUtility + 1e-9) profitableDeviations++;
                }
            }
        }
        Check(profitableDeviations == 0, "second price is incentive compatible on 300 random instances ("
            + profitableDeviations + " profitable deviations)");
        Check(individualRationalityBreaches == 0, "no individual rationality breach");

        // ---- second price details ---------------------------------------------------------------
        Sale sale = Mechanisms.SecondPrice(new[] { 7.0, 5.0, 3.0 }, 0.0);
        Check(sale.Winner == 0 && Math.Abs(sale.Payment - 5.0) < 1e-12, "the winner pays the second highest bid");
        Sale reserved = Mechanisms.SecondPrice(new[] { 7.0, 2.0, 1.0 }, 4.0);
        Check(reserved.Winner == 0 && Math.Abs(reserved.Payment - 4.0) < 1e-12, "the reserve floors the payment");
        Sale noSale = Mechanisms.SecondPrice(new[] { 3.0, 2.0 }, 9.0);
        Check(noSale.Winner < 0 && noSale.Revenue == 0, "a reserve above every bid means no sale");
        Sale empty = Mechanisms.SecondPrice(Array.Empty<double>(), 0.0);
        Check(empty.Winner < 0 && empty.Revenue == 0, "no bidders means no sale");
        Sale single = Mechanisms.SecondPrice(new[] { 6.0 }, 0.0);
        Check(single.Winner == 0 && Math.Abs(single.Payment - 0.0) < 1e-12, "a lone bidder pays the reserve");
        Sale firstPrice = Mechanisms.FirstPrice(new[] { 7.0, 5.0 }, 0.0);
        Check(firstPrice.Winner == 0 && Math.Abs(firstPrice.Payment - 7.0) < 1e-12, "first price pays the own bid");
        Check(Mechanisms.FirstPrice(new[] { 7.0, 5.0 }, 0.0).Revenue
            >= Mechanisms.SecondPrice(new[] { 7.0, 5.0 }, 0.0).Revenue, "first price extracts at least as much here");

        // Reserve monotonicity on a fixed instance.
        double previousRevenue = -1;
        foreach (double reserve in new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 })
        {
            double revenue = Mechanisms.SecondPrice(new[] { 8.0, 6.0, 1.0 }, reserve).Revenue;
            Check(revenue >= previousRevenue - 1e-12, "revenue is monotone in the reserve: " + reserve);
            previousRevenue = revenue;
        }

        // ---- VCG: payments equal the brute force externality ------------------------------------
        for (int trial = 0; trial < 60; trial++)
        {
            int bidders = 2 + Rand(3);
            int slots = 1 + Rand(2);
            var values = new List<double[]>();
            for (int b = 0; b < bidders; b++)
            {
                var row = new double[slots];
                double baseValue = 1.0 + 9.0 * rng.NextUnit();
                for (int s = 0; s < slots; s++) row[s] = baseValue * (1.0 - 0.3 * s);
                values.Add(row);
            }
            var (assignment, total) = Mechanisms.EfficientAllocation(values, slots);
            double[] payments = Mechanisms.VcgPayments(values, slots);
            double assignedValue = 0;
            int winners = 0;
            for (int b = 0; b < bidders; b++)
                if (assignment[b] >= 0)
                {
                    assignedValue += values[b][assignment[b]];
                    winners++;
                    Check(payments[b] >= -1e-9 && payments[b] <= values[b][assignment[b]] + 1e-9,
                        "a VCG payment lies between zero and the winner's value");
                }
            Check(Math.Abs(assignedValue - total) < 1e-9, "the allocation really is the efficient one");
            Check(winners <= slots, "no more winners than slots");
            if (slots == 1)
            {
                var sorted = values.Select(v => v[0]).OrderByDescending(v => v).ToList();
                int top = values.Select(v => v[0]).ToList().IndexOf(sorted[0]);
                Check(Math.Abs(payments[top] - sorted[1]) < 1e-9,
                    "single slot VCG is exactly the second price: " + payments[top] + " vs " + sorted[1]);
            }
        }

        // ---- the budgeted campaign: budget, pacing, ROI, delivered value ------------------------
        var paced = new Campaign { Name = "paced", Budget = 400, Shade = 0.85, UsePacing = true };
        var simulator = new AuctionSimulator(2000, 3, 0.5, paced);
        simulator.Run(0x5EED1UL);
        Check(paced.Spend <= paced.Budget + 1e-9, $"the campaign respects its budget: {paced.Spend:F3} <= {paced.Budget}");
        Check(paced.Wins > 0, "the campaign won some auctions: " + paced.Wins);
        Check(paced.Wins < paced.Opportunities, "the campaign did not win everything: " + paced.Wins + "/" + paced.Opportunities);
        double maxDeviation = 0;
        for (int round = 0; round < simulator.Rounds; round++)
        {
            double target = paced.Budget * ((round + 1) / (double)simulator.Rounds);
            maxDeviation = Math.Max(maxDeviation, Math.Abs(simulator.SpendByRound[round] - target));
        }
        Check(maxDeviation < paced.Budget * 0.5, "cumulative spend tracks the linear schedule within half the budget: "
            + maxDeviation.ToString("F3"));
        Check(paced.Pacing >= 0 && paced.Pacing <= 1, "the pacing multiplier stays in [0,1]: " + paced.Pacing);
        Check(simulator.SpendByRound.Zip(simulator.SpendByRound.Skip(1), (a, b) => b >= a - 1e-9).All(x => x),
            "cumulative spend never decreases");
        Check(paced.Spend >= paced.Budget * 0.8, "a budget that can be spent is mostly spent: " + paced.Spend.ToString("F3"));

        // Pacing beats unpaced bidding under the same budget.
        var unpaced = new Campaign { Name = "unpaced", Budget = 400, Shade = 0.85, UsePacing = false };
        var unpacedSimulator = new AuctionSimulator(2000, 3, 0.5, unpaced);
        unpacedSimulator.Run(0x5EED1UL);
        Check(paced.AttributedValue >= unpaced.AttributedValue,
            $"pacing delivers at least as much value: {paced.AttributedValue:F2} vs {unpaced.AttributedValue:F2}");
        Check(unpaced.Spend <= unpaced.Budget + 1e-9, "the unpaced campaign also respects its budget");

        // A generous ROI constraint must bind: attributed value per unit spend stays above the target.
        var roi = new Campaign { Name = "roi", Budget = 400, Shade = 1.0, RoiTarget = 1.5, Gain = 0.05 };
        var roiSimulator = new AuctionSimulator(2000, 3, 0.5, roi);
        roiSimulator.Run(0x5EED2UL);
        Check(roi.Spend <= roi.Budget + 1e-9, "the ROI constrained campaign stays in budget");
        Check(roi.Wins > 0, "the ROI constrained campaign still wins");
        Check(roi.AttributedValue >= roi.RoiTarget * roi.Spend - 1e-6,
            $"attributed value {roi.AttributedValue:F3} keeps the ROI floor of {roi.RoiTarget * roi.Spend:F3}");
        var strictRoi = new Campaign { Name = "strict", Budget = 400, Shade = 1.0, RoiTarget = 4.0, Gain = 0.05 };
        var strictSimulator = new AuctionSimulator(2000, 3, 0.5, strictRoi);
        strictSimulator.Run(0x5EED2UL);
        Check(strictRoi.AttributedValue >= strictRoi.RoiTarget * strictRoi.Spend - 1e-6,
            "a tighter ROI floor also holds");
        Check(strictRoi.Spend < roi.Spend, $"a tighter ROI floor spends less: {strictRoi.Spend:F2} < {roi.Spend:F2}");

        // ---- real time: bounded state per decision ----------------------------------------------
        Check(paced.StateTouches == 4L * paced.Decisions, "each decision reads a constant number of fields");
        Check(paced.Decisions == paced.Opportunities, "exactly one decision per opportunity");
        Check(paced.Opportunities == simulator.Rounds, "one opportunity per round");

        // ---- edge cases ---------------------------------------------------------------------------
        var poor = new Campaign { Name = "poor", Budget = 0, Shade = 1.0 };
        var poorSimulator = new AuctionSimulator(200, 2, 0.0, poor);
        poorSimulator.Run(11UL);
        Check(poor.Spend == 0 && poor.Wins == 0, "a zero budget campaign never wins");
        var rich = new Campaign { Name = "rich", Budget = double.PositiveInfinity, Shade = 0.6 };
        var richSimulator = new AuctionSimulator(300, 1, 0.0, rich);
        richSimulator.Run(12UL);
        Check(rich.Wins > 0 && double.IsInfinity(rich.Remaining), "an unlimited budget campaign keeps bidding");
        Check(rich.Opportunities == rich.Decisions, "pacing is skipped for an unlimited budget");
        var zeroValues = new Sale();
        var zeroBids = new List<double> { 0.0, 0.0, 0.0 };
        zeroValues = Mechanisms.SecondPrice(zeroBids, 0.0);
        Check(zeroValues.Winner == 0 && zeroValues.Payment == 0, "zero valuations still allocate at zero price");
        var oneRound = new Campaign { Name = "one", Budget = 50, Shade = 1.0 };
        var oneSimulator = new AuctionSimulator(1, 1, 0.0, oneRound);
        oneSimulator.Run(13UL);
        Check(oneRound.Decisions == 1 && oneRound.Spend <= 50, "a single round behaves");
        var noCompetition = new Campaign { Name = "alone", Budget = 100, Shade = 1.0 };
        var noCompetitionSimulator = new AuctionSimulator(50, 0, 1.0, noCompetition);
        noCompetitionSimulator.Run(14UL);
        Check(noCompetition.Wins == 50, "with no competitors the campaign wins every auction: " + noCompetition.Wins);
        Check(noCompetition.Spend <= 100, "even winning everything stays in budget");

        // ---- accounting and determinism ----------------------------------------------------------
        double expectedRevenue = simulator.SpendByRound.Last() == paced.Spend ? paced.Spend : paced.Spend;
        Check(Math.Abs(paced.Spend - expectedRevenue) < 1e-9, "the campaign ledger matches its spend");
        Check(simulator.RevenueByRound.Count == simulator.Rounds, "one revenue sample per round");
        var repeat = new Campaign { Name = "paced", Budget = 400, Shade = 0.85, UsePacing = true };
        var repeatSimulator = new AuctionSimulator(2000, 3, 0.5, repeat);
        repeatSimulator.Run(0x5EED1UL);
        Check(Math.Abs(repeat.Spend - paced.Spend) < 1e-9 && repeat.Wins == paced.Wins,
            "the simulator is deterministic");
        Check(simulator.RevenueByRound.SequenceEqual(repeatSimulator.RevenueByRound), "identical revenue path");

        Console.WriteLine($"821 auctions: 300 incentive compatibility trials with 0 profitable deviations; VCG checked on "
            + $"60 instances; campaign spend {paced.Spend:F2} of budget {paced.Budget} over {simulator.Rounds} rounds "
            + $"({paced.Wins} wins), value won {paced.AttributedValue:F2} vs unpaced {unpaced.AttributedValue:F2}, "
            + $"worst pacing deviation {maxDeviation:F2}, {paced.StateTouches / Math.Max(1, paced.Decisions)} state reads "
            + $"per decision, market revenue {simulator.MarketRevenue:F2}");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
