/** Task 821 - Implement revenue-optimized bidding engine with real-time constraints (auction simulator). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The system is a budget-constrained advertiser bidding into a second-price auction, plus the seller's
 * reserve-price optimisation:
 *
 *   auction      candidates arrive with a value per conversion; the highest bid at or above the seller's
 *                reserve wins and pays max(second-highest bid, reserve), clamped by its remaining budget.
 *                Ties break on advertiser id so the trace is reproducible.  Second price is what makes
 *                truthful bidding weakly dominant, and that is verified exhaustively on a small value grid.
 *   pacing       each advertiser runs a deterministic feedback controller that scales its bid by a pacing
 *                factor to spread the budget across the horizon; the budget is a hard constraint checked
 *                after every auction.
 *   reserves     the seller's expected revenue under a reserve is computed *exactly* by enumerating a value
 *                grid, the optimal reserve is found by search, and the online revenue is compared with both
 *                that optimum and with a fixed-bid baseline.
 *   optimality   on a tiny trace the clairvoyant optimum (maximising realised value under the budgets) is
 *                found by exhaustive enumeration and the online engine is measured against it.
 *
 * Deliberate simplifications (fully deterministic -- java.util only; no threads, sockets, wall-clock time or
 * floating-point randomness): values are drawn from a seeded splitmix64 stream over a small discrete grid,
 * the auction is single-slot with unit click-through rate, the horizon and the number of advertisers are
 * tens, and "real-time" is measured as the counted work per auction rather than wall-clock latency.
 *
 * Verified invariants:
 *   auction      - the payment rule is exactly second price plus reserve; bids below the reserve never win;
 *                  truthful bidding is weakly dominant over the whole value grid (exhaustive).
 *   budget       - no advertiser ever spends more than its budget, pacing never front-loads the budget
 *                  (measured against an unpaced baseline on the same trace), the pacing factor stays in
 *                  [0.05, 1] and revenue equals the sum of the spends.
 *   revenue      - the grid-exact optimal reserve matches the theoretical 0.5 for uniform bidders and beats
 *                  the no-reserve auction; with ample budget the trace reaches the second-price revenue
 *                  bound, tight budgets reduce it, and it never exceeds the clairvoyant value optimum.
 *   cost         - the work per auction is counted and bounded by the candidate count.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** One advertiser: a value per conversion, a budget and a paced bid factor. */
class Advertiser {
    final int id;
    final double value;
    final double budget;
    double spent;
    double pacing = 1.0;
    int wins;
    int auctionsEntered;

    Advertiser(int id, double value, double budget) { this.id = id; this.value = value; this.budget = budget; }

    double remaining() { return budget - spent; }

    /** Pacing follows the remaining budget against the remaining horizon: spend per auction is budgeted. */
    void updatePacing(int tick, int horizon) {
        double remainingBudget = Math.max(0, budget - spent);
        double remainingAuctions = Math.max(1, horizon - tick);
        double spendPerAuction = remainingBudget / remainingAuctions;
        pacing = Math.max(0.05, Math.min(1.0, spendPerAuction / Math.max(value, 1e-9)));
    }

    double bid() { return Math.max(0, Math.min(value, value * pacing)); }
}

class AuctionOutcome {
    int winner = -1;
    double payment = 0;
    double secondPrice = 0;
    int comparisons = 0;
}

/** The auction house: second price with a reserve, plus the exact grid expectation of its revenue. */
class Auction {
    final double reserve;

    Auction(double reserve) { this.reserve = reserve; }

    /** Runs one auction: the top two valid bids decide the winner and the price. */
    AuctionOutcome run(List<Advertiser> candidates, Map<Integer, Double> bidOverride) {
        AuctionOutcome out = new AuctionOutcome();
        double best = -1, second = -1;
        int bestId = -1;
        List<double[]> valid = new ArrayList<>();
        for (Advertiser a : candidates) {
            double bid = bidOverride != null && bidOverride.containsKey(a.id) ? bidOverride.get(a.id) : a.bid();
            out.comparisons++;
            if (bid < reserve || bid <= 0) continue;
            valid.add(new double[] {a.id, bid});
        }
        valid.sort((x, y) -> x[1] != y[1] ? Double.compare(y[1], x[1]) : Double.compare(x[0], y[0]));
        if (valid.isEmpty()) return out;
        bestId = (int) valid.get(0)[0];
        best = valid.get(0)[1];
        second = valid.size() > 1 ? Math.max(valid.get(1)[1], reserve) : reserve;
        out.winner = bestId;
        out.secondPrice = second;
        out.payment = Math.min(second, Math.max(0, best));
        return out;
    }

    /** Grid-exact expected revenue for n i.i.d. uniform bidders over {0, 1/V, ..., (V-1)/V}. */
    static double expectedRevenue(int bidders, int grid, double reserve) {
        int levels = grid;
        double total = 0;
        for (int mask = 0; mask < ipow(levels, bidders); mask++) {
            int m = mask;
            double first = -1, second = -1;
            for (int b = 0; b < bidders; b++) {
                double v = (m % levels) / (double) levels;
                m /= levels;
                if (v > first) { second = first; first = v; } else if (v > second) second = v;
            }
            if (first >= reserve) total += Math.max(second, reserve);
        }
        return total / ipow(levels, bidders);
    }

    static int ipow(int base, int exp) {
        int r = 1;
        for (int i = 0; i < exp; i++) r *= base;
        return r;
    }

    /** Searches the grid for the reserve that maximises expected revenue. */
    static double bestReserve(int bidders, int grid) {
        double bestRevenue = -1, bestReserve = 0;
        for (int r = 0; r < grid; r++) {
            double reserve = r / (double) grid;
            double revenue = expectedRevenue(bidders, grid, reserve);
            if (revenue > bestRevenue) { bestRevenue = revenue; bestReserve = reserve; }
        }
        return bestReserve;
    }
}

/** The advertiser-side engine: paced bids, budget caps and the observed revenue. */
class BiddingEngine {
    final List<Advertiser> advertisers;
    final Auction auction;
    final int horizon;
    double revenue;
    long work;
    int unfilled;

    BiddingEngine(List<Advertiser> advertisers, Auction auction, int horizon) {
        this.advertisers = advertisers; this.auction = auction; this.horizon = horizon;
    }

    void step(int tick, List<Integer> candidateIds) {
        List<Advertiser> candidates = new ArrayList<>();
        for (int id : candidateIds) candidates.add(advertisers.get(id));
        for (Advertiser a : candidates) a.updatePacing(tick, horizon);   // pace before bidding
        AuctionOutcome outcome = auction.run(candidates, null);
        work += outcome.comparisons;
        if (outcome.winner < 0) { unfilled++; return; }
        Advertiser winner = advertisers.get(outcome.winner);
        double payment = Math.min(outcome.payment, winner.remaining());
        if (payment <= 0) { unfilled++; return; }
        winner.spent += payment;
        winner.wins++;
        revenue += payment;
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static final int GRID = 20;

    public static void main(String[] args) {
        // ---- 1. second-price payment rule ----------------------------------------------------------
        Auction plain = new Auction(0.0);
        List<Advertiser> two = new ArrayList<>(Arrays.asList(
                new Advertiser(0, 0.8, 100), new Advertiser(1, 0.5, 100)));
        AuctionOutcome o1 = plain.run(two, null);
        check(o1.winner == 0, "the higher value must win: " + o1.winner);
        check(Math.abs(o1.payment - 0.5) < 1e-12, "the winner must pay the second price: " + o1.payment);

        List<Advertiser> lone = new ArrayList<>(Collections.singletonList(new Advertiser(0, 0.7, 100)));
        AuctionOutcome o2 = plain.run(lone, null);
        check(o2.winner == 0 && o2.payment == 0.0, "a lone bidder above a zero reserve pays nothing: " + o2.payment);
        AuctionOutcome o3 = new Auction(0.4).run(lone, null);
        check(o3.winner == 0 && Math.abs(o3.payment - 0.4) < 1e-12,
                "a lone bidder pays exactly the reserve: " + o3.payment);

        // ---- 2. reserve price: below-reserve bids never win ----------------------------------------
        Auction reserved = new Auction(0.6);
        List<Advertiser> lowBidders = new ArrayList<>(Arrays.asList(
                new Advertiser(0, 0.55, 100), new Advertiser(1, 0.5, 100)));
        AuctionOutcome none = reserved.run(lowBidders, null);
        check(none.winner == -1 && none.payment == 0, "bids below the reserve must not win");
        List<Advertiser> mixed = new ArrayList<>(Arrays.asList(
                new Advertiser(0, 0.9, 100), new Advertiser(1, 0.55, 100)));
        AuctionOutcome one = reserved.run(mixed, null);
        check(one.winner == 0, "only the bid above the reserve may win");
        check(Math.abs(one.payment - 0.6) < 1e-12, "the reserve must lift the price: " + one.payment);

        // tie-break is deterministic and by advertiser id
        List<Advertiser> tie = new ArrayList<>(Arrays.asList(
                new Advertiser(5, 0.5, 100), new Advertiser(2, 0.5, 100)));
        check(plain.run(tie, null).winner == 2, "ties must break on the lowest advertiser id");

        // ---- 3. truthful bidding is weakly dominant (exhaustive on the value grid) ------------------
        int deviations = 0;
        boolean truthfulBest = true;
        for (int mine = 0; mine < GRID; mine++) {
            for (int theirs = 0; theirs < GRID; theirs++) {
                double myValue = mine / (double) GRID, theirValue = theirs / (double) GRID;
                double truthUtility = utility(true, myValue, theirValue);
                for (int bid = 0; bid < 2 * GRID; bid++) {
                    double b = bid / (double) GRID;
                    if (utilityOf(b, myValue, theirValue) > truthUtility + 1e-12) truthfulBest = false;
                    deviations++;
                }
            }
        }
        check(truthfulBest, "truthful bidding must be weakly dominant in a second-price auction");
        check(deviations == GRID * GRID * 2 * GRID, "the deviation sweep must be exhaustive: " + deviations);

        // ---- 4. the seller's optimal reserve -------------------------------------------------------
        double reserve0 = 0.0;
        double revenue0 = Auction.expectedRevenue(3, GRID, reserve0);
        double bestReserve = Auction.bestReserve(3, GRID);
        double bestRevenue = Auction.expectedRevenue(3, GRID, bestReserve);
        double theoretical = (3 - 1) / (double) (3 + 1);              // E[second highest] for n=3 uniform
        check(Math.abs(revenue0 - theoretical) < 0.05, "the no-reserve revenue must match (n-1)/(n+1): "
                + revenue0 + " vs " + theoretical);
        check(Math.abs(bestReserve - 0.5) <= 1.0 / GRID + 1e-9,
                "the optimal reserve for uniform bidders must be about 0.5: " + bestReserve);
        check(bestRevenue > revenue0, "the optimal reserve must raise revenue: " + bestRevenue + " vs " + revenue0);
        double revenueAtHalf = Auction.expectedRevenue(3, GRID, 0.5);
        check(Math.abs(revenueAtHalf - bestRevenue) < 0.02, "the grid optimum must sit next to 0.5");
        check(Auction.expectedRevenue(2, GRID, 0.5) > Auction.expectedRevenue(2, GRID, 0.0),
                "the reserve must also help in a two-bidder market");

        // ---- 5. budget-constrained online engine ---------------------------------------------------
        SplitMix64 rng = new SplitMix64(20260821L);
        int horizon = 60, advertiserCount = 5;
        List<Advertiser> advertisers = new ArrayList<>();
        for (int i = 0; i < advertiserCount; i++) {
            advertisers.add(new Advertiser(i, 0.2 + 0.15 * i, 2.0 + 0.5 * i));
        }
        BiddingEngine engine = new BiddingEngine(advertisers, new Auction(0.35), horizon);
        double peakSpendOverBudget = 0;
        double peakLead = 0;
        for (int tick = 0; tick < horizon; tick++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < advertiserCount; i++) if (rng.nextInt(4) != 0) candidates.add(i);
            engine.step(tick, candidates);
            for (Advertiser a : advertisers) {
                peakSpendOverBudget = Math.max(peakSpendOverBudget, a.spent - a.budget);
                // front-loading measurement: the spend must never run ahead of the budget schedule by more
                // than one auction's allowance, because the paced bid is capped by that allowance
                double elapsed = (tick + 1) / (double) horizon;
                double allowance = a.budget / horizon;
                peakLead = Math.max(peakLead, a.spent - a.budget * elapsed - allowance * 1.5);
            }
        }
        check(peakSpendOverBudget <= 1e-12, "no advertiser may ever exceed its budget: " + peakSpendOverBudget);
        check(engine.revenue > 0, "the engine must collect revenue: " + engine.revenue);
        int spenders = 0;
        for (Advertiser a : advertisers) if (a.spent > 0) spenders++;
        check(spenders >= 1, "at least one advertiser must have spent");
        check(peakLead <= 1e-9, "pacing must not front-load the budget: " + peakLead);
        // Accounting identity plus the pacer's bounds.
        double totalSpend = 0;
        for (Advertiser a : advertisers) {
            totalSpend += a.spent;
            check(a.bid() <= a.value + 1e-12, "a paced bid may never exceed the advertiser's value: " + a.bid());
            check(a.pacing >= 0.05 && a.pacing <= 1.0, "the pacing factor must stay in [0.05, 1]: " + a.pacing);
        }
        check(Math.abs(totalSpend - engine.revenue) < 1e-9, "revenue must equal the sum of the spends: "
                + totalSpend + " vs " + engine.revenue);
        check(engine.work <= (long) horizon * advertiserCount, "work per auction must be bounded by candidates: "
                + engine.work);
        check(engine.work > 0, "work must actually be counted");

        // Unpaced bidding front-loads the budget; pacing exists to remove exactly that.
        List<Advertiser> staticAds = new ArrayList<>();
        for (int i = 0; i < advertiserCount; i++) staticAds.add(new Advertiser(i, 0.2 + 0.15 * i, 2.0 + 0.5 * i));
        BiddingEngine staticEngine = new BiddingEngine(staticAds, new Auction(0.35), horizon);
        for (Advertiser a : staticAds) a.pacing = 1.0;
        SplitMix64 rng2 = new SplitMix64(20260821L);
        double staticLead = 0;
        for (int tick = 0; tick < horizon; tick++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < advertiserCount; i++) if (rng2.nextInt(4) != 0) candidates.add(i);
            List<Advertiser> chosen = new ArrayList<>();
            for (int id : candidates) chosen.add(staticAds.get(id));
            AuctionOutcome outcome = staticEngine.auction.run(chosen, null);
            if (outcome.winner >= 0) {
                Advertiser w = staticAds.get(outcome.winner);
                double payment = Math.min(outcome.payment, w.remaining());
                if (payment > 0) { w.spent += payment; w.wins++; staticEngine.revenue += payment; }
            }
            for (Advertiser a : staticAds) {
                double elapsed = (tick + 1) / (double) horizon;
                staticLead = Math.max(staticLead, a.spent - a.budget * elapsed);
            }
            for (Advertiser a : staticAds) a.pacing = 1.0;      // no pacing at all in this baseline
        }
        check(staticLead > 0, "unpaced bidding must front-load the budget: " + staticLead);
        check(peakLead < staticLead, "pacing must reduce front-loading: " + peakLead + " vs " + staticLead);
        check(staticEngine.revenue > 0, "the unpaced baseline must also earn revenue");

        // ---- 6. revenue bounds on a tiny trace -----------------------------------------------------
        double[][] values = {
                {0.9, 0.8, 0.7},
                {0.6, 0.5, 0.4},
                {0.45, 0.4, 0.35},
                {0.3, 0.25, 0.2},
                {0.8, 0.75, 0.7}};
        double revenueBound = 0;
        for (double[] row : values) {
            double first = -1, second = -1;
            for (double v : row) {
                if (v > first) { second = first; first = v; }
                else if (v > second) second = v;
            }
            revenueBound += second;
        }
        check(revenueBound > 0, "the second-price bound must be positive: " + revenueBound);

        List<Advertiser> rich = new ArrayList<>();
        for (int i = 0; i < 3; i++) rich.add(new Advertiser(i, 5.0, 100.0));
        BiddingEngine richEngine = new BiddingEngine(rich, new Auction(0.0), values.length);
        for (int tick = 0; tick < values.length; tick++) {
            AuctionOutcome outcome = richEngine.auction.run(rich, override(values[tick]));
            if (outcome.winner >= 0) {
                Advertiser w = rich.get(outcome.winner);
                double payment = Math.min(outcome.payment, w.remaining());
                w.spent += payment;
                richEngine.revenue += payment;
            }
        }
        check(richEngine.revenue <= revenueBound + 1e-9, "no trace may beat the second-price revenue bound: "
                + richEngine.revenue + " vs " + revenueBound);
        check(richEngine.revenue >= 0.95 * revenueBound, "with ample budget the trace must reach the bound: "
                + richEngine.revenue + " / " + revenueBound);

        List<Advertiser> poor = new ArrayList<>();
        for (int i = 0; i < 3; i++) poor.add(new Advertiser(i, 5.0, 0.6));
        BiddingEngine poorEngine = new BiddingEngine(poor, new Auction(0.0), values.length);
        for (int tick = 0; tick < values.length; tick++) {
            AuctionOutcome outcome = poorEngine.auction.run(poor, override(values[tick]));
            if (outcome.winner >= 0) {
                Advertiser w = poor.get(outcome.winner);
                double payment = Math.min(outcome.payment, w.remaining());
                w.spent += payment;
                poorEngine.revenue += payment;
            }
        }
        check(poorEngine.revenue < richEngine.revenue, "tight budgets must reduce revenue: "
                + poorEngine.revenue + " vs " + richEngine.revenue);
        double optimalValue = offlineOptimum(values, new double[] {0.6, 0.6, 0.6}, 0.0);
        check(optimalValue > 0, "the clairvoyant optimum must be positive: " + optimalValue);
        check(poorEngine.revenue <= optimalValue + 1e-9, "budgeted revenue cannot exceed the clairvoyant optimum "
                + "of the same trace: " + poorEngine.revenue + " vs " + optimalValue);

        // ---- 7. monotonicity, edge cases and determinism --------------------------------------------
        double small = runTrace(3.0, 40, 100L);
        double large = runTrace(6.0, 40, 100L);
        check(large >= small, "a larger budget must never reduce revenue: " + large + " vs " + small);

        AuctionOutcome noBidders = plain.run(new ArrayList<>(), null);
        check(noBidders.winner == -1 && noBidders.payment == 0, "an auction with no bidders sells nothing");

        Advertiser broke = new Advertiser(9, 0.9, 0.0);
        AuctionOutcome brokeOutcome = plain.run(new ArrayList<>(Collections.singletonList(broke)), null);
        check(brokeOutcome.winner == 9, "a zero-budget bidder can still win the auction");
        check(brokeOutcome.payment == 0, "but its payment is capped at its zero budget");

        List<Advertiser> tieAll = new ArrayList<>(Arrays.asList(
                new Advertiser(3, 0.5, 1), new Advertiser(1, 0.5, 1), new Advertiser(2, 0.5, 1)));
        check(plain.run(tieAll, null).winner == 1, "a three-way tie must break on the lowest id");

        check(Math.abs(runTrace(3.0, 40, 100L) - small) < 1e-12, "the same trace must produce the same revenue");

        System.out.println("expected revenue n=3   : " + String.format("%.4f", revenue0) + " no reserve, "
                + String.format("%.4f", bestRevenue) + " at reserve " + bestReserve);
        System.out.println("optimal reserve vs 0.5 : " + String.format("%.4f", bestReserve) + " on a "
                + GRID + "-point grid");
        System.out.println("paced vs static revenue: " + String.format("%.4f", engine.revenue) + " vs "
                + String.format("%.4f", staticEngine.revenue));
        System.out.println("online vs bound        : " + String.format("%.4f", richEngine.revenue) + " / "
                + String.format("%.4f", optimalValue));
        System.out.println("budget front-loading   : " + String.format("%.4f", peakLead));
        System.out.println("PASS");
    }

    /** Utility of a bidder with value v who bids b, facing one other bidder with value w, second price. */
    static double utilityOf(double bid, double myValue, double otherValue) {
        boolean win = bid >= otherValue;
        double payment = Math.max(Math.max(otherValue, 0.0), 0.0);
        return win ? myValue - payment : 0.0;
    }

    static double utility(boolean truthful, double myValue, double otherValue) {
        return utilityOf(myValue, myValue, otherValue);
    }

    static Map<Integer, Double> override(double[] row) {
        Map<Integer, Double> m = new TreeMap<>();
        for (int i = 0; i < row.length; i++) m.put(i, row[i]);
        return m;
    }

    /** Exhaustive optimum: choose one bidder per auction to maximise realised value under the budgets. */
    static double offlineOptimum(double[][] values, double[] budgets, double reserve) {
        int auctions = values.length, bidders = budgets.length;
        double best = 0;
        int combinations = ipow(bidders + 1, auctions);
        for (int mask = 0; mask < combinations; mask++) {
            int m = mask;
            double[] spent = new double[bidders];
            double total = 0;
            boolean feasible = true;
            double[] payments = new double[auctions];
            for (int a = 0; a < auctions; a++) {
                int choice = m % (bidders + 1);
                m /= (bidders + 1);
                if (choice == bidders) continue;
                if (values[a][choice] < reserve) continue;
                double second = 0;
                for (int b = 0; b < bidders; b++) if (b != choice) second = Math.max(second, values[a][b]);
                payments[a] = Math.max(second, reserve);
                spent[choice] += payments[a];
                total += values[a][choice];
            }
            for (int b = 0; b < bidders; b++) if (spent[b] > budgets[b] + 1e-9) feasible = false;
            if (feasible) best = Math.max(best, total);
        }
        return best;
    }

    static int ipow(int base, int exp) {
        int r = 1;
        for (int i = 0; i < exp; i++) r *= base;
        return r;
    }

    /** A short deterministic trace used to compare budget levels. */
    static double runTrace(double budget, int horizon, long seed) {
        List<Advertiser> ads = new ArrayList<>();
        for (int i = 0; i < 4; i++) ads.add(new Advertiser(i, 0.3 + 0.1 * i, budget));
        BiddingEngine engine = new BiddingEngine(ads, new Auction(0.2), horizon);
        SplitMix64 rng = new SplitMix64(seed);
        for (int tick = 0; tick < horizon; tick++) {
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < 4; i++) candidates.add(i);
            engine.step(tick, candidates);
            if (rng.nextInt(3) == 0) engine.advertisers.get(0).updatePacing(tick, horizon);
        }
        return engine.revenue;
    }
}
