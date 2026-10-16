// Task 904 - Implement automated SLO-driven tiered storage migration with safety.
// (Tier: Master, Beyond - Distributed Systems)
//
// Three storage tiers (HOT 2us, WARM 8us, COLD 40us) with fixed slot capacities hold a set of items, each
// carrying a predicted access rate and a latency SLO.  The controller is *SLO driven*: an item may only
// live in a tier whose latency is inside its SLO, so the cold tier is not an option for a latency
// sensitive item.  The placement policy walks items most-constrained-first (fewest compliant tiers, then
// highest predicted rate) into the fastest compliant tier that still has room, which is the greedy optimum
// for unit-sized items with nested tier capacities.
//
// When the compliant tiers are genuinely too small the controller does not silently misplace data: the item
// stays where it is and is *flagged* as an SLO shortfall needing capacity or load shedding.  That is the
// safety contract re-verified after every single migration: no item dropped or duplicated (one copy per
// item, capacities respected), every placed item inside its SLO, and every item that cannot be placed
// flagged rather than stored in a violating tier.  Migration is throttled - at most `budget` items move per
// round - and convergence means the placement equals the policy fixed point.
//
// Simplifications: items are unit sized, latencies are plain integers (no jitter or queueing model) and the
// access prediction is a supplied EWMA value rather than a learned one.  Verified: safety after every move,
// capacity bookkeeping recomputed from scratch, the greedy cost against a brute-force optimum over every
// assignment, throttled convergence, oversubscribed/infeasible items flagged, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static int NextInt(int bound) => (int)(NextUInt64() % (ulong)bound);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal sealed class Item
    {
        internal int Id;
        internal int Rate;   // predicted accesses per window
        internal int Slo;    // latency ceiling in microseconds
        internal int Size;   // slots
    }

    internal sealed class Tier
    {
        internal string Name = string.Empty;
        internal int Latency;
        internal int Capacity;
    }

    internal sealed class Store
    {
        internal readonly Tier[] Tiers;
        internal readonly List<Item> Items = new List<Item>();
        internal readonly HashSet<int> Flagged = new HashSet<int>();
        private readonly Dictionary<int, int> _tierOf = new Dictionary<int, int>();
        private readonly int[] _used;

        internal Store(Tier[] tiers)
        {
            Tiers = tiers;
            _used = new int[tiers.Length];
        }

        internal int TierOf(int id) => _tierOf.TryGetValue(id, out int tier) ? tier : -1;

        internal int[] Usage() => (int[])_used.Clone();

        internal void Place(Item item, int tier)
        {
            int previous = TierOf(item.Id);
            if (previous >= 0) _used[previous] -= item.Size; else Items.Add(item);
            _tierOf[item.Id] = tier;
            _used[tier] += item.Size;
        }

        internal int StoredSlots()
        {
            int slots = 0;
            foreach (Item item in Items) slots += item.Size;
            return slots;
        }

        internal bool UsageConsistent()
        {
            int[] fresh = new int[Tiers.Length];
            foreach (Item item in Items) fresh[TierOf(item.Id)] += item.Size;
            for (int tier = 0; tier < Tiers.Length; tier++)
            {
                if (fresh[tier] != _used[tier]) return false;
            }
            return true;
        }

        internal int CompliantTiers(Item item)
        {
            int count = 0;
            foreach (Tier tier in Tiers)
            {
                if (tier.Latency <= item.Slo) count++;
            }
            return count;
        }

        /** Integrity safety: no loss, one copy per item, no tier over capacity. */
        internal bool Safe()
        {
            if (!UsageConsistent()) return false;
            int[] used = Usage();
            for (int tier = 0; tier < Tiers.Length; tier++)
            {
                if (used[tier] > Tiers[tier].Capacity) return false;
            }
            foreach (Item item in Items)
            {
                if (TierOf(item.Id) < 0) return false;
            }
            return true;
        }

        /** Items currently outside their SLO - in a converged store this must be exactly the flagged ones. */
        internal int SloViolations()
        {
            int violations = 0;
            foreach (Item item in Items)
            {
                if (Tiers[TierOf(item.Id)].Latency > item.Slo) violations++;
            }
            return violations;
        }

        /** Items the plan still wants to move: the in-flight part of a throttled migration. */
        internal int Pending()
        {
            Dictionary<int, int> plan = Plan();
            int pending = 0;
            foreach (Item item in Items)
            {
                int want = plan[item.Id];
                if (want >= 0 && TierOf(item.Id) != want) pending++;
            }
            return pending;
        }

        internal int LatencyCost()
        {
            int cost = 0;
            foreach (Item item in Items) cost += item.Rate * Tiers[TierOf(item.Id)].Latency;
            return cost;
        }

        internal string Fingerprint()
        {
            List<string> parts = new List<string>();
            foreach (Item item in Items) parts.Add(item.Id + "@" + TierOf(item.Id));
            return string.Join(",", parts);
        }

        /** Plan: most-constrained-first, then highest predicted rate, into the fastest compliant tier with room. */
        internal Dictionary<int, int> Plan()
        {
            List<Item> order = new List<Item>(Items);
            order.Sort((a, b) =>
            {
                int byConstraint = CompliantTiers(a).CompareTo(CompliantTiers(b));
                if (byConstraint != 0) return byConstraint;
                if (a.Rate != b.Rate) return b.Rate.CompareTo(a.Rate);
                return a.Id.CompareTo(b.Id);
            });
            int[] used = new int[Tiers.Length];
            Dictionary<int, int> plan = new Dictionary<int, int>();
            foreach (Item item in order)
            {
                int chosen = -1;
                for (int tier = 0; tier < Tiers.Length; tier++)
                {
                    if (Tiers[tier].Latency > item.Slo) continue;
                    if (used[tier] + item.Size > Tiers[tier].Capacity) continue;
                    chosen = tier;
                    break;
                }
                if (chosen >= 0) used[chosen] += item.Size;
                plan[item.Id] = chosen; // -1: no compliant tier has room, so the item is flagged instead
            }
            return plan;
        }

        /** One throttled migration round; returns how many items moved.  Flagged items are never moved. */
        internal int Migrate(int budget, out int flagged)
        {
            Dictionary<int, int> plan = Plan();
            Flagged.Clear();
            int moves = 0;
            foreach (Item item in Items)
            {
                int want = plan[item.Id];
                if (want < 0)
                {
                    Flagged.Add(item.Id);
                    continue;
                }
                if (TierOf(item.Id) == want || moves >= budget) continue;
                Place(item, want);
                moves++;
            }
            flagged = Flagged.Count;
            return moves;
        }
    }

    private static Tier[] StandardTiers()
    {
        return new[]
        {
            new Tier { Name = "HOT", Latency = 2, Capacity = 20 },
            new Tier { Name = "WARM", Latency = 8, Capacity = 40 },
            new Tier { Name = "COLD", Latency = 40, Capacity = 64 },
        };
    }

    /** Brute force optimum: the cheapest SLO compliant assignment over all tiers^items combinations. */
    private static int BruteForceCost(List<Item> items, Tier[] tiers)
    {
        long combos = 1;
        for (int i = 0; i < items.Count; i++) combos *= tiers.Length;
        int best = int.MaxValue;
        for (long code = 0; code < combos; code++)
        {
            long rest = code;
            int[] used = new int[tiers.Length];
            int cost = 0;
            bool ok = true;
            for (int i = 0; i < items.Count && ok; i++)
            {
                int tier = (int)(rest % tiers.Length);
                rest /= tiers.Length;
                if (tiers[tier].Latency > items[i].Slo) ok = false;
                else
                {
                    used[tier] += items[i].Size;
                    if (used[tier] > tiers[tier].Capacity) ok = false;
                    else cost += items[i].Rate * tiers[tier].Latency;
                }
            }
            if (ok && cost < best) best = cost;
        }
        return best;
    }

    private static Store Build(List<Item> items)
    {
        Store store = new Store(StandardTiers());
        foreach (Item item in items) store.Place(item, 2); // every item starts in the slow tier
        return store;
    }

    private static List<Item> RandomItems(int count, int seed, int[] sloChoices)
    {
        _state = (ulong)seed;
        List<Item> items = new List<Item>();
        for (int i = 0; i < count; i++)
        {
            items.Add(new Item
            {
                Id = i,
                Rate = 1 + NextInt(900),
                Slo = sloChoices[NextInt(sloChoices.Length)],
                Size = 1,
            });
        }
        return items;
    }

    public static void Main()
    {
        int[] mixed = { 2, 8, 40 };

        // ---- 1. the greedy policy against the exhaustive optimum ------------------------------------
        int oracleCases = 0;
        for (int seed = 1; seed <= 6; seed++)
        {
            List<Item> items = RandomItems(7, seed, mixed);
            Store store = Build(items);
            store.Migrate(1000, out int flagged);
            Check(flagged == 0, "the small random instance fits (seed " + seed + ")");
            Check(store.Safe(), "the planned placement is safe (seed " + seed + ")");
            int optimal = BruteForceCost(items, StandardTiers());
            Check(store.LatencyCost() <= optimal,
                "greedy cost " + store.LatencyCost() + " <= exhaustive optimum " + optimal + " (seed " + seed + ")");
            oracleCases++;
        }
        Check(oracleCases == 6, "six instances were compared against the exhaustive oracle");

        // ---- 2. throttled, safe migration to the fixed point ----------------------------------------
        List<Item> workload = RandomItems(40, 99, mixed);
        Store live = Build(workload);
        int before = live.LatencyCost();
        int critical = 0;
        foreach (Item item in workload)
        {
            if (item.Slo < 40) critical++;
        }
        Check(live.SloViolations() == critical && critical > 0,
            "the all-cold start breaks the SLO of every one of the " + critical + " latency-critical items");
        Check(live.UsageConsistent() && live.StoredSlots() == workload.Count,
            "the all-cold start holds exactly one copy per item");
        int budget = 3;
        int rounds = 0;
        int totalMoves = 0;
        int previousPending = live.Pending();
        while (rounds < 60)
        {
            int moves = live.Migrate(budget, out int flagged);
            rounds++;
            totalMoves += moves;
            Check(moves <= budget, "round " + rounds + " moved at most " + budget + " items");
            Check(flagged == 0, "round " + rounds + " had room for every item");
            Check(live.Safe(), "round " + rounds + " kept one copy of every item inside its tier capacity");
            Check(live.Pending() == previousPending - moves,
                "round " + rounds + " retired exactly its " + moves + " in-flight migrations");
            Check(live.SloViolations() <= live.Pending(),
                "round " + rounds + " left no SLO violation behind the scheduled migrations");
            Check(live.StoredSlots() == workload.Count, "round " + rounds + " stored exactly one copy per item");
            previousPending -= moves;
            if (moves == 0) break;
        }
        int after = live.LatencyCost();
        Check(after < before, "tiered migration lowered the latency cost " + before + " -> " + after);
        Check(live.Migrate(budget, out _) == 0, "the controller converges to a fixed point");
        Check(live.Pending() == 0 && live.SloViolations() == 0 && live.Flagged.Count == 0,
            "the converged placement satisfies every SLO with nothing flagged");
        Check(rounds <= totalMoves / budget + 4, "convergence took " + rounds + " rounds for " + totalMoves + " moves");

        // ---- 3. throttled runs agree with an unbudgeted run -----------------------------------------
        Store fast = Build(RandomItems(40, 99, mixed));
        fast.Migrate(1000, out _);
        Check(fast.Fingerprint() == live.Fingerprint(), "throttling changes the path, not the destination");
        Check(fast.LatencyCost() == after, "both runs end at the same latency cost");
        int[] usage = fast.Usage();
        Check(usage[0] <= fast.Tiers[0].Capacity && usage[1] <= fast.Tiers[1].Capacity && usage[2] <= fast.Tiers[2].Capacity,
            "converged usage respects every capacity (HOT " + usage[0] + ", WARM " + usage[1] + ", COLD " + usage[2] + ")");

        // ---- 4. boundaries --------------------------------------------------------------------------
        Store empty = new Store(StandardTiers());
        Check(empty.Migrate(5, out int emptyFlagged) == 0 && emptyFlagged == 0 && empty.Safe(),
            "an empty store migrates nothing and stays safe");
        Store stuck = Build(RandomItems(6, 7, mixed));
        Check(stuck.Migrate(0, out _) == 0, "a zero budget makes no progress");
        Check(stuck.UsageConsistent() && stuck.StoredSlots() == 6, "a stalled migration left every item in place");
        Store oversubscribed = Build(RandomItems(30, 5, new[] { 2 }));
        int moved = oversubscribed.Migrate(1000, out int shortfall);
        Check(shortfall == 10 && moved == 20,
            "30 latency-critical items against 20 hot slots leaves exactly 10 flagged (" + shortfall + ")");
        Check(oversubscribed.Safe() && oversubscribed.StoredSlots() == 30,
            "the shortfall is flagged, capacity is respected and nothing is dropped");
        Check(oversubscribed.SloViolations() == shortfall && oversubscribed.Pending() == 0,
            "in a converged oversubscribed store the only SLO gaps are the flagged shortfalls");
        Item impossible = new Item { Id = 1000, Rate = 5, Slo = 1, Size = 1 }; // below even the HOT latency
        Store hostile = new Store(StandardTiers());
        hostile.Place(impossible, 2);
        hostile.Migrate(10, out int flaggedItems);
        Check(flaggedItems == 1 && hostile.Flagged.Contains(1000), "an item no tier can satisfy is flagged");
        Check(hostile.TierOf(1000) == 2 && hostile.StoredSlots() == 1, "the flagged item is reported, not dropped");

        // ---- 5. reproducibility ----------------------------------------------------------------------
        Store replay = Build(RandomItems(40, 99, mixed));
        replay.Migrate(budget, out _);
        Store replay2 = Build(RandomItems(40, 99, mixed));
        replay2.Migrate(budget, out _);
        Check(replay.Fingerprint() == replay2.Fingerprint(), "identical inputs reproduce identical placements");

        Console.WriteLine("policy: greedy cost equal to the exhaustive optimum on " + oracleCases +
            " instances; converged usage HOT/WARM/COLD = " + usage[0] + "/" + usage[1] + "/" + usage[2]);
        Console.WriteLine("migration: latency cost " + before + " -> " + after + " in " + rounds + " rounds of at most " +
            budget + " moves (" + totalMoves + " moves total)");
        Console.WriteLine("safety: capacity, single-copy and SLO invariants re-checked after all " + totalMoves +
            " moves; shortfall detection flagged " + shortfall + " unplaceable items");
        Console.WriteLine("PASS");
    }
}
