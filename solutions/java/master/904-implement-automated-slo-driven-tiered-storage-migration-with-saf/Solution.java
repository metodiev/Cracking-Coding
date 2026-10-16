/** Task 904 - Implement automated SLO-driven tiered storage migration with safety.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A storage fleet has three tiers (hot SSD, warm disk, cold object store) with different latency and price.
 * Each block carries a size, an access heat and an SLO (the latency it must stay under). The planner picks,
 * for every block, the *cheapest* tier that still satisfies its SLO, then migrates the mis-placed blocks in
 * throttled, atomic batches.
 *
 * Safety is the interesting part and is what the checks exercise:
 *
 *   placement rule   the chosen tier is the minimum-price tier with latency <= SLO (verified against a
 *                    brute-force oracle that enumerates every tier);
 *   dual write       a migration copies first and only then switches the source of truth, so a block is
 *                    readable - and its checksum verifiable - at every intermediate step; the checks
 *                    re-derive the copy set and the source of truth after every single block move;
 *   throttle         at most `budgetBytes` are copied per round and each round makes progress, so the plan
 *                    needs at least ceil(bytes/need/budget) rounds;
 *   idempotence      an already-correct placement, an empty corpus, or a block larger than the budget
 *                    produces an empty plan (never a half-migrated block).
 *
 * Simplification: the latencies and prices are small integers rather than measured p99s, and the "fleet" is a
 * few maps in memory. Invariants verified: zero SLO violations after the run (many before it), strictly
 * lower monthly cost, checksum preservation, SLO-safe destinations, bounded per-round bytes and a finite
 * number of rounds.
 */
import java.util.*;

/** A storage tier: name, read latency in microseconds, monthly price per GiB in tenths of a cent. */
final class Tier {
    final String name;
    final int latencyUs;
    final int pricePerGiB;

    Tier(String name, int latencyUs, int pricePerGiB) {
        this.name = name;
        this.latencyUs = latencyUs;
        this.pricePerGiB = pricePerGiB;
    }
}

/** One stored block: identity, size, access heat, SLO, and the checksum the client committed to. */
final class Block {
    final String id;
    final int sizeGiB;
    final long accesses;
    final int sloUs;
    final long checksum;

    Block(String id, int sizeGiB, long accesses, int sloUs, long checksum) {
        this.id = id;
        this.sizeGiB = sizeGiB;
        this.accesses = accesses;
        this.sloUs = sloUs;
        this.checksum = checksum;
    }
}

/** One throttled batch: the blocks whose source of truth is switched at the end of the round. */
final class MigrationBatch {
    final int round;
    final List<String> blockIds = new ArrayList<>();
    long bytesCopied;

    MigrationBatch(int round) { this.round = round; }

    void add(Block block) {
        blockIds.add(block.id);
        bytesCopied += block.sizeGiB;
    }
}

public class Solution {
    static final Tier HOT = new Tier("hot", 200, 120);
    static final Tier WARM = new Tier("warm", 2000, 30);
    static final Tier COLD = new Tier("cold", 40000, 4);
    static final List<Tier> TIERS = List.of(HOT, WARM, COLD);

    public static void main(String[] args) {
        List<Block> blocks = new ArrayList<>();
        SplitMix64 rng = new SplitMix64(0x904ABCDEL);
        int[] slos = {HOT.latencyUs, WARM.latencyUs, COLD.latencyUs};
        for (int i = 0; i < 18; i++) {
            int slo = slos[i % 3];
            int size = 4 + (int) Math.floorMod(rng.nextLong(), 60);
            long accesses = 10 + Math.floorMod(rng.nextLong(), 500);
            blocks.add(new Block("blk-" + i, size, accesses, slo, checksumOf("blk-" + i, size)));
        }

        // ---- 1. the placement rule, checked against a brute-force oracle -----------------------------
        Map<String, Tier> plan = new LinkedHashMap<>();
        for (Block block : blocks) plan.put(block.id, cheapestTierFor(block.sloUs));
        check(plan.get("blk-0") == HOT, "a 200us SLO can only be met by the hot tier");
        check(plan.get("blk-1") == WARM, "a 2000us SLO is met by warm at the lowest price");
        check(plan.get("blk-2") == COLD, "a 40000us SLO belongs in cold storage");
        for (Block block : blocks) {
            check(bruteForceTier(block.sloUs) == plan.get(block.id), "oracle disagrees for " + block.id);
            check(plan.get(block.id).latencyUs <= block.sloUs, block.id + " would violate its own SLO");
        }

        // ---- 2. the starting point: everything cold, so most blocks violate their SLO -----------------
        Map<String, Tier> placement = new LinkedHashMap<>();
        for (Block block : blocks) placement.put(block.id, COLD);
        long violationsBefore = countViolations(blocks, placement);
        long costBefore = monthlyCost(blocks, placement);
        int misplaced = 0;
        long bytesToMove = 0;
        for (Block block : blocks) {
            if (plan.get(block.id) != COLD) {
                misplaced++;
                bytesToMove += block.sizeGiB;
            }
        }
        check(violationsBefore == misplaced, "violations must equal the number of mis-placed blocks: "
                + violationsBefore + " vs " + misplaced);
        check(violationsBefore > 0 && violationsBefore < blocks.size(), "the fixture must be partly mis-placed");
        Map<String, Tier> allHot = new LinkedHashMap<>();
        for (Block block : blocks) allHot.put(block.id, HOT);
        long naiveCost = monthlyCost(blocks, allHot);
        check(countViolations(blocks, allHot) == 0, "parking everything in hot is SLO-safe but wasteful");
        check(naiveCost > costBefore, "the naive SLO-safe placement must be the expensive option");

        // ---- 3. throttled migration with dual writes and per-step safety -----------------------------
        long budgetBytes = 90;
        List<MigrationBatch> batches = planMigration(blocks, placement, plan, budgetBytes);
        long expectedRounds = (bytesToMove + budgetBytes - 1) / budgetBytes;
        check(batches.size() >= expectedRounds,
                "the throttle forces at least " + expectedRounds + " rounds, got " + batches.size());
        Map<String, List<Tier>> copies = new LinkedHashMap<>();
        Map<String, Tier> source = new LinkedHashMap<>();
        for (Block block : blocks) {
            copies.put(block.id, new ArrayList<>(List.of(COLD)));
            source.put(block.id, COLD);
        }
        int moved = 0;
        for (MigrationBatch batch : batches) {
            for (String id : batch.blockIds) {
                Block block = blockById(blocks, id);
                Tier destination = plan.get(id);
                check(!copies.get(id).contains(destination), id + " must not be copied twice");
                copies.get(id).add(destination);                       // copy completes, dual write begins
                check(copies.get(id).size() == 2, id + " must be dual-written before the switch");
                check(copies.get(id).contains(source.get(id)), id + " must stay readable from its old tier");
                check(checksumOf(id, block.sizeGiB) == block.checksum, "copy of " + id + " is corrupt");
                check(destination.latencyUs <= block.sloUs, "destination of " + id + " breaks its SLO");
                copies.get(id).remove(source.get(id));                 // switch the source of truth
                source.put(id, destination);
                moved++;
                check(copies.get(id).equals(List.of(destination)),
                        id + " must hold exactly one verified copy after the switch");
                check(!copies.get(id).isEmpty(), id + " became unreadable");
                check(countViolations(blocks, source) == violationsBefore - moved,
                        "each switched block must remove exactly one violation, " + (violationsBefore - moved)
                                + " expected");
            }
            check(batch.bytesCopied <= budgetBytes, "round " + batch.round + " copied " + batch.bytesCopied
                    + " GiB, above the budget of " + budgetBytes);
            check(!batch.blockIds.isEmpty(), "an empty round would be a livelock");
        }
        check(moved == misplaced, "every mis-placed block must move exactly once, moved " + moved);
        long bytesCopied = 0;
        for (MigrationBatch batch : batches) bytesCopied += batch.bytesCopied;
        check(bytesCopied == bytesToMove, "copied bytes must equal the mis-placed bytes: " + bytesCopied);

        // ---- 4. the invariants the migration was supposed to establish --------------------------------
        check(countViolations(blocks, source) == 0, "no block may violate its SLO after the migration");
        check(source.equals(plan), "the live placement must equal the planned tier for every block");
        long costAfter = monthlyCost(blocks, source);
        check(costAfter < naiveCost, "SLO compliance must cost less than parking everything in hot: "
                + costAfter + " vs " + naiveCost);
        check(costAfter > costBefore, "the cheap-but-violating start cannot be kept as is");
        for (Block block : blocks) {
            check(checksumOf(block.id, block.sizeGiB) == block.checksum, "payload of " + block.id + " changed");
        }

        // ---- 5. independent oracle: re-derive the plan from scratch and compare -----------------------
        long oracleBytes = 0;
        Set<String> oracleIds = new LinkedHashSet<>();
        Map<String, Tier> oracleTarget = new LinkedHashMap<>();
        for (Block block : blocks) {
            Tier best = bruteForceTier(block.sloUs);
            oracleTarget.put(block.id, best);
            if (best != COLD) {
                check(oracleIds.add(block.id), "the oracle must not schedule " + block.id + " twice");
                oracleBytes += block.sizeGiB;
            }
        }
        check(oracleIds.size() == misplaced, "the oracle moves the same number of blocks");
        check(oracleBytes == bytesCopied, "the oracle moves the same number of bytes");
        check(oracleTarget.equals(source), "the oracle ends in the same placement");
        int oracleRounds = 0;
        int cursor = 0;
        List<Block> pending = new ArrayList<>();
        for (Block block : blocks) if (oracleTarget.get(block.id) != COLD) pending.add(block);
        while (cursor < pending.size()) {
            long fill = 0;
            while (cursor < pending.size() && fill + pending.get(cursor).sizeGiB <= budgetBytes) {
                fill += pending.get(cursor).sizeGiB;
                cursor++;
            }
            oracleRounds++;
        }
        check(oracleRounds == batches.size(), "the oracle needs " + oracleRounds + " rounds, planner used "
                + batches.size());

        // ---- 6. idempotence and boundary cases --------------------------------------------------------
        check(planMigration(blocks, source, plan, budgetBytes).isEmpty(),
                "a correctly placed fleet must produce an empty plan");
        check(planMigration(List.of(), new LinkedHashMap<>(), new LinkedHashMap<>(), budgetBytes).isEmpty(),
                "an empty corpus must produce an empty plan");
        check(monthlyCost(blocks, new LinkedHashMap<>()) == 0, "an empty placement costs nothing");
        check(countViolations(blocks, new LinkedHashMap<>()) == blocks.size(),
                "a missing placement means every block is unreadable");
        Block oversized = new Block("blk-big", 500, 1, HOT.latencyUs, checksumOf("blk-big", 500));
        check(planMigration(List.of(oversized), Map.of("blk-big", COLD), Map.of("blk-big", HOT), 4).isEmpty(),
                "a block bigger than the whole budget is never half-migrated");

        // ---- 7. the policy must refuse SLO-violating destinations and tolerate misses -----------------
        Block lowSlo = blockById(blocks, "blk-0");
        Block highSlo = blockById(blocks, "blk-2");
        check(rejectPlacement(lowSlo, COLD) && !rejectPlacement(lowSlo, HOT),
                "the policy must refuse to demote a 200us block to cold");
        check(!rejectPlacement(highSlo, HOT) && rejectPlacement(highSlo, new Tier("glacial", 50000, 1)),
                "the policy must refuse a tier that is simply too slow");
        check(rejectsMissing(cheapestTierFor(50)) && rejectsMissing(bruteForceTier(50)),
                "an SLO no tier can meet must yield no placement rather than a violating one");
        check(rejectPlacement(new Block("blk-impossible", 1, 1, 50, 0), HOT),
                "an impossible SLO is rejected by the policy on every tier");

        // ---- 8. determinism ----------------------------------------------------------------------------
        List<MigrationBatch> replay = planMigration(blocks, placement, plan, budgetBytes);
        check(replay.size() == batches.size(), "the plan length must be reproducible");
        for (int i = 0; i < batches.size(); i++) {
            check(replay.get(i).blockIds.equals(batches.get(i).blockIds), "batch " + i + " must be identical");
            check(replay.get(i).bytesCopied == batches.get(i).bytesCopied, "batch " + i + " bytes must match");
        }

        System.out.println(blocks.size() + " blocks, " + totalBytesOf(blocks) + " GiB, budget " + budgetBytes
                + " GiB/round, plan of " + batches.size() + " rounds (" + expectedRounds + " minimum)");
        System.out.println("SLO violations " + violationsBefore + " -> 0; monthly cost " + costBefore
                + " (violating) / " + costAfter + " (planner) / " + naiveCost
                + " (all-hot), in tenths of a cent");
        System.out.println("blocks moved " + moved + " / " + blocks.size() + ", bytes copied " + bytesCopied
                + ", tail round " + batches.get(batches.size() - 1).bytesCopied + " GiB");
        System.out.println("PASS");
    }

    /** Cheapest tier whose latency still satisfies the SLO; TIERS is ordered hottest to coldest. */
    static Tier cheapestTierFor(int sloUs) {
        Tier best = null;
        for (Tier tier : TIERS) {
            if (tier.latencyUs > sloUs) continue;
            if (best == null || tier.pricePerGiB < best.pricePerGiB) best = tier;
        }
        return best;
    }

    /** Independent oracle: enumerate every affordable tier and keep one no other tier beats on price. */
    static Tier bruteForceTier(int sloUs) {
        List<Tier> affordable = new ArrayList<>();
        for (Tier tier : TIERS) if (sloUs >= tier.latencyUs) affordable.add(tier);
        for (Tier candidate : affordable) {
            boolean beaten = false;
            for (Tier other : affordable) {
                if (other != candidate && other.pricePerGiB < candidate.pricePerGiB) beaten = true;
            }
            if (!beaten) return candidate;
        }
        return null;
    }

    /** Greedy throttled planner: a block is never split across rounds, so every batch stays atomic. */
    static List<MigrationBatch> planMigration(List<Block> blocks, Map<String, Tier> current,
            Map<String, Tier> target, long budgetBytes) {
        List<Block> pending = new ArrayList<>();
        for (Block block : blocks) {
            Tier currentTier = current.get(block.id);
            if (currentTier != target.get(block.id) && block.sizeGiB <= budgetBytes) pending.add(block);
        }
        List<MigrationBatch> batches = new ArrayList<>();
        int index = 0;
        while (index < pending.size()) {
            MigrationBatch batch = new MigrationBatch(batches.size() + 1);
            while (index < pending.size() && batch.bytesCopied + pending.get(index).sizeGiB <= budgetBytes) {
                batch.add(pending.get(index));
                index++;
            }
            check(batch.bytesCopied > 0, "a round that copies nothing would be a livelock");
            batches.add(batch);
        }
        return batches;
    }

    static long countViolations(List<Block> blocks, Map<String, Tier> placement) {
        long violations = 0;
        for (Block block : blocks) {
            Tier tier = placement.get(block.id);
            if (tier == null || tier.latencyUs > block.sloUs) violations++;
        }
        return violations;
    }

    static long monthlyCost(List<Block> blocks, Map<String, Tier> placement) {
        long cost = 0;
        for (Block block : blocks) {
            Tier tier = placement.get(block.id);
            if (tier != null) cost += (long) block.sizeGiB * tier.pricePerGiB;
        }
        return cost;
    }

    static long totalBytesOf(List<Block> blocks) {
        long total = 0;
        for (Block block : blocks) total += block.sizeGiB;
        return total;
    }

    static Block blockById(List<Block> blocks, String id) {
        for (Block block : blocks) if (block.id.equals(id)) return block;
        return null;
    }

    /** The policy refuses any destination tier that would break the block's SLO. */
    static boolean rejectPlacement(Block block, Tier destination) {
        return destination.latencyUs > block.sloUs;
    }

    static boolean rejectsMissing(Tier placement) { return placement == null; }

    static long checksumOf(String id, int sizeGiB) {
        long h = 0xC0FFEEL * (sizeGiB + 1);
        for (int i = 0; i < id.length(); i++) h = (h ^ id.charAt(i)) * 0xBF58476D1CE4E5B9L;
        return h ^ (h >>> 31);
    }

    static final class SplitMix64 {
        private long state;

        SplitMix64(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
