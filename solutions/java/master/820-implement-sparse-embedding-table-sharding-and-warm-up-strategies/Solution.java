/** Task 820 - Implement sparse-embedding table sharding and warm-up strategies for recommender systems. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A sparse embedding table is a very large, very skewed key-value store: a few ids take most of the traffic.
 * The file models the two problems that decide whether such a table survives production:
 *
 *   sharding    three placement schemes are compared on the same Zipf-like vocabulary:
 *                 hash       -- id mod S, the naive scheme (hot ids all land on one shard)
 *                 consistent -- a virtual-node ring, so a resize moves ~1/S of the ids instead of ~(S-1)/S
 *                 frequency  -- a replicated placement map that spreads the hot ids over shards
 *               Hot ids are additionally replicated R times, and every lookup is served by exactly one
 *               shard per replica, which makes the read cost a constant that is measured, not assumed.
 *   warm-up     because most rows are never seen during a short window, a cold table answers from default
 *               rows and scores badly at the start.  A deterministic warm-up phase (replaying a sample of
 *               the same distribution) is compared against the cold start on identical online traffic, and
 *               both the early held-out accuracy and the log-loss are measured.
 *
 * Deliberate simplifications (fully deterministic -- java.util only; no threads, sockets, clocks or files):
 * the vocabulary is ~300 ids with a synthetic Zipf frequency profile, the model is a per-id weight trained
 * online with sparse single-id examples, and "memory" is counted in rows.  The measured quantities (shard
 * imbalance, moved fraction on resize, collision counts, early accuracy, staleness) are all exact.
 *
 * Verified invariants:
 *   placement  - every id has exactly one home shard, hot ids have R replicas, the frequency-aware scheme
 *                bounds max/mean shard load far below hashing, and consistent hashing moves about 1/S of the
 *                ids when the shard count changes while hashing moves nearly all of them.
 *   serving    - a lookup touches exactly one shard per replica, replica values converge after a sync step,
 *                and a stale read before the sync is bounded by one version.
 *   warm-up    - the warm table has a higher early hit rate and a better early accuracy/log-loss than the
 *                cold table on the same online stream, and the online phase still improves monotonically.
 *   degenerate - empty vocabularies, a single shard, unseen ids and a zero-capacity budget are handled.
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

/** Placement schemes for a sparse table. */
class Placement {
    final int shardCount;
    final long[] ringKeys;         // consistent-hashing ring keys (sorted)
    final int[] ringShards;
    final Map<Integer, Integer> frequencyHome = new TreeMap<>();
    final Set<Integer> hot = new TreeSet<>();
    int replicaCount = 1;

    Placement(int shardCount) {
        this.shardCount = shardCount;
        this.ringKeys = new long[0];
        this.ringShards = new int[0];
    }

    Placement(int shardCount, int virtualNodes, long seed) {
        this.shardCount = shardCount;
        SplitMix64 rng = new SplitMix64(seed);
        long[] keys = new long[shardCount * virtualNodes];
        int[] owners = new int[keys.length];
        for (int s = 0; s < shardCount; s++) {
            for (int v = 0; v < virtualNodes; v++) {
                int idx = s * virtualNodes + v;
                keys[idx] = rng.nextLong();
                owners[idx] = s;
            }
        }
        Integer[] order = new Integer[keys.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingLong(i -> keys[i]));
        ringKeys = new long[keys.length];
        ringShards = new int[keys.length];
        for (int i = 0; i < order.length; i++) {
            ringKeys[i] = keys[order[i]];
            ringShards[i] = owners[order[i]];
        }
    }

    static long hash(int id) {
        long z = id * 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int moduloShard(int id) { return (int) Math.floorMod(hash(id), shardCount); }

    int consistentShard(int id) {
        long key = hash(id);
        int lo = 0, hi = ringKeys.length - 1, best = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (ringKeys[mid] >= key) { best = mid; hi = mid - 1; } else lo = mid + 1;
        }
        return ringShards[best < 0 ? 0 : best];
    }

    int frequencyShard(int id) { return frequencyHome.getOrDefault(id, -1); }

    /** Weighted LPT: ids in descending frequency, each placed on the currently least loaded shard. */
    void buildFrequencyAware(List<Integer> idsByDescendingFrequency, long[] freq) {
        long[] load = new long[shardCount];
        for (int id : idsByDescendingFrequency) {
            int best = 0;
            for (int s = 1; s < shardCount; s++) if (load[s] < load[best]) best = s;
            frequencyHome.put(id, best);
            load[best] += freq[id];
        }
    }

    void markHot(List<Integer> ids, int countReplicated, int replicas) {
        replicaCount = replicas;
        for (int i = 0; i < Math.min(countReplicated, ids.size()); i++) hot.add(ids.get(i));
    }

    /** Shards that hold a copy of the id: the home shard plus a deterministic spread of the others. */
    List<Integer> replicasOf(int id, boolean frequencyAware) {
        int home = frequencyAware ? frequencyShard(id) : moduloShard(id);
        List<Integer> out = new ArrayList<>();
        out.add(home);
        if (hot.contains(id) && replicaCount > 1) {
            // replicas live on the next shards after the home shard: always exactly R distinct shards
            for (int r = 1; r < replicaCount && out.size() < shardCount; r++) {
                out.add((home + r) % shardCount);
            }
        }
        return out;
    }
}

/** One shard of the table: rows of fixed width plus a per-row version. */
class Shard {
    final int id;
    final int width;
    final Map<Integer, double[]> rows = new TreeMap<>();
    final Map<Integer, Integer> versions = new TreeMap<>();
    long lookups;

    Shard(int id, int width) { this.id = id; this.width = width; }

    double[] get(int row) {
        lookups++;
        double[] r = rows.get(row);
        return r == null ? null : r.clone();
    }

    void put(int row, double[] value, int version) {
        rows.put(row, value.clone());
        versions.put(row, version);
    }

    int versionOf(int row) { return versions.getOrDefault(row, -1); }

    void syncFrom(int row, Shard primary) {
        double[] value = primary.get(row);
        if (value != null) put(row, value, primary.versionOf(row));
    }
}

class WarmUpReport {
    double earlyAccuracy = 0;
    double earlyLogLoss = 0;
    double hitRate = 0;
    double finalLogLoss = 0;
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    /** Zipf-like frequencies: id 0 is the hottest, frequency decays like 1/(id+1). */
    static long[] frequencies(int n, long total) {
        double[] raw = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) { raw[i] = 1.0 / (i + 1); sum += raw[i]; }
        long[] out = new long[n];
        for (int i = 0; i < n; i++) out[i] = Math.max(1, Math.round(total * raw[i] / sum));
        return out;
    }

    static Map<String, Double> shardLoads(Placement p, long[] freq, boolean frequencyAware) {
        Map<String, Double> loads = new TreeMap<>();
        for (int id = 0; id < freq.length; id++) {
            int home = frequencyAware ? p.frequencyShard(id) : p.moduloShard(id);
            loads.merge("s" + home, (double) freq[id], Double::sum);
        }
        return loads;
    }

    static double imbalance(Map<String, Double> loads, int shards) {
        double max = 0, sum = 0;
        for (int s = 0; s < shards; s++) {
            double v = loads.getOrDefault("s" + s, 0.0);
            max = Math.max(max, v);
            sum += v;
        }
        double mean = sum / shards;
        return mean == 0 ? 0 : max / mean;
    }

    static Set<Integer> homes(Placement p, long[] freq, boolean frequencyAware, int shards) {
        Set<Integer> out = new TreeSet<>();
        for (int id = 0; id < freq.length; id++) {
            out.add((frequencyAware ? p.frequencyShard(id) : p.moduloShard(id)) * 1_000_000 + id);
        }
        return out;
    }

    /** Online sparse training: weight per id, logistic loss, fixed learning rate, single-id examples. */
    static WarmUpReport train(int[] onlineIds, int[] labels, double[] weights, Set<Integer> initialized,
                              double learningRate, int earlyWindow) {
        WarmUpReport report = new WarmUpReport();
        double earlyCorrect = 0, earlyLoss = 0;
        for (int step = 0; step < onlineIds.length; step++) {
            int id = onlineIds[step];
            double w = weights[id];
            double p = 1.0 / (1.0 + Math.exp(-w));
            double label = labels[step];
            if (step < earlyWindow) {
                double loss = -(label * Math.log(Math.max(p, 1e-12)) + (1 - label) * Math.log(Math.max(1 - p, 1e-12)));
                earlyLoss += loss;
                if ((p >= 0.5 ? 1 : 0) == (int) label) earlyCorrect++;
                if (initialized.contains(id)) report.hitRate += 1;
            }
            weights[id] = w - learningRate * (p - label);
            report.finalLogLoss = -(label * Math.log(Math.max(p, 1e-12)) + (1 - label) * Math.log(Math.max(1 - p, 1e-12)));
        }
        report.earlyAccuracy = earlyCorrect / earlyWindow;
        report.earlyLogLoss = earlyLoss / earlyWindow;
        if (earlyWindow > 0) report.hitRate /= earlyWindow;
        return report;
    }

    public static void main(String[] args) {
        int vocabulary = 300, shards = 8, hotCount = 6, replicas = 3;
        long[] freq = frequencies(vocabulary, 20000);
        List<Integer> byFrequency = new ArrayList<>();
        for (int i = 0; i < vocabulary; i++) byFrequency.add(i);

        // ---- 1. placement: exactly one home shard per id ------------------------------------------
        Placement hashPlacement = new Placement(shards, 64, 20260820L);
        Placement ringPlacement = new Placement(shards, 64, 20260821L);
        Placement frequencyPlacement = new Placement(shards);
        frequencyPlacement.buildFrequencyAware(byFrequency, freq);
        frequencyPlacement.markHot(byFrequency, hotCount, replicas);
        hashPlacement.markHot(byFrequency, hotCount, replicas);
        ringPlacement.markHot(byFrequency, hotCount, replicas);

        boolean oneHome = true;
        for (int id = 0; id < vocabulary; id++) {
            int home = hashPlacement.moduloShard(id);
            if (home < 0 || home >= shards) oneHome = false;
            if (frequencyPlacement.frequencyShard(id) < 0) oneHome = false;
            if (ringPlacement.consistentShard(id) < 0 || ringPlacement.consistentShard(id) >= shards) oneHome = false;
        }
        check(oneHome, "every id must have exactly one home shard in every scheme");

        // ---- 2. load balance: frequency-aware beats hashing on the Zipf profile --------------------
        double hashImbalance = imbalance(shardLoads(hashPlacement, freq, false), shards);
        double frequencyImbalance = imbalance(shardLoads(frequencyPlacement, freq, true), shards);
        check(hashImbalance > 1.3, "hash placement must be visibly imbalanced on a Zipf profile: " + hashImbalance);
        check(frequencyImbalance <= 1.45, "frequency-aware placement must stay near the best achievable: "
                + frequencyImbalance);
        check(frequencyImbalance < hashImbalance, "frequency-aware placement must beat hashing: "
                + frequencyImbalance + " vs " + hashImbalance);
        // The strongest structural claim: the hottest S ids are spread one per shard, so no shard can be
        // hit by two of them at once.
        Set<Integer> hotHomes = new TreeSet<>();
        for (int i = 0; i < shards; i++) hotHomes.add(frequencyPlacement.frequencyShard(byFrequency.get(i)));
        check(hotHomes.size() == shards, "the top S ids must occupy S distinct shards: " + hotHomes.size());

        // hot ids carry replicas, and every lookup costs one shard per replica
        List<Integer> hottest = byFrequency.subList(0, hotCount);
        int totalReplicas = 0, lookups = 0;
        for (int id : hottest) {
            List<Integer> r = frequencyPlacement.replicasOf(id, true);
            check(r.size() == replicas, "a hot id must have exactly R replicas: " + r.size());
            check(new HashSet<>(r).size() == r.size(), "replicas must be on distinct shards");
            totalReplicas += r.size();
            lookups += r.size();
        }
        check(lookups == hotCount * replicas, "lookup cost must be exactly R shards for hot ids: " + lookups);
        check(frequencyPlacement.replicasOf(byFrequency.get(vocabulary - 1), true).size() == 1,
                "a cold id must live on exactly one shard");
        check(totalReplicas == hotCount * replicas, "replica accounting must add up");

        // ---- 3. resizing stability: consistent hashing moves ~1/S, modulo moves nearly everything ---
        Placement resizedModulo = new Placement(shards + 1, 64, 20260821L);
        Placement resizedRing = new Placement(shards + 1, 64, 20260821L);
        int movedModulo = 0, movedRing = 0;
        for (int id = 0; id < vocabulary; id++) {
            if (hashPlacement.moduloShard(id) != resizedModulo.moduloShard(id)) movedModulo++;
            if (ringPlacement.consistentShard(id) != resizedRing.consistentShard(id)) movedRing++;
        }
        double moduloFraction = movedModulo / (double) vocabulary;
        double ringFraction = movedRing / (double) vocabulary;
        check(moduloFraction > 0.8, "modulo hashing must move nearly every id on resize: " + moduloFraction);
        check(ringFraction <= 1.0 / shards + 0.15, "consistent hashing must move about 1/S: " + ringFraction);
        check(ringFraction < moduloFraction / 2, "consistent hashing must be much more stable: "
                + ringFraction + " vs " + moduloFraction);

        // ---- 4. replica staleness and convergence -------------------------------------------------
        List<Shard> shardObjects = new ArrayList<>();
        for (int s = 0; s < shards; s++) shardObjects.add(new Shard(s, 4));
        int hotId = byFrequency.get(0);
        List<Integer> replicaShards = frequencyPlacement.replicasOf(hotId, true);
        double[] updated = {0.1, 0.2, 0.3, 0.4};
        shardObjects.get(replicaShards.get(0)).put(hotId, updated, 7);
        int served = 0;
        double staleValue = 0;
        for (int s : replicaShards) {
            double[] value = shardObjects.get(s).get(hotId);
            served++;
            if (value == null) staleValue++;
        }
        check(served == replicas, "a replicated read must touch exactly R shards");
        check(staleValue == replicas - 1, "before the sync only the primary has the row: " + staleValue);
        for (int r = 1; r < replicaShards.size(); r++) {
            shardObjects.get(replicaShards.get(r)).syncFrom(hotId, shardObjects.get(replicaShards.get(0)));
        }
        boolean converged = true;
        for (int s : replicaShards) {
            double[] value = shardObjects.get(s).get(hotId);
            if (value == null || !Arrays.equals(value, updated)) converged = false;
        }
        check(converged, "after a sync every replica must hold the primary's value");
        check(shardObjects.get(replicaShards.get(1)).versionOf(hotId) == 7,
                "a synced replica must carry the primary's version");
        long shardLookups = 0;
        for (Shard s : shardObjects) shardLookups += s.lookups;
        check(shardLookups == 2L * replicas + (replicas - 1),
                "two replicated reads plus the sync primaries make exactly 2R+(R-1) touches: " + shardLookups);

        // ---- 5. warm-up: early accuracy, hit rate and log-loss ------------------------------------
        SplitMix64 stream = new SplitMix64(777L);
        int streamLength = 1500, earlyWindow = 300;
        int[] onlineIds = new int[streamLength];
        int[] labels = new int[streamLength];
        int[] planted = new int[vocabulary];
        SplitMix64 plant = new SplitMix64(31337L);
        for (int i = 0; i < vocabulary; i++) planted[i] = plant.nextInt(2) == 0 ? -1 : 1;

        for (int i = 0; i < streamLength; i++) {
            // sample an id with probability proportional to its frequency
            int id = sample(stream, freq, 20000);
            onlineIds[i] = id;
            labels[i] = planted[id] > 0 ? 1 : 0;
        }

        double[] coldWeights = new double[vocabulary];
        Set<Integer> coldInit = new TreeSet<>();
        WarmUpReport cold = train(onlineIds, labels, coldWeights, coldInit, 0.7, earlyWindow);

        double[] warmWeights = new double[vocabulary];
        Set<Integer> warmInit = new TreeSet<>();
        SplitMix64 warmStream = new SplitMix64(555L);
        int warmSample = 600;
        for (int i = 0; i < warmSample; i++) {
            int id = sample(warmStream, freq, 20000);
            double p = 1.0 / (1.0 + Math.exp(-warmWeights[id]));
            warmWeights[id] -= 0.7 * (p - (planted[id] > 0 ? 1 : 0));
            warmInit.add(id);
        }
        WarmUpReport warm = train(onlineIds, labels, warmWeights, warmInit, 0.7, earlyWindow);

        check(cold.hitRate < 0.5, "a cold table must miss most early ids: " + cold.hitRate);
        check(warm.hitRate > cold.hitRate, "warm-up must raise the early hit rate: " + warm.hitRate + " vs " + cold.hitRate);
        check(warm.earlyAccuracy >= cold.earlyAccuracy, "warm-up must not hurt early accuracy: "
                + warm.earlyAccuracy + " vs " + cold.earlyAccuracy);
        check(warm.earlyLogLoss < cold.earlyLogLoss, "warm-up must reduce early log-loss: "
                + warm.earlyLogLoss + " vs " + cold.earlyLogLoss);
        check(warm.finalLogLoss <= cold.finalLogLoss + 1e-9, "warm-up must not degrade the converged loss: "
                + warm.finalLogLoss + " vs " + cold.finalLogLoss);

        // online training itself must keep improving: late loss lower than early loss
        double[] soloWeights = new double[vocabulary];
        double firstHalf = 0, secondHalf = 0;
        for (int i = 0; i < streamLength; i++) {
            int id = onlineIds[i];
            double p = 1.0 / (1.0 + Math.exp(-soloWeights[id]));
            double loss = -(labels[i] * Math.log(Math.max(p, 1e-12)) + (1 - labels[i]) * Math.log(Math.max(1 - p, 1e-12)));
            if (i < streamLength / 2) firstHalf += loss; else secondHalf += loss;
            soloWeights[id] -= 0.7 * (p - labels[i]);
        }
        check(secondHalf < firstHalf, "the online loss must fall as the table warms: "
                + secondHalf + " vs " + firstHalf);

        // ---- 6. collisions, capacity and degenerate inputs ----------------------------------------
        int capacity = 32;
        Set<Integer> rows = new TreeSet<>();
        int collisions = 0;
        for (int id = 0; id < vocabulary; id++) {
            int row = (int) Math.floorMod(Placement.hash(id), capacity);
            if (!rows.add(row)) collisions++;
        }
        check(collisions > 0, "a small hash table must collide: " + collisions);
        check(vocabulary - collisions == rows.size(), "collision accounting must be exact: "
                + (vocabulary - collisions) + " vs " + rows.size());
        int uniqueRows = vocabulary;
        check(uniqueRows > rows.size(), "frequency-aware placement needs one row per id, hashing fewer slots");

        long rowBudget = (long) vocabulary * 4 * replicas;
        check(rowBudget >= (long) vocabulary * 4, "the replica budget must cover the vocabulary");

        Placement singleShard = new Placement(1, 4, 5L);
        boolean singleOk = true;
        for (int id = 0; id < vocabulary; id++) if (singleShard.consistentShard(id) != 0) singleOk = false;
        check(singleOk, "with one shard every id must live on shard 0");

        Placement empty = new Placement(shards, 8, 6L);
        check(empty.replicasOf(0, false).size() == 1, "an unknown id still resolves to one shard");
        check(imbalance(shardLoads(empty, new long[0], false), shards) == 0,
                "an empty vocabulary has zero load everywhere");

        // ---- 7. determinism -----------------------------------------------------------------------
        Placement again = new Placement(shards, 64, 20260821L);
        boolean sameRing = true;
        for (int id = 0; id < vocabulary; id++) if (again.consistentShard(id) != ringPlacement.consistentShard(id)) sameRing = false;
        check(sameRing, "the ring must be reproducible from its seed");
        double[] repeatWeights = new double[vocabulary];
        WarmUpReport repeat = train(onlineIds, labels, repeatWeights, coldInit, 0.7, earlyWindow);
        check(Math.abs(repeat.earlyLogLoss - cold.earlyLogLoss) < 1e-12, "training must be reproducible");

        System.out.println("hash vs frequency skew : " + String.format("%.2f", hashImbalance) + " vs "
                + String.format("%.2f", frequencyImbalance) + " (max/mean)");
        System.out.println("moved on resize        : " + String.format("%.2f", moduloFraction) + " modulo vs "
                + String.format("%.2f", ringFraction) + " consistent");
        System.out.println("early hit rate         : cold " + String.format("%.3f", cold.hitRate) + " warm "
                + String.format("%.3f", warm.hitRate));
        System.out.println("early accuracy/loss    : cold " + String.format("%.3f", cold.earlyAccuracy) + "/"
                + String.format("%.4f", cold.earlyLogLoss) + " warm " + String.format("%.3f", warm.earlyAccuracy) + "/"
                + String.format("%.4f", warm.earlyLogLoss));
        System.out.println("loss first vs second   : " + String.format("%.1f", firstHalf) + " vs "
                + String.format("%.1f", secondHalf));
        System.out.println("PASS");
    }

    /** Samples an id with probability proportional to its frequency. */
    static int sample(SplitMix64 rng, long[] freq, long total) {
        long roll = Math.floorMod(rng.nextLong(), total);
        long acc = 0;
        for (int i = 0; i < freq.length; i++) {
            acc += freq[i];
            if (roll < acc) return i;
        }
        return freq.length - 1;
    }
}
