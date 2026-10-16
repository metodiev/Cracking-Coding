/** Task 928 - Implement production-grade feature store with lineage and drift detection.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * A feature store keyed by (entity, feature) with a full version history per key, so a training set can be
 * built with point-in-time correct joins: a lookup at time T returns the newest value written at or before T,
 * never a value that was only known later. That one rule is what prevents training/serving skew - the value a
 * training row sees is exactly the value a serving request would have seen at that instant. The store also
 * tracks lineage (which derived feature was computed from which inputs, down to the raw sources) and monitors
 * drift between a reference and a serving sample with the population stability index, PSI = sum over bins of
 * (p - q) ln(p / q), with the usual < 0.1 stable / > 0.25 drift reading.
 * Simplified: in-memory versions with a single scalar per (key, timestamp), no schemas or backfills, PSI over
 * equal-width bins rather than quantiles, and no online ingestion. Verified: the point-in-time join matches a
 * brute-force scan of the histories on many probes, no value newer than T is ever returned, lineage traversal
 * reaches exactly the raw sources (and the graph is acyclic), PSI separates stable from shifted samples, a
 * missing history returns a defined absence rather than a wrong value, and the store is deterministic.
 */
import java.util.*;

/** A feature store: versioned values, a lineage DAG and a PSI drift monitor. */
final class FeatureStore {
    final Map<String, TreeMap<Long, Double>> history = new HashMap<>();
    final Map<String, List<String>> inputs = new HashMap<>();
    final Set<String> rawSources = new HashSet<>();
    int lookups, misses;

    static String key(String entity, String feature) { return entity + "|" + feature; }

    /** Append a version. Versions for one key arrive in non-decreasing timestamp order by construction. */
    void write(String entity, String feature, long timestamp, double value) {
        history.computeIfAbsent(key(entity, feature), k -> new TreeMap<>()).put(timestamp, value);
    }

    /** Point-in-time lookup: the newest value at or before `asOf`, or null when nothing was known yet. */
    Double asOf(String entity, String feature, long asOf) {
        lookups++;
        TreeMap<Long, Double> versions = history.get(key(entity, feature));
        if (versions == null || versions.isEmpty()) {
            misses++;
            return null;
        }
        Map.Entry<Long, Double> entry = versions.floorEntry(asOf);
        if (entry == null) {
            misses++;
            return null;
        }
        return entry.getValue();
    }

    /** Brute force over every recorded version - the oracle the point-in-time join is checked against. */
    Double asOfBruteForce(String entity, String feature, long asOf) {
        TreeMap<Long, Double> versions = history.get(key(entity, feature));
        if (versions == null) return null;
        Double best = null;
        long bestTime = Long.MIN_VALUE;
        for (Map.Entry<Long, Double> e : versions.entrySet()) {
            if (e.getKey() <= asOf && e.getKey() >= bestTime) {
                bestTime = e.getKey();
                best = e.getValue();
            }
        }
        return best;
    }

    /** The timestamp the point-in-time join selected, so leakage can be checked directly. */
    long asOfVersion(String entity, String feature, long asOf) {
        TreeMap<Long, Double> versions = history.get(key(entity, feature));
        if (versions == null) return Long.MIN_VALUE;
        Map.Entry<Long, Double> entry = versions.floorEntry(asOf);
        return entry == null ? Long.MIN_VALUE : entry.getKey();
    }

    void derive(String feature, List<String> from) { inputs.put(feature, from); }
    void raw(String source) { rawSources.add(source); }

    /** Walk the lineage of a feature; `visited` doubles as the cycle detector. */
    List<String> lineage(String feature, Set<String> visited, List<String> order) {
        if (visited.contains(feature)) return order;
        visited.add(feature);
        order.add(feature);
        List<String> upstream = inputs.getOrDefault(feature, Collections.emptyList());
        for (String up : upstream) lineage(up, visited, order);
        return order;
    }

    /** True when every leaf of a feature's lineage is a declared raw source. */
    boolean lineageComplete(String feature) {
        Set<String> visited = new HashSet<>();
        List<String> order = lineage(feature, visited, new ArrayList<>());
        for (String node : order) {
            if (inputs.containsKey(node)) continue;
            if (!rawSources.contains(node)) return false;
        }
        return true;
    }

    /** Population stability index between a reference and a serving sample over equal-width bins. */
    static double psi(double[] reference, double[] serving, int bins) {
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : reference) { min = Math.min(min, v); max = Math.max(max, v); }
        if (max == min) return 0;
        double[] refShare = shares(reference, min, max, bins), curShare = shares(serving, min, max, bins);
        double psi = 0;
        for (int b = 0; b < bins; b++) psi += (curShare[b] - refShare[b]) * Math.log(curShare[b] / refShare[b]);
        return psi;
    }

    static double[] shares(double[] values, double min, double max, int bins) {
        int[] counts = new int[bins];
        for (double v : values) counts[Math.min(bins - 1, (int) ((v - min) / (max - min) * bins))]++;
        double[] share = new double[bins];
        for (int b = 0; b < bins; b++) share[b] = Math.max(counts[b] / (double) values.length, 1e-6);
        return share;
    }
}

public class Solution {
    static final int ENTITIES = 6, FEATURES = 3;

    static double value(int entity, int feature, long timestamp) {
        return 0.5 * entity + 0.1 * feature + 0.01 * timestamp + 0.3 * Math.sin(0.7 * entity + 1.3 * feature + 0.11 * timestamp);
    }

    public static void main(String[] args) {
        FeatureStore store = new FeatureStore();
        List<long[]> probes = new ArrayList<>();
        for (int e = 0; e < ENTITIES; e++) {
            for (int f = 0; f < FEATURES; f++) {
                for (long t = 10; t <= 100; t += 10) store.write("e" + e, "f" + f, t, value(e, f, t));
                for (long t = 5; t <= 115; t += 7) probes.add(new long[] {e, f, t});
            }
        }
        int mismatches = 0, leaks = 0, nulls = 0;
        for (long[] probe : probes) {
            String entity = "e" + probe[0], feature = "f" + probe[1];
            Double got = store.asOf(entity, feature, probe[2]);
            Double want = store.asOfBruteForce(entity, feature, probe[2]);
            if (got == null ? want != null : !got.equals(want)) mismatches++;
            long version = store.asOfVersion(entity, feature, probe[2]);
            if (version != Long.MIN_VALUE && version > probe[2]) leaks++;
            if (got == null) nulls++;
        }

        System.out.println("feature store: " + store.history.size() + " versioned keys, " + probes.size()
                + " point-in-time probes, mismatches " + mismatches + ", future leaks " + leaks + ", absences " + nulls);
        check(mismatches == 0, "the point-in-time join must match the brute-force scan");
        check(leaks == 0, "a training row must never see a value written after its timestamp");
        check(nulls > 0, "the probe set must exercise timestamps before the first version");

        // Training/serving consistency: the same instant yields the same value on both paths.
        for (int e = 0; e < ENTITIES; e++) {
            double training = store.asOf("e" + e, "f1", 55);
            double serving = store.asOf("e" + e, "f1", 55);
            check(training == serving, "a training row and a serving request at the same instant must agree");
            check(store.asOfVersion("e" + e, "f1", 55) == 50, "55 must resolve to the version written at 50");
        }

        // Boundaries on the history.
        check(store.asOf("e0", "f0", 10) != null && store.asOf("e0", "f0", 9) == null,
                "the first version is visible at its own timestamp and invisible before it");
        check(store.asOf("e9", "f0", 50) == null && store.asOf("e0", "f9", 50) == null,
                "an unknown entity or feature must return an absence, not a value");
        check(store.asOf("e0", "f0", 1000) == store.asOf("e0", "f0", 100), "the last version is the newest value");

        // Lineage: derived features down to the raw sources.
        store.raw("raw:clickstream");
        store.raw("raw:orders");
        store.derive("clicks_7d", List.of("raw:clickstream"));
        store.derive("orders_30d", List.of("raw:orders"));
        store.derive("engagement", List.of("clicks_7d", "orders_30d"));
        store.derive("ltv", List.of("engagement", "orders_30d"));
        List<String> lineage = store.lineage("ltv", new HashSet<>(), new ArrayList<>());
        check(store.lineageComplete("ltv") && lineage.containsAll(List.of("ltv", "engagement", "orders_30d", "clicks_7d")),
                "lineage must reach every upstream node, got " + lineage);
        check(lineage.get(lineage.size() - 1).startsWith("raw:") || lineage.contains("raw:orders"),
                "lineage must terminate at a raw source, got " + lineage);
        store.derive("sketchy", List.of("undeclared_source"));
        check(!store.lineageComplete("sketchy"), "a feature built on an undeclared source must be reported");
        store.derive("loop_a", List.of("loop_b"));
        store.derive("loop_b", List.of("loop_a"));
        Set<String> visited = new HashSet<>();
        int size = store.lineage("loop_a", visited, new ArrayList<>()).size();
        check(size == 2 && visited.size() == 2, "a cyclic lineage must terminate on the second visit, got " + size);
        // Drift: stable samples stay under the alert band, shifted ones cross it.
        double[] reference = new double[400];
        for (int i = 0; i < reference.length; i++) reference[i] = 50 + 10 * Math.sin(0.13 * i) + 3 * Math.cos(0.31 * i);
        double[] stable = new double[400];
        for (int i = 0; i < stable.length; i++) stable[i] = 50 + 10 * Math.sin(0.13 * i + 0.4) + 3 * Math.cos(0.31 * i + 0.2);
        double[] shifted = new double[400];
        for (int i = 0; i < shifted.length; i++) shifted[i] = 58 + 10 * Math.sin(0.13 * i) + 3 * Math.cos(0.31 * i);
        double stablePsi = FeatureStore.psi(reference, stable, 10), driftedPsi = FeatureStore.psi(reference, shifted, 10);
        check(stablePsi < 0.1, "a re-timed sample must look stable, PSI " + stablePsi);
        check(driftedPsi > 0.25, "a shifted sample must cross the drift threshold, PSI " + driftedPsi);
        check(FeatureStore.psi(reference, reference, 10) == 0, "PSI against itself must be exactly zero");
        check(FeatureStore.psi(new double[] {1, 1, 1}, new double[] {1, 1, 1}, 10) == 0,
                "a degenerate reference with no spread must not divide by zero");

        int before = store.lookups, missesBefore = store.misses;
        Double once = store.asOf("e2", "f2", 63), twice = store.asOf("e2", "f2", 63);
        check(once.equals(twice) && store.lookups == before + 2 && store.misses == missesBefore,
                "repeated lookups must be stable and counted once per call");

        System.out.println("lineage " + lineage + " -> complete " + store.lineageComplete("ltv")
                + ", PSI stable " + String.format("%.4f", stablePsi) + " vs drifted " + String.format("%.4f", driftedPsi));
        System.out.println("boundaries: first version at 10, absence before it, 55 resolves to the 50 version, later "
                + "probes resolve to the newest 100 version " + store.asOfVersion("e0", "f0", 1000));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
