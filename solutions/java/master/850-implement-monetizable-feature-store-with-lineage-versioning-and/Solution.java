/** Task 850 - Implement monetizable feature store with lineage, versioning, and online/offline sync. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A feature store has to answer four questions that pull in different directions: what did a feature look
 * like at training time (no leakage), what does it look like right now (online, with a freshness bound),
 * where did it come from (lineage), and who is allowed to read it (metering).
 *
 *   offline     - append-only records (entity, timestamp, value, version). A point-in-time read returns the
 *                 newest record at or before the query timestamp, which is what makes a training set free
 *                 of future leakage, and it can be pinned to a feature version.
 *   online      - the materialised latest value per entity with a logical clock and a TTL: a read inside
 *                 the TTL returns the value, a read after it returns null (stale), so consistency between
 *                 the two stores is a testable statement rather than a hope.
 *   lineage     - a DAG from each feature to its upstream features and raw sources. Registration rejects
 *                 cycles, and every leaf must be a declared raw source, so the provenance of any served
 *                 feature is complete and reproducible.
 *   monetization- per-tenant quota and billing: a served (or cached) read is billed at the tenant's unit
 *                 price, a rejected read is not, and the totals are checked against a hand-computed oracle.
 *
 * Simplifications, stated honestly: one process, in-memory stores, no network, no persistence and no
 * scheduler - materialisation is an explicit call with an explicit logical timestamp, which is exactly what
 * makes TTL expiry deterministic here. Values are doubles, entities are integers, and the "clock" is a
 * counter supplied by the caller, never wall-clock time.
 *
 * Invariants verified below. No leakage: a point-in-time read agrees with a brute-force scan and never
 * returns a record stamped after the query time (checked over every entity and timestamp). Sync: after
 * materialisation the online value equals the offline value, and it goes stale exactly when the TTL is
 * exceeded. Lineage: every feature's transitive closure reaches only declared raw sources, equals a DFS
 * closure, and registering a cycle is rejected. Versioning: a pinned read returns the older version's value
 * and a rollback restores it online. Metering: the billing totals equal the hand-computed oracle, rejected
 * reads are not billed, and the quota is enforced exactly. Cost: every store read is counted.
 */
import java.util.*;

/** splitmix64 - used only to build the deterministic test data. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** One offline observation. */
final class FeatureRecord {
    final int entity;
    final long timestamp;
    final double value;
    final int version;

    FeatureRecord(int entity, long timestamp, double value, int version) {
        this.entity = entity;
        this.timestamp = timestamp;
        this.value = value;
        this.version = version;
    }
}

/** The offline store for one feature view: append-only records, point-in-time reads. */
final class FeatureView {
    final String name;
    final List<FeatureRecord> records = new ArrayList<>();
    int versions;
    long reads;

    FeatureView(String name) {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("a view needs a name");
        this.name = name;
    }

    /** Appends an observation to the newest version of the view. */
    void append(int entity, long timestamp, double value) {
        if (entity < 0) throw new IllegalArgumentException("entities are non-negative");
        records.add(new FeatureRecord(entity, timestamp, value, versions));
    }

    void newVersion() { versions++; }

    /** The newest record at or before the timestamp, pinned to the newest version <= pinnedVersion. */
    FeatureRecord asOf(int entity, long timestamp, int pinnedVersion) {
        reads++;
        FeatureRecord best = null;
        for (FeatureRecord record : records) {
            if (record.entity != entity || record.timestamp > timestamp) continue;
            if (record.version > pinnedVersion) continue;
            if (best == null || record.timestamp > best.timestamp
                    || (record.timestamp == best.timestamp && record.version > best.version)) {
                best = record;
            }
        }
        return best;
    }

    /** Brute-force oracle: same rule, expressed independently. */
    FeatureRecord asOfOracle(int entity, long timestamp, int pinnedVersion) {
        List<FeatureRecord> eligible = new ArrayList<>();
        for (FeatureRecord record : records) {
            if (record.entity == entity && record.timestamp <= timestamp
                    && record.version <= pinnedVersion) eligible.add(record);
        }
        eligible.sort((a, b) -> a.timestamp != b.timestamp ? Long.compare(a.timestamp, b.timestamp)
                : Integer.compare(a.version, b.version));
        return eligible.isEmpty() ? null : eligible.get(eligible.size() - 1);
    }
}

/** The online store: materialised values with a TTL under a caller-supplied logical clock. */
final class OnlineStore {
    final Map<String, Map<Integer, double[]>> values = new TreeMap<>();
    final Map<String, Long> materialisedAt = new TreeMap<>();
    Map<String, Integer> materialisedVersion = new TreeMap<>();
    int reads;

    void materialize(FeatureView view, int version, long now, int limit) {
        Map<Integer, double[]> snapshot = new TreeMap<>();
        Set<Integer> entities = new TreeSet<>();
        for (FeatureRecord record : view.records) entities.add(record.entity);
        for (int entity : entities) {
            FeatureRecord record = view.asOf(entity, now, version);
            if (record == null) continue;
            snapshot.put(entity, new double[]{record.value, record.timestamp});
        }
        values.put(view.name, snapshot);
        materialisedAt.put(view.name, now);
        materialisedVersion.put(view.name, version);
    }

    /** Reads a served value, or null when it is stale or unknown. */
    Double read(String view, int entity, long now, int ttl) {
        reads++;
        Map<Integer, double[]> snapshot = values.get(view);
        if (snapshot == null || !snapshot.containsKey(entity)) return null;
        Long stamp = materialisedAt.get(view);
        if (stamp == null || now > stamp + ttl) return null;
        return snapshot.get(entity)[0];
    }

    int versionOf(String view) { return materialisedVersion.getOrDefault(view, -1); }
}

/** Lineage DAG: feature -> upstream features or raw sources. */
final class Lineage {
    final Map<String, Set<String>> upstreams = new TreeMap<>();
    final Set<String> rawSources = new TreeSet<>();
    int traversals;

    void declareRaw(String source) { rawSources.add(source); }

    /** Registers a feature with its upstreams, rejecting cycles and unknown raw sources. */
    void register(String feature, Set<String> upstream) {
        for (String parent : upstream) {
            if (parent.equals(feature)) throw new IllegalArgumentException("a feature cannot feed itself");
            if (upstreams.containsKey(parent) && reaches(parent, feature)) {
                throw new IllegalArgumentException("cycle through " + parent);
            }
            if (!upstreams.containsKey(parent) && !rawSources.contains(parent)) {
                throw new IllegalArgumentException("unknown upstream " + parent);
            }
        }
        upstreams.put(feature, new TreeSet<>(upstream));
    }

    /** True when `from` depends transitively on `target`. */
    boolean reaches(String from, String target) {
        Deque<String> stack = new ArrayDeque<>();
        Set<String> seen = new HashSet<>();
        stack.push(from);
        while (!stack.isEmpty()) {
            String current = stack.pop();
            if (!seen.add(current)) continue;
            if (current.equals(target)) return true;
            for (String parent : upstreams.getOrDefault(current, Set.of())) stack.push(parent);
        }
        return false;
    }

    /** The transitive closure over upstreams: every feature and raw source the feature depends on. */
    Set<String> closure(String feature) {
        traversals++;
        Set<String> result = new TreeSet<>();
        Deque<String> stack = new ArrayDeque<>();
        stack.push(feature);
        while (!stack.isEmpty()) {
            String current = stack.pop();
            for (String parent : upstreams.getOrDefault(current, Set.of())) {
                if (result.add(parent)) stack.push(parent);
            }
        }
        return result;
    }

    /** All leaves of the closure must be declared raw sources. */
    boolean complete(String feature) {
        for (String parent : closure(feature)) {
            if (!upstreams.containsKey(parent) && !rawSources.contains(parent)) return false;
        }
        return true;
    }
}

/** Metering: per-tenant quota, unit price and billing. */
final class Metering {
    final Map<String, Integer> quota = new TreeMap<>();
    final Map<String, Double> price = new TreeMap<>();
    final Map<String, Integer> served = new TreeMap<>();
    final Map<String, Integer> rejected = new TreeMap<>();
    final Map<String, Integer> cachedHits = new TreeMap<>();

    void register(String tenant, int limit, double unitPrice) {
        quota.put(tenant, limit);
        price.put(tenant, unitPrice);
    }

    /** Returns true when the read is served (and billed). */
    boolean serve(String tenant, boolean cacheHit) {
        int used = served.getOrDefault(tenant, 0);
        if (used >= quota.getOrDefault(tenant, 0)) {
            rejected.merge(tenant, 1, Integer::sum);
            return false;
        }
        served.merge(tenant, 1, Integer::sum);
        if (cacheHit) cachedHits.merge(tenant, 1, Integer::sum);
        return true;
    }

    double bill(String tenant) { return served.getOrDefault(tenant, 0) * price.getOrDefault(tenant, 0.0); }

    int rejectedCount(String tenant) { return rejected.getOrDefault(tenant, 0); }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(1701L);
        FeatureView clicks = new FeatureView("clicks_7d");
        int entities = 6, steps = 8;
        for (int entity = 0; entity < entities; entity++) {
            for (int t = 0; t < steps; t++) {
                clicks.append(entity, 100L * (t + 1), 10 * entity + rng.nextInt(5));
            }
        }
        clicks.newVersion();
        for (int entity = 0; entity < entities; entity++) {
            for (int t = 0; t < steps; t++) {
                clicks.append(entity, 100L * (t + 1) + 50, 100 + 10 * entity + rng.nextInt(5));
            }
        }
        check(clicks.versions == 1 && clicks.records.size() == entities * steps * 2,
                "the view must hold two versions, saw " + clicks.versions);

        // ---- 1. point-in-time reads: no leakage, and agreement with a brute-force scan -----------
        int compared = 0;
        for (int entity = 0; entity < entities; entity++) {
            for (long timestamp = 0; timestamp <= 900; timestamp += 37) {
                for (int pinned : new int[]{0, 1}) {
                    FeatureRecord mine = clicks.asOf(entity, timestamp, pinned);
                    FeatureRecord oracle = clicks.asOfOracle(entity, timestamp, pinned);
                    if (mine == null) {
                        check(oracle == null, "the oracle must agree on the missing case");
                    } else {
                        check(mine.timestamp <= timestamp,
                                "a point-in-time read must never return a future record");
                        check(mine.version <= pinned, "the pinned version must be respected");
                        check(oracle != null && mine.timestamp == oracle.timestamp
                                        && mine.version == oracle.version
                                        && Math.abs(mine.value - oracle.value) < 1e-12,
                                "the point-in-time read must match the oracle at " + entity + "@" + timestamp);
                    }
                    compared++;
                }
            }
        }
        check(compared >= 100, "the sweep must cover many (entity, timestamp, version) triples, saw "
                + compared);
        check(clicks.asOf(0, 50, 1) == null,
                "before the first record a point-in-time read is empty");
        check(clicks.asOf(0, 150, 0).value < 100 && clicks.asOf(0, 150, 1).value >= 100,
                "the pinned version must select the right family of values");
        check(clicks.asOf(-1, 100, 1) == null, "an unknown entity has no value");

        // ---- 2. online/offline sync and TTL ------------------------------------------------------
        OnlineStore online = new OnlineStore();
        long now = 900;
        int ttl = 100;
        online.materialize(clicks, 1, now, ttl);
        for (int entity = 0; entity < entities; entity++) {
            FeatureRecord offline = clicks.asOf(entity, now, 1);
            Double served = online.read(clicks.name, entity, now, ttl);
            check(served != null && Math.abs(served - offline.value) < 1e-12,
                    "the online value must equal the offline value for entity " + entity);
        }
        check(online.read(clicks.name, entities + 5, now, ttl) == null,
                "an entity that was never materialised reads null");
        check(online.read(clicks.name, 0, now + ttl, ttl) != null, "inside the TTL the value is fresh");
        check(online.read(clicks.name, 0, now + ttl + 1, ttl) == null,
                "one tick past the TTL the value must be stale");
        check(online.versionOf(clicks.name) == 1, "the online store must record the materialised version");

        // ---- 3. versioning and rollback ----------------------------------------------------------
        online.materialize(clicks, 0, now + ttl + 5, ttl);
        check(online.versionOf(clicks.name) == 0, "the rollback must pin the older version");
        for (int entity = 0; entity < entities; entity++) {
            FeatureRecord old = clicks.asOf(entity, now + ttl + 5, 0);
            Double served = online.read(clicks.name, entity, now + ttl + 5, ttl);
            check(served != null && Math.abs(served - old.value) < 1e-12,
                    "after the rollback the online store must serve the old version");
        }
        online.materialize(clicks, 1, now + ttl + 10, ttl);
        check(Math.abs(online.read(clicks.name, 0, now + ttl + 10, ttl)
                - clicks.asOf(0, now + ttl + 10, 1).value) < 1e-12,
                "re-materialising a newer version must restore the new values");

        // ---- 4. lineage: completeness, acyclicity and reproducibility ----------------------------
        Lineage lineage = new Lineage();
        lineage.declareRaw("events");
        lineage.declareRaw("profiles");
        lineage.register("clicks_7d", Set.of("events"));
        lineage.register("sessions", Set.of("events"));
        lineage.register("engagement", Set.of("clicks_7d", "sessions"));
        lineage.register("audience", Set.of("engagement", "profiles"));
        check(lineage.complete("audience"), "every leaf of the closure must be a declared raw source");
        check(lineage.closure("audience").equals(new TreeSet<>(Set.of("engagement", "profiles",
                "clicks_7d", "sessions", "events"))),
                "the lineage closure must be exactly the ancestry, got " + lineage.closure("audience"));
        check(lineage.reaches("audience", "events") && !lineage.reaches("events", "audience"),
                "the dependency direction must be respected");
        expectThrows(() -> lineage.register("engagement", Set.of("audience")), "a lineage cycle");
        expectThrows(() -> lineage.register("mystery", Set.of("nobody")), "an unknown upstream");
        expectThrows(() -> lineage.register("selfish", Set.of("selfish")), "a self-referential feature");
        check(lineage.closure("events").isEmpty(), "a raw source has no ancestors");
        Map<String, Integer> recordedVersions = new TreeMap<>();
        recordedVersions.put(clicks.name, online.versionOf(clicks.name));
        check(recordedVersions.get(clicks.name) == 1,
                "the materialisation record must let a value be recomputed later");
        FeatureRecord recomputed = clicks.asOf(0, now + ttl + 10, recordedVersions.get(clicks.name));
        check(Math.abs(recomputed.value - online.read(clicks.name, 0, now + ttl + 10, ttl)) < 1e-12,
                "the recorded version must reproduce the served value exactly");

        // ---- 5. metering and quotas --------------------------------------------------------------
        Metering metering = new Metering();
        metering.register("free", 5, 0.0);
        metering.register("pro", 100, 0.25);
        int proServed = 0, proRejected = 0, freeServed = 0;
        for (int i = 0; i < 60; i++) {
            boolean cached = i % 3 == 0;
            if (metering.serve("pro", cached)) proServed++;
            else proRejected++;
        }
        for (int i = 0; i < 9; i++) if (metering.serve("free", false)) freeServed++;
        check(proServed == 60 && proRejected == 0, "the pro quota must absorb all 60 reads, saw " + proServed);
        int secondBatchRejected = 0;
        for (int i = 0; i < 60; i++) if (!metering.serve("pro", false)) secondBatchRejected++;
        check(metering.rejectedCount("pro") == 20 && secondBatchRejected == 20,
                "reads past the quota must be rejected, saw " + metering.rejectedCount("pro"));
        check(Math.abs(metering.bill("pro") - 100 * 0.25) < 1e-12,
                "the pro bill must be the served count times the price, got " + metering.bill("pro"));
        check(metering.served.get("pro") == 100, "the quota must be exactly exhausted, saw "
                + metering.served.get("pro"));
        check(freeServed == 5 && Math.abs(metering.bill("free")) < 1e-12,
                "the free tier must allow exactly five reads and bill nothing, saw " + freeServed);
        int cachedTotal = 0;
        for (int count : metering.cachedHits.values()) cachedTotal += count;
        check(cachedTotal == 20, "twenty of the pro reads were cache hits, saw " + cachedTotal);

        // ---- 6. degenerate views, validation and determinism -------------------------------------
        FeatureView empty = new FeatureView("empty_view");
        check(empty.asOf(0, 1000, 0) == null && empty.asOfOracle(0, 1000, 0) == null,
                "an empty view has no values");
        OnlineStore emptyOnline = new OnlineStore();
        emptyOnline.materialize(empty, 0, 0, 10);
        check(emptyOnline.read("empty_view", 0, 0, 10) == null,
                "materialising an empty view leaves an empty online store");
        expectThrows(() -> new FeatureView(""), "an empty view name");
        expectThrows(() -> clicks.append(-1, 0, 1), "a negative entity");
        check(clicks.reads > 0 && online.reads > 0,
                "store reads must be counted, saw " + clicks.reads + " and " + online.reads);
        long readsBefore = clicks.reads;
        clicks.asOf(0, 500, 1);
        check(clicks.reads == readsBefore + 1, "every point-in-time read must be counted");
        FeatureRecord first = clicks.asOf(0, 100, 1);
        FeatureRecord second = clicks.asOf(0, 100, 1);
        check(first.timestamp == second.timestamp && first.value == second.value
                        && first.version == second.version,
                "the same query must be reproducible");
        check(clicks.asOf(0, 150, 0).timestamp == 100 && clicks.asOf(0, 150, 1).timestamp == 150,
                "version 1 must read the value appended after the version bump");

        System.out.println("offline: " + clicks.records.size() + " records, " + (compared / 2)
                + " point-in-time queries matched the brute-force scan in both versions");
        System.out.println("online/offline sync: exact at materialisation, fresh at ttl, stale at ttl+1;"
                + " rollback to version 0 and back to 1 verified");
        System.out.println("lineage: closure of audience = " + lineage.closure("audience") + ", cycles and"
                + " unknown upstreams rejected; metering bill pro = " + metering.bill("pro")
                + " for " + metering.served.get("pro") + " served and " + metering.rejectedCount("pro")
                + " rejected");
        System.out.println("PASS");
    }

    static void expectThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected a rejection for " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
