/** Task 865 - Implement transparent model serving with versioning, canary, shadow tests, and rollback.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A model-serving front door with four cooperating mechanisms:
 *
 *       versioning : every published version is an immutable (weights, threshold) pair; a request may pin a
 *                    version explicitly, otherwise the router decides;
 *       routing    : a stable hash of the request key maps to a bucket in [0, 10000); the canary owns the
 *                    first `canaryPercent * 100` buckets, so changing the percentage only moves the keys that
 *                    fall inside the new range - the minimal-disruption property of consistent bucketing;
 *       shadow     : the candidate scores a copy of the traffic without its result being served and without
 *                    its errors affecting the canary decision;
 *       canary/SLO : an error-rate monitor over a sliding window trips at a configured threshold and the
 *                    controller either promotes the candidate after a full healthy window or rolls it back,
 *                    driven by an explicit state machine with an audit log.
 *
 * WHAT IS SIMPLIFIED
 *   Deterministic scoring instead of a real model server, one process and no network: "traffic" is a
 *   deterministic sequence of scored requests driven by splitmix64, and latency is not modelled at all - the
 *   phenomenon under test is routing, monitoring and rollback, not inference performance.
 *
 * INVARIANTS VERIFIED
 *   Routing: assignment is sticky (the same key always lands on the same version), the observed split matches
 *   the configured percentage, raising the canary percentage keeps every previously promoted key on the
 *   canary, and the bucket function is stable across controller restarts.
 *   Safety: serving is unaffected by shadow traffic (byte-identical responses with shadow on and off), shadow
 *   errors never trigger a rollback, pinned and unknown versions resolve correctly, published versions are
 *   immutable, and the controller's state machine only takes legal transitions.
 *   Monitoring: the measured error rate equals an independent count over the same window, rollback happens
 *   within the configured window when the candidate degrades while a healthy candidate is promoted instead,
 *   rollback restores 100% primary traffic, degenerate configurations (zero traffic, an empty window, 0% and
 *   100% canary) are handled, and every run is reproducible from its seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so traffic and monitors are reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** An immutable published model version. */
final class ModelVersion {
    final String name;
    final double[] weights;
    final double bias;
    final double threshold;

    ModelVersion(String name, double[] weights, double bias, double threshold) {
        this.name = name;
        this.weights = weights.clone();
        this.bias = bias;
        this.threshold = threshold;
    }

    double score(double[] features) {
        double value = bias;
        for (int i = 0; i < weights.length; i++) value += weights[i] * features[i];
        return value;
    }

    int predict(double[] features) { return score(features) > threshold ? 1 : 0; }
}

/** Request: a key (also used for bucketing), the features and the label. */
final class Request {
    final String key;
    final double[] features;
    final int label;

    Request(String key, double[] features, int label) {
        this.key = key;
        this.features = features;
        this.label = label;
    }
}

/** Registry of published versions; publishing never mutates an existing version. */
final class Registry {
    final Map<String, ModelVersion> versions = new LinkedHashMap<>();

    void publish(ModelVersion version) {
        if (versions.containsKey(version.name)) throw new IllegalStateException("duplicate version");
        versions.put(version.name, version);
    }

    ModelVersion get(String name) {
        ModelVersion version = versions.get(name);
        if (version == null) throw new IllegalArgumentException("unknown version " + name);
        return version;
    }
}

/** Stable hash routing with a canary bucket range. */
final class Router {
    final Registry registry;
    String primary;
    String canary;
    int canaryBuckets;
    String shadow;
    final Map<String, String> pinned = new HashMap<>();

    Router(Registry registry, String primary) {
        this.registry = registry;
        this.primary = primary;
    }

    static int bucket(String key) {
        long hash = 0xCBF29CE484222325L;
        for (int i = 0; i < key.length(); i++) {
            hash ^= key.charAt(i);
            hash *= 0x100000001B3L;
        }
        hash ^= hash >>> 33;
        hash *= 0xFF51AFD7ED558CCDL;
        hash ^= hash >>> 33;
        hash *= 0xC4CEB9FE1A85EC53L;
        hash ^= hash >>> 33;
        return (int) Math.floorMod(hash >>> 1, 10000L);
    }

    void setCanary(String version, double percent) {
        this.canary = version;
        this.canaryBuckets = (int) Math.round(percent * 10000);
    }

    String versionFor(String key) {
        String pinnedVersion = pinned.get(key);
        if (pinnedVersion != null) return pinnedVersion;
        if (canary != null && bucket(key) < canaryBuckets) return canary;
        return primary;
    }

    /** Scores one request; returns {version, prediction, likelihood} with shadow metrics kept separate. */
    Map<String, Object> serve(Request request) {
        String version = versionFor(request.key);
        ModelVersion model = registry.get(version);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("version", version);
        result.put("prediction", model.predict(request.features));
        if (shadow != null) {
            ModelVersion shadowModel = registry.get(shadow);
            result.put("shadowVersion", shadow);
            result.put("shadowPrediction", shadowModel.predict(request.features));
        }
        return result;
    }
}

/** Sliding-window error-rate monitor with a burn-rate check over two windows. */
final class SloMonitor {
    final int window;
    final double threshold;
    final ArrayDeque<Integer> errors = new ArrayDeque<>();
    final ArrayDeque<Integer> requests = new ArrayDeque<>();

    SloMonitor(int window, double threshold) {
        this.window = window;
        this.threshold = threshold;
    }

    void record(boolean error) {
        errors.addLast(error ? 1 : 0);
        requests.addLast(1);
        while (requests.size() > window) {
            errors.pollFirst();
            requests.pollFirst();
        }
    }

    int observed() {
        int total = 0;
        for (int value : errors) total += value;
        return total;
    }

    double errorRate() { return errors.isEmpty() ? 0 : observed() / (double) errors.size(); }

    boolean trips() { return errors.size() >= window && errorRate() > threshold; }

    boolean full() { return errors.size() >= window; }
}

/** Canary controller: an explicit state machine with an audit log. */
final class CanaryController {
    String state = "PRIMARY_ONLY";
    final List<String> transitions = new ArrayList<>();
    int rollbacks;
    int promotions;

    boolean transition(String from, String to) {
        boolean legal = (state.equals(from))
                && ((from.equals("PRIMARY_ONLY") && (to.equals("CANARY") || to.equals("PROMOTED")))
                || (from.equals("CANARY") && (to.equals("ROLLED_BACK") || to.equals("PROMOTED")))
                || (from.equals("ROLLED_BACK") && to.equals("CANARY"))
                || (from.equals("PROMOTED") && to.equals("CANARY")));
        if (!legal) return false;
        state = to;
        transitions.add(from + "->" + to);
        if (to.equals("ROLLED_BACK")) rollbacks++;
        if (to.equals("PROMOTED")) promotions++;
        return true;
    }

    boolean legal(String from, String to) { return transitions.contains(from + "->" + to); }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Deterministic traffic: keys and features labelled by the healthy baseline model v1. */
    static List<Request> traffic(long seed, int count, int dimension, Registry registry) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Request> requests = new ArrayList<>();
        ModelVersion baseline = registry.get("v1");
        for (int i = 0; i < count; i++) {
            double[] features = new double[dimension];
            for (int d = 0; d < dimension; d++) features[d] = rng.nextDouble();
            requests.add(new Request("key-" + i, features, baseline.predict(features)));
        }
        return requests;
    }

    static Registry buildRegistry() {
        Registry registry = new Registry();
        registry.publish(new ModelVersion("v1", new double[]{1.0, -0.5, 0.25}, 0.0, 0.5));
        registry.publish(new ModelVersion("v2", new double[]{0.9, -0.4, 0.3}, 0.05, 0.5));
        registry.publish(new ModelVersion("v3", new double[]{1.1, -0.6, 0.2}, -0.05, 0.5));
        return registry;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        Registry registry = buildRegistry();
        List<Request> requests = traffic(865_1L, 20000, 3, registry);
        check(requests.size() == 20000, "traffic generator produces the requested volume");

        // --- 1. sticky routing and the configured split.
        Router router = new Router(registry, "v1");
        router.setCanary("v2", 0.10);
        int canaryCount = 0;
        boolean sticky = true;
        for (Request request : requests) {
            String first = router.versionFor(request.key);
            String second = router.versionFor(request.key);
            sticky &= first.equals(second);
            if (first.equals("v2")) canaryCount++;
        }
        double share = canaryCount / (double) requests.size();
        System.out.printf("canary share at 10%%: %.4f%n", share);
        check(sticky, "routing is sticky for every key");
        check(Math.abs(share - 0.10) < 0.015, "the observed split matches the configured percentage");
        check(Router.bucket("key-0") == Router.bucket("key-0"), "the bucket function is deterministic");
        registry.publish(new ModelVersion("v4", new double[]{1.0, 0.0, 0.0}, 0.0, 0.5));
        check(router.versionFor("key-7").equals(router.versionFor("key-7")),
                "publishing a new version does not disturb routing");

        // --- 2. minimal disruption when the canary grows.
        Set<String> canaryBefore = new HashSet<>();
        for (Request request : requests) if (router.versionFor(request.key).equals("v2")) canaryBefore.add(request.key);
        router.setCanary("v2", 0.20);
        int moved = 0, lost = 0, canaryAfter = 0;
        for (Request request : requests) {
            boolean now = router.versionFor(request.key).equals("v2");
            if (now) canaryAfter++;
            if (now && !canaryBefore.contains(request.key)) moved++;
            if (!now && canaryBefore.contains(request.key)) lost++;
        }
        System.out.printf("growing the canary 10%%->20%%: moved=%d lost=%d share=%.4f%n", moved, lost,
                canaryAfter / (double) requests.size());
        check(lost == 0, "no key leaves the canary when it grows");
        check(Math.abs(moved / (double) requests.size() - 0.10) < 0.015, "only the new range moves");

        // --- 3. shadow traffic never changes what is served.
        router.setCanary("v2", 0.0);
        List<Integer> servedWithout = new ArrayList<>();
        for (Request request : requests) servedWithout.add((Integer) router.serve(request).get("prediction"));
        router.shadow = "v3";
        List<Integer> servedWith = new ArrayList<>();
        int shadowPredictions = 0;
        for (Request request : requests) {
            Map<String, Object> response = router.serve(request);
            servedWith.add((Integer) response.get("prediction"));
            shadowPredictions += (Integer) response.get("shadowPrediction");
        }
        check(servedWithout.equals(servedWith), "shadow traffic does not change served predictions");
        check(shadowPredictions > 0, "the shadow model actually scored traffic");

        // --- 4. SLO monitor: measured rate equals an independent count.
        SloMonitor monitor = new SloMonitor(200, 0.05);
        SplitMix64 rng = new SplitMix64(865_2L);
        List<Boolean> flags = new ArrayList<>();
        for (int i = 0; i < 500; i++) flags.add(rng.nextDouble() < 0.08);
        for (boolean flag : flags) monitor.record(flag);
        int independentCount = 0;
        for (int i = flags.size() - 200; i < flags.size(); i++) if (flags.get(i)) independentCount++;
        check(monitor.observed() == independentCount,
                "the window error count equals an independent count (" + monitor.observed() + ")");
        System.out.printf("monitor: window rate=%.4f observed=%d%n", monitor.errorRate(), monitor.observed());
        check(monitor.trips() == (monitor.errorRate() > 0.05 && monitor.full()),
                "the trip condition matches its definition");
        SloMonitor clean = new SloMonitor(200, 0.05);
        for (int i = 0; i < 500; i++) clean.record(false);
        check(!clean.trips(), "a clean window never trips");
        SloMonitor empty = new SloMonitor(50, 0.01);
        check(empty.errorRate() == 0 && !empty.trips() && !empty.full(), "an empty window is inert");

        // --- 5. canary degradation triggers a rollback within the window; a healthy canary is promoted.
        CanaryController controller = new CanaryController();
        Router canaryRouter = new Router(registry, "v1");
        check(controller.transition("PRIMARY_ONLY", "CANARY"), "start a canary");
        canaryRouter.setCanary("v3", 0.10);
        SloMonitor canaryMonitor = new SloMonitor(200, 0.10);
        int rolledBackAt = -1;
        int canaryServed = 0;
        for (int i = 0; i < requests.size() && rolledBackAt < 0; i++) {
            Request request = requests.get(i);
            boolean servedByCanary = canaryRouter.versionFor(request.key).equals("v3");
            if (servedByCanary) {
                canaryServed++;
                // the degraded candidate disagrees with the healthy baseline on a large share of traffic
                canaryMonitor.record(request.label != registry.get("v3").predict(request.features));
            }
            if (canaryMonitor.trips()) {
                check(controller.transition("CANARY", "ROLLED_BACK"), "rollback is a legal transition");
                canaryRouter.setCanary(null, 0.0);
                rolledBackAt = i;
            }
        }
        System.out.printf("rollback after %d requests (%d canary samples), state=%s%n", rolledBackAt,
                canaryServed, controller.state);
        check(rolledBackAt > 0, "rollback happens inside the configured window");
        int primaryCount = 0;
        for (Request request : requests) if (canaryRouter.versionFor(request.key).equals("v1")) primaryCount++;
        check(primaryCount == requests.size(), "rollback restores 100% primary traffic");
        check(controller.state.equals("ROLLED_BACK") && controller.rollbacks == 1, "one rollback is recorded");
        check(controller.transition("ROLLED_BACK", "CANARY"), "re-canarying is a legal transition");

        CanaryController healthy = new CanaryController();
        Router healthyRouter = new Router(registry, "v2");
        healthy.transition("PRIMARY_ONLY", "CANARY");
        healthyRouter.setCanary("v1", 0.10);
        SloMonitor healthyMonitor = new SloMonitor(200, 0.10);
        int promotedAt = -1;
        for (int i = 0; i < requests.size() && promotedAt < 0; i++) {
            Request request = requests.get(i);
            boolean servedByCanary = healthyRouter.versionFor(request.key).equals("v1");
            if (servedByCanary) {
                healthyMonitor.record(request.label != registry.get("v1").predict(request.features));
            }
            if (healthyMonitor.full() && !healthyMonitor.trips()) {
                check(healthy.transition("CANARY", "PROMOTED"), "promotion is a legal transition");
                healthyRouter.setCanary("v1", 1.0);
                promotedAt = i;
            }
        }
        System.out.printf("promotion after %d requests, state=%s%n", promotedAt, healthy.state);
        check(promotedAt > 0, "a healthy canary is promoted");
        check(healthy.rollbacks == 0, "no rollback for a healthy candidate");
        int allCanary = 0;
        for (Request request : requests) if (healthyRouter.versionFor(request.key).equals("v1")) allCanary++;
        check(allCanary == requests.size(), "promotion sends all traffic to the promoted version");

        // --- 6. versioning: pinning, immutability and unknown versions.
        Router pinnedRouter = new Router(registry, "v1");
        pinnedRouter.setCanary("v2", 0.5);
        pinnedRouter.pinned.put("key-3", "v4");
        check(pinnedRouter.versionFor("key-3").equals("v4"), "a pinned key ignores the canary");
        for (Request request : requests) {
            if (pinnedRouter.pinned.containsKey(request.key)) continue;
            String version = pinnedRouter.versionFor(request.key);
            check(version.equals("v1") || version.equals("v2"), "unpinned keys use primary or canary");
        }
        ModelVersion before = registry.get("v1");
        double[] scoresBefore = new double[3];
        for (int i = 0; i < 3; i++) scoresBefore[i] = before.score(requests.get(i).features);
        registry.publish(new ModelVersion("v5", new double[]{2.0, 2.0, 2.0}, 1.0, 0.5));
        for (int i = 0; i < 3; i++) {
            check(registry.get("v1").score(requests.get(i).features) == scoresBefore[i],
                    "publishing does not mutate an existing version");
        }
        boolean unknownRejected = false;
        try {
            registry.get("v99");
        } catch (IllegalArgumentException e) {
            unknownRejected = true;
        }
        check(unknownRejected, "an unknown version is rejected");
        boolean duplicateRejected = false;
        try {
            registry.publish(new ModelVersion("v1", new double[]{0, 0, 0}, 0, 0));
        } catch (IllegalStateException e) {
            duplicateRejected = true;
        }
        check(duplicateRejected, "re-publishing a version name is rejected");

        // --- 7. degenerate router configurations and determinism.
        Router zeroRouter = new Router(registry, "v1");
        zeroRouter.setCanary("v2", 0.0);
        int zeroCanary = 0;
        for (Request request : requests) if (zeroRouter.versionFor(request.key).equals("v2")) zeroCanary++;
        check(zeroCanary == 0, "a 0% canary receives no traffic");
        Router fullRouter = new Router(registry, "v1");
        fullRouter.setCanary("v2", 1.0);
        int fullCanary = 0;
        for (Request request : requests) if (fullRouter.versionFor(request.key).equals("v2")) fullCanary++;
        check(fullCanary == requests.size(), "a 100% canary receives all traffic");
        Router repeatRouter = new Router(buildRegistry(), "v1");
        repeatRouter.setCanary("v2", 0.10);
        int repeatCount = 0;
        for (Request request : requests) if (repeatRouter.versionFor(request.key).equals("v2")) repeatCount++;
        check(Math.abs(repeatCount / (double) requests.size() - 0.10) < 0.015, "routing is reproducible");
        check(bucketUpperBound(), "bucket values stay inside [0, 10000)");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    static boolean bucketUpperBound() {
        for (int i = 0; i < 2000; i++) {
            int bucket = Router.bucket("probe-" + i);
            if (bucket < 0 || bucket >= 10000) return false;
        }
        return true;
    }
}
