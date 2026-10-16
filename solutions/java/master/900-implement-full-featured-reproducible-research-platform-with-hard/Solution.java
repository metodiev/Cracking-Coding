import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Task 900 - Implement full-featured reproducible research platform with hardware config capture.
 * (Tier: Master, CTCI Beyond - Software Engineering &amp; Tooling)
 *
 * <p>Model: a content-addressed experiment pipeline. Every stage declares its inputs, parameters,
 * an implementation revision and the hardware fields it is sensitive to; its address is
 * {@code H(stage, revision, canonical params, input addresses, hardware fingerprint)}. Stage costs
 * are declared constants, so "at scale" is typed parameters (stages, milliseconds), never measured
 * time.
 *
 * <p><b>What is simplified.</b> Stages are content-addressed, not executed; payload sizes come from
 * a splitmix64 stream keyed by the artifact hash. "Hardware" is a fixed capture (cpu, cores, mem,
 * gpu, driver, blas, threads, image) - no probing, scheduler, executor, network, threads, sleeping,
 * wall-clock time or {@code Math.random}. A cache is a hash-to-artifact map; invalidation is
 * implicit (a changed digest is a miss) and explained by a separate brute-force dependants walk.
 *
 * <p><b>Verified invariants.</b> Determinism (two cold runs on the same hardware agree bit for bit);
 * equivalence with an always-recompute reference that never consults a cache; memoisation soundness
 * (a warm re-run recomputes nothing and never changes a result hash); invalidation exactness (the
 * recomputed set equals the brute-force transitive dependants of the changed stages, and its cost
 * equals that frontier's declared cost, not the whole pipeline's); hardware capture (bumping a
 * field invalidates exactly the stages declaring it plus their dependants); auditability (provenance
 * verifies, manifests reproduce on the same machine, drift names exactly the affected stages, and
 * tampering with a parameter, address or hardware fingerprint is detected with Merkle taint on all
 * dependants); and degenerate shapes (empty pipeline, single stage, diamond DAG, cycles, unknown
 * dependencies, unknown hardware fields, unsorted input lists).
 */
final class SplitMix64 {
    private long state;

    SplitMix64(long seed) {
        this.state = seed;
    }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }
}

/** Deterministic, order-sensitive digest used to content-address every artifact. */
final class Digest {
    private long h = 0x243F6A8885A308D3L;

    Digest add(long value) {
        h ^= value * 0x9E3779B97F4A7C15L;
        h = Long.rotateLeft(h, 27) * 0xBF58476D1CE4E5B9L;
        h ^= h >>> 29;
        return this;
    }

    Digest add(String text) {
        add(text.length());
        for (int i = 0; i < text.length(); i++) add(text.charAt(i));
        return this;
    }

    String hex() {
        long z = h;
        z ^= z >>> 33;
        z *= 0xFF51AFD7ED558CCDL;
        z ^= z >>> 33;
        z *= 0xC4CEB9FE1A85EC53L;
        z ^= z >>> 33;
        return String.format("%016x", z);
    }

    static String of(Object... parts) {
        Digest digest = new Digest();
        for (Object part : parts) {
            if (part instanceof Iterable<?>) {
                digest.add("[");
                for (Object item : (Iterable<?>) part) digest.add(item == null ? "~null" : item.toString());
                digest.add("]");
            } else {
                digest.add(part == null ? "~null" : part.toString());
            }
        }
        return digest.hex();
    }
}

/** Captured hardware configuration; only the declared fields enter a stage's hash. */
final class HardwareConfig {
    static final List<String> KEYS = List.of("cpu", "cores", "mem", "gpu", "driver", "blas", "threads", "image");

    private final TreeMap<String, String> values = new TreeMap<>();

    static HardwareConfig baseline() {
        HardwareConfig hw = new HardwareConfig();
        hw.values.put("cpu", "x86-64 EPYC 7763");
        hw.values.put("cores", "64");
        hw.values.put("mem", "512GiB");
        hw.values.put("gpu", "NVIDIA A100-SXM4-80GB");
        hw.values.put("driver", "535.104.05");
        hw.values.put("blas", "openblas-0.3.24");
        hw.values.put("threads", "16");
        hw.values.put("image", "lab-base@sha256:2f1c9ab4");
        return hw;
    }

    HardwareConfig with(String key, String value) {
        if (!KEYS.contains(key)) throw new IllegalArgumentException("unknown hardware field: " + key);
        HardwareConfig copy = new HardwareConfig();
        copy.values.putAll(values);
        copy.values.put(key, value);
        return copy;
    }

    String get(String key) {
        if (!KEYS.contains(key)) throw new IllegalArgumentException("unknown hardware field: " + key);
        return values.getOrDefault(key, "~unset");
    }

    String fingerprint(Collection<String> keys) {
        List<String> parts = new ArrayList<>();
        for (String key : new TreeSet<>(keys)) parts.add(key + "=" + get(key));
        return Digest.of("hardware", parts);
    }

    String full() {
        return fingerprint(KEYS);
    }

    List<String> captured() {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<String, String> entry : values.entrySet()) lines.add(entry.getKey() + "\t" + entry.getValue());
        return lines;
    }
}

/** One pipeline stage: declared inputs, parameters, revision, cost and hardware sensitivity. */
final class Stage {
    final String id;
    final List<String> inputs;
    final TreeMap<String, String> params = new TreeMap<>();
    final long baseCostMs;
    final Set<String> hwKeys;
    String revision = "r1";

    Stage(String id, List<String> inputs, Map<String, String> params, long baseCostMs, Collection<String> hwKeys) {
        this.id = id;
        List<String> declared = new ArrayList<>(inputs);
        if (!new ArrayList<>(new TreeSet<>(declared)).equals(declared))
            throw new IllegalArgumentException("inputs of " + id + " must be sorted and deduplicated: " + declared);
        this.inputs = List.copyOf(declared);
        this.params.putAll(params);
        this.baseCostMs = baseCostMs;
        TreeSet<String> keys = new TreeSet<>(hwKeys);
        for (String key : keys)
            if (!HardwareConfig.KEYS.contains(key))
                throw new IllegalArgumentException("stage " + id + " declares unknown hardware field " + key);
        this.hwKeys = Collections.unmodifiableSet(keys);
    }

    Stage revision(String newRevision) {
        this.revision = newRevision;
        return this;
    }

    String canonicalParams() {
        List<String> parts = new ArrayList<>();
        for (Map.Entry<String, String> entry : params.entrySet()) parts.add(entry.getKey() + "=" + entry.getValue());
        return String.join(";", parts);
    }

    String outputHash(HardwareConfig hardware, Map<String, String> inputHashes) {
        List<String> wired = new ArrayList<>();
        for (String input : inputs) wired.add(input + ":" + inputHashes.getOrDefault(input, "~missing"));
        return Digest.of("artifact", id, revision, canonicalParams(), wired, hardware.fingerprint(hwKeys));
    }
}

/** An artifact addressed by its content hash; parameters and hashes are writable only to model tampering. */
final class Artifact {
    String hash;
    final String stageId;
    final List<String> inputs;
    String params;
    String hwFingerprint;
    final long sizeBytes;

    Artifact(String hash, String stageId, List<String> inputs, String params, String hwFingerprint, long sizeBytes) {
        this.hash = hash;
        this.stageId = stageId;
        this.inputs = List.copyOf(inputs);
        this.params = params;
        this.hwFingerprint = hwFingerprint;
        this.sizeBytes = sizeBytes;
    }
}

/** A DAG of stages with deterministic topological order and brute-force invalidation queries. */
final class Pipeline {
    private final TreeMap<String, Stage> stages = new TreeMap<>();

    Pipeline add(Stage stage) {
        if (stages.containsKey(stage.id)) throw new IllegalArgumentException("duplicate stage " + stage.id);
        stages.put(stage.id, stage);
        return this;
    }

    int size() {
        return stages.size();
    }

    Stage stage(String id) {
        Stage found = stages.get(id);
        if (found == null) throw new IllegalArgumentException("unknown stage " + id);
        return found;
    }

    List<String> ids() {
        return new ArrayList<>(stages.keySet());
    }

    List<String> topology() {
        for (Stage stage : stages.values())
            for (String input : stage.inputs)
                if (!stages.containsKey(input))
                    throw new IllegalArgumentException("stage " + stage.id + " depends on unknown stage " + input);
        TreeMap<String, Integer> indegree = new TreeMap<>();
        for (Stage stage : stages.values()) indegree.put(stage.id, stage.inputs.size());
        ArrayDeque<String> ready = new ArrayDeque<>();
        for (Map.Entry<String, Integer> entry : indegree.entrySet())
            if (entry.getValue() == 0) ready.add(entry.getKey());
        List<String> order = new ArrayList<>();
        while (!ready.isEmpty()) {
            String id = ready.poll();
            order.add(id);
            for (Stage stage : stages.values())
                if (stage.inputs.contains(id) && indegree.merge(stage.id, -1, Integer::sum) == 0) ready.add(stage.id);
        }
        if (order.size() != stages.size()) throw new IllegalArgumentException("the pipeline DAG contains a cycle");
        return order;
    }

    /** Transitive dependants (roots included) - the brute-force invalidated-set oracle. */
    Set<String> dependants(Collection<String> roots) {
        TreeSet<String> closure = new TreeSet<>();
        ArrayDeque<String> queue = new ArrayDeque<>();
        for (String root : roots) {
            if (!stages.containsKey(root)) throw new IllegalArgumentException("unknown stage " + root);
            closure.add(root);
            queue.add(root);
        }
        while (!queue.isEmpty()) {
            String id = queue.poll();
            for (Stage stage : stages.values())
                if (stage.inputs.contains(id) && closure.add(stage.id)) queue.add(stage.id);
        }
        return closure;
    }

    Set<String> hardwareSensitive(Collection<String> changedKeys) {
        TreeSet<String> out = new TreeSet<>();
        for (Stage stage : stages.values())
            for (String key : changedKeys)
                if (stage.hwKeys.contains(key)) out.add(stage.id);
        return out;
    }

    /** Copy-on-write parameter change; the original pipeline is untouched. */
    Pipeline withParam(String stageId, String key, String value) {
        if (!stages.containsKey(stageId)) throw new IllegalArgumentException("unknown stage " + stageId);
        Pipeline copy = new Pipeline();
        for (Stage stage : stages.values()) {
            Stage clone = new Stage(stage.id, stage.inputs, stage.params, stage.baseCostMs, stage.hwKeys);
            clone.revision(stage.revision);
            if (stage.id.equals(stageId)) clone.params.put(key, value);
            copy.add(clone);
        }
        return copy;
    }

    long costOf(Collection<String> ids) {
        long total = 0;
        for (String id : ids) total += stage(id).baseCostMs;
        return total;
    }

    long totalCost() {
        return costOf(ids());
    }
}

/** Memoising executor: a stage is recomputed only when its content address is absent from the cache. */
final class Engine {
    static final class Report {
        final TreeMap<String, String> byStage = new TreeMap<>();
        final List<String> recomputed = new ArrayList<>();
        long cost;
        int hits;
    }

    private final Map<String, Artifact> cache = new HashMap<>();
    private final Map<String, Artifact> latest = new TreeMap<>();
    private long totalCost;

    Report run(Pipeline pipeline, HardwareConfig hardware) {
        Report report = new Report();
        Map<String, String> hashes = new HashMap<>();
        for (String id : pipeline.topology()) {
            Stage stage = pipeline.stage(id);
            Map<String, String> inputHashes = new LinkedHashMap<>();
            List<String> wiring = new ArrayList<>();
            for (String input : stage.inputs) {
                inputHashes.put(input, hashes.get(input));
                wiring.add(hashes.get(input));
            }
            String hash = stage.outputHash(hardware, inputHashes);
            Artifact artifact = cache.get(hash);
            if (artifact == null) {
                long size = 4096L + Math.floorMod(new SplitMix64(Long.parseUnsignedLong(hash, 16)).nextLong(),
                        1L << 16);
                artifact = new Artifact(hash, id, wiring, stage.canonicalParams(), hardware.fingerprint(stage.hwKeys),
                        size);
                cache.put(hash, artifact);
                report.recomputed.add(id);
                report.cost += stage.baseCostMs;
                totalCost += stage.baseCostMs;
            } else {
                report.hits++;
            }
            latest.put(id, artifact);
            hashes.put(id, hash);
            report.byStage.put(id, hash);
        }
        return report;
    }

    Map<String, Artifact> store() {
        return new TreeMap<>(latest);
    }

    Map<String, Artifact> cache() {
        return new TreeMap<>(cache);
    }

    long totalCost() {
        return totalCost;
    }
}

/** Provenance / integrity audit over a run's artifact store. */
final class Audit {
    static List<String> verify(Pipeline pipeline, HardwareConfig hardware, Map<String, Artifact> store) {
        Set<String> known = new HashSet<>();
        for (Artifact artifact : store.values()) known.add(artifact.hash);
        Set<String> problems = new LinkedHashSet<>();
        for (Artifact artifact : store.values())
            for (String input : artifact.inputs)
                if (!known.contains(input)) problems.add(artifact.stageId + ":dangling-input");
        TreeSet<String> broken = new TreeSet<>();
        for (String id : pipeline.topology()) {
            Stage stage = pipeline.stage(id);
            Artifact artifact = store.get(id);
            if (artifact == null) {
                problems.add(id + ":missing");
                broken.add(id);
                continue;
            }
            Map<String, String> inputHashes = new LinkedHashMap<>();
            boolean wired = true;
            for (String input : stage.inputs) {
                Artifact upstream = store.get(input);
                if (upstream == null) {
                    problems.add(input + ":missing");
                    wired = false;
                    break;
                }
                inputHashes.put(input, upstream.hash);
            }
            if (!artifact.params.equals(stage.canonicalParams())) {
                problems.add(id + ":params");
                broken.add(id);
            }
            if (!artifact.hwFingerprint.equals(hardware.fingerprint(stage.hwKeys))) {
                problems.add(id + ":hardware");
                broken.add(id);
            }
            if (wired && !stage.outputHash(hardware, inputHashes).equals(artifact.hash)) {
                problems.add(id + ":hash");
                broken.add(id);
            }
        }
        // Merkle taint: whatever was built on a broken artifact cannot be trusted either.
        Set<String> reach = pipeline.dependants(broken);
        for (String id : pipeline.topology())
            if (reach.contains(id) && !broken.contains(id)) problems.add(id + ":tainted");
        return new ArrayList<>(problems);
    }
}

/** Canonical, textual reproduction record: captured hardware plus one content address per stage. */
final class Manifest {
    static String of(Pipeline pipeline, HardwareConfig hardware, Map<String, Artifact> store) {
        StringBuilder text = new StringBuilder("manifest 900-reproducible-platform v1\n");
        text.append("hardware\t").append(hardware.full()).append('\n');
        for (String capture : hardware.captured()) text.append("hwfield\t").append(capture).append('\n');
        for (String id : pipeline.topology()) {
            Artifact artifact = store.get(id);
            text.append("stage\t").append(id).append('\t').append(artifact.hash).append('\t')
                    .append(artifact.hwFingerprint).append('\t').append(artifact.params).append('\n');
        }
        return text.toString();
    }

    static TreeMap<String, String> addresses(String manifest) {
        TreeMap<String, String> out = new TreeMap<>();
        for (String line : manifest.split("\n")) {
            String[] fields = line.split("\t");
            if (fields.length > 2 && "stage".equals(fields[0])) out.put(fields[1], fields[2]);
        }
        return out;
    }

    static List<String> drift(String left, String right) {
        TreeMap<String, String> a = addresses(left);
        TreeMap<String, String> b = addresses(right);
        List<String> out = new ArrayList<>();
        for (String id : a.keySet())
            if (!Objects.equals(a.get(id), b.get(id))) out.add(id);
        return out;
    }
}

public class Solution {
    static int checks;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    /** Boundary probe: the action must be rejected with an IllegalArgumentException. */
    static void reject(Runnable action, String label) {
        try {
            action.run();
        } catch (IllegalArgumentException expected) {
            check(true, label + " must be rejected");
            return;
        }
        throw new AssertionError(label + " was accepted");
    }

    static Pipeline researchPipeline() {
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Stage("ingest", List.of(), Map.of("dataset", "corpus-v3", "shards", "2048"), 12_000L,
                List.of()));
        pipeline.add(new Stage("clean", List.of("ingest"), Map.of("minTokens", "32"), 8_000L, List.of()));
        pipeline.add(new Stage("features", List.of("clean"), Map.of("dim", "768"), 45_000L, List.of("blas")));
        pipeline.add(new Stage("train", List.of("features"), Map.of("lr", "0.05", "epochs", "30", "seed", "7"),
                180_000L, List.of("gpu", "driver")));
        pipeline.add(new Stage("eval-a", List.of("train"), Map.of("split", "dev"), 9_000L, List.of()));
        pipeline.add(new Stage("eval-b", List.of("train"), Map.of("split", "test"), 9_000L, List.of()));
        pipeline.add(new Stage("calibrate", List.of("eval-a", "eval-b"), Map.of("method", "isotonic"), 6_000L,
                List.of()));
        pipeline.add(new Stage("report", List.of("calibrate"), Map.of("format", "pdf"), 5_000L, List.of("cpu")));
        return pipeline;
    }

    static Pipeline scalePipeline(int width) {
        SplitMix64 rnd = new SplitMix64(0xC0FFEE1234L);
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Stage("ingest", List.of(), Map.of("dataset", "shard-set", "shards", "4096"), 21_000L,
                List.of()));
        for (int layer = 0; layer < 4; layer++) {
            for (int i = 0; i < width; i++) {
                TreeSet<String> inputs = new TreeSet<>();
                if (layer == 0) {
                    inputs.add("ingest");
                } else {
                    inputs.add("l" + (layer - 1) + "-" + (i % width));
                    inputs.add("l" + (layer - 1) + "-" + ((i * 7 + 3) % width));
                }
                Map<String, String> params = new TreeMap<>();
                params.put("lr", "0.01");
                params.put("seed", Integer.toString(rnd.nextInt(1 << 20)));
                params.put("shard", Integer.toString(i));
                pipeline.add(new Stage("l" + layer + "-" + i, new ArrayList<>(inputs), params, 4_000L +
                        rnd.nextInt(4_000), layer == 3 ? List.of("gpu", "driver") : List.of()));
            }
        }
        pipeline.add(new Stage("report", List.of("l3-0", "l3-25"), Map.of("format", "html"), 9_000L, List.of("cpu")));
        return pipeline;
    }

    static Map<String, Artifact> copyStore(Map<String, Artifact> store) {
        TreeMap<String, Artifact> out = new TreeMap<>();
        for (Map.Entry<String, Artifact> entry : store.entrySet()) {
            Artifact artifact = entry.getValue();
            out.put(entry.getKey(), new Artifact(artifact.hash, artifact.stageId, artifact.inputs, artifact.params,
                    artifact.hwFingerprint, artifact.sizeBytes));
        }
        return out;
    }

    /** Always-recompute reference implementation: no cache is ever consulted. */
    static Map<String, String> naiveHashes(Pipeline pipeline, HardwareConfig hardware) {
        TreeMap<String, String> out = new TreeMap<>();
        for (String id : pipeline.topology()) out.put(id, pipeline.stage(id).outputHash(hardware, out));
        return out;
    }

    public static void main(String[] args) {
        HardwareConfig labA = HardwareConfig.baseline();
        HardwareConfig labB = labA.with("driver", "545.23.06").with("cpu", "x86-64 EPYC 9654");

        // 1. Shape, declared cost model and topological order of the reference pipeline.
        Pipeline pipeline = researchPipeline();
        List<String> order = pipeline.topology();
        check(order.size() == 8, "the reference pipeline must have 8 stages, found " + order.size());
        check(order.equals(List.of("ingest", "clean", "features", "train", "eval-a", "eval-b", "calibrate",
                "report")), "unexpected topological order: " + order);
        check(pipeline.totalCost() == 274_000L, "declared pipeline cost must be 274000ms, got " +
                pipeline.totalCost());
        check(pipeline.dependants(List.of("train")) .equals(new TreeSet<>(List.of("train", "eval-a", "eval-b",
                "calibrate", "report"))), "the transitive dependants of train are wrong: " +
                        pipeline.dependants(List.of("train")));
        System.out.println("pipeline: " + pipeline.size() + " stages, declared cost " + pipeline.totalCost() +
                "ms, order " + order);

        // 2. Degenerate shapes and rejected graphs (boundaries).
        Pipeline empty = new Pipeline();
        Engine.Report emptyRun = new Engine().run(empty, labA);
        check(empty.topology().isEmpty() && emptyRun.byStage.isEmpty() && emptyRun.cost == 0L && empty.totalCost() == 0L, "an empty pipeline must run as a no-op");

        Pipeline single = new Pipeline().add(new Stage("only", List.of(), Map.of("k", "v"), 5L, List.of()));
        Engine singleEngine = new Engine();
        Engine.Report firstSingle = singleEngine.run(single, labA);
        Engine.Report secondSingle = singleEngine.run(single, labA);
        check(firstSingle.recomputed.size() == 1 && secondSingle.recomputed.isEmpty() && secondSingle.byStage.get("only").equals(firstSingle.byStage.get("only")), "a single-stage pipeline must compute once and then hit the cache");

        Pipeline cyclic = new Pipeline()
                .add(new Stage("a", List.of("b"), Map.of(), 1L, List.of()))
                .add(new Stage("b", List.of("a"), Map.of(), 1L, List.of()));
        Pipeline dangling = new Pipeline().add(new Stage("a", List.of("ghost"), Map.of(), 1L, List.of()));
        reject(() -> cyclic.topology(), "a cyclic pipeline");
        reject(() -> dangling.topology(), "a stage depending on an unknown stage");
        reject(() -> new Stage("bad", List.of(), Map.of(), 1L, List.of("gpuUuid")), "an unknown hardware field");
        reject(() -> new Stage("bad2", List.of("b", "a"), Map.of(), 1L, List.of()), "an unsorted input list");

        // 3. Cold run: every stage executes, cost equals the declared model, provenance verifies.
        Engine engine = new Engine();
        Engine.Report cold = engine.run(pipeline, labA);
        check(cold.recomputed.size() == order.size() && cold.hits == 0,
                "the cold run must compute every stage exactly once: " + cold.recomputed);
        check(cold.cost == pipeline.totalCost() && engine.totalCost() == pipeline.totalCost(),
                "the cold run must cost the full declared pipeline: " + cold.cost);
        check(Collections.frequency(cold.recomputed, "train") == 1,
                "the diamond's shared dependency must be computed exactly once");
        check(engine.cache().size() == order.size(), "content addressing must keep 8 distinct artifacts, got " +
                engine.cache().size());
        check(Audit.verify(pipeline, labA, engine.store()).isEmpty(), "the cold run's provenance does not verify: " +
                Audit.verify(pipeline, labA, engine.store()));
        check(Manifest.of(pipeline, labA, engine.store()).contains("hwfield\tcpu\tx86-64 EPYC 7763"),
                "the manifest must capture the hardware configuration verbatim");
        System.out.println("cold run: recomputed " + cold.recomputed.size() + ", cost " + cold.cost +
                "ms, artifacts " + engine.cache().size() + ", provenance ok");

        // 4. Determinism and equivalence with the always-recompute reference implementation.
        check(cold.byStage.equals(naiveHashes(pipeline, labA)),
                "the memoising engine disagrees with the always-recompute oracle on a cold run");
        Engine independent = new Engine();
        Engine.Report independentRun = independent.run(pipeline, labA);
        check(independentRun.byStage.equals(cold.byStage),
                "two independent runs on identical hardware must agree bit for bit");
        Engine.Report rememoised = independent.run(pipeline, labA);
        check(rememoised.recomputed.isEmpty() && rememoised.cost == 0L && rememoised.hits == order.size(),
                "an unchanged re-run on the same hardware must be a pure cache hit");
        check(rememoised.byStage.equals(cold.byStage), "reusing cached artifacts must not change any hash");
        System.out.println("determinism: warm re-run recomputed 0, saved " + pipeline.totalCost() + "ms (100%)");

        // 5. Parameter change: invalidation equals the brute-force dependants oracle, cost equals that frontier.
        Pipeline tuned = pipeline.withParam("train", "epochs", "40");
        Set<String> dirty = pipeline.dependants(List.of("train"));
        Engine.Report incremental = engine.run(tuned, labA);
        check(new TreeSet<>(incremental.recomputed).equals(new TreeSet<>(dirty)),
                "the recomputed set must equal the transitive dependants of the changed stage: got " +
                        incremental.recomputed + " want " + dirty);
        check(incremental.cost == pipeline.costOf(dirty) && incremental.cost == 209_000L,
                "the incremental cost must equal the dirty frontier cost, got " + incremental.cost);
        check(incremental.cost == pipeline.totalCost() - pipeline.costOf(List.of("ingest", "clean", "features")),
                "the incremental run must skip exactly the unaffected upstream stages: " + incremental.cost);
        check(incremental.hits == order.size() - dirty.size(),
                "every clean stage must be served from the content-addressed cache: " + incremental.hits);
        check(!incremental.byStage.get("train").equals(cold.byStage.get("train")) && incremental.byStage.get("ingest").equals(cold.byStage.get("ingest")), "the changed stage must move while its ancestors stay put");
        check(incremental.byStage.equals(naiveHashes(tuned, labA)),
                "memoisation must not change the result of the modified pipeline");
        System.out.println("param change: recomputed " + incremental.recomputed + ", cost " + incremental.cost +
                "ms (saving " + (pipeline.totalCost() - incremental.cost) + "ms), hits " + incremental.hits);

        // 6. Boundary: rewriting a parameter with its current value is not a change.
        Pipeline unchanged = pipeline.withParam("train", "epochs", "30");
        Engine.Report noop = engine.run(unchanged, labA);
        check(noop.recomputed.isEmpty() && noop.hits == order.size() && noop.cost == 0L,
                "setting a parameter to its current value must not invalidate anything: " + noop.recomputed);

        // 7. Hardware capture: exactly the stages declaring a changed field, plus their dependants.
        Engine hardwareEngine = new Engine();
        Engine.Report onA = hardwareEngine.run(pipeline, labA);
        Set<String> hwRoots = pipeline.hardwareSensitive(List.of("driver", "cpu"));
        Set<String> expectedBump = pipeline.dependants(hwRoots);
        Engine.Report onB = hardwareEngine.run(pipeline, labB);
        check(hwRoots.equals(new TreeSet<>(List.of("train", "report"))),
                "driver/cpu sensitivity must be declared by train and report, got " + hwRoots);
        check(new TreeSet<>(onB.recomputed).equals(new TreeSet<>(expectedBump)),
                "a hardware bump must invalidate exactly the sensitive stages and their dependants: got " +
                        onB.recomputed + " want " + expectedBump);
        check(onB.cost == pipeline.costOf(expectedBump),
                "the hardware bump must cost the invalidated frontier, got " + onB.cost);
        check(onB.byStage.get("ingest").equals(onA.byStage.get("ingest")) && onB.byStage.get("clean").equals(onA.byStage.get("clean")) && onB.byStage.get("features").equals(onA.byStage.get("features")), "hardware-independent stages must be reused unchanged across machines");
        check(!onB.byStage.get("train").equals(onA.byStage.get("train")),
                "a GPU/driver sensitive stage must produce a different content address on new hardware");
        System.out.println("hardware bump: roots " + hwRoots + ", recomputed " + onB.recomputed.size() + " of " +
                order.size() + " stages, cost " + onB.cost + "ms");

        // 8. Reproducibility manifests: identical hardware reproduces, other hardware drifts detectably.
        String manifestA = Manifest.of(pipeline, labA, engine.store());
        Engine tunedEngine = new Engine();
        tunedEngine.run(tuned, labA);
        String manifestTuned = Manifest.of(tuned, labA, tunedEngine.store());
        check(Manifest.addresses(manifestA).size() == order.size(),
                "a manifest must list one content address per stage");
        check(!Manifest.drift(manifestA, manifestTuned).isEmpty(),
                "a manifest built from changed parameters must drift");
        Engine otherBox = new Engine();
        Engine.Report coldB = otherBox.run(pipeline, labB);
        String manifestB = Manifest.of(pipeline, labB, otherBox.store());
        check(new TreeSet<>(Manifest.drift(manifestA, manifestB)).equals(new TreeSet<>(expectedBump)),
                "the reproduction diff must name exactly the hardware-affected stages: " + Manifest.drift(manifestA,
                        manifestB));
        check(coldB.byStage.equals(naiveHashes(pipeline, labB)), "the machine-B cold run disagrees with the oracle");
        System.out.println("manifest drift A->B: " + Manifest.drift(manifestA, manifestB));

        // 9. Tamper detection, including the Merkle taint cascading to dependants (negative checks).
        Map<String, Artifact> pristine = copyStore(engine.store());
        check(Audit.verify(pipeline, labA, pristine).isEmpty(), "the untouched store must verify: " +
                Audit.verify(pipeline, labA, pristine));
        Map<String, Artifact> tamperedParams = copyStore(engine.store());
        tamperedParams.get("train").params = tamperedParams.get("train").params + ";epochs=99";
        check(Audit.verify(pipeline, labA, tamperedParams).equals(List.of("train:params", "eval-a:tainted",
                "eval-b:tainted", "calibrate:tainted", "report:tainted")),
                        "a recorded parameter edited out of band must be caught: " + Audit.verify(pipeline, labA,
                                tamperedParams));
        Map<String, Artifact> tamperedHash = copyStore(engine.store());
        tamperedHash.get("features").hash = "0000000000000000";
        check(Audit.verify(pipeline, labA, tamperedHash).equals(List.of("train:dangling-input", "features:hash",
                "train:hash", "eval-a:tainted", "eval-b:tainted", "calibrate:tainted", "report:tainted")),
                        "swapping an intermediate artifact must be detected on it and downstream: " +
                                Audit.verify(pipeline, labA, tamperedHash));
        Map<String, Artifact> tamperedHw = copyStore(engine.store());
        tamperedHw.get("eval-a").hwFingerprint = "ffffffffffffffff";
        check(Audit.verify(pipeline, labA, tamperedHw) .equals(List.of("eval-a:hardware", "calibrate:tainted",
                "report:tainted")), "a forged hardware fingerprint must be caught: " + Audit.verify(pipeline, labA,
                        tamperedHw));
        Map<String, Artifact> missing = copyStore(engine.store());
        missing.remove("calibrate");
        check(Audit.verify(pipeline, labA, missing).contains("calibrate:missing") && Audit.verify(pipeline, labA,
                missing).contains("report:tainted"), "a missing artifact must be reported: " + Audit.verify(pipeline,
                        labA, missing));
        System.out.println("tamper checks: params, swapped artifact, hardware fingerprint, missing artifact and " +
                "taint cascades all detected");

        // 10. Scale: 202 stages, brute-force invalidation oracle and incremental cost bound.
        Pipeline scale = scalePipeline(50);
        Engine scaleEngine = new Engine();
        Engine.Report wide = scaleEngine.run(scale, labA);
        check(scale.size() == 202 && wide.recomputed.size() == scale.size() && wide.cost == scale.totalCost(),
                "the cold run over " + scale.size() + " stages must compute and charge every stage");
        Pipeline perturbed = scale.withParam("l1-7", "lr", "0.02");
        Set<String> scaleDirty = scale.dependants(List.of("l1-7"));
        Engine.Report partial = scaleEngine.run(perturbed, labA);
        check(partial.recomputed.size() == scaleDirty.size() && new TreeSet<>(partial.recomputed).equals(new TreeSet<>(scaleDirty)), "the recomputed set at scale must match the brute-force dependants oracle: got " + partial.recomputed.size() + " want " + scaleDirty.size());
        check(partial.recomputed.size() * 4 < scale.size(),
                "the dirty frontier must stay far below the pipeline size: " + partial.recomputed.size() + "/" +
                        scale.size());
        check(partial.cost == scale.costOf(scaleDirty), "the incremental cost must equal the dirty frontier, got " +
                partial.cost);
        check(partial.cost * 9 < wide.cost, "an incremental update must cost at least 9x less than the cold run: " +
                partial.cost + " vs " + wide.cost);
        check(partial.hits == scale.size() - scaleDirty.size(), "every clean stage must be a cache hit: " +
                partial.hits);
        check(partial.byStage.equals(naiveHashes(perturbed, labA)),
                "at scale, memoisation must reproduce the oracle's hashes exactly");
        Engine.Report scaleBump = scaleEngine.run(perturbed, labB);
        Set<String> scaleHwRoots = scale.hardwareSensitive(List.of("driver", "cpu"));
        check(new TreeSet<>(scaleBump.recomputed).equals(new TreeSet<>(scale.dependants(scaleHwRoots))),
                "at scale the hardware bump must invalidate exactly the sensitive frontier: " +
                        scaleBump.recomputed.size() + " vs " + scale.dependants(scaleHwRoots).size());
        check(scaleBump.byStage.get("ingest").equals(partial.byStage.get("ingest")) && scaleBump.cost == scale.costOf(scale.dependants(scaleHwRoots)), "hardware-independent stages must survive a machine change at scale");
        System.out.println("scale: " + scale.size() + " stages, cold " + wide.cost + "ms, incremental " +
                partial.cost + "ms over " + scaleDirty.size() + " stages, hardware frontier " +
                        scaleBump.recomputed.size() + " stages");

        System.out.println("hardware captured: " + labA.captured());
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
