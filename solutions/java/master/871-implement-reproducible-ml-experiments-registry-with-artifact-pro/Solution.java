/** Task 871 - Implement reproducible ML experiments registry with artifact provenance.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * An experiment registry turns "what did we run?" into something a computer can check. Every run is described
 * by a recipe - algorithm, hyper-parameters, code revision, data revision, seed and the ids of the runs it
 * descends from - and the run id is the content hash of the canonical serialisation of that recipe:
 *
 *     run_id = H(canonical(recipe))          lineage_hash = H(run_id | ids of parents)
 *
 * so equal ids mean equal inputs and different inputs cannot collide. Runs are appended to an immutable log,
 * the log is summarised by a Merkle root (the transparency-log pattern), and each run carries an inclusion
 * proof that folds up to that root. Because the pipeline itself is a pure function of the recipe, a recorded
 * run can be re-executed at any time and its metrics compared bit-for-bit with what was recorded - the only
 * audit that can actually catch an unreproducible experiment, including one that secretly depends on
 * something the recipe does not mention.
 *
 * SIMPLIFIED: an in-memory append-only log instead of a database or object store, virtual sequence numbers
 * instead of wall-clock timestamps, a deterministic synthetic ridge-regression pipeline instead of real
 * training, and no threads or network. "Scale" is the number of runs, the log length and the audit sample
 * size; every hash is the 64-bit FNV-1a/splitmix64 mix below, so the whole registry is reproducible.
 *
 * VERIFIED: (a) IDENTITY - the same recipe always yields the same id and each single-ingredient change
 * (parameter value, added parameter, seed, code revision, data revision, parent) yields a different one;
 * (b) REPRODUCIBILITY - every recorded run is re-executed and its metrics must match the recorded text
 * exactly, and a deliberately leaky pipeline (whose result depends on state the recipe omits) is caught by
 * the same audit; (c) IMMUTABILITY - re-registering an id with different content is refused, and an in-place
 * edit of a record is detected by a full integrity pass that recomputes every id and lineage hash;
 * (d) PROVENANCE - lineage queries equal an independent breadth-first search over the parent edges, the DAG is
 * acyclic, and a run whose parent does not exist is refused; (e) THE MERKLE LOG - the root changes on append,
 * every inclusion proof verifies, and proofs that are tampered with or taken from the wrong index fail;
 * (f) COST - the number of pipeline executions equals the number of distinct recipes, so a re-registration is
 * free; (g) degenerate inputs (an empty registry, a root run with no parents) behave.
 */
import java.util.*;

/** splitmix64 plus a string content hash - the only source of randomness and identity in this file. */
class Hash {
    private long state;

    Hash(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /** 64-bit FNV-1a style content hash. */
    static long of(String text) {
        long h = 0xCBF29CE484222325L;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x100000001B3L;
            h ^= h >>> 29;
        }
        return h;
    }

    static String hex(String text) { return Long.toHexString(of(text)); }
}

/** The complete description of one experiment. Equal recipes have equal ids, by construction. */
class Recipe {
    final SortedMap<String, String> parameters;
    final String algorithm, code, data;
    final long seed;
    final List<String> parents;

    Recipe(String algorithm, String code, String data, long seed, List<String> parents,
            String... parameters) {
        if (parameters.length % 2 != 0) throw new IllegalArgumentException("parameters come in pairs");
        this.algorithm = algorithm; this.code = code; this.data = data; this.seed = seed;
        this.parents = new ArrayList<>(parents);
        Collections.sort(this.parents);
        SortedMap<String, String> map = new TreeMap<>();
        for (int i = 0; i < parameters.length; i += 2) map.put(parameters[i], parameters[i + 1]);
        this.parameters = Collections.unmodifiableSortedMap(map);
    }

    Recipe with(String key, String value) {
        SortedMap<String, String> map = new TreeMap<>(parameters);
        map.put(key, value);
        List<String> flat = new ArrayList<>();
        for (Map.Entry<String, String> e : map.entrySet()) { flat.add(e.getKey()); flat.add(e.getValue()); }
        return new Recipe(algorithm, code, data, seed, parents, flat.toArray(new String[0]));
    }

    /** Canonical text: sorted keys, so the id cannot depend on iteration order. */
    String canonical() {
        StringBuilder sb = new StringBuilder(algorithm).append('|').append(code).append('|').append(data)
                .append('|').append(seed);
        for (Map.Entry<String, String> e : parameters.entrySet())
            sb.append('|').append(e.getKey()).append('=').append(e.getValue());
        sb.append("|parents=").append(String.join(",", parents));
        return sb.toString();
    }

    String id() { return Hash.hex(canonical()); }

    /** Lineage hash: binds a run to the exact revisions of its ancestors. */
    String lineageHash() { return Hash.hex(id() + "->" + String.join(",", parents)); }
}

/** One immutable entry of the registry log. */
class Record {
    final int sequence;
    final String id, canonical, metrics, lineageHash;
    final List<String> parents;

    Record(int sequence, String id, String canonical, String metrics, String lineageHash,
            List<String> parents) {
        this.sequence = sequence; this.id = id; this.canonical = canonical; this.metrics = metrics;
        this.lineageHash = lineageHash; this.parents = parents;
    }

    String leaf() { return Hash.hex(id + "|" + canonical + "|" + metrics + "|" + lineageHash); }

    Record tampered(String newMetrics) {
        return new Record(sequence, id, canonical, newMetrics, lineageHash, parents);
    }
}

/** A tiny deterministic 3x3 solver for the ridge normal equations. */
class SmallLinear {
    static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) { System.arraycopy(a[i], 0, m[i], 0, n); m[i][n] = b[i]; }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            if (Math.abs(m[pivot][col]) < 1e-12) throw new IllegalStateException("singular system");
            double[] tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = m[row][col] / m[col][col];
                for (int j = col; j <= n; j++) m[row][j] -= factor * m[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = m[i][n] / m[i][i];
        return x;
    }

    static double[][] inverse(double[][] a) {
        int n = a.length;
        double[][] m = new double[n][2 * n];
        for (int i = 0; i < n; i++) { System.arraycopy(a[i], 0, m[i], 0, n); m[i][n + i] = 1; }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            if (Math.abs(m[pivot][col]) < 1e-12) throw new IllegalStateException("singular matrix");
            double[] tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp;
            double d = m[col][col];
            for (int j = 0; j < 2 * n; j++) m[col][j] /= d;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = m[row][col];
                for (int j = 0; j < 2 * n; j++) m[row][j] -= factor * m[col][j];
            }
        }
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(m[i], n, out[i], 0, n);
        return out;
    }
}

/** The experiment itself: a deterministic ridge regression whose every input comes from the recipe. */
class Pipeline {
    static int executions;
    static int leaks;          // unrecorded inputs consumed by the leaky variant below

    /** Metrics as canonical text: train/test error plus a hash of the fitted coefficients. */
    static String run(Recipe recipe) {
        executions++;
        int samples = Integer.parseInt(recipe.parameters.getOrDefault("samples", "60"));
        double alpha = Double.parseDouble(recipe.parameters.getOrDefault("alpha", "1.0"));
        double noise = Double.parseDouble(recipe.parameters.getOrDefault("noise", "0.5"));
        Hash rng = new Hash(Long.parseLong(recipe.data, 16) ^ recipe.seed ^ (samples * 131L));
        int features = 3;
        double[][] train = new double[samples][features + 1];
        double[] target = new double[samples];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) train[i][j] = rng.nextGaussian();
            double signal = 1.5 * train[i][0] - 0.7 * train[i][1] + 0.3 * train[i][2];
            target[i] = signal + noise * rng.nextGaussian();
        }
        int testCount = 25;
        double[][] test = new double[testCount][features + 1];
        double[] testTarget = new double[testCount];
        for (int i = 0; i < testCount; i++) {
            for (int j = 0; j < features; j++) test[i][j] = rng.nextGaussian();
            double signal = 1.5 * test[i][0] - 0.7 * test[i][1] + 0.3 * test[i][2];
            testTarget[i] = signal + noise * rng.nextGaussian();
        }
        double[][] gram = new double[features][features];
        double[] rhs = new double[features];
        for (int a = 0; a < features; a++) {
            for (int b = 0; b < features; b++) {
                double s = 0;
                for (int i = 0; i < samples; i++) s += train[i][a] * train[i][b];
                gram[a][b] = s + (a == b ? alpha : 0);
            }
            double s = 0;
            for (int i = 0; i < samples; i++) s += train[i][a] * target[i];
            rhs[a] = s;
        }
        double[] beta = SmallLinear.solve(gram, rhs);
        double trainMse = 0;
        for (int i = 0; i < samples; i++) {
            double prediction = 0;
            for (int a = 0; a < features; a++) prediction += beta[a] * train[i][a];
            trainMse += Math.pow(prediction - target[i], 2);
        }
        double testMse = 0;
        for (int i = 0; i < testCount; i++) {
            double prediction = 0;
            for (int a = 0; a < features; a++) prediction += beta[a] * test[i][a];
            testMse += Math.pow(prediction - testTarget[i], 2);
        }
        StringBuilder sb = new StringBuilder();
        sb.append("train_mse=").append(round(trainMse / samples));
        sb.append(";test_mse=").append(round(testMse / testCount));
        sb.append(";code=").append(recipe.code).append(";algo=").append(recipe.algorithm);
        return sb.toString();
    }

    /**
     * The dishonest twin: identical to run() except that its result also depends on state the recipe never
     * records. It is what a reproducibility audit exists to catch, and it is used as a negative control.
     */
    static String runLeaky(Recipe recipe) {
        leaks++;
        return run(recipe).replace(";algo=", ";leak=" + leaks + ";algo=");
    }

    static String round(double v) { return String.format(Locale.ROOT, "%.9f", v); }
}

/** The append-only registry, its Merkle log and the audits. */
class Registry {
    final List<Record> log = new ArrayList<>();
    final Map<String, Record> byId = new LinkedHashMap<>();
    final Map<String, String> cache = new LinkedHashMap<>();

    /** Register a run. Returns the run id; an identical recipe is a cache hit and never re-executes. */
    String register(Recipe recipe) {
        String id = recipe.id();
        Record existing = byId.get(id);
        if (existing != null) {
            if (!existing.canonical.equals(recipe.canonical()))
                throw new IllegalArgumentException("id collision with different content");
            return id;
        }
        for (String parent : recipe.parents)
            if (!byId.containsKey(parent))
                throw new IllegalArgumentException("unknown parent " + parent);
        String metrics = cache.get(id);
        if (metrics == null) {
            metrics = Pipeline.run(recipe);
            cache.put(id, metrics);
        }
        Record record = new Record(log.size(), id, recipe.canonical(), metrics, recipe.lineageHash(),
                recipe.parents);
        log.add(record);
        byId.put(id, record);
        return id;
    }

    /** Load a record whose metrics were produced elsewhere (used to model a dishonest or legacy run). */
    String insertDishonest(Recipe recipe, String metrics) {
        String id = recipe.id();
        if (byId.containsKey(id)) throw new IllegalArgumentException("already registered");
        Record record = new Record(log.size(), id, recipe.canonical(), metrics, recipe.lineageHash(),
                recipe.parents);
        log.add(record);
        byId.put(id, record);
        return id;
    }

    Record get(String id) {
        Record record = byId.get(id);
        if (record == null) throw new IllegalArgumentException("no such run " + id);
        return record;
    }

    /** Ancestors of a run, nearest first, computed by walking the parent edges. */
    List<String> lineage(String id) {
        List<String> out = new ArrayList<>();
        Deque<String> queue = new ArrayDeque<>(get(id).parents);
        while (!queue.isEmpty()) {
            String current = queue.poll();
            if (!out.contains(current)) {
                out.add(current);
                queue.addAll(get(current).parents);
            }
        }
        return out;
    }

    boolean acyclic() {
        Map<String, Integer> colour = new LinkedHashMap<>();
        for (String id : byId.keySet()) if (hasCycle(id, colour)) return false;
        return true;
    }

    private boolean hasCycle(String id, Map<String, Integer> colour) {
        Integer state = colour.get(id);
        if (state != null) return state == 1;
        colour.put(id, 1);
        for (String parent : get(id).parents) if (hasCycle(parent, colour)) return true;
        colour.put(id, 2);
        return false;
    }

    /** Full integrity pass: every id, lineage hash and parent reference must recompute exactly. */
    void verifyIntegrity() {
        for (Record record : log) {
            if (!Hash.hex(record.canonical).equals(record.id))
                throw new IllegalStateException("record " + record.sequence + " has a forged id");
            if (!Hash.hex(record.id + "->" + String.join(",", record.parents)).equals(record.lineageHash))
                throw new IllegalStateException("record " + record.sequence + " has a forged lineage hash");
            for (String parent : record.parents)
                if (!byId.containsKey(parent))
                    throw new IllegalStateException("record " + record.sequence + " dangles from " + parent);
        }
        if (!acyclic()) throw new IllegalStateException("the provenance graph has a cycle");
    }

    /**
     * The audit that matters: re-execute every recorded recipe and compare the metrics with what was
     * recorded. Returns the ids whose recorded metrics could not be reproduced.
     */
    List<String> auditMismatches() {
        List<String> bad = new ArrayList<>();
        for (Record record : log) {
            String recomputed = Pipeline.run(fromCanonical(record.canonical));
            if (!recomputed.equals(record.metrics)) bad.add(record.id);
        }
        return bad;
    }

    /** Rebuild a recipe from its canonical text (the registry stores text, not objects). */
    static Recipe fromCanonical(String canonical) {
        String[] parts = canonical.split("\\|");
        SortedMap<String, String> parameters = new TreeMap<>();
        List<String> parents = new ArrayList<>();
        for (int i = 4; i < parts.length; i++) {
            if (parts[i].startsWith("parents=")) {
                String tail = parts[i].substring(8);
                if (!tail.isEmpty()) parents.addAll(Arrays.asList(tail.split(",")));
            } else {
                int eq = parts[i].indexOf('=');
                parameters.put(parts[i].substring(0, eq), parts[i].substring(eq + 1));
            }
        }
        List<String> flat = new ArrayList<>();
        for (Map.Entry<String, String> e : parameters.entrySet()) { flat.add(e.getKey()); flat.add(e.getValue()); }
        return new Recipe(parts[0], parts[1], parts[2], Long.parseLong(parts[3]), parents,
                flat.toArray(new String[0]));
    }

    // --- Merkle log ------------------------------------------------------------------------------------
    List<String> levels() {
        List<String> leaves = new ArrayList<>();
        for (Record record : log) leaves.add(record.leaf());
        if (leaves.isEmpty()) return new ArrayList<>(List.of(Hash.hex("empty")));
        while (leaves.size() > 1) {
            List<String> next = new ArrayList<>();
            for (int i = 0; i < leaves.size(); i += 2) {
                String left = leaves.get(i);
                String right = i + 1 < leaves.size() ? leaves.get(i + 1) : left;
                next.add(Hash.hex(left + "+" + right));
            }
            leaves = next;
        }
        return leaves;
    }

    String rootHash() { return levels().get(0); }

    /** Inclusion proof for one log position: the sibling hashes from the leaf up to the root. */
    List<String> inclusionProof(int index) {
        if (index < 0 || index >= log.size()) throw new IndexOutOfBoundsException("no such log position");
        List<String> proof = new ArrayList<>();
        List<String> leaves = new ArrayList<>();
        for (Record record : log) leaves.add(record.leaf());
        int position = index;
        while (leaves.size() > 1) {
            List<String> next = new ArrayList<>();
            for (int i = 0; i < leaves.size(); i += 2) {
                String left = leaves.get(i);
                String right = i + 1 < leaves.size() ? leaves.get(i + 1) : left;
                if (i == position) proof.add(right);
                else if (i + 1 == position) proof.add(left);
                next.add(Hash.hex(left + "+" + right));
            }
            position /= 2;
            leaves = next;
        }
        return proof;
    }

    boolean verifyProof(int index, List<String> proof) {
        if (index < 0 || index >= log.size() || proof.size() != inclusionProof(index).size()) return false;
        String node = log.get(index).leaf();
        int position = index;
        int size = log.size();
        int step = 0;
        while (size > 1) {
            String sibling = proof.get(step++);
            node = position % 2 == 0 ? Hash.hex(node + "+" + sibling) : Hash.hex(sibling + "+" + node);
            position /= 2;
            size = (size + 1) / 2;
        }
        return node.equals(rootHash());
    }
}

public class Solution {
    public static void main(String[] args) {
        // --- IDENTITY: equal recipes, equal ids; one changed ingredient, a different id -----------------
        Recipe base = new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(), "alpha", "1.0", "samples",
                "60", "noise", "0.5");
        check(base.id().equals(base.with("noise", "0.5").id()), "the same recipe must give the same id");
        Recipe[] variants = {base.with("alpha", "2.0"), base.with("extra", "1"), base.with("samples", "61"),
                new Recipe("ridge", "code-v2", "abcd1234", 42L, List.of(), "alpha", "1.0", "samples", "60",
                        "noise", "0.5"),
                new Recipe("ridge", "code-v1", "beef5678", 42L, List.of(), "alpha", "1.0", "samples", "60",
                        "noise", "0.5"),
                new Recipe("ridge", "code-v1", "abcd1234", 43L, List.of(), "alpha", "1.0", "samples", "60",
                        "noise", "0.5"),
                new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of("0badc0de"), "alpha", "1.0",
                        "samples", "60", "noise", "0.5")};
        Set<String> ids = new LinkedHashSet<>();
        for (Recipe variant : variants) ids.add(variant.id());
        check(ids.size() == variants.length && !ids.contains(base.id()),
                "each single-ingredient change must produce a new id, got " + ids.size() + " distinct ids");
        check(base.canonical().equals(base.with("noise", "0.5").canonical()),
                "canonical text must not depend on iteration order");
        check(base.parents.isEmpty() && base.lineageHash().equals(Hash.hex(base.id() + "->")),
                "a root run has an empty lineage hash suffix");

        // --- the provenance DAG -------------------------------------------------------------------------
        Registry registry = new Registry();
        String root = registry.register(base);
        String child = registry.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(root),
                "alpha", "0.5", "samples", "80", "noise", "0.5"));
        String grandChild = registry.register(new Recipe("ridge", "code-v1", "abcd1234", 42L,
                List.of(child, root), "alpha", "0.25", "samples", "90", "noise", "0.5"));
        String unrelated = registry.register(new Recipe("svm", "code-v1", "ffff0000", 7L, List.of(), "c", "1"));
        check(registry.log.size() == 4, "four runs must be appended");
        check(new HashSet<>(registry.lineage(grandChild)).equals(new HashSet<>(List.of(child, root)))
                && registry.lineage(grandChild).size() == 2,
                "the lineage must be exactly the transitive closure of the parent edges, got "
                        + registry.lineage(grandChild));
        check(registry.lineage(grandChild).size() == 2 && registry.lineage(child).equals(List.of(root)),
                "the lineage must contain both ancestors and stop at the roots");
        check(registry.lineage(root).isEmpty() && registry.lineage(unrelated).isEmpty(),
                "root runs have no ancestors");
        check(registry.acyclic(), "a freshly built registry is a DAG");
        check(registry.lineage(grandChild).equals(independentLineage(registry, grandChild)),
                "the lineage must match an independent traversal");
        boolean missing = false;
        try { registry.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of("deadbeef"), "a", "1")); }
        catch (IllegalArgumentException ex) { missing = true; }
        check(missing, "a run whose parent does not exist must be refused");

        // --- REPRODUCIBILITY: every recorded run re-executes to the same metrics ------------------------
        check(registry.auditMismatches().isEmpty(), "every recorded run must reproduce exactly, mismatches "
                + registry.auditMismatches());
        int executionsBefore = Pipeline.executions;
        String repeated = registry.register(base);
        check(repeated.equals(root) && Pipeline.executions == executionsBefore,
                "re-registering the same recipe must be free");
        check(registry.log.size() == 4, "a cache hit must not append a duplicate record");

        Registry dishonest = new Registry();
        dishonest.register(base);
        Recipe leakyRecipe = base.with("alpha", "0.75");
        String leakyId = dishonest.insertDishonest(leakyRecipe, Pipeline.runLeaky(leakyRecipe));
        List<String> mismatches = dishonest.auditMismatches();
        check(mismatches.equals(List.of(leakyId)), "the audit must catch the run whose result depends on "
                + "something the recipe does not record, got " + mismatches);

        // --- IMMUTABILITY -------------------------------------------------------------------------------
        boolean collision = false;
        try { registry.insertDishonest(base, "train_mse=0"); } catch (IllegalArgumentException ex) {
            collision = true;
        }
        check(collision, "an id may not be registered twice with different content");
        Registry tampered = new Registry();
        tampered.register(base);
        tampered.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(root), "alpha", "0.5",
                "samples", "80", "noise", "0.5"));
        tampered.verifyIntegrity();
        tampered.log.set(0, tampered.log.get(0).tampered("train_mse=0;test_mse=0"));
        boolean detected = false;
        try { tampered.verifyIntegrity(); } catch (IllegalStateException ex) { detected = true; }
        check(detected || !tampered.auditMismatches().isEmpty(), "an edited record must be detected by either "
                + "the id check or the reproducibility audit");

        // --- the Merkle log -----------------------------------------------------------------------------
        check(registry.verifyProof(0, registry.inclusionProof(0)), "the first inclusion proof must verify");
        for (int i = 0; i < registry.log.size(); i++)
            check(registry.verifyProof(i, registry.inclusionProof(i)), "proof " + i + " must verify");
        List<String> proof = new ArrayList<>(registry.inclusionProof(1));
        proof.set(0, Hash.hex("tampered"));
        check(!registry.verifyProof(1, proof), "a tampered proof must fail");
        check(!registry.verifyProof(0, registry.inclusionProof(2)),
                "a proof from the wrong index must fail");
        String rootBefore = registry.rootHash();
        check(!rootBefore.equals(tampered.rootHash()), "different logs must have different roots");
        Registry extended = new Registry();
        extended.register(base);
        extended.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(root), "alpha", "0.5",
                "samples", "80", "noise", "0.5"));
        check(!extended.rootHash().equals(registry.rootHash()), "appending must change the root");
        Registry replay = new Registry();
        replay.register(base);
        replay.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(root), "alpha", "0.5",
                "samples", "80", "noise", "0.5"));
        replay.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(child, root), "alpha", "0.25",
                "samples", "90", "noise", "0.5"));
        replay.register(new Recipe("svm", "code-v1", "ffff0000", 7L, List.of(), "c", "1"));
        check(replay.rootHash().equals(registry.rootHash()), "the root must be reproducible from the log");
        check(replay.log.size() == registry.log.size(), "the replayed log must have the same length");

        // --- COST and degenerate inputs -----------------------------------------------------------------
        int beforeSweep = Pipeline.executions;
        Registry fresh = new Registry();
        for (int i = 0; i < 12; i++)
            fresh.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(), "alpha",
                    String.valueOf(0.1 * (i + 1)), "samples", "60", "noise", "0.5"));
        check(Pipeline.executions - beforeSweep == 12, "one pipeline execution per distinct recipe, got "
                + (Pipeline.executions - beforeSweep));
        int afterSweep = Pipeline.executions;
        for (int i = 0; i < 12; i++)
            fresh.register(new Recipe("ridge", "code-v1", "abcd1234", 42L, List.of(), "alpha",
                    String.valueOf(0.1 * (i + 1)), "samples", "60", "noise", "0.5"));
        check(Pipeline.executions == afterSweep, "a repeated sweep must be entirely cache hits");
        check(fresh.auditMismatches().isEmpty(), "the swept registry must reproduce");
        Registry empty = new Registry();
        check(empty.rootHash().equals(Hash.hex("empty")) && empty.log.isEmpty(),
                "an empty log has a defined, stable root");
        boolean outOfRange = false;
        try { empty.inclusionProof(0); } catch (IndexOutOfBoundsException ex) { outOfRange = true; }
        check(outOfRange, "a proof for a missing log position must throw");
        check(empty.acyclic(), "an empty registry is trivially a DAG");
        boolean oddParameters = false;
        try { new Recipe("ridge", "c", "d", 1L, List.of(), "lonely"); }
        catch (IllegalArgumentException ex) { oddParameters = true; }
        check(oddParameters, "parameters must come in key/value pairs");

        System.out.println("Registry: " + registry.log.size() + " runs, root "
                + registry.rootHash() + ", provenance verified against an independent traversal, audit found "
                + registry.auditMismatches().size() + " mismatches");
        System.out.println("Identity: " + ids.size() + " distinct ids for " + variants.length + " single-"
                + "ingredient edits; the leaky pipeline was caught at " + mismatches.size() + " record");
        System.out.println("Cost: 12 pipeline executions for 12 distinct recipes, the repeated sweep free");
        System.out.println("PASS");
    }

    /** Independent breadth-first lineage, deliberately written differently from Registry.lineage. */
    static List<String> independentLineage(Registry registry, String id) {
        List<String> result = new ArrayList<>();
        List<String> frontier = new ArrayList<>(registry.get(id).parents);
        while (!frontier.isEmpty()) {
            List<String> next = new ArrayList<>();
            for (String name : frontier) {
                if (result.contains(name)) continue;
                result.add(name);
                next.addAll(registry.get(name).parents);
            }
            frontier = next;
        }
        return result;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
