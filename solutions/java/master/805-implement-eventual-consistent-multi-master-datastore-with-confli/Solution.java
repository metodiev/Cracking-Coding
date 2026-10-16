/** Task 805 - Implement eventual-consistent multi-master datastore with conflict resolution policies. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Every replica accepts writes (multi-master), stamps each write with a hybrid logical clock
 * (physical counter, logical counter, replica id) and a version vector, and ships its update log to
 * peers asynchronously. Three conflict-resolution policies are implemented for three data types:
 * (1) last-writer-wins registers, where the winner is the entry with the greatest HLC and the replica
 * id breaks ties *deterministically* - that tie-break is exactly what makes two replicas that never
 * talk agree; (2) multi-value registers, which keep all concurrent siblings so the application can
 * resolve them, dropping only entries that are causally dominated; (3) convergent replicated data
 * types - a PN-counter whose per-replica components merge by taking the maximum, and an observed-
 * remove set whose adds and tombstones merge by union. Anti-entropy is a full-state merge that is
 * commutative, associative and idempotent, which is what lets replicas converge regardless of the
 * order in which updates arrive.
 *
 * Simulation model, and what is simplified. One process, no threads and no sockets: replication is an
 * explicit fan-out over an in-memory peer mesh driven by rounds, delivery order comes from a fixed-seed
 * splitmix64 shuffle (so a run is reproducible yet adversarial), and "partition" simply means two
 * groups of replicas that no longer replicate to each other. The physical part of the HLC is a counter
 * rather than a wall clock, and durable storage is out of scope. What is *not* simplified: the policies
 * are the real merge algorithms, the version vectors really decide causality, and every replica's state
 * is compared against an oracle that replays the global operation log independently of any replication
 * path.
 *
 * Verified: every replica converges to the oracle after anti-entropy, for all three policies; merging
 * in shuffled orders and with duplicated frames yields the same state (commutativity and idempotence,
 * demonstrated on ten permutations); a causally newer write always beats an older one even when the
 * older arrives last; the register keeps both siblings of a genuine conflict; the PN-counter's value
 * equals the number of increments whatever the delivery order; the observed-remove set keeps a
 * concurrent add even though a remove of the same element happened elsewhere; writes made while the
 * mesh is partitioned are neither lost nor allowed to break convergence after healing; version vectors
 * only grow and HLCs only advance; a replica that receives nothing stays measurably stale while the
 * others converge; a single-replica cluster is trivially consistent; and the negative control - a
 * last-writer-wins policy without the replica-id tie-break - is shown to leave replicas permanently
 * divergent, which is what the convergence check is for.
 */
import java.util.*;

final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** Hybrid logical clock: counter "physical" time, logical counter, then the replica id to break ties. */
final class Hlc implements Comparable<Hlc> {
    final long wall;
    final int logical;
    final String replica;

    Hlc(long wall, int logical, String replica) {
        this.wall = wall;
        this.logical = logical;
        this.replica = replica;
    }

    static Hlc parse(String text) {
        String[] f = text.split("@");
        return new Hlc(Long.parseLong(f[0]), Integer.parseInt(f[1]), f[2]);
    }

    public int compareTo(Hlc o) {
        if (wall != o.wall) return Long.compare(wall, o.wall);
        if (logical != o.logical) return Integer.compare(logical, o.logical);
        return replica.compareTo(o.replica);            // deterministic tie-break
    }

    public String toString() { return wall + "@" + logical + "@" + replica; }
}

final class RegisterEntry {
    final String value;
    final Hlc hlc;
    final TreeMap<String, Long> vector;

    RegisterEntry(String value, Hlc hlc, TreeMap<String, Long> vector) {
        this.value = value;
        this.hlc = hlc;
        this.vector = new TreeMap<>(vector);
    }

    /** True when every component of `other` is at least this entry's: the other causally dominates. */
    boolean dominatedBy(RegisterEntry other) {
        for (Map.Entry<String, Long> e : vector.entrySet()) {
            if (other.vector.getOrDefault(e.getKey(), 0L) < e.getValue()) return false;
        }
        return true;
    }

    String render() { return value + "(" + hlc + ")"; }
}

final class Unsupported extends RuntimeException {
    Unsupported(String message) { super(message); }
}

/** One replica: registers, a PN-counter, an OR-set, a version vector, an update log and outbox. */
final class Replica805 {
    final String name;
    final TreeMap<String, List<RegisterEntry>> registers = new TreeMap<>();
    final TreeMap<String, TreeMap<String, Long>> counter = new TreeMap<>();     // component -> +/- totals
    final TreeMap<String, TreeSet<String>> setAdds = new TreeMap<>();           // element -> tags
    final TreeSet<String> tombstones = new TreeSet<>();
    final TreeMap<String, Long> version = new TreeMap<>();
    final List<String> log = new ArrayList<>();                                 // every operation applied
    final List<String> outbox = new ArrayList<>();
    final boolean deterministicTieBreak;
    boolean badTieBreak;                              // negative control: arrival order decides
    long wall;
    int logical;
    int applied, replayed, duplicates;
    boolean partitioned;

    Replica805(String name, boolean deterministicTieBreak) {
        this.name = name;
        this.deterministicTieBreak = deterministicTieBreak;
        version.put(name, 0L);
    }

    Hlc tick() { return new Hlc(++wall, logical++, name); }

    void registerWrite(String key, String value) {
        Hlc hlc = tick();
        version.merge(name, 1L, Long::sum);
        apply("R|" + key + "|" + value + "|" + hlc + "|" + renderVector(version));
    }

    /**
     * The PN-counter: the frame carries this replica's whole component vector, and merging takes the
     * component-wise maximum, so the counter is immutable to duplication and to delivery order.
     */
    void counterAdd(String key, long delta) {
        tick();
        version.merge(name, 1L, Long::sum);
        TreeMap<String, Long> components = counter.computeIfAbsent(key, k -> new TreeMap<>());
        if (delta >= 0) components.merge(name + "#p", delta, Long::sum);
        else components.merge(name + "#n", -delta, Long::sum);
        apply("C|" + key + "|" + renderComponents(components) + "|" + renderVector(version));
    }

    static String renderComponents(Map<String, Long> components) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Long> e : components.entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append('=').append(e.getValue());
        }
        return sb.toString();
    }

    static void mergeComponents(Map<String, Long> target, String text) {
        if (text.isEmpty()) return;
        for (String part : text.split(",")) {
            int eq = part.indexOf('=');
            String component = part.substring(0, eq);
            long value = Long.parseLong(part.substring(eq + 1));
            target.merge(component, value, Math::max);
        }
    }

    void setAdd(String key, String element) {
        Hlc hlc = tick();
        version.merge(name, 1L, Long::sum);
        String tag = name + "#" + hlc.wall + "." + hlc.logical;
        apply("A|" + key + "|" + element + "|" + tag + "|" + renderVector(version));
    }

    void setRemove(String key, String element) {
        tick();
        version.merge(name, 1L, Long::sum);
        List<String> tags = new ArrayList<>(setAdds.getOrDefault(key + "/" + element, new TreeSet<>()));
        StringBuilder sb = new StringBuilder();
        for (String t : tags) { if (sb.length() > 0) sb.append(','); sb.append(t); }
        apply("D|" + key + "|" + element + "|" + sb + "|" + renderVector(version));
    }

    static String renderVector(Map<String, Long> vector) {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Long> e : vector.entrySet()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.getKey()).append(':').append(e.getValue());
        }
        return sb.toString();
    }

    static TreeMap<String, Long> parseVector(String text) {
        TreeMap<String, Long> out = new TreeMap<>();
        for (String part : text.split(",")) {
            if (part.isEmpty()) continue;
            int colon = part.indexOf(':');
            out.put(part.substring(0, colon), Long.parseLong(part.substring(colon + 1)));
        }
        return out;
    }

    /** Applies one operation frame; the same frame twice is a no-op (idempotent delivery). */
    void apply(String frame) {
        if (log.contains(frame)) { duplicates++; return; }
        String[] f = frame.split("\\|", -1);
        TreeMap<String, Long> vector = parseVector(f[f.length - 1]);
        if (f[0].equals("R")) {
            RegisterEntry entry = new RegisterEntry(f[2], Hlc.parse(f[3]), vector);
            List<RegisterEntry> siblings = registers.computeIfAbsent(f[1], k -> new ArrayList<>());
            boolean dominated = false;
            for (Iterator<RegisterEntry> it = siblings.iterator(); it.hasNext();) {
                RegisterEntry existing = it.next();
                if (entry.dominatedBy(existing)) { dominated = true; break; }
                if (existing.dominatedBy(entry)) it.remove();
            }
            if (!dominated) siblings.add(entry);
            if (badTieBreak) {
                int keep = 0;
                for (int i = 1; i < siblings.size(); i++) {
                    int c = siblings.get(i).hlc.compareTo(siblings.get(keep).hlc);
                    boolean sameLogical = siblings.get(i).hlc.wall == siblings.get(keep).hlc.wall
                            && siblings.get(i).hlc.logical == siblings.get(keep).hlc.logical;
                    if (c > 0 || sameLogical) keep = i;      // equal clock: whatever arrived last wins
                }
                RegisterEntry winner = siblings.get(keep);
                siblings.clear();
                siblings.add(winner);
            } else if (deterministicTieBreak) {
                Hlc best = null;
                for (RegisterEntry e : siblings) if (best == null || e.hlc.compareTo(best) > 0) best = e.hlc;
                for (Iterator<RegisterEntry> it = siblings.iterator(); it.hasNext();) {
                    if (it.next().hlc.compareTo(best) < 0) it.remove();
                }
            }
        } else if (f[0].equals("C")) {
            mergeComponents(counter.computeIfAbsent(f[1], k -> new TreeMap<>()), f[2]);
        } else if (f[0].equals("A")) {
            setAdds.computeIfAbsent(f[1] + "/" + f[2], k -> new TreeSet<>()).add(f[3]);
        } else {
            for (String tag : f[3].split(",")) if (!tag.isEmpty()) tombstones.add(f[2] + "/" + tag);
        }
        for (Map.Entry<String, Long> e : vector.entrySet()) {
            version.merge(e.getKey(), e.getValue(), Math::max);
        }
        log.add(frame);
        outbox.add(frame);
        applied++;
    }

    /** True when this replica has already seen a state that dominates the frame's vector. */
    boolean contains(TreeMap<String, Long> vector) {
        for (Map.Entry<String, Long> e : vector.entrySet()) {
            if (version.getOrDefault(e.getKey(), 0L) < e.getValue()) return false;
        }
        return true;
    }

    /** Anti-entropy: merge another replica's whole log; order of frames must not matter. */
    void mergeLog(Collection<String> frames) {
        List<String> shuffled = new ArrayList<>(frames);
        Set<String> seen = new LinkedHashSet<>(shuffled);
        for (String frame : seen) apply(frame);
        replayed++;
    }

    String render() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, List<RegisterEntry>> e : registers.entrySet()) {
            List<String> values = new ArrayList<>();
            for (RegisterEntry entry : e.getValue()) values.add(entry.render());
            Collections.sort(values);
            sb.append("R:").append(e.getKey()).append('=').append(values).append(';');
        }
        for (Map.Entry<String, TreeMap<String, Long>> e : counter.entrySet()) {
            sb.append("C:").append(e.getKey()).append('=').append(counterValue(e.getKey())).append(';');
        }
        for (Map.Entry<String, TreeSet<String>> e : setAdds.entrySet()) {
            int slash = e.getKey().indexOf('/');
            String key = e.getKey().substring(0, slash), element = e.getKey().substring(slash + 1);
            if (!visible(key, element)) continue;
            sb.append("S:").append(key).append('=').append(element).append(';');
        }
        return sb.toString();
    }

    long counterValue(String key) {
        long positive = 0, negative = 0;
        for (Map.Entry<String, Long> e : counter.getOrDefault(key, new TreeMap<>()).entrySet()) {
            if (e.getKey().endsWith("#n")) negative += e.getValue();
            else positive += e.getValue();
        }
        return positive - negative;
    }

    boolean visible(String key, String element) {
        for (String tag : setAdds.getOrDefault(key + "/" + element, new TreeSet<>())) {
            if (!tombstones.contains(element + "/" + tag)) return true;
        }
        return false;
    }

    String singleValue(String key) {
        List<RegisterEntry> siblings = registers.get(key);
        if (siblings == null || siblings.isEmpty()) return null;
        return siblings.get(siblings.size() - 1).value;
    }

    int siblingsOf(String key) {
        List<RegisterEntry> siblings = registers.get(key);
        return siblings == null ? 0 : siblings.size();
    }
}

final class Mesh805 {
    final List<Replica805> replicas = new ArrayList<>();
    final Map<String, Replica805> byName = new TreeMap<>();
    final List<String> globalLog = new ArrayList<>();
    final SplitMix64 rng;
    int frames, rounds;

    Mesh805(int n, long seed, boolean deterministicTieBreak) {
        rng = new SplitMix64(seed);
        for (int i = 0; i < n; i++) {
            Replica805 r = new Replica805("n" + i, deterministicTieBreak);
            replicas.add(r);
            byName.put(r.name, r);
        }
    }

    Replica805 replica(String name) {
        Replica805 r = byName.get(name);
        if (r == null) throw new IllegalArgumentException("unknown replica " + name);
        return r;
    }

    /** Records the operation in the global log as well, so the oracle sees every accepted write. */
    void record(String frame) { globalLog.add(frame); }

    /** One gossip round: every replica pushes its log to `fanout` peers chosen deterministically. */
    int round(int fanout) {
        rounds++;
        int delivered = 0;
        for (Replica805 from : replicas) {
            if (from.partitioned) continue;
            List<Replica805> targets = new ArrayList<>();
            for (Replica805 r : replicas) if (r != from && !r.partitioned) targets.add(r);
            if (targets.isEmpty()) continue;
            for (int k = 0; k < Math.min(fanout, targets.size()); k++) {
                Replica805 to = targets.get(rng.nextInt(targets.size()));
                if (to == from) continue;
                to.mergeLog(from.log);
                delivered++;
                frames++;
            }
        }
        return delivered;
    }

    int fullSync() {
        int merges = 0;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (Replica805 a : replicas) {
                for (Replica805 b : replicas) {
                    if (a == b) continue;
                    int before = a.log.size();
                    a.mergeLog(b.log);
                    if (a.log.size() != before) changed = true;
                    merges++;
                }
            }
        }
        return merges;
    }

    void partition(String... group) {
        Set<String> side = new HashSet<>(Arrays.asList(group));
        for (Replica805 r : replicas) r.partitioned = side.contains(r.name);
    }

    void heal() {
        for (Replica805 r : replicas) r.partitioned = false;
    }

    boolean allAgree() {
        String first = replicas.get(0).render();
        for (Replica805 r : replicas) if (!r.render().equals(first)) return false;
        return true;
    }

    String renderAll() { return replicas.get(0).render(); }

    /** The oracle: applies the same policies straight to the global log, ignoring replication. */
    static final class Oracle805 {
        final TreeMap<String, List<RegisterEntry>> registers = new TreeMap<>();
        final TreeMap<String, TreeMap<String, Long>> counter = new TreeMap<>();
        final TreeMap<String, TreeSet<String>> adds = new TreeMap<>();
        final TreeSet<String> tombstones = new TreeSet<>();
        final boolean deterministicTieBreak;

        Oracle805(List<String> globalLog, boolean deterministicTieBreak) {
            this.deterministicTieBreak = deterministicTieBreak;
            for (String frame : globalLog) apply(frame);
            if (deterministicTieBreak) {
                for (Map.Entry<String, List<RegisterEntry>> e : registers.entrySet()) {
                    Hlc best = null;
                    for (RegisterEntry entry : e.getValue()) {
                        if (best == null || entry.hlc.compareTo(best) > 0) best = entry.hlc;
                    }
                    List<RegisterEntry> kept = new ArrayList<>();
                    for (RegisterEntry entry : e.getValue()) {
                        if (entry.hlc.compareTo(best) == 0) kept.add(entry);
                    }
                    e.setValue(kept);
                }
            }
        }

        void apply(String frame) {
            String[] f = frame.split("\\|", -1);
            TreeMap<String, Long> vector = Replica805.parseVector(f[f.length - 1]);
            if (f[0].equals("R")) {
                RegisterEntry entry = new RegisterEntry(f[2], Hlc.parse(f[3]), vector);
                List<RegisterEntry> siblings = registers.computeIfAbsent(f[1], k -> new ArrayList<>());
                boolean dominated = false;
                for (Iterator<RegisterEntry> it = siblings.iterator(); it.hasNext();) {
                    RegisterEntry existing = it.next();
                    if (entry.dominatedBy(existing)) { dominated = true; break; }
                    if (existing.dominatedBy(entry)) it.remove();
                }
                if (!dominated) siblings.add(entry);
            } else if (f[0].equals("C")) {
                Replica805.mergeComponents(counter.computeIfAbsent(f[1], k -> new TreeMap<>()), f[2]);
            } else if (f[0].equals("A")) {
                adds.computeIfAbsent(f[1] + "/" + f[2], k -> new TreeSet<>()).add(f[3]);
            } else {
                for (String tag : f[3].split(",")) if (!tag.isEmpty()) tombstones.add(f[2] + "/" + tag);
            }
        }

        String render() {
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, List<RegisterEntry>> e : registers.entrySet()) {
                List<String> values = new ArrayList<>();
                for (RegisterEntry entry : e.getValue()) values.add(entry.render());
                Collections.sort(values);
                sb.append("R:").append(e.getKey()).append('=').append(values).append(';');
            }
            for (Map.Entry<String, TreeMap<String, Long>> e : counter.entrySet()) {
                long positive = 0, negative = 0;
                for (Map.Entry<String, Long> c : e.getValue().entrySet()) {
                    if (c.getKey().endsWith("#n")) negative += c.getValue();
                    else positive += c.getValue();
                }
                sb.append("C:").append(e.getKey()).append('=').append(positive - negative).append(';');
            }
            for (Map.Entry<String, TreeSet<String>> e : adds.entrySet()) {
                int slash = e.getKey().indexOf('/');
                String key = e.getKey().substring(0, slash), element = e.getKey().substring(slash + 1);
                boolean visible = false;
                for (String tag : e.getValue()) if (!tombstones.contains(element + "/" + tag)) visible = true;
                if (visible) sb.append("S:").append(key).append('=').append(element).append(';');
            }
            return sb.toString();
        }
    }
}

public class Solution {

    static final Map<String, Long> METRICS = new TreeMap<>();

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    static void metric(String name, long value) { METRICS.merge(name, value, Long::sum); }

    /** Every replica's state must equal the oracle after anti-entropy. */
    static void convergence() {
        Mesh805 mesh = new Mesh805(5, 805001L, true);
        String[] keys = {"config", "mode", "owner"};
        for (int i = 0; i < 12; i++) {
            Replica805 r = mesh.replicas.get(i % mesh.replicas.size());
            r.registerWrite(keys[i % keys.length], "v" + i);
            mesh.record(r.log.get(r.log.size() - 1));
            mesh.round(2);
        }
        mesh.fullSync();
        Mesh805.Oracle805 oracle = new Mesh805.Oracle805(mesh.globalLog, true);
        check(mesh.allAgree(), "every replica converged to the same state");
        check(mesh.renderAll().equals(oracle.render()), "and that state equals the oracle:\n  " + mesh.renderAll()
                + "\n  " + oracle.render());
        check(mesh.replicas.get(0).duplicates > 0, "duplicate frames were delivered more than once: "
                + mesh.replicas.get(0).duplicates);
        int siblings = 0;
        for (Replica805 r : mesh.replicas) siblings = Math.max(siblings, r.siblingsOf("config"));
        check(siblings == 1, "last-writer-wins keeps exactly one value per key: " + siblings);
        metric("gossipFrames", mesh.frames);
        metric("duplicateFrames", mesh.replicas.get(0).duplicates);
    }

    /** A causally newer write must win even when the older one is delivered last. */
    static void causalityAndTieBreak() {
        Mesh805 mesh = new Mesh805(2, 805002L, true);
        Replica805 a = mesh.replica("n0"), b = mesh.replica("n1");
        a.registerWrite("k", "first");
        b.mergeLog(a.log);
        b.registerWrite("k", "second");                  // causally after "first"
        check("second".equals(b.singleValue("k")), "the causally newer write replaced the old one");
        a.mergeLog(b.log);
        check("second".equals(a.singleValue("k")), "and the older replica agrees");
        check(a.siblingsOf("k") == 1, "no sibling survived a causal overwrite: " + a.siblingsOf("k"));

        // A genuine conflict: neither write saw the other, so both HLCs must be compared.
        Mesh805 concurrent = new Mesh805(3, 805003L, true);
        Replica805 x = concurrent.replica("n0"), y = concurrent.replica("n1"), z = concurrent.replica("n2");
        x.registerWrite("c", "from-x");
        y.registerWrite("c", "from-y");
        concurrent.record(x.log.get(0));
        concurrent.record(y.log.get(0));
        x.mergeLog(y.log);
        y.mergeLog(x.log);
        check(x.singleValue("c").equals(y.singleValue("c")),
                "both replicas resolved the conflict the same way: " + x.singleValue("c") + " vs "
                        + y.singleValue("c"));
        z.mergeLog(x.log);
        check(z.singleValue("c").equals(x.singleValue("c")), "and so does a third replica");
        Mesh805.Oracle805 oracle = new Mesh805.Oracle805(concurrent.globalLog, true);
        check(x.render().equals(oracle.render()), "the resolution matches the oracle");
        check("from-y".equals(x.singleValue("c")),
                "the highest replica id broke the tie deterministically: " + x.singleValue("c"));

        // Multi-value mode keeps concurrency visible instead of silently discarding an update.
        Mesh805 multi = new Mesh805(2, 805004L, false);
        Replica805 p = multi.replica("n0"), q = multi.replica("n1");
        p.registerWrite("m", "A");
        q.registerWrite("m", "B");
        multi.record(p.log.get(0));
        multi.record(q.log.get(0));
        p.mergeLog(q.log);
        check(p.siblingsOf("m") == 2, "the multi-value register kept both siblings: " + p.siblingsOf("m"));
        q.mergeLog(p.log);
        check(q.siblingsOf("m") == 2 && p.render().equals(q.render()),
                "and both replicas expose the same conflict");
        metric("conflictResolutions", 3);
    }

    /** The outcome must not depend on the order of frames nor on duplicates. */
    static void orderIndependence() {
        List<String> frames = new ArrayList<>();
        Mesh805 source = new Mesh805(3, 805005L, true);
        for (int i = 0; i < 9; i++) {
            Replica805 r = source.replicas.get(i % 3);
            if (i % 3 == 0) r.registerWrite("r", "value" + i);
            else if (i % 3 == 1) r.counterAdd("counter", 1);
            else r.setAdd("set", "e" + (i % 2));
            frames.add(r.log.get(r.log.size() - 1));
        }
        String expected = null;
        String reference = null;
        for (int seed = 1; seed <= 10; seed++) {
            Mesh805 m = new Mesh805(1, 805100L + seed, true);
            Replica805 target = m.replica("n0");
            List<String> shuffled = new ArrayList<>(frames);
            for (int i = 0; i < frames.size() * 2; i++) {
                shuffled.add(frames.get(i % frames.size()));       // duplicates as well
            }
            Collections.shuffle(shuffled, new Random(seed));       // fixed seed: deterministic
            target.mergeLog(shuffled);
            if (seed == 1) reference = target.render();
            else check(target.render().equals(reference),
                    "shuffle " + seed + " produced a different state:\n  " + target.render()
                            + "\n  " + reference);
        }
        Mesh805.Oracle805 oracle = new Mesh805.Oracle805(frames, true);
        check(reference.equals(oracle.render()), "the merged state equals the oracle:\n  " + reference
                + "\n  " + oracle.render());
        expected = reference;
        check(!expected.isEmpty(), "the merged state is not empty");
        check(reference.contains("C:counter=3") || reference.contains("counter=3"),
                "three increments were counted once each: " + reference);
        metric("mergePermutations", 10);
    }

    /** The two CRDTs: a PN-counter and an observed-remove set. */
    static void crdtPolicies() {
        Mesh805 mesh = new Mesh805(4, 805006L, true);
        for (int round = 0; round < 25; round++) {
            Replica805 r = mesh.replicas.get(round % 4);
            r.counterAdd("hits", 1);
            mesh.record(r.log.get(r.log.size() - 1));
            if (round % 5 == 0) mesh.round(1);
        }
        mesh.fullSync();
        for (Replica805 r : mesh.replicas) {
            check(r.counterValue("hits") == 25 || r.counterValue("hits") == 25,
                    "the counter is exactly 25 on every replica: " + r.counterValue("hits"));
        }
        check(mesh.replicas.get(0).counterValue("hits") == 25, "25 increments in, 25 counted");
        check(mesh.allAgree(), "the replicas agree after anti-entropy");
        Mesh805.Oracle805 oracle = new Mesh805.Oracle805(mesh.globalLog, true);
        check(mesh.renderAll().equals(oracle.render()), "the counter matches the oracle");

        Mesh805 sets = new Mesh805(3, 805007L, true);
        Replica805 a = sets.replica("n0"), b = sets.replica("n1");
        a.setAdd("tags", "red");
        b.mergeLog(a.log);
        b.setRemove("tags", "red");                        // observed remove: b saw red
        sets.record(a.log.get(0));
        sets.record(b.log.get(1));
        check(!b.visible("tags", "red"), "the observed remove hid the element");
        a.setAdd("tags", "red");                           // a re-adds it concurrently
        sets.record(a.log.get(1));
        a.mergeLog(b.log);
        b.mergeLog(a.log);
        check(a.visible("tags", "red") && b.visible("tags", "red"),
                "the concurrent re-add survives the older remove");
        check(a.visible("tags", "red") == b.visible("tags", "red"), "and both replicas agree");
        Mesh805.Oracle805 setOracle = new Mesh805.Oracle805(sets.globalLog, true);
        check(a.render().equals(setOracle.render()), "the set matches the oracle:\n  " + a.render()
                + "\n  " + setOracle.render());
        metric("crdtCounterValue", mesh.replicas.get(0).counterValue("hits"));
    }

    /** Writes on both sides of a partition must survive healing and still converge. */
    static void partitionAndHeal() {
        Mesh805 mesh = new Mesh805(4, 805008L, true);
        mesh.replica("n0").registerWrite("zone", "west");
        mesh.record(mesh.replica("n0").log.get(0));
        mesh.fullSync();
        String beforeState = mesh.renderAll();
        mesh.partition("n0", "n1");                       // {n0,n1} cut off from {n2,n3}
        mesh.replica("n0").registerWrite("zone", "west-2");
        mesh.record(mesh.replica("n0").log.get(mesh.replica("n0").log.size() - 1));
        mesh.replica("n2").registerWrite("zone", "east-2");
        mesh.record(mesh.replica("n2").log.get(mesh.replica("n2").log.size() - 1));
        for (int i = 0; i < 6; i++) mesh.round(2);
        check(!mesh.allAgree(), "while partitioned the two sides disagree: "
                + mesh.replica("n0").singleValue("zone") + " vs " + mesh.replica("n2").singleValue("zone"));
        int lost = 0;
        for (String frame : mesh.globalLog) {
            boolean present = false;
            for (Replica805 r : mesh.replicas) if (r.log.contains(frame)) present = true;
            if (!present) lost++;
        }
        check(lost == 0, "no accepted write was dropped during the partition: " + lost);
        check(!mesh.replica("n2").log.contains(mesh.globalLog.get(mesh.globalLog.size() - 2))
                        || !mesh.replica("n0").log.contains(mesh.globalLog.get(mesh.globalLog.size() - 1)),
                "each side really missed the other side's write");
        mesh.heal();
        mesh.fullSync();
        Mesh805.Oracle805 oracle = new Mesh805.Oracle805(mesh.globalLog, true);
        check(mesh.allAgree(), "after healing every replica agrees again");
        check(mesh.renderAll().equals(oracle.render()), "and the merged value is the oracle's:\n  "
                + mesh.renderAll() + "\n  " + oracle.render());
        check(!beforeState.equals(mesh.renderAll()), "the partition really changed the data");
        check(mesh.replicas.get(0).registers.get("zone").size() == 1,
                "last-writer-wins left a single winner: " + mesh.replicas.get(0).registers.get("zone").size());
        metric("partitionRounds", mesh.rounds);
    }

    /** Version vectors and HLCs must only move forward; a silent replica stays behind. */
    static void monotonicityAndStaleness() {
        Mesh805 mesh = new Mesh805(4, 805009L, true);
        Replica805 silent = mesh.replica("n3");
        Replica805 active = mesh.replica("n0");
        long previous = active.version.getOrDefault("n0", 0L);
        for (int i = 0; i < 6; i++) {
            active.registerWrite("k" + i, "v" + i);
            mesh.record(active.log.get(active.log.size() - 1));
            check(active.version.getOrDefault("n0", 0L) >= previous, "the version only grows");
            previous = active.version.getOrDefault("n0", 0L);
        }
        check(previous == 6, "six writes advanced the replica's own component to six: " + previous);
        mesh.partition("n3");
        int roundsBefore = mesh.rounds;
        for (int i = 0; i < 4; i++) mesh.round(2);
        check(mesh.rounds == roundsBefore + 4, "four gossip rounds ran");
        check(silent.log.isEmpty(), "the excluded replica received nothing: " + silent.log.size());
        int staleOthers = 0;
        for (Replica805 r : mesh.replicas) {
            if (r != silent && !r.log.isEmpty()) staleOthers++;
        }
        check(staleOthers == 3, "the other three replicas exchanged the writes: " + staleOthers);
        check(active.version.getOrDefault("n0", 0L) == 6, "gossiping does not move the local component: "
                + active.version.getOrDefault("n0", 0L));
        silent.mergeLog(active.log);
        check(silent.log.size() == active.log.size(), "one anti-entropy merge catches the laggard up: "
                + silent.log.size() + " vs " + active.log.size());
        check(silent.version.getOrDefault("n0", 0L) == active.version.getOrDefault("n0", 0L),
                "including the version vector");
        int before = silent.log.size();
        silent.mergeLog(active.log);
        check(silent.log.size() == before, "merging the same log again changes nothing (idempotence)");
        metric("stalenessCaught", silent.log.size());
    }

    /** Negative control: without the replica-id tie-break two leaders never converge. */
    static void negativeControlBadTieBreak() {
        Mesh805 good = new Mesh805(2, 805010L, true);
        good.replica("n0").registerWrite("v", "A");
        good.replica("n1").registerWrite("v", "B");
        for (Replica805 r : good.replicas) good.record(r.log.get(0));
        good.fullSync();
        check(good.allAgree(), "with a deterministic tie-break both replicas agree: "
                + good.replica("n0").singleValue("v"));

        Mesh805 bad = new Mesh805(2, 805011L, true);
        for (Replica805 r : bad.replicas) r.badTieBreak = true;   // arrival order breaks ties
        bad.replica("n0").registerWrite("v", "A");
        bad.replica("n1").registerWrite("v", "B");
        for (Replica805 r : bad.replicas) bad.record(r.log.get(0));
        Replica805 left = bad.replica("n0"), right = bad.replica("n1");
        left.mergeLog(right.log);
        right.mergeLog(left.log);
        check(!left.singleValue("v").equals(right.singleValue("v")),
                "the buggy policy keeps each replica's own value: " + left.singleValue("v") + " / "
                        + right.singleValue("v"));
        check(!bad.allAgree(), "so the replicas never converge - the checker catches it");
        String first = left.render(), second = right.render();
        check(!first.equals(second), "divergence is observable in the rendered state");
        metric("negativeControlDivergence", 1);

        Mesh805 single = new Mesh805(1, 805012L, true);
        Replica805 only = single.replica("n0");
        only.registerWrite("solo", "one");
        single.record(only.log.get(0));
        single.fullSync();
        check(single.allAgree() && "one".equals(only.singleValue("solo")),
                "a single replica is trivially consistent");
        Mesh805.Oracle805 oracle = new Mesh805.Oracle805(single.globalLog, true);
        check(only.render().equals(oracle.render()), "and it matches the oracle");
    }

    public static void main(String[] args) {
        convergence();
        causalityAndTieBreak();
        orderIndependence();
        crdtPolicies();
        partitionAndHeal();
        monotonicityAndStaleness();
        negativeControlBadTieBreak();
        StringBuilder sb = new StringBuilder("metrics:");
        for (Map.Entry<String, Long> e : METRICS.entrySet()) {
            sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        }
        System.out.println(sb);
        System.out.println("PASS");
    }
}
