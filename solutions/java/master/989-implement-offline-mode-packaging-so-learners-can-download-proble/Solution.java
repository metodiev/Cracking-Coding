/** Task 989 - Implement offline mode packaging so learners can download problem sets + tests.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Offline packaging has three duties. (1) Closure: an asset (problem, test harness, fixture) declares
 *  the assets it requires; the packer takes the transitive closure over those edges so a bundle is
 *  self-contained - after every insertion it re-checks that every requirement is inside the bundle or
 *  still queued, and cycles terminate via the visited set. (2) Budget and priority: the bundle must fit
 *  a byte budget, so seeds are considered in (priority desc, id asc) order and a seed whose closure no
 *  longer fits is skipped and reported rather than half-included. (3) Integrity and delta: the packer
 *  records an FNV-1a checksum per asset plus the byte total, verification recomputes both against the
 *  current assets (edits, deletions and size changes are caught), and an incremental update is a delta
 *  (added/changed/removed) that, applied to the old manifest, reproduces the new manifest exactly while
 *  transferring fewer bytes than a full download.
 *  Simplified: assets live in memory with synthetic sizes and a fixed priority; closure, budget,
 *  checksum and delta arithmetic are the real ones.
 *  Verified: closure completeness re-checked after every insertion, priority-ordered budget packing with
 *  a reported skip, checksum/size/vanished tamper detection, delta application as a fixed point plus its
 *  transfer advantage, the empty and unchanged boundaries, cyclic requirements, and an independent
 *  fixed-point closure oracle.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static String fnv(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= 0x100000001b3L; }
        return String.format("%016x", h);
    }

    static final class Asset {
        final String id, content;
        final int size, priority;
        final List<String> requires;
        Asset(String id, int size, int priority, String content, List<String> requires) {
            this.id = id; this.size = size; this.priority = priority; this.content = content; this.requires = requires;
        }
        String checksum() { return fnv(content); }
    }

    static Map<String, Asset> index(List<Asset> assets) {
        Map<String, Asset> byId = new TreeMap<>();
        for (Asset a : assets) byId.put(a.id, a);
        return byId;
    }

    /**
     * Transitive closure of `seeds` over `requires`, or null for an unknown asset id. The closure
     * invariant - every requirement of every member is a member or still queued - is re-checked after
     * every insertion, so a broken closure cannot escape the packer.
     */
    static Set<String> closure(Collection<String> seeds, Map<String, Asset> byId) {
        Set<String> included = new TreeSet<>();
        Deque<String> queue = new ArrayDeque<>(new TreeSet<>(seeds));
        while (!queue.isEmpty()) {
            String id = queue.poll();
            Asset a = byId.get(id);
            if (a == null) return null;
            if (!included.add(id)) continue;
            for (String req : a.requires) if (!included.contains(req)) queue.add(req);
            for (String member : included)
                for (String req : byId.get(member).requires)
                    check(included.contains(req) || queue.contains(req), "closure invariant broken: " + member + " requires " + req);
        }
        return included;
    }

    static int bytesOf(Collection<String> ids, Map<String, Asset> byId) {
        int total = 0;
        for (String id : ids) total += byId.get(id).size;
        return total;
    }

    static final class Bundle {
        final Set<String> ids;
        final List<String> skipped;
        final int bytes;
        final String error;
        final Map<String, String> manifest;      // id -> checksum, recorded at pack time
        Bundle(Set<String> ids, List<String> skipped, int bytes, String error, Map<String, String> manifest) {
            this.ids = ids; this.skipped = skipped; this.bytes = bytes; this.error = error; this.manifest = manifest;
        }
        boolean ok() { return error == null; }
    }

    /** Packs seeds in (priority desc, id asc) order while the running closure stays inside `budget`. */
    static Bundle pack(List<Asset> assets, List<String> seeds, int budget) {
        Map<String, Asset> byId = index(assets);
        List<String> order = new ArrayList<>(new TreeSet<>(seeds));
        for (String seed : order)
            if (!byId.containsKey(seed)) return new Bundle(null, null, 0, "unknown asset: " + seed, null);
        order.sort((a, b) -> byId.get(a).priority != byId.get(b).priority
                ? Integer.compare(byId.get(b).priority, byId.get(a).priority) : a.compareTo(b));
        Set<String> chosen = new TreeSet<>();
        List<String> skipped = new ArrayList<>();
        for (String seed : order) {
            Set<String> candidate = new TreeSet<>(chosen);
            candidate.add(seed);
            candidate = closure(candidate, byId);
            if (bytesOf(candidate, byId) <= budget) chosen = candidate;
            else skipped.add(seed);
        }
        for (String id : chosen)
            for (String req : byId.get(id).requires)
                check(chosen.contains(req) && byId.containsKey(req), "packed bundle must stay closed after " + id + " (missing " + req + ")");
        Map<String, String> manifest = new TreeMap<>();
        for (String id : chosen) manifest.put(id, byId.get(id).checksum());
        return new Bundle(chosen, skipped, bytesOf(chosen, byId), null, manifest);
    }

    /** Recomputes every recorded checksum and the byte total against the assets as they are now. */
    static String verify(Bundle bundle, List<Asset> assets) {
        if (!bundle.ok()) return "bundle not packed";
        Map<String, Asset> byId = index(assets);
        int total = 0;
        for (Map.Entry<String, String> entry : bundle.manifest.entrySet()) {
            Asset a = byId.get(entry.getKey());
            if (a == null) return "asset vanished: " + entry.getKey();
            if (!a.checksum().equals(entry.getValue())) return "checksum mismatch for " + entry.getKey();
            total += a.size;
        }
        return total == bundle.bytes ? "ok" : "size mismatch";
    }

    static final class Delta {
        final List<String> added = new ArrayList<>(), changed = new ArrayList<>(), removed = new ArrayList<>();
        int transferBytes;
        /** Rebuilds the new manifest from the old one plus this delta; true when reconstruction is exact. */
        boolean apply(Map<String, String> oldManifest, Map<String, String> target, Map<String, String> newChecksums) {
            Map<String, String> rebuilt = new TreeMap<>(oldManifest);
            for (String id : removed) rebuilt.remove(id);
            for (String id : added) rebuilt.put(id, newChecksums.get(id));
            for (String id : changed) rebuilt.put(id, newChecksums.get(id));
            return rebuilt.equals(target);
        }
    }

    /** Incremental update plan: only added and changed assets travel, removals are instructions. */
    static Delta diff(Map<String, String> oldManifest, Map<String, String> newManifest, Map<String, Asset> byId) {
        Delta delta = new Delta();
        for (Map.Entry<String, String> e : newManifest.entrySet()) {
            String before = oldManifest.get(e.getKey());
            if (before == null) { delta.added.add(e.getKey()); delta.transferBytes += byId.get(e.getKey()).size; }
            else if (!before.equals(e.getValue())) {
                delta.changed.add(e.getKey());
                delta.transferBytes += byId.get(e.getKey()).size;
            }
        }
        for (String id : oldManifest.keySet()) if (!newManifest.containsKey(id)) delta.removed.add(id);
        return delta;
    }

    static List<Asset> catalog() {
        return List.of(
            new Asset("fixtures", 400, 3, "fixtures-v1", List.of()),
            new Asset("harness", 900, 3, "harness-v1", List.of("fixtures")),
            new Asset("tests-two-sum", 1500, 2, "tests-two-sum-v1", List.of("harness")),
            new Asset("problem-two-sum", 3000, 2, "two-sum-v1", List.of("fixtures")),
            new Asset("problem-bfs", 3200, 1, "bfs-v1", List.of("harness")),
            new Asset("solutions-bfs", 2600, 1, "bfs-solution-v1", List.of("problem-bfs")));
    }

    public static void main(String[] args) {
        List<Asset> assets = catalog();
        Map<String, Asset> byId = index(assets);
        // --- happy path: the bundle is closed, budgeted and self-contained.
        Bundle full = pack(assets, List.of("problem-two-sum", "problem-bfs"), 1 << 20);
        check(full.ok() && full.skipped.isEmpty() && full.bytes == 400 + 900 + 3000 + 3200, "an unconstrained bundle must pull the whole closure: " + full.bytes + " bytes");
        check(full.ids.equals(new TreeSet<>(List.of("fixtures", "harness", "problem-two-sum", "problem-bfs"))),
                "the closure must include every transitive requirement: " + full.ids);
        check(verify(full, assets).equals("ok"), "a packed bundle must verify clean: " + verify(full, assets));
        for (String id : full.ids)
            for (String req : byId.get(id).requires)
                check(full.ids.contains(req), "offline invariant: " + id + " requires " + req + " which is absent");
        // --- budget: the low-priority seed whose closure no longer fits is skipped and reported.
        Bundle tight = pack(assets, List.of("problem-two-sum", "problem-bfs"), 400 + 3000);
        check(tight.ok() && tight.bytes == 400 + 3000 && tight.skipped.equals(List.of("problem-bfs")),
                "the over-budget seed must be skipped, never half-included: " + tight.skipped + " / " + tight.bytes);
        check(tight.ids.equals(new TreeSet<>(List.of("fixtures", "problem-two-sum"))) && verify(tight, assets).equals("ok") && tight.bytes < full.bytes, "the surviving seed keeps its full closure and verifies exactly at its budget: " + tight.ids);
        // --- priority beats alphabetical order: the priority-3 closure is admitted, the other is skipped.
        Bundle priority = pack(assets, List.of("solutions-bfs", "tests-two-sum"), 400 + 900 + 1500);
        check(priority.skipped.equals(List.of("solutions-bfs")) && priority.ids.equals(new TreeSet<>(List.of("fixtures", "harness", "tests-two-sum"))),
                "the higher-priority closure must be admitted first: " + priority.ids + " skipped " + priority.skipped);
        // --- tamper detection: edited content, a vanished asset and a rewritten byte count.
        List<Asset> tampered = new ArrayList<>(assets);
        tampered.set(3, new Asset("problem-two-sum", 3000, 2, "two-sum-v2", List.of("fixtures")));
        List<Asset> shrunk = new ArrayList<>();
        for (Asset a : assets) if (!a.id.equals("problem-bfs")) shrunk.add(a);
        List<Asset> resized = new ArrayList<>(assets);
        resized.set(0, new Asset("fixtures", 401, 3, "fixtures-v1", List.of()));
        check(verify(full, tampered).equals("checksum mismatch for problem-two-sum"), "edited content must break its recorded checksum: " + verify(full, tampered));
        check(verify(full, shrunk).equals("asset vanished: problem-bfs"), "a deleted asset must be reported by name: " + verify(full, shrunk));
        check(verify(full, resized).equals("size mismatch"), "a size change with identical content must be caught by the byte total: " + verify(full, resized));
        // --- incremental update: added/changed/removed, and applying the delta is a fixed point.
        List<Asset> updated = new ArrayList<>();
        for (Asset a : tampered) if (!a.id.equals("problem-bfs")) updated.add(a);
        updated.add(new Asset("problem-dp", 2000, 2, "dp-v1", List.of("fixtures")));
        Bundle rebuilt = pack(updated, List.of("problem-two-sum", "problem-dp"), 1 << 20);
        Delta delta = diff(full.manifest, rebuilt.manifest, index(updated));
        check(delta.added.equals(List.of("problem-dp")) && delta.changed.equals(List.of("problem-two-sum")) && delta.removed.equals(List.of("harness", "problem-bfs")),
                "the delta must classify one add, one change and two removals: " + delta.added + "/" + delta.changed + "/" + delta.removed);
        Map<String, String> newChecksums = new TreeMap<>();
        for (Asset a : updated) newChecksums.put(a.id, a.checksum());
        check(delta.apply(full.manifest, rebuilt.manifest, newChecksums), "applying the delta to the old manifest must reproduce the new manifest exactly");
        check(delta.transferBytes == 2000 + 3000 && delta.transferBytes < full.bytes, "an incremental update transfers only the add and the change: " + delta.transferBytes + " of " + full.bytes);
        // --- boundary: identical bundles produce an empty delta, and no seeds an empty bundle.
        Delta none = diff(full.manifest, full.manifest, byId);
        check(none.added.isEmpty() && none.changed.isEmpty() && none.removed.isEmpty() && none.transferBytes == 0, "an unchanged bundle must produce an empty delta: " + none.added + "/" + none.changed + "/" + none.removed);
        Bundle empty = pack(assets, List.of(), 0);
        check(empty.ok() && empty.ids.isEmpty() && empty.bytes == 0 && empty.manifest.isEmpty() && verify(empty, assets).equals("ok") && pack(assets, List.of("ghost"), 1 << 20) .error.equals("unknown asset: ghost"),
                "an empty bundle must verify and an unknown seed must fail closed");
        // --- cyclic requirements must terminate and still close over the cycle.
        List<Asset> cyclic = List.of(new Asset("a", 10, 1, "a", List.of("b")), new Asset("b", 20, 1, "b", List.of("a")));
        Bundle cycleBundle = pack(cyclic, List.of("a"), 100);
        check(cycleBundle.ok() && cycleBundle.ids.equals(new TreeSet<>(List.of("a", "b"))) && cycleBundle.bytes == 30, "a cyclic requirement must terminate with both members: " + cycleBundle.ids);
        // --- independent oracle: fixed-point closure must agree with the BFS closure.
        for (List<String> seeds : List.of(List.of("problem-bfs"), List.of("tests-two-sum"), List.of("fixtures"),
                List.of("problem-two-sum", "solutions-bfs")))
            check(new TreeSet<>(closure(seeds, byId)).equals(oracleClosure(seeds, byId)), "closure oracle mismatch for " + seeds);

        System.out.println("full bundle: " + full.ids.size() + " assets, " + full.bytes + " bytes; budgeted: "
                + tight.bytes + " bytes, skipped " + tight.skipped);
        System.out.println("PASS");
    }

    /** Oracle: naive fixed-point closure - repeat "add every requirement of every member" until stable. */
    static Set<String> oracleClosure(Collection<String> seeds, Map<String, Asset> byId) {
        Set<String> members = new TreeSet<>(seeds);
        boolean changed = true;
        while (changed) {
            changed = false;
            for (String id : new ArrayList<>(members)) {
                Asset a = byId.get(id);
                if (a == null) return null;
                for (String req : a.requires) if (members.add(req)) changed = true;
            }
        }
        return members;
    }
}
