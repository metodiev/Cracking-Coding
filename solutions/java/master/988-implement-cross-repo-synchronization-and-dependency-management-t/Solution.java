/** Task 988 - Implement cross-repo synchronization and dependency management tool for exercises.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Two coupled jobs. (1) Sync order: repos depend on repos, so a Kahn topological sort with a
 *  name-ordered ready queue produces a deterministic build order, and a leftover node means a cycle,
 *  which is reported (with the members) instead of being silently ordered. (2) Version management:
 *  every repo pins library constraints (^1.4.0, ~1.4.0, >=1.1.0, =1.2.0, exact); the resolver
 *  intersects all constraints for a library over a published catalog and picks the highest satisfying
 *  version, which is the "sync drift" fix - after a sync every repo agrees on one version per library.
 *  Unpinned ("*") constraints, empty intersections and cycles fail closed with a diagnostic, so no
 *  partially pinned plan is ever produced.
 *  Simplified: in-memory repos and a three-field semver instead of git remotes; the order constraints,
 *  the intersection arithmetic and the fail-closed behaviour are the real ones.
 *  Verified: the order respects every dependency edge and matches an independent min-selection oracle,
 *  two-repo and self cycles are detected, drift collapses to a single resolved version, impossible and
 *  unpinned constraints are rejected, the empty repo is a legal boundary, and the "dependencies before
 *  dependents" invariant is re-checked after every applied step.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Repo {
        final String name;
        final List<String> deps;
        final Map<String, String> pins;
        Repo(String name, List<String> deps, Map<String, String> pins) {
            this.name = name; this.deps = deps; this.pins = pins;
        }
    }

    static int[] parse(String v) {
        String[] parts = v.trim().split("\\.");
        int[] out = new int[3];
        for (int i = 0; i < 3; i++) out[i] = i < parts.length ? Integer.parseInt(parts[i]) : 0;
        return out;
    }

    static int compare(int[] a, int[] b) {
        for (int i = 0; i < 3; i++) if (a[i] != b[i]) return Integer.compare(a[i], b[i]);
        return 0;
    }

    static boolean satisfies(String constraint, int[] version) {
        String c = constraint.trim();
        if (c.isEmpty() || c.equals("*")) return true;                    // caller decides about unpinned
        if (c.startsWith("^")) {
            int[] base = parse(c.substring(1));
            return version[0] == base[0] && compare(version, base) >= 0;
        }
        if (c.startsWith("~")) {
            int[] base = parse(c.substring(1));
            return version[0] == base[0] && version[1] == base[1] && compare(version, base) >= 0;
        }
        if (c.startsWith(">=")) return compare(version, parse(c.substring(2))) >= 0;
        if (c.startsWith("=")) return compare(version, parse(c.substring(1))) == 0;
        return compare(version, parse(c)) == 0;
    }

    static final class Order {
        final List<String> order;
        final String error;
        Order(List<String> order, String error) { this.order = order; this.error = error; }
    }

    /** Kahn sort with an alphabetically ordered ready queue: deterministic and cycle aware. */
    static Order syncOrder(List<Repo> repos) {
        Map<String, Integer> indegree = new TreeMap<>();
        Map<String, List<String>> dependents = new TreeMap<>();
        for (Repo r : repos) { indegree.putIfAbsent(r.name, 0); dependents.putIfAbsent(r.name, new ArrayList<>()); }
        for (Repo r : repos)
            for (String dep : r.deps) {
                if (!indegree.containsKey(dep)) return new Order(null, "unknown dependency: " + dep + " of " + r.name);
                indegree.merge(r.name, 1, Integer::sum);
                dependents.get(dep).add(r.name);
            }
        PriorityQueue<String> ready = new PriorityQueue<>();
        for (Map.Entry<String, Integer> e : indegree.entrySet()) if (e.getValue() == 0) ready.add(e.getKey());
        List<String> order = new ArrayList<>();
        while (!ready.isEmpty()) {
            String current = ready.poll();
            order.add(current);
            for (String next : dependents.get(current)) if (indegree.merge(next, -1, Integer::sum) == 0) ready.add(next);
        }
        if (order.size() != indegree.size()) {
            List<String> stuck = new ArrayList<>();
            for (Map.Entry<String, Integer> e : indegree.entrySet()) if (e.getValue() > 0 && !order.contains(e.getKey())) stuck.add(e.getKey());
            return new Order(null, "dependency cycle among " + stuck);
        }
        return new Order(order, null);
    }

    static final class Resolution {
        final Map<String, String> versions;      // library -> resolved version
        final Map<String, List<String>> drift;   // library -> repos that must be re-pinned
        final String error;
        Resolution(Map<String, String> versions, Map<String, List<String>> drift, String error) {
            this.versions = versions; this.drift = drift; this.error = error;
        }
        boolean ok() { return error == null; }
    }

    /** Intersects every repo's constraint for each library and picks the highest catalog version left. */
    static Resolution resolve(List<Repo> repos, Map<String, List<String>> catalog) {
        Map<String, Map<String, String>> wanted = new TreeMap<>();
        for (Repo r : repos)
            for (Map.Entry<String, String> pin : r.pins.entrySet())
                wanted.computeIfAbsent(pin.getKey(), k -> new TreeMap<>()).put(r.name, pin.getValue());
        Map<String, String> chosen = new TreeMap<>();
        Map<String, List<String>> drift = new TreeMap<>();
        for (Map.Entry<String, Map<String, String>> lib : wanted.entrySet()) {
            List<String> available = catalog.get(lib.getKey());
            if (available == null) return new Resolution(null, null, "unknown library: " + lib.getKey());
            for (Map.Entry<String, String> pin : lib.getValue().entrySet()) {
                String c = pin.getValue().trim();
                if (c.isEmpty() || c.equals("*"))
                    return new Resolution(null, null, "unpinned dependency in " + pin.getKey() + ": " + lib.getKey());
            }
            String best = null;
            for (String candidate : available) {
                int[] version = parse(candidate);
                boolean fits = true;
                for (String constraint : lib.getValue().values()) if (!satisfies(constraint, version)) { fits = false; break; }
                if (fits && (best == null || compare(parse(candidate), parse(best)) > 0)) best = candidate;
            }
            if (best == null)
                return new Resolution(null, null, "no version of " + lib.getKey() + " satisfies "
                        + new TreeSet<>(lib.getValue().values()));
            chosen.put(lib.getKey(), best);
            List<String> mismatched = new ArrayList<>();
            for (Map.Entry<String, String> pin : lib.getValue().entrySet())
                if (!pin.getValue().equals(best) && !pin.getValue().equals("=" + best)) mismatched.add(pin.getKey());
            if (!mismatched.isEmpty()) drift.put(lib.getKey(), mismatched);   // repos to re-pin after the sync
        }
        return new Resolution(chosen, drift, null);
    }

    /** Applies the plan step by step, re-checking that every dependency is already synchronised. */
    static List<String> applyPlan(List<Repo> repos, Order order, Map<String, String> versions) {
        Set<String> synced = new LinkedHashSet<>();
        List<String> log = new ArrayList<>();
        for (String name : order.order) {
            Repo repo = null;
            for (Repo r : repos) if (r.name.equals(name)) repo = r;
            for (String dep : repo.deps)
                check(synced.contains(dep), "invariant: " + name + " synced before its dependency " + dep);
            synced.add(name);
            StringBuilder sb = new StringBuilder("synced " + name + " ->");
            for (Map.Entry<String, String> pin : new TreeMap<>(repo.pins).entrySet())
                sb.append(' ').append(pin.getKey()).append('@').append(versions.get(pin.getKey()));
            log.add(sb.toString());
        }
        return log;
    }

    static List<Repo> fleet() {
        return List.of(
            new Repo("kernel", List.of(), Map.of("json", "^2.0.0")),
            new Repo("runtime", List.of("kernel"), Map.of("cli", ">=1.1.0")),
            new Repo("compiler", List.of("kernel"), Map.of()),
            new Repo("stdlib", List.of("runtime"), Map.of()),
            new Repo("web", List.of("stdlib", "compiler"), Map.of("http", "^1.4.0", "json", "^2.0.0")),
            new Repo("tools", List.of("compiler"), Map.of("http", "~1.4.0", "cli", "=1.2.0")),
            new Repo("docs", List.of(), Map.of()));
    }

    static Map<String, List<String>> catalog() {
        return Map.of("http", List.of("1.4.0", "1.4.2", "1.5.0", "2.0.0"),
                      "json", List.of("1.0.0", "2.0.0", "2.1.0"),
                      "cli", List.of("1.0.0", "1.2.0", "1.3.0"));
    }

    public static void main(String[] args) {
        List<Repo> fleet = fleet();
        Order order = syncOrder(fleet);
        check(order.error == null && order.order.size() == 7, "a cycle-free fleet must order every repo: " + order.error);
        Map<String, Integer> position = new HashMap<>();
        for (int i = 0; i < order.order.size(); i++) position.put(order.order.get(i), i);
        boolean respect = true;
        for (Repo r : fleet) for (String dep : r.deps) respect &= position.get(dep) < position.get(r.name);
        check(respect, "every dependency must precede its dependent: " + order.order);
        check(order.order.equals(List.of("docs", "kernel", "compiler", "runtime", "stdlib", "tools", "web")),
                "the ready queue must be alphabetical for determinism: " + order.order);
        // --- independent oracle: a linear min-selection sort over the same edges must agree exactly.
        check(oracleOrder(fleet).equals(order.order), "min-selection oracle order mismatch: " + oracleOrder(fleet));
        // --- cycle detection: mutual and self dependencies must be named, never silently ordered.
        Order mutual = syncOrder(List.of(new Repo("a", List.of("b"), Map.of()), new Repo("b", List.of("a"), Map.of())));
        Order selfCycle = syncOrder(List.of(new Repo("s", List.of("s"), Map.of())));
        check(mutual.order == null && mutual.error.contains("[a, b]"), "a mutual cycle must be reported: " + mutual.error);
        check(selfCycle.order == null && selfCycle.error.contains("[s]"), "a self dependency must be reported: " + selfCycle.error);
        // --- drift: two different constraints resolve to one version and both repos are flagged.
        Resolution resolution = resolve(fleet, catalog());
        check(resolution.ok(), "the fleet must resolve cleanly: " + resolution.error);
        check(resolution.versions.equals(Map.of("cli", "1.2.0", "http", "1.4.2", "json", "2.1.0")),
                "the highest satisfying version must win for every library: " + resolution.versions);
        check(resolution.drift.equals(Map.of("cli", List.of("runtime"), "http", List.of("tools", "web"), "json", List.of("kernel", "web"))),
                "every repo whose pin is not the resolved version must be flagged for re-pinning: " + resolution.drift);
        check(resolution.drift.get("http").size() == 2 && satisfies("~1.4.0", parse("1.4.2")) && !satisfies("~1.4.0", parse("1.5.0")),
                "tilde must stay inside the minor line: " + resolution.versions.get("http"));
        check(!satisfies("^1.4.0", parse("2.0.0")) && satisfies("^1.4.0", parse("1.9.9")) && satisfies(">=1.1.0", parse("3.0.0")) && !satisfies("=1.2.0", parse("1.2.1")),
                "caret/at-least/exact constraints must behave per semver");
        // --- fail-closed paths: impossible intersections, unpinned constraints, unknown libraries.
        List<Repo> impossible = List.of(new Repo("x", List.of(), Map.of("json", "^1.0.0")),
                new Repo("y", List.of(), Map.of("json", "^2.0.0")));
        Resolution clash = resolve(impossible, catalog());
        check(!clash.ok() && clash.error.contains("no version of json") && clash.error.contains("[^1.0.0, ^2.0.0]"),
                "an empty intersection must fail closed: " + clash.error);
        Resolution unpinned = resolve(List.of(new Repo("z", List.of(), Map.of("http", "*"))), catalog());
        check(!unpinned.ok() && unpinned.error.equals("unpinned dependency in z: http"), "an unpinned dependency must be refused: " + unpinned.error);
        Resolution unknown = resolve(List.of(new Repo("z", List.of(), Map.of("ghost", "1.0.0"))), catalog());
        check(!unknown.ok() && unknown.error.equals("unknown library: ghost"), "unknown libraries must be refused");
        Order unknownDep = syncOrder(List.of(new Repo("m", List.of("nope"), Map.of())));
        check(unknownDep.order == null && unknownDep.error.equals("unknown dependency: nope of m"), "an unknown dependency edge must be refused: " + unknownDep.error);
        // --- boundary: a single repo with no deps and no pins syncs trivially.
        List<Repo> lone = List.of(new Repo("only", List.of(), Map.of()));
        Order loneOrder = syncOrder(lone);
        Resolution loneResolution = resolve(lone, catalog());
        check(loneOrder.error == null && loneOrder.order.equals(List.of("only")) && loneResolution.ok() && loneResolution.versions.isEmpty() && applyPlan(lone, loneOrder, loneResolution.versions).size() == 1, "a single repo must sync with an empty version plan");
        // --- plan execution keeps the "dependencies first" invariant after every applied step.
        List<String> log = applyPlan(fleet, order, resolution.versions);
        check(log.size() == 7 && log.get(log.size() - 1).startsWith("synced web ->") && log.get(log.size() - 1).contains("http@1.4.2") && log.get(log.size() - 1).contains("json@2.1.0"),
                "the plan must sync web last with the resolved versions: " + log.get(log.size() - 1));
        check(log.get(0).equals("synced docs ->"), "docs must sync first: " + log.get(0));

        System.out.println("PASS");
    }

    /** Oracle topological order: repeatedly take the alphabetically smallest node whose deps are done. */
    static List<String> oracleOrder(List<Repo> repos) {
        List<String> done = new ArrayList<>();
        List<String> remaining = new ArrayList<>();
        for (Repo r : repos) remaining.add(r.name);
        Collections.sort(remaining);
        while (!remaining.isEmpty()) {
            String picked = null;
            for (String name : remaining) {
                Repo repo = null;
                for (Repo r : repos) if (r.name.equals(name)) repo = r;
                if (done.containsAll(repo.deps)) { picked = name; break; }
            }
            if (picked == null) return List.of();
            remaining.remove(picked);
            done.add(picked);
        }
        return done;
    }
}
