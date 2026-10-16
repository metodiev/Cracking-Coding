/** Task 981 - Implement reproducible deployment of full platform via infra-as-code.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A declarative infrastructure engine. A desired state is a set of typed resources with properties, an
 *  immutable property set and dependency edges; the planner validates the graph (unknown dependency or
 *  cycle rejected before anything is touched), orders resources topologically and diffs against the current
 *  state: absent resources are created, changed mutable properties updated, changed immutable properties
 *  replaced, and resources that only exist in the current state are deleted after every create/update. The
 *  applier is transactional and idempotent: applying the plan yields exactly the desired state and a second
 *  plan is empty.
 *  Simplified: an in-memory state store with string properties, no providers, no remote API and no rollback
 *  of a partially failed provider call (validation failures abort before any mutation); the diff semantics,
 *  the dependency ordering, idempotence and drift detection are the real ones.
 *  Verified: creation order, an independent kind-comparison oracle for the plan, apply == desired state,
 *  no-op second plan, out-of-band drift detection and repair, immutable replacement, deletion after
 *  creates, dependency/cycle rejection without mutation, the destroy ordering and reproducibility.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Resource {
        final String name, type; final Map<String, String> props; final List<String> dependsOn; final Set<String> immutable;
        Resource(String name, String type, Map<String, String> props, List<String> dependsOn, Set<String> immutable) {
            this.name = name; this.type = type; this.props = new TreeMap<>(props);
            this.dependsOn = List.copyOf(dependsOn); this.immutable = Set.copyOf(immutable);
        }
        boolean sameAs(Resource other) { return type.equals(other.type) && props.equals(other.props); }
    }

    static final class Action {
        final String kind, name, reason; final Resource resource;
        Action(String kind, String name, Resource resource, String reason) {
            this.kind = kind; this.name = name; this.resource = resource; this.reason = reason;
        }
        String render() { return kind + " " + name + " (" + reason + ")"; }
    }

    /** Kahn topological order over the desired graph; a missing dependency or a cycle is rejected. */
    static List<Resource> order(List<Resource> desired) {
        Map<String, Resource> byName = new TreeMap<>();
        for (Resource r : desired) byName.put(r.name, r);
        if (byName.size() != desired.size()) throw new IllegalStateException("duplicate resource name");
        Map<String, Integer> indegree = new TreeMap<>();
        Map<String, List<String>> after = new TreeMap<>();
        for (Resource r : desired) { indegree.put(r.name, r.dependsOn.size()); after.put(r.name, new ArrayList<>()); }
        for (Resource r : desired) {
            for (String dependency : r.dependsOn) {
                if (!byName.containsKey(dependency)) throw new IllegalArgumentException("unknown dependency " + dependency);
                after.get(dependency).add(r.name);
            }
        }
        PriorityQueue<String> ready = new PriorityQueue<>();
        for (Map.Entry<String, Integer> e : indegree.entrySet()) if (e.getValue() == 0) ready.add(e.getKey());
        List<Resource> out = new ArrayList<>();
        while (!ready.isEmpty()) {
            String name = ready.poll();
            out.add(byName.get(name));
            for (String next : after.get(name)) if (indegree.merge(next, -1, Integer::sum) == 0) ready.add(next);
        }
        if (out.size() != desired.size()) throw new IllegalStateException("dependency cycle in desired state");
        return out;
    }

    /** Plan: creates/updates/replaces in dependency order, then deletions, dependents first. */
    static List<Action> plan(Map<String, Resource> current, List<Resource> desired) {
        List<Action> actions = new ArrayList<>();
        for (Resource want : order(desired)) {
            Resource have = current.get(want.name);
            if (have == null) {
                actions.add(new Action("create", want.name, want, "absent"));
            } else if (changedImmutable(have, want)) {
                actions.add(new Action("replace", want.name, want, "immutable property changed"));
            } else if (!have.sameAs(want)) {
                actions.add(new Action("update", want.name, want, "properties drifted"));
            }
        }
        Set<String> wanted = new TreeSet<>();
        for (Resource r : desired) wanted.add(r.name);
        List<String> deletions = new ArrayList<>();
        for (String name : current.keySet()) if (!wanted.contains(name)) deletions.add(name);
        for (String name : reverse(deletions, current)) {
            actions.add(new Action("delete", name, null, "not declared in desired state"));
        }
        return actions;
    }

    /** Independent oracle for the plan: pairwise comparison of current against desired, name by name. */
    static Map<String, String> kindsOracle(Map<String, Resource> current, List<Resource> desired) {
        Map<String, String> kinds = new TreeMap<>();
        for (Resource want : desired) {
            Resource have = current.get(want.name);
            if (have == null) kinds.put(want.name, "create");
            else if (changedImmutable(have, want)) kinds.put(want.name, "replace");
            else if (!have.type.equals(want.type) || !have.props.equals(want.props)) kinds.put(want.name, "update");
        }
        for (String name : current.keySet()) {
            boolean declared = false;
            for (Resource want : desired) declared |= want.name.equals(name);
            if (!declared) kinds.put(name, "delete");
        }
        return kinds;
    }

    static boolean changedImmutable(Resource have, Resource want) {
        for (String key : have.immutable) {
            if (!String.valueOf(have.props.get(key)).equals(String.valueOf(want.props.get(key)))) return true;
        }
        return false;
    }

    /** Deletions are emitted dependents-first, so a resource is removed before what it relies on. */
    static List<String> reverse(List<String> names, Map<String, Resource> current) {
        List<String> sorted = new ArrayList<>(names);
        sorted.sort((a, b) -> depth(b, current) - depth(a, current));
        return sorted;
    }

    static int depth(String name, Map<String, Resource> current) {
        Resource r = current.get(name);
        int best = 0;
        for (String dependency : r.dependsOn) if (current.containsKey(dependency)) best = Math.max(best, depth(dependency, current) + 1);
        return best;
    }

    static Map<String, Resource> apply(Map<String, Resource> current, List<Action> actions) {
        Map<String, Resource> next = new TreeMap<>(current);
        for (Action action : actions) {
            if (action.kind.equals("delete")) next.remove(action.name);
            else if (!action.kind.equals("none")) next.put(action.name, action.resource);
        }
        return next;
    }

    static List<Resource> desired() {
        return List.of(
                new Resource("network", "vpc", Map.of("cidr", "10.0.0.0/16"), List.of(), Set.of("cidr")),
                new Resource("cluster", "k8s", Map.of("size", "3", "region", "eu-west-1"), List.of("network"), Set.of("region")),
                new Resource("service", "app", Map.of("image", "app:1.0", "replicas", "2"), List.of("cluster"), Set.of("image")),
                new Resource("database", "sql", Map.of("engine", "postgres", "size", "10"), List.of("cluster"), Set.of("engine")));
    }

    static Map<String, Resource> initial() {
        Map<String, Resource> state = new TreeMap<>();
        state.put("network", new Resource("network", "vpc", Map.of("cidr", "10.0.0.0/16"), List.of(), Set.of("cidr")));
        state.put("cluster", new Resource("cluster", "k8s", Map.of("size", "2", "region", "eu-west-1"), List.of("network"), Set.of("region")));
        state.put("legacy", new Resource("legacy", "vm", Map.of("ami", "old"), List.of(), Set.of()));
        return state;
    }

    public static void main(String[] args) {
        Map<String, Resource> state = initial();
        List<Action> actions = plan(state, desired());

        Map<String, String> oracle = kindsOracle(state, desired());
        Map<String, String> actual = new TreeMap<>();
        for (Action action : actions) actual.put(action.name, action.kind);
        check(actual.equals(oracle) && actions.size() == 4 && actual.get("cluster").equals("update")
                        && actual.get("service").equals("create") && actual.get("legacy").equals("delete"),
                "the plan must match the independent kind oracle: " + actions.stream().map(Action::render).toList());

        int cluster = indexOf(actions, "cluster"), service = indexOf(actions, "service");
        int database = indexOf(actions, "database"), legacy = indexOf(actions, "legacy");
        check(cluster < service && cluster < database && legacy > service && legacy > database
                        && actions.get(actions.size() - 1).kind.equals("delete"),
                "creates must follow dependencies and deletions must come last: "
                        + actions.stream().map(Action::render).toList());

        Map<String, Resource> deployed = apply(state, actions);
        List<Action> second = plan(deployed, desired());
        check(second.isEmpty() && deployed.size() == 4 && deployed.get("cluster").props.get("size").equals("3")
                        && !deployed.containsKey("legacy"),
                "applying must converge on the desired state and the second plan must be empty: "
                        + second.stream().map(Action::render).toList());

        Map<String, Resource> drifted = new TreeMap<>(deployed);
        drifted.put("service", new Resource("service", "app", Map.of("image", "app:1.0", "replicas", "9"),
                List.of("cluster"), Set.of("image")));
        List<Action> repair = plan(drifted, desired());
        check(repair.size() == 1 && repair.get(0).kind.equals("update") && repair.get(0).name.equals("service")
                        && plan(apply(drifted, repair), desired()).isEmpty(),
                "out-of-band drift must be detected and repaired by one update: "
                        + repair.stream().map(Action::render).toList());

        Map<String, Resource> moved = new TreeMap<>(deployed);
        moved.put("cluster", new Resource("cluster", "k8s", Map.of("size", "3", "region", "us-east-1"),
                List.of("network"), Set.of("region")));
        List<Action> replacement = plan(moved, desired());
        check(replacement.size() == 1 && replacement.get(0).kind.equals("replace")
                        && apply(moved, replacement).get("cluster").props.get("region").equals("eu-west-1"),
                "changing an immutable property must replace the resource: "
                        + replacement.stream().map(Action::render).toList());

        List<Action> destroy = plan(deployed, List.of());
        check(destroy.size() == 4 && destroy.stream().allMatch(a -> a.kind.equals("delete"))
                        && apply(deployed, destroy).isEmpty()
                        && depth(destroy.get(0).name, deployed) >= depth(destroy.get(3).name, deployed)
                        && depth(destroy.get(3).name, deployed) == 0,
                "an empty desired state must destroy everything, dependents first: "
                        + destroy.stream().map(Action::render).toList());

        boolean missingDependency = false;
        List<Resource> brokenDepends = List.of(new Resource("service", "app", Map.of(), List.of("ghost"), Set.of()));
        try {
            plan(state, brokenDepends);
        } catch (IllegalArgumentException e) {
            missingDependency = e.getMessage().contains("ghost");
        }
        List<Resource> cyclic = List.of(new Resource("a", "x", Map.of(), List.of("b"), Set.of()),
                new Resource("b", "x", Map.of(), List.of("a"), Set.of()));
        boolean cycle = false;
        try {
            plan(state, cyclic);
        } catch (IllegalStateException e) {
            cycle = e.getMessage().contains("cycle");
        }
        check(missingDependency && cycle && state.size() == 3 && !state.containsKey("service"),
                "a broken graph must be rejected before any mutation");

        check(plan(Map.of(), List.of()).isEmpty() && apply(Map.of(), List.of()).isEmpty()
                        && plan(state, desired()).stream().map(Action::render).toList()
                        .equals(actions.stream().map(Action::render).toList()),
                "empty inputs must plan nothing and planning must be reproducible");

        System.out.println("actions=" + actions.stream().map(Action::render).toList());
        System.out.println("deployed=" + deployed.keySet() + " driftRepairs=" + repair.size()
                + " destroyed=" + destroy.size());
        System.out.println("PASS");
    }

    static int indexOf(List<Action> actions, String name) {
        for (int i = 0; i < actions.size(); i++) if (actions.get(i).name.equals(name)) return i;
        return -1;
    }
}
