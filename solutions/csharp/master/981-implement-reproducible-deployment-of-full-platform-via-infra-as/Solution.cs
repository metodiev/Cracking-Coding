// Task 981 - Implement reproducible deployment of full platform via infra-as-code.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The desired platform is a graph of resources (id, type, dependencies, settings) and the actual
// platform is the same shape observed live. Planning diffs the two: unknowns become creates, changed
// settings become updates, and resources nobody declares anymore become deletes. Creates and updates
// are ordered dependencies-first, deletes dependents-first, and the plan text is hashed so two runs
// over the same inputs must agree bit for bit. Applying the plan and re-planning must be empty, which
// is the idempotency property infrastructure-as-code is trusted for.
//
// Verified: the exact create/update/delete set, dependency order for creates and reverse order for
// deletes, the empty second plan after apply, cycle rejection, the all-delete and empty-diff
// boundaries, hash stability across two differently ordered constructions of the same state, and an
// independent transitive-closure oracle that no resource is created before its dependencies exist.
// Simplification: the provider is in-memory with no cloud API, no drift between plan and apply, and
// settings are compared as sorted key/value pairs rather than by a provider-specific differ.
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private sealed class Resource
    {
        public readonly string Id;
        public readonly string Type;
        public readonly string[] DependsOn;
        public readonly SortedDictionary<string, string> Settings;

        public Resource(string id, string type, string[] dependsOn, params string[] settings)
        {
            Id = id;
            Type = type;
            DependsOn = dependsOn;
            Settings = new SortedDictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i + 1 < settings.Length; i += 2) Settings[settings[i]] = settings[i + 1];
        }
    }

    private sealed class Change
    {
        public readonly string Kind;
        public readonly string Id;

        public Change(string kind, string id)
        {
            Kind = kind;
            Id = id;
        }
    }

    /** Kahn with an id tie-break; null when the graph contains a cycle. */
    private static List<string> DependenciesFirst(Dictionary<string, Resource> graph)
    {
        var indegree = new Dictionary<string, int>(StringComparer.Ordinal);
        var dependents = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Resource> pair in graph)
        {
            indegree[pair.Key] = 0;
            dependents[pair.Key] = new List<string>();
        }
        foreach (KeyValuePair<string, Resource> pair in graph)
            foreach (string dependency in pair.Value.DependsOn)
            {
                if (!graph.ContainsKey(dependency)) continue;
                indegree[pair.Key]++;
                dependents[dependency].Add(pair.Key);
            }
        var ready = new List<string>();
        foreach (KeyValuePair<string, int> pair in indegree) if (pair.Value == 0) ready.Add(pair.Key);
        ready.Sort(StringComparer.Ordinal);
        var order = new List<string>();
        while (ready.Count > 0)
        {
            string next = ready[0];
            ready.RemoveAt(0);
            order.Add(next);
            foreach (string dependent in dependents[next])
            {
                indegree[dependent]--;
                if (indegree[dependent] == 0) ready.Add(dependent);
            }
            ready.Sort(StringComparer.Ordinal);
        }
        return order.Count == graph.Count ? order : null;
    }

    private static bool SameSettings(Resource left, Resource right)
    {
        if (left.Type != right.Type || left.Settings.Count != right.Settings.Count) return false;
        foreach (KeyValuePair<string, string> pair in left.Settings)
            if (!right.Settings.TryGetValue(pair.Key, out string value) || value != pair.Value) return false;
        return true;
    }

    private static List<Change> Plan(Dictionary<string, Resource> desired, Dictionary<string, Resource> actual)
    {
        List<string> creationOrder = DependenciesFirst(desired);
        List<string> deletionOrder = DependenciesFirst(actual);
        if (creationOrder == null || deletionOrder == null) return null;
        var changes = new List<Change>();
        foreach (string id in creationOrder)
        {
            if (!actual.ContainsKey(id)) changes.Add(new Change("create", id));
            else if (!SameSettings(desired[id], actual[id])) changes.Add(new Change("update", id));
        }
        for (int i = deletionOrder.Count - 1; i >= 0; i--)
            if (!desired.ContainsKey(deletionOrder[i])) changes.Add(new Change("delete", deletionOrder[i]));
        return changes;
    }

    private static Dictionary<string, Resource> Apply(
        List<Change> plan, Dictionary<string, Resource> desired, Dictionary<string, Resource> actual)
    {
        var next = new Dictionary<string, Resource>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Resource> pair in actual) next[pair.Key] = pair.Value;
        foreach (Change change in plan)
        {
            if (change.Kind == "delete") next.Remove(change.Id);
            else next[change.Id] = desired[change.Id];
        }
        return next;
    }

    private static string PlanText(List<Change> plan)
    {
        var lines = new List<string>();
        foreach (Change change in plan) lines.Add(change.Kind + ":" + change.Id);
        lines.Sort(StringComparer.Ordinal);
        return string.Join(";", lines);
    }
    private static ulong PlanHash(List<Change> plan)
    {
        ulong hash = 0xCBF29CE484222325UL;
        unchecked
        {
            foreach (char character in PlanText(plan))
            {
                hash = (hash ^ character) * 0x100000001B3UL;
                hash ^= hash >> 29;
            }
        }
        return hash;
    }

    private static Dictionary<string, Resource> ActualPlatform()
    {
        return new Dictionary<string, Resource>(StringComparer.Ordinal)
        {
            { "vpc", new Resource("vpc", "network", new string[0], "cidr", "10.0.0.0/16") },
            { "db", new Resource("db", "database", new[] { "vpc" }, "size", "small") },
            { "app", new Resource("app", "service", new[] { "db" }, "replicas", "2") },
            { "stale", new Resource("stale", "service", new[] { "app" }, "replicas", "1") },
        };
    }

    private static Dictionary<string, Resource> DesiredPlatform()
    {
        return new Dictionary<string, Resource>(StringComparer.Ordinal)
        {
            { "vpc", new Resource("vpc", "network", new string[0], "cidr", "10.0.0.0/16") },
            { "cache", new Resource("cache", "cache", new[] { "vpc" }, "maxmemory", "512mb") },
            { "db", new Resource("db", "database", new[] { "vpc" }, "size", "large") },
            { "app", new Resource("app", "service", new[] { "db" }, "replicas", "2") },
        };
    }

    public static void Main()
    {
        Dictionary<string, Resource> actual = ActualPlatform();
        Dictionary<string, Resource> desired = DesiredPlatform();
        List<Change> plan = Plan(desired, actual);
        Check(plan != null, "the acyclic platform produces a plan");
        Check(PlanText(plan) == "create:cache;delete:stale;update:db",
            "the plan contains exactly one create, one update and one delete (got '" + PlanText(plan) + "')");
        Check(PlanText(plan).Contains("update:app") == false && PlanText(plan).Contains("delete:app") == false,
            "an unchanged resource appears nowhere in the plan");

        // Order: creates and updates follow dependencies, deletes run dependents-first.
        var position = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < plan.Count; i++) position[plan[i].Id] = i;
        Check(position["cache"] < position["db"],
            "the new cache is created before the database that shares its parent is updated");
        Check(plan[0].Kind == "create" && plan[plan.Count - 1].Kind == "delete",
            "deletes are emitted last so nothing is removed while it is still referenced");

        // Oracle: transitive closure says no created resource has a missing dependency at create time.
        var existing = new HashSet<string>(actual.Keys, StringComparer.Ordinal);
        var violated = new List<string>();
        foreach (Change change in plan)
        {
            if (change.Kind == "delete")
            {
                existing.Remove(change.Id);
                continue;
            }
            foreach (string dependency in desired[change.Id].DependsOn)
                if (!existing.Contains(dependency)) violated.Add(change.Id + "->" + dependency);
            existing.Add(change.Id);
        }
        Check(violated.Count == 0, "every create and update runs after its dependencies exist ("
            + string.Join(",", violated) + ")");

        // Idempotency: replanning after apply must be empty.
        Dictionary<string, Resource> applied = Apply(plan, desired, actual);
        List<Change> second = Plan(desired, applied);
        Check(second != null && second.Count == 0, "the second plan after apply is empty, so apply is idempotent");
        Check(Plan(desired, Apply(second, desired, applied)).Count == 0 && Plan(actual, actual).Count == 0,
            "a third plan stays empty, and an unchanged platform yields an empty diff");

        // Reproducibility: the same state built in a different insertion order hashes identically.
        var reordered = new Dictionary<string, Resource>(StringComparer.Ordinal);
        foreach (string id in new[] { "app", "db", "cache", "vpc" }) reordered[id] = desired[id];
        List<Change> reorderedPlan = Plan(reordered, actual);
        Check(PlanHash(reorderedPlan) == PlanHash(plan) && PlanHash(Plan(desired, actual)) == PlanHash(plan),
            "the plan hash ignores declaration order and is stable across replans");

        // Deletion ordering on a full teardown: dependents disappear before their dependencies.
        List<Change> teardown = Plan(new Dictionary<string, Resource>(StringComparer.Ordinal),
            new Dictionary<string, Resource>(StringComparer.Ordinal)
            {
                { "vpc", new Resource("vpc", "network", new string[0]) },
                { "db", new Resource("db", "database", new[] { "vpc" }) },
                { "app", new Resource("app", "service", new[] { "db" }) },
            });
        Check(teardown.Count == 3 && teardown[0].Id == "app" && teardown[1].Id == "db" && teardown[2].Id == "vpc",
            "tearing everything down deletes the service, then the database, then the network");

        // Boundaries: identical states, an empty desired state and a dependency cycle.
        Check(Plan(new Dictionary<string, Resource>(StringComparer.Ordinal),
            new Dictionary<string, Resource>(StringComparer.Ordinal)).Count == 0,
            "planning from nothing to nothing yields an empty diff");
        var cyclic = new Dictionary<string, Resource>(StringComparer.Ordinal)
        {
            { "a", new Resource("a", "service", new[] { "b" }) },
            { "b", new Resource("b", "service", new[] { "a" }) },
        };
        Check(Plan(cyclic, new Dictionary<string, Resource>(StringComparer.Ordinal)) == null,
            "a dependency cycle is rejected instead of deployed in an arbitrary order");

        Console.WriteLine("plan: " + PlanText(plan) + " (hash " + PlanHash(plan).ToString("X8") + ")");
        Console.WriteLine("teardown order: " + string.Join(" > ", teardown.ConvertAll(c => c.Id)));
        Console.WriteLine("PASS");
    }
}
