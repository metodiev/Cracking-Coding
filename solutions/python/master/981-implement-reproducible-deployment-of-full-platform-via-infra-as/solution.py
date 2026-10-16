"""Task 981 - Implement reproducible deployment of full platform via infra-as-code.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A miniature infrastructure-as-code engine.  A configuration declares resources (network, database, cache,
queue, api, workers, dashboard) with properties and explicit dependencies; the engine computes a plan by
diffing the desired resources against the recorded state, applies it in dependency order, deletes in reverse
dependency order when a resource is dropped from the configuration, and hashes the canonical form of the
configuration so the same input always yields the same fingerprint.  Applying an unchanged configuration
plans nothing (idempotence), while a mutated state entry shows up as drift that the next apply repairs.

Simplified versus Terraform: resources are dictionaries in memory instead of cloud APIs, there is no remote
state or locking, providers are a single in-memory kind table and destructive updates are modelled as
delete-plus-create.  Verified: create ordering against the dependency graph, idempotence, drift detection and
repair for updates/creations/removals, reverse-order destruction to an empty state, hash stability and
insensitivity to declaration order, rejection of cycles and unknown dependencies, plan replay against the
resulting state, empty-config and single-resource boundaries and reproducibility.
"""
import hashlib
import json
from collections import namedtuple

Resource = namedtuple("Resource", "kind name properties depends_on")
# a small but realistic platform: network -> data stores -> services -> edge
CONFIG = (
    Resource("network", "core", {"cidr": "10.0.0.0/16"}, ()),
    Resource("database", "primary", {"engine": "postgres", "size": "m5.large"}, ("network.core",)),
    Resource("cache", "sessions", {"engine": "redis", "size": "m5.small"}, ("network.core",)),
    Resource("queue", "jobs", {"engine": "kafka", "partitions": 6}, ("network.core",)),
    Resource("service", "api", {"replicas": 3, "image": "api:1.4"}, ("database.primary", "cache.sessions")),
    Resource("service", "worker", {"replicas": 2, "image": "worker:1.4"},
             ("queue.jobs", "database.primary")),
    Resource("dashboard", "ops", {"replicas": 1, "image": "dash:0.9"}, ("service.api",)),
)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect(error, call, message):
    try:
        call()
    except error:
        return
    raise AssertionError(message)


def ident(resource):
    return f"{resource.kind}.{resource.name}"


def fingerprint(resource):
    """The canonical payload of one resource: what the state must match for the resource to be in sync."""
    return (resource.kind, resource.name, tuple(sorted(resource.properties.items())),
            tuple(sorted(resource.depends_on)))


def config_hash(config):
    """SHA-256 over the canonical, declaration-order independent form of the configuration."""
    canonical = json.dumps(sorted(fingerprint(resource) for resource in config), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deploy_order(config, reverse=False):
    """Kahn's algorithm over the dependency edges; a cycle or an unknown dependency is an error."""
    resources = {ident(resource): resource for resource in config}
    successors = {key: [] for key in resources}
    indegree = {key: 0 for key in resources}
    for key, resource in resources.items():
        for dependency in resource.depends_on:
            if dependency not in resources:
                raise ValueError(f"resource {key} depends on unknown resource {dependency}")
            successors[dependency].append(key)
            indegree[key] += 1
    ready = sorted(key for key in resources if indegree[key] == 0)
    order = []
    while ready:
        key = ready.pop(0)
        order.append(key)
        for successor in sorted(successors[key]):
            indegree[successor] -= 1
            if not indegree[successor]:
                ready.append(successor)
        ready.sort()
    if len(order) != len(resources):
        raise ValueError("the configuration has a dependency cycle")
    return list(reversed(order)) if reverse else order


def plan(config, state):
    """[(action, key)]: creates and updates in deployment order, deletions in reverse dependency order."""
    desired = {ident(resource): resource for resource in config}
    actions = []
    for key in deploy_order(config):
        if key not in state:
            actions.append(("create", key))
        elif state[key] != fingerprint(desired[key]):
            actions.append(("update", key))
    for key in deletion_order(state, desired):
        actions.append(("delete", key))
    return actions


def deletion_order(state, desired):
    """Recorded resources that left the configuration, ordered so dependents go before dependencies."""
    leaving = {key: entry for key, entry in state.items() if key not in desired}
    if not leaving:
        return []
    graph = tuple(Resource(kind, name, dict(properties),
                           tuple(dependency for dependency in dependencies if dependency in leaving))
                  for kind, name, properties, dependencies in
                  (leaving[key] for key in sorted(leaving)))
    return deploy_order(graph, reverse=True)


def apply(config, state):
    """Run the plan and return (actions, new state); the state maps key -> canonical fingerprint."""
    actions = plan(config, state)
    new_state = dict(state)
    desired = {ident(resource): resource for resource in config}
    for action, key in actions:
        if action == "delete":
            new_state.pop(key, None)
        else:
            new_state[key] = fingerprint(desired[key])
    return actions, new_state


def drift(state, key, **changes):
    """Simulate out-of-band change: a property of a recorded resource is edited behind the engine."""
    mutated = dict(state)
    kind, name, properties, dependencies = mutated[key]
    updated = dict(properties)
    updated.update(changes)
    mutated[key] = (kind, name, tuple(sorted(updated.items())), dependencies)
    return mutated


def main():
    base_actions, base_state = apply(CONFIG, {})
    # 1) the first apply creates everything, in an order where every dependency precedes its dependent.
    dependencies = {ident(resource): resource.depends_on for resource in CONFIG}
    created = [key for action, key in base_actions if action == "create"]
    position = {key: index for index, key in enumerate(created)}
    check(len(base_actions) == len(CONFIG) == 7 and all(action == "create" for action, _key in base_actions)
          and all(position[dependency] < position[key]
                  for key, deps in dependencies.items() for dependency in deps),
          f"the creation order is wrong: {created}")
    # 2) idempotence: applying the same configuration again plans and changes nothing.
    repeat_actions, repeat_state = apply(CONFIG, base_state)
    check(repeat_actions == [] and repeat_state == base_state and plan(CONFIG, base_state) == [],
          f"a second apply was not a no-op: {repeat_actions}")
    # 3) drift: an out-of-band property edit plans exactly one update, and the repair restores the hash.
    drifted = drift(base_state, "database.primary", size="m5.xlarge")
    drift_actions, repaired = apply(CONFIG, drifted)
    check(drift_actions == [("update", "database.primary")] and repaired == base_state
          and hash_signature(repaired) != hash_signature(drifted),
          f"drift detection is wrong: {drift_actions}")
    # 4) removals and additions: dropping a service and adding a new one plans a delete and a create, and
    #    the deletion is ordered after everything that depends on it.
    trimmed = tuple(resource for resource in CONFIG if ident(resource) != "service.worker")
    extra = trimmed + (Resource("service", "reports", {"replicas": 1, "image": "reports:0.3"},
                                ("database.primary",)),)
    changed_actions, changed_state = apply(extra, base_state)
    check(("delete", "service.worker") in changed_actions
          and ("create", "service.reports") in changed_actions
          and sorted(changed_actions) == sorted([("delete", "service.worker"), ("create", "service.reports")])
          and "service.worker" not in changed_state and "service.reports" in changed_state,
          f"add/remove planning is wrong: {changed_actions}")
    # 5) destruction: an empty configuration deletes everything in reverse dependency order and leaves an
    #    empty state.
    destroy_actions, empty_state = apply((), base_state)
    deleted = [key for action, key in destroy_actions if action == "delete"]
    reverse_position = {key: index for index, key in enumerate(deleted)}
    check(len(destroy_actions) == len(CONFIG) == 7 and all(action == "delete" for action, _key in destroy_actions)
          and all(reverse_position[key] < reverse_position[dependency]
                  for key, deps in dependencies.items() for dependency in deps)
          and empty_state == {} and plan((), {}) == [],
          f"the destroy order is wrong: {deleted}")
    # 6) the hash is stable, changes when the configuration changes, and does not depend on declaration
    #    order - the reproducibility property the deployment pipeline relies on.
    shuffled = tuple(reversed(CONFIG))
    edited = tuple(Resource(r.kind, r.name, dict(r.properties, **({"replicas": 5} if r.kind == "service"
                                                                 else {})), r.depends_on)
                   for r in CONFIG)
    check(config_hash(CONFIG) == config_hash(shuffled) == config_hash(CONFIG)
          and config_hash(edited) != config_hash(CONFIG)
          and len(config_hash(CONFIG)) == 64 and config_hash(()) == config_hash(()),
          "the configuration hash is not reproducible")
    # 7) oracle: replaying the planned actions reproduces the state the engine returned, for every case.
    for config, state in ((CONFIG, {}), (CONFIG, base_state), (extra, base_state), ((), base_state)):
        actions, produced = apply(config, state)
        replay_state = dict(state)
        desired = {ident(resource): fingerprint(resource) for resource in config}
        for action, key in actions:
            if action == "delete":
                replay_state.pop(key, None)
            else:
                replay_state[key] = desired[key]
        check(replay_state == produced and plan(config, replay_state) == [],
              f"replaying the plan did not reproduce the state for {len(config)} resources")
    # 8) rejected inputs and boundaries: cycles, unknown dependencies, a dangling state entry that no
    #    longer appears in the configuration, and a single dependency-free resource.
    cyclic = (Resource("network", "core", {"cidr": "x"}, ("network.edge",)),
              Resource("network", "edge", {"cidr": "10.1.0.0/16"}, ("network.core",)))
    expect(ValueError, lambda: deploy_order(cyclic), "a dependency cycle must be rejected")
    expect(ValueError, lambda: deploy_order((Resource("db", "x", {}, ("network.missing",)),)),
           "an unknown dependency must be rejected")
    lone = (Resource("storage", "bucket", {"class": "cold"}, ()),)
    lone_actions, lone_state = apply(lone, {})
    check(lone_actions == [("create", "storage.bucket")] and apply(lone, lone_state)[0] == []
          and deploy_order(()) == [] and deletion_order({}, {}) == []
          and plan(CONFIG, {"ghost.resource": ("ghost", "resource", (), ())})[-1] == ("delete", "ghost.resource"),
          "the empty and single-resource boundaries are wrong")
    # 9) determinism: plans, hashes and states are byte-identical across runs.
    check(apply(CONFIG, {}) == (base_actions, base_state) and plan(CONFIG, base_state) == repeat_actions
          and config_hash(CONFIG) == config_hash(CONFIG) and deploy_order(CONFIG) == deploy_order(shuffled)
          and deletion_order(base_state, {}) == deleted,
          "the engine must be reproducible")
    print(f"resources={len(CONFIG)} hash={config_hash(CONFIG)[:16]} created={len(created)} "
          f"drift_actions={drift_actions} destroy_order={deleted[:3]}...")
    print(f"state_keys={sorted(base_state)}")
    print("PASS")


def hash_signature(state):
    """Fingerprint of a recorded state - used to show that drift really changed something."""
    return hashlib.sha256(json.dumps(sorted((key, value) for key, value in state.items()),
                                     sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
