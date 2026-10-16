"""Task 666 - Implement a conflict-free replicated JSON structure (complex CRDT).  (Tier: Expert, Ch 9 - System Design and Scalability)

A replicated JSON document is a tree of nodes with globally unique ids: a map node keeps, per key, the
set of children written for it, a list node keeps RGA-ordered items (each item records the item it was
inserted after, siblings ordered by id descending), and every node carries a last-writer-wins register
value and a last-writer-wins deletion timestamp. Resolution is deterministic - a key or list slot shows
the child with the greatest (counter, site) stamp, and a node whose deletion stamp beats its write stamp
stays deleted - so replicas agree without coordination. Merging is a per-node union (maximum stamps,
unioned children and items), which makes it idempotent, commutative and associative, and any two replicas
that saw the same edits materialise the same canonical JSON.

Verified: LWW on concurrent writes with the site tie-break, deletion versus concurrent write in both merge
orders, subtree deletion, concurrent list insertion at one anchor, element removal with tombstones,
idempotence/commutativity/associativity, canonical-JSON convergence and deterministic randomised sessions.
Simplified: maps, lists and registers only (no counters, sets or text), and whole-state merges.
"""
import json


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


ROOT = (0, "")
ZERO = (0, "")


def kind_of(value):
    if isinstance(value, dict):
        return "map"
    if isinstance(value, list):
        return "list"
    return "register"


class JsonCrdt:
    """One replica: the node table plus local editing operations."""

    def __init__(self, site):
        self.site = site
        self.counter = 0
        self.nodes = {ROOT: {"type": "map", "value": (ZERO, None), "deleted": None,
                             "entries": {}, "items": {}, "anchor": None}}

    # ------------------------------------------------------------------ plumbing
    def _stamp(self):
        self.counter += 1
        return (self.counter, self.site)

    def _new_node(self, value, anchor=None, stamp=None):
        node_id = (self.counter, self.site) if stamp is None else stamp
        return node_id, {"type": kind_of(value), "value": (self._stamp() if stamp is None else stamp, value),
                         "deleted": None, "entries": {}, "items": {}, "anchor": anchor}

    def _visible_children(self, node_id, key):
        """The child that wins a map key: newest write among the ones that are not deleted."""
        candidates = self.nodes[node_id]["entries"].get(key, [])
        alive = [child for child in candidates if not self._is_deleted(child)]
        if not alive:
            return None
        return max(alive, key=lambda child: self.nodes[child]["value"][0])

    def _is_deleted(self, node_id):
        deletion = self.nodes[node_id]["deleted"]
        stamp = self.nodes[node_id]["value"][0]
        return deletion is not None and deletion >= stamp

    def _resolve(self, path):
        """Walk `path` through the visible structure; returns the node id or None."""
        node_id = ROOT
        for step in path:
            node = self.nodes[node_id]
            if node["type"] == "map":
                node_id = self._visible_children(node_id, step)
            elif node["type"] == "list":
                visible = self.list_ids(node_id)
                check(isinstance(step, int) and 0 <= step < len(visible), "list index out of range")
                node_id = visible[step]
            else:
                raise ValueError("cannot descend into a scalar value")
            if node_id is None:
                return None
        return node_id

    def _anchor_for(self, node_id, index):
        visible = self.list_ids(node_id)
        check(0 <= index <= len(visible), "list index out of range")
        return visible[index - 1] if index > 0 else None

    # ------------------------------------------------------------------ list ordering
    def list_ids(self, node_id):
        """RGA materialisation: children ordered by anchor, siblings by id descending."""
        items = self.nodes[node_id]["items"]
        children = {}
        for item_id, anchor in items.items():
            children.setdefault(anchor, []).append(item_id)
        for siblings in children.values():
            siblings.sort(reverse=True)
        order = []
        alive = [item_id for item_id in items if not self._is_deleted(item_id)]
        stack = [item_id for item_id in reversed(children.get(None, []))]
        while stack:
            item_id = stack.pop()
            if item_id in alive:
                order.append(item_id)
            stack.extend(reversed(children.get(item_id, [])))
        return order

    # ------------------------------------------------------------------ local operations
    def _populate(self, node_id, value):
        """Expand a container payload into child nodes so nested maps/lists arereplicated."""
        if isinstance(value, dict):
            for key in sorted(value):
                child_value = value[key]
                child_id, child_node = self._new_node(child_value, stamp=self._stamp())
                self.nodes[child_id] = child_node
                self.nodes[node_id]["entries"].setdefault(key, []).append(child_id)
                self._populate(child_id, child_value)
        elif isinstance(value, list):
            for item in value:
                anchor = self._anchor_for(node_id, len(self.nodes[node_id]["items"]))
                item_id, item_node = self._new_node(item, anchor=anchor, stamp=self._stamp())
                self.nodes[item_id] = item_node
                self.nodes[node_id]["items"][item_id] = anchor
                self._populate(item_id, item)

    def set(self, path, value, key=None):
        """Write a value at `path`; `key` names the map key that receives it."""
        stamp = self._stamp()
        target = self._resolve(path) if path else ROOT
        check(target is not None, "the path does not exist")
        node_id, node = self._new_node(value, stamp=stamp)
        self.nodes[node_id] = node
        parent = self.nodes[target]
        if parent["type"] == "map":
            check(key is not None, "a map write needs a key")
            parent["entries"].setdefault(key, []).append(node_id)
        elif parent["type"] == "list":
            parent["items"][node_id] = self._anchor_for(target, len(parent["items"]))
        else:
            raise ValueError("a scalar cannot hold children")
        self._populate(node_id, value)
        return node_id

    def append(self, path, value):
        stamp = self._stamp()
        target = self._resolve(path) if path else ROOT
        check(target is not None, "the path does not exist")
        check(self.nodes[target]["type"] == "list", "append needs a list")
        node_id, node = self._new_node(value, anchor=self._anchor_for(target, len(self.nodes[target]["items"])),
                                       stamp=stamp)
        self.nodes[node_id] = node
        self.nodes[target]["items"][node_id] = node["anchor"]
        self._populate(node_id, value)
        return node_id

    def delete(self, path):
        target = self._resolve(path) if path else ROOT
        check(target is not None, "the path does not exist")
        check(target != ROOT, "the root cannot be deleted")
        self.nodes[target]["deleted"] = self._stamp()
        return target

    # ------------------------------------------------------------------ merge and output
    def merge(self, other):
        for node_id, node in other.nodes.items():
            self.counter = max(self.counter, node_id[0], node["value"][0][0])
            if node["deleted"] is not None:
                self.counter = max(self.counter, node["deleted"][0])
            if node_id not in self.nodes:
                self.nodes[node_id] = {"type": node["type"],
                                       "value": node["value"],
                                       "deleted": node["deleted"],
                                       "entries": {key: list(children)
                                                   for key, children in node["entries"].items()},
                                       "items": dict(node["items"]),
                                       "anchor": node["anchor"]}
                continue
            mine = self.nodes[node_id]
            if node["value"][0] > mine["value"][0]:
                mine["value"] = node["value"]
                mine["type"] = node["type"]
            if node["deleted"] is not None and (mine["deleted"] is None or node["deleted"] > mine["deleted"]):
                mine["deleted"] = node["deleted"]
            for key, children in node["entries"].items():
                known = mine["entries"].setdefault(key, [])
                for child in children:
                    if child not in known:
                        known.append(child)
                known.sort()
            for item_id, anchor in node["items"].items():
                mine["items"].setdefault(item_id, anchor)
        for node_id in self.nodes:
            self.nodes[node_id]["items"] = dict(sorted(self.nodes[node_id]["items"].items()))
            for key in self.nodes[node_id]["entries"]:
                self.nodes[node_id]["entries"][key].sort()
        return self

    def materialize(self, node_id=ROOT):
        node = self.nodes[node_id]
        if self._is_deleted(node_id):
            return None
        if node["type"] == "map":
            result = {}
            for key in sorted(node["entries"]):
                child = self._visible_children(node_id, key)
                if child is None:
                    continue
                value = self.materialize(child)
                if value is not None:
                    result[key] = value
            return result
        if node["type"] == "list":
            return [value for value in (self.materialize(item) for item in self.list_ids(node_id))
                    if value is not None]
        return node["value"][1]

    def to_json(self):
        return json.dumps(self.materialize(), sort_keys=True)

    def clone(self, site=None):
        copy = JsonCrdt(site if site is not None else self.site)
        copy.counter = self.counter
        copy.nodes = {}
        for node_id, node in self.nodes.items():
            copy.nodes[node_id] = {"type": node["type"], "value": node["value"], "deleted": node["deleted"],
                                   "entries": {key: list(children)
                                               for key, children in node["entries"].items()},
                                   "items": dict(node["items"]), "anchor": node["anchor"]}
        return copy


def main():
    # 1. basic map and list editing
    replica = JsonCrdt("A")
    replica.set([], {"name": "ada"}, key="user")
    replica.set(["user"], "paris", key="city")
    replica.set([], [1, 2, 3], key="scores")
    replica.append(["scores"], 4)
    check(replica.to_json() == json.dumps({"scores": [1, 2, 3, 4], "user": {"city": "paris", "name": "ada"}},
                                          sort_keys=True),
          "nested maps and lists must materialise correctly")
    check(replica.materialize()["scores"] == [1, 2, 3, 4], "list order must follow insertion order")
    replica.set(["user"], "rome", key="city")
    check(replica.materialize()["user"]["city"] == "rome",
          "the newest write to a key must win")
    check(replica.to_json().count("paris") == 0, "the superseded value must not be visible")

    # 2. deletion removes the subtree from the output
    victim = JsonCrdt("A")
    victim.set([], {"inner": 1}, key="branch")
    victim.set(["branch"], "leaf", key="leaf")
    check(victim.materialize() == {"branch": {"leaf": "leaf", "inner": 1}}, "the subtree is present")
    victim.delete(["branch"])
    check(victim.materialize() == {}, "a deleted subtree must vanish from the output")
    check("branch" not in victim.to_json(), "the key must not appear at all")

    # 3. last-writer-wins on concurrent writes to one key: timestamp order, tie broken by site
    left, right = JsonCrdt("A"), JsonCrdt("B")
    left.set([], "seed", key="k")                  # stamp (1, "A")
    right.merge(left)                              # replica B adopts the node; its counter becomes 1
    right.set([], "right", key="k")                # stamp (2, "B")
    left.set([], "left", key="k")                  # stamp (2, "A") - same counter, lower site
    order_one = left.merge(right.clone("B"))
    order_two = right.clone("B").merge(left)
    check(order_one.to_json() == order_two.to_json(),
          "the winner must not depend on the merge order")
    check(order_one.materialize()["k"] == "right",
          "on an equal counter the higher site id must win")
    older = JsonCrdt("A")
    older.merge(order_one)
    older.set([], "newer", key="k")                # a strictly newer stamp on this replica
    check(older.materialize()["k"] == "newer",
          "a write with a higher counter must win regardless of the site")

    # 4. deletion wins or loses on timestamps; a concurrent write is a new node and survives
    producer = JsonCrdt("B")
    producer.set([], "value", key="k")             # node stamp (1, "B")
    stale = JsonCrdt("A")
    stale.merge(producer)                          # A adopts B's node
    stale.delete(["k"])                            # deletion stamp (2, "A") beats the node's (1, "B")
    check(stale.materialize().get("k") is None, "a newer deletion must remove the key")
    check(stale.clone("A").merge(producer).materialize().get("k") is None,
          "the reverse merge order must agree on the deletion")
    rival = JsonCrdt("C")
    rival.set([], "rival", key="k")                # a concurrent write unaware of the deletion
    for index in range(3):
        rival.set([], index, key=f"pad{index}")    # advance C's counter
    rival.merge(producer)
    contested = rival.clone("C").merge(stale)
    check(contested.materialize().get("k") == "rival",
          "the surviving write must win the key")
    check(stale.clone("A").merge(rival).to_json() == contested.to_json(),
          "the winner must not depend on the merge order")
    late = JsonCrdt("D")
    late.merge(contested)                          # D's counter overtakes every stamp it has seen
    late.delete(["k"])                             # so this deletion is strictly newer
    check(late.clone("D").merge(contested).materialize().get("k") is None,
          "a strictly newer deletion must win")
    check(contested.clone("C").merge(late).materialize().get("k") is None,
          "the reverse order must agree on the newer deletion")

    # 5. concurrent writes to different paths both survive
    alice, bob = JsonCrdt("A"), JsonCrdt("B")
    alice.set([], {"x": 1}, key="one")
    bob.merge(alice.clone("B"))
    alice.set([], {"y": 2}, key="two")
    bob.set([], {"z": 3}, key="three")
    alice.merge(bob)
    bob.merge(alice)
    check(alice.to_json() == bob.to_json(), "both replicas must converge")
    check(alice.materialize() == {"one": {"x": 1}, "three": {"z": 3}, "two": {"y": 2}},
          "writes to different keys must all be preserved")

    # 6. concurrent list insertions at the same anchor and element removal
    left_list, right_list = JsonCrdt("A"), JsonCrdt("B")
    left_list.set([], [], key="items")
    right_list.merge(left_list.clone("B"))
    left_list.append(["items"], "a")
    right_list.append(["items"], "b")
    left_list.merge(right_list)
    right_list.merge(left_list)
    check(left_list.to_json() == right_list.to_json(), "list order must converge")
    check(sorted(left_list.materialize()["items"]) == ["a", "b"],
          "both concurrent insertions must survive")
    index = left_list.list_ids(left_list._visible_children(ROOT, "items"))[0]
    left_list.nodes[index]["deleted"] = left_list._stamp()
    check(len(left_list.materialize()["items"]) == 1, "a removed element must disappear")
    check(len(left_list.nodes[left_list._visible_children(ROOT, "items")]["items"]) == 2,
          "the removed element stays as a tombstone")

    # 7. merge is idempotent, commutative and associative
    base = JsonCrdt("A")
    base.set([], {"n": 1}, key="cfg")
    other = base.clone("B")
    other.set(["cfg"], True, key="flag")
    third = base.clone("C")
    third.set(["cfg"], "extra", key="note")
    once = base.clone().merge(other)
    check(once.clone().merge(other).to_json() == once.to_json(), "merge must be idempotent")
    check(base.clone().merge(other).to_json() == other.clone().merge(base).to_json(),
          "merge must be commutative")
    left_assoc = base.clone().merge(other).merge(third)
    right_assoc = base.clone().merge(other.clone().merge(third))
    check(left_assoc.to_json() == right_assoc.to_json(), "merge must be associative")
    check(base.clone().merge(JsonCrdt("Z")).to_json() == base.to_json(),
          "merging an empty replica must be the identity")

    # 8. randomised three-replica session with shuffled merge orders converges
    def session(seed):
        replicas = [JsonCrdt(name) for name in ("A", "B", "C")]
        replicas[0].set([], {}, key="root")
        for replica in replicas[1:]:
            replica.merge(replicas[0])
        keys = ["root"]
        for _ in range(30):
            seed, r1 = splitmix64(seed)
            seed, r2 = splitmix64(seed)
            seed, r3 = splitmix64(seed)
            replica = replicas[r1 % 3]
            if r3 % 4 == 0 and len(keys) > 1:
                replica.delete([keys[r2 % len(keys)]])
            else:
                replica.set([keys[r2 % len(keys)]], (r1 % 100), key=f"k{r2 % 5}")
            order = list(range(3))
            for step in range(r1 % 3):
                order.append(order.pop(0))
            for target in order:
                for source in order:
                    if target != source:
                        replicas[target].merge(replicas[source])
        return replicas

    first = session(0x5EED)
    documents = {replica.to_json() for replica in first}
    check(len(documents) == 1, f"all replicas must converge, got {documents}")
    check(first[0].materialize() != {}, "the session must produce a non-empty document")
    second = session(0x5EED)
    check({replica.to_json() for replica in second} == documents, "the session must be deterministic")
    other_seed = session(0x1234)
    check(len({replica.to_json() for replica in other_seed}) == 1,
          "another session must converge as well")

    # 9. convergence after a long chain of nested merges
    chained = first[0].clone().merge(first[1]).merge(first[2]).merge(first[0])
    check(chained.to_json() == first[0].to_json(), "chained merges must be stable")

    print(f"666 converged document={first[0].to_json()}")
    print(f"666 nodes={len(first[0].nodes)} keys={sorted(first[0].materialize())}")
    print("PASS")


if __name__ == "__main__":
    main()
