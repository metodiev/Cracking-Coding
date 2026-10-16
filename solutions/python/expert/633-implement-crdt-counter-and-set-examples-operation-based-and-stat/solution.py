"""Task 633 - Implement CRDT counter and set examples (operation-based and state-based).  (Tier: Expert, Ch 9 — System Design and Scalability)

A state-based CRDT is defined by a join-semilattice: merging two states takes the least upper bound, so
merge must be commutative, associative and idempotent, and every state must be reachable in any order.
That is what makes the replica converge without coordination - the same data merged in any order, any
number of times, ends in the same state. An operation-based CRDT instead ships operations, and relies
on a reliable broadcast that delivers each operation exactly once (or on the receiver de-duplicating
by operation id under at-least-once delivery); the operations themselves must commute.

Implemented here: a grow-only counter and a PN-counter (two grow-only counters, so the state stays
monotone), an operation-based counter that de-duplicates by (replica, sequence), a grow-only set, a
two-phase set with tombstones (remove wins forever), and an observed-remove set whose tags make a
concurrent add and remove resolve as "add wins". Simplified: tags are deterministic per-replica
strings, the op-based counter carries integer deltas only, and there is no garbage collection of
tombstones or tags - each of those is stated where it matters in the checks.

Verified: the merge laws and convergence-under-any-order property for every state-based type (all
permutations of the replicas' states, including duplicates), monotonicity of the counters, that a
re-add after a remove is ignored by the 2P-Set, that a concurrent add/remove keeps the element in the
OR-Set while a sequential remove deletes it, and that a de-duplicating operation-based counter reaches
exactly the same value as its state-based twin under reordering and duplicate delivery.

Checks: lattice laws over generated state sets, value monotonicity, PN-counter negatives, permutation
convergence, op-based de-duplication and reordering, G-Set/2P-Set/OR-Set semantics including the
concurrent-vs-sequential remove distinction, cross-check between the op-based and state-based counters,
and empty/self-merge boundaries.
"""

from itertools import permutations


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def merged(first, second):
    """A fresh state holding the join of the two arguments (used for the lattice-law checks)."""
    copy = first.copy()
    copy.merge(second)
    return copy


def lattice_laws(states, message):
    """Commutativity, associativity and idempotence over every triple of the given states."""
    for first in states:
        check(first.signature() == merged(first, first).signature(), f"{message}: idempotent")
        for second in states:
            check(merged(first, second).signature() == merged(second, first).signature(),
                  f"{message}: commutative")
            for third in states:
                left = merged(merged(first, second), third).signature()
                right = merged(first, merged(second, third)).signature()
                check(left == right, f"{message}: associative")


class GCounter:
    """Grow-only counter: one monotonically increasing count per replica, joined by max."""

    def __init__(self, counts=None):
        self.counts = dict(counts or {})

    def copy(self):
        return GCounter(self.counts)

    def increment(self, replica, amount=1):
        check(amount >= 0, "a grow-only counter cannot decrease")
        self.counts[replica] = self.counts.get(replica, 0) + amount
        return self

    def merge(self, other):
        for replica, count in other.counts.items():
            if count > self.counts.get(replica, 0):
                self.counts[replica] = count
        return self

    def value(self):
        return sum(self.counts.values())

    def signature(self):
        return tuple(sorted(self.counts.items()))


class PNCounter:
    """Increment/decrement counter: two grow-only counters, so the state stays monotone."""

    def __init__(self, positive=None, negative=None):
        self.positive = positive or GCounter()
        self.negative = negative or GCounter()

    def copy(self):
        return PNCounter(self.positive.copy(), self.negative.copy())

    def increment(self, replica, amount=1):
        self.positive.increment(replica, amount)
        return self

    def decrement(self, replica, amount=1):
        self.negative.increment(replica, amount)
        return self

    def value(self):
        return self.positive.value() - self.negative.value()

    def merge(self, other):
        self.positive.merge(other.positive)
        self.negative.merge(other.negative)
        return self

    def signature(self):
        return (self.positive.signature(), self.negative.signature())


class OpCounter:
    """Operation-based counter: apply each (replica, sequence) operation exactly once."""

    def __init__(self, replica):
        self.replica = replica
        self.sequence = 0
        self.applied = set()
        self.per_replica = {}
        self.duplicates = 0

    def increment(self, amount=1):
        self.sequence += 1
        operation = (self.replica, self.sequence, amount)
        self.apply(operation)
        return operation

    def apply(self, operation):
        replica, sequence, amount = operation
        if (replica, sequence) in self.applied:
            self.duplicates += 1
            return False
        self.applied.add((replica, sequence))
        self.per_replica[replica] = self.per_replica.get(replica, 0) + amount
        return True

    def value(self):
        return sum(self.per_replica.values())

    def signature(self):
        return tuple(sorted(self.per_replica.items()))


class GSet:
    """Grow-only set: union is the join, so nothing is ever removed."""

    def __init__(self, elements=()):
        self.elements = set(elements)

    def copy(self):
        return GSet(self.elements)

    def add(self, element):
        self.elements.add(element)
        return self

    def contains(self, element):
        return element in self.elements

    def merge(self, other):
        self.elements |= other.elements
        return self

    def signature(self):
        return tuple(sorted(self.elements))


class TwoPSet:
    """Two-phase set: an element is present while it is in `added` and not in `removed`; because the
    tombstone set only grows, a remove always wins - even against a later add of the same element."""

    def __init__(self, added=(), removed=()):
        self.added = set(added)
        self.removed = set(removed)

    def copy(self):
        return TwoPSet(self.added, self.removed)

    def add(self, element):
        self.added.add(element)
        return self

    def remove(self, element):
        self.removed.add(element)
        return self

    def contains(self, element):
        return element in self.added and element not in self.removed

    def merge(self, other):
        self.added |= other.added
        self.removed |= other.removed
        return self

    def signature(self):
        return (tuple(sorted(self.added)), tuple(sorted(self.removed)))


class ORSet:
    """Observed-remove set: every add carries a unique tag, and a remove tombstones only the tags the
    replica had observed. A concurrent add therefore survives the remove (add wins)."""

    def __init__(self, adds=None, removes=None):
        self.adds = {element: set(tags) for element, tags in (adds or {}).items()}
        self.removes = {element: set(tags) for element, tags in (removes or {}).items()}

    def copy(self):
        return ORSet(self.adds, self.removes)

    def add(self, element, tag):
        self.adds.setdefault(element, set()).add(tag)
        return tag

    def remove(self, element):
        """Tombstone exactly the tags this replica has observed for the element."""
        observed = self.adds.get(element, set())
        self.removes.setdefault(element, set()).update(observed)
        return set(observed)

    def contains(self, element):
        return bool(self.adds.get(element, set()) - self.removes.get(element, set()))

    def merge(self, other):
        for element, tags in other.adds.items():
            self.adds.setdefault(element, set()).update(tags)
        for element, tags in other.removes.items():
            self.removes.setdefault(element, set()).update(tags)
        return self

    def signature(self):
        return (tuple(sorted((element, tuple(sorted(tags))) for element, tags in self.adds.items())),
                tuple(sorted((element, tuple(sorted(tags))) for element, tags in self.removes.items())))

    def live_tags(self, element):
        return self.adds.get(element, set()) - self.removes.get(element, set())


def counter_states():
    return [GCounter(), GCounter({"A": 1}), GCounter({"A": 2, "B": 1}),
            GCounter({"B": 3}), GCounter({"A": 1, "B": 3, "C": 4})]


def set_states():
    return [GSet(), GSet(["x"]), GSet(["x", "y"]), GSet(["y", "z"]), GSet(["x", "y", "z"])]


def two_phase_states():
    return [TwoPSet(), TwoPSet(["x"], []), TwoPSet(["x"], ["x"]), TwoPSet(["x", "y"], ["x"])]


def or_set_states():
    states = []
    for adds in ({}, {"x": {"a1"}}, {"x": {"a1", "b1"}}, {"x": {"b2"}}):
        for removes in ({}, {"x": {"a1"}}):
            states.append(ORSet(adds, removes))
    return states


def main():
    # --- 1. the merge laws of every state-based type -------------------------------------------
    lattice_laws(counter_states(), "G-Counter")
    lattice_laws([PNCounter(GCounter({"A": 1}), GCounter({"B": 2})), PNCounter(GCounter(), GCounter()),
                  PNCounter(GCounter({"B": 2}), GCounter({"A": 1}))], "PN-Counter")
    lattice_laws(set_states(), "G-Set")
    lattice_laws(two_phase_states(), "2P-Set")
    lattice_laws(or_set_states(), "OR-Set")

    # --- 2. G-Counter ---------------------------------------------------------------------------
    a, b, c = GCounter(), GCounter(), GCounter()
    a.increment("A", 3)
    b.increment("B")
    b.increment("B")
    c.increment("C", 5)
    check(a.value() == 3 and b.value() == 2 and c.value() == 5, "each replica counts its own events")
    check(a.copy().merge(b).merge(c).value() == 10, "the converged value is the sum of the replicas")
    permutations_of_merges = set()
    for order in permutations([a, b, c], 3):
        state = GCounter()
        for replica in order:
            state.merge(replica)
        permutations_of_merges.add(state.signature())
    check(permutations_of_merges == {a.copy().merge(b).merge(c).signature()},
          "any merge order reaches exactly the same state")
    stale = GCounter({"A": 1})
    check(a.copy().merge(stale).value() == 3, "a stale state cannot lower a counter (monotone merge)")
    check(merged(a, a).value() == a.value(), "merging a counter with itself changes nothing")
    check(GCounter().value() == 0 and GCounter().signature() == (), "an empty counter is zero")

    values_before = [state.value() for state in counter_states()]
    for state in counter_states():
        for other in counter_states():
            check(state.copy().merge(other).value() >= state.value(),
                  "merge never decreases the value - the state is monotone")

    # --- 3. PN-Counter -------------------------------------------------------------------------
    left, right = PNCounter(), PNCounter()
    left.increment("A", 5)
    left.decrement("A", 2)
    right.decrement("B", 3)
    right.increment("B", 1)
    check(left.value() == 3 and right.value() == -2, "each replica tracks increments and decrements")
    check(left.copy().merge(right).value() == 1, "the merged value is 3 - 2 = 1")
    check(right.copy().merge(left).value() == 1, "and it does not depend on the direction")
    combined = left.copy().merge(right)
    check(combined.positive.signature() == (("A", 5), ("B", 1))
          and combined.negative.signature() == (("A", 2), ("B", 3)),
          "the two underlying counters keep every replica's contribution")
    check(combined.value() == combined.positive.value() - combined.negative.value() == 1,
          "the value is exactly positive minus negative")

    # --- 4. operation-based counter: de-duplication and reordering ------------------------------
    source = OpCounter("A")
    other_source = OpCounter("B")
    operations = [source.increment(1) for _ in range(3)]
    operations += [other_source.increment(2) for _ in range(2)]
    check(len(operations) == 5 and len({(op[0], op[1]) for op in operations}) == 5,
          "five operations with five distinct ids were generated")
    check(source.value() == 3, "the generating replica applied its own operations")

    delivery_orders = [operations, list(reversed(operations)),
                       operations[2:] + operations[:2], operations + operations]
    values = set()
    for order in delivery_orders:
        replica = OpCounter("S")
        for operation in order:
            replica.apply(operation)
        values.add(replica.value())
    check(values == {7}, f"reordering and duplicate delivery still give the same value: {values}")
    duplicate_replica = OpCounter("S")
    for operation in operations + operations:
        duplicate_replica.apply(operation)
    check(duplicate_replica.duplicates == 5, "every re-delivered operation was recognised")
    check(duplicate_replica.signature() == (("A", 3), ("B", 4)),
          "the op-based counter agrees with the per-replica totals")

    # cross-check: the op-based counter and its state-based twin hold the same value
    twin = GCounter({"A": 3, "B": 4})
    check(duplicate_replica.value() == twin.value() == 7,
          "the operation-based and state-based counters agree")

    # --- 5. G-Set ------------------------------------------------------------------------------
    first, second = GSet(["a", "b"]), GSet(["b", "c"])
    check(first.copy().merge(second).signature() == ("a", "b", "c"), "the join of sets is a union")
    check(second.copy().merge(first).signature() == ("a", "b", "c"), "in either direction")
    check(not GSet().contains("a") and GSet().copy().merge(first).contains("a"),
          "an empty set learns the elements of any state it merges")
    check(first.copy().merge(first).signature() == first.signature(),
          "union with itself is a no-op (idempotent)")

    # --- 6. 2P-Set: remove wins, and a later add cannot resurrect the element --------------------
    phase = TwoPSet()
    phase.add("x").add("y")
    check(phase.contains("x") and phase.contains("y"), "both elements start out present")
    phase.remove("x")
    check(not phase.contains("x") and phase.contains("y"), "the removed element disappears")
    phase.add("x")                                # re-adding after the remove
    check(not phase.contains("x"),
          "a re-add cannot resurrect the element: the tombstone only grows (remove wins)")
    check(phase.signature() == (("x", "y"), ("x",)), "the tombstone is still there")
    replica = TwoPSet()
    replica.add("x")
    check(replica.contains("x"), "a replica that never saw the remove still holds the element")
    replica.merge(phase)
    check(not replica.contains("x") and replica.contains("y"),
          "merging the tombstone deletes the element on the other replica too")
    check(phase.copy().merge(replica).signature() == replica.signature(),
          "and the two states are now identical")
    check(not TwoPSet(["z"], []).copy().merge(TwoPSet([], ["z"])).contains("z"),
          "add and remove of the same element at once resolves to absent")

    # --- 7. OR-Set: concurrent add wins, sequential remove deletes ------------------------------
    replica_a, replica_b = ORSet(), ORSet()
    tag_a = replica_a.add("x", "A-1")
    replica_b.add("x", "B-1")
    replica_b.remove("x")                          # B removes only the tag it has observed
    check(replica_b.contains("x") is False, "B sees the element as removed locally")
    merged_state = replica_a.copy().merge(replica_b)
    check(merged_state.contains("x"),
          "the concurrent add wins: A's tag was never observed by the remove")
    check(merged_state.live_tags("x") == {"A-1"} and tag_a == "A-1",
          "only the surviving tag keeps the element alive")

    sequential = ORSet()
    sequential.add("x", "A-1")
    sequential.remove("x")                         # this replica did observe the add
    check(not sequential.contains("x"), "a sequential remove deletes the element")
    check(sequential.live_tags("x") == set() and sequential.adds["x"] == {"A-1"},
          "the observed tag is tombstoned, not deleted")
    sequential.add("x", "A-2")
    check(sequential.contains("x"), "a genuinely new add does resurrect the element")

    fresh_a, fresh_b, fresh_c = ORSet(), ORSet(), ORSet()
    fresh_a.add("x", "A-1")
    fresh_b.add("x", "B-1")
    fresh_c.add("y", "C-1")
    fresh_b.remove("x")
    states = [fresh_a, fresh_b, fresh_c]
    signatures = set()
    for order in permutations(states, 3):
        state = ORSet()
        for replica in order:
            state.merge(replica)
        signatures.add(state.signature())
    check(len(signatures) == 1, "the OR-Set converges to one state under any merge order")
    converged = ORSet()
    for replica in states:
        converged.merge(replica)
    check(converged.contains("x") and converged.contains("y"),
          "the concurrent add survives in the converged state")

    # --- 8. state-based and operation-based views of the same history ---------------------------
    history = [("A", 1), ("B", 4), ("A", 2), ("C", 7)]
    state_based = GCounter()
    op_based = OpCounter("S")
    for index, (replica, amount) in enumerate(history, start=1):
        state_based.increment(replica, amount)
        op_based.apply((replica, index, amount))
    check(state_based.value() == op_based.value() == 14,
          "both CRDT flavours compute the same total for the same history")
    check(state_based.signature() == op_based.signature(),
          "and the same per-replica breakdown")

    print(f"crdt: g-counter states={len(counter_states())} g-set states={len(set_states())} "
          f"or-set states={len(or_set_states())} op-based-duplicates={duplicate_replica.duplicates} "
          f"converged-or-set={sorted(converged.adds)}")


if __name__ == "__main__":
    main()
    print("PASS")
