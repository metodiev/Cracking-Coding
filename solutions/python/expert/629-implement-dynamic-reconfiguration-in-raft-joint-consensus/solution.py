"""Task 629 - Implement dynamic reconfiguration in Raft (joint consensus).  (Tier: Expert, Ch 9 — System Design and Scalability)

Raft changes its membership in two steps: first the leader replicates a *joint* configuration
C_old,new whose quorum is "a majority of C_old **and** a majority of C_new", and only after that
entry commits does it replicate C_new. Every decision taken while the joint configuration is active
therefore needs a super-quorum, and because any two majorities of C_old intersect, any two joint
quorums intersect - which is exactly what stops two independent leaders from committing conflicting
entries in the middle of the change. Once C_new alone is committed the old servers are irrelevant:
quorums are computed from C_new only.

This is a deterministic model rather than a full consensus run (the message-level Raft is task 628):
a `Configuration` holds the active phase and judges voter sets, so every quorum argument can be
checked exhaustively instead of argued informally. Verified: any two joint quorums intersect (for
every pair of small configurations, by brute force over all subsets); a single-configuration majority
is *not* sufficient during the joint phase (a concrete counterexample is exhibited); the minimum
joint quorum is maj(C_old) + maj(C_new) - |C_old n C_new|, so the two configurations must overlap for
the change to be possible at all; elections and entries need the active configuration's quorum; and
after C_new commits only C_new counts, while a second reconfiguration is rejected until the first
one finishes.

Checks: exhaustive quorum-intersection and minimum-size properties, the phase transition OLD ->
JOINT -> NEW, rejected quorums in each phase (including the "old majority only" and "new majority
only" negatives), election quorums, a one-change-at-a-time negative check, and a liveness run that
shows nothing can commit while only the overlap node is reachable.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


OLD, JOINT, NEW = "C_old", "C_old,new", "C_new"


def majority_size(config):
    return len(config) // 2 + 1


def is_majority(config, voters):
    return len(set(voters) & set(config)) >= majority_size(config)


def all_subsets(nodes):
    nodes = sorted(nodes)
    for mask in range(1 << len(nodes)):
        yield frozenset(node for index, node in enumerate(nodes) if mask >> index & 1)


def single_quorums(config):
    """Every set of voters that is a majority of one configuration."""
    return [voters for voters in all_subsets(config) if is_majority(config, voters)]


def joint_quorums(old, new):
    """Every set of voters that is a majority of C_old *and* of C_new."""
    return [voters for voters in all_subsets(set(old) | set(new))
            if is_majority(old, voters) and is_majority(new, voters)]


class Configuration:
    """The membership state machine: which phase is active decides what a quorum means."""

    def __init__(self, old, new):
        self.old = frozenset(old)
        self.new = frozenset(new)
        self.phase = OLD
        self.log = [f"{OLD}={sorted(self.old)}"]
        self.entries = []                      # committed configuration entries
        self.rejected = []
        self.pending_change = None

    # ---- quorum rules ----------------------------------------------------------------------
    def active_configs(self):
        if self.phase == JOINT:
            return (self.old, self.new)
        return (self.new, ) if self.phase == NEW else (self.old, )

    def is_quorum(self, voters):
        voters = set(voters)
        return all(is_majority(config, voters) for config in self.active_configs())

    def quorum_size_floor(self):
        """Smallest possible quorum in the active phase."""
        if self.phase == JOINT:
            return majority_size(self.old) + majority_size(self.new) - len(self.old & self.new)
        config = self.active_configs()[0]
        return majority_size(config)

    def outsiders(self, voters):
        return set(voters) - (self.old | self.new)

    # ---- protocol steps --------------------------------------------------------------------
    def start_change(self, new_config):
        check(self.pending_change is None,
              "a second reconfiguration cannot start while one is in flight")
        self.pending_change = frozenset(new_config)
        return f"propose {sorted(self.pending_change)} (joint phase starts at commit)"

    def commit(self, entry, voters):
        """Try to commit a log entry; True when the voters form a quorum of the active phase.

        The two configuration entries are only meaningful in the phase they lead to, so a stale
        C_old,new or a premature C_new is refused before the quorum is even considered.
        """
        voters = set(voters)
        expected = {"C_old,new": OLD, "C_new": JOINT}
        if (entry in expected and self.phase != expected[entry]) or not self.is_quorum(voters):
            self.rejected.append((self.phase, entry, sorted(voters)))
            return False
        self.entries.append((self.phase, entry))
        if entry == "C_old,new":
            self.phase = JOINT
        elif entry == "C_new":
            self.phase = NEW
            self.pending_change = None
        return True

    def elect(self, candidate, voters):
        """An election needs a quorum in the active configuration, and during the transition the
        candidate must belong to C_new: a leader that the change is about to remove could not
        finish the change, so Raft requires the transitional leader to be a member of C_new."""
        config = self.old if self.phase == OLD else self.new
        return (candidate in config and candidate in set(voters)
                and self.is_quorum(voters))

    def describe(self):
        return f"{self.phase}(old={sorted(self.old)}, new={sorted(self.new)})"


def main():
    old = frozenset([1, 2, 3])
    new = frozenset([3, 4, 5])

    # --- 1. the exhaustive quorum theory -------------------------------------------------------
    check(majority_size(old) == 2 and majority_size(new) == 2, "both configurations need 2 of 3")
    for config in (old, new):
        majorities = single_quorums(config)
        check(len(majorities) == 4, "a 3-node configuration has exactly four majorities")
        for first in majorities:
            for second in majorities:
                check(first & second, "two majorities of the same configuration always intersect")

    quorums = joint_quorums(old, new)
    check(frozenset([1, 2, 3, 4, 5]) in quorums and frozenset([1, 3, 4]) in quorums,
          "the whole cluster and any 2+2 super-quorum are joint quorums")
    check(frozenset([1, 2]) not in quorums and frozenset([3, 4, 5]) not in quorums,
          "a majority of only one side is not a joint quorum")
    for first in quorums:
        for second in quorums:
            check(first & second,
                  f"joint quorums {sorted(first)} and {sorted(second)} must intersect")
    sizes = sorted(len(q) for q in quorums)
    check(sizes[0] == 3 == majority_size(old) + majority_size(new) - len(old & new),
          "the smallest joint quorum needs maj(old) + maj(new) - |old n new| = 3 voters")
    check(any(node not in q for q in quorums for node in (1, 2, 4, 5)),
          "no single node - not even the overlap node - is required by every quorum")

    disjoint_old, disjoint_new = frozenset([1, 2, 3]), frozenset([4, 5, 6])
    disjoint_quorums = joint_quorums(disjoint_old, disjoint_new)
    check(majority_size(disjoint_old) == 2
          and min(len(q) for q in disjoint_quorums) == 4
          and all(is_majority(disjoint_old, q) and is_majority(disjoint_new, q)
                  for q in disjoint_quorums),
          "with disjoint configurations every joint quorum needs four voters")
    for first in disjoint_quorums:
        for second in disjoint_quorums:
            check(first & second, "overlap still comes from the two old-configuration majorities")

    # --- 2. phase OLD: a plain majority of C_old commits the joint entry ------------------------
    config = Configuration(old, new)
    check(config.phase == OLD and not config.is_quorum([1]), "one vote is never a quorum")
    check(config.is_quorum([1, 2]) and config.is_quorum([2, 3]) and not config.is_quorum([1, 4]),
          "in phase C_old a 2-of-3 majority of the old configuration is required")
    check(config.start_change(new).startswith("propose"), "the leader proposes the new config")
    check(config.commit("C_old,new", [1, 2]), "the joint entry commits on a C_old majority")
    check(config.phase == JOINT and config.entries == [(OLD, "C_old,new")],
          "committing it switches the cluster into the joint phase")

    # --- 3. phase JOINT: both majorities are required ------------------------------------------
    check(not config.is_quorum([1, 2]) and not config.is_quorum([3, 4, 5]),
          "neither side alone can decide during the transition")
    check(config.is_quorum([1, 3, 4]) and config.is_quorum([1, 2, 3, 4, 5]),
          "a super-quorum with a majority of both configurations can")
    check(config.is_quorum([2, 3, 4]) and not config.is_quorum([2, 4, 5]),
          "the old majority must include a node that is also in C_new (node 3 here)")
    check(config.quorum_size_floor() == 3, "the joint phase needs three voters at least")
    check(not config.commit("client=x", [1, 2]),
          "a client entry cannot commit on the old majority alone during the transition")
    check(not config.commit("client=x", [3, 4, 5]),
          "nor on the new majority alone - the two halves cannot run apart")
    check(config.commit("client=x", [1, 3, 4]),
          "a genuine joint quorum commits client entries as well")
    check(len(config.rejected) == 2 and all(phase == JOINT for phase, _e, _v in config.rejected),
          "both rejected attempts were recorded in the joint phase")

    # --- 4. elections during the transition need the joint quorum ------------------------------
    check(config.elect(3, [1, 3, 4]) and config.elect(4, [1, 2, 3, 4, 5]),
          "a candidate from C_new wins with a joint quorum")
    check(not config.elect(1, [1, 2, 3, 4, 5]),
          "a node the change is about to remove cannot lead the joint configuration")
    check(not config.elect(1, [1, 2]) and not config.elect(4, [3, 4, 5]),
          "nobody can be elected with only one configuration's majority")
    check(not config.elect(7, [1, 2, 3, 4, 5, 7]),
          "a candidate outside both configurations cannot win even with every vote")
    check(config.outsiders([6, 7]) == {6, 7} and config.outsiders([1, 6]) == {6},
          "nodes outside C_old u C_new are not voters")

    # --- 5. phase NEW: only the new configuration counts ----------------------------------------
    check(not config.commit("C_new", [3, 4, 5]),
          "even the C_new entry needs a joint quorum while the joint phase is still active")
    check(config.commit("C_new", [1, 3, 4]), "with a joint quorum the change completes")
    check(config.phase == NEW and config.pending_change is None,
          "the change finished exactly when C_new committed")
    check(tuple(config.active_configs()) == (new,), "the active configuration is C_new alone")
    check(config.is_quorum([3, 4, 5]) and config.is_quorum([4, 5]) and not config.is_quorum([3]),
          "a 2-of-3 majority of C_new is enough again")
    check(not config.is_quorum([1, 2]) and not config.is_quorum([1, 2, 3]),
          "the removed nodes no longer contribute - their majority is worthless")
    check(config.quorum_size_floor() == 2, "quorums shrink back to a plain majority")
    check(config.commit("client=y", [4, 5]) and config.entries[-1] == (NEW, "client=y"),
          "entries commit on the new configuration's majority")
    check(not config.commit("C_old,new", [4, 5]),
          "the joint entry cannot be re-committed after the transition")
    check(len(config.rejected) == 4, "exactly the four attempts above were rejected")

    # --- 6. one change at a time ---------------------------------------------------------------
    config = Configuration(old, frozenset([3, 4, 5]))
    config.start_change(frozenset([3, 4, 5]))
    try:
        config.start_change(frozenset([3, 4, 5, 6]))
        raise AssertionError("a second reconfiguration must be refused while one is in flight")
    except AssertionError as error:
        check("second reconfiguration" in str(error), "the refusal comes from the one-change rule")
    config.commit("C_old,new", [1, 2, 3])
    config.commit("C_new", [1, 3, 4])
    check(config.phase == NEW and config.start_change(frozenset([3, 4, 5, 6])).startswith("propose"),
          "a new change may start once the previous one has finished")

    # --- 7. liveness: the transition needs the overlap ------------------------------------------
    config = Configuration(old, new)
    config.commit("C_old,new", [1, 2])
    reachable = [1, 2]                         # only the two doomed nodes are up
    check(not config.commit("C_new", reachable),
          "the transition cannot finish while only the old majority is reachable")
    reachable_one_new = [1, 2, 4]
    check(not config.commit("C_new", reachable_one_new),
          "one new node is not enough: node 3 links the two configurations")
    check(config.commit("C_new", [1, 2, 3, 4, 5]), "with everything up the change goes through")
    check(config.phase == NEW, "and the cluster is on the new configuration")

    config = Configuration(old, new)
    config.commit("C_old,new", [1, 2, 3])
    overlapping = [3]                          # the only node both configurations share
    check(config.commit("C_new", overlapping) is False and config.phase == JOINT,
          "the overlap node alone is not a quorum - it is the bridge, not the majority")
    check(config.commit("C_new", overlapping + [4, 5]) is False,
          "the overlap node plus a new majority is still short of the old majority")
    check(config.commit("C_new", [1, 2, 3, 4]),
          "the change finishes as soon as a joint quorum is reachable again")

    print(f"reconfig: joint-quorums(old=3,new=3)={len(quorums)} min-size={sizes[0]} "
          f"disjoint-min-size={min(len(q) for q in disjoint_quorums)} "
          f"entries={len(config.entries)}")


if __name__ == "__main__":
    main()
    print("PASS")
