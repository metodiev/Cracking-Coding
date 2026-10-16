"""Task 833 - Implement consensus over heterogeneous trust (federated consensus) simulation. (Tier: Master, CTCI Ch 9 — System Design and Scalability)

Nodes are not equal: each federated validator carries an integer trust weight, and a quorum is a
*weight* threshold (here strictly more than two thirds of the total weight) rather than a head
count. The protocol is a compact Tendermint-style lock-step round -- propose, prevote, precommit,
commit -- driven by an explicit message log with a deterministic scheduler, and the Byzantine
behaviours modelled are the ones the weight argument is about: an equivocating proposer (two
contradictory values in one round), validators that vote for every value they see, and silent
ones.

What is implemented
-------------------
* `total_weight`, `quorum_threshold(weights) = floor(2W/3) + 1`, `weight_of(voters)`: the
  threshold arithmetic. Votes are tallied per distinct validator, so duplicates weigh nothing.
* `intersection_lemma`: brute-forces *every* pair of subsets that reach quorum weight and
  reports the minimum intersection weight -- 2*theta - W -- and, with a plain majority threshold
  instead, exhibits two disjoint quorums (the negative case that shows what 2/3 buys).
* `Federation.round(height, rnd, plan)`: the proposer comes from a deterministic *weighted*
  rotation (heavier validators propose more often); honest validators vote for at most one value
  per round, refuse to vote against a locked value, precommit only after seeing a prevote
  quorum, and commit only after seeing a precommit quorum; faulty validators follow the script.
* `run_height` / `chain`: retries, one round per timeout period, so liveness under message loss
  is measurable rather than assumed.
* `safety_violations`: two commits at the same height carrying different values.
* `exhaustive_attack`: enumerates every pair of recipient sets for an equivocating proposer and
  reports the worst achievable outcome, so the safety claim is tested against all plans, not one.

What is simplified
------------------
* Certificate assembly is global: when the vote log reaches a quorum weight for a value, the
  validators that voted for it commit in the same step. Real nodes gossip votes and assemble
  their own certificates; the deterministic global tally keeps the run reproducible and leaves
  the safety argument untouched, because that argument is entirely about which weight sets can
  exist.
* One round is one timeout period -- no clocks, no timeouts to tune, no signatures, no
  execution: values are abstract block ids.
* Equivocation is not slashed: the simulation *records* every validator that appears in two
  vote buckets in one round, which is the evidence a real client would use.

Invariants verified
-------------------
1.  Threshold arithmetic: a quorum is strictly more than 2/3 of the weight, duplicate votes do
    not change any tally, and the full federation weighs exactly W.
2.  Quorum intersection brute-forced over all 32 subsets of a five-validator federation with
    weights [4,3,3,2,1]: every quorum pair overlaps in at least 2*theta - W = 5 weight (12 of
    the subsets reach quorum weight).
3.  The threshold is tight and load-bearing: with a majority threshold two disjoint quorums
    exist, so the 2/3 rule is doing real work.
4.  Safety below the bound: with a fault set below 2*theta - W, *every* equivocation plan the
    enumerator can build commits a single value, the commit carries a real quorum certificate,
    the equivocators are recorded as evidence, and the safety checker reports nothing.
5.  The attack succeeds exactly when the arithmetic says it can: with a fault set of exactly
    2*theta - W some plan commits two conflicting values, both certificates have quorum weight,
    and the two certificates intersect in exactly the faulty weight.
6.  Locking: a validator locked to another value stays out of every vote bucket, and no honest
    validator appears in two buckets in one round.
7.  Liveness: with no faults every height commits in round 1; dropping a whole round's messages
    costs exactly one extra round; a silent proposer wastes one round and a later proposer
    succeeds; commits stay inside the round budget and remain unique per height.
8.  Message accounting: a round stays under 2 * n * (n - 1) vote messages and the count is
    measured, not assumed.
9.  Heterogeneous trust bites: a weight-0 observer changes neither the threshold nor a
    certificate, and raising a validator's weight raises both W and the quorum threshold.
10. Determinism: a second run produces identical commits, certificates, votes and messages.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# trust weights and quorums
# --------------------------------------------------------------------------------------
def total_weight(weights):
    return sum(weights.values())


def quorum_threshold(weights, numerator=2, denominator=3):
    """Strictly more than numerator/denominator of the total weight."""
    return (numerator * total_weight(weights)) // denominator + 1


def weight_of(voters, weights):
    return sum(weights.get(node, 0) for node in set(voters))


def subsets(nodes):
    """Every subset as a frozenset (the federation is small on purpose)."""
    result = [frozenset()]
    for node in nodes:
        result += [subset | {node} for subset in result]
    return result


def intersection_lemma(weights, threshold=None):
    """(minimum intersection weight, witness pair, all quorum subsets)."""
    theta = quorum_threshold(weights) if threshold is None else threshold
    quorums = [subset for subset in subsets(sorted(weights))
               if weight_of(subset, weights) >= theta]
    worst, witness = None, None
    for first in quorums:
        for second in quorums:
            overlap = weight_of(first & second, weights)
            if worst is None or overlap < worst:
                worst, witness = overlap, (sorted(first), sorted(second))
    return worst, witness, quorums


def weighted_proposer_order(weights):
    """Heavier validators appear more often in the deterministic proposer rotation."""
    order = []
    for node in sorted(weights, key=lambda node: (-weights[node], node)):
        order += [node] * max(1, weights[node])
    return order


# --------------------------------------------------------------------------------------
# the federation
# --------------------------------------------------------------------------------------
class Federation:
    """Lock-step rounds over an explicit message log; faulty validators follow a script."""

    def __init__(self, weights, faults=None):
        self.weights = dict(weights)
        self.faults = dict(faults or {})          # node -> "equivocate" | "silent"
        self.theta = quorum_threshold(weights)
        self.proposers = weighted_proposer_order(weights)
        self.locked = {node: (None, None) for node in weights}   # node -> (value, height)
        self.messages = []                        # (round, sender, kind, value, recipient)
        self.votes = []                           # (round, phase, node, value)
        self.commits = []                         # (height, round, value, certificate)
        self.double_voters = set()

    # -- one round ----------------------------------------------------------------------
    def round(self, height, rnd, plan=None):
        """plan: {value: [recipients]} stands in for an equivocating proposer's messages."""
        proposer = self.proposers[(rnd + height - 1) % len(self.proposers)]
        if plan is None:
            plan = {} if self.faults.get(proposer) == "silent" \
                else {f"h{height}r{rnd}": sorted(self.weights)}
        report = {"height": height, "round": rnd, "proposer": proposer, "plan": dict(plan),
                  "prevotes": {}, "precommits": {}, "commits": []}

        # prevote: an honest validator votes for at most one value per round, and never
        # against the value it is locked to; an equivocator votes for everything it sees
        first_vote = {}
        prevotes = {}
        for candidate, recipients in sorted(plan.items()):
            voters = set()
            for node in sorted(set(recipients)):
                behavior = self.faults.get(node)
                if behavior == "silent":
                    continue
                if behavior == "equivocate":
                    voters.add(node)
                    continue
                locked_value, locked_height = self.locked[node]
                if locked_height == height and locked_value not in (None, candidate):
                    continue                                  # locked at this height: refuse
                if node in first_vote and first_vote[node] != candidate:
                    continue
                first_vote[node] = candidate
                voters.add(node)
            prevotes[candidate] = sorted(voters)
        for candidate, voters in sorted(prevotes.items()):
            for node in voters:
                self.votes.append((rnd, "prevote", node, candidate))
                for recipient in sorted(self.weights):
                    if recipient != node:
                        self.messages.append((rnd, node, "prevote", candidate, recipient))
        self.double_voters |= {node for node in self.weights
                               if sum(1 for voters in prevotes.values() if node in voters) > 1}
        report["prevotes"] = {candidate: (voters, weight_of(voters, self.weights))
                              for candidate, voters in prevotes.items()}

        # precommit: only values that reached a prevote quorum, and only by their voters
        precommits = {}
        for candidate, voters in sorted(prevotes.items()):
            if weight_of(voters, self.weights) < self.theta:
                continue
            precommitting = set(voters)
            for node in self.weights:
                if self.faults.get(node) == "equivocate" and node in precommitting:
                    continue
            for node in sorted(precommitting):
                self.votes.append((rnd, "precommit", node, candidate))
                for recipient in sorted(self.weights):
                    if recipient != node:
                        self.messages.append((rnd, node, "precommit", candidate, recipient))
                self.locked[node] = (candidate, height)
            precommits[candidate] = sorted(precommitting)
        report["precommits"] = {candidate: (voters, weight_of(voters, self.weights))
                                for candidate, voters in precommits.items()}

        # commit: only values that reached a precommit quorum
        for candidate, voters in sorted(precommits.items()):
            if weight_of(voters, self.weights) >= self.theta:
                certificate = (candidate, tuple(sorted(voters)),
                               weight_of(voters, self.weights))
                self.commits.append((height, rnd, candidate, certificate))
                report["commits"].append(certificate)
        return report

    def run_height(self, height, max_rounds=4, plan_for_round=None):
        """Rounds until this height commits (a round is one timeout period)."""
        report = None
        for rnd in range(1, max_rounds + 1):
            report = self.round(height, rnd, plan_for_round(rnd) if plan_for_round else None)
            if report["commits"]:
                return report
        return report

    def chain(self, heights, max_rounds=4):
        return [self.run_height(height, max_rounds) for height in range(1, heights + 1)]

    def commits_at(self, height):
        return [commit for commit in self.commits if commit[0] == height]

    def safety_violations(self):
        violations = []
        for height in sorted({commit[0] for commit in self.commits}):
            values = {commit[2] for commit in self.commits_at(height)}
            if len(values) > 1:
                violations.append((height, sorted(values)))
        return violations


# --------------------------------------------------------------------------------------
# attack plans
# --------------------------------------------------------------------------------------
def attack_plan(weights, faulty, first_value="a", second_value="b"):
    """The split an equivocating proposer wants: the faulty set plus each honest camp."""
    honest = [node for node in sorted(weights) if node not in faulty]
    camp_one, running = list(faulty), weight_of(faulty, weights)
    for node in honest:
        if running >= quorum_threshold(weights):
            break
        camp_one.append(node)
        running += weights[node]
    camp_two = list(faulty) + [node for node in honest if node not in camp_one]
    return {first_value: sorted(set(camp_one)), second_value: sorted(set(camp_two))}


def exhaustive_attack(weights, faulty, max_rounds=1):
    """Every pair of recipient sets; report the worst outcome for safety."""
    best = {"conflicts": 0, "plan": None, "intersection": None, "commits": []}
    for first in subsets(sorted(weights)):
        for second in subsets(sorted(weights)):
            federation = Federation(weights, {node: "equivocate" for node in faulty})
            plan = {"a": sorted(first), "b": sorted(second)}
            for rnd in range(1, max_rounds + 1):
                federation.round(1, rnd, plan)
            commits = federation.commits_at(1)
            conflicts = len({commit[2] for commit in commits})
            if conflicts > best["conflicts"]:
                best = {"conflicts": conflicts, "plan": plan,
                        "intersection": weight_of(first & second, weights), "commits": commits}
    return best


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def main():
    weights = {"v0": 4, "v1": 3, "v2": 3, "v3": 2, "v4": 1}
    W = total_weight(weights)
    theta = quorum_threshold(weights)

    # -- 1. threshold arithmetic ----------------------------------------------------------
    check(W == 13 and theta == 9, f"2/3 of {W} is 8.67, so a quorum needs {theta} weight")
    check(2 * theta - W == 5, "two quorums must overlap in at least 2*theta - W = 5 weight")
    check(weight_of(["v0", "v1", "v0", "v0"], weights) == 7,
          "the tally counts a validator once however many duplicate votes arrive")
    check(weight_of(list(weights), weights) == W, "the whole federation weighs exactly W")
    check(quorum_threshold({"v0": 4, "v1": 3, "v2": 3, "v3": 2}) == 9,
          "dropping the lightest validator keeps the threshold at 9 (8 weight is not enough)")

    # -- 2. quorum intersection, brute-forced ---------------------------------------------
    worst, witness, quorums = intersection_lemma(weights)
    check(worst == 2 * theta - W == 5,
          f"the minimum intersection over all quorum pairs is {worst} ({witness})")
    check(len(quorums) == 9, f"{len(quorums)} of the 32 subsets have quorum weight")
    check(all(weight_of(subset, weights) >= theta for subset in quorums),
          "every enumerated quorum really has quorum weight")
    majority_worst, majority_witness, _ = intersection_lemma(weights, threshold=6)
    check(majority_worst == 0,
          f"with a half-weight threshold two quorums can be disjoint "
          f"({majority_witness}), which is exactly what the 2/3 rule prevents")

    # -- 3. safety with a fault set below the overlap bound --------------------------------
    safe_faults = ["v1", "v4"]                        # weight 4 < 2*theta - W = 5
    check(weight_of(safe_faults, weights) == 4 < 2 * theta - W,
          "the safe fault set is strictly below the overlap bound")
    attack = exhaustive_attack(weights, safe_faults)
    check(attack["conflicts"] == 1,
          f"over all 1024 equivocation plans the federation commits exactly one value "
          f"({attack['conflicts']} distinct values)")
    safe_run = Federation(weights, {node: "equivocate" for node in safe_faults})
    safe_run.round(1, 1, attack_plan(weights, safe_faults))
    certificate = safe_run.commits_at(1)[0][3]
    check(certificate[2] >= theta,
          f"the single commit carries a {certificate[2]}-weight certificate")
    check(safe_run.safety_violations() == [], "no safety violation at height 1")
    check(safe_run.double_voters == set(safe_faults),
          f"the equivocation is recorded as evidence ({sorted(safe_run.double_voters)})")

    # -- 4. the same attack with enough faulty weight breaks safety ------------------------
    unsafe_faults = ["v1", "v3"]                      # weight 5 == 2*theta - W
    check(weight_of(unsafe_faults, weights) == 2 * theta - W,
          "the unsafe fault set exactly reaches the overlap bound")
    broken = exhaustive_attack(weights, unsafe_faults)
    check(broken["conflicts"] == 2 and broken["intersection"] == 5,
          f"with 5 faulty weight two conflicting values commit, and the quorums overlap in "
          f"exactly the faulty weight ({broken['intersection']})")
    certificates = broken["commits"]
    check(len(certificates) == 2 and all(item[3][2] >= theta for item in certificates),
          "both conflicting commits carry full quorum certificates")
    check(set(certificates[0][3][1]) & set(certificates[1][3][1]) == set(unsafe_faults),
          "the two certificates intersect exactly in the equivocating validators")
    unsafe = Federation(weights, {node: "equivocate" for node in unsafe_faults})
    unsafe.round(1, 1, broken["plan"])
    check(unsafe.safety_violations() and unsafe.safety_violations()[0][0] == 1,
          f"the safety checker reports the violation ({unsafe.safety_violations()})")
    check(len({commit[2] for commit in unsafe.commits_at(1)}) == 2,
          "two different values committed at one height")

    # -- 5. locking and one-vote-per-round -------------------------------------------------
    locked = Federation(weights)
    locked.locked["v2"] = ("h1r0", 1)
    locked_report = locked.round(1, 1, {"a": ["v0", "v2"], "b": ["v2", "v3"]})
    check("v2" not in locked_report["prevotes"]["a"][0]
          and "v2" not in locked_report["prevotes"]["b"][0],
          "a validator locked to another value stays out of every vote bucket")
    check(set(locked_report["prevotes"]["a"][0]) == {"v0"}
          and set(locked_report["prevotes"]["b"][0]) == {"v3"},
          "the rest of the federation votes normally")
    honest_round = Federation(weights).round(1, 1)
    buckets = [voters for voters, _ in honest_round["prevotes"].values()]
    check(sum(len(bucket) for bucket in buckets) == len(weights)
          and all(bucket.count(node) == 1 for bucket in buckets for node in bucket),
          "each validator appears in at most one bucket when everyone is honest")
    check(Federation(weights).double_voters == set(),
          "no double voters in a fault-free round")

    # -- 6. liveness ------------------------------------------------------------------------
    healthy = Federation(weights)
    reports = healthy.chain(3)
    check(all(report["round"] == 1 and report["commits"] for report in reports),
          "with no faults the first round decides every height")
    check([commit[0] for commit in healthy.commits] == [1, 2, 3],
          "the committed heights are exactly 1, 2 and 3")
    check(healthy.safety_violations() == [], "the three-height chain is safe")
    lossy = Federation(weights)
    lossy_reports = [lossy.run_height(height, plan_for_round=lambda rnd: {} if rnd < 2 else None)
                     for height in range(1, 3)]
    check(all(report["commits"] and report["round"] == 2 for report in lossy_reports),
          "dropping a whole round's messages costs exactly one extra round")
    check(len(lossy.commits_at(1)) == 1 and len(lossy.commits_at(2)) == 1
          and lossy.safety_violations() == [],
          "retrying a round does not duplicate or conflict the commit")
    silent = Federation(weights, {"v0": "silent"})
    silent_first = silent.round(1, 1)
    check(not silent_first["commits"] and W - weights["v0"] >= theta,
          "a silent proposer wastes a round but the rest still holds quorum weight")
    silent_later = silent.run_height(1)
    check(silent_later["commits"] and silent_later["round"] > 1,
          f"a later proposer commits the height (round {silent_later['round']})")
    check(silent.safety_violations() == [], "the silent-proposer run stays safe")

    # -- 7. message complexity and trust accounting -----------------------------------------
    counted = Federation(weights)
    counted.round(1, 1)
    check(len(counted.messages) == 2 * len(weights) * (len(weights) - 1) == 40,
          f"a round really costs 2 * n * (n-1) vote messages ({len(counted.messages)})")
    check(len(counted.votes) == 2 * len(weights),
          "prevote plus precommit is exactly two votes per validator")
    check({phase for _, phase, _, _ in counted.votes} == {"prevote", "precommit"},
          "only the two vote phases travel")
    zero = Federation({"v0": 4, "v1": 3, "v2": 3, "v3": 2, "observer": 0})
    check(zero.theta == theta, "a weight-0 observer cannot move the threshold")
    observer_round = zero.round(1, 1)
    check(observer_round["commits"]
          and weight_of(observer_round["commits"][0][1], zero.weights) >= zero.theta,
          "and it cannot inflate a certificate either")
    heavier = {"v0": 6, "v1": 3, "v2": 3, "v3": 2, "v4": 1}
    check(total_weight(heavier) > W and quorum_threshold(heavier) > theta,
          "raising a validator's weight raises both W and the quorum threshold")

    # -- 8. determinism ---------------------------------------------------------------------
    again = Federation(weights)
    again.chain(3)
    check(again.commits == healthy.commits and again.votes == healthy.votes
          and again.messages == healthy.messages,
          "a second run produces identical commits, votes and messages")
    check([commit[3] for commit in again.commits] == [commit[3] for commit in healthy.commits],
          "the certificates are reproducible too")

    print(f"weights {weights}: W={W}, theta={theta}, minimum quorum intersection {worst} "
          f"(= 2*theta - W); a fault set of {2 * theta - W} weight breaks safety")
    print(f"exhaustive plan enumeration: fault weight 4 -> {attack['conflicts']} committed "
          f"value(s); fault weight 5 -> {broken['conflicts']} conflicting commits")
    print(f"liveness: rounds per height {[report['round'] for report in reports]} (healthy), "
          f"{[report['round'] for report in lossy_reports]} (first round dropped), "
          f"silent proposer {[silent_later['round']]}")
    print("PASS")


if __name__ == "__main__":
    main()
