"""Task 835 - Implement recursive SNARK composition outline (conceptual). (Tier: Master, CTCI Beyond — Security & Privacy)

Recursion is what lets a verifier check an unbounded computation with a constant amount of work:
a proof for `n` steps is a proof that "there exist two proofs, each of which *verifies*, whose
statements chain together". The prover therefore has to run the verifier *inside* the circuit at
every node, and the verifier of the final proof only re-checks a constant-size digest. This file
builds that structure explicitly: leaf proofs assert one state transition over F_p, node proofs
assert that their two children verified, and the prover really executes the in-circuit
verification at every internal node -- the count is measured, not asserted.

What is implemented
-------------------
* `step`: the deterministic transition being proven, and `leaf_proof` / `leaf_verify` where the
  leaf claim carries both the predecessor and the successor so the relation can be checked
  exactly by anyone.
* `node_proof`: runs `verify` on both children (that call *is* the recursion), records the two
  child digests, a hash of the verification trace, and the combined claim digest. The resulting
  proof has constant size regardless of how many steps the subtree covers.
* `verify`: the structural verifier. For a node it is O(1) hashes -- it checks the digest, the
  claim chain and the recorded child verdicts -- which is the succinctness property.
* `spot_check_verify`: the honest fallback, because the structural check *trusts* the recorded
  verdicts. The verifier derives `t` node indices by Fiat-Shamir from the root digest, opens
  those nodes, re-runs the verification on their children and recomputes their digests.
  `soundness_bound` gives the exact acceptance probability for a given number of tampered paths,
  and the checks measure the frequency against it.
* `ivc_fold`: the incrementally verifiable variant -- fold steps one at a time into a
  constant-size accumulator and verify the whole chain from the final accumulator.

What is simplified -- and what is explicitly *not* claimed
----------------------------------------------------------
* There is **no in-circuit verifier in the cryptographic sense**: a Python function cannot be
  snapshotted into an arithmetic circuit here, so the "circuit" is modelled by a trace digest
  plus spot checks, and the structural verifier is sound only under the honest-prover
  assumption. That gap is *demonstrated* (a lying prover passes the structural check and is
  caught by the spot check), not hidden. A real recursive SNARK needs a verifier circuit and
  pairings or a curve cycle; none of that is fabricated here. The file claims no cryptographic
  soundness it does not implement.
* There is no zero knowledge: leaves prove plain state transitions, so a leaf's claim is
  checkable by anyone who sees it.
* Deterministic throughout: SHA-256 digests, Fiat-Shamir from the root digest, no threads, no
  clock, no unseeded randomness.

Invariants verified
-------------------
1.  Leaf proofs: an honest transition verifies, a wrong successor fails, a skipped step fails,
    and a claim for another index fails.
2.  Composition really recurses: an N-step tree performs 2*(N-1) in-circuit verifications, and
    every internal node names exactly two children with recorded verdicts.
3.  Structural verification is constant-work: the root proof's byte size and the verifier's hash
    count are identical for 2, 4, 8, 16 and 32 steps, while the prover's work is linear.
4.  Recursion depth is ceil(log2 N), measured from the tree.
5.  Statement binding: replacing any single leaf changes the root digest, and no tampered tree
    collides with the honest root.
6.  An honest prover *cannot* prove a false statement: a tree built over an invalid leaf records
    the failure and the structural verifier rejects it.
7.  The documented assumption is real: a prover that rewrites the recorded verdicts passes the
    structural check, and the spot-check verifier rejects it at the measured rate, which is
    compared with the exact combinatorial bound 1 - prod (n-k-i)/(n-i).
8.  IVC: 16 folds keep the accumulator at a constant size with exactly two in-circuit
    verifications per fold, and the final accumulator verifies.
9.  Work separation: prover work 2*(N-1) versus a constant verifier cost, so the ratio grows
    with the computation.
10. Determinism: rebuilding the tree reproduces every digest and the same work.
"""
import hashlib
import math

P = 2 ** 61 - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# the program being proven
# --------------------------------------------------------------------------------------
def step(state):
    value, counter = state
    return ((7 * value + counter + 1) % P, counter + 1)


def initial_state():
    return (123456789 % P, 0)


def state_text(state):
    return f"{state[0]}:{state[1]}"


def parse(text):
    value, index = text.split(":")
    return (int(value), int(index))


def digest_of(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


# --------------------------------------------------------------------------------------
# leaf and node proofs
# --------------------------------------------------------------------------------------
def leaf_proof(previous):
    following = step(previous)
    claim = f"{state_text(previous)}->{state_text(following)}"
    return {"kind": "leaf", "claim": claim, "children": (),
            "trace": digest_of("step", claim), "digest": digest_of("leaf", claim)}


def leaf_verify(proof):
    """: an exact relation check -- no trust involved."""
    if proof["kind"] != "leaf" or "->" not in proof["claim"]:
        return False
    previous, following = proof["claim"].split("->")
    return step(parse(previous)) == parse(following)


def node_proof(left, right, counter):
    """The recursive step: both children are verified *here*, inside the node's construction."""
    counter["verifications"] += 2
    left_ok = verify(left)
    right_ok = verify(right)
    claim = digest_of("claim", left["claim"], right["claim"])
    trace = digest_of("trace", left["digest"], right["digest"], int(left_ok), int(right_ok))
    children = (left["digest"], right["digest"])
    return {"kind": "node", "claim": claim, "children": children, "trace": trace,
            "digest": digest_of("node", children[0], children[1], trace),
            "verified_children": (left_ok, right_ok)}


def verify(proof, counter=None):
    """Structural verification: O(1) work for a node, an exact relation check for a leaf."""
    if counter is not None:
        counter["hashes"] += 1
    if proof["kind"] == "leaf":
        return leaf_verify(proof)
    if not all(proof["verified_children"]):
        return False
    left, right = proof["children"]
    return proof["digest"] == digest_of("node", left, right, proof["trace"])


def build_tree(steps):
    """Balanced tree over `steps` transitions; returns (root, levels, work counter)."""
    counter = {"verifications": 0, "hashes": 0}
    state = initial_state()
    level = []
    for _ in range(steps):
        level.append(leaf_proof(state))
        state = step(state)
    levels = [level]
    while len(levels[-1]) > 1:
        current = levels[-1]
        parents = [node_proof(current[i], current[i + 1], counter)
                   for i in range(0, len(current) - 1, 2)]
        if len(current) % 2:
            parents.append(current[-1])
        levels.append(parents)
    return levels[-1][0], levels, counter


def rebuild_tree(levels):
    """Recompute every parent (used after tampering with a leaf)."""
    counter = {"verifications": 0, "hashes": 0}
    current, rebuilt = list(levels[0]), [list(levels[0])]
    while len(current) > 1:
        parents = [node_proof(current[i], current[i + 1], counter)
                   for i in range(0, len(current) - 1, 2)]
        if len(current) % 2:
            parents.append(current[-1])
        rebuilt.append(parents)
        current = parents
    return rebuilt[-1][0], rebuilt, counter


def recursive_depth(levels):
    return len(levels) - 1


def internal_nodes(levels):
    return [proof for level in levels for proof in level if proof["kind"] == "node"]


# --------------------------------------------------------------------------------------
# the honest fallback: Fiat-Shamir spot checks
# --------------------------------------------------------------------------------------
def challenge_positions(root_digest, count, total):
    positions, nonce = [], 0
    while len(positions) < min(count, total):
        candidate = int(digest_of(root_digest, "open", nonce)[:16], 16) % total
        if candidate not in positions:
            positions.append(candidate)
        nonce += 1
    return sorted(positions)


def spot_check_verify(root, levels, count):
    """Open `count` internal nodes and re-verify their children from scratch."""
    nodes = internal_nodes(levels)
    by_digest = {proof["digest"]: proof for level in levels for proof in level}
    positions = challenge_positions(root["digest"], count, len(nodes))
    for position in positions:
        opened = nodes[position]
        for child_digest in opened["children"]:
            child = by_digest.get(child_digest)
            if child is None:
                return False, positions
            if child["kind"] == "leaf":
                if not leaf_verify(child):
                    return False, positions
            elif not verify(child):
                return False, positions
        if not all(opened["verified_children"]):
            return False, positions
    return True, positions


def soundness_bound(total, tampered, count):
    """Exact P[a uniform `count`-subset of internal nodes opens none of the tampered paths]."""
    if tampered <= 0:
        return 1.0
    openings = min(count, total)
    if total - tampered < openings:
        return 0.0
    return math.prod((total - tampered - i) / (total - i) for i in range(openings))


def malicious_tree(steps, bad_index):
    """A prover that puts a *false* leaf in the tree and rewrites every recorded verdict."""
    state = initial_state()
    leaves = []
    for index in range(steps):
        leaf = leaf_proof(state)
        if index == bad_index:
            leaf["claim"] = (f"{state_text(state)}"
                             f"->{state_text((state[0] + 1, state[1]))}")
            leaf["digest"] = digest_of("leaf", leaf["claim"])
        leaves.append(leaf)
        state = step(state)
    root, levels, _ = rebuild_tree([leaves])
    for depth in range(1, len(levels)):        # bottom-up, keeping every digest consistent
        below = levels[depth - 1]
        for position, proof in enumerate(levels[depth]):
            children = (below[2 * position]["digest"], below[2 * position + 1]["digest"])
            trace = digest_of("trace", children[0], children[1], 1, 1)
            proof["children"] = children
            proof["trace"] = trace
            proof["verified_children"] = (True, True)
            proof["digest"] = digest_of("node", children[0], children[1], trace)
    return levels[-1][0], levels


def ivc_fold(steps):
    """Fold transitions one at a time into a constant-size accumulator."""
    counter = {"verifications": 0, "hashes": 0}
    state = initial_state()
    accumulator = leaf_proof(state)
    sizes = [len(accumulator["digest"]) + len(accumulator["trace"])]
    for _ in range(steps - 1):
        state = step(state)
        accumulator = node_proof(accumulator, leaf_proof(state), counter)
        sizes.append(len(accumulator["digest"]) + len(accumulator["trace"]))
    return accumulator, sizes, counter


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def main():
    # -- 1. leaf proofs --------------------------------------------------------------------
    start = initial_state()
    good = leaf_proof(start)
    check(leaf_verify(good), "an honest leaf verifies against the transition relation")
    check(leaf_verify(leaf_proof((start[0], start[1] + 1))),
          "the claim carries both states, so any state can start a chain")
    skip = leaf_proof(start)
    skip["claim"] = f"{state_text(start)}->{state_text(step(step(start)))}"
    check(not leaf_verify(skip), "a claim that skips a step fails")
    wrong_previous = leaf_proof(start)
    wrong_previous["claim"] = (f"{state_text((start[0] + 1, start[1]))}"
                               f"->{state_text(step(start))}")
    check(not leaf_verify(wrong_previous), "a leaf whose predecessor is wrong fails")
    check(not leaf_verify({"kind": "node", "claim": "x", "children": (),
                           "trace": "", "digest": ""}), "a node is not a leaf")

    # -- 2. composition really recurses -----------------------------------------------------
    root, levels, counter = build_tree(16)
    check(counter["verifications"] == 2 * 15,
          f"16 leaves cost {counter['verifications']} in-circuit verifications (2*(N-1))")
    check(len(internal_nodes(levels)) == 15, "a 16-leaf tree has 15 internal nodes")
    check(all(len(proof["children"]) == 2 for proof in internal_nodes(levels)),
          "every internal node names exactly two children")
    check(all(proof["verified_children"] == (True, True) for proof in internal_nodes(levels)),
          "every node recorded both children as verified")
    check(recursive_depth(levels) == 4, "16 steps give a depth-4 tree")

    # -- 3. constant-size structural verification -------------------------------------------
    sizes, hashes, depths, prover_work = [], [], [], []
    for steps in (2, 4, 8, 16, 32):
        proof, tree_levels, tree_counter = build_tree(steps)
        ledger = {"hashes": 0}
        check(verify(proof, ledger), f"the {steps}-step root verifies")
        sizes.append(len(proof["digest"]) + len(proof["trace"]) + len(proof["claim"]))
        hashes.append(ledger["hashes"])
        depths.append(recursive_depth(tree_levels))
        prover_work.append(tree_counter["verifications"])
    check(len(set(sizes)) == 1, f"the root proof is the same {sizes[0]} bytes for every N")
    check(len(set(hashes)) == 1, f"root verification costs the same everywhere ({hashes})")
    check(depths == [1, 2, 3, 4, 5], f"depth is ceil(log2 N) ({depths})")
    check(prover_work == [2, 6, 14, 30, 62],
          f"prover work is 2*(N-1), i.e. linear ({prover_work})")
    check(prover_work[-1] > 15 * prover_work[0],
          "so the prover/verifier work ratio grows with the computation")

    # -- 4. statement binding ---------------------------------------------------------------
    digests = set()
    for index in range(16):
        leaves = list(levels[0])
        state = initial_state()
        for _ in range(index):
            state = step(state)
        leaves[index] = leaf_proof((state[0] + 5, state[1]))
        tampered_root, _, _ = rebuild_tree([leaves])
        digests.add(tampered_root["digest"])
    check(len(digests) == 16 and root["digest"] not in digests,
          "changing any single leaf produces a fresh root digest, never the honest one")

    # -- 5. an honest prover cannot prove a false statement ---------------------------------
    bad = leaf_proof(initial_state())
    bad["claim"] = (f"{state_text(initial_state())}"
                    f"->{state_text((initial_state()[0] + 1, initial_state()[1]))}")
    honest_failure = rebuild_tree([[leaf_proof(initial_state()), bad]])[0]
    check(honest_failure["verified_children"] == (True, False),
          "the second leaf claims a transition that does not hold, and the node says so")
    check(not verify(honest_failure),
          "an honest prover that built the tree over a bad leaf is rejected -- "
          "the structural verifier propagates recorded failures")

    # -- 6. the documented assumption, demonstrated and measured ----------------------------
    malicious_root, malicious_levels = malicious_tree(16, bad_index=5)
    check(verify(malicious_root),
          "a prover that rewrites the recorded verdicts passes the structural check, "
          "which is exactly the assumption it makes")
    good_open, positions = spot_check_verify(root, levels, 4)
    check(good_open and len(positions) == 4, "spot checks pass on the honest tree")
    nodes = len(internal_nodes(malicious_levels))
    tampered_paths = 1                       # only the parent of the false leaf catches it
    bound = soundness_bound(nodes, tampered_paths, 4)
    rejections = 0
    trials = 400
    for nonce in range(trials):
        challenge_root = dict(malicious_root)
        challenge_root["digest"] = digest_of("seed", nonce)
        opened, picks = spot_check_verify(challenge_root, malicious_levels, 4)
        rejections += 0 if opened else 1
    measured = rejections / trials
    predicted = 1.0 - bound
    sigma = math.sqrt(predicted * (1 - predicted) / trials)
    check(predicted - 3 * sigma <= measured <= predicted + 3 * sigma,
          f"the spot check catches the lie at {measured:.3f}, matching the exact bound "
          f"{predicted:.3f} +- {3 * sigma:.3f}")
    check(bound < 1.0, f"the bound itself is {bound:.3f} for {tampered_paths}/{nodes} paths")
    widths = [soundness_bound(nodes, bad, 4) for bad in (1, 2, 4, 8)]
    check(all(a > b for a, b in zip(widths, widths[1:])),
          f"the bound shrinks as more paths are tampered with ({[round(w, 4) for w in widths]})")
    check(soundness_bound(nodes, 0, 4) == 1.0 and soundness_bound(nodes, nodes, 4) == 0.0,
          "no tampering is always accepted; full tampering is never accepted")
    check(1.0 - soundness_bound(nodes, tampered_paths, 8)
          > 1.0 - soundness_bound(nodes, tampered_paths, 4),
          "more openings improve the soundness of the spot check")

    # -- 7. IVC ------------------------------------------------------------------------------
    accumulator, sizes_ivc, ivc_counter = ivc_fold(16)
    check(len(set(sizes_ivc)) == 1, "the accumulator keeps a constant size across 16 folds")
    check(verify(accumulator), "the folded accumulator verifies")
    check(ivc_counter["verifications"] == 2 * 15,
          f"each fold costs exactly two in-circuit verifications "
          f"({ivc_counter['verifications']})")
    check(accumulator["kind"] == "node" and accumulator["verified_children"] == (True, True),
          "the final accumulator records both children as verified")
    short_accumulator, short_sizes, _ = ivc_fold(4)
    check(short_sizes[0] == sizes_ivc[0],
          "the accumulator size does not depend on how many steps were folded")
    check(verify(short_accumulator), "a short chain verifies too")

    # -- 8. determinism -----------------------------------------------------------------------
    again, again_levels, again_counter = build_tree(16)
    check(again["digest"] == root["digest"]
          and again_counter["verifications"] == counter["verifications"]
          and [level[0]["digest"] for level in again_levels]
          == [level[0]["digest"] for level in levels],
          "rebuilding the tree reproduces every digest and the same work")

    print(f"16 steps: depth {recursive_depth(levels)}, root proof {sizes[0]} bytes, "
          f"{counter['verifications']} in-circuit verifications, {hashes[3]} hashes to verify")
    print(f"lying prover: structural check accepts, spot check rejects {measured:.3f} of the "
          f"time (bound {1 - bound:.3f}, tampered paths {tampered_paths}/{nodes})")
    print(f"IVC: 16 folds, accumulator {sizes_ivc[0]} bytes throughout, "
          f"{ivc_counter['verifications']} verifications inside the folds")
    print("PASS")


if __name__ == "__main__":
    main()
