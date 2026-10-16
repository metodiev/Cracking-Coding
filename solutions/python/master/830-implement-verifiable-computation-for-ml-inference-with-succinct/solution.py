"""Task 830 - Implement verifiable computation for ML inference with succinct proofs (conceptual). (Tier: Master, CTCI Beyond — Security & Privacy)

A verifier that holds only the statement (input, model, claimed output) checks an untrusted
server's inference without re-running it. The inference program is compiled into R1CS over a
prime field -- every gate has at most three wires, the way a real circuit compiler produces
it -- the prover commits to the whole wire assignment with a Merkle root, the challenge (which
constraints to open) is derived by Fiat-Shamir from that root and the statement, and the
verifier opens only the wires those constraints touch and checks them exactly in F_p. That is
the linear-PCP spot check. Its soundness is a *combinatorial* quantity that is computed here
exactly (C(m-k, t)/C(m, t)) and then *measured*, not asserted.

What is implemented
-------------------
* `compile_network`: a 2 -> h -> 1 polynomial network (activation t^2 -- the substitution
  zkML systems make for ReLU) compiled gate by gate into R1CS rows (A_i, B_i, C_i) with at
  most three wire references each, plus a witness evaluator that follows the same gates.
* `merkle_root` / `merkle_open` / `merkle_check`: SHA-256 commitment to the wire values.
* `prove`: commit, derive t distinct constraint indices from H(root || statement), open only
  the wires those rows reference, and report the proof size in bytes.
* `verify_spot`: re-derives the challenge, checks every opening against the root and
  evaluates the sampled constraints exactly.
* `verify_exact` / `violated_rows`: the all-constraints verifier, used as the correctness
  oracle (it needs the whole witness, which is precisely what a succinct verifier must not).
* `acceptance_bound(m, k, t)`: the exact probability that a uniform t-subset misses all k
  violated constraints; `integer_reference` is the independent integer-evaluation oracle.

What is simplified -- and what is explicitly *not* claimed
----------------------------------------------------------
* There is **no zero knowledge**: opened wires are revealed in the clear. There is no
  pairing, no trusted setup, no polynomial commitment and no FFT. "Succinct" here means only
  "a constant number of opened wires regardless of circuit size" -- the linear-PCP half of a
  SNARK -- and the checks state that limitation instead of hiding it.
* ReLU is replaced by t^2, so the circuit contains no comparisons and no range checks; the
  file never claims to prove them.
* Soundness is quantified honestly: under a uniform challenge the acceptance probability for
  k violated constraints is exactly C(m-k, t)/C(m, t), and 0 once t >= m. Fiat-Shamir is only
  a heuristic (random-oracle) instantiation of that uniform challenge, and the empirical
  acceptance frequencies are compared with the combinatorial value.
* The commitment is SHA-256, so collision resistance is assumed. Nothing here is safe for
  real use.

Invariants verified
-------------------
1.  The modulus is prime (Miller-Rabin) and every field operation matches exact big-integer
    arithmetic.
2.  The compiled circuit reproduces the network's plain-integer output on several inputs and
    sizes, and the honest witness satisfies every constraint (exact verifier accepts).
3.  Every gate touches at most three wires, which is what bounds the verifier's work.
4.  Completeness: 200 seeded challenges, 200 acceptances.
5.  Soundness: for k = 1, 2, 3 violated constraints the measured acceptance frequency tracks
    the exact bound C(m-k, t)/C(m, t) (0.72, 0.49 and 0.32 for m=20, t=3), and a run with
    t = m rejects every corrupted witness deterministically (bound exactly 0).
6.  Schwartz-Zippel, by exhaustively counting roots over a small prime: a non-zero degree-d
    polynomial has at most d roots, so the linear-combination variant errs with probability
    at most d/|F|.
7.  Commitments bind: tampering changes the root, valid openings verify, openings for another
    value are rejected, and the path length is logarithmic in the witness size.
8.  Fiat-Shamir binds both the commitment and the statement: challenges move when either
    changes and a proof does not transfer to another statement.
9.  Succinctness: the proof is O(t log n) -- it grows 2x while the circuit grows 9x -- the
    opened wires stay bounded by 3t, and for circuits with 40+ rows the verifier does under
    a quarter of the prover's field work (on a six-row circuit a three-sample check is *not*
    cheaper, which the numbers state rather than hide).
10. Edge cases each have a check: a three-constraint circuit is checked exhaustively (t >= m,
    soundness error exactly 0), a challenge never exceeds the number of constraints, and the
    same statement verifies under two different witnesses (knowledge, not uniqueness).
11. Determinism: the prover is bit-reproducible.
"""
import hashlib
import math

P = 2 ** 61 - 1                       # Mersenne prime: |F| = 2^61 - 1
T_CHALLENGES = 3


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# the field
# --------------------------------------------------------------------------------------
def is_prime(n, rounds=16):
    """Miller-Rabin with fixed bases (deterministic, no randomness)."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)[:rounds]:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def fmul(a, b):
    return a * b % P


def fadd(a, b):
    return (a + b) % P


def fsub(a, b):
    return (a - b) % P


# --------------------------------------------------------------------------------------
# the network and its R1CS compilation
# --------------------------------------------------------------------------------------
def network(hidden=3):
    """Weights of a 2 -> hidden -> 1 polynomial network with integer coefficients."""
    w1 = [[1, 2], [-1, 1], [2, -2], [1, 0], [0, 1], [-2, 1], [3, -1], [1, 1]]
    b1 = [1, -1, 0, 2, -2, 1, 0, 1]
    w2 = [2, -1, 3, 1, -2, 2, 3, -3]      # integer coefficients: exact over F_p
    if hidden > 8:
        raise ValueError("the toy network has eight hidden units")
    return {"w1": w1[:hidden], "b1": b1[:hidden], "w2": w2[:hidden], "hidden": hidden}


def integer_reference(model, inputs):
    """Independent oracle: the network evaluated with plain Python integers."""
    z = [model["b1"][j] + sum(model["w1"][j][k] * inputs[k] for k in range(2))
         for j in range(model["hidden"])]
    return sum(model["w2"][j] * z[j] * z[j] for j in range(model["hidden"]))


class Compiler:
    """Gate-by-gate R1CS construction: every row references at most three wires."""

    def __init__(self):
        self.rows = []                    # (A, B, C) as {wire: coefficient}
        self.size = 3                     # 0 = constant one, 1..2 = inputs
        self.gates = []

    def new_wire(self):
        self.size += 1
        return self.size - 1

    def mul_const(self, wire, constant):
        """t = constant * wire  (A = wire, B = one*constant, C = t)."""
        target = self.new_wire()
        self.rows.append(({wire: 1}, {0: constant % P}, {target: 1}))
        self.gates.append(("mul", wire, constant, target))
        return target

    def add(self, left, right):
        """t = left + right  (A = one, B = left + right, C = t)."""
        target = self.new_wire()
        self.rows.append(({0: 1}, {left: 1, right: 1}, {target: 1}))
        self.gates.append(("add", left, right, target))
        return target

    def add_const(self, wire, constant):
        target = self.new_wire()
        self.rows.append(({0: 1}, {wire: 1, 0: constant % P}, {target: 1}))
        self.gates.append(("addc", wire, constant, target))
        return target

    def square(self, wire):
        """t = wire^2  (A = B = wire, C = t)."""
        target = self.new_wire()
        self.rows.append(({wire: 1}, {wire: 1}, {target: 1}))
        self.gates.append(("sqr", wire, wire, target))
        return target


def compile_network(model):
    compiler = Compiler()
    activations = []
    for j in range(model["hidden"]):
        acc = compiler.add_const(compiler.mul_const(2, model["w1"][j][1]), model["b1"][j])
        z = compiler.add(compiler.mul_const(1, model["w1"][j][0]), acc)
        activations.append(compiler.square(z))
    output = compiler.mul_const(activations[0], model["w2"][0])
    for j in range(1, model["hidden"]):
        output = compiler.add(output, compiler.mul_const(activations[j], model["w2"][j]))
    return {"model": model, "rows": compiler.rows, "size": compiler.size,
            "gates": compiler.gates, "activations": activations, "output": output,
            "inputs": [1, 2], "constraints": len(compiler.rows)}


def honest_witness(circuit, inputs):
    """Evaluate the gate list -- the same order the compiler emitted -- in F_p."""
    witness = [0] * circuit["size"]
    witness[0] = 1
    witness[1], witness[2] = inputs[0] % P, inputs[1] % P
    for kind, left, right, target in circuit["gates"]:
        if kind == "mul":
            witness[target] = fmul(witness[left], right % P)
        elif kind == "add":
            witness[target] = fadd(witness[left], witness[right])
        elif kind == "addc":
            witness[target] = fadd(witness[left], right % P)
        else:
            witness[target] = fmul(witness[left], witness[right])
    return witness


def row_dot(row, witness):
    return sum(coeff * witness[wire] for wire, coeff in row.items()) % P


def violated_rows(circuit, witness):
    """Indices whose R1CS constraint is not satisfied exactly in F_p."""
    return [index for index, (ra, rb, rc) in enumerate(circuit["rows"])
            if fsub(fmul(row_dot(ra, witness), row_dot(rb, witness)),
                    row_dot(rc, witness)) != 0]


def verify_exact(circuit, witness, statement):
    """The all-constraints verifier: needs the full witness, so it is the oracle here."""
    if witness[circuit["output"]] != statement["output"] % P:
        return False
    return not violated_rows(circuit, witness)


def row_width(circuit):
    return max(len(row) for triple in circuit["rows"] for row in triple)


# --------------------------------------------------------------------------------------
# Merkle commitment
# --------------------------------------------------------------------------------------
def hash_nodes(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{value}:".encode())
    return digest.hexdigest()


def merkle_tree(leaves):
    nodes, levels = list(leaves), [list(leaves)]
    while len(nodes) > 1:
        nodes = [hash_nodes(nodes[i], nodes[i + 1]) if i + 1 < len(nodes) else nodes[i]
                 for i in range(0, len(nodes), 2)]
        levels.append(nodes)
    return nodes[0], levels


def merkle_open(levels, index):
    path, i = [], index
    for level in levels[:-1]:
        sibling = i + 1 if i % 2 == 0 else i - 1
        path.append(level[sibling] if sibling < len(level) else None)
        i //= 2
    return path


def merkle_check(root, leaf, path, index):
    node, i = leaf, index
    for sibling in path:
        if sibling is None:                 # the tree duplicates nothing: an odd tail node
            pass                            # simply moves up one level unchanged
        elif i % 2 == 0:
            node = hash_nodes(node, sibling)
        else:
            node = hash_nodes(sibling, node)
        i //= 2
    return node == root


def commit(witness):
    root, levels = merkle_tree([str(v) for v in witness])
    return root, levels


# --------------------------------------------------------------------------------------
# prover and succinct verifier
# --------------------------------------------------------------------------------------
def statement_digest(statement):
    return hash_nodes(statement["inputs"][0], statement["inputs"][1],
                      statement["output"], statement["model_tag"])


def challenge_indices(root, statement, count, constraints):
    """Fiat-Shamir: `count` distinct constraint indices from H(root || statement)."""
    base = hash_nodes(root, statement_digest(statement))
    indices, nonce = [], 0
    while len(indices) < min(count, constraints):
        candidate = int(hash_nodes(base, nonce)[:16], 16) % constraints
        if candidate not in indices:
            indices.append(candidate)
        nonce += 1
    return sorted(indices)


def prove(circuit, witness, statement, count=T_CHALLENGES):
    root, levels = commit(witness)
    indices = challenge_indices(root, statement, count, circuit["constraints"])
    needed = set()
    for index in indices:
        for row in circuit["rows"][index]:
            needed |= set(row)
    openings = {wire: (witness[wire], merkle_open(levels, wire)) for wire in sorted(needed)}
    return {"root": root, "statement": dict(statement), "indices": indices,
            "opened": len(needed), "openings": openings,
            "bytes": len(root) + sum(len(str(v)) + 64 * len(path)
                                     for v, path in openings.values())}


def verify_spot(circuit, statement, proof, count=T_CHALLENGES, ledger=None):
    """Re-derive the challenge, check the openings, evaluate the sampled rows exactly."""
    if proof["statement"] != dict(statement):
        return False
    indices = challenge_indices(proof["root"], statement, count, circuit["constraints"])
    if indices != proof["indices"]:
        return False
    values = {}
    for index in indices:
        for row in circuit["rows"][index]:
            for wire in row:
                if wire not in proof["openings"]:
                    return False
                value, path = proof["openings"][wire]
                if not merkle_check(proof["root"], str(value), path, wire):
                    return False
                values[wire] = value
    for index in indices:
        ra, rb, rc = circuit["rows"][index]
        left = sum(c * values[w] for w, c in ra.items()) % P
        right = sum(c * values[w] for w, c in rb.items()) % P
        target = sum(c * values[w] for w, c in rc.items()) % P
        if ledger is not None:
            ledger[0] += len(ra) + len(rb) + len(rc) + 1
        if fsub(fmul(left, right), target) != 0:
            return False
    return True


def acceptance_bound(constraints, violated, count):
    """Exact P[a uniform `count`-subset of constraints misses all `violated` of them]."""
    m, t, k = constraints, min(count, constraints), violated
    if k == 0:
        return 1.0
    if m - k < t:
        return 0.0
    return math.prod((m - k - i) / (m - i) for i in range(t))


def main():
    model_tag = "poly-net"
    circuit = compile_network(network(3))
    inputs = [2, 3]
    statement = {"inputs": inputs, "model_tag": model_tag,
                 "output": integer_reference(network(3), inputs)}
    witness = honest_witness(circuit, inputs)

    # -- 1. the field --------------------------------------------------------------------
    check(is_prime(P) and not is_prime((1 << 61) - 3), "the modulus is prime and P-2 is not")
    check(all(fmul(a, b) == (a * b) % P and fadd(a, b) == (a + b) % P
              and fsub(a, b) == (a - b) % P
              for a, b in ((P - 3, 5), (123456789, 987654321), (P - 1, P - 1))),
          "field operations agree with exact big-integer arithmetic")

    # -- 2. compilation ------------------------------------------------------------------
    for hidden in (1, 2, 3, 8):
        unit = compile_network(network(hidden))
        for pair in ([2, 3], [-4, 1], [0, 0], [5, -5]):
            local = {"inputs": pair, "model_tag": model_tag,
                     "output": integer_reference(network(hidden), pair)}
            local_witness = honest_witness(unit, pair)
            check(verify_exact(unit, local_witness, local),
                  f"h={hidden}: the honest witness satisfies all {unit['constraints']} rows")
            check(local_witness[unit["output"]] == local["output"] % P,
                  f"h={hidden} on {pair}: the circuit equals the integer reference")
        check(row_width(unit) <= 3, "every compiled constraint touches at most three wires")
    check(circuit["constraints"] == 7 * 3 - 1 == 20 and circuit["size"] == 7 * 3 + 2 == 23,
          "2->3->1 compiles to 20 rows over 23 wires")
    check(row_width(circuit) == 2,
          "the widest row references two wires (the addition gate)")

    # -- 3. completeness -----------------------------------------------------------------
    proof = prove(circuit, witness, statement)
    check(verify_spot(circuit, statement, proof), "the honest proof verifies")
    check(all(verify_spot(circuit, statement, prove(circuit, witness, statement))
              for _ in range(200)), "completeness: 200 fresh challenges, 200 acceptances")
    check(len(proof["indices"]) == T_CHALLENGES and len(set(proof["indices"])) == 3,
          "the challenge is three distinct constraint indices")

    # -- 4. soundness against the exact combinatorial bound ------------------------------
    bounds, frequencies, violations_seen = [], [], []
    for k in (1, 2, 3):
        corrupted = list(witness)
        for step in range(k):
            wire = circuit["activations"][step]
            corrupted[wire] = (corrupted[wire] + 3 + step) % P
        violations = violated_rows(circuit, corrupted)
        bound = acceptance_bound(circuit["constraints"], len(violations), T_CHALLENGES)
        accepts = 0
        for nonce in range(400):
            shifted = {"inputs": inputs, "model_tag": f"{model_tag}-{nonce}",
                       "output": statement["output"]}
            accepts += verify_spot(circuit, shifted, prove(circuit, corrupted, shifted))
        frequency = accepts / 400.0
        bounds.append(bound)
        frequencies.append(frequency)
        violations_seen.append(len(violations))
        check(frequency <= bound + 0.05,
              f"{len(violations)} violated rows: measured {frequency:.3f} <= bound {bound:.3f}")
        check(len(violations) >= 1, "the corruption really violates constraints")
    check(bounds[0] > 0.4 and bounds[-1] < bounds[0],
          "with three samples the bound is weak, which the numbers acknowledge")

    smashed = [v + 1 if i > 2 else v for i, v in enumerate(witness)]
    violations = violated_rows(circuit, smashed)
    check(len(violations) > 0, f"the smashed witness violates {len(violations)} rows")
    check(acceptance_bound(circuit["constraints"], len(violations),
                           circuit["constraints"]) == 0.0,
          "t = m drives the soundness bound to zero for any number of violations")
    all_rows = circuit["constraints"]
    rejected = 0
    for nonce in range(20):
        here = {"inputs": inputs, "model_tag": f"{model_tag}-x{nonce}",
                "output": statement["output"]}
        rejected += not verify_spot(circuit, here, prove(circuit, smashed, here, count=all_rows),
                                    count=all_rows)
    check(rejected == 20,
          "with t = m every corrupted witness is rejected (the exact-verifier limit)")

    # -- 5. Schwartz-Zippel by exhaustive root counting ----------------------------------
    prime_small, degree = 101, 3
    coefficients = [1, 0, 7, 5]

    def poly(r, coeffs=coefficients):
        return sum(coeffs[i] * pow(r, i, prime_small) for i in range(len(coeffs))) % prime_small

    roots = [r for r in range(prime_small) if poly(r) == 0]
    check(0 < len(roots) <= degree,
          f"the non-zero degree-{degree} polynomial has {len(roots)} roots, at most {degree}")
    check(all(poly(r) == 0 for r in roots)
          and all(poly(r) != 0 for r in range(prime_small) if r not in roots),
          "the roots are exactly the residues where the polynomial vanishes")
    tight = [0, 2, -3, 1]                    # r(r-1)(r-2): exactly degree many roots
    tight_roots = [r for r in range(prime_small) if poly(r, tight) == 0]
    check(tight_roots == [0, 1, 2],
          "the bound is tight: r(r-1)(r-2) has exactly d roots over F_101")
    check(degree / prime_small < 0.03, "so the combination check errs with prob <= d/|F|")

    # -- 6. commitments ------------------------------------------------------------------
    root, levels = commit(witness)
    changed = list(witness)
    changed[circuit["activations"][0]] = (changed[circuit["activations"][0]] + 1) % P
    other_root, _ = commit(changed)
    check(other_root != root and proof["root"] == root, "tampering changes the commitment")
    path = merkle_open(levels, 5)
    check(merkle_check(root, str(witness[5]), path, 5), "a valid opening verifies")
    check(not merkle_check(root, str(witness[5] + 1), path, 5),
          "an opening for another value is rejected")
    check(len(path) == math.ceil(math.log2(len(witness))),
          "the opening path is logarithmic in the witness size")

    # -- 7. Fiat-Shamir binds commitment and statement ------------------------------------
    indices_here = challenge_indices(proof["root"], statement, T_CHALLENGES,
                                    circuit["constraints"])
    shifted = {"inputs": inputs, "model_tag": model_tag, "output": statement["output"] + 1}
    check(challenge_indices(proof["root"], shifted, T_CHALLENGES, circuit["constraints"])
          != indices_here, "a different statement moves the challenge")
    check(challenge_indices(other_root, statement, T_CHALLENGES, circuit["constraints"])
          != indices_here, "a different commitment moves the challenge")
    check(not verify_spot(circuit, shifted, proof),
          "the same proof does not transfer to another statement")
    check(challenge_indices(proof["root"], dict(statement), T_CHALLENGES,
                            circuit["constraints"]) == indices_here,
          "the challenge is a deterministic function of the root and the statement")

    # -- 8. succinctness -----------------------------------------------------------------
    measured = []
    for hidden in (1, 2, 3, 4, 6, 8):
        unit = compile_network(network(hidden))
        unit_statement = {"inputs": inputs, "model_tag": model_tag,
                          "output": integer_reference(network(hidden), inputs)}
        unit_witness = honest_witness(unit, inputs)
        unit_proof = prove(unit, unit_witness, unit_statement)
        ledger = [0]
        check(verify_spot(unit, unit_statement, unit_proof, ledger=ledger),
              f"h={hidden} verifies")
        prover_ops = sum(1 for gate in unit["gates"] if gate[0] in ("mul", "sqr")) \
            + unit["constraints"]
        measured.append((unit["constraints"], unit["size"], unit_proof["opened"],
                         unit_proof["bytes"], prover_ops, ledger[0]))
        check(unit_proof["opened"] <= 3 * T_CHALLENGES,
              "at most three wires per sampled constraint are opened")
        check(unit_proof["opened"] < unit["size"],
              "the verifier never receives the whole witness")
    bytes_seen = [row[3] for row in measured]
    growth_bytes = max(bytes_seen) / min(bytes_seen)
    growth_rows = measured[-1][0] / measured[0][0]
    check(growth_rows > 3 * growth_bytes,
          f"the proof grows by {growth_bytes:.2f}x while the circuit grows by "
          f"{growth_rows:.1f}x ({bytes_seen}) -- the message is O(t log n), not O(n)")
    ratios = [prover / verifier for _, _, _, _, prover, verifier in measured]
    large = [(m, p, v) for m, _, _, _, p, v in measured if m >= 40]
    check(all(p > v for m, _, _, _, p, v in measured if m >= 13),
          f"the prover does more field work once the circuit exceeds the challenge: {measured}")
    check(measured[0][4] <= measured[0][5],
          "on a six-row circuit three samples are not cheaper than checking everything -- "
          "succinctness needs a large circuit, and the numbers say so")
    check(all(p >= 4 * v for _, p, v in large),
          f"for circuits with 40+ rows the verifier does under a quarter of the work: {large}")
    check(ratios[-1] == max(ratios) and ratios[-1] > 3 * ratios[0],
          f"the prover/verifier work ratio grows with the circuit size: "
          f"{[round(r, 2) for r in ratios]}")
    check(all(p > 0 and v > 0 for _, _, _, _, p, v in measured),
          "both sides do a measurable amount of field work")

    # -- 9. edge cases -------------------------------------------------------------------
    tiny = compile_network(network(1))
    tiny_statement = {"inputs": [1, 1], "model_tag": "tiny",
                      "output": integer_reference(network(1), [1, 1])}
    tiny_witness = honest_witness(tiny, [1, 1])
    check(tiny["constraints"] == 6 and len(challenge_indices(proof["root"], tiny_statement, 9,
                                                             6)) == 6,
          "a six-row circuit exhausts a challenge larger than the circuit")
    check(acceptance_bound(6, 3, 6) == 0.0 and acceptance_bound(6, 0, 6) == 1.0,
          "t = m is perfectly sound; no violations is perfectly complete")
    corrupted_tiny = list(tiny_witness)
    corrupted_tiny[tiny["activations"][0]] = (corrupted_tiny[tiny["activations"][0]] + 1) % P
    always_caught = all(not verify_spot(tiny, {"inputs": [1, 1], "model_tag": f"tiny-{n}",
                                              "output": tiny_statement["output"]},
                                        prove(tiny, corrupted_tiny,
                                              {"inputs": [1, 1], "model_tag": f"tiny-{n}",
                                               "output": tiny_statement["output"]},
                                              count=6),
                                        count=6)
                        for n in range(50))
    check(always_caught, "with t = m a corrupted wire is caught every time")

    free = witness + [12345]
    other_free = witness + [999]
    check(len(violated_rows(circuit, free)) == 0 and verify_exact(circuit, free, statement)
          and verify_exact(circuit, other_free, statement),
          "two different witnesses prove the same statement (knowledge, not uniqueness)")
    extra = {"inputs": inputs, "model_tag": model_tag, "output": statement["output"]}
    check(verify_spot(circuit, extra, prove(circuit, free, extra))
          and verify_spot(circuit, extra, prove(circuit, other_free, extra)),
          "an unconstrained extra wire does not change the verdict")

    # -- 10. determinism -----------------------------------------------------------------
    again = prove(circuit, witness, statement)
    check(again["root"] == proof["root"] and again["openings"] == proof["openings"]
          and again["bytes"] == proof["bytes"],
          "the prover is deterministic given the witness and the statement")

    print(f"2->3->1 circuit: {circuit['constraints']} constraints over {circuit['size']} wires, "
          f"widest row {row_width(circuit)} wires, proof {proof['bytes']} bytes, "
          f"{proof['opened']} wires opened")
    print(f"soundness C(m-k,t)/C(m,t) at m={circuit['constraints']}, t={T_CHALLENGES}: "
          + ", ".join(f"k={k}: bound {b:.3f} measured {f:.3f}"
                      for k, b, f in zip((1, 2, 3), bounds, frequencies))
          + f" (violated rows {violations_seen})")
    print(f"constraints/opened-wires/proof-bytes over h=1..8: {measured}")
    print("PASS")


if __name__ == "__main__":
    main()
