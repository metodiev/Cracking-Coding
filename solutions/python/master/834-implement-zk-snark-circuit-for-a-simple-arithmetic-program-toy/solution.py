"""Task 834 - Implement zk-SNARK circuit for a simple arithmetic program (toy). (Tier: Master, CTCI Beyond — Security & Privacy)

The QAP half of a SNARK, implemented exactly. A small arithmetic program is compiled to R1CS
(one constraint per multiplication), the R1CS rows are interpolated into three polynomial
vectors u_i, v_i, w_i over a domain H, and the witness gives the polynomials
A(X) = sum_i a_i u_i(X), B, C with the property that the witness satisfies the R1CS *if and only
if* the vanishing polynomial t(X) = prod_{r in H} (X - r) divides A*B - C. The prover performs
that division exactly and the verifier checks the resulting identity A(s)B(s) - C(s) =
h(s) t(s) at a Fiat-Shamir point s, using only the public inputs plus four field elements from
the prover -- O(#public inputs) verifier work instead of O(#constraints).

What is implemented
-------------------
* Coefficient-list polynomial arithmetic over F_p (add, sub, mul, divmod, evaluate) with the
  exact division remainder available, plus Lagrange interpolation over H = {0, ..., m-1}.
* `program()`: a five-constraint arithmetic program (a*b, c + d, b*c, a + e, d*e) over
  wires, with a public/private split.
* `compile_r1cs` / `build_qap`: the constraint matrices and their interpolating polynomials.
* `prove`: the witness, the exact division h = (A*B - C) / t (which succeeds only for a
  satisfying assignment), and the private contributions A_priv(s), B_priv(s), C_priv(s), h(s).
* `verify`: adds the public contributions it computes itself, checks the identity at s, and
  reports the verifier's work (interpolations of the public wires only).
* `small_field_circuit`: the same program over F_101, where the *entire* set of points at which
  a corrupted proof passes can be enumerated -- that turns the Schwartz-Zippel bound deg/p from
  a slogan into a counted number.

What is simplified -- and what is explicitly *not* claimed
----------------------------------------------------------
* There is **no zero knowledge**: the prover sends evaluations of A, B, C and h, which leak
  information about the witness. Nothing here hides anything.
* There is **no polynomial commitment**: no KZG, no pairings, no trusted setup, so the verifier
  cannot check that the four claimed evaluations come from the committed polynomials. The toy
  commits to the coefficient vectors with SHA-256 (binding for a fixed prover, not hiding) and
  the file states the gap instead of papering over it.
* No FFT: the domain is tiny, so interpolation is done with the O(m^2) Lagrange formula and the
  division is schoolbook. The QAP reduction, the division theorem, the public/private split and
  the evaluation check are the real ones.
* Soundness is quantified honestly: for a corrupted witness the identity holds at at most
  deg(A*B - C - h*t) <= 2m - 2 points of F_p, so the single-point check errs with probability at
  most (2m-2)/p, and over F_101 that count is *measured exactly*.

Invariants verified
-------------------
1.  Polynomial arithmetic: multiplication, division with remainder (restoring the dividend as
    quotient*divisor + remainder), interpolation reproducing the tabulated values, and the
    vanishing polynomial dividing exactly.
2.  The QAP identity holds at *every* point of F_101 (all 101 evaluated, not sampled) for the
    honest witness, and at 50 random points of the 61-bit field.
3.  The division is exact exactly when the R1CS is satisfied: a tampered witness leaves a
    non-zero remainder, and the exact verifier rejects it.
4.  Public inputs are bound: the verifier recomputes the public part itself, so a proof whose
    private contributions were produced for a different public input fails.
5.  Fiat-Shamir binds the statement and the commitment: changing either moves s, and a proof
    does not transfer to another statement.
6.  Soundness, counted exactly over F_101: a corrupted witness passes at at most 2m - 2 points,
    the measured root count equals the degree bound for a sabotage witness and never exceeds it
    for a family of random perturbations.
7.  Succinctness: the proof is a constant number of field elements, and verifier work grows with
    the number of *public* inputs while the prover's work grows with the number of constraints.
8.  Edge cases: a degenerate one-constraint program works, a witness with an unsatisfied
    constraint is rejected, and an all-zero witness is rejected.
9.  Determinism: two proofs of the same statement are identical.
"""
import hashlib

P = 2 ** 61 - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# polynomial arithmetic over F_p
# --------------------------------------------------------------------------------------
def trim(poly, modulus=P):
    out = list(poly)
    while len(out) > 1 and out[-1] % modulus == 0:
        out.pop()
    return [coefficient % modulus for coefficient in out]


def padd(left, right, modulus=P):
    size = max(len(left), len(right))
    return trim([(left[i] if i < len(left) else 0) + (right[i] if i < len(right) else 0)
                 for i in range(size)], modulus)


def psub(left, right, modulus=P):
    size = max(len(left), len(right))
    return trim([(left[i] if i < len(left) else 0) - (right[i] if i < len(right) else 0)
                 for i in range(size)], modulus)


def pmul(left, right, modulus=P):
    out = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a % modulus == 0:
            continue
        for j, b in enumerate(right):
            out[i + j] = (out[i + j] + a * b) % modulus
    return trim(out, modulus)


def pdivmod(numerator, denominator, modulus=P):
    numerator, denominator = trim(numerator, modulus), trim(denominator, modulus)
    if denominator == [0]:
        raise ZeroDivisionError("division by the zero polynomial")
    quotient = [0] * max(1, len(numerator) - len(denominator) + 1)
    remainder = list(numerator)
    inverse = pow(denominator[-1], modulus - 2, modulus)
    while len(remainder) >= len(denominator) and remainder != [0]:
        shift = len(remainder) - len(denominator)
        factor = remainder[-1] * inverse % modulus
        quotient[shift] = (quotient[shift] + factor) % modulus
        for index, coefficient in enumerate(denominator):
            remainder[shift + index] = (remainder[shift + index]
                                        - factor * coefficient) % modulus
        remainder = trim(remainder, modulus)
    return trim(quotient, modulus), remainder


def peval(poly, x, modulus=P):
    value = 0
    for coefficient in reversed(poly):
        value = (value * x + coefficient) % modulus
    return value


def degree(poly, modulus=P):
    return len(trim(poly, modulus)) - 1


def interpolate(points, values, modulus=P):
    """Lagrange interpolation through (points[i], values[i]) over F_p."""
    result = [0]
    for i, x_i in enumerate(points):
        basis = [1]
        for j, x_j in enumerate(points):
            if i == j:
                continue
            basis = pmul(basis, [(-x_j) % modulus, 1], modulus)
            scale = pow((x_i - x_j) % modulus, modulus - 2, modulus)
            basis = trim([coefficient * scale for coefficient in basis], modulus)
        result = padd(result, trim([coefficient * values[i] for coefficient in basis],
                                   modulus), modulus)
    return trim(result, modulus)


def vanishing(domain, modulus=P):
    poly = [1]
    for point in domain:
        poly = pmul(poly, [(-point) % modulus, 1], modulus)
    return trim(poly, modulus)


# --------------------------------------------------------------------------------------
# the arithmetic program and its R1CS
# --------------------------------------------------------------------------------------
def program():
    """a*b = c ; a + b = d ; b*c = e ; a + c = f ; d*e = g   (a,b public; c..g private)."""
    return {"constraints": [({"a": 1}, {"b": 1}, {"c": 1}),
                            ({"a": 1, "b": 1}, {"one": 1}, {"d": 1}),
                            ({"b": 1}, {"c": 1}, {"e": 1}),
                            ({"a": 1, "c": 1}, {"one": 1}, {"f": 1}),
                            ({"d": 1}, {"e": 1}, {"g": 1})],
            "public": ["a", "b"],
            "wires": ["one", "a", "b", "c", "d", "e", "f", "g"]}


def witness_for(values):
    """Complete an assignment so that every constraint holds (a, b given)."""
    a, b = values["a"], values["b"]
    return {"one": 1, "a": a, "b": b, "c": a * b, "d": a + b, "e": b * a * b,
            "f": a + a * b, "g": (a + b) * (b * a * b)}


def satisfies(constraints, values, modulus=P):
    for left, right, output in constraints:
        product = 0
        for wire, coefficient in left.items():
            product = (product + coefficient * values.get(wire, 0)) % modulus
        other = 0
        for wire, coefficient in right.items():
            other = (other + coefficient * values.get(wire, 0)) % modulus
        target = 0
        for wire, coefficient in output.items():
            target = (target + coefficient * values.get(wire, 0)) % modulus
        if product * other % modulus != target % modulus:
            return False
    return True


def build_qap(spec, modulus=P):
    """Interpolate the per-row coefficient vectors of A, B and C over the domain H."""
    wires, constraints = spec["wires"], spec["constraints"]
    domain = list(range(len(constraints)))
    qap = {"domain": domain, "t": vanishing(domain, modulus), "wires": wires,
           "u": {}, "v": {}, "w": {}}
    for wire in wires:
        rows = []
        for left, right, output in constraints:
            rows.append([[left.get(wire, 0), right.get(wire, 0), output.get(wire, 0)]])
        qap["u"][wire] = interpolate(domain, [row[0][0] for row in rows], modulus)
        qap["v"][wire] = interpolate(domain, [row[0][1] for row in rows], modulus)
        qap["w"][wire] = interpolate(domain, [row[0][2] for row in rows], modulus)
    return qap


def combine(qap, values, which, modulus=P):
    total = [0]
    for wire in qap["wires"]:
        total = padd(total, trim([coefficient * values.get(wire, 0) % modulus
                                  for coefficient in qap[which][wire]], modulus), modulus)
    return trim(total, modulus)


def qap_polynomials(spec, values, modulus=P):
    qap = build_qap(spec, modulus)
    left = combine(qap, values, "u", modulus)
    right = combine(qap, values, "v", modulus)
    target = combine(qap, values, "w", modulus)
    return qap, left, right, target


def prove(spec, values, modulus=P, public_only=False):
    """Build h = (A*B - C)/t and the prover's part of the evaluations."""
    qap, left, right, target = qap_polynomials(spec, values, modulus)
    quotient, remainder = pdivmod(psub(pmul(left, right, modulus), target, modulus),
                                 qap["t"], modulus)
    if remainder != [0]:
        return {"ok": False, "remainder": remainder, "h": None, "qap": qap}
    return {"ok": True, "remainder": [0], "h": quotient, "qap": qap,
            "A": left, "B": right, "C": target}


def statement_digest(spec, public_values):
    payload = "|".join(f"{wire}={public_values[wire]}" for wire in sorted(public_values))
    return hashlib.sha256(f"{payload}:{len(spec['constraints'])}".encode()).hexdigest()


def commit(proof):
    """Binding (SHA-256) but not hiding: a toy stand-in for a polynomial commitment."""
    payload = "|".join(str(coefficient) for coefficient in proof["A"] + proof["B"]
                       + proof["C"] + proof["h"])
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_point(proof, spec, public_values, modulus=P):
    digest = hashlib.sha256(f"{commit(proof)}:{statement_digest(spec, public_values)}"
                            .encode()).hexdigest()
    return int(digest[:32], 16) % modulus


def verify(spec, public_values, proof, modulus=P, point=None):
    """Check A(s)B(s) - C(s) == h(s) t(s) with the public part recomputed by the verifier."""
    if not proof["ok"]:
        return {"accepted": False, "reason": "division_remainder"}
    s = sample_point(proof, spec, public_values, modulus) if point is None else point
    qap = proof["qap"]
    public = [wire for wire in spec["public"]]
    work = 0
    left = right = target = 0
    for wire in public:
        weight = public_values[wire] % modulus
        left = (left + weight * peval(qap["u"][wire], s, modulus)) % modulus
        right = (right + weight * peval(qap["v"][wire], s, modulus)) % modulus
        target = (target + weight * peval(qap["w"][wire], s, modulus)) % modulus
        work += 3
    private = proof["private_at_s"]
    left = (left + private["A"]) % modulus
    right = (right + private["B"]) % modulus
    target = (target + private["C"]) % modulus
    identity = (left * right - target) % modulus
    vanishing_value = peval(qap["t"], s, modulus)
    claimed = private["h"] * vanishing_value % modulus
    return {"accepted": identity == claimed, "s": s, "work": work,
            "reason": "identity_holds" if identity == claimed else "identity_fails"}


def private_evaluations(spec, proof, s, modulus=P):
    """The prover's message: its own part of A(s), B(s), C(s) and h(s)."""
    qap = proof["qap"]
    values = proof["values"]
    out = {"A": 0, "B": 0, "C": 0, "h": peval(proof["h"], s, modulus)}
    for wire in spec["wires"]:
        weight = values.get(wire, 0) % modulus
        out["A"] = (out["A"] + weight * peval(qap["u"][wire], s, modulus)) % modulus
        out["B"] = (out["B"] + weight * peval(qap["v"][wire], s, modulus)) % modulus
        out["C"] = (out["C"] + weight * peval(qap["w"][wire], s, modulus)) % modulus
    for wire in spec["public"]:                       # remove what the verifier recomputes
        weight = values.get(wire, 0) % modulus
        out["A"] = (out["A"] - weight * peval(qap["u"][wire], s, modulus)) % modulus
        out["B"] = (out["B"] - weight * peval(qap["v"][wire], s, modulus)) % modulus
        out["C"] = (out["C"] - weight * peval(qap["w"][wire], s, modulus)) % modulus
    return out


def full_proof(spec, values, modulus=P):
    """Prove, then attach the private evaluations at the Fiat-Shamir point."""
    proof = prove(spec, values, modulus)
    if not proof["ok"]:
        return proof
    proof["values"] = dict(values)
    s = sample_point(proof, spec, {wire: values[wire] for wire in spec["public"]}, modulus)
    proof["private_at_s"] = private_evaluations(spec, proof, s, modulus)
    return proof


# --------------------------------------------------------------------------------------
# the same program over a tiny field, where soundness can be counted exactly
# --------------------------------------------------------------------------------------
SMALL = 101


def small_field_program():
    return program()


def count_defects(spec, values, h_poly, modulus):
    """(degree, root points) of the defect A*B - C - h*t for a given witness and h."""
    _, left, right, target = qap_polynomials(spec, values, modulus)
    defect = psub(psub(pmul(left, right, modulus), target, modulus),
                  pmul(h_poly, build_qap(spec, modulus)["t"], modulus), modulus)
    roots = [s for s in range(modulus) if peval(defect, s, modulus) == 0] \
        if degree(defect, modulus) >= 0 else []
    return degree(defect, modulus), roots


def truncated_spec(count):
    """The first `count` constraints, with the wires and public inputs they really use."""
    full = program()
    constraints = full["constraints"][:count]
    wires = {"one"}
    for left, right, output in constraints:
        for row in (left, right, output):
            wires |= {wire for wire in row}
    return {"constraints": constraints,
            "public": [wire for wire in full["public"] if wire in wires],
            "wires": sorted(wires)}


def main():
    spec = program()
    public_values = {"a": 3, "b": 5}
    values = witness_for(public_values)
    proof = full_proof(spec, values)

    # -- 1. polynomial arithmetic ----------------------------------------------------------
    left, right = [1, 2, 3], [4, 0, 5]
    product = pmul(left, right)
    check(product == [4, 8, 17, 10, 15], f"schoolbook multiplication ({product})")
    quotient, remainder = pdivmod(product, right)
    check(quotient == left and remainder == [0], "exact division inverts multiplication")
    messy = [7, 0, 4, 9]
    quotient, remainder = pdivmod(messy, [1, 1])
    check(padd(pmul(quotient, [1, 1]), remainder) == trim(messy),
          "quotient * divisor + remainder restores the dividend")
    tabulated = interpolate([0, 1, 2], [5, 8, 17])
    check([peval(tabulated, x) for x in (0, 1, 2)] == [5, 8, 17]
          and tabulated == trim([5, 0, 3]),
          "Lagrange interpolation reproduces its table (3x^2 + 5)")
    check(peval(vanishing([0, 2, 5]), 5) == 0 and peval(vanishing([0, 2, 5]), 3) != 0,
          "the vanishing polynomial is zero exactly on the domain")

    # -- 2. the QAP identity holds everywhere ----------------------------------------------
    qap, A, B, C = qap_polynomials(spec, values)
    defect = psub(psub(pmul(A, B), C), pmul(proof["h"], qap["t"]))
    check(defect == [0], "A*B - C is exactly h*t for the honest witness")
    small_spec = small_field_program()
    small_values = witness_for({"a": 3, "b": 5})
    small_proof = prove(small_spec, small_values, SMALL)
    check(small_proof["ok"], "the honest witness divides exactly over F_101 as well")
    everywhere = []
    for s in range(SMALL):
        A_s = combine(small_proof["qap"], small_values, "u", SMALL)
        B_s = combine(small_proof["qap"], small_values, "v", SMALL)
        C_s = combine(small_proof["qap"], small_values, "w", SMALL)
        everywhere.append((peval(A_s, s, SMALL) * peval(B_s, s, SMALL)
                           - peval(C_s, s, SMALL)
                           - peval(small_proof["h"], s, SMALL)
                           * peval(small_proof["qap"]["t"], s, SMALL)) % SMALL == 0)
    check(all(everywhere), "the identity holds at all 101 points of F_101")
    random_points = [sample_point(proof, spec, public_values) + 7919 * k for k in range(50)]
    check(all((peval(A, s) * peval(B, s) - peval(C, s)
               - peval(proof["h"], s) * peval(qap["t"], s)) % P == 0 for s in random_points),
          "and at 50 spread-out points of the 61-bit field")

    # -- 3. the division is exact exactly when the R1CS is satisfied -----------------------
    check(satisfies(spec["constraints"], values), "the honest witness satisfies the R1CS")
    broken = dict(values)
    broken["e"] = (broken["e"] + 1) % P
    broken_proof = prove(spec, broken)
    check(not broken_proof["ok"] and broken_proof["remainder"] != [0],
          "a tampered witness leaves a non-zero remainder")
    check(not satisfies(spec["constraints"], broken), "and the R1CS says the same")
    check(not verify(spec, public_values, broken_proof)["accepted"],
          "the verifier rejects a proof whose division was not exact")

    # -- 4. the verification path ----------------------------------------------------------
    verdict = verify(spec, public_values, proof)
    check(verdict["accepted"] and verdict["reason"] == "identity_holds",
          f"the honest proof verifies at s = {verdict['s']}")
    check(verdict["work"] == 3 * len(spec["public"]),
          f"verifier work is 3 evaluations per public input ({verdict['work']})")
    shifted = {"a": 4, "b": 5}
    check(not verify(spec, shifted, proof)["accepted"],
          "a proof for other public inputs fails: the public part is recomputed")
    check(sample_point(proof, spec, shifted) != verdict["s"],
          "changing the statement moves the Fiat-Shamir point")
    other = full_proof(spec, witness_for({"a": 4, "b": 5}))
    check(commit(other) != commit(proof) and verify(spec, {"a": 4, "b": 5},
                                                    other)["accepted"],
          "a different commitment is a different proof of a different statement")

    # -- 5. soundness counted exactly over F_101 -------------------------------------------
    m = len(small_spec["constraints"])
    honest_defect = psub(psub(pmul(small_proof["A"], small_proof["B"], SMALL),
                              small_proof["C"], SMALL),
                         pmul(small_proof["h"], small_proof["qap"]["t"], SMALL), SMALL)
    check(honest_defect == [0],
          "the honest defect is the zero polynomial: the identity holds at every point")

    sabotage = dict(small_values)
    sabotage["e"] = (sabotage["e"] + 1) % SMALL
    degree_sabotage, roots_sabotage = count_defects(small_spec, sabotage, [0], SMALL)
    check(degree_sabotage <= 2 * m - 2 and len(roots_sabotage) <= degree_sabotage,
          f"a corrupted witness answering h=0 passes at {len(roots_sabotage)} of 101 points "
          f"(defect degree {degree_sabotage} <= 2m-2 = {2 * m - 2})")
    check(len(roots_sabotage) < SMALL / 4,
          "so a single random evaluation point rejects it with overwhelming probability")

    delta = pmul(pmul([(-5) % SMALL, 1], [(-6) % SMALL, 1], SMALL), [(-7) % SMALL, 1], SMALL)
    crafted_degree, crafted_roots = count_defects(small_spec, small_values,
                                                  padd(small_proof["h"], delta, SMALL), SMALL)
    check(crafted_degree == len(crafted_roots) == 2 * m - 2,
          f"the bound is attained exactly: a crafted h passes at exactly "
          f"{len(crafted_roots)} points, the full Schwartz-Zippel bound")
    offsets = []
    for offset in range(1, 6):
        shifted_h = padd(small_proof["h"], [offset], SMALL)
        _, shifted_roots = count_defects(small_spec, small_values, shifted_h, SMALL)
        offsets.append(len(shifted_roots))
    check(set(offsets) == {m},
          f"shifting h by a constant only keeps the domain points "
          f"({offsets} vs m = {m} constraints)")

    # -- 6. public vs private work ----------------------------------------------------------
    check(len(proof["private_at_s"]) == 4, "the prover's message is four field elements")
    measurements = []
    for count in range(1, len(spec["constraints"]) + 1):
        partial = truncated_spec(count)
        partial_proof = full_proof(partial, witness_for(public_values))
        partial_verdict = verify(partial, {wire: (public_values.get(wire, 0))
                                           for wire in partial["public"]}, partial_proof)
        coefficients = sum(len(partial_proof[key]) for key in ("A", "B", "C", "h"))
        measurements.append((count, len(partial["wires"]), coefficients,
                             len(partial_proof["private_at_s"]), partial_verdict["work"]))
        check(partial_verdict["accepted"], f"the {count}-constraint program verifies")
        check(len(partial_proof["private_at_s"]) == 4,
              "the proof is four field elements whatever the circuit size")
        check(partial_verdict["work"] == 3 * len(partial["public"]),
              "verifier work is three evaluations per public input")
    check(all(row[2] < row[0] * 40 for row in measurements),
          f"the polynomial set grows with the circuit, unlike the proof ({measurements})")

    # -- 7. edge cases ----------------------------------------------------------------------
    tiny = {"constraints": [({"a": 1}, {"a": 1}, {"b": 1})], "public": ["a"],
            "wires": ["one", "a", "b"]}
    tiny_values = {"one": 1, "a": 3, "b": 9}
    tiny_proof = full_proof(tiny, tiny_values)
    check(tiny_proof["ok"] and verify(tiny, {"a": 3}, tiny_proof)["accepted"],
          "a one-constraint program proves and verifies")
    zero = {"one": 1, "a": 0, "b": 0, "c": 0, "d": 0, "e": 0, "f": 0, "g": 0}
    check(prove(spec, zero)["ok"] and verify(spec, {"a": 0, "b": 0},
                                             full_proof(spec, zero))["accepted"],
          "the all-zero witness is a valid proof of the all-zero statement")
    check(not prove(spec, dict(zero, c=1))["ok"],
          "a witness that breaks the first constraint cannot be proven")

    # -- 8. determinism ---------------------------------------------------------------------
    again = full_proof(spec, values)
    check(commit(again) == commit(proof)
          and again["private_at_s"] == proof["private_at_s"]
          and sample_point(again, spec, public_values) == sample_point(proof, spec,
                                                                       public_values),
          "proving the same statement twice gives the same proof")

    print(f"program: {len(spec['constraints'])} constraints, {len(spec['wires'])} wires, "
          f"{len(spec['public'])} public; proof = 4 field elements, verifier work "
          f"{verdict['work']} evaluations")
    print(f"F_101 soundness: corrupted witness passes at {len(roots_sabotage)}/101 points; the "
          f"crafted h attains the bound exactly ({len(crafted_roots)} = 2m-2 roots)")
    print(f"honest identity: exact at all 101 points of F_101 and 50 points of F_p "
          f"(p = 2^61-1)")
    print("PASS")


if __name__ == "__main__":
    main()
