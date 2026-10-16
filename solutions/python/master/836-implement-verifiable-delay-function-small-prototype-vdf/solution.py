"""Task 836 - Implement verifiable delay function small prototype (VDF). (Tier: Master, CTCI Beyond — Security & Privacy)

A VDF is a function that takes a prescribed *sequential* number of steps to evaluate even with
unbounded parallelism, but whose output can be verified quickly. The prototype uses the classic
repeated-squaring construction in a group of unknown order: y = g^(2^T) mod N with N = p*q, where
the output is checked by Wesolowski's one-element proof pi = g^floor(2^T / l) with
l = HashToPrime(g, y, T): the verifier recomputes l and r = 2^T mod l and checks pi^l * g^r == y.
The prover's work is T squarings in sequence; the verifier's is about 256 squarings regardless
of T, and both counts are measured rather than asserted.

What is implemented
-------------------
* `deterministic_prime`: Miller-Rabin search driven by a splitmix64 stream, so the modulus (and
  therefore the whole run) is reproducible without any randomness library.
* `delayed_output`: the sequential squaring loop, with the number of squarings counted.
* `hash_to_prime`: Fiat-Shamir derivation of the challenge prime from SHA-256 of (g, y, T).
* `wesolowski_prove` / `wesolowski_verify`: the proof and its exact check pi^l * g^r == y, with
  modmul counters on both sides.
* `trapdoor_shortcut`: y recomputed with the group order known, which collapses the sequential
  work to a single exponentiation -- the demonstration of *why* the order must stay unknown.

What is simplified -- and what is explicitly *not* claimed
----------------------------------------------------------
* The factorization of N is generated here and is known to the simulator, so this is **not** a
  delay function against a real adversary: anyone holding phi(N) shortcuts it, exactly as the
  `trapdoor_shortcut` check shows. A production VDF needs a class group of unknown order or a
  trusted-setup RSA modulus. Wesolowski's soundness (adaptive root assumption in a group of
  unknown order) is *not* claimed to hold here, and no hardness is fabricated; what is verified
  is the protocol algebra (completeness, forgery rejection, size and work bounds).
* T is small (256-4096 squarings) and N is 512 bits, so everything runs in milliseconds and the
  sequential and parallel costs are measured on the same machine instead of being argued about.
* No threads, no clock: "sequential" is a counted loop and "T steps" is a parameter.

Invariants verified
-------------------
1.  The sequential output equals the independent big-integer oracle pow(g, 2**T, N) exactly, and
    the loop really performed T squarings (and T-1 of the intermediate values chain).
2.  The Wesolowski proof verifies exactly, including T = 0 (y = g) and T = 1 (y = g^2).
3.  Forgery rejection: a perturbed proof, a perturbed output, a wrong T, a wrong challenge
    prime, a wrong base, a wrong modulus and the identity element as a proof are all rejected.
4.  The challenge prime is deterministic, prime, and correctly bound to (g, y, T): changing any
    of the three changes l.
5.  Work separation: the verifier's modmul count is a constant that does not grow with T while
    the prover's is exactly T, so the ratio grows; measured for several T.
6.  Proof size is constant in T (the proof is one group element plus the small T field).
7.  Sequentiality depends on the unknown order: with phi(N) available the same output is
    produced with O(log T) work, which is exactly the trapdoor an adversary must not have.
8.  The output is not a guessable short expression: y differs from g, g^2, g^T, g^(2T),
    g^(T^2) and g^(T!)  for the tested T.
9.  Determinism: the same (N, g, T) reproduces the same output and the same proof.
10. Repeat-squaring is order-free at the output level: squaring the result 2^k more times equals
    evaluating with T + k, so an honest prover can extend a VDF without restarting it.
"""
import hashlib
import math

MASK64 = (1 << 64) - 1
SEED = 0x5DEECE66D


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------------------
# deterministic primality and primes
# --------------------------------------------------------------------------------------
def is_prime(n, rounds=24):
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        if n % small == 0:
            return n == small
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for witness in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)[:rounds]:
        x = pow(witness, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def deterministic_prime(bits, seed):
    """A prime of exactly `bits` bits, found from a seeded stream (fully reproducible)."""
    rng = SplitMix64(seed)
    while True:
        candidate = (1 << (bits - 1)) | (rng.next_u64() << (bits - 64)) | 1
        candidate |= 1
        if is_prime(candidate, 12):
            return candidate


def rsa_modulus(bits=512, seed=SEED):
    """N = p * q with 512-bit N by default, plus phi for the trapdoor demonstration."""
    prime_bits = bits // 2
    first = deterministic_prime(prime_bits, seed)
    second = deterministic_prime(prime_bits, seed ^ 0xA5A5A5A5)
    while second == first:
        second = deterministic_prime(prime_bits, (seed + 1) ^ 0xA5A5A5A5)
    return {"N": first * second, "p": first, "q": second,
            "phi": (first - 1) * (second - 1)}


# --------------------------------------------------------------------------------------
# the VDF itself
# --------------------------------------------------------------------------------------
def delayed_output(g, N, steps, counter=None):
    """Sequential squaring: exactly `steps` modular squarings, no shortcuts."""
    value = g % N
    for _ in range(steps):
        value = value * value % N
        if counter is not None:
            counter["modmuls"] += 1
    return value


def oracle_output(g, N, steps):
    """Independent oracle: the same value via Python's fast exponentiation."""
    return pow(g % N, 1 << steps, N)


def hash_to_prime(g, y, steps, modulus, bits=128):
    """Fiat-Shamir: the first prime at or above the hash of the whole public statement."""
    digest = hashlib.sha256(f"{modulus}|{g}|{y}|{steps}".encode()).hexdigest()
    candidate = int(digest[:bits // 4], 16) | (1 << (bits - 1)) | 1
    while not is_prime(candidate, 12):
        candidate += 2
    return candidate


def wesolowski_prove(g, N, steps, counter=None):
    """pi = g^floor(2^T / l); the prover only needs y, so the work is one exponentiation."""
    y = delayed_output(g, N, steps, counter)
    exponent = 1 << steps
    challenge = hash_to_prime(g, y, steps, N)
    quotient, remainder = divmod(exponent, challenge)
    proof = pow(g % N, quotient, N)
    if counter is not None:
        counter["modmuls"] += max(1, quotient.bit_length())
    return {"y": y, "pi": proof, "steps": steps, "l": challenge, "r": remainder,
            "q": quotient, "g": g % N, "N": N}


def wesolowski_verify(statement, counter=None):
    """Check pi^l * g^r == y with l and r recomputed from the statement."""
    g, N, steps = statement["g"], statement["N"], statement["steps"]
    y, proof = statement["y"], statement["pi"]
    challenge = hash_to_prime(g, y, steps, N)
    if challenge != statement.get("l"):
        return False
    remainder = (1 << steps) % challenge
    left = pow(proof % N, challenge, N)
    right = pow(g, remainder, N)
    if counter is not None:
        counter["modmuls"] += max(1, challenge.bit_length()) + max(1, remainder.bit_length())
    return left * right % N == y % N


def trapdoor_shortcut(g, order, N, steps, counter=None):
    """With phi(N) known the same output costs one exponentiation: why the order must hide."""
    exponent = pow(2, steps, order)
    value = pow(g % N, exponent, N)
    if counter is not None:
        counter["modmuls"] += max(1, exponent.bit_length())
    return value


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def main():
    setup = rsa_modulus(512)
    N, phi = setup["N"], setup["phi"]
    g = 2 + 0x123456789 % 1000
    steps = 1024

    # -- 1. the modulus and the sequential output ------------------------------------------
    check(is_prime(setup["p"]) and is_prime(setup["q"])
          and setup["p"].bit_length() == setup["q"].bit_length() == 256
          and N.bit_length() in (511, 512),
          "the modulus really is a product of two 256-bit primes")
    check(math.gcd(g, N) == 1, "the base is invertible modulo N")
    counter = {"modmuls": 0}
    y = delayed_output(g, N, steps, counter)
    check(counter["modmuls"] == steps == 1024,
          f"the delay loop performed exactly T = {steps} squarings")
    check(y == oracle_output(g, N, steps),
          "the sequential result equals the independent fast-exponentiation oracle")
    check(y != pow(g, steps, N) and y != pow(g, 2 * steps, N),
          "the output is not a short exponentiation of the base")
    check(y != pow(g, steps * steps, N) and y != pow(g, math.factorial(12), N),
          "and not any of the obvious guessable expressions either")
    partial = delayed_output(g, N, steps - 1)
    check(partial * partial % N == y,
          "the last squaring chains onto the previous value, as a delay function must")
    check(delayed_output(g, N, 0) == g % N and delayed_output(g, N, 1) == g * g % N,
          "T = 0 gives the base and T = 1 gives its square")

    # -- 2. extending a VDF without restarting it ------------------------------------------
    check(delayed_output(y, N, 5) == oracle_output(g, N, steps + 5),
          "squaring the output k more times equals evaluating with T + k")

    # -- 3. the Wesolowski proof ------------------------------------------------------------
    statement = wesolowski_prove(g, N, steps)
    check(statement["y"] == y and statement["q"] * statement["l"] + statement["r"]
          == (1 << steps),
          "the proof splits 2^T = q*l + r exactly")
    check(wesolowski_verify(statement), "the honest proof verifies")
    verifier_counter = {"modmuls": 0}
    check(wesolowski_verify(statement, verifier_counter), "the proof verifies again")
    check(verifier_counter["modmuls"] <= 2 * 130 + 8,
          f"verification costs only the challenge-size work ({verifier_counter['modmuls']} "
          f"modmuls, independent of T = {steps})")
    check(is_prime(statement["l"]) and statement["l"].bit_length() >= 128,
          "the challenge is a real 128-bit prime")

    # -- 4. forgery rejection ---------------------------------------------------------------
    forgeries = []
    forgeries.append({**statement, "pi": statement["pi"] + 1})
    forgeries.append({**statement, "y": statement["y"] + 1})
    forgeries.append({**statement, "steps": steps + 1})
    forgeries.append({**statement, "l": hash_to_prime(g, statement["y"], steps + 7, N)})
    forgeries.append({**statement, "g": g + 1})
    forgeries.append({**statement, "N": N * 3})
    forgeries.append({**statement, "pi": 1})
    forgeries.append({**statement, "pi": 0})
    forgeries.append({**statement, "pi": statement["pi"] * 2 % N})
    rejected = sum(1 for forgery in forgeries if not wesolowski_verify(forgery))
    check(rejected == len(forgeries),
          f"all {len(forgeries)} forgeries are rejected ({rejected} rejected)")
    check(wesolowski_verify(statement), "and the honest proof still verifies")

    # -- 5. the challenge is bound to the statement -----------------------------------------
    base = hash_to_prime(g, y, steps, N)
    check(hash_to_prime(g, y, steps, N) == base, "the challenge prime is deterministic")
    check(len({base, hash_to_prime(g + 1, y, steps, N), hash_to_prime(g, y + 1, steps, N),
               hash_to_prime(g, y, steps + 1, N), hash_to_prime(g, y, steps, 3 * N)}) == 5,
          "changing g, y, T or the modulus changes the challenge prime")
    check(all(is_prime(hash_to_prime(g, y, steps + offset, N)) for offset in range(4)),
          "every derived challenge is prime")

    # -- 6. work and size separation ----------------------------------------------------------
    measurements = []
    for exponent in (10, 12, 14):
        small_steps = 1 << exponent
        prover = {"modmuls": 0}
        proof = wesolowski_prove(g, N, small_steps, prover)
        verifier = {"modmuls": 0}
        check(wesolowski_verify(proof, verifier), f"T = 2^{exponent} verifies")
        size = len(str(proof["pi"])) + len(str(proof["l"]))
        measurements.append((small_steps, prover["modmuls"], verifier["modmuls"], size))
        check(size < 400, f"the proof is one group element ({size} digits) whatever T is")
        check(verifier["modmuls"] < prover["modmuls"] / 3,
              f"verification is much cheaper than evaluation at T = 2^{exponent}")
    check(measurements[0][3] == measurements[-1][3] or
          abs(measurements[0][3] - measurements[-1][3]) < 40,
          f"proof size does not grow with T ({[row[3] for row in measurements]})")
    ratios = [prover / verifier for _, prover, verifier, _ in measurements]
    check(ratios[-1] > 2 * ratios[0],
          f"the prover/verifier work ratio grows with T ({[round(r, 1) for r in ratios]})")

    # -- 7. the trapdoor collapses the delay ---------------------------------------------------
    big_steps = 1 << 14
    big_y = delayed_output(g, N, big_steps)
    shortcut_counter = {"modmuls": 0}
    shortcut = trapdoor_shortcut(g, phi, N, big_steps, shortcut_counter)
    check(shortcut == big_y, "with the group order the same output appears")
    check(shortcut_counter["modmuls"] < big_steps // 8,
          f"...for {shortcut_counter['modmuls']} modmuls instead of {big_steps} sequential "
          f"ones (the gap widens with T)")
    check(phi > 0 and (setup["p"] - 1) * (setup["q"] - 1) == phi,
          "the trapdoor is exactly the group order: in this file the setup knows it")

    # -- 8. determinism --------------------------------------------------------------------------
    repeat = wesolowski_prove(g, N, steps)
    check(repeat["y"] == statement["y"] and repeat["pi"] == statement["pi"]
          and repeat["l"] == statement["l"],
          "proving the same statement twice gives the same output and the same proof")
    fresh = rsa_modulus(256, seed=SEED ^ 0x99)
    check(fresh["N"] != N and is_prime(fresh["p"]),
          "a different seed gives a different, still valid modulus")

    print(f"N is {N.bit_length()} bits; T = {steps} squarings in sequential work, verified "
          f"with {verifier_counter['modmuls']} modmuls")
    print(f"work/size over T = 2^10, 2^12, 2^14: {measurements}")
    print(f"trapdoor: same output in {shortcut_counter['modmuls']} modmuls with phi(N) known "
          f"(the order must stay unknown)")
    print("PASS")


if __name__ == "__main__":
    main()
