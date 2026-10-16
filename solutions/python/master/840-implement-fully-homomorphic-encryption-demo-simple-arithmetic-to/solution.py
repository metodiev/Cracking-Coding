"""Task 840 - Implement fully homomorphic encryption demo simple arithmetic (toy).  (Tier: Master, CTCI Beyond — Security & Privacy)

A real (if deliberately tiny) leveled BFV-style scheme over the negacyclic ring
R = Z_q[x]/(x^n + 1): ciphertexts are genuinely encrypted, arithmetic happens on the
ciphertexts alone, and the *only* thing the evaluator ever sees is noise-carrying ciphertext.

The scheme, as implemented:
  * plaintext space R_t with t = 2, ciphertext space R_q, scaling factor delta = q // t;
  * secret key s with ternary coefficients, public key (b, a) with b = -(a*s + e), so
    b + a*s + e = 0 is a real relation the checks re-verify;
  * encryption c = (b*u + e1 + delta*m, a*u + e2) with ternary u - the standard BFV form;
  * decryption computes v = sum_i c_i * s^i in R_q and returns round(t * [v]_q / q) mod t,
    which is correct exactly while the noise term [v - delta*m]_q stays below delta/2;
  * addition is componentwise; multiplication is the naive tensor product followed by BFV's
    scaling step (multiply by t/q), so a ciphertext of degree k (k+1 ring elements) becomes
    degree 2k and decryption needs s^2, s^4, ...  No relinearisation key is built, which is
    the main simplification: a production BFV keeps ciphertexts at degree 1 by key switching,
    at the cost of that extra machinery.
  * `modulus_switch` performs BFV's other noise-management step: every coefficient is scaled
    from q down to a smaller p, which preserves the message and multiplies the noise by p/q.
    That is what makes the scheme "fully" homomorphic modulo a bounded circuit depth: the
    depth is limited by noise, and both modulus switching and a larger q buy depth.

What is simplified: n = 8, q ~ 2**42, t = 2, small ternary/error distributions drawn from a
seeded splitmix64 stream (no library, no sampling from a real discrete Gaussian), no
relinearisation, no batching/NTT, no secret-key re-use across runs.  Nothing here is
deployable or secure; the point is that every step is real arithmetic and every claim below
is measured rather than asserted.

Verified, mostly against a plaintext oracle that recomputes the same circuit in R_t:
  * the ring itself: the naive negacyclic product equals an independent Kronecker-substitution
    big-integer product, and x^n = -1 holds;
  * key relation b + a*s + e = 0, and a fresh ciphertext's noise stays below delta/2;
  * round trip, homomorphic addition, homomorphic multiplication and the mixed circuit
    a*b + c all decrypt to exactly the plaintext oracle's answer;
  * noise accounting: fresh < add < multiply, the measured noise tracks the correctness
    boundary (a chain deep enough to push the noise past delta/2 decrypts wrongly - the check
    that the noise bound is not decoration), and a larger q buys exactly the extra headroom;
  * modulus switching preserves the message, cuts the noise by the predicted factor p/q and
    shrinks every coefficient by log2(q/p) bits, while a too-aggressive switch is detected by
    the canary vector instead of silently corrupting the answer;
  * cost: ciphertext degree doubles per multiplication, so the evaluator's work is measured,
    and the same seed reproduces identical ciphertexts.
"""

import hashlib

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
SMALL_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


class Rng:
    """Deterministic splitmix64 stream - the only randomness source in this file."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        z = (self.state + GOLDEN) & MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        self.state = (z ^ (z >> 31)) & MASK64
        return self.state


def is_prime(n):
    if n < 2:
        return False
    for p in SMALL_BASES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for base in SMALL_BASES:
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def center(value, q):
    """Symmetric representative of value mod q, in (-q/2, q/2]."""
    value %= q
    return value - q if value > q // 2 else value


def scale(value, source, target):
    """Round value * target / source to the nearest integer (half away from zero)."""
    product = value * target
    if product >= 0:
        return (product + source // 2) // source
    return -((-product + source // 2) // source)


# --------------------------------------------------------------------------------------
# ring arithmetic
# --------------------------------------------------------------------------------------
def poly_add(left, right, q):
    return [(a + b) % q for a, b in zip(left, right)]


def poly_sub(left, right, q):
    return [(a - b) % q for a, b in zip(left, right)]


def poly_mul(left, right, q, n):
    """Negacyclic product (x^n = -1) - the schoolbook O(n^2) implementation."""
    out = [0] * n
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            k = i + j
            if k >= n:
                k -= n
                out[k] = (out[k] - a * b) % q
            else:
                out[k] = (out[k] + a * b) % q
    return out


def poly_mul_kronecker(left, right, q, n):
    """Independent oracle for the same product via Kronecker substitution.

    Both polynomials are packed into big integers with a base large enough that no carry
    ever crosses a coefficient boundary, multiplied with exact big-integer arithmetic, and
    the 2n-1 raw coefficients are then folded back with x^n = -1.  Nothing here shares code
    with `poly_mul`, so agreement is real evidence.
    """
    bits = (2 * q.bit_length() + n.bit_length() + 8)
    base = 1 << bits
    packed_left = sum(a << (bits * i) for i, a in enumerate(left))
    packed_right = sum(b << (bits * j) for j, b in enumerate(right))
    product = packed_left * packed_right
    raw = [(product >> (bits * k)) & (base - 1) for k in range(2 * n - 1)]
    out = [0] * n
    for k, value in enumerate(raw):
        if k < n:
            out[k] += value
        else:
            out[k - n] -= value
    return [value % q for value in out]


class Params:
    def __init__(self, n=8, q=(1 << 42) + 4389, t=2):
        self.n = n
        self.q = q
        self.t = t
        self.delta = q // t

    def switched(self, new_q):
        return Params(self.n, new_q, self.t)


# --------------------------------------------------------------------------------------
# BFV
# --------------------------------------------------------------------------------------
def sample(params, rng, bound):
    width = 2 * bound + 1
    return [(rng.next() % width) - bound for _ in range(params.n)]


def keygen(params, rng):
    secret = sample(params, rng, 1)
    public_a = [rng.next() % params.q for _ in range(params.n)]
    error = sample(params, rng, 2)
    product = poly_mul(public_a, secret, params.q, params.n)
    public_b = [(-value) % params.q for value in poly_add(product, error, params.q)]
    return {"s": secret, "a": public_a, "b": public_b, "e": error}


def encrypt(params, key, message, rng):
    u = sample(params, rng, 1)
    e1 = sample(params, rng, 2)
    e2 = sample(params, rng, 2)
    scaled = [(params.delta * value) % params.q for value in message]
    c0 = poly_add(poly_add(poly_mul(key["b"], u, params.q, params.n), e1, params.q),
                  scaled, params.q)
    c1 = poly_add(poly_mul(key["a"], u, params.q, params.n), e2, params.q)
    return [c0, c1]


def add_ciphertexts(params, left, right):
    size = max(len(left), len(right))
    zero = [0] * params.n
    return [poly_add(left[i] if i < len(left) else zero,
                     right[i] if i < len(right) else zero, params.q) for i in range(size)]


def poly_mul_exact(left, right, n):
    """Negacyclic product over the *integers*: the larger modulus a product really needs.

    A ciphertext multiplication cannot be done modulo q, because the delta^2 * m term of the
    product is much larger than q and would wrap; real implementations therefore compute in
    an extended modulus (or an RNS basis of one).  Here the extension is exact integer
    arithmetic over the centred representatives, which is the same thing for a toy size.
    """
    out = [0] * n
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            k = i + j
            if k >= n:
                out[k - n] -= a * b
            else:
                out[k] += a * b
    return out


def tensor_product(params, left, right):
    """Exact tensor product: degree k1 + k2, so the evaluator's cost grows per level."""
    out = [[0] * params.n for _ in range(len(left) + len(right) - 1)]
    for i, a in enumerate(left):
        centred_a = [center(value, params.q) for value in a]
        for j, b in enumerate(right):
            centred_b = [center(value, params.q) for value in b]
            product = poly_mul_exact(centred_a, centred_b, params.n)
            out[i + j] = [x + y for x, y in zip(out[i + j], product)]
    return out


def rescale(params, ciphertext):
    """BFV's multiplication scaling: multiply every coefficient by t/q and reduce to q.

    Without it a product decrypts to delta * m1 * m2 (the plaintext factor squares); the
    scaling is what brings the factor back to delta, and it also cuts the noise by t/q.
    """
    return [[scale(value, params.q, params.t) % params.q for value in component]
            for component in ciphertext]


def multiply_ciphertexts(params, left, right):
    return rescale(params, tensor_product(params, left, right))


def secret_power(secret, degree, params):
    result = [1] + [0] * (params.n - 1)
    for _ in range(degree):
        result = poly_mul(result, secret, params.q, params.n)
    return result


def decrypt_raw(params, secret, ciphertext):
    """v = sum_i c_i * s^i in R_q - the value the noise is measured against."""
    value = [0] * params.n
    for degree, component in enumerate(ciphertext):
        value = poly_add(value, poly_mul(component, secret_power(secret, degree, params),
                                         params.q, params.n), params.q)
    return value


def decrypt(params, secret, ciphertext):
    value = decrypt_raw(params, secret, ciphertext)
    return [round(params.t * center(entry, params.q) / params.q) % params.t
            for entry in value]


def noise(params, secret, ciphertext, message):
    """Max coefficient of [v - delta * m]_q: the quantity the correctness bound limits."""
    value = decrypt_raw(params, secret, ciphertext)
    return max(abs(center(center(entry, params.q) - params.delta * m, params.q))
               for entry, m in zip(value, message))


def modulus_switch(params, ciphertext, new_params):
    """Scale every coefficient down to the smaller modulus: the message survives, noise shrinks."""
    return [[scale(center(value, params.q), params.q, new_params.q) % new_params.q
             for value in component] for component in ciphertext]


def plain_add(left, right, t):
    return [(a + b) % t for a, b in zip(left, right)]


def plain_mul(left, right, t, n):
    return poly_mul(left, right, t, n)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def canary_ok(params, secret, value, message):
    """A known-answer canary: corruption is detected instead of silently accepted."""
    return decrypt(params, secret, value) == message


def main():
    params = Params()
    n, q, t = params.n, params.q, params.t
    check(n & (n - 1) == 0 and n >= 4,
          "the ring dimension must be a power of two for x^n + 1 to be the standard ring")
    check(params.delta == q // t and params.delta > t,
          "delta = q // t must be a real scaling factor, not a rounding artefact")

    # ---- 1. the ring, against an independent oracle -----------------------
    rng = Rng(2024)
    for step in range(24):
        left = [rng.next() % q for _ in range(n)]
        right = [rng.next() % q for _ in range(n)]
        check(poly_mul(left, right, q, n) == poly_mul_kronecker(left, right, q, n),
              "negacyclic product must match the Kronecker oracle (step %d)" % step)
    one = [1] + [0] * (n - 1)
    x = [0, 1] + [0] * (n - 2)
    check(poly_mul(one, left, q, n) == left, "one is the ring identity")
    check(poly_mul(x, [0] * (n - 1) + [1], q, n) == [(q - 1) % q] + [0] * (n - 1),
          "x^n = -1 is the defining relation")
    check(poly_mul(left, right, q, n) == poly_mul(right, left, q, n),
          "the ring is commutative")
    check(poly_add(left, [(-value) % q for value in left], q) == [0] * n,
          "additive inverses exist coefficientwise")

    # ---- 2. keys and the public-key relation ------------------------------
    key = keygen(params, Rng(7))
    relation = poly_add(poly_add(poly_mul(key["a"], key["s"], q, n), key["e"], q),
                        key["b"], q)
    check(relation == [0] * n, "b + a*s + e must vanish in R_q")
    other = keygen(params, Rng(8))
    check(other["s"] != key["s"] and other["b"] != key["b"], "key generation is randomised")

    # ---- 3. round trip over many messages ---------------------------------
    messages = [[(rng.next()) % t for _ in range(n)] for _ in range(16)]
    messages.append([0] * n)
    messages.append([t - 1] * n)
    fresh_noises = []
    for index, message in enumerate(messages):
        ciphertext = encrypt(params, key, message, Rng(100 + index))
        check(decrypt(params, key["s"], ciphertext) == message,
              "round trip of message %d" % index)
        fresh_noises.append(noise(params, key["s"], ciphertext, message))
    check(max(fresh_noises) < params.delta // 2,
          "fresh noise must stay below delta/2 (%d)" % max(fresh_noises))
    wrong_key = keygen(params, Rng(9))
    check(decrypt(params, wrong_key["s"], encrypt(params, key, messages[1], Rng(101)))
          != messages[1], "decryption with the wrong secret key must not match")

    # ---- 4. homomorphic arithmetic versus the plaintext oracle ------------
    left, right, third = messages[2], messages[3], messages[4]
    ct_left = encrypt(params, key, left, Rng(11))
    ct_right = encrypt(params, key, right, Rng(12))
    ct_third = encrypt(params, key, third, Rng(13))
    sum_ct = add_ciphertexts(params, ct_left, ct_right)
    check(decrypt(params, key["s"], sum_ct) == plain_add(left, right, t),
          "homomorphic addition must match the oracle")
    check(noise(params, key["s"], sum_ct, plain_add(left, right, t)) <=
          fresh_noises[2] + fresh_noises[3] + 2,
          "additive noise grows at most like the sum of the two inputs")
    product_ct = multiply_ciphertexts(params, ct_left, ct_right)
    expected_product = plain_mul(left, right, t, n)
    check(len(product_ct) == 3, "a depth-1 product has degree 2 (no relinearisation)")
    check(decrypt(params, key["s"], product_ct) == expected_product,
          "homomorphic multiplication must match the oracle")
    circuit_ct = add_ciphertexts(params, product_ct, ct_third)
    expected_circuit = plain_add(expected_product, third, t)
    check(decrypt(params, key["s"], circuit_ct) == expected_circuit,
          "the mixed circuit a*b + c must match the oracle")
    check(canary_ok(params, key["s"], circuit_ct, expected_circuit),
          "the known-answer canary accepts a correct evaluation")

    # ---- 5. noise accounting and the correctness boundary -----------------
    n_add = noise(params, key["s"], sum_ct, plain_add(left, right, t))
    n_mul = noise(params, key["s"], product_ct, expected_product)
    pair_noise = max(fresh_noises[2], fresh_noises[3])
    check(1 <= n_add <= fresh_noises[2] + fresh_noises[3] + 2,
          "additive noise tracks the sum of the input noises (%d)" % n_add)
    check(n_mul > pair_noise,
          "multiplication must dominate the fresh noise (%d vs %d)" % (n_mul, pair_noise))
    check(n_mul < params.delta // 2, "a depth-1 product must still decrypt correctly")

    # push the same product through further multiplications until the noise crosses delta/2
    deep = product_ct
    deep_expected = expected_product
    levels = 1
    while True:
        grown = multiply_ciphertexts(params, deep, deep)
        grown_expected = plain_mul(deep_expected, deep_expected, t, n)
        measured = noise(params, key["s"], grown, grown_expected)
        correct = decrypt(params, key["s"], grown) == grown_expected
        if not correct or measured >= params.delta // 2:
            break
        levels += 1
        deep, deep_expected = grown, grown_expected
        check(levels + 1 < 8, "the toy parameters must not support unbounded depth")
    check(measured >= params.delta // 2 and not correct,
          "crossing delta/2 must actually break decryption (noise %d, bound %d)"
          % (measured, params.delta // 2))
    check(levels >= 2, "the toy parameters must support at least two multiplicative levels")

    # a larger q buys headroom: the same circuit keeps its correctness margin
    wider = Params(n, q << 12, t)
    wide_key = keygen(wider, Rng(7))
    wide_fresh = noise(wider, wide_key["s"],
                       encrypt(wider, wide_key, left, Rng(11)), left)
    wide_margin = wider.delta // 2 - wide_fresh
    check(wide_margin > params.delta // 2 - fresh_noises[2],
          "a larger ciphertext modulus must raise the noise headroom")
    check(wide_fresh <= fresh_noises[2] * 8 + 16,
          "fresh noise grows slowly with q, not proportionally")

    # ---- 6. modulus switching ---------------------------------------------
    switched_params = params.switched(params.q >> 10)
    switched = modulus_switch(params, product_ct, switched_params)
    check(decrypt(switched_params, key["s"], switched) == expected_product,
          "modulus switching must preserve the message")
    switched_noise = noise(switched_params, key["s"], switched, expected_product)
    check(switched_noise < n_mul,
          "modulus switching must reduce the noise (%d -> %d)" % (n_mul, switched_noise))
    check(switched_noise <= n_mul * (switched_params.q / params.q) * 2 + 4,
          "the measured noise reduction must track the modulus ratio")
    before_bits = max(value.bit_length() for value in product_ct[0])
    after_bits = max(value.bit_length() for value in switched[0])
    check(before_bits - after_bits >= 8,
          "coefficients must shrink by about log2(q/p) bits (%d -> %d)"
          % (before_bits, after_bits))
    saturated = params.switched(8)
    squeezed = modulus_switch(params, product_ct, saturated)
    check(decrypt(saturated, key["s"], squeezed) != expected_product,
          "an aggressive switch must be caught by the canary, not silently trusted")
    check(saturated.delta == 4 and switched_noise < params.delta // 2,
          "the breaking switch is only one that leaves less than half the noise budget")
    check(not canary_ok(saturated, key["s"], squeezed, expected_product),
          "the canary must reject a corrupted evaluation")

    # ---- 7. cost, determinism and edge cases ------------------------------
    check(len(tensor_product(params, deep, ct_left)) == len(deep) + 1,
          "the tensor product is additive in the two degrees")
    check(len(deep) == 2 ** levels + 1,
          "ciphertext degree doubles per multiplication level (%d components)" % len(deep))
    first = encrypt(params, key, messages[5], Rng(55))
    again = encrypt(params, key, messages[5], Rng(55))
    check(first == again, "the same seed must reproduce the same ciphertext")
    check(encrypt(params, key, messages[5], Rng(56)) != first,
          "a different seed must give a different ciphertext")
    zero_ct = encrypt(params, key, [0] * n, Rng(57))
    check(decrypt(params, key["s"], zero_ct) == [0] * n, "zero encrypts to zero")
    check(decrypt(params, key["s"], multiply_ciphertexts(params, zero_ct, ct_left))
          == [0] * n, "multiplying by the zero ciphertext stays zero")
    additive_identity = multiply_ciphertexts(params, encrypt(params, key,
                                                             [1] + [0] * (n - 1), Rng(58)),
                                             ct_left)
    check(decrypt(params, key["s"], additive_identity) == left,
          "the plaintext one multiplies as the identity")
    check(all(0 <= value < q for component in deep for value in component),
          "every ciphertext coefficient is a residue of q")

    print("ring: n=%d, q=%d (%d bits), t=%d, delta=%d" % (n, q, q.bit_length(), t,
                                                          params.delta))
    print("noise: fresh %d/%d, add %d, multiply %d, bound delta/2 = %d"
          % (fresh_noises[2], fresh_noises[3], n_add, n_mul, params.delta // 2))
    print("depth: exact through %d levels, broken at %d (noise %d)"
          % (levels, levels + 1, measured))
    print("modulus switch: %d -> %d, noise %d -> %d, coefficients %d -> %d bits"
          % (params.q, switched_params.q, n_mul, switched_noise, before_bits, after_bits))
    print("PASS")


if __name__ == "__main__":
    main()
