"""Task 711 - Implement homomorphic encryption simple operations demo (Paillier).  (Tier: Expert, CTCI Beyond - Security & Privacy)

Paillier is an additively homomorphic public-key cryptosystem.  For n = p*q with gcd(n, phi(n)) =
1, g = n + 1 and lambda = lcm(p-1, q-1), the maps

    Enc(m) = g^m * r^n mod n^2        (r random in Z_n*, so encryption is randomised)
    Dec(c) = L(c^lambda mod n^2) * mu mod n,   L(x) = (x - 1) / n,  mu = L(g^lambda)^{-1} mod n

satisfy Dec(Enc(m)) = m, and -- the point of the whole construction --

    Enc(a) * Enc(b) = Enc(a + b mod n),      Enc(m)^k = Enc(m*k mod n),
    Enc(m) * g^k    = Enc(m + k mod n).

Multiplication of two *ciphertexts* is not supported: that is the price of additive-only
homomorphism, and the file exercises exactly the operations that do exist.

What is implemented: deterministic prime generation (splitmix64 + Miller-Rabin, audited against
trial division on every number small enough to test exhaustively), key generation with the
lcm/phi bookkeeping, randomised encryption whose randomiser is drawn from the PRNG and required to
be invertible, and the four homomorphic operations (ciphertext + ciphertext, ciphertext +
plaintext, ciphertext * plaintext, ciphertext - ciphertext).  The demo closes with an e-voting
tally: 24 ballots are encrypted individually, summed homomorphically and only the tally is
decrypted, after which the same tally is recomputed from the plaintext ballots as an audit.

Simplified versus a real deployment: the modulus is only a couple of hundred bits (a demo key,
not a secure one), there is no serialisation, no padding and no zero-knowledge proof of correct
encryption, and one deterministic PRNG stream supplies both key material and randomisers.
Verified: round trips for random and boundary messages, randomised ciphertexts that decrypt alike,
all four homomorphic identities including the modular wraparound case, Enc(0) as the additive
identity, the negative checks (foreign key, out-of-range message, bad randomiser), and the tally
audit.
"""
import math
import sys

MASK64 = (1 << 64) - 1

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def bits(self, width):
        """A deterministic integer of `width` bits with the top bit set (so it is that long)."""
        value = 0
        while value.bit_length() < width:
            self.state, r = splitmix64(self.state)
            value = (value << 32) | (r & 0xFFFFFFFF)
        return (value & ((1 << width) - 1)) | (1 << (width - 1))

    def below(self, bound):
        """A deterministic integer in [1, bound - 1]."""
        check(bound >= 2, "the bound must be at least 2")
        self.state, r = splitmix64(self.state)
        return r % (bound - 1) + 1

    def coin(self):
        """A deterministic fair bit."""
        self.state, r = splitmix64(self.state)
        return r & 1


# --------------------------------------------------------------------------- primality
def is_probable_prime(candidate, rng, rounds=24):
    """Small trial division, then Miller-Rabin with deterministic-but-unstructured bases."""
    if candidate < 2:
        return False
    for prime in SMALL_PRIMES:
        if candidate % prime == 0:
            return candidate == prime
    exponent = candidate - 1
    twos = 0
    while exponent % 2 == 0:
        exponent //= 2
        twos += 1
    for _ in range(rounds):
        witness = pow(rng.below(candidate), exponent, candidate)
        if witness in (1, candidate - 1):
            continue
        for _ in range(twos - 1):
            witness = witness * witness % candidate
            if witness == candidate - 1:
                break
        else:
            return False
    return True


def is_prime_by_trial_division(candidate):
    """The independent oracle; only ever called on numbers small enough to finish."""
    if candidate < 2:
        return False
    if candidate % 2 == 0:
        return candidate == 2
    divisor = 3
    while divisor * divisor <= candidate:
        if candidate % divisor == 0:
            return False
        divisor += 2
    return True


def has_no_small_factor(candidate):
    return all(candidate % prime for prime in SMALL_PRIMES if prime != candidate)


def generate_prime(bits, rng):
    while True:
        candidate = rng.bits(bits) | 1
        if is_probable_prime(candidate, rng):
            return candidate


# --------------------------------------------------------------------------- Paillier
class Paillier:
    """Key pair plus the homomorphic operations; small demo key sizes only."""

    def __init__(self, bits=96, seed=0x5A11E):
        rng = Rng(seed)
        while True:
            p = generate_prime(bits, rng)
            q = generate_prime(bits, rng)
            if p != q and math.gcd(p * q, (p - 1) * (q - 1)) == 1:
                break
        self.bits = bits
        self.p = p
        self.q = q
        self.n = p * q
        self.n_square = self.n * self.n
        self.g = self.n + 1
        self.lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
        self.mu = pow(self._l(pow(self.g, self.lam, self.n_square)), -1, self.n)
        self.rng = Rng(seed ^ 0x9E3779B97F4A7C15)
        self.encryptions = 0

    def _l(self, value):
        """L(x) = (x - 1) / n; exact for any x congruent to 1 modulo n."""
        check((value - 1) % self.n == 0, "L is only defined on residues congruent to 1 mod n")
        return (value - 1) // self.n

    def encrypt(self, message, randomiser=None):
        check(0 <= message < self.n, "the message must lie in [0, n)")
        if randomiser is None:
            while True:
                randomiser = self.rng.below(self.n)
                if math.gcd(randomiser, self.n) == 1:
                    break
        check(1 <= randomiser < self.n and math.gcd(randomiser, self.n) == 1,
              "the randomiser must be invertible modulo n")
        self.encryptions += 1
        return pow(self.g, message, self.n_square) * pow(randomiser, self.n, self.n_square) \
            % self.n_square

    def decrypt(self, ciphertext):
        check(0 < ciphertext < self.n_square, "the ciphertext must lie in (0, n^2)")
        return self._l(pow(ciphertext, self.lam, self.n_square)) * self.mu % self.n

    # ---------------------------------------------------------------- homomorphic API
    def add_ciphertexts(self, first, second):
        """Enc(a) * Enc(b) = Enc(a + b mod n)."""
        return first * second % self.n_square

    def add_plaintext(self, ciphertext, constant):
        """Enc(m) * g^k = Enc(m + k mod n)."""
        check(0 <= constant < self.n, "the constant must lie in [0, n)")
        return ciphertext * pow(self.g, constant, self.n_square) % self.n_square

    def multiply_plaintext(self, ciphertext, factor):
        """Enc(m)^k = Enc(m * k mod n)."""
        check(0 <= factor < self.n, "the factor must lie in [0, n)")
        return pow(ciphertext, factor, self.n_square)

    def subtract_ciphertexts(self, first, second):
        """Enc(a) * Enc(b)^{-1} = Enc(a - b mod n)."""
        return first * pow(second, -1, self.n_square) % self.n_square


# --------------------------------------------------------------------------- checks
def test_primality_audit():
    for candidate in range(2, 600):
        expected = is_prime_by_trial_division(candidate)
        check(is_probable_prime(candidate, Rng(candidate + 1)) == expected,
              f"primality disagreement at {candidate}")
    for product in (121, 323, 899, 1763, 2491, 3599, 5293):
        check(is_probable_prime(product, Rng(product)) == is_prime_by_trial_division(product),
              f"Miller-Rabin must reject the composite {product}")
    check(is_probable_prime(4, Rng(1)) is False, "an even number is not prime")
    check(is_probable_prime(1, Rng(1)) is False and is_probable_prime(0, Rng(1)) is False,
          "0 and 1 are not prime")

    rng = Rng(0x1234)
    prime = generate_prime(64, rng)
    check(prime.bit_length() == 64, "the requested bit length is respected")
    check(has_no_small_factor(prime), "a generated prime has no small factor")
    for seed in (11, 22, 33):
        check(is_probable_prime(prime, Rng(seed)), f"independent Miller-Rabin run with seed {seed}")
    composite = prime * 3
    check(not is_probable_prime(composite, Rng(44)), "a multiple of a generated prime is rejected")
    return prime


def test_key_generation():
    key = Paillier(bits=96, seed=0xABCDEF)
    check(key.p != key.q, "p and q must differ")
    check(key.p.bit_length() == 96 and key.q.bit_length() == 96, "both factors have the requested size")
    for seed in (1, 2, 3):
        check(is_probable_prime(key.p, Rng(seed)) and is_probable_prime(key.q, Rng(seed + 50)),
              "both factors pass independent primality tests")
    check(has_no_small_factor(key.p) and has_no_small_factor(key.q), "no small factors")
    check(key.n == key.p * key.q, "n = p*q")
    check(key.n_square == key.n * key.n, "the modulus of the ciphertext space is n^2")
    check(key.lam == (key.p - 1) * (key.q - 1) // math.gcd(key.p - 1, key.q - 1),
          "lambda = lcm(p-1, q-1)")
    check(math.gcd(key.n, (key.p - 1) * (key.q - 1)) == 1, "the Paillier condition gcd(n, phi) = 1")
    check(key.g == key.n + 1, "the standard generator g = n+1")
    check(pow(key.g, key.lam, key.n_square) % key.n == 1, "g^lambda is congruent to 1 mod n")
    check(key._l(pow(key.g, key.lam, key.n_square)) * key.mu % key.n == 1,
          "mu must invert L(g^lambda) modulo n")
    return key


def test_round_trip():
    key = Paillier(bits=96, seed=0xC0FFEE)
    for message in (0, 1, 2, key.n - 1, key.n // 2, 12345):
        check(key.decrypt(key.encrypt(message)) == message,
              f"Dec(Enc({message})) must return {message}")
    rng = Rng(0xFACE)
    for _ in range(12):
        message = rng.below(key.n)
        check(key.decrypt(key.encrypt(message)) == message, "random messages round-trip")
    first = key.encrypt(7, randomiser=1)
    check(first == key.encrypt(7, randomiser=1), "the same randomiser gives the same ciphertext")
    third = key.encrypt(7, randomiser=5)
    check(third != first, "a different randomiser gives a different ciphertext")
    check(key.decrypt(third) == 7 and key.decrypt(first) == 7, "both decrypt to 7")


def test_randomisation():
    key = Paillier(bits=96, seed=0xBEEF)
    ciphertexts = {key.encrypt(3) for _ in range(30)}
    check(len(ciphertexts) >= 27, f"encryption must be randomised, got {len(ciphertexts)} distinct")
    check(all(key.decrypt(value) == 3 for value in ciphertexts), "all of them decrypt to 3")
    for value in sorted(ciphertexts)[:10]:
        check(math.gcd(value, key.n) == 1, "a ciphertext is invertible modulo n")
    check(key.encryptions >= 30, "the encryption counter tracks the work")
    return key


def test_homomorphic_operations():
    key = Paillier(bits=96, seed=0x1234)
    rng = Rng(0x99)
    for _ in range(10):
        a = rng.below(key.n)
        b = rng.below(key.n)
        summed = key.add_ciphertexts(key.encrypt(a), key.encrypt(b))
        check(key.decrypt(summed) == (a + b) % key.n,
              f"Enc(a)*Enc(b) must decrypt to a+b mod n for {a},{b}")
        factor = rng.below(1000)
        check(key.decrypt(key.multiply_plaintext(key.encrypt(a), factor)) == a * factor % key.n,
              f"Enc(a)^k must decrypt to a*k mod n for {a},{factor}")
        check(key.decrypt(key.add_plaintext(key.encrypt(a), b)) == (a + b) % key.n,
              "Enc(m)*g^k must decrypt to m+k mod n")

    # boundaries: wraparound, the additive identity, and scaling to zero
    a, b = key.n - 1, 5
    check(key.decrypt(key.add_ciphertexts(key.encrypt(a), key.encrypt(b))) == (a + b) % key.n,
          "the homomorphic sum wraps around modulo n")
    identity = key.encrypt(0)
    message = key.encrypt(442)
    check(key.decrypt(key.add_ciphertexts(message, identity)) == 442,
          "Enc(0) is the additive identity")
    check(key.decrypt(identity) == 0, "zero round-trips")
    check(key.decrypt(key.multiply_plaintext(message, 0)) == 0, "scaling by zero gives Enc(0)")
    check(key.decrypt(key.multiply_plaintext(message, key.n - 1)) == (-442) % key.n,
          "scaling by n-1 is negation modulo n")
    try:
        key.multiply_plaintext(message, key.n)
        raise AssertionError("a factor of n must be rejected (0 has the same effect mod n)")
    except AssertionError as error:
        check("factor" in str(error), f"unexpected message: {error}")
    check(key.decrypt(key.add_plaintext(message, 0)) == 442, "adding zero is a no-op")
    check(key.decrypt(key.add_plaintext(message, key.n - 1)) == 441, "adding n-1 subtracts one")
    check(key.decrypt(key.subtract_ciphertexts(key.encrypt(900), message)) == 900 - 442,
          "subtraction works by inverting the second ciphertext")
    check(key.decrypt(key.subtract_ciphertexts(message, message)) == 0,
          "a ciphertext minus itself is Enc(0)")

    # a foreign key must not recover the message (negative check)
    foreign = Paillier(bits=96, seed=0x999999)
    check(foreign.n != key.n, "the two demo keys differ")
    failures = 0
    for plain in (1234, 5678, 9012):
        reduced = key.encrypt(plain) % foreign.n_square
        if reduced == 0:
            reduced = 2
        if foreign.decrypt(reduced) != plain:
            failures += 1
    check(failures >= 2, f"a foreign key must fail to decrypt, only {failures} failures out of 3")
    return key


def test_plaintext_arithmetic_matches_homomorphic():
    key = Paillier(bits=96, seed=0x777)
    rng = Rng(0x3141)
    values = [rng.below(key.n // 8) for _ in range(12)]
    aggregate = key.encrypt(0)
    for value in values:
        aggregate = key.add_ciphertexts(aggregate, key.encrypt(value))
    plain = sum(values) % key.n
    check(key.decrypt(aggregate) == plain, "the product of ciphertexts decrypts to the plain sum")
    check(key.decrypt(key.multiply_plaintext(aggregate, 3)) == 3 * plain % key.n,
          "scaling the aggregate scales the sum")
    check(key.decrypt(key.add_plaintext(aggregate, 10)) == (plain + 10) % key.n,
          "shifting the aggregate shifts the sum")
    check(plain == sum(values), "no wraparound happened in this small instance")


def test_ballot_tally_demo():
    key = Paillier(bits=112, seed=0xE0E)
    rng = Rng(0xE1E)
    ballots = [rng.coin() for _ in range(24)]
    encrypted = [key.encrypt(ballot) for ballot in ballots]
    tally = key.encrypt(0)
    for ciphertext in encrypted:
        tally = key.add_ciphertexts(tally, ciphertext)
    decrypted_tally = key.decrypt(tally)
    check(decrypted_tally == sum(ballots), "the homomorphic tally equals the plaintext count")
    check(0 < sum(ballots) < len(ballots), "the ballots are a mix of yes and no")
    for ciphertext, ballot in zip(encrypted, ballots):
        check(ciphertext not in (0, 1, key.g, key.n), "a ciphertext is a large unrelated residue")
        check(key.decrypt(ciphertext) == ballot, "but it decrypts to its own ballot")
    check(decrypted_tally == sum(key.decrypt(ciphertext) for ciphertext in encrypted),
          "audit: the aggregate decrypts to the sum of the individual decryptions")
    return len(ballots), decrypted_tally


def test_boundaries_and_validation():
    key = Paillier(bits=96, seed=0x55)
    for bad, headline in ((key.n, "must lie in"), (-1, "must lie in")):
        try:
            key.encrypt(bad)
            raise AssertionError("an out-of-range message must be rejected")
        except AssertionError as error:
            check(headline in str(error), f"unexpected message: {error}")
    try:
        key.encrypt(5, randomiser=key.p)
        raise AssertionError("a randomiser sharing a factor with n must be rejected")
    except AssertionError as error:
        check("invertible" in str(error), f"unexpected message: {error}")
    for bad in (0, key.n_square):
        try:
            key.decrypt(bad)
            raise AssertionError("an out-of-range ciphertext must be rejected")
        except AssertionError as error:
            check("ciphertext" in str(error), f"unexpected message: {error}")
    try:
        key.add_plaintext(key.encrypt(1), key.n)
        raise AssertionError("an out-of-range constant must be rejected")
    except AssertionError as error:
        check("constant" in str(error), f"unexpected message: {error}")
    try:
        key.multiply_plaintext(key.encrypt(1), key.n + 1)
        raise AssertionError("an out-of-range factor must be rejected")
    except AssertionError as error:
        check("factor" in str(error), f"unexpected message: {error}")

    fresh = Paillier(bits=96, seed=0x55)
    check((fresh.n, fresh.p, fresh.q, fresh.lam, fresh.mu) == (key.n, key.p, key.q, key.lam, key.mu),
          "key generation is deterministic for a fixed seed")
    check(Paillier(bits=96, seed=0x55).encrypt(9) == Paillier(bits=96, seed=0x55).encrypt(9),
          "the whole pipeline is reproducible")
    check(Paillier(bits=96, seed=0x56).n != key.n, "a different seed gives a different key")


def main():
    prime = test_primality_audit()
    key = test_key_generation()
    test_round_trip()
    demo_key = test_randomisation()
    test_homomorphic_operations()
    test_plaintext_arithmetic_matches_homomorphic()
    ballots, tally = test_ballot_tally_demo()
    test_boundaries_and_validation()

    print(f"generated 64-bit prime audit sample: {prime}; demo modulus: {key.n.bit_length()} bits")
    print(f"ballots={ballots} homomorphic_tally={tally} encryptions={demo_key.encryptions}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
