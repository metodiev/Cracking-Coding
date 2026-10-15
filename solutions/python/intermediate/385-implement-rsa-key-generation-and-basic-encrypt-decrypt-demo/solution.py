"""Task 385 - Implement RSA key generation and basic encrypt/decrypt (demo). (Tier: Intermediate, CTCI Beyond — Security & Privacy)

Textbook (unpadded) toy RSA: with primes p, q, n = p*q and phi = (p-1)*(q-1), pick a public
exponent e with gcd(e, phi) = 1 and set d = e^(-1) (mod phi). Encryption is c = m^e (mod n),
decryption is m = c^d (mod n). This is a demo: real RSA uses large keys, padding (OAEP) and
more. Two documented quirks: 0 and 1 are fixed points, and "decrypting" a ciphertext with the
public exponent (the wrong key) usually does not return the message, but nothing guarantees
that for every message -- the assertions below only pin the concrete examples.
"""


def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a, modulus):
    g, x, _ = extended_gcd(a % modulus, modulus)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % modulus


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def is_small_prime(n):
    """Trial division, used only to reject obviously bad demo primes."""
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def keygen(p, q, e):
    """Return (n, e, d, phi); raises ValueError on bad demo parameters."""
    if not is_small_prime(p) or not is_small_prime(q) or p == q:
        raise ValueError("p and q must be distinct primes")
    n = p * q
    phi = (p - 1) * (q - 1)
    if gcd(e, phi) != 1:
        raise ValueError("public exponent must be coprime with phi(n)")
    return n, e, mod_inverse(e, phi), phi


def encrypt(message, e, n):
    if not 0 <= message < n:
        raise ValueError("message must satisfy 0 <= m < n")
    return pow(message, e, n)


def decrypt(ciphertext, d, n):
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must satisfy 0 <= c < n")
    return pow(ciphertext, d, n)


def main():
    # classic small keypair p=61, q=53 (n = 3233, phi = 3120, e = 17, d = 2753)
    n, e, d, phi = keygen(61, 53, 17)
    assert (n, e, d, phi) == (3233, 17, 2753, 3120)
    assert n == 61 * 53 and phi == 60 * 52
    assert e * d % phi == 1, "e*d = 1 (mod phi(n))"
    assert d != e

    # round trip for several messages, including the boundaries 0 and n-1
    for message in (0, 1, 2, 42, 65, 1000, 3232):
        ciphertext = encrypt(message, e, n)
        assert decrypt(ciphertext, d, n) == message, message
    assert encrypt(0, e, n) == 0 and encrypt(1, e, n) == 1        # documented fixed points
    assert encrypt(n - 1, e, n) == n - 1                          # (-1)^e = -1 for odd e
    for message in (2, 42, 65, 1000):
        assert encrypt(message, e, n) != message

    # wrong key: applying the public exponent to the ciphertext does not recover the message
    ciphertext42 = encrypt(42, e, n)
    assert decrypt(ciphertext42, e, n) == 1001 and decrypt(ciphertext42, e, n) != 42

    # second keypair with e = 3: p=101, q=113 -> n=11413, phi=11200, d=7467
    n2, e2, d2, phi2 = keygen(101, 113, 3)
    assert (n2, e2, d2, phi2) == (11413, 3, 7467, 11200)
    assert e2 * d2 % phi2 == 1
    for message in (0, 1, 2, 5, 42, 999, 11412):
        assert decrypt(encrypt(message, e2, n2), d2, n2) == message, message
    ciphertext7 = encrypt(7, e2, n2)
    assert decrypt(ciphertext7, e2, n2) == 8652 and decrypt(ciphertext7, e2, n2) != 7

    # keygen rejects bad demo parameters (documented errors)
    for bad in ((7, 13, 3), (9, 53, 17), (61, 61, 17), (61, 53, 4)):
        try:
            keygen(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))

    # encryption domain is checked
    for bad in (3233, -1):
        try:
            encrypt(bad, e, n)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for message %d" % bad)


if __name__ == "__main__":
    main()
    print("PASS")
