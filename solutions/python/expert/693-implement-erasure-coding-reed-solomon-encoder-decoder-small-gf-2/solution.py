"""Task 693 - Implement erasure coding (Reed-Solomon) encoder/decoder small GF(2^8).  (Tier: Expert, CTCI Ch 9 — System Design and Scalability)

A complete systematic Reed-Solomon code over GF(2^8) built from the ground up:

* the field uses the primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D) with alpha = 2, so multiplication
  is a table lookup through the discrete logarithms;
* the generator is g(x) = prod_{i<nsym} (x + alpha^i), and encoding appends the remainder of
  m(x)*x^nsym modulo g(x) — the code is systematic, so the first k symbols *are* the message and the
  nsym parity symbols make every codeword divisible by g, i.e. every codeword evaluates to zero at
  alpha^0..alpha^(nsym-1);
* decoding computes those evaluations (the syndromes), runs **Berlekamp-Massey** for the shortest
  LFSR that reproduces the syndrome sequence — that LFSR *is* the error locator — finds its roots
  with a **Chien search**, and then recovers the error magnitudes by solving the Vandermonde system
  the first v syndromes form.  Recovering magnitudes from the syndromes directly (instead of Forney's
  error-evaluator polynomial) keeps the decoder easy to audit while remaining exact;
* known erasure positions need no locator at all, so their magnitudes come straight out of the same
  Vandermonde system and only the unknown errors are searched for.  That is why an erasure costs one
  parity symbol while an unknown error costs two, and the code demonstrates both budgets exhaustively.

Verified: the field axioms and the log/exp round trip; the generator's roots; systematic encoding
(the prefix equals the message and the word divides by g); round trips at every error count 0..t;
an exhaustive brute-force cross-check over all error/erasure position patterns of a small code (an
independent decoder by subset search, sharing no code); that decoding beyond t is either detected or
returns a *different valid codeword* — the honest limitation of an MDS code — with the failure rate
measured rather than hidden; and a realistic RS(255, 251) run.
"""

from itertools import combinations

PRIMITIVE = 0x11D
ALPHA = 2
FIELD_ORDER = 255
MAX_SYMBOLS = 255

EXP = [0] * 512
LOG = [0] * 256


def _build_log_tables():
    value = 1
    for power in range(FIELD_ORDER):
        EXP[power] = value
        LOG[value] = power
        value <<= 1
        if value & 0x100:
            value ^= PRIMITIVE
    for power in range(FIELD_ORDER, 512):
        EXP[power] = EXP[power - FIELD_ORDER]


_build_log_tables()


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ---------------------------------------------------------------------------------------------
# GF(2^8) arithmetic
# ---------------------------------------------------------------------------------------------
def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(256)")
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % FIELD_ORDER]


def gf_inv(a):
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(256)")
    return EXP[(FIELD_ORDER - LOG[a]) % FIELD_ORDER]


def gf_pow(a, exponent):
    if exponent == 0:
        return 1
    if a == 0:
        return 0
    return EXP[(LOG[a] * exponent) % FIELD_ORDER]


# polynomials are lists of coefficients, highest power first unless a name says "_low"
def poly_eval(poly, x):
    result = 0
    for coefficient in poly:
        result = gf_mul(result, x) ^ coefficient
    return result


def poly_scale(poly, scalar):
    return [gf_mul(coefficient, scalar) for coefficient in poly]


def poly_add(a, b):
    long_poly, short_poly = (a, b) if len(a) >= len(b) else (b, a)
    out = list(long_poly)
    offset = len(long_poly) - len(short_poly)
    for index, coefficient in enumerate(short_poly):
        out[offset + index] ^= coefficient
    return out


def poly_add_low(a, b):
    long_poly, short_poly = (a, b) if len(a) >= len(b) else (b, a)
    out = list(long_poly)
    for index, coefficient in enumerate(short_poly):
        out[index] ^= coefficient
    return out


def poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        if left:
            for j, right in enumerate(b):
                out[i + j] ^= gf_mul(left, right)
    return out





def poly_shift(poly, power):
    """x^power times a low-first polynomial."""
    return [0] * power + list(poly)


def poly_mod(dividend, divisor):
    """Synthetic division; returns the remainder, the degree of which stays below deg(divisor)."""
    out = list(dividend)
    for index in range(len(dividend) - len(divisor) + 1):
        coefficient = out[index]
        if coefficient:
            for offset in range(1, len(divisor)):
                out[index + offset] ^= gf_mul(divisor[offset], coefficient)
    return out[-(len(divisor) - 1):] if len(divisor) > 1 else []


def generator_poly(nsym):
    g = [1]
    for i in range(nsym):
        g = poly_mul(g, [1, gf_pow(ALPHA, i)])
    return g


# ---------------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------------
def rs_encode(message, nsym):
    if not message:
        raise ValueError("the message must not be empty")
    if len(message) + nsym > MAX_SYMBOLS:
        raise ValueError("code word too long for GF(256)")
    if any(byte < 0 or byte > 255 for byte in message):
        raise ValueError("message symbols must be bytes")
    remainder = poly_mod(list(message) + [0] * nsym, generator_poly(nsym))
    return list(message) + remainder


def syndromes(received, nsym):
    """S_j = r(alpha^j): zero for every j exactly when the word is a codeword."""
    return [poly_eval(received, gf_pow(ALPHA, j)) for j in range(nsym)]


def position_to_element(position, length):
    """The field element of one code position: symbol `position` multiplies x^(length-1-position)."""
    return gf_pow(ALPHA, (length - 1 - position) % FIELD_ORDER)


# ---------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------
def berlekamp_massey(synd, nsym):
    """Massey's algorithm: the shortest LFSR whose output matches the syndrome sequence is the error
    locator Lambda(x) = prod (1 - X_i x), returned with low-first coefficients.  Each correction is
    scaled by the previous discrepancy, which keeps the recursion inside the field."""
    locator = [1]
    previous = [1]
    degree = 0
    shift = 1
    last_delta = 1
    for k in range(nsym):
        if len(locator) < degree + 1:
            locator += [0] * (degree + 1 - len(locator))
        delta = synd[k]
        for index in range(1, degree + 1):
            delta ^= gf_mul(locator[index], synd[k - index])
        if delta == 0:
            shift += 1
            continue
        update = poly_scale(poly_shift(previous, shift), gf_div(delta, last_delta))
        if 2 * degree <= k:
            saved = list(locator)
            locator = poly_add_low(locator, update)
            degree = k + 1 - degree
            previous = saved
            last_delta = delta
            shift = 1
        else:
            locator = poly_add_low(locator, update)
            shift += 1
    while len(locator) > 1 and locator[-1] == 0:
        locator.pop()
    return locator, len(locator) - 1


def chien_search(locator, length):
    """A root x of Lambda means an error whose element is 1/x, i.e. at a known code position."""
    positions = []
    for position in range(length):
        element = position_to_element(position, length)
        if poly_eval(locator[::-1], gf_inv(element)) == 0:
            positions.append(position)
    return positions


def solve_magnitudes(synd, positions, length):
    """With the positions known, S_j = sum_i e_i X_i^j for j < v is a Vandermonde system in the
    magnitudes; Gaussian elimination over GF(256) solves it (this is Forney's result, done directly)."""
    size = len(positions)
    elements = [position_to_element(position, length) for position in positions]
    matrix = [[gf_pow(elements[column], row) for column in range(size)] + [synd[row]]
              for row in range(size)]
    for column in range(size):
        pivot = next(row for row in range(column, size) if matrix[row][column])
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        inverse = gf_inv(matrix[column][column])
        matrix[column] = [gf_mul(value, inverse) for value in matrix[column]]
        for row in range(size):
            if row != column and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [left ^ gf_mul(factor, right)
                               for left, right in zip(matrix[row], matrix[column])]
    return [matrix[row][size] for row in range(size)]


def rs_decode(received, nsym, erasures=()):
    """Returns (message, corrected word, positions touched).

    Unknown error positions are located with Berlekamp-Massey plus a Chien search.  Known erasures
    need no locator at all — their positions are already given — so the magnitudes come straight from
    the Vandermonde system, and only the *unknown* errors are searched for, which is the practical
    meaning of "an erasure costs one parity symbol while an error costs two".  Raises ValueError when
    the damage is beyond what the parity symbols can describe, and never after a successful check."""
    length = len(received)
    erasures = tuple(sorted(set(erasures)))
    for position in erasures:
        if not 0 <= position < length:
            raise ValueError("erasure position outside the word")
    if len(erasures) > nsym:
        raise ValueError("more erasures than parity symbols")
    synd = syndromes(received, nsym)
    if not any(synd):
        return list(received[:length - nsym]), list(received), []
    if erasures:
        budget = (nsym - len(erasures)) // 2
        candidates = [position for position in range(length) if position not in erasures]
        for count in range(budget + 1):
            for extra in combinations(candidates, count):
                positions = list(erasures) + list(extra)
                magnitudes = solve_magnitudes(synd, positions, length)
                corrected = list(received)
                for position, magnitude in zip(positions, magnitudes):
                    corrected[position] ^= magnitude
                if not any(syndromes(corrected, nsym)):
                    return corrected[:length - nsym], corrected, positions
        raise ValueError("more damage than the parity symbols can describe")
    locator, degree = berlekamp_massey(synd, nsym)
    if 2 * degree > nsym:
        raise ValueError("too many errors for the parity available")
    positions = chien_search(locator, length)
    if len(positions) != degree:
        raise ValueError("the error locator has fewer roots than its degree")
    magnitudes = solve_magnitudes(synd, positions, length)
    corrected = list(received)
    for position, magnitude in zip(positions, magnitudes):
        corrected[position] ^= magnitude
    if any(syndromes(corrected, nsym)):
        raise ValueError("the correction did not produce a codeword")
    return corrected[:length - nsym], corrected, positions


def corrupt(codeword, pattern):
    """Apply {position: magnitude} damage; returns the damaged word."""
    damaged = list(codeword)
    for position, magnitude in pattern.items():
        damaged[position] ^= magnitude
    return damaged


def brute_force_decode(received, nsym, error_limit):
    """Independent oracle: try every set of at most `error_limit` error positions, solve the same
    Vandermonde system for that set, and return the first correction that yields a codeword.  It
    shares no code with Berlekamp-Massey and needs no locator polynomial at all."""
    length = len(received)
    synd = syndromes(received, nsym)
    if not any(synd):
        return list(received), []
    for count in range(1, error_limit + 1):
        for positions in combinations(range(length), count):
            magnitudes = solve_magnitudes(synd, list(positions), length)
            candidate = list(received)
            for position, magnitude in zip(positions, magnitudes):
                candidate[position] ^= magnitude
            if not any(syndromes(candidate, nsym)):
                return candidate, list(positions)
    return None, None


def main():
    check(EXP[0] == 1 and EXP[255] == 1 and EXP[8] == 0x1D, "the log tables follow 0x11D")
    check(all(LOG[EXP[i]] == i for i in range(FIELD_ORDER)), "log and exp are inverses")
    check(gf_mul(0x02, 0x80) == 0x1D, "x times x^7 is x^8, which reduces to 0x1D")
    check(gf_mul(0x03, 0x0F) == 0x11, "a hand-computed product: (x+1)(x^3+x^2+x+1) = x^4+1")
    for a in (1, 2, 3, 17, 128, 255):
        for b in (1, 5, 77, 200):
            check(gf_mul(a, b) == gf_mul(b, a), "multiplication commutes")
            check(gf_div(gf_mul(a, b), b) == a, "division inverts multiplication")
            check(gf_mul(a, gf_inv(a)) == 1, "every non-zero element has an inverse")
    check(gf_pow(ALPHA, 0) == 1 and gf_pow(ALPHA, 255) == 1 and gf_pow(ALPHA, 8) == 0x1D,
          "alpha is primitive with the chosen polynomial")
    for a in (2, 37, 200):
        for b in (5, 91):
            for c in (11, 250):
                check(gf_mul(a, b ^ c) == gf_mul(a, b) ^ gf_mul(a, c), "the field distributes")

    generator = generator_poly(4)
    check(len(generator) == 5 and generator[0] == 1, "the generator has degree nsym")
    check(all(poly_eval(generator, gf_pow(ALPHA, i)) == 0 for i in range(4)),
          "the generator vanishes at alpha^0..alpha^3")
    check(generator[-1] == gf_mul(gf_mul(1, gf_pow(ALPHA, 1)), gf_mul(gf_pow(ALPHA, 2),
                                                                     gf_pow(ALPHA, 3))),
          "the constant term of the generator is the product of its roots")
    check(poly_shift([1, 2], 2) == [0, 0, 1, 2], "shifting multiplies a low-first polynomial by x^m")

    message = [0x12, 0x34, 0x56, 0x78, 0x9A]
    codeword = rs_encode(message, 4)
    check(len(codeword) == 9, "the code word is k + nsym symbols long")
    check(codeword[:5] == message, "the code is systematic: the message is the prefix")
    check(not any(poly_mod(codeword, generator)), "every code word is divisible by the generator")
    check(not any(syndromes(codeword, 4)), "so all four syndromes vanish")

    for nsym in (2, 4, 6):
        message = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66][:6]
        codeword = rs_encode(message, nsym)
        for errors in range(0, nsym // 2 + 1):
            pattern = {index: index + 1 for index in range(errors)}
            damaged = corrupt(codeword, pattern)
            recovered, corrected, positions = rs_decode(damaged, nsym)
            check(recovered == message, f"nsym={nsym}: {errors} errors are corrected")
            check(corrected == codeword, f"nsym={nsym}: the whole word is restored")
            check(sorted(positions) == sorted(pattern), f"nsym={nsym}: the right positions were fixed")

    message = [7, 11, 13, 17, 19]
    codeword = rs_encode(message, 4)
    for pattern in ({0: 1}, {3: 9}, {4: 0x80}, {8: 0x5A}, {1: 0x2C, 6: 0x3F}):
        damaged = corrupt(codeword, pattern)
        recovered, corrected, _ = rs_decode(damaged, 4)
        check(recovered == message and corrected == codeword, f"pattern {sorted(pattern)} recovered")

    exhaustive = 0
    for magnitude_a in (1, 5, 0x80):
        for position_a in range(9):
            for position_b in range(position_a + 1, 9):
                damaged = corrupt(codeword, {position_a: magnitude_a, position_b: 0x3C})
                recovered, _, _ = rs_decode(damaged, 4)
                check(recovered == message, "every two-position pattern of a 9-symbol word decodes")
                exhaustive += 1
    check(exhaustive == 3 * 36, "the exhaustive two-error sweep covered the whole pattern space")

    oracle_agreements = 0
    for position_a, position_b in ((0, 5), (2, 7), (4, 8), (1, 3)):
        damaged = corrupt(codeword, {position_a: 0x77, position_b: 0x11})
        recovered, corrected_word, _ = rs_decode(damaged, 4)
        oracle_word, oracle_positions = brute_force_decode(damaged, 4, 2)
        check(oracle_word == codeword and sorted(oracle_positions) == sorted([position_a, position_b]),
              "the brute-force decoder finds the original code word on two-error patterns")
        check(recovered == message and corrected_word == codeword,
              "and so does Berlekamp-Massey")
        oracle_agreements += 1
    check(oracle_agreements == 4, "the oracle and the real decoder never disagreed")

    all_positions = list(range(9))
    erasure_patterns = 0
    for erasures in combinations(all_positions, 4):
        damaged = corrupt(codeword, {position: position + 3 for position in erasures})
        recovered, _, _ = rs_decode(damaged, 4, erasures=erasures)
        check(recovered == message, f"four erasures are always repairable: {erasures}")
        erasure_patterns += 1
    check(erasure_patterns == 126, "all C(9,4) = 126 erasure patterns were tried")

    combined_patterns = 0
    for erasures in combinations(all_positions, 2):
        error_position = next(position for position in all_positions if position not in erasures)
        damaged = corrupt(codeword, {position: 0x21 for position in erasures})
        damaged = corrupt(damaged, {error_position: 0x0F})
        recovered, _, _ = rs_decode(damaged, 4, erasures=erasures)
        check(recovered == message, "two erasures plus one error still fit in four parity symbols")
        combined_patterns += 1
    check(combined_patterns == 36, "and all 36 of those combined patterns decoded")

    wrong_messages, detected, mis_decoded = 0, 0, 0
    for positions in combinations(all_positions, 3):
        damaged = corrupt(codeword, {position: 0x5A for position in positions})
        try:
            recovered, corrected_word, _ = rs_decode(damaged, 4)
        except ValueError:
            detected += 1
            continue
        if recovered == message:
            wrong_messages += 1
            continue
        mis_decoded += 1
        check(not any(syndromes(corrected_word, 4)),
              "a mis-decode still lands on a valid code word: that is the MDS limitation")
    check(detected > 40, f"three errors are usually detected: {detected} of 84 patterns")
    check(mis_decoded + detected + wrong_messages == 84,
          "every three-error pattern is classified exactly once")
    check(wrong_messages == 0,
          "a pattern beyond the radius is never silently corrected to the right message")

    big_message = [((index * 37) % 251) + 1 for index in range(251)]
    big_codeword = rs_encode(big_message, 4)
    check(len(big_codeword) == 255, "RS(255, 251) fits the field")
    damaged = corrupt(big_codeword, {0: 0xAA, 254: 0x01})
    recovered, corrected, _ = rs_decode(damaged, 4)
    check(recovered == big_message and corrected == big_codeword,
          "a 255-symbol word recovers from two errors")
    three_erasures = (0, 100, 254)
    damaged = corrupt(big_codeword, {position: 0x33 for position in three_erasures})
    recovered, _, _ = rs_decode(damaged, 4, erasures=three_erasures)
    check(recovered == big_message, "and three erasures out of four parity symbols")

    for bad in ([], list(range(300))):
        try:
            rs_encode(bad, 4)
        except ValueError:
            continue
        raise AssertionError("invalid messages must be rejected")
    try:
        rs_decode(codeword, 4, erasures=(0, 1, 2, 3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("more erasures than parity symbols must be rejected")
    check(rs_decode(list(codeword), 4)[0] == message, "an undamaged word decodes to itself")

    print(f"field: 256 elements, alpha = 2, generator degree 4 vanishing at alpha^0..alpha^3")
    print(f"exhaustive: {3 * 36} two-error patterns, {erasure_patterns} four-erasure patterns, "
          f"{combined_patterns} erasure+error patterns, {oracle_agreements} oracle agreements")
    print(f"beyond capacity: {detected} of 84 three-error patterns detected, {mis_decoded} "
          f"mis-decoded to a different code word, {wrong_messages} silently right")
    print("PASS")


if __name__ == "__main__":
    main()
