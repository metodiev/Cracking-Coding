"""Task 540 - Implement Möbius transform on subset lattice (SOS). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

The zeta (subset-sum, "sum over subsets") transform F[S] = sum over T subset of S of f[T], its
inverse - the Möbius transform f[S] = sum over T subset of S of (-1)^(|S| - |T|) F[T] - and the
mirror pair for supersets, all in O(2^n * n) with the standard SOS recurrence: for each bit i, in
increasing order of the mask, F[mask] += F[mask without bit i] (subsets) or
F[mask] += F[mask with bit i] (supersets). Each transform is written twice: once by the recurrence
and once by brute force over all submasks or supermasks of each mask, so the O(2^n n) sweep is
validated against its definition rather than against itself.

Applications checked in the file: (1) the algebra is used as an OR-convolution, since the zeta
transform is a ring homomorphism taking OR-convolution to pointwise multiplication, so
mobius(zeta(f) * zeta(g)) must equal the brute-force OR-convolution, and AND-convolution is the
same construction on complemented masks; (2) counting elements of a list that are subsets (or
supersets) of a query mask, which is the classic "SOS DP on frequencies" trick and is compared
with a direct scan; (3) the Boolean-lattice Möbius function sum over T subset of S of
(-1)^(|S|-|T|) = [S is empty], obtained by Möbius-transforming the all-ones vector.

Checks: the hand-checked n = 2 example f = [1, 2, 3, 4] with zeta [1, 3, 4, 10]; zeta and Möbius
inverse to each other for n = 0..8 on LCG integer arrays; zeta of the delta at the empty set equal
to the all-ones vector and zeta of all-ones equal to 2^popcount(mask); superset transforms checked
against the brute-force supermask scan; OR-convolution and AND-convolution against brute force;
the subset/superset counting applications against direct scans; and the degenerate cases n = 0 and
the zero vector.
"""


def size_of(n):
    return 1 << n


def zeta_subset(f, n):
    """F[S] = sum_{T subset S} f[T], by the SOS recurrence over bits."""
    out = list(f)
    for i in range(n):
        bit = 1 << i
        for mask in range(size_of(n)):
            if mask & bit:
                out[mask] += out[mask ^ bit]
    return out


def mobius_subset(f, n):
    """Inverse of zeta_subset: f[S] = sum_{T subset S} (-1)^(|S|-|T|) F[T]."""
    out = list(f)
    for i in range(n):
        bit = 1 << i
        for mask in range(size_of(n)):
            if mask & bit:
                out[mask] -= out[mask ^ bit]
    return out


def zeta_superset(f, n):
    """F[S] = sum_{T superset S} f[T]."""
    out = list(f)
    for i in range(n):
        bit = 1 << i
        for mask in range(size_of(n)):
            if not mask & bit:
                out[mask] += out[mask | bit]
    return out


def mobius_superset(f, n):
    """Inverse of zeta_superset: f[S] = sum_{T superset S} (-1)^(|T|-|S|) F[T]."""
    out = list(f)
    for i in range(n):
        bit = 1 << i
        for mask in range(size_of(n)):
            if not mask & bit:
                out[mask] -= out[mask | bit]
    return out


def submasks(mask):
    sub = mask
    while True:
        yield sub
        if sub == 0:
            return
        sub = (sub - 1) & mask


def supersets(mask, n):
    rest = ((1 << n) - 1) ^ mask
    sub = rest
    while True:
        yield mask | sub
        if sub == 0:
            return
        sub = (sub - 1) & rest


def brute_zeta_subset(f, n):
    return [sum(f[t] for t in submasks(s)) for s in range(size_of(n))]


def brute_zeta_superset(f, n):
    return [sum(f[t] for t in supersets(s, n)) for s in range(size_of(n))]


def or_convolve(f, g, n):
    """Brute-force OR-convolution: h[m] = sum_{s | t = m} f[s] g[t]."""
    size = size_of(n)
    out = [0] * size
    for s in range(size):
        if f[s] == 0:
            continue
        for t in range(size):
            if g[t]:
                out[s | t] += f[s] * g[t]
    return out


def or_convolve_via_zeta(f, g, n):
    zf = zeta_subset(f, n)
    zg = zeta_subset(g, n)
    return mobius_subset([a * b for (a, b) in zip(zf, zg)], n)


def and_convolve(f, g, n):
    """Brute-force AND-convolution: h[m] = sum_{s & t = m} f[s] g[t]."""
    size = size_of(n)
    out = [0] * size
    for s in range(size):
        if f[s] == 0:
            continue
        for t in range(size):
            if g[t]:
                out[s & t] += f[s] * g[t]
    return out


def and_convolve_via_zeta(f, g, n):
    """AND-convolution = OR-convolution of the bitwise complements."""
    size = size_of(n)
    full = size - 1
    rf = [0] * size
    rg = [0] * size
    for mask in range(size):
        rf[full ^ mask] = f[mask]
        rg[full ^ mask] = g[mask]
    result = or_convolve_via_zeta(rf, rg, n)
    out = [0] * size
    for mask in range(size):
        out[full ^ mask] = result[mask]
    return out


def count_subsets_present(values, n):
    """For every mask, how many input values are submasks of it (SOS on frequencies)."""
    freq = [0] * size_of(n)
    for v in values:
        freq[v] += 1
    return zeta_subset(freq, n)


def count_subsets_present_direct(values, n):
    return [sum(1 for v in values if v & s == v) for s in range(size_of(n))]


def count_supersets_present(values, n):
    freq = [0] * size_of(n)
    for v in values:
        freq[v] += 1
    return zeta_superset(freq, n)


def count_supersets_present_direct(values, n):
    return [sum(1 for v in values if v & s == s) for s in range(size_of(n))]


def lcg_array(size, seed, modulus=21):
    out = []
    for _ in range(size):
        seed = (seed * 75 + 74) % 65537
        out.append(seed % modulus)
    return out, seed


def main():
    # --- Hand-checked 2-bit example -----------------------------------------------------
    f = [1, 2, 3, 4]
    assert zeta_subset(f, 2) == [1, 3, 4, 10], "F[11] = 1+2+3+4 = 10"
    assert mobius_subset([1, 3, 4, 10], 2) == f
    assert zeta_superset(f, 2) == [10, 6, 7, 4], "the dual sums"
    assert mobius_superset([10, 6, 7, 4], 2) == f
    assert brute_zeta_subset(f, 2) == [1, 3, 4, 10]
    assert brute_zeta_superset(f, 2) == [10, 6, 7, 4]

    # --- Round trips and brute-force definitions ---------------------------------------
    seed = 20251016
    for n in range(0, 9):
        size = size_of(n)
        values, seed = lcg_array(size, seed)
        signed, seed = lcg_array(size, seed, modulus=13)
        signed = [v - 6 for v in signed]
        assert zeta_subset(values, n) == brute_zeta_subset(values, n), "SOS vs submasks"
        assert zeta_superset(values, n) == brute_zeta_superset(values, n), "SOS vs supermasks"
        assert mobius_subset(zeta_subset(values, n), n) == values, "Mobius inverts zeta"
        assert mobius_superset(zeta_superset(values, n), n) == values
        assert zeta_subset(mobius_subset(values, n), n) == values, "the other order too"
        # Signed data exercises the alternating signs of the Mobius transform.
        assert mobius_subset(zeta_subset(signed, n), n) == signed
        assert mobius_superset(zeta_superset(signed, n), n) == signed
        # zeta of a delta at the empty set is the all-ones vector; zeta of all-ones counts
        # the submasks of each mask, i.e. 2^popcount.
        delta = [0] * size
        delta[0] = 1
        assert zeta_subset(delta, n) == [1] * size
        ones_zeta = zeta_subset([1] * size, n)
        assert ones_zeta == [1 << bin(mask).count("1") for mask in range(size)]
        # The Boolean-lattice Mobius function: sum over T subset S of (-1)^(|S|-|T|) = [S empty].
        assert mobius_subset([1] * size, n) == [1] + [0] * (size - 1)
        assert mobius_superset([1] * size, n) == [0] * (size - 1) + [1]
        assert zeta_subset([0] * size, n) == [0] * size

    # --- OR- and AND-convolution through the transform ---------------------------------
    seed = 777
    for n in range(0, 9):
        size = size_of(n)
        f, seed = lcg_array(size, seed, modulus=4)
        g, seed = lcg_array(size, seed, modulus=3)
        assert or_convolve_via_zeta(f, g, n) == or_convolve(f, g, n), "zeta is a ring map for OR"
        assert and_convolve_via_zeta(f, g, n) == and_convolve(f, g, n), "AND via complements"
        ip = indicator_from(f, 2)
        iq = indicator_from(g, 2)
        assert or_convolve(ip, iq, n) == or_convolve_via_zeta(ip, iq, n)
        assert sum(or_convolve(ip, iq, n)) == sum(ip) * sum(iq), "mass conservation for OR"
        assert sum(and_convolve_via_zeta(ip, iq, n)) == sum(ip) * sum(iq), "and for AND"

    # --- Counting applications ---------------------------------------------------------
    seed = 909090
    for n in (3, 4, 5, 6, 7):
        size = size_of(n)
        pool, seed = lcg_array(size, seed, modulus=size)
        values = pool[:size // 2 + 1]
        assert count_subsets_present(values, n) == count_subsets_present_direct(values, n), \
            "SOS counts the elements that are submasks"
        assert count_supersets_present(values, n) == count_supersets_present_direct(values, n), \
            "the superset version counts the other direction"
        # Sanity: the full mask contains every value, the empty mask only the zeros.
        assert count_subsets_present(values, n)[size - 1] == len(values)
        assert count_subsets_present(values, n)[0] == sum(1 for v in values if v == 0)
        assert count_supersets_present(values, n)[0] == len(values)
        assert count_supersets_present(values, n)[size - 1] == sum(1 for v in values
                                                                  if v == size - 1)
        # A value is always counted at its own mask.
        for v in values:
            assert count_subsets_present(values, n)[v] >= 1

    # --- Boundary details --------------------------------------------------------------
    assert list(submasks(0b1010)) == [10, 8, 2, 0], "submasks in decreasing order"
    assert sorted(supersets(0b0000, 2)) == [0, 1, 2, 3]
    assert sorted(supersets(0b0011, 2)) == [3]
    assert list(submasks(0)) == [0]
    zero = zeta_subset([0, 0, 0, 0], 2)
    assert zero == [0, 0, 0, 0] and mobius_subset(zero, 2) == [0, 0, 0, 0]
    assert zeta_subset([5], 0) == [5] and mobius_subset([5], 0) == [5]
    assert zeta_superset([5], 0) == [5] and mobius_superset([5], 0) == [5]
    # The n = 3 example with a single element: submasks of every mask stay exact.
    single = [0] * 8
    single[0b0101] = 7
    got = zeta_subset(single, 3)
    assert [mask for mask in range(8) if got[mask] == 7] == [5, 7]
    assert all(got[mask] in (0, 7) for mask in range(8))


def indicator_from(arr, modulus):
    """Turn a count array into a 0/1 indicator array (keeps the OR-convolution checks binary)."""
    return [1 if value % modulus else 0 for value in arr]


if __name__ == "__main__":
    main()
    print("PASS")
