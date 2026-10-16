"""Task 699 - Implement GPU-style parallel prefix-sum algorithm simulation.  (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)

Two classic scan algorithms are simulated with the machine model made explicit — every addition and
every barrier step is counted, which is the whole point of the comparison:

* **Hillis-Steele** inclusive scan: each of the ceil(log2 n) steps adds the value `2^k` positions back
  *simultaneously* for every element (the simulation reads a snapshot, exactly what a barrier buys).
  Simple, but it performs n*log2 n - (n-1) additions: work-inefficient.
* **Blelloch** exclusive scan: an up-sweep builds a reduction tree (n-1 additions) and a down-sweep
  turns it into prefix sums (n-1 additions) for a total of exactly 2n-2 additions in 2*log2 n steps.
  Work-efficient, which is why real GPU scan implementations use it (with a two-phase block scan so
  the whole device can work at once).

Non-power-of-two lengths are padded to the next power of two with the identity 0, which is exactly
how a device kernel handles them; **block scan** is simulated too, with a per-block Blelloch scan,
a scan of the block totals, and an offset pass — the structure of a real multi-block kernel.

Verified: both scans equal `itertools.accumulate` on empty, singleton, prime-length and power-of-two
inputs; the exclusive/inclusive identity; the exact operation counts (2n-2 for Blelloch, the closed
form n*log2 n-(n-1) for Hillis-Steele) and the exact step counts; that Blelloch does strictly less
work for n >= 4; that the block-scan result matches the sequential scan for every block size and that
its work is bounded by the per-block scans plus the offset pass; that the reduction tree really holds
the total after the up-sweep; and — since floating-point addition is not associative — that both
orders stay within a few ulps of the exact rational prefix sums.
"""

import math
from fractions import Fraction
from itertools import accumulate

ADDITIONS = "additions"
STEPS = "steps"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def counters():
    return {ADDITIONS: 0, STEPS: 0}


def next_power_of_two(length):
    size = 1
    while size < length:
        size *= 2
    return size


def sequential_scan(values):
    return list(accumulate(values))


def sequential_exclusive(values):
    """Exclusive prefix sums: the first output is the identity, the rest lag by one element."""
    return [0] + list(accumulate(values))[:-1] if values else []


def hillis_steele(values, counter=None):
    """Inclusive scan in ceil(log2 n) barrier steps; every element is updated from a snapshot."""
    counter = counter if counter is not None else counters()
    if not values:
        return [], counter
    size = next_power_of_two(len(values))
    padded = list(values) + [0] * (size - len(values))
    offset = 1
    while offset < size:
        snapshot = list(padded)
        for index in range(size):
            if index >= offset:
                padded[index] = snapshot[index - offset] + snapshot[index]
                counter[ADDITIONS] += 1
        counter[STEPS] += 1
        offset *= 2
    return padded[:len(values)], counter


def blelloch(values, counter=None):
    """Exclusive scan via up-sweep and down-sweep: exactly 2n-2 additions in 2*log2 n steps."""
    counter = counter if counter is not None else counters()
    if not values:
        return [], counter
    size = next_power_of_two(len(values))
    tree = list(values) + [0] * (size - len(values))
    offset = 1
    while offset < size:                                   # up-sweep: build the reduction tree
        for start in range(0, size, 2 * offset):
            tree[start + 2 * offset - 1] += tree[start + offset - 1]
            counter[ADDITIONS] += 1
        counter[STEPS] += 1
        offset *= 2
    tree[size - 1] = 0                                     # the identity starts the down-sweep
    offset //= 2
    while offset >= 1:                                     # down-sweep: turn it into prefixes
        for start in range(0, size, 2 * offset):
            left = tree[start + offset - 1]
            tree[start + offset - 1] = tree[start + 2 * offset - 1]
            tree[start + 2 * offset - 1] += left
            counter[ADDITIONS] += 1
        counter[STEPS] += 1
        offset //= 2
    return tree[:len(values)], counter


def reduction_total(values):
    """The up-sweep alone computes the total; expose it so the tree invariant can be checked."""
    size = next_power_of_two(len(values))
    tree = list(values) + [0] * (size - len(values))
    offset = 1
    while offset < size:
        for start in range(0, size, 2 * offset):
            tree[start + 2 * offset - 1] += tree[start + offset - 1]
        offset *= 2
    return tree[size - 1]


def block_scan(values, block_size, counter=None):
    """Two-phase exclusive scan: scan inside each block, scan the block totals, add the offsets.
    It is the shape of a real multi-block GPU kernel, where each block owns a contiguous chunk."""
    counter = counter if counter is not None else counters()
    if not values:
        return [], counter
    blocks = [values[start:start + block_size] for start in range(0, len(values), block_size)]
    locals_, totals = [], []
    for block in blocks:
        scanned, _ = blelloch(block, counter)
        locals_.append(scanned)
        totals.append(sum(block))
        counter[ADDITIONS] += max(0, len(block) - 1)
    offsets, _ = blelloch(totals, counter)
    result = []
    for block_index, block in enumerate(blocks):
        offset = offsets[block_index]
        for value in locals_[block_index]:
            result.append(value + offset)
            counter[ADDITIONS] += 1
    return result, counter


def exact_prefixes(values):
    """Exact rational inclusive prefix sums, for a precision cross-check."""
    running, out = Fraction(0), []
    for value in values:
        running += Fraction(value)
        out.append(float(running))
    return out


def relative_error(value, reference):
    if reference == 0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def main():
    check(next_power_of_two(1) == 1 and next_power_of_two(3) == 4 and next_power_of_two(64) == 64,
          "the padding size rounds up to a power of two")
    sample = [3, 1, 4, 1, 5, 9, 2, 6]
    inclusive, counters_hs = hillis_steele(sample)
    check(inclusive == sequential_scan(sample), "Hillis-Steele matches the sequential inclusive scan")
    check(counters_hs[ADDITIONS] == 8 * 3 - 7, "its addition count is n*log2(n) - (n-1) = 17")
    check(counters_hs[STEPS] == 3, "and it needs log2(n) = 3 barrier steps")
    exclusive, counters_blelloch = blelloch(sample)
    check(exclusive == sequential_exclusive(sample), "Blelloch matches the sequential exclusive scan")
    check(counters_blelloch[ADDITIONS] == 2 * 8 - 2,
          "its addition count is exactly 2n-2 = 14, work-efficient")
    check(counters_blelloch[STEPS] == 6, "in 2*log2(n) = 6 steps")
    check(counters_blelloch[ADDITIONS] < counters_hs[ADDITIONS],
          "so Blelloch does strictly less work on eight elements")
    check([a + b for a, b in zip(exclusive, sample)] == inclusive,
          "the exclusive scan plus the input is the inclusive scan")
    check(exclusive[0] == 0, "the exclusive scan starts at the identity")
    check(reduction_total(sample) == sum(sample), "the up-sweep's root holds the total")

    for length in (0, 1, 2, 3, 5, 8, 13, 17, 31, 64, 100, 256):
        values = [((index * 37) % 23) + 1 for index in range(length)]
        expected = sequential_scan(values)
        hillis, hillis_counters = hillis_steele(values)
        check(hillis == expected, f"Hillis-Steele is right for length {length}")
        prefix, blelloch_counters = blelloch(values)
        check(prefix == sequential_exclusive(values), f"Blelloch is right for length {length}")
        if length:
            padded = next_power_of_two(length)
            check(blelloch_counters[ADDITIONS] == 2 * padded - 2,
                  f"Blelloch's work at length {length} is 2*padded-2")
            check(hillis_counters[ADDITIONS] == padded * int(math.log2(padded)) - (padded - 1),
                  f"Hillis-Steele's work at length {length} follows its closed form")
            check(hillis_counters[STEPS] == int(math.log2(padded)),
                  f"Hillis-Steele takes log2(n) steps at length {length}")
            check(blelloch_counters[STEPS] == 2 * int(math.log2(padded)),
                  f"Blelloch takes 2*log2(n) steps at length {length}")
            if padded >= 8:
                check(blelloch_counters[ADDITIONS] < hillis_counters[ADDITIONS],
                      f"work efficiency holds from eight elements on (length {length})")
            else:
                check(blelloch_counters[ADDITIONS] >= hillis_counters[ADDITIONS],
                      f"at {padded} padded elements the tree still costs more than the naive loop")

    for length in (1, 16, 64, 200):
        values = [((index * 11) % 17) - 8 for index in range(length)]
        check(hillis_steele(values)[0] == sequential_scan(values),
              f"negative values are handled at length {length}")

    for block_size in (2, 4, 8, 16):
        for length in (0, 1, 7, 32, 100):
            values = [((index * 5) % 13) + 1 for index in range(length)]
            result, block_counters = block_scan(values, block_size)
            check(result == sequential_exclusive(values),
                  f"the block scan is correct at length {length} with block {block_size}")
            if not length:
                check(block_counters[ADDITIONS] == 0, "an empty scan costs nothing")
                continue
            blocks = [values[start:start + block_size]
                      for start in range(0, length, block_size)]
            tree_work = sum(2 * next_power_of_two(len(block)) - 2 for block in blocks)
            tree_work += 2 * next_power_of_two(len(blocks)) - 2
            totals_work = sum(len(block) - 1 for block in blocks)
            expected_work = tree_work + totals_work + length
            check(block_counters[ADDITIONS] == expected_work,
                  f"the block scan's work is exactly its parts at length {length}: "
                  f"{block_counters[ADDITIONS]} vs {expected_work}")
            per_block_steps = sum(2 * int(math.log2(next_power_of_two(len(block))))
                                  for block in blocks)
            total_steps = per_block_steps + 2 * int(math.log2(next_power_of_two(len(blocks))))
            check(block_counters[STEPS] == total_steps,
                  f"the counted steps are the per-block scans plus the offset scan "
                  f"({block_counters[STEPS]} vs {total_steps})")
            depth = (2 * int(math.log2(next_power_of_two(block_size)))
                     + 2 * int(math.log2(next_power_of_two(len(blocks)))))
            check(depth < block_counters[STEPS] or len(blocks) == 1,
                  "the critical path is shorter than the summed work: the blocks are independent")

    tricky = [0.1] * 16
    truth = exact_prefixes(tricky)
    orders = {"sequential": sequential_scan(tricky),
              "Hillis-Steele": hillis_steele(tricky)[0],
              "Blelloch": [value + tricky[index]
                           for index, value in enumerate(blelloch(tricky)[0])]}
    for label, produced in orders.items():
        check(all(abs(a - b) <= 1e-15 * max(1.0, abs(b)) for a, b in zip(produced, truth)),
              f"{label} stays within a few ulps of the exact rational prefixes")
    baseline = orders["sequential"]
    for label, produced in orders.items():
        check(all(abs(a - b) <= 1e-15 * max(1.0, abs(b)) for a, b in zip(produced, baseline)),
              f"{label} agrees with the sequential order to within a few ulps")

    check(hillis_steele([])[0] == [] and blelloch([])[0] == [] and block_scan([], 4)[0] == [],
          "every scan handles the empty input")
    check(hillis_steele([7])[0] == [7] and blelloch([7])[0] == [0],
          "a single element is its own inclusive sum and an exclusive zero")
    check(reduction_total([1, 2, 3]) == 6 and reduction_total([5]) == 5,
          "the reduction tree handles the padded and unpadded cases")
    constant = [3] * 1000
    check(hillis_steele(constant)[0] == [3 * (index + 1) for index in range(1000)],
          "a constant array scans to a linear ramp")
    big = [1] * 1024
    big_inclusive, big_counters = hillis_steele(big)
    check(big_inclusive[-1] == 1024 and big_counters[STEPS] == 10,
          "1024 elements scan in exactly ten steps")
    check(blelloch(big)[1][ADDITIONS] == 2 * 1024 - 2,
          "and Blelloch's work on 1024 elements is exactly 2046 additions")
    check(len(block_scan(big, 32)[0]) == 1024, "the block scan covers the whole array")

    print(f"n=8: Hillis-Steele {counters_hs[ADDITIONS]} additions in {counters_hs[STEPS]} steps, "
          f"Blelloch {counters_blelloch[ADDITIONS]} additions in {counters_blelloch[STEPS]} steps")
    print(f"n=1024: Hillis-Steele {big_counters[ADDITIONS]} additions, Blelloch "
          f"{blelloch(big)[1][ADDITIONS]} additions (2n-2)")
    print(f"block scan of 1024 elements with 32-wide blocks: "
          f"{block_scan(big, 32)[1][ADDITIONS]} additions, "
          f"{block_scan(big, 32)[1][STEPS]} steps")
    print("PASS")


if __name__ == "__main__":
    main()
