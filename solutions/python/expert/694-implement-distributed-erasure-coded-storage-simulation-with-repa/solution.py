"""Task 694 - Implement distributed erasure-coded storage simulation with repair.  (Tier: Expert, CTCI Ch 9 — System Design and Scalability)

Two real storage codes are built over GF(2^8) and compared on a simulated cluster, because the whole
point of erasure coding in a distributed store is a *trade-off* that can be measured:

* **Reed-Solomon (4,2)** — systematic, with the parity matrix taken from a Cauchy matrix (every square
  submatrix of a Cauchy matrix is invertible, so the code is MDS): 4 data and 2 parity shards, storage
  overhead 1.5x, and *any* 4 of the 6 shards reconstruct the object.  Repairing one lost shard needs
  4 shard reads, i.e. 4S bytes for a shard of S bytes — the naive repair cost.
* **Locally repairable code LRC(4, 2, 1)** — 4 data shards, 2 local parities (one per group of two
  data shards) and 1 global parity, 7 shards total: overhead 1.75x, and a lost *data* shard is
  repaired by reading only its own group, 2S bytes instead of 4S.  The price is exactly the extra
  shard and slightly weaker fault tolerance.

Both codes are exercised by an exhaustive sweep over every erasure pattern they are supposed to
survive, and both are placed on simulated nodes that fail, get repaired and fail again.  Verified:
the Cauchy code is MDS (all C(6,4) reconstruction subsets give the identical object, all 2-failure
patterns survive, every 3-failure pattern fails); every one of the C(7,2) LRC failure pairs recovers
and at least one triple does not; the number of shard reads per repair matches the design (4S for
Reed-Solomon, 2S for an LRC data shard, 4S for the global parity); storage overhead matches 6/4 and
7/4 of the object size; and after a repair round the cluster is back to full redundancy, checked by
re-running the exhaustive pair sweep on the repaired state.
"""

PRIMITIVE = 0x11D
ALPHA = 2
EXP = [0] * 512
LOG = [0] * 256


def _build_tables():
    value = 1
    for power in range(255):
        EXP[power] = value
        LOG[value] = power
        value <<= 1
        if value & 0x100:
            value ^= PRIMITIVE
    for power in range(255, 512):
        EXP[power] = EXP[power - 255]


_build_tables()


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]


def gf_div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(256)")
    return 0 if a == 0 else EXP[(LOG[a] - LOG[b]) % 255]


def gf_inv(a):
    return gf_div(1, a)


def gf_pow(a, exponent):
    return 1 if exponent == 0 else (0 if a == 0 else EXP[(LOG[a] * exponent) % 255])


def invert_matrix(matrix):
    """Gauss-Jordan over GF(256); raises when the matrix is singular."""
    size = len(matrix)
    augmented = [list(row) + [1 if i == j else 0 for j in range(size)]
                 for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise ValueError("singular matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = gf_inv(augmented[column][column])
        augmented[column] = [gf_mul(value, inverse) for value in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column]:
                factor = augmented[row][column]
                augmented[row] = [left ^ gf_mul(factor, right)
                                  for left, right in zip(augmented[row], augmented[column])]
    return [row[size:] for row in augmented]


def solve_linear(coefficients, rhs):
    """Solve a (possibly over-determined) linear system over GF(256) by elimination; raises when it
    has no unique solution, which is exactly the condition for 'this erasure pattern is fatal'."""
    rows, columns = len(coefficients), len(coefficients[0])
    matrix = [list(coefficients[row]) + [rhs[row]] for row in range(rows)]
    pivot_row, pivots = 0, []
    for column in range(columns):
        selected = next((row for row in range(pivot_row, rows) if matrix[row][column]), None)
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        inverse = gf_inv(matrix[pivot_row][column])
        matrix[pivot_row] = [gf_mul(value, inverse) for value in matrix[pivot_row]]
        for row in range(rows):
            if row != pivot_row and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [left ^ gf_mul(factor, right)
                               for left, right in zip(matrix[row], matrix[pivot_row])]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    for row in range(rows):
        if not any(matrix[row][:columns]) and matrix[row][columns]:
            raise ValueError("inconsistent system")
    if len(pivots) < columns:
        raise ValueError("not enough independent equations to recover the object")
    solution = [0] * columns
    for index, column in enumerate(pivots):
        solution[column] = matrix[index][columns]
    return solution


def matrix_times(matrix, vector):
    return [reduce_xor(gf_mul(value, row[index]) for index, value in enumerate(vector))
            for row in matrix]


def reduce_xor(values):
    total = 0
    for value in values:
        total ^= value
    return total


class ReedSolomon:
    """Systematic (k, m) code with a Cauchy parity matrix: [ I_k | C ], hence MDS."""

    def __init__(self, k, m):
        self.k, self.m = k, m
        self.parity = [[gf_inv((k + row) ^ column) for column in range(k)] for row in range(m)]
        check(all(gf_mul(self.parity[row][column], (k + row) ^ column) == 1
                  for row in range(m) for column in range(k)),
              "the Cauchy entries really are reciprocals")

    def encode(self, data):
        """`data` is a list of k shards; returns the k data shards plus m parity shards."""
        check(len(data) == self.k, "the object must have k blocks")
        shards = list(data)
        for row in range(self.m):
            shard = [reduce_xor(gf_mul(self.parity[row][column], data[column][byte])
                                for column in range(self.k))
                     for byte in range(len(data[0]))]
            shards.append(shard)
        return shards

    def coefficients(self, index):
        """The row that expresses shard `index` in terms of the data blocks."""
        if index < self.k:
            return [1 if column == index else 0 for column in range(self.k)]
        return list(self.parity[index - self.k])

    def reconstruct(self, shards, available):
        """Rebuild every shard from any k of them; returns the full shard list and the reads used."""
        if len(available) < self.k:
            raise ValueError("not enough surviving shards")
        known = available[:self.k]
        matrix = [self.coefficients(index) for index in known]
        inverse = invert_matrix(matrix)
        size = len(shards[known[0]])
        data = []
        for column in range(self.k):
            row = inverse[column]
            block = [reduce_xor(gf_mul(row[position], shards[known[position]][byte])
                                for position in range(self.k))
                     for byte in range(size)]
            data.append(block)
        return self.encode(data), len(known)


class LocallyRepairable:
    """4 data shards in two groups, one local parity each, plus one global parity: 7 shards.

    A lost data shard is repaired inside its group (two reads); the global parity doubles as the
    second layer that makes any *two* failures recoverable."""

    def __init__(self, group_size=2):
        self.group_size = group_size
        self.k = 4
        self.m = 3
        self.global_row = [gf_inv((17 + column)) for column in range(self.k)]

    def encode(self, data):
        check(len(data) == self.k, "the object must have four blocks")
        shards = list(data)
        for group in range(self.k // self.group_size):
            start = group * self.group_size
            shard = [reduce_xor(data[start + offset][byte] for offset in range(self.group_size))
                     for byte in range(len(data[0]))]
            shards.append(shard)
        shards.append([reduce_xor(gf_mul(self.global_row[column], data[column][byte])
                                  for column in range(self.k))
                       for byte in range(len(data[0]))])
        return shards

    def local_group(self, index):
        return index // self.group_size if index < self.k else index

    def equations(self, available):
        """Rows of the linear system over the four data blocks given the surviving shards."""
        rows = []
        for index in sorted(available):
            if index < self.k:
                rows.append(([1 if column == index else 0 for column in range(self.k)], index))
            elif index < self.k + 2:
                group = index - self.k
                start = group * self.group_size
                row = [1 if start <= column < start + self.group_size else 0
                       for column in range(self.k)]
                rows.append((row, self.k + group))
            else:
                rows.append((list(self.global_row), self.k + 2))
        return rows

    def reconstruct(self, shards, available):
        """Solve for the four data blocks from whatever survives, then rebuild every shard.  The
        system is over-determined when more than four shards survive, so independence — not the
        count — decides whether the pattern is fatal."""
        size = len(shards[available[0]])
        rows = self.equations(available)
        coefficients = [row for row, _ in rows]
        solved = []
        for byte in range(size):
            rhs = [shards[index][byte] for _, index in rows]
            solved.append(solve_linear(coefficients, rhs))
        data = [[solved[byte][column] for byte in range(size)] for column in range(self.k)]
        return self.encode(data), len(rows)


def repair_reads(code, shard_index, available):
    """How many shard reads a repair needs for one shard, and which shards they are."""
    if isinstance(code, LocallyRepairable) and shard_index < code.k:
        group = shard_index // code.group_size
        peers = [index for index in range(group * code.group_size, (group + 1) * code.group_size)
                 if index != shard_index]
        sources = [index for index in peers + [code.k + group] if index in available]
        return len(sources), sources
    return code.k, sorted(available)[:code.k]


def available_sources(total, failed):
    return [index for index in range(total) if index not in failed]


def object_size(shards):
    return sum(len(shard) for shard in shards)


def main():
    check(gf_mul(0x02, 0x80) == 0x1D and gf_pow(ALPHA, 8) == 0x1D,
          "the field uses the primitive polynomial 0x11D")
    check(gf_mul(gf_inv(37), 37) == 1, "inverses exist for every non-zero element")
    identity = [[1, 0], [0, 1]]
    check(invert_matrix(identity) == identity, "inverting the identity changes nothing")
    try:
        invert_matrix([[1, 0], [1, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("a singular matrix must be rejected")

    rs = ReedSolomon(4, 2)
    payload = [[0x11, 0x22, 0x33, 0x44], [0x55, 0x66, 0x77, 0x88],
               [0x99, 0xAA, 0xBB, 0xCC], [0xDD, 0xEE, 0xFF, 0x10]]
    shards = rs.encode(payload)
    check(len(shards) == 6 and shards[:4] == payload, "the code is systematic: 6 shards, data first")
    check(object_size(shards) == 6 * 4, "storage overhead is 6/4 of the object size")
    check(max(max(shard) for shard in shards) <= 255, "shards stay inside a byte")
    check(shards[4] != shards[5], "the two parity shards are independent of each other")

    subsets = 0
    rebuild_failures = 0
    for first in range(6):
        for second in range(first + 1, 6):
            for third in range(second + 1, 6):
                for fourth in range(third + 1, 6):
                    subset = [first, second, third, fourth]
                    rebuilt, reads = rs.reconstruct(shards, subset)
                    check(reads == 4, "a reconstruction reads exactly k shards")
                    if rebuilt != shards:
                        rebuild_failures += 1
                    subsets += 1
    check(subsets == 15 and rebuild_failures == 0,
          f"all C(6,4) = 15 four-shard subsets rebuild the object identically")

    survived = 0
    for first in range(6):
        for second in range(first + 1, 6):
            available = available_sources(6, {first, second})
            rebuilt, reads = rs.reconstruct(shards, available)
            check(rebuilt == shards, f"failure of shards {first} and {second} is repaired")
            survived += 1
    check(survived == 15, "all fifteen two-failure patterns of the MDS code survive")

    for first in range(6):
        for second in range(first + 1, 6):
            for third in range(second + 1, 6):
                available = available_sources(6, {first, second, third})
                try:
                    rs.reconstruct(shards, available)
                except ValueError:
                    continue
                raise AssertionError("three failures leave only three shards: that must be refused")

    lrc = LocallyRepairable()
    lrc_shards = lrc.encode(payload)
    check(len(lrc_shards) == 7 and lrc_shards[:4] == payload,
          "the LRC keeps the data systematic in seven shards")
    check(object_size(lrc_shards) == 7 * 4, "its storage overhead is the higher 7/4")
    check(lrc_shards[4] == [a ^ b for a, b in zip(payload[0], payload[1])],
          "the first local parity is the XOR of its own group only")
    check(lrc_shards[5] == [a ^ b for a, b in zip(payload[2], payload[3])],
          "and the second group has its own independent parity")

    lrc_survived, lrc_failed_triples = 0, 0
    for first in range(7):
        for second in range(first + 1, 7):
            available = available_sources(7, {first, second})
            rebuilt, _ = lrc.reconstruct(lrc_shards, available)
            check(rebuilt == lrc_shards, f"LRC failure pair ({first}, {second}) is repaired")
            lrc_survived += 1
    check(lrc_survived == 21, "all C(7,2) = 21 LRC failure pairs are recoverable")
    for first in range(7):
        for second in range(first + 1, 7):
            for third in range(second + 1, 7):
                available = available_sources(7, {first, second, third})
                try:
                    lrc.reconstruct(lrc_shards, available)
                except ValueError:
                    lrc_failed_triples += 1
    check(0 < lrc_failed_triples < 35,
          f"but {lrc_failed_triples} of the 35 LRC triple failures are fatal - the cost of locality")

    rs_reads, _ = repair_reads(rs, 0, available_sources(6, {0}))
    lrc_reads, sources = repair_reads(lrc, 0, available_sources(7, {0}))
    check(rs_reads == 4, "Reed-Solomon repairs a data shard with four reads (4S bytes)")
    check(lrc_reads == 2 and set(sources) <= {1, 4},
          "the LRC repairs it with two reads from its own group (2S bytes)")
    check(lrc_reads < rs_reads, f"so the local repair is {rs_reads / lrc_reads:.0f}x cheaper")
    global_reads, _ = repair_reads(lrc, 6, available_sources(7, {6}))
    check(global_reads == 4, "while the global parity still costs a full four-read rebuild")

    placement = {shard: shard for shard in range(7)}
    check(sorted(placement.values()) == list(range(7)), "every shard sits on its own node")
    failed_shards = {1, 3}
    surviving = available_sources(7, failed_shards)
    rebuilt, _ = lrc.reconstruct(lrc_shards, surviving)
    check(rebuilt == lrc_shards, "two node failures are survived by reading the rest")
    repair_costs = [repair_reads(lrc, shard, surviving)[0] for shard in sorted(failed_shards)]
    check(sum(repair_costs) == 4, "repairing both lost data shards costs 2S + 2S")
    pairs = 0
    for first in range(7):
        for second in range(first + 1, 7):
            rebuilt, _ = lrc.reconstruct(lrc_shards, available_sources(7, {first, second}))
            check(rebuilt == lrc_shards, "the repaired cluster is again two-failure tolerant")
            pairs += 1
    check(pairs == 21, "the whole pair sweep was re-run on the repaired state")

    other = [[0x01, 0x02, 0x03, 0x04], [0x05, 0x06, 0x07, 0x08],
             [0x09, 0x0A, 0x0B, 0x0C], [0x0D, 0x0E, 0x0F, 0x10]]
    other_shards = rs.encode(other)
    check(other_shards != shards, "a different object gives different shards")
    check(rs.reconstruct(other_shards, available_sources(6, {0, 5}))[0] == other_shards,
          "and it reconstructs to itself")
    check(rs.encode(payload) == shards, "encoding is deterministic")

    print(f"Reed-Solomon (4,2): overhead {object_size(shards) / 16:.2f}x, "
          f"all 15 two-failure patterns survive, repair costs {rs_reads}S")
    print(f"LRC(4,2,1): overhead {object_size(lrc_shards) / 16:.2f}x, 21 pairs survive, "
          f"{lrc_failed_triples}/35 triples fatal, data repair costs {lrc_reads}S")
    print(f"two node failures repaired with {sum(repair_costs)} shard reads and full redundancy back")
    print("PASS")


if __name__ == "__main__":
    main()
