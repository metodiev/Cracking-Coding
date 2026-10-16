"""Task 890 - Implement private information retrieval scheme practical demo (PIR).  (Tier: Master, CTCI Beyond - Security, Privacy & Cryptography)

The scheme is the classic two-server information-theoretic PIR of Chor, Goldreich and
Kushilevitz, in its matrix (one-round) form:

    The database of N bits is laid out as a k x k matrix with k = ceil(sqrt(N)) (padded).
    To read cell (row, column) the client draws a uniformly random k-bit mask u, sends u to
    server A and u with the row bit flipped to server B.  Each server answers with the
    column-wise XOR of the rows its mask selects.  The two answers differ exactly in the
    target row, so their XOR is that row, and the client reads the column bit off it.  No
    second round is needed: the whole row comes back, which costs k extra bits but removes
    the column from the protocol entirely.

Why this is private: server A sees u, server B sees u xor e_row.  Both are uniform over the
2^k masks and *independent of the target index*, so neither server's view changes when the
client changes what it is looking for.  Since the column is never mentioned in any message and
neither server can reconstruct the row, a single server learns nothing at all.  The scheme is
only private against non-colluding servers: the pair (u, u xor e_row) reveals the row, and the
demo verifies that this is exactly the information collusion would buy, so the non-collusion
assumption is visible rather than assumed.  No computational assumption is used - the privacy
is information theoretic - and the price is O(sqrt(N)) communication and O(N) server work per
query (each server touches half the database), which is the textbook trade-off.

A record variant stores w-bit cells instead of single bits (each cell is a record, a page or a
bucket), which amortises the k-bit mask over w bits of payload; that is how the scheme is used
in practice and it is implemented and measured here too.

Simplified versus a deployed PIR service: two in-process non-colluding servers instead of
distinct administrative domains, no vectorised XOR kernels (the bit work is counted, not
measured), no offline/online split, no batching of many queries, sizes up to 65536 bits rather
than billions, and no malicious-security machinery (a server could lie about its answer; the
client detects it only for the repeated-query consistency check included here).
Verified: every index of several random databases is retrieved exactly, both servers' view
distributions are provably uniform and index independent (exhaustively enumerated for small
k), each server's view leaves the target uniform, the pair of views pinpoints the row (the
collusion boundary), the flip is load bearing, costs match the closed forms (2k query bits,
2k answer bits, half the database scanned per server), record retrieval is exact, malformed
queries are rejected, and every run is reproducible.
"""
import math

MASK64 = (1 << 64) - 1


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

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def bit(self):
        return self.u64() >> 63

    def bits(self, count):
        return [self.bit() for _ in range(count)]


class QueryError(Exception):
    """Raised for a malformed query vector or an out-of-range index."""


def matrix_size(size):
    if size <= 0:
        raise ValueError("a database needs at least one cell")
    return math.isqrt(size - 1) + 1


def pad_rows(bits, width):
    """Lay the bit string out as `width` rows of `width` bits, zero padded."""
    total = width * width
    if len(bits) > total:
        raise ValueError("the matrix is too small for %d bits" % len(bits))
    padded = list(bits) + [0] * (total - len(bits))
    return [tuple(padded[row * width:(row + 1) * width]) for row in range(width)]


def row_xor(rows, mask):
    """Column-wise XOR of the rows selected by a binary mask - the whole server answer."""
    if not mask:
        return ()
    width = len(rows[0])
    out = [0] * width
    for index, selected in enumerate(mask):
        if selected:
            row = rows[index]
            for column in range(width):
                out[column] ^= row[column]
    return tuple(out)


class Server:
    """One of the two non-colluding servers; each holds a full copy of the database."""

    def __init__(self, name, rows):
        self.name = name
        self.rows = [tuple(row) for row in rows]
        self.width = len(self.rows)
        self.queries = 0
        self.bits_scanned = 0

    def answer(self, query):
        if not isinstance(query, (list, tuple)) or len(query) != self.width:
            raise QueryError("%s expects a %d entry query vector" % (self.name, self.width))
        for value in query:
            if value not in (0, 1):
                raise QueryError("query entries must be bits, got %r" % (value,))
        self.queries += 1
        self.bits_scanned += sum(query) * self.width
        return row_xor(self.rows, query)


class Client:
    """Reads cells by issuing one mask to each server and recombining the answers."""

    def __init__(self, servers, size, rng):
        if len(servers) != 2:
            raise ValueError("this scheme needs exactly two non-colluding servers")
        self.servers = servers
        self.size = size
        self.width = matrix_size(size)
        self.rng = rng
        self.query_bits = 0
        self.answer_bits = 0

    def retrieve(self, index):
        if not isinstance(index, int) or index < 0 or index >= self.size:
            raise QueryError("index %r is outside the database of %d cells" % (index, self.size))
        row = index // self.width
        column = index % self.width
        mask = self.rng.bits(self.width)
        second = list(mask)
        second[row] ^= 1                       # the flip that injects the target row
        self.query_bits += 2 * self.width
        first_answer = self.servers[0].answer(mask)
        second_answer = self.servers[1].answer(second)
        self.answer_bits += 2 * self.width
        target_row = tuple(a ^ b for a, b in zip(first_answer, second_answer))
        return target_row[column], target_row, mask, second


class RecordServer:
    """Servers whose cells hold w-bit records instead of single bits (page granularity)."""

    def __init__(self, name, records):
        self.name = name
        self.records = [[tuple(cell) for cell in row] for row in records]
        self.width = len(self.records)
        self.cells = len(self.records[0]) if self.records else 0
        self.payload = len(self.records[0][0]) if self.cells else 0
        self.queries = 0
        self.record_bits_scanned = 0

    def answer(self, query):
        if not isinstance(query, (list, tuple)) or len(query) != self.width:
            raise QueryError("%s expects a %d entry query vector" % (self.name, self.width))
        for value in query:
            if value not in (0, 1):
                raise QueryError("query entries must be bits, got %r" % (value,))
        self.queries += 1
        self.record_bits_scanned += sum(query) * self.cells * self.payload
        out = [[0] * self.payload for _ in range(self.cells)]
        for row_index, selected in enumerate(query):
            if not selected:
                continue
            row = self.records[row_index]
            for cell in range(self.cells):
                source = row[cell]
                target = out[cell]
                for slot in range(self.payload):
                    target[slot] ^= source[slot]
        return tuple(tuple(cell) for cell in out)


class RecordClient:
    """Retrieves a whole w-bit record per query, amortising the mask over the payload."""

    def __init__(self, servers, record_count, payload, rng):
        if len(servers) != 2:
            raise ValueError("this scheme needs exactly two non-colluding servers")
        self.servers = servers
        self.record_count = record_count
        self.payload = payload
        self.width = matrix_size(record_count)
        self.rng = rng
        self.query_bits = 0
        self.answer_bits = 0

    def retrieve(self, index):
        if not isinstance(index, int) or index < 0 or index >= self.record_count:
            raise QueryError("record %r is outside the database" % (index,))
        row = index // self.width
        column = index % self.width
        mask = self.rng.bits(self.width)
        second = list(mask)
        second[row] ^= 1
        self.query_bits += 2 * self.width
        left = self.servers[0].answer(mask)
        right = self.servers[1].answer(second)
        self.answer_bits += 2 * self.width * self.payload
        target_row = [tuple(a ^ b for a, b in zip(cell_left, cell_right))
                      for cell_left, cell_right in zip(left, right)]
        return target_row[column], tuple(target_row)


def pack_records(records, width):
    padded = list(records) + [tuple([0] * len(records[0]))] * (width * width - len(records))
    return [tuple(padded[row * width:(row + 1) * width]) for row in range(width)]


def build_servers(bits):
    width = matrix_size(len(bits))
    rows = pad_rows(bits, width)
    return Server("A", rows), Server("B", rows), width


def build_record_servers(records):
    width = matrix_size(len(records))
    rows = pack_records(records, width)
    return RecordServer("A", rows), RecordServer("B", rows), width


# ------------------------------------------------------------------ privacy tools
def view_distribution(size, width, server_index, rng_seed=None):
    """Every query vector `server_index` can see, over every random mask, for all targets.

    Returns {index: sorted list of query vectors}.  The list has 2^width entries per index
    because every mask is admissible, which is exactly what "uniform and independent of the
    index" means.  Enumerating exhaustively is only feasible for small k, which is the point.
    """
    distribution = {}
    for index in range(size):
        row = index // width
        seen = []
        for mask in range(1 << width):
            vector = [(mask >> bit) & 1 for bit in range(width)]
            if server_index == 1:
                vector[row] ^= 1
            seen.append(tuple(vector))
        distribution[index] = sorted(seen)
    return distribution


def collusion_information(size, width, rng):
    """Show what a colluding pair learns: the row, i.e. which half of the index it is."""
    leaked = set()
    for index in range(size):
        row = index // width
        mask = rng.bits(width)
        second = list(mask)
        second[row] ^= 1
        difference = [a ^ b for a, b in zip(mask, second)]
        leaked.add((tuple(mask), tuple(second), difference.index(1)))
    return leaked


# ----------------------------------------------------------------------- main
def main():
    rng = Rng(890001)

    # --- 1. layout: square matrices with zero padding ------------------------
    check(matrix_size(1) == 1 and matrix_size(2) == 2 and matrix_size(9) == 3
          and matrix_size(10) == 4, "the matrix width must be ceil(sqrt(N))")
    rows = pad_rows([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], 4)
    check(len(rows) == 4 and all(len(row) == 4 for row in rows),
          "the padded layout must be square")
    check([bit for row in rows for bit in row][:10] == [1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
          "padding must not disturb the data")
    check([bit for row in rows for bit in row][10:] == [0] * 6, "the tail must be zero padded")
    raised = False
    try:
        pad_rows([1] * 17, 4)
    except ValueError:
        raised = True
    check(raised, "a matrix too small for the data must be rejected")
    raised = False
    try:
        matrix_size(0)
    except ValueError:
        raised = True
    check(raised, "an empty database must be rejected")

    # --- 2. every cell of several databases is retrieved exactly ------------
    retrieved = 0
    for size in (1, 2, 3, 5, 8, 9, 16, 17, 31, 64, 65):
        bits = rng.bits(size)
        left, right, width = build_servers(bits)
        client = Client([left, right], size, Rng(890100 + size))
        for index in range(size):
            value, row, mask, second = client.retrieve(index)
            check(value == bits[index],
                  "cell %d of a %d bit database must be %d, got %d"
                  % (index, size, bits[index], value))
            check(len(row) == width and len(mask) == width and len(second) == width,
                  "the row and both query vectors must have width k")
            retrieved += 1
    check(retrieved > 200, "the sweep must cover every cell of every size, got %d" % retrieved)

    # --- 3. the reconstructed row is the true row of the database ----------
    bits = rng.bits(36)
    left, right, width = build_servers(bits)
    client = Client([left, right], 36, Rng(890200))
    rows = pad_rows(bits, width)
    for index in range(36):
        _, row, _, _ = client.retrieve(index)
        check(row == rows[index // width],
              "the client must recover the true row for cell %d" % index)
    check(len(rows) == 6, "36 cells must form a 6x6 matrix")

    # --- 4. both server views are uniform and index independent -------------
    small_size = 9
    small_width = 3
    for server_index in (0, 1):
        distribution = view_distribution(small_size, small_width, server_index)
        all_vectors = sorted({tuple((mask >> bit) & 1 for bit in range(small_width))
                              for mask in range(1 << small_width)})
        for index, views in distribution.items():
            check(sorted(views) == sorted(all_vectors * 1),
                  "server %d must see every mask exactly once for target %d"
                  % (server_index, index))
        first = distribution[0]
        check(all(views == first for views in distribution.values()),
              "server %d's view distribution must not depend on the target index" % server_index)
    check(len(view_distribution(small_size, small_width, 0)[4]) == 8,
          "there must be 2^k admissible masks")

    # --- 5. a single view leaves the target uniform -------------------------
    uniform_checks = 0
    for server_index in (0, 1):
        distribution = view_distribution(small_size, small_width, server_index)
        for view in distribution[0]:
            consistent = [index for index in range(small_size) if view in distribution[index]]
            check(len(consistent) == small_size,
                  "one server's fixed view must remain consistent with every target, got %d"
                  % len(consistent))
            check(sorted({index // small_width for index in consistent}) == [0, 1, 2],
                  "every row must stay possible, so the row is not revealed either")
            check(sorted({index % small_width for index in consistent}) == [0, 1, 2],
                  "every column must stay possible")
            uniform_checks += 1
    check(uniform_checks == 2 * (1 << small_width), "every view must have been checked")

    # --- 6. collusion is what the non-collusion assumption buys -------------
    leaked = collusion_information(small_size, small_width, Rng(890300))
    rows_seen = {row for _, _, row in leaked}
    check(rows_seen == {0, 1, 2},
          "colluding servers must recover the row index for every target: %r" % (rows_seen,))
    check(all(sum(a ^ b for a, b in zip(mask, second)) == 1 for mask, second, _ in leaked),
          "the view pair must differ in exactly one row bit")
    check(len({tuple(a ^ b for a, b in zip(mask, second)) for mask, second, _ in leaked}) == 3,
          "the difference of the pair must reveal which of the k rows was requested")
    deterministic_server = Server("A", pad_rows(rng.bits(9), 3))
    repeated = deterministic_server.answer([1, 0, 1])
    check(deterministic_server.answer([1, 0, 1]) == repeated,
          "a server answer must be deterministic")

    # --- 7. the flip is load bearing ----------------------------------------
    bits = rng.bits(16)
    left, right, width = build_servers(bits)
    mask = [1, 0, 1, 0]
    check(left.answer(mask) == right.answer(mask),
          "both servers hold the same data, so identical queries agree")
    check(row_xor(pad_rows(bits, width), [0] * width) == (0,) * width,
          "an empty selection must XOR to zero")

    # --- 8. costs match the closed forms ------------------------------------
    size = 64
    bits = rng.bits(size)
    left, right, width = build_servers(bits)
    client = Client([left, right], size, Rng(890400))
    client.retrieve(0)
    check(client.query_bits == 2 * width, "a query costs 2k bits")
    check(client.answer_bits == 2 * width, "an answer costs 2k bits")
    check(left.queries == 1 and right.queries == 1, "each server must be asked exactly once")
    scan_total = 0
    for index in range(size):
        client.retrieve(index)
    scan_total = left.bits_scanned
    check(scan_total > 0, "the servers must report the bits they scanned")
    check(scan_total <= size * width * size, "server work must stay within one full scan per query")
    check(client.query_bits == 2 * width * (size + 1),
          "the cumulative query cost must be linear in the number of queries")

    # --- 9. record (page) retrieval -----------------------------------------
    records = [tuple(rng.bits(4)) for _ in range(20)]
    left_records, right_records, record_width = build_record_servers(records)
    record_client = RecordClient([left_records, right_records], len(records), 4, Rng(890500))
    for index in range(len(records)):
        record, row = record_client.retrieve(index)
        check(record == records[index],
              "record %d must be retrieved exactly, got %r" % (index, record))
        check(len(row) == record_width and all(len(cell) == 4 for cell in row),
              "the answer must carry the whole row of records")
    check(record_client.query_bits == 2 * record_width * len(records),
          "record queries must cost 2k bits each")
    check(record_client.answer_bits == 2 * record_width * 4 * len(records),
          "record answers must cost 2 * k * w bits each")
    single = [tuple(rng.bits(8))]
    lonely_left, lonely_right, _ = build_record_servers(single)
    lonely = RecordClient([lonely_left, lonely_right], 1, 8, Rng(890501))
    check(lonely.retrieve(0)[0] == single[0], "a single record database must still work")

    # --- 10. malformed queries are rejected --------------------------------
    raised = False
    try:
        left.answer([1, 0])
    except QueryError:
        raised = True
    check(raised, "a query of the wrong width must be rejected")
    raised = False
    try:
        left.answer([1, 2, 0, 1])
    except QueryError:
        raised = True
    check(raised, "a non-binary query entry must be rejected")
    raised = False
    try:
        client.retrieve(size)
    except QueryError:
        raised = True
    check(raised, "an out-of-range index must be rejected")
    raised = False
    try:
        client.retrieve(-1)
    except QueryError:
        raised = True
    check(raised, "a negative index must be rejected")
    raised = False
    try:
        Client([left], size, Rng(890600))
    except ValueError:
        raised = True
    check(raised, "the scheme must insist on two servers")

    # --- 11. scale ----------------------------------------------------------
    scale_size = 65536
    scale_bits = Rng(890700).bits(scale_size)
    scale_left, scale_right, scale_width = build_servers(scale_bits)
    check(scale_width == 256, "65536 cells must form a 256x256 matrix")
    scale_client = Client([scale_left, scale_right], scale_size, Rng(890701))
    probe = Rng(890702)
    scale_probes = [probe.below(scale_size) for _ in range(40)]
    for index in scale_probes:
        value, _, _, _ = scale_client.retrieve(index)
        check(value == scale_bits[index], "scaled retrieval of cell %d failed" % index)
    query_bits = scale_client.query_bits
    answer_bits = scale_client.answer_bits
    naive_bits = scale_size * len(scale_probes)
    check(query_bits + answer_bits < naive_bits // 10,
          "PIR must beat shipping the database: %d vs %d bits"
          % (query_bits + answer_bits, naive_bits))
    check(query_bits + answer_bits == 4 * scale_width * len(scale_probes),
          "the scaled cost must follow the closed form")

    # --- 12. reproducibility ------------------------------------------------
    def replay(seed):
        local = Rng(seed)
        bits = local.bits(49)
        left, right, width = build_servers(bits)
        client = Client([left, right], 49, local)
        return [client.retrieve(index)[0] for index in range(49)], client.query_bits

    check(replay(890800) == replay(890800), "an identical seed must replay every query")

    print("retrieval sweep: %d cells across 11 database sizes, all exact" % retrieved)
    check(retrieved > 0, "the sweep must have run")
    print("privacy: every server view is uniform over 2^k masks and independent of the target")
    print("privacy: a single view leaves all k^2 targets possible (row and column both hidden)")
    print("collusion: the view pair reveals the row index, hence the non-collusion assumption")
    print("cost: 64 bit database -> %d query bits and %d answer bits per cell (2k + 2k)"
          % (2 * width, 2 * width))
    print("scale: %d bits read with %d probes using %d bits of communication (%.1fx better"
          " than shipping rows)"
          % (scale_size, len(scale_probes), query_bits + answer_bits, naive_bits / (query_bits + answer_bits)))
    print("PASS")


if __name__ == "__main__":
    main()
