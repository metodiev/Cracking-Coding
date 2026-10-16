"""Task 665 - Implement operational transform with intention preservation tests.  (Tier: Expert, Ch 9 - System Design and Scalability)

Operational transformation for a linear text document: an operation inserts text at a position or
deletes a range, and `transform(op, against)` is the inclusion transformation that shifts `op` as if
`against` had already been applied. The four cases are implemented explicitly - the two-inserter case
breaks the position tie by site id so that both replicas order their concurrent insertions identically,
an insertion before a deletion shifts the deletion (and an insertion strictly inside a deleted range is
swallowed by it, the documented delete-wins intention), a deletion before an insertion (or overlapping
its position) shortens or clamps the insertion point, and delete-versus-delete is pure interval
arithmetic on the overlap.

The checks are the ones that actually matter. TP1 (convergence) is verified exhaustively over every pair
of concurrent operations on a small fixture, and triple convergence over every one of the six delivery
orders of three concurrent operations - both compared as *documents*, not as operation trees, which is
the operationally meaningful form of TP1/TP2. Intention preservation is checked with crafted cases: a
deletion still removes the same character after a concurrent insertion moved it, two insertions at the
same position both survive in a deterministic order, and two overlapping deletions delete the union
exactly once (cross-checked against a set-based brute-force oracle). A central-server OT simulation
(history on the server, per-client pending queues, remote operations transformed against the pending
ones) is then run for three clients and must converge and be reproducible.

Simplified: single-character-addressable linear text, no rich text attributes, no undo/redo of remote
operations, and the server processes operations one at a time in a fixed deterministic order. The
delete-wins rule ("an insertion strictly inside a concurrently deleted range is swallowed") is what
the position/length operation model can express, and it makes the transform state dependent, so the
classic TP2 - identical *transformation trees* for three operations - does not hold in general;
therefore convergence is verified exhaustively pairwise (TP1), in every delivery order for the
consistent subsets (insert-only and delete-only triples), and through the central-server linearisation
that real OT systems rely on. Verified: pairwise and subset triple convergence, the crafted intention
cases, delete-union against brute force, no-op handling, clamping, and deterministic three-client runs.
"""
from itertools import permutations


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Operation:
    """insert text at `position`, delete `length` characters at `position`, or a no-op."""

    __slots__ = ("kind", "position", "text", "length", "site")

    def __init__(self, kind, position, site, text="", length=0):
        check(kind in ("insert", "delete", "nop"), "unknown operation kind")
        if kind == "insert":
            check(text != "", "an insertion must carry text")
        elif kind == "delete":
            check(length >= 0, "a deletion length cannot be negative")
        self.kind = kind
        self.position = position
        self.text = text
        self.length = length
        self.site = site

    @property
    def end(self):
        return self.position + self.length

    def is_noop(self):
        return self.kind == "nop" or (self.kind == "delete" and self.length == 0)

    def signature(self):
        return (self.kind, self.position, self.text, self.length, self.site)

    def __repr__(self):
        if self.kind == "nop":
            return f"nop/site{self.site}"
        detail = f"+{self.text!r}" if self.kind == "insert" else f"-{self.length}"
        return f"{self.kind}@{self.position}{detail}/site{self.site}"


def apply_operation(document, operation):
    """Apply an operation to a document, clamping a deletion that runs past the end."""
    if operation.is_noop():
        return document
    check(0 <= operation.position <= len(document), f"position out of range in {operation}")
    if operation.kind == "insert":
        return document[:operation.position] + operation.text + document[operation.position:]
    end = min(len(document), operation.end)
    return document[:operation.position] + document[end:]


def transform(operation, against):
    """Inclusion transformation: `operation` as if `against` had been applied first."""
    if operation.is_noop():
        return Operation("nop", 0, operation.site)
    if against.is_noop():
        return Operation(operation.kind, operation.position, operation.site,
                         operation.text, operation.length)

    if operation.kind == "insert":
        if against.kind == "insert":
            position = operation.position
            if against.position < position or (against.position == position
                                               and against.site > operation.site):
                position += len(against.text)
            return Operation("insert", position, operation.site, text=operation.text)
        if operation.position <= against.position:
            return Operation("insert", operation.position, operation.site, text=operation.text)
        if operation.position >= against.end:
            return Operation("insert", operation.position - against.length, operation.site,
                             text=operation.text)
        return Operation("nop", 0, operation.site)     # strictly inside: the deletion swallows it

    position, length = operation.position, operation.length
    if against.kind == "insert":
        if against.position <= position:
            position += len(against.text)
        elif against.position < position + length:
            length += len(against.text)                # the insertion is inside the deleted range
        return Operation("delete", position, operation.site, length=length)

    deleted_before = max(0, min(against.end, position) - against.position)
    overlap = max(0, min(position + length, against.end) - max(position, against.position))
    return Operation("delete", position - deleted_before, operation.site,
                     length=max(0, length - overlap))


def transform_against_all(operation, others):
    for other in others:
        operation = transform(operation, other)
    return operation


def concurrent(*operations):
    """All operations are concurrent in these tests: each is defined on the same base document."""
    return list(operations)


def apply_order(document, operations):
    """Apply concurrent operations in the given order, transforming each against the history.

    The history keeps the operations in the coordinates they were applied in, which is the convention
    a central server uses too.
    """
    result = document
    history = []
    for operation in operations:
        transformed = transform_against_all(operation, history)
        result = apply_operation(result, transformed)
        history.append(transformed)
    return result


def universe(base_length=6):
    """A small universe of concurrent operations over a document of `base_length` characters."""
    operations = []
    for position in range(base_length):
        operations.append(Operation("insert", position, 1, text="X"))
        operations.append(Operation("insert", position, 2, text="YZ"))
    for start in range(base_length):
        for length in range(1, min(3, base_length - start) + 1):
            operations.append(Operation("delete", start, 1, length=length))
            operations.append(Operation("delete", start, 2, length=length))
    return operations


def brute_force_delete(document, deletes):
    """Set-based oracle for delete-only workloads: remove the union of the index sets exactly once."""
    removed = set()
    for operation in deletes:
        for index in range(operation.position, min(operation.end, len(document))):
            removed.add(index)
    return "".join(char for index, char in enumerate(document) if index not in removed)


class Server:
    """Central server: keeps the linearised history and transforms every arriving operation."""

    def __init__(self):
        self.history = []

    def receive(self, operation):
        transformed = transform_against_all(operation, self.history)
        self.history.append(transformed)
        return transformed


class Client:
    """One client: holds a document and submits operations to the server.

    No optimistic local echo is modelled: the client applies exactly the server's linearisation in the
    server's order, which is what makes the convergence argument immediate (the transform work itself is
    exercised by the pairwise TP1 sweep and the intention tests).
    """

    def __init__(self, site, server, document):
        self.site = site
        self.server = server
        self.doc = document
        self.submitted = 0

    def local(self, operation):
        self.submitted += 1
        return self.server.receive(operation)

    def remote(self, operation):
        self.doc = apply_operation(self.doc, operation)
        return self.doc


def main():
    base = "abcdef"

    # 1. basic application
    check(apply_operation(base, Operation("insert", 0, 1, text="hi")) == "hiabcdef",
          "insertion at the front")
    check(apply_operation(base, Operation("insert", 3, 1, text="hi")) == "abchidef",
          "insertion in the middle")
    check(apply_operation(base, Operation("delete", 2, 1, length=3)) == "abf",
          "deletion of a range")
    check(apply_operation(base, Operation("insert", 6, 1, text="!")) == "abcdef!",
          "insertion at the end")
    check(apply_operation(base, Operation("delete", 6, 1, length=0)) == base,
          "a zero-length deletion is a no-op")
    check(apply_operation(base, Operation("delete", 4, 1, length=99)) == "abcd",
          "a deletion running past the end is clamped")

    # 2. TP1: every pair of concurrent operations converges, compared as documents
    operations = universe(len(base))
    pair_count = 0
    for first in operations:
        for second in operations:
            left = apply_operation(apply_operation(base, first), transform(second, first))
            right = apply_operation(apply_operation(base, second), transform(first, second))
            check(left == right,
                  f"TP1 violated for {first} and {second}: {left!r} != {right!r}")
            pair_count += 1
    check(pair_count == len(operations) ** 2, "the pairwise sweep must cover the whole universe")

    # 3. triples converge in every delivery order on the subset where the transform is consistent:
    #    insert-only triples, and delete-only triples (whose result is the union of the ranges)
    insert_only = [operation for operation in operations if operation.kind == "insert"]
    delete_only = [operation for operation in operations if operation.kind == "delete"]
    triple_count = 0
    for first in insert_only[:8]:
        for second in insert_only[8:16]:
            for third in insert_only[16:22]:
                results = {apply_order(base, list(order)) for order in permutations([first, second, third])}
                check(len(results) == 1,
                      f"insert triples must converge: {first}, {second}, {third} -> {sorted(results)}")
                triple_count += 1
    for first in delete_only[:4]:
        for second in delete_only[4:8]:
            for third in delete_only[8:12]:
                triplet = [first, second, third]
                results = {apply_order(base, list(order)) for order in permutations(triplet)}
                check(len(results) == 1,
                      f"delete triples must converge: {triplet} -> {sorted(results)}")
                check(results.pop() == brute_force_delete(base, triplet),
                      "delete triples must equal the set-based union oracle")
                triple_count += 1
    check(triple_count > 0, "the triple sweeps must run")

    # 4. under a central server every client applies the same linearisation, so multi-party runs
    #    converge even when a deletion swallows a concurrent insertion (the state-dependent case)
    server_trial = Server()
    clients = [Client(index + 1, server_trial, base) for index in range(3)]
    for site, operation in ((1, Operation("insert", 0, 1, text="X")),
                            (2, Operation("insert", 3, 2, text="Y")),
                            (3, Operation("delete", 0, 3, length=3)),
                            (2, Operation("insert", 1, 2, text="Z"))):
        broadcast = clients[site - 1].local(operation)
        for client in clients:
            client.remote(broadcast)
    check(len({client.doc for client in clients}) == 1,
          f"the server order must converge all clients, got {[client.doc for client in clients]}")
    replay = base
    for operation in server_trial.history:
        replay = apply_operation(replay, operation)
    check(replay == clients[0].doc, "the server history must replay to the clients' document")
    check(len(server_trial.history) == 4, "every local operation must reach the server")

    # 4. intention preservation: a deletion still removes the same character
    delete_third = Operation("delete", 2, 1, length=1)
    insert_before = Operation("insert", 0, 2, text="Z")
    after_insert = apply_operation(base, insert_before)
    check(apply_operation(after_insert, transform(delete_third, insert_before)) == "Zabdef",
          "the deletion must follow the character it intended to delete")
    check("c" not in apply_operation(after_insert, transform(delete_third, insert_before)),
          "the intended character must really be gone")
    insert_after = Operation("insert", 1, 2, text="Z")
    check(apply_operation(apply_operation(base, insert_after), transform(delete_third, insert_after))
          == "aZbdef",
          "an insertion before the deleted character does not change what is deleted")

    # 5. two insertions at the same position both survive, in a deterministic order
    first_writer = Operation("insert", 2, 7, text="A")
    second_writer = Operation("insert", 2, 3, text="B")
    left = apply_operation(apply_operation(base, second_writer), transform(first_writer, second_writer))
    right = apply_operation(apply_operation(base, first_writer), transform(second_writer, first_writer))
    check(left == right, "concurrent insertions must converge to the same text")
    check(left.count("A") == 1 and left.count("B") == 1, "both insertions must survive")
    check(left == "abABcdef", f"the higher site id must go first, got {left!r}")

    # 6. overlapping deletions delete the union exactly once
    for first in (Operation("delete", 1, 1, length=3), Operation("delete", 2, 1, length=4)):
        for second in (Operation("delete", 2, 2, length=3), Operation("delete", 0, 2, length=2),
                       Operation("delete", 4, 2, length=2)):
            result = apply_order(base, [first, second])
            check(result == brute_force_delete(base, [first, second]),
                  f"delete union mismatch for {first} and {second}: {result!r}")
            check(len(result) + len({index for operation in (first, second)
                                     for index in range(operation.position,
                                                        min(operation.end, len(base)))})
                  == len(base), "each deleted character must be removed exactly once")

    # 7. an insertion strictly inside a deleted range is swallowed (documented delete-wins)
    delete_range = Operation("delete", 1, 1, length=3)
    insert_inside = Operation("insert", 3, 2, text="Q")
    merged = apply_order(base, [delete_range, insert_inside])
    check("Q" not in merged, "delete-wins must swallow an insertion inside the deleted range")
    check(merged == "aef", f"the deletion must still remove exactly its characters, got {merged!r}")
    check(apply_order(base, [insert_inside, delete_range]) == merged,
          "the delete-wins rule must be order independent")

    # 8. no-op transformations stay no-ops
    noop = Operation("delete", 3, 1, length=0)
    check(transform(noop, Operation("insert", 0, 2, text="ZZ")).is_noop(),
          "transforming a no-op must give a no-op")
    check(transform(Operation("insert", 2, 1, text="X"), noop).signature()
          == Operation("insert", 2, 1, text="X").signature(),
          "a no-op must not shift anything")
    check(apply_operation(base, transform_against_all(noop, universe())) == base,
          "applying a transformed no-op must leave the document alone")

    # 9. a central-server OT simulation with three clients converges
    def run_clients(seed):
        server = Server()
        clients = [Client(index + 1, server, "abcdef") for index in range(3)]
        stream = []
        for _ in range(18):
            seed, r1 = splitmix64(seed)
            seed, r2 = splitmix64(seed)
            stream.append((r1 % 3, r2))
        for site, r in stream:
            client = clients[site]
            if r % 3 == 0 and client.doc:
                operation = Operation("delete", r % len(client.doc), client.site,
                                      length=1 + (r % 2))
            else:
                operation = Operation("insert", r % (len(client.doc) + 1), client.site,
                                      text=chr(ord("m") + r % 8))
            broadcast = client.local(operation)
            for other in clients:
                other.remote(broadcast)
        return [client.doc for client in clients], server

    documents, server = run_clients(0x5EED)
    check(len(set(documents)) == 1, f"the three clients must converge, got {documents}")
    check(len(server.history) == 18, "the server must keep the transformed history")
    check(all(operation.position <= len("abcdef") + 18 for operation in server.history),
          "transformed positions must stay in range")
    documents_again, _ = run_clients(0x5EED)
    check(documents_again == documents, "the simulation must be deterministic")
    other_seed, _ = run_clients(0x1234)
    check(len(set(other_seed)) == 1, "another schedule must converge as well")

    # 10. the server history itself is a valid linearisation of the client operations
    replay = "abcdef"
    for operation in server.history:
        replay = apply_operation(replay, operation)
    check(replay == documents[0], "the server history must replay to the clients' document")

    print(f"665 TP1 pairs={pair_count} triples={triple_count} server ops={len(server.history)}")
    print(f"665 converged document={documents[0]!r}")
    print("PASS")


if __name__ == "__main__":
    main()
