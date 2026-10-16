"""Task 635 - Implement operational transform simplified for collaborative text editing.  (Tier: Expert, Ch 9 — System Design and Scalability)

Operational transformation keeps replicas convergent by *transforming* an operation against the
operations that were applied before it. The transformation is written as `transform(a, b)`: the
operation that site A wants to perform, adjusted for the fact that site B's operation `b` has already
been applied. For single-character edits the rules are short: an insert shifts everything at or after
its position one to the right (so a later insert at a position past it moves too), a delete shrinks the
text by one (so a later position past it moves left), two deletes of the same character leave the
second one with nothing to do (a no-op), and two inserts at the same position are ordered by site id so
that both orders agree.

The property that makes this work is TP1: for any pair of concurrent operations a and b created on the
same document, applying b and then a-transformed-against-b must reach the same document as applying a
and then b-transformed-against-a. The test checks that exhaustively - every pair of operations on every
document up to three characters over a two-letter alphabet - rather than on a handful of examples.

Simplifications, stated explicitly: operations are single-character inserts and deletes, there is no
operation *composition* (TP2) because the central server serialises every operation, and clients do not
apply their own edits optimistically - an edit is echoed back by the server already transformed against
whatever happened in between, so the client's text is always a prefix of the server's history. What is
verified: the transform pair rules (insert/insert including the site tie-break, insert/delete,
delete/insert, delete/delete including the no-op case), TP1 exhaustively, no-op stability, the server
scenario where three sites converge on the server's document, that no edit is lost or duplicated, and
determinism of the whole run.

Checks: apply semantics on boundaries (start, end, past-the-end delete, empty document), exhaustive TP1
over small documents including the same-position insert tie-break, no-op handling, a three-site
concurrent session converging on the server's text, a seeded multi-round session with client intent
preserved, and determinism.
"""

from collections import Counter
from itertools import product


def check(ok, message):
    if not ok:
        raise AssertionError(message)


INSERT, DELETE = "insert", "delete"
NOOP = ("noop", 0, None, None)


def insert(position, character, site):
    return (INSERT, position, character, site)


def delete(position, site):
    return (DELETE, position, None, site)


def is_noop(operation):
    return operation[0] == "noop"


def apply_op(text, operation):
    """Apply one operation; positions past the end clamp (an insert appends, a delete does nothing)."""
    kind, position, character, _site = operation
    if kind == INSERT:
        return text[:position] + character + text[position:]
    if kind == DELETE:
        return text[:position] + text[position + 1:]
    return text


def transform(operation, other):
    """Return `operation` adjusted for `other` having been applied first (TP1's other half)."""
    if is_noop(operation) or is_noop(other):
        return operation
    kind, position, character, site = operation
    other_kind, other_position, _other_character, other_site = other
    if kind == INSERT and other_kind == INSERT:
        if other_position < position:
            return insert(position + 1, character, site)
        if other_position == position and site > other_site:
            return insert(position + 1, character, site)     # the greater site id goes second
        return operation
    if kind == INSERT and other_kind == DELETE:
        return insert(position - 1, character, site) if other_position < position else operation
    if kind == DELETE and other_kind == INSERT:
        return delete(position + 1, site) if other_position <= position else operation
    if kind == DELETE and other_kind == DELETE:
        if other_position < position:
            return delete(position - 1, site)
        if other_position == position:
            return NOOP                                       # that character is already gone
        return operation
    return operation


def transform_against(operation, others):
    """Transform an operation against a sequence of already applied operations."""
    result = operation
    for other in others:
        result = transform(result, other)
    return result


def documents(alphabet="ab", max_length=3):
    for length in range(max_length + 1):
        for characters in product(alphabet, repeat=length):
            yield "".join(characters)


def operations_for(text, sites=("A", "B"), characters="xy"):
    """Every *valid* operation a site could create on this document."""
    found = []
    for site in sites:
        for position in range(len(text) + 1):
            found.append(insert(position, characters[0], site))
            if position < len(text):
                found.append(delete(position, site))
    return found


class Server:
    """Central OT server: serialises operations, transforming each one against what it missed."""

    def __init__(self, text):
        self.text = text
        self.applied = []                    # operations, in the order they were applied
        self.received = 0

    def receive(self, site, operation, known):
        """`known` is how many of `self.applied` the sending client had already seen."""
        concurrent = self.applied[known:]
        transformed = transform_against(operation, concurrent)
        self.received += 1
        if not is_noop(transformed):
            self.text = apply_op(self.text, transformed)
        self.applied.append(transformed)
        return transformed


class Client:
    """A site that sends edits to the server and applies the transformed echo (no local echo)."""

    def __init__(self, site, server):
        self.site = site
        self.server = server
        self.text = server.text
        self.known = len(server.applied)
        self.sent = 0

    def edit(self, kind, position, character=None):
        operation = insert(position, character, self.site) if kind == INSERT else delete(
            position, self.site)
        self.sent += 1
        echo = self.server.receive(self.site, operation, self.known)
        # Replay what happened in between, then the transformed echo: that keeps this client's text
        # equal to the matching prefix of the server's history.
        for missed in self.server.applied[self.known:-1]:
            self.text = apply_op(self.text, missed)
        self.text = apply_op(self.text, echo)
        self.known = len(self.server.applied)
        return operation, echo

    def sync(self):
        """Catch up on everything applied since this client last looked."""
        for operation in self.server.applied[self.known:]:
            self.text = apply_op(self.text, operation)
        self.known = len(self.server.applied)
        return self.text


class Session:
    """Deterministic driver: a fixed arrival order of client edits, then a final sync."""

    def __init__(self, text, sites=("A", "B", "C")):
        self.server = Server(text)
        self.clients = {site: Client(site, self.server) for site in sites}

    def deliver(self, site, kind, position, character=None):
        return self.clients[site].edit(kind, position, character)

    def sync_all(self):
        return {site: client.sync() for site, client in self.clients.items()}

    def converged(self):
        return len(set(self.sync_all().values()) | {self.server.text}) == 1


def seeded_runs(seed, rounds=6, sites=("A", "B", "C")):
    """A deterministic multi-round session: each round every site edits its own current text."""
    state = seed

    def pick(bound):
        nonlocal state
        state = (state * 75 + 74) % 65537
        return state % bound

    session = Session("hello", sites)
    for round_index in range(rounds):
        for site in sites:
            client = session.clients[site]
            text = client.text
            if pick(3) == 0 and text:
                position = pick(len(text))
                session.deliver(site, DELETE, position)
            else:
                position = pick(len(text) + 1)
                session.deliver(site, INSERT, position, chr(ord("a") + (round_index + pick(3))))
    session.sync_all()
    return session


def main():
    # --- 1. apply semantics and boundaries -------------------------------------------------------
    check(apply_op("", insert(0, "x", "A")) == "x", "an insert into an empty document is the char")
    check(apply_op("abc", insert(0, "x", "A")) == "xabc", "an insert at the front")
    check(apply_op("abc", insert(3, "x", "A")) == "abcx", "an insert at the end")
    check(apply_op("abc", insert(99, "x", "A")) == "abcx", "an insert past the end clamps to append")
    check(apply_op("abc", delete(1, "A")) == "ac", "a delete removes exactly one character")
    check(apply_op("abc", delete(5, "A")) == "abc", "a delete past the end does nothing")
    check(apply_op("abc", NOOP) == "abc", "a no-op leaves the document alone")
    check(is_noop(transform(delete(1, "A"), delete(1, "B"))),
          "two deletes of the same character leave the second one nothing to do")

    # --- 2. TP1 exhaustively over every small document and every valid operation pair -----------
    pairs = 0
    tie_break_cases = 0
    for text in documents():
        operations = operations_for(text)
        for first in operations:
            for second in operations:
                left = apply_op(apply_op(text, second), transform(first, second))
                right = apply_op(apply_op(text, first), transform(second, first))
                check(left == right,
                      f"TP1 violated on {text!r}: {first} / {second} gave {left!r} and {right!r}")
                pairs += 1
                if first[0] == INSERT and second[0] == INSERT and first[1] == second[1] \
                        and first[3] != second[3]:
                    tie_break_cases += 1
                    check(len(left) == len(text) + 2, "both concurrent inserts survive the tie-break")
    check(pairs > 500, f"the exhaustive sweep really ran ({pairs} operation pairs)")
    check(tie_break_cases > 0, f"same-position inserts were exercised ({tie_break_cases} pairs)")

    # --- 3. hand-checked transforms --------------------------------------------------------------
    check(transform(insert(2, "x", "A"), insert(0, "y", "B")) == insert(3, "x", "A"),
          "a later insert is shifted right by a concurrent insert before it")
    check(transform(insert(0, "x", "A"), insert(2, "y", "B")) == insert(0, "x", "A"),
          "an earlier insert is unaffected by one after it")
    check(transform(insert(1, "x", "A"), delete(0, "B")) == insert(0, "x", "A"),
          "an insert after a deleted character moves left")
    check(transform(delete(2, "A"), insert(1, "y", "B")) == delete(3, "A"),
          "a delete after a concurrent insert moves right")
    check(transform(delete(1, "A"), insert(5, "y", "B")) == delete(1, "A"),
          "and stays put when the insert is elsewhere")
    check(transform(delete(2, "A"), delete(1, "B")) == delete(1, "A"),
          "a delete after another delete moves left")
    check(transform(transform(delete(1, "A"), delete(1, "B")), insert(0, "z", "C"))
          == transform(delete(1, "A"), delete(1, "B")),
          "a no-op stays a no-op however it is transformed further")
    check(transform_against(insert(2, "x", "A"), [insert(0, "y", "B"), delete(3, "C")])
          == insert(3, "x", "A"),
          "a chain of concurrent operations shifts the insert by the inserts before it only")

    # --- 4. a three-site session converges on the server's document ------------------------------
    session = Session("abcd")
    session.clients["A"].sync()
    first_op, first_echo = session.deliver("A", DELETE, 0, None)      # A: "abcd" -> "bcd"
    second_op, second_echo = session.deliver("B", INSERT, 1, "X")     # B saw "abcd": insert after 'a'
    third_op, third_echo = session.deliver("C", INSERT, 4, "Y")       # C saw "abcd": append
    check(session.server.text == "XbcdY",
          f"the server serialises the three edits: {session.server.text!r}")
    check(session.converged(), "every site ends on the server's document")
    check(len({session.clients[site].text for site in session.clients}) == 1,
          "all client texts are identical")
    check(session.clients["A"].text == session.server.text,
          "including the site whose delete was transformed against the inserts")
    check(first_echo == first_op and second_echo == insert(0, "X", "B"),
          "A's delete applied as-is, while B's insert moved left because of the earlier delete")
    check(third_echo == insert(4, "Y", "C"),
          "C's append was shifted by the delete and by B's insert")
    check(len(session.server.applied) == 3 and session.server.received == 3,
          "one applied operation per client edit - nothing was dropped")

    # --- 5. a delete that becomes a no-op --------------------------------------------------------
    session = Session("ab")
    session.deliver("A", DELETE, 0)                # server: "b"
    _, echo = session.deliver("B", DELETE, 0)      # B also deletes the first character it saw
    check(is_noop(echo), "the second delete of the same character becomes a no-op")
    check(session.server.text == "b", "and the document keeps the other character")
    check(session.converged() and len(session.server.applied) == 2,
          "the no-op is still recorded in the history, and everybody agrees")

    # --- 6. seeded multi-round session: convergence and no lost edits ---------------------------
    session = seeded_runs(seed=20250, rounds=6)
    check(session.converged(), "a six-round session with three sites converges")
    inserts = [operation for operation in session.server.applied if operation[0] == INSERT]
    deletes = [operation for operation in session.server.applied if operation[0] == DELETE]
    check(len(session.server.applied) == sum(client.sent for client in session.clients.values()),
          "the history holds one entry per client edit")
    check(len(session.server.text) == 5 + len(inserts) - len(deletes),
          f"the length follows the applied inserts and deletes: {session.server.text!r}")
    as_multiset = Counter(session.server.text)
    budget = Counter("hello") + Counter(operation[2] for operation in inserts)
    check(all(count <= budget[character] for character, count in as_multiset.items()),
          "every character in the final text was either in the original or inserted once")
    check(all(client.text == session.server.text for client in session.clients.values()),
          "every client finished on the server's text")

    twin = seeded_runs(seed=20250, rounds=6)
    check(twin.server.text == session.server.text
          and twin.server.applied == session.server.applied,
          "the same seed reproduces the same session exactly")

    # --- 7. boundaries ---------------------------------------------------------------------------
    empty = Session("")
    empty.deliver("A", DELETE, 0)                  # deleting past the end of an empty document
    empty.deliver("B", INSERT, 0, "z")
    check(empty.server.text == "z" and empty.converged(),
          "an empty document accepts an insert and shrugs off a delete")
    single = Session("x")
    single.deliver("A", DELETE, 0)
    single.deliver("B", INSERT, 0, "y")
    check(single.server.text == "y" and single.converged(),
          "a delete and a concurrent insert on a one-character document converge")
    check(all(client.text == single.server.text for client in single.clients.values()),
          "every replica holds the server's text after the boundary case")

    print(f"ot: tp1-pairs={pairs} tie-break-pairs={tie_break_cases} "
          f"seeded-session={session.server.text!r} applied={len(session.server.applied)}")


if __name__ == "__main__":
    main()
    print("PASS")
