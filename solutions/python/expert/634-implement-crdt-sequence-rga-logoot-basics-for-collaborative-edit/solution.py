"""Task 634 - Implement CRDT sequence (RGA / Logoot) basics for collaborative editing.  (Tier: Expert, Ch 9 — System Design and Scalability)

Two ways to make a text sequence a CRDT. **RGA** gives every inserted character a unique id (a
per-site counter plus the site name) and an *anchor* - the id of the character it was typed after -
so the document is a tree rooted at a virtual head. Children of the same anchor are ordered by
descending id, which puts a newly typed character immediately before its older siblings and keeps a
run of characters typed by one site contiguous: the anchor of each character is the previous character
of the same site, so the run forms one chain. **Logoot** instead gives every character a dense
position (a list of (digit, site) pairs) generated strictly between the positions of its neighbours;
the document is the positions in lexicographic order, so no anchors are needed and insertion never
needs to know about unseen characters.

Both structures are order-independent: an operation whose anchor or predecessor has not arrived yet is
buffered and replayed when it does, a duplicate delivery is idempotent (RGA keys the tree by id,
Logoot by position), and deletions are tombstones. What is verified: the final text is identical under
every delivery permutation of a concurrent edit script (the convergence property, checked exhaustively
on small scripts), buffered operations are drained exactly once, tombstones survive, RGA's sibling
order keeps concurrent runs contiguous (the no-interleaving property for chains), and every Logoot
position is strictly between its neighbours with positions growing only logarithmically when text is
appended at the end.

Simplifications, stated explicitly: the site counter is a deterministic integer (no wall clock), the
document is kept in memory and tombstones are never garbage-collected, and Logoot uses a fixed base of
256 digits with a single site component per digit.
"""

from itertools import permutations


def check(ok, message):
    if not ok:
        raise AssertionError(message)


HEAD = None
BASE = 256
START = ((0, "\x00"),)                       # a virtual position below every real Logoot position


# ---------------------------------------------------------------------------------------------
# RGA
# ---------------------------------------------------------------------------------------------
class RGA:
    """Replicated growable array: characters hang off the id of the character they follow."""

    def __init__(self, site):
        self.site = site
        self.counter = 0
        self.anchors = {HEAD: []}            # anchor id -> child ids, descending
        self.values = {}                     # id -> character
        self.deleted = set()
        self.pending = []
        self.applied = set()

    def _new_id(self):
        self.counter += 1
        return (self.counter, self.site)

    def insert(self, anchor, character):
        identifier = self._new_id()
        operation = ("insert", identifier, anchor, character)
        check(self.apply(operation), "a locally generated insert always applies")
        return operation

    def delete(self, identifier):
        operation = ("delete", identifier)
        self.apply(operation)
        return operation

    def apply(self, operation):
        if operation in self.applied:
            return True                      # duplicate delivery: idempotent
        kind = operation[0]
        if kind == "insert":
            _, identifier, anchor, character = operation
            if anchor is not HEAD and anchor not in self.values:
                self.pending.append(operation)
                return False                 # the anchor has not arrived yet
            self.values[identifier] = character
            self.anchors.setdefault(identifier, [])
            siblings = self.anchors.setdefault(anchor, [])
            siblings.append(identifier)
            siblings.sort(reverse=True)      # descending id: the newest char goes first
        else:
            _, identifier = operation
            if identifier not in self.values:
                self.pending.append(operation)
                return False
            self.deleted.add(identifier)
        self.applied.add(operation)
        return True

    def drain(self):
        """Replay the buffer until no buffered operation can be applied (returns #applied)."""
        applied = 0
        progress = True
        while progress:
            progress = False
            remaining = []
            for operation in list(self.pending):     # snapshot: apply() may re-buffer an operation
                if self.apply(operation):
                    applied += 1
                    progress = True
                else:
                    remaining.append(operation)
            self.pending = remaining
        return applied

    def text(self):
        out = []
        stack = [(HEAD, 0)]
        while stack:
            anchor, index = stack.pop()
            children = self.anchors.get(anchor, [])
            if index >= len(children):
                continue
            child = children[index]
            stack.append((anchor, index + 1))
            if child not in self.deleted:
                out.append(self.values[child])
            stack.append((child, 0))
        return "".join(out)

    def ids(self):
        return [identifier for identifier in self.values]


# ---------------------------------------------------------------------------------------------
# Logoot
# ---------------------------------------------------------------------------------------------
def compare_positions(first, second):
    for (digit_a, site_a), (digit_b, site_b) in zip(first, second):
        if digit_a != digit_b:
            return -1 if digit_a < digit_b else 1
        if site_a != site_b:
            return -1 if site_a < site_b else 1
    if len(first) == len(second):
        return 0
    return -1 if len(first) < len(second) else 1


class Logoot:
    """Dense-position sequence: the document is the positions in lexicographic order."""

    def __init__(self, site):
        self.site = site
        self.positions = {}                  # position -> character
        self.deleted = set()
        self.applied = set()

    def generate(self, left, right):
        """A position strictly between `left` and `right` (None = open end)."""
        if right is None:                    # appending at the end: just step the last digit
            digits = list(left)
            if digits[-1][0] + 1 < BASE:
                digits[-1] = (digits[-1][0] + 1, self.site)
                return tuple(digits)
            digits.append((BASE // 2, self.site))
            return tuple(digits)
        digits = []
        index = 0
        while True:
            low = left[index][0] if index < len(left) else 0
            high = right[index][0] if index < len(right) else BASE
            if high - low > 1:
                digits.append(((low + high) // 2, self.site))
                return tuple(digits)
            digits.append(left[index])
            index += 1

    def insert_between(self, left, right, character):
        position = self.generate(START if left is None else left, right)
        operation = ("insert", position, character)
        self.apply(operation)
        return operation

    def delete(self, position):
        operation = ("delete", position)
        self.apply(operation)
        return operation

    def apply(self, operation):
        if operation in self.applied:
            return True
        if operation[0] == "insert":
            _, position, character = operation
            self.positions.setdefault(position, character)   # keyed by position: idempotent
        else:
            self.deleted.add(operation[1])
        self.applied.add(operation)
        return True

    def order(self):
        return sorted(self.positions, key=lambda position: position)

    def text(self):
        return "".join(self.positions[position] for position in self.order()
                       if position not in self.deleted)

    def positions_between(self, left, right):
        return [position for position in self.order() if compare_positions(left, position) < 0
                and (right is None or compare_positions(position, right) < 0)]


def lcg(seed, modulus=65537):
    state = seed

    def nxt(bound):
        nonlocal state
        state = (state * 75 + 74) % modulus
        return state % bound

    return nxt


def type_word(replica, word, anchor=HEAD):
    """Type a word character by character, each character anchored on the previous one."""
    operations = []
    current = anchor
    for character in word:
        current_op = replica.insert(current, character)
        operations.append(current_op)
        current = current_op[1]
    return operations


def deliver(replica, operations):
    for operation in operations:
        replica.apply(operation)
    replica.drain()
    return replica


def main():
    # --- 1. RGA: sequential typing and mid-document insertion ----------------------------------
    alice = RGA("A")
    check(type_word(alice, "lo") and alice.text() == "lo",
          "typing two characters in order yields them in order")
    check(alice.pending == [] and len(alice.ids()) == 2, "both characters are in the tree")

    anchor = alice.anchors[HEAD][-1]         # the first character, "l"
    alice.insert(anchor, "X")                # inserted right after "l"
    check(alice.text() == "lXo", f"an insert after an existing character lands in place: {alice.text()}")
    check(alice.text().index("X") == 1, "and immediately follows its anchor")

    # --- 2. RGA: out-of-order delivery is buffered ---------------------------------------------
    writer = RGA("W")
    prefix = type_word(writer, "l")
    more = type_word(writer, "ef", prefix[-1][1])   # "e" after "l", then "f" after "e"
    late = RGA("C")
    check(late.apply(more[-1]) is False and len(late.pending) == 1,
          "a character whose anchor is missing is buffered, not applied")
    check(late.apply(prefix[0]) is True, "the first anchor arrives and is applied")
    check(late.drain() == 0, "the buffered character still waits for its own anchor")
    check(late.apply(more[0]) is True and late.drain() == 1 and late.pending == [],
          "the missing anchor then unblocks and drains the buffer exactly once")
    check(late.text() == "lef", f"the buffered character appears in the right place: {late.text()}")
    check(deliver(RGA("D"), prefix + more).text() == "lef", "in-order delivery gives the same text")
    check(deliver(RGA("E"), list(reversed(prefix + more))).text() == "lef",
          "reverse delivery gives the same text too")

    # --- 3. RGA: concurrent runs stay contiguous, and every order converges --------------------
    base = RGA("BASE")
    prefix = type_word(base, "a")
    left_site = deliver(RGA("L"), prefix)
    right_site = deliver(RGA("R"), prefix)
    anchor = prefix[-1][1]                   # both sites type after the same character
    left_ops = type_word(left_site, "xy", anchor)
    right_ops = type_word(right_site, "pq", anchor)
    script = prefix + left_ops + right_ops
    combined = deliver(RGA("M"), script)
    check(combined.text() in (left_site.text() + "pq", right_site.text() + "xy"),
          f"the two runs are contiguous - no interleaving: {combined.text()}")
    check("xp" not in combined.text() and "px" not in combined.text(),
          f"characters of the two runs never interleave: {combined.text()}")

    texts = set()
    orders = list(permutations(script, len(script)))
    for order in orders:
        replica = RGA("P")
        for operation in order:
            replica.apply(operation)
            replica.drain()
        texts.add(replica.text())
    check(len(texts) == 1, f"all delivery orders converge on one text: {texts}")
    check(texts.pop() == combined.text(), "and it is the text the local site computed")
    check(len(orders) == 120, "five operations produce 120 distinct delivery orders")

    # --- 4. RGA: duplicates, deletes and tombstones --------------------------------------------
    replica = RGA("Z")
    deliver(replica, script)
    before = replica.text()
    deliver(replica, script)                 # every operation delivered twice
    check(replica.text() == before, "re-delivering the whole script changes nothing")
    first_id = next(identifier for identifier in replica.ids()
                    if replica.values[identifier] == "a")
    replica.delete(first_id)
    check(replica.text() == before.replace("a", "", 1),
          f"a delete removes exactly one character: {replica.text()}")
    replica.delete(first_id)
    check(replica.text() == before.replace("a", "", 1), "a repeated delete is idempotent")

    fresh = RGA("Y")
    insert_operation = fresh.insert(HEAD, "!")
    other = RGA("X")
    check(other.apply(("delete", insert_operation[1])) is False and other.pending,
          "a delete for an unknown id is buffered")
    check(other.apply(insert_operation) is True, "the character itself applies normally")
    check(other.drain() == 1 and other.text() == "",
          "the buffered delete then removes it, leaving the document empty")

    # --- 5. Logoot: ordering, betweenness and density -------------------------------------------
    doc = Logoot("A")
    positions = []
    previous = None
    for character in "hello":
        operation = doc.insert_between(previous, None, character)
        positions.append(operation[1])
        previous = operation[1]
    check(doc.text() == "hello", "appending at the end keeps the text in order")
    check(all(compare_positions(positions[index], positions[index + 1]) < 0
              for index in range(len(positions) - 1)), "positions are strictly increasing")
    check(max(len(position) for position in positions) <= 2,
          "appending at the end grows the position by at most one digit here")

    doc = Logoot("B")
    middle = doc.insert_between(None, None, "m")[1]
    before = doc.insert_between(None, middle, "b")[1]
    after = doc.insert_between(middle, None, "z")[1]
    check([before, middle, after] == doc.order(), "an insert between two positions lands between them")
    check(doc.text() == "bmz", f"the document follows the position order: {doc.text()}")
    inner = doc.insert_between(before, middle, "c")[1]
    check(compare_positions(before, inner) < 0 and compare_positions(inner, middle) < 0,
          "the generated position is strictly between its neighbours")
    check(doc.text() == "bcmz", f"and the new character appears in the middle: {doc.text()}")

    pick = lcg(4321)
    for _ in range(60):                      # stress the generator: always strictly between
        order = doc.order()
        left_index = pick(len(order) + 1) - 1
        left = order[left_index] if left_index >= 0 else None
        right = order[left_index + 1] if left_index + 1 < len(order) else None
        new_position = doc.generate(START if left is None else left, right)
        if left is not None:
            check(compare_positions(left, new_position) < 0, "generated position is above its left")
        if right is not None:
            check(compare_positions(new_position, right) < 0, "generated position is below its right")
        doc.positions[new_position] = "?"
    check(len(doc.order()) == 4 + 60, "every generated position was unique")
    check(doc.order() == sorted(doc.order(), key=lambda position: position),
          "the document order is the sorted position order")

    # --- 6. Logoot: convergence under any delivery order ----------------------------------------
    source = Logoot("S")
    first = source.insert_between(None, None, "a")[1]
    second = source.insert_between(first, None, "c")[1]
    script = [("insert", first, "a"), ("insert", second, "c"),
              ("insert", source.generate(first, second), "b"), ("delete", second)]
    texts = set()
    for order in permutations(script, len(script)):
        replica = Logoot("T")
        for operation in order:
            replica.apply(operation)
        texts.add(replica.text())
    check(texts == {"ab"}, f"all delivery orders converge on one text: {texts}")
    check(len(list(permutations(script, len(script)))) == 24,
          "four operations produce 24 distinct delivery orders")

    repeated = Logoot("U")
    for operation in script + script:
        repeated.apply(operation)
    check(repeated.text() == "ab", "duplicate deliveries do not duplicate characters")
    check(len(repeated.positions) == 3, "the document holds exactly three positions")

    # --- 7. boundaries --------------------------------------------------------------------------
    empty = RGA("E")
    check(empty.text() == "" and empty.drain() == 0, "an empty RGA is empty and has no buffer")
    check(Logoot("F").text() == "" and Logoot("F").order() == [],
          "an empty Logoot document has no positions")
    solo = RGA("G")
    only = solo.insert(HEAD, "s")
    check(solo.text() == "s", "a one-character document is just that character")
    solo.delete(only[1])
    check(solo.text() == "", "deleting the only character empties the document")
    check(compare_positions(START, START) == 0 and compare_positions(START, ((1, "A"),)) < 0,
          "the start sentinel compares below every real position")

    print(f"rga: converged={combined.text()!r} delivery-orders={len(orders)} "
          f"logoot: letters={len(doc.positions)} max-position-length="
          f"{max(len(position) for position in doc.positions)}")


if __name__ == "__main__":
    main()
    print("PASS")
