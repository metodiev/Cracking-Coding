"""Task 664 - Implement a CRDT-based collaborative editor with undo support.  (Tier: Expert, Ch 9 - System Design and Scalability)

The editor is RGA: every character is an immutable item with a unique id (counter, site), the id of the
item it was inserted after, and a deleted flag. Integration only needs the anchor to exist, tombstones are
never removed (so a late concurrent insertion still places correctly), and materialising walks the tree
with siblings ordered by id descending - a total order derived from ids, hence independent of arrival
order, which is why merging is idempotent, commutative and associative. A site's consecutive insertions
anchor on each other, so another site's block cannot interleave into them.

Undo tombstones an insertion, and undoing a deletion re-inserts the character with a *fresh* id anchored
at the original left neighbour: resurrecting the tombstone id instead would apply once on some replicas
and twice on others once a merge replays it. Redo also allocates a fresh id. Undo stacks are per site and
LIFO, and an undo is an ordinary CRDT operation that propagates through the normal merge.

Verified: convergence under shuffled and repeated merges, idempotent/commutative/associative merge,
non-interleaving, tombstone deletes, undo/redo of inserts and deletes, undo propagation, causal buffering,
out-of-range rejection and determinism. Simplified: single-symbol characters, buffering only for a missing
anchor, whole-state merges instead of deltas.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Document:
    """One replica of the document: items keyed by id plus the causal buffer."""

    def __init__(self, site):
        self.site = site
        self.counter = 0
        self.items = {}                     # id -> {"char", "left", "deleted"}
        self.buffered = {}                  # id -> item waiting for its anchor

    # ------------------------------------------------------------------ integration
    def next_id(self):
        self.counter += 1
        return (self.counter, self.site)

    def integrate(self, item_id, char, left, deleted=False):
        """Idempotent: an id that is already known keeps its content."""
        if item_id in self.items:
            if deleted:
                self.items[item_id]["deleted"] = True
            return False
        if left is not None and left not in self.items:
            self.buffered[item_id] = (char, left, deleted)
            return False
        self.items[item_id] = {"char": char, "left": left, "deleted": deleted}
        self._drain_buffer()
        return True

    def _drain_buffer(self):
        progress = True
        while progress:
            progress = False
            for item_id in sorted(self.buffered):
                char, left, deleted = self.buffered[item_id]
                if left is None or left in self.items:
                    del self.buffered[item_id]
                    self.integrate(item_id, char, left, deleted)
                    progress = True
                    break

    # ------------------------------------------------------------------ ordering
    def children(self):
        """Successors of every anchor, ordered by id descending (independent of arrival order)."""
        table = {None: []}
        for item_id, item in self.items.items():
            table.setdefault(item["left"], []).append(item_id)
            table.setdefault(item_id, [])
        for siblings in table.values():
            siblings.sort(reverse=True)
        return table

    def order(self):
        """Depth-first walk over the id-ordered tree: the convergent document order."""
        table = self.children()
        result = []
        stack = list(reversed(table.get(None, [])))
        while stack:
            item_id = stack.pop()
            result.append(item_id)
            for child in reversed(table.get(item_id, [])):
                stack.append(child)
        check(len(result) == len(self.items), "every item must be reachable from the root")
        return result

    def text(self):
        return "".join(self.items[item_id]["char"] for item_id in self.order()
                       if not self.items[item_id]["deleted"])

    def visible(self):
        return [item_id for item_id in self.order() if not self.items[item_id]["deleted"]]

    def anchor_for(self, index):
        """The left neighbour for an insertion at `index` in the visible sequence."""
        check(0 <= index <= len(self.visible()), "insert index out of range")
        if index == 0:
            return None
        return self.visible()[index - 1]

    def tombstone_count(self):
        return sum(1 for item in self.items.values() if item["deleted"])

    # ------------------------------------------------------------------ merge
    def merge(self, other):
        """Union of items: idempotent, commutative and associative because ids are unique."""
        for item_id in sorted(set(self.items) | set(other.items) | set(self.buffered) | set(other.buffered)):
            if item_id in other.items:
                char, left = other.items[item_id]["char"], other.items[item_id]["left"]
                deleted = self.items.get(item_id, {}).get("deleted", False) or other.items[item_id]["deleted"]
                if item_id in self.items:
                    self.items[item_id]["deleted"] = deleted
                else:
                    self.integrate(item_id, char, left, deleted)
            elif item_id not in self.items and item_id in other.buffered:
                char, left, deleted = other.buffered[item_id]
                self.integrate(item_id, char, left, deleted)
        return self

    def clone(self, site=None):
        copy = Document(site if site is not None else self.site)
        copy.counter = self.counter
        copy.items = {item_id: dict(item) for item_id, item in self.items.items()}
        copy.buffered = dict(self.buffered)
        return copy


class Editor:
    """One user: a document, a local insert/delete API and per-site undo/redo stacks."""

    def __init__(self, site):
        self.site = site
        self.doc = Document(site)
        self.undo_stack = []
        self.redo_stack = []

    def insert(self, index, char, record=True):
        item_id = self.doc.next_id()
        self.doc.integrate(item_id, char, self.doc.anchor_for(index))
        if record:
            self.undo_stack.append({"kind": "insert", "id": item_id, "char": char,
                                    "left": self.doc.items[item_id]["left"]})
            self.redo_stack = []
        return item_id

    def delete(self, index, record=True):
        visible = self.doc.visible()
        check(0 <= index < len(visible), "delete index out of range")
        item_id = visible[index]
        self.doc.items[item_id]["deleted"] = True
        if record:
            self.undo_stack.append({"kind": "delete", "id": item_id,
                                    "char": self.doc.items[item_id]["char"],
                                    "left": self.doc.items[item_id]["left"]})
            self.redo_stack = []
        return item_id

    def text(self):
        return self.doc.text()

    def undo(self):
        """Invert the newest local operation; the inverse is itself an ordinary CRDT operation."""
        check(bool(self.undo_stack), "nothing to undo")
        operation = self.undo_stack.pop()
        if operation["kind"] == "insert":
            self.doc.items[operation["id"]]["deleted"] = True
            self.redo_stack.append(dict(operation))
        else:
            item_id = self.doc.next_id()
            self.doc.integrate(item_id, operation["char"], operation["left"])
            restored = dict(operation)
            restored["id"] = item_id
            self.redo_stack.append(restored)
        return operation

    def redo(self):
        check(bool(self.redo_stack), "nothing to redo")
        operation = self.redo_stack.pop()
        if operation["kind"] == "insert":
            item_id = self.doc.next_id()                      # a fresh id: the old one is tombstoned
            self.doc.integrate(item_id, operation["char"], operation["left"])
            replay = dict(operation)
            replay["id"] = item_id
            self.undo_stack.append(replay)
        else:
            self.doc.items[operation["id"]]["deleted"] = True
            self.undo_stack.append(dict(operation))
        return operation

    def merge(self, other):
        self.doc.merge(other.doc if isinstance(other, Editor) else other)
        return self


def run_sites(editors, operations, seed):
    """Apply local operations round-robin and merge everything in a shuffled order.

    Indices are resolved against the current text so a randomly generated stream stays valid.
    """
    for site, kind, payload in operations:
        editor = editors[site]
        if kind == "insert":
            index, char = payload
            editor.insert(index % (len(editor.text()) + 1), char)
        elif kind == "delete":
            if editor.text():
                editor.delete(payload % len(editor.text()))
        elif kind == "undo":
            if editor.undo_stack:
                editor.undo()
        elif editor.redo_stack:
            editor.redo()
        seed, r = splitmix64(seed)
        order = list(range(len(editors)))
        for step in range(r % len(editors)):
            order.append(order.pop(0))
        for target in order:
            for source in order:
                if target != source:
                    editors[target].merge(editors[source])
    return seed


def main():
    # 1. local editing
    editor = Editor("A")
    for char in "hello":
        editor.insert(len(editor.text()), char)
    check(editor.text() == "hello", "characters must appear in insertion order")
    editor.insert(0, "X")
    check(editor.text() == "Xhello", "insert at the front must work")
    editor.insert(3, "_")
    check(editor.text() == "Xhe_llo", "insert in the middle must work")
    editor.delete(3)
    check(editor.text() == "Xhello", "delete must remove exactly that character")
    check(editor.doc.tombstone_count() == 1, "a delete must leave a tombstone, not remove the item")
    check(len(editor.doc.items) == 7, "tombstones stay in the item map")
    try:
        editor.delete(99)
        raise AssertionError("deleting past the end must be rejected")
    except AssertionError as error:
        check("out of range" in str(error), "the bounds guard must fire")
    try:
        editor.insert(99, "z")
        raise AssertionError("inserting past the end must be rejected")
    except AssertionError as error:
        check("out of range" in str(error), "the insert guard must fire")

    # 2. convergence of concurrent edits at the same position
    alpha, beta, gamma = Editor("A"), Editor("B"), Editor("C")
    for char in "cat":
        alpha.insert(len(alpha.text()), char)
    beta.merge(alpha.doc.clone())                    # both peers start from alpha's document
    gamma.merge(alpha.doc.clone())
    alpha.insert(1, "A")                             # concurrent inserts around the shared text
    beta.insert(1, "B")
    gamma.insert(3, "C")
    for editor in (alpha, beta, gamma):
        for other in (alpha, beta, gamma):
            editor.merge(other)
    check(alpha.text() == beta.text() == gamma.text(),
          "all replicas must converge on the same text")
    check(sorted(alpha.text()) == sorted("catABC"), "convergence may not lose or invent characters")
    check(alpha.text().startswith("c"),
          "characters concurrency cannot move must keep their position")
    check(alpha.text().index("c") < alpha.text().index("t"),
          "the original relative order must be preserved")

    # 3. merge is idempotent, commutative and associative
    snapshot = alpha.text()
    alpha.merge(beta).merge(beta)
    check(alpha.text() == snapshot, "merging the same state twice must change nothing")
    left = alpha.doc.clone().merge(beta.doc.clone())
    right = beta.doc.clone().merge(alpha.doc.clone())
    check(left.text() == right.text(), "merge must be commutative")
    third = Editor("D")
    third.insert(0, "z")
    first = alpha.doc.clone().merge(beta.doc.clone()).merge(third.doc.clone())
    second = alpha.doc.clone().merge(beta.doc.clone().merge(third.doc.clone()))
    check(first.text() == second.text(), "merge must be associative")
    check(alpha.doc.clone().merge(Document("X")).text() == alpha.text(),
          "merging an empty replica must be the identity")

    # 4. non-interleaving: one site's block stays contiguous
    left_site, right_site = Editor("A"), Editor("B")
    left_site.insert(0, "x")                       # a shared anchor both sites agree on
    right_site.merge(left_site.doc.clone())
    for offset, char in enumerate("alice"):
        left_site.insert(1 + offset, char)
    for offset, char in enumerate("bob"):
        right_site.insert(1 + offset, char)
    left_site.merge(right_site)
    blocks = left_site.text()
    check("alice" in blocks and "bob" in blocks, "both blocks must survive the merge")
    check(blocks.count("alice") == 1 and blocks.count("bob") == 1, "blocks must not be duplicated")
    check(blocks.startswith("x") and len(blocks) == 9, "the shared anchor and both blocks remain")
    check(blocks[1:] in ("alicebob", "bobalice"),
          "each site's block must stay contiguous: no character-level interleaving")

    # 5. undo and redo of an insertion
    writer = Editor("A")
    writer.insert(0, "a")
    writer.insert(1, "b")
    writer.insert(2, "c")
    check(writer.text() == "abc", "the three characters are in place")
    writer.undo()
    check(writer.text() == "ab", "undoing an insert must remove that character")
    writer.undo()
    check(writer.text() == "a", "the undo stack must be LIFO")
    writer.redo()
    check(writer.text() == "ab", "redo must re-apply the undone insertion")
    writer.redo()
    check(writer.text() == "abc", "redo must restore the whole text")
    check(len(writer.undo_stack) == 3, "redo must put the operations back on the undo stack")

    # 6. undo of a deletion restores the character at its place
    deleter = Editor("A")
    for char in "hello":
        deleter.insert(len(deleter.text()), char)
    deleter.delete(1)
    check(deleter.text() == "hllo", "the character is deleted")
    deleter.undo()
    check(deleter.text() == "hello", "undoing a delete must restore the character")
    deleter.redo()
    check(deleter.text() == "hllo", "redoing the delete must remove it again")
    check(deleter.doc.tombstone_count() >= 1, "the delete still shows up as a tombstone")

    # 7. undo produces an ordinary CRDT operation that propagates
    alice, bob = Editor("A"), Editor("B")
    for char in "hi":
        alice.insert(len(alice.text()), char)
    bob.merge(alice)
    check(bob.text() == "hi", "the peer receives the text")
    alice.delete(0)
    check(alice.text() == "i", "alice deletes the first character")
    alice.undo()
    check(alice.text() == "hi", "alice restores it")
    bob.merge(alice)
    check(bob.text() == alice.text(), "the undo must propagate to the peer")
    check(bob.doc.tombstone_count() == alice.doc.tombstone_count(),
          "tombstones must propagate as well")

    # 8. causal buffering: an item that arrives before its anchor waits
    latecomer = Document("L")
    early = Document("E")
    anchor = early.next_id()
    child = early.next_id()
    early.integrate(anchor, "a", None)
    early.integrate(child, "b", anchor)
    latecomer.integrate(child, "b", anchor)
    check(latecomer.text() == "", "an item without its anchor must be buffered")
    check(len(latecomer.buffered) == 1, "the buffered item must be kept")
    latecomer.integrate(anchor, "a", None)
    check(latecomer.text() == "ab", "the buffer must drain once the anchor arrives")
    check(not latecomer.buffered, "the buffer must be empty afterwards")
    check(latecomer.text() == early.text(), "the late replica must converge")

    # 9. shuffled operation streams converge, and the run is deterministic
    operations = []
    seed = 0x5EED
    for step in range(24):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        site = r1 % 3
        kind = r2 % 4
        if kind == 0:
            operations.append((site, "delete", r1))
        elif kind == 1:
            operations.append((site, "undo", None))
        elif kind == 2:
            operations.append((site, "redo", None))
        else:
            operations.append((site, "insert", (r1, chr(ord("a") + r2 % 26))))

    def run(seed):
        editors = [Editor(name) for name in ("A", "B", "C")]
        for editor in editors:
            for char in "base":
                editor.insert(len(editor.text()), char)
        run_sites(editors, operations, seed)
        return editors

    first_run, second_run = run(0x1234), run(0x1234)
    check([editor.text() for editor in first_run] == [editor.text() for editor in second_run],
          "the same operation stream and seed must reproduce the same text")
    texts = {editor.text() for editor in first_run}
    check(len(texts) == 1, "every replica must converge after the shuffled merges")
    check(all(editor.doc.buffered == {} for editor in first_run),
          "no anchor may be left buffered once everything merged")
    orders = {tuple(editor.doc.order()) for editor in first_run}
    check(len(orders) == 1, "the item order itself must converge, not just the visible text")

    # 10. tombstone bookkeeping survives repeated merges
    survivor = first_run[0]
    before = survivor.text()
    for editor in first_run[1:]:
        survivor.merge(editor)
    check(survivor.text() == before, "re-merging must not change the text")
    check(not survivor.doc.buffered, "re-merging must leave no buffered items")

    print(f"664 converged text={first_run[0].text()!r} items={len(first_run[0].doc.items)} "
          f"tombstones={first_run[0].doc.tombstone_count()}")
    print(f"664 concurrent blocks={blocks!r}")
    print("PASS")


if __name__ == "__main__":
    main()
