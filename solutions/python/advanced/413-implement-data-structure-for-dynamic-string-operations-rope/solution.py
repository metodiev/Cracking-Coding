"""Task 413 - Implement a data structure for dynamic string operations (rope). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A rope stores a string as a sequence of chunks (leaves of a binary tree). The real
structure is a balanced tree of chunks with O(log n) split/concat; this implementation
keeps the chunk sequence FLAT and balanced instead, a documented simulation with the
same interface:
  * chunks hold at most CHUNK = 64 characters and are rebalanced so their count stays
    within ~2x of the ideal ceil(n / 64),
  * index/char_at/slice locate by walking the chunk lengths (O(#chunks + CHUNK)),
  * insert/delete/split/concat/reverse splice chunks and rebalance (O(n + CHUNK));
    a tree-based rope would perform the same operations by re-linking subtrees.
All operations are functional: they return NEW ropes, leaving the receiver untouched.
"""

CHUNK = 64


def _normalize(chunks):
    """Drop empty chunks and re-chunk so the count is close to ceil(n / CHUNK)."""
    whole = "".join(c for c in chunks if c)
    n = len(whole)
    if n == 0:
        return []
    target = max(1, (n + CHUNK - 1) // CHUNK)
    size = (n + target - 1) // target
    return [whole[i:i + size] for i in range(0, n, size)]


def _maybe_rebalance(chunks):
    kept = [c for c in chunks if c]
    n = sum(len(c) for c in kept)
    target = max(1, (n + CHUNK - 1) // CHUNK) if n else 1
    if len(kept) > 2 * target + 2:
        return _normalize(kept)
    return kept


class Rope:
    def __init__(self, text=""):
        self.chunks = _normalize([text] if text else [])

    @staticmethod
    def _of(chunks):
        r = Rope.__new__(Rope)
        r.chunks = _maybe_rebalance(chunks)
        return r

    @property
    def length(self):
        return sum(len(c) for c in self.chunks)

    def to_string(self):
        return "".join(self.chunks)

    def _locate(self, pos):
        """Return (chunk_index, offset) for character pos, or (len(chunks), 0) at the end."""
        for idx, c in enumerate(self.chunks):
            if pos <= len(c):
                return idx, pos
            pos -= len(c)
        return len(self.chunks), 0

    def char_at(self, pos):
        if pos < 0 or pos >= self.length:
            raise IndexError("rope index out of range")
        for c in self.chunks:
            if pos < len(c):
                return c[pos]
            pos -= len(c)
        raise IndexError("rope index out of range")

    def slice(self, start, end):
        if start < 0 or end > self.length or start > end:
            raise IndexError("rope slice out of range")
        out = []
        need = end - start
        idx, off = self._locate(start)
        while need > 0:
            take = self.chunks[idx][off:off + need]
            out.append(take)
            need -= len(take)
            idx += 1
            off = 0
        return "".join(out)

    def insert(self, pos, text):
        if not 0 <= pos <= self.length:
            raise IndexError("rope insert out of range")
        idx, off = self._locate(pos)
        chunks = list(self.chunks[:idx])
        if off > 0:
            chunks.append(self.chunks[idx][:off])
        chunks.extend(_normalize([text]) if text else [])
        if idx < len(self.chunks) and off < len(self.chunks[idx]):
            chunks.append(self.chunks[idx][off:])
        chunks.extend(self.chunks[idx + 1:])
        return Rope._of(chunks)

    def delete(self, start, end):
        if start < 0 or end > self.length or start > end:
            raise IndexError("rope delete out of range")
        if start == end:
            return Rope._of(list(self.chunks))
        li, lo = self._locate(start)
        ri, ro = self._locate(end)
        chunks = list(self.chunks[:li])
        if lo > 0:
            chunks.append(self.chunks[li][:lo])
        if ro < len(self.chunks[ri]):
            chunks.append(self.chunks[ri][ro:])
        chunks.extend(self.chunks[ri + 1:])
        return Rope._of(chunks)

    def split(self, pos):
        if not 0 <= pos <= self.length:
            raise IndexError("rope split out of range")
        idx, off = self._locate(pos)
        left = list(self.chunks[:idx])
        if off > 0:
            left.append(self.chunks[idx][:off])
        right = list(self.chunks[idx + 1:])
        if idx < len(self.chunks) and off < len(self.chunks[idx]):
            right.insert(0, self.chunks[idx][off:])
        return Rope._of(left), Rope._of(right)

    def concat(self, other):
        return Rope._of(list(self.chunks) + list(other.chunks))

    def reverse(self):
        return Rope._of([c[::-1] for c in reversed(self.chunks)])


class Lcg:
    def __init__(self, state):
        self.state = state

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def main():
    r = Rope("hello")
    assert r.length == 5
    assert r.char_at(0) == "h" and r.char_at(4) == "o" and r.char_at(2) == "l"
    assert r.to_string() == "hello"
    assert r.slice(1, 4) == "ell"
    assert Rope("abef").insert(2, "cd").to_string() == "abcdef"
    assert Rope("hello").insert(0, "xyz").to_string() == "xyzhello"
    # "xyzhello" (8 chars) minus [3, 6) 'h','e','l' leaves "xyzlo"
    assert Rope("hello").insert(0, "xyz").delete(3, 6).to_string() == "xyzlo"
    left, right = Rope("xyzlo").split(2)
    assert left.to_string() == "xy" and right.to_string() == "zlo"
    assert left.concat(right).to_string() == "xyzlo"
    assert Rope("xyzlo").reverse().to_string() == "olzyx"
    assert Rope("").length == 0 and Rope("").to_string() == ""
    assert Rope("").insert(0, "abc").to_string() == "abc"
    assert Rope("abc").delete(1, 2).to_string() == "ac"
    assert Rope("abc").delete(0, 0).to_string() == "abc"
    assert Rope("abc").split(0)[1].to_string() == "abc"
    assert Rope("abc").split(3)[0].to_string() == "abc"
    # functional semantics: the receiver never changes
    original = Rope("keepme")
    original.insert(0, "X")
    original.delete(0, 3)
    original.reverse()
    assert original.to_string() == "keepme"

    # cross-chunk operations on a 200-character rope
    big = "".join(chr(ord("a") + i % 26) for i in range(200))
    rb = Rope(big)
    assert rb.length == 200 and rb.to_string() == big
    assert rb.char_at(0) == big[0] and rb.char_at(199) == big[199]
    assert rb.char_at(63) == big[63] and rb.char_at(64) == big[64]
    assert rb.slice(0, 70) == big[:70] and rb.slice(60, 150) == big[60:150]
    oracle = big[:130] + "XYZ" + big[130:]
    ins = rb.insert(130, "XYZ")
    assert ins.to_string() == oracle and ins.length == 203
    dele = ins.delete(125, 135)
    assert dele.to_string() == oracle[:125] + oracle[135:]
    left, right = Rope(big).split(64)
    assert left.length == 64 and right.length == 136
    assert left.to_string() == big[:64] and right.to_string() == big[64:]
    assert left.concat(right).to_string() == big
    assert Rope(big).reverse().to_string() == big[::-1]

    # chunk length invariant after structural operations
    probe = Rope(big).insert(0, "z" * 40).delete(10, 90).reverse().concat(Rope("q" * 300))
    assert all(len(c) <= CHUNK for c in probe.chunks), probe.chunks
    assert probe.length == 200 + 40 - 80 + 300

    # LCG-driven fuzz against a plain-string oracle; every operation must agree,
    # and no op script can desynchronise the rope from the oracle string.
    gen = Lcg(31337)
    alphabet = "abcdef"
    rope, oracle_str = Rope(""), ""
    for step in range(320):
        kind = gen.next() % 5
        if kind == 0 or kind == 1:
            pos = gen.next() % (len(oracle_str) + 1)
            ln = 1 + gen.next() % 12
            text = "".join(alphabet[gen.next() % len(alphabet)] for _ in range(ln))
            rope = rope.insert(pos, text)
            oracle_str = oracle_str[:pos] + text + oracle_str[pos:]
        elif kind == 2:
            if oracle_str:
                a = gen.next() % (len(oracle_str) + 1)
                b = gen.next() % (len(oracle_str) + 1)
                if a > b:
                    a, b = b, a
                rope = rope.delete(a, b)
                oracle_str = oracle_str[:a] + oracle_str[b:]
        elif kind == 3:
            rope = rope.reverse()
            oracle_str = oracle_str[::-1]
        else:
            a = gen.next() % (len(oracle_str) + 1)
            b = gen.next() % (len(oracle_str) + 1)
            if a > b:
                a, b = b, a
            assert rope.slice(a, b) == oracle_str[a:b], (step, a, b)
            if oracle_str:
                pos = gen.next() % len(oracle_str)
                assert rope.char_at(pos) == oracle_str[pos], (step, pos)
            mid = a
            left, right = rope.split(mid)
            assert left.to_string() == oracle_str[:mid], step
            assert right.to_string() == oracle_str[mid:], step
            rope = left.concat(right)
        assert rope.to_string() == oracle_str, step
        assert rope.length == len(oracle_str), step

    # final sweep: index and slice for every position of the fuzz result
    for i in range(len(oracle_str)):
        assert rope.char_at(i) == oracle_str[i], i
    assert rope.slice(0, len(oracle_str)) == oracle_str


if __name__ == "__main__":
    main()
    print("PASS")
