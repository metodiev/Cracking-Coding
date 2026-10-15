
"""Task 080 - Implement string builder/concatenation efficient. (Tier: Beginner, CTCI Ch 1)

O(n) via a join buffer instead of repeated O(n) immutable concatenations.
"""


class StringBuilder:
    def __init__(self):
        self._parts = []

    def append(self, text):
        self._parts.append(str(text))
        return self

    def build(self):
        return "".join(self._parts)

    def __len__(self):
        return sum(len(p) for p in self._parts)


def main():
    sb = StringBuilder()
    sb.append("Hello").append(", ").append("world").append("!").append(42)
    assert sb.build() == "Hello, world!42"
    assert len(sb) == 15

    # O(n) bulk build from many pieces
    parts = str(10000)
    big = "".join(str(i) for i in range(10000))
    assert len(big) == len(str("".join(map(str, range(10000)))))
    assert StringBuilder().build() == ""


if __name__ == "__main__":
    main()
    print("PASS")
