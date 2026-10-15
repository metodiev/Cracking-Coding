"""Task 289 - Restore IP addresses from a digit string. (Tier: Intermediate, CTCI Ch 8)

Parts are extended left to right, trying segment lengths 1, 2, 3 in that order, so the
emitted addresses are deterministic. A segment is valid when it is 1-3 digits, at most
255, and has no leading zero unless it is exactly "0". Cross-checked against an oracle
that enumerates every cut position triple.
"""

def is_valid_segment(segment):
    if not segment or len(segment) > 3:
        return False
    if segment[0] == "0":
        return segment == "0"
    return int(segment) <= 255


def restore_ip_addresses(s):
    """All valid IP addresses formed by inserting three dots, in DFS order."""
    if len(s) < 4 or len(s) > 12:
        return []
    results = []
    parts = []

    def backtrack(index):
        if len(parts) == 4:
            if index == len(s):
                results.append(".".join(parts))
            return
        for length in (1, 2, 3):
            if index + length > len(s):
                break
            segment = s[index:index + length]
            if not is_valid_segment(segment):
                continue
            parts.append(segment)
            backtrack(index + length)
            parts.pop()

    backtrack(0)
    return results


def brute_force_ip_addresses(s):
    """Oracle: try every triple of cut positions and validate the four segments."""
    results = []
    n = len(s)
    for i in range(1, min(4, n)):
        for j in range(i + 1, min(i + 4, n)):
            for k in range(j + 1, min(j + 4, n)):
                parts = [s[:i], s[i:j], s[j:k], s[k:]]
                if all(is_valid_segment(part) for part in parts):
                    results.append(".".join(parts))
    return sorted(results)


def is_valid_ip(text, s):
    parts = text.split(".")
    return (len(parts) == 4 and all(is_valid_segment(part) for part in parts)
            and "".join(parts) == s)


class Lcg:
    """Deterministic generator: next = (next * 75 + 74) % 65537."""

    def __init__(self, seed=13579):
        self.state = seed

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def main():
    assert restore_ip_addresses("25525511135") == ["255.255.11.135", "255.255.111.35"]
    assert restore_ip_addresses("0000") == ["0.0.0.0"]
    assert restore_ip_addresses("1111") == ["1.1.1.1"]
    assert restore_ip_addresses("010010") == ["0.10.0.10", "0.100.1.0"]
    assert restore_ip_addresses("101023") == ["1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3"]
    assert restore_ip_addresses("255255255255") == ["255.255.255.255"]

    assert restore_ip_addresses("111") == []            # too short for four parts
    assert restore_ip_addresses("") == []
    assert restore_ip_addresses("1111111111111") == []  # 13 digits: too long
    assert restore_ip_addresses("256256256256") == []   # every segment exceeds 255
    assert restore_ip_addresses("111111111111111111111") == []

    rng = Lcg()
    for trial in range(200):
        length = 1 + rng.next() % 14
        s = "".join(str(rng.next() % 10) for _ in range(length))
        results = restore_ip_addresses(s)
        assert sorted(results) == brute_force_ip_addresses(s), s
        for text in results:
            assert is_valid_ip(text, s), (s, text)
        assert len(results) == len(set(results)), s


if __name__ == "__main__":
    main()
    print("PASS")
