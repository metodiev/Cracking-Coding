
"""Task 175 - Multiply large integers as strings. (Tier: Beginner, CTCI Ch 6)

Schoolbook multiplication into a digit array: O(n*m) time.
Signs and leading zeros are handled; the result is verified against Python's own
arbitrary-precision ints on large random inputs.
"""


def multiply_strings(a, b):
    if a == "0" or b == "0":
        return "0"
    negative = (a[0] == "-") ^ (b[0] == "-")
    da = [int(ch) for ch in a.lstrip("-")]
    db = [int(ch) for ch in b.lstrip("-")]
    result = [0] * (len(da) + len(db))
    for i in range(len(da) - 1, -1, -1):
        for j in range(len(db) - 1, -1, -1):
            result[i + j + 1] += da[i] * db[j]
    for k in range(len(result) - 1, 0, -1):      # normalize carries
        result[k - 1] += result[k] // 10
        result[k] %= 10
    digits = "".join(str(d) for d in result).lstrip("0") or "0"
    if digits == "0":
        return "0"
    return "-" + digits if negative else digits


def main():
    assert multiply_strings("123", "456") == "56088"
    assert multiply_strings("0", "999") == "0"
    assert multiply_strings("-12", "12") == "-144"
    assert multiply_strings("-12", "-12") == "144"
    assert multiply_strings("2", "3") == "6"
    assert multiply_strings("99999", "99999") == "9999800001"

    import random
    rng = random.Random(11)
    for _ in range(30):
        x = rng.randrange(0, 10 ** rng.randint(1, 40))
        y = rng.randrange(0, 10 ** rng.randint(1, 40))
        assert multiply_strings(str(x), str(y)) == str(x * y), (x, y)

    a = "9" * 100
    assert multiply_strings(a, a) == str(int(a) * int(a))


if __name__ == "__main__":
    main()
    print("PASS")
