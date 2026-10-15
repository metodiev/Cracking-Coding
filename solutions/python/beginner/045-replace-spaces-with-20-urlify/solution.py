"""Task 045 - Replace spaces with %20 (URLify). (Tier: Beginner, CTCI Ch 1)"""

def urlify(chars, length):
    """Replace spaces with %20 in a char list padded with trailing capacity (in place)."""
    write = len(chars) - 1
    for read in range(length - 1, -1, -1):
        if chars[read] == " ":
            chars[write - 2:write + 1] = ["%", "2", "0"]
            write -= 3
        else:
            chars[write] = chars[read]
            write -= 1
    return chars


def main():
    chars = list("a b c" + " " * 4)   # 5 real chars + 4 spare slots for two %20s
    assert "".join(urlify(chars, 5)) == "a%20b%20c"
    chars = list("abc")
    assert "".join(urlify(chars, 3)) == "abc"


if __name__ == "__main__":
    main()
    print("PASS")
