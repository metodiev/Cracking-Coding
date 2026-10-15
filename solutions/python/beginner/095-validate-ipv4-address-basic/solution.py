"""Task 095 - Validate IPv4 address (basic). (Tier: Beginner, CTCI Ch 1)

Rules: exactly 4 dot-separated parts, ASCII digits only, each 0-255.
Choice: leading zeros are rejected ("1.2.3.04" invalid), except the value "0" itself.
"""


def is_valid_ipv4(s):
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part or len(part) > 3:
            return False
        for c in part:
            if c < "0" or c > "9":
                return False
        if len(part) > 1 and part[0] == "0":
            return False
        if int(part) > 255:
            return False
    return True


def main():
    assert is_valid_ipv4("192.168.1.1") is True
    assert is_valid_ipv4("0.0.0.0") is True
    assert is_valid_ipv4("255.255.255.255") is True
    assert is_valid_ipv4("10.0.0.255") is True
    assert is_valid_ipv4("1.2.3.0") is True
    assert is_valid_ipv4("256.1.1.1") is False
    assert is_valid_ipv4("1.2.3.256") is False
    assert is_valid_ipv4("1.2.3") is False
    assert is_valid_ipv4("1.2.3.4.5") is False
    assert is_valid_ipv4("") is False
    assert is_valid_ipv4(".1.2.3") is False
    assert is_valid_ipv4("1.2.3.") is False
    assert is_valid_ipv4("1..2.3") is False
    assert is_valid_ipv4("1.2.3.a") is False
    assert is_valid_ipv4("1.2.3.4x") is False
    assert is_valid_ipv4("01.2.3.4") is False
    assert is_valid_ipv4("1.2.3.00") is False
    assert is_valid_ipv4("1.2.3.1234") is False
    assert is_valid_ipv4("-1.2.3.4") is False
    assert is_valid_ipv4("1.2.3.4 ") is False


if __name__ == "__main__":
    main()
    print("PASS")
