
"""Task 168 - Implement basic checksum (sum bytes). (Tier: Beginner, CTCI Ch 5)

Two simple checksums: 8-bit sum (mod 256) and 16-bit sum (mod 65536).
These DETECT accidental corruption; they are not cryptographic digests.
"""


def checksum8(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return sum(data) % 256


def checksum16(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return sum(data) % 65536


def main():
    assert checksum8(b"") == 0
    assert checksum8("abc") == (97 + 98 + 99) % 256 == 38
    assert checksum16("abc") == 294
    assert checksum8(b"\xff\x01") == 0            # wraps around
    assert checksum16(b"\xff\xff") == 510

    # single-byte corruption almost always changes an 8-bit checksum
    original = bytes(range(0, 50))
    corrupt = bytearray(original)
    corrupt[10] ^= 0x01
    assert checksum8(original) != checksum8(bytes(corrupt))
    # but a swap of two differing bytes does not (order-insensitive sum)
    swapped = bytearray(original)
    swapped[3], swapped[4] = swapped[4], swapped[3]
    assert checksum8(bytes(swapped)) == checksum8(original)


if __name__ == "__main__":
    main()
    print("PASS")
