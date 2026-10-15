
"""Task 163 - Implement basic encryption/decryption with XOR. (Tier: Beginner, CTCI Ch 5)

XOR is its own inverse: cipher = plain ^ key, plain = cipher ^ key.
EDUCATIONAL ONLY - repeating-key XOR offers no real security.
"""


def xor_bytes(data, key):
    if not key:
        raise ValueError("key must not be empty")
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def xor_text(text, key, encoding="utf-8"):
    if not key:
        raise ValueError("key must not be empty")
    key_bytes = key.encode(encoding)
    cipher = xor_bytes(text.encode(encoding), key_bytes)
    return cipher.hex()


def decrypt_text(hex_cipher, key, encoding="utf-8"):
    if not key:
        raise ValueError("key must not be empty")
    key_bytes = key.encode(encoding)
    plain = xor_bytes(bytes.fromhex(hex_cipher), key_bytes)
    return plain.decode(encoding)


def main():
    message = "attack at dawn"
    key = "s3cret"
    cipher = xor_text(message, key)
    assert cipher != message and all(c in "0123456789abcdef" for c in cipher)
    assert decrypt_text(cipher, key) == message

    assert xor_text("", key) == "" and decrypt_text("", key) == ""
    # XOR is self-inverse
    raw = b"\x00\xff\x7f"
    assert xor_bytes(xor_bytes(raw, b"k"), b"k") == raw
    # single-byte key equals key of length 1
    assert xor_text("abc", "x") == xor_text("abc", "x")
    # wrong key yields garbage, not the plaintext
    assert decrypt_text(cipher, "other") != message


if __name__ == "__main__":
    main()
    print("PASS")
