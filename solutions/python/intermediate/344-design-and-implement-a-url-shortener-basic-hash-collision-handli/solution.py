"""Task 344 - URL shortener: deterministic hash, base62 codes, linear probing. (Tier: Intermediate, CTCI Ch 7)

code(url) = base62 of (additive hash of the URL) % 62^6, zero-padded to six
characters (MSB first, alphabet 0-9a-zA-Z documented below).  The hash is
deliberately simple - the sum of code points - so the tests can engineer
collisions: "…/ab" and "…/ba" hash identically.  Collisions are resolved by
linear probing: index, index+1, ... (mod 62^6); the first free slot wins.
shorten() is idempotent for the same URL, expand() returns None for an unknown
code (documented) and distinct URLs always receive distinct codes.
"""

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE = 62
CODE_WIDTH = 6
SPACE = BASE ** CODE_WIDTH


def hash_url(url):
    total = 0
    for ch in url:
        total += ord(ch)
    return total % SPACE


def encode_base62(value):
    value %= SPACE
    digits = []
    for _ in range(CODE_WIDTH):
        digits.append(ALPHABET[value % BASE])
        value //= BASE
    return "".join(reversed(digits))


def decode_base62(code):
    """Independent inverse, used by the tests to verify the probing order."""
    value = 0
    for ch in code:
        index = ALPHABET.find(ch)
        if index < 0:
            raise ValueError("invalid base62 character %r" % ch)
        value = value * BASE + index
    return value


class URLShortener:
    def __init__(self):
        self._url_to_code = {}
        self._code_to_url = {}
        self.probes = 0

    def shorten(self, url):
        if url in self._url_to_code:
            return self._url_to_code[url]           # idempotent
        index = hash_url(url)
        while True:
            code = encode_base62(index)
            if code not in self._code_to_url:
                self._url_to_code[url] = code
                self._code_to_url[code] = url
                return code
            index = (index + 1) % SPACE             # linear probing
            self.probes += 1

    def expand(self, code):
        return self._code_to_url.get(code)          # None when unknown

    def size(self):
        return len(self._url_to_code)


def main():
    shortener = URLShortener()
    url = "http://example.com/articles/42"
    code = shortener.shorten(url)
    assert len(code) == CODE_WIDTH
    assert all(ch in ALPHABET for ch in code)
    assert code == encode_base62(hash_url(url))     # initial slot free -> no probing
    assert shortener.expand(code) == url
    assert shortener.shorten(url) == code           # idempotent
    assert shortener.size() == 1
    assert shortener.probes == 0

    # unknown codes return None (documented)
    assert shortener.expand("zzzzzz") is None
    assert shortener.expand("") is None
    assert shortener.expand("!!!!!!") is None

    # encode/decode round trips, including the boundaries
    assert encode_base62(0) == "000000"
    assert encode_base62(SPACE - 1) == "ZZZZZZ"
    seed = 9137
    for _ in range(200):
        seed = (seed * 75 + 74) % 65537
        value = seed * 12345 % SPACE
        assert decode_base62(encode_base62(value)) == value

    # engineered collision: anagrams hash to the same slot; linear probing escapes
    u1, u2 = "http://x.com/pages/ab", "http://x.com/pages/ba"
    assert hash_url(u1) == hash_url(u2)
    collision = URLShortener()
    c1 = collision.shorten(u1)
    c2 = collision.shorten(u2)
    assert c1 != c2
    assert decode_base62(c1) == hash_url(u1)
    assert decode_base62(c2) == (decode_base62(c1) + 1) % SPACE   # exactly one probe step
    assert collision.probes == 1
    assert collision.expand(c1) == u1 and collision.expand(c2) == u2

    # many URLs, every one in a known-colliding pair
    batch = []
    for k in range(250):
        pair = ("http://bench/%d/ab" % k, "http://bench/%d/ba" % k)
        assert hash_url(pair[0]) == hash_url(pair[1])
        batch.append(pair)
    batch_shortener = URLShortener()
    codes = []
    for pair in batch:
        for u in pair:
            c = batch_shortener.shorten(u)
            assert batch_shortener.shorten(u) == c          # idempotent
            assert batch_shortener.expand(c) == u
            assert len(c) == CODE_WIDTH and all(ch in ALPHABET for ch in c)
            codes.append(c)
    assert len(codes) == 500
    assert len(set(codes)) == 500                           # distinct despite collisions
    assert batch_shortener.probes > 0                       # collisions were exercised
    assert batch_shortener.size() == 500

    # deterministic: a fresh instance reproduces the same codes in the same order
    again = URLShortener()
    assert [again.shorten(u) for pair in batch for u in pair] == codes


if __name__ == "__main__":
    main()
    print("PASS")
