r"""Task 199 - Deserialize JSON with arrays & objects (basic). (Tier: Beginner, CTCI Ch 7)

Recursive-descent JSON parser (no stdlib json): objects, arrays, strings with escapes,
integers, true/false/null. Numbers are integers only (floats are rejected, documented);
\uXXXX escapes are decoded to single code units. Malformed input raises JsonParseError
with the offending index. The self-test cross-checks against stdlib json where useful.
"""


class JsonParseError(ValueError):
    pass


class JsonParser:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def parse(self):
        value = self._parse_value()
        self._skip_ws()
        if self.pos != len(self.text):
            self._fail("trailing characters after JSON value")
        return value

    def _fail(self, message):
        raise JsonParseError("%s at index %d" % (message, self.pos))

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def _peek(self):
        if self.pos >= len(self.text):
            self._fail("unexpected end of input")
        return self.text[self.pos]

    def _expect(self, ch):
        if self._peek() != ch:
            self._fail("expected %r" % ch)
        self.pos += 1

    def _parse_value(self):
        self._skip_ws()
        ch = self._peek()
        if ch == "{":
            return self._parse_object()
        if ch == "[":
            return self._parse_array()
        if ch == '"':
            return self._parse_string()
        if ch in "tfn":
            return self._parse_literal()
        if ch == "-" or ch.isdigit():
            return self._parse_number()
        self._fail("unexpected character %r" % ch)

    def _parse_literal(self):
        for word, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(word, self.pos):
                self.pos += len(word)
                return value
        self._fail("invalid literal")

    def _parse_number(self):
        start = self.pos
        if self._peek() == "-":
            self.pos += 1
        if self.pos >= len(self.text) or not self.text[self.pos].isdigit():
            self._fail("invalid number")
        if self.text[self.pos] == "0":
            self.pos += 1
            if self.pos < len(self.text) and self.text[self.pos].isdigit():
                self._fail("leading zeros are not allowed")
        else:
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        if self.pos < len(self.text) and self.text[self.pos] in ".eE":
            self._fail("only integer numbers are supported")
        return int(self.text[start:self.pos])

    def _parse_string(self):
        self._expect('"')
        out = []
        while True:
            if self.pos >= len(self.text):
                self._fail("unterminated string")
            ch = self.text[self.pos]
            self.pos += 1
            if ch == '"':
                return "".join(out)
            if ch == "\\":
                out.append(self._parse_escape())
            elif ord(ch) < 0x20:
                self._fail("unescaped control character in string")
            else:
                out.append(ch)

    def _parse_escape(self):
        if self.pos >= len(self.text):
            self._fail("unterminated escape")
        esc = self.text[self.pos]
        self.pos += 1
        simple = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        if esc in simple:
            return simple[esc]
        if esc == "u":
            if self.pos + 4 > len(self.text):
                self._fail("incomplete \\u escape")
            digits = self.text[self.pos:self.pos + 4]
            if any(d not in "0123456789abcdefABCDEF" for d in digits):
                self._fail("invalid \\u escape")
            self.pos += 4
            return chr(int(digits, 16))
        self._fail("invalid escape %r" % esc)

    def _parse_object(self):
        self._expect("{")
        result = {}
        self._skip_ws()
        if self._peek() == "}":
            self.pos += 1
            return result
        while True:
            self._skip_ws()
            key = self._parse_string()
            self._skip_ws()
            self._expect(":")
            result[key] = self._parse_value()
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.pos += 1
                self._skip_ws()
                if self._peek() == "}":
                    self._fail("trailing comma in object")
                continue
            if ch == "}":
                self.pos += 1
                return result
            self._fail("expected ',' or '}'")

    def _parse_array(self):
        self._expect("[")
        result = []
        self._skip_ws()
        if self._peek() == "]":
            self.pos += 1
            return result
        while True:
            result.append(self._parse_value())
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.pos += 1
                self._skip_ws()
                if self._peek() == "]":
                    self._fail("trailing comma in array")
                continue
            if ch == "]":
                self.pos += 1
                return result
            self._fail("expected ',' or ']'")


def parse_json(text):
    return JsonParser(text).parse()


def main():
    import json  # stdlib json is used only as a cross-check oracle in this self-test

    documents = [
        "{}",
        "[]",
        "null",
        "true",
        '{"a":1,"b":[true,false,null],"c":"hi\\n","d":{"e":[]}}',
        '{"nested":[1,[2,[3]]],"u":"\\u0041","esc":"a\\"b\\\\c"}',
        "  {  \"x\" : [ 1 , 2 ]  } ",
        "-42",
        "0",
    ]
    for text in documents:
        assert parse_json(text) == json.loads(text), "mismatch for %r" % text
    assert parse_json('{"a": 1}') == {"a": 1}

    bad = ['{"a":1,}', "[1,]", '"abc', '{"a" 1}', "tru", "@", "", "1 2", "01", "1.5",
           '{"a":}', "[", "]", "[1 2]", '"\\x"', '"\\u12"']
    for text in bad:
        try:
            parse_json(text)
        except JsonParseError:
            pass
        else:
            raise AssertionError("expected JsonParseError for %r" % text)


if __name__ == "__main__":
    main()
    print("PASS")
