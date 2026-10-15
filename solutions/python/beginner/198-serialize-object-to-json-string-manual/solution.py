r"""Task 198 - Serialize object to JSON string (manual). (Tier: Beginner, CTCI Ch 7)

Hand-rolled JSON serializer (no stdlib json) for None, booleans, integers, strings,
lists/tuples and dicts with string keys. Map keys are emitted in sorted order so the
output is deterministic. Strings escape \" \\ \n \t \r \b \f and other control
characters become \u00XX. Output is compact (no spaces). The self-test uses stdlib
json only as a cross-check oracle.
"""

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
    "\b": "\\b",
    "\f": "\\f",
}


def _serialize_string(value):
    parts = ['"']
    for ch in value:
        if ch in _ESCAPES:
            parts.append(_ESCAPES[ch])
        elif ord(ch) < 0x20:
            parts.append("\\u%04x" % ord(ch))
        else:
            parts.append(ch)
    parts.append('"')
    return "".join(parts)


def to_json(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _serialize_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(to_json(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("object keys must be strings")
        return "{" + ",".join(_serialize_string(key) + ":" + to_json(value[key])
                              for key in sorted(value)) + "}"
    raise TypeError("unsupported type: %s" % type(value).__name__)


def main():
    import json  # stdlib json is used only as a cross-check oracle in this self-test

    assert to_json(None) == "null"
    assert to_json(True) == "true"
    assert to_json(False) == "false"
    assert to_json(42) == "42"
    assert to_json(-7) == "-7"
    assert to_json("") == '""'
    assert to_json([]) == "[]"
    assert to_json({}) == "{}"
    assert to_json("a\"b\\c\nd\te\rf") == '"a\\"b\\\\c\\nd\\te\\rf"'
    assert to_json({"b": 1, "a": [True, None, "x"]}) == '{"a":[true,null,"x"],"b":1}'
    assert to_json({"a": {"b": {"c": [1, [2, [3, None]]]}}}) == '{"a":{"b":{"c":[1,[2,[3,null]]]}}}'

    doc = {"users": [{"id": 2, "name": "Bob"}, {"id": 1, "name": "Ann"}],
           "count": 2, "ok": True, "tags": []}
    expected = '{"count":2,"ok":true,"tags":[],"users":[{"id":2,"name":"Bob"},{"id":1,"name":"Ann"}]}'
    assert to_json(doc) == expected
    assert json.loads(to_json(doc)) == doc          # oracle round-trip

    for bad in ({"k": object()}, {1: "x"}, 3.5):
        try:
            to_json(bad)
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError for %r" % (bad,))


if __name__ == "__main__":
    main()
    print("PASS")
