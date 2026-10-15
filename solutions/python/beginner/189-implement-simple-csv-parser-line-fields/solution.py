"""Task 189 - Implement simple CSV parser (line -> fields). (Tier: Beginner, CTCI Ch 7)

RFC-4180 basics for a single line:
- ',' separates fields outside quotes;
- a field starting with '"' is quoted, and "" inside a quoted field is one literal quote;
- only ',' or the end of the line may follow a closing quote — anything else is an error;
- an empty line yields one empty field and a trailing comma yields a final empty field.

Raises ValueError for an unterminated quoted field, junk after a closing quote, or a
quote inside an unquoted field.
"""


def parse_csv_line(line):
    """Split one CSV line into its fields."""
    fields = []
    i = 0
    while True:
        chunk = []
        if i < len(line) and line[i] == '"':
            i += 1
            while True:
                if i >= len(line):
                    raise ValueError("unterminated quoted field")
                if line[i] == '"':
                    if i + 1 < len(line) and line[i + 1] == '"':
                        chunk.append('"')
                        i += 2
                        continue
                    i += 1                       # closing quote
                    break
                chunk.append(line[i])
                i += 1
            if i < len(line) and line[i] != ',':
                raise ValueError("unexpected character after closing quote")
        else:
            while i < len(line) and line[i] != ',':
                if line[i] == '"':
                    raise ValueError("quote inside unquoted field")
                chunk.append(line[i])
                i += 1
        fields.append("".join(chunk))
        if i >= len(line):
            return fields
        i += 1                                   # skip the separating comma


def main():
    assert parse_csv_line("a,b,c") == ["a", "b", "c"]
    assert parse_csv_line('a,"b,c",d') == ["a", "b,c", "d"]
    assert parse_csv_line('"a""b",c') == ['a"b', "c"]
    assert parse_csv_line("") == [""]
    assert parse_csv_line("a,,c") == ["a", "", "c"]
    assert parse_csv_line("a,b,") == ["a", "b", ""]
    assert parse_csv_line(",,") == ["", "", ""]
    assert parse_csv_line('"",x') == ["", "x"]
    assert parse_csv_line('"line""break"') == ['line"break']
    assert parse_csv_line('"a,b","c,d"') == ["a,b", "c,d"]
    assert parse_csv_line(" , ") == [" ", " "]
    assert parse_csv_line('"  spaced  ",tail') == ["  spaced  ", "tail"]
    assert parse_csv_line("single") == ["single"]

    for bad in ('a,"b,c', '"', 'a,"b"c,d', 'a"b', '"unterminated'):
        try:
            parse_csv_line(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % bad)


if __name__ == "__main__":
    main()
    print("PASS")
