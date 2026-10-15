"""Task 196 - Implement file line reader that yields lines lazily. (Tier: Beginner, CTCI Ch 7)

read_lines(path) is a generator: the file is opened only when iteration starts and
lines are pulled one at a time (nothing is buffered up front). A trailing '\n' is
stripped; a final line without a newline is still yielded; blank lines yield "".
The self-test creates a temp file, reads it lazily, then removes every artifact.
"""
import os
import tempfile


def read_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def main():
    workdir = tempfile.mkdtemp(prefix="line-reader-")
    path = os.path.join(workdir, "sample.txt")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("alpha\nbeta\n\ngamma\ndelta")     # 5 lines, last one unterminated

        # Laziness: the generator does not even open the file until it is advanced.
        gen = read_lines(path)
        os.remove(path)
        try:
            next(gen)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected the file to be opened only when iterating")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("alpha\nbeta\n\ngamma\ndelta")

        gen = read_lines(path)
        assert next(gen) == "alpha"
        assert list(gen) == ["beta", "", "gamma", "delta"]
        assert list(read_lines(path)) == ["alpha", "beta", "", "gamma", "delta"]
    finally:
        if os.path.exists(path):
            os.remove(path)
        os.rmdir(workdir)

    assert not os.path.exists(path), "temp file must be removed"
    assert not os.path.exists(workdir), "temp dir must be removed"


if __name__ == "__main__":
    main()
    print("PASS")
