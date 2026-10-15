/** Task 196 - Implement file line reader that yields lines lazily. (Tier: Beginner, CTCI Ch 7) */
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class Solution {
    static class LineIterator implements Iterator<String>, AutoCloseable {
        private final String path;
        private BufferedReader reader;
        private String pending;
        private boolean finished;

        LineIterator(String path) {
            this.path = path;
        }

        @Override
        public boolean hasNext() {
            if (pending != null) return true;
            if (finished) return false;
            try {
                if (reader == null) {                       // opened on first use, not in the constructor
                    reader = Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8);
                }
                String line = reader.readLine();
                if (line == null) {
                    finished = true;
                    close();
                    return false;
                }
                pending = line;
                return true;
            } catch (IOException exc) {
                throw new UncheckedIOException(exc);
            }
        }

        @Override
        public String next() {
            if (!hasNext()) throw new NoSuchElementException();
            String line = pending;
            pending = null;
            return line;
        }

        @Override
        public void close() {
            finished = true;
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException ignored) {
                    // nothing sensible to do while closing
                }
                reader = null;
            }
        }
    }

    static LineIterator readLines(String path) {
        return new LineIterator(path);
    }

    public static void main(String[] args) throws IOException {
        Path dir = Files.createTempDirectory("line-reader-");
        Path file = dir.resolve("sample.txt");
        try {
            Files.write(file, "alpha\nbeta\n\ngamma\ndelta".getBytes(StandardCharsets.UTF_8));

            // Laziness: the file is not opened until the iterator is advanced.
            LineIterator lazy = readLines(file.toString());
            Files.delete(file);
            checkThrows(lazy);
            lazy.close();

            Files.write(file, "alpha\nbeta\n\ngamma\ndelta".getBytes(StandardCharsets.UTF_8));

            List<String> lines = new ArrayList<>();
            try (LineIterator iterator = readLines(file.toString())) {
                check(iterator.hasNext(), "first hasNext");
                lines.add(iterator.next());
                while (iterator.hasNext()) lines.add(iterator.next());
            }
            check(lines.equals(Arrays.asList("alpha", "beta", "", "gamma", "delta")), "lines match");
        } finally {
            Files.deleteIfExists(file);
            Files.deleteIfExists(dir);
        }
        check(!Files.exists(file), "temp file removed");
        check(!Files.exists(dir), "temp dir removed");
        System.out.println("PASS");
    }

    static void checkThrows(LineIterator iterator) {
        try {
            iterator.next();
        } catch (UncheckedIOException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected the file to be opened only when iterating");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
