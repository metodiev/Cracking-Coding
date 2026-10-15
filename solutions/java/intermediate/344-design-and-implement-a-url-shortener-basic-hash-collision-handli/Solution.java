/**
 * Task 344 - URL shortener: deterministic hash, base62 codes, linear probing. (Tier: Intermediate, CTCI Ch 7)
 *
 * code(url) = base62 of (sum of the URL's code points) % 62^6, zero-padded to six characters.
 * The deliberately simple hash lets the tests engineer collisions ("…/ab" vs "…/ba"); linear
 * probing (index+1 mod 62^6) finds the first free slot, so distinct URLs always get distinct
 * codes.  shorten() is idempotent and expand() returns null for unknown codes.
 */
import java.util.*;

class UrlShortener {
    static final String ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static final long BASE = 62;
    static final int CODE_WIDTH = 6;
    static final long SPACE = 62L * 62 * 62 * 62 * 62 * 62;      // 62^6 = 56_800_235_584

    private final Map<String, String> urlToCode = new HashMap<>();
    private final Map<String, String> codeToUrl = new HashMap<>();
    long probes = 0;

    static long hashUrl(String url) {
        long total = 0;
        for (int i = 0; i < url.length(); i++) total += url.charAt(i);
        return total % SPACE;
    }

    static String encodeBase62(long value) {
        value %= SPACE;
        char[] digits = new char[CODE_WIDTH];
        for (int i = CODE_WIDTH - 1; i >= 0; i--) {
            digits[i] = ALPHABET.charAt((int) (value % BASE));
            value /= BASE;
        }
        return new String(digits);
    }

    /** Independent inverse; used by the tests to verify the probing order. */
    static long decodeBase62(String code) {
        long value = 0;
        for (int i = 0; i < code.length(); i++) {
            int index = ALPHABET.indexOf(code.charAt(i));
            if (index < 0) throw new IllegalArgumentException("invalid base62 character '" + code.charAt(i) + "'");
            value = value * BASE + index;
        }
        return value;
    }

    String shorten(String url) {
        String existing = urlToCode.get(url);
        if (existing != null) return existing;                   // idempotent
        long index = hashUrl(url);
        while (true) {
            String code = encodeBase62(index);
            if (!codeToUrl.containsKey(code)) {
                urlToCode.put(url, code);
                codeToUrl.put(code, url);
                return code;
            }
            index = (index + 1) % SPACE;                         // linear probing
            probes++;
        }
    }

    String expand(String code) {
        return codeToUrl.get(code);                              // null when unknown
    }

    int size() {
        return urlToCode.size();
    }
}

public class Solution {
    public static void main(String[] args) {
        UrlShortener shortener = new UrlShortener();
        String url = "http://example.com/articles/42";
        String code = shortener.shorten(url);
        check(code.length() == UrlShortener.CODE_WIDTH, "code width");
        for (int i = 0; i < code.length(); i++) {
            check(UrlShortener.ALPHABET.indexOf(code.charAt(i)) >= 0, "code alphabet");
        }
        check(code.equals(UrlShortener.encodeBase62(UrlShortener.hashUrl(url))), "initial slot free, no probing");
        check(url.equals(shortener.expand(code)), "expand round trip");
        check(code.equals(shortener.shorten(url)), "idempotent");
        check(shortener.size() == 1, "size 1");
        check(shortener.probes == 0, "no probes yet");
        check(shortener.expand("zzzzzz") == null, "unknown code returns null");
        check(shortener.expand("") == null, "empty code unknown");
        check(shortener.expand("!!!!!!") == null, "invalid code unknown");

        check(UrlShortener.encodeBase62(0).equals("000000"), "encode 0");
        check(UrlShortener.encodeBase62(UrlShortener.SPACE - 1).equals("ZZZZZZ"), "encode max");
        int seed = 9137;
        for (int i = 0; i < 200; i++) {
            seed = (seed * 75 + 74) % 65537;
            long value = seed * 12345L % UrlShortener.SPACE;
            check(UrlShortener.decodeBase62(UrlShortener.encodeBase62(value)) == value, "base62 round trip");
        }

        // engineered collision: anagram paths hash to the same slot
        String u1 = "http://x.com/pages/ab";
        String u2 = "http://x.com/pages/ba";
        check(UrlShortener.hashUrl(u1) == UrlShortener.hashUrl(u2), "engineered collision");
        UrlShortener collision = new UrlShortener();
        String c1 = collision.shorten(u1);
        String c2 = collision.shorten(u2);
        check(!c1.equals(c2), "distinct codes despite collision");
        check(UrlShortener.decodeBase62(c1) == UrlShortener.hashUrl(u1), "c1 sits at the initial slot");
        check(UrlShortener.decodeBase62(c2) == (UrlShortener.decodeBase62(c1) + 1) % UrlShortener.SPACE, "one probe step");
        check(collision.probes == 1, "exactly one probe");
        check(u1.equals(collision.expand(c1)) && u2.equals(collision.expand(c2)), "both expand correctly");

        // many URLs, every one in a known-colliding pair
        List<String> codes = new ArrayList<>();
        UrlShortener batch = new UrlShortener();
        for (int k = 0; k < 250; k++) {
            String ab = "http://bench/" + k + "/ab";
            String ba = "http://bench/" + k + "/ba";
            check(UrlShortener.hashUrl(ab) == UrlShortener.hashUrl(ba), "pair collision " + k);
            for (String u : new String[]{ab, ba}) {
                String c = batch.shorten(u);
                check(c.equals(batch.shorten(u)), "idempotent batch " + u);
                check(u.equals(batch.expand(c)), "round trip batch " + u);
                check(c.length() == UrlShortener.CODE_WIDTH, "batch width");
                codes.add(c);
            }
        }
        check(codes.size() == 500, "500 codes");
        check(new HashSet<>(codes).size() == 500, "all distinct despite collisions");
        check(batch.probes > 0, "collisions were exercised");
        check(batch.size() == 500, "batch size");

        UrlShortener again = new UrlShortener();
        List<String> reproduced = new ArrayList<>();
        for (int k = 0; k < 250; k++) {
            reproduced.add(again.shorten("http://bench/" + k + "/ab"));
            reproduced.add(again.shorten("http://bench/" + k + "/ba"));
        }
        check(reproduced.equals(codes), "deterministic replay");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
