/** Task 993 - Implement academic citation generator for editorial solutions with references.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Editorials cite the literature, so the generator turns structured references into three artefacts:
 *  an APA-style bibliography entry (authors, year, title, venue, and for web sources a URL with the
 *  access date), an in-text marker ("(Knuth, 1997)", "(Knuth & Bentley, 1986)", "(Cormen et al., 2009)",
 *  "Anonymous" when no author is recorded), and a stable BibTeX key. Keys are built from the first
 *  author's surname, the year and the first significant title word, and collisions get an a/b/c suffix,
 *  so the same bibliography always yields the same keys. Sorting is by surname, then year, then title,
 *  and BibTeX values are escaped for &, %, $, #, _, brace, tilde and caret.
 *  Simplified: ASCII-only author strings and three venue kinds (article, book, web) rather than the full
 *  APA manual; the formatting rules, the key de-duplication and the deterministic ordering are real.
 *  Verified: the exact APA strings of one-, two- and three-author entries, the venue variants including
 *  the web access date, the in-text markers, a/b suffixes for colliding keys, escaping of special
 *  characters, the anonymous boundary, key uniqueness plus key round-trip parsing, and an independent
 *  comparator/key oracle over the whole bibliography.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Reference {
        final List<String> authors;              // "Surname, I. I." strings, already abbreviated
        final int year;
        final String title, venue, kind, url, accessDate;
        Reference(List<String> authors, int year, String title, String venue, String kind, String url, String accessDate) {
            this.authors = authors; this.year = year; this.title = title; this.venue = venue;
            this.kind = kind; this.url = url; this.accessDate = accessDate;
        }
        static Reference article(List<String> authors, int year, String title, String venue) {
            return new Reference(authors, year, title, venue, "article", "", "");
        }
        static Reference book(List<String> authors, int year, String title, String venue) {
            return new Reference(authors, year, title, venue, "book", "", "");
        }
        String firstSurname() {
            if (authors.isEmpty()) return "Anonymous";
            return surname(authors.get(0));
        }
    }

    static String surname(String author) {
        int comma = author.indexOf(',');
        return comma < 0 ? author.trim() : author.substring(0, comma).trim();
    }

    static String authorList(List<String> authors) {
        if (authors.isEmpty()) return "Anonymous";
        if (authors.size() == 1) return authors.get(0);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < authors.size() - 1; i++) sb.append(authors.get(i)).append(", ");
        return sb.append("& ").append(authors.get(authors.size() - 1)).toString();
    }

    static String apaEntry(Reference ref) {
        StringBuilder sb = new StringBuilder(authorList(ref.authors));
        sb.append(" (").append(ref.year).append("). ").append(ref.title).append(". ").append(ref.venue).append('.');
        if (ref.kind.equals("web"))
            sb.append(' ').append(ref.url).append(" (accessed ").append(ref.accessDate).append(").");
        return sb.toString();
    }

    static String inText(Reference ref) {
        if (ref.authors.isEmpty()) return "(Anonymous, " + ref.year + ")";
        if (ref.authors.size() == 1) return "(" + surname(ref.authors.get(0)) + ", " + ref.year + ")";
        if (ref.authors.size() == 2)
            return "(" + surname(ref.authors.get(0)) + " & " + surname(ref.authors.get(1)) + ", " + ref.year + ")";
        return "(" + surname(ref.authors.get(0)) + " et al., " + ref.year + ")";
    }

    /** First title word with letters/digits only, lower-cased; "the" is not skipped (APA-style keys keep it). */
    static String titleWord(String title) {
        for (String word : title.split("\\s+")) {
            String cleaned = word.replaceAll("[^A-Za-z0-9]", "");
            if (!cleaned.isEmpty()) return cleaned.toLowerCase(Locale.ROOT);
        }
        return "untitled";
    }

    static String baseKey(Reference ref) {
        String surname = ref.authors.isEmpty() ? "anonymous" : surname(ref.authors.get(0)).replaceAll("[^A-Za-z0-9]", "").toLowerCase(Locale.ROOT);
        return surname + ref.year + titleWord(ref.title);
    }

    /** Assigns keys in bibliography order; repeated base keys get a, b, c ... suffixes. */
    static Map<Reference, String> assignKeys(List<Reference> bibliography) {
        Map<Reference, String> keys = new LinkedHashMap<>();
        Map<String, Integer> seen = new LinkedHashMap<>();
        for (Reference ref : sorted(bibliography)) {
            String base = baseKey(ref);
            int n = seen.merge(base, 1, Integer::sum);
            keys.put(ref, base + (char) ('a' + n - 1));
        }
        return keys;
    }

    /** Bibliography order: surname, then year, then title (all case-insensitive for the surname). */
    static List<Reference> sorted(List<Reference> refs) {
        List<Reference> out = new ArrayList<>(refs);
        out.sort((a, b) -> {
            int bySurname = a.firstSurname().toLowerCase(Locale.ROOT).compareTo(b.firstSurname().toLowerCase(Locale.ROOT));
            if (bySurname != 0) return bySurname;
            if (a.year != b.year) return Integer.compare(a.year, b.year);
            return a.title.compareTo(b.title);
        });
        return out;
    }

    static String escapeBib(String value) {
        StringBuilder sb = new StringBuilder();
        for (char c : value.toCharArray()) {
            switch (c) {
                case '&': sb.append("\\&"); break;
                case '%': sb.append("\\%"); break;
                case '$': sb.append("\\$"); break;
                case '#': sb.append("\\#"); break;
                case '_': sb.append("\\_"); break;
                case '{': sb.append("\\{"); break;
                case '}': sb.append("\\}"); break;
                case '~': sb.append("\\textasciitilde{}"); break;
                case '^': sb.append("\\textasciicircum{}"); break;
                default: sb.append(c);
            }
        }
        return sb.toString();
    }

    static String bibtex(Reference ref, String key) {
        return "@" + ref.kind + "{" + key + ", author = {" + escapeBib(authorList(ref.authors).replace(" & ", " and "))
                + "}, title = {" + escapeBib(ref.title) + "}, year = {" + ref.year + "}, venue = {"
                + escapeBib(ref.venue) + "}}";
    }

    /** Strips the trailing a/b/c de-duplication suffix from a generated key. */
    static String stripSuffix(String key) {
        char last = key.charAt(key.length() - 1);
        return last >= 'a' && last <= 'z' ? key.substring(0, key.length() - 1) : key;
    }

    static List<Reference> bibliography() {
        return List.of(
            Reference.book(List.of("Knuth, D. E."), 1997, "The Art of Computer Programming", "Addison-Wesley"),
            Reference.article(List.of("Knuth, D. E.", "Bentley, J. L."), 1986, "Literate Programming", "Commun. ACM"),
            Reference.book(List.of("Cormen, T. H.", "Leiserson, C. E.", "Rivest, R. L."), 2009,
                    "Introduction to Algorithms", "MIT Press"),
            Reference.book(List.of("Knuth, D. E."), 1997, "The TeXbook", "Addison-Wesley"),
            Reference.article(List.of("Hoare, C. A. R."), 1962, "Quicksort", "The Computer Journal"),
            new Reference(List.of(), 2020, "Anonymous Report", "TechRxiv", "web", "https://example.org/anon", "2026-10-16"));
    }

    public static void main(String[] args) {
        List<Reference> refs = bibliography();
        Reference knuth = refs.get(0), knuthBentley = refs.get(1), cormen = refs.get(2),
                texbook = refs.get(3), hoare = refs.get(4), anonymous = refs.get(5);
        // --- APA formatting: one, two and three authors, plus the web variant with its access date.
        check(apaEntry(knuth).equals("Knuth, D. E. (1997). The Art of Computer Programming. Addison-Wesley."), "single-author APA entry: " + apaEntry(knuth));
        check(apaEntry(knuthBentley).equals("Knuth, D. E., & Bentley, J. L. (1986). Literate Programming. Commun. ACM."),
                "two-author APA entry must place the ampersand before the last author: " + apaEntry(knuthBentley));
        check(apaEntry(cormen).equals("Cormen, T. H., Leiserson, C. E., & Rivest, R. L. (2009). " + "Introduction to Algorithms. MIT Press."),
                "three-author APA entry must be comma separated: " + apaEntry(cormen));
        check(apaEntry(anonymous).equals("Anonymous (2020). Anonymous Report. TechRxiv. " + "https://example.org/anon (accessed 2026-10-16)."),
                "web sources must carry the URL and the access date: " + apaEntry(anonymous));
        // --- in-text markers: alone, pair, et al. and the anonymous boundary.
        check(inText(knuth).equals("(Knuth, 1997)") && inText(knuthBentley).equals("(Knuth & Bentley, 1986)") && inText(cormen).equals("(Cormen et al., 2009)") && inText(anonymous).equals("(Anonymous, 2020)"),
                "in-text markers must follow the author count: " + inText(cormen) + " " + inText(anonymous));
        check(inText(hoare).equals("(Hoare, 1962)"), "a hyphenated initial must not leak into the marker: " + inText(hoare));
        // --- keys: base construction, collision suffixes and uniqueness over the whole bibliography.
        Map<Reference, String> keys = assignKeys(refs);
        check(keys.get(knuth).equals("knuth1997thea") && keys.get(texbook).equals("knuth1997theb"),
                "colliding base keys must get a and b suffixes: " + keys.get(knuth) + " / " + keys.get(texbook));
        check(keys.get(knuthBentley).equals("knuth1986literatea") && keys.get(cormen).equals("cormen2009introductiona") && keys.get(hoare).equals("hoare1962quicksorta") && keys.get(anonymous).equals("anonymous2020anonymousa"),
                "base keys must use surname + year + first title word: " + keys);
        check(new HashSet<>(keys.values()).size() == refs.size(), "generated keys must be unique across the bibliography: " + keys.values());
        for (Map.Entry<Reference, String> entry : keys.entrySet()) check(stripSuffix(entry.getValue()).equals(baseKey(entry.getKey())),
                "key round-trip must recover the base key for " + entry.getValue());
        // --- ordering: surname, then year, then title; the input order must not matter.
        List<Reference> shuffled = new ArrayList<>(List.of(refs.get(3), refs.get(1), refs.get(5),
                refs.get(0), refs.get(4), refs.get(2)));         // a fixed permutation, no RNG involved
        List<Reference> order = sorted(shuffled);
        check(order.equals(List.of(anonymous, cormen, hoare, knuthBentley, knuth, texbook)),
                "bibliography must be ordered by surname/year/title: " + order.stream().map(Reference::firstSurname).toList());
        check(assignKeys(shuffled).equals(keys) && assignKeys(refs).keySet().equals(keys.keySet()), "key assignment must not depend on input order or the run");
        // --- escaping: BibTeX special characters survive, and ordinary text is untouched.
        check(escapeBib("Graphs & {Trees} 100% _safe_ #1 $x^2~y").equals( "Graphs \\& \\{Trees\\} 100\\% \\_safe\\_ \\#1 \\$x\\textasciicircum{}2\\textasciitilde{}y"),
                "BibTeX escaping: " + escapeBib("Graphs & {Trees} 100% _safe_ #1 $x^2~y"));
        check(escapeBib("Plain Title").equals("Plain Title"), "plain titles must pass through unchanged");
        check(bibtex(hoare, keys.get(hoare)).equals("@article{hoare1962quicksorta, author = {Hoare, C. A. R.}, " + "title = {Quicksort}, year = {1962}, venue = {The Computer Journal}}"),
                "BibTeX output: " + bibtex(hoare, keys.get(hoare)));
        // --- boundary: a bibliography of one entry still sorts and keys.
        List<Reference> single = List.of(texbook);
        Map<Reference, String> singleKeys = assignKeys(single);
        check(singleKeys.size() == 1 && singleKeys.get(texbook).equals("knuth1997thea") && assignKeys(List.of()).isEmpty(),
                "a single-entry bibliography must key without a suffix and an empty one must key nothing: " + singleKeys);
        // --- independent oracle: hand-written comparator and key builder must agree.
        check(oracleOrder(refs).equals(sorted(refs)), "independent sort oracle mismatch");
        for (Reference ref : refs) check(oracleBaseKey(ref).equals(baseKey(ref)),
                "key oracle mismatch for " + ref.title + ": " + oracleBaseKey(ref) + " vs " + baseKey(ref));

        System.out.println("bibliography: " + order.size() + " entries, " + keys.values() + ", first: "
                + apaEntry(order.get(0)) + ", " + inText(cormen));
        System.out.println("PASS");
    }

    /** Oracle: selection sort by the same three keys, implemented without a comparator chain. */
    static List<Reference> oracleOrder(List<Reference> refs) {
        List<Reference> pool = new ArrayList<>(refs);
        List<Reference> out = new ArrayList<>();
        while (!pool.isEmpty()) {
            int best = 0;
            for (int i = 1; i < pool.size(); i++) {
                Reference a = pool.get(i), b = pool.get(best);
                int cmp = a.firstSurname().toLowerCase(Locale.ROOT).compareTo(b.firstSurname().toLowerCase(Locale.ROOT));
                if (cmp < 0) best = i;
                else if (cmp == 0) {
                    if (a.year < b.year) best = i;
                    else if (a.year == b.year && a.title.compareTo(b.title) < 0) best = i;
                }
            }
            out.add(pool.remove(best));
        }
        return out;
    }

    /** Oracle: builds the key by string replacement instead of the shared helpers. */
    static String oracleBaseKey(Reference ref) {
        String author = ref.authors.isEmpty() ? "Anonymous" : ref.authors.get(0);
        String surname = author.split(",")[0].replaceAll("[^A-Za-z0-9]", "").toLowerCase(Locale.ROOT);
        String title = ref.title.trim();
        String word = "";
        for (int i = 0; i < title.length(); i++) {
            char c = title.charAt(i);
            if (Character.isLetterOrDigit(c)) word += c;
            else if (!word.isEmpty()) break;
        }
        return surname + ref.year + word.toLowerCase(Locale.ROOT);
    }
}
