/** Task 968 - Implement localization pipeline to translate problems into multiple languages.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A message-catalog localization pipeline. A problem field is a keyed message: one template with flat
 *  {arg} placeholders, or a plural message holding one template per CLDR category. A locale resolves a key
 *  by walking its fallback chain (pt-BR -> pt -> en) and accepts the first candidate whose placeholder
 *  parity equals the source's, so a translation can never silently drop or invent a slot. Plural
 *  categories: en/pt one|other, id other-only, ru one|few|many; a category a translation lacks degrades to
 *  `other`, and a message with no renderable candidate degrades to the source locale.
 *  Simplified: an in-memory catalog with hand-written plural rules for five locales, no translation memory
 *  and no MT step (ASCII transliterations keep the file encoding-safe); the fallback order, the parity
 *  gate, plural selection and coverage accounting are the real ones.
 *  Verified: exact hit, region fallback, chain exhaustion, parity rejection, plural categories, branch
 *  degradation, coverage vs an independent count, a naive-substitution oracle, determinism, empty catalog.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final String SOURCE = "en";
    static final class Message {
        final String key;
        final Map<String, String> flat = new TreeMap<>();                    // locale -> {arg} template
        final Map<String, Map<String, String>> plur = new TreeMap<>();       // locale -> category -> template
        Message(String key, String src) { this.key = key; flat.put(SOURCE, src); }
        Message(String key) { this.key = key; }                              // plural-only message
        Message tr(String locale, String t) { flat.put(locale, t); return this; }
        Message plural(String locale, String category, String t) {
            plur.computeIfAbsent(locale, k -> new TreeMap<>()).put(category, t);
            return this;
        }
        boolean isPlural() { return !plur.isEmpty(); }
        boolean has(String locale) { return flat.containsKey(locale) || plur.containsKey(locale); }
    }
    static final class Catalog {
        final Map<String, Message> byKey = new TreeMap<>();
        Catalog add(Message m) { byKey.put(m.key, m); return this; }
        int size() { return byKey.size(); }
        Message get(String key) { if (!byKey.containsKey(key)) throw new IllegalArgumentException(key); return byKey.get(key); }
    }

    /** CLDR-flavoured plural category for the locales this pipeline supports. */
    static String plural(String locale, int n) {
        String lang = locale.contains("-") ? locale.substring(0, locale.indexOf('-')) : locale;
        int a = Math.abs(n);
        if (lang.equals("id") || lang.equals("ja") || lang.equals("zh") || lang.equals("ko")) return "other";
        if (lang.equals("ru")) {
            if (a % 10 == 1 && a % 100 != 11) return "one";
            if (a % 10 >= 2 && a % 10 <= 4 && !(a % 100 >= 12 && a % 100 <= 14)) return "few";
            return "many";
        }
        return a == 1 ? "one" : "other";
    }

    /** Argument names one flat template references: every `{name}` span. */
    static SortedSet<String> argNames(String template) {
        SortedSet<String> out = new TreeSet<>();
        for (int i = 0; i < template.length(); i++) {
            if (template.charAt(i) != '{') continue;
            int close = template.indexOf('}', i);
            if (close < 0) throw new IllegalArgumentException("unbalanced: " + template);
            out.add(template.substring(i + 1, close).trim());
            i = close;
        }
        return out;
    }

    /** Argument names visible for one locale: its flat template, or the union over its plural categories. */
    static SortedSet<String> argNames(Message m, String locale) {
        if (!m.isPlural()) return argNames(m.flat.get(locale));
        SortedSet<String> out = new TreeSet<>();
        for (String t : m.plur.get(locale).values()) out.addAll(argNames(t));
        return out;
    }

    /** Substitutes {arg} spans; a missing argument is a hard error for the candidate locale. */
    static String substitute(String template, Map<String, String> texts) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < template.length(); i++) {
            if (template.charAt(i) != '{') { out.append(template.charAt(i)); continue; }
            int close = template.indexOf('}', i);
            String name = template.substring(i + 1, close).trim();
            if (!texts.containsKey(name)) throw new IllegalStateException("missing argument {" + name + "}");
            out.append(texts.get(name));
            i = close;
        }
        return out.toString();
    }
    /** Category template for locale+counter; a category the locale lacks degrades to `other`. */
    static String pick(Map<String, String> categories, String locale, int n) {
        String t = categories.getOrDefault(plural(locale, n), categories.get("other"));
        if (t == null) throw new IllegalStateException("no 'other' category for " + locale);
        return t;
    }

    static final class Resolved {
        final String locale, text; final List<String> notes = new ArrayList<>();
        Resolved(String locale, String text) { this.locale = locale; this.text = text; }
    }
    /** Resolves one key: first chain locale with matching placeholder parity that also renders. */
    static Resolved resolve(Catalog catalog, String key, String locale, int n, Map<String, String> texts) {
        Message msg = catalog.get(key);
        SortedSet<String> want = argNames(msg, SOURCE);
        Map<String, String> args = new TreeMap<>(texts);
        args.putIfAbsent("n", Integer.toString(n));
        List<String> chain = new ArrayList<>(List.of(locale));
        int dash = locale.indexOf('-');
        if (dash > 0) chain.add(locale.substring(0, dash));
        if (!chain.contains(SOURCE)) chain.add(SOURCE);
        List<String> notes = new ArrayList<>();
        for (String candidate : chain) {
            if (!msg.has(candidate)) continue;
            if (!argNames(msg, candidate).equals(want)) {
                notes.add("rejected " + candidate + ": placeholder parity " + argNames(msg, candidate) + " != " + want);
                continue;
            }
            try {
                String template = msg.isPlural() ? pick(msg.plur.get(candidate), candidate, n) : msg.flat.get(candidate);
                Resolved r = new Resolved(candidate, substitute(template, args));
                r.notes.addAll(notes);
                return r;
            } catch (RuntimeException e) {
                notes.add("rejected " + candidate + ": " + e.getMessage());
            }
        }
        String src = msg.isPlural() ? pick(msg.plur.get(SOURCE), SOURCE, n) : msg.flat.get(SOURCE);
        Resolved r = new Resolved(SOURCE, substitute(src, args));
        r.notes.addAll(notes);
        return r;
    }

    static final class Field {
        final String key; final int n; final Map<String, String> texts = new TreeMap<>();
        Field(String key) { this(key, 1); }
        Field(String key, int n) { this.key = key; this.n = n; }
        Field txt(String k, String v) { texts.put(k, v); return this; }
    }
    /** Six fields across two problems: title x2, hint, tags, revisit(n=3) and revisit(n=1). */
    static List<Field> fields() {
        return List.of(
                new Field("problem.title").txt("count", "2").txt("name", "Two Sum"),
                new Field("problem.hint").txt("structure", "hash-map"),
                new Field("problem.revisit", 3),
                new Field("problem.title").txt("count", "3").txt("name", "Maze"),
                new Field("problem.tags").txt("topic", "graphs"),
                new Field("problem.revisit", 1));
    }

    /** Localizes every field; `audit` records the resolved locale plus any rejected candidates. */
    static List<String> localize(Catalog catalog, List<Field> fs, String locale, List<String> audit) {
        List<String> out = new ArrayList<>();
        for (Field f : fs) {
            Resolved r = resolve(catalog, f.key, locale, f.n, f.texts);
            out.add(r.text);
            audit.add(f.key + " -> " + r.locale + (r.notes.isEmpty() ? "" : " " + r.notes));
        }
        return out;
    }

    /** Fraction of catalog keys carrying a template for the locale itself (no fallback credit). */
    static Map<String, Double> coverage(Catalog catalog, List<String> locales) {
        Map<String, Double> out = new TreeMap<>();
        for (String locale : locales) {
            int hit = 0;
            for (Message m : catalog.byKey.values()) if (m.has(locale)) hit++;
            out.put(locale, catalog.size() == 0 ? 0.0 : (double) hit / catalog.size());
        }
        return out;
    }
    /** Independent oracle: repeated brace substitution over a key-sorted argument map. */
    static String naive(String template, Map<String, String> texts) {
        String out = template;
        for (Map.Entry<String, String> e : texts.entrySet()) out = out.replace("{" + e.getKey() + "}", e.getValue());
        return out;
    }

    static final List<String> LOCALES = List.of("en", "pt", "pt-BR", "id", "ru", "de");

    static Catalog catalog() {
        return new Catalog()
                .add(new Message("problem.title", "Sum {count} numbers named {name}")
                        .tr("pt", "Soma de {count} numeros chamados {name}")
                        .tr("pt-BR", "Soma de numeros chamados assim"))       // drops both slots: parity error
                .add(new Message("problem.hint", "Try a {structure} approach")
                        .tr("pt", "Tente uma abordagem com {structure}")
                        .tr("pt-BR", "Tente uma abordagem com {structure}"))
                .add(new Message("problem.tags", "Topics: {topic}"))
                .add(new Message("problem.revisit")
                        .plural(SOURCE, "one", "Revisit {n} problem")
                        .plural(SOURCE, "other", "Revisit {n} problems")
                        .plural("pt", "one", "Revise {n} problema")
                        .plural("pt", "other", "Revise {n} problemas")
                        .plural("id", "other", "Ulangi {n} soal")
                        .plural("ru", "one", "Povtorite {n} zadachu")
                        .plural("ru", "few", "Povtorite {n} zadachi")
                        .plural("ru", "other", "Povtorite {n} zadach"));
    }

    public static void main(String[] args) {
        Catalog catalog = catalog();
        List<String> audit = new ArrayList<>();
        List<String> ptBr = localize(catalog, fields(), "pt-BR", audit);
        check(ptBr.get(1).equals("Tente uma abordagem com hash-map") && audit.get(1).equals("problem.hint -> pt-BR")
                        && ptBr.get(0).equals("Soma de 2 numeros chamados Two Sum") && audit.get(0).contains("-> pt [")
                        && audit.get(0).contains("rejected pt-BR: placeholder parity [] != [count, name]"),
                "a complete template is used as-is; a parity-broken region template falls back to its parent: "
                        + audit.subList(0, 2));
        check(ptBr.get(4).equals("Topics: graphs") && audit.get(4).equals("problem.tags -> en"),
                "an untranslated key must resolve to the source locale without notes: " + audit.get(4));
        check(localize(catalog, fields(), "de", new ArrayList<>()).equals(List.of("Sum 2 numbers named Two Sum",
                        "Try a hash-map approach", "Revisit 3 problems", "Sum 3 numbers named Maze",
                        "Topics: graphs", "Revisit 1 problem")),
                "an unsupported locale must degrade to source text");
        List<String> ruAudit = new ArrayList<>();
        check(plural("en", 1).equals("one") && plural("pt-BR", 0).equals("other") && plural("id", 1).equals("other")
                        && plural("ru", 1).equals("one") && plural("ru", 3).equals("few")
                        && plural("ru", 5).equals("many") && plural("ru", 11).equals("many")
                        && plural("ru", 21).equals("one") && plural("ru", 22).equals("few")
                        && localize(catalog, List.of(new Field("problem.revisit", 5)), "ru", ruAudit)
                        .equals(List.of("Povtorite 5 zadach")) && ruAudit.get(0).equals("problem.revisit -> ru")
                        && localize(catalog, List.of(new Field("problem.revisit", 3)), "ru", new ArrayList<>())
                        .equals(List.of("Povtorite 3 zadachi"))
                        && localize(catalog, List.of(new Field("problem.revisit", 1)), "id", new ArrayList<>())
                        .equals(List.of("Ulangi 1 soal")) && ptBr.get(2).equals("Revise 3 problemas"),
                "plural categories must follow the language rule and a missing branch must degrade to `other`");
        Map<String, Double> stats = coverage(catalog, LOCALES);
        Map<String, String> titleArgs = new TreeMap<>(Map.of("count", "2", "name", "Two Sum"));
        Map<String, String> hintArgs = new TreeMap<>(Map.of("structure", "two-pointers"));
        check(stats.equals(new TreeMap<>(Map.of("en", 1.0, "pt", 0.75, "pt-BR", 0.5, "id", 0.25, "ru", 0.25,
                        "de", 0.0)))
                        && naive(catalog.get("problem.title").flat.get("pt"), titleArgs)
                        .equals(resolve(catalog, "problem.title", "pt", 1, titleArgs).text)
                        && naive(catalog.get("problem.hint").flat.get("pt-BR"), hintArgs)
                        .equals("Tente uma abordagem com two-pointers"),
                "coverage must match an independent count; single templates must match the oracle: " + stats);
        List<String> audit2 = new ArrayList<>();
        check(localize(catalog, fields(), "pt-BR", audit2).equals(ptBr) && audit2.equals(audit)
                        && coverage(new Catalog(), List.of("en")).get("en") == 0.0
                        && coverage(new Catalog(), LOCALES).size() == LOCALES.size(),
                "two runs must be byte-identical and an empty catalog must report zero coverage");
        System.out.println("locales=" + LOCALES.size() + " keys=" + catalog.size() + " fields=" + ptBr.size()
                + " coverage=" + stats);
        System.out.println("PASS");
    }
}
