/** Task 967 - Implement accessibility features for lesson content (screen-reader friendly).
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory accessibility auditor for lesson documents. A document is an ordered list of typed blocks
 *  (headings, paragraphs, images, links, code, buttons, tables); the auditor walks it in reading order and
 *  reports the defects a screen-reader user would hit: a missing level-one heading, heading level jumps,
 *  images without alt text, empty or vague link text, icon buttons without an accessible name, code blocks
 *  without a language and tables without a header row. Text colours are checked against the WCAG relative
 *  luminance formula with the 4.5:1 AA threshold, and an outline shows the reading order a screen reader
 *  would follow.
 *  Simplified: the document model and the outline are in-memory structures, not HTML/DOM, and the audit
 *  covers a fixed rule set; the contrast maths, threshold and ordering rules are the real ones.
 *  Verified: contrast references (black on white is 21:1, white on white is 1:1, mid grey lands in the AA
 *  band) hold; a clean document reports nothing; heading order, alt text, link text, button names, code
 *  languages and table headers are all enforced; the outline equals the document order of the blocks it
 *  covers; empty and paragraph-only documents are handled; 20 generated documents report exactly their
 *  planted defects and an independent luminance table reproduces every ratio.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static long splitmix64(long x) {
        long z = x + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final int INK = 0x1A1A1A, PAPER = 0xFFFFFF, GREY = 0x767676, LIGHT = 0xBBBBBB, WHITE = 0xFFFFFF;
    static final double AA_NORMAL_TEXT = 4.5;

    record Block(String kind, int level, String text, String target, int foreground, int background, boolean decorative) {
        static Block heading(int level, String text) { return new Block("heading", level, text, "", INK, PAPER, false); }
        static Block paragraph(String text) { return new Block("paragraph", 0, text, "", INK, PAPER, false); }
        static Block tintedParagraph(String text, int foreground) { return new Block("paragraph", 0, text, "", foreground, PAPER, false); }
        static Block image(String alt) { return new Block("image", 0, alt, "", INK, PAPER, false); }
        static Block decorativeImage() { return new Block("image", 0, "", "", INK, PAPER, true); }
        static Block link(String text, String href) { return new Block("link", 0, text, href, INK, PAPER, false); }
        static Block code(String source, String language) { return new Block("code", 0, source, language, INK, PAPER, false); }
        static Block button(String label, String ariaLabel) { return new Block("button", 0, label, ariaLabel, INK, PAPER, false); }
        static Block table(String headerRow) { return new Block("table", 0, headerRow, "", INK, PAPER, false); }
    }

    static final Set<String> VAGUE_LINK_TEXT = Set.of("click here", "here", "read more", "link", "this");

    static double channel(int component) {
        double value = component / 255.0;
        return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
    }

    static double luminance(int rgb) {
        return 0.2126 * channel((rgb >> 16) & 0xFF) + 0.7152 * channel((rgb >> 8) & 0xFF) + 0.0722 * channel(rgb & 0xFF);
    }

    static double contrast(int first, int second) {
        double a = luminance(first), b = luminance(second);
        return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    }

    /** Independent implementation: a 256 entry sRGB table plus a sorted luminance pair. */
    static final double[] SRGB_TABLE = buildTable();

    static double[] buildTable() {
        double[] table = new double[256];
        for (int i = 0; i < 256; i++) table[i] = i / 255.0 <= 0.03928 ? i / 255.0 / 12.92 : Math.pow((i / 255.0 + 0.055) / 1.055, 2.4);
        return table;
    }

    static double contrastOracle(int first, int second) {
        double a = 0.2126 * SRGB_TABLE[(first >> 16) & 0xFF] + 0.7152 * SRGB_TABLE[(first >> 8) & 0xFF] + 0.0722 * SRGB_TABLE[first & 0xFF];
        double b = 0.2126 * SRGB_TABLE[(second >> 16) & 0xFF] + 0.7152 * SRGB_TABLE[(second >> 8) & 0xFF] + 0.0722 * SRGB_TABLE[second & 0xFF];
        double hi = Math.max(a, b) + 0.05, lo = Math.min(a, b) + 0.05;
        return hi / lo;
    }

    static List<String> audit(List<Block> document) {
        List<String> issues = new ArrayList<>();
        int previousHeading = 0;
        for (int i = 0; i < document.size(); i++) {
            Block block = document.get(i);
            switch (block.kind()) {
                case "heading":
                    if (previousHeading == 0 && block.level() != 1) issues.add(i + ": first heading must be level 1");
                    else if (previousHeading > 0 && block.level() > previousHeading + 1)
                        issues.add(i + ": heading level jumps from " + previousHeading + " to " + block.level());
                    previousHeading = block.level();
                    break;
                case "image":
                    if (!block.decorative() && block.text().isBlank()) issues.add(i + ": image without alt text");
                    break;
                case "link":
                    if (block.text().isBlank()) issues.add(i + ": link without text");
                    else if (VAGUE_LINK_TEXT.contains(block.text().toLowerCase(Locale.ROOT)))
                        issues.add(i + ": vague link text '" + block.text() + "'");
                    break;
                case "button":
                    if (block.target().isBlank()) issues.add(i + ": icon button without an accessible name");
                    break;
                case "code":
                    if (block.target().isBlank()) issues.add(i + ": code block without a language");
                    break;
                case "table":
                    if (block.text().isBlank()) issues.add(i + ": table without a header row");
                    break;
                default:
                    break;
            }
            if ((block.kind().equals("paragraph") || block.kind().equals("heading") || block.kind().equals("link"))
                    && contrast(block.foreground(), block.background()) < AA_NORMAL_TEXT)
                issues.add(i + ": contrast " + String.format("%.2f", contrast(block.foreground(), block.background()))
                        + ":1 is below " + AA_NORMAL_TEXT);
        }
        return issues;
    }

    /** Reading order a screen reader follows, one entry per block that carries text or a name. */
    static List<String> outline(List<Block> document) {
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < document.size(); i++) {
            Block block = document.get(i);
            switch (block.kind()) {
                case "heading": lines.add("h" + block.level() + ": " + block.text()); break;
                case "paragraph": lines.add("p: " + block.text()); break;
                case "image": lines.add("img: " + (block.decorative() ? "(decorative)" : block.text())); break;
                case "link": lines.add("a: " + block.text()); break;
                case "button": lines.add("button: " + (block.target().isBlank() ? block.text() : block.target())); break;
                case "code": lines.add("code(" + block.target() + ")"); break;
                case "table": lines.add("table: " + block.text()); break;
                default: break;
            }
        }
        return lines;
    }

    static List<Block> cleanLesson() {
        return List.of(Block.heading(1, "Two pointers"), Block.paragraph("Walk from both ends."),
                Block.image("Two pointers moving towards each other"), Block.heading(2, "Complexity"),
                Block.paragraph("Linear time."), Block.code("while (i < j) { i++; j--; }", "java"),
                Block.link("two pointers editorial", "https://example.invalid/editorial"),
                Block.table("case | result"), Block.button("", "run the sample"), Block.decorativeImage());
    }

    public static void main(String[] args) {
        // 1. contrast references and the AA decision
        check(Math.abs(contrast(0x000000, WHITE) - 21.0) < 0.01 && Math.abs(contrast(WHITE, WHITE) - 1.0) < 1e-9
                        && contrast(GREY, WHITE) > AA_NORMAL_TEXT && contrast(GREY, WHITE) < 4.6
                        && contrast(LIGHT, WHITE) < AA_NORMAL_TEXT && contrast(INK, PAPER) > AA_NORMAL_TEXT,
                "contrast references hold: black on white 21:1, white on white 1:1, grey in the AA band");

        // 2. a clean lesson reports nothing
        List<Block> clean = cleanLesson();
        check(audit(clean).isEmpty(), "a well formed lesson reports no accessibility issues");

        // 3. heading order: a jump and a missing level one heading are both reported
        List<Block> jump = List.of(Block.heading(1, "A"), Block.heading(3, "C"), Block.paragraph("text"));
        List<Block> noTop = List.of(Block.heading(2, "A"), Block.paragraph("text"));
        check(audit(jump).size() == 1 && audit(jump).get(0).contains("jumps from 1 to 3")
                        && audit(noTop).size() == 1 && audit(noTop).get(0).contains("must be level 1"),
                "heading level jumps and a missing level one heading are reported");

        // 4. images need alt text unless they are explicitly decorative
        List<Block> images = List.of(Block.heading(1, "T"), Block.image(""), Block.decorativeImage());
        check(audit(images).size() == 1 && audit(images).get(0).contains("without alt text")
                        && outline(images).get(1).equals("img: ") && outline(images).get(2).equals("img: (decorative)"),
                "an image without alt text is reported while a decorative image is exempt");

        // 5. link text, button names, code languages and table headers are validated
        List<Block> sloppy = List.of(Block.heading(1, "T"), Block.link("click here", "x"), Block.link("", "y"),
                Block.button("", ""), Block.code("print(1);", ""), Block.table(""));
        List<String> sloppyIssues = audit(sloppy);
        check(sloppyIssues.size() == 5 && sloppyIssues.get(0).contains("vague link text")
                        && sloppyIssues.get(1).contains("link without text")
                        && sloppyIssues.get(2).contains("accessible name")
                        && sloppyIssues.get(3).contains("without a language")
                        && sloppyIssues.get(4).contains("without a header row"),
                "vague links, unnamed buttons, language-less code and header-less tables are reported");

        // 6. the outline follows document order
        List<String> lines = outline(clean);
        boolean ordered = lines.size() == clean.size() && lines.get(0).equals("h1: Two pointers")
                && lines.get(2).startsWith("img: Two pointers") && lines.get(6).equals("a: two pointers editorial")
                && lines.get(7).equals("table: case | result");
        check(ordered, "the screen-reader outline reproduces the document order of every block");

        // 7. boundaries: empty document and a single paragraph
        check(audit(List.of()).isEmpty() && outline(List.of()).isEmpty()
                        && audit(List.of(Block.paragraph("just text"))).isEmpty()
                        && outline(List.of(Block.paragraph("just text"))).equals(List.of("p: just text"))
                        && audit(List.of(Block.tintedParagraph("low contrast", LIGHT))).size() == 1,
                "empty and paragraph-only documents are handled without false positives");

        // 8. independent oracle: planted defects and a table-backed luminance implementation
        boolean oracleOk = true;
        for (int i = 0; i < 20; i++) {
            long state = splitmix64(7000 + i);
            List<Block> document = new ArrayList<>();
            document.add(Block.heading(1, "Lesson " + i));
            int planted = 0;
            int count = 3 + (int) Math.floorMod(state, 4);
            for (int j = 0; j < count; j++) {
                state = splitmix64(state + j * 7);
                switch ((int) Math.floorMod(state, 5)) {
                    case 0: document.add(Block.paragraph("body " + j)); break;
                    case 1: document.add(Block.image("")); planted++; break;
                    case 2: document.add(Block.image("alt " + j)); break;
                    case 3: document.add(Block.link("here", "u" + j)); planted++; break;
                    default: document.add(Block.code("x = " + j, "java")); break;
                }
            }
            oracleOk &= audit(document).size() == planted;
        }
        for (int i = 0; i < 40; i++) {
            int first = (int) (0x808080 + i * 0x000101);
            oracleOk &= Math.abs(contrast(first, WHITE) - contrastOracle(first, WHITE)) < 1e-12;
        }
        check(oracleOk, "20 generated lessons report exactly their planted defects and the luminance table agrees");

        System.out.println("clean lesson issues: " + audit(clean).size() + ", outline lines: " + lines.size());
        System.out.println("contrast: ink " + String.format("%.2f", contrast(INK, PAPER))
                + ":1, grey " + String.format("%.2f", contrast(GREY, PAPER))
                + ":1, light " + String.format("%.2f", contrast(LIGHT, PAPER)) + ":1");
        System.out.println("planted-defect documents checked: 20");
        System.out.println("PASS");
    }
}
