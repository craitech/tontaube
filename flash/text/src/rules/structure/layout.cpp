// Layout: lines and what their starts and ends become. Heading and quote markers, bullets, numbered items and
// check boxes, dividers and code fences, tables and label-value rows, the period after a heading, lines broken
// inside a sentence, and the spaces and tabs between tokens.
#include "rules/structure/common.h"

namespace {

using namespace st;

enum Kind2 : uint8_t { PLAIN, EMPTY, DELETED, TABLE, COMMENT };

struct Info {
    Line l;
    size_t first = 0, last = 0;  // first and last visible tokens; first > last when nothing is visible
    size_t content = 0;          // first token after the line-start markers
    Kind2 kind = PLAIN;
    bool md_heading = false, html_heading = false, list_item = false, quote = false, separators = false;
    bool ends_punct = false, image_end = false, colon = false, all_caps = false, lower_start = false;
    bool has_word = false, reference = false, arrows = false;
    int words = 0;
    bool section = false;  // starts with a section number: "2.1 Study design"
    bool emoji_end = false;
    bool dots = false;      // has middle-dot separators
    bool md_divider = false;  // a Markdown table divider row
    bool break_end = false; // ends with a <br> tag
    bool title_start = false;
    bool bold = false;      // the whole line is in bold
    bool period = false, join = false, join_hyphen = false;
};

bool sp(const Doc& d, size_t i) { return d[i].space_before; }

// The index of the '<' that opens a tag ending at token e (a '>'), or npos.
size_t tag_ending_at(const Doc& d, size_t a, size_t e) {
    for (size_t k = e; k-- > a && e - k < 70;) {
        if (p(d, k, '<') || p(d, k, '[')) return tag_at(d, k) == e ? k : Doc::npos;
        if (d[k].newline_before) break;
    }
    return Doc::npos;
}

// The first token of "![](src)" ending at the ')' e, or npos.
size_t empty_image_ending_at(const Doc& d, size_t a, size_t e) {
    if (!p(d, e, ')')) return Doc::npos;
    size_t k = e;
    while (k > a && !p(d, k, '(')) --k;
    if (k < a + 3 || !p(d, k, '(') || !p(d, k - 1, ']') || !p(d, k - 2, '[') || !p(d, k - 3, '!')) return Doc::npos;
    return k - 3;
}

// The line before the one starting at token a also starts with code point c.
bool line_start_cp(const Doc& d, size_t a, uint32_t c) {
    size_t k = a;
    while (k > 0 && !d[k].newline_before) --k;  // k: start of the line of token a
    if (k == 0) return false;
    size_t j = k - 1;
    while (j > 0 && !d[j].newline_before) --j;
    return p(d, j, c);
}

// A symbol in quotes of its own ("a '|' symbol"): named, not markup.
bool quoted_symbol(const Doc& d, size_t k) {
    if (k == 0 || k + 1 >= d.size() || !d.joined(k) || !d.joined(k + 1)) return false;
    uint32_t a = d[k - 1].cp, b = d[k + 1].cp;
    return is_punct(d, k - 1) && is_punct(d, k + 1) &&
           ((a == '\'' && b == '\'') || (a == '"' && b == '"') || (a == 0x2018 && b == 0x2019) ||
            (a == 0x201C && b == 0x201D) || (a == '`' && b == '`'));
}

// The number of whitespace characters before token k (a thin space is one character of three bytes).
uint32_t gap_chars(const Doc& d, size_t k) {
    uint32_t n = 0;
    for (uint32_t q = d[k - 1].end; q < d[k].start; ++q) n += (static_cast<unsigned char>(d.text[q]) & 0xC0) != 0x80;
    return n;
}

bool has_pipe(const Doc& d, size_t a, size_t b) {
    for (size_t k = a; k < b; ++k)
        if (p(d, k, '|')) return true;
    return false;
}

// ① to ⑳, ❶ to ❿, ➀ to ➉, ➊ to ➓: the number, 0 for anything else.
int circled_number(uint32_t cp) {
    if (cp >= 0x2460 && cp <= 0x2473) return static_cast<int>(cp - 0x2460 + 1);
    if (cp >= 0x2776 && cp <= 0x277F) return static_cast<int>(cp - 0x2776 + 1);
    if (cp >= 0x2780 && cp <= 0x2789) return static_cast<int>(cp - 0x2780 + 1);
    if (cp >= 0x278A && cp <= 0x2793) return static_cast<int>(cp - 0x278A + 1);
    return 0;
}

bool is_bullet(uint32_t cp) {
    return cp == '-' || cp == '*' || cp == '+' || cp == 0x2022 || cp == 0x25AA || cp == 0x25E6 || cp == 0x2023 || cp == 0x2013 ||
           cp == 0x25CF || cp == 0x25CB || cp == 0x2043 || cp == 0x2219;
}

bool divider_char(uint32_t cp) {
    return cp == '-' || cp == '*' || cp == '_' || cp == '=' || cp == '~' || cp == 0x2014 || cp == 0x2013 ||
           cp == 0x2500 || cp == 0x2550 || cp == 0x2501;
}

// Words that number a part of a document ("Section 4.2: Results").
const rv::WordSet SECTION_WORDS{"chapter", "section", "part", "appendix", "unit", "lesson", "module", "phase",
                                "stage", "week", "day", "book", "act", "scene", "annex", "article", "topic"};

// Words after which a sentence cannot end.
const rv::WordSet CONTINUES{
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "with", "from", "by", "on", "at", "as", "that", "see",
    "is", "are", "was", "were", "its", "their", "our", "this", "these", "those", "into", "than", "per", "via",
    "under", "over", "between", "about", "after", "before", "during", "without", "which", "whose", "has",
    "have", "had", "be", "been", "will", "would", "can", "could", "may", "might", "must", "should", "do",
    "does", "did", "if", "when", "while", "because", "since", "whether", "both", "either", "each", "every",
    "such", "very", "his", "her", "my", "your", "nor", "but", "so", "then", "where", "how", "what", "who"};

const rv::WordSet SMALL{"a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with", "at", "by", "as",
                        "vs", "nor", "but", "via", "from"};
bool has_vowel(std::string_view w) {
    for (char c : w)
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y') return true;
    return false;
}

// Title case for one all-caps word of a heading; "" when it stays.
std::string title_word(const Doc& d, size_t k, bool first) {
    std::string w(d.low(k));
    if (w == "a" && !first) return w;
    if (d.str(k).size() < 2 || initialism(w) || !has_vowel(w)) return "";
    if (!first && SMALL.contains(w)) return w;
    return rv::capital(w);
}

struct Layout {
    const Doc& d;
    Out& out;
    std::vector<Info> info;
    bool markdown = false;  // the text has Markdown headings or bold markers

    Layout(const Doc& doc, Out& o) : d(doc), out(o) {
        markdown = has(d, "**") || has(d, "\n# ") || has(d, "\n## ") || has(d, "\n### ") || d.text.rfind("# ", 0) == 0 ||
                   d.text.rfind("## ", 0) == 0 || d.text.rfind("### ", 0) == 0;
    }

    void analyse(Info& f);
    bool value_list(const Info& f) const;
    void tables();
    void decide();
    void emit_line(size_t li);
    void line_end(size_t li);
    void gaps();

    // the next / previous line that is not a comment line (deleted lines count)
    size_t next_line(size_t li) const {
        for (size_t k = li + 1; k < info.size(); ++k)
            if (info[k].kind != COMMENT) return k;
        return Doc::npos;
    }
    size_t prev_line(size_t li) const {
        for (size_t k = li; k-- > 0;)
            if (info[k].kind != COMMENT && info[k].kind != DELETED) return k;
        return Doc::npos;
    }
    size_t next_kept(size_t li) const {
        for (size_t k = li + 1; k < info.size(); ++k)
            if (info[k].kind != COMMENT && info[k].kind != DELETED) return k;
        return Doc::npos;
    }
};

// "- Label: 4x": a dash, one to three words, a colon, then a number (or ".4").
bool label_number_item(const Doc& d, const Line& l) {
    size_t a = l.a, k = a + 1;
    if (!p(d, a, '-') || k >= l.b || !sp(d, k)) return false;
    while (k < l.b && k - a <= 5 &&
           (d.word(k) || (p(d, k, '-') && d.joined(k) && d.word(k - 1) && d.word(k + 1) && d.joined(k + 1))))
        ++k;  // one to three words, "Sign-ups" counts as one
    if (k == a + 1 || !p(d, k, ':') || !d.joined(k) || k + 1 >= l.b) return false;
    return d.number(k + 1) || (p(d, k + 1, '.') && d.number(k + 2) && d.joined(k + 2));
}

// The line is in a list of "- Label: number" items under a line that ends with a colon; the labels keep the
// dashes of such short value lists ("Results:\n- Compile: 4x\n- Link: 7x").
bool Layout::value_list(const Info& f) const {
    size_t li = static_cast<size_t>(&f - info.data()), s = li, e = li;
    while (s > 0 && label_number_item(d, info[s - 1].l)) --s;
    while (e + 1 < info.size() && label_number_item(d, info[e + 1].l)) ++e;
    if (s == 0 || !p(d, info[s - 1].l.b - 1, ':') || e - s + 1 > 3) return false;
    for (size_t k = s; k <= e; ++k)
        if (!label_number_item(d, info[k].l)) return false;
    return true;
}

void Layout::analyse(Info& f) {
    size_t a = f.l.a, b = f.l.b;  // [a, b)
    // whole-line forms
    {
        bool div = b - a >= 3, fence = false;
        for (size_t k = a; k < b && div; ++k) div = is_punct(d, k) && divider_char(d[k].cp) && d[k].cp == d[a].cp;
        if (b - a >= 3 && (p(d, a, '`') || p(d, a, '~')) && p(d, a + 1, d[a].cp) && p(d, a + 2, d[a].cp) &&
            d.joined(a + 1) && d.joined(a + 2))
            fence = true;
        bool md_div = !div && !fence;  // a table divider row: | - : only, three dashes or more
        int dashes = 0;
        for (size_t k = a; k < b && md_div; ++k) {
            md_div = p(d, k, '|') || p(d, k, '-') || p(d, k, ':') || p(d, k, '=');
            dashes += p(d, k, '-');
        }
        md_div = md_div && dashes >= 3 && (p(d, a, '|') || p(d, b - 1, '|') || has_pipe(d, a, b));
        // an empty table cell among other lines: a line of one or two dashes, a lone question mark, an arrow "-->"
        bool cell = b - a <= 3 && d.size() > b - a, arrow = false;
        for (size_t k = a; k < b && cell; ++k) {
            arrow |= p(d, k, '>') || p(d, k, '<');
            cell = is_punct(d, k) && (d[k].cp == '-' || d[k].cp == 0x2013 || d[k].cp == 0x2014 || d[k].cp == 0x2015 ||
                                      (d[k].cp == '?' && b - a == 1) || d[k].cp == '>' || d[k].cp == '<');
        }
        cell = cell && (b - a <= 2 || arrow);
        // the edit link after a heading of a wiki page: "edit", "[edit]", "edit source"
        size_t w = a + (p(d, a, '[') ? 1 : 0);
        if (d.size() > b - a && w < b && d.str(w) == "edit") {
            size_t z = w + 1 + (d.is(w + 1, "source") ? 1 : 0);
            cell |= (z == b && w == a) || (w > a && z + 1 == b && p(d, z, ']'));
        }
        if (div || fence || md_div || cell) {
            f.kind = DELETED;
            f.md_divider = md_div;
            return;
        }
        if (p(d, a, '<') && p(d, a + 1, '!') && p(d, a + 2, '-') && p(d, b - 1, '>') && p(d, b - 2, '-')) {
            f.kind = COMMENT;
            return;
        }
        if (empty_image_ending_at(d, a, b - 1) == a) {  // a line that is only "![](spacer.gif)"
            f.kind = DELETED;
            return;
        }
    }
    // start: invisible characters, decorations, emoji, opening tags, quote markers, then one list marker
    size_t i = a;
    for (bool again = true; again && i < b;) {
        again = false;
        bool star = p(d, i, 0x2605) && i + 1 < b && d.word(i + 1) && sp(d, i + 1);
        if ((p(d, i, 0x2610) || p(d, i, 0x2611) || p(d, i, 0x2612)) && i + 1 < b && sp(d, i + 1) &&
            (d.word(i + 1) || d.number(i + 1))) {  // ballot boxes: an open or a ticked item
            out.tokens(i, i, p(d, i, 0x2610) ? "Open:" : "Done:", 80);
            f.list_item = true;
            ++i;
            break;
        }
        if (is_invisible(d, i) || is_emoji(d, i) || (is_punct(d, i) && (decor_cp(d[i].cp) || star) && i + 1 < b)) {
            // bullet-like symbols, and a decoration that also starts a neighbouring line, mark a list item; so
            // does an emoji when the line before or after starts with one too
            uint32_t c = d[i].cp;
            bool neighbour = (f.l.b < d.size() && p(d, f.l.b, c)) || (a > 0 && is_punct(d, a) && i == a &&
                                                                     line_start_cp(d, a, c));
            if (is_emoji(d, i) && i == a) {
                size_t pa = a;  // the first token of the line before
                if (a > 0) {
                    pa = a - 1;
                    while (pa > 0 && !d[pa].newline_before) --pa;
                }
                if ((f.l.b < d.size() && is_emoji(d, f.l.b)) || (a > 0 && is_emoji(d, pa))) f.list_item = true;
            }
            if (is_punct(d, i) && (c == 0x25AA || c == 0x25CF || c == 0x25E6 || c == 0x25CB || c == 0x2023 ||
                                   (decor_cp(c) && neighbour)))
                f.list_item = true;
            ++i;
            again = true;
        } else if (size_t e = tag_at(d, i)) {
            if (e < b) {
                if (d.word(i + 1) && d.str(i + 1).size() == 1 && (d.str(i + 1)[0] | 0x20) == 'h' && d.number(i + 2))
                    f.html_heading = true;
                i = e + 1;
                again = true;
            }
        } else if (is_punct(d, i) && (d[i].cp == '*' || d[i].cp == '_' || d[i].cp == '~') && i + 1 < b &&
                   p(d, b - 1, d[i].cp) && b - 1 > i + 1) {
            size_t e = i;  // **Scope of Coverage**: the opener of an emphasis that closes at the line end
            while (e + 1 < b && p(d, e + 1, d[i].cp) && d.joined(e + 1)) ++e;
            if (e + 1 < b && d.joined(e + 1) && d.word(e + 1)) {
                // a whole line in bold is a heading ("**Pros**" over a list)
                if (i == a && e == i + 1 && d[i].cp != '~' && b - 2 > e && p(d, b - 2, d[i].cp) && d.joined(b - 1))
                    f.bold = true;
                i = e + 1;
                again = true;
            }
        } else if (p(d, i, '>') && (i + 1 >= b || sp(d, i + 1) || p(d, i + 1, '>'))) {
            out.add(d[i].start, i + 1 < b ? d[i + 1].start : d[i].end, "", 80);
            f.quote = true;
            ++i;
            again = true;
        }
    }
    f.first = i;
    if (i > a && i < b && !f.quote && d[i].space_before) {
        bool tags_only = true;
        for (size_t k = a; k < i && tags_only; ++k) tags_only = !is_emoji(d, k) && !(is_punct(d, k) && decor_cp(d[k].cp));
        if (tags_only) out.add(d[a].start, d[i].start, "", 93);
    }
    // "o First item" over "o Second item": the letter o as a bullet (pasted from a word processor)
    if (i + 1 < b && i == a && d.str(i) == "o" && sp(d, i + 1) && d.word(i + 1) && d[i + 1].casing != rv::LOWER) {
        size_t pa = a;
        if (a > 0) {
            pa = a - 1;
            while (pa > 0 && !d[pa].newline_before) --pa;
        }
        bool next_o = f.l.b + 1 < d.size() && d.str(f.l.b) == "o" && sp(d, f.l.b + 1) && !d[f.l.b + 1].newline_before;
        bool prev_o = a > 0 && d.str(pa) == "o" && pa + 1 < a && sp(d, pa + 1);
        if (next_o || prev_o) {
            out.add(d[i].start, d[i + 1].start, "", 75);
            f.list_item = true;
            ++i;
        }
    }
    if (i < b && p(d, i, '#')) {  // ## Heading
        size_t e = i;
        while (e + 1 < b && p(d, e + 1, '#') && d.joined(e + 1)) ++e;
        if (e - i < 6 && e + 1 < b && sp(d, e + 1)) {
            out.add(d[i].start, d[e + 1].start, "", 80);
            f.md_heading = true;
            i = e + 1;
        }
    } else if (i + 2 < b && p(d, i, '=') && p(d, i + 1, '=') && d.joined(i + 1)) {  // == Heading ==
        size_t e = i + 1;
        while (e + 1 < b && p(d, e + 1, '=') && d.joined(e + 1)) ++e;
        size_t z = b - 1;
        if (p(d, z, '=') && e + 1 < b && sp(d, e + 1)) {
            out.add(d[i].start, d[e + 1].start, "", 80);
            f.md_heading = true;
            i = e + 1;
        }
    } else if (i + 1 < b && is_punct(d, i) && is_bullet(d[i].cp) &&
               (sp(d, i + 1) || d[i].cp == 0x2022 || d[i].cp == 0x25AA) &&
               !(d[i].cp == '-' && p(d, i + 1, '-'))) {
        bool emph = false;  // "* text *" is not a bullet when another marker closes it
        if (d[i].cp == '*' && !sp(d, i + 1))
            for (size_t k = i + 1; k < b && !emph; ++k) emph = p(d, k, '*');
        if (!emph) {
            if (!(i == a && d[i].cp == '-' && value_list(f))) out.add(d[i].start, d[i + 1].start, "", 75);
            f.list_item = true;
            ++i;
        }
    } else if (i + 1 < b && p(d, i, '*') && d.joined(i + 1) && d.word(i + 1)) {  // *Footnote text
        bool closed = false;
        for (size_t k = i + 1; k < b && !closed; ++k) closed = p(d, k, '*');
        if (!closed) {
            out.add(d[i].start, d[i + 1].start, "", 75);
            f.list_item = true;
            ++i;
        }
    }
    // a check box, also after a bullet
    if (i + 2 < b && p(d, i, '[') && ((p(d, i + 2, ']') && (d.is(i + 1, "x")) && d.joined(i + 1) && d.joined(i + 2)) ||
                                      (p(d, i + 1, ']') && sp(d, i + 1) && d[i + 1].start == d[i].end + 1))) {
        size_t e = p(d, i + 1, ']') ? i + 1 : i + 2;
        out.tokens(i, e, e == i + 1 ? "Open:" : "Done:", 80);
        f.list_item = true;
        i = e + 1;
    } else if (!f.md_heading && !f.list_item && i + 2 < b) {
        // numbered and lettered items: "1." "1)" "1-" "(a)" "a)" "ii."
        size_t e = 0;
        std::string spoken;
        if (d.number(i) && d.str(i).size() <= 3 && (p(d, i + 1, '.') || p(d, i + 1, ')') || p(d, i + 1, '-')) &&
            d.joined(i + 1) && sp(d, i + 2) && !d.number(i + 2)) {
            e = i + 1;
            spoken = cap_cardinal(std::stoull(std::string(d.str(i)))) + ",";
        } else if (p(d, i, '(') && d.word(i + 1) && d.str(i + 1).size() == 1 && p(d, i + 2, ')') &&
                   d.joined(i + 1) && d.joined(i + 2) && i + 3 < b && sp(d, i + 3)) {
            e = i + 2;
            spoken = std::string(1, static_cast<char>(d.str(i + 1)[0] & ~0x20)) + ",";
        } else if (d.word(i) && d.str(i).size() == 1 && d[i].casing == rv::LOWER && (p(d, i + 1, ')') || p(d, i + 1, '.')) &&
                   d.joined(i + 1) && sp(d, i + 2)) {
            e = i + 1;
            spoken = std::string(1, static_cast<char>(d.str(i)[0] & ~0x20)) + ",";
        } else if (d.word(i) && d.str(i).size() == 1 && d[i].casing == rv::UPPER && d.str(i)[0] >= 'A' && d.str(i)[0] <= 'H' &&
                   p(d, i + 1, ')') && d.joined(i + 1) && sp(d, i + 2) && d.word(i + 2)) {
            e = i + 1;  // "A) Keep the plan."
            spoken = std::string(d.str(i)) + ",";
        } else if (circled_number(d[i].cp) && is_punct(d, i) && sp(d, i + 1) && (d.word(i + 1) || d.number(i + 1))) {
            e = i;  // "① Open the box."
            spoken = cap_cardinal(static_cast<uint64_t>(circled_number(d[i].cp))) + ",";
        } else if (d.word(i) && (p(d, i + 1, '.') || p(d, i + 1, ')')) && d.joined(i + 1) && sp(d, i + 2) &&
                   d.word(i + 2)) {
            int v = rv::roman_value(d.str(i));
            bool upper = d[i].casing == rv::UPPER;
            // "I." and "i." count only when the next line goes on with "II." / "ii."
            bool one = (d.str(i) == "I" || d.str(i) == "i") && f.l.b < d.size() && d.str(f.l.b) == (upper ? "II" : "ii") &&
                       p(d, f.l.b + 1, d[i + 1].cp);
            if (((v >= 2 && v <= 30 && d.str(i).size() >= 2) || one) && (upper || d[i].casing == rv::LOWER)) {
                e = i + 1;
                spoken = cap_cardinal(static_cast<uint64_t>(v)) + ",";
            }
        }
        if (!spoken.empty()) {
            // in a Markdown document a list number is markup like a bullet; elsewhere it is read
            if (markdown && d.number(i)) out.add(d[i].start, d[e + 1].start, "", 50);
            else out.tokens(i, e, spoken, 50);
            f.list_item = true;
            i = e + 1;
        }
    }
    // Q: and A: labels
    if (i + 1 < b && d.word(i) && d.str(i).size() == 1 && (d.str(i) == "Q" || d.str(i) == "A") && p(d, i + 1, ':') &&
        d.joined(i + 1)) {
        out.tokens(i, i, d.str(i) == "Q" ? "Question" : "Answer", 80);
    }
    f.content = i;
    // end: invisible characters, emoji, closing tags, emphasis closers, closing hashes, empty images
    size_t z = b;  // one past the last visible token
    for (bool again = true; again && z > i;) {
        again = false;
        size_t t = z - 1;
        bool value = t > i && (p(d, t - 1, ':') || p(d, t - 1, '|')) &&
                     (p(d, t, 0x2705) || p(d, t, 0x274C) || p(d, t, 0x2716) || p(d, t, 0x274E));  // "Status: ✅"
        if ((is_invisible(d, t) || is_emoji(d, t)) && !value) {
            --z;
            again = true;
        } else if (p(d, t, '>') || p(d, t, ']')) {
            size_t s = tag_ending_at(d, i, t);
            if (s != Doc::npos) {
                if (d.is(s + 1, "br")) f.break_end = true;
                z = s;
                again = true;
            }
        } else if (is_punct(d, t) && (d[t].cp == '*' || d[t].cp == '_' || d[t].cp == '~') && d.joined(t) &&
                   t > i && !(d[t].cp == '*' && d.number(t - 1))) {
            size_t s = t;
            while (s > i && p(d, s - 1, d[t].cp) && d.joined(s)) --s;
            bool opener = false;  // the closer of an emphasis opened earlier on the line, not a mask (f***)
            for (size_t k = i; k + 1 < s && !opener; ++k)
                opener = p(d, k, d[t].cp) && (k == i || sp(d, k) || is_punct(d, k - 1)) && d.joined(k + 1) &&
                         !p(d, k + 1, d[t].cp) && d[k + 1].kind != Kind::Punct;
            if (opener || (s > i && !d.word(s - 1))) {
                z = s;
                again = true;
            }
        } else if ((p(d, t, '#') && f.md_heading) || (p(d, t, '=') && f.md_heading)) {
            while (z > i && is_punct(d, z - 1) && d[z - 1].cp == d[t].cp) --z;
            again = true;
        } else if (p(d, t, ')')) {
            size_t s = empty_image_ending_at(d, i, t);
            if (s != Doc::npos) {
                z = s;
                again = true;
            }
        }
    }
    if (z <= i) {
        f.first = 1;
        f.last = 0;
        f.kind = EMPTY;
        return;
    }
    f.last = z - 1;
    for (size_t k = z; k < b && !f.emoji_end; ++k) f.emoji_end = is_emoji(d, k);
    // content features
    uint32_t lc = is_punct(d, f.last) ? d[f.last].cp : 0;
    // a closing quote or bracket, or a footnote mark that goes ("incidents.†"), may follow the sentence end
    bool after_end = lc == '"' || lc == 0x201D || lc == ')' || lc == '\'' || lc == 0x2019 || lc == 0x2020 ||
                     lc == 0x2021 || lc == '*' || lc == 0xB9 || lc == 0xB2 || lc == 0xB3 || (lc >= 0x2074 && lc <= 0x2079);
    f.ends_punct = lc && (terminal_cp(lc) || (after_end && f.last > i && is_punct(d, f.last - 1) &&
                                              terminal_cp(d[f.last - 1].cp)));
    if (d[f.last].cp == 0xFDD0) {  // a marked formula whose source ends with a mark ("x = 1.")
        std::string_view s = d.str(f.last);
        size_t e = s.size() >= 6 && s.substr(s.size() - 3) == "\xEF\xB7\x91" ? s.size() - 3 : s.size();
        while (e > 3 && s[e - 1] == ' ') --e;
        f.ends_punct = e > 3 && (s[e - 1] == '.' || s[e - 1] == ',' || s[e - 1] == ';' || s[e - 1] == ':');
    }
    if (p(d, f.last, ')')) {  // ![alt](src) reads "Image: alt."
        size_t k = f.last;
        while (k > i && !p(d, k, '(')) --k;
        for (size_t m = k; m > i; --m)
            if (p(d, m, '[')) {
                if (m > i && p(d, m - 1, '!') && p(d, k - 1, ']')) f.image_end = f.ends_punct = true;
                break;
            }
    }
    size_t sec_end = i;  // last token of a leading section number "2.1"
    if (d.number(i)) {
        bool ok = d.str(i).size() <= 2;
        while (ok && p(d, sec_end + 1, '.') && d.number(sec_end + 2) && d.joined(sec_end + 1) &&
               d.joined(sec_end + 2)) {
            sec_end += 2;
            ok = d.str(sec_end).size() <= 2;
        }
        if (!ok || sec_end + 1 > f.last || !sp(d, sec_end + 1)) sec_end = i - 1;
    } else {
        sec_end = i - 1;
    }
    bool all_caps = true, any_word = false, dots = false, digits = false;
    int caps_words = 0;
    bool real_word = false, long_word = false;
    for (size_t k = i; k <= f.last; ++k) {
        const rv::Token& t = d[k];
        if (t.kind == Kind::Word) {
            f.has_word = true;
            ++f.words;
            any_word = true;
            if (t.casing != rv::UPPER) all_caps = false;
            else if (t.end - t.start >= 2) {
                ++caps_words;
                std::string_view w = d.low(k);
                if (w.size() >= 3 && has_vowel(w) && !initialism(w)) {
                    real_word = true;
                    long_word |= w.size() >= 5;
                }
            }
            continue;
        }
        if (t.kind == Kind::Number) {
            if (k > sec_end || sec_end + 1 == i) digits = true;
            continue;
        }
        if (p(d, k, ':') && k > i && !(d.number(k - 1) && d.number(k + 1) && d.joined(k + 1))) f.colon = true;
        if (p(d, k, 0x2192) || (p(d, k, '>') && k > i && (p(d, k - 1, '-') || (sp(d, k) && k < f.last && sp(d, k + 1)))))
            f.arrows = true;
        if ((p(d, k, 0xB7) || (p(d, k, 0x2022) && k > i)) && sp(d, k) && k + 1 <= f.last && sp(d, k + 1) &&
            !(is_punct(d, k - 1) && terminal_cp(d[k - 1].cp) && d[k - 1].cp != ',')) {
            size_t v = k - 1;
            while (v > i && (is_emoji(d, v) || is_invisible(d, v))) --v;
            out.add(d[v].end, d[k + 1].start, ", ", 60);
            dots = true;
        }
    }
    if (dots && f.colon) f.separators = true;
    f.dots = dots;
    // label-value rows separated by ';': "Workers: 12 ; Jobs: 480"
    if (f.colon)
        for (size_t k = i + 1; k + 1 <= f.last; ++k) {
            if (!p(d, k, ';') || !sp(d, k + 1)) continue;
            bool label_after = false;
            for (size_t m = k + 1; m <= f.last && m < k + 6 && !label_after; ++m)
                label_after = p(d, m, ':') && d.joined(m) && !(d.number(m - 1) && d.number(m + 1) && d.joined(m + 1));
            if (!label_after) continue;
            out.add(d[k - 1].end, d[k + 1].start, ", ", 60);
            f.separators = true;
        }
    // "RISK REGISTER", "Q2 FY2025 RESULTS", "LIMITATIONS"; not "NASA" or "API 2"
    f.all_caps = any_word && all_caps && real_word && (caps_words >= 2 || long_word) && (!digits || f.words >= 2 || long_word);
    f.title_start = (d.word(i) && d[i].casing != rv::LOWER) || p(d, i, '"') || p(d, i, 0x201C);
    if (sec_end + 1 != i && !f.list_item) {
        size_t k = sec_end;
        f.section = d.word(k + 1) && d[k + 1].casing != rv::LOWER && f.words <= 8 && !f.colon;
    }
    // "1st Pass Results": an ordinal start counts as a capitalised start
    if (d.number(i) && d.word(i + 1) && d.joined(i + 1) && d[i + 1].casing == rv::LOWER && d.str(i + 1).size() == 2 &&
        i + 2 <= f.last && d.word(i + 2) && d[i + 2].casing != rv::LOWER)
        f.title_start = true;
    f.lower_start = d.word(i) && d[i].casing == rv::LOWER;
    f.reference = p(d, i, '[') && d.number(i + 1) && p(d, i + 2, ']');
}

// A table: consecutive rows with cells separated by '|' or tabs. With a header row (a divider under it, or a
// first row without digits over rows with digits) each row reads "H1 v1: H2 v2, H3 v3."; without one,
// "v1, v2, v3." A header row is never spoken on its own.
struct Cell {
    size_t a, b;  // tokens a..b inclusive; a > b when empty
};

// Units in a column header ("Rain (mm)", "Speed (km/h)").
const rv::PhraseMap HEADER_UNITS{
    {"%", "percent"}, {"$", "dollars"}, {"\xE2\x82\xAC", "euros"}, {"\xC2\xA3", "pounds"}, {"usd", "US dollars"},
    {"eur", "euros"}, {"gbp", "pounds"}, {"mm", "millimeters"}, {"cm", "centimeters"}, {"m", "meters"},
    {"km", "kilometers"}, {"kg", "kilograms"}, {"g", "grams"}, {"mg", "milligrams"}, {"lb", "pounds"},
    {"lbs", "pounds"}, {"oz", "ounces"}, {"ms", "milliseconds"}, {"s", "seconds"}, {"sec", "seconds"},
    {"min", "minutes"}, {"h", "hours"}, {"hrs", "hours"}, {"kw", "kilowatts"}, {"kwh", "kilowatt hours"},
    {"mw", "megawatts"}, {"w", "watts"}, {"gb", "gigabytes"}, {"mb", "megabytes"}, {"tb", "terabytes"},
    {"hz", "hertz"}, {"khz", "kilohertz"}, {"mhz", "megahertz"}, {"ghz", "gigahertz"}, {"db", "decibels"},
    {"km/h", "kilometers per hour"}, {"mph", "miles per hour"}, {"m/s", "meters per second"},
    {"\xC2\xB0" "c", "degrees Celsius"}, {"\xC2\xB0" "f", "degrees Fahrenheit"}, {"ppm", "parts per million"},
    {"l", "liters"}, {"ml", "milliliters"}, {"ft", "feet"}, {"in", "inches"}, {"mi", "miles"}};

// The words of a column header, which the reading of every row repeats: words as written, numbers and codes read
// ("P95" -> "P ninety-five", "Q1" -> "Q one"), a unit in brackets in full ("Rain (mm)" -> "Rain (millimeters)").
std::string header_words(const Doc& d, size_t a, size_t b) {
    std::string out;
    for (size_t k = a; k <= b; ++k) {
        if (k > a && d[k].space_before) out += ' ';
        if (p(d, k, '(') && k + 2 <= b) {  // (unit)
            size_t e = k + 1;
            while (e <= b && !p(d, e, ')')) ++e;
            if (e <= b && e > k + 1) {
                const std::string_view* u = HEADER_UNITS.find(lower(d.slice(d[k + 1].start, d[e - 1].end)));
                if (u) {
                    out += "(" + std::string(*u) + ")";
                    k = e;
                    continue;
                }
            }
        }
        if (d.number(k)) {
            std::string w = rv::read_number(d.str(k));
            if (w.empty()) w = rv::digits(d.str(k));
            if (k > a && d.word(k - 1) && d.joined(k)) out += ' ';  // P95 -> P ninety-five
            out += w;
        } else if (p(d, k, '%')) {
            out += (k > a && d.number(k - 1) ? " percent" : "percent");
        } else if (p(d, k, '&')) {
            out += "and";
        } else if (p(d, k, '#') && !(k < b && d.joined(k + 1) && d.word(k + 1))) {
            out += "number";
        } else if (d.word(k) && d[k].casing == rv::UPPER && d.str(k).size() >= 3 && has_vowel(d.low(k)) &&
                   !initialism(d.low(k))) {
            out += rv::capital(lower(d.str(k)));  // NAME -> Name
        } else {
            out += std::string(d.str(k));
        }
    }
    return out;
}

void Layout::tables() {
    size_t nl = info.size();
    // cells of each line; empty when the line is not a row
    std::vector<std::vector<Cell>> cells(nl);
    std::vector<char> divider(nl, 0);
    for (size_t li = 0; li < nl; ++li) {
        Info& f = info[li];
        if (f.md_divider) divider[li] = 1;
        if (f.kind != PLAIN) continue;
        size_t a = f.l.a, b = f.l.b;
        bool pipes = false, tabs = false;
        int wiki = 0;
        for (size_t k = a; k < b; ++k) {
            if (p(d, k, '[') && p(d, k + 1, '[')) ++wiki;
            if (p(d, k, ']') && p(d, k + 1, ']') && wiki > 0) --wiki;
            if (p(d, k, '|') && !wiki && !(p(d, k + 1, '|') && d.joined(k + 1)) && !(k > a && p(d, k - 1, '|') && d.joined(k)) &&
                !quoted_symbol(d, k) &&
                !(p(d, k + 1, '>') && d.joined(k + 1)) && !(k > a && p(d, k - 1, '<') && d.joined(k)))
                pipes = true;
            if (k > a && d.text.substr(d[k - 1].end, d[k].start - d[k - 1].end).find('\t') != std::string_view::npos)
                tabs = true;
        }
        // a row of values set apart by runs of spaces: "6  1.00  0  30.1s"
        bool wide = false;
        if (!pipes && !tabs && b - a >= 3 && b - a <= 24) {
            int cells_n = 1, numeric = d.number(a) || (is_punct(d, a) && d.number(a + 1)) ? 1 : 0, words_in = 0, longest = 0;
            bool any_gap = false;
            for (size_t k = a + 1; k < b; ++k) {
                uint32_t g = gap_chars(d, k);
                if (g >= 2) {
                    any_gap = true;
                    ++cells_n;
                    longest = std::max(longest, words_in);
                    words_in = 0;
                    numeric += d.number(k) || (is_punct(d, k) && d.number(k + 1) && d.joined(k + 1));
                } else if (g == 1) {
                    ++words_in;
                }
            }
            longest = std::max(longest, words_in);
            wide = any_gap && cells_n >= 3 && cells_n <= 8 && 2 * numeric >= cells_n + 1 && longest <= 1;
        }
        if (!pipes && !tabs && !wide) continue;
        std::vector<Cell> row;
        auto tab_before = [&](size_t k) {
            if (k <= a) return false;
            if (wide) return gap_chars(d, k) >= 2;
            return d.text.substr(d[k - 1].end, d[k].start - d[k - 1].end).find('\t') != std::string_view::npos;
        };
        size_t s = a, end = b;
        if (pipes && p(d, a, '|')) s = a + 1;
        if (pipes && b - 1 > a && p(d, b - 1, '|')) end = b - 1;
        for (size_t k = s; k < end; ++k) {
            if (pipes ? (p(d, k, '|') && !(p(d, k + 1, '|') && d.joined(k + 1))) : (k > s && tab_before(k))) {
                row.push_back({s, k});
                s = pipes ? k + 1 : k;
            }
        }
        row.push_back({s, end});
        for (Cell& c : row) {  // [a, b) -> inclusive, a > b when empty; trailing emoji are not content
            while (c.b > c.a && ((is_emoji(d, c.b - 1) && !(c.b - 1 == c.a && (p(d, c.a, 0x2705) || p(d, c.a, 0x274C)))) ||
                                 is_invisible(d, c.b - 1)))
                --c.b;
            if (c.a < c.b) c.b -= 1;
            else c = {1, 0};
        }
        size_t cells_ok = 0, longest = 0;
        for (const Cell& c : row)
            if (c.a <= c.b) {
                ++cells_ok;
                longest = std::max(longest, c.b - c.a + 1);
            }
        if (row.size() < 2 || cells_ok < 2) continue;
        if (!pipes) {  // tabs in prose are spacing, not cells; neither are "Label:<tab>value" lines
            if (row[0].a <= row[0].b && p(d, row[0].b, ':')) continue;
            size_t z = b - 1;
            if (is_punct(d, z) && (d[z].cp == '.' || d[z].cp == '!' || d[z].cp == '?')) continue;
            if (longest > 8 || row.size() < 3) {
                bool neighbour = false;
                for (size_t m : {li - 1, li + 1})
                    if (m < nl && info[m].kind == PLAIN) {
                        size_t ma = info[m].l.a, mb = info[m].l.b;
                        for (size_t k = ma + 1; k < mb && !neighbour; ++k)
                            neighbour = d.text.substr(d[k - 1].end, d[k].start - d[k - 1].end).find('\t') !=
                                        std::string_view::npos;
                    }
                if (longest > 8 || !neighbour) continue;
            }
        }
        cells[li] = row;
    }
    // blocks
    for (size_t li = 0; li < nl;) {
        if (cells[li].empty()) {
            ++li;
            continue;
        }
        std::vector<size_t> rows{li};
        size_t div_line = Doc::npos;
        size_t k = li + 1;
        while (k < nl && info[k].l.blank_before == 0 && (!cells[k].empty() || (divider[k] && div_line == Doc::npos && rows.size() == 1))) {
            if (divider[k]) div_line = k;
            else rows.push_back(k);
            ++k;
        }
        bool header = false;
        if (rows.size() >= 2) {
            if (div_line != Doc::npos) {
                header = true;
            } else {
                // a header row: every cell starts with a letter and has no colon, over rows with a cell that
                // starts with a digit, or with a lower-case value under short capitalised headers
                bool first_words = true, first_colon = false, short_titles = true, later_digit = false,
                     later_lower = false;
                for (const Cell& c : cells[rows[0]]) {
                    if (c.a > c.b) continue;
                    first_words &= d.word(c.a);
                    short_titles &= d.word(c.a) && d[c.a].casing != rv::LOWER && c.b - c.a <= 2;
                    for (size_t m = c.a; m <= c.b; ++m) first_colon |= p(d, m, ':');
                }
                // a column of values: digits in at least half of the rows below
                std::vector<int> digit_rows(cells[rows[0]].size() + 8, 0);
                for (size_t r = 1; r < rows.size(); ++r)
                    for (size_t ci = 0; ci < cells[rows[r]].size(); ++ci) {
                        const Cell& c = cells[rows[r]][ci];
                        if (c.a > c.b) continue;
                        bool dg = d.number(c.a) || (is_punct(d, c.a) && d.number(c.a + 1) && d.joined(c.a + 1));
                        if (dg && ci < digit_rows.size()) ++digit_rows[ci];
                        later_lower |= d.word(c.a) && d[c.a].casing == rv::LOWER;
                    }
                for (int k : digit_rows) later_digit |= k > 0 && 2 * static_cast<size_t>(k) > rows.size() - 1;
                header = first_words && !first_colon && (later_digit || (short_titles && later_lower));
            }
        }
        std::vector<std::string> heads;
        if (header)
            for (const Cell& c : cells[rows[0]])
                heads.push_back(c.a <= c.b ? header_words(d, c.a, c.b) : std::string());
        size_t r0 = header ? 1 : 0;
        if (header) info[rows[0]].kind = TABLE;
        for (size_t r = r0; r < rows.size(); ++r) {
            const std::vector<Cell>& row = cells[rows[r]];
            Info& f = info[rows[r]];
            f.kind = TABLE;
            // the start of the row
            size_t c0 = 0;
            while (c0 < row.size() && row[c0].a > row[c0].b) ++c0;
            if (c0 == row.size()) continue;
            std::string prefix;
            if (header && c0 < heads.size() && !heads[c0].empty()) prefix = rv::capital(heads[c0]) + " ";
            uint32_t from;
            std::string spoken;
            if (r == r0) {
                from = header ? d[info[rows[0]].l.a].start : d[f.l.a].start;
                spoken = prefix;
            } else {
                const std::vector<Cell>& prev = cells[rows[r - 1]];
                size_t lastc = prev.size();
                while (lastc > 0 && prev[lastc - 1].a > prev[lastc - 1].b) --lastc;
                from = d[prev[lastc - 1].b].end;
                spoken = is_punct(d, prev[lastc - 1].b) && terminal_cp(d[prev[lastc - 1].b].cp) ? "" : ".";
                for (uint32_t q = from; q < d[row[c0].a].start; ++q)
                    if (d.text[q] == '\n') spoken += '\n';
                spoken += prefix;
            }
            if (from < d[row[c0].a].start) out.add(from, d[row[c0].a].start, spoken, 93);
            // separators between cells
            bool first_pair = true;
            for (size_t c = c0 + 1; c < row.size(); ++c) {
                if (row[c].a > row[c].b) continue;
                size_t pc = c - 1;
                while (row[pc].a > row[pc].b) --pc;
                std::string s = header ? (first_pair ? ": " : ", ") : ", ";
                if (header && c < heads.size() && !heads[c].empty()) s += heads[c] + " ";
                first_pair = false;
                out.add(d[row[pc].b].end, d[row[c].a].start, s, 93);
            }
            // the end of the last row (the rows between are closed by the next row's start)
            if (r + 1 == rows.size()) {
                size_t lastc = row.size();
                while (lastc > 0 && row[lastc - 1].a > row[lastc - 1].b) --lastc;
                size_t v = row[lastc - 1].b;
                size_t nk = next_kept(rows[r]);
                uint32_t to = nk != Doc::npos ? d[info[nk].l.a].start : static_cast<uint32_t>(d.text.size());
                std::string end = ".";
                bool last_text = nk == Doc::npos;
                if (rows.size() == 1 && last_text) {  // a single row at the end: a period only for values
                    bool values = false;
                    for (const Cell& c : row)
                        for (size_t m = c.a; m <= c.b && c.a <= c.b; ++m) values |= d.number(m) || p(d, m, ':');
                    if (!values) end = "";
                }
                bool ends = is_punct(d, v) && terminal_cp(d[v].cp);
                if (ends) end = "";
                if (!last_text)
                    for (uint32_t q = d[v].end; q < to; ++q)
                        if (d.text[q] == '\n') end += '\n';
                if (d[v].end < to) {
                    if (end != std::string(d.slice(d[v].end, to))) out.add(d[v].end, to, end, 93);
                } else if (!end.empty()) {
                    out.add(d[v].end, d[v].end, end, 50);  // the text ends with the row
                }
            }
        }
        li = k;
    }
}

// Which lines get a period, and which lines continue on the next one.
void Layout::decide() {
    size_t nl = info.size();
    bool listing = false;  // the line is in a block under a line that ends with a colon: a list of entries
    for (size_t li = 0; li < nl; ++li) {
        Info& f = info[li];
        if (f.l.blank_before > 0) listing = false;
        if (li > 0 && info[li - 1].kind == PLAIN && info[li - 1].last >= info[li - 1].first &&
            p(d, info[li - 1].last, ':'))
            listing = true;
        if (f.kind != PLAIN) continue;
        size_t nx = next_kept(li);
        size_t pv = prev_line(li);
        bool has_next = nx != Doc::npos;
        bool blank_after = has_next && info[nx].l.blank_before > 0;
        // a word broken across lines, or a sentence that goes on in lower case
        if (has_next && !blank_after && info[nx].kind == PLAIN && !f.md_heading && !f.html_heading &&
            !f.break_end && info[nx].lower_start && info[nx].first == info[nx].content) {
            size_t v = f.last;
            const Info& g = info[nx];
            bool label_next = g.first + 1 <= g.last && p(d, g.first + 1, ':');
            if (p(d, v, '-') && v > f.first && d.word(v - 1) && d.joined(v) && v == f.l.b - 1) {
                f.join = f.join_hyphen = true;
                continue;
            }
            bool page = f.first == f.last && d.number(v);
            // lines of a log or list that start alike are separate lines ("retry 1/4 ..." / "retry 2/4 ...")
            bool alike = (d.low(f.first) == d.low(g.first) && !CONTINUES.contains(d.low(f.first))) ||
                         (d.number(f.first) && d.number(g.first));
            if ((d.word(v) || p(d, v, ',') || page) && !f.colon && !label_next && !g.colon && !f.list_item && !alike &&
                !listing) {
                f.join = true;
                continue;
            }
        }
        // a line that cannot end a sentence ("... in the") goes on, whatever the next line starts with; so does a
        // long unfinished line before a line of digits or a running header
        if (has_next && !blank_after && info[nx].kind == PLAIN && !f.md_heading && !f.html_heading &&
            !f.list_item && !f.colon && d.word(f.last) && d[f.last].casing != rv::UPPER &&
            info[nx].first == info[nx].content && !info[nx].md_heading && !info[nx].html_heading &&
            !info[nx].quote) {
            const Info& g = info[nx];
            if (CONTINUES.contains(d.low(f.last)) && f.l.b - 1 == f.last) {
                f.join = true;
                continue;
            }
            // (only such a word joins when written with a capital: "See")
            bool lower_end = d[f.last].casing == rv::LOWER;
            if (lower_end && f.words >= 6 && d.number(g.first) && g.words > 0 && !g.colon && !d.number(f.first) &&
                !listing) {
                f.join = true;
                continue;
            }
            if (lower_end && f.words >= 6 && g.has_word && !g.ends_punct && g.last - g.first < 6 && !g.lower_start) {
                size_t n2 = next_kept(nx);
                size_t n3 = n2;
                if (n2 != Doc::npos && info[n2].kind == PLAIN && info[n2].first == info[n2].last &&
                    d.number(info[n2].first) && info[n2].l.blank_before == 0)
                    n3 = next_kept(n2);
                if (n3 != Doc::npos && info[n3].l.blank_before == 0 && info[n3].kind == PLAIN && info[n3].lower_start) {
                    f.join = true;
                    info[nx].join = true;
                    if (n3 != n2) info[n2].join = true;
                    li = (n3 != n2 ? n2 : nx);
                    continue;
                }
            }
        }
        // a page number between two halves of a sentence
        if (has_next && !blank_after && info[nx].kind == PLAIN && info[nx].first == info[nx].last &&
            d.number(info[nx].first) && !f.ends_punct && !f.colon) {
            size_t n2 = next_kept(nx);
            if (n2 != Doc::npos && info[n2].l.blank_before == 0 && info[n2].lower_start) {
                f.join = true;
                info[nx].join = true;
                info[nx].kind = PLAIN;
                ++li;
                continue;
            }
        }
        bool cut_off = f.ends_punct && (p(d, f.last, 0x2026) || (p(d, f.last, '.') && p(d, f.last - 1, '.')));
        if (f.md_heading && cut_off) {
            f.period = true;
            continue;
        }
        if (f.ends_punct || !f.has_word || f.list_item) continue;
        // "Good work 🎉" is no heading, but a title with an emoji over a divider is
        if (f.dots && !f.colon && !f.md_heading && !f.html_heading) continue;
        if (f.emoji_end && !f.md_heading && !f.html_heading && !(li + 1 < nl && info[li + 1].kind == DELETED) &&
            !(blank_after && f.title_start && f.words <= 8 && !f.colon))
            continue;
        if (f.section) {
            f.period = true;
            continue;
        }
        if (f.lower_start && !f.md_heading && !f.html_heading) continue;
        if (f.arrows && !f.md_heading && !f.html_heading && !f.all_caps) continue;
        if (f.md_heading || f.html_heading || f.separators || (f.bold && f.title_start && !f.colon && has_next)) {
            f.period = true;
            continue;
        }
        if (f.reference) {
            f.period = has_next;
            continue;
        }
        // a line with a blank line on both sides that starts with a number and ends with a word: a title or a
        // value of its own ("2022 FIFA World Cup final", "64% humidity")
        bool isolated = (f.l.blank_before > 0 || pv == Doc::npos) && blank_after;
        bool number_title = isolated && d.number(f.first) && d.word(f.last) && !f.colon && f.words >= 1 &&
                            sp(d, f.last);
        if (!has_next || !(f.title_start || number_title)) continue;
        // a running header before a page number ("Insurance Panel Study" / "19"), and a page line ("Page 27")
        if (info[nx].kind == PLAIN && info[nx].first == info[nx].last && d.number(info[nx].first)) continue;
        if (f.first + 1 == f.last && d.is(f.first, "page") && d.number(f.last)) continue;
        if (info[nx].kind == PLAIN && info[nx].lower_start && !blank_after) continue;
        if (d.word(f.last) && d[f.last].casing == rv::LOWER && CONTINUES.contains(d.low(f.last))) continue;
        bool first = pv == Doc::npos;
        if (f.colon) {
            // "Model card: Atlas Mini" before a blank line is a title; "Install path: ✓" is a value
            size_t c = f.first;
            while (c <= f.last && !p(d, c, ':')) ++c;
            int words = 0;
            bool digits = false, other = false;
            for (size_t k = c + 1; k <= f.last; ++k) {
                if (d.word(k)) ++words;
                else if (d.number(k)) digits = true;
                else other = true;
            }
            // "Section 4.2: Results", "Chapter 3: The Storm": a numbered section title
            bool numbered_title = c >= f.first + 2 && SECTION_WORDS.contains(d.low(f.first)) && d[f.first].casing != rv::LOWER &&
                                  (d.number(f.first + 1) || (d.word(f.first + 1) && d[f.first + 1].casing == rv::UPPER)) &&
                                  words >= 1 && !digits && !other && c + 1 <= f.last && d.word(c + 1) &&
                                  d[c + 1].casing != rv::LOWER;
            if (!numbered_title && !((blank_after || first) && words >= 2 && !digits && !other && !f.arrows)) continue;
            if (numbered_title) {
                f.period = true;
                continue;
            }
            if (!blank_after && info[nx].colon) continue;  // a run of "Label: value" lines
            f.period = true;
            continue;
        }
        if (first) {
            f.period = true;
            continue;
        }
        const Info& pf = info[pv];
        bool closed = f.l.blank_before > 0 || pf.ends_punct || pf.period || pf.kind == TABLE || pf.kind == EMPTY ||
                      pf.image_end || pf.quote || (li > 0 && info[li - 1].kind == DELETED);
        if (!closed && !f.all_caps) continue;
        const Info& nf = info[nx];
        bool next_cut = nf.ends_punct && (p(d, nf.last, 0x2026) || (p(d, nf.last, '.') && p(d, nf.last - 1, '.')));
        bool next_plain_u = nf.kind == PLAIN && (!nf.ends_punct || next_cut) && !nf.md_heading && !nf.html_heading &&
                            !nf.separators && !nf.list_item && !nf.reference && nf.has_word && !blank_after &&
                            !nf.all_caps;
        if (next_plain_u && !f.all_caps) {
            // the next line ends a paragraph only if the line after it is closed
            continue;
        }
        f.period = true;
    }
}

void Layout::emit_line(size_t li) {
    Info& f = info[li];
    if (f.kind != PLAIN) return;
    // section number in front of a heading: "3.2 Methods" -> "Three point two, Methods"
    if ((f.period || f.md_heading) && (f.section || f.md_heading) && d.number(f.content)) {
        size_t k = f.content;
        std::string words;
        bool ok = true;
        while (ok) {
            if (d.str(k).size() > 2) ok = false;
            else words += (words.empty() ? "" : " point ") + rv::cardinal(std::stoull(std::string(d.str(k))));
            if (ok && p(d, k + 1, '.') && d.number(k + 2) && d.joined(k + 1) && d.joined(k + 2)) k += 2;
            else break;
        }
        if (ok && k + 1 <= f.last && sp(d, k + 1) && d.word(k + 1) && d[k + 1].casing != rv::LOWER)
            out.tokens(f.content, k, words + ",", 75);
    }
    // all-caps headings and labels in title case
    bool heading = f.period || f.md_heading || f.html_heading;
    if (heading && f.all_caps) {
        bool first = true;
        for (size_t k = f.content; k <= f.last; ++k) {
            if (!d.word(k)) continue;
            std::string t = title_word(d, k, first);
            first = false;
            if (!t.empty() && t != d.str(k)) out.tokens(k, k, t, 72);
        }
    } else if (d.word(f.content) && d[f.content].casing == rv::UPPER) {
        size_t k = f.content;
        while (k <= f.last && (d.word(k) && d[k].casing == rv::UPPER) && k < f.content + 6) {
            if (k + 1 <= f.last && !(sp(d, k + 1) && d.word(k + 1)) && !p(d, k + 1, ':')) break;
            ++k;
        }
        bool value_ok = k + 1 > f.last || !(d.number(k + 1) || (d.word(k + 1) && d[k + 1].casing == rv::UPPER &&
                                                              d.str(k + 1).size() > 1));
        if (k <= f.last && p(d, k, ':') && d.joined(k) && value_ok) {
            bool any_long = false;
            for (size_t m = f.content; m < k; ++m) any_long |= d.str(m).size() >= 3 && has_vowel(lower(d.str(m)));
            if (any_long)
                for (size_t m = f.content; m < k; ++m) {
                    std::string t = title_word(d, m, m == f.content);
                    if (!t.empty() && t != d.str(m)) out.tokens(m, m, t, 72);
                }
        }
    }
}

// The end of a kept line: a period, a join, removed trailing markup, removed deleted lines after it.
void Layout::line_end(size_t li) {
    Info& f = info[li];
    if (f.kind != PLAIN && f.kind != EMPTY) return;
    size_t b = f.l.b;
    size_t nx = next_line(li);
    // the following lines up to the next kept one; trailing deleted lines go with the last kept line
    size_t nk = next_kept(li);
    if (f.kind == PLAIN && nk != Doc::npos) {  // lines of tags or emoji at the end go with the last spoken line
        bool spoken_after = false;
        for (size_t k = nk; k < info.size() && !spoken_after; ++k) spoken_after = info[k].kind == PLAIN || info[k].kind == TABLE;
        if (!spoken_after) nk = Doc::npos;
    }
    uint32_t to;
    std::string nls;
    bool deleted_between = false;
    if (nk == Doc::npos) {
        to = static_cast<uint32_t>(d.text.size());
        deleted_between = nx != Doc::npos;
    } else {
        // deleted lines before nk are covered here; comment lines are left to the comment rule
        size_t stop = nx != Doc::npos ? nx : nk;
        if (info[stop].kind == DELETED) {
            size_t k = stop;
            while (k < info.size() && info[k].kind == DELETED) ++k;
            if (k < info.size() && info[k].kind != COMMENT) {
                to = d[info[k].l.a].start;
                deleted_between = true;
            } else {
                to = d[info[stop].l.a].start;
            }
        } else {
            to = d[info[stop].l.a].start;
        }
        for (uint32_t q = d[b - 1].end; q < d[info[stop].l.a].start; ++q)
            if (d.text[q] == '\n') nls += '\n';
        if (deleted_between) {  // a blank line on either side of a removed divider stays
            std::string after;
            for (uint32_t q = to; q > d[b - 1].end; --q) {
                char ch = d.text[q - 1];
                if (ch == '\n') after += '\n';
                else if (ch != ' ' && ch != '\t' && ch != '\r') break;
            }
            if (after.size() > nls.size()) nls = after;
        }
    }
    if (f.kind == EMPTY) {
        bool first = true;  // nothing is spoken before: a line of tags or emoji at the start goes with its line break
        for (size_t k = 0; k < li && first; ++k) first = info[k].kind != PLAIN && info[k].kind != TABLE;
        if (first && nk != Doc::npos) out.add(d[f.l.a].start, d[info[nk].l.a].start, "", 93);
        else if (deleted_between && d[b - 1].end < to) out.add(d[b - 1].end, to, nk == Doc::npos ? "" : nls, 93);
        return;
    }
    // "Dec. 24: 9:00 a.m.–2:00 p.m." among lines without a final period: the period of "p.m." goes too
    size_t v = f.last;
    if (!f.period && !f.join && v >= f.first + 3 && p(d, v, '.') && d.is(v - 1, "m") && p(d, v - 2, '.') &&
        (d.is(v - 3, "a") || d.is(v - 3, "p")) && d.joined(v) && d.joined(v - 1) && d.joined(v - 2)) {
        bool plain_neighbour = false, period_neighbour = false;
        for (size_t m : {li - 1, li + 1}) {
            if (m >= info.size() || info[m].kind != PLAIN || info[m].last < info[m].first) continue;
            if ((m == li + 1 ? info[m] : f).l.blank_before > 0) continue;
            if (info[m].ends_punct && !p(d, info[m].last, ':')) period_neighbour = true;
            else plain_neighbour = true;
        }
        if (plain_neighbour && !period_neighbour) out.tokens(v, v, "", 50);
    }
    uint32_t from = d[f.last].end;
    std::string spoken;
    if (f.join_hyphen) {
        from = d[f.last].start;
        spoken = "";
    } else {
        if (f.period) spoken += ".";
        spoken += f.join ? " " : (nk == Doc::npos ? "" : nls);
    }
    bool trailing = f.last + 1 < b;
    if (from >= to) {
        if (f.period) out.add(from, from, ".", 50);  // the text ends with the line
        return;
    }
    if (spoken == d.slice(from, to)) return;
    int score = (f.period || f.join || trailing || deleted_between) ? 93 : 5;
    out.add(from, to, spoken, score);
}

// Spaces and tabs between tokens: one space; none before a comma or period that ends a clause; none at the
// start or end of a line. Line breaks are kept (line ends handle their own gaps). Lowest score.
void Layout::gaps() {
    size_t n = d.size();
    for (size_t i = 0; i <= n; ++i) {
        uint32_t s = gap_start(d, i), e = gap_end(d, i);
        if (s >= e) continue;
        if (e - s == 1 && i > 0 && i < n) {  // the common gaps: one space between words, one line break
            char c = d.text[s];
            if (c == '\n') continue;
            if (c == ' ' && (d[i].kind != Kind::Punct || !attaches_left(d, i)) &&
                (d[i - 1].kind != Kind::Punct || (d[i - 1].cp != '(' && d[i - 1].cp != '[')))
                continue;
        }
        std::string_view g = d.slice(s, e);
        bool nl = g.find('\n') != std::string_view::npos;
        std::string want;
        if (i == 0 || i == n) {
            want = i == n ? gap_newlines(d, i) : "";
            if (i == n) want = "";
        } else if (nl) {
            want = gap_newlines(d, i);
        } else if ((p(d, i, ',') || p(d, i, '.') || p(d, i, ';') || p(d, i, '!') || p(d, i, '?') || p(d, i, ':')) &&
                   (d.word(i - 1) || d.number(i - 1) || p(d, i - 1, ')') || p(d, i - 1, '"') || p(d, i - 1, '\'') ||
                    p(d, i - 1, 0x201D) || p(d, i - 1, 0x2019)) &&
                   (i + 1 >= n || d[i + 1].space_before ||
                    ((p(d, i + 1, '"') || p(d, i + 1, 0x201D) || p(d, i + 1, ')')) && (i + 2 >= n || d[i + 2].space_before))) &&
                   !p(d, i + 1, '.')) {
            want = "";
        } else if (((p(d, i - 1, '(') || p(d, i - 1, '[')) && (i < 2 || d[i - 1].space_before) && !p(d, i, ')')) ||
                   ((p(d, i, ')') || p(d, i, ']')) && (i + 1 >= n || d[i + 1].space_before || attaches_left(d, i + 1)) &&
                    !p(d, i - 1, '('))) {
            want = "";  // "( all fields are required )"
        } else {
            want = " ";
        }
        if (want == g) continue;
        out.add(s, e, want, 5);
    }
}

// A letter marker at token i before a word: "(a)" / "(A)", or "a." / "a)" in lower case. Returns its last token,
// or 0.
size_t letter_marker(const Doc& d, size_t i) {
    size_t e = 0;
    if (p(d, i, '(') && d.word(i + 1) && d.str(i + 1).size() == 1 && p(d, i + 2, ')') && d.joined(i + 1) &&
        d.joined(i + 2))
        e = i + 2;
    else if (d.word(i) && d.str(i).size() == 1 && d[i].casing == rv::LOWER && (p(d, i + 1, '.') || p(d, i + 1, ')')) &&
             d.joined(i + 1))
        e = i + 1;
    if (!e || e + 1 >= d.size() || !sp(d, e + 1) || !d.word(e + 1) || d[e + 1].newline_before) return 0;
    return e;
}

// " a. " or " a) ": a lettered item inside a line.
bool letter_a_marker(std::string_view t) {
    for (size_t k = 2; k + 1 < t.size(); ++k)
        if ((t[k] == '.' || t[k] == ')') && t[k - 1] == 'a' && t[k - 2] == ' ' && t[k + 1] == ' ') return true;
    return false;
}

// List markers inside a line: "provide: (a) your name, (b) your e-mail, and (c) your address" -> "A, your name,
// B, ...", and check boxes inside a line: "Checklist: [x] Calibrate. [ ] Replace" -> "Done: Calibrate. Open:
// Replace". Line starts are handled by the layout rule.
RV_RULE("structure", inline_list) {
    if (!(has_any(doc, "([:") || has(doc, "1)") || has(doc, "1-") || has(doc, "1.") || letter_a_marker(doc.text))) return;
    size_t n = doc.size();
    char expect = 0;      // the letter the next marker of the current series has
    uint64_t expect_n = 0;  // the number the next numbered marker has
    bool question = false;  // a "Q:" label came before
    if (doc.number(0) && doc.str(0) == "1" && (p(doc, 1, ')') || p(doc, 1, '-') || p(doc, 1, '.')) && doc.joined(1))
        expect_n = 2;  // a list that starts the text (read by the layout rule) goes on inside the line
    for (size_t i = 1; i + 3 < n; ++i) {
        if (doc.word(i) && doc[i].end - doc[i].start > 1) continue;  // markers start with a bracket, a digit or a letter
        if (doc[i].newline_before && doc.number(i) && doc.str(i) == "1" && doc.joined(i + 1) &&
            (p(doc, i + 1, ')') || p(doc, i + 1, '-') || p(doc, i + 1, '.')))
            expect_n = 2;
        if (doc.word(i) && doc.str(i).size() == 1 && (doc.str(i) == "Q" || doc.str(i) == "A") && p(doc, i + 1, ':') &&
            doc.joined(i + 1)) {  // Q: and A: labels inside a line: "... final? A: No, ..."
            bool q = doc.str(i) == "Q";
            if (!doc[i].newline_before && doc.sentence_start(i) && (q || question || p(doc, i - 1, '?')))
                out.tokens(i, i, q ? "Question" : "Answer", 80);
            question |= q;
        }
        if (doc[i].newline_before || !sp(doc, i)) continue;
        // numbered markers: "1)", "1-", "1.", "(1)" before a word, after ":", ",", ";", a sentence end or "and"
        {
            size_t a = i, num = i, e = i + 1;
            bool paren = p(doc, i, '(') && doc.number(i + 1) && doc.joined(i + 1) && p(doc, i + 2, ')') &&
                         doc.joined(i + 2);
            if (paren) num = i + 1, e = i + 2;
            bool plain = !paren && doc.number(i) && doc.joined(i + 1) &&
                         (p(doc, i + 1, ')') || p(doc, i + 1, '-') || p(doc, i + 1, '.'));
            if ((paren || plain) && doc.str(num).size() <= 2 && e + 1 < n && sp(doc, e + 1) && doc.word(e + 1) &&
                !doc[e + 1].newline_before) {
                uint64_t v = std::stoull(std::string(doc.str(num)));
                bool sep = a > 0 && (p(doc, a - 1, ':') || p(doc, a - 1, ',') || p(doc, a - 1, ';') ||
                                     p(doc, a - 1, '?') || p(doc, a - 1, '!') ||
                                     (p(doc, a - 1, '.') && !(a >= 2 && doc.number(a - 2))));
                bool link = a > 0 && (doc.is(a - 1, "and") || doc.is(a - 1, "or") || doc.is(a - 1, "then"));
                bool dot_ok = !p(doc, e, '.') || doc[e + 1].casing != rv::LOWER;  // "1. Check", not "1. the"
                // "(1)" after any word starts a series when "(2)" follows on the line ("included (1) a, (2) b")
                bool series = false;
                if (paren && v == 1 && !sep && doc.word(a - 1) && !doc.joined(a))
                    for (size_t k = e + 1; k + 2 < n && !doc[k].newline_before && k < e + 60 && !series; ++k)
                        series = p(doc, k, '(') && doc.str(k + 1) == "2" && p(doc, k + 2, ')') && doc.joined(k + 1) &&
                                 doc.joined(k + 2) && doc[k].space_before;
                if (dot_ok && ((sep && v == 1) || series || (expect_n && v == expect_n && (sep || link)))) {
                    out.tokens(a, e, cap_cardinal(v) + ",", 50);
                    expect_n = v + 1;
                    i = e;
                    continue;
                }
                if (v != expect_n) expect_n = 0;
            }
        }
        if (p(doc, i, '[')) {  // a check box before a word
            bool open = p(doc, i + 1, ']') && doc[i + 1].start == doc[i].end + 1;
            bool done = doc.is(i + 1, "x") && p(doc, i + 2, ']') && doc.joined(i + 1) && doc.joined(i + 2);
            size_t e = open ? i + 1 : i + 2;
            if ((open || done) && e + 1 < n && doc.word(e + 1) && sp(doc, e + 1) && !doc[e + 1].newline_before)
                out.tokens(i, e, open ? "Open:" : "Done:", 50);
            continue;
        }
        // letter markers: "(a)", and "a." / "a)" in lower case
        size_t e = letter_marker(doc, i);
        if (!e) continue;
        size_t letter = e - 1;
        char c = static_cast<char>(doc.str(letter)[0] | 0x20);
        bool paren = p(doc, i, '(');
        bool sep = p(doc, i - 1, ':') || p(doc, i - 1, ',') || p(doc, i - 1, ';') || p(doc, i - 1, '.');
        bool link = doc.is(i - 1, "and") || doc.is(i - 1, "or") || doc.is(i - 1, "then");
        // "(a)" to "(c)" after a separator; later letters, and markers after "and" / "or", only in a series; a
        // series also starts with "a" anywhere when its "b" follows on the line ("consider (a) ..., and (b) ...")
        bool series = false;
        if (c == 'a' && (!sep || !paren))
            for (size_t k = e + 1; k + 2 < n && !doc[k].newline_before && k < e + 80 && !series; ++k) {
                size_t f = sp(doc, k) ? letter_marker(doc, k) : 0;
                series = f && (doc.str(f - 1)[0] | 0x20) == 'b' && p(doc, k, '(') == paren;
            }
        bool ok = (sep && c <= 'c' && paren) || (c == 'a' && series) || (expect && c == expect && (sep || link));
        if (!ok) {
            expect = 0;
            continue;
        }
        out.tokens(i, e, std::string(1, static_cast<char>(c - 32)) + ",", 50);
        expect = static_cast<char>(c + 1);
        i = e;
    }
}

// Outline markers. A chain at the start of a sentence: "I. A. 1." -> "Part One, Section A, Number One." (a period
// before a capital, a comma before lower case), "A. 1. a." -> "Section A, Number One, letter A,". In a text laid
// out as an outline (a capital letter marker "A." with "B." or "1." after it), single markers are read too:
// "A. Project scope, B. Budget, and C. Timeline." -> "Section A, Project scope, Section B, ...", a numbered marker
// under a letter is "Number One,", and a lower-case letter right under a number is "letter A,".
const rv::WordSet TITLE_ABBR{"dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "gen", "col", "capt", "rev", "fig",
                             "no", "vol", "ch", "sec", "art", "pt", "app", "eq", "p", "pp"};

struct Marker {
    size_t i, last;  // first token and the period (or hyphen) after it
    char type;       // 'R' Roman numeral in capitals, 'L' capital letter, 'N' number, 'l' lower-case letter
    int value;
};

// The text may hold an outline marker: one or two letters or digits after a space or line start, then ". " or "- ".
bool maybe_marker(std::string_view t) {
    for (size_t k = t.find('.'); k != std::string_view::npos; k = t.find('.', k + 1)) {
        if (k == 0 || k + 1 >= t.size() || t[k + 1] != ' ') continue;
        size_t a = k;
        while (a > 0 && k - a < 5 && rv::is_ascii_alnum(t[a - 1])) --a;
        if (a == k || !(a == 0 || t[a - 1] == ' ' || t[a - 1] == '\n' || t[a - 1] == '.')) continue;
        std::string_view w = t.substr(a, k - a);  // "1", "12", "A", "a", "IV"
        bool digits = w.size() <= 2 && rv::is_ascii_digit(w[0]) && rv::is_ascii_digit(w.back());
        bool roman = w.find_first_not_of("IVX") == std::string_view::npos;
        if (digits || w.size() == 1 || roman) return true;
    }
    return false;
}

RV_RULE("structure", outline) {
    if (!maybe_marker(doc.text)) return;
    size_t n = doc.size();
    // a marker position: the start of a line or sentence, after a colon, semicolon, comma, "and", "or", "then"
    auto position = [&](size_t k) {
        if (k == 0 || doc[k].newline_before) return true;
        size_t q = k - 1;
        if (!doc[k].space_before) return false;
        if (p(doc, q, ':') || p(doc, q, ';') || p(doc, q, ',')) return true;
        if (doc.is(q, "and") || doc.is(q, "or") || doc.is(q, "then")) return true;
        if (p(doc, q, '!') || p(doc, q, '?')) return true;
        if (!p(doc, q, '.') || q == 0) return false;
        // after a period that ends a sentence, not after an initial or a title ("J. A. Smith", "Dr. A. Smith")
        size_t w = q - 1;
        if (doc.word(w) && TITLE_ABBR.contains(doc.low(w))) return false;
        if (doc.word(w) && doc.str(w).size() == 1) {
            // the period of a Roman numeral marker in a chain ("I. A. 1.") counts, not an initial ("J. A. Smith")
            bool roman = doc.str(w) == "I" || doc.str(w) == "V" || doc.str(w) == "X";
            return roman && (w == 0 || doc[w].newline_before || p(doc, w - 1, ':') || p(doc, w - 1, '.'));
        }
        return true;
    };
    std::vector<Marker> ms;
    for (size_t k = 0; k + 2 < n; ++k) {
        // "1)", "(1)", "ii)": markers of a plain list, which end a numbered series under a letter
        if ((doc.number(k) || doc.word(k)) && p(doc, k + 1, ')') && doc.joined(k + 1) && doc.str(k).size() <= 4) {
            if (!ms.empty()) ms.push_back({k, k + 1, 'P', 0});
            continue;
        }
        if (!(doc.word(k) || doc.number(k))) continue;
        bool dot = p(doc, k + 1, '.') && doc.joined(k + 1);
        bool hyphen = doc.number(k) && p(doc, k + 1, '-') && doc.joined(k + 1);
        if (!(dot || hyphen) || !doc[k + 2].space_before || doc[k + 2].newline_before) continue;
        if (!(doc.word(k + 2) || (dot && (doc.number(k + 2) || doc.word(k + 2))))) continue;
        bool chained = !ms.empty() && ms.back().last + 1 == k && ms.back().type != 'P';  // "I. A. 1."
        if (!chained && !position(k)) continue;
        std::string_view w = doc.str(k);
        Marker m{k, k + 1, 0, 0};
        if (doc.number(k)) {
            if (w.size() > 2 || w[0] == '0') continue;
            m.type = 'N';
            m.value = std::stoi(std::string(w));
        } else if (w.size() == 1 && w[0] >= 'A' && w[0] <= 'H' && w[0] != 'I') {
            m.type = 'L';
            m.value = w[0] - 'A' + 1;
        } else if (w.size() == 1 && w[0] >= 'a' && w[0] <= 'h' && w[0] != 'i') {
            m.type = 'l';
            m.value = w[0] - 'a' + 1;
        } else if (doc[k].casing == rv::UPPER && rv::roman_value(w) > 0 && rv::roman_value(w) <= 20) {
            m.type = 'R';
            m.value = rv::roman_value(w);
        } else {
            continue;
        }
        if (hyphen && m.type != 'N') continue;
        ms.push_back(m);
    }
    if (ms.empty()) return;
    // chains: markers that follow each other directly ("I. A. 1. a.")
    auto next_in_chain = [&](size_t a, size_t b) { return b < ms.size() && ms[b].i == ms[a].last + 1; };
    // is the text an outline? a capital "A." with "B." or a number marker after it
    bool outline = false;
    for (size_t a = 0; a < ms.size() && !outline; ++a) {
        if (ms[a].type != 'L' || ms[a].value != 1) continue;
        for (size_t b = a + 1; b < ms.size() && !outline; ++b)
            outline = (ms[b].type == 'L' && ms[b].value == 2) || (ms[b].type == 'N' && ms[b].value == 1);
    }
    char prev_type = 0;   // the type of the last marker read
    bool numbered = false;  // a number series under a letter is going on
    for (size_t a = 0; a < ms.size(); ++a) {
        const Marker& m = ms[a];
        // a chain of two or more levels
        size_t e = a;
        while (next_in_chain(e, e + 1) && e - a < 3) ++e;
        bool chain = false;
        if (e > a) {
            std::string seq;
            for (size_t k = a; k <= e; ++k) seq += ms[k].type;
            chain = seq == "RLN" || seq == "RLNl" || seq == "LN" || seq == "LNl";
            if (!chain && (seq.rfind("RLN", 0) == 0 || seq.rfind("LNl", 0) == 0)) {
                e = a + (seq[0] == 'R' ? 2 : 2);
                chain = true;
            } else if (!chain && seq.rfind("LN", 0) == 0) {
                e = a + 1;
                chain = true;
            }
        }
        if (chain) {
            std::string words;
            for (size_t k = a; k <= e; ++k) {
                const Marker& c = ms[k];
                if (!words.empty()) words += ", ";
                if (c.type == 'R') words += "Part " + cap_cardinal(static_cast<uint64_t>(c.value));
                else if (c.type == 'L') words += "Section " + std::string(1, static_cast<char>('A' + c.value - 1));
                else if (c.type == 'N') words += "Number " + cap_cardinal(static_cast<uint64_t>(c.value));
                else words += "letter " + std::string(1, static_cast<char>('A' + c.value - 1));
            }
            size_t after = ms[e].last + 1;
            bool part = ms[a].type == 'R' && ms[e].type == 'N';
            bool cap_next = doc.word(after) && doc[after].casing != rv::LOWER;
            words += part && cap_next ? "." : ",";
            out.tokens(ms[a].i, ms[e].last, words, 60);
            prev_type = ms[e].type;
            // after "I. A. 1." the numbers go on as a plain list ("2. Staffing" -> "Two,")
            numbered = ms[a].type != 'R' && (ms[e].type == 'N' || (e > a && ms[e - 1].type == 'N'));
            if (ms[a].type == 'R') prev_type = 'P';
            a = e;
            continue;
        }
        if (!outline) continue;
        std::string words;
        if (m.type == 'L') {
            words = "Section " + std::string(1, static_cast<char>('A' + m.value - 1)) + ",";
            numbered = false;
        } else if (m.type == 'N' && (prev_type == 'L' || (prev_type == 'l' && numbered) || (prev_type == 'N' && numbered))) {
            if (m.value != 1 && !numbered) continue;
            words = "Number " + cap_cardinal(static_cast<uint64_t>(m.value)) + ",";
            numbered = true;
        } else if (m.type == 'l' && prev_type == 'N' && numbered) {
            words = "letter " + std::string(1, static_cast<char>('A' + m.value - 1)) + ",";
        } else if (m.type == 'l' && prev_type == 'l' && numbered && doc[m.i].newline_before) {
            words = "letter " + std::string(1, static_cast<char>('A' + m.value - 1)) + ",";
        } else {
            if (m.type == 'N') numbered = false;
            prev_type = m.type;
            continue;
        }
        out.tokens(m.i, m.last, words, 60);
        prev_type = m.type;
    }
}

RV_RULE("structure", layout) {
    if (doc.size() == 0) return;
    Layout L(doc, out);
    std::vector<Line> lines = split_lines(doc);
    L.info.resize(lines.size());
    for (size_t k = 0; k < lines.size(); ++k) L.info[k].l = lines[k];
    for (Info& f : L.info) L.analyse(f);
    bool wide_gap = false;  // two spaces or more between tokens of a line: maybe a row of values
    for (size_t k = 1; k < doc.size() && !wide_gap; ++k) wide_gap = doc[k].start - doc[k - 1].end >= 2 && !doc[k].newline_before;
    if (wide_gap || has_any(doc, "|\t")) L.tables();
    L.decide();
    for (size_t li = 0; li < L.info.size(); ++li) {
        L.emit_line(li);
        L.line_end(li);
    }
    // deleted lines at the start of the text, or between table rows and kept lines, go on their own
    for (size_t li = 0; li < L.info.size(); ++li) {
        if (L.info[li].kind != DELETED) continue;
        size_t pv = li;
        bool covered = false;
        while (pv-- > 0) {
            if (L.info[pv].kind == DELETED || L.info[pv].kind == COMMENT) continue;
            covered = true;
            break;
        }
        if (covered) continue;
        const Line& l = L.info[li].l;
        uint32_t to = li + 1 < L.info.size() ? doc[L.info[li + 1].l.a].start : doc[l.b - 1].end;
        if (li + 1 >= L.info.size() && li > 0) {
            out.add(doc[l.a - 1].end, doc[l.b - 1].end, "", 80);
            continue;
        }
        out.add(doc[l.a].start, to, "", 80);
    }
    L.gaps();
}

}  // namespace
