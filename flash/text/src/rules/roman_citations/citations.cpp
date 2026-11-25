// Citation and reference markers: bracketed and parenthesized numeric citations, superscript and caret markers,
// footnote symbols and leading footnote marks, [sic] and wiki tags, author-year citations, et al., ibid.,
// reference-list entries, arXiv / PMID / ISBN / ISSN numbers and DOIs; brackets that are not citations: intervals
// ("[0, 1]" -> zero to one), values ("[0.5]") and index variables ("e[i]").
#include <vector>

#include "rules/roman_citations/common.h"

namespace {

using namespace rc;

// Words after which a bracketed citation is part of the sentence: "listed in [16]" -> listed in reference sixteen.
const rv::WordSet REF_CUES{"across", "as", "by", "cf", "cite", "cited", "cites", "citing", "follow", "followed",
                           "following", "follows", "from", "in", "of", "on", "per", "see", "to", "under", "with"};
// Words before "word[3]" that make it an index: "stored in records[5]", "logged as run[4]", "rows[0] through rows[9]".
const rv::WordSet INDEX_CUES{"and", "as", "at", "by", "called", "from", "in", "index", "into", "labeled", "labelled",
                             "marked", "named", "numbered", "or", "reads", "see", "through", "thru", "to", "under"};
// Words after which a bracket with two numbers is an interval: "values in [0, 1]". The strong ones decide alone;
// after the others a close pair of numbers above one is a citation ("the mean spread [12, 14],").
const rv::WordSet INTERVAL_CUES{"are", "across", "as", "at", "band", "bands", "between", "ci", "in", "inside",
                                "interval", "intervals", "into", "is", "last", "of", "outside", "range", "ranged",
                                "ranges", "scale", "sits", "span", "spans", "spread", "such", "values", "was",
                                "window", "within"};
const rv::WordSet STRONG_INTERVAL_CUES{"across", "band", "bands", "between", "ci", "inside", "interval", "intervals",
                                       "outside", "range", "ranged", "ranges", "span", "spans", "window", "within"};
// Words after a bracket that make its two numbers a range of a quantity: "[3, 9] minutes", "[60, 95] percent".
const rv::WordSet INTERVAL_UNITS{
    "cases", "cm", "cycles", "days", "degrees", "faults", "feet", "ft", "g", "h", "hours", "hz", "items", "kg", "km",
    "m", "meters", "metres", "mg", "miles", "min", "minutes", "mm", "months", "ms", "participants", "patients",
    "people", "percent", "points", "rounds", "runs", "s", "samples", "seconds", "steps", "times", "units", "users",
    "weeks", "years"};
// Words before a bracket after a period that is not a sentence end: "Smith et al. [14] as".
const rv::WordSet NOT_SENTENCE_END{"al", "cf", "eq", "eqs", "etc", "fig", "figs", "ref", "refs", "vs"};
// Words before "(14)" that make it a label, not a citation: "Eq. (5)", "Stage (2)".
const rv::WordSet PAREN_LABELS{
    "art", "article", "appendix", "case", "chapter", "claim", "claims", "condition", "criterion", "definition", "eq",
    "eqs", "equation", "equations", "example", "exercise", "fig", "figure", "figures", "footnote", "formula",
    "hypothesis", "item", "lemma", "line", "lines", "model", "no", "note", "number", "option", "p", "page", "part",
    "phase", "pp", "problem", "proposition", "question", "rule", "scenario", "section", "stage", "step", "steps",
    "table", "theorem", "type", "version"};
// Label words that name a figure or table only when capitalized: "in the appendix (9)." is a citation.
const rv::WordSet WEAK_LABELS{"appendix", "figure", "figures", "item", "model", "note", "number", "table", "type",
                              "version"};
// Words after "(6)" that go on with the sentence, so the bracket closed a clause: "on usability (6) and was".
const rv::WordSet CLAUSE_GOES_ON{"after", "although", "and", "as", "at", "because", "before", "but", "by", "for",
                                 "from", "in", "on", "or", "since", "so", "than", "that", "though", "to", "when",
                                 "where", "which", "while", "with"};
// Words before "(1)" that introduce a list item or a quantity, not a cited claim.
const rv::WordSet BEFORE_LIST{
    "a", "about", "across", "address", "addressed", "after", "against", "along", "among", "an", "and", "are", "around",
    "as", "at", "be", "been", "before", "between", "by", "consider", "cover", "covered", "covers", "discuss",
    "discussed", "during", "for", "from", "highlighted", "in", "include", "included", "includes", "including", "into",
    "is", "like", "list", "listed", "lists", "of", "on", "onto", "or", "over", "per", "review", "reviewed", "see",
    "the", "then", "through", "to", "toward", "towards", "under", "upon", "via", "was", "were", "with", "within"};

// Short words that carry a superscript power, not a citation: "sec²".
const rv::WordSet UNIT_WORDS{"deg", "lbs", "min", "mol", "rad", "sec"};
// First words of a parenthesis that is not an author-year citation.
const rv::WordSet NOT_AUTHOR{
    "accessed", "age", "ages", "april", "aug", "august", "autumn", "born", "circa", "data", "dec", "december", "died",
    "est", "established", "fall", "feb", "february", "fig", "figure", "founded", "fy", "jan", "january", "jul",
    "july", "jun", "june", "mar", "march", "may", "note", "nov", "november", "oct", "october", "published",
    "retrieved", "revised", "section", "sep", "sept", "september", "since", "source", "sources", "spring", "summer",
    "table", "updated", "version", "winter"};
// Lower-case name particles allowed inside an author name: "van der Berg".
const rv::WordSet PARTICLES{"al", "and", "da", "de", "del", "der", "des", "di", "dos", "du", "et", "la", "le", "van",
                            "von"};

// ---------------------------------------------------------------------------------------------------------------
// Bracketed numeric groups: "[3]", "[1, 2, 7]", "[3–5]", "[4, 8–10]".
struct Group {
    size_t open = 0, close = 0;
    std::vector<std::pair<uint32_t, uint32_t>> items;
    bool ranged = false, zero = false;
};

bool parse_group(const Doc& d, size_t i, Group& g) {
    if (!d.punct(i, '[') && !d.punct(i, 0xFF3B)) return false;  // also the full-width "［3］"
    g = Group{};
    g.open = i;
    size_t k = i + 1;
    for (int guard = 0; guard < 40; ++guard) {
        if (!d.number(k) || d.str(k).size() > 3) return false;
        uint32_t a = num(d, k), b = a;
        if (dash_tok(d, k + 1) && d.number(k + 2) && d.str(k + 2).size() <= 3) {
            b = num(d, k + 2);
            g.ranged = true;
            k += 2;
        }
        g.items.emplace_back(a, b);
        if (a == 0) g.zero = true;
        if (d.punct(k + 1, ']') || d.punct(k + 1, 0xFF3D)) {
            g.close = k + 1;
            return true;
        }
        if (!d.punct(k + 1, ',') && !d.punct(k + 1, ';')) return false;
        k += 2;
    }
    return false;
}

std::string ref_items(const std::vector<std::pair<uint32_t, uint32_t>>& items, bool& plural) {
    std::vector<std::string> parts;
    for (auto [a, b] : items) parts.push_back(a == b ? rv::cardinal(a) : rv::cardinal(a) + " to " + rv::cardinal(b));
    plural = items.size() > 1 || items[0].first != items[0].second;
    std::string out;
    for (size_t k = 0; k < parts.size(); ++k) {
        if (k) out += k + 1 == parts.size() ? " and " : ", ";
        out += parts[k];
    }
    return out;
}

// Next token after k, skipping citation markers that are removed (brackets, superscripts): used to decide whether
// a comma has to follow a narrative citation.
size_t after_markers(const Doc& d, size_t k) {
    size_t n = k + 1;
    for (int guard = 0; guard < 8 && n < d.size(); ++guard) {
        Group g;
        if (parse_group(d, n, g)) n = g.close + 1;
        else if (superscript(d, n)) ++n;
        else break;
    }
    return n;
}

// A signed number at token k inside brackets: "-5", "−0.3", "12.5%" -> words. Sets last and whether it has a
// decimal part; "" if there is none.
std::string signed_value(const Doc& d, size_t k, size_t& last, bool& decimal) {
    std::string w;
    if (punct_tok(d, k) && (d[k].cp == '-' || d[k].cp == 0x2212) && d.number(k + 1) && d.joined(k + 1)) {
        w = "minus ";
        ++k;
    }
    rv::NumberSpan n = rv::parse_number(d, k);
    if (!n.ok) return "";
    std::string r = rv::read_number(n);
    if (r.empty()) return "";
    decimal = !n.fraction.empty();
    w += r;
    last = n.last;
    if (d.punct(last + 1, '%') && d.joined(last + 1)) {
        w += " percent";
        ++last;
    }
    return w;
}

// Brackets around values that are not citations: "[-5, 10]", "[0.2–0.6]", "[−15%, −9%]" -> minus five to ten;
// a lone decimal "[0.5]" -> zero point five. A cue rule may read them first.
size_t value_bracket_at(const Doc& doc, Out& out, size_t i) {
    if (i > 0 && doc.joined(i) && (doc.word(i - 1) || doc.number(i - 1))) return i;  // an index
    size_t a_last = 0, b_last = 0;
    bool da = false, db = false;
    std::string a = signed_value(doc, i + 1, a_last, da);
    if (a.empty()) return i;
    size_t k = a_last + 1;
    if (doc.punct(k, ']') && da) {
        out.tokens(i, k, a, 84);
        return k;
    }
    if (!(doc.punct(k, ',') || doc.punct(k, ';') || dash_tok(doc, k))) return i;
    std::string b = signed_value(doc, k + 1, b_last, db);
    if (b.empty() || !doc.punct(b_last + 1, ']')) return i;
    out.tokens(i, b_last + 1, a + " to " + b, 84);
    return b_last + 1;
}

// The bracket at i continues a list of intervals: "[0, 40], [40, 70] and [70, 100]".
bool after_interval(const Doc& doc, size_t i) {
    size_t k = i;
    if (k > 0 && (doc.is(k - 1, "and") || doc.is(k - 1, "or"))) --k;
    if (k > 0 && doc.punct(k - 1, ',')) --k;
    if (k == i || k == 0 || !doc.punct(k - 1, ']')) return false;
    size_t o = k - 1;
    for (int guard = 0; guard < 6 && o > 0 && !doc.punct(o, '['); ++guard) --o;
    Group h;
    if (!parse_group(doc, o, h) || h.close != k - 1 || h.items.size() != 2 || h.ranged) return false;
    return h.zero || h.items[1].first - h.items[0].first > 3 || h.items[1].first <= h.items[0].first;
}

// A group and the groups joined to it by commas, "and", "or" or nothing: "[3] and [5]", "[5], [6] and [9]",
// "[2][4]". Collects the items and returns the last closing bracket.
size_t chain(const Doc& doc, const Group& g, std::vector<std::pair<uint32_t, uint32_t>>& items) {
    items = g.items;
    size_t last = g.close, k = g.close + 1;
    for (int guard = 0; guard < 20; ++guard) {
        size_t c = k;
        if (doc.punct(c, ',')) {
            ++c;
            if (doc.is(c, "and") || doc.is(c, "or")) ++c;
        } else if (doc.is(c, "and") || doc.is(c, "or")) {
            ++c;
        } else if (!(doc.punct(c, '[') && doc.joined(c))) {
            break;
        }
        Group h;
        if (!parse_group(doc, c, h) || h.zero || doc[c].newline_before) break;
        items.insert(items.end(), h.items.begin(), h.items.end());
        last = h.close;
        k = h.close + 1;
    }
    return last;
}

// "[3]" decorating a sentence is removed; "in [3] and [5]" is read as references.
size_t bracket_citation_at(const Doc& doc, Out& out, size_t i) {
    Group g;
    if (!parse_group(doc, i, g)) {
        // an index variable: "e[i]" -> e i, "avg[t]" -> avg t
        if (i > 0 && doc.joined(i) && doc.word(i - 1) && doc.punct(i + 2, ']') && doc.joined(i + 1) &&
            doc.joined(i + 2) && lower(doc, i + 1)) {
            char c = single_letter(doc, i + 1);
            if (c == 'i' || c == 'j' || c == 'k' || c == 'n' || c == 't' || c == 'x' || c == 'y' || c == 'z') {
                out.tokens(i, i + 2, std::string(1, c), 55);
                return i + 2;
            }
        }
        return value_bracket_at(doc, out, i);
    }
    size_t next = g.close + 1;
    bool next_word = next < doc.size() && !doc[next].newline_before && (doc.word(next) || doc.number(next));
    // a reference-list entry "[3] Smith — Title" (read by reference_entry)
    if (line_start(doc, i) && next < doc.size() && !doc[next].newline_before) {
        return g.close;
    }
    // joined to a word: an index "row[7]", "x[0]", "stored in records[5]", "rows[0] through rows[9]" (read by
    // electronic), or a citation that decorates the word before punctuation: "field team[1]."
    if (i > 0 && doc.joined(i) && (doc.word(i - 1) || doc.number(i - 1))) {
        size_t w = i - 1;
        bool chained = doc.punct(g.close + 1, '[') && doc.joined(g.close + 1);
        bool index_cue = w > 0 && doc.word(w - 1) && INDEX_CUES.contains(doc.low(w - 1));
        bool single = g.items.size() == 1 && !g.ranged;
        // an index goes on with the sentence ("readings[2] stayed") or repeats its name ("run[2] and run[3]")
        bool index = single && !chained && (index_cue || next_word);
        if (single && !chained && !index && doc.word(w)) {
            size_t from = i > 64 ? i - 64 : 1, to = std::min(doc.size(), i + 64);  // a window keeps the rule linear
            for (size_t q = from; q < to && !index; ++q)
                index = q != i && doc.punct(q, '[') && doc.joined(q) && doc.word(q - 1) && doc.low(q - 1) == doc.low(w);
        }
        if (doc.word(w) && doc.str(w).size() >= 2 && !g.zero && !index && !doc.joined(w))
            out.tokens(i, g.close, "", 60);
        return g.close;
    }
    size_t p = i - 1;
    bool spaced = i > 0 && doc[i].space_before;
    std::string_view prev = spaced && doc.word(p) ? doc.low(p) : std::string_view();
    bool unit_next = next < doc.size() && !doc[next].newline_before &&
                     ((doc.word(next) && INTERVAL_UNITS.contains(doc.low(next))) || doc.punct(next, 0xB0));
    // a range in brackets after "range", "between"...: "the earlier range [5–7]" -> range five to seven
    if (g.items.size() == 1 && g.ranged &&
        (prev == "range" || prev == "ranges" || prev == "ranged" || prev == "between" || unit_next)) {
        out.tokens(i, g.close, rv::cardinal(g.items[0].first) + " to " + rv::cardinal(g.items[0].second), 84);
        return g.close;
    }
    // two numbers after "in", "range", "CI"... are an interval, unless a close pair ends the sentence ("the protocol
    // in [18, 19].")
    if (g.items.size() == 2 && !g.ranged) {
        uint32_t a = g.items[0].first, b = g.items[1].first;
        bool ends = next >= doc.size() || doc[next].newline_before || doc.punct(next, '.') ||
                    doc.punct(next, ';') || doc.punct(next, '!') || doc.punct(next, '?');
        bool clause = ends || doc.punct(next, ',') || doc.punct(next, ':') || superscript(doc, next) ||
                      doc.punct(next, '[') || doc.punct(next, '(');
        // words after it: close numbers are references ("reported [14, 15] and"), a wide pair is an interval
        bool weak_cue = INTERVAL_CUES.contains(prev) && !STRONG_INTERVAL_CUES.contains(prev);
        bool interval = g.zero || b <= a || unit_next || STRONG_INTERVAL_CUES.contains(prev) || after_interval(doc, i) ||
                        (!clause && (b - a > 3 || weak_cue)) ||
                        (clause && weak_cue && !(prev == "in" && b - a <= 2 && ends) && (a <= 1 || b - a >= 4));
        if (interval) {
            // read as a range ("values in [0, 1]" -> values in zero to one) unless a cue rule reads it already
            out.tokens(i, g.close, rv::cardinal(a) + " to " + rv::cardinal(b), 84);
            return g.close;
        }
    }
    if (g.zero) {
        return g.close;
    }
    // "see reference [6]": the word is written, only the number is read ("the reference [4]" is decorated)
    if ((prev == "reference" || prev == "references") &&
        !(p > 0 && (doc.is(p - 1, "the") || doc.is(p - 1, "a") || doc.is(p - 1, "this") || doc.is(p - 1, "our")))) {
        bool plural;
        out.tokens(i, g.close, ref_items(g.items, plural), 70);
        return g.close;
    }
    // "see refs. [4, 5]" -> see references four and five
    if (doc.punct(p, '.') && doc.joined(p) && p > 0 && (doc.is(p - 1, "ref") || doc.is(p - 1, "refs")) &&
        !(p > 1 && doc.joined(p - 1))) {
        std::vector<std::pair<uint32_t, uint32_t>> items;
        size_t last = chain(doc, g, items);
        bool plural;
        std::string words = ref_items(items, plural);
        plural = plural || doc.is(p - 1, "refs");
        std::string label = plural ? "references " : "reference ";
        if (doc.text[doc[p - 1].start] == 'R') label[0] = 'R';
        out.tokens(p - 1, last, label + words, 75);
        return last;
    }
    if (REF_CUES.contains(prev)) {
        // "[3] and [5]", "[5], [6] and [9]", "[2][4]"
        std::vector<std::pair<uint32_t, uint32_t>> items;
        size_t last = chain(doc, g, items);
        bool plural;
        std::string words = ref_items(items, plural);
        out.tokens(i, last, (plural ? "references " : "reference ") + words, 70);
        return last;
    }
    // a citation that starts a sentence in running text and is its subject: leave it ("et al. [14] as" goes on)
    bool abbreviation = doc.punct(p, '.') && p > 0 && doc.word(p - 1) && NOT_SENTENCE_END.contains(doc.low(p - 1));
    if (spaced && doc.sentence_start(i) && next_word && !abbreviation) {
        return g.close;
    }
    // decorations go with the connectors between them: "robust [1], [2, 3] and [4–6]." -> robust.
    std::vector<std::pair<uint32_t, uint32_t>> items;
    size_t last = chain(doc, g, items);
    bool ranged = false;
    for (auto [a, b] : items) ranged = ranged || a != b;
    out.add(lead(doc, i), doc[last].end, "", ranged ? 67 : 60);  // above "5–7" read as a score
    return last;
}

// "(14)" or "(1, 2)" closing a clause is a citation: "in most patients (14)." -> in most patients.
size_t paren_citation_at(const Doc& doc, Out& out, size_t i) {
    if (i < 1) return i;
    if (!(i + 2 < doc.size())) return i;
    if (!doc.punct(i, '(') || !doc[i].space_before || doc[i].newline_before) return i;
    size_t k = i + 1;
    bool ok = true;
    for (int guard = 0; guard < 20; ++guard) {
        if (!doc.number(k) || doc.str(k).size() > 3 || num(doc, k) == 0) {
            ok = false;
            break;
        }
        if (dash_tok(doc, k + 1) && doc.number(k + 2) && doc.str(k + 2).size() <= 3) k += 2;
        if (doc.punct(k + 1, ')')) break;
        if (!doc.punct(k + 1, ',')) {
            ok = false;
            break;
        }
        k += 2;
    }
    if (!ok) return i;
    size_t close = k + 1, next = close + 1;
    size_t p = i - 1;
    // what follows: the end of the clause, or a word that goes on with the sentence after a noun ("rated it 4.4
    // on usability (6) and was"), but not "(1) and (2)"
    bool ends = next >= doc.size() || doc[next].newline_before || superscript(doc, next) ||
                (punct_tok(doc, next) && (doc[next].cp == '.' || doc[next].cp == ',' || doc[next].cp == ';' ||
                                          doc[next].cp == ':' || doc[next].cp == '!' || doc[next].cp == '?'));
    bool noun_before = lower(doc, p) && doc.str(p).size() >= 3 && !BEFORE_LIST.contains(doc.low(p)) &&
                       !PAREN_LABELS.contains(doc.low(p));
    bool figure_before = doc.punct(p, '%') || (year_like(doc, p) && doc[p].space_before);  // "by 40% (2, 4) in"
    bool goes_on = !ends && doc.word(next) && doc[next].space_before && lower(doc, next) &&
                   CLAUSE_GOES_ON.contains(doc.low(next)) && !doc.punct(next + 1, '(') && (noun_before || figure_before);
    if ((!ends && !goes_on) || (doc.punct(next, '.') && doc.number(next + 1) && doc.joined(next + 1))) return i;
    if (doc.word(p) && PAREN_LABELS.contains(doc.low(p)) && !(lower(doc, p) && WEAK_LABELS.contains(doc.low(p))))
        return i;
    if (doc.punct(p, '.') && p > 0 && doc.word(p - 1) && doc[p - 1].end - doc[p - 1].start <= 4) return i;  // Eq. (5)
    // after a number or a percent sign the clause ends too: "in late 2019 (7).", "by 30% (2, 4),"
    bool numeric = (doc.number(p) && doc[i].space_before) || doc.punct(p, '%');
    if (!(doc.word(p) || numeric || doc.punct(p, ']') || doc.punct(p, ')') || superscript(doc, p) ||
          doc.punct(p, '\'') || doc.punct(p, '"') || doc[p].cp == 0x201D || doc[p].cp == 0x2019))
        return i;
    out.add(lead(doc, i), doc[close].end, "", 60);
    return close;
}

// Superscript and caret footnote numbers after a word or the punctuation after it: "the results¹²",
// "was confirmed.^3", "the claim[^1]". A superscript on a number, a unit or a variable is a power.
size_t superscript_citation_at(const Doc& doc, Out& out, size_t i) {
    if (i < 1) return i;
    if (!punct_tok(doc, i)) return i;
    size_t p = i - 1;
    // a lone superscript digit from 2 up is a power ("m²", "x⁴"); "¹" or a run of digits on a short word is a
    // marker ("180 ms¹", "43.8 s¹⁶")
    bool short_ok = superscript(doc, i) && !(superscript_digit(doc[i].cp) >= 2 && !superscript(doc, i + 1));
    auto attached_ok = [&](size_t q) {
        if (doc.word(q)) {
            size_t len = doc[q].end - doc[q].start;
            return (len >= 3 && !UNIT_WORDS.contains(doc.low(q))) || short_ok;
        }
        // after a number only a lone "¹" is a marker ("on January 20, 2026¹."); "10²" is a power, but a long
        // number other than a power of ten carries a marker: "ISO 50001³"
        if (doc.number(q)) {
            std::string_view s = doc.str(q);
            bool long_number = superscript(doc, i) && s.size() >= 4 && s.find_first_not_of('0', 1) != std::string_view::npos;
            return (doc[i].cp == 0xB9 && !superscript(doc, i + 1)) || long_number;
        }
        if (!punct_tok(doc, q)) return false;
        uint32_t c = doc[q].cp;
        return c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?' || c == ')' || c == ']' ||
               c == '"' || c == '\'' || c == 0x201D || c == 0x2019 || c == 0x2020 || c == 0x2021 || c == '*' ||
               c == '%';
    };
    auto free_after = [&](size_t n) {
        if (n >= doc.size() || doc[n].space_before) return true;
        if (doc.word(n) || doc.number(n)) return false;
        return !(doc.punct(n, '.') && doc.number(n + 1) && doc.joined(n + 1));
    };
    if (superscript(doc, i) && !superscript(doc, p) && doc.joined(i) && attached_ok(p)) {
        size_t j = i;
        // a run of markers, with ranges and lists inside: "¹²", "¹⁻³", "²,⁵"
        for (;;) {
            if (superscript(doc, j + 1) && doc.joined(j + 1)) ++j;
            else if ((doc.punct(j + 1, 0x207B) || doc.punct(j + 1, ',') || dash_tok(doc, j + 1)) && doc.joined(j + 1) &&
                     superscript(doc, j + 2) && doc.joined(j + 2))
                j += 2;
            else break;
        }
        if (free_after(j + 1)) out.add(doc[i].start, doc[j].end, "", 62);
        return j;
    }
    // caret: "summaries.^12"
    if (doc.punct(i, '^') && doc.joined(i) && doc.number(i + 1) && doc.joined(i + 1) && attached_ok(p) &&
        !doc.punct(p, '[') && free_after(i + 2)) {
        out.add(doc[i].start, doc[i + 1].end, "", 62);
        return i + 1;
    }
    // markdown footnote reference: "[^1]"
    if (doc.punct(p, '[') && doc.punct(i, '^') && doc.joined(i) && doc.number(i + 1) && doc.joined(i + 1) &&
        doc.punct(i + 2, ']') && doc.joined(i + 2) && p > 0 && !line_start(doc, p)) {
        out.add(lead(doc, p), doc[i + 2].end, "", 62);
        i += 2;
    }
    return i;
}

// Footnote symbols at the end of a word ("revenue‡", "default*.") and the leading mark of a footnote or legend line
// ("† Adjusted for inflation.", "*IGR = ..."). Asterisks that pair up as emphasis, bold, masked words and
// multiplications are left to other rules.
RV_RULE("roman_citations", footnote_symbol) {
    bool any = false;
    for (size_t i = 0; i < doc.size() && !any; ++i) {
        uint32_t c = doc[i].cp;
        any = punct_tok(doc, i) && (c == '*' || c == 0x2020 || c == 0x2021 || c == 0xA7 || c == 0xB6 ||
                                    superscript_digit(c) >= 0);
    }
    if (!any) return;
    // a dagger named in the text is read: "See footnote †", "marked with a †", "a small '†'"
    static const rv::WordSet NAMES_IT{"a", "footnote", "in", "marked", "see", "symbol", "the", "with"};
    for (size_t i = 1; i < doc.size(); ++i) {
        if (!(doc.punct(i, 0x2020) || doc.punct(i, 0x2021))) continue;
        size_t p = i - 1;
        bool quoted = punct_tok(doc, p) && (doc[p].cp == '\'' || doc[p].cp == '"' || doc[p].cp == 0x2018 ||
                                            doc[p].cp == 0x201C) && doc.joined(i) && doc[p].space_before &&
                      punct_tok(doc, i + 1) && doc.joined(i + 1);
        if (quoted) p = p > 0 ? p - 1 : p;
        bool named = (doc[i].space_before || quoted) && doc.word(p) && (NAMES_IT.contains(doc.low(p)) || quoted);
        if (named) out.tokens(i, i, doc.punct(i, 0x2020) ? "dagger" : "double dagger", 70);
    }
    // pair emphasis asterisks per line: an opener ("*word") closed later on the line by a closer ("word*")
    std::vector<char> role(doc.size(), 0);  // 'o' opener, 'c' closer, 'e' emphasis (paired), 'x' ignore
    std::vector<char> in_code(doc.size(), 0);  // inside `inline code` on the line
    std::vector<size_t> open_stack;
    bool code = false;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].newline_before) open_stack.clear(), code = false;
        if (doc.punct(i, '`')) code = !code;
        in_code[i] = code;
        if (!doc.punct(i, '*')) continue;
        bool star_prev = i > 0 && doc.punct(i - 1, '*') && doc.joined(i);
        bool star_next = doc.punct(i + 1, '*') && doc.joined(i + 1);
        if (star_prev || star_next) {
            role[i] = 'x';
            continue;
        }
        bool left_open = i == 0 || doc[i].space_before || doc[i].newline_before ||
                         (punct_tok(doc, i - 1) && (doc[i - 1].cp == '(' || doc[i - 1].cp == '"' ||
                                                    doc[i - 1].cp == 0x201C || doc[i - 1].cp == '['));
        bool right_open = i + 1 >= doc.size() || doc[i + 1].space_before || doc[i + 1].newline_before;
        if (!left_open && !right_open) {
            // "sh*t", "3*4", or a closer followed by punctuation: "default*."
            bool punct_next = punct_tok(doc, i + 1) && !doc.punct(i + 1, '*');
            role[i] = punct_next ? 'c' : 'x';
        } else if (left_open && !right_open) {
            role[i] = 'o';
        } else if (!left_open && right_open) {
            role[i] = 'c';
        } else {
            role[i] = 'l';  // lone "*" with spaces on both sides
        }
        if (role[i] == 'o') open_stack.push_back(i);
        else if (role[i] == 'c' && !open_stack.empty()) {
            role[open_stack.back()] = 'e';
            role[i] = 'e';
            open_stack.pop_back();
        }
    }
    for (size_t i = 0; i < doc.size(); ++i) {
        uint32_t c = doc[i].cp;
        bool dagger = punct_tok(doc, i) && (c == 0x2020 || c == 0x2021);
        bool star = doc.punct(i, '*') && role[i] != 'x' && role[i] != 'e';
        bool sup_lead = superscript(doc, i);
        bool section_lead = punct_tok(doc, i) && (c == 0xA7 || c == 0xB6) && line_start(doc, i) && doc.word(i + 1);
        if (!dagger && !star && !sup_lead && !section_lead) continue;
        if (line_start(doc, i)) {
            // leading mark of a footnote line, with the space after it
            size_t n = i + 1;
            if (sup_lead) {
                while (superscript(doc, n) && doc.joined(n)) ++n;
                if (n >= doc.size() || !doc[n].space_before || doc[n].newline_before || !doc.word(n)) continue;
            }
            if (n >= doc.size() || doc[n].newline_before) continue;
            if (!(doc.word(n) || doc.number(n) || doc.punct(n, '(') || doc.punct(n, '"'))) continue;
            out.add(doc[i].start, doc[n].start, "", 64);
            continue;
        }
        if (sup_lead) continue;
        // a dagger set off by a space before the end of the sentence: "on page 47 †."
        if (dagger && !doc.joined(i) && i > 0 && !doc[i].newline_before &&
            (i + 1 >= doc.size() || doc[i + 1].newline_before || doc.punct(i + 1, '.') || doc.punct(i + 1, ','))) {
            out.add(doc[i - 1].end, doc[i].end, "", 64);
            continue;
        }
        // attached to the end of a word, number or closing punctuation, free after
        if (!doc.joined(i)) continue;
        size_t p = i - 1;
        bool attached = doc.word(p) || doc.number(p) || doc.punct(p, '.') || doc.punct(p, ')') ||
                        doc.punct(p, ']') || doc.punct(p, '\'') || doc.punct(p, '"') || doc[p].cp == 0x201D ||
                        doc[p].cp == 0x2019 || doc[p].cp == '%';
        if (!attached) continue;
        size_t n = i + 1;
        bool free = n >= doc.size() || doc[n].space_before;
        if (!free && punct_tok(doc, n)) {
            uint32_t q = doc[n].cp;
            bool closing = q == '.' || q == ',' || q == ';' || q == ':' || q == '!' || q == '?' || q == ')' ||
                           q == ']' || q == '"' || q == '\'' || q == 0x201D || q == 0x2019;
            // "draft*.docx", "4.2*.5": a file pattern or a number goes on
            bool goes_on = n + 1 < doc.size() && doc.joined(n + 1) && !punct_tok(doc, n + 1);
            free = closing && !goes_on;
        }
        if (!free) continue;
        if (star && role[i] != 'c') continue;
        if (star && in_code[i]) continue;
        out.add(doc[i].start, doc[i].end, "", 64);
    }
}

// Inline maintenance tags of wikis: "the claim[citation needed]." -> the claim.
const rv::WordSet WIKI_TAGS{"better source needed", "by whom?", "citation needed", "clarification needed",
                            "dead link", "failed verification", "page needed", "verification needed", "when?",
                            "which?", "who?"};

// "[sic]" and wiki tags are removed with the space before them.
size_t sic_marker_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 2 < doc.size())) return i;
    if (doc.punct(i, '[') && doc.word(i + 1) && !doc.is(i + 1, "sic")) {
        uint32_t s = doc[i + 1].start;
        size_t k = i + 1;
        while (k < doc.size() && k < i + 5 && !doc.punct(k, ']') && !doc[k].newline_before) ++k;
        if (doc.punct(k, ']') && WIKI_TAGS.contains(std::string_view(doc.lower).substr(s, doc[k].start - s))) {
            out.add(lead(doc, i), doc[k].end, "", 85);
            return k;
        }
        return i;
    }
    if (!doc.punct(i, '[') || !doc.is(i + 1, "sic")) return i;
    size_t close = i + 2;
    if (doc.punct(close, '!')) ++close;
    if (!doc.punct(close, ']')) return i;
    out.add(lead(doc, i), doc[close].end, "", 85);
    return close;
}

// Reference-list entries: "[3] Smith — Title · Notes" and "4. Kowalski — Notes | Papers" ->
// Reference three, Smith, Title, Notes. The separators become commas and the entry ends with a period.
RV_RULE("roman_citations", reference_entry) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!line_start(doc, i)) continue;
        size_t end = i + 1;  // end of the line (exclusive)
        bool separator = false;
        while (end < doc.size() && !doc[end].newline_before) {
            uint32_t c = doc[end].cp;
            if (punct_tok(doc, end) && (c == 0x2014 || c == 0xB7 || c == '|') && doc[end].space_before) separator = true;
            ++end;
        }
        size_t body;
        uint32_t n;
        if (doc.punct(i, '[') && doc.number(i + 1) && doc.str(i + 1).size() <= 3 && doc.punct(i + 2, ']') &&
            i + 3 < end && doc[i + 3].space_before) {
            n = num(doc, i + 1);
            body = i + 3;
            out.tokens(i, i + 2, "Reference " + rv::cardinal(n) + ",", 75);
        } else if (doc.number(i) && doc.str(i).size() <= 3 && doc.punct(i + 1, '.') && doc.joined(i + 1) &&
                   i + 2 < end && doc[i + 2].space_before && doc.word(i + 2) && separator) {
            n = num(doc, i);
            body = i + 2;
            out.tokens(i, i + 1, "Reference " + rv::cardinal(n) + ",", 75);
        } else {
            i = end - 1;
            continue;
        }
        if (n == 0) {
            i = end - 1;
            continue;
        }
        for (size_t k = body; k < end; ++k) {
            uint32_t c = doc[k].cp;
            if (punct_tok(doc, k) && (c == 0x2014 || c == 0xB7 || c == '|') && doc[k].space_before && k > body)
                out.add(doc[k - 1].end, doc[k].end, ",", 70);
        }
        // a final period where the entry has no closing punctuation: on the line break when one follows (no other
        // rule reads whitespace), else on the last word
        size_t last = end - 1;
        uint32_t lc = doc[last].cp;
        bool closed = punct_tok(doc, last) && (lc == '.' || lc == '!' || lc == '?' || lc == ':');
        if (last > body && !closed && end < doc.size()) {
            uint32_t gap = doc[last].end, next = doc[end].start;
            out.add(gap, next, "." + std::string(doc.slice(gap, next)), 50);
        } else if (last > body && !closed) {
            if (doc.word(last) && doc.joined(last)) {
                // after a word that ends an address another rule reads: "tontaube.ai" -> tontaube dot AI.
                out.add(doc[last].end, doc[last].end, ".", 50);
            } else if (doc.word(last) && doc[last].casing != rv::UPPER) {
                out.tokens(last, last, std::string(doc.str(last)) + ".", 50);
            } else if (doc.number(last) && doc[last].space_before && !(last > 0 && doc.joined(last))) {
                std::string w = year_like(doc, last) ? year_words(doc, last) : plain(doc, last);
                if (!w.empty()) out.tokens(last, last, w + ".", 50);
            }
        }
        i = end - 1;
    }
}

// arXiv, PMID, ISBN and ISSN numbers digit by digit: "arXiv:2609.08703" -> Arxiv two six zero nine dot zero
// eight seven zero three; "ISBN 978-3-16-148410-0" -> ISBN nine seven eight, three, one six, ...
size_t catalogue_number_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 1 < doc.size())) return i;
    if (!doc.word(i)) return i;
    char f = first_lower(doc, i);
    if (f != 'a' && f != 'p' && f != 'i') return i;
    std::string_view w = doc.low(i);
    bool arxiv = w == "arxiv", pmid = w == "pmid", isbn = w == "isbn", issn = w == "issn";
    if (!arxiv && !pmid && !isbn && !issn) return i;
    if (!arxiv && doc[i].casing != rv::UPPER) return i;
    size_t k = i + 1;
    if (doc.punct(k, ':') && doc.joined(k)) ++k;
    std::string label = arxiv ? "Arxiv" : pmid ? "PMID" : isbn ? "ISBN" : "ISSN";
    if (isbn && dash_tok(doc, k) && doc.joined(k) && doc.number(k + 1) && doc.joined(k + 1)) {
        label += " " + plain(doc, k + 1);  // "ISBN-13:" -> ISBN thirteen
        k += 2;
        if (doc.punct(k, ':') && doc.joined(k)) ++k;
    }
    if (!doc.number(k)) return i;
    std::string words;
    size_t last = k;
    if (arxiv) {
        if (!(doc.punct(k + 1, '.') && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2))) return i;
        words = rv::digits(doc.str(k)) + " dot " + rv::digits(doc.str(k + 2));
        last = k + 2;
        if (doc.word(last + 1) && doc.joined(last + 1) && doc.is(last + 1, "v") && doc.number(last + 2) &&
            doc.joined(last + 2)) {
            words += " V " + plain(doc, last + 2);
            last += 2;
        }
    } else if (pmid) {
        words = rv::digits(doc.str(k));
    } else {
        words = rv::digits(doc.str(k));
        while (dash_tok(doc, last + 1) && doc.joined(last + 1) && doc.joined(last + 2) &&
               (doc.number(last + 2) || doc.is(last + 2, "x"))) {
            words += ", " + (doc.number(last + 2) ? rv::digits(doc.str(last + 2)) : std::string("X"));
            last += 2;
        }
        if (doc.is(last + 1, "x") && doc.joined(last + 1)) {  // check digit joined: "0-306-40615-X"
            words += " X";
            last += 1;
        }
    }
    out.tokens(i, last, label + " " + words, 85);
    return last;
}

// DOIs: the label is read "DOI" and the identifier is copied unchanged: "doi:10.1038/s41586-021" ->
// DOI 10.1038/s41586-021.
size_t doi_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 1 < doc.size())) return i;
    if (!doc.word(i) || first_lower(doc, i) != 'd' || !doc.is(i, "doi")) return i;
    size_t k = i + 1;
    bool colon = doc.punct(k, ':') && doc.joined(k);
    if (colon) ++k;
    if (!doc.number(k) || doc.str(k) != "10" || !doc.punct(k + 1, '.') || !doc.joined(k + 1)) return i;
    if (!colon && !doc[k].space_before) return i;
    // the identifier runs to the next space, without closing punctuation
    uint32_t s = doc[k].start, e = s;
    const std::string_view t = doc.text;
    while (e < t.size() && t[e] != ' ' && t[e] != '\n' && t[e] != '\t' && t[e] != '\r') ++e;
    int depth = 0;
    for (uint32_t b = s; b < e; ++b) depth += t[b] == '(' ? 1 : t[b] == ')' ? -1 : 0;
    while (e > s) {
        char c = t[e - 1];
        if (c == '.' || c == ',' || c == ';' || c == ':' || c == '"' || c == '\'' || c == ']') --e;
        else if (c == ')' && depth < 0) --e, ++depth;
        else break;
    }
    if (e <= s + 3) return i;
    size_t label_end = colon ? i + 1 : i;
    if (i > 0 && doc.is(i - 1, "doi") && doc[i].space_before && !doc[i].newline_before)
        out.add(doc[i].start, doc[label_end].end, "", 90);  // "the DOI doi:10.1145/..." says the label once
    else
        out.tokens(i, label_end, "DOI", 90);
    out.add(s, e, std::string(t.substr(s, e - s)), 95);
    return doc.token_at(e) - 1;
}

// ---------------------------------------------------------------------------------------------------------------
// Author-year citations.

// Inside "(...)" from token o: "Smith et al., 2020", "Varga and Lind, 2019; Moreau et al., 2021", optionally
// with "e.g.," or "see" before and ", p. 45" after a year. Returns the closing token, npos if not a citation.
size_t author_year_paren(const Doc& d, size_t o) {
    size_t k = o + 1;
    if (d.is(k, "e") && d.punct(k + 1, '.') && d.is(k + 2, "g") && d.punct(k + 3, '.')) {
        k += 4;
        if (d.punct(k, ',')) ++k;
    } else if (d.is(k, "see") || d.is(k, "cf")) {
        ++k;
        if (d.punct(k, '.')) ++k;
    }
    for (int seg = 0; seg < 8; ++seg) {
        // author part
        if (!d.word(k) || d.text[d[k].start] < 'A' || d.text[d[k].start] > 'Z' || NOT_AUTHOR.contains(d.low(k)))
            return npos;
        int words = 0;
        while (k < d.size() && words < 14) {
            if (d.word(k) && (d.text[d[k].start] >= 'A' && d.text[d[k].start] <= 'Z')) {
                ++words;
                ++k;
            } else if (d.word(k) && PARTICLES.contains(d.low(k))) {
                ++k;
            } else if (punct_tok(d, k) && (d[k].cp == '.' || d[k].cp == ',' || d[k].cp == '&' || dash(d[k].cp) ||
                                          d[k].cp == '\'' || d[k].cp == 0x2019)) {
                ++k;
            } else {
                break;
            }
        }
        if (!words || !year_like(d, k)) return npos;
        ++k;
        if (single_letter(d, k) && d.joined(k)) ++k;  // 2020a
        // more years of the same authors: "Garcia et al., 2019a, 2019b"
        while (d.punct(k, ',') && year_like(d, k + 1)) {
            k += 2;
            if (single_letter(d, k) && d.joined(k)) ++k;
        }
        if (d.punct(k, ',') && (d.is(k + 1, "p") || d.is(k + 1, "pp")) && d.punct(k + 2, '.') && d.number(k + 3)) {
            k += 4;
            if (dash_tok(d, k) && d.number(k + 1)) k += 2;
        }
        if (d.punct(k, ')')) return k;
        if (!d.punct(k, ';')) return npos;
        ++k;
    }
    return npos;
}

// "(Smith et al., 2020)" is removed with the space before it.
size_t author_year_parenthetical_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 3 < doc.size())) return i;
    if (!doc.punct(i, '(')) return i;
    size_t close = author_year_paren(doc, i);
    if (close == npos) return i;
    out.add(lead(doc, i), doc[close].end, "", 82);  // above its parts ("e.g.", "p. 45")
    return close;
}

// Narrative citations keep the author and set the year off with commas: "Okafor et al. (2021) found" ->
// Okafor and others, twenty twenty-one, found; "Miller (2005, p. 45)," -> Miller, two thousand five, page
// forty-five,.
size_t author_year_narrative_at(const Doc& doc, Out& out, size_t o) {
    if (o < 1) return o;
    if (!(o + 2 < doc.size())) return o;
    if (!doc.punct(o, '(') || !doc[o].space_before || doc[o].newline_before || !year_like(doc, o + 1)) return o;
    size_t k = o + 2;
    std::string year = year_words(doc, o + 1);
    if (char c = single_letter(doc, k); c && doc.joined(k)) {
        year += " " + letter_cap(c);
        ++k;
    }
    std::string page;
    if (doc.punct(k, ',') && (doc.is(k + 1, "p") || doc.is(k + 1, "pp")) && doc.punct(k + 2, '.') &&
        doc.number(k + 3)) {
        bool pp = doc.is(k + 1, "pp");
        page = std::string(", ") + (pp ? "pages " : "page ") + plain(doc, k + 3);
        k += 4;
        if (dash_tok(doc, k) && doc.number(k + 1)) {
            page += " to " + plain(doc, k + 1);
            k += 2;
        }
    }
    if (!doc.punct(k, ')')) return o;
    size_t close = k, p = o - 1;
    // a law report: "410 U.S. 113 (1973)" -> ..., nineteen seventy-three
    if (page.empty() && doc.number(p) && doc[o].space_before && doc[p].space_before && p > 0 && doc.punct(p - 1, '.') &&
        doc.joined(p - 1)) {
        size_t n = after_markers(doc, close);
        bool comma = n < doc.size() && !doc[n].newline_before && (doc.word(n) || doc.number(n));
        out.add(doc[p].end, doc[close].end, ", " + year + (comma ? "," : ""), 82);
        return close;
    }
    // "**Okafor** (2019)": the name ends before its emphasis marks
    while (p > 0 && (doc.punct(p, '*') || doc.punct(p, '_')) && doc.joined(p)) --p;
    bool etal = false;
    size_t name;
    if (p >= 3 && doc.punct(p, '.') && doc.is(p - 1, "al") && doc.is(p - 2, "et")) {
        etal = true;
        name = p - 3;
        if (doc.punct(name, ',') && name > 0) --name;  // "Smith, et al. (2022)"
    } else if (p >= 2 && (doc.is(p, "colleagues") || doc.is(p, "coworkers")) && doc.is(p - 1, "and")) {
        name = p - 2;
    } else {
        name = p;
    }
    if (name >= doc.size() || !doc.word(name) || doc.text[doc[name].start] < 'A' ||
        doc.text[doc[name].start] > 'Z' || doc[name].casing == rv::UPPER)
        return o;
    if (NOT_AUTHOR.contains(doc.low(name))) return o;
    // "Treaty of Veldmark (1763)", "The Godfather (1972)", "1. The Ash Crown (1994)": a named thing, not an author
    for (size_t b = name, guard = 0; guard < 6; --b, ++guard) {
        if (doc.is(b, "the") || doc.is(b, "a") || doc.is(b, "an")) return o;
        if (b == 0 || !(title(doc, b - 1) || doc.is(b - 1, "the") || doc.is(b - 1, "a") || doc.is(b - 1, "an")) ||
            doc[b].newline_before)
            break;
    }
    if (name > 0) {
        size_t b = name - 1;
        if (doc.is(b, "of") && b > 0 && title(doc, b - 1)) return o;
    }
    // "Smith (2022), Jones (2021), etc.": a list of sources keeps its brackets
    if (doc.punct(close + 1, ',') && doc.word(close + 2) && doc.punct(close + 3, '(') && year_like(doc, close + 4))
        return o;
    if (name > 1 && doc.punct(name - 1, ',') && doc.punct(name - 2, ')')) return o;
    size_t n = after_markers(doc, close);
    bool comma = n < doc.size() && !doc[n].newline_before && (doc.word(n) || doc.number(n));
    std::string words = etal ? "and others, " + year : ", " + year;
    words += page;
    if (comma) words += ",";
    uint32_t start = etal ? doc[p - 2].start : doc[p].end;
    out.add(start, doc[close].end, words, 82);
    return close;
}

// "et al." -> and others; "ibid." -> ibidem. The period stays where it also ends the sentence.
size_t et_al_ibid_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i) || (first_lower(doc, i) != 'e' && first_lower(doc, i) != 'i')) return i;
    // "et. al." written with a period too many
    if (doc.is(i, "et") && lower(doc, i) && doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.is(i + 2, "al") &&
        doc[i + 2].space_before) {
        size_t last = i + 2;
        if (doc.punct(i + 3, '.') && doc.joined(i + 3) && !period_ends_sentence(doc, i + 3)) last = i + 3;
        out.tokens(i, last, "and others", 70);
        return last;
    }
    if (doc.is(i, "et") && doc.is(i + 1, "al") && lower(doc, i)) {
        size_t last = i + 1;
        if (doc.punct(i + 2, '.') && doc.joined(i + 2) && !period_ends_sentence(doc, i + 2)) last = i + 2;
        // a possessive: "Smith et al.'s study" -> Smith and others' study
        size_t a = last + 1;
        if (doc.punct(last, '.') && (doc.punct(a, '\'') || doc.punct(a, 0x2019)) && doc.joined(a) && doc.is(a + 1, "s") &&
            doc.joined(a + 1)) {
            out.tokens(i, a + 1, "and others" + std::string(doc.str(a)), 70);
            return a + 1;
        }
        out.tokens(i, last, "and others", 70);
        i = last;
    } else if (doc.is(i, "ibid")) {
        size_t last = i;
        if (doc.punct(i + 1, '.') && doc.joined(i + 1) && !period_ends_sentence(doc, i + 1)) last = i + 1;
        out.tokens(i, last, doc.text[doc[i].start] == 'I' ? "Ibidem" : "ibidem", 70);
        i = last;
    }
    return i;
}

// One scan for the citation rules that start at a single token: brackets, parentheses, superscripts and the words
// et al., ibid., arXiv, PMID, ISBN, ISSN and DOI. Each rule resumes after the tokens it read.
RV_RULE("roman_citations", citation_marker) {
    enum { BRACKET, SIC, PAREN, AUTHOR_PAREN, AUTHOR_NARRATIVE, SUPERSCRIPT, ET_AL, CATALOGUE, DOI, COUNT };
    size_t next[COUNT] = {};
    auto run = [&](int k, size_t (*fn)(const Doc&, Out&, size_t), size_t i) {
        if (i >= next[k]) next[k] = fn(doc, out, i) + 1;
    };
    for (size_t i = 0; i < doc.size(); ++i) {
        if (punct_tok(doc, i)) {
            if (doc[i].cp == '[' || doc[i].cp == 0xFF3B) {
                run(BRACKET, bracket_citation_at, i);
                run(SIC, sic_marker_at, i);
            } else if (doc[i].cp == '(') {
                run(PAREN, paren_citation_at, i);
                run(AUTHOR_PAREN, author_year_parenthetical_at, i);
                run(AUTHOR_NARRATIVE, author_year_narrative_at, i);
            }
            run(SUPERSCRIPT, superscript_citation_at, i);
            continue;
        }
        if (!doc.word(i)) continue;
        switch (first_lower(doc, i)) {
            case 'e': run(ET_AL, et_al_ibid_at, i); break;
            case 'i': run(ET_AL, et_al_ibid_at, i); run(CATALOGUE, catalogue_number_at, i); break;
            case 'a': case 'p': run(CATALOGUE, catalogue_number_at, i); break;
            case 'd': run(DOI, doi_at, i); break;
            default: break;
        }
    }
}

}  // namespace
