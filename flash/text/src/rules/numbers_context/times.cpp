// "x" and "×" with numbers: dimensions "by" (a 4 x 6 card, 1920×1080, 2 × 3 m), multiplication "times"
// (76 x 4 = 304, the formula 8 x 15, 2 x 500 ml) and multipliers (6x louder, 2.5× faster, a 10X return, 2-3x,
// 2x/day, ×10); foot and inch marks (4' x 8', a 27" monitor).
// Model names (Ryzen 7 5800X), hex (0x80070002), algebra (2x = 20) and scientific notation (1 x 10^3) are left
// to their owners.
#include "rules/numbers_context/common.h"

namespace {

using namespace nctx;

const rv::WordSet DIM_CUES{
    "array", "bay", "board", "canvas", "card", "cards", "cell", "cm", "display", "feet", "floor", "foot", "frame",
    "frames", "ft", "grid", "image", "in", "inch", "inches", "kernel", "layout", "lumber", "m", "map", "matrix",
    "meters", "metres", "mm", "monitor", "mosaic", "pan", "panel", "panels", "photo", "picture", "pixel",
    "pixels", "plot", "poster", "preview", "print", "px", "resolution", "room", "rug", "screen", "sheet", "sign",
    "tile", "tiles", "wall", "yards"};

// Unit words after the last part of a dimension; the unit's own reading (units_symbols) takes that number.
const rv::WordSet UNIT_AFTER{"cm", "ft", "in", "km", "m", "mi", "mm", "yd", "px", "nm", "um", "dm"};
const rv::PhraseMap UNIT_WORDS{{"cm", "centimeter"}, {"ft", "foot"}, {"in", "inch"}, {"km", "kilometer"},
                               {"m", "meter"}, {"mi", "mile"}, {"mm", "millimeter"}, {"yd", "yard"}, {"px", "pixel"},
                               {"nm", "nanometer"}, {"dm", "decimeter"}};

const rv::WordSet TIMES_CUES{"arithmetic", "calculate", "calculated", "calculating", "calculation", "compute",
                             "computed", "equation", "formula", "math", "multiplied", "multiply", "multiplying",
                             "solve", "what's"};

// Units of a counted quantity, not of a length: "2 x 500 ml bottles", "3 x 2TB drives" are multiplications.
const rv::WordSet COUNTED_UNITS{"cl", "dl", "g", "gb", "kb", "kg", "l", "lb", "lbs", "mah", "mb", "mg", "ml",
                                "oz", "pb", "tb", "w", "kw", "v", "tablets", "capsules", "caps", "tabs"};

// Quantities multiplied in a formula written with words: "length x width", "rate x time".
const rv::WordSet FORMULA_WORDS{"base", "breadth", "depth", "distance", "height", "length", "mass", "price",
                                "principal", "quantity", "rate", "speed", "time", "width"};

// Words after "x" with a number that make it a rate: "2x/day", "3x/week".
const rv::WordSet PER_TIME{"day", "d", "week", "wk", "month", "mo", "year", "yr", "hour", "hr", "h", "night",
                           "session", "dose"};
const rv::PhraseMap PER_TIME_WORD{{"d", "day"}, {"wk", "week"}, {"mo", "month"}, {"yr", "year"}, {"hr", "hour"},
                                  {"h", "hour"}};

const rv::WordSet COMPARATIVE{
    "bigger", "brighter", "cheaper", "cleaner", "costlier", "denser", "faster", "fewer", "greater", "harder",
    "heavier", "higher", "larger", "less", "lighter", "longer", "louder", "lower", "more", "quicker", "quieter",
    "safer", "shorter", "slower", "smaller", "smoother", "stronger", "the", "as", "speedup", "speed-up",
    "increase", "improvement", "gain", "return", "growth", "zoom", "magnification", "boost", "reduction",
    "better", "worse", "over", "what", "that", "their", "its", "our"};

bool times_op(const Doc& d, size_t j) {
    if (j >= d.size()) return false;
    if (d.word(j)) return d.low(j) == "x";
    return d.punct(j, 0xD7) || d.punct(j, 0x2715) || d.punct(j, 0x2716);
}

bool power_after(const Doc& d, size_t last) {
    size_t j = last + 1;
    if (!d.joined(j)) return false;
    uint32_t c = d[j].cp;
    return d[j].kind == Kind::Punct && (c == '^' || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0x2070 && c <= 0x209F));
}

// The operator at j joins the numbers on both sides: "4x6", "4 x 6", "2×4", "2 × 4", "2 ×4".
bool op_between(const Doc& d, size_t j) {
    bool before = d[j].space_before, after = d[j + 1].space_before;
    if (d.word(j)) return before == after;
    return true;
}

// Feet and inch marks in a dimension: "4'x8'" four foot by eight foot (before a noun), "12' x 14'" twelve feet
// by fourteen feet, "12″ × 18″".
char dim_mark(const Doc& d, size_t last) {
    size_t j = last + 1;
    if (j >= d.size() || !d.joined(j) || d[j].kind != Kind::Punct) return 0;
    uint32_t c = d[j].cp;
    return c == '\'' || c == 0x2032 || c == 0x2019 ? 'f' : c == '"' || c == 0x2033 || c == 0x201D ? 'i' : 0;
}

// Returns the last token read (e when nothing was read).
size_t marked_dimension(const Doc& doc, rv::Out& out, size_t i, size_t e) {
    char m0 = dim_mark(doc, e);
    Num parts[3];
    char marks[3];
    size_t n = 0, j = i;
    parts[n] = num_at(doc, i);
    marks[n++] = m0;
    if (!free_left(doc, i) || parts[0].whole.empty() ||
        (m0 == 'i' && doc.punct(e + 1, '"') && quote_open_before(doc, e + 1, '"')))
        return e;
    j = e + 2;  // the sign
    bool ok = true;
    while (n < 3 && times_op(doc, j) && doc.number(j + 1)) {
        bool letter = doc.word(j);
        if (letter && doc[j].space_before != doc[j + 1].space_before) break;
        Num b = num_at(doc, j + 1);
        if (b.whole.empty()) {
            ok = false;
            break;
        }
        char m = dim_mark(doc, b.last);
        parts[n] = b;
        marks[n++] = m;
        j = b.last + 1 + (m ? 1 : 0);
        if (!m) break;
    }
    size_t last = j - 1;
    if (!ok || n < 2 || !marks[n - 1] || (doc.joined(last + 1) && (doc.word(last + 1) || doc.number(last + 1))))
        return last;
    bool adj = adjectival_after(doc, last + 1);
    std::string words;
    for (size_t k = 0; k < n; ++k) {
        std::string w = num_words(parts[k]);
        if (w.empty()) {
            ok = false;
            break;
        }
        bool one = parts[k].whole == "1" && parts[k].frac.empty();
        bool sg = one || adj;
        words += (k ? " by " : "") + w + " " +
                 (marks[k] == 'f' ? (sg ? "foot" : "feet") : (sg ? "inch" : "inches"));
    }
    if (ok) out.tokens(i, last, words, 70);
    return last;
}

// "a 6.1" display", "the 27” monitor", "12" across": a number with an inch mark that closes no quotation.
// Returns e.
size_t inch_measure(const Doc& doc, rv::Out& out, size_t i, size_t e) {
    size_t m = e + 1;
    if (m >= doc.size() || !doc.joined(m) || !(doc.punct(m, '"') || doc.punct(m, 0x201D))) return e;
    if (doc.joined(m + 1) && (doc.word(m + 1) || doc.number(m + 1))) return e;
    if (quote_open_before(doc, m, '"') || !free_left(doc, i)) return e;
    // a closing quote has its opening one on the line; also no word right before an opening ("he said 5")
    Num n = num_at(doc, i);
    std::string w = num_words(n);
    if (!n.ok || n.last != e || w.empty() || n.grouped) return e;
    bool one = n.whole == "1" && n.frac.empty();
    bool adj = adjectival_after(doc, m + 1);
    out.tokens(i, m, w + (one || adj ? " inch" : " inches"), 60);
    return e;
}

RV_RULE("numbers_context", dimensions) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        size_t e = num_end(doc, i);
        if (!times_op(doc, e + 1)) {
            if (dim_mark(doc, e) && times_op(doc, e + 2)) i = marked_dimension(doc, out, i, e);
            else i = inch_measure(doc, out, i, e);
            continue;
        }
        Num a = num_at(doc, i);
        size_t op = a.last + 1;
        if (!times_op(doc, op) || !doc.number(op + 1) || !op_between(doc, op) || a.whole.empty()) {
            i = a.last;
            continue;
        }
        if (!free_left(doc, a.first) || (a.whole == "0" && doc.word(op) && doc.joined(op))) {
            i = a.last;
            continue;
        }
        // the parts: a x b [x c ...]
        Num parts[4];
        size_t nparts = 0;
        parts[nparts++] = a;
        size_t j = op;
        bool ok = true;
        while (nparts < 4 && times_op(doc, j) && doc.number(j + 1) && op_between(doc, j)) {
            Num b = num_at(doc, j + 1);
            if (b.whole.empty()) {
                ok = false;
                break;
            }
            parts[nparts++] = b;
            j = b.last + 1;
        }
        size_t last = parts[nparts - 1].last;
        // "3 x 2TB drives", "2 x 500 ml bottles": a count times a quantity
        if (ok && nparts == 2 && a.is_int() && a.ndigits() <= 2 && doc.word(last + 1) &&
            COUNTED_UNITS.contains(doc.low(last + 1)) && !(doc.word(a.last + 1) && doc.joined(a.last + 1) &&
                                                            !times_op(doc, a.last + 1))) {
            std::string w = num_words(a);
            if (!w.empty()) out.tokens(a.first, op, w + " times", 64);
            i = last;
            continue;
        }
        if (!ok || power_after(doc, last)) {
            // "3 x 10^8": the sign is "times"; the power is read by its own rule
            if (ok && nparts == 2 && parts[1].whole == "10" && parts[1].frac.empty() && free_left(doc, a.first)) {
                std::string w = num_words(a);
                if (!w.empty()) out.tokens(a.first, op, w + " times", 60);
            }
            i = last;
            continue;
        }
        // what follows the last part: free, or a unit word (joined or spaced)
        size_t nx = last + 1;
        bool unit_after = nx < doc.size() && doc.word(nx) && UNIT_AFTER.contains(doc.low(nx));
        // "2 x 4s": the plural of the last number (two by fours)
        if (nparts == 2 && doc.word(nx) && doc.joined(nx) && doc.str(nx) == "s" && free_right(doc, nx) &&
            parts[0].is_int() && parts[1].is_int()) {
            std::string a0 = num_words(parts[0]), a1 = num_words(parts[1]);
            if (!a0.empty() && !a1.empty()) {
                char e1 = a1.back();
                a1 = e1 == 'x' ? a1 + "es" : e1 == 'y' ? a1.substr(0, a1.size() - 1) + "ies" : a1 + "s";
                out.tokens(a.first, nx, a0 + " by " + a1, 66);
            }
            i = nx;
            continue;
        }
        // "a 9x13-inch pan": the hyphenated measure word stays as written
        bool hyphen_word = doc.punct(nx, '-') && doc.joined(nx) && doc.word(nx + 1) && doc.joined(nx + 1);
        if (!free_right(doc, last) && !unit_after && !hyphen_word) {
            i = last;
            continue;
        }
        if (hyphen_word) unit_after = false;
        bool four = false;
        for (size_t k = 0; k < nparts; ++k) four |= parts[k].ndigits() == 4 && parts[k].is_int();
        // multiplication: "76 x 4 = 304", "the formula 8 x 15"
        bool what_is = doc.is(prev_w(doc, a.first) == Doc::npos ? doc.size() : prev_w(doc, a.first), "is") &&
                       doc.is(prev_w(doc, a.first, 2) == Doc::npos ? doc.size() : prev_w(doc, a.first, 2), "what");
        bool times = doc.punct(nx, '=') || any_prev(doc, a.first, 4, TIMES_CUES) || what_is ||
                     (a.first >= 1 && doc.punct(a.first - 1, '='));
        const char* how = times ? " times " : " by ";
        std::string words;
        bool bad = false;
        for (size_t k = 0; k < nparts; ++k) {
            std::string w = times ? num_words(parts[k]) : dim_words(parts[k], four && parts[k].ndigits() != 4);
            if (w.empty()) bad = true;
            if (k) words += how;
            words += w;
        }
        if (bad) {
            i = last;
            continue;
        }
        bool cue = sentence_has(doc, a.first, last, DIM_CUES, 6) || unit_after;
        bool letter_spaced = doc.word(op) && doc[op].space_before;
        int score = times ? 64 : cue ? 66 : letter_spaced ? 56 : 60;
        // "4x400m": the unit is stuck to a number that is stuck to the sign; no unit reading sees it, so read it here
        const std::string_view* stuck = unit_after && !times && doc.joined(nx) && doc.joined(parts[nparts - 1].first)
                                            ? UNIT_WORDS.find(doc.low(nx)) : nullptr;
        if (stuck && free_right(doc, nx)) {
            std::string u(*stuck);
            const Num& lp = parts[nparts - 1];
            bool one = lp.whole == "1" && lp.frac.empty();
            if (!one && !adjectival_after(doc, nx + 1)) u = u == "foot" ? "feet" : u == "inch" ? "inches" : u + "s";
            out.tokens(a.first, nx, words + " " + u, score);
        } else if (unit_after && !times) {
            // the unit's reading takes the last number: propose "2 x" -> "two by" and leave "3 m"
            std::string head;
            for (size_t k = 0; k + 1 < nparts; ++k) {
                std::string w = dim_words(parts[k], four && parts[k].ndigits() != 4);
                head += (k ? " by " : "") + w;
            }
            head += " by";
            size_t lastop = parts[nparts - 1].first - 1;
            out.tokens(a.first, lastop, head, score);
            // a joined unit ("12in") is stuck for the fallback; read the number when nothing else does. A
            // four-digit part keeps its pairs against the unit's cardinal ("1600×1200 px" twelve hundred pixels).
            const Num& lp = parts[nparts - 1];
            bool pairs = lp.is_int() && lp.ndigits() == 4 && !lp.grouped;
            out.tokens(lp.first, lp.last, dim_words(lp, four && lp.ndigits() != 4), pairs ? 40 : 12);
        } else {
            out.tokens(a.first, last, words, score);
        }
        i = last;
    }
}

// "2 m × 3 m": a lone sign between two measured quantities.
RV_RULE("numbers_context", dimension_sign) {
    for (size_t i = 2; i + 1 < doc.size(); ++i) {
        if (!times_op(doc, i) || !doc[i].space_before || !doc[i + 1].space_before) continue;
        // "length x width" in a formula
        if (doc.word(i - 1) && doc.word(i + 1) && doc.low(i) == "x" && FORMULA_WORDS.contains(doc.low(i - 1)) &&
            FORMULA_WORDS.contains(doc.low(i + 1))) {
            out.tokens(i, i, "times", 60);
            continue;
        }
        if (!doc.number(i + 1)) continue;
        // "$10 x 3": a price times a count
        if (doc.number(i - 1)) {
            size_t f = i - 1;
            while (f >= 2 && doc.joined(f) && (doc.punct(f - 1, ',') || doc.punct(f - 1, '.')) && doc.number(f - 2)) f -= 2;
            if (f >= 1 && doc.joined(f) && doc[f - 1].kind == Kind::Punct &&
                (doc[f - 1].cp == '$' || doc[f - 1].cp == 0x20AC || doc[f - 1].cp == 0xA3) && free_right(doc, i + 1))
                out.tokens(i, i, "times", 60);
            continue;
        }
        if (!doc.word(i - 1) || !doc.number(i - 2)) continue;
        std::string_view u = doc.low(i - 1);
        if (!UNIT_AFTER.contains(u) && u != "inches" && u != "feet" && u != "meters" && u != "cm") continue;
        out.tokens(i, i, "by", 55);
    }
}

RV_RULE("numbers_context", multipliers) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        // "zoom ×10": the sign before the number
        if (i >= 1 && doc.joined(i) && doc.punct(i - 1, 0xD7) && !doc.joined(i - 1) && !doc.number(i - 2)) {
            Num m = num_at(doc, i);
            std::string w = num_words(m);
            if (m.ok && !w.empty() && free_right(doc, m.last)) out.tokens(i - 1, m.last, "times " + w, 66);
            i = m.ok ? m.last : i;
            continue;
        }
        size_t e = num_end(doc, i);
        if (!times_op(doc, e + 1)) {
            i = e;
            continue;
        }
        Num a = num_at(doc, i);
        size_t op = a.last + 1;
        if (op >= doc.size() || !times_op(doc, op) || a.whole.empty()) {
            i = a.last;
            continue;
        }
        bool letter = doc.word(op);
        // "6x", "2.5×", "50 ×" (a spaced sign only for ×); a spaced letter only before a comparison: "3 x faster"
        bool spaced_cmp = letter && !doc.joined(op) && doc.word(op + 1) && doc[op + 1].space_before &&
                          COMPARATIVE.contains(doc.low(op + 1)) && doc.low(op + 1) != "that" &&
                          doc.low(op + 1) != "what" && doc.low(op + 1) != "over";
        if ((letter && !doc.joined(op) && !spaced_cmp) || (doc.number(op + 1) && op_between(doc, op))) {
            i = a.last;
            continue;
        }
        // range: "2-3x", "2–3×"
        size_t start = a.first;
        std::string words = num_words(a);
        if (a.first >= 2 && doc.joined(a.first) && (doc.punct(a.first - 1, '-') || doc.punct(a.first - 1, 0x2013)) &&
            doc.number(a.first - 2) && doc.joined(a.first - 1)) {
            size_t f = a.first - 2;
            while (f >= 2 && doc.joined(f) && (doc.punct(f - 1, '.') || doc.punct(f - 1, ',')) && doc.number(f - 2)) f -= 2;
            Num r = num_at(doc, f);
            if (r.ok && r.last == a.first - 2 && free_left(doc, r.first) && !num_words(r).empty()) {
                start = r.first;
                words = num_words(r) + " to " + words;
            }
        }
        if (start == a.first && !free_left(doc, a.first)) {
            i = op;
            continue;
        }
        // model numbers: Ryzen 7 5800X, the 9950X
        if (letter && doc.str(op) == "X" && a.ndigits() >= 3) {
            i = op;
            continue;
        }
        // "2x/day", "3x/week": times per day
        if (doc.punct(op + 1, '/') && doc.joined(op + 1) && doc.word(op + 2) && doc.joined(op + 2) &&
            PER_TIME.contains(doc.low(op + 2)) && free_right(doc, op + 2) && !words.empty()) {
            const std::string_view* full = PER_TIME_WORD.find(doc.low(op + 2));
            out.tokens(start, op + 2, words + " times per " + std::string(full ? *full : doc.low(op + 2)), 70);
            i = op + 2;
            continue;
        }
        // what follows the sign: nothing joined; not algebra (2x = 20, 3x + 1) or a power
        size_t nx = op + 1;
        if (nx < doc.size()) {
            if (doc.joined(nx) && !free_right(doc, op) && !(doc.punct(nx, '-') && doc.word(nx + 1))) {
                i = op;
                continue;
            }
            const rv::Token& t = doc[nx];
            bool algebra = t.cp == '=' || t.cp == '+' || t.cp == '^' || t.cp == '*' || t.cp == 0xB2 || t.cp == 0xB3 ||
                           ((t.cp == '(' || t.cp == '/') && doc.joined(nx));
            if (t.kind == Kind::Punct && algebra) {
                i = op;
                continue;
            }
            if (doc.number(nx)) {
                i = op;
                continue;
            }
        }
        if (words.empty()) {
            i = op;
            continue;
        }
        size_t w = next_w(doc, op);
        bool cue = w == op + 1 && COMPARATIVE.contains(doc.low(w));
        // "11.1×": the sign itself is unambiguous
        bool sign = !letter && doc.joined(op);
        out.tokens(start, op, words + " times", sign ? 74 : cue ? 70 : 62);
        i = op;
    }
}

}  // namespace
