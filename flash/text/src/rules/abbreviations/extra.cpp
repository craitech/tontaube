// Rules of the module "abbreviations" around numbers: ordinal street names ("5th Ave." Fifth Avenue), numbered
// freeways ("the 101 Fwy." one oh one), circa before any number ("c. 80 BC"), cups in recipes ("2 c. flour"), and
// life dates ("b. 1769", "d. 1821", "fl. 1630s", "r. 1509-1547").
#include "core/rule.h"
#include "rules/abbreviations/common.h"

namespace {

using abbr::clean_end;
using abbr::clean_start;
using abbr::lower_first;
using abbr::upper_first;
using rv::Doc;
using rv::Out;

uint32_t value_of(std::string_view s) {
    uint32_t n = 0;
    for (char c : s) n = n * 10 + static_cast<uint32_t>(c - '0');
    return n;
}

// Token i is a plain integer: not part of a grouped number, a decimal, a time or a code.
bool plain_number(const Doc& d, size_t i) {
    if (!d.number(i)) return false;
    if (i > 0 && d.joined(i) && d[i - 1].kind == rv::Kind::Punct &&
        (d[i - 1].cp == '.' || d[i - 1].cp == ',' || d[i - 1].cp == ':' || d[i - 1].cp == '-' ||
         d[i - 1].cp == '/' || d[i - 1].cp == '#'))
        return false;
    if (i > 0 && d.joined(i) && !(d[i - 1].kind == rv::Kind::Punct)) return false;
    size_t n = i + 1;
    if (n < d.size() && d.joined(n) && d[n].kind == rv::Kind::Punct &&
        (d[n].cp == '.' || d[n].cp == ',' || d[n].cp == ':' || d[n].cp == '/') && d.number(n + 1) && d.joined(n + 1))
        return false;
    return true;
}

// Abbreviated street words after an ordinal: "5th Ave.", "W. 14th St." (labels keep "42nd Street" lower case).
// "9th Cir." is a court, not a street.
const rv::WordSet ORDINAL_STREETS{"ave", "blvd", "ct", "dr", "ln", "pkwy", "pl", "rd", "st", "ter"};
// With their period only: "2nd Pl" and "the 9th Ct" are unclear.
const rv::WordSet ABBR_ONLY{"ct", "dr", "pl"};

void ordinal_street(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i + 1) || !doc.joined(i + 1)) return;
    std::string_view suf = doc.low(i + 1);
    if (!(suf == "st" || suf == "nd" || suf == "rd" || suf == "th")) return;
    if (!plain_number(doc, i) || doc.str(i).size() > 3 || doc.str(i)[0] == '0') return;
    size_t s = i + 2;
    if (!doc.word(s) || !doc[s].space_before || doc[s].newline_before || !upper_first(doc, s) ||
        doc[s].casing == rv::UPPER || !ORDINAL_STREETS.contains(doc.low(s)))
        return;
    bool dotted = doc.punct(s + 1, '.') && doc.joined(s + 1);
    if (ABBR_ONLY.contains(doc.low(s)) && !dotted) return;
    if (!clean_end(doc, i + 1)) return;
    out.tokens(i, i + 1, rv::capital(rv::ordinal(value_of(doc.str(i)))), 66);
}

// Freeway words after a route number: "the 101 Fwy.", "the 405 freeway", "the 101 N. Fwy."
const rv::WordSet FREEWAYS{"expressway", "expy", "freeway", "fwy", "tollway"};

void numbered_freeway(const Doc& doc, Out& out, size_t i) {
    size_t k = i + 1;
    if (!doc.word(k) || !doc[k].space_before || doc[k].newline_before) return;
    if (!plain_number(doc, i) || !clean_start(doc, i)) return;
    std::string_view s = doc.str(i);
    if (s.size() > 4 || s[0] == '0') return;
    // a compass letter between: "101 N Fwy.", "101 N. Fwy."
    if (doc[k].casing == rv::UPPER && doc.str(k).size() == 1) {
        char c = doc.low(k)[0];
        if (c != 'n' && c != 's' && c != 'e' && c != 'w') return;
        ++k;
        if (doc.punct(k, '.') && doc.joined(k)) ++k;
        if (!doc.word(k) || !doc[k].space_before) return;
    }
    if (!FREEWAYS.contains(doc.low(k)) || doc[k].casing == rv::UPPER) return;
    uint32_t n = value_of(s);
    out.tokens(i, i, s.size() >= 3 ? rv::pairs(n) : rv::cardinal(n), 66);
}

RV_RULE("abbreviations", numbered_roads) {
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        ordinal_street(doc, out, i);
        numbered_freeway(doc, out, i);
    }
}

// "c. 80 BC", "ca. 470-399 BC", "c.1450": circa before a number. A year after it is also read by the date rules,
// which take the whole span when they know the construct.
void circa(const Doc& doc, Out& out, size_t i) {
    std::string_view w = doc.str(i);
    if (!(w == "c" || w == "ca" || w == "circ") || !clean_start(doc, i)) return;
    if (i > 0 && doc.number(i - 1)) return;  // "2 c. flour" is a cup
    size_t k = i + 2;
    if (!doc.number(k) || doc[k].newline_before) return;
    out.tokens(i, i + 1, "circa", 60);
}

bool fraction_glyph(uint32_t cp) { return (cp >= 0xBC && cp <= 0xBE) || (cp >= 0x2150 && cp <= 0x215E); }

// "2 c. flour", "1/2 c. milk", "1 c. sugar": cups before an ingredient.
void cups(const Doc& doc, Out& out, size_t i) {
    if (i == 0 || !doc.is(i, "c") || !doc[i].space_before) return;
    size_t k = i + 2;
    if (!lower_first(doc, k) || !doc[k].space_before || doc[k].newline_before) return;
    size_t p = i - 1;
    bool glyph = doc[p].kind == rv::Kind::Punct && fraction_glyph(doc[p].cp);
    if (!doc.number(p) && !glyph) return;
    // the quantity: "1" and a bare fraction take the singular, everything else the plural
    size_t a = p;
    bool slash = glyph;
    while (a > 0 && doc.joined(a) &&
           (doc.number(a - 1) || doc.punct(a - 1, '/') || doc.punct(a - 1, '.') || doc.punct(a - 1, ','))) {
        slash |= doc.punct(a - 1, '/');
        --a;
    }
    if (glyph && a > 0 && doc.number(a - 1) && doc.joined(a)) {  // "1½"
        slash = false;
        --a;
    }
    bool whole_before = a > 0 && doc.number(a - 1) && doc[a].space_before;  // "1 1/2"
    bool one = (a == p && doc.str(p) == "1") || (slash && !whole_before);
    out.tokens(i, i + 1, one ? "cup" : "cups", 70);
    // the quantity of a list item is no date: "2 c. flour, 1/2 c. butter" one half
    if (!glyph && a + 2 == p && doc.number(a) && doc.punct(a + 1, '/') && doc.joined(a + 1) && doc.joined(p) &&
        doc.str(a).size() <= 2 && doc.str(p).size() <= 2 && (a == 0 || !doc.joined(a))) {
        uint32_t num = value_of(doc.str(a)), den = value_of(doc.str(p));
        if (num == 0 || den < 2 || den > 16 || num >= den) return;
        std::string unit = den == 2 ? "half" : den == 4 ? "quarter" : rv::ordinal(den);
        if (whole_before && !(a >= 2 && doc.joined(a - 1) && !doc.number(a - 2) && doc[a - 2].kind != rv::Kind::Punct)) {
            std::string part = num == 1 ? std::string(den == 8 ? "an " : "a ") + unit
                                        : rv::cardinal(num) + " " + (den == 2 ? "halves" : unit + "s");
            out.tokens(a - 1, p, rv::cardinal(value_of(doc.str(a - 1))) + " and " + part, 66);
        } else {
            std::string words = rv::cardinal(num) + " " + (num == 1 ? unit : den == 2 ? "halves" : unit + "s");
            out.tokens(a, p, words, 66);
        }
    }
}

// Life and reign dates: "(b. 1769, d. 1821)", "Rembrandt (fl. 1630s)", "Henry VIII (r. 1509-1547)".
void life_date(const Doc& doc, Out& out, size_t i) {
    std::string_view w = doc.str(i);
    const char* spoken = w == "b" ? "born" : w == "d" ? "died" : w == "fl" ? "flourished" :
                         w == "r" ? "reigned" : w == "m" ? "married" : nullptr;
    // after an opening bracket, a comma or a semicolon ("b. 2000" at a line start is a list item)
    if (!spoken || i == 0) return;
    bool opens = ((doc.punct(i - 1, '(') || doc.punct(i - 1, '[')) && doc.joined(i)) ||
                 ((doc.punct(i - 1, ',') || doc.punct(i - 1, ';')) && doc[i].space_before);
    if (!opens) return;
    size_t k = i + 2;
    if (!doc[k].space_before || doc[k].newline_before) return;
    bool year = doc.number(k) && doc.str(k).size() >= 3 && doc.str(k).size() <= 4;
    bool circa = (doc.is(k, "c") || doc.is(k, "ca")) && doc.punct(k + 1, '.') && doc.number(k + 2);
    if (!year && !circa) return;
    out.tokens(i, i + 1, spoken, 70);
}

// Lower-case letters with a period before a number or an ingredient: circa, cups, life dates.
RV_RULE("abbreviations", letter_dots) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.word(i) || doc[i].casing != rv::LOWER || doc[i].end - doc[i].start > 4) continue;
        if (!doc.punct(i + 1, '.') || !doc.joined(i + 1)) continue;
        circa(doc, out, i);
        cups(doc, out, i);
        life_date(doc, out, i);
    }
}

}  // namespace
