#pragma once
// Shared helpers of the module "roman_citations".
#include <string>
#include <string_view>

#include "core/rule.h"

namespace rc {

using rv::Doc;
using rv::Out;
constexpr size_t npos = Doc::npos;

inline bool title(const Doc& d, size_t i) { return d.word(i) && d[i].casing == rv::TITLE; }
inline bool upper(const Doc& d, size_t i) { return d.word(i) && d[i].casing == rv::UPPER; }
inline bool lower(const Doc& d, size_t i) { return d.word(i) && d[i].casing == rv::LOWER; }
inline bool line_start(const Doc& d, size_t i) { return i == 0 || d[i].newline_before; }
inline char first_lower(const Doc& d, size_t i) { return d.lower[d[i].start]; }
inline bool punct_tok(const Doc& d, size_t i) { return i < d.size() && d[i].kind == rv::Kind::Punct; }

// Where a removal that starts at token i begins: the end of the previous token when only spaces (no line break)
// lie between, so that "the base [3]." loses " [3]" and keeps its period.
inline uint32_t lead(const Doc& d, size_t i) {
    if (i > 0 && i < d.size() && d[i].space_before && !d[i].newline_before) return d[i - 1].end;
    return d[i].start;
}

// "One hundred twelve", "the Sixteenth".
inline std::string card_cap(int v) { return rv::capital(rv::cardinal(static_cast<uint64_t>(v))); }
inline std::string the_ord(int v) { return "the " + rv::capital(rv::ordinal(static_cast<uint64_t>(v))); }

// A Roman numeral word, possibly with a possessive "'s" / "’s" ("VI's").
struct Roman {
    int value = 0;
    bool upper = false;
    std::string_view base, suffix;
};
inline Roman roman(const Doc& d, size_t i) {
    Roman r;
    if (!d.word(i)) return r;
    switch (d.lower[d[i].start]) {  // most words stop at the first letter
        case 'i': case 'v': case 'x': case 'l': case 'c': case 'd': case 'm': break;
        default: return r;
    }
    std::string_view s = d.str(i);
    if (s.size() > 2 && (s.substr(s.size() - 2) == "'s")) r.suffix = s.substr(s.size() - 2);
    else if (s.size() > 4 && s.substr(s.size() - 4) == "\xE2\x80\x99s") r.suffix = s.substr(s.size() - 4);
    r.base = s.substr(0, s.size() - r.suffix.size());
    if (r.base.empty() || r.base.size() > 15) return r;
    for (char c : r.base) {  // only numeral letters: most words stop here, before the full check
        switch (c | 0x20) {
            case 'i': case 'v': case 'x': case 'l': case 'c': case 'd': case 'm': break;
            default: return r;
        }
    }
    r.value = rv::roman_value(r.base);
    r.upper = !r.base.empty() && r.base[0] >= 'A' && r.base[0] <= 'Z';
    return r;
}

// Digit value of a superscript digit code point, -1 otherwise.
inline int superscript_digit(uint32_t cp) {
    switch (cp) {
        case 0x2070: return 0;
        case 0xB9: return 1;
        case 0xB2: return 2;
        case 0xB3: return 3;
        default: return cp >= 0x2074 && cp <= 0x2079 ? static_cast<int>(cp - 0x2070) : -1;
    }
}
inline bool superscript(const Doc& d, size_t i) { return punct_tok(d, i) && superscript_digit(d[i].cp) >= 0; }

inline bool dash(uint32_t cp) { return cp == '-' || cp == 0x2013 || cp == 0x2014; }
inline bool dash_tok(const Doc& d, size_t i) { return punct_tok(d, i) && dash(d[i].cp); }

// Integer value of a Number token (0 if it has more than 9 digits).
inline uint32_t num(const Doc& d, size_t i) {
    std::string_view s = d.str(i);
    if (s.size() > 9) return 0;
    uint32_t v = 0;
    for (char c : s) v = v * 10 + static_cast<uint32_t>(c - '0');
    return v;
}

// A whole number written plainly (no leading zero), read in full: "1100" -> one thousand one hundred.
inline std::string plain(const Doc& d, size_t i) { return rv::read_number(d.str(i)); }

// A year in pairs, the way citations read it: 2021 -> twenty twenty-one, 2005 -> two thousand five.
inline std::string year_words(const Doc& d, size_t i) { return rv::pairs(num(d, i)); }
inline bool year_like(const Doc& d, size_t i) {
    if (!d.number(i) || d.str(i).size() != 4) return false;
    uint32_t y = num(d, i);
    return y >= 1500 && y <= 2099;
}

// Token p is a period that also ends the sentence: end of text, a line break, or a capitalized word, a quote or a
// digit after a space.
inline bool period_ends_sentence(const Doc& d, size_t p) {
    size_t n = p + 1;
    if (n >= d.size() || d[n].newline_before) return true;
    if (!d[n].space_before) return false;
    if (d.word(n)) return d[n].casing != rv::LOWER;
    uint32_t c = d[n].cp;
    return c == '"' || c == 0x201C || c == '\'' || c == 0x2018;
}

// The lower-case letter of a single-letter word ("b" in "(b)"), 0 otherwise.
inline char single_letter(const Doc& d, size_t i) {
    if (!d.word(i) || d[i].end - d[i].start != 1) return 0;
    char c = d.text[d[i].start];
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c;
}
inline std::string letter_cap(char c) { return std::string(1, static_cast<char>(c >= 'a' && c <= 'z' ? c - 32 : c)); }

// Numbers with "oh" for a zero: 107 -> one oh seven.
inline std::string digits_oh(std::string_view s, bool final_zero_oh) {
    static const char* const W[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    std::string out;
    for (size_t k = 0; k < s.size(); ++k) {
        char c = s[k];
        if (c < '0' || c > '9') continue;
        if (!out.empty()) out += ' ';
        bool oh = c == '0' && (k + 1 < s.size() || final_zero_oh);
        out += oh ? "oh" : W[c - '0'];
    }
    return out;
}

// Sections of laws and codes: a three- or four-digit whole number in pairs (1001 -> ten oh one), shorter ones in
// full, further dotted parts as numbers. Before a dotted part the whole number is in pairs only where pairs are
// natural: 226.1 -> two twenty-six point one, 1508.1 -> fifteen oh eight point one, but 201.3 -> two hundred one
// point three, 430.10 -> four hundred thirty point ten, 1026.34 -> one thousand twenty-six point thirty-four.
inline std::string section_part(std::string_view s, bool first, bool before_dot = false) {
    if (s.empty()) return "";
    if (s.size() > 1 && s[0] == '0') return rv::digits(s);
    bool natural = (s.size() == 3 && s[1] != '0' && s[2] != '0') || (s.size() == 4 && s[1] != '0');
    if ((first || (before_dot && natural)) && (s.size() == 3 || s.size() == 4)) {
        uint32_t v = 0;
        for (char c : s) v = v * 10 + static_cast<uint32_t>(c - '0');
        return rv::pairs(v);
    }
    return rv::cardinal_digits(s);
}

// A section number starting at Number token i: "1001" (pairs), "430.2" (four hundred thirty point two), "226.1"
// (two twenty-six point one), "3.14" (three point one four), "4.2.1", "3B", "78a". Sets last to its last token.
inline std::string section_number(const Doc& d, size_t i, size_t& last) {
    last = i;
    bool dotted = d.punct(i + 1, '.') && d.joined(i + 1) && d.number(i + 2) && d.joined(i + 2);
    std::string out = section_part(d.str(i), !dotted, dotted);
    if (out.empty()) return "";
    bool short_whole = d.str(i).size() == 1;
    int parts = 0;
    while (d.punct(last + 1, '.') && d.joined(last + 1) && d.number(last + 2) && d.joined(last + 2)) {
        std::string_view f = d.str(last + 2);
        out += " point " + (short_whole && parts == 0 && f.size() > 1 && !(d.punct(last + 3, '.') && d.number(last + 4))
                                ? rv::digits(f)
                                : section_part(f, false));
        last += 2;
        ++parts;
    }
    if (char c = single_letter(d, last + 1); c && d.joined(last + 1)) {
        out += " " + letter_cap(c);
        last += 1;
    }
    return out;
}

// Parenthesized subdivisions after a reference: "(2)(b)" -> ", paragraph two, letter B". Sets last; "" if none.
inline std::string subdivisions(const Doc& d, size_t after, size_t& last, bool letters_plain = false) {
    std::string out;
    last = after;
    size_t k = after + 1;
    while (d.punct(k, '(') && d.joined(k) && k + 2 < d.size() && d.punct(k + 2, ')') && d.joined(k + 1) &&
           d.joined(k + 2)) {
        if (d.number(k + 1) && d.str(k + 1).size() <= 3) {
            out += ", paragraph " + plain(d, k + 1);
        } else if (char c = single_letter(d, k + 1)) {
            out += letters_plain ? " " + letter_cap(c) : ", letter " + letter_cap(c);
        } else {
            break;
        }
        last = k + 2;
        k += 3;
    }
    return out;
}

}  // namespace rc
