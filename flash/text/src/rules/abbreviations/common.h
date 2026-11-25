#pragma once
// Helpers shared by the files of the module "abbreviations".
#include "core/rule.h"

namespace abbr {

using rv::Doc;
using rv::Out;

inline bool upper_first(const Doc& d, size_t i) { return d.word(i) && rv::is_upper(d[i].cp); }
inline bool lower_first(const Doc& d, size_t i) { return d.word(i) && rv::is_lower(d[i].cp); }

inline bool closing_mark(uint32_t cp) {
    switch (cp) {
        case '"': case '\'': case ')': case ']': case 0x201D: case 0x2019: case 0xBB:
            return true;
        default:
            return false;
    }
}

// Token i may start an abbreviation: whitespace or an opening mark before it, not the middle of a domain, a path or
// an identifier ("example.co", "/St", "x@Co").
inline bool clean_start(const Doc& d, size_t i) {
    if (!d.joined(i)) return true;
    const rv::Token& p = d[i - 1];
    if (p.kind != rv::Kind::Punct) return false;
    switch (p.cp) {
        case '(': case '[': case '"': case '\'': case 0x201C: case 0x2018: case 0xAB: case '*': case '_':
        case 0x2014: case 0x2013: case '-': case '{': case '<': case '>': case '`':
            return true;
        default:
            return false;
    }
}

// Nothing letter- or digit-like is glued to the end of the span that ends with token last.
inline bool clean_end(const Doc& d, size_t last) {
    size_t n = last + 1;
    if (n >= d.size() || !d.joined(n)) return true;
    if (d.word(n) || d.number(n)) return false;
    uint32_t cp = d[n].cp;
    if (cp == '/' || cp == '_' || cp == '@' || cp == '\\') return false;
    bool alnum_after = n + 1 < d.size() && d.joined(n + 1) && (d.word(n + 1) || d.number(n + 1));
    return !((cp == '.' || cp == ':') && alnum_after);  // "k8s.yaml" is a file; "k8s-native" keeps its hyphen
}

// The first letter of spoken follows the first letter of the written form.
inline std::string match_case(std::string spoken, bool upper) {
    if (upper) return rv::capital(std::move(spoken));
    return spoken;
}

}  // namespace abbr
