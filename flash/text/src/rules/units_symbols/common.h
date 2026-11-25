#pragma once
// Helpers shared by the files of the module "units_symbols".
#include "core/rule.h"

namespace us {

using rv::Doc;
using rv::Kind;
using rv::Out;
using rv::Token;

// Value of a superscript / subscript digit, -1 for anything else.
inline int super_digit(uint32_t cp) {
    switch (cp) {
        case 0x2070: return 0;
        case 0xB9: return 1;
        case 0xB2: return 2;
        case 0xB3: return 3;
        default: return cp >= 0x2074 && cp <= 0x2079 ? static_cast<int>(cp - 0x2070) : -1;
    }
}
inline int sub_digit(uint32_t cp) { return cp >= 0x2080 && cp <= 0x2089 ? static_cast<int>(cp - 0x2080) : -1; }

inline bool punct(const Doc& d, size_t i) { return i < d.size() && d[i].kind == Kind::Punct; }
inline bool super_tok(const Doc& d, size_t i) { return punct(d, i) && super_digit(d[i].cp) >= 0; }
inline bool sub_tok(const Doc& d, size_t i) { return punct(d, i) && sub_digit(d[i].cp) >= 0; }
// Token i follows token i-1 on the same line, with a space or joined.
inline bool same_line(const Doc& d, size_t i) { return i < d.size() && !d[i].newline_before; }
// Token i is stuck to a letter or digit before it.
inline bool stuck_before(const Doc& d, size_t i) {
    return d.joined(i) && d[i - 1].kind != Kind::Punct;
}
// Token i is followed directly by a letter or digit.
inline bool stuck_after(const Doc& d, size_t i) {
    return i + 1 < d.size() && d.joined(i + 1) && d[i + 1].kind != Kind::Punct;
}
// Run of superscript (or subscript) digit tokens starting at i, joined; returns the last index, or npos.
inline size_t digit_run(const Doc& d, size_t i, bool super, std::string& digits) {
    size_t last = Doc::npos;
    for (size_t k = i; k < d.size() && (k == i || d.joined(k)); ++k) {
        int v = super ? (punct(d, k) ? super_digit(d[k].cp) : -1) : (punct(d, k) ? sub_digit(d[k].cp) : -1);
        if (v < 0) break;
        digits += static_cast<char>('0' + v);
        last = k;
    }
    return last;
}
// "squared", "cubed" or "to the power of N" for an exponent given as digits ('10³' is said with "to the power of").
inline std::string power_words(const std::string& digits, bool minus, bool allow_cubed) {
    if (!minus && digits == "2") return "squared";
    if (!minus && allow_cubed && digits == "3") return "cubed";
    std::string n = rv::cardinal_digits(digits);
    if (n.empty()) return "";
    return std::string("to the power of ") + (minus ? "minus " : "") + n;
}
// A currency sign: $ € £ ¥ ₹ ₩ ₽ ₺ ₪ ₿.
inline bool currency_sign(uint32_t cp) {
    return cp == '$' || cp == 0x20AC || cp == 0xA3 || cp == 0xA5 || cp == 0x20B9 || cp == 0x20A9 || cp == 0x20BD ||
           cp == 0x20BA || cp == 0x20AA || cp == 0x20BF;
}
// Text of token i, or "" past the end.
inline std::string_view tok(const Doc& d, size_t i) { return i < d.size() ? d.str(i) : std::string_view(); }

}  // namespace us
