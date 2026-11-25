// Rules of the module "chars": characters outside the speech model's alphabet that no other rule read. Every
// reading here scores 5, below every other reading, so it applies only where no rule took the character
// (README: "Characters the model does not know"). Texts of ASCII and known characters pay one byte scan.
#include <cstring>

#include "core/rule.h"

namespace {

using rv::Doc;
using rv::Kind;
using rv::Out;

constexpr int SCORE = 5;

// Characters the speech model knows beyond ASCII: £ § ° » ¼ ½ ¾, the accented letters below, Œ œ, – — ‘ ’ “ ” „ … €.
bool known(uint32_t cp) {
    switch (cp) {
        case 0xA3: case 0xA7: case 0xB0: case 0xBB: case 0xBC: case 0xBD: case 0xBE:
        case 0xC0: case 0xC1: case 0xC2: case 0xC6: case 0xC7: case 0xC9: case 0xCA: case 0xD6: case 0xDC:
        case 0xE0: case 0xE1: case 0xE2: case 0xE3: case 0xE4: case 0xE6: case 0xE7: case 0xE8: case 0xE9:
        case 0xEA: case 0xEB: case 0xEC: case 0xED: case 0xEE: case 0xEF: case 0xF1: case 0xF2: case 0xF3:
        case 0xF4: case 0xF5: case 0xF6: case 0xF9: case 0xFA: case 0xFB: case 0xFC: case 0xFF:
        case 0x152: case 0x153: case 0x2013: case 0x2014: case 0x2018: case 0x2019: case 0x201C: case 0x201D:
        case 0x201E: case 0x2026: case 0x20AC:
            return true;
        default:
            return false;
    }
}

enum Act : uint8_t {
    WORDS,  // read as words
    CHAR,   // replaced by a character the model knows
    SEP,    // a separator inside a line: a comma pause
    DROP,   // removed
    SET,    // a number set: the real numbers; R two before a superscript
    TIMES,  // a middle dot: times between digits, a pause elsewhere
};

struct Reading {
    uint32_t cp;
    Act act;
    const char* s;
};

// Sorted by code point. Relations read as in the math module; the plain-text readings of units_symbols
// (≤ at most, ≈ approximately) are kept for the signs it reads.
const Reading READINGS[] = {
    {0xA1, DROP, ""},                        // ¡
    {0xA2, WORDS, "cents"},                  // ¢
    {0xA4, DROP, ""},                        // ¤
    {0xA5, WORDS, "yen"},                    // ¥
    {0xA6, SEP, ","},                        // ¦
    {0xA8, DROP, ""},                        // ¨
    {0xA9, WORDS, "copyright"},              // ©
    {0xAB, CHAR, "\xE2\x80\x9C"},            // « -> “
    {0xAC, WORDS, "not"},                    // ¬
    {0xAE, WORDS, "registered trademark"},   // ®
    {0xAF, DROP, ""},                        // ¯
    {0xB1, WORDS, "plus or minus"},          // ±
    {0xB4, CHAR, "'"},                       // ´ as an apostrophe
    {0xB6, WORDS, "paragraph"},              // ¶
    {0xB7, TIMES, ""},                       // ·
    {0xB8, DROP, ""},                        // ¸
    {0xBF, DROP, ""},                        // ¿
    {0xD7, WORDS, "times"},                  // ×
    {0xF7, WORDS, "divided by"},             // ÷
    {0x2B9, CHAR, "'"},                      // ʹ
    {0x2BB, CHAR, "'"},                      // ʻ (Hawaiʻi)
    {0x2BC, CHAR, "'"},                      // ʼ
    {0x2BE, CHAR, "'"},                      // ʾ
    {0x2BF, CHAR, "'"},                      // ʿ
    {0x2C6, DROP, ""},                       // ˆ
    {0x2C7, DROP, ""},                       // ˇ
    {0x2DA, DROP, ""},                       // ˚
    {0x2DC, DROP, ""},                       // ˜
    {0x2012, CHAR, "\xE2\x80\x93"},          // ‒ -> –
    {0x2015, CHAR, "\xE2\x80\x94"},          // ― -> —
    {0x2016, DROP, ""},                      // ‖
    {0x2017, DROP, ""},                      // ‗
    {0x201A, CHAR, "\xE2\x80\x98"},          // ‚ -> ‘
    {0x201B, CHAR, "\xE2\x80\x98"},          // ‛ -> ‘
    {0x201F, CHAR, "\xE2\x80\x9C"},          // ‟ -> “
    {0x2020, DROP, ""},                      // †
    {0x2021, DROP, ""},                      // ‡
    {0x2022, SEP, ","},                      // •
    {0x2023, SEP, ","},                      // ‣
    {0x2024, CHAR, "."},                     // ․
    {0x2025, CHAR, "\xE2\x80\xA6"},          // ‥ -> …
    {0x2027, DROP, ""},                      // ‧
    {0x2031, WORDS, "per ten thousand"},     // ‱
    {0x2035, CHAR, "\xE2\x80\x98"},          // ‵
    {0x2036, CHAR, "\xE2\x80\x9C"},          // ‶
    {0x2039, CHAR, "\xE2\x80\x98"},          // ‹ -> ‘
    {0x203A, CHAR, "\xE2\x80\x99"},          // › -> ’
    {0x203B, DROP, ""},                      // ※
    {0x203C, CHAR, "!"},                     // ‼
    {0x203E, DROP, ""},                      // ‾
    {0x2043, SEP, ","},                      // ⁃
    {0x2044, CHAR, "/"},                     // ⁄
    {0x2047, CHAR, "?"},                     // ⁇
    {0x2048, CHAR, "?"},                     // ⁈
    {0x2049, CHAR, "!"},                     // ⁉
    {0x204E, DROP, ""},                      // ⁎
    {0x20A1, WORDS, "colones"},              // ₡
    {0x20A6, WORDS, "naira"},                // ₦
    {0x20A9, WORDS, "won"},                  // ₩
    {0x20AA, WORDS, "shekels"},              // ₪
    {0x20AB, WORDS, "dong"},                 // ₫
    {0x20B1, WORDS, "pesos"},                // ₱
    {0x20B4, WORDS, "hryvnias"},             // ₴
    {0x20B9, WORDS, "rupees"},               // ₹
    {0x20BA, WORDS, "lira"},                 // ₺
    {0x20BD, WORDS, "rubles"},               // ₽
    {0x20BF, WORDS, "bitcoin"},              // ₿
    {0x2102, SET, "the complex numbers"},    // ℂ
    {0x2103, WORDS, "degrees Celsius"},      // ℃
    {0x2109, WORDS, "degrees Fahrenheit"},   // ℉
    {0x210D, SET, "the quaternions"},        // ℍ
    {0x210F, WORDS, "h bar"},                // ℏ
    {0x2111, WORDS, "imaginary part"},       // ℑ
    {0x2113, WORDS, "L"},                    // ℓ
    {0x2115, SET, "the natural numbers"},    // ℕ
    {0x2116, WORDS, "number"},               // №
    {0x2117, WORDS, "sound recording copyright"},  // ℗
    {0x2118, WORDS, "P"},                    // ℘
    {0x2119, SET, "P"},                      // ℙ
    {0x211A, SET, "the rational numbers"},   // ℚ
    {0x211C, WORDS, "real part"},            // ℜ
    {0x211D, SET, "the real numbers"},       // ℝ
    {0x211E, WORDS, "prescription"},         // ℞
    {0x2120, WORDS, "service mark"},         // ℠
    {0x2122, WORDS, "trademark"},            // ™
    {0x2124, SET, "the integers"},           // ℤ
    {0x2126, WORDS, "Omega"},                // Ω (ohm sign)
    {0x212A, CHAR, "K"},                     // K (kelvin sign)
    {0x212B, CHAR, "A"},                     // Å (angstrom sign)
    {0x212E, DROP, ""},                      // ℮
    {0x2135, WORDS, "aleph"},                // ℵ
    {0x2150, WORDS, "one seventh"},          // ⅐
    {0x2151, WORDS, "one ninth"},            // ⅑
    {0x2152, WORDS, "one tenth"},            // ⅒
    {0x2153, WORDS, "one third"},            // ⅓
    {0x2154, WORDS, "two thirds"},           // ⅔
    {0x2155, WORDS, "one fifth"},            // ⅕
    {0x2156, WORDS, "two fifths"},           // ⅖
    {0x2157, WORDS, "three fifths"},         // ⅗
    {0x2158, WORDS, "four fifths"},          // ⅘
    {0x2159, WORDS, "one sixth"},            // ⅙
    {0x215A, WORDS, "five sixths"},          // ⅚
    {0x215B, WORDS, "one eighth"},           // ⅛
    {0x215C, WORDS, "three eighths"},        // ⅜
    {0x215D, WORDS, "five eighths"},         // ⅝
    {0x215E, WORDS, "seven eighths"},        // ⅞
    {0x2190, WORDS, "from"},                 // ←
    {0x2191, WORDS, "up"},                   // ↑
    {0x2192, WORDS, "to"},                   // →
    {0x2193, WORDS, "down"},                 // ↓
    {0x2194, WORDS, "to and from"},          // ↔
    {0x2197, WORDS, "up"},                   // ↗
    {0x2198, WORDS, "down"},                 // ↘
    {0x21A6, WORDS, "maps to"},              // ↦
    {0x21A9, DROP, ""},                      // ↩
    {0x21AA, DROP, ""},                      // ↪
    {0x21C4, WORDS, "to and from"},          // ⇄
    {0x21D0, WORDS, "is implied by"},        // ⇐
    {0x21D2, WORDS, "implies"},              // ⇒
    {0x21D4, WORDS, "if and only if"},       // ⇔
    {0x2200, WORDS, "for all"},              // ∀
    {0x2201, WORDS, "the complement of"},    // ∁
    {0x2202, WORDS, "partial"},              // ∂
    {0x2203, WORDS, "there exists"},         // ∃
    {0x2204, WORDS, "there is no"},          // ∄
    {0x2205, WORDS, "the empty set"},        // ∅
    {0x2206, WORDS, "delta"},                // ∆
    {0x2207, WORDS, "del"},                  // ∇
    {0x2208, WORDS, "is in"},                // ∈
    {0x2209, WORDS, "is not in"},            // ∉
    {0x220A, WORDS, "is in"},                // ∊
    {0x220B, WORDS, "contains"},             // ∋
    {0x220C, WORDS, "does not contain"},     // ∌
    {0x220E, DROP, ""},                      // ∎
    {0x220F, WORDS, "the product of"},       // ∏
    {0x2210, WORDS, "the coproduct of"},     // ∐
    {0x2211, WORDS, "the sum of"},           // ∑
    {0x2212, CHAR, "-"},                     // −
    {0x2213, WORDS, "minus or plus"},        // ∓
    {0x2214, WORDS, "plus"},                 // ∔
    {0x2215, CHAR, "/"},                     // ∕
    {0x2216, WORDS, "minus"},                // ∖
    {0x2217, WORDS, "times"},                // ∗
    {0x2218, WORDS, "composed with"},        // ∘
    {0x2219, TIMES, ""},                     // ∙
    {0x221A, WORDS, "the square root of"},   // √
    {0x221B, WORDS, "the cube root of"},     // ∛
    {0x221C, WORDS, "the fourth root of"},   // ∜
    {0x221D, WORDS, "is proportional to"},   // ∝
    {0x221E, WORDS, "infinity"},             // ∞
    {0x221F, WORDS, "right angle"},          // ∟
    {0x2220, WORDS, "angle"},                // ∠
    {0x2221, WORDS, "angle"},                // ∡
    {0x2223, WORDS, "divides"},              // ∣
    {0x2224, WORDS, "does not divide"},      // ∤
    {0x2225, WORDS, "is parallel to"},       // ∥
    {0x2226, WORDS, "is not parallel to"},   // ∦
    {0x2227, WORDS, "and"},                  // ∧
    {0x2228, WORDS, "or"},                   // ∨
    {0x2229, WORDS, "intersection"},         // ∩
    {0x222A, WORDS, "union"},                // ∪
    {0x222B, WORDS, "the integral of"},      // ∫
    {0x222C, WORDS, "the double integral of"},   // ∬
    {0x222D, WORDS, "the triple integral of"},   // ∭
    {0x222E, WORDS, "the contour integral of"},  // ∮
    {0x2234, WORDS, "therefore"},            // ∴
    {0x2235, WORDS, "because"},              // ∵
    {0x2236, CHAR, ":"},                     // ∶
    {0x2237, WORDS, "as"},                   // ∷
    {0x2238, WORDS, "minus"},                // ∸
    {0x223C, WORDS, "approximately"},        // ∼
    {0x223D, WORDS, "is similar to"},        // ∽
    {0x2241, WORDS, "is not similar to"},    // ≁
    {0x2243, WORDS, "is approximately equal to"},  // ≃
    {0x2245, WORDS, "is congruent to"},      // ≅
    {0x2248, WORDS, "approximately"},        // ≈
    {0x2249, WORDS, "is not approximately equal to"},  // ≉
    {0x224D, WORDS, "is equivalent to"},     // ≍
    {0x2250, WORDS, "approaches"},           // ≐
    {0x2252, WORDS, "is approximately equal to"},  // ≒
    {0x2254, WORDS, "is defined as"},        // ≔
    {0x225C, WORDS, "is defined as"},        // ≜
    {0x225D, WORDS, "is defined as"},        // ≝
    {0x2260, WORDS, "not equal to"},         // ≠
    {0x2261, WORDS, "is equivalent to"},     // ≡
    {0x2262, WORDS, "is not equivalent to"},  // ≢
    {0x2264, WORDS, "at most"},              // ≤
    {0x2265, WORDS, "at least"},             // ≥
    {0x2266, WORDS, "at most"},              // ≦
    {0x2267, WORDS, "at least"},             // ≧
    {0x226A, WORDS, "much less than"},       // ≪
    {0x226B, WORDS, "much greater than"},    // ≫
    {0x226E, WORDS, "is not less than"},     // ≮
    {0x226F, WORDS, "is not greater than"},  // ≯
    {0x2270, WORDS, "is not less than or equal to"},  // ≰
    {0x2271, WORDS, "is not greater than or equal to"},  // ≱
    {0x2272, WORDS, "is less than or approximately equal to"},  // ≲
    {0x2273, WORDS, "is greater than or approximately equal to"},  // ≳
    {0x227A, WORDS, "precedes"},             // ≺
    {0x227B, WORDS, "succeeds"},             // ≻
    {0x2282, WORDS, "is a subset of"},       // ⊂
    {0x2283, WORDS, "is a superset of"},     // ⊃
    {0x2284, WORDS, "is not a subset of"},   // ⊄
    {0x2285, WORDS, "is not a superset of"},  // ⊅
    {0x2286, WORDS, "is a subset of or equal to"},  // ⊆
    {0x2287, WORDS, "is a superset of or equal to"},  // ⊇
    {0x2288, WORDS, "is not a subset of"},   // ⊈
    {0x2289, WORDS, "is not a superset of"},  // ⊉
    {0x228A, WORDS, "is a proper subset of"},  // ⊊
    {0x228B, WORDS, "is a proper superset of"},  // ⊋
    {0x228E, WORDS, "union"},                // ⊎
    {0x2293, WORDS, "intersection"},         // ⊓
    {0x2294, WORDS, "union"},                // ⊔
    {0x2295, WORDS, "direct sum"},           // ⊕
    {0x2296, WORDS, "circled minus"},        // ⊖
    {0x2297, WORDS, "tensor"},               // ⊗
    {0x2299, WORDS, "elementwise times"},    // ⊙
    {0x22A2, WORDS, "proves"},               // ⊢
    {0x22A4, WORDS, "top"},                  // ⊤
    {0x22A5, WORDS, "is perpendicular to"},  // ⊥
    {0x22A7, WORDS, "models"},               // ⊧
    {0x22A8, WORDS, "models"},               // ⊨
    {0x22BB, WORDS, "xor"},                  // ⊻
    {0x22C0, WORDS, "and"},                  // ⋀
    {0x22C1, WORDS, "or"},                   // ⋁
    {0x22C2, WORDS, "the intersection of"},  // ⋂
    {0x22C3, WORDS, "the union of"},         // ⋃
    {0x22C4, DROP, ""},                      // ⋄
    {0x22C5, WORDS, "times"},                // ⋅
    {0x22C6, WORDS, "star"},                 // ⋆
    {0x22C8, WORDS, "join"},                 // ⋈
    {0x22EE, CHAR, "\xE2\x80\xA6"},          // ⋮ -> …
    {0x22EF, CHAR, "\xE2\x80\xA6"},          // ⋯ -> …
    {0x22F0, CHAR, "\xE2\x80\xA6"},          // ⋰
    {0x22F1, CHAR, "\xE2\x80\xA6"},          // ⋱
    {0x2300, WORDS, "diameter"},             // ⌀
    {0x2308, DROP, ""},                      // ⌈
    {0x2309, DROP, ""},                      // ⌉
    {0x230A, DROP, ""},                      // ⌊
    {0x230B, DROP, ""},                      // ⌋
    {0x25A0, SEP, ","},                      // ■
    {0x25A1, SEP, ","},                      // □
    {0x25AA, SEP, ","},                      // ▪
    {0x25AB, SEP, ","},                      // ▫
    {0x25B3, WORDS, "triangle"},             // △
    {0x25B6, SEP, ","},                      // ▶
    {0x25B8, SEP, ","},                      // ▸
    {0x25BA, SEP, ","},                      // ►
    {0x25C6, SEP, ","},                      // ◆
    {0x25C7, SEP, ","},                      // ◇
    {0x25CB, SEP, ","},                      // ○
    {0x25CF, SEP, ","},                      // ●
    {0x25E6, SEP, ","},                      // ◦
    {0x27E8, CHAR, "("},                     // ⟨
    {0x27E9, CHAR, ")"},                     // ⟩
    {0x27F5, WORDS, "from"},                 // ⟵
    {0x27F7, WORDS, "to and from"},          // ⟷
    {0x27F8, WORDS, "is implied by"},        // ⟸
    {0x27F9, WORDS, "implies"},              // ⟹
    {0x27FA, WORDS, "if and only if"},       // ⟺
    {0x27FC, WORDS, "maps to"},              // ⟼
    {0x2E3A, CHAR, "\xE2\x80\x94"},          // ⸺ -> —
    {0x2E3B, CHAR, "\xE2\x80\x94"},          // ⸻ -> —
};

const Reading* reading_of(uint32_t cp) {
    const Reading* e = READINGS + sizeof(READINGS) / sizeof(READINGS[0]);
    const Reading* r = std::lower_bound(READINGS, e, cp, [](const Reading& a, uint32_t c) { return a.cp < c; });
    return r != e && r->cp == cp ? r : nullptr;
}

// Superscripts: the character they stand for (digits, letters, + - = ( )), 0 for anything else.
char super_char(uint32_t cp) {
    switch (cp) {
        case 0x2070: return '0';
        case 0xB9: return '1';
        case 0xB2: return '2';
        case 0xB3: return '3';
        case 0x2071: return 'i';
        case 0x207A: return '+';
        case 0x207B: return '-';
        case 0x207C: return '=';
        case 0x207D: return '(';
        case 0x207E: return ')';
        case 0x207F: return 'n';
        case 0x2B0: return 'h';
        case 0x2B2: return 'j';
        case 0x2B3: return 'r';
        case 0x2B7: return 'w';
        case 0x2B8: return 'y';
        case 0x2E1: return 'l';
        case 0x2E2: return 's';
        case 0x2E3: return 'x';
        case 0x1D2C: return 'A';
        case 0x1D2E: return 'B';
        case 0x1D30: return 'D';
        case 0x1D31: return 'E';
        case 0x1D33: return 'G';
        case 0x1D34: return 'H';
        case 0x1D35: return 'I';
        case 0x1D36: return 'J';
        case 0x1D37: return 'K';
        case 0x1D38: return 'L';
        case 0x1D39: return 'M';
        case 0x1D3A: return 'N';
        case 0x1D3C: return 'O';
        case 0x1D3E: return 'P';
        case 0x1D3F: return 'R';
        case 0x1D40: return 'T';
        case 0x1D41: return 'U';
        case 0x1D42: return 'W';
        case 0x1D43: return 'a';
        case 0x1D47: return 'b';
        case 0x1D48: return 'd';
        case 0x1D49: return 'e';
        case 0x1D4D: return 'g';
        case 0x1D4F: return 'k';
        case 0x1D50: return 'm';
        case 0x1D52: return 'o';
        case 0x1D56: return 'p';
        case 0x1D57: return 't';
        case 0x1D58: return 'u';
        case 0x1D5B: return 'v';
        case 0x1D9C: return 'c';
        case 0x1DA0: return 'f';
        case 0x1DBB: return 'z';
        case 0x2C7D: return 'V';
        default: return cp >= 0x2074 && cp <= 0x2079 ? static_cast<char>('0' + cp - 0x2070) : 0;
    }
}

// Subscripts: the character they stand for, 0 for anything else.
char sub_char(uint32_t cp) {
    if (cp >= 0x2080 && cp <= 0x2089) return static_cast<char>('0' + cp - 0x2080);
    switch (cp) {
        case 0x208A: return '+';
        case 0x208B: return '-';
        case 0x208C: return '=';
        case 0x208D: return '(';
        case 0x208E: return ')';
        case 0x2090: return 'a';
        case 0x2091: return 'e';
        case 0x2092: return 'o';
        case 0x2093: return 'x';
        case 0x2094: return 'e';
        case 0x2095: return 'h';
        case 0x2096: return 'k';
        case 0x2097: return 'l';
        case 0x2098: return 'm';
        case 0x2099: return 'n';
        case 0x209A: return 'p';
        case 0x209B: return 's';
        case 0x209C: return 't';
        case 0x1D62: return 'i';
        case 0x1D63: return 'r';
        case 0x1D64: return 'u';
        case 0x1D65: return 'v';
        case 0x2C7C: return 'j';
        default: return 0;
    }
}

bool combining(uint32_t cp) { return cp >= 0x300 && cp <= 0x36F; }

// A letter and a combining mark that make a letter the model knows: e + U+0301 -> é.
const char* compose(uint32_t base, uint32_t mark) {
    struct C {
        char b;
        uint16_t m;
        const char* s;
    };
    static const C TABLE[] = {
        {'A', 0x300, "\xC3\x80"}, {'A', 0x301, "\xC3\x81"}, {'A', 0x302, "\xC3\x82"}, {'C', 0x327, "\xC3\x87"},
        {'E', 0x301, "\xC3\x89"}, {'E', 0x302, "\xC3\x8A"}, {'O', 0x308, "\xC3\x96"}, {'U', 0x308, "\xC3\x9C"},
        {'a', 0x300, "\xC3\xA0"}, {'a', 0x301, "\xC3\xA1"}, {'a', 0x302, "\xC3\xA2"}, {'a', 0x303, "\xC3\xA3"},
        {'a', 0x308, "\xC3\xA4"}, {'c', 0x327, "\xC3\xA7"}, {'e', 0x300, "\xC3\xA8"}, {'e', 0x301, "\xC3\xA9"},
        {'e', 0x302, "\xC3\xAA"}, {'e', 0x308, "\xC3\xAB"}, {'i', 0x300, "\xC3\xAC"}, {'i', 0x301, "\xC3\xAD"},
        {'i', 0x302, "\xC3\xAE"}, {'i', 0x308, "\xC3\xAF"}, {'n', 0x303, "\xC3\xB1"}, {'o', 0x300, "\xC3\xB2"},
        {'o', 0x301, "\xC3\xB3"}, {'o', 0x302, "\xC3\xB4"}, {'o', 0x303, "\xC3\xB5"}, {'o', 0x308, "\xC3\xB6"},
        {'u', 0x300, "\xC3\xB9"}, {'u', 0x301, "\xC3\xBA"}, {'u', 0x302, "\xC3\xBB"}, {'u', 0x308, "\xC3\xBC"},
        {'y', 0x308, "\xC3\xBF"},
    };
    for (const C& c : TABLE)
        if (static_cast<unsigned char>(c.b) == base && c.m == mark) return c.s;
    return nullptr;
}

// The token that contains byte b.
size_t token_of(const Doc& d, uint32_t b) {
    size_t k = d.token_at(b);
    if (k < d.size() && d[k].start == b) return k;
    return k - 1;
}

bool word_part(const Doc& d, size_t k) {
    return (d.word(k) && d[k].cp != 0xFDD0) || (d[k].kind == Kind::Punct && combining(d[k].cp));
}

// A run of letters with letters the model does not know, spelled with letters it knows: 'Łódź' -> 'Lódz',
// 'straße' -> 'strasse', 'cafe' + U+0301 -> 'café'. One or two Greek letters alone are said by name ('αβ');
// longer Greek words and other scripts stay. Returns the token after the run.
size_t letters(const Doc& d, Out& out, size_t k) {
    size_t a = k, b = k;
    while (a > 0 && d.joined(a) && word_part(d, a - 1)) --a;
    while (b + 1 < d.size() && d.joined(b + 1) && word_part(d, b + 1)) ++b;
    std::string_view src = d.slice(d[a].start, d[b].end);
    std::string res;
    int greek = 0, other = 0;
    uint32_t prev = 0;
    for (size_t i = 0; i < src.size();) {
        size_t j = i;
        uint32_t cp = rv::decode(src, j);
        if (cp >= 0x370 && cp <= 0x3FF) ++greek;
        else if (!combining(cp)) ++other;
        if (cp < 0x80 || known(cp)) {
            res.append(src.substr(i, j - i));
        } else if (combining(cp)) {
            const char* c = prev < 0x80 ? compose(prev, cp) : nullptr;
            if (c && !res.empty()) {
                res.pop_back();
                res += c;
            }
        } else if (const char* f = rv::latin_fold(cp)) {
            res += f;
        } else {
            res.append(src.substr(i, j - i));
        }
        prev = cp;
        i = j;
    }
    if (greek > 0 && other == 0 && greek <= 2) {  // 'αβ', 'ϰ'
        res.clear();
        for (size_t i = 0; i < src.size();) {
            std::string g = rv::variable_name(rv::decode(src, i));
            if (g.empty()) return b + 1;
            if (!res.empty()) res += ' ';
            res += g;
        }
    }
    if (!res.empty() && res != src) out.add(d[a].start, d[b].end, res, SCORE);
    return b + 1;
}

// A run of superscripts or subscripts starting at token k: 'xⁿ' -> 'x to the power of n', 'Na⁺' -> 'Na plus',
// 'aᵢ' -> 'a sub i', '1ˢᵗ' -> 'first'. Returns the token after the run.
size_t scripts(const Doc& d, Out& out, size_t k, bool super) {
    size_t e = k;
    std::string s;
    while (e < d.size() && (e == k || d.joined(e)) && d[e].kind == Kind::Punct) {
        char c = super ? super_char(d[e].cp) : sub_char(d[e].cp);
        if (!c) break;
        s += c;
        ++e;
    }
    if (s.empty()) return k + 1;
    size_t last = e - 1;
    bool after = k > 0 && d.joined(k) && (d[k - 1].kind != Kind::Punct || d.punct(k - 1, ')'));
    // ordinal suffixes after a number: 1ˢᵗ, 2ⁿᵈ, 3ʳᵈ, 4ᵗʰ
    if (super && after && d.number(k - 1) && (s == "st" || s == "nd" || s == "rd" || s == "th") &&
        d.str(k - 1).size() <= 6) {
        out.add(d[k - 1].start, d[last].end, rv::ordinal(std::stoull(std::string(d.str(k - 1)))), 40);
        return e;
    }
    // after a word of three letters or more a superscript is a footnote mark
    if (super && after && d.word(k - 1) && d.str(k - 1).size() > 2) {
        out.add(d[k].start, d[last].end, "", SCORE);
        return e;
    }
    std::string words;
    bool has_letter = false;
    for (size_t i = 0; i < s.size();) {
        std::string w;
        char c = s[i];
        if (c >= '0' && c <= '9') {
            size_t j = i;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
            w = rv::cardinal_digits(s.substr(i, j - i));
            i = j;
        } else {
            w = c == '+' ? "plus" : c == '-' ? "minus" : c == '=' ? "equals" : (c == '(' || c == ')') ? "" : rv::variable_name(static_cast<unsigned char>(c));
            has_letter |= (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
            ++i;
        }
        if (w.empty()) continue;
        if (!words.empty()) words += ' ';
        words += w;
    }
    if (words.empty()) {
        out.add(d[k].start, d[last].end, "", SCORE);
        return e;
    }
    if (after && super) {
        if (s == "2") words = "squared";
        else if (s == "3") words = "cubed";
        else if (s != "+" && s != "-") words = "to the power of " + words;
    } else if (after && has_letter) {
        words = "sub " + words;
    }
    out.add(d[k].start, d[last].end, words, SCORE);
    return e;
}

// Mathematical alphanumerics (𝐱, 𝑓, 𝟐) the engine did not fold: a joined run in plain letters.
size_t math_letters(const Doc& d, Out& out, size_t k) {
    size_t e = k;
    std::string res;
    while (e < d.size() && (e == k || d.joined(e)) && d[e].kind == Kind::Punct) {
        uint32_t p = rv::plain_math_alnum(d[e].cp);
        if (!p) break;
        rv::append_utf8(res, p);
        ++e;
    }
    out.add(d[k].start, d[e - 1].end, res, SCORE);
    return e;
}

// '∂f/∂x' -> 'the partial derivative of f with respect to x' (clearly math, never prose).
bool partial_derivative(const Doc& d, Out& out, size_t k) {
    size_t f = k + 1, s = k + 2, p = k + 3, x = k + 4;
    if (x >= d.size() || !d.word(f) || !d.punct(s, '/') || !d.punct(p, 0x2202) || !d.word(x)) return false;
    if (d.str(f).size() > 3 || d.str(x).size() > 3 || !d.joined(f) || !d.joined(x)) return false;
    if (x + 1 < d.size() && d.joined(x + 1) && d[x + 1].kind != Kind::Punct) return false;
    auto name = [&](size_t t) {  // a one-letter variable as the speech model says it best
        std::string_view w = d.str(t);
        size_t q = 0;
        uint32_t cp = rv::decode(w, q);
        std::string v = q == w.size() ? rv::variable_name(cp) : "";
        return v.empty() ? std::string(w) : v;
    };
    out.add(d[k].start, d[x].end, "the partial derivative of " + name(f) + " with respect to " + name(x), 50);
    return true;
}

void symbol(const Doc& d, Out& out, size_t k, Act act, const char* s) {
    const rv::Token& t = d[k];
    switch (act) {
        case WORDS: out.tokens(k, k, s, SCORE); return;
        case CHAR:
            if (t.cp == 0x2212 && k + 1 < d.size() && d.number(k + 1) && d.joined(k + 1))
                out.tokens(k, k, "minus", SCORE);  // '−5' where no number rule read it
            else
                out.tokens(k, k, s, SCORE);
            return;
        case DROP: out.tokens(k, k, "", SCORE); return;
        case SEP: {
            bool line_start = k == 0 || t.newline_before;
            bool line_end = k + 1 >= d.size() || d[k + 1].newline_before;
            if (line_start || line_end || d[k - 1].kind == Kind::Punct) out.tokens(k, k, "", SCORE);
            else out.add(d[k - 1].end, t.end, ",", SCORE);
            return;
        }
        case TIMES: {
            bool joined = k > 0 && d.joined(k) && k + 1 < d.size() && d.joined(k + 1);
            if (joined && d.number(k - 1) && d.number(k + 1)) out.tokens(k, k, "times", SCORE);
            else if (joined && d.word(k - 1) && d.word(k + 1)) out.tokens(k, k, " ", SCORE);  // N·m
            else symbol(d, out, k, SEP, ",");
            return;
        }
        case SET: {
            // ℝ², ℝⁿ: R two, R n
            if (k + 1 < d.size() && d.joined(k + 1) && d[k + 1].kind == Kind::Punct && super_char(d[k + 1].cp)) {
                std::string w = t.cp == 0x211D ? "R" : t.cp == 0x2115 ? "N" : t.cp == 0x2124 ? "Z" : t.cp == 0x211A ? "Q"
                                : t.cp == 0x2102 ? "C" : t.cp == 0x210D ? "H" : "P";
                size_t e = k + 1;
                while (e < d.size() && d.joined(e) && d[e].kind == Kind::Punct && super_char(d[e].cp)) {
                    char c = super_char(d[e].cp);
                    w += ' ';
                    w += c >= '0' && c <= '9' ? rv::cardinal(static_cast<uint64_t>(c - '0')) : rv::variable_name(static_cast<unsigned char>(c));
                    ++e;
                }
                out.add(t.start, d[e - 1].end, w, SCORE);
                return;
            }
            out.tokens(k, k, s, SCORE);
            return;
        }
    }
}

// Primes: 'f′' -> 'f prime', '5′' -> 'five feet' (minutes after degrees), 'don′t' -> "don't".
void prime(const Doc& d, Out& out, size_t k) {
    uint32_t cp = d[k].cp;
    int n = cp == 0x2032 ? 1 : cp == 0x2033 ? 2 : 3;
    bool after = k > 0 && d.joined(k);
    if (after && d.word(k - 1) && k + 1 < d.size() && d.joined(k + 1) && d.word(k + 1) && n == 1) {
        out.tokens(k, k, "'", SCORE);  // an apostrophe
        return;
    }
    if (after && d.number(k - 1) && n < 3) {
        bool degrees = false;
        for (size_t j = k - 1; j-- > 0 && j + 5 > k && !d[j + 1].newline_before;) degrees |= d.punct(j, 0xB0);
        bool one = d.str(k - 1) == "1";
        const char* w = degrees ? (n == 1 ? (one ? "minute" : "minutes") : (one ? "second" : "seconds"))
                                : (n == 1 ? (one ? "foot" : "feet") : (one ? "inch" : "inches"));
        out.tokens(k, k, w, SCORE);
        return;
    }
    if (after && d[k - 1].kind != Kind::Punct) {
        static const char* const P[] = {"", "prime", "double prime", "triple prime"};
        out.tokens(k, k, P[n], SCORE);
        return;
    }
    out.tokens(k, k, n == 1 ? "'" : "\"", SCORE);  // used as a quotation mark
}

// Plain-text quotation marks as the curly quotes the model knows, in pairs: ``...'' -> “...”, ‚...‘ -> ‘...’,
// «...» and »...« -> “...”; a backtick inside a word is an apostrophe ("it`s"); a spaced » between words is a
// separator ("Home » News").
// (Quotations in other languages keep their marks; ‹ › and the rarer marks are in the table above.)
RV_RULE("chars", quotes) {
    std::string_view t = doc.text;
    if (!(t.find("\xE2\x80\x9A") != std::string_view::npos || t.find('`') != std::string_view::npos ||
          t.find("\xC2\xBB") != std::string_view::npos || t.find("\xC2\xAB") != std::string_view::npos))
        return;
    size_t n = doc.size();
    auto line_end = [&](size_t k) {
        while (k + 1 < n && !doc[k + 1].newline_before) ++k;
        return k;
    };
    for (size_t i = 0; i < n; ++i) {
        if (doc[i].kind != Kind::Punct) continue;
        uint32_t c = doc[i].cp;
        if (c == '`') {
            if (i > 0 && doc.joined(i) && doc.word(i - 1) && doc.word(i + 1) && doc.joined(i + 1)) {
                out.tokens(i, i, "'", SCORE);
                continue;
            }
            if (!(doc.punct(i + 1, '`') && doc.joined(i + 1))) continue;
            for (size_t k = i + 2, e = line_end(i); k + 1 <= e; ++k) {
                if (doc.punct(k, '\'') && doc.punct(k + 1, '\'') && doc.joined(k + 1)) {
                    out.tokens(i, i + 1, "\xE2\x80\x9C", SCORE);
                    out.tokens(k, k + 1, "\xE2\x80\x9D", SCORE);
                    i = k + 1;
                    break;
                }
            }
        } else if (c == 0x201A) {
            size_t e = line_end(i), k = i + 1;
            while (k <= e && !doc.punct(k, 0x2018) && !doc.punct(k, 0x2019)) ++k;
            if (k <= e && doc.punct(k, 0x2018)) {
                out.tokens(i, i, "\xE2\x80\x98", SCORE);
                out.tokens(k, k, "\xE2\x80\x99", SCORE);
                i = k;
            }
        } else if (c == 0xAB || c == 0xBB) {  // «...» and »...« as the outer marks of a pair
            uint32_t other = c == 0xAB ? 0xBB : 0xAB;
            size_t e = line_end(i), k = i + 1;
            while (k <= e && !doc.punct(k, other)) ++k;
            if (k <= e) {
                out.tokens(i, i, "\xE2\x80\x9C", SCORE);
                out.tokens(k, k, "\xE2\x80\x9D", SCORE);
                i = k;
                continue;
            }
        }
        if (c == 0xBB && i > 0 && doc[i].space_before && !doc[i].newline_before && doc.word(i - 1) && i + 1 < n &&
            doc[i + 1].space_before && !doc[i + 1].newline_before && doc.word(i + 1)) {
            bool guillemet = false;  // a quotation in guillemets on the line
            for (size_t k = i; k-- > 0 && !doc[k + 1].newline_before && !guillemet;) guillemet = doc.punct(k, 0xAB);
            for (size_t k = i + 1, e = line_end(i); k <= e && !guillemet; ++k) guillemet = doc.punct(k, 0xAB);
            if (!guillemet) out.add(doc[i - 1].end, doc[i].end, ",", SCORE);
        }
    }
}

RV_RULE("chars", unknown) {
    std::string_view t = doc.text;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(t.data());
    size_t n = t.size(), i = 0;
    while (i + 8 <= n) {  // ASCII eight bytes at a time
        uint64_t w;
        std::memcpy(&w, b + i, 8);
        if (w & 0x8080808080808080ull) break;
        i += 8;
    }
    while (i < n && b[i] < 0x80) ++i;
    size_t done = 0;  // tokens before this one are read
    while (i < n) {
        if (b[i] < 0x80) {
            ++i;
            continue;
        }
        size_t j = i;
        uint32_t cp = rv::decode(t, j);
        if (known(cp)) {
            i = j;
            continue;
        }
        if (rv::is_space(cp)) {  // no-break and thin spaces are spaces
            out.add(static_cast<uint32_t>(i), static_cast<uint32_t>(j), " ", SCORE);
            i = j;
            continue;
        }
        size_t k = token_of(doc, static_cast<uint32_t>(i));
        if (k < done || k >= doc.size()) {
            i = j;
            continue;
        }
        size_t next = k + 1;
        if (doc[k].cp == 0xFDD0) {
            // a marked formula: the math module reads it
        } else if (doc.word(k) || combining(cp)) {
            next = letters(doc, out, k);
        } else if (rv::plain_math_alnum(cp)) {
            next = math_letters(doc, out, k);
        } else if (super_char(cp) && !(cp >= 0x2B0 && cp <= 0x2E3 && !doc.joined(k))) {
            next = scripts(doc, out, k, true);
        } else if (sub_char(cp)) {
            next = scripts(doc, out, k, false);
        } else if (cp == 0x2032 || cp == 0x2033 || cp == 0x2034) {
            prime(doc, out, k);
        } else if (cp == 0x2202 && partial_derivative(doc, out, k)) {
            next = k + 5;
        } else if (const Reading* r = reading_of(cp)) {
            symbol(doc, out, k, r->act, r->s);
        }
        done = next;
        i = next < doc.size() ? doc[next].start : n;
    }
}

}  // namespace
