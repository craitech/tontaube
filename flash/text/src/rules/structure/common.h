#pragma once
// Shared helpers of the module "structure": character classes, lines, and deletions that tidy the spaces
// around what they remove.
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/rule.h"

namespace st {

using rv::Doc;
using rv::Kind;
using rv::Out;

// Zero-width spaces and joiners, soft hyphen, BOM, direction marks, variation selectors, tag characters.
inline bool invisible_cp(uint32_t cp) {
    return cp == 0xAD || cp == 0x34F || cp == 0x180E || (cp >= 0x200B && cp <= 0x200F) ||
           (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2060 && cp <= 0x2064) || (cp >= 0x2066 && cp <= 0x2069) ||
           cp == 0xFEFF || (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xE0000 && cp <= 0xE007F) ||
           (cp >= 0xE0100 && cp <= 0xE01EF) || cp == 0x20E3;
}

// Emoji and pictographs that are removed wherever they stand. Stars, check marks, arrows, keyboard symbols and
// trademark signs are not in the list: other rules read them.
inline bool emoji_cp(uint32_t cp) {
    if (cp >= 0x1F000 && cp <= 0x1FAFF) return true;
    if (cp >= 0x2600 && cp <= 0x26FF) return cp != 0x2605 && cp != 0x2606;  // misc symbols, not the stars
    if (cp >= 0x2700 && cp <= 0x27BF)  // dingbats, not the check marks, crosses, circled numbers and arrows
        return !(cp == 0x2713 || cp == 0x2714 || cp == 0x2717 || cp == 0x2718 || (cp >= 0x2776 && cp <= 0x2793) ||
                 (cp >= 0x2794 && cp <= 0x27BF && cp != 0x27B0 && cp != 0x27BF));
    switch (cp) {
        case 0x231A: case 0x231B: case 0x23E9: case 0x23EA: case 0x23EB: case 0x23EC: case 0x23ED: case 0x23EE:
        case 0x23EF: case 0x23F0: case 0x23F1: case 0x23F2: case 0x23F3: case 0x23F8: case 0x23F9: case 0x23FA:
        case 0x2B1B: case 0x2B1C: case 0x2B55: case 0x2B50: case 0x3030: case 0x303D: case 0x3297: case 0x3299: case 0x2139:
            return true;
        default:
            return false;
    }
}

// Symbols that only decorate the start of a line or label: arrows heads, squares, diamonds, circles.
inline bool decor_cp(uint32_t cp) {
    switch (cp) {
        case 0x25A0: case 0x25A1: case 0x25AA: case 0x25AB: case 0x25B2: case 0x25B6: case 0x25B8: case 0x25B9:
        case 0x25BA: case 0x25BC: case 0x25B7: case 0x25C6: case 0x25C7: case 0x25C9: case 0x25CB: case 0x25CF: case 0x25E6:
        case 0x2726: case 0x2727: case 0x272A: case 0x2731: case 0x2733: case 0x271A: case 0x2756: case 0x2794:
        case 0x279C: case 0x27A4: case 0x27A2: case 0x27A3: case 0x276F: case 0x2023: case 0x2043: case 0x261B:
        case 0x261E: case 0x2192: case 0x21D2: case 0x2713: case 0x2714: case 0x2717: case 0x2718:
            return true;
        default:
            return false;
    }
}

inline bool is_emoji(const Doc& d, size_t i) { return i < d.size() && d[i].kind == Kind::Punct && emoji_cp(d[i].cp); }
inline bool is_invisible(const Doc& d, size_t i) {
    return i < d.size() && d[i].kind == Kind::Punct && invisible_cp(d[i].cp);
}
inline bool is_punct(const Doc& d, size_t i) { return i < d.size() && d[i].kind == Kind::Punct; }
inline bool p(const Doc& d, size_t i, uint32_t cp) { return d.punct(i, cp); }

// Sentence punctuation that ends a line properly.
inline bool terminal_cp(uint32_t cp) {
    return cp == '.' || cp == '!' || cp == '?' || cp == ':' || cp == ';' || cp == ',' || cp == 0x2026 ||
           cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F;
}

// A line of tokens [a, b): b > a. nl is the byte offset of the line break after the line (text size if none).
struct Line {
    size_t a, b;
    uint32_t nl;
    int blank_before;  // number of empty lines before this line (0 for the first line)
};
std::vector<Line> split_lines(const Doc& d);

// The whitespace between token i-1 and token i (i == size(): after the last token; i == 0: before the first).
inline uint32_t gap_start(const Doc& d, size_t i) { return i == 0 ? 0 : d[i - 1].end; }
inline uint32_t gap_end(const Doc& d, size_t i) {
    return i < d.size() ? d[i].start : static_cast<uint32_t>(d.text.size());
}
// The newlines of a gap, without the spaces and tabs around them; "" for a gap without line breaks.
std::string gap_newlines(const Doc& d, size_t i);

// Byte span that deletes tokens first..last and tidies the spaces around: at the start or end of a line, or
// before punctuation that attaches to the left, the neighbouring spaces go too; between two words one stays.
struct Span {
    uint32_t start, end;
};
Span deletion(const Doc& d, size_t first, size_t last);
// Token i is punctuation that attaches to the word before it (, . ; : ! ? ) ] ” ’ …).
bool attaches_left(const Doc& d, size_t i);

// Tokens a..b are Latin text whose only parts in another script are names or pronunciations that the glosses rule
// removes ("Nepali: सगरमाथा", "सगरमाथा (Nepali)"); the language rule does not keep such a sentence.
bool gloss_sentence(const Doc& d, size_t a, size_t b);

// An HTML or XML tag (<b>, </p>, <a href="...">, <tool_call>), a chat-template token (<|im_end|>) or an
// instruction marker ([INST], [/INST]) that starts at token i: the index of its last token, or 0. block is set
// for tags that separate text (p, br, div, li, h1...).
size_t tag_at(const Doc& d, size_t i, bool* block = nullptr);

// Quick checks on the raw text, so that a rule whose trigger characters are absent returns at once.
// (memchr for the first byte; string_view::find with a longer needle scans byte by byte)
inline bool has(const Doc& d, std::string_view s) {
    const char* q = d.text.data();
    const char* end = q + d.text.size();
    while (static_cast<size_t>(end - q) >= s.size()) {
        const void* hit = std::memchr(q, s[0], static_cast<size_t>(end - q) - s.size() + 1);
        if (!hit) return false;
        q = static_cast<const char*>(hit);
        if (std::memcmp(q, s.data(), s.size()) == 0) return true;
        ++q;
    }
    return false;
}
inline bool has_any(const Doc& d, const char* bytes) {
    for (const char* b = bytes; *b; ++b)
        if (std::memchr(d.text.data(), *b, d.text.size())) return true;
    return false;
}

// Calls f(k) for every byte offset k of the character c in the text (memchr), until f returns true; true if it did.
template <class F>
bool each_byte(std::string_view t, char c, F f) {
    const char* b = t.data();
    const char* e = b + t.size();
    for (const char* q = b; q < e;) {
        const void* hit = std::memchr(q, c, static_cast<size_t>(e - q));
        if (!hit) return false;
        size_t k = static_cast<size_t>(static_cast<const char*>(hit) - b);
        if (f(k)) return true;
        q = static_cast<const char*>(hit) + 1;
    }
    return false;
}

// The text may contain an invisible character, a ligature or a special hyphen (see invisible_cp).
inline bool maybe_invisible(const Doc& d) {
    std::string_view t = d.text;
    for (size_t k = 0; k + 1 < t.size(); ++k) {
        unsigned char c = static_cast<unsigned char>(t[k]), n = static_cast<unsigned char>(t[k + 1]);
        if (c < 0xC2) continue;
        if ((c == 0xC2 && n == 0xAD) || c == 0xCD || (c == 0xE1 && n == 0xA0) || c == 0xF3 ||
            (c == 0xEF && (n == 0xBB || n == 0xB8 || n == 0xAC)) || (c == 0xE2 && (n == 0x81 || n == 0x83)))
            return true;
        if (c == 0xE2 && n == 0x80 && k + 2 < t.size()) {
            unsigned char b = static_cast<unsigned char>(t[k + 2]);
            if ((b >= 0x8B && b <= 0x91) || (b >= 0xAA && b <= 0xAE)) return true;
        }
    }
    return false;
}
// The text may contain an emoji, a pictograph or a decorative symbol (U+2100 and up, outside punctuation), or one
// of the bullets ‣ and ⁃.
inline bool maybe_symbol(const Doc& d) {
    std::string_view t = d.text;
    if (std::memchr(t.data(), '\xF0', t.size()) || std::memchr(t.data(), '\xE3', t.size())) return true;
    return each_byte(t, '\xE2', [&](size_t k) {
        if (k + 2 >= t.size()) return false;
        unsigned char n = static_cast<unsigned char>(t[k + 1]), m = static_cast<unsigned char>(t[k + 2]);
        return n >= 0x84 || (n == 0x80 && m == 0xA3) || (n == 0x81 && m == 0x83);
    });
}

// A word of up to 7 bytes packed into an integer (length in the top byte); 0 for longer words.
inline uint64_t pack7(std::string_view w) {
    if (w.empty() || w.size() > 7) return 0;
    uint64_t v = static_cast<uint64_t>(w.size()) << 56;
    for (size_t k = 0; k < w.size(); ++k) v |= static_cast<uint64_t>(static_cast<unsigned char>(w[k])) << (8 * k);
    return v;
}

// A small set of short words with a value each, looked up by one multiply and an integer compare or two. The
// rules look up most words of a text, so this is cheaper than a binary search over strings.
class ShortWords {
public:
    ShortWords() = default;
    ShortWords(std::initializer_list<std::pair<std::string_view, int>> items) {
        for (const auto& [w, v] : items) add(w, v);
    }
    void add(std::string_view w, int v) {
        uint64_t k = pack7(w);
        if (!k) return;
        uint32_t h = slot(k);
        while (keys_[h] && keys_[h] != k) h = (h + 1) & (N - 1);
        keys_[h] = k;
        vals_[h] |= v;
        filter_[(k * 0xC2B2AE3D27D4EB4Full) >> 58] |= 1ull << (k & 63);
    }
    // The value of w (OR of the values it was given), 0 if absent.
    int get(std::string_view w) const { return get_packed(pack7(w)); }
    int get_packed(uint64_t k) const {
        if (!k || !(filter_[(k * 0xC2B2AE3D27D4EB4Full) >> 58] & (1ull << (k & 63)))) return 0;
        for (uint32_t h = slot(k);; h = (h + 1) & (N - 1)) {
            if (keys_[h] == k) return vals_[h];
            if (!keys_[h]) return 0;
        }
    }

private:
    static constexpr uint32_t N = 1024;
    static uint32_t slot(uint64_t k) { return static_cast<uint32_t>((k * 0x9E3779B97F4A7C15ull) >> 54); }
    uint64_t keys_[N] = {};
    int vals_[N] = {};
    uint64_t filter_[64] = {};  // a quick "certainly absent" test before the table
};

// Initialisms said as letters: kept in capitals in an all-caps heading, written in capitals in a log tag.
bool initialism(std::string_view lower_word);

// Lower-case ASCII copy.
std::string lower(std::string_view s);
// "One" for 1: a capitalised cardinal.
std::string cap_cardinal(uint64_t n);

}  // namespace st
