#include "core/utf8.h"

namespace rv {

uint32_t decode(std::string_view s, size_t& i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        ++i;
        return c;
    }
    int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : -1;
    if (n < 0 || i + static_cast<size_t>(n) >= s.size()) {
        ++i;
        return 0xFFFD;
    }
    uint32_t cp = c & (0x3F >> n);
    for (int k = 1; k <= n; ++k) {
        unsigned char d = static_cast<unsigned char>(s[i + k]);
        if ((d & 0xC0) != 0x80) {
            ++i;
            return 0xFFFD;
        }
        cp = (cp << 6) | (d & 0x3F);
    }
    i += n + 1;
    return cp;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

bool is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    if (cp == 0xAA || cp == 0xB5 || cp == 0xBA) return true;
    if (cp >= 0xC0 && cp <= 0x24F) return cp != 0xD7 && cp != 0xF7;
    if (cp >= 0x386 && cp <= 0x3FF) return cp != 0x387;
    if (cp >= 0x400 && cp <= 0x52F) return cp < 0x482 || cp > 0x489;
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;   // kana
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;   // CJK
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;   // Hangul
    return false;
}

bool is_upper(uint32_t cp) {
    if (cp < 0x80) return cp >= 'A' && cp <= 'Z';
    if (cp >= 0xC0 && cp <= 0xDE) return cp != 0xD7;
    if (cp >= 0x100 && cp <= 0x17F) return cp % 2 == 0;
    if (cp >= 0x391 && cp <= 0x3A9) return true;
    if (cp >= 0x400 && cp <= 0x42F) return true;
    return false;
}

bool is_lower(uint32_t cp) {
    if (cp < 0x80) return cp >= 'a' && cp <= 'z';
    if (cp >= 0xDF && cp <= 0xFF) return cp != 0xF7;
    if (cp >= 0x100 && cp <= 0x17F) return cp % 2 == 1;
    if (cp >= 0x3AC && cp <= 0x3CE) return true;
    if (cp >= 0x430 && cp <= 0x45F) return true;
    return false;
}

bool is_invisible(uint32_t cp) {
    return cp == 0xAD || cp == 0x34F || cp == 0x180E || (cp >= 0x200B && cp <= 0x200F) ||
           (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2060 && cp <= 0x2064) || (cp >= 0x2066 && cp <= 0x2069) ||
           cp == 0xFEFF || (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0xE0000 && cp <= 0xE007F) ||
           (cp >= 0xE0100 && cp <= 0xE01EF) || cp == 0x20E3;
}

bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v' || cp == 0xA0 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

}  // namespace rv
