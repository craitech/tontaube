#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace rv {

// Decodes the code point at s[i] and advances i past it. Invalid bytes decode as U+FFFD, one byte at a time.
uint32_t decode(std::string_view s, size_t& i);
void append_utf8(std::string& out, uint32_t cp);

// Letters of the scripts a text may contain (Latin with accents, Greek, Cyrillic, CJK, Hangul).
bool is_letter(uint32_t cp);
bool is_upper(uint32_t cp);
bool is_lower(uint32_t cp);
// Whitespace, including no-break and thin spaces (U+00A0, U+2000-200A, U+202F, U+3000).
bool is_space(uint32_t cp);
// Characters that are never spoken and are removed before tokenizing: zero-width spaces and joiners, soft hyphen,
// BOM, direction marks, variation selectors, tag characters, the combining keycap.
bool is_invisible(uint32_t cp);

// The plain letter or digit of a mathematical alphanumeric symbol (U+1D400-U+1D7FF: bold, italic, script,
// fraktur, double-struck, sans-serif, monospace; Greek ones give the Greek letter) and of the letterlike
// symbols that fill the holes of those alphabets (italic h, script and fraktur capitals). 0 for anything else.
uint32_t plain_math_alnum(uint32_t cp);
// The spelling with the letters a speech model knows of a Latin letter with a diacritic or a special Latin
// letter (U+00C0-U+024F, U+1E00-U+1EFF): 'Ł' -> "L", 'ß' -> "ss", 'ő' -> "ö", 'þ' -> "th". nullptr for letters
// the model knows as they are (é, ü, ñ, æ, Œ, ...) and for anything else.
const char* latin_fold(uint32_t cp);

inline bool is_ascii_digit(char c) { return c >= '0' && c <= '9'; }
inline bool is_ascii_alnum(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

}  // namespace rv
