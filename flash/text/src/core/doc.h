#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rv {

// Word: letters, with apostrophes between letters ("can't", "O'Neil"); also a whole marked formula (see
// formula_end), so that no rule reads its source. Number: a run of ASCII digits. Punct: any other single code
// point (punctuation, symbols, emoji, invisible characters). Whitespace is not a token; it shows up as
// space_before / newline_before on the next token.
enum class Kind : uint8_t { Word, Number, Punct };

// Casing of a Word token.
enum Case : uint8_t {
    LOWER = 1,  // "port"
    UPPER = 2,  // "FBI", "A", "I"
    TITLE = 4,  // "Room"
    MIXED = 8,  // "iPhone", "GitLab"
};

struct Token {
    uint32_t start, end;  // byte offsets in Doc::text
    uint32_t cp;          // first code point
    Kind kind;
    uint8_t casing;       // Case flag for words, 0 otherwise
    bool space_before;    // whitespace before this token (true for the first token)
    bool newline_before;  // a line break before this token
};

class Doc {
public:
    explicit Doc(std::string_view text);

    std::string_view text;
    std::string lower;           // text with ASCII letters lowered; same byte offsets as text
    std::vector<Token> tokens;

    size_t size() const { return tokens.size(); }
    const Token& operator[](size_t i) const { return tokens[i]; }

    std::string_view str(size_t i) const;    // the token's text
    std::string_view low(size_t i) const;    // the token's text, ASCII lowered
    std::string_view slice(uint32_t start, uint32_t end) const { return text.substr(start, end - start); }

    bool word(size_t i) const { return i < size() && tokens[i].kind == Kind::Word; }
    bool number(size_t i) const { return i < size() && tokens[i].kind == Kind::Number; }
    // Token i is the single code point cp.
    bool punct(size_t i, uint32_t cp) const {
        return i < size() && tokens[i].kind == Kind::Punct && tokens[i].cp == cp;
    }
    // Token i equals w, ignoring ASCII case (pass w in lower case).
    bool is(size_t i, std::string_view w) const { return i < size() && low(i) == w; }
    // Token i directly follows token i-1 with no whitespace between.
    bool joined(size_t i) const { return i > 0 && i < size() && !tokens[i].space_before; }

    // The nth Word or Number token before / after token i, skipping punctuation; npos if none.
    size_t prev_word(size_t i, int n = 1) const;
    size_t next_word(size_t i, int n = 1) const;
    // Token i starts a sentence: start of text, a line break, or . ! ? … before it (quotes and brackets between).
    bool sentence_start(size_t i) const;
    // Index of the first token that starts at or after byte offset b (size() if none).
    size_t token_at(uint32_t b) const;
    // The text at byte offset b starts with s (exact bytes) / with s ignoring ASCII case (pass s in lower case).
    bool text_at(uint32_t b, std::string_view s) const { return text.substr(b, s.size()) == s; }
    bool lower_at(uint32_t b, std::string_view s) const {
        return std::string_view(lower).substr(b, s.size()) == s;
    }

    static constexpr size_t npos = static_cast<size_t>(-1);
};

// A formula the page extractor marked, U+FDD0 TeX U+FDD1, starts at byte b: the offset after its closing
// marker, or without one the end of the line or the next opening marker. Its token has cp U+FDD0, casing 0.
size_t formula_end(std::string_view text, size_t b);

// A written number of the common US form starting at token i, its tokens joined without spaces:
// "42", "1,234,567", "3.14", "1,234.5". ok is false when token i is not a Number.
struct NumberSpan {
    bool ok = false;
    size_t first = 0, last = 0;  // tokens
    std::string whole;           // digits of the whole part, separators removed
    std::string fraction;        // digits after the point, empty if none
    bool grouped = false;        // written with thousands separators
};
NumberSpan parse_number(const Doc& doc, size_t i);

}  // namespace rv
