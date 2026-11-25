#pragma once
// Shared parts of the module numbers_core: a number scanner that knows every written form the module reads
// (thousands separators of several styles, a leading point), and number words.
#include <string>
#include <string_view>

#include "core/rule.h"

namespace numbers_core {

using rv::Doc;

// Invisible characters are removed before tokenizing (rv::prepare), so none sits inside a number any more. These
// helpers keep the rules' wording: the next and previous token, and token j right after token k.
inline bool invisible_tok(const Doc&, size_t) { return false; }
inline size_t next_tok(const Doc&, size_t k, bool* = nullptr) { return k + 1; }
inline size_t prev_tok(const Doc& d, size_t k, bool* = nullptr) { return k == 0 || k > d.size() ? Doc::npos : k - 1; }
inline bool glued(const Doc& d, size_t k, size_t j) { return j == k + 1 && j < d.size() && !d[j].space_before; }
// Exactly one space (plain, no-break or narrow) between tokens a and b.
bool one_space(const Doc& d, size_t a, size_t b);
inline int score_span(const Doc&, size_t, size_t, int score) { return score; }

struct Num {
    size_t first = 0, last = 0;  // tokens, inclusive
    std::string whole;           // digits of the whole part, separators removed ("" for ".5")
    std::string frac;            // digits after the point
    bool point = false;          // written with a decimal point
    char sep = 0;                // thousands separator: ',', ' ', '\'', '_', or 0
    bool invisible = false;      // invisible characters inside
    bool integer() const { return !point && !whole.empty(); }
};

// A cheap look at the number starting at Number token i, without building it: the last token it may reach,
// how many digits it has, and whether it uses a form only this module reads (space, apostrophe or underscore
// groups, invisible characters). Rules use it to skip numbers before scanning them.
struct Quick {
    size_t end = 0;
    size_t digits = 0;
    bool special = false;
};
Quick quick(const Doc& d, size_t i);

// Scans a written number starting at token i: a Number, or a '.' directly followed by one (".5"). Accepts
// "1,234,567.89", "1 234 567" (one space, no-break or thin space between groups), "27'530", "1_000_000".
// Returns false when token i does not start a number. Says nothing about what stands around the number.
bool scan(const Doc& d, size_t i, Num& n);

// Nothing sticks to the front of token i: no letter, digit, '.' or ',' directly before it, and no hyphen or
// sign that itself sticks to a letter or digit ("Qwen3-0.6B").
bool free_before(const Doc& d, size_t i);
// Number i is the second end of a range: "10–15%", "1st–3rd", "1980s–1990s".
bool after_range_dash(const Doc& d, size_t i);
// Nothing sticks to the back of token k: no letter or digit directly after it, and no '.' / ',' followed by a
// digit.
bool free_after(const Doc& d, size_t k);

// Number words. Whole numbers of any length up to 36 digits in full, longer ones digit by digit.
std::string whole_words(std::string_view digits);
std::string num_words(const Num& n);  // "" if the number cannot be read (leading zeros, empty)
std::string plural_of(std::string words);  // "nineteen sixty" -> "nineteen sixties"
uint32_t value4(const std::string& digits);  // value of up to 9 digits; 0 for longer

// An HTML entity starting at token i ("&ndash;", "&#8211;", "&#x2013;"): its code point, or 0; last is its ';'.
// Only the entities of the signs this module reads are known. (The core decodes spacing entities only.)
uint32_t entity_at(const Doc& d, size_t i, size_t& last);

// Words of tables shared by several rules.
bool count_word(std::string_view lower_word);  // nouns that are counted ("units", "people") and unit names
bool unit_word(std::string_view lower_word);   // unit names only ("px", "km", "miles")
bool scale_word(std::string_view lower_word);  // hundred, thousand, million, ...

}  // namespace numbers_core
