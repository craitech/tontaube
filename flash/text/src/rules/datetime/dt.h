#pragma once
// Shared helpers of the module "datetime": token tests, month and weekday tables, and the readings of
// dates, years, clock times and durations.
#include <string>
#include <string_view>

#include "core/rule.h"

namespace dt {

using rv::Doc;
constexpr size_t NPOS = Doc::npos;

// Scores (README score table).
enum : int {
    S_WEAK = 28,      // a shape alone that often means something else (5/9 without a cue)
    S_SHAPE = 50,     // the shape decides (10:50)
    S_CUE = 70,       // shape and a cue word
    S_SURE = 83,      // dates with a month name: they must win over the readings of their parts (Oct., 28th)
    S_CONSTRUCT = 85, // ISO dates and timestamps
};

// --- token tests -------------------------------------------------------------------------------------

inline bool is_dash_cp(uint32_t cp) {
    return cp == '-' || cp == 0x2010 || cp == 0x2011 || cp == 0x2012 || cp == 0x2013;
}
inline bool is_minus_cp(uint32_t cp) { return is_dash_cp(cp) || cp == 0x2212 || cp == 0x2014; }
inline bool dash(const Doc& d, size_t i) {
    return i < d.size() && d[i].kind == rv::Kind::Punct && is_dash_cp(d[i].cp);
}
inline bool punct_any(const Doc& d, size_t i) { return i < d.size() && d[i].kind == rv::Kind::Punct; }
inline bool apostrophe_cp(uint32_t cp) { return cp == '\'' || cp == 0x2019 || cp == 0x2018; }

// Value of a Number token with at most `max_len` digits, -1 otherwise.
inline int num(const Doc& d, size_t i, size_t max_len = 9) {
    if (!d.number(i)) return -1;
    std::string_view s = d.str(i);
    if (s.size() > max_len || s.size() > 9) return -1;
    int v = 0;
    for (char c : s) v = v * 10 + (c - '0');
    return v;
}
inline size_t len(const Doc& d, size_t i) { return d[i].end - d[i].start; }

// A range dash at i, with or without spaces around it, followed by token i+1 written the same way:
// "21–24", "21 – 24", but not "21 –24". Returns the token after the dash, or NPOS.
size_t range_dash(const Doc& d, size_t i);

// Token i is not glued to a letter, digit or number punctuation before it.
bool free_before(const Doc& d, size_t i);
// Token i is not glued to a letter, digit or number punctuation after it.
bool free_after(const Doc& d, size_t i);

// The lower-cased word before token i, skipping one ',' or '.' and quotes or brackets; "" if none.
std::string_view word_before(const Doc& d, size_t i);
size_t word_before_index(const Doc& d, size_t i);

// The '.' at token i ends the sentence as well (end of text, a line break, or a capitalized word after it
// that does not continue a date). Then a rule leaves the period in the text.
bool period_ends_sentence(const Doc& d, size_t i);
// Last token of an abbreviation at i: i+1 if a joined '.' follows that does not end the sentence, else i.
size_t abbrev_end(const Doc& d, size_t i);

// --- tables ------------------------------------------------------------------------------------------

extern const char* const MONTH_NAMES[13];
// Month at token i: a capitalized full name or abbreviation (May, Jan, Sept, ...). 0 if none. *abbr is set for
// abbreviations.
int month_at(const Doc& d, size_t i, bool* abbr = nullptr);
// A full month name in lower case (chat: "june 3"), except march (the verb); may is too short. 0 if none.
int month_lower_at(const Doc& d, size_t i);
// Weekday abbreviation (Mon, Tue, Tues, ..., capitalized) at token i: its full name, or nullptr.
const char* weekday_abbrev_at(const Doc& d, size_t i);
bool weekday_full_at(const Doc& d, size_t i);
bool valid_day(int month, int day);

// --- readings ----------------------------------------------------------------------------------------

std::string year_words(int y);             // 1870 eighteen seventy, 2005 two thousand five, 476 four hundred seventy-six
std::string year_pairs(int y);             // like year_words, but 3-digit years in pairs too (180 one eighty)
std::string two_digit_year(int yy);        // 24 twenty-four, 5 oh five
std::string decade_words(int y);           // 1990 nineteen nineties
std::string day_words(int d);              // ordinal
std::string clock_words(int h, int m);     // 9:05 nine oh five, 6:00 six o'clock, 14:00 fourteen hundred
std::string clock_short(int h, int m);     // in a dash range: on the hour 1-12 is the hour alone
std::string hm_words(int h, int m);        // 3:20 three twenty, 7:00 seven (with AM / PM after it)
std::string plural(int n, const char* unit);  // 1 hour, 2 hours

// Ordinal suffix st / nd / rd / th joined to the number at i: index of the suffix, or NPOS.
size_t ordinal_suffix(const Doc& d, size_t i);

// A four-digit year 1000-2999 at token i.
inline bool year4(const Doc& d, size_t i) {
    int v = num(d, i, 4);
    return v >= 1000 && v <= 2999 && len(d, i) == 4;
}

// AM / PM at token i in any spelling: AM, am, a.m., A.M., a.m; sets *last to its last token (leaving a
// sentence-final period out) and returns "AM" / "PM", or nullptr.
const char* ampm_at(const Doc& d, size_t i, size_t* last);

// A clock time starting at Number token i: H:MM, H:MM:SS(.frac), optional joined Z.
struct Clock {
    bool ok = false;
    size_t first = 0, last = 0;  // tokens of H:MM[:SS[.frac]]
    int h = 0, m = 0, s = -1;
    bool zulu = false;           // a joined 'Z' follows (included in last)
};
Clock parse_clock(const Doc& d, size_t i, int max_h = 24);  // max_h 99 for durations (56:14)

// The word after token `last` makes a colon pair a ratio, not a time (16:10 format, a 4:3 layout).
bool ratio_after(const Doc& d, size_t last);
// The word before token i makes a colon pair a score or a ratio: the score was 21:14, a ratio of 10:30, won 3:1.
bool ratio_before(const Doc& d, size_t i);
bool zone_word(std::string_view w);     // UTC, CET, PST, ... (as written)
bool time_cue(std::string_view w);      // at, by, until, from, ...
bool duration_cue(std::string_view w);  // took, lasted, runs, ...

}  // namespace dt
