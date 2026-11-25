#include "rules/datetime/dt.h"

namespace dt {

using rv::Kind;

const char* const MONTH_NAMES[13] = {"",     "January", "February", "March",     "April",   "May",     "June",
                                     "July", "August",  "September", "October", "November", "December"};

namespace {

const rv::PhraseMap FULL_MONTHS{{"january", "1"}, {"february", "2"}, {"march", "3"},     {"april", "4"},
                                {"may", "5"},     {"june", "6"},     {"july", "7"},      {"august", "8"},
                                {"september", "9"}, {"october", "10"}, {"november", "11"}, {"december", "12"}};
const rv::PhraseMap ABBR_MONTHS{{"jan", "1"}, {"feb", "2"}, {"mar", "3"},  {"apr", "4"},  {"jun", "6"},
                                {"jul", "7"}, {"aug", "8"}, {"sep", "9"},  {"sept", "9"}, {"oct", "10"},
                                {"nov", "11"}, {"dec", "12"}};
const rv::PhraseMap WEEKDAY_ABBR{{"mon", "Monday"},   {"tue", "Tuesday"},  {"tues", "Tuesday"},
                                 {"wed", "Wednesday"}, {"thu", "Thursday"}, {"thur", "Thursday"},
                                 {"thurs", "Thursday"}, {"fri", "Friday"},  {"sat", "Saturday"},
                                 {"sun", "Sunday"}};
const rv::WordSet WEEKDAYS{"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"};

const rv::WordSet TIME_CUES{"at",      "by",      "until",   "till",    "til",      "from",     "to",
                            "between", "before",  "after",   "around",  "about",    "approximately",
                            "since",   "for",     "starting", "starts", "start",    "begins",   "begin",
                            "began",   "opens",   "open",    "opened",  "closes",   "close",    "closed",
                            "ends",    "ended",   "arrive",  "arrives", "arrived",  "arriving", "depart",
                            "departs", "departed", "departing", "due",  "alarm",    "through",  "thru",
                            "every",   "daily",   "nightly", "noon",    "or",       "and"};
const rv::WordSet DURATION_CUES{"took",    "take",     "takes",   "taking",  "lasted",  "last",   "lasts",
                                "lasting", "runs",     "ran",     "run",     "running", "runtime", "length",
                                "duration", "elapsed", "clocked", "timed",   "in",      "spent",  "uptime",
                                "averaged", "average", "averaging", "averages", "totaled", "totalled", "totals", "plays",
                                "played",  "within",   "under",   "needs",   "needed",  "requires", "required",
                                "wait",    "waits",    "waited",  "waiting"};

const rv::WordSet ZONES{"UTC", "GMT", "CET", "CEST", "EET", "EEST", "WET", "WEST", "BST", "IST", "MSK", "EST",
                        "EDT", "CST", "CDT", "MST", "MDT", "PST", "PDT", "AKST", "AKDT", "HST", "AEST", "AEDT",
                        "ACST", "AWST", "NZST", "NZDT", "JST", "KST", "HKT", "SGT", "SAST", "WAT", "EAT",
                        "ET", "CT", "MT", "PT", "Z"};

int lookup(const rv::PhraseMap& m, std::string_view w) {
    const std::string_view* v = m.find(w);
    if (!v) return 0;
    int n = 0;
    for (char c : *v) n = n * 10 + (c - '0');
    return n;
}

bool capitalized(const Doc& d, size_t i) { return d[i].casing == rv::TITLE || d[i].casing == rv::UPPER; }

bool quote_or_bracket(uint32_t cp) {
    switch (cp) {
        case '"': case '\'': case '(': case ')': case '[': case ']': case '*': case '_':
        case 0x201C: case 0x201D: case 0x2018: case 0x2019: case 0xAB: case 0xBB:
            return true;
        default:
            return false;
    }
}

}  // namespace

size_t range_dash(const Doc& d, size_t i) {
    if (!dash(d, i) || i + 1 >= d.size()) return NPOS;
    bool before = d[i].space_before, after = d[i + 1].space_before;
    if (before != after) return NPOS;
    if (d[i].newline_before || d[i + 1].newline_before) return NPOS;
    return i + 1;
}

bool free_before(const Doc& d, size_t i) {
    if (!d.joined(i)) return true;
    const rv::Token& p = d[i - 1];
    if (p.kind != Kind::Punct) return false;
    switch (p.cp) {
        case '.': case ',': case '/': case ':': case '_': case '\\': case '@': case '#': case '$': case '+':
        case '=': case '-': case 0x2013: case 0x2212: case 0x20AC: case 0xA3: case '~': case '^':
            return false;
        default:
            return true;
    }
}

bool free_after(const Doc& d, size_t i) {
    size_t n = i + 1;
    if (n >= d.size() || !d.joined(n)) return true;
    const rv::Token& t = d[n];
    if (t.kind != Kind::Punct) return false;
    bool glued_on = n + 1 < d.size() && d.joined(n + 1) && d[n + 1].kind != Kind::Punct;
    switch (t.cp) {
        case '/': case '_': case '\\': case '@': case '%': case '+': case '^': case 0xB0:
            return false;
        case '.':
            // "2026-01-19.The review": a sentence end with the space missing
            return !glued_on || (d[n + 1].kind == Kind::Word && d[n + 1].casing == rv::TITLE && len(d, n + 1) >= 2);
        case ',': case ':': case '-': case 0x2013:
            return !glued_on;
        default:
            return true;
    }
}

size_t word_before_index(const Doc& d, size_t i) {
    size_t j = i;
    while (j > 0 && d[j - 1].kind == Kind::Punct && quote_or_bracket(d[j - 1].cp)) --j;
    if (j > 0 && (d.punct(j - 1, ',') || d.punct(j - 1, '.'))) --j;
    if (j > 0 && d[j - 1].kind == Kind::Word) return j - 1;
    return NPOS;
}

std::string_view word_before(const Doc& d, size_t i) {
    size_t j = word_before_index(d, i);
    return j == NPOS ? std::string_view() : d.low(j);
}

bool period_ends_sentence(const Doc& d, size_t i) {
    size_t n = i + 1;
    if (n >= d.size()) return true;
    if (d[n].newline_before) return true;
    if (!d[n].space_before) return false;
    while (n < d.size() && d[n].kind == Kind::Punct && quote_or_bracket(d[n].cp) && d[n].cp != ')' &&
           d[n].cp != ']')
        ++n;
    if (n >= d.size()) return true;
    if (d[n].kind != Kind::Word || !capitalized(d, n)) return false;
    if (month_at(d, n) || weekday_abbrev_at(d, n) || weekday_full_at(d, n)) return false;
    std::string_view w = d.low(n);
    if (w == "am" || w == "pm" || zone_word(d.str(n))) return false;
    return true;
}

size_t abbrev_end(const Doc& d, size_t i) {
    if (d.punct(i + 1, '.') && d.joined(i + 1) && !period_ends_sentence(d, i + 1)) return i + 1;
    return i;
}

namespace {

// The first two letters of a month or weekday name: a cheap test before the table lookups.
bool month_prefix(std::string_view w) {
    switch (w[0]) {
        case 'j': return w[1] == 'a' || w[1] == 'u';
        case 'f': return w[1] == 'e';
        case 'm': return w[1] == 'a';
        case 'a': return w[1] == 'p' || w[1] == 'u';
        case 's': return w[1] == 'e';
        case 'o': return w[1] == 'c';
        case 'n': return w[1] == 'o';
        case 'd': return w[1] == 'e';
        default: return false;
    }
}
bool weekday_prefix(std::string_view w) {
    switch (w[0]) {
        case 'm': return w[1] == 'o';
        case 't': return w[1] == 'u' || w[1] == 'h';
        case 'w': return w[1] == 'e';
        case 'f': return w[1] == 'r';
        case 's': return w[1] == 'a' || w[1] == 'u';
        default: return false;
    }
}

}  // namespace

int month_at(const Doc& d, size_t i, bool* abbr) {
    if (!d.word(i)) return 0;
    std::string_view w = d.low(i);
    if (w.size() < 3 || w.size() > 9 || !month_prefix(w)) return 0;
    int m = lookup(FULL_MONTHS, w);
    if (m) {
        // lower case is the modal "may", the verb "march", or another language ("12 september 2026")
        if (!capitalized(d, i)) return 0;
        if (abbr) *abbr = false;
        return m;
    }
    m = lookup(ABBR_MONTHS, w);
    if (m && capitalized(d, i)) {
        if (abbr) *abbr = true;
        return m;
    }
    return 0;
}

int month_lower_at(const Doc& d, size_t i) {
    if (!d.word(i) || d[i].casing != rv::LOWER || len(d, i) < 4 || len(d, i) > 9) return 0;
    std::string_view w = d.low(i);
    if (w == "march" || !month_prefix(w)) return 0;
    return lookup(FULL_MONTHS, w);
}

const char* weekday_abbrev_at(const Doc& d, size_t i) {
    if (!d.word(i) || !capitalized(d, i) || len(d, i) < 3 || len(d, i) > 5 || !weekday_prefix(d.low(i))) return nullptr;
    const std::string_view* v = WEEKDAY_ABBR.find(d.low(i));
    return v ? v->data() : nullptr;
}

bool weekday_full_at(const Doc& d, size_t i) {
    if (!d.word(i) || len(d, i) < 6 || len(d, i) > 9) return false;
    std::string_view w = d.low(i);
    return w.substr(w.size() - 3) == "day" && WEEKDAYS.contains(w);
}

bool valid_day(int month, int day) {
    static const int DAYS[13] = {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return month >= 1 && month <= 12 && day >= 1 && day <= DAYS[month];
}

std::string year_words(int y) { return y >= 1000 ? rv::pairs(static_cast<uint32_t>(y)) : rv::cardinal(y); }

std::string year_pairs(int y) { return y >= 100 ? rv::pairs(static_cast<uint32_t>(y)) : rv::cardinal(y); }

std::string two_digit_year(int yy) {
    if (yy <= 0 || yy > 99) return "";
    return yy < 10 ? "oh " + rv::cardinal(yy) : rv::cardinal(yy);
}

std::string decade_words(int y) {
    std::string w = year_words(y);
    if (!w.empty() && w.back() == 'y') return w.substr(0, w.size() - 1) + "ies";
    return w + "s";
}

std::string day_words(int d) { return rv::ordinal(static_cast<uint64_t>(d)); }

namespace {
std::string minute_words(int m) { return m < 10 ? "oh " + rv::cardinal(m) : rv::cardinal(m); }
}  // namespace

std::string clock_words(int h, int m) {
    if (m == 0) {
        if (h == 0) return "zero hundred";
        if (h <= 12) return rv::cardinal(h) + " o'clock";
        return rv::cardinal(h) + " hundred";
    }
    return rv::cardinal(h) + " " + minute_words(m);
}

std::string clock_short(int h, int m) {
    if (m == 0 && h >= 1 && h <= 12) return rv::cardinal(h);
    return clock_words(h, m);
}

std::string hm_words(int h, int m) {
    return m == 0 ? rv::cardinal(h) : rv::cardinal(h) + " " + minute_words(m);
}

std::string plural(int n, const char* unit) { return n == 1 ? std::string(unit) : std::string(unit) + "s"; }

size_t ordinal_suffix(const Doc& d, size_t i) {
    size_t n = i + 1;
    if (!d.word(n) || !d.joined(n)) return NPOS;
    std::string_view w = d.low(n);
    if (w == "st" || w == "nd" || w == "rd" || w == "th") return n;
    return NPOS;
}

const char* ampm_at(const Doc& d, size_t i, size_t* last) {
    if (!d.word(i)) return nullptr;
    std::string_view w = d.low(i);
    if (w == "am" || w == "pm") {
        if (d[i].casing == rv::MIXED) return nullptr;
        *last = i;
        return w == "am" ? "AM" : "PM";
    }
    if ((w == "a" || w == "p") && d.punct(i + 1, '.') && d.joined(i + 1) && d.is(i + 2, "m") && d.joined(i + 2)) {
        *last = i + 2;
        // the period stays where it ends a sentence, but not before a line break inside a list ("2:00 p.m.\nDec. 25")
        if (d.punct(i + 3, '.') && d.joined(i + 3) &&
            (!period_ends_sentence(d, i + 3) || (i + 4 < d.size() && d[i + 4].newline_before)))
            *last = i + 3;
        if (d.joined(*last + 1) && !punct_any(d, *last + 1)) return nullptr;
        return w == "a" ? "AM" : "PM";
    }
    return nullptr;
}

Clock parse_clock(const Doc& d, size_t i, int max_h) {
    Clock c;
    int h = num(d, i, 2);
    if (h < 0 || h > max_h) return c;
    if (!d.punct(i + 1, ':') || !d.joined(i + 1) || !d.joined(i + 2)) return c;
    int m = num(d, i + 2, 2);
    if (m < 0 || m > 59 || len(d, i + 2) != 2) return c;
    c.first = i;
    c.last = i + 2;
    c.h = h;
    c.m = m;
    if (d.punct(c.last + 1, ':') && d.joined(c.last + 1) && d.joined(c.last + 2)) {
        int s = num(d, c.last + 2, 2);
        if (s < 0 || s > 59 || len(d, c.last + 2) != 2) return c;
        c.s = s;
        c.last += 2;
        if (d.punct(c.last + 1, '.') && d.joined(c.last + 1) && d.number(c.last + 2) && d.joined(c.last + 2))
            c.last += 2;
    }
    if (d.word(c.last + 1) && d.joined(c.last + 1) && d.str(c.last + 1) == "Z") {
        c.zulu = true;
        c.last += 1;
    }
    if (d.punct(c.last + 1, ':') && d.joined(c.last + 1) && d.number(c.last + 2) && d.joined(c.last + 2)) return c;
    if (!c.zulu && d.joined(c.last + 1) && d.word(c.last + 1)) {
        // "10:30am" is handled by the caller through ampm_at; other letters glued on are not a time
        size_t dummy;
        if (!ampm_at(d, c.last + 1, &dummy)) return c;
    }
    c.ok = true;
    return c;
}

const rv::WordSet RATIO_NOUNS{"format",  "formats", "aspect",  "ratio",   "ratios", "screen",  "screens", "display",
                              "displays", "layout", "layouts", "frame",  "frames", "crop",    "monitor", "monitors",
                              "panel",   "panels",  "mix",     "blend",   "split",  "odds",    "majority", "lead",
                              "win",     "victory", "defeat",  "loss",    "scoreline", "draw", "ratio"};

bool ratio_after(const Doc& d, size_t last) {
    size_t n = last + 1;
    return d.word(n) && d[n].space_before && !d[n].newline_before && RATIO_NOUNS.contains(d.low(n));
}

const rv::WordSet RATIO_CUES{"score", "scores", "ratio", "ratios", "odds", "won", "win", "wins", "lost", "lose",
                             "loses", "beat", "beats", "led", "leads", "trailed", "trails", "tied", "defeated",
                             "edged", "scoreline", "proportion", "aspect", "drew", "victory"};

bool ratio_before(const Doc& d, size_t i) {
    size_t w = word_before_index(d, i);
    if (w == NPOS || !d[i].space_before) return false;
    std::string_view x = d.low(w);
    if ((x == "was" || x == "is" || x == "of" || x == "were") && w > 0) {  // the score was 21:14, a ratio of 10:30
        size_t w2 = word_before_index(d, w);
        if (w2 != NPOS && RATIO_CUES.contains(d.low(w2))) return true;
    }
    return RATIO_CUES.contains(x);
}

bool zone_word(std::string_view w) { return ZONES.contains(w); }
bool time_cue(std::string_view w) { return TIME_CUES.contains(w); }
bool duration_cue(std::string_view w) { return DURATION_CUES.contains(w); }

}  // namespace dt
