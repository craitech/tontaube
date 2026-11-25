#pragma once
// Shared helpers of the module "numbers_context": number spans, context words, fraction and digit readings.
#include <cstdint>
#include <string>
#include <string_view>

#include "core/rule.h"

namespace nctx {

using rv::Doc;
using rv::Kind;

// ---------------------------------------------------------------------------------------------------------
// Numbers

// A written number starting at token i: "42", "1,234", "3.14", and ".5" when token i follows a lone joined '.'.
struct Num {
    bool ok = false;
    size_t first = 0, last = 0;  // tokens (first is the '.' of ".5")
    std::string whole, frac;     // digits; whole is empty for ".5"
    bool grouped = false;
    bool is_int() const { return ok && frac.empty() && !whole.empty(); }
    // The whole part as an integer; 0 if it does not fit.
    uint64_t value() const {
        if (whole.empty() || whole.size() > 18) return 0;
        uint64_t v = 0;
        for (char c : whole) v = v * 10 + static_cast<uint64_t>(c - '0');
        return v;
    }
    size_t ndigits() const { return whole.size(); }
};

// The last token of the number starting at token i, as parse_number reads it, without building strings.
inline size_t num_end(const Doc& d, size_t i) {
    size_t j = i;
    if (d.str(i).size() <= 3)
        while (d.punct(j + 1, ',') && d.joined(j + 1) && d.number(j + 2) && d.joined(j + 2) && d.str(j + 2).size() == 3)
            j += 2;
    if (d.punct(j + 1, '.') && d.joined(j + 1) && d.number(j + 2) && d.joined(j + 2)) j += 2;
    return j;
}

inline Num num_at(const Doc& d, size_t i) {
    Num n;
    if (!d.number(i)) return n;
    rv::NumberSpan s = rv::parse_number(d, i);
    n.ok = true;
    n.first = s.first;
    n.last = s.last;
    n.whole = s.whole;
    n.frac = s.fraction;
    n.grouped = s.grouped;
    // ".5": a '.' joined before, with no digit joined before the '.'
    if (!n.grouped && n.frac.empty() && i >= 1 && d.punct(i - 1, '.') && d.joined(i) &&
        !(i >= 2 && d.joined(i - 1) && d[i - 2].kind != Kind::Punct)) {
        n.first = i - 1;
        n.frac = n.whole;
        n.whole.clear();
    }
    return n;
}

// The usual reading: "1,234.5" -> one thousand two hundred thirty-four point five; ".5" -> point five.
// "" for a whole part with a leading zero ("007").
inline std::string num_words(const Num& n) {
    if (!n.ok) return "";
    if (n.whole.empty()) return n.frac.empty() ? "" : "point " + rv::digits(n.frac);
    if (n.whole.size() > 1 && n.whole[0] == '0') return "";
    std::string w = rv::cardinal_digits(n.whole);
    if (w.empty()) return "";
    if (!n.frac.empty()) w += " point " + rv::digits(n.frac);
    return w;
}

// 100..9999 read in pairs, as for rooms and ports: 115 one fifteen, 305 three oh five, 2006 twenty oh six,
// 1900 nineteen hundred, 3000 three thousand.
inline std::string pair_words(uint64_t n) {
    if (n >= 2001 && n <= 2009) return "twenty oh " + rv::cardinal(n % 10);
    return rv::pairs(static_cast<uint32_t>(n));
}

// A dimension or resolution part: four digits in pairs (1920 nineteen twenty), others as a number.
inline std::string dim_words(const Num& n, bool partner_has_four_digits) {
    if (n.is_int() && !n.grouped && n.whole[0] != '0') {
        if (n.ndigits() == 4) return pair_words(n.value());
        if (n.ndigits() == 3 && partner_has_four_digits && n.whole.substr(1) != "00") return pair_words(n.value());
    }
    return num_words(n);
}

// ---------------------------------------------------------------------------------------------------------
// Tokens around a number

// Nothing that makes the number part of a larger token is joined before token i (a letter, digit, '.', ',',
// '/', '-', ':', a currency sign ...). Opening brackets, quotes and a few signs are fine.
inline bool free_left(const Doc& d, size_t i) {
    if (!d.joined(i)) return true;
    const rv::Token& p = d[i - 1];
    if (p.kind != Kind::Punct) return false;
    switch (p.cp) {
        case '(': case '[': case '{': case '"': case '\'': case '*': case '~': case '<': case '>':
        case 0x201C: case 0x2018: case 0xAB: case 0x2248: case 0x2264: case 0x2265: case 0x2014:
            return true;
        default:
            return false;
    }
}

// Nothing that makes the number part of a larger token is joined after token last. Closing punctuation is fine;
// a '.' or ',' or ':' only when no digit follows it.
inline bool free_right(const Doc& d, size_t last) {
    size_t j = last + 1;
    if (!d.joined(j)) return true;
    const rv::Token& t = d[j];
    if (t.kind != Kind::Punct) return false;
    switch (t.cp) {
        case ',': case '.': case ':':
            return !(d.number(j + 1) && d.joined(j + 1));
        case ';': case ')': case ']': case '}': case '"': case '\'': case '!': case '?': case '*': case '<':
        case 0x201D: case 0x2019: case 0xBB: case 0x2026:
            return true;
        default:
            return false;
    }
}

inline bool is_sentence_stop(const Doc& d, size_t j) {
    if (d[j].kind != Kind::Punct) return false;
    uint32_t c = d[j].cp;
    if (c == '!' || c == '?' || c == ';' || c == 0x2026) return true;
    if (c == '.' && j + 1 < d.size() && d[j + 1].space_before && d.word(j + 1) &&
        (d[j + 1].casing & (rv::TITLE | rv::UPPER)))
        return true;
    return false;
}

// The nth Word/Number token before token i in the same sentence (punctuation skipped); npos if none.
inline size_t prev_w(const Doc& d, size_t i, int n = 1) {
    for (size_t j = i; j-- > 0;) {
        if (d[j + 1].newline_before) return Doc::npos;
        if (d[j].kind == Kind::Punct) {
            if (is_sentence_stop(d, j)) return Doc::npos;
            continue;
        }
        if (--n == 0) return j;
    }
    return Doc::npos;
}

// The nth Word/Number token after token i in the same sentence; npos if none.
inline size_t next_w(const Doc& d, size_t i, int n = 1) {
    for (size_t j = i + 1; j < d.size(); ++j) {
        if (d[j].newline_before) return Doc::npos;
        if (d[j].kind == Kind::Punct) {
            if (is_sentence_stop(d, j)) return Doc::npos;
            continue;
        }
        if (--n == 0) return j;
    }
    return Doc::npos;
}

inline std::string_view low_or_empty(const Doc& d, size_t j) {
    return j == Doc::npos ? std::string_view() : d.low(j);
}

// A quotation that the mark at token j would close is open on its line: scanning back, an opening quote (space
// before it, a word or digit joined after it) that no closing quote (joined to what precedes it) matches.
inline bool quote_open_before(const Doc& d, size_t j, uint32_t mark) {
    auto is_quote = [&](uint32_t c) {
        return mark == '"' ? (c == '"' || c == 0x201C || c == 0x201D) : (c == '\'' || c == 0x2018 || c == 0x2019);
    };
    int closed = 0;
    for (size_t k = j; k-- > 0;) {
        if (d[k].kind == Kind::Punct && is_quote(d[k].cp)) {
            bool opening = (d[k].space_before || k == 0 || d[k].cp == 0x201C || d[k].cp == 0x2018) &&
                           d.joined(k + 1) && d[k].cp != 0x201D && d[k].cp != 0x2019;
            if (opening) {
                if (closed == 0) return true;
                --closed;
            } else if (k > 0 && d.joined(k)) {
                ++closed;
            }
        }
        if (d[k].newline_before) break;
    }
    return false;
}

// A foot or inch mark joined after token last: ′ ″ always, ' and " when they do not close a quotation and no
// letter or digit is joined after them. Returns 'f' (feet), 'i' (inches) or 0.
inline char length_mark(const Doc& d, size_t last) {
    size_t j = last + 1;
    if (j >= d.size() || !d.joined(j) || d[j].kind != Kind::Punct) return 0;
    uint32_t c = d[j].cp;
    if (c == 0x2032) return 'f';
    if (c == 0x2033) return 'i';
    if (c != '"' && c != '\'') return 0;
    if (d.joined(j + 1) && (d.word(j + 1) || d.number(j + 1))) return 0;
    if (quote_open_before(d, j, c)) return 0;
    return c == '"' ? 'i' : 'f';
}

// The measure ending before token j modifies the noun at j ("a 2 1/2" hole", "4' x 8' sheets"): its unit is
// singular. Function words and measuring adjectives ("3 1/2" thick") keep the plural.
inline bool adjectival_after(const Doc& d, size_t j) {
    static const rv::WordSet NOT_NOUN{
        "a", "above", "across", "after", "along", "an", "and", "apart", "are", "around", "as", "at", "away",
        "before", "below", "beside", "between", "but", "by", "deep", "down", "each", "for", "from", "high", "in",
        "into", "is", "long", "maximum", "minimum", "more", "of", "off", "on", "or", "over", "per", "short", "so",
        "tall", "than", "that", "the", "thick", "to", "total", "under", "up", "was", "were", "which", "wide",
        "will", "with", "x"};
    return j < d.size() && d.word(j) && d[j].space_before && !d[j].newline_before && !NOT_NOUN.contains(d.low(j));
}

// One of the k words before token i (same sentence) is in the set.
inline bool any_prev(const Doc& d, size_t i, int k, const rv::WordSet& set) {
    for (int n = 1; n <= k; ++n) {
        size_t j = prev_w(d, i, n);
        if (j == Doc::npos) return false;
        if (set.contains(d.low(j))) return true;
    }
    return false;
}

// One of the words within `window` tokens around [first, last] in the same sentence is in the set.
inline bool sentence_has(const Doc& d, size_t first, size_t last, const rv::WordSet& set, int window = 24) {
    int seen = 0;
    for (size_t j = first; j-- > 0 && seen < window; ++seen) {
        if (d[j + 1].newline_before || is_sentence_stop(d, j)) break;
        if (d.word(j) && set.contains(d.low(j))) return true;
    }
    seen = 0;
    for (size_t j = last + 1; j < d.size() && seen < window; ++j, ++seen) {
        if (d[j].newline_before || is_sentence_stop(d, j)) break;
        if (d.word(j) && set.contains(d.low(j))) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------
// Readings

// 1/2 one half, 3/4 three quarters, 2/3 two thirds, 5/8 five eighths; singular: three quarter (before a unit).
inline std::string fraction_words(uint64_t n, uint64_t den, bool singular = false) {
    std::string num = rv::cardinal(n), unit;
    bool plural = n != 1 && !singular;
    if (den == 2) unit = plural ? "halves" : "half";
    else if (den == 4) unit = plural ? "quarters" : "quarter";
    else unit = rv::ordinal(den) + (plural ? "s" : "");
    if (unit.compare(0, 4, "one ") == 0) unit.erase(0, 4);  // 1/1000: one thousandth, not one one thousandth
    return num + " " + unit;
}

// A fraction as the first part of a hyphenated adjective: three-quarter (inch), five-eighth.
inline std::string adjective_fraction(uint64_t n, uint64_t den) {
    return rv::cardinal(n) + "-" + (den == 2 ? std::string("half") : den == 4 ? std::string("quarter") : rv::ordinal(den));
}

// The part after "and" in a mixed number: a half, a quarter, an eighth, three quarters.
inline std::string mixed_tail(uint64_t n, uint64_t den) {
    if (n != 1) return fraction_words(n, den);
    std::string unit = den == 2 ? "half" : den == 4 ? "quarter" : rv::ordinal(den);
    return (unit[0] == 'e' ? "an " : "a ") + unit;
}

// Digits one by one: "4502917" -> four five zero two nine one seven.
inline std::string digit_words(std::string_view s) { return rv::digits(s); }

inline bool fraction_denominator(uint64_t den) {
    return den == 2 || den == 3 || den == 4 || den == 8 || den == 16 || den == 32 || den == 64;
}

// ---------------------------------------------------------------------------------------------------------
// Word tables shared by several rules

// Measures and nouns that a fraction measures: "1/2 cup", "3/4 mile", "1/4 of".
inline const rv::WordSet FRACTION_NEXT{
    "acre", "acres", "bag", "bar", "batch", "bottle", "bowl", "box", "bushel", "can", "carat", "cent",
    "century", "clove", "cm", "cup", "cups", "day", "days", "decade", "dollar", "dose", "dozen", "drop",
    "fathom", "feet", "foot", "ft", "g", "gallon", "gallons", "gram", "grams", "hour", "hours", "hp", "inch",
    "inches", "jar", "kg", "kilo", "kilogram", "kilometer", "km", "kwh", "lb", "lbs", "length", "liter",
    "liters", "litre", "litres", "loaf", "majority", "mile", "miles", "minute", "minutes", "ml", "mm",
    "month", "months", "oz", "ounce", "ounces", "pack", "page", "percent", "piece", "pinch", "pint", "pints",
    "portion", "pound", "pounds", "quart", "quarts", "ream", "scale", "scoop", "second", "seconds", "sheet",
    "size", "slice", "spoon", "stick", "tablespoon", "tablespoons", "tablet", "tbsp", "teaspoon",
    "teaspoons", "ton", "tons", "tonne", "tsp", "turn", "wavelength", "week", "weeks", "yard", "yards",
    "year", "years", "inch-thick", "lemon", "lime", "banana", "onion", "apple", "egg", "pie", "cake",
    "sandwich", "pizza", "point"};

// Unit abbreviations after a fraction, with their singular reading: "1/2 tsp" one half teaspoon. "in" needs a
// period or a measuring word after it (see fraction_unit).
inline const rv::PhraseMap FRACTION_UNITS{
    {"tsp", "teaspoon"}, {"tsps", "teaspoon"}, {"tbsp", "tablespoon"}, {"tbsps", "tablespoon"}, {"tbs", "tablespoon"},
    {"tbl", "tablespoon"}, {"oz", "ounce"}, {"lb", "pound"}, {"lbs", "pound"}, {"in", "inch"}, {"ft", "foot"},
    {"mi", "mile"}, {"yd", "yard"}, {"yds", "yard"}, {"hr", "hour"}, {"hrs", "hour"}, {"min", "minute"},
    {"mins", "minute"}, {"sec", "second"}, {"secs", "second"}, {"kg", "kilogram"}, {"g", "gram"},
    {"mg", "milligram"}, {"ml", "milliliter"}, {"l", "liter"}, {"gal", "gallon"}, {"qt", "quart"}, {"pt", "pint"},
    {"cm", "centimeter"}, {"mm", "millimeter"}, {"km", "kilometer"}, {"c", "cup"}};

// Words after "in" that make it the inch: "3/4 in thick", "1/2 in diameter".
inline const rv::WordSet INCH_AFTER{"thick", "wide", "long", "deep", "tall", "high", "diameter", "dia", "pipe",
                                    "plywood", "drill", "bit", "bolt", "screw", "screws", "board", "gap", "hole",
                                    "thickness", "width", "length", "depth", "height", "square", "round", "x"};

// The unit abbreviation after a fraction ending at token last: its singular word, and in unit_last the unit's
// last token (with its period), or nullptr.
inline const std::string_view* fraction_unit(const Doc& d, size_t last, size_t& unit_last) {
    size_t u = last + 1;
    if (!d.word(u) || (d[u].casing & rv::MIXED)) return nullptr;
    const std::string_view* w = FRACTION_UNITS.find(d.low(u));
    if (!w) return nullptr;
    if (d.joined(u + 1) && (d.word(u + 1) || d.number(u + 1))) return nullptr;
    unit_last = u;
    bool dot = d.punct(u + 1, '.') && d.joined(u + 1);
    if (d.low(u) == "c" && !dot && !(d.word(u + 1) && (d[u + 1].casing & rv::LOWER))) return nullptr;  // "1/2 c. sugar"
    if (d.low(u) == "in") {
        bool measure = !dot && d.word(u + 1) && INCH_AFTER.contains(d.low(u + 1));
        bool end = !dot && (u + 1 >= d.size() || d.punct(u + 1, ',') || d.punct(u + 1, ')') || d.punct(u + 1, ';'));
        if (!dot && !measure && !end) return nullptr;
    }
    // "1/2 in. wide": the period belongs to the abbreviation unless it ends the sentence
    if (dot && !(u + 2 >= d.size() || d[u + 2].newline_before ||
                 (d.word(u + 2) && (d[u + 2].casing & (rv::TITLE | rv::UPPER)))))
        unit_last = u + 1;
    return w;
}

// Function words after a fraction: "1/2 of", "1/2 a cup", "1/2 as much". Weaker than a score cue.
inline const rv::WordSet FRACTION_WEAK_NEXT{"of", "a", "an", "as", "the"};

// Measures after which a fraction behind "a" is an adjective in the singular: "a 3/4 pound bag".
inline const rv::WordSet SINGULAR_UNITS{
    "acre", "cup", "day", "foot", "gallon", "hour", "inch", "liter", "litre", "mile", "minute", "ounce",
    "pint", "pound", "quart", "second", "tablespoon", "teaspoon", "ton", "week", "yard", "year"};

// Words that make a nearby number a date reading (datetime owns the date itself).
inline const rv::WordSet DATE_PREV{
    "on", "by", "due", "until", "till", "til", "since", "from", "through", "thru", "after", "before", "dated",
    "starting", "starts", "start", "ends", "ending", "effective", "expires", "expiring", "deadline",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mon", "tue", "tues", "wed",
    "thu", "thur", "thurs", "fri", "sat", "sun", "opens", "closes", "launch", "launches", "ships", "arrives",
    "arriving", "delivered", "delivery", "eta", "posted", "updated", "logged", "born", "died", "between"};

// IDs read digit by digit from three digits on.
inline const rv::WordSet ID_ALWAYS{"ean", "ein", "iban", "imei", "isbn", "issn", "npi", "otp", "passcode", "password",
                            "pin", "pmcid", "pmid", "postcode", "ref", "routing", "serial", "sku", "sn", "ssn", "upc", "vin",
                            "zip", "zipcode"};
// IDs read digit by digit from four digits on.
inline const rv::WordSet ID_FOUR{"accession", "account", "acct", "application", "badge", "booking", "card", "certificate",
                          "claim", "client", "code", "codes", "confirmation", "coupon", "customer", "employee", "id",
                          "ids", "identifier", "license", "licence", "login", "member", "membership",
                          "mls", "order", "parcel", "patient", "permit", "po", "policy", "receipt", "ref",
                          "reference", "registration", "reservation", "shipment", "shipping", "student", "tail", "tracking",
                          "transaction", "user", "voucher", "memo", "security", "access", "verification", "auth"};
// IDs read digit by digit from five digits on (shorter ones are counts or model numbers).
inline const rv::WordSet ID_FIVE{"catalog", "catalogue", "model", "part", "product", "ticket", "lot", "batch", "codex", "invoice",
                          "unit", "container", "package", "tag"};
// Words between the cue and the number: "account number is 123", "ID: 42", "Serial No. 7"
inline const rv::WordSet ID_FILLER{"number", "no", "num", "nr", "is", "was", "nos", "which", "that", "like"};


// The cue before token i: skips "number", "no.", "is", "#", ":" and similar; returns the cue's token or npos.
inline size_t id_cue(const Doc& d, size_t i, int& min_digits) {
    size_t j = i;
    bool copula = false;
    for (int steps = 0; steps < 4; ++steps) {
        j = prev_w(d, j);
        if (j == Doc::npos || !d.word(j)) return Doc::npos;
        std::string_view w = d.low(j);
        if (w == "order" && d.is(prev_w(d, j) == Doc::npos ? d.size() : prev_w(d, j), "in")) return Doc::npos;  // in order
        if (ID_ALWAYS.contains(w)) return min_digits = 3, j;
        if (ID_FOUR.contains(w)) return min_digits = 4, j;
        if (ID_FIVE.contains(w)) return min_digits = 5, j;
        copula |= w == "is" || w == "was";
        if (ID_FILLER.contains(w)) continue;
        // "The access code for the main door is 87291": a cue before a short phrase with for / to / of
        if (!copula) return Doc::npos;
        for (int k = 0; k < 4; ++k) {
            j = prev_w(d, j);
            if (j == Doc::npos || !d.word(j)) return Doc::npos;
            std::string_view p = d.low(j);
            if (p == "for" || p == "to" || p == "of" || p == "on") {
                size_t c = prev_w(d, j);
                if (c == Doc::npos) return Doc::npos;
                std::string_view cw = d.low(c);
                if (ID_ALWAYS.contains(cw)) return min_digits = 3, c;
                if (ID_FOUR.contains(cw)) return min_digits = 4, c;
                return Doc::npos;
            }
        }
        return Doc::npos;
    }
    return Doc::npos;
}

}  // namespace nctx
