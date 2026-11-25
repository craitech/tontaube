// Years in pairs ("in 1962", "2010", "1066 AD"), four-digit numbers that count something ("1100 units"), year
// ranges ("1941-1945", "2019–24"), decades ("1960s", "the 70s", "'90s"), short years ("'91") and "mid-2025".
#include "core/rule.h"
#include "rules/numbers_core/num.h"

namespace {

using namespace numbers_core;
using rv::Kind;

bool apostrophe(uint32_t c) { return c == '\'' || c == 0x2019 || c == 0x2018; }

const rv::WordSet MONTHS{"january", "february", "march", "april", "may", "june", "july", "august", "september",
                         "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
                         "sep", "sept", "oct", "nov", "dec"};

// Words before a four-digit number that make it a year.
const rv::WordSet YEAR_BEFORE{
    "in", "since", "from", "until", "till", "til", "through", "thru", "during", "circa", "ca", "year", "years",
    "early", "mid", "late", "spring", "summer", "autumn", "fall", "winter", "before", "after", "around", "between",
    "born", "died", "founded", "established", "built", "dated", "vintage", "anno", "by", "season", "class",
    "census", "election", "elections", "olympics", "christmas", "easter", "fiscal", "cohort", "remembered",
    "into", "pre", "post", "beginning", "end", "start",
};
// Words before a number that name a thing by its number and are read in full ("Flight 1207", "Part 1308").
const rv::WordSet NAME_BEFORE{"flight", "part"};
// Street words after a house number ("1400 Elm St.", "1428 N. Main Street").
const rv::WordSet STREET{"st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "ln", "lane", "dr",
                         "drive", "way", "ct", "court", "pl", "place", "pkwy", "parkway", "hwy", "terrace", "ter",
                         "circle", "cir"};
const rv::WordSet DIRECTION{"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west"};

// "1400 Elm St.", "1428 N. Main St.": a house number, not a year.
bool street_after(const Doc& d, size_t e) {
    size_t k = e + 1;
    if (d.word(k) && DIRECTION.contains(d.low(k)) && d[k].casing != rv::LOWER) {
        ++k;
        if (d.punct(k, '.')) ++k;
    }
    for (int words = 0; words < 2 && d.word(k) && d[k].casing == rv::TITLE; ++words) {
        if (d.word(k + 1) && STREET.contains(d.low(k + 1))) return true;
        ++k;
    }
    return false;
}

// Words before "of" that make the number after it a year ("the class of 1999", "the summer of 1969").
const rv::WordSet YEAR_OF{"class", "summer", "spring", "autumn", "fall", "winter", "year", "end", "start",
                          "beginning", "middle", "half", "events", "election", "war", "crisis", "flood",
                          "revolution", "act", "treaty", "games", "olympics", "harvest", "vintage"};
// Words linking two years ("from 1990 to 2000", "1950 and 1960").
const rv::WordSet YEAR_LINK{"to", "and", "or", "through", "until", "till", "thru", "versus", "vs"};
// Adjectives that may stand between a number and what it counts ("1500 new users").
const rv::WordSet COUNT_ADJ{
    "new", "more", "active", "other", "additional", "extra", "unique", "registered", "paying", "daily", "monthly",
    "weekly", "individual", "different", "separate", "small", "large", "full", "distinct", "spare", "online",
    "local", "foreign", "international", "young", "eligible", "confirmed", "reported", "known", "free", "fewer",
    "raw", "labeled", "labelled", "annotated", "verified", "valid", "failed", "successful", "concurrent", "random",
    "square", "cubic", "archived", "loyalty", "test", "nightly", "subscriber", "unread", "hand", "real", "live",
};

const rv::WordSet DETERMINERS{"the", "its", "their", "our", "his", "her", "this", "that", "these", "those",
                              "your", "my"};
const rv::WordSet YEAR_PREPS{"of", "in", "for", "from", "on", "across", "over", "by", "with", "during", "through"};

bool upper_is(const Doc& d, size_t i, std::string_view w) { return i < d.size() && d.str(i) == w; }

// "AD", "BC", "BCE", "CE", "A.D.", "B.C." starting at token i.
bool era_at(const Doc& d, size_t i) {
    if (!d.word(i) || !d[i].space_before) return false;
    if (upper_is(d, i, "AD") || upper_is(d, i, "BC") || upper_is(d, i, "BCE") || upper_is(d, i, "CE")) return true;
    return (upper_is(d, i, "A") || upper_is(d, i, "B")) && d.punct(i + 1, '.') && !d[i + 1].space_before &&
           (upper_is(d, i + 2, "D") || upper_is(d, i + 2, "C")) && !d[i + 2].space_before;
}

// Units and counted nouns in a compound after a hyphen ("1500-byte frames", "a 1200-page report").
const rv::WordSet COMPOUND_COUNT{"byte", "bit", "page", "day", "hour", "minute", "second", "word", "line", "mile",
                                 "meter", "metre", "foot", "pound", "ton", "acre", "seat", "room", "bed", "person",
                                 "member", "piece", "step", "point", "year", "km", "kg", "gram", "liter", "litre",
                                 "watt", "volt", "pixel", "px", "item", "node", "core", "unit", "kilometer",
                                 "square", "sq", "character", "token", "frame", "employee", "student"};
// Adjectives after which any plural counts ("1250 new initiatives").
const rv::WordSet COUNT_ADJ_ANY{"new", "more", "additional", "extra", "other", "fewer"};
// Plural nouns that count after a year-like number below 2000 only ("1500 records"; "the 2024 records" is a year).
const rv::WordSet COUNT_BEFORE_2000{"records", "orders", "claims", "cases", "points", "runs", "calls", "steps",
                                    "tests", "events", "trials", "questions", "responses", "answers", "games"};

bool plural_like(std::string_view w) { return w.size() > 3 && w.back() == 's' && w[w.size() - 2] != 's'; }

// Plural nouns that a year names rather than counts ("2024 figures", "the regulator's 2024 rules").
const rv::WordSet YEAR_NOUNS{
    "figures", "rules", "targets", "invoices", "results", "numbers", "sales", "earnings", "returns", "revenues",
    "estimates", "forecasts", "projections", "budgets", "plans", "filings", "accounts", "reports", "statistics",
    "stats", "levels", "prices", "rates", "guidelines", "standards", "regulations", "elections", "games",
    "olympics", "models", "editions", "versions", "releases", "cards", "series", "counts", "totals", "trends",
    "taxes", "fees", "costs", "salaries", "wages", "payments", "deadlines", "goals", "priorities", "highlights",
    "winners", "nominees", "graduates", "recipients", "cohorts", "champions", "specs", "renewals", "bonuses",
    "dues", "data", "awards", "nominations", "policies", "dates", "holidays", "midterms", "primaries",
    "playoffs", "finals", "classes", "rankings", "ratings", "reviews", "summaries", "updates", "changes",
    "objectives", "outlooks", "guidance", "contracts", "imports", "exports", "harvests", "vintages", "wines",
    "collections", "launches", "rollouts", "patches", "builds", "defects", "audits", "surveys", "studies",
    "cases", "census", "issues", "events", "festivals", "tours", "seasons", "fixtures", "draws", "matches",
    "races", "lists", "allocations", "limits", "caps", "thresholds",
    // events named by their year ("remembering 1968 protests")
    "protests", "riots", "floods", "strikes", "wars", "earthquakes", "uprisings", "hurricanes", "droughts",
    "bombings", "reforms", "laws", "treaties", "accords", "sanctions", "tariffs", "crises", "recessions",
    "lockdowns", "mandates", "polls", "debates", "campaigns", "revolutions", "crashes", "famines", "outbreaks",
    "pandemics", "amendments", "revisions", "statutes", "migrations"};
// Words before a four-digit number after which it is not a count of the noun that follows ("against 2025
// invoices" is a year): prepositions and determiners.
const rv::WordSet COUNT_BLOCK_BEFORE{
    "the", "its", "their", "our", "his", "her", "this", "that", "these", "those", "your", "my", "a", "an",
    "against", "of", "for", "to", "and", "or", "with", "per", "via", "vs", "versus", "than", "as", "like",
    "is", "was", "are", "were", "be", "been", "any", "each", "every", "no", "next", "last", "same", "whose",
    "which", "what", "fy", "cy", "model", "type", "series", "version", "number", "item",
    // a model series: "the older 1440 displays", "the new 2080 cards"
    "old", "older", "new", "newer", "latest", "earlier", "later", "original", "classic", "legacy", "current",
    "previous", "updated", "upgraded", "refurbished", "used",
    // conjunctions and adverbs before a clause that starts with a year: "so 1971 becomes ..."
    "so", "then", "but", "when", "if", "because", "while", "although", "though", "where", "once", "now",
    "also", "until", "since", "after", "before", "whereas", "yet", "thus", "hence"};
// Verbs in -s that can follow a year ("1971 becomes", "2020 marks").
const rv::WordSet VERBS_S{
    "becomes", "means", "shows", "includes", "contains", "remains", "seems", "looks", "sounds", "gets", "goes",
    "comes", "makes", "takes", "gives", "says", "keeps", "needs", "marks", "starts", "ends", "runs", "brings",
    "sees", "covers", "follows", "leads", "reads", "turns", "lists", "holds", "puts", "sets", "stands", "falls",
    "begins", "appears", "applies", "matters", "counts", "works", "lasts", "feels", "promises", "offers",
    "introduces", "requires", "allows", "adds", "uses", "has", "does", "was", "is"};

// "planted 2000 trees", "numbered 1100 knights": a four-digit number between a lower-case word that is not a
// preposition or determiner and a plural noun counts it. Years between 2005 and 2035 count only after a verb in
// -ed or -ing or a quantity word ("producing 2023 units", "about 2010 entries"); otherwise they stay years.
bool plural_count(const Doc& d, size_t i, size_t e, uint32_t v) {
    size_t k = e + 1;
    if (!d.word(k) || !d[k].space_before || d[k].casing != rv::LOWER) return false;
    std::string_view w = d.low(k);
    if (!plural_like(w) || YEAR_NOUNS.contains(w) || VERBS_S.contains(w)) return false;
    if (d.word(k + 1) && !d[k + 1].space_before) return false;  // "years'"
    if (i == 0 || !d[i].space_before || !d.word(i - 1) || d[i - 1].casing != rv::LOWER) return false;
    std::string_view b = d.low(i - 1);
    if (COUNT_BLOCK_BEFORE.contains(b) || YEAR_BEFORE.contains(b) || MONTHS.contains(b) ||
        b.find('\'') != std::string_view::npos ||
        b.find("\xE2\x80\x99") != std::string_view::npos)
        return false;
    if (v >= 2005 && v <= 2035) {
        bool verb = b.size() > 3 && (b.substr(b.size() - 2) == "ed" || b.substr(b.size() - 3) == "ing");
        return verb || b == "about" || b == "over" || b == "nearly" || b == "almost" || b == "some";
    }
    return true;
}

// What follows token e counts something: a counted noun, a unit, a scale word, an adjective and a counted noun, or
// a compound ("1500-byte"). v is the number's value.
bool counted_after(const Doc& d, size_t e, uint32_t v = 0) {
    size_t k = e + 1;
    if (d.punct(k, '-') && !d[k].space_before && d.word(k + 1) && !d[k + 1].space_before)
        return COMPOUND_COUNT.contains(d.low(k + 1));
    if (!d.word(k) || !d[k].space_before) return false;
    std::string_view w = d.low(k);
    if (count_word(w) || scale_word(w) || w == "per") return true;
    if (v && v < 2000 && COUNT_BEFORE_2000.contains(w)) return true;
    if (!d.word(k + 1) || !d[k + 1].space_before) return false;
    std::string_view w2 = d.low(k + 1);
    return (COUNT_ADJ.contains(w) && count_word(w2)) || (COUNT_ADJ_ANY.contains(w) && plural_like(w2));
}

// Labels before "Users: 1240", "x = 1156": the number is a value, not a year.
const rv::WordSet VALUE_LABEL{"total", "count", "score", "sessions", "users", "tests", "items", "rows", "entries",
                              "requests", "visits", "views", "clicks", "errors", "warnings", "samples", "size",
                              "length", "width", "height", "steps", "iterations", "epochs", "tokens", "records",
                              "files", "pages", "words", "lines", "members", "participants", "subscribers"};

// A list of numbers that are not years before token i: "65, 97, 128, 1234 and 65535".
bool in_count_list(const Doc& d, size_t i) {
    int items = 0;
    for (size_t j = i; j >= 2 && items < 2; j -= 2) {
        if (!d.punct(j - 1, ',') || !d.number(j - 2) || !d[j].space_before || d[j - 1].space_before) break;
        std::string_view s = d.str(j - 2);
        if (s.size() == 4 || (j >= 3 && !d[j - 2].space_before)) break;
        ++items;
    }
    return items == 2;
}

bool value_context(const Doc& d, size_t i) {
    if (in_count_list(d, i)) return true;
    if (i == 0 || !d[i].space_before || d[i - 1].kind != Kind::Punct) return false;
    uint32_t c = d[i - 1].cp;
    if (c == '=' || c == 0x2248) return true;
    if (c != ':' || i < 2 || d[i - 1].space_before) return false;
    return d.punct(i - 2, '"') || (d.word(i - 2) && VALUE_LABEL.contains(d.low(i - 2)));  // "timeout_ms": 1500
}

// The word right before token i, separated by a space, or npos.
size_t word_before(const Doc& d, size_t i) {
    if (i == 0 || !d[i].space_before || !d.word(i - 1)) return Doc::npos;
    return i - 1;
}

// A four-digit year-like number at token i, read whole: value, or 0.
struct Year {
    uint32_t v = 0;
    size_t first = 0, last = 0;
};
Year year_at(const Doc& d, size_t i) {
    Year y;
    Num n;
    if (!d.number(i) || !scan(d, i, n) || !n.integer() || n.sep || n.whole.size() != 4 || n.whole[0] == '0')
        return y;
    y.v = value4(n.whole);
    y.first = n.first;
    y.last = n.last;
    return y;
}

// A neighbour glued to the number makes it something else: a date or time part, money, a fraction, an index.
bool glued_context(const Doc& d, const Year& y) {
    if (y.first > 0 && !d[y.first].space_before) {
        size_t p = prev_tok(d, y.first);
        if (p != Doc::npos && d[p].kind == Kind::Punct) {
            uint32_t c = d[p].cp;
            if (c == '/' || c == ':' || c == '#' || c == '$' || c == 0x20AC || c == 0xA3 || c == 0xA5 ||
                c == '-' || c == 0x2013 || c == 0x2212 || c == '+' || c == '^' || c == 0xD7 || c == '*' ||
                c == '=' || c == '@' || c == '\\' || c == '_' || c == 0xA7)
                return true;
        }
    }
    size_t k = next_tok(d, y.last);
    if (k < d.size() && glued(d, y.last, k) && d[k].kind == Kind::Punct) {
        uint32_t c = d[k].cp;
        if (c == '/' || c == '%' || c == 0xD7 || c == '^' || c == '=' || c == '_' || c == 0xB0 || c == '+')
            return true;
        // "20:25" is a time, "Taxes for 2025: $1,091" a label, "2024*3" a product, "in 2024*." a footnote
        if ((c == '-' || c == 0x2013 || c == '.' || c == ':' || c == '*') && glued(d, k, k + 1) && d.number(k + 1))
            return true;
    }
    return false;
}

// The 4-digit number at token i is a year by the words around it: 0 none, else the score of the year reading.
int year_cue(const Doc& d, size_t i, const Year& y) {
    size_t e = y.last;
    if (era_at(d, next_tok(d, e))) return 75;
    size_t w = word_before(d, i);
    if (w != Doc::npos) {
        std::string_view lw = d.low(w);
        if (upper_is(d, w, "AD") || upper_is(d, w, "CE")) return 75;
        if (YEAR_BEFORE.contains(lw) || MONTHS.contains(lw)) return 70;
        if (lw == "of" && w > 0 && d.word(w - 1) && (YEAR_OF.contains(d.low(w - 1)) || MONTHS.contains(d.low(w - 1))))
            return 70;
        if (YEAR_LINK.contains(lw)) {
            size_t p = d.prev_word(w);
            if (p != Doc::npos && d.number(p) && d.str(p).size() == 4) {
                uint32_t pv = value4(std::string(d.str(p)));
                if (pv >= 1100 && pv <= 2099) return 70;
            }
        }
    }
    // "c. 1450", "ca. 1450"
    if (i >= 2 && d[i].space_before && d.punct(i - 1, '.') && (d.is(i - 2, "c") || d.is(i - 2, "ca"))) return 70;
    return 0;
}

std::string year_words(uint32_t v) { return rv::pairs(v); }

// Years in pairs, and four-digit numbers read in full where they count something.
RV_RULE("numbers_core", years) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        Quick q = quick(doc, i);
        if (q.digits != 4 || !free_before(doc, i)) {
            i = q.end;
            continue;
        }
        Year y = year_at(doc, i);
        if (!y.v) continue;
        size_t e = y.last;
        if (!free_after(doc, e) || glued_context(doc, y)) {
            i = e;
            continue;
        }
        // "the 2019 recordings", "2025 defects were fixed": a recent year before a plural noun is attributive
        size_t wb = word_before(doc, i);
        // "the full archive of 2026 measurements", "from 2024 surveys": also after a preposition
        bool prep = wb != Doc::npos && YEAR_PREPS.contains(doc.low(wb)) && y.v >= 2001;
        bool attributive = y.v >= 2010 - (prep ? 9 : 0) && y.v <= 2035 &&
                           (doc.sentence_start(i) || (wb != Doc::npos && DETERMINERS.contains(doc.low(wb))) || prep) &&
                           !(doc.word(e + 1) && unit_word(doc.low(e + 1)));
        if (street_after(doc, e)) {  // a house number: numbers_context reads it
            i = e;
            continue;
        }
        // "from 1200 to 1500 units": the first end of a counted range counts too
        bool counted_range = doc.word(e + 1) && YEAR_LINK.contains(doc.low(e + 1)) && doc.number(e + 2) &&
                             doc.str(e + 2).size() == 4 && free_after(doc, e + 2) && counted_after(doc, e + 2);
        if ((!attributive && (counted_after(doc, e, y.v) || counted_range || plural_count(doc, i, e, y.v))) ||
            value_context(doc, i)) {
            out.tokens(i, e, whole_words(std::to_string(y.v)), score_span(doc, i, e, 40));
            i = e;
            continue;
        }
        // a word around it makes it a year; the shape alone does from 1100 to 2099
        if (wb != Doc::npos && NAME_BEFORE.contains(doc.low(wb))) {
            i = e;
            continue;
        }
        int score = year_cue(doc, i, y);
        if (score == 70 && (y.v < 1000 || y.v > 2199)) score = 0;
        if (!score && y.v >= 1100 && y.v <= 2099 && y.v != 2048) score = 30;
        if (score) out.tokens(i, e, year_words(y.v), score_span(doc, i, e, score));
        i = e;
    }
}

// "1941-1945", "2019–24", "2026–27": both ends as years, the dash as "to".
RV_RULE("numbers_core", year_ranges) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        Quick q = quick(doc, i);
        size_t dash = q.end + 1;
        if (q.digits == 3 && q.end == i && word_before(doc, i) != Doc::npos && doc.is(i - 1, "period") &&
            (doc.punct(dash, '-') || doc.punct(dash, 0x2013)) && !doc[dash].space_before && doc.number(dash + 1) &&
            !doc[dash + 1].space_before && doc.str(dash + 1).size() == 3 && free_after(doc, dash + 1)) {
            // "the period 180-284": three-digit years
            uint32_t a = value4(std::string(doc.str(i))), b = value4(std::string(doc.str(dash + 1)));
            if (a >= 100 && b > a) {
                out.tokens(i, dash + 1, year_words(a) + " to " + year_words(b), 70);
                i = dash + 1;
                continue;
            }
        }
        if (q.digits != 4 || dash >= doc.size() || doc[dash].space_before || doc[dash].kind != Kind::Punct ||
            !free_before(doc, i)) {
            i = q.end;
            continue;
        }
        Year a = year_at(doc, i);
        if (!a.v || a.v < 1000 || a.v > 2099) continue;
        if (i > 0 && !doc[i].space_before) {
            size_t p = prev_tok(doc, i);
            if (p != Doc::npos && (doc.punct(p, '/') || doc.punct(p, ':') || doc.punct(p, '-'))) continue;
        }
        size_t t = next_tok(doc, a.last);
        if (t >= doc.size() || !glued(doc, a.last, t) ||
            !(doc.punct(t, '-') || doc.punct(t, 0x2013) || doc.punct(t, 0x2014)))
            continue;
        size_t k = next_tok(doc, t);
        if (!doc.number(k) || !glued(doc, t, k)) continue;
        std::string_view s = doc.str(k);
        if (!free_after(doc, k)) continue;
        size_t after = next_tok(doc, k);
        if (after < doc.size() && glued(doc, k, after) &&
            (doc.punct(after, '-') || doc.punct(after, 0x2013) || doc.punct(after, '/') || doc.punct(after, '.')) &&
            doc.number(next_tok(doc, after)))
            continue;
        if (doc.punct(after, '%')) continue;
        if (s.size() == 4 && s[0] != '0' && counted_after(doc, k)) {  // "1200-1500 units": counts, in full
            out.tokens(i, k, whole_words(std::to_string(a.v)) + " to " + whole_words(std::string(s)), 50);
            i = k;
            continue;
        }
        uint32_t b = 0;
        if (s.size() == 4 && s[0] != '0') {
            b = value4(std::string(s));
            if (b <= a.v || b > a.v + 200) continue;
        } else if (s.size() == 2) {
            uint32_t lo = value4(std::string(s));
            if (lo <= a.v % 100) continue;
            b = a.v / 100 * 100 + lo;
        } else {
            continue;
        }
        int score = 60;
        out.tokens(i, a.last, year_words(a.v), score_span(doc, i, a.last, score));
        out.tokens(t, t, "to", score);
        out.tokens(k, k, year_words(b), score);
        i = a.last;
    }
}

bool bc_at(const Doc& d, size_t i) {
    return era_at(d, i) && (upper_is(d, i, "BC") || upper_is(d, i, "BCE") || upper_is(d, i, "B"));
}

// Words of a year before an era: four digits in pairs, fewer in full (as datetime.era reads them).
std::string era_year_words(uint32_t v) { return v >= 1000 ? year_words(v) : rv::cardinal(v); }

// The dash or "to" between two years at token t; the next year's token, or npos.
size_t year_link_at(const Doc& d, size_t t) {
    if (d.is(t, "to") && d[t].space_before && d.number(t + 1) && d[t + 1].space_before) return t + 1;
    if ((d.punct(t, '-') || d.punct(t, 0x2013) || d.punct(t, 0x2014)) && d.number(t + 1) &&
        (glued(d, t - 1, t) ? glued(d, t, t + 1) : one_space(d, t - 1, t) && one_space(d, t, t + 1)))
        return t + 1;
    return Doc::npos;
}

// "1332-1323 BC", "c. 80–51 BC": a falling range before BC; "476 AD": a lone three-digit year that is no multiple
// of ten, in pairs (labels 19:7; "450 AD" is read in full). Both are left out by datetime.era. The era itself stays
// as written.
RV_RULE("numbers_core", era_years) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        std::string_view s = doc.str(i);
        if (s.size() < 2 || s.size() > 4 || s[0] == '0') continue;
        if (s.size() == 3 && s[2] != '0' && era_at(doc, i + 1) && free_before(doc, i) && free_after(doc, i) &&
            !(i > 0 && (doc.punct(i - 1, '-') || doc.punct(i - 1, 0x2013) || doc.punct(i - 1, '.')))) {
            out.tokens(i, i, rv::pairs(value4(std::string(s))), 76);
            continue;
        }
        size_t j = year_link_at(doc, i + 1);
        if (j == Doc::npos || !bc_at(doc, j + 1)) continue;
        if (!free_before(doc, i) || !free_after(doc, i)) continue;
        size_t first = i;
        bool circa = i >= 2 && doc.punct(i - 1, '.') && !doc[i - 1].space_before && doc[i].space_before &&
                     (upper_is(doc, i - 2, "c") || upper_is(doc, i - 2, "ca"));
        if (circa) first = i - 2;
        std::string_view t = doc.str(j);
        if (t.size() < 2 || t.size() > 4 || t[0] == '0' || !free_after(doc, j)) continue;
        uint32_t a = value4(std::string(s)), b = value4(std::string(t));
        if (b < a) {  // "1332-1323 BC", "from 80 to 51 BC"
            out.tokens(first, j, (circa ? "circa " : "") + era_year_words(a) + " to " + era_year_words(b), 77);
            i = j;
        }
    }
}

// Words after a two-digit "s" form that make it seconds, not a decade ("the 90s timeout").
const rv::WordSet SECONDS_AFTER{
    "timeout", "timeouts", "delay", "window", "interval", "limit", "wait", "cooldown", "grace", "default",
    "baseline", "slot", "slots", "clip", "clips", "stress", "sample", "heartbeat", "pause", "buffer", "gap",
    "cycle", "ttl", "poll", "polling", "backoff", "retry", "budget", "cap", "threshold", "target", "video",
    "spot", "ad", "ads", "segment", "snippet", "preview", "trailer", "intro", "countdown", "break", "rest", "lap",
    "sprint", "burst", "test", "run", "warmup", "warm", "read", "product", "max", "maximum", "minimum", "timer",
    "deadline", "expiry", "period", "duration", "cutoff", "increments", "mark", "exposure", "shot", "shutter",
    "hold", "plank", "rule", "recording", "time", "loop", "refresh", "sleep", "tick", "step",
    "intervals", "delays", "breaks", "pauses", "gaps", "windows", "loops", "rests", "sprints", "bursts",
    "segments", "spots", "snippets", "previews", "trailers", "commercials", "cycles", "exposures", "polls",
    "ticks", "waits", "warmups", "laps", "chunks", "spins", "drills", "commercial", "block", "chunk", "spin",
    "drill", "warning", "alarm", "beep", "chime",
};
// Words after a two-digit decade that make it one ("80s pop culture").
const rv::WordSet DECADE_AFTER{"pop", "rock", "music", "fashion", "style", "culture", "movies", "films", "tv",
                               "era", "nostalgia", "kids", "hits", "songs", "vibe", "vibes", "aesthetic", "hair",
                               "disco", "punk", "hip", "decor", "design", "cartoons", "sitcoms", "classics",
                               "revival", "trends", "icons", "bands", "synth", "neon", "retro", "posters",
                               "records", "albums", "cars", "clothes", "outfits", "jeans", "furniture",
                               "wallpaper", "slang", "babies", "generation", "anime", "arcade", "sitcom",
                               "commercials", "vinyl", "glam", "grunge", "rap", "soul", "funk", "jazz", "country",
                               "dance", "metal", "hairstyles", "makeup", "throwback", "throwbacks", "child",
                               "children", "teens", "teenagers", "suburbia", "architecture", "furnishings",
                               "television", "cinema", "magazines", "toys", "computers", "hit", "song", "band",
                               "album", "movie", "film", "show", "classic", "icon", "star", "singer", "kid",
                               "look", "sound", "tune", "anthem", "ballad", "heartthrob", "sci", "horror",
                               "comedy", "nostalgic", "revivals", "remake", "remakes", "playlist", "mixtape"};
// Words after a hyphen that make a two-digit "s" form a decade ("60s-style", "a 90s-era bridge").
const rv::WordSet DECADE_COMPOUND{"style", "era", "inspired", "themed", "vintage", "retro", "influenced",
                                  "flavored", "flavoured", "sounding", "looking", "styled", "type", "vibe",
                                  "chic", "nostalgia", "born"};
// A single digit with "s" read as a plural number word ("his 3s and 4s").
const char* const DIGIT_PLURALS[] = {"zeros", "ones",   "twos",   "threes", "fours",
                                     "fives", "sixes",  "sevens", "eights", "nines"};
// Words before a two-digit decade ("the 70s", "their 40s", "the late 80s").
const rv::WordSet DECADE_BEFORE{"the", "their", "his", "her", "my", "our", "your", "its", "those", "these",
                                "early", "mid", "late", "upper", "lower", "low", "high", "roaring", "swinging",
                                "golden", "turbulent"};

// The token after the number n is a plural "s" (also "'s"): returns the token of the "s", or npos.
size_t plural_s(const Doc& d, size_t e, bool& apos) {
    apos = false;
    size_t k = next_tok(d, e);
    if (k >= d.size() || !glued(d, e, k)) return Doc::npos;
    if (apostrophe(d[k].cp) && d[k].kind == Kind::Punct) {
        size_t m = next_tok(d, k);
        if (!d.word(m) || !glued(d, k, m) || d.str(m) != "s") return Doc::npos;
        apos = true;
        return m;
    }
    return d.word(k) && d.str(k) == "s" ? k : Doc::npos;
}

// "1960s", "the 1900s", "the 70s", "'90s", "1960's".
RV_RULE("numbers_core", decades) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        std::string_view s = doc.str(i);
        if (s.size() < 2 || s.size() > 4 || s.back() != '0' || s[0] == '0') continue;
        if (s.size() == 3 && s.substr(1) != "00") continue;
        // "'90s": an apostrophe before a two-digit decade, itself free
        size_t first = i;
        bool short_form = false;
        if (s.size() == 2 && i > 0 && !doc[i].space_before && doc[i - 1].kind == Kind::Punct &&
            apostrophe(doc[i - 1].cp)) {
            size_t a = i - 1;
            if (a == 0 || doc[a].space_before || (doc[a - 1].kind == Kind::Punct && doc[a - 1].cp != '.' &&
                                                  !apostrophe(doc[a - 1].cp))) {
                first = a;
                short_form = true;
            }
        }
        if (!short_form && !free_before(doc, i) && !(after_range_dash(doc, i) && doc.str(i - 2) == "s")) continue;
        bool apos = false;
        size_t k = plural_s(doc, i, apos);
        if (k == Doc::npos || !free_after(doc, k)) continue;
        uint32_t v = value4(std::string(s));
        std::string words;
        int score = 0;
        if (s.size() == 4) {
            if (v < 1000 || v > 2090) continue;
            words = plural_of(year_words(v));
            // never seconds: above the time-unit readings
            score = apos ? 50 : 64;
        } else if (s.size() == 3) {  // "the 100s", "the 500s"
            size_t w = word_before(doc, first);
            if (w == Doc::npos || !DECADE_BEFORE.contains(doc.low(w))) continue;
            words = plural_of(year_words(v));
            score = 60;
        } else {
            words = plural_of(rv::cardinal(v));
            size_t nx = k + 1;
            bool seconds = doc.word(nx) && doc[nx].space_before && SECONDS_AFTER.contains(doc.low(nx));
            size_t w = word_before(doc, first);
            bool after_decade = after_range_dash(doc, first) && doc.str(first - 2) == "s";  // "the 1960s-70s"
            bool compound = doc.punct(nx, '-') && !doc[nx].space_before && doc.word(nx + 1) &&
                            !doc[nx + 1].space_before && DECADE_COMPOUND.contains(doc.low(nx + 1));
            if (short_form || after_decade) {
                score = seconds ? 0 : 65;
            } else if (seconds) {
                score = 0;
            } else if (compound) {
                score = 63;  // a cue decides: above the time-unit readings
            } else if (w != Doc::npos && DECADE_BEFORE.contains(doc.low(w))) {
                score = 65;
            } else if (doc.word(nx) && doc[nx].space_before && DECADE_AFTER.contains(doc.low(nx))) {
                score = 69;  // "a 70s hit": above seconds after an article (datetime, 68)
            } else if (doc.sentence_start(first)) {
                score = 40;
            }
            if (!score && !seconds && w != Doc::npos && YEAR_LINK.contains(doc.low(w))) {
                // "the 60s and 70s", "the 80’s and 90’s"; "15s and 30s" are seconds
                size_t p = doc.prev_word(w);
                size_t q = p != Doc::npos && p > 0 && apostrophe(doc[p - 1].cp) && doc[p - 1].kind == Kind::Punct
                               ? p - 2
                               : p - 1;
                if (p != Doc::npos && doc.str(p) == "s" && p > 0 && q < doc.size() && doc.number(q) &&
                    doc.str(q).back() == '0')
                    score = 65;
            }
            // "70s posters": before a plural noun that is no stretch of time, a decade
            if (!score && !seconds && doc.word(nx) && doc[nx].space_before && doc[nx].casing == rv::LOWER &&
                plural_like(doc.low(nx)) && !SECONDS_AFTER.contains(doc.low(nx)))
                score = 45;
            // (not a decade: seconds, read by datetime)
        }
        if (score) out.tokens(first, k, words, score_span(doc, first, k, score));
        i = k;
    }
}

// "1000s of people", "100s of messages", "10,000s of users": a round number made plural before "of".
RV_RULE("numbers_core", round_plurals) {
    static const std::pair<const char*, const char*> ROUND[] = {
        {"10", "tens"},          {"100", "hundreds"},        {"1000", "thousands"},
        {"10000", "tens of thousands"}, {"100000", "hundreds of thousands"}, {"1000000", "millions"}};
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) || doc.str(i)[0] != '1') continue;
        Quick q = quick(doc, i);
        bool qa = false;
        if (plural_s(doc, q.end, qa) == Doc::npos || !free_before(doc, i)) {
            i = q.end;
            continue;
        }
        Num n;
        scan(doc, i, n);
        if (!n.integer() || (n.sep && n.sep != ',')) continue;
        bool apos = false;
        size_t k = plural_s(doc, n.last, apos);
        if (k == Doc::npos || !free_after(doc, k) || !doc.is(k + 1, "of") || !doc[k + 1].space_before) continue;
        // "within 10s of a load change" is a time; "10s of thousands" is not
        size_t m = k + 2;
        if (!doc.word(m) || DETERMINERS.contains(doc.low(m)) || doc.is(m, "a") || doc.is(m, "an")) continue;
        if (n.whole == "10" && !scale_word(doc.low(m))) continue;
        for (const auto& r : ROUND)
            if (n.whole == r.first) out.tokens(i, k, r.second, 66);
        i = k;
    }
}

const rv::WordSet POSSESSIVE{"his", "her", "their", "my", "your", "our", "its"};

// "his 3s and 4s", "mixed up her 3s and 8s": single digits made plural, in pairs or after a possessive.
RV_RULE("numbers_core", digit_plurals) {
    auto plural_at = [&](size_t i) -> const char* {
        if (!doc.number(i)) return nullptr;
        std::string_view s = doc.str(i);
        if ((s.size() != 1 && s != "10") || !free_before(doc, i)) return nullptr;
        bool apos = false;
        size_t k = plural_s(doc, i, apos);
        if (k == Doc::npos || !free_after(doc, k)) return nullptr;
        return s == "10" ? "tens" : DIGIT_PLURALS[s[0] - '0'];
    };
    for (size_t i = 0; i < doc.size(); ++i) {
        const char* a = plural_at(i);
        if (!a) continue;
        bool apos = false;
        size_t k = plural_s(doc, i, apos);
        size_t link = k + 1;
        const char* b = nullptr;
        size_t j = Doc::npos;
        if (doc.word(link) && (doc.is(link, "and") || doc.is(link, "or")) && doc.number(link + 1) &&
            doc[link + 1].space_before) {
            j = link + 1;
            b = plural_at(j);
        }
        size_t w = word_before(doc, i);
        bool owner = w != Doc::npos && POSSESSIVE.contains(doc.low(w));
        bool the = w != Doc::npos && (doc.is(w, "the") || doc.is(w, "all") || doc.is(w, "those"));
        if (b && (owner || the)) {
            bool aj = false;
            size_t kb = plural_s(doc, j, aj);
            out.tokens(i, k, a, 66);
            out.tokens(j, kb, b, 66);
            i = kb;
        } else if (owner && !(doc.word(k + 1) && SECONDS_AFTER.contains(doc.low(k + 1)))) {
            out.tokens(i, k, a, 66);
            i = k;
        }  // (otherwise seconds, "the 3s warning": datetime)
    }
}

// Words before a short year ("the class of '98", "summer '24").
const rv::WordSet SHORT_YEAR_BEFORE{
    "of", "in", "since", "from", "until", "till", "by", "to", "the", "summer", "spring", "fall", "autumn",
    "winter", "class", "season", "back", "early", "late", "mid", "circa", "around", "before", "after", "during",
    "through", "and", "or", "vintage", "cohort", "batch", "christmas", "easter", "election", "festival", "tour",
    "championship", "cup", "world", "olympics", "graduated", "grad", "alumni", "alum"};

// "'91", "’98", "'05": a two-digit year after an apostrophe.
RV_RULE("numbers_core", short_years) {
    for (size_t i = 1; i < doc.size(); ++i) {
        if (!doc.number(i) || doc.str(i).size() != 2 || doc[i].space_before) continue;
        size_t a = i - 1;
        if (doc[a].kind != Kind::Punct || !apostrophe(doc[a].cp)) continue;
        if (!(a == 0 || doc[a].space_before ||
              (doc[a - 1].kind == Kind::Punct && doc[a - 1].cp != '.' && !apostrophe(doc[a - 1].cp))))
            continue;
        if (!free_after(doc, i)) continue;
        size_t k = next_tok(doc, i);
        if (k < doc.size() && glued(doc, i, k) && doc[k].kind == Kind::Punct && apostrophe(doc[k].cp)) continue;
        std::string_view s = doc.str(i);
        std::string words = s[0] == '0' ? "oh " + rv::digits(s.substr(1)) : rv::cardinal(value4(std::string(s)));
        size_t w = word_before(doc, a);
        int score = 0;
        if (w != Doc::npos && (SHORT_YEAR_BEFORE.contains(doc.low(w)) || MONTHS.contains(doc.low(w)))) {
            score = 65;
        } else {
            // no cue: a quotation if a closing quote (glued to the word before it) follows soon
            bool closed = false;
            for (size_t j = i + 1; j < doc.size() && j < i + 12 && !closed; ++j) {
                if (doc[j].kind == Kind::Punct && (doc[j].cp == '.' || doc[j].cp == '!' || doc[j].cp == '?')) break;
                closed = doc[j].kind == Kind::Punct && apostrophe(doc[j].cp) && !doc[j].space_before;
            }
            score = closed ? 0 : 35;
        }
        if (score) out.tokens(a, i, words, score);
    }
}

// "mid-2025", "late-2024", "early-1990s", "mid-80s", "pre-2020": the year or decade after the hyphen, which
// becomes a space.
const rv::WordSet YEAR_PREFIX{"mid", "early", "late", "pre", "post"};

RV_RULE("numbers_core", prefixed_years) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        size_t t = i + 1, k = i + 2;
        if (!(doc.punct(t, '-') || doc.punct(t, 0x2013)) || !doc.word(i) || doc[t].space_before ||
            doc[k].space_before || !YEAR_PREFIX.contains(doc.low(i)))
            continue;
        size_t first_num = k;
        if (doc[k].kind == Kind::Punct && apostrophe(doc[k].cp)) first_num = k + 1;
        if (!doc.number(first_num) || doc[first_num].space_before) continue;
        std::string_view s = doc.str(first_num);
        bool apos = false;
        size_t sk = plural_s(doc, first_num, apos);
        std::string words;
        size_t last = first_num;
        uint32_t v = value4(std::string(s));
        if (sk != Doc::npos && s.back() == '0' && (s.size() == 2 || (s.size() == 4 && v >= 1000 && v <= 2090))) {
            words = plural_of(s.size() == 4 ? year_words(v) : rv::cardinal(v));
            last = sk;
        } else if (s.size() == 4 && first_num == k && v >= 1000 && v <= 2099) {
            words = year_words(v);
        } else {
            continue;
        }
        if (!free_after(doc, last)) continue;
        out.tokens(t, last, words, 70);
        i = last;
    }
}

}  // namespace
