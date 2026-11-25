// Rules of the module "numbers_core": numbers in every written form, ordinals, signs, approximations, ranges of
// plain numbers, scientific notation, percent. Years and decades: numbers_core/years.cpp. Shared scanner and
// number words: numbers_core/num.h.
#include "core/rule.h"
#include "rules/numbers_core/num.h"

namespace {

using namespace numbers_core;
using rv::Kind;

bool percent_sign(uint32_t c) { return c == '%' || c == 0x2030 || c == 0xFF05 || c == 0xFE6A || c == 0x066A; }
bool currency(uint32_t c) { return c == '$' || c == 0x20AC || c == 0xA3 || c == 0xA5 || c == 0x20B9; }

// A '.' that starts a number (".5"): not stuck to a word, a number, another point or a currency sign.
bool lead_point(const Doc& d, size_t i) {
    if (!d.punct(i, '.')) return false;
    size_t k = next_tok(d, i);
    if (!d.number(k) || d[k].space_before) return false;
    if (i == 0 || d[i].space_before) return true;
    size_t p = prev_tok(d, i);
    if (p == Doc::npos || d[p + 1].space_before) return true;
    if (d[p].kind != Kind::Punct) return false;
    uint32_t c = d[p].cp;
    return c != '.' && c != ',' && !currency(c) && c != '/' && c != '\\' && c != ':' && c != '_' && c != '#';
}

const rv::WordSet ORDINAL_SUFFIX{"st", "nd", "rd", "th"};

// Dashes and hyphens between the ends of a range: - ‐ ‑ ‒ – —
bool dash_char(uint32_t c) { return c == '-' || (c >= 0x2010 && c <= 0x2014); }

// Token i starts a number this module may read: a free Number, the second end of a range, or a leading point.
bool number_start(const Doc& d, size_t i) {
    if (d.number(i)) return free_before(d, i) || after_range_dash(d, i);
    return lead_point(d, i);
}

// The token right after a number ending at token e is c, glued to it.
bool punct_after(const Doc& d, size_t e, uint32_t c) {
    size_t t = e + 1;
    while (invisible_tok(d, t)) ++t;
    return d.punct(t, c) && !d[t].space_before;
}

// Words after a number that do not show that it counts something.
const rv::WordSet FUNCTION_AFTER{"and", "or", "to", "of", "in", "on", "at", "by", "for", "from", "with", "is",
                                 "was", "are", "were", "be", "the", "a", "an", "as", "than", "then", "but",
                                 "if", "so", "nor", "vs", "versus", "per", "via", "into", "onto", "over", "under"};

// "1 234 567", "27'530", "1_000_000", ".5", numbers with invisible characters inside, numbers longer than the
// core reads. The core fallback reads the other plain forms.
RV_RULE("numbers_core", forms) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) && !doc.punct(i, '.')) continue;
        if (doc.number(i)) {
            Quick q = quick(doc, i);
            if (!q.special && q.digits <= 24) {
                i = q.end;
                continue;
            }
        }
        if (!number_start(doc, i)) continue;
        Num n;
        if (!scan(doc, i, n)) continue;
        size_t nx = next_tok(doc, n.last);
        bool unit_after = n.sep && n.sep != ',' && doc.word(nx) && !doc[nx].space_before;  // "1 000km"
        if (free_after(doc, n.last) || unit_after) {
            int score = 0;
            // three groups or more ("58 400 000") cannot be anything else, nor can groups before what they count
            // ("412 900 hours"): above the readings of the parts (a status code, a pair)
            if (n.sep == ' ') {
                bool counts = doc.word(nx) && doc[nx].space_before && doc[nx].casing == rv::LOWER &&
                              !FUNCTION_AFTER.contains(doc.low(nx));
                score = counts ? 71 : n.whole.size() > 6 ? 60 : 40;
            }
            else if (n.sep == '\'' || n.sep == '_') score = 45;
            else if (n.whole.empty()) score = 40;
            else if (n.whole.size() > 24) score = 15;
            else if (n.invisible) score = 12;
            if (score) {
                // a long digit string without separators is a code: digit by digit
                std::string w = n.whole.size() > 24 && !n.sep ? rv::digits(n.whole) : num_words(n);
                if (!w.empty()) out.tokens(n.first, n.last, w, score_span(doc, n.first, n.last, score));
            }
        }
        i = n.last;
    }
}

// "35,00,000", "1,23,456", "2,50,00,000": Indian grouping (pairs of digits before the last three), read in lakh
// and crore as it is written.
RV_RULE("numbers_core", lakh_grouping) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) || doc.str(i).size() > 2 || doc.str(i)[0] == '0' || !doc.punct(i + 1, ',') ||
            !doc.number(i + 2) || doc.str(i + 2).size() != 2 || !free_before(doc, i))
            continue;
        std::string groups[8];
        size_t g = 0, last = i;
        groups[g++] = std::string(doc.str(i));
        while (g < 8 && doc.punct(last + 1, ',') && !doc[last + 1].space_before && doc.number(last + 2) &&
               !doc[last + 2].space_before) {
            groups[g++] = std::string(doc.str(last + 2));
            last += 2;
        }
        if (g < 3 || groups[g - 1].size() != 3 || !free_after(doc, last)) {
            i = last;
            continue;
        }
        bool pairs_ok = true;
        for (size_t t = 1; t + 1 < g; ++t) pairs_ok &= groups[t].size() == 2;
        if (!pairs_ok || g > 5) {
            i = last;
            continue;
        }
        // groups: [crore...] lakh-part thousand-part hundreds; read from the right
        std::string digits;
        for (size_t t = 0; t < g; ++t) digits += groups[t];
        std::string w;
        size_t len = digits.size();
        uint64_t crore = len > 7 ? std::stoull(digits.substr(0, len - 7)) : 0;
        uint64_t lakh = len > 5 ? std::stoull(digits.substr(len > 7 ? len - 7 : 0, len > 7 ? 2 : len - 5)) : 0;
        uint64_t rest = std::stoull(digits.substr(len - 5));
        auto add = [&](const std::string& part) {
            if (!w.empty()) w += ' ';
            w += part;
        };
        if (crore) add(rv::cardinal(crore) + " crore");
        if (lakh) add(rv::cardinal(lakh) + " lakh");
        if (rest) add(rv::cardinal(rest));
        if (!w.empty()) out.tokens(i, last, w, 40);
        i = last;
    }
}

// "１２３", "３．５": full-width digits (each a token of its own), read like ASCII digits.
RV_RULE("numbers_core", full_width) {
    auto fw = [&](size_t t) { return t < doc.size() && doc[t].kind == Kind::Punct && doc[t].cp >= 0xFF10 &&
                                     doc[t].cp <= 0xFF19; };
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!fw(i)) continue;
        if (i > 0 && !doc[i].space_before && (doc.word(i - 1) || doc.number(i - 1))) continue;
        std::string whole, frac;
        size_t last = i;
        bool point = false;
        for (size_t t = i; t < doc.size() && (t == i || !doc[t].space_before); ++t) {
            if (fw(t)) {
                (point ? frac : whole) += static_cast<char>('0' + (doc[t].cp - 0xFF10));
                last = t;
            } else if (!point && (doc[t].cp == '.' || doc[t].cp == 0xFF0E) && fw(t + 1) && !doc[t + 1].space_before) {
                point = true;
            } else if (!point && (doc[t].cp == ',' || doc[t].cp == 0xFF0C) && fw(t + 1) && fw(t + 2) && fw(t + 3) &&
                       !fw(t + 4)) {
                continue;  // a thousands separator
            } else {
                break;
            }
        }
        if (whole.size() > 24 || (whole.size() > 1 && whole[0] == '0')) {
            i = last;
            continue;
        }
        Num n;
        n.whole = whole;
        n.frac = frac;
        n.point = point;
        std::string w = num_words(n);
        if (w.empty() || (doc.word(last + 1) && !doc[last + 1].space_before)) {
            i = last;
            continue;
        }
        if (last + 1 < doc.size() && doc[last + 1].kind == Kind::Punct && percent_sign(doc[last + 1].cp) &&
            (glued(doc, last, last + 1) || one_space(doc, last, last + 1))) {
            w += doc[last + 1].cp == 0x2030 ? " per mille" : " percent";
            ++last;
        }
        out.tokens(i, last, w, 40);
        i = last;
    }
}

// "14%", "14 %", "0.0001%", "7.5&nbsp;%", "99.9+%", "3‰".
RV_RULE("numbers_core", percent) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) && !doc.punct(i, '.')) continue;
        if (doc.number(i)) {
            Quick q = quick(doc, i);
            size_t t = q.end + 1;
            while (invisible_tok(doc, t)) ++t;
            if (t >= doc.size() || doc[t].kind != Kind::Punct ||
                !(percent_sign(doc[t].cp) || doc[t].cp == '+' || doc[t].cp == '&')) {
                i = q.end;
                continue;
            }
        }
        if (!number_start(doc, i)) continue;
        Num n;
        if (!scan(doc, i, n)) continue;
        size_t e = n.last, k = next_tok(doc, e), j = Doc::npos;
        bool plus = false;
        if (k < doc.size() && doc[k].kind == Kind::Punct) {
            if (doc[k].cp == '+' && glued(doc, e, k)) {
                size_t m = next_tok(doc, k);
                if (m < doc.size() && percent_sign(doc[m].cp) && glued(doc, k, m)) j = m, plus = true;
            } else if (percent_sign(doc[k].cp) && (glued(doc, e, k) || one_space(doc, e, k))) {
                j = k;
            } else if (doc[k].cp == '&' && glued(doc, e, k) && doc.is(k + 1, "nbsp") && !doc[k + 1].space_before &&
                       doc.punct(k + 2, ';') && !doc[k + 2].space_before && k + 3 < doc.size() &&
                       percent_sign(doc[k + 3].cp) && !doc[k + 3].space_before) {
                j = k + 3;
            }
        }
        if (j != Doc::npos && free_after(doc, j)) {
            std::string w = num_words(n);
            if (!w.empty()) {
                if (plus) w += " plus";
                w += doc[j].cp == 0x2030 ? " per mille" : " percent";
                out.tokens(n.first, j, w, score_span(doc, n.first, j, 50));
            }
        }
        i = n.last;
    }
}

// A percent sign whose number another rule reads ("7.5&nbsp;%" where the entity is read on its own).
RV_RULE("numbers_core", percent_sign_alone) {
    for (size_t i = 1; i < doc.size(); ++i) {
        if (doc[i].kind != Kind::Punct || !percent_sign(doc[i].cp)) continue;
        size_t p = prev_tok(doc, i);
        if (p == Doc::npos) continue;
        bool near = glued(doc, p, i) || one_space(doc, p, i);
        if (doc.punct(p, ';') && p >= 3 && doc.is(p - 1, "nbsp") && doc.punct(p - 2, '&') && !doc[p].space_before &&
            !doc[p - 1].space_before && !doc[i].space_before) {
            p -= 3;
            near = !doc[p + 1].space_before;
        }
        if (!near || !doc.number(p) || !free_after(doc, i)) continue;
        size_t start = p;  // the number must stand free ("的有效率为63.5%" is not read)
        while (start >= 2 && !doc[start].space_before && (doc.punct(start - 1, '.') || doc.punct(start - 1, ',')) &&
               !doc[start - 1].space_before && doc.number(start - 2))
            start -= 2;
        Num n;
        if (!free_before(doc, start) || !scan(doc, start, n) || n.last != p) continue;  // "99,9 %" stays
        out.tokens(i, i, doc[i].cp == 0x2030 ? "per mille" : "percent", 14);
    }
}

// Scale abbreviations after a number that the product rules do not read: "3bn", "2.4 bn", "40mn", "1.2 tn",
// "12 mln", and a spaced "M" or lower-case "k" ("2 M users", "2.5 k").
const rv::PhraseMap SCALE_ABBR{{"bn", "billion"}, {"bln", "billion"}, {"mn", "million"}, {"mln", "million"},
                               {"tn", "trillion"}, {"trn", "trillion"}};
// Words near a spaced "M" that make it molar ("a 0.5 M solution").
const rv::WordSet MOLAR_CUES{"concentration", "concentrations", "solution", "solutions", "molar", "molarity",
                             "buffer", "titrated", "dissolved", "diluted", "stock", "reagent", "acid", "base",
                             "nacl", "hcl", "kcl", "naoh", "tris", "edta", "sulfate", "chloride", "aqueous"};

// Words before "10 M" that make the number and the letter a name: a size, an exit, a version.
const rv::WordSet NAMED_BEFORE_M{"size", "sizes", "exit", "version", "gate", "platform", "route", "row", "model",
                                 "level", "grade", "type", "class", "series", "phase", "section", "unit"};

bool molar_context(const Doc& d, size_t i, size_t m) {
    for (size_t j = i > 6 ? i - 6 : 0; j < i; ++j)
        if (d.word(j) && MOLAR_CUES.contains(d.low(j))) return true;
    size_t k = m + 1;
    return d.word(k) && (MOLAR_CUES.contains(d.low(k)) || d[k].casing == rv::MIXED ||
                         (d[k].casing == rv::UPPER && d.number(k + 1) && !d[k + 1].space_before));
}

RV_RULE("numbers_core", scale_abbreviations) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        Quick q = quick(doc, i);
        size_t m = q.end + 1;
        if (!doc.word(m) || doc[m].end - doc[m].start > 3) {
            i = q.end;
            continue;
        }
        if (!number_start(doc, i)) continue;
        if (i > 0 && !doc[i].space_before && doc[i - 1].kind == Kind::Punct && currency(doc[i - 1].cp)) continue;
        if (i > 0 && doc[i].space_before && doc.word(i - 1) && doc[i - 1].casing == rv::UPPER &&
            doc[i - 1].end - doc[i - 1].start == 3)
            continue;  // "USD 4.5bn": money
        Num n;
        scan(doc, i, n);
        m = n.last + 1;
        if (!doc.word(m)) continue;
        bool spaced = doc[m].space_before;
        if (spaced && !one_space(doc, n.last, m)) continue;
        std::string_view w = doc.str(m);
        const std::string_view* abbr = SCALE_ABBR.find(doc.low(m));
        const char* word = nullptr;
        if (doc.low(m) == "pct" && doc[m].casing == rv::LOWER) word = "percent";  // "5 pct", "0.5pct"
        else if (abbr && (doc[m].casing == rv::LOWER || doc[m].casing == rv::UPPER)) word = abbr->data();
        else if (spaced && w == "M" && !molar_context(doc, i, m) &&
                 !(i > 0 && doc.word(i - 1) && NAMED_BEFORE_M.contains(doc.low(i - 1))))
            word = "million";  // "size 10 M" is a shoe width, "Exit 3 M" an exit
        else if (spaced && w == "k") word = "thousand";
        if (!word) continue;
        // what follows: the end, punctuation that does not glue on, or a lower-case word
        size_t k = m + 1;
        if (k < doc.size() && !doc[k].space_before) {
            if (doc[k].kind != Kind::Punct) continue;
            uint32_t c = doc[k].cp;
            if (c == '/' || c == '-' || c == '^' || c == 0xB2 || c == 0xB3 || c == '\'' || c == 0x2019) continue;
        } else if (doc.word(k) && doc[k].casing != rv::LOWER && spaced) {
            continue;  // "5 M Street", "3 M Company"
        }
        // a currency code after it belongs to the money rules ("2 bn EUR")
        if (doc.word(k) && doc[k].casing == rv::UPPER && doc[k].end - doc[k].start == 3) continue;
        std::string nw = num_words(n);
        if (nw.empty()) continue;
        out.tokens(i, m, nw + " " + word, 55);
        i = m;
    }
}

// Unit names after which "101st" is read "one oh first" ("the 101st Airborne", "the 303rd Squadron").
const rv::WordSet MILITARY_UNIT{"airborne", "squadron", "fighter", "air", "infantry", "division", "wing",
                                "regiment", "battalion", "brigade", "reconnaissance", "cavalry", "armored",
                                "artillery", "marine", "marines", "fleet", "corps", "bomb", "bomber", "rescue"};

// Title-case words after which a one-word ordinal is part of a name ("Fifth Avenue", "First District").
const rv::WordSet ORDINAL_NAMES{"street", "st", "avenue", "ave", "district", "dist", "armored", "infantry",
                                "division", "div", "platoon", "precinct", "battalion", "regiment", "brigade",
                                "fleet", "army", "corps", "cavalry", "congressional", "marine", "marines", "ward",
                                "airborne", "squadron", "artillery", "company", "battery", "panzer", "parachute"};

// "1st", "23rd", "101st", "1,000,000th", "21st century".
RV_RULE("numbers_core", ordinals) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        Quick q = quick(doc, i);
        size_t t = q.end + 1;
        while (invisible_tok(doc, t)) ++t;
        if (!doc.word(t) || doc[t].space_before || doc[t].end - doc[t].start < 2 || doc[t].end - doc[t].start > 3) {
            i = q.end;
            continue;
        }
        if (!number_start(doc, i)) continue;
        Num n;
        scan(doc, i, n);
        size_t k = next_tok(doc, n.last);
        // "three 1sts and two 2nds": a plural ordinal
        bool plural = doc.word(k) && doc.low(k).size() == 3 && doc.low(k).back() == 's' &&
                      ORDINAL_SUFFIX.contains(doc.low(k).substr(0, 2));
        if (n.integer() && doc.word(k) && glued(doc, n.last, k) &&
            (ORDINAL_SUFFIX.contains(doc.low(k)) || plural) && free_after(doc, k)) {
            std::string w = num_words(n);
            if (!w.empty()) {
                // "the 101st Airborne", "Unit 105th": a military unit with a zero in the middle, "one oh first"
                const std::string& d3 = n.whole;
                size_t nx = k + 1;
                if (d3.size() == 3 && d3[1] == '0' && d3[2] != '0' &&
                    ((doc.word(nx) && doc[nx].casing == rv::TITLE && MILITARY_UNIT.contains(doc.low(nx))) ||
                     (n.first > 0 && doc.is(n.first - 1, "unit"))))
                    w = whole_words(d3.substr(0, 1)) + " oh " + rv::ordinal(static_cast<uint64_t>(d3[2] - '0'));
                else
                    w = rv::ordinal_of(w);
                // "the 1st District", "W. 14th St.", "the 7th Armored": a one-word ordinal in a name has a capital
                if (doc.word(nx) && doc[nx].space_before && doc[nx].casing == rv::TITLE &&
                    ORDINAL_NAMES.contains(doc.low(nx)) && w.find_first_of(" -") == std::string::npos)
                    w = rv::capital(w);
                if (plural) w += 's';
                out.tokens(n.first, k, w, score_span(doc, n.first, k, 60));
            }
        }
        i = n.last;
    }
}

bool sign_char(uint32_t c) {
    return c == '-' || c == 0x2212 || c == 0x2013 || c == '+' || c == 0xB1 || c == 0x2213 || c == 0xFF0D ||
           c == 0xFE63 || c == 0xFF0B;  // full-width and small minus and plus
}

// The sign stands where a sign can stand: at the start, after a space, or after an opening bracket, a quote or
// an operator; never stuck to a word or a number ("Qwen3-0.6B", "UTC-5", "10-15").
bool sign_position(const Doc& d, size_t i) {
    if (i == 0 || d[i].space_before) {
        if (d[i].cp != 0x2013) return true;
        size_t p = prev_tok(d, i);  // "10 –15" is a range, not a negative number
        return p == Doc::npos || d[p].kind != Kind::Number;
    }
    size_t p = prev_tok(d, i);
    if (p == Doc::npos || d[p + 1].space_before) return true;
    if (d[p].kind != Kind::Punct) return false;
    switch (d[p].cp) {
        case '(': case '[': case '{': case '"': case '\'': case 0x201C: case 0x2018: case 0xAB: case '=':
        case ':': case '/': case ';': case '<': case '>': case 0x2264: case 0x2265: case '|': case '~':
        case 0x2248:
            return true;
        default:
            return false;
    }
}

// "-23", "−2.4%", "–3%", "+12%", "±5%": the sign alone, so that whatever reads the number still can.
RV_RULE("numbers_core", signs) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].kind != Kind::Punct || !sign_char(doc[i].cp)) continue;
        uint32_t c = doc[i].cp;
        size_t k = next_tok(doc, i);
        if (!glued(doc, i, k)) continue;
        bool ok = doc.number(k) || lead_point(doc, k);
        if (!ok && c != 0x2013 && k < doc.size() && currency(doc[k].cp)) {
            size_t m = next_tok(doc, k);
            ok = glued(doc, k, m) && (doc.number(m) || doc.punct(m, '.'));
        }
        if (!ok || !sign_position(doc, i)) continue;
        const char* w = c == '+' || c == 0xFF0B ? "plus"
                        : c == 0xB1            ? "plus or minus"
                        : c == 0x2213          ? "minus or plus"
                                               : "minus";
        out.tokens(i, i, w, score_span(doc, i, k, c == 0x2013 ? 40 : 45));
        // "±3mm": a unit glued to the number is read on its own; the number must not stay as digits
        Num n;
        if (doc.number(k) && scan(doc, k, n) && doc.word(n.last + 1) && !doc[n.last + 1].space_before) {
            std::string nw = num_words(n);
            if (!nw.empty()) out.tokens(k, n.last, nw, 11);
        }
    }
}

// Scale letters glued to a number that may carry a plus ("2.4B+ tokens", "10k+ users").
bool scale_letter(std::string_view w) {
    return w == "k" || w == "K" || w == "M" || w == "B" || w == "bn" || w == "m" || w == "mn" || w == "T";
}

// The signs of this module written as HTML entities: "1&ndash;2 hours", "&minus;5", "12&#37;", "&asymp;3". Above
// the generic entity reading (structure), which would leave the character unread.
RV_RULE("numbers_core", sign_entities) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.punct(i, '&')) continue;
        size_t last = 0;
        uint32_t cp = entity_at(doc, i, last);
        if (!cp) continue;
        bool num_before = i > 0 && doc.number(i - 1) && !doc[i].space_before;
        bool num_after = doc.number(last + 1) && !doc[last + 1].space_before;
        const char* w = nullptr;
        if ((cp == 0x2013 || cp == 0x2014) && num_before && num_after) w = "to";
        else if ((cp == 0x2212 || cp == 0xB1) && !num_before && num_after) w = cp == 0xB1 ? "plus or minus" : "minus";
        else if ((cp == '%' || cp == 0x2030) && num_before) w = cp == '%' ? "percent" : "per mille";
        else if ((cp == 0x2248 || cp == 0x223C) && !num_before &&
                 (num_after || (doc.number(last + 1) && one_space(doc, last, last + 1))))
            w = "approximately";
        if (w) out.tokens(i, last, w, 86);
        i = last;
    }
}

// "+/- 0.3", "+/-5%", "+-2": the plus-minus sign typed in ASCII.
RV_RULE("numbers_core", ascii_plus_minus) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.punct(i, '+') || !sign_position(doc, i)) continue;
        size_t last = i + 1;
        if (doc.punct(last, '/') && !doc[last].space_before) ++last;
        if (!doc.punct(last, '-') || doc[last].space_before) continue;
        size_t k = last + 1;
        if (k >= doc.size() || !(glued(doc, last, k) || one_space(doc, last, k))) continue;
        if (!(doc.number(k) || lead_point(doc, k) || currency(doc[k].cp))) continue;
        out.tokens(i, last, "plus or minus", 46);
        i = last;
    }
}

// "100+ languages", "30,000+", "2.4B+ tokens", "15%+": the plus after a number.
RV_RULE("numbers_core", plus_after) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        Quick q = quick(doc, i);
        size_t t = q.end + 1;  // a scale letter or word or a percent sign between the number and the plus
        if (t < doc.size() && !doc[t].space_before &&
            ((doc.word(t) && scale_letter(doc.str(t))) || (doc[t].kind == Kind::Punct && percent_sign(doc[t].cp))))
            ++t;
        else if (doc.word(t) && doc[t].space_before && scale_word(doc.low(t)) && doc.punct(t + 1, '+'))
            ++t;  // "1 million+ downloads"
        if (!doc.punct(t, '+') || doc[t].space_before || !free_before(doc, i)) {
            i = q.end;
            continue;
        }
        Num n;
        scan(doc, i, n);
        size_t e = n.last;
        if (t == n.last + 2) e = n.last + 1;
        if (t == n.last + 2 && doc[t - 1].space_before) {  // after a scale word the plus may stand at the end
            if (!glued(doc, t - 1, t)) continue;
            out.tokens(t, t, "plus", 45);
            i = t;
            continue;
        }
        size_t k = next_tok(doc, e);
        if (doc.punct(k, '+') && glued(doc, e, k)) {
            size_t m = next_tok(doc, k);
            bool stuck = m < doc.size() && !doc[m].space_before &&
                         (doc[m].kind != Kind::Punct || doc[m].cp == '+' || doc[m].cp == '=');
            if (!stuck) out.tokens(k, k, "plus", 45);
        }
        i = n.last;
    }
}

// Words after "~" that it makes approximate ("~midnight", "~half the users").
const rv::WordSet APPROX_WORDS{"midnight", "noon", "midday", "dawn", "dusk", "half", "a", "an", "one", "two",
                               "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve",
                               "twenty", "thirty", "forty", "fifty", "hundred", "dozen", "twice", "double",
                               "triple", "third", "quarter", "once", "thrice", "zero"};

// "~37", "≈ 90%", "~$5", "~midnight": approximately. After the word "approximately" the sign is left out.
RV_RULE("numbers_core", approx) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].kind != Kind::Punct) continue;
        uint32_t c = doc[i].cp;
        if (c != '~' && c != 0x2248 && c != 0x223C && c != 0x2243 && c != 0x2252 && c != 0xFF5E) continue;
        size_t k = next_tok(doc, i);
        bool near = k < doc.size() && (glued(doc, i, k) || one_space(doc, i, k));
        if (c == '~' && near && doc.word(k) && glued(doc, i, k) && APPROX_WORDS.contains(doc.low(k)) &&
            (i == 0 || doc[i].space_before) && !doc.punct(k + 1, '~')) {
            out.tokens(i, i, "approximately", 50);
            continue;
        }
        bool ok = near && (doc.number(k) || lead_point(doc, k));
        if (ok && i > 0 && doc[i].space_before && (doc.is(i - 1, "approximately") || doc.is(i - 1, "approx"))) {
            out.tokens(i, i, "", 55);  // "approximately ~1.5 miles"
            continue;
        }
        if (!ok && near && (currency(doc[k].cp) || sign_char(doc[k].cp))) {
            size_t m = next_tok(doc, k);
            ok = glued(doc, k, m) && doc.number(m);
        }
        if (!ok && c == 0x2248) {  // "≈ equal", "≈ zero"
            out.tokens(i, i, "approximately", 45);
            continue;
        }
        if (!ok) continue;
        if (c != 0x2248 && i > 0 && !doc[i].space_before) {  // "5~10", "x~5"
            size_t p = prev_tok(doc, i);
            if (p != Doc::npos && !doc[p + 1].space_before && doc[p].kind != Kind::Punct) continue;
        }
        out.tokens(i, i, "approximately", 50);
    }
}

// "ca. 30 people", "ca. 2.5 km": circa before a small number (datetime.era reads the years from 100 on).
RV_RULE("numbers_core", circa_small) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.is(i, "ca") || !doc.punct(i + 1, '.') || doc[i + 1].space_before || !doc.number(i + 2) ||
            !one_space(doc, i + 1, i + 2) || (i > 0 && !doc[i].space_before && doc[i - 1].kind != Kind::Punct))
            continue;
        Num n;
        scan(doc, i + 2, n);
        if (!free_after(doc, n.last) || n.sep) continue;
        if (n.integer() && n.whole.size() > 2) continue;
        std::string w = num_words(n);
        if (!w.empty()) out.tokens(i, n.last, "circa " + w, 66);
        i = n.last;
    }
}

// End of the operand of a range that starts with the number n: the number, or the number with an ordinal
// suffix or a percent sign.
size_t operand_end(const Doc& d, const Num& n) {
    size_t k = next_tok(d, n.last);
    if (k < d.size() && glued(d, n.last, k)) {
        if (d.word(k) && ORDINAL_SUFFIX.contains(d.low(k)) && n.integer()) return k;
        if (d.word(k) && d.str(k) == "s" && n.integer() && n.whole.back() == '0') return k;  // "1980s–1990s"
        if (d[k].kind == Kind::Punct && percent_sign(d[k].cp)) return k;
    }
    return n.last;
}

// "10–15", "10-15", "6–8 field visits", "1st–3rd", "10%–15%": the dash between two plain numbers is "to".
RV_RULE("numbers_core", ranges) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        {
            Quick q = quick(doc, i);
            size_t t = q.end + 1;
            while (invisible_tok(doc, t)) ++t;
            if (t < doc.size() && !doc[t].space_before &&
                ((doc.word(t) && doc[t].end - doc[t].start <= 2) || percent_sign(doc[t].cp)))
                ++t;
            if (t >= doc.size() || doc[t].kind != Kind::Punct || !dash_char(doc[t].cp)) {
                i = q.end;
                continue;
            }
        }
        if (!free_before(doc, i)) continue;
        if (i > 0 && !doc[i].space_before) {  // "1/2-3/4", "10:30-11"
            size_t p = prev_tok(doc, i);
            if (p != Doc::npos && (doc.punct(p, '/') || doc.punct(p, ':'))) continue;
        }
        Num a;
        scan(doc, i, a);
        size_t ea = operand_end(doc, a), t = next_tok(doc, ea);
        if (t >= doc.size() || doc[t].kind != Kind::Punct || !dash_char(doc[t].cp)) continue;
        bool tight = glued(doc, ea, t);
        size_t k = next_tok(doc, t);
        if (k >= doc.size() || !doc.number(k)) continue;
        if (tight ? !glued(doc, t, k)
                  : (doc[t].cp == '-' || !one_space(doc, ea, t) || !one_space(doc, t, k)))
            continue;
        Num b;
        scan(doc, k, b);
        size_t eb = operand_end(doc, b), after = next_tok(doc, eb);
        if (after < doc.size() && glued(doc, eb, after) && doc[after].kind == Kind::Punct) {
            uint32_t c = doc[after].cp;  // "2026-03-24", "555-123-4567", "1-2/3"
            if ((dash_char(c) || c == '/' || c == ':' || c == '.') && doc.number(next_tok(doc, after))) continue;
        }
        if (num_words(a).empty() || num_words(b).empty()) continue;
        if (tight && ea == a.last && eb == b.last && a.integer() && b.integer() &&
            ((a.whole == "50" && b.whole == "50") || (a.whole == "24" && b.whole == "7"))) {
            i = b.last;  // "a 50-50 chance", "open 24-7": no range (numbers_context)
            continue;
        }
        int score = !tight ? 35 : doc[t].cp == 0x2013 ? 40 : 35;
        // "Home record: 31-10", "a season record of 7-3": wins and losses, "and"
        bool record = false;
        for (size_t p = i, words = 0; p > 0 && words < 5 && !record; --p) {
            if (doc.punct(p - 1, '.') || doc.punct(p - 1, ';') || doc[p].newline_before) break;
            if (doc.word(p - 1)) {
                ++words;
                record = doc.is(p - 1, "record") || doc.is(p - 1, "records");
            }
        }
        if (record && doc.word(eb + 1) && doc[eb + 1].casing == rv::LOWER && doc.low(eb + 1).back() == 's')
            record = false;  // "a record 10-15 points": a range of something counted
        out.tokens(t, t, record && a.integer() && b.integer() ? "and" : "to", score_span(doc, ea, k, score));
        i = a.last;
    }
}

// A superscript digit's value, -1 if c is none.
int superscript_digit(uint32_t c) {
    if (c == 0x2070) return 0;
    if (c == 0xB9) return 1;
    if (c == 0xB2) return 2;
    if (c == 0xB3) return 3;
    if (c >= 0x2074 && c <= 0x2079) return static_cast<int>(c - 0x2070);
    return -1;
}

// The exponent after the "10" at token ten: "^6", "^-6", "^−6", "⁶", "⁻⁶". Returns its last token, or npos.
size_t power_of_ten(const Doc& d, size_t ten, std::string& exp, bool& neg) {
    exp.clear();
    neg = false;
    size_t k = next_tok(d, ten);
    if (!glued(d, ten, k)) return Doc::npos;
    if (d.punct(k, '^')) {
        size_t m = next_tok(d, k);
        if (!glued(d, k, m)) return Doc::npos;
        if (d.punct(m, '-') || d.punct(m, 0x2212) || d.punct(m, '+')) {
            neg = !d.punct(m, '+');
            size_t q = next_tok(d, m);
            if (!glued(d, m, q)) return Doc::npos;
            m = q;
        }
        if (!d.number(m) || d.str(m).size() > 3) return Doc::npos;
        exp = std::string(d.str(m));
        return m;
    }
    if (d.punct(k, 0x207B) || d.punct(k, 0x207A)) {
        neg = d.punct(k, 0x207B);
        size_t q = next_tok(d, k);
        if (!glued(d, k, q)) return Doc::npos;
        k = q;
    }
    size_t last = Doc::npos;
    while (k < d.size() && d[k].kind == Kind::Punct && superscript_digit(d[k].cp) >= 0 &&
           (last == Doc::npos || glued(d, last, k))) {
        exp += static_cast<char>('0' + superscript_digit(d[k].cp));
        last = k;
        k = next_tok(d, k);
    }
    return exp.empty() || exp.size() > 3 ? Doc::npos : last;
}

std::string power_words(const std::string& exp, bool neg) {
    return std::string("ten to the power of ") + (neg ? "minus " : "") + whole_words(exp);
}

// Token t may continue a number into scientific notation: "e", "x", "×", "·", "*", "^" or a superscript.
bool times_or_power(const Doc& d, size_t t) {
    if (t >= d.size()) return false;
    if (d[t].kind == Kind::Word) return d[t].end - d[t].start == 1;
    if (d[t].kind != Kind::Punct) return false;
    uint32_t c = d[t].cp;
    return c == 0xD7 || c == 0xB7 || c == '*' || c == 0x22C5 || c == '^' || c == 0x207B || c == 0x207A ||
           superscript_digit(c) >= 0;
}

// "2.3e4", "1.5E-6", "1.2×10^5", "3 x 10^8", "10^6", "10⁶".
RV_RULE("numbers_core", scientific) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        {
            Quick q = quick(doc, i);
            size_t t = q.end + 1;
            while (invisible_tok(doc, t)) ++t;
            if (!times_or_power(doc, t)) {
                i = q.end;
                continue;
            }
        }
        if (!free_before(doc, i)) continue;
        Num n;
        scan(doc, i, n);
        std::string exp;
        bool neg = false;
        size_t k = next_tok(doc, n.last);
        if (k >= doc.size() || (doc[k].kind == Kind::Number) || (n.whole != "10" && doc[k].kind == Kind::Punct &&
                                                                  doc[k].cp != 0xD7 && doc[k].cp != 0xB7 &&
                                                                  doc[k].cp != '*' && doc[k].cp != 0x22C5)) {
            i = n.last;
            continue;
        }
        std::string mantissa = num_words(n);
        // e-notation: the mantissa, "e" or "E", an optional sign, at most three digits
        if (!n.sep && !mantissa.empty() && doc.word(k) && glued(doc, n.last, k) && doc.low(k) == "e") {
            size_t m = next_tok(doc, k);
            bool sign = false;
            if (glued(doc, k, m) && (doc.punct(m, '-') || doc.punct(m, 0x2212) || doc.punct(m, '+'))) {
                neg = !doc.punct(m, '+');
                sign = true;
                size_t q = next_tok(doc, m);
                m = glued(doc, m, q) ? q : Doc::npos;
            } else if (!glued(doc, k, m)) {
                m = Doc::npos;
            }
            if (m != Doc::npos && doc.number(m) && doc.str(m).size() <= 3 && free_after(doc, m)) {
                // a decimal mantissa or a signed exponent is no product code ("3.5E-3")
                int score = sign || n.point ? 64 : 60;
                out.tokens(i, m, mantissa + " times " + power_words(std::string(doc.str(m)), neg), score);
                i = m;
                continue;
            }
        }
        // mantissa × 10^N
        if (!mantissa.empty() && k < doc.size() && (glued(doc, n.last, k) || one_space(doc, n.last, k))) {
            bool times = doc[k].kind == Kind::Punct
                             ? (doc[k].cp == 0xD7 || doc[k].cp == 0xB7 || doc[k].cp == '*' || doc[k].cp == 0x22C5)
                             : doc.is(k, "x");
            size_t ten = next_tok(doc, k);
            if (times && doc.number(ten) && (glued(doc, k, ten) || one_space(doc, k, ten)) && doc.str(ten) == "10") {
                size_t last = power_of_ten(doc, ten, exp, neg);
                if (last != Doc::npos && free_after(doc, last)) {
                    out.tokens(i, last, mantissa + " times " + power_words(exp, neg), 65);
                    i = last;
                    continue;
                }
            }
        }
        // 10^N alone
        if (n.integer() && !n.sep && n.whole == "10") {
            size_t last = power_of_ten(doc, n.last, exp, neg);
            if (last != Doc::npos && free_after(doc, last)) {
                // the labels read a superscript two on ten as "ten squared", a caret as a power
                bool squared = exp == "2" && !neg && doc[last].cp == 0xB2;
                out.tokens(i, last, squared ? "ten squared" : power_words(exp, neg), 55);
                i = last;
                continue;
            }
        }
        i = n.last;
    }
}

// "2.1.4": the three parts of a dotted number with "point" between ("section 2.1.4"). Versions, dates and
// addresses belong to other modules, which outscore this.
RV_RULE("numbers_core", dotted) {
    auto part = [&](size_t t) {
        std::string_view s = doc.str(t);
        return s.size() > 1 && s[0] == '0' ? rv::digits(s) : whole_words(s);
    };
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) || !free_before(doc, i)) continue;
        size_t last = i;
        while (doc.punct(last + 1, '.') && !doc[last + 1].space_before && doc.number(last + 2) &&
               !doc[last + 2].space_before)
            last += 2;
        // "1.000.000": groups of three after the first, a number grouped with points
        bool grouped = last >= i + 4 && doc.str(i).size() <= 3 && doc.str(i)[0] != '0';
        for (size_t t = i + 2; grouped && t <= last; t += 2) grouped = doc.str(t).size() == 3;
        if (grouped && free_after(doc, last)) {
            std::string all;
            for (size_t t = i; t <= last; t += 2) all += doc.str(t);
            std::string w = whole_words(all);
            if (!w.empty()) out.tokens(i, last, w, 25);
        } else if (last == i + 4 && free_after(doc, last)) {  // three parts; four are an address (electronic)
            std::string w = part(i);
            for (size_t t = i + 2; t <= last; t += 2) w += " point " + part(t);
            out.tokens(i, last, w, 20);
        }
        i = last;
    }
}

// Words before a bracketed pair that make it an interval, not a citation ("values in [0, 1]", "95% CI [0.1, 0.3]").
const rv::WordSet INTERVAL_CUE{"in", "of", "range", "ranges", "interval", "intervals", "ci", "inside", "within",
                               "as", "was", "were", "is", "are", "band", "into", "window", "between", "bounds",
                               "spans", "span", "from", "to", "on", "over", "hdi", "cri", "pi"};

// A signed number inside an interval at token i: "0.1", "-2", "−0.5". Returns the last token, npos if none.
size_t signed_number(const Doc& d, size_t i, std::string& words) {
    bool neg = false;
    if (i < d.size() && (d.punct(i, '-') || d.punct(i, 0x2212))) {
        size_t k = next_tok(d, i);
        if (!glued(d, i, k)) return Doc::npos;
        neg = true;
        i = k;
    }
    Num n;
    if (!(d.number(i) || d.punct(i, '.')) || !scan(d, i, n)) return Doc::npos;
    words = num_words(n);
    if (words.empty()) return Doc::npos;
    if (neg) words = "minus " + words;
    return n.last;
}

// "[0.1, 0.3]", "values in [0, 1]", "a range of [18, 22]": an interval is read as a range, brackets left out.
// Whole numbers in brackets without such a word before them are citations (roman_citations).
RV_RULE("numbers_core", intervals) {
    for (size_t i = 0; i + 4 < doc.size(); ++i) {
        if (!doc.punct(i, '[') && !doc.punct(i, '(')) continue;
        size_t k = next_tok(doc, i);
        if (k >= doc.size() || !(glued(doc, i, k) || one_space(doc, i, k))) continue;
        std::string a, b;
        size_t ea = signed_number(doc, k, a);
        if (ea == Doc::npos) continue;
        // "[9%, 15%]": percentages, the sign said once at the end
        bool pa = ea + 1 < doc.size() && doc[ea + 1].kind == Kind::Punct && percent_sign(doc[ea + 1].cp) &&
                  !doc[ea + 1].space_before;
        if (pa) ++ea;
        size_t c = next_tok(doc, ea);
        if (!doc.punct(c, ',') || !glued(doc, ea, c)) continue;
        size_t m = next_tok(doc, c);
        if (m >= doc.size() || !(glued(doc, c, m) || one_space(doc, c, m))) continue;
        size_t eb = signed_number(doc, m, b);
        if (eb == Doc::npos) continue;
        bool pb = eb + 1 < doc.size() && doc[eb + 1].kind == Kind::Punct && percent_sign(doc[eb + 1].cp) &&
                  !doc[eb + 1].space_before;
        if (pa && !pb) continue;
        if (pb) {
            ++eb;
            b += " percent";
            // a signed start takes its own: "[−15%, −9%]" minus fifteen percent to minus nine percent
            if (pa && a.rfind("minus ", 0) == 0) a += " percent";
        }
        size_t close = next_tok(doc, eb);
        if (close >= doc.size() || !(doc.punct(close, ']') || doc.punct(close, ')')) ||
            !(glued(doc, eb, close) || one_space(doc, eb, close)))
            continue;
        bool decimal = a.find(" point ") != std::string::npos || b.find(" point ") != std::string::npos ||
                       a.rfind("point ", 0) == 0 || b.rfind("point ", 0) == 0;
        size_t w = i > 0 && doc[i].space_before && doc.word(i - 1) ? i - 1 : Doc::npos;
        bool cue = w != Doc::npos && INTERVAL_CUE.contains(doc.low(w));
        if (doc.punct(i, '(') ? !(decimal && cue) : !(decimal || cue)) continue;
        // whole numbers that a reference list below names ("[18] Ostrander, ...") are citations
        if (!decimal && doc.punct(i, '[') &&
            doc.text.find("\n[" + std::string(doc.str(k)) + "]") != std::string_view::npos)
            continue;
        out.tokens(i, close, a + " to " + b, 85);
        i = close;
    }
}

// "95% CI [0.1, 0.3]", "95% CI 0.1-0.3": a confidence interval after a percentage.
RV_RULE("numbers_core", confidence_interval) {
    for (size_t i = 1; i + 1 < doc.size(); ++i) {
        if (doc[i].kind != Kind::Word || doc[i].casing != rv::UPPER || !percent_sign(doc[i - 1].cp) ||
            doc.str(i) != "CI" || !doc[i].space_before)
            continue;
        size_t k = i + 1;
        if (!doc[k].space_before) continue;
        if (doc.punct(k, '[') || doc.punct(k, '(')) ++k;
        if (doc.number(k) || doc.punct(k, '-') || doc.punct(k, 0x2212) || doc.punct(k, '.'))
            out.tokens(i, i, "confidence interval", 60);
    }
}

// "COVID-19", "top-5", "sub-200 ms", "carbon-14": the hyphen between a word and a number is left out (the number
// reading then gets a space before it).
RV_RULE("numbers_core", word_hyphen) {
    for (size_t i = 1; i + 1 < doc.size(); ++i) {
        if (!doc.punct(i, '-') || doc[i].space_before || !doc.word(i - 1)) continue;
        if (i >= 2 && !doc[i - 1].space_before && doc.punct(i - 2, '-')) continue;  // "6-foot-5"
        size_t k = next_tok(doc, i);
        if (!doc.number(k) || !glued(doc, i, k)) continue;
        if (doc.is(i - 1, "utc") || doc.is(i - 1, "gmt")) continue;
        Num n;
        scan(doc, k, n);
        if (!free_after(doc, n.last)) continue;
        size_t after = next_tok(doc, n.last);
        if (after < doc.size() && glued(doc, n.last, after) && doc[after].kind == Kind::Punct &&
            doc.number(next_tok(doc, after)))
            continue;  // "AB-123-456", "ISBN-978-3"
        out.tokens(i, i, "", 20);
    }
}

// "mind your p's and q's", "your i's and j's": single lower-case letters made plural, in pairs.
const char* letter_plural(std::string_view w) {
    static const char* const NAMES[] = {"ays",  "bees", "cees", "dees",    "ees",  "efs",  "gees", "aitches", "eyes",
                                        "jays", "kays", "els",  "ems",     "ens",  "ohs",  "pees", "cues",    "ars",
                                        "esses", "tees", "yous", "vees", "double-yous", "exes", "wyes", "zees"};
    if (w.size() != 3 && w.size() != 5) return nullptr;
    if (w.back() != 's' || !(w.substr(1, w.size() - 2) == "'" || w.substr(1, w.size() - 2) == "’")) return nullptr;
    char c = static_cast<char>(w[0] | 0x20);
    return c >= 'a' && c <= 'z' ? NAMES[c - 'a'] : nullptr;
}

RV_RULE("numbers_core", letter_plurals) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        uint32_t len = doc[i].end - doc[i].start;
        if ((len != 3 && len != 5) || !doc.word(i) || !doc.word(i + 2) || !doc.is(i + 1, "and")) continue;
        std::string_view a = doc.str(i), b = doc.str(i + 2);
        bool lower = a[0] >= 'a' && b[0] >= 'a', upper = a[0] < 'a' && b[0] < 'a';
        const char *x = letter_plural(a), *y = letter_plural(b);
        if (!x || !y || !(lower || (upper && doc.low(i) == "p's" && doc.low(i + 2) == "q's"))) continue;
        if (doc.low(i) == "t's" && doc.low(i + 2) == "c's") x = "Ts", y = "Cs";  // terms and conditions
        out.tokens(i, i, x, 60);
        out.tokens(i + 2, i + 2, y, 60);
        i += 2;
    }
}

}  // namespace
