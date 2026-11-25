// Years in date constructs: eras (300 BC, AD 79), circa (c. 1450), quarters, halves and fiscal years (Q3 2026, H1,
// FY2025, FY25). Plain years, year ranges, short years, decades, falling ranges before BC and lone three-digit
// years before an era ("476 AD") belong to numbers_core.
#include "rules/datetime/dt.h"

namespace {

using namespace dt;

// An era marker at i: BC, AD, BCE, CE, B.C., A.D., B.C.E., C.E. Sets *last (a sentence-final period stays).
const char* era_at(const Doc& d, size_t i, size_t* last) {
    if (!d.word(i)) return nullptr;
    std::string_view w = d.str(i);
    if (w == "BC" || w == "AD" || w == "BCE" || w == "CE") {
        if (d.joined(i + 1) && d.word(i + 1)) return nullptr;
        *last = i;
        return w == "BC" ? "BC" : w == "AD" ? "AD" : w == "BCE" ? "BCE" : "CE";
    }
    // dotted: B . C . [E .]
    static const char* const DOTTED[][4] = {{"B", "C", "E", "BCE"}, {"B", "C", nullptr, "BC"},
                                            {"A", "D", nullptr, "AD"}, {"C", "E", nullptr, "CE"}};
    for (const auto& f : DOTTED) {
        size_t k = i;
        bool ok = true;
        for (int p = 0; p < 3 && f[p]; ++p) {
            if (!(d.word(k) && d.str(k) == f[p] && (p == 0 || d.joined(k)) && d.punct(k + 1, '.') && d.joined(k + 1))) {
                ok = false;
                break;
            }
            k += 2;
        }
        if (!ok) continue;
        *last = k - 1;
        if (period_ends_sentence(d, k - 1)) *last = k - 2;
        if (d.joined(k) && d.word(k)) continue;
        return f[3];
    }
    return nullptr;
}

// A year as written before an era: up to four digits, no leading zero.
bool era_year(const Doc& d, size_t i, int* y) {
    if (!d.number(i) || len(d, i) > 4 || d.str(i)[0] == '0') return false;
    *y = num(d, i, 4);
    return *y >= 1;
}

bool spaced(const Doc& d, size_t i) { return i < d.size() && d[i].space_before && !d[i].newline_before; }

// circa: c. 1450, ca. 1450, circa 1450. Returns the year token and sets the words for the marker.
size_t circa_at(const Doc& d, size_t i, std::string* words, bool* abbr) {
    if (d.word(i) && (d.str(i) == "c" || d.str(i) == "ca") && d.punct(i + 1, '.') && d.joined(i + 1) &&
        d.number(i + 2) && !(d.joined(i) && !punct_any(d, i - 1))) {
        *words = "circa ";
        *abbr = true;
        return i + 2;
    }
    if (d.is(i, "circa") && d.number(i + 1) && spaced(d, i + 1)) {
        *words = std::string(d.str(i)) + " ";
        *abbr = false;
        return i + 1;
    }
    return NPOS;
}

// Eras and circa: 300 BC, 1066 AD, AD 79, A.D. 79, 100-200 AD, c. 1450, ca. 1250 AD, circa 1450.
RV_RULE("datetime", era) {
    for (size_t i = 0; i < doc.size(); ++i) {
        const rv::Token& t = doc[i];
        if (t.kind == rv::Kind::Punct) continue;
        if (t.kind == rv::Kind::Word && t.casing != rv::UPPER && t.cp != 'c') continue;
        if (t.kind == rv::Kind::Number && !(spaced(doc, i + 1) && doc.word(i + 1) && doc[i + 1].casing == rv::UPPER) &&
            !dash(doc, i + 1))
            continue;
        int y1 = 0, y2 = 0;
        size_t el;
        // AD 79, A.D. 79
        const char* e = t.kind == rv::Kind::Word ? era_at(doc, i, &el) : nullptr;
        if (e && e[0] == 'A' && spaced(doc, el + 1) && era_year(doc, el + 1, &y1) && free_after(doc, el + 1) &&
            !(doc.joined(el + 2) && doc.word(el + 2))) {
            out.tokens(i, el + 1, std::string(e) + " " + year_words(y1), 76);
            i = el + 1;
            continue;
        }
        std::string words;
        bool abbr = false;
        size_t at = circa_at(doc, i, &words, &abbr);
        if (at == NPOS) {
            if (!doc.number(i) || !free_before(doc, i)) continue;
            at = i;
        }
        if (!era_year(doc, at, &y1)) continue;
        bool circa = at != i;  // after circa a three-digit year is read in pairs too: "c. 196 BC"
        words += circa ? year_pairs(y1) : year_words(y1);
        size_t last = at;
        size_t k = range_dash(doc, at + 1);
        if (k != NPOS && era_year(doc, k, &y2) && y2 > y1 &&
            ((spaced(doc, k + 1) && era_at(doc, k + 1, &el)) || (circa && y1 >= 100 && free_after(doc, k)))) {
            words += " to " + (circa ? year_pairs(y2) : year_words(y2));
            last = k;
        }
        e = spaced(doc, last + 1) ? era_at(doc, last + 1, &el) : nullptr;
        // a lone three-digit year that is no multiple of ten ("476 AD") is read in pairs by numbers_core
        if (e && !circa && last == at && y1 >= 100 && y1 <= 999 && y1 % 10 != 0) continue;
        if (e) {
            out.tokens(i, el, words + " " + e, at != i ? 78 : 76);
            i = el;
            continue;
        }
        if (circa && free_after(doc, last) && y1 >= 100 && y1 <= 2999 && !(doc.joined(last + 1) && doc.word(last + 1))) {
            out.tokens(i, last, words, abbr ? 78 : 65);
            i = last;
        }
    }
}

// Quarters, halves and fiscal years: Q3 Q three, Q3 2026, Q1–Q3, H1 2025, FY2025, FY25, FY 2024–25.
RV_RULE("datetime", fiscal) {
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        if (!doc.word(i) || doc[i].casing != rv::UPPER || len(doc, i) > 2 || (doc.joined(i) && !punct_any(doc, i - 1)))
            continue;
        std::string_view w = doc.str(i);
        if (w == "Q" || w == "H") {
            int n = num(doc, i + 1, 1);
            if (!doc.joined(i + 1) || n < 1 || n > (w == "Q" ? 4 : 2) || (doc.joined(i + 2) && doc.word(i + 2)))
                continue;
            std::string words = std::string(w) + " " + rv::cardinal(n);
            size_t last = i + 1;
            size_t k = range_dash(doc, last + 1);
            if (k != NPOS && doc.word(k) && doc.str(k) == w && doc.joined(k + 1) && num(doc, k + 1, 1) > n &&
                num(doc, k + 1, 1) <= (w == "Q" ? 4 : 2)) {
                words += " to " + std::string(w) + " " + rv::cardinal(num(doc, k + 1, 1));
                last = k + 1;
            }
            bool year = false;
            if (last + 1 < doc.size() && doc[last + 1].space_before && year4(doc, last + 1) && free_after(doc, last + 1)) {
                words += " " + year_words(num(doc, last + 1, 4));
                last += 1;
                year = true;
            } else if (dash(doc, last + 1) && doc.joined(last + 1) && doc.joined(last + 2) && year4(doc, last + 2) &&
                       free_after(doc, last + 2)) {  // Q3-2026
                words += " " + year_words(num(doc, last + 2, 4));
                last += 2;
                year = true;
            } else if (last + 2 < doc.size() && punct_any(doc, last + 1) && apostrophe_cp(doc[last + 1].cp) &&
                       doc.number(last + 2) && doc.joined(last + 2) && len(doc, last + 2) == 2 &&
                       free_after(doc, last + 2)) {  // Q4 '24, Q4'24
                std::string yy = two_digit_year(num(doc, last + 2, 2));
                if (!yy.empty()) {
                    words += " " + yy;
                    last += 2;
                    year = true;
                }
            }
            if (!free_after(doc, last)) continue;
            std::string_view cue = word_before(doc, i);
            int score = year ? 76 : (w == "Q" ? 72 : (cue == "in" || cue == "the" || cue == "first" || cue == "for" ||
                                                     cue == "during" || cue == "and" || cue == "by" ? 60 : 40));
            out.tokens(i, last, words, score);
            i = last;
            continue;
        }
        if (w == "FY") {
            size_t k = i + 1;
            // FY'25
            if (punct_any(doc, k) && apostrophe_cp(doc[k].cp) && doc.joined(k + 1) && doc.number(k + 1) &&
                len(doc, k + 1) == 2)
                ++k;
            if (!doc.number(k) || doc[k].newline_before) continue;
            size_t l = len(doc, k);
            std::string first;
            int y = num(doc, k, 4);
            if (l == 4 && y >= 1900 && y <= 2099) first = year_words(y);
            else if (l == 2) first = two_digit_year(y);
            if (first.empty()) continue;
            std::string words = "FY " + first;
            size_t last = k;
            size_t r = range_dash(doc, k + 1);
            if (r == NPOS && doc.punct(k + 1, '/') && doc.joined(k + 1) && doc.joined(k + 2)) r = k + 2;  // FY2025/26
            if (r != NPOS && doc.number(r) && (len(doc, r) == 2 || len(doc, r) == 4) && free_after(doc, r)) {
                std::string second = len(doc, r) == 4 ? year_words(num(doc, r, 4)) : two_digit_year(num(doc, r, 2));
                if (!second.empty()) {
                    words += " to " + second;
                    last = r;
                }
            }
            if (!free_after(doc, last) || (doc.joined(last + 1) && doc.word(last + 1))) continue;
            out.tokens(i, last, words, 78);
            i = last;
        }
    }
}

}  // namespace
