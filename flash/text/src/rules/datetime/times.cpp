// Clock times: 3:20 PM, 9:05, 6:00, 14:00, 3PM, 9 a.m., ranges (9:00-5:00, 9am-5pm, 9-11 a.m., 9-to-5),
// military times (0800, at 1530), time zone offsets (UTC+9, GMT−3).
#include "rules/datetime/dt.h"

namespace {

using namespace dt;

bool zone_at(const Doc& d, size_t i) {
    return d.word(i) && d[i].space_before && zone_word(d.str(i));
}

// One side of a time range or a single time at token i.
struct Side {
    bool ok = false;
    size_t first = 0, last = 0;
    int h = 0, m = 0;
    bool clock = false;   // written with minutes (H:MM)
    bool seconds = false;
    bool zulu = false;
    bool padded = false;  // written with a leading zero (08:00): a 24-hour clock
    const char* ap = nullptr;
};

Side side_at(const Doc& d, size_t i, bool bare_hour) {
    Side s;
    if (!d.number(i)) return s;
    Clock c = parse_clock(d, i);
    size_t last;
    if (c.ok) {
        s.h = c.h;
        s.m = c.m;
        s.clock = true;
        s.seconds = c.s >= 0;
        s.zulu = c.zulu;
        last = c.last;
    } else if (d.punct(i + 1, '.') && d.joined(i + 1) && d.number(i + 2) && d.joined(i + 2) && len(d, i + 2) == 2 &&
               len(d, i) <= 2) {
        // 9.30am: only with AM / PM
        int h = num(d, i, 2), m = num(d, i + 2, 2);
        size_t a = i + 3, al;
        bool ap = a < d.size() && !d[a].newline_before && ampm_at(d, a, &al);
        if (h < 1 || h > 12 || m > 59 || !ap) return s;
        s.h = h;
        s.m = m;
        s.clock = true;
        last = i + 2;
    } else {
        if (len(d, i) > 2) return s;
        int h = num(d, i, 2);
        if (h < 1 || h > 12) return s;
        s.h = h;
        last = i;
    }
    if (!s.zulu && last + 1 < d.size() && !d[last + 1].newline_before) {
        size_t al;
        const char* ap = ampm_at(d, last + 1, &al);
        if (ap) {
            s.ap = ap;
            last = al;
        }
    }
    if (!s.clock && !s.ap && !bare_hour) return s;
    s.padded = len(d, i) == 2 && d.str(i)[0] == '0';
    s.first = i;
    s.last = last;
    s.ok = true;
    return s;
}

std::string side_words(const Side& s, bool short_form) {
    std::string w;
    if (s.ap) w = hm_words(s.h, s.m) + " " + s.ap;
    else if (!s.clock) w = rv::cardinal(s.h);
    else if (!short_form && s.padded && s.m == 0 && s.h >= 1) w = rv::cardinal(s.h) + " hundred";  // 08:00
    else w = short_form ? clock_short(s.h, s.m) : clock_words(s.h, s.m);
    if (s.zulu) w += " UTC";
    return w;
}

bool side_free_after(const Doc& d, const Side& s) {
    if (s.ap) return !(d.joined(s.last + 1) && !punct_any(d, s.last + 1));
    return free_after(d, s.last);
}

// "~13:00": a free tilde before the number ("approximately" is read by its own rule).
bool tilde_before(const Doc& d, size_t i) {
    return d.joined(i) && d.punct(i - 1, '~') && (i == 1 || !d.joined(i - 1));
}

// "9am–noon", "10:00-midnight": the range ends at noon or midnight. Returns the word's token, or NPOS.
size_t noon_after(const Doc& d, size_t last) {
    size_t k = range_dash(d, last + 1);
    if (k == NPOS || !d.word(k) || !(d.is(k, "noon") || d.is(k, "midnight"))) return NPOS;
    return free_after(d, k) ? k : NPOS;
}

// Clock times written with minutes, alone or as the first part of a range.
RV_RULE("datetime", clock_time) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) || !(free_before(doc, i) || tilde_before(doc, i))) continue;
        Side a = side_at(doc, i, false);
        if (!a.ok || !a.clock) continue;
        if (!a.ap && !a.seconds && (ratio_after(doc, a.last) || ratio_before(doc, i)))
            continue;  // a score or a ratio (numbers_context)
        size_t nn = noon_after(doc, a.last);
        if (nn != NPOS) {
            out.tokens(i, nn, side_words(a, true) + " to " + std::string(doc.str(nn)), 76);
            i = nn;
            continue;
        }
        std::string_view cue = word_before(doc, i);
        bool zone = zone_at(doc, a.last + 1);
        // a range: 9:00-5:00, 10:00 AM - 12:00 PM, 8:00 a.m.–5:30 p.m., 10:00 a.m. to 12:00
        size_t k = range_dash(doc, a.last + 1);
        // a range into the next day: "Sat. 23:00–Sun. 03:00": the dash is "to" and the day reads itself
        if (k != NPOS && (weekday_abbrev_at(doc, k) || weekday_full_at(doc, k))) {
            out.tokens(i, a.last + 1, side_words(a, false) + " to", 72);
            i = a.last + 1;
            continue;
        }
        bool dashed = k != NPOS;
        if (!dashed && doc.is(a.last + 1, "to") && doc[a.last + 1].space_before && a.last + 2 < doc.size() &&
            doc[a.last + 2].space_before)
            k = a.last + 2;
        Side b;
        if (k != NPOS) b = side_at(doc, k, dashed);
        if (b.ok && side_free_after(doc, b) && !a.seconds && !b.seconds && (b.clock || b.ap)) {
            std::string wa, wb;
            if (dashed) {
                wa = side_words(a, true);
                wb = side_words(b, true);
            } else {
                // "10:00 a.m. to 12:00": an hour without AM / PM next to one with it is the hour alone
                wa = (!a.ap && b.ap && a.m == 0 && a.h >= 1 && a.h <= 12) ? rv::cardinal(a.h) : side_words(a, false);
                wb = (!b.ap && a.ap && b.m == 0 && b.h >= 1 && b.h <= 12) ? rv::cardinal(b.h) : side_words(b, false);
            }
            out.tokens(i, b.last, wa + " to " + wb, a.ap || b.ap ? 80 : 74);
            i = b.last;
            continue;
        }
        // a numeric offset after the time: 14:05:33 +0200, 09:30 -05:00
        std::string offset;
        size_t ol = a.last;
        if (!a.ap && !a.zulu && a.last + 2 < doc.size() && doc[a.last + 1].space_before && doc.joined(a.last + 2) &&
            punct_any(doc, a.last + 1) && (doc[a.last + 1].cp == '+' || is_minus_cp(doc[a.last + 1].cp)) &&
            doc.number(a.last + 2)) {
            size_t n = a.last + 2;
            int oh = -1, om = 0;
            size_t last = n;
            if (len(doc, n) == 4) {
                oh = num(doc, n, 4) / 100;
                om = num(doc, n, 4) % 100;
            } else if (len(doc, n) == 2 && doc.punct(n + 1, ':') && doc.joined(n + 1) && doc.number(n + 2) &&
                       len(doc, n + 2) == 2) {
                oh = num(doc, n, 2);
                om = num(doc, n + 2, 2);
                last = n + 2;
            }
            if (oh >= 0 && oh <= 14 && om < 60 && free_after(doc, last)) {
                offset = oh == 0 && om == 0 ? " UTC"
                                            : std::string(doc[a.last + 1].cp == '+' ? " plus " : " minus ") + hm_words(oh, om);
                ol = last;
            }
        }
        if (!offset.empty()) {
            out.tokens(i, ol, side_words(a, false) + offset, 72);
            i = ol;
            continue;
        }
        if (!side_free_after(doc, a)) continue;
        int score;
        if (a.ap || a.zulu) score = 80;
        else if (a.seconds) score = zone || (time_cue(cue) && len(doc, i) == 2) ? S_CUE : 45;
        else if (zone) score = 72;
        else if (time_cue(cue)) score = S_CUE;
        else score = S_SHAPE;
        out.tokens(i, a.last, side_words(a, false), score);
        i = a.last;
    }
}

const rv::WordSet HOURS_CUES{"open", "opens", "opened", "hours", "work", "works", "worked", "working", "shift",
                             "shifts", "staffed", "closed", "daily", "weekdays", "manned", "available"};

// An hour with AM / PM: 3PM, 9 a.m., and ranges 9am-5pm, 9-11 a.m., 3-5 PM, 9 AM - 5:30 PM, open 9–5.
RV_RULE("datetime", hour_ampm) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) || len(doc, i) > 2 || !(free_before(doc, i) || tilde_before(doc, i))) continue;
        if (doc.punct(i + 1, ':') && doc.joined(i + 1)) continue;
        Side a = side_at(doc, i, true);
        if (!a.ok) continue;
        size_t nn = a.ap ? noon_after(doc, a.last) : NPOS;
        if (nn != NPOS) {  // 9am–noon
            out.tokens(i, nn, side_words(a, true) + " to " + std::string(doc.str(nn)), 80);
            i = nn;
            continue;
        }
        size_t k = range_dash(doc, a.last + 1);
        Side b;
        if (k != NPOS) b = side_at(doc, k, true);
        if (b.ok && b.ap && side_free_after(doc, b)) {
            out.tokens(i, b.last, side_words(a, true) + " to " + side_words(b, true), 80);
            i = b.last;
            continue;
        }
        if (b.ok && !a.ap && !a.clock && !b.clock && side_free_after(doc, b) && HOURS_CUES.contains(word_before(doc, i))) {
            out.tokens(i, b.last, side_words(a, true) + " to " + side_words(b, true), 45);  // open 9–5
            i = b.last;
            continue;
        }
        if (!a.ap || !side_free_after(doc, a)) continue;
        out.tokens(i, a.last, side_words(a, false), 80);
        i = a.last;
    }
}

// noon–2 p.m., midnight–6:00: noon to two PM.
RV_RULE("datetime", noon_range) {
    if (doc.lower.find("noon") == std::string::npos && doc.lower.find("midnight") == std::string::npos) return;
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!dash(doc, i + 1) || !doc.word(i) || !(doc.is(i, "noon") || doc.is(i, "midnight"))) continue;
        size_t k = range_dash(doc, i + 1);
        if (k == NPOS || !doc.number(k)) continue;
        Side b = side_at(doc, k, false);
        if (!b.ok || b.seconds || !side_free_after(doc, b)) continue;
        out.tokens(i, b.last, std::string(doc.str(i)) + " to " + side_words(b, true), 80);
        i = b.last;
    }
}

// Shift ranges with one-letter AM / PM: 7a–7p, 11p-7a (seven AM to seven PM).
RV_RULE("datetime", shift_ap) {
    for (size_t i = 0; i + 4 < doc.size(); ++i) {
        if (!doc.number(i) || len(doc, i) > 2 || !free_before(doc, i)) continue;
        int h1 = num(doc, i, 2);
        if (h1 < 1 || h1 > 12 || !doc.word(i + 1) || !doc.joined(i + 1) || len(doc, i + 1) != 1) continue;
        char a1 = doc.lower[doc[i + 1].start];
        if (a1 != 'a' && a1 != 'p') continue;
        size_t k = range_dash(doc, i + 2);
        if (k == NPOS || !doc.number(k) || len(doc, k) > 2 || !doc.word(k + 1) || !doc.joined(k + 1) ||
            len(doc, k + 1) != 1)
            continue;
        int h2 = num(doc, k, 2);
        char a2 = doc.lower[doc[k + 1].start];
        if (h2 < 1 || h2 > 12 || (a2 != 'a' && a2 != 'p') || a1 == a2 || !free_after(doc, k + 1) ||
            (doc.joined(k + 2) && doc.word(k + 2)))
            continue;
        out.tokens(i, k + 1, rv::cardinal(h1) + (a1 == 'a' ? " AM to " : " PM to ") + rv::cardinal(h2) +
                                 (a2 == 'a' ? " AM" : " PM"), 78);
        i = k + 1;
    }
}

// 9-to-5, 9:00-to-5:00: nine to five.
RV_RULE("datetime", nine_to_five) {
    for (size_t i = 0; i + 4 < doc.size(); ++i) {
        if (!doc.number(i) || !free_before(doc, i)) continue;
        Side a = side_at(doc, i, true);
        if (!a.ok || a.seconds) continue;
        size_t k = a.last + 1;
        if (!dash(doc, k) || !doc.joined(k) || !doc.is(k + 1, "to") || !doc.joined(k + 1) || !dash(doc, k + 2) ||
            !doc.joined(k + 2) || !doc.joined(k + 3))
            continue;
        Side b = side_at(doc, k + 3, true);
        if (!b.ok || b.seconds || !side_free_after(doc, b)) continue;
        out.tokens(i, b.last, side_words(a, true) + " to " + side_words(b, true), 75);
        i = b.last;
    }
}

const rv::WordSet MILITARY_CUES{"at", "by", "until", "till", "before", "after", "from", "to", "between", "and", "for",
                                "the", "around", "since", "@"};
// Words after a military time that keep it a time: by 1600 at the latest, until 1700 to finish.
const rv::WordSet MILITARY_NEXT{"sharp", "hours", "hrs",  "tomorrow", "today", "tonight", "this", "on",   "at",
                                "for",   "to",    "and",  "or",       "local", "so",      "but",  "when", "with",
                                "then",  "zulu",  "please", "until",  "every", "each",    "daily", "military",
                                "time",  "next",  "instead", "if",    "as",    "while"};
// Nouns a military time describes: the 1530 briefing.
const rv::WordSet MILITARY_NOUNS{"briefing", "report", "meeting", "call", "shift", "flight", "train", "bus",
                                 "departure", "deadline", "slot", "standup", "check", "session", "appointment",
                                 "launch", "update", "sync", "review", "deployment", "rollout", "cutoff", "curfew",
                                 "arrival", "wakeup", "formation", "muster", "inspection", "patrol", "watch", "start",
                                 "hours", "hrs"};

// Words before "2000 hrs" that make it a count of hours: every 2000 hrs, logged 1050 hours.
const rv::WordSet QUANTITY_CUES{"every", "each", "of", "over", "under", "than", "about", "approximately", "around",
                                "another", "additional", "extra", "first", "last", "next", "logged", "tested", "ran",
                                "run", "took", "takes", "lasted", "spent", "stores", "needs", "within", "per",
                                "total", "roughly", "nearly", "almost", "some", "all", "those", "these", "only",
                                "just", "up", "past", "beyond", "exceeds", "exceeded", "reached", "hit", "clocked"};

bool military_shape(const Doc& d, size_t i, int* h, int* m) {
    if (!d.number(i) || len(d, i) != 4) return false;
    int v = num(d, i, 4);
    *h = v / 100;
    *m = v % 100;
    return *h <= 23 && *m <= 59;
}

std::string military_words(const Doc& d, size_t i, int h, int m) {
    std::string w;
    if (h == 0) w = "zero";
    else if (d.str(i)[0] == '0') w = "oh " + rv::cardinal(h);
    else w = rv::cardinal(h);
    if (m == 0) w += " hundred";
    else w += m < 10 ? " oh " + rv::cardinal(m) : " " + rv::cardinal(m);
    return w;
}

// What follows token i: 0 nothing that keeps a time (a number, a unit), 1 the end or punctuation, 2 a word like
// "sharp" or "tomorrow", 3 a noun like "briefing".
int military_follow(const Doc& d, size_t i) {
    size_t n = i + 1;
    if (n >= d.size() || d[n].newline_before) return 1;
    if (d[n].kind == rv::Kind::Punct) return free_after(d, i) && !dash(d, n) && d[n].cp != 0xD7 ? 1 : 0;
    if (d[n].kind == rv::Kind::Number) return 0;
    if (MILITARY_NOUNS.contains(d.low(n))) return 3;
    return MILITARY_NEXT.contains(d.low(n)) || zone_at(d, n) ? 2 : 0;
}

// Military times: 0800 oh eight hundred, at 1530 fifteen thirty, by 1600 at the latest, the 1530 briefing,
// 0800-1000. Four digits without a leading zero are a time only with a strong cue: they are years and counts too.
RV_RULE("datetime", military_time) {
    int doc_military = -1;  // the text has a military time with a leading zero (0800): computed when needed
    for (size_t i = 0; i < doc.size(); ++i) {
        int h, m;
        if (!military_shape(doc, i, &h, &m) || !free_before(doc, i)) continue;
        bool zero = doc.str(i)[0] == '0';
        std::string_view cue = word_before(doc, i);
        size_t k = range_dash(doc, i + 1);
        int h2, m2;
        if (k != NPOS && military_shape(doc, k, &h2, &m2) && (zero || doc.str(k)[0] == '0') && military_follow(doc, k)) {
            out.tokens(i, k, military_words(doc, i, h, m) + " to " + military_words(doc, k, h2, m2), 72);
            i = k;
            continue;
        }
        int follow = military_follow(doc, i);
        if (!follow) continue;
        // "0900 hrs": the abbreviation is read with the time
        size_t last = i;
        std::string hrs;
        if ((doc.is(i + 1, "hrs") || doc.is(i + 1, "hr")) && doc[i + 1].space_before && free_after(doc, i + 1)) {
            last = abbrev_end(doc, i + 1);
            hrs = " hours";
        }
        if (zero) {
            out.tokens(i, last, military_words(doc, i, h, m) + hrs, MILITARY_CUES.contains(cue) ? S_CUE : 45);
            i = last;
            continue;
        }
        if (doc_military < 0) {
            doc_military = 0;
            for (size_t k2 = 0; k2 < doc.size() && !doc_military; ++k2)
                if (doc.number(k2) && len(doc, k2) == 4 && doc.str(k2)[0] == '0' && military_shape(doc, k2, &h2, &m2) &&
                    free_before(doc, k2) && military_follow(doc, k2))
                    doc_military = 1;
        }
        size_t wi = word_before_index(doc, i);
        bool after_military = wi != NPOS && wi > 0 && doc.number(wi - 1) && len(doc, wi - 1) == 4 &&
                              doc.str(wi - 1)[0] == '0';  // 0800 and 1000
        bool ok;
        if (cue == "at" || cue == "@") ok = true;
        else if (cue == "by" || cue == "until" || cue == "till" || cue == "before" || cue == "after")
            ok = follow >= 2 || doc_military;
        else if (cue == "for") ok = follow == 3 || doc.is(i + 1, "sharp") || doc_military;
        else if (cue == "the") ok = follow == 3;
        else if (cue == "and" || cue == "to" || cue == "between" || cue == "from") ok = after_military || doc_military;
        else ok = (doc.is(i + 1, "hours") || doc.is(i + 1, "hrs")) && !QUANTITY_CUES.contains(cue);
        if (ok && h >= 1) {
            out.tokens(i, last, military_words(doc, i, h, m) + hrs, 66);
            i = last;
        }
    }
}

// Time zone offsets: UTC+9 UTC plus nine, GMT−3 GMT minus three, UTC+5:30 UTC plus five thirty, PST+8.
RV_RULE("datetime", zone_offset) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.word(i) || doc[i].casing != rv::UPPER || len(doc, i) < 3 || len(doc, i) > 4 ||
            !(doc.punct(i + 1, '+') || (punct_any(doc, i + 1) && is_minus_cp(doc[i + 1].cp))) ||
            !zone_word(doc.str(i)) || (doc.joined(i) && !punct_any(doc, i - 1)))
            continue;
        size_t s = i + 1;
        const rv::Token& t = doc[s];
        if (t.kind != rv::Kind::Punct || (t.cp != '+' && !is_minus_cp(t.cp))) continue;
        if (!doc.joined(s + 1) || !doc.number(s + 1)) continue;
        int h = -1, m = 0;
        size_t last = s + 1;
        if (len(doc, s + 1) <= 2) {
            h = num(doc, s + 1, 2);
            if (doc.punct(s + 2, ':') && doc.joined(s + 2) && doc.number(s + 3) && doc.joined(s + 3) &&
                len(doc, s + 3) == 2) {
                m = num(doc, s + 3, 2);
                last = s + 3;
            }
        } else if (len(doc, s + 1) == 4) {
            int v = num(doc, s + 1, 4);
            h = v / 100;
            m = v % 100;
        }
        if (h < 0 || h > 14 || m > 59 || !free_after(doc, last)) continue;
        std::string words = std::string(doc.str(i)) + (t.cp == '+' ? " plus " : " minus ") + hm_words(h, m);
        out.tokens(i, last, words, 80);
        i = last;
    }
}

}  // namespace
