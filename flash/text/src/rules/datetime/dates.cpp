// Dates: month names and abbreviations, day-first dates, numeric dates (m/d, m/d/y, d.m.y, mm/yyyy), ISO dates
// and timestamps, date ranges, weekday and month abbreviations.
#include "rules/datetime/dt.h"

namespace {

using namespace dt;

// A month as spoken: an abbreviation in full, a full name as written (it may be "june" in a chat message).
std::string month_name(const Doc& d, size_t i, int m, bool abbr) {
    return abbr || d[i].casing == rv::UPPER ? std::string(MONTH_NAMES[m]) : std::string(d.str(i));  // MARCH 12
}

// A month word at i with its optional joined period; returns the last token, or NPOS.
size_t month_token(const Doc& d, size_t i, int* m, bool* abbr) {
    bool a = false;
    int mm = month_at(d, i, &a);
    if (!mm) return NPOS;
    if (d.joined(i) && !punct_any(d, i - 1)) return NPOS;
    *m = mm;
    if (abbr) *abbr = a;
    if (a && d.punct(i + 1, '.') && d.joined(i + 1)) return i + 1;
    return i;
}

struct Day {
    size_t last = 0;
    int d = 0;
    bool suffix = false;
};

// A day of the month at Number i: 1-31, one or two digits, optional st / nd / rd / th.
bool day_at(const Doc& d, size_t i, Day* out) {
    if (!d.number(i) || len(d, i) > 2) return false;
    int v = num(d, i, 2);
    if (v < 1 || v > 31) return false;
    size_t suf = ordinal_suffix(d, i);
    out->d = v;
    out->suffix = suf != NPOS;
    out->last = suf != NPOS ? suf : i;
    if (suf == NPOS && d.joined(i + 1) && d.word(i + 1)) return false;
    return true;
}

// ", 2026" / " 2026" after token `last`: the year token, or NPOS.
size_t year_after(const Doc& d, size_t last, int* y) {
    size_t k = last + 1;
    if (d.punct(k, ',') && d.joined(k)) ++k;
    if (k >= d.size() || !d[k].space_before || d[k].newline_before || !year4(d, k) || !free_after(d, k))
        return NPOS;
    if (d.joined(k + 1) && d.word(k + 1)) return NPOS;
    *y = num(d, k, 4);
    return k;
}

const rv::WordSet DAY_LINKS{"to", "through", "thru", "until", "till", "and", "or"};

// Month-first dates: May 4, May 31, 1870, Oct. 28, Apr 21–24, Dec 18–Jan 4, Sept. 2026, Mar–Jun 2025,
// Jun 2025–Jan 2026.
// The first letter of a month name (lower case): a cheap test before the table lookups.
inline bool month_initial(char c) {
    return c == 'j' || c == 'f' || c == 'm' || c == 'a' || c == 's' || c == 'o' || c == 'n' || c == 'd';
}

RV_RULE("datetime", month_date) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.word(i) || doc[i].casing == rv::MIXED || len(doc, i) < 3 || len(doc, i) > 9 ||
            !month_initial(doc.lower[doc[i].start]))
            continue;
        int m = 0;
        bool abbr = false;
        size_t j = month_token(doc, i, &m, &abbr);
        bool lower = false;
        if (j == NPOS) {
            // "june 3" in a chat message: a lower-case full name only with a day after it
            m = month_lower_at(doc, i);
            if (!m || (doc.joined(i) && !punct_any(doc, i - 1))) continue;
            j = i;
            lower = true;
        }
        bool dotted = j > i;
        std::string month = month_name(doc, i, m, abbr);
        size_t n = j + 1;
        Day d1;
        // "June the 30th"
        std::string the;
        if (doc.is(n, "the") && doc[n].space_before && n + 1 < doc.size() && doc[n + 1].space_before &&
            day_at(doc, n + 1, &d1) && d1.suffix) {
            the = " the";
            ++n;
        }
        if (n < doc.size() && (doc[n].space_before || dotted) && !doc[n].newline_before && day_at(doc, n, &d1) &&
            valid_day(m, d1.d)) {
            std::string words = month + the + " " + day_words(d1.d);
            size_t last = d1.last;
            int score = abbr && !dotted ? S_SURE - 2 : S_SURE;
            size_t k = range_dash(doc, last + 1);
            Day d2;
            int m2 = 0;
            bool abbr2 = false;
            if (k != NPOS && day_at(doc, k, &d2) && d2.d > d1.d && valid_day(m, d2.d)) {
                words += " to " + day_words(d2.d);
                last = d2.last;
            } else if (k != NPOS) {
                size_t j2 = month_token(doc, k, &m2, &abbr2);
                if (j2 != NPOS && j2 + 1 < doc.size() && (doc[j2 + 1].space_before || j2 > k) &&
                    day_at(doc, j2 + 1, &d2) && valid_day(m2, d2.d)) {
                    words += " to " + month_name(doc, k, m2, abbr2) + " " + day_words(d2.d);
                    last = d2.last;
                }
            } else if (doc.word(last + 1) && DAY_LINKS.contains(doc.low(last + 1)) && doc[last + 1].space_before &&
                       day_at(doc, last + 2, &d2) && doc[last + 2].space_before && d2.d > d1.d &&
                       valid_day(m, d2.d) && free_after(doc, d2.last) && !month_at(doc, d2.last + 1) &&
                       !(doc.word(d2.last + 1) && !doc.punct(d2.last + 1, ','))) {
                // "May 3 to 5": the second day is an ordinal too; it must not start a phrase of its own
                words += " " + std::string(doc.low(last + 1)) + " " + day_words(d2.d);
                last = d2.last;
            }
            int y = 0;
            size_t yk = year_after(doc, last, &y);
            if (yk != NPOS) {
                words += ", " + year_words(y);
                last = yk;
                score = S_SURE + 2;
            }
            if (!free_after(doc, last)) continue;
            // "May 4:30" is not a date; "Dec. 24: closed" is
            if (doc.joined(last + 1) && doc.punct(last + 1, ':') && doc.number(last + 2) && doc.joined(last + 2))
                continue;
            if (lower && !d1.suffix && yk == NPOS && doc.word(last + 1) && doc[last + 1].space_before &&
                (doc.is(last + 1, "times") || doc.is(last + 1, "people") || doc.is(last + 1, "of")))
                continue;  // "june 3 times": not a date
            out.tokens(i, last, words, lower ? S_SURE - 8 : score);
            i = last;
            continue;
        }
        if (lower || !the.empty()) continue;
        // month and a short year: Nov '24, Sept. ’25
        if (n + 1 < doc.size() && doc[n].space_before && punct_any(doc, n) && apostrophe_cp(doc[n].cp) &&
            doc.number(n + 1) && doc.joined(n + 1) && len(doc, n + 1) == 2 && free_after(doc, n + 1) &&
            !(doc.joined(n + 2) && doc.word(n + 2))) {
            std::string yy = two_digit_year(num(doc, n + 1, 2));
            if (!yy.empty()) {
                out.tokens(i, n + 1, month + " " + yy, S_SURE - 3);
                i = n + 1;
                continue;
            }
        }
        if (n < doc.size() && doc[n].space_before && !doc[n].newline_before && year4(doc, n) &&
            !(doc.joined(n + 1) && doc.word(n + 1))) {
            // month and year, possibly a range of them: Jun 2025–Jan 2026
            std::string words = month + " " + year_words(num(doc, n, 4));
            size_t last = n;
            size_t k = range_dash(doc, n + 1);
            int m2 = 0;
            bool abbr2 = false;
            size_t j2 = k != NPOS ? month_token(doc, k, &m2, &abbr2) : NPOS;
            if (j2 != NPOS && j2 + 1 < doc.size() && doc[j2 + 1].space_before && year4(doc, j2 + 1) &&
                free_after(doc, j2 + 1)) {
                words += " to " + month_name(doc, k, m2, abbr2) + " " + year_words(num(doc, j2 + 1, 4));
                last = j2 + 1;
            }
            if (!free_after(doc, last)) continue;
            out.tokens(i, last, words, S_SURE - 3);
            i = last;
            continue;
        }
        size_t k = range_dash(doc, j + 1);
        int m2 = 0;
        bool abbr2 = false;
        size_t j2 = k != NPOS ? month_token(doc, k, &m2, &abbr2) : NPOS;
        if (j2 != NPOS && m2 != m) {
            // a range of months: Mar–Jun 2025, Oct.–Dec 2026, June–August
            std::string words = month + " to " + month_name(doc, k, m2, abbr2);
            size_t last = j2;
            if (abbr2 && j2 > k && period_ends_sentence(doc, j2)) last = k;
            if (j2 + 1 < doc.size() && doc[j2 + 1].space_before && year4(doc, j2 + 1) && free_after(doc, j2 + 1) &&
                !(doc.joined(j2 + 2) && doc.word(j2 + 2))) {
                words += " " + year_words(num(doc, j2 + 1, 4));
                last = j2 + 1;
            }
            out.tokens(i, last, words, S_SURE - 3);
            i = last;
        }
    }
}

const rv::WordSet HOLIDAY_NOUNS{"barbecue", "bbq", "celebration", "celebrations", "concert", "cookout", "display",
                                "event", "festivities", "fireworks", "holiday", "parade", "parades", "party",
                                "picnic", "sale", "sales", "weekend"};

const rv::WordSet DAY_RANGE_LINKS{"to", "through", "thru", "till", "until"};

// Day-first dates: 4 July 2024, 1 Aug 2022, 12 June, the 3rd of December. A day-first range is read month
// first like the labels do: 10-12 March, from 9 to 11 Sep 2026 (March tenth to twelfth).
RV_RULE("datetime", day_month_date) {
    for (size_t i = 0; i < doc.size(); ++i) {
        Day d1;
        if (!doc.number(i) || !free_before(doc, i) || !day_at(doc, i, &d1)) continue;
        size_t last = d1.last;
        Day d2;
        size_t k = range_dash(doc, last + 1);
        bool range = false;
        std::string link = "to";
        if (k == NPOS && doc.word(last + 1) && doc[last + 1].space_before && doc.number(last + 2) &&
            doc[last + 2].space_before &&
            (DAY_RANGE_LINKS.contains(doc.low(last + 1)) || (doc.is(last + 1, "and") && word_before(doc, i) == "between"))) {
            k = last + 2;
            link = std::string(doc.low(last + 1));
        }
        if (k != NPOS && day_at(doc, k, &d2) && d2.d > d1.d) {
            last = d2.last;
            range = true;
        }
        size_t n = last + 1;
        if (n >= doc.size() || !doc[n].space_before || doc[n].newline_before) continue;
        bool of = false;
        if (doc.is(n, "of") && (range ? d2.suffix : d1.suffix)) {
            of = true;
            ++n;
            if (n >= doc.size() || !doc[n].space_before) continue;
        }
        int m = 0;
        bool abbr = false;
        size_t j = month_token(doc, n, &m, &abbr);
        if (j == NPOS || !valid_day(m, range ? d2.d : d1.d)) continue;
        if (j == n && doc.number(j + 1) && doc[j + 1].space_before && !year4(doc, j + 1))
            continue;  // "5 May 3": the month starts a date of its own
        last = j;
        bool after_the = word_before(doc, i) == "the" && !doc.punct(i - 1, ',');
        std::string words;
        if (range && !of && !after_the) {
            words = month_name(doc, n, m, abbr) + " " + day_words(d1.d) + " " + link + " " + day_words(d2.d);
        } else {
            words = after_the ? "" : "the ";
            words += day_words(d1.d);
            if (range) words += " " + link + " " + day_words(d2.d);
            words += " of " + month_name(doc, n, m, abbr);
        }
        // a range to a second day-first date: 30 June–4 July 2025
        if (!range && !of) {
            size_t k2 = range_dash(doc, j + 1);
            Day e;
            int m2 = 0;
            bool abbr2 = false;
            size_t j2 = NPOS;
            if (k2 != NPOS && day_at(doc, k2, &e) && e.last + 1 < doc.size() && doc[e.last + 1].space_before)
                j2 = month_token(doc, e.last + 1, &m2, &abbr2);
            if (j2 != NPOS && m2 != m && valid_day(m2, e.d)) {
                words += " to the " + day_words(e.d) + " of " + month_name(doc, e.last + 1, m2, abbr2);
                j = last = j2;
                n = e.last + 1;
            }
        }
        int y = 0;
        size_t yk = year_after(doc, j, &y);
        if (yk != NPOS) {
            words += ", " + year_words(y);
            last = yk;
        } else if (j > n && period_ends_sentence(doc, j)) {
            last = n;  // "on 3 Dec." keeps its sentence period
        }
        if (!free_after(doc, last)) continue;
        if (of && !range && yk == NPOS && d1.d == 4 && m == 7 && doc.word(last + 1) &&
            HOLIDAY_NOUNS.contains(doc.low(last + 1)))
            words[words.find("fourth")] = 'F';  // the holiday: the Fourth of July parade
        out.tokens(i, last, words, yk != NPOS ? S_SURE + 2 : (of ? S_SURE : S_SURE - 2));
        i = last;
    }
}

const rv::WordSet DATE_CUES{"on",       "by",       "due",      "since",    "until",    "till",    "til",
                            "from",     "through",  "thru",     "after",    "before",   "dated",   "starting",
                            "ending",   "ends",     "ended",    "eta",      "effective", "expires", "expired",
                            "expiring", "deadline", "delivered", "delivery", "arriving", "arrives", "arrival",
                            "ships",    "shipped",  "opens",    "closes",   "closed",   "opened",  "starts",
                            "started",  "begins",   "began",    "scheduled", "held",    "moved",   "date",
                            "dates",    "monday",   "tuesday",  "wednesday", "thursday", "friday", "saturday",
                            "sunday",   "mon",      "tue",      "tues",     "wed",      "thu",     "thur",
                            "thurs",    "fri",      "sat",      "sun",      "tonight",  "tomorrow", "today",
                            "postponed", "rescheduled", "birthday", "week",  "launch",   "launches", "returns",
                            "return",   "back",     "posted",   "updated",  "signed",   "filed",   "submitted",
                            "received", "paid",     "renewal",  "renews",   "valid",    "issued",  "draw",
                            "drawn",    "appointment", "visit", "pickup",   "dropoff",  "closing", "opening", "available",
                            "avail",    "lands",    "landing",  "departs",  "departing", "leaves",  "leaving",  "est"};
// Words around a short slash that make it a fraction or a score, not a date: scored 7/10, 1/2 cup, 10 1/2.
const rv::WordSet FRACTION_BEFORE{
    "scored",   "score",    "scores",   "rated",     "rating",   "gave",      "reached",  "completed", "got",
    "earned",   "scoring",  "odds",     "about",     "approximately", "nearly", "almost", "only",     "roughly",
    "over",     "under",    "just",     "add",       "added",    "needs",     "need",     "needed",    "use",
    "used",     "had",      "has",      "have",      "spread",   "barely",    "mere",     "half",      "take",
    "another",  "gives",    "give",     "giving",    "awarded",  "graded",    "ranked",   "a",         "an",
    "with",     "at",       "retry",    "retries",   "attempt",  "attempts",  "step",     "status",    "progress",
    "processed", "reporting", "averaged", "remaining", "failed", "passed",    "showed",   "shows",     "restart",
    "restarts", "exactly",  "below",    "above",     "currently", "ready",    "whopping", "mark",      "grade",
    "verdict",  "voted",    "won",      "lost",      "record",   "went",      "finished", "solved",    "answered"};
const rv::WordSet FRACTION_AFTER{
    "cup",      "cups",     "teaspoon", "teaspoons", "tsp",      "tablespoon", "tablespoons", "tbsp",   "inch",
    "inches",   "mile",     "miles",    "pound",     "pounds",   "lb",        "lbs",      "ounce",     "ounces",
    "oz",       "odds",     "steps",    "tasks",     "replicas", "nodes",     "stars",    "points",    "marks",
    "done",     "complete", "completed", "correct",  "shoe",     "lemon",     "lime",     "orders",    "cases",
    "of",       "tablet",   "tablets",  "pill",      "pills",    "slice",     "slices",   "completion", "majority",
    "increase", "decrease", "reduction", "discount", "share",    "split",     "chance",   "targets",   "logins",
    "attempts", "items",    "questions", "answers",  "votes",    "pages",     "chapters", "runs",      "tests",
    "checks",   "passed",   "failed",   "ready",     "healthy",  "available", "pods",     "servers",   "players",
    "games",    "matches",  "shots",    "workers",   "jobs",     "rating",    "overall",  "stake",     "vote",
    "speed",    "time",     "size",     "scale",     "strength", "dose",      "turn",     "inning",    "innings"};
// Nouns before "of" that make the slash after it a date: the vote of 3/9, the week of 3/9.
const rv::WordSet EVENT_OF{"vote",    "week",    "meeting", "session", "election", "letter", "order",   "memo",
                           "call",    "email",   "message", "decision", "ruling",  "notice", "release", "report",
                           "update",  "statement", "invoice", "event", "night",    "morning", "evening", "weekend",
                           "minutes", "hearing", "shipment", "batch",  "storm",    "outage", "incident", "close"};
// Nouns before "is" / "was" that make the slash after it a date: the deadline is 3/4.
const rv::WordSet DATE_NOUNS{"date",     "deadline", "appointment", "birthday", "delivery", "arrival",  "eta",
                             "due",      "launch",   "release",     "start",    "end",      "opening",  "closing",
                             "visit",    "checkup",  "anniversary", "renewal",  "expiry",   "expiration", "cutoff",
                             "meeting",  "pickup",   "dropoff",     "surgery",  "exam",     "interview", "hearing"};
const rv::WordSet MOVED{"moved", "pushed", "postponed", "rescheduled", "delayed", "changed", "shifted", "brought",
                        "scheduled", "planned", "booked", "set", "slated", "due", "deadline", "pencilled",
                        "penciled", "extended", "advanced"};
const rv::WordSet DATE_LINKS{"and", "or", "to", "through", "thru", "until", "till", "between"};

// A slash date "a/b" whose month and day are valid, read month first unless a is above twelve.
bool slash_md(int a, int b, int* month, int* day) {
    if (a >= 1 && a <= 12 && valid_day(a, b)) {
        *month = a;
        *day = b;
        return true;
    }
    if (a >= 13 && a <= 31 && b >= 1 && b <= 12 && valid_day(b, a)) {
        *month = b;
        *day = a;
        return true;
    }
    return false;
}

std::string date_words(const Doc& doc, size_t i, int a, int month, int day, const std::string& year) {
    std::string w;
    if (a == month) {
        w = std::string(MONTH_NAMES[month]) + " " + day_words(day);
    } else {
        w = word_before(doc, i) == "the" ? "" : "the ";
        w += day_words(day) + " of " + MONTH_NAMES[month];
    }
    if (!year.empty()) w += ", " + year;
    return w;
}

// A short slash "a/b" at token i (a 1-2 digit number, a joined '/', a 1-2 digit number) that is a valid date.
bool short_slash(const Doc& doc, size_t i) {
    int month, day;
    return doc.number(i) && len(doc, i) <= 2 && doc.punct(i + 1, '/') && doc.joined(i + 1) && doc.joined(i + 2) &&
           doc.number(i + 2) && len(doc, i + 2) <= 2 && !(doc.punct(i + 3, '/') && doc.joined(i + 3)) &&
           slash_md(num(doc, i, 2), num(doc, i + 2, 2), &month, &day);
}

// What follows a short slash ending at token `last` makes it a date: a time after it ("4/9 at 1:30",
// "6/12 8:00 pm", "12/24 from 8:00 AM").
bool slash_after_cue(const Doc& doc, size_t last) {
    size_t n = last + 1;
    if (n >= doc.size() || doc[n].newline_before) return false;
    if (doc.punct(n, ',') && doc.joined(n)) ++n;
    size_t t = n;
    if (doc.is(n, "at") || doc.punct(n, '@') || doc.is(n, "from") || doc.is(n, "starting") || doc.is(n, "between") ||
        doc.is(n, "by") || doc.is(n, "until") || doc.is(n, "before") || doc.is(n, "after"))
        t = n + 1;
    if (!doc.number(t) || len(doc, t) > 2) return false;
    if (doc.punct(t + 1, ':') && doc.joined(t + 1) && doc.number(t + 2)) return true;  // a clock time
    size_t al;
    if (t + 1 < doc.size() && ampm_at(doc, t + 1, &al)) return true;                  // 8 pm
    return t > n && (doc.is(n, "at") || doc.punct(n, '@'));                             // at 9
}

// A short slash at the start of a line that continues like a dated entry: a capitalized word, a time, or
// ':', '|', ',' or a dash ("6/11 Ferncliff Tavern", "12/5: 11:00"), not "1/2 cup flour".
bool date_line_shape(const Doc& doc, size_t i) {
    size_t n = i + 3;
    if (n >= doc.size() || doc[n].newline_before) return true;
    const rv::Token& t = doc[n];
    if (t.kind == rv::Kind::Number) return true;
    if (t.kind == rv::Kind::Punct) return t.cp == ':' || t.cp == '|' || t.cp == ',' || is_dash_cp(t.cp) || t.cp == 0x2014;
    return t.casing == rv::TITLE || t.casing == rv::UPPER;
}

// Short slashes at the start of two or more lines: a list of dates ("6/11 Ferncliff Tavern, 9:20 pm").
bool date_lines(const Doc& doc) {
    int lines = 0;
    for (size_t k = 0; k + 2 < doc.size(); ++k)
        if ((k == 0 || doc[k].newline_before) && short_slash(doc, k) && date_line_shape(doc, k) && ++lines >= 2)
            return true;
    return false;
}

// The cue before a short slash date: on 5/9, due 3/4, Sat. 11/21, Date: 5/9, as of 5/9, the vote of 3/9, the
// deadline is 3/4, and the second date of "on 5/9 and 5/12".
bool slash_date_cue(const Doc& doc, size_t i, int depth = 0) {
    std::string_view w = word_before(doc, i);
    if (DATE_CUES.contains(w)) return true;
    size_t wi = word_before_index(doc, i);
    std::string_view w2 = wi != NPOS && wi > 0 && doc.word(wi - 1) ? doc.low(wi - 1) : std::string_view();
    if (w == "of" && (w2 == "as" || EVENT_OF.contains(w2))) return true;
    if ((w == "is" || w == "was" || w == "be" || w == "s") && DATE_NOUNS.contains(w2)) return true;
    if (w == "live" && (w2 == "goes" || w2 == "go" || w2 == "went" || w2 == "going" || w2 == "gone")) return true;
    // moved to 3/26, scheduled for 10/21
    if ((w == "to" || w == "for") && MOVED.contains(w2)) return true;
    // the second date of a list: "on 5/9 and 5/12" (the first one must be a date itself)
    if (DATE_LINKS.contains(w) && wi >= 3 && doc.number(wi - 1) && doc.punct(wi - 2, '/') && doc.number(wi - 3) &&
        depth == 0 && short_slash(doc, wi - 3) && (slash_date_cue(doc, wi - 3, 1) || slash_after_cue(doc, wi - 1)))
        return true;
    if (i >= 2 && doc.punct(i - 1, ':') && doc.word(i - 2)) {
        std::string_view l = doc.low(i - 2);
        return l == "date" || l == "due" || l == "deadline" || l == "delivery" || l == "arrival" || l == "eta" ||
               l == "when" || l == "day" || l == "start" || l == "end" || l == "expires" || l == "dates";
    }
    if (i >= 1 && range_dash(doc, i - 1) == i && i >= 4 && doc.number(i - 2) && doc.punct(i - 3, '/')) return true;
    return false;
}

// Words after a short slash that make it a date even after "a" or "the": "a 1/4 deadline".
const rv::WordSet DATE_NOUN_AFTER{"appointment", "birthday", "call", "deadline", "delivery", "event", "flight",
                                  "launch", "meeting", "release", "session", "visit", "red"};

// A short slash between numbers that reads as a fraction or a score: scored 7/10, gave the design 4/5,
// 1/2 cup, 10 1/2, 9/10 on the dry runs, a 3/4 majority, (3/10), odds of 3/1.
bool fraction_context(const Doc& doc, size_t i) {
    // "a 1/4 deadline", "the 2/27 red-eye": a date before a word for an event
    if (doc.word(i + 3) && doc[i + 3].space_before && DATE_NOUN_AFTER.contains(doc.low(i + 3))) return false;
    // a mixed number "10 1/2": the whole part on the same line, not the end of a clock time
    if (i > 0 && doc.number(i - 1) && !doc[i].newline_before && !(i >= 2 && doc.punct(i - 2, ':'))) return true;
    size_t w1 = word_before_index(doc, i);
    if (w1 != NPOS && FRACTION_BEFORE.contains(doc.low(w1))) return true;
    if (w1 != NPOS && w1 > 0) {
        size_t w2 = word_before_index(doc, w1);
        if (w2 != NPOS && FRACTION_BEFORE.contains(doc.low(w2))) return true;
    }
    if (doc.word(i + 3) && (FRACTION_AFTER.contains(doc.low(i + 3)) || doc.is(i + 3, "for") || doc.is(i + 3, "on")))
        return true;
    if (doc.punct(i - 1, '(') && doc.joined(i) && doc.punct(i + 3, ')') && doc.joined(i + 3)) return true;  // (3/10)
    if (doc.str(i + 2) == "1" && !(doc.punct(i + 3, ':') && doc.joined(i + 3))) {  // 3/1: odds
        std::string_view w = word_before(doc, i);
        if (w == "at" || w == "were" || w == "was" || w == "are" || w == "is" || w == "of" || w == "odds" ||
            w == "the" || w == "a" || w == "priced" || w == "paying" || w == "pays")
            return !(w == "the" && doc.word(i + 3) && !doc.is(i + 3, "odds") && !doc.is(i + 3, "shot") &&
                     !doc.is(i + 3, "favorite") && !doc.is(i + 3, "favourite") && !doc.is(i + 3, "underdog"));
        if (doc.word(i + 3) && (doc.is(i + 3, "odds") || doc.is(i + 3, "against") || doc.is(i + 3, "ratio") ||
                                doc.is(i + 3, "shot") || doc.is(i + 3, "favorite") || doc.is(i + 3, "favourite")))
            return true;
    }
    if (doc.punct(i - 1, '|') && doc.punct(i + 3, '|')) return true;                          // | 8/10 |
    return false;
}

// A score or fraction after another one with the same denominator that is not a date: "7/10 at the start,
// 8/10 by the end", "a 3/4 majority and 3/4 support".
bool same_denominator(const Doc& doc, size_t i) {
    std::string_view b = doc.str(i + 2);
    int bv = num(doc, i + 2, 2);
    for (size_t k = i >= 14 ? i - 14 : 0; k + 2 < i; ++k)
        if (doc.number(k) && doc.punct(k + 1, '/') && doc.joined(k + 1) && doc.joined(k + 2) && doc.str(k + 2) == b &&
            (k == 0 || !doc.punct(k - 1, '/')) && !doc.punct(k + 3, '/') && !slash_date_cue(doc, k, 1) &&
            (bv >= 5 || fraction_context(doc, k)))
            return true;
    return false;
}

// Numeric dates: 09/14/2025, 28/10/2026, 14.03.2024, 14-03-2024, on 5/9, 04/2026.
RV_RULE("datetime", numeric_date) {
    int doc_dates = -1;  // the text is a list of dated lines: computed when needed
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.number(i) || len(doc, i) > 2 || !free_before(doc, i)) continue;
        const rv::Token& sep = doc[i + 1];
        if (sep.kind != rv::Kind::Punct || !doc.joined(i + 1) || !doc.joined(i + 2) || !doc.number(i + 2)) continue;
        uint32_t s = sep.cp;
        if (s != '/' && s != '.' && s != '-') continue;
        int a = num(doc, i, 2);
        bool three = doc.punct(i + 3, s) && doc.joined(i + 3) && doc.number(i + 4) && doc.joined(i + 4);
        if (three) {
            if (len(doc, i + 2) > 2) continue;
            int b = num(doc, i + 2, 2);
            size_t yl = len(doc, i + 4);
            if (!free_after(doc, i + 4) || (yl != 4 && yl != 2) || (yl == 2 && s != '/')) continue;
            int y = num(doc, i + 4, 4);
            if (yl == 4 && (y < 1000 || y > 2999)) continue;
            int month = 0, day = 0;
            if (!slash_md(a, b, &month, &day)) continue;
            std::string year = yl == 4 ? year_words(y) : two_digit_year(y);
            if (year.empty()) continue;
            int score = s == '-' ? 65 : (yl == 4 ? 82 : 60);
            out.tokens(i, i + 4, date_words(doc, i, a, month, day, year), score);
            i += 4;
            continue;
        }
        if (s != '/' || !free_after(doc, i + 2)) continue;
        if (doc.joined(i + 3) && doc.word(i + 3)) continue;
        if (len(doc, i + 2) == 4) {
            int y = num(doc, i + 2, 4);
            if (a >= 1 && a <= 12 && y >= 1900 && y <= 2099) {
                out.tokens(i, i + 2, std::string(MONTH_NAMES[a]) + " " + year_words(y), 75);
                i += 2;
            }
            continue;
        }
        if (len(doc, i + 2) > 2) continue;
        int b = num(doc, i + 2, 2), month = 0, day = 0;
        if (!slash_md(a, b, &month, &day)) continue;
        bool cue = slash_date_cue(doc, i) || slash_after_cue(doc, i + 2);
        bool line = false;
        if (i == 0 || doc[i].newline_before) {
            // the start of a line: "12/5: 11:00–1:00", a list of dated lines
            if (doc_dates < 0) doc_dates = date_lines(doc);
            line = date_line_shape(doc, i) &&
                   (doc_dates || (doc.punct(i + 3, ':') && doc.joined(i + 3)) || doc.punct(i + 3, '|'));
            cue = cue || line;
        }
        // a zero-padded month (09/05) is never a fraction
        bool padded = len(doc, i) == 2 && doc.str(i)[0] == '0' && a == month;
        // a fraction, odds or a score: numbers_context
        if (!cue && !padded && fraction_context(doc, i)) continue;
        if (!cue && !padded && same_denominator(doc, i)) {
            i += 2;
            continue;
        }
        // a date range "5/9–5/12"
        size_t k = range_dash(doc, i + 3);
        std::string words = date_words(doc, i, a, month, day, "");
        size_t last = i + 2;
        int a2 = num(doc, k == NPOS ? 0 : k, 2), month2 = 0, day2 = 0;
        if (k != NPOS && a2 > 0 && doc.punct(k + 1, '/') && doc.joined(k + 1) && doc.joined(k + 2) &&
            len(doc, k + 2) <= 2 && free_after(doc, k + 2) && slash_md(a2, num(doc, k + 2, 2), &month2, &day2) &&
            (a2 == month2) == (a == month)) {
            words += " to " + date_words(doc, k, a2, month2, day2, "");
            last = k + 2;
        }
        out.tokens(i, last, words, line ? S_CUE + 4 : cue ? S_CUE : (padded ? 60 : S_WEAK));
        i = last;
    }
}

// ISO dates and timestamps: 2026-10-28, 2026-10-28T10:50:57.871740+00:00, 2026-10-28 10:50 UTC.
RV_RULE("datetime", iso_date) {
    for (size_t i = 0; i + 4 < doc.size(); ++i) {
        if (!year4(doc, i) || !free_before(doc, i)) continue;
        uint32_t s = doc[i + 1].cp;
        if (!(doc.punct(i + 1, '-') || doc.punct(i + 1, '/')) || !doc.joined(i + 1)) continue;
        if (!doc.number(i + 2) || !doc.joined(i + 2) || len(doc, i + 2) != 2) continue;
        if (!doc.punct(i + 3, s) || !doc.joined(i + 3) || !doc.number(i + 4) || !doc.joined(i + 4) ||
            len(doc, i + 4) != 2)
            continue;
        int y = num(doc, i, 4), m = num(doc, i + 2, 2), d = num(doc, i + 4, 2);
        if (!valid_day(m, d)) continue;
        std::string words = std::string(MONTH_NAMES[m]) + " " + day_words(d) + ", " + year_words(y);
        size_t last = i + 4;
        // the time: joined 'T' or after a space
        size_t t = NPOS;
        if (doc.word(i + 5) && doc.joined(i + 5) && doc.str(i + 5) == "T" && doc.joined(i + 6)) t = i + 6;
        else if (i + 5 < doc.size() && doc[i + 5].space_before && !doc[i + 5].newline_before) t = i + 5;
        Clock c;
        if (t != NPOS) c = parse_clock(doc, t);
        if (c.ok && c.h <= 23) {
            std::string zone;
            size_t cl = c.last;
            if (c.zulu) {
                zone = " UTC";
            } else if (cl + 2 < doc.size() && doc.joined(cl + 1) && doc[cl + 1].kind == rv::Kind::Punct &&
                       (doc[cl + 1].cp == '+' || is_minus_cp(doc[cl + 1].cp)) && doc.number(cl + 2) &&
                       doc.joined(cl + 2)) {
                int oh = -1, om = 0;
                size_t ol = cl + 2;
                if (len(doc, cl + 2) == 2) {
                    oh = num(doc, cl + 2, 2);
                    if (doc.punct(cl + 3, ':') && doc.joined(cl + 3) && doc.number(cl + 4) && doc.joined(cl + 4) &&
                        len(doc, cl + 4) == 2) {
                        om = num(doc, cl + 4, 2);
                        ol = cl + 4;
                    }
                } else if (len(doc, cl + 2) == 4) {
                    int v = num(doc, cl + 2, 4);
                    oh = v / 100;
                    om = v % 100;
                }
                if (oh >= 0 && oh <= 14 && om < 60) {
                    cl = ol;
                    if (oh == 0 && om == 0) zone = " UTC";
                    else zone = std::string(doc[c.last + 1].cp == '+' ? " plus " : " minus ") + hm_words(oh, om);
                }
            }
            if (free_after(doc, cl)) {
                words += word_before(doc, i) == "at" ? ", " : ", at ";
                bool padded = len(doc, t) == 2 && doc.str(t)[0] == '0';
                words += (padded && c.m == 0 && c.h >= 1 ? rv::cardinal(c.h) + " hundred" : clock_words(c.h, c.m)) + zone;
                last = cl;
            }
        }
        if (!free_after(doc, last)) continue;
        out.tokens(i, last, words, S_CONSTRUCT);
        i = last;
    }
}

const rv::WordSet NOT_BEFORE_WEEKDAY{"the", "a", "an", "our", "my", "his", "her", "their", "its", "your"};

// Weekday abbreviations with a period: Mon. Monday, Thurs. Thursday; ranges Mon.–Fri. Monday to Friday.
// Full weekday ranges: Monday–Friday.
RV_RULE("datetime", weekday) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.word(i) || doc[i].casing == rv::LOWER) continue;
        const char* full = doc.punct(i + 1, '.') ? weekday_abbrev_at(doc, i) : nullptr;
        // "the Sun.", "our Sat." are other words; "this Thu." is a day
        std::string_view low = full ? doc.low(i) : std::string_view();
        bool word_like = low == "sun" || low == "sat" || low == "wed";
        if (full && doc.joined(i + 1) && !(word_like && NOT_BEFORE_WEEKDAY.contains(word_before(doc, i))) &&
            !(doc.joined(i) && !punct_any(doc, i - 1))) {
            std::string words = full;
            size_t last = abbrev_end(doc, i);
            size_t k = range_dash(doc, i + 2);
            const char* full2 = k != NPOS ? weekday_abbrev_at(doc, k) : nullptr;
            if (full2) {
                words += std::string(" to ") + full2;
                last = k;
                if (doc.punct(k + 1, '.') && doc.joined(k + 1)) last = abbrev_end(doc, k);
            } else if (k != NPOS && weekday_full_at(doc, k)) {
                words += " to " + std::string(doc.str(k));
                last = k;
            }
            out.tokens(i, last, words, 75);
            i = last;
            continue;
        }
        // a range of abbreviations without periods: Mon–Fri, Tue-Thu
        const char* a1 = dash(doc, i + 1) ? weekday_abbrev_at(doc, i) : nullptr;
        if (a1 && !(doc.joined(i) && !punct_any(doc, i - 1))) {
            size_t k = dash(doc, i + 1) && doc.joined(i + 1) ? i + 2 : NPOS;
            const char* a2 = k != NPOS && doc.joined(k) ? weekday_abbrev_at(doc, k) : nullptr;
            if (a2 && a2 != a1 && !(doc.joined(k + 1) && doc.word(k + 1))) {
                size_t last = doc.punct(k + 1, '.') && doc.joined(k + 1) ? abbrev_end(doc, k) : k;  // Mon–Fri.
                if (last > k || free_after(doc, k)) {
                    out.tokens(i, last, std::string(a1) + " to " + a2, 72);
                    i = last;
                    continue;
                }
            }
        }
        if (weekday_full_at(doc, i)) {
            size_t k = range_dash(doc, i + 1);
            if (k != NPOS && (weekday_full_at(doc, k) || weekday_abbrev_at(doc, k))) {
                const char* full2 = weekday_abbrev_at(doc, k);
                size_t last = k;
                std::string second = full2 ? full2 : std::string(doc.str(k));
                if (full2 && doc.punct(k + 1, '.') && doc.joined(k + 1)) last = abbrev_end(doc, k);
                else if (full2) continue;
                out.tokens(i, last, std::string(doc.str(i)) + " to " + second, 65);
                i = last;
            }
        }
    }
}

// Month abbreviations with a period outside a date: the Aug. conference, July, Aug., and Sept. In capitals
// they are initialisms: "a high APR."
RV_RULE("datetime", month_abbrev) {
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        if (!doc.punct(i + 1, '.') || !doc.joined(i + 1) || doc[i].casing == rv::UPPER) continue;
        bool abbr = false;
        int m = month_at(doc, i, &abbr);
        if (!m || !abbr) continue;
        if (doc.joined(i) && !punct_any(doc, i - 1)) continue;
        out.tokens(i, abbrev_end(doc, i), MONTH_NAMES[m], 72);
        ++i;
    }
}

}  // namespace
