// Durations: clock-style durations (2:07:19, took 4:17, at 03:19 in the recording), compact durations and
// time units (1h 30m, 45s, 3d, 36 hrs, 50 min, 2 wks), 24/7.
#include "rules/datetime/dt.h"

namespace {

using namespace dt;

const rv::WordSet MEDIA{"recording", "audio",   "video",    "clip",      "file",    "transcript", "replay",
                        "track",     "footage", "podcast",  "episode",   "stream",  "tape",       "interview",
                        "film",      "movie",   "reel",     "broadcast", "livestream", "talk",    "lecture",
                        "webinar",   "song",    "album",    "keynote",   "segment", "show",       "session",
                        "cut",       "edit",    "playback", "trailer",   "documentary", "mix",    "set",
                        "speech",    "presentation", "demo", "tutorial", "vod",     "timeline",   "take",
                        "footage",   "excerpt",  "scene",    "chapter",   "marker"};
// Recordings, unambiguous: positions in them are durations.
const rv::WordSet RECORDINGS{"recording", "recordings", "video", "audio", "clip", "clips", "footage", "tape",
                             "episode", "podcast", "film", "movie", "track", "transcript", "broadcast", "livestream",
                             "lecture", "interview", "webinar", "trailer", "documentary", "reel", "take",
                             "playback", "vod", "soundtrack", "album", "song"};
// Other nouns whose "at 04:38" is a position: the file at 04:38, the cut at 12:10. Not events that are
// scheduled (a keynote at 09:30, the session at 14:00).
const rv::WordSet AT_MEDIA{"file", "replay", "cut", "edit", "mix", "timeline", "excerpt", "scene", "chapter",
                           "marker", "stream", "markers"};
const rv::WordSet ELAPSED_FROM{"ignition", "launch", "liftoff", "lift-off", "takeoff", "start", "kickoff",
                               "whistle", "tip-off", "gun", "beginning", "onset", "injection", "dosing", "boot",
                               "startup", "reset", "restart", "warm", "warmup", "intro", "opening", "trigger"};
const rv::WordSet FOR_VERBS{"ran", "run", "runs", "running", "lasted", "lasts", "last", "lasting", "played",
                            "plays", "recorded", "went", "continued", "continues", "stayed", "waited", "wait",
                            "held", "streamed", "talked", "spoke", "trained", "worked", "paused", "idled", "slept"};
const rv::WordSet OF_NOUNS{"runtime", "length", "duration", "total", "time", "delay", "gap", "lag", "wait",
                           "pause", "outage", "downtime", "uptime", "average", "mean", "median", "span", "window",
                           "runtimes", "lengths", "durations", "totals", "delays", "gaps", "lags", "waits", "pauses",
                           "outages", "breaks", "break", "stoppage", "stoppages", "section", "sections", "part",
                           "portion", "stretch", "times", "splits", "laps"};
const rv::WordSet CLOCK_FOR{"scheduled", "planned", "booked", "slated", "reserved", "moved", "rescheduled",
                            "postponed", "delayed", "pushed", "arranged", "called", "fixed", "organized",
                            "organised", "confirmed", "requested", "set", "timed", "penciled", "pencilled",
                            "announced", "listed", "posted", "advertised", "expected", "needed", "wanted"};
const rv::WordSet BE_NOUNS{"lap", "split", "pace", "record", "best", "finish", "intro", "outro", "trailer",
                           "longest", "shortest", "total", "runtime", "length", "duration", "clip", "cut"};
const rv::WordSet SEEK_WORDS{"ahead", "skip", "jump", "jumps", "forward", "rewind", "back", "scrub", "seek", "fast-forward",
                             "advance", "cue"};
const rv::WordSet BE_WORDS{"is", "was", "are", "were", "be", "being", "been"};
// Verbs before "in" that make it a phrasal verb, not a duration: gets in around 12:05, checked in at 3:00.
const rv::WordSet IN_VERBS{"get",  "gets",  "got",  "getting", "come",    "comes",  "came",   "coming", "pull",
                           "pulls", "pulled", "check", "checks", "checked", "clock", "clocks", "clocked", "fly",
                           "flies", "flew", "roll", "rolls", "rolled", "log",    "logs",   "logged", "sign",
                           "signs", "signed", "drop", "drops", "dropped", "move", "moves",  "moved",  "lands",
                           "land",  "landed", "punch", "punched", "call",   "calls",  "called", "tune",   "tuned"};
// Verbs before "back to" that move a time: pushed back to 4:30.
const rv::WordSet BACK_VERBS{"pushed", "push", "moved", "move", "set", "put", "pulled", "bumped", "shifted", "changed",
                             "scaled", "dialed", "turned", "brought", "come", "came", "went", "go"};
// Verbs before "the intro to 02:30": an edit that sets a length.
const rv::WordSet CUT_VERBS{"cut", "cuts", "trimmed", "trim", "trims", "shortened", "shorten", "reduced", "edited",
                            "extended", "extend", "stretched", "capped", "cap", "kept", "keep", "limited"};

// The word that governs a duration at token i, walking back over a list of durations ("took 2:44, 5:09
// and 1:58", "lasted 4:05 and the second 11:30").
bool duration_context(const Doc& d, size_t i) {
    size_t j = i;
    for (int steps = 0; j > 0 && steps < 24; ++steps) {
        const rv::Token& p = d[j - 1];
        bool list_part = p.kind == rv::Kind::Number || (p.kind == rv::Kind::Punct && (p.cp == ':' || p.cp == ',')) ||
                         (p.kind == rv::Kind::Word && (d.low(j - 1) == "and" || d.low(j - 1) == "or"));
        // the list has a duration with hours in it: "1:10:05 and 2:40"
        if (p.kind == rv::Kind::Number && j >= 5 && j - 1 != i && d.punct(j - 2, ':') && d.number(j - 3) &&
            d.punct(j - 4, ':') && d.number(j - 5) && d.joined(j - 1) && !(j >= 6 && d.joined(j - 5) && d.word(j - 6)))
            return true;
        // "and the second", "and the final lap": up to two lower-case words after "the"
        if (!list_part && p.kind == rv::Kind::Word) {
            size_t t = NPOS;
            if (j >= 4 && d.is(j - 2, "the")) t = j - 2;
            else if (j >= 5 && d.is(j - 3, "the") && d.word(j - 2) && d[j - 2].casing == rv::LOWER) t = j - 3;
            if (t != NPOS && d[j - 1].casing == rv::LOWER && (d.is(t - 1, "and") || d.punct(t - 1, ',')) &&
                d.number(t - 2) && !time_cue(d.low(j - 1)) && !(t + 1 < j - 1 && time_cue(d.low(t + 1)))) {
                j = t;
                continue;
            }
        }
        if (!list_part) break;
        --j;
    }
    if (j == 0 || !d.word(j - 1)) return false;
    std::string_view w = d.low(j - 1);
    size_t b = j - 1;
    std::string_view w2 = b > 0 && d.word(b - 1) ? d.low(b - 1) : std::string_view();
    if (w == "in") {
        if (b > 0 && d.joined(b) && dash(d, b - 1)) return false;  // Check-in 3:00, log-in
        if (IN_VERBS.contains(w2)) return false;                    // gets in 12:05
    }
    if (duration_cue(w)) return true;
    std::string_view w3 = b > 1 && d.word(b - 1) && d.word(b - 2) ? d.low(b - 2) : std::string_view();
    if (w == "for")  // ran for 3:05:27, burned for 2:41; but scheduled for 4:30
        return FOR_VERBS.contains(w2) ||
               (w2.size() > 4 && w2.substr(w2.size() - 2) == "ed" && !CLOCK_FOR.contains(w2));
    if (w == "of") return OF_NOUNS.contains(w2) || MEDIA.contains(w2);
    if (BE_WORDS.contains(w))
        return (OF_NOUNS.contains(w2) && w2 != "time" && w2 != "times") || MEDIA.contains(w2) || BE_NOUNS.contains(w2);
    if (w == "to") {
        if (w2 == "back" && BACK_VERBS.contains(w3)) return false;  // pushed back to 4:30
        if ((MEDIA.contains(w2) || BE_NOUNS.contains(w2)) && b > 1) {
            // cut the intro to 02:30
            size_t v = b - 1;
            if (v > 0 && d.is(v - 1, "the")) --v;
            if (v > 0 && d.word(v - 1) && CUT_VERBS.contains(d.low(v - 1))) return true;
        }
        return SEEK_WORDS.contains(w2);
    }
    if (w == "at") return RECORDINGS.contains(w2) || AT_MEDIA.contains(w2);  // the recording at 04:38
    if (w == "about" || w == "around" || w == "approximately" || w == "roughly" || w == "nearly" ||
        w == "almost" || w == "just" || w == "only" || w == "another" || w == "over" || w == "between") {
        if (w2 == "in" && IN_VERBS.contains(w3)) return false;  // gets in around 12:05
        return duration_cue(w2) || w2 == "for";
    }
    return false;
}

// Words after a clock that make it a time of day: 17:41 on 2025-07-07, 22:40 CET, 09:15 and 17:40 each day.
// With seconds (2:07:19) only a zone or AM / PM does.
bool time_after(const Doc& d, size_t last, bool seconds) {
    size_t n = last + 1;
    for (int steps = 0; n < d.size() && steps < 10; ++steps) {  // over the rest of a list of times
        const rv::Token& t = d[n];
        bool list_part = t.kind == rv::Kind::Number || (t.kind == rv::Kind::Punct && (t.cp == ':' || t.cp == ',')) ||
                         (t.kind == rv::Kind::Word && (d.low(n) == "and" || d.low(n) == "or" || d.low(n) == "to"));
        if (!list_part || t.newline_before) break;
        ++n;
    }
    if (!d.word(n) || d[n].newline_before) return false;
    if (zone_word(d.str(n))) return true;
    size_t al;
    if (ampm_at(d, n, &al)) return true;
    if (seconds) return false;
    std::string_view w = d.low(n);
    if (w == "daily" || w == "nightly" || w == "today" || w == "tonight" || w == "tomorrow" || w == "local" ||
        w == "sharp" || w == "yesterday")
        return true;
    // that morning, this evening, in the afternoon
    size_t p = (w == "in" && d.is(n + 1, "the")) ? n + 2 : n + 1;
    if ((w == "that" || w == "this" || w == "in" || w == "the" || w == "tomorrow" || w == "yesterday") &&
        d.word(p) && (d.is(p, "morning") || d.is(p, "afternoon") || d.is(p, "evening") || d.is(p, "night")))
        return true;
    if ((w == "each" || w == "every") && d.word(n + 1)) {
        std::string_view x = d.low(n + 1);
        return x == "day" || x == "night" || x == "morning" || x == "evening" || x == "afternoon" || x == "week" ||
               x == "weekday" || x == "monday" || x == "tuesday" || x == "wednesday" || x == "thursday" ||
               x == "friday" || x == "saturday" || x == "sunday";
    }
    if (w == "on") {  // 17:41 on 2025-07-07, on Monday, on the final day
        size_t k = n + 1;
        if (k < d.size() && (d.number(k) || month_at(d, k) || weekday_full_at(d, k) || weekday_abbrev_at(d, k)))
            return true;
        if (d.is(k, "weekdays") || d.is(k, "weekends") || d.is(k, "workdays") || d.is(k, "mondays") ||
            d.is(k, "fridays") || d.is(k, "sundays") || d.is(k, "saturdays"))
            return true;
        if (d.is(k, "the") && (d.is(k + 1, "final") || d.is(k + 1, "last") || d.is(k + 1, "first")) &&
            d.is(k + 2, "day"))
            return true;
    }
    return false;
}

// The sentence before token i has a duration with hours, minutes and seconds (2:41:19) and a recording:
// "The recording runs 2:41:19, and the public comment starts at 07:45." A zero-padded position (01:14) needs
// only the duration: "The run lasted 5:20:30, and the checkpoint was saved at 01:14."
bool media_sentence(const Doc& d, size_t i) {
    bool padded = len(d, i) == 2 && d.str(i)[0] == '0';
    bool hms = false, media = false;
    for (size_t k = i; k-- > 0;) {
        if (d[k + 1].newline_before) break;
        const rv::Token& t = d[k];
        if (t.kind == rv::Kind::Punct && (t.cp == '.' || t.cp == '!' || t.cp == '?') && d[k + 1].space_before) break;
        if (d.word(k) && RECORDINGS.contains(d.low(k))) media = true;
        if (d.number(k) && k + 4 < i && d.punct(k + 1, ':') && d.joined(k + 1) && d.number(k + 2) &&
            d.punct(k + 3, ':') && d.joined(k + 3) && d.number(k + 4) && !(k > 0 && d.joined(k) && d.word(k - 1)))
            hms = true;
        // an earlier duration read by its cue: "The episode runs 52:10, and the interview starts at 03:15."
        if (!hms && d.number(k) && k + 2 < i && d.punct(k + 1, ':') && d.joined(k + 1) && d.number(k + 2) &&
            len(d, k + 2) == 2 && !(k > 0 && (d.punct(k - 1, ':') || (d.joined(k) && d.word(k - 1)))) && k > 0 && d.word(k - 1) &&
            duration_cue(d.low(k - 1)))
            hms = true;
        if (i - k > 60) break;
    }
    return hms && (media || padded);
}

// Words after a duration that make it one: "long", "in the recording", "into the video", "mark", "remaining".
bool duration_after(const Doc& d, size_t last) {
    size_t n = last + 1;
    // over the rest of a list: "at 06:21 and 18:37 in the cut"; a list with 1:07:32 in it is durations
    bool hms = false;
    for (int steps = 0; n < d.size() && steps < 12; ++steps) {
        const rv::Token& t = d[n];
        bool list_part = t.kind == rv::Kind::Number || (t.kind == rv::Kind::Punct && (t.cp == ':' || t.cp == ',')) ||
                         (t.kind == rv::Kind::Word && (d.low(n) == "and" || d.low(n) == "or"));
        if (!list_part) break;
        if (t.kind == rv::Kind::Number && d.punct(n + 1, ':') && d.number(n + 2) && d.punct(n + 3, ':') &&
            d.number(n + 4) && d.joined(n + 4))
            hms = true;
        ++n;
    }
    if (hms) return true;
    if (!d.word(n)) return false;
    std::string_view w = d.low(n);
    if (w == "long" || w == "mark" || w == "remaining" || w == "left" || w == "runtime" || w == "elapsed")
        return true;
    if (w == "per" || w == "a") {  // a pace: 6:42 per mile, 4:30 a kilometer
        size_t k = n + 1;
        if (d.word(k) && (d.is(k, "mile") || d.is(k, "km") || d.is(k, "kilometer") || d.is(k, "kilometre") ||
                          d.is(k, "lap") || d.is(k, "k") || d.is(k, "mi")))
            return true;
        if (w == "per" && d.number(k) && (d.is(k + 1, "m") || d.is(k + 1, "meters") || d.is(k + 1, "metres")))
            return true;  // per 100 m
    }
    if (w == "of" || w == "into") {  // ended at 2:41 of round 3, at 12:05 of the second period
        size_t k = d.is(n + 1, "the") ? n + 2 : n + 1;
        if (d.word(k) && (d.is(k, "round") || d.is(k, "period") || d.is(k, "quarter") || d.is(k, "half") ||
                          d.is(k, "overtime") || d.is(k, "inning") || d.is(k, "set") || d.is(k, "game") ||
                          ((d.is(k, "first") || d.is(k, "second") || d.is(k, "third") || d.is(k, "fourth") ||
                            d.is(k, "final")) &&
                           d.word(k + 1) && (d.is(k + 1, "round") || d.is(k + 1, "period") || d.is(k + 1, "quarter") ||
                                             d.is(k + 1, "half") || d.is(k + 1, "set")))))
            return true;
    }
    if (w == "after" || w == "into" || w == "from") {  // separated at 03:55 after ignition
        size_t k = d.is(n + 1, "the") ? n + 2 : n + 1;
        if (d.word(k) && ELAPSED_FROM.contains(d.low(k))) return true;
    }
    if (w == "in" || w == "into" || w == "of" || w == "on") {
        // in the recording, on the same tape, in the shared audio file
        for (size_t k = n + 1; k < d.size() && k <= n + 4 && d.word(k) && d[k].space_before; ++k)
            if (MEDIA.contains(d.low(k)) || d.is(k, "timer") || d.is(k, "stopwatch") || d.is(k, "countdown") ||
                d.is(k, "scoreboard") || d.is(k, "timestamps") || d.is(k, "timestamp") || d.is(k, "timecode"))
                return true;
    }
    return false;
}

// 2:07:19 two hours, seven minutes and nineteen seconds; zero parts are said (1:45:00 ... and zero seconds),
// except a zero hour (0:45:10 forty-five minutes and ten seconds).
std::string hms_words(int h, int m, int s) {
    std::string mins = rv::cardinal(m) + " " + plural(m, "minute");
    std::string secs = rv::cardinal(s) + " " + plural(s, "second");
    if (h == 0) return m == 0 ? secs : mins + " and " + secs;
    return rv::cardinal(h) + " " + plural(h, "hour") + ", " + mins + " and " + secs;
}

std::string ms_words(int m, int s) {
    if (m == 0) return s ? rv::cardinal(s) + " " + plural(s, "second") : "";
    std::string w = rv::cardinal(m) + " " + plural(m, "minute");
    if (s) w += " " + rv::cardinal(s) + " " + plural(s, "second");
    return w;
}

// Clock-style durations: 2:07:19, took 4:17, at 03:19 in the recording.
RV_RULE("datetime", clock_duration) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i) || !free_before(doc, i)) continue;
        Clock c = parse_clock(doc, i, 99);
        if (!c.ok || c.zulu) continue;
        bool frac = c.s >= 0 ? c.last != i + 4 : false;
        size_t al;
        // a pace: 5:09/km (the unit after the slash is read by its own rule)
        bool pace = doc.punct(c.last + 1, '/') && doc.joined(c.last + 1) && doc.word(c.last + 2) &&
                    doc.joined(c.last + 2) && (doc.is(c.last + 2, "km") || doc.is(c.last + 2, "mi") ||
                                               doc.is(c.last + 2, "mile") || doc.is(c.last + 2, "k"));
        if (frac || (!pace && !free_after(doc, c.last)) || ratio_after(doc, c.last) ||
            (c.s < 0 && ratio_before(doc, i)))
            continue;
        if (c.last + 1 < doc.size() && ampm_at(doc, c.last + 1, &al)) continue;
        size_t k = range_dash(doc, c.last + 1);
        if (k != NPOS && doc.number(k)) continue;  // a range of clock times: runs 9:00-5:00
        if (c.s < 0 && doc.is(c.last + 1, "to") && doc.number(c.last + 2) && parse_clock(doc, c.last + 2).ok) continue;
        // after a date with a year it is the time of a timestamp: "Mon, 30 Jun 2025 14:05:33 +0200"
        bool timeish = time_after(doc, c.last, c.s >= 0) || (i > 0 && doc[i].space_before && year4(doc, i - 1));
        bool context = pace || (!timeish && (duration_context(doc, i) || duration_after(doc, c.last) ||
                                             (c.s < 0 && media_sentence(doc, i))));
        std::string words;
        int score;
        if (c.s >= 0) {
            words = hms_words(c.h, c.m, c.s);
            std::string_view cue = word_before(doc, i);
            score = context ? 75 : ((time_cue(cue) && len(doc, i) == 2) || timeish ? 0 : 55);
        } else {
            words = ms_words(c.h, c.m);
            // "0:10" with a one-digit zero is a count, not a time of day (that is 00:10)
            score = context ? 75 : (c.h == 0 && len(doc, i) == 1 && !timeish ? 55 : 0);
        }
        if (!score || words.empty()) continue;
        size_t last = c.last;
        if (pace) {  // 5:09/km five minutes nine seconds per kilometer
            std::string_view u = doc.low(c.last + 2);
            words += u == "km" ? " per kilometer" : u == "k" ? " per K" : " per mile";
            last = c.last + 2;
        }
        out.tokens(i, last, words, score);
        i = last;
    }
}

struct Unit {
    const char* written;
    const char* name;
};
// Units of time after a number. Single letters need more context (see unit_score).
const Unit UNITS[] = {{"h", "hour"},   {"hr", "hour"},    {"hrs", "hour"},   {"hour", "hour"},   {"hours", "hour"},
                      {"m", "minute"}, {"min", "minute"}, {"mins", "minute"}, {"s", "second"},   {"sec", "second"},
                      {"secs", "second"}, {"d", "day"},   {"w", "week"},     {"wk", "week"},    {"wks", "week"},    {"mo", "month"},
                      {"mos", "month"}, {"yr", "year"},   {"yrs", "year"},   {"minute", "minute"},
                      {"minutes", "minute"}, {"second", "second"}, {"seconds", "second"}, {"day", "day"},
                      {"days", "day"}};

const Unit* unit_of(const Doc& d, size_t i) {
    if (!d.word(i)) return nullptr;
    std::string_view w = d.str(i);
    for (const Unit& u : UNITS)
        if (w == u.written) return &u;
    return nullptr;
}

// Units written as words (hours, minutes): only part of a compound ("1 hour 30m") is read by this rule.
bool word_unit(const Unit* u) {
    std::string_view w = u->written;
    return w.size() > 3 && w != "mins" && w != "secs";
}

const rv::WordSet DAY_NEXT{"and", "or", "to", "after", "before", "of", "ago", "later", "left", "in", "on", "at",
                           "for", "from", "with", "the", "until", "till", "between", "per", "each", "a", "by",
                           "now", "max", "tops", "remaining"};
const rv::WordSet DETERMINERS{"a", "an", "the", "each", "every", "this", "that", "another", "per", "one", "any"};
const rv::WordSet FUNCTION_WORDS{
    "of",     "to",     "in",      "on",      "at",      "for",     "from",    "with",    "and",    "or",
    "but",    "per",    "each",    "a",       "an",      "the",     "before",  "after",   "later",  "ago",
    "long",   "left",   "remaining", "total", "max",     "maximum", "min",     "minimum", "is",     "was",
    "are",    "were",   "be",      "been",    "will",    "would",   "can",     "could",   "should", "may",
    "might",  "must",   "do",      "does",    "did",     "has",     "have",    "had",     "that",   "which",
    "who",    "while",  "when",    "where",   "if",      "so",      "then",    "than",    "as",     "by",
    "into",   "over",   "under",   "until",   "till",    "since",   "between", "through", "during", "without",
    "within", "across", "around",  "about",   "rest",    "apart",   "off",     "out",     "up",     "down",
    "away",   "back",   "behind",  "ahead",   "early",   "late",    "earlier", "more",    "less",   "too",
    "this",   "these",  "those",   "its",     "their",   "his",     "her",     "our",     "my",     "your",
    "overnight", "straight", "now",  "ago",   "old",     "faster",  "slower",  "longer",  "shorter", "past",
    "last",   "instead", "moving", "used",   "every",   "once",    "because", "including", "followed", "remains",
    "needs",  "keeps",  "next",    "get",     "gets",    "rather",  "unless",  "goes",    "means",   "came",
    "counts", "bought", "keep",    "set",     "apiece",  "flat",    "tops",    "plus",    "minus",   "versus",
    "vs",     "worth",  "already", "still",   "yet",     "even",    "just",    "only",    "again",   "also",
    "though", "although", "thus",  "whereas", "not",     "no",      "it",      "they",    "we",      "you",
    "he",     "she",    "there",   "here",    "what",    "how",     "why",     "all",     "both",    "either",
    "neither", "some",  "most",    "many",    "much",    "few",     "several", "per",     "via",     "against",
    "toward", "towards", "upon",   "onto",    "beyond",  "except",  "like",    "unlike",  "near",    "along",
    "among",  "amid",   "beside",  "besides", "despite", "inside",  "outside", "underneath", "whether", "nor",
    "overall", "altogether", "apart", "combined", "flat", "total", "respectively", "elapsed", "remaining",
    "lol",    "omg",    "btw",     "tbh",     "imo",     "fyi",     "thx",     "pls",     "smh",     "idk"};

// Nouns that make a joined m or d a time unit: a 45m break, 3d outages.
const rv::WordSet TIME_NOUNS{
    "pause",  "timeout", "window",  "delay",   "break",   "grace",   "buffer",  "clip",     "retry",  "check",
    "hold",   "interval", "intervals", "jog",  "jogs",    "wait",    "drop",    "drops",    "spike",  "limit",
    "block",  "blocks", "checkout", "sampling", "outage", "outages", "test",    "cooldown", "cycle",  "slot",
    "slots",  "response", "soak",   "regimen", "countdown", "timer", "polling", "sync",     "backoff", "recovery",
    "reset",  "warmup", "warm",     "session", "meeting", "call",    "walk",    "drive",    "ride",   "commute",
    "nap",    "rest",   "shift",    "shifts",  "trial",   "sprint",  "event",   "route",    "schedule", "batch",
    "briefing", "cool", "lag",      "latency", "gap",     "cutoff",  "deadline", "notice",  "lockout", "trip",
    "flight", "journey", "stay",    "period",  "phase",   "fast",    "lesson",  "class",    "course", "workshop",
    "webinar", "video", "podcast",  "episode", "segment", "show",    "film",    "movie",    "stretch", "span",
    "horizon", "rollback", "retention", "expiry", "lead", "run",     "runs",    "cadence",  "loop",   "turnaround",
    "benchmark", "warning", "default", "average", "sequence", "card", "intro",  "outro",   "spot",   "commercial",
    "ad",     "ads",    "teaser",   "preview", "snippet", "burst",   "exposure", "dwell",   "soak",   "heartbeat",
    "ttl",    "refresh", "rotation", "stop",   "halt",    "freeze",  "silence", "lockdown", "boil",  "simmer",
    "steep",  "bloom",  "plank",    "hang",    "sprints", "intervals", "repeats", "rounds", "splits", "laps",
    "task",   "tasks",  "clips",    "breaks",  "pauses",  "delays",  "timeouts", "windows", "jingle"};

// The word after a number and its unit is a noun the pair describes ("a 45 min break"): then the unit is
// singular.
// Chat words that are not nouns: "fasting 10-12 hrs b4", "45 min thx".
const rv::WordSet CHAT_WORDS{"b4", "thx", "pls", "plz", "lol", "btw", "imo", "tbh", "idk", "fyi", "asap", "rn",
                             "tmrw", "smh", "omg", "ttyl", "brb", "ok", "okay", "ty", "np", "jk"};

// Token n follows token n-1 after a gap of fields (a tab or several spaces): "12.3 min   range: 2.5 km".
bool field_gap(const Doc& d, size_t n) {
    if (n == 0 || n >= d.size()) return false;
    int spaces = 0;
    for (uint32_t k = d[n - 1].end; k < d[n].start; ++k) {
        if (d.text[k] == '\t') return true;
        if (d.text[k] == ' ') ++spaces;
    }
    return spaces >= 2;
}

bool noun_after(const Doc& d, size_t last) {
    size_t n = last + 1;
    if (n >= d.size() || !d.word(n) || !d[n].space_before || d[n].newline_before || d[n].casing != rv::LOWER ||
        field_gap(d, n))
        return false;
    if (d.joined(n + 1) && d.number(n + 1)) return false;  // b4, w8
    std::string_view w = d.low(n);
    if (w.size() > 3 && w.substr(w.size() - 2) == "ly") return false;  // adverbs: exactly, roughly
    if (w == "end" && d.is(n + 1, "to")) return false;                // 1h 15m end to end
    // adjectives after the duration: "delays of 5 min common at night"
    if (w == "common" || w == "typical" || w == "normal" || w == "usual" || w == "possible" || w == "likely" ||
        w == "expected" || w == "required" || w == "allowed" || w == "needed" || w == "maximum" || w == "minimum")
        return false;
    return !FUNCTION_WORDS.contains(w) && !duration_cue(w) && !CHAT_WORDS.contains(w);
}

// the 90s, '80s, 1990s: a decade, not seconds. A decade is written with the s joined and no decimals.
bool decade_like(const Doc& d, size_t i) {
    size_t u = i + 1;
    if (!d.word(u) || !d.joined(u) || (d.punct(i + 1, '.') && d.joined(i + 1))) return false;
    int v = num(d, i, 4);
    if (len(d, i) == 4) return v % 10 == 0;
    if (len(d, i) == 2 && v % 10 == 0) return true;
    return d.joined(i) && apostrophe_cp(d[i - 1].cp);
}

// The unit group at Number k is a compact duration that is not a decade look-alike (30s with "the" before it).
bool clear_duration_at(const Doc& d, size_t k);

// The sentence around token i has another compact duration (1h, 30s, 2d, 45 min): then "25m" is minutes.
bool duration_nearby(const Doc& d, size_t i) {
    size_t lo = i > 16 ? i - 16 : 0, hi = std::min(d.size(), i + 16);
    for (size_t k = lo; k + 1 < hi; ++k) {
        if (k == i || !d.number(k) || (k > 0 && d.punct(k - 1, '.') && d.joined(k))) continue;
        if (clear_duration_at(d, k)) return true;
    }
    return false;
}

bool clear_duration_at(const Doc& d, size_t k) {
    const Unit* u = unit_of(d, k + 1);
    if (!u || word_unit(u) || (u->written[0] == 'm' && !d.joined(k + 1))) return false;
    if (u->written[0] == 's' && u->written[1] == '\0' && (decade_like(d, k) || len(d, k) >= 4)) return false;
    return d.joined(k + 1) || std::string_view(u->written).size() > 1;
}

// A duration written in words close to token i ("30s, two minutes, 90s"): a 2-digit "90s" is then seconds.
bool duration_words_nearby(const Doc& d, size_t i) {
    size_t lo = i > 8 ? i - 8 : 0, hi = std::min(d.size(), i + 8);
    for (size_t k = lo; k < hi; ++k) {
        if (!d.word(k)) continue;
        std::string_view w = d.low(k);
        if (w == "seconds" || w == "minutes" || w == "second" || w == "minute" || w == "secs" || w == "mins")
            return true;
    }
    return false;
}

struct Group {
    size_t first, last;  // number .. unit (and its period)
    size_t unit_tok;
    std::string number;  // words
    bool one;
    const Unit* unit;
    bool joined;
};

bool group_at(const Doc& d, size_t i, Group* g) {
    if (!d.number(i)) return false;
    rv::NumberSpan n = rv::parse_number(d, i);
    size_t u = n.last + 1;
    const Unit* unit = unit_of(d, u);
    if (!unit || d[u].newline_before) return false;
    std::string words = rv::read_number(n);
    if (words.empty() && n.whole.size() == 2 && n.whole[0] == '0' && n.fraction.empty())
        words = rv::cardinal(static_cast<uint64_t>(n.whole[1] - '0'));  // 6m 03s
    if (words.empty()) return false;
    g->first = i;
    g->last = u;
    g->unit_tok = u;
    g->number = words;
    g->one = n.whole == "1" && n.fraction.empty();
    g->unit = unit;
    g->joined = d.joined(u);
    // "4 day(s)", "2 hour(s)": the optional plural is read as a plural
    if (d.punct(u + 1, '(') && d.joined(u + 1) && d.is(u + 2, "s") && d.joined(u + 2) && d.punct(u + 3, ')') &&
        d.joined(u + 3)) {
        g->last = u + 3;
        return free_after(d, g->last);
    }
    if (std::string_view(unit->written).size() > 1 && !word_unit(unit)) {
        g->last = abbrev_end(d, u);
        // "delayed 2 hrs. btw the forecast": the period ends a chat sentence
        if (g->last > u && d.word(g->last + 1) && CHAT_WORDS.contains(d.low(g->last + 1))) g->last = u;
    }
    if (!free_after(d, g->last) && !(d.joined(g->last + 1) && d.number(g->last + 1))) return false;
    return true;
}

const rv::WordSet DURATION_PREPS{"for",    "after",  "every",   "of",     "another", "about", "approximately",
                                 "within", "around", "roughly", "nearly", "over",    "under", "than",
                                 "per",    "extra",  "additional", "just", "only",    "allows", "allow"};
const rv::WordSet METER_NEXT{"long", "wide", "tall", "high", "deep", "thick", "away", "above", "below", "across",
                             "in", "of", "cable", "wall", "pool", "tower", "dive", "drop", "sprint", "race",
                             "radius", "diameter", "length", "width", "height", "depth", "deep", "run", "dash"};
const rv::WordSet AFTER_MINUTES{"before", "after", "later", "ago", "early", "late", "left", "remaining", "to",
                                "on", "for", "and", "or", "per", "each", "total", "max", "at", "with", "while"};

// Words after "20s" or "90s" that make it seconds, not a decade: 20s per gate change, 90s later.
const rv::WordSet SECONDS_NEXT{"per", "each", "between", "later", "before", "after", "apart", "ago", "elapsed",
                               "long", "max", "maximum", "min", "minimum", "tops", "flat", "remaining", "left"};
// Words before "30s" that make it a decade: in their 40s, the early 90s.
const rv::WordSet DECADE_BEFORE{"the", "their", "his", "her", "my", "our", "your", "its", "early", "mid", "late",
                                "since", "during", "throughout", "classic", "golden", "roaring", "swinging"};
// Words before the cue word that make "N s" seconds: timeout of 30s, the window was 80s, timeout to 10s.
const rv::WordSet SECONDS_NOUNS{"timeout", "timeouts", "window", "delay", "interval", "wait", "pause", "ttl",
                                "latency", "limit", "backoff", "cooldown", "duration", "timer", "lag", "gap",
                                "expiry", "refresh", "polling", "heartbeat", "deadline", "budget", "hold", "dwell",
                                "rest", "break", "warmup", "warm-up", "grace", "buffer", "cutoff"};

// The word before the cue at the start of a unit group: "timeout of 30s" gives "timeout".
std::string_view word_before_cue(const Doc& d, size_t first) {
    size_t c = word_before_index(d, first);
    if (c == NPOS) return std::string_view();
    size_t w = word_before_index(d, c);
    return w == NPOS ? std::string_view() : d.low(w);
}

// "N s" at Number k has a context that makes it seconds even though it looks like a decade: a cue or a
// preposition before it, an article, a time noun after it.
bool strong_seconds(const Doc& d, size_t k, size_t last) {
    std::string_view cue = word_before(d, k);
    if (duration_cue(cue) || DURATION_PREPS.contains(cue)) return true;
    if (cue == "a" || cue == "an" || cue == "each" || cue == "every" || cue == "per") return true;
    std::string_view wb = word_before_cue(d, k);
    if (SECONDS_NOUNS.contains(wb) && !DECADE_BEFORE.contains(cue)) return true;
    // set to 30s, cut to 90s, lowered to 10s
    if (cue == "to" && (wb == "set" || wb == "cut" || wb == "trimmed" || wb == "reduced" || wb == "lowered" ||
                        wb == "raised" || wb == "increased" || wb == "extended" || wb == "capped" || wb == "limited" ||
                        wb == "shortened" || wb == "bumped" || wb == "changed" || wb == "dropped"))
        return true;
    size_t n = last + 1;
    if (d.word(n) && (TIME_NOUNS.contains(d.low(n)) || SECONDS_NEXT.contains(d.low(n)))) return true;
    return false;
}

// The same "30s" appears elsewhere in the text in a context that makes it seconds ("the 30s default" after
// "a timeout of 30s"): one token is read one way throughout.
bool same_token_seconds(const Doc& d, size_t i) {
    std::string_view s = d.str(i);
    if (!noun_after(d, i + 1)) return false;  // only "the 30s default", not "30s and 40s"
    for (size_t k = 0; k + 1 < d.size(); ++k)
        if (k != i && d.number(k) && d.str(k) == s && d.is(k + 1, "s") && d.joined(k + 1) && free_after(d, k + 1) &&
            strong_seconds(d, k, k + 1))
            return true;
    return false;
}

// A single-letter unit is a time unit only with more context.
int unit_score(const Doc& d, const Group& g, bool compound, std::string_view cue) {
    char c = g.unit->written[0];
    bool single = g.unit->written[1] == '\0';
    bool cued = duration_cue(cue) || DURATION_PREPS.contains(cue);
    size_t n = g.last + 1;
    bool end = n >= d.size() || d[n].kind == rv::Kind::Punct || d[n].newline_before || field_gap(d, n);
    bool noun = noun_after(d, g.last);
    // the noun the duration describes, one word on: "a 20s stress test"
    bool time_noun = d.word(n) && (TIME_NOUNS.contains(d.low(n)) || (noun && d.word(n + 1) && d[n + 1].space_before &&
                                                                     TIME_NOUNS.contains(d.low(n + 1))));
    if (word_unit(g.unit)) return compound ? 62 : 0;
    if (!single) return 62;
    if (c == 'h')
        return g.joined || compound || cued || end || d.is(n, "of") || noun ||
                       (d.word(n) && AFTER_MINUTES.contains(d.low(n)))
                   ? 62
                   : 0;
    if (c == 'm' || c == 'w') {
        if (!g.joined) return 0;
        if (c == 'm' && !compound && !time_noun && g.number.find("point") != std::string::npos)
            return 0;  // "15.3m long", "6.22m": minutes are not written with decimals
        if (compound || cued || time_noun) return 62;
        // "300m tall", "2m in length", "4m of extra lanes": meters; "45m of baking": minutes
        if (d.word(n) && (d.is(n, "of") || d.is(n, "in"))) {
            std::string_view x = d.word(n + 1) ? d.low(n + 1) : std::string_view();
            if (d.is(n, "in") && (x == "length" || x == "height" || x == "width" || x == "depth" || x == "diameter" ||
                                  x == "size" || x == "total"))
                return 0;
            bool activity = (x.size() > 4 && x.substr(x.size() - 3) == "ing") || TIME_NOUNS.contains(x) ||
                            x == "prep" || x == "work" || x == "play" || x == "sleep" || x == "practice" ||
                            x == "downtime" || x == "delays" || x == "silence" || x == "daylight" || x == "the";
            if (d.is(n, "of") && !activity) return 0;
        } else if (d.word(n) && METER_NEXT.contains(d.low(n))) {
            return 0;
        }
        if (duration_nearby(d, g.first)) return 62;
        if (c == 'w') return 0;
        if (DETERMINERS.contains(cue) || g.number.find("hundred") != std::string::npos) return 0;
        return end || (d.word(n) && AFTER_MINUTES.contains(d.low(n))) ? 50 : 0;
    }
    if (c == 's') {
        // "Alpha 7s III", "iPhone 6s Plus": a product name
        if (g.joined && !compound && !cued && d.word(n) && d[n].space_before && !d[n].newline_before &&
            (d[n].casing == rv::UPPER || d[n].casing == rv::TITLE) && !zone_word(d.str(n)) && !month_at(d, n) &&
            !weekday_full_at(d, n) && !weekday_abbrev_at(d, n))
            return 0;
        bool dec = decade_like(d, g.first);
        if (dec && len(d, g.first) == 4) return compound ? 62 : 0;  // 1980s, 2010s
        if (dec) {
            if (compound) return 62;
            // 60s-style, a 90s-era bridge
            if (d.punct(n, '-') && d.joined(n) && d.word(n + 1) && d.joined(n + 1) && !d.is(n + 1, "long") &&
                !TIME_NOUNS.contains(d.low(n + 1)))
                return 0;
            // seconds must beat the decade reading of numbers_core (65); "the 20s stress benchmark"
            if (strong_seconds(d, g.first, g.last)) return 68;
            if (noun && d.word(n + 1) && d[n + 1].space_before &&
                (d.is(n + 1, "benchmark") || d.is(n + 1, "task") || d.is(n + 1, "test") || d.is(n + 1, "timeout") ||
                 d.is(n + 1, "delay") || d.is(n + 1, "countdown") || d.is(n + 1, "interval") || d.is(n + 1, "pause")))
                return 68;
            if (DECADE_BEFORE.contains(cue) || (g.first > 0 && d.joined(g.first) && apostrophe_cp(d[g.first - 1].cp)))
                return 0;
            // "in their 50s and 60s": the second decade of a pair
            size_t ci = word_before_index(d, g.first);
            if (ci != NPOS && (cue == "and" || cue == "or" || cue == "to") && ci >= 2 && d.is(ci - 1, "s") &&
                d.joined(ci - 1) && d.number(ci - 2) && decade_like(d, ci - 2))
                return 0;
            if (same_token_seconds(d, g.first)) return 68;
            // "the 5s and 10s": plural numbers in a pair (numbers_core)
            if (ci != NPOS && (cue == "and" || cue == "or") && ci >= 2 && d.is(ci - 1, "s") && d.joined(ci - 1) &&
                d.number(ci - 2) && DECADE_BEFORE.contains(word_before(d, ci - 2)))
                return 0;
            // seconds without a cue ("bloom 40s, then pour"): low, so a decade read by its cue wins
            return duration_nearby(d, g.first) || duration_words_nearby(d, g.first) ? 68 : 20;
        }
        if (cue == "the" && !time_noun && !noun) return 0;  // "the 5s": fives
        // "his 3s and 4s": plurals of numbers
        if (DECADE_BEFORE.contains(cue) && cue != "the" && !time_noun) return 0;
        {
            size_t ci = word_before_index(d, g.first);
            if (ci != NPOS && (cue == "and" || cue == "or") && ci >= 2 && d.is(ci - 1, "s") && d.joined(ci - 1) &&
                d.number(ci - 2) && DECADE_BEFORE.contains(word_before(d, ci - 2)))
                return 0;
        }
        if (g.joined || compound || cued) return 62;
        return end || (d.word(n) && d[n].casing == rv::LOWER) ? 62 : 0;
    }
    if (c == 'd') {
        if (!g.joined) return 0;
        if (compound || cued || time_noun) return 62;
        return end || DAY_NEXT.contains(d.low(n)) ? 60 : 0;
    }
    return 0;
}

// Compact durations and time units: 1h 30m, 2m 15s, 45s, 3d, 36 hrs, 50 min, 2 hrs 30 min, 10-12 hrs.
// "~30 min", "sub-10 min": a prefix that other rules read ("approximately", "sub").
bool prefixed(const Doc& d, size_t i) {
    if (!d.joined(i)) return false;
    if (d.punct(i - 1, '~')) return i == 1 || !d.joined(i - 1);
    return d.punct(i - 1, '-') && i >= 2 && d.joined(i - 1) && d.is(i - 2, "sub");
}

// A decimal written without its zero: ".9s", "(.75 s)".
bool leading_dot(const Doc& d, size_t i) {
    if (!d.punct(i - 1, '.') || !d.joined(i)) return false;
    if (i == 1 || !d.joined(i - 1)) return true;
    const rv::Token& p = d[i - 2];
    return p.kind == rv::Kind::Punct && (p.cp == '(' || p.cp == '[' || p.cp == '~' || p.cp == '"' || p.cp == 0x201C);
}

RV_RULE("datetime", time_units) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        bool dot = leading_dot(doc, i);
        if (!(free_before(doc, i) || prefixed(doc, i) || dot)) continue;
        // "a 45-min slot", "2-hr drive": an adjective with a hyphen, singular unit
        if (!dot && dash(doc, i + 1) && doc.joined(i + 1) && doc.word(i + 2) && doc.joined(i + 2) &&
            free_after(doc, i + 2) && !(i > 0 && doc.joined(i) && doc.number(i - 1))) {
            const Unit* u = unit_of(doc, i + 2);
            std::string_view w = u ? std::string_view(u->written) : std::string_view();
            std::string n = rv::read_number(rv::parse_number(doc, i));
            if (u && w.size() > 1 && !word_unit(u) && !n.empty() && rv::parse_number(doc, i).last == i) {
                out.tokens(i, i + 2, n + "-" + u->name, 64);
                i += 2;
                continue;
            }
        }
        // "60s-style", "90s-era": a decade in a compound (numbers_core)
        if (!dot && len(doc, i) == 2 && num(doc, i, 2) % 10 == 0 && doc.word(i + 1) && doc.str(i + 1) == "s" &&
            doc.joined(i + 1) &&
            dash(doc, i + 2) && doc.joined(i + 2) && doc.word(i + 3) && doc.joined(i + 3) && !doc.is(i + 3, "long")) {
            i += 1;
            continue;
        }
        // a range of numbers before the unit: 10-12 hrs
        std::string prefix;
        size_t start = i, at = i;
        if (!dot) {
            rv::NumberSpan n0 = rv::parse_number(doc, i);
            size_t k = range_dash(doc, n0.last + 1);
            Group probe;
            if (k != NPOS && !unit_of(doc, n0.last + 1) && group_at(doc, k, &probe)) {
                std::string w0 = rv::read_number(n0);
                if (w0.empty()) continue;
                prefix = w0 + " to ";
                at = k;
            }
        }
        Group gs[4];
        int count = 0;
        size_t k = at;
        while (count < 4 && group_at(doc, k, &gs[count])) {
            ++count;
            size_t n = gs[count - 1].last + 1;
            if (n >= doc.size() || doc[n].newline_before || !doc.number(n)) break;
            k = n;
        }
        if (count == 0) continue;
        if (dot) {
            if (!rv::parse_number(doc, i).fraction.empty()) continue;
            gs[0].number = "point " + rv::digits(doc.str(i));
            gs[0].one = false;
            start = i - 1;
        }
        // keep the groups up to the first one whose unit lacks the context it needs
        std::string_view cue = word_before(doc, start);
        bool compound = count > 1;
        int ok = 0, score = 100;
        for (int g = 0; g < count; ++g) {
            int sc = unit_score(doc, gs[g], compound, cue);
            if (sc <= 0) break;
            score = std::min(score, sc);
            ok = g + 1;
        }
        if (ok == 1 && compound) {
            compound = false;
            score = unit_score(doc, gs[0], false, cue);
            if (score < 0) score = 0;
        }
        // not a time unit: a product name ("Alpha 7s III", letters_products), plural numbers ("his 3s and 8s",
        // numbers_core), a decade ("70s posters", numbers_core)
        if (ok == 0 || score == 0) continue;
        // before a noun the unit is singular: a 45 min break, a 2m 20s clip, 5 min intervals, the 5s average
        size_t after = gs[ok - 1].last + 1;
        bool det = DETERMINERS.contains(cue);
        bool singular =
            gs[ok - 1].last == gs[ok - 1].unit_tok &&
            (noun_after(doc, gs[ok - 1].last) ||
             ((doc.is(after, "average") || doc.is(after, "rest")) && doc[after].space_before && det) ||
             // "0.25 s average latency", "a 90 min daily outage": an adjective before the noun
             (doc.word(after) && doc[after].casing == rv::LOWER && !doc[after].newline_before &&
              (doc.is(after, "average") || doc.is(after, "median") || doc.is(after, "mean") ||
               (det && doc.low(after).size() > 4 && doc.low(after).substr(doc.low(after).size() - 2) == "ly")) &&
              doc[after].space_before &&
              (noun_after(doc, after) || (doc.word(after + 1) && doc[after + 1].casing == rv::LOWER &&
                                          !FUNCTION_WORDS.contains(doc.low(after + 1))))));
        std::string words = prefix;
        for (int g = 0; g < ok; ++g) {
            if (g) words += ' ';
            words += gs[g].number + " " + (gs[g].one || singular ? gs[g].unit->name : plural(2, gs[g].unit->name));
        }
        if (compound) score = 66;
        out.tokens(start, gs[ok - 1].last, words, score);
        i = gs[ok - 1].last;
    }
}

// 24/7: twenty-four seven.
// Words before "18/7" that make it hours a day and days a week, like 24/7: staffing was 18/7, runs 20/7.
const rv::WordSet AVAILABILITY{"was", "is", "ran", "runs", "run", "operates", "operated", "operating", "open",
                               "staffed", "available", "staffing", "coverage", "support", "service", "works",
                               "working", "worked", "manned", "monitored", "online", "trade", "trades",
                               "traded", "trading"};

RV_RULE("datetime", twenty_four_seven) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!doc.number(i) || !doc.punct(i + 1, '/') || !doc.joined(i + 1) || !doc.joined(i + 2) ||
            !doc.number(i + 2) || !free_before(doc, i) || !free_after(doc, i + 2))
            continue;
        if (doc.str(i) == "24" && doc.str(i + 2) == "7") {
            out.tokens(i, i + 2, "twenty-four seven", 85);
            i += 2;
            continue;
        }
        int h = num(doc, i, 2), days = num(doc, i + 2, 1);
        // futures trade 23/5
        if (h >= 8 && h <= 24 && days >= 5 && days <= 7 && AVAILABILITY.contains(word_before(doc, i))) {
            out.tokens(i, i + 2, rv::cardinal(h) + " " + rv::cardinal(days), 72);
            i += 2;
        }
    }
}

}  // namespace
