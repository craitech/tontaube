// A slash between numbers, read by what it means there: fractions (1/2 cup, 1 1/2, ½), scores (10/10, 4.5/5),
// progress counts (15/20 tasks), odds (9/2 odds), splits (60/40, 50/50), 24/7, blood pressure (120/80 mmHg),
// and a spaced slash between two listed values (0.08 / 0.02 -> and); a fraction with its measure (1/2 tsp,
// 3/4" thick, 1/4%), ¹⁄₂, f/2.8. The date reading of m/d belongs to datetime: date-shaped pairs without a cue
// get a score below the date's.
#include "rules/numbers_context/common.h"

namespace {

using namespace nctx;

const rv::WordSet SCORE_PREV{
    "achieved", "average", "averaged", "averages", "award", "awarded", "awards", "came", "earn", "earned",
    "earning", "earns", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "got", "grade",
    "graded", "grades", "mark", "marked", "marks", "rate", "rated", "rates", "rating", "ratings", "received",
    "receive", "receives", "result", "results", "score", "scorecard", "scored", "scores", "scoring", "tally",
    "reached", "reaches", "reaching", "hit", "hits"};

// Score verbs that reach over an object: "rated it 8/10", "gave the draft 3/4 for clarity".
const rv::WordSet SCORE_VERBS{"awarded", "earned", "gave", "give", "gives", "giving", "graded", "rate", "rated",
                              "rates", "rating", "score", "scored", "scores", "scoring"};

const rv::WordSet SCORE_NEXT{"average", "avg", "overall", "points", "pts", "rating", "ratings", "score", "scores",
                             "star", "stars"};

const rv::WordSet PROGRESS_PREV{
    "attempt", "attempts", "batch", "batches", "block", "chapter", "checks", "chunk", "disc", "disk", "episode",
    "epoch", "epochs", "exercise", "file", "files", "frame", "game", "heat", "hole", "inning", "issue", "item",
    "items", "iteration", "job", "jobs", "lap", "laps", "leg", "level", "match", "node", "nodes", "page", "part",
    "phase", "pods", "problem", "progress", "question", "record", "replica", "replicas", "restart", "restarts",
    "retries", "retry", "round", "rounds", "row", "run", "sample", "samples", "set", "shard", "shards", "slide",
    "stage", "step", "steps", "task", "tasks", "test", "tests", "track", "tries", "try", "volume", "worker",
    "workers"};

// Aspects that a review scores: "Overall 9/10", "Value for money 3/5", "Sleep quality: 6/10".
const rv::WordSet REVIEW_PREV{
    "acting", "ambiance", "ambience", "atmosphere", "battery", "build", "camera", "cleanliness", "comfort",
    "design", "difficulty", "display", "durability", "energy", "fit", "flavor", "flavour", "food", "gameplay",
    "graphics", "location", "money", "mood", "music", "overall", "pain", "performance", "plot", "presentation",
    "price", "quality", "service", "sleep", "sound", "story", "stress", "taste", "value", "verdict", "visuals"};

// A fraction as the subject of a verb: "1/4 found it confusing", "2/3 agreed".
const rv::WordSet FRACTION_VERB_NEXT{
    "agree", "agreed", "agrees", "are", "believe", "believed", "came", "chose", "could", "did", "felt", "found",
    "had", "have", "prefer", "preferred", "reported", "said", "say", "says", "think", "thought", "voted", "want",
    "wanted", "were", "will", "would"};

// Verbs of change before "by" and a fraction: "fell by 1/3".
const rv::WordSet CHANGE_VERBS{"climbed", "cut", "decreased", "declined", "dropped", "fell", "grew", "increased",
                               "jumped", "lowered", "raised", "reduced", "rose", "shrank", "slashed", "trimmed",
                               "down", "up"};

// A part of a size that is done: "Downloaded 1.2/4.5 GB", "used 3.1/7.8 GiB".
const rv::WordSet TRANSFER_PREV{"copied", "disk", "downloaded", "downloading", "loaded", "memory", "progress", "ram",
                                "storage", "transferred", "uploaded", "uploading", "used"};
const rv::WordSet SIZE_UNITS{"gb", "gib", "kb", "kib", "mb", "mib", "tb", "tib", "km", "mi", "miles"};

// Nouns after "a 1/4 ..." that make it a date: "a 1/4 deadline".
const rv::WordSet DATE_NOUN_NEXT{"appointment", "birthday", "call", "deadline", "delivery", "event", "flight",
                                 "game", "launch", "meeting", "release", "session", "start", "visit"};

// After a fraction of a price: "1/2 off", "1/2 price".
const rv::WordSet DEAL_NEXT{"off", "price", "priced"};

const rv::WordSet NINE_ELEVEN_NEXT{"attack", "attacks", "commission", "era", "families", "hijackers", "memorial",
                                   "museum", "responders", "survivors", "terror", "terrorist", "victims"};
const rv::WordSet VISION_WORDS{"acuity", "eye", "eyes", "eyesight", "hindsight", "sight", "vision"};

// Words that make a line a list of scores: "Team scores, first round. Ana, 8/10 ..."
const rv::WordSet SCORE_LINE{"grade", "graded", "grades", "rated", "rating", "ratings", "score", "scored",
                             "scores"};

const rv::WordSet COMPLETE_NEXT{
    "active", "approved", "available", "booked", "complete", "completed", "correct", "done", "empty", "failed",
    "filled", "finished", "full", "healthy", "hit", "left", "met", "occupied", "online", "passed", "ready",
    "remaining", "right", "sold", "successful", "up", "used", "wrong"};

const rv::WordSet QTY_PREV{
    "about", "add", "added", "adding", "additional", "almost", "an", "another", "approx", "approximately",
    "ate", "barely", "drank", "eat", "exactly", "extra", "had", "has", "have", "just", "least",
    "less", "lost", "measure", "measured", "mere", "more", "nearly", "need", "needed", "needs", "only",
    "over", "pour", "remaining", "require", "required", "requires", "roughly", "some", "spent", "stir",
    "take", "takes", "than", "the", "took", "under", "use", "used", "using", "whole"};

const rv::WordSet NOT_COUNT_NOUNS{"always", "as", "has", "his", "is", "its", "perhaps", "plus", "this", "thus",
                                  "us", "was", "yes", "versus", "whereas", "towards", "afterwards"};

const rv::WordSet SPLIT_NEXT{"allocation", "arrangement", "balance", "bet", "blend", "call", "chance", "chances",
                             "deal", "distribution", "division", "mix", "odds", "partnership", "proposition",
                             "ratio", "rule", "share", "shot", "split", "vote", "weighting"};

const rv::WordSet ODDS_NEXT{"odds", "shot", "favorite", "favourite", "underdog", "outsider", "longshot"};
const rv::WordSet ODDS_PREV{"odds", "bookie", "bookies", "bookmaker", "bookmakers", "betting", "wager", "shot",
                            "favorite", "favourite", "longshot", "outsider"};

// Words right before "5/1" that make it odds: "won at 5/1", "priced at 3/1", "were 7/1".
const rv::WordSet ODDS_AT{"at", "priced", "paying", "pays", "were", "was", "are", "is"};

const rv::WordSet PRESSURE{"pressure", "bp"};
const rv::WordSet PRESSURE_LINK{"at", "came", "is", "measured", "of", "read", "reading", "reads", "remained",
                                "stayed", "was", "were"};

const rv::WordSet LEGAL_PREV{"regulation", "directive", "eu", "decision", "case", "resolution", "doi", "no"};

bool is_slash(const Doc& d, size_t j) {
    return d.punct(j, '/') || d.punct(j, 0x2044) || d.punct(j, 0x2215);
}

// The next word is a plural noun that counts things: "15/20 tasks", "8/10 dentists".
bool count_noun(const Doc& d, size_t j) {
    if (j == Doc::npos || !d.word(j) || !(d[j].casing & rv::LOWER)) return false;
    std::string_view w = d.low(j);
    if (w.size() < 4 || w.back() != 's' || w[w.size() - 2] == 's' || w[w.size() - 2] == 'u') return false;
    return !NOT_COUNT_NOUNS.contains(w) && !FRACTION_NEXT.contains(w);
}

const rv::WordSet NOT_ADJECTIVE{"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on",
                                "or", "the", "to", "was", "were", "with"};

// "4/5 active cases": one describing word, then a plural count noun.
bool count_noun_after_adjective(const Doc& d, size_t j) {
    if (j == Doc::npos || !d.word(j) || !(d[j].casing & rv::LOWER) || NOT_ADJECTIVE.contains(d.low(j))) return false;
    if (!d.word(j + 1) || d[j + 1].newline_before) return false;
    return count_noun(d, j + 1);
}

// An unmistakable score on the line of token i, or a score word: then a date-shaped n/10 or n/5 on that line
// is a score too. Cached per line by the caller.
bool line_has_scores(const Doc& d, size_t i, size_t& line_first, size_t& line_last);

// "2/10 at 12:15", "4/9 at 1:30": a pair followed by "at" and a clock time is a date.
bool time_after(const Doc& d, size_t last) {
    size_t j = last + 1;
    if (d.punct(j, ',')) ++j;
    return d.is(j, "at") && d.number(j + 1);
}

bool score_like(const Doc& d, size_t slash) {
    if (!d.number(slash - 1) || !d.number(slash + 1) || !d.joined(slash) || !d.joined(slash + 1)) return false;
    size_t first = slash - 1;
    bool decimal = slash >= 3 && d.punct(slash - 2, '.') && d.joined(slash - 1) && d.number(slash - 3);
    if (decimal) first = slash - 3;
    if (DATE_PREV.contains(low_or_empty(d, prev_w(d, first))) || (!decimal && time_after(d, slash + 1))) return false;
    if (decimal) return true;  // 4.5/5
    std::string_view den = d.str(slash + 1);
    return (den == "5" || den == "10" || den == "100") && d.str(slash - 1).size() <= den.size();
}

// Another score (4.5/5, 7/10) in the same sentence: "4.5/5 for the budget and 2/4 for the staffing plan".
bool sibling_score(const Doc& d, size_t first, size_t last) {
    int seen = 0;
    for (size_t j = first; j-- > 1 && seen < 24; ++seen) {
        if (d[j + 1].newline_before || is_sentence_stop(d, j)) break;
        if (is_slash(d, j) && score_like(d, j)) return true;
    }
    seen = 0;
    for (size_t j = last + 1; j + 1 < d.size() && seen < 24; ++j, ++seen) {
        if (d[j].newline_before || is_sentence_stop(d, j)) break;
        if (is_slash(d, j) && score_like(d, j)) return true;
    }
    return false;
}

// The line of token i: its first and last token; table says whether it has '|' or tab cells.
void line_of(const Doc& d, size_t i, size_t& first, size_t& last, bool& table) {
    first = i;
    while (first > 0 && !d[first].newline_before) --first;
    last = i;
    while (last + 1 < d.size() && !d[last + 1].newline_before) ++last;
    table = false;
    for (size_t j = first; j <= last && !table; ++j) {
        table = d.punct(j, '|');
        if (j > first && !table) {
            std::string_view gap = d.slice(d[j - 1].end, d[j].start);
            table = gap.find('\t') != std::string_view::npos;
        }
    }
}

bool line_has_scores(const Doc& d, size_t i, size_t& line_first, size_t& line_last) {
    bool table = false;
    line_of(d, i, line_first, line_last, table);
    size_t from = line_first, to = line_last;
    if (table) {  // a table: every row of it, and the line before it (its title)
        for (int rows = 0; from > 0 && rows < 30; ++rows) {
            size_t f, l;
            bool t;
            line_of(d, from - 1, f, l, t);
            from = f;
            if (!t) break;
        }
        for (int rows = 0; to + 1 < d.size() && rows < 30; ++rows) {
            size_t f, l;
            bool t;
            line_of(d, to + 1, f, l, t);
            if (!t) break;
            to = l;
        }
    }
    for (size_t j = from; j <= to; ++j) {
        if (d.word(j) && SCORE_LINE.contains(d.low(j))) return true;
        if (!is_slash(d, j) || j == 0 || !d.number(j - 1) || !d.number(j + 1) || !d.joined(j) || !d.joined(j + 1))
            continue;
        std::string_view num = d.str(j - 1), den = d.str(j + 1);
        bool decimal = j >= 3 && d.punct(j - 2, '.') && d.joined(j - 1) && d.number(j - 3);
        bool score_den = den == "5" || den == "10" || den == "100";
        if (score_den && (decimal || num == den || den == "100")) return true;
    }
    return false;
}

struct LineCache {
    size_t first = 1, last = 0;  // empty
    bool scores = false;
};

// Reads a/b (or whole a/b) and decides how sure the reading is from the words around it.
void slash_pair(const Doc& d, rv::Out& out, const Num& a, const Num& b, size_t start, const Num* whole,
                LineCache& line) {
    bool ints = a.is_int() && b.is_int() && !a.grouped && !b.grouped;
    uint64_t n = ints ? a.value() : 0, den = ints ? b.value() : 0;
    std::string aw = num_words(a), bw = num_words(b);
    if (aw.empty() || bw.empty()) return;

    // "(12/15 tasks)", "a mere (1/3) of": a count in brackets of its own; the brackets go (see the end)
    bool in_paren = start >= 1 && d.punct(start - 1, '(') && d.joined(start);
    size_t paren_close = Doc::npos;
    if (in_paren) {
        size_t j = b.last + 1;
        int words = 0;
        while (j < d.size() && d.word(j) && (d[j].casing & rv::LOWER) && !d[j].newline_before && words < 3) ++j, ++words;
        if (d.punct(j, ')') && d.joined(j)) {
            bool ok = words == 0 || count_noun(d, b.last + 1);
            for (size_t k = b.last + 2; ok && k < j; ++k) ok = count_noun(d, k) || COMPLETE_NEXT.contains(d.low(k));
            if (ok) paren_close = j;
        }
    }
    size_t after = paren_close == b.last + 1 ? paren_close : b.last;  // "(1/3) of": the word after the bracket counts

    size_t nx = next_w(d, after), px = prev_w(d, start), px2 = prev_w(d, start, 2);
    std::string_view next = low_or_empty(d, nx), prev = low_or_empty(d, px), prev2 = low_or_empty(d, px2);
    // the next word only counts when it directly follows (no comma or bracket between)
    bool next_close = nx == after + 1 || (nx == after + 2 && (d.punct(after + 1, '-')));
    if (!next_close) next = std::string_view();
    bool hyphen_adj = d.punct(b.last + 1, '-') && d.joined(b.last + 1) && d.word(b.last + 2) && d.joined(b.last + 2);

    bool date_shape = ints && ((n >= 1 && n <= 12 && den >= 1 && den <= 31) || (n >= 13 && n <= 31 && den >= 1 && den <= 12));
    // "fell by 1/3", "cut by 1/2": a change by a fraction, not a date
    bool change_by = prev == "by" && CHANGE_VERBS.contains(prev2);
    // (a cue word of the clause before does not count: "7/10 at the start, 8/10 by the end")
    bool cue_close = px != Doc::npos;
    for (size_t k = px + 1; cue_close && k < start; ++k) cue_close = !d.punct(k, ',') && !d.punct(k, ';');
    bool date_cue = (DATE_PREV.contains(prev) && cue_close && !change_by) || d.punct(b.last + 1, '/') ||
                    (ints && time_after(d, b.last));
    bool proper = ints && n >= 1 && n < den && den <= 100;
    // the article "a" or "an", not the letter A ("Reviewer A | 8/10")
    bool article = (prev == "a" || prev == "an") && (d[px].casing != rv::UPPER || d.sentence_start(px)) &&
                   !(d[px].casing == rv::UPPER && d.str(px).size() == 1 && px + 1 < d.size() && d[px + 1].kind == Kind::Punct);
    // the number after an inch or foot mark or a unit abbreviation: "3/4" thick", "1/2 tsp"
    char mark = ints ? length_mark(d, b.last) : 0;
    size_t unit_last = Doc::npos;
    const std::string_view* unit = ints && !mark && !hyphen_adj ? fraction_unit(d, b.last, unit_last) : nullptr;

    bool is_frac = false;
    auto frac = [&](bool singular) -> std::string {
        is_frac = true;
        if (hyphen_adj && n != 1) {  // an adjective: "three-quarter" (the hyphen after it is swallowed)
            std::string f = adjective_fraction(n, den);
            return whole ? num_words(*whole) + " and " + f : f;
        }
        if (whole) return num_words(*whole) + " and " + mixed_tail(n, den);
        if (hyphen_adj && n == 1 && den == 2) return "half";
        return fraction_words(n, den, singular);
    };
    std::string out_of = aw + " out of " + bw;

    if (LEGAL_PREV.contains(prev) || (ints && n >= 1000 && den >= 1000)) return;
    if (ints && n >= 1 && n <= 12 && den >= 1900 && den <= 2100) return;  // 10/2026: month and year (datetime)

    // a fraction of a percent: "rates rose 1/4%", "a 1/2 % fee"
    size_t pct = d.punct(b.last + 1, '%') ? b.last + 1 : Doc::npos;
    if (pct != Doc::npos && proper && (fraction_denominator(den) || den <= 10) && free_right(d, pct)) {
        out.tokens(start, pct, frac(false) + " percent", 74);
        return;
    }
    // a fraction with its measure: "3/4" thick" three quarters inch, "2 1/2"" two and a half inches, "1/2 tsp"
    if ((mark || unit) && proper) {
        std::string f = frac(article && !whole);
        std::string u = mark == 'f' ? "foot" : mark == 'i' ? "inch" : std::string(*unit);
        size_t end = mark ? b.last + 1 : unit_last;
        bool length = mark || u == "inch" || u == "foot" || u == "yard" || u == "mile" || u == "centimeter" ||
                      u == "millimeter" || u == "kilometer";
        if (whole && !(adjectival_after(d, end + 1) && (length || article)))  // a two and a half inch hole
            u = u == "foot" ? "feet" : u == "inch" ? "inches" : u + "s";
        out.tokens(start, end, f + " " + u, 74);
        return;
    }

    std::string words;
    int score = 0;
    if (whole) {  // "1 1/2": a mixed number is a fraction whatever surrounds it
        if (!proper) return;
        words = frac(false), score = 76;
    } else if (ints && n == 24 && den == 7) {
        words = "twenty-four seven", score = 80;
    } else if (ints && n == 9 && den == 11 && (NINE_ELEVEN_NEXT.contains(next) || prev == "post" || prev == "pre")) {
        words = "nine eleven", score = 80;
    } else if (ints && n == 20 && den >= 10 && den <= 400 && den % 5 == 0 &&
               sentence_has(d, start, b.last, VISION_WORDS, 8)) {
        words = aw + " " + bw, score = 74;  // 20/20 vision
    } else if (ints && (PRESSURE.contains(prev) || (PRESSURE.contains(prev2) && PRESSURE_LINK.contains(prev)) ||
                        next == "mmhg" || (next == "mm" && d.is(nx + 1, "hg")))) {
        words = aw + " over " + bw, score = 74;
    } else if (ints && n + den == 100 && n > 0 && den > 0 &&
               (SPLIT_NEXT.contains(next) || any_prev(d, start, 4, ODDS_PREV))) {
        words = aw + " " + bw, score = 72;
    } else if (ints && (ODDS_NEXT.contains(next) || any_prev(d, start, 4, ODDS_PREV) ||
                        (den == 1 && n >= 2 && ODDS_AT.contains(prev)) ||
                        (den <= 10 && !FRACTION_WEAK_NEXT.contains(next) && !FRACTION_NEXT.contains(next) &&
                         sentence_has(d, start, b.last, ODDS_PREV, den == 1 ? 30 : 12)))) {
        words = aw + " to " + bw, score = 74;
    } else if (proper && (FRACTION_NEXT.contains(next) || hyphen_adj)) {
        bool singular = article && SINGULAR_UNITS.contains(next);
        words = frac(singular), score = 72;
    } else if (proper && FRACTION_WEAK_NEXT.contains(next) && !SCORE_VERBS.contains(prev)) {
        words = frac(false), score = 70;  // "get 3/4 of the way"
    } else if (proper && change_by && (fraction_denominator(den) || den <= 10)) {
        words = frac(false), score = 72;  // "costs fell by 1/3"
    } else if (proper && den <= 4 && (DEAL_NEXT.contains(next) || next == "past")) {
        words = frac(false), score = 70;  // "1/2 off", "1/4 past the hour"
    } else if (proper && article && (fraction_denominator(den) || den == 5 || den == 10) && !SCORE_NEXT.contains(next) &&
               !DATE_NOUN_NEXT.contains(next) && !any_prev(d, start, 5, SCORE_VERBS)) {
        words = frac(false), score = 72;  // "a 1/2 chance", "a 3/4 increase": no date takes an article
    } else if ((SCORE_PREV.contains(prev) || SCORE_NEXT.contains(next) ||
                ((prev == "was" || prev == "is" || prev == "of" || prev == "at" || prev == "were") &&
                 SCORE_PREV.contains(prev2)) ||
                any_prev(d, start, 5, SCORE_VERBS)) && (!ints || n <= den)) {
        words = out_of, score = 72;
    } else if (REVIEW_PREV.contains(prev) && (!ints || (n <= den && (den == 5 || den == 10 || den == 100))) &&
               (nx == Doc::npos || !next_close || next == "would" || next == "stars")) {
        words = out_of, score = 70;  // "Overall 9/10, would buy again", "Value for money 3/5"
    } else if (count_noun(d, nx) && next_close && (!ints || n <= den)) {
        if (date_cue) return;
        words = out_of, score = 66;
    } else if (PROGRESS_PREV.contains(prev) && (!ints || n <= den)) {
        words = out_of, score = 70;
    } else if (ints && n + den == 100 && n > 0 && den > 0 && a.ndigits() == 2 && b.ndigits() == 2) {
        words = aw + " " + bw, score = 58;
    } else if ((!ints || n <= den) && !date_cue && sibling_score(d, start, b.last)) {
        words = out_of, score = 64;
    } else if (proper && QTY_PREV.contains(prev) && !date_cue && (den <= 10 || fraction_denominator(den))) {
        words = frac(false), score = 64;
    } else if (COMPLETE_NEXT.contains(next) && (!ints || n <= den)) {
        if (date_cue) return;
        words = proper && (den <= 4) ? frac(false) : out_of, score = 62;
    } else if (paren_close == b.last + 1 && ints && n <= den && !date_cue) {
        words = out_of, score = 64;  // "Half the team (3/6) voted yes": a count in brackets of its own
    } else if (proper && den <= 10 && !date_cue && prev != "for" && prev != "to" && prev != "at" && prev != "in" &&
               (FRACTION_VERB_NEXT.contains(next) ||
                (next.size() > 4 && next.substr(next.size() - 2) == "ed" && (d[nx].casing & rv::LOWER)))) {
        words = frac(false), score = 62;  // "while 1/4 found it confusing"
    } else if (count_noun_after_adjective(d, nx) && next_close && (!ints || n <= den) && !date_cue) {
        words = out_of, score = 60;  // "4/5 active cases"
    } else if (ints && n <= den && start >= 1 && d.punct(start - 1, '[') && d.punct(b.last + 1, ']')) {
        words = out_of, score = 45;  // "[2/5] Building"
        if (d.sentence_start(start - 1)) words = rv::capital(words);  // a step counter opens its line
    } else if (!ints && !a.frac.empty() && !b.frac.empty() && !a.whole.empty() && !b.whole.empty() &&
               (a.value() < b.value() || (a.value() == b.value() && a.frac < b.frac)) && !date_cue &&
               (TRANSFER_PREV.contains(prev) || SIZE_UNITS.contains(next))) {
        words = out_of, score = 55;  // "Downloaded 1.2/4.5 GB"
    } else if (a.is_int() && a.whole == "1" && !b.frac.empty() && !b.whole.empty()) {
        words = aw + " over " + bw, score = 50;  // a sensor of 1/2.3 inch
    } else if (!ints) {  // 4.5/5
        if (a.frac.empty() || !b.frac.empty() || a.whole.empty() || b.value() > 100 || a.value() > b.value()) return;
        words = out_of, score = 60;
    } else if (n == den && n >= 2) {
        if (date_cue) return;
        words = out_of, score = date_shape ? 40 : 50;
    } else if (!date_shape && n < den && den <= 1000) {
        bool unit_frac = n == 1 && (den == 100 || den == 1000) && FRACTION_WEAK_NEXT.contains(next);
        words = (proper && fraction_denominator(den)) || unit_frac ? frac(false) : out_of, score = 45;
    } else if (date_shape && n < den && !date_cue) {
        if (start < line.first || start > line.last) {
            line.scores = line_has_scores(d, start, line.first, line.last);
        }
        if (line.scores && (den == 5 || den == 10)) words = out_of, score = 40;  // a line of scores
        else if (fraction_denominator(den)) words = frac(false), score = 18;
        else words = out_of, score = 15;
    } else {
        return;
    }
    if (words.empty()) return;
    // "3/4-inch": the adjective form takes the hyphen ("three-quarter inch"); "1/2-day" keeps it ("half-day")
    out.tokens(start, hyphen_adj && n != 1 && is_frac ? b.last + 1 : b.last, words, score);
    if (paren_close != Doc::npos && score >= 45) {
        out.tokens(start - 1, start - 1, "", 60);
        out.tokens(paren_close, paren_close, "", 60);
    }
}

// "1/10th of the budget", "3/10ths": a fraction with the ordinal written on its denominator.
bool ordinal_fraction(const Doc& doc, rv::Out& out, const Num& a, const Num& b) {
    size_t s = b.last + 1;
    if (!doc.word(s) || !doc.joined(s) || !a.is_int() || !b.is_int() || a.grouped || b.grouped) return false;
    std::string_view suf = doc.low(s);
    if (suf != "th" && suf != "ths" && suf != "rd" && suf != "rds" && suf != "nd" && suf != "nds" && suf != "st")
        return false;
    uint64_t n = a.value(), den = b.value();
    if (n < 1 || n >= den || den > 1000 || !free_right(doc, s) || !free_left(doc, a.first)) return false;
    out.tokens(a.first, s, fraction_words(n, den), 70);
    return true;
}

// "0.08 / 0.02" in a list of values: and; "BP 130 / 85": over. i starts the first number, e ends it.
void spaced_slash(const Doc& doc, rv::Out& out, size_t i, size_t e) {
    if (!doc.punct(e + 1, '/') || !doc[e + 1].space_before || !doc.number(e + 2) || doc.joined(e + 2)) return;
    Num a = num_at(doc, i);
    size_t s = a.last + 1;
    Num b = num_at(doc, s + 1);
    bool chain = (a.first >= 1 && doc.punct(a.first - 1, '/')) || doc.punct(b.last + 1, '/');
    std::string aw = num_words(a), bw = num_words(b);
    if (chain || !free_left(doc, a.first) || !free_right(doc, b.last) || aw.empty() || bw.empty()) return;
    size_t px = prev_w(doc, a.first), nx = next_w(doc, b.last);
    bool pressure = a.is_int() && b.is_int() &&
                    (PRESSURE.contains(low_or_empty(doc, px)) || PRESSURE.contains(low_or_empty(doc, prev_w(doc, a.first, 2))) ||
                     (nx == b.last + 1 && doc.is(nx, "mmhg")));
    if (a.whole == "24" && b.whole == "7" && a.frac.empty() && b.frac.empty())
        out.tokens(a.first, b.last, "twenty-four seven", 70);
    else
        out.tokens(a.first, b.last, aw + (pressure ? " over " : " and ") + bw, pressure ? 70 : 50);
}

// "f/2.8": an aperture; i is the number after the slash. Returns the last token read.
size_t aperture(const Doc& doc, rv::Out& out, size_t i) {
    size_t f = i - 2;
    if (!doc.word(f) || doc.low(f) != "f" || !free_left(doc, f)) return i;
    Num n = num_at(doc, i);
    if (!n.ok || n.whole.empty() || !free_right(doc, n.last) || n.ndigits() > 2) return i;
    std::string w = num_words(n);
    if (w.empty()) return i;
    out.tokens(f, n.last, std::string(doc.str(f)) + " " + w, 70);
    return n.last;
}

RV_RULE("numbers_context", slash_numbers) {
    LineCache line;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        if (i >= 2 && doc.joined(i) && is_slash(doc, i - 1) && doc.joined(i - 1) && doc.word(i - 2)) {
            i = aperture(doc, out, i);
            continue;
        }
        size_t e = num_end(doc, i);
        if (!is_slash(doc, e + 1) || !doc.joined(e + 1) || !doc.number(e + 2) || !doc.joined(e + 2)) {
            spaced_slash(doc, out, i, e);
            i = e;
            continue;
        }
        Num a = num_at(doc, i);
        size_t s = a.last + 1;
        Num b = num_at(doc, s + 1);
        if (b.whole.empty()) continue;
        // part of a longer chain (1/2/2026, a/b/c paths, 2026/1/2) or of a larger token
        bool chain_before = a.first >= 1 && doc.joined(a.first) && is_slash(doc, a.first - 1);
        bool chain_after = is_slash(doc, b.last + 1) && doc.joined(b.last + 1);
        bool hyphen_after = doc.punct(b.last + 1, '-') && doc.joined(b.last + 1) && doc.word(b.last + 2) &&
                            doc.joined(b.last + 2);
        // "24/7/365": open all day, every day of the year
        if (chain_after && !chain_before && a.whole == "24" && a.frac.empty() && b.whole == "7" && b.frac.empty() &&
            doc.number(b.last + 2) && doc.str(b.last + 2) == "365" && doc.joined(b.last + 2) &&
            free_right(doc, b.last + 2) && free_left(doc, a.first)) {
            out.tokens(a.first, b.last + 2, "twenty-four seven three sixty-five", 80);
            i = b.last + 2;
            continue;
        }
        if (!chain_before && !chain_after && ordinal_fraction(doc, out, a, b)) {
            i = b.last + 1;
            continue;
        }
        bool mark_after = length_mark(doc, b.last) != 0 || doc.punct(b.last + 1, '%');
        if (chain_before || chain_after || !(free_right(doc, b.last) || hyphen_after || mark_after)) {
            i = b.last;
            continue;
        }
        // a mixed number: "1 1/2" or "1-1/2"
        const Num* whole = nullptr;
        Num w;
        size_t start = a.first;
        if (a.is_int() && a.first >= 1) {
            size_t k = a.first - 1;
            bool spaced = !doc.joined(a.first) && doc.number(k);
            bool hyph = doc.joined(a.first) && doc.punct(k, '-') && doc.joined(k) && k >= 1 && doc.number(k - 1);
            size_t wi = spaced ? k : hyph ? k - 1 : Doc::npos;
            if (wi != Doc::npos && !(wi >= 1 && doc.joined(wi) && doc.number(wi - 1))) {
                // find where that number starts
                size_t f = wi;
                while (f >= 2 && doc.joined(f) && doc.punct(f - 1, ',') && doc.number(f - 2)) f -= 2;
                w = num_at(doc, f);
                if (w.ok && w.last == wi && w.is_int() && !w.grouped && w.ndigits() <= 3 && free_left(doc, w.first) &&
                    a.value() < b.value() && a.value() >= 1) {
                    whole = &w;
                    start = w.first;
                }
            }
        }
        // a value after '=' ("retries=3/3") is read like any other
        bool after_equals = doc.joined(a.first) && a.first >= 1 && doc.punct(a.first - 1, '=');
        if (!whole && !free_left(doc, a.first) && !after_equals) {
            i = b.last;
            continue;
        }
        slash_pair(doc, out, a, b, start, whole, line);
        i = b.last;
    }
}



// ½ ¼ ¾ ⅓ ...: one half, three quarters; after a number: one and a half.
struct Vulgar {
    uint32_t cp;
    uint8_t n, d;
};
const Vulgar VULGAR[] = {{0xBD, 1, 2},   {0xBC, 1, 4},   {0xBE, 3, 4},   {0x2153, 1, 3}, {0x2154, 2, 3},
                         {0x2155, 1, 5}, {0x2156, 2, 5}, {0x2157, 3, 5}, {0x2158, 4, 5}, {0x2159, 1, 6},
                         {0x215A, 5, 6}, {0x2150, 1, 7}, {0x215B, 1, 8}, {0x215C, 3, 8}, {0x215D, 5, 8},
                         {0x215E, 7, 8}, {0x2151, 1, 9}, {0x2152, 1, 10}};

int super_digit(uint32_t c) {
    if (c == 0xB9) return 1;
    if (c == 0xB2) return 2;
    if (c == 0xB3) return 3;
    if (c == 0x2070) return 0;
    if (c >= 0x2074 && c <= 0x2079) return static_cast<int>(c - 0x2070);
    return -1;
}
int sub_digit(uint32_t c) { return c >= 0x2080 && c <= 0x2089 ? static_cast<int>(c - 0x2080) : -1; }

// "¹⁄₂", "³/₄": a fraction in superscript and subscript digits. Returns the last token read.
size_t script_fraction(const Doc& doc, rv::Out& out, size_t i) {
    uint64_t n = 0, den = 0;
    size_t j = i;
    int digits = 0;
    while (j < doc.size() && doc[j].kind == Kind::Punct && super_digit(doc[j].cp) >= 0 && digits < 3 &&
           (j == i || doc.joined(j)))
        n = n * 10 + static_cast<uint64_t>(super_digit(doc[j].cp)), ++j, ++digits;
    if (!digits || !(doc.punct(j, 0x2044) || doc.punct(j, '/')) || !doc.joined(j)) return i;
    size_t slash = j++;
    digits = 0;
    while (j < doc.size() && doc[j].kind == Kind::Punct && sub_digit(doc[j].cp) >= 0 && digits < 3 && doc.joined(j))
        den = den * 10 + static_cast<uint64_t>(sub_digit(doc[j].cp)), ++j, ++digits;
    if (!digits || j == slash + 1 || n == 0 || n >= den || !free_right(doc, j - 1)) return i;
    if (doc.joined(i) && i >= 1 && doc[i - 1].kind == Kind::Word) return i;  // x¹/₂ is algebra
    std::string words = fraction_words(n, den);
    if (i >= 1 && doc.number(i - 1) && doc.joined(i)) {  // "1¹⁄₂": one and a half
        Num w = num_at(doc, i - 1);
        if (w.ok && w.last == i - 1 && w.is_int() && free_left(doc, w.first) && !num_words(w).empty()) {
            out.tokens(w.first, j - 1, num_words(w) + " and " + mixed_tail(n, den), 76);
            return j - 1;
        }
    }
    out.tokens(i, j - 1, words, 70);
    return j - 1;
}

RV_RULE("numbers_context", vulgar_fractions) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].kind == Kind::Punct && super_digit(doc[i].cp) >= 0) {
            i = script_fraction(doc, out, i);
            continue;
        }
        if (doc[i].kind != Kind::Punct || doc[i].cp < 0xBC) continue;
        const Vulgar* v = nullptr;
        for (const auto& x : VULGAR)
            if (x.cp == doc[i].cp) v = &x;
        if (!v) continue;
        bool hyphen_adj = doc.punct(i + 1, '-') && doc.joined(i + 1) && doc.word(i + 2) && doc.joined(i + 2);
        char mark = length_mark(doc, i);
        size_t unit_last = Doc::npos;
        const std::string_view* unit = mark || hyphen_adj ? nullptr : fraction_unit(doc, i, unit_last);
        bool pct = doc.punct(i + 1, '%') && free_right(doc, i + 1);  // "a ½% fee"
        if (!(free_right(doc, i) || hyphen_adj || mark || pct)) continue;
        if (pct) {
            bool whole_before = i >= 1 && doc.number(i - 1) && doc.joined(i);
            if (whole_before) {
                Num w = num_at(doc, i - 1);
                if (w.ok && w.last == i - 1 && w.is_int() && free_left(doc, w.first) && !num_words(w).empty())
                    out.tokens(w.first, i + 1, num_words(w) + " and " + mixed_tail(v->n, v->d) + " percent", 76);
            } else if (!doc.joined(i) || free_left(doc, i)) {
                out.tokens(i, i + 1, fraction_words(v->n, v->d) + " percent", 74);
            }
            continue;
        }
        size_t last = mark ? i + 1 : unit ? unit_last : i;
        // a whole number joined or one space before
        if (i >= 1 && doc.number(i - 1)) {
            size_t f = i - 1;
            while (f >= 2 && doc.joined(f) && doc.punct(f - 1, ',') && doc.number(f - 2)) f -= 2;
            Num w = num_at(doc, f);
            if (w.ok && w.last == i - 1 && w.is_int() && free_left(doc, w.first)) {
                std::string ww = num_words(w);
                if (ww.empty()) continue;
                ww += " and " + mixed_tail(v->n, v->d);
                if (mark || unit) {  // "2½" thick" two and a half inches
                    std::string u = mark == 'f' ? "foot" : mark == 'i' ? "inch" : std::string(*unit);
                    std::string_view pw = low_or_empty(doc, prev_w(doc, w.first));
                    bool adj = adjectival_after(doc, last + 1) &&
                               (mark || u == "inch" || u == "foot" || pw == "a" || pw == "an");
                    if (!adj) u = u == "foot" ? "feet" : u == "inch" ? "inches" : u + "s";
                    out.tokens(w.first, last, ww + " " + u, 76);
                } else {
                    out.tokens(w.first, i, ww, 76);
                }
                continue;
            }
        }
        if (doc.joined(i) && !free_left(doc, i)) continue;
        size_t px = prev_w(doc, i), nx = next_w(doc, i);
        std::string_view prev = low_or_empty(doc, px), next = nx == i + 1 ? low_or_empty(doc, nx) : std::string_view();
        if (mark || unit) {  // "¾" plywood" three quarters inch, "½ tsp" one half teaspoon
            std::string u = mark == 'f' ? "foot" : mark == 'i' ? "inch" : std::string(*unit);
            out.tokens(i, last, fraction_words(v->n, v->d, prev == "a" || prev == "an") + " " + u, 74);
            continue;
        }
        std::string words;
        if (hyphen_adj && v->n == 1 && v->d == 2) words = "half";
        else if (hyphen_adj && v->n != 1) words = adjective_fraction(v->n, v->d);
        else words = fraction_words(v->n, v->d, (prev == "a" || prev == "an") && SINGULAR_UNITS.contains(next));
        out.tokens(i, hyphen_adj && v->n != 1 ? i + 1 : i, words, 66);
    }
}

}  // namespace
