// A colon between numbers that is not a clock time or a verse: a ratio "to" (a 3:1 ratio of sand, 1:4 scale,
// 9/2 odds) or an aspect ratio "by" (16:9, a 4:3 display, 9:16 portrait). Clock times and verses belong to
// datetime and roman_citations: a time-shaped pair (10:50) is proposed only with a ratio or display cue
// (16:10 among screens, a ratio of 1:12). Also the ratio sign (16∶9) and a spaced colon after a ratio word.
#include "rules/numbers_context/common.h"

namespace {

using namespace nctx;

const rv::WordSet DISPLAY_NEXT{
    "aspect", "backdrop", "banner", "banners", "bezels", "camera", "cameras", "canvas", "card", "cards", "chart",
    "cinema", "cinematic", "crop", "crops", "curved", "cut", "cuts", "dashboard", "design", "deck", "display",
    "displays", "export", "exports", "feed", "film", "footage", "format", "formats", "frame", "frames",
    "framing", "graphic", "graphics", "header", "image", "images", "input", "kiosk", "kiosks", "landscape",
    "laptop", "layout", "layouts", "lens", "letterbox", "mockup", "mode", "model", "monitor", "monitors",
    "mount", "output", "panel", "panels", "panorama", "photo", "photos", "picture", "pictures", "portrait",
    "poster", "posters", "presentation", "preview", "previews", "print", "prints", "projector", "projectors",
    "resolution", "scanner", "screen", "screens", "sensor", "sheet", "sheets", "slide", "slides", "source",
    "square", "stories", "story", "tablet", "tablets", "television", "template", "thumbnail",
    "thumbnails", "touch", "trailer", "trailers", "teaser", "clip", "clips", "reel", "tv", "tvs", "ultrawide",
    "unit", "units", "variant", "version", "vertical", "video",
    "videos", "view", "wall", "widescreen"};

const rv::WordSet DISPLAY_SENT{
    "aspect", "banner", "banners", "camera", "cameras", "canvas", "canvases", "cinema", "cinematic", "clip", "clips",
    "crop", "cropped", "cropping", "trailer", "teaser", "framing", "widescreen", "ultrawide", "kiosk", "kiosks",
    "display", "displays", "film", "films", "footage", "format", "formats", "frame", "frames", "image",
    "images", "kiosk", "kiosks", "landscape", "layout", "layouts", "lens", "letterbox", "letterboxed",
    "mockup", "mockups", "monitor", "monitors", "panel", "panels", "phone", "phones", "photo", "photos",
    "picture", "pictures", "pixels", "portrait", "poster", "posters", "print", "prints", "projector",
    "projectors", "printed", "resolution", "screen", "screens", "signage", "slides", "tablet", "tablets",
    "television",
    "thumbnail", "thumbnails", "tv", "tvs", "ultrawide", "video", "videos", "widescreen"};

// The display words that make even a time-shaped pair (16:10, 9:16) an aspect ratio.
const rv::WordSet DISPLAY_STRICT{"aspect", "banner", "banners", "canvas", "crop", "display", "displays", "format",
                                 "frame", "frames", "image", "images", "landscape", "layout", "monitor", "monitors",
                                 "panel", "panels", "photo", "photos", "portrait", "poster", "posters", "print",
                                 "prints", "projector", "projectors", "screen", "screens", "tablet", "tablets",
                                 "thumbnail", "thumbnails", "tv", "tvs", "ultrawide", "video", "widescreen"};

const rv::WordSet RATIO_NEXT{"advantage", "basis", "blend", "chance", "comparison", "compression", "contrast",
                             "decision", "defeat", "dilution", "edge", "favorite", "gear", "lead", "loss", "majority",
                             "margin", "mix", "mixture", "odds", "preference", "reduction", "sampling", "scale",
                             "split", "victory", "vote", "win", "zoom"};

const rv::WordSet RATIO_PREV{"beat", "defeated", "dilute", "diluted", "dilution", "drew", "final", "lead", "led",
                             "lost", "mix", "mixed", "mixing", "odds", "proportion", "proportions", "ratio", "ratios",
                             "scale", "score", "scored", "split", "tie", "tied", "trailing", "won"};

const rv::WordSet AFTER_RATIO_TO{"of", "between", "for", "in", "compared", "against", "versus", "vs", "from"};

// Words in the sentence before a time-shaped pair that make it a ratio: "an insulin-to-carb ratio of 1:12 at
// breakfast and 1:15 at dinner", "1 part bleach to 10 parts water (1:10)".
const rv::WordSet RATIO_SENT{"dilute", "diluted", "dilution", "part", "parts", "proportion", "ratio", "ratios"};
const rv::WordSet RATIO_NEAR{"dilute", "diluted", "dilution", "proportion", "ratio", "ratios"};
const rv::WordSet RATIO_FAR{"dilution", "mix", "mixture", "proportion", "proportions", "ratio", "ratios", "recipe"};

// Before "at": verbs and nouns that put an aspect ratio there ("runs at 16:10", "the console at 9:16").
const rv::WordSet ASPECT_AT{"console", "display", "displayed", "displays", "kiosk", "monitor", "output", "outputs",
                            "panel", "rendered", "renders", "run", "running", "runs", "screen", "set", "shoot",
                            "shoots", "shot", "shown", "ships", "sits", "stays", "tablet", "wall"};

// Strong clock cues that keep a time-shaped pair a time even in a ratio or display sentence.
const rv::WordSet STRONG_TIME_PREV{"until", "till", "til", "from", "since", "between", "before", "after", "around",
                                   "by", "noon", "starts", "opens", "closes", "ends", "begins", "during"};

bool is_colon(const Doc& d, size_t j) { return d.punct(j, ':') || d.punct(j, 0x2236); }

// Another colon pair in the sentence that no clock can read (16:9, 4:3, 2.39:1): "16:9 and 9:16 canvases".
bool sibling_ratio(const Doc& d, size_t first, size_t last) {
    auto ratio_at = [&](size_t j) {
        if (!is_colon(d, j) || j == 0 || !d.number(j - 1) || !d.number(j + 1) || !d.joined(j) || !d.joined(j + 1))
            return false;
        if (d.punct(j + 2, ':') && d.joined(j + 2)) return false;
        if (d.punct(j + 2, '.') && d.joined(j + 2) && d.number(j + 3) && d.joined(j + 3)) return true;  // 1:0.8
        return d.str(j + 1).size() == 1 || d.punct(j, 0x2236);
    };
    int seen = 0;
    for (size_t j = first; j-- > 0 && seen < 30; ++seen) {
        if (d[j + 1].newline_before || is_sentence_stop(d, j)) break;
        if (ratio_at(j)) return true;
    }
    seen = 0;
    for (size_t j = last + 1; j < d.size() && seen < 30; ++j, ++seen) {
        if (d[j].newline_before || is_sentence_stop(d, j)) break;
        if (ratio_at(j)) return true;
    }
    return false;
}

// Another clock time in the sentence (9:00, 12:05, 4:30): then a pair like 23:10 is a time too.
bool sibling_time(const Doc& d, size_t first, size_t last) {
    auto time_at = [&](size_t j) {
        if (!d.punct(j, ':') || j == 0 || !d.number(j - 1) || !d.number(j + 1) || !d.joined(j) || !d.joined(j + 1))
            return false;
        std::string_view b = d.str(j + 1);
        return b.size() == 2 && b != "10" && b != "16" && b[0] <= '5' && d.str(j - 1).size() <= 2;
    };
    int seen = 0;
    for (size_t j = first; j-- > 0 && seen < 40; ++seen) {
        if (d[j + 1].newline_before || is_sentence_stop(d, j)) break;
        if (time_at(j)) return true;
    }
    seen = 0;
    for (size_t j = last + 1; j < d.size() && seen < 40; ++j, ++seen) {
        if (d[j].newline_before || is_sentence_stop(d, j)) break;
        if (time_at(j)) return true;
    }
    return false;
}

const rv::WordSet TIME_NOUN_NEXT{"call", "class", "deadline", "departure", "flight", "kickoff", "meeting", "session",
                                 "shift", "show", "slot", "start", "train", "window"};
const rv::WordSet CLOCK_WORDS{"clock", "clocks", "time", "timer", "timestamp", "watch"};

// A word of the set among the n words before token i in its sentence.
bool sentence_before(const Doc& d, size_t i, const rv::WordSet& set, int n) {
    for (int k = 1; k <= n; ++k) {
        size_t j = prev_w(d, i, k);
        if (j == Doc::npos) return false;
        if (d.word(j) && set.contains(d.low(j))) return true;
    }
    return false;
}

const rv::WordSet TIME_NEXT{"am", "pm", "utc", "gmt", "et", "est", "edt", "pt", "pst", "pdt", "ct", "cst", "cdt",
                            "mt", "mst", "mdt", "cet", "cest", "bst", "ist", "jst", "aest", "hrs", "h", "sharp",
                            "o'clock", "local", "today", "tomorrow", "tonight", "daily"};
const rv::WordSet TIME_PREV{"at", "by", "until", "till", "til", "from", "around", "after", "before", "since",
                            "between", "noon", "starts", "start", "opens", "closes", "ends", "begins"};

const rv::WordSet SCRIPTURE{
    "acts", "amos", "chapter", "chronicles", "colossians", "corinthians", "daniel", "deuteronomy",
    "ecclesiastes", "ephesians", "esther", "exodus", "ezekiel", "ezra", "galatians", "genesis", "habakkuk",
    "haggai", "hebrews", "hosea", "isaiah", "james", "jeremiah", "job", "joel", "john", "jonah", "joshua",
    "jude", "judges", "kings", "lamentations", "leviticus", "luke", "malachi", "mark", "matthew", "micah",
    "nahum", "nehemiah", "numbers", "obadiah", "peter", "philemon", "philippians", "proverbs", "psalm",
    "psalms", "quran", "revelation", "revelations", "romans", "ruth", "samuel", "song", "sura", "surah",
    "thessalonians", "timothy", "titus", "verse", "verses", "zechariah", "zephaniah"};

struct Pair {
    const char* a;
    const char* b;
};
const Pair COMMON_ASPECT[] = {{"16", "9"}, {"4", "3"},  {"21", "9"}, {"16", "10"}, {"3", "2"},  {"9", "16"},
                              {"32", "9"}, {"5", "4"},  {"18", "9"}, {"19", "9"},  {"20", "9"}, {"3", "4"},
                              {"4", "5"},  {"2.39", "1"}, {"2.35", "1"}, {"1.85", "1"}, {"2.4", "1"}};

std::string written(const Num& n) { return n.whole + (n.frac.empty() ? "" : "." + n.frac); }

RV_RULE("numbers_context", colon_ratio) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        size_t e = num_end(doc, i);
        // "2:1", "16∶9" (the ratio sign), and "2 : 1" with a space on both sides
        bool spaced = is_colon(doc, e + 1) && doc[e + 1].space_before && doc.number(e + 2) && doc[e + 2].space_before &&
                      !doc[e + 1].newline_before && !doc[e + 2].newline_before;
        if (!is_colon(doc, e + 1) || !doc.number(e + 2) || (!spaced && (!doc.joined(e + 1) || !doc.joined(e + 2)))) {
            i = e;
            continue;
        }
        bool ratio_sign = doc.punct(e + 1, 0x2236);
        Num a = num_at(doc, i);
        size_t s = a.last + 1;
        Num b = num_at(doc, s + 1);
        i = b.last;
        bool chain = (a.first >= 1 && doc.joined(a.first) && is_colon(doc, a.first - 1)) ||
                     (is_colon(doc, b.last + 1) && doc.number(b.last + 2) && doc.joined(b.last + 2));
        bool power = doc.joined(b.last + 1) && (doc.punct(b.last + 1, '^') || doc.punct(b.last + 1, 0xB2) ||
                                                doc.punct(b.last + 1, 0xB3) || doc[b.last + 1].cp >= 0x2070);
        if (chain || power || a.whole.empty() || b.whole.empty() || !free_left(doc, a.first) ||
            !free_right(doc, b.last))
            continue;
        std::string aw = num_words(a), bw = num_words(b);
        if (aw.empty() || bw.empty()) continue;

        size_t px = prev_w(doc, a.first), nx = next_w(doc, b.last);
        std::string_view prev = low_or_empty(doc, px);
        std::string_view next = nx == b.last + 1 ? low_or_empty(doc, nx) : std::string_view();
        if (SCRIPTURE.contains(prev)) continue;
        if (TIME_NEXT.contains(next) || ((next == "a" || next == "p") && doc.punct(nx + 1, '.'))) continue;

        if (spaced) {  // "the ratio 2 : 1", "mixed 3 : 1": only with a ratio or display word
            bool display = DISPLAY_NEXT.contains(next) || sentence_has(doc, a.first, b.last, DISPLAY_STRICT, 8);
            bool ratio = RATIO_NEXT.contains(next) || next == "ratio" || sentence_before(doc, a.first, RATIO_PREV, 3) ||
                         sentence_before(doc, a.first, RATIO_SENT, 3);
            if (ratio && !display) out.tokens(a.first, b.last, aw + " to " + bw, 66);
            else if (display && !ratio) out.tokens(a.first, b.last, aw + " by " + bw, 66);
            continue;
        }

        // a clock time or minutes and seconds (10:50, 52:10): only adjacent cues make it a ratio
        bool time_shaped = !ratio_sign && a.is_int() && b.is_int() && b.ndigits() == 2 && a.ndigits() <= 2 &&
                           b.value() <= 59;
        if (time_shaped && !STRONG_TIME_PREV.contains(prev)) {
            // 16:10, 9:16, 21:10 among screens and other ratios are aspect ratios ("runs at 16:10", "9:16 canvases")
            uint64_t av = a.value();
            bool aspect_shape = (b.whole == "10" && av >= 13 && av <= 37) || (av == 9 && b.whole == "16");
            bool at_ok = prev != "at" || ASPECT_AT.contains(low_or_empty(doc, prev_w(doc, a.first, 2)));
            if (aspect_shape && at_ok && !TIME_NOUN_NEXT.contains(next) && !sibling_time(doc, a.first, b.last) &&
                !sentence_before(doc, a.first, CLOCK_WORDS, 3) &&
                (sentence_has(doc, a.first, b.last, DISPLAY_SENT) || DISPLAY_NEXT.contains(next) ||
                 sibling_ratio(doc, a.first, b.last))) {
                out.tokens(a.first, b.last, aw + " by " + bw, 72);
                continue;
            }
            // "a ratio of 1:12 at breakfast and 1:15 at dinner", "1 part bleach to 10 parts water (1:10)"
            if (prev != "at" && ((av == 1 && sentence_before(doc, a.first, RATIO_SENT, 12)) ||
                                 sentence_before(doc, a.first, RATIO_NEAR, 3))) {
                out.tokens(a.first, b.last, aw + " to " + bw, 72);
                continue;
            }
        }
        if (time_shaped && TIME_PREV.contains(prev)) continue;
        if (time_shaped && a.value() <= 24 && (next == "on" || next == "and" || next == "to")) continue;
        if (a.whole == "50" && b.whole == "50" && a.frac.empty() && b.frac.empty() && next != "ratio") {
            out.tokens(a.first, b.last, "fifty fifty", 66);  // "a 50:50 decision"
            continue;
        }

        // "a 17:10 source" (an article before) takes every display word; a bare time-shaped pair only the strict ones
        bool display_next = time_shaped && prev != "a" && prev != "an" ? DISPLAY_STRICT.contains(next)
                                                                       : DISPLAY_NEXT.contains(next);
        bool ratio_next = RATIO_NEXT.contains(next);
        bool ratio_word_next = next == "ratio" || next == "ratios";
        // "the map reads 2:1 for the scale": a ratio word a few words on, before any screen word
        if (!ratio_next && !ratio_word_next && !display_next && !time_shaped) {
            for (int k = 1; k <= 3; ++k) {
                size_t w = next_w(doc, b.last, k);
                if (w == Doc::npos) break;
                if (doc.is(w, "for") || doc.is(w, "the") || doc.is(w, "on") || doc.is(w, "of")) continue;
                ratio_next = doc.is(w, "scale") || doc.is(w, "ratio") || doc.is(w, "odds") || doc.is(w, "dilution");
                break;
            }
        }
        // "the final score was 21:14", "a ratio of 10:30"
        bool be_ratio = (prev == "was" || prev == "is" || prev == "were" || prev == "of") &&
                        RATIO_PREV.contains(low_or_empty(doc, prev_w(doc, a.first, 2)));
        bool ratio_prev = time_shaped ? RATIO_PREV.contains(prev) || be_ratio : any_prev(doc, a.first, 3, RATIO_PREV);
        // "the aspect ratio was 16:9", "display ratio 4:3": a display ratio is "by"
        bool display_ratio = false;
        for (int k = 1; k <= 3; ++k) {
            size_t r = prev_w(doc, a.first, k);
            if (r == Doc::npos) break;
            if (doc.is(r, "ratio") || doc.is(r, "ratios")) {
                std::string_view before = low_or_empty(doc, prev_w(doc, r));
                display_ratio = before == "aspect" || before == "display";
                break;
            }
        }
        bool display_sent = sentence_has(doc, a.first, b.last, DISPLAY_SENT);
        bool common = false;
        for (const auto& p : COMMON_ASPECT)
            if (written(a) == p.a && written(b) == p.b) common = true;

        const char* how = nullptr;
        int score = 0;
        if (display_next || display_ratio) {
            how = " by ", score = 70;
        } else if (ratio_word_next) {
            bool of_after = AFTER_RATIO_TO.contains(low_or_empty(doc, nx + 1 < doc.size() && doc.word(nx + 1) ? nx + 1 : Doc::npos));
            if (display_sent && !of_after) how = " by ", score = 70;
            else how = " to ", score = 70;
        } else if (ratio_next || ratio_prev) {
            how = " to ", score = 70;
        } else if (!display_sent && !time_shaped && sentence_before(doc, a.first, RATIO_FAR, 12)) {
            how = " to ", score = 60;  // "The ratio of acid to base in the recipe is 4:5"
        } else if (display_sent) {
            how = " by ", score = time_shaped ? 0 : 60;
        } else if (common && !time_shaped) {
            how = " by ", score = 50;
        } else if (!time_shaped) {
            how = " to ", score = ratio_sign ? 60 : 45;
        }
        if (!how || score == 0) continue;
        out.tokens(a.first, b.last, aw + how + bw, score);
    }
}

}  // namespace
