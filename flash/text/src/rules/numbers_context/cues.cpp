// Numbers whose reading comes from the word before them: pairs after room, suite, route, highway, exit and port
// (Room 115 one fifteen, Route 101 one oh one, port 8080 eighty eighty, port 443 four four three), "#12" and
// "No. 5", "№ 5" and "# of" as "number", win-loss records (a 7-3 record: seven and three), game scores (won 3-2:
// three to two) and splits (a 50-50 chance).
#include "rules/numbers_context/common.h"

namespace {

using namespace nctx;

const rv::WordSet ABBR_CUES{"apt", "bldg", "expy", "fwy", "hwy", "pkwy", "rm", "rte", "ste"};
const rv::WordSet PAIR_CUES{"apartment", "apt", "bldg", "exit", "expy", "fwy", "hangar", "highway", "hwy",
                            "interstate", "pkwy", "room", "rooms", "rm", "route", "routes", "rte", "sector", "ste",
                            "suite", "suites", "tail"};

const rv::WordSet MUSIC{"ballade", "cantata", "concerto", "concertos", "divertimento", "etude", "fugue", "impromptu",
                        "mass", "mazurka", "nocturne", "overture", "partita", "polonaise", "prelude", "quartet",
                        "quintet", "rhapsody", "scherzo", "serenade", "sextet", "sonata", "sonatas", "symphony",
                        "trio", "waltz", "\xC3\x89tude", "\xC3\xA9tude"};

const rv::WordSet SPORTS_WORDS{"beat", "beats", "conference", "defeated", "division", "face", "faces", "game", "games",
                               "host", "hosts", "league", "match", "matchup", "playoff", "playoffs", "season",
                               "standings", "visit", "visits", "win", "wins", "won"};
const rv::WordSet SPLIT_WORDS{"call", "chance", "chances", "deal", "mix", "odds", "proposition", "ratio", "shot",
                              "split", "vote"};
const rv::WordSet RECORD_NEXT{"record", "records", "mark", "start", "finish"};
const rv::WordSet GAME_PREV{"beat", "beating", "beats", "defeat", "defeated", "defeating", "defeats", "drew",
                            "edged", "ended", "final", "finished", "game", "lead", "leading", "leads", "led", "lose",
                            "loses", "losing", "loss", "lost", "match", "routed", "score", "scored", "scoreline",
                            "scores", "thrashed", "tie", "tied", "trail", "trailed", "trailing", "victory", "win",
                            "winning", "wins", "won"};
const rv::WordSet GAME_NEXT{"win", "victory", "loss", "defeat", "lead", "final", "scoreline", "drubbing", "rout",
                            "thrashing", "upset", "comeback", "triumph", "decision", "verdict"};

// The digits of an integer token that stands alone (no grouping, decimals or letters around it).
bool plain_int(const Doc& d, size_t i) {
    if (!d.number(i)) return false;
    if (d.joined(i + 1) && (d.punct(i + 1, '.') || d.punct(i + 1, ',') || d.punct(i + 1, ':')) && d.number(i + 2) &&
        d.joined(i + 2))
        return false;
    return true;
}

void pairs_after_cue(const Doc& doc, rv::Out& out, size_t i) {
    if (i == 0 || !free_left(doc, i)) return;
    std::string_view s = doc.str(i);
    if (s.size() < 3 || s.size() > 4 || s[0] == '0') return;
    // "Room 12A" keeps its letter; anything else joined after the number is not a room number
    bool letter_after = doc.joined(i + 1) && doc.word(i + 1) && (doc[i + 1].casing & rv::UPPER) &&
                        doc.str(i + 1).size() == 1;
    if (!free_right(doc, i) && !letter_after) return;
    if (doc.punct(i - 1, '#') && doc.joined(i)) return;
    // the cue stands right before the number, or before "number" / "No." ("Room No. 412"); "is" / "was"
    // only when the number ends the clause ("the room is 203."; "the room is 412 m²" is an area)
    size_t j = i - 1;
    if (doc.is(j, "to") && j >= 2 && doc.is(j - 1, "set") && doc.word(j - 2)) j -= 2;  // "port set to 443"
    else if (doc.word(j) && doc.is(j, "number") && j > 0) --j;
    else if (doc.punct(j, '.') && j >= 2 && doc.is(j - 1, "no")) j -= 2;
    else if (doc.punct(j, '.') && j >= 1 && doc.word(j - 1) && doc.joined(j) && ABBR_CUES.contains(doc.low(j - 1)))
        --j;  // "Ste. 400", "Apt. 1204"
    else if ((doc.is(j, "is") || doc.is(j, "was")) && j > 0 && (i + 1 >= doc.size() || doc.punct(i + 1, '.') ||
                                                              doc.punct(i + 1, ',') || doc.punct(i + 1, ';')))
        --j;
    if (!doc.word(j) || !doc[i].space_before) return;
    std::string_view cue = doc.low(j);
    uint64_t v = 0;
    for (char c : s) v = v * 10 + static_cast<uint64_t>(c - '0');
    if (PAIR_CUES.contains(cue)) {
        bool three_only = cue == "tail" || cue == "sector" || cue == "hangar";
        if (three_only && s.size() != 3) return;
        out.tokens(i, i, pair_words(v), 70);
    } else if (cue == "port" || cue == "ports") {
        out.tokens(i, i, s.size() == 3 ? digit_words(s) : pair_words(v), 70);
    }
}

const rv::WordSet COMPASS{"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west"};

// "the 101 Fwy.", "the 405 freeway": a road number before its road word, in pairs.
const rv::WordSet ROAD_AFTER{"expressway", "expy", "freeway", "fwy", "highway", "hwy", "interstate", "motorway",
                             "pkwy", "tollway", "turnpike"};

void road_before(const Doc& doc, rv::Out& out, size_t i) {
    if (i == 0 || !doc.word(i + 1) || !doc[i + 1].space_before) return;
    size_t r = i + 1;
    if (COMPASS.contains(doc.low(r)) && doc[r].casing != rv::LOWER) {  // "the 101 N Fwy."
        ++r;
        if (doc.punct(r, '.') && doc.joined(r)) ++r;
    }
    if (!doc.word(r) || !ROAD_AFTER.contains(doc.low(r))) return;
    if (!free_left(doc, i) || !free_right(doc, i)) return;
    std::string_view s = doc.str(i);
    if (s.size() != 3 || s[0] == '0' || !doc.is(i - 1, "the")) return;
    uint64_t v = 0;
    for (char c : s) v = v * 10 + static_cast<uint64_t>(c - '0');
    out.tokens(i, i, pair_words(v), 66);
}

// House numbers before a street name: "145 King St." one forty-five, "1428 N. Main St." one four two eight; a
// run of rising digits is a placeholder read digit by digit ("123 Elm St." one two three).
const rv::WordSet STREET_WORDS{"ave", "avenue", "blvd", "boulevard", "cir", "circle", "court", "ct", "dr", "drive",
                               "lane", "ln", "parkway", "pkwy", "pl", "place", "rd", "road", "sq", "square", "st",
                               "street", "ter", "terrace", "way"};

void house_numbers(const Doc& doc, rv::Out& out, size_t i) {
    if (i + 2 >= doc.size() || !free_left(doc, i) || !free_right(doc, i)) return;
    std::string_view s = doc.str(i);
    if (s.size() < 3 || s.size() > 5 || s[0] == '0') return;
    size_t k = i + 1;
    if (doc.word(k) && COMPASS.contains(doc.low(k)) && doc[k].casing != rv::LOWER) {
        ++k;
        if (doc.punct(k, '.') && doc.joined(k)) ++k;
    }
    bool street = false;
    for (int w = 0; w < 3 && doc.word(k) && doc[k].casing == rv::TITLE && doc[k].space_before; ++w, ++k)
        if (w > 0 && STREET_WORDS.contains(doc.low(k))) street = true;
    if (!street && doc.word(k) && doc[k].space_before && STREET_WORDS.contains(doc.low(k)) && k > i + 1) street = true;
    if (!street) return;
    bool rising = true;
    for (size_t c = 1; c < s.size(); ++c) rising &= s[c] == s[c - 1] + 1;
    uint64_t v = 0;
    for (char c : s) v = v * 10 + static_cast<uint64_t>(c - '0');
    out.tokens(i, i, s.size() == 3 && !rising ? pair_words(v) : digit_words(s), 65);
}

// Numbers read from the words around them: room and port pairs, road numbers, house numbers.
RV_RULE("numbers_context", cue_numbers) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!plain_int(doc, i)) continue;
        pairs_after_cue(doc, out, i);
        road_before(doc, out, i);
        house_numbers(doc, out, i);
    }
}

// "#12", "ticket #611", "the #12 bus", "apartment #3B": number twelve. IDs after '#' digit by digit
// ("order #8830417"). Returns the last token read.
size_t hash_number(const Doc& doc, rv::Out& out, size_t i) {
    // "Item # 55821": a spaced '#' between an ID word and the number
    bool spaced = doc.number(i + 1) && !doc.joined(i + 1) && doc[i].space_before && i >= 1 && doc.word(i - 1) &&
                  !doc[i + 1].newline_before &&
                  (ID_ALWAYS.contains(doc.low(i - 1)) || ID_FOUR.contains(doc.low(i - 1)) ||
                   ID_FIVE.contains(doc.low(i - 1)) || doc.is(i - 1, "item") || doc.is(i - 1, "issue") ||
                   doc.is(i - 1, "room") || doc.is(i - 1, "unit") || doc.is(i - 1, "ticket"));
    if (!doc.number(i + 1) || (!doc.joined(i + 1) && !spaced)) return i;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return i;  // C#, F#5
    if (doc.joined(i) && (doc.punct(i - 1, '#') || doc.punct(i - 1, '&'))) return i;
    Num n = num_at(doc, i + 1);
    if (n.whole.empty() || !n.frac.empty()) return i;
    bool letter_after = doc.joined(n.last + 1) && doc.word(n.last + 1) && (doc[n.last + 1].casing & rv::UPPER) &&
                        doc.str(n.last + 1).size() <= 2;
    if (!free_right(doc, n.last) && !letter_after) return i;
    int min_digits = 0;
    size_t cue = id_cue(doc, i + 1, min_digits);
    bool by_digit = n.ndigits() >= 5 || (cue != Doc::npos && static_cast<int>(n.ndigits()) >= min_digits);
    if (n.grouped) by_digit = false;
    std::string words = by_digit ? digit_words(n.whole) : num_words(n);
    if (words.empty()) words = digit_words(n.whole);
    out.tokens(i, n.last, "number " + words, 72);
    return n.last;
}

// "# of users": the number of users. "Acct #: 4455", "Item #: 12": '#' between an ID word and a colon.
void hash_word(const Doc& doc, rv::Out& out, size_t i) {
    if (doc.is(i + 1, "of") && doc[i + 1].space_before && !doc[i + 1].newline_before && doc.word(i + 2) &&
        !(doc.joined(i) && doc[i - 1].kind != Kind::Punct)) {
        out.tokens(i, i, doc.sentence_start(i) ? "Number" : "number", 70);
        return;
    }
    if (i == 0 || !doc.punct(i + 1, ':') || !doc.joined(i + 1) || !doc.word(i - 1)) return;
    std::string_view w = doc.low(i - 1);
    if (ID_ALWAYS.contains(w) || ID_FOUR.contains(w) || ID_FIVE.contains(w) || w == "item" || w == "room" ||
        w == "unit" || w == "invoice" || w == "phone" || w == "case")
        out.tokens(i, i, "number", 70);
}

// "№ 5", "nº 5", "N° 3": the numero sign is "number". Returns the last token read.
size_t numero_sign(const Doc& doc, rv::Out& out, size_t i) {
    size_t k = Doc::npos;
    bool title = false;
    if (doc.punct(i, 0x2116)) {  // №
        k = i + 1;
        title = true;
    } else if (doc.word(i) && (doc.low(i) == "n" || doc.str(i) == "n\xC2\xBA" || doc.str(i) == "N\xC2\xBA") &&
               (doc.str(i).size() > 1 || ((doc.punct(i + 1, 0xBA) || doc.punct(i + 1, 0xB0)) && doc.joined(i + 1)))) {
        k = doc.str(i).size() > 1 ? i + 1 : i + 2;  // "nº" may be one word (º is a letter) or n + °
        title = doc.str(i)[0] == 'N';
    }
    if (k == Doc::npos || !doc.number(k) || (doc.joined(i) && doc[i - 1].kind != Kind::Punct)) return i;
    if (doc.punct(k - 1, '.') && doc.joined(k - 1)) return i;
    Num n = num_at(doc, k);
    if (n.whole.empty() || !free_right(doc, n.last)) return i;
    std::string words = num_words(n);
    if (words.empty()) return i;
    out.tokens(i, n.last, std::string(title ? "Number " : "number ") + words, 72);
    return n.last;
}

// "No. 5" number five, "Serial No. 7319024" Serial Number seven three one ..., "Symphony No. 7" Number Seven.
// Returns the last token read.
size_t no_number(const Doc& doc, rv::Out& out, size_t i) {
    if (i + 2 >= doc.size() || !doc.punct(i + 1, '.') || !doc.joined(i + 1)) return i;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return i;
    size_t k = i + 2;
    std::string head = doc[i].casing & (rv::TITLE | rv::UPPER) ? "Number" : "number";
    if (!doc.number(k)) {
        // "The serial no. is 123456789": no number right after, but an ID word before; "No. of employees"
        size_t px = prev_w(doc, i);
        int min_digits = 0;
        if ((px != Doc::npos && px + 1 == i && id_cue(doc, i, min_digits) == px && min_digits <= 4 &&
             (doc.is(k, "is") || doc.is(k, "was") || doc.punct(k, ':'))) ||
            (doc.is(k, "of") && doc[k].space_before && doc.word(k + 1)))
            out.tokens(i, i + 1, head, 70);
        return i;
    }
    head += ' ';
    Num n = num_at(doc, k);
    if (n.whole.empty()) return i;
    // "Lot no. 4471-B": a letter after the number
    size_t suffix = Doc::npos;
    if (!free_right(doc, n.last) && doc.punct(n.last + 1, '-') && doc.joined(n.last + 1) && doc.word(n.last + 2) &&
        doc.joined(n.last + 2) && doc[n.last + 2].casing == rv::UPPER && doc.str(n.last + 2).size() <= 2 &&
        free_right(doc, n.last + 2))
        suffix = n.last + 2;
    if (!free_right(doc, n.last) && suffix == Doc::npos) return i;
    int min_digits = 0;
    size_t cue = id_cue(doc, k, min_digits);
    bool by_digit = cue != Doc::npos && static_cast<int>(n.ndigits()) >= min_digits && n.frac.empty() && !n.grouped;
    std::string words = by_digit ? digit_words(n.whole) : num_words(n);
    if (words.empty()) return i;
    size_t px = prev_w(doc, i);
    if (!by_digit && px != Doc::npos && MUSIC.contains(doc.low(px)) && (doc[px].casing & rv::TITLE))
        words = rv::capital(words);
    // "Is it done? No. 5 more minutes": a "No." that answers keeps its reading; "No.1" and a line that starts
    // with "No. 5" are numbers
    bool answer = doc.sentence_start(i) && i > 0 && !doc[i].newline_before && !doc.joined(k);
    if (suffix != Doc::npos) {
        out.tokens(i, suffix, head + words + " " + std::string(doc.str(suffix)), answer ? 64 : 75);
        return suffix;
    }
    out.tokens(i, n.last, head + words, answer ? 64 : 75);
    // the number alone, for when another rule reads "No." itself
    out.tokens(k, n.last, words, 68);
    return n.last;
}

// '#', '№', "nº" and "No." before a number: one pass for all of them.
RV_RULE("numbers_context", number_signs) {
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        const rv::Token& t = doc[i];
        if (t.kind == Kind::Punct) {
            if (t.cp == '#') {
                size_t last = hash_number(doc, out, i);
                if (last == i) hash_word(doc, out, i);
                i = last;
            } else if (t.cp == 0x2116) {
                i = numero_sign(doc, out, i);
            }
        } else if (t.kind == Kind::Word && (doc.text[t.start] | 0x20) == 'n') {
            if (doc.is(i, "no")) i = no_number(doc, out, i);
            else i = numero_sign(doc, out, i);
        }
    }
}

// "a 7-3 record": seven and three; "won 3-2": three to two.
RV_RULE("numbers_context", records_and_scores) {
    for (size_t i = 0; i + 2 < doc.size(); ++i) {
        if (!plain_int(doc, i) || !free_left(doc, i)) continue;
        size_t parts[3] = {i, 0, 0};
        size_t np = 1, j = i;
        while (np < 3 && (doc.punct(j + 1, '-') || doc.punct(j + 1, 0x2013)) && doc.joined(j + 1) &&
               doc.number(j + 2) && doc.joined(j + 2)) {
            parts[np++] = j + 2;
            j += 2;
        }
        if (np < 2 || !free_right(doc, j) || (i >= 1 && doc.punct(i - 1, '[') && doc.joined(i))) {  // "[4–6]" cites
            i = j;
            continue;
        }
        bool small = true;
        for (size_t k = 0; k < np; ++k) {
            std::string_view s = doc.str(parts[k]);
            small &= s.size() <= 3 && (s.size() == 1 || s[0] != '0');
        }
        if (!small) {
            i = j;
            continue;
        }
        size_t nx = next_w(doc, j);
        std::string_view next = nx == j + 1 ? doc.low(nx) : std::string_view();
        std::string words;
        auto num = [&](size_t t) { return rv::cardinal_digits(doc.str(t)); };
        bool record = RECORD_NEXT.contains(next) || (next == "win" && (RECORD_NEXT.contains(low_or_empty(doc, next_w(doc, nx, 1))) ||
                                                RECORD_NEXT.contains(low_or_empty(doc, next_w(doc, nx, 2)))));
        // "Home record: 31-10", "a record of 7-3", "their record is 12-4"; "went 10-2 this season"
        if (!record && np >= 2) {
            // "Their season record is a respectable 7-3": the record word up to four words back
            for (int k = 1; k <= 4 && !record; ++k) {
                std::string_view w = low_or_empty(doc, prev_w(doc, i, k));
                if (w.empty()) break;
                record = w == "record" || w == "records";
            }
            // "The record drop was 10-15 points": a range of something counted
            if (record && next.size() > 3 && next.back() == 's' && (doc[nx].casing & rv::LOWER) &&
                !RECORD_NEXT.contains(next) && !GAME_NEXT.contains(next))
                record = false;
        }
        if (!record && np >= 2) {
            std::string_view p = low_or_empty(doc, prev_w(doc, i));
            std::string_view n2 = low_or_empty(doc, next_w(doc, j, 2));
            bool season_after = (next == "this" || next == "last") && (n2 == "season" || n2 == "year");
            record = season_after && (p == "went" || p == "are" || p == "is" || p == "were" || p == "finished");
            record |= next == "season" || next == "campaign" || next == "overall";
            // "improved to 8-2", "fell to 3-5"
            std::string_view p2 = low_or_empty(doc, prev_w(doc, i, 2));
            bool clause_end = next.empty() || next == "after" || next == "with" || next == "following" ||
                              next == "since" || next == "on" || next == "this" || next == "in";
            record |= clause_end && p == "to" &&
                      (p2 == "improved" || p2 == "improves" || p2 == "fell" || p2 == "falls" || p2 == "dropped" ||
                       p2 == "drops" || p2 == "moved" || p2 == "moves" || p2 == "climbed" || p2 == "slipped" ||
                       p2 == "evened");
        }
        bool split = np == 2 && doc.str(parts[0]).size() == 2 && doc.str(parts[1]).size() == 2 &&
                     (doc.str(parts[0])[0] - '0') * 10 + (doc.str(parts[0])[1] - '0') +
                             (doc.str(parts[1])[0] - '0') * 10 + (doc.str(parts[1])[1] - '0') == 100 &&
                     SPLIT_WORDS.contains(next);
        // "The Warriors (42-18) host the Suns": a team's record in brackets after its name
        if (!record && np >= 2 && i >= 2 && doc.punct(i - 1, '(') && doc.joined(i) && doc.punct(j + 1, ')') &&
            doc.joined(j + 1) && doc.word(i - 2) && (doc[i - 2].casing & rv::TITLE) &&
            sentence_has(doc, i, j, SPORTS_WORDS, 20))
            record = true;
        bool game = any_prev(doc, i, 3, GAME_PREV) || GAME_NEXT.contains(next);
        if (np == 2 && doc.str(parts[0]) == "24" && doc.str(parts[1]) == "7" && !game) {  // "open 24-7"
            out.tokens(i, j, "twenty-four seven", 70);
        } else if (split) {  // "a 50-50 chance": fifty-fifty
            out.tokens(i, j, num(parts[0]) + "-" + num(parts[1]), 70);
        } else if (record) {
            for (size_t k = 0; k < np; ++k) words += (k ? " and " : "") + num(parts[k]);
            out.tokens(i, j, words, 72);
        } else if (np == 2 && (any_prev(doc, i, 3, GAME_PREV) || GAME_NEXT.contains(next))) {
            // a range of counted things ("won 3-5 games") stays a range
            if (nx == j + 1 && doc.word(nx) && next.size() > 3 && next.back() == 's' && !GAME_NEXT.contains(next) &&
                next != "against")
            {
                i = j;
                continue;
            }
            out.tokens(i, j, num(parts[0]) + " to " + num(parts[1]), 66);
        }
        i = j;
    }
}

}  // namespace
