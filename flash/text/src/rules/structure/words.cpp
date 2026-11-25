// Masked words, chat shorthand, TL;DR, optional plurals "(s)" and the slash between two words.
#include "rules/structure/common.h"

namespace {

using namespace st;

// Words that texts mask; the first match of a pattern wins, so the commoner word comes first.
const char* const PROFANITY[] = {
    "fuck",     "fucking",  "fucked",   "fucker",    "fuckers",      "fuckery",   "fucks",     "motherfucker",
    "motherfucking", "shit", "shitty",  "shithole",  "shits",        "bullshit",  "horseshit", "bitch",
    "bitches",  "bitching", "bitchin",  "bastard",   "bastards",     "asshole",   "assholes",  "ass",
    "damn",     "damned",   "dammit",   "goddamn",   "goddamned",    "hell",      "crap",      "crappy",
    "piss",     "pissed",   "dick",     "dickhead",  "cock",         "cunt",      "prick",     "bugger",
    "bollocks", "wanker",   "twat",     "slut",      "whore",        "arse",      "arsehole",  "douche",
    "douchebag", "jackass", "frigging", "freaking"};

bool mask_cp(uint32_t cp) {
    return cp == '*' || cp == '@' || cp == '#' || cp == '$' || cp == '!' || cp == '%' || cp == 0x2014 ||
           cp == 0x2013;
}
bool dash_cp(uint32_t cp) { return cp == 0x2014 || cp == 0x2013 || cp == '-'; }
// A mask character at token k; hyphens only in a run of two or more ("f---", "F---ing").
bool mask_at(const Doc& d, size_t k) {
    if (!is_punct(d, k)) return false;
    if (d[k].cp != '-') return mask_cp(d[k].cp);
    return (p(d, k + 1, '-') && d.joined(k + 1)) || (k > 0 && p(d, k - 1, '-') && d.joined(k));
}

const rv::WordSet TITLES{"mr",    "mrs",  "ms",    "miss",  "mister", "dr",      "sir",     "lady",
                         "lord",  "capt", "captain", "col", "colonel", "gen",    "general", "prof",
                         "professor", "rev", "father", "judge", "madame", "monsieur", "count", "countess"};

// The pattern's letters at known positions; '?' for each masked letter, or "prefix" + "~" + "suffix" when the
// number of masked letters is unknown (a dash).
const char* match_word(const std::string& pat) {
    size_t tilde = pat.find('~');
    const char* best = nullptr;  // a dash with nothing after it ("f---"): the shortest word, the commoner first
    for (const char* w : PROFANITY) {
        std::string_view word(w);
        if (tilde != std::string::npos) {
            std::string_view pre(pat.data(), tilde), suf(pat.data() + tilde + 1, pat.size() - tilde - 1);
            if (pre.empty()) return nullptr;
            if (word.size() > pre.size() + suf.size() && word.substr(0, pre.size()) == pre &&
                word.substr(word.size() - suf.size()) == suf) {
                if (!suf.empty()) return w;
                if (!best || word.size() < std::string_view(best).size()) best = w;
            }
            continue;
        }
        if (word.size() != pat.size()) continue;
        bool ok = true;
        for (size_t k = 0; k < word.size() && ok; ++k) ok = pat[k] == '?' || pat[k] == word[k];
        if (ok) return w;
    }
    return best;
}

// f***, sh*t, f@ck, b****!, d—n, a**hole: the full word. Mr. S——, Mr. P—tter: the initial.
RV_RULE("structure", masked) {
    bool bang_inside = false;  // sh!t: a ! between letters
    for (size_t k = doc.text.find('!'); k != std::string_view::npos && !bang_inside; k = doc.text.find('!', k + 1))
        bang_inside = k > 0 && k + 1 < doc.text.size() && rv::is_ascii_alnum(doc.text[k + 1]) &&
                      rv::is_ascii_alnum(doc.text[k - 1]);
    if (!(bang_inside || has_any(doc, "*@#$%") || has(doc, "\xE2\x80\x93") || has(doc, "\xE2\x80\x94") ||
          has(doc, "--")))
        return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        // a run of joined letters and mask characters with at least one of each
        if (!doc.word(i) && !(is_punct(doc, i) && (doc[i].cp == '*' || doc[i].cp == '@'))) continue;
        if (i > 0 && doc.joined(i) && (doc.word(i - 1) || doc.number(i - 1) || is_punct(doc, i - 1))) {
            if (!(is_punct(doc, i - 1) && (doc[i - 1].cp == '"' || doc[i - 1].cp == '(' || doc[i - 1].cp == 0x201C)))
                continue;
        }
        size_t j = i;
        while (j + 1 < n && doc.joined(j + 1) && !doc[j + 1].newline_before && (doc.word(j + 1) || mask_at(doc, j + 1)))
            ++j;
        if (p(doc, i, '@') && j == i + 2 && p(doc, i + 1, '$') && p(doc, i + 2, '$') &&
            !(j + 1 < n && doc.joined(j + 1) && !attaches_left(doc, j + 1))) {
            out.tokens(i, j, "ass", 70);  // "@$$"
            i = j;
            continue;
        }
        bool has_word = false, has_mask = false, dash = false;
        for (size_t k = i; k <= j; ++k) {
            if (doc.word(k)) has_word = true;
            else {
                has_mask = true;
                dash |= dash_cp(doc[k].cp);
            }
        }
        bool dotted = p(doc, j + 1, '.') && j + 2 < n && doc.joined(j + 2) && (doc.word(j + 2) || doc.number(j + 2));
        if (!has_word || !has_mask || (j + 1 < n && doc.joined(j + 1) && (doc.number(j + 1) || p(doc, j + 1, '-') ||
                                                                         dotted || p(doc, j + 1, '_')))) {
            i = j;
            continue;
        }
        // "*word*" is emphasis, not a mask
        if (!doc.word(i) && !doc.word(j)) {
            i = j;
            continue;
        }
        // a name after a title: Mr. S——, Mr. P—tter
        if (dash && doc.word(i) && rv::is_upper(doc[i].cp)) {
            size_t t = i;
            if (t > 0 && p(doc, t - 1, '.')) --t;
            if (t > 0 && doc.word(t - 1) && TITLES.contains(doc.low(t - 1))) {
                out.tokens(i, j, std::string(doc.str(i).substr(0, 1)), 70);
                i = j;
                continue;
            }
        }
        // a final ! is sentence punctuation when the word is complete without it; try that first
        size_t bare = j;
        while (bare > i && p(doc, bare, '!')) --bare;
        for (size_t end : {bare, j}) {
            std::string pat, tail;
            bool upper_all = true;
            for (size_t k = i; k <= end; ++k) {
                if (doc.word(k)) {
                    std::string w = lower(doc.str(k));
                    if (k == end && w.size() > 2 && (w.compare(w.size() - 2, 2, "'s") == 0)) {
                        tail = std::string(doc.str(k).substr(w.size() - 2));
                        w.resize(w.size() - 2);
                    } else if (k == end && w.size() > 4 && w.compare(w.size() - 4, 4, "\xE2\x80\x99s") == 0) {
                        tail = std::string(doc.str(k).substr(w.size() - 4));
                        w.resize(w.size() - 4);
                    }
                    pat += w;
                    upper_all &= doc[k].casing == rv::UPPER;
                } else if (dash_cp(doc[k].cp)) {
                    if (pat.empty() || pat.back() != '~') pat += '~';
                } else {
                    pat += '?';
                }
            }
            if (pat.find('?') == std::string::npos && pat.find('~') == std::string::npos) continue;
            const char* w = match_word(pat);
            if (!w) continue;
            std::string s(w);
            if (upper_all && doc.str(i).size() > 1) {
                for (char& c : s) c = static_cast<char>(c - 32);
            } else if (doc.word(i) && rv::is_upper(doc[i].cp) &&
                       !(i > 0 && doc.word(i - 1) && doc[i - 1].casing == rv::LOWER)) {
                s = rv::capital(s);  // after a lower-case word the capital only marks the mask: "what the F---"
            }
            out.tokens(i, end, s + tail, 70);
            break;
        }
        i = j;
    }
}

// Chat shorthand: said as letters (capitals) or written out. A '!' marks the forms that count only in lower case,
// because they are also names or other words when written with a capital ("BC"); a '^' marks the ones that stand
// only at the edge of a clause, as a reply of their own ("np, happy to help", not "import numpy as np").
const rv::PhraseMap SLANG{
    {"lol", "LOL"},          {"omg", "OMG"},           {"fyi", "FYI"},          {"asap", "ASAP"},
    {"btw", "BTW"},          {"dm", "DM"},             {"brb", "be right back"}, {"tbh", "to be honest"},
    {"idk", "I don't know"}, {"ttyl", "talk to you later"}, {"imo", "in my opinion"},
    {"imho", "in my humble opinion"}, {"smh", "shaking my head"}, {"thx", "thanks"}, {"pls", "please"},
    {"plz", "please"},       {"u", "you"},             {"ur", "your"},          {"tmrw", "tomorrow"},
    {"rn", "right now"},     {"lmao", "LMAO"},         {"wtf", "WTF"},          {"irl", "in real life"},
    {"ngl", "not gonna lie"}, {"afaik", "as far as I know"}, {"iirc", "if I remember correctly"},
    {"np", "^no problem"},   {"ty", "^thank you"},     {"omw", "on my way"},    {"lmk", "let me know"},
    {"jk", "^just kidding"}, {"nvm", "never mind"},    {"ikr", "I know, right"}, {"icymi", "in case you missed it"},
    {"fwiw", "for what it's worth"}, {"yw", "^you're welcome"}, {"hbu", "how about you"}, {"bc", "!because"},
    {"ppl", "people"},       {"abt", "!about"},        {"bday", "birthday"},    {"thnx", "thanks"},
    {"thanx", "thanks"},     {"gtg", "got to go"},     {"idc", "I don't care"}, {"smth", "something"},
    {"ofc", "of course"},    {"tmr", "tomorrow"},      {"2moro", "tomorrow"},   {"cya", "see you"},
    {"eod", "EOD"},          {"eow", "EOW"},           {"hmu", "hit me up"},    {"rly", "really"},
    {"tbf", "to be fair"},   {"ttys", "talk to you soon"}, {"tysm", "thank you so much"}, {"wfh", "WFH"}};

const ShortWords SLANG_INDEX = [] {
    ShortWords t;
    int k = 1;
    for (const auto& [w, spoken] : SLANG.items()) t.add(w, k++);
    return t;
}();

// "2nite", "2day", "2moro", "4ever": a digit and letters for a word.
const rv::PhraseMap DIGIT_WORDS{{"2day", "today"}, {"2moro", "tomorrow"}, {"2morrow", "tomorrow"},
                                {"2nite", "tonight"}, {"4ever", "forever"}};
const rv::WordSet ARE_SUBJECTS{"u", "you", "we", "they", "ya"};

RV_RULE("structure", slang) {
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        const rv::Token& t = doc[i];
        if (t.kind == Kind::Punct || t.end - t.start > 5) continue;
        bool glued_before = i > 0 && doc.joined(i) && !is_punct(doc, i - 1);
        if (t.kind == Kind::Number) {  // 2nite, 2day, 4ever
            if ((doc.str(i) == "2" || doc.str(i) == "4") && doc.word(i + 1) && doc.joined(i + 1) && doc[i + 1].casing == rv::LOWER &&
                !glued_before && !(i + 2 < n && doc.joined(i + 2) && !attaches_left(doc, i + 2))) {
                std::string key = std::string(doc.str(i)) + std::string(doc.str(i + 1));
                if (const std::string_view* w = DIGIT_WORDS.find(key)) out.tokens(i, i + 1, std::string(*w), 70);
            }
            continue;
        }
        // most words are not shorthand: one table lookup decides, short forms of one or two letters go on
        uint32_t len = t.end - t.start;
        int idx = (t.casing == rv::LOWER || t.casing == rv::TITLE) ? SLANG_INDEX.get(doc.low(i)) : 0;
        if (!idx && len > 2) continue;
        // w/ w/o b/c ("w/ milk", "w/milk", "W/O"; "0.5 W/mK" is a unit)
        if (p(doc, i + 1, '/') && doc.joined(i + 1)) {
            if (glued_before || t.end - t.start != 1) continue;
            std::string_view w = doc.low(i);
            bool next_joined = i + 2 < n && doc.joined(i + 2);
            auto free_after = [&](size_t k) { return !(k < n && doc.joined(k) && !attaches_left(doc, k)); };
            size_t m = i + 2;
            if (w == "w" && next_joined && doc.is(m, "o") && doc.str(m).size() == 1 && free_after(i + 3)) {
                out.tokens(i, i + 2, "without", 70);
                i += 2;
            } else if (w == "b" && next_joined && doc.str(m) == "c" && free_after(i + 3)) {
                out.tokens(i, i + 2, "because", 70);
                i += 2;
            } else if (w == "w") {
                bool space_word = m < n && doc[m].space_before && !doc[m].newline_before &&
                                  (doc.word(m) || doc.number(m) || p(doc, m, '#') || p(doc, m, '$'));
                bool glued_word = doc.str(i) == "w" && doc.word(m) && next_joined && doc.str(m).size() > 1 &&
                                  !(i > 0 && doc.number(i - 1));
                if (space_word || glued_word) {
                    out.tokens(i, i + 1, "with", 70);
                    ++i;
                }
            }
            continue;
        }
        // b4 gr8 l8r
        if (doc.number(i + 1) && doc.joined(i + 1)) {
            if (glued_before) continue;
            if (doc.str(i) == "l" && doc.str(i + 1) == "8" && doc.is(i + 2, "r") && doc.joined(i + 2) &&
                !(i + 3 < n && doc.joined(i + 3) && !attaches_left(doc, i + 3))) {
                out.tokens(i, i + 2, "later", 70);
                continue;
            }
            if (doc.str(i) == "g" && doc.str(i + 1) == "2" && doc.str(i + 2) == "g" && doc.joined(i + 2) &&
                !(i + 3 < n && doc.joined(i + 3) && !attaches_left(doc, i + 3))) {
                out.tokens(i, i + 2, "got to go", 70);
                continue;
            }
            if (i + 2 < n && doc.joined(i + 2) && !attaches_left(doc, i + 2)) continue;
            if (doc.str(i) == "b" && doc.str(i + 1) == "4") out.tokens(i, i + 1, "before", 70);
            if (doc.str(i) == "gr" && doc.str(i + 1) == "8") out.tokens(i, i + 1, "great", 70);
            continue;
        }
        if (doc[i].casing != rv::LOWER && doc[i].casing != rv::TITLE) continue;
        // "r u ok", "how r you", "u r right": r for "are"
        if (doc.str(i) == "r" && !glued_before && !(i + 1 < n && doc.joined(i + 1) && !attaches_left(doc, i + 1)) &&
            ((i + 1 < n && doc.word(i + 1) && ARE_SUBJECTS.contains(doc.low(i + 1)) && doc[i + 1].casing == rv::LOWER &&
              !doc[i + 1].newline_before) ||
             (i > 0 && doc.word(i - 1) && ARE_SUBJECTS.contains(doc.low(i - 1)) && doc[i - 1].casing == rv::LOWER &&
              i + 1 < n && doc.word(i + 1) && !doc[i + 1].newline_before))) {
            out.tokens(i, i, "are", 70);
            continue;
        }
        if (!idx) continue;
        std::string_view s = SLANG.items()[idx - 1].second;
        bool lower_only = s[0] == '!', edge = s[0] == '^';
        if (lower_only || edge) s.remove_prefix(1);
        if (lower_only && doc[i].casing != rv::LOWER) continue;
        if (edge && !(doc.sentence_start(i) || (i > 0 && is_punct(doc, i - 1) && terminal_cp(doc[i - 1].cp)) ||
                      i + 1 >= n || (is_punct(doc, i + 1) && terminal_cp(doc[i + 1].cp))))
            continue;
        bool letters = s.size() > 1 && s[1] >= 'A' && s[1] <= 'Z';  // LOL, FYI: a capitalised "Lol" stays
        if (doc[i].casing == rv::TITLE && (!doc.sentence_start(i) || letters)) continue;
        // "Ty Burrell starred": a capitalised form before a name is the name
        if (doc[i].casing == rv::TITLE && doc.word(i + 1) && doc[i + 1].space_before && rv::is_upper(doc[i + 1].cp))
            continue;
        // the tail of a word broken across lines: "ea-\nrly"
        if (i >= 2 && p(doc, i - 1, '-') && doc.word(i - 2) && doc.joined(i - 1)) continue;
        // "import numpy as np", "5 np": code
        if (doc.is(i, "np") && i > 0 && (doc.is(i - 1, "as") || doc.number(i - 1))) continue;
        // standalone: not glued to letters, digits, slashes, @, # or a dot-word (file names, handles, paths)
        bool glued_left = i > 0 && doc.joined(i) &&
                          (doc.word(i - 1) || doc.number(i - 1) || p(doc, i - 1, '/') || p(doc, i - 1, '@') ||
                           p(doc, i - 1, '#') || p(doc, i - 1, '.') || p(doc, i - 1, '-') || p(doc, i - 1, '_'));
        bool glued_right = i + 1 < n && doc.joined(i + 1) &&
                           (p(doc, i + 1, '/') || p(doc, i + 1, '@') || p(doc, i + 1, '-') || p(doc, i + 1, '_') ||
                            p(doc, i + 1, '\'') || p(doc, i + 1, 0x2019) || p(doc, i + 1, '(') || p(doc, i + 1, '=') ||
                            (p(doc, i + 1, '.') && (doc.word(i + 2) || doc.number(i + 2)) && doc.joined(i + 2)));
        if (glued_left || glued_right) continue;
        std::string_view w = doc.low(i);
        if ((w == "dm" || w == "u" || w == "bc") && i > 0 && doc.number(i - 1)) continue;  // 5 dm, 3 u, 300 bc
        out.tokens(i, i, std::string(s), 70);
    }
}

// Emoticons are left out like emoji: ":)", ":-(", ";)", ":D", "xD", "<3" at the end of a clause, "^_^",
// "¯\_(ツ)_/¯". They must stand apart: "help:)" counts, "http://" and "1:3" do not.
const char* const EMOTICONS[] = {":-)", ":-(", ":-D", ":-P", ":-p", ":-/", ";-)", ":')", ":'(", ":)", ":(", ":D",
                                 ":P", ":p", ";)", ";D", ":/", ":O", ":o", "=)", "=(", "=D", "xD", "XD", "<3",
                                 "</3", "^_^", "^^", "-_-", "T_T", ">_<", "o_O", "O_o", ";P",
                                 "\xC2\xAF\\_(\xE3\x83\x84)_/\xC2\xAF"};

// The text may hold an emoticon: one of its first marks followed by a mark that can follow it.
bool maybe_emoticon(std::string_view t) {
    auto next = [&](size_t k) { return k + 1 < t.size() ? t[k + 1] : '\0'; };
    auto eyes = [&](size_t k) {
        char d = next(k);
        return d == ')' || d == '(' || d == 'D' || d == 'P' || d == 'p' || d == '-' || d == '/' || d == 'O' || d == 'o' ||
               d == '\'';
    };
    return each_byte(t, ':', eyes) || each_byte(t, ';', eyes) || each_byte(t, '=', eyes) ||
           each_byte(t, '<', [&](size_t k) { return next(k) == '3' || next(k) == '/'; }) ||
           each_byte(t, '^', [&](size_t k) { return next(k) == '_' || next(k) == '^'; }) ||
           each_byte(t, '_', [&](size_t k) { return k > 0 && (t[k - 1] == '-' || t[k - 1] == '>' || t[k - 1] == '^' ||
                                                              t[k - 1] == 'o' || t[k - 1] == 'O' || t[k - 1] == 'T'); }) ||
           each_byte(t, 'D', [&](size_t k) { return k > 0 && (t[k - 1] == 'x' || t[k - 1] == 'X'); }) ||
           each_byte(t, '\xAF', [&](size_t k) { return k > 0 && t[k - 1] == '\xC2'; });
}

RV_RULE("structure", emoticons) {
    if (!maybe_emoticon(doc.text)) return;
    std::string_view t = doc.text;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        uint32_t c = doc[i].cp;
        if (doc[i].kind == Kind::Word) {  // only "xD" and the letters of "o_O", "T_T"
            uint32_t len = doc[i].end - doc[i].start;
            if (!((len == 2 && (c == 'x' || c == 'X') && t[doc[i].start + 1] == 'D') ||
                  (len == 1 && (c == 'o' || c == 'O' || c == 'T') && p(doc, i + 1, '_'))))
                continue;
        } else if (!(c == ':' || c == ';' || c == '=' || c == '<' || c == '^' || c == '-' || c == '>' || c == 0xAF)) {
            continue;
        }
        uint32_t s = doc[i].start;
        for (const char* e : EMOTICONS) {
            std::string_view em(e);
            if (t.substr(s, em.size()) != em) continue;
            uint32_t end = s + static_cast<uint32_t>(em.size());
            size_t last = doc.token_at(end);
            if (last == 0 || (last < n && doc[last].start != end && doc[last - 1].end > end)) break;  // inside a token
            --last;
            if (doc[last].end != end) break;
            // left: a space, the start of a line, or a word for the ':' forms ("help:)"); never a digit
            bool left = i == 0 || doc[i].space_before ||
                        (em[0] == ':' && doc.word(i - 1) && doc[i - 1].casing != rv::UPPER);
            // right: the end, a space, sentence punctuation, another emoticon or an emoji
            bool right = last + 1 >= n || doc[last + 1].space_before ||
                         (is_punct(doc, last + 1) && (terminal_cp(doc[last + 1].cp) || emoji_cp(doc[last + 1].cp)));
            if (!left || !right) break;
            // "xD", "o_O", "T_T" look like words: only apart and at the end of a clause ("the xD card" stays)
            bool word_form = em[0] == 'x' || em[0] == 'X' || em[0] == 'o' || em[0] == 'O' || em[0] == 'T';
            bool clause_end = last + 1 >= n || doc[last + 1].newline_before ||
                              (is_punct(doc, last + 1) && (terminal_cp(doc[last + 1].cp) || emoji_cp(doc[last + 1].cp)));
            if (word_form && (!(i == 0 || doc[i].space_before) || !clause_end)) break;
            // "<3" is a heart only at the end of a clause ("love it <3"), not before a word or number ("<3 hrs",
            // "<3.0", "<3 %")
            if (em[0] == '<' &&
                !((last + 1 >= n || doc[last + 1].newline_before ||
                   (is_punct(doc, last + 1) && terminal_cp(doc[last + 1].cp) && doc[last + 1].cp != ':' &&
                    !(last + 2 < n && doc.joined(last + 2) && doc.number(last + 2)))) &&
                  i > 0 && doc.word(i - 1) && doc[i].space_before))
                break;
            if ((em == ":/" || em == ":o" || em == ":O" || em == ":P" || em == ":p") && !(i == 0 || doc[i].space_before))
                break;
            if (em[0] == '-' && !(i == 0 || doc[i].space_before)) break;
            // the space before goes with it (the space rule would drop it before a mark that looks attached)
            bool prev = i > 0 && !doc[i].newline_before, next = last + 1 < n && !doc[last + 1].newline_before;
            if (prev) out.add(doc[i - 1].end, doc[last].end, "", 60);
            else out.add(doc[i].start, next ? doc[last + 1].start : doc[last].end, "", 60);
            i = last;
            break;
        }
    }
}

// TL;DR is said "TLDR".
RV_RULE("structure", tldr) {
    if (!(has(doc, ";"))) return;
    for (size_t i = 0; i + 2 < doc.size(); ++i)
        if (doc.is(i, "tl") && p(doc, i + 1, ';') && doc.is(i + 2, "dr") && doc.joined(i + 1) && doc.joined(i + 2))
            out.tokens(i, i + 2, "TLDR", 80);
}

// An optional plural takes the plural: iteration(s) -> iterations, match(es) -> matches.
RV_RULE("structure", optional_plural) {
    if (!(has(doc, "(s)") || has(doc, "(es)"))) return;
    for (size_t i = 0; i + 3 < doc.size(); ++i) {
        if (!doc.word(i) || !p(doc, i + 1, '(') || !doc.joined(i + 1) || !doc.joined(i + 2) ||
            !p(doc, i + 3, ')') || !doc.joined(i + 3))
            continue;
        std::string_view suffix = doc.str(i + 2);
        if (suffix != "s" && suffix != "es") continue;
        out.tokens(i, i + 3, std::string(doc.str(i)) + std::string(suffix), 60);
        i += 3;
    }
}

// Pairs whose slash is read as a space.
const rv::WordSet SPACE_PAIRS{"client/server", "input/output", "encoder/decoder", "read/write",
                              "on/off", "start/stop", "send/receive", "upload/download", "import/export",
                              "request/response", "key/value", "publish/subscribe"};
// Words after a slash that make it a rate ("per"), and words before one that are units.
const rv::WordSet RATE_WORDS{"s", "sec", "second", "h", "hr", "hrs", "hour", "min", "minute", "day", "d",
                             "week", "wk", "month", "mo", "year", "yr", "night", "shift", "run", "user",
                             "unit", "person", "capita", "kg", "km", "mi", "l", "dl", "ml", "m", "g", "mol",
                             "cm", "mm", "lb", "gal", "kwh", "mwh", "core", "gpu", "node", "seat", "request",
                             "call", "query", "page", "token", "sample", "frame", "item", "head", "acre", "ha"};
const rv::WordSet REPO_CUES{"repo", "repository", "fork", "clone", "org", "organization", "github", "gitlab",
                            "package", "image", "branch", "subreddit", "namespace", "bucket", "channel"};

// word/word: "or", the fixed technical pairs and initialism pairs a space, single letters joined (I/O, A/B),
// "and/or" "and or". Paths, rates, units and handles are left for their rules.
RV_RULE("structure", slash) {
    if (!(has(doc, "/"))) return;
    size_t n = doc.size();
    // hyphenated choices: "full-time/part-time" -> "full-time or part-time"
    for (size_t k = 3; k + 3 < n; ++k) {
        if (!p(doc, k, '/') || !doc.joined(k) || !doc.joined(k + 1)) continue;
        auto compound = [&](size_t w, int dir) {  // word - word joined, all lower case, from w outwards
            size_t h = dir < 0 ? w - 1 : w + 1, v = dir < 0 ? w - 2 : w + 2;
            return doc.word(w) && doc[w].casing == rv::LOWER && p(doc, h, '-') && doc.word(v) &&
                   doc[v].casing == rv::LOWER && doc.joined(dir < 0 ? w : h) && doc.joined(dir < 0 ? h : v) &&
                   !(dir < 0 ? (v > 0 && doc.joined(v)) : (v + 1 < n && doc.joined(v + 1)));
        };
        if (compound(k - 1, -1) && compound(k + 1, 1)) out.tokens(k, k, "or", 45);
    }
    for (size_t i = 0; i + 2 < n; ++i) {
        // a spaced slash between two words: "Country / Region" -> "Country or Region"
        if (doc.word(i) && p(doc, i + 1, '/') && doc.word(i + 2) && doc[i + 1].space_before && doc[i + 2].space_before &&
            !doc[i + 1].newline_before && !doc[i + 2].newline_before && doc.str(i).size() > 1 && doc.str(i + 2).size() > 1 &&
            (doc[i].casing == rv::LOWER || doc[i].casing == rv::TITLE) &&
            (doc[i + 2].casing == rv::LOWER || doc[i + 2].casing == rv::TITLE)) {
            out.add(doc[i].end, doc[i + 2].start, doc.is(i, "and") && doc.is(i + 2, "or") ? " " : " or ", 45);
            continue;
        }
        if (!doc.word(i) || !p(doc, i + 1, '/') || !doc.word(i + 2) || !doc.joined(i + 1) || !doc.joined(i + 2))
            continue;
        // the pair must stand alone: no path, URL, handle or longer chain around it
        auto glue = [&](size_t k) {
            return p(doc, k, '/') || p(doc, k, '.') || p(doc, k, ':') || p(doc, k, '@') || p(doc, k, '_') ||
                   p(doc, k, '~') || p(doc, k, '\\') || p(doc, k, '#') || p(doc, k, '=') || doc.number(k) ||
                   doc.word(k);
        };
        if (i > 0 && doc.joined(i) && (glue(i - 1) || p(doc, i - 1, '-'))) continue;
        if (i + 3 < n && doc.joined(i + 3) &&
            (glue(i + 3) && !(p(doc, i + 3, '.') && !(doc.joined(i + 4) && (doc.word(i + 4) || doc.number(i + 4))))))
            continue;
        if (i + 3 < n && doc.joined(i + 3) && p(doc, i + 3, '-')) continue;
        std::string a = lower(doc.str(i)), b = lower(doc.str(i + 2));
        size_t prev = doc.prev_word(i);
        if (prev != Doc::npos && REPO_CUES.contains(doc.low(prev))) continue;
        if (RATE_WORDS.contains(b) || (i > 0 && doc.number(i - 1))) continue;
        if ((a == "n" && b == "a") || (a == "w" && b == "o") || (a == "b" && b == "c") || a == "r" || a == "u")
            continue;
        uint8_t ca = doc[i].casing, cb = doc[i + 2].casing;
        std::string spoken;
        if (a == "and" && b == "or") {
            spoken = std::string(doc.str(i)) + " " + std::string(doc.str(i + 2));
        } else if (SPACE_PAIRS.contains(a + "/" + b)) {
            spoken = std::string(doc.str(i)) + " " + std::string(doc.str(i + 2));
        } else if (ca == rv::UPPER && cb == rv::UPPER) {
            bool single = doc.str(i).size() == 1 && doc.str(i + 2).size() == 1;
            spoken = std::string(doc.str(i)) + (single ? "" : " ") + std::string(doc.str(i + 2));
        } else if ((ca == rv::LOWER || ca == rv::TITLE) && (cb == rv::LOWER || cb == rv::TITLE)) {
            if (a.size() < 2 || b.size() < 2) continue;
            spoken = std::string(doc.str(i)) + " or " + std::string(doc.str(i + 2));
        } else if (ca == rv::UPPER && cb != rv::UPPER) {
            spoken = std::string(doc.str(i)) + " " + std::string(doc.str(i + 2));
        } else {
            continue;
        }
        out.tokens(i, i + 2, spoken, 45);
        i += 2;
    }
}

}  // namespace

namespace {

using namespace st;

// A one-line input that starts with a number and breaks off without closing punctuation is a fragment ("974
// accuracy on the held-out split, with the"): the number keeps a lower-case start. (The engine's fragment test
// covers only inputs of one or two words; this covers a plain number before a lower-case word.)
RV_RULE("structure", fragment_number) {
    size_t n = doc.size();
    if (n < 3 || !doc.number(0)) return;
    std::string_view t = doc.text;
    if (t.find('\n') != std::string_view::npos) return;
    for (size_t k = 1; k + 1 < n; ++k)  // one sentence, broken off
        if (is_punct(doc, k) && (doc[k].cp == '.' || doc[k].cp == '!' || doc[k].cp == '?') && doc[k + 1].space_before &&
            doc.word(k + 1) && doc[k + 1].casing != rv::LOWER)
            return;
    const rv::Token& z = doc[n - 1];
    if (z.kind == Kind::Punct && (terminal_cp(z.cp) && z.cp != ',' && z.cp != ';')) return;
    if (z.kind == Kind::Punct && (z.cp == '"' || z.cp == 0x201D || z.cp == ')' || z.cp == '\'' || z.cp == 0x2019) &&
        is_punct(doc, n - 2) && terminal_cp(doc[n - 2].cp) && doc[n - 2].cp != ',')
        return;
    rv::NumberSpan s = rv::parse_number(doc, 0);
    size_t w = s.last + 1;
    if (!s.ok || w >= n || !doc[w].space_before || !doc.word(w) || doc[w].casing != rv::LOWER) return;
    if (s.fraction.empty() && !s.grouped && s.whole.size() == 4 && s.whole >= "1000" && s.whole < "2100") return;  // a year?
    std::string words = rv::read_number(s);
    if (words.empty()) return;
    out.tokens(0, s.last, words, 45, true);
}

}  // namespace
