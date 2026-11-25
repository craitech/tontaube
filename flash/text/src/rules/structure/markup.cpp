// Markup the narrator never says: invisible characters, emoji and decorations, Markdown emphasis, links and
// images, wiki syntax, HTML tags, comments and entities, chat-template tokens.
#include <cstring>

#include "rules/structure/common.h"

namespace {

using namespace st;

// Typographic variants read as their plain form.
const char* ligature(uint32_t cp) {
    switch (cp) {
        case 0xFB00: return "ff";
        case 0xFB01: return "fi";
        case 0xFB02: return "fl";
        case 0xFB03: return "ffi";
        case 0xFB04: return "ffl";
        case 0xFB05: case 0xFB06: return "st";
        default: return nullptr;
    }
}

// Invisible characters are removed. Ligatures are spelled out with the letters joined to them (a replacement
// that ends in a letter would otherwise get a space before the next letter), and non-breaking hyphens become
// plain hyphens.
RV_RULE("structure", invisible) {
    if (!maybe_invisible(doc)) return;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (is_invisible(doc, i)) {
            out.tokens(i, i, "", 90);
        } else if (is_punct(doc, i) && (doc[i].cp == 0x2011 || doc[i].cp == 0x2010)) {
            out.tokens(i, i, "-", 90);
        } else if (is_punct(doc, i) && ligature(doc[i].cp)) {
            size_t a = i, b = i;
            while (a > 0 && doc.joined(a) && (doc.word(a - 1) || is_invisible(doc, a - 1) ||
                                              (is_punct(doc, a - 1) && ligature(doc[a - 1].cp))))
                --a;
            while (b + 1 < doc.size() && doc.joined(b + 1) &&
                   (doc.word(b + 1) || is_invisible(doc, b + 1) || (is_punct(doc, b + 1) && ligature(doc[b + 1].cp))))
                ++b;
            std::string s;
            for (size_t k = a; k <= b; ++k) {
                if (is_invisible(doc, k)) continue;
                const char* l = is_punct(doc, k) ? ligature(doc[k].cp) : nullptr;
                s += l ? std::string(l) : std::string(doc.str(k));
            }
            out.tokens(a, b, s, 91);
            i = b;
        }
    }
}

// Emoji and pictographs are removed, never described; a run of them (with joiners, skin tones and variation
// selectors) goes as one, and so do decorative symbols at the start of a line.
RV_RULE("structure", emoji) {
    if (!maybe_symbol(doc)) return;
    for (size_t i = 0; i < doc.size(); ++i) {
        bool line_start = i == 0 || doc[i].newline_before;
        bool after_sentence = i > 0 && !line_start && doc[i].space_before && is_punct(doc, i - 1) &&
                              (doc[i - 1].cp == '.' || doc[i - 1].cp == '!' || doc[i - 1].cp == '?');
        // a single star before a word is a decoration too; a row of stars is a rating
        bool star = line_start && p(doc, i, 0x2605) && doc.word(i + 1) && doc[i + 1].space_before &&
                    !doc[i + 1].newline_before;
        bool deco = (line_start || after_sentence) && is_punct(doc, i) && (decor_cp(doc[i].cp) || star) &&
                    i + 1 < doc.size() && !doc[i + 1].newline_before &&
                    !(after_sentence && (doc[i].cp == 0x2192 || doc[i].cp == 0x21D2 || doc[i].cp == 0x2713 ||
                                         doc[i].cp == 0x2714 || doc[i].cp == 0x2717 || doc[i].cp == 0x2718));
        if (!is_emoji(doc, i) && !deco) continue;
        size_t j = i;
        while (j + 1 < doc.size() && !doc[j + 1].newline_before &&
               (is_emoji(doc, j + 1) || is_invisible(doc, j + 1) || (deco && decor_cp(doc[j + 1].cp))))
            ++j;
        Span s = deletion(doc, i, j);
        out.add(s.start, s.end, "", 60);
        i = j;
    }
}

// A run of two to five emoji stars is a rating, read like a run of black stars: "⭐⭐⭐⭐⭐ Best purchase" -> "five
// out of five stars Best purchase". A single one is a decoration and goes with the emoji.
RV_RULE("structure", emoji_stars) {
    if (!has(doc, "\xE2\xAD\x90")) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        if (!p(doc, i, 0x2B50)) continue;
        size_t j = i;
        while (j + 1 < n && p(doc, j + 1, 0x2B50) && doc.joined(j + 1)) ++j;
        size_t count = j - i + 1;
        if (count >= 2 && count <= 5) {
            std::string c = rv::cardinal(count);
            out.tokens(i, j, c + " out of " + c + " stars", 61);
        }
        i = j;
    }
}

// Markdown emphasis (** __ * _ ~~) and wiki emphasis ('' '''): the markers go, the text stays.
const rv::WordSet DUNDERS{"init", "main", "name", "dict", "call", "repr", "str", "len", "file", "all",
                          "slots", "class", "doc", "version", "future", "new", "del", "enter", "exit", "iter"};

bool marker_char(uint32_t cp) { return cp == '*' || cp == '_' || cp == '~' || cp == '\''; }

RV_RULE("structure", emphasis) {
    if (!(has_any(doc, "*_~") || has(doc, "''"))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        if (!is_punct(doc, i) || !marker_char(doc[i].cp)) continue;
        uint32_t c = doc[i].cp;
        size_t j = i;
        while (j + 1 < n && doc.joined(j + 1) && p(doc, j + 1, c)) ++j;
        size_t len = j - i + 1;
        bool ok_len = c == '~' ? len == 2 : c == '\'' ? (len == 2 || len == 3) : len <= 3;
        // an opener: at a word boundary on the left, text right after it
        // (a marker of another kind may enclose it: "**_Important:_**")
        bool left_ok = i == 0 || doc[i].space_before || doc[i].newline_before ||
                       (is_punct(doc, i - 1) && doc[i - 1].cp != '\\' &&
                        (!marker_char(doc[i - 1].cp) || (c != '\'' && doc[i - 1].cp != c && doc[i - 1].cp != '\'')));
        if (!ok_len || !left_ok || j + 1 >= n || !doc.joined(j + 1) || doc[j + 1].newline_before ||
            (c == '\'' && doc.word(i - 1) && doc.joined(i))) {
            i = j;
            continue;
        }
        // find the closer of the same length on the same line
        size_t k = j + 1, close = 0;
        for (; k < n && !doc[k].newline_before; ++k) {
            if (!p(doc, k, c) || !doc.joined(k)) continue;
            size_t e = k;
            while (e + 1 < n && doc.joined(e + 1) && p(doc, e + 1, c)) ++e;
            bool right_ok = e + 1 >= n || !doc.joined(e + 1) || doc[e + 1].newline_before ||
                            (is_punct(doc, e + 1) && (!marker_char(doc[e + 1].cp) ||
                                                      (c != '\'' && doc[e + 1].cp != c && doc[e + 1].cp != '\'')));
            if (e - k + 1 == len && right_ok) {
                close = k;
                break;
            }
            k = e;
        }
        if (!close) {
            i = j;
            continue;
        }
        // "__init__" and other identifiers stay for the code rules
        bool dunder = close == j + 2 && doc.word(j + 1) && c == '_' && len == 2 &&
                      ((i > 0 && p(doc, i - 1, '.') && doc.joined(i)) ||
                       (p(doc, close + len, '(') && doc.joined(close + len)) || DUNDERS.contains(doc.low(j + 1)));
        if (dunder) {
            i = close + len - 1;
            continue;
        }
        out.tokens(i, j, "", 50);
        out.tokens(close, close + len - 1, "", 50);
        i = j;
    }
}

// The token after i on the same line, joined to it.
bool joined_same_line(const Doc& d, size_t i) { return i < d.size() && d.joined(i) && !d[i].newline_before; }

// Index of the token that closes the bracket opened at i (same line), or 0.
size_t closing(const Doc& d, size_t i, uint32_t open, uint32_t close) {
    int depth = 0;
    for (size_t k = i; k < d.size(); ++k) {
        if (k > i && d[k].newline_before) return 0;
        if (p(d, k, open)) ++depth;
        else if (p(d, k, close) && --depth == 0) return k;
    }
    return 0;
}

// [text](url) reads as its text. ![alt](src) reads as "Image: alt." and goes away when the alt text is empty.
// [[Target|text]] reads as its text, [[Target]] as the target.
RV_RULE("structure", links) {
    if (!(has(doc, "["))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        if (!p(doc, i, '[')) continue;
        if (p(doc, i + 1, '[') && joined_same_line(doc, i + 1)) {  // wiki link
            size_t e = i + 2;
            while (e + 1 < n && !doc[e].newline_before && !(p(doc, e, ']') && p(doc, e + 1, ']'))) ++e;
            if (e + 1 >= n || doc[e].newline_before || e == i + 2) continue;
            size_t bar = 0;
            for (size_t k = i + 2; k < e; ++k)
                if (p(doc, k, '|')) bar = k;
            if (bar && bar + 1 < e) out.add(doc[i].start, doc[bar + 1].start, "", 85);
            else out.tokens(i, i + 1, "", 85);
            out.tokens(e, e + 1, "", 85);
            i = e + 1;
            continue;
        }
        size_t close = closing(doc, i, '[', ']');
        if (!close || !p(doc, close + 1, '(') || !joined_same_line(doc, close + 1)) continue;
        size_t paren = closing(doc, close + 1, '(', ')');
        if (!paren) continue;
        bool image = i > 0 && p(doc, i - 1, '!') && doc.joined(i);
        if (image) {
            if (close == i + 1) {  // empty alt text
                Span s = deletion(doc, i - 1, paren);
                out.add(s.start, s.end, "", 92);
            } else {
                out.tokens(i - 1, i, "Image: ", 85);
                bool ends = (is_punct(doc, close - 1) && terminal_cp(doc[close - 1].cp)) ||
                            (paren + 1 < n && doc.joined(paren + 1) && is_punct(doc, paren + 1) &&
                             terminal_cp(doc[paren + 1].cp)) ||  // "as ![Score plot](s.svg)." has its period
                            (paren + 1 < n && !doc[paren + 1].newline_before && doc.word(paren + 1) &&
                             doc[paren + 1].casing == rv::LOWER);  // "![Power chart](p.png) shows the draw"

                out.tokens(close, paren, ends ? "" : ".", 92);
            }
        } else if (close > i + 1) {
            out.tokens(i, i, "", 85);
            out.tokens(close, paren, "", 92);
        }
        i = paren;
    }
}

const rv::WordSet LOG_LEVELS{"debug", "error", "fatal", "info", "notice", "trace", "warn", "warning", "critical"};

// Brackets around words: an editorial insertion keeps its words ("[the company]" -> "the company", "[REDACTED]"
// -> "redacted"); a tag in a log line is read as its words and a comma, as written ("ERROR [line-4] signal" ->
// "Error line four, signal", "[edge-7] 84.212..." -> "edge seven, eighty-four ...", "[ERROR] 2026-10-02" ->
// "Error, October ..."). Numeric groups, check boxes, links, wiki links and indexes are other rules' business.
RV_RULE("structure", brackets) {
    if (!(has(doc, "["))) return;
    size_t n = doc.size();
    for (size_t i = 0; i + 2 < n; ++i) {
        if (!p(doc, i, '[') || (i > 0 && p(doc, i - 1, '[')) || p(doc, i + 1, '[')) continue;
        if (i > 0 && doc.joined(i) && !is_punct(doc, i - 1)) continue;  // an index or a glued citation
        size_t close = 0;
        for (size_t k = i + 1; k < n && k < i + 40 && !doc[k].newline_before; ++k) {
            if (p(doc, k, '[')) break;
            if (p(doc, k, ']')) {
                close = k;
                break;
            }
        }
        if (!close || close < i + 2) continue;
        if (close + 1 < n && doc.joined(close + 1) && (p(doc, close + 1, '(') || p(doc, close + 1, ':'))) continue;
        bool letters = false, spaced = false, tag_sep = false, other = false;
        for (size_t k = i + 1; k < close; ++k) {
            if (doc.word(k)) letters |= doc.str(k).size() >= 2;
            else if (is_punct(doc, k) && (p(doc, k, '-') || p(doc, k, '_'))) tag_sep = true;
            else if (is_punct(doc, k) && !(p(doc, k, ',') || p(doc, k, '\'') || p(doc, k, '.') || p(doc, k, 0x2019)))
                other = true;
            if (k > i + 1 && doc[k].space_before) spaced = true;
        }
        if (!letters || other || doc.is(i + 1, "sic") || doc.is(i + 1, "inst")) continue;
        size_t pw = i > 0 ? doc.prev_word(i) : Doc::npos;
        bool log = pw != Doc::npos && pw + 1 == i && LOG_LEVELS.contains(doc.low(pw)) && doc[pw].casing == rv::UPPER;
        size_t nx = close + 1;
        bool more = nx < n && !doc[nx].newline_before && (doc.word(nx) || doc.number(nx) || p(doc, nx, '['));
        log |= more && doc.word(nx) && LOG_LEVELS.contains(doc.low(nx)) && doc[nx].casing == rv::UPPER;
        bool line_start = i == 0 || doc[i].newline_before;
        bool shout = !spaced && close == i + 2 && doc[i + 1].casing == rv::UPPER;  // [REDACTED], [ERROR]
        if ((!spaced && (tag_sep || log)) || (log && close <= i + 5) || (shout && line_start && more)) {
            // a tag: words split at '-' and '_', numbers in full, initialisms in capitals, a comma before the message
            std::string words;
            for (size_t k = i + 1; k < close; ++k) {
                if (!doc.word(k) && !doc.number(k)) continue;
                std::string w;
                if (doc.number(k)) {
                    w = rv::read_number(doc.str(k));
                    if (w.empty()) w = rv::digits(doc.str(k));
                } else if (initialism(doc.low(k))) {
                    w = std::string(doc.str(k));
                    for (char& c : w) c = static_cast<char>(c >= 'a' && c <= 'z' ? c - 32 : c);
                } else if (doc[k].casing == rv::UPPER && doc.str(k).size() > 1) {
                    w = rv::capital(lower(doc.str(k)));  // [ERROR] -> Error
                } else if (doc.str(k).size() == 1 && doc.number(k + 1) && doc.joined(k + 1)) {
                    w = std::string(1, static_cast<char>(doc.str(k)[0] & ~0x20));  // [api-v2] -> API V two
                } else {
                    w = std::string(doc.str(k));
                }
                if (!words.empty()) words += ' ';
                words += w;
            }
            out.tokens(i, close, words + (more ? "," : ""), 65, true);
            // when an identifier rule reads the inside ("[checkout_v2]"), the brackets still go
            out.tokens(i, i, "", 64);
            out.tokens(close, close, more ? "," : "", 64);
        } else if (shout) {
            out.tokens(i, close, lower(doc.str(i + 1)), 60);  // [REDACTED]
        } else {
            out.tokens(i, i, "", 60);
            out.tokens(close, close, "", 60);
        }
        i = close;
    }
}

// Deletes tokens a..b; between two non-space neighbours a block tag leaves a space.
void drop(const Doc& d, Out& out, size_t a, size_t b, bool block, int score) {
    bool tight_left = a > 0 && !d[a].space_before && !d[a].newline_before;
    bool tight_right = b + 1 < d.size() && !d[b + 1].space_before && !d[b + 1].newline_before;
    if (block && tight_left && tight_right && !attaches_left(d, b + 1)) {
        out.tokens(a, b, " ", score);
        return;
    }
    if (tight_left || tight_right) {
        out.tokens(a, b, "", score);
        return;
    }
    Span s = deletion(d, a, b);
    out.add(s.start, s.end, "", score);
}

const rv::WordSet REFER_CUES{"to", "in", "see", "under", "from", "of", "per", "on", "at", "with", "and"};

// HTML and XML-like tags go and their text stays; comments go whole; chat-template tokens (<|im_end|>, <s>,
// [INST]) go. Bracketed metadata such as <Figure 3.1: Sales> is not narration and goes whole.
RV_RULE("structure", tags) {
    if (!(has(doc, "<") || has(doc, "INST]"))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        if (p(doc, i, '[')) {  // [INST] [/INST]
            bool block = false;
            size_t e = tag_at(doc, i, &block);
            if (e) {
                drop(doc, out, i, e, true, 90);
                i = e;
            }
            continue;
        }
        if (!p(doc, i, '<') || !joined_same_line(doc, i + 1)) continue;
        // <!-- comment -->
        if (p(doc, i + 1, '!') && p(doc, i + 2, '-') && p(doc, i + 3, '-')) {
            size_t e = i + 4;
            while (e + 2 < n && !(p(doc, e, '-') && p(doc, e + 1, '-') && p(doc, e + 2, '>'))) ++e;
            if (e + 2 >= n) continue;
            size_t last = e + 2;
            bool whole_line = (i == 0 || doc[i].newline_before) && (last + 1 >= n || doc[last + 1].newline_before);
            if (whole_line) {
                uint32_t s = doc[i].start, t = last + 1 < n ? doc[last + 1].start : doc[last].end;
                if (last + 1 >= n && i > 0) s = doc[i - 1].end;
                out.add(s, t, "", 92);
            } else {
                drop(doc, out, i, last, false, 92);
            }
            i = last;
            continue;
        }
        // <https://example.com>, <mailto:...>, <ana@example.com>: an autolink is read as its address (electronic
        // reads it with its brackets and outscores this; the brackets go when it cannot)
        bool web = doc.is(i + 1, "http") || doc.is(i + 1, "https") || doc.is(i + 1, "www") || doc.is(i + 1, "mailto");
        if ((web || doc.word(i + 1) || doc.number(i + 1)) && doc.joined(i + 2)) {
            size_t e = i + 2;
            bool at = false;
            while (e < n && doc.joined(e) && !doc[e].newline_before && !p(doc, e, '>') && !p(doc, e, '<')) at |= p(doc, e++, '@');
            if ((web || at) && e < n && p(doc, e, '>') && doc.joined(e)) {
                out.tokens(i, i, "", 80);
                out.tokens(e, e, "", 80);
                i = e;
                continue;
            }
        }
        // <sup>2</sup> after a letter or a number is a power; after a word it is a footnote mark and goes;
        // <sub>2</sub> is a subscript, read apart from its letters ("H<sub>2</sub>O" -> "H two O")
        if ((doc.is(i + 1, "sup") || doc.is(i + 1, "sub")) && p(doc, i + 2, '>') && doc.joined(i + 2) && i > 0 &&
            doc.joined(i) && p(doc, i + 4, '<') && p(doc, i + 5, '/') && doc.is(i + 6, doc.low(i + 1)) && p(doc, i + 7, '>') &&
            doc.joined(i + 4) && doc.joined(i + 5) && doc.joined(i + 6) && doc.joined(i + 7) && doc.joined(i + 3) &&
            (doc.number(i + 3) || doc.word(i + 3))) {
            bool sup = doc.is(i + 1, "sup");
            bool power_base = (doc.number(i - 1) && doc.str(i - 1).size() <= 3) ||
                              (doc.word(i - 1) && doc.str(i - 1).size() == 1) || p(doc, i - 1, ')');
            if (sup && !power_base) {
                out.tokens(i, i + 7, "", 90);  // a footnote mark
            } else if (sup && doc.number(i + 3)) {
                std::string_view v = doc.str(i + 3);
                std::string w = v == "2" ? "squared" : v == "3" ? "cubed" : "to the power of " + rv::read_number(v);
                out.tokens(i, i + 7, w, 90);
            } else {
                out.tokens(i, i + 2, " ", 90);
                out.tokens(i + 4, i + 7, i + 8 < n && doc.joined(i + 8) && !attaches_left(doc, i + 8) ? " " : "", 90);
            }
            i += 7;
            continue;
        }
        bool block = false;
        size_t e = tag_at(doc, i, &block);
        if (e) {
            // a run of tags ("</p><p>", "<div><p><strong>") goes as one
            bool b2 = false;
            size_t e2;
            while (e + 1 < n && doc.joined(e + 1) && !doc[e + 1].newline_before && (e2 = tag_at(doc, e + 1, &b2))) {
                block |= b2;
                e = e2;
            }
            drop(doc, out, i, e, block, 90);
            i = e;
            continue;
        }
        size_t k = i + 1;
        if (p(doc, k, '/')) continue;
        if (!doc.word(k)) continue;
        size_t after = k;
        while (after < n && doc.joined(after) && (doc.word(after) || doc.number(after))) ++after;
        e = after;
        while (e < n && !doc[e].newline_before && !p(doc, e, '>') && !p(doc, e, '<') && e < after + 60) ++e;
        if (e >= n || !p(doc, e, '>') || doc[e].newline_before) continue;
        // <Figure 3.1: Quarterly Sales> <End Chapter> <See Appendix A.>: metadata, removed with its text
        // (<Report_ID_73B> is a placeholder: electronic)
        if (doc[k].casing == rv::TITLE && e > after && doc.joined(e) && e - i >= 3 && !p(doc, after, '_')) {
            // part of the sentence ("refer to <Appendix A> for", "<Figure 3.1> shows"): only the brackets go
            size_t pw = i > 0 && !doc[i].newline_before ? i - 1 : Doc::npos;
            bool inside = (pw != Doc::npos && doc.word(pw) && REFER_CUES.contains(doc.low(pw))) ||
                          (e + 1 < n && !doc[e + 1].newline_before && doc.word(e + 1) && doc[e + 1].casing == rv::LOWER);
            if (inside) {
                out.tokens(i, i, "", 90);
                out.tokens(e, e, "", 90);
                i = e;
                continue;
            }
            Span s = deletion(doc, i, e);
            if (i > 0 && p(doc, i - 1, ',') && p(doc, e + 1, ',')) s.end = doc[e + 1].end;
            // "note: <See Appendix A>. The" and "net. <Figure 3>. The": the period after it goes too
            // (at the end of a line, a period after an abbreviation or an ellipsis stays: "P.S. <Adjourned>." -> "PS.")
            bool line_end = e + 2 >= n || doc[e + 2].newline_before;
            bool abbr = i > 1 && p(doc, i - 1, '.') && (p(doc, i - 2, '.') || (doc.word(i - 2) && doc.str(i - 2).size() == 1));
            if (i > 0 && is_punct(doc, i - 1) && (doc[i - 1].cp == ':' || doc[i - 1].cp == '.') && p(doc, e + 1, '.') &&
                doc.joined(e + 1) && !p(doc, e + 2, '.'))
                s = {doc[i - 1].end, line_end && abbr ? doc[e].end : doc[e + 1].end};
            out.add(s.start, s.end, "", 90);
            i = e;
        }
    }
}

// HTML entities become the character they stand for, or its words when the character is a symbol.
const rv::PhraseMap ENTITIES{
    {"amp", "&"}, {"nbsp", " "}, {"quot", "\""}, {"apos", "'"}, {"lsquo", "\xE2\x80\x98"}, {"rsquo", "\xE2\x80\x99"},
    {"sbquo", "\xE2\x80\x9A"}, {"ldquo", "\xE2\x80\x9C"}, {"rdquo", "\xE2\x80\x9D"}, {"bdquo", "\xE2\x80\x9E"},
    {"hellip", "\xE2\x80\xA6"}, {"mdash", "\xE2\x80\x94"}, {"ndash", "\xE2\x80\x93"}, {"minus", "\xE2\x88\x92"},
    {"lt", "<"}, {"gt", ">"}, {"le", "\xE2\x89\xA4"}, {"ge", "\xE2\x89\xA5"}, {"ne", "\xE2\x89\xA0"},
    {"thinsp", " "}, {"ensp", " "}, {"emsp", " "}, {"shy", ""}, {"zwj", ""}, {"zwnj", ""}, {"laquo", "\xC2\xAB"},
    {"raquo", "\xC2\xBB"}, {"middot", "\xC2\xB7"}, {"bull", "\xE2\x80\xA2"}, {"copy", "\xC2\xA9"},
    {"reg", "\xC2\xAE"}, {"trade", "\xE2\x84\xA2"}, {"plusmn", "\xC2\xB1"}, {"sect", "\xC2\xA7"},
    {"para", "\xC2\xB6"}, {"frac12", "\xC2\xBD"}, {"frac14", "\xC2\xBC"}, {"frac34", "\xC2\xBE"},
    {"asymp", "\xE2\x89\x88"}, {"rarr", "\xE2\x86\x92"}, {"uarr", "\xE2\x86\x91"}, {"darr", "\xE2\x86\x93"},
    {"deg", "\xC2\xB0"}, {"cent", "\xC2\xA2"}, {"times", "\xC3\x97"}, {"divide", "\xC3\xB7"},
    {"infin", "\xE2\x88\x9E"}, {"prime", "\xE2\x80\xB2"}};

// Words for the symbols an entity can stand for; nullptr for a character that stays as it is.
const char* symbol_words(uint32_t cp) {
    switch (cp) {
        case '&': return "and";
        case '<': return "less than";
        case '>': return "more than";
        case 0x2264: return "less than or equal to";
        case 0x2265: return "greater than or equal to";
        case 0x2260: return "not equal to";
        case 0xA9: return "copyright";
        case 0xAE: return "registered trademark";
        case 0x2122: return "trademark";
        case 0xB1: return "plus or minus";
        case 0xA7: return "Section";
        case 0xB6: return "paragraph";
        case 0xBD: return "one half";
        case 0xBC: return "one quarter";
        case 0xBE: return "three quarters";
        case 0x2248: return "approximately";
        case 0x2192: return "to";
        case 0x2191: return "up";
        case 0x2193: return "down";
        case 0xB7: case 0x2022: return ",";
        case 0xB0: return "degrees";
        case 0xA2: return "cents";
        case 0xD7: return "times";
        case 0xF7: return "divided by";
        case 0x221E: return "infinity";
        case 0x2212: return "minus";
        default: return nullptr;
    }
}

// &atilde; &eacute; &uuml; &ccedil; ...: the accented Latin-1 letter, 0 if the name is not one.
uint32_t accented_entity(std::string_view name) {
    static const char* const MARKS[] = {"grave", "acute", "circ", "tilde", "uml", "ring", "cedil"};
    // Latin-1 code points of the vowels (and c, n, y) with each mark, upper case; lower case is +0x20
    static const char BASES[] = "AEIOUCNY";
    static const uint16_t CODES[7][8] = {
        {0xC0, 0xC8, 0xCC, 0xD2, 0xD9, 0, 0, 0},     {0xC1, 0xC9, 0xCD, 0xD3, 0xDA, 0, 0, 0xDD},
        {0xC2, 0xCA, 0xCE, 0xD4, 0xDB, 0, 0, 0},     {0xC3, 0, 0, 0xD5, 0, 0, 0xD1, 0},
        {0xC4, 0xCB, 0xCF, 0xD6, 0xDC, 0, 0, 0x178}, {0xC5, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0xC7, 0, 0}};
    if (name.size() < 3) return 0;
    char base = name[0];
    bool upper = base >= 'A' && base <= 'Z';
    char ub = upper ? base : static_cast<char>(base - 32);
    const char* bp = std::strchr(BASES, ub);
    if (!bp || !*bp) return 0;
    for (int m = 0; m < 7; ++m) {
        if (name.substr(1) != MARKS[m]) continue;
        uint32_t c = CODES[m][bp - BASES];
        if (!c) return 0;
        if (c == 0x178) return upper ? 0x178 : 0xFF;
        return upper ? c : c + 0x20;
    }
    if (name == "szlig") return 0xDF;
    return 0;
}

RV_RULE("structure", entities) {
    if (!(has(doc, "&"))) return;
    size_t n = doc.size();
    std::string_view t = doc.text;
    for (size_t i = 0; i + 2 < n; ++i) {
        if (!p(doc, i, '&') || !doc.joined(i + 1)) continue;
        uint32_t cp = 0;
        bool none = false;  // an entity for nothing (&shy;)
        size_t last;
        if (doc.word(i + 1)) {  // &amp; &frac12; ...: letters and digits up to ';'
            uint32_t b = doc[i + 1].start, e = b;
            while (e < t.size() && e - b < 10 && rv::is_ascii_alnum(t[e])) ++e;
            if (e >= t.size() || t[e] != ';') continue;
            last = doc.token_at(e);
            if (last >= n || doc[last].start != e) continue;
            std::string_view name = t.substr(b, e - b);
            const std::string_view* s = ENTITIES.find(lower(name));
            cp = s ? 0 : accented_entity(name);
            if (!s && !cp) continue;
            if (s && s->empty()) none = true;
            else if (s) {
                size_t k = 0;
                cp = rv::decode(*s, k);
            }
        } else if (p(doc, i + 1, '#') && doc.joined(i + 2)) {  // &#8217; &#x2014;
            uint32_t b = doc[i + 1].end, e = b;
            bool hex = b < t.size() && (t[b] == 'x' || t[b] == 'X');
            if (hex) ++e;
            uint32_t v = 0, digits = 0;
            for (; e < t.size() && digits < 7; ++e, ++digits) {
                char c = t[e];
                int dv = c >= '0' && c <= '9' ? c - '0' : hex && (c | 0x20) >= 'a' && (c | 0x20) <= 'f' ? (c | 0x20) - 'a' + 10 : -1;
                if (dv < 0) break;
                v = v * (hex ? 16 : 10) + static_cast<uint32_t>(dv);
            }
            if (!digits || e >= t.size() || t[e] != ';' || v < 32 || v >= 0x110000) continue;
            last = doc.token_at(e);
            if (last >= n || doc[last].start != e) continue;
            cp = v == 0xAD || v == 0x200B || v == 0x200C || v == 0x200D || v == 0xFEFF ? 0 : v;
            none = cp == 0;
        } else {
            continue;
        }
        bool sp_left = doc[i].space_before, sp_right = last + 1 < n && doc[last + 1].space_before;
        bool line_start = i == 0 || doc[i].newline_before;
        bool num_left = i > 0 && doc.number(i - 1) && doc.joined(i);
        bool num_right = last + 1 < n && doc.number(last + 1) && doc.joined(last + 1);
        std::string spoken;
        const char* words = none ? nullptr : symbol_words(cp);
        if (none || cp == 0xA0 || cp == ' ') {
            spoken = (sp_left || sp_right || i == 0 || last + 1 >= n) ? "" : " ";
        } else if (cp == '>' && line_start && sp_right) {  // "&gt; quoted text": a quote marker
            out.add(doc[i].start, doc[last + 1].start, "", 85);
            i = last;
            continue;
        } else if (cp == '&' && p(doc, last + 1, '&') && doc.joined(last + 1) && doc.is(last + 2, "amp") &&
                   p(doc, last + 3, ';')) {  // &amp;&amp; is one "and"
            out.tokens(i, last + 3, "and", 85);
            i = last + 3;
            continue;
        } else if ((cp == '>' || cp == '<') && p(doc, last + 1, '=') && doc.joined(last + 1)) {
            out.tokens(i, last + 1, cp == '>' ? "greater than or equal to" : "less than or equal to", 85);
            i = last + 1;
            continue;
        } else if ((cp == 0x2014 || cp == 0x2013) && sp_left && sp_right && i > 0 && !line_start &&
                   !doc[last + 1].newline_before) {
            out.add(doc[i - 1].end, doc[last + 1].start, ", ", 85);  // a spaced dash sets off an insertion
            i = last;
            continue;
        } else if (cp == 0x2026 && !sp_left && (last + 1 >= n || doc[last + 1].newline_before || attaches_left(doc, last + 1))) {
            spoken = "";  // a trailing ellipsis goes, as written out
        } else if (cp == 0x2026 && !sp_left && sp_right && doc.word(last + 1)) {
            spoken = ",";  // a pause inside the sentence
        } else if (cp == 0xD7 && num_left && num_right) {
            spoken = "by";  // 1024&times;768
        } else if (cp == 0xB0 && last + 1 < n && doc.joined(last + 1) && (doc.str(last + 1) == "C" || doc.str(last + 1) == "F")) {
            out.tokens(i, last + 1, doc.str(last + 1) == "C" ? "degrees Celsius" : "degrees Fahrenheit", 85);
            i = last + 1;
            continue;
        } else if (cp == 0xA2 && i > 0 && doc.is(i - 1, "1") && num_left) {
            spoken = "cent";
        } else if (words) {
            spoken = words;
            if (spoken == ",") {
                out.add(doc[i - 1].end, last + 1 < n ? doc[last + 1].start : doc[last].end, ", ", 85);
                i = last;
                continue;
            }
        } else {
            rv::append_utf8(spoken, cp);
        }
        if (spoken.empty()) {
            Span s = deletion(doc, i, last);
            out.add(s.start, s.end, "", 85);
        } else {
            out.tokens(i, last, spoken, 85);
        }
        i = last;
    }
}

}  // namespace
