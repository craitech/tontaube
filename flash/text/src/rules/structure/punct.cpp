// Punctuation and spoken flow: dashes, ellipses, repeated marks, arrows, check marks, missing spaces.
#include "rules/structure/common.h"

namespace {

using namespace st;

bool same_line(const Doc& d, size_t i) { return i < d.size() && !d[i].newline_before; }
bool capital_word(const Doc& d, size_t i) {
    return i != Doc::npos && d.word(i) && (d[i].casing == rv::TITLE || d[i].casing == rv::UPPER) && d.str(i).size() > 1;
}

// A dash that sets off an insertion or an attribution is a comma: spaced en and em dashes, an em dash or a
// double hyphen between two words. Dashes between numbers are ranges and stay for the number rules.
RV_RULE("structure", dashes) {
    if (!(has(doc, "\xE2\x80\x93") || has(doc, "\xE2\x80\x94") || has(doc, "--") || has(doc, "\xE2\x80\x92") ||
          has(doc, "\xE2\x80\x95")))
        return;
    size_t n = doc.size();
    for (size_t i = 1; i + 1 < n; ++i) {
        if (!is_punct(doc, i) || !same_line(doc, i) || !same_line(doc, i + 1)) continue;
        uint32_t c = doc[i].cp;
        size_t last = i;
        bool dbl = c == '-' && p(doc, i + 1, '-') && doc.joined(i + 1) && !p(doc, i + 2, '-');
        if (dbl) last = i + 1;
        if (c == 0x2012 || c == 0x2015) c = 0x2014;  // figure dash, horizontal bar
        if (c != 0x2014 && c != 0x2013 && !dbl) continue;
        if (last + 1 >= n || !same_line(doc, last + 1)) continue;
        bool sp_left = doc[i].space_before, sp_right = doc[last + 1].space_before;
        size_t a = i - 1, b = last + 1;
        if ((doc.number(a) && (doc.number(b) || !(sp_left && sp_right))) || (doc.number(b) && !(sp_left && sp_right))) {
            i = last;
            continue;
        }
        if (sp_left && sp_right) {
            if (attaches_left(doc, a) && !is_punct(doc, a)) continue;
            if (is_punct(doc, a) && (doc[a].cp == ',' || doc[a].cp == ';' || doc[a].cp == ':')) continue;
            out.add(doc[a].end, doc[b].start, ", ", 45);
        } else if (!sp_left && !sp_right && (c == 0x2014 || dbl) && doc.word(a) && doc.word(b)) {
            // d—n and Mr. S—— are masks, read by their own rule; a lower-case one-letter word is one too
            if (doc.str(a).size() == 1 || (doc.str(b).size() == 1 && !doc.is(b, "a") && doc.str(b) != "I")) continue;
            out.tokens(i, last, ", ", 45);
        }
        i = last;
    }
}

// A bullet inside a line after the end of a sentence starts the next item and goes: "Read the warnings. • Wear
// eye protection. - Mix the solution." -> "Read the warnings. Wear eye protection. Mix the solution."
bool inline_bullet_cp(uint32_t c) {
    return c == 0x2022 || c == 0x25AA || c == 0x25E6 || c == 0x2023 || c == 0x25CF || c == 0x25CB || c == 0x25A0 ||
           c == 0x25BA || c == 0x27A4 || c == 0x2219 || c == '-' || c == '*' || c == '+' || c == 0x2013 || c == 0x2014;
}

RV_RULE("structure", inline_bullets) {
    std::string_view t = doc.text;
    auto bullet_after = [&](size_t k) {  // ". " or "! " or "? " and then a mark
        return k + 2 < t.size() && t[k + 1] == ' ' &&
               (t[k + 2] == '-' || t[k + 2] == '*' || t[k + 2] == '+' || static_cast<unsigned char>(t[k + 2]) >= 0xE2);
    };
    if (!(each_byte(t, '.', bullet_after) || each_byte(t, '!', bullet_after) || each_byte(t, '?', bullet_after))) return;
    size_t n = doc.size();
    for (size_t i = 2; i + 1 < n; ++i) {
        if (!is_punct(doc, i) || !inline_bullet_cp(doc[i].cp) || !same_line(doc, i) || !same_line(doc, i + 1)) continue;
        if (!doc[i].space_before || !doc[i + 1].space_before) continue;
        uint32_t c = doc[i - 1].kind == Kind::Punct ? doc[i - 1].cp : 0;
        if (!(c == '.' || c == '!' || c == '?')) continue;
        if (p(doc, i - 2, '.') && doc.word(i - 3) && doc.str(i - 3).size() == 1) continue;  // "J. - "
        bool next_cap = doc.word(i + 1) && doc[i + 1].casing != rv::LOWER;
        if (!next_cap) continue;
        out.add(doc[i - 1].end, doc[i + 1].start, " ", 62);
    }
}

// An ellipsis inside a sentence is a short pause (a comma); one that marks a cut-off title or entry goes, and so
// does one before a closing quote, a comma or the end of the line.
RV_RULE("structure", ellipsis) {
    if (!(has(doc, "..") || has(doc, "\xE2\x80\xA6") || has(doc, ". . ."))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        size_t e = i;
        if (p(doc, i, 0x2026)) {
            while (e + 1 < n && doc.joined(e + 1) && (p(doc, e + 1, 0x2026) || p(doc, e + 1, '.'))) ++e;
        } else if (p(doc, i, '.') && p(doc, i + 1, '.') && doc.joined(i + 1)) {
            while (e + 1 < n && doc.joined(e + 1) && (p(doc, e + 1, '.') || p(doc, e + 1, 0x2026))) ++e;
        } else if (p(doc, i, '.') && p(doc, i + 1, '.') && p(doc, i + 2, '.') && doc[i + 1].start == doc[i].end + 1 &&
                   doc[i + 2].start == doc[i + 1].end + 1 && same_line(doc, i + 2)) {
            e = i + 2;  // ". . ." spaced
            if (i > 0 && same_line(doc, i) && doc[i].space_before && e + 1 < n && same_line(doc, e + 1) && doc.word(e + 1)) {
                out.add(doc[i - 1].end, doc[e + 1].start, ", ", 40);  // "paused . . . then" -> "paused, then"
                i = e;
                continue;
            }
        } else {
            continue;
        }
        bool has_prev = i > 0 && same_line(doc, i);
        bool has_next = same_line(doc, e + 1);
        if (!has_prev && !has_next) {
            i = e;
            continue;
        }
        if (has_prev && has_next && doc.number(i - 1) && doc.number(e + 1) && doc.joined(i) && doc.joined(e + 1)) {
            i = e;
            continue;
        }
        if (!has_prev) {  // "...it holds" at the start of a line
            out.add(doc[i].start, doc[e + 1].start, "", 40);
        } else if (doc[i].space_before) {  // "Hierarchical ..." cut-off
            out.add(doc[i - 1].end, doc[e].end, "", 40);
        } else if (!has_next || attaches_left(doc, e + 1) || p(doc, e + 1, '"') || p(doc, e + 1, '\'') ||
                   p(doc, e + 1, 0x201D) || p(doc, e + 1, 0x2019) || p(doc, e + 1, '<')) {
            out.tokens(i, e, "", 40);
        } else if (doc[e + 1].space_before && i >= 2 && capital_word(doc, i - 1) && capital_word(doc, doc.prev_word(i - 1)) &&
                   doc.word(e + 1) && doc[e + 1].casing == rv::LOWER) {
            out.tokens(i, e, "", 40);  // a cut-off title inside a sentence: "Synthesis for Edge Devices… for the"
        } else {
            out.tokens(i, e, doc[e + 1].space_before ? "," : ", ", 40);
        }
        i = e;
    }
}

// Repeated ! and ? become one mark ("!!!1" in chat too), and a doubled comma or semicolon one.
RV_RULE("structure", repeated_marks) {
    if (!(has(doc, "!!") || has(doc, "??") || has(doc, "!?") || has(doc, "?!") || has(doc, ",,") || has(doc, ";;")))
        return;
    size_t n = doc.size();
    for (size_t i = 0; i + 1 < n; ++i) {
        if ((p(doc, i, ',') || p(doc, i, ';')) && i > 0 && doc.joined(i) && !is_punct(doc, i - 1) && p(doc, i + 1, doc[i].cp) &&
            doc.joined(i + 1) && (i + 2 >= n || doc[i + 2].space_before)) {
            out.tokens(i, i + 1, std::string(1, static_cast<char>(doc[i].cp)), 40);  // "Seriously,, why"
            ++i;
            continue;
        }
        if (!(p(doc, i, '!') || p(doc, i, '?'))) continue;
        size_t e = i;
        while (e + 1 < n && doc.joined(e + 1) && (p(doc, e + 1, '!') || p(doc, e + 1, '?'))) ++e;
        if (e >= i + 2 && doc.number(e + 1) && doc.joined(e + 1) &&
            doc.str(e + 1).find_first_not_of('1') == std::string_view::npos && (e + 2 >= n || doc[e + 2].space_before))
            ++e;  // the chat "!!!1"
        bool question = false;  // "?!" and "!?" ask: one question mark
        for (size_t k = i; k <= e; ++k) question |= p(doc, k, '?');
        if (e > i && (i > 0 && same_line(doc, i))) out.tokens(i, e, question ? "?" : "!", 40);
        i = e;
    }
}

bool arrow_at(const Doc& d, size_t i, size_t& last) {
    last = i;
    if (!is_punct(d, i)) return false;
    uint32_t c = d[i].cp;
    if (c == 0x2192 || c == 0x27F6 || c == 0x21E8 || c == 0x2794 || c == 0x279C) return true;
    if (c == '-' || c == '=') {
        size_t k = i;
        if (c == '-' && p(d, k + 1, '-') && d.joined(k + 1)) ++k;
        if (p(d, k + 1, '>') && d.joined(k + 1) && !p(d, k + 2, '>')) {
            last = k + 1;
            // "=>" is an arrow only spaced between words, in a chain of two or more ("draft => review => done")
            if (c == '=') {
                if (!d[i].space_before || last + 1 >= d.size() || !d[last + 1].space_before) return false;
                for (size_t m = last + 2; m < d.size() && m < last + 10 && !d[m].newline_before; ++m)
                    if (p(d, m, '=') && p(d, m + 1, '>') && d.joined(m + 1) && d[m].space_before) return true;
                for (size_t m = i; m-- > 0 && m + 10 > i && !d[m + 1].newline_before;)
                    if (p(d, m, '>') && m > 0 && p(d, m - 1, '=') && d.joined(m)) return true;
                return false;
            }
            return true;
        }
    }
    return false;
}

bool edge_arrow(uint32_t c) {
    return c == 0x2190 || c == 0x2192 || c == 0x27F5 || c == 0x27F6 || c == 0x21E6 || c == 0x21E8 || c == 0x2B05 ||
           c == 0x27A1 || c == 0x2794 || c == 0x279C || c == 0x21D0 || c == 0x21D2;
}

// The first token of an item in a chain starts with a capital letter.
bool capital_item(const Doc& d, size_t i) {
    return d.word(i) && (d[i].casing == rv::TITLE || d[i].casing == rv::MIXED ||
                         (d[i].casing == rv::UPPER && d.str(i).size() <= 4));
}

const rv::WordSet NAV_CUES{"open", "opens", "select", "selecting", "tap", "click", "choose", "go", "navigate",
                           "under", "via", "in", "press", "visit", "find", "at", "page", "menu", "path"};

// Arrows between steps or states are "to"; between menu steps (every item capitalised, after a navigation word)
// they are ", then". An up or down arrow before a number is "up" or "down". (Menu paths written with '>' are
// electronic's.)
RV_RULE("structure", arrows) {
    if (!(has_any(doc, ">") || has(doc, "\xE2\x86") || has(doc, "\xE2\x87") || has(doc, "\xE2\x9F\xB6") || has(doc, "\xE2\xAC") || has(doc, "\xE2\x9E"))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        // an arrow sign at the start or the end of a line points to a link ("← 2018", "2026 →", "Next ⟶"): it goes
        if (is_punct(doc, i) && edge_arrow(doc[i].cp) && n > 1 &&
            (i == 0 || doc[i].newline_before || i + 1 == n || doc[i + 1].newline_before)) {
            Span s = deletion(doc, i, i);
            out.add(s.start, s.end, "", 53);
            continue;
        }
        if (p(doc, i, 0x2191) || p(doc, i, 0x2193) || p(doc, i, 0x2B06) || p(doc, i, 0x2B07)) {
            bool up = doc[i].cp == 0x2191 || doc[i].cp == 0x2B06;
            bool said = i > 0 && doc.is(i - 1, up ? "up" : "down");  // "is up ↑3%"
            out.tokens(i, i, said ? "" : up ? "up" : "down", 52);
            continue;
        }
        size_t last;
        if (!arrow_at(doc, i, last)) continue;
        bool symbol = last == i;  // a single arrow sign (not "->") is "to" wherever it stands
        if (i == 0 || !same_line(doc, i) || last + 1 >= n || !same_line(doc, last + 1) ||
            (doc.number(i - 1) && doc.number(last + 1))) {  // 3->4 stays for the number rules
            if (symbol) out.tokens(i, i, "to", 52);
            i = last;
            continue;
        }
        // the chain on this line: items separated by arrows
        std::vector<std::pair<size_t, size_t>> seps{{i, last}};
        bool all_cap = capital_item(doc, i - 1);
        size_t k = last + 1;
        while (true) {
            all_cap &= capital_item(doc, k);
            size_t m = k, lst = k;
            bool found = false;
            while (m < n && same_line(doc, m) && m - k < 8) {
                if (arrow_at(doc, m, lst)) {
                    found = m > k;
                    break;
                }
                if (is_punct(doc, m) && !p(doc, m, '-') && !p(doc, m, '&')) break;
                ++m;
            }
            if (!found || lst + 1 >= n || !same_line(doc, lst + 1)) break;
            seps.push_back({m, lst});
            k = lst + 1;
        }
        size_t first_item = i - 1;
        while (first_item > 0 && same_line(doc, first_item) && capital_item(doc, first_item - 1) &&
               !doc[first_item].newline_before)
            --first_item;
        bool cue = false;  // a navigation word among the three words before the path ("Tap Menu", "go to Library")
        for (size_t k = first_item + 1, seen = 0; k-- > 0 && seen < 3 && !cue && (k + 1 == first_item + 1 || !doc[k + 1].newline_before);) {
            if (doc[k].kind == Kind::Punct) continue;
            ++seen;
            cue = NAV_CUES.contains(doc.low(k));
        }
        bool then = all_cap && cue;  // "tap Profile → Edit" is a menu path; "Status: Ordered → Packed" is not
        // a menu path must beat the plain readings of the arrow
        for (auto [a, b] : seps) out.add(doc[a - 1].end, doc[b + 1].start, then ? ", then " : " to ", then ? 65 : 52);
        i = seps.back().second;
    }
}

// A symbol quoted on its own is named: "a '|' symbol" -> "a 'pipe' symbol", "a small '†'" -> "a small 'dagger'".
const char* symbol_name(uint32_t c) {
    switch (c) {
        case '|': return "pipe";
        case '^': return "caret";
        case 0x2020: return "dagger";
        case 0x2021: return "double dagger";
        case 0xB6: return "paragraph";
        case 0xA7: return "section";
        case '*': return "asterisk";
        case '~': return "tilde";
        case '\\': return "backslash";
        case '_': return "underscore";
        case '#': return "hash";
        case '@': return "at";
        case '&': return "and";
        case '+': return "plus";
        case '=': return "equals";
        case '%': return "percent";
        case '/': return "slash";
        case '<': return "less than";
        case '>': return "greater than";
        default: return nullptr;
    }
}

RV_RULE("structure", quoted_symbols) {
    std::string_view t = doc.text;
    auto quoted = [&](size_t k) {  // 'x' with a one-byte symbol, or a quote before a character of several bytes
        return k + 2 < t.size() && ((t[k + 2] == t[k] && !rv::is_ascii_alnum(t[k + 1]) && t[k + 1] != ' ') ||
                                    static_cast<unsigned char>(t[k + 1]) >= 0xC2);
    };
    if (!(each_byte(t, '\'', quoted) || each_byte(t, '"', quoted) || each_byte(t, '`', quoted) ||
          has(doc, "\xE2\x80\x98") || has(doc, "\xE2\x80\x9C")))
        return;
    size_t n = doc.size();
    for (size_t i = 1; i + 1 < n; ++i) {
        if (!is_punct(doc, i) || !doc.joined(i) || !doc.joined(i + 1) || !is_punct(doc, i - 1) || !is_punct(doc, i + 1))
            continue;
        uint32_t a = doc[i - 1].cp, b = doc[i + 1].cp;
        bool quoted = (a == '\'' && b == '\'') || (a == '"' && b == '"') || (a == 0x2018 && b == 0x2019) ||
                      (a == 0x201C && b == 0x201D) || (a == '`' && b == '`');
        if (!quoted) continue;
        // the quotes stand apart: a space or a line start before, a space or punctuation after
        bool left = i - 1 == 0 || doc[i - 1].space_before;
        bool right = i + 2 >= n || doc[i + 2].space_before || is_punct(doc, i + 2);
        const char* name = symbol_name(doc[i].cp);
        if (!name || !left || !right) continue;
        out.tokens(i, i, name, 75);
        ++i;
    }
}

// Label-value pairs inside a line: a spaced colon before a value is a pause ("Validation loss : 0.045" ->
// "Validation loss, zero point zero four five"), and a wide gap before the next label separates the pairs
// ("humidity: 62%   pressure: 1012 hPa" -> "..., pressure: ...").
RV_RULE("structure", label_values) {
    bool any = false;  // a spaced colon, or a gap of three spaces
    for (size_t p = doc.text.find(':'); p != std::string_view::npos && !any; p = doc.text.find(':', p + 1))
        any = p > 0 && doc.text[p - 1] == ' ';
    for (size_t i = 1; i < doc.size() && !any; ++i) any = doc[i].start >= doc[i - 1].end + 3;
    if (!any) return;
    size_t n = doc.size();
    for (size_t i = 1; i + 1 < n; ++i) {
        if (!same_line(doc, i) || !same_line(doc, i + 1)) continue;
        if (p(doc, i, ':') && doc[i].space_before && doc[i + 1].space_before && doc.word(i - 1)) {
            size_t v = i + 1;
            bool value = doc.number(v) || ((p(doc, v, '.') || p(doc, v, '$') || p(doc, v, 0x20AC) || p(doc, v, 0xA3) ||
                                            p(doc, v, '-') || p(doc, v, '+')) &&
                                           doc.number(v + 1) && doc.joined(v + 1));
            if (value) out.add(doc[i - 1].end, doc[v].start, ", ", 45);
            continue;
        }
        // a gap of three or more spaces before "label:" (not right after a colon)
        if (doc.word(i) && (p(doc, i + 1, ':') || (doc.word(i + 1) && p(doc, i + 2, ':') && doc.joined(i + 2))) &&
            !p(doc, i - 1, ':') && doc[i].start >= doc[i - 1].end + 3) {
            std::string_view gap = doc.slice(doc[i - 1].end, doc[i].start);
            if (gap.find_first_not_of(' ') == std::string_view::npos && !attaches_left(doc, i - 1))
                out.add(doc[i - 1].end, doc[i].start, ", ", 40);
        }
    }
}

// A check mark or cross that stands as a value is "yes" or "no".
RV_RULE("structure", check_marks) {
    if (!(has(doc, "\xE2\x9C") || has(doc, "\xE2\x98") || has(doc, "\xE2\x9D"))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        if (!is_punct(doc, i)) continue;
        uint32_t c = doc[i].cp;
        bool yes = c == 0x2713 || c == 0x2714 || c == 0x2611 || c == 0x2705;
        bool no = c == 0x2717 || c == 0x2718 || c == 0x2612 || c == 0x274C || c == 0x2716 || c == 0x274E;
        if (!yes && !no) continue;
        bool line_start = i == 0 || doc[i].newline_before;
        bool alone = !same_line(doc, i + 1);
        if (line_start && !alone) continue;  // a decoration, removed by the emoji rule
        bool emoji = c == 0x2705 || c == 0x274C || c == 0x2716 || c == 0x274E;
        // an emoji check mark is a value only after a label or in a cell ("Status: ✅", "| ✅ |")
        if (emoji && !(i > 0 && (p(doc, i - 1, ':') || p(doc, i - 1, '|')) && (alone || p(doc, i + 1, '|') ||
                                                                              p(doc, i + 1, ',') || p(doc, i + 1, ';'))))
            continue;
        out.tokens(i, i, yes ? "yes" : "no", 61);
    }
}

// A circled number inside a line is its number ("press ② to confirm" -> "press two to confirm"); at the start of a
// list line the layout rule reads it as a list number.
RV_RULE("structure", circled_numbers) {
    if (!(has(doc, "\xE2\x91") || has(doc, "\xE2\x9D") || has(doc, "\xE2\x9E"))) return;
    for (size_t i = 0; i < doc.size(); ++i) {
        uint32_t c = doc[i].cp;
        int v = !is_punct(doc, i) ? 0
                : (c >= 0x2460 && c <= 0x2473) ? static_cast<int>(c - 0x2460 + 1)
                : (c >= 0x2776 && c <= 0x277F) ? static_cast<int>(c - 0x2776 + 1)
                : (c >= 0x2780 && c <= 0x2789) ? static_cast<int>(c - 0x2780 + 1)
                : (c >= 0x278A && c <= 0x2793) ? static_cast<int>(c - 0x278A + 1) : 0;
        if (v) out.tokens(i, i, rv::cardinal(static_cast<uint64_t>(v)), 20);
    }
}

// Full-width punctuation in an English sentence is the plain mark: "Results，as expected，were good！" -> "Results,
// as expected, were good!". (Text in Chinese or Japanese is kept whole by the language rule.)
const char* fullwidth(uint32_t c) {
    switch (c) {
        case 0xFF0C: case 0x3001: return ",";
        case 0xFF01: return "!";
        case 0xFF1F: return "?";
        case 0xFF1A: return ":";
        case 0xFF1B: return ";";
        case 0xFF0E: case 0x3002: return ".";
        default: return nullptr;
    }
}

RV_RULE("structure", fullwidth_marks) {
    if (!has_any(doc, "\xEF\xE3") || !(has(doc, "\xEF\xBC") || has(doc, "\xE3\x80"))) return;
    size_t n = doc.size();
    for (size_t i = 1; i < n; ++i) {
        const char* m = is_punct(doc, i) ? fullwidth(doc[i].cp) : nullptr;
        if (!m || !(doc.word(i - 1) || doc.number(i - 1)) || static_cast<unsigned char>(doc.str(i - 1)[0]) >= 0x80) continue;
        bool glued_next = i + 1 < n && doc.joined(i + 1) && (doc.word(i + 1) || doc.number(i + 1));
        out.tokens(i, i, std::string(m) + (glued_next ? " " : ""), 40);
    }
}

const rv::WordSet TLDS{"com", "org", "net", "io", "gov", "edu", "co", "uk", "de", "ai", "dev", "app", "info", "biz"};

// A missing space after a sentence period ("ports.Its") or after a comma between words ("once,and").
RV_RULE("structure", missing_space) {
    size_t n = doc.size();
    for (size_t i = 1; i + 1 < n; ++i) {
        if (!doc.joined(i + 1)) continue;
        if (p(doc, i, '.') && doc.word(i + 1) && doc[i + 1].casing == rv::TITLE && !doc.joined(i) &&
            doc[i].space_before && !doc[i].newline_before && (doc.word(i - 1) || doc.number(i - 1))) {
            out.add(doc[i - 1].end, doc[i].end, ". ", 35);  // "today .The"
            continue;
        }
        if (p(doc, i, ',') && !doc.joined(i) && doc[i].space_before && !doc[i].newline_before && doc.word(i + 1) &&
            (doc.word(i - 1) || doc.number(i - 1))) {
            out.add(doc[i - 1].end, doc[i].end, ", ", 35);  // "incomplete ,and"
            continue;
        }
        if (!doc.joined(i)) continue;
        if ((p(doc, i, '!') || p(doc, i, '?')) && doc.word(i - 1) && doc[i - 1].casing != rv::UPPER &&
            doc.str(i - 1).size() > 1 && doc.word(i + 1) && doc[i + 1].casing == rv::TITLE && doc.str(i + 1).size() > 1 &&
            !(i >= 2 && doc.joined(i - 1) && is_punct(doc, i - 2))) {
            out.tokens(i, i, std::string(1, static_cast<char>(doc[i].cp)) + " ", 35);  // "ready?Yes" -> "ready? Yes"
            continue;
        }
        bool closer = is_punct(doc, i - 1) && (doc[i - 1].cp == '"' || doc[i - 1].cp == 0x201D ||
                                               doc[i - 1].cp == 0x2019 || doc[i - 1].cp == '\'' ||
                                               doc[i - 1].cp == ')' || doc[i - 1].cp == '%');
        if (p(doc, i, '.') && (doc.word(i - 1) || doc.number(i - 1) || closer) && doc.word(i + 1) &&
            doc[i + 1].casing == rv::TITLE) {
            if (doc.word(i - 1) && doc.str(i - 1).size() == 1) continue;           // J.R.Tolkien
            if (i >= 2 && p(doc, i - 2, '.') && doc.joined(i - 1)) continue;        // a.b.Com
            if (p(doc, i + 2, '.') && doc.joined(i + 2) && doc.word(i + 3)) continue;  // Example.Com.au
            if (TLDS.contains(doc.low(i + 1)) && doc.word(i - 1) && !(i + 2 < n && doc.joined(i + 2) && doc.word(i + 2)))
                continue;  // example.Com
            if (doc.number(i - 1) && i >= 2 && doc.joined(i - 1) && !is_punct(doc, i - 2)) continue;
            out.tokens(i, i, ". ", 35);
        } else if ((p(doc, i, ',') || p(doc, i, ';')) && doc.word(i - 1) && doc.word(i + 1) && doc.str(i - 1).size() > 1 &&
                   doc.str(i + 1).size() > 1 && doc[i + 1].casing != rv::UPPER) {
            out.tokens(i, i, std::string(1, static_cast<char>(doc[i].cp)) + " ", 20);
        }
    }
}

}  // namespace
