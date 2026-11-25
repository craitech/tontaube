// Menu paths written with '>' (Settings > Privacy > Clear), code operators (>=, !=, &&, ||), identifiers that
// stay as written (ORD-7731-QX9P-0042, DOIs, password hashes).
#include "rules/electronic/common.h"

namespace {

using namespace rv::el;
using rv::Doc;
using rv::Kind;

// A '>' at token i that separates menu steps: spaced on both sides, on one line. "&gt;" counts as '>'.
// Returns the last token of the separator, or Doc::npos.
size_t nav_sep(const Doc& doc, size_t i) {
    if (i == 0 || i + 1 >= doc.size() || !doc[i].space_before || doc[i].newline_before) return Doc::npos;
    size_t last = i;
    if (doc.punct(i, '&')) {
        if (!doc.is(i + 1, "gt") || !doc.joined(i + 1) || !doc.punct(i + 2, ';') || !doc.joined(i + 2)) return Doc::npos;
        last = i + 2;
    } else if (!doc.punct(i, '>')) {
        return Doc::npos;
    }
    if (last + 1 >= doc.size() || !doc[last + 1].space_before || doc[last + 1].newline_before) return Doc::npos;
    return last;
}

// A step of a menu path from token k: up to four words or numbers (with '-' and '&' inside); the first word has a
// capital unless lower is allowed. Returns the last token of the step, or Doc::npos.
size_t nav_item(const Doc& doc, size_t k, bool lower) {
    if (k >= doc.size() || !doc.word(k)) return Doc::npos;
    if (!lower && !(doc[k].casing & (rv::TITLE | rv::UPPER | rv::MIXED))) return Doc::npos;
    size_t e = k, words = 1;
    while (e + 1 < doc.size() && !doc[e + 1].newline_before) {
        size_t n = e + 1;
        if ((doc.punct(n, '-') || doc.punct(n, '&')) && doc.joined(n) && doc[n + 1].kind != Kind::Punct &&
            doc.joined(n + 1)) {
            e = n + 1;  // self-service, Q&A, Lumen-7B
            while (e + 1 < doc.size() && doc.joined(e + 1) && doc[e + 1].kind != Kind::Punct) ++e;
            continue;
        }
        if (doc[n].kind == Kind::Punct || ++words > 4) break;
        e = n;
    }
    return e;
}

const rv::WordSet NAV_CUES{"choose", "click", "find", "menu", "navigate", "open", "opens", "path", "press",
                           "select", "tap", "under", "via"};
const rv::WordSet GO_VERBS{"go", "goes", "head", "jump", "navigate", "switch"};

// One '>' between two menu names (Settings > Privacy): both steps start with a capital and are not single
// letters (X > Y compares), after a menu verb ("open", "go to", "under") or inside quotes.
bool single_step(const Doc& doc, size_t sep, size_t last) {
    size_t a = sep - 1, b = last + 1;
    if (!(doc[a].casing & (rv::TITLE | rv::MIXED)) || !(doc[b].casing & (rv::TITLE | rv::MIXED))) return false;
    size_t first = a;
    while (first > 0 && doc.word(first - 1) && !doc[first].newline_before && (doc[first - 1].casing & rv::TITLE) &&
           a - first < 3)
        --first;
    if (first > 0 && (doc.punct(first - 1, '\'') || doc.punct(first - 1, '"') || doc.punct(first - 1, 0x2018) ||
                      doc.punct(first - 1, 0x201C)))
        return true;
    size_t pw = doc.prev_word(first);
    if (pw == Doc::npos || pw + 1 != first) return false;
    if (NAV_CUES.contains(doc.low(pw))) return true;
    size_t p2 = doc.prev_word(pw);
    return doc.is(pw, "to") && p2 != Doc::npos && GO_VERBS.contains(doc.low(p2));
}

// Menu paths: "the self-service tab > Requests > Summary" -> "the self-service tab, then Requests, then Summary",
// "Pay &gt; Transfers &gt; New". Every step after the first starts with a capital; two separators at least, or one
// after a menu verb.
// (structure reads the paths whose every step has a capital; this one adds a lower-case first step and the
// escaped sign.)
RV_RULE("electronic", menu_path) {
    const std::string_view t = doc.text;
    if (t.find('>') == std::string_view::npos && t.find("&gt;") == std::string_view::npos) return;
    size_t done = 0;
    for (size_t i = 1; i < doc.size(); ++i) {
        if (i < done || !(doc.punct(i, '>') || doc.punct(i, '&'))) continue;
        size_t last = nav_sep(doc, i);
        if (last == Doc::npos || !doc.word(i - 1)) continue;
        // the steps after the first
        size_t seps[8][2];
        int n = 0;
        size_t s = i, l = last;
        while (n < 8) {
            size_t e = nav_item(doc, l + 1, false);
            if (e == Doc::npos) break;
            seps[n][0] = s;
            seps[n][1] = l;
            ++n;
            size_t ns = e + 1;
            if (ns >= doc.size() || (l = nav_sep(doc, ns)) == Doc::npos) break;
            s = ns;
        }
        if (n == 1 && !single_step(doc, i, seps[0][1])) n = 0;
        if (n == 0) {
            done = last + 1;
            continue;
        }
        for (int k = 0; k < n; ++k) out.add(doc[seps[k][0] - 1].end, doc[seps[k][1] + 1].start, ", then ", 86);
        done = seps[n - 1][1] + 1;
    }
}

}  // namespace

namespace {

const char* operator_words(char a, char b) {
    if (b == '=') {
        switch (a) {
            case '>': return "greater than or equal to";
            case '<': return "less than or equal to";
            case '!': return "not equal to";
            case '=': return "equals";
            case '+': return "plus equals";
            case '-': return "minus equals";
            default: return nullptr;
        }
    }
    if (a == '&' && b == '&') return "and";
    if (a == '|' && b == '|') return "or";
    return nullptr;
}

// Two-character operators of code where the symbol rules leave them: joined on both sides (buffer>=10), joined
// to a number after a space (>=0.25, <=120), alone in backticks (`||`).
RV_RULE("electronic", code_operator) {
    const std::string_view t = doc.text;
    auto at = [&](size_t i) {
        if (i + 1 >= doc.size() || !doc.joined(i + 1) || doc[i + 1].kind != Kind::Punct) return;
        const char* w = operator_words(t[doc[i].start], t[doc[i + 1].start]);
        if (!w || (i + 2 < doc.size() && doc.joined(i + 2) && doc[i + 2].kind == Kind::Punct &&
                   (doc.punct(i + 2, '=') || doc.punct(i + 2, '>') || doc.punct(i + 2, '<') || doc.punct(i + 2, '|') ||
                    doc.punct(i + 2, '&'))))
            return;
        if (i > 0 && doc.joined(i) && doc[i - 1].kind == Kind::Punct &&
            !(doc.punct(i - 1, '`') || doc.punct(i - 1, ')') || doc.punct(i - 1, ']')))
            return;
        bool left = i > 0 && doc.joined(i), right = i + 2 < doc.size() && doc.joined(i + 2);
        bool tick = left && right && doc.punct(i - 1, '`') && doc.punct(i + 2, '`');
        bool glued = left && right && doc[i - 1].kind != Kind::Punct &&
                     (doc[i + 2].kind != Kind::Punct || (doc.punct(i + 2, '-') && doc.number(i + 3)));
        bool value = !left && right && (doc.number(i + 2) || (doc.punct(i + 2, '.') && doc.number(i + 3)) ||
                                        (doc.punct(i + 2, '-') && doc.number(i + 3)));
        // && and || with spaces: one word for the pair, never "and and"
        char c0 = t[doc[i].start];
        bool logic = (c0 == '&' || c0 == '|') && doc[i].space_before && i > 0 && i + 2 < doc.size() &&
                     doc[i + 2].space_before && !doc[i + 2].newline_before;
        if (!(tick || glued || value || logic)) return;
        if (glued && (t[doc[i].start] == '<' || t[doc[i].start] == '>') && doc.word(i - 1) && doc.word(i + 2) &&
            !idc(t[doc[i - 1].end - 1]))
            return;
        out.tokens(i, i + 1, w, 46);
    };
    // compound assignment with spaces: x += 1, total -= cost
    auto spaced = [&](size_t i) {
        if (i == 0 || !doc.punct(i + 1, '=') || !doc.joined(i + 1) || !doc[i].space_before || i + 2 >= doc.size() ||
            !doc[i + 2].space_before || doc[i + 2].newline_before)
            return;
        if (doc.joined(i + 2) || !(doc.word(i - 1) || doc.punct(i - 1, ']') || doc.punct(i - 1, ')'))) return;
        if (!(doc.word(i + 2) || doc.number(i + 2))) return;
        out.tokens(i, i + 1, t[doc[i].start] == '+' ? "plus equals" : "minus equals", 46);
    };
    // increments: i++ -> i plus plus (not C++, which is a name)
    auto incr = [&](size_t i) {
        if (i == 0 || !doc.punct(i + 1, '+') || !doc.joined(i) || !doc.joined(i + 1) || !doc.word(i - 1)) return;
        if (doc.str(i - 1).size() == 1 && up(doc.str(i - 1)[0])) return;  // C++
        if (i + 2 < doc.size() && doc.joined(i + 2) && !(doc[i + 2].kind == Kind::Punct && !doc.punct(i + 2, '+')))
            return;
        if (doc.punct(i + 2, '+')) return;
        if (!idc(t[doc[i - 1].start]) || !lo(t[doc[i - 1].start])) return;
        out.tokens(i, i + 1, "plus plus", 46);
    };
    // one pass over the punctuation that is joined to the next token (none of the signs: nothing to do)
    bool any = false;
    for (char c : {'=', '&', '|', '+'}) any = any || t.find(c) != std::string_view::npos;
    if (!any) return;
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        if (doc[i].kind != Kind::Punct || doc[i + 1].kind != Kind::Punct || !doc.joined(i + 1)) continue;
        switch (doc[i].cp) {
            case '>': case '<': case '!': case '=': case '&': case '|':
                at(i);
                break;
            case '+':
                at(i);
                spaced(i);
                incr(i);
                break;
            case '-':
                at(i);
                spaced(i);
                break;
            default:
                break;
        }
    }
}

}  // namespace

namespace {

// Reference codes of capitals and digit groups stay as written: ORD-7731-QX9P-0042, AB-93817-XQ44, QX-88-4491.
// Three or more groups, the first of capitals, two with digits, one of them long (four digits, or letters and
// digits); never a lower-case word (GPT-4o-mini) or a year range (FY-2025-26).
bool code_id(std::string_view s) {
    std::string_view seg[12];
    int n = 0;
    size_t k = 0;
    while (k <= s.size() && n < 12) {
        size_t j = s.find('-', k);
        if (j == std::string_view::npos) j = s.size();
        seg[n++] = s.substr(k, j - k);
        k = j + 1;
    }
    if (n < 3 || k <= s.size() || seg[0].size() > 5 || !all_upper(seg[0])) return false;
    int digit_groups = 0;
    bool strong = false;
    for (int m = 0; m < n; ++m) {
        std::string_view g = seg[m];
        if (g.empty()) return false;
        bool d = false, l = false, u = false;
        for (char c : g) {
            if (dg(c)) d = true;
            else if (lo(c)) l = true;
            else if (up(c)) u = true;
            else return false;
        }
        if (l && !d && !u) return false;
        if (d) ++digit_groups;
        if ((d && !l && !u && g.size() >= 4) || (d && (l || u) && g.size() >= 4)) strong = true;
    }
    if (n == 3 && all_digits(seg[2]) && seg[2].size() <= 2) return false;
    return digit_groups >= 2 && strong;
}

// Identifiers kept as written: reference codes, DOIs without their label (10.5281/zenodo.1234567), password
// hashes ($2b$12$...).
RV_RULE("electronic", keep_codes) {
    const std::string_view t = doc.text;
    auto run_end = [&](uint32_t b) {
        uint32_t e = b;
        while (e < t.size() && static_cast<unsigned char>(t[e]) > ' ' && t[e] != '"' && t[e] != '<' && t[e] != '`')
            ++e;
        return trim_tail(t, b, e);
    };
    each_mark(doc, '-', [&](size_t k) {
        if (k == 0 || !doc.joined(k) || !doc.word(k - 1) || doc[k - 1].casing != rv::UPPER) return;
        uint32_t b = doc[k - 1].start;
        if (k >= 2 && doc.joined(k - 1) && doc[k - 2].kind != Kind::Punct) return;
        if (!left_edge(doc, b)) return;
        uint32_t e = b;
        while (e < t.size() && (an(t[e]) || t[e] == '-')) ++e;
        if (!right_edge(doc, e) || (e + 1 < t.size() && (t[e] == '/' || t[e] == '.') && an(t[e + 1]))) return;
        if (code_id(t.substr(b, e - b))) out.add(b, e, std::string(t.substr(b, e - b)), 92);
    });
    each_mark(doc, '/', [&](size_t k) {
        // 10.NNNN/suffix
        if (k < 3 || !doc.joined(k) || !doc.number(k - 1) || !doc.punct(k - 2, '.') || !doc.number(k - 3) ||
            doc.str(k - 3) != "10" || !doc.joined(k - 1) || !doc.joined(k - 2))
            return;
        size_t n = doc.str(k - 1).size();
        uint32_t b = doc[k - 3].start;
        if (n < 4 || n > 9 || !left_edge(doc, b)) return;
        uint32_t e = run_end(doc[k].end);
        if (e < doc[k].end + 2) return;
        out.add(b, e, std::string(t.substr(b, e - b)), 92);
    });
    each_mark(doc, '$', [&](size_t k) {
        uint32_t b = doc[k].start;
        if (!left_edge(doc, b)) return;
        bool hash = doc.text_at(b, "$2a$") || doc.text_at(b, "$2b$") || doc.text_at(b, "$2y$") ||
                    doc.text_at(b, "$argon2") || doc.text_at(b, "$6$") || doc.text_at(b, "$5$") || doc.text_at(b, "$1$");
        if (!hash) return;
        uint32_t e = run_end(b);
        if (e - b >= 20) out.add(b, e, std::string(t.substr(b, e - b)), 92);
    });
}

}  // namespace

namespace {

bool hex_c(char c) { return dg(c) || (lower_c(c) >= 'a' && lower_c(c) <= 'f'); }

}  // namespace

namespace rv::el {

// IPv6 addresses stay as written: 2001:db8::1, fe80::1ff:fe23:4567:890a (hex groups and colons; "::" or five
// groups and more, never a clock time).
void ipv6_at(const Doc& doc, rv::Out& out, size_t k, uint32_t& last) {
    const std::string_view t = doc.text;
    uint32_t c = doc[k].start;
    if (c < last) return;
    uint32_t b = c;
    while (b > 0 && (hex_c(t[b - 1]) || t[b - 1] == ':')) --b;
    uint32_t e = c;
    while (e < t.size() && (hex_c(t[e]) || t[e] == ':')) ++e;
    last = e;
    if (!left_edge(doc, b) || !right_edge(doc, e) || e - b < 6) return;
    std::string_view s = t.substr(b, e - b);
    if (e < t.size() && t[e] == '.' && e + 1 < t.size() && dg(t[e + 1])) return;  // ::ffff:192.0.2.1 is rare
    int groups = 0, doubles = 0;
    bool letter = false;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == ':') {
            if (i + 1 < s.size() && s[i + 1] == ':') {
                ++doubles;
                i += 2;
            } else {
                ++i;
            }
            continue;
        }
        size_t j = i;
        while (j < s.size() && hex_c(s[j])) {
            letter |= !dg(s[j]);
            ++j;
        }
        if (j - i > 4) return;
        ++groups;
        i = j;
    }
    if (doubles > 1 || s.find(":::") != std::string_view::npos) return;
    if (!(doubles == 1 ? groups >= 2 : groups >= 5) || (!letter && doubles == 0)) return;
    out.add(b, e, std::string(s), 92);
}

}  // namespace rv::el

namespace {

}  // namespace
