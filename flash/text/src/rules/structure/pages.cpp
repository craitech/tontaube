// What web pages leave in their text: pronunciations in phonetic letters and names in scripts the speech model
// cannot read, in brackets or after a language label; empty brackets and stray commas where parts of the page were
// left out.
#include "rules/structure/common.h"

namespace st {

namespace {

// Letters that only a phonetic transcription has: IPA letters, stress and length marks, raised h j r w y.
bool ipa_cp(uint32_t cp) {
    return (cp >= 0x250 && cp <= 0x2B8) || cp == 0x2C8 || cp == 0x2CC || cp == 0x2D0 || cp == 0x2D1;
}

// Letters and marks of scripts other than Latin and Greek.
bool script_cp(uint32_t cp) {
    return (cp >= 0x400 && cp <= 0x52F) || (cp >= 0x530 && cp <= 0x8FF) || (cp >= 0x900 && cp <= 0xDFF) ||
           (cp >= 0xE00 && cp <= 0xFFF) || (cp >= 0x1000 && cp <= 0x11FF) || (cp >= 0x1200 && cp <= 0x139F) ||
           (cp >= 0x1780 && cp <= 0x17FF) || (cp >= 0x3040 && cp <= 0x31FF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x20000 && cp <= 0x2FFFF);
}
// Punctuation that goes with those scripts inside a name (、・, full-width forms); not the sentence ends 。！？.
bool script_mark(uint32_t cp) {
    return (cp >= 0x3000 && cp <= 0x303F && cp != 0x3002) || (cp >= 0xFF00 && cp <= 0xFFEF && cp != 0xFF01 && cp != 0xFF1F &&
                                                            cp != 0xFF0E);
}

bool latin_word(const Doc& d, size_t k) { return d.word(k) && d[k].cp < 0x250; }
bool comma_or_semicolon(const Doc& d, size_t k) { return p(d, k, ',') || p(d, k, ';'); }

// The line before the one that starts at token a is a label of one to three words ("Kanji.", "Hiragana:").
bool label_line_before(const Doc& d, size_t a) {
    if (a == 0 || !d[a].newline_before) return false;
    size_t s = a - 1;
    while (s > 0 && !d[s].newline_before) --s;
    size_t e = a;
    if (p(d, e - 1, '.') || p(d, e - 1, ':')) --e;
    if (e == s || e - s > 3) return false;
    for (size_t k = s; k < e; ++k)
        if (!latin_word(d, k)) return false;
    return true;
}
bool delim(const Doc& d, size_t k) { return p(d, k, '(') || p(d, k, ')') || p(d, k, ',') || p(d, k, ';'); }

// A phonetic span on the line [a, b): '/' or '[' at token s, the closing mark at token e, IPA letters inside.
struct Ipa {
    size_t s, e;
};

std::vector<Ipa> ipa_spans(const Doc& d, size_t a, size_t b) {
    std::vector<Ipa> v;
    for (size_t k = a; k < b; ++k) {
        if (!(p(d, k, '/') || p(d, k, '[')) || k + 1 >= b || !d.joined(k + 1)) continue;
        uint32_t close = d[k].cp == '[' ? ']' : '/';
        bool twice = close == ']' && p(d, k + 1, '[');  // "[[t͡ɕʰòmó]]" from wiki markup
        bool ipa = false;
        size_t e = k + 1 + twice;
        for (; e < b && e < k + 80 && !p(d, e, close) && !p(d, e, '(') && !p(d, e, ')') && !(close == '/' && p(d, e, '['));
             ++e)
            ipa |= ipa_cp(d[e].cp) && d[e].kind == Kind::Punct;
        if (!ipa || e >= b || !p(d, e, close)) continue;
        if (twice) {
            if (!p(d, e + 1, ']') || !d.joined(e + 1)) continue;
            ++e;
        }
        if (e + 1 < b && d.joined(e + 1) && (d.word(e + 1) || d.number(e + 1))) continue;
        v.push_back({k, e});
        k = e;
    }
    return v;
}

// The first token of an item's content after a label of one to three words and a colon ("French:", "Nepali
// pronunciation:"), or after "pronounced"; s when there is none. *pron: the label speaks of a pronunciation.
size_t after_label(const Doc& d, size_t s, size_t e, bool* pron) {
    *pron = false;
    if (d.is(s, "pronounced") && s < e) {
        *pron = true;
        return s + 1;
    }
    size_t k = s;
    while (k <= e && k < s + 3 && latin_word(d, k)) ++k;
    if (k == s || k > e || !p(d, k, ':') || !d.joined(k)) return s;
    for (size_t m = s; m < k; ++m) *pron |= d.is(m, "pronunciation") || d.is(m, "ipa");
    return k + 1;
}

// Tokens s..e are only letters of other scripts and their marks (at least one letter, at most 90 bytes).
bool script_run(const Doc& d, size_t s, size_t e) {
    if (s > e || d[e].end - d[s].start > 90) return false;
    bool letter = false;
    for (size_t k = s; k <= e; ++k) {
        if (k > s && d[k].newline_before) return false;
        if (script_cp(d[k].cp)) letter = true;
        else if (!(d[k].kind == Kind::Punct && script_mark(d[k].cp))) return false;
    }
    return letter;
}

struct Item {
    size_t s, e;          // tokens of the item
    size_t cut_s, cut_e;  // what goes when only a part goes (a pronunciation before a respelling)
    int depth;            // brackets around it
    bool whole = false, part = false;
};

// The items of the line [a, b) between brackets, commas and semicolons, with what of each goes.
std::vector<Item> items(const Doc& d, size_t a, size_t b) {
    std::vector<Ipa> ipa = ipa_spans(d, a, b);
    std::vector<Item> v;
    size_t s = a, q = 0;
    int depth = 0;
    for (size_t k = a; k <= b; ++k) {
        while (q < ipa.size() && ipa[q].e < k) ++q;
        bool inside = q < ipa.size() && ipa[q].s < k && k <= ipa[q].e;
        if (k < b && (inside || !delim(d, k))) continue;
        if (k > s) {  // a sentence period after the item is not part of it
            size_t e = k - 1;
            while (e > s && (p(d, e, '.') || p(d, e, '!') || p(d, e, '?')) && d.joined(e)) --e;
            v.push_back({s, e, 0, 0, depth});
        }
        if (k < b && p(d, k, '(')) ++depth;
        if (k < b && p(d, k, ')') && depth > 0) --depth;
        s = k + 1;
    }
    // a line without a sentence end: a name, its romanization and its pronunciation ("東京都, Tōkyō-to; ...")
    bool open_line = b > a && !(p(d, b - 1, '.') || p(d, b - 1, '!') || p(d, b - 1, '?') || p(d, b - 1, 0x3002));
    q = 0;
    for (Item& it : v) {
        bool pron;
        size_t c = after_label(d, it.s, it.e, &pron);
        while (q < ipa.size() && ipa[q].s < c) ++q;
        if (q < ipa.size() && ipa[q].s == c && ipa[q].e <= it.e) {
            if (ipa[q].e == it.e) it.whole = true;
            else it.part = true, it.cut_s = it.s, it.cut_e = ipa[q].e;
            continue;
        }
        // a name in another script: in brackets, after a label, or before a bracket at the start of a line
        bool place = it.depth > 0 || c > it.s || (it.s == a && it.e + 1 < b && p(d, it.e + 1, '(')) ||
                     (it.s == a && open_line && it.e + 1 < b && comma_or_semicolon(d, it.e + 1)) ||
                     (it.s == a && it.e + 1 >= b && d[it.e].end - d[it.s].start <= 30 && label_line_before(d, a));
        if (place && c <= it.e && script_run(d, c, it.e)) {
            it.whole = true;
            continue;
        }
        // "t 珠穆朗瑪峰", "s 圣母峰": a letter that names the form, then the name
        if (it.depth > 0 && c == it.s && latin_word(d, c) && d.str(c).size() <= 2 && c < it.e && d[c].casing == rv::LOWER &&
            script_run(d, c + 1, it.e)) {
            it.whole = true;
            continue;
        }
        if (pron && c <= it.e) {  // "Tibetan pronunciation: tɕʰòmó lángmá"
            bool any = false;
            for (size_t k = c; k <= it.e; ++k) any |= ipa_cp(d[k].cp) && d[k].kind == Kind::Punct;
            it.whole = any;
        }
    }
    return v;
}

bool line_has_trigger(const Doc& d, size_t a, size_t b) {
    for (size_t k = a; k < b; ++k)
        if ((d[k].kind == Kind::Punct && ipa_cp(d[k].cp)) || script_cp(d[k].cp)) return true;
    return false;
}

// The text may hold IPA letters or another script: a lead byte of U+0240-U+1FFF, U+3000-U+DFFF or above U+FFFF
// (one pass; not punctuation and symbols U+2000-U+2FFF, nor U+E000-U+FFFF).
bool maybe_trigger(const Doc& d) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(d.text.data());
    size_t n = d.text.size(), i = 0;
    while (i < n) {
        if (i + 8 <= n) {  // ASCII eight bytes at a time
            uint64_t w;
            std::memcpy(&w, b + i, 8);
            if (!(w & 0x8080808080808080ull)) {
                i += 8;
                continue;
            }
        }
        unsigned char c = b[i++];
        if (c >= 0xC9 && c != 0xE2 && c != 0xEF && c != 0xEE) return true;
    }
    return false;
}

}  // namespace

bool gloss_sentence(const Doc& d, size_t a, size_t b) {
    if (!line_has_trigger(d, a, b + 1)) return false;
    bool gloss = false, latin = false, other = false;
    for (const Item& it : items(d, a, b + 1)) {
        gloss |= it.whole || it.part;
        bool script = false;
        for (size_t k = it.s; k <= it.e; ++k) {
            script |= script_cp(d[k].cp);
            latin |= latin_word(d, k) && d.str(k).size() > 2;
        }
        if (script && !it.whole) return false;
        other |= script;
    }
    // Latin words must say what a name in another script is ("Nepali: सगरमाथा", or a label on the line before); a
    // text in that script alone stays
    return gloss && (latin || !other || label_line_before(d, a));
}

}  // namespace st

namespace {

using namespace st;

bool same_line(const Doc& d, size_t i) { return i < d.size() && !d[i].newline_before; }
bool sep(const Doc& d, size_t i) { return p(d, i, ',') || p(d, i, ';'); }

// Pronunciations and names in other scripts go: "Everest (/ˈɛvərɪst/) is" -> "Everest is"; "(/ˈkjʊəri/ KURE-ee;
// French: [maʁi kyʁi]; née ...)" -> "(KURE-ee; née ...)"; "Qomolangma (ཇོ་མོ་གླང་མ, lit. ...)" -> "Qomolangma (lit.
// ...)"; the line "Nepali: सगरमाथा" goes. The items that go are taken together with the brackets around them when
// nothing else is inside, else with one separator. Sentences and quotations in another language stay (the
// language rule keeps them).
RV_RULE("structure", glosses) {
    if (!maybe_trigger(doc)) return;
    std::vector<Line> lines = split_lines(doc);
    for (const Line& l : lines) {
        if (!line_has_trigger(doc, l.a, l.b)) continue;
        std::vector<Item> v = items(doc, l.a, l.b);
        for (size_t x = 0; x < v.size(); ++x) {
            if (v[x].part) {  // "(/ˈiːvrɪst/ EEV-rist)" -> "(EEV-rist)"
                size_t c = v[x].cut_s, e = v[x].cut_e;
                if (c > l.a && p(doc, c - 1, '(') && doc.joined(c)) out.add(doc[c].start, doc[e + 1].start, "", 85);
                else {
                    Span s = deletion(doc, c, e);
                    out.add(s.start, s.end, "", 85);
                }
                continue;
            }
            if (!v[x].whole) continue;
            size_t y = x;  // a run of items that go, separated by commas or semicolons only
            while (y + 1 < v.size() && v[y + 1].whole && v[y + 1].s == v[y].e + 2 && sep(doc, v[y].e + 1)) ++y;
            size_t first = v[x].s, last = v[y].e;
            bool line_start = first == l.a, line_end = last + 1 == l.b;
            bool open = !line_start && p(doc, first - 1, '('), close = !line_end && p(doc, last + 1, ')');
            bool sep_before = !line_start && sep(doc, first - 1), sep_after = !line_end && sep(doc, last + 1);
            uint32_t s0 = doc[first].start, e0 = doc[last].end;
            if (open && doc.joined(first - 1) && first - 1 > l.a) {
                x = y;  // "¯\_(ツ)_/¯", "f(x)": brackets glued to the word before are no gloss
                continue;
            }
            // the line holds nothing else but its final period: all of it goes
            bool rest_period = last + 2 == l.b && (p(doc, last + 1, '.') || p(doc, last + 1, '!') || p(doc, last + 1, '?'));
            if (line_start && rest_period) {
                out.add(s0, doc[last + 1].end, "", 85);
                x = y;
                continue;
            }
            if (open && close) {
                Span s = deletion(doc, first - 1, last + 1);
                s0 = s.start, e0 = s.end;
            } else if (sep_after && (open || sep_before || line_start)) {
                e0 = last + 2 < l.b ? doc[last + 2].start : doc[last + 1].end;
            } else if (sep_before && (close || line_end || p(doc, last + 1, '.'))) {
                s0 = doc[first - 2].end;
            } else if (line_start && line_end) {
                // the whole line
            } else {
                Span s = deletion(doc, first, last);
                s0 = s.start, e0 = s.end;
            }
            out.add(s0, e0, "", 85);
            x = y;
        }
    }
}

// Empty brackets and stray commas where page parts (code, links) were left out: "the statement () binds" ->
// "the statement binds", "the operators (, , ,) and" -> "the operators and", "The, , and" -> "The, and",
// "for example, ." -> "for example."; an omission mark "[...]" is a pause between words and goes elsewhere.
RV_RULE("structure", debris) {
    std::string_view t = doc.text;
    auto bracket = [&](size_t k) {  // "()", "( ", "(..", "(…", and the same after '['
        if (k + 1 >= t.size()) return false;
        char c = t[k + 1];
        return c == ')' || c == ']' || c == ' ' || c == ',' || (c == '.' && k + 2 < t.size() && t[k + 2] == '.') ||
               t.substr(k + 1, 3) == "\xE2\x80\xA6";
    };
    auto comma = [&](size_t k) {  // ", ,", ", .", ",)"
        return k + 1 < t.size() && (t[k + 1] == ')' || (t[k + 1] == ' ' && k + 2 < t.size() && (t[k + 2] == ',' || t[k + 2] == '.')));
    };
    if (!(each_byte(t, '(', bracket) || each_byte(t, '[', bracket) || each_byte(t, ',', comma))) return;
    size_t n = doc.size();
    for (size_t i = 0; i < n; ++i) {
        if ((p(doc, i, '(') || p(doc, i, '[')) && (i == 0 || doc[i].space_before)) {
            uint32_t close = doc[i].cp == '(' ? ')' : ']';
            size_t e = i + 1;
            int dots = 0;
            while (e < n && same_line(doc, e) &&
                   (p(doc, e, ',') || p(doc, e, ';') || p(doc, e, '.') || p(doc, e, 0x2026))) {
                dots += p(doc, e, '.') ? 1 : p(doc, e, 0x2026) ? 3 : 0;
                ++e;
            }
            // "[ ]" is a check box; "[]" and "( )" are empty; "[...]" marks words left out of a quotation
            bool box = close == ']' && e == i + 1 && e < n && doc[e].space_before;
            bool omission = dots >= 2;
            if ((dots > 0 && !omission) || e >= n || !same_line(doc, e) || !p(doc, e, close) || box ||
                !(e + 1 >= n || !doc.joined(e + 1) || attaches_left(doc, e + 1))) {
                continue;
            }
            if (omission && i > 0 && same_line(doc, i) && doc.word(i - 1) && e + 1 < n && same_line(doc, e + 1) &&
                doc.word(e + 1))
                out.add(doc[i - 1].end, doc[e].end, ",", 60);  // a pause where words were left out
            else {
                Span s = deletion(doc, i, e);
                out.add(s.start, s.end, "", 60);
            }
            i = e;
            continue;
        }
        if (!p(doc, i, ',') || i == 0 || !same_line(doc, i + 1) || !doc[i + 1].space_before) continue;
        size_t e = i;
        while (p(doc, e + 1, ',') && same_line(doc, e + 1) && doc[e + 1].space_before) ++e;
        if (e > i) {  // ", , ," -> ","
            out.add(doc[i].end, doc[e].end, "", 60);
            i = e;
        } else if (p(doc, i + 1, '.') && !p(doc, i + 2, '.') && (i + 2 >= n || !doc.joined(i + 2))) {
            out.add(doc[i - 1].end, doc[i + 1].start, "", 60);  // ", ." -> "."
        }
    }
}

}  // namespace
