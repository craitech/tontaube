#include "core/engine.h"

#include <algorithm>
#include <cstdio>
#include <numeric>

#include "rv/verbalizer.h"

namespace rv {

namespace {

bool alnum(char c) { return is_ascii_alnum(c); }

bool closing(char c) { return c == ' ' || c == '\t' || c == '"' || c == '\'' || c == ')' || c == ']' || c == '*' || c == '_'; }

// The output so far ends a sentence (or is empty): . ! ? … or a line break, then only spaces, quotes, brackets.
bool ends_sentence(const std::string& out) {
    size_t i = out.size();
    while (i > 0) {
        char c = out[i - 1];
        if (closing(c) || c == '(' || c == '[') {
            --i;
            continue;
        }
        if (i >= 3 && static_cast<unsigned char>(out[i - 3]) == 0xE2 && static_cast<unsigned char>(out[i - 2]) == 0x80) {
            unsigned char k = static_cast<unsigned char>(out[i - 1]);
            if (k == 0x9C || k == 0x9D || k == 0x98 || k == 0x99) {  // “ ” ‘ ’
                i -= 3;
                continue;
            }
            if (k == 0xA6) return true;  // …
        }
        break;
    }
    if (i == 0) return true;
    char c = out[i - 1];
    return c == '.' || c == '!' || c == '?' || c == '\n';
}

// The span [start, end) is a fragment: alone on its line, or the start of an input of at most two words that does
// not end with closing punctuation. Labels keep fragments in lower case ("0.600" -> "zero point six zero zero").
bool fragment(std::string_view t, size_t start, size_t end) {
    size_t a = start, b = end;
    while (a > 0 && (t[a - 1] == ' ' || t[a - 1] == '\t')) --a;
    while (b < t.size() && (t[b] == ' ' || t[b] == '\t' || t[b] == '\r')) ++b;
    if ((a == 0 || t[a - 1] == '\n') && (b == t.size() || t[b] == '\n')) return true;
    for (size_t k = 0; k < start; ++k)
        if (!closing(t[k]) && t[k] != '(' && t[k] != '[' && t[k] != '\n') return false;
    size_t e = t.size();
    while (e > 0 && (t[e - 1] == '\n' || t[e - 1] == '\r' || t[e - 1] == ' ')) --e;
    // a one-line input that starts with a digit counts as a fragment however long it is
    bool digit_line = t[start] >= '0' && t[start] <= '9' && t.substr(start, e - start).find('\n') == std::string_view::npos;
    int words = 0;
    bool in_word = false;
    for (char c : t) {
        bool space = c == ' ' || c == '\t' || c == '\n' || c == '\r';
        if (!space && !in_word && ++words > 2 && !digit_line) return false;
        in_word = !space;
    }
    e = t.size();
    while (e > 0 && closing(t[e - 1])) --e;
    while (e > 0 && (t[e - 1] == '\n' || t[e - 1] == '\r')) --e;
    char last = e ? t[e - 1] : 0;
    bool ellipsis = e >= 3 && t.substr(e - 3, 3) == "\xE2\x80\xA6";
    return !(last == '.' || last == '!' || last == '?' || last == ':' || ellipsis);
}

struct Entity {
    const char* name;  // after '&', up to and including ';', lower case
    const char* with;  // UTF-8 replacement ("" removes)
};
const Entity SPACING[] = {{"nbsp;", "\xC2\xA0"}, {"#160;", "\xC2\xA0"}, {"#xa0;", "\xC2\xA0"},
                          {"thinsp;", "\xE2\x80\x89"}, {"#8201;", "\xE2\x80\x89"}, {"ensp;", " "},
                          {"emsp;", " "}, {"shy;", ""}, {"#173;", ""}, {"#xad;", ""}, {"zwj;", ""},
                          {"zwnj;", ""}, {"#8203;", ""}, {"#x200b;", ""}, {"#8204;", ""}, {"#8205;", ""},
                          {"#65279;", ""}, {"#xfeff;", ""}};

// Text with Chinese, Japanese or Korean characters keeps its full-width forms (labels keep "，" as written).
bool fold_full_width(std::string_view t) {
    if (t.find("\xEF\xBC") == std::string_view::npos && t.find("\xEF\xBD") == std::string_view::npos &&
        t.find("\xE3\x80\x80") == std::string_view::npos)
        return false;
    for (size_t i = 0; i < t.size();) {
        uint32_t cp = decode(t, i);
        if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) ||
            (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF))
            return false;
    }
    return true;
}

}  // namespace

std::string prepare(std::string_view t, std::vector<uint32_t>* src) {
    std::string out;
    out.reserve(t.size());
    if (src) src->clear(), src->reserve(t.size() + 1);
    // the bytes appended since o0 came from input byte i: copied byte for byte, or made from it
    auto mark = [&](size_t o0, size_t i, bool copied) {
        if (src)
            for (size_t k = o0; k < out.size(); ++k) src->push_back(static_cast<uint32_t>(copied ? i + (k - o0) : i));
    };
    bool full_width = fold_full_width(t);
    size_t i = 0;
    while (i < t.size()) {
        unsigned char c = static_cast<unsigned char>(t[i]);
        size_t o0 = out.size();
        if (c == '&') {
            bool done = false;
            for (const Entity& e : SPACING) {
                size_t n = std::char_traits<char>::length(e.name);
                if (i + 1 + n > t.size()) continue;
                bool same = true;
                for (size_t k = 0; k < n && same; ++k) {
                    char x = t[i + 1 + k];
                    same = (x >= 'A' && x <= 'Z' ? x + 32 : x) == e.name[k];
                }
                if (same) {
                    out += e.with;
                    mark(o0, i, false);
                    i += 1 + n;
                    done = true;
                    break;
                }
            }
            if (done) continue;
        }
        if (c < 0x80) {
            out += static_cast<char>(c);
            mark(o0, i, true);
            ++i;
            continue;
        }
        size_t j = i;
        uint32_t cp = decode(t, j);
        bool copied = false;
        if (cp == 0x2010 || cp == 0x2011) out += '-';  // hyphen, no-break hyphen
        else if (uint32_t m = plain_math_alnum(cp)) append_utf8(out, m);  // 𝑥 𝐳 𝟐: plain letters and digits
        else if (full_width && cp >= 0xFF01 && cp <= 0xFF5E) out += static_cast<char>(cp - 0xFEE0);
        else if (full_width && cp == 0x3000) out += ' ';
        else if (!is_invisible(cp)) out.append(t.substr(i, j - i)), copied = true;
        mark(o0, i, copied);
        i = j;
    }
    if (src) src->push_back(static_cast<uint32_t>(t.size()));
    return out;
}

std::vector<const RuleDef*> select_modules(std::string_view csv) {
    const auto& all = rules();
    if (csv.empty()) return all;
    std::vector<std::string> wanted{"core"};
    size_t p = 0;
    while (p <= csv.size()) {
        size_t q = csv.find(',', p);
        if (q == std::string_view::npos) q = csv.size();
        if (q > p) wanted.emplace_back(csv.substr(p, q - p));
        p = q + 1;
    }
    std::vector<const RuleDef*> out;
    for (const RuleDef* r : all)
        if (std::find(wanted.begin(), wanted.end(), r->module) != wanted.end()) out.push_back(r);
    return out;
}

void resolve(Run& r) {
    auto& c = r.candidates;
    std::vector<int> idx(c.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const Candidate &x = c[a], &y = c[b];
        bool xi = x.start == x.end, yi = y.start == y.end;
        if (xi != yi) return yi;  // replacements first, insertions after
        if (x.score != y.score) return x.score > y.score;
        uint32_t lx = x.end - x.start, ly = y.end - y.start;
        if (lx != ly) return lx > ly;
        if (x.start != y.start) return x.start < y.start;
        return x.rule->order < y.rule->order;
    });
    r.beaten_by.assign(c.size(), -1);
    std::vector<int> kept;  // winning replacements sorted by start; they never overlap
    std::vector<int> inserted;
    for (int k : idx) {
        auto it = std::lower_bound(kept.begin(), kept.end(), c[k].start,
                                   [&](int w, uint32_t s) { return c[w].start < s; });
        int rival = -1;
        if (c[k].start == c[k].end) {  // an insertion: not strictly inside a replacement, one per position
            if (it != kept.begin() && c[*(it - 1)].end > c[k].start) rival = *(it - 1);
            for (int w : inserted)
                if (c[w].start == c[k].start) rival = w;
            if (rival < 0) inserted.push_back(k);
            else r.beaten_by[k] = rival;
            continue;
        }
        if (it != kept.begin() && c[*(it - 1)].end > c[k].start) rival = *(it - 1);
        else if (it != kept.end() && c[*it].start < c[k].end) rival = *it;
        if (rival < 0) kept.insert(it, k);
        else r.beaten_by[k] = rival;
    }
}

namespace {

bool blank(char c) { return c == '\n' || c == ' ' || c == '\t' || c == '\r'; }

// More than one blank line in a row (also where rules removed whole lines) becomes one blank line: the first two
// line breaks of the run stay, the rest of the run goes. Line breaks left at the start or the end by removed lines
// go when the text itself does not start or end with one. The pieces from index p0 on are cut to match.
void squeeze_blank_lines(std::string_view text, std::string& out, std::vector<Piece>* map, size_t p0) {
    std::vector<uint8_t> drop;
    if (!out.empty() && blank(out[0]) && !text.empty() && !blank(text[0])) {
        size_t j = 0;
        while (j < out.size() && blank(out[j])) ++j;
        if (j < out.size() && out.find('\n') < j) drop.assign(out.size(), 0), std::fill(drop.begin(), drop.begin() + j, 1);
    }
    if (!out.empty() && blank(out.back()) && !text.empty() && !blank(text.back())) {
        size_t j = out.size();
        while (j > 0 && blank(out[j - 1])) --j;
        if (j > 0 && out.find('\n', j) != std::string::npos) {
            if (drop.empty()) drop.assign(out.size(), 0);
            std::fill(drop.begin() + j, drop.end(), 1);
        }
    }
    for (size_t i = out.find('\n'); i != std::string::npos; i = out.find('\n', i)) {
        size_t j = i, nl = 0;
        while (j < out.size() && blank(out[j])) nl += out[j++] == '\n';
        if (nl >= 3) {
            if (drop.empty()) drop.assign(out.size(), 0);
            for (size_t k = i, kept = 0; k < j; ++k) {
                if (out[k] == '\n' && kept < 2 && !drop[k]) ++kept;
                else drop[k] = 1;
            }
        }
        i = j;
    }
    if (drop.empty()) return;
    std::vector<uint32_t> at(out.size() + 1);  // new offset of each old offset
    std::string s;
    s.reserve(out.size());
    for (size_t k = 0; k < out.size(); ++k) {
        at[k] = static_cast<uint32_t>(s.size());
        if (!drop[k]) s += out[k];
    }
    at[out.size()] = static_cast<uint32_t>(s.size());
    if (map) {
        std::vector<Piece> pieces(map->begin(), map->begin() + p0);
        for (size_t q = p0; q < map->size(); ++q) {
            const Piece& p = (*map)[q];
            if (!p.copied) {
                if (at[p.out_end] > at[p.out_begin]) pieces.push_back({at[p.out_begin], at[p.out_end], p.in_begin, p.in_end, false});
                continue;
            }
            for (uint32_t k = p.out_begin; k < p.out_end;) {  // runs of kept bytes, still byte for byte
                if (drop[k]) { ++k; continue; }
                uint32_t e = k;
                while (e < p.out_end && !drop[e]) ++e;
                uint32_t ib = p.in_begin + (k - p.out_begin);
                pieces.push_back({at[k], at[e], ib, ib + (e - k), true});
                k = e;
            }
        }
        map->swap(pieces);
    }
    out.swap(s);
}

}  // namespace

std::string assemble(std::string_view text, const Run& r, std::vector<Piece>* map) {
    std::vector<int> win;
    for (size_t k = 0; k < r.candidates.size(); ++k)
        if (r.beaten_by[k] < 0) win.push_back(static_cast<int>(k));
    std::sort(win.begin(), win.end(), [&](int a, int b) {
        const Candidate &x = r.candidates[a], &y = r.candidates[b];
        return x.start != y.start ? x.start < y.start : x.end < y.end;  // an insertion before a span at its spot
    });
    std::string out;
    out.reserve(text.size() + text.size() / 2 + 16);
    const size_t p0 = map ? map->size() : 0;
    // the output appended since o0 came from text [b, e)
    auto piece = [&](size_t o0, size_t b, size_t e, bool copied = false) {
        if (map && out.size() > o0)
            map->push_back(Piece{static_cast<uint32_t>(o0), static_cast<uint32_t>(out.size()), static_cast<uint32_t>(b),
                                 static_cast<uint32_t>(e), copied});
    };
    auto copy = [&](size_t b, size_t e) {
        size_t o0 = out.size();
        out.append(text.substr(b, e - b));
        piece(o0, b, e, true);
    };
    size_t pos = 0;
    for (int k : win) {
        const Candidate& c = r.candidates[k];
        copy(pos, c.start);
        pos = c.end;
        std::string_view original = text.substr(c.start, c.end - c.start);
        if (c.spoken.empty()) {
            if (!out.empty() && out.back() == ' ' && pos < text.size() && text[pos] == ' ') ++pos;
            continue;
        }
        if (c.spoken == original) {
            copy(c.start, c.end);
            continue;
        }
        bool upper_original = !original.empty() && original[0] >= 'A' && original[0] <= 'Z';
        bool cap = !c.keep_case && ends_sentence(out) && (upper_original || !fragment(text, c.start, c.end));
        size_t o0 = out.size();
        if (!out.empty() && alnum(out.back()) && alnum(c.spoken.front())) out += ' ';
        piece(o0, c.start, c.start);
        size_t at = out.size();
        out += c.spoken;
        if (cap && out[at] >= 'a' && out[at] <= 'z') out[at] = static_cast<char>(out[at] - 32);
        piece(at, c.start, c.end);
        o0 = out.size();
        if (pos < text.size() && alnum(text[pos]) && alnum(c.spoken.back())) out += ' ';
        piece(o0, c.end, c.end);
    }
    copy(pos, text.size());
    squeeze_blank_lines(text, out, map, p0);
    return out;
}

namespace {

// Pieces over the prepared text -> pieces over the input (src from prepare). A copied piece is cut where prepare
// changed bytes, so every piece with equal lengths maps byte for byte.
std::vector<Piece> to_input(const std::vector<Piece>& pieces, const std::vector<uint32_t>& src) {
    std::vector<Piece> out;
    out.reserve(pieces.size() + 8);
    auto add = [&](uint32_t ob, uint32_t oe, uint32_t ib, uint32_t ie, bool copied) {
        if (copied && ie - ib != oe - ob) copied = false;  // a character prepare() changed
        if (copied && !out.empty()) {
            Piece& l = out.back();
            if (l.copied && l.out_end == ob && l.in_end == ib) {
                l.out_end = oe, l.in_end = ie;
                return;
            }
        }
        out.push_back(Piece{ob, oe, ib, ie, copied});
    };
    for (const Piece& p : pieces) {
        if (!p.copied) {  // a reading: its whole span
            add(p.out_begin, p.out_end, src[p.in_begin], src[p.in_end], false);
            continue;
        }
        uint32_t k0 = p.in_begin;
        for (uint32_t k = p.in_begin; k < p.in_end; ++k) {
            if (src[k + 1] == src[k] + 1) continue;  // byte for byte up to k
            uint32_t ob = p.out_begin + (k0 - p.in_begin);
            if (k > k0) add(ob, ob + (k - k0), src[k0], src[k0] + (k - k0), true);
            add(ob + (k - k0), ob + (k - k0) + 1, src[k], src[k + 1], true);
            k0 = k + 1;
        }
        if (k0 < p.in_end) {
            uint32_t ob = p.out_begin + (k0 - p.in_begin);
            add(ob, p.out_end, src[k0], src[k0] + (p.in_end - k0), true);
        }
    }
    return out;
}

}  // namespace

Run run(std::string_view input, const std::vector<const RuleDef*>& active, std::vector<Piece>* map) {
    Run r;
    std::vector<uint32_t> src;
    r.text = prepare(input, map ? &src : nullptr);
    if (r.text.find_first_not_of(" \t\r\n") == std::string::npos) {  // nothing visible: the input stays as it is
        r.text = std::string(input);
        r.spoken = r.text;
        if (map) {
            map->clear();
            uint32_t n = static_cast<uint32_t>(input.size());
            if (n) map->push_back(Piece{0, n, 0, n, true});
        }
        return r;
    }
    Doc doc(r.text);
    r.candidates.reserve(doc.size() * 2 + 4);
    Out out(doc, r.candidates);
    for (const RuleDef* rule : active) {
        out.rule = rule;
        rule->fn(doc, out);
    }
    resolve(r);
    if (!map) {
        r.spoken = assemble(r.text, r);
        return r;
    }
    std::vector<Piece> pieces;
    r.spoken = assemble(r.text, r, &pieces);
    *map = to_input(pieces, src);
    return r;
}

std::string describe(const Run& r) {
    std::string_view text = r.text;
    std::vector<int> idx(r.candidates.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const Candidate &x = r.candidates[a], &y = r.candidates[b];
        return x.start != y.start ? x.start < y.start : x.score > y.score;
    });
    std::string s;
    char buf[64];
    for (int k : idx) {
        const Candidate& c = r.candidates[k];
        std::snprintf(buf, sizeof buf, "%c %3d ", r.beaten_by[k] < 0 ? '+' : '-', c.score);
        s += buf;
        std::string name = c.rule->name;
        name.resize(std::max<size_t>(name.size(), 34), ' ');
        s += name + " '" + std::string(text.substr(c.start, c.end - c.start)) + "' -> '" + c.spoken + "'";
        if (r.beaten_by[k] >= 0) {
            const Candidate& w = r.candidates[r.beaten_by[k]];
            std::snprintf(buf, sizeof buf, " (%d)", w.score);
            s += "   lost to " + w.rule->name + buf;
        }
        s += '\n';
    }
    return s;
}

std::string verbalize(std::string_view text) { return run(text, rules()).spoken; }

std::string verbalize(std::string_view text, std::vector<Piece>& map) { return run(text, rules(), &map).spoken; }

}  // namespace rv
