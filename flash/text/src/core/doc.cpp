#include "core/doc.h"

#include <algorithm>

#include "core/utf8.h"

namespace rv {

namespace {

bool apostrophe(uint32_t cp) { return cp == '\'' || cp == 0x2019; }

bool quote_or_bracket(uint32_t cp) {
    switch (cp) {
        case '"': case '\'': case '(': case ')': case '[': case ']': case '*': case '_':
        case 0x201C: case 0x201D: case 0x2018: case 0x2019: case 0xAB: case 0xBB:
            return true;
        default:
            return false;
    }
}

}  // namespace

Doc::Doc(std::string_view t) : text(t), lower(t) {
    for (char& c : lower)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    tokens.reserve(t.size() / 3 + 1);
    bool space = true, newline = false;
    size_t i = 0;
    while (i < t.size()) {
        size_t start = i;
        uint32_t cp = decode(t, i);
        if (is_space(cp)) {
            space = true;
            if (cp == '\n') newline = true;
            continue;
        }
        Token tok{static_cast<uint32_t>(start), 0, cp, Kind::Punct, 0, space, newline};
        if (cp == 0xFDD0) {
            i = formula_end(t, start);
            tok.kind = Kind::Word;
        } else if (cp >= '0' && cp <= '9') {
            while (i < t.size() && is_ascii_digit(t[i])) ++i;
            tok.kind = Kind::Number;
        } else if (is_letter(cp)) {
            tok.kind = Kind::Word;
            bool first_upper = is_upper(cp), lower_after = false, upper_after = false;
            while (i < t.size()) {
                size_t k = i;
                uint32_t c = decode(t, k);
                if (is_letter(c)) {
                    upper_after |= is_upper(c);
                    lower_after |= is_lower(c);
                    i = k;
                    continue;
                }
                if (apostrophe(c) && k < t.size()) {
                    size_t m = k;
                    if (is_letter(decode(t, m))) {
                        i = k;
                        continue;
                    }
                }
                break;
            }
            if (first_upper && !lower_after) tok.casing = UPPER;
            else if (first_upper && !upper_after) tok.casing = TITLE;
            else if (!first_upper && !upper_after) tok.casing = LOWER;
            else tok.casing = MIXED;
        }
        tok.end = static_cast<uint32_t>(i);
        tokens.push_back(tok);
        space = newline = false;
    }
}

size_t formula_end(std::string_view t, size_t b) {
    for (size_t k = b + 3; k < t.size(); ++k) {
        if (t[k] == '\n') return k;
        if (t[k] == '\xEF' && k + 2 < t.size() && t[k + 1] == '\xB7') {
            if (t[k + 2] == '\x91') return k + 3;
            if (t[k + 2] == '\x90') return k;
        }
    }
    return t.size();
}

std::string_view Doc::str(size_t i) const { return slice(tokens[i].start, tokens[i].end); }

std::string_view Doc::low(size_t i) const {
    return std::string_view(lower).substr(tokens[i].start, tokens[i].end - tokens[i].start);
}

size_t Doc::prev_word(size_t i, int n) const {
    for (size_t j = i; j-- > 0;)
        if (tokens[j].kind != Kind::Punct && --n == 0) return j;
    return npos;
}

size_t Doc::next_word(size_t i, int n) const {
    for (size_t j = i + 1; j < size(); ++j)
        if (tokens[j].kind != Kind::Punct && --n == 0) return j;
    return npos;
}

bool Doc::sentence_start(size_t i) const {
    if (i >= size()) return false;
    if (tokens[i].newline_before) return true;
    size_t j = i;
    while (j > 0 && tokens[j - 1].kind == Kind::Punct && quote_or_bracket(tokens[j - 1].cp)) {
        --j;
        if (tokens[j].newline_before) return true;
    }
    if (j == 0) return true;
    const Token& p = tokens[j - 1];
    return p.kind == Kind::Punct && (p.cp == '.' || p.cp == '!' || p.cp == '?' || p.cp == 0x2026);
}

size_t Doc::token_at(uint32_t b) const {
    auto it = std::lower_bound(tokens.begin(), tokens.end(), b,
                               [](const Token& t, uint32_t v) { return t.start < v; });
    return static_cast<size_t>(it - tokens.begin());
}

NumberSpan parse_number(const Doc& d, size_t i) {
    NumberSpan n;
    if (!d.number(i)) return n;
    n.ok = true;
    n.first = n.last = i;
    n.whole = std::string(d.str(i));
    // thousands groups: a first group of 1-3 digits, then ",ddd" joined
    if (n.whole.size() <= 3) {
        size_t j = i;
        while (d.punct(j + 1, ',') && d.joined(j + 1) && d.number(j + 2) && d.joined(j + 2) &&
               d.str(j + 2).size() == 3) {
            n.whole += d.str(j + 2);
            n.grouped = true;
            j += 2;
        }
        n.last = j;
    }
    if (d.punct(n.last + 1, '.') && d.joined(n.last + 1) && d.number(n.last + 2) && d.joined(n.last + 2)) {
        n.fraction = std::string(d.str(n.last + 2));
        n.last += 2;
    }
    return n;
}

}  // namespace rv
