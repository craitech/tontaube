// Rules of the module "units_symbols": math and comparison operators, powers and subscripts, Greek letters,
// complexity notation, stars, arrows, and the signs & @ © ® ™.
#include "rules/units_symbols/common.h"

namespace {

using namespace us;

bool spaced_both(const Doc& d, size_t i) {
    return i > 0 && i + 1 < d.size() && d[i].space_before && d[i + 1].space_before && !d[i].newline_before &&
           !d[i + 1].newline_before;
}

// A single letter used as a variable: 'x', 'n', 'R'.
bool variable(const Doc& d, size_t i) {
    return d.word(i) && rv::is_letter(d[i].cp) && d.str(i).size() <= 2 && !stuck_before(d, i) &&
           (d.str(i).size() == 1 || (d[i].cp >= 0x80 && d.str(i).size() == 2));
}

// An operand ends at token i / starts at token i: a number, a variable, a closing or opening bracket.
bool operand_end(const Doc& d, size_t i) {
    if (i >= d.size()) return false;
    if (d.number(i) || variable(d, i)) return true;
    return d.punct(i, ')') || super_tok(d, i) || sub_tok(d, i) || d.punct(i, '%');
}
bool operand_start(const Doc& d, size_t i) {
    if (i >= d.size()) return false;
    if (d.number(i) || variable(d, i)) return true;
    if (d.punct(i, '(') || d.punct(i, '$') || d.punct(i, 0x221A) || d.punct(i, 0x221B)) return true;
    if ((d.punct(i, '-') || d.punct(i, 0x2212) || d.punct(i, '.')) && d.number(i + 1) && d.joined(i + 1)) return true;
    return false;
}

const char* const GREEK_LOWER[] = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                                   "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
                                   "rho", "final sigma", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"};

// Name of a Greek letter code point, "" if none. Capitals: 'Psi', but 'delta' for Δ as the usual difference sign.
std::string greek_name(uint32_t cp) {
    if (cp >= 0x3B1 && cp <= 0x3C9) return cp == 0x3C2 ? "sigma" : GREEK_LOWER[cp - 0x3B1];
    if (cp >= 0x391 && cp <= 0x3A9 && cp != 0x3A2) {
        std::string n = GREEK_LOWER[cp - 0x391];
        if (cp == 0x394) return "delta";
        return rv::capital(n);
    }
    if (cp == 0xB5) return "mu";
    if (cp == 0x3D5) return "phi";
    if (cp == 0x3F5) return "epsilon";
    if (cp == 0x3D1) return "theta";
    if (cp == 0x3F1) return "rho";
    if (cp == 0x3D6) return "pi";
    return "";
}

// A Greek letter written together with Latin letters: 'πr' -> 'pi r', 'Δt' -> 'delta t', 'ΔCO' -> 'delta CO'.
// "" if token i is not one.
std::string greek_pair_words(const Doc& d, size_t i) {
    if (!d.word(i)) return "";
    std::string_view w = d.str(i);
    size_t p = 0;
    uint32_t a = rv::decode(w, p);
    std::string ga = greek_name(a);
    if (!ga.empty() && p < w.size()) {
        std::string_view rest = w.substr(p);
        bool latin = rest.size() <= (a == 0x394 ? 3u : 1u);
        for (char c : rest) latin &= (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        if (latin) return ga + " " + std::string(rest);
    }
    if (w.size() >= 3 && ((w[0] >= 'a' && w[0] <= 'z') || (w[0] >= 'A' && w[0] <= 'Z'))) {
        size_t q = 1;
        uint32_t b = rv::decode(w, q);
        std::string gb = greek_name(b);
        if (!gb.empty() && q == w.size()) return std::string(1, w[0]) + " " + gb;
    }
    return "";
}
bool greek_pair(const Doc& d, size_t i) { return !greek_pair_words(d, i).empty(); }

bool greek_letter(const Doc& d, size_t i) {
    if (!d.word(i) || (d[i].cp < 0x391 && d[i].cp != 0xB5) || d[i].cp > 0x3F5) return false;
    size_t p = 0;
    rv::decode(d.str(i), p);
    return p == d.str(i).size() && !greek_name(d[i].cp).empty();
}

// Two lower-case letters in a formula after '=' are said as letters: 'E = mc²' -> 'E equals M C squared'.
// "" if token i is not such a word.
std::string formula_letters(const Doc& d, size_t i) {
    if (!d.word(i) || d.str(i).size() != 2 || !(d[i].casing & rv::LOWER) || stuck_before(d, i)) return "";
    if (!(super_tok(d, i + 1) || d.punct(i + 1, '^')) || !d.joined(i + 1)) return "";
    if (i == 0 || !(d.punct(i - 1, '=') || d.is(i - 1, "equals"))) return "";
    std::string_view w = d.str(i);
    std::string s;
    s += static_cast<char>(w[0] - 32);
    s += ' ';
    s += static_cast<char>(w[1] - 32);
    return s;
}

const rv::WordSet VERBISH{"is", "was", "are", "were", "be", "been", "being", "stays", "stayed", "stay", "remains",
                          "remained", "remain", "kept", "keeps", "keep", "held", "holds", "hold", "consistently",
                          "always", "clearly", "obviously", "still", "reached", "reach", "reaches", "hit", "hits",
                          "exceeds", "exceeded", "exceed", "fell", "falls", "rose", "rises", "grew", "grows", "sits",
                          "sat", "ran", "runs", "need", "needs", "needed", "with", "at", "of", "in", "for", "to",
                          "from", "under", "over", "by", "and", "or", "but", "only", "about", "around", "just",
                          "showed", "shows", "show", "had", "has", "have", "got", "gets", "get", "flagged", "took",
                          "takes", "measured", "averaged", "averages", "scored", "logged", "returned", "dropped",
                          "drops", "within", "below", "above", "near", "when", "if", "while", "where", "whether",
                          "every", "each", "any", "all", "means", "mean", "indicates", "indicate", "should", "must",
                          "will", "would", "can", "could", "may", "might", "typically", "usually", "often", "now",
                          "also", "even", "yet", "already", "nearly", "roughly", "approximately", "becomes",
                          "became", "hovered", "hovers", "peaked", "peaks", "targets", "target", "requires",
                          "require", "required", "allows", "allow", "allowed", "used", "uses", "use", "set", "sets",
                          "anything", "something", "everything", "nothing"};

// Comparison and math signs: 'x > 9', '<2%', '≤ 5', '≈ 40', '≠ 0', '±3%', '∞', '√2', '2 + 2 = 4', '7 - 2'.
void operators_at(const Doc& doc, Out& out, size_t i) {
    uint32_t cp = doc[i].cp;
    bool at_line_start = i == 0 || doc[i].newline_before;
    switch (cp) {
        case '<':
        case '>': {
            bool less = cp == '<';
            // never quote markers, arrows, tags, '<<', '->', '=>', '<=', '>='
            if (at_line_start) break;
            if (i > 0 && doc.joined(i) && (doc.punct(i - 1, '-') || doc.punct(i - 1, '=') || doc.punct(i - 1, '<') ||
                                           doc.punct(i - 1, '>') || doc.punct(i - 1, '|')))
                break;
            if (i + 1 < doc.size() && doc.joined(i + 1) &&
                (doc.punct(i + 1, '<') || doc.punct(i + 1, '>') || doc.punct(i + 1, '-') || doc.punct(i + 1, '|')))
                break;
            // a quantity starts at token k: '5', '.05', '$30', '€30', '~40', '-3'
            auto quantity = [&](size_t k) {
                if (doc.number(k)) return true;
                return k + 1 < doc.size() && doc.joined(k + 1) && doc.number(k + 1) && punct(doc, k) &&
                       (currency_sign(doc[k].cp) || doc[k].cp == '.' || doc[k].cp == '~' || doc[k].cp == '-' ||
                        doc[k].cp == 0x2212);
            };
            if (doc.punct(i + 1, '=') && doc.joined(i + 1)) {
                // '>=' and '<=', spaced or written tight before a number: 'buffer>=4', 'is >=0.25'
                bool tight = i + 2 < doc.size() && doc.joined(i + 2) && quantity(i + 2) &&
                             (!doc.joined(i) || doc.word(i - 1) || doc.number(i - 1));
                if (!tight && !spaced_both(doc, i) &&
                    !(doc[i].space_before && i + 2 < doc.size() && doc[i + 2].space_before))
                    break;
                out.tokens(i, i + 1, less ? "less than or equal to" : "greater than or equal to", 45);
                break;
            }
            // '<2%', '>90%', '<200 ms', '<€30': a sign joined to a number after a space or an opening bracket
            if (doc.joined(i + 1) && (!doc.joined(i) || doc.punct(i - 1, '(') || doc.punct(i - 1, '[')) && quantity(i + 1)) {
                out.tokens(i, i, less ? "less than" : "more than", 45);
                break;
            }
            // 'p<.001', 'x>5': a sign written tight after a variable
            if (doc.joined(i) && variable(doc, i - 1) && !stuck_before(doc, i - 1) && quantity(i + 1) &&
                doc.joined(i + 1) == doc.joined(i)) {
                bool p_value = doc.is(i - 1, "p");
                out.tokens(i, i, less ? (p_value ? "less than" : "is less than")
                                      : (p_value ? "greater than" : "is greater than"), 45);
                break;
            }
            if (!spaced_both(doc, i)) break;
            size_t r = i + 1;
            // 'Settings > Privacy', 'Dosing schedule > Week 1 > Week 2' are paths (structure), not comparisons
            if (!less) {
                bool title_right = doc.word(r) && (doc[r].casing & (rv::TITLE | rv::UPPER)) && !variable(doc, r);
                bool title_left = doc.word(i - 1) && (doc[i - 1].casing & (rv::TITLE | rv::UPPER)) &&
                                  !variable(doc, i - 1);
                int arrows = 0;  // other spaced '>' on the line, within 30 tokens
                for (size_t k = i; k-- > 0 && k + 30 > i && !doc[k + 1].newline_before;)
                    arrows += doc.punct(k, '>') && doc[k].space_before;
                for (size_t k = i + 1; k < doc.size() && k < i + 30 && !doc[k].newline_before; ++k)
                    arrows += doc.punct(k, '>') && doc[k].space_before;
                if (title_right && (title_left || arrows >= 1)) break;
            }
            if (!(operand_start(doc, r) || doc.word(r) || doc.punct(r, '$') || doc.punct(r, 0x20AC) ||
                  doc.punct(r, 0xA3)))
                break;
            size_t l = i - 1;
            std::string_view lw = doc.word(l) ? doc.low(l) : std::string_view();
            static const rv::WordSet BE{"is", "was", "are", "were", "be", "been"};
            std::string words;
            if (greek_letter(doc, l)) words = less ? "less than" : "more than";
            else if (lw == "p" && !stuck_before(doc, l)) words = less ? "less than" : "greater than";
            else if (lw == "value" && l >= 2 && doc.punct(l - 1, '-') && doc.is(l - 2, "p"))  // 'p-value < 0.05'
                words = less ? "less than" : "greater than";
            else if (VERBISH.contains(lw) || (lw.size() > 4 && lw.substr(lw.size() - 2) == "ly"))
                words = less ? "less than" : BE.contains(lw) ? "greater than" : "more than";
            else if (doc.word(l) || operand_end(doc, l) || doc.punct(l, '\'') || doc.punct(l, '"') ||
                     doc.punct(l, 0x2019) || doc.punct(l, 0x201D))
                words = less ? "is less than" : "is greater than";
            else
                break;
            out.tokens(i, i, words, 45);
            break;
        }
        case 0x2264:  // ≤
        case 0x2265: {  // ≥
            bool le = cp == 0x2264;
            size_t p = doc.prev_word(i);
            bool after_at = p != Doc::npos && p + 1 == i && doc.is(p, "at");
            if (after_at) {
                out.tokens(p, i, le ? "at or below" : "at or above", 46);
            } else {
                out.tokens(i, i, le ? "at most" : "at least", 45);
            }
            break;
        }
        case 0x2248:  // ≈
        case 0x223C:  // ∼
            out.tokens(i, i, "approximately", 45);
            break;
        case 0x2260:  // ≠
            out.tokens(i, i, "not equal to", 45);
            break;
        case 0x2261:  // ≡
            out.tokens(i, i, "is equivalent to", 45);
            break;
        case 0x226A:  // ≪
            out.tokens(i, i, "much less than", 45);
            break;
        case 0x226B:  // ≫
            out.tokens(i, i, "much greater than", 45);
            break;
        case 0xB1:  // ±
            out.tokens(i, i, "plus or minus", 40);
            break;
        case 0x2213:  // ∓
            out.tokens(i, i, "minus or plus", 40);
            break;
        case 0xF7:  // ÷
            out.tokens(i, i, "divided by", 45);
            break;
        case 0x221E:  // ∞
            out.tokens(i, i, "infinity", 45);
            break;
        case 0x221A:  // √
            out.tokens(i, i, i > 0 && doc.is(i - 1, "the") ? "square root of" : "the square root of", 45);
            break;
        case 0x221B:  // ∛
        case 0x221C:  // ∜
            out.tokens(i, i, std::string(i > 0 && doc.is(i - 1, "the") ? "" : "the ") +
                                 (cp == 0x221B ? "cube root of" : "fourth root of"), 45);
            break;
        case 0x221D:  // ∝
            out.tokens(i, i, "is proportional to", 45);
            break;
        case 0x2211: {  // ∑
            bool of = doc.is(i + 1, "of"), the = i > 0 && doc.is(i - 1, "the");
            out.tokens(i, i, of ? (the ? "sum" : "the sum") : (the ? "sum of" : "the sum of"), 40);
            break;
        }
        case 0x222B:  // ∫
            out.tokens(i, i, i > 0 && doc.is(i - 1, "the") ? "integral of" : "the integral of", 40);
            break;
        case '~':  // '~=' written for 'approximately equal' in prose
            if (doc.punct(i + 1, '=') && doc.joined(i + 1) && spaced_both(doc, i + 1) && doc[i].space_before)
                out.tokens(i, i + 1, "approximately", 45);
            break;
        case 0x2206:  // ∆ increment
            out.tokens(i, i, "delta", 45);
            break;
        case 0x2030:  // ‰ without a number before
            if (!(i > 0 && doc.number(i - 1) && doc.joined(i))) out.tokens(i, i, "per mille", 40);
            break;
        case '=': {
            if (at_line_start) break;
            bool dbl = doc.punct(i + 1, '=') && doc.joined(i + 1);
            if (i > 0 && doc.joined(i) && (doc.punct(i - 1, '=') || doc.punct(i - 1, '<') || doc.punct(i - 1, '>')))
                break;
            if (doc.punct(i + 1, '>') && doc.joined(i + 1)) break;  // '=>'
            size_t last = dbl ? i + 1 : i;
            if (dbl && doc.punct(i + 2, '=') && doc.joined(i + 2)) break;
            if (i > 0 && doc.joined(i) && doc.punct(i - 1, '!')) {
                out.tokens(i - 1, last, "not equal to", 45);
                break;
            }
            // 'key=value' inside a word run (URLs, code) is left to other modules unless it reads as math
            bool tight = doc.joined(i) && last + 1 < doc.size() && doc.joined(last + 1);
            bool half = last + 1 < doc.size() && doc.joined(i) != doc.joined(last + 1);
            if (half && !((doc.number(i - 1) || variable(doc, i - 1)) && operand_start(doc, last + 1)))
                break;  // base64 padding 'Q=', 'x =5' is rare
            if (tight && i >= 2 && doc.joined(i - 1) && punct(doc, i - 2) &&
                (doc[i - 2].cp == '?' || doc[i - 2].cp == '&' || doc[i - 2].cp == '/'))
                break;  // a query string
            // markup headings: '== Next steps =='
            if (dbl && (doc[i].newline_before || last + 1 >= doc.size() || doc[last + 1].newline_before)) break;
            if (tight && !((operand_end(doc, i - 1) || doc.word(i - 1)) &&
                           (operand_start(doc, last + 1) || !formula_letters(doc, last + 1).empty())))
                break;
            if (tight && doc.word(i - 1) && doc.str(i - 1).size() > 1 && !doc.number(last + 1)) break;
            if (i == 0 || last + 1 >= doc.size()) break;
            out.tokens(i, last, "equals", 45);
            // a one-letter variable before '=' is said as a letter: 'n = 400' -> 'N equals'; 'a' only before a number
            // ('a = 1.4'); x, y, z, i and a variable with a power ('k² = 0.76') stay
            size_t l = i - 1;
            std::string_view v = tok(doc, l);
            bool number_after = operand_start(doc, last + 1) && !variable(doc, last + 1);
            if (doc.word(l) && v.size() == 1 && (doc[l].casing & rv::LOWER) && !stuck_before(doc, l) &&
                !(l > 0 && doc.joined(l) && punct(doc, l - 1) && doc[l - 1].cp != '(') &&
                v != "x" && v != "y" && v != "z" && v != "i" && (v != "a" || (number_after && doc[i].space_before)))
                out.tokens(l, l, rv::capital(std::string(v)), 44);
            break;
        }
        case '+': {
            if (at_line_start) break;
            static const rv::WordSet KEYS{"shift", "ctrl", "control", "alt", "cmd", "command", "option", "opt",
                                          "fn", "win", "windows", "meta", "super", "tab", "enter", "esc"};
            if (i > 0 && doc.word(i - 1) && KEYS.contains(doc.low(i - 1))) break;
            if (doc.punct(i + 1, '+') || (i > 0 && doc.punct(i - 1, '+'))) break;  // C++
            // '$84 +, depending on': a plus after an amount, before punctuation
            if (doc[i].space_before && i > 0 && doc.number(i - 1) && i + 1 < doc.size() && doc.joined(i + 1) &&
                punct(doc, i + 1) && (doc[i + 1].cp == ',' || doc[i + 1].cp == '.' || doc[i + 1].cp == ';' ||
                                      doc[i + 1].cp == ')')) {
                out.tokens(i, i, "plus", 42);
                break;
            }
            // '163 (+ 2 below ground)': a plus at the start of a bracket, spaced from its number
            if (i > 0 && doc.joined(i) && doc.punct(i - 1, '(') && i + 1 < doc.size() && doc[i + 1].space_before &&
                !doc[i + 1].newline_before && doc.number(i + 1)) {
                out.tokens(i, i, "plus", 42);
                break;
            }
            if (spaced_both(doc, i)) {
                size_t r = i + 1;
                if (r < doc.size() && (doc.word(r) || operand_start(doc, r) || doc.punct(r, '$') ||
                                       doc.punct(r, 0x20AC) || doc.punct(r, 0xA3) || doc.punct(r, '"')))
                    out.tokens(i, i, "plus", 42);
                break;
            }
            // '10k+ members', '15%+ reduction', '$84+,': a plus after a quantity, before a space or punctuation
            if (i > 0 && doc.joined(i) && !(i + 1 < doc.size() && doc.joined(i + 1) && !punct(doc, i + 1))) {
                static const rv::WordSet SCALES{"k", "K", "M", "B", "bn", "m", "mn", "T", "tn"};
                bool scale = doc.word(i - 1) && SCALES.contains(doc.str(i - 1)) && doc.joined(i - 1) &&
                             doc.number(i - 2);
                bool money = doc.number(i - 1) && i >= 2 && (currency_sign(doc[i - 2].cp) || doc.number(i - 2) ||
                                                             doc.punct(i - 2, '.') || doc.punct(i - 2, ','));
                // '30 kg+': a plus after a unit of a measure
                bool unit = doc.word(i - 1) && doc.str(i - 1).size() <= 4 && i >= 2 && doc.number(i - 2) &&
                            !doc.joined(i - 1) && !(doc[i - 1].casing & rv::TITLE);
                if (scale || money || unit || (doc.punct(i - 1, '%') && doc.number(i - 2) && doc.joined(i - 1))) {
                    out.tokens(i, i, "plus", 42);
                    break;
                }
            }
            // '2+2': joined between numbers
            if (i > 0 && doc.joined(i) && doc.joined(i + 1) && doc.number(i - 1) && doc.number(i + 1) &&
                !(i >= 2 && doc.joined(i - 1) && doc.punct(i - 2, ':')) && !(doc.punct(i + 2, ':') && doc.joined(i + 2)))
                out.tokens(i, i, "plus", 42);  // not a time offset '23:15:00+00:00'
            break;
        }
        case '-':
        case 0x2212: {
            // a calculation: '7 - 2 = 5', '10 − 3'
            if (!spaced_both(doc, i)) {
                if (cp == 0x2212 && i > 0 && doc.joined(i) && doc.joined(i + 1) && operand_end(doc, i - 1) &&
                    operand_start(doc, i + 1) && !doc.punct(i - 1, '%'))
                    out.tokens(i, i, "minus", 42);
                // 'subtract 5-3', '7-3 equals 4': a calculation, not a range
                else if (cp == '-' && i > 0 && doc.joined(i) && doc.joined(i + 1) && doc.number(i - 1) &&
                         doc.number(i + 1) && !stuck_before(doc, i - 1) && !stuck_after(doc, i + 1) &&
                         !(i >= 2 && doc.joined(i - 1) && punct(doc, i - 2)) && !doc.punct(i + 2, '-')) {
                    size_t p = doc.prev_word(i - 1);
                    bool cue = (p != Doc::npos && (doc.is(p, "subtract") || doc.is(p, "calculate") ||
                                                   doc.is(p, "compute"))) ||
                               doc.is(i + 2, "equals") || (doc.punct(i + 2, '=') && !doc.joined(i + 2));
                    if (cue) out.tokens(i, i, "minus", 66);
                }
                break;
            }
            bool l = doc.number(i - 1) || variable(doc, i - 1) || doc.punct(i - 1, ')') || super_tok(doc, i - 1);
            bool r = doc.number(i + 1) || variable(doc, i + 1) || doc.punct(i + 1, '(');
            // a calculation outranks a range: '7 - 3 = 4', 'subtract 5 - 3'
            bool calc = false;
            if (l && r && doc.number(i - 1) && doc.number(i + 1)) {
                size_t e = i + 1;
                while (e + 2 < doc.size() && doc.joined(e + 1) && (doc.punct(e + 1, '.') || doc.punct(e + 1, ',')) &&
                       doc.number(e + 2))
                    e += 2;
                size_t p = doc.prev_word(i - 1);
                calc = doc.punct(e + 1, '=') ||
                       (p != Doc::npos && (doc.is(p, "subtract") || doc.is(p, "calculate") || doc.is(p, "compute")));
            }
            // words in a formula after '=': 'profit = revenue - expenses'
            if (!l && !r && doc.word(i - 1) && doc.word(i + 1) && (doc[i + 1].casing & rv::LOWER)) {
                for (size_t b = i - 1, n = 0; b-- > 0 && n < 4 && !doc[b + 1].newline_before; ++n)
                    if (doc.punct(b, '=') && doc[b].space_before) l = r = true;
            }
            if (cp == 0x2212 ? (l || r) : (l && r)) out.tokens(i, i, "minus", calc ? 66 : 42);
            break;
        }
        case 0xD7:
            // '2 × π': numbers_context owns dimensions and multipliers; scientific forms are in sci_times
            if (operand_end(doc, i - 1) && (variable(doc, i + 1) || doc.punct(i + 1, '(')) && !doc.is(i + 1, "x"))
                out.tokens(i, i, "times", 40);
            break;
        case '*':
            if (at_line_start) break;
            if (doc.joined(i) && doc.is(i - 1, "asterisk") && !stuck_after(doc, i)) {
                out.tokens(i, i, "", 35);  // 'an asterisk* in'
                break;
            }
            if ((spaced_both(doc, i) || (doc.joined(i) && doc.joined(i + 1))) && operand_end(doc, i - 1) &&
                operand_start(doc, i + 1) && !doc.punct(i - 1, '*') && !doc.punct(i + 1, '*'))
                out.tokens(i, i, "times", 42);
            else if (spaced_both(doc, i) && doc.word(i - 1) && doc.word(i + 1))
                out.tokens(i, i, doc.is(i - 1, "asterisk") ? "" : "asterisk", 35);
            break;
        case 0x21D2:  // ⇒ (the arrows → ↑ ↓ belong to structure.arrows)
        case 0x27F9:  // ⟹
            out.tokens(i, i, "implies", 52);
            break;
        default:
            break;
    }
}

// 'x ^ 2' / '10^6' -> 'to the power of'; a caret on its own is 'caret'; '^3' after a word is a footnote.
void caret_at(const Doc& doc, Out& out, size_t i) {
    // inside inline code the caret is a symbol: '`x ^ y`'
    int ticks = 0;
    for (size_t b = i; b-- > 0 && b + 40 > i && !doc[b + 1].newline_before;)
        if (doc.punct(b, '`')) ++ticks;
    if (ticks % 2 == 1) {
        out.tokens(i, i, "caret", 47);
        return;
    }
    bool base = i > 0 && operand_end(doc, i - 1) && !doc.punct(i - 1, '%');
    size_t e = i + 1;
    bool minus = false;
    if (e + 1 < doc.size() && (doc.punct(e, '-') || doc.punct(e, 0x2212)) && doc.joined(e) && doc.joined(e + 1) &&
        doc.number(e + 1)) {
        minus = true;
        ++e;
    }
    bool joined = doc.joined(i) && e < doc.size() && doc.joined(minus ? e - 1 : e);
    bool spaced = i > 0 && doc[i].space_before && e < doc.size() && doc[minus ? e - 1 : e].space_before;
    bool letters = i > 0 && !formula_letters(doc, i - 1).empty();
    if ((base || letters) && (joined || spaced) && e < doc.size() && (doc.number(e) || variable(doc, e))) {
        std::string ex;
        if (doc.number(e) && doc.punct(e + 1, '.') && doc.joined(e + 1) && doc.number(e + 2)) {
            // a decimal exponent: 'n^0.5'; a version is not one ('^1.2.3')
            if (doc.punct(e + 3, '.') && doc.joined(e + 3)) return;
            rv::NumberSpan sp = rv::parse_number(doc, e);
            ex = rv::read_number(sp);
            e = sp.last;
        }
        if (stuck_after(doc, e) && !doc.number(e)) return;
        if (doc.number(e) && stuck_after(doc, e)) return;
        // footnote after a word: 'load.^4', 'results^2' are citation markers
        if (doc.word(i - 1) && doc.str(i - 1).size() > 1 && !letters) return;
        if (ex.empty()) ex = doc.number(e) ? rv::cardinal_digits(doc.str(e)) : std::string(doc.str(e));
        if (ex.empty()) return;
        // a single letter exponent stays as written: '2^n' -> 'two to the power of n'
        out.tokens(i, e, std::string("to the power of ") + (minus ? "minus " : "") + ex, 46);
        return;
    }
    // a caret named as a symbol: 'the ^ symbol', '(^)', '^10'
    bool open_before = i == 0 || doc[i].space_before ||
                       (punct(doc, i - 1) && (doc[i - 1].cp == '(' || doc[i - 1].cp == '\'' ||
                                              doc[i - 1].cp == '"' || doc[i - 1].cp == '`' || doc[i - 1].cp == 0x2018 ||
                                              doc[i - 1].cp == 0x201C));
    if (open_before && !(doc.punct(i - 1, '.') || doc.punct(i - 1, '[')))
        out.tokens(i, i, "caret", 40);
}

// Superscript powers on numbers and one-letter variables: 'x²' -> 'x squared', '10⁶' -> 'ten to the power of
// six', 'R²'. A superscript after a longer word is a footnote (roman_citations).
void superscripts_at(const Doc& doc, Out& out, size_t i) {
    if (!super_tok(doc, i) && !(doc.punct(i, 0x207B) && super_tok(doc, i + 1) && doc.joined(i + 1))) return;
    if (!doc.joined(i)) return;
    size_t b = i - 1;
    bool num = doc.number(b);
    bool var = variable(doc, b) || doc.punct(b, ')') || greek_letter(doc, b) || !formula_letters(doc, b).empty() ||
               greek_pair(doc, b);
    if (!num && !var) return;
    bool minus = doc.punct(i, 0x207B);
    std::string dg;
    size_t last = digit_run(doc, minus ? i + 1 : i, true, dg);
    if (last == Doc::npos || stuck_after(doc, last)) return;
    // a decimal or grouped number before: '1.5²' is fine, '2023¹' after a year is a footnote
    bool ten = num && doc.str(b) == "10";
    if (num && !ten && doc.str(b).size() >= 4) return;
    if (num && b > 0 && doc.joined(b) && doc[b - 1].kind == Kind::Word) return;  // 'page42¹'
    std::string w = power_words(dg, minus, !ten);
    if (w.empty()) return;
    // after 'πr' or 'mc' the power must beat the footnote reading of a superscript after a word
    bool formula = !num && (greek_pair(doc, b) || !formula_letters(doc, b).empty());
    out.tokens(i, last, w, formula ? 63 : 45);
}

// Subscript digits after a letter: 'x₁' -> 'x one', 'CO₂' -> 'CO two', 'N₂O' -> 'N two O'.
void subscripts_at(const Doc& doc, Out& out, size_t i) {
    if (!sub_tok(doc, i) || !doc.joined(i)) return;
    if (!(doc.word(i - 1) || doc.punct(i - 1, ')'))) return;
    std::string dg;
    size_t last = digit_run(doc, i, false, dg);
    std::string w = rv::cardinal_digits(dg);
    if (w.empty()) return;
    out.tokens(i, last, w, 45);
}

// Greek letters standing alone: 'α = 0.05' -> 'alpha equals', 'σ', 'λ > 5', 'β-blocker'.
void greek_at(const Doc& doc, Out& out, size_t i) {
    if (!greek_letter(doc, i)) {
        // 'πr²', '2πr', 'Δt = 0.005', 'ΔCO₂'
        std::string pw = greek_pair_words(doc, i);
        if (pw.empty() || (stuck_before(doc, i) && !doc.number(i - 1)) ||
            (i + 1 < doc.size() && doc.joined(i + 1) && doc.word(i + 1)))
            return;
        out.tokens(i, i, pw, 45);
        if (stuck_before(doc, i)) {  // '2πr': the core leaves a number stuck to letters
            size_t f = i - 1;
            while (f >= 2 && doc.joined(f) && doc.punct(f - 1, '.') && doc.joined(f - 1) && doc.number(f - 2)) f -= 2;
            rv::NumberSpan sp = rv::parse_number(doc, f);
            std::string nw = rv::read_number(sp);
            if (sp.last == i - 1 && !nw.empty() && !stuck_before(doc, f)) out.tokens(f, i - 1, nw, 38);
        }
        return;
    }
    if (stuck_before(doc, i) || stuck_after(doc, i)) return;
    // inside Greek text the letters are words
    size_t p = doc.prev_word(i), n = doc.next_word(i);
    auto greek_word = [&](size_t k) {
        return k != Doc::npos && doc.word(k) && doc[k].cp >= 0x370 && doc[k].cp <= 0x3FF && doc.str(k).size() > 2;
    };
    if (greek_word(p) || greek_word(n)) return;
    out.tokens(i, i, greek_name(doc[i].cp), 45);
}

// Complexity notation: 'O(n)' -> 'O of N', 'O(n log n)', 'O(p²)' -> 'O of P squared', 'O(1)'.
void big_o_at(const Doc& doc, Out& out, size_t i) {
    std::string_view h = doc.str(i);
    if (!(doc.word(i) && (h == "O" || h == "Θ" || h == "Ω" || h == "o")) || stuck_before(doc, i)) return;
    if (!doc.punct(i + 1, '(') || !doc.joined(i + 1)) return;
    std::string words = std::string(h == "Θ" ? "Theta" : h == "Ω" ? "Omega" : h) + " of";
    size_t k = i + 2;
    bool ok = true, any = false;
    for (; k < doc.size() && !doc.punct(k, ')'); ++k) {
        if (doc[k].newline_before || k > i + 24) {
            ok = false;
            break;
        }
        const Token& t = doc[k];
        std::string piece;
        if (t.kind == Kind::Word) {
            std::string_view w = doc.str(k);
            if (w == "log" || w == "lg" || w == "ln" || w == "sqrt" || w == "exp" || w == "poly") piece = std::string(w);
            else if (greek_letter(doc, k)) piece = greek_name(t.cp);
            else if (w.size() <= 2) {
                for (char c : w) piece += static_cast<char>(c >= 'a' && c <= 'z' ? c - 32 : c);
            } else {
                ok = false;
                break;
            }
        } else if (t.kind == Kind::Number) {
            rv::NumberSpan s = rv::parse_number(doc, k);
            piece = rv::read_number(s);
            k = s.last;
        } else if (super_tok(doc, k)) {
            std::string dg;
            k = digit_run(doc, k, true, dg);
            piece = power_words(dg, false, true);
        } else if (t.cp == '^' && k + 1 < doc.size()) {
            ++k;
            if (doc.number(k)) {
                piece = "to the power of " + rv::cardinal_digits(doc.str(k));
            } else if (doc.word(k) && doc.str(k).size() == 1) {
                char c = doc.str(k)[0];
                piece = "to the power of " + std::string(1, c >= 'a' && c <= 'z' ? static_cast<char>(c - 32) : c);
            } else {
                ok = false;
                break;
            }
        } else if (t.cp == '+') piece = "plus";
        else if (t.cp == '-' || t.cp == 0x2212) piece = "minus";
        else if (t.cp == '*' || t.cp == 0xB7 || t.cp == 0xD7 || t.cp == 0x22C5) piece = "times";
        else if (t.cp == '/') piece = "over";
        else if (t.cp == '!') piece = "factorial";
        else if (t.cp == ',') piece = ",";
        else if (t.cp == '(' || t.cp == 0x221A) piece = t.cp == 0x221A ? "the square root of" : "";
        else {
            ok = false;
            break;
        }
        if (piece.empty()) continue;
        if (piece == ",") words += ",";
        else words += " " + piece;
        any = true;
    }
    if (!ok || !any || k >= doc.size()) return;
    out.tokens(i, k, words, 72);  // above the parts ('u' as chat shorthand)
}

// Stars: '4.4★' -> 'four point four stars', '★★☆☆☆' -> 'two out of five stars'.
void stars_at(const Doc& doc, Out& out, size_t i) {
    if (!(doc.punct(i, 0x2605) || doc.punct(i, 0x2606))) return;
    if (doc.joined(i) && (doc.punct(i - 1, 0x2605) || doc.punct(i - 1, 0x2606))) return;  // inside a run
    size_t k = i;
    int full = 0, total = 0;
    while (k < doc.size() && (k == i || doc.joined(k)) && (doc.punct(k, 0x2605) || doc.punct(k, 0x2606))) {
        full += doc.punct(k, 0x2605);
        ++total;
        ++k;
    }
    size_t last = k - 1;
    if (total >= 2) {
        out.tokens(i, last, rv::cardinal(full) + " out of " + rv::cardinal(total) + " stars", 50);
    } else if (i > 0 && doc.number(i - 1) && !doc[i].newline_before) {
        rv::NumberSpan s = rv::parse_number(doc, i - 1);
        size_t f = i - 1;
        while (f >= 2 && doc.joined(f) && doc.punct(f - 1, '.') && doc.number(f - 2)) f -= 2;
        s = rv::parse_number(doc, f);
        bool one = s.whole == "1" && s.fraction.empty();
        out.tokens(i, i, one ? "star" : "stars", 50);
    }
}

// Trademark and copyright signs, '&' and '@': 'Bolt®' -> 'Bolt registered trademark', 'P&L' -> 'P and L'.
void signs_at(const Doc& doc, Out& out, size_t i) {
    uint32_t cp = doc[i].cp;
    const char* w = nullptr;
    switch (cp) {
        case 0xAE: w = "registered trademark"; break;
        case 0x2122: w = "trademark"; break;
        case 0xA9: w = "copyright"; break;
        case 0x2120: w = "service mark"; break;
        case 0x2117: w = "sound recording copyright"; break;
        default: break;
    }
    if (cp == 0xA9 && ((i > 0 && doc.is(i - 1, "copyright")) || doc.is(i + 1, "copyright"))) {
        out.tokens(i, i, "", 50);  // 'Copyright © 2021' says the word once
        return;
    }
    if (w) {
        // after a closing quote the word needs its own space: '"Leap"™' -> '"Leap" trademark'
        std::string s = w;
        if (i >= 2 && doc.joined(i) && punct(doc, i - 1) && doc.joined(i - 1) && doc[i - 2].kind != Kind::Punct)
            s = " " + s;
        out.tokens(i, i, s, 50);
        return;
    }
    if (cp == '&') {
        if (doc.joined(i) && doc.punct(i - 1, '&')) return;  // second half of '&&'
        // not an HTML entity ('&amp;', '&#39;') or a query string
        if (doc.joined(i + 1) && (doc.word(i + 1) || doc.punct(i + 1, '#')) && doc.punct(i + 2, ';')) return;
        if (doc.joined(i + 1) && doc.punct(i + 1, '#')) return;
        if (doc.punct(i + 1, '&') && doc.joined(i + 1)) {
            out.tokens(i, i + 1, "and", 45);
            return;
        }
        if (i + 1 >= doc.size() || i == 0) return;
        if (doc.joined(i) && doc.joined(i + 1) && doc.punct(i + 2, '=') && doc.joined(i + 2)) return;  // a=1&b=2
        out.tokens(i, i, "and", 45);
        return;
    }
    if (cp == '@') {
        if (i == 0 || i + 1 >= doc.size()) return;
        bool before_free = doc[i].space_before;
        bool after_free = doc[i + 1].space_before;
        if (before_free && (after_free || doc.number(i + 1) || doc.punct(i + 1, '$')))
            out.tokens(i, i, "at", 40);
    }
}

// '(c) 2020 Smith & Sons', '(C) 2019': the copyright sign written with letters before a year.
void copyright_letters_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.punct(i, '(') || !doc.word(i + 1) || !(doc.str(i + 1) == "c" || doc.str(i + 1) == "C") ||
        !doc.punct(i + 2, ')') || !doc.joined(i + 1) || !doc.joined(i + 2) || !doc.number(i + 3) ||
        doc.str(i + 3).size() != 4 || !doc[i + 3].space_before || stuck_before(doc, i))
        return;
    std::string_view y = doc.str(i + 3);
    if (y[0] != '1' && y[0] != '2') return;
    out.tokens(i, i + 2, "copyright", 55);
}

// Probability notation: 'P(A)' -> 'P A', 'P(A|B)' -> 'P A given B'.
void probability_at(const Doc& doc, Out& out, size_t i) {
    if (doc.str(i) != "P" || stuck_before(doc, i) || !doc.punct(i + 1, '(') || !doc.joined(i + 1)) return;
    size_t a = i + 2;
    if (!doc.word(a) || doc.str(a).size() != 1 || !(doc[a].casing & rv::UPPER) || !doc.joined(a)) return;
    std::string words = "P " + std::string(doc.str(a));
    size_t k = a + 1;
    if (doc.punct(k, '|') && doc.joined(k) && doc.word(k + 1) && doc.str(k + 1).size() == 1 &&
        (doc[k + 1].casing & rv::UPPER)) {
        words += " given " + std::string(doc.str(k + 1));
        k += 2;
    }
    if (!doc.punct(k, ')') || !doc.joined(k) || stuck_after(doc, k)) return;
    out.tokens(i, k, words, 60);
}

// A confidence interval after a percentage where numbers_core does not read it: '95% CI, 0.65–0.94',
// '95% CI: 0.12–0.34', 'the 95% CI for the mean', 'the 95 % CI was 1.1–2.3'.
void ci_at(const Doc& doc, Out& out, size_t i) {
    if (doc.str(i) != "CI" || i == 0 || !doc.punct(i - 1, '%') || !doc[i].space_before) return;
    // 'the 95% CI for the mean', 'the 80% CI on the central case', 'the 95% CI was 1.1–2.3'
    if (doc.is(i + 1, "for") || doc.is(i + 1, "on") || doc.is(i + 1, "of")) {
        out.tokens(i, i, "confidence interval", 60);
        return;
    }
    size_t k = i + 2;
    if ((doc.is(i + 1, "was") || doc.is(i + 1, "is")) && (doc.number(k) || doc.punct(k, '['))) {
        out.tokens(i, i, "confidence interval", 60);
        return;
    }
    if (!(doc.punct(i + 1, ',') || doc.punct(i + 1, ':')) || !doc.joined(i + 1)) return;
    if (doc.punct(k, '[') || doc.punct(k, '(')) ++k;
    if (doc.number(k) || ((doc.punct(k, '-') || doc.punct(k, 0x2212) || doc.punct(k, '.')) && doc.number(k + 1)))
        out.tokens(i, i, "confidence interval", 60);
}

// A coefficient in an equation: 'y = 2.1x + 0.4' -> 'two point one x', '= 3n - 1'. i: the number start.
void coefficient_at(const Doc& doc, Out& out, size_t i) {
    if (i == 0 || !doc.punct(i - 1, '=') || !doc[i].space_before || !doc[i - 1].space_before) return;
    rv::NumberSpan sp = rv::parse_number(doc, i);
    size_t v = sp.last + 1;
    if (!doc.word(v) || doc.str(v).size() != 1 || !doc.joined(v) || !(doc[v].casing & rv::LOWER) ||
        stuck_after(doc, v))
        return;
    size_t k = v + 1;
    bool end_ok = k >= doc.size() || doc[k].newline_before ||
                  ((doc.punct(k, '+') || doc.punct(k, '-') || doc.punct(k, 0x2212)) && doc[k].space_before) ||
                  (punct(doc, k) && (doc[k].cp == ',' || doc[k].cp == '.' || doc[k].cp == ';' || doc[k].cp == ')'));
    std::string w = rv::read_number(sp);
    if (!end_ok || w.empty()) return;
    out.tokens(i, v, w + " " + std::string(doc.str(v)), 55);
}

// 'She has MS.', 'works at MS.': the initialism before a sentence period is not the title 'Ms.' (A13).
void ms_at(const Doc& doc, Out& out, size_t i) {
    if (doc.str(i) != "MS" || !doc.punct(i + 1, '.') || !doc.joined(i + 1) || i == 0 || stuck_before(doc, i)) return;
    static const rv::WordSet BEFORE{"about", "against", "at", "by", "diagnosed", "for", "from", "had", "has", "have",
                                    "join", "joined", "joins", "of", "on", "with"};
    if (doc.word(i - 1) && BEFORE.contains(doc.low(i - 1))) out.keep(i, i, 90);
}

// '1:10^6' -> 'one to ten to the power of six': a ratio whose second part is a power of ten.
void ratio_power_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.punct(i, ':') || !doc.joined(i) || !doc.joined(i + 1) || !doc.number(i - 1) || doc.str(i + 1) != "10" ||
        !doc.joined(i + 2) || !(doc.punct(i + 2, '^') || super_tok(doc, i + 2)) || stuck_before(doc, i - 1))
        return;
    out.tokens(i, i, "to", 47);
}

// One scan for every symbol.
RV_RULE("units_symbols", symbols) {
    for (size_t i = 0; i < doc.size(); ++i) {
        const Token& t = doc[i];
        if (t.kind == Kind::Number) {
            if (i > 0 && doc.punct(i - 1, '=')) coefficient_at(doc, out, i);
            continue;
        }
        if (t.kind == Kind::Word) {
            if (t.cp >= 0x391 || t.cp == 0xB5) greek_at(doc, out, i);
            if (t.cp >= 'a' && t.cp <= 'z' && t.end - t.start == 2) {
                std::string fl = formula_letters(doc, i);
                if (!fl.empty()) out.tokens(i, i, fl, 45);
            }
            if ((t.cp == 'O' || t.cp == 'o' || t.cp == 0x398 || t.cp == 0x3A9) && i + 3 < doc.size()) big_o_at(doc, out, i);
            if (t.cp == 'P' && i + 3 < doc.size()) probability_at(doc, out, i);
            if (t.cp == 'C' && i + 2 < doc.size()) ci_at(doc, out, i);
            if (t.cp == 'M' && i + 1 < doc.size()) ms_at(doc, out, i);
            continue;
        }
        if (t.kind != Kind::Punct) continue;
        uint32_t cp = t.cp;
        if (cp == '(' && i + 3 < doc.size()) copyright_letters_at(doc, out, i);
        if (cp == ':' && i + 2 < doc.size() && i > 0) ratio_power_at(doc, out, i);
        if (cp == '^') caret_at(doc, out, i);
        else if (super_digit(cp) >= 0 || cp == 0x207B) superscripts_at(doc, out, i);
        else if (sub_digit(cp) >= 0) subscripts_at(doc, out, i);
        else if (cp == 0x2605 || cp == 0x2606) stars_at(doc, out, i);
        else if (cp == 0xAE || cp == 0x2122 || cp == 0xA9 || cp == 0x2120 || cp == 0x2117 || cp == '&' || cp == '@')
            signs_at(doc, out, i);
        else if (cp == '|' && doc.punct(i + 1, '|') && doc.joined(i + 1) && t.space_before && i + 2 < doc.size() &&
                 doc[i + 2].space_before && i > 0 && !doc[i].newline_before && !doc[i + 2].newline_before)
            out.tokens(i, i + 1, "or", 45);  // 'retries > 3 || timeout': logical or, like '&&'
        else operators_at(doc, out, i);
    }
}

}  // namespace
