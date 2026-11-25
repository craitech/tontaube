// TeX source -> tree. A hand-written tokenizer and recursive-descent parser for the math-mode TeX that web
// pages carry (MathJax, KaTeX, Wikipedia, MathML converted to TeX), Unicode math symbols included. Unknown
// commands keep their arguments; unbalanced braces and delimiters are recovered; nothing throws.
#include <cstring>

#include "core/readers.h"
#include "core/utf8.h"
#include "rules/math/math.h"

namespace mth {

namespace {

bool is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_digit(char c) { return c >= '0' && c <= '9'; }

Node make(K k) {
    Node n;
    n.k = k;
    return n;
}
Node make_sym(const Sym* s) {
    Node n = make(K::Sym);
    n.s = s;
    return n;
}
Node make_letter(uint32_t cp, uint8_t upright = 0) {
    Node n = make(K::Letter);
    n.cp = cp;
    n.n = upright;
    return n;
}
Node make_text(K k, std::string t) {
    Node n = make(k);
    n.t = std::move(t);
    return n;
}
// A row of one item is that item.
Node unwrap(Node row) {
    if (row.k == K::Row && row.kids.size() == 1) {
        Node one = std::move(row.kids[0]);
        return one;
    }
    return row;
}

int super_value(uint32_t cp) {
    switch (cp) {
        case 0x2070: return 0;
        case 0xB9: return 1;
        case 0xB2: return 2;
        case 0xB3: return 3;
        default: return cp >= 0x2074 && cp <= 0x2079 ? static_cast<int>(cp - 0x2070) : -1;
    }
}
int sub_value(uint32_t cp) { return cp >= 0x2080 && cp <= 0x2089 ? static_cast<int>(cp - 0x2080) : -1; }

// Delimiter written after \left, \right, \big...: its id, or "" for none ('.'), or "?" when not a delimiter.
enum class Side { Open, Close, Either };
struct Delim {
    std::string id;
    Side side;
};

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}

    Node run() {
        Node t = table_body("", /*top=*/true);
        if (t.kids.size() == 1 && t.kids[0].kids.size() == 1) return unwrap(std::move(t.kids[0].kids[0]));
        return t;
    }

private:
    enum class Stop { End, Brace, Right, Cell, Bracket };

    static constexpr int MAX_DEPTH = 32;  // rows and atoms on the way down; keeps the stack small (WebAssembly)

    std::string_view s_;
    size_t p_ = 0;
    int depth_ = 0;

    bool at_end() const { return p_ >= s_.size(); }
    char peek(size_t k = 0) const { return p_ + k < s_.size() ? s_[p_ + k] : '\0'; }
    void skip_ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\t' || s_[p_] == '\n' || s_[p_] == '\r')) ++p_;
    }
    // The command name at p_ (after the backslash), without consuming it.
    std::string_view peek_cmd() const {
        if (peek() != '\\' || p_ + 1 >= s_.size()) return {};
        size_t q = p_ + 1;
        if (!is_alpha(s_[q])) return s_.substr(q, 1);
        while (q < s_.size() && is_alpha(s_[q])) ++q;
        if (q < s_.size() && s_[q] == '*') ++q;  // \operatorname*
        return s_.substr(p_ + 1, q - p_ - 1);
    }
    std::string_view read_cmd() {
        std::string_view c = peek_cmd();
        p_ += 1 + c.size();
        return c;
    }
    // Raw content of a brace group at p_ (spaces before it skipped); consumes it. Unbalanced: to the end.
    std::string_view raw_group() {
        skip_ws();
        if (peek() != '{') {
            if (at_end()) return {};
            size_t a = p_;  // a single token: \cmd or one character
            if (peek() == '\\') read_cmd();
            else rv::decode(s_, p_);
            return s_.substr(a, p_ - a);
        }
        size_t a = ++p_;
        int level = 1;
        while (p_ < s_.size()) {
            char c = s_[p_];
            if (c == '\\' && p_ + 1 < s_.size()) {
                p_ += 2;
                continue;
            }
            if (c == '{') ++level;
            if (c == '}' && --level == 0) break;
            ++p_;
        }
        std::string_view g = s_.substr(a, p_ - a);
        if (p_ < s_.size()) ++p_;
        return g;
    }
    void skip_group() { raw_group(); }
    // An optional [..] argument: its raw text, consumed ("" if none).
    bool optional_arg(Node* out) {
        skip_ws();
        if (peek() != '[') return false;
        ++p_;
        Node r = row(Stop::Bracket);
        if (peek() == ']') ++p_;
        if (out) *out = unwrap(std::move(r));
        return true;
    }

    // A command argument: a brace group, a command with its own arguments, or one character (one digit).
    Node arg() {
        skip_ws();
        if (at_end()) return make(K::Empty);
        char c = peek();
        if (c == '{') {
            ++p_;
            Node r = row(Stop::Brace);
            if (peek() == '}') ++p_;
            if (r.kids.empty()) return make(K::Empty);
            return unwrap(std::move(r));
        }
        if (is_digit(c)) {
            ++p_;
            return make_text(K::Num, std::string(1, c));
        }
        if (c == '}' || c == '&' || c == '^' || c == '_') return make(K::Empty);
        return atom();
    }

    // A row of items up to the stop.
    Node row(Stop stop) {
        Node r = make(K::Row);
        if (++depth_ > MAX_DEPTH) {  // nested deeper than any real formula: the rest is not read
            --depth_;
            p_ = s_.size();
            return r;
        }
        int split = -1;
        bool binom = false;
        while (true) {
            skip_ws();
            if (at_end()) break;
            char c = peek();
            if (c == '}') {
                if (stop == Stop::Brace || stop == Stop::Cell) break;
                ++p_;  // a stray closing brace
                continue;
            }
            if (c == ']' && stop == Stop::Bracket) break;
            if (c == '&') {
                if (stop == Stop::Cell) break;
                ++p_;
                continue;
            }
            if (c == '\\') {
                std::string_view cmd = peek_cmd();
                if (cmd == "\\" || cmd == "cr" || cmd == "newline") {
                    if (stop == Stop::Cell) break;
                    read_cmd();
                    optional_arg(nullptr);  // \\[2pt]
                    Node sp = make(K::Space);
                    sp.n = 1;
                    r.kids.push_back(sp);
                    continue;
                }
                if (cmd == "right") {
                    if (stop == Stop::Right) break;
                    read_cmd();
                    delimiter();
                    continue;
                }
                if (cmd == "end") {
                    if (stop == Stop::Cell) break;
                    read_cmd();
                    skip_group();
                    continue;
                }
                if (cmd == "over" || cmd == "choose" || cmd == "atop" || cmd == "brack" || cmd == "brace") {
                    read_cmd();
                    if (split < 0) {
                        split = static_cast<int>(r.kids.size());
                        binom = cmd == "choose";
                    }
                    continue;
                }
                if (cmd == "hline" || cmd == "hdashline" || cmd == "midrule" || cmd == "toprule" ||
                    cmd == "bottomrule") {
                    read_cmd();
                    continue;
                }
            }
            size_t before = p_;
            Node it = item();
            if (it.k != K::Empty) r.kids.push_back(std::move(it));
            if (p_ == before) ++p_;  // never stall
        }
        --depth_;
        if (split >= 0) {
            Node num = make(K::Row), den = make(K::Row);
            for (size_t k = 0; k < r.kids.size(); ++k)
                (static_cast<int>(k) < split ? num : den).kids.push_back(std::move(r.kids[k]));
            Node f = make(K::Frac);
            f.n = binom;
            fences(num);
            fences(den);
            f.kids.push_back(unwrap(std::move(num)));
            f.kids.push_back(unwrap(std::move(den)));
            r.kids.clear();
            r.kids.push_back(std::move(f));
            return r;
        }
        fences(r);
        return r;
    }

    // An atom and its scripts and primes.
    Node item() {
        Node base = atom();
        Node sub = make(K::Empty), sup = make(K::Empty);
        int primes = 0;
        bool any = false;
        while (true) {
            skip_ws();
            char c = peek();
            if (c == '^' || c == '_') {
                ++p_;
                Node a = arg();
                Node& slot = c == '^' ? sup : sub;
                if (slot.k != K::Empty) {  // x^a^b: nest
                    base = scripts(std::move(base), std::move(sub), std::move(sup), primes);
                    sub = make(K::Empty);
                    sup = make(K::Empty);
                    primes = 0;
                }
                (c == '^' ? sup : sub) = std::move(a);
                any = true;
                continue;
            }
            if (c == '\'') {
                ++p_;
                ++primes;
                any = true;
                continue;
            }
            if (c == '\\') {
                std::string_view cmd = peek_cmd();
                if (cmd == "limits" || cmd == "nolimits") {
                    read_cmd();
                    continue;
                }
                break;
            }
            if (static_cast<unsigned char>(c) >= 0x80) {
                size_t q = p_;
                uint32_t cp = rv::decode(s_, q);
                if (cp == 0x2032 || cp == 0x2033 || cp == 0x2034) {
                    primes += static_cast<int>(cp - 0x2031);
                    p_ = q;
                    any = true;
                    continue;
                }
                int sv = super_value(cp), bv = sub_value(cp);
                if (sv >= 0 || bv >= 0) {
                    std::string digits;
                    bool super = sv >= 0;
                    size_t r = p_;
                    while (r < s_.size()) {
                        size_t q2 = r;
                        uint32_t d = rv::decode(s_, q2);
                        int v = super ? super_value(d) : sub_value(d);
                        if (v < 0) break;
                        digits += static_cast<char>('0' + v);
                        r = q2;
                    }
                    p_ = r;
                    (super ? sup : sub) = make_text(K::Num, digits);
                    any = true;
                    continue;
                }
            }
            break;
        }
        if (!any) return base;
        return scripts(std::move(base), std::move(sub), std::move(sup), primes);
    }

    static Node scripts(Node base, Node sub, Node sup, int primes) {
        // \underbrace{x}_{label}: the label is not read
        if (base.k == K::Accent && base.t == "brace") return base;
        Node n = make(K::Scripts);
        n.n = static_cast<uint8_t>(primes > 3 ? 3 : primes);
        n.kids.push_back(std::move(base));
        n.kids.push_back(std::move(sub));
        n.kids.push_back(std::move(sup));
        return n;
    }

    Node number() {
        size_t a = p_;
        std::string t;
        auto digits = [&]() {
            while (p_ < s_.size() && is_digit(s_[p_])) t += s_[p_++];
        };
        auto three_digits_at = [&](size_t q) {
            return q + 3 <= s_.size() && is_digit(s_[q]) && is_digit(s_[q + 1]) && is_digit(s_[q + 2]) &&
                   (q + 3 == s_.size() || !is_digit(s_[q + 3]));
        };
        digits();
        bool group_ok = p_ - a <= 3 && !t.empty();
        while (group_ok) {  // 1{,}000  1\,000  1,000
            if (s_.compare(p_, 3, "{,}") == 0 && three_digits_at(p_ + 3)) p_ += 3;
            else if (s_.compare(p_, 2, "\\,") == 0 && three_digits_at(p_ + 2)) p_ += 2;
            else if (peek() == ',' && three_digits_at(p_ + 1)) p_ += 1;
            else break;
            t += ',';
            digits();
        }
        if ((peek() == '.' && is_digit(peek(1))) || (s_.compare(p_, 3, "{.}") == 0 && p_ + 3 < s_.size() &&
                                                     is_digit(s_[p_ + 3]))) {
            p_ += peek() == '.' ? 1 : 3;
            t += '.';
            digits();
        }
        return make_text(K::Num, t);
    }

    Node atom() {
        skip_ws();
        if (at_end()) return make(K::Empty);
        if (depth_ >= MAX_DEPTH) {  // nested deeper than any real formula: the rest is not read
            p_ = s_.size();
            return make(K::Empty);
        }
        struct Level {
            int& d;
            explicit Level(int& x) : d(x) { ++d; }
            ~Level() { --d; }
        } level(depth_);
        char c = peek();
        if (c == '{') {
            ++p_;
            Node r = row(Stop::Brace);
            if (peek() == '}') ++p_;
            if (r.kids.empty()) return make(K::Empty);  // {}^{14}C
            return unwrap(std::move(r));
        }
        if (is_digit(c) || (c == '.' && is_digit(peek(1)))) return number();
        if (is_alpha(c)) return letters();
        if (c == '\\') return command();
        if (static_cast<unsigned char>(c) >= 0x80) return unicode();
        ++p_;
        switch (c) {
            case '(': case '[': return make_text(K::Open, std::string(1, c));
            case ')': return make_text(K::Close, "(");
            case ']': return make_text(K::Close, "[");
            case '|': return make_text(K::Bar, "|");
            case '~': return make_sym(sym_cp('~'));  // a tie, or a tilde operator between two operands
            case '^': case '_': --p_; return make(K::Empty);  // scripts without a base
            case '%': return make_text(K::Word, "percent");
            case '.':
                if (peek() == '.') {  // ... written with periods
                    while (peek() == '.') ++p_;
                    return make_sym(sym_tex("ldots"));
                }
                break;
            default: break;
        }
        const Sym* s = sym_cp(static_cast<unsigned char>(c));
        return s ? make_sym(s) : make(K::Empty);
    }

    // One letter, or a function name written in plain letters ("sin x" without a backslash).
    Node letters() {
        size_t q = p_;
        while (q < s_.size() && is_alpha(s_[q])) ++q;
        std::string_view run = s_.substr(p_, q - p_);
        if (run.size() >= 2 && run.size() <= 3) {
            static const char* const PLAIN[] = {"sin", "cos", "tan", "log", "ln", "exp", "lim", "max", "min", "det"};
            for (const char* w : PLAIN)
                if (run == w) {
                    p_ = q;
                    const Sym* s = sym_tex(run);
                    return s ? make_sym(s) : make_text(K::Word, std::string(run));
                }
        }
        return make_letter(static_cast<unsigned char>(s_[p_++]));
    }

    Node unicode() {
        uint32_t cp = rv::decode(s_, p_);
        if (uint32_t plain = rv::plain_math_alnum(cp)) cp = plain;
        if (cp < 0x80) {
            if (is_digit(static_cast<char>(cp))) return make_text(K::Num, std::string(1, static_cast<char>(cp)));
            return make_letter(cp);
        }
        if (cp == 0x2202 || cp == 0x2207) return make_sym(sym_cp(cp));
        if ((cp >= 0x391 && cp <= 0x3C9 && cp != 0x3A2) || cp == 0x3D1 || cp == 0x3D5 || cp == 0x3D6 || cp == 0x3F0 ||
            cp == 0x3F1 || cp == 0x3F4 || cp == 0x3F5 || cp == 0xB5)
            return make_letter(cp);
        switch (cp) {
            case 0x211D: case 0x2115: case 0x2124: case 0x211A: case 0x2102: {
                Node n = make(K::Set);
                n.cp = cp == 0x211D ? 'R' : cp == 0x2115 ? 'N' : cp == 0x2124 ? 'Z' : cp == 0x211A ? 'Q' : 'C';
                return n;
            }
            case 0x2113: return make_letter('l');
            case 0x2016: case 0x2225: return make_text(K::Bar, "\xE2\x80\x96");
            case 0x2223: return make_text(K::Bar, "mid");
            case 0x27E8: case 0x2329: return make_text(K::Open, "<");
            case 0x27E9: case 0x232A: return make_text(K::Close, "<");
            case 0x230A: return make_text(K::Open, "floor");
            case 0x230B: return make_text(K::Close, "floor");
            case 0x2308: return make_text(K::Open, "ceil");
            case 0x2309: return make_text(K::Close, "ceil");
            case 0x2061: case 0x2062: case 0x2063: case 0x2064: case 0xA0: case 0x2009: case 0x200A: case 0x202F:
                return make(K::Space);
            default: break;
        }
        if (const Sym* s = sym_cp(cp)) return make_sym(s);
        if (const char* f = rv::latin_fold(cp)) return make_text(K::Word, f);
        if (rv::is_letter(cp) && cp < 0x250) {  // an accented letter the model knows: keep it as a word
            std::string w;
            rv::append_utf8(w, cp);
            return make_text(K::Word, w);
        }
        return make(K::Empty);
    }

    // The delimiter after \left, \right, \big...
    Delim delimiter() {
        skip_ws();
        if (at_end()) return {"", Side::Either};
        char c = peek();
        if (c == '\\') {
            std::string_view d = read_cmd();
            if (d == "{" || d == "lbrace") return {"{", Side::Open};
            if (d == "}" || d == "rbrace") return {"{", Side::Close};
            if (d == "langle") return {"<", Side::Open};
            if (d == "rangle") return {"<", Side::Close};
            if (d == "lfloor") return {"floor", Side::Open};
            if (d == "rfloor") return {"floor", Side::Close};
            if (d == "lceil") return {"ceil", Side::Open};
            if (d == "rceil") return {"ceil", Side::Close};
            if (d == "lvert") return {"|", Side::Open};
            if (d == "rvert") return {"|", Side::Close};
            if (d == "lVert") return {"\xE2\x80\x96", Side::Open};
            if (d == "rVert") return {"\xE2\x80\x96", Side::Close};
            if (d == "|" || d == "Vert") return {"\xE2\x80\x96", Side::Either};
            if (d == "vert") return {"|", Side::Either};
            if (d == "lbrack") return {"[", Side::Open};
            if (d == "rbrack") return {"[", Side::Close};
            if (d == "lparen") return {"(", Side::Open};
            if (d == "rparen") return {"(", Side::Close};
            if (d == "mid") return {"mid", Side::Either};
            return {"", Side::Either};
        }
        if (static_cast<unsigned char>(c) >= 0x80) {
            size_t q = p_;
            uint32_t cp = rv::decode(s_, q);
            p_ = q;
            switch (cp) {
                case 0x27E8: case 0x2329: return {"<", Side::Open};
                case 0x27E9: case 0x232A: return {"<", Side::Close};
                case 0x230A: return {"floor", Side::Open};
                case 0x230B: return {"floor", Side::Close};
                case 0x2308: return {"ceil", Side::Open};
                case 0x2309: return {"ceil", Side::Close};
                case 0x2016: case 0x2225: return {"\xE2\x80\x96", Side::Either};
                default: return {"", Side::Either};
            }
        }
        ++p_;
        switch (c) {
            case '(': return {"(", Side::Open};
            case ')': return {"(", Side::Close};
            case '[': return {"[", Side::Open};
            case ']': return {"[", Side::Close};
            case '|': return {"|", Side::Either};
            case '<': return {"<", Side::Open};
            case '>': return {"<", Side::Close};
            default: return {"", Side::Either};  // '.', '/', anything else: no delimiter
        }
    }

    Node delimiter_atom() {
        Delim d = delimiter();
        if (d.id.empty()) return make(K::Empty);
        if (d.id == "mid") return make_text(K::Bar, "mid");
        if (d.side == Side::Either) return make_text(K::Bar, d.id);
        return make_text(d.side == Side::Open ? K::Open : K::Close, d.id);
    }

    // The words of \text{...}: letters, digits and sentence punctuation; $...$ inside is math.
    Node text(std::string_view raw) {
        Node r = make(K::Row);
        size_t i = 0;
        std::string cur;
        auto flush = [&]() {
            std::string w;  // single spaces, none at the ends
            for (char ch : cur)
                if (ch != ' ' || (!w.empty() && w.back() != ' ')) w += ch;
            while (!w.empty() && w.back() == ' ') w.pop_back();
            if (!w.empty()) r.kids.push_back(make_text(K::Word, w));
            cur.clear();
        };
        while (i < raw.size()) {
            char c = raw[i];
            if (c == '$') {
                size_t e = raw.find('$', i + 1);
                if (e == std::string_view::npos) e = raw.size();
                flush();
                Node m = Parser(raw.substr(i + 1, e - i - 1)).run();
                r.kids.push_back(std::move(m));
                i = e + 1;
                continue;
            }
            if (c == '\\') {  // \, \  \% \& \_ inside text
                size_t q = i + 1;
                if (q < raw.size() && is_alpha(raw[q])) {
                    while (q < raw.size() && is_alpha(raw[q])) ++q;
                } else {
                    char e = q < raw.size() ? raw[q] : ' ';
                    if (e == '%') cur += " percent";
                    else if (e == '&') cur += " and";
                    ++q;
                }
                cur += ' ';
                i = q;
                continue;
            }
            if (static_cast<unsigned char>(c) >= 0x80) {
                size_t q = i;
                uint32_t cp = rv::decode(raw, q);
                if (uint32_t pl = rv::plain_math_alnum(cp)) cp = pl;
                if (cp < 0x80) cur += static_cast<char>(cp);
                else if (const char* f = rv::latin_fold(cp)) cur += f;
                else if (rv::is_letter(cp) && cp < 0x250) cur.append(raw.substr(i, q - i));
                else if (cp == 0x2019) cur += '\'';
                else cur += ' ';
                i = q;
                continue;
            }
            cur += (c == '{' || c == '}' || c == '~') ? ' ' : c;
            ++i;
        }
        flush();
        return unwrap(std::move(r));
    }

    // \mathrm{...}, \operatorname{...}: a function name, a word, an upright letter, or math.
    Node upright(std::string_view raw) {
        // a unit with a slash or a power: m/s^2, km/h, m^2
        if (raw.find_first_of("/^") != std::string_view::npos && raw.size() < 24) {
            std::string u;
            bool ok = true;
            for (char ch : raw) {
                if (is_alpha(ch) || is_digit(ch) || ch == '/' || ch == '^') u += ch;
                else if (ch != '{' && ch != '}' && ch != ' ') ok = false;
            }
            if (ok && !u.empty() && is_alpha(u[0])) return make_text(K::Word, u);
        }
        std::string w;
        bool letters_only = !raw.empty();
        for (size_t i = 0; i < raw.size(); ++i) {
            char c = raw[i];
            if (is_alpha(c)) w += c;
            else if (c == ' ') continue;
            else if (c == '\\' && i + 1 < raw.size() && !is_alpha(raw[i + 1])) ++i;  // arg\,max
            else {
                letters_only = false;
                break;
            }
        }
        if (letters_only && !w.empty()) {
            if (w.size() == 1) return make_letter(static_cast<unsigned char>(w[0]), 1);
            if (const Sym* f = func_word(w)) return make_sym(f);
            return make_text(K::Word, w);
        }
        return unwrap(Parser(raw).run());
    }

    // \ce{H2O}: element symbols, digits as numbers.
    Node chemistry(std::string_view raw) {
        std::string out;
        for (size_t i = 0; i < raw.size();) {
            char c = raw[i];
            if (is_digit(c)) {
                size_t q = i;
                while (q < raw.size() && is_digit(raw[q])) ++q;
                out += ' ';
                out += rv::cardinal_digits(raw.substr(i, q - i));
                out += ' ';
                i = q;
            } else if (c >= 'A' && c <= 'Z') {
                out += ' ';
                out += c;
                ++i;
            } else if (c >= 'a' && c <= 'z') {
                out += c;
                ++i;
            } else if (c == '+') {
                out += " plus ";
                ++i;
            } else if (c == '-' && i + 1 < raw.size() && raw[i + 1] == '>') {
                out += " yields ";
                i += 2;
            } else if (c == '=') {
                out += " yields ";
                ++i;
            } else {
                out += ' ';
                ++i;
            }
        }
        return make_text(K::Word, out);
    }

    Node environment() {
        std::string_view name = raw_group();
        std::string env(name);
        if (env == "array" || env == "subarray" || env == "alignat" || env == "alignat*" || env == "alignedat")
            skip_group();
        return table_body(env, false);
    }

    // Rows (\\) of cells (&) up to \end{env} (or the end of the input at the top level).
    Node table_body(const std::string& env, bool top) {
        Node t = make(K::Table);
        t.t = env.empty() ? "lines" : env;
        Node cur = make(K::Row);
        while (true) {
            Node cell = row(Stop::Cell);
            cur.kids.push_back(unwrap(std::move(cell)));
            skip_ws();
            if (at_end()) break;
            char c = peek();
            if (c == '&') {
                ++p_;
                continue;
            }
            if (c == '}') {
                if (top) {
                    ++p_;
                    continue;
                }
                break;  // a missing \end inside a group
            }
            std::string_view cmd = peek_cmd();
            if (cmd == "\\" || cmd == "cr" || cmd == "newline") {
                read_cmd();
                optional_arg(nullptr);
                t.kids.push_back(std::move(cur));
                cur = make(K::Row);
                continue;
            }
            if (cmd == "end") {
                read_cmd();
                skip_group();
                if (!top) break;
                continue;
            }
            break;
        }
        t.kids.push_back(std::move(cur));
        // drop empty rows (a trailing \\)
        std::vector<Node> rows;
        for (Node& r : t.kids) {
            bool empty = true;
            for (const Node& c : r.kids) empty &= c.k == K::Empty || (c.k == K::Row && c.kids.empty());
            if (!empty) rows.push_back(std::move(r));
        }
        t.kids = std::move(rows);
        if (top && t.kids.empty()) t.kids.push_back(make(K::Row));
        return t;
    }

    Node command() {
        std::string_view c = read_cmd();
        if (c.empty()) return make(K::Empty);
        if (uint32_t g = greek_tex(c)) return make_letter(g);
        if (c.size() == 1 && !is_alpha(c[0])) {
            switch (c[0]) {
                case ',': case ':': case ';': case '!': case ' ': case '>': return make(K::Space);
                case '{': return make_text(K::Open, "{");
                case '}': return make_text(K::Close, "{");
                case '|': return make_text(K::Bar, "\xE2\x80\x96");
                case '%': return make_text(K::Word, "percent");
                case '&': return make_text(K::Word, "and");
                case '#': return make_text(K::Word, "number");
                case '$': return make_text(K::Word, "dollars");
                default: return make(K::Empty);
            }
        }
        if (c == "frac" || c == "dfrac" || c == "tfrac" || c == "cfrac" || c == "binom" || c == "dbinom" ||
            c == "tbinom") {
            if (c == "cfrac") optional_arg(nullptr);
            Node f = make(K::Frac);
            f.n = c[c.size() - 1] == 'm';  // binom
            f.kids.push_back(arg());
            f.kids.push_back(arg());
            return f;
        }
        if (c == "sqrt") {
            Node idx = make(K::Empty);
            optional_arg(&idx);
            Node r = make(K::Root);
            r.kids.push_back(arg());
            r.kids.push_back(std::move(idx));
            return r;
        }
        if (c == "left") {
            Delim open = delimiter();
            Node body = row(Stop::Right);
            Delim close{"", Side::Either};
            if (peek_cmd() == "right") {
                read_cmd();
                close = delimiter();
            }
            if (open.id.empty() && close.id.empty()) return unwrap(std::move(body));
            Node f = make(K::Fence);
            f.t = open.id;
            f.u = close.id;
            f.kids.push_back(std::move(body));
            return f;
        }
        if (c == "big" || c == "Big" || c == "bigg" || c == "Bigg" || c == "bigl" || c == "bigr" || c == "Bigl" ||
            c == "Bigr" || c == "biggl" || c == "biggr" || c == "Biggl" || c == "Biggr" || c == "bigm" ||
            c == "Bigm" || c == "biggm" || c == "Biggm" || c == "middle") {
            Node d = delimiter_atom();
            if (c == "middle" && d.k == K::Bar) d.t = "mid";
            return d;
        }
        if (c == "begin") return environment();
        if (c == "text" || c == "textrm" || c == "textit" || c == "textbf" || c == "textsf" || c == "texttt" ||
            c == "textnormal" || c == "mbox" || c == "hbox" || c == "textup" || c == "emph" || c == "textmd" ||
            c == "text*")
            return text(raw_group());
        if (c == "mathrm" || c == "operatorname" || c == "operatorname*" || c == "mathit" || c == "mathsf" ||
            c == "mathtt" || c == "mathop" || c == "mathnormal" || c == "textsc" || c == "mathsfit")
            return upright(raw_group());
        if (c == "mathbf" || c == "boldsymbol" || c == "bm" || c == "mathcal" || c == "mathscr" || c == "mathfrak" ||
            c == "pmb" || c == "mathbfit" || c == "symbf" || c == "symit" || c == "symcal" || c == "mathbold" ||
            c == "boxed" || c == "fbox" || c == "cancel" || c == "bcancel" || c == "xcancel" || c == "sout" ||
            c == "smash" || c == "mathord" || c == "mathbin" || c == "mathrel" || c == "mathopen" ||
            c == "mathclose" || c == "mathpunct" || c == "mathinner" || c == "displaylines" || c == "ensuremath")
            return arg();
        if (c == "mathbb" || c == "Bbb" || c == "mathds" || c == "symbb" || c == "mathbbm") {
            std::string_view g = raw_group();
            std::string w;
            for (char ch : g)
                if (is_alpha(ch) || is_digit(ch)) w += ch;
            if (w.size() == 1) {
                char l = w[0];
                if (l == 'R' || l == 'N' || l == 'Z' || l == 'Q' || l == 'C') {
                    Node n = make(K::Set);
                    n.cp = static_cast<unsigned char>(l);
                    return n;
                }
                if (l == 'E') return make_sym(func_word("E"));
                if (l == '1') return make_text(K::Word, "the indicator");
                return make_letter(static_cast<unsigned char>(l));
            }
            return unwrap(Parser(g).run());
        }
        if (c == "hat" || c == "widehat" || c == "bar" || c == "overline" || c == "vec" || c == "overrightarrow" ||
            c == "tilde" || c == "widetilde" || c == "dot" || c == "ddot" || c == "dddot" || c == "check" ||
            c == "widecheck" || c == "breve" || c == "acute" || c == "grave" || c == "mathring" ||
            c == "underline" || c == "overleftarrow" || c == "overleftrightarrow" || c == "utilde" ||
            c == "underbar" || c == "underbrace" || c == "overbrace" || c == "underleftarrow" ||
            c == "underrightarrow" || c == "overgroup" || c == "undergroup") {
            Node a = make(K::Accent);
            if (c == "widehat") a.t = "hat";
            else if (c == "overline") a.t = "bar";
            else if (c == "overrightarrow" || c == "overleftrightarrow") a.t = "vec";
            else if (c == "widetilde") a.t = "tilde";
            else if (c == "widecheck") a.t = "check";
            else if (c == "underbrace" || c == "overbrace" || c == "underline" || c == "underbar" || c == "utilde" ||
                     c == "acute" || c == "grave" || c == "overleftarrow" || c == "underleftarrow" ||
                     c == "underrightarrow" || c == "overgroup" || c == "undergroup")
                a.t = c == "underbrace" || c == "overbrace" ? "brace" : "plain";
            else a.t = std::string(c);
            a.kids.push_back(arg());
            return a;
        }
        if (c == "overset" || c == "stackrel" || c == "underset") {
            Node top = arg();
            Node base = arg();
            if (c == "underset") return scripts(std::move(base), std::move(top), make(K::Empty), 0);
            // =, defined: "is defined as"
            if (base.k == K::Sym && base.s && base.s->kind == R_EQ &&
                ((top.k == K::Word && (top.t == "def" || top.t == "!" || top.t == "Def")) ||
                 (top.k == K::Sym && top.s && top.s->cp == 0x25B3) || (top.k == K::Row && !top.kids.empty())))
                return make_sym(sym_tex("coloneqq"));
            return base;
        }
        if (c == "color") {
            skip_group();
            return make(K::Empty);
        }
        if (c == "textcolor" || c == "colorbox") {
            skip_group();
            return arg();
        }
        if (c == "fcolorbox") {
            skip_group();
            skip_group();
            return arg();
        }
        if (c == "phantom" || c == "hphantom" || c == "vphantom" || c == "tag" || c == "tag*" || c == "label" ||
            c == "leftroot" || c == "uproot" ||
            c == "eqref" || c == "ref" || c == "hspace" || c == "hspace*" || c == "vspace" || c == "kern" ||
            c == "mkern" || c == "mskip" || c == "hskip" || c == "mspace" || c == "unicode" || c == "href" ||
            c == "class" || c == "cssId" || c == "style" || c == "htmlClass" || c == "htmlId" ||
            c == "htmlStyle" || c == "raisebox" || c == "rule") {
            if (c == "href" || c == "class" || c == "cssId" || c == "style" || c == "htmlClass" || c == "htmlId" ||
                c == "htmlStyle" || c == "raisebox") {
                skip_group();
                return arg();
            }
            if (c == "rule") {
                optional_arg(nullptr);
                skip_group();
            }
            if (c == "kern" || c == "mkern" || c == "mskip" || c == "hskip") {  // a dimension without braces
                skip_ws();
                if (peek() == '{') skip_group();
                else
                    while (!at_end() && (is_digit(peek()) || is_alpha(peek()) || peek() == '.' || peek() == '-')) ++p_;
                return make(K::Empty);
            }
            skip_group();
            return make(K::Empty);
        }
        if (c == "quad" || c == "qquad") {
            Node s = make(K::Space);
            s.n = 1;
            return s;
        }
        if (c == "enspace" || c == "thinspace" || c == "medspace" || c == "thickspace" || c == "negthinspace" ||
            c == "negmedspace" || c == "negthickspace" || c == "enskip" || c == "space" || c == "nobreakspace" ||
            c == "displaystyle" || c == "textstyle" || c == "scriptstyle" || c == "scriptscriptstyle" ||
            c == "limits" || c == "nolimits" || c == "mathstrut" || c == "strut" || c == "nonumber" ||
            c == "notag" || c == "allowbreak" || c == "nobreak" || c == "relax" || c == "bf" || c == "rm" ||
            c == "it" || c == "cal" || c == "sf" || c == "tt" || c == "normalsize" || c == "small" ||
            c == "large" || c == "Large" || c == "huge" || c == "Huge" || c == "scriptsize" || c == "tiny" ||
            c == "footnotesize" || c == "medskip" || c == "smallskip" || c == "bigskip")
            return make(K::Space);
        if (c == "not") {
            Node next = atom();
            if (next.k == K::Sym && next.s && next.s->cat == Cat::Rel) next.n = 1;
            return next;
        }
        if (c == "pmod" || c == "pod") {
            Node r = make(K::Row);
            r.kids.push_back(make_sym(sym_tex("mod")));
            r.kids.push_back(arg());
            Node f = make(K::Fence);
            f.t = "(";
            f.u = "(";
            f.kids.push_back(std::move(r));
            return f;
        }
        if (c == "ell") return make_letter('l');
        if (c == "imath") return make_letter('i');
        if (c == "jmath") return make_letter('j');
        if (c == "lbrace") return make_text(K::Open, "{");
        if (c == "rbrace") return make_text(K::Close, "{");
        if (c == "langle") return make_text(K::Open, "<");
        if (c == "rangle") return make_text(K::Close, "<");
        if (c == "lfloor") return make_text(K::Open, "floor");
        if (c == "rfloor") return make_text(K::Close, "floor");
        if (c == "lceil") return make_text(K::Open, "ceil");
        if (c == "rceil") return make_text(K::Close, "ceil");
        if (c == "lbrack") return make_text(K::Open, "[");
        if (c == "rbrack") return make_text(K::Close, "[");
        if (c == "lvert") return make_text(K::Open, "|");
        if (c == "rvert") return make_text(K::Close, "|");
        if (c == "lVert") return make_text(K::Open, "\xE2\x80\x96");
        if (c == "rVert") return make_text(K::Close, "\xE2\x80\x96");
        if (c == "vert") return make_text(K::Bar, "|");
        if (c == "Vert") return make_text(K::Bar, "\xE2\x80\x96");
        if (c == "mid") return make_text(K::Bar, "mid");
        if (c == "ce") return chemistry(raw_group());
        if (c == "xrightarrow" || c == "xleftarrow" || c == "xRightarrow" || c == "xLeftarrow" ||
            c == "xleftrightarrow" || c == "xLeftrightarrow" || c == "xmapsto") {  // the label goes
            optional_arg(nullptr);
            skip_group();
            std::string_view base = c == "xrightarrow" ? "to" : c == "xleftarrow" ? "gets" : c == "xRightarrow" ? "Rightarrow"
                                    : c == "xLeftarrow" ? "Leftarrow" : c == "xmapsto" ? "mapsto" : "iff";
            return make_sym(sym_tex(base));
        }
        if (c == "substack") {
            Node a = arg();  // its \\ became wide spaces: they read as "and"
            return a;
        }
        if (const Sym* s = sym_tex(c)) return make_sym(s);
        // unknown: read its arguments, drop the name
        Node r = make(K::Row);
        while (true) {
            skip_ws();
            if (peek() != '{') break;
            ++p_;
            Node g = row(Stop::Brace);
            if (peek() == '}') ++p_;
            for (Node& k : g.kids) r.kids.push_back(std::move(k));
        }
        if (r.kids.empty()) return make(K::Empty);
        return unwrap(std::move(r));
    }

    // Pairs the delimiters of a row into fences: ( ) [ ] { } < > floor ceil, and | and ‖ by position (a bar
    // after an operand closes an open bar, elsewhere it opens one). What stays unpaired is dropped; a bar that
    // can neither open nor close stands alone (such that, given, divides).
    static bool operand_end(const Node& n) {
        switch (n.k) {
            case K::Sym:
                return n.s && (n.s->cat == Cat::Const || n.s->cat == Cat::Dots || n.s->cat == Cat::Prime ||
                               (n.s->cat == Cat::Punct && n.s->say == "factorial"));
            case K::Open: case K::Bar: case K::Space: case K::Empty: return false;
            default: return true;
        }
    }

    static void fences(Node& r) {
        bool any = false;
        for (const Node& k : r.kids) any |= k.k == K::Open || k.k == K::Close || k.k == K::Bar;
        if (!any) return;
        std::vector<Node> out;
        std::vector<size_t> stack;  // positions in out of open delimiters (Open, or Bar used as an opener)
        for (size_t ix = 0; ix < r.kids.size(); ++ix) {
            Node& item = r.kids[ix];
            // (a+b)^2, \|x\|_2: the scripts of a closing delimiter belong to the whole fence
            bool wrapped = item.k == K::Scripts && (item.kids[0].k == K::Close || item.kids[0].k == K::Bar);
            Node& k = wrapped ? item.kids[0] : item;
            if ((k.k == K::Open || k.k == K::Bar) && stack.size() >= MAX_DEPTH) continue;  // absurd nesting
            if (k.k == K::Open) {
                stack.push_back(out.size());
                out.push_back(std::move(k));
                continue;
            }
            bool bar = k.k == K::Bar && k.t != "mid";
            bool closing = k.k == K::Close;
            if (bar) {
                bool after_operand = false;
                for (size_t q = out.size(); q-- > 0;) {
                    if (out[q].k == K::Space) continue;
                    after_operand = operand_end(out[q]);
                    break;
                }
                bool open_bar_on_top = !stack.empty() && out[stack.back()].k == K::Bar && out[stack.back()].t == k.t;
                // |a| |b|: a bar after an operand opens a new pair when another bar of its kind follows
                bool another = false;
                for (size_t q = ix + 1; q < r.kids.size() && !another; ++q) {
                    const Node& z = r.kids[q].k == K::Scripts ? r.kids[q].kids[0] : r.kids[q];
                    another = z.k == K::Bar && z.t == k.t;
                }
                if (after_operand && open_bar_on_top) closing = true;
                else if (!after_operand || another) {
                    stack.push_back(out.size());
                    out.push_back(std::move(k));
                    continue;
                } else {
                    k.t = "mid";
                    out.push_back(std::move(k));
                    continue;
                }
            }
            if (closing) {
                // a closing bracket pairs with the nearest open bracket; open bars on the way stand alone
                if (!bar) {
                    size_t q = stack.size();
                    while (q > 0 && out[stack[q - 1]].k != K::Open) --q;
                    if (q == 0) continue;  // unpaired: dropped
                    for (size_t m = q; m < stack.size(); ++m) out[stack[m]].t = "mid";
                    stack.resize(q);
                }
                size_t at = stack.back();
                stack.pop_back();
                Node f = make(K::Fence);
                f.t = out[at].t;
                f.u = k.t;
                Node body = make(K::Row);
                for (size_t q = at + 1; q < out.size(); ++q) body.kids.push_back(std::move(out[q]));
                out.resize(at);
                f.kids.push_back(std::move(body));
                if (wrapped) {
                    item.kids[0] = std::move(f);
                    out.push_back(std::move(item));
                } else {
                    out.push_back(std::move(f));
                }
                continue;
            }
            out.push_back(std::move(k));
        }
        // unpaired openers: a bar stands alone, other delimiters go
        std::vector<Node> clean;
        for (Node& k : out) {
            if (k.k == K::Open) continue;
            if (k.k == K::Bar) k.t = "mid";
            clean.push_back(std::move(k));
        }
        r.kids = std::move(clean);
    }
};

}  // namespace

Node parse(std::string_view tex) { return Parser(tex).run(); }

}  // namespace mth
