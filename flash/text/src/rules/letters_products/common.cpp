// Readers shared by the rules of the module "letters_products".
#include <cstdlib>

#include "rules/letters_products/lp.h"

namespace lp {

namespace {

bool vowel(char c) { return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'; }

bool onset(std::string_view s) {
    static const rv::WordSet SET{
        "",   "b",  "bl",  "br",  "c",   "ch", "chr", "cl",  "cr",  "d",  "dr", "dw", "f",  "fl", "fr", "g",
        "gh", "gl", "gn",  "gr",  "h",   "j",  "k",   "kl",  "kn",  "kr", "l",  "m",  "n",  "p",  "ph", "phr",
        "pl", "pr", "ps",  "q",   "qu",  "r",  "rh",  "s",   "sc",  "sch", "scr", "sh", "shr", "sk", "sl",
        "sm", "sn", "sp",  "sph", "spl", "spr", "squ", "st",  "str", "sw", "t",  "th", "thr", "tr", "tw", "v",
        "w",  "wh", "wr",  "x",   "y",   "z"};
    return SET.contains(s);
}

bool coda(std::string_view s) {
    static const rv::WordSet SET{
        "",    "b",   "bs",  "c",   "ch",  "ck",  "cks", "ct",  "d",   "ds",  "f",   "ff",  "ft",  "g",   "gh",
        "ght", "gn",  "gs",  "h",   "hn",  "k",   "ks",  "l",   "lch", "ld",  "lds", "lf",  "lk",  "ll",  "lm",  "lp",
        "ls",  "lt",  "lth", "lts", "m",   "mb",  "mn",  "mp",  "mps", "mpts", "ms", "n",   "nc",  "nce", "nch",
        "nd",  "nds", "ng",  "ngs", "ngth", "nk", "nks", "ns",  "nt",  "nth", "nts", "p",   "ps",  "pt",  "pth",
        "pts", "r",   "rb",  "rc",  "rch", "rd",  "rds", "rf",  "rg",  "rk",  "rks", "rl",  "rm",  "rms", "rn",
        "rns", "rp",  "rs",  "rst", "rt",  "rth", "rts", "s",   "sh",  "sk",  "sm",  "sp",  "ss",  "st",  "sts",
        "t",   "tch", "th",  "ts",  "tz",  "v",   "w",   "wd",  "wl",  "wn",  "ws",  "x",   "xt",  "ys",  "z",
        "zz"};
    return SET.contains(s);
}

}  // namespace

bool pronounceable(std::string_view w) {
    if (w.empty() || w.size() > 24) return false;
    for (char c : w)
        if (c < 'a' || c > 'z') return false;
    // "USDA", "USPS": "US" and a consonant start an initialism; "AARP", "IIRC": a doubled vowel at the start
    if (w.size() >= 3 && w[0] == 'u' && w[1] == 's' && !vowel(w[2])) return false;
    if (w.size() >= 2 && w[0] == w[1] && (w[0] == 'a' || w[0] == 'i' || w[0] == 'u')) return false;
    bool any_vowel = false;
    size_t i = 0, groups = 0;
    while (i < w.size()) {
        auto is_v = [&](size_t k) { return vowel(w[k]) || (w[k] == 'y' && k > 0); };
        bool v = is_v(i);
        size_t j = i + 1;
        while (j < w.size() && is_v(j) == v) ++j;
        std::string_view g = w.substr(i, j - i);
        if (v) {
            any_vowel = true;
            if (g.size() > 3 || (g.size() == 3 && g[0] == g[1] && g[1] == g[2])) return false;
        } else if (groups == 0) {
            if (!onset(g)) return false;
        } else if (j == w.size()) {
            if (!coda(g) && !(g.back() == 's' && coda(g.substr(0, g.size() - 1)))) return false;
        } else {
            bool split = false;
            for (size_t c = 0; c <= g.size() && !split; ++c) split = coda(g.substr(0, c)) && onset(g.substr(c));
            if (!split) return false;
        }
        ++groups;
        i = j;
    }
    return any_vowel;
}

namespace {

// Letter triples of the english() words with the word edges (27 symbols: a-z and the edge), one bit each.
class Triples {
public:
    Triples() {
        for (std::string_view w : english_words().words()) {
            bool letters = !w.empty();
            for (char c : w) letters &= c >= 'a' && c <= 'z';
            if (!letters) continue;
            for (size_t k = 0; k < w.size(); ++k) set(at(w, k - 1), at(w, k), at(w, k + 1));
        }
    }
    bool all(std::string_view w) const {
        for (size_t k = 0; k < w.size(); ++k)
            if (!get(at(w, k - 1), at(w, k), at(w, k + 1))) return false;
        return true;
    }

private:
    // the letter at k, or 26 for the edge before the first and after the last letter (k wraps below zero)
    static unsigned at(std::string_view w, size_t k) { return k < w.size() ? static_cast<unsigned>(w[k] - 'a') : 26; }
    void set(unsigned a, unsigned b, unsigned c) {
        unsigned i = (a * 27 + b) * 27 + c;
        bits_[i / 8] = static_cast<uint8_t>(bits_[i / 8] | (1u << (i % 8)));
    }
    bool get(unsigned a, unsigned b, unsigned c) const {
        unsigned i = (a * 27 + b) * 27 + c;
        return bits_[i / 8] & (1u << (i % 8));
    }
    uint8_t bits_[27 * 27 * 27 / 8 + 1] = {};
};

}  // namespace

bool english_letters(std::string_view w) {
    static const Triples T;
    for (char c : w)
        if (c < 'a' || c > 'z') return false;
    return !w.empty() && T.all(w);
}

std::string two_words(const std::string& w) {
    if (english(w)) return "";
    for (size_t c = 2; c + 2 <= w.size(); ++c)
        if (english(w.substr(0, c)) && english(w.substr(c)) && (c >= 3 || w.size() - c >= 3))
            return w.substr(0, c) + " " + w.substr(c);
    return "";
}

bool word_like(std::string_view w) {
    if (w.size() < 4) return false;
    std::string low = lower(w);
    if (w.size() < 6) return name(low);
    // random letters rarely form eight or more pronounceable letters; long words keep the syllable test alone
    return pronounceable(low) && (w.size() >= 8 || english_letters(low));
}

std::string upper(std::string_view w) {
    std::string s(w);
    for (char& c : s)
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
    return s;
}

std::string lower(std::string_view w) {
    std::string s(w);
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}

std::string title(std::string_view w) { return rv::capital(lower(w)); }

bool is_upper_word(std::string_view w) {
    if (w.empty()) return false;
    for (char c : w)
        if (c < 'A' || c > 'Z') return false;
    return true;
}

std::string model_number(std::string_view d) {
    if (d.empty() || d.size() > 9) return d.empty() ? "" : rv::digits(d);
    if (d.size() > 1 && d[0] == '0') return rv::digits(d);
    uint32_t v = static_cast<uint32_t>(std::strtoul(std::string(d).c_str(), nullptr, 10));
    if (d.size() <= 2) return rv::cardinal(v);
    if (d.size() <= 4) return rv::pairs(v);
    if (d.size() == 5) {
        uint32_t hi = v / 1000, lo = v % 1000;
        if (lo == 0) return rv::cardinal(v);
        // "17025" -> seventeen zero twenty-five, "17005" -> seventeen zero zero five
        if (lo < 10) return rv::cardinal(hi) + " zero zero " + rv::cardinal(lo);
        if (lo < 100) return rv::cardinal(hi) + " zero " + rv::cardinal(lo);
        return rv::cardinal(hi) + " " + rv::pairs(lo);
    }
    return rv::digits(d);
}

std::string standard_number(std::string_view d) {
    if (d.size() < 4 || d.size() > 5 || d[0] == '0') return model_number(d);
    uint32_t v = static_cast<uint32_t>(std::strtoul(std::string(d).c_str(), nullptr, 10));
    if (v % 1000 <= 1) return rv::cardinal(v);
    return model_number(d);
}

std::string version_part(std::string_view d) {
    if (d.size() > 1 && d[0] == '0') return rv::digits(d);
    return rv::cardinal_digits(d);
}

std::string version_words(std::string_view w) {
    std::string out;
    size_t p = 0;
    while (p <= w.size()) {
        size_t q = w.find('.', p);
        if (q == std::string_view::npos) q = w.size();
        std::string_view digits = w.substr(p, q - p);
        // a year as the first part is read in pairs: "Unity 2022.3" -> twenty twenty-two point three
        bool year = p == 0 && digits.size() == 4 && (digits.substr(0, 2) == "19" || digits.substr(0, 2) == "20");
        std::string part = year ? rv::pairs(static_cast<uint32_t>(std::strtoul(std::string(digits).c_str(), nullptr, 10)))
                                : version_part(digits);
        if (part.empty()) return "";
        if (!out.empty()) out += " point ";
        out += part;
        p = q + 1;
    }
    return out;
}

std::string caps_word(std::string_view w) {
    if (const auto* r = respelling(w)) return std::string(*r);
    if (log_level(w)) return title(w);
    if (!keep_caps(w) && word_like(w)) return title(w);
    return std::string(w);
}

namespace {

// Segments of a mixed-case word: a Title part ("Git"), capitals ("AI"), lower case ("mac").
struct Seg {
    std::string_view s;
    char kind;  // 'T' title, 'U' capitals, 'L' lower case, 'W' an English word in lower case
};

bool up(char c) { return c >= 'A' && c <= 'Z'; }
bool low(char c) { return c >= 'a' && c <= 'z'; }

// Capitals inside a name: the respelling, a pronounceable word of four or more letters with one capital, or
// the letters.
std::string caps_part(std::string_view s, bool after_prefix) {
    if (const auto* r = respelling(s)) return std::string(*r);
    size_t min = after_prefix ? 3 : 4;
    if (s.size() >= min && !keep_caps(s) && pronounceable(lower(s))) return title(s);
    return std::string(s);
}

}  // namespace

std::string mixed_word(std::string_view w) {
    if (const auto* r = respelling(w)) return std::string(*r);
    if (w.size() < 4 || w.size() > 40) return "";
    for (char c : w)
        if (!up(c) && !low(c)) return "";
    if (w == "XBoxSeriesX") return "Xbox Series X";
    // capitals with an English ending stay as written: "LOLing", "PDFed"; one lower-case letter after
    // capitals is spelled: "PCIe" -> PCI E (like "NVMe")
    {
        size_t k = 0;
        while (k < w.size() && up(w[k])) ++k;
        std::string_view end = w.substr(k);
        if (k >= 2 && (end == "ing" || end == "ed" || end == "er" || end == "ers" || end == "es" || end == "ish"))
            return "";
        if (k >= 3 && end.size() == 1 && end != "s" && end != "v") return std::string(w.substr(0, k)) + " " + upper(end);
    }
    if ((w.substr(0, 2) == "Mc" || w.substr(0, 3) == "Mac") && w.size() > 3) return "";
    Seg segs[24];
    size_t n = 0, i = 0;
    while (i < w.size()) {
        if (n == 24) return "";
        size_t j = i;
        if (low(w[i])) {
            while (j < w.size() && low(w[j])) ++j;
            segs[n++] = {w.substr(i, j - i), 'L'};
        } else {
            while (j < w.size() && up(w[j])) ++j;
            size_t caps = j - i;
            if (j < w.size() && caps >= 2) {
                size_t k = j;
                while (k < w.size() && low(w[k])) ++k;
                std::string tail = lower(w.substr(j, k - j));
                bool suffix = tail == "ing" || tail == "ers" || tail == "est" || tail == "ish" || tail == "ies";
                if (tail.size() >= 3 && !suffix && english(tail) && !english(lower(w.substr(j - 1, k - j + 1)))) {
                    segs[n++] = {w.substr(i, caps), 'U'};  // "OCRengine": OCR, engine
                    if (n == 24) return "";
                    segs[n++] = {w.substr(j, k - j), 'W'};
                    i = k;
                    continue;
                }
            }
            if (j < w.size() && caps >= 2 && !(j + 1 == w.size() && (w[j] == 'v' || w[j] == 's'))) {
                // "AIModel": the last capital starts the next part
                segs[n++] = {w.substr(i, caps - 1), 'U'};
                i = j - 1;
                continue;
            }
            if (j < w.size() && caps == 1) {
                size_t k = j;
                while (k < w.size() && low(w[k])) ++k;
                segs[n++] = {w.substr(i, k - i), 'T'};
                j = k;
            } else {
                segs[n++] = {w.substr(i, caps), 'U'};
            }
        }
        i = j;
    }
    std::string out;
    size_t k = 0;
    auto add = [&](const std::string& s) {
        if (!out.empty()) out += ' ';
        out += s;
    };
    // "LOLing", "PDFed": capitals with an English ending stay; "ETag": one capital and a word stays
    if (n == 2 && segs[0].kind == 'U' && segs[1].kind == 'L') return "";
    if (segs[0].kind == 'U' && segs[0].s.size() == 1 && n >= 2 && segs[1].kind == 'T') return "";
    if (segs[0].kind == 'L') {
        std::string_view p = segs[0].s;
        if (n < 2) return "";
        if (p.size() == 1) {
            if (segs[1].kind == 'T' && p == "i") {
                std::string word = "I";
                for (k = 1; k < n && segs[k].kind == 'T'; ++k) word += lower(segs[k].s);
                add(word);
            } else {
                add(upper(p));
                k = 1;
                if (segs[1].kind == 'U') {
                    add(caps_part(segs[1].s, true));
                    k = 2;
                }
            }
        } else if (segs[1].kind == 'U' && segs[1].s == "OS" && n == 2) {
            add(pronounceable(p) ? title(p) : upper(p));
            add("OS");
            k = 2;
        } else {
            return "";
        }
    }
    while (k < n) {
        const Seg& s = segs[k];
        if (s.kind == 'T') {
            std::string word(s.s);
            for (++k; k < n && segs[k].kind == 'T'; ++k) word += lower(segs[k].s);
            add(title(word));
        } else if (s.kind == 'U') {
            add(caps_part(s.s, false));
            ++k;
        } else if (s.kind == 'W') {
            add(std::string(s.s));
            ++k;
        } else {
            if (s.s == "s" && segs[k - 1].kind == 'U') out += 's';  // "NeuroGPUs"
            else add(s.s.size() == 1 ? upper(s.s) : std::string(s.s));
            ++k;
        }
    }
    return out == w ? "" : out;
}

std::string name_word(std::string_view w) {
    if (const auto* r = respelling(w)) return std::string(*r);
    bool has_upper = false, has_lower = false;
    for (char c : w) {
        has_upper |= up(c);
        has_lower |= low(c);
    }
    if (has_upper && !has_lower) return caps_word(w);
    if (!has_upper) {
        if (w.size() == 1) return upper(w);
        if (w == "amd" || english(w) || (w.size() >= 3 && pronounceable(w))) return std::string(w);
        return upper(w);
    }
    if (up(w[0])) {
        bool rest_lower = true;
        for (size_t k = 1; k < w.size(); ++k) rest_lower &= low(w[k]);
        if (rest_lower) return std::string(w);
    }
    std::string m = mixed_word(w);
    if (!m.empty()) return m;
    // "TontaubeV", "IPv", "YOLOv": the name, then the letter V
    char last = w.back();
    if ((last == 'v' || last == 'V') && w.size() >= 3) {
        std::string head = name_word(w.substr(0, w.size() - 1));
        return head + " V";
    }
    return std::string(w);
}

bool code_context(const Doc& d, size_t i) {
    auto punct_cp = [&](size_t k) -> uint32_t { return d[k].kind == rv::Kind::Punct ? d[k].cp : 0; };
    if (d.joined(i)) {
        uint32_t p = punct_cp(i - 1);
        if (p == '_' || p == '$' || p == '@' || p == '#' || p == '\\' || p == '~' || p == '`' || p == '=' ||
            p == '<' || p == '%')
            return true;
        if (p == '.' && i >= 2 && d.joined(i - 1) && d[i - 2].kind != rv::Kind::Punct) return true;
        if (p == '/' && !(i >= 2 && d.joined(i - 1) && d.word(i - 2) && d[i - 2].casing == rv::UPPER)) return true;
    }
    if (i + 1 < d.size() && d.joined(i + 1)) {
        uint32_t p = punct_cp(i + 1);
        if (p == '_' || p == '(' || p == '\\' || p == '=' || p == '`' || p == '>' || p == '[') return true;
        if (p == '.' && d.joined(i + 2) && d.word(i + 2)) return true;
        if (p == '/' && !(d.joined(i + 2) && d.word(i + 2) && d[i + 2].casing == rv::UPPER)) return true;
    }
    return false;
}

}  // namespace lp
