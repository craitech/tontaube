#include "core/readers.h"

#include "core/utf8.h"

namespace rv {

namespace {

const char* const ONES[] = {"zero",    "one",     "two",       "three",    "four",     "five",    "six",
                            "seven",   "eight",   "nine",      "ten",      "eleven",   "twelve",  "thirteen",
                            "fourteen", "fifteen", "sixteen",  "seventeen", "eighteen", "nineteen"};
const char* const TENS[] = {"", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char* const SCALES[] = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
                              "sextillion", "septillion"};

void below_thousand(unsigned n, std::string& out) {
    if (n >= 100) {
        out += ONES[n / 100];
        out += " hundred";
        n %= 100;
        if (n) out += ' ';
    }
    if (n >= 20) {
        out += TENS[n / 10];
        if (n % 10) {
            out += '-';
            out += ONES[n % 10];
        }
    } else if (n > 0) {
        out += ONES[n];
    }
}

}  // namespace

std::string cardinal_digits(std::string_view d) {
    if (d.empty() || d.size() > 24) return "";
    for (char c : d)
        if (!is_ascii_digit(c)) return "";
    size_t nz = d.find_first_not_of('0');
    if (nz == std::string_view::npos) return "zero";
    d.remove_prefix(nz);
    std::string out;
    size_t groups = (d.size() + 2) / 3;
    size_t pos = 0, first = d.size() - (groups - 1) * 3;
    for (size_t g = 0; g < groups; ++g) {
        size_t len = g == 0 ? first : 3;
        unsigned v = 0;
        for (size_t k = 0; k < len; ++k) v = v * 10 + static_cast<unsigned>(d[pos + k] - '0');
        pos += len;
        if (v == 0) continue;
        if (!out.empty()) out += ' ';
        below_thousand(v, out);
        size_t scale = groups - 1 - g;
        if (scale) {
            out += ' ';
            out += SCALES[scale];
        }
    }
    return out;
}

std::string cardinal(uint64_t n) { return cardinal_digits(std::to_string(n)); }

std::string ordinal_of(std::string w) {
    size_t cut = w.find_last_of(" -");
    size_t at = cut == std::string::npos ? 0 : cut + 1;
    std::string last = w.substr(at);
    static const struct { const char* from; const char* to; } IRREGULAR[] = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
        {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}};
    for (const auto& r : IRREGULAR)
        if (last == r.from) return w.substr(0, at) + r.to;
    if (!last.empty() && last.back() == 'y') return w.substr(0, w.size() - 1) + "ieth";
    return w + "th";
}

std::string ordinal(uint64_t n) { return ordinal_of(cardinal(n)); }

std::string digits(std::string_view s) {
    std::string out;
    for (char c : s) {
        if (!is_ascii_digit(c)) continue;
        if (!out.empty()) out += ' ';
        out += ONES[c - '0'];
    }
    return out;
}

std::string pairs(uint32_t n) {
    uint32_t hi = n / 100, lo = n % 100;
    if (n < 100) return cardinal(n);
    if (lo == 0) return hi % 10 ? cardinal(hi) + " hundred" : cardinal(n);
    if (n >= 2000 && n <= 2009) return std::string("two thousand ") + ONES[lo];
    return cardinal(hi) + " " + (lo < 10 ? std::string("oh ") + ONES[lo] : cardinal(lo));
}

std::string read_number(std::string_view w) {
    std::string whole, frac;
    size_t i = 0, run = 0;
    bool grouped = false;
    for (; i < w.size() && w[i] != '.'; ++i) {
        char c = w[i];
        if (is_ascii_digit(c)) {
            whole += c;
            ++run;
        } else if (c == ',') {
            if ((grouped ? run != 3 : (run == 0 || run > 3))) return "";
            grouped = true;
            run = 0;
        } else {
            return "";
        }
    }
    if (whole.empty() || (grouped && run != 3)) return "";
    if (i < w.size()) {
        frac = std::string(w.substr(i + 1));
        if (frac.empty()) return "";
        for (char c : frac)
            if (!is_ascii_digit(c)) return "";
    }
    if ((whole.size() > 1 && whole[0] == '0') || whole.size() > 24) return "";
    std::string out = cardinal_digits(whole);
    if (!frac.empty()) out += " point " + digits(frac);
    return out;
}

std::string read_number(const NumberSpan& n) {
    if (!n.ok || n.whole.empty() || (n.whole.size() > 1 && n.whole[0] == '0') || n.whole.size() > 24) return "";
    std::string out = cardinal_digits(n.whole);
    if (!n.fraction.empty()) out += " point " + digits(n.fraction);
    return out;
}

int roman_value(std::string_view s) {
    if (s.empty() || s.size() > 15) return 0;
    bool upper = s[0] >= 'A' && s[0] <= 'Z';
    auto value = [](char c) {
        switch (c | 0x20) {
            case 'i': return 1; case 'v': return 5; case 'x': return 10; case 'l': return 50;
            case 'c': return 100; case 'd': return 500; case 'm': return 1000; default: return 0;
        }
    };
    int total = 0;
    for (size_t k = 0; k < s.size(); ++k) {
        char c = s[k];
        if (!value(c) || ((c >= 'A' && c <= 'Z') != upper)) return 0;
        int v = value(c), next = k + 1 < s.size() ? value(s[k + 1]) : 0;
        total += next > v ? -v : v;
    }
    if (total <= 0 || total >= 4000) return 0;
    static const struct { int v; const char* r; } STEPS[] = {
        {1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"}, {100, "C"}, {90, "XC"}, {50, "L"},
        {40, "XL"}, {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"}};
    std::string canon;
    int n = total;
    for (const auto& st : STEPS)
        while (n >= st.v) canon += st.r, n -= st.v;
    if (canon.size() != s.size()) return 0;
    for (size_t k = 0; k < s.size(); ++k)
        if ((s[k] & ~0x20) != canon[k]) return 0;
    return total;
}

std::string capital(std::string s) {
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 32);
    return s;
}

namespace {
const char* const GREEK[] = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                             "iota", "kappa", "lambda", "mu", "nu", "zai", "omicron", "pie",
                             "rho", "sigma", "sigma", "tau", "upsilon", "phi", "kai", "psi", "omega"};
const char* const GREEK_CAPITALS[] = {"Alpha", "Beta", "Gamma", "delta", "Epsilon", "Zeta", "Eta", "Theta",
                                      "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
                                      "Rho", "", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"};
}  // namespace

// The speech model often misreads a lone lower-case letter ("f of x" -> "fear of"), a capital is
// said as its name; m and s need their spelling. Capital and small x therefore sound alike.
std::string variable_name(uint32_t cp) {
    if (cp < 0x80) {
        char c = static_cast<char>(cp);
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
        if (c < 'A' || c > 'Z') return "";
        if (c == 'M') return "emm";
        if (c == 'S') return "ess";
        return std::string(1, c);
    }
    if (cp >= 0x3B1 && cp <= 0x3C9) return GREEK[cp - 0x3B1];
    if (cp >= 0x391 && cp <= 0x3A9) return GREEK_CAPITALS[cp - 0x391];
    switch (cp) {
        case 0xB5: return "mu";
        case 0x3D0: return "beta";
        case 0x3D1: return "theta";
        case 0x3D2: return "Upsilon";
        case 0x3D5: return "phi";
        case 0x3D6: return "pie";
        case 0x3DC: case 0x3DD: return "digamma";
        case 0x3F0: return "kappa";
        case 0x3F1: return "rho";
        case 0x3F4: return "Theta";
        case 0x3F5: return "epsilon";
        default: return "";
    }
}

}  // namespace rv
