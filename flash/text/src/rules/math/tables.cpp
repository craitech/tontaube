// Tables of the module "math": TeX commands and Unicode symbols with their words, Greek letters, function
// names, and the one table of letter spellings.
#include <algorithm>
#include <vector>

#include "core/readers.h"
#include "rules/math/math.h"

namespace mth {

namespace {

// One entry per symbol; aliases have no code point. Wording: ClearSpeak (see README).
const Sym SYMS[] = {
    // operators
    {"", '+', Cat::Op, "plus", O_ADD},
    {"", '-', Cat::Op, "minus", O_ADD},
    {"", 0x2212, Cat::Op, "minus", O_ADD},
    {"pm", 0xB1, Cat::Op, "plus or minus", O_ADD},
    {"mp", 0x2213, Cat::Op, "minus or plus", O_ADD},
    {"times", 0xD7, Cat::Op, "times", 0},
    {"cdot", 0x22C5, Cat::Op, "times", 0},
    {"", 0xB7, Cat::Op, "times", 0},
    {"ast", 0x2217, Cat::Op, "times", C_STAR},
    {"", '*', Cat::Op, "times", C_STAR},
    {"star", 0x22C6, Cat::Op, "star", C_STAR},
    {"div", 0xF7, Cat::Op, "divided by", 0},
    {"", '/', Cat::Op, "over", O_DIV},
    {"", 0x2215, Cat::Op, "over", O_DIV},
    {"circ", 0x2218, Cat::Op, "composed with", C_DEGREE},
    {"bullet", 0x2219, Cat::Op, "times", 0},
    {"otimes", 0x2297, Cat::Op, "tensor", 0},
    {"oplus", 0x2295, Cat::Op, "direct sum", 0},
    {"ominus", 0x2296, Cat::Op, "circled minus", 0},
    {"odot", 0x2299, Cat::Op, "elementwise times", 0},
    {"cup", 0x222A, Cat::Op, "union", 0},
    {"cap", 0x2229, Cat::Op, "intersection", 0},
    {"sqcup", 0x2294, Cat::Op, "union", 0},
    {"sqcap", 0x2293, Cat::Op, "intersection", 0},
    {"uplus", 0x228E, Cat::Op, "union", 0},
    {"setminus", 0x2216, Cat::Op, "minus", 0},
    {"smallsetminus", 0, Cat::Op, "minus", 0},
    {"wedge", 0x2227, Cat::Op, "and", 0},
    {"land", 0, Cat::Op, "and", 0},
    {"vee", 0x2228, Cat::Op, "or", 0},
    {"lor", 0, Cat::Op, "or", 0},
    {"bmod", 0, Cat::Op, "mod", 0},
    {"mod", 0, Cat::Op, "mod", 0},
    {"amalg", 0x2A3F, Cat::Op, "coproduct", 0},
    // relations
    {"", '=', Cat::Rel, "equals", R_EQ},
    {"ne", 0x2260, Cat::Rel, "is not equal to", 0},
    {"neq", 0, Cat::Rel, "is not equal to", 0},
    {"", '<', Cat::Rel, "is less than", 0},
    {"lt", 0, Cat::Rel, "is less than", 0},
    {"", '>', Cat::Rel, "is greater than", 0},
    {"gt", 0, Cat::Rel, "is greater than", 0},
    {"le", 0x2264, Cat::Rel, "is less than or equal to", 0},
    {"leq", 0, Cat::Rel, "is less than or equal to", 0},
    {"leqslant", 0x2A7D, Cat::Rel, "is less than or equal to", 0},
    {"ge", 0x2265, Cat::Rel, "is greater than or equal to", 0},
    {"geq", 0, Cat::Rel, "is greater than or equal to", 0},
    {"geqslant", 0x2A7E, Cat::Rel, "is greater than or equal to", 0},
    {"nless", 0x226E, Cat::Rel, "is not less than", 0},
    {"ngtr", 0x226F, Cat::Rel, "is not greater than", 0},
    {"nleq", 0x2270, Cat::Rel, "is not less than or equal to", 0},
    {"ngeq", 0x2271, Cat::Rel, "is not greater than or equal to", 0},
    {"ll", 0x226A, Cat::Rel, "is much less than", 0},
    {"gg", 0x226B, Cat::Rel, "is much greater than", 0},
    {"approx", 0x2248, Cat::Rel, "is approximately equal to", 0},
    {"simeq", 0x2243, Cat::Rel, "is approximately equal to", 0},
    {"cong", 0x2245, Cat::Rel, "is congruent to", 0},
    {"equiv", 0x2261, Cat::Rel, "is equivalent to", 0},
    {"sim", 0x223C, Cat::Rel, "is similar to", R_SIM},
    {"backsim", 0x223D, Cat::Rel, "is similar to", R_SIM},
    {"", '~', Cat::Rel, "is similar to", R_SIM},
    {"", 0x2DC, Cat::Rel, "is similar to", R_SIM},
    {"", 0x301C, Cat::Rel, "is similar to", R_SIM},
    {"propto", 0x221D, Cat::Rel, "is proportional to", 0},
    {"asymp", 0x224D, Cat::Rel, "is asymptotic to", 0},
    {"lesssim", 0x2272, Cat::Rel, "is less than or approximately equal to", 0},
    {"gtrsim", 0x2273, Cat::Rel, "is greater than or approximately equal to", 0},
    {"prec", 0x227A, Cat::Rel, "precedes", 0},
    {"succ", 0x227B, Cat::Rel, "succeeds", 0},
    {"preceq", 0x2AAF, Cat::Rel, "precedes or equals", 0},
    {"succeq", 0x2AB0, Cat::Rel, "succeeds or equals", 0},
    {"parallel", 0x2225, Cat::Rel, "is parallel to", 0},
    {"perp", 0x22A5, Cat::Rel, "is perpendicular to", 0},
    {"nmid", 0x2224, Cat::Rel, "does not divide", 0},
    {"in", 0x2208, Cat::Rel, "is in", 0},
    {"notin", 0x2209, Cat::Rel, "is not in", 0},
    {"ni", 0x220B, Cat::Rel, "contains", 0},
    {"subset", 0x2282, Cat::Rel, "is a subset of", 0},
    {"subseteq", 0x2286, Cat::Rel, "is a subset of or equal to", 0},
    {"subsetneq", 0x228A, Cat::Rel, "is a proper subset of", 0},
    {"nsubseteq", 0x2288, Cat::Rel, "is not a subset of", 0},
    {"supset", 0x2283, Cat::Rel, "is a superset of", 0},
    {"supseteq", 0x2287, Cat::Rel, "is a superset of or equal to", 0},
    {"supsetneq", 0x228B, Cat::Rel, "is a proper superset of", 0},
    {"to", 0x2192, Cat::Rel, "to", R_ARROW},
    {"rightarrow", 0, Cat::Rel, "to", R_ARROW},
    {"longrightarrow", 0x27F6, Cat::Rel, "to", R_ARROW},
    {"mapsto", 0x21A6, Cat::Rel, "maps to", 0},
    {"longmapsto", 0x27FC, Cat::Rel, "maps to", 0},
    {"Rightarrow", 0x21D2, Cat::Rel, "implies", 0},
    {"implies", 0x27F9, Cat::Rel, "implies", 0},
    {"Longrightarrow", 0, Cat::Rel, "implies", 0},
    {"Leftarrow", 0x21D0, Cat::Rel, "is implied by", 0},
    {"impliedby", 0x27F8, Cat::Rel, "is implied by", 0},
    {"Longleftarrow", 0, Cat::Rel, "is implied by", 0},
    {"Leftrightarrow", 0x21D4, Cat::Rel, "if and only if", 0},
    {"iff", 0x27FA, Cat::Rel, "if and only if", 0},
    {"Longleftrightarrow", 0, Cat::Rel, "if and only if", 0},
    {"leftarrow", 0x2190, Cat::Rel, "gets", 0},
    {"gets", 0, Cat::Rel, "gets", 0},
    {"longleftarrow", 0x27F5, Cat::Rel, "gets", 0},
    {"leftrightarrow", 0x2194, Cat::Rel, "corresponds to", 0},
    {"uparrow", 0x2191, Cat::Rel, "up", 0},
    {"downarrow", 0x2193, Cat::Rel, "down", 0},
    {"coloneqq", 0x2254, Cat::Rel, "is defined as", R_DEFINE},
    {"triangleq", 0x225C, Cat::Rel, "is defined as", R_DEFINE},
    {"doteq", 0x2250, Cat::Rel, "equals", R_EQ},
    {"models", 0x22A8, Cat::Rel, "models", 0},
    {"vdash", 0x22A2, Cat::Rel, "proves", 0},
    // before their operand
    {"neg", 0xAC, Cat::Prefix, "not", 0},
    {"lnot", 0, Cat::Prefix, "not", 0},
    {"forall", 0x2200, Cat::Prefix, "for all", P_QUANT},
    {"exists", 0x2203, Cat::Prefix, "there exists", P_QUANT},
    {"nexists", 0x2204, Cat::Prefix, "there does not exist", P_QUANT},
    {"nabla", 0x2207, Cat::Prefix, "del", P_NABLA},
    {"partial", 0x2202, Cat::Prefix, "partial", P_PARTIAL},
    {"angle", 0x2220, Cat::Prefix, "angle", 0},
    {"measuredangle", 0x2221, Cat::Prefix, "angle", 0},
    {"triangle", 0x25B3, Cat::Prefix, "triangle", 0},
    {"therefore", 0x2234, Cat::Prefix, "therefore", 0},
    {"because", 0x2235, Cat::Prefix, "because", 0},
    // nouns
    {"infty", 0x221E, Cat::Const, "infinity", 0},
    {"emptyset", 0x2205, Cat::Const, "the empty set", 0},
    {"varnothing", 0, Cat::Const, "the empty set", 0},
    {"hbar", 0x210F, Cat::Const, "h bar", 0},
    {"hslash", 0, Cat::Const, "h bar", 0},
    {"aleph", 0x2135, Cat::Const, "aleph", 0},
    {"beth", 0x2136, Cat::Const, "beth", 0},
    {"wp", 0x2118, Cat::Const, "p", 0},
    {"top", 0x22A4, Cat::Const, "top", C_TOP},
    {"intercal", 0x22BA, Cat::Const, "transpose", C_TOP},
    {"bot", 0, Cat::Const, "bottom", 0},
    {"dagger", 0x2020, Cat::Const, "dagger", C_DAGGER},
    {"ddagger", 0x2021, Cat::Const, "double dagger", 0},
    {"degree", 0xB0, Cat::Const, "degrees", C_DEGREE},
    {"prime", 0x2032, Cat::Prime, "prime", 0},
    {"", 0x2033, Cat::Prime, "double prime", 0},
    {"", 0x2034, Cat::Prime, "triple prime", 0},
    // ellipses
    {"ldots", 0x2026, Cat::Dots, "dot dot dot", 0},
    {"cdots", 0x22EF, Cat::Dots, "dot dot dot", 0},
    {"dots", 0, Cat::Dots, "dot dot dot", 0},
    {"dotsc", 0, Cat::Dots, "dot dot dot", 0},
    {"dotsb", 0, Cat::Dots, "dot dot dot", 0},
    {"dotsm", 0, Cat::Dots, "dot dot dot", 0},
    {"dotsi", 0, Cat::Dots, "dot dot dot", 0},
    {"dotso", 0, Cat::Dots, "dot dot dot", 0},
    {"mathellipsis", 0, Cat::Dots, "dot dot dot", 0},
    {"vdots", 0x22EE, Cat::Dots, "dot dot dot", 0},
    {"ddots", 0x22F1, Cat::Dots, "dot dot dot", 0},
    {"iddots", 0x22F0, Cat::Dots, "dot dot dot", 0},
    {"", 0x2025, Cat::Dots, "dot dot dot", 0},
    // punctuation
    {"", ',', Cat::Punct, ",", 0},
    {"", ';', Cat::Punct, ";", 0},
    {"", ':', Cat::Punct, ":", 0},
    {"colon", 0x2236, Cat::Punct, ":", 0},
    {"", '!', Cat::Punct, "factorial", 0},
    {"", '.', Cat::Punct, ".", 0},
    // large operators
    {"sum", 0x2211, Cat::Big, "sum", 0},
    {"prod", 0x220F, Cat::Big, "product", 0},
    {"coprod", 0x2210, Cat::Big, "coproduct", 0},
    {"int", 0x222B, Cat::Big, "integral", B_INTEGRAL},
    {"intop", 0, Cat::Big, "integral", B_INTEGRAL},
    {"iint", 0x222C, Cat::Big, "double integral", B_INTEGRAL},
    {"iiint", 0x222D, Cat::Big, "triple integral", B_INTEGRAL},
    {"oint", 0x222E, Cat::Big, "contour integral", B_INTEGRAL},
    {"bigcup", 0x22C3, Cat::Big, "union", 0},
    {"bigcap", 0x22C2, Cat::Big, "intersection", 0},
    {"bigsqcup", 0x2A06, Cat::Big, "union", 0},
    {"biguplus", 0x2A04, Cat::Big, "union", 0},
    {"bigoplus", 0x2A01, Cat::Big, "direct sum", 0},
    {"bigotimes", 0x2A02, Cat::Big, "tensor product", 0},
    {"bigvee", 0x22C1, Cat::Big, "disjunction", 0},
    {"bigwedge", 0x22C0, Cat::Big, "conjunction", 0},
    // named functions
    {"sin", 0, Cat::Func, "sine", F_TRIG},
    {"cos", 0, Cat::Func, "cosine", F_TRIG},
    {"tan", 0, Cat::Func, "tangent", F_TRIG},
    {"sec", 0, Cat::Func, "secant", F_TRIG},
    {"csc", 0, Cat::Func, "cosecant", F_TRIG},
    {"cot", 0, Cat::Func, "cotangent", F_TRIG},
    {"arcsin", 0, Cat::Func, "arcsine", F_TRIG},
    {"arccos", 0, Cat::Func, "arccosine", F_TRIG},
    {"arctan", 0, Cat::Func, "arctangent", F_TRIG},
    {"sinh", 0, Cat::Func, "hyperbolic sine", F_TRIG},
    {"cosh", 0, Cat::Func, "hyperbolic cosine", F_TRIG},
    {"tanh", 0, Cat::Func, "hyperbolic tangent", F_TRIG},
    {"coth", 0, Cat::Func, "hyperbolic cotangent", F_TRIG},
    {"log", 0, Cat::Func, "log", F_LOG},
    {"lg", 0, Cat::Func, "log", F_LOG},
    {"ln", 0, Cat::Func, "natural log", F_LN},
    {"exp", 0, Cat::Func, "exponential", F_PLAIN},
    {"lim", 0, Cat::Func, "limit", F_LIMIT},
    {"liminf", 0, Cat::Func, "limit inferior", F_LIMIT},
    {"limsup", 0, Cat::Func, "limit superior", F_LIMIT},
    {"varliminf", 0, Cat::Func, "limit inferior", F_LIMIT},
    {"varlimsup", 0, Cat::Func, "limit superior", F_LIMIT},
    {"max", 0, Cat::Func, "maximum", F_OVER},
    {"min", 0, Cat::Func, "minimum", F_OVER},
    {"sup", 0, Cat::Func, "supremum", F_OVER},
    {"inf", 0, Cat::Func, "infimum", F_OVER},
    {"argmax", 0, Cat::Func, "arg max", F_OVER},
    {"argmin", 0, Cat::Func, "arg min", F_OVER},
    {"arg", 0, Cat::Func, "argument", F_ARG},
    {"det", 0, Cat::Func, "determinant", F_PLAIN},
    {"dim", 0, Cat::Func, "dimension", F_PLAIN},
    {"ker", 0, Cat::Func, "kernel", F_PLAIN},
    {"deg", 0, Cat::Func, "degree", F_PLAIN},
    {"gcd", 0, Cat::Func, "greatest common divisor", F_PLAIN},
    {"Pr", 0, Cat::Func, "probability", F_PLAIN},
    {"sgn", 0, Cat::Func, "sign", F_PLAIN},
    {"Re", 0x211C, Cat::Func, "real part", F_PLAIN},
    {"Im", 0x2111, Cat::Func, "imaginary part", F_PLAIN},
};

// Function names written as words (\operatorname{tr}, \mathrm{Var}).
const Sym WORD_FUNCS[] = {
    {"Cov", 0, Cat::Func, "covariance", F_PLAIN},       {"E", 0, Cat::Func, "expected value", F_EXPECT},
    {"Im", 0, Cat::Func, "imaginary part", F_PLAIN},    {"Pr", 0, Cat::Func, "probability", F_PLAIN},
    {"Re", 0, Cat::Func, "real part", F_PLAIN},         {"Tr", 0, Cat::Func, "trace", F_PLAIN},
    {"Var", 0, Cat::Func, "variance", F_PLAIN},         {"arccos", 0, Cat::Func, "arccosine", F_TRIG},
    {"arcsin", 0, Cat::Func, "arcsine", F_TRIG},        {"arctan", 0, Cat::Func, "arctangent", F_TRIG},
    {"arg", 0, Cat::Func, "argument", F_ARG},           {"argmax", 0, Cat::Func, "arg max", F_OVER},
    {"argmin", 0, Cat::Func, "arg min", F_OVER},        {"cos", 0, Cat::Func, "cosine", F_TRIG},
    {"cosh", 0, Cat::Func, "hyperbolic cosine", F_TRIG}, {"cot", 0, Cat::Func, "cotangent", F_TRIG},
    {"cov", 0, Cat::Func, "covariance", F_PLAIN},       {"csc", 0, Cat::Func, "cosecant", F_TRIG},
    {"det", 0, Cat::Func, "determinant", F_PLAIN},      {"diag", 0, Cat::Func, "diagonal", F_PLAIN},
    {"dim", 0, Cat::Func, "dimension", F_PLAIN},        {"erf", 0, Cat::Func, "error function", F_PLAIN},
    {"erfc", 0, Cat::Func, "complementary error function", F_PLAIN},
    {"exp", 0, Cat::Func, "exponential", F_PLAIN},      {"gcd", 0, Cat::Func, "greatest common divisor", F_PLAIN},
    {"inf", 0, Cat::Func, "infimum", F_OVER},           {"ker", 0, Cat::Func, "kernel", F_PLAIN},
    {"lcm", 0, Cat::Func, "least common multiple", F_PLAIN},
    {"lim", 0, Cat::Func, "limit", F_LIMIT},            {"ln", 0, Cat::Func, "natural log", F_LN},
    {"log", 0, Cat::Func, "log", F_LOG},                {"max", 0, Cat::Func, "maximum", F_OVER},
    {"min", 0, Cat::Func, "minimum", F_OVER},           {"rank", 0, Cat::Func, "rank", F_PLAIN},
    {"sec", 0, Cat::Func, "secant", F_TRIG},            {"sgn", 0, Cat::Func, "sign", F_PLAIN},
    {"sign", 0, Cat::Func, "sign", F_PLAIN},            {"sin", 0, Cat::Func, "sine", F_TRIG},
    {"sinh", 0, Cat::Func, "hyperbolic sine", F_TRIG},  {"sup", 0, Cat::Func, "supremum", F_OVER},
    {"supp", 0, Cat::Func, "support", F_PLAIN},         {"tan", 0, Cat::Func, "tangent", F_TRIG},
    {"tanh", 0, Cat::Func, "hyperbolic tangent", F_TRIG}, {"tr", 0, Cat::Func, "trace", F_PLAIN},
    {"var", 0, Cat::Func, "variance", F_PLAIN},
};

struct GreekCmd {
    std::string_view tex;
    uint32_t cp;
};
const GreekCmd GREEK[] = {
    {"Alpha", 0x391},   {"Beta", 0x392},    {"Chi", 0x3A7},      {"Delta", 0x394},   {"Epsilon", 0x395},
    {"Eta", 0x397},     {"Gamma", 0x393},   {"Iota", 0x399},     {"Kappa", 0x39A},   {"Lambda", 0x39B},
    {"Mu", 0x39C},      {"Nu", 0x39D},      {"Omega", 0x3A9},    {"Omicron", 0x39F}, {"Phi", 0x3A6},
    {"Pi", 0x3A0},      {"Psi", 0x3A8},     {"Rho", 0x3A1},      {"Sigma", 0x3A3},   {"Tau", 0x3A4},
    {"Theta", 0x398},   {"Upsilon", 0x3A5}, {"Xi", 0x39E},       {"Zeta", 0x396},    {"alpha", 0x3B1},
    {"beta", 0x3B2},    {"chi", 0x3C7},     {"delta", 0x3B4},    {"digamma", 0x3DD}, {"epsilon", 0x3B5},
    {"eta", 0x3B7},     {"gamma", 0x3B3},   {"iota", 0x3B9},     {"kappa", 0x3BA},   {"lambda", 0x3BB},
    {"mu", 0x3BC},      {"nu", 0x3BD},      {"omega", 0x3C9},    {"omicron", 0x3BF}, {"phi", 0x3C6},
    {"pi", 0x3C0},      {"psi", 0x3C8},     {"rho", 0x3C1},      {"sigma", 0x3C3},   {"tau", 0x3C4},
    {"theta", 0x3B8},   {"upsilon", 0x3C5}, {"varDelta", 0x394}, {"varGamma", 0x393}, {"varLambda", 0x39B},
    {"varOmega", 0x3A9}, {"varPhi", 0x3A6}, {"varPi", 0x3A0},    {"varPsi", 0x3A8},  {"varSigma", 0x3A3},
    {"varTheta", 0x398}, {"varUpsilon", 0x3A5}, {"varXi", 0x39E}, {"varepsilon", 0x3B5}, {"varkappa", 0x3BA},
    {"varphi", 0x3C6},  {"varpi", 0x3C0},   {"varrho", 0x3C1},   {"varsigma", 0x3C3}, {"vartheta", 0x3B8},
    {"xi", 0x3BE},      {"zeta", 0x3B6},
};


struct Index {
    std::vector<const Sym*> by_tex, by_cp;
    Index() {
        for (const Sym& s : SYMS) {
            if (!s.tex.empty()) by_tex.push_back(&s);
            if (s.cp) by_cp.push_back(&s);
        }
        std::sort(by_tex.begin(), by_tex.end(), [](const Sym* a, const Sym* b) { return a->tex < b->tex; });
        std::stable_sort(by_cp.begin(), by_cp.end(), [](const Sym* a, const Sym* b) { return a->cp < b->cp; });
    }
};
const Index& index() {
    static const Index ix;
    return ix;
}

}  // namespace

const Sym* sym_tex(std::string_view name) {
    const auto& v = index().by_tex;
    auto it = std::lower_bound(v.begin(), v.end(), name, [](const Sym* a, std::string_view n) { return a->tex < n; });
    return it != v.end() && (*it)->tex == name ? *it : nullptr;
}

const Sym* sym_cp(uint32_t cp) {
    const auto& v = index().by_cp;
    auto it = std::lower_bound(v.begin(), v.end(), cp, [](const Sym* a, uint32_t c) { return a->cp < c; });
    return it != v.end() && (*it)->cp == cp ? *it : nullptr;
}

uint32_t greek_tex(std::string_view name) {
    auto it = std::lower_bound(std::begin(GREEK), std::end(GREEK), name,
                               [](const GreekCmd& g, std::string_view n) { return g.tex < n; });
    return it != std::end(GREEK) && it->tex == name ? it->cp : 0;
}

const Sym* func_word(std::string_view word) {
    auto it = std::lower_bound(std::begin(WORD_FUNCS), std::end(WORD_FUNCS), word,
                               [](const Sym& s, std::string_view w) { return s.tex < w; });
    return it != std::end(WORD_FUNCS) && it->tex == word ? &*it : nullptr;
}

namespace {

struct Unit {
    std::string_view sym;
    const char* one;
    const char* many;
};
// Sorted by symbol.
const Unit UNITS[] = {
    {"A", "ampere", "amperes"},       {"C", "coulomb", "coulombs"},     {"F", "farad", "farads"},
    {"GHz", "gigahertz", "gigahertz"}, {"GeV", "giga electron volt", "giga electron volts"},
    {"Hz", "hertz", "hertz"},         {"J", "joule", "joules"},         {"K", "kelvin", "kelvins"},
    {"L", "liter", "liters"},         {"MHz", "megahertz", "megahertz"}, {"MPa", "megapascal", "megapascals"},
    {"MW", "megawatt", "megawatts"},  {"MeV", "mega electron volt", "mega electron volts"},
    {"N", "newton", "newtons"},       {"Pa", "pascal", "pascals"},      {"T", "tesla", "teslas"},
    {"V", "volt", "volts"},           {"W", "watt", "watts"},           {"atm", "atmosphere", "atmospheres"},
    {"cal", "calorie", "calories"},   {"cm", "centimeter", "centimeters"}, {"dB", "decibel", "decibels"},
    {"eV", "electron volt", "electron volts"}, {"ft", "foot", "feet"}, {"g", "gram", "grams"},
    {"h", "hour", "hours"},           {"kHz", "kilohertz", "kilohertz"}, {"kJ", "kilojoule", "kilojoules"},
    {"kPa", "kilopascal", "kilopascals"}, {"kW", "kilowatt", "kilowatts"}, {"kcal", "kilocalorie", "kilocalories"},
    {"keV", "kilo electron volt", "kilo electron volts"}, {"kg", "kilogram", "kilograms"},
    {"km", "kilometer", "kilometers"}, {"lb", "pound", "pounds"},       {"m", "meter", "meters"},
    {"mA", "milliampere", "milliamperes"}, {"mL", "milliliter", "milliliters"}, {"mV", "millivolt", "millivolts"},
    {"mg", "milligram", "milligrams"}, {"mi", "mile", "miles"},         {"min", "minute", "minutes"},
    {"mm", "millimeter", "millimeters"}, {"mol", "mole", "moles"},      {"ms", "millisecond", "milliseconds"},
    {"nm", "nanometer", "nanometers"}, {"ns", "nanosecond", "nanoseconds"}, {"rad", "radian", "radians"},
    {"s", "second", "seconds"},       {"yr", "year", "years"},
};

const Unit* find_unit(std::string_view u) {
    auto it = std::lower_bound(std::begin(UNITS), std::end(UNITS), u,
                               [](const Unit& x, std::string_view k) { return x.sym < k; });
    return it != std::end(UNITS) && it->sym == u ? &*it : nullptr;
}

// "m^2" -> "m" and '2'.
std::string_view split_power(std::string_view u, char* pw) {
    *pw = 0;
    size_t c = u.find('^');
    if (c != std::string_view::npos && c + 2 == u.size() && (u[c + 1] == '2' || u[c + 1] == '3')) {
        *pw = u[c + 1];
        u = u.substr(0, c);
    }
    return u;
}

}  // namespace

std::string unit_words(std::string_view unit, bool plural) {
    while (!unit.empty() && unit.front() == ' ') unit.remove_prefix(1);
    while (!unit.empty() && unit.back() == ' ') unit.remove_suffix(1);
    std::string_view num = unit, den;
    size_t slash = unit.find('/');
    if (slash != std::string_view::npos) {
        num = unit.substr(0, slash);
        den = unit.substr(slash + 1);
    }
    char np = 0, dp = 0;
    const Unit* a = find_unit(split_power(num, &np));
    if (!a) return "";
    std::string s = std::string(np == '2' ? "square " : np == '3' ? "cubic " : "") + (plural ? a->many : a->one);
    if (slash != std::string_view::npos) {
        const Unit* b = find_unit(split_power(den, &dp));
        if (!b) return "";
        s += std::string(" per ") + b->one + (dp == '2' ? " squared" : dp == '3' ? " cubed" : "");
    }
    return s;
}

std::string letter(uint32_t cp) { return rv::variable_name(cp); }

std::string letter_ordinal(uint32_t cp) {
    std::string l = letter(cp);
    return l.empty() ? l : l + "-th";  // the N-th root: the letter stays a word of its own
}

}  // namespace mth
