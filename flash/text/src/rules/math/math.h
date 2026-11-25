#pragma once
// The module "math": a formula the page extractor marked (U+FDD0, TeX source, U+FDD1) read in words. A small
// TeX tokenizer and recursive-descent parser build a tree (parse.cpp); the speaker reads the tree with the
// ClearSpeak wording (speak.cpp; conventions and deviations in the README).
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mth {

enum class Cat : uint8_t {
    Op,      // binary or unary operator: + - × · ∪ ∧ mod
    Rel,     // relation: = < ≤ ∈ ⊂ → ⇒
    Const,   // a symbol read as a noun: infinity, the empty set, h bar
    Func,    // a named function: sin, log, lim, max, det
    Big,     // a large operator with bounds: sum, product, integral, union
    Prefix,  // an operator before its operand: nabla, partial, not, for all, there exists
    Dots,    // ellipsis
    Punct,   // , ; : ! .
    Prime,   // ′ and \prime
};

// Kinds of functions and large operators.
enum : uint8_t {
    F_TRIG = 1,  // sine x, the sine of ...
    F_LOG,       // log x, log base b of x
    F_LN,        // natural log x
    F_LIMIT,     // the limit as x approaches a of ...
    F_OVER,      // the maximum over x of ...
    F_PLAIN,     // the determinant of ...
    F_EXPECT,    // the expected value of ...
    F_ARG,       // \arg: arg max with a following max or min
    B_INTEGRAL,  // integrals: the integrand runs to its differential
    P_NABLA,
    P_PARTIAL,
    P_QUANT,     // for all, there exists: a following relation reads like a bound (for all x in A)
    O_ADD,       // + - ± ∓: these end a term
    O_DIV,       // the slash
    R_ARROW,     // → in a limit reads "approaches"
    R_SIM,       // ∼: "is distributed as" before a capital, "is similar to" otherwise
    R_EQ,
    R_DEFINE,
    C_STAR,      // * as a superscript: star
    C_DAGGER,
    C_TOP,       // ⊤ as a superscript: transpose
    C_DEGREE,    // ∘ and ° as a superscript: degrees
};

struct Sym {
    std::string_view tex;  // command name without the backslash, "" if none
    uint32_t cp;           // code point, 0 if none
    Cat cat;
    std::string_view say;  // the words
    uint8_t kind;          // F_ / B_ / P_ / O_ / R_ / C_ above, 0 if none
};
const Sym* sym_tex(std::string_view name);
const Sym* sym_cp(uint32_t cp);
// The Greek letter of a command (\alpha, \varphi, \Gamma), 0 if none.
uint32_t greek_tex(std::string_view name);
// A function name written in letters (\operatorname{tr}, \mathrm{Var}, sin in plain letters), nullptr if none.
const Sym* func_word(std::string_view word);

enum class K : uint8_t {
    Row,      // kids: a sequence of items
    Letter,   // cp: a Latin or Greek letter; n = 1 when upright (\mathrm{d})
    Num,      // t: digits as written, grouping kept ("3.14", "1,000")
    Word,     // t: words (\text{if}, \mathrm{softmax})
    Sym,      // s: a symbol of the table; n = 1 when negated (\not\in)
    Open,     // t: delimiter: ( [ { | ‖ < floor ceil
    Close,
    Bar,      // a | or ‖ that may open, close, or stand alone; t: "|", "‖", "mid"
    Fence,    // kids[0]: body; t: open delimiter, u: close delimiter
    Frac,     // kids: numerator, denominator; n = 1 for a binomial coefficient
    Root,     // kids: body, index (Empty if none)
    Scripts,  // kids: base, subscript, superscript (Empty if absent); n: number of primes
    Accent,   // kids: body; t: hat, bar, vec, tilde, dot, ddot, brace...
    Set,      // a number set; cp: R N Z Q C
    Table,    // kids: rows (Row of cells); t: environment name
    Space,    // n = 1 for a wide space (\quad) or a line break outside an environment
    Empty,
};

struct Node {
    K k = K::Empty;
    uint8_t n = 0;
    uint32_t cp = 0;
    const Sym* s = nullptr;
    std::string t, u;
    std::vector<Node> kids;
};

// The TeX source between the markers -> tree (a Row). Never fails: what it cannot read becomes Empty.
Node parse(std::string_view tex);
// The tree -> spoken words: ASCII letters, digits, spaces and , . ; : ' - only. "" when nothing is readable.
std::string speak(const Node& root);

// The written form of a one-letter variable (Latin or Greek): rv::variable_name, the one table for every letter.
std::string letter(uint32_t cp);
// "N-th", "K-th": a letter as an ordinal (roots and derivatives of order n).
std::string letter_ordinal(uint32_t cp);
// A unit after a number written upright (\mathrm{kg}, \text{m/s}, \mathrm{m/s^2}): "kilograms", "meters per
// second squared"; "" if it is not a unit.
std::string unit_words(std::string_view unit, bool plural);

}  // namespace mth
