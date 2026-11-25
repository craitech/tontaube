// Tree -> words, with the ClearSpeak wording (README: "Reading formulas"). Each reading reports whether it is
// simple (ClearSpeak's simple expression), one term (no operator, relation or comma at its top level), and
// open (it ends in a part whose end the listener cannot hear: what follows gets a comma).
#include <algorithm>

#include "core/readers.h"
#include "rules/math/math.h"

namespace mth {

namespace {

struct Say {
    std::string s;
    bool simple = false;
    bool term = true;
    bool open = false;
    bool single = false;  // one operand, nothing else at its top level
};

struct Ctx {
    bool bounds = false;  // bounds of a sum, a limit, a set builder: relations without "is" (i in S)
    bool lim = false;     // a limit's bound: an arrow reads "approaches"
    bool set = false;     // inside set braces: a bar or a colon reads "such that"
    bool args = false;    // inside function arguments: a bar reads "given"
    bool dims = false;      // the exponent of a number set: times reads "by" (R n by m)
    bool interval = false;  // after "is in": (a, b) is an interval
    bool range = false;     // a subscript: a colon between operands reads "to" (x_{1:T})
};

using Items = std::vector<const Node*>;

bool digit(char c) { return c >= '0' && c <= '9'; }

const Node* strip(const Node& n) { return n.k == K::Row && n.kids.size() == 1 ? &n.kids[0] : &n; }
bool sym_is(const Node& n, Cat cat) { return n.k == K::Sym && n.s && n.s->cat == cat; }
bool sym_kind(const Node& n, uint8_t kind) { return n.k == K::Sym && n.s && n.s->kind == kind; }
bool is_num(const Node& n, std::string_view t) { return n.k == K::Num && n.t == t; }
bool is_minus(const Node& n) { return sym_is(n, Cat::Op) && n.s->kind == O_ADD && n.s->say == "minus"; }
bool comma_sym(const Node& n) { return sym_is(n, Cat::Punct) && (n.s->say == "," || n.s->say == ";"); }
bool wide_space(const Node& n) { return n.k == K::Space && n.n == 1; }
bool blank(const Node& n) { return n.k == K::Space || n.k == K::Empty; }
bool upper(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') || (cp >= 0x391 && cp <= 0x3A9); }

// The items of a row without spaces (or the node itself).
Items items_of(const Node& n) {
    Items v;
    if (n.k == K::Row) {
        for (const Node& k : n.kids)
            if (!blank(k)) v.push_back(&k);
    } else if (n.k != K::Empty) {
        v.push_back(&n);
    }
    return v;
}
// The items of a row (or the node itself), spaces kept: they separate lines and bounds.
Items ptrs(const Node& n) {
    Items v;
    if (n.k == K::Row)
        for (const Node& k : n.kids) v.push_back(&k);
    else if (n.k != K::Empty)
        v.push_back(&n);
    return v;
}

std::string number_words(const std::string& t) {
    if (t.empty()) return "";
    if (t[0] == '.') return "point " + rv::digits(t.substr(1));
    std::string w = rv::read_number(t);
    if (!w.empty()) return w;
    std::string whole, frac;
    size_t dot = t.find('.');
    for (size_t k = 0; k < t.size() && k < dot; ++k)
        if (digit(t[k])) whole += t[k];
    if (dot != std::string::npos) frac = t.substr(dot + 1);
    std::string out = rv::digits(whole);
    if (!frac.empty()) out += " point " + rv::digits(frac);
    return out;
}

// 1/2 one half, 3/4 three quarters, 5/8 five eighths: numerators up to 19, denominators 2 to 10.
std::string common_fraction(const Node& num, const Node& den) {
    const Node *a = strip(num), *b = strip(den);
    if (a->k != K::Num || b->k != K::Num || a->t.find_first_not_of("0123456789") != std::string::npos ||
        b->t.find_first_not_of("0123456789") != std::string::npos || a->t.size() > 2 || b->t.size() > 2 ||
        a->t[0] == '0' || b->t[0] == '0')
        return "";
    int n = std::stoi(a->t), d = std::stoi(b->t);
    if (n < 1 || n > 19 || d < 2 || d > 10) return "";
    static const char* const ORD[] = {"", "", "half", "third", "quarter", "fifth", "sixth", "seventh", "eighth",
                                      "ninth", "tenth"};
    std::string den_w = ORD[d];
    if (n > 1) den_w = d == 2 ? "halves" : den_w + "s";
    return rv::cardinal(static_cast<uint64_t>(n)) + " " + den_w;
}

bool simple(const Node& node);

// 1/2 written with a slash: the three items num / den.
bool slash_fraction(const Items& v, size_t k, std::string* words) {
    if (k + 2 >= v.size() || v[k]->k != K::Num || !(v[k + 1]->k == K::Sym && v[k + 1]->s->kind == O_DIV) ||
        v[k + 2]->k != K::Num)
        return false;
    std::string w = common_fraction(*v[k], *v[k + 2]);
    if (w.empty()) return false;
    if (words) *words = w;
    return true;
}

// ClearSpeak's simple expression over a sequence: a negative letter or number, two letters, a number with one
// or two letters (2x).
bool simple_items(const Items& v) {
    if (v.size() == 1) return simple(*v[0]);
    if (v.size() == 2 && is_minus(*v[0]) && (v[1]->k == K::Letter || v[1]->k == K::Num)) return true;
    if (v.size() == 2 && v[0]->k == K::Letter && v[1]->k == K::Letter) return true;
    if ((v.size() == 2 || v.size() == 3) && v[0]->k == K::Num && v[1]->k == K::Letter &&
        (v.size() == 2 || v[2]->k == K::Letter))
        return true;
    return false;
}

// ClearSpeak's simple expression: a number, a letter, a common fraction, and the sequences above; here also a
// letter with a simple subscript, primes or an accent.
bool simple(const Node& node) {
    const Node& n = *strip(node);
    switch (n.k) {
        case K::Num: case K::Letter: case K::Set: return true;
        case K::Word: return n.t.find(' ') == std::string::npos;
        case K::Sym: return n.s && n.s->cat == Cat::Const;
        case K::Accent: return strip(n.kids[0])->k == K::Letter || (n.t != "bar" && simple(n.kids[0]));
        case K::Frac: return n.n == 0 && !common_fraction(n.kids[0], n.kids[1]).empty();
        case K::Scripts: {
            const Node &b = *strip(n.kids[0]), &sub = *strip(n.kids[1]);
            if (n.kids[2].k != K::Empty) return false;
            if (!(b.k == K::Letter || b.k == K::Accent || b.k == K::Word)) return false;
            return sub.k == K::Empty || sub.k == K::Num || sub.k == K::Letter || sub.k == K::Word ||
                   (sub.k == K::Row && simple_items(items_of(sub)) && !is_minus(*items_of(sub)[0]));
        }
        case K::Row: return simple_items(items_of(n));
        default: return false;
    }
}

// An exponent read as "to the power of E": a number, a letter, their negatives, a common fraction, infinity.
bool simple_exponent(const Node& node) {
    const Node& n = *strip(node);
    switch (n.k) {
        case K::Num: case K::Letter: return true;
        case K::Word: return n.t.find(' ') == std::string::npos;
        case K::Sym: return n.s && n.s->cat == Cat::Const;
        case K::Frac: case K::Accent: case K::Scripts: return simple(n);
        case K::Row: {
            Items v = items_of(n);
            if (v.size() == 3 && slash_fraction(v, 0, nullptr)) return true;
            return v.size() == 2 && is_minus(*v[0]) && (v[1]->k == K::Num || v[1]->k == K::Letter);
        }
        default: return false;
    }
}

// d, \mathrm{d} or ∂, with an order as superscript when order is given: the start of a derivative.
bool diff_mark(const Node& n, bool* partial, const Node** order) {
    const Node* b = &n;
    if (order) *order = nullptr;
    if (n.k == K::Scripts) {
        if (!order || n.kids[1].k != K::Empty || n.kids[2].k == K::Empty || n.n) return false;
        b = strip(n.kids[0]);
        *order = strip(n.kids[2]);
    }
    bool d = (b->k == K::Letter && b->cp == 'd') || (b->k == K::Word && b->t == "d");
    bool p = sym_kind(*b, P_PARTIAL);
    if (partial) *partial = p;
    return d || p;
}

// A variable of differentiation or integration: x, \theta, x_i, \mathbf{x}, x^2 (the power is the order).
bool variable(const Node& n) {
    const Node* b = &n;
    if (n.k == K::Scripts) b = strip(n.kids[0]);
    return b->k == K::Letter || (b->k == K::Accent && strip(b->kids[0])->k == K::Letter);
}

std::string ordinal_words(const Node& n) {
    const Node* o = strip(n);
    if (o->k == K::Num && o->t.find_first_not_of("0123456789") == std::string::npos && o->t.size() < 4)
        return rv::ordinal(std::stoull(o->t));
    if (o->k == K::Letter) return letter_ordinal(o->cp);
    return "";
}

std::string attach(const std::string& s, bool open, const std::string& w) {
    if (s.empty()) return w;
    return s + (open ? ", " : " ") + w;
}

std::string lower_first_word(const std::string& t) {
    std::string w = t.substr(0, t.find(' '));
    for (char& ch : w)
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch + 32);
    return w;
}

// Relations: "is less than" loses its "is" in bounds (the sum over i less than n), and \not negates.
std::string relation_words(const Node& n, const Ctx& c, bool quant, const Node* next, bool applied) {
    const Sym* s = n.s;
    std::string w(s->say);
    if (s->kind == R_ARROW && c.lim) w = "approaches";
    if (s->kind == R_SIM && next) {  // x ~ N(0, 1), x ~ p(x): a distribution
        const Node* r = next->k == K::Scripts ? strip(next->kids[0]) : next;
        bool cap = (r->k == K::Letter && upper(r->cp)) ||
                   (r->k == K::Word && !r->t.empty() && r->t[0] >= 'A' && r->t[0] <= 'Z');
        if (cap || applied || c.bounds) w = "is distributed as";
    }
    if (n.n == 1) {
        if (w == "equals") w = "is not equal to";
        else if (w.compare(0, 3, "is ") == 0) w = "is not " + w.substr(3);
        else w = "not " + w;
    }
    if ((c.bounds || quant) && w.compare(0, 3, "is ") == 0) w = w.substr(3);
    return w;
}

std::vector<std::pair<size_t, size_t>> split_commas(const Items& v) {
    std::vector<std::pair<size_t, size_t>> parts;
    size_t a = 0;
    for (size_t k = 0; k < v.size(); ++k)
        if (comma_sym(*v[k])) {
            parts.push_back({a, k});
            a = k + 1;
        }
    parts.push_back({a, v.size()});
    return parts;
}

class Speaker {
public:
    Say node(const Node& n, Ctx c);
    Say row(const Items& v, size_t a, size_t b, Ctx c);
    Say fence(const Node& f, Ctx c, bool force_quantity);
    Say table(const Node& t, Ctx c, const std::string& as);
    Say scripts(const Node& n, Ctx c);
    Say frac(const Node& n, Ctx c);
    Say root(const Node& n, Ctx c);
    Say accent(const Node& n, Ctx c);
    // The words of a superscript p on base b: squared, to the power of n, transpose, inverse, prime...
    std::string sup_words(const Node* b, const Node* p, Ctx c);
    // Function arguments: "x, y" (open), "x", "two x", "the quantity x plus h" (open).
    std::string args(const Node& body, Ctx c, bool* open, bool* simple_arg);
};

// Reads a row left to right: operands, operators, relations, punctuation; function names, large operators
// and derivative operators take the term after them.
class RowReader {
public:
    RowReader(Speaker& sp, const Items& v, size_t a, size_t b, Ctx c) : sp_(sp), v_(v), i_(a), b_(b), c_(c) {}

    Say read() {
        Items mine;
        for (size_t k = i_; k < b_; ++k)
            if (!blank(*v_[k])) mine.push_back(v_[k]);
        while (i_ < b_) step();
        Say r;
        bool prev_open = false;
        for (const Piece& p : pieces_) {
            if (p.punct) {
                r.s += p.s;
                prev_open = false;
                continue;
            }
            if (!r.s.empty()) {
                if (prev_open) r.s += ',';
                r.s += ' ';
            }
            r.s += p.s;
            prev_open = p.open;
        }
        r.open = prev_open;
        r.term = !top_ops_;
        r.simple = !mine.empty() && simple_items(mine);
        r.single = operands_ == 1 && pieces_.size() == 1;
        return r;
    }

private:
    struct Piece {
        std::string s;
        bool open = false;
        bool punct = false;
    };
    enum Prev { NONE, OPERAND, OPERATOR };

    Speaker& sp_;
    const Items& v_;
    size_t i_, b_;
    Ctx c_;
    std::vector<Piece> pieces_;
    Prev prev_ = NONE;
    bool top_ops_ = false;
    bool quant_ = false;
    bool last_quantity_ = false;  // the last operand was read as "the quantity ..."
    const Node* last_ = nullptr;
    int operands_ = 0;
    const Node* pending_sup_ = nullptr;  // a power on the argument's bracket, read after the function
    bool last_rel_in_ = false;           // the last relation was "is in": (a, b) after it is an interval
    bool colon_ = false;                 // a colon came before (f: X \to Y)

    const Node& at(size_t k) const { return *v_[k]; }

    void add(std::string s, bool open = false) {
        if (s.empty()) return;
        pieces_.push_back({std::move(s), open, false});
    }
    void punct(const char* p) {
        if (pieces_.empty() || pieces_.back().punct) return;
        pieces_.push_back({p, false, true});
    }
    size_t next_index(size_t k) const {
        while (k < b_ && blank(at(k))) ++k;
        return k;
    }
    const Node* next_item(size_t k) const {
        k = next_index(k);
        return k < b_ ? v_[k] : nullptr;
    }

    // End of the term starting at k: up to + or - between operands, a relation, punctuation or a wide space;
    // with mult false also up to a product sign.
    size_t term_end(size_t k, bool mult) const {
        size_t e = k;
        bool seen = false;
        while (e < b_) {
            const Node& n = at(e);
            if (sym_is(n, Cat::Op)) {
                if (!seen && n.s->kind == O_ADD) {
                    ++e;
                    continue;
                }
                if (n.s->kind == O_ADD || n.s->kind == O_DIV || !mult) break;
            } else if (sym_is(n, Cat::Rel) || (sym_is(n, Cat::Punct) && n.s->say != "factorial") ||
                       n.k == K::Bar || wide_space(n) || (n.k == K::Word && connective(n.t))) {
                break;
            }
            if (!blank(n)) seen = true;
            ++e;
        }
        return e;
    }

    Say range(size_t a, size_t e, Ctx c) { return RowReader(sp_, v_, a, e, c).read(); }

    void step() {
        const Node& n = at(i_);
        switch (n.k) {
            case K::Empty: ++i_; return;
            case K::Space:
                if (n.n == 1 && prev_ == OPERAND && next_item(i_ + 1)) {
                    if (c_.bounds) add("and");
                    else punct(",");
                    prev_ = OPERATOR;
                    top_ops_ = true;
                }
                ++i_;
                return;
            case K::Bar: {  // a bar between two operands: such that, given, divides
                const Node* nx = next_item(i_ + 1);
                if (prev_ == OPERAND && nx && nx->k != K::Bar && !sym_is(*nx, Cat::Rel) && !sym_is(*nx, Cat::Punct)) {
                    add(c_.set ? "such that" : c_.args ? "given" : "divides");
                    prev_ = OPERATOR;
                    top_ops_ = true;
                }
                ++i_;
                return;
            }
            case K::Word:
                if (connective(n.t) && n.t != "the indicator") {
                    add(n.t);
                    quant_ = lower_first_word(n.t) == "for";  // for x in A: like "for all"
                    prev_ = OPERATOR;
                    top_ops_ = true;
                    ++i_;
                    return;
                }
                break;
            case K::Sym:
                switch (n.s->cat) {
                    case Cat::Op: op(n); return;
                    case Cat::Rel: rel(n); return;
                    case Cat::Punct: punct_sym(n); return;
                    case Cat::Prime:
                        if (prev_ == OPERAND) add(std::string(n.s->say));
                        ++i_;
                        return;
                    case Cat::Prefix:
                        if (n.s->kind != P_NABLA && n.s->kind != P_PARTIAL) {
                            add(std::string(n.s->say));
                            quant_ = n.s->kind == P_QUANT;
                            prev_ = OPERATOR;
                            ++i_;
                            return;
                        }
                        break;
                    default: break;
                }
                break;
            default: break;
        }
        operand();
    }

    static bool connective(const std::string& t) {
        std::string w = lower_first_word(t);
        static const char* const C[] = {"if", "for", "when", "where", "otherwise", "else", "unless", "and", "or",
                                        "as", "with", "such", "then", "since", "provided", "while", "whenever",
                                        "is", "are", "so", "but", "iff", "on", "in", "at", "all", "some", "not",
                                        "of", "to", "from", "by", "over", "the"};
        for (const char* x : C)
            if (w == x) return true;
        return false;
    }

    void op(const Node& n) {
        const Sym* s = n.s;
        bool unary = prev_ != OPERAND;
        if (s->kind == C_STAR && prev_ == OPERAND && !next_item(i_ + 1)) {  // x^* written as x*
            add("star");
            ++i_;
            return;
        }
        std::string w(s->say);
        if (c_.dims && w == "times") w = "by";
        add(w);
        if (!unary && !(w == "times" || w == "divided by" || w == "tensor" || w == "elementwise times" || w == "star"))
            top_ops_ = true;
        prev_ = OPERATOR;
        ++i_;
    }

    void rel(const Node& n) {
        last_rel_in_ = n.s->cp == 0x2208 || n.s->tex == "in";
        const Node* nx = next_item(i_ + 1);
        // a_n \to 0, x \to \infty: approaches; f: X \to Y stays "to"
        if (n.s->kind == R_ARROW && nx && !colon_ && n.n == 0) {
            size_t k = next_index(i_ + 1);
            const Node* v = is_minus(*nx) ? next_item(k + 1) : nx;
            if (v && (v->k == K::Num || (v->k == K::Sym && v->s->say == "infinity"))) {
                add("approaches");
                quant_ = false;
                prev_ = OPERATOR;
                top_ops_ = true;
                ++i_;
                return;
            }
        }
        if (n.s->cp == '~' && !(prev_ == OPERAND && nx && operand_like(*nx))) {  // a tie: a space
            ++i_;
            return;
        }
        size_t k2 = nx ? next_index(next_index(i_ + 1) + 1) : b_;
        const Node* nb = nx && nx->k == K::Scripts ? strip(nx->kids[0]) : nx;
        bool applied = nb && (nb->k == K::Letter || nb->k == K::Word) && k2 < b_ && at(k2).k == K::Fence &&
                       at(k2).t == "(";
        add(relation_words(n, c_, quant_, nx, applied));
        quant_ = false;
        prev_ = OPERATOR;
        top_ops_ = true;
        ++i_;
    }

    void punct_sym(const Node& n) {
        std::string_view p = n.s->say;
        ++i_;
        if (p != "factorial") quant_ = false;
        if (p == "factorial") {
            if (prev_ == OPERAND) add("factorial");
            return;
        }
        if (p == ":" && c_.range && prev_ == OPERAND && next_item(i_)) {  // x_{1:T}
            add("to");
            prev_ = OPERATOR;
            top_ops_ = true;
            return;
        }
        if (p == ":") {
            colon_ = true;
            size_t k = next_index(i_);
            if (k < b_ && sym_kind(at(k), R_EQ)) {  // :=
                i_ = k + 1;
                add("is defined as");
            } else if (c_.set) {
                add("such that");
            } else {
                punct(":");
            }
            prev_ = OPERATOR;
            top_ops_ = true;
            return;
        }
        if (p == ".") {
            punct(".");
            return;
        }
        punct(p == ";" ? ";" : ",");
        prev_ = OPERATOR;
        top_ops_ = true;
    }

    // f(x) -> "f of x", g(x, y) -> "g of x, y", f(x+1) -> "f of the quantity x plus one". A letter used as a
    // coefficient before a quantity is a product (a(b+c)), as in ClearSpeak.
    bool applies(const Node& name, const Node& f) const {
        if (f.k != K::Fence || !(f.t == "(" || f.t == "[") || f.kids.empty()) return false;
        const Node* b = &name;
        if (name.k == K::Scripts) {
            const Node* sup = strip(name.kids[2]);
            if (sup->k != K::Empty && !sym_is(*sup, Cat::Prime) && !(sup->k == K::Fence && sup->t == "(")) {
                Items pv = items_of(*sup);
                bool inverse = pv.size() == 2 && is_minus(*pv[0]) && is_num(*pv[1], "1");
                if (!inverse) return false;
            }
            b = strip(name.kids[0]);
        }
        if (b->k == K::Accent) b = strip(b->kids[0]);
        if (b->k == K::Word) return true;
        if (b->k != K::Letter) return false;
        const Node& body = f.kids[0];
        bool list = false;
        for (const Node& k : body.kids) list |= comma_sym(k);
        if (list || simple(body)) return true;
        static const std::string_view COEF = "abckmnrstxyz";
        return !(b->cp < 0x80 && COEF.find(static_cast<char>(b->cp)) != std::string_view::npos);
    }

    Say apply(const Node& name, Say s) {
        size_t k = next_index(i_);
        if (k >= b_) return s;
        // softmax(z)_i: the scripts of the argument's bracket come after it
        const Node *sub = nullptr, *sup = nullptr;
        const Node* fp = &at(k);
        if (fp->k == K::Scripts && fp->n == 0 && strip(fp->kids[0])->k == K::Fence) {
            if (fp->kids[1].k != K::Empty) sub = &fp->kids[1];
            if (fp->kids[2].k != K::Empty) sup = strip(fp->kids[2]);
            fp = strip(fp->kids[0]);
        }
        if (!applies(name, *fp)) return s;
        const Node& f = *fp;
        i_ = k + 1;
        bool open = false, simple_arg = false;
        Say r;
        if (name.k == K::Letter && name.cp == 'P' && has_relation(f.kids[0])) {  // P(X = x)
            r.s = "the probability that " + sp_.node(f.kids[0], c_).s;
            r.open = true;
            return r;
        }
        std::string a = sp_.args(f.kids[0], c_, &open, &simple_arg);
        r.s = a.empty() ? s.s : attach(s.s, s.open, "of " + a);
        r.open = open;
        if (sub) {
            Say S = sp_.node(*sub, c_);
            r.s = attach(r.s, r.open, "sub " + S.s);
            r.open = !S.simple;
        }
        if (sup) {  // f(x)^2: f of x, squared
            r.s = attach(r.s, true, sp_.sup_words(fp, sup, c_));
            r.open = false;
        }
        return r;
    }

    static bool operand_like(const Node& n) {
        switch (n.k) {
            case K::Letter: case K::Num: case K::Scripts: case K::Fence: case K::Frac: case K::Root: case K::Accent:
            case K::Set:
                return true;
            case K::Word: return !connective(n.t);
            case K::Sym: return n.s->cat == Cat::Const || n.s->cat == Cat::Func || n.s->cat == Cat::Big;
            default: return false;
        }
    }

    static bool has_relation(const Node& body) {
        for (const Node* q : items_of(body))
            if (sym_is(*q, Cat::Rel)) return true;
        return false;
    }

    void operand() {
        const Node& n = at(i_);
        bool quantity_before = last_quantity_;
        const Node* before = prev_ == OPERAND ? last_ : nullptr;
        bool open_before = !pieces_.empty() && pieces_.back().open;
        Say o;
        std::string fw;
        bool slash = slash_fraction(v_, i_, &fw);
        std::string unit = before ? unit_after(*before, n) : "";
        if (!unit.empty()) {  // 9.8\,\mathrm{m/s^2}: units after a number
            ++i_;
            add(unit);
            last_ = &n;
            return;
        }
        if (n.k == K::Word && n.t == "dollars" && i_ + 1 < b_ && at(i_ + 1).k == K::Num) {  // \$5
            add(number_words(at(i_ + 1).t) + " dollars");
            i_ += 2;
            prev_ = OPERAND;
            ++operands_;
            last_ = &n;
            return;
        }
        if (slash) {
            o.s = fw;
            i_ += 3;
        } else {
            o = operand_at();
        }
        if (o.s.empty()) return;
        last_quantity_ = o.s.compare(0, 13, "the quantity ") == 0;
        if (before) {
            bool mixed = before->k == K::Num && ((n.k == K::Frac && n.n == 0 &&
                                                  !common_fraction(n.kids[0], n.kids[1]).empty()) || slash);
            if (mixed) add("and");  // 2 1/2: two and one half
            else if (last_quantity_ || quantity_before || open_before || (before->k == K::Num && n.k == K::Num) ||
                     o.s.compare(0, 4, "the ") == 0)
                add("times");
        }
        add(o.s, o.open);
        prev_ = OPERAND;
        ++operands_;
        last_ = &n;
    }

    Say operand_at() {
        const Node& n = at(i_);
        Say s;
        switch (n.k) {
            case K::Num:
                ++i_;
                s.s = number_words(n.t);
                s.simple = true;
                return s;
            case K::Letter: {
                Say d;
                if (slash_derivative(d)) return d;
                ++i_;
                s.s = letter(n.cp);
                s.simple = true;
                return apply(n, s);
            }
            case K::Word: {
                ++i_;
                size_t k = next_index(i_);
                if (n.t == "the indicator" && k < b_ && at(k).k == K::Fence &&
                    (at(k).t == "{" || at(k).t == "(" || at(k).t == "[")) {  // \mathbb{1}\{y \neq k\}
                    i_ = k + 1;
                    Ctx ic = c_;
                    ic.set = false;
                    s.s = "the indicator of " + sp_.node(at(k).kids[0], ic).s;
                    s.open = true;
                    return s;
                }
                s.s = n.t;
                s.simple = n.t.find(' ') == std::string::npos;
                return apply(n, s);
            }
            case K::Set: ++i_; return set_words(n, nullptr);
            case K::Scripts: return scripts_at();
            case K::Frac: return frac_at();
            case K::Root: ++i_; return sp_.root(n, c_);
            case K::Accent: {
                ++i_;
                Say a = sp_.accent(n, c_);
                return apply(n, a);
            }
            case K::Fence: {
                ++i_;
                Ctx fc = c_;
                fc.interval = last_rel_in_;
                return sp_.fence(n, fc, false);
            }
            case K::Table: ++i_; return sp_.table(n, c_, "");
            case K::Row: ++i_; return sp_.node(n, c_);
            case K::Sym:
                switch (n.s->cat) {
                    case Cat::Const:
                    case Cat::Dots:
                        ++i_;
                        s.s = n.s->say;
                        s.simple = n.s->cat == Cat::Const;
                        return s;
                    case Cat::Func: return func_at(n, nullptr, nullptr);
                    case Cat::Big: return big_at(n, nullptr, nullptr);
                    case Cat::Prefix: {
                        Say d;
                        if (n.s->kind == P_PARTIAL && slash_derivative(d)) return d;
                        return prefix_at(n, nullptr, nullptr);
                    }
                    default: ++i_; return s;
                }
            default: ++i_; return s;
        }
    }

    // A unit written upright after a number ("kilograms"), or Celsius after degrees; "" if none.
    static std::string unit_after(const Node& before, const Node& n) {
        const Node* num = &before;
        if (before.k == K::Scripts) {
            const Node* p = strip(before.kids[2]);
            if (sym_kind(*p, C_DEGREE) && n.k == K::Letter && n.n == 1 && (n.cp == 'C' || n.cp == 'F'))
                return n.cp == 'C' ? "Celsius" : "Fahrenheit";
            num = strip(before.kids[0]);
            if (!(num->k == K::Num && num->t == "10")) return "";  // 3 \times 10^8\,\mathrm{m}
        }
        if (num->k != K::Num) return "";
        bool plural = !(before.k == K::Num && before.t == "1");
        std::string u;
        if (n.k == K::Word) {
            u = n.t;
        } else if (n.k == K::Letter && n.n == 1 && n.cp < 0x80) {
            u = std::string(1, static_cast<char>(n.cp));
        } else if (n.k == K::Scripts && n.kids[1].k == K::Empty && n.n == 0) {  // \mathrm{m}^2
            const Node *b = strip(n.kids[0]), *p = strip(n.kids[2]);
            if (!(p->k == K::Num && (p->t == "2" || p->t == "3"))) return "";
            if (b->k == K::Word) u = b->t;
            else if (b->k == K::Letter && b->n == 1 && b->cp < 0x80) u = std::string(1, static_cast<char>(b->cp));
            u += "^" + p->t;
        }
        return u.empty() ? "" : unit_words(u, plural);
    }

    Say set_words(const Node& n, const Node* sup) {
        Say s;
        if (sup) {
            Ctx dc = c_;
            dc.dims = true;
            s.s = letter(n.cp) + " " + sp_.node(*sup, dc).s;  // R n, R n by m
            return s;
        }
        switch (n.cp) {
            case 'R': s.s = "the real numbers"; break;
            case 'N': s.s = "the natural numbers"; break;
            case 'Z': s.s = "the integers"; break;
            case 'Q': s.s = "the rational numbers"; break;
            default: s.s = "the complex numbers"; break;
        }
        s.simple = true;
        return s;
    }

    Say scripts_at() {
        const Node& n = at(i_);
        const Node* base = strip(n.kids[0]);
        const Node* sub = n.kids[1].k == K::Empty ? nullptr : &n.kids[1];
        const Node* sup = n.kids[2].k == K::Empty ? nullptr : &n.kids[2];
        if (base->k == K::Sym && n.n == 0) {
            switch (base->s->cat) {
                case Cat::Big: return big_at(*base, sub, sup);
                case Cat::Func: return func_at(*base, sub, sup);
                case Cat::Prefix:
                    if (base->s->kind == P_NABLA || base->s->kind == P_PARTIAL) return prefix_at(*base, sub, sup);
                    break;
                default: break;
            }
        }
        if (base->k == K::Set && sup && !sub) {
            ++i_;
            return set_words(*base, sup);
        }
        ++i_;
        Say s = sp_.scripts(n, c_);
        return apply(n, s);
    }

    // "dy/dx", "∂f/∂x", "d/dx f": a derivative written with a slash.
    bool slash_derivative(Say& out) {
        size_t k = i_;
        bool partial = false;
        if (!diff_mark(at(k), &partial, nullptr)) return false;
        size_t x = 0;
        bool has_x = false;
        ++k;
        if (k + 1 < b_ && variable(at(k)) && sym_kind(at(k + 1), O_DIV)) {
            x = k++;
            has_x = true;
        }
        if (k + 2 >= b_ || !sym_kind(at(k), O_DIV)) return false;
        bool p2 = false;
        if (!diff_mark(at(k + 1), &p2, nullptr) || p2 != partial || !variable(at(k + 2))) return false;
        std::string head = partial ? "the partial derivative" : "the derivative";
        const Node& vn = at(k + 2);
        std::string var = sp_.node(vn.k == K::Scripts && vn.kids[2].k != K::Empty ? vn.kids[0] : vn, c_).s;
        i_ = k + 3;
        if (has_x) {
            out.s = head + " of " + sp_.node(at(x), c_).s + " with respect to " + var;
            return true;
        }
        size_t e = term_end(i_, true);
        out.s = head + " with respect to " + var;
        if (e > i_) {
            out.s += " of " + range(i_, e, c_).s;
            out.open = true;
            i_ = e;
        }
        return true;
    }

    Say frac_at() {
        const Node& n = at(i_++);
        if (n.n == 0) {
            Items ni = items_of(n.kids[0]), di = items_of(n.kids[1]);
            bool partial = false;
            const Node* order = nullptr;
            if (!ni.empty() && !di.empty() && diff_mark(*ni[0], &partial, &order)) {
                std::vector<std::string> vars;
                bool ok = di.size() % 2 == 0;
                for (size_t k = 0; k + 1 < di.size() && ok; k += 2) {
                    bool p2 = false;
                    ok = diff_mark(*di[k], &p2, nullptr) && p2 == partial && variable(*di[k + 1]);
                    if (ok) {
                        const Node* var = di[k + 1];
                        if (var->k == K::Scripts && var->kids[2].k != K::Empty) var = &var->kids[0];
                        vars.push_back(sp_.node(*var, c_).s);
                    }
                }
                std::string ord = order ? ordinal_words(*order) : "";
                if (ok && !vars.empty() && (order == nullptr || !ord.empty())) {
                    if (ord == "first") ord.clear();
                    std::string head = "the " + (ord.empty() ? std::string() : ord + " ") +
                                       (partial ? "partial derivative" : "derivative");
                    std::string wrt = vars[0];
                    for (size_t k = 1; k < vars.size(); ++k) wrt += (k + 1 == vars.size() ? " and " : ", ") + vars[k];
                    Say s;
                    if (ni.size() == 1) {  // d/dx: takes the term after it
                        size_t e = term_end(i_, true);
                        s.s = head + " with respect to " + wrt;
                        if (e > i_) {
                            s.s += " of " + range(i_, e, c_).s;
                            s.open = true;
                            i_ = e;
                        }
                        return s;
                    }
                    Items of(ni.begin() + 1, ni.end());
                    s.s = head + " of " + sp_.row(of, 0, of.size(), c_).s + " with respect to " + wrt;
                    return s;
                }
            }
        }
        return sp_.frac(n, c_);
    }

    // The argument after a function name: a bracketed group, or the term after it.
    bool take_argument(bool mult, std::string* a, bool* simple_arg, bool* open) {
        *open = false;
        size_t k = next_index(i_);
        const Node* f = k < b_ ? &at(k) : nullptr;
        pending_sup_ = nullptr;
        if (f && f->k == K::Scripts && f->n == 0 && f->kids[1].k == K::Empty && strip(f->kids[0])->k == K::Fence) {
            pending_sup_ = strip(f->kids[2]);  // E[X]^2: the power goes after the whole reading
            f = strip(f->kids[0]);
        }
        if (f && f->k == K::Fence && (f->t == "(" || f->t == "[")) {
            i_ = k + 1;
            *a = sp_.args(f->kids[0], c_, open, simple_arg);
            return !a->empty();
        }
        pending_sup_ = nullptr;
        size_t e = term_end(i_, mult);
        if (next_index(i_) >= e) return false;
        Say s = range(i_, e, c_);
        i_ = e;
        *a = s.s;
        *simple_arg = s.simple;
        *open = !s.simple;
        return !s.s.empty();
    }

    Say func_at(const Node& f, const Node* sub, const Node* sup) {
        ++i_;
        const Sym* s = f.s;
        std::string name(s->say);
        uint8_t kind = s->kind;
        // \arg\max_x: arg max
        if (kind == F_ARG) {
            size_t k = next_index(i_);
            if (k < b_) {
                const Node& nx = at(k);
                const Node* nb = nx.k == K::Scripts ? strip(nx.kids[0]) : &nx;
                if (nb->k == K::Sym && nb->s && (nb->s->say == "maximum" || nb->s->say == "minimum")) {
                    name = nb->s->say == "maximum" ? "arg max" : "arg min";
                    kind = F_OVER;
                    if (nx.k == K::Scripts) {
                        if (nx.kids[1].k != K::Empty) sub = &nx.kids[1];
                        if (nx.kids[2].k != K::Empty) sup = &nx.kids[2];
                    }
                    i_ = k + 1;
                }
            }
        }
        std::string a;
        bool simple_arg = false, open = false;
        bool mult = kind != F_TRIG && kind != F_LOG && kind != F_LN;
        bool has = take_argument(mult, &a, &simple_arg, &open);
        Say r;
        r.open = true;
        if (kind == F_TRIG || kind == F_LOG || kind == F_LN) {
            std::string pw, base, nth;
            bool inverse = false;
            if (sup) {
                const Node* p = strip(*sup);
                Items pv = items_of(*p);
                if (is_num(*p, "2")) pw = " squared";
                else if (is_num(*p, "3")) pw = " cubed";
                else if (pv.size() == 2 && is_minus(*pv[0]) && is_num(*pv[1], "1")) inverse = true;
                else nth = ordinal_words(*p);
                if (!inverse && pw.empty() && nth.empty()) pw = " to the power of " + sp_.node(*p, c_).s;
            }
            if (sub && kind != F_TRIG) base = " base " + sp_.node(*sub, c_).s;
            if (!has) {
                r.s = name + pw + base;
                r.open = false;
                return r;
            }
            if (inverse) r.s = "the inverse " + name + " of " + a;
            else if (!nth.empty()) r.s = "the " + nth + " power of " + name + " " + a;
            else if (!base.empty()) r.s = "the " + name + base + " of " + a;
            else if (simple_arg) r.s = name + pw + " " + a;
            else r.s = "the " + name + pw + " of " + a;
            r.open = !simple_arg || open;
            return r;
        }
        std::string head = "the " + name;
        if (sub) {
            Ctx lc = c_;
            lc.bounds = true;
            lc.lim = kind == F_LIMIT;
            head += (kind == F_LIMIT ? " as " : " over ") + sp_.node(*sub, lc).s;
        }
        if (!has) {
            r.s = sub ? head : name;
            r.open = sub != nullptr;
            return r;
        }
        r.s = head + " of " + a;
        if (name == "probability" && a.compare(0, 13, "the quantity ") == 0) r.s = "the probability that " + a.substr(13);
        r.open = kind == F_LIMIT || kind == F_OVER || kind == F_EXPECT || open || !simple_arg;
        if (pending_sup_) {
            r.s = attach(r.s, true, sp_.sup_words(&f, pending_sup_, c_));
            r.open = false;
            pending_sup_ = nullptr;
        }
        return r;
    }

    Say big_at(const Node& b, const Node* sub, const Node* sup) {
        ++i_;
        Ctx bc = c_;
        bc.bounds = true;
        std::string head = "the " + std::string(b.s->say);
        if (sub && sup) head += " from " + sp_.node(*sub, bc).s + " to " + sp_.node(*sup, bc).s;
        else if (sub) head += " over " + sp_.node(*sub, bc).s;
        else if (sup) head += " to " + sp_.node(*sup, bc).s;
        Say r;
        r.open = true;
        auto differential = [&](size_t k) {
            size_t q = next_index(k + 1);
            return q < b_ && !blank(at(k)) && diff_mark(at(k), nullptr, nullptr) && at(k).k != K::Scripts &&
                   !sym_kind(at(k), P_PARTIAL) && variable(at(q));
        };
        if (b.s->kind == B_INTEGRAL) {
            // the integrand runs to its differential: d x, \mathrm{d}x, d\theta
            size_t k = i_;
            bool found = false;
            for (; k < b_; ++k) {
                const Node& n = at(k);
                if (sym_is(n, Cat::Rel) || comma_sym(n) || wide_space(n)) break;
                if (differential(k)) {
                    found = true;
                    break;
                }
            }
            if (found) {
                std::string integrand = range(i_, k, c_).s;
                std::string diffs;
                while (k < b_ && differential(k)) {
                    size_t q = next_index(k + 1);
                    diffs += (diffs.empty() ? "" : " ") + letter('d') + " " + sp_.node(at(q), c_).s;
                    k = next_index(q + 1);
                }
                i_ = k;
                if (integrand.empty()) {  // \int dx\, f(x)
                    size_t e2 = term_end(i_, true);
                    integrand = range(i_, e2, c_).s;
                    i_ = e2;
                }
                r.s = head + (integrand.empty() ? "" : " of " + integrand) + ", " + diffs;
                return r;
            }
        }
        size_t e = term_end(i_, true);
        if (next_index(i_) < e) {
            r.s = head + " of " + range(i_, e, c_).s;
            i_ = e;
        } else {
            r.s = head;
        }
        return r;
    }

    Say prefix_at(const Node& p, const Node* sub, const Node* sup) {
        ++i_;
        Say r;
        r.open = true;
        if (p.s->kind == P_NABLA) {
            std::string head;
            size_t k = next_index(i_);
            const Node* nx = k < b_ ? &at(k) : nullptr;
            if (sup && is_num(*strip(*sup), "2")) {
                head = "the Laplacian of";
            } else if (nx && sym_is(*nx, Cat::Op) && (nx->s->cp == 0x22C5 || nx->s->cp == 0xB7)) {
                head = "the divergence of";
                i_ = k + 1;
            } else if (nx && sym_is(*nx, Cat::Op) && nx->s->cp == 0xD7) {
                head = "the curl of";
                i_ = k + 1;
            } else if (sub) {
                head = "the gradient with respect to " + sp_.node(*sub, c_).s + " of";
            } else {
                head = "the gradient of";
            }
            size_t e = term_end(i_, false);
            if (next_index(i_) >= e) {
                r.s = "del";
                r.open = false;
                return r;
            }
            r.s = head + " " + range(i_, e, c_).s;
            i_ = e;
            return r;
        }
        // ∂_x f: the partial derivative with respect to x of f
        if (sub) {
            std::string ord = sup ? ordinal_words(*sup) : "";
            std::string head = "the " + (ord.empty() ? std::string() : ord + " ") +
                               "partial derivative with respect to " + sp_.node(*sub, c_).s;
            size_t e = term_end(i_, false);
            if (next_index(i_) < e) {
                head += " of " + range(i_, e, c_).s;
                i_ = e;
            } else {
                r.open = false;
            }
            r.s = head;
            return r;
        }
        r.s = "partial";
        r.open = false;
        return r;
    }
};

Say Speaker::row(const Items& v, size_t a, size_t b, Ctx c) { return RowReader(*this, v, a, b, c).read(); }

Say Speaker::node(const Node& n, Ctx c) {
    if (n.k == K::Empty) return Say{};
    if (n.k == K::Table) return table(n, c, "");
    Items v = ptrs(n);
    return row(v, 0, v.size(), c);
}

std::string Speaker::args(const Node& body, Ctx c, bool* open, bool* simple_arg) {
    Ctx ac = c;
    ac.args = true;
    ac.set = false;
    Items v = ptrs(body);
    auto parts = split_commas(v);
    *simple_arg = false;
    if (parts.size() > 1) {
        std::string s;
        for (auto [a, e] : parts) {
            Say p = row(v, a, e, ac);
            if (!s.empty()) s += ", ";
            s += p.s;
        }
        *open = true;
        return s;
    }
    Say one = row(v, 0, v.size(), ac);
    *simple_arg = one.simple;
    bool given = false;
    for (const Node* q : v) given |= q->k == K::Bar;
    if (given) {
        *open = true;
        return one.s;
    }
    if (one.simple || one.single || (one.term && !one.open)) {  // one operand needs no "the quantity"
        *open = !one.simple;
        return one.s;
    }
    *open = true;
    return "the quantity " + one.s;
}

Say Speaker::scripts(const Node& n, Ctx c) {
    const Node& base = n.kids[0];
    const Node* sub = n.kids[1].k == K::Empty ? nullptr : &n.kids[1];
    const Node* sup = n.kids[2].k == K::Empty ? nullptr : &n.kids[2];
    const Node* b = strip(base);
    const Node* p = sup ? strip(*sup) : nullptr;
    // ‖x‖_2: the two norm of x
    // \{x_t\}_{t=1}^{T}: an indexed sequence, the subscript assigns the index (the T above is no transpose)
    if (b->k == K::Fence && sub) {
        bool assigns = false;
        for (const Node* q : items_of(*sub)) assigns |= sym_kind(*q, R_EQ);
        if (assigns) {
            Ctx bc = c;
            bc.bounds = true;
            bc.set = false;
            Say r;
            r.s = "the sequence of " + node(b->kids[0], bc).s + ", for " + node(*sub, bc).s +
                  (p ? " to " + node(*p, bc).s : "");
            r.open = true;
            return r;
        }
    }
    if (b->k == K::Fence && b->t == "\xE2\x80\x96" && sub) {
        Say r;
        r.s = "the " + node(*sub, c).s + " norm of " + node(b->kids[0], c).s;
        r.open = true;
        if (p) r.s += is_num(*p, "2") ? ", squared" : ", to the power of " + node(*p, c).s;
        return r;
    }
    if (b->k == K::Empty) {  // {}^{14}C: the script alone
        Say r;
        if (sub) r.s = node(*sub, c).s;
        if (p) r.s = attach(r.s, false, node(*p, c).s);
        return r;
    }
    if (b->k == K::Word && b->t == "the indicator" && sub && !p) {  // \mathbb{1}_{x > 0}
        Say r;
        bool rel = false;
        for (const Node* q : items_of(*sub)) rel |= sym_is(*q, Cat::Rel);
        r.s = (rel ? "the indicator that " : "the indicator of ") + node(*sub, c).s;
        r.open = true;
        return r;
    }
    bool letterish = b->k == K::Letter || b->k == K::Accent || b->k == K::Word;
    // f^{(n)}: the derivative of order n of f
    if (p && p->k == K::Fence && p->t == "(" && letterish && simple(p->kids[0]) && !sub) {
        Say r;
        std::string o = ordinal_words(p->kids[0]);
        r.s = (o.empty() ? "the derivative of order " + node(p->kids[0], c).s : "the " + o + " derivative") + " of " +
              node(base, c).s;
        return r;
    }
    Say B;
    if (b->k == K::Fence && (b->t == "(" || b->t == "[") && p) B = fence(*b, c, true);
    else if (b->k != K::Empty) B = node(base, c);
    std::string s = B.s;
    bool open = B.open;
    static const char* const PRIMES[] = {"", "prime", "double prime", "triple prime"};
    if (n.n) {
        s = attach(s, open, PRIMES[n.n]);
        open = false;
    }
    Say r;
    r.simple = simple(n);
    if (sub) {
        Ctx sc = c;
        sc.range = true;
        Say S = node(*sub, sc);
        s = attach(s, open, "sub " + S.s);
        open = !S.simple;
    }
    if (p) {
        s = attach(s, open, sup_words(b, p, c));
        open = false;
    }
    r.s = s;
    r.open = open;
    return r;
}

std::string Speaker::sup_words(const Node* b, const Node* p, Ctx c) {
    static const char* const PRIMES[] = {"", "prime", "double prime", "triple prime"};
    Items pv = items_of(*p);
    bool all_primes = !pv.empty() && pv.size() <= 3;
    for (const Node* q : pv) all_primes &= sym_is(*q, Cat::Prime) && q->s->say == "prime";
    bool inverse_base = (b->k == K::Letter && (upper(b->cp) || b->cp == 'f' || b->cp == 'g' || b->cp == 'h')) ||
                        b->k == K::Accent || b->k == K::Fence;
    if (sym_kind(*p, C_TOP) || (p->k == K::Letter && p->cp == 'T' && b->k != K::Num) ||
        (p->k == K::Word && p->t == "T"))
        return "transpose";
    if (sym_kind(*p, C_DEGREE)) return "degrees";
    if (sym_kind(*p, C_STAR)) return "star";
    if (sym_kind(*p, C_DAGGER)) return "dagger";
    if (sym_is(*p, Cat::Prime)) return std::string(p->s->say);
    if (all_primes) return PRIMES[pv.size()];
    if (pv.size() == 2 && is_minus(*pv[0]) && is_num(*pv[1], "1") && inverse_base) return "inverse";
    if (sym_is(*p, Cat::Op) && p->s->kind == O_ADD) return std::string(p->s->say);
    if (is_num(*p, "2")) return "squared";
    if (is_num(*p, "3")) return "cubed";
    if (simple_exponent(*p)) return "to the power of " + node(*p, c).s;
    return "raised to the " + node(*p, c).s + " power";
}

Say Speaker::frac(const Node& n, Ctx c) {
    Say r;
    if (n.n == 1) {  // binomial coefficient
        Say a = node(n.kids[0], c), b = node(n.kids[1], c);
        if (a.s.empty() || b.s.empty()) return a.s.empty() ? b : a;
        r.s = a.s + (a.simple ? " " : ", ") + "choose " + b.s;
        r.open = true;
        return r;
    }
    std::string common = common_fraction(n.kids[0], n.kids[1]);
    if (!common.empty()) {
        r.s = common;
        r.simple = true;
        return r;
    }
    Say a = node(n.kids[0], c), b = node(n.kids[1], c);
    if (a.s.empty() || b.s.empty()) return a.s.empty() ? b : a;
    // one operand on each side reads "a over b"; a fraction inside a fraction never does
    auto nested = [](const Node& x) { return strip(x)->k == K::Frac && !simple(x); };
    bool a_short = (a.simple || a.single) && !a.open && !nested(n.kids[0]);
    bool b_short = (b.simple || b.single) && !b.open && !nested(n.kids[1]);
    if (a_short && b_short) {
        r.s = a.s + " over " + b.s;
        r.open = !(a.simple && b.simple);
        return r;
    }
    r.open = true;
    r.term = false;
    if (b_short) r.s = a.s + ", all over " + b.s;
    else r.s = "the fraction with numerator " + a.s + ", and denominator " + b.s;
    return r;
}

Say Speaker::root(const Node& n, Ctx c) {
    Say body = node(n.kids[0], c);
    if (body.s.empty()) return body;
    const Node* idx = strip(n.kids[1]);
    std::string head = "the square root of";
    if (is_num(*idx, "3")) {
        head = "the cube root of";
    } else if (idx->k != K::Empty && !is_num(*idx, "2")) {
        std::string o = ordinal_words(*idx);
        head = o.empty() ? "the root of order " + node(*idx, c).s + " of" : "the " + o + " root of";
    }
    Say r;
    r.s = head + " " + body.s;
    r.open = true;  // the end of the radicand is a pause: the square root of x, plus one
    return r;
}

Say Speaker::accent(const Node& n, Ctx c) {
    const Node& body = n.kids[0];
    Say b = node(body, c);
    if (b.s.empty()) return b;
    Say r;
    r.simple = b.simple;
    const std::string& t = n.t;
    if (t == "plain" || t == "brace") return b;
    if (t == "vec") {
        r.s = "vector " + b.s;
        r.open = b.open;
        return r;
    }
    if (t == "bar") {
        Items v = items_of(*strip(body));
        if (v.size() == 2 && v[0]->k == K::Letter && v[1]->k == K::Letter && upper(v[0]->cp) && upper(v[1]->cp)) {
            r.s = "the line segment " + b.s;
            return r;
        }
        if (!b.simple) {
            r.s = "the complex conjugate of " + b.s;
            r.open = true;
            r.simple = false;
            return r;
        }
    }
    static const char* const SUFFIX[][2] = {{"hat", "hat"},     {"bar", "bar"},         {"tilde", "tilda"},
                                            {"dot", "dot"},     {"ddot", "double dot"}, {"dddot", "triple dot"},
                                            {"check", "check"}, {"breve", "breve"},     {"mathring", "ring"}};
    for (const auto& sfx : SUFFIX)
        if (t == sfx[0]) {
            r.s = attach(b.s, b.open, sfx[1]);
            return r;
        }
    return b;
}

Say Speaker::fence(const Node& f, Ctx c, bool force_quantity) {
    const Node& body = f.kids[0];
    const std::string& o = f.t.empty() ? f.u : f.t;
    Say r;
    r.open = true;
    Items v = ptrs(body);
    if (o == "{") {
        Ctx sc = c;
        sc.set = true;
        sc.args = false;
        size_t split = v.size();
        for (size_t k = 0; k < v.size(); ++k)
            if (v[k]->k == K::Bar || (sym_is(*v[k], Cat::Punct) && v[k]->s->say == ":")) {
                split = k;
                break;
            }
        if (split < v.size()) {
            Ctx lc = sc;
            lc.bounds = true;
            r.s = "the set of all " + row(v, 0, split, lc).s + " such that " + row(v, split + 1, v.size(), sc).s;
            return r;
        }
        std::string s;
        for (auto [a, e] : split_commas(v)) {
            Say p = row(v, a, e, sc);
            if (p.s.empty()) continue;
            if (!s.empty()) s += ", ";
            s += p.s;
        }
        r.s = s.empty() ? "the empty set" : "the set " + s;
        r.open = !s.empty();
        return r;
    }
    if (o == "|") {
        const Node* in = strip(body);
        if (in->k == K::Table) {
            r.s = "the determinant of " + table(*in, c, "matrix").s;
            return r;
        }
        Say b = node(body, c);
        r.s = "the absolute value of " + b.s;
        r.open = !b.simple;
        return r;
    }
    if (o == "\xE2\x80\x96" || o == "floor" || o == "ceil") {
        Say b = node(body, c);
        r.s = std::string(o == "floor" ? "the floor of " : o == "ceil" ? "the ceiling of " : "the norm of ") + b.s;
        r.open = !b.simple;
        return r;
    }
    if (o == "<") {
        auto parts = split_commas(v);
        if (parts.size() == 2) {
            r.s = "the inner product of " + row(v, parts[0].first, parts[0].second, c).s + " and " +
                  row(v, parts[1].first, parts[1].second, c).s;
            return r;
        }
        r.s = "the expected value of " + node(body, c).s;
        return r;
    }
    // ( ) and [ ]
    const Node* in = strip(body);
    if (in->k == K::Table) return table(*in, c, "matrix");
    Items it = items_of(body);
    if (!it.empty() && sym_is(*it[0], Cat::Op) && it[0]->s->say == "mod") {  // (mod n)
        Say m = node(body, c);
        m.open = false;
        return m;
    }
    auto parts = split_commas(v);
    if (parts.size() > 1) {
        std::vector<std::string> items;
        for (auto [a, e] : parts) items.push_back(row(v, a, e, c).s);
        if (parts.size() == 2 && (f.t == "[" || f.u == "[" || f.t != f.u || c.interval)) {
            r.s = "the interval from " + items[0] + " to " + items[1];
            return r;
        }
        std::string s;
        for (const auto& x : items) s += (s.empty() ? "" : ", ") + x;
        r.s = s;
        r.term = false;
        return r;
    }
    Say b = node(body, c);
    // before an exponent only a number or a letter goes without "the quantity"
    bool atom = it.size() == 1 && (it[0]->k == K::Num || it[0]->k == K::Letter || it[0]->k == K::Accent ||
                                   (it[0]->k == K::Scripts && simple(*it[0])));
    if (force_quantity ? atom : (b.simple || (b.term && !b.open))) return b;
    r.s = "the quantity " + b.s;
    r.term = false;
    return r;
}

Say Speaker::table(const Node& t, Ctx c, const std::string& as) {
    static const char* const MATRICES[] = {"matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
                                           "smallmatrix", "pmatrix*", "bmatrix*"};
    const std::string& env = as.empty() ? t.t : as;
    bool matrix = false, det = t.t == "vmatrix" || t.t == "Vmatrix";
    for (const char* m : MATRICES) matrix |= env == m;
    bool cases = env == "cases" || env == "dcases" || env == "rcases" || env == "cases*";
    Say r;
    r.open = true;
    r.term = false;
    if (matrix || cases) {
        std::vector<std::vector<std::string>> rows;
        bool dots = false;
        size_t cols = 0;
        for (const Node& rw : t.kids) {
            std::vector<std::string> cells;
            for (const Node& cell : rw.kids) {
                for (const Node* q : items_of(cell)) dots |= sym_is(*q, Cat::Dots);
                cells.push_back(node(cell, c).s);
            }
            cols = std::max(cols, cells.size());
            rows.push_back(std::move(cells));
        }
        if (matrix) {
            std::string s = det ? "the determinant of the " : "the ";
            if (rows.size() == 1 || cols == 1) {
                std::string cells;
                for (const auto& rw : rows)
                    for (const auto& cell : rw) cells += (cells.empty() ? "" : ", ") + cell;
                r.s = s + (rows.size() == 1 ? "row vector " : "column vector ") + cells;
                return r;
            }
            if (!dots) s += rv::cardinal(rows.size()) + " by " + rv::cardinal(cols) + " ";
            s += "matrix";
            for (size_t k = 0; k < rows.size(); ++k) {
                s += (k == 0 ? ", row " : "; row ") + rv::cardinal(k + 1) + ":";
                std::string cells;
                for (const auto& cell : rows[k]) cells += (cells.empty() ? " " : ", ") + cell;
                s += cells;
            }
            r.s = s;
            return r;
        }
        std::string s = rv::cardinal(rows.size()) + (rows.size() == 1 ? " case" : " cases");
        for (size_t k = 0; k < rows.size(); ++k) {
            s += (k == 0 ? ": case " : "; case ") + rv::cardinal(k + 1) + ",";
            const auto& rw = rows[k];
            if (!rw.empty()) s += " " + rw[0];
            if (rw.size() > 1 && !rw[1].empty()) {
                std::string first = lower_first_word(rw[1]);
                static const char* const LEAD[] = {"if", "for", "when", "where", "otherwise", "else", "unless", "as",
                                                   "provided", "whenever", "with", "and", "or", "on", "in", "at"};
                bool lead = false;
                for (const char* l : LEAD) lead |= first == l;
                s += (lead ? " " : " if ") + rw[1];
            }
        }
        r.s = s;
        return r;
    }
    // lines: aligned equations, gathered lines, arrays; the cells of a row read as one
    std::string s;
    for (const Node& rw : t.kids) {
        Items items;
        for (const Node& cell : rw.kids) {
            if (cell.k == K::Row)
                for (const Node& k : cell.kids) items.push_back(&k);
            else
                items.push_back(&cell);
        }
        Say line = row(items, 0, items.size(), c);
        if (line.s.empty()) continue;
        if (!s.empty()) s += c.bounds ? " and " : "; ";
        s += line.s;
    }
    r.s = s;
    r.open = t.kids.size() > 1;
    return r;
}

// Only words and , . ; : go out; spaces and doubled marks tidied.
std::string finish(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char ch : in) {
        bool word = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || digit(ch) || ch == '\'' || ch == '-' ||
                    static_cast<unsigned char>(ch) >= 0x80;
        if (word) {
            out += ch;
            continue;
        }
        if (ch == ',' || ch == ';' || ch == ':' || ch == '.') {
            while (!out.empty() && out.back() == ' ') out.pop_back();
            if (out.empty()) continue;
            char last = out.back();
            if (last == ',' || last == ';' || last == ':' || last == '.') {
                if (ch != ',' && last == ',') out.back() = ch;  // ", ;" -> ";"
                continue;
            }
            out += ch;
            continue;
        }
        if (!out.empty() && out.back() != ' ') out += ' ';  // a space, or anything else
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == ',' || out.back() == ';' || out.back() == ':'))
        out.pop_back();
    return out;
}

}  // namespace

std::string speak(const Node& root) {
    Speaker sp;
    return finish(sp.node(root, Ctx{}).s);
}

}  // namespace mth
