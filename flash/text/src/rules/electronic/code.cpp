// Code and identifiers: snake_case, camelCase and UPPER_CASE names, dotted names, assignments, quoted
// configuration keys, inline code in backticks, command-line flags, function calls, placeholders, keyboard
// shortcuts, short hex identifiers, and long random strings kept as written.
#include "rules/electronic/common.h"

namespace {

using namespace rv::el;
using rv::Doc;
using rv::Kind;

// End of the identifier at b ([A-Za-z0-9_]).
uint32_t ident_end(std::string_view t, uint32_t b) {
    uint32_t e = b;
    while (e < t.size() && idc(t[e])) ++e;
    return e;
}

// Start of the identifier that ends at e.
uint32_t ident_start(std::string_view t, uint32_t e) {
    uint32_t b = e;
    while (b > 0 && idc(t[b - 1])) --b;
    return b;
}

// A dotted name goes on at e: ".lower", "._x", ".CamelCase", ".UPPER"; a ".Word" with one capital starts a
// sentence after a missing space ("user_id.Then").
bool dotted_next(std::string_view t, uint32_t e) {
    if (e + 1 >= t.size() || t[e] != '.') return false;
    char c = t[e + 1];
    if (lo(c) || c == '_') return true;
    if (!up(c)) return false;
    uint32_t q = e + 2;
    bool inner = false, lower = false;
    while (q < t.size() && idc(t[q])) {
        if (up(t[q])) inner = true;
        else if (lo(t[q])) lower = true;
        ++q;
    }
    return inner || !lower;
}

bool snake(std::string_view s) {
    if (s.size() < 3 || s.front() == '_' || s.back() == '_') return false;
    bool us = false, alpha = false;
    for (size_t k = 0; k < s.size(); ++k) {
        if (s[k] == '_') {
            if (s[k + 1] == '_') return false;
            us = true;
        } else if (al(s[k])) {
            alpha = true;
        }
    }
    return us && alpha;
}

bool has_lower(std::string_view s) {
    for (char c : s)
        if (lo(c)) return true;
    return false;
}

Words ident_mode(std::string_view s) { return has_lower(s) ? Words::Code : Words::Upper; }

const rv::WordSet CAMEL_NOT{"arxiv", "biorxiv", "chemrxiv", "medrxiv", "mmhg", "psyarxiv"};

// lower camelCase: two or more lower-case letters, then words with a capital ("maxTokens", "customerId").
bool camel(std::string_view s) {
    if (s.size() < 4 || !lo(s[0]) || !lo(s[1])) return false;
    bool hump = false;
    for (size_t k = 1; k < s.size(); ++k) {
        if (!an(s[k])) return false;
        if (up(s[k]) && k + 1 < s.size() && lo(s[k + 1])) hump = true;
    }
    if (!hump && s.size() > 4 && s.substr(s.size() - 2) == "ID" && lo(s[s.size() - 3])) hump = true;  // userID
    return hump && !CAMEL_NOT.contains(lower_s(s));
}

// snake_case and UPPER_CASE identifiers: max_batch_size -> max batch size, HF_TOKEN -> HF token.
RV_RULE("electronic", snake_case) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    each_mark(doc, '_', [&](size_t k) {
        uint32_t u = doc[k].start;
        if (u < last) return;
        uint32_t b = ident_start(t, u), e = ident_end(t, u);
        last = e;
        std::string_view s = t.substr(b, e - b);
        if (!snake(s) || !al(s[0])) return;
        if (!left_edge(doc, b) && !(b > 0 && (t[b - 1] == '{' || t[b - 1] == '$'))) return;
        if (!right_edge(doc, e)) return;
        if (dotted_next(t, e)) return;  // a dotted name
        if (e + 1 < t.size() && t[e] == '-' && al(t[e + 1])) return;
        add_lc(doc, out, b, e, ident_words(s, ident_mode(s)), 80);
    });
}

// lower camelCase identifiers: maxTokens -> max tokens, customerId -> customer ID.
RV_RULE("electronic", camel_case) {
    const std::string_view t = doc.text;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].casing != rv::MIXED || doc[i].kind != Kind::Word) continue;
        uint32_t b = doc[i].start;
        if (!left_edge(doc, b)) continue;
        uint32_t e = b;
        while (e < t.size() && an(t[e])) ++e;
        std::string_view s = t.substr(b, e - b);
        if (!camel(s) || !right_edge(doc, e)) continue;
        if (dotted_next(t, e)) continue;
        if (e < t.size() && (t[e] == ':' || t[e] == '-') && e + 1 < t.size() && dg(t[e + 1])) continue;  // arXiv:2401
        add_lc(doc, out, b, e, ident_words(s, Words::Code), 78);
    }
}

const rv::WordSet MODULES{"argparse", "asyncio", "collections", "csv", "dataclasses", "datetime", "errors",
                          "functools", "io", "itertools", "jax", "json", "keras", "logging", "math", "multiprocessing",
                          "nn", "np", "numpy", "os", "pandas", "pathlib", "pd", "pickle", "pillow", "random", "re",
                          "requests", "scipy", "shutil", "sklearn", "socket", "statsmodels", "subprocess", "sys",
                          "tf", "threading", "time", "torch", "transformers", "typing", "unittest"};

// Words that start a sentence after a missing space ("local time.The reminder"), never a class name.
const rv::WordSet STARTERS{"after", "all", "an", "and", "as", "at", "but", "each", "every", "for", "he", "her", "his",
                           "if", "in", "it", "its", "most", "my", "no", "on", "once", "one", "our", "she", "so",
                           "some", "that", "the", "their", "then", "there", "these", "they", "this", "those", "to",
                           "we", "when", "while", "with", "you", "your"};

// Dotted names in code: torch.nn.Linear -> torch dot NN dot Linear, os.path.join -> OS dot path dot join.
RV_RULE("electronic", dotted_name) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    each_mark(doc, '.', [&](size_t k) {
        if (k == 0 || !doc.word(k - 1) || !doc.joined(k) || !glued(doc, k) || doc[k].start < last) return;
        uint32_t b = ident_start(t, doc[k].start);  // node_status.get starts at "node"
        if (!lo(t[b]) || !left_edge(doc, b)) return;
        std::string_view parts[16];
        int n = 0;
        uint32_t p = b, e = b;
        bool ok = true;
        while (true) {
            uint32_t q = ident_end(t, p);
            if (q == p) break;
            std::string_view part = t.substr(p, q - p);
            if (n < 16) parts[n] = part;
            ++n;
            if (part.size() < 2 || all_digits(part) || !(al(part[0]) || part[0] == '_')) ok = false;
            e = q;
            if (q + 1 < t.size() && t[q] == '.' && (al(t[q + 1]) || t[q + 1] == '_')) p = q + 1;
            else break;
        }
        last = e;
        if (n < 2 || n > 16 || !ok || !right_edge(doc, e)) return;
        std::string_view end_part = parts[n - 1];
        bool call = e < t.size() && t[e] == '(';
        // code, not a domain: a module name first, or a part with a capital or an underscore (tf.keras.Model.fit)
        bool code = MODULES.contains(parts[0]);
        for (int c = 0; c < n && c < 16 && !code; ++c)
            for (char ch : parts[c]) code |= up(ch) || ch == '_';
        if (!call && (is_ext(end_part) || (is_tld(end_part) && !code))) return;  // a file name or a domain
        if (n == 2 && up(end_part[0])) {
            bool inner_cap = false;
            for (size_t c = 1; c < end_part.size(); ++c)
                if (up(end_part[c])) inner_cap = true;
            if (!MODULES.contains(parts[0]) && (!inner_cap || all_upper(end_part))) return;  // ports.Its
            if (!inner_cap && STARTERS.contains(lower_s(end_part))) return;                  // time.The
        }
        std::string spoken;
        for (int c = 0; c < n; ++c) {
            if (c) spoken += " dot ";
            spoken += code_part(parts[c]);
        }
        add_lc(doc, out, b, e, spoken, 81);
    });
}

// Indexes with a variable: lambda[t] -> lambda t, matrix[i][j] -> matrix i j, data[idx] -> data idx.
const rv::WordSet INDEX_VARS{"i", "idx", "index", "j", "k", "key", "n", "t", "x", "y"};

// Indexes: x[0] -> x zero, tokens[3] -> tokens three (a bracket after a word that roman_citations does not read
// as a citation); a variable as written: lambda[t] -> lambda t.
RV_RULE("electronic", index) {
    each_mark(doc, '[', [&](size_t k) {
        // a variable: lambda[t], matrix[i][j]
        if (k > 0 && doc.joined(k) && doc.word(k + 1) && doc.joined(k + 1) && doc.punct(k + 2, ']') &&
            doc.joined(k + 2) && (doc.word(k - 1) || doc.punct(k - 1, ']')) && INDEX_VARS.contains(doc.str(k + 1))) {
            out.tokens(k, k + 2, std::string(doc.str(k + 1)), 55);
            return;
        }
        if (k == 0 || !doc.joined(k) || !doc.word(k - 1) || !doc.number(k + 1) || !doc.joined(k + 1) ||
            !doc.punct(k + 2, ']') || !doc.joined(k + 2) || doc.str(k + 1).size() > 3)
            return;
        std::string words = rv::read_number(doc.str(k + 1));
        if (!words.empty()) out.tokens(k, k + 2, words, 55);
    });
}

// Relative git refs: HEAD~3 -> HEAD tilde three, HEAD^ -> HEAD caret.
RV_RULE("electronic", git_ref) {
    each_mark(doc, '~', [&](size_t k) {
        if (k == 0 || !doc.joined(k) || doc.str(k - 1) != "HEAD") return;
        if (doc.number(k + 1) && doc.joined(k + 1) && doc.str(k + 1).size() <= 3)
            out.tokens(k, k + 1, "tilde " + num_words(doc.str(k + 1)), 75);
        else if (k + 1 >= doc.size() || !doc.joined(k + 1))
            out.tokens(k, k, "tilde", 75);
    });
    each_mark(doc, '^', [&](size_t k) {
        if (k == 0 || !doc.joined(k) || doc.str(k - 1) != "HEAD") return;
        if (doc.number(k + 1) && doc.joined(k + 1) && doc.str(k + 1).size() <= 2)
            out.tokens(k, k + 1, "caret " + num_words(doc.str(k + 1)), 75);
        else if (k + 1 >= doc.size() || !doc.joined(k + 1) || doc[k + 1].kind == Kind::Punct)
            out.tokens(k, k, "caret", 75);
    });
}

}  // namespace

namespace rv::el {

// Scope operators of C++, Rust and Ruby: std::vector -> std colon colon vector, Foo::bar -> Foo colon colon bar.
void scope_operator_at(const Doc& doc, rv::Out& out, size_t k) {
    if (k == 0 || !doc.punct(k + 1, ':') || !doc.joined(k) || !doc.joined(k + 1) || !doc.joined(k + 2)) return;
    if (!doc.word(k - 1) || !doc.word(k + 2)) return;
    if (doc.punct(k + 3, ':') && doc.joined(k + 3) && !doc.punct(k + 4, ':')) return;
    out.tokens(k, k + 1, "colon colon", 70);
}

}  // namespace rv::el

namespace {

// Assignments: ENABLE_VERBALIZATION=1 -> enable verbalization equals (the value is read by its own rule),
// batch_size=8 -> batch size equals; PORT=9090 -> port equals ninety ninety.
RV_RULE("electronic", assignment) {
    const std::string_view t = doc.text;
    each_mark(doc, '=', [&](size_t k) {
        uint32_t eq = doc[k].start;
        if (eq + 1 >= t.size() || t[eq + 1] == '=' || t[eq + 1] == ' ' || t[eq + 1] == '\n' || eq == 0) return;
        if (t[eq - 1] == '=' || t[eq - 1] == '!' || t[eq - 1] == '<' || t[eq - 1] == '>') return;
        uint32_t e = eq;
        if (t[e - 1] == '`') --e;  // `key`=5
        uint32_t b = ident_start(t, e);
        if (e - b < 2 || !al(t[b])) return;
        if (e < eq && !(b > 0 && t[b - 1] == '`')) return;
        if (b > 0 && t[b - 1] == '-') return;  // --flag=value: the flag rule
        if (!left_edge(doc, b)) return;
        std::string_view key = t.substr(b, e - b);
        std::string spoken = ident_words(key, ident_mode(key));
        append_word(spoken, "equals");
        uint32_t end = eq + 1;
        std::string l = lower_s(key);
        if (l == "port" || (l.size() > 5 && l.substr(l.size() - 5) == "_port")) {
            uint32_t r = end;
            while (r < t.size() && dg(t[r])) ++r;
            if (r > end && r - end <= 5 && right_edge(doc, r)) {
                append_word(spoken, port_words(t.substr(end, r - end)));
                end = r;
            }
        }
        add_lc(doc, out, b, end, spoken, 82);
    });
}

// A JSON value at v after spaces: a number, true, false, null or a quoted string.
bool literal_value(std::string_view t, uint32_t v) {
    while (v < t.size() && t[v] == ' ') ++v;
    if (v >= t.size()) return false;
    if (t[v] == '"' || dg(t[v]) || (t[v] == '-' && v + 1 < t.size() && dg(t[v + 1]))) return true;
    for (std::string_view w : {"true", "false", "null"})
        if (t.substr(v, w.size()) == w && (v + w.size() >= t.size() || !idc(t[v + w.size()]))) return true;
    return false;
}

// Quoted configuration keys: "use_verbalization": true -> use verbalization set to true; the quotes of a
// string value are left out.
RV_RULE("electronic", config_key) {
    const std::string_view t = doc.text;
    uint32_t set_end = 0;  // end of the last key read after the verb "set"
    each_mark(doc, '"', [&](size_t k) {
        uint32_t q = doc[k].start;
        if (!left_edge(doc, q)) return;
        uint32_t s = q + 1, e = s;
        int spaces = 0;  // "Sync interval": 15 -> Sync interval set to fifteen
        while (e < t.size() && (idc(t[e]) || t[e] == '-' || (t[e] == ' ' && e > s && t[e - 1] != ' ' && ++spaces <= 4)))
            ++e;
        if (e == s || e + 1 >= t.size() || t[e] != '"' || t[e + 1] != ':' || !al(t[s]) || t[e - 1] == ' ') return;
        uint32_t colon = e + 1;
        std::string_view key = t.substr(s, e - s);
        if (spaces && !literal_value(t, colon + 1)) return;
        // after the verb: Set "timeout": 30 and "retries": 3 -> Set timeout to thirty and retries to three
        size_t pw = doc.prev_word(k);
        bool after_set = pw != Doc::npos && doc.is(pw, "set");
        if (!after_set && pw != Doc::npos && doc.is(pw, "and") && set_end && q > set_end && q - set_end < 40 &&
            t.substr(set_end, q - set_end).find_first_of(".;\n") == std::string_view::npos)
            after_set = true;
        set_end = after_set ? colon + 1 : 0;
        std::string words = spaces ? std::string(key) : ident_words(key, ident_mode(key));
        add_lc(doc, out, q, colon + 1, words + (after_set ? " to" : " set to"), 82);
        uint32_t v = colon + 1;
        while (v < t.size() && t[v] == ' ') ++v;
        if (v < t.size() && t[v] == '"') {
            uint32_t close = v + 1;
            while (close < t.size() && close - v < 200 && t[close] != '"' && t[close] != '\n') ++close;
            if (close < t.size() && t[close] == '"' && close > v + 1) {
                out.add(v, v + 1, "", 82);
                out.add(close, close + 1, "", 82);
            }
        }
    });
}

const rv::WordSet PARTICLES{"away", "back", "by", "down", "in", "off", "on", "out", "over", "through", "up"};

// A kebab-case key at the start of a line or with a plain value: "max-tokens: 512" -> "max tokens: five
// hundred twelve"; in running text "the request-timeout setting" stays.
RV_RULE("electronic", kebab_key) {
    const std::string_view t = doc.text;
    each_mark(doc, ':', [&](size_t k) {
        uint32_t e = doc[k].start;
        if (e == 0 || e + 1 >= t.size() || t[e + 1] != ' ' || !an(t[e - 1])) return;
        uint32_t b = e;
        bool hy = false, upper = false;
        while (b > 0 && (an(t[b - 1]) || t[b - 1] == '-')) {
            if (t[b - 1] == '-') hy = true;
            if (up(t[b - 1])) upper = true;
            --b;
        }
        if (!hy || upper || !al(t[b]) || !left_edge(doc, b)) return;
        size_t i = tok_at(doc, b);
        bool line = i == 0 || (i < doc.size() && doc[i].newline_before) || (b > 0 && t[b - 1] == '`');
        if (!line) {
            // in running text only with a plain value: "retry-limit: 4", "dry-run: true"; never a compound
            // that ends in a particle (warm-up: 11, follow-up: 23)
            uint32_t h = e;
            while (h > b && t[h - 1] != '-') --h;
            if (PARTICLES.contains(t.substr(h, e - h))) return;
            uint32_t v = e + 2, w = v;
            while (w < t.size() && an(t[w])) ++w;
            std::string_view val = t.substr(v, w - v);
            if (!(all_digits(val) || val == "true" || val == "false" || val == "null" || val == "on" || val == "off"))
                return;
        }
        add_lc(doc, out, b, e, ident_words(t.substr(b, e - b), Words::Code), 80);
    });
}

// Inline code: backticks are left out; a kebab-case name inside them is read as words.
RV_RULE("electronic", backticks) {
    const std::string_view t = doc.text;
    size_t open = Doc::npos;
    each_mark(doc, '`', [&](size_t i) {
        bool fence = (i > 0 && doc.punct(i - 1, '`') && doc.joined(i)) ||
                     (i + 1 < doc.size() && doc.punct(i + 1, '`') && doc.joined(i + 1));
        if (fence) return;
        // a backtick typed for an apostrophe: "it`s", "don`t"
        if (i > 0 && doc.joined(i) && doc.word(i - 1) && doc.word(i + 1) && doc.joined(i + 1) &&
            (doc.is(i + 1, "s") || doc.is(i + 1, "t") || doc.is(i + 1, "ll") || doc.is(i + 1, "re") ||
             doc.is(i + 1, "ve") || doc.is(i + 1, "d") || doc.is(i + 1, "m")) &&
            !(i + 2 < doc.size() && doc.joined(i + 2) && doc.punct(i + 2, '`')))
            return;
        out.tokens(i, i, "", 80);
        if (open != Doc::npos &&
            t.substr(doc[open].end, doc[i].start - doc[open].end).find('\n') != std::string_view::npos)
            open = Doc::npos;  // a pair never spans lines
        if (open == Doc::npos) {
            open = i;
            return;
        }
        // kebab-case names between the pair
        for (size_t k = open + 1; k < i; ++k) {
            if (!doc.word(k)) continue;
            uint32_t b = doc[k].start;
            if (!(space_before(doc, b) || t[b - 1] == '`')) continue;
            uint32_t e = b;
            bool hy = false;
            while (e < doc[i].start && (an(t[e]) || t[e] == '-' || t[e] == '_')) {
                if (t[e] == '-') hy = true;
                ++e;
            }
            if (!hy || t[e - 1] == '-' || !right_edge(doc, e)) continue;
            if (e < t.size() && (t[e] == '.' || t[e] == '/') && e + 1 < t.size() && an(t[e + 1])) continue;
            add_lc(doc, out, b, e, ident_words(t.substr(b, e - b), Words::Code), 80);
            skip_to(doc, k, e);
        }
        open = Doc::npos;
    });
}

const rv::WordSet SUFFIXES{"al", "an", "ed", "er", "es", "est", "ful", "ic", "ing", "ion", "ise", "ish", "ism",
                           "ist", "ity", "ive", "ize", "ly", "ous", "th"};

// Command-line flags: --no-cache -> dash dash no cache, -v -> dash V, --format=csv -> dash dash format equals CSV.
RV_RULE("electronic", flag) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    each_mark(doc, '-', [&](size_t i) {
        uint32_t b = doc[i].start;
        if (b < last || i + 1 >= doc.size() || !doc.joined(i + 1) || !left_edge(doc, b)) return;
        if (doc.punct(i + 1, '-') && i + 2 < doc.size() && doc.joined(i + 2) && doc.word(i + 2)) {
            uint32_t s = doc[i + 2].start, e = s;
            while (e < t.size() && (an(t[e]) || t[e] == '-' || t[e] == '_')) ++e;
            while (e > s && (t[e - 1] == '-' || t[e - 1] == '_')) --e;
            std::string_view name = t.substr(s, e - s);
            std::string spoken = "dash dash";
            size_t k = 0;
            while (k < name.size()) {
                size_t j = k;
                while (j < name.size() && an(name[j])) ++j;
                std::string_view w = name.substr(k, j - k);
                if (!w.empty()) append_word(spoken, all_digits(w) ? num_words(w) : std::string(w));
                k = j + 1;
            }
            uint32_t end = e;
            bool format = lower_s(name).find("format") != std::string::npos;
            if (e < t.size() && t[e] == '=') {
                append_word(spoken, "equals");
                end = e + 1;
                if (format) {
                    uint32_t v = end;
                    while (v < t.size() && an(t[v])) ++v;
                    if (v > end && is_ext(t.substr(end, v - end)) && right_edge(doc, v)) {
                        append_word(spoken, ext_word(t.substr(end, v - end)));
                        end = v;
                    }
                }
            } else if (!right_edge(doc, e)) {
                return;
            } else if (format && e + 1 < t.size() && t[e] == ' ') {
                uint32_t v = e + 1, w = v;
                while (w < t.size() && an(t[w])) ++w;
                if (w > v && is_ext(t.substr(v, w - v)) && right_edge(doc, w) && all_lower(t.substr(v, w - v)))
                    out.add(v, w, ext_word(t.substr(v, w - v)), 70);
            }
            add_lc(doc, out, b, end, spoken, 80);
            last = end;
            return;
        }
        if (!doc.word(i + 1)) return;
        std::string_view w = doc.str(i + 1);
        uint32_t e = doc[i + 1].end;
        if (!right_edge(doc, e)) return;
        if (e < t.size() && (t[e] == '-' || t[e] == '\'' || t[e] == '/' || t[e] == '.') && e + 1 < t.size() &&
            an(t[e + 1]))
            return;
        bool ok = (w.size() == 1 && al(w[0])) || (w.size() <= 3 && all_lower(w) && !SUFFIXES.contains(w));
        if (!ok) return;
        add_lc(doc, out, b, e, "dash " + upper_s(w), 80);
    });
}

// Function calls with empty parentheses: generate() -> generate.
RV_RULE("electronic", call) {
    each_mark(doc, '(', [&](size_t i) {
        if (i == 0 || !doc.punct(i + 1, ')') || !doc.joined(i) || !doc.joined(i + 1)) return;
        if (doc[i - 1].kind == Kind::Punct && !doc.punct(i - 1, '_')) return;
        out.tokens(i, i + 1, "", 80);
    });
}

// Single words in angle brackets that are placeholders, not markup tags.
const rv::WordSet PLACEHOLDER_WORDS{"branch", "bucket", "command", "date", "dir", "directory", "email", "file",
                                    "filename", "host", "hostname", "id", "key", "model", "name", "password",
                                    "path", "port", "project", "query", "region", "repo", "secret", "tag",
                                    "token", "url", "user", "username", "value", "version"};

void placeholder_at(const Doc& doc, rv::Out& out, size_t i, bool angle) {
    const std::string_view t = doc.text;
    if (i + 2 >= doc.size() || !doc.joined(i + 1) || doc[i + 1].kind == Kind::Punct) return;
    uint32_t s = doc[i].end, e = s;
    while (e < t.size() && (idc(t[e]) || t[e] == '-')) ++e;
    char close = angle ? '>' : '}';
    if (e >= t.size() || t[e] != close || e == s) return;
    std::string_view w = t.substr(s, e - s);
    if (!al(w[0])) return;
    bool sep = w.find('-') != std::string_view::npos || w.find('_') != std::string_view::npos;
    if (angle) {
        std::string l = lower_s(w);
        if (!sep && !PLACEHOLDER_WORDS.contains(l)) return;
        if (l.rfind("tool", 0) == 0 || l.rfind("im_", 0) == 0 || l.rfind("function", 0) == 0 ||
            l.rfind("end_", 0) == 0 || l.rfind("start_", 0) == 0)
            return;
    } else if (s >= 2 && t[s - 2] == '{') {
        return;  // {{template}}
    }
    add_lc(doc, out, doc[i].start, e + 1, ident_words(w, Words::Placeholder), 80);
    // <org-name>/<repo-name>
    if (angle && e + 2 < t.size() && t[e + 1] == '/' && t[e + 2] == '<') out.add(e + 1, e + 2, " slash ", 80);
}

// Placeholders: <your-api-key> -> your API key, {user_name} -> user name.
RV_RULE("electronic", placeholder) {
    each_mark(doc, '<', [&](size_t i) { placeholder_at(doc, out, i, true); });
    each_mark(doc, '{', [&](size_t i) { placeholder_at(doc, out, i, false); });
}

const rv::PhraseMap MODIFIERS{{"alt", "Alt"},       {"cmd", "Command"}, {"command", "Command"},
                              {"control", "Control"}, {"ctrl", "Control"}, {"fn", "Fn"},
                              {"meta", "Meta"},     {"opt", "Option"},  {"option", "Option"},
                              {"shift", "Shift"},   {"super", "Super"}, {"win", "Win"},
                              {"windows", "Windows"}};
const rv::PhraseMap KEYS{{"backspace", "Backspace"}, {"click", "Click"},   {"del", "Del"},
                         {"delete", "Delete"},       {"down", "Down"},     {"drag", "Drag"},
                         {"end", "End"},             {"enter", "Enter"},   {"esc", "Escape"},
                         {"escape", "Escape"},       {"home", "Home"},     {"ins", "Insert"},
                         {"insert", "Insert"},       {"left", "Left"},     {"minus", "Minus"},
                         {"pagedown", "Page Down"},  {"pageup", "Page Up"}, {"pgdn", "Page Down"},
                         {"pgup", "Page Up"},        {"plus", "Plus"},     {"print", "Print"},
                         {"return", "Return"},       {"right", "Right"},   {"scroll", "Scroll"},
                         {"space", "Space"},         {"tab", "Tab"},       {"up", "Up"}};

// One key: a modifier, a named key, a letter, a digit, an F key, another key name with one capital.
bool key_name(std::string_view k, std::string& spoken, bool& modifier) {
    std::string l = lower_s(k);
    modifier = false;
    if (const std::string_view* m = MODIFIERS.find(l)) {
        spoken = std::string(*m);
        modifier = true;
        return true;
    }
    if (const std::string_view* m = KEYS.find(l)) {
        spoken = std::string(*m);
        return true;
    }
    if (k.size() == 1 && an(k[0])) {
        spoken = dg(k[0]) ? num_words(k) : upper_s(k);
        return true;
    }
    if (k.size() >= 2 && k.size() <= 3 && (k[0] == 'F' || k[0] == 'f') && all_digits(k.substr(1))) {
        spoken = "F " + num_words(k.substr(1));
        return true;
    }
    if (k.size() >= 3 && up(k[0]) && all_lower(k.substr(1))) {  // Ctrl+Period, Ctrl+Pause
        spoken = std::string(k);
        return true;
    }
    return false;
}

// Keyboard shortcuts: Ctrl+C -> Control C, Cmd+Shift+P -> Command Shift P, Alt+F4 -> Alt F four.
// A punctuation key after '+': Cmd+, -> Command comma, Ctrl+/ -> Control slash.
const char* punct_key(char c) {
    switch (c) {
        case ',': return "comma";
        case '.': return "period";
        case '/': return "slash";
        case '\\': return "backslash";
        case ';': return "semicolon";
        case '\'': return "quote";
        case '`': return "backtick";
        case '[': return "left bracket";
        case ']': return "right bracket";
        case '-': return "minus";
        case '=': return "equals";
        case '+': return "plus";
        default: return nullptr;
    }
}

// Mac key symbols: ⌘ Command, ⇧ Shift, ⌥ Option, ⌃ Control (modifiers), and ⏎ ↩ ⌫ ⎋ ⇥ keys.
const char* mac_key(uint32_t cp, bool& modifier) {
    modifier = true;
    switch (cp) {
        case 0x2318: return "Command";
        case 0x21E7: return "Shift";
        case 0x2325: return "Option";
        case 0x2303: return "Control";
        default: break;
    }
    modifier = false;
    switch (cp) {
        case 0x23CE: case 0x21A9: case 0x21B5: return "Return";
        case 0x232B: return "Delete";
        case 0x238B: return "Escape";
        case 0x21E5: return "Tab";
        default: return nullptr;
    }
}

// Keyboard shortcuts: Ctrl+C -> Control C, Cmd+Shift+P -> Command Shift P, Alt+F4 -> Alt F four; also with
// hyphens (Ctrl-C), spaced (Ctrl + S), a punctuation key (Cmd+,) and Mac symbols (⌘K, ⇧⌘P).
RV_RULE("electronic", shortcut) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    auto parse = [&](size_t k0, char sep, bool spaced) {
        uint32_t b = doc[k0].start, p = b, end = 0;
        if (b < last) return;
        std::string spoken;
        int keys = 0;
        bool last_mod = true, after_plus = false;
        while (true) {
            uint32_t q = p;
            while (q < t.size() && an(t[q])) ++q;
            std::string w;
            bool mod = false;
            if (q == p && after_plus && p < t.size() && punct_key(t[p]) &&
                (p + 1 >= t.size() || t[p + 1] == ' ' || t[p + 1] == '\n' || t[p + 1] == ')')) {
                w = punct_key(t[p]);  // Cmd+, Ctrl+/
                q = p + 1;
            } else if (q == p || !key_name(t.substr(p, q - p), w, mod)) {
                break;
            }
            // with hyphens only single keys and named keys after a modifier written with a capital
            if (sep == '-' && keys > 0 && !mod && q - p > 1 && !(up(t[p]) && (KEYS.find(lower_s(t.substr(p, q - p))) ||
                                                                             (t[p] == 'F' && dg(t[p + 1])))))
                break;
            append_word(spoken, w);
            ++keys;
            last_mod = mod;
            end = q;
            if (!mod) break;
            after_plus = false;
            if (!spaced && q + 1 < t.size() && t[q] == sep && (an(t[q + 1]) || (sep == '+' && punct_key(t[q + 1])))) {
                p = q + 1;
                after_plus = sep == '+';
            } else if (size_t pk = spaced ? doc.token_at(q) : doc.size();
                       pk + 1 < doc.size() && doc.punct(pk, '+') && doc[pk].space_before && doc[pk + 1].space_before &&
                       !doc[pk + 1].newline_before && an(t[doc[pk + 1].start])) {
                p = doc[pk + 1].start;  // Ctrl + S (also with no-break spaces)
            } else {
                break;
            }
        }
        if (keys < 2 || last_mod || (!right_edge(doc, end) && !(end > 0 && !an(t[end - 1])))) return;
        out.add(b, end, spoken, 85);
        last = end;
    };
    each_mark(doc, '+', [&](size_t k) {
        if (k == 0 || !doc.word(k - 1) || !left_edge(doc, doc[k - 1].start) || !MODIFIERS.find(doc.low(k - 1))) return;
        if (doc.joined(k)) parse(k - 1, '+', false);
        else if (doc[k].space_before && k + 1 < doc.size() && doc[k + 1].space_before && !doc[k + 1].newline_before)
            parse(k - 1, '+', true);
    });
    each_mark(doc, '-', [&](size_t k) {
        if (k == 0 || !doc.word(k - 1) || !doc.joined(k) || !left_edge(doc, doc[k - 1].start)) return;
        if (!up(t[doc[k - 1].start]) || !MODIFIERS.find(doc.low(k - 1))) return;
        parse(k - 1, '-', false);
    });
    // Mac symbols
    each_mark(doc, '\xE2', [&](size_t k) {
        bool mod = false;
        if (doc[k].start < last || doc[k].kind != Kind::Punct || !mac_key(doc[k].cp, mod) || !mod ||
            !left_edge(doc, doc[k].start))
            return;
        std::string spoken;
        size_t j = k;
        bool last_mod = true;
        uint32_t end = 0;
        while (j < doc.size() && (j == k || doc.joined(j))) {
            if (doc.punct(j, '+') && j > k && last_mod) {
                ++j;
                continue;
            }
            const char* m = doc[j].kind == Kind::Punct ? mac_key(doc[j].cp, mod) : nullptr;
            if (m) {
                append_word(spoken, m);
                last_mod = mod;
                end = doc[j].end;
                ++j;
                if (!mod) break;
                continue;
            }
            std::string w;
            if (doc[j].kind == Kind::Punct || !key_name(doc.str(j), w, mod)) break;
            append_word(spoken, w);
            last_mod = mod;
            end = doc[j].end;
            ++j;
            if (!mod) break;
        }
        if (last_mod || !end || !right_edge(doc, end)) return;
        out.add(doc[k].start, end, spoken, 85);
        last = end;
    });
}

const rv::WordSet FKEY_CUES{"hit", "hold", "key", "keys", "press", "pressed", "presses", "pressing", "tap", "use",
                            "with"};

// Function keys on their own: F12 -> F twelve (after a key word, or before "key").
RV_RULE("electronic", function_key) {
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        if (doc[i].end - doc[i].start != 1 || (doc.text[doc[i].start] | 0x20) != 'f') continue;
        if (!doc.number(i + 1) || !doc.joined(i + 1)) continue;
        if (!left_edge(doc, doc[i].start) || !right_edge(doc, doc[i + 1].end)) continue;
        std::string_view d = doc.str(i + 1);
        if (d.size() > 2 || d[0] == '0' || std::stoi(std::string(d)) > 24) continue;
        size_t pw = doc.prev_word(i), nw = doc.next_word(i + 1);
        bool cue = (pw != Doc::npos && FKEY_CUES.contains(doc.low(pw))) ||
                   (nw != Doc::npos && (doc.is(nw, "key") || doc.is(nw, "opens") || doc.is(nw, "toggles")));
        if (cue) out.tokens(i, i + 1, "F " + num_words(d), 70);
    }
}

const rv::WordSet HEX_CUES{"build", "checksum", "commit", "commits", "hash", "id", "image", "job", "ref",
                           "revision", "rev", "sha", "tag", "version"};

bool hexc(char c) { return dg(c) || (c >= 'a' && c <= 'f'); }

// Short hexadecimal identifiers: commit ef254ca -> commit EF two five four CA.
RV_RULE("electronic", hex_id) {
    const std::string_view t = doc.text;
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        // a hex run mixes digits and letters, so it is a Number glued to a Word or the other way round
        if (doc[i].kind == Kind::Punct || !doc.joined(i + 1) || doc[i + 1].kind == Kind::Punct ||
            !hexc(t[doc[i].start]))
            continue;
        uint32_t b = doc[i].start;
        if (!left_edge(doc, b)) continue;
        uint32_t e = b;
        bool letter = false, digit = false;
        while (e < t.size() && hexc(t[e])) {
            if (dg(t[e])) digit = true;
            else letter = true;
            ++e;
        }
        skip_to(doc, i, e);
        if (!letter || !digit || e - b < 4 || e - b > 8 || !right_edge(doc, e)) continue;
        if (e < t.size() && (t[e] == '-' || t[e] == '.' || t[e] == ':') && e + 1 < t.size() && an(t[e + 1])) continue;
        std::string_view s = t.substr(b, e - b);
        // not a number with a unit letter (14b, 3e5) or a letter with a number (bf16): runs alternate
        size_t letters = 0, runs = 0;
        for (size_t k = 0; k < s.size(); ++k) {
            if (!dg(s[k])) ++letters;
            if (k == 0 || dg(s[k]) != dg(s[k - 1])) ++runs;
        }
        size_t pw = doc.prev_word(tok_at(doc, b));
        bool cue = pw != Doc::npos && HEX_CUES.contains(doc.low(pw));
        bool shape = runs >= 3 && (s.size() >= 6 || (letters >= 2 && s.size() - letters >= 1));
        if (cue || shape) out.add(b, e, hex_words(s), 75);
    }
}

// The class of a byte for randomness: 1 lower, 2 upper, 3 digit, 0 other.
int cls(char c) { return lo(c) ? 1 : up(c) ? 2 : dg(c) ? 3 : 0; }

// A segment that looks random: 9+ characters, letters and digits (or both cases) in short runs.
bool random_segment(std::string_view s) {
    if (s.size() < 9) return false;
    size_t switches = 0, run = 0, max_letter_run = 0;
    bool digit = false, lower = false, upper = false;
    for (size_t k = 0; k < s.size(); ++k) {
        int c = cls(s[k]);
        if (c == 0) return false;
        digit |= c == 3;
        lower |= c == 1;
        upper |= c == 2;
        if (k > 0 && c != cls(s[k - 1])) {
            ++switches;
            run = 0;
        }
        ++run;
        if (c != 3) max_letter_run = std::max(max_letter_run, run);
    }
    if (!(digit || (lower && upper))) return false;
    if (!digit && switches < 8) return false;
    return switches >= 5 && (max_letter_run <= 5 || switches * 3 >= s.size());
}

bool hex_run(std::string_view s, size_t n) {
    if (s.size() != n) return false;
    for (char c : s)
        if (!(dg(c) || (lower_c(c) >= 'a' && lower_c(c) <= 'f'))) return false;
    return true;
}

// Whether a run of joined characters is a random string: hash, UUID, key, base64, MAC address, product key.
bool random_string(std::string_view s) {
    std::string_view seg[24];
    int n = 0;
    char sep = 0;
    bool one_sep = true;
    size_t k = 0;
    while (k <= s.size() && n < 24) {
        size_t j = k;
        while (j < s.size() && an(s[j])) ++j;
        seg[n++] = s.substr(k, j - k);
        if (j < s.size()) {
            if (sep && s[j] != sep) one_sep = false;
            sep = s[j];
        }
        k = j + 1;
    }
    for (int m = 0; m < n; ++m)
        if (random_segment(seg[m])) return true;
    // UUID 8-4-4-4-12
    if (n == 5 && sep == '-' && one_sep && hex_run(seg[0], 8) && hex_run(seg[1], 4) && hex_run(seg[2], 4) &&
        hex_run(seg[3], 4) && hex_run(seg[4], 12))
        return true;
    // MAC address: six or eight hex pairs
    if ((n == 6 || n == 8) && one_sep && (sep == ':' || sep == '-')) {
        bool mac = true;
        for (int m = 0; m < n; ++m)
            if (!hex_run(seg[m], 2)) mac = false;
        if (mac) return true;
    }
    // product keys: three or more groups of 4-6 capitals and digits
    if (n >= 3 && one_sep && sep == '-') {
        int mixed = 0;
        bool all = true;
        for (int m = 0; m < n; ++m) {
            bool d = false, a = false;
            if (seg[m].size() < 4 || seg[m].size() > 6) all = false;
            for (char c : seg[m]) {
                if (dg(c)) d = true;
                else if (up(c)) a = true;
                else all = false;
            }
            if (d && a) ++mixed;
        }
        if (all && mixed >= 2) return true;
    }
    // a machine identifier: a segment of 12+ characters with letters and six or more digits (20261104T103021),
    // or one of 14+ characters with ten or more digits (a tracking number: 1Z9999W99999999999)
    for (int m = 0; m < n; ++m)
        if (seg[m].size() >= 12) {
            size_t d = 0, a = 0;
            for (char c : seg[m]) (dg(c) ? d : a) += 1;
            if (d >= 6 && a >= 1 && n >= 3) return true;
            if (seg[m].size() >= 14 && d >= 10 && a >= 1) return true;
        }
    // 12+ lower-case hex digits with letters
    for (int m = 0; m < n; ++m)
        if (seg[m].size() >= 12) {
            bool hex = true, letter = false, digit = false;
            for (char c : seg[m]) {
                if (dg(c)) digit = true;
                else if (c >= 'a' && c <= 'f') letter = true;
                else hex = false;
            }
            if (hex && letter && digit) return true;
        }
    return false;
}

// Long random strings stay as written: hashes, UUIDs, keys, base64, MAC addresses.
RV_RULE("electronic", keep_random) {
    const std::string_view t = doc.text;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].kind == Kind::Punct) continue;
        if (!glued(doc, i) && !(doc[i].casing == rv::MIXED && doc[i].end - doc[i].start >= 9)) continue;
        if (glued(doc, i) && doc[i + 1].kind == Kind::Punct && !glued(doc, i + 1) && doc[i].end - doc[i].start < 9)
            continue;  // "word." "word,"
        uint32_t b = doc[i].start;
        if (!left_edge(doc, b) && !(b > 0 && t[b - 1] == ':')) continue;
        if (b > 0 && (t[b - 1] == '=' || t[b - 1] == '&')) {
            // a value in the query of a web address belongs to the address
            uint32_t w = b;
            while (w > 0 && static_cast<unsigned char>(t[w - 1]) > ' ') --w;
            std::string_view before = t.substr(w, b - w);
            if (before.find('?') != std::string_view::npos && before.find('.') != std::string_view::npos) continue;
        }
        uint32_t e = b;
        while (e < t.size() && (an(t[e]) || t[e] == '_' || t[e] == '-' || t[e] == '+' || t[e] == '/' || t[e] == '.' ||
                                t[e] == ':'))
            ++e;
        while (e < t.size() && t[e] == '=') ++e;  // base64 padding
        skip_to(doc, i, e);
        uint32_t end = trim_tail(t, b, e);
        while (end > b && (t[end - 1] == '-' || t[end - 1] == '/' || t[end - 1] == '_')) --end;
        std::string_view s = t.substr(b, end - b);
        if (s.size() < 9 || s.find("://") != std::string_view::npos || s.substr(0, 4) == "www.") continue;
        if (!random_string(s)) continue;
        Host h = parse_host(t, b, end, true);
        if (h.end && (h.end == end || t[h.end] == '/' || t[h.end] == ':')) continue;  // a web address
        out.add(b, end, std::string(s), 92);
    }
}

}  // namespace
