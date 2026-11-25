// File names, file types with a leading dot, file patterns, Unix, relative and Windows paths, API routes.
#include "rules/electronic/common.h"

namespace {

using namespace rv::el;
using rv::Doc;
using rv::Kind;

bool file_char(char c) { return an(c) || c == '_' || c == '-' || c == '.'; }

// Start of the letter that ends at byte b when it is a non-ASCII letter (dúvidas.txt), else b.
uint32_t letter_before(std::string_view t, uint32_t b) {
    if (b == 0 || static_cast<unsigned char>(t[b - 1]) < 0x80) return b;
    uint32_t k = b - 1;
    while (k > 0 && (static_cast<unsigned char>(t[k]) & 0xC0) == 0x80) --k;
    size_t m = k;
    return rv::is_letter(rv::decode(t, m)) && m == b ? k : b;
}

// A file name: name parts, then a known extension (report_v2.pdf, archive.tar.gz).
bool file_name(std::string_view f) {
    size_t dot = f.rfind('.');
    if (dot == std::string_view::npos || dot == 0 || dot + 1 >= f.size()) return false;
    if (!is_ext(f.substr(dot + 1))) return false;
    if (!an(f[0]) && static_cast<unsigned char>(f[0]) < 0x80) return false;
    bool initials = true;  // D.C, U.S: single letters between periods are an abbreviation
    for (size_t k = 0; k + 1 < f.size(); ++k) {
        if (f[k] == '.' && f[k + 1] == '.') return false;
        if (f[k] != '.' && f[k + 1] != '.') initials = false;
    }
    return !initials;
}

// Words of running text that a file type can stick to when a space is missing.
const rv::WordSet GLUE_WORDS{"accept", "accepts", "and", "as", "convert", "converts", "download", "downloads",
                             "export", "exports", "from", "import", "imports", "in", "into", "like", "of", "open",
                             "opens", "or", "read", "reads", "save", "saves", "support", "supports", "the", "to",
                             "upload", "uploads", "use", "uses", "with"};

// File names: report.txt -> report dot TXT, config.json -> config dot Json.
RV_RULE("electronic", file) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    each_mark(doc, '.', [&](size_t k) {
        uint32_t d = doc[k].start;
        if (d < last || d == 0 || !an(t[d - 1]) || !doc.joined(k) || !glued(doc, k)) return;
        uint32_t b = d;
        while (b > 0) {
            if (file_char(t[b - 1])) --b;
            else if (uint32_t k = letter_before(t, b); k < b) b = k;
            else break;
        }
        while (b < d && !an(t[b]) && static_cast<unsigned char>(t[b]) < 0x80) ++b;
        uint32_t e = d;
        while (e < t.size() && file_char(t[e])) ++e;
        last = e;
        if (!left_edge(doc, b)) return;
        uint32_t end = trim_tail(t, b, e);
        if (!right_edge(doc, e)) return;
        if (e < t.size() && (t[e] == '/' || t[e] == '@' || t[e] == ':' || t[e] == '\\') && end == e) return;
        std::string_view f = t.substr(b, end - b);
        if (!file_name(f)) return;
        if (up(f[0]) && f.size() > 3 && lower_s(f.substr(f.size() - 3)) == ".js" && f.find('.') == f.size() - 3)
            return;  // Node.js, Vue.js: product names
        // a missing space before a file type: "accepts.TXT", "and.PDF" -> accepts TXT, and PDF
        size_t dot = f.find('.');
        if (dot != std::string_view::npos && dot == f.rfind('.') && all_upper(f.substr(dot + 1)) &&
            all_lower(f.substr(0, dot)) && GLUE_WORDS.contains(f.substr(0, dot))) {
            out.add(b + static_cast<uint32_t>(dot), end, " " + ext_word(f.substr(dot + 1)), 80);
            return;
        }
        // a header in angle brackets: #include <stdio.h> -> stdio dot H
        uint32_t lb = b, le = end;
        if (lb > 0 && t[lb - 1] == '<' && le < t.size() && t[le] == '>') --lb, ++le;
        add_lc(doc, out, lb, le, file_words(f), 80);
    });
}

// A known file type with a leading dot shortly before token i: ".VTT or .SBV".
bool listed_type(const Doc& doc, size_t i) {
    for (size_t k = i >= 8 ? i - 8 : 0; k + 1 < i; ++k)
        if (doc.punct(k, '.') && doc.joined(k + 1) && doc.word(k + 1) && space_before(doc, doc[k].start) &&
            is_ext(doc.str(k + 1)))
            return true;
    return false;
}

bool path_body(std::string_view p, std::string& spoken);

// File types named with a leading dot: .PDF -> PDF, .EPUB -> E Pub; a domain ending: .com -> dot com.
void dot_type_at(const Doc& doc, rv::Out& out, size_t i) {
    const std::string_view t = doc.text;
    if (i + 1 >= doc.size() || !doc.joined(i + 1) || doc[i + 1].kind == Kind::Punct) return;
    uint32_t d = doc[i].start;
    uint32_t s = d + 1, e = s;
    while (e < t.size() && an(t[e])) ++e;
    if (!right_edge(doc, e)) return;
    if (e < t.size() && (t[e] == '.' || t[e] == '/') && e + 1 < t.size() && an(t[e + 1])) return;
    std::string_view w = t.substr(s, e - s);
    if (!al(w[0])) return;
    std::string spoken;
    if (is_ext(w)) spoken = ext_word(w);
    else if (is_tld(w)) spoken = "dot " + tld_word(w);
    else if (all_upper(w) && w.size() >= 2 && w.size() <= 5 && listed_type(doc, i)) spoken = w;
    else return;
    if (d > 0 && t[d - 1] == ',') spoken.insert(0, " ");  // "TXT,.EPUB" -> TXT, E Pub
    out.add(d, e, spoken, 75);
}

// Hidden files and folders: .gitignore -> dot gitignore, .env.example -> dot env dot example,
// .github/workflows -> dot github slash workflows (a lone type such as .pdf or .com is read as a type).
void dotfile_at(const Doc& doc, rv::Out& out, size_t i) {
    const std::string_view t = doc.text;
    if (i + 1 >= doc.size() || !doc.joined(i + 1) || !doc.word(i + 1) || doc[i + 1].casing != rv::LOWER) return;
    uint32_t d = doc[i].start;
    if (d > 0 && (t[d - 1] == '.' || t[d - 1] == ',')) return;
    uint32_t e = d + 1;
    while (e < t.size() && (lo(t[e]) || dg(t[e]) || t[e] == '_' || t[e] == '-' || t[e] == '.' || t[e] == '/')) ++e;
    uint32_t end = trim_tail(t, d + 1, e);
    while (end > d + 1 && t[end - 1] == '/') --end;
    if (!right_edge(doc, e) || end < d + 4) return;
    std::string_view name = t.substr(d + 1, end - d - 1);
    bool compound = name.find_first_of("./") != std::string_view::npos;
    if (!compound && (is_ext(name) || is_tld(name))) return;
    std::string spoken = "dot";
    if (name.find('/') != std::string_view::npos) {
        if (!path_body(name, spoken)) return;
    } else {
        append_word(spoken, compound ? file_words(name) : std::string(name));
    }
    add_lc(doc, out, d, end, spoken, 80);
}

// "cd .." -> cd dot dot (the parent folder in a command).
void parent_dir_at(const Doc& doc, rv::Out& out, size_t k) {
    if (k == 0 || !doc.is(k - 1, "cd") || !doc[k].space_before || !doc.punct(k + 1, '.') || !doc.joined(k + 1)) return;
    if (k + 2 < doc.size() && doc.joined(k + 2) && !doc.punct(k + 2, '`') && !doc.punct(k + 2, ',')) return;
    out.tokens(k, k + 1, "dot dot", 80);
}

// A dot that starts a word: a file type, a domain ending, a hidden file, "cd ..".
RV_RULE("electronic", dot_type) {
    const std::string_view t = doc.text;
    each_mark(doc, '.', [&](size_t i) {
        uint32_t d = doc[i].start;
        if (!left_edge(doc, d) && !(d > 0 && t[d - 1] == ',')) return;
        dot_type_at(doc, out, i);
        if (left_edge(doc, d)) dotfile_at(doc, out, i);
        parent_dir_at(doc, out, i);
    });
}

// File patterns: *.txt -> star dot TXT.
RV_RULE("electronic", file_pattern) {
    const std::string_view t = doc.text;
    each_mark(doc, '*', [&](size_t i) {
        if (i + 2 >= doc.size() || !doc.punct(i + 1, '.') || !doc.joined(i + 1) || !doc.joined(i + 2)) return;
        uint32_t s = doc[i].start;  // a prefix before the star stays: contract*.docx
        uint32_t e = s + 1;
        while (e < t.size() && file_char(t[e])) ++e;
        e = trim_tail(t, s + 1, e);
        if (e < s + 3) return;
        std::string_view f = t.substr(s + 1, e - s - 1);  // ".tar.gz"
        std::string spoken = "star";
        size_t k = 0;
        bool ok = !f.empty();
        while (k < f.size() && ok) {
            size_t n = f.find('.', k + 1);
            if (n == std::string_view::npos) n = f.size();
            std::string_view x = f.substr(k + 1, n - k - 1);
            if (x.empty()) ok = false;
            append_word(spoken, "dot");
            append_word(spoken, ext_word(x));
            k = n;
        }
        if (!ok) return;
        add_lc(doc, out, s, e, spoken, 80);
    });
}

bool path_char(char c) {
    return an(c) || c == '_' || c == '-' || c == '.' || c == '/' || c == '{' || c == '}' || c == '+';
}

// The spoken segments of a path body (after its root), segments separated by '/'.
bool path_body(std::string_view p, std::string& spoken) {
    size_t i = 0;
    bool first = true;
    bool api = p.size() >= 2 && p[0] == 'v' && dg(p[1]);  // v1/speech/generate
    while (i < p.size()) {
        size_t j = p.find('/', i);
        if (j == std::string_view::npos) j = p.size();
        std::string_view seg = p.substr(i, j - i);
        if (!seg.empty()) {
            if (seg == "..") {
                if (!first) append_word(spoken, "slash");
                append_word(spoken, "dot dot");
            } else if (seg == ".") {
                if (!first) append_word(spoken, "slash");
                append_word(spoken, "dot");
            } else {
                if (!first) append_word(spoken, "slash");
                append_word(spoken, path_part(seg, api, true));
            }
            first = false;
        } else if (j < p.size() && i != 0) {
            return false;  // "//"
        }
        i = j + 1;
    }
    return true;
}

// Unix paths and API routes: /usr/local/bin -> slash usr slash local slash bin, ~/models -> home slash models,
// /v1/speech/generate -> slash V one slash speech slash generate.
RV_RULE("electronic", unix_path) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    each_mark(doc, '/', [&](size_t k) {
        uint32_t sl = doc[k].start;
        if (sl < last) return;
        size_t i = k;
        if (k > 0 && doc.joined(k) && doc.punct(k - 1, '~')) i = k - 1;
        else if (k > 1 && doc.joined(k) && doc.joined(k - 1) && doc.punct(k - 1, '.') && doc.punct(k - 2, '.'))
            i = k - 2;
        else if (k > 0 && doc.joined(k) && doc.punct(k - 1, '.')) i = k - 1;
        uint32_t b = doc[i].start;
        bool root = i == k;
        bool home = doc.punct(i, '~');
        if (!root && !home && !doc.punct(i, '.')) return;
        if (!left_edge(doc, b) && !(b > 0 && t[b - 1] == ':' && i > 0 && doc.word(i - 1))) return;
        std::string spoken;
        uint32_t body = 0;
        if (root) {
            spoken = "slash";
            body = 1;
        } else if (home) {
            spoken = "home slash";
            body = 2;
        } else if (doc.text_at(b, "../")) {
            spoken = "dot dot slash";
            body = 3;
        } else {
            spoken = "dot slash";
            body = 2;
        }
        uint32_t e = b + body;
        while (e < t.size() && path_char(t[e])) ++e;
        uint32_t end = std::max(trim_tail(t, b + body, e), b + body);
        std::string_view raw = t.substr(b + body, e - b - body);
        if (!root && !raw.empty() && raw.find_first_not_of("./") == std::string_view::npos && raw.size() >= 2 &&
            raw.substr(0, 2) == ".." && raw.find("...") == std::string_view::npos)
            end = e;  // ../.. keeps its last dots
        if (end == e && !right_edge(doc, e) && e < t.size() && t[e] != ':' && t[e] != ')') return;
        std::string_view p = t.substr(b, end - b);
        std::string_view rest = p.substr(body);
        if (rest.empty() && !home) return;
        // a path needs a name: "/2", "/s" and a lone "/" are something else
        size_t letters = 0;
        for (char c : rest)
            if (al(c)) ++letters;
        bool parents = !rest.empty() && rest.find_first_not_of("./") == std::string_view::npos &&
                       rest.find("...") == std::string_view::npos && !root;  // ../..
        if (letters < 2 && !(home && rest.empty()) && !parents) return;
        if (root && !(al(rest[0]) || rest[0] == '{' || rest[0] == '_' || rest[0] == '.')) return;
        if (parents && rest.back() == '/') rest.remove_suffix(1);
        if (home && rest.empty()) {
            add_lc(doc, out, b, b + 2, "home", 80);
            return;
        }
        if (!rest.empty() && rest.back() == '/') rest.remove_suffix(1);
        if (!path_body(rest, spoken)) return;
        add_lc(doc, out, b, b + static_cast<uint32_t>(p.size()), spoken, 80);
        last = end;
    });
}

const rv::WordSet DIR_ROOTS{"app", "assets", "bin", "build", "conf", "config", "configs", "data", "dist", "docs",
                            "examples", "lib", "logs", "models", "notebooks", "out", "output", "outputs", "pkg",
                            "reports", "results", "scripts", "src", "static", "test", "tests", "tmp", "tools"};

// Relative paths: src/main.py -> src slash main dot PY, data/raw/train -> data slash raw slash train.
RV_RULE("electronic", relative_path) {
    const std::string_view t = doc.text;
    uint32_t last = 0;
    each_mark(doc, '/', [&](size_t k) {
        uint32_t sl = doc[k].start;
        if (sl < last || k == 0 || !doc.word(k - 1) || !doc.joined(k)) return;
        uint32_t b = sl;
        while (b > 0 && path_char(t[b - 1])) --b;
        size_t i = tok_at(doc, b);
        uint32_t e = sl;
        while (e < t.size() && path_char(t[e])) ++e;
        last = e;
        if (i >= doc.size() || !doc.word(i) || !left_edge(doc, b)) return;
        uint32_t end = trim_tail(t, b, e);
        if (!right_edge(doc, e) && end == e) return;
        std::string_view p = t.substr(b, end - b);
        size_t slashes = 0;
        for (char c : p)
            if (c == '/') ++slashes;
        if (slashes == 0 || p.back() == '/') return;
        size_t first = p.find('/');
        if (first <= 1) return;  // r/, u/, w/
        std::string_view tail = p.substr(p.rfind('/') + 1);
        bool ext = file_name(tail);
        bool dir = slashes >= 2 && DIR_ROOTS.contains(lower_s(p.substr(0, first)));
        if (!ext && !dir) return;
        if (i > 0 && doc.number(i - 1)) return;
        std::string spoken;
        if (!path_body(p, spoken)) return;
        add_lc(doc, out, b, end, spoken, 80);
    });
}

// Network shares: \\server\share\folder -> backslash backslash server backslash share backslash folder.
void unc_path_at(const Doc& doc, rv::Out& out, size_t k) {
    const std::string_view t = doc.text;
    {
        if (!doc.punct(k + 1, '\\') || !doc.joined(k + 1) || !doc.word(k + 2) || !doc.joined(k + 2)) return;
        uint32_t b = doc[k].start;
        if (!left_edge(doc, b) || (b > 0 && t[b - 1] == '\\')) return;
        uint32_t e = b + 2;
        while (e < t.size() && (path_char(t[e]) || t[e] == '\\') && t[e] != '/') ++e;
        uint32_t end = trim_tail(t, b, e);
        std::string_view rest = t.substr(b + 2, end - b - 2);
        if (rest.find('\\') == std::string_view::npos) return;
        std::string spoken = "backslash backslash";
        size_t c = 0;
        bool first = true;
        while (c < rest.size()) {
            size_t j = rest.find('\\', c);
            if (j == std::string_view::npos) j = rest.size();
            if (j > c) {
                if (!first) append_word(spoken, "backslash");
                append_word(spoken, path_part(rest.substr(c, j - c), false, true));
                first = false;
            }
            c = j + 1;
        }
        add_lc(doc, out, b, end, spoken, 80);
    }
}

// Windows paths: C:\Users\Admin -> C colon backslash Users backslash Admin.
RV_RULE("electronic", windows_path) {
    const std::string_view t = doc.text;
    each_mark(doc, '\\', [&](size_t k) {
        unc_path_at(doc, out, k);
        if (k < 2) return;
        size_t i = k - 2;
        if (!doc.word(i) || doc.str(i).size() != 1 || !up(doc.str(i)[0]) || !left_edge(doc, doc[i].start)) return;
        if (!doc.punct(i + 1, ':') || !doc.joined(i + 1) || !doc.joined(i + 2)) return;
        uint32_t b = doc[i].start, e = doc[i + 2].end;
        while (e < t.size() && (path_char(t[e]) || t[e] == '\\') && t[e] != '/') ++e;
        uint32_t end = trim_tail(t, b, e);
        std::string spoken = std::string(doc.str(i)) + " colon backslash";
        std::string_view rest = t.substr(doc[i + 2].end, end - doc[i + 2].end);
        size_t c = 0;
        bool first = true;
        while (c < rest.size()) {
            size_t j = rest.find('\\', c);
            if (j == std::string_view::npos) j = rest.size();
            std::string_view seg = rest.substr(c, j - c);
            if (!seg.empty()) {
                if (!first) append_word(spoken, "backslash");
                append_word(spoken, path_part(seg, false, true));
                first = false;
            }
            c = j + 1;
        }
        out.add(b, end, spoken, 80);
    });
}

}  // namespace
