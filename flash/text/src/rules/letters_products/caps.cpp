// Capital-letter conventions: initialisms spelled out in capitals, acronyms said as a word with one capital,
// mixed-case names, shouted and emphasised words, initialisms with periods, slashes and ampersands.
#include <vector>

#include "rules/letters_products/lp.h"

namespace {

using lp::Doc;

bool alnum_tok(const Doc& d, size_t k) { return k < d.size() && d[k].kind != rv::Kind::Punct; }

bool sentence_end(const Doc& d, size_t k) {
    if (d[k].kind != rv::Kind::Punct) return false;
    uint32_t c = d[k].cp;
    if (c == '!' || c == '?' || c == 0x2026) return true;
    // a period that is not inside a number or a name ("3.5", "U.S.")
    return c == '.' && !(d.joined(k + 1) && d[k + 1].kind != rv::Kind::Punct);
}

// An all-capital word, or one with a plural or possessive ending ("GPUs", "CEO's"). Splits it into the
// capitals and the ending; false for anything else.
// A shouted contraction: "DON'T", "WE'RE", "I'LL", "YOU’VE".
bool contraction(std::string_view w) {
    size_t ap = w.find('\'');
    size_t len = 1;
    if (ap == std::string_view::npos) ap = w.find("\xE2\x80\x99"), len = 3;
    if (ap == std::string_view::npos || ap == 0 || !lp::is_upper_word(w.substr(0, ap))) return false;
    std::string_view t = w.substr(ap + len);
    return t == "T" || t == "LL" || t == "RE" || t == "VE" || t == "D" || t == "M";
}

bool caps_form(std::string_view w, std::string_view& base, std::string_view& tail) {
    size_t cut = w.size();
    size_t ap = w.find('\'');
    if (ap == std::string_view::npos) ap = w.find("\xE2\x80\x99");
    if (ap != std::string_view::npos) {
        std::string_view t = w.substr(ap);
        if (contraction(w)) {
            base = w, tail = std::string_view();
            return true;
        }
        if (t != "'s" && t != "'S" && t != "\xE2\x80\x99s" && t != "\xE2\x80\x99S") return false;
        cut = ap;
    } else if (w.size() >= 3 && w.back() == 's') {
        cut = w.size() - 1;
    }
    base = w.substr(0, cut);
    tail = w.substr(cut);
    return lp::is_upper_word(base) && (base.size() >= 2 || (base.size() == 1 && tail.empty()));
}

// Lines that consist of capitals only and stand alone as a heading belong to the heading rules. Looked up line
// by line as the tokens are visited, so the whole check stays linear. caps_line(): a line written in capitals
// with two or more English words ("ACME CORP. AND GLOBEX INC. MERGE"), shouted even where periods split it.
class HeadingLines {
public:
    explicit HeadingLines(const Doc& d) : d_(d) {}
    bool operator()(size_t i) {
        line(i);
        return heading_;
    }
    bool caps_line(size_t i) {
        line(i);
        return caps_;
    }

private:
    void line(size_t i) {
        if (i >= a_ && i < b_) return;
        if (many_ < 0) {
            many_ = 0;
            for (size_t k = 1; k < d_.size() && !many_; ++k) many_ = d_[k].newline_before;
        }
        a_ = i;
        while (a_ > 0 && !d_[a_].newline_before) --a_;
        b_ = i + 1;
        while (b_ < d_.size() && !d_[b_].newline_before) ++b_;
        bool lower = false;
        int words = 0, english = 0;
        for (size_t k = a_; k < b_ && !lower; ++k)
            if (d_.word(k)) {
                ++words;
                lower |= d_[k].casing != rv::UPPER;
            }
        for (size_t k = a_; k < b_ && !lower && english < 2; ++k)
            english += d_.word(k) && d_.str(k).size() >= 2 && lp::english(d_.low(k));
        bool sentence = d_[b_ - 1].kind == rv::Kind::Punct &&
                        (d_[b_ - 1].cp == '.' || d_[b_ - 1].cp == '!' || d_[b_ - 1].cp == '?');
        heading_ = many_ && !lower && words > 0 && words <= 12 && !sentence;
        caps_ = !lower && english >= 2;
    }

    const Doc& d_;
    size_t a_ = 0, b_ = 0;
    int many_ = -1;
    bool heading_ = false, caps_ = false;
};

// A log level stands at the start of a line or sentence, in brackets, or before a colon: "ERROR:", "[WARN]".
bool log_context(const Doc& d, size_t i) {
    // after a timestamp ("04:12:09Z ERROR", "13:27 INFO") or a bracketed tag ("[pit-2] INFO")
    bool after_stamp = i > 0 && (d.number(i - 1) || d.punct(i - 1, ']') || d.punct(i - 1, '|') ||
                                 (i > 1 && d.word(i - 1) && d.joined(i - 1) && d.number(i - 2)));
    return d.sentence_start(i) || after_stamp || (i > 0 && d.punct(i - 1, '[')) || d.punct(i + 1, ':') ||
           d.punct(i + 1, ']') || (d.punct(i + 1, '[') && d[i + 1].space_before);  // "07:41:56Z ERROR [line-4]"
}

// A long word with an English ending ("ENFORCED", "DEFINITELY") is an emphasised word, not an acronym.
bool english_like(const std::string& w) {
    static const char* const ENDS[] = {"ed", "ing", "ly", "tion", "tions", "sion", "ment", "ments", "ness",
                                       "able", "ible", "ive", "ous", "ful", "less", "ity", "ance", "ence"};
    if (w.size() < 6 || !lp::pronounceable(w)) return false;
    for (const char* e : ENDS) {
        std::string_view end(e);
        if (w.size() > end.size() + 2 && w.compare(w.size() - end.size(), end.size(), end) == 0) return true;
    }
    return false;
}

// A stressed syllable of a respelled pronunciation: capitals joined by a hyphen to a lower-case part that is
// not an English word ("KURE-ee"); "BESS-based" is not one.
bool stressed_syllable(const Doc& d, size_t i) {
    auto part = [&](size_t k) { return d.word(k) && d[k].casing == rv::LOWER && !lp::english(d.low(k)); };
    return (d.joined(i + 1) && d.punct(i + 1, '-') && d.joined(i + 2) && part(i + 2)) ||
           (i >= 2 && d.joined(i) && d.punct(i - 1, '-') && d.joined(i - 1) && part(i - 2));
}

const rv::WordSet CODE_CUES{"code", "codes", "coupon", "promo", "voucher", "password", "keyword"};
// Words before an identifier: an unlisted capital word after them is spelled ("number NIOFEI").
const rv::WordSet ID_CUES{"code",       "codes", "coupon", "id",  "keyword",   "number",    "passcode",
                          "passphrase", "password", "pin", "promo", "reference", "serial", "voucher"};
// Words before "IT" or "US" that make it the initialism in a shouted line: "THE IT DESK".
const rv::WordSet DETERMINERS{"our", "the", "their", "your"};
// Product words after a model number keep one capital in a shouted line: "IPHONE 15 PRO MAX".
const rv::WordSet PRODUCT_WORDS{"AIR", "LITE", "MAX", "MINI", "PLUS", "PRO", "ULTRA"};

// All-capital words. Respellings apply everywhere; in a run of shouted words the ordinary words go to lower
// case and other pronounceable words get one capital; a lone emphasised word goes to lower case; elsewhere an
// unlisted word gets one capital only when lp::word_like() says it reads as a word, else it is spelled.
RV_RULE("letters_products", caps_words) {
    const size_t n = doc.size();
    HeadingLines heading(doc);
    std::vector<size_t> run;
    auto flush = [&]() {
        int english = 0, total = static_cast<int>(run.size());
        if (total >= 2)
            for (size_t i : run) {
                std::string_view base, tail;
                caps_form(doc.str(i), base, tail);
                bool eng = lp::english(lp::lower(base));
                if (contraction(base) || (base.size() >= 2 && eng && (tail.empty() || tail[tail.size() - 1] == 'S')))
                    ++english;
            }
        bool shouted = english >= 2 && english * 2 >= total;
        for (size_t i : run) {
            std::string_view w = doc.str(i), base, tail;
            caps_form(w, base, tail);
            // the possessive ending is written in lower case: "NASA'S" -> Nasa's, "CEO'S" -> CEO's
            std::string end(tail);
            if (!end.empty() && end.back() == 'S') end.back() = 's';
            std::string spoken;
            int score = 55;
            if (w == "SOC" && doc.number(i + 1) && doc[i + 1].space_before) {
                out.tokens(i, i, "Soc", 55);  // "SOC 2"; the security team "SOC" stays
                continue;
            }
            if (contraction(base)) {
                // "DON'T", "WE'RE": always words
                out.tokens(i, i, lp::lower(base), 45);
                continue;
            }
            size_t prev_tok = i > 0 && doc[i].space_before ? i - 1 : Doc::npos;
            bool after_number = prev_tok != Doc::npos &&
                                (doc.number(prev_tok) || (doc.word(prev_tok) && PRODUCT_WORDS.contains(doc.str(prev_tok))));
            if (const auto* r = lp::respelling(w)) {
                spoken = std::string(*r);
            } else if (const auto* rb = tail.empty() ? nullptr : lp::respelling(base)) {
                spoken = std::string(*rb) + end;
            } else if (heading(i)) {
                continue;
            } else if (lp::log_level(base) && log_context(doc, i)) {
                spoken = shouted ? lp::lower(base) : lp::title(base);
                spoken += end;
            } else if (base.size() == 1) {
                if (shouted && base == "A") spoken = "a", score = 45;
            } else if (base == "GET" && tail.empty() && i > 0 && doc.number(i - 1) && doc.number(i + 1)) {
                spoken = "get", score = 45;  // "Buy 2 GET 1 free"
            } else if (const auto* pl = shouted && tail.empty() && base.back() == 'S'
                                            ? lp::lower_initialism(lp::lower(base))
                                            : nullptr;
                       pl && pl->back() == 's') {
                spoken = std::string(*pl);  // "GPUS" -> GPUs
            } else if (shouted && (base == "IT" || base == "US") && tail.empty() && i > 0 && doc.word(i - 1) &&
                       DETERMINERS.contains(doc.low(i - 1))) {
                continue;  // "THE IT DESK", "THE US MARKET"
            } else if (lp::keep_caps(base) && !(shouted && tail.empty() && lp::english(lp::lower(base)))) {
                if (end != tail) spoken = std::string(base) + end;
            } else {
                std::string low = lp::lower(base);
                bool eng = lp::english(low) || english_like(low);
                size_t prev = doc.prev_word(i);
                bool code_cue = prev != Doc::npos && CODE_CUES.contains(doc.low(prev));
                bool id_cue = prev != Doc::npos && ID_CUES.contains(doc.low(prev));
                // promo words after a code cue: "use code SPRING", "code SAVEBIG" (Save big); "code MIDO" is spelled
                std::string two = code_cue && !eng && tail.empty() ? lp::two_words(low) : std::string();
                size_t cut = two.find(' ');
                if (code_cue && base.size() >= 4 && eng) {
                    spoken = lp::title(base), score = 45;
                } else if (cut != std::string::npos && cut >= 3 && two.size() - cut - 1 >= 3) {
                    spoken = rv::capital(two), score = 45;
                } else if (shouted && eng && tail.empty() && after_number && PRODUCT_WORDS.contains(base)) {
                    spoken = lp::title(base), score = 45;  // "IPHONE 15 PRO" -> Pro
                } else if (shouted && eng && (tail.empty() || (tail.back() == 'S' && base.size() >= 2))) {
                    spoken = low + end, score = 45;  // "IT'S", "THAT'S" -> it's, that's
                } else if (!shouted && tail.empty() && lp::emphasis(base)) {
                    spoken = low, score = 45;
                } else if (!shouted && tail.empty() && eng && base.size() >= 4) {
                    spoken = low, score = 40;
                } else if (base.size() >= 4 && !eng &&
                           ((lp::pronounceable(low) &&
                             (shouted || heading.caps_line(i) || stressed_syllable(doc, i))) ||
                            (!id_cue && lp::word_like(base)))) {
                    spoken = lp::title(base) + end, score = 35;
                } else if (end != tail) {
                    spoken = std::string(base) + end;
                }
            }
            if (!spoken.empty() && spoken != w) out.tokens(i, i, spoken, score);
        }
        run.clear();
    };
    for (size_t i = 0; i < n; ++i) {
        const rv::Token& t = doc[i];
        if (t.newline_before && !run.empty()) flush();
        if (t.kind == rv::Kind::Punct) {
            if (!run.empty() && sentence_end(doc, i)) flush();
            continue;
        }
        if (t.kind == rv::Kind::Number) continue;
        std::string_view base, tail;
        bool caps = (t.casing == rv::UPPER || t.casing == rv::MIXED) && caps_form(doc.str(i), base, tail);
        if (!caps) {
            if (!run.empty()) flush();
            continue;
        }
        // glued to a number or part of a dotted initialism, identifier, path or handle: other rules
        bool glued = (doc.joined(i) && doc.number(i - 1)) || (doc.joined(i + 1) && doc.number(i + 1)) ||
                     (base.size() == 1 && doc.punct(i + 1, '.') && doc.joined(i + 1)) ||
                     (doc.joined(i) && doc.punct(i - 1, '.') && i >= 2 && doc.joined(i - 1) &&
                      doc.word(i - 2) && doc.str(i - 2).size() == 1);
        if (glued || lp::code_context(doc, i)) continue;
        run.push_back(i);
    }
    if (!run.empty()) flush();
}

// Mixed-case names: one capital for what is read as a word, capitals for what is spelled ("GitLab" ->
// "Gitlab", "OpenAI" -> "Open AI", "macOS" -> "Mac OS").
void mixed_at(const Doc& doc, rv::Out& out, size_t i) {
    std::string_view w = doc.str(i), base, tail;
    if (caps_form(w, base, tail)) return;  // "GPUs", "CEO's": caps_words
    if (lp::code_context(doc, i) || lp::unit_word(w) || lp::shorthand(w)) return;
    if (doc.joined(i) && (doc.number(i - 1) || doc.punct(i - 1, '/'))) return;  // "5kWh", "u/TechGuru"
    if (doc.joined(i + 1) && doc.number(i + 1)) return;  // "TontaubeV1": codes; "aGVsbG8...": copied
    // a possessive name keeps its ending: "OpenAI's"
    size_t ap = w.find('\'');
    if (ap == std::string_view::npos) ap = w.find("\xE2\x80\x99");
    std::string_view name = ap == std::string_view::npos ? w : w.substr(0, ap);
    std::string_view end = ap == std::string_view::npos ? std::string_view() : w.substr(ap);
    if (!end.empty() && end != "'s" && end != "\xE2\x80\x99s") return;
    std::string spoken = lp::mixed_word(name);
    if (spoken.empty()) return;
    out.tokens(i, i, spoken + std::string(end), lp::respelling(name) ? 55 : 45);
}

// Initialisms written in lower case get their capitals ("nda" -> "NDA"), chat shorthand said as letters too
// ("lol" -> "LOL").
void lower_at(const Doc& doc, rv::Out& out, size_t i) {
    std::string_view w = doc.str(i);
    if (w.size() < 2 || w.size() > 7) return;
    const std::string_view* r = lp::lower_initialism(w);
    std::string tail;
    if (!r) {
        size_t ap = w.find('\'');
        if (ap == std::string_view::npos || w.substr(ap) != "'s") return;
        r = lp::lower_initialism(w.substr(0, ap));
        tail = "'s";
        if (!r) return;
    }
    if (lp::code_context(doc, i)) return;
    if (doc.joined(i) && doc[i - 1].kind != rv::Kind::Punct) return;
    if (doc.joined(i + 1) && doc[i + 1].kind != rv::Kind::Punct) return;
    if (doc.joined(i + 1) && (doc.punct(i + 1, '.') || doc.punct(i + 1, ':')) && doc.joined(i + 2) &&
        doc[i + 2].kind != rv::Kind::Punct)
        return;  // "api.example.com", "doi:10..."
    if (i > 0 && doc.number(i - 1)) return;  // a unit after a number
    out.tokens(i, i, std::string(*r) + tail, 55);
}

// Initialisms with periods: "U.S.A." -> "USA", "U.K." -> "UK", "a.k.a." -> "AKA". The last period stays when
// it also ends the sentence. Returns the last token used.
size_t dotted_at(const Doc& doc, rv::Out& out, size_t i) {
    static const rv::WordSet LOWER_FORMS{"a.k.a.", "n.b.", "p.s."};
    const size_t n = doc.size();
    if (doc.joined(i) && (doc[i - 1].kind != rv::Kind::Punct || doc.punct(i - 1, '.'))) return i;
    size_t j = i;
    std::string letters;
    while (j + 1 < n && doc.word(j) && doc.str(j).size() == 1 && doc.punct(j + 1, '.') && doc.joined(j + 1) &&
           (j == i || doc.joined(j))) {
        letters += doc.str(j);
        if (j + 2 < n && doc.joined(j + 2) && doc.word(j + 2) && doc.str(j + 2).size() == 1) j += 2;
        else break;
    }
    size_t last_dot = j + 1;
    if (letters.size() >= 2 && j > i && !(doc.punct(last_dot, '.') && doc.joined(last_dot)) &&
        doc.str(j).size() == 1 && doc.joined(j) && !(doc.joined(j + 1) && alnum_tok(doc, j + 1))) {
        // "U.S.A" without the last period; "M.D.s" -> MDs
        std::string all = letters + std::string(doc.str(j));
        if (lp::is_upper_word(all)) out.tokens(i, j, all, 60);
        else if (doc.str(j) == "s" && lp::is_upper_word(letters)) out.tokens(i, j, all, 63);
        return j;
    }
    if (letters.size() < 2 || last_dot >= n || !doc.punct(last_dot, '.')) return j;
    if (doc.joined(last_dot + 1) && doc[last_dot + 1].kind != rv::Kind::Punct) {
        // "M.D.s" -> MDs; "I.Q.5" is something else
        if (doc.word(last_dot + 1) && doc.str(last_dot + 1) == "s" && lp::is_upper_word(letters) &&
            !(doc.joined(last_dot + 2) && doc[last_dot + 2].kind != rv::Kind::Punct))
            out.tokens(i, last_dot + 1, letters + "s", 63);
        return last_dot;
    }
    bool all_upper = lp::is_upper_word(letters);
    std::string written(doc.slice(doc[i].start, doc[last_dot].end));
    if (!all_upper && (!LOWER_FORMS.contains(lp::lower(written)) || lp::lower(written) != written)) return last_dot;
    // the last period ends the sentence: end of text or line, or a word that opens a sentence
    bool keep_dot = false;
    size_t next = last_dot + 1;
    while (next < n && doc[next].kind == rv::Kind::Punct && !doc[next].newline_before &&
           (doc[next].cp == '"' || doc[next].cp == 0x201D || doc[next].cp == ')' || doc[next].cp == '*' ||
            doc[next].cp == '_' || doc[next].cp == 0x2019))
        ++next;
    bool after_markup = next > last_dot + 1;
    if (next < n && after_markup && doc[next].kind == rv::Kind::Punct && (doc[next].cp == '*' || doc[next].cp == '_'))
        while (next < n && doc[next].kind == rv::Kind::Punct && (doc[next].cp == '*' || doc[next].cp == '_')) ++next;
    if (next >= n || doc[next].newline_before) {
        keep_dot = !(i == 0 && next >= n);  // "U.S.A." alone is a fragment: "USA"
    } else if ((next == last_dot + 1 || after_markup) && doc.word(next)) {
        std::string_view nw = doc.str(next);
        std::string l = lp::lower(letters);
        // a sentence opener, or a Roman numeral heading such as "II." after it
        bool roman = rv::roman_value(nw) > 0 && lp::is_upper_word(nw) && doc.punct(next + 1, '.');
        keep_dot = nw[0] >= 'A' && nw[0] <= 'Z' && (lp::sentence_opener(nw) || l == "ps" || l == "nb" || roman);
    }
    std::string spoken = lp::upper(letters);
    // "P.S. I hope": a postscript before "I" takes a comma
    if (keep_dot && lp::lower(letters) == "ps" && doc.word(last_dot + 1) && doc.str(last_dot + 1) == "I") {
        out.tokens(i, last_dot, "PS,", 63);
        return last_dot;
    }
    // the period that also ends the sentence stays: "from the U.K. The FBI" -> UK. The
    out.tokens(i, keep_dot ? last_dot - 1 : last_dot, spoken, keep_dot ? 63 : 60);
    return last_dot;
}

// Initialisms joined by '/' or '&': "CI/CD" -> "CI CD", "I/O" -> "IO", "R&D" -> "R and D". Returns the last
// token used.
size_t joined_at(const Doc& doc, rv::Out& out, size_t i) {
    const size_t n = doc.size();
    if (doc.str(i).size() > 5) return i;
    if (doc.joined(i) && doc[i - 1].kind != rv::Kind::Punct) return i;
    if (doc.joined(i) && (doc.punct(i - 1, '/') || doc.punct(i - 1, '&') || doc.punct(i - 1, '.'))) return i;
    if (!(doc.joined(i + 1) && (doc.punct(i + 1, '/') || doc.punct(i + 1, '&')))) return i;
    size_t parts[8] = {i};
    char seps[8];
    size_t np = 1, j = i;
    while (np < 8 && j + 2 < n && doc.joined(j + 1) && doc.joined(j + 2) && doc.word(j + 2) &&
           doc[j + 2].casing == rv::UPPER && doc.str(j + 2).size() <= 5) {
        if (doc.punct(j + 1, '/')) seps[np - 1] = '/';
        else if (doc.punct(j + 1, '&')) seps[np - 1] = '&';
        else break;
        j += 2;
        parts[np++] = j;
    }
    // "R&amp;D"
    if (np == 1 && j + 4 < n && doc.punct(j + 1, '&') && doc.is(j + 2, "amp") && doc.punct(j + 3, ';') &&
        doc.word(j + 4) && doc[j + 4].casing == rv::UPPER && doc.joined(j + 4)) {
        seps[0] = '&';
        j += 4;
        parts[np++] = j;
    }
    if (np < 2 || (doc.joined(j + 1) && doc[j + 1].kind != rv::Kind::Punct) ||
        (doc.joined(j + 1) && (doc.punct(j + 1, '/') || doc.punct(j + 1, '&'))))
        return j;
    if (np == 2 && seps[0] == '/' && doc.str(parts[0]) == "N" && doc.str(parts[1]) == "A") return j;  // N/A
    std::string spoken;
    for (size_t k = 0; k < np; ++k) {
        std::string_view w = doc.str(parts[k]);
        if (k > 0) {
            bool glue = seps[k - 1] == '/' && w.size() == 1 && doc.str(parts[k - 1]).size() == 1;
            spoken += seps[k - 1] == '&' ? " and " : glue ? "" : " ";
        }
        spoken += lp::caps_word(w);
    }
    out.tokens(i, j, spoken, 60);
    return j;
}

// Product names joined by a hyphen to a model word: "Pico-Lite" -> Pico Lite, "Mistral-Nemo" -> Mistral Nemo.
// Returns the last token used.
size_t name_hyphen_at(const Doc& doc, rv::Out& out, size_t i) {
    static const rv::WordSet SECOND{"air", "base", "chat", "code", "coder", "core", "edge", "flash", "go",
                                    "large", "lite", "llama", "max", "medium", "mini", "mobile", "nano", "nemo",
                                    "neo", "nova", "plus", "prime", "pro", "small", "speech", "tiny", "turbo",
                                    "ultra", "vision"};
    if (i + 2 >= doc.size() || !doc.punct(i + 1, '-') || !doc.joined(i + 1) || !doc.word(i + 2) ||
        !doc.joined(i + 2))
        return i;
    if (doc.joined(i) && doc[i - 1].kind != rv::Kind::Punct) return i;
    if (doc.joined(i + 3) && (doc[i + 3].kind != rv::Kind::Punct || doc.punct(i + 3, '-'))) return i;
    // "Halo-S", "Rouge-L": a name and a model letter; "Edge-A" (an English word) keeps its hyphen
    bool letter = doc.str(i + 2).size() == 1 && doc[i + 2].casing == rv::UPPER && !lp::english(doc.low(i)) &&
                  doc.str(i).size() >= 3;
    if (!letter && (doc[i + 2].casing != rv::TITLE || !SECOND.contains(doc.low(i + 2)))) return i;
    out.tokens(i, i + 2, std::string(doc.str(i)) + " " + std::string(doc.str(i + 2)), 50);
    return i + 2;
}

// A name with a plus: "OPEC+" -> Opec plus, "Disney+" -> Disney plus, "A+" -> A plus.
void plus_at(const Doc& doc, rv::Out& out, size_t i) {
    if (doc.joined(i) && doc[i - 1].kind != rv::Kind::Punct) return;
    if (doc.joined(i) && (doc.punct(i - 1, '+') || doc.punct(i - 1, '.') || doc.punct(i - 1, '/') ||
                          doc.punct(i - 1, '-') || doc.punct(i - 1, '_')))
        return;
    // "Notepad++": two plus signs
    bool twice = doc.punct(i + 2, '+') && doc.joined(i + 2) && doc.str(i).size() >= 2 &&
                 !(doc.joined(i + 3) && (alnum_tok(doc, i + 3) || doc.punct(i + 3, '+') || doc.punct(i + 3, '=')));
    if (!twice && doc.joined(i + 2) && (alnum_tok(doc, i + 2) || doc.punct(i + 2, '+') || doc.punct(i + 2, '=') ||
                                        doc.punct(i + 2, '/')))
        return;
    std::string_view w = doc.str(i);
    if (lp::code_context(doc, i)) return;
    std::string name;
    if (doc[i].casing == rv::UPPER) name = w.size() == 1 ? std::string(w) : lp::caps_word(w);
    else if (doc[i].casing == rv::TITLE) name = std::string(w);
    else if (doc[i].casing == rv::MIXED) name = lp::name_word(w);
    else return;
    if (twice) out.tokens(i, i + 2, name + " plus plus", 60);
    else out.tokens(i, i + 1, name + " plus", 60);
}

// Words: mixed-case names, lower-case initialisms, dotted and joined initialisms, hyphenated product names.
RV_RULE("letters_products", words) {
    size_t next_dotted = 0, next_joined = 0, next_hyphen = 0;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.word(i)) continue;
        uint8_t c = doc[i].casing;
        if (c == rv::MIXED) mixed_at(doc, out, i);
        else if (c == rv::LOWER && doc[i].end - doc[i].start <= 7) lower_at(doc, out, i);
        if (!doc.joined(i + 1) || doc[i + 1].kind != rv::Kind::Punct) continue;
        uint32_t p = doc[i + 1].cp;
        if (p == '+') plus_at(doc, out, i);
        if (p == '.' && i >= next_dotted && doc.str(i).size() == 1) next_dotted = dotted_at(doc, out, i) + 1;
        if ((p == '/' || p == '&') && i >= next_joined && c == rv::UPPER) next_joined = joined_at(doc, out, i) + 1;
        if (p == '-' && i >= next_hyphen && c == rv::TITLE) next_hyphen = name_hyphen_at(doc, out, i) + 1;
    }
}

}  // namespace
