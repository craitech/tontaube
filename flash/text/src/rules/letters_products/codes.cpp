// Product names, model numbers, alphanumeric codes, versions, model sizes and standards.
#include <cstdlib>
#include <vector>

#include "rules/letters_products/lp.h"

namespace {

using lp::Doc;
using rv::Kind;

bool alnum_tok(const Doc& d, size_t k) { return k < d.size() && d[k].kind != Kind::Punct; }

bool lower_word(std::string_view w) {
    for (char c : w)
        if (c < 'a' || c > 'z') return false;
    return !w.empty();
}

// "sixteen" -> "sixteens", "ninety" -> "nineties", "six" -> "sixes"
std::string plural(std::string w) {
    if (!w.empty() && w.back() == 'y') return w.substr(0, w.size() - 1) + "ies";
    if (!w.empty() && (w.back() == 'x' || w.back() == 's')) return w + "es";
    return w + "s";
}

// Lower-case words that stay as written inside a product or format name.
const rv::WordSet NAME_WORDS{"air", "alpha", "amd", "arm", "base", "beta", "carbon", "chat",
                             "delta", "dev", "double", "flash", "float", "gamma", "int", "large",
                             "lite", "max", "medium", "micro", "mini", "nano", "neo", "omega", "plus", "preview",
                             "pro", "small", "tiny", "top", "turbo", "uint", "ultra", "vitamin"};
// Names written with one capital that are English words but cue a version when capitalized ("Go 1.22").
const rv::WordSet CAP_VERSION_CUES{"Blender", "Bun", "Dart", "Deno", "Go", "Godot", "Haskell", "Hugo", "Julia",
                                   "Neovim", "Unity", "Vim", "Zig"};
// Names before a model number.
const rv::WordSet MODEL_PREFIX{"gt", "gtx", "mx", "rtx", "rx", "quadro", "radeon", "geforce", "boeing",
                                    "airbus", "cessna", "embraer", "xbox", "rfc", "http", "ryzen", "core",
                                    "xeon", "epyc", "threadripper", "snapdragon", "exynos", "dimensity",
                                    "tegra", "titan", "model", "series"};
// Model prefixes that cue a model number even when written in lower case ("my rtx 4090").
const rv::WordSet LOWER_PREFIX{"geforce", "gtx", "radeon", "rtx", "ryzen", "snapdragon", "threadripper", "xeon"};
// Letters after a model number that belong to the model, not to a serial ("MI300X", "RTX4090Ti").
const rv::WordSet MODEL_SUFFIX{"GT", "GTS", "GTX", "HS", "HX", "KF", "KS", "LE", "Max", "Mini", "Plus", "Pro", "RS",
                               "SE", "SUPER", "Super", "Ti", "Ultra", "XT", "XTX"};
// Words before a code that make it a serial read digit by digit ("license plate JKL987M").
const rv::WordSet SERIAL_CUES{"id", "licence", "license", "no", "number", "plate", "plates", "reg", "registration",
                              "serial", "tag", "tail", "tracking", "vin"};
// Lower-case model names written in a code: one capital.
const rv::WordSet MODEL_NAMES{"deepseek", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "qwen"};
// Words in front of a hyphen and a number that make a compound, not a name ("pre-2020").
const rv::WordSet PREFIXES{"anti", "by", "ca", "circa", "early", "ex", "late", "mid", "non", "post", "pre",
                           "since", "sub", "until"};
// Lower-case letters after a number that are units, ordinals, times or multipliers.
const rv::WordSet NUMBER_ENDINGS{
    "am", "bn", "cm", "d", "day", "days", "ft", "g", "h", "hr", "hrs", "in", "k", "kb", "kg", "km", "l", "lb",
    "lbs", "m", "mb", "mg", "mi", "min", "mins", "ml", "mm", "mn", "mo", "mph", "ms", "nd", "nm", "ns", "oz",
    "pm", "px", "rd", "s", "sec", "secs", "st", "t", "th", "tb", "v", "w", "wk", "wks", "x", "y", "yr", "yrs",
    "e"};
const rv::WordSet SLANG{"b4", "gr8", "2nite", "2day", "l8r", "w8", "m8", "4u", "h8", "2moro", "4ever", "sk8"};

// Numeronyms, read by abbreviations: a code reading must not take them.
const rv::WordSet NUMERONYMS{"a11n", "a11y", "a12n", "c11n", "c14n", "e2e", "f2f", "g10n", "g11n", "i18n", "k8s",
                             "l10n", "m17n", "o11y", "p11n", "p13n", "s10n", "s11n", "t9n", "v12n"};

// Chemical element symbols: a code made of them may be a formula, which the units rules read.
bool element_letters(std::string_view w) {
    static const rv::WordSet EL{"Al", "Ar", "B", "Br", "C", "Ca", "Cl", "Co", "Cr", "Cu", "F", "Fe", "H", "He",
                                "I", "K", "Li", "Mg", "Mn", "N", "Na", "Ne", "Ni", "O", "P", "Pb", "S", "Si",
                                "Sn", "Ti", "U", "Zn"};
    size_t i = 0;
    while (i < w.size()) {
        if (i + 1 < w.size() && w[i + 1] >= 'a' && w[i + 1] <= 'z' && EL.contains(w.substr(i, 2))) i += 2;
        else if (EL.contains(w.substr(i, 1))) i += 1;
        else return false;
    }
    return true;
}

const rv::WordSet SMALL_MOLECULES{"CH4", "CO2", "CO3", "H2", "H2O", "H2O2", "H2S", "N2", "N2O", "NH3", "NO2",
                                  "NO3", "O2", "O3", "PO4", "SO2", "SO3", "SO4"};

// ---------------------------------------------------------------------------------------------------------
// Codes: letters and numbers joined, with hyphens between the parts ("Qwen3-1.7B", "i7-12700K", "SHA-256",
// "COVID-19", "A1B2C3", "IPv4", "x86-64", "CB1-CB3").

struct Item {
    size_t first, last;  // tokens: a word, or a number with its dotted parts
    bool number;
};

struct Segment {
    std::vector<Item> items;
    char sep = 0;  // separator before this segment: '-', '_' or 0x2013 (en dash), 0 for the first
};

// Short function words and abbreviations that are letters in a code unless a code word is near: "HER2" (HER
// two), "REF3B9" (REF three B nine).
const rv::WordSet SHORT_FUNCTION{"ARE", "CAN", "DID", "FOR", "HAD", "HAS", "HER", "HIM", "HIS", "ITS", "MAY", "NOR",
                                 "OUR", "REF", "SHE", "THE", "WAS", "WHO", "YOU"};
// Short English words that start promo codes ("FUN50", "GET25OFF").
const rv::WordSet PROMO_SHORT{"ACE", "ADD", "BIG", "BUY", "FLY", "FUN", "GET", "HOT", "JOY", "NEW", "NOW", "OFF",
                              "TRY", "USE", "WIN", "YAY", "ZEN"};

// Reads one segment's items into words.
std::string read_segment(const Doc& d, const Segment& s, bool serial, bool& said_word, bool promo) {
    std::string out;
    for (size_t k = 0; k < s.items.size(); ++k) {
        const Item& it = s.items[k];
        std::string w;
        if (it.number) {
            std::string written(d.slice(d[it.first].start, d[it.last].end));
            if (it.last > it.first) {
                w = lp::version_words(written);
            } else if (serial && written.size() >= 3) {
                w = rv::digits(written);
            } else if (written.size() == 5 && !(k + 1 < s.items.size() && !s.items[k + 1].number)) {
                w = rv::digits(written);  // a five-digit serial without a suffix letter
            } else {
                w = lp::model_number(written);
            }
        } else {
            std::string_view word = d.str(it.first);
            if (word == "SOC" && k + 1 < s.items.size() && s.items[k + 1].number) {
                w = "Soc";
            } else if (MODEL_NAMES.contains(word)) {
                w = rv::capital(std::string(word));
            } else if (lower_word(word) && word.size() >= 2) {
                bool keep = NAME_WORDS.contains(word) || (word.size() >= 3 && lp::english(word));
                w = keep ? std::string(word) : lp::upper(word);
            } else if (lp::is_upper_word(word) && !lp::respelling(word) && lp::english(lp::lower(word)) &&
                       (PROMO_SHORT.contains(word) || (word.size() >= 4 && !lp::keep_caps(word)) ||
                        (word.size() == 3 && lp::pronounceable(lp::lower(word)) && !lp::keep_caps(word) &&
                         (promo || !SHORT_FUNCTION.contains(word))))) {
                // words in a promo code: the first with one capital, the rest in lower case ("Save fifteen now")
                w = said_word ? lp::lower(word) : lp::title(word);
            } else if (lp::is_upper_word(word) && word.size() >= 5 && !lp::respelling(word) &&
                       !lp::two_words(lp::lower(word)).empty()) {
                w = lp::two_words(lp::lower(word));  // "BUYNOW" -> Buy now
                if (!said_word) w = rv::capital(w);
            } else if (d[it.first].casing == rv::MIXED && word[0] >= 'a' && word[0] <= 'z') {
                // "23andMe": a lower-case word and a name part
                size_t cut = 0;
                while (cut < word.size() && word[cut] >= 'a' && word[cut] <= 'z') ++cut;
                std::string head(word.substr(0, cut));
                w = cut >= 2 && lp::english(head) ? head + " " + lp::name_word(word.substr(cut)) : lp::name_word(word);
            } else {
                w = lp::name_word(word);
            }
            said_word = true;
        }
        if (w.empty()) return "";
        if (!out.empty()) out += ' ';
        out += w;
    }
    return out;
}

// The letters in front of the first number of a segment ("CB" in "CB1"), "" if it does not start with letters.
std::string_view prefix_letters(const Doc& d, const Segment& s) {
    if (s.items.size() != 2 || s.items[0].number || !s.items[1].number) return "";
    return d.str(s.items[0].first);
}

// Reads the run of letters and numbers that starts at token i; returns the last token it looked at.
size_t code_at(const Doc& doc, rv::Out& out, size_t i) {
    {
        if (!alnum_tok(doc, i)) return i;
        if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return i;  // inside a run
        {
            // quick look first: a code needs a word and a number in one run
            size_t j = i;
            bool w = false, num = false;
            while (true) {
                w |= doc.word(j);
                num |= doc.number(j);
                if (alnum_tok(doc, j + 1) && doc.joined(j + 1)) {
                    ++j;
                    continue;
                }
                bool conn = doc.punct(j + 1, '-') || doc.punct(j + 1, '.') || doc.punct(j + 1, '_') ||
                            doc.punct(j + 1, 0x2013) || doc.punct(j + 1, 0x2011);
                if (conn && doc.joined(j + 1) && alnum_tok(doc, j + 2) && doc.joined(j + 2)) {
                    j += 2;
                    continue;
                }
                break;
            }
            if (!w || !num) return j;
        }
        // the run: words and numbers joined, with '-' / '_' / en dash between parts and '.' between numbers
        std::vector<Segment> segs(1);
        size_t j = i;
        bool has_word = false, has_number = false;
        while (true) {
            Segment& s = segs.back();
            if (doc.number(j)) {
                size_t e = j;
                while (doc.punct(e + 1, '.') && doc.joined(e + 1) && doc.number(e + 2) && doc.joined(e + 2)) e += 2;
                s.items.push_back({j, e, true});
                has_number = true;
                j = e;
            } else {
                s.items.push_back({j, j, false});
                has_word = true;
            }
            if (alnum_tok(doc, j + 1) && doc.joined(j + 1)) {
                ++j;
                continue;
            }
            bool hyphen = doc.punct(j + 1, '-') || doc.punct(j + 1, 0x2013) || doc.punct(j + 1, '_') ||
                          doc.punct(j + 1, 0x2011);
            if (hyphen && doc.joined(j + 1) && alnum_tok(doc, j + 2) && doc.joined(j + 2)) {
                char sep = doc.punct(j + 1, '-') || doc.punct(j + 1, 0x2011) ? '-' : doc.punct(j + 1, '_') ? '_' : '\x13';
                // "GPT–4": an en dash between a name and its number is a hyphen
                if (sep == '\x13' && doc.word(j) && doc.str(j).size() >= 2 && doc.number(j + 2) &&
                    doc[j].casing != rv::LOWER && !lp::unit_word(doc.str(j)) && !(j > 0 && doc.number(j - 1)))
                    sep = '-';
                bool after_number = doc.number(j);
                std::string_view next = doc.word(j + 2) ? doc.str(j + 2) : std::string_view();
                // a compound keeps its hyphen: "4-bit", "COVID-19-related", "7B-derived", "a 4-GPU node"
                if (lower_word(next) && !NAME_WORDS.contains(next) && next != "rc" && next != "it" &&
                    (after_number ? next.size() >= 2 : next.size() >= 3 && lp::english(next)))
                    break;
                if (after_number && segs.size() == 1 && s.items.size() == 1 && next.size() >= 2 &&
                    lp::is_upper_word(next) && !lp::unit_word(next) &&
                    doc.word(j + 3) && doc[j + 3].space_before && doc[j + 3].casing == rv::LOWER &&
                    !lp::model_noun(doc.low(j + 3)))
                    break;
                if (sep == '_' && !(doc.is(j - 1, "x") && doc.str(j) == "86")) break;
                if (sep == '\x13' && !doc.number(j)) break;
                segs.push_back(Segment{});
                segs.back().sep = sep;
                j += 2;
                continue;
            }
            break;
        }
        const size_t last = j;
        if (!has_word || !has_number) return last;
        std::string written(doc.slice(doc[segs[0].items[0].first].start, doc[last].end));
        for (const Segment& s : segs)
            for (const Item& it : s.items)
                if (!it.number)
                    for (char c : doc.str(it.first))
                        if (static_cast<unsigned char>(c) >= 0x80) return last;  // letters of another language
        std::string low = lp::lower(written);
        if (SLANG.contains(written) || NUMERONYMS.contains(low)) return last;
        // part of a larger construct: a path, handle, hashtag, file name, amount or domain
        size_t first = segs[0].items[0].first;
        if (doc.joined(first)) {
            uint32_t p = doc[first - 1].cp;
            if (p == '/' || p == '@' || p == '#' || p == '.' || p == '_' || p == '\\' || p == ':' || p == '$' ||
                p == 0x20AC || p == 0xA3 || p == 0xA5 || p == '~' || p == '=' || p == '+' || p == '-')
                return last;
        }
        // "Galaxy S24+", "SAVE15+": a plus after the code is said
        bool plus = doc.punct(last + 1, '+') && doc.joined(last + 1) &&
                    !(doc.joined(last + 2) && (alnum_tok(doc, last + 2) || doc.punct(last + 2, '+') ||
                                               doc.punct(last + 2, '=')));
        if (doc.joined(last + 1) && !plus) {
            uint32_t p = doc[last + 1].cp;
            if (p == '/' || p == '@' || p == '_' || p == '\\' || p == '%' || p == 0xB0 || p == '=' || p == '+' ||
                p == '(' || ((p == '.' || p == ':') && alnum_tok(doc, last + 2) && doc.joined(last + 2)))
                return last;
        }
        const Segment& s0 = segs[0];
        std::string_view w0 = s0.items[0].number ? std::string_view() : doc.str(s0.items[0].first);
        if (w0 == "UTC" || w0 == "GMT") return last;
        if (!w0.empty() && doc[s0.items[0].first].casing == rv::TITLE && lp::not_model_cue(lp::lower(w0)))
            return last;  // "Mar-2026", "Page-4": dates and references
        if (w0 == "x" && segs.size() == 1 && s0.items.size() == 2 && s0.items[1].number &&
            s0.items[1].first == s0.items[1].last && doc.str(s0.items[1].last).size() <= 2 &&
            doc.str(s0.items[1].last) != "86" && doc.str(s0.items[1].last) != "64" &&
            doc.str(s0.items[1].last) != "32")
            return last;  // "x2": a multiplier
        // a lone 'v' before a number is a version: the version rule
        if (w0 == "v" && s0.items.size() >= 2 && s0.items[1].number) return last;
        // a number followed by a unit, ordinal, time, scale or multiplier is not a code
        bool skip = false;
        bool letters_first = !s0.items[0].number;
        for (const Segment& s : segs)
            for (size_t k = 0; k + 1 < s.items.size(); ++k) {
                if (!s.items[k].number || s.items[k + 1].number) continue;
                std::string_view w = doc.str(s.items[k + 1].first);
                std::string_view digs = doc.str(s.items[k].last);
                bool lone = s.items.size() == 2 && segs.size() == 1;
                bool tail = k == 0 && k + 2 == s.items.size();  // the part is a number and one word
                if (tail && !letters_first &&
                    (lp::unit_word(w) || w == "AM" || w == "PM" || w == "ST" || w == "ND" || w == "RD" ||
                     w == "TH" || w == "S"))
                    skip = true;
                if (lone && (w == "K" || w == "M" || w == "B" || w == "T") && digs.size() <= 3)
                    skip = true;  // scale rule
                // "10x" is a multiplier; "777X" (a Boeing family) and "Boeing 777X" are models
                bool jet = digs.size() == 3 && digs[0] == '7' && digs[2] == '7' && w == "X";
                bool after_prefix = first > 0 && doc.word(first - 1) && doc[first].space_before &&
                                    MODEL_PREFIX.contains(doc.low(first - 1)) && doc[first - 1].casing != rv::LOWER;
                if ((w == "x" || w == "X") && digs.size() <= 3 && s.items[k].first == s.items[k].last && k == 0 &&
                    !letters_first && !jet && !after_prefix)
                    skip = true;
                // "iPhone 16e": a model letter after a product name
                bool model_e = w == "e" && lone && first > 0 && doc[first].space_before && doc.word(first - 1) &&
                               doc[first - 1].casing == rv::MIXED;
                // "7800x3d": an x inside a model code; "1280x720" is a dimension
                bool inner_x = s.items.size() == 4 && !letters_first && doc.is(s.items[1].first, "x") &&
                               s.items[2].number && doc.str(s.items[2].last).size() <= 2 && !s.items[3].number;
                if (lower_word(w) && NUMBER_ENDINGS.contains(w) && !(w == "s" && letters_first && k + 2 == s.items.size()) &&
                    !(letters_first && w.size() == 1 && w != "s") && !model_e && !inner_x)
                    skip = true;
                if (w == "p" && lone) skip = true;  // resolution: its own rule
                if (lower_word(w) && w.size() >= 2 && !NAME_WORDS.contains(w) && w != "rc" && k == 0 &&
                    s.items.size() == 2 && !letters_first)
                    skip = true;  // "3rd", "2nite"
            }
        // "2.4–3.1M", "10-15k": a range of amounts with a scale letter belongs to the scale rule
        if (segs.size() == 2 && s0.items.size() == 1 && s0.items[0].number && segs[1].items.size() == 2 &&
            segs[1].items[0].number && !segs[1].items[1].number) {
            std::string_view sc = doc.str(segs[1].items[1].first);
            if (sc == "K" || sc == "M" || sc == "B" || sc == "T" || sc == "k") return last;
        }
        if (skip) return last;
        // measures belong to the units rules: "16-kHz", "1.6-kW", "50 m2", "12 tCO2e"
        if (segs.size() == 2 && s0.items.size() == 1 && s0.items[0].number && segs[1].items.size() == 1 &&
            !segs[1].items[0].number && doc.str(segs[1].items[0].first).size() >= 2 &&  // "Form 20-F" is a code
            lp::unit_word(doc.str(segs[1].items[0].first)))
            return last;
        if (segs.size() == 1 && s0.items.size() == 2 && letters_first && s0.items[1].number && first > 0 &&
            doc.number(first - 1) && (doc.str(s0.items[1].first) == "2" || doc.str(s0.items[1].first) == "3") &&
            (w0 == "m" || w0 == "cm" || w0 == "mm" || w0 == "km" || w0 == "ft" || w0 == "in" || w0 == "yd"))
            return last;
        if (w0.size() >= 3 && w0.substr(w0.size() - 2) == "CO" && s0.items.size() >= 2 && s0.items[1].number) {
            std::string_view mass = w0.substr(0, w0.size() - 2);
            if (mass == "t" || mass == "kt" || mass == "Mt" || mass == "Gt" || mass == "kg" || mass == "g")
                return last;
        }
        // a compound such as "pre-2020"
        if (segs.size() >= 2 && s0.items.size() == 1 && !s0.items[0].number &&
            PREFIXES.contains(doc.low(s0.items[0].first)))
            return last;
        // a long or random-looking string is copied as written (identifier rules): many alternations with
        // mixed-case or lower-case letter groups, a long hex string, scientific notation, "base64"
        if (written.size() > 20 || low == "base64" || low == "base32") return last;
        bool random = false;
        for (const Segment& s : segs) {
            if (s.items.size() < 4) continue;
            int odd = 0;
            for (const Item& it : s.items)
                if (!it.number) {
                    std::string_view w = doc.str(it.first);
                    if (!lp::is_upper_word(w) && w.size() >= 2) ++odd;
                    if (doc[it.first].casing == rv::MIXED || doc[it.first].casing == rv::TITLE) odd += 2;
                    if (lower_word(w) && w.size() == 1) ++odd;
                }
            random |= odd >= 3 || s.items.size() >= 9;
        }
        bool long_hex = written.size() >= 8;
        for (char c : written) long_hex &= (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        // identifiers with several code parts: "CVE-2026-0417", "ORD-7731-QX9P-0042", "4A7C-91FD"
        if (segs.size() >= 2) {
            bool id = false;
            for (const Segment& s : segs) {
                id |= s.items.size() >= 4;
                for (const Item& it : s.items) id |= it.number && segs.size() >= 3 && doc.str(it.last).size() >= 4;
            }
            if (id && !(lp::is_upper_word(w0) == false && doc[s0.items[0].first].casing == rv::TITLE)) return last;
        }
        if (random || long_hex || low.substr(0, 2) == "0x") return last;
        if (s0.items.size() >= 3 && s0.items[0].number && (doc.is(s0.items[1].first, "e")) && s0.items[2].number)
            return last;  // 2.3e5
        // a serial: letters, numbers and letters again in one part ("JKL987M"), or a hex string
        size_t max_items = 0;
        for (const Segment& s : segs) max_items = std::max(max_items, s.items.size());
        bool hex = written.size() >= 6 && segs.size() == 1;
        for (char c : written) hex &= (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        bool serial = max_items >= 3 || hex;
        if (serial && !hex && segs.size() == 1 && s0.items.size() == 3 && letters_first && s0.items[1].number) {
            // letters, a model number and a model letter or suffix: "MI300X", "TR850X", "A320neo"; a serial only
            // after a cue word ("tracking number PQR987M") or with two letters at both ends ("QW345RT")
            std::string_view tail = doc.str(s0.items[2].first);
            bool suffix = MODEL_SUFFIX.contains(tail) || NAME_WORDS.contains(tail) || tail == "s" ||
                          (tail.size() == 1 && tail[0] >= 'A' && tail[0] <= 'Z');
            bool cue = false;
            for (size_t k = first, seen = 0; k > 0 && seen < 4 && !cue; --k)
                if (doc.word(k - 1)) cue = SERIAL_CUES.contains(doc.low(k - 1)), ++seen;
            if (suffix && !cue) serial = false;
        }
        if (serial && !hex && segs.size() == 1 && !letters_first && s0.items.size() >= 3) {
            // a model number with a short suffix code: "7800X3D", "9950X3D"
            std::string_view d0 = doc.str(s0.items[0].last);
            bool model = s0.items[0].first == s0.items[0].last && d0.size() >= 3 && d0.size() <= 5 && d0[0] != '0';
            for (size_t k = 1; k < s0.items.size() && model; ++k)
                model = doc.str(s0.items[k].first).size() <= (s0.items[k].number ? 2u : 3u) &&
                        s0.items[k].first == s0.items[k].last;
            if (model) serial = false;
        }
        if (serial && !hex) {
            // promo codes with words and formulas or virus names read their numbers whole
            bool words = false, elements_only = true;
            for (const Segment& s : segs)
                for (const Item& it : s.items) {
                    if (it.number) continue;
                    std::string_view w = doc.str(it.first);
                    std::string l = lp::lower(w);
                    words |= w.size() >= 3 && (lp::english(l) || !lp::two_words(l).empty());
                    elements_only &= element_letters(w);
                }
            if (words || elements_only) serial = false;
        }
        // a chemical formula is left to the units rules: element symbols with mixed case ("NaCl2"), a known
        // small molecule ("CO2"), or element symbols alternating with numbers ("C6H12O6", "H2SO4")
        // ("CO2e": a formula and "e" for equivalent)
        bool formula = segs.size() == 1 && !s0.items[0].number;
        bool mixed_symbol = false;
        size_t n_items = s0.items.size();
        if (n_items >= 3 && !s0.items.back().number && s0.items[n_items - 2].number &&
            (doc.str(s0.items.back().first) == "e" || doc.str(s0.items.back().first) == "eq"))
            --n_items;
        for (size_t k = 0; k < n_items; ++k)
            if (!s0.items[k].number) {
                formula &= element_letters(doc.str(s0.items[k].first));
                mixed_symbol |= doc[s0.items[k].first].casing == rv::TITLE || doc[s0.items[k].first].casing == rv::MIXED;
            }
        std::string_view molecule(doc.slice(doc[s0.items[0].first].start, doc[s0.items[n_items - 1].last].end));
        formula &= mixed_symbol || n_items >= 3 || SMALL_MOLECULES.contains(molecule);
        // words in capitals make a promo code, not a formula: "FUN50OFF"
        if (formula && !mixed_symbol)
            for (size_t k = 0; k < n_items && formula; ++k)
                if (!s0.items[k].number && doc.str(s0.items[k].first).size() >= 3 &&
                    lp::english(lp::lower(doc.str(s0.items[k].first))))
                    formula = false;
        // a code word just before: a three-letter word in the code is a word ("code SKY2"), else letters ("HER2")
        static const rv::WordSet PROMO_CUES{"code", "codes", "coupon", "enter", "promo", "promotion", "use",
                                            "voucher"};
        bool promo = false;
        for (size_t k = first, seen = 0; k > 0 && seen < 3 && !promo; --k)
            if (doc.word(k - 1)) promo = PROMO_CUES.contains(doc.low(k - 1)), ++seen;
        // read the parts
        std::string spoken;
        bool said_word = false, ok = true;
        for (size_t k = 0; k < segs.size() && ok; ++k) {
            const Segment& s = segs[k];
            std::string part = read_segment(doc, s, serial, said_word, promo);
            if (part.empty()) {
                ok = false;
                break;
            }
            if (k > 0) {
                const Segment& p = segs[k - 1];
                bool prev_num = p.items.back().number, this_num = s.items.front().number;
                std::string_view pa = prefix_letters(doc, p), pb = prefix_letters(doc, s);
                if (!pa.empty() && pa == pb) spoken += " to ";  // "CB1-CB3"
                else if (s.sep == '\x13') spoken += " to ";
                else if (prev_num && this_num && !(p.items.size() == 2 && doc.is(p.items[0].first, "x")))
                    spoken += ", ";  // "Qwen3-1.7B": three, one point seven B
                else spoken += ' ';
            }
            // plural of a model: "F-16s" -> F sixteens
            if (k + 1 == segs.size() && s.items.size() >= 2 && doc.str(s.items.back().first) == "s" &&
                !s.items.back().number && s.items[s.items.size() - 2].number) {
                size_t cut = part.rfind(' ');
                part = plural(part.substr(0, cut));
            }
            spoken += part;
        }
        if (!ok || spoken.empty()) return last;
        if (plus && !formula) {
            out.tokens(first, last + 1, spoken + " plus", 62);
            return last + 1;
        }
        out.tokens(first, last, spoken, formula ? 45 : 62);
        return last;
    }
}

// Virus lineages near a variant word: "Omicron BA.5" -> BA five, "B.1.1.529 variant" -> B one one five two nine.
void lineage_at(const Doc& doc, rv::Out& out, size_t i) {
    static const rv::WordSet CUES{"delta", "lineage", "lineages", "omicron", "strain", "strains", "sublineage",
                                  "subvariant", "subvariants", "variant", "variants"};
    if (!doc.word(i) || doc.str(i).size() > 3 || doc[i].casing != rv::UPPER) return;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return;
    size_t e = i;
    std::string spoken(doc.str(i));
    while (doc.punct(e + 1, '.') && doc.joined(e + 1) && doc.number(e + 2) && doc.joined(e + 2)) {
        spoken += " " + rv::digits(doc.str(e + 2));
        e += 2;
    }
    if (e == i || (doc.joined(e + 1) && doc[e + 1].kind != Kind::Punct)) return;
    if (doc.joined(e + 1) && doc.punct(e + 1, '.') && doc.joined(e + 2) && doc[e + 2].kind != Kind::Punct) return;
    bool cue = false;
    for (size_t k = i, seen = 0; k > 0 && seen < 3 && !cue; --k)
        if (doc.word(k - 1)) cue = CUES.contains(doc.low(k - 1)), ++seen;
    for (size_t k = e + 1, seen = 0; k < doc.size() && seen < 2 && !cue; ++k)
        if (doc.word(k)) cue = CUES.contains(doc.low(k)), ++seen;
    if (cue) out.tokens(i, e, spoken, 70);
}

// A network generation written in lower case: "the 5g was fine", "4g coverage" -> five G, four G.
void network_at(const Doc& doc, rv::Out& out, size_t i) {
    static const rv::WordSet NEXT{"band", "bands", "connection", "coverage", "data", "modem", "network", "networks",
                                  "phone", "phones", "plan", "plans", "reception", "rollout", "service", "signal",
                                  "speed", "speeds", "tower", "towers"};
    static const rv::WordSet VERBS{"is", "was", "works", "worked", "dropped", "died"};
    if (!doc.number(i) || doc.str(i).size() != 1 || doc.str(i)[0] < '2' || doc.str(i)[0] > '6') return;
    if (!doc.word(i + 1) || !doc.joined(i + 1) || doc.str(i + 1) != "g") return;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return;
    if (alnum_tok(doc, i + 2) && doc.joined(i + 2)) return;
    size_t nx = doc.next_word(i + 1);
    bool net = nx == i + 2 && doc[nx].space_before &&
               (NEXT.contains(doc.low(nx)) || (VERBS.contains(doc.low(nx)) && i > 0 && doc.is(i - 1, "the")));
    if (net) out.tokens(i, i + 1, rv::cardinal_digits(doc.str(i)) + " G", 65);
}

// Team names made of a number and "ers": "76ers" -> seventy-sixers, "49ers" -> forty-niners.
void ers_at(const Doc& doc, rv::Out& out, size_t i) {
    if (!doc.number(i) || doc.str(i).size() != 2 || doc.str(i)[1] == '0' || !doc.word(i + 1) || !doc.joined(i + 1) ||
        doc.str(i + 1) != "ers")
        return;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return;
    if (alnum_tok(doc, i + 2) && doc.joined(i + 2)) return;
    std::string w = rv::cardinal_digits(doc.str(i));
    if (w.empty()) return;
    w += w.back() == 'e' ? "rs" : "ers";
    out.tokens(i, i + 1, w, 65);
}

// Business and network shorthand with '2': "B2B" -> B to B, "P2P" -> P to P.
void two_shorthand_at(const Doc& doc, rv::Out& out, size_t i) {
    static const rv::WordSet LETTERS{"A", "B", "C", "D", "E", "G", "M", "O", "P", "R", "T", "X"};
    if (i + 2 >= doc.size()) return;
    if (!doc.word(i) || !doc.joined(i + 1) || !doc.joined(i + 2) || doc.str(i + 1) != "2") return;
    if (!doc.word(i + 2) || doc.str(i).size() != 1 || doc.str(i + 2).size() != 1) return;
    bool same = doc.str(i) == doc.str(i + 2) && doc[i].casing == rv::UPPER;
    if (!same && (!LETTERS.contains(doc.str(i)) || !LETTERS.contains(doc.str(i + 2)))) return;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return;
    if (alnum_tok(doc, i + 3) && doc.joined(i + 3)) return;
    out.tokens(i, i + 2, std::string(doc.str(i)) + " to " + std::string(doc.str(i + 2)), 70);
}

// ---------------------------------------------------------------------------------------------------------
// Model numbers after a name: "RTX 4090" -> RTX forty ninety, "Boeing 737" -> Boeing seven thirty-seven,
// "Model 9000" -> Model nine thousand, "the 4080 card".

// Capitals before a number that make it an amount, a year, a time or an identifier.
const rv::WordSet NOT_MODEL_CAPS{
    "AD", "AED", "AM", "AUD", "BC", "BCE", "BRL", "CAD", "CE", "CET", "CHF", "CNY", "DKK", "DOI", "EST", "EUR",
    "FTSE", "FY", "GBP", "GMT", "HKD", "ID", "INR", "ISBN", "ISSN", "JPY", "KRW", "MXN", "NOK", "NZD", "PIN",
    "PM", "PMID", "PST", "PT", "Q", "RMB", "SEK", "SGD", "SKU", "USD", "UTC", "ZAR", "ZIP"};


// The number at token i stands alone: no decimals, separators, units or ranges attached ("s" may follow).
bool lone_number(const Doc& doc, size_t i) {
    if (i == 0 || !doc.number(i) || !doc[i].space_before) return false;
    std::string_view digs = doc.str(i);
    if (digs.size() < 3 || digs.size() > 5 || digs[0] == '0') return false;
    if (!doc.joined(i + 1)) return true;
    if (doc.punct(i + 1, '.') && !(alnum_tok(doc, i + 2) && doc.joined(i + 2))) return true;
    if (doc.punct(i + 1, ',') && !doc.joined(i + 2)) return true;
    if (doc.punct(i + 1, ':') && !(doc.number(i + 2) && doc.joined(i + 2))) return true;  // "RTX 4090: 24 GB"
    // "RTX 4090-class cards": a compound with a lower-case word
    if (doc.punct(i + 1, '-') && doc.word(i + 2) && doc.joined(i + 2) && doc[i + 2].casing == rv::LOWER &&
        doc.str(i + 2).size() >= 3)
        return true;
    if (doc.word(i + 1) && doc.str(i + 1) == "s") return true;
    return !(alnum_tok(doc, i + 1) || doc.punct(i + 1, '.') || doc.punct(i + 1, ',') || doc.punct(i + 1, ':') ||
             doc.punct(i + 1, '/') || doc.punct(i + 1, '-') || doc.punct(i + 1, 0x2013) || doc.punct(i + 1, '%'));
}

// The score of reading the lone number at token i as a model number from the word before it; 0 if it is not
// one ("RTX 4090", "Boeing 737", "Ridge 880", "the 4080 card").
int model_score(const Doc& doc, size_t i) {
    static const rv::WordSet AFTER{"board", "boards", "box", "boxes", "card", "cards", "chip", "chips", "gpu",
                                   "gpus", "host", "hosts", "laptop", "laptops", "line", "machine", "machines",
                                   "model", "models", "name", "rig", "rigs", "series", "server", "servers",
                                   "workstation", "workstations"};
    std::string_view digs = doc.str(i);
    size_t p = i - 1;
    // a model word after the number: "30 737 MAX 8 jets", "the 4080 Super"
    if (doc.word(i + 1) && doc[i + 1].space_before && digs.size() >= 3 &&
        (MODEL_SUFFIX.contains(doc.str(i + 1)) || doc.str(i + 1) == "MAX"))
        return 60;
    // a model number after a model name and number: "Dell XPS 13 9340"
    if (digs.size() == 4 && doc.number(p) && doc[p].space_before && doc.str(p).size() <= 2 && p > 0 &&
        doc.word(p - 1) && (doc[p - 1].casing == rv::UPPER || doc[p - 1].casing == rv::MIXED) &&
        doc.str(p - 1).size() >= 2 && !NOT_MODEL_CAPS.contains(doc.str(p - 1)) &&
        !(doc.word(i + 1) && (lp::count_noun(doc.low(i + 1)) || lp::unit_word(doc.str(i + 1)))))
        return 60;
    if (!doc.word(p)) return 0;
    std::string_view pw = doc.str(p);
    std::string_view pl = doc.low(p);
    const rv::Token& pt = doc[p];
    int score = 0;
    if (MODEL_PREFIX.contains(pl) && (pt.casing != rv::LOWER || LOWER_PREFIX.contains(pl))) {
        score = 70;
    } else if (pw == "P" && p >= 2 && doc.punct(p - 1, '&') && doc.joined(p) && doc.is(p - 2, "s")) {
        score = 60;  // S&P 500
    } else if (pl == "the" || pl == "a" || pl == "an" || pl == "older" || pl == "newer" || pl == "single") {
        size_t nx = i + 1;
        if (digs.size() == 4 && doc.word(nx) && AFTER.contains(doc.low(nx))) score = 60;
        // the shape of a graphics card number ("the 4090", "a 5080 setup") unless it counts something
        bool card_shape = digs.size() == 4 && digs[1] == '0' && digs[2] >= '5' && digs[3] == '0';
        if (!score && card_shape && !(doc.word(nx) && (lp::count_noun(doc.low(nx)) || lp::unit_word(doc.str(nx)) ||
                                                       (doc.low(nx).size() > 3 && doc.low(nx).back() == 's'))))
            score = 55;
        if (digs.size() == 5 && digs.substr(3) == "00" && doc.word(nx) && !lp::unit_word(doc.str(nx))) score = 50;
    } else if (digs.size() == 5 && !(pt.casing == rv::UPPER || pt.casing == rv::MIXED)) {
        return 0;
    } else if (pl == "iso" || pl == "iec" || lp::not_model_cue(pl) || lp::version_cue(pl)) {
        return 0;  // standards, references, dates, versions
    } else if (pt.casing == rv::UPPER && pw.size() >= 2 && pw.size() <= 5 && !NOT_MODEL_CAPS.contains(pw)) {
        score = 60;  // "RTX 4070", "FTSE 100"
    } else if (pt.casing == rv::MIXED || (pt.casing == rv::TITLE && !doc.sentence_start(p))) {
        score = 60;
    } else if (pt.casing == rv::TITLE && !lp::english(pl)) {
        score = 60;
    }
    // a quantity after a name is still a quantity when a unit follows ("Atlas 990 mAh")
    if (score && doc.word(i + 1) && lp::unit_word(doc.str(i + 1))) return 0;
    return score;
}

// Model numbers: after a name, and the same number again elsewhere in the text ("The 880 offered more").
// A variant number after a model number: "Boeing 787-9" -> seven eighty-seven, nine; "737-800".
void variant_at(const Doc& doc, rv::Out& out, size_t i) {
    if (i == 0 || !doc[i].space_before || !doc.word(i - 1) || doc[i - 1].casing == rv::LOWER) return;
    std::string_view a = doc.str(i);
    if (a.size() < 3 || a.size() > 4 || a[0] == '0' || !MODEL_PREFIX.contains(doc.low(i - 1))) return;
    if (!doc.punct(i + 1, '-') || !doc.joined(i + 1) || !doc.number(i + 2) || !doc.joined(i + 2)) return;
    std::string_view b = doc.str(i + 2);
    if (b.size() > 4 || (b.size() > 1 && b[0] == '0')) return;
    if (doc.joined(i + 3) && (alnum_tok(doc, i + 3) || ((doc.punct(i + 3, '.') || doc.punct(i + 3, '-')) &&
                                                         alnum_tok(doc, i + 4) && doc.joined(i + 4))))
        return;
    out.tokens(i, i + 2, lp::model_number(a) + ", " + lp::model_number(b), 70);
}

// Versions or models joined by a slash after a product name: "Windows 10/11" -> ten slash eleven.
void slash_versions_at(const Doc& doc, rv::Out& out, size_t i) {
    if (i == 0 || !doc[i].space_before || !doc.word(i - 1)) return;
    std::string_view a = doc.str(i);
    if (a.size() > 2 || !doc.punct(i + 1, '/') || !doc.joined(i + 1) || !doc.number(i + 2) || !doc.joined(i + 2))
        return;
    std::string_view b = doc.str(i + 2);
    if (b.size() > 2 || (doc.joined(i + 3) && (alnum_tok(doc, i + 3) || doc.punct(i + 3, '/') ||
                                                (doc.punct(i + 3, '.') && alnum_tok(doc, i + 4)))))
        return;
    std::string_view pw = doc.str(i - 1);
    const rv::Token& pt = doc[i - 1];
    bool cue = (lp::version_cue(doc.low(i - 1)) && pt.casing != rv::LOWER) || CAP_VERSION_CUES.contains(pw) ||
               pt.casing == rv::MIXED || (MODEL_PREFIX.contains(doc.low(i - 1)) && pt.casing != rv::LOWER);
    if (!cue) return;
    out.tokens(i, i + 2, lp::model_number(a) + " slash " + lp::model_number(b), 75);
}

void model_at(const Doc& doc, rv::Out& out, size_t i, const std::vector<std::string_view>& models) {
    slash_versions_at(doc, out, i);
    if (!lone_number(doc, i)) {
        variant_at(doc, out, i);
        return;
    }
    std::string_view digs = doc.str(i);
    bool plural_s = doc.word(i + 1) && doc.joined(i + 1) && doc.str(i + 1) == "s";
    if (plural_s && digs.size() == 4 && (digs[0] >= '3' || (digs[0] == '2' && digs[1] != '0'))) {
        out.tokens(i, i + 1, plural(lp::model_number(digs)), 55);  // "two 3090s": thirty nineties
        return;
    }
    // "40 737s": models counted by a number before them
    if (plural_s && digs.size() == 3 && i >= 1 && doc.number(i - 1) && doc[i].space_before &&
        doc.str(i - 1).size() <= 3) {
        out.tokens(i, i + 1, plural(lp::model_number(digs)), 65);
        return;
    }
    int score = model_score(doc, i);
    if (!score) {
        bool seen = false;
        for (std::string_view m : models) seen |= m == digs;
        if (!seen || (doc.word(i + 1) && (lp::count_noun(doc.low(i + 1)) || lp::unit_word(doc.str(i + 1)))))
            return;
        size_t p = i - 1;
        // the same number after a determiner or a name ("Office 365 ... Microsoft 365")
        bool det = doc.word(p) && (doc.is(p, "the") || doc.is(p, "a") || doc.is(p, "an") || doc.is(p, "and") ||
                                   doc.is(p, "or") || doc.is(p, "vs") || doc.is(p, "than") || doc.is(p, "older") ||
                                   doc.is(p, "newer") || doc.is(p, "uses") || doc.is(p, "with") ||
                                   (doc[p].casing != rv::LOWER && !lp::not_model_cue(doc.low(p))));
        if (!det) return;
        score = 55;
    }
    std::string spoken = lp::model_number(digs);
    size_t end = i;
    if (plural_s) {
        spoken = plural(spoken);
        end = i + 1;
    }
    out.tokens(i, end, spoken, score);
}

// ---------------------------------------------------------------------------------------------------------
// Model sizes and scales after a number: "7B model" -> seven B, "2.9B parameters" -> two point nine billion,
// "1.5M tokens", "12k users", "4K video" -> four K, "10-15k" -> ten to fifteen thousand.

const rv::WordSet CURRENCY_CODES{"AED", "AUD", "BRL", "CAD", "CHF", "CNY", "DKK", "EUR", "GBP", "HKD", "INR",
                                 "JPY", "KRW", "MXN", "NOK", "NZD", "RMB", "SEK", "SGD", "USD", "ZAR"};

// "1.5B, 7B and 14B parameters": sizes in a list share the counted noun after the last one.
bool coordinated_count(const Doc& doc, size_t k) {
    std::string_view letter = doc.str(k);
    size_t p = k + 1;
    bool linked = false;
    for (int step = 0; step < 8 && p < doc.size(); ++step) {
        if (doc.punct(p, ',') || doc.is(p, "and") || doc.is(p, "or")) {
            ++p;
            linked = true;
            continue;
        }
        if (!linked || !doc.number(p)) return false;
        rv::NumberSpan n = rv::parse_number(doc, p);
        size_t q = n.last + 1;
        if (!doc.word(q) || !doc.joined(q) || doc.str(q) != letter) return false;
        p = q + 1;
        linked = false;
        if (doc.word(p) && doc[p].space_before && (doc.punct(p - 1, ',') == false)) {
            std::string_view w = doc.low(p);
            if (lp::count_noun(w)) return true;
        }
    }
    return false;
}

// "3M" at token i (k: the M) is the company: a possessive, a company verb after it, "the 3M plant", or the
// start of a sentence when it counts nothing.
bool company_3m(const Doc& doc, size_t i, size_t k, const std::string& next, bool count) {
    static const rv::WordSet COMPANY_VERBS{"announced", "brand", "cut", "factory", "has", "headquarters", "hired",
                                           "is", "launched", "launches", "logo", "makes", "plans", "plant",
                                           "ranks", "released", "reported", "said", "says", "sells", "sends",
                                           "sent", "settled", "spokesperson", "stock", "sued", "unveiled", "was"};
    return (doc.joined(k + 1) && (doc.punct(k + 1, '\'') || doc.punct(k + 1, 0x2019))) ||
           COMPANY_VERBS.contains(next) ||
           (i > 0 && doc.is(doc.prev_word(i) == Doc::npos ? i : doc.prev_word(i), "the") && !count) ||
           (doc.sentence_start(i) && !count);
}

size_t scale_at(const Doc& doc, rv::Out& out, size_t i, bool seen_3m) {
    {
        if (!doc.number(i)) return i;
        if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return i;
        if (doc.joined(i) && (doc.punct(i - 1, '$') || doc.punct(i - 1, 0x20AC) || doc.punct(i - 1, 0xA3) ||
                              doc.punct(i - 1, 0xA5) || doc.punct(i - 1, '.') || doc.punct(i - 1, '-') ||
                              doc.punct(i - 1, 0x2013) || doc.punct(i - 1, '/')))
            return i;
        rv::NumberSpan a = rv::parse_number(doc, i);
        size_t k = a.last + 1;
        // a range: "10-15k"
        rv::NumberSpan b;
        if ((doc.punct(k, '-') || doc.punct(k, 0x2013)) && doc.joined(k) && doc.number(k + 1) && doc.joined(k + 1)) {
            b = rv::parse_number(doc, k + 1);
            k = b.last + 1;
        }
        // (a spaced "2 M" is a scale of numbers_core)
        if (!doc.word(k) || !doc.joined(k)) return a.last;
        std::string_view s = doc.str(k);
        if (s != "k" && s != "K" && s != "M" && s != "B" && s != "T") return i;
        {
            // an amount with a currency code belongs to the money rules: "EUR 150k", "150k USD"
            size_t pw = doc.prev_word(i), nw = doc.next_word(k);
            if ((pw != Doc::npos && pw + 1 == i && CURRENCY_CODES.contains(doc.str(pw))) ||
                (nw != Doc::npos && nw == k + 1 && CURRENCY_CODES.contains(doc.str(nw))))
                return k;
        }
        if (!a.grouped && a.fraction.empty() && a.whole.size() >= 4 && !b.ok) return i;  // "13900K": a model
        // "Core Ultra 9 285K": a processor model after its tier number
        if (s == "K" && !b.ok && a.fraction.empty() && !a.grouped && a.whole.size() == 3 && i >= 2 &&
            doc[i].space_before && doc.number(i - 1) && doc.str(i - 1).size() == 1 && doc.word(i - 2) &&
            (MODEL_PREFIX.contains(doc.low(i - 2)) || doc.is(i - 2, "ultra")) &&
            !(alnum_tok(doc, k + 1) && doc.joined(k + 1))) {
            out.tokens(i, k, lp::model_number(a.whole) + " K", 62);
            return k;
        }
        if (alnum_tok(doc, k + 1) && doc.joined(k + 1)) return i;
        if (doc.joined(k + 1) && (doc.punct(k + 1, '/') || doc.punct(k + 1, '-')) && alnum_tok(doc, k + 2) &&
            doc.joined(k + 2) && !(doc.punct(k + 1, '-') && doc.word(k + 2) && lower_word(doc.str(k + 2))))
            return i;
        size_t nx = doc.next_word(k);
        std::string next = nx != Doc::npos && nx == k + 1 + (doc.punct(k + 1, '-') ? 1 : 0) ? lp::lower(doc.str(nx))
                           : nx != Doc::npos && doc[nx].space_before && nx == k + 1 ? lp::lower(doc.str(nx))
                                                                                    : std::string();
        static const rv::WordSet NOT_PLURAL{"across", "as", "benchmarks", "evals", "has", "his", "is", "its",
                                            "less", "plus", "results", "scores", "sizes", "this", "thus", "versus",
                                            "was", "yes"};
        static const rv::WordSet COUNTED{"active", "data", "daily", "hourly", "monthly", "parameter", "scanned",
                                         "synthetic", "token", "unique", "weekly"};
        static const rv::WordSet FUNCTION{"across", "and", "at", "by", "for", "from", "in", "of", "on", "or",
                                          "over", "per", "than", "to", "under", "with"};
        bool possessive_next = next.find('\'') != std::string::npos || next.find("\xE2\x80\x99") != std::string::npos;
        bool plural_noun = next.size() >= 4 && next.back() == 's' && !NOT_PLURAL.contains(next) &&
                           !lp::model_noun(next) && !possessive_next;
        bool count = !next.empty() && (lp::count_noun(next) || plural_noun || COUNTED.contains(next));
        if (!count && !next.empty() && !lp::model_noun(next) && !FUNCTION.contains(next) && nx != Doc::npos) {
            // "2.7B charge cycles": the counted noun one word later
            size_t nx2 = doc.next_word(nx);
            std::string_view n2 = nx2 == nx + 1 && doc[nx2].space_before ? doc.low(nx2) : std::string_view();
            if (lp::count_noun(n2) ||
                (!a.fraction.empty() && doc[nx].casing == rv::LOWER && doc[nx2].casing == rv::LOWER &&
                 n2.size() >= 4 && n2.back() == 's' && !NOT_PLURAL.contains(n2) && !lp::model_noun(n2)))
                count = true;
        }
        bool whole = a.fraction.empty() && (!b.ok || b.fraction.empty());
        // a model name right before the size: "Kestrel Edge 3B hosts", "Lumen 0.6B, 42 ms"
        bool name_before = false;
        if (i > 0 && doc.word(i - 1) && doc[i].space_before && !b.ok) {
            const rv::Token& pt = doc[i - 1];
            std::string_view pw = doc.str(i - 1);
            bool possessive = pw.find('\'') != std::string_view::npos || pw.find("\xE2\x80\x99") != std::string_view::npos;
            // a capitalized word mid-sentence is a name; at a sentence start only if it is not an English word
            name_before = !possessive && !CURRENCY_CODES.contains(pw) &&
                          (pt.casing == rv::MIXED || (pt.casing == rv::UPPER && pw.size() >= 2) ||
                           (pt.casing == rv::TITLE && (!doc.sentence_start(i - 1) || !lp::english(doc.low(i - 1))) &&
                            !lp::not_model_cue(doc.low(i - 1))));
        }
        std::string word;
        if (s == "k") {
            word = "thousand";
        } else if (s == "K") {
            // a screen or sensor resolution: "a 4K monitor", "a 6.5K sensor", "the 4K documentary channel"
            static const rv::WordSet RES_NEXT{
                "cable", "cables", "camera", "cameras", "caption", "captions", "channel", "clip", "clips", "content",
                "display", "displays", "documentary", "export", "exports", "feed", "feeds", "film", "footage", "frame",
                "frames", "hdr", "imaging", "master", "microscope", "monitor", "monitors", "panel", "panels",
                "playback", "projector", "projectors", "recording", "recordings", "render", "renders", "resolution",
                "screen", "screens", "sensor", "sensors", "source", "sources", "stream", "streaming", "test", "texture",
                "textures", "trailer", "trailers", "tv", "video", "videos"};
            bool res = (whole && !b.ok && (a.whole == "2" || a.whole == "4" || a.whole == "5" || a.whole == "8")) ||
                       (!b.ok && RES_NEXT.contains(next));
            bool name = nx != Doc::npos && nx == k + 1 && doc[nx].casing == rv::TITLE;  // "2K Games"
            // "a 2K-vial lot": a hyphenated count; "Context 4K": a context length in tokens
            bool compound = doc.punct(k + 1, '-') && doc.joined(k + 1) && !RES_NEXT.contains(next) && !next.empty();
            size_t pk = doc.prev_word(i);
            if (pk != Doc::npos && pk + 2 >= i && (doc.is(pk, "context") || doc.is(pk, "window"))) compound = true;
            word = res && !compound && (!count || name || RES_NEXT.contains(next)) ? "K" : "thousand";
        } else if (s == "M") {
            word = "million";
            bool company = !b.ok && a.whole == "3" && whole &&
                           (company_3m(doc, i, k, next, count) || (seen_3m && (!count || next == "shares")));
            if (company) word = "M";
        } else if (s == "B") {
            // a measure or currency after the number counts: "1.5B euro fraud", "2.1B tonne baseline"
            static const rv::WordSet MEASURES{"barrel", "barrels", "dollar", "dollars", "euro", "euros", "pound",
                                              "pounds", "ton", "tonne", "tonnes", "tons", "yen", "yuan"};
            // a decimal is a model size only before a model word ("a 2.7B model", "the 0.8B edge model") or
            // after a name; before other words it counts something ("2.3B event feed")
            bool model = !next.empty() && lp::model_noun(next);
            if (!model && nx != Doc::npos && doc.word(nx) && doc[nx].casing == rv::LOWER) {
                size_t nx2 = doc.next_word(nx);
                model = nx2 == nx + 1 && doc[nx2].space_before && lp::model_noun(doc.low(nx2));
            }
            // a label after a reference word: "Apartment 3B", "Box 7B", "Ref. 14B"
            static const rv::WordSet LABELS{"apartment", "apt", "bay", "bed", "block", "box", "building", "cell",
                                            "dock", "exhibit", "fig", "figure", "flat", "gate", "item", "level",
                                            "locker", "lot", "pier", "platform", "ref", "room", "row", "seat",
                                            "slot", "stall", "suite", "table", "terminal", "unit", "ward"};
            size_t pw = doc.prev_word(i);
            bool label = whole && pw != Doc::npos && pw + 2 >= i && LABELS.contains(doc.low(pw)) &&
                         (pw + 1 == i || doc.punct(pw + 1, '.'));
            // money words just before: "revenue hit 2B", "raised 1.5B"
            static const rv::WordSet MONEY{"budget", "cost", "costs", "debt", "deficit", "earned", "earnings",
                                           "funding", "income", "loss", "losses", "profit", "profits", "raised",
                                           "revenue", "revenues", "sales", "spent", "turnover", "valuation",
                                           "valued", "worth"};
            bool money = false;
            for (size_t k = i, seen = 0; k > 0 && seen < 3 && !money; --k)
                if (doc.word(k - 1)) money = MONEY.contains(doc.low(k - 1)), ++seen;
            // "the 70B weights": the weights of a model; "the 14B needs 24 GB": a model doing something
            static const rv::WordSet MODEL_VERBS{"answers", "beat", "beats", "fits", "handles", "loads", "needed",
                                                 "needs", "ran", "reached", "requires", "runs", "scored", "scores",
                                                 "takes", "took", "uses", "won"};
            bool after_the = pw != Doc::npos && pw + 1 == i && doc.is(pw, "the");
            bool the_weights = after_the && ((whole && next == "weights") || MODEL_VERBS.contains(next));
            // "Size 2.5B": a size label
            if (pw != Doc::npos && (doc.is(pw, "size") || doc.is(pw, "sizes")) && pw + 2 >= i) model = true;
            bool counted = !label && !the_weights &&
                           ((count && (lp::count_noun(next) || !name_before)) || MEASURES.contains(next) ||
                            (money && !model) || (!name_before && coordinated_count(doc, k)));
            word = counted ? "billion" : label || model || name_before || whole ? "B" : "billion";
        } else {
            word = "trillion";
        }
        std::string spoken = rv::read_number(a);
        if (spoken.empty()) return i;
        if (b.ok) {
            std::string hi = rv::read_number(b);
            if (hi.empty()) return i;
            spoken += " to " + hi;
        }
        spoken += " " + word;
        // "10k+ transactions": the plus is said
        if (doc.punct(k + 1, '+') && doc.joined(k + 1) &&
            !(doc.joined(k + 2) && (alnum_tok(doc, k + 2) || doc.punct(k + 2, '+')))) {
            out.tokens(i, k + 1, spoken + " plus", 60);
            return k + 1;
        }
        out.tokens(i, k, spoken, 60);
        return k;
    }
}

// Video resolutions: "1080p" -> ten eighty P, "720p" -> seven twenty P.
void resolution_at(const Doc& doc, rv::Out& out, size_t i) {
    static const rv::WordSet RES{"144", "240", "360", "480", "540", "720", "1080", "1440", "2160", "4320"};
    if (i + 1 >= doc.size()) return;
    if (!doc.number(i) || !doc.joined(i + 1) || doc.str(i + 1) != "p" || !RES.contains(doc.str(i))) return;
    if (doc.joined(i) && doc[i - 1].kind != Kind::Punct) return;
    if (alnum_tok(doc, i + 2) && doc.joined(i + 2)) return;
    out.tokens(i, i + 1, lp::model_number(doc.str(i)) + " P", 60);
}

// ---------------------------------------------------------------------------------------------------------
// Versions: "v2.3" -> version two point three, "1.0.1", "Python 3.12" -> Python three point twelve,
// "2.0.0-rc1" -> two point zero point zero RC one, "v1.0-beta".

// A pre-release tag after a version: "-rc1", "-beta", "-alpha2". Returns the words and moves end.
std::string prerelease(const Doc& d, size_t& end) {
    if (!d.punct(end + 1, '-') || !d.joined(end + 1) || !d.word(end + 2) || !d.joined(end + 2)) return "";
    std::string_view t = d.str(end + 2);
    std::string tl = lp::lower(t);
    std::string words;
    if (tl == "rc") words = "RC";
    else if (tl == "beta" || tl == "alpha" || tl == "preview" || tl == "dev" || tl == "final" || tl == "pre")
        words = tl;
    else return "";
    size_t e = end + 2;
    if (d.number(e + 1) && d.joined(e + 1)) {
        words += " " + lp::version_part(d.str(e + 1));
        ++e;
    } else if (d.punct(e + 1, '.') && d.joined(e + 1) && d.number(e + 2) && d.joined(e + 2)) {
        words += " " + lp::version_part(d.str(e + 2));  // "-beta.1"
        e += 2;
    }
    if (alnum_tok(d, e + 1) && d.joined(e + 1)) return "";
    end = e;
    return words;
}

// Token k starts its line, with only punctuation before it ("> v3.4.1", "- v2.0").
bool line_start(const Doc& d, size_t k) {
    while (k > 0 && !d[k].newline_before) {
        if (d[k - 1].kind != Kind::Punct) return false;
        --k;
    }
    return true;
}

// Operators between a cue word and its version: "Python >= 3.10", "python==3.12".
bool version_operator(const Doc& d, size_t k) {
    if (d[k].kind != Kind::Punct) return false;
    uint32_t c = d[k].cp;
    return c == '>' || c == '<' || c == '=' || c == '~' || c == '^' || c == '!' || c == 0x2265 || c == 0x2264;
}


// Tokens between two versions that make the second one a version too: "3.9–3.12", "3.10 and 3.11", "3.11/3.12".
bool version_link(const Doc& d, size_t a, size_t b) {
    if (b == a + 2) {
        const rv::Token& t = d[a + 1];
        if (t.kind == Kind::Punct && (t.cp == '-' || t.cp == 0x2013 || t.cp == '/'))
            return d.joined(a + 1) == d.joined(b) || t.cp == 0x2013;
        if (t.kind == Kind::Punct && t.cp == ',') return true;
        return d.is(a + 1, "to") || d.is(a + 1, "or") || d.is(a + 1, "and") || d.is(a + 1, "through");
    }
    return b == a + 3 && d.punct(a + 1, ',') && (d.is(a + 2, "and") || d.is(a + 2, "or"));
}

// The word before the number at token i cues a version: "Python 3.12", "Go 1.22", "macOS 14.2", "Python >= 3.10".
bool version_cue_before(const Doc& doc, size_t i) {
    size_t p = doc.prev_word(i);
    if (p == Doc::npos || !doc.word(p)) return false;
    // operators between the cue word and the number: "Python >= 3.10"
    size_t gap = 1;
    if (p + 1 < i) {
        bool ops = p + 4 >= i;
        for (size_t k = p + 1; k < i && ops; ++k) ops = version_operator(doc, k);
        if (ops) gap = i - p;
    }
    if (p + gap != i) return false;
    std::string pl = lp::lower(doc.str(p));
    const rv::Token& pt = doc[p];
    if (lp::version_cue(pl) || CAP_VERSION_CUES.contains(doc.str(p)) ||
        (!lp::not_model_cue(pl) &&
         (pt.casing == rv::MIXED || (pt.casing == rv::TITLE && !lp::english(pl) && pl.size() > 2))))
        return true;
    // "upgraded from 4.1", "pinned to 2.11", "stays on 1.22", "Upgrade the toolchain to 5.22"
    static const rv::WordSet PREPS{"from", "on", "to"};
    static const rv::WordSet VERBS{"bump", "bumped", "deployed", "downgrade", "downgraded", "migrate", "migrated",
                                   "pin", "pinned", "pins", "revert", "reverted", "rolled", "shipped", "upgrade",
                                   "upgraded", "upgrades", "upgrading"};
    // verbs that cue a version only with "on": "stays on 1.22", "fails on 5.8"
    static const rv::WordSet ON_VERBS{"fail", "failed", "fails", "remain", "remains", "run", "running", "runs",
                                      "stay", "stayed", "stays", "work", "worked", "works"};
    if (gap != 1 || !PREPS.contains(pl)) return false;
    if (pl == "on" && p > 0 && doc.word(p - 1) && ON_VERBS.contains(doc.low(p - 1))) return true;
    for (size_t k = p, seen = 0; k > 0 && seen < 6; --k) {
        const rv::Token& t = doc[k - 1];
        if (t.kind == Kind::Punct && (t.cp == '!' || t.cp == '?' || t.cp == ';' || (t.cp == '.' && !doc.joined(k))))
            return false;
        if (t.kind != Kind::Word) continue;
        if (VERBS.contains(doc.low(k - 1))) return true;
        ++seen;
    }
    return false;
}

// The word after a dotted number cues a version: "the 1.13 branch", "the 7.21 release".
bool version_cue_after(const Doc& doc, size_t e) {
    static const rv::WordSet AFTER{"branch", "branches", "build", "builds", "driver", "drivers", "firmware",
                                   "hotfix", "kernel", "lts", "patch", "release", "releases", "update", "upgrade"};
    return doc.word(e + 1) && doc[e + 1].space_before && AFTER.contains(doc.low(e + 1));
}

size_t version_at(const Doc& doc, rv::Out& out, size_t i, size_t& last_version,
                  const std::vector<std::string_view>& known_versions) {
    {
        if (!doc.number(i)) return i;
        bool chained = last_version != Doc::npos && last_version < i && version_link(doc, last_version, i);
        bool v_prefix = i > 0 && doc.joined(i) && doc.word(i - 1) && (doc.str(i - 1) == "v") &&
                        !(i > 1 && doc.joined(i - 1) && doc[i - 2].kind != Kind::Punct);
        // "v.3.1.4": the letter, a period, the version
        bool v_dot = !v_prefix && i >= 2 && doc.punct(i - 1, '.') && doc.joined(i) && doc.joined(i - 1) &&
                     doc.word(i - 2) && doc.str(i - 2) == "v" && !(i > 2 && doc.joined(i - 2) && alnum_tok(doc, i - 3));
        size_t vstart = v_dot ? i - 2 : i - 1;
        v_prefix = v_prefix || v_dot;
        // "large-v3": the letter V inside a name, read with the name
        if (v_prefix && i >= 3 && doc.joined(i - 1) && doc.punct(i - 2, '-') && doc.joined(i - 2) && doc.word(i - 3))
            return i;
        if (doc.joined(i) && !v_prefix && doc[i - 1].kind != Kind::Punct) return i;
        if (doc.joined(i) && !v_prefix && !chained &&
            (doc.punct(i - 1, '.') || doc.punct(i - 1, '/') || doc.punct(i - 1, '-') || doc.punct(i - 1, '$') ||
             doc.punct(i - 1, ':') || doc.punct(i - 1, '#') || doc.punct(i - 1, '@')))
            return i;
        if (v_prefix && i > 1 && doc.joined(i - 1) &&
            (doc.punct(i - 2, '/') || doc.punct(i - 2, '.') || doc.punct(i - 2, '@') || doc.punct(i - 2, '#')))
            return i;  // "/v2/": a path
        // dotted parts
        size_t e = i;
        int parts = 1;
        while (doc.punct(e + 1, '.') && doc.joined(e + 1) && doc.number(e + 2) && doc.joined(e + 2)) {
            e += 2;
            ++parts;
        }
        if (alnum_tok(doc, e + 1) && doc.joined(e + 1)) {
            return e;
        }
        // "1.x", "v2.x": a version with a wildcard part
        bool wildcard = doc.punct(e + 1, '.') && doc.joined(e + 1) && doc.is(e + 2, "x") && doc.joined(e + 2) &&
                        !(doc.joined(e + 3) && (alnum_tok(doc, e + 3) || (doc.punct(e + 3, '.') && alnum_tok(doc, e + 4))));
        if (doc.joined(e + 1) && !wildcard &&
            (doc.punct(e + 1, '.') || doc.punct(e + 1, '/') || doc.punct(e + 1, '%') || doc.punct(e + 1, ':')) &&
            alnum_tok(doc, e + 2) && doc.joined(e + 2) && !(chained && doc.punct(e + 1, '/'))) {
            return e;
        }
        if (doc.joined(e + 1) && (doc.punct(e + 1, '%') || doc.punct(e + 1, 0xB0))) {
            return e;
        }
        std::string written(doc.slice(doc[i].start, doc[e].end));
        size_t p = doc.prev_word(i);
        // the same dotted number as a version elsewhere in the text ("Python 3.12 ... the 3.12 release")
        bool known = false;
        static const rv::WordSet QUANTITY{"days", "hours", "hrs", "meters", "miles", "min", "mins", "minutes", "ms",
                                          "percent", "points", "s", "sec", "seconds", "secs", "times", "x"};
        bool quantity = doc.word(e + 1) && (lp::unit_word(doc.str(e + 1)) || lp::count_noun(doc.low(e + 1)) ||
                                            QUANTITY.contains(doc.low(e + 1)));
        if (parts == 2 && !v_prefix && !quantity)
            for (std::string_view k : known_versions) known |= k == std::string_view(written);
        bool cue = chained || known || (!v_prefix && !quantity && (version_cue_before(doc, i) ||
                                                                     version_cue_after(doc, e)));
        if (wildcard) {
            std::string spoken = lp::version_words(written);
            if (spoken.empty()) return e;
            spoken += " point X";
            size_t first = i;
            if (v_prefix) spoken = "version " + spoken, first = vstart;
            out.tokens(first, e + 2, spoken, 70, v_prefix && line_start(doc, vstart));
            last_version = e + 2;
            return e + 2;
        }
        if (!cue && p != Doc::npos && !v_prefix && p + 2 == i && doc.punct(p + 1, '.') && doc.word(p) &&
            (doc.is(p, "js") || doc.is(p, "net")))
            cue = true;  // "Node.js 20.11"
        static const rv::WordSet VERBS{"pinned", "pins", "require", "required", "requires", "running", "runs",
                                       "support", "supports", "targets"};
        static const rv::WordSet LATER{"above", "earlier", "higher", "later", "newer", "older"};
        if (!cue && !v_prefix && p != Doc::npos && p + 1 == i && doc.word(p) && VERBS.contains(doc.low(p)))
            cue = true;
        if (!cue && !v_prefix && doc.is(e + 1, "or") && doc.word(e + 2) && LATER.contains(doc.low(e + 2)))
            cue = true;
        // "Llama 3.1 8B": the version, a comma, the size
        bool size_next = parts <= 2 && doc.number(e + 1) && doc[e + 1].space_before && doc.word(e + 2) &&
                         doc.joined(e + 2) && (doc.str(e + 2) == "B" || doc.str(e + 2) == "b") &&
                         !(alnum_tok(doc, e + 3) && doc.joined(e + 3)) && p != Doc::npos && p + 1 == i &&
                         doc.word(p) && (doc[p].casing != rv::LOWER || MODEL_NAMES.contains(doc.str(p)));
        if (size_next) cue = true;
        if (size_next && doc[p].casing == rv::LOWER) out.tokens(p, p, rv::capital(std::string(doc.str(p))), 50);
        if (parts == 1 && !v_prefix && size_next && doc.str(i).size() <= 2) {
            out.tokens(i, i, rv::cardinal_digits(doc.str(i)) + ",", 65);  // "Gemma 2 9B": two, nine B
            return i;
        }
        if (parts == 1 && !v_prefix) return i;
        bool four_part_ip = parts == 4;
        if (four_part_ip && !(v_prefix || cue)) {
            return e;
        }
        if (!v_prefix && !cue && parts < 3) {
            return e;
        }
        if (parts >= 3) {
            // a date such as 1.2.2026 is not a version
            std::string_view lastp = doc.str(e);
            if (lastp.size() == 4 && !v_prefix && !cue) {
                return e;
            }
        }
        std::string spoken = lp::version_words(written);
        if (spoken.empty()) {
            return e;
        }
        if (size_next) spoken += ",";
        size_t end = e;
        std::string pre = prerelease(doc, end);
        if (!pre.empty()) spoken += " " + pre;
        size_t first = i;
        bool keep = false;
        if (v_prefix) {
            // "version v2.1" says the word once; a lower-case "v" stays lower case at a line start
            size_t pv = doc.prev_word(i - 1);
            if (!(pv != Doc::npos && pv + 2 == i && (doc.is(pv, "version") || doc.is(pv, "ver"))))
                spoken = "version " + spoken;
            first = vstart;
            keep = line_start(doc, vstart);
            // "v2.0–v2.3": a range of versions
            if (i >= 3 && last_version == i - 3 && doc.joined(i - 1) && doc.joined(i - 2) &&
                (doc.punct(i - 2, 0x2013) || doc.punct(i - 2, '-'))) {
                spoken = "to " + spoken;
                first = i - 2;
                keep = false;
            }
        }
        out.tokens(first, end, spoken, v_prefix || cue ? 70 : 55, keep);
        last_version = end;
        return end;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Standards and names read in a fixed way: "ISO 9001", "IEEE 802.11ax", "H.264", "HTTP/2", "C++", "C#",
// ".NET", "Node.js", "401(k)", "top-k", "7-Eleven", "iPhone X".


// Numbers of standards: "ISO 9001", "802.11ax", "401(k)", "7-Eleven". Returns the last token used.
// Words before a standard number with a part or year after it: "FIPS 140-2", "NIST SP 800-53", "ISO 9001:2015".
const rv::WordSet STANDARD_CUES{"ANSI", "AS", "ASME", "BS", "CSA", "DIN", "EN", "FIPS", "IEC", "IEEE", "ISO", "MIL", "NIST",
                                "NFPA", "SP", "STD", "UL"};

// The part after a standard number: "-2" -> two, ":2015" / "-2008" -> twenty fifteen / two thousand eight.
std::string standard_part(std::string_view d) {
    if (d.size() == 4 && (d.substr(0, 2) == "19" || d.substr(0, 2) == "20"))
        return rv::pairs(static_cast<uint32_t>(std::strtoul(std::string(d).c_str(), nullptr, 10)));
    return d.size() <= 4 ? lp::model_number(d) : "";
}

size_t standard_number_at(const Doc& doc, rv::Out& out, size_t i) {
    const rv::Token& t = doc[i];
    // a part or year after a standard number: "FIPS 140-2" -> FIPS one forty, two; "ISO 9001:2015"
    if (i >= 1 && t.space_before && doc.word(i - 1) && STANDARD_CUES.contains(doc.str(i - 1)) &&
        doc.str(i).size() >= 2 && doc.str(i).size() <= 5 && doc.str(i)[0] != '0' &&
        (doc.punct(i + 1, '-') || doc.punct(i + 1, ':')) && doc.joined(i + 1) && doc.number(i + 2) &&
        doc.joined(i + 2) && !(doc.joined(i + 3) && (alnum_tok(doc, i + 3) || doc.punct(i + 3, '-') ||
                                                      doc.punct(i + 3, ':') ||
                                                      (doc.punct(i + 3, '.') && doc.number(i + 4))))) {
        bool iso = doc.str(i - 1) == "ISO" || doc.str(i - 1) == "IEC";
        std::string part = standard_part(doc.str(i + 2));
        bool year = doc.str(i + 2).size() == 4;
        if (!part.empty() && (doc.punct(i + 1, '-') || year)) {
            std::string head = iso ? lp::standard_number(doc.str(i)) : lp::model_number(doc.str(i));
            out.tokens(i, i + 2, head + ", " + part, 70);
            return i + 2;
        }
    }
    // ISO 9001, ISO/IEC 27001
    if (i >= 1 && t.space_before && doc.word(i - 1) && (doc.str(i - 1) == "ISO" || doc.str(i - 1) == "IEC") &&
        doc.str(i).size() >= 3 && doc.str(i).size() <= 5 && !(alnum_tok(doc, i + 1) && doc.joined(i + 1)) &&
        !(doc.punct(i + 1, '.') && doc.number(i + 2) && doc.joined(i + 2))) {
        out.tokens(i, i, lp::standard_number(doc.str(i)), 70);
        return i;
    }
    // 802.11ax -> eight oh two dot eleven AX
    if (doc.str(i) == "802" && doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.number(i + 2) &&
        doc.joined(i + 2) && !(doc.joined(i) && alnum_tok(doc, i - 1))) {
        std::string spoken = "eight oh two dot " + lp::model_number(doc.str(i + 2));
        size_t end = i + 2;
        if (doc.word(end + 1) && doc.joined(end + 1) && lower_word(doc.str(end + 1)) && doc.str(end + 1).size() <= 3) {
            spoken += " " + lp::upper(doc.str(end + 1));
            ++end;
        }
        if (alnum_tok(doc, end + 1) && doc.joined(end + 1)) return i;
        out.tokens(i, end, spoken, 75);
        return end;
    }
    // 401(k), 403(b), 457(b)
    bool plan = doc.str(i) == "401" || doc.str(i) == "403" || doc.str(i) == "457";
    if (plan && doc.punct(i + 1, '(') && doc.joined(i + 1) && doc.word(i + 2) && doc.str(i + 2).size() == 1 &&
        doc.joined(i + 2) && doc.punct(i + 3, ')') && doc.joined(i + 3)) {
        out.tokens(i, i + 3, lp::model_number(doc.str(i)) + " " + lp::upper(doc.str(i + 2)), 80);
        return i + 3;
    }
    if (doc.str(i) == "7" && doc.punct(i + 1, '-') && doc.joined(i + 1) && doc.word(i + 2) &&
        doc.str(i + 2) == "Eleven" && doc.joined(i + 2)) {
        out.tokens(i, i + 2, "Seven Eleven", 80);
        return i + 2;
    }
    return i;
}

// Names read in a fixed way: "H.264", "HTTP/2", "C++", "C#", ".NET", "Node.js", "top-k", "iPhone X".
// Returns the last token used.
size_t standard_word_at(const Doc& doc, rv::Out& out, size_t i) {
    const rv::Token& t = doc[i];
    if (t.kind == Kind::Punct) {
        // .NET
        if (t.cp == '.' && doc.word(i + 1) && doc.str(i + 1) == "NET" && doc.joined(i + 1) &&
            !(doc.joined(i) && alnum_tok(doc, i - 1)) && !(alnum_tok(doc, i + 2) && doc.joined(i + 2))) {
            out.tokens(i, i + 1, "dot net", 80);
            return i + 1;
        }
        return i;
    }
    if (t.kind != Kind::Word) return i;
    std::string_view w = doc.str(i);
    bool joined_punct = doc.joined(i + 1) && doc[i + 1].kind == Kind::Punct;
    if (!joined_punct && w != "iPhone" && w != "OS") return i;
    bool free_before = !(doc.joined(i) && (alnum_tok(doc, i - 1) || doc.punct(i - 1, '.')));
    // H.264, G.711: a capital, a period and three digits
    if (w.size() == 1 && w[0] >= 'A' && w[0] <= 'Z' && free_before && doc.punct(i + 1, '.') && doc.joined(i + 1) &&
        doc.number(i + 2) && doc.joined(i + 2) && doc.str(i + 2).size() == 3 &&
        !(alnum_tok(doc, i + 3) && doc.joined(i + 3)) && !(doc.punct(i + 3, '.') && doc.number(i + 4) && doc.joined(i + 4))) {
        out.tokens(i, i + 2, std::string(w) + " " + lp::model_number(doc.str(i + 2)), 70);
        return i + 2;
    }
    // HTTP/2, HTTP/1.1
    if ((w == "HTTP" || w == "http") && free_before && doc.punct(i + 1, '/') && doc.joined(i + 1) &&
        doc.number(i + 2) && doc.joined(i + 2) && doc.str(i + 2).size() == 1) {
        size_t end = i + 2;
        std::string ver(doc.str(end));
        if (doc.punct(end + 1, '.') && doc.joined(end + 1) && doc.number(end + 2) && doc.joined(end + 2)) {
            ver += "." + std::string(doc.str(end + 2));
            end += 2;
        }
        if (alnum_tok(doc, end + 1) && doc.joined(end + 1)) return i;
        out.tokens(i, end, "HTTP " + lp::version_words(ver), 70);
        return end;
    }
    // C++, C#, F#, Q#
    if ((w == "C" || w == "F" || w == "Q") && free_before) {
        if (w == "C" && doc.punct(i + 1, '+') && doc.punct(i + 2, '+') && doc.joined(i + 2) &&
            !(doc.word(i + 3) && doc.joined(i + 3))) {
            // "C++17", "C++20": the standard year after the name
            if (doc.number(i + 3) && doc.joined(i + 3) && doc.str(i + 3).size() == 2 &&
                !(alnum_tok(doc, i + 4) && doc.joined(i + 4))) {
                out.tokens(i, i + 3, "C plus plus " + rv::cardinal_digits(doc.str(i + 3)), 80);
                return i + 3;
            }
            if (!(alnum_tok(doc, i + 3) && doc.joined(i + 3))) {
                out.tokens(i, i + 2, "C plus plus", 80);
                return i + 2;
            }
        }
        if (doc.punct(i + 1, '#') && !(alnum_tok(doc, i + 2) && doc.joined(i + 2))) {
            out.tokens(i, i + 1, std::string(w) + " sharp", 80);
            return i + 1;
        }
    }
    // Node.js, Vue.js, ASP.NET
    if (free_before && doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.word(i + 2) && doc.joined(i + 2) &&
        !(alnum_tok(doc, i + 3) && doc.joined(i + 3)) && !(doc.punct(i + 3, '.') && alnum_tok(doc, i + 4) && doc.joined(i + 4))) {
        std::string_view ext = doc.str(i + 2);
        if (ext == "js" && w[0] >= 'A' && w[0] <= 'Z') {
            out.tokens(i, i + 2, lp::name_word(w) + " JS", 80);
            return i + 2;
        }
        if (ext == "NET" && lp::is_upper_word(w) && w.size() <= 3) {
            out.tokens(i, i + 2, std::string(w) + " dot net", 80);
            return i + 2;
        }
    }
    // top-k, top-p
    if (doc.is(i, "top") && free_before && doc.punct(i + 1, '-') && doc.joined(i + 1) && doc.word(i + 2) &&
        doc.joined(i + 2) && doc.str(i + 2).size() == 1 && !(alnum_tok(doc, i + 3) && doc.joined(i + 3))) {
        out.tokens(i, i + 2, std::string(w) + " " + lp::upper(doc.str(i + 2)), 60);
        return i + 2;
    }
    // iPhone X -> Iphone Ten, Mac OS X -> Mac OS Ten
    if ((w == "iPhone" || w == "OS") && doc.word(i + 1) && doc.str(i + 1) == "X" && doc[i + 1].space_before &&
        !(alnum_tok(doc, i + 2) && doc.joined(i + 2)) && !(doc.joined(i + 2) && doc.punct(i + 2, '-'))) {
        out.tokens(i, i + 1, w == "OS" ? "OS Ten" : "Iphone Ten", 70);
        return i + 1;
    }
    return i;
}

// ---------------------------------------------------------------------------------------------------------
// The rules: one pass over the tokens each.

// Codes of letters and numbers ("Qwen3-1.7B", "A1B2C3"), "B2B", resolutions ("1080p"). Numeronyms ("k8s") are
// left to abbreviations.
RV_RULE("letters_products", codes) {
    size_t next = 0;
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc[i].kind == Kind::Punct || (!doc.joined(i + 1) && !doc.joined(i))) continue;
        if (i >= next) next = code_at(doc, out, i) + 1;
        two_shorthand_at(doc, out, i);
        resolution_at(doc, out, i);
        ers_at(doc, out, i);
        network_at(doc, out, i);
        lineage_at(doc, out, i);
    }
}

// Numbers: model numbers after a name ("RTX 4090"), scales ("7B", "1.5M"), versions ("v2.3", "Python 3.12"),
// numbers of standards ("ISO 9001", "802.11ax").
RV_RULE("letters_products", numbers) {
    const size_t n = doc.size();
    // numbers read as model numbers anywhere in the text: the same number elsewhere is read the same way
    std::vector<std::string_view> models;
    for (size_t i = 1; i < n; ++i)
        if (doc.number(i) && lone_number(doc, i) && model_score(doc, i) >= 60) models.push_back(doc.str(i));
    size_t next_scale = 0, next_version = 0, next_standard = 0, last_version = Doc::npos;
    // two-part versions after a cue word, read the same way wherever they come back
    std::vector<std::string_view> known_versions;
    for (size_t i = 0; i + 2 < n; ++i)
        if (doc.number(i) && doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.number(i + 2) && doc.joined(i + 2) &&
            !(i > 0 && doc.joined(i)) && !(doc.joined(i + 3) && doc[i + 3].kind != Kind::Punct) &&
            !(doc.punct(i + 3, '.') && doc.number(i + 4) && doc.joined(i + 4)) &&
            (version_cue_before(doc, i) || version_cue_after(doc, i + 2)))
            known_versions.push_back(doc.slice(doc[i].start, doc[i + 2].end));
    // "3M" read as the company once is the company everywhere in the text
    bool seen_3m = false;
    for (size_t i = 0; i + 1 < n && !seen_3m; ++i) {
        if (doc.str(i) == "3" && doc.joined(i + 1) && (doc.str(i + 1) == "M's" || doc.str(i + 1) == "M\xE2\x80\x99s") &&
            !(i > 0 && doc.joined(i)))
            seen_3m = true;  // "3M's home page"
        if (doc.str(i) == "3" && doc.str(i + 1) == "M" && doc.joined(i + 1) && !(i > 0 && doc.joined(i)) &&
            !(i + 2 < n && doc.joined(i + 2) && doc[i + 2].kind != Kind::Punct)) {
            size_t nx = doc.next_word(i + 1);
            std::string next = nx != Doc::npos && nx == i + 2 ? lp::lower(doc.str(nx)) : std::string();
            bool count = !next.empty() && (lp::count_noun(next) || (next.size() >= 4 && next.back() == 's' &&
                                                                    next.find('\'') == std::string::npos));
            seen_3m = company_3m(doc, i, i + 1, next, count);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (!doc.number(i)) continue;
        model_at(doc, out, i, models);
        if (i >= next_scale) next_scale = scale_at(doc, out, i, seen_3m) + 1;
        if (i >= next_version) next_version = version_at(doc, out, i, last_version, known_versions) + 1;
        if (i >= next_standard) next_standard = standard_number_at(doc, out, i) + 1;
    }
}

// Names read in a fixed way: "H.264", "HTTP/2", "C++", ".NET", "Node.js", "top-k".
RV_RULE("letters_products", names) {
    size_t next = 0;
    for (size_t i = 0; i < doc.size(); ++i)
        if (i >= next && (doc.joined(i + 1) || doc.str(i) == "iPhone" || doc.str(i) == "OS"))
            next = standard_word_at(doc, out, i) + 1;
}

}  // namespace
