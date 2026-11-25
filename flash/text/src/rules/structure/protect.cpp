// Text that stays as written: input of punctuation only, and text in another language (whole texts, sentences
// and quotations), digits included.
#include "rules/structure/common.h"

namespace {

using namespace st;

bool plain_punct(uint32_t cp) {
    return (cp < 0x80) || cp == 0x2013 || cp == 0x2014 || cp == 0x2026 || cp == 0x2018 || cp == 0x2019 ||
           cp == 0x201C || cp == 0x201D || cp == 0xAB || cp == 0xBB || cp == 0xB7 || cp == 0x2022;
}

// An input without words or numbers, made of ordinary punctuation, is copied as it is.
RV_RULE("structure", punctuation_only) {
    if (doc.size() == 0) return;
    for (size_t i = 0; i < doc.size(); ++i)
        if (!is_punct(doc, i) || !(plain_punct(doc[i].cp) || invisible_cp(doc[i].cp))) return;
    out.keep(0, doc.size() - 1, 99);
}

// Letters of scripts that are not English: CJK, kana, Hangul, Cyrillic, Greek, Arabic, Hebrew, Thai,
// Devanagari.
bool foreign_script(uint32_t cp) {
    if (cp < 0x370) return false;
    return (cp >= 0x370 && cp <= 0x3FF) || (cp >= 0x400 && cp <= 0x52F) || (cp >= 0x590 && cp <= 0x6FF) ||
           (cp >= 0x900 && cp <= 0x97F) || (cp >= 0xE00 && cp <= 0xE7F) || (cp >= 0x3040 && cp <= 0x30FF) ||
           (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x3400 && cp <= 0x4DBF);
}

const char* const ENGLISH[] = {"the", "and", "of", "to", "are", "were", "be", "been", "it", "that",
                          "this", "with", "for", "on", "by", "from", "at", "as", "has", "have", "had", "not",
                          "but", "which", "will", "we", "they", "he", "she", "you", "its", "their", "our",
                          "or", "after", "before", "said", "says", "than"};
// Function words of German, French, Spanish, Italian, Dutch, Portuguese and Polish that English does not use.
const char* const FOREIGN[] = {
    "der", "das", "und", "ist", "nicht", "mit", "von", "den", "dem", "ein", "eine", "einer", "im", "zu", "auf",
    "f\xC3\xBCr", "wird", "wurde", "sich", "auch", "bei", "nach", "\xC3\xBC" "ber", "aus", "sind", "wir", "ich",
    "sie", "noch", "oder", "aber", "durch", "nur", "le", "la", "les", "des", "du", "et", "est", "une", "dans",
    "pour", "pas", "sur", "au", "aux", "avec", "cette", "qui", "que", "sont", "\xC3\xA9t\xC3\xA9", "nous", "vous",
    "il", "elle", "ils", "ne", "el", "los", "las", "del", "y", "es", "por", "con", "una", "para", "su", "lo",
    "como", "m\xC3\xA1s", "pero", "sus", "fue", "este", "esta", "di", "che", "\xC3\xA8", "e", "sono", "gli",
    "della", "delle", "dei", "nel", "nella", "alla", "ai", "tutti", "stati", "het", "een", "van", "en", "op",
    "te", "dat", "niet", "voor", "zijn", "er", "aan", "ook", "bij", "door", "naar", "nog", "worden", "wordt",
    "os", "da", "dos", "das", "um", "uma", "n\xC3\xA3o", "na", "mais", "foi", "s\xC3\xA3o", "de", "si\xC4\x99",
    "nie", "\xC5\xBC" "e", "jest", "jak", "po", "s\xC4\x85", "od", "za", "ze", "przez", "dla", "oraz", "ou",
    "em", "ao", "pelo", "pela", "uno", "degli", "zich", "werden", "hat", "haben", "kann", "seit", "die",
    "als", "eines", "einem", "wurden", "keine", "kein", "mehr", "sehr", "beim", "vom", "zum", "zur", "tot",
    "wij", "ze", "ont", "avait", "sera", "entre", "sobre", "tiene", "est\xC3\xA1", "hanno", "senza", "anche",
    "molto", "questo", "questa", "ha", "loro", "com", "z", "w", "jest", "oraz", "ist", "sind", "nicht", "poi",
    "siamo", "pi\xC3\xB9", "perch\xC3\xA9", "muy", "tambi\xC3\xA9n", "porque", "hay", "nos", "aussi",
    "tr\xC3\xA8s", "uns", "wenn", "dass", "weil", "cuando", "donde", "sempre", "quando", "dove", "ich"};

// Both lists in one table: 1 English, 2 another language.
const ShortWords FUNCTION_WORDS = [] {
    ShortWords t;
    for (const char* w : ENGLISH) t.add(w, 1);
    for (const char* w : FOREIGN) t.add(w, 2);
    return t;
}();

// Particles of names: "Bernard de Prez", "dos Santos", "van der Waals", "Baron de La Queue".
const rv::WordSet PARTICLES{"da", "das", "de", "del", "della", "den", "der", "des", "di", "do", "dos", "du",
                            "la", "las", "le", "los", "ten", "ter", "van", "von"};

bool name_word(const Doc& d, size_t k) {
    return d.word(k) && (d[k].casing == rv::TITLE || d[k].casing == rv::UPPER) && d[k].cp < 0x370;
}

// Token k is part of a name, not a function word: a particle before a capitalised word, after a capitalised word,
// a particle or a comma ("Gonzalez de Mejia E", "Brovini EM, dos Santos JA"); or initials written in capitals
// next to words that are not ("Brovini EM").
bool in_name(const Doc& d, size_t k) {
    const rv::Token& t = d[k];
    if (t.casing == rv::UPPER && t.end - t.start >= 2) {
        bool shout = (k > 0 && d.word(k - 1) && d[k - 1].casing == rv::UPPER) ||
                     (d.word(k + 1) && d[k + 1].casing == rv::UPPER);
        return !shout;
    }
    if (!PARTICLES.contains(d.low(k)) || !d[k].space_before) return false;
    size_t n = k + 1;
    while (n < d.size() && d.word(n) && PARTICLES.contains(d.low(n)) && d[n].space_before && n < k + 3) ++n;
    if (!name_word(d, n) || !d[n].space_before || d[n].newline_before) return false;
    if (k == 0 || d[k].newline_before) return false;
    bool before = name_word(d, k - 1) || (d.word(k - 1) && PARTICLES.contains(d.low(k - 1))) || d.punct(k - 1, ',') ||
                  d.punct(k - 1, '(');
    return before && (t.casing == rv::LOWER || name_word(d, k - 1) || PARTICLES.contains(d.low(k - 1)));
}

struct Seg {
    size_t a, b;  // tokens a..b inclusive
};

enum Lang { ENG, FOR, UNSURE };

struct Counts {
    int words = 0, en = 0, fr = 0, script = 0, accented = 0;
};

void count(const Doc& d, size_t k, Counts& c) {
    const rv::Token& t = d.tokens[k];
    if (t.kind != Kind::Word) {
        if (t.kind == Kind::Punct && foreign_script(t.cp)) ++c.script;
        return;
    }
    const char* w = d.lower.data() + t.start;
    size_t len = t.end - t.start;
    // letters glued to digits are parts of an identifier (9d1e4f20), not words
    if ((d.number(k - 1) && d.joined(k)) || (d.number(k + 1) && d.joined(k + 1))) return;
    if (foreign_script(t.cp)) {
        // a single Greek or Cyrillic letter is a symbol (alpha, sigma), not a word
        if (len > 2 || t.cp >= 0x590) c.script += 2;
        return;
    }
    if (len == 2 && w[0] == 'e' && w[1] == 't' && d.is(k + 1, "al")) return;  // "et al." in English citations
    ++c.words;
    if (len <= 7) {
        uint64_t key = pack7(std::string_view(w, len));
        c.accented += (key & 0x0080808080808080ull) != 0;
        int f = FUNCTION_WORDS.get_packed(key);
        // a capital letter after a name or before a mark or number is a label ("Route E", "Plan A:", "E 4"), and
        // "van 12" is an English van; "E poi siamo" and "Y luego" are words
        bool label = len == 1 && t.casing == rv::UPPER &&
                     ((k > 0 && d.word(k - 1) && d[k - 1].casing != rv::LOWER && !d[k].newline_before) || k + 1 >= d.size() ||
                      !d.word(k + 1) || d[k + 1].casing != rv::LOWER);
        if (label || (len == 3 && w[0] == 'v' && w[1] == 'a' && w[2] == 'n' && d.number(k + 1))) f = 0;
        if ((f & 2) && in_name(d, k)) f = 0;
        c.en += f & 1;
        c.fr += (f >> 1) & 1;
    } else {
        bool accent = false;
        for (size_t m = 0; m < len; ++m) accent |= static_cast<unsigned char>(w[m]) >= 0x80;
        c.accented += accent;
    }
}

Lang classify(const Counts& c) {
    if (c.script >= 2 && c.script >= c.words) return FOR;
    if (c.words >= 3 && c.fr >= 3 && c.fr > 2 * c.en) return FOR;
    if (c.words >= 4 && c.fr >= 2 && c.en == 0) return FOR;
    if (c.words >= 3 && c.en == 0 && c.accented * 3 >= c.words && c.accented >= 2) return FOR;
    if (c.en <= 1 && c.accented >= 3 && c.accented * 4 >= c.words) return FOR;
    if (c.en >= 2 && c.en > c.fr) return ENG;
    if (c.en >= 1 && c.fr == 0 && c.script == 0) return ENG;
    return UNSURE;
}

Lang language(const Doc& d, size_t a, size_t b) {
    Counts c;
    for (size_t k = a; k <= b; ++k) count(d, k, c);
    return classify(c);
}

bool foreign(const Doc& d, size_t a, size_t b) { return language(d, a, b) == FOR; }

bool closing_cp(uint32_t c) {
    return c == '"' || c == 0x201D || c == 0x201C || c == 0xBB || c == ')' || c == '\'' || c == 0x2019 ||
           c == 0x300D;
}

bool terminator(const Doc& d, size_t i) {
    if (!is_punct(d, i)) return false;
    uint32_t c = d[i].cp;
    if (c == 0x3002 || c == 0xFF01 || c == 0xFF1F || c == 0x061F || c == 0x06D4) return true;
    if (c != '.' && c != '!' && c != '?') return false;
    size_t k = i + 1;
    while (k < d.size() && d.joined(k) && is_punct(d, k) && closing_cp(d[k].cp)) ++k;
    return k >= d.size() || d[k].space_before;
}

bool open_quote(uint32_t cp) { return cp == 0x201E || cp == 0x201C || cp == 0xAB || cp == '"' || cp == 0x300C; }
bool close_quote_for(uint32_t open, uint32_t cp) {
    switch (open) {
        case 0x201E: return cp == 0x201C || cp == 0x201D;
        case 0x201C: return cp == 0x201D;
        case 0xAB: return cp == 0xBB;
        case '"': return cp == '"';
        case 0x300C: return cp == 0x300D;
        default: return false;
    }
}

// Sentences in another language are protected whole; in an English sentence, a quotation in another language
// is protected.
// Splits the text into sentences (and lines) and calls f(a, b, language) for each.
template <class F>
void sentences(const Doc& doc, F f) {
    size_t n = doc.size();
    Counts c;
    size_t a = 0;
    uint32_t quote = 0;  // the opening quote of a quotation being skipped: it is judged on its own
    for (size_t i = 0; i < n; ++i) {
        if (quote) {
            if (is_punct(doc, i) && close_quote_for(quote, doc[i].cp)) quote = 0;
        } else if (is_punct(doc, i) && open_quote(doc[i].cp) && (doc[i].cp != '"' || i == 0 || doc[i].space_before) &&
                   i > a) {
            quote = doc[i].cp;
        } else {
            count(doc, i, c);
        }
        bool end = i + 1 == n || doc[i + 1].newline_before || terminator(doc, i);
        if (!end) continue;
        size_t b = i;
        while (b + 1 < n && doc.joined(b + 1) && !doc[b + 1].newline_before && is_punct(doc, b + 1) &&
               closing_cp(doc[b + 1].cp))
            ++b;
        f(a, b, classify(c));
        c = Counts();
        quote = 0;
        i = b;
        a = b + 1;
    }
}

RV_RULE("structure", foreign_language) {
    if (doc.size() == 0) return;
    bool any_english = false, any_foreign = false;
    sentences(doc, [&](size_t sa, size_t sb, Lang l) {
        any_english |= l == ENG;
        any_foreign |= l == FOR && !gloss_sentence(doc, sa, sb);
    });
    bool quotes = has(doc, "\"") || has(doc, "\xE2\x80\x9C") || has(doc, "\xE2\x80\x9E") ||
                  has(doc, "\xC2\xAB") || has(doc, "\xE3\x80\x8C");
    if (!any_foreign && !quotes) return;
    sentences(doc, [&](size_t sa, size_t sb, Lang l) {
        if (l != ENG && gloss_sentence(doc, sa, sb)) return;
        if (l == FOR || (l == UNSURE && any_foreign && !any_english)) {
            out.keep(sa, sb, 99);
            return;
        }
        for (size_t k = sa; k <= sb; ++k) {  // a quotation in another language inside an English sentence
            if (!is_punct(doc, k) || !open_quote(doc[k].cp)) continue;
            size_t e = k + 1;  // the quotation may end after the sentence: « ... 2023. »
            while (e < doc.size() && !(e > sb && doc[e].newline_before) &&
                   !(is_punct(doc, e) && close_quote_for(doc[k].cp, doc[e].cp)))
                ++e;
            if (e >= doc.size() || doc[e].newline_before || !is_punct(doc, e)) break;
            if (e > k + 1 && foreign(doc, k + 1, e - 1)) out.keep(k, e, 99);
            k = e;
        }
    });
}

}  // namespace
