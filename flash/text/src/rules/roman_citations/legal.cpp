// Legal, scripture and page references: § and §§, U.S.C. and C.F.R. with their sections, Art. 14(2)(b),
// Rule 12(b)(6), Amend., bill and act numbers, canons, Bible verses (John 3:16, 1 Cor 13:4–7, Ps. 23), Surah 2:255,
// Talmud folios (Berakhot 2a), p. 12 and pp. 12–15.
#include "rules/roman_citations/common.h"

namespace {

using namespace rc;

// Capitalized words after "U.S.C." that continue the reference, so its period is not a sentence end.
const rv::WordSet LEGAL_NEXT{"app", "appendix", "ch", "chapter", "part", "sec", "section", "sections", "subpart",
                             "title"};

// Bible books: written form (lower case, no period) -> spoken name.
const rv::PhraseMap BOOKS{
    {"acts", "Acts"}, {"amos", "Amos"}, {"baruch", "Baruch"}, {"chr", "Chronicles"}, {"chron", "Chronicles"},
    {"chronicles", "Chronicles"}, {"col", "Colossians"}, {"colossians", "Colossians"}, {"cor", "Corinthians"},
    {"corinthians", "Corinthians"}, {"dan", "Daniel"}, {"daniel", "Daniel"}, {"deut", "Deuteronomy"},
    {"deuteronomy", "Deuteronomy"}, {"eccl", "Ecclesiastes"}, {"eccles", "Ecclesiastes"},
    {"ecclesiastes", "Ecclesiastes"}, {"eph", "Ephesians"}, {"ephesians", "Ephesians"}, {"esth", "Esther"},
    {"esther", "Esther"}, {"ex", "Exodus"}, {"exo", "Exodus"}, {"exod", "Exodus"}, {"exodus", "Exodus"},
    {"ezek", "Ezekiel"}, {"ezekiel", "Ezekiel"},
    {"ezra", "Ezra"}, {"gal", "Galatians"}, {"galatians", "Galatians"}, {"gen", "Genesis"}, {"genesis", "Genesis"},
    {"hab", "Habakkuk"}, {"habakkuk", "Habakkuk"}, {"hag", "Haggai"}, {"haggai", "Haggai"}, {"heb", "Hebrews"},
    {"hebrews", "Hebrews"}, {"hos", "Hosea"}, {"hosea", "Hosea"}, {"isa", "Isaiah"}, {"isaiah", "Isaiah"},
    {"jas", "James"}, {"james", "James"}, {"jer", "Jeremiah"}, {"jeremiah", "Jeremiah"}, {"jn", "John"},
    {"job", "Job"}, {"joel", "Joel"}, {"john", "John"}, {"jonah", "Jonah"}, {"josh", "Joshua"},
    {"joshua", "Joshua"}, {"jude", "Jude"}, {"judg", "Judges"}, {"judges", "Judges"}, {"judith", "Judith"},
    {"kgs", "Kings"}, {"kings", "Kings"}, {"lam", "Lamentations"}, {"lamentations", "Lamentations"},
    {"lev", "Leviticus"}, {"leviticus", "Leviticus"}, {"lk", "Luke"}, {"luke", "Luke"}, {"macc", "Maccabees"},
    {"maccabees", "Maccabees"}, {"mal", "Malachi"}, {"malachi", "Malachi"}, {"mark", "Mark"}, {"matt", "Matthew"},
    {"matthew", "Matthew"}, {"mic", "Micah"}, {"micah", "Micah"}, {"mk", "Mark"}, {"mt", "Matthew"},
    {"nah", "Nahum"}, {"nahum", "Nahum"}, {"neh", "Nehemiah"}, {"nehemiah", "Nehemiah"}, {"num", "Numbers"},
    {"numbers", "Numbers"}, {"obad", "Obadiah"}, {"obadiah", "Obadiah"}, {"pet", "Peter"}, {"peter", "Peter"},
    {"phil", "Philippians"}, {"philem", "Philemon"}, {"philemon", "Philemon"}, {"philippians", "Philippians"},
    {"phlm", "Philemon"}, {"solomon", "Solomon"}, {"songs", "Songs"},
    {"prov", "Proverbs"}, {"proverbs", "Proverbs"}, {"ps", "Psalm"}, {"psa", "Psalm"}, {"psalm", "Psalm"},
    {"psalms", "Psalms"}, {"rev", "Revelation"}, {"revelation", "Revelation"}, {"rom", "Romans"},
    {"romans", "Romans"}, {"ruth", "Ruth"}, {"sam", "Samuel"}, {"samuel", "Samuel"}, {"sir", "Sirach"},
    {"sirach", "Sirach"}, {"thess", "Thessalonians"}, {"thessalonians", "Thessalonians"}, {"tim", "Timothy"},
    {"timothy", "Timothy"}, {"tit", "Titus"}, {"titus", "Titus"}, {"tobit", "Tobit"}, {"zech", "Zechariah"},
    {"zechariah", "Zechariah"}, {"zeph", "Zephaniah"}, {"zephaniah", "Zephaniah"}};
// Books that come in numbered parts: "1 Cor", "2 Kings", "First Peter".
const rv::WordSet NUMBERED_BOOKS{"chr", "chron", "chronicles", "cor", "corinthians", "john", "jn", "kgs", "kings",
                                 "macc", "maccabees", "pet", "peter", "sam", "samuel", "thess", "thessalonians",
                                 "tim", "timothy"};
// Abbreviations that are read as the book also without a verse ("Ps. 23" -> Psalm twenty-three).
const rv::WordSet BOOK_ABBR_ALONE{"ps", "psa"};
// Names of the Quran's chapters.
const rv::WordSet SURAH{"koran", "quran", "qur'an", "qur\xE2\x80\x99" "an", "sura", "surah"};
// Talmud tractates (the last word of a two-word name).
const rv::WordSet TRACTATES{
    "arakhin", "avodah", "batra", "beitzah", "bekhorot", "berakhot", "chagigah", "chullin", "eruvin", "gittin",
    "hashanah", "horayot", "kamma", "katan", "keritot", "ketubot", "kiddushin", "makkot", "megillah", "meilah",
    "menachot", "metzia", "nazir", "nedarim", "niddah", "pesachim", "sanhedrin", "shabbat", "shevuot", "sotah",
    "sukkah", "sukkot", "taanit", "tamid", "temurah", "yevamot", "yoma", "zarah", "zevachim"};

// The token after a name, past a closing emphasis mark: "*Berakhot* 3b".
size_t past_emphasis(const Doc& d, size_t k) {
    return (d.punct(k, '*') || d.punct(k, '_')) && d.joined(k) ? k + 1 : k;
}

// Abbreviation with periods ending at token last ('.'): replaced by spoken; the final period stays when it also
// ends the sentence.
void abbreviation(const Doc& d, Out& out, size_t first, size_t last, const char* spoken, int score) {
    bool keep = d.punct(last, '.') && period_ends_sentence(d, last);
    if (keep) {
        size_t n = last + 1;
        if (n < d.size() && !d[n].newline_before && d.word(n) && LEGAL_NEXT.contains(d.low(n))) keep = false;
    }
    out.tokens(first, keep ? last - 1 : last, spoken, score);
}

// "U.S.C." / "C.F.R." / "H.R." style: single capital letters joined by periods, spelled as given (e.g. "USC").
// Returns the index of the last '.', npos if the letters do not match.
size_t dotted_letters(const Doc& d, size_t i, std::string_view letters) {
    size_t k = i;
    for (char c : letters) {
        if (!d.word(k) || d[k].end - d[k].start != 1 || d.text[d[k].start] != c) return npos;
        if (k > i && !d.joined(k)) return npos;
        if (!d.punct(k + 1, '.') || !d.joined(k + 1)) return npos;
        k += 2;
    }
    return k - 1;
}

// A section number after "§", "Section", "CFR": 1001 -> ten oh one, 4.2.1 -> four point two point one, with
// "(b)(6)" subdivisions and ranges "10-15".
std::string section_reference(const Doc& d, size_t k, size_t& last, bool letters_plain, bool plural) {
    std::string words = section_number(d, k, last);
    if (words.empty()) return "";
    size_t sub_last;
    words += subdivisions(d, last, sub_last, letters_plain);
    last = sub_last;
    if (dash_tok(d, last + 1) && d.number(last + 2) && (d.joined(last + 1) || d[last + 1].space_before)) {
        size_t l2;
        std::string w2 = section_number(d, last + 2, l2);
        if (!w2.empty() && (plural || d.joined(last + 1))) {
            // "§§ 10-15" is a range; "§ 78j-1" is one section whose parts are read one after the other
            words += (plural ? " to " : " ") + w2;
            last = l2;
        }
    }
    return words;
}

// "§ 241" -> Section two forty-one, "§§ 10-15" -> Sections ten to fifteen, "§ 4(b)" -> Section four B.
size_t section_sign_at(const Doc& doc, Out& out, size_t i) {
    bool para = doc.punct(i, 0xB6);
    if (!doc.punct(i, 0xA7) && !para) return i;
    size_t k = i + 1;
    bool plural = doc.punct(k, doc[i].cp) && doc.joined(k);
    if (plural) ++k;
    std::string label = para ? (plural ? "paragraphs" : "paragraph") : plural ? "Sections" : "Section";
    // a quoted pilcrow is named: "marked with a '¶'"
    if (para && i > 0 && doc.joined(i) && punct_tok(doc, i - 1) && doc[i - 1].space_before && k < doc.size() &&
        doc.joined(k) && punct_tok(doc, k)) {
        uint32_t q = doc[i - 1].cp;
        if (q == '\'' || q == '"' || q == 0x2018 || q == 0x201C) {
            out.tokens(i, i, "paragraph", 70);
            return i;
        }
    }
    // a footnote mark at the end of a word or sentence: "this year.§"
    if (!plural && i > 0 && doc.joined(i) && (k >= doc.size() || doc[k].space_before || punct_tok(doc, k)) &&
        !doc.number(k)) {
        out.add(doc[i].start, doc[i].end, "", 64);
        return i;
    }
    // a link to a section of a page: "See also: Energy drink § Regulations" -> "Energy drink, section Regulations",
    // "(see § 21st-century surveys)" -> "(see section twenty-first-century surveys)"
    bool ordinal_title = doc.number(k) && doc.word(k + 1) && doc.joined(k + 1) &&
                         (doc.is(k + 1, "st") || doc.is(k + 1, "nd") || doc.is(k + 1, "rd") || doc.is(k + 1, "th"));
    if (!para && !plural && i > 0 && doc[i].space_before && !doc[i].newline_before && k < doc.size() &&
        doc[k].space_before && !doc[k].newline_before &&
        ((doc.word(k) && doc[k].casing != rv::LOWER && !rv::roman_value(doc.str(k))) || ordinal_title)) {
        if (doc.is(i - 1, "see") || doc.is(i - 1, "in") || doc.punct(i - 1, '(')) {
            out.tokens(i, i, "section", 70);
            return i;
        }
        if (doc.word(i - 1) || doc.punct(i - 1, ')')) {
            out.add(doc[i - 1].end, doc[i].end, ", section", 70);
            return i;
        }
    }
    if (doc.number(k)) {
        size_t last;
        std::string words = section_reference(doc, k, last, true, plural);
        if (words.empty()) return i;
        out.tokens(i, last, label + " " + words, 78);
        i = last;
    } else if (para && !plural && i > 0 && doc[i].space_before && !doc[i].newline_before) {
        // a pilcrow named or standing in the text: "a lone ¶," -> paragraph; before a capitalized word it starts
        // one: "read: ¶ See appendix A" -> Paragraph See
        bool leads = title(doc, k) && doc[k].space_before;
        out.tokens(i, i, leads ? "Paragraph" : "paragraph", 70);
    } else {
        Roman R = roman(doc, k);
        if (R.value && R.upper && R.suffix.empty() && !para) {
            out.tokens(i, k, label + " " + card_cap(R.value), 78);
            i = k;
        }
    }
    return i;
}

// "U.S.C." -> USC, "C.F.R." -> CFR, and the section number after them in pairs: "18 U.S.C. 1001" -> eighteen USC
// ten oh one, "40 CFR 191.13" -> forty CFR one ninety-one point thirteen.
size_t code_abbreviation_at(const Doc& doc, Out& out, size_t i) {
    size_t len = doc[i].end - doc[i].start;
    if (!doc.word(i) || (len != 1 && len != 3) || (first_lower(doc, i) != 'u' && first_lower(doc, i) != 'c')) return i;
    size_t last = npos, code_end = npos;
    const char* spoken = nullptr;
    if ((last = dotted_letters(doc, i, "USC")) != npos) spoken = "USC";
    else if ((last = dotted_letters(doc, i, "CFR")) != npos) spoken = "CFR";
    if (spoken) {
        if (i > 0 && doc.joined(i) && doc.word(i - 1)) return i;
        abbreviation(doc, out, i, last, spoken, 80);
        code_end = last;
    } else if (doc.is(i, "usc") || doc.is(i, "cfr")) {
        if (!upper(doc, i)) return i;
        code_end = i;
    } else {
        return i;
    }
    size_t k = code_end + 1;
    if (doc.number(k) && doc[k].space_before && !doc[k].newline_before) {
        size_t l;
        std::string words = section_reference(doc, k, l, false, false);
        if (!words.empty()) out.tokens(k, l, words, 70);
    }
    return code_end;
}

// "Section 230" -> Section two thirty: sections of laws with three or four digits in pairs; nested numbers and
// subdivisions by their parts.
size_t section_word_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 1 < doc.size())) return i;
    // "Sec. 3(b)" -> Section three B: with its subdivisions, above the abbreviation rule that reads "Sec. 3"
    if ((doc.is(i, "sec") || doc.is(i, "secs")) && doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.number(i + 2) &&
        doc[i + 2].space_before && doc.punct(i + 3, '(') && doc.joined(i + 3)) {
        bool plural = doc.is(i, "secs");
        size_t last;
        std::string words = section_reference(doc, i + 2, last, true, plural);
        if (words.empty() || last <= i + 2) return i;
        std::string label = doc.text[doc[i].start] == 'S' ? "Section" : "section";
        out.tokens(i, last, label + (plural ? "s " : " ") + words, 81);
        return last;
    }
    // "Clause 7(a)" -> Clause seven A, like a section
    if ((doc.is(i, "clause") || doc.is(i, "subsection") || doc.is(i, "subparagraph")) && doc.number(i + 1) &&
        doc[i + 1].space_before && doc.punct(i + 2, '(') && doc.joined(i + 2) && doc.str(i + 1).size() <= 3) {
        size_t last;
        std::string words = section_reference(doc, i + 1, last, true, false);
        if (!words.empty() && last > i + 1) out.tokens(i + 1, last, words, 66);
        return last;
    }
    if (!doc.number(i + 1) || (!doc.is(i, "section") && !doc.is(i, "sections"))) return i;
    size_t k = i + 1;
    if (!doc[k].space_before) return i;
    if (doc.joined(k + 1) && doc.word(k + 1) && !single_letter(doc, k + 1)) return i;
    size_t len = doc.str(k).size();
    bool nested = doc.punct(k + 1, '.') && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2) &&
                  doc.punct(k + 3, '.') && doc.joined(k + 3) && doc.number(k + 4) && doc.joined(k + 4);
    bool sub = doc.punct(k + 1, '(') && doc.joined(k + 1);
    if (!(len == 3 || len == 4 || nested || sub)) return i;
    size_t last;
    std::string words = section_reference(doc, k, last, true, doc.is(i, "sections"));
    if (words.empty()) return i;
    out.tokens(k, last, words, 66);
    // "Sections 3(a) and 3(b)", "Sections 201.3, 201.4 and 201.7": the other numbers in the list
    for (int guard = 0; guard < 6 && doc.is(i, "sections"); ++guard) {
        size_t c = last + 1;
        if (doc.punct(c, ',')) ++c;
        if (doc.is(c, "and") || doc.is(c, "or")) ++c;
        if (c == last + 1 || !doc.number(c) || !doc[c].space_before) break;
        size_t l2;
        std::string w2 = section_reference(doc, c, l2, true, true);
        if (w2.empty()) break;
        out.tokens(c, l2, w2, 66);
        last = l2;
    }
    return last;
}

// "Art. 14(2)(b)" -> Article fourteen, paragraph two, letter B; "Article 15(1)(h)"; "Rule 12(b)(6)" -> Rule twelve,
// letter B, paragraph six.
size_t article_reference_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 2 < doc.size())) return i;
    size_t len = doc[i].end - doc[i].start;
    if (!doc.word(i) || (len != 3 && len != 4 && len != 7 && len != 8 && len != 10) ||
        (first_lower(doc, i) != 'a' && first_lower(doc, i) != 'r'))
        return i;
    size_t k;
    bool art_abbr = doc.is(i, "art") && doc.punct(i + 1, '.') && doc.joined(i + 1);
    if (art_abbr) {
        k = i + 2;
        if (!doc[k].space_before && !doc.number(k)) return i;
        Roman R = roman(doc, k);
        if (!doc.number(k) && !(R.value && R.upper)) return i;
        out.tokens(i, i + 1, doc.text[doc[i].start] == 'A' ? "Article" : "article", 76);
    } else if (doc.is(i, "article") || doc.is(i, "articles") || doc.is(i, "rule") || doc.is(i, "regulation")) {
        k = i + 1;
        if (!doc[k].space_before) return i;
    } else {
        return i;
    }
    if (!doc.number(k) || doc.str(k).size() > 4) return i;
    std::string words = plain(doc, k);
    if (words.empty()) return i;
    size_t last;
    std::string sub = subdivisions(doc, k, last);
    if (sub.empty()) return i;
    out.tokens(k, last, words + sub, 76);
    return last;
}

// "1st Amend." -> First Amendment, "Amend." -> Amendment.
size_t amendment_at(const Doc& doc, Out& out, size_t i) {
    size_t len = doc[i].end - doc[i].start;
    if (!doc.word(i) || (len != 5 && len != 9) || first_lower(doc, i) != 'a') return i;
    bool abbr = doc.is(i, "amend") && doc.punct(i + 1, '.') && doc.joined(i + 1) && title(doc, i);
    bool full = doc.is(i, "amendment") && title(doc, i);
    if (!abbr && !full) return i;
    size_t last = abbr ? i + 1 : i;
    if (abbr && period_ends_sentence(doc, last)) --last;
    // an ordinal before it: "5th Amend." -> Fifth Amendment
    if (i >= 2 && doc.word(i - 1) && doc.number(i - 2) && doc.joined(i - 1) && doc[i].space_before) {
        std::string_view suf = doc.low(i - 1);
        if (suf == "st" || suf == "nd" || suf == "rd" || suf == "th") {
            std::string w = rv::ordinal(num(doc, i - 2));
            if (doc.str(i - 2).size() <= 2) {
                out.tokens(i - 2, last, rv::capital(w) + " Amendment", 76);
                return last;
            }
        }
    }
    if (abbr) out.tokens(i, last, "Amendment", 75);
    return last;
}

// Bill and act numbers digit by digit: "Bill H.R. 1234" -> Bill HR one two three four, "Act 107" -> Act one oh
// seven, "Bill S.1042" -> Bill S one oh four two.
size_t bill_number_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 1 < doc.size())) return i;
    size_t len = doc[i].end - doc[i].start;
    if (!doc.word(i) || (len != 1 && len != 3 && len != 4)) return i;
    char f = first_lower(doc, i);
    if (f != 'h' && f != 's' && f != 'a' && f != 'b') return i;
    size_t last = npos, k = npos;
    const char* label = nullptr;
    if ((last = dotted_letters(doc, i, "HR")) != npos) label = "HR";
    else if ((last = dotted_letters(doc, i, "SB")) != npos) label = "SB";
    else if ((last = dotted_letters(doc, i, "HB")) != npos) label = "HB";
    else if ((last = dotted_letters(doc, i, "AB")) != npos) label = "AB";
    std::string letters;
    if (!label && i > 0 && doc.is(i - 1, "bill")) {
        // any dotted capitals after "Bill": "S.", "S.R.", "S.F."
        size_t k = i;
        while (letters.size() < 3 && doc.word(k) && doc[k].end - doc[k].start == 1 && upper(doc, k) &&
               (k == i || doc.joined(k)) && doc.punct(k + 1, '.') && doc.joined(k + 1)) {
            letters += doc.text[doc[k].start];
            last = k + 1;
            k += 2;
        }
        if (!letters.empty()) label = letters.c_str();
    }
    if (label) {
        k = last + 1;
        if (!doc.number(k) || doc.str(k).size() < 2 || doc.str(k).size() > 5) return i;
        out.tokens(i, last, label, 70);
        out.tokens(k, k, digits_oh(doc.str(k), false), 70);
        return k;
    }
    bool bill = doc.is(i, "bill") && title(doc, i);
    bool act = doc.is(i, "act") && title(doc, i);
    if (!bill && !act) return i;
    k = i + 1;
    if (!doc.number(k) || !doc[k].space_before || doc.str(k).size() < 3 || doc.str(k).size() > 5) return i;
    if (doc.joined(k + 1) && (doc.word(k + 1) || (doc.punct(k + 1, '.') && doc.number(k + 2)))) return i;
    out.tokens(k, k, digits_oh(doc.str(k), act), 70);
    return k;
}

// A spaced subdivision after its word: "subsection (b)" -> subsection B, "paragraph (2)" -> paragraph two.
size_t subdivision_word_at(const Doc& doc, Out& out, size_t i) {
    static const rv::WordSet WORDS{"clause", "paragraph", "section", "subparagraph", "subsection"};
    if (!(i + 3 < doc.size())) return i;
    if (!doc.punct(i + 1, '(') || !doc.word(i) || !WORDS.contains(doc.low(i)) || !doc[i + 1].space_before ||
        !doc.punct(i + 3, ')') || !doc.joined(i + 2) || !doc.joined(i + 3))
        return i;
    std::string w;
    if (char c = single_letter(doc, i + 2)) w = letter_cap(c);
    else if (doc.number(i + 2) && doc.str(i + 2).size() <= 3 && doc.word(i + 4)) w = plain(doc, i + 2);
    if (w.empty()) return i;  // "each draft clause (4)." is a citation
    out.tokens(i + 1, i + 3, w, 66);
    return i + 3;
}

// "subsection 2b" -> subsection two B.
size_t subdivision_letter_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 2 < doc.size())) return i;
    if (!doc.number(i + 1) || !doc.word(i + 2)) return i;
    if (!doc.is(i, "subsection") && !doc.is(i, "paragraph") && !doc.is(i, "clause")) return i;
    size_t k = i + 1;
    char c = single_letter(doc, k + 1);
    if (!doc.number(k) || !doc[k].space_before || !c || !doc.joined(k + 1) || doc.str(k).size() > 3) return i;
    if (doc.joined(k + 2) && (doc.word(k + 2) || doc.number(k + 2))) return i;
    out.tokens(k, k + 1, plain(doc, k) + " " + letter_cap(c), 66);
    return i;
}

// "Canon 101" -> Canon one hundred one (a canon of law, never a camera model number).
size_t canon_number_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 1 < doc.size())) return i;
    if (!doc.number(i + 1) || !doc.is(i, "canon") || !title(doc, i) || !doc[i + 1].space_before) return i;
    if (doc.joined(i + 2) && (doc.word(i + 2) || (doc.punct(i + 2, '.') && doc.number(i + 3) && doc.joined(i + 3))))
        return i;
    std::string w = plain(doc, i + 1);
    if (!w.empty()) out.tokens(i + 1, i + 1, w, 65);
    return i;
}

// Bible verses: "John 3:16" -> John chapter three, sixteen; "1 Cor 13:4–7" -> First Corinthians chapter thirteen,
// four to seven; "Ps. 23" -> Psalm twenty-three. Quran: "Surah 2:255" -> Surah two, two hundred fifty-five.
size_t scripture_at(const Doc& doc, Out& out, size_t i) {
    static const char* const ORD[] = {"", "First", "Second", "Third"};
    if (!(i + 1 < doc.size())) return i;
    if (!doc.word(i) || doc.text[doc[i].start] < 'A' || doc.text[doc[i].start] > 'Z') return i;
    {  // a number follows (after a period or an emphasis mark): checked before the book lookup
        size_t q = i + 1;
        if (doc.punct(q, '.') || doc.punct(q, '*') || doc.punct(q, '_')) ++q;
        if (doc.punct(q, '*') || doc.punct(q, '_')) ++q;
        if (!doc.number(q)) return i;
    }
    std::string_view key = doc.low(i);
    bool surah = SURAH.contains(key);
    const std::string_view* book = surah ? nullptr : BOOKS.find(key);
    if (!surah && !book) return i;
    // "Song of Solomon 2:1", "Song of Songs 2:1": the name is written, only the reference is read
    if ((key == "solomon" || key == "songs") && !(i >= 2 && doc.is(i - 1, "of") && doc.is(i - 2, "song"))) return i;
    size_t k = i + 1;
    bool period = doc.punct(k, '.') && doc.joined(k);
    if (period) ++k;
    k = past_emphasis(doc, k);
    if (!doc.number(k) || !doc[k].space_before || doc[k].newline_before) return i;
    // a numbered book: "1 Cor", "2 Kings", "II Samuel", "First Peter"
    size_t first = i;
    std::string prefix;
    if (book && NUMBERED_BOOKS.contains(key) && i > 0 && doc[i].space_before) {
        size_t p = i - 1;
        int n = 0;
        bool verse_after = doc.punct(k + 1, ':') && doc.joined(k + 1) && doc.number(k + 2);
        if (doc.number(p) && doc.str(p).size() == 1) n = static_cast<int>(num(doc, p));
        // "1st Corinthians 13:4"
        else if (p > 0 && doc.word(p) && doc.number(p - 1) && doc.joined(p) && doc.str(p - 1).size() == 1 &&
                 (doc.is(p, "st") || doc.is(p, "nd") || doc.is(p, "rd")) && verse_after) {
            n = static_cast<int>(num(doc, p - 1));
            --p;
        }
        // "II Kings 2:11"; a lone "I" only before a verse: "I Cor. 13:4"
        else if (Roman R = roman(doc, p); R.value && R.value <= 3 && R.upper && (R.base.size() > 1 || verse_after))
            n = R.value;
        if (n >= 1 && n <= 3 && !(p > 0 && doc.joined(p))) {
            prefix = std::string(ORD[n]) + " ";
            first = p;
        }
    }
    bool verse = doc.punct(k + 1, ':') && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2);
    if (!verse) {
        // chapter only: "Ps. 23" and numbered books ("1 Cor 13") get the book's name, the number stays
        if (book && ((period && BOOK_ABBR_ALONE.contains(key)) || !prefix.empty()))
            out.add(doc[first].start, doc[period ? i + 1 : i].end, prefix + std::string(*book), 66);
        return i;
    }
    // a clock time ("Mark 4:30 pm") is not a verse
    size_t after = k + 3;
    if (doc.word(after) && (doc.is(after, "am") || doc.is(after, "pm") || doc.is(after, "a") || doc.is(after, "p")))
        return i;
    std::string words;
    if (surah) words = std::string(doc.str(i)) + " " + plain(doc, k) + ", " + plain(doc, k + 2);
    else words = prefix + std::string(*book) + " chapter " + plain(doc, k) + ", " + plain(doc, k + 2);
    size_t last = k + 2;
    if (dash_tok(doc, last + 1) && doc.joined(last + 1) && doc.number(last + 2) && doc.joined(last + 2)) {
        if (!surah && doc.punct(last + 3, ':') && doc.joined(last + 3) && doc.number(last + 4)) {
            words += " to chapter " + plain(doc, last + 2) + ", " + plain(doc, last + 4);
            last += 4;
        } else {
            words += " to " + plain(doc, last + 2);
            last += 2;
        }
    }
    // above the parts it covers: "Rev." as an abbreviation, "21:4" as a ratio or a time
    out.add(doc[first].start, doc[last].end, words, 82);
    return last;
}

// Talmud folios: "Berakhot 2a" -> Berakhot two A.
size_t talmud_folio_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 2 < doc.size())) return i;
    if (!(doc.number(i + 1) || doc.number(i + 2)) || !title(doc, i) || !TRACTATES.contains(doc.low(i))) return i;
    size_t k = past_emphasis(doc, i + 1);
    if (!doc.number(k) || !doc[k].space_before) return i;
    char c = single_letter(doc, k + 1);
    if ((c != 'a' && c != 'b') || !doc.joined(k + 1)) return i;
    if (doc.joined(k + 2) && (doc.word(k + 2) || doc.number(k + 2))) return i;
    out.tokens(k, k + 1, plain(doc, k) + " " + letter_cap(c), 70);
    return k + 1;
}

// One part of a regulation number in pairs: 2021 twenty twenty-one, 1874 eighteen seventy-four, 2006 twenty oh
// six, 745 seven forty-five, 65 sixty-five.
std::string regulation_part(const Doc& d, size_t i) {
    uint32_t v = num(d, i);
    if (v >= 2001 && v <= 2009) return "twenty oh " + rv::cardinal(v % 10);
    return v >= 100 ? rv::pairs(v) : rv::cardinal(v);
}

// European legislation: "Regulation (EU) 2021/1874" -> Regulation EU twenty twenty-one slash eighteen seventy-four,
// "Directive 2019/790", "Regulation 1907/2006".
const rv::WordSet REGULATION_WORDS{"decision", "directive", "regulation", "regulations", "directives"};

size_t regulation_at(const Doc& doc, Out& out, size_t i) {
    if (!REGULATION_WORDS.contains(doc.low(i))) return i;
    size_t k = i + 1;
    std::string words;
    bool paren = doc.punct(k, '(') && doc[k].space_before && upper(doc, k + 1) && doc.punct(k + 2, ')') &&
                 doc.joined(k + 1) && doc.joined(k + 2);
    size_t body = paren ? k + 3 : k;
    if (paren || (upper(doc, k) && (doc.is(k, "eu") || doc.is(k, "ec") || doc.is(k, "eec")))) {
        std::string_view code = doc.str(paren ? k + 1 : k);
        if (code != "EU" && code != "EC" && code != "EEC") return i;
        words = std::string(code) + " ";
        if (!paren) body = k + 1;
    }
    if (!doc.number(body) || !doc[body].space_before || !doc.punct(body + 1, '/') || !doc.joined(body + 1) ||
        !doc.number(body + 2) || !doc.joined(body + 2))
        return i;
    size_t a = body, b = body + 2;
    if (doc.str(a).size() < 2 || doc.str(a).size() > 4 || doc.str(b).size() > 4 || doc.str(a)[0] == '0' ||
        doc.str(b)[0] == '0')
        return i;
    if (doc.joined(b + 1) && !(doc.punct(b + 1, '.') || doc.punct(b + 1, ',') || doc.punct(b + 1, ')') ||
                               doc.punct(b + 1, ';') || doc.punct(b + 1, ':')))
        return i;
    if ((doc.punct(b + 1, '.') || doc.punct(b + 1, ',')) && doc.number(b + 2) && doc.joined(b + 2)) return i;
    words += regulation_part(doc, a) + " slash " + regulation_part(doc, b);
    out.tokens(paren || !words.empty() ? k : body, b, words, 80);
    return b;
}

// Page references: "p. 12" -> page twelve, "pp. 12–15" -> pages twelve to fifteen.
size_t page_reference_at(const Doc& doc, Out& out, size_t i) {
    if (!(i + 2 < doc.size())) return i;
    if (!doc.punct(i + 1, '.') || !doc.word(i) || first_lower(doc, i) != 'p') return i;
    bool pg = (doc.is(i, "pg") || doc.is(i, "pgs")) && lower(doc, i);
    bool p = (doc.is(i, "p") || doc.is(i, "pg")) && lower(doc, i);
    bool pp = (doc.is(i, "pp") || doc.is(i, "pgs")) && lower(doc, i);
    if (!p && !pp) return i;
    if (!doc.punct(i + 1, '.') || !doc.joined(i + 1)) return i;
    if (i > 0 && doc.joined(i) && !doc.punct(i - 1, '(') && !doc.punct(i - 1, '[')) return i;
    size_t k = i + 2;
    // "p.12" with the number joined: only digits, not "p.m." or "p.s."
    bool joined_number = !doc[k].space_before && doc.number(k) && !pg;
    if ((!doc[k].space_before && !joined_number) || doc[k].newline_before) return i;
    Roman R = roman(doc, k);
    bool roman_page = R.value && !R.upper && R.suffix.empty();
    if (!doc.number(k) && !roman_page) return i;
    out.tokens(i, i + 1, pp ? "pages" : "page", 70);
    if (doc.number(k) && dash_tok(doc, k + 1) && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2)) {
        std::string a = plain(doc, k), b = plain(doc, k + 2);
        if (!a.empty() && !b.empty()) out.tokens(k, k + 2, a + " to " + b, 70);
        i = k + 2;
    } else if (joined_number) {
        std::string a = plain(doc, k);
        if (!a.empty()) out.tokens(k, k, a, 70);
        i = k;
    }
    // "pp. 45 ff." -> pages forty-five and following
    size_t f = doc.number(i) ? i + 1 : k + 1;
    if (doc.number(f - 1) && doc.is(f, "ff") && doc[f].space_before && lower(doc, f)) {
        bool dot = doc.punct(f + 1, '.') && doc.joined(f + 1) && !period_ends_sentence(doc, f + 1);
        out.tokens(f, dot ? f + 1 : f, "and following", 70);
    }
    return i;
}

// One scan for every legal, scripture and page rule: each token goes to the rules its first byte can start, and each
// rule resumes after the tokens it read.
RV_RULE("roman_citations", legal_reference) {
    enum { SECTION_SIGN, CODE, SECTION_WORD, ARTICLE, AMENDMENT, BILL, SUBDIVISION, SUB_LETTER, CANON, SCRIPTURE,
           TALMUD, PAGE, REGULATION, COUNT };
    size_t next[COUNT] = {};
    auto run = [&](int k, size_t (*fn)(const Doc&, Out&, size_t), size_t i) {
        if (i >= next[k]) next[k] = fn(doc, out, i) + 1;
    };
    for (size_t i = 0; i < doc.size(); ++i) {
        if (punct_tok(doc, i)) {
            run(SECTION_SIGN, section_sign_at, i);
            continue;
        }
        if (!doc.word(i)) continue;
        switch (first_lower(doc, i)) {
            case 'a': run(ARTICLE, article_reference_at, i); run(AMENDMENT, amendment_at, i); run(BILL, bill_number_at, i); break;
            case 'd': run(REGULATION, regulation_at, i); break;
            case 'b': case 'h': run(BILL, bill_number_at, i); break;
            case 'c': run(CODE, code_abbreviation_at, i); run(SECTION_WORD, section_word_at, i);
                      run(SUBDIVISION, subdivision_word_at, i);
                      run(SUB_LETTER, subdivision_letter_at, i); run(CANON, canon_number_at, i); break;
            case 'p': run(PAGE, page_reference_at, i); run(SUBDIVISION, subdivision_word_at, i);
                      run(SUB_LETTER, subdivision_letter_at, i); break;
            case 'r': run(ARTICLE, article_reference_at, i); run(REGULATION, regulation_at, i); break;
            case 's': run(SECTION_WORD, section_word_at, i); run(SUBDIVISION, subdivision_word_at, i);
                      run(SUB_LETTER, subdivision_letter_at, i); run(BILL, bill_number_at, i); break;
            case 'u': run(CODE, code_abbreviation_at, i); break;
            default: break;
        }
        if (doc.text[doc[i].start] >= 'A' && doc.text[doc[i].start] <= 'Z') {
            run(SCRIPTURE, scripture_at, i);
            run(TALMUD, talmud_folio_at, i);
        }
    }
}

}  // namespace
