// Rules of the module "units_symbols": chemical formulas, isotopes, virus and disease names with numbers,
// particle and substance names, pH, musical keys.
#include "rules/units_symbols/common.h"

namespace {

using namespace us;

const rv::WordSet ELEMENTS{
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

// The letters of w split into element symbols ('NaCl' -> Na Cl, 'CO' -> C O); false if impossible.
bool elements_only(std::string_view w, int& count) {
    size_t i = 0;
    count = 0;
    while (i < w.size()) {
        if (!(w[i] >= 'A' && w[i] <= 'Z')) return false;
        if (i + 1 < w.size() && w[i + 1] >= 'a' && w[i + 1] <= 'z' && ELEMENTS.contains(w.substr(i, 2))) {
            i += 2;
        } else if (ELEMENTS.contains(w.substr(i, 1))) {
            i += 1;
        } else {
            return false;
        }
        ++count;
    }
    return count > 0;
}

// Molecules written as one symbol group and one number.
const rv::WordSet MOLECULES{"CH4", "CO2", "CO3", "H2", "N2", "NH3", "NH4", "NO2", "NO3", "O2", "O3", "PO4", "SO2",
                            "SO3", "SO4", "B12", "B6", "D3", "K2", "C4", "C60"};

// Chemical formulas with ASCII or subscript digits: 'H2O' -> 'H two O', 'CO2' -> 'CO two', 'Fe2O3', 'C6H12O6',
// 'CO2e' -> 'CO two equivalent'. Codes that only look like formulas read the same way ('B12', 'H5N1').
void chemistry_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i) || stuck_before(doc, i) || doc[i].cp < 'A' || doc[i].cp > 'Z') return;
    // a digit must follow directly (ASCII or subscript)
    if (!(i + 1 < doc.size() && doc.joined(i + 1) && (doc.number(i + 1) || sub_tok(doc, i + 1)))) return;
    if (i > 0 && doc.joined(i) && (doc.punct(i - 1, '-') || doc.punct(i - 1, '/') || doc.punct(i - 1, '.') ||
                                   doc.punct(i - 1, '_') || doc.punct(i - 1, '@')))
        return;
    std::vector<std::string> parts;
    size_t k = i;
    int elements = 0, groups = 0;
    bool ascii = false, ok = true, equivalent = false;
    size_t last = i;
    while (k < doc.size() && (k == i || doc.joined(k))) {
        if (doc.word(k)) {
            std::string_view w = doc.str(k);
            int n = 0;
            if (!parts.empty() && (w == "e" || w == "eq") && !stuck_after(doc, k) && groups > 0) {
                equivalent = true;
                last = k;
                ++k;
                break;
            }
            if (!elements_only(w, n)) {
                ok = false;
                break;
            }
            bool all_upper = true;
            for (char c : w) all_upper &= c >= 'A' && c <= 'Z';
            if (all_upper && w.size() > 3) {
                ok = false;
                break;
            }
            elements += n;
            parts.emplace_back(w);
            last = k;
            ++k;
        } else if (doc.number(k)) {
            std::string_view n = doc.str(k);
            if (parts.empty() || n.size() > 2 || n[0] == '0') {
                ok = false;
                break;
            }
            parts.push_back(rv::cardinal_digits(n));
            ascii = true;
            ++groups;
            last = k;
            ++k;
        } else if (sub_tok(doc, k)) {
            std::string dg;
            size_t e = digit_run(doc, k, false, dg);
            if (parts.empty()) {
                ok = false;
                break;
            }
            parts.push_back(rv::cardinal_digits(dg));
            ++groups;
            last = e;
            k = e + 1;
        } else {
            break;
        }
    }
    if (!ok || groups == 0 || elements < 1 || stuck_after(doc, last)) return;
    // one word in capitals and one number is a code ("COP28", "SKY2", "CB1") unless it is a known molecule; a
    // formula has element symbols in mixed case, two symbol groups or two numbers ("NaCl2", "H2O", "C6H12O6")
    {
        int words = 0;
        bool mixed = false;
        for (size_t k = i; k <= last; ++k)
            if (doc.word(k) && !(equivalent && k == last)) {
                ++words;
                for (char c : doc.str(k)) mixed |= c >= 'a' && c <= 'z';
            }
        if (words == 1 && groups == 1 && !mixed && doc.number(i + 1) &&
            !MOLECULES.contains(std::string(doc.str(i)) + std::string(doc.str(i + 1))))
            return;  // (subscript digits are always chemistry: "SO₂")
    }
    // a decimal after the formula is something else ('PM2.5' is handled apart)
    if (doc.punct(last + 1, '.') && doc.joined(last + 1) && doc.number(last + 2) && doc.joined(last + 2)) return;
    // 'B2B', 'P2P': letter 2 letter is 'to', not a formula (letters_products); keep water and friends
    if (ascii && parts.size() == 3 && parts[1] == "two" && parts[0].size() == 1 && parts[2].size() == 1) {
        std::string f = parts[0] + "2" + parts[2];
        if (f != "H2O" && f != "N2O" && f != "H2S" && f != "D2O") return;
    }
    std::string words;
    for (size_t p = 0; p < parts.size(); ++p) {
        std::string piece = parts[p];
        // the hydroxyl group at the end is spelled: 'C2H5OH' -> 'C two H five O H'
        if (piece == "OH" && p + 1 == parts.size() && p > 0) piece = "O H";
        if (!words.empty()) words += ' ';
        words += piece;
    }
    if (equivalent) words += " equivalent";
    out.tokens(i, last, words, 61);
}

const rv::WordSet ELEMENT_NAMES{
    "hydrogen", "deuterium", "tritium", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen",
    "fluorine", "neon", "sodium", "magnesium", "aluminum", "aluminium", "silicon", "phosphorus", "sulfur", "sulphur",
    "chlorine", "argon", "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
    "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton",
    "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium", "ruthenium", "rhodium",
    "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium", "iodine", "xenon", "cesium",
    "caesium", "barium", "lanthanum", "cerium", "neodymium", "promethium", "samarium", "europium", "gadolinium",
    "terbium", "holmium", "erbium", "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten",
    "rhenium", "osmium", "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium",
    "astatine", "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
    "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium", "nobelium"};

// Mass number of an isotope: two digits as a number, three digits in pairs ('U-235' -> 'two thirty-five').
std::string mass_number(std::string_view n) {
    if (n.size() == 3) return rv::pairs(static_cast<uint32_t>(std::stoi(std::string(n))));
    return rv::cardinal_digits(n);
}

// Isotopes: 'carbon-14' -> 'carbon fourteen', 'iodine-131' -> 'iodine one thirty-one', 'U-235'.
void isotopes_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i) || stuck_before(doc, i) || !doc.punct(i + 1, '-') || !doc.joined(i + 1) ||
        !doc.number(i + 2) || !doc.joined(i + 2))
        return;
    std::string_view n = doc.str(i + 2);
    if (n.size() > 3 || n[0] == '0') return;
    size_t last = i + 2;
    bool meta = doc.word(last + 1) && doc.joined(last + 1) && doc.str(last + 1) == "m";
    if (meta) ++last;
    if (stuck_after(doc, last)) return;
    if (doc.punct(last + 1, '.') && doc.joined(last + 1) && doc.number(last + 2)) return;
    if (doc.punct(last + 1, '-') && doc.joined(last + 1)) return;  // 'Llama-3-8B'
    std::string_view w = doc.str(i);
    bool name = ELEMENT_NAMES.contains(doc.low(i)) && !(doc[i].casing & rv::UPPER);
    bool symbol = w.size() <= 2 && ELEMENTS.contains(w) && w != "In" && w != "At" && w != "Be" && w != "No" &&
                  w != "Am" && w != "As" && w != "Es" && w != "Md" && w != "Ho" && w != "He" && w != "Pa";
    if (!name && !symbol) return;
    std::string words = std::string(w) + " " + mass_number(n) + (meta ? " M" : "");
    out.tokens(i, last, words, name ? 62 : 61);
}

// Virus names said as words or letters, with the number apart: 'COVID-19' -> 'Covid nineteen',
// 'SARS-CoV-2' -> 'Sars Cov two', 'HIV-1' -> 'HIV one', 'HPV-16' -> 'HPV sixteen'.
const rv::PhraseMap VIRUS_WORDS{{"COVID", "Covid"}, {"Covid", "Covid"}, {"SARS", "Sars"}, {"MERS", "Mers"},
                                {"CoV", "Cov"}, {"nCoV", "N Cov"}};
const rv::WordSet VIRUS_LETTERS{"HIV", "HPV", "HTLV", "HSV", "HHV", "EBV", "CMV", "RSV", "HCV", "HBV", "HAV", "HEV",
                                "SIV", "FIV", "EV", "PCV", "HRV", "HMPV", "PIV", "DENV", "ZIKV", "WNV", "CVB",
                                "HCoV", "HMPXV", "MPXV", "JEV", "VZV"};

void virus_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i) || doc[i].cp > 'Z' || doc.str(i).size() < 2 || doc.str(i).size() > 5) return;
    if ((doc[i].casing & rv::TITLE) && doc.str(i) != "Covid") return;
    if (stuck_before(doc, i)) return;
    std::string_view w = doc.str(i);
    const std::string_view* said = VIRUS_WORDS.find(w);
    bool letters = VIRUS_LETTERS.contains(w);
    if (!said && !letters) return;
    std::string words = said ? std::string(*said) : std::string(w);
    size_t k = i, last = i;
    // further parts joined by hyphens: SARS-CoV-2, MERS-CoV, COVID-19, HIV-1
    while (k + 2 < doc.size() && doc.punct(k + 1, '-') && doc.joined(k + 1) && doc.joined(k + 2)) {
        size_t n = k + 2;
        if (doc.number(n) && doc.str(n).size() <= 2) {
            words += " " + rv::cardinal_digits(doc.str(n));
            last = n;
            break;
        }
        // a letter and a number: 'EV-A71' -> 'EV A seventy-one'
        if (doc.word(n) && doc.str(n).size() == 1 && (doc[n].casing & rv::UPPER) && doc.number(n + 1) &&
            doc.joined(n + 1) && doc.str(n + 1).size() <= 2) {
            words += " " + std::string(doc.str(n)) + " " + rv::cardinal_digits(doc.str(n + 1));
            last = n + 1;
            break;
        }
        const std::string_view* s2 = doc.word(n) ? VIRUS_WORDS.find(doc.str(n)) : nullptr;
        if (!s2) break;
        words += " " + std::string(*s2);
        last = n;
        k = n;
    }
    if (stuck_after(doc, last)) return;
    if (last == i && !said) return;          // 'HIV' alone needs nothing
    if (last == i && words == w) return;     // 'Covid' alone
    out.tokens(i, last, words, 61);
}

// Particle sizes, vitamins and fatty acids: 'PM2.5' -> 'PM two point five', 'PM10', 'vitamin D3' -> 'vitamin D
// three', 'omega-3' -> 'omega three'.
void substances_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i)) return;
    uint32_t c0 = doc[i].cp | 0x20;
    if ((c0 != 'p' && c0 != 'o' && c0 != 'v') || stuck_before(doc, i)) return;
    std::string_view w = doc.str(i);
    if ((w == "PM" || w == "pm") && doc.number(i + 1) && doc.joined(i + 1)) {
        rv::NumberSpan s = rv::parse_number(doc, i + 1);
        if (stuck_after(doc, s.last) || s.whole.size() > 2) return;
        std::string n = rv::read_number(s);  // 'PM0.3' is fine, 'PM05' is not a particle size
        if (n.empty()) return;
        out.tokens(i, s.last, "PM " + n, 75);
        return;
    }
    if ((doc.is(i, "omega") || doc.is(i, "Omega")) && doc.punct(i + 1, '-') && doc.joined(i + 1) &&
        doc.number(i + 2) && doc.joined(i + 2) && doc.str(i + 2).size() <= 2 && !stuck_after(doc, i + 2)) {
        out.tokens(i, i + 2, std::string(w) + " " + rv::cardinal_digits(doc.str(i + 2)), 61);
        return;
    }
    if (doc.is(i, "vitamin") || doc.is(i, "vitamins")) {
        // 'vitamin D3', 'vitamins B6 and B12'
        for (size_t k = i + 1; k + 1 < doc.size() && k <= i + 6; ++k) {
            if (doc.word(k) && doc.str(k).size() == 1 && doc.str(k)[0] >= 'A' && doc.str(k)[0] <= 'K' &&
                doc.number(k + 1) && doc.joined(k + 1) && doc.str(k + 1).size() <= 2 && !stuck_after(doc, k + 1)) {
                out.tokens(k, k + 1, std::string(doc.str(k)) + " " + rv::cardinal_digits(doc.str(k + 1)), 61);
                ++k;
                continue;
            }
            if (!(doc.is(k, "and") || doc.is(k, "or") || doc.punct(k, ',') || doc.punct(k, '/') ||
                  (doc.word(k) && doc.str(k).size() == 1)))
                break;
        }
    }
}

// pH is said as letters: 'pH 7.4' -> 'PH seven point four', 'a pH of 7.5'.
void ph_at(const Doc& doc, Out& out, size_t i) {
    if (doc.str(i) == "pH" && !stuck_after(doc, i) && !stuck_before(doc, i)) out.tokens(i, i, "PH", 50);
}

// Musical keys: 'C# minor' -> 'C sharp minor', 'Bb major' -> 'B flat major', 'F♯', 'E♭'.
const rv::WordSet KEY_CUES{"major", "minor", "maj", "min", "chord", "chords", "key", "scale", "string", "note",
                           "notes", "tuning", "dorian", "lydian", "mixolydian", "phrygian", "aeolian", "locrian",
                           "pentatonic", "blues", "sharp", "flat", "triad", "arpeggio", "harmonic", "melodic"};

// Instruments and pieces: 'the Bb clarinet', 'a rare Bb concerto'.
const rv::WordSet INSTRUMENTS{"bass", "cello", "clarinet", "clarinets", "concerto", "cornet", "etude", "flute",
                              "fugue", "horn", "oboe", "piece", "prelude", "sax", "saxophone", "sonata", "symphony",
                              "trombone", "trumpet", "trumpets", "tuba", "violin"};
// Words that make a text about music.
const rv::WordSet MUSIC_WORDS{"chord", "chords", "chorus", "clarinet", "composer", "concerto", "guitar", "harmony",
                              "key", "major", "melody", "minor", "music", "musical", "note", "notes", "orchestra",
                              "piano", "saxophone", "scale", "song", "sonata", "symphony", "trumpet", "tune",
                              "violin"};

bool music_text(const Doc& doc) {
    for (size_t k = 0; k < doc.size(); ++k)
        if (doc.word(k) && MUSIC_WORDS.contains(doc.low(k))) return true;
    return false;
}

void music_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i) || doc[i].cp < 'A' || doc[i].cp > 'G' || doc.str(i).size() > 2 || stuck_before(doc, i)) return;
    std::string_view w = doc.str(i);
    // 'Bb', 'Eb' as one word
    if (w.size() == 2 && w[0] >= 'A' && w[0] <= 'G' && w[1] == 'b' && !stuck_after(doc, i) &&
        !(doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.word(i + 2))) {
        size_t n = doc.next_word(i);
        bool next = n != Doc::npos && n == i + 1 && (KEY_CUES.contains(doc.low(n)) || INSTRUMENTS.contains(doc.low(n)));
        if (next || music_text(doc)) out.tokens(i, i, std::string(1, w[0]) + " flat", 55);
        return;
    }
    if (w.size() != 1 || w[0] < 'A' || w[0] > 'G' || i + 1 >= doc.size() || !doc.joined(i + 1)) return;
    uint32_t cp = doc[i + 1].cp;
    bool sharp = doc.punct(i + 1, '#') || doc.punct(i + 1, 0x266F);
    bool flat = doc.punct(i + 1, 0x266D);
    if (!sharp && !flat) return;
    if (stuck_after(doc, i + 1)) return;
    size_t n = i + 2;
    bool cue = n < doc.size() && doc.word(n) && !doc.joined(n) && KEY_CUES.contains(doc.low(n));
    size_t p = doc.prev_word(i);
    bool key_before = p != Doc::npos && (doc.is(p, "key") || doc.is(p, "in") || doc.is(p, "of"));
    // 'C#' alone is the language (electronic); the symbols ♯ ♭ are always music
    if (!cue && !key_before && cp == '#') return;
    out.tokens(i, i + 1, std::string(w) + (sharp ? " sharp" : " flat"), 60);
}

// Locants of a chemical name: "1,3-dimethylurea" -> "one three-dimethylurea", "1,3,7-Trimethylxanthine" -> "one
// three seven-Trimethylxanthine", "5,6-ring": numbers of one or two digits joined by commas, then a hyphen and a word.
RV_RULE("units_symbols", locants) {
    size_t n = doc.size();
    for (size_t i = 0; i + 3 < n; ++i) {
        if (!doc.number(i) || !doc.punct(i + 1, ',') || !doc.joined(i + 1) || doc.str(i).size() > 2 ||
            (i > 0 && doc.joined(i) && !doc.punct(i - 1, '(') && !doc.punct(i - 1, '-')))
            continue;
        size_t k = i;
        std::string words = rv::cardinal(std::stoull(std::string(doc.str(i))));
        while (doc.punct(k + 1, ',') && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2) &&
               doc.str(k + 2).size() <= 2) {
            words += " " + rv::cardinal(std::stoull(std::string(doc.str(k + 2))));
            k += 2;
        }
        if (k == i || !doc.punct(k + 1, '-') || !doc.joined(k + 1) || !doc.word(k + 2) || !doc.joined(k + 2) ||
            doc.str(k + 2).size() < 3) {
            i = k;
            continue;
        }
        out.tokens(i, k + 1, words + "-", 65);
        i = k + 1;
    }
}

// One scan for every science form.
RV_RULE("units_symbols", science) {
    for (size_t i = 0; i < doc.size(); ++i) {
        const Token& t = doc[i];
        if (t.kind != Kind::Word || t.cp > 'z') continue;
        if (t.cp >= 'A' && t.cp <= 'Z') {
            chemistry_at(doc, out, i);
            virus_at(doc, out, i);
            if (t.cp <= 'G') music_at(doc, out, i);
        }
        if (doc.punct(i + 1, '-')) isotopes_at(doc, out, i);
        uint32_t c0 = t.cp | 0x20;
        if (c0 == 'p' || c0 == 'o' || c0 == 'v') substances_at(doc, out, i);
        if (t.cp == 'p') ph_at(doc, out, i);
    }
}

}  // namespace
