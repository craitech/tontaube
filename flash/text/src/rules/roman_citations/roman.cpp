// Roman numerals: cardinals after structural and series words (Chapter IV), ordinals after rulers, popes and
// names (Louis XVI, Henry the VIII, John Smith III), sequels (Rocky IV, Lanternfest XCIV), centuries, list markers
// (II., iii., (iv)), dotted references (Othello IV.i.58, Summa I.Q.3.Art.2, Ethics III.V.12), lower-case page
// numbers and the numeral characters (Ⅳ). Letters that only look like a numeral (I, CD, MD, the IV drip,
// Vitamin C, Professor X) get no candidate.
#include "rules/roman_citations/common.h"

namespace {

using namespace rc;

// Words after which a numeral is a cardinal (lower case; "war" and "bowl" need "World" / "Super").
const rv::WordSet CUES{
    "act", "acts", "amendment", "amendments", "annex", "appendices", "appendix", "arena", "article", "articles",
    "book", "books", "bowl", "build", "canto", "cantos", "category", "categories", "chapter", "chapters", "class",
    "clause", "cup", "division", "dynasty", "edition", "episode", "episodes", "exhibit", "exhibits", "experiment",
    "festival", "figure", "figures", "grade", "group", "groups", "introduction", "lecture", "lesson", "lessons", "level",
    "levels", "mark", "mk", "module", "movement", "night", "olympiad", "panel", "paragraph", "part", "parts",
    "patrol", "phase", "phases", "preface", "prototype", "psalm", "realms", "regatta", "round", "saga", "scene",
    "scenes", "schedule", "season", "section", "sections", "sector", "series", "set", "sets", "siege", "sprint",
    "stage", "stages", "subchapter", "subpart", "symphony", "table", "tables", "tier", "title", "titles", "tome",
    "topic", "topics", "type", "types", "unit", "vatican", "volume", "volumes", "war", "wars", "wave", "zone",
    "zones"};
// Abbreviated cues, written with a period: "Vol. IV", "Pt. II".
const rv::WordSet CUE_ABBR{"app", "art", "bk", "bks", "ch", "chap", "chs", "mk", "pt", "pts", "sc", "sec", "vol",
                           "vols"};
// Cues after which a lone L is a numeral (Act L, Volume L); C, D and M alone are always letters.
const rv::WordSet CUES_FOR_L{"act", "book", "canto", "chapter", "part", "preface", "volume", "vol"};
// Cues whose numeral is a page number: lower-case numerals are numbers after these.
const rv::WordSet PAGE_CUES{"page", "pages", "p", "pp"};

// Titles before a ruler's or a person's name.
const rv::WordSet TITLES{
    "abbot", "admiral", "archbishop", "archduchess", "archduke", "baron", "baroness", "bishop", "brother",
    "caliph", "captain", "cardinal", "colonel", "count", "countess", "czar", "czarina", "dame", "don", "dr", "duchess",
    "duke", "earl", "elector", "emir", "emperor", "empress", "father", "general", "gov", "governor", "hon",
    "infanta", "infante", "judge", "kaiser", "khan", "king", "lady", "landgrave", "lord", "madame", "margrave",
    "marquess", "marquis", "master", "mayor", "mister", "monsieur", "mr", "mrs", "ms", "patriarch", "pharaoh", "pope",
    "president", "prince", "princess", "prof", "professor", "queen", "rabbi", "regent", "rev", "reverend", "saint",
    "senator", "shah", "sheikh", "sir", "st", "sultan", "tsar", "tsarina", "viceroy", "viscount"};
// Titles that also mark a ruler when written in lower case ("the emperor Hadrian II").
const rv::WordSet LOWER_TITLES{"archbishop", "bishop", "czar", "duke", "emperor", "empress", "kaiser", "king",
                               "pharaoh", "pope", "prince", "princess", "queen", "sultan", "tsar"};
// Abbreviated titles, written with a period: "Mr. Thompson III".
const rv::WordSet TITLE_ABBR{"capt", "col", "dr", "fr", "gen", "gov", "hon", "lt", "mr", "mrs", "ms", "pres", "prof",
                             "rev", "sen", "sgt", "st"};

// Given and regnal names: a numeral after a name run that contains one of these is an ordinal. Rulers, popes and
// common first names (a generational suffix: "Kevin Brooks III").
const rv::WordSet GIVEN{
    "aaron", "abbas", "abdulaziz", "abdulhamid", "abdullah", "abdulmejid", "adam", "adeodatus", "adolf", "adrian",
    "afonso", "agapetus", "agatho", "agrippa", "ahmed", "akihito", "alan", "alaric", "albert", "albrecht", "alexander",
    "alexandra", "alexei", "alexios", "alexius", "alfonso", "alfred", "alice", "amadeo", "amadeus", "amalia",
    "ambrose", "amenemhat", "amenhotep", "amy", "anacletus", "anastasius", "andrew", "andronicus", "andronikos",
    "angela", "ann", "anna", "anne", "anthony", "antiochus", "arthur", "artaxerxes", "ashurnasirpal", "attalus",
    "augustus", "aurelius", "baldwin", "barbara", "bartholomew", "basil", "bayezid", "beatrice", "beatrix", "bela",
    "benedict", "benjamin", "bernard", "betty", "birger", "blanche", "bohemond", "boleslaw", "boniface", "boris",
    "brian", "bruce", "callistus", "callixtus", "calixtus", "canute", "carl", "carlos", "carol", "casimir",
    "catalina", "catherine", "celestine", "charles", "charlotte", "childeric", "chilperic", "christian", "christina",
    "christine", "christopher", "claudius", "clement", "cleopatra", "clovis", "cnut", "conon", "conrad", "constans",
    "constantine", "constantius", "cornelius", "cosimo", "craig", "cyrus", "dagobert", "damasus", "daniel", "darius",
    "david", "demetrius", "dennis", "diana", "dmitri", "donald", "donus", "dorothea", "dorothy", "douglas", "duarte",
    "edgar", "edmund", "edward", "edwin", "elisabeth", "eleanor", "elias", "elizabeth", "emily", "emma", "emmanuel",
    "enrique", "eric", "erik", "ernest", "ernst", "eugene", "eugenius", "evelyn", "fahd", "faisal", "farouk", "felipe",
    "felix", "ferdinand", "fernando", "filip", "formosus", "frances", "francesco", "francis", "francisco", "frank",
    "franz", "fred", "frederica", "frederick", "frederik", "friedrich", "fuad", "fyodor", "gabriel", "gary",
    "gelasius", "geoffrey", "george", "gerald", "giovanni", "godfrey", "gregory", "guillaume", "gustaf", "gustav",
    "gustavus", "guy", "haakon", "hadrian", "hakon", "hans", "harald", "harold", "harry", "hassan", "heinrich",
    "helen", "henri", "henrique", "henry", "heraclius", "herod", "hilarius", "hormisdas", "honorius", "howard", "hugh",
    "humbert", "hussein", "idris", "inge", "innocent", "isaac", "isabel", "isabella", "ismail", "ivan", "jack",
    "jacob", "jaime", "james", "jan", "jane", "jason", "jean", "jeffrey", "jennifer", "joachim", "joan", "joanna",
    "joao", "johan", "johann", "johanna", "johannes", "john", "johnathan", "jonathan", "jose", "joseph", "joshua",
    "josiah", "juan", "julia", "julian", "juliana", "julius", "justin", "justinian", "kamehameha", "karl",
    "katherine", "kazimierz", "keith", "kenneth", "kevin", "konrad", "ladislaus", "lajos", "lando", "laura",
    "lawrence", "leo", "leonard", "leopold", "letsie", "liberius", "linus", "llywelyn", "lothair", "louis", "louise",
    "lucius", "ludovico", "ludwig", "luis", "magnus", "mahmoud", "mahmud", "malcolm", "manfred", "manuel",
    "marcellus", "marcus", "margaret", "margrethe", "maria", "marie", "marinus", "martha", "martin", "mary",
    "matilda", "matthew", "matthias", "maud", "maurice", "maximilian", "mehmed", "menelik", "michael", "miguel",
    "milan", "mircea", "mithridates", "mohammed", "moshoeshoe", "mswati", "muhammad", "murad", "mustafa",
    "napoleon", "nathan", "nathaniel", "nebuchadnezzar", "nicholas", "nicolas", "nikephoros", "nikolai", "olaf",
    "olav", "olga", "oscar", "osman", "otto", "ottokar", "paschal", "patrick", "paul", "pedro", "pelagius", "pepin",
    "peter", "philip", "philippa", "philippe", "phillip", "pius", "psamtik", "ptolemy", "radu", "rainier", "rama",
    "ramesses", "ramiro", "ramses", "ranavalona", "raymond", "rebecca", "reginald", "richard", "robert", "roger",
    "romanos", "romanus", "ronald", "rudolf", "rudolph", "ruth", "samuel", "sancho", "sarah", "sargon", "scott",
    "sebastian", "seleucus", "selim", "sergius", "sesostris", "seti", "severinus", "shalmaneser", "shapur",
    "sigismund", "sigurd", "silverius", "silvester", "simeon", "simplicius", "sixtus", "sobhuza", "sophia",
    "stanislaus", "stefan", "stephan", "stephen", "steven", "suleiman", "susan", "sverre", "sweyn", "sylvester",
    "symmachus", "tamar", "tancred", "tewodros", "thaddeus", "theobald", "theodora", "theodore", "theodoric",
    "theodosius", "thomas", "thutmose", "tiberius", "tigranes", "timothy", "tupou", "ulrich", "ulrika", "umberto",
    "urban", "urraca", "valdemar", "valentine", "valentinian", "vasili", "vasily", "victor", "victoria",
    "vigilius", "vitalian", "vittorio", "vladimir", "waldemar", "walter", "wenceslas", "wilhelm", "wilhelmina",
    "willem", "william", "xerxes", "yaroslav", "yohannes", "zachary", "zeno", "zephyrinus", "zoe",
    "bolesław", "stanisław", "władysław"};

// Capitalized words that never start a name or a title ("The IV drip", "Washington DC").
const rv::WordSet STOP{
    "a", "about", "after", "all", "also", "an", "and", "any", "as", "at", "before", "both", "but", "by", "during",
    "each", "every", "for", "from", "her", "here", "his", "how", "however", "if", "in", "into", "is", "it", "its",
    "my", "no", "not", "note", "now", "of", "on", "or", "our", "over", "per", "please", "see", "since", "so", "some",
    "that", "the", "their", "then", "there", "these", "this", "those", "to", "today", "under", "until", "via", "was",
    "washington", "we", "what", "when", "where", "which", "while", "who", "why", "with", "your"};

// Works divided into books: "Confessions IX.Ch.13" -> Book Nine, Chapter Thirteen.
const rv::WordSet BOOK_WORKS{"aeneid", "annals", "confessions", "ethics", "georgics", "histories", "iliad",
                             "institutes", "metamorphoses", "nicomachea", "odyssey", "republic"};

// Numerals that a sequel reading must not take: initialisms and sizes (XL, LV, LI, LX, XXX).
const rv::WordSet SEQUEL_LOOKALIKE{"li", "lv", "lx", "xl", "xxx"};
// Numerals that a name must not take ("Charles CD", "John MD").
const rv::WordSet NAME_LOOKALIKE{"cc", "cd", "cl", "cli", "cm", "cmv", "cv", "dc", "di", "dl", "dvd", "mc", "mcv",
                                 "md", "mdx", "mi", "ml", "mm"};
// Lower-case words that are never list markers.
const rv::WordSet LOWER_LOOKALIKE{"cc", "cd", "civ", "cl", "cm", "cv", "dc", "di", "dix", "dl", "li", "liv", "lv",
                                  "mc", "md", "mi", "mix", "ml", "mm", "xl"};
// Words after "IV" that make it an intravenous line, not a numeral.
const rv::WordSet IV_NOUNS{
    "access", "administration", "antibiotics", "bag", "bags", "cannula", "catheter", "dose", "doses", "drip",
    "drips", "drug", "drugs", "fluid", "fluids", "form", "formulation", "hydration", "infusion", "infusions",
    "insertion", "iron", "line", "lines", "medication", "medications", "nurse", "pole", "push", "room", "route",
    "saline", "site", "team", "therapy", "tubing", "use"};
// Nouns after a lower-case cue and a lone "I" that make it a numeral ("type I error"), not the pronoun.
const rv::WordSet I_NOUNS{"cancer", "collagen", "diabetes", "disease", "error", "errors", "hypersensitivity",
                          "interferon", "receptor", "receptors", "study", "studies", "supernova", "trial", "trials"};
// Verbs before a name that make a lone "I" after it the pronoun.
const rv::WordSet OBJECT_VERBS{"ask", "asked", "assured", "dear", "gave", "give", "hello", "hey", "hi", "let", "ok",
                               "okay", "promised", "reminded", "show", "showed", "sorry", "tell", "thank", "thanked",
                               "thanks", "told", "warned", "wish"};
// Words after "Name I" that make "I" the pronoun without an object verb before the name ("Mike I think so").
const rv::WordSet PRONOUN_NEXT{"agree", "am", "appreciate", "believe", "bet", "can", "can't", "cannot", "could",
                               "couldn't", "did", "didn't", "do", "don't", "feel", "guess", "have", "haven't", "hope",
                               "just", "know", "love", "mean", "might", "miss", "must", "need", "owe", "really",
                               "shall", "should", "suppose", "think", "thought", "want", "will", "won't", "would",
                               "wouldn't"};

std::string lower_str(std::string_view s) {
    std::string out(s);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return out;
}

// Index of the cue word before numeral r ("Chapter IV", "Vol. IV", "World War II"), npos if none.
size_t cue_before(const Doc& d, size_t r) {
    if (r == 0 || !d[r].space_before) return npos;
    size_t c = r - 1;
    if (d.punct(c, '.') && c > 0 && d.joined(c) && d.word(c - 1) && CUE_ABBR.contains(d.low(c - 1))) return c - 1;
    if (!d.word(c) || d[c].casing == rv::MIXED) return npos;
    std::string_view w = d.low(c);
    if (!CUES.contains(w)) return npos;
    if (w == "war" || w == "wars") {
        size_t p = d.prev_word(c);
        if (p == npos || !d.is(p, "world")) return npos;
    }
    if (w == "bowl") {
        size_t p = d.prev_word(c);
        if (p == npos || !d.is(p, "super")) return npos;
    }
    if (w == "series") {  // Xbox Series X
        size_t p = d.prev_word(c);
        if (p != npos && d.is(p, "xbox")) return npos;
    }
    return c;
}

// Name run before numeral r: walk back over capitalized words, initials, "Mr." and quoted nicknames.
// 2: a title is in the run (King, Pope, Mr.), 1: a given or regnal name, 0: no name. first: first token of the run.
int name_before(const Doc& d, size_t r, size_t& first, bool* name_word = nullptr) {
    first = npos;
    if (r == 0 || !d[r].space_before) return 0;
    size_t j = r;
    int words = 0, quotes = 0;
    bool given = false, titled = false, named = false;
    while (j > 0 && words < 6) {
        size_t k = j - 1;
        if (d.word(k) && (d[k].casing == rv::TITLE || d[k].casing == rv::MIXED) && d.text[d[k].start] >= 'A' &&
            d.text[d[k].start] <= 'Z') {
            std::string_view w = d.low(k);
            if (STOP.contains(w) || CUES.contains(w)) break;
            if (TITLES.contains(w)) titled = true;
            else named = true, given = given || GIVEN.contains(w);
            ++words;
            first = j = k;
            // a hyphenated name: "Louis-Philippe I", "Hans-Adam II"
            if (k >= 2 && d.punct(k - 1, '-') && d.joined(k) && d.joined(k - 1) && title(d, k - 2)) {
                if (GIVEN.contains(d.low(k - 2))) given = true;
                first = j = k - 2;
            }
            continue;
        }
        if (d.punct(k, '.') && k > 0 && d.joined(k) && d.word(k - 1)) {
            // a title needs a name after it: "Main St. I." is a street and a list marker
            if (TITLE_ABBR.contains(d.low(k - 1)) && words > 0) {
                titled = true;
                first = j = k - 1;
                ++words;
                continue;
            }
            if ((d.is(k - 1, "jr") || d.is(k - 1, "sr")) && words == 0) {  // "Robert Downey Jr. III"
                first = j = k - 1;
                continue;
            }
            if (upper(d, k - 1) && d[k - 1].end - d[k - 1].start == 1 && words > 0) {  // initial: "Arthur P. Finch"
                first = j = k - 1;
                continue;
            }
            break;
        }
        uint32_t c = d[k].cp;
        if (punct_tok(d, k) && (c == '"' || c == 0x201C || c == 0x201D) && words > 0 && k > 0 && title(d, k - 1)) {
            ++quotes;  // Robert "Bob" Johnson
            j = k;
            continue;
        }
        break;
    }
    if (words == 0) return 0;
    if (name_word) *name_word = named;
    if (quotes == 2 && words >= 3) given = true;  // a quoted nickname inside a name
    if (!titled && first > 0) {
        size_t p = first - 1;
        if (d.word(p) && LOWER_TITLES.contains(d.low(p))) titled = true;
    }
    return titled ? 2 : given ? 1 : 0;
}

// A lone "I" after a name is the pronoun after an object verb ("told Henry I would") or before a verb that goes with
// the pronoun ("Mike I think"); elsewhere it is the numeral ("Napoleon I faced", "and Francis I of France").
bool pronoun_after_name(const Doc& d, size_t r, size_t first) {
    size_t b = first > 0 ? first - 1 : npos;
    size_t n = r + 1;
    if (b != npos && d.word(b) && OBJECT_VERBS.contains(d.low(b))) return true;
    if (first != npos && first + 1 < r && OBJECT_VERBS.contains(d.low(first))) return true;  // "Tell Mary I said"
    return n < d.size() && d.word(n) && PRONOUN_NEXT.contains(d.low(n));
}

// A numeral stands alone: not glued to a digit or a letter run ("V3", "3D", "MI6").
bool free_standing(const Doc& d, size_t r) {
    if (r > 0 && d.joined(r) && (d.word(r - 1) || d.number(r - 1))) return false;
    size_t n = r + 1;
    if (d.punct(n, '.') && d.joined(n) && d.number(n + 1) && d.joined(n + 1)) return false;  // "II.3.1"
    return !(n < d.size() && d.joined(n) && (d.word(n) || d.number(n)));
}

bool roman_word(const Doc& d, size_t i, bool want_upper, Roman& out) {
    out = roman(d, i);
    return out.value > 0 && out.upper == want_upper && out.suffix.empty();
}

// A lone letter that is only a numeral after a cue: I, V and X; L after a few cues; never C, D or M.
bool single_ok(const Roman& R, std::string_view cue) {
    if (R.base.size() != 1) return true;
    char c = static_cast<char>(R.base[0] | 0x20);
    if (c == 'i' || c == 'v' || c == 'x') return true;
    return c == 'l' && CUES_FOR_L.contains(cue);
}

// Continue a cardinal after a cue through a range or list: "Chapters I through IV", "Parts II–IV", "Acts I and II".
void continue_list(const Doc& d, Out& out, size_t r, bool want_upper, int score) {
    size_t k = r;
    for (int guard = 0; guard < 8; ++guard) {
        size_t c = k + 1;
        if (c >= d.size()) return;
        bool connector = (d.word(c) && (d.is(c, "and") || d.is(c, "or") || d.is(c, "to") || d.is(c, "through") ||
                                        d.is(c, "thru"))) ||
                         d.punct(c, ',') || (dash_tok(d, c) && d[c].space_before);
        if (!connector) return;
        size_t n = c + 1;
        if (d.punct(c, ',') && d.is(n, "and")) ++n;
        Roman R;
        if (!roman_word(d, n, want_upper, R) || !d[n].space_before || !single_ok(R, "act")) return;
        // "In Part II, I discuss": a lone "I" continues the list only before punctuation or another connector
        if (R.base == "I" && d.word(n + 1) && !d.is(n + 1, "and") && !d.is(n + 1, "or") && !d.is(n + 1, "through"))
            return;
        if (dash_tok(d, c)) out.tokens(c, n, "to " + card_cap(R.value), score);
        else out.tokens(n, n, card_cap(R.value), score);
        k = n;
    }
}

// In a citation that starts with a Roman numeral, the next capitalized cue with a one-digit number keeps the
// capital when the reference goes on after it: "Title IV, Section 3, all" -> Title Four, Section Three, all. A third
// level, a number that ends the sentence or one followed by a verb stays lower case ("Article VI, Section 2
// requires", "Title X, Article Y, Section 4 of").
void continue_citation(const Doc& d, Out& out, size_t k, size_t cue, int score) {
    static const rv::WordSet CHAIN{"act", "article", "book", "canto", "chapter", "clause", "part", "scene",
                                   "section", "title", "volume"};
    if (cue >= 2 && d.punct(cue - 1, ',') && roman(d, cue - 2).value > 0) return;  // already a second level
    size_t c = k + 2;
    if (!d.punct(k + 1, ',') || !title(d, c) || !CHAIN.contains(d.low(c))) return;
    size_t n = c + 1;
    if (!d.number(n) || !d[n].space_before || d.str(n).size() != 1) return;
    if (n + 1 < d.size() && d.joined(n + 1) && (d.word(n + 1) || d.number(n + 2) || d.punct(n + 1, '('))) return;
    bool scene = d.is(c, "scene") && d.is(cue, "act");  // "Act I, Scene 2." keeps the capital everywhere
    if (!(scene || d.punct(n + 1, ',') || d.punct(n + 1, ';') || d.punct(n + 1, ')') || d.is(n + 1, "of"))) return;
    std::string w = plain(d, n);
    if (!w.empty()) out.tokens(n, n, rv::capital(w), score);
}

// Numerals read as cardinals after a cue, with joined ranges ("II–IV" -> Two to Four). Returns true if it proposed.
bool after_cue(const Doc& doc, Out& out, size_t r, const Roman& R) {
    size_t c = cue_before(doc, r);
    bool page = false;
    if (c == npos) {
        // lower-case page numbers: "page xiv", "pp. xi–xxxiv"
        size_t p = r - 1;
        if (!R.upper && doc[r].space_before) {
            if (doc.word(p) && (doc.is(p, "page") || doc.is(p, "pages"))) page = true;
            else if (doc.punct(p, '.') && p > 0 && doc.joined(p) && (doc.is(p - 1, "p") || doc.is(p - 1, "pp")))
                page = true;
        }
        if (!page) return false;
    }
    std::string_view cue = page ? std::string_view("page") : doc.low(c);
    if (!R.upper && !page) {
        // lower-case numerals after a capitalized cue: "Chapter iv", "Part ii"; a lone one only for the scene of an
        // act: "Act II, Scene i"
        bool scene = (cue == "scene" || cue == "sc") && c >= 3 && doc.punct(c - 1, ',') && doc.is(c - 3, "act") &&
                     roman(doc, c - 2).value > 0;
        if (!(title(doc, c) || cue == "sc") || (R.base.size() < 2 && !scene) || LOWER_LOOKALIKE.contains(R.base))
            return false;
    }
    // "Act V, sc. ii" -> Act Five, scene Two
    if (cue == "sc") out.tokens(c, c + 1, doc.text[doc[c].start] == 'S' ? "Scene" : "scene", 72);
    if (cue == "mk") out.tokens(c, doc.punct(c + 1, '.') ? c + 1 : c, doc.text[doc[c].start] == 'M' ? "Mark" : "mark", 72);
    if (R.upper && page) return false;
    if (!single_ok(R, cue)) return false;
    // a lone "I" after a lower-case cue is usually the pronoun ("in part I agree")
    if (R.base == "I" && lower(doc, c)) {
        size_t n = r + 1;
        bool ok = n >= doc.size() || (punct_tok(doc, n) && !doc.punct(n, '\'')) ||
                  (doc.word(n) && I_NOUNS.contains(doc.low(n)));
        if (!ok && doc.word(n) && (doc.is(n, "and") || doc.is(n, "or") || doc.is(n, "to") || doc.is(n, "through"))) {
            Roman Q;
            ok = roman_word(doc, n + 1, true, Q);
        }
        if (!ok) return false;
    }
    std::string words = card_cap(R.value) + std::string(R.suffix);
    size_t last = r;
    Roman Q;
    if (R.suffix.empty() && dash_tok(doc, r + 1) && doc.joined(r + 1) && doc.joined(r + 2) &&
        roman_word(doc, r + 2, R.upper, Q)) {
        words += " to " + card_cap(Q.value);
        last = r + 2;
    } else if (R.suffix.empty() && doc.punct(r + 1, '(') && doc.joined(r + 1)) {
        // "Section V(a)" -> Section Five A, "Article IV(2)" -> Article Four, paragraph two
        size_t sub_last;
        std::string sub = subdivisions(doc, r, sub_last, cue == "section");
        if (!sub.empty()) {
            words += sub;
            last = sub_last;
        }
    }
    out.tokens(r, last, words, 72);
    if (R.suffix.empty()) continue_list(doc, out, last, R.upper, 72);
    if (R.suffix.empty() && R.upper) {
        continue_citation(doc, out, last, c, 72);
        // "Section 7, Article IV": the number of the cue just before keeps a capital too
        if (!page && c >= 3 && doc.punct(c - 1, ',') && doc.number(c - 2) && title(doc, c - 3) &&
            (doc.is(c - 3, "section") || doc.is(c - 3, "scene")) && doc.str(c - 2).size() <= 2 &&
            doc[c - 2].space_before) {
            std::string w = plain(doc, c - 2);
            if (!w.empty()) out.tokens(c - 2, c - 2, rv::capital(w), 72);
        }
    }
    return true;
}

// Numerals after a ruler's, pope's or person's name are ordinals: "Louis XVI" -> Louis the Sixteenth,
// "Pope John Paul II's" -> Pope John Paul the Second's, "Robert Sterling IV" -> Robert Sterling the Fourth.
bool after_name(const Doc& doc, Out& out, size_t r, const Roman& R) {
    if (!R.upper || NAME_LOOKALIKE.contains(lower_str(R.base))) return false;
    size_t first;
    bool named = false;
    int kind = name_before(doc, r, first, &named);
    if (!kind) return false;
    if (R.base.size() == 1) {
        char c = R.base[0];
        if (c != 'I' && c != 'V' && c != 'X') return false;
        if (!named) return false;  // a title alone with a letter: "Professor X", "Agent V"
        if (c == 'X' && doc.is(r - 1, "malcolm")) return false;
        // the pronoun never takes a possessive or ends a clause: "Napoleon I's reforms", "and Francis I."
        bool clause_end = r + 1 >= doc.size() || doc.punct(r + 1, '.') || doc.punct(r + 1, ',') ||
                          doc.punct(r + 1, ';') || doc.punct(r + 1, ')');
        if (c == 'I' && kind == 1 && R.suffix.empty() && !clause_end && pronoun_after_name(doc, r, first))
            return false;
    }
    out.tokens(r, r, the_ord(R.value) + std::string(R.suffix), 72);
    // "Louis XIV or XV" -> Louis the Fourteenth or the Fifteenth
    size_t c = r + 1;
    Roman Q;
    if (R.suffix.empty() && (doc.is(c, "or") || doc.is(c, "and")) && roman_word(doc, c + 1, true, Q) &&
        Q.base.size() > 1 && free_standing(doc, c + 1))
        out.tokens(c + 1, c + 1, the_ord(Q.value), 72);
    return true;
}

// A numeral after a capitalized title word is a sequel or a numbered event: "Rocky IV" -> Rocky Four,
// "Final Fantasy VII" -> Final Fantasy Seven. Only I, V, X and L, at least two letters.
void sequel(const Doc& doc, Out& out, size_t r, const Roman& R) {
    if (!R.upper || R.base.size() < 2) return;
    // C only inside a longer numeral that also has smaller letters ("Lanternfest XCIV"; not CV, CL, CIV, CLI)
    bool has_c = false;
    for (char c : R.base) {
        if (c == 'C') has_c = true;
        else if (c != 'I' && c != 'V' && c != 'X' && c != 'L') return;
    }
    if (has_c && (R.base.size() < 3 || R.base == "CIV" || R.base == "CLI" || R.base == "CLV" || R.base == "CXL"))
        return;
    size_t p = r - 1;
    // "WrestleMania XL": a size or initialism only after a word written in one case
    bool mixed = doc.word(p) && doc[p].casing == rv::MIXED && doc.text[doc[p].start] >= 'A' &&
                 doc.text[doc[p].start] <= 'Z';
    if (SEQUEL_LOOKALIKE.contains(lower_str(R.base)) && !mixed) return;
    if (!(title(doc, p) || mixed) || !doc[r].space_before || STOP.contains(doc.low(p))) return;
    size_t n = r + 1;
    if (R.base == "IV" && doc.word(n) && IV_NOUNS.contains(doc.low(n))) return;
    if (upper(doc, n) && doc[n].space_before) return;  // a run of initialisms
    out.tokens(r, r, card_cap(R.value) + std::string(R.suffix), 55);
    if (R.suffix.empty()) continue_list(doc, out, r, true, 55);  // "Rocky II, III and IV"
}

// Roman numerals read as numbers: a century ("the XIX century" -> the nineteenth century), a cardinal after a cue,
// an ordinal after a name, a cardinal after a title.
RV_RULE("roman_citations", roman_numeral) {
    for (size_t r = 0; r < doc.size(); ++r) {
        Roman R = roman(doc, r);
        if (!R.value || !free_standing(doc, r)) continue;
        size_t n = r + 1;
        if (dash_tok(doc, n) && doc.joined(n) && doc.joined(n + 1)) ++n;
        if (R.upper && R.suffix.empty() && R.value <= 40 && (doc.is(n, "century") || doc.is(n, "centuries"))) {
            out.tokens(r, r, rv::capital(rv::ordinal(static_cast<uint64_t>(R.value))), 74);
            continue;
        }
        if (r == 0) continue;
        // "Mac OS X" -> Mac OS Ten
        if (R.base == "X" && doc[r].space_before && doc.is(r - 1, "os") && upper(doc, r - 1)) {
            out.tokens(r, r, "Ten" + std::string(R.suffix), 72);
            continue;
        }
        // "WW II" -> World War Two
        if (R.upper && (R.value == 1 || R.value == 2) && doc[r].space_before && upper(doc, r - 1) &&
            doc.str(r - 1) == "WW") {
            out.tokens(r - 1, r, "World War " + card_cap(R.value) + std::string(R.suffix), 72);
            continue;
        }
        // "Henry the VIII" -> Henry the Eighth: the article is written already
        if (R.upper && R.base.size() >= 2 && r >= 2 && doc.is(r - 1, "the") && lower(doc, r - 1) &&
            !NAME_LOOKALIKE.contains(lower_str(R.base))) {
            size_t first;
            if (title(doc, r - 2) && name_before(doc, r - 1, first)) {
                out.tokens(r, r, rv::capital(rv::ordinal(static_cast<uint64_t>(R.value))) + std::string(R.suffix), 72);
                continue;
            }
        }
        if (after_cue(doc, out, r, R) || cue_before(doc, r) != npos) continue;
        if (after_name(doc, out, r, R)) continue;
        size_t first;
        if (name_before(doc, r, first)) continue;
        sequel(doc, out, r, R);
    }
}

// Roman numeral characters: "Chapter Ⅳ" -> Chapter Four, "Henry Ⅷ" -> Henry the Eighth; alone they are cardinals.
RV_RULE("roman_citations", roman_numeral_sign) {
    static const int VALUE[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 100, 500, 1000};
    for (size_t r = 0; r < doc.size(); ++r) {
        uint32_t cp = doc[r].cp;
        if (!punct_tok(doc, r) || cp < 0x2160 || cp > 0x217F) continue;
        int v = VALUE[(cp - 0x2160) % 16];
        size_t first;
        bool named = false;
        if (r > 0 && cue_before(doc, r) != npos) out.tokens(r, r, card_cap(v), 72);
        else if (r > 0 && name_before(doc, r, first, &named) && named) out.tokens(r, r, the_ord(v), 72);
        else out.tokens(r, r, card_cap(v), 50);
    }
}

// Token p starts a line, past the markup that may open it: heading marks, a bullet, a quote mark, an emphasis
// opener ("### II.", "- (ii)", "> iii.", "**I.**").
bool line_lead(const Doc& d, size_t p) {
    size_t k = p;
    for (int guard = 0; guard < 8; ++guard) {
        if (line_start(d, k)) return true;
        size_t b = k - 1;
        uint32_t c = d[b].cp;
        bool markup = punct_tok(d, b) && (c == '#' || c == '-' || c == '*' || c == '_' || c == '>' || c == 0x2022 ||
                                          c == 0x2023 || c == 0x25E6 || c == 0x2013);
        if (!markup) return false;
        k = b;
    }
    return false;
}

// List markers: "I) Gather", "ii. secure", "(iv) the budget" -> One, / Two, / Four,.
RV_RULE("roman_citations", roman_list_marker) {
    // values of the markers in the text, upper and lower case apart (bit v for value v up to 63)
    uint64_t seen[2] = {0, 0};
    for (size_t r = 0; r + 2 < doc.size(); ++r) {
        if (!doc.punct(r + 1, '.') && !doc.punct(r + 1, ')')) continue;
        Roman R = roman(doc, r);
        if (R.value && R.value < 64 && R.suffix.empty() && doc.joined(r + 1) &&
            (doc[r + 2].space_before || (doc.punct(r + 2, ',') && doc.joined(r + 2))) &&
            !LOWER_LOOKALIKE.contains(lower_str(R.base)))
            seen[R.upper] |= uint64_t(1) << R.value;
    }
    // a marker has siblings when the one before or after it in the count is in the text too ("i." and "ii.")
    auto in_sequence = [&](const Roman& R) {
        if (R.value >= 64) return seen[0] != 0 || seen[1] != 0;
        uint64_t near = (uint64_t(1) << (R.value - 1)) | (R.value < 63 ? uint64_t(1) << (R.value + 1) : 0);
        return ((seen[0] | seen[1]) & near) != 0;
    };
    static const rv::WordSet MARKER_PREPS{"from", "in", "see", "to", "under"};
    size_t named_end = npos;
    for (size_t r = 0; r + 1 < doc.size(); ++r) {
        if (!doc.punct(r + 1, '.') && !doc.punct(r + 1, ')')) continue;
        Roman R = roman(doc, r);
        if (!R.value || !R.suffix.empty()) continue;
        bool siblings = in_sequence(R);
        size_t t = r + 1;
        bool paren = doc.punct(t, ')'), dot = doc.punct(t, '.');
        if (!doc.joined(t)) continue;
        bool open = paren && r > 0 && doc.punct(r - 1, '(') && doc.joined(r);
        size_t p = open ? r - 1 : r;
        // at the start of a line, after heading marks, a bullet or a bold opener: "### II. Methods", "- (ii) clean",
        // "**I.** Gather"
        bool lead = line_lead(doc, p);
        if (open && p > 0 && !doc[p].space_before && !lead) continue;  // "O(v)", "f(x)"
        if (!open && r > 0 && !doc[r].space_before && !doc[r].newline_before && !lead) continue;
        // what follows: a word starting the item, or a comma when a marker is named in passing ("step, I., involves")
        size_t n = t + 1;
        while (lead && (doc.punct(n, '*') || doc.punct(n, '_')) && doc.joined(n)) ++n;  // "I.**"
        bool inline_ref = doc.punct(n, ',') && doc.joined(n);
        if (!inline_ref && (n >= doc.size() || !doc.word(n) || !doc[n].space_before)) continue;
        // position
        bool strong = lead || doc.punct(p - 1, ':') || doc.punct(p - 1, ';') ||
                      (doc.sentence_start(p) && p > 0 && !doc.punct(p - 1, '('));
        // a marker named in passing: "the points under I) and II) in the appendix" -> under One and Two
        bool named = p > 0 && ((doc.word(p - 1) && MARKER_PREPS.contains(doc.low(p - 1))) ||
                               (p >= 2 && named_end == p - 2 && (doc.is(p - 1, "and") || doc.is(p - 1, "or"))));
        bool weak = p > 0 && (doc.punct(p - 1, ',') || doc.is(p - 1, "and") || doc.is(p - 1, "or") || named);
        // in a list of lower-case markers, a marker may follow any word: "we'll cover i. budgets and ii. staff"; an
        // upper-case one when its item starts in lower case: "including I. equipment failure, II. delays"
        bool after_word = siblings && (dot || open) && R.base.size() <= 4 && doc.word(p - 1) &&
                          (!R.upper || (dot && lower(doc, n)));
        if (!strong && !weak && !after_word) continue;
        std::string low = lower_str(R.base);
        bool single = R.base.size() == 1;
        if (single && low != "i" && low != "v" && low != "x") continue;
        if (LOWER_LOOKALIKE.contains(low) && !(siblings && (lead || doc.punct(p - 1, ':')))) continue;
        if (R.upper) {
            if (dot && single) {
                // "I. M. Pei", "I. A. 1." (an outline, read by the structure rules)
                if (upper(doc, n) && doc[n].end - doc[n].start == 1 && doc.punct(n + 1, '.')) continue;
                if (low != "i" && !siblings && !lead && !doc.punct(p - 1, ':')) continue;
            }
            if (!strong && !siblings && (single || R.value > 39)) continue;
        } else {
            // a lone "i." only at a line start, after a colon, or opening a sentence before a capital ("cut it. i.
            // Next, fold it")
            static const rv::WordSet ABBR{"al", "app", "art", "cf", "ch", "eq", "etc", "fig", "figs", "no", "p", "pp",
                                          "ref", "sec", "vol", "vs"};
            bool after_abbr = doc.punct(p - 1, '.') && p >= 2 && doc.word(p - 2) && ABBR.contains(doc.low(p - 2));
            bool opens = dot && strong && !after_abbr && doc.word(n) && doc[n].casing != rv::LOWER;
            if (single && !siblings && !lead && !doc.punct(p - 1, ':') && !opens) continue;
            if (!strong && !siblings && R.value > 39) continue;
            if (!strong && !siblings && single) continue;
        }
        if (named) named_end = t;
        std::string words = card_cap(R.value);
        if (!inline_ref && !named) words += ',';
        out.tokens(p, t, words, 70);
    }
}

// Dotted references: "Othello IV.i.58" -> Othello, Act Four, Scene One, line fifty-eight;
// "Summa Theologica I.Q.3.Art.2" -> ..., Part One, Question Three, Article Two; "II-II.Q.4" -> Part Two-Two, ...
RV_RULE("roman_citations", roman_dotted_reference) {
    for (size_t r = 0; r + 2 < doc.size(); ++r) {
        if (!(doc.punct(r + 1, '.') || dash_tok(doc, r + 1)) || !doc.joined(r + 1)) continue;
        Roman R;
        if (!roman_word(doc, r, true, R)) continue;
        // standing alone or opening a bracket: "Macbeth (II.ii.60–63)"
        if (r > 0 && !doc[r].space_before && !doc[r].newline_before &&
            !((doc.punct(r - 1, '(') || doc.punct(r - 1, '[')) && (r == 1 || doc[r - 1].space_before)))
            continue;
        std::string first = card_cap(R.value);
        size_t k = r;
        Roman R2;
        if (dash_tok(doc, k + 1) && doc.joined(k + 1) && doc.joined(k + 2) && roman_word(doc, k + 2, true, R2)) {
            first += "-" + card_cap(R2.value);
            k += 2;
        }
        if (!doc.punct(k + 1, '.') || !doc.joined(k + 1) || !doc.joined(k + 2)) continue;
        std::vector<std::string> parts;
        size_t x = k + 2;
        Roman S;
        // the title before: a word, or the word before its closing emphasis mark
        size_t before_title = r > 0 && doc[r].space_before && !doc[r].newline_before ? r - 1 : npos;
        if (before_title != npos && (doc.punct(before_title, '*') || doc.punct(before_title, '_')) && before_title > 0)
            --before_title;
        static const rv::WordSet NOT_WORK{"build", "class", "figure", "firmware", "grade", "item", "level", "model",
                                          "note", "option", "patch", "phase", "release", "revision", "rule", "stage",
                                          "step", "table", "type", "update", "version"};
        bool titled = before_title != npos && doc.word(before_title) &&
                      (title(doc, before_title) || doc[before_title].casing == rv::MIXED) &&
                      !NOT_WORK.contains(doc.low(before_title)) && !STOP.contains(doc.low(before_title));
        Roman S2;
        bool plain_parts = (roman_word(doc, x, true, S2) || (doc.number(x) && doc.str(x).size() <= 3)) &&
                           doc.punct(x + 1, '.') && doc.joined(x + 1) && doc.number(x + 2) && doc.joined(x + 2) &&
                           doc.str(x + 2).size() <= 3 && !(doc.punct(x + 3, '.') && doc.joined(x + 3) &&
                                                           !punct_tok(doc, x + 4) && doc.joined(x + 4));
        if (plain_parts && titled && R2.value == 0) {
            // "Nicomachean Ethics III.V.12" -> Book Three, Chapter Five, section twelve; "De Anima III.2.4" -> Part
            // Three, Section Two, Number Four
            bool book = BOOK_WORKS.contains(doc.low(before_title));
            int second = doc.number(x) ? static_cast<int>(num(doc, x)) : S2.value;
            std::string third = plain(doc, x + 2);
            if (third.empty()) continue;
            parts.push_back(std::string(book ? "Book " : "Part ") + first);
            parts.push_back(std::string(book ? "Chapter " : "Section ") + card_cap(second));
            parts.push_back(std::string(book ? "section " : "Number ") + (book ? third : rv::capital(third)));
            k = x + 2;
        } else if (roman_word(doc, x, false, S) && !(doc.punct(x + 1, '.') && doc.word(x + 2) && doc.joined(x + 2))) {
            // act, scene, line
            parts.push_back("Act " + first);
            parts.push_back("Scene " + card_cap(S.value));
            k = x;
            if (doc.punct(k + 1, '.') && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2)) {
                std::string line = plain(doc, k + 2);
                if (line.empty()) continue;
                k += 2;
                if (dash_tok(doc, k + 1) && doc.joined(k + 1) && doc.number(k + 2) && doc.joined(k + 2)) {
                    parts.push_back("lines " + line + " to " + plain(doc, k + 2));
                    k += 2;
                } else {
                    parts.push_back("line " + line);
                }
            }
        } else {
            // Part, then labelled parts: Q (Question), Art (Article), Ch (Chapter), Sec (Section), Para (Paragraph)
            parts.push_back("Part " + first);
            bool labelled = false;
            while (doc.punct(k + 1, '.') && doc.joined(k + 1) && doc.joined(k + 2)) {
                size_t y = k + 2;
                Roman T;
                if (roman_word(doc, y, true, T) && !labelled) {  // II.II.Q.77: a second part
                    parts.push_back("Part " + card_cap(T.value));
                    k = y;
                    continue;
                }
                if (!doc.word(y)) break;
                std::string_view w = doc.low(y);
                const char* label = w == "q" ? "Question" : w == "art" ? "Article" : w == "ch" ? "Chapter"
                                  : w == "sec" ? "Section" : w == "para" ? "Paragraph" : nullptr;
                if (!label) break;
                size_t z = y + 1;
                if (doc.punct(z, '.') && doc.joined(z)) ++z;
                if (!doc.number(z) || !doc.joined(z)) break;
                std::string n = plain(doc, z);
                parts.push_back(std::string(label) + " " + (w == "para" ? n : rv::capital(n)));  // "Paragraph five"
                labelled = true;
                k = z;
            }
            if (!labelled) continue;
        }
        // after a cue ("Chapter III.Sec.2") the first part has its name already; works in books say "Book"
        size_t before = r > 0 ? r - 1 : npos;
        bool cue = before != npos && doc.word(before) && CUES.contains(doc.low(before));
        if (cue) parts[0] = parts[0].substr(parts[0].find(' ') + 1);
        else if (parts[0].compare(0, 5, "Part ") == 0 && before != npos) {
            size_t t = (doc.punct(before, '*') || doc.punct(before, '_')) && before > 0 ? before - 1 : before;
            if (doc.word(t) && BOOK_WORKS.contains(doc.low(t))) parts[0] = "Book" + parts[0].substr(4);
        }
        // a comma after a title, none after a cue or a comma
        uint32_t start = doc[r].start;
        std::string words;
        if (!cue && r > 0 && doc[r].space_before && !doc[r].newline_before) {
            size_t p = r - 1;
            uint32_t c = doc[p].cp;
            // after a closing quote no comma: "Oedipus Rex" III.iv.12
            bool after_title = doc.word(p) || doc.number(p) || c == '*' || c == '_';
            if (after_title) {
                start = doc[p].end;
                words = ", ";
            }
        }
        for (size_t q = 0; q < parts.size(); ++q) words += (q ? ", " : "") + parts[q];
        out.add(start, doc[k].end, words, 80);
        r = k;
    }
}

}  // namespace
