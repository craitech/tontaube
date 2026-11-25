// Rules of the module "abbreviations": words written with periods and written-only short forms. Titles, Latin
// forms, address words, company suffixes, reference words before numbers, measures and report shorthand, dotted
// initialisms, state names, compass directions in addresses, numeronyms and business shorthand. The rules around
// numbers (ordinal streets, freeways, circa, cups, life dates) live in abbreviations/extra.cpp; chat shorthand is
// the structure module's.
// See README.md for the conventions.
//
// One period both ends an abbreviation and a sentence: when the sentence ends there, the span stops before the
// period and the period stays in the text; inside a sentence the period is part of the span and goes.
#include <algorithm>
#include <vector>

#include "core/rule.h"
#include "rules/abbreviations/common.h"

namespace {

using abbr::clean_end;
using abbr::clean_start;
using abbr::closing_mark;
using abbr::lower_first;
using abbr::match_case;
using abbr::upper_first;
using rv::Doc;
using rv::Out;

// ------------------------------------------------------------------------------------------------ the word table

enum Flag : uint16_t {
    CAP = 1,      // only written with a capital first letter (Mr., Inc., Ave.)
    BARE = 2,     // also written without the final period
    TITLE = 4,    // before a name: the period never ends the sentence
    PREFIX = 8,   // before what it qualifies (vs., approx.): the period never ends the sentence
    SUFFIX = 16,  // after what it qualifies (Inc., Jr., etc.): a capitalized word after the period starts a sentence
    REF = 32,     // reference word: the number after it is read with it (Fig. 3, Sec. 3.2.1, Figs. 2-4)
    NUMONLY = 64, // only before a number (p. 7, l. 12, Fl. 3)
    CAPNUM = 128, // the number after it gets a capital (Volume Three, Part Two)
    ADDR = 256,   // address cue: a number after it is read in pairs (Suite three oh five, Highway one oh one)
    CTX = 512,    // a context handler decides the reading
    FIXED = 1024, // spoken keeps its own capitals whatever the written case
};

struct Entry {
    std::string_view key;     // lower case, inner periods kept, final period dropped: "e.g", "mr", "w.r.t"
    std::string_view spoken;  // the first letter follows the written case unless FIXED or CAP
    uint16_t flags;
};

// clang-format off
const Entry TABLE[] = {
    // titles before a name
    {"mr", "Mister", CAP | TITLE | BARE}, {"mrs", "Missus", CAP | TITLE | BARE}, {"ms", "Miss", CAP | TITLE | BARE},
    {"mx", "Mix", CAP | TITLE | BARE}, {"messrs", "Messieurs", CAP | TITLE}, {"mmes", "Mesdames", CAP | TITLE},
    {"dr", "Doctor", CAP | TITLE | BARE | CTX}, {"prof", "Professor", CAP | TITLE | BARE},
    {"rev", "Reverend", CAP | TITLE | BARE | CTX}, {"revd", "Reverend", CAP | TITLE | BARE},
    {"hon", "Honorable", CAP | TITLE}, {"gen", "General", CAP | TITLE | CTX},
    {"lt", "Lieutenant", CAP | TITLE | BARE}, {"col", "Colonel", CAP | TITLE | BARE},
    {"capt", "Captain", CAP | TITLE | BARE}, {"sgt", "Sergeant", CAP | TITLE | BARE},
    {"cpl", "Corporal", CAP | TITLE | BARE}, {"pvt", "Private", CAP | TITLE | BARE},
    {"maj", "Major", CAP | TITLE}, {"adm", "Admiral", CAP | TITLE}, {"amb", "Ambassador", CAP | TITLE},
    {"gov", "Governor", CAP | TITLE}, {"sen", "Senator", CAP | TITLE}, {"rep", "Representative", CAP | TITLE | CTX},
    {"pres", "President", CAP | TITLE}, {"supt", "Superintendent", CAP | TITLE}, {"insp", "Inspector", CAP | TITLE},
    {"det", "Detective", CAP | TITLE}, {"cmdr", "Commander", CAP | TITLE | BARE}, {"brig", "Brigadier", CAP | TITLE},
    {"msgr", "Monsignor", CAP | TITLE}, {"atty", "Attorney", CAP | TITLE}, {"st", "Saint", CAP | TITLE | BARE | CTX},
    {"drs", "Doctors", CAP | TITLE}, {"profs", "Professors", CAP | TITLE}, {"sts", "Saints", CAP | TITLE},
    {"revs", "Reverends", CAP | TITLE},
    {"mt", "Mount", CAP | TITLE | BARE | CTX}, {"ft", "Fort", CAP | TITLE | CTX}, {"fr", "Father", CAP | TITLE | CTX},
    {"asst", "assistant", PREFIX}, {"supv", "supervisor", PREFIX}, {"dir", "Director", CAP | TITLE},
    {"emer", "Emeritus", CAP | TITLE}, {"ops", "operations", SUFFIX | CTX}, {"mktg", "marketing", SUFFIX | CTX},
    {"engr", "engineer", SUFFIX | CTX},
    // after a name
    {"jr", "Junior", CAP | SUFFIX | BARE | CTX}, {"sr", "Senior", CAP | SUFFIX | BARE | CTX}, {"esq", "Esquire", CAP | SUFFIX},
    {"bros", "Brothers", CAP | SUFFIX},
    // companies and organisations
    {"inc", "Incorporated", SUFFIX | BARE | CTX}, {"corp", "Corporation", CAP | SUFFIX | BARE | CTX},
    {"ltd", "Limited", CAP | SUFFIX | BARE | CTX},
    {"co", "County", CAP | SUFFIX | CTX}, {"intl", "international", SUFFIX | CTX}, {"natl", "national", SUFFIX | CTX},
    {"univ", "university", SUFFIX | CTX}, {"dept", "department", SUFFIX | CTX}, {"depts", "departments", SUFFIX | CTX},
    {"govt", "government", SUFFIX | CTX}, {"gov't", "government", SUFFIX | BARE},
    {"gov\xE2\x80\x99t", "government", SUFFIX | BARE}, {"assn", "association", SUFFIX | CTX},
    {"assoc", "association", SUFFIX | CTX},
    {"mfg", "manufacturing", SUFFIX | CTX}, {"grp", "group", SUFFIX | CTX}, {"hldgs", "holdings", SUFFIX | CTX},
    {"inst", "Institute", CAP | SUFFIX | CTX}, {"soc", "Society", CAP | SUFFIX | CTX}, {"sys", "Systems", CAP | SUFFIX | CTX},
    {"coop", "cooperative", SUFFIX | CTX}, {"mgmt", "management", SUFFIX | CTX}, {"admin", "Administration", CAP | SUFFIX | CTX},
    {"dist", "District", CAP | SUFFIX | CTX}, {"div", "Division", CAP | SUFFIX | CTX}, {"svcs", "services", SUFFIX | CTX},
    {"ctr", "Center", CAP | SUFFIX | CTX}, {"fdn", "Foundation", CAP | SUFFIX | CTX},
    {"ent", "Enterprises", CAP | SUFFIX | CTX}, {"ind", "Industries", CAP | SUFFIX | CTX},
    {"acad", "Academy", CAP | SUFFIX | CTX}, {"hosp", "Hospital", CAP | SUFFIX | CTX},
    {"cmte", "Committee", CAP | SUFFIX | CTX}, {"commn", "Commission", CAP | SUFFIX | CTX},
    {"agcy", "Agency", CAP | SUFFIX | CTX},
    // addresses
    {"ave", "Avenue", CAP | SUFFIX | BARE}, {"blvd", "Boulevard", CAP | SUFFIX | BARE}, {"rd", "Road", CAP | SUFFIX | BARE},
    {"ln", "Lane", CAP | SUFFIX | BARE}, {"ct", "Court", CAP | SUFFIX | BARE}, {"pl", "Place", CAP | SUFFIX | BARE},
    {"hwy", "Highway", CAP | SUFFIX | ADDR | BARE}, {"pkwy", "Parkway", CAP | SUFFIX | ADDR | BARE},
    {"expy", "Expressway", CAP | SUFFIX | ADDR}, {"fwy", "Freeway", CAP | SUFFIX | ADDR},
    {"rte", "Route", CAP | SUFFIX | ADDR}, {"sq", "Square", CAP | SUFFIX | BARE | CTX}, {"ter", "Terrace", CAP | SUFFIX},
    {"cir", "Circle", CAP | SUFFIX}, {"hts", "Heights", CAP | SUFFIX}, {"jct", "Junction", CAP | SUFFIX},
    {"plz", "Plaza", CAP | SUFFIX}, {"bldg", "Building", CAP | SUFFIX | ADDR}, {"ste", "Suite", CAP | SUFFIX | ADDR | CTX},
    {"apt", "Apartment", CAP | SUFFIX | ADDR}, {"rm", "Room", CAP | SUFFIX | ADDR}, {"fl", "Floor", CAP | NUMONLY | ADDR},
    // Latin and report prose
    {"e.g", "for example", PREFIX | BARE}, {"i.e", "that is", PREFIX | BARE}, {"etc", "et cetera", SUFFIX | BARE},
    {"vs", "versus", PREFIX | BARE}, {"v", "versus", PREFIX | CTX}, {"cf", "compare", PREFIX}, {"viz", "namely", PREFIX},
    {"lit", "literally", PREFIX | CTX},
    {"approx", "approximately", PREFIX | BARE}, {"incl", "including", PREFIX}, {"excl", "excluding", PREFIX},
    {"w.r.t", "with respect to", PREFIX}, {"esp", "especially", PREFIX}, {"avg", "average", PREFIX},
    {"est", "estimated", PREFIX | CTX}, {"max", "maximum", PREFIX | CTX}, {"min", "minimum", PREFIX | CTX},
    {"cum", "cumulative", PREFIX}, {"tot", "total", PREFIX}, {"prev", "previous", PREFIX}, {"ea", "each", SUFFIX},
    {"rel", "relative", PREFIX}, {"ann", "annual", PREFIX}, {"misc", "miscellaneous", SUFFIX},
    {"resp", "respectively", SUFFIX}, {"temp", "temperature", PREFIX | CTX}, {"pop", "population", PREFIX | CTX},
    {"doz", "dozen", SUFFIX}, {"pcs", "pieces", SUFFIX}, {"pc", "piece", SUFFIX | CTX}, {"pkg", "package", SUFFIX},
    {"pkgs", "packages", SUFFIX}, {"wt", "weight", PREFIX | CTX}, {"ht", "height", PREFIX | CTX},
    {"dims", "dimensions", PREFIX | CTX}, {"mfr", "manufacturer", PREFIX}, {"mfrs", "manufacturers", PREFIX},
    {"encl", "enclosure", SUFFIX}, {"mkt", "market", SUFFIX}, {"doc", "document", REF | NUMONLY}, {"appt", "appointment", SUFFIX | BARE}, {"appts", "appointments", SUFFIX | BARE},
    {"reqd", "required", SUFFIX},
    // business shorthand, with or without the period
    {"mtg", "meeting", SUFFIX | BARE}, {"mtgs", "meetings", SUFFIX | BARE}, {"mgr", "manager", SUFFIX | BARE},
    {"mgrs", "managers", SUFFIX | BARE}, {"acct", "account", SUFFIX | BARE}, {"accts", "accounts", SUFFIX | BARE},
    {"msg", "message", SUFFIX | BARE}, {"msgs", "messages", SUFFIX | BARE}, {"qtr", "quarter", SUFFIX | BARE},
    {"qtrs", "quarters", SUFFIX | BARE}, {"amt", "amount", SUFFIX | BARE},
    // box-score columns after a number: "34 pts., 11 reb., 8 ast."
    {"pts", "points", SUFFIX | BARE | CTX}, {"reb", "rebounds", SUFFIX | CTX}, {"ast", "assists", SUFFIX | CTX},
    {"stl", "steals", SUFFIX | CTX}, {"blk", "blocks", SUFFIX | CTX},
    {"a.k.a", "AKA", PREFIX | FIXED}, {"ph.d", "PHD", SUFFIX | FIXED}, {"m.sc", "MSC", SUFFIX | FIXED},
    {"b.sc", "BSC", SUFFIX | FIXED}, {"ll.b", "LLB", SUFFIX | FIXED}, {"ll.m", "LLM", SUFFIX | FIXED},
    {"no", "number", SUFFIX | CTX}, {"nos", "numbers", SUFFIX},
    {"intro", "Introduction", REF | CAP | CAPNUM}, {"introd", "Introduction", REF | CAP | CAPNUM},
    {"pref", "Preface", REF | CAP | CAPNUM}, {"qty", "quantity", SUFFIX}, {"std", "standard", PREFIX},
    {"ed", "Editor", CAP | SUFFIX | CTX}, {"eds", "Editors", CAP | SUFFIX},
    // reference words before numbers
    {"fig", "figure", REF}, {"figs", "figures", REF}, {"eq", "equation", REF}, {"eqs", "equations", REF},
    {"eqn", "equation", REF}, {"eqns", "equations", REF}, {"mk", "Mark", REF | CAP | CAPNUM | NUMONLY},
    {"tab", "table", REF | NUMONLY}, {"tbl", "table", REF | NUMONLY}, {"sec", "section", REF},
    {"secs", "sections", REF | NUMONLY}, {"sect", "section", REF | NUMONLY}, {"ch", "chapter", REF},
    {"chap", "chapter", REF}, {"chs", "chapters", REF | NUMONLY}, {"vol", "volume", REF | CAPNUM},
    {"vols", "volumes", REF}, {"alg", "algorithm", REF | CAP}, {"thm", "theorem", REF | CAP},
    {"app", "appendix", REF | CAP}, {"appx", "appendix", REF | CAP}, {"pt", "part", REF | CAP | CAPNUM | NUMONLY},
    {"def", "definition", REF | CAP | NUMONLY}, {"lem", "lemma", REF | CAP | NUMONLY},
    {"prop", "proposition", REF | CAP | NUMONLY}, {"cor", "corollary", REF | CAP | NUMONLY},
    {"ex", "example", REF | NUMONLY}, {"para", "paragraph", REF | NUMONLY}, {"paras", "paragraphs", REF | NUMONLY},
    {"fn", "footnote", REF | NUMONLY},
    {"l", "line", REF | NUMONLY}, {"ll", "lines", REF | NUMONLY}, {"bk", "book", REF | CAP | NUMONLY},
    {"art", "Article", CAP | CTX}, {"arts", "Articles", CAP | CTX}, {"ref", "Reference", CAP | CTX},
    {"refs", "References", CAP | CTX}, {"op", "Opus", CAP | CTX},
};
// clang-format on

const std::vector<const Entry*>& sorted_table() {
    static const std::vector<const Entry*> v = [] {
        std::vector<const Entry*> s;
        for (const Entry& e : TABLE) s.push_back(&e);
        std::sort(s.begin(), s.end(), [](const Entry* a, const Entry* b) { return a->key < b->key; });
        return s;
    }();
    return v;
}

const Entry* find_entry(std::string_view key) {
    const auto& t = sorted_table();
    auto it = std::lower_bound(t.begin(), t.end(), key, [](const Entry* e, std::string_view k) { return e->key < k; });
    return it != t.end() && (*it)->key == key ? *it : nullptr;
}

// US state abbreviations, read in full only after "City, " ("Springfield, Ill."): alone many are words.
const rv::PhraseMap STATES{
    {"ala", "Alabama"},     {"ariz", "Arizona"},       {"ark", "Arkansas"},      {"calif", "California"},
    {"colo", "Colorado"},   {"conn", "Connecticut"},   {"del", "Delaware"},      {"fla", "Florida"},
    {"ga", "Georgia"},      {"ill", "Illinois"},       {"ind", "Indiana"},       {"kan", "Kansas"},
    {"ky", "Kentucky"},     {"la", "Louisiana"},       {"md", "Maryland"},       {"mass", "Massachusetts"},
    {"mich", "Michigan"},   {"minn", "Minnesota"},     {"miss", "Mississippi"},  {"mo", "Missouri"},
    {"mont", "Montana"},    {"n.c", "North Carolina"}, {"n.d", "North Dakota"},  {"n.h", "New Hampshire"},
    {"n.j", "New Jersey"},  {"n.m", "New Mexico"},     {"n.y", "New York"},      {"neb", "Nebraska"},
    {"nev", "Nevada"},      {"okla", "Oklahoma"},      {"ore", "Oregon"},        {"pa", "Pennsylvania"},
    {"r.i", "Rhode Island"}, {"s.c", "South Carolina"}, {"s.d", "South Dakota"}, {"tenn", "Tennessee"},
    {"w.va", "West Virginia"}, {"penn", "Pennsylvania"}, {"me", "Maine"},
    {"tex", "Texas"},       {"va", "Virginia"},        {"vt", "Vermont"},        {"wash", "Washington"},
    {"wis", "Wisconsin"},   {"wyo", "Wyoming"},
};

const rv::WordSet US_OFFICES{"amb", "atty", "gen", "gov", "pres", "rep", "sen"};

// State abbreviations that are no English word, read in full after a place preposition too.
const rv::WordSet SAFE_STATES{"ariz", "calif", "colo", "conn", "fla", "mich", "minn", "neb", "nev", "okla", "tenn",
                              "wis", "wyo"};
const rv::WordSet STATE_PREPS{"across", "around", "at", "central", "eastern", "from", "in", "near", "northern", "of",
                              "outside", "southern", "throughout", "to", "western"};

// Two-word forms spoken as one: "op. cit.", "pro tem.", "Rev. Ed.", "Ed. Note".
struct Pair {
    std::string_view first, second, spoken;
    bool first_period;   // the first word has its own period ("op." but "pro")
    bool second_period;  // the second word ends with a period ("cit." but "Note")
    bool fixed;          // spoken keeps its capitals; written only with a capital ("Rev. Ed.")
};
const Pair PAIRS[] = {
    {"op", "cit", "opus citatum", true, true, false},     {"loc", "cit", "loco citato", true, true, false},
    {"et", "seq", "et sequentes", false, true, false},    {"pro", "tem", "pro tempore", false, true, false},
    {"ad", "lib", "ad libitum", false, true, false},      {"rev", "ed", "Revised Edition", true, true, true},
    {"rt", "hon", "Right Honorable", true, true, true},   {"ed", "note", "Editor's Note", true, false, true},
    {"std", "dev", "standard deviation", true, true, false},
};

// ------------------------------------------------------------------------------------------------ context words

// Capitalized words that start sentences and are not names: after "Elm St." they show that the sentence ended.
const rv::WordSet STARTERS{
    "a", "about", "according", "after", "again", "all", "also", "although", "an", "and", "another", "any", "are",
    "as", "at", "because", "before", "both", "but", "by", "call", "can", "check", "come", "continue", "could",
    "did", "do", "does", "don't", "during", "each", "even", "every", "finally", "first", "for", "from", "go",
    "had", "has", "have", "he", "head", "her", "here", "his", "how", "however", "i", "i'll", "i'm", "if", "in",
    "is", "it", "it's", "its", "just", "last", "let", "let's", "many", "may", "meanwhile", "most", "my", "next",
    "no", "not", "now", "of", "on", "once", "one", "only", "or", "our", "please", "she", "should", "since", "so",
    "some", "still", "take", "thanks", "that", "the", "their", "then", "there", "these", "they", "this", "those",
    "to", "today", "turn", "two", "until", "was", "we", "we'll", "we're", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "yes", "yet", "you", "you'll", "your"};

// Capitalized words before "St." / "Dr." that are not street names.
const rv::WordSet NOT_STREET{
    "about", "after", "and", "ask", "asked", "at", "attn", "before", "both", "by", "call", "cc", "congrats",
    "contact", "dear", "for", "from", "good", "hello", "hey", "hi", "in", "like", "meet", "met", "near", "of",
    "on", "or", "please", "said", "says", "see", "thank", "thanks", "the", "to", "visit", "visited", "visiting",
    "welcome", "when", "while", "with"};

// Capitalized words before "St." that put a saint's name after it: "Lake St. Clair", "Hotel St. Regis".
const rv::WordSet SAINT_BEFORE{"abbey", "academy", "basilica", "bay", "calle", "cape", "cathedral", "chapel",
                               "church", "college", "fort", "hospital", "hotel", "isle", "lake", "mount", "parish",
                               "port", "rue", "sault", "school", "via"};

// Words that make "Co." a company rather than a county.
const rv::WordSet BUSINESS{
    "airlines", "baking", "bank", "brewing", "chemical", "coal", "coffee", "construction", "dairy", "electric",
    "energy", "engineering", "freight", "furniture", "gas", "glass", "gold", "hardware", "holding", "insurance",
    "investment", "iron", "logistics", "lumber", "manufacturing", "mfg", "mills", "mining", "motor", "motors",
    "mutual", "oil", "packing", "paper", "power", "products", "publishing", "railroad", "railway", "robotics",
    "rubber", "shipping", "silver", "software", "steel", "sugar", "supply", "tea", "technology", "telephone",
    "textile", "tobacco", "tool", "toy", "trading", "transport", "trust", "water"};

// Words before "Ed." that make it "Edition".
const rv::WordSet EDITION_BEFORE{
    "abridged", "anniversary", "current", "deluxe", "digital", "expanded", "first", "fourth", "illustrated",
    "international", "last", "latest", "limited", "new", "next", "original", "previous", "print", "revised",
    "second", "special", "student", "third", "this", "unabridged", "updated"};

const rv::WordSet POSSESSIVES{"her", "his", "my", "our", "their", "your"};
const rv::WordSet DETERMINERS{"a", "an", "any", "each", "every", "her", "his", "its", "my", "our", "that", "the",
                              "their", "this", "your"};

// Words before "no." that make it "number".
const rv::WordSet NUMBER_BEFORE{
    "a", "account", "an", "any", "case", "catalog", "catalogue", "each", "every", "file", "her", "his", "invoice",
    "item", "its", "model", "my", "order", "our", "part", "phone", "policy", "product", "ref", "reference",
    "registration", "serial", "that", "the", "their", "this", "total", "tracking", "true", "your"};

// Street words that may follow a compass point (N. Main St.) or precede one (Main St. NE).
const rv::WordSet STREET_WORDS{
    "ave", "avenue", "blvd", "boulevard", "cir", "circle", "court", "ct", "dr", "drive", "expressway", "expy",
    "freeway", "fwy", "highway", "hwy", "lane", "ln", "parkway", "pkwy", "pl", "place", "rd", "road", "sq",
    "square", "st", "street", "ter", "terrace", "way"};

const rv::WordSet UNIT_AFTER_SQ{"ft", "feet", "foot", "in", "inch", "inches", "km", "m", "mi", "miles", "yd", "yds"};

const rv::WordSet ORDINAL_SUFFIX{"st", "nd", "rd", "th"};

const rv::WordSet ORG_WORDS{"intl", "natl", "univ", "dept", "depts", "govt", "assn", "assoc", "mfg", "grp",
                            "hldgs", "inst", "soc", "sys", "coop", "mgmt", "admin", "dist", "div", "svcs",
                            "ctr", "fdn", "ent", "ind", "acad", "hosp", "cmte", "commn", "agcy"};

// Job and rank words: "Sr. Vice President", "Jr. Analyst", "Assoc. Prof." qualify them rather than a name.
const rv::WordSet JOB_WORDS{
    "accountant", "administrator", "advisor", "analyst", "architect", "associate", "attorney", "auditor", "buyer",
    "chemist", "clerk", "consultant", "coordinator", "counsel", "dean", "designer", "developer", "director", "editor",
    "engineer", "executive", "fellow", "investigator", "lecturer", "librarian", "manager", "member", "officer",
    "partner", "planner", "producer", "prof", "professor", "programmer", "researcher", "scientist", "specialist",
    "staff", "strategist", "technician", "vice", "writer", "mgr", "dir", "engr", "asst", "assoc"};

// Locative words before a county name: "in Cook Co.", "through Lincoln Co."
const rv::WordSet COUNTY_BEFORE{"across", "around", "at", "by", "for", "from", "in", "inside", "into", "near", "of",
                                "outside", "rural", "through", "throughout", "to", "western", "eastern", "northern",
                                "southern", "within"};
// Words before a single-word name that make "Co." the company: "is Carrow Co.", "called Brix Co."
const rv::WordSet COMPANY_BEFORE{"acquired", "bought", "called", "client", "competitor", "contractor", "customer",
                                 "distributor", "employer", "firm", "hired", "is", "joined", "maker", "manufacturer",
                                 "named", "partner", "reseller", "retailer", "rival", "subsidiary", "supplier",
                                 "vendor", "was", "wholesaler"};

// Words before "temp." that make it a temperature: "body temp. 38°C".
const rv::WordSet TEMP_BEFORE{"air", "average", "body", "core", "oven", "room", "skin", "surface", "water"};

// Lower-case words before "Ed.", "Pref.", "Intro." that make them common nouns: "the new edition".
const rv::WordSet COMMON_NOUN_BEFORE{"brief", "new", "short", "special"};

// Words before a street name or a compass point: "on Maple Dr.", "onto W. Elm Street".
const rv::WordSet DIRECTION_PREPS{"along", "at", "down", "for", "from", "near", "of", "off", "on", "onto",
                                  "take", "to", "toward", "towards", "up", "via"};
// Distance units before a compass point: "0.5 mi. W. of the mill", "two blocks N."
const rv::WordSet DISTANCE_UNITS{"blocks", "feet", "ft", "km", "kilometers", "m", "meters", "mi", "mile", "miles",
                                 "yards", "yd"};

bool is_title_entry(const Doc& d, size_t k);
bool continues_name(const Doc& d, size_t k);

bool direction_word(std::string_view low) {
    return low == "n" || low == "s" || low == "e" || low == "w" || low == "ne" || low == "nw" || low == "se" ||
           low == "sw";
}

// Token k (after an abbreviation) is a capitalized word that names someone or something, not a word that starts a
// new sentence, a title or a compass point.
bool name_at(const Doc& d, size_t k) {
    if (k >= d.size() || !upper_first(d, k) || !d[k].space_before || d[k].newline_before) return false;
    std::string_view low = d.low(k);
    if (STARTERS.contains(low) || is_title_entry(d, k)) return false;
    if (d[k].casing == rv::UPPER && direction_word(low)) return false;
    return true;
}

// The word before token i can be a street name: "Elm St.", "Cherry Tree Rd.", "5th St.".
bool street_name_before(const Doc& d, size_t i) {
    if (i == 0 || !d[i].space_before || d[i].newline_before) return false;
    size_t p = i - 1;
    if (p >= 1 && d.word(p) && ORDINAL_SUFFIX.contains(d.low(p)) && d.number(p - 1) && d.joined(p)) return true;
    // an abbreviated name: "the Mkt. St. station", "Univ. Ave."
    if (p >= 1 && d.punct(p, '.') && d.joined(p) && upper_first(d, p - 1) && d[p - 1].casing == rv::TITLE &&
        d.str(p - 1).size() >= 3 && find_entry(d.low(p - 1)) != nullptr && !is_title_entry(d, p - 1))
        return true;
    if (!upper_first(d, p) || d[p].casing == rv::UPPER) return false;
    return !NOT_STREET.contains(d.low(p));
}

// ------------------------------------------------------------------------------------------------ sentence ends

enum class End { NEVER, WEAK, STRONG };

// The period at token p also ends the sentence. STRONG: any capitalized word after it starts a sentence. WEAK:
// only a sentence-starter word does. NEVER: only the end of the text or of a line.
bool period_ends_sentence(const Doc& d, size_t p, End mode) {
    size_t n = p + 1;
    if (n >= d.size() || d[n].newline_before) return true;
    if (!d[n].space_before && d.punct(n, '.')) return true;  // "etc..." keeps its dots
    size_t k = n;
    bool closed = false;
    while (k < d.size() && !d[k].space_before && d[k].kind == rv::Kind::Punct &&
           (closing_mark(d[k].cp) || d[k].cp == '*' || d[k].cp == '_')) {
        ++k;
        closed = true;
    }
    if (closed && (k >= d.size() || d[k].newline_before)) return true;
    if (mode == End::NEVER) return false;
    // shouted text: "ACME CORP. AND GLOBEX INC. MERGE" goes on
    if (p > 0 && d.word(p - 1) && d[p - 1].casing == rv::UPPER && d.str(p - 1).size() >= 2 && k < d.size() &&
        d.word(k) && d[k].casing == rv::UPPER && d.str(k).size() >= 2 && d[k].space_before)
        return false;
    if (k < d.size() && d[k].space_before && d.punct(k, 0xB6)) return true;  // a pilcrow: "Washington Sq. ¶ For"
    if (k < d.size() && d[k].space_before && d.punct(k, '<')) {
        // a tag that is left out: "Elm Pl. <Footnote 3: ...>"; its own final period keeps ours from doubling
        for (size_t t = k + 1; t < d.size() && t < k + 48; ++t)
            if (d.punct(t, '>')) return !d.punct(t + 1, '.');
        return false;
    }
    // a list number: "Turn right on Elm St. 2. Proceed to Maple Ct.", "at 123 N. Main St. I. The first item"
    bool list_label = d.number(k) ? d.str(k).size() <= 2
                                  : d.word(k) && d[k].casing == rv::UPPER && rv::roman_value(d.str(k)) > 0;
    if (k + 2 < d.size() && d[k].space_before && list_label && (d.punct(k + 1, '.') || d.punct(k + 1, ')')) &&
        d.joined(k + 1) && d[k + 2].space_before && upper_first(d, k + 2))
        return true;
    // an opening quote before the next word: 'Sunset Rd. "We'll head'; a bracket is an aside ("Inc. (AAPL)")
    if (!closed && k < d.size() && d[k].space_before && d[k].kind == rv::Kind::Punct && d.word(k + 1) &&
        d.joined(k + 1) && (d[k].cp == '"' || d[k].cp == 0x201C || d[k].cp == 0x2018))
        return upper_first(d, k + 1);
    if (k >= d.size() || !d[k].space_before || !upper_first(d, k)) return false;
    if (d[k].casing == rv::UPPER && direction_word(d.low(k))) return false;  // "Main St. NE"
    if (mode == End::STRONG) return !continues_name(d, k);
    return STARTERS.contains(d.low(k)) || is_title_entry(d, k);
}

// ------------------------------------------------------------------------------------------------ numbers

std::string upper_letter(std::string_view w) { return std::string(1, static_cast<char>(w[0] & ~0x20)); }

// A reference number starting at token k: "3", "2.1", "3.2.1", "5b", "(5)". Sets last; "" if none.
std::string ref_number(const Doc& d, size_t k, size_t& last) {
    // a lettered number: "App. C.1" -> C point one
    if (d.word(k) && d.str(k).size() == 1 && d[k].casing == rv::UPPER && d.punct(k + 1, '.') && d.joined(k + 1) &&
        d.number(k + 2) && d.joined(k + 2)) {
        std::string rest = ref_number(d, k + 2, last);
        if (!rest.empty()) return std::string(d.str(k)) + " point " + rest;
    }
    bool paren = d.punct(k, '(') && d.number(k + 1) && d.joined(k + 1);
    size_t s = paren ? k + 1 : k;
    if (!d.number(s)) return "";
    std::string_view parts[8];
    size_t n = 0, j = s;
    parts[n++] = d.str(s);
    while (n < 8 && d.punct(j + 1, '.') && d.joined(j + 1) && d.number(j + 2) && d.joined(j + 2)) {
        parts[n++] = d.str(j + 2);
        j += 2;
    }
    std::string words;
    if (n == 2) {
        words = rv::read_number(std::string(parts[0]) + "." + std::string(parts[1]));
        if (words.empty()) return "";
    } else {
        for (size_t t = 0; t < n; ++t) {
            std::string w = rv::read_number(parts[t]);
            if (w.empty()) w = rv::digits(parts[t]);
            if (!words.empty()) words += " point ";
            words += w;
        }
    }
    // one letter glued to the number: "2a" -> "two A"
    if (d.word(j + 1) && d.joined(j + 1) && d.str(j + 1).size() == 1) {
        words += " " + upper_letter(d.str(j + 1));
        ++j;
    }
    if (paren) {
        if (!d.punct(j + 1, ')') || !d.joined(j + 1)) return "";
        ++j;
    }
    if (!clean_end(d, j)) return "";
    last = j;
    return words;
}

// A number after an address word: "305" -> "three oh five", "4B" -> "four B", "12" -> "twelve".
std::string address_number(const Doc& d, size_t k, size_t& last) {
    if (!d.number(k)) return "";
    std::string_view s = d.str(k);
    if (s.size() > 4 || (s.size() > 1 && s[0] == '0')) return "";
    uint32_t n = 0;
    for (char c : s) n = n * 10 + static_cast<uint32_t>(c - '0');
    std::string words = s.size() >= 3 ? rv::pairs(n) : rv::cardinal(n);
    size_t j = k;
    if (d.word(j + 1) && d.joined(j + 1) && d.str(j + 1).size() == 1) {
        words += " " + upper_letter(d.str(j + 1));
        ++j;
    }
    if ((d.punct(j + 1, ',') || d.punct(j + 1, '.')) && d.joined(j + 1) && d.number(j + 2) && d.joined(j + 2))
        return "";  // a grouped quantity or a decimal
    if (!clean_end(d, j)) return "";
    last = j;
    return words;
}

// A four-digit year after "est." ("est. 1992", "est. in 1992"): the founding year, so "established".
bool year_at(const Doc& d, size_t k) {
    if (!d.number(k) || d.str(k).size() != 4) return false;
    if (d.punct(k + 1, ',') && d.number(k + 2) && d.joined(k + 2)) return false;
    char c = d.str(k)[0];
    return c == '1' || c == '2';
}

// ------------------------------------------------------------------------------------------------ the word rule

// Words that go on with a name after an abbreviation inside it: "Elm St. Parkway", "Halvorsen Intl. Holdings".
const rv::WordSet NAME_GOES_ON{
    "academy", "agency", "airport", "apartment", "apartments", "arena", "avenue", "boulevard", "bridge", "building",
    "campus", "cathedral", "cemetery", "center", "centre", "chapel", "church", "class", "clinic", "college",
    "corporation", "series",
    "court", "day", "elementary", "expressway", "freeway", "garden", "gardens", "group", "hall", "high", "highway",
    "holdings", "hospital", "hotel", "lab", "labs", "lane", "library", "mall", "market", "memorial", "museum", "park",
    "parkway", "partners", "place", "plaza", "press", "road", "school", "square", "stadium", "station", "street",
    "suite", "terminal", "theater", "theatre", "tower", "transport", "tunnel"};

struct Match {
    size_t first, last_word;  // tokens of the written form, without the final period
    bool period;              // a final period follows last_word
    bool upper;               // written with a capital first letter
};

bool is_title_entry(const Doc& d, size_t k) {
    if (!upper_first(d, k) || !d.punct(k + 1, '.') || !d.joined(k + 1)) return false;
    const Entry* e = find_entry(d.low(k));
    return e && (e->flags & TITLE);
}

// The capitalized word at k goes on with the name before it: another abbreviation of a name (not a title) or a
// word such as "Parkway" or "Holdings".
bool continues_name(const Doc& d, size_t k) {
    if (NAME_GOES_ON.contains(d.low(k))) return true;
    if (d[k].casing == rv::UPPER && d.str(k) != "I" && rv::roman_value(d.str(k)) > 0) return true;  // "Jr. III"
    if (!d.punct(k + 1, '.') || !d.joined(k + 1)) return false;
    // "the Mkt. St. station"; not "Elm St. Dr. Smith was waiting"
    if (STREET_WORDS.contains(d.low(k)) && !(is_title_entry(d, k) && name_at(d, k + 2))) return true;
    const Entry* e = find_entry(d.low(k));
    return e && (e->flags & SUFFIX) && (e->flags & CAP);
}

void emit(const Doc& d, Out& out, const Match& m, std::string spoken, End mode, int score) {
    // an input that is only the abbreviation loses its period: "Inc." -> "Incorporated"
    bool alone = m.first == 0 && m.last_word + 2 == d.size();
    // a spaced ellipsis after the period, then the sentence goes on: "on Elm Rd. ...it holds" -> "Road, it"
    size_t n = m.last_word + 2;
    if (m.period && n < d.size() && d[n].space_before && !d[n].newline_before) {
        size_t e = Doc::npos;
        if (d.punct(n, 0x2026)) e = n;
        else if (d.punct(n, '.') && d.punct(n + 1, '.') && d.punct(n + 2, '.') && d.joined(n + 1) && d.joined(n + 2))
            e = n + 2;
        if (e != Doc::npos && lower_first(d, e + 1)) {
            out.tokens(m.first, e, spoken + (d.joined(e + 1) ? ", " : ","), score);
            return;
        }
    }
    bool keep = m.period && !alone && period_ends_sentence(d, m.last_word + 1, mode);
    size_t last = m.period && !keep ? m.last_word + 1 : m.last_word;
    out.tokens(m.first, last, std::move(spoken), score);
}

// Words that follow a numeral but never the pronoun: "Preface I for details" against "Preface. I hope".
const rv::WordSet AFTER_NUMERAL{"and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"};

// A reference word, with the number after it when there is one ("Fig. 5b", "Figs. 2-4", "Eq. (3)").
void reference(const Doc& d, Out& out, const Match& m, const Entry& e, std::string word) {
    size_t k = m.period ? m.last_word + 2 : m.last_word + 1;
    bool next_ok = k < d.size() && !d[k].newline_before && (d[k].space_before || (m.period && d.number(k)));
    if (next_ok) {
        size_t last = 0;
        std::string num = ref_number(d, k, last);
        if (!num.empty()) {
            if (e.flags & CAPNUM) num = rv::capital(num);
            size_t dash = last + 1;
            if (dash < d.size() && d[dash].kind == rv::Kind::Punct &&
                (d[dash].cp == '-' || d[dash].cp == 0x2013 || d[dash].cp == 0x2014) &&
                d.joined(dash) == d.joined(dash + 1)) {
                size_t last2 = 0;
                std::string to = ref_number(d, dash + 1, last2);
                if (!to.empty()) {
                    num += " to " + to;
                    last = last2;
                }
            }
            // more numbers in brackets ("Eqs. (2) and (3)", "Eq. (4), (5) or (6)"): the bracket would read as a
            // citation marker
            bool paren = d.punct(k, '(');
            for (int n = 0; paren && n < 6; ++n) {
                size_t t = last + 1;
                std::string sep;
                if (d.punct(t, ',') && d.joined(t)) {
                    sep = ",";
                    ++t;
                }
                if (d.word(t) && d[t].space_before && (d.is(t, "and") || d.is(t, "or") || d.is(t, "to"))) {
                    sep += " " + std::string(d.str(t));
                    ++t;
                }
                if (sep.empty() || !d.punct(t, '(') || !d[t].space_before) break;
                size_t last2 = 0;
                std::string more = ref_number(d, t, last2);
                if (more.empty()) break;
                num += sep + " " + more;
                last = last2;
            }
            out.tokens(m.first, last, word + " " + num, 80);
            return;
        }
    }
    if (!m.period) return;
    if (e.key == "sec" && m.first > 0 && d.number(m.first - 1) && d[m.first].space_before) return;  // "1 sec."
    // a numeral or a letter after it is read by other rules: "Vol. III", "App. C"; "I hope" is a new sentence
    bool label_after = next_ok && d.word(k) && d[k].casing == rv::UPPER &&
                       !(d.str(k) == "I" && d.word(k + 1) && d[k + 1].space_before && lower_first(d, k + 1) &&
                         !AFTER_NUMERAL.contains(d.low(k + 1)));
    if ((e.flags & NUMONLY) && !label_after) return;
    // "Spitfire Mk. IX": no other rule takes a numeral after "Mk."
    if (e.key == "mk" && label_after && d.str(k).size() > 1 && clean_end(d, k)) {
        int v = rv::roman_value(d.str(k));
        if (v > 0) {
            out.tokens(m.first, k, word + " " + rv::capital(rv::cardinal(static_cast<uint64_t>(v))), 80);
            return;
        }
    }
    // before a numeral another rule reads, the word alone scores low so that a reading of the whole wins
    emit(d, out, m, std::move(word), label_after ? End::NEVER : End::STRONG, label_after ? 60 : 80);
}

// Context decisions: the spoken form ("" when the abbreviation does not apply here); may change the period mode.
std::string context(const Doc& d, const Match& m, const Entry& e, End& mode) {
    std::string_view key = e.key;
    size_t i = m.first, k = m.period ? m.last_word + 2 : m.last_word + 1;  // k: first token after the form
    bool prev_number = i > 0 && d.number(i - 1) && d[i].space_before;
    if (key == "dr") {
        bool address = i >= 2 && (d.number(i - 2) || (lower_first(d, i - 2) && DIRECTION_PREPS.contains(d.low(i - 2))));
        if (street_name_before(d, i) && (!name_at(d, k) || address)) {
            mode = End::STRONG;
            return "Drive";
        }
        return m.period || name_at(d, k) ? "Doctor" : "";
    }
    if (key == "st") {
        // "Main St." Street; "Hotel St. Regis", "Rue St. Denis" Saint before a name, unless the text is an address
        bool address = i >= 2 && (d.number(i - 2) || (lower_first(d, i - 2) && DIRECTION_PREPS.contains(d.low(i - 2))));
        if (i > 0 && SAINT_BEFORE.contains(d.low(i - 1))) address = false;  // "Lake St. Clair", "Rue St. Denis"
        if (street_name_before(d, i) &&
            (!name_at(d, k) || address || NAME_GOES_ON.contains(d.low(k)) || continues_name(d, k))) {
            mode = End::STRONG;
            return "Street";
        }
        if (upper_first(d, k) && d[k].space_before && !d[k].newline_before) return "Saint";
        if (!m.period) return "";
        mode = End::STRONG;
        return "Street";
    }
    if (key == "rev") {
        if (!m.period) return name_at(d, k) ? "Reverend" : "";
        return d.number(k) && d[k].space_before ? "Revision" : "Reverend";
    }
    if (key == "rep" && d.is(k, "of")) return "Republic";  // "Rep. of Korea"
    if (key == "gen") {
        if (d.number(k) && d.punct(k + 1, ':')) return "";  // "Gen. 1:1" is scripture
        // "Gen. Z", "Gen. Alpha" are generations
        if (d.word(k) && ((d.str(k).size() == 1 && d[k].casing == rv::UPPER) || d.is(k, "alpha"))) return "";
        return "General";
    }
    if (key == "mt" || key == "ft") {  // "2 Mt." megatonnes, "5 ft." feet
        if (prev_number || !upper_first(d, k) || !d[k].space_before || STARTERS.contains(d.low(k)))
            return "";
        if (!m.period && (d.str(k).size() < 2 || d[k].casing == rv::UPPER)) return "";
        if (key == "ft" && i >= 2 && d.is(i - 2, "sq")) return "";
        return std::string(e.spoken);
    }
    if (key == "fr") {  // "Fr. Brennan" Father; "Fr." alone is French or Friday
        if (!name_at(d, k) || d.str(k).size() < 2 || d[k].casing == rv::UPPER) return "";
        return "Father";
    }
    if (key == "sr" || key == "jr") {
        // "Sr. Vice President", "Jr. Analyst", "Sr. Software Engineer": a rank before a job title
        for (size_t t = k, n = 0; n < 3 && d.word(t) && d[t].space_before && !d[t].newline_before; ++t, ++n) {
            if (JOB_WORDS.contains(d.low(t))) {
                mode = End::NEVER;
                return std::string(e.spoken);
            }
            if (!upper_first(d, t)) break;
        }
        // "Sr. Mary Joseph": before a name, not after one, it is Sister
        bool after_name = i > 0 && d[i].space_before && !d[i].newline_before &&
                          (upper_first(d, i - 1) || d.punct(i - 1, ',')) && !STARTERS.contains(d.low(i - 1));
        if (key == "sr" && !after_name && name_at(d, k) && d.str(k).size() > 1 && d[k].casing != rv::UPPER) {
            mode = End::NEVER;
            return "Sister";
        }
        return std::string(e.spoken);
    }
    if (key == "sq") {  // "500 Sq. Ft." is an area
        if (d.word(k) && UNIT_AFTER_SQ.contains(d.low(k)) && !d.is(k, "in")) return "";
        if (prev_number && !upper_first(d, k)) return "";
        if (!m.period && !street_name_before(d, i)) return "";
        if (prev_number && name_at(d, k)) mode = End::NEVER;  // "15 Sq. Gardenia"
        return "Square";
    }
    if (key == "temp") {  // "max. temp. 35°C"; "I hired a temp." is a word
        if (!m.period || k >= d.size() || d[k].newline_before) return "";
        bool after_measure = i >= 2 && d.punct(i - 1, '.') && d.word(i - 2) &&
                             (d.is(i - 2, "max") || d.is(i - 2, "min") || d.is(i - 2, "avg"));
        // "the temp. dropped", "body temp. 38°C"; "Temp. agency staff" is temporary
        bool before = i > 0 && d.word(i - 1) && d[i].space_before &&
                      (DETERMINERS.contains(d.low(i - 1)) || TEMP_BEFORE.contains(d.low(i - 1)));
        if (!(after_measure || d.number(k) || d.punct(k, ':') || (before && lower_first(d, k)))) return "";
        return "temperature";
    }
    if (key == "lit") {  // a gloss: "(ཇོ་མོ་གླང་མ, lit. "holy mother")"; "the lamp was lit." is a word
        if (!m.period || k >= d.size() || d[k].newline_before || !d[k].space_before) return "";
        bool after = i == 0 || d[i].newline_before || d.punct(i - 1, '(') || d.punct(i - 1, ',') || d.punct(i - 1, ';');
        bool before = d.punct(k, '"') || d.punct(k, '\'') || d.punct(k, 0x201C) || d.punct(k, 0x2018) ||
                      (d.word(k) && lower_first(d, k));
        return after && before ? "literally" : "";
    }
    if (key == "pop") {  // "pop. 4,500" population; "the balloon went pop." is a word
        if (!m.period || !d.number(k) || !d[k].space_before) return "";
        mode = End::NEVER;
        return "population";
    }
    if (key == "pc") {  // "1 pc." one piece
        return prev_number && m.period && !m.upper ? "piece" : "";
    }
    if (key == "ops" || key == "mktg" || key == "engr") {  // "Dir. of Mktg. Sarah Lee": a name may follow
        mode = End::WEAK;
        return std::string(e.spoken);
    }
    if (key == "blk" && m.upper && m.period && d.number(k) && d[k].space_before) {  // "Lot 7, Blk. 3"
        mode = End::NEVER;
        return "Block";
    }
    if (key == "pts" || key == "reb" || key == "ast" || key == "stl" || key == "blk") {
        if (!prev_number || m.upper || d.is(k, "of")) return "";  // "3 pts. of cream" are pints
        std::string s(e.spoken);
        if (d.str(i - 1) == "1" && !(i >= 2 && d.punct(i - 2, '.'))) s.pop_back();  // "1 reb." one rebound
        return s;
    }
    if (key == "ste") {  // "Ste. 400" Suite; "Sault Ste. Marie" Sainte, before the name
        if (!name_at(d, k)) return "Suite";
        mode = End::NEVER;
        return "Sainte";
    }
    if (key == "co") {
        size_t p = i > 0 && d[i].space_before && !d[i].newline_before ? i - 1 : Doc::npos;
        bool company = p != Doc::npos && (d.punct(p, '&') || (d.word(p) && BUSINESS.contains(d.low(p))) ||
                                          (p >= 1 && d.is(p - 1, "and") && upper_first(d, p)));
        if (d.is(k, "inc") || d.is(k, "ltd") || (d.punct(k, ',') && d.is(k + 1, "ltd"))) company = true;
        // the capitalized name before it: "Johnson & Johnson Co." and "The Walt Disney Co." are companies,
        // "in Cook Co." and "through Los Angeles Co." are counties
        size_t q = p, words = 0;
        bool amp = false;
        while (q != Doc::npos && words < 4) {
            if (upper_first(d, q) && !STARTERS.contains(d.low(q))) ++words;
            else if (d.punct(q, '&')) amp = true;
            else break;
            q = q > 0 && d[q].space_before && !d[q].newline_before ? q - 1 : Doc::npos;
        }
        size_t before = q;  // the token before the name
        bool locative = before != Doc::npos && words > 0 && d.word(before) && lower_first(d, before) &&
                        COUNTY_BEFORE.contains(d.low(before));
        if (amp) company = true;
        if (!company && !locative && words > 0) {
            if (words >= 2) company = true;
            else if (before != Doc::npos && d.word(before) && COMPANY_BEFORE.contains(d.low(before))) company = true;
            else if (before != Doc::npos && before > 0 && d.punct(before, ',') && d.word(before - 1) &&
                     COMPANY_BEFORE.contains(d.low(before - 1)))
                company = true;  // "Our reseller, Brix Co., ships"
        }
        if (p == Doc::npos || !upper_first(d, p) || STARTERS.contains(d.low(p)))
            mode = End::WEAK;  // before the name: "the Co. Kerry coast"; "for that Co. It was" ends
        return company ? "Company" : "County";
    }
    if (key == "inc" && !m.upper) {  // "inc. 1902" incorporated in that year
        if (m.period && i > 0 && d.is(i - 1, "net")) return "income";  // "Net inc. fell 3%"
        if (!m.period || !year_at(d, k) || !d[k].space_before) return "";
        mode = End::NEVER;
        return "incorporated";
    }
    if (key == "inc" || key == "corp" || key == "ltd") {
        bool closes = i > 0 && d[i].space_before && !d[i].newline_before &&
                      (upper_first(d, i - 1) || d.punct(i - 1, ',') || d.punct(i - 1, '&')) &&
                      !STARTERS.contains(d.low(i - 1));
        // without the period only after a name: "Acme Inc reported", "Northwind Ltd"
        if (!m.period && (!closes || d[i].casing != rv::TITLE || !upper_first(d, i - 1) || name_at(d, k))) return "";
        if (!closes) mode = End::NEVER;
        return std::string(e.spoken);
    }
    if (ORG_WORDS.contains(key)) {
        if (!m.period) return "";
        // after a capitalized name it closes the name ("Vertex Intl."), else it starts one ("Intl. Logistics",
        // "The Intl. Monetary Fund")
        bool closes = i > 0 && d[i].space_before && !d[i].newline_before && upper_first(d, i - 1) &&
                      !STARTERS.contains(d.low(i - 1));
        if (key == "assoc" && d.word(k) && d[k].space_before && JOB_WORDS.contains(d.low(k))) {
            mode = End::NEVER;  // "Assoc. Prof. Kim"
            return "associate";
        }
        mode = closes ? End::STRONG : End::NEVER;
        return std::string(e.spoken);
    }
    if (key == "v") {  // "Smith v. Jones", "rock v. pop", "No. 789 v. State"
        // after an abbreviation too: "Miller et al. v. State", "U.S. v. Nixon"
        bool abbr_before = i >= 2 && d.punct(i - 1, '.') && d.word(i - 2) && d.joined(i - 1);
        if (!m.period || m.upper || i == 0 || !(d.word(i - 1) || d.number(i - 1) || abbr_before) || !d[i].space_before)
            return "";
        if (!d.word(k) || !d[k].space_before || d[k].newline_before) return "";
        return "versus";
    }
    if (key == "est") {  // all capitals it is the time zone: "9 a.m. EST."
        if (d[i].casing == rv::UPPER) return "";
        return year_at(d, k) || (d.is(k, "in") && year_at(d, k + 1)) ? "established" : "estimated";
    }
    if (key == "min" || key == "max") {  // "15 min." is a duration; "the min. Then" ends a sentence
        mode = End::WEAK;  // "Max. Load: 500 lbs" is a label
        return prev_number || !m.period ? "" : std::string(e.spoken);
    }
    if (key == "no") {  // "the serial no. is ..."; "No. 7" belongs to the number rules
        if (!m.period || d.number(k) || d.punct(k, '#')) return "";
        if (d.is(k, "of") && lower_first(d, k) && d[k].space_before) {  // "No. of users: 500"
            mode = End::NEVER;
            return "number";
        }
        if (i == 0 || !d.word(i - 1) || !d[i].space_before || !NUMBER_BEFORE.contains(d.low(i - 1))) return "";
        // after a bare determiner something must follow: "the no. of items", but "a yes or a no."
        if (DETERMINERS.contains(d.low(i - 1)) &&
            !(k < d.size() && d[k].space_before && !d[k].newline_before && (lower_first(d, k) || d.punct(k, ':'))))
            return "";
        return "number";
    }
    if (key == "ed") {
        if (d.is(k, "by")) return "edited";
        if (d.is(k, "board") && d[k].space_before) return "Editorial";
        bool edition = d.number(k) && d[k].space_before;
        if (i > 0 && d[i].space_before && d.word(i - 1)) {
            size_t p = i - 1;
            if (EDITION_BEFORE.contains(d.low(p)) ||
                (p >= 1 && ORDINAL_SUFFIX.contains(d.low(p)) && d.number(p - 1) && d.joined(p)))
                edition = true;
        }
        if (!edition && (name_at(d, k) || is_title_entry(d, k))) mode = End::NEVER;  // "Ed. Dr. Lena Hansen"
        if (edition && i > 0 && d.word(i - 1) && COMMON_NOUN_BEFORE.contains(d.low(i - 1))) return "edition";
        return edition ? "Edition" : "Editor";
    }
    if (key == "art" || key == "arts" || key == "op" || key == "ref" || key == "refs") {
        // only as a label before its number or numeral: "Art. 5", "Op. 9", "Ref. A-12"
        if (!m.period || k >= d.size() || !d[k].space_before ||
            !(d.number(k) || (d.word(k) && d[k].casing == rv::UPPER) ||
              (key == "ref" && d.word(k) && d.is(k, "doc") && d.punct(k + 1, '.'))))
            return "";
        mode = End::NEVER;
        return std::string(e.spoken);
    }
    if (key == "wt" || key == "ht" || key == "dims") {  // "Wt.: 5 kg", "Approx. Wt. 4 lbs."
        if (!m.period || !(d.punct(k, ':') || (d.number(k) && d[k].space_before))) return "";
        mode = End::NEVER;
        return std::string(e.spoken);
    }
    return std::string(e.spoken);
}

// Token i sits in code: brackets, backticks or an assignment right around it.
bool code_around(const Doc& d, size_t i) {
    if (d.joined(i) && (d.punct(i - 1, '(') || d.punct(i - 1, '[') || d.punct(i - 1, '`') || d.punct(i - 1, '.')))
        return true;
    size_t n = i + 1;
    if (n < d.size() && d.joined(n) &&
        (d.punct(n, ')') || d.punct(n, ']') || d.punct(n, '`') || d.punct(n, '(') || d.punct(n, '[') ||
         d.punct(n, '=')))
        return true;
    return n < d.size() && d.punct(n, '=') && d[n].space_before;
}

// The text around token i (an all-capitals abbreviation) is shouted: the word before it, or at the start the word
// after it (token after), is in capitals too and is not AM or PM.
bool shouted(const Doc& d, size_t i, size_t after) {
    size_t w = d.prev_word(i);
    if (w == Doc::npos || d[i].newline_before) w = after < d.size() && d.word(after) ? after : Doc::npos;
    if (w == Doc::npos || !d.word(w) || d[w].casing != rv::UPPER || d.str(w).size() < 2) return false;
    return !(d.is(w, "am") || d.is(w, "pm"));
}

// "u.s.a" -> true: single letters with periods between them.
bool all_single_letters(std::string_view key) {
    if (key.size() < 3 || key.size() % 2 == 0) return false;
    for (size_t k = 0; k < key.size(); ++k) {
        char c = key[k];
        if (k % 2 == 0 ? !(c >= 'a' && c <= 'z') : c != '.') return false;
    }
    return true;
}

// The written form at token i, whatever it is; returns where the scan continues.
size_t one_form(const Doc& doc, Out& out, size_t i) {
    // a title glued to a name, the space missing: "Mr.Smith", "Dr.Jones", "St.Louis" (not "St.Mary.org")
    if (doc[i].casing == rv::TITLE && doc.punct(i + 1, '.') && doc.joined(i + 1) && doc.word(i + 2) &&
        doc.joined(i + 2) && doc[i + 2].casing == rv::TITLE && doc.str(i + 2).size() > 1 &&
        !(doc.joined(i + 3) && i + 3 < doc.size() && (doc.punct(i + 3, '.') || doc.punct(i + 3, '/')) &&
          (doc.word(i + 4) || doc.number(i + 4)) && doc.joined(i + 4))) {
        std::string_view t = doc.low(i);
        const char* s = t == "mr" ? "Mister" : t == "mrs" ? "Missus" : t == "ms" ? "Miss" : t == "dr" ? "Doctor" :
                        t == "prof" ? "Professor" : t == "st" ? "Saint" : nullptr;
        if (s) {
            out.tokens(i, i + 1, s, 80);
            return i + 1;
        }
    }
    // "e. g." and "i. e." written with a space
    if (doc[i].casing == rv::LOWER && (doc.str(i) == "e" || doc.str(i) == "i") && doc.punct(i + 1, '.') &&
        doc.joined(i + 1) && doc.word(i + 2) && doc[i + 2].space_before && !doc[i + 2].newline_before &&
        doc.str(i + 2) == (doc.str(i) == "e" ? "g" : "e") && doc.punct(i + 3, '.') && doc.joined(i + 3) &&
        clean_end(doc, i + 3)) {
        Match sm{i, i + 2, true, false};
        emit(doc, out, sm, doc.str(i) == "e" ? "for example" : "that is", End::NEVER, 82);
        return i + 3;
    }
    // letters, then ".letters" groups, all glued together
    size_t j = i;
    while (doc.punct(j + 1, '.') && doc.joined(j + 1) && doc.word(j + 2) && doc.joined(j + 2)) j += 2;
    bool period = doc.punct(j + 1, '.') && doc.joined(j + 1);
    size_t after = period ? j + 2 : j + 1;
    bool glued_number = false;
    if (after < doc.size() && doc.joined(after) && (doc.word(after) || doc.number(after))) {
        // "v.3.1.4", "Mr.Smith": something else; only a reference word takes a glued number ("fig.0.5")
        if (!(period && j == i && doc.number(after))) return j;
        glued_number = true;
    }
    std::string_view key(doc.lower.data() + doc[i].start, doc[j].end - doc[i].start);
    Match m{i, j, period, rv::is_upper(doc[i].cp)};

    // a plural with a lower-case s: "three M.D.s", "two Ph.D.s"
    if (j >= i + 4 && doc.str(j) == "s" && !glued_number && clean_end(doc, period ? j + 1 : j)) {
        std::string_view base = key.substr(0, key.size() - 2);
        std::string letters;
        if (all_single_letters(base) && rv::is_upper(doc[j - 2].cp)) {
            for (size_t t = i; t + 2 <= j; t += 2) letters += static_cast<char>(doc.str(t)[0] & ~0x20);
        } else if (const Entry* pe = find_entry(base); pe && (pe->flags & FIXED) && m.upper) {
            letters = std::string(pe->spoken);
        }
        if (!letters.empty()) {
            out.tokens(i, j, letters + "s", 62);
            return j;
        }
    }

    // two-word forms: "op. cit.", "pro tem.", "Rev. Ed."
    if (j == i) {
        for (const Pair& p : PAIRS) {
            if (key != p.first || period != p.first_period || (p.fixed && !m.upper)) continue;
            size_t w = period ? j + 2 : j + 1;
            if (!doc.word(w) || !doc[w].space_before || doc[w].newline_before || doc.low(w) != p.second) continue;
            bool second_period = doc.punct(w + 1, '.') && doc.joined(w + 1);
            if (second_period != p.second_period || !clean_end(doc, second_period ? w + 1 : w)) continue;
            Match pm{i, w, second_period, m.upper};
            std::string spoken = p.fixed ? std::string(p.spoken) : match_case(std::string(p.spoken), m.upper);
            emit(doc, out, pm, spoken, p.first == "rt" || p.first == "ed" ? End::NEVER : End::STRONG, 82);
            return second_period ? w + 1 : w;
        }
    }

    // "Springfield, Ill." -> "Springfield, Illinois"
    bool city_before = i >= 2 && (upper_first(doc, i - 2) ||
                                  (i >= 3 && doc.punct(i - 2, '.') && doc.joined(i - 2) && upper_first(doc, i - 3)));
    if (period && m.upper && !glued_number && i >= 2 && doc.punct(i - 1, ',') && doc[i].space_before &&
        city_before && doc.joined(i - 1)) {
        if (const std::string_view* state = STATES.find(key)) {
            if (clean_end(doc, j + 1)) {
                emit(doc, out, m, std::string(*state), End::STRONG, 81);
                return j;
            }
        }
    }
    // the forms that are no English word: "moved to Calif. in 2010", "Sen. Kyl (R-Ariz.)"
    if (period && m.upper && !glued_number && i >= 1 && STATES.find(key) && clean_end(doc, j + 1)) {
        bool party = i >= 3 && doc.punct(i - 1, '-') && doc.joined(i - 1) && doc.joined(i) && doc.word(i - 2) &&
                     doc.str(i - 2).size() == 1 && doc[i - 2].casing == rv::UPPER && doc.punct(i - 3, '(');
        bool place = j == i && SAFE_STATES.contains(key) && doc[i].space_before && lower_first(doc, i - 1) &&
                     STATE_PREPS.contains(doc.low(i - 1));
        if (party || place) {
            emit(doc, out, m, std::string(*STATES.find(key)), End::STRONG, 81);
            return j;
        }
    }

    // "my dr appt", "the dr. said": a doctor in chat
    if (j == i && key == "dr" && doc[i].casing == rv::LOWER && i > 0 && doc.word(i - 1) && doc[i].space_before &&
        DETERMINERS.contains(doc.low(i - 1)) && (period || POSSESSIVES.contains(doc.low(i - 1))) && !glued_number &&
        clean_end(doc, period ? j + 1 : j)) {
        Match dm{i, j, period, false};
        emit(doc, out, dm, "doctor", End::STRONG, 80);
        return j;
    }
    // lower-case forms of capitalized entries in report and chart notes
    if (period && j == i && !glued_number && clean_end(doc, j + 1)) {
        const char* spoken = nullptr;
        End mode = End::STRONG;
        size_t k = j + 2;
        bool next_lower = k < doc.size() && doc[k].space_before && !doc[k].newline_before && lower_first(doc, k);
        bool num_before = i > 0 && doc.number(i - 1) && doc[i].space_before;
        bool one_before = num_before && doc.str(i - 1) == "1" && !(i >= 2 && doc.punct(i - 2, '.'));
        bool clause_start = i == 0 || doc.sentence_start(i) || doc.punct(i - 1, ',') || doc.punct(i - 1, ';');
        if (key == "pt" && !m.upper && i > 0 &&
            ((doc.word(i - 1) && DETERMINERS.contains(doc.low(i - 1))) || (clause_start && next_lower))) {
            spoken = "patient";  // "the pt. was seen", "38.5°C, pt. alert and oriented"
        } else if (key == "pt" && m.upper && doc.sentence_start(i) && next_lower) {
            spoken = "Patient";  // "Pt. presents with"
        } else if ((key == "tab" || key == "tabs") && !m.upper && num_before && next_lower) {
            spoken = one_before ? "tablet" : "tablets";  // "Take 1 tab. by mouth"
        } else if (key == "rev" && !m.upper && doc.word(k) && next_lower && i > 0 && doc[i].space_before) {
            spoken = "revenue";  // "rev. rose to $4.2B"
        }
        if (spoken) {
            emit(doc, out, m, spoken, mode, 80);
            return j;
        }
    }

    // "Ref. No. 12345": the number words are ours, "No." after "Ref." reads as the start of a sentence
    if (key == "ref" && period && j == i && doc.is(after, "no") && doc.punct(after + 1, '.') &&
        doc.joined(after + 1) && doc.number(after + 2) && doc[after + 2].space_before && clean_end(doc, after + 2) &&
        !(doc.punct(after + 3, ',') && doc.joined(after + 3) && doc.number(after + 4))) {
        std::string_view n = doc.str(after + 2);
        std::string words = n.size() >= 5 || n[0] == '0' ? rv::digits(n) : rv::read_number(n);
        out.tokens(i, after + 2, std::string(m.upper ? "Reference Number " : "reference number ") + words, 80);
        return after + 2;
    }

    const Entry* e = find_entry(key);
    if (e && !period && !(e->flags & BARE)) e = nullptr;
    if (e && (e->flags & CAP) && !m.upper) e = nullptr;
    if (e && j == i && doc[i].casing == rv::MIXED) e = nullptr;  // "PCs", "CoS" are other words
    // all capitals only in shouted text: "MR. SMITH", "MAIN ST."; "5:00 p.m. CT." is a time zone
    if (e && j == i && doc[i].casing == rv::UPPER && doc.str(i).size() >= 2 && !shouted(doc, i, after)) e = nullptr;
    // "1200 Sq. Ft." square feet (the units rules read the lower-case "sq. ft.")
    if (e && key == "sq" && m.upper && i > 0 && doc.number(i - 1) && doc[i].space_before) {
        size_t u = period ? j + 2 : j + 1;
        const char* unit = nullptr;
        bool one_unit = false;
        if (doc.word(u) && doc[u].space_before && !lower_first(doc, u)) {
            std::string_view ul = doc.low(u);
            if (ul == "ft" || ul == "feet") unit = "feet", one_unit = true;
            else if (ul == "yd" || ul == "yds") unit = "yards";
            else if (ul == "mi") unit = "miles";
            else if (ul == "km") unit = "kilometers";
            else if (ul == "m") unit = "meters";
        }
        if (unit) {
            bool uperiod = doc.punct(u + 1, '.') && doc.joined(u + 1);
            size_t n = uperiod ? u + 2 : u + 1;
            // before a noun after "a": "a 1200 Sq. Ft. office" square foot
            size_t s = i - 1;
            while (s > 0 && doc.joined(s) && (doc.number(s - 1) || doc.punct(s - 1, ','))) --s;
            bool adjective = s > 0 && (doc.is(s - 1, "a") || doc.is(s - 1, "an")) && lower_first(doc, n) &&
                             doc[n].space_before && !AFTER_NUMERAL.contains(doc.low(n));
            std::string words = std::string("square ") + (adjective && one_unit ? "foot" : unit);
            if (adjective && !one_unit) words.pop_back();
            Match um{i, u, uperiod, true};
            emit(doc, out, um, words, End::STRONG, 70);
            return uperiod ? u + 1 : u;
        }
    }
    // "Pt. Reyes" is a point of land; "Pt. 3", "Pt. IV" are parts
    if (e && key == "pt" && period && m.upper && name_at(doc, after) && doc.str(after).size() > 1 &&
        doc[after].casing != rv::UPPER && clean_end(doc, j + 1)) {
        emit(doc, out, m, "Point", End::NEVER, 80);
        return j;
    }
    if (e && glued_number && !(e->flags & REF)) e = nullptr;
    if (e && !glued_number && !clean_end(doc, period ? j + 1 : j)) e = nullptr;
    if (e) {
        End mode = (e->flags & (TITLE | PREFIX)) ? End::NEVER : End::STRONG;
        std::string spoken(e->spoken);
        if (e->flags & CTX) {
            spoken = context(doc, m, *e, mode);
            if (spoken.empty()) return j;
        }
        if (!period && (e->flags & TITLE) && spoken != "Street" && spoken != "Drive" &&
            (doc[i].casing != rv::TITLE || !name_at(doc, after) || doc.str(after).size() < 2 ||
             doc[after].casing == rv::UPPER))
            return j;  // "Mr Smith" yes; "Dr" alone, "MS Office" and "Rev B" no
        if (!period && (e->flags & SUFFIX) && (e->flags & CAP) &&
            !(street_name_before(doc, i) && !name_at(doc, after)) &&
            !((e->flags & ADDR) && doc.number(after) && doc[after].space_before))
            return j;  // "Maple Ave", "Hwy 101"; not "Schubert's Ave Maria"
        if (!period && j == i && doc[i].casing == rv::UPPER) return j;  // "VS Code", "ETC" are names
        if (key == "col" && period && doc.number(after) && doc[after].space_before) {  // "Col. 3" of a table
            size_t last = 0;
            if (!ref_number(doc, after, last).empty()) reference(doc, out, m, *e, m.upper ? "Column" : "column");
            return j;
        }
        if ((e->flags & TITLE) && doc.number(after) && key != "rev" && spoken != "Street" && spoken != "Drive")
            return j;  // "Col. 3:16", "Sen. 5": not before a name
        if (e->flags & REF) {
            if ((key == "p" || key == "l") && m.upper) return j;
            reference(doc, out, m, *e, match_case(spoken, m.upper));
            return j;
        }
        if (e->flags & ADDR) {
            // the number after it is read by numbers_context ("Suite three oh five"); only the word is ours
            size_t last = 0;
            if (after < doc.size() && doc[after].space_before && !doc[after].newline_before &&
                !address_number(doc, after, last).empty()) {
                emit(doc, out, m, spoken, End::NEVER, 80);
                return j;
            }
            if (e->flags & NUMONLY) return j;
            // a letter after it is a label, not a new sentence: "Bldg. A, right off"
            if (period && after < doc.size() && doc.word(after) && doc.str(after).size() == 1 &&
                doc[after].casing == rv::UPPER && !(doc.word(after + 1) && doc[after + 1].space_before &&
                                                    lower_first(doc, after + 1)))
                mode = End::NEVER;
        }
        // "(9th Cir. 1994)": a federal circuit court
        if (key == "cir" && i >= 2 && doc.word(i - 1) && ORDINAL_SUFFIX.contains(doc.low(i - 1)) &&
            doc.number(i - 2) && doc.joined(i - 1))
            spoken = "Circuit";
        // code, not prose: "print(msg)", "amt = 5", "`acct`"
        if (!period && !(e->flags & (TITLE | CAP)) && code_around(doc, i)) return j;
        if ((key == "pkg" || key == "pc") && i > 0 && doc.number(i - 1) && doc.str(i - 1) != "1")
            spoken += "s";  // "2 pkg. butter" two packages
        if (spoken == "established" && year_at(doc, after) && doc[after].space_before && clean_end(doc, after)) {
            // the founding year is read as a year: "est. 1992" -> "established nineteen ninety-two"
            std::string_view y = doc.str(after);
            uint32_t n = 0;
            for (char ch : y) n = n * 10 + static_cast<uint32_t>(ch - '0');
            out.tokens(i, after, match_case(spoken, m.upper) + " " + rv::pairs(n), 80);
            return after;
        }
        if (!(e->flags & (FIXED | CAP))) spoken = match_case(spoken, m.upper);
        // "Art. 6(1)(a)", "Op. 9", "Ref. A-12": the label is read by other rules, which may take it in
        bool label = key == "art" || key == "arts" || key == "op" || key == "ref" || key == "refs";
        emit(doc, out, m, spoken, mode, label ? 60 : 80);
        return j;
    }

    // dotted initialisms: "U.S.", "U.K.", "N.B.", "P.S.", "J.P." -> letters in capitals
    if (!glued_number && all_single_letters(key) && key != "a.m" && key != "p.m" &&
        clean_end(doc, period ? j + 1 : j)) {
        std::string letters;
        for (size_t t = i; t <= j; t += 2) letters += static_cast<char>(doc.str(t)[0] & ~0x20);
        // "N.B. the results" and "P.S. I hope" take a comma in place of the period
        if (period && (key == "n.b" || key == "p.s") && doc.word(after) && doc[after].space_before &&
            !doc[after].newline_before &&
            (key == "n.b" ? lower_first(doc, after) : doc.str(after) == "I")) {
            out.tokens(i, j + 1, letters + ",", 62);
            return j + 1;
        }
        // "U.S. Navy" goes on; "P.S. Don't", "the U.S.A. Data" start sentences
        End mode = (key == "p.s" || key == "n.b" || key == "p.p.s" || key == "u.s.a") ? End::STRONG : End::WEAK;
        // "U.S. Rep. Jim Jordan", "U.S. Atty. Gen.": an office of the country
        if (key == "u.s" && period && doc.word(after) && doc[after].space_before && is_title_entry(doc, after) &&
            US_OFFICES.contains(doc.low(after)))
            mode = End::NEVER;
        emit(doc, out, m, letters, mode, 62);  // a citation construct ("18 U.S.C. § 241") may take it in
    }
    return j;
}

// Words that are read without their period too ("Mr Smith", "Maple Ave", "vs"), and first words of two-word forms.
const rv::WordSet BARE_KEYS{
    "acct", "accts", "ad", "amt", "appt", "appts", "approx", "ave", "blvd", "capt", "cmdr", "col", "corp", "cpl",
    "ct", "dr", "et", "etc", "gov't", "gov\xE2\x80\x99t", "hwy", "inc", "jr", "ln", "lt", "ltd", "mgr", "mgrs",
    "mr", "mrs", "ms", "msg", "msgs", "mt", "mtg", "mtgs", "mx", "pkwy", "pl", "pro", "prof", "pts", "pvt", "qtr",
    "qtrs", "rd", "rev", "revd", "sgt", "sq", "sr", "st", "vs"};

// The first two letters of the keys above, as bits: most words are ruled out without a string comparison.
const std::vector<bool>& bare_prefixes() {
    static const std::vector<bool> bits = [] {
        std::vector<bool> b(26 * 26, false);
        for (const char* k : {"ac", "ad", "am", "ap", "av", "bl", "ca", "cm", "co", "cp", "ct", "dr", "et", "go", "hw",
                              "in", "jr", "ln", "lt", "mg", "mr", "ms", "mt", "mx", "pk", "pl", "pr", "pt", "pv", "qt",
                              "rd", "re", "sg", "sq", "sr", "st", "vs"})
            b[(k[0] - 'a') * 26 + (k[1] - 'a')] = true;
        return b;
    }();
    return bits;
}

bool bare_key(std::string_view w) {
    if (w.size() < 2 || w.size() > 7 || w[0] < 'a' || w[0] > 'z' || w[1] < 'a' || w[1] > 'z') return false;
    if (!bare_prefixes()[(w[0] - 'a') * 26 + (w[1] - 'a')]) return false;
    return BARE_KEYS.contains(w);
}

RV_RULE("abbreviations", words) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.word(i)) continue;
        // fast path: listed words are at most six letters, and without a period only a few are listed
        size_t len = doc[i].end - doc[i].start;
        bool dot = doc.punct(i + 1, '.') && doc.joined(i + 1);
        if (len > 7 || (!dot && !bare_key(doc.low(i))) || !clean_start(doc, i)) continue;
        i = one_form(doc, out, i);
    }
}

// ------------------------------------------------------------------------------------------------ compass points

// "14 N. Elm St." -> North, "Main St. S." -> South, "S. Dakota" -> South, "Main St NE" -> Northeast;
// "Robert E. Lee" and "the 2026 Arlo SE" stay.
const rv::WordSet DIRECTION_REGIONS{"africa", "african", "america", "american", "asia", "asian", "atlantic",
                                    "carolina", "dakota", "europe", "european", "ireland", "korea", "korean",
                                    "pacific", "pole", "sudan", "virginia", "wales"};
const rv::WordSet DIRECTION_VERBS{"blocks", "drive", "driving", "facing", "go", "going", "head", "heading",
                                  "headed", "miles", "move", "moves", "moving", "proceed", "travel", "traveling",
                                  "turn", "turned", "turning", "walk", "walking", "drove", "flowing", "flows",
                                  "runs", "running", "veer", "veering", "bear", "continue", "continuing"};
const rv::WordSet AREA_AFTER{"coast", "corner", "direction", "door", "edge", "end", "entrance", "exit", "gate",
                             "part", "quadrant", "region", "section", "side", "suburbs", "wing"};

std::string compass(std::string_view low) {
    if (low == "n") return "North";
    if (low == "s") return "South";
    if (low == "e") return "East";
    if (low == "w") return "West";
    if (low == "ne") return "Northeast";
    if (low == "nw") return "Northwest";
    if (low == "se") return "Southeast";
    if (low == "sw") return "Southwest";
    return "";
}

bool street_word_at(const Doc& d, size_t k) { return d.word(k) && STREET_WORDS.contains(d.low(k)); }

// Words between a compass letter and the road it names: "Go N on Rte. 9".
const rv::WordSet ROUTE_PREPS{"along", "on", "onto", "via"};
const rv::WordSet ROAD_CUES{"hwy", "highway", "i", "interstate", "route", "rte", "sr", "us"};

bool street_ahead(const Doc& d, size_t k);

// A road starts at token k: a street name ("Elm St.") or a numbered road ("Rte. 9", "Hwy 20", "I-5").
bool road_ahead(const Doc& d, size_t k) {
    if (street_ahead(d, k)) return true;
    if (!d.word(k) || !d[k].space_before || !ROAD_CUES.contains(d.low(k)) || d[k].casing == rv::LOWER) return false;
    size_t n = k + 1;
    if (d.punct(n, '.') && d.joined(n)) ++n;
    if (d.punct(n, '-') && d.joined(n)) ++n;
    return d.number(n);
}

// A street name starting at token k: one to three capitalized words or an ordinal ("14th"), then a street word.
bool street_ahead(const Doc& d, size_t k) {
    for (int n = 0; n < 4 && k < d.size(); ++n) {
        if (n > 0 && street_word_at(d, k) && upper_first(d, k)) return true;
        if (d.number(k) && d.word(k + 1) && d.joined(k + 1) && ORDINAL_SUFFIX.contains(d.low(k + 1))) {
            k += 2;
            continue;
        }
        if (!upper_first(d, k)) return false;
        ++k;
    }
    return false;
}

RV_RULE("abbreviations", directions) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.word(i) || doc[i].casing != rv::UPPER || doc.str(i).size() > 2 || !clean_start(doc, i)) continue;
        std::string_view low = doc.low(i);
        std::string words = compass(low);
        if (words.empty()) continue;
        bool period = doc.punct(i + 1, '.') && doc.joined(i + 1);
        // dotted two-letter points: "1600 Pennsylvania Ave. N.W." Northwest
        size_t last_letter = i;
        if (period && low.size() == 1 && doc.word(i + 2) && doc.joined(i + 2) && doc.str(i + 2).size() == 1 &&
            doc[i + 2].casing == rv::UPPER && doc.punct(i + 3, '.') && doc.joined(i + 3)) {
            std::string two = std::string(low) + std::string(doc.low(i + 2));
            std::string w2 = compass(two);
            if (w2.empty() || two == "ns" || two == "ew") continue;
            words = w2;
            last_letter = i + 2;
        }
        size_t k = period ? last_letter + 2 : i + 1;
        if (k < doc.size() && doc.joined(k) && (doc.word(k) || doc.number(k))) continue;
        size_t p = i > 0 ? i - 1 : Doc::npos;  // the token right before
        if (p != Doc::npos && !doc[i].space_before) continue;
        size_t pw = doc.prev_word(i);  // the word before, across a period and a comma
        bool after_street = pw != Doc::npos && street_word_at(doc, pw) && upper_first(doc, pw) && i - pw <= 3;
        for (size_t t = pw + 1; after_street && t < i; ++t)
            after_street = doc.punct(t, '.') || doc.punct(t, ',');
        // "Highway 101 W.", "Hwy. 101 NE"
        bool after_road_number = false;
        if (p != Doc::npos && doc.number(p)) {
            size_t q = doc.prev_word(p);
            after_road_number = q != Doc::npos && (doc.is(q, "hwy") || doc.is(q, "highway") || doc.is(q, "route") ||
                                                   doc.is(q, "rte") || doc.is(q, "interstate"));
            // "off I-95 N.", "Take US-101 S."
            if (p >= 2 && doc.joined(p) && doc.punct(p - 1, '-') && doc.joined(p - 1) &&
                (doc.str(p - 2) == "I" || doc.str(p - 2) == "US" || doc.str(p - 2) == "SR"))
                after_road_number = true;
        }
        // a direction verb before, maybe with an adverb between: "turning sharply NE"
        size_t verb = p != Doc::npos && doc.word(p) && lower_first(doc, p) ? p : Doc::npos;
        if (verb != Doc::npos && doc.str(verb).size() > 4 && doc.low(verb).substr(doc.str(verb).size() - 2) == "ly" &&
            verb > 0 && doc.word(verb - 1) && doc[verb].space_before)
            verb = verb - 1;
        bool after_verb = verb != Doc::npos && DIRECTION_VERBS.contains(doc.low(verb));
        bool yes = false;
        if (last_letter != i) {
            yes = after_street || after_road_number || after_verb || (p != Doc::npos && doc.number(p));
        } else if (low.size() == 1) {
            if (!period) {
                // "the 101 N Fwy.", "the I-5 N"; "Go N on Rte. 9"
                yes = p != Doc::npos && doc.number(p) &&
                      (street_word_at(doc, k) ||
                       (p >= 2 && doc.joined(p) && doc.punct(p - 1, '-') && doc.joined(p - 1) && doc.is(p - 2, "i")));
                if (!yes && p != Doc::npos && doc.word(p) && doc.word(k) && doc[k].space_before &&
                    ROUTE_PREPS.contains(doc.low(k)) && road_ahead(doc, k + 1))
                    yes = true;
            } else if (doc.word(k) && doc[k].space_before && DIRECTION_REGIONS.contains(doc.low(k))) {
                yes = true;  // "S. Dakota"
                after_street = after_road_number = false;
            } else if (after_street || after_road_number) {
                yes = true;  // "Main St. S.", "Highway 101 W."
            } else if (p != Doc::npos && doc.number(p) &&
                       (street_ahead(doc, k) || (street_word_at(doc, k) && upper_first(doc, k)))) {
                yes = true;  // "14 N. Elm St.", "the 101 N. Fwy.", not "650 W. Peak power"
            } else if (p != Doc::npos && doc.word(p) && DIRECTION_PREPS.contains(doc.low(p)) &&
                       lower_first(doc, p) && street_ahead(doc, k)) {
                yes = true;  // "onto W. Elm Street"
            } else if (p != Doc::npos && doc.word(p) && DIRECTION_VERBS.contains(doc.low(p)) &&
                       lower_first(doc, p) && !upper_first(doc, k)) {
                yes = true;  // "head N. on Elm Street"
            } else if (pw != Doc::npos && DISTANCE_UNITS.contains(doc.low(pw)) && lower_first(doc, pw) &&
                       doc.number(doc.prev_word(pw)) && (pw == p || (pw + 1 == p && doc.punct(p, '.')))) {
                yes = true;  // "0.5 mi. W. to the old mill"
            } else if (p != Doc::npos && doc.is(k, "of") && (lower_first(doc, p) || doc.number(p))) {
                yes = true;  // "just W. of the old mill"
            } else if (doc.word(k) && doc[k].space_before && AREA_AFTER.contains(doc.low(k))) {
                yes = true;  // "on the E. side", "E. Wing"
            }
        } else {
            if (after_street) {
                yes = true;  // "Main St NE", "Elm St., SE"
            } else if (after_road_number) {
                yes = true;  // "Hwy. 101 NE"
            } else if (after_verb || (p != Doc::npos && doc.word(p) && DIRECTION_VERBS.contains(doc.low(p)))) {
                yes = true;  // "heading NE", "moves SE from", "turning sharply NE"
            } else if (p != Doc::npos && doc.is(p, "the") && doc.word(k) && AREA_AFTER.contains(doc.low(k))) {
                yes = true;  // "the SE quadrant"
            } else if (p != Doc::npos && doc.is(k, "of") && doc[k].space_before &&
                       ((lower_first(doc, p) && !doc.is(p, "the")) || doc.number(p) ||
                        (doc.punct(p, '.') && pw != Doc::npos && DISTANCE_UNITS.contains(doc.low(pw))))) {
                yes = true;  // "40 miles NE of Denver", "just SW of the river", "40 mi. NE of Boise"
            }
        }
        if (!yes) continue;
        // before the street name the period goes ("N. Elm St."); after it, it may end the sentence ("Main St. S.")
        size_t dot = last_letter + 1;
        bool keep = !period || (low.size() == 2) ||
                    period_ends_sentence(doc, dot, after_street || after_road_number ? End::STRONG : End::NEVER);
        // (the period of a unit before it, "0.5 mi. W.", goes with the unit's reading)
        out.tokens(i, keep ? last_letter : dot, words, 78);
        i = last_letter;
    }
}

// ------------------------------------------------------------------------------------------------ short forms

// Numeronyms: written-only shorthand that cannot be said as written.
const rv::PhraseMap NUMERONYMS{
    {"a11n", "authorization"},      {"a11y", "accessibility"},        {"a12n", "authentication"},
    {"c11n", "customization"},      {"c14n", "canonicalization"},     {"e2e", "end to end"},
    {"f2f", "face to face"},        {"g10n", "globalization"},        {"p11n", "personalization"},
    {"s10n", "sanitization"},
    {"g11n", "globalization"},      {"i18n", "internationalization"}, {"k8s", "Kubernetes"},
    {"l10n", "localization"},       {"m17n", "multilingualization"},  {"o11y", "observability"},
    {"p13n", "personalization"},    {"s11n", "serialization"},        {"t9n", "translation"},
    {"v12n", "virtualization"},
};

// Business shorthand said in full; exact spellings.
const rv::PhraseMap BUSINESS_SHORT{
    {"YoY", "year over year"},       {"yoy", "year over year"},       {"YOY", "year over year"},
    {"QoQ", "quarter over quarter"}, {"qoq", "quarter over quarter"}, {"QOQ", "quarter over quarter"},
    {"MoM", "month over month"},     {"HoH", "half over half"},       {"YTD", "year to date"},
    {"ytd", "year to date"},         {"MTD", "month to date"},        {"QTD", "quarter to date"},
    {"WTD", "week to date"},         {"TTM", "trailing twelve months"},
};
// Only after a number or a change ("7% WoW") or before a word for a figure; alone they are names (World of
// Warcraft, the DoD).
const rv::PhraseMap BUSINESS_AFTER_NUMBER{{"WoW", "week over week"}, {"DoD", "day over day"}};
const rv::WordSet FIGURE_WORDS{"basis", "change", "churn", "comparison", "decline", "decrease", "drop", "figures",
                               "gain", "growth", "increase", "metric", "metrics", "numbers", "revenue", "rise",
                               "sales", "trading"};

const rv::PhraseMap WORLD_WARS{{"WWI", "World War One"}, {"WWII", "World War Two"}, {"WWIII", "World War Three"}};

// A number, a percent sign or a percentage word in the two tokens before i.
bool numeric_before(const Doc& d, size_t i) {
    for (size_t back = 1; back <= 2 && back <= i; ++back) {
        size_t t = i - back;
        if (d.number(t) || d.punct(t, '%') || d.is(t, "percent") || d.is(t, "pp") || d.is(t, "bps") ||
            d.is(t, "points"))
            return true;
    }
    return false;
}

RV_RULE("abbreviations", short_forms) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.word(i)) continue;
        // the second of "l10n/i18n" follows a slash
        if (!clean_start(doc, i) &&
            !(i >= 4 && doc.punct(i - 1, '/') && doc.joined(i - 1) && doc.joined(i) && doc.word(i - 2) &&
              doc.number(i - 3) && doc.joined(i - 2) && doc.word(i - 4) && doc.joined(i - 3)))
            continue;
        std::string_view w = doc.str(i);
        std::string_view low = doc.low(i);
        // numeronyms: letters, a number, letters, glued together ("k8s", "i18n")
        if (doc.number(i + 1) && doc.joined(i + 1) && doc.word(i + 2) && doc.joined(i + 2)) {
            // "k8s", "K8s", "K8S", "I18N"; not "a11Y"
            bool caps = doc[i].casing == rv::UPPER && doc[i + 2].casing == rv::UPPER;
            if (doc.str(i).size() == 1 && (lower_first(doc, i + 2) || caps)) {
                std::string key(doc.lower.data() + doc[i].start, doc[i + 2].end - doc[i].start);
                const std::string_view* s = NUMERONYMS.find(key);
                // "l10n/i18n": a slash between two of them
                bool slash_pair = doc.punct(i + 3, '/') && doc.joined(i + 3) && doc.word(i + 4) && doc.joined(i + 4) &&
                                  doc.number(i + 5) && doc.joined(i + 5);
                if (s && (clean_end(doc, i + 2) || slash_pair)) out.tokens(i, i + 2, std::string(*s), 80);
            }
            i += 2;
            continue;
        }
        // "WW1", "WW2"
        if (w == "WW" && doc.number(i + 1) && doc.joined(i + 1)) {
            std::string_view n = doc.str(i + 1);
            const char* war = n == "1" ? "World War One" : n == "2" ? "World War Two" : nullptr;
            if (war && clean_end(doc, i + 1)) out.tokens(i, i + 1, war, 80);
            ++i;
            continue;
        }
        // slash forms: "n/a", "N/A", "c/o" ("w/", "w/o", "b/c" are chat shorthand: structure.slang)
        if (w.size() == 1 && doc.punct(i + 1, '/') && doc.joined(i + 1)) {
            bool second = doc.word(i + 2) && doc.joined(i + 2) && doc.str(i + 2).size() == 1;
            std::string_view b = second ? doc.str(i + 2) : std::string_view();
            const char* spoken = nullptr;
            size_t last = i + 2;
            if (second && clean_end(doc, i + 2)) {
                if ((w == "n" && b == "a") || (w == "N" && (b == "A" || b == "a"))) spoken = "not available";
                else if (w == "c" && b == "o") spoken = "care of";
            }
            if (spoken) {
                out.tokens(i, last, spoken, 80);
                i = last;
            }
            continue;
        }
        // "England v France", "R v Smith": a bare "v" between two names
        if (w == "v" && i > 0 && doc[i].space_before && doc.word(i - 1) && rv::is_upper(doc[i - 1].cp) &&
            doc[i - 1].casing != rv::MIXED && doc.word(i + 1) && doc[i + 1].space_before &&
            !doc[i + 1].newline_before && upper_first(doc, i + 1) && doc.str(i + 1).size() > 1) {
            out.tokens(i, i, "versus", 70);
            continue;
        }
        if (w.size() < 2 || w.size() > 9) continue;
        switch (w[0]) {
            case 'Y': case 'Q': case 'M': case 'H': case 'W': case 'D': case 'T': case 'y': case 'q': case 'a':
            case 'b': case 'p':
                break;
            default:
                continue;
        }
        // a possessive stays: "WWII's" -> "World War Two's"
        std::string_view poss;
        for (std::string_view suffix : {std::string_view("'s"), std::string_view("\xE2\x80\x99s")})
            if (w.size() > suffix.size() + 1 && w.substr(w.size() - suffix.size()) == suffix) {
                poss = w.substr(w.size() - suffix.size());
                w.remove_suffix(suffix.size());
            }
        // first letters and lengths of the forms below
        bool candidate = false;
        switch (w[0]) {
            case 'Y': case 'Q': case 'M': case 'H': case 'W': case 'D': case 'T':
                candidate = w.size() >= 3 && w.size() <= 5;
                break;
            case 'y': case 'q': case 'a': case 'b':
                candidate = w.size() == 3;
                break;
            case 'p':
                candidate = w.size() == 2;
                break;
            default:
                break;
        }
        if (!candidate || !clean_end(doc, i)) continue;
        if (const std::string_view* s = BUSINESS_SHORT.find(w)) {
            out.tokens(i, i, std::string(*s) + std::string(poss), 80);
        } else if (const std::string_view* s2 = BUSINESS_AFTER_NUMBER.find(w)) {
            if (numeric_before(doc, i) || (doc.word(i + 1) && FIGURE_WORDS.contains(doc.low(i + 1))))
                out.tokens(i, i, std::string(*s2) + std::string(poss), 80);
        } else if (const std::string_view* s3 = WORLD_WARS.find(w)) {
            out.tokens(i, i, std::string(*s3) + std::string(poss), 80);
        } else if (w == "aka") {
            out.tokens(i, i, "AKA", 80);
        } else if ((low == "pp" || low == "bps") && w == low && i > 0 && doc.number(i - 1) &&
                   !(doc.punct(i + 1, '.') && doc.number(i + 2))) {
            // "3.2 pp" percentage points, "25 bps" basis points; singular after a plain "1"
            bool one = doc.str(i - 1) == "1" && !(i >= 2 && doc.punct(i - 2, '.') && doc.joined(i - 1));
            // before a noun after "a": "a 3.5 pp margin lift" -> "percentage point"
            size_t t = rv::parse_number(doc, i - 1).first;  // the number starts at or before i - 1
            for (size_t b = i - 1; b > 0 && (doc.number(b - 1) || doc.punct(b - 1, '.') || doc.punct(b - 1, ',')) &&
                                   doc.joined(b);
                 --b)
                t = b - 1;
            while (t > 0 && doc.joined(t) && (doc.punct(t - 1, '+') || doc.punct(t - 1, '-') ||
                                              doc.punct(t - 1, 0x2212) || doc.punct(t - 1, '~')))
                --t;
            if (t > 0 && (doc.is(t - 1, "a") || doc.is(t - 1, "an")) && doc.word(i + 1) && lower_first(doc, i + 1))
                one = true;
            std::string s = w == "pp" ? "percentage point" : "basis point";
            out.tokens(i, i, one ? s : s + "s", 70);
        } else if (w == "bps" && !(i > 0 && doc.number(i - 1))) {
            out.tokens(i, i, "basis points", 70);  // "in bps", "the bps impact"
        }
    }
}

}  // namespace
