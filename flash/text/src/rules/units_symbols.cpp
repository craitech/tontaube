// Rules of the module "units_symbols": measurement units after numbers. Symbols, powers and science forms are
// in units_symbols/*.cpp. See README.md for the conventions.
//
// A measure is proposed in two parts that compose with the other modules: the number (with its sign) at a low
// score, so that numbers_core and numbers_context keep their own readings of ranges, dimensions and
// approximations, and the unit at a high score. The unit chooses singular or plural from the number before it
// and the word after it ('a 24 GB card', 'one kilogram', 'five kilograms').
#include "rules/units_symbols/common.h"

namespace {

using namespace us;

enum : uint16_t {
    LEN = 1,      // a length: takes 'square' and 'cubic'
    TEMP = 2,     // keeps the plural before a noun: temperatures, frames per second
    INV = 4,      // said the same after any number (RPM, DPI, IU)
    SPACED = 8,   // only after a space (K, t, J, Pa)
    CTX = 16,     // needs a context check (in, m joined, A, W, bar, Mt)
    DOT = 32,     // an abbreviation that may carry a period (lbs., oz.)
    TIME = 64,    // a time unit: only as the denominator of a rate (time readings belong to datetime)
    NOBARE = 128,  // never read without a number
    PART = 256     // only inside a compound unit (N·m, MT/s)
};

struct UnitDef {
    std::string_view key, sg, pl;  // pl empty: sg + "s"
    uint16_t flags;
};

const UnitDef UNIT_LIST[] = {
    // length
    {"nm", "nanometer", "", LEN}, {"µm", "micrometer", "", LEN}, {"μm", "micrometer", "", LEN},
    {"mm", "millimeter", "", LEN}, {"cm", "centimeter", "", LEN}, {"dm", "decimeter", "", LEN},
    {"m", "meter", "", LEN | CTX | NOBARE}, {"km", "kilometer", "", LEN}, {"Km", "kilometer", "", LEN | NOBARE},
    {"Å", "angstrom", "", LEN | NOBARE},
    {"in", "inch", "inches", LEN | CTX | DOT | NOBARE}, {"ft", "foot", "feet", LEN | DOT},
    {"yd", "yard", "", LEN | DOT}, {"yds", "yard", "yards", LEN | DOT}, {"mi", "mile", "", LEN | DOT | NOBARE},
    // mass
    {"mg", "milligram", "", 0}, {"µg", "microgram", "", 0}, {"μg", "microgram", "", 0}, {"mcg", "microgram", "", 0},
    {"ng", "nanogram", "", 0}, {"g", "gram", "", NOBARE}, {"kg", "kilogram", "", 0}, {"kgs", "kilogram", "kilograms", 0},
    {"Kg", "kilogram", "", NOBARE}, {"t", "tonne", "", SPACED | NOBARE}, {"Mt", "megatonne", "", CTX | NOBARE},
    {"Gt", "gigatonne", "", NOBARE}, {"lb", "pound", "", DOT}, {"lbs", "pound", "pounds", DOT},
    {"oz", "ounce", "", DOT},
    // volume
    {"mL", "milliliter", "", 0}, {"ml", "milliliter", "", 0}, {"µL", "microliter", "", 0}, {"μL", "microliter", "", 0},
    {"L", "liter", "", NOBARE}, {"dL", "deciliter", "", 0}, {"cL", "centiliter", "", 0},
    {"l", "liter", "", NOBARE | CTX}, {"dl", "deciliter", "", NOBARE}, {"cl", "centiliter", "", NOBARE},
    {"gal", "gallon", "", DOT}, {"qt", "quart", "", DOT}, {"tsp", "teaspoon", "", DOT}, {"tbsp", "tablespoon", "", DOT},
    {"Tbsp", "tablespoon", "", DOT},
    // area
    {"ha", "hectare", "", NOBARE}, {"sqft", "square foot", "square feet", 0},
    // data
    {"KB", "kilobyte", "", 0}, {"kB", "kilobyte", "", 0}, {"MB", "megabyte", "", 0}, {"GB", "gigabyte", "", 0},
    {"TB", "terabyte", "", 0}, {"PB", "petabyte", "", 0}, {"EB", "exabyte", "", NOBARE},
    {"KiB", "kibibyte", "", 0}, {"MiB", "mebibyte", "", 0}, {"GiB", "gibibyte", "", 0}, {"TiB", "tebibyte", "", 0},
    {"PiB", "pebibyte", "", 0}, {"Kb", "kilobit", "", NOBARE}, {"kb", "kilobit", "", NOBARE},
    {"Mb", "megabit", "", NOBARE}, {"Gb", "gigabit", "", NOBARE}, {"Tb", "terabit", "", NOBARE},
    {"kbit", "kilobit", "", 0}, {"Kbit", "kilobit", "", 0}, {"Mbit", "megabit", "", 0}, {"Gbit", "gigabit", "", 0},
    // data rates
    {"kbps", "kilobit per second", "kilobits per second", 0}, {"Kbps", "kilobit per second", "kilobits per second", 0},
    {"Mbps", "megabit per second", "megabits per second", 0}, {"Gbps", "gigabit per second", "gigabits per second", 0},
    {"Tbps", "terabit per second", "terabits per second", 0}, {"MBps", "megabyte per second", "megabytes per second", 0},
    {"GBps", "gigabyte per second", "gigabytes per second", 0},
    // frequency, rates
    {"Hz", "hertz", "hertz", 0}, {"kHz", "kilohertz", "kilohertz", 0}, {"MHz", "megahertz", "megahertz", 0},
    {"GHz", "gigahertz", "gigahertz", 0}, {"THz", "terahertz", "terahertz", 0},
    {"rpm", "RPM", "RPM", INV}, {"dpi", "DPI", "DPI", INV}, {"ppi", "PPI", "PPI", INV},
    {"fps", "frame per second", "frames per second", TEMP}, {"px", "pixel", "", 0},
    // power, energy
    {"W", "watt", "", CTX | NOBARE}, {"mW", "milliwatt", "", 0}, {"µW", "microwatt", "", 0}, {"kW", "kilowatt", "", 0},
    {"MW", "megawatt", "", 0}, {"GW", "gigawatt", "", 0}, {"TW", "terawatt", "", NOBARE},
    {"Wh", "watt hour", "", 0}, {"mWh", "milliwatt hour", "", 0}, {"kWh", "kilowatt hour", "", 0},
    {"MWh", "megawatt hour", "", 0}, {"GWh", "gigawatt hour", "", 0}, {"TWh", "terawatt hour", "", 0},
    {"J", "joule", "", SPACED | NOBARE}, {"kJ", "kilojoule", "", 0}, {"MJ", "megajoule", "", 0},
    {"GJ", "gigajoule", "", 0}, {"cal", "calorie", "", NOBARE}, {"kcal", "kilocalorie", "", 0},
    {"Kcal", "kilocalorie", "", 0}, {"eV", "electronvolt", "", 0}, {"keV", "kiloelectronvolt", "", 0},
    {"MeV", "megaelectronvolt", "", 0}, {"GeV", "gigaelectronvolt", "", 0}, {"TeV", "teraelectronvolt", "", 0},
    {"Btu", "British thermal unit", "", 0}, {"BTU", "British thermal unit", "", NOBARE},
    {"hp", "horsepower", "horsepower", NOBARE},
    // electrical, sound
    {"V", "volt", "", NOBARE}, {"mV", "millivolt", "", 0}, {"µV", "microvolt", "", 0}, {"kV", "kilovolt", "", 0},
    {"A", "amp", "", CTX | SPACED | NOBARE}, {"mA", "milliamp", "", 0}, {"µA", "microamp", "", 0},
    {"μA", "microamp", "", 0}, {"kA", "kiloamp", "", 0}, {"mAh", "milliamp hour", "", 0}, {"Ah", "amp hour", "", NOBARE},
    {"Ω", "ohm", "", 0}, {"kΩ", "kilohm", "", 0}, {"MΩ", "megohm", "", 0}, {"mΩ", "milliohm", "", 0},
    {"µF", "microfarad", "", 0}, {"μF", "microfarad", "", 0}, {"nF", "nanofarad", "", 0}, {"pF", "picofarad", "", 0},
    {"mH", "millihenry", "millihenries", 0}, {"µH", "microhenry", "microhenries", 0},
    {"dB", "decibel", "", 0},
    // pressure
    {"Pa", "pascal", "", SPACED | NOBARE}, {"kPa", "kilopascal", "", 0}, {"MPa", "megapascal", "", 0},
    {"GPa", "gigapascal", "", 0}, {"hPa", "hectopascal", "", 0}, {"bar", "bar", "bar", CTX | NOBARE},
    {"mbar", "millibar", "", 0}, {"atm", "atmosphere", "", 0}, {"psi", "PSI", "PSI", INV}, {"PSI", "PSI", "PSI", INV},
    {"mmHg", "millimeter of mercury", "millimeters of mercury", 0},
    {"inHg", "inch of mercury", "inches of mercury", 0}, {"Torr", "torr", "torr", 0},
    // temperature (the degree sign is parsed apart)
    {"K", "kelvin", "kelvin", TEMP | SPACED | CTX | NOBARE}, {"ºC", "degree Celsius", "degrees Celsius", TEMP},
    {"ºF", "degree Fahrenheit", "degrees Fahrenheit", TEMP},
    // speed
    {"kph", "kilometer per hour", "kilometers per hour", 0}, {"mph", "mile per hour", "miles per hour", 0},
    // amount, concentration, medicine
    {"ppm", "parts per million", "parts per million", INV}, {"ppb", "parts per billion", "parts per billion", INV},
    {"mol", "mole", "", 0}, {"mmol", "millimole", "", 0}, {"µmol", "micromole", "", 0}, {"mM", "millimolar", "millimolar", 0},
    {"µM", "micromolar", "micromolar", 0}, {"nM", "nanomolar", "nanomolar", 0}, {"IU", "IU", "IU", INV | NOBARE},
    {"mEq", "milliequivalent", "", 0},
    // radiation, light, force
    {"Gy", "gray", "", NOBARE}, {"mGy", "milligray", "", 0}, {"Sv", "sievert", "", 0}, {"mSv", "millisievert", "", 0},
    {"µSv", "microsievert", "", 0}, {"μSv", "microsievert", "", 0}, {"Bq", "becquerel", "", 0},
    {"lm", "lumen", "", NOBARE}, {"lx", "lux", "lux", NOBARE}, {"kN", "kilonewton", "", 0}, {"Nm", "newton meter", "", 0},
    // fractions of a second are measures; the other time units only as the denominator of a rate
    {"ms", "millisecond", "", 0}, {"µs", "microsecond", "", 0}, {"μs", "microsecond", "", 0},
    {"ns", "nanosecond", "", 0},
    {"s", "second", "", TIME}, {"sec", "second", "", TIME},
    {"min", "minute", "", TIME}, {"h", "hour", "", TIME}, {"hr", "hour", "", TIME}, {"hrs", "hour", "", TIME},
    {"d", "day", "", TIME}, {"wk", "week", "", TIME}, {"mo", "month", "", TIME}, {"yr", "year", "", TIME},
    {"msec", "millisecond", "", 0}, {"msecs", "millisecond", "milliseconds", 0}, {"usec", "microsecond", "", 0},
    {"nsec", "nanosecond", "", 0},
    // spellings of informal and older writing, ASCII 'u' for micro; only after a number
    {"KG", "kilogram", "", NOBARE}, {"Kgs", "kilogram", "kilograms", NOBARE}, {"KGs", "kilogram", "kilograms", NOBARE},
    {"KGS", "kilogram", "kilograms", NOBARE}, {"LB", "pound", "", DOT | NOBARE}, {"Lb", "pound", "", DOT | NOBARE},
    {"LBS", "pound", "pounds", DOT | NOBARE}, {"Lbs", "pound", "pounds", DOT | NOBARE},
    {"OZ", "ounce", "", DOT | NOBARE}, {"Oz", "ounce", "", DOT | NOBARE}, {"ozs", "ounce", "ounces", DOT | NOBARE},
    {"KM", "kilometer", "", LEN | NOBARE}, {"kms", "kilometer", "kilometers", LEN | NOBARE},
    {"Kms", "kilometer", "kilometers", LEN | NOBARE}, {"CM", "centimeter", "", LEN | NOBARE},
    {"cms", "centimeter", "centimeters", LEN | NOBARE},
    {"ML", "milliliter", "", NOBARE}, {"Ml", "milliliter", "", NOBARE}, {"mls", "milliliter", "milliliters", NOBARE},
    {"MG", "milligram", "", NOBARE}, {"mgs", "milligram", "milligrams", NOBARE}, {"gm", "gram", "", NOBARE},
    {"gms", "gram", "grams", NOBARE}, {"ltr", "liter", "", NOBARE},
    {"ltrs", "liter", "liters", NOBARE}, {"Ltr", "liter", "", NOBARE}, {"gals", "gallon", "gallons", DOT},
    {"qts", "quart", "quarts", DOT}, {"tsps", "teaspoon", "teaspoons", DOT},
    {"tbsps", "tablespoon", "tablespoons", DOT},
    {"Tbsps", "tablespoon", "tablespoons", DOT}, {"TBSP", "tablespoon", "", DOT | NOBARE},
    {"TSP", "teaspoon", "", DOT | NOBARE}, {"Tsp", "teaspoon", "", DOT | NOBARE},
    {"tbs", "tablespoon", "", DOT | NOBARE},
    {"Tbs", "tablespoon", "", DOT | NOBARE},
    {"ug", "microgram", "", NOBARE}, {"uL", "microliter", "", NOBARE}, {"µl", "microliter", "", 0},
    {"μl", "microliter", "", 0}, {"um", "micrometer", "", LEN | NOBARE | CTX}, {"uA", "microamp", "", NOBARE},
    {"uV", "microvolt", "", NOBARE}, {"μV", "microvolt", "", 0}, {"uF", "microfarad", "", NOBARE},
    {"uH", "microhenry", "microhenries", NOBARE}, {"μH", "microhenry", "microhenries", 0},
    {"umol", "micromole", "", NOBARE}, {"μmol", "micromole", "", 0}, {"uM", "micromolar", "micromolar", NOBARE},
    {"μM", "micromolar", "micromolar", 0}, {"uSv", "microsievert", "", NOBARE}, {"uW", "microwatt", "", NOBARE},
    {"μW", "microwatt", "", 0}, {"pg", "picogram", "", NOBARE}, {"nmol", "nanomole", "", 0},
    {"pmol", "picomole", "", 0},
    {"MPH", "mile per hour", "miles per hour", NOBARE}, {"Mph", "mile per hour", "miles per hour", NOBARE},
    {"KPH", "kilometer per hour", "kilometers per hour", NOBARE},
    {"kmh", "kilometer per hour", "kilometers per hour", NOBARE},
    {"KMH", "kilometer per hour", "kilometers per hour", NOBARE},
    {"mbps", "megabit per second", "megabits per second", NOBARE},
    {"gbps", "gigabit per second", "gigabits per second", NOBARE}, {"gb", "gigabyte", "", NOBARE},
    {"tb", "terabyte", "", NOBARE}, {"Ghz", "gigahertz", "gigahertz", NOBARE},
    {"GHZ", "gigahertz", "gigahertz", NOBARE},
    {"ghz", "gigahertz", "gigahertz", NOBARE}, {"Mhz", "megahertz", "megahertz", NOBARE},
    {"MHZ", "megahertz", "megahertz", NOBARE}, {"mhz", "megahertz", "megahertz", NOBARE},
    {"KHz", "kilohertz", "kilohertz", NOBARE}, {"Khz", "kilohertz", "kilohertz", NOBARE},
    {"khz", "kilohertz", "kilohertz", NOBARE}, {"hz", "hertz", "hertz", NOBARE}, {"HZ", "hertz", "hertz", NOBARE},
    {"KW", "kilowatt", "", NOBARE}, {"Kw", "kilowatt", "", NOBARE}, {"kw", "kilowatt", "", NOBARE},
    {"KWh", "kilowatt hour", "", NOBARE}, {"Kwh", "kilowatt hour", "", NOBARE}, {"kwh", "kilowatt hour", "", NOBARE},
    {"KWH", "kilowatt hour", "", NOBARE}, {"MAH", "milliamp hour", "", NOBARE}, {"mah", "milliamp hour", "", NOBARE},
    {"deg", "degree", "", NOBARE}, {"degs", "degree", "degrees", NOBARE}, {"º", "degree", "", NOBARE},
    // said as letters, like RPM and DPI
    {"bpm", "BPM", "BPM", INV | NOBARE}, {"cc", "CC", "CC", INV | NOBARE}, {"cfm", "CFM", "CFM", INV | NOBARE},
    {"FPS", "frame per second", "frames per second", TEMP | NOBARE},
    {"TFLOPS", "teraflops", "teraflops", INV}, {"PFLOPS", "petaflops", "petaflops", INV},
    {"GFLOPS", "gigaflops", "gigaflops", INV},
    {"sqm", "square meter", "square meters", NOBARE}, {"nmi", "nautical mile", "", NOBARE},
    {"bbl", "barrel", "", NOBARE}, {"bpd", "barrel per day", "barrels per day", NOBARE},
    // newtons in a compound or after a force word; transfers only in a rate ('6400 MT/s')
    {"N", "newton", "", CTX | SPACED | NOBARE}, {"MT", "megatransfer", "", PART}, {"GT", "gigatransfer", "", PART},
    {"mpg", "mile per gallon", "miles per gallon", NOBARE}, {"MPG", "mile per gallon", "miles per gallon", NOBARE},
    {"kgf", "kilogram-force", "kilograms-force", 0}, {"lbf", "pound-force", "pounds-force", 0},
};

class UnitTable {
public:
    UnitTable() : items_(std::begin(UNIT_LIST), std::end(UNIT_LIST)) {
        std::sort(items_.begin(), items_.end(), [](const UnitDef& a, const UnitDef& b) { return a.key < b.key; });
    }
    const UnitDef* find(std::string_view k) const {
        auto it = std::lower_bound(items_.begin(), items_.end(), k,
                                   [](const UnitDef& a, std::string_view v) { return a.key < v; });
        return it != items_.end() && it->key == k ? &*it : nullptr;
    }

private:
    std::vector<UnitDef> items_;
};

const UnitTable& units() {
    static const UnitTable t;
    return t;
}

// A length unit in any letter case, after 'sq' or 'cu' ('Sq. Ft.', 'SQ M').
const UnitDef* length_unit(std::string_view w) {
    if (const UnitDef* u = units().find(w)) return u;
    char buf[8];
    if (w.size() >= sizeof buf) return nullptr;
    for (size_t k = 0; k < w.size(); ++k) buf[k] = w[k] >= 'A' && w[k] <= 'Z' ? static_cast<char>(w[k] + 32) : w[k];
    return units().find(std::string_view(buf, w.size()));
}

std::string plural_of(const UnitDef& u) { return u.pl.empty() ? std::string(u.sg) + "s" : std::string(u.pl); }

// Plain words that may stand as the denominator of a rate ('kg/day', 'mg/serving').
const rv::WordSet DEN_WORDS{"second", "minute", "hour", "day", "week", "month", "year", "night", "person", "capita",
                            "user", "patient", "visit", "unit", "serving", "dose", "cycle", "shift", "trip",
                            "seat", "node", "core", "gpu", "rack", "site", "head", "hectare", "acre", "liter",
                            "litre", "gallon", "mile", "meter", "metre", "page", "query", "request", "call",
                            "decade", "century", "lap", "game", "match", "season", "student", "employee",
                            "household", "ton", "tonne", "barrel", "pack", "bottle", "item", "piece", "share",
                            "room", "bed", "flight", "passenger", "vehicle", "car", "kid", "child", "adult",
                            "word", "token", "image", "frame", "sample", "run", "step", "epoch", "batch"};

// Words after a unit that show the number and unit do not modify a noun ('5 kg of', '3 m wide').
const rv::WordSet NOT_NOUN{"a", "about", "above", "according", "across", "add", "adds", "after", "again", "against",
    "ago", "all", "allocated", "along", "already", "also", "although", "altogether", "among", "an", "and",
    "annually", "any", "apart", "apiece", "appear", "appears", "are", "around", "as", "at", "away", "back", "be",
    "because", "become", "becomes", "been", "before", "behind", "being", "below", "beneath", "beside", "besides",
    "better", "between", "beyond", "bigger", "both", "but", "by", "colder", "combined", "compared", "continuous",
    "continuously", "could", "cumulative", "daily", "deep", "deeper", "despite", "diagonally", "did", "do", "does",
    "down", "downhill", "downstream", "dry", "during", "each", "earlier", "east", "eastward", "either", "else",
    "empty", "end", "enough", "equivalent", "even", "every", "excluding", "faster", "fewer", "first", "for", "free",
    "from", "full", "gives", "greater", "gross", "had", "has", "have", "he", "heavier", "her", "here", "high",
    "higher", "his", "hotter", "hourly", "however", "i", "idle", "if", "in", "including", "inside", "instead",
    "into", "is", "it", "its", "just", "larger", "last", "later", "least", "left", "less", "lighter", "like",
    "long", "longer", "lower", "max", "maximum", "may", "me", "might", "min", "minimum", "minus", "monthly", "more",
    "most", "much", "must", "my", "narrower", "near", "nearly", "neither", "net", "new", "next", "no", "nominal",
    "nor", "north", "northeast", "northward", "northwest", "not", "now", "of", "off", "on", "once", "only", "onto",
    "or", "otherwise", "our", "out", "outside", "over", "overall", "overnight", "past", "peak", "per", "plus",
    "rather", "remain", "remained", "remains", "respectively", "right", "rougher", "shall", "she", "shorter",
    "should", "since", "slower", "smaller", "so", "some", "south", "southeast", "southward", "southwest", "square",
    "still", "such", "sustained", "tall", "taller", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "thick", "thicker", "thin", "thinner", "this", "those", "though", "through", "throughout",
    "thus", "till", "times", "to", "today", "together", "tomorrow", "tonight", "too", "total", "toward", "towards",
    "twice", "under", "unless", "until", "up", "upon", "upstream", "us", "using", "versus", "via", "vs", "warmer",
    "was", "we", "weekly", "were", "west", "westward", "wet", "what", "when", "where", "whereas", "whether",
    "which", "while", "who", "whose", "why", "wide", "wider", "will", "with", "within", "without", "worse", "would",
    "x", "yearly", "yesterday", "yet", "you", "your"};
// Verbs that often follow a measure as the subject ('126 mg/dL triggers a repeat test').
const rv::WordSet VERBS_AFTER{"acts", "added", "adds", "allows", "amounts", "applies", "arrives", "became",
    "becomes", "bought", "brings", "brought", "buys", "came", "comes", "contains", "corresponds", "cost", "costs",
    "covers", "dropped", "drops", "enters", "equaled", "equals", "exceeded", "exceeds", "failed", "fails", "falls",
    "fell", "fits", "flows", "gave", "gets", "gives", "goes", "got", "held", "helps", "hits", "holds", "includes",
    "indicates", "keeps", "kept", "lasted", "lasts", "leaves", "left", "lets", "lies", "looks", "made", "makes",
    "marks", "matches", "means", "meant", "measured", "measures", "moves", "needs", "passed", "passes", "powered",
    "powers", "produces", "provides", "pulls", "pushes", "puts", "ran", "reached", "reaches", "reads", "remains",
    "represents", "required", "requires", "rises", "rose", "runs", "sat", "saved", "saves", "seems", "sets",
    "showed", "shows", "sits", "spans", "stands", "stayed", "stays", "suffice", "suffices", "supports", "takes",
    "took", "translates", "travels", "triggered", "triggers", "uses", "weighed", "weighs", "went", "works"};

// Irregular participles and verbs in the base form: after a measure they start a predicate ('2,140 kcal eaten',
// 'frames below 20 ms reduce jitter'), so the unit keeps its plural.
const rv::WordSet PREDICATE{"ate", "became", "become", "began", "begun", "bitten", "blown", "born", "borne",
    "bought", "brought", "built", "burnt", "came", "caught", "cause", "chosen", "contain", "dealt", "did", "done",
    "drawn", "drew", "driven", "drove", "eaten", "enable", "exceed", "fallen", "felt", "flew", "flown",
    "forgotten", "found", "froze", "frozen", "gave", "given", "gone", "gotten", "grew", "grown", "hidden",
    "improve", "indicate", "knew", "known", "laid", "led", "lent", "lost", "matter", "meant", "paid", "produce",
    "provide", "reduce", "remain", "represent", "require", "ridden", "risen", "rode", "said", "sank", "saw",
    "scatter", "seem", "seen", "sent", "shown", "shrank", "shrunk", "sold", "spent", "spoke", "spoken", "stay",
    "stole", "stolen", "struck", "stuck", "suggest", "sunk", "swam", "swum", "taken", "taught", "thrown",
    "told", "took", "tore", "torn", "trigger", "won", "wore", "worn", "written", "wrote"};
// What is measured, after the measure without 'of' ('6 mg/dL glucose', '500 g flour', '2 tsp vanilla').
const rv::WordSet SUBSTANCES{"acetaminophen", "adrenaline", "alcohol", "almonds", "apples", "aspirin", "beans",
    "beef", "berries", "blueberries", "broccoli", "butter", "caffeine", "calcium", "carbohydrate", "carbohydrates",
    "carbs", "carrots", "cheese", "cherries", "chicken", "chickpeas", "cholesterol", "cocoa", "codeine", "cream",
    "dates", "dextrose", "epinephrine", "ethanol", "fat", "fentanyl", "fiber", "fibre", "flour", "garlic",
    "ginger", "glucose", "grapes", "heparin", "honey", "insulin", "iron", "ketamine", "lentils", "magnesium",
    "melatonin", "milk", "mince", "morphine", "mushrooms", "naloxone", "nicotine", "noodles", "nuts", "oats",
    "oil", "onions", "pasta", "peas", "peppers", "potassium", "potatoes", "prawns", "protein", "raisins", "rice",
    "saline", "salt", "shrimp", "sodium", "spinach", "strawberries", "sugar", "sugars", "tomatoes", "vanilla",
    "walnuts", "water", "yeast", "yogurt", "zinc"};
// Drug names by their usual endings ('500 mg amoxicillin', '20 mg atorvastatin').
bool drug_name(std::string_view w) {
    static const char* const ENDINGS[] = {"cillin", "mycin", "profen", "olol", "pril", "sartan", "statin", "azole",
                                          "caine", "phine", "azepam", "formin", "amol", "isone", "oxacin", "vir"};
    for (const char* e : ENDINGS) {
        std::string_view x(e);
        if (w.size() > x.size() + 2 && w.substr(w.size() - x.size()) == x) return true;
    }
    return false;
}
// Adjectives that follow a measure ('120 dB loud', '143 mmHg systolic').
const rv::WordSet ADJ_AFTER{"abroad", "ahead", "alone", "behind", "brighter", "cheaper", "citywide", "cooler",
    "diastolic", "downtown", "early", "fast", "heavy", "indoors", "inland", "late", "loud", "louder", "nationwide",
    "nearby", "old", "older", "outdoors", "overseas", "overweight", "quiet", "quieter", "short", "slow",
    "statewide", "systolic", "underweight", "uptown", "whenever", "wherever", "worldwide", "younger"};
// Determiners that do not follow a noun directly ('every day', 'this week' and 'that' do).
const rv::WordSet DETERMINERS{"a", "an", "his", "its", "my", "our", "the", "their", "your"};

// Words after a number and unit that make an ambiguous unit a length or an inch ('80 in high').
const rv::WordSet DIM_WORDS{"above", "across", "ahead", "apart", "away", "behind", "below", "broad", "cable",
    "dash", "deep", "diagonally", "diameter", "downstream", "freestyle", "high", "hurdles", "length", "lengths",
    "long", "offshore", "race", "radius", "relay", "sprint", "swim", "tall", "thick", "underground", "upstream",
    "wide"};
// Words that make a joined 'm' a length: nouns after it, cue words shortly before it.
const rv::WordSet LENGTH_NOUNS{"antenna", "beam", "bridge", "cable", "cliff", "coastline", "dive", "drop", "fabric",
                               "fence", "fencing", "hose", "jump", "ladder", "lane", "lanes", "mast", "pipe",
                               "pipeline", "pole", "pool", "rope", "shoreline", "statue", "tower", "track", "trail",
                               "tunnel", "wall", "waves", "wire", "yacht"};
const rv::WordSet LENGTH_CUES{"altitude", "climbed", "deep", "depth", "diameter", "dived", "elevation", "height",
                              "high", "jumped", "length", "measured", "measures", "measuring", "radius", "spanning",
                              "spans", "sprinted", "standing", "stands", "stood", "swam", "tall", "threw",
                              "towering", "towers", "width", "wingspan"};
// Plural nouns that are counted in millions: 'm' before them is a million, not meters or minutes ('5m people').
const rv::WordSet MILLION_NOUNS{"acres", "adults", "barrels", "cars", "children", "copies", "customers", "devices",
    "downloads", "followers", "hectares", "homes", "households", "jobs", "members", "passengers", "patients",
    "people", "readers", "residents", "shares", "streams", "students", "subscribers", "tonnes", "tons", "units",
    "users", "vehicles", "viewers", "views", "visitors", "voters", "votes", "workers", "years"};

// Words shortly before a number that make 'A' amps: other electrical measures count ('12 V and 2 A').
const rv::WordSet AMP_CUES{"amperage", "breaker", "breakers", "charger", "circuit", "current", "fuse", "fuses",
                           "rated", "rating", "V", "volts", "voltage", "mA", "kV", "mV", "W", "watts", "Ω", "ohms"};
const rv::WordSet AMP_NEXT{"fuse", "breaker", "charger", "current", "circuit", "max", "maximum", "peak", "continuous",
                           "output", "input", "load", "draw", "rating", "supply", "adapter", "cable", "socket",
                           "outlet", "service", "panel", "port", "limit"};
const rv::WordSet AMP_PREV{"draws", "draw", "drew", "drawing", "pulls", "pulled", "current", "rated", "carries",
                           "supplies", "delivers", "deliver", "output", "outputs", "charges", "charging"};
const rv::WordSet BAR_NEXT{"of", "pressure", "at", "above", "below", "and", "or", "in", "on", "for", "with", "during",
                           "gauge", "max", "maximum", "overpressure", "absolute", "to", "when", "while", "is", "was"};
const rv::WordSet RES_NEXT{"blu-ray", "camera", "cameras", "capture", "content", "display", "displays", "encode",
    "export", "exports", "film", "films", "footage", "gaming", "grade", "hdr", "image", "images", "monitor",
    "monitors", "movie", "movies", "output", "panel", "panels", "playback", "projector", "recording", "recordings",
    "render", "renders", "resolution", "screen", "screens", "stream", "streaming", "streams", "television",
    "televisions", "tv", "tvs", "upscaling", "video", "videos", "viewer"};

bool currency(uint32_t cp) {
    return cp == '$' || cp == 0x20AC || cp == 0xA3 || cp == 0xA5 || cp == 0x20B9 || cp == 0x20A9 || cp == 0x20BD ||
           cp == 0x20BA || cp == 0x20AA || cp == 0x20BF;
}

// A unit expression starting at token j.
struct UnitMatch {
    bool ok = false;
    size_t last = 0;      // last token of the unit
    std::string sg, pl;   // spoken forms
    uint16_t flags = 0;   // flags of the first unit
    bool rate = false;    // has a denominator
};

// A power after the unit ending at token k: '²', '³', ASCII '2'/'3' (lengths only), '^2'. Returns digits or "".
std::string unit_power(const Doc& d, size_t k, bool length, size_t& last) {
    if (k + 1 < d.size() && d.joined(k + 1)) {
        if (super_tok(d, k + 1)) {
            std::string dg;
            size_t e = digit_run(d, k + 1, true, dg);
            if ((dg == "2" || dg == "3") && !stuck_after(d, e)) {
                last = e;
                return dg;
            }
            return "";
        }
        if (length && d.number(k + 1) && (tok(d, k + 1) == "2" || tok(d, k + 1) == "3") && !stuck_after(d, k + 1) &&
            !(d.punct(k + 2, '.') && d.joined(k + 2) && d.number(k + 3))) {
            last = k + 1;
            return std::string(tok(d, k + 1));
        }
        if (d.punct(k + 1, '^') && d.number(k + 2) && d.joined(k + 2) &&
            (tok(d, k + 2) == "2" || tok(d, k + 2) == "3") && !stuck_after(d, k + 2)) {
            last = k + 2;
            return std::string(tok(d, k + 2));
        }
    }
    return "";
}

// One unit (no denominator) at token j: table word, 'sq ft', 'fl oz', the degree sign, 'tCO2e'.
bool simple_unit(const Doc& d, size_t j, bool den, size_t& last, std::string& sg, std::string& pl, uint16_t& flags) {
    if (j >= d.size()) return false;
    const Token& t = d[j];
    if (t.kind == Kind::Punct) {
        if (den) return false;
        if (t.cp == 0x2103 || t.cp == 0x2109) {  // ℃ ℉
            last = j;
            sg = t.cp == 0x2103 ? "degree Celsius" : "degree Fahrenheit";
            pl = t.cp == 0x2103 ? "degrees Celsius" : "degrees Fahrenheit";
            flags = TEMP;
            return true;
        }
        if (t.cp == 0x2030) {  // ‰
            last = j;
            sg = pl = "per mille";
            flags = INV;
            return true;
        }
        if (t.cp != 0xB0) return false;
        // degree sign: °C, ° C, °F, °K, ° N (a bearing), or degrees alone
        size_t k = j + 1;
        flags = TEMP;
        if (k < d.size() && d.word(k) && same_line(d, k) && !stuck_after(d, k) &&
            (d.joined(k) || d.str(k).size() == 1)) {
            std::string_view w = d.str(k);
            if (d.joined(k) && (w == "c" || w == "f")) w = w == "c" ? "C" : "F";  // '20 °c'
            const char* name = w == "C" ? "Celsius" : w == "F" ? "Fahrenheit" : w == "K" ? "kelvin" : nullptr;
            if (name) {
                last = k;
                if (w == "K") {
                    sg = pl = "kelvin";
                } else {
                    sg = std::string("degree ") + name;
                    pl = std::string("degrees ") + name;
                }
                return true;
            }
            const char* dir = w == "N" ? "North" : w == "S" ? "South" : w == "E" ? "East" : w == "W" ? "West" : nullptr;
            if (dir && !(d.punct(k + 1, '.') && d.joined(k + 1) && d.word(k + 2))) {
                last = k;
                sg = std::string("degree ") + dir;
                pl = std::string("degrees ") + dir;
                flags = 0;
                return true;
            }
        }
        last = j;
        sg = "degree";
        pl = "degrees";
        flags = 0;
        return true;
    }
    if (t.kind != Kind::Word) return false;
    std::string_view w = d.str(j);
    // square / cubic prefix: 'sq ft', 'sq. m', 'cu ft', 'Sq. Ft.', 'SQ FT'
    if (!den && (w == "sq" || w == "Sq" || w == "SQ" || w == "cu" || w == "Cu" || w == "CU")) {
        size_t k = j + 1;
        if (d.punct(k, '.') && d.joined(k)) ++k;
        bool spaced_ok = k == j + 1 ? !d.joined(k) : true;
        if (k < d.size() && d.word(k) && same_line(d, k) && spaced_ok) {
            const UnitDef* u = length_unit(d.str(k));
            if (u && (u->flags & LEN) && !stuck_after(d, k)) {
                std::string pre = w[0] == 'c' || w[0] == 'C' ? "cubic " : "square ";
                last = k;
                sg = pre + std::string(u->sg);
                pl = pre + plural_of(*u);
                flags = u->flags & ~(CTX | NOBARE);
                return true;
            }
        }
        return false;
    }
    // fluid ounces
    if (!den && w == "fl") {
        size_t k = j + 1;
        if (d.punct(k, '.') && d.joined(k)) ++k;
        if (k < d.size() && d.is(k, "oz") && same_line(d, k)) {
            last = k;
            sg = "fluid ounce";
            pl = "fluid ounces";
            flags = DOT;
            return true;
        }
        return false;
    }
    // emissions: tCO2e, MtCO₂e, kgCO2
    if (!den && w.size() >= 3 && w.substr(w.size() - 2) == "CO") {
        std::string_view pre = w.substr(0, w.size() - 2);
        const char* mass = pre == "t" ? "tonne" : pre == "Mt" ? "megatonne" : pre == "kt" ? "kilotonne"
                         : pre == "Gt" ? "gigatonne" : pre == "kg" ? "kilogram" : pre == "g" ? "gram" : nullptr;
        size_t k = j + 1;
        bool two = k < d.size() && d.joined(k) && ((d.number(k) && d.str(k) == "2") || (punct(d, k) && d[k].cp == 0x2082));
        if (mass && two) {
            last = k;
            std::string tail = " of CO two";
            if (d.word(k + 1) && d.joined(k + 1) && (d.str(k + 1) == "e" || d.str(k + 1) == "eq")) {
                last = k + 1;
                tail += " equivalent";
            }
            if (stuck_after(d, last)) return false;
            sg = std::string(mass) + tail;
            pl = std::string(mass) + "s" + tail;
            flags = 0;
            return true;
        }
        return false;
    }
    // degrees with a scale letter: '68 degrees F', '20 deg C'
    if (!den && (w == "degrees" || w == "degree" || w == "deg" || w == "Degrees") && d.word(j + 1) &&
        !d.joined(j + 1) && same_line(d, j + 1) && (d.str(j + 1) == "C" || d.str(j + 1) == "F") &&
        !stuck_after(d, j + 1) && !(d.punct(j + 2, '.') && d.joined(j + 2) && d.word(j + 3))) {
        const char* name = d.str(j + 1) == "C" ? "Celsius" : "Fahrenheit";
        last = j + 1;
        sg = std::string("degree ") + name;
        pl = std::string(w == "degree" ? "degree " : "degrees ") + name;
        flags = TEMP;
        return true;
    }
    // torque written with a hyphen: 'ft-lb', 'lb-ft', 'in-lbs', 'N-m'
    if (!den && d.punct(j + 1, '-') && d.joined(j + 1) && d.word(j + 2) && d.joined(j + 2) && !stuck_after(d, j + 2)) {
        std::string_view b = d.str(j + 2);
        bool pound = b == "lb" || b == "lbs" || b == "lbf";
        const char *s1 = nullptr, *p1 = nullptr;
        if ((w == "ft" || w == "foot") && pound) s1 = "foot-pound", p1 = "foot-pounds";
        else if ((w == "lb" || w == "lbs" || w == "lbf") && (b == "ft" || b == "feet"))
            s1 = "pound-foot", p1 = "pound-feet";
        else if (w == "in" && pound) s1 = "inch-pound", p1 = "inch-pounds";
        else if (w == "N" && b == "m") s1 = "newton-meter", p1 = "newton-meters";
        if (s1) {
            last = j + 2;
            sg = s1;
            pl = p1;
            flags = 0;
            return true;
        }
    }
    // thermal conductivity: 'W/mK'
    if (den && w == "mK") {
        last = j;
        sg = pl = "meter kelvin";
        flags = 0;
        return true;
    }
    // millimeters of mercury written apart: 'mm Hg'
    if (!den && w == "mm" && j + 1 < d.size() && d.str(j + 1) == "Hg" && same_line(d, j + 1) && !d.joined(j + 1) &&
        !stuck_after(d, j + 1)) {
        last = j + 1;
        sg = "millimeter of mercury";
        pl = "millimeters of mercury";
        flags = 0;
        return true;
    }
    if (!den && w.size() > 6) return false;  // no unit is longer
    const UnitDef* u = units().find(w);
    if (!u) {
        if (den && DEN_WORDS.contains(d.low(j)) && (d[j].casing & rv::LOWER)) {
            last = j;
            sg = pl = std::string(w);
            flags = 0;
            return true;
        }
        return false;
    }
    if ((u->flags & TIME) && !den) return false;
    last = j;
    flags = u->flags;
    size_t pl_last = j;
    std::string p = unit_power(d, j, (u->flags & LEN) || (den && (u->flags & TIME)), pl_last);  // 'm/s2'
    if (!p.empty()) {
        last = pl_last;
        if (u->flags & LEN) {
            std::string pre = p == "2" ? "square " : "cubic ";
            sg = pre + std::string(u->sg);
            pl = pre + plural_of(*u);
        } else {
            std::string post = p == "2" ? " squared" : " cubed";
            sg = std::string(u->sg) + post;
            pl = plural_of(*u) + post;
        }
        flags &= ~(CTX | NOBARE);
        return true;
    }
    sg = std::string(u->sg);
    pl = plural_of(*u);
    return true;
}

// A unit with a negative superscript power at token k: 's⁻¹' -> 'second', 'm⁻²' -> 'square meter', 's⁻²' ->
// 'second squared'. last: the last token.
bool inverse_unit(const Doc& d, size_t k, size_t& last, std::string& words) {
    if (!d.word(k) || !d.punct(k + 1, 0x207B) || !d.joined(k + 1) || !super_tok(d, k + 2) || !d.joined(k + 2))
        return false;
    const UnitDef* u = units().find(d.str(k));
    if (!u || (u->flags & PART)) return false;
    std::string dg;
    size_t e = digit_run(d, k + 2, true, dg);
    if (stuck_after(d, e) || (dg != "1" && dg != "2" && dg != "3")) return false;
    last = e;
    if (dg == "1") words = std::string(u->sg);
    else if (u->flags & LEN) words = std::string(dg == "2" ? "square " : "cubic ") + std::string(u->sg);
    else words = std::string(u->sg) + (dg == "2" ? " squared" : " cubed");
    return true;
}

// A unit with optional denominators: 'km/h', 'mg/kg/day', 'm/s²', 'N·m'.
UnitMatch unit_expr(const Doc& d, size_t j) {
    UnitMatch m;
    size_t last;
    std::string sg, pl;
    uint16_t flags;
    std::string inv;
    size_t l2;
    if (inverse_unit(d, j, l2, inv)) {  // '0.5 s⁻¹' -> 'per second'
        sg = pl = "per " + inv;
        flags = INV;
        last = l2;
        m.rate = true;
    } else if (!simple_unit(d, j, false, last, sg, pl, flags)) {
        return m;
    }
    m.flags = flags;
    while (last + 2 < d.size() && d.joined(last + 1) && d.joined(last + 2) &&
           (d.punct(last + 1, '/') || d.punct(last + 1, 0xB7) || d.punct(last + 1, 0x22C5))) {
        bool per = d.punct(last + 1, '/');
        std::string s2, p2;
        uint16_t f2;
        if (!per && inverse_unit(d, last + 2, l2, inv)) break;  // 'mg·L⁻¹' is read below
        if (!simple_unit(d, last + 2, per, l2, s2, p2, f2)) break;
        if (d.str(j) == "N" && d.str(last + 2) == "A") break;  // 'N/A'
        if (per) {
            sg += " per " + s2;
            pl += " per " + s2;
        } else if (m.rate) {
            sg += " " + s2;  // 'watts per meter kelvin'
            pl += " " + s2;
        } else {
            pl = sg + " " + p2;  // 'newton meters'
            sg += " " + s2;
        }
        m.rate = true;
        last = l2;
        m.flags &= ~(CTX | NOBARE | SPACED);
    }
    // a rate per a counted unit: 'L/100 km', 'g/100 g'
    if (last + 3 < d.size() && d.punct(last + 1, '/') && d.joined(last + 1) && d.number(last + 2) &&
        d.joined(last + 2) && !d[last + 3].newline_before) {
        rv::NumberSpan ns = rv::parse_number(d, last + 2);
        size_t k = ns.last + 1, l2;
        std::string s2, p2, nw = rv::read_number(ns);
        uint16_t f2;
        // a round count ('100 km', '1,000 people'), not a second measure ('5kg/11lb')
        bool round = ns.fraction.empty() && !ns.whole.empty() && ns.whole[0] == '1' &&
                     ns.whole.find_first_not_of('0', 1) == std::string::npos;
        if (round && !nw.empty() && simple_unit(d, k, false, l2, s2, p2, f2) && !(f2 & (TIME | PART)) &&
            !stuck_after(d, l2)) {
            std::string tail = " per " + nw + " " + (ns.whole == "1" && ns.fraction.empty() ? s2 : p2);
            sg += tail;
            pl += tail;
            m.rate = true;
            last = l2;
            m.flags &= ~(CTX | NOBARE | SPACED);
        }
    }
    // negative exponents after the unit: 'm·s⁻¹', 'W m⁻²', 'J kg⁻¹ K⁻¹'
    while (last + 1 < d.size() && !d[last + 1].newline_before) {
        size_t k = last + 1;
        if ((d.punct(k, 0xB7) || d.punct(k, 0x22C5)) && d.joined(k) && d.joined(k + 1)) ++k;
        else if (d.joined(k)) break;
        if (!inverse_unit(d, k, l2, inv)) break;
        sg += " per " + inv;
        pl += " per " + inv;
        last = l2;
        m.rate = true;
        m.flags &= ~(CTX | NOBARE | SPACED);
    }
    if ((flags & PART) && !m.rate) return m;  // 'N' alone is not a newton
    if (m.flags & INV) m.flags &= ~(CTX | NOBARE);
    m.ok = true;
    m.last = last;
    m.sg = sg;
    m.pl = pl;
    return m;
}

// An adverb in -ly ('nearly', 'roughly'), not a noun like 'supply' or 'assembly'.
bool adverb(std::string_view w) {
    if (w.size() <= 4 || w.substr(w.size() - 2) != "ly") return false;
    char c = w[w.size() - 3];
    if (c == 'p' || (c == 'b' && w[w.size() - 4] != 'a' && w[w.size() - 4] != 'i')) return false;  // supply, assembly
    static const rv::WordSet NOUNS{"family", "rally", "belly", "jelly", "italy", "anomaly", "monopoly", "butterfly",
                                   "firefly", "dragonfly", "homily", "lily", "bully", "gully", "folly", "holly"};
    return !NOUNS.contains(w);
}

// Token k is the start of the text after a measure: is the measure an adjective before a noun? det: a determiner
// stands before the measure ('the 15–20 km transect the team used'), so the next word is a noun, not a verb.
bool adjectival(const Doc& d, size_t k, bool det = false) {
    if (k >= d.size()) return false;
    if (d.punct(k, '-') && d.joined(k) && d.word(k + 1) && d.joined(k + 1)) return true;
    if (!d.word(k) || d.joined(k) || d[k].newline_before) return false;
    std::string_view low = d.low(k);
    uint8_t c = d[k].casing;
    if (c & rv::TITLE) return false;
    if (c & rv::UPPER) {
        static const rv::WordSet NOT{"am", "pm", "utc", "gmt", "cet", "est", "pst", "and", "or", "ac", "dc", "rms",
                                     "to", "in", "of", "per", "at", "for", "is", "was", "i", "a", "ne", "nw", "se",
                                     "sw", "nne", "ene", "ese", "sse", "ssw", "wsw", "wnw", "nnw", "iv", "im", "po",
                                     "sc", "sl", "bid", "tid", "qid", "qd", "prn", "od"};
        // 'a 2 kW AC unit' but '230 V AC'
        if ((low == "ac" || low == "dc") && d.word(k + 1) && !d.joined(k + 1) && (d[k + 1].casing & rv::LOWER) &&
            !NOT_NOUN.contains(d.low(k + 1)))
            return true;
        return low.size() >= 2 && !NOT.contains(low);
    }
    // a key of 'key=value', a label ('gold: 3.55 kg  silver: 12.8 kg') and a contraction ('that's') do not take a
    // modifier
    if ((d.punct(k + 1, '=') || (d.punct(k + 1, ':') && !det)) && d.joined(k + 1)) return false;
    if (low.find('\'') != std::string_view::npos || low.find("\xE2\x80\x99") != std::string_view::npos) return false;
    // a compound modifier before a noun: '200 ms first-token latency', 'a 5 kg high-density block'
    if (d.punct(k + 1, '-') && d.joined(k + 1) && d.word(k + 2) && d.joined(k + 2) && (d[k + 2].casing & rv::LOWER)) {
        static const rv::WordSet DIRECTION{"north", "south", "east", "west", "up", "per", "re", "non", "so"};
        return !DIRECTION.contains(low);
    }
    if (low == "first") {  // '200 ms first token' but '5 kg first, then'
        return d.word(k + 1) && !d.joined(k + 1) && (d[k + 1].casing & rv::LOWER) && !NOT_NOUN.contains(d.low(k + 1));
    }
    if (NOT_NOUN.contains(low) || VERBS_AFTER.contains(low) || ADJ_AFTER.contains(low) || SUBSTANCES.contains(low) ||
        drug_name(low))
        return false;
    if (d.is(k + 1, "over") && d.word(k + 2) && d.low(k + 2) == low) return false;  // 'quarter over quarter'
    if (d.number(k + 1) && d.joined(k + 1)) return false;                           // '250 ms p99'
    if (!det) {
        if (PREDICATE.contains(low)) return false;
        // a participle or verb before a preposition: '300 ms set for mobile clients', '2 x 5 m mean for'
        static const rv::WordSet AMBIGUOUS{"mean", "put", "set", "shut"};
        static const rv::WordSet PREP{"across", "as", "at", "by", "during", "for", "from", "in", "into", "on", "to",
                                      "with"};
        if (AMBIGUOUS.contains(low) && d.word(k + 1) && PREP.contains(d.low(k + 1))) return false;
        // a verb: the word after it is a determiner ('12 GB trigger a flush'), but not in 'a glass a day'
        static const rv::WordSet PER_WORDS{"day", "game", "go", "head", "hour", "minute", "month", "night", "person",
                                           "piece", "pop", "second", "serving", "session", "side", "time", "week",
                                           "year"};
        if (d.word(k + 1) && !d.joined(k + 1) && DETERMINERS.contains(d.low(k + 1)) && !(d[k + 1].casing & rv::UPPER) &&
            !((d.is(k + 1, "a") || d.is(k + 1, "an")) && d.word(k + 2) && PER_WORDS.contains(d.low(k + 2))))
            return false;
        if (low == "read" && (d.punct(k + 1, ',') || (d.word(k + 1) && NOT_NOUN.contains(d.low(k + 1))))) return false;
    }
    if (low == "can" && d.word(k + 1) && !d.joined(k + 1)) {  // 'a 28 oz can' but '5 kg can be'
        static const rv::WordSet VERB_AFTER{"be", "not", "also", "only", "still", "easily", "hold", "store", "fit",
                                            "run", "carry", "reach", "power", "last", "handle", "take", "make"};
        if (VERB_AFTER.contains(d.low(k + 1))) return false;
    }
    if (adverb(low)) return false;
    if (low.size() > 4 && low.substr(low.size() - 2) == "ed") {
        static const rv::WordSet NOUNS{"speed", "feed", "seed", "need", "breed", "shred", "steed", "embed"};
        if (!NOUNS.contains(low)) return false;
    }
    return true;
}

// The token before a measure starting at token first, skipping a range ('the 15–20 km') and 'sub-'; npos if none.
size_t before_measure(const Doc& d, size_t first) {
    size_t b = first;
    if (b >= 3 && d.joined(b) && (d.punct(b - 1, '-') || d.punct(b - 1, 0x2013)) && d.number(b - 2)) {
        b -= 2;
        while (b >= 2 && d.joined(b) && (d.punct(b - 1, '.') || d.punct(b - 1, ',')) && d.number(b - 2)) b -= 2;
    } else if (b >= 2 && d.joined(b) && d.punct(b - 1, '-') && d.is(b - 2, "sub")) {
        b -= 2;
    }
    return b > 0 && d[b].space_before ? b - 1 : Doc::npos;
}

// A determiner before the measure: 'the 15–20 km transect', 'a sub-180 ms read'.
bool det_before(const Doc& d, size_t first) {
    static const rv::WordSet DET{"a", "an", "each", "every", "her", "his", "its", "my", "our", "that", "the",
                                 "their", "these", "this", "those", "your"};
    size_t p = before_measure(d, first);
    return p != Doc::npos && d.word(p) && DET.contains(d.low(p));
}

// "a 75 ms maximum", "a 900 MB processed copy": after "a" the measure describes the word after it.
bool after_article(const Doc& d, size_t first, size_t k) {
    size_t p = before_measure(d, first);
    if (p == Doc::npos || !(d.is(p, "a") || d.is(p, "an"))) return false;
    if (!d.word(k) || d.joined(k) || d[k].newline_before || !(d[k].casing & rv::LOWER)) return false;
    std::string_view w = d.low(k);
    return w != "of" && w != "per" && w != "and" && w != "or" && w != "to" && w != "in" && w != "at" && w != "for";
}

// Previous token is a range dash or a dimension sign between two numbers.
bool after_range_or_dim(const Doc& d, size_t first) {
    if (first < 2) return false;
    size_t p = first - 1;
    const Token& t = d[p];
    if (t.kind == Kind::Punct && (t.cp == '-' || t.cp == 0x2013 || t.cp == 0xD7)) {
        size_t q = p - 1;
        return d[q].kind == Kind::Number || (d[q].kind == Kind::Word && d.str(q).size() <= 3 && units().find(d.str(q)));
    }
    if (t.kind == Kind::Word && (d.str(p) == "x" || d.str(p) == "X" || d.str(p) == "by" || d.str(p) == "to"))
        return d.number(p - 1);
    return false;
}

// A number span before a unit; sign token index (npos if none).
struct Num {
    bool ok = false;
    size_t first = 0, last = 0, sign = Doc::npos;
    std::string words;
    bool one = false;
};

Num number_at(const Doc& d, size_t i) {
    Num n;
    rv::NumberSpan s = rv::parse_number(d, i);
    if (!s.ok) return n;
    n.first = i;
    n.last = s.last;
    n.words = rv::read_number(s);
    n.one = s.whole == "1" && s.fraction.empty();
    if (i > 0 && d.joined(i) && punct(d, i - 1)) {
        uint32_t c = d[i - 1].cp;
        bool lone = !d.joined(i - 1) || (d[i - 2].kind == Kind::Punct && d[i - 2].cp != '-' && d[i - 2].cp != '.' &&
                                         d[i - 2].cp != ',');
        if (lone && c == '.' && s.fraction.empty()) {  // '.5 mm'
            n.sign = i - 1;
            n.words = "point " + rv::digits(s.whole);
            n.one = false;
        } else if (lone && (c == '-' || c == 0x2212)) {
            n.sign = i - 1;
            n.words = "minus " + n.words;
        } else if (lone && c == '+') {
            n.sign = i - 1;
            n.words = "plus " + n.words;
        } else if (lone && c == 0xB1) {
            n.sign = i - 1;
            n.words = "plus or minus " + n.words;
        }
    }
    n.ok = !n.words.empty();
    return n;
}

const rv::WordSet SCALE{"k", "K", "M", "B", "bn", "mn", "T"};
const rv::WordSet SCALE_WORDS{"thousand", "million", "billion", "trillion"};

// Context gates for ambiguous units. j: unit token, k: token after the unit, num_last: last token of the number.
bool unit_allowed(const Doc& d, const UnitMatch& u, size_t j, size_t num_last, size_t num_first, bool joined) {
    std::string_view w = tok(d, j);
    size_t k = u.last + 1;
    if ((u.flags & SPACED) && joined) return false;
    if (!(u.flags & CTX)) {
        if (w == "t" || w == "L" || w == "V" || w == "g" || w == "J") {
            if (d.punct(k, '-') && d.joined(k)) return false;  // 't-shirts'
            if (d.punct(k, '\'') && d.joined(k) && stuck_after(d, k)) return false;  // but '7 V' means'
        }
        return true;
    }
    std::string_view next = d.word(k) && !d.joined(k) ? d.low(k) : std::string_view();
    if (w == "in") {
        if (k >= d.size()) return false;  // '... and 4 in' may be a cut fragment
        if (joined) return true;          // '12in'
        if (next == "by" || next == "x" || d.punct(k, 0xD7)) return true;  // '12 in by 8 in'
        // after feet: '6 ft 8 in'
        if (num_first >= 2 && (d.is(num_first - 1, "ft") || d.is(num_first - 1, "feet") || d.punct(num_first - 1, '.')) &&
            d.number(num_first - 2 - (d.punct(num_first - 1, '.') ? 1 : 0)))
            return true;
        if (d.punct(k, '.') && d.joined(k)) return true;
        if (punct(d, k) && (d[k].cp == ',' || d[k].cp == ';' || d[k].cp == ')' || d[k].cp == ':' || d[k].cp == 0x2014))
            return true;
        if (d[k].newline_before) return true;
        return (DIM_WORDS.contains(next) && next != "in") || next == "of";
    }
    if (w == "m") {
        if (MILLION_NOUNS.contains(next)) return false;  // '5 m people'
        if (!joined) {
            // '5 m 30 s' is a duration
            if (d.number(k) && (d.is(k + 1, "s") || d.is(k + 1, "sec"))) return false;
            return true;
        }
        if (u.rate || (u.last > j)) return true;  // m/s, m²
        // '1h 50m' is a duration
        if (num_first >= 2 && d.word(num_first - 1) && d.is(num_first - 1, "h") && d.number(num_first - 2)) return false;
        if (DIM_WORDS.contains(next)) return true;
        if (num_last > num_first) return true;  // '6.22m': minutes are not written with decimals
        if (d.is(k, "x") || d.punct(k, 0xD7) || d.is(k, "by")) return true;
        if (num_first >= 1 && (d.is(num_first - 1, "x") || d.punct(num_first - 1, 0xD7))) return true;
        if (LENGTH_NOUNS.contains(next)) return true;  // 'a 10m wall'
        // a race: 'he ran the 100m', 'the 400m hurdles'
        {
            static const rv::WordSet RACES{"50", "60", "100", "110", "200", "400", "800", "1500", "3000", "5000",
                                           "10000"};
            if (num_first > 0 && d.is(num_first - 1, "the") && num_last == num_first &&
                RACES.contains(d.str(num_first)))
                return true;
        }
        // '4m of extra lanes'
        if (next == "of")
            for (size_t b = k + 1; b <= k + 2 && d.word(b); ++b)
                if (LENGTH_NOUNS.contains(d.low(b))) return true;
        // 'measuring 2m in length'
        if (next == "in" && d.word(k + 1) && LENGTH_CUES.contains(d.low(k + 1))) return true;
        // a length word shortly before: 'standing at 300m', 'its height would be 200m'
        for (size_t b = num_first, words = 0; b-- > 0 && words < 4;) {
            if (d[b + 1].newline_before) break;
            if (punct(d, b)) {
                if (d[b].cp == '.' || d[b].cp == ';' || d[b].cp == '!' || d[b].cp == '?') break;
                continue;
            }
            ++words;
            if (d.word(b) && LENGTH_CUES.contains(d.low(b))) return true;
        }
        return false;
    }
    if (w == "W") {
        // 'Highway 101 W' is a direction
        static const rv::WordSet ROADS{"highway", "hwy", "route", "rte", "interstate", "i", "us", "exit", "sr", "road",
                                       "rd", "street", "st", "avenue", "ave", "county", "state"};
        size_t p = d.prev_word(num_first);
        if (p != Doc::npos && p + 1 == num_first && ROADS.contains(d.low(p))) return false;
        if (d.word(k) && (d[k].casing & rv::TITLE) && !d.joined(k)) return false;  // '45 W Main St'
        static const rv::WordSet STREET{"st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "dr",
                                        "drive", "ln", "lane", "way", "pl", "place", "ct", "court", "pkwy", "hwy"};
        if (d.punct(k, '.') && d.joined(k) && d.word(k + 1) && (d[k + 1].casing & rv::TITLE) &&
            (d.word(k + 2) && STREET.contains(d.low(k + 2))))
            return false;
        return true;
    }
    if (w == "Mt") return !(d.word(k) && (d[k].casing & rv::TITLE) && !d.joined(k));
    if (w == "um" || w == "l") {
        // 'um' is also a filler ('like 5 um maybe 6'), 'l)' a list letter ('Article 5 l)')
        static const rv::WordSet FILLER{"about", "actually", "and", "basically", "but", "he", "i", "it", "like",
                                        "maybe", "or", "she", "so", "they", "we", "well", "yeah", "you"};
        if (d.punct(k, ')') || (num_first > 0 && d.punct(num_first - 1, '('))) return false;
        if (FILLER.contains(next) || (num_first > 0 && d.punct(num_first - 1, ','))) return false;
        if (w == "l" && num_first > 0 && d.word(num_first - 1)) {
            static const rv::WordSet LEGAL{"art", "article", "clause", "item", "para", "paragraph", "point", "section"};
            if (LEGAL.contains(d.low(num_first - 1))) return false;
        }
        return true;
    }
    if (w == "N") {  // newtons in a compound or after a force word: 'a force of 9.8 N'
        if (u.rate) return true;
        static const rv::WordSet FORCE{"force", "forces", "thrust", "tension", "load", "drag", "lift", "weight",
                                       "pull", "push", "newtons", "torque", "friction", "weighs", "weigh",
                                       "weighed", "exerts", "exerted", "applies", "applied"};
        for (size_t b = num_first, words = 0; b-- > 0 && words < 5;) {
            if (punct(d, b)) {
                if (d[b].cp == '.' || d[b].cp == ';' || d[b + 1].newline_before) break;
                continue;
            }
            ++words;
            if (d.word(b) && FORCE.contains(d.low(b))) return !(d.word(k) && (d[k].casing & rv::TITLE));
        }
        return false;
    }
    if (w == "A") {
        size_t p = d.prev_word(num_first);
        if (AMP_NEXT.contains(next) || (p != Doc::npos && AMP_PREV.contains(d.low(p)))) return true;
        // 'the fuse rating is 10 A', 'a breaker above 15 A.'
        for (size_t b = num_first, words = 0; b-- > 0 && words < 6;) {
            if (punct(d, b)) {
                if (d[b].cp == '.' || d[b].cp == ';' || d[b].cp == '!' || d[b].cp == '?' || d[b + 1].newline_before)
                    break;
                continue;
            }
            ++words;
            if (d.word(b) && (AMP_CUES.contains(d.low(b)) || AMP_CUES.contains(d.str(b))))
                return !(d.word(k) && (d[k].casing & rv::TITLE));
        }
        return false;
    }
    if (w == "bar") return next.empty() ? !d.word(k) : BAR_NEXT.contains(next);
    if (w == "K") {
        // '4 K video' is a resolution; '$5 K' is money; 'ran 5 K' is a race
        if (RES_NEXT.contains(next)) return false;
        static const rv::WordSet RACE{"ran", "run", "runs", "running", "walked", "jogged", "raced", "race",
                                      "finished", "completed"};
        size_t pw = d.prev_word(num_first);
        if (pw != Doc::npos && pw + 1 == num_first && RACE.contains(d.low(pw))) return false;
        if (num_first > 0 && d.joined(num_first) && punct(d, num_first - 1) &&
            (d[num_first - 1].cp == '$' || d[num_first - 1].cp == 0x20AC || d[num_first - 1].cp == 0xA3))
            return false;
        return true;
    }
    return true;
}

// The number of a measure and, for a range or a dimension, the number and sign before it: '20–30 kg',
// '3 × 4 m'. These are fallbacks: numbers_core and numbers_context outrank them with their own readings.
void measure_numbers(const Doc& doc, Out& out, const Num& n) {
    if (!n.words.empty()) out.tokens(n.sign != Doc::npos ? n.sign : n.first, n.last, n.words, 38);
    if (n.sign != Doc::npos || n.first < 2) return;
    size_t p = n.first - 1;
    const Token& t = doc[p];
    bool dash = t.kind == Kind::Punct && ((t.cp == '-' && doc.joined(p) && doc.joined(n.first)) || t.cp == 0x2013);
    bool times = (t.kind == Kind::Punct && t.cp == 0xD7) || (t.kind == Kind::Word && doc.str(p) == "x");
    if (!dash && !times) return;
    // the number before: its last token is p-1
    size_t q = p - 1;
    if (!doc.number(q)) return;
    size_t f = q;
    while (f >= 2 && doc.joined(f) && (doc.punct(f - 1, '.') || doc.punct(f - 1, ',')) && doc.joined(f - 1) &&
           doc.number(f - 2))
        f -= 2;
    if (stuck_before(doc, f)) return;
    Num left = number_at(doc, f);
    if (!left.ok || left.last != q) return;
    out.tokens(p, p, dash ? "to" : "by", 20);
    out.tokens(left.sign != Doc::npos ? left.sign : left.first, left.last, left.words, 38);
}

// Last token of the number starting at i, as parse_number finds it, without building strings.
size_t number_end(const Doc& d, size_t i) {
    size_t last = i;
    if (d.str(i).size() <= 3)
        while (d.punct(last + 1, ',') && d.joined(last + 1) && d.number(last + 2) && d.joined(last + 2) &&
               d.str(last + 2).size() == 3)
            last += 2;
    if (d.punct(last + 1, '.') && d.joined(last + 1) && d.number(last + 2) && d.joined(last + 2)) last += 2;
    return last;
}

// Token j can start a unit (cheap test before the real parse).
bool maybe_unit_start(const Doc& d, size_t j) {
    const Token& t = d[j];
    if (t.kind == Kind::Word) return true;
    if (t.kind != Kind::Punct) return false;
    return t.cp == 0xB0 || t.cp == '-' || t.cp == 0x2010 || t.cp == 0x2011 || t.cp == 0x2013 || t.cp == 0x2103 ||
           t.cp == 0x2109 || t.cp == 0x2030 || t.cp == 0x200B ||
           t.cp == 0xAD || t.cp == 0x2060 || t.cp == 0xFEFF || t.cp == 0xFE0F;
}

// Units after numbers: '12 lbs', '8kg', '20°C', '9.8 m/s²', 'a 24 GB card', '1.6-kW', '-5°C'. i: a number start.
void measure_at(const Doc& doc, Out& out, size_t i) {
    if (doc.joined(i) && doc.punct(i - 1, '.') && i >= 2 && doc.joined(i - 1) && doc[i - 2].kind != Kind::Punct)
        return;  // inside '1.2.3' or 'v2.5'
    size_t e = number_end(doc, i);
    size_t j = e + 1;
    if (j >= doc.size() || doc[j].newline_before || !maybe_unit_start(doc, j)) return;
    if (doc.joined(i) && punct(doc, i - 1) && currency(doc[i - 1].cp)) return;  // money
    bool joined = doc.joined(j);
    bool hyphen = false, scaled = false;
    if ((doc.punct(j, '-') || doc.punct(j, 0x2010) || doc.punct(j, 0x2011) || doc.punct(j, 0x2013)) && joined &&
        j + 1 < doc.size() && doc.joined(j + 1) && doc.word(j + 1)) {
        // '1.6-kW plant': the hyphen stays, the unit is singular
        hyphen = true;
        ++j;
    } else if (joined && doc.word(j) && SCALE.contains(doc.str(j)) && j + 1 < doc.size() && !doc.joined(j + 1) &&
               same_line(doc, j + 1)) {
        // '2.4M kWh': the scale belongs to numbers_core, the unit to us
        scaled = true;
        ++j;
        joined = false;
    } else if (!joined && doc.word(j) && SCALE_WORDS.contains(doc.low(j)) && j + 1 < doc.size() &&
               !doc.joined(j + 1) && same_line(doc, j + 1)) {
        scaled = true;  // '9.5 million bpd', '1.2 billion kWh'
        ++j;
    }
    // invisible characters between the number and the unit
    while (j < doc.size() && punct(doc, j) &&
           (doc[j].cp == 0x200B || doc[j].cp == 0xAD || doc[j].cp == 0x2060 || doc[j].cp == 0xFEFF ||
            doc[j].cp == 0xFE0F))
        ++j;
    if (j >= doc.size() || doc[j].newline_before) return;
    UnitMatch u = unit_expr(doc, j);
    if (!u.ok) return;
    // '40°26′46″': degrees, then arc minutes (primes)
    if (stuck_after(doc, u.last) && !(doc.punct(u.last, 0xB0) && doc.number(u.last + 1))) return;
    Num n = number_at(doc, i);
    if (!unit_allowed(doc, u, j, n.last, n.first, joined && !hyphen)) return;
    if (hyphen && (u.flags & SPACED)) return;  // '10-K' is a filing
    // abbreviation period: '3 lbs. 4 oz. of' drops the period inside a sentence
    size_t end = u.last;
    const UnitDef* last_def = doc.word(u.last) ? length_unit(doc.str(u.last)) : nullptr;
    if (last_def && (last_def->flags & DOT) && doc.punct(end + 1, '.') && doc.joined(end + 1)) {
        size_t k = end + 2;
        std::string_view dw = doc.word(k) ? doc.str(k) : std::string_view();
        bool direction = dw == "N" || dw == "S" || dw == "E" || dw == "W" || dw == "NE" || dw == "NW" || dw == "SE" ||
                         dw == "SW";
        if (k < doc.size() && !doc[k].newline_before &&
            (doc.number(k) || (doc.word(k) && (doc[k].casing & rv::LOWER)) || direction ||
             (punct(doc, k) && (doc[k].cp == ',' || doc[k].cp == ';' || doc[k].cp == ':' || doc[k].cp == ')'))))
            ++end;
    }
    bool one = n.one && !scaled && !after_range_or_dim(doc, n.first);
    // a simple fraction takes the singular: '1/2 tsp salt' -> one half teaspoon salt (not '1 1/2 cups')
    if (n.sign == Doc::npos && n.first >= 2 && doc.joined(n.first) && doc.punct(n.first - 1, '/') &&
        doc.joined(n.first - 1) && doc.number(n.first - 2) && !stuck_before(doc, n.first - 2) &&
        !(n.first >= 3 && doc.number(n.first - 3)) && n.last == n.first) {
        static const rv::WordSet DENOMS{"2", "3", "4", "5", "6", "8", "10", "12", "16", "32", "64"};
        std::string_view a = doc.str(n.first - 2), b = doc.str(n.first);
        if (DENOMS.contains(b) && (a.size() < b.size() || (a.size() == b.size() && a < b))) one = true;
    }
    std::string spoken;
    if ((u.flags & INV) || one || hyphen) spoken = u.sg;
    else if ((u.flags & TEMP) || u.pl.find(" of ") != std::string::npos) spoken = u.pl;  // '130 mmHg vacuum line'
    else {
        // measures joined by a dash, 'by', 'and' or 'or' share the noun after the last one: '5 kg–10 kg loads',
        // 'the 10 ft × 12 ft shed', 'its 5 kW and 10 kW versions'
        size_t at = end + 1;
        std::string_view unit_word = doc.str(j);
        for (int hop = 0; hop < 4; ++hop) {
            size_t c = at;
            bool comma = doc.punct(c, ',');
            bool dash = (doc.punct(c, 0x2013) || doc.punct(c, '-')) && doc.joined(c);
            bool word = doc.is(c, "and") || doc.is(c, "or") || doc.is(c, "to") || doc.is(c, "by") || doc.is(c, "x") ||
                        doc.punct(c, 0xD7) || (doc.punct(c, ',') && doc.joined(c));
            if (!dash && !word) break;
            size_t nn = c + 1;
            if (!doc.number(nn) || (dash && !doc.joined(nn))) break;
            size_t e2 = number_end(doc, nn) + 1;
            if (e2 >= doc.size() || !doc.word(e2) || doc[e2].newline_before) break;
            UnitMatch u2 = unit_expr(doc, e2);
            if (!u2.ok || (u2.flags & (TIME | CTX)) || stuck_after(doc, u2.last)) break;
            if (comma && doc.str(e2) != unit_word) break;  // '48 kg, 310 km range' are two items
            unit_word = doc.str(e2);
            at = u2.last + 1;
        }
        spoken = adjectival(doc, at, det_before(doc, n.first)) || after_article(doc, n.first, at) ? u.sg : u.pl;
    }
    // a unit after a number outranks generic code readings ('20ºC', '16-kHz'); a joined 'm' with a length cue
    // outranks the minutes reading ('15.3m long')
    // 's⁻¹' over '0.5 s'; '1-L bottle' over the code reading of '1-L'
    int score = doc.str(j) == "m" && joined ? 64
                : doc.punct(j + 1, 0x207B) || (hyphen && doc.word(end + 1) && (doc[end + 1].casing & rv::LOWER)) ? 63
                                                                                                             : 61;
    out.tokens(j, end, spoken, score);
    // a footnote after the unit is not a power: '180 ms¹', '120 GB¹²'
    if (end == u.last && super_tok(doc, end + 1) && doc.joined(end + 1)) {
        std::string dg;
        size_t f = digit_run(doc, end + 1, true, dg);
        if (dg != "2" && dg != "3" && !stuck_after(doc, f)) out.tokens(end + 1, f, "", 45);
    }
    if (scaled) return;
    measure_numbers(doc, out, n);
    // a decimal after a name is a quantity, not a version: 'Fairhollow 2.14 in'
    if (n.sign == Doc::npos && n.last >= n.first + 2 && doc.str(n.last).size() >= 2 && n.first > 0 &&
        doc.word(n.first - 1) && doc[n.first].space_before && (doc[n.first - 1].casing & (rv::TITLE | rv::UPPER)) &&
        !doc.punct(end + 1, '-') && !doc.punct(end + 1, 0x2013))
        out.tokens(n.first, n.last, n.words, 71);
}

// Units with no number before them, after a superscript power, 'N+', an interval, 'per' or 'in': '3 × 10⁸ m/s',
// 'measured in kPa', 'per kWh', '40+ fps'. j: a word or a degree sign.
void bare_unit_at(const Doc& doc, Out& out, size_t j) {
    if (j == 0 || doc[j].newline_before) return;
    size_t p = j - 1;
    const Token& pt = doc[p];
    bool after_power = false, plus = false, bracket = false, cue = false;
    std::string_view prevw;
    if (pt.kind == Kind::Punct) {
        after_power = super_digit(pt.cp) >= 0;
        plus = pt.cp == '+' && p >= 1 && doc.number(p - 1) && doc.joined(p);
        bracket = pt.cp == ']';
    } else if (pt.kind == Kind::Number) {
        after_power = p >= 2 && doc.punct(p - 1, '^') && doc.joined(p);
    } else if (doc.str(p).size() <= 5 && !doc.joined(j)) {
        prevw = doc.low(p);
        cue = prevw == "in" || prevw == "per" || prevw == "unit" || prevw == "units";  // 'Unit kg.'
    } else if (doc.punct(p, ':') && p > 0 && (doc.is(p - 1, "unit") || doc.is(p - 1, "units"))) {
        prevw = "unit";
        cue = true;
    }
    if (!after_power && !plus && !bracket && !cue) return;
    if (doc.punct(j, 0xB0) && !(after_power || plus || bracket)) return;
    if (stuck_before(doc, j)) return;
    UnitMatch u = unit_expr(doc, j);
    if (!u.ok || stuck_after(doc, u.last)) return;
    std::string_view w = doc.str(j);
    if (cue && ((u.flags & (NOBARE | CTX | SPACED)) || (w.size() < 2 && u.last == j))) return;  // 'per m²'
    if ((after_power || plus || bracket) && (u.flags & (CTX | SPACED)) && !(w == "m" && !doc.joined(j))) return;
    if (cue && prevw != "per" && doc.word(u.last + 1) && !doc.joined(u.last + 1) &&
        (doc[u.last + 1].casing & rv::LOWER) && adjectival(doc, u.last + 1) && !u.rate)
        return;  // 'in GB terms' is too unsure
    std::string spoken = (u.flags & INV) ? u.sg : prevw == "per" ? u.sg : u.pl;
    out.tokens(j, u.last, spoken, after_power || plus || bracket ? 50 : 45);
}

// Rates with a plain word: '30 tokens/s', '1,200 req/s', 'the tokens/s figure'. j: a word.
const rv::PhraseMap RATE_ABBR{{"req", "requests"}, {"reqs", "requests"}, {"tok", "tokens"}, {"toks", "tokens"},
                              {"msg", "messages"}, {"msgs", "messages"}, {"hrs", "hours"}, {"pkt", "packets"},
                              {"pkts", "packets"}, {"txn", "transactions"}, {"txns", "transactions"},
                              {"it", "iterations"}, {"img", "images"}, {"imgs", "images"}};

void word_rate_at(const Doc& doc, Out& out, size_t j) {
    if (j + 2 >= doc.size() || !doc.punct(j + 1, '/') || !doc.joined(j + 1) || !doc.joined(j + 2)) return;
    if (stuck_before(doc, j) || (j > 0 && doc.joined(j) && (doc.punct(j - 1, '/') || doc.punct(j - 1, '.')))) return;
    if (!(doc[j].casing & rv::LOWER) || doc.str(j).size() < 2) return;
    const UnitDef* num_unit = units().find(doc.str(j));
    if (num_unit && !(num_unit->flags & TIME)) return;  // km/h: a unit
    // a count per unit: plural nouns and their short forms ('tokens/s', 'req/s'), not 'admin/user' or 'and/or'
    std::string_view lw = doc.low(j);
    static const rv::WordSet TIME_DEN{"day", "hour", "hr", "month", "mo", "night", "week", "wk", "year", "yr"};
    if (lw.back() != 's' && !RATE_ABBR.find(lw) && !(j > 0 && doc.number(j - 1)) &&
        !TIME_DEN.contains(doc.low(j + 2)))
        return;
    size_t last;
    std::string sg, pl;
    uint16_t f;
    if (!simple_unit(doc, j + 2, true, last, sg, pl, f)) return;
    if (stuck_after(doc, last) || (doc.punct(last + 1, '/') && doc.joined(last + 1))) return;
    const std::string_view* abbr = RATE_ABBR.find(doc.str(j));
    std::string num = abbr ? std::string(*abbr) : std::string(doc.str(j));
    out.tokens(j, last, num + " per " + sg, 61);
}

// A pace or rate after a clock-like number: '5:09/km' -> 'per kilometer'. i: the number after the colon.
void pace_at(const Doc& doc, Out& out, size_t i) {
    if (i < 2 || !doc.joined(i) || !doc.punct(i - 1, ':') || !doc.number(i - 2) || !doc.punct(i + 1, '/') ||
        !doc.joined(i + 1) || !doc.joined(i + 2))
        return;
    size_t last;
    std::string sg, pl;
    uint16_t f;
    if (!simple_unit(doc, i + 2, true, last, sg, pl, f) || stuck_after(doc, last)) return;
    out.tokens(i + 1, last, "per " + sg, 61);
}

// Inches written with a plain double quote: '27" monitor' -> twenty-seven inch monitor, 'a 6.1" screen'. A quote
// that closes an open quotation is left alone. i: a number start.
void inch_quote_at(const Doc& doc, Out& out, size_t i) {
    size_t e = number_end(doc, i), k = e + 1;
    if (!doc.punct(k, '"') || !doc.joined(k) || stuck_after(doc, k)) return;
    if (doc.number(k + 1) && doc.joined(k + 1)) return;
    int quotes = 0;  // quotation marks before on the line, not counting other inch marks
    for (size_t b = i; b-- > 0 && !doc[b + 1].newline_before;)
        quotes += doc.punct(b, '"') && !(b > 0 && doc.joined(b) && doc.number(b - 1));
    if (quotes % 2 == 1) return;  // '"I have 5" he said'
    if (i > 0 && doc.joined(i) && doc.punct(i - 1, '\'')) return;  // 5'11" is read by primes_at
    Num n = number_at(doc, i);
    if (!n.ok) return;
    bool adj = !n.one && adjectival(doc, k + 1, det_before(doc, n.first));
    out.tokens(k, k, n.one || adj ? "inch" : "inches", 61);
    out.tokens(n.sign != Doc::npos ? n.sign : n.first, n.last, n.words, 38);
}

// Celsius or Fahrenheit written without the degree sign, after a temperature word: 'GPU temp hit 83C',
// 'temp=71C', 'highs of 30 C'. 'Box 7C' and 'Seat 22C' stay. i: a number start.
void bare_celsius_at(const Doc& doc, Out& out, size_t i) {
    size_t e = number_end(doc, i), k = e + 1;
    if (!doc.word(k) || !(doc.str(k) == "C" || doc.str(k) == "F") || doc[k].newline_before || stuck_after(doc, k))
        return;
    if (doc.punct(k + 1, '.') && doc.joined(k + 1) && doc.word(k + 2)) return;
    static const rv::WordSet TEMP_WORDS{"ambient", "cold", "cooled", "degrees", "freezer", "fridge", "heat",
                                        "heated", "high", "highs", "hot", "low", "lows", "oven", "temp", "temps",
                                        "temperature", "temperatures", "thermal", "warm", "weather"};
    bool cue = false;
    for (size_t b = i, words = 0; b-- > 0 && words < 5 && !cue;) {
        if (doc[b + 1].newline_before) break;
        if (!doc.word(b)) continue;
        ++words;
        cue = TEMP_WORDS.contains(doc.low(b));
    }
    if (!cue) return;
    Num n = number_at(doc, i);
    if (!n.ok) return;
    bool c = doc.str(k) == "C";
    // above the code reading of '83C': the temperature word decides
    const char* sg = c ? "degree Celsius" : "degree Fahrenheit";
    const char* pl = c ? "degrees Celsius" : "degrees Fahrenheit";
    out.tokens(k, k, n.one ? sg : pl, 63);
    out.tokens(n.sign != Doc::npos ? n.sign : n.first, n.last, n.words, 38);
}

// Feet and inches with primes: '5′11″', '5'11"', '12″'; after a degree sign they are arc minutes and seconds.
void primes_at(const Doc& doc, Out& out, size_t i) {
    size_t e = number_end(doc, i);
    size_t k = e + 1;
    if (!(k < doc.size() && doc.joined(k) && (doc.punct(k, 0x2032) || doc.punct(k, 0x2033) || doc.punct(k, '\''))))
        return;
    rv::NumberSpan s = rv::parse_number(doc, i);
    uint32_t cp = doc[k].cp;
    size_t inch_num = Doc::npos, inch_mark = Doc::npos;
    if (cp == '\'') {
        // ASCII only as a pair: 5'11"
        if (!doc.number(k + 1) || !doc.joined(k + 1)) return;
        size_t e2 = number_end(doc, k + 1);
        if (!doc.punct(e2 + 1, '"') || !doc.joined(e2 + 1) || stuck_after(doc, e2 + 1)) return;
        inch_num = k + 1;
        inch_mark = e2 + 1;
    }
    bool arc = false;
    for (size_t b = i + 1; b-- > 0;) {
        if (doc.punct(b, 0xB0)) {
            arc = true;
            break;
        }
        if (b == 0 || !doc.joined(b)) break;
    }
    std::string words = rv::read_number(s);
    if (words.empty()) return;
    bool one = s.whole == "1" && s.fraction.empty();
    bool feet = cp == 0x2032 || cp == '\'';
    // a direction after arc minutes or seconds: '40°26′46″N'
    const char* dir = nullptr;
    if (arc && stuck_after(doc, k) && doc.word(k + 1) && !stuck_after(doc, k + 1)) {
        std::string_view d1 = doc.str(k + 1);
        dir = d1 == "N" ? "North" : d1 == "S" ? "South" : d1 == "E" ? "East" : d1 == "W" ? "West" : nullptr;
    }
    if (stuck_after(doc, k) && !(feet && doc.number(k + 1)) && !dir) return;
    size_t after = inch_mark != Doc::npos ? inch_mark + 1 : k + 1;
    bool adj = !arc && !one && inch_mark == Doc::npos && adjectival(doc, after);
    const char* unit = arc ? (feet ? (one ? "minute" : "minutes") : (one ? "second" : "seconds"))
                           : (feet ? (one || adj ? "foot" : "feet") : (one || adj ? "inch" : "inches"));
    out.tokens(i, s.last, words, 38);
    if (dir) out.tokens(k, k + 1, std::string(unit) + " " + dir, 61);
    else out.tokens(k, k, unit, 61);
    if (inch_mark != Doc::npos) {
        rv::NumberSpan s2 = rv::parse_number(doc, inch_num);
        bool one2 = s2.whole == "1" && s2.fraction.empty();
        out.tokens(inch_num, s2.last, rv::read_number(s2), 38);
        out.tokens(inch_mark, inch_mark, one2 ? "inch" : "inches", 61);
    }
}

// 'sub-200 ms' -> 'sub two hundred milliseconds': the hyphen goes, the rest is read by the measure rules.
void sub_prefix_at(const Doc& doc, Out& out, size_t i) {
    if (i >= 2 && doc.joined(i) && doc.punct(i - 1, '-') && doc.joined(i - 1) && doc.is(i - 2, "sub") &&
        !stuck_before(doc, i - 2))
        out.tokens(i - 2, i - 1, std::string(doc.str(i - 2)), 60);
}

// Between two measures a dash is 'to' and a dimension sign is 'by': '20 kg–30 kg', '4 m × 3 m'. i: the number
// after the sign.
void between_at(const Doc& doc, Out& out, size_t i) {
    if (i < 3) return;
    size_t k = i - 1;
    bool dash = doc.punct(k, 0x2013) || (doc.punct(k, '-') && doc.joined(k) && doc.joined(i));
    bool by = doc.punct(k, 0xD7) || doc.is(k, "x");
    if ((!dash && !by) || !doc.number(k - 2) || !doc.word(k - 1)) return;
    const UnitDef* u = units().find(doc.str(k - 1));
    if (!u || (u->flags & TIME)) return;
    size_t e = number_end(doc, i) + 1;
    if (e >= doc.size() || !doc.word(e) || !units().find(doc.str(e))) return;
    out.tokens(k, k, dash ? "to" : "by", 61);
}

// Scientific notation: '3 x 10^8' / '1.5 × 10⁸' -> 'three times ten to the power of eight'. i: the number '10'.
void sci_times_at(const Doc& doc, Out& out, size_t i) {
    if (i < 2 || doc.str(i) != "10" || !doc.joined(i + 1) || !(super_tok(doc, i + 1) || doc.punct(i + 1, '^'))) return;
    size_t x = i - 1;
    if (!(doc.punct(x, 0xD7) || doc.punct(x, '*') || doc.is(x, "x")) || !doc.number(x - 1)) return;
    if (doc[x].space_before != doc[i].space_before) return;
    out.tokens(x, x, "times", 47);
    if (doc.joined(x)) {  // '3x10^8': the core refuses the stuck number
        size_t f = x - 1;
        while (f >= 2 && doc.joined(f) && doc.punct(f - 1, '.') && doc.number(f - 2)) f -= 2;
        rv::NumberSpan s = rv::parse_number(doc, f);
        if (s.last == x - 1 && !stuck_before(doc, f)) out.tokens(f, s.last, rv::read_number(s), 38);
    }
}

// Screen resolutions and network generations: '4K video' -> 'four K video', '5G network', '3D printing'.
void letter_at(const Doc& doc, Out& out, size_t i) {
    if (!doc.word(i + 1) || !doc.joined(i + 1) || doc.str(i).size() != 1 || doc.str(i + 1).size() != 1) return;
    if (i > 0 && doc.joined(i) && (doc.punct(i - 1, '.') || doc.punct(i - 1, '$') || doc.punct(i - 1, ','))) return;
    std::string_view n = doc.str(i), w = doc.str(i + 1);
    if (stuck_after(doc, i + 1)) return;
    if (doc.punct(i + 2, '.') && doc.joined(i + 2) && doc.number(i + 3)) return;
    size_t k = i + 2;
    std::string_view next = doc.word(k) && !doc.joined(k) ? doc.low(k) : std::string_view();
    bool ok = false;
    int score = 50;
    if (w == "D" && n >= "1" && n <= "4") ok = true;
    else if (w == "G" && n >= "2" && n <= "6") ok = true;
    else if (w == "K" && (n == "2" || n == "4" || n == "5" || n == "6" || n == "8")) {
        static const rv::WordSet RES_PREV{"in", "at", "native", "true", "shoots", "shot", "records", "recorded",
                                          "supports", "support", "upscaled", "upscale", "outputs", "playback",
                                          "streams", "stream", "full", "and", "or"};
        size_t p = doc.prev_word(i);
        ok = RES_NEXT.contains(next) || (k < doc.size() && doc.punct(k, '/') && doc.joined(k)) ||
             (p != Doc::npos && RES_PREV.contains(doc.low(p)) && !adjectival(doc, k));
        score = 60;
    }
    if (ok) out.tokens(i, i + 1, rv::cardinal_digits(n) + " " + std::string(w), score);
}

// 'M' after a number and a space is a million in a count ('5 M users', '| 4 M |', '2 M in reserves') and molar in
// chemistry ('0.5 M HCl', 'a 3.0e-4 M solution'), where it stays. i: the number start.
void million_at(const Doc& doc, Out& out, size_t i) {
    size_t e = number_end(doc, i), k = e + 1;
    if (doc.word(k) && doc.str(k) == "m" && !doc[k].newline_before && !stuck_after(doc, k) && doc.word(k + 1) &&
        !doc.joined(k + 1) && MILLION_NOUNS.contains(doc.low(k + 1)) &&
        !(i > 0 && doc.joined(i) && punct(doc, i - 1) && currency(doc[i - 1].cp))) {
        out.tokens(k, k, "million", 64);  // '5m people', '2.3 m users'
        Num n = number_at(doc, i);
        if (n.ok && n.last == e) out.tokens(n.sign != Doc::npos ? n.sign : n.first, n.last, n.words, 38);
        return;
    }
    if (!doc.word(k) || doc.str(k) != "M" || !doc[k].space_before || doc[k].newline_before || stuck_after(doc, k))
        return;
    if (doc.punct(k + 1, '.') && doc.joined(k + 1) && doc.word(k + 2)) return;  // 'M. Smith'
    if (i > 0 && doc.joined(i) && punct(doc, i - 1) && currency(doc[i - 1].cp)) return;  // money
    if (doc.joined(i) && doc.punct(i - 1, '-') && i >= 2 && doc.word(i - 2) && doc.str(i - 2).size() == 1)
        return;  // '3.0e-4 M'
    static const rv::WordSet CHEM{"acid", "buffer", "concentration", "dilute", "hcl", "kcl", "koh", "molar",
                                  "molarity", "nacl", "naoh", "solution", "solutions", "solvent", "titrant"};
    static const rv::WordSet SIZE{"size", "sizes", "shoe", "shoes", "width"};
    // a label before the number: 'Exit 3 M', 'Studio 5 M', 'Version 2 M'
    static const rv::WordSet LABELS{"apartment", "apt", "area", "bay", "block", "box", "building", "chapter", "class",
        "court", "dock", "door", "exit", "figure", "floor", "form", "gate", "grade", "group", "hall", "highway",
        "item", "lane", "level", "line", "lot", "model", "page", "part", "phase", "pier", "platform", "room", "route",
        "row", "seat", "section", "sector", "series", "stage", "step", "studio", "suite", "table", "terminal", "tier",
        "track", "type", "unit", "version", "ward", "zone"};
    if (i > 0 && doc.word(i - 1) && LABELS.contains(doc.low(i - 1))) return;
    size_t nx = k + 1;
    if (doc.word(nx) && !doc.joined(nx) && (CHEM.contains(doc.low(nx)) || doc[nx].casing == rv::MIXED)) return;
    for (size_t b = i, words = 0; b-- > 0 && words < 8;) {
        if (doc[b + 1].newline_before) break;
        if (!doc.word(b)) continue;
        ++words;
        if (CHEM.contains(doc.low(b)) || (words == 1 && SIZE.contains(doc.low(b)))) return;
    }
    out.tokens(k, k, "million", 61);
}

// A unit after a vulgar fraction: '½ tsp salt' -> one half teaspoon salt, '¾ in drill bit', '2½ ft long'. The
// fraction alone takes the singular, a whole number and a fraction the plural. i: the fraction sign.
void fraction_unit_at(const Doc& doc, Out& out, size_t i) {
    size_t j = i + 1;
    if (j >= doc.size() || doc[j].newline_before || !doc.word(j)) return;
    bool mixed = i > 0 && doc.joined(i) && doc.number(i - 1);
    if (i > 0 && doc.joined(i) && !mixed && !punct(doc, i - 1)) return;
    UnitMatch u = unit_expr(doc, j);
    if (!u.ok || stuck_after(doc, u.last) || (u.flags & (SPACED | TIME))) return;
    std::string_view w = doc.str(j);
    if (u.flags & CTX) {
        // after a fraction 'in' is inches unless a phrase follows ('½ in the bowl')
        if (w != "in" || (doc.word(u.last + 1) && DETERMINERS.contains(doc.low(u.last + 1)))) return;
    }
    bool adj = mixed && adjectival(doc, u.last + 1, det_before(doc, i - 1));
    out.tokens(j, u.last, !mixed || adj || (u.flags & INV) ? u.sg : u.pl, 61);
}

// A unit after a number in scientific notation: '6.02e23 mol⁻¹', '1.6e-19 J'. i: the exponent number.
void sci_unit_at(const Doc& doc, Out& out, size_t i) {
    size_t e = i;
    if (!(i >= 2 && doc.joined(i))) return;
    size_t b = i - 1;
    if ((doc.punct(b, '-') || doc.punct(b, '+') || doc.punct(b, 0x2212)) && doc.joined(b)) --b;
    if (!(doc.word(b) && (doc.str(b) == "e" || doc.str(b) == "E") && doc.joined(b) && doc.number(b - 1))) return;
    size_t j = e + 1;
    if (j >= doc.size() || !doc[j].space_before || doc[j].newline_before || !doc.word(j)) return;
    UnitMatch u = unit_expr(doc, j);
    if (!u.ok || stuck_after(doc, u.last) || (u.flags & (CTX | TIME))) return;
    out.tokens(j, u.last, (u.flags & (INV | TEMP)) ? (u.flags & INV ? u.sg : u.pl)
                          : adjectival(doc, u.last + 1) ? u.sg : u.pl, 61);
}

// One scan for every unit form.
RV_RULE("units_symbols", units) {
    for (size_t i = 0; i < doc.size(); ++i) {
        const Token& t = doc[i];
        if (t.kind == Kind::Number) {
            sci_times_at(doc, out, i);
            sci_unit_at(doc, out, i);
            if (stuck_before(doc, i)) continue;  // 'A320', 'x86'
            if (doc.joined(i) && i >= 2 && (doc.punct(i - 1, ',') || doc.punct(i - 1, '.')) && doc.joined(i - 1) &&
                doc.number(i - 2))
                continue;  // the rest of '1,234' or '3.14'
            measure_at(doc, out, i);
            million_at(doc, out, i);
            pace_at(doc, out, i);
            inch_quote_at(doc, out, i);
            bare_celsius_at(doc, out, i);
            primes_at(doc, out, i);
            letter_at(doc, out, i);
            sub_prefix_at(doc, out, i);
            between_at(doc, out, i);
        } else if (t.kind == Kind::Word) {
            bare_unit_at(doc, out, i);
            word_rate_at(doc, out, i);
        } else if (t.cp == 0xBD || t.cp == 0xBC || t.cp == 0xBE || (t.cp >= 0x2153 && t.cp <= 0x215E)) {
            fraction_unit_at(doc, out, i);  // ½ ¼ ¾ ⅓ ⅔ ⅕ … ⅞
        } else if (t.cp == 0xB0) {
            bare_unit_at(doc, out, i);
            // the scale named without a number: 'Units: °C', 'Thermal limits (°C)'
            if (doc.word(i + 1) && doc.joined(i + 1) && (doc.str(i + 1) == "C" || doc.str(i + 1) == "F") &&
                !stuck_after(doc, i + 1) && !(i > 0 && doc.joined(i) && !punct(doc, i - 1)) &&
                !(i > 0 && doc.number(i - 1)))
                out.tokens(i, i + 1, doc.str(i + 1) == "C" ? "degrees Celsius" : "degrees Fahrenheit", 44);
        } else if ((t.cp == 0x2103 || t.cp == 0x2109) && !(i > 0 && doc.joined(i) && !punct(doc, i - 1))) {
            if (!(i > 0 && doc.number(i - 1)) && !(i > 1 && doc.number(i - 2) && !doc.joined(i)))
                out.tokens(i, i, t.cp == 0x2103 ? "degrees Celsius" : "degrees Fahrenheit", 44);
        }
    }
}

}  // namespace
