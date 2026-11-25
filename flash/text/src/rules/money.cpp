// Rules of the module "money": every currency form (symbols, prefixed dollars, ISO codes before or after the
// amount, currency abbreviations such as Rs. or kr, the cent sign, pence), cents, scales (K, M, bn, million, lakh
// ...), negatives and accounting brackets, ranges, per-unit prices and the adjectival singular ('a $10 coupon').
// See README.md for the conventions.
#include "core/rule.h"

namespace {

using rv::Doc;
using rv::Kind;

constexpr size_t NPOS = Doc::npos;

// ------------------------------------------------------------------------------------------------ currencies

struct Cur {
    std::string_view pl, sg, adj;              // main unit: "dollars", "one dollar", "a ten dollar coupon"
    std::string_view sub_pl, sub_sg, sub_adj;  // two decimals as a subunit; empty: the decimals are read out
    bool dot_groups = false;                   // no decimals in use, a dot groups thousands ("Rp 50.000")
};

constexpr Cur DOLLAR{"dollars", "dollar", "dollar", "cents", "cent", "cent"};
constexpr Cur EURO{"euros", "euro", "euro", "cents", "cent", "cent"};
constexpr Cur POUND{"pounds", "pound", "pound", "pence", "penny", "pence"};
constexpr Cur YEN{"yen", "yen", "yen", "", "", ""};
constexpr Cur RUPEE{"rupees", "rupee", "rupee", "paise", "paisa", "paisa"};
constexpr Cur WON{"won", "won", "won", "", "", ""};
constexpr Cur RUBLE{"rubles", "ruble", "ruble", "kopecks", "kopeck", "kopeck"};
constexpr Cur LIRA{"lira", "lira", "lira", "", "", ""};
constexpr Cur PESO{"pesos", "peso", "peso", "", "", ""};
constexpr Cur SHEKEL{"shekels", "shekel", "shekel", "", "", ""};
constexpr Cur NAIRA{"naira", "naira", "naira", "", "", ""};
constexpr Cur BITCOIN{"bitcoin", "bitcoin", "bitcoin", "", "", ""};
constexpr Cur DONG{"dong", "dong", "dong", "", "", "", true};
constexpr Cur BAHT{"baht", "baht", "baht", "", "", ""};
constexpr Cur HRYVNIA{"hryvnias", "hryvnia", "hryvnia", "", "", ""};
constexpr Cur CEDI{"cedis", "cedi", "cedi", "", "", ""};
constexpr Cur TENGE{"tenge", "tenge", "tenge", "", "", ""};
constexpr Cur LARI{"lari", "lari", "lari", "", "", ""};
constexpr Cur MANAT{"manats", "manat", "manat", "", "", ""};
constexpr Cur GUARANI{"guaranis", "guarani", "guarani", "", "", ""};
constexpr Cur KIP{"kip", "kip", "kip", "", "", ""};
constexpr Cur TUGRIK{"tugriks", "tugrik", "tugrik", "", "", ""};
constexpr Cur FRANC{"francs", "franc", "franc", "centimes", "centime", "centime"};
constexpr Cur KRONE{"kroner", "krone", "krone", "", "", ""};
constexpr Cur ZLOTY{"zloty", "zloty", "zloty", "", "", ""};
constexpr Cur KORUNA{"koruna", "koruna", "koruna", "", "", ""};
constexpr Cur RUPIAH{"rupiah", "rupiah", "rupiah", "", "", "", true};
constexpr Cur RINGGIT{"ringgit", "ringgit", "ringgit", "", "", ""};
constexpr Cur CENTS{"cents", "cent", "cent", "", "", ""};    // 75¢
constexpr Cur PENCE{"pence", "penny", "pence", "", "", ""};  // 50p

constexpr Cur USD{"US dollars", "US dollar", "US dollar", "cents", "cent", "cent"};
constexpr Cur AUD{"Australian dollars", "Australian dollar", "Australian dollar", "cents", "cent", "cent"};
constexpr Cur BRL{"Brazilian reals", "Brazilian real", "Brazilian real", "", "", ""};
constexpr Cur CAD{"Canadian dollars", "Canadian dollar", "Canadian dollar", "cents", "cent", "cent"};
constexpr Cur CHF{"Swiss francs", "Swiss franc", "Swiss franc", "centimes", "centime", "centime"};
constexpr Cur CNY{"Chinese yuan", "Chinese yuan", "Chinese yuan", "", "", ""};
constexpr Cur DKK{"Danish kroner", "Danish krone", "Danish krone", "", "", ""};
constexpr Cur GBP{"pounds", "pound", "pound", "pence", "penny", "pence"};
constexpr Cur HKD{"Hong Kong dollars", "Hong Kong dollar", "Hong Kong dollar", "cents", "cent", "cent"};
constexpr Cur INR{"Indian rupees", "Indian rupee", "Indian rupee", "paise", "paisa", "paisa"};
constexpr Cur JPY{"Japanese yen", "Japanese yen", "Japanese yen", "", "", ""};
constexpr Cur KRW{"South Korean won", "South Korean won", "South Korean won", "", "", ""};
constexpr Cur MXN{"Mexican pesos", "Mexican peso", "Mexican peso", "", "", ""};
constexpr Cur NOK{"Norwegian kroner", "Norwegian krone", "Norwegian krone", "", "", ""};
constexpr Cur NZD{"New Zealand dollars", "New Zealand dollar", "New Zealand dollar", "cents", "cent", "cent"};
constexpr Cur PLN{"Polish zloty", "Polish zloty", "Polish zloty", "", "", ""};
constexpr Cur RMB{"yuan", "yuan", "yuan", "", "", ""};
constexpr Cur RUB{"Russian rubles", "Russian ruble", "Russian ruble", "kopecks", "kopeck", "kopeck"};
constexpr Cur SEK{"Swedish kronor", "Swedish krona", "Swedish krona", "", "", ""};
constexpr Cur SGD{"Singapore dollars", "Singapore dollar", "Singapore dollar", "cents", "cent", "cent"};
constexpr Cur TRY{"Turkish lira", "Turkish lira", "Turkish lira", "", "", ""};
constexpr Cur TWD{"New Taiwan dollars", "New Taiwan dollar", "New Taiwan dollar", "cents", "cent", "cent"};
constexpr Cur ZAR{"South African rand", "South African rand", "South African rand", "cents", "cent", "cent"};
constexpr Cur AED{"UAE dirhams", "UAE dirham", "UAE dirham", "", "", ""};
constexpr Cur ARS{"Argentine pesos", "Argentine peso", "Argentine peso", "", "", ""};
constexpr Cur CLP{"Chilean pesos", "Chilean peso", "Chilean peso", "", "", ""};
constexpr Cur CZK{"Czech koruna", "Czech koruna", "Czech koruna", "", "", ""};
constexpr Cur EGP{"Egyptian pounds", "Egyptian pound", "Egyptian pound", "", "", ""};
constexpr Cur HUF{"Hungarian forints", "Hungarian forint", "Hungarian forint", "", "", ""};
constexpr Cur IDR{"Indonesian rupiah", "Indonesian rupiah", "Indonesian rupiah", "", "", "", true};
constexpr Cur ILS{"Israeli shekels", "Israeli shekel", "Israeli shekel", "", "", ""};
constexpr Cur KES{"Kenyan shillings", "Kenyan shilling", "Kenyan shilling", "", "", ""};
constexpr Cur MYR{"Malaysian ringgit", "Malaysian ringgit", "Malaysian ringgit", "", "", ""};
constexpr Cur NGN{"Nigerian naira", "Nigerian naira", "Nigerian naira", "", "", ""};
constexpr Cur PKR{"Pakistani rupees", "Pakistani rupee", "Pakistani rupee", "", "", ""};
constexpr Cur SAR{"Saudi riyals", "Saudi riyal", "Saudi riyal", "", "", ""};
constexpr Cur THB{"Thai baht", "Thai baht", "Thai baht", "", "", ""};
constexpr Cur UAH{"Ukrainian hryvnias", "Ukrainian hryvnia", "Ukrainian hryvnia", "", "", ""};
constexpr Cur VND{"Vietnamese dong", "Vietnamese dong", "Vietnamese dong", "", "", "", true};
constexpr Cur BDT{"Bangladeshi taka", "Bangladeshi taka", "Bangladeshi taka", "", "", ""};
constexpr Cur BGN{"Bulgarian leva", "Bulgarian lev", "Bulgarian lev", "", "", ""};
constexpr Cur COP{"Colombian pesos", "Colombian peso", "Colombian peso", "", "", ""};
constexpr Cur DOP{"Dominican pesos", "Dominican peso", "Dominican peso", "", "", ""};
constexpr Cur ETB{"Ethiopian birr", "Ethiopian birr", "Ethiopian birr", "", "", ""};
constexpr Cur FJD{"Fijian dollars", "Fijian dollar", "Fijian dollar", "cents", "cent", "cent"};
constexpr Cur GHS{"Ghanaian cedis", "Ghanaian cedi", "Ghanaian cedi", "", "", ""};
constexpr Cur ISK{"Icelandic kronur", "Icelandic krona", "Icelandic krona", "", "", ""};
constexpr Cur JMD{"Jamaican dollars", "Jamaican dollar", "Jamaican dollar", "cents", "cent", "cent"};
constexpr Cur KWD{"Kuwaiti dinars", "Kuwaiti dinar", "Kuwaiti dinar", "", "", ""};
constexpr Cur KZT{"Kazakhstani tenge", "Kazakhstani tenge", "Kazakhstani tenge", "", "", ""};
constexpr Cur LKR{"Sri Lankan rupees", "Sri Lankan rupee", "Sri Lankan rupee", "", "", ""};
constexpr Cur QAR{"Qatari riyals", "Qatari riyal", "Qatari riyal", "", "", ""};
constexpr Cur RON{"Romanian lei", "Romanian leu", "Romanian leu", "", "", ""};
constexpr Cur TTD{"Trinidad and Tobago dollars", "Trinidad and Tobago dollar", "Trinidad and Tobago dollar", "cents",
                  "cent", "cent"};
constexpr Cur TZS{"Tanzanian shillings", "Tanzanian shilling", "Tanzanian shilling", "", "", ""};
constexpr Cur UGX{"Ugandan shillings", "Ugandan shilling", "Ugandan shilling", "", "", ""};
constexpr Cur XCD{"East Caribbean dollars", "East Caribbean dollar", "East Caribbean dollar", "cents", "cent",
                  "cent"};
constexpr Cur NAD{"Namibian dollars", "Namibian dollar", "Namibian dollar", "cents", "cent", "cent"};
constexpr Cur BZD{"Belize dollars", "Belize dollar", "Belize dollar", "cents", "cent", "cent"};
constexpr Cur ZWL{"Zimbabwean dollars", "Zimbabwean dollar", "Zimbabwean dollar", "cents", "cent", "cent"};

struct Named {
    std::string_view key;
    const Cur* cur;
};

const Named CODES[] = {
    {"AED", &AED}, {"ARS", &ARS}, {"AUD", &AUD}, {"BDT", &BDT}, {"BGN", &BGN}, {"BRL", &BRL}, {"CAD", &CAD},
    {"CHF", &CHF}, {"CLP", &CLP}, {"CNY", &CNY}, {"COP", &COP}, {"CZK", &CZK}, {"DKK", &DKK}, {"DOP", &DOP},
    {"EGP", &EGP}, {"ETB", &ETB}, {"EUR", &EURO}, {"FJD", &FJD}, {"GBP", &GBP}, {"GHS", &GHS}, {"HKD", &HKD},
    {"HUF", &HUF}, {"IDR", &IDR}, {"ILS", &ILS}, {"INR", &INR}, {"ISK", &ISK}, {"JMD", &JMD}, {"JPY", &JPY},
    {"KES", &KES}, {"KRW", &KRW}, {"KWD", &KWD}, {"KZT", &KZT}, {"LKR", &LKR}, {"MXN", &MXN}, {"MYR", &MYR},
    {"NGN", &NGN}, {"NOK", &NOK}, {"NZD", &NZD}, {"PKR", &PKR}, {"PLN", &PLN}, {"QAR", &QAR}, {"RMB", &RMB},
    {"RON", &RON}, {"RUB", &RUB}, {"SAR", &SAR}, {"SEK", &SEK}, {"SGD", &SGD}, {"THB", &THB}, {"TRY", &TRY},
    {"TTD", &TTD}, {"TWD", &TWD}, {"TZS", &TZS}, {"UAH", &UAH}, {"UGX", &UGX}, {"USD", &USD}, {"VND", &VND},
    {"XCD", &XCD}, {"ZAR", &ZAR},
};

// Written right before '$', with no space: US$5, C$20, A$1.2M, CDN$15, USD$50.
const Named DOLLAR_PREFIXES[] = {
    {"A", &AUD},   {"AU", &AUD},   {"AUD", &AUD}, {"BZ", &BZD},  {"C", &CAD},   {"CA", &CAD},  {"CAD", &CAD},
    {"CAN", &CAD}, {"CDN", &CAD},  {"EC", &XCD},  {"FJ", &FJD},  {"HK", &HKD},  {"HKD", &HKD}, {"J", &JMD},
    {"Mex", &MXN}, {"MX", &MXN},   {"N", &NAD},   {"NT", &TWD},  {"NZ", &NZD},  {"NZD", &NZD}, {"R", &BRL},
    {"RD", &DOP},  {"S", &SGD},    {"SG", &SGD},  {"SGD", &SGD}, {"TT", &TTD},  {"US", &USD},  {"USD", &USD},
    {"Z", &ZWL},
};

// The same prefixes written with a space before '$' ("US $5 million"); not after a word that makes them a place
// ("in the US $5 buys a coffee").
const rv::WordSet SPACED_PREFIX_BLOCK{"across", "around", "in", "inside", "outside", "the", "throughout", "within"};

template <size_t N>
const Cur* lookup(const Named (&table)[N], std::string_view key) {
    for (const Named& n : table)
        if (n.key == key) return n.cur;
    return nullptr;
}

const Cur* symbol_currency(uint32_t cp) {
    switch (cp) {
        case '$': case 0xFF04: case 0xFE69: return &DOLLAR;
        case 0x20AC: return &EURO;
        case 0xA3: case 0xFFE1: return &POUND;
        case 0xA5: case 0xFFE5: return &YEN;
        case 0x20B9: case 0x20A8: return &RUPEE;
        case 0x20A9: case 0xFFE6: return &WON;
        case 0x20BD: return &RUBLE;
        case 0x20BA: return &LIRA;
        case 0x20B1: return &PESO;
        case 0x20AA: return &SHEKEL;
        case 0x20A6: return &NAIRA;
        case 0x20BF: return &BITCOIN;
        case 0x20AB: return &DONG;
        case 0x0E3F: return &BAHT;
        case 0x20B4: return &HRYVNIA;
        case 0x20B5: return &CEDI;
        case 0x20B8: return &TENGE;
        case 0x20BE: return &LARI;
        case 0x20BC: return &MANAT;
        case 0x20B2: return &GUARANI;
        case 0x20AD: return &KIP;
        case 0x20AE: return &TUGRIK;
        default: return nullptr;
    }
}

// Currency abbreviations written as words before the amount ("Rs. 500", "kr 60", "KSh 1,200", "Fr. 20"). The
// key is the exact word; dot: the abbreviation may carry a period.
struct WordMarker {
    std::string_view key;
    const Cur* cur;
    bool dot;
};
const WordMarker WORD_BEFORE[] = {
    {"Fr", &FRANC, true},  {"KSh", &KES, true}, {"Ksh", &KES, true}, {"Rp", &RUPIAH, true},
    {"Rs", &RUPEE, true},  {"SFr", &CHF, true}, {"kr", &KRONE, true}, {"sFr", &CHF, true},
    {"z\xC5\x82", &ZLOTY, false},
};
// ... and after the amount ("60 kr", "50 zł", "100元", "500 Rs").
const WordMarker WORD_AFTER[] = {
    {"K\xC4\x8D", &KORUNA, false}, {"Rs", &RUPEE, false},        {"kr", &KRONE, false},
    {"z\xC5\x82", &ZLOTY, false},  {"\xE5\x85\x83", &RMB, false}, {"\xE5\x86\x86", &YEN, false},
};
// ISO codes written in lower case after a number in chat and notes ("50 usd").
const Named LOWER_CODES[] = {
    {"aud", &AUD}, {"cad", &CAD}, {"chf", &CHF}, {"cny", &CNY}, {"eur", &EURO}, {"gbp", &GBP}, {"hkd", &HKD},
    {"inr", &INR}, {"jpy", &JPY}, {"mxn", &MXN}, {"nzd", &NZD}, {"sek", &SEK}, {"sgd", &SGD}, {"usd", &USD},
};

// Currency words written out after an amount with a magnitude letter ("15–20M dollars").
const rv::WordSet CURRENCY_WORDS{"dollar", "dollars", "euro", "euros", "pound", "pounds", "yen", "rupee", "rupees",
                                 "francs", "pesos", "yuan", "won", "rubles", "roubles", "reais", "reals", "kronor",
                                 "kroner", "rand", "shekels", "lira", "naira", "dirhams", "riyals", "bucks"};

// A magnitude written right after the number: "5M", "3.2bn", "20k", "5million".
std::string_view scale_suffix(std::string_view s) {
    static const rv::PhraseMap SUFFIX{
        {"B", "billion"},   {"BN", "billion"},   {"Bn", "billion"},     {"K", "thousand"},
        {"M", "million"},   {"MM", "million"},   {"MN", "million"},     {"Mn", "million"},
        {"T", "trillion"},  {"TN", "trillion"},  {"Tn", "trillion"},    {"b", "billion"},
        {"billion", "billion"}, {"bln", "billion"}, {"bn", "billion"},  {"k", "thousand"},
        {"m", "million"},   {"mil", "million"},  {"million", "million"}, {"mio", "million"}, {"mm", "million"},
        {"mln", "million"}, {"mn", "million"},  {"thousand", "thousand"}, {"tn", "trillion"},
        {"trillion", "trillion"}, {"trn", "trillion"}};
    const std::string_view* w = SUFFIX.find(s);
    return w ? *w : std::string_view();
}

// A magnitude word after a space: "5 million", "3.2 bn", "5 lakh" (pass the word in lower case).
std::string_view scale_word(std::string_view lower) {
    static const rv::PhraseMap WORDS{
        {"billion", "billion"}, {"bln", "billion"}, {"bn", "billion"},     {"crore", "crore"},
        {"crores", "crore"},    {"lakh", "lakh"},   {"lakhs", "lakh"},     {"mil", "million"},
        {"million", "million"}, {"mio", "million"}, {"mln", "million"},    {"mn", "million"},
        {"thousand", "thousand"}, {"tn", "trillion"}, {"trillion", "trillion"}, {"trn", "trillion"}};
    const std::string_view* w = WORDS.find(lower);
    return w ? *w : std::string_view();
}

bool ends_with(std::string_view s, std::string_view tail) {
    return s.size() >= tail.size() && s.substr(s.size() - tail.size()) == tail;
}

bool starts_with(std::string_view s, std::string_view head) { return s.substr(0, head.size()) == head; }

bool all_zero(std::string_view digits) { return digits.find_first_not_of('0') == std::string_view::npos; }

// ----------------------------------------------------------------------------------------------- token view

// Zero-width and format characters inside an amount ("$4.​99") are skipped; the span covers them.
bool invisible(uint32_t cp) {
    return cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0x2060 || cp == 0xFEFF || cp == 0xAD ||
           cp == 0xFE0F || cp == 0xFE0E;
}

// The tokens of a Doc with invisible characters skipped. Indexes are Doc indexes; NPOS is "none".
struct V {
    const Doc& d;
    explicit V(const Doc& doc) : d(doc) {}

    bool ok(size_t i) const { return i < d.size(); }
    bool hidden(size_t i) const { return ok(i) && d[i].kind == Kind::Punct && invisible(d[i].cp); }
    size_t next(size_t i) const {
        if (!ok(i)) return NPOS;
        do ++i;
        while (ok(i) && hidden(i));
        return ok(i) ? i : NPOS;
    }
    size_t prev(size_t i) const {
        while (ok(i) && i > 0) {
            --i;
            if (!hidden(i)) return i;
        }
        return NPOS;
    }
    // No whitespace between token i and the visible token before it.
    bool tight(size_t i) const {
        if (!ok(i) || i == 0) return false;
        for (size_t j = i; j > 0; --j) {
            if (d[j].space_before) return false;
            if (!hidden(j - 1)) return true;
        }
        return false;
    }
    bool word(size_t i) const { return ok(i) && d[i].kind == Kind::Word; }
    bool number(size_t i) const { return ok(i) && d[i].kind == Kind::Number; }
    bool punct(size_t i, uint32_t cp) const { return ok(i) && d.punct(i, cp); }
    uint32_t cp(size_t i) const { return ok(i) && d[i].kind == Kind::Punct ? d[i].cp : 0; }
    std::string_view str(size_t i) const { return d.str(i); }
    std::string_view low(size_t i) const { return d.low(i); }
    size_t len(size_t i) const { return d[i].end - d[i].start; }
    // Exactly one space character (plain, no-break, thin or narrow) between tokens a and b.
    bool one_space(size_t a, size_t b) const {
        if (!ok(a) || !ok(b) || b <= a || d[b].newline_before) return false;
        size_t p = d[a].end;
        if (d[b].start <= p) return false;
        uint32_t c = rv::decode(d.text, p);
        return p == d[b].start && (c == ' ' || c == 0xA0 || (c >= 0x2005 && c <= 0x200A) || c == 0x202F);
    }
};

// CODES is sorted by key: binary search.
template <size_t N>
const Cur* sorted_lookup(const Named (&table)[N], std::string_view key) {
    const Named* it =
        std::lower_bound(table, table + N, key, [](const Named& n, std::string_view k) { return n.key < k; });
    return it != table + N && it->key == key ? it->cur : nullptr;
}

const Cur* code_currency(const V& v, size_t i) {
    if (!v.word(i) || v.d[i].casing != rv::UPPER || v.len(i) != 3) return nullptr;
    return sorted_lookup(CODES, v.str(i));
}

bool is_dash(const V& v, size_t i) {
    uint32_t c = v.cp(i);
    return c == '-' || c == 0x2010 || c == 0x2012 || c == 0x2013 || c == 0x2014;
}

// A currency word marker after the amount: "kr", "zł", "元", or a code in lower case ("50 usd").
const Cur* word_after(const V& v, size_t i) {
    if (!v.word(i)) return nullptr;
    size_t n = v.len(i);
    if (n != 2 && n != 3) return nullptr;
    std::string_view s = v.str(i);
    if (n == 3 && v.d[i].casing == rv::LOWER)
        if (const Cur* c = sorted_lookup(LOWER_CODES, s)) return c;
    for (const WordMarker& w : WORD_AFTER)
        if (w.key == s) return w.cur;
    return nullptr;
}

// ---------------------------------------------------------------------------------------------------- amounts

struct Amount {
    bool ok = false;
    size_t first = 0, last = 0;  // tokens: the number through its magnitude
    std::string whole, frac;     // digits without separators; whole is empty for ".75"
    std::string_view scale;      // "thousand", "million", ... or empty
};

struct Parse {
    bool pence = false;   // a stuck "p" may follow ("50p")
    bool indian = false;  // lakh grouping is allowed ("₹1,00,000")
    bool dots = false;    // a dot groups thousands ("Rp 50.000")
};

// Token i is a letter or digit glued to the token before it.
bool glued(const V& v, size_t i) { return v.tight(i) && (v.word(i) || v.number(i)); }

// A word that may stick to the number without a space: a magnitude, a currency code or word, "pp" (per person).
bool glued_word_ok(const V& v, size_t k, const Parse& o) {
    std::string_view s = v.str(k);
    if (code_currency(v, k) || word_after(v, k) || s == "pp" || s == "psf") return true;
    return o.pence && s == "p";
}

// Digit groups after the first group (at most three digits): "1,250,000", "1 250 000", "1'250'000". Returns the
// last token; adds the digits to whole. sep is the separator used, 0 if none.
size_t groups(const V& v, size_t j, std::string& whole, char& sep, const Parse& o) {
    for (;;) {
        size_t c = v.next(j), g;
        char st;
        uint32_t cp = v.cp(c);
        if ((cp == ',' || cp == '\'' || cp == 0x2019 || (cp == '.' && o.dots)) && v.tight(c)) {
            st = cp == ',' ? ',' : cp == '.' ? '.' : '\'';
            g = v.next(c);
            if (!v.number(g) || !v.tight(g)) break;
        } else if (v.number(c) && v.one_space(j, c)) {
            st = ' ';
            g = c;
        } else {
            break;
        }
        if ((sep && st != sep) || v.len(g) != 3) break;
        whole += v.str(g);
        sep = st;
        j = g;
    }
    return j;
}

// Indian grouping: "1,00,000", "12,34,56,789" (pairs, then a final group of three).
size_t indian_groups(const V& v, size_t j, std::string& whole) {
    std::string w = whole;
    size_t k = j;
    int pairs = 0;
    for (;;) {
        size_t c = v.next(k), g = v.next(c);
        if (!(v.punct(c, ',') && v.tight(c) && v.number(g) && v.tight(g))) return j;
        size_t n = v.len(g);
        w += v.str(g);
        k = g;
        if (n == 3) break;
        if (n != 2) return j;
        ++pairs;
    }
    if (!pairs) return j;
    whole = w;
    return k;
}

// European grouping that a US reading cannot have: "2.345,67" (dots, then a decimal comma) and "1.234.567" (two
// or more dot groups). Returns the last token, adds to whole and frac; j if the shape is not there.
size_t eu_dots(const V& v, size_t j, std::string& whole, std::string& frac) {
    std::string w = whole;
    size_t k = j;
    int n = 0;
    for (;;) {
        size_t c = v.next(k), g = v.next(c);
        if (!(v.punct(c, '.') && v.tight(c) && v.number(g) && v.tight(g) && v.len(g) == 3)) break;
        w += v.str(g);
        k = g;
        ++n;
    }
    if (!n) return j;
    size_t c = v.next(k), f = v.next(c);
    std::string fr;
    if (v.punct(c, ',') && v.tight(c) && v.number(f) && v.tight(f) && v.len(f) == 2) {
        fr = std::string(v.str(f));
        k = f;
    } else if (n < 2) {
        return j;
    }
    size_t after = v.next(k);
    if (v.tight(after) && (v.number(after) || ((v.punct(after, '.') || v.punct(after, ',')) &&
                                               v.number(v.next(after)) && v.tight(v.next(after)))))
        return j;
    whole = w;
    frac = fr;
    return k;
}

// "5", "1,250,000", "4.99", ".75", "7.4M", "3.2bn", "5 million", "1 250 000", "2,50" (a decimal comma), "5,-".
// Not ok when the number sticks to letters or digits it cannot own ("5x", "1,234", "1.2.3", "$2b$12$...").
Amount parse_amount(const V& v, size_t i, const Parse& o = Parse()) {
    Amount a;
    size_t j;
    if (v.punct(i, '.') && v.number(v.next(i)) && v.tight(v.next(i))) {
        j = v.next(i);
        a.frac = std::string(v.str(j));
    } else if (v.number(i)) {
        j = i;
        a.whole = std::string(v.str(i));
        char sep = 0;
        if (a.whole.size() <= 3) {
            if (o.indian && a.whole.size() <= 2) j = indian_groups(v, j, a.whole);
            if (j == i) {
                j = eu_dots(v, j, a.whole, a.frac);
                if (j != i) sep = '.';
            }
            if (j == i) {
                std::string w = a.whole;
                size_t e = groups(v, j, w, sep, o);
                if (sep == ' ') {  // "+49 30 123 4567", "5 100-meter races": a chain of numbers, not one
                    size_t s = v.next(e), p = v.prev(i);
                    bool after = (v.number(s) && v.one_space(e, s)) || (is_dash(v, s) && v.tight(s));
                    bool before = v.number(p) && v.one_space(p, i);
                    if (after || before) {
                        e = i;
                        w = a.whole;
                        sep = 0;
                    }
                }
                j = e;
                a.whole = w;
            }
        }
        size_t p = v.next(j), f = v.next(p);
        if (!a.frac.empty()) {
        } else if (v.punct(p, '.') && v.tight(p) && v.number(f) && v.tight(f) && sep != '.') {
            a.frac = std::string(v.str(f));
            j = f;
        } else if (v.punct(p, ',') && v.tight(p) && v.number(f) && v.tight(f) && v.len(f) <= 2 && sep != ',') {
            size_t q = v.next(f);  // "€2,50", "€1,5 billion": a decimal comma, never a group of three
            bool scale_next = v.word(q) && !v.d[q].newline_before &&
                              (!scale_word(v.low(q)).empty() || (v.tight(q) && !scale_suffix(v.str(q)).empty()));
            if (!((v.punct(q, ',') || v.punct(q, '.')) && v.tight(q) && v.number(v.next(q))) &&
                (v.len(f) == 2 || scale_next)) {
                a.frac = std::string(v.str(f));
                j = f;
            }
        } else if ((v.punct(p, ',') || v.punct(p, '.')) && v.tight(p) && v.tight(f) &&
                   (v.punct(f, '-') || v.punct(f, 0x2013) || v.punct(f, 0x2014)) && !glued(v, v.next(f))) {
            j = f;  // "€5,-": a whole amount
        }
        if ((a.whole.size() > 1 && a.whole[0] == '0') || a.whole.size() > 24) return a;
    } else {
        return a;
    }
    a.first = i;
    a.last = j;
    size_t k = v.next(j);
    if (o.indian && v.word(k) && !v.d[k].newline_before) {  // "₹15L", "Rs 2 Cr", "₹5 lacs"
        std::string_view w = v.str(k);
        std::string_view sc = w == "L" || w == "lac" || w == "lacs" || w == "Lac" ? "lakh"
                              : w == "Cr" || w == "cr" || w == "Crore" ? "crore" : "";
        if (!sc.empty() && (v.tight(k) || w != "L") && !glued(v, v.next(k))) {
            a.scale = sc;
            a.last = k;
            a.ok = true;
            return a;
        }
    }
    if (v.tight(k)) {
        if (v.word(k)) {
            std::string_view sc = scale_suffix(v.str(k));
            if (sc.empty()) sc = scale_word(v.low(k));
            if (!sc.empty()) {
                a.scale = sc;
                a.last = k;
            } else if (!glued_word_ok(v, k, o)) {
                return a;
            }
        } else if (v.number(k) || (v.punct(k, '$') && glued(v, v.next(k)))) {
            return a;  // "$12$Eix..." in a hash; "100$ discount" is fine
        } else if ((v.punct(k, ',') || v.punct(k, '.')) && v.number(v.next(k)) && v.tight(v.next(k))) {
            return a;
        }
    } else if (v.word(k) && !v.d[k].newline_before) {
        std::string_view sc = scale_word(v.low(k));
        if (!sc.empty()) {
            a.scale = sc;
            a.last = k;
        }
    }
    if (!a.scale.empty()) {
        size_t m = v.next(a.last);
        if (v.tight(m) && (v.word(m) || v.number(m) || v.punct(m, '$'))) {
            if (!(v.word(m) && glued_word_ok(v, m, o))) return a;
        }
        // "€4 Mio. for the project": an abbreviated magnitude takes its period inside a sentence
        size_t w = v.next(m);
        std::string_view x = v.low(a.last);
        if (v.punct(m, '.') && v.tight(m) && v.word(w) && !v.tight(w) && v.d[w].casing == rv::LOWER &&
            !v.d[w].newline_before && (x == "mio" || x == "mln" || x == "bln" || x == "trn" || x == "mil" || x == "bn"))
            a.last = m;
    }
    a.ok = true;
    return a;
}

// The number with its magnitude, no currency: "seven point four million", "point seven five", "two hundred".
std::string number_words(const Amount& a, bool with_scale = true) {
    std::string s;
    if (a.whole.empty()) {
        s = "point " + rv::digits(a.frac);
    } else {
        s = rv::cardinal_digits(a.whole);
        if (!a.frac.empty()) s += " point " + rv::digits(a.frac);
    }
    if (with_scale && !a.scale.empty()) {
        s += ' ';
        s += a.scale;
    }
    return s;
}

// Two decimals and no magnitude: read as a subunit ("$4.99", "$0.45", "$.50") where the currency has one.
bool cents_form(const Amount& a, const Cur& c) { return a.scale.empty() && a.frac.size() == 2 && !c.sub_pl.empty(); }

// A main unit and a subunit: "$4.99" (not "$0.99", "$5.00").
bool mixed_form(const Amount& a, const Cur& c) {
    return cents_form(a, c) && !all_zero(a.whole) && !all_zero(a.frac);
}

unsigned two_digits(std::string_view f) { return static_cast<unsigned>((f[0] - '0') * 10 + (f[1] - '0')); }

// "$4.99" -> four dollars and ninety-nine cents, "$5.00" -> five dollars, "$0.45" -> forty-five cents,
// "$7.4M" -> seven point four million dollars, "¥145.75" -> one hundred forty-five point seven five yen.
// adj: before a noun, singular units ("a ten dollar coupon"). code: written with an ISO code, where an amount
// below one keeps its decimals ("0.45 USD" -> zero point four five US dollars).
std::string read_money(const Amount& a, const Cur& c, bool adj, bool code) {
    std::string unit(adj ? c.adj : c.pl);
    if (!a.scale.empty()) return number_words(a) + " " + unit;
    bool zero_whole = all_zero(a.whole);
    bool cents = cents_form(a, c);
    if (a.frac.empty() || (cents && all_zero(a.frac))) {
        if (a.whole == "1") return "one " + std::string(c.sg);
        return rv::cardinal_digits(a.whole.empty() ? "0" : a.whole) + " " + unit;
    }
    if (cents && !(code && zero_whole)) {
        unsigned ct = two_digits(a.frac);
        std::string sub = rv::cardinal(ct) + " " + std::string(adj ? c.sub_adj : ct == 1 ? c.sub_sg : c.sub_pl);
        if (zero_whole) return sub;
        std::string main = a.whole == "1" ? "one " + std::string(c.sg) : rv::cardinal_digits(a.whole) + " " + unit;
        return main + " and " + sub;
    }
    return number_words(a) + " " + unit;
}

// "$200-350" -> two hundred to three hundred fifty dollars, "$3–5M" -> three to five million dollars,
// "$1.50-$2.75" -> one dollar and fifty cents to two dollars and seventy-five cents.
std::string read_range(Amount a, Amount b, const Cur& c, bool adj, bool code) {
    bool scaled = !a.scale.empty() || !b.scale.empty();  // "$1.18–1.22bn": both halves share the magnitude
    bool ca = !scaled && cents_form(a, c) && !all_zero(a.frac);
    bool cb = !scaled && cents_form(b, c) && !all_zero(b.frac);
    if (ca || cb) {
        if (ca && cb && all_zero(a.whole) && all_zero(b.whole))
            return rv::cardinal(two_digits(a.frac)) + " to " + rv::cardinal(two_digits(b.frac)) + " " +
                   std::string(adj ? c.sub_adj : c.sub_pl);
        return read_money(a, c, false, code) + " to " + read_money(b, c, adj, code);
    }
    if (!scaled && cents_form(a, c)) a.frac.clear();
    if (!scaled && cents_form(b, c)) b.frac.clear();
    bool shared = !a.scale.empty() && a.scale == b.scale;  // "$500M–800M": the magnitude once, at the end
    return number_words(a, !shared) + " to " + number_words(b) + " " + std::string(adj ? c.adj : c.pl);
}

// ---------------------------------------------------------------------------------------------- per units

const rv::PhraseMap PER_ABBR{
    {"GB", "gigabyte"},   {"GWh", "gigawatt hour"}, {"K", "thousand"},  {"L", "liter"},
    {"M", "million"},     {"MB", "megabyte"},    {"MMBTU", "million British thermal units"},
    {"MMBtu", "million British thermal units"}, {"MTok", "million tokens"}, {"MWh", "megawatt hour"},
    {"SF", "square foot"}, {"TB", "terabyte"},   {"bbl", "barrel"},    {"bu", "bushel"},
    {"cwt", "hundredweight"}, {"doz", "dozen"},  {"ft", "foot"},       {"g", "gram"},
    {"gal", "gallon"},    {"h", "hour"},         {"hr", "hour"},       {"hrs", "hour"},
    {"k", "thousand"},    {"kWh", "kilowatt hour"}, {"kg", "kilogram"}, {"km", "kilometer"},
    {"l", "liter"},       {"lb", "pound"},       {"lbs", "pound"},     {"mL", "milliliter"},
    {"mg", "milligram"},  {"mi", "mile"},        {"min", "minute"},    {"mins", "minute"},
    {"ml", "milliliter"}, {"mmBtu", "million British thermal units"}, {"mo", "month"}, {"mos", "month"},
    {"mth", "month"},     {"oz", "ounce"},       {"pc", "piece"},      {"pcs", "piece"},
    {"pp", "person"},     {"qt", "quart"},       {"qtr", "quarter"},   {"s", "second"},
    {"sec", "second"},    {"sf", "square foot"}, {"sqft", "square foot"}, {"sqm", "square meter"},
    {"t", "tonne"},       {"wk", "week"},        {"yd", "yard"},       {"yr", "year"},
    {"yrs", "year"}};

// Units that may stand after a slash with spaces around it ("$5 / month", "$5 /mo").
const rv::WordSet PER_SPACED{"barrel", "day",   "gallon", "guest",  "head",   "hour", "item",  "kilo",
                             "liter",  "litre", "mile",   "month",  "night",  "ounce", "page", "person",
                             "piece",  "pound", "room",   "seat",   "session", "share", "ticket", "ton",
                             "tonne",  "unit",  "user",   "visit",  "week",   "year"};

// "sq ft", "sq. ft.", "sq m" after a slash; end is the last token used.
bool square_unit(const V& v, size_t u, std::string& w, size_t& end) {
    if (!v.word(u) || v.low(u) != "sq") return false;
    size_t n = v.next(u);
    if (v.punct(n, '.') && v.tight(n)) n = v.next(n);
    if (!v.word(n) || v.d[n].newline_before) return false;
    std::string_view x = v.low(n);
    if (x != "ft" && x != "m" && x != "mi" && x != "km" && x != "yd") return false;
    w = x == "ft" ? "square foot" : x == "m" ? "square meter" : x == "mi" ? "square mile"
        : x == "km" ? "square kilometer" : "square yard";
    end = n;
    size_t dot = v.next(n), after = v.next(dot);  // "sq. ft." keeps its period only at the end of a sentence
    if (v.punct(dot, '.') && v.tight(dot) && v.ok(after) && !v.d[after].newline_before &&
        !(v.word(after) && (v.d[after].casing & (rv::UPPER | rv::TITLE))))
        end = dot;
    return true;
}

// "/mo" -> " per month", "/1M" -> " per one million", "/user/yr" -> " per user per year", "/ea" -> " each".
// Moves last to the final unit token; "" if no unit follows.
std::string per_units(const V& v, size_t& last, bool* spaced_any = nullptr) {
    std::string s;
    for (;;) {
        size_t sl = v.next(last), u = v.next(sl);
        if (v.word(sl) && v.low(sl) == "per" && !v.tight(sl) && v.word(u) && !v.tight(u) && !v.d[u].newline_before) {
            const std::string_view* e = PER_ABBR.find(v.str(u));  // "$15 per hr"
            std::string w;
            size_t end = u;
            if (square_unit(v, u, w, end) || (e && v.str(u).size() > 1 && *e != v.str(u))) {
                size_t after = v.next(end);
                if (v.tight(after) && (v.word(after) || v.number(after))) break;
                s += " per ";
                s += w.empty() ? std::string(*e) : w;
                last = end;
                continue;
            }
            break;
        }
        if (!v.punct(sl, '/') || !v.ok(u) || v.d[u].newline_before || v.d[sl].newline_before) break;
        bool spaced = !v.tight(sl) || !v.tight(u);
        std::string w;
        size_t end = u;
        if (square_unit(v, u, w, end)) {
        } else if (v.word(u)) {
            if (code_currency(v, u)) break;  // "EUR/USD": a currency pair
            const std::string_view* e = PER_ABBR.find(v.str(u));
            if (spaced && !e && !PER_SPACED.contains(v.low(u))) break;
            w = e ? std::string(*e) : std::string(v.str(u));
            size_t p = v.next(u);
            if (v.tight(p) && (v.cp(p) == 0xB2 || v.cp(p) == 0xB3)) {  // m², ft³
                if (v.str(u) == "m") w = "meter";
                w = (v.cp(p) == 0xB2 ? "square " : "cubic ") + w;
                end = p;
            } else if (v.tight(p) && v.number(p) && (v.str(p) == "2" || v.str(p) == "3") &&
                       (v.str(u) == "m" || v.str(u) == "ft")) {  // m2, ft3
                w = std::string(v.str(p) == "2" ? "square " : "cubic ") + (v.str(u) == "m" ? "meter" : "foot");
                end = p;
            }
            if (v.str(u) == "ea") {  // "$5/ea": each
                size_t after = v.next(end);
                if (v.tight(after) && (v.word(after) || v.number(after))) break;
                s += " each";
                last = end;
                continue;
            }
        } else if (v.number(u) && !spaced) {
            Amount q = parse_amount(v, u);
            if (!q.ok || !q.frac.empty()) break;
            size_t after = v.next(q.last);
            if (q.scale.empty() && !(v.word(after) && !v.tight(after))) break;  // "/1M", "/100 words"
            w = number_words(q);
            end = q.last;
        } else {
            break;
        }
        size_t after = v.next(end);
        if (v.tight(after) && (v.word(after) || v.number(after))) break;
        s += " per ";
        s += w;
        last = end;
        if (spaced && spaced_any) *spaced_any = true;
    }
    return s;
}

// After the amount: "+" (plus), "pp" / "p.p." (per person), "psf" (per square foot), "p.a." / "pa" (per annum).
// Moves last past what it reads.
std::string tail_once(const V& v, size_t& last) {
    size_t n = v.next(last);
    if (!v.ok(n) || v.d[n].newline_before) return "";
    if (v.punct(n, '+')) {  // "$50+", "$1M+", "$84 +,"; not "$5+$6", "$5 + $6"
        size_t after = v.next(n);
        if (v.tight(n) && v.tight(after) && (v.word(after) || v.number(after) || symbol_currency(v.cp(after))))
            return "";
        if (!v.tight(n) && !(after == NPOS || v.d[after].newline_before ||
                             (v.tight(after) && (v.punct(after, ',') || v.punct(after, '.') || v.punct(after, ';')))))
            return "";
        last = n;
        return " plus";
    }
    if (!v.word(n)) return "";
    std::string_view w = v.str(n);
    size_t dot = v.next(n), a = v.next(dot);
    if ((w == "p" || w == "P") && v.punct(dot, '.') && v.tight(dot) && v.word(a) && v.tight(a)) {
        std::string_view x = v.low(a);
        size_t dot2 = v.next(a);
        if ((x == "p" || x == "a") && v.punct(dot2, '.') && v.tight(dot2) && !glued(v, v.next(dot2))) {
            // "p.p.", "p.a."; at the end of a sentence the period stays for it
            size_t n2 = v.next(dot2);
            bool final = n2 == NPOS || v.d[n2].newline_before ||
                         (v.word(n2) && (v.d[n2].casing & (rv::UPPER | rv::TITLE)));
            last = final ? a : dot2;
            return x == "p" ? " per person" : " per annum";
        }
        return "";
    }
    size_t after = v.next(n);
    if (v.tight(after) && (v.word(after) || v.number(after))) return "";
    if (w == "pp" || w == "PP") {
        last = n;
        return " per person";
    }
    if (w == "psf" || w == "PSF") {
        last = n;
        return " per square foot";
    }
    if (w == "pa" && !v.tight(n)) {
        last = n;
        return " per annum";
    }
    return "";
}

std::string tail(const V& v, size_t& last) {
    std::string s = tail_once(v, last);
    if (s == " plus") s += tail_once(v, last);  // "£90k+ p.a."
    return s;
}

// ------------------------------------------------------------------------------------------------- context

const rv::WordSet DETERMINERS{"a",   "an",  "another", "her", "his",  "its",
                              "my",  "our", "that",    "the", "their", "this", "your"};

// Contractions that look like possessives: "it's 10¢ cheaper".
const rv::WordSet CONTRACTED{"he", "here", "it", "let", "she", "that", "there", "what", "where", "who"};

// Words that may stand between a determiner and the amount: "an initial $5 deposit".
const rv::WordSet BRIDGE{"additional", "annual", "average", "combined", "estimated", "extra",   "final",
                         "first",      "flat",   "full",    "hefty",    "initial",   "massive", "mere",
                         "modest",     "monthly", "new",    "one-time", "proposed",  "second",  "weekly",
                         "whopping",   "yearly"};

// Adjectives that need a noun after them before the amount counts as adjectival ("the $5M annual budget").
const rv::WordSet NEEDS_NOUN{"annual", "daily", "flat", "hourly", "monthly", "nightly", "quarterly", "weekly",
                             "yearly"};

// Words after an amount that show it is not used as an adjective: "the $96 for", "the $1.5M total".
const rv::WordSet NOT_NOUN{
    "a", "about", "above", "according", "across", "after", "again", "against", "ago", "ahead", "all", "along",
    "already", "also", "although", "altogether", "always", "am", "among", "an", "and", "apart", "apiece", "are",
    "around", "as", "at", "away", "back", "be", "became", "because", "becomes", "been", "before", "behind", "being",
    "below", "better", "between", "beyond", "bigger", "both", "but", "by", "came", "can", "cash", "cheaper",
    "comes", "costlier", "could", "counts", "despite", "did", "do", "does", "down", "during", "each", "earlier",
    "either", "even", "every", "excluding", "extra", "fewer", "for", "from", "gets", "goes", "got", "greater",
    "gross", "had", "has", "have", "he", "her", "here", "higher", "him", "his", "i", "if", "in", "including",
    "instead", "into", "is", "it", "its", "just", "larger", "last", "later", "left", "less", "lower", "made",
    "makes", "may", "me", "might", "minus", "more", "must", "my", "neither", "net", "never", "next", "nor", "not",
    "now", "of", "off", "on", "once", "only", "onto", "or", "our", "out", "outright", "over", "overall", "per",
    "plus", "pricier", "remaining", "remains", "richer", "seems", "shall", "she", "short", "should", "shy", "since",
    "smaller", "so", "stays", "still", "takes", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "though", "through", "throughout", "till", "times", "to", "today", "tomorrow",
    "tonight", "too", "took", "total", "toward", "towards", "u", "under", "unless", "until", "up", "upfront",
    "upon", "ur", "us", "versus", "via", "vs", "was", "we", "went", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whom", "whose", "will", "with", "within", "without", "worse", "worth", "would",
    "yesterday", "yet", "you", "your"};

bool noun_word(const V& v, size_t i) {
    if (!v.word(i) || v.tight(i) || v.d[i].newline_before) return false;
    return !NOT_NOUN.contains(v.low(i)) && !code_currency(v, i);
}

bool noun_ahead(const V& v, size_t i) {
    if (!noun_word(v, i)) return false;
    std::string_view w = v.low(i);
    if (NEEDS_NOUN.contains(w) || ends_with(w, "ly") || ends_with(w, "ed")) {
        size_t j = v.next(i);
        if (!noun_word(v, j)) return false;
        std::string_view x = v.low(j);
        return !ends_with(x, "ly") && !ends_with(x, "ed");
    }
    return true;
}

bool possessive(std::string_view w) {
    size_t cut = ends_with(w, "'s") ? 2 : ends_with(w, "\xE2\x80\x99s") ? 4 : 0;
    return cut && !CONTRACTED.contains(w.substr(0, w.size() - cut));
}

// S02: money before a noun takes the singular: "a $10 coupon", "its €2.6m budget", "($15 rental)". The labels keep
// the plural after "the" when the amount has a scale ("the $6M budget": six million dollars budget) or a period
// adjective follows ("the $159 monthly plan").
bool adjectival(const V& v, size_t first, size_t last, bool scaled) {
    size_t n = v.next(last), p = v.prev(first);
    if (p == NPOS) return false;
    if (v.punct(p, '('))  // "($15 rental)", not "($6,250,000 guaranteed)"
        return noun_word(v, n) && !ends_with(v.low(n), "ed") && v.punct(v.next(n), ')');
    if (v.punct(n, '-') && v.tight(n)) {  // a compound modifier: "a $5-off coupon", "a $10,000-a-year raise"
        size_t w = v.next(n);
        return v.word(w) && v.tight(w) && v.d[w].casing == rv::LOWER && scale_word(v.low(w)).empty() &&
               scale_suffix(v.str(w)).empty();
    }
    if (!noun_ahead(v, n)) return false;
    for (int hop = 0; hop < 6 && p != NPOS; ++hop, p = v.prev(p)) {
        if (v.word(p)) {
            std::string_view w = v.low(p);
            if (DETERMINERS.contains(w) || possessive(w))
                return !(w == "the" && (scaled || NEEDS_NOUN.contains(v.low(n))));
            bool compound_tail = v.tight(p) && is_dash(v, v.prev(p));  // "a 5-year, $262M extension"
            if (!BRIDGE.contains(w) && !compound_tail) return false;
        } else if (!(v.punct(p, ',') || is_dash(v, p) || v.number(p))) {
            return false;
        }
    }
    return false;
}

// Words before "($1.2M)" that make the brackets an accounting negative rather than an aside
// ("the reserve showed ($950,000)" against "the first night ($162.40)").
const rv::WordSet NEG_CUES{
    "a",        "an",       "and",      "appear",   "appears",  "as",       "at",       "booked",
    "books",    "by",       "came",     "carried",  "carries",  "carry",    "closed",   "closes",
    "cost",     "costs",    "declined", "dropped",  "equals",   "fell",     "from",     "gave",
    "gives",    "hit",      "in",       "into",     "is",       "list",     "listed",   "lists",
    "logged",   "lost",     "minus",    "net",      "of",       "or",       "post",     "posted",
    "posts",    "printed",  "reached",  "read",     "reads",    "record",   "recorded", "records",
    "report",   "reported", "reports",  "returned", "show",     "showed",   "showing",  "shown",
    "shows",    "stood",    "stands",   "swung",    "the",      "to",       "total",    "totaled",
    "totalled", "totals",   "was",      "were",     "widened",  "with",     "yielded"};

bool accounting(const V& v, size_t open) {
    const rv::Token& t = v.d[open];
    size_t p = v.prev(open);
    if (p == NPOS || t.newline_before) return true;
    if (t.start > 0 && v.d.text[t.start - 1] == '\t') return true;  // a table cell
    if (v.word(p)) return NEG_CUES.contains(v.low(p)) || v.d[p].newline_before || v.prev(p) == NPOS;
    uint32_t c = v.cp(p);
    return c == '|' || c == ':' || c == ',' || c == ';' || c == '=';
}

// A sign written right before the money: "-$7.4M", "−$0.31", "+$2,120,406.50"; not a hyphen after a word or
// a number ("$200-$350").
const char* sign_before(const V& v, size_t first, size_t& sign) {
    sign = v.prev(first);
    if (sign == NPOS || !v.tight(first)) return nullptr;
    uint32_t c = v.cp(sign);
    if (c != '-' && c != 0x2212 && c != 0x2013 && c != '+') return nullptr;
    size_t q = v.prev(sign);
    if (v.tight(sign) && (v.word(q) || v.number(q) || v.punct(q, '/'))) return nullptr;
    return c == '+' ? "plus " : "minus ";
}

// Words around "$1" that make it a placeholder in code or a regular expression, not money.
const rv::WordSet CODE_CUES{"argv",        "awk",          "backreference", "backreferences", "bash",   "capture",
                            "captured",    "echo",         "perl",          "placeholder",    "placeholders",
                            "positional",  "printf",       "regex",         "regexp",         "script", "scripts",
                            "sed",         "shell",        "substitution"};

bool sentence_end(const V& v, size_t i) {
    uint32_t c = v.cp(i);
    return (c == '.' || c == '!' || c == '?') && v.ok(i + 1) && v.d[i + 1].space_before;
}

// Looks for a cue in the sentence around token k (at most 16 tokens each way).
bool code_placeholder(const V& v, size_t k, const Amount& a) {
    if (!a.frac.empty() || !a.scale.empty() || a.whole.size() != 1) return false;
    for (size_t i = k, n = 0; i > 0 && n < 16; ++n) {
        --i;
        if (sentence_end(v, i) || v.d[i + 1].newline_before) break;
        if (v.word(i) && CODE_CUES.contains(v.low(i))) return true;
    }
    for (size_t i = k + 1, n = 0; i < v.d.size() && n < 16 && !v.d[i].newline_before; ++i, ++n) {
        if (v.word(i) && CODE_CUES.contains(v.low(i))) return true;
        if (sentence_end(v, i)) break;
    }
    return false;
}

bool superscript_digit(uint32_t c) {
    return c == 0xB9 || c == 0xB2 || c == 0xB3 || c == 0x2070 || (c >= 0x2074 && c <= 0x2079);
}

// ----------------------------------------------------------------------------------------------- proposals

struct Money {
    size_t first = 0, last = 0;              // tokens of the whole expression
    size_t plain_first = 0, plain_last = 0;  // the first amount with its currency alone (a fallback)
    bool has_plain = false;
    const Cur* cur = nullptr;
    Amount a, b;             // b.ok: a range from a to b
    bool code = false;       // written with an ISO code
    bool after = false;      // the currency marker follows the number ("100$", "500 €")
    bool interval = false;   // "[$4M, $6M]"
    const char* inner = "";  // a sign inside the expression: "CHF -4.2 million", "JPY (300 million)"
    std::string per;         // " per month"
    bool per_spaced = false; // "$5 /month": a slash with a space, read before a path could take it
    bool entity = false;     // the currency is an HTML entity ("&#163;12"), whose decoding alone must lose
    std::string word;        // a currency word read as written: "15–20M dollars"
};

bool has_hidden(const V& v, size_t first, size_t last) {
    for (size_t i = first + 1; i < last; ++i)
        if (v.hidden(i)) return true;
    return false;
}

// "a $80 fee" -> "an eighty dollar fee": the article before the amount follows its first sound.
bool vowel_start(std::string_view w) {
    return starts_with(w, "eight") || starts_with(w, "eleven");
}

void emit(const V& v, rv::Out& out, Money& m) {
    // a footnote mark stuck to the amount is left out: "$1.8M¹ committed"
    for (size_t n = v.next(m.last); v.tight(n) && superscript_digit(v.cp(n)); n = v.next(n)) m.last = n;
    // "Rs. 5,000/-": a whole amount, no subunit
    size_t sl = v.next(m.last), dash = v.next(sl);
    if (m.per.empty() && v.punct(sl, '/') && v.tight(sl) && v.punct(dash, '-') && v.tight(dash) &&
        !glued(v, v.next(dash)))
        m.last = dash;
    std::string extra = tail(v, m.last);
    size_t first = m.first, last = m.last, s = NPOS;
    std::string sign = m.inner;
    // A number with a currency marker is money whatever the words around it say ("priced at 1200€" is not a
    // military time); longer constructs score above the readings of their parts.
    int score = 70;
    if (m.b.ok) score = 72;
    // "62 000 kr": above the reading of the space-grouped number alone (numbers_core, 71)
    for (size_t k = m.a.first + 1; k <= m.a.last && score < 72; ++k)
        if (v.number(k) && !v.tight(k)) score = 72;
    if (m.interval) score = 74;
    if (!m.per.empty() || !extra.empty()) score = 76;
    if ((!extra.empty() && extra != " plus") || m.per_spaced) score = 82;  // over "p.a." and "/month" alone
    if (!m.interval && sign.empty()) {
        if (const char* w = sign_before(v, first, s)) {
            sign = w;
            first = s;
            score = std::max(score, 72);
        } else {
            size_t o = v.prev(first), c = v.next(last);
            if (v.punct(o, '(') && v.tight(first) && v.punct(c, ')') && v.tight(c) && accounting(v, o)) {
                sign = "minus ";
                first = o;
                last = c;
                score = std::max(score, 72);
            } else if ((v.punct(o, '<') || v.punct(o, '>')) && v.tight(first) && !v.tight(o) &&
                       !v.d[o].newline_before && v.prev(o) != NPOS) {
                sign = v.punct(o, '<') ? "less than " : "more than ";  // "<€30", like "<$20"
                first = o;
            }
        }
    }
    // a scaled amount with a sign keeps the plural too: "a -$4M loss" (minus four million dollars loss); so do a
    // main unit with a subunit ("a $4.99 dessert") and a currency written after the number ("the 500€ grant")
    bool scaled = !m.a.scale.empty() && m.a.scale != "billion";
    bool plural_only = mixed_form(m.a, *m.cur) || (m.b.ok && mixed_form(m.b, *m.cur)) ||
                       (m.after && m.cur != &CENTS && m.cur != &PENCE) || !m.word.empty();
    bool adj = m.per.empty() && extra.empty() && !plural_only && !(scaled && !sign.empty()) &&
               adjectival(v, first, last, scaled);
    std::string words;
    if (!m.word.empty())
        words = (m.b.ok ? number_words(m.a, !(m.a.scale == m.b.scale)) + " to " + number_words(m.b)
                        : number_words(m.a)) + " " + m.word;
    else
        words = m.b.ok ? read_range(m.a, m.b, *m.cur, adj, m.code) : read_money(m.a, *m.cur, adj, m.code);
    words = sign + words + m.per + extra;
    // An invisible character inside the amount must not split it: outrank its deletion (structure, 90) and a
    // number reading across it (numbers_core, 91).
    if (has_hidden(v, first, last)) score = 92;
    if (m.entity) score = std::max(score, 86);
    size_t art = v.prev(first);
    if (vowel_start(words) && v.word(art) && (v.str(art) == "a" || v.str(art) == "A") && !v.tight(first) &&
        v.one_space(art, first)) {
        words = (v.str(art) == "A" ? "An " : "an ") + words;
        first = art;
    }
    out.add(v.d[first].start, v.d[last].end, words, score);
    // The first amount alone, in case a longer reading loses on one of its parts.
    if (m.has_plain && (first != m.plain_first || last != m.plain_last))
        out.add(v.d[m.plain_first].start, v.d[m.plain_last].end, read_money(m.a, *m.cur, false, m.code),
                has_hidden(v, m.plain_first, m.plain_last) ? 91 : 68);
}

// The tokens after a bare second amount let it share the currency of the first: the end, closing punctuation, or
// a word that is no unit of its own ("$5 to 10 a night", not "$528 or 90 days").
const rv::WordSet RANGE_TAIL{"a",    "an",   "and", "apiece", "at",    "but",  "depending", "each", "for",
                             "from", "in",   "on",  "or",     "over",  "per",  "plus",      "so",   "that",
                             "this", "under", "when", "which", "while", "with"};

bool shares_unit(const V& v, size_t last) {
    size_t n = v.next(last);
    if (n == NPOS || v.d[n].newline_before) return true;
    if (v.word(n)) return !v.tight(n) && RANGE_TAIL.contains(v.low(n));
    if (v.number(n)) return false;
    uint32_t c = v.cp(n);
    if (c == ':' || c == '.' || c == ',') return !(v.tight(v.next(n)) && v.number(v.next(n)));  // "10:30"
    return c == ';' || c == ')' || c == ']' || c == '!' || c == '?' || c == '/' || c == '"' || c == 0x201D ||
           c == 0x2019 || c == 0x2014 || c == 0x2013 || c == 0x2026;
}

// "$200-350", "$200–$350", "$5 – $10", "$3–5M", "$5 - 10" after amount a; sym is the currency symbol accepted
// before the second number (0: none).
void range_after(const V& v, Money& m, uint32_t sym) {
    size_t d = v.next(m.a.last);
    if (!is_dash(v, d)) return;
    size_t s = v.next(d);
    bool has_sym = sym && v.punct(s, sym);
    size_t n = has_sym ? v.next(s) : s;
    if (has_sym && !v.tight(n)) return;
    bool joined = v.tight(d) && v.tight(s);
    bool spaced = !v.tight(d) && !v.tight(s) && v.ok(s) && !v.d[s].newline_before;
    if (!(joined || spaced)) return;
    Amount b = parse_amount(v, n);
    if (!b.ok || (spaced && !has_sym && !shares_unit(v, b.last))) return;
    if (spaced && (v.punct(v.next(b.last), '=') || v.punct(v.next(b.last), '+'))) return;  // "$40 - $5 = $35"
    m.b = b;
    m.last = b.last;
}

// "$5 to 10", "USD 5 to 10 million": a bare second amount after "to" shares the currency.
void range_to(const V& v, Money& m) {
    size_t t = v.next(m.a.last), n = v.next(t);
    if (!v.word(t) || v.low(t) != "to" || v.tight(t) || !v.number(n) || v.tight(n) || v.d[n].newline_before) return;
    Amount b = parse_amount(v, n);
    if (!b.ok || !shares_unit(v, b.last)) return;
    if (!m.a.scale.empty() && b.scale.empty()) return;  // "$5M to 10": not one range
    m.b = b;
    m.last = b.last;
}

// "[$4M, $6M]" -> four to six million dollars.
bool interval_around(const V& v, Money& m, uint32_t sym) {
    size_t o = v.prev(m.first), c = v.next(m.a.last), s = v.next(c), n = v.next(s);
    if (!sym || !(v.punct(o, '[') && v.tight(m.first) && v.punct(c, ',') && v.tight(c) && v.punct(s, sym) &&
                  v.tight(n)))
        return false;
    Amount b = parse_amount(v, n);
    size_t e = b.ok ? v.next(b.last) : NPOS;
    if (!v.punct(e, ']') || !v.tight(e)) return false;
    m.b = b;
    m.first = o;
    m.last = e;
    m.interval = true;
    return true;
}

// A currency word or code repeated after the amount: "$75 dollars", "$5 USD" (the code decides the name).
void repeated_unit(const V& v, Money& m) {
    size_t u = v.next(m.last);
    if (!v.word(u) || v.d[u].newline_before) return;
    const Cur* c = code_currency(v, u);
    if (!c && v.len(u) == 3 && v.d[u].casing == rv::LOWER) c = sorted_lookup(LOWER_CODES, v.str(u));  // "$5 usd"
    if (c) {
        if (ends_with(c->pl, m.cur->pl)) {
            m.cur = c;
            m.last = u;
        }
    } else if (!v.tight(u) && (v.low(u) == m.cur->pl || v.low(u) == m.cur->sg)) {
        m.last = u;
    }
}

Parse parse_for(const Cur* c) {
    Parse o;
    o.indian = c == &RUPEE || c == &INR;
    o.dots = c && c->dot_groups;
    return o;
}

// The amount after a currency marker that ends at token k, with a sign or accounting brackets between them
// ("CHF -4.2 million", "JPY (300 million)"). spaced: one space is allowed before the amount.
bool amount_after(const V& v, Money& m, size_t k, bool spaced, const Cur* cur) {
    size_t n = v.next(k);
    if (!v.ok(n) || v.d[n].newline_before || (!spaced && !v.tight(n))) return false;
    uint32_t c = v.cp(n);
    Parse o = parse_for(cur);
    if ((c == '-' || c == 0x2212 || c == '(') && v.tight(v.next(n))) {
        m.a = parse_amount(v, v.next(n), o);
        if (!m.a.ok) return false;
        if (c == '(') {
            size_t close = v.next(m.a.last);
            if (!v.punct(close, ')') || !v.tight(close)) return false;
            m.last = close;
        } else {
            m.last = m.a.last;
        }
        m.inner = "minus ";
        return true;
    }
    m.a = parse_amount(v, n, o);
    m.last = m.a.last;
    return m.a.ok;
}

// "&euro;59", "&#36;5", "&#x20AC;59", "50&cent;": a currency symbol written as an HTML entity; returns its ';'
// token or NPOS.
size_t entity(const V& v, size_t amp, const Cur*& cur) {
    struct E {
        std::string_view name;
        const Cur* cur;
    };
    static const E ENTITIES[] = {{"euro;", &EURO},   {"pound;", &POUND},  {"yen;", &YEN},     {"dollar;", &DOLLAR},
                                 {"cent;", &CENTS},  {"#36;", &DOLLAR},   {"#8364;", &EURO},  {"#163;", &POUND},
                                 {"#165;", &YEN},    {"#162;", &CENTS},   {"#x24;", &DOLLAR}, {"#x20ac;", &EURO},
                                 {"#xa3;", &POUND},  {"#xa5;", &YEN},     {"#xa2;", &CENTS}};
    uint32_t b = v.d[amp].end;
    for (const E& e : ENTITIES) {
        if (!v.d.lower_at(b, e.name)) continue;
        size_t semi = v.d.token_at(b + static_cast<uint32_t>(e.name.size()) - 1);
        if (!v.punct(semi, ';')) return NPOS;
        cur = e.cur;
        return semi;
    }
    return NPOS;
}

// A unit in a table header or a note, no number: "Revenue ($M)", "(US$m)", "($000s)", "in $ millions", "in €bn"
// -> "millions of dollars". first is the marker's first token (a prefix included); end its last.
void unit_header(const V& v, rv::Out& out, size_t first, size_t end, const Cur* cur) {
    size_t w = v.next(end), p = v.prev(first);
    if (!v.ok(w) || v.d[w].newline_before) return;
    std::string_view many;
    size_t last = w;
    if (v.word(w) && v.tight(w)) {
        std::string_view sc = scale_suffix(v.str(w));
        many = sc == "million" ? "millions" : sc == "billion" ? "billions" : sc == "thousand" ? "thousands" : "";
    } else if (v.number(w) && v.tight(w) && v.str(w) == "000") {
        many = "thousands";
        size_t s2 = v.next(w);
        if (v.word(s2) && v.tight(s2) && v.str(s2) == "s") last = s2;
    } else if (v.word(w) && !v.tight(w)) {
        std::string_view x = v.low(w);
        many = x == "millions" ? "millions" : x == "billions" ? "billions" : x == "thousands" ? "thousands" : "";
    }
    if (many.empty()) return;
    size_t n = v.next(last);
    if (v.ok(n) && v.tight(n) && (v.word(n) || v.number(n))) return;
    bool bracket = v.punct(p, '(') && v.tight(first) && v.punct(n, ')') && v.tight(n);
    bool after_in = v.word(p) && v.low(p) == "in" && !v.tight(first);
    if (!bracket && !after_in) return;
    out.add(v.d[first].start, v.d[last].end, std::string(many) + " of " + std::string(cur->pl), 70);
}

// $5, €2.50, £20k, US$5, US $5, € 12, -$7.4M, ($1.2M), $200-350, $5 to 10, $0.45/1M, [$4M, $6M], &euro;59. The
// marker runs from token k to token end; sym is its code point (0 for an entity).
void symbol_led(const V& v, rv::Out& out, size_t k, size_t end, const Cur* cur, uint32_t sym) {
    size_t first = k, p = v.prev(k);
    if (v.tight(k)) {
        if (v.word(p)) {
            if (sym != '$' || !(cur = lookup(DOLLAR_PREFIXES, v.str(p)))) return;
            if (v.tight(p) && (v.word(v.prev(p)) || v.number(v.prev(p)))) return;
            first = p;
        } else if (v.number(p)) {
            return;  // "5$": the number reads it
        }
    } else if (sym == '$' && v.word(p) && v.len(p) >= 2 && v.d[p].casing == rv::UPPER && v.one_space(p, k)) {
        const Cur* c = lookup(DOLLAR_PREFIXES, v.str(p));  // "US $5 million"
        size_t q = v.prev(p);
        if (c && v.tight(v.next(k)) && !(v.word(q) && SPACED_PREFIX_BLOCK.contains(v.low(q))) && !v.tight(p)) {
            cur = c;
            first = p;
        }
    }
    // "€ 12": one space, not after a number ("5 € 12" belongs to the 5)
    Money m;
    if (!amount_after(v, m, end, !v.number(p), cur)) {
        if (sym) unit_header(v, out, first, end, cur);
        return;
    }
    if (sym == '$' && !*m.inner && first == k && v.tight(m.a.first) && code_placeholder(v, k, m.a)) {
        out.keep(k, m.a.last, 90);  // "replace it with $1": a placeholder in a regular expression
        return;
    }
    m.first = m.plain_first = first;
    m.plain_last = m.last;
    m.has_plain = !*m.inner;
    m.cur = cur;
    m.entity = !sym;
    if (!*m.inner && !interval_around(v, m, sym)) {
        range_after(v, m, sym);
        if (!m.b.ok) range_to(v, m);
        repeated_unit(v, m);
        m.per = per_units(v, m.last, &m.per_spaced);
    }
    if (!m.b.ok && sym) {  // "$40 - $5 = $35": a subtraction, not a range
        size_t d = v.next(m.last), s2 = v.next(d);
        if (v.punct(d, '-') && !v.tight(d) && !v.tight(s2) && v.punct(s2, sym)) {
            Amount b = parse_amount(v, v.next(s2));
            if (b.ok && v.tight(b.first) && v.punct(v.next(b.last), '=')) out.tokens(d, d, "minus", 45);
        }
    }
    emit(v, out, m);
}

// USD 5 million, CHF 4.80, EUR 200k, -USD 3.4 million, USD 15–20 million, CHF 12/mo; also currency
// abbreviations before the amount: Rs. 500, kr 60, KSh 1,200 (the marker runs from token k to token end).
void code_led(const V& v, rv::Out& out, size_t k, size_t end, const Cur* cur, bool code) {
    size_t p = v.prev(k);
    if (v.tight(k) && (v.word(p) || v.number(p) || v.punct(p, '/'))) return;  // "EUR/USD 1.08"
    if (v.number(p) && !code) return;                                          // "5 kr 10"
    // glued to the number only the common codes: "EUR50", not "COP28" (a conference)
    static const rv::WordSet GLUED{"AUD", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "INR", "JPY", "NZD", "SGD",
                                   "USD"};
    if (code && v.tight(v.next(end)) && !GLUED.contains(v.str(k))) return;
    Money m;
    if (!amount_after(v, m, end, true, cur)) return;
    m.cur = cur;
    if (m.cur == &TRY && m.a.whole.size() < 3 && m.a.scale.empty()) return;  // "TRY 5 TIMES": shouted text
    m.code = code;
    m.first = m.plain_first = k;
    m.plain_last = m.last;
    m.has_plain = !*m.inner;
    if (!*m.inner) {
        range_after(v, m, 0);
        if (!m.b.ok) range_to(v, m);
        m.per = per_units(v, m.last, &m.per_spaced);
    }
    emit(v, out, m);
}

// A currency abbreviation written as a word before the amount; returns the token that ends it, or NPOS.
size_t word_before(const V& v, size_t k, const Cur*& cur) {
    size_t len = v.len(k);
    if (len != 2 && len != 3) return NPOS;
    char c0 = v.d.text[v.d[k].start];
    if (c0 != 'F' && c0 != 'K' && c0 != 'R' && c0 != 'S' && c0 != 'k' && c0 != 's' && c0 != 'z') return NPOS;
    std::string_view s = v.str(k);
    for (const WordMarker& w : WORD_BEFORE) {
        if (w.key != s) continue;
        size_t end = k, n = v.next(k);
        if (w.dot && v.punct(n, '.') && v.tight(n)) {
            end = n;
            n = v.next(n);
        }
        if (!v.number(n) || v.d[n].newline_before) return NPOS;
        if (s == "Fr" && end == k) return NPOS;  // "Fr 20" without a period: not clear enough
        cur = w.cur;
        return end;
    }
    if (s == "RM" && v.number(v.next(k)) && v.tight(v.next(k))) {  // "RM50": Malaysian ringgit
        size_t p = v.prev(k), n = v.next(k), d = v.next(n);
        if (v.word(p) && (v.low(p) == "room" || v.low(p) == "in")) return NPOS;
        if ((v.len(n) == 1 && !v.punct(d, '.')) || (is_dash(v, d) && v.tight(d))) return NPOS;  // "RM2–RM5"
        cur = &RINGGIT;
        return k;
    }
    return NPOS;
}

// A currency marker can follow the number at token i (a cheap test before parsing it).
bool currency_ahead(const V& v, size_t i) {
    bool scaled = false;
    for (int k = 0; k < 12; ++k) {
        i = v.next(i);
        if (i == NPOS) return false;
        const rv::Token& t = v.d[i];
        if (t.kind == Kind::Number) continue;
        if (t.kind == Kind::Punct) {
            if (symbol_currency(t.cp) || t.cp == 0xA2) return true;
            if (t.cp == '&') {
                const Cur* c = nullptr;
                return entity(v, i, c) != NPOS;
            }
            if (t.cp != ',' && t.cp != '.' && t.cp != '\'' && t.cp != 0x2019 && !is_dash(v, i)) return false;
            continue;
        }
        if (code_currency(v, i) || word_after(v, i) || v.str(i) == "p") return true;
        if (scaled && CURRENCY_WORDS.contains(v.str(i))) return true;
        if (!scale_suffix(v.str(i)).empty()) {
            scaled = true;
            continue;
        }
        if (scale_word(v.low(i)).empty()) return false;
    }
    return false;
}

// The currency marker right after an amount that ends at token e: ¢, a symbol, a code, a currency word or code
// ("60 kr", "50 usd"), pence ("50p"); spaced or not. Sets m.cur (and m.code); returns its token or NPOS.
size_t marker_after(const V& v, Money& m, size_t e, const Amount& a, bool scaled) {
    size_t s = v.next(e);
    if (!v.ok(s) || v.d[s].newline_before) return NPOS;
    uint32_t c = v.cp(s);
    const Cur* ent = nullptr;
    size_t ent_end = c == '&' ? entity(v, s, ent) : NPOS;
    if (ent_end != NPOS) {  // "50&cent;", "20&euro;"
        m.cur = ent;
        m.entity = true;
        return ent_end;
    } else if (c == 0xA2) {
        m.cur = &CENTS;
    } else if (const Cur* sc = symbol_currency(c)) {
        size_t after = v.next(s);
        if (v.tight(after) && (v.number(after) || (v.punct(after, '.') && v.number(v.next(after))))) return NPOS;
        m.cur = sc;
    } else if (const Cur* cc = code_currency(v, s)) {
        size_t sl = v.next(s);
        if (v.punct(sl, '/') && v.tight(sl) && code_currency(v, v.next(sl))) return NPOS;  // "1.08 EUR/USD": a rate
        m.cur = cc;
        m.code = true;
    } else if (const Cur* wc = word_after(v, s)) {
        if (glued(v, v.next(s))) return NPOS;
        m.cur = wc;
        m.code = v.d[s].casing == rv::LOWER && v.len(s) == 3 && wc != &KRONE;
    } else if (v.word(s) && v.tight(s) && v.str(s) == "p" && a.frac.empty() && a.scale.empty() &&
               a.whole.size() <= 2 && !all_zero(a.whole)) {
        size_t after = v.next(s);
        if (v.tight(after) && (v.word(after) || v.number(after))) return NPOS;
        if (v.punct(after, '.') && v.tight(after) && v.word(v.next(after)) && v.tight(v.next(after))) return NPOS;
        if (v.word(after) && (v.low(after) == "orbital" || v.low(after) == "orbitals")) return NPOS;
        m.cur = &PENCE;
    } else if (scaled && v.word(s) && !v.tight(s) && CURRENCY_WORDS.contains(v.str(s))) {
        m.cur = &DOLLAR;  // "15–20M dollars": the word is read as written
        m.word = std::string(v.str(s));
    } else {
        return NPOS;
    }
    return s;
}

// 50 EUR, 150€, 15.50 €, 50,000¥, 75¢, 50p, 15–20 €, 5 - 10 €, 5€–10€, 5 USD/unit, 60 kr, 100元, 15–20M dollars.
void number_led(const V& v, rv::Out& out, size_t i) {
    if (!currency_ahead(v, i)) return;
    size_t p = v.prev(i);
    if (v.tight(i)) {
        if (v.word(p) || v.number(p)) return;
        uint32_t c = v.cp(p);
        if (symbol_currency(c) || c == '.' || c == ',' || c == ';' || c == '\'') return;
        if (is_dash(v, p) && v.tight(p) && v.number(v.prev(p))) return;  // the second half of a range
    } else if (code_currency(v, p) || symbol_currency(v.cp(p))) {
        return;  // "USD 5", "€ 12"
    } else if (is_dash(v, p) && !v.tight(p) && v.number(v.prev(p))) {
        return;  // "5 - 10 €": the 5 reads the range
    }
    if (v.word(p) && !v.tight(i)) {
        const Cur* c = nullptr;
        if (word_before(v, p, c) == p) return;  // "kr 60"
    }
    Money m;
    Parse o;
    o.pence = true;
    m.a = parse_amount(v, i, o);
    if (!m.a.ok) return;
    m.first = i;
    m.last = m.a.last;
    size_t d = v.next(m.last);
    Amount b;
    if (is_dash(v, d) && ((v.tight(d) && v.tight(v.next(d))) || (!v.tight(d) && !v.tight(v.next(d))))) {
        b = parse_amount(v, v.next(d), o);
        if (b.ok) m.last = b.last;
    }
    bool scaled = !m.a.scale.empty() || (b.ok && !b.scale.empty());
    size_t s = marker_after(v, m, m.last, b.ok ? b : m.a, scaled);
    if (s == NPOS) return;
    if (m.cur->dot_groups && !b.ok && m.a.frac.size() == 3) {  // "45.000₫": a dot groups thousands
        Parse od = o;
        od.dots = true;
        Amount a2 = parse_amount(v, i, od);
        if (a2.ok && a2.last == m.a.last) m.a = a2;
    }
    if (b.ok) {
        m.b = b;
    } else {
        m.plain_first = m.first;
        m.plain_last = s;
        m.has_plain = m.word.empty();
        // "5€-10€", "5 € – 10 €": the marker on both halves
        size_t d2 = v.next(s);
        if (is_dash(v, d2) && v.d[s].kind == Kind::Punct) {
            size_t n = v.next(d2);
            bool joined = v.tight(d2) && v.tight(n), spaced = !v.tight(d2) && !v.tight(n);
            Amount b2 = (joined || spaced) && v.number(n) ? parse_amount(v, n, o) : Amount();
            size_t s2 = b2.ok ? v.next(b2.last) : NPOS;
            if (b2.ok && v.punct(s2, v.d[s].cp) && (v.tight(s2) == v.tight(s))) {
                m.b = b2;
                s = s2;
            }
        }
    }
    m.after = true;
    m.last = s;
    m.per = per_units(v, m.last, &m.per_spaced);
    emit(v, out, m);
}

const rv::WordSet NUMBER_WORDS{
    "billion", "eight",   "eighteen", "eighty",   "eleven",    "fifteen", "fifty",  "five",    "forty",
    "four",    "fourteen", "hundred", "million",  "nine",      "nineteen", "ninety", "one",    "seven",
    "seventeen", "seventy", "six",    "sixteen",  "sixty",     "ten",     "thirteen", "thirty", "thousand",
    "three",   "trillion", "twelve",  "twenty",   "two"};

// A symbol after an amount in words: "twenty €" -> twenty euros.
void after_words(const V& v, rv::Out& out, size_t k, const Cur* cur) {
    size_t p = v.prev(k), n = v.next(k);
    if (!v.word(p) || !NUMBER_WORDS.contains(v.low(p)) || glued(v, n) || v.d[k].newline_before) return;
    out.tokens(k, k, std::string(v.low(p) == "one" ? cur->sg : cur->pl), 50);
}

RV_RULE("money", amounts) {
    V v(doc);
    for (size_t i = 0; i < doc.size(); ++i) {
        const rv::Token& t = doc[i];
        if (t.kind == Kind::Punct) {
            if (const Cur* c = symbol_currency(t.cp)) {
                symbol_led(v, out, i, i, c, t.cp);
                after_words(v, out, i, c);
            } else if (t.cp == 0xA2) {
                after_words(v, out, i, &CENTS);
            } else if (t.cp == '&') {
                size_t end = entity(v, i, c);
                if (end != NPOS && c != &CENTS) symbol_led(v, out, i, end, c, 0);
            }
        } else if (t.kind == Kind::Number) {
            number_led(v, out, i);
        } else if (const Cur* c = code_currency(v, i)) {
            code_led(v, out, i, i, c, true);
        } else {
            const Cur* w = nullptr;
            size_t end = word_before(v, i, w);
            if (end != NPOS) code_led(v, out, i, end, w, false);
        }
    }
}

}  // namespace
