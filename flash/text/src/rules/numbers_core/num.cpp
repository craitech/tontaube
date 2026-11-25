#include "rules/numbers_core/num.h"

#include <algorithm>

namespace numbers_core {

namespace {

using rv::Kind;

// Digits of Number token j.
size_t digit_run(const Doc& d, size_t j, std::string& out, bool& inv) {
    out += d.str(j);
    for (;;) {
        bool v = false;
        size_t k = next_tok(d, j, &v);
        if (!v || !d.number(k) || d[k].space_before) break;
        out += d.str(k);
        inv = true;
        j = k;
    }
    return j;
}

const char* const SCALES[] = {"",           "thousand",    "million",    "billion",   "trillion",
                              "quadrillion", "quintillion", "sextillion", "septillion", "octillion",
                              "nonillion",   "decillion"};

const rv::WordSet COUNT_WORDS{
    "people", "persons", "users", "customers", "clients", "members", "employees", "staff", "workers", "students",
    "pupils", "children", "kids", "adults", "patients", "participants", "respondents", "visitors", "guests",
    "passengers", "residents", "citizens", "voters", "readers", "subscribers", "viewers", "listeners", "fans",
    "followers", "attendees", "delegates", "volunteers", "soldiers", "troops", "players", "applicants",
    "candidates", "teachers", "doctors", "nurses", "engineers", "developers", "researchers", "scientists",
    "farmers", "families", "households", "homes", "houses", "dwellings", "apartments", "inhabitants", "men",
    "women", "girls", "boys", "babies", "athletes", "runners", "riders", "drivers", "pilots", "accounts",
    "subscriptions", "contributors", "maintainers", "speakers", "authors", "artists", "animals", "birds",
    "species", "trees", "plants", "cows", "cattle", "sheep", "horses", "dogs", "cats", "fish", "units", "items",
    "pieces", "parts", "products", "copies", "books", "pages", "words", "lines", "characters", "chars", "tokens",
    "samples", "rows", "entries", "files", "documents", "images", "photos", "pictures", "videos", "clips",
    "frames", "songs", "tracks", "utterances", "sentences", "recordings", "votes", "signatures", "buildings",
    "rooms", "beds", "seats", "tickets", "cars", "vehicles", "trucks", "bikes", "bicycles", "ships", "boats",
    "planes", "aircraft", "flights", "trips", "transactions", "requests", "queries", "messages", "emails",
    "downloads", "installs", "sessions", "visits", "clicks", "impressions", "jobs", "tasks", "cycles",
    "iterations", "epochs", "batches", "nodes", "servers", "machines", "devices", "sensors", "cameras", "cells",
    "boxes", "crates", "pallets", "bottles", "cans", "bags", "packages", "parcels", "kits", "laptops", "computers",
    "phones", "smartphones", "monitors", "chips", "cores", "threads", "apples", "oranges", "eggs", "meals",
    "calories", "deaths", "infections", "incidents", "errors", "faults", "failures", "defects", "bugs", "stores",
    "shops", "restaurants", "schools", "classrooms", "hospitals", "clinics", "locations", "branches", "offices",
    "stations", "villages", "towns", "cities", "countries", "labels", "planks", "sachets", "artifacts",
    "artefacts", "packets", "channels", "cabinets", "sites", "objects", "readings", "measurements", "parameters",
    "dimensions", "interviews", "titles", "hours", "minutes", "seconds", "days", "weeks", "months", "years",
    "decades", "centuries", "dollars", "euros", "yen", "rupees", "coins", "laps", "gpus", "cpus", "tpus", "vms",
    "containers", "pods", "clusters", "racks", "disks", "drives", "panels", "turbines", "batteries", "modules",
    "cables", "wells", "plates", "tubes", "doses", "vials", "pills", "tablets", "subjects", "mice", "rats",
    "courses", "concerts", "screens", "stamps", "barrels", "cubes", "blocks", "bricks", "tiles", "sheets", "rolls",
    "pairs", "packs", "cartons",
};

// Units written after a number.
const rv::WordSet UNIT_WORDS{
    "pixels", "px", "rpm", "dpi", "ppi", "fps", "kb", "mb", "gb", "tb", "kbps", "mbps", "hz", "khz", "mhz", "ghz",
    "kw", "mw", "kwh", "mwh", "mah", "km", "m", "cm", "mm", "mi", "ft", "yd", "kg", "g", "mg", "lb", "lbs", "oz",
    "ml", "sq", "square", "cubic", "ppm", "qps", "rps", "btu", "hpa", "kpa", "psi", "°c", "°f", "msgs", "req",
    "ms", "sec", "secs", "min", "mins", "hrs", "miles", "kilometers", "kilometres", "meters", "metres", "feet",
    "inches", "yards", "pounds", "ounces", "tons", "tonnes", "grams", "kilograms", "liters", "litres", "gallons",
    "acres", "hectares", "mph", "kph", "knots", "kmh",
};

const rv::WordSet SCALE_WORDS{"hundred", "thousand", "million", "billion", "trillion", "quadrillion",
                              "lakh", "crore", "dozen", "thousands", "millions", "billions", "hundreds"};

}  // namespace

bool one_space(const Doc& d, size_t a, size_t b) {
    if (b <= a || b >= d.size()) return false;
    uint32_t from = d[a].end, to = d[b].start;
    if (to <= from) return false;
    size_t p = from;
    uint32_t cp = rv::decode(d.text, p);
    if (p != to) return false;
    return cp == ' ' || cp == 0xA0 || (cp >= 0x2005 && cp <= 0x200A) || cp == 0x202F;
}

Quick quick(const Doc& d, size_t i) {
    Quick q;
    q.end = i;
    q.digits = d[i].end - d[i].start;
    for (size_t j = i + 1; j < d.size(); ++j) {
        const rv::Token& t = d[j];
        if (t.kind == Kind::Number) {
            if (t.space_before) {
                if (t.end - t.start != 3 || !one_space(d, j - 1, j)) break;
                q.special = true;
            }
            q.digits += t.end - t.start;
        } else if (t.kind == Kind::Punct && !t.space_before && d.number(j + 1) && !d[j + 1].space_before &&
                   (t.cp == '.' || t.cp == ',' || t.cp == '\'' || t.cp == 0x2019 || t.cp == '_')) {
            q.special |= t.cp != '.' && t.cp != ',';
        } else {
            break;
        }
        q.end = j;
    }
    return q;
}

bool scan(const Doc& d, size_t i, Num& n) {
    n = Num{};
    if (i >= d.size()) return false;
    if (d.punct(i, '.')) {
        bool v = false;
        size_t k = next_tok(d, i, &v);
        if (!d.number(k) || d[k].space_before) return false;
        n.first = i;
        n.point = true;
        n.invisible = v;
        n.last = digit_run(d, k, n.frac, n.invisible);
        return true;
    }
    if (!d.number(i)) return false;
    n.first = i;
    size_t j = digit_run(d, i, n.whole, n.invisible);
    if (n.whole.size() <= 3) {
        const size_t j0 = j;
        const std::string whole0 = n.whole;
        const bool inv0 = n.invisible;
        for (;;) {
            bool v = false;
            size_t s = next_tok(d, j, &v);
            if (s >= d.size()) break;
            char style = 0;
            size_t g = 0;
            if (!d[s].space_before && d[s].kind == Kind::Punct) {
                uint32_t cp = d[s].cp;
                style = cp == ',' ? ',' : (cp == '\'' || cp == 0x2019) ? '\'' : cp == '_' ? '_' : 0;
                if (!style) break;
                g = next_tok(d, s, &v);
                if (!d.number(g) || d[g].space_before) break;
            } else if (!v && d.number(s) && d[s].space_before && one_space(d, j, s)) {
                style = ' ';
                g = s;
            } else {
                break;
            }
            if (n.sep && n.sep != style) break;
            std::string grp;
            bool gv = false;
            size_t e = digit_run(d, g, grp, gv);
            if (grp.size() != 3) break;
            n.whole += grp;
            n.sep = style;
            n.invisible |= v || gv;
            j = e;
        }
        if (n.sep == ' ') {
            // "+49 30 123 4567", "1 2 345", "5 100-meter races": something around the chain makes it something
            // else than one number
            size_t s = next_tok(d, j), p = prev_tok(d, i);
            bool after = (d.number(s) && d[s].space_before) ||
                         (s < d.size() && !d[s].space_before && d[s].kind == Kind::Punct && d[s].cp == '-');
            bool before = p != Doc::npos && d.number(p) && d[p + 1].space_before && one_space(d, p, i);
            if (after || before) {
                n.whole = whole0;
                n.sep = 0;
                n.invisible = inv0;
                j = j0;
            }
        }
    }
    bool v = false;
    size_t s = next_tok(d, j, &v);
    if (d.punct(s, '.') && !d[s].space_before) {
        bool v2 = false;
        size_t k = next_tok(d, s, &v2);
        if (d.number(k) && !d[k].space_before) {
            n.point = true;
            n.invisible |= v || v2;
            j = digit_run(d, k, n.frac, n.invisible);
        }
    }
    n.last = j;
    return true;
}

namespace {
bool sign_cp(uint32_t c) { return c == '-' || c == 0x2212 || c == 0x2013 || c == '+' || c == 0xB1; }
}  // namespace

bool free_before(const Doc& d, size_t i) {
    if (i == 0 || i >= d.size() || d[i].space_before) return true;
    size_t p = prev_tok(d, i);
    if (p == Doc::npos || d[p + 1].space_before) return true;
    const rv::Token& t = d[p];
    if (t.kind != Kind::Punct) return false;
    if (t.cp == '.' || t.cp == ',' || t.cp == '_') return false;  // "v2.0", "var_1"
    if (sign_cp(t.cp)) {
        if (p == 0 || d[p].space_before) return true;
        size_t q = prev_tok(d, p);
        if (q == Doc::npos || d[q + 1].space_before) return true;
        return d[q].kind == Kind::Punct && d[q].cp != '.' && d[q].cp != ',' && d[q].cp != ')' && d[q].cp != ']' &&
               d[q].cp != '%' && !sign_cp(d[q].cp);
    }
    return true;
}

bool after_range_dash(const Doc& d, size_t i) {
    if (i < 2 || i >= d.size() || d[i].space_before) return false;
    size_t t = prev_tok(d, i);
    if (t == Doc::npos || d[t + 1].space_before || d[t].kind != Kind::Punct) return false;
    uint32_t c = d[t].cp;
    if (c != '-' && !(c >= 0x2010 && c <= 0x2014)) return false;
    size_t p = prev_tok(d, t);
    if (p == Doc::npos || d[p + 1].space_before) return false;
    if (d.number(p) || (d[p].kind == Kind::Punct && (d[p].cp == '%' || d[p].cp == 0x2030))) return true;
    if (!d.word(p) || p == 0 || !d.number(p - 1) || d[p].space_before) return false;
    std::string_view w = d.low(p);
    return w == "st" || w == "nd" || w == "rd" || w == "th" || w == "s";
}

bool free_after(const Doc& d, size_t k) {
    size_t j = next_tok(d, k);
    if (j >= d.size() || d[j].space_before) return true;
    const rv::Token& t = d[j];
    if (t.kind != Kind::Punct) return false;
    if (t.cp == '.' || t.cp == ',') {
        size_t m = next_tok(d, j);
        if (d.number(m) && !d[m].space_before) return false;
    }
    return true;
}

std::string whole_words(std::string_view digits) {
    size_t nz = digits.find_first_not_of('0');
    if (nz == std::string_view::npos) return digits.empty() ? "" : "zero";
    std::string_view v = digits.substr(nz);
    if (v.size() <= 24) return rv::cardinal_digits(v);
    if (v.size() > 36) return rv::digits(v);
    std::string out;
    size_t groups = (v.size() + 2) / 3, first = v.size() - (groups - 1) * 3, pos = 0;
    for (size_t g = 0; g < groups; ++g) {
        size_t len = g == 0 ? first : 3;
        std::string_view part = v.substr(pos, len);
        pos += len;
        if (part.find_first_not_of('0') == std::string_view::npos) continue;
        if (!out.empty()) out += ' ';
        out += rv::cardinal_digits(part);
        size_t scale = groups - 1 - g;
        if (scale) {
            out += ' ';
            out += SCALES[scale];
        }
    }
    return out;
}

std::string num_words(const Num& n) {
    std::string w;
    if (!n.whole.empty()) {
        if (n.whole.size() > 1 && n.whole[0] == '0') return "";
        w = whole_words(n.whole);
    } else if (!n.point) {
        return "";
    }
    if (n.point) {
        if (n.frac.empty()) return "";
        w += w.empty() ? "point " : " point ";
        w += rv::digits(n.frac);
    }
    return w;
}

std::string plural_of(std::string w) {
    if (!w.empty() && w.back() == 'y') {
        w.pop_back();
        return w + "ies";
    }
    return w + "s";
}

uint32_t value4(const std::string& digits) {
    if (digits.empty() || digits.size() > 9) return 0;
    uint32_t v = 0;
    for (char c : digits) v = v * 10 + static_cast<uint32_t>(c - '0');
    return v;
}

uint32_t entity_at(const Doc& d, size_t i, size_t& last) {
    static const std::pair<std::string_view, uint32_t> NAMED[] = {
        {"ndash", 0x2013}, {"mdash", 0x2014}, {"minus", 0x2212}, {"plusmn", 0xB1},
        {"percnt", '%'},   {"asymp", 0x2248}, {"sim", 0x223C},   {"permil", 0x2030}};
    if (!d.punct(i, '&') || i + 2 >= d.size()) return 0;
    uint32_t cp = 0;
    size_t j = i + 1;
    if (d.word(j) && !d[j].space_before) {
        for (const auto& e : NAMED)
            if (d.low(j) == e.first) cp = e.second;
        if (!cp) return 0;
    } else if (d.punct(j, '#') && !d[j + 1].space_before) {
        ++j;
        if (d.number(j)) {
            for (char c : d.str(j)) cp = cp * 10 + static_cast<uint32_t>(c - '0');
        } else if (d.is(j, "x") && d.number(j + 1) && !d[j + 1].space_before) {
            ++j;
            for (char c : d.str(j)) cp = cp * 16 + static_cast<uint32_t>(c - '0');
        } else {
            return 0;
        }
    } else {
        return 0;
    }
    if (!d.punct(j + 1, ';') || d[j + 1].space_before) return 0;
    last = j + 1;
    return cp;
}

bool count_word(std::string_view w) { return COUNT_WORDS.contains(w) || UNIT_WORDS.contains(w); }
bool unit_word(std::string_view w) { return UNIT_WORDS.contains(w); }
bool scale_word(std::string_view w) { return SCALE_WORDS.contains(w); }

}  // namespace numbers_core
