// Numbers read digit by digit: phone numbers in every format (with a comma where the written number has a
// hyphen, a dot or a closing parenthesis, nothing at a space; 800 as "eight hundred"), extensions, IDs after a
// cue word (serial, account, order #, PIN, ZIP, PMID, ISBN), codes with a letter prefix after a cue
// (model XZ-987654321, license plate 5XF903P), the last digits of an account (card ending 7730), dialed short
// numbers (call 911, text STOP to 55555), ZIP codes after a state, and numbers no cardinal can read (a leading
// zero, more than 24 digits).
#include "rules/numbers_context/common.h"

namespace {

using namespace nctx;

const rv::WordSet PHONE_CUES{"call", "called", "calling", "calls", "cell", "contact", "dial", "fax", "helpline",
                             "hotline", "line", "mobile", "phone", "phones", "reach", "reached", "ring", "sms",
                             "tel", "telephone", "text", "texts", "whatsapp", "number", "desk", "office", "support",
                             "questions", "info", "inquiries", "enquiries", "reservations", "showings", "orders"};

const rv::WordSet PLATE_CUES{"plate", "plates"};
const rv::WordSet PLATE_BEFORE{"license", "licence", "registration", "car", "vehicle", "number", "his", "her", "its",
                              "their", "the", "a", "car's", "vehicle's", "truck's", "van's", "sedan's", "suspect's"};
const rv::WordSet CODE_CUES{"model", "serial", "part", "product", "identifier", "id", "sku", "code", "plate",
                            "plates", "item", "reference", "ref", "unit", "device", "number"};

const rv::WordSet STATES{"AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL",
                         "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
                         "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
                         "VA", "VT", "WA", "WI", "WV", "WY", "PR"};

// ---------------------------------------------------------------------------------------------------------
// Phone numbers

struct Group {
    size_t first, last;  // tokens of the digits (with the parentheses for "(312)")
    std::string_view digits;
    char sep;            // separator after the group: '-', '.', ' ', ')' or 0 at the end
};

bool sep_dash(const Doc& d, size_t j) { return d.punct(j, '-') || d.punct(j, 0x2010) || d.punct(j, 0x2011); }
// A dash that writers also put into phone numbers: en dash, em dash, figure dash.
bool wide_dash(const Doc& d, size_t j) { return d.punct(j, 0x2012) || d.punct(j, 0x2013) || d.punct(j, 0x2014); }

// Reads digit groups from token i: "(312) 660-4198", "1-800-555-1212", "+44 20 7946 0391", "312.660.4198".
size_t read_groups(const Doc& d, size_t i, Group* g, size_t cap, size_t& n) {
    n = 0;
    size_t pos = i;
    while (n < cap) {
        Group x{};
        bool paren = d.punct(pos, '(') && d.number(pos + 1) && d.joined(pos + 1) && d.punct(pos + 2, ')') &&
                     d.joined(pos + 2);
        if (paren) {
            x = {pos, pos + 2, d.str(pos + 1), ')'};
            pos += 3;
        } else if (d.number(pos)) {
            x = {pos, pos, d.str(pos), 0};
            pos += 1;
        } else {
            break;
        }
        g[n++] = x;
        Group& cur = g[n - 1];
        // separator to the next group
        bool next_starts = pos < d.size() && (d.number(pos) || d.punct(pos, '('));
        bool after_dash = (sep_dash(d, pos) || wide_dash(d, pos) || d.punct(pos, '.')) && d.joined(pos) &&
                          pos + 1 < d.size() && d.joined(pos + 1) && d.number(pos + 1);
        if (paren) {
            if (after_dash) {
                ++pos;  // "(312)-660"
            } else if (!(next_starts && (d.joined(pos) || d[pos].space_before))) {
                cur.sep = 0;
                break;
            }
            continue;
        }
        if (after_dash) {
            cur.sep = d.punct(pos, '.') ? '.' : wide_dash(d, pos) ? '~' : '-';
            ++pos;
        } else if (next_starts && d[pos].space_before && !d[pos].newline_before) {
            cur.sep = ' ';
        } else {
            cur.sep = 0;
            break;
        }
    }
    if (n > 0 && g[n - 1].sep != 0 && g[n - 1].sep != ')') g[n - 1].sep = 0;
    return pos;
}

RV_RULE("numbers_context", phone_numbers) {
    Group g[8];
    for (size_t i = 0; i < doc.size(); ++i) {
        bool plus = doc.punct(i, '+') && doc.number(i + 1) && doc.joined(i + 1);
        bool start = plus || doc.number(i) || (doc.punct(i, '(') && doc.number(i + 1));
        if (!start) continue;
        if (!plus && !free_left(doc, i) && !doc.punct(i, '(')) continue;
        if (plus && doc.joined(i) && doc[i - 1].kind != Kind::Punct) continue;
        size_t n = 0;
        read_groups(doc, plus ? i + 1 : i, g, 8, n);
        if (n < 2) continue;
        // trim trailing groups that do not belong (a spaced number after the phone number)
        size_t len[8];
        size_t total = 0;
        for (size_t k = 0; k < n; ++k) total += (len[k] = g[k].digits.size());
        bool has_dash_or_dot = false, has_paren = false, spaces_only = true;
        for (size_t k = 0; k + 1 < n; ++k) {
            if (g[k].sep == '-' || g[k].sep == '.' || g[k].sep == '~') has_dash_or_dot = true, spaces_only = false;
            if (g[k].sep == ')') has_paren = true, spaces_only = false;
        }
        bool cue = total >= 7 && any_prev(doc, i, 3, PHONE_CUES);
        size_t use = 0;  // groups that form the number
        int score = 0;
        auto shape = [&](size_t off, std::initializer_list<size_t> want) {
            if (off + want.size() > n) return false;
            size_t k = off;
            for (size_t w : want)
                if (len[k++] != w) return false;
            return true;
        };
        if (plus) {
            // +CC then groups: 8..15 digits, or a NANP shape
            size_t t = 0, k = 0;
            for (; k < n && t < 15; ++k) t += len[k];
            if (t >= 8 && t <= 15 && len[0] <= 3) use = k, score = 70;
        } else if (shape(0, {1, 3, 3, 4}) && g[0].digits == "1" && !spaces_only) {
            use = 4, score = cue ? 72 : 64;
        } else if (shape(0, {3, 3, 4}) && (has_dash_or_dot || has_paren)) {
            use = 3, score = cue ? 72 : 62;
        } else if (shape(0, {3, 3, 4}) && spaces_only && cue) {
            use = 3, score = 66;
        } else if ((shape(0, {3, 4, 4}) || shape(0, {4, 3, 4}) || shape(0, {5, 6})) && g[0].digits[0] == '0' &&
                   has_dash_or_dot && !has_paren) {
            use = shape(0, {5, 6}) ? 2 : 3, score = cue ? 70 : 60;  // UK style: 020-7946-0911
        } else if (shape(0, {3, 4}) && (g[0].sep == '-' || g[0].sep == '.') && g[0].digits[0] >= '2' &&
                   (cue || g[0].digits == "555")) {
            use = 2, score = cue ? 70 : 50;  // "555-1234"; "pages 200-3000" is a range
        } else if (!plus && g[0].digits[0] == '0' && len[0] >= 2 && len[0] <= 5 && !has_dash_or_dot &&
                   (g[0].sep == ' ' || g[0].sep == ')')) {
            // a national number with its trunk zero: "020 3816 4492", "07700 900123", "(020) 7946 0958"
            size_t t = len[0], k = 1;
            for (; k < n && t < 11 && len[k] >= 3 && len[k] <= 6 && (k == 1 || g[k - 1].sep == ' '); ++k) t += len[k];
            if (k >= 2 && t >= 9 && t <= 11) use = k, score = cue ? 70 : 60;
        }
        if (!use) continue;
        // the number must end cleanly
        size_t last_tok = g[use - 1].last;
        if (use < n && g[use - 1].sep != ' ') continue;
        if (!free_right(doc, last_tok) && !(doc.joined(last_tok + 1) && doc.word(last_tok + 1) &&
                                            (doc.low(last_tok + 1) == "x" || doc.low(last_tok + 1) == "ext")))
            continue;
        if (total > 20) continue;
        // reading
        std::string words = plus ? "plus " : "";
        size_t area = use >= 3 ? use - 3 : Doc::npos;  // the area code of a NANP number
        bool nanp = use >= 3 && len[use - 1] == 4 && len[use - 2] == 3 && len[use - 3] == 3;
        for (size_t k = 0; k < use; ++k) {
            bool eight_hundred = nanp && k == area && g[k].digits == "800";
            words += eight_hundred ? "eight hundred" : digit_words(g[k].digits);
            if (k + 1 < use) {
                char sep = g[k].sep;
                bool one_800 = nanp && k + 1 == area && g[k].digits == "1" && g[k + 1].digits == "800" && !plus;
                words += (sep == ' ' || one_800) ? " " : ", ";
            }
        }
        size_t first_tok = plus ? i : g[0].first;
        out.tokens(first_tok, last_tok, words, score);
        i = last_tok;
    }
}

// "ext. 52", "extension 204", "x52" after a phone number: extension fifty-two.
RV_RULE("numbers_context", extensions) {
    for (size_t i = 0; i + 1 < doc.size(); ++i) {
        if (!doc.word(i)) continue;
        std::string_view w = doc.low(i);
        size_t j = i + 1;
        bool joined_x = false;
        if (w == "ext" || w == "extension") {
            if (doc.punct(j, '.') && doc.joined(j)) ++j;
            if (doc.punct(j, '#')) ++j;
        } else if (w == "x" && doc.number(j) && doc[i].space_before) {
            // "x52" or "x 52" after a phone number: "312-660-4198 x52"
            if (i < 3 || !doc.number(i - 1) || doc.str(i - 1).size() != 4 ||
                !(sep_dash(doc, i - 2) || doc.punct(i - 2, '.')) || !doc.joined(i - 1))
                continue;
            joined_x = true;
        } else {
            continue;
        }
        if (!doc.number(j) || !free_right(doc, j) || doc.str(j).size() > 6) continue;
        if (!joined_x && doc.joined(i) && doc[i - 1].kind != Kind::Punct) continue;
        std::string_view digits = doc.str(j);
        std::string n = digits[0] == '0' ? digit_words(digits) : rv::cardinal_digits(digits);
        bool after_phone = (i >= 1 && doc.number(i - 1)) || (i >= 2 && doc.punct(i - 1, ',') && doc.number(i - 2)) ||
                           (i >= 2 && doc.punct(i - 1, '(') && doc.number(i - 2));
        std::string word = doc[i].casing & (rv::TITLE | rv::UPPER) ? "Extension" : "extension";
        out.tokens(i, j, word + " " + n, after_phone || joined_x ? 72 : 66);
        i = j;
    }
}

// ---------------------------------------------------------------------------------------------------------
// IDs after a cue word

RV_RULE("numbers_context", id_digits) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (!doc.number(i)) continue;
        if (!free_left(doc, i)) continue;  // "#12" is read by the rule for '#'

        // the number, possibly in hyphenated groups (ISBN 978-0-306-40615-7) or in spaced groups of four
        // (card 4111 1111 1111 1111)
        size_t last = i;
        size_t ndig = doc.str(i).size();
        while (sep_dash(doc, last + 1) && doc.joined(last + 1) && doc.number(last + 2) && doc.joined(last + 2)) {
            ndig += doc.str(last + 2).size();
            last += 2;
        }
        bool spaced_groups = false;
        if (last == i && ndig == 4) {
            size_t k = i;
            while (doc.number(k + 1) && doc[k + 1].space_before && !doc[k + 1].newline_before &&
                   doc.str(k + 1).size() <= 4 && free_right(doc, k))
                ++k;
            if (k >= i + 2) spaced_groups = true, last = k;
            for (size_t m = i + 1; m <= last && spaced_groups; ++m) ndig += doc.str(m).size();
        }
        // "Ref. 123-A": a letter suffix after the digits
        size_t suffix = Doc::npos;
        if (sep_dash(doc, last + 1) && doc.joined(last + 1) && doc.word(last + 2) && doc.joined(last + 2) &&
            doc[last + 2].casing == rv::UPPER && doc.str(last + 2).size() <= 3 && free_right(doc, last + 2))
            suffix = last + 2;
        bool iso_date = last == i + 4 && doc.str(i).size() == 4 && doc.str(i + 2).size() <= 2 &&
                        doc.str(i + 4).size() <= 2 && !spaced_groups;
        if ((!free_right(doc, last) && suffix == Doc::npos) || iso_date) {
            i = last;
            continue;
        }
        int min_digits = 0;
        size_t cue = ndig >= 3 ? id_cue(doc, i, min_digits) : Doc::npos;
        // "License Plate 918834"
        if (cue == Doc::npos && ndig >= 3) {
            size_t p = prev_w(doc, i);
            if (p != Doc::npos && (doc.is(p, "plate") || doc.is(p, "plates")) &&
                PLATE_BEFORE.contains(low_or_empty(doc, prev_w(doc, p))) && !doc.is(prev_w(doc, p), "the") &&
                !doc.is(prev_w(doc, p), "a"))
                cue = p, min_digits = 3;
        }
        if (cue == Doc::npos || static_cast<int>(ndig) < min_digits || (spaced_groups && min_digits >= 5)) {
            i = last;
            continue;
        }
        std::string_view first = doc.str(i);
        bool year = last == i && ndig == 4 && (first.substr(0, 2) == "19" || first.substr(0, 2) == "20");
        if (year && prev_w(doc, i) == cue && !ID_ALWAYS.contains(doc.low(cue))) {
            i = last;
            continue;
        }
        // "code 404" (an HTTP status) is read in pairs, not here
        std::string_view cw = doc.low(cue);
        if (cw == "code" && ndig == 3) {
            i = last;
            continue;
        }
        std::string words = digit_words(doc.str(i));
        if (spaced_groups)
            for (size_t k = i + 1; k <= last; ++k) words += " " + digit_words(doc.str(k));
        else
            for (size_t k = i + 2; k <= last; k += 2) words += ", " + digit_words(doc.str(k));
        if (suffix != Doc::npos) {
            words += " " + std::string(doc.str(suffix));
            last = suffix;
        }
        out.tokens(i, last, words, 70);
        i = last;
    }
}

// "card ending 7730", "Savings ending in 0275", "the card that ends in 4821": the last digits of an account.
const rv::WordSet ACCOUNT_WORDS{"account", "acct", "amex", "brokerage", "card", "cards", "checking", "credit",
                                "debit", "discover", "ira", "loan", "mastercard", "mortgage", "number", "phone",
                                "savings", "visa", "wallet"};

void account_ending(const Doc& doc, rv::Out& out, size_t i) {
    if (i == 0 || i + 1 >= doc.size() || !(doc.is(i, "ending") || doc.is(i, "ends"))) return;
    size_t k = i + 1;
    if (doc.is(k, "in")) ++k;
    if (doc.punct(k, ':')) ++k;
    if (!doc.number(k) || !doc[k].space_before || !free_right(doc, k)) return;
    std::string_view s = doc.str(k);
    if (s.size() < 2 || s.size() > 6) return;
    bool account = false;
    for (int m = 1; m <= 4 && !account; ++m) {
        size_t p = prev_w(doc, i, m);
        if (p == Doc::npos) break;
        account = ACCOUNT_WORDS.contains(doc.low(p));
    }
    bool year = s.size() == 4 && (s.substr(0, 2) == "19" || s.substr(0, 2) == "20");
    if (!account && (year || doc.is(i, "ends") || s.size() < 4)) return;
    if (!account && !(doc[i - 1].kind == Kind::Word && (doc[i - 1].casing & rv::TITLE))) return;
    out.tokens(k, k, digit_words(s), 70);
}

// Short numbers that are dialed: "call 911", "dial 999", "9-1-1"; text short codes: "text STOP to 55555".
const rv::WordSet DIAL_WORDS{"call", "called", "calling", "calls", "dial", "dialed", "dialing", "dialled", "dialling",
                             "phone", "ring", "contact"};
const rv::WordSet EMERGENCY{"911", "999", "112", "211", "311", "411", "511", "611", "711", "811", "988", "101",
                            "111"};

// Returns the last token read.
size_t dialed_number(const Doc& doc, rv::Out& out, size_t i) {
    if (!free_left(doc, i)) return i;
    // "9-1-1"
    if (doc.str(i) == "9" && sep_dash(doc, i + 1) && doc.joined(i + 1) && doc.is(i + 2, "1") && doc.joined(i + 2) &&
        sep_dash(doc, i + 3) && doc.joined(i + 3) && doc.is(i + 4, "1") && doc.joined(i + 4) &&
        free_right(doc, i + 4)) {
        out.tokens(i, i + 4, "nine one one", 75);
        return i + 4;
    }
    std::string_view s = doc.str(i);
    bool emergency = s.size() == 3 && EMERGENCY.contains(s);
    bool short_code = s.size() >= 4 && s.size() <= 6 && doc.is(i - 1, "to");
    if (!emergency && !short_code) return i;
    if (!free_right(doc, i) || (doc.joined(i + 1) && doc.punct(i + 1, '.') && doc.number(i + 2))) return i;
    if (emergency) {
        size_t p = prev_w(doc, i), p2 = prev_w(doc, i, 2);
        if (DIAL_WORDS.contains(low_or_empty(doc, p)) || DIAL_WORDS.contains(low_or_empty(doc, p2)))
            out.tokens(i, i, digit_words(s), 72);
        return i;
    }
    // "text JOIN to 55555": the keyword between "text" and "to" is written in capitals or quoted
    size_t kw = prev_w(doc, i, 2);
    std::string_view v = low_or_empty(doc, prev_w(doc, i, 3));
    bool keyword = kw != Doc::npos && doc.word(kw) &&
                   (doc[kw].casing == rv::UPPER ||
                    (kw >= 1 && doc[kw - 1].kind == Kind::Punct && doc[kw - 1].cp != ','));
    if (keyword && (v == "text" || v == "txt" || v == "sms" || v == "texting")) out.tokens(i, i, digit_words(s), 72);
    return i;
}

// A code with a letter prefix after a cue: "model number XZ987654321", "Model XZ-987654321", "part number
// XY456789", "license plate 5XF903P": letters as written, digits one by one.
RV_RULE("numbers_context", coded_ids) {
    for (size_t i = 1; i < doc.size(); ++i) {
        // a code starts after a space with a letter group or digits, and has the other joined to it
        if ((!doc.word(i) && !doc.number(i)) || doc.joined(i) || !doc.joined(i + 1)) continue;
        // the joined run of letters and digits (one hyphen allowed between a letter group and digits)
        size_t j = i, letters = 0, digit_tokens = 0, longest = 0;
        bool ok = true;
        for (; j < doc.size(); ++j) {
            if (j > i && !doc.joined(j)) break;
            if (doc.word(j)) {
                if (!(doc[j].casing & rv::UPPER) || doc.str(j).size() > 4) {
                    ok = false;
                    break;
                }
                ++letters;
            } else if (doc.number(j)) {
                ++digit_tokens;
                longest = std::max(longest, doc.str(j).size());
            } else if (!(sep_dash(doc, j) && j > i && doc.word(j - 1) && doc.number(j + 1) && doc.joined(j + 1))) {
                break;
            }
        }
        size_t last = j > i ? j - 1 : i;
        // "Serial Number A2B4C6D8": single letters and short digit groups taking turns, after a code word
        bool alternating = ok && letters >= 2 && digit_tokens >= 2 && letters + digit_tokens <= 12 && longest <= 2 &&
                           free_right(doc, last);
        for (size_t k = i; k <= last && alternating; ++k)
            alternating = (doc.word(k) && doc.str(k).size() == 1) || doc.number(k);
        if (alternating) {
            bool code = false;
            for (int k = 1; k <= 3 && !code; ++k) {
                size_t px = prev_w(doc, i, k);
                if (px == Doc::npos) break;
                std::string_view w = doc.low(px);
                code = (ID_ALWAYS.contains(w) || ID_FOUR.contains(w) || w == "number") && w != "model";
            }
            if (code) {
                std::string words;
                for (size_t k = i; k <= last; ++k) {
                    if (!words.empty()) words += ' ';
                    words += doc.word(k) ? std::string(doc.str(k)) : digit_words(doc.str(k));
                }
                out.tokens(i, last, words, 70);
            }
            i = last;
            continue;
        }
        if (!ok || letters == 0 || digit_tokens == 0 || letters + digit_tokens > 6 || longest < 2 ||
            !free_right(doc, last)) {
            i = last;
            continue;
        }
        // a cue among the six words before ("the license plate, which read 7GZ891")
        bool plate = false, code = false;
        for (int k = 1; k <= 6 && !plate && !code; ++k) {
            size_t px = prev_w(doc, i, k);
            if (px == Doc::npos) break;
            // "license plate", "the plate", not "assay plate"
            plate = PLATE_CUES.contains(doc.low(px)) && PLATE_BEFORE.contains(low_or_empty(doc, prev_w(doc, px)));
            code = CODE_CUES.contains(doc.low(px)) && !PLATE_CUES.contains(doc.low(px));
        }
        if ((!plate && !code) || (!plate && (longest < 5 || !doc.word(i) || letters + digit_tokens > 3))) {
            // "Xeon 13900K" is a model name, "1Z9R3W2A0380123456" a tracking string kept as written
            i = last;
            continue;
        }
        std::string words;
        for (size_t k = i; k <= last; ++k) {
            if (doc[k].kind == Kind::Punct) {
                if (plate) words += ',';  // "5GH-421JK": five GH, four two one JK
                continue;
            }
            if (!words.empty()) words += ' ';
            words += doc.word(k) ? std::string(doc.str(k)) : digit_words(doc.str(k));
        }
        out.tokens(i, last, words, 70);
        i = last;
    }
}

// "Springfield, IL 62704": a ZIP code after a state.
void zip_code(const Doc& doc, rv::Out& out, size_t i) {
    if (i < 2 || doc.str(i).size() != 5 || doc.joined(i) || !doc.word(i - 1)) return;
    if (!STATES.contains(doc.str(i - 1)) || !(doc.punct(i - 2, ',') || doc.word(i - 2))) return;
    size_t last = i;
    std::string words = digit_words(doc.str(i));
    if (sep_dash(doc, i + 1) && doc.joined(i + 1) && doc.number(i + 2) && doc.str(i + 2).size() == 4 &&
        doc.joined(i + 2)) {
        words += ", " + digit_words(doc.str(i + 2));
        last = i + 2;
    }
    if (!free_right(doc, last)) return;
    out.tokens(i, last, words, 66);
}

// A number no cardinal reads: "007", "0042", more than 24 digits.
void unreadable_digits(const Doc& doc, rv::Out& out, size_t i) {
    std::string_view s = doc.str(i);
    if (!((s.size() >= 2 && s[0] == '0') || s.size() > 24)) return;
    // "2023-CV-01234": the part after a letter group of a code
    bool code_part = i >= 2 && doc.joined(i) && sep_dash(doc, i - 1) && doc.joined(i - 1) && doc.word(i - 2) &&
                     doc[i - 2].casing == rv::UPPER;
    if ((!free_left(doc, i) && !code_part) || !free_right(doc, i)) return;
    if (s.find_first_not_of('0') == std::string_view::npos) return;
    if (s.size() == 4 && s[2] <= '5') return;  // 0800: a clock time (datetime)
    if ((i >= 1 && doc.number(i - 1)) || doc.number(i + 1)) return;  // "200 000", "1 050"
    if (doc.joined(i + 1) && (doc.punct(i + 1, ':') || doc.punct(i + 1, '/') || doc.punct(i + 1, '.')) &&
        (!code_part || doc.number(i + 2)))
        return;
    out.tokens(i, i, digit_words(s), 12);
}

// The last digits of an account, dialed numbers, ZIP codes after a state, unreadable digit strings: one pass.
RV_RULE("numbers_context", digit_numbers) {
    for (size_t i = 0; i < doc.size(); ++i) {
        if (doc.number(i)) {
            zip_code(doc, out, i);
            unreadable_digits(doc, out, i);
            i = dialed_number(doc, out, i);
        } else if (doc.word(i) && (doc.text[doc[i].start] | 0x20) == 'e') {
            account_ending(doc, out, i);
        }
    }
}

}  // namespace
