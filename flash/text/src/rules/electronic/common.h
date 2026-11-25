#pragma once
// Shared helpers of the module "electronic": byte classes, boundaries, word tables and the readings of the
// pieces (domain labels, extensions, identifiers, ports, IP parts, hex). Most rules find the mark their construct
// needs ('.', '/', '@', '_', '=') with each_mark, parse the bytes of the run of joined tokens around it, and
// propose one candidate per construct. Rules: web.cpp, files.cpp, code.cpp.
#include <string>
#include <string_view>

#include "core/rule.h"

namespace rv::el {

inline bool lo(char c) { return c >= 'a' && c <= 'z'; }
inline bool up(char c) { return c >= 'A' && c <= 'Z'; }
inline bool al(char c) { return lo(c) || up(c); }
inline bool dg(char c) { return c >= '0' && c <= '9'; }
inline bool an(char c) { return al(c) || dg(c); }
inline bool idc(char c) { return an(c) || c == '_'; }
inline char lower_c(char c) { return up(c) ? static_cast<char>(c + 32) : c; }
inline char upper_c(char c) { return lo(c) ? static_cast<char>(c - 32) : c; }
std::string lower_s(std::string_view s);
std::string upper_s(std::string_view s);
bool all_digits(std::string_view s);
bool all_lower(std::string_view s);   // ASCII lower-case letters only, non-empty
bool all_upper(std::string_view s);   // ASCII upper-case letters only, non-empty

// Byte b starts a construct: start of text, whitespace before it, or an opening bracket, quote, backtick, '=',
// ',', ';', '|', '*' or a dash before it.
bool left_edge(const Doc& doc, uint32_t b);
// Byte e ends a construct: end of text or anything but a letter, digit or '_' at e.
bool right_edge(const Doc& doc, uint32_t e);
// The byte before b is whitespace (or b is 0).
bool space_before(const Doc& doc, uint32_t b);
// [s, e) without trailing sentence punctuation (. , ; : ! ? and unbalanced closing brackets and quotes).
uint32_t trim_tail(std::string_view text, uint32_t s, uint32_t e);
// End of the run of bytes from b that may belong to a URL, path or file name (ASCII, no whitespace, no quotes).
uint32_t url_run(std::string_view text, uint32_t b);
// Moves a scan index i so that the next step starts at the first token at or after byte e (never backwards).
inline void skip_to(const Doc& doc, size_t& i, uint32_t e) {
    while (i + 1 < doc.size() && doc[i + 1].start < e) ++i;
}
// Calls f(i) for every token i that starts with the ASCII byte c, in text order. The bytes are found with
// find (memchr), so a rule that needs a rare mark ('@', '#', '`', '=') costs next to nothing on other text.
template <class F>
void each_mark(const Doc& doc, char c, F&& f) {
    std::string_view t = doc.text;
    size_t i = 0;
    for (size_t p = t.find(c); p != std::string_view::npos; p = t.find(c, p + 1)) {
        // a few steps forward, then a binary search: rare marks cost nothing on long texts
        for (int k = 0; k < 4 && i < doc.size() && doc[i].start < p; ++k) ++i;
        if (i < doc.size() && doc[i].start < p) {
            size_t lo = i, hi = doc.size();
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (doc[mid].start < p) lo = mid + 1;
                else hi = mid;
            }
            i = lo;
        }
        if (i < doc.size() && doc[i].start == p) f(i);
    }
}
// Token i is followed directly (no whitespace) by another token.
inline bool glued(const Doc& doc, size_t i) { return i + 1 < doc.size() && !doc[i + 1].space_before; }
// The token whose span starts at byte b, or doc.size().
size_t tok_at(const Doc& doc, uint32_t b);

// Tables.
bool is_tld(std::string_view label);          // a label that can end a domain in running text
std::string tld_word(std::string_view label);  // io -> IO, edu -> EDU, co -> co, com -> com
std::string label_word(std::string_view label);
// Letters and digits of a name, letters as written: web01 -> web zero one, s3 -> s three, node-7 -> node seven.
std::string mixed_words(std::string_view s);
std::string ext_word(std::string_view ext);    // known spoken form, else spelled when short, else capitalized
bool is_ext(std::string_view ext);             // a known file extension (any case)

// Readings.
enum class Words { Code, Upper, Handle, Placeholder };
// Words of an identifier: snake_case, kebab-case, camelCase, UPPER_CASE, digits read as numbers; "id" -> "ID".
std::string ident_words(std::string_view s, Words mode = Words::Code);
// A name with its capitals kept (handles, hashtags, Reddit and repository names): split at '_' and '-', CamelCase
// split, digits as numbers; up2: two lower-case letters that are not a word are an initialism (ai -> AI).
std::string name_words(std::string_view s, bool up2);
// One part of a dotted name in code: os -> OS, read_csv -> read csv, DataFrame -> Dataframe.
std::string code_part(std::string_view s);
// One path or URL path segment: v1 -> V one, sim_runs -> sim runs, report.md -> report dot MD.
// In an API route (api) a lower-case spelled extension or TTS is an initialism: /v5/exports/csv -> CSV.
// hyphen: a hyphen between two words stays (file system paths: bank-two).
std::string path_part(std::string_view s, bool api = false, bool hyphen = false);
// Letters as written, digit runs as numbers (mixed_words): vault2 -> vault two, q3 -> q three.
void alnum_words(std::string& out, std::string_view w);
// A file name with its extensions: report_v2.pdf -> report V two dot PDF.
std::string file_words(std::string_view s);
std::string port_words(std::string_view digits);  // 8080 eighty eighty, 443 four four three, 3000 three thousand
std::string ip_part(std::string_view digits);     // 192 one ninety-two, 0 zero, 100 one hundred
std::string hex_words(std::string_view s);        // ef254ca -> EF two five four CA
std::string num_words(std::string_view digits);   // a plain number in full; digit by digit with a leading zero
std::string name_num(std::string_view digits);    // the same, but 1900-2099 as a year (a name, tag or file name)

// Host parsing: labels separated by '.', [A-Za-z0-9-]. Returns the end of the longest prefix whose last label
// is a TLD (running text) or any alphabetic label of 2+ letters (strict = false, after a scheme), 0 if none.
struct Host {
    uint32_t end = 0;
    int labels = 0;
    bool ip = false;
};
Host parse_host(std::string_view text, uint32_t b, uint32_t limit, bool strict);
// Spoken host: labels with "dot", "www" as WWW, an IPv4 address part by part.
std::string host_words(std::string_view host);
// An IPv4 address at b (four parts of up to three digits, each up to 255); returns the end or 0.
uint32_t parse_ipv4(std::string_view text, uint32_t b, uint32_t limit);

void append_word(std::string& out, std::string_view w);  // adds w with a space before unless out is empty

// Readings at a colon (token k), called from one scan in web.cpp: IPv6 addresses kept as written (last: the end
// of the last one seen), scope operators (std::vector).
void ipv6_at(const Doc& doc, Out& out, size_t k, uint32_t& last);
void scope_operator_at(const Doc& doc, Out& out, size_t k);

// Proposes a reading for [b, e) that keeps its lower case at the start of a sentence (identifiers, addresses,
// file names, flags). The assembly capitalizes a replacement there, so the span then takes in the whitespace
// (and an opening backtick) before it and the spoken text starts with that whitespace. At the very start of the
// text this is not possible.
void add_lc(const Doc& doc, Out& out, uint32_t b, uint32_t e, std::string spoken, int score);

}  // namespace rv::el
