// Text -> chunks the way Tontaube splits text (app/text.py: sanitize_text_for_tts' whitespace rules, split_long_text
// with the sentence-first cuts and the short-final-sentence rule, then merge_short_pieces): chunks of at most 350
// characters, pieces under 50 joined to a neighbour (up to 400). Text as code points; character classes cover Latin,
// Greek and Cyrillic (Python's Unicode rules elsewhere differ).
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace chunker {

typedef std::u32string S;

inline bool is_space(char32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x1F) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool is_upper(char32_t c) {
    if (c < 128) return c >= 'A' && c <= 'Z';
    if (c >= 0xC0 && c <= 0xDE) return c != 0xD7;
    if (c >= 0x100 && c <= 0x137) return c % 2 == 0;   // Latin Extended-A: pairs, with a few shifts
    if (c >= 0x139 && c <= 0x148) return c % 2 == 1;
    if (c >= 0x14A && c <= 0x177) return c % 2 == 0;
    if (c == 0x178) return true;
    if (c >= 0x179 && c <= 0x17E) return c % 2 == 1;
    if (c == 0x386 || (c >= 0x388 && c <= 0x38F)) return c != 0x38B && c != 0x38D;
    if (c >= 0x391 && c <= 0x3AB) return c != 0x3A2;
    if (c >= 0x400 && c <= 0x42F) return true;
    return false;
}
inline bool is_lower(char32_t c) {
    if (c < 128) return c >= 'a' && c <= 'z';
    if (c >= 0xDF && c <= 0xFF) return c != 0xF7;
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c >= 0x100 && c <= 0x17F) return !is_upper(c);
    if (c >= 0x3AC && c <= 0x3CE) return true;
    if (c >= 0x430 && c <= 0x45F) return true;
    return false;
}
inline bool is_alpha(char32_t c) {
    if (c < 128) return (c | 32) >= 'a' && (c | 32) <= 'z';
    if (c == 0xAA || c == 0xB5 || c == 0xBA) return true;
    if (c >= 0xC0 && c <= 0x24F) return c != 0xD7 && c != 0xF7;
    if (c >= 0x250 && c <= 0x2AF) return true;
    if (c >= 0x386 && c <= 0x3FF) return c != 0x387 && c != 0x3F6;
    if (c >= 0x400 && c <= 0x481) return true;
    if (c >= 0x48A && c <= 0x52F) return true;
    return false;
}
inline bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }
inline bool is_alnum(char32_t c) { return is_alpha(c) || is_digit(c); }
inline char32_t lower(char32_t c) {
    if (c < 128) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 32;
    if (c == 0x178) return 0xFF;
    if (c >= 0x100 && c <= 0x17F && is_upper(c)) return c + 1;
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 32;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    return c;
}
inline bool in(char32_t c, const char32_t *set) { for (; *set; set++) if (*set == c) return true; return false; }

static const char32_t CLOSERS[] = U"\"'”’»)]";
static const char32_t OPENERS[] = U"\"'“‘«([";
static const char32_t TRAILING[] = U".,;:!?\"'”’»)]";
static const char32_t DIALOGUE[] = U"\"'“”„‟«»‘’‚‛‹›「『—―";
static const char *ABBREV[] = {
    "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "sra", "srta", "jr", "st", "ste", "mt", "rev", "hon", "gen", "col",
    "capt", "lt", "sgt", "gov", "sen", "rep", "pres", "vs", "approx", "dept", "est", "fig", "figs", "vol", "vols",
    "cf", "al", "ca", "fr", "jan", "feb", "aug", "sep", "sept", "oct", "nov", "dec",
    "nr", "bzw", "vgl", "ggf", "evtl", "inkl", "zzgl", "sog", "hr", "frl", "str",
    "mme", "mlle", "mm", "av", "bd", "dra", "ud", "uds", "sig", "dott", "ing", "avv", "dhr", "mevr", "bijv", "blz", nullptr};
static const char *LEFT_EXCLUDE[] = {"inc", "ltd", "llc", "corp", "etc", "bros", "ave", "blvd", "univ", "assn", "intl", "natl",
                                     "govt", "sec", "abs", "art", "ibid", "viz", "pts", "nos", nullptr};
static const char *RIGHT_EXCLUDE[] = {
    "com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "io", "ai", "app", "dev", "co", "uk", "de", "fr",
    "es", "it", "nl", "eu", "us", "ch", "at", "be", "tv", "me", "ly", "js", "ts", "py", "txt", "pdf", "html", "htm",
    "php", "exe", "jpg", "png", "gif", "csv", "json", "xml", "doc", "docx", "md", nullptr};

inline S lower_s(const S &s) { S o = s; for (auto &c : o) c = lower(c); return o; }
inline bool in_list(const S &w, const char **list) {
    for (; *list; list++) {
        const char *p = *list;
        size_t i = 0;
        for (; p[i] && i < w.size(); i++) if ((char32_t)(unsigned char)p[i] != w[i]) break;
        if (!p[i] && i == w.size()) return true;
    }
    return false;
}
inline S lstrip_set(const S &s, const char32_t *set) { size_t i = 0; while (i < s.size() && in(s[i], set)) i++; return s.substr(i); }
inline S rstrip_set(const S &s, const char32_t *set) { size_t e = s.size(); while (e > 0 && in(s[e - 1], set)) e--; return s.substr(0, e); }
inline S strip(const S &s) {
    size_t a = 0, e = s.size();
    while (a < e && is_space(s[a])) a++;
    while (e > a && is_space(s[e - 1])) e--;
    return s.substr(a, e - a);
}
inline S lstrip(const S &s) { size_t a = 0; while (a < s.size() && is_space(s[a])) a++; return s.substr(a); }
inline bool all_alpha(const S &s) { if (s.empty()) return false; for (auto c : s) if (!is_alpha(c)) return false; return true; }
inline bool all_digit(const S &s) { if (s.empty()) return false; for (auto c : s) if (!is_digit(c)) return false; return true; }
inline bool str_islower(const S &s) {   // Python str.islower: a cased character, none upper
    bool cased = false;
    for (auto c : s) { if (is_upper(c)) return false; if (is_lower(c)) cased = true; }
    return cased;
}
inline bool any_alnum(const S &s) { for (auto c : s) if (is_alnum(c)) return true; return false; }
// last index of space / tab / newline in [0, stop), -1 if none
inline int64_t rfind_ws(const S &t, int64_t stop) {
    for (int64_t i = std::min<int64_t>(stop, t.size()) - 1; i >= 0; i--) if (t[i] == ' ' || t[i] == '\t' || t[i] == '\n') return i;
    return -1;
}
inline int64_t rfind_any(const S &t, const char32_t *set, int64_t a, int64_t b) {   // last index in [a, b) of a char in set
    for (int64_t i = std::min<int64_t>(b, t.size()) - 1; i >= a; i--) if (in(t[i], set)) return i;
    return -1;
}

inline int word_count(const S &t) {
    int n = 0;
    size_t i = 0;
    while (i < t.size()) {
        while (i < t.size() && is_space(t[i])) i++;
        bool al = false;
        size_t j = i;
        while (j < t.size() && !is_space(t[j])) { al |= is_alnum(t[j]); j++; }
        if (j > i && al) n++;
        i = j;
    }
    return n;
}

inline bool dotted_abbrev(const S &w) {   // (?:[^\W\d_]{1,3}\.)+[^\W\d_]{1,3}
    int parts = 0;
    size_t i = 0;
    while (true) {
        size_t j = i;
        while (j < w.size() && is_alpha(w[j])) j++;
        const size_t len = j - i;
        if (len < 1 || len > 3) return false;
        parts++;
        if (j == w.size()) return parts >= 2;
        if (w[j] != '.') return false;
        i = j + 1;
    }
}

inline bool abbreviation_word(const S &word, const S &after) {
    const S lw = lower_s(word);
    return in_list(lw, ABBREV) || (word.size() == 1 && is_alpha(word[0])) || dotted_abbrev(word) ||
           (all_digit(word) && word.size() <= 2) ||
           ((lw == U"no" || lw == U"nr") && !after.empty() && is_digit(after[0]));
}

inline bool no_space_sentence_end(const S &t, int64_t stop) {
    const int64_t start = rfind_ws(t, stop) + 1;
    int64_t end = (int64_t)t.size();
    for (int64_t i = stop; i < (int64_t)t.size(); i++) if (t[i] == ' ' || t[i] == '\t' || t[i] == '\n') { end = i; break; }
    const S left = lstrip_set(t.substr(start, stop - start), OPENERS);
    const S right = rstrip_set(t.substr(stop + 1, end - stop - 1), TRAILING);
    S letters;
    for (auto c : left) if (c != '\'' && c != 0x2019) letters.push_back(c);
    if (!all_alpha(letters) || letters.size() < 3 || !is_lower(left.back())) return false;
    if (!(letters.size() >= 4 || !is_upper(left[0]))) return false;
    const S ll = lower_s(left);
    if (in_list(ll, ABBREV) || in_list(ll, LEFT_EXCLUDE)) return false;
    return right.size() >= 2 && all_alpha(right) && is_upper(right[0]) && str_islower(right.substr(1)) && !in_list(lower_s(right), RIGHT_EXCLUDE);
}

// (index of the terminator, cut offset after it) for each sentence end
inline std::vector<std::pair<int64_t, int64_t>> sentence_cuts(const S &t, bool no_space = true) {
    std::vector<std::pair<int64_t, int64_t>> cuts;
    const int64_t n = t.size();
    int64_t i = 0;
    while (i < n) {
        if (!in(t[i], U".!?")) { i++; continue; }
        int64_t j = i;
        while (j < n && in(t[j], U".!?")) j++;
        const int64_t run_end = j;
        while (j < n && in(t[j], CLOSERS)) j++;
        if (!(j == n || is_space(t[j]))) { i++; continue; }
        const int64_t ms = i, me = j;
        i = j;
        const S run = t.substr(ms, run_end - ms);
        const S after = lstrip(t.substr(me, 32));
        const bool lower_next = !after.empty() && is_lower(after[0]);
        if (lower_next && (run_end != me || run.back() == '!' || run.back() == '?')) continue;
        if (run == U".") {
            const S word = lstrip_set(t.substr(rfind_ws(t, ms) + 1, ms - (rfind_ws(t, ms) + 1)), OPENERS);
            if (abbreviation_word(word, after) || (lower_next && (word.size() <= 4 || (!word.empty() && is_upper(word[0]))))) continue;
        } else {
            bool dots = true;
            for (auto c : run) dots &= c == '.';
            if (dots) continue;
        }
        cuts.push_back({ms, me});
    }
    if (!no_space) return cuts;
    bool added = false;
    for (int64_t k = 1; k + 1 < n; k++)
        if (t[k] == '.' && is_alpha(t[k - 1]) && is_alpha(t[k + 1]) && no_space_sentence_end(t, k)) { cuts.push_back({k, k + 1}); added = true; }
    if (added) std::sort(cuts.begin(), cuts.end());
    return cuts;
}

inline bool ends_on_short_sentence(const S &chunk0) {
    const S chunk = strip(chunk0);
    int64_t last = -1;
    for (auto &c : sentence_cuts(chunk)) if (c.second < (int64_t)chunk.size()) last = c.second;
    const S fin = last >= 0 ? strip(chunk.substr(last)) : chunk;
    const int w = word_count(fin);
    return w > 0 && w <= 2 && fin.size() <= 40;
}

inline bool one_word_before_dialogue(const S &moved, const S &following, bool spaced = true) {
    const S head = lstrip(following);
    if (head.empty()) return false;
    return word_count(moved) == 1 && in(head[0], DIALOGUE) && head.size() >= 2 && !is_space(head[1]) &&
           ((!following.empty() && is_space(following[0])) || !spaced);
}

inline int64_t cut_before_short_final_sentence(const S &t, int64_t split_at, int64_t window_start) {
    const S head = t.substr(0, split_at);
    if (!ends_on_short_sentence(head)) return split_at;
    int64_t end = head.size();
    while (end > 0 && is_space(head[end - 1])) end--;
    auto cuts = sentence_cuts(head, false);
    for (auto it = cuts.rbegin(); it != cuts.rend(); ++it) {
        if (it->first < window_start) break;
        if (it->second < end && !ends_on_short_sentence(head.substr(0, it->second))) {
            if (one_word_before_dialogue(head.substr(it->second), t.substr(split_at))) return split_at;
            return it->second;
        }
    }
    return split_at;
}

inline bool known_non_end(const S &t, int64_t stop) {
    if (t[stop] != '.') return false;
    const int64_t n = t.size();
    int64_t end = stop + 1;
    while (end < n && in(t[end], CLOSERS)) end++;
    if ((stop > 0 && t[stop - 1] == '.') || (end < n && t[end] == '.')) return false;
    S word = lstrip_set(t.substr(rfind_ws(t, stop) + 1, stop - (rfind_ws(t, stop) + 1)), OPENERS);
    if (end == stop + 1 && end < n && !is_space(t[end])) {
        if (!dotted_abbrev(word)) { const size_t d = word.rfind(U'.'); if (d != S::npos) word = word.substr(d + 1); }
        return !is_upper(t[end]) || abbreviation_word(word, t.substr(end));
    }
    return abbreviation_word(word, lstrip(t.substr(end, 32)));
}

inline bool inside_number(const S &t, int64_t i) { return i > 0 && i < (int64_t)t.size() - 1 && is_digit(t[i - 1]) && is_digit(t[i + 1]); }

inline int64_t last_sentence_cut(const S &t, int64_t start, int64_t limit) {
    int64_t best = -1;
    for (auto &c : sentence_cuts(t, false)) {
        if (c.first >= limit) break;
        if (c.first >= start && c.second <= limit) best = c.second;
    }
    if (best >= 0) return best;
    for (int64_t stop = limit - 1; stop >= start; stop--)
        if (in(t[stop], U".!?") && !known_non_end(t, stop)) {
            int64_t cut = stop + 1;
            while (cut < limit && in(t[cut], CLOSERS)) cut++;
            return cut;
        }
    return -1;
}

inline std::vector<S> split_segment(const S &segment, int64_t max_chars, bool avoid) {
    S rem = strip(segment);
    std::vector<S> pieces;
    while ((int64_t)rem.size() > max_chars) {
        const int64_t limit = max_chars;
        int64_t split_at = last_sentence_cut(rem, (int64_t)(limit * 0.3), limit);
        if (split_at < 0) {
            int64_t boundary = -1;
            for (const char32_t *chars : {U";:", U","}) {
                boundary = rfind_any(rem, chars, 0, limit);
                while (boundary > 0 && inside_number(rem, boundary)) boundary = rfind_any(rem, chars, 0, boundary);
                if (boundary >= 0) break;
            }
            if (boundary >= 0) split_at = boundary + 1;
            else {
                const int64_t ws = rfind_any(rem, U" \t\n", 0, limit + 1);
                split_at = ws > 0 ? ws : limit;
            }
        }
        if (avoid) split_at = cut_before_short_final_sentence(rem, split_at, (int64_t)(limit * 0.3));
        pieces.push_back(strip(rem.substr(0, split_at)));
        rem = strip(rem.substr(split_at));
    }
    if (!rem.empty()) pieces.push_back(rem);
    std::vector<S> merged;
    for (auto &p : pieces) {
        if (!merged.empty() && (!any_alnum(p) || !any_alnum(merged.back()))) merged.back() += p;
        else merged.push_back(p);
    }
    if (avoid) {
        bool over = false;
        for (auto &p : merged) over |= (int64_t)p.size() > max_chars;
        if (over) {
            auto plain = split_segment(segment, max_chars, false);
            bool ok = true;
            for (auto &p : plain) ok &= (int64_t)p.size() <= max_chars;
            if (ok) return plain;
        }
    }
    return merged;
}

inline void merge_short_pieces(std::vector<S> &pieces, int64_t min_chars, int64_t max_chars, bool avoid) {
    bool changed = true;
    while (changed && pieces.size() > 1) {
        changed = false;
        for (size_t index = 0; index < pieces.size(); index++) {
            if ((int64_t)pieces[index].size() >= min_chars) continue;
            int64_t order[2] = {(int64_t)index - 1, (int64_t)index};
            if (avoid && ends_on_short_sentence(pieces[index]) &&
                !(index + 1 < pieces.size() && one_word_before_dialogue(pieces[index], pieces[index + 1], false))) {
                order[0] = index; order[1] = (int64_t)index - 1;
            }
            for (int64_t left : order) {
                if (left < 0 || left + 1 >= (int64_t)pieces.size()) continue;
                const S joined = pieces[left] + U" " + pieces[left + 1];
                if ((int64_t)joined.size() <= max_chars) {
                    pieces[left] = joined;
                    pieces.erase(pieces.begin() + left + 1);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    }
}

// whitespace as sanitize_text_for_tts: runs with no newline -> " ", one newline -> "\n", more -> "\n\n"; strip; a
// final ". ! ? ; : ," added if missing
inline S sanitize(const S &t) {
    S o;
    size_t i = 0;
    while (i < t.size()) {
        const char32_t c = t[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
            int nl = 0;
            while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r' || t[i] == '\v' || t[i] == '\f')) {
                nl += t[i] == '\n' || t[i] == '\v' || t[i] == '\f';
                if (t[i] == '\r' && !(i + 1 < t.size() && t[i + 1] == '\n')) nl++;
                i++;
            }
            o += nl == 0 ? U" " : nl == 1 ? U"\n" : U"\n\n";
        } else {
            o.push_back(c);
            i++;
        }
    }
    o = strip(o);
    if (!o.empty() && !in(o.back(), U".!?;:,")) o.push_back('.');
    return o;
}

inline std::vector<S> chunks(const S &text, int64_t max_chars = 350, int64_t short_chars = 50) {
    const S t = sanitize(text);
    if (t.empty()) return {};
    std::vector<S> p = split_segment(t, max_chars, true);
    if (p.size() > 1) merge_short_pieces(p, short_chars, max_chars + short_chars, true);
    return p;
}

inline S from_utf8(const std::string &s) {
    S o;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        char32_t cp = c;
        int more = 0;
        if (c >= 0xF0) { cp = c & 7; more = 3; } else if (c >= 0xE0) { cp = c & 15; more = 2; } else if (c >= 0xC0) { cp = c & 31; more = 1; }
        i++;
        for (; more > 0 && i < s.size(); more--, i++) cp = (cp << 6) | (s[i] & 63);
        o.push_back(cp);
    }
    return o;
}
inline std::string to_utf8(const S &s) {
    std::string o;
    for (char32_t c : s) {
        if (c < 0x80) o.push_back((char)c);
        else if (c < 0x800) { o.push_back((char)(0xC0 | (c >> 6))); o.push_back((char)(0x80 | (c & 63))); }
        else if (c < 0x10000) { o.push_back((char)(0xE0 | (c >> 12))); o.push_back((char)(0x80 | ((c >> 6) & 63))); o.push_back((char)(0x80 | (c & 63))); }
        else { o.push_back((char)(0xF0 | (c >> 18))); o.push_back((char)(0x80 | ((c >> 12) & 63))); o.push_back((char)(0x80 | ((c >> 6) & 63))); o.push_back((char)(0x80 | (c & 63))); }
    }
    return o;
}

}  // namespace chunker
